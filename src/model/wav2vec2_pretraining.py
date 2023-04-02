"""
A wrapper of Wav2Vec2 for training phase.
"""
from typing import Tuple, List, Optional
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningModule
import einops
from torchmetrics import MeanMetric

from .modules import (
    ContextEncoder,
    FeatureExtractor,
    QuantizationModule,
    Wav2Vec2Processor,
)
from src.utils import init_module_weights, instantiate


class Wav2Vec2(LightningModule):
    def __init__(
        self,
        feature_extractor: DictConfig,
        context_encoder: DictConfig,
        quantizer: Optional[DictConfig] = None,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        train_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.processor = Wav2Vec2Processor()
        self.context_encoder = ContextEncoder(**context_encoder)
        self.feature_extractor = FeatureExtractor(**feature_extractor)
        if quantizer is not None:
            self.quantizer = QuantizationModule(**quantizer)
        else:
            self.quantizer = None
        self.train_cfg = train_cfg

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = MeanMetric()

    def init_weights(self):
        init_module_weights(self.context_encoder)
        init_module_weights(self.feature_extractor)

    def forward(
        self,
        waveforms: List[torch.Tensor],
        mask_time_indices: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            waveforms (List[torch.Tensor]): List containing waveforms. Each waveform has shape: (wave_length,).

        Returns:
            torch.Tensor: The context features of waveforms. Shape: (batch_size, num_frames, hidden_size).
            wave_lengths (torch.Tensor): The lengths of waveforms. Shape: (batch_size,).
        """
        waveforms, wave_lengths = self.processor(waveforms)

        # features.shape == (batch_size, num_frames, hidden_size)
        features, seq_lengths = self.feature_extractor(waveforms, wave_lengths)
        attention_mask = self._compute_attention_mask(seq_lengths)

        context_features = self.context_encoder(
            features, attention_mask=attention_mask, mask_time_indices=mask_time_indices
        )

        return context_features, seq_lengths

    def loss(self, waveforms: List[torch.Tensor]):
        """
        Args:
            waveforms (Tuple[torch.Tensor]): The waveforms. Shape: (batch_size, wave_length).

        Returns:
            loss: The loss of the model. Contrastive loss + Diversity loss.
        """
        assert self.train_cfg is not None, "`train_cfg` is required for training phase."
        assert self.quantizer is not None, "`quantizer` is required for training phase."

        waveforms, wave_lengths = self.processor(waveforms)

        # features.shape == (batch_size, num_frames, hidden_size)
        features, num_frames = self.feature_extractor(waveforms, wave_lengths)

        attention_mask = self._compute_attention_mask(num_frames)
        mask_time_indices = self._compute_mask_span(
            shape=features.shape[:-1],
            mask_prob=self.train_cfg.mask_prob,
            mask_length=self.train_cfg.mask_length,
            attention_mask=attention_mask,
            device=features.device,
            min_masks=self.train_cfg.min_masks,
        )

        context_features = self.context_encoder(
            features, attention_mask=attention_mask, mask_time_indices=mask_time_indices
        )

        quantized_features, perplexity = self.quantizer(features, attention_mask)

        negative_quantized_features = self._sample_negatives(
            quantized_features,
            num_negatives=self.train_cfg.num_negatives,
            attention_mask=attention_mask,
        )

        # (batch_size, num_frames, num_negatives + 1)
        contrastive_logits = self._compute_contrastive_logits(
            context_features,
            quantized_features,
            negative_quantized_features,
            self.train_cfg.contrastive_logits_temperature,
        ).flatten(0, -2)

        # compute contrastive loss
        # positive indices are always the first one
        targets = (1 - mask_time_indices.long().flatten()) * -100

        contrastive_loss = torch.nn.functional.cross_entropy(
            contrastive_logits, targets, reduction="sum"
        )

        # compute diversity loss
        diversity_loss = 1 - perplexity / self.quantizer.total_codewords

        loss = contrastive_loss + diversity_loss * self.train_cfg.diversity_loss_weight

        return loss

    @staticmethod
    def _sample_negatives(
        features: torch.Tensor,
        num_negatives: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Sampling negative features from quantized features to compute the contrastive loss.

        Args:
            features (torch.Tensor): The quantized features. Shape: (batch_size, num_frames, d_model).
            num_negatives (int): The number of negative samples.
            attention_mask (Optional[torch.Tensor]): The mask for valid frames. `True` is invalid. Shape: (batch_size, num_frames).

        Returns:
            sampled_negatives (torch.Tensor): The sampled negative features. Shape: (batch_size, num_frames, num_negatives, d_model).
        """

        batch_size, num_frames, d_model = features.shape

        features = features.view(-1, d_model)  # (batch_size * num_frames, d_model)

        with torch.no_grad():
            sampled_ids = []

            for batch_idx in range(batch_size):
                num_valid_frames = int(
                    features.size(1)
                    if attention_mask is None
                    else (~attention_mask[batch_idx]).sum().item()
                )

                sampled_ids.append(
                    torch.randint(
                        0,
                        num_valid_frames - 1,
                        (num_frames * num_negatives,),
                        device=features.device,
                    )
                )

            # (batch_size, num_frames * num_negatives)
            sampled_ids = torch.stack(sampled_ids, dim=0)

            feature_ids = einops.repeat(
                torch.arange(num_frames, device=features.device),
                "f -> (f n)",
                n=num_negatives,
            )

            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_ids[sampled_ids >= feature_ids] += 1

        # correct for batch size
        # E.g [[0, 1, 2], [0, 1, 2]] -> [0, 1, 2, 3, 4, 5]
        sampled_ids += torch.arange(
            0, batch_size * num_frames, num_frames, device=features.device
        ).unsqueeze_(-1)

        sampled_negatives = features[sampled_ids.view(-1)]
        sampled_negatives = einops.rearrange(
            sampled_negatives,
            "(b f n) d -> b f n d",
            b=batch_size,
            f=num_frames,
            n=num_negatives,
        )

        return sampled_negatives

    @staticmethod
    def _compute_contrastive_logits(
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        negative_features: torch.Tensor,
        temperature: int = 1,
    ):
        """
        Compute the logits for contrastive loss.

        Args:
            predicted_features (torch.Tensor): The predicted features. Shape: (batch_size, num_frames, d_model).
            target_features (torch.Tensor): The target features. Shape: (batch_size, num_frames, d_model).
            negative_features (torch.Tensor): The negative features. Shape: (batch_size, num_frames, num_negatives, d_model).
            temperature (int): The temperature for contrastive loss.

        Returns:
            logits (torch.Tensor): The logits for contrastive loss. Shape: (batch_size, num_frames, num_negatives + 1).
        """

        # (batch_size, num_frames, num_negatives + 1, d_model)
        target_features = torch.cat(
            (target_features.unsqueeze_(2), negative_features), dim=2
        )

        # (batch_size, num_frames, 1, d_model)
        predicted_features = predicted_features.unsqueeze_(2)

        # (batch_size, num_frames, num_negatives + 1)
        logits = torch.cosine_similarity(predicted_features, target_features, dim=-1)
        logits /= temperature

        return logits

    @staticmethod
    def _compute_mask_span(
        shape: Tuple[int, int],
        mask_prob: float = 0.065,
        mask_length: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
        min_masks: int = 0,
    ):
        """
        Compute the mask span for contrastive task.

        Args:
            shape (Tuple[int, int]): The shape of the mask span. Shape: (batch_size, num_frames).
            mask_prob (float): The probability of choosing a frame to be the start of masking position.
            mask_length (int): The length of the mask span.
            attention_mask (Optional[torch.Tensor]): The mask for valid frames. `True` is invalid. Shape: (batch_size, num_frames).
            device (torch.device): The device of the mask span.
            min_masks (int): The minimum number of masks.

        Returns:
            mask_span (torch.Tensor): The mask span. Shape: (batch_size, num_frames).
        """

        batch_size, num_frames = shape

        # NOTE: num_frames / mask_length: the number of spans in one waveform
        num_masked_spans = int(
            mask_prob * num_frames / mask_length + torch.rand(1).item()
        )
        num_masked_spans = max(num_masked_spans, min_masks)

        # make sure num masked indices <= num frames
        if num_masked_spans * mask_length > num_frames:
            num_masked_spans = num_frames // mask_length

        # uniform distribution to sample from
        # NOTE: num_frames - (mask_length - 1): the number of start positions of the span
        uniform_dist = torch.ones(
            (batch_size, num_frames - (mask_length - 1)), device=device
        )

        # (batch_size, num_masked_spans)
        mask_span_ids = torch.multinomial(uniform_dist, num_masked_spans)

        # (batch_size, num_masked_spans * mask_length)
        mask_span_ids = einops.repeat(mask_span_ids, "b n -> b (n l)", l=mask_length)

        offsets = einops.repeat(
            torch.arange(mask_length, device=device),
            "l -> b (n l)",
            b=batch_size,
            n=num_masked_spans,
        )

        mask_span_ids = mask_span_ids + offsets

        mask_span = torch.zeros(shape, device=device, dtype=torch.bool)
        mask_span = mask_span.scatter_(1, mask_span_ids, True)

        if attention_mask is not None:
            # Make sure the invalid frames are not masked
            mask_span = torch.where(attention_mask.bool(), mask_span, False)

        return mask_span

    @staticmethod
    def _compute_attention_mask(length: torch.Tensor):
        """
        Args:
            length (Tensor): The length of valid frames. Shape: (batch)
            max_length (int): The maximum length of the frames.

        Returns:
            attention_mask (BoolTensor): The mask for the valid frames. `True` is invalid. Shape: (batch, num_frames)
        """
        max_length = int(length.max().item())

        mask = (
            torch.arange(max_length, device=length.device).expand(
                length.size(0), max_length
            )
            >= length[:, None]
        )

        return mask

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)

        self.train_loss(loss)

        if batch_idx % 100 == 0:
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer is not None:
            optimizer = instantiate(self.optimizer, partial=True)(self.parameters())

            if self.lr_scheduler is not None:
                assert "scheduler" in self.lr_scheduler, "Please specify the scheduler."
                scheduler = instantiate(self.lr_scheduler.pop("scheduler"), partial=True)(
                    optimizer
                )
                return dict(
                    optimizer=optimizer,
                    lr_scheduler=dict(
                        scheduler=scheduler,
                        **self.lr_scheduler,
                    ),
                )
            else:
                return optimizer
