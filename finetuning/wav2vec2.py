from typing import Tuple
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import (
    Wav2Vec2ForPreTraining,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)

from src.utils.metrics import character_error_rate, word_error_rate
from src.utils.scheduler import TriStateScheduler


class SpeechRecognizer(LightningModule):
    def __init__(
        self,
        wav2vec2: Wav2Vec2ForPreTraining,
        tokenizer: Wav2Vec2CTCTokenizer,
        feature_extractor: Wav2Vec2FeatureExtractor,
        adam_config: dict,
        tristate_scheduler_config: dict,
    ):
        super().__init__()

        self.hidden_size = wav2vec2.config.proj_codevector_dim
        self.vocab_size = tokenizer.vocab_size

        self.wav2vec2 = wav2vec2
        self.wav2vec2.freeze_feature_encoder()
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.adam_config = adam_config
        self.tristate_scheduler_config = tristate_scheduler_config

        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size // 2, self.vocab_size),
        )

        self.criterion = torch.nn.CTCLoss(blank=tokenizer.pad_token_id, zero_infinity=True)

        self.train_loss = MeanMetric()

        self.save_hyperparameters(ignore=["wav2vec2", "tokenizer", "feature_extractor"])

    def forward(self, waveforms: Tuple[torch.Tensor], transcripts: Tuple[str] = None):
        # convert torch.Tensor to numpy.ndarray
        waveforms = tuple(waveform.cpu().numpy() for waveform in waveforms)

        input_values, attention_mask = self.feature_extractor(
            waveforms,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        ).values()

        input_values = input_values.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # hidden_states.shape == (batch_size, sequence_length, hidden_size)
        hidden_states = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
        )[0]

        hidden_states = self.dropout(hidden_states)

        # logits.shape == (batch_size, sequence_length, vocab_size)
        logits = self.fc(hidden_states)

        # get the length of valids sequence
        input_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).long()

        if transcripts is not None:
            # tokenize transcripts
            target_ids, target_lengths = self.tokenizer(
                transcripts,
                padding=True,
                return_length=True,
                return_attention_mask=False,
                return_tensors="pt",
            ).values()

            target_ids = target_ids.to(self.device)
            assert (
                target_ids < self.tokenizer.vocab_size
            ).all(), "target_ids is out of range"

            target_lengths = target_lengths.to(self.device)
            assert (
                target_lengths <= logits.size(1)
            ).all(), "target_lengths is out of range"

            # (batch_size, sequence_length, vocab_size) -> (sequence_length, batch_size, vocab_size)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)

            # compute loss
            loss = self.criterion(log_probs, target_ids, input_lengths, target_lengths)

            return loss, logits, input_lengths
        else:
            return logits, input_lengths

    @staticmethod
    def _get_predicted_ids(logits: torch.Tensor, lengths: torch.Tensor):
        # logits.shape == (batch_size, sequence_length, vocab_size)
        # lengths.shape == (batch_size, )

        # get the max value of logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # remove the padding
        predicted_ids = [
            predicted_id[:length]
            for predicted_id, length in zip(predicted_ids, lengths)
        ]

        return predicted_ids

    def training_step(self, batch, batch_idx):
        transcripts, waveforms = batch

        loss = self(waveforms, transcripts)[0]

        self.train_loss(loss)

        if self.global_step % 500 == 0:
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        transcripts, waveforms = batch

        logits, seq_lengths = self(waveforms)

        predicted_ids = self._get_predicted_ids(logits, seq_lengths)
        predicted_texts = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        wer = word_error_rate(predicted_texts, transcripts)
        cer = character_error_rate(predicted_texts, transcripts)

        return wer, cer

    def validation_epoch_end(self, outputs):
        wer, cer = zip(*outputs)

        wer = sum(wer) / len(wer)
        cer = sum(cer) / len(cer)

        self.log("val/wer", wer, on_epoch=True)
        self.log("val/cer", cer, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[
                {
                    "params": self.wav2vec2.parameters(),
                    "lr": self.adam_config["wav2vec2_lr"],
                },
                {
                    "params": self.fc.parameters(),
                    "lr": self.adam_config["classifier_lr"],
                },
            ],
            weight_decay=self.adam_config["weight_decay"],
        )

        scheduler = TriStateScheduler(optimizer, **self.tristate_scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
