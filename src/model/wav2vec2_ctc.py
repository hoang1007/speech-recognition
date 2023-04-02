from typing import List, Optional
from warnings import warn
from omegaconf import DictConfig

import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric

try:
    from transformers import AutoTokenizer
except ImportError as e:
    warn("Please install transformers to use Wav2Vec2ForCTC: `pip install transformers`")

from .wav2vec2_pretraining import Wav2Vec2
from src.utils import instantiate, load_pretrained_model
from src.utils.metrics import character_error_rate, word_error_rate


class Wav2Vec2ForCTC(LightningModule):
    def __init__(
            self, 
            wav2vec: DictConfig,
            transcript_tokenizer: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: Optional[DictConfig] = None,
            pretrain: bool = False):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.wav2vec = Wav2Vec2(**wav2vec)
        self.transcript_tokenizer = AutoTokenizer.from_pretrained(**transcript_tokenizer)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if pretrain:
            self.wav2vec = load_pretrained_model(self.wav2vec) # type: ignore
        
        hidden_size = self.get_embedding_dim()
        vocab_size = self.transcript_tokenizer.vocab_size # type: ignore
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, vocab_size),
        )

        assert self.transcript_tokenizer.pad_token_id is not None
        self.criterion = nn.CTCLoss(
            blank=self.transcript_tokenizer.pad_token_id,
            zero_infinity=True)
        
        self.train_loss = MeanMetric()

    def get_embedding_dim(self):
        dummy_input = torch.randn(16000)
        embedding_dim = int(self.wav2vec(dummy_input).size(-1))
        return embedding_dim
    
    def forward(self, waveforms: List[Tensor], transcripts: Optional[List[str]] = None):
        features, seq_out_lengths = self.wav2vec(waveforms)
        features = self.dropout(features)

        logits = self.fc(features)

        if self.training:
            assert transcripts is not None, "transcripts must be provided in training state"
            # tokenize transcripts
            target_ids, target_lengths = self.transcript_tokenizer(
                transcripts,
                padding=True,
                return_length=True,
                return_attention_mask=False,
                return_tensors="pt",
            ).values()

            target_ids = target_ids.to(logits.device)
            assert (
                target_ids < self.transcript_tokenizer.vocab_size
            ).all(), "target_ids is out of range"

            target_lengths = target_lengths.to(logits.device)
            assert (
                target_lengths <= logits.size(1)
            ).all(), "target_lengths is out of range"

            # (batch_size, seq_len, vocab_size) -> (seq_len, batch_size, vocab_size)
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = self.criterion(log_probs, target_ids, seq_out_lengths, target_lengths)

            return loss, logits, seq_out_lengths
        else:
            return logits, seq_out_lengths
    
    def _get_predicted_token_ids(self, logits: Tensor, lengths: Tensor) -> List[Tensor]:
        predicted_token_ids = logits.argmax(dim=-1)

        # remove padding
        predicted_token_ids = [
            token_ids[:length]
            for token_ids, length in zip(predicted_token_ids, lengths)
        ]
        return predicted_token_ids
    
    def training_step(self, batch):
        waveforms, transcripts = batch
        loss, _, _ = self(waveforms, transcripts)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.train_loss.reset()

    def validation_step(self, batch):
        waveforms, transcripts = batch

        logits, seq_out_lengths = self(waveforms)
        predicted_token_ids = self._get_predicted_token_ids(logits, seq_out_lengths)
        pred_transcripts = self.transcript_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

        cer = character_error_rate(pred_transcripts, transcripts)
        wer = word_error_rate(pred_transcripts, transcripts)

        self.log("val/cer", cer, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, waveforms: List[Tensor]):
        logits, seq_out_lengths = self(waveforms)
        predicted_token_ids = self._get_predicted_token_ids(logits, seq_out_lengths)
        pred_transcripts = self.transcript_tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

        return pred_transcripts

    def configure_optimizers(self):
        params = [
            {"params": self.wav2vec.parameters(), "lr": self.optimizer.lr * 0.1},
            {"params": self.fc.parameters(), "lr": self.optimizer.lr},
        ]
        optimizer = instantiate(self.optimizer, partial=True)(params)

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