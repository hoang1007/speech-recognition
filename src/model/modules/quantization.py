from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import einops


class QuantizationModule(nn.Module):
    def __init__(
        self, config
    ):
        """
        Args:
            x (Tensor): The extracted features from waveforms. Shape: (batch, num_frames, in_features)
            mask (BoolTensor): The mask for the valid frames. `True` is invalid. Shape: (batch, num_frames)

        Returns:
            out (Tensor): The quantized features. Shape: (batch, num_frames, d_model)
            perplexity (Tensor): The perplexity of the quantized features. Shape: (1)
        """
        super().__init__()

        assert (
            config.d_model % config.num_codebooks == 0
        ), "d_model must be divisible by num_codebooks"

        self.num_codebooks = config.num_codebooks
        self.num_codewords = config.num_codewords
        self.d_model = config.d_model
        self.codeword_dim = config.d_model // config.num_codebooks

        self.codebooks = self._init_codebooks()

        self.projection = nn.Linear(
            config.in_features, self.num_codebooks * self.num_codewords
        )

        self.tau = 1  # temperature factor

    def _init_codebooks(self):
        codebooks = torch.randn(
            1, 1, self.num_codebooks, self.num_codewords, self.codeword_dim
        )
        nn.init.xavier_uniform_(codebooks)

        return nn.Parameter(codebooks)

    @property
    def total_codewords(self):
        return self.num_codebooks * self.num_codewords

    @staticmethod
    def _compute_perplexity(probs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Computes the perplexity of the quantized features. (Diversity loss)

        Args:
            probs (Tensor): The probability distribution of words in codebooks. Shape: (batch, num_frames, num_codebooks, num_codewords)
            mask (BoolTensor): The mask for the valid frames. `True` is invalid. Shape: (batch, num_frames)
        """
        if mask is not None:
            probs = (
                probs * ~mask[..., None, None]
            )  # Turn invalid frames' probability to 0
            marginal_probs = (
                einops.reduce(probs, "b nf nb nw -> nb nw", "sum") / mask.sum()
            )
        else:
            marginal_probs = einops.reduce(probs, "b nf nb nw -> nb nw", "mean")

        perplexity = torch.exp(
            -torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)
        ).sum()
        return perplexity

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, num_frames, _ = x.shape

        logits = self.projection(x)
        logits = logits.view(
            batch_size, num_frames, self.num_codebooks, self.num_codewords
        )

        if self.training:
            word_probs = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            word_soft_probs = F.softmax(logits, dim=-1)

            perplexity = self._compute_perplexity(word_soft_probs, mask=mask)
        else:
            word_ids = torch.argmax(logits, dim=-1, keepdim=True)
            word_probs = torch.zeros_like(logits).scatter_(-1, word_ids, 1.0)  # One-hot

            perplexity = self._compute_perplexity(word_probs, mask=mask)

        # (batch, num_frames, num_codebooks, num_codewords, 1) x (1, 1, num_codebooks, num_codewords, codeword_dim)
        # -> (batch, num_frames, num_codebooks x codeword_dim)
        quantized = einops.reduce(
            word_probs.unsqueeze_(-1) * self.codebooks,
            "b nf nb nw d -> b nf (nb d)",
            reduction="sum",
        )

        return quantized, perplexity
