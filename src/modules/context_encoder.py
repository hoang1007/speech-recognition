from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from .transformers import EncoderLayer


class FeatureProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """
        Projects the extracted features to the encoder dimension.

        Args:
            x (Tensor): The input features. Shape: (batch, num_frames, in_features)

        Returns:
            hiddens (Tensor): The latent features. Shape: (batch, num_frames, out_features)
        """
        super().__init__()

        self.projection = nn.Linear(in_features, out_features)
        self.layernorm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        hiddens = self.layernorm(x)
        hiddens = self.projection(x)
        hiddens = self.dropout(hiddens)
        return hiddens


class RelativePositionalEmbedding(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int, groups: int, dropout: float = 0.1
    ):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, d_model)

        Returns:
            out (Tensor): The output which encoded the relative positional information. Shape: (batch, num_frames, d_model)
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor):
        # (batch, channels=d_model, num_frames)
        out = x.transpose(1, 2)

        out = self.conv(out)

        if self.num_remove > 0:
            out = out[..., : -self.num_remove]

        out = F.gelu(out)

        # (batch, num_frames, channels=d_model)
        out = out.transpose_(1, 2)
        out = out + x
        out = self.dropout(out)

        return out


class TranformerEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, d_model)
            mask (Tensor): The mask for the valid frames. Shape: (batch, num_frames)

        Returns:
            out (Tensor): The output of the transformer encoder. Shape: (batch, num_frames, d_model)
        """
        super().__init__()

        self.pos_embedding = RelativePositionalEmbedding(**config.pos_embedding)
        self.layernorm = nn.LayerNorm(config.d_model)
        self.layer_drop = config.layer_drop

        self.layers = nn.ModuleList(
            EncoderLayer(**config.layer) for _ in range(config.num_layers)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        out = self.pos_embedding(x)

        for layer in self.layers:
            skip_layer = self.training and torch.rand(1).item() < self.layer_drop

            if skip_layer:
                continue
            else:
                out, _ = layer(out, attention_mask=mask)

        out = self.layernorm(out)

        return out


class ContextEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, in_features)
            attention_mask (BoolTensor): The mask for the valid frames. `True` is invalid. Shape: (batch, num_frames)
        """
        super().__init__()

        self.feature_projection = FeatureProjection(**config.feature_projection)
        self.encoder = TranformerEncoder(config.encoder)
        self.masked_spec_embed = nn.Parameter(
            torch.FloatTensor(config.feature_projection.out_features).uniform_()
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        mask_time_indices: torch.Tensor = None,
    ):
        x = self.feature_projection(x)

        if mask_time_indices is not None:
            x[mask_time_indices] = self.masked_spec_embed.to(x.dtype)

        if attention_mask is not None:
            x[attention_mask] = 0.0  # turn invalid frames to zero

            attention_mask = attention_mask[:, None, None, :]
            # (batch, 1, num_frames, num_frames)
            # mask = mask[:, None, None, :].repeat(1, 1, mask.size(1), 1) # TODO: check this
            attention_mask = (
                torch.maximum(attention_mask, attention_mask.transpose(2, 3)) * 1e-6
            )

        x = self.encoder(x, mask=attention_mask)

        return x
