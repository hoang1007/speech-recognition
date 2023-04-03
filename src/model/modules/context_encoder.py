from typing import Optional
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F
from .transformers import EncoderLayer


class FeatureProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        """
        Projects the extracted features to the encoder dimension.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            dropout (float): The dropout probability.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): The input features. Shape: (batch, num_frames, in_features)

        Returns:
            hiddens (Tensor): The latent features. Shape: (batch, num_frames, out_features)
        """
        x = self.layer_norm(x)
        hiddens = self.projection(x)
        hiddens = self.dropout(hiddens)
        return hiddens


class RelativePositionalConvEmbedding(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, groups: int):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)  # type: ignore
        self.activation = nn.GELU()
        self.num_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, d_model)

        Returns:
            out (Tensor): The output which encoded the relative positional information. Shape: (batch, num_frames, d_model)
        """
        # (batch, channels=d_model, num_frames)
        out = x.transpose(1, 2)

        out = self.conv(out)
        if self.num_remove > 0:
            out = out[..., : -self.num_remove]

        out = self.activation(out)

        # (batch, num_frames, channels=d_model)
        out = out.transpose_(1, 2)

        return out


class TranformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_embedding: DictConfig,
        enc_layer: DictConfig,
        num_enc_layers: int,
        layer_drop_prob: float = 0.1,
        dropout: float = 0.1,
        stable_layer_norm: bool = False,
    ):
        """
        The transformer encoder.

        Args:
            d_model (int): The dimension of the encoder.
            pos_embedding (DictConfig): The config of the positional embedding.
            enc_layer (DictConfig): The config of the encoder layer.
            num_layers (int): The number of encoder layers.
            layer_drop_prob (float): The probability of dropping the encoder layer.
        """
        super().__init__()

        self.pos_conv_embed = RelativePositionalConvEmbedding(d_model, **pos_embedding)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_drop_prob = layer_drop_prob
        self.stable_layer_norm = stable_layer_norm

        self.layers = nn.ModuleList(
            EncoderLayer(d_model, **enc_layer) for _ in range(num_enc_layers)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, d_model)
            mask (Tensor): The mask for the valid frames. Shape: (batch, num_frames)

        Returns:
            out (Tensor): The output of the transformer encoder. Shape: (batch, num_frames, d_model)
        """
        pos_embedding = self.pos_conv_embed(x)
        out = x + pos_embedding
        if not self.stable_layer_norm:
            out = self.layer_norm(out)
        out = self.dropout(out)

        for layer in self.layers:
            # Random dropout probability from a uniform distribution
            skip_layer = self.training and torch.rand(1).item() < self.layer_drop_prob

            if skip_layer:
                continue
            else:
                out, _ = layer(out, attention_mask=mask)

        if self.stable_layer_norm:
            out = self.layer_norm(out)

        return out


class ContextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        feature_projection: DictConfig,
        transformer_encoder: DictConfig,
    ):
        super().__init__()

        self.feature_projection = FeatureProjection(
            out_features=d_model, **feature_projection
        )
        self.encoder = TranformerEncoder(d_model=d_model, **transformer_encoder)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(d_model).uniform_())

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): The extracted features. Shape: (batch, num_frames, in_features)
            attention_mask (Optional[BoolTensor]): The mask for the valid frames. `True` is invalid. Shape: (batch, num_frames)
            mask_time_indices (Optional[LongTensor]): The indices of the masked frames. Shape: (batch, num_masked_frames)
        """
        x = self.feature_projection(x)

        if mask_time_indices is not None:
            x[mask_time_indices] = self.masked_spec_embed.to(x.dtype)

        if attention_mask is not None:
            x[attention_mask] = 0.0  # turn invalid frames to zero

            attention_mask = attention_mask[:, None, None, :]
            # (batch, 1, num_frames, num_frames)
            attention_mask = (attention_mask * torch.finfo(x.dtype).min).repeat(
                1, 1, attention_mask.size(1), 1
            )
            # TODO: check this
            # attention_mask = (
            #     torch.maximum(attention_mask, attention_mask.transpose(2, 3)) * -1e6
            # )

        x = self.encoder(x, mask=attention_mask)

        return x
