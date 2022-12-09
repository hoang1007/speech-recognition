"""
This file contains the implementation of the Transformer Encoder layer.
Source: https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/components.py
"""
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module


class SelfAttention(Module):
    """Multihead Self Attention module
    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(
                f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). "
                f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(
                    f"The expected attention mask shape is {shape_}. "
                    f"Found {attention_mask.size()}."
                )

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        # scale down q to avoid value overflow.
        weights = (self.scaling * q) @ k  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask
        # subtracting a constant value from the tensor won't change the output of softmax.
        # apply the subtraction to avoid value overflow in torch.nn.functional.softmax.
        # for more details, please see Equation 7 in https://arxiv.org/abs/2112.08778
        weights = weights - weights.max(dim=-1, keepdim=True)[0]

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output, None  # Necessary for compatibility with WavLMSelAttention


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        layer_norm_first: bool,
        feed_forward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = SelfAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = FeedForward(d_model, feed_forward_dim, dropout, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x, position_bias = self.attention(
            x,
            attention_mask=attention_mask,
            position_bias=position_bias,
            key_padding_mask=key_padding_mask,
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x, position_bias
