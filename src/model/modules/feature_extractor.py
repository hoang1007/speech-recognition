from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class TransposeLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layernorm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3, "Input must be 3D tensor."

        x = x.transpose_(1, 2)
        x = self.layernorm(x)
        x = x.transpose_(1, 2)
        return x


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: str = "none",
    ):
        """
        A single convolutional layer.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int): The stride of the convolution.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            bias=False,
        )

        if norm_type == "layer":
            self.layer_norm = TransposeLayerNorm(out_channels)
        elif norm_type == "group":
            self.layer_norm = nn.GroupNorm(out_channels, out_channels, affine=True)
        elif norm_type == "none":
            self.layer_norm = nn.Identity()
        else:
            raise ValueError(
                f"norm_type must be one of 'none', 'layer', 'group'. Got {norm_type}"
            )
        
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        """
        Args:
            x (Tensor): The ouput. Shape: (batch, in_channels, in_frames)
            length (Tensor): The valid length of each sample. Shape: (batch)

        Returns:
            x (Tensor): The output. Shape: (batch, out_channels, out_frames)
            length (Tensor): The valid length of each sample. Shape: (batch)
        """

        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.activation(x)

        length = (
            torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
        )
        # prevent negative output length when input length is 0
        length = length.clamp_min_(min=0)
        return x, length


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()

        assert (
            len(hidden_channels) == len(kernel_sizes) == len(strides)
        ), "The number of layers must be the same for all parameters"

        self.conv_layers = nn.ModuleList(
            (
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels[0],
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    norm_type="group",
                ),
            )
        )

        for i in range(1, len(hidden_channels)):
            self.conv_layers.append(
                Conv1DBlock(
                    in_channels=hidden_channels[i - 1],
                    out_channels=hidden_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                )
            )

    def forward(self, waveforms: torch.Tensor, wavelength: torch.Tensor):
        """
        Extracts features from the waveform.

        Args:
            waveforms (Tensor): The waveform to extract features from. Shape: (batch, wavelength)
            wavelength (Tensor): The valid length of each waveform. Shape: (batch)

        Returns:
            features (Tensor): The extracted features. Shape: (batch, num_frames, num_channels)
            num_frames (Tensor): The valid length of each feature. Shape: (batch)
        """

        features = waveforms.unsqueeze(1)
        seq_length = wavelength

        for conv_layer in self.conv_layers:
            features, seq_length = conv_layer(features, seq_length)

        # (batch, num_channels, num_frames) -> (batch, num_frames, num_channels)
        features = features.transpose(1, 2)
        return features, seq_length
