from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class _Conv1DLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        """
        Args:
            x (Tensor): The ouput. Shape: (batch, in_channels, in_frames)
            length (Tensor): The valid length of each sample. Shape: (batch)

        Returns:
            x (Tensor): The output. Shape: (batch, out_channels, out_frames)
            length (Tensor): The valid length of each sample. Shape: (batch)
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

        self.layernorm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, length: torch.Tensor):
        x = self.conv(x)
        x = x.transpose_(1, 2)
        x = self.layernorm(x)
        x = x.transpose_(1, 2)
        x = F.gelu(x)

        length = (length - self.kernel_size) // self.stride + 1
        length = length.clamp_min_(min=0)  # prevent negative lengths
        return x, length


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        """
        Extracts features from the waveform.

        Args:
            waveforms (Tensor): The waveform to extract features from. Shape: (batch, wavelength)
            wavelength (Tensor): The valid length of each waveform. Shape: (batch)

        Returns:
            features (Tensor): The extracted features. Shape: (batch, num_frames, num_channels)
            num_frames (Tensor): The valid length of each feature. Shape: (batch)
        """
        super().__init__()

        num_channels = config.num_channels
        kernel_sizes = config.kernel_sizes
        strides = config.strides

        assert (
            len(num_channels) == len(kernel_sizes) == len(strides)
        ), "The number of layers must be the same for all parameters"

        self.conv_layers = nn.ModuleList(
            (
                _Conv1DLayer(
                    in_channels=1,
                    out_channels=num_channels[0],
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                ),
            )
        )

        for i in range(1, len(num_channels)):
            self.conv_layers.append(
                _Conv1DLayer(
                    in_channels=num_channels[i - 1],
                    out_channels=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                )
            )

    def forward(self, waveforms: torch.Tensor, wavelength: torch.Tensor):
        features = waveforms.unsqueeze(1)

        for conv_layer in self.conv_layers:
            features, wavelength = conv_layer(features, wavelength)

        # (batch, num_channels, num_frames) -> (batch, num_frames, num_channels)
        features = features.transpose(1, 2)
        return features, wavelength
