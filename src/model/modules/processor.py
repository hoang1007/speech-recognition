from typing import Tuple
import torch
from torch import nn


class Wav2Vec2Processor(nn.Module):
    def __init__(self):
        """
        Convert tuple of waveforms whose length is different to a batch.

        Args:
            waveforms (Tuple[torch.Tensor]): The waveforms. Shape: (batch_size, wave_length).

        Returns:
            waveforms (torch.Tensor): The batched waveforms. Shape: (batch_size, max_wave_length).
            wave_lengths (torch.Tensor): The wave length of each waveform. Shape: (batch_size,).
        """
        super().__init__()

    def forward(self, waveforms: Tuple[torch.Tensor, ...]):
        device = waveforms[0].device
        wave_lengths = torch.tensor(
            tuple(waveform.size(0) for waveform in waveforms), device=device
        )

        max_length = wave_lengths.max().item()

        padded = []

        for waveform in waveforms:
            padded.append(
                nn.functional.pad(
                    waveform,
                    (0, max_length - waveform.size(0)),
                    mode="constant",
                    value=0.0,
                )
            )

        batched_waveforms = torch.stack(padded, dim=0)

        return batched_waveforms, wave_lengths
