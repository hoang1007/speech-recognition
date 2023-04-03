from typing import Tuple, List
import torch
from torch import nn


class Wav2Vec2Processor(nn.Module):
    def __init__(self):
        """
        Convert waveforms whose length is different to a batch.
        """
        super().__init__()

    def forward(self, waveforms: List[torch.Tensor]):
        """
        Args:
            waveforms (Tuple[torch.Tensor]): The waveforms. Shape: (batch_size, wave_length).

        Returns:
            waveforms (torch.Tensor): The batched waveforms. Shape: (batch_size, max_wave_length).
            wave_lengths (torch.Tensor): The wave length of each waveform. Shape: (batch_size,).
        """
        for waveform in waveforms:
            assert waveform.ndim == 2, "waveform must be 2D tensor"
            assert waveform.size(0) == 1, "Only support mono waveform"

        device = waveforms[0].device
        wave_lengths = torch.tensor(
            tuple(waveform.size(1) for waveform in waveforms), device=device
        )

        max_length = int(wave_lengths.max().item())

        padded: List[torch.Tensor] = []

        for waveform in waveforms:
            padded.append(
                nn.functional.pad(
                    waveform,
                    (0, max_length - waveform.size(1)),
                    mode="constant",
                    value=0.0,
                )
            )

        batched_waveforms = torch.cat(padded, dim=0)

        return batched_waveforms, wave_lengths
