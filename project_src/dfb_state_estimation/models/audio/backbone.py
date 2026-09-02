from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AudioBackboneConfig:
    in_channels: int = 4
    hidden_channels: int = 32
    out_channels: int = 64
    kernel_size: int = 5


class AudioBackbone(nn.Module):
    def __init__(self, config: AudioBackboneConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv1d(
                config.in_channels,
                config.hidden_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(config.hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                config.hidden_channels,
                config.out_channels,
                kernel_size=config.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(config.out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    @property
    def out_channels(self) -> int:
        return self.config.out_channels

    def forward(self, audio_window_binaural: Tensor) -> Tensor:
        if audio_window_binaural.ndim != 3:
            raise ValueError("audio_window_binaural must be [B, T, 2]")
        if audio_window_binaural.shape[-1] != 2:
            raise ValueError("audio_window_binaural must have trailing channel dim 2")
        left = audio_window_binaural[..., 0]
        right = audio_window_binaural[..., 1]
        waveform_feature = torch.stack(
            [
                left,
                right,
                left + right,
                left - right,
            ],
            dim=1,
        )
        if waveform_feature.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected trailing audio channel dim {self.config.in_channels}, "
                f"got {waveform_feature.shape[1]}"
            )
        encoded = self.encoder(waveform_feature)
        pooled = self.pool(encoded).squeeze(-1)
        return pooled
