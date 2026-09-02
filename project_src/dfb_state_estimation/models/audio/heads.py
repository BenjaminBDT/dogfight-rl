from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AudioHeadConfig:
    structured_feature_dim: int = 14
    hidden_dim: int = 128
    embedding_dim: int = 64


class AudioCueEncoder(nn.Module):
    def __init__(
        self,
        config: AudioHeadConfig,
    ) -> None:
        super().__init__()
        self.structured_norm = nn.LayerNorm(config.structured_feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.structured_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, structured_feature: Tensor) -> Tensor:
        structured_feature = self.structured_norm(structured_feature)
        return self.proj(structured_feature)


class AudioFusionTrunk(nn.Module):
    def __init__(self, backbone_channels: int, config: AudioHeadConfig) -> None:
        super().__init__()
        self.waveform_norm = nn.LayerNorm(backbone_channels)
        self.cue_norm = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(backbone_channels + config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        waveform_feature: Tensor,
        cue_embedding: Tensor,
    ) -> Tensor:
        waveform_feature = self.waveform_norm(waveform_feature)
        cue_embedding = self.cue_norm(cue_embedding)
        fused = torch.cat([waveform_feature, cue_embedding], dim=-1)
        return self.proj(fused)


class AudioEmbeddingHead(nn.Module):
    def __init__(self, config: AudioHeadConfig) -> None:
        super().__init__()
        self.latent_norm = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

    def forward(self, audio_latent: Tensor) -> Tensor:
        audio_latent = self.latent_norm(audio_latent)
        return self.proj(audio_latent)
