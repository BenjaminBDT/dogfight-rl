from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import AudioBackbone, AudioBackboneConfig
from .heads import AudioCueEncoder, AudioEmbeddingHead, AudioFusionTrunk, AudioHeadConfig


@dataclass(frozen=True)
class SingleStepAudioConfig:
    backbone: AudioBackboneConfig = field(default_factory=AudioBackboneConfig)
    heads: AudioHeadConfig = field(default_factory=AudioHeadConfig)
    energy_scale: float = 0.05
    gcc_scale: float = 0.5
    ild_scale: float = 3.0
    reverb_scale: float = 0.5
    directness_scale: float = 0.5


@dataclass(frozen=True)
class SingleStepAudioOutput:
    doa_unit_vector_body: Tensor
    doa_conf: Tensor
    log_distance_scalar: Tensor
    dist_conf: Tensor
    binaural_energy_t: Tensor
    binaural_cue_vector_t: Tensor
    raw_audio_evidence_strength: Tensor
    audio_embedding: Tensor


def compute_audio_evidence_terms(
    *,
    binaural_energy_t: Tensor,
    binaural_cue_vector_t: Tensor,
    config: SingleStepAudioConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    energy_sum = torch.relu(binaural_energy_t[:, 2])
    gcc_peak_value = torch.relu(binaural_cue_vector_t[:, 1])
    ild_abs = binaural_cue_vector_t[:, 2].abs()
    interaural_coherence = binaural_cue_vector_t[:, 5].clamp(0.0, 1.0)
    reverb_ratio_proxy = torch.relu(binaural_cue_vector_t[:, 6])
    directness_proxy = torch.relu(binaural_cue_vector_t[:, 7])
    ild_low_abs = binaural_cue_vector_t[:, 8].abs()
    ild_high_abs = binaural_cue_vector_t[:, 9].abs()

    a_energy = 1.0 - torch.exp(-energy_sum / config.energy_scale)
    s_gcc = 1.0 - torch.exp(-gcc_peak_value / config.gcc_scale)
    s_ild = 1.0 - torch.exp(
        -torch.maximum(ild_abs, torch.maximum(ild_low_abs, ild_high_abs)) / config.ild_scale
    )
    s_reverb = 1.0 - torch.exp(-reverb_ratio_proxy / config.reverb_scale)
    s_directness = 1.0 - torch.exp(-directness_proxy / config.directness_scale)
    s_direct = 0.5 * (s_reverb + s_directness)
    a_cue = (s_gcc * interaural_coherence * s_direct * s_ild).clamp_min(0.0).pow(1.0 / 4.0)
    raw_audio_evidence_strength = 0.5 * a_energy + 0.5 * a_cue
    return a_energy, a_cue, raw_audio_evidence_strength


class SingleStepAudioModule(nn.Module):
    def __init__(self, config: SingleStepAudioConfig | None = None) -> None:
        super().__init__()
        config = config or SingleStepAudioConfig()
        self.config = config
        self.backbone = AudioBackbone(config.backbone)
        self.cue_encoder = AudioCueEncoder(config.heads)
        self.fusion_trunk = AudioFusionTrunk(self.backbone.out_channels, config.heads)
        self.embedding_head = AudioEmbeddingHead(config.heads)
        self.spatial_trunk = nn.Sequential(
            nn.LayerNorm(config.heads.hidden_dim),
            nn.Linear(config.heads.hidden_dim, config.heads.hidden_dim),
            nn.GELU(),
        )
        self.doa_vector_head = nn.Linear(config.heads.hidden_dim, 3)
        self.doa_conf_head = nn.Linear(config.heads.hidden_dim, 1)
        self.distance_head = nn.Linear(config.heads.hidden_dim, 1)
        self.dist_conf_head = nn.Linear(config.heads.hidden_dim, 1)

    def forward(
        self,
        audio_window_binaural: Tensor,
        binaural_energy_t: Tensor,
        binaural_cue_vector_t: Tensor,
    ) -> SingleStepAudioOutput:
        if audio_window_binaural.ndim != 3 or audio_window_binaural.shape[-1] != 2:
            raise ValueError("audio_window_binaural must be [B, T, 2]")
        if binaural_energy_t.ndim != 2 or binaural_energy_t.shape[-1] != 4:
            raise ValueError("binaural_energy_t must be [B, 4]")
        if binaural_cue_vector_t.ndim != 2 or binaural_cue_vector_t.shape[-1] != 10:
            raise ValueError("binaural_cue_vector_t must be [B, 10]")

        waveform_feature = self.backbone(audio_window_binaural)
        structured_feature = torch.cat([binaural_energy_t, binaural_cue_vector_t], dim=-1)
        cue_embedding = self.cue_encoder(structured_feature)
        audio_latent = self.fusion_trunk(waveform_feature, cue_embedding)
        audio_embedding = self.embedding_head(audio_latent)
        spatial_hidden = self.spatial_trunk(audio_latent)
        doa_unit_vector_body = F.normalize(
            self.doa_vector_head(spatial_hidden),
            dim=-1,
            eps=1e-6,
        )
        doa_conf = torch.sigmoid(self.doa_conf_head(spatial_hidden).squeeze(-1))
        log_distance_scalar = self.distance_head(spatial_hidden).squeeze(-1)
        dist_conf = torch.sigmoid(self.dist_conf_head(spatial_hidden).squeeze(-1))

        _, _, raw_audio_evidence_strength = compute_audio_evidence_terms(
            binaural_energy_t=binaural_energy_t,
            binaural_cue_vector_t=binaural_cue_vector_t,
            config=self.config,
        )

        return SingleStepAudioOutput(
            doa_unit_vector_body=doa_unit_vector_body,
            doa_conf=doa_conf,
            log_distance_scalar=log_distance_scalar,
            dist_conf=dist_conf,
            binaural_energy_t=binaural_energy_t,
            binaural_cue_vector_t=binaural_cue_vector_t,
            raw_audio_evidence_strength=raw_audio_evidence_strength,
            audio_embedding=audio_embedding,
        )
