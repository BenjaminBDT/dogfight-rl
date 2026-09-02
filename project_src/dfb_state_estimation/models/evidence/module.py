from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dfb_state_estimation.models.audio.module import SingleStepAudioOutput
from dfb_state_estimation.models.vision.module import SingleStepVisionOutput


@dataclass(frozen=True)
class SingleStepEvidenceConfig:
    visual_embedding_dim: int = 128
    audio_embedding_dim: int = 64
    hidden_dim: int = 256
    position_refine_scale: float = 1.0


@dataclass(frozen=True)
class EvidenceStateOutput:
    relative_position: Tensor
    relative_orientation: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor
    pos_valid: Tensor
    ori_valid: Tensor
    pos_valid_probability: Tensor
    ori_valid_probability: Tensor


@dataclass(frozen=True)
class EvidenceFeaturesOutput:
    visual_embedding: Tensor
    audio_embedding: Tensor
    view_valid_probability: Tensor
    visual_relative_position: Tensor
    visual_position_confidence: Tensor
    visual_body_pose_9d: Tensor
    visual_geometry_info: Tensor
    audio_relative_position: Tensor
    audio_position_confidence: Tensor
    raw_visual_evidence_strength: Tensor
    raw_audio_evidence_strength: Tensor


@dataclass(frozen=True)
class SingleStepEvidenceOutput:
    evidence_state: EvidenceStateOutput
    evidence: EvidenceFeaturesOutput


class SingleStepEvidenceModule(nn.Module):
    _AUDIO_POSITION_CONFIDENCE_VALID_THRESHOLD = 0.15
    _AUDIO_EVIDENCE_STRENGTH_VALID_THRESHOLD = 0.05

    def __init__(self, config: SingleStepEvidenceConfig | None = None) -> None:
        super().__init__()
        config = config or SingleStepEvidenceConfig()
        self.config = config
        visual_geo_dim = 9 + 4
        visual_dim = config.visual_embedding_dim + 1 + visual_geo_dim
        position_fusion_dim = (
            config.visual_embedding_dim
            + config.audio_embedding_dim
            + visual_geo_dim
            + 3
            + 3
            + 1
            + 1
            + 1
            + 1
        )
        self.visual_trunk = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.visual_position_head = nn.Linear(config.hidden_dim, 3)
        self.visual_orientation_head = nn.Linear(config.hidden_dim, 6)
        self.visual_position_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.visual_orientation_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.visual_view_valid_head = nn.Linear(config.hidden_dim, 1)
        self.visual_orientation_valid_head = nn.Linear(config.hidden_dim, 1)
        self.position_fusion_trunk = nn.Sequential(
            nn.LayerNorm(position_fusion_dim),
            nn.Linear(position_fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.position_refine_head = nn.Linear(config.hidden_dim, 3)
        self.position_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.position_valid_head = nn.Linear(config.hidden_dim, 1)

    def _decode_visual_state(
        self,
        vision_output: SingleStepVisionOutput,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        view_valid = vision_output.selected_candidate.view_valid.unsqueeze(-1)
        pos_valid = vision_output.selected_candidate.pos_valid.unsqueeze(-1)
        ori_valid = vision_output.selected_candidate.ori_valid.unsqueeze(-1)
        visual_geometry_info = torch.stack(
            [
                vision_output.selected_candidate.pnp_success,
                vision_output.selected_candidate.reprojection_error,
                vision_output.selected_candidate.v_sup,
                vision_output.selected_candidate.v_rep,
            ],
            dim=-1,
        ) * view_valid
        visual_input = torch.cat(
            [
                vision_output.selected_candidate.visual_embedding * view_valid,
                vision_output.selected_candidate.raw_visual_evidence_strength.unsqueeze(-1) * view_valid,
                vision_output.selected_candidate.body_pose_9d * pos_valid,
                visual_geometry_info,
            ],
            dim=-1,
        )
        visual_hidden = self.visual_trunk(visual_input)
        visual_relative_position = self.visual_position_head(visual_hidden) * pos_valid
        visual_relative_orientation = self.visual_orientation_head(visual_hidden) * ori_valid
        visual_position_confidence = torch.sigmoid(
            self.visual_position_confidence_head(visual_hidden).squeeze(-1)
        ) * vision_output.selected_candidate.pos_valid
        visual_orientation_confidence = torch.sigmoid(
            self.visual_orientation_confidence_head(visual_hidden).squeeze(-1)
        ) * vision_output.selected_candidate.ori_valid
        view_valid_probability = torch.sigmoid(
            self.visual_view_valid_head(visual_hidden).squeeze(-1)
        )
        ori_valid_probability = torch.sigmoid(
            self.visual_orientation_valid_head(visual_hidden).squeeze(-1)
        )
        return (
            visual_relative_position,
            visual_relative_orientation,
            visual_position_confidence,
            visual_orientation_confidence,
            visual_geometry_info,
            view_valid_probability,
            ori_valid_probability,
        )

    def _decode_audio_position(
        self,
        audio_output: SingleStepAudioOutput,
    ) -> tuple[Tensor, Tensor]:
        audio_relative_position = (
            F.normalize(audio_output.doa_unit_vector_body, dim=-1, eps=1e-6)
            * torch.exp(audio_output.log_distance_scalar).unsqueeze(-1)
        )
        audio_position_confidence = torch.sqrt(
            (audio_output.doa_conf * audio_output.dist_conf).clamp_min(0.0)
        )
        return audio_relative_position, audio_position_confidence

    def forward(
        self,
        vision_output: SingleStepVisionOutput,
        audio_output: SingleStepAudioOutput,
    ) -> SingleStepEvidenceOutput:
        (
            visual_relative_position,
            visual_relative_orientation,
            visual_position_confidence,
            visual_orientation_confidence,
            visual_geometry_info,
            view_valid_probability,
            visual_ori_valid_probability,
        ) = self._decode_visual_state(vision_output)
        audio_relative_position, audio_position_confidence = self._decode_audio_position(
            audio_output
        )
        audio_pos_valid = (
            (audio_position_confidence > self._AUDIO_POSITION_CONFIDENCE_VALID_THRESHOLD)
            & (audio_output.raw_audio_evidence_strength > self._AUDIO_EVIDENCE_STRENGTH_VALID_THRESHOLD)
        ).to(dtype=audio_position_confidence.dtype)
        audio_relative_position = audio_relative_position * audio_pos_valid.unsqueeze(-1)
        audio_position_confidence = audio_position_confidence * audio_pos_valid

        visual_weight = visual_position_confidence.unsqueeze(-1)
        audio_weight = audio_position_confidence.unsqueeze(-1)
        fused_position_base = (
            visual_weight * visual_relative_position
            + audio_weight * audio_relative_position
        ) / (visual_weight + audio_weight + 1e-6)

        fusion_input = torch.cat(
            [
                vision_output.selected_candidate.visual_embedding,
                audio_output.audio_embedding,
                vision_output.selected_candidate.body_pose_9d,
                visual_geometry_info,
                visual_relative_position,
                audio_relative_position,
                visual_position_confidence.unsqueeze(-1),
                audio_position_confidence.unsqueeze(-1),
                vision_output.selected_candidate.raw_visual_evidence_strength.unsqueeze(-1),
                audio_output.raw_audio_evidence_strength.unsqueeze(-1),
            ],
            dim=-1,
        )
        fusion_hidden = self.position_fusion_trunk(fusion_input)
        fused_relative_position = fused_position_base + (
            self.position_refine_head(fusion_hidden) * float(self.config.position_refine_scale)
        )
        pos_valid_probability = torch.sigmoid(
            self.position_valid_head(fusion_hidden).squeeze(-1)
        )
        evidence_pos_valid = torch.maximum(
            vision_output.selected_candidate.pos_valid,
            audio_pos_valid,
        )
        evidence_ori_valid = vision_output.selected_candidate.ori_valid
        fused_relative_position = fused_relative_position * evidence_pos_valid.unsqueeze(-1)
        visual_relative_orientation = (
            visual_relative_orientation * evidence_ori_valid.unsqueeze(-1)
        )
        fused_position_confidence = torch.sigmoid(
            self.position_confidence_head(fusion_hidden).squeeze(-1)
        ) * evidence_pos_valid
        visual_orientation_confidence = visual_orientation_confidence * evidence_ori_valid
        evidence_state = EvidenceStateOutput(
            relative_position=fused_relative_position,
            relative_orientation=visual_relative_orientation,
            position_confidence=fused_position_confidence,
            orientation_confidence=visual_orientation_confidence,
            pos_valid=evidence_pos_valid,
            ori_valid=evidence_ori_valid,
            pos_valid_probability=pos_valid_probability,
            ori_valid_probability=visual_ori_valid_probability,
        )
        evidence = EvidenceFeaturesOutput(
            visual_embedding=vision_output.selected_candidate.visual_embedding
            * vision_output.selected_candidate.view_valid.unsqueeze(-1),
            audio_embedding=audio_output.audio_embedding,
            view_valid_probability=view_valid_probability,
            visual_relative_position=visual_relative_position,
            visual_position_confidence=visual_position_confidence,
            visual_body_pose_9d=vision_output.selected_candidate.body_pose_9d
            * vision_output.selected_candidate.pos_valid.unsqueeze(-1),
            visual_geometry_info=visual_geometry_info,
            audio_relative_position=audio_relative_position,
            audio_position_confidence=audio_position_confidence,
            raw_visual_evidence_strength=vision_output.selected_candidate.raw_visual_evidence_strength
            * vision_output.selected_candidate.view_valid,
            raw_audio_evidence_strength=audio_output.raw_audio_evidence_strength,
        )
        return SingleStepEvidenceOutput(
            evidence_state=evidence_state,
            evidence=evidence,
        )
