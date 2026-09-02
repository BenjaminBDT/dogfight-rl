from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
import torch
from torch import Tensor, nn

from .modality import CoarseStateOutput


@dataclass(frozen=True)
class BeliefUpdateConfig:
    token_dim: int = 28
    time_dim: int = 2
    hidden_dim: int = 256


@dataclass(frozen=True)
class BeliefUpdateInputs:
    coarse_state_t: CoarseStateOutput
    context_relative_position: Tensor
    context_relative_orientation: Tensor
    context_position_confidence: Tensor
    context_orientation_confidence: Tensor
    visual_evidence_strength_t: Tensor
    audio_evidence_strength_t: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    dt_to_prev: Tensor
    time_from_now: Tensor


@dataclass(frozen=True)
class BeliefUpdateTokenOutput:
    delta_position: Tensor
    delta_orientation: Tensor
    belief_update_tokens: Tensor


@dataclass(frozen=True)
class BeliefUpdateBackboneOutput:
    hidden_states: Tensor


@dataclass(frozen=True)
class BeliefStateOutput:
    relative_position: Tensor
    relative_orientation: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor
    track_confidence: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor


@dataclass(frozen=True)
class TemporalBeliefUpdateStageOutput:
    token_output: BeliefUpdateTokenOutput
    backbone: BeliefUpdateBackboneOutput
    belief_state: BeliefStateOutput


@dataclass(frozen=True)
class PolicyViewOutput:
    relative_position: Tensor
    relative_orientation: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    track_confidence: Tensor


class BeliefUpdateTokenBuilder(nn.Module):
    def __init__(self, config: BeliefUpdateConfig | None = None) -> None:
        super().__init__()
        config = config or BeliefUpdateConfig()
        self.config = config
        self.projector = nn.Sequential(
            nn.LayerNorm(config.token_dim + config.time_dim),
            nn.Linear(config.token_dim + config.time_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.token_type = nn.Parameter(torch.zeros(config.hidden_dim))

    def forward(self, inputs: BeliefUpdateInputs) -> BeliefUpdateTokenOutput:
        coarse_position = self._replace_last_step(
            inputs.context_relative_position,
            inputs.coarse_state_t.relative_position,
        )
        coarse_orientation = self._replace_last_step(
            inputs.context_relative_orientation,
            inputs.coarse_state_t.relative_orientation,
        )
        coarse_position_confidence = self._replace_last_step_scalar(
            inputs.context_position_confidence,
            inputs.coarse_state_t.position_confidence,
        )
        coarse_orientation_confidence = self._replace_last_step_scalar(
            inputs.context_orientation_confidence,
            inputs.coarse_state_t.orientation_confidence,
        )
        visual_evidence_strength = self._last_step_only_scalar(
            inputs.visual_evidence_strength_t,
            inputs.time_from_now,
        )
        audio_evidence_strength = self._last_step_only_scalar(
            inputs.audio_evidence_strength_t,
            inputs.time_from_now,
        )
        delta_position = self._delta(coarse_position)
        delta_orientation = self._delta(coarse_orientation)
        time_feature = torch.stack([inputs.dt_to_prev, inputs.time_from_now], dim=-1)
        token_feature = torch.cat(
            [
                coarse_position,
                coarse_orientation,
                coarse_position_confidence.unsqueeze(-1),
                coarse_orientation_confidence.unsqueeze(-1),
                visual_evidence_strength.unsqueeze(-1),
                audio_evidence_strength.unsqueeze(-1),
                delta_position,
                delta_orientation,
                inputs.linear_velocity,
                inputs.angular_velocity,
                time_feature,
            ],
            dim=-1,
        )
        belief_update_tokens = self.projector(token_feature) + self.token_type
        return BeliefUpdateTokenOutput(
            delta_position=delta_position,
            delta_orientation=delta_orientation,
            belief_update_tokens=belief_update_tokens,
        )

    @staticmethod
    def _delta(sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must be [B, T, D]")
        delta = sequence[:, 1:, :] - sequence[:, :-1, :]
        zeros = torch.zeros_like(sequence[:, :1, :])
        return torch.cat([zeros, delta], dim=1)

    @staticmethod
    def _replace_last_step(sequence: Tensor, current_value: Tensor) -> Tensor:
        if sequence.ndim != 3 or current_value.ndim != 2:
            raise ValueError("expected [B, T, D] sequence and [B, D] current_value")
        updated = sequence.clone()
        updated[:, -1, :] = current_value
        return updated

    @staticmethod
    def _replace_last_step_scalar(sequence: Tensor, current_value: Tensor) -> Tensor:
        if sequence.ndim != 2 or current_value.ndim != 1:
            raise ValueError("expected [B, T] sequence and [B] current_value")
        updated = sequence.clone()
        updated[:, -1] = current_value
        return updated

    @staticmethod
    def _last_step_only_scalar(current_value: Tensor, time_from_now: Tensor) -> Tensor:
        if current_value.ndim != 1 or time_from_now.ndim != 2:
            raise ValueError("expected [B] current_value and [B, T] time_from_now")
        output = torch.zeros_like(time_from_now)
        output[:, -1] = current_value
        return output


class TemporalBeliefUpdateTransformer(nn.Module):
    def __init__(
        self,
        config: BeliefUpdateConfig | None = None,
        *,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = config or BeliefUpdateConfig()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, belief_update_tokens: Tensor) -> BeliefUpdateBackboneOutput:
        if belief_update_tokens.ndim != 3:
            raise ValueError("belief_update_tokens must be [B, T, H]")
        if belief_update_tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"expected hidden dim {self.config.hidden_dim}, "
                f"got {belief_update_tokens.shape[-1]}"
            )
        steps = belief_update_tokens.shape[1]
        causal_mask = self._build_causal_mask(
            steps,
            device=belief_update_tokens.device,
            dtype=belief_update_tokens.dtype,
        )
        hidden_states = self.encoder(belief_update_tokens, mask=causal_mask)
        return BeliefUpdateBackboneOutput(hidden_states=hidden_states)

    @staticmethod
    def _build_causal_mask(
        steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mask = torch.full((steps, steps), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)


class BeliefStateHeads(nn.Module):
    def __init__(self, config: BeliefUpdateConfig | None = None) -> None:
        super().__init__()
        config = config or BeliefUpdateConfig()
        self.config = config
        self.state_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.relative_position_head = nn.Linear(config.hidden_dim, 3)
        self.relative_orientation_head = nn.Linear(config.hidden_dim, 6)
        self.position_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.orientation_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.track_confidence_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        backbone_output: BeliefUpdateBackboneOutput,
        *,
        context_relative_position: Tensor,
        context_relative_orientation: Tensor,
        dt_to_prev: Tensor,
    ) -> BeliefStateOutput:
        previous_position = self._previous_step_or_current(
            context_relative_position,
            fallback_dim=3,
        )
        previous_orientation = self._previous_step_or_current(
            context_relative_orientation,
            fallback_dim=6,
        )
        hidden_t = self.state_head(backbone_output.hidden_states[:, -1, :])
        relative_position = self.relative_position_head(hidden_t)
        relative_orientation = self.relative_orientation_head(hidden_t)
        position_confidence = torch.sigmoid(
            self.position_confidence_head(hidden_t).squeeze(-1)
        )
        orientation_confidence = torch.sigmoid(
            self.orientation_confidence_head(hidden_t).squeeze(-1)
        )
        track_confidence = torch.sigmoid(
            self.track_confidence_head(hidden_t).squeeze(-1)
        )
        linear_velocity = self._derive_linear_velocity(
            relative_position,
            previous_position,
            dt_to_prev[:, -1],
        )
        angular_velocity = self._derive_angular_velocity(
            relative_orientation,
            previous_orientation,
            dt_to_prev[:, -1],
        )
        return BeliefStateOutput(
            relative_position=relative_position,
            relative_orientation=relative_orientation,
            position_confidence=position_confidence,
            orientation_confidence=orientation_confidence,
            track_confidence=track_confidence,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

    @staticmethod
    def _previous_step_or_current(sequence: Tensor, *, fallback_dim: int) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("expected [B, T, D] sequence")
        if sequence.shape[-1] != fallback_dim:
            raise ValueError(f"expected trailing dim {fallback_dim}")
        if sequence.shape[1] >= 2:
            return sequence[:, -2, :]
        return sequence[:, -1, :]

    @staticmethod
    def _derive_linear_velocity(
        relative_position_t: Tensor,
        previous_position: Tensor,
        dt_to_prev_t: Tensor,
    ) -> Tensor:
        delta = relative_position_t - previous_position
        dt = dt_to_prev_t.unsqueeze(-1).clamp_min(1e-6)
        return delta / dt

    @classmethod
    def _derive_angular_velocity(
        cls,
        relative_orientation_t: Tensor,
        previous_orientation: Tensor,
        dt_to_prev_t: Tensor,
    ) -> Tensor:
        prev = cls._rotation_6d_to_matrix(previous_orientation)
        curr = cls._rotation_6d_to_matrix(relative_orientation_t)
        rel = torch.matmul(curr, prev.transpose(-1, -2))
        rotvec = cls._so3_log_map(rel)
        dt = dt_to_prev_t.unsqueeze(-1).clamp_min(1e-6)
        return rotvec / dt

    @staticmethod
    def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
        a1 = rotation_6d[..., 0:3]
        a2 = rotation_6d[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        proj = (b1 * a2).sum(dim=-1, keepdim=True)
        b2 = F.normalize(a2 - proj * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def _so3_log_map(rotation_matrix: Tensor) -> Tensor:
        trace = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        vee = torch.stack(
            [
                rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2],
                rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0],
                rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1],
            ],
            dim=-1,
        )
        sin_theta = torch.sin(theta).unsqueeze(-1)
        scale = theta.unsqueeze(-1) / (2.0 * sin_theta.clamp_min(1e-6))
        return scale * vee


class TemporalBeliefUpdateStage(nn.Module):
    def __init__(
        self,
        config: BeliefUpdateConfig | None = None,
        *,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = config or BeliefUpdateConfig()
        self.token_builder = BeliefUpdateTokenBuilder(config)
        self.backbone = TemporalBeliefUpdateTransformer(
            config,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.heads = BeliefStateHeads(config)

    def forward(self, inputs: BeliefUpdateInputs) -> TemporalBeliefUpdateStageOutput:
        token_output = self.token_builder(inputs)
        backbone_output = self.backbone(token_output.belief_update_tokens)
        belief_state = self.heads(
            backbone_output,
            context_relative_position=inputs.context_relative_position,
            context_relative_orientation=inputs.context_relative_orientation,
            dt_to_prev=inputs.dt_to_prev,
        )
        return TemporalBeliefUpdateStageOutput(
            token_output=token_output,
            backbone=backbone_output,
            belief_state=belief_state,
        )


class PolicyViewAdapter(nn.Module):
    def forward(self, belief_state: BeliefStateOutput) -> PolicyViewOutput:
        return PolicyViewOutput(
            relative_position=belief_state.relative_position,
            relative_orientation=belief_state.relative_orientation,
            position_confidence=belief_state.position_confidence,
            orientation_confidence=belief_state.orientation_confidence,
            linear_velocity=belief_state.linear_velocity,
            angular_velocity=belief_state.angular_velocity,
            track_confidence=belief_state.track_confidence,
        )
