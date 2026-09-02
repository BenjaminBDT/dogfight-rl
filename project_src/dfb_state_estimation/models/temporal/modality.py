from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TemporalModalityConfig:
    state_dim: int = 13
    visual_dim: int = 132
    visual_seg_diff_dim: int = 17
    visual_kp_delta_dim: int = 20
    audio_dim: int = 85
    time_dim: int = 2
    hidden_dim: int = 256


@dataclass(frozen=True)
class TemporalModalityInputs:
    relative_position: Tensor
    relative_orientation: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor
    pos_valid: Tensor
    ori_valid: Tensor
    visual_embedding: Tensor
    audio_embedding: Tensor
    raw_visual_evidence_strength: Tensor
    view_valid: Tensor
    selected_segmentation_difference_t: Tensor
    selected_segmentation_diff_valid_t: Tensor
    selected_keypoint_delta_t: Tensor
    selected_keypoint_delta_support_summary_t: Tensor
    selected_keypoint_delta_valid_t: Tensor
    raw_audio_evidence_strength: Tensor
    binaural_energy_t: Tensor
    binaural_cue_vector_t: Tensor
    delta_binaural_cue_t: Tensor
    dt_to_prev: Tensor
    time_from_now: Tensor


@dataclass(frozen=True)
class TemporalModalityProjectionOutput:
    state_tokens: Tensor
    visual_tokens: Tensor
    audio_tokens: Tensor
    stacked_tokens: Tensor


@dataclass(frozen=True)
class TemporalModalityBackboneOutput:
    hidden_tokens: Tensor
    state_hidden: Tensor
    visual_hidden: Tensor
    audio_hidden: Tensor


@dataclass(frozen=True)
class CoarseStateOutput:
    relative_position: Tensor
    relative_orientation: Tensor
    position_confidence: Tensor
    orientation_confidence: Tensor


@dataclass(frozen=True)
class TemporalModalityStageOutput:
    projected_tokens: TemporalModalityProjectionOutput
    backbone: TemporalModalityBackboneOutput
    coarse_state: CoarseStateOutput
    visual_evidence_strength: Tensor
    audio_evidence_strength: Tensor


def select_view_target_probability(
    front_segmentation_logits: Tensor,
    rear_segmentation_logits: Tensor,
    target_class_ids: Tensor,
    selected_view_index: Tensor,
) -> Tensor:
    if front_segmentation_logits.shape != rear_segmentation_logits.shape:
        raise ValueError("front/rear segmentation logits must share shape")
    if front_segmentation_logits.ndim != 4:
        raise ValueError("segmentation logits must be [B, C, H, W]")
    batch_size, num_classes, _, _ = front_segmentation_logits.shape
    if target_class_ids.shape != (batch_size,):
        raise ValueError("target_class_ids must be [B]")
    if selected_view_index.shape != (batch_size,):
        raise ValueError("selected_view_index must be [B]")

    front_probs = torch.softmax(front_segmentation_logits, dim=1)
    rear_probs = torch.softmax(rear_segmentation_logits, dim=1)
    gather_index = target_class_ids.to(device=front_probs.device, dtype=torch.long).view(
        batch_size, 1, 1, 1
    )
    gather_index = gather_index.expand(-1, 1, front_probs.shape[2], front_probs.shape[3])
    front_target = front_probs.gather(dim=1, index=gather_index).squeeze(1)
    rear_target = rear_probs.gather(dim=1, index=gather_index).squeeze(1)
    selected = torch.zeros_like(front_target)
    front_mask = selected_view_index == 0
    rear_mask = selected_view_index == 1
    if front_mask.any():
        selected[front_mask] = front_target[front_mask]
    if rear_mask.any():
        selected[rear_mask] = rear_target[rear_mask]
    return selected


def compute_selected_segmentation_difference_t(
    selected_target_probability_t: Tensor,
    selected_view_index_t: Tensor,
) -> tuple[Tensor, Tensor]:
    if selected_target_probability_t.ndim != 4:
        raise ValueError("selected_target_probability_t must be [B, T, H, W]")
    if selected_view_index_t.ndim != 2:
        raise ValueError("selected_view_index_t must be [B, T]")
    batch_size, steps, height, width = selected_target_probability_t.shape
    if selected_view_index_t.shape != (batch_size, steps):
        raise ValueError("selected_view_index_t must match [B, T]")

    same_view = torch.zeros(
        (batch_size, steps),
        dtype=selected_target_probability_t.dtype,
        device=selected_target_probability_t.device,
    )
    difference = torch.zeros(
        (batch_size, steps, 2, height, width),
        dtype=selected_target_probability_t.dtype,
        device=selected_target_probability_t.device,
    )
    if steps <= 1:
        return difference, same_view

    prev_view = selected_view_index_t[:, :-1]
    curr_view = selected_view_index_t[:, 1:]
    valid_prev = prev_view < 2
    valid_curr = curr_view < 2
    same = (prev_view == curr_view) & valid_prev & valid_curr
    same_view[:, 1:] = same.to(dtype=selected_target_probability_t.dtype)

    prev_mask = selected_target_probability_t[:, :-1]
    curr_mask = selected_target_probability_t[:, 1:]
    appear = torch.relu(curr_mask - prev_mask)
    disappear = torch.relu(prev_mask - curr_mask)
    same_mask = same.to(dtype=selected_target_probability_t.dtype).unsqueeze(-1).unsqueeze(-1)
    difference[:, 1:, 0] = appear * same_mask
    difference[:, 1:, 1] = disappear * same_mask
    return difference, same_view


class SegmentationDifferenceEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        hidden_dim = max(output_dim - 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, hidden_dim)

    def forward(
        self,
        selected_segmentation_difference_t: Tensor,
        selected_segmentation_diff_valid_t: Tensor,
    ) -> Tensor:
        if selected_segmentation_difference_t.ndim != 5:
            raise ValueError("selected_segmentation_difference_t must be [B, T, 2, H, W]")
        if selected_segmentation_diff_valid_t.ndim != 2:
            raise ValueError("selected_segmentation_diff_valid_t must be [B, T]")
        batch_size, steps, channels, _, _ = selected_segmentation_difference_t.shape
        if channels != 2:
            raise ValueError("selected_segmentation_difference_t channel dim must be 2")
        if selected_segmentation_diff_valid_t.shape != (batch_size, steps):
            raise ValueError("selected_segmentation_diff_valid_t must match [B, T]")
        flat = selected_segmentation_difference_t.reshape(batch_size * steps, channels, *selected_segmentation_difference_t.shape[-2:])
        encoded = self.conv(flat).flatten(1)
        encoded = self.proj(encoded).reshape(batch_size, steps, -1)
        diff_valid = selected_segmentation_diff_valid_t.unsqueeze(-1).to(dtype=encoded.dtype)
        encoded = encoded * diff_valid
        return torch.cat([encoded, diff_valid], dim=-1)


def select_view_tensor(
    front_value: Tensor,
    rear_value: Tensor,
    selected_view_index: Tensor,
) -> Tensor:
    if front_value.shape != rear_value.shape:
        raise ValueError("front_value and rear_value must share shape")
    if selected_view_index.ndim != 1 or selected_view_index.shape[0] != front_value.shape[0]:
        raise ValueError("selected_view_index must be [B] and match batch dim")
    selected = torch.zeros_like(front_value)
    front_mask = selected_view_index == 0
    rear_mask = selected_view_index == 1
    if front_mask.any():
        selected[front_mask] = front_value[front_mask]
    if rear_mask.any():
        selected[rear_mask] = rear_value[rear_mask]
    return selected


def compute_selected_keypoint_delta_t(
    selected_keypoints_xy_t: Tensor,
    selected_keypoint_support_t: Tensor,
    selected_view_index_t: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if selected_keypoints_xy_t.ndim != 4:
        raise ValueError("selected_keypoints_xy_t must be [B, T, K, 2]")
    if selected_keypoint_support_t.ndim != 3:
        raise ValueError("selected_keypoint_support_t must be [B, T, K]")
    if selected_view_index_t.ndim != 2:
        raise ValueError("selected_view_index_t must be [B, T]")
    batch_size, steps, num_keypoints, xy_dim = selected_keypoints_xy_t.shape
    if xy_dim != 2:
        raise ValueError("selected_keypoints_xy_t last dim must be 2")
    if selected_keypoint_support_t.shape != (batch_size, steps, num_keypoints):
        raise ValueError("selected_keypoint_support_t must match [B, T, K]")
    if selected_view_index_t.shape != (batch_size, steps):
        raise ValueError("selected_view_index_t must match [B, T]")

    delta = torch.zeros_like(selected_keypoints_xy_t)
    support_summary = torch.zeros(
        (batch_size, steps),
        dtype=selected_keypoint_support_t.dtype,
        device=selected_keypoint_support_t.device,
    )
    valid = torch.zeros(
        (batch_size, steps),
        dtype=selected_keypoint_support_t.dtype,
        device=selected_keypoint_support_t.device,
    )
    if steps <= 1:
        return delta, support_summary, valid

    prev_view = selected_view_index_t[:, :-1]
    curr_view = selected_view_index_t[:, 1:]
    same = (prev_view == curr_view) & (prev_view < 2) & (curr_view < 2)
    valid[:, 1:] = same.to(dtype=selected_keypoint_support_t.dtype)
    delta[:, 1:] = (
        selected_keypoints_xy_t[:, 1:] - selected_keypoints_xy_t[:, :-1]
    ) * valid[:, 1:].unsqueeze(-1).unsqueeze(-1)
    prev_support_mean = selected_keypoint_support_t[:, :-1].mean(dim=-1)
    curr_support_mean = selected_keypoint_support_t[:, 1:].mean(dim=-1)
    support_summary[:, 1:] = 0.5 * (prev_support_mean + curr_support_mean) * valid[:, 1:]
    return delta, support_summary, valid


class TemporalModalityProjection(nn.Module):
    def __init__(self, config: TemporalModalityConfig | None = None) -> None:
        super().__init__()
        config = config or TemporalModalityConfig()
        self.config = config
        self.state_proj = self._build_projector(config.state_dim + config.time_dim)
        self.visual_seg_diff_encoder = SegmentationDifferenceEncoder(config.visual_seg_diff_dim)
        self.visual_proj = self._build_projector(
            config.visual_dim + config.visual_seg_diff_dim + config.visual_kp_delta_dim + config.time_dim
        )
        self.audio_proj = self._build_projector(config.audio_dim + config.time_dim)
        self.state_token_type = nn.Parameter(torch.zeros(config.hidden_dim))
        self.visual_token_type = nn.Parameter(torch.zeros(config.hidden_dim))
        self.audio_token_type = nn.Parameter(torch.zeros(config.hidden_dim))

    def _build_projector(self, input_dim: int) -> nn.Sequential:
        hidden_dim = self.config.hidden_dim
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        inputs: TemporalModalityInputs,
    ) -> TemporalModalityProjectionOutput:
        time_feature = torch.stack([inputs.dt_to_prev, inputs.time_from_now], dim=-1)

        state_feature = torch.cat(
            [
                inputs.relative_position,
                inputs.relative_orientation,
                inputs.position_confidence.unsqueeze(-1),
                inputs.orientation_confidence.unsqueeze(-1),
                inputs.pos_valid.unsqueeze(-1),
                inputs.ori_valid.unsqueeze(-1),
                time_feature,
            ],
            dim=-1,
        )
        visual_feature = torch.cat(
            [
                inputs.visual_embedding,
                inputs.raw_visual_evidence_strength.unsqueeze(-1),
                inputs.view_valid.unsqueeze(-1),
                inputs.pos_valid.unsqueeze(-1),
                inputs.ori_valid.unsqueeze(-1),
                self.visual_seg_diff_encoder(
                    inputs.selected_segmentation_difference_t,
                    inputs.selected_segmentation_diff_valid_t,
                ),
                inputs.selected_keypoint_delta_t.flatten(start_dim=-2),
                inputs.selected_keypoint_delta_support_summary_t.unsqueeze(-1),
                inputs.selected_keypoint_delta_valid_t.unsqueeze(-1),
                time_feature,
            ],
            dim=-1,
        )
        audio_feature = torch.cat(
            [
                inputs.audio_embedding,
                inputs.raw_audio_evidence_strength.unsqueeze(-1),
                inputs.binaural_energy_t,
                inputs.binaural_cue_vector_t,
                inputs.delta_binaural_cue_t,
                time_feature,
            ],
            dim=-1,
        )

        state_tokens = self.state_proj(state_feature) + self.state_token_type
        visual_tokens = self.visual_proj(visual_feature) + self.visual_token_type
        audio_tokens = self.audio_proj(audio_feature) + self.audio_token_type

        stacked_tokens = torch.stack(
            [state_tokens, visual_tokens, audio_tokens],
            dim=2,
        )
        return TemporalModalityProjectionOutput(
            state_tokens=state_tokens,
            visual_tokens=visual_tokens,
            audio_tokens=audio_tokens,
            stacked_tokens=stacked_tokens,
        )


class TemporalModalityTransformer(nn.Module):
    def __init__(
        self,
        config: TemporalModalityConfig | None = None,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = config or TemporalModalityConfig()
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

    def forward(self, stacked_tokens: Tensor) -> TemporalModalityBackboneOutput:
        if stacked_tokens.ndim != 4:
            raise ValueError("stacked_tokens must be [B, T, 3, H]")
        batch, steps, token_types, hidden_dim = stacked_tokens.shape
        if token_types != 3:
            raise ValueError("expected three token types per step")
        if hidden_dim != self.config.hidden_dim:
            raise ValueError(
                f"expected hidden dim {self.config.hidden_dim}, got {hidden_dim}"
            )
        flat_tokens = stacked_tokens.reshape(batch, steps * token_types, hidden_dim)
        causal_mask = self._build_block_causal_mask(
            steps=steps,
            token_types=token_types,
            device=flat_tokens.device,
            dtype=flat_tokens.dtype,
        )
        encoded = self.encoder(flat_tokens, mask=causal_mask)
        hidden_tokens = encoded.reshape(batch, steps, token_types, hidden_dim)
        return TemporalModalityBackboneOutput(
            hidden_tokens=hidden_tokens,
            state_hidden=hidden_tokens[:, :, 0, :],
            visual_hidden=hidden_tokens[:, :, 1, :],
            audio_hidden=hidden_tokens[:, :, 2, :],
        )

    @staticmethod
    def _build_block_causal_mask(
        *,
        steps: int,
        token_types: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        total_tokens = steps * token_types
        mask = torch.zeros((total_tokens, total_tokens), device=device, dtype=dtype)
        step_indices = torch.arange(total_tokens, device=device) // token_types
        future_step = step_indices.unsqueeze(0) > step_indices.unsqueeze(1)
        mask = mask.masked_fill(future_step, float("-inf"))
        return mask


def compute_delta_binaural_cue_t(binaural_cue_vector_t: Tensor) -> Tensor:
    if binaural_cue_vector_t.ndim != 3:
        raise ValueError("binaural_cue_vector_t must be [B, T, C]")
    delta = torch.zeros_like(binaural_cue_vector_t)
    delta[:, 1:, :] = binaural_cue_vector_t[:, 1:, :] - binaural_cue_vector_t[:, :-1, :]
    return delta


class TemporalModalityCalibrationHeads(nn.Module):
    def __init__(self, config: TemporalModalityConfig | None = None) -> None:
        super().__init__()
        config = config or TemporalModalityConfig()
        self.config = config
        self.coarse_state_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.relative_position_head = nn.Linear(config.hidden_dim, 3)
        self.relative_orientation_head = nn.Linear(config.hidden_dim, 6)
        self.position_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.orientation_confidence_head = nn.Linear(config.hidden_dim, 1)
        self.visual_evidence_strength_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )
        self.audio_evidence_strength_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        backbone_output: TemporalModalityBackboneOutput,
    ) -> tuple[CoarseStateOutput, Tensor, Tensor]:
        state_hidden_t = backbone_output.state_hidden[:, -1, :]
        visual_hidden_t = backbone_output.visual_hidden[:, -1, :]
        audio_hidden_t = backbone_output.audio_hidden[:, -1, :]
        coarse_hidden = self.coarse_state_head(state_hidden_t)
        coarse_state = CoarseStateOutput(
            relative_position=self.relative_position_head(coarse_hidden),
            relative_orientation=self.relative_orientation_head(coarse_hidden),
            position_confidence=torch.sigmoid(
                self.position_confidence_head(coarse_hidden).squeeze(-1)
            ),
            orientation_confidence=torch.sigmoid(
                self.orientation_confidence_head(coarse_hidden).squeeze(-1)
            ),
        )
        visual_evidence_strength = torch.sigmoid(
            self.visual_evidence_strength_head(visual_hidden_t).squeeze(-1)
        )
        audio_evidence_strength = torch.sigmoid(
            self.audio_evidence_strength_head(audio_hidden_t).squeeze(-1)
        )
        return coarse_state, visual_evidence_strength, audio_evidence_strength


class TemporalModalityCalibrationStage(nn.Module):
    def __init__(
        self,
        config: TemporalModalityConfig | None = None,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config = config or TemporalModalityConfig()
        self.projection = TemporalModalityProjection(config)
        self.backbone = TemporalModalityTransformer(
            config,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.heads = TemporalModalityCalibrationHeads(config)

    def forward(
        self,
        inputs: TemporalModalityInputs,
    ) -> TemporalModalityStageOutput:
        projected = self.projection(inputs)
        backbone_output = self.backbone(projected.stacked_tokens)
        coarse_state, visual_evidence_strength, audio_evidence_strength = self.heads(
            backbone_output
        )
        return TemporalModalityStageOutput(
            projected_tokens=projected,
            backbone=backbone_output,
            coarse_state=coarse_state,
            visual_evidence_strength=visual_evidence_strength,
            audio_evidence_strength=audio_evidence_strength,
        )
