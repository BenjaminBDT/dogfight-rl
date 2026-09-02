from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.temporal import TemporalModalityStageOutput


@dataclass(frozen=True)
class TemporalSupervisionTargets:
    gt_relative_position: Tensor
    gt_relative_orientation: Tensor
    target_pos_conf: Tensor
    target_ori_conf: Tensor


@dataclass(frozen=True)
class TemporalLossWeights:
    position: float = 1.0
    orientation: float = 1.0
    position_confidence: float = 0.5
    orientation_confidence: float = 0.5


def compute_temporal_modality_loss(
    output: TemporalModalityStageOutput,
    targets: TemporalSupervisionTargets,
    *,
    weights: TemporalLossWeights = TemporalLossWeights(),
) -> dict[str, Tensor]:
    position_loss = F.smooth_l1_loss(
        output.coarse_state.relative_position,
        targets.gt_relative_position,
    )
    orientation_loss = F.smooth_l1_loss(
        output.coarse_state.relative_orientation,
        targets.gt_relative_orientation,
    )
    position_confidence_loss = F.mse_loss(
        output.coarse_state.position_confidence,
        targets.target_pos_conf,
    )
    orientation_confidence_loss = F.mse_loss(
        output.coarse_state.orientation_confidence,
        targets.target_ori_conf,
    )
    total = (
        weights.position * position_loss
        + weights.orientation * orientation_loss
        + weights.position_confidence * position_confidence_loss
        + weights.orientation_confidence * orientation_confidence_loss
    )
    return {
        "total": total,
        "position": position_loss,
        "orientation": orientation_loss,
        "position_confidence": position_confidence_loss,
        "orientation_confidence": orientation_confidence_loss,
    }
