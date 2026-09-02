from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.temporal import TemporalBeliefUpdateStageOutput


@dataclass(frozen=True)
class BeliefSupervisionTargets:
    gt_relative_position: Tensor
    gt_relative_orientation: Tensor
    gt_linear_velocity: Tensor
    gt_angular_velocity: Tensor
    target_pos_conf: Tensor
    target_ori_conf: Tensor


@dataclass(frozen=True)
class BeliefLossWeights:
    position: float = 1.0
    orientation: float = 1.0
    linear_velocity: float = 0.5
    angular_velocity: float = 0.5
    position_confidence: float = 0.5
    orientation_confidence: float = 0.5


def compute_temporal_belief_loss(
    output: TemporalBeliefUpdateStageOutput,
    targets: BeliefSupervisionTargets,
    *,
    weights: BeliefLossWeights = BeliefLossWeights(),
) -> dict[str, Tensor]:
    belief_state = output.belief_state
    position_loss = F.smooth_l1_loss(
        belief_state.relative_position,
        targets.gt_relative_position,
    )
    orientation_loss = F.smooth_l1_loss(
        belief_state.relative_orientation,
        targets.gt_relative_orientation,
    )
    linear_velocity_loss = F.smooth_l1_loss(
        belief_state.linear_velocity,
        targets.gt_linear_velocity,
    )
    angular_velocity_loss = F.smooth_l1_loss(
        belief_state.angular_velocity,
        targets.gt_angular_velocity,
    )
    position_confidence_loss = F.mse_loss(
        belief_state.position_confidence,
        targets.target_pos_conf,
    )
    orientation_confidence_loss = F.mse_loss(
        belief_state.orientation_confidence,
        targets.target_ori_conf,
    )
    total = (
        weights.position * position_loss
        + weights.orientation * orientation_loss
        + weights.linear_velocity * linear_velocity_loss
        + weights.angular_velocity * angular_velocity_loss
        + weights.position_confidence * position_confidence_loss
        + weights.orientation_confidence * orientation_confidence_loss
    )
    return {
        "total": total,
        "position": position_loss,
        "orientation": orientation_loss,
        "linear_velocity": linear_velocity_loss,
        "angular_velocity": angular_velocity_loss,
        "position_confidence": position_confidence_loss,
        "orientation_confidence": orientation_confidence_loss,
    }
