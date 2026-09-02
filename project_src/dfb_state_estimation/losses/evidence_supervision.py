from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.evidence import SingleStepEvidenceOutput


@dataclass(frozen=True)
class EvidenceSupervisionTargets:
    gt_relative_position: Tensor
    gt_relative_orientation: Tensor
    target_pos_conf: Tensor
    target_ori_conf: Tensor
    view_valid_target: Tensor
    pos_valid_target: Tensor
    ori_valid_target: Tensor


@dataclass(frozen=True)
class EvidenceLossWeights:
    position: float = 1.0
    orientation: float = 1.0
    position_confidence: float = 0.5
    orientation_confidence: float = 0.5
    view_valid: float = 0.5
    pos_valid: float = 0.5
    ori_valid: float = 0.5
    valid_positive_weight: float = 2.0


def _masked_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    if prediction.shape[:-1] != mask.shape and prediction.shape != mask.shape:
        raise ValueError("mask shape must match prediction batch dims")
    expanded_mask = mask
    while expanded_mask.ndim < prediction.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.to(dtype=prediction.dtype)
    valid = expanded_mask.sum()
    if float(valid.detach().cpu().item()) == 0.0:
        return prediction.new_zeros(())
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * expanded_mask
    return loss.sum() / expanded_mask.sum().clamp_min(1.0)


def _masked_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    if prediction.shape != mask.shape:
        raise ValueError("mask shape must match scalar prediction shape")
    mask = mask.to(dtype=prediction.dtype)
    valid = mask.sum()
    if float(valid.detach().cpu().item()) == 0.0:
        return prediction.new_zeros(())
    loss = F.mse_loss(prediction, target, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _weighted_bce(
    prediction: Tensor,
    target: Tensor,
    *,
    positive_weight: float,
) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    weight = torch.where(
        target > 0.5,
        prediction.new_full(prediction.shape, float(positive_weight)),
        prediction.new_ones(prediction.shape),
    )
    return F.binary_cross_entropy(prediction, target, weight=weight)


def compute_single_step_evidence_loss(
    output: SingleStepEvidenceOutput,
    targets: EvidenceSupervisionTargets,
    *,
    weights: EvidenceLossWeights = EvidenceLossWeights(),
) -> dict[str, Tensor]:
    position_loss = _masked_smooth_l1(
        output.evidence_state.relative_position,
        targets.gt_relative_position,
        targets.pos_valid_target,
    )
    orientation_loss = _masked_smooth_l1(
        output.evidence_state.relative_orientation,
        targets.gt_relative_orientation,
        targets.ori_valid_target,
    )
    position_confidence_loss = _masked_mse(
        output.evidence_state.position_confidence,
        targets.target_pos_conf,
        targets.pos_valid_target,
    )
    orientation_confidence_loss = _masked_mse(
        output.evidence_state.orientation_confidence,
        targets.target_ori_conf,
        targets.ori_valid_target,
    )
    view_valid_loss = _weighted_bce(
        output.evidence.view_valid_probability,
        targets.view_valid_target,
        positive_weight=weights.valid_positive_weight,
    )
    pos_valid_loss = _weighted_bce(
        output.evidence_state.pos_valid_probability,
        targets.pos_valid_target,
        positive_weight=weights.valid_positive_weight,
    )
    ori_valid_loss = _weighted_bce(
        output.evidence_state.ori_valid_probability,
        targets.ori_valid_target,
        positive_weight=weights.valid_positive_weight,
    )
    total = (
        weights.position * position_loss
        + weights.orientation * orientation_loss
        + weights.position_confidence * position_confidence_loss
        + weights.orientation_confidence * orientation_confidence_loss
        + weights.view_valid * view_valid_loss
        + weights.pos_valid * pos_valid_loss
        + weights.ori_valid * ori_valid_loss
    )
    return {
        "total": total,
        "position": position_loss,
        "orientation": orientation_loss,
        "position_confidence": position_confidence_loss,
        "orientation_confidence": orientation_confidence_loss,
        "view_valid": view_valid_loss,
        "pos_valid": pos_valid_loss,
        "ori_valid": ori_valid_loss,
    }
