from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.audio import SingleStepAudioOutput


@dataclass(frozen=True)
class AudioSupervisionTargets:
    gt_doa_unit_vector_body: Tensor
    gt_log_distance_scalar: Tensor


@dataclass(frozen=True)
class AudioLossWeights:
    doa: float = 1.0
    distance: float = 1.0
    doa_confidence: float = 0.5
    distance_confidence: float = 0.5


@dataclass(frozen=True)
class AudioConfidenceConfig:
    doa_half_angle: float = math.radians(15.0)
    dist_half_error: float = 0.25


def compute_audio_confidence_targets(
    output: SingleStepAudioOutput,
    targets: AudioSupervisionTargets,
    *,
    config: AudioConfidenceConfig = AudioConfidenceConfig(),
) -> dict[str, Tensor]:
    pred_doa = F.normalize(output.doa_unit_vector_body, dim=-1, eps=1e-6)
    gt_doa = F.normalize(targets.gt_doa_unit_vector_body, dim=-1, eps=1e-6)
    doa_dot = (pred_doa * gt_doa).sum(dim=-1).clamp(-1.0, 1.0)
    doa_angle_error = doa_dot.acos()
    log_distance_error = (output.log_distance_scalar - targets.gt_log_distance_scalar).abs()
    target_doa_conf = torch.exp(
        -math.log(2.0) * (doa_angle_error / config.doa_half_angle).pow(2)
    ).clamp(0.0, 1.0)
    target_dist_conf = torch.exp(
        -math.log(2.0) * (log_distance_error / config.dist_half_error).pow(2)
    ).clamp(0.0, 1.0)
    return {
        "doa_angle_error": doa_angle_error,
        "log_distance_error": log_distance_error,
        "target_doa_conf": target_doa_conf,
        "target_dist_conf": target_dist_conf,
    }


def compute_single_step_audio_loss(
    output: SingleStepAudioOutput,
    targets: AudioSupervisionTargets,
    *,
    weights: AudioLossWeights = AudioLossWeights(),
    confidence: AudioConfidenceConfig = AudioConfidenceConfig(),
) -> dict[str, Tensor]:
    confidence_targets = compute_audio_confidence_targets(
        output,
        targets,
        config=confidence,
    )
    doa_loss = F.smooth_l1_loss(
        output.doa_unit_vector_body,
        targets.gt_doa_unit_vector_body,
    )
    distance_loss = F.smooth_l1_loss(
        output.log_distance_scalar,
        targets.gt_log_distance_scalar,
    )
    doa_confidence_loss = F.mse_loss(
        output.doa_conf,
        confidence_targets["target_doa_conf"],
    )
    distance_confidence_loss = F.mse_loss(
        output.dist_conf,
        confidence_targets["target_dist_conf"],
    )
    total = (
        weights.doa * doa_loss
        + weights.distance * distance_loss
        + weights.doa_confidence * doa_confidence_loss
        + weights.distance_confidence * distance_confidence_loss
    )
    return {
        "total": total,
        "doa": doa_loss,
        "distance": distance_loss,
        "doa_confidence": doa_confidence_loss,
        "distance_confidence": distance_confidence_loss,
        "doa_angle_error": confidence_targets["doa_angle_error"].mean(),
        "log_distance_error": confidence_targets["log_distance_error"].mean(),
        "target_doa_conf_mean": confidence_targets["target_doa_conf"].mean(),
        "target_dist_conf_mean": confidence_targets["target_dist_conf"].mean(),
    }
