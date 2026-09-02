from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.vision.module import SingleStepVisionOutput
from dfb_state_estimation.models.vision.single_view_segmentation_module import (
    SingleViewSegmentationOutput,
)


@dataclass(frozen=True)
class VisionSupervisionTargets:
    target_class_ids: Tensor
    front_segmentation: Tensor
    rear_segmentation: Tensor
    front_keypoints_xy: Tensor
    rear_keypoints_xy: Tensor
    front_keypoint_xy_mask: Tensor
    rear_keypoint_xy_mask: Tensor
    front_keypoint_voting_pixels: Tensor
    rear_keypoint_voting_pixels: Tensor
    front_keypoint_voting_unit_vectors: Tensor
    rear_keypoint_voting_unit_vectors: Tensor
    front_keypoint_voting_mask: Tensor
    rear_keypoint_voting_mask: Tensor
    front_keypoint_xy_weights: Tensor | None = None
    rear_keypoint_xy_weights: Tensor | None = None
    front_segmentation_valid: Tensor | None = None
    rear_segmentation_valid: Tensor | None = None


@dataclass(frozen=True)
class SingleViewSegmentationTargets:
    target_class_ids: Tensor
    segmentation: Tensor


@dataclass(frozen=True)
class VisionLossWeights:
    segmentation: float = 1.0
    keypoints: float = 1.0
    keypoint_voting: float = 1.0
    segmentation_background: float = 1.0
    segmentation_other_aircraft: float = 1.0
    segmentation_target: float = 1.0
    segmentation_loss_mode: str = "ce"
    segmentation_focal_gamma: float = 2.0
    segmentation_dice_weight: float = 0.0


def _target_aware_segmentation_loss_per_sample(
    logits: Tensor,
    target: Tensor,
    target_class_ids: Tensor,
    *,
    background_weight: float,
    other_aircraft_weight: float,
    target_weight: float,
    loss_mode: str,
    focal_gamma: float,
) -> Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must be [B, C, H, W]")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("target segmentation shape must match logits spatial shape")
    if target_class_ids.shape != (logits.shape[0],):
        raise ValueError("target_class_ids must be [B]")
    log_probs = F.log_softmax(logits, dim=1)
    gathered_log_probs = log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1)
    gathered = -gathered_log_probs
    pt = gathered_log_probs.exp()

    class_weights = torch.full(
        (logits.shape[0], logits.shape[1]),
        float(other_aircraft_weight),
        dtype=logits.dtype,
        device=logits.device,
    )
    class_weights[:, 0] = float(background_weight)
    class_weights.scatter_(
        1,
        target_class_ids.long().unsqueeze(1),
        torch.full(
            (logits.shape[0], 1),
            float(target_weight),
            dtype=logits.dtype,
            device=logits.device,
        ),
    )
    pixel_weights = class_weights.gather(1, target.long().view(logits.shape[0], -1)).view_as(target)
    if loss_mode == "ce":
        per_pixel = gathered
    elif loss_mode == "focal":
        focal_factor = (1.0 - pt).clamp_min(0.0).pow(float(focal_gamma))
        per_pixel = focal_factor * gathered
    else:
        raise ValueError(f"unsupported segmentation loss mode: {loss_mode}")
    weighted = per_pixel * pixel_weights
    return weighted.flatten(1).sum(dim=1) / pixel_weights.flatten(1).sum(dim=1).clamp_min(1.0)


def _target_soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    target_class_ids: Tensor,
    *,
    smooth: float = 1.0,
) -> Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must be [B, C, H, W]")
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("target segmentation shape must match logits spatial shape")
    if target_class_ids.shape != (logits.shape[0],):
        raise ValueError("target_class_ids must be [B]")
    probs = torch.softmax(logits, dim=1)
    gather_index = target_class_ids.long().view(-1, 1, 1, 1).expand(-1, 1, logits.shape[2], logits.shape[3])
    target_probs = probs.gather(1, gather_index).squeeze(1)
    target_mask = (
        target == target_class_ids.view(-1, 1, 1).to(device=target.device, dtype=target.dtype)
    ).to(dtype=logits.dtype)
    intersection = (target_probs * target_mask).sum(dim=(1, 2))
    denom = target_probs.sum(dim=(1, 2)) + target_mask.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice


def _masked_keypoint_l1(
    pred_xy: Tensor,
    target_xy: Tensor,
    supervision_weights: Tensor,
) -> Tensor:
    mask = supervision_weights.unsqueeze(-1).to(dtype=pred_xy.dtype)
    denom = mask.sum().clamp_min(1.0)
    return ((pred_xy - target_xy).abs() * mask).sum() / denom


def _sparse_voting_smooth_l1(
    pred_voting_field: Tensor,
    voting_pixels: Tensor,
    target_unit_vectors: Tensor,
    voting_mask: Tensor,
) -> Tensor:
    batch_size, num_keypoints, _, height, width = pred_voting_field.shape
    if voting_pixels.shape[1] == 0:
        return pred_voting_field.new_zeros(())
    pred_vectors = pred_voting_field.permute(0, 3, 4, 1, 2)
    pred_vectors = torch.nn.functional.normalize(pred_vectors, dim=-1, eps=1.0e-6)

    batch_index = torch.arange(batch_size, device=pred_voting_field.device).view(batch_size, 1)
    x = voting_pixels[..., 0].long().clamp_(0, width - 1)
    y = voting_pixels[..., 1].long().clamp_(0, height - 1)
    sampled = pred_vectors[batch_index, y, x]

    per_component = torch.nn.functional.smooth_l1_loss(
        sampled,
        target_unit_vectors,
        reduction="none",
    )
    per_vector = per_component.mean(dim=-1)
    mask = voting_mask.to(dtype=pred_voting_field.dtype).unsqueeze(-1)
    denom = (mask.sum() * num_keypoints).clamp_min(1.0)
    return (per_vector * mask).sum() / denom


def compute_single_step_vision_loss(
    output: SingleStepVisionOutput,
    targets: VisionSupervisionTargets,
    *,
    weights: VisionLossWeights = VisionLossWeights(),
) -> dict[str, Tensor]:
    front_seg_loss_per_sample = _target_aware_segmentation_loss_per_sample(
        output.front_segmentation_logits,
        targets.front_segmentation.long(),
        targets.target_class_ids.long(),
        background_weight=weights.segmentation_background,
        other_aircraft_weight=weights.segmentation_other_aircraft,
        target_weight=weights.segmentation_target,
        loss_mode=weights.segmentation_loss_mode,
        focal_gamma=weights.segmentation_focal_gamma,
    )
    rear_seg_loss_per_sample = _target_aware_segmentation_loss_per_sample(
        output.rear_segmentation_logits,
        targets.rear_segmentation.long(),
        targets.target_class_ids.long(),
        background_weight=weights.segmentation_background,
        other_aircraft_weight=weights.segmentation_other_aircraft,
        target_weight=weights.segmentation_target,
        loss_mode=weights.segmentation_loss_mode,
        focal_gamma=weights.segmentation_focal_gamma,
    )
    front_kp_loss = _masked_keypoint_l1(
        output.front_keypoints_xy,
        targets.front_keypoints_xy,
        targets.front_keypoint_xy_weights
        if targets.front_keypoint_xy_weights is not None
        else targets.front_keypoint_xy_mask.to(dtype=output.front_keypoints_xy.dtype),
    )
    rear_kp_loss = _masked_keypoint_l1(
        output.rear_keypoints_xy,
        targets.rear_keypoints_xy,
        targets.rear_keypoint_xy_weights
        if targets.rear_keypoint_xy_weights is not None
        else targets.rear_keypoint_xy_mask.to(dtype=output.rear_keypoints_xy.dtype),
    )
    front_voting_loss = _sparse_voting_smooth_l1(
        output.front_voting_field,
        targets.front_keypoint_voting_pixels,
        targets.front_keypoint_voting_unit_vectors,
        targets.front_keypoint_voting_mask,
    )
    rear_voting_loss = _sparse_voting_smooth_l1(
        output.rear_voting_field,
        targets.rear_keypoint_voting_pixels,
        targets.rear_keypoint_voting_unit_vectors,
        targets.rear_keypoint_voting_mask,
    )
    front_valid = (
        targets.front_segmentation_valid.to(device=output.front_segmentation_logits.device, dtype=output.front_segmentation_logits.dtype)
        if targets.front_segmentation_valid is not None
        else torch.ones_like(front_seg_loss_per_sample, dtype=output.front_segmentation_logits.dtype)
    )
    rear_valid = (
        targets.rear_segmentation_valid.to(device=output.rear_segmentation_logits.device, dtype=output.rear_segmentation_logits.dtype)
        if targets.rear_segmentation_valid is not None
        else torch.ones_like(rear_seg_loss_per_sample, dtype=output.rear_segmentation_logits.dtype)
    )
    segmentation_terms: list[Tensor] = []
    if bool((front_valid > 0).any()):
        segmentation_terms.append(
            (front_seg_loss_per_sample * front_valid).sum() / front_valid.sum().clamp_min(1.0)
        )
    if bool((rear_valid > 0).any()):
        segmentation_terms.append(
            (rear_seg_loss_per_sample * rear_valid).sum() / rear_valid.sum().clamp_min(1.0)
        )
    segmentation_loss = (
        torch.stack(segmentation_terms).mean()
        if segmentation_terms
        else output.front_segmentation_logits.new_zeros(())
    )
    if weights.segmentation_dice_weight > 0.0:
        front_dice_loss = _target_soft_dice_loss(
            output.front_segmentation_logits,
            targets.front_segmentation.long(),
            targets.target_class_ids.long(),
        )
        rear_dice_loss = _target_soft_dice_loss(
            output.rear_segmentation_logits,
            targets.rear_segmentation.long(),
            targets.target_class_ids.long(),
        )
        dice_terms: list[Tensor] = []
        if bool((front_valid > 0).any()):
            dice_terms.append(
                (front_dice_loss * front_valid).sum() / front_valid.sum().clamp_min(1.0)
            )
        if bool((rear_valid > 0).any()):
            dice_terms.append(
                (rear_dice_loss * rear_valid).sum() / rear_valid.sum().clamp_min(1.0)
            )
        if dice_terms:
            segmentation_loss = segmentation_loss + weights.segmentation_dice_weight * torch.stack(dice_terms).mean()
    keypoint_loss = 0.5 * (front_kp_loss + rear_kp_loss)
    keypoint_voting_loss = 0.5 * (front_voting_loss + rear_voting_loss)
    total = (
        weights.segmentation * segmentation_loss
        + weights.keypoints * keypoint_loss
        + weights.keypoint_voting * keypoint_voting_loss
    )
    return {
        "total": total,
        "segmentation": segmentation_loss,
        "keypoints": keypoint_loss,
        "keypoint_voting": keypoint_voting_loss,
    }


def compute_single_view_segmentation_loss(
    output: SingleViewSegmentationOutput,
    targets: SingleViewSegmentationTargets,
    *,
    weights: VisionLossWeights = VisionLossWeights(),
) -> dict[str, Tensor]:
    segmentation_loss = _target_aware_segmentation_loss_per_sample(
        output.segmentation_logits,
        targets.segmentation.long(),
        targets.target_class_ids.long(),
        background_weight=weights.segmentation_background,
        other_aircraft_weight=weights.segmentation_other_aircraft,
        target_weight=weights.segmentation_target,
        loss_mode=weights.segmentation_loss_mode,
        focal_gamma=weights.segmentation_focal_gamma,
    ).mean()
    if weights.segmentation_dice_weight > 0.0:
        dice_loss = _target_soft_dice_loss(
            output.segmentation_logits,
            targets.segmentation.long(),
            targets.target_class_ids.long(),
        ).mean()
        segmentation_loss = segmentation_loss + weights.segmentation_dice_weight * dice_loss
    return {
        "total": weights.segmentation * segmentation_loss,
        "segmentation": segmentation_loss,
    }
