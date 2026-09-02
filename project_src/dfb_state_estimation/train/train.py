from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Sampler

from dfb_state_estimation.datasets import (
    AudioOnlyRecordingDataset,
    remap_segmentation_mask,
    segmentation_num_classes,
    StepDataset,
    SyntheticSegmentationDataset,
    SyntheticSingleStepDataset,
    SyntheticSingleViewSegmentationDataset,
    WindowDataset,
    target_segmentation_class_id,
)
from dfb_state_estimation.losses import (
    AudioConfidenceConfig,
    AudioLossWeights,
    AudioSupervisionTargets,
    BeliefSupervisionTargets,
    EvidenceSupervisionTargets,
    SingleViewSegmentationTargets,
    TemporalSupervisionTargets,
    VisionSupervisionTargets,
    compute_single_view_segmentation_loss,
    compute_single_step_audio_loss,
    compute_single_step_evidence_loss,
    compute_single_step_vision_loss,
    compute_temporal_belief_loss,
    compute_temporal_modality_loss,
)
from dfb_state_estimation.losses.vision_supervision import VisionLossWeights
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.evidence import SingleStepEvidenceConfig, SingleStepEvidenceModule
from dfb_state_estimation.models.temporal import (
    BeliefUpdateInputs,
    TemporalBeliefUpdateStage,
    TemporalModalityCalibrationStage,
    TemporalModalityInputs,
    compute_selected_keypoint_delta_t,
    compute_selected_segmentation_difference_t,
    compute_delta_binaural_cue_t,
    reroute_vision_output_with_inertia,
    select_view_tensor,
    select_view_target_probability,
)
from dfb_state_estimation.models.vision import (
    DeepLabSingleViewSegmentationConfig,
    DeepLabSingleViewSegmentationModule,
    DeepLabSingleStepVisionConfig,
    DeepLabSingleStepVisionModule,
    GeometryValidationConfig,
    SingleViewSegmentationConfig,
    SingleViewSegmentationModule,
    SingleStepVisionConfig,
    SingleStepVisionModule,
    VisionHeadConfig,
)
from dfb_state_estimation.models.vision.backbone import VisionBackboneConfig
from dfb_state_estimation.train.config import TrainConfig, load_train_config


def _single_step_total_loss(
    config: TrainConfig,
    *,
    vision_total: torch.Tensor,
    audio_total: torch.Tensor,
    evidence_total: torch.Tensor,
) -> torch.Tensor:
    weights = config.single_step_loss_weights
    return (
        weights.vision * vision_total
        + weights.audio * audio_total
        + weights.evidence * evidence_total
    )


def _mixed_aggregation_foreground_from_targets(
    targets: VisionSupervisionTargets,
    *,
    mix: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if mix <= 0.0:
        return None, None
    target = targets.target_class_ids.view(-1, 1, 1)
    front_gt = (targets.front_segmentation == target).to(dtype=torch.float32).unsqueeze(1)
    rear_gt = (targets.rear_segmentation == target).to(dtype=torch.float32).unsqueeze(1)
    return front_gt, rear_gt

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Part 2 trainer entry.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["single_step", "segmentation_only", "temporal_modality", "temporal_belief", "audio_only"],
    )
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser


def _rgba_image_to_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.to(dtype=torch.float32)
    else:
        tensor = torch.tensor(image, dtype=torch.float32)
    return tensor[..., :3].permute(2, 0, 1) / 255.0


def _pad_nested_sequence_batch(
    values: list[Any],
    *,
    dtype: torch.dtype,
    trailing_shape: tuple[int, ...],
) -> torch.Tensor:
    max_len = max((len(value) for value in values), default=0)
    output = torch.zeros((len(values), max_len, *trailing_shape), dtype=dtype)
    for batch_index, value in enumerate(values):
        if len(value) == 0:
            continue
        tensor = torch.tensor(value, dtype=dtype)
        expected_shape = (len(value), *trailing_shape)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"expected nested sequence with shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        output[batch_index, : len(value)] = tensor
    return output


def _pad_flat_sequence_batch(
    values: list[Any],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_len = max((len(value) for value in values), default=0)
    output = torch.zeros((len(values), max_len), dtype=dtype)
    for batch_index, value in enumerate(values):
        if len(value) == 0:
            continue
        tensor = torch.tensor(value, dtype=dtype)
        if tensor.ndim != 1 or tensor.shape[0] != len(value):
            raise ValueError(f"expected flat sequence of length {len(value)}, got {tuple(tensor.shape)}")
        output[batch_index, : len(value)] = tensor
    return output


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _orientation_geodesic_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_r = _rotation_6d_to_matrix(prediction)
    target_r = _rotation_6d_to_matrix(target)
    rel = torch.matmul(pred_r.transpose(-1, -2), target_r)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_theta))


def _remap_segmentation_mask_to_tensor(
    mask: Any,
    *,
    observed_role: str,
    label_mode: str,
) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.to(dtype=torch.long)
        if label_mode == "multiclass_absolute":
            return tensor
        if label_mode != "binary_target":
            raise ValueError(f"unsupported segmentation label mode: {label_mode}")
        absolute_target_class_id = target_segmentation_class_id(
            observed_role,
            label_mode="multiclass_absolute",
        )
        return (tensor == int(absolute_target_class_id)).to(dtype=torch.long)
    return torch.tensor(
        remap_segmentation_mask(
            mask,
            observed_role=observed_role,
            label_mode=label_mode,
        ),
        dtype=torch.long,
    )


def _apply_segmentation_roi_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_class_id: int,
    context_scale: float,
    min_crop_size: int,
    square: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3 or mask.ndim != 2:
        raise ValueError("image must be [C,H,W] and mask must be [H,W]")
    height, width = mask.shape
    target_coords = torch.nonzero(mask == int(target_class_id), as_tuple=False)
    if target_coords.numel() == 0:
        return image, mask

    y_min = int(target_coords[:, 0].min().item())
    y_max = int(target_coords[:, 0].max().item())
    x_min = int(target_coords[:, 1].min().item())
    x_max = int(target_coords[:, 1].max().item())

    box_h = max(1, y_max - y_min + 1)
    box_w = max(1, x_max - x_min + 1)
    crop_h = max(float(min_crop_size), float(box_h) * float(context_scale))
    crop_w = max(float(min_crop_size), float(box_w) * float(context_scale))
    if square:
        crop_side = max(crop_h, crop_w)
        crop_h = crop_side
        crop_w = crop_side

    center_y = 0.5 * (y_min + y_max)
    center_x = 0.5 * (x_min + x_max)
    top = int(round(center_y - crop_h / 2.0))
    left = int(round(center_x - crop_w / 2.0))
    crop_h_int = min(height, max(1, int(round(crop_h))))
    crop_w_int = min(width, max(1, int(round(crop_w))))
    top = max(0, min(top, height - crop_h_int))
    left = max(0, min(left, width - crop_w_int))
    bottom = top + crop_h_int
    right = left + crop_w_int

    image_crop = image[:, top:bottom, left:right].unsqueeze(0)
    mask_crop = mask[top:bottom, left:right].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    image_resized = F.interpolate(
        image_crop,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask_resized = F.interpolate(
        mask_crop,
        size=(height, width),
        mode="nearest",
    ).squeeze(0).squeeze(0).to(dtype=mask.dtype)
    return image_resized, mask_resized


def _apply_target_centered_patch_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_class_id: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3 or mask.ndim != 2:
        raise ValueError("image must be [C,H,W] and mask must be [H,W]")
    height, width = mask.shape
    target_coords = torch.nonzero(mask == int(target_class_id), as_tuple=False)
    if target_coords.numel() == 0:
        return image, mask

    center_y = int(torch.round(target_coords[:, 0].to(dtype=torch.float32).mean()).item())
    center_x = int(torch.round(target_coords[:, 1].to(dtype=torch.float32).mean()).item())
    patch = max(8, int(patch_size))
    crop_h = min(height, patch)
    crop_w = min(width, patch)
    top = max(0, min(center_y - crop_h // 2, height - crop_h))
    left = max(0, min(center_x - crop_w // 2, width - crop_w))
    bottom = top + crop_h
    right = left + crop_w

    image_crop = image[:, top:bottom, left:right].unsqueeze(0)
    mask_crop = mask[top:bottom, left:right].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    image_resized = F.interpolate(
        image_crop,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mask_resized = F.interpolate(
        mask_crop,
        size=(height, width),
        mode="nearest",
    ).squeeze(0).squeeze(0).to(dtype=mask.dtype)
    return image_resized, mask_resized


def _mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    ious: list[torch.Tensor] = []
    for class_index in range(num_classes):
        pred_mask = pred == class_index
        target_mask = target == class_index
        intersection = (pred_mask & target_mask).sum().to(dtype=torch.float32)
        union = (pred_mask | target_mask).sum().to(dtype=torch.float32)
        if union.item() == 0.0:
            ious.append(torch.ones((), dtype=torch.float32, device=logits.device))
        else:
            ious.append(intersection / union)
    return torch.stack(ious).mean()


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask
    while expanded_mask.ndim < prediction.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.to(dtype=prediction.dtype)
    if float(expanded_mask.sum().detach().cpu().item()) == 0.0:
        return prediction.new_zeros(())
    loss = torch.abs(prediction - target) * expanded_mask
    return loss.sum() / expanded_mask.sum().clamp_min(1.0)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    if float(mask.sum().detach().cpu().item()) == 0.0:
        return prediction.new_zeros(())
    loss = ((prediction - target) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _target_iou(logits: torch.Tensor, target: torch.Tensor, target_class_ids: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    target_class = target_class_ids.view(-1, 1, 1).to(device=pred.device, dtype=pred.dtype)
    pred_mask = pred == target_class
    target_mask = target == target_class
    intersection = (pred_mask & target_mask).sum(dim=(1, 2)).to(dtype=torch.float32)
    union = (pred_mask | target_mask).sum(dim=(1, 2)).to(dtype=torch.float32)
    iou = torch.where(union > 0.0, intersection / union.clamp_min(1.0), torch.zeros_like(union))
    return iou.mean()


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "train config requested device='cuda', but CUDA is not available in the "
                "current execution environment"
            )
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device setting: {name}")


def _build_optimizer(parameters, *, name: str, lr: float, weight_decay: float):
    if name.lower() != "adam":
        raise ValueError(f"unsupported optimizer: {name}")
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def _step_indices(length: int, *, step: int, batch_size: int, base_index: int = 0) -> list[int]:
    start = (base_index + step * batch_size) % max(length, 1)
    return [((start + offset) % length) for offset in range(batch_size)]


class _WrappedStepBatchSampler(Sampler[list[int]]):
    def __init__(self, *, length: int, batch_size: int, num_steps: int, base_index: int = 0) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self._length = length
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._base_index = base_index

    def __iter__(self):
        for step in range(self._num_steps):
            yield _step_indices(
                self._length,
                step=step,
                batch_size=self._batch_size,
                base_index=self._base_index,
            )

    def __len__(self) -> int:
        return self._num_steps


class _WeightedStepBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        weights: list[float],
        batch_size: int,
        num_steps: int,
        random_seed: int,
        draw_offset: int = 0,
    ) -> None:
        if not weights:
            raise ValueError("weights must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if draw_offset < 0:
            raise ValueError("draw_offset must be >= 0")
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("all weights must be positive")
        self._weights = torch.tensor(weights, dtype=torch.float64)
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._random_seed = random_seed
        self._draw_offset = draw_offset

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._random_seed)
        total_draws = self._draw_offset + self._num_steps * self._batch_size
        draws = torch.multinomial(
            self._weights,
            num_samples=total_draws,
            replacement=True,
            generator=generator,
        ).tolist()
        draws = draws[self._draw_offset :]
        for step_index in range(self._num_steps):
            start = step_index * self._batch_size
            end = start + self._batch_size
            yield draws[start:end]

    def __len__(self) -> int:
        return self._num_steps


def _single_step_sampling_weights(config: TrainConfig, dataset: Any) -> list[float] | None:
    sampling = config.single_step_sampling
    if (
        sampling.small_target_weight <= 1.0
        and sampling.medium_target_weight <= 1.0
    ):
        return None
    active_target_areas = getattr(dataset, "active_target_areas", None)
    if active_target_areas is None:
        return None
    if len(active_target_areas) != len(dataset):
        raise ValueError("single-step dataset active_target_areas length mismatch")
    if (
        sampling.medium_target_area_threshold > 0
        and sampling.small_target_area_threshold > 0
        and sampling.medium_target_area_threshold <= sampling.small_target_area_threshold
    ):
        raise ValueError(
            "single_step_sampling.medium_target_area_threshold must be > "
            "single_step_sampling.small_target_area_threshold"
        )
    weights: list[float] = []
    for area in active_target_areas:
        weight = 1.0
        if (
            sampling.small_target_area_threshold > 0
            and area <= sampling.small_target_area_threshold
        ):
            weight = sampling.small_target_weight
        elif (
            sampling.medium_target_area_threshold > 0
            and area <= sampling.medium_target_area_threshold
        ):
            weight = sampling.medium_target_weight
        weights.append(max(weight, 1e-6))
    if all(abs(weight - 1.0) < 1e-9 for weight in weights):
        return None
    return weights


def _single_step_batch_sampler(
    *,
    dataset: Any,
    config: TrainConfig,
    num_steps: int,
    start_step: int,
) -> Sampler[list[int]]:
    weights = _single_step_sampling_weights(config, dataset)
    if weights is not None:
        return _WeightedStepBatchSampler(
            weights=weights,
            batch_size=config.schedule.batch_size,
            num_steps=num_steps - start_step,
            random_seed=config.seed,
            draw_offset=start_step * config.schedule.batch_size,
        )
    return _WrappedStepBatchSampler(
        length=len(dataset),
        batch_size=config.schedule.batch_size,
        num_steps=num_steps - start_step,
        base_index=config.single_step.step_index + start_step * config.schedule.batch_size,
    )


def _audio_sampling_cache_path(config: TrainConfig, dataset: Any) -> Path:
    payload = {
        "dataset_root": str(getattr(dataset, "dataset_root", config.dataset_root)),
        "audio_sampling": asdict(config.audio_sampling),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    dataset_root = Path(getattr(dataset, "dataset_root", config.dataset_root))
    return dataset_root / f".audio_sampling_weights_{digest}.pt"


def _audio_only_sampling_weights(config: TrainConfig, dataset: Any) -> list[float] | None:
    sampling = config.audio_sampling
    if not sampling.enabled:
        return None
    cache_path = _audio_sampling_cache_path(config, dataset)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        weights = [float(value) for value in cached["weights"]]
        if len(weights) != len(dataset):
            raise ValueError("audio sampling weight cache length mismatch")
        return weights

    weights: list[float] = []
    total = len(dataset)
    for index in range(total):
        sample = dataset[index]
        energy = sample.audio_features["binaural_energy_t"]
        cues = sample.audio_features["binaural_cue_vector_t"]
        energy_sum = float(energy[2])
        ild_abs = abs(float(cues[2]))
        gcc_peak = float(cues[1])
        coherence = float(cues[5])
        ild_low_abs = abs(float(cues[8]))
        ild_high_abs = abs(float(cues[9]))

        weight = 1.0
        if ild_abs <= sampling.low_ild_abs_threshold:
            weight *= sampling.low_ild_weight
        elif ild_abs <= sampling.medium_ild_abs_threshold:
            weight *= sampling.medium_ild_weight
        if gcc_peak >= sampling.high_gcc_peak_threshold:
            weight *= sampling.high_gcc_peak_weight
        if coherence >= sampling.high_coherence_threshold:
            weight *= sampling.high_coherence_weight
        if ild_low_abs >= sampling.high_lowband_ild_abs_threshold:
            weight *= sampling.high_lowband_ild_weight
        if ild_high_abs >= sampling.high_highband_ild_abs_threshold:
            weight *= sampling.high_highband_ild_weight
        if energy_sum <= sampling.low_energy_sum_threshold:
            weight *= sampling.low_energy_weight
        weights.append(max(1e-6, min(weight, sampling.max_combined_weight)))
        if (index + 1) % 5000 == 0 or index + 1 == total:
            print(
                f"audio sampling weight scan: {index + 1}/{total} "
                f"({100.0 * (index + 1) / max(total, 1):.1f}%)"
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": weights}, cache_path)
    return weights


def _audio_only_batch_sampler(
    *,
    dataset: Any,
    config: TrainConfig,
    num_steps: int,
    start_step: int,
) -> Sampler[list[int]]:
    weights = _audio_only_sampling_weights(config, dataset)
    if weights is not None:
        return _WeightedStepBatchSampler(
            weights=weights,
            batch_size=config.schedule.batch_size,
            num_steps=num_steps - start_step,
            random_seed=config.seed,
            draw_offset=start_step * config.schedule.batch_size,
        )
    return _WrappedStepBatchSampler(
        length=len(dataset),
        batch_size=config.schedule.batch_size,
        num_steps=num_steps - start_step,
        base_index=config.single_step.step_index + start_step * config.schedule.batch_size,
    )


def _single_step_optimizer_parameters(
    *,
    vision: Any,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    base_lr = config.optimizer.lr
    group_scales = config.optimizer_group_scales
    parameter_groups: list[dict[str, Any]] = []
    trainable_parameters: list[torch.nn.Parameter] = []
    seen_parameter_ids: set[int] = set()

    def add_group(module: Any, *, lr_scale: float) -> None:
        if module is None or lr_scale <= 0.0:
            return
        group_parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in seen_parameter_ids
        ]
        if not group_parameters:
            return
        for parameter in group_parameters:
            seen_parameter_ids.add(id(parameter))
            trainable_parameters.append(parameter)
        parameter_groups.append(
            {
                "params": group_parameters,
                "lr": base_lr * lr_scale,
                "weight_decay": config.optimizer.weight_decay,
            }
        )

    add_group(getattr(vision, "backbone", None), lr_scale=group_scales.vision_backbone)
    add_group(
        getattr(vision, "segmentation_head", None),
        lr_scale=group_scales.vision_segmentation_head,
    )
    add_group(getattr(vision, "keypoint_head", None), lr_scale=group_scales.vision_keypoint_head)
    add_group(
        getattr(vision, "embedding_head", None),
        lr_scale=group_scales.vision_embedding_head,
    )
    add_group(audio, lr_scale=group_scales.audio)
    add_group(evidence, lr_scale=group_scales.evidence)

    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after applying freeze settings")
    return parameter_groups, trainable_parameters


def _dataloader_kwargs(config: TrainConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": config.schedule.num_workers,
        "pin_memory": config.schedule.pin_memory,
    }
    if config.schedule.num_workers > 0:
        kwargs["persistent_workers"] = config.schedule.persistent_workers
        if config.schedule.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.schedule.prefetch_factor
    return kwargs


def _single_view_segmentation_collate_fn(
    samples: list[Any],
    *,
    config: TrainConfig,
):
    return _collate_single_view_segmentation_batch(
        samples,
        segmentation_label_mode=config.segmentation_label_mode,
        segmentation_roi_crop_enabled=config.segmentation_roi_crop.enabled,
        segmentation_roi_crop_context_scale=config.segmentation_roi_crop.context_scale,
        segmentation_roi_crop_min_crop_size=config.segmentation_roi_crop.min_crop_size,
        segmentation_roi_crop_square=config.segmentation_roi_crop.square,
        segmentation_target_patch_enabled=config.segmentation_target_patch.enabled,
        segmentation_target_patch_size=config.segmentation_target_patch.patch_size,
    )


def _build_step_dataset(config: TrainConfig):
    if config.dataset_format == "packed":
        return StepDataset(config.dataset_root)
    if config.dataset_format == "synthetic_segmentation":
        if config.stage != "segmentation_only":
            raise ValueError(
                "dataset_format='synthetic_segmentation' is only supported for stage='segmentation_only'"
            )
        return SyntheticSingleViewSegmentationDataset(config.dataset_root)
    if config.dataset_format == "synthetic_single_step":
        if config.stage != "single_step":
            raise ValueError(
                "dataset_format='synthetic_single_step' is only supported for stage='single_step'"
            )
        return SyntheticSingleStepDataset(config.dataset_root)
    if config.dataset_format == "audio_only_recordings":
        if config.stage != "audio_only":
            raise ValueError(
                "dataset_format='audio_only_recordings' is only supported for stage='audio_only'"
            )
        return AudioOnlyRecordingDataset(config.dataset_root)
    raise ValueError(f"unsupported dataset_format: {config.dataset_format}")


def _vision_loss_weights_from_config(config: TrainConfig) -> VisionLossWeights:
    return VisionLossWeights(
        segmentation=config.vision_loss_weights.segmentation,
        keypoints=config.vision_loss_weights.keypoints,
        keypoint_voting=config.vision_loss_weights.keypoint_voting,
        segmentation_background=config.segmentation_class_weights.background,
        segmentation_other_aircraft=config.segmentation_class_weights.other_aircraft,
        segmentation_target=config.segmentation_class_weights.target,
        segmentation_loss_mode=config.segmentation_loss.mode,
        segmentation_focal_gamma=config.segmentation_loss.focal_gamma,
        segmentation_dice_weight=config.segmentation_loss.dice_weight,
    )


def _audio_loss_weights_from_config(config: TrainConfig) -> AudioLossWeights:
    return AudioLossWeights(
        doa=config.audio_loss_weights.doa,
        distance=config.audio_loss_weights.distance,
        doa_confidence=config.audio_loss_weights.doa_confidence,
        distance_confidence=config.audio_loss_weights.distance_confidence,
    )


def _audio_confidence_from_config(config: TrainConfig) -> AudioConfidenceConfig:
    return AudioConfidenceConfig(
        doa_half_angle=math.radians(config.audio_confidence.doa_half_angle_degrees),
        dist_half_error=config.audio_confidence.dist_half_error,
    )


def _build_vision_module_for_config(config: TrainConfig):
    num_segmentation_classes = segmentation_num_classes(config.segmentation_label_mode)
    if config.stage == "segmentation_only":
        if config.vision_backbone.name == "deeplabv3_resnet50":
            return DeepLabSingleViewSegmentationModule(
                DeepLabSingleViewSegmentationConfig(
                    num_segmentation_classes=num_segmentation_classes,
                    pretrained=config.vision_backbone.pretrained,
                )
            )
        return SingleViewSegmentationModule(
            SingleViewSegmentationConfig(
                backbone=VisionBackboneConfig(
                    name=config.vision_backbone.name,
                    pretrained=config.vision_backbone.pretrained,
                    in_channels=3,
                ),
                heads=VisionHeadConfig(num_segmentation_classes=num_segmentation_classes),
            )
        )
    if config.vision_backbone.name == "deeplabv3_resnet50":
        if config.stage != "segmentation_only":
            raise ValueError("deeplabv3_resnet50 is currently only supported for stage='segmentation_only'")
        return DeepLabSingleStepVisionModule(
            DeepLabSingleStepVisionConfig(
                num_segmentation_classes=num_segmentation_classes,
                pretrained=config.vision_backbone.pretrained,
            )
        )
    vision_config = SingleStepVisionConfig(
        backbone=VisionBackboneConfig(
            name=config.vision_backbone.name,
            pretrained=config.vision_backbone.pretrained,
            in_channels=3,
        ),
        heads=VisionHeadConfig(num_segmentation_classes=num_segmentation_classes),
        geometry=GeometryValidationConfig(
            pnp_top_k_points=config.vision_geometry.pnp_top_k_points,
            pnp_success_reprojection_threshold=config.vision_geometry.pnp_success_reprojection_threshold,
            pnp_success_min_selected_support_mean=config.vision_geometry.pnp_success_min_selected_support_mean,
            pnp_success_min_camera_depth=config.vision_geometry.pnp_success_min_camera_depth,
            pnp_success_max_camera_depth=config.vision_geometry.pnp_success_max_camera_depth,
            pnp_success_max_camera_translation_norm=config.vision_geometry.pnp_success_max_camera_translation_norm,
        ),
    )
    return SingleStepVisionModule(vision_config)


def _build_evidence_module_for_config(config: TrainConfig) -> SingleStepEvidenceModule:
    evidence_config = SingleStepEvidenceConfig(
        position_refine_scale=config.evidence.position_refine_scale
    )
    return SingleStepEvidenceModule(evidence_config)


def _collate_step_batch(
    samples: list[Any],
    *,
    segmentation_label_mode: str = "multiclass_absolute",
    keypoint_visible_weight: float = 1.0,
    keypoint_projectable_inframe_weight: float = 0.25,
    segmentation_roi_crop_enabled: bool = False,
    segmentation_roi_crop_context_scale: float = 2.0,
    segmentation_roi_crop_min_crop_size: int = 64,
    segmentation_roi_crop_square: bool = True,
    segmentation_target_patch_enabled: bool = False,
    segmentation_target_patch_size: int = 96,
    segmentation_target_patch_positive_views_only: bool = True,
):
    front_images: list[torch.Tensor] = []
    rear_images: list[torch.Tensor] = []
    front_segmentation_masks: list[torch.Tensor] = []
    rear_segmentation_masks: list[torch.Tensor] = []
    front_segmentation_valids: list[float] = []
    rear_segmentation_valids: list[float] = []
    audio_window = torch.tensor(
        [sample.core["audio_window_binaural"] for sample in samples],
        dtype=torch.float32,
    )
    binaural_energy_t = torch.tensor(
        [sample.audio_features["binaural_energy_t"] for sample in samples],
        dtype=torch.float32,
    )
    binaural_cue_vector_t = torch.tensor(
        [sample.audio_features["binaural_cue_vector_t"] for sample in samples],
        dtype=torch.float32,
    )
    target_class_ids = torch.tensor(
        [
            target_segmentation_class_id(
                sample.ref.observed_role,
                label_mode=segmentation_label_mode,
            )
            for sample in samples
        ],
        dtype=torch.long,
    )
    for sample, target_class_id in zip(samples, target_class_ids.tolist(), strict=True):
        front_image = _rgba_image_to_tensor(sample.core["front_camera_image"])
        rear_image = _rgba_image_to_tensor(sample.core["rear_camera_image"])
        front_segmentation = _remap_segmentation_mask_to_tensor(
            sample.vision_labels["segmentation_mask_front"],
            observed_role=sample.ref.observed_role,
            label_mode=segmentation_label_mode,
        )
        rear_segmentation = _remap_segmentation_mask_to_tensor(
            sample.vision_labels["segmentation_mask_rear"],
            observed_role=sample.ref.observed_role,
            label_mode=segmentation_label_mode,
        )
        front_has_target = bool((front_segmentation == target_class_id).any().item())
        rear_has_target = bool((rear_segmentation == target_class_id).any().item())
        front_valid = 1.0
        rear_valid = 1.0
        if segmentation_target_patch_enabled:
            if front_has_target:
                front_image, front_segmentation = _apply_target_centered_patch_crop(
                    front_image,
                    front_segmentation,
                    target_class_id=target_class_id,
                    patch_size=segmentation_target_patch_size,
                )
            if rear_has_target:
                rear_image, rear_segmentation = _apply_target_centered_patch_crop(
                    rear_image,
                    rear_segmentation,
                    target_class_id=target_class_id,
                    patch_size=segmentation_target_patch_size,
                )
            if segmentation_target_patch_positive_views_only and (front_has_target or rear_has_target):
                front_valid = 1.0 if front_has_target else 0.0
                rear_valid = 1.0 if rear_has_target else 0.0
        if segmentation_roi_crop_enabled:
            front_image, front_segmentation = _apply_segmentation_roi_crop(
                front_image,
                front_segmentation,
                target_class_id=target_class_id,
                context_scale=segmentation_roi_crop_context_scale,
                min_crop_size=segmentation_roi_crop_min_crop_size,
                square=segmentation_roi_crop_square,
            )
            rear_image, rear_segmentation = _apply_segmentation_roi_crop(
                rear_image,
                rear_segmentation,
                target_class_id=target_class_id,
                context_scale=segmentation_roi_crop_context_scale,
                min_crop_size=segmentation_roi_crop_min_crop_size,
                square=segmentation_roi_crop_square,
            )
        front_images.append(front_image)
        rear_images.append(rear_image)
        front_segmentation_masks.append(front_segmentation)
        rear_segmentation_masks.append(rear_segmentation)
        front_segmentation_valids.append(front_valid)
        rear_segmentation_valids.append(rear_valid)
    front = torch.stack(front_images, dim=0)
    rear = torch.stack(rear_images, dim=0)
    scale = torch.tensor(
        [front.shape[-1] - 1, front.shape[-2] - 1],
        dtype=torch.float32,
    )
    front_keypoints_xy_raw = torch.tensor(
        [sample.vision_labels["keypoints_2d_front"] for sample in samples],
        dtype=torch.float32,
    )
    rear_keypoints_xy_raw = torch.tensor(
        [sample.vision_labels["keypoints_2d_rear"] for sample in samples],
        dtype=torch.float32,
    )
    front_keypoint_visibility = torch.tensor(
        [sample.vision_labels["keypoint_visibility_front"] for sample in samples],
        dtype=torch.long,
    )
    rear_keypoint_visibility = torch.tensor(
        [sample.vision_labels["keypoint_visibility_rear"] for sample in samples],
        dtype=torch.long,
    )
    front_keypoint_projectable = torch.tensor(
        [sample.vision_labels["keypoint_projectable_front"] for sample in samples],
        dtype=torch.long,
    )
    rear_keypoint_projectable = torch.tensor(
        [sample.vision_labels["keypoint_projectable_rear"] for sample in samples],
        dtype=torch.long,
    )
    max_x = float(front.shape[-1] - 1)
    max_y = float(front.shape[-2] - 1)
    front_inframe = (
        (front_keypoints_xy_raw[..., 0] >= 0.0)
        & (front_keypoints_xy_raw[..., 0] <= max_x)
        & (front_keypoints_xy_raw[..., 1] >= 0.0)
        & (front_keypoints_xy_raw[..., 1] <= max_y)
    ).to(dtype=torch.float32)
    rear_inframe = (
        (rear_keypoints_xy_raw[..., 0] >= 0.0)
        & (rear_keypoints_xy_raw[..., 0] <= max_x)
        & (rear_keypoints_xy_raw[..., 1] >= 0.0)
        & (rear_keypoints_xy_raw[..., 1] <= max_y)
    ).to(dtype=torch.float32)
    front_projectable_inframe = (
        front_keypoint_projectable.to(dtype=torch.float32) * front_inframe
    )
    rear_projectable_inframe = (
        rear_keypoint_projectable.to(dtype=torch.float32) * rear_inframe
    )
    front_keypoint_xy_weights = (
        front_projectable_inframe * keypoint_projectable_inframe_weight
        + front_keypoint_visibility.to(dtype=torch.float32)
        * (keypoint_visible_weight - keypoint_projectable_inframe_weight)
    )
    rear_keypoint_xy_weights = (
        rear_projectable_inframe * keypoint_projectable_inframe_weight
        + rear_keypoint_visibility.to(dtype=torch.float32)
        * (keypoint_visible_weight - keypoint_projectable_inframe_weight)
    )
    vision_targets = VisionSupervisionTargets(
        target_class_ids=target_class_ids,
        front_segmentation=torch.stack(front_segmentation_masks, dim=0),
        rear_segmentation=torch.stack(rear_segmentation_masks, dim=0),
        front_segmentation_valid=torch.tensor(front_segmentation_valids, dtype=torch.float32),
        rear_segmentation_valid=torch.tensor(rear_segmentation_valids, dtype=torch.float32),
        front_keypoints_xy=front_keypoints_xy_raw / scale,
        rear_keypoints_xy=rear_keypoints_xy_raw / scale,
        front_keypoint_xy_mask=front_keypoint_visibility,
        rear_keypoint_xy_mask=rear_keypoint_visibility,
        front_keypoint_xy_weights=front_keypoint_xy_weights,
        rear_keypoint_xy_weights=rear_keypoint_xy_weights,
        front_keypoint_voting_pixels=_pad_nested_sequence_batch(
            [sample.vision_labels["keypoint_voting_pixels_front"] for sample in samples],
            dtype=torch.long,
            trailing_shape=(2,),
        ),
        rear_keypoint_voting_pixels=_pad_nested_sequence_batch(
            [sample.vision_labels["keypoint_voting_pixels_rear"] for sample in samples],
            dtype=torch.long,
            trailing_shape=(2,),
        ),
        front_keypoint_voting_unit_vectors=_pad_nested_sequence_batch(
            [sample.vision_labels["keypoint_voting_unit_vectors_front"] for sample in samples],
            dtype=torch.float32,
            trailing_shape=(front_keypoint_visibility.shape[-1], 2),
        ),
        rear_keypoint_voting_unit_vectors=_pad_nested_sequence_batch(
            [sample.vision_labels["keypoint_voting_unit_vectors_rear"] for sample in samples],
            dtype=torch.float32,
            trailing_shape=(rear_keypoint_visibility.shape[-1], 2),
        ),
        front_keypoint_voting_mask=_pad_flat_sequence_batch(
            [sample.vision_labels["keypoint_voting_mask_front"] for sample in samples],
            dtype=torch.float32,
        ),
        rear_keypoint_voting_mask=_pad_flat_sequence_batch(
            [sample.vision_labels["keypoint_voting_mask_rear"] for sample in samples],
            dtype=torch.float32,
        ),
    )
    evidence_targets = EvidenceSupervisionTargets(
        gt_relative_position=torch.tensor(
            [sample.core["gt_relative_position"] for sample in samples],
            dtype=torch.float32,
        ),
        gt_relative_orientation=torch.tensor(
            [sample.core["gt_relative_orientation"] for sample in samples],
            dtype=torch.float32,
        ),
        target_pos_conf=torch.tensor(
            [sample.rule_targets["target_pos_conf"] for sample in samples],
            dtype=torch.float32,
        ),
        target_ori_conf=torch.tensor(
            [sample.rule_targets["target_ori_conf"] for sample in samples],
            dtype=torch.float32,
        ),
        view_valid_target=torch.tensor(
            [
                float(
                    any(
                        int(value) == int(target_segmentation_class_id(sample.ref.observed_role))
                        for row in sample.vision_labels["segmentation_mask_front"]
                        for value in row
                    )
                    or any(
                        int(value) == int(target_segmentation_class_id(sample.ref.observed_role))
                        for row in sample.vision_labels["segmentation_mask_rear"]
                        for value in row
                    )
                )
                for sample in samples
            ],
            dtype=torch.float32,
        ),
        pos_valid_target=torch.tensor(
            [float(sample.rule_targets["target_pos_conf"] > 0.0) for sample in samples],
            dtype=torch.float32,
        ),
        ori_valid_target=torch.tensor(
            [float(sample.rule_targets["target_ori_conf"] > 0.0) for sample in samples],
            dtype=torch.float32,
        ),
    )
    audio_targets = AudioSupervisionTargets(
        gt_doa_unit_vector_body=torch.tensor(
            [sample.rule_targets["gt_doa_unit_vector_body"] for sample in samples],
            dtype=torch.float32,
        ),
        gt_log_distance_scalar=torch.tensor(
            [sample.rule_targets["gt_log_distance_scalar"] for sample in samples],
            dtype=torch.float32,
        ),
    )
    return {
        "front": front,
        "rear": rear,
        "audio_window": audio_window,
        "binaural_energy_t": binaural_energy_t,
        "binaural_cue_vector_t": binaural_cue_vector_t,
        "target_class_ids": target_class_ids,
        "vision_targets": vision_targets,
        "evidence_targets": evidence_targets,
        "audio_targets": audio_targets,
    }


def _single_step_collate_fn(
    samples: list[Any],
    *,
    config: TrainConfig,
):
    return _collate_step_batch(
        samples,
        segmentation_label_mode=config.segmentation_label_mode,
        keypoint_visible_weight=config.keypoint_supervision.visible_weight,
        keypoint_projectable_inframe_weight=config.keypoint_supervision.projectable_inframe_weight,
        segmentation_roi_crop_enabled=config.segmentation_roi_crop.enabled,
        segmentation_roi_crop_context_scale=config.segmentation_roi_crop.context_scale,
        segmentation_roi_crop_min_crop_size=config.segmentation_roi_crop.min_crop_size,
        segmentation_roi_crop_square=config.segmentation_roi_crop.square,
        segmentation_target_patch_enabled=config.segmentation_target_patch.enabled,
        segmentation_target_patch_size=config.segmentation_target_patch.patch_size,
        segmentation_target_patch_positive_views_only=config.segmentation_target_patch.positive_views_only,
    )


def _collate_audio_only_batch(samples: list[Any]) -> dict[str, Any]:
    return {
        "audio_window": torch.tensor(
            [sample.audio_features["audio_window_binaural"] for sample in samples],
            dtype=torch.float32,
        ),
        "binaural_energy_t": torch.tensor(
            [sample.audio_features["binaural_energy_t"] for sample in samples],
            dtype=torch.float32,
        ),
        "binaural_cue_vector_t": torch.tensor(
            [sample.audio_features["binaural_cue_vector_t"] for sample in samples],
            dtype=torch.float32,
        ),
        "audio_targets": AudioSupervisionTargets(
            gt_doa_unit_vector_body=torch.tensor(
                [sample.rule_targets["gt_doa_unit_vector_body"] for sample in samples],
                dtype=torch.float32,
            ),
            gt_log_distance_scalar=torch.tensor(
                [sample.rule_targets["gt_log_distance_scalar"] for sample in samples],
                dtype=torch.float32,
            ),
        ),
    }


def _audio_only_collate_fn(
    samples: list[Any],
    *,
    config: TrainConfig,
):
    del config
    return _collate_audio_only_batch(samples)


def _selected_view_from_segmentation_masks(
    front_segmentation: torch.Tensor,
    rear_segmentation: torch.Tensor,
    target_class_ids: torch.Tensor,
) -> torch.Tensor:
    if front_segmentation.shape != rear_segmentation.shape:
        raise ValueError("front/rear segmentation must share shape")
    if front_segmentation.ndim != 3:
        raise ValueError("segmentation tensors must be [B, H, W]")
    batch_size = front_segmentation.shape[0]
    if target_class_ids.shape != (batch_size,):
        raise ValueError("target_class_ids must be [B]")
    target = target_class_ids.view(-1, 1, 1).to(device=front_segmentation.device, dtype=front_segmentation.dtype)
    front_area = (front_segmentation == target).sum(dim=(1, 2))
    rear_area = (rear_segmentation == target).sum(dim=(1, 2))
    selected_view_index = torch.full_like(front_area, 2, dtype=torch.long)
    selected_view_index = torch.where(
        (front_area > rear_area) & (front_area > 0),
        torch.zeros_like(selected_view_index),
        selected_view_index,
    )
    selected_view_index = torch.where(
        (rear_area >= front_area) & (rear_area > 0),
        torch.ones_like(selected_view_index),
        selected_view_index,
    )
    return selected_view_index


def _build_temporal_inputs_from_window_batch(
    samples: list[Any],
    *,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
    device: torch.device,
) -> TemporalModalityInputs:
    front = torch.stack(
        [
            torch.stack(
                [_rgba_image_to_tensor(image) for image in sample.core["front_camera_image"]],
                dim=0,
            )
            for sample in samples
        ],
        dim=0,
    ).to(device)
    rear = torch.stack(
        [
            torch.stack(
                [_rgba_image_to_tensor(image) for image in sample.core["rear_camera_image"]],
                dim=0,
            )
            for sample in samples
        ],
        dim=0,
    ).to(device)
    audio_window = torch.tensor(
        [sample.core["audio_window_binaural"] for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    binaural_energy_t = torch.tensor(
        [sample.audio_features["binaural_energy_t"] for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    binaural_cue_vector_t = torch.tensor(
        [sample.audio_features["binaural_cue_vector_t"] for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    delta_binaural_cue_t = compute_delta_binaural_cue_t(binaural_cue_vector_t)

    batch_size, steps, channels, height, width = front.shape
    flat_front = front.reshape(batch_size * steps, channels, height, width)
    flat_rear = rear.reshape(batch_size * steps, channels, height, width)
    flat_audio_window = audio_window.reshape(batch_size * steps, *audio_window.shape[2:])
    flat_binaural_energy = binaural_energy_t.reshape(batch_size * steps, binaural_energy_t.shape[-1])
    flat_binaural_cue = binaural_cue_vector_t.reshape(batch_size * steps, binaural_cue_vector_t.shape[-1])
    flat_target_class_ids = torch.tensor(
        [
            target_segmentation_class_id(sample.ref.observed_role)
            for sample in samples
            for _ in range(steps)
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        vision_output = vision(flat_front, flat_rear, flat_target_class_ids)
        vision_output = reroute_vision_output_with_inertia(
            vision_output,
            batch_size=batch_size,
            steps=steps,
            device=device,
        )
        audio_output = audio(
            flat_audio_window,
            flat_binaural_energy,
            flat_binaural_cue,
        )
        evidence_output = evidence(vision_output, audio_output)

    selected_target_probability_t = select_view_target_probability(
        vision_output.front_segmentation_logits,
        vision_output.rear_segmentation_logits,
        flat_target_class_ids,
        vision_output.selected_view_index,
    ).reshape(batch_size, steps, height, width)
    selected_view_index_t = vision_output.selected_view_index.reshape(batch_size, steps)
    selected_segmentation_difference_t, selected_segmentation_diff_valid_t = (
        compute_selected_segmentation_difference_t(
            selected_target_probability_t,
            selected_view_index_t,
        )
    )
    selected_keypoints_xy_t = select_view_tensor(
        vision_output.front_keypoints_xy,
        vision_output.rear_keypoints_xy,
        vision_output.selected_view_index,
    ).reshape(batch_size, steps, vision_output.front_keypoints_xy.shape[1], 2)
    selected_keypoint_support_t = select_view_tensor(
        vision_output.front_keypoint_support,
        vision_output.rear_keypoint_support,
        vision_output.selected_view_index,
    ).reshape(batch_size, steps, vision_output.front_keypoint_support.shape[1])
    (
        selected_keypoint_delta_t,
        selected_keypoint_delta_support_summary_t,
        selected_keypoint_delta_valid_t,
    ) = compute_selected_keypoint_delta_t(
        selected_keypoints_xy_t,
        selected_keypoint_support_t,
        selected_view_index_t,
    )

    return TemporalModalityInputs(
        relative_position=evidence_output.evidence_state.relative_position.reshape(batch_size, steps, 3),
        relative_orientation=evidence_output.evidence_state.relative_orientation.reshape(batch_size, steps, 6),
        position_confidence=evidence_output.evidence_state.position_confidence.reshape(batch_size, steps),
        orientation_confidence=evidence_output.evidence_state.orientation_confidence.reshape(batch_size, steps),
        pos_valid=evidence_output.evidence_state.pos_valid.reshape(batch_size, steps),
        ori_valid=evidence_output.evidence_state.ori_valid.reshape(batch_size, steps),
        visual_embedding=evidence_output.evidence.visual_embedding.reshape(batch_size, steps, -1),
        audio_embedding=evidence_output.evidence.audio_embedding.reshape(batch_size, steps, -1),
        raw_visual_evidence_strength=evidence_output.evidence.raw_visual_evidence_strength.reshape(batch_size, steps),
        view_valid=vision_output.selected_candidate.view_valid.reshape(batch_size, steps),
        selected_segmentation_difference_t=selected_segmentation_difference_t,
        selected_segmentation_diff_valid_t=selected_segmentation_diff_valid_t,
        selected_keypoint_delta_t=selected_keypoint_delta_t,
        selected_keypoint_delta_support_summary_t=selected_keypoint_delta_support_summary_t,
        selected_keypoint_delta_valid_t=selected_keypoint_delta_valid_t,
        raw_audio_evidence_strength=evidence_output.evidence.raw_audio_evidence_strength.reshape(batch_size, steps),
        binaural_energy_t=binaural_energy_t,
        binaural_cue_vector_t=binaural_cue_vector_t,
        delta_binaural_cue_t=delta_binaural_cue_t,
        dt_to_prev=torch.tensor([sample.dt_to_prev for sample in samples], dtype=torch.float32, device=device),
        time_from_now=torch.tensor([sample.time_from_now for sample in samples], dtype=torch.float32, device=device),
    )


def _collate_single_view_segmentation_batch(
    samples: list[Any],
    *,
    segmentation_label_mode: str = "multiclass_absolute",
    segmentation_roi_crop_enabled: bool = False,
    segmentation_roi_crop_context_scale: float = 2.0,
    segmentation_roi_crop_min_crop_size: int = 64,
    segmentation_roi_crop_square: bool = True,
    segmentation_target_patch_enabled: bool = False,
    segmentation_target_patch_size: int = 96,
):
    images: list[torch.Tensor] = []
    segmentation_masks: list[torch.Tensor] = []
    view_indices: list[int] = []
    gt_target_areas: list[int] = []
    target_class_ids = torch.tensor(
        [
            target_segmentation_class_id(
                sample.observed_role,
                label_mode=segmentation_label_mode,
            )
            for sample in samples
        ],
        dtype=torch.long,
    )
    for sample, target_class_id in zip(samples, target_class_ids.tolist(), strict=True):
        image = _rgba_image_to_tensor(sample.image)
        segmentation = _remap_segmentation_mask_to_tensor(
            sample.segmentation_mask,
            observed_role=sample.observed_role,
            label_mode=segmentation_label_mode,
        )
        if segmentation_target_patch_enabled:
            image, segmentation = _apply_target_centered_patch_crop(
                image,
                segmentation,
                target_class_id=target_class_id,
                patch_size=segmentation_target_patch_size,
            )
        if segmentation_roi_crop_enabled:
            image, segmentation = _apply_segmentation_roi_crop(
                image,
                segmentation,
                target_class_id=target_class_id,
                context_scale=segmentation_roi_crop_context_scale,
                min_crop_size=segmentation_roi_crop_min_crop_size,
                square=segmentation_roi_crop_square,
            )
        images.append(image)
        segmentation_masks.append(segmentation)
        view_indices.append(0 if sample.active_view == "front" else 1)
        gt_target_areas.append(int((segmentation == target_class_id).sum().item()))
    return {
        "image": torch.stack(images, dim=0),
        "target_class_ids": target_class_ids,
        "view_index": torch.tensor(view_indices, dtype=torch.long),
        "gt_target_area": torch.tensor(gt_target_areas, dtype=torch.long),
        "segmentation_targets": SingleViewSegmentationTargets(
            target_class_ids=target_class_ids,
            segmentation=torch.stack(segmentation_masks, dim=0),
        ),
    }


def _move_single_view_segmentation_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    image = batch["image"].to(device, non_blocking=True)
    target_class_ids = batch["target_class_ids"].to(device, non_blocking=True)
    view_index = batch["view_index"].to(device, non_blocking=True)
    gt_target_area = batch["gt_target_area"].to(device, non_blocking=True)
    segmentation = batch["segmentation_targets"].segmentation.to(device, non_blocking=True)
    return {
        "image": image,
        "target_class_ids": target_class_ids,
        "view_index": view_index,
        "gt_target_area": gt_target_area,
        "segmentation_targets": SingleViewSegmentationTargets(
            target_class_ids=target_class_ids,
            segmentation=segmentation,
        ),
    }


def _move_single_step_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    front = batch["front"].to(device, non_blocking=True)
    rear = batch["rear"].to(device, non_blocking=True)
    audio_window = batch["audio_window"].to(device, non_blocking=True)
    binaural_energy_t = batch["binaural_energy_t"].to(device, non_blocking=True)
    binaural_cue_vector_t = batch["binaural_cue_vector_t"].to(device, non_blocking=True)
    target_class_ids = batch["target_class_ids"].to(device, non_blocking=True)

    vision_targets_cpu = batch["vision_targets"]
    vision_targets = VisionSupervisionTargets(
        target_class_ids=target_class_ids,
        front_segmentation=vision_targets_cpu.front_segmentation.to(device, non_blocking=True),
        rear_segmentation=vision_targets_cpu.rear_segmentation.to(device, non_blocking=True),
        front_segmentation_valid=vision_targets_cpu.front_segmentation_valid.to(device, non_blocking=True),
        rear_segmentation_valid=vision_targets_cpu.rear_segmentation_valid.to(device, non_blocking=True),
        front_keypoints_xy=vision_targets_cpu.front_keypoints_xy.to(device, non_blocking=True),
        rear_keypoints_xy=vision_targets_cpu.rear_keypoints_xy.to(device, non_blocking=True),
        front_keypoint_xy_mask=vision_targets_cpu.front_keypoint_xy_mask.to(device, non_blocking=True),
        rear_keypoint_xy_mask=vision_targets_cpu.rear_keypoint_xy_mask.to(device, non_blocking=True),
        front_keypoint_xy_weights=(
            vision_targets_cpu.front_keypoint_xy_weights.to(device, non_blocking=True)
            if vision_targets_cpu.front_keypoint_xy_weights is not None
            else None
        ),
        rear_keypoint_xy_weights=(
            vision_targets_cpu.rear_keypoint_xy_weights.to(device, non_blocking=True)
            if vision_targets_cpu.rear_keypoint_xy_weights is not None
            else None
        ),
        front_keypoint_voting_pixels=vision_targets_cpu.front_keypoint_voting_pixels.to(device, non_blocking=True),
        rear_keypoint_voting_pixels=vision_targets_cpu.rear_keypoint_voting_pixels.to(device, non_blocking=True),
        front_keypoint_voting_unit_vectors=vision_targets_cpu.front_keypoint_voting_unit_vectors.to(device, non_blocking=True),
        rear_keypoint_voting_unit_vectors=vision_targets_cpu.rear_keypoint_voting_unit_vectors.to(device, non_blocking=True),
        front_keypoint_voting_mask=vision_targets_cpu.front_keypoint_voting_mask.to(device, non_blocking=True),
        rear_keypoint_voting_mask=vision_targets_cpu.rear_keypoint_voting_mask.to(device, non_blocking=True),
    )

    evidence_targets_cpu = batch["evidence_targets"]
    evidence_targets = EvidenceSupervisionTargets(
        gt_relative_position=evidence_targets_cpu.gt_relative_position.to(device, non_blocking=True),
        gt_relative_orientation=evidence_targets_cpu.gt_relative_orientation.to(device, non_blocking=True),
        target_pos_conf=evidence_targets_cpu.target_pos_conf.to(device, non_blocking=True),
        target_ori_conf=evidence_targets_cpu.target_ori_conf.to(device, non_blocking=True),
        view_valid_target=evidence_targets_cpu.view_valid_target.to(device, non_blocking=True),
        pos_valid_target=evidence_targets_cpu.pos_valid_target.to(device, non_blocking=True),
        ori_valid_target=evidence_targets_cpu.ori_valid_target.to(device, non_blocking=True),
    )

    audio_targets_cpu = batch["audio_targets"]
    audio_targets = AudioSupervisionTargets(
        gt_doa_unit_vector_body=audio_targets_cpu.gt_doa_unit_vector_body.to(device, non_blocking=True),
        gt_log_distance_scalar=audio_targets_cpu.gt_log_distance_scalar.to(device, non_blocking=True),
    )

    return {
        "front": front,
        "rear": rear,
        "audio_window": audio_window,
        "binaural_energy_t": binaural_energy_t,
        "binaural_cue_vector_t": binaural_cue_vector_t,
        "target_class_ids": target_class_ids,
        "vision_targets": vision_targets,
        "evidence_targets": evidence_targets,
        "audio_targets": audio_targets,
    }


def _move_audio_only_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    audio_targets_cpu = batch["audio_targets"]
    return {
        "audio_window": batch["audio_window"].to(device, non_blocking=True),
        "binaural_energy_t": batch["binaural_energy_t"].to(device, non_blocking=True),
        "binaural_cue_vector_t": batch["binaural_cue_vector_t"].to(device, non_blocking=True),
        "audio_targets": AudioSupervisionTargets(
            gt_doa_unit_vector_body=audio_targets_cpu.gt_doa_unit_vector_body.to(
                device, non_blocking=True
            ),
            gt_log_distance_scalar=audio_targets_cpu.gt_log_distance_scalar.to(
                device, non_blocking=True
            ),
        ),
    }


def _selected_view_from_single_view_areas(
    *,
    view_index: torch.Tensor,
    target_area: torch.Tensor,
) -> torch.Tensor:
    if view_index.shape != target_area.shape:
        raise ValueError("view_index and target_area must share shape")
    return torch.where(
        target_area > 0,
        view_index.long(),
        torch.full_like(view_index, 2, dtype=torch.long),
    )


def _collate_temporal_targets(samples: list[Any], device: torch.device) -> TemporalSupervisionTargets:
    return TemporalSupervisionTargets(
        gt_relative_position=torch.tensor(
            [sample.core["gt_relative_position"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        gt_relative_orientation=torch.tensor(
            [sample.core["gt_relative_orientation"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        target_pos_conf=torch.tensor(
            [sample.rule_targets["target_pos_conf"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        target_ori_conf=torch.tensor(
            [sample.rule_targets["target_ori_conf"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    )


def _collate_belief_targets(samples: list[Any], device: torch.device) -> BeliefSupervisionTargets:
    return BeliefSupervisionTargets(
        gt_relative_position=torch.tensor(
            [sample.core["gt_relative_position"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        gt_relative_orientation=torch.tensor(
            [sample.core["gt_relative_orientation"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        gt_linear_velocity=torch.tensor(
            [sample.core["gt_linear_velocity"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        gt_angular_velocity=torch.tensor(
            [sample.core["gt_angular_velocity"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        target_pos_conf=torch.tensor(
            [sample.rule_targets["target_pos_conf"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        target_ori_conf=torch.tensor(
            [sample.rule_targets["target_ori_conf"][-1] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    )


def _build_belief_inputs(
    samples: list[Any],
    stage1_output,
    *,
    device: torch.device,
) -> BeliefUpdateInputs:
    return BeliefUpdateInputs(
        coarse_state_t=stage1_output.coarse_state,
        context_relative_position=torch.tensor(
            [sample.core["gt_relative_position"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        context_relative_orientation=torch.tensor(
            [sample.core["gt_relative_orientation"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        context_position_confidence=torch.tensor(
            [sample.rule_targets["target_pos_conf"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        context_orientation_confidence=torch.tensor(
            [sample.rule_targets["target_ori_conf"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        visual_evidence_strength_t=stage1_output.visual_evidence_strength,
        audio_evidence_strength_t=stage1_output.audio_evidence_strength,
        linear_velocity=torch.tensor(
            [sample.core["gt_linear_velocity"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        angular_velocity=torch.tensor(
            [sample.core["gt_angular_velocity"] for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        dt_to_prev=torch.tensor([sample.dt_to_prev for sample in samples], dtype=torch.float32, device=device),
        time_from_now=torch.tensor([sample.time_from_now for sample in samples], dtype=torch.float32, device=device),
    )


def _resolved_config_dict(
    config: TrainConfig,
    *,
    stage: str,
    num_steps: int,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    data = asdict(config)
    data["dataset_root"] = str(config.dataset_root)
    data["output_root"] = str(output_root)
    data["init_vision_from"] = (
        str(config.init_vision_from) if config.init_vision_from is not None else None
    )
    data["init_audio_from"] = (
        str(config.init_audio_from) if config.init_audio_from is not None else None
    )
    data["init_segmentation_from"] = (
        str(config.init_segmentation_from)
        if config.init_segmentation_from is not None
        else None
    )
    data["stage"] = stage
    data["device"] = device
    data["schedule"]["num_steps"] = num_steps
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_checkpoint(
    path: Path,
    *,
    stage: str,
    step: int,
    optimizer: torch.optim.Optimizer,
    modules: dict[str, torch.nn.Module],
    output_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "step": step,
        "global_step": step + 1,
        "output_root": str(output_root),
        "optimizer": optimizer.state_dict(),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "torch_rng_state": torch.get_rng_state(),
        "saved_at_utc": _utc_now_iso(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _initialize_vision_from_checkpoint(
    *,
    vision: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, list[str]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = payload.get("modules") if isinstance(payload, dict) else None
    if isinstance(module_states, dict) and "vision" in module_states:
        state_dict = module_states["vision"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"unsupported checkpoint payload for init_vision_from: {checkpoint_path}")
    incompatible = vision.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _initialize_audio_from_checkpoint(
    *,
    audio: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, list[str]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = payload.get("modules") if isinstance(payload, dict) else None
    if isinstance(module_states, dict) and "audio" in module_states:
        state_dict = module_states["audio"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"unsupported checkpoint payload for init_audio_from: {checkpoint_path}")
    current_state = audio.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    incompatible = audio.load_state_dict(filtered_state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _initialize_segmentation_submodules_from_checkpoint(
    *,
    vision: SingleStepVisionModule,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, list[str]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = payload.get("modules") if isinstance(payload, dict) else None
    if isinstance(module_states, dict) and "vision" in module_states:
        state_dict = module_states["vision"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(
            f"unsupported checkpoint payload for init_segmentation_from: {checkpoint_path}"
        )
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("backbone.") or key.startswith("segmentation_head.")
    }
    incompatible = vision.load_state_dict(filtered_state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _freeze_single_step_segmentation_module(vision: SingleStepVisionModule) -> None:
    for module in (vision.backbone, vision.segmentation_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def _load_checkpoint(
    path: Path,
    *,
    expected_stage: str,
    optimizer: torch.optim.Optimizer,
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    allow_partial_module_load: set[str] | None = None,
) -> dict[str, Any]:
    allow_partial_module_load = allow_partial_module_load or set()
    payload = torch.load(path, map_location=device, weights_only=False)
    checkpoint_stage = str(payload["stage"])
    if checkpoint_stage != expected_stage:
        raise ValueError(
            f"checkpoint stage mismatch: expected {expected_stage}, got {checkpoint_stage}"
        )
    module_states = payload["modules"]
    used_partial_load = False
    for name, module in modules.items():
        if name not in module_states:
            raise ValueError(f"missing module state in checkpoint: {name}")
        state_dict = module_states[name]
        try:
            module.load_state_dict(state_dict)
        except RuntimeError:
            if name not in allow_partial_module_load:
                raise
            current_state = module.state_dict()
            filtered_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            incompatible = module.load_state_dict(filtered_state, strict=False)
            skipped_count = len(state_dict) - len(filtered_state)
            used_partial_load = True
            print(
                f"warning: partially loaded checkpoint module {name} from {path} "
                f"(loaded={len(filtered_state)}, skipped={skipped_count}, "
                f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
            )
    if not used_partial_load:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except (RuntimeError, ValueError) as exc:
            print(
                f"warning: skipped optimizer state restore from {path} due to optimizer mismatch: {exc}"
            )
    else:
        print(
            f"warning: skipped optimizer state restore from {path} after partial module load"
        )
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and "torch_cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda_rng_state_all"]])
    return payload


def _maybe_save_checkpoint(
    *,
    step: int,
    num_steps: int,
    save_interval: int,
    checkpoint_dir: Path,
    stage: str,
    optimizer: torch.optim.Optimizer,
    modules: dict[str, torch.nn.Module],
    output_root: Path,
) -> None:
    current_step = step + 1
    should_save = current_step % save_interval == 0 or current_step == num_steps
    if not should_save:
        return
    numbered_path = checkpoint_dir / f"step_{current_step:06d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    _save_checkpoint(
        numbered_path,
        stage=stage,
        step=step,
        optimizer=optimizer,
        modules=modules,
        output_root=output_root,
    )
    _save_checkpoint(
        latest_path,
        stage=stage,
        step=step,
        optimizer=optimizer,
        modules=modules,
        output_root=output_root,
    )


def _maybe_write_eval_snapshot(
    *,
    step: int,
    num_steps: int,
    eval_interval: int,
    eval_dir: Path,
    payload: dict[str, Any],
) -> None:
    current_step = step + 1
    should_eval = current_step % eval_interval == 0 or current_step == num_steps
    if not should_eval:
        return
    _write_json(eval_dir / f"eval_step_{current_step:06d}.json", payload)


def _ensure_final_checkpoint(
    *,
    completed_final_step: int | None,
    num_steps: int,
    checkpoint_dir: Path,
    stage: str,
    optimizer: torch.optim.Optimizer,
    modules: dict[str, torch.nn.Module],
    output_root: Path,
) -> None:
    if completed_final_step is None:
        return
    if completed_final_step + 1 != num_steps:
        return
    _save_checkpoint(
        checkpoint_dir / f"step_{num_steps:06d}.pt",
        stage=stage,
        step=completed_final_step,
        optimizer=optimizer,
        modules=modules,
        output_root=output_root,
    )
    _save_checkpoint(
        checkpoint_dir / "latest.pt",
        stage=stage,
        step=completed_final_step,
        optimizer=optimizer,
        modules=modules,
        output_root=output_root,
    )


def _capture_single_step_eval_snapshot(
    *,
    dataset: Any,
    config: TrainConfig,
    device: torch.device,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
) -> dict[str, Any]:
    modules = (vision, audio, evidence)
    previous_modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            sample = dataset[config.single_step.step_index % len(dataset)]
            batch = _collate_step_batch(
                [sample],
                segmentation_label_mode=config.segmentation_label_mode,
                segmentation_roi_crop_enabled=config.segmentation_roi_crop.enabled,
                segmentation_roi_crop_context_scale=config.segmentation_roi_crop.context_scale,
                segmentation_roi_crop_min_crop_size=config.segmentation_roi_crop.min_crop_size,
                segmentation_roi_crop_square=config.segmentation_roi_crop.square,
                segmentation_target_patch_enabled=config.segmentation_target_patch.enabled,
                segmentation_target_patch_size=config.segmentation_target_patch.patch_size,
                segmentation_target_patch_positive_views_only=config.segmentation_target_patch.positive_views_only,
            )
            batch = _move_single_step_batch_to_device(batch, device)
            vision_output = vision(batch["front"], batch["rear"], batch["target_class_ids"])
            audio_output = audio(
                batch["audio_window"],
                batch["binaural_energy_t"],
                batch["binaural_cue_vector_t"],
            )
            evidence_output = evidence(vision_output, audio_output)
            vision_losses = compute_single_step_vision_loss(
                vision_output,
                batch["vision_targets"],
                weights=_vision_loss_weights_from_config(config),
            )
            audio_losses = compute_single_step_audio_loss(
                audio_output,
                batch["audio_targets"],
                weights=_audio_loss_weights_from_config(config),
                confidence=_audio_confidence_from_config(config),
            )
            evidence_losses = compute_single_step_evidence_loss(
                evidence_output, batch["evidence_targets"]
            )
            evidence_position_l1 = _masked_l1(
                evidence_output.evidence_state.relative_position,
                batch["evidence_targets"].gt_relative_position,
                batch["evidence_targets"].pos_valid_target,
            )
            evidence_orientation_l1 = _masked_l1(
                evidence_output.evidence_state.relative_orientation,
                batch["evidence_targets"].gt_relative_orientation,
                batch["evidence_targets"].ori_valid_target,
            )
            evidence_pos_conf_mse = _masked_mse(
                evidence_output.evidence_state.position_confidence,
                batch["evidence_targets"].target_pos_conf,
                batch["evidence_targets"].pos_valid_target,
            )
            evidence_ori_conf_mse = _masked_mse(
                evidence_output.evidence_state.orientation_confidence,
                batch["evidence_targets"].target_ori_conf,
                batch["evidence_targets"].ori_valid_target,
            )
            audio_position_l1 = F.l1_loss(
                evidence_output.evidence.audio_relative_position,
                batch["evidence_targets"].gt_relative_position,
            )
            evidence_position_error_norm = torch.linalg.vector_norm(
                evidence_output.evidence_state.relative_position
                - batch["evidence_targets"].gt_relative_position,
                dim=-1,
            )
            evidence_orientation_geodesic_deg = _orientation_geodesic_degrees(
                evidence_output.evidence_state.relative_orientation,
                batch["evidence_targets"].gt_relative_orientation,
            )
            return {
                "stage": "single_step",
                "dataset_root": str(config.dataset_root),
                "num_samples": 1,
                "sample_pose_debug": {
                    "pred_relative_position": evidence_output.evidence_state.relative_position[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "gt_relative_position": batch["evidence_targets"].gt_relative_position[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "pred_relative_orientation_6d": evidence_output.evidence_state.relative_orientation[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "gt_relative_orientation_6d": batch["evidence_targets"].gt_relative_orientation[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "pred_view_valid_probability": float(
                        evidence_output.evidence.view_valid_probability[0].detach().cpu().item()
                    ),
                    "pred_pos_valid_probability": float(
                        evidence_output.evidence_state.pos_valid_probability[0].detach().cpu().item()
                    ),
                    "pred_ori_valid_probability": float(
                        evidence_output.evidence_state.ori_valid_probability[0].detach().cpu().item()
                    ),
                    "target_view_valid": float(
                        batch["evidence_targets"].view_valid_target[0].detach().cpu().item()
                    ),
                    "target_pos_valid": float(
                        batch["evidence_targets"].pos_valid_target[0].detach().cpu().item()
                    ),
                    "target_ori_valid": float(
                        batch["evidence_targets"].ori_valid_target[0].detach().cpu().item()
                    ),
                },
                "metrics": {
                    "vision_total_loss": float(vision_losses["total"].detach().cpu().item()),
                    "vision_segmentation_loss": float(
                        vision_losses["segmentation"].detach().cpu().item()
                    ),
                    "vision_keypoint_loss": float(
                        vision_losses["keypoints"].detach().cpu().item()
                    ),
                    "vision_keypoint_voting_loss": float(
                        vision_losses["keypoint_voting"].detach().cpu().item()
                    ),
                    "audio_total_loss": float(audio_losses["total"].detach().cpu().item()),
                    "evidence_total_loss": float(evidence_losses["total"].detach().cpu().item()),
                    "evidence_position_loss": float(
                        evidence_losses["position"].detach().cpu().item()
                    ),
                    "evidence_orientation_loss": float(
                        evidence_losses["orientation"].detach().cpu().item()
                    ),
                    "evidence_position_confidence_loss": float(
                        evidence_losses["position_confidence"].detach().cpu().item()
                    ),
                    "evidence_orientation_confidence_loss": float(
                        evidence_losses["orientation_confidence"].detach().cpu().item()
                    ),
                    "evidence_view_valid_loss": float(
                        evidence_losses["view_valid"].detach().cpu().item()
                    ),
                    "evidence_pos_valid_loss": float(
                        evidence_losses["pos_valid"].detach().cpu().item()
                    ),
                    "evidence_ori_valid_loss": float(
                        evidence_losses["ori_valid"].detach().cpu().item()
                    ),
                    "weighted_vision_loss": float(
                        (
                            config.single_step_loss_weights.vision
                            * vision_losses["total"]
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "weighted_audio_loss": float(
                        (
                            config.single_step_loss_weights.audio
                            * audio_losses["total"]
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "weighted_evidence_loss": float(
                        (
                            config.single_step_loss_weights.evidence
                            * evidence_losses["total"]
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "total_loss": float(
                        _single_step_total_loss(
                            config,
                            vision_total=vision_losses["total"],
                            audio_total=audio_losses["total"],
                            evidence_total=evidence_losses["total"],
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "audio_doa_loss": float(audio_losses["doa"].detach().cpu().item()),
                    "audio_distance_loss": float(audio_losses["distance"].detach().cpu().item()),
                    "audio_doa_conf_loss": float(
                        audio_losses["doa_confidence"].detach().cpu().item()
                    ),
                    "audio_dist_conf_loss": float(
                        audio_losses["distance_confidence"].detach().cpu().item()
                    ),
                    "audio_doa_angle_error": float(
                        audio_losses["doa_angle_error"].detach().cpu().item()
                    ),
                    "audio_log_distance_error": float(
                        audio_losses["log_distance_error"].detach().cpu().item()
                    ),
                    "audio_doa_conf_mse": float(
                        audio_losses["doa_confidence"].detach().cpu().item()
                    ),
                    "audio_dist_conf_mse": float(
                        audio_losses["distance_confidence"].detach().cpu().item()
                    ),
                    "audio_doa_conf_target_mean": float(
                        audio_losses["target_doa_conf_mean"].detach().cpu().item()
                    ),
                    "audio_dist_conf_target_mean": float(
                        audio_losses["target_dist_conf_mean"].detach().cpu().item()
                    ),
                    "evidence_position_l1": float(evidence_position_l1.detach().cpu().item()),
                    "evidence_position_error_norm": float(
                        evidence_position_error_norm.mean().detach().cpu().item()
                    ),
                    "evidence_orientation_l1": float(
                        evidence_orientation_l1.detach().cpu().item()
                    ),
                    "evidence_orientation_geodesic_deg": float(
                        evidence_orientation_geodesic_deg.mean().detach().cpu().item()
                    ),
                    "evidence_pos_conf_mse": float(
                        evidence_pos_conf_mse.detach().cpu().item()
                    ),
                    "evidence_ori_conf_mse": float(
                        evidence_ori_conf_mse.detach().cpu().item()
                    ),
                    "audio_position_l1": float(audio_position_l1.detach().cpu().item()),
                    "pred_view_valid_mean": float(
                        evidence_output.evidence.view_valid_probability.mean().detach().cpu().item()
                    ),
                    "pred_pos_valid_mean": float(
                        evidence_output.evidence_state.pos_valid_probability.mean().detach().cpu().item()
                    ),
                    "pred_ori_valid_mean": float(
                        evidence_output.evidence_state.ori_valid_probability.mean().detach().cpu().item()
                    ),
                    "target_view_valid_mean": float(
                        batch["evidence_targets"].view_valid_target.mean().detach().cpu().item()
                    ),
                    "target_pos_valid_mean": float(
                        batch["evidence_targets"].pos_valid_target.mean().detach().cpu().item()
                    ),
                    "target_ori_valid_mean": float(
                        batch["evidence_targets"].ori_valid_target.mean().detach().cpu().item()
                    ),
                    "view_valid_match": float(
                        (
                            (
                                evidence_output.evidence.view_valid_probability >= 0.5
                            ).to(dtype=torch.float32)
                            == batch["evidence_targets"].view_valid_target
                        )
                        .to(dtype=torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "pos_valid_match": float(
                        (
                            (
                                evidence_output.evidence_state.pos_valid_probability >= 0.5
                            ).to(dtype=torch.float32)
                            == batch["evidence_targets"].pos_valid_target
                        )
                        .to(dtype=torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "ori_valid_match": float(
                        (
                            (
                                evidence_output.evidence_state.ori_valid_probability >= 0.5
                            ).to(dtype=torch.float32)
                            == batch["evidence_targets"].ori_valid_target
                        )
                        .to(dtype=torch.float32)
                        .mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "raw_visual_evidence_mean": float(
                        vision_output.raw_visual_evidence_strength.mean().detach().cpu().item()
                    ),
                    "raw_audio_evidence_mean": float(
                        audio_output.raw_audio_evidence_strength.mean().detach().cpu().item()
                    ),
                    "doa_conf_mean": float(audio_output.doa_conf.mean().detach().cpu().item()),
                    "dist_conf_mean": float(audio_output.dist_conf.mean().detach().cpu().item()),
                    "audio_position_confidence_mean": float(
                        evidence_output.evidence.audio_position_confidence.mean()
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "pnp_success_rate": float(
                        (
                            0.5
                            * (
                                vision_output.front_pnp_success.mean()
                                + vision_output.rear_pnp_success.mean()
                            )
                        )
                        .detach()
                        .cpu()
                        .item()
                    ),
                },
            }
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)


def _capture_audio_only_eval_snapshot(
    *,
    dataset: Any,
    config: TrainConfig,
    device: torch.device,
    audio: SingleStepAudioModule,
) -> dict[str, Any]:
    was_training = audio.training
    audio.eval()
    try:
        with torch.no_grad():
            sample = dataset[config.single_step.step_index % len(dataset)]
            batch = _collate_audio_only_batch([sample])
            batch = _move_audio_only_batch_to_device(batch, device)
            audio_output = audio(
                batch["audio_window"],
                batch["binaural_energy_t"],
                batch["binaural_cue_vector_t"],
            )
            audio_losses = compute_single_step_audio_loss(
                audio_output,
                batch["audio_targets"],
                weights=_audio_loss_weights_from_config(config),
                confidence=_audio_confidence_from_config(config),
            )
            return {
                "stage": "audio_only",
                "dataset_root": str(config.dataset_root),
                "num_samples": 1,
                "metrics": {
                    "total_loss": float(audio_losses["total"].detach().cpu().item()),
                    "audio_total_loss": float(audio_losses["total"].detach().cpu().item()),
                    "audio_doa_loss": float(audio_losses["doa"].detach().cpu().item()),
                    "audio_distance_loss": float(audio_losses["distance"].detach().cpu().item()),
                    "audio_doa_conf_loss": float(
                        audio_losses["doa_confidence"].detach().cpu().item()
                    ),
                    "audio_dist_conf_loss": float(
                        audio_losses["distance_confidence"].detach().cpu().item()
                    ),
                    "audio_doa_angle_error": float(
                        audio_losses["doa_angle_error"].detach().cpu().item()
                    ),
                    "audio_log_distance_error": float(
                        audio_losses["log_distance_error"].detach().cpu().item()
                    ),
                    "audio_doa_conf_target_mean": float(
                        audio_losses["target_doa_conf_mean"].detach().cpu().item()
                    ),
                    "audio_dist_conf_target_mean": float(
                        audio_losses["target_dist_conf_mean"].detach().cpu().item()
                    ),
                    "raw_audio_evidence_mean": float(
                        audio_output.raw_audio_evidence_strength.mean().detach().cpu().item()
                    ),
                    "doa_conf_mean": float(audio_output.doa_conf.mean().detach().cpu().item()),
                    "dist_conf_mean": float(audio_output.dist_conf.mean().detach().cpu().item()),
                },
            }
    finally:
        audio.train(was_training)


def _capture_segmentation_only_eval_snapshot(
    *,
    dataset: Any,
    config: TrainConfig,
    device: torch.device,
    vision,
) -> dict[str, Any]:
    was_training = vision.training
    vision.eval()
    try:
        with torch.no_grad():
            num_classes = segmentation_num_classes(config.segmentation_label_mode)
            sample = dataset[config.single_step.step_index % len(dataset)]
            batch = _collate_single_view_segmentation_batch(
                [sample],
                segmentation_label_mode=config.segmentation_label_mode,
                segmentation_roi_crop_enabled=config.segmentation_roi_crop.enabled,
                segmentation_roi_crop_context_scale=config.segmentation_roi_crop.context_scale,
                segmentation_roi_crop_min_crop_size=config.segmentation_roi_crop.min_crop_size,
                segmentation_roi_crop_square=config.segmentation_roi_crop.square,
                segmentation_target_patch_enabled=config.segmentation_target_patch.enabled,
                segmentation_target_patch_size=config.segmentation_target_patch.patch_size,
            )
            batch = _move_single_view_segmentation_batch_to_device(batch, device)
            vision_output = vision(batch["image"], batch["target_class_ids"])
            vision_losses = compute_single_view_segmentation_loss(
                vision_output,
                batch["segmentation_targets"],
                weights=_vision_loss_weights_from_config(config),
            )
            pred = vision_output.segmentation_logits.argmax(dim=1)
            gt = batch["segmentation_targets"].segmentation
            segmentation_iou = _mean_iou(
                vision_output.segmentation_logits,
                gt,
                num_classes,
            )
            target_iou = _target_iou(
                vision_output.segmentation_logits,
                gt,
                batch["target_class_ids"],
            )
            gt_selected_view = _selected_view_from_single_view_areas(
                view_index=batch["view_index"],
                target_area=batch["gt_target_area"],
            )
            pred_target_area = vision_output.pred_target_area
            pred_selected_view = _selected_view_from_single_view_areas(
                view_index=batch["view_index"],
                target_area=pred_target_area,
            )
            selected_view_agreement = (gt_selected_view == pred_selected_view).to(torch.float32).mean()
            target = batch["target_class_ids"].view(-1, 1, 1)
            tp = ((pred == target) & (gt == target)).sum().to(torch.float32)
            pred_pos = (pred == target).sum().to(torch.float32)
            gt_pos = (gt == target).sum().to(torch.float32)
            target_precision = tp / pred_pos.clamp_min(1.0)
            target_recall = tp / gt_pos.clamp_min(1.0)
            return {
                "stage": "segmentation_only",
                "dataset_root": str(config.dataset_root),
                "num_samples": 1,
                "metrics": {
                    "total_loss": float(vision_losses["segmentation"].detach().cpu().item()),
                    "vision_total_loss": float(vision_losses["total"].detach().cpu().item()),
                    "vision_segmentation_loss": float(
                        vision_losses["segmentation"].detach().cpu().item()
                    ),
                    "segmentation_iou_mean": float(segmentation_iou.detach().cpu().item()),
                    "target_iou": float(target_iou.detach().cpu().item()),
                    "target_precision": float(target_precision.detach().cpu().item()),
                    "target_recall": float(target_recall.detach().cpu().item()),
                    "selected_view_agreement": float(
                        selected_view_agreement.detach().cpu().item()
                    ),
                    "pred_selected_view_front": float((pred_selected_view == 0).to(torch.float32).mean().detach().cpu().item()),
                    "pred_selected_view_rear": float((pred_selected_view == 1).to(torch.float32).mean().detach().cpu().item()),
                    "pred_selected_view_none": float((pred_selected_view == 2).to(torch.float32).mean().detach().cpu().item()),
                    "gt_selected_view_front": float((gt_selected_view == 0).to(torch.float32).mean().detach().cpu().item()),
                    "gt_selected_view_rear": float((gt_selected_view == 1).to(torch.float32).mean().detach().cpu().item()),
                    "gt_selected_view_none": float((gt_selected_view == 2).to(torch.float32).mean().detach().cpu().item()),
                },
            }
    finally:
        vision.train(was_training)


def _capture_segmentation_only_aggregate_eval_snapshot(
    *,
    dataset: Any,
    config: TrainConfig,
    device: torch.device,
    vision,
) -> dict[str, Any]:
    was_training = vision.training
    vision.eval()
    try:
        with torch.no_grad():
            num_classes = segmentation_num_classes(config.segmentation_label_mode)
            intersection = torch.zeros(num_classes, dtype=torch.float64)
            union = torch.zeros(num_classes, dtype=torch.float64)
            target_intersection = torch.zeros((), dtype=torch.float64)
            target_union = torch.zeros((), dtype=torch.float64)
            tp = torch.zeros((), dtype=torch.float64)
            pred_pos = torch.zeros((), dtype=torch.float64)
            gt_pos = torch.zeros((), dtype=torch.float64)
            selected_view_matches = torch.zeros((), dtype=torch.float64)
            sample_count = 0
            pred_selected_front = 0
            pred_selected_rear = 0
            pred_selected_none = 0
            total_segmentation_loss = 0.0
            total_vision_loss = 0.0

            eval_loader = DataLoader(
                dataset,
                batch_size=config.schedule.batch_size,
                shuffle=False,
                collate_fn=partial(_single_view_segmentation_collate_fn, config=config),
                **_dataloader_kwargs(config),
            )
            for batch in eval_loader:
                batch = _move_single_view_segmentation_batch_to_device(batch, device)
                vision_output = vision(batch["image"], batch["target_class_ids"])
                vision_losses = compute_single_view_segmentation_loss(
                    vision_output,
                    batch["segmentation_targets"],
                    weights=_vision_loss_weights_from_config(config),
                )
                pred = vision_output.segmentation_logits.argmax(dim=1)
                gt = batch["segmentation_targets"].segmentation
                target = batch["target_class_ids"].view(-1, 1, 1)

                batch_size = pred.shape[0]
                sample_count += batch_size
                total_segmentation_loss += float(vision_losses["segmentation"].detach().cpu().item()) * batch_size
                total_vision_loss += float(vision_losses["total"].detach().cpu().item()) * batch_size

                for class_index in range(num_classes):
                    pred_mask = pred == class_index
                    gt_mask = gt == class_index
                    intersection[class_index] += (pred_mask & gt_mask).sum().item()
                    union[class_index] += (pred_mask | gt_mask).sum().item()

                gt_target_mask = gt == target
                pred_target_mask = pred == target
                target_intersection += (pred_target_mask & gt_target_mask).sum().item()
                target_union += (pred_target_mask | gt_target_mask).sum().item()
                tp += (pred_target_mask & gt_target_mask).sum().item()
                pred_pos += pred_target_mask.sum().item()
                gt_pos += gt_target_mask.sum().item()

                gt_selected_view = _selected_view_from_single_view_areas(
                    view_index=batch["view_index"],
                    target_area=batch["gt_target_area"],
                )
                pred_selected_view = _selected_view_from_single_view_areas(
                    view_index=batch["view_index"],
                    target_area=vision_output.pred_target_area,
                )
                selected_view_matches += (gt_selected_view == pred_selected_view).to(torch.float64).sum().item()
                pred_selected_front += int((pred_selected_view == 0).sum().item())
                pred_selected_rear += int((pred_selected_view == 1).sum().item())
                pred_selected_none += int((pred_selected_view == 2).sum().item())

            class_iou = torch.where(
                union > 0.0,
                intersection / union.clamp_min(1.0),
                torch.ones_like(union),
            )
            segmentation_iou_mean = class_iou.mean()
            target_iou = torch.where(
                target_union > 0.0,
                target_intersection / target_union.clamp_min(1.0),
                torch.zeros_like(target_union),
            )
            target_precision = tp / pred_pos.clamp_min(1.0)
            target_recall = tp / gt_pos.clamp_min(1.0)
            selected_view_agreement = selected_view_matches / max(sample_count, 1)

            return {
                "stage": "segmentation_only",
                "dataset_root": str(config.dataset_root),
                "num_samples": int(sample_count),
                "metrics": {
                    "total_loss": total_segmentation_loss / max(sample_count, 1),
                    "vision_total_loss": total_vision_loss / max(sample_count, 1),
                    "vision_segmentation_loss": total_segmentation_loss / max(sample_count, 1),
                    "segmentation_iou_mean": float(segmentation_iou_mean.detach().cpu().item()),
                    "target_iou": float(target_iou.detach().cpu().item()),
                    "target_precision": float(target_precision.detach().cpu().item()),
                    "target_recall": float(target_recall.detach().cpu().item()),
                    "selected_view_agreement": float(selected_view_agreement),
                    "pred_selected_view_front": float(pred_selected_front / max(sample_count, 1)),
                    "pred_selected_view_rear": float(pred_selected_rear / max(sample_count, 1)),
                    "pred_selected_view_none": float(pred_selected_none / max(sample_count, 1)),
                },
            }
    finally:
        vision.train(was_training)


def _capture_temporal_modality_eval_snapshot(
    *,
    dataset: WindowDataset,
    config: TrainConfig,
    device: torch.device,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
    stage1: TemporalModalityCalibrationStage,
) -> dict[str, Any]:
    modules = (vision, audio, evidence, stage1)
    previous_modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            sample = dataset[config.temporal_modality.window_index % len(dataset)]
            modality_inputs = _build_temporal_inputs_from_window_batch(
                [sample], vision=vision, audio=audio, evidence=evidence, device=device
            )
            targets = _collate_temporal_targets([sample], device)
            output = stage1(modality_inputs)
            losses = compute_temporal_modality_loss(output, targets)
            return {
                "stage": "temporal_modality",
                "dataset_root": str(config.dataset_root),
                "num_samples": 1,
                "metrics": {
                    "total_loss": float(losses["total"].detach().cpu().item()),
                    "position_loss": float(losses["position"].detach().cpu().item()),
                    "orientation_loss": float(losses["orientation"].detach().cpu().item()),
                    "position_confidence_loss": float(
                        losses["position_confidence"].detach().cpu().item()
                    ),
                    "orientation_confidence_loss": float(
                        losses["orientation_confidence"].detach().cpu().item()
                    ),
                    "visual_evidence_mean": float(
                        output.visual_evidence_strength.mean().detach().cpu().item()
                    ),
                    "audio_evidence_mean": float(
                        output.audio_evidence_strength.mean().detach().cpu().item()
                    ),
                },
            }
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)


def _capture_temporal_belief_eval_snapshot(
    *,
    dataset: WindowDataset,
    config: TrainConfig,
    device: torch.device,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
    stage1: TemporalModalityCalibrationStage,
    stage2: TemporalBeliefUpdateStage,
) -> dict[str, Any]:
    modules = (vision, audio, evidence, stage1, stage2)
    previous_modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            sample = dataset[config.temporal_belief.window_index % len(dataset)]
            modality_inputs = _build_temporal_inputs_from_window_batch(
                [sample], vision=vision, audio=audio, evidence=evidence, device=device
            )
            stage1_output = stage1(modality_inputs)
            belief_inputs = _build_belief_inputs([sample], stage1_output, device=device)
            targets = _collate_belief_targets([sample], device)
            output = stage2(belief_inputs)
            losses = compute_temporal_belief_loss(output, targets)
            return {
                "stage": "temporal_belief",
                "dataset_root": str(config.dataset_root),
                "num_samples": 1,
                "metrics": {
                    "total_loss": float(losses["total"].detach().cpu().item()),
                    "position_loss": float(losses["position"].detach().cpu().item()),
                    "orientation_loss": float(losses["orientation"].detach().cpu().item()),
                    "linear_velocity_loss": float(
                        losses["linear_velocity"].detach().cpu().item()
                    ),
                    "angular_velocity_loss": float(
                        losses["angular_velocity"].detach().cpu().item()
                    ),
                    "position_confidence_loss": float(
                        losses["position_confidence"].detach().cpu().item()
                    ),
                    "orientation_confidence_loss": float(
                        losses["orientation_confidence"].detach().cpu().item()
                    ),
                    "track_confidence_mean": float(
                        output.belief_state.track_confidence.mean().detach().cpu().item()
                    ),
                },
            }
    finally:
        for module, was_training in zip(modules, previous_modes, strict=True):
            module.train(was_training)


def run_training(
    config: TrainConfig,
    *,
    stage_override: str | None = None,
    num_steps_override: int | None = None,
    output_root_override: Path | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    stage = stage_override or config.stage
    num_steps = num_steps_override or config.schedule.num_steps
    output_root = output_root_override or config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    eval_dir = output_root / "eval"
    train_log_path = log_dir / "train_log.jsonl"

    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if stage == "single_step" and config.dataset_format == "synthetic_single_step":
        if config.segmentation_target_patch.enabled:
            raise ValueError(
                "dataset_format='synthetic_single_step' currently requires segmentation_target_patch.enabled=false "
                "because keypoint/voting labels remain in full-frame coordinates"
            )
        if config.segmentation_roi_crop.enabled:
            raise ValueError(
                "dataset_format='synthetic_single_step' currently requires segmentation_roi_crop.enabled=false "
                "because keypoint/voting labels remain in full-frame coordinates"
            )
    _write_json(
        output_root / "resolved_train_config.json",
        _resolved_config_dict(
            config,
            stage=stage,
            num_steps=num_steps,
            output_root=output_root,
            device=str(device),
        ),
    )

    history: list[dict[str, float]] = []
    resumed_from: str | None = None
    start_step = 0

    if stage == "audio_only":
        dataset = _build_step_dataset(config)
        audio = SingleStepAudioModule().to(device).train()
        optimizer = _build_optimizer(
            audio.parameters(),
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        modules = {
            "audio": audio,
        }
        if resume is not None:
            checkpoint = _load_checkpoint(
                resume,
                expected_stage=stage,
                optimizer=optimizer,
                modules=modules,
                device=device,
                allow_partial_module_load={"audio"},
            )
            resumed_from = str(resume)
            start_step = int(checkpoint["step"]) + 1
        first_loss = None
        last_loss = None
        last_completed_step: int | None = None
        train_loader = DataLoader(
            dataset,
            batch_sampler=_audio_only_batch_sampler(
                dataset=dataset,
                config=config,
                num_steps=num_steps,
                start_step=start_step,
            ),
            collate_fn=partial(_audio_only_collate_fn, config=config),
            **_dataloader_kwargs(config),
        )
        for loader_step, batch in enumerate(train_loader):
            step = start_step + loader_step
            batch = _move_audio_only_batch_to_device(batch, device)
            last_completed_step = step

            optimizer.zero_grad(set_to_none=True)
            audio_output = audio(
                batch["audio_window"],
                batch["binaural_energy_t"],
                batch["binaural_cue_vector_t"],
            )
            audio_losses = compute_single_step_audio_loss(
                audio_output,
                batch["audio_targets"],
                weights=_audio_loss_weights_from_config(config),
                confidence=_audio_confidence_from_config(config),
            )
            total = audio_losses["total"]
            total.backward()
            if config.schedule.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(audio.parameters(), config.schedule.grad_clip_norm)
            optimizer.step()
            loss_value = float(total.detach().cpu().item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            history.append({"step": float(step), "total_loss": loss_value})
            current_step = step + 1
            if step % config.schedule.log_interval == 0 or step == num_steps - 1:
                print(
                    f"step={step:04d} stage=audio_only total={loss_value:.6f} "
                    f"doa={audio_losses['doa'].item():.6f} "
                    f"distance={audio_losses['distance'].item():.6f}"
                )
                _append_jsonl(
                    train_log_path,
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "stage": stage,
                        "step": current_step,
                        "total_loss": loss_value,
                        "audio_total_loss": loss_value,
                        "audio_doa_loss": float(audio_losses["doa"].detach().cpu().item()),
                        "audio_distance_loss": float(
                            audio_losses["distance"].detach().cpu().item()
                        ),
                        "audio_doa_conf_loss": float(
                            audio_losses["doa_confidence"].detach().cpu().item()
                        ),
                        "audio_dist_conf_loss": float(
                            audio_losses["distance_confidence"].detach().cpu().item()
                        ),
                        "audio_doa_angle_error": float(
                            audio_losses["doa_angle_error"].detach().cpu().item()
                        ),
                        "audio_log_distance_error": float(
                            audio_losses["log_distance_error"].detach().cpu().item()
                        ),
                        "doa_conf_mean": float(audio_output.doa_conf.mean().detach().cpu().item()),
                        "dist_conf_mean": float(
                            audio_output.dist_conf.mean().detach().cpu().item()
                        ),
                        "raw_audio_evidence_mean": float(
                            audio_output.raw_audio_evidence_strength.mean().detach().cpu().item()
                        ),
                    },
                )
            _maybe_save_checkpoint(
                step=step,
                num_steps=num_steps,
                save_interval=config.schedule.save_interval,
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                optimizer=optimizer,
                modules=modules,
                output_root=output_root,
            )
            _maybe_write_eval_snapshot(
                step=step,
                num_steps=num_steps,
                eval_interval=config.schedule.eval_interval,
                eval_dir=eval_dir,
                payload=_capture_audio_only_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    audio=audio,
                ),
            )
        _ensure_final_checkpoint(
            completed_final_step=last_completed_step,
            num_steps=num_steps,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            optimizer=optimizer,
            modules=modules,
            output_root=output_root,
        )

    elif stage == "single_step":
        dataset = _build_step_dataset(config)
        vision = _build_vision_module_for_config(config).to(device)
        audio = SingleStepAudioModule().to(device)
        evidence = _build_evidence_module_for_config(config).to(device)
        if config.vision_freeze.segmentation_module:
            _freeze_single_step_segmentation_module(vision)
        parameter_groups, trainable_parameters = _single_step_optimizer_parameters(
            vision=vision,
            audio=audio,
            evidence=evidence,
            config=config,
        )
        optimizer = _build_optimizer(
            parameter_groups,
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        modules = {
            "vision": vision,
            "audio": audio,
            "evidence": evidence,
        }
        if resume is not None:
            allow_partial_module_load: set[str] = set()
            if config.single_step_loss_weights.audio == 0.0:
                allow_partial_module_load.add("audio")
            if config.single_step_loss_weights.evidence == 0.0:
                allow_partial_module_load.add("evidence")
            checkpoint = _load_checkpoint(
                resume,
                expected_stage=stage,
                optimizer=optimizer,
                modules=modules,
                device=device,
                allow_partial_module_load=allow_partial_module_load,
            )
            resumed_from = str(resume)
            start_step = int(checkpoint["step"]) + 1
        else:
            if config.init_vision_from is not None:
                init_result = _initialize_vision_from_checkpoint(
                    vision=vision,
                    checkpoint_path=config.init_vision_from,
                    device=device,
                )
                print(
                    f"initialized vision from {config.init_vision_from} "
                    f"(missing={len(init_result['missing_keys'])}, unexpected={len(init_result['unexpected_keys'])})"
                )
            if config.init_audio_from is not None:
                init_result = _initialize_audio_from_checkpoint(
                    audio=audio,
                    checkpoint_path=config.init_audio_from,
                    device=device,
                )
                print(
                    f"initialized audio from {config.init_audio_from} "
                    f"(missing={len(init_result['missing_keys'])}, unexpected={len(init_result['unexpected_keys'])})"
                )
            if config.init_segmentation_from is not None:
                init_result = _initialize_segmentation_submodules_from_checkpoint(
                    vision=vision,
                    checkpoint_path=config.init_segmentation_from,
                    device=device,
                )
                print(
                    f"initialized segmentation submodules from {config.init_segmentation_from} "
                    f"(missing={len(init_result['missing_keys'])}, unexpected={len(init_result['unexpected_keys'])})"
                )
        first_loss = None
        last_loss = None
        last_completed_step: int | None = None
        train_loader = DataLoader(
            dataset,
            batch_sampler=_single_step_batch_sampler(
                dataset=dataset,
                config=config,
                num_steps=num_steps,
                start_step=start_step,
            ),
            collate_fn=partial(_single_step_collate_fn, config=config),
            **_dataloader_kwargs(config),
        )
        for loader_step, batch in enumerate(train_loader):
            step = start_step + loader_step
            batch = _move_single_step_batch_to_device(batch, device)
            last_completed_step = step

            optimizer.zero_grad(set_to_none=True)
            front_aggregation_foreground, rear_aggregation_foreground = (
                _mixed_aggregation_foreground_from_targets(
                    batch["vision_targets"],
                    mix=config.aggregation_training.gt_foreground_mix,
                )
            )
            vision_output = vision(
                batch["front"],
                batch["rear"],
                batch["target_class_ids"],
                front_aggregation_foreground=front_aggregation_foreground,
                rear_aggregation_foreground=rear_aggregation_foreground,
                aggregation_foreground_mix=config.aggregation_training.gt_foreground_mix,
            )
            audio_output = audio(
                batch["audio_window"],
                batch["binaural_energy_t"],
                batch["binaural_cue_vector_t"],
            )
            evidence_output = evidence(vision_output, audio_output)
            vision_losses = compute_single_step_vision_loss(vision_output, batch["vision_targets"])
            audio_losses = compute_single_step_audio_loss(
                audio_output,
                batch["audio_targets"],
                weights=_audio_loss_weights_from_config(config),
                confidence=_audio_confidence_from_config(config),
            )
            evidence_losses = compute_single_step_evidence_loss(evidence_output, batch["evidence_targets"])
            total = _single_step_total_loss(
                config,
                vision_total=vision_losses["total"],
                audio_total=audio_losses["total"],
                evidence_total=evidence_losses["total"],
            )
            total.backward()
            if config.schedule.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    config.schedule.grad_clip_norm,
                )
            optimizer.step()
            loss_value = float(total.detach().cpu().item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            history.append({"step": float(step), "total_loss": loss_value})
            current_step = step + 1
            if step % config.schedule.log_interval == 0 or step == num_steps - 1:
                print(
                    f"step={step:04d} stage=single_step total={loss_value:.6f} "
                    f"vision={vision_losses['total'].item():.6f} "
                    f"audio={audio_losses['total'].item():.6f} "
                    f"evidence={evidence_losses['total'].item():.6f} "
                    f"weighted_evidence={(config.single_step_loss_weights.evidence * evidence_losses['total']).item():.6f}"
                )
                _append_jsonl(
                    train_log_path,
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "stage": stage,
                        "step": current_step,
                        "total_loss": loss_value,
                        "vision_loss": float(vision_losses["total"].detach().cpu().item()),
                        "audio_loss": float(audio_losses["total"].detach().cpu().item()),
                        "evidence_loss": float(evidence_losses["total"].detach().cpu().item()),
                        "weighted_vision_loss": float(
                            (
                                config.single_step_loss_weights.vision
                                * vision_losses["total"]
                            )
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "weighted_audio_loss": float(
                            (
                                config.single_step_loss_weights.audio
                                * audio_losses["total"]
                            )
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "weighted_evidence_loss": float(
                            (
                                config.single_step_loss_weights.evidence
                                * evidence_losses["total"]
                            )
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "vision_segmentation_loss": float(
                            vision_losses["segmentation"].detach().cpu().item()
                        ),
                        "vision_keypoint_loss": float(
                            vision_losses["keypoints"].detach().cpu().item()
                        ),
                        "vision_keypoint_voting_loss": float(
                            vision_losses["keypoint_voting"].detach().cpu().item()
                        ),
                        "audio_doa_loss": float(audio_losses["doa"].detach().cpu().item()),
                        "audio_distance_loss": float(
                            audio_losses["distance"].detach().cpu().item()
                        ),
                        "audio_doa_conf_loss": float(
                            audio_losses["doa_confidence"].detach().cpu().item()
                        ),
                        "audio_dist_conf_loss": float(
                            audio_losses["distance_confidence"].detach().cpu().item()
                        ),
                        "position_loss": float(
                            evidence_losses["position"].detach().cpu().item()
                        ),
                        "orientation_loss": float(
                            evidence_losses["orientation"].detach().cpu().item()
                        ),
                        "position_confidence_loss": float(
                            evidence_losses["position_confidence"].detach().cpu().item()
                        ),
                        "orientation_confidence_loss": float(
                            evidence_losses["orientation_confidence"].detach().cpu().item()
                        ),
                        "view_valid_loss": float(
                            evidence_losses["view_valid"].detach().cpu().item()
                        ),
                        "pos_valid_loss": float(
                            evidence_losses["pos_valid"].detach().cpu().item()
                        ),
                        "ori_valid_loss": float(
                            evidence_losses["ori_valid"].detach().cpu().item()
                        ),
                        "pred_view_valid_mean": float(
                            evidence_output.evidence.view_valid_probability.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "pred_pos_valid_mean": float(
                            evidence_output.evidence_state.pos_valid_probability.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "pred_ori_valid_mean": float(
                            evidence_output.evidence_state.ori_valid_probability.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "target_view_valid_mean": float(
                            batch["evidence_targets"].view_valid_target.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "target_pos_valid_mean": float(
                            batch["evidence_targets"].pos_valid_target.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "target_ori_valid_mean": float(
                            batch["evidence_targets"].ori_valid_target.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "doa_conf_mean": float(audio_output.doa_conf.mean().detach().cpu().item()),
                        "dist_conf_mean": float(audio_output.dist_conf.mean().detach().cpu().item()),
                        "audio_position_confidence_mean": float(
                            evidence_output.evidence.audio_position_confidence.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "raw_visual_evidence_mean": float(
                            vision_output.raw_visual_evidence_strength.mean()
                            .detach()
                            .cpu()
                            .item()
                        ),
                        "raw_audio_evidence_mean": float(
                            audio_output.raw_audio_evidence_strength.mean().detach().cpu().item()
                        ),
                    },
                )
            _maybe_save_checkpoint(
                step=step,
                num_steps=num_steps,
                save_interval=config.schedule.save_interval,
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                optimizer=optimizer,
                modules=modules,
                output_root=output_root,
            )
            _maybe_write_eval_snapshot(
                step=step,
                num_steps=num_steps,
                eval_interval=config.schedule.eval_interval,
                eval_dir=eval_dir,
                payload=_capture_single_step_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    vision=vision,
                    audio=audio,
                    evidence=evidence,
                ),
            )
        _ensure_final_checkpoint(
            completed_final_step=last_completed_step,
            num_steps=num_steps,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            optimizer=optimizer,
            modules=modules,
            output_root=output_root,
        )

    elif stage == "segmentation_only":
        dataset = _build_step_dataset(config)
        vision = _build_vision_module_for_config(config).to(device).train()
        optimizer = _build_optimizer(
            vision.parameters(),
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        modules = {
            "vision": vision,
        }
        if resume is not None:
            checkpoint = _load_checkpoint(
                resume,
                expected_stage=stage,
                optimizer=optimizer,
                modules=modules,
                device=device,
            )
            resumed_from = str(resume)
            start_step = int(checkpoint["step"]) + 1
        train_loader = DataLoader(
            dataset,
            batch_sampler=_WrappedStepBatchSampler(
                length=len(dataset),
                batch_size=config.schedule.batch_size,
                num_steps=num_steps - start_step,
                base_index=config.single_step.step_index + start_step * config.schedule.batch_size,
            ),
            collate_fn=partial(_single_view_segmentation_collate_fn, config=config),
            **_dataloader_kwargs(config),
        )
        first_loss = None
        last_loss = None
        best_target_recall = float("-inf")
        best_target_iou = float("-inf")
        last_completed_step: int | None = None
        for loader_step, batch in enumerate(train_loader):
            step = start_step + loader_step
            batch = _move_single_view_segmentation_batch_to_device(batch, device)
            last_completed_step = step

            optimizer.zero_grad(set_to_none=True)
            vision_output = vision(batch["image"], batch["target_class_ids"])
            vision_losses = compute_single_view_segmentation_loss(
                vision_output,
                batch["segmentation_targets"],
                weights=_vision_loss_weights_from_config(config),
            )
            total = vision_losses["segmentation"]
            total.backward()
            if config.schedule.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(vision.parameters(), config.schedule.grad_clip_norm)
            optimizer.step()
            loss_value = float(total.detach().cpu().item())
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            history.append({"step": float(step), "total_loss": loss_value})
            current_step = step + 1
            if step % config.schedule.log_interval == 0 or step == num_steps - 1:
                print(
                    f"step={step:04d} stage=segmentation_only total={loss_value:.6f}"
                )
                _append_jsonl(
                    train_log_path,
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "stage": stage,
                        "step": current_step,
                        "total_loss": loss_value,
                        "vision_total_loss": float(
                            vision_losses["total"].detach().cpu().item()
                        ),
                        "vision_segmentation_loss": float(
                            vision_losses["segmentation"].detach().cpu().item()
                        ),
                    },
                )
            _maybe_save_checkpoint(
                step=step,
                num_steps=num_steps,
                save_interval=config.schedule.save_interval,
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                optimizer=optimizer,
                modules=modules,
                output_root=output_root,
            )
            _maybe_write_eval_snapshot(
                step=step,
                num_steps=num_steps,
                eval_interval=config.schedule.eval_interval,
                eval_dir=eval_dir,
                payload=_capture_segmentation_only_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    vision=vision,
                ),
            )
            current_step = step + 1
            should_eval = current_step % config.schedule.eval_interval == 0 or current_step == num_steps
            if should_eval:
                aggregate_payload = _capture_segmentation_only_aggregate_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    vision=vision,
                )
                _write_json(
                    eval_dir / f"aggregated_eval_step_{current_step:06d}.json",
                    aggregate_payload,
                )
                aggregate_metrics = aggregate_payload["metrics"]
                target_recall = float(aggregate_metrics["target_recall"])
                target_iou = float(aggregate_metrics["target_iou"])
                if target_recall > best_target_recall:
                    best_target_recall = target_recall
                    _save_checkpoint(
                        checkpoint_dir / "best_target_recall.pt",
                        stage=stage,
                        step=step,
                        optimizer=optimizer,
                        modules=modules,
                        output_root=output_root,
                    )
                    _write_json(
                        eval_dir / "best_target_recall.json",
                        {
                            "step": current_step,
                            "metric": "target_recall",
                            "value": target_recall,
                            "checkpoint": str(checkpoint_dir / "best_target_recall.pt"),
                            "payload": aggregate_payload,
                        },
                    )
                if target_iou > best_target_iou:
                    best_target_iou = target_iou
                    _save_checkpoint(
                        checkpoint_dir / "best_target_iou.pt",
                        stage=stage,
                        step=step,
                        optimizer=optimizer,
                        modules=modules,
                        output_root=output_root,
                    )
                    _write_json(
                        eval_dir / "best_target_iou.json",
                        {
                            "step": current_step,
                            "metric": "target_iou",
                            "value": target_iou,
                            "checkpoint": str(checkpoint_dir / "best_target_iou.pt"),
                            "payload": aggregate_payload,
                        },
                    )
        _ensure_final_checkpoint(
            completed_final_step=last_completed_step,
            num_steps=num_steps,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            optimizer=optimizer,
            modules=modules,
            output_root=output_root,
        )
        if last_completed_step is not None and last_completed_step + 1 == num_steps:
            aggregate_payload = _capture_segmentation_only_aggregate_eval_snapshot(
                dataset=dataset,
                config=config,
                device=device,
                vision=vision,
            )
            _write_json(
                eval_dir / f"aggregated_eval_step_{num_steps:06d}.json",
                aggregate_payload,
            )

    elif stage == "temporal_modality":
        stage_cfg = config.temporal_modality
        dataset = WindowDataset(
            config.dataset_root,
            max_steps=stage_cfg.max_steps,
            min_stride_steps=stage_cfg.min_stride_steps,
            max_stride_steps=stage_cfg.max_stride_steps,
            random_seed=config.seed,
        )
        vision = _build_vision_module_for_config(config).to(device).eval()
        audio = SingleStepAudioModule().to(device).eval()
        evidence = _build_evidence_module_for_config(config).to(device).eval()
        for module in (vision, audio, evidence):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        stage1 = TemporalModalityCalibrationStage().to(device).train()
        optimizer = _build_optimizer(
            stage1.parameters(),
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        modules = {
            "temporal_modality_stage": stage1,
        }
        if resume is not None:
            checkpoint = _load_checkpoint(
                resume,
                expected_stage=stage,
                optimizer=optimizer,
                modules=modules,
                device=device,
            )
            resumed_from = str(resume)
            start_step = int(checkpoint["step"]) + 1
        first_loss = None
        last_loss = None
        last_completed_step: int | None = None
        for step in range(start_step, num_steps):
            batch_indices = _step_indices(
                len(dataset),
                step=step,
                batch_size=config.schedule.batch_size,
                base_index=stage_cfg.window_index,
            )
            samples = [dataset[index] for index in batch_indices]
            modality_inputs = _build_temporal_inputs_from_window_batch(
                samples, vision=vision, audio=audio, evidence=evidence, device=device
            )
            targets = _collate_temporal_targets(samples, device)

            optimizer.zero_grad(set_to_none=True)
            output = stage1(modality_inputs)
            losses = compute_temporal_modality_loss(output, targets)
            total = losses["total"]
            total.backward()
            if config.schedule.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(stage1.parameters(), config.schedule.grad_clip_norm)
            optimizer.step()
            loss_value = float(total.detach().cpu().item())
            last_completed_step = step
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            history.append({"step": float(step), "total_loss": loss_value})
            current_step = step + 1
            if step % config.schedule.log_interval == 0 or step == num_steps - 1:
                print(
                    f"step={step:04d} stage=temporal_modality total={loss_value:.6f} "
                    f"position={losses['position'].item():.6f} orientation={losses['orientation'].item():.6f}"
                )
                _append_jsonl(
                    train_log_path,
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "stage": stage,
                        "step": current_step,
                        "total_loss": loss_value,
                        "position_loss": float(losses["position"].detach().cpu().item()),
                        "orientation_loss": float(losses["orientation"].detach().cpu().item()),
                        "position_confidence_loss": float(
                            losses["position_confidence"].detach().cpu().item()
                        ),
                        "orientation_confidence_loss": float(
                            losses["orientation_confidence"].detach().cpu().item()
                        ),
                    },
                )
            _maybe_save_checkpoint(
                step=step,
                num_steps=num_steps,
                save_interval=config.schedule.save_interval,
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                optimizer=optimizer,
                modules=modules,
                output_root=output_root,
            )
            _maybe_write_eval_snapshot(
                step=step,
                num_steps=num_steps,
                eval_interval=config.schedule.eval_interval,
                eval_dir=eval_dir,
                payload=_capture_temporal_modality_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    vision=vision,
                    audio=audio,
                    evidence=evidence,
                    stage1=stage1,
                ),
            )
        _ensure_final_checkpoint(
            completed_final_step=last_completed_step,
            num_steps=num_steps,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            optimizer=optimizer,
            modules=modules,
            output_root=output_root,
        )

    elif stage == "temporal_belief":
        stage_cfg = config.temporal_belief
        dataset = WindowDataset(
            config.dataset_root,
            max_steps=stage_cfg.max_steps,
            min_stride_steps=stage_cfg.min_stride_steps,
            max_stride_steps=stage_cfg.max_stride_steps,
            random_seed=config.seed,
        )
        vision = _build_vision_module_for_config(config).to(device).eval()
        audio = SingleStepAudioModule().to(device).eval()
        evidence = _build_evidence_module_for_config(config).to(device).eval()
        for module in (vision, audio, evidence):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        stage1 = TemporalModalityCalibrationStage().to(device).train()
        stage2 = TemporalBeliefUpdateStage().to(device).train()
        optimizer = _build_optimizer(
            list(stage1.parameters()) + list(stage2.parameters()),
            name=config.optimizer.name,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        modules = {
            "temporal_modality_stage": stage1,
            "temporal_belief_stage": stage2,
        }
        if resume is not None:
            checkpoint = _load_checkpoint(
                resume,
                expected_stage=stage,
                optimizer=optimizer,
                modules=modules,
                device=device,
            )
            resumed_from = str(resume)
            start_step = int(checkpoint["step"]) + 1
        first_loss = None
        last_loss = None
        last_completed_step: int | None = None
        for step in range(start_step, num_steps):
            batch_indices = _step_indices(
                len(dataset),
                step=step,
                batch_size=config.schedule.batch_size,
                base_index=stage_cfg.window_index,
            )
            samples = [dataset[index] for index in batch_indices]
            modality_inputs = _build_temporal_inputs_from_window_batch(
                samples, vision=vision, audio=audio, evidence=evidence, device=device
            )
            targets = _collate_belief_targets(samples, device)

            optimizer.zero_grad(set_to_none=True)
            stage1_output = stage1(modality_inputs)
            belief_inputs = _build_belief_inputs(samples, stage1_output, device=device)
            stage2_output = stage2(belief_inputs)
            losses = compute_temporal_belief_loss(stage2_output, targets)
            total = losses["total"]
            total.backward()
            if config.schedule.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(stage1.parameters()) + list(stage2.parameters()),
                    config.schedule.grad_clip_norm,
                )
            optimizer.step()
            loss_value = float(total.detach().cpu().item())
            last_completed_step = step
            if first_loss is None:
                first_loss = loss_value
            last_loss = loss_value
            history.append({"step": float(step), "total_loss": loss_value})
            current_step = step + 1
            if step % config.schedule.log_interval == 0 or step == num_steps - 1:
                print(
                    f"step={step:04d} stage=temporal_belief total={loss_value:.6f} "
                    f"position={losses['position'].item():.6f} orientation={losses['orientation'].item():.6f} "
                    f"linear_velocity={losses['linear_velocity'].item():.6f} angular_velocity={losses['angular_velocity'].item():.6f}"
                )
                _append_jsonl(
                    train_log_path,
                    {
                        "timestamp_utc": _utc_now_iso(),
                        "stage": stage,
                        "step": current_step,
                        "total_loss": loss_value,
                        "position_loss": float(losses["position"].detach().cpu().item()),
                        "orientation_loss": float(losses["orientation"].detach().cpu().item()),
                        "linear_velocity_loss": float(
                            losses["linear_velocity"].detach().cpu().item()
                        ),
                        "angular_velocity_loss": float(
                            losses["angular_velocity"].detach().cpu().item()
                        ),
                        "position_confidence_loss": float(
                            losses["position_confidence"].detach().cpu().item()
                        ),
                        "orientation_confidence_loss": float(
                            losses["orientation_confidence"].detach().cpu().item()
                        ),
                    },
                )
            _maybe_save_checkpoint(
                step=step,
                num_steps=num_steps,
                save_interval=config.schedule.save_interval,
                checkpoint_dir=checkpoint_dir,
                stage=stage,
                optimizer=optimizer,
                modules=modules,
                output_root=output_root,
            )
            _maybe_write_eval_snapshot(
                step=step,
                num_steps=num_steps,
                eval_interval=config.schedule.eval_interval,
                eval_dir=eval_dir,
                payload=_capture_temporal_belief_eval_snapshot(
                    dataset=dataset,
                    config=config,
                    device=device,
                    vision=vision,
                    audio=audio,
                    evidence=evidence,
                    stage1=stage1,
                    stage2=stage2,
                ),
            )
        _ensure_final_checkpoint(
            completed_final_step=last_completed_step,
            num_steps=num_steps,
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            optimizer=optimizer,
            modules=modules,
            output_root=output_root,
        )
    else:
        raise ValueError(f"unsupported stage: {stage}")

    summary = {
        "stage": stage,
        "dataset_root": str(config.dataset_root),
        "output_root": str(output_root),
        "device": str(device),
        "num_steps": num_steps,
        "start_step": start_step,
        "completed_steps": len(history),
        "resumed_from": resumed_from,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "checkpoint_dir": str(checkpoint_dir),
        "train_log_path": str(train_log_path),
        "history": history,
    }
    _write_json(output_root / "train_summary.json", summary)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    config = load_train_config(args.config)
    run_training(
        config,
        stage_override=args.stage,
        num_steps_override=args.num_steps,
        output_root_override=args.output_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
