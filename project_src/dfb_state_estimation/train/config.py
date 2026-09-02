from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dfb_state_estimation.datasets.semantics import SegmentationLabelMode


StageName = Literal["single_step", "segmentation_only", "temporal_modality", "temporal_belief", "audio_only"]
DatasetFormat = Literal["packed", "synthetic_segmentation", "synthetic_single_step", "audio_only_recordings"]
SegmentationLossMode = Literal["ce", "focal"]
VisionBackboneName = Literal["conv", "resnet18", "deeplabv3_resnet50"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class ScheduleConfig:
    num_steps: int
    batch_size: int
    grad_clip_norm: float
    log_interval: int
    eval_interval: int
    save_interval: int
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None


@dataclass(frozen=True)
class SingleStepTrainConfig:
    step_index: int


@dataclass(frozen=True)
class SingleStepLossWeightsConfig:
    vision: float = 1.0
    audio: float = 1.0
    evidence: float = 1.0


@dataclass(frozen=True)
class AudioLossWeightsConfig:
    doa: float = 1.0
    distance: float = 1.0
    doa_confidence: float = 0.5
    distance_confidence: float = 0.5


@dataclass(frozen=True)
class AudioConfidenceConfig:
    doa_half_angle_degrees: float = 15.0
    dist_half_error: float = 0.25


@dataclass(frozen=True)
class AudioSamplingConfig:
    enabled: bool = False
    low_ild_abs_threshold: float = 0.2
    low_ild_weight: float = 4.0
    medium_ild_abs_threshold: float = 1.0
    medium_ild_weight: float = 2.0
    high_gcc_peak_threshold: float = 0.999
    high_gcc_peak_weight: float = 1.5
    high_coherence_threshold: float = 0.999
    high_coherence_weight: float = 1.5
    high_lowband_ild_abs_threshold: float = 1.0
    high_lowband_ild_weight: float = 1.5
    high_highband_ild_abs_threshold: float = 1.0
    high_highband_ild_weight: float = 1.5
    low_energy_sum_threshold: float = 0.05
    low_energy_weight: float = 1.25
    max_combined_weight: float = 6.0


@dataclass(frozen=True)
class VisionLossComponentWeightsConfig:
    segmentation: float = 1.0
    keypoints: float = 1.0
    keypoint_voting: float = 1.0


@dataclass(frozen=True)
class KeypointSupervisionConfig:
    visible_weight: float = 1.0
    projectable_inframe_weight: float = 0.25


@dataclass(frozen=True)
class AggregationTrainingConfig:
    gt_foreground_mix: float = 0.0


@dataclass(frozen=True)
class EvidenceTrainConfig:
    position_refine_scale: float = 1.0


@dataclass(frozen=True)
class VisionGeometryTrainConfig:
    pnp_top_k_points: int = 6
    pnp_success_reprojection_threshold: float = 8.0
    pnp_success_min_selected_support_mean: float = 0.6
    pnp_success_min_camera_depth: float = 1.0
    pnp_success_max_camera_depth: float = 10000.0
    pnp_success_max_camera_translation_norm: float = 10000.0


@dataclass(frozen=True)
class VisionFreezeConfig:
    segmentation_module: bool = False


@dataclass(frozen=True)
class OptimizerGroupScalesConfig:
    vision_backbone: float = 1.0
    vision_segmentation_head: float = 1.0
    vision_keypoint_head: float = 1.0
    vision_embedding_head: float = 1.0
    audio: float = 1.0
    evidence: float = 1.0


@dataclass(frozen=True)
class SingleStepSamplingConfig:
    small_target_area_threshold: int = 0
    small_target_weight: float = 1.0
    medium_target_area_threshold: int = 0
    medium_target_weight: float = 1.0


@dataclass(frozen=True)
class SegmentationClassWeightsConfig:
    background: float = 1.0
    other_aircraft: float = 1.0
    target: float = 1.0


@dataclass(frozen=True)
class SegmentationLossConfig:
    mode: SegmentationLossMode = "ce"
    focal_gamma: float = 2.0
    dice_weight: float = 0.0


@dataclass(frozen=True)
class SegmentationRoiCropConfig:
    enabled: bool = False
    context_scale: float = 2.0
    min_crop_size: int = 64
    square: bool = True


@dataclass(frozen=True)
class SegmentationTargetPatchConfig:
    enabled: bool = False
    patch_size: int = 96
    positive_views_only: bool = True


@dataclass(frozen=True)
class VisionBackboneTrainConfig:
    name: VisionBackboneName = "conv"
    pretrained: bool = False


@dataclass(frozen=True)
class TemporalStageTrainConfig:
    window_index: int
    max_steps: int
    min_stride_steps: int
    max_stride_steps: int


@dataclass(frozen=True)
class TrainConfig:
    config_version: str
    run_name: str
    dataset_root: Path
    dataset_format: DatasetFormat
    output_root: Path
    init_vision_from: Path | None
    init_audio_from: Path | None
    init_segmentation_from: Path | None
    seed: int
    device: str
    stage: StageName
    segmentation_label_mode: SegmentationLabelMode
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    single_step: SingleStepTrainConfig
    single_step_loss_weights: SingleStepLossWeightsConfig
    audio_loss_weights: AudioLossWeightsConfig
    audio_confidence: AudioConfidenceConfig
    audio_sampling: AudioSamplingConfig
    vision_loss_weights: VisionLossComponentWeightsConfig
    keypoint_supervision: KeypointSupervisionConfig
    aggregation_training: AggregationTrainingConfig
    evidence: EvidenceTrainConfig
    vision_geometry: VisionGeometryTrainConfig
    vision_freeze: VisionFreezeConfig
    optimizer_group_scales: OptimizerGroupScalesConfig
    single_step_sampling: SingleStepSamplingConfig
    segmentation_class_weights: SegmentationClassWeightsConfig
    segmentation_loss: SegmentationLossConfig
    segmentation_roi_crop: SegmentationRoiCropConfig
    segmentation_target_patch: SegmentationTargetPatchConfig
    vision_backbone: VisionBackboneTrainConfig
    temporal_modality: TemporalStageTrainConfig
    temporal_belief: TemporalStageTrainConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _build_temporal_stage_config(name: str, payload: dict[str, Any]) -> TemporalStageTrainConfig:
    cfg = TemporalStageTrainConfig(
        window_index=int(payload["window_index"]),
        max_steps=int(payload["max_steps"]),
        min_stride_steps=int(payload["min_stride_steps"]),
        max_stride_steps=int(payload["max_stride_steps"]),
    )
    _require_positive_int(f"{name}.max_steps", cfg.max_steps)
    _require_positive_int(f"{name}.min_stride_steps", cfg.min_stride_steps)
    _require_positive_int(f"{name}.max_stride_steps", cfg.max_stride_steps)
    if cfg.max_stride_steps < cfg.min_stride_steps:
        raise ValueError(
            f"{name}.max_stride_steps must be >= {name}.min_stride_steps"
        )
    return cfg


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    stage = payload["stage"]
    if stage not in {"single_step", "segmentation_only", "temporal_modality", "temporal_belief", "audio_only"}:
        raise ValueError(f"unsupported stage: {stage}")
    dataset_format = str(payload.get("dataset_format", "packed"))
    if dataset_format not in {"packed", "synthetic_segmentation", "synthetic_single_step", "audio_only_recordings"}:
        raise ValueError(f"unsupported dataset_format: {dataset_format}")
    segmentation_label_mode = str(payload.get("segmentation_label_mode", "multiclass_absolute"))
    if segmentation_label_mode not in {"multiclass_absolute", "binary_target"}:
        raise ValueError(f"unsupported segmentation_label_mode: {segmentation_label_mode}")

    optimizer = OptimizerConfig(
        name=str(payload["optimizer"]["name"]),
        lr=float(payload["optimizer"]["lr"]),
        weight_decay=float(payload["optimizer"]["weight_decay"]),
    )
    schedule = ScheduleConfig(
        num_steps=int(payload["schedule"]["num_steps"]),
        batch_size=int(payload["schedule"]["batch_size"]),
        grad_clip_norm=float(payload["schedule"]["grad_clip_norm"]),
        log_interval=int(payload["schedule"]["log_interval"]),
        eval_interval=int(payload["schedule"]["eval_interval"]),
        save_interval=int(payload["schedule"]["save_interval"]),
        num_workers=int(payload["schedule"].get("num_workers", 0)),
        pin_memory=bool(payload["schedule"].get("pin_memory", False)),
        persistent_workers=bool(payload["schedule"].get("persistent_workers", False)),
        prefetch_factor=(
            int(payload["schedule"]["prefetch_factor"])
            if payload["schedule"].get("prefetch_factor") is not None
            else None
        ),
    )
    _require_positive_int("schedule.num_steps", schedule.num_steps)
    _require_positive_int("schedule.batch_size", schedule.batch_size)
    _require_positive_int("schedule.log_interval", schedule.log_interval)
    _require_positive_int("schedule.eval_interval", schedule.eval_interval)
    _require_positive_int("schedule.save_interval", schedule.save_interval)
    if schedule.num_workers < 0:
        raise ValueError("schedule.num_workers must be >= 0")
    if schedule.prefetch_factor is not None and schedule.prefetch_factor <= 0:
        raise ValueError("schedule.prefetch_factor must be positive when provided")
    if schedule.persistent_workers and schedule.num_workers == 0:
        raise ValueError("schedule.persistent_workers requires schedule.num_workers > 0")
    if schedule.prefetch_factor is not None and schedule.num_workers == 0:
        raise ValueError("schedule.prefetch_factor requires schedule.num_workers > 0")

    single_step = SingleStepTrainConfig(
        step_index=int(payload["single_step"]["step_index"]),
    )
    single_step_loss_weights_payload = payload.get("single_step_loss_weights", {})
    single_step_loss_weights = SingleStepLossWeightsConfig(
        vision=float(single_step_loss_weights_payload.get("vision", 1.0)),
        audio=float(single_step_loss_weights_payload.get("audio", 1.0)),
        evidence=float(single_step_loss_weights_payload.get("evidence", 1.0)),
    )
    audio_loss_weights_payload = payload.get("audio_loss_weights", {})
    audio_loss_weights = AudioLossWeightsConfig(
        doa=float(audio_loss_weights_payload.get("doa", 1.0)),
        distance=float(audio_loss_weights_payload.get("distance", 1.0)),
        doa_confidence=float(audio_loss_weights_payload.get("doa_confidence", 0.5)),
        distance_confidence=float(
            audio_loss_weights_payload.get("distance_confidence", 0.5)
        ),
    )
    audio_confidence_payload = payload.get("audio_confidence", {})
    audio_confidence = AudioConfidenceConfig(
        doa_half_angle_degrees=float(
            audio_confidence_payload.get("doa_half_angle_degrees", 15.0)
        ),
        dist_half_error=float(audio_confidence_payload.get("dist_half_error", 0.25)),
    )
    audio_sampling_payload = payload.get("audio_sampling", {})
    audio_sampling = AudioSamplingConfig(
        enabled=bool(audio_sampling_payload.get("enabled", False)),
        low_ild_abs_threshold=float(audio_sampling_payload.get("low_ild_abs_threshold", 0.2)),
        low_ild_weight=float(audio_sampling_payload.get("low_ild_weight", 4.0)),
        medium_ild_abs_threshold=float(
            audio_sampling_payload.get("medium_ild_abs_threshold", 1.0)
        ),
        medium_ild_weight=float(audio_sampling_payload.get("medium_ild_weight", 2.0)),
        high_gcc_peak_threshold=float(
            audio_sampling_payload.get("high_gcc_peak_threshold", 0.999)
        ),
        high_gcc_peak_weight=float(audio_sampling_payload.get("high_gcc_peak_weight", 1.5)),
        high_coherence_threshold=float(
            audio_sampling_payload.get("high_coherence_threshold", 0.999)
        ),
        high_coherence_weight=float(
            audio_sampling_payload.get("high_coherence_weight", 1.5)
        ),
        high_lowband_ild_abs_threshold=float(
            audio_sampling_payload.get("high_lowband_ild_abs_threshold", 1.0)
        ),
        high_lowband_ild_weight=float(
            audio_sampling_payload.get("high_lowband_ild_weight", 1.5)
        ),
        high_highband_ild_abs_threshold=float(
            audio_sampling_payload.get("high_highband_ild_abs_threshold", 1.0)
        ),
        high_highband_ild_weight=float(
            audio_sampling_payload.get("high_highband_ild_weight", 1.5)
        ),
        low_energy_sum_threshold=float(
            audio_sampling_payload.get("low_energy_sum_threshold", 0.05)
        ),
        low_energy_weight=float(audio_sampling_payload.get("low_energy_weight", 1.25)),
        max_combined_weight=float(audio_sampling_payload.get("max_combined_weight", 6.0)),
    )
    vision_loss_weights_payload = payload.get("vision_loss_weights", {})
    vision_loss_weights = VisionLossComponentWeightsConfig(
        segmentation=float(vision_loss_weights_payload.get("segmentation", 1.0)),
        keypoints=float(vision_loss_weights_payload.get("keypoints", 1.0)),
        keypoint_voting=float(vision_loss_weights_payload.get("keypoint_voting", 1.0)),
    )
    keypoint_supervision_payload = payload.get("keypoint_supervision", {})
    keypoint_supervision = KeypointSupervisionConfig(
        visible_weight=float(keypoint_supervision_payload.get("visible_weight", 1.0)),
        projectable_inframe_weight=float(
            keypoint_supervision_payload.get("projectable_inframe_weight", 0.25)
        ),
    )
    aggregation_training_payload = payload.get("aggregation_training", {})
    aggregation_training = AggregationTrainingConfig(
        gt_foreground_mix=float(aggregation_training_payload.get("gt_foreground_mix", 0.0))
    )
    evidence_payload = payload.get("evidence", {})
    evidence = EvidenceTrainConfig(
        position_refine_scale=float(evidence_payload.get("position_refine_scale", 1.0))
    )
    vision_geometry_payload = payload.get("vision_geometry", {})
    vision_geometry = VisionGeometryTrainConfig(
        pnp_top_k_points=int(vision_geometry_payload.get("pnp_top_k_points", 6)),
        pnp_success_reprojection_threshold=float(
            vision_geometry_payload.get("pnp_success_reprojection_threshold", 8.0)
        ),
        pnp_success_min_selected_support_mean=float(
            vision_geometry_payload.get("pnp_success_min_selected_support_mean", 0.6)
        ),
        pnp_success_min_camera_depth=float(
            vision_geometry_payload.get("pnp_success_min_camera_depth", 1.0)
        ),
        pnp_success_max_camera_depth=float(
            vision_geometry_payload.get("pnp_success_max_camera_depth", 10000.0)
        ),
        pnp_success_max_camera_translation_norm=float(
            vision_geometry_payload.get("pnp_success_max_camera_translation_norm", 10000.0)
        ),
    )
    _require_positive_int("vision_geometry.pnp_top_k_points", vision_geometry.pnp_top_k_points)
    vision_freeze_payload = payload.get("vision_freeze", {})
    vision_freeze = VisionFreezeConfig(
        segmentation_module=bool(vision_freeze_payload.get("segmentation_module", False))
    )
    optimizer_group_scales_payload = payload.get("optimizer_group_scales", {})
    optimizer_group_scales = OptimizerGroupScalesConfig(
        vision_backbone=float(optimizer_group_scales_payload.get("vision_backbone", 1.0)),
        vision_segmentation_head=float(
            optimizer_group_scales_payload.get("vision_segmentation_head", 1.0)
        ),
        vision_keypoint_head=float(
            optimizer_group_scales_payload.get("vision_keypoint_head", 1.0)
        ),
        vision_embedding_head=float(
            optimizer_group_scales_payload.get("vision_embedding_head", 1.0)
        ),
        audio=float(optimizer_group_scales_payload.get("audio", 1.0)),
        evidence=float(optimizer_group_scales_payload.get("evidence", 1.0)),
    )
    single_step_sampling_payload = payload.get("single_step_sampling", {})
    single_step_sampling = SingleStepSamplingConfig(
        small_target_area_threshold=int(
            single_step_sampling_payload.get("small_target_area_threshold", 0)
        ),
        small_target_weight=float(single_step_sampling_payload.get("small_target_weight", 1.0)),
        medium_target_area_threshold=int(
            single_step_sampling_payload.get("medium_target_area_threshold", 0)
        ),
        medium_target_weight=float(
            single_step_sampling_payload.get("medium_target_weight", 1.0)
        ),
    )
    segmentation_class_weights_payload = payload.get("segmentation_class_weights", {})
    segmentation_class_weights = SegmentationClassWeightsConfig(
        background=float(segmentation_class_weights_payload.get("background", 1.0)),
        other_aircraft=float(segmentation_class_weights_payload.get("other_aircraft", 1.0)),
        target=float(segmentation_class_weights_payload.get("target", 1.0)),
    )
    segmentation_loss_payload = payload.get("segmentation_loss", {})
    segmentation_loss_mode = str(segmentation_loss_payload.get("mode", "ce"))
    if segmentation_loss_mode not in {"ce", "focal"}:
        raise ValueError(f"unsupported segmentation_loss.mode: {segmentation_loss_mode}")
    segmentation_loss = SegmentationLossConfig(
        mode=segmentation_loss_mode,
        focal_gamma=float(segmentation_loss_payload.get("focal_gamma", 2.0)),
        dice_weight=float(segmentation_loss_payload.get("dice_weight", 0.0)),
    )
    segmentation_roi_crop_payload = payload.get("segmentation_roi_crop", {})
    segmentation_roi_crop = SegmentationRoiCropConfig(
        enabled=bool(segmentation_roi_crop_payload.get("enabled", False)),
        context_scale=float(segmentation_roi_crop_payload.get("context_scale", 2.0)),
        min_crop_size=int(segmentation_roi_crop_payload.get("min_crop_size", 64)),
        square=bool(segmentation_roi_crop_payload.get("square", True)),
    )
    segmentation_target_patch_payload = payload.get("segmentation_target_patch", {})
    segmentation_target_patch = SegmentationTargetPatchConfig(
        enabled=bool(segmentation_target_patch_payload.get("enabled", False)),
        patch_size=int(segmentation_target_patch_payload.get("patch_size", 96)),
        positive_views_only=bool(
            segmentation_target_patch_payload.get("positive_views_only", True)
        ),
    )
    vision_backbone_payload = payload.get("vision_backbone", {})
    vision_backbone_name = str(vision_backbone_payload.get("name", "conv"))
    if vision_backbone_name not in {"conv", "resnet18", "deeplabv3_resnet50"}:
        raise ValueError(f"unsupported vision_backbone.name: {vision_backbone_name}")
    vision_backbone = VisionBackboneTrainConfig(
        name=vision_backbone_name,
        pretrained=bool(vision_backbone_payload.get("pretrained", False)),
    )
    temporal_modality = _build_temporal_stage_config(
        "temporal_modality", payload["temporal_modality"]
    )
    temporal_belief = _build_temporal_stage_config(
        "temporal_belief", payload["temporal_belief"]
    )

    return TrainConfig(
        config_version=str(payload["config_version"]),
        run_name=str(payload["run_name"]),
        dataset_root=_resolve_path(str(payload["dataset_root"])),
        dataset_format=dataset_format,
        output_root=_resolve_path(str(payload["output_root"])),
        init_vision_from=(
            _resolve_path(str(payload["init_vision_from"]))
            if payload.get("init_vision_from") is not None
            else None
        ),
        init_audio_from=(
            _resolve_path(str(payload["init_audio_from"]))
            if payload.get("init_audio_from") is not None
            else None
        ),
        init_segmentation_from=(
            _resolve_path(str(payload["init_segmentation_from"]))
            if payload.get("init_segmentation_from") is not None
            else None
        ),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        stage=stage,
        segmentation_label_mode=segmentation_label_mode,
        optimizer=optimizer,
        schedule=schedule,
        single_step=single_step,
        single_step_loss_weights=single_step_loss_weights,
        audio_loss_weights=audio_loss_weights,
        audio_confidence=audio_confidence,
        audio_sampling=audio_sampling,
        vision_loss_weights=vision_loss_weights,
        keypoint_supervision=keypoint_supervision,
        aggregation_training=aggregation_training,
        evidence=evidence,
        vision_geometry=vision_geometry,
        vision_freeze=vision_freeze,
        optimizer_group_scales=optimizer_group_scales,
        single_step_sampling=single_step_sampling,
        segmentation_class_weights=segmentation_class_weights,
        segmentation_loss=segmentation_loss,
        segmentation_roi_crop=segmentation_roi_crop,
        segmentation_target_patch=segmentation_target_patch,
        vision_backbone=vision_backbone,
        temporal_modality=temporal_modality,
        temporal_belief=temporal_belief,
    )
