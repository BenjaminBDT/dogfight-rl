from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.datasets import (
    StepDataset,
    WindowDataset,
    target_segmentation_class_id,
)
from dfb_state_estimation.losses import AudioSupervisionTargets, compute_audio_confidence_targets
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.audio.module import SingleStepAudioConfig, compute_audio_evidence_terms
from dfb_state_estimation.models.evidence import SingleStepEvidenceModule
from dfb_state_estimation.models.temporal import (
    BeliefUpdateInputs,
    TemporalBeliefUpdateStage,
    TemporalModalityCalibrationStage,
    TemporalModalityInputs,
    compute_selected_keypoint_delta_t,
    compute_selected_segmentation_difference_t,
    compute_delta_binaural_cue_t,
    select_view_tensor,
    select_view_target_probability,
)
from dfb_state_estimation.models.vision import SingleStepVisionModule

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


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _orientation_geodesic_degrees(prediction: Tensor, target: Tensor) -> Tensor:
    pred_r = _rotation_6d_to_matrix(prediction)
    target_r = _rotation_6d_to_matrix(target)
    rel = torch.matmul(pred_r.transpose(-1, -2), target_r)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_theta))


STAGES = ("single_step", "temporal_modality", "temporal_belief")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified eval runner for Part 2 stages.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=STAGES,
        required=True,
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser


def _rgba_image_to_tensor(image: list[list[list[int]]]) -> Tensor:
    return torch.tensor(image, dtype=torch.float32)[..., :3].permute(2, 0, 1) / 255.0


def _mean_iou(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    pred = logits.argmax(dim=1)
    ious: list[Tensor] = []
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


def _masked_keypoint_l1(pred_xy: Tensor, target_xy: Tensor, visibility_mask: Tensor) -> Tensor:
    mask = visibility_mask.unsqueeze(-1).to(dtype=pred_xy.dtype)
    denom = mask.sum().clamp_min(1.0)
    return ((pred_xy - target_xy).abs() * mask).sum() / denom


def _float(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _format_summary_text(result: dict[str, Any]) -> str:
    lines = [
        f"stage: {result['stage']}",
        f"dataset_root: {result['dataset_root']}",
        f"num_samples: {result['num_samples']}",
        "",
        "metrics:",
    ]
    metrics = result["metrics"]
    for key in sorted(metrics):
        lines.append(f"  {key}: {metrics[key]:.6f}")
    return "\n".join(lines) + "\n"


def _build_window_temporal_inputs(
    sample,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
) -> tuple[TemporalModalityInputs, Any]:
    front = torch.stack(
        [_rgba_image_to_tensor(image) for image in sample.core["front_camera_image"]],
        dim=0,
    )
    rear = torch.stack(
        [_rgba_image_to_tensor(image) for image in sample.core["rear_camera_image"]],
        dim=0,
    )
    audio_window = torch.tensor(sample.core["audio_window_binaural"], dtype=torch.float32)
    binaural_energy_t = torch.tensor(sample.audio_features["binaural_energy_t"], dtype=torch.float32)
    binaural_cue_vector_t = torch.tensor(
        sample.audio_features["binaural_cue_vector_t"],
        dtype=torch.float32,
    )
    delta_binaural_cue_t = compute_delta_binaural_cue_t(binaural_cue_vector_t.unsqueeze(0)).squeeze(0)

    with torch.no_grad():
        target_class_ids = torch.tensor(
            [target_segmentation_class_id(sample.ref.observed_role)] * front.shape[0],
            dtype=torch.long,
        )
        vision_output = vision(front, rear, target_class_ids)
        audio_output = audio(
            audio_window,
            binaural_energy_t,
            binaural_cue_vector_t,
        )
        evidence_output = evidence(vision_output, audio_output)

    selected_target_probability_t = select_view_target_probability(
        vision_output.front_segmentation_logits,
        vision_output.rear_segmentation_logits,
        target_class_ids,
        vision_output.selected_view_index,
    ).unsqueeze(0)
    selected_view_index_t = vision_output.selected_view_index.unsqueeze(0)
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
    ).unsqueeze(0)
    selected_keypoint_support_t = select_view_tensor(
        vision_output.front_keypoint_support,
        vision_output.rear_keypoint_support,
        vision_output.selected_view_index,
    ).unsqueeze(0)
    (
        selected_keypoint_delta_t,
        selected_keypoint_delta_support_summary_t,
        selected_keypoint_delta_valid_t,
    ) = compute_selected_keypoint_delta_t(
        selected_keypoints_xy_t,
        selected_keypoint_support_t,
        selected_view_index_t,
    )

    inputs = TemporalModalityInputs(
        relative_position=evidence_output.evidence_state.relative_position.unsqueeze(0),
        relative_orientation=evidence_output.evidence_state.relative_orientation.unsqueeze(0),
        position_confidence=evidence_output.evidence_state.position_confidence.unsqueeze(0),
        orientation_confidence=evidence_output.evidence_state.orientation_confidence.unsqueeze(0),
        pos_valid=evidence_output.evidence_state.pos_valid.unsqueeze(0),
        ori_valid=evidence_output.evidence_state.ori_valid.unsqueeze(0),
        visual_embedding=evidence_output.evidence.visual_embedding.unsqueeze(0),
        audio_embedding=evidence_output.evidence.audio_embedding.unsqueeze(0),
        raw_visual_evidence_strength=evidence_output.evidence.raw_visual_evidence_strength.unsqueeze(0),
        view_valid=vision_output.selected_candidate.view_valid.unsqueeze(0),
        selected_segmentation_difference_t=selected_segmentation_difference_t,
        selected_segmentation_diff_valid_t=selected_segmentation_diff_valid_t,
        selected_keypoint_delta_t=selected_keypoint_delta_t,
        selected_keypoint_delta_support_summary_t=selected_keypoint_delta_support_summary_t,
        selected_keypoint_delta_valid_t=selected_keypoint_delta_valid_t,
        raw_audio_evidence_strength=evidence_output.evidence.raw_audio_evidence_strength.unsqueeze(0),
        binaural_energy_t=binaural_energy_t.unsqueeze(0),
        binaural_cue_vector_t=binaural_cue_vector_t.unsqueeze(0),
        delta_binaural_cue_t=delta_binaural_cue_t.unsqueeze(0),
        dt_to_prev=torch.tensor(sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
        time_from_now=torch.tensor(sample.time_from_now, dtype=torch.float32).unsqueeze(0),
    )
    return inputs, evidence_output


def _evaluate_single_step(dataset_root: Path, num_samples: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset = StepDataset(dataset_root)
    vision = SingleStepVisionModule().eval()
    audio = SingleStepAudioModule().eval()
    evidence = SingleStepEvidenceModule().eval()
    metrics: dict[str, float] = {}
    sample_debug: list[dict[str, Any]] = []
    count = min(num_samples, len(dataset))
    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0)
            rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0)
            audio_window = torch.tensor(sample.core["audio_window_binaural"], dtype=torch.float32).unsqueeze(0)
            binaural_energy_t = torch.tensor(sample.audio_features["binaural_energy_t"], dtype=torch.float32).unsqueeze(0)
            binaural_cue_vector_t = torch.tensor(
                sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32
            ).unsqueeze(0)
            target_class_ids = torch.tensor(
                [target_segmentation_class_id(sample.ref.observed_role)],
                dtype=torch.long,
            )
            vision_output = vision(front, rear, target_class_ids)
            audio_output = audio(
                audio_window,
                binaural_energy_t,
                binaural_cue_vector_t,
            )
            a_energy, a_cue, _ = compute_audio_evidence_terms(
                binaural_energy_t=binaural_energy_t,
                binaural_cue_vector_t=binaural_cue_vector_t,
                config=SingleStepAudioConfig(),
            )
            gt_doa = torch.tensor(
                sample.rule_targets["gt_doa_unit_vector_body"], dtype=torch.float32
            ).unsqueeze(0)
            gt_log_distance = torch.tensor(
                sample.rule_targets["gt_log_distance_scalar"], dtype=torch.float32
            ).unsqueeze(0)
            audio_conf_targets = compute_audio_confidence_targets(
                audio_output,
                AudioSupervisionTargets(
                    gt_doa_unit_vector_body=gt_doa,
                    gt_log_distance_scalar=gt_log_distance,
                ),
            )
            evidence_output = evidence(vision_output, audio_output)

            front_seg_target = torch.tensor(
                sample.vision_labels["segmentation_mask_front"], dtype=torch.long
            ).unsqueeze(0)
            rear_seg_target = torch.tensor(
                sample.vision_labels["segmentation_mask_rear"], dtype=torch.long
            ).unsqueeze(0)
            front_vis_target = torch.tensor(
                sample.vision_labels["keypoint_visibility_front"], dtype=torch.float32
            ).unsqueeze(0)
            rear_vis_target = torch.tensor(
                sample.vision_labels["keypoint_visibility_rear"], dtype=torch.float32
            ).unsqueeze(0)
            scale = torch.tensor([front.shape[-1] - 1, front.shape[-2] - 1], dtype=torch.float32)
            front_kp_target = (
                torch.tensor(sample.vision_labels["keypoints_2d_front"], dtype=torch.float32).unsqueeze(0) / scale
            )
            rear_kp_target = (
                torch.tensor(sample.vision_labels["keypoints_2d_rear"], dtype=torch.float32).unsqueeze(0) / scale
            )
            gt_pos = torch.tensor(sample.core["gt_relative_position"], dtype=torch.float32).unsqueeze(0)
            gt_ori = torch.tensor(sample.core["gt_relative_orientation"], dtype=torch.float32).unsqueeze(0)
            pos_conf = torch.tensor(sample.rule_targets["target_pos_conf"], dtype=torch.float32).unsqueeze(0)
            ori_conf = torch.tensor(sample.rule_targets["target_ori_conf"], dtype=torch.float32).unsqueeze(0)
            pos_valid_mask = torch.tensor(
                [float(sample.rule_targets["target_pos_conf"] > 0.0)], dtype=torch.float32
            ).to(gt_pos.device)
            ori_valid_mask = torch.tensor(
                [float(sample.rule_targets["target_ori_conf"] > 0.0)], dtype=torch.float32
            ).to(gt_ori.device)
            evidence_position_error_norm = torch.linalg.vector_norm(
                evidence_output.evidence_state.relative_position - gt_pos,
                dim=-1,
            )
            evidence_orientation_geodesic_deg = _orientation_geodesic_degrees(
                evidence_output.evidence_state.relative_orientation,
                gt_ori,
            )

            sample_metrics = {
                "segmentation_ce": 0.5
                * (
                    F.cross_entropy(vision_output.front_segmentation_logits, front_seg_target)
                    + F.cross_entropy(vision_output.rear_segmentation_logits, rear_seg_target)
                ),
                "segmentation_iou_mean": 0.5
                * (
                    _mean_iou(vision_output.front_segmentation_logits, front_seg_target, 3)
                    + _mean_iou(vision_output.rear_segmentation_logits, rear_seg_target, 3)
                ),
                "keypoint_l1_visible": 0.5
                * (
                    _masked_keypoint_l1(vision_output.front_keypoints_xy, front_kp_target, front_vis_target)
                    + _masked_keypoint_l1(vision_output.rear_keypoints_xy, rear_kp_target, rear_vis_target)
                ),
                "pnp_success_rate": 0.5
                * (vision_output.front_pnp_success.mean() + vision_output.rear_pnp_success.mean()),
                "reprojection_error_mean": 0.5
                * (
                    vision_output.front_reprojection_error.mean()
                    + vision_output.rear_reprojection_error.mean()
                ),
                "evidence_position_l1": _masked_l1(
                    evidence_output.evidence_state.relative_position, gt_pos, pos_valid_mask
                ),
                "evidence_position_error_norm": _masked_l1(
                    evidence_position_error_norm.unsqueeze(-1),
                    torch.zeros_like(evidence_position_error_norm).unsqueeze(-1),
                    pos_valid_mask,
                ),
                "evidence_orientation_l1": _masked_l1(
                    evidence_output.evidence_state.relative_orientation, gt_ori, ori_valid_mask
                ),
                "evidence_orientation_geodesic_deg": _masked_l1(
                    evidence_orientation_geodesic_deg.unsqueeze(-1),
                    torch.zeros_like(evidence_orientation_geodesic_deg).unsqueeze(-1),
                    ori_valid_mask,
                ),
                "evidence_pos_conf_mse": _masked_mse(
                    evidence_output.evidence_state.position_confidence, pos_conf, pos_valid_mask
                ),
                "evidence_ori_conf_mse": _masked_mse(
                    evidence_output.evidence_state.orientation_confidence, ori_conf, ori_valid_mask
                ),
                "raw_visual_evidence_mean": vision_output.raw_visual_evidence_strength.mean(),
                "a_energy_mean": a_energy.mean(),
                "a_cue_mean": a_cue.mean(),
                "doa_conf_mean": audio_output.doa_conf.mean(),
                "dist_conf_mean": audio_output.dist_conf.mean(),
                "audio_position_confidence_mean": evidence_output.evidence.audio_position_confidence.mean(),
                "log_distance_mean": audio_output.log_distance_scalar.mean(),
                "audio_doa_angle_error": audio_conf_targets["doa_angle_error"].mean(),
                "audio_log_distance_error": audio_conf_targets["log_distance_error"].mean(),
                "audio_doa_conf_target_mean": audio_conf_targets["target_doa_conf"].mean(),
                "audio_dist_conf_target_mean": audio_conf_targets["target_dist_conf"].mean(),
                "audio_doa_conf_mse": F.mse_loss(
                    audio_output.doa_conf, audio_conf_targets["target_doa_conf"]
                ),
                "audio_dist_conf_mse": F.mse_loss(
                    audio_output.dist_conf, audio_conf_targets["target_dist_conf"]
                ),
                "audio_position_l1": F.l1_loss(
                    evidence_output.evidence.audio_relative_position, gt_pos
                ),
                "raw_audio_evidence_mean": audio_output.raw_audio_evidence_strength.mean(),
                "front_raw_visual_evidence_mean": vision_output.front_raw_visual_evidence_strength.mean(),
                "rear_raw_visual_evidence_mean": vision_output.rear_raw_visual_evidence_strength.mean(),
                "view_valid_match": (
                    (
                        evidence_output.evidence.view_valid_probability >= 0.5
                    ).to(dtype=torch.float32)
                    == torch.tensor(
                        [float(any(int(value) == int(target_segmentation_class_id(sample.ref.observed_role)) for row in sample.vision_labels["segmentation_mask_front"] for value in row) or any(int(value) == int(target_segmentation_class_id(sample.ref.observed_role)) for row in sample.vision_labels["segmentation_mask_rear"] for value in row))],
                        dtype=torch.float32,
                    )
                )
                .to(dtype=torch.float32)
                .mean(),
                "pos_valid_match": (
                    (evidence_output.evidence_state.pos_valid_probability >= 0.5).to(dtype=torch.float32)
                    == pos_valid_mask
                )
                .to(dtype=torch.float32)
                .mean(),
                "ori_valid_match": (
                    (evidence_output.evidence_state.ori_valid_probability >= 0.5).to(dtype=torch.float32)
                    == ori_valid_mask
                )
                .to(dtype=torch.float32)
                .mean(),
            }
            for key, value in sample_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + _float(value)
            sample_debug.append(
                {
                    "sample_index": index,
                    "episode_id": sample.ref.episode_id,
                    "observed_role": sample.ref.observed_role,
                    "chunk_id": sample.ref.chunk_id,
                    "chunk_step_offset": sample.ref.chunk_step_offset,
                    "global_model_step_index": sample.ref.global_model_step_index,
                    "simulation_step_index": sample.ref.simulation_step_index,
                    "pred_relative_position": evidence_output.evidence_state.relative_position[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "gt_relative_position": gt_pos[0].detach().cpu().tolist(),
                    "pred_relative_orientation_6d": evidence_output.evidence_state.relative_orientation[0]
                    .detach()
                    .cpu()
                    .tolist(),
                    "gt_relative_orientation_6d": gt_ori[0].detach().cpu().tolist(),
                    "position_error_norm": float(evidence_position_error_norm[0].detach().cpu().item()),
                    "orientation_geodesic_deg": float(
                        evidence_orientation_geodesic_deg[0].detach().cpu().item()
                    ),
                    "pred_view_valid_probability": float(
                        evidence_output.evidence.view_valid_probability[0].detach().cpu().item()
                    ),
                    "pred_pos_valid_probability": float(
                        evidence_output.evidence_state.pos_valid_probability[0].detach().cpu().item()
                    ),
                    "pred_ori_valid_probability": float(
                        evidence_output.evidence_state.ori_valid_probability[0].detach().cpu().item()
                    ),
                    "target_pos_valid": float(pos_valid_mask[0].detach().cpu().item()),
                    "target_ori_valid": float(ori_valid_mask[0].detach().cpu().item()),
                }
            )
    return {key: value / count for key, value in metrics.items()}, sample_debug


def _evaluate_temporal_modality(dataset_root: Path, num_samples: int, max_steps: int) -> dict[str, float]:
    dataset = WindowDataset(dataset_root, max_steps=max_steps)
    vision = SingleStepVisionModule().eval()
    audio = SingleStepAudioModule().eval()
    evidence = SingleStepEvidenceModule().eval()
    stage = TemporalModalityCalibrationStage().eval()
    metrics: dict[str, float] = {}
    count = min(num_samples, len(dataset))
    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            inputs, _ = _build_window_temporal_inputs(sample, vision, audio, evidence)
            output = stage(inputs)
            gt_pos = torch.tensor(sample.core["gt_relative_position"][-1], dtype=torch.float32).unsqueeze(0)
            gt_ori = torch.tensor(sample.core["gt_relative_orientation"][-1], dtype=torch.float32).unsqueeze(0)
            pos_conf = torch.tensor(sample.rule_targets["target_pos_conf"][-1], dtype=torch.float32).unsqueeze(0)
            ori_conf = torch.tensor(sample.rule_targets["target_ori_conf"][-1], dtype=torch.float32).unsqueeze(0)
            sample_metrics = {
                "coarse_position_l1": F.l1_loss(output.coarse_state.relative_position, gt_pos),
                "coarse_orientation_l1": F.l1_loss(output.coarse_state.relative_orientation, gt_ori),
                "coarse_pos_conf_mse": F.mse_loss(output.coarse_state.position_confidence, pos_conf),
                "coarse_ori_conf_mse": F.mse_loss(output.coarse_state.orientation_confidence, ori_conf),
                "visual_evidence_mean": output.visual_evidence_strength.mean(),
                "audio_evidence_mean": output.audio_evidence_strength.mean(),
            }
            for key, value in sample_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + _float(value)
    return {key: value / count for key, value in metrics.items()}


def _evaluate_temporal_belief(dataset_root: Path, num_samples: int, max_steps: int) -> dict[str, float]:
    dataset = WindowDataset(dataset_root, max_steps=max_steps)
    vision = SingleStepVisionModule().eval()
    audio = SingleStepAudioModule().eval()
    evidence = SingleStepEvidenceModule().eval()
    stage1 = TemporalModalityCalibrationStage().eval()
    stage2 = TemporalBeliefUpdateStage().eval()
    metrics: dict[str, float] = {}
    count = min(num_samples, len(dataset))
    with torch.no_grad():
        for index in range(count):
            sample = dataset[index]
            modality_inputs, _ = _build_window_temporal_inputs(sample, vision, audio, evidence)
            stage1_output = stage1(modality_inputs)
            belief_inputs = BeliefUpdateInputs(
                coarse_state_t=stage1_output.coarse_state,
                context_relative_position=torch.tensor(
                    sample.core["gt_relative_position"], dtype=torch.float32
                ).unsqueeze(0),
                context_relative_orientation=torch.tensor(
                    sample.core["gt_relative_orientation"], dtype=torch.float32
                ).unsqueeze(0),
                context_position_confidence=torch.tensor(
                    sample.rule_targets["target_pos_conf"], dtype=torch.float32
                ).unsqueeze(0),
                context_orientation_confidence=torch.tensor(
                    sample.rule_targets["target_ori_conf"], dtype=torch.float32
                ).unsqueeze(0),
                visual_evidence_strength_t=stage1_output.visual_evidence_strength,
                audio_evidence_strength_t=stage1_output.audio_evidence_strength,
                linear_velocity=torch.tensor(sample.core["gt_linear_velocity"], dtype=torch.float32).unsqueeze(0),
                angular_velocity=torch.tensor(sample.core["gt_angular_velocity"], dtype=torch.float32).unsqueeze(0),
                dt_to_prev=torch.tensor(sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
                time_from_now=torch.tensor(sample.time_from_now, dtype=torch.float32).unsqueeze(0),
            )
            output = stage2(belief_inputs)
            gt_pos = torch.tensor(sample.core["gt_relative_position"][-1], dtype=torch.float32).unsqueeze(0)
            gt_ori = torch.tensor(sample.core["gt_relative_orientation"][-1], dtype=torch.float32).unsqueeze(0)
            gt_lin = torch.tensor(sample.core["gt_linear_velocity"][-1], dtype=torch.float32).unsqueeze(0)
            gt_ang = torch.tensor(sample.core["gt_angular_velocity"][-1], dtype=torch.float32).unsqueeze(0)
            pos_conf = torch.tensor(sample.rule_targets["target_pos_conf"][-1], dtype=torch.float32).unsqueeze(0)
            ori_conf = torch.tensor(sample.rule_targets["target_ori_conf"][-1], dtype=torch.float32).unsqueeze(0)
            sample_metrics = {
                "belief_position_l1": F.l1_loss(output.belief_state.relative_position, gt_pos),
                "belief_orientation_l1": F.l1_loss(output.belief_state.relative_orientation, gt_ori),
                "belief_linear_velocity_l1": F.l1_loss(output.belief_state.linear_velocity, gt_lin),
                "belief_angular_velocity_l1": F.l1_loss(output.belief_state.angular_velocity, gt_ang),
                "belief_pos_conf_mse": F.mse_loss(output.belief_state.position_confidence, pos_conf),
                "belief_ori_conf_mse": F.mse_loss(output.belief_state.orientation_confidence, ori_conf),
                "track_confidence_mean": output.belief_state.track_confidence.mean(),
            }
            for key, value in sample_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + _float(value)
    return {key: value / count for key, value in metrics.items()}


def run_eval_stage(
    *,
    dataset_root: Path,
    stage: str,
    num_samples: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if stage == "single_step":
        metrics, sample_debug = _evaluate_single_step(dataset_root, num_samples)
        actual_num_samples = min(num_samples, len(StepDataset(dataset_root)))
    elif stage == "temporal_modality":
        metrics = _evaluate_temporal_modality(dataset_root, num_samples, max_steps)
        actual_num_samples = min(num_samples, len(WindowDataset(dataset_root, max_steps=max_steps)))
        sample_debug = None
    elif stage == "temporal_belief":
        metrics = _evaluate_temporal_belief(dataset_root, num_samples, max_steps)
        actual_num_samples = min(num_samples, len(WindowDataset(dataset_root, max_steps=max_steps)))
        sample_debug = None
    else:
        raise ValueError(f"unsupported stage: {stage}")

    result = {
        "stage": stage,
        "dataset_root": str(dataset_root),
        "num_samples": actual_num_samples,
        "metrics": metrics,
    }
    if sample_debug is not None:
        result["sample_debug"] = sample_debug
    return result


def main() -> None:
    args = _build_parser().parse_args()
    result = run_eval_stage(
        dataset_root=args.dataset_root,
        stage=args.stage,
        num_samples=args.num_samples,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
        (args.output_dir / "summary.txt").write_text(
            _format_summary_text(result),
            encoding="utf-8",
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
