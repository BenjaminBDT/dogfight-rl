from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from dfb_state_estimation.datasets import (
    StepDataset,
    WindowDataset,
    target_segmentation_class_id,
)
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.evidence import SingleStepEvidenceModule
from dfb_state_estimation.models.temporal import (
    BeliefUpdateInputs,
    PolicyViewAdapter,
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


SEGMENTATION_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([40, 180, 40], dtype=np.uint8),
    2: np.array([40, 80, 220], dtype=np.uint8),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export eval sample visuals for Part 2.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-index", type=int)
    parser.add_argument("--window-index", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _rgba_image_to_bgr(image: Sequence[Sequence[Sequence[int]]]) -> np.ndarray:
    rgba = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _rgba_image_to_tensor(image: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
    return torch.tensor(image, dtype=torch.float32)[..., :3].permute(2, 0, 1) / 255.0


def _draw_points(
    image: np.ndarray,
    points_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    out = image.copy()
    for point in points_xy:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(out, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_reprojection_overlay(
    image: np.ndarray,
    predicted_xy: np.ndarray,
    reprojected_xy: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    for pred, reproj in zip(predicted_xy, reprojected_xy):
        p = (int(round(float(pred[0]))), int(round(float(pred[1]))))
        r = (int(round(float(reproj[0]))), int(round(float(reproj[1]))))
        cv2.line(out, p, r, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, p, 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, r, 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return out


def _segmentation_overlay(base_bgr: np.ndarray, logits: torch.Tensor) -> np.ndarray:
    pred = logits.argmax(dim=0).detach().cpu().numpy()
    color = np.zeros_like(base_bgr)
    for class_index, class_color in SEGMENTATION_COLORS.items():
        color[pred == class_index] = class_color
    return cv2.addWeighted(base_bgr, 0.7, color, 0.3, 0.0)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_curve(values: Sequence[float], *, title: str, width: int = 640, height: int = 240) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    if not values:
        return canvas
    v = np.asarray(values, dtype=np.float32)
    v_min = float(v.min())
    v_max = float(v.max())
    if abs(v_max - v_min) < 1e-6:
        v_max = v_min + 1.0
    x_coords = np.linspace(40, width - 20, num=len(v), dtype=np.float32)
    y_coords = (height - 30) - ((v - v_min) / (v_max - v_min)) * (height - 70)
    points = np.stack([x_coords, y_coords], axis=1).astype(np.int32)
    cv2.polylines(canvas, [points], isClosed=False, color=(30, 90, 220), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"min={v_min:.3f}", (16, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"max={v_max:.3f}", (180, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def _write_index_html(output_dir: Path, summary: dict) -> None:
    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>DFB State Estimation Eval Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111; background: #f7f7f7; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ margin: 0 0 20px; color: #444; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .card img {{ width: 100%; height: auto; display: block; background: #eee; }}
    pre {{ background: #111; color: #f2f2f2; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>DFB State Estimation Eval Report</h1>
  <div class="meta">
    step_index={summary["step_index"]}, window_index={summary["window_index"]}, window_end_model_step_index={summary["window_end_model_step_index"]}
  </div>
  <h2>Visual Overlays</h2>
  <div class="grid">
    <div class="card"><div>Front Segmentation</div><img src="front_segmentation_overlay.png" alt="front segmentation"></div>
    <div class="card"><div>Rear Segmentation</div><img src="rear_segmentation_overlay.png" alt="rear segmentation"></div>
    <div class="card"><div>Front Keypoints</div><img src="front_keypoints_overlay.png" alt="front keypoints"></div>
    <div class="card"><div>Rear Keypoints</div><img src="rear_keypoints_overlay.png" alt="rear keypoints"></div>
    <div class="card"><div>Front Reprojection</div><img src="front_reprojection_overlay.png" alt="front reprojection"></div>
    <div class="card"><div>Rear Reprojection</div><img src="rear_reprojection_overlay.png" alt="rear reprojection"></div>
  </div>
  <h2>Window Curves</h2>
  <div class="grid">
    <div class="card"><div>GT Position Norm</div><img src="window_position_norm.png" alt="position norm"></div>
    <div class="card"><div>GT Velocity Norm</div><img src="window_velocity_norm.png" alt="velocity norm"></div>
  </div>
  <h2>Summary</h2>
  <pre>{json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)}</pre>
</body>
</html>
"""
    _write_text(output_dir / "index.html", html)


def _build_window_temporal_inputs(
    sample,
    vision: SingleStepVisionModule,
    audio: SingleStepAudioModule,
    evidence: SingleStepEvidenceModule,
) -> tuple[TemporalModalityInputs, object]:
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
            [target_segmentation_class_id(step_sample.ref.observed_role)] * front.shape[0],
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


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_dataset = StepDataset(args.dataset_root)
    window_dataset = WindowDataset(args.dataset_root, max_steps=args.max_steps)
    window_sample = window_dataset[args.window_index]
    step_index = args.step_index
    if step_index is None:
        step_index = window_sample.ref.window_end_model_step_index
    step_sample = step_dataset[step_index]

    vision = SingleStepVisionModule().eval()
    audio = SingleStepAudioModule().eval()
    evidence = SingleStepEvidenceModule().eval()
    stage1 = TemporalModalityCalibrationStage().eval()
    stage2 = TemporalBeliefUpdateStage().eval()
    policy = PolicyViewAdapter().eval()

    with torch.no_grad():
        front_tensor = _rgba_image_to_tensor(step_sample.core["front_camera_image"]).unsqueeze(0)
        rear_tensor = _rgba_image_to_tensor(step_sample.core["rear_camera_image"]).unsqueeze(0)
        audio_window = torch.tensor(step_sample.core["audio_window_binaural"], dtype=torch.float32).unsqueeze(0)
        binaural_energy_t = torch.tensor(step_sample.audio_features["binaural_energy_t"], dtype=torch.float32).unsqueeze(0)
        binaural_cue_vector_t = torch.tensor(
            step_sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32
        ).unsqueeze(0)

        target_class_ids = torch.tensor(
            [target_segmentation_class_id(step_sample.ref.observed_role)],
            dtype=torch.long,
        )
        vision_output = vision(front_tensor, rear_tensor, target_class_ids)
        audio_output = audio(audio_window, binaural_energy_t, binaural_cue_vector_t)
        evidence_output = evidence(vision_output, audio_output)

        front_bgr = _rgba_image_to_bgr(step_sample.core["front_camera_image"])
        rear_bgr = _rgba_image_to_bgr(step_sample.core["rear_camera_image"])
        cv2.imwrite(
            str(args.output_dir / "front_segmentation_overlay.png"),
            _segmentation_overlay(front_bgr, vision_output.front_segmentation_logits[0]),
        )
        cv2.imwrite(
            str(args.output_dir / "rear_segmentation_overlay.png"),
            _segmentation_overlay(rear_bgr, vision_output.rear_segmentation_logits[0]),
        )

        front_scale = np.array([front_bgr.shape[1] - 1, front_bgr.shape[0] - 1], dtype=np.float32)
        rear_scale = np.array([rear_bgr.shape[1] - 1, rear_bgr.shape[0] - 1], dtype=np.float32)
        front_pred_xy = (vision_output.front_keypoints_xy[0].detach().cpu().numpy() * front_scale).astype(np.float32)
        rear_pred_xy = (vision_output.rear_keypoints_xy[0].detach().cpu().numpy() * rear_scale).astype(np.float32)
        front_gt_xy = np.asarray(step_sample.vision_labels["keypoints_2d_front"], dtype=np.float32)
        rear_gt_xy = np.asarray(step_sample.vision_labels["keypoints_2d_rear"], dtype=np.float32)

        cv2.imwrite(
            str(args.output_dir / "front_keypoints_overlay.png"),
            _draw_points(_draw_points(front_bgr, front_gt_xy, color=(0, 255, 0)), front_pred_xy, color=(0, 0, 255)),
        )
        cv2.imwrite(
            str(args.output_dir / "rear_keypoints_overlay.png"),
            _draw_points(_draw_points(rear_bgr, rear_gt_xy, color=(0, 255, 0)), rear_pred_xy, color=(0, 0, 255)),
        )

        front_projection = vision.geometry_validator.estimate_projection_batch(
            vision_output.front_keypoints_xy,
            vision_output.front_keypoint_support,
            image_height=front_bgr.shape[0],
            image_width=front_bgr.shape[1],
        )
        rear_projection = vision.geometry_validator.estimate_projection_batch(
            vision_output.rear_keypoints_xy,
            vision_output.rear_keypoint_support,
            image_height=rear_bgr.shape[0],
            image_width=rear_bgr.shape[1],
        )
        cv2.imwrite(
            str(args.output_dir / "front_reprojection_overlay.png"),
            _draw_reprojection_overlay(
                front_bgr,
                front_pred_xy,
                front_projection.projected_keypoints_px[0].detach().cpu().numpy(),
            ),
        )
        cv2.imwrite(
            str(args.output_dir / "rear_reprojection_overlay.png"),
            _draw_reprojection_overlay(
                rear_bgr,
                rear_pred_xy,
                rear_projection.projected_keypoints_px[0].detach().cpu().numpy(),
            ),
        )

        modality_inputs, temporal_evidence_output = _build_window_temporal_inputs(
            window_sample, vision, audio, evidence
        )
        stage1_output = stage1(modality_inputs)
        belief_inputs = BeliefUpdateInputs(
            coarse_state_t=stage1_output.coarse_state,
            context_relative_position=torch.tensor(window_sample.core["gt_relative_position"], dtype=torch.float32).unsqueeze(0),
            context_relative_orientation=torch.tensor(window_sample.core["gt_relative_orientation"], dtype=torch.float32).unsqueeze(0),
            context_position_confidence=torch.tensor(window_sample.rule_targets["target_pos_conf"], dtype=torch.float32).unsqueeze(0),
            context_orientation_confidence=torch.tensor(window_sample.rule_targets["target_ori_conf"], dtype=torch.float32).unsqueeze(0),
            visual_evidence_strength_t=stage1_output.visual_evidence_strength,
            audio_evidence_strength_t=stage1_output.audio_evidence_strength,
            linear_velocity=torch.tensor(window_sample.core["gt_linear_velocity"], dtype=torch.float32).unsqueeze(0),
            angular_velocity=torch.tensor(window_sample.core["gt_angular_velocity"], dtype=torch.float32).unsqueeze(0),
            dt_to_prev=torch.tensor(window_sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
            time_from_now=torch.tensor(window_sample.time_from_now, dtype=torch.float32).unsqueeze(0),
        )
        stage2_output = stage2(belief_inputs)
        policy_output = policy(stage2_output.belief_state)
    position_norm = [
        float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
        for v in window_sample.core["gt_relative_position"]
    ]
    velocity_norm = [
        float(np.linalg.norm(np.asarray(v, dtype=np.float32)))
        for v in window_sample.core["gt_linear_velocity"]
    ]
    cv2.imwrite(str(args.output_dir / "window_position_norm.png"), _render_curve(position_norm, title="GT Position Norm"))
    cv2.imwrite(str(args.output_dir / "window_velocity_norm.png"), _render_curve(velocity_norm, title="GT Linear Velocity Norm"))

    summary = {
        "step_index": step_index,
        "window_index": args.window_index,
        "window_end_model_step_index": window_sample.ref.window_end_model_step_index,
        "front_pnp_success": float(vision_output.front_pnp_success[0].detach().cpu().item()),
        "rear_pnp_success": float(vision_output.rear_pnp_success[0].detach().cpu().item()),
        "front_reprojection_error": float(vision_output.front_reprojection_error[0].detach().cpu().item()),
        "rear_reprojection_error": float(vision_output.rear_reprojection_error[0].detach().cpu().item()),
        "front_raw_visual_evidence_strength": float(
            vision_output.front_raw_visual_evidence_strength[0].detach().cpu().item()
        ),
        "rear_raw_visual_evidence_strength": float(
            vision_output.rear_raw_visual_evidence_strength[0].detach().cpu().item()
        ),
        "raw_audio_evidence_strength": float(audio_output.raw_audio_evidence_strength[0].detach().cpu().item()),
        "audio_state_t": {
            "doa_unit_vector_body": audio_output.doa_unit_vector_body[0].detach().cpu().tolist(),
            "doa_conf": float(audio_output.doa_conf[0].detach().cpu().item()),
            "log_distance_scalar": float(audio_output.log_distance_scalar[0].detach().cpu().item()),
            "dist_conf": float(audio_output.dist_conf[0].detach().cpu().item()),
            "audio_relative_position": evidence_output.evidence.audio_relative_position[0]
            .detach()
            .cpu()
            .tolist(),
            "audio_position_confidence": float(
                evidence_output.evidence.audio_position_confidence[0]
                .detach()
                .cpu()
                .item()
            ),
        },
        "evidence_state_t": {
            "relative_position": evidence_output.evidence_state.relative_position[0].detach().cpu().tolist(),
            "relative_orientation": evidence_output.evidence_state.relative_orientation[0].detach().cpu().tolist(),
            "position_confidence": float(evidence_output.evidence_state.position_confidence[0].detach().cpu().item()),
            "orientation_confidence": float(evidence_output.evidence_state.orientation_confidence[0].detach().cpu().item()),
            "visual_relative_position": evidence_output.evidence.visual_relative_position[0]
            .detach()
            .cpu()
            .tolist(),
            "visual_position_confidence": float(
                evidence_output.evidence.visual_position_confidence[0]
                .detach()
                .cpu()
                .item()
            ),
        },
        "coarse_state_t": {
            "relative_position": stage1_output.coarse_state.relative_position[0].detach().cpu().tolist(),
            "relative_orientation": stage1_output.coarse_state.relative_orientation[0].detach().cpu().tolist(),
            "position_confidence": float(stage1_output.coarse_state.position_confidence[0].detach().cpu().item()),
            "orientation_confidence": float(stage1_output.coarse_state.orientation_confidence[0].detach().cpu().item()),
            "visual_evidence_strength": float(stage1_output.visual_evidence_strength[0].detach().cpu().item()),
            "audio_evidence_strength": float(stage1_output.audio_evidence_strength[0].detach().cpu().item()),
        },
        "belief_state_t": {
            "relative_position": stage2_output.belief_state.relative_position[0].detach().cpu().tolist(),
            "relative_orientation": stage2_output.belief_state.relative_orientation[0].detach().cpu().tolist(),
            "linear_velocity": stage2_output.belief_state.linear_velocity[0].detach().cpu().tolist(),
            "angular_velocity": stage2_output.belief_state.angular_velocity[0].detach().cpu().tolist(),
            "position_confidence": float(stage2_output.belief_state.position_confidence[0].detach().cpu().item()),
            "orientation_confidence": float(stage2_output.belief_state.orientation_confidence[0].detach().cpu().item()),
            "track_confidence": float(stage2_output.belief_state.track_confidence[0].detach().cpu().item()),
        },
        "policy_view_t": {
            "relative_position": policy_output.relative_position[0].detach().cpu().tolist(),
            "relative_orientation": policy_output.relative_orientation[0].detach().cpu().tolist(),
            "position_confidence": float(policy_output.position_confidence[0].detach().cpu().item()),
            "orientation_confidence": float(policy_output.orientation_confidence[0].detach().cpu().item()),
            "linear_velocity": policy_output.linear_velocity[0].detach().cpu().tolist(),
            "angular_velocity": policy_output.angular_velocity[0].detach().cpu().tolist(),
            "track_confidence": float(policy_output.track_confidence[0].detach().cpu().item()),
        },
        "window_evidence_state_t": {
            "relative_position": temporal_evidence_output.evidence_state.relative_position[-1].detach().cpu().tolist(),
            "relative_orientation": temporal_evidence_output.evidence_state.relative_orientation[-1].detach().cpu().tolist(),
            "position_confidence": float(temporal_evidence_output.evidence_state.position_confidence[-1].detach().cpu().item()),
            "orientation_confidence": float(temporal_evidence_output.evidence_state.orientation_confidence[-1].detach().cpu().item()),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_index_html(args.output_dir, summary)


if __name__ == "__main__":
    main()
