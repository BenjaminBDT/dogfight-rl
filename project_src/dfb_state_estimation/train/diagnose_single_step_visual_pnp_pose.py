from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from dfb_state_estimation.datasets import StepDataset, target_segmentation_class_id
from dfb_state_estimation.models.vision import SingleStepVisionModule
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import _build_vision_module_for_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose single-step visual PnP body pose against GT."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _rgba_image_to_tensor(image: list[list[list[int]]]) -> torch.Tensor:
    tensor = torch.tensor(image, dtype=torch.float32)[..., :3].permute(2, 0, 1)
    return tensor / 255.0


def _load_vision_module(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> SingleStepVisionModule:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = payload.get("modules")
    config = load_train_config(config_path)
    vision = _build_vision_module_for_config(config).to(device).eval()
    if isinstance(module_states, dict) and "vision" in module_states:
        vision.load_state_dict(module_states["vision"], strict=False)
    else:
        vision.load_state_dict(payload, strict=False)
    return vision


def _sample_indices(dataset_len: int, num_samples: int) -> list[int]:
    if dataset_len <= 0 or num_samples <= 0:
        return []
    if num_samples >= dataset_len:
        return list(range(dataset_len))
    if num_samples == 1:
        return [0]
    step = (dataset_len - 1) / float(num_samples - 1)
    indices: list[int] = []
    for i in range(num_samples):
        index = int(round(i * step))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _orientation_geodesic_degrees(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_r = _rotation_6d_to_matrix(prediction.unsqueeze(0))[0]
    target_r = _rotation_6d_to_matrix(target.unsqueeze(0))[0]
    rel = torch.matmul(pred_r.transpose(-1, -2), target_r)
    trace = rel[0, 0] + rel[1, 1] + rel[2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos_theta)).detach().cpu().item())


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor).detach().cpu().item())


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _rate(values: list[float]) -> float:
    return _mean(values)


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_train_config(args.config)
    dataset = StepDataset(args.dataset_root)
    vision = _load_vision_module(args.checkpoint, args.config, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_position_errors: list[float] = []
    selected_orientation_errors: list[float] = []
    front_position_errors: list[float] = []
    front_orientation_errors: list[float] = []
    rear_position_errors: list[float] = []
    rear_orientation_errors: list[float] = []
    front_successes: list[float] = []
    rear_successes: list[float] = []
    selected_successes: list[float] = []
    sample_entries: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in _sample_indices(len(dataset), args.num_samples):
            sample = dataset[index]
            front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0).to(device)
            rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0).to(device)
            gt_position = torch.tensor(sample.core["gt_relative_position"], dtype=torch.float32, device=device)
            gt_orientation = torch.tensor(
                sample.core["gt_relative_orientation"],
                dtype=torch.float32,
                device=device,
            )
            target_class_ids = torch.tensor(
                [
                    target_segmentation_class_id(
                        sample.ref.observed_role,
                        label_mode=config.segmentation_label_mode,
                    )
                ],
                dtype=torch.long,
                device=device,
            )
            output = vision(front, rear, target_class_ids)

            front_success = float(output.front_candidate.pnp_success[0].detach().cpu().item())
            rear_success = float(output.rear_candidate.pnp_success[0].detach().cpu().item())
            selected_success = float(output.selected_candidate.pos_valid[0].detach().cpu().item())
            front_successes.append(front_success)
            rear_successes.append(rear_success)
            selected_successes.append(selected_success)

            selected_pose = output.selected_candidate.body_pose_9d[0]
            selected_position = selected_pose[:3]
            selected_orientation = selected_pose[3:]
            selected_position_error = None
            selected_orientation_error = None
            if selected_success > 0.5:
                selected_position_error = _norm(selected_position - gt_position)
                selected_orientation_error = _orientation_geodesic_degrees(
                    selected_orientation,
                    gt_orientation,
                )
                selected_position_errors.append(selected_position_error)
                selected_orientation_errors.append(selected_orientation_error)

            front_pose = output.front_candidate.body_pose_9d[0]
            if front_success > 0.5:
                front_position_errors.append(_norm(front_pose[:3] - gt_position))
                front_orientation_errors.append(
                    _orientation_geodesic_degrees(front_pose[3:], gt_orientation)
                )

            rear_pose = output.rear_candidate.body_pose_9d[0]
            if rear_success > 0.5:
                rear_position_errors.append(_norm(rear_pose[:3] - gt_position))
                rear_orientation_errors.append(
                    _orientation_geodesic_degrees(rear_pose[3:], gt_orientation)
                )

            sample_entries.append(
                {
                    "index": index,
                    "episode_id": sample.ref.episode_id,
                    "observed_role": sample.ref.observed_role,
                    "selected_view_index": int(output.selected_candidate.view_index[0].detach().cpu().item()),
                    "front_pnp_success": front_success,
                    "rear_pnp_success": rear_success,
                    "selected_pnp_success": selected_success,
                    "gt_relative_position": gt_position.detach().cpu().tolist(),
                    "gt_relative_orientation_6d": gt_orientation.detach().cpu().tolist(),
                    "selected_body_pose_9d": selected_pose.detach().cpu().tolist(),
                    "selected_position_error_norm": selected_position_error,
                    "selected_orientation_geodesic_deg": selected_orientation_error,
                    "front_body_pose_9d": front_pose.detach().cpu().tolist(),
                    "rear_body_pose_9d": rear_pose.detach().cpu().tolist(),
                    "front_reprojection_error": float(output.front_candidate.reprojection_error[0].detach().cpu().item()),
                    "rear_reprojection_error": float(output.rear_candidate.reprojection_error[0].detach().cpu().item()),
                    "front_v_sup": float(output.front_candidate.v_sup[0].detach().cpu().item()),
                    "rear_v_sup": float(output.rear_candidate.v_sup[0].detach().cpu().item()),
                    "front_v_rep": float(output.front_candidate.v_rep[0].detach().cpu().item()),
                    "rear_v_rep": float(output.rear_candidate.v_rep[0].detach().cpu().item()),
                }
            )

    sample_entries.sort(
        key=lambda entry: (
            entry["selected_position_error_norm"]
            if entry["selected_position_error_norm"] is not None
            else -1.0
        ),
        reverse=True,
    )
    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_samples": len(sample_entries),
        "front_pnp_success_rate": _rate(front_successes),
        "rear_pnp_success_rate": _rate(rear_successes),
        "selected_pnp_success_rate": _rate(selected_successes),
        "selected_valid_count": len(selected_position_errors),
        "front_valid_count": len(front_position_errors),
        "rear_valid_count": len(rear_position_errors),
        "mean_selected_position_error_norm": _mean(selected_position_errors),
        "mean_selected_orientation_geodesic_deg": _mean(selected_orientation_errors),
        "mean_front_position_error_norm": _mean(front_position_errors),
        "mean_front_orientation_geodesic_deg": _mean(front_orientation_errors),
        "mean_rear_position_error_norm": _mean(rear_position_errors),
        "mean_rear_orientation_geodesic_deg": _mean(rear_orientation_errors),
        "top_selected_failures": sample_entries[:8],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "samples.json").write_text(
        json.dumps(sample_entries, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
