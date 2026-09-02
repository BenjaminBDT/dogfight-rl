from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dfb_state_estimation.datasets import StepDataset, target_segmentation_class_id
from dfb_state_estimation.models.vision import SingleStepVisionModule
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import _build_vision_module_for_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose per-keypoint visual quality against GT keypoints."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hard-error-threshold-px", type=float, default=16.0)
    parser.add_argument(
        "--hard-sample-quantile",
        type=float,
        default=0.75,
        help="Samples above this selected position-error quantile are treated as hard cases.",
    )
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


def _load_keypoint_labels(schema_path: str) -> list[str]:
    payload = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    labels = payload.get("point_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"invalid keypoint schema labels in {schema_path}")
    return [str(label) for label in labels]


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


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * float(len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    t = pos - float(lo)
    return ordered[lo] * (1.0 - t) + ordered[hi] * t


def _norm_per_keypoint(pred_px: torch.Tensor, gt_px: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred_px - gt_px, dim=-1)


def _new_stat_bucket() -> dict[str, list[float] | float]:
    return {
        "selected_visible_errors_px": [],
        "selected_projectable_errors_px": [],
        "selected_visible_support": [],
        "selected_projectable_support": [],
        "front_visible_errors_px": [],
        "rear_visible_errors_px": [],
        "front_projectable_errors_px": [],
        "rear_projectable_errors_px": [],
        "front_visible_support": [],
        "rear_visible_support": [],
        "front_projectable_support": [],
        "rear_projectable_support": [],
        "selected_visible_count": 0.0,
        "selected_projectable_count": 0.0,
        "front_visible_count": 0.0,
        "rear_visible_count": 0.0,
        "front_projectable_count": 0.0,
        "rear_projectable_count": 0.0,
        "selected_hard_visible_error_count": 0.0,
    }


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_train_config(args.config)
    dataset = StepDataset(args.dataset_root)
    vision = _load_vision_module(args.checkpoint, args.config, device)
    keypoint_labels = _load_keypoint_labels(vision.config.geometry.keypoint_schema_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_keypoints = len(keypoint_labels)
    per_keypoint = [_new_stat_bucket() for _ in range(num_keypoints)]
    sample_entries: list[dict[str, Any]] = []
    selected_position_errors: list[float] = []

    with torch.no_grad():
        for index in _sample_indices(len(dataset), args.num_samples):
            sample = dataset[index]
            front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0).to(device)
            rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0).to(device)
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

            height = front.shape[-2]
            width = front.shape[-1]
            scale = torch.tensor([width, height], dtype=torch.float32, device=device)

            front_pred_px = output.front_keypoints_xy[0] * scale
            rear_pred_px = output.rear_keypoints_xy[0] * scale
            front_gt_px = torch.tensor(sample.vision_labels["keypoints_2d_front"], dtype=torch.float32, device=device)
            rear_gt_px = torch.tensor(sample.vision_labels["keypoints_2d_rear"], dtype=torch.float32, device=device)
            front_visible = torch.tensor(sample.vision_labels["keypoint_visibility_front"], dtype=torch.bool, device=device)
            rear_visible = torch.tensor(sample.vision_labels["keypoint_visibility_rear"], dtype=torch.bool, device=device)
            front_projectable = torch.tensor(sample.vision_labels["keypoint_projectable_front"], dtype=torch.bool, device=device)
            rear_projectable = torch.tensor(sample.vision_labels["keypoint_projectable_rear"], dtype=torch.bool, device=device)

            front_support = output.front_keypoint_support[0].detach().cpu()
            rear_support = output.rear_keypoint_support[0].detach().cpu()
            front_errors = _norm_per_keypoint(front_pred_px, front_gt_px).detach().cpu()
            rear_errors = _norm_per_keypoint(rear_pred_px, rear_gt_px).detach().cpu()

            selected_view_index = int(output.selected_view_index[0].detach().cpu().item())
            selected_position_error = None
            if float(output.selected_candidate.pos_valid[0].detach().cpu().item()) > 0.5:
                gt_position = torch.tensor(sample.core["gt_relative_position"], dtype=torch.float32, device=device)
                selected_position = output.selected_candidate.body_pose_9d[0, :3]
                selected_position_error = float(
                    torch.linalg.vector_norm(selected_position - gt_position).detach().cpu().item()
                )
                selected_position_errors.append(selected_position_error)

            if selected_view_index == 0:
                selected_errors = front_errors
                selected_support = front_support
                selected_visible = front_visible.detach().cpu()
                selected_projectable = front_projectable.detach().cpu()
            elif selected_view_index == 1:
                selected_errors = rear_errors
                selected_support = rear_support
                selected_visible = rear_visible.detach().cpu()
                selected_projectable = rear_projectable.detach().cpu()
            else:
                selected_errors = torch.zeros_like(front_errors)
                selected_support = torch.zeros_like(front_support)
                selected_visible = torch.zeros_like(front_visible.detach().cpu())
                selected_projectable = torch.zeros_like(front_projectable.detach().cpu())

            sample_entries.append(
                {
                    "index": index,
                    "episode_id": sample.ref.episode_id,
                    "observed_role": sample.ref.observed_role,
                    "selected_view_index": selected_view_index,
                    "selected_position_error_norm": selected_position_error,
                    "selected_visible_errors_px": selected_errors.tolist(),
                    "selected_support": selected_support.tolist(),
                    "selected_visible_mask": selected_visible.tolist(),
                    "selected_projectable_mask": selected_projectable.tolist(),
                }
            )

            front_visible_cpu = front_visible.detach().cpu()
            rear_visible_cpu = rear_visible.detach().cpu()
            front_projectable_cpu = front_projectable.detach().cpu()
            rear_projectable_cpu = rear_projectable.detach().cpu()

            for keypoint_index in range(num_keypoints):
                bucket = per_keypoint[keypoint_index]
                fe = float(front_errors[keypoint_index].item())
                re = float(rear_errors[keypoint_index].item())
                fs = float(front_support[keypoint_index].item())
                rs = float(rear_support[keypoint_index].item())

                if bool(front_visible_cpu[keypoint_index].item()):
                    bucket["front_visible_errors_px"].append(fe)
                    bucket["front_visible_support"].append(fs)
                    bucket["front_visible_count"] += 1.0
                if bool(rear_visible_cpu[keypoint_index].item()):
                    bucket["rear_visible_errors_px"].append(re)
                    bucket["rear_visible_support"].append(rs)
                    bucket["rear_visible_count"] += 1.0
                if bool(front_projectable_cpu[keypoint_index].item()):
                    bucket["front_projectable_errors_px"].append(fe)
                    bucket["front_projectable_support"].append(fs)
                    bucket["front_projectable_count"] += 1.0
                if bool(rear_projectable_cpu[keypoint_index].item()):
                    bucket["rear_projectable_errors_px"].append(re)
                    bucket["rear_projectable_support"].append(rs)
                    bucket["rear_projectable_count"] += 1.0

                se = float(selected_errors[keypoint_index].item())
                ss = float(selected_support[keypoint_index].item())
                if bool(selected_visible[keypoint_index].item()):
                    bucket["selected_visible_errors_px"].append(se)
                    bucket["selected_visible_support"].append(ss)
                    bucket["selected_visible_count"] += 1.0
                if bool(selected_projectable[keypoint_index].item()):
                    bucket["selected_projectable_errors_px"].append(se)
                    bucket["selected_projectable_support"].append(ss)
                    bucket["selected_projectable_count"] += 1.0

    hard_threshold = _quantile(
        [value for value in selected_position_errors if value is not None],
        args.hard_sample_quantile,
    )

    for entry in sample_entries:
        position_error = entry["selected_position_error_norm"]
        if position_error is None or position_error < hard_threshold:
            continue
        visible_mask = entry["selected_visible_mask"]
        projectable_mask = entry["selected_projectable_mask"]
        errors = entry["selected_visible_errors_px"]
        supports = entry["selected_support"]
        for keypoint_index in range(num_keypoints):
            if bool(visible_mask[keypoint_index]):
                bucket = per_keypoint[keypoint_index]
                bucket["selected_hard_visible_error_count"] += float(
                    errors[keypoint_index] >= args.hard_error_threshold_px
                )
                bucket.setdefault("selected_hard_visible_errors_px", []).append(errors[keypoint_index])
                bucket.setdefault("selected_hard_visible_support", []).append(supports[keypoint_index])
            elif bool(projectable_mask[keypoint_index]):
                bucket = per_keypoint[keypoint_index]
                bucket.setdefault("selected_hard_projectable_errors_px", []).append(errors[keypoint_index])
                bucket.setdefault("selected_hard_projectable_support", []).append(supports[keypoint_index])

    per_keypoint_summary: list[dict[str, Any]] = []
    for keypoint_index, label in enumerate(keypoint_labels):
        bucket = per_keypoint[keypoint_index]
        selected_visible_count = float(bucket["selected_visible_count"])
        selected_projectable_count = float(bucket["selected_projectable_count"])
        entry = {
            "keypoint_index": keypoint_index,
            "keypoint_label": label,
            "selected_visible_rate": selected_visible_count / float(len(sample_entries)) if sample_entries else 0.0,
            "selected_projectable_rate": selected_projectable_count / float(len(sample_entries)) if sample_entries else 0.0,
            "mean_selected_visible_error_px": _mean(bucket["selected_visible_errors_px"]),
            "mean_selected_projectable_error_px": _mean(bucket["selected_projectable_errors_px"]),
            "mean_selected_visible_support": _mean(bucket["selected_visible_support"]),
            "mean_selected_projectable_support": _mean(bucket["selected_projectable_support"]),
            "selected_visible_hard_error_rate": (
                float(bucket["selected_hard_visible_error_count"]) / selected_visible_count
                if selected_visible_count > 0.0
                else 0.0
            ),
            "mean_selected_hard_visible_error_px": _mean(bucket.get("selected_hard_visible_errors_px", [])),
            "mean_selected_hard_visible_support": _mean(bucket.get("selected_hard_visible_support", [])),
            "mean_front_visible_error_px": _mean(bucket["front_visible_errors_px"]),
            "mean_rear_visible_error_px": _mean(bucket["rear_visible_errors_px"]),
            "mean_front_projectable_error_px": _mean(bucket["front_projectable_errors_px"]),
            "mean_rear_projectable_error_px": _mean(bucket["rear_projectable_errors_px"]),
            "mean_front_visible_support": _mean(bucket["front_visible_support"]),
            "mean_rear_visible_support": _mean(bucket["rear_visible_support"]),
        }
        per_keypoint_summary.append(entry)

    per_keypoint_summary.sort(
        key=lambda entry: (
            entry["mean_selected_visible_error_px"],
            entry["selected_visible_hard_error_rate"],
        ),
        reverse=True,
    )

    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_samples": len(sample_entries),
        "hard_error_threshold_px": float(args.hard_error_threshold_px),
        "hard_sample_quantile": float(args.hard_sample_quantile),
        "hard_sample_position_error_threshold": hard_threshold,
        "per_keypoint_selected": per_keypoint_summary,
        "top_keypoints_by_selected_visible_error_px": per_keypoint_summary[:5],
        "top_keypoints_by_selected_visible_hard_error_rate": sorted(
            per_keypoint_summary,
            key=lambda entry: entry["selected_visible_hard_error_rate"],
            reverse=True,
        )[:5],
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
