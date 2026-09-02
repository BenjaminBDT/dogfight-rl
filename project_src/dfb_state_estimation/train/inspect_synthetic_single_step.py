from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dfb_state_estimation.datasets import SyntheticSingleStepDataset
from dfb_state_estimation.models.vision.geometry import (
    GeometryValidationConfig,
    VisualGeometryValidator,
)

SEGMENTATION_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([40, 220, 40], dtype=np.uint8),
    2: np.array([40, 80, 255], dtype=np.uint8),
}
KEYPOINT_VISIBLE_COLOR = (60, 255, 60)
KEYPOINT_HIDDEN_COLOR = (60, 60, 255)
KEYPOINT_PROJECTABLE_COLOR = (0, 220, 255)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export audit visuals for synthetic single-step datasets."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated sample indices to export. Use 'all' for every sample.",
    )
    parser.add_argument(
        "--max-voting-arrows-per-keypoint",
        type=int,
        default=64,
        help="Maximum sparse voting pixels rendered per keypoint/view.",
    )
    return parser


def _segmentation_to_color(mask: list[list[int]]) -> np.ndarray:
    class_ids = np.asarray(mask, dtype=np.uint8)
    color = np.zeros((class_ids.shape[0], class_ids.shape[1], 3), dtype=np.uint8)
    for class_id, class_color in SEGMENTATION_COLORS.items():
        color[class_ids == class_id] = class_color
    return color


def _rgba_to_bgr(image: list[list[list[int]]]) -> np.ndarray:
    rgba = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _write_segmentation_artifacts(path_prefix: Path, image, mask) -> None:
    bgr = _rgba_to_bgr(image)
    color = _segmentation_to_color(mask)
    overlay = cv2.addWeighted(bgr, 0.7, color, 0.3, 0.0)
    _write_image(path_prefix.with_name(path_prefix.name + "_rgb.png"), bgr)
    _write_image(path_prefix.with_name(path_prefix.name + "_seg_color.png"), color)
    _write_image(path_prefix.with_name(path_prefix.name + "_seg_overlay.png"), overlay)


def _draw_keypoints(
    bgr: np.ndarray,
    keypoints_px: np.ndarray,
    visibility: np.ndarray,
    projectable: np.ndarray,
    labels: list[str],
    *,
    projected_px: np.ndarray | None = None,
) -> np.ndarray:
    canvas = bgr.copy()
    height, width = canvas.shape[:2]
    for index, (point, visible, can_project) in enumerate(
        zip(keypoints_px, visibility, projectable, strict=False)
    ):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        if int(visible) != 0:
            color = KEYPOINT_VISIBLE_COLOR
        elif int(can_project) != 0:
            color = KEYPOINT_PROJECTABLE_COLOR
        else:
            color = KEYPOINT_HIDDEN_COLOR
        cv2.circle(canvas, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 7, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{index}:{labels[index]}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
        if projected_px is not None:
            px = int(round(float(projected_px[index, 0])))
            py = int(round(float(projected_px[index, 1])))
            px = min(max(px, 0), width - 1)
            py = min(max(py, 0), height - 1)
            cv2.circle(canvas, (px, py), 3, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x, y), (px, py), (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return canvas


def _target_only_background(mask: list[list[int]], *, target_class_id: int) -> np.ndarray:
    class_ids = np.asarray(mask, dtype=np.uint8)
    canvas = np.zeros((class_ids.shape[0], class_ids.shape[1], 3), dtype=np.uint8)
    canvas[class_ids == target_class_id] = np.array([32, 32, 32], dtype=np.uint8)
    return canvas


def _draw_sparse_voting_pixels(bgr: np.ndarray, pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = bgr.copy()
    valid = pixels[mask != 0]
    for point in valid:
        x = int(point[0])
        y = int(point[1])
        cv2.circle(canvas, (x, y), 1, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def _draw_sparse_voting_vectors(
    bgr: np.ndarray,
    pixels: np.ndarray,
    vectors: np.ndarray,
    mask: np.ndarray,
    *,
    keypoint_index: int,
    max_arrows: int,
) -> np.ndarray:
    canvas = bgr.copy()
    valid_indices = np.flatnonzero(mask != 0)
    if valid_indices.size == 0:
        return canvas
    if valid_indices.size > max_arrows:
        sample = np.linspace(0, valid_indices.size - 1, num=max_arrows, dtype=np.int64)
        valid_indices = valid_indices[sample]
    for index in valid_indices:
        x = int(pixels[index, 0])
        y = int(pixels[index, 1])
        dx = float(vectors[index, keypoint_index, 0])
        dy = float(vectors[index, keypoint_index, 1])
        end_x = int(round(x + dx * 10.0))
        end_y = int(round(y + dy * 10.0))
        cv2.arrowedLine(
            canvas,
            (x, y),
            (end_x, end_y),
            (0, 255, 255),
            thickness=1,
            line_type=cv2.LINE_AA,
            tipLength=0.25,
        )
    return canvas


def _normalized_keypoints(keypoints_2d: list[list[float]], width: int, height: int) -> torch.Tensor:
    keypoints = torch.tensor(keypoints_2d, dtype=torch.float32).unsqueeze(0)
    scale = torch.tensor([max(width - 1, 1), max(height - 1, 1)], dtype=torch.float32)
    return keypoints / scale


def _camera_audit_summary(
    validator: VisualGeometryValidator,
    *,
    keypoints_2d: list[list[float]],
    visibility: list[int],
    width: int,
    height: int,
) -> tuple[dict, np.ndarray]:
    keypoints_xy = _normalized_keypoints(keypoints_2d, width, height)
    support = torch.tensor(visibility, dtype=torch.float32).unsqueeze(0)
    projection = validator.estimate_projection_batch(
        keypoints_xy,
        support,
        image_height=height,
        image_width=width,
    )
    projected = projection.projected_keypoints_px[0].detach().cpu().numpy()
    return (
        {
            "pnp_success": bool(float(projection.pnp_success[0].item()) != 0.0),
            "reprojection_error": float(projection.reprojection_error[0].item()),
            "visible_count": int(sum(int(v) for v in visibility)),
        },
        projected,
    )


def _export_sample(
    sample_dir: Path,
    dataset: SyntheticSingleStepDataset,
    sample_index: int,
    *,
    max_arrows: int,
    validator: VisualGeometryValidator,
) -> dict:
    sample = dataset[sample_index]
    sample_dir.mkdir(parents=True, exist_ok=True)

    width = dataset.width
    height = dataset.height
    observed_role = sample.ref.observed_role
    target_role = "fighter2" if observed_role == "fighter1" else "fighter1"
    target_class_id = 2 if target_role == "fighter2" else 1

    front_summary, front_projected = _camera_audit_summary(
        validator,
        keypoints_2d=sample.vision_labels["keypoints_2d_front"],
        visibility=sample.vision_labels["keypoint_visibility_front"],
        width=width,
        height=height,
    )
    rear_summary, rear_projected = _camera_audit_summary(
        validator,
        keypoints_2d=sample.vision_labels["keypoints_2d_rear"],
        visibility=sample.vision_labels["keypoint_visibility_rear"],
        width=width,
        height=height,
    )

    for view_name, image_key, seg_key, kp_key, vis_key, voting_pixels_key, voting_vectors_key, voting_mask_key, projected in (
        (
            "front",
            "front_camera_image",
            "segmentation_mask_front",
            "keypoints_2d_front",
            "keypoint_visibility_front",
            "keypoint_voting_pixels_front",
            "keypoint_voting_unit_vectors_front",
            "keypoint_voting_mask_front",
            front_projected,
        ),
        (
            "rear",
            "rear_camera_image",
            "segmentation_mask_rear",
            "keypoints_2d_rear",
            "keypoint_visibility_rear",
            "keypoint_voting_pixels_rear",
            "keypoint_voting_unit_vectors_rear",
            "keypoint_voting_mask_rear",
            rear_projected,
        ),
    ):
        image = sample.core[image_key]
        mask = sample.vision_labels[seg_key]
        _write_segmentation_artifacts(sample_dir / view_name, image, mask)

        bgr = _rgba_to_bgr(image)
        target_only_bgr = _target_only_background(mask, target_class_id=target_class_id)
        keypoints = np.asarray(sample.vision_labels[kp_key], dtype=np.float32)
        visibility = np.asarray(sample.vision_labels[vis_key], dtype=np.int64)
        projectable = np.asarray(
            sample.vision_labels[vis_key.replace("visibility", "projectable")],
            dtype=np.int64,
        )
        _write_image(
            sample_dir / f"{view_name}_keypoints_overlay.png",
            _draw_keypoints(bgr, keypoints, visibility, projectable, dataset.point_labels),
        )
        _write_image(
            sample_dir / f"{view_name}_keypoints_target_only_overlay.png",
            _draw_keypoints(target_only_bgr, keypoints, visibility, projectable, dataset.point_labels),
        )
        _write_image(
            sample_dir / f"{view_name}_pnp_projection_overlay.png",
            _draw_keypoints(
                bgr,
                keypoints,
                visibility,
                projectable,
                dataset.point_labels,
                projected_px=projected,
            ),
        )
        _write_image(
            sample_dir / f"{view_name}_pnp_projection_target_only_overlay.png",
            _draw_keypoints(
                target_only_bgr,
                keypoints,
                visibility,
                projectable,
                dataset.point_labels,
                projected_px=projected,
            ),
        )

        pixels = np.asarray(sample.vision_labels[voting_pixels_key], dtype=np.int64)
        vectors = np.asarray(sample.vision_labels[voting_vectors_key], dtype=np.float32)
        voting_mask = np.asarray(sample.vision_labels[voting_mask_key], dtype=np.int64)
        _write_image(
            sample_dir / f"{view_name}_voting_pixels_overlay.png",
            _draw_sparse_voting_pixels(bgr, pixels, voting_mask),
        )
        _write_image(
            sample_dir / f"{view_name}_voting_pixels_target_only_overlay.png",
            _draw_sparse_voting_pixels(target_only_bgr, pixels, voting_mask),
        )
        for keypoint_index, label in enumerate(dataset.point_labels):
            _write_image(
                sample_dir / f"{view_name}_voting_vectors_k{keypoint_index:02d}_{label}.png",
                _draw_sparse_voting_vectors(
                    bgr,
                    pixels,
                    vectors,
                    voting_mask,
                    keypoint_index=keypoint_index,
                    max_arrows=max_arrows,
                ),
            )
            _write_image(
                sample_dir
                / f"{view_name}_voting_vectors_target_only_k{keypoint_index:02d}_{label}.png",
                _draw_sparse_voting_vectors(
                    target_only_bgr,
                    pixels,
                    vectors,
                    voting_mask,
                    keypoint_index=keypoint_index,
                    max_arrows=max_arrows,
                ),
            )

    sample_summary = {
        "sample_index": sample_index,
        "observed_role": observed_role,
        "target_role": target_role,
        "front": front_summary,
        "rear": rear_summary,
        "front_projectable_count": int(sum(sample.vision_labels["keypoint_projectable_front"])),
        "rear_projectable_count": int(sum(sample.vision_labels["keypoint_projectable_rear"])),
        "front_voting_pixel_count": int(sum(sample.vision_labels["keypoint_voting_mask_front"])),
        "rear_voting_pixel_count": int(sum(sample.vision_labels["keypoint_voting_mask_rear"])),
        "front_target_area": int(
            sum(
                1
                for row in sample.vision_labels["segmentation_mask_front"]
                for value in row
                if int(value) == target_class_id
            )
        ),
        "rear_target_area": int(
            sum(
                1
                for row in sample.vision_labels["segmentation_mask_rear"]
                for value in row
                if int(value) == target_class_id
            )
        ),
    }
    _write_json(sample_dir / "summary.json", sample_summary)
    return sample_summary


def main() -> None:
    args = _build_parser().parse_args()
    dataset = SyntheticSingleStepDataset(args.dataset_root)
    validator = VisualGeometryValidator(GeometryValidationConfig())

    if args.sample_indices.strip().lower() == "all":
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = [int(value) for value in args.sample_indices.split(",") if value.strip()]

    aggregate = {
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_count": len(sample_indices),
        "samples": [],
    }
    for sample_index in sample_indices:
        sample_output_dir = args.output_dir / f"sample_{sample_index:06d}"
        aggregate["samples"].append(
            _export_sample(
                sample_output_dir,
                dataset,
                sample_index,
                max_arrows=args.max_voting_arrows_per_keypoint,
                validator=validator,
            )
        )
    _write_json(args.output_dir / "summary.json", aggregate)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
