from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dfb_state_estimation.datasets import (
    remap_segmentation_mask,
    segmentation_num_classes,
    SegmentationLabelMode,
    StepDataset,
    SyntheticSegmentationDataset,
    SyntheticSingleStepDataset,
    target_segmentation_class_id,
)
from dfb_state_estimation.losses import (
    VisionSupervisionTargets,
    compute_single_step_vision_loss,
)
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import _build_vision_module_for_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-validate the single-step vision module with random tensors."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--num-keypoints", type=int, default=9)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _rgba_image_to_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.to(dtype=torch.float32)
    else:
        tensor = torch.tensor(image, dtype=torch.float32)
    tensor = tensor[..., :3].permute(2, 0, 1)
    return tensor / 255.0


def _build_step_dataset(dataset_root: Path):
    if (dataset_root / "manifest.json").exists():
        payload = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
        dataset_format = str(payload.get("dataset_format", ""))
        if dataset_format == "synthetic_single_step_v1":
            return SyntheticSingleStepDataset(dataset_root)
        return SyntheticSegmentationDataset(dataset_root)
    return StepDataset(dataset_root)


MULTICLASS_SEGMENTATION_COLORS = {
    0: (0, 0, 0),
    1: (0, 220, 0),
    2: (0, 80, 255),
}

BINARY_SEGMENTATION_COLORS = {
    0: (0, 0, 0),
    1: (0, 80, 255),
}


def _segmentation_to_color(mask: np.ndarray, *, label_mode: SegmentationLabelMode) -> np.ndarray:
    colors = (
        BINARY_SEGMENTATION_COLORS
        if label_mode == "binary_target"
        else MULTICLASS_SEGMENTATION_COLORS
    )
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, class_color in colors.items():
        color[mask == class_id] = class_color
    return color


def _write_segmentation_images(
    output_dir: Path,
    prefix: str,
    rgba_image: list[list[list[int]]],
    mask: np.ndarray,
    *,
    label_mode: SegmentationLabelMode,
) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(rgba_image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    color = _segmentation_to_color(mask, label_mode=label_mode)
    overlay = cv2.addWeighted(bgr, 0.7, color, 0.3, 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{prefix}_segmentation_pred_color.png"), color)
    cv2.imwrite(str(output_dir / f"{prefix}_segmentation_pred_overlay.png"), overlay)
    return bgr, overlay


def _draw_sparse_voting_vectors(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    vectors: np.ndarray,
    *,
    keypoint_index: int,
    mask: np.ndarray | None = None,
    color: tuple[int, int, int] = (255, 255, 0),
    max_arrows: int = 64,
    arrow_scale: float = 14.0,
) -> np.ndarray:
    canvas = image_bgr.copy()
    if mask is None:
        valid = np.ones((pixels.shape[0],), dtype=bool)
    else:
        valid = mask.astype(bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return canvas
    stride = max(1, valid_indices.size // max_arrows)
    chosen = valid_indices[::stride][:max_arrows]
    for idx in chosen:
        x = int(pixels[idx, 0])
        y = int(pixels[idx, 1])
        dx = float(vectors[idx, keypoint_index, 0])
        dy = float(vectors[idx, keypoint_index, 1])
        end_x = int(round(x + dx * arrow_scale))
        end_y = int(round(y + dy * arrow_scale))
        cv2.arrowedLine(
            canvas,
            (x, y),
            (end_x, end_y),
            color,
            1,
            tipLength=0.25,
        )
    return canvas


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    if args.config is not None:
        train_config = load_train_config(args.config)
        label_mode: SegmentationLabelMode = train_config.segmentation_label_mode
        model = _build_vision_module_for_config(train_config)
        num_classes = segmentation_num_classes(label_mode)
    else:
        train_config = None
        label_mode = "multiclass_absolute"
        from dfb_state_estimation.models.vision import SingleStepVisionModule
        model = SingleStepVisionModule()
        num_classes = 3
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        module_states = payload.get("modules")
        if isinstance(module_states, dict) and "vision" in module_states:
            model.load_state_dict(module_states["vision"])
        else:
            model.load_state_dict(payload)
    model.train()

    if args.dataset_root is not None:
        step_dataset = _build_step_dataset(args.dataset_root)
        sample = step_dataset[args.step_index]
        front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0)
        rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0)
        target_class_id = target_segmentation_class_id(
            sample.ref.observed_role,
            label_mode=label_mode,
        )
        target_class_ids = torch.tensor(
            [target_class_id],
            dtype=torch.long,
        )
        args.batch_size = 1
        args.height = front.shape[-2]
        args.width = front.shape[-1]
        args.num_keypoints = len(sample.vision_labels["keypoints_2d_front"])
    else:
        front = torch.rand(args.batch_size, 3, args.height, args.width, dtype=torch.float32)
        rear = torch.rand(args.batch_size, 3, args.height, args.width, dtype=torch.float32)
        default_target_class_id = 1 if label_mode == "binary_target" else 2
        target_class_ids = torch.full(
            (args.batch_size,),
            default_target_class_id,
            dtype=torch.long,
        )

    output = model(front, rear, target_class_ids)
    print("front_segmentation_logits:", tuple(output.front_segmentation_logits.shape))
    print("rear_segmentation_logits:", tuple(output.rear_segmentation_logits.shape))
    print("front_keypoints_xy:", tuple(output.front_keypoints_xy.shape))
    print("rear_keypoints_xy:", tuple(output.rear_keypoints_xy.shape))
    print("front_keypoint_support:", tuple(output.front_keypoint_support.shape))
    print("rear_keypoint_support:", tuple(output.rear_keypoint_support.shape))
    print("selected_view_index:", output.selected_view_index.tolist())
    print("selected_view_onehot:", output.selected_view_onehot.tolist())
    print("selected_view_changed:", output.selected_view_changed.tolist())
    print("front_pred_target_area:", output.front_pred_target_area.tolist())
    print("rear_pred_target_area:", output.rear_pred_target_area.tolist())
    print("visual_embedding:", tuple(output.visual_embedding.shape))
    print("front_pnp_success:", output.front_pnp_success.tolist())
    print("rear_pnp_success:", output.rear_pnp_success.tolist())
    print("front_reprojection_error:", output.front_reprojection_error.tolist())
    print("rear_reprojection_error:", output.rear_reprojection_error.tolist())
    print("front_v_sup:", output.front_v_sup.tolist())
    print("rear_v_sup:", output.rear_v_sup.tolist())
    print("front_v_rep:", output.front_v_rep.tolist())
    print("rear_v_rep:", output.rear_v_rep.tolist())
    print("front_keypoint_support_mean:", output.front_keypoint_support.mean(dim=1).tolist())
    print("rear_keypoint_support_mean:", output.rear_keypoint_support.mean(dim=1).tolist())
    print(
        "raw_visual_evidence_strength:",
        output.raw_visual_evidence_strength.tolist(),
    )

    if args.dataset_root is not None:
        front_gt_target_area = sum(
            1
            for row in sample.vision_labels["segmentation_mask_front"]
            for value in row
            if int(value) == target_class_id
        )
        rear_gt_target_area = sum(
            1
            for row in sample.vision_labels["segmentation_mask_rear"]
            for value in row
            if int(value) == target_class_id
        )
        print("target_class_id:", target_class_id)
        print("front_gt_target_area:", front_gt_target_area)
        print("rear_gt_target_area:", rear_gt_target_area)
        front_keypoints_xy = (
            torch.tensor(sample.vision_labels["keypoints_2d_front"], dtype=torch.float32)
            .unsqueeze(0)
        )
        rear_keypoints_xy = (
            torch.tensor(sample.vision_labels["keypoints_2d_rear"], dtype=torch.float32)
            .unsqueeze(0)
        )
        scale = torch.tensor([args.width - 1, args.height - 1], dtype=torch.float32)
        front_keypoints_xy = front_keypoints_xy / scale
        rear_keypoints_xy = rear_keypoints_xy / scale
        targets = VisionSupervisionTargets(
            target_class_ids=target_class_ids,
            front_segmentation=torch.tensor(
                remap_segmentation_mask(
                    sample.vision_labels["segmentation_mask_front"],
                    observed_role=sample.ref.observed_role,
                    label_mode=label_mode,
                ),
                dtype=torch.long,
            ).unsqueeze(0),
            rear_segmentation=torch.tensor(
                remap_segmentation_mask(
                    sample.vision_labels["segmentation_mask_rear"],
                    observed_role=sample.ref.observed_role,
                    label_mode=label_mode,
                ),
                dtype=torch.long,
            ).unsqueeze(0),
            front_keypoints_xy=front_keypoints_xy,
            rear_keypoints_xy=rear_keypoints_xy,
            front_keypoint_xy_mask=torch.tensor(
                sample.vision_labels["keypoint_visibility_front"],
                dtype=torch.long,
            ).unsqueeze(0),
            rear_keypoint_xy_mask=torch.tensor(
                sample.vision_labels["keypoint_visibility_rear"],
                dtype=torch.long,
            ).unsqueeze(0),
            front_keypoint_voting_pixels=torch.tensor(
                sample.vision_labels["keypoint_voting_pixels_front"],
                dtype=torch.long,
            ).unsqueeze(0),
            rear_keypoint_voting_pixels=torch.tensor(
                sample.vision_labels["keypoint_voting_pixels_rear"],
                dtype=torch.long,
            ).unsqueeze(0),
            front_keypoint_voting_unit_vectors=torch.tensor(
                sample.vision_labels["keypoint_voting_unit_vectors_front"],
                dtype=torch.float32,
            ).unsqueeze(0),
            rear_keypoint_voting_unit_vectors=torch.tensor(
                sample.vision_labels["keypoint_voting_unit_vectors_rear"],
                dtype=torch.float32,
            ).unsqueeze(0),
            front_keypoint_voting_mask=torch.tensor(
                sample.vision_labels["keypoint_voting_mask_front"],
                dtype=torch.float32,
            ).unsqueeze(0),
            rear_keypoint_voting_mask=torch.tensor(
                sample.vision_labels["keypoint_voting_mask_rear"],
                dtype=torch.float32,
            ).unsqueeze(0),
        )
    else:
        sparse_points = 16
        targets = VisionSupervisionTargets(
            target_class_ids=target_class_ids,
            front_segmentation=torch.randint(
                low=0,
                high=num_classes,
                size=(args.batch_size, args.height, args.width),
            ),
            rear_segmentation=torch.randint(
                low=0,
                high=num_classes,
                size=(args.batch_size, args.height, args.width),
            ),
            front_keypoints_xy=torch.rand(args.batch_size, args.num_keypoints, 2),
            rear_keypoints_xy=torch.rand(args.batch_size, args.num_keypoints, 2),
            front_keypoint_xy_mask=torch.randint(
                low=0,
                high=2,
                size=(args.batch_size, args.num_keypoints),
            ),
            rear_keypoint_xy_mask=torch.randint(
                low=0,
                high=2,
                size=(args.batch_size, args.num_keypoints),
            ),
            front_keypoint_voting_pixels=torch.randint(
                low=0,
                high=min(args.width, args.height),
                size=(args.batch_size, sparse_points, 2),
            ),
            rear_keypoint_voting_pixels=torch.randint(
                low=0,
                high=min(args.width, args.height),
                size=(args.batch_size, sparse_points, 2),
            ),
            front_keypoint_voting_unit_vectors=torch.nn.functional.normalize(
                torch.rand(args.batch_size, sparse_points, args.num_keypoints, 2) * 2.0 - 1.0,
                dim=-1,
                eps=1.0e-6,
            ),
            rear_keypoint_voting_unit_vectors=torch.nn.functional.normalize(
                torch.rand(args.batch_size, sparse_points, args.num_keypoints, 2) * 2.0 - 1.0,
                dim=-1,
                eps=1.0e-6,
            ),
            front_keypoint_voting_mask=torch.ones(
                args.batch_size, sparse_points, dtype=torch.float32
            ),
            rear_keypoint_voting_mask=torch.ones(
                args.batch_size, sparse_points, dtype=torch.float32
            ),
        )

    losses = compute_single_step_vision_loss(output, targets)
    total = losses["total"]
    total.backward()
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")
    print("backward: ok")

    if args.dataset_root is not None and args.output_dir is not None:
        front_pred_mask = output.front_segmentation_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        rear_pred_mask = output.rear_segmentation_logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        _, front_seg_overlay = _write_segmentation_images(
            args.output_dir,
            "front",
            sample.core["front_camera_image"],
            front_pred_mask,
            label_mode=label_mode,
        )
        _, rear_seg_overlay = _write_segmentation_images(
            args.output_dir,
            "rear",
            sample.core["rear_camera_image"],
            rear_pred_mask,
            label_mode=label_mode,
        )
        point_labels = getattr(step_dataset, "point_labels", [f"k{i:02d}" for i in range(output.front_voting_field.shape[1])])
        target_only_front = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        target_only_rear = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        target_only_front[front_pred_mask == int(target_class_id)] = (0, 80, 255)
        target_only_rear[rear_pred_mask == int(target_class_id)] = (0, 80, 255)
        front_field = (
            torch.nn.functional.normalize(output.front_voting_field[0].detach().cpu(), dim=1, eps=1.0e-6)
            .permute(2, 3, 0, 1)
            .numpy()
        )
        rear_field = (
            torch.nn.functional.normalize(output.rear_voting_field[0].detach().cpu(), dim=1, eps=1.0e-6)
            .permute(2, 3, 0, 1)
            .numpy()
        )
        front_pixels = np.argwhere(front_pred_mask == int(target_class_id))
        rear_pixels = np.argwhere(rear_pred_mask == int(target_class_id))
        if front_pixels.size > 0:
            front_pixels_xy = np.stack([front_pixels[:, 1], front_pixels[:, 0]], axis=1)
            front_mask = np.ones((front_pixels_xy.shape[0],), dtype=np.uint8)
            for keypoint_index, label in enumerate(point_labels):
                cv2.imwrite(
                    str(args.output_dir / f"front_pred_voting_vectors_k{keypoint_index:02d}_{label}.png"),
                    _draw_sparse_voting_vectors(
                        target_only_front,
                        front_pixels_xy,
                        front_field[front_pixels[:, 0], front_pixels[:, 1]],
                        keypoint_index=keypoint_index,
                        mask=front_mask,
                    ),
                )
                cv2.imwrite(
                    str(args.output_dir / f"front_segmentation_and_voting_overlay_k{keypoint_index:02d}_{label}.png"),
                    _draw_sparse_voting_vectors(
                        front_seg_overlay,
                        front_pixels_xy,
                        front_field[front_pixels[:, 0], front_pixels[:, 1]],
                        keypoint_index=keypoint_index,
                        mask=front_mask,
                    ),
                )
        if rear_pixels.size > 0:
            rear_pixels_xy = np.stack([rear_pixels[:, 1], rear_pixels[:, 0]], axis=1)
            rear_mask = np.ones((rear_pixels_xy.shape[0],), dtype=np.uint8)
            for keypoint_index, label in enumerate(point_labels):
                cv2.imwrite(
                    str(args.output_dir / f"rear_pred_voting_vectors_k{keypoint_index:02d}_{label}.png"),
                    _draw_sparse_voting_vectors(
                        target_only_rear,
                        rear_pixels_xy,
                        rear_field[rear_pixels[:, 0], rear_pixels[:, 1]],
                        keypoint_index=keypoint_index,
                        mask=rear_mask,
                    ),
                )
                cv2.imwrite(
                    str(args.output_dir / f"rear_segmentation_and_voting_overlay_k{keypoint_index:02d}_{label}.png"),
                    _draw_sparse_voting_vectors(
                        rear_seg_overlay,
                        rear_pixels_xy,
                        rear_field[rear_pixels[:, 0], rear_pixels[:, 1]],
                        keypoint_index=keypoint_index,
                        mask=rear_mask,
                    ),
                )
        summary = {
            "label_mode": label_mode,
            "num_classes": num_classes,
            "selected_view_index": output.selected_view_index.tolist(),
            "front_pred_target_area": output.front_pred_target_area.tolist(),
            "rear_pred_target_area": output.rear_pred_target_area.tolist(),
            "front_v_sup": output.front_v_sup.tolist(),
            "rear_v_sup": output.rear_v_sup.tolist(),
            "front_v_rep": output.front_v_rep.tolist(),
            "rear_v_rep": output.rear_v_rep.tolist(),
        }
        (args.output_dir / "forward_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    geometry_validator = getattr(model, "geometry_validator", None)
    min_pnp_points = (
        geometry_validator.config.min_pnp_points if geometry_validator is not None else None
    )
    pnp_top_k_points = (
        geometry_validator.config.pnp_top_k_points if geometry_validator is not None else None
    )
    if args.dataset_root is not None:
        front_gt_visible_count = torch.tensor(
            [sum(sample.vision_labels["keypoint_visibility_front"])], dtype=torch.long
        )
        rear_gt_visible_count = torch.tensor(
            [sum(sample.vision_labels["keypoint_visibility_rear"])], dtype=torch.long
        )
    else:
        front_gt_visible_count = targets.front_keypoint_xy_mask.sum(dim=1)
        rear_gt_visible_count = targets.rear_keypoint_xy_mask.sum(dim=1)
    front_selected_count = (
        min(output.front_keypoint_support.shape[1], pnp_top_k_points)
        if pnp_top_k_points is not None
        else 0
    )
    rear_selected_count = (
        min(output.rear_keypoint_support.shape[1], pnp_top_k_points)
        if pnp_top_k_points is not None
        else 0
    )
    if front_selected_count > 0:
        front_selected_support = torch.topk(
            output.front_keypoint_support.detach(), k=front_selected_count, dim=1
        ).values
        rear_selected_support = torch.topk(
            output.rear_keypoint_support.detach(), k=rear_selected_count, dim=1
        ).values
    else:
        front_selected_support = torch.zeros(
            output.front_keypoint_support.shape[0], 1, dtype=output.front_keypoint_support.dtype
        )
        rear_selected_support = torch.zeros(
            output.rear_keypoint_support.shape[0], 1, dtype=output.rear_keypoint_support.dtype
        )
    print("front_gt_visible_count:", front_gt_visible_count.tolist())
    print("rear_gt_visible_count:", rear_gt_visible_count.tolist())
    if min_pnp_points is not None:
        print("front_gt_pnp_usable:", (front_gt_visible_count >= min_pnp_points).to(dtype=torch.float32).tolist())
        print("rear_gt_pnp_usable:", (rear_gt_visible_count >= min_pnp_points).to(dtype=torch.float32).tolist())
    print("front_selected_support_mean:", front_selected_support.mean(dim=1).tolist())
    print("rear_selected_support_mean:", rear_selected_support.mean(dim=1).tolist())
    print(
        "front_pred_pnp_usable:",
        torch.ones_like(output.front_pnp_success, dtype=torch.float32).tolist(),
    )
    print(
        "rear_pred_pnp_usable:",
        torch.ones_like(output.rear_pnp_success, dtype=torch.float32).tolist(),
    )
    def _gt_geometry_from_visible_mask(
        keypoints_xy: torch.Tensor,
        visible_mask: torch.Tensor,
        visible_count: torch.Tensor,
    ) -> tuple[list[float], list[float]]:
        pnp_success = torch.zeros_like(visible_count, dtype=torch.float32)
        reprojection_error = torch.full_like(visible_count, float("inf"), dtype=torch.float32)
        if geometry_validator is None or min_pnp_points is None:
            return pnp_success.tolist(), reprojection_error.tolist()
        usable = visible_count >= min_pnp_points
        if bool(usable.any()):
            geometry = model.geometry_validator.evaluate_batch(
                keypoints_xy[usable],
                visible_mask[usable].float(),
                image_height=args.height,
                image_width=args.width,
            )
            pnp_success[usable] = geometry.pnp_success
            reprojection_error[usable] = geometry.reprojection_error
        return pnp_success.tolist(), reprojection_error.tolist()

    gt_front_pnp_success, gt_front_reprojection_error = _gt_geometry_from_visible_mask(
        targets.front_keypoints_xy,
        targets.front_keypoint_xy_mask,
        front_gt_visible_count,
    )
    gt_rear_pnp_success, gt_rear_reprojection_error = _gt_geometry_from_visible_mask(
        targets.rear_keypoints_xy,
        targets.rear_keypoint_xy_mask,
        rear_gt_visible_count,
    )
    print("gt_front_pnp_success:", gt_front_pnp_success)
    print("gt_rear_pnp_success:", gt_rear_pnp_success)
    print("gt_front_reprojection_error:", gt_front_reprojection_error)
    print("gt_rear_reprojection_error:", gt_rear_reprojection_error)


if __name__ == "__main__":
    main()
