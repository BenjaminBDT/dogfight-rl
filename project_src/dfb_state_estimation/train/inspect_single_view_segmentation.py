from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dfb_state_estimation.datasets import (
    SyntheticSingleViewSegmentationDataset,
    remap_segmentation_mask,
    SegmentationLabelMode,
)
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import (
    _build_vision_module_for_config,
    _collate_single_view_segmentation_batch,
)


BINARY_SEGMENTATION_COLORS = {
    0: (0, 0, 0),
    1: (0, 80, 255),
}

MULTICLASS_SEGMENTATION_COLORS = {
    0: (0, 0, 0),
    1: (0, 220, 0),
    2: (0, 80, 255),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect true single-view segmentation with raw/patched/pred outputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


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


def _write_mask_bundle(
    *,
    output_dir: Path,
    prefix: str,
    bgr_image: np.ndarray,
    mask: np.ndarray,
    label_mode: SegmentationLabelMode,
) -> None:
    color = _segmentation_to_color(mask, label_mode=label_mode)
    overlay = cv2.addWeighted(bgr_image, 0.7, color, 0.3, 0.0)
    cv2.imwrite(str(output_dir / f"{prefix}_color.png"), color)
    cv2.imwrite(str(output_dir / f"{prefix}_overlay.png"), overlay)


def _tensor_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    chw = image.detach().cpu().clamp(0.0, 1.0).numpy()
    hwc = np.transpose(chw, (1, 2, 0))
    rgb = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_train_config(args.config)
    dataset = SyntheticSingleViewSegmentationDataset(args.dataset_root)
    sample = dataset[args.sample_index]

    model = _build_vision_module_for_config(config).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module_states = payload.get("modules")
    if not isinstance(module_states, dict) or "vision" not in module_states:
        raise ValueError("checkpoint does not contain modules['vision']")
    model.load_state_dict(module_states["vision"])

    batch = _collate_single_view_segmentation_batch(
        [sample],
        torch.device("cpu"),
        segmentation_label_mode=config.segmentation_label_mode,
        segmentation_roi_crop_enabled=config.segmentation_roi_crop.enabled,
        segmentation_roi_crop_context_scale=config.segmentation_roi_crop.context_scale,
        segmentation_roi_crop_min_crop_size=config.segmentation_roi_crop.min_crop_size,
        segmentation_roi_crop_square=config.segmentation_roi_crop.square,
        segmentation_target_patch_enabled=config.segmentation_target_patch.enabled,
        segmentation_target_patch_size=config.segmentation_target_patch.patch_size,
    )

    with torch.no_grad():
        output = model(batch["image"], batch["target_class_ids"])
        pred_mask = output.segmentation_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target_prob = torch.softmax(output.segmentation_logits, dim=1)[0, 1].cpu().numpy()
    target_class_id = int(batch["target_class_ids"][0].item())

    raw_mask = np.asarray(
        remap_segmentation_mask(
            sample.segmentation_mask,
            observed_role=sample.observed_role,
            label_mode=config.segmentation_label_mode,
        ),
        dtype=np.uint8,
    )
    patched_mask = batch["segmentation_targets"].segmentation[0].cpu().numpy().astype(np.uint8)

    raw_rgba = np.asarray(sample.image, dtype=np.uint8)
    raw_bgr = cv2.cvtColor(raw_rgba, cv2.COLOR_RGBA2BGR)
    patched_bgr = _tensor_image_to_bgr(batch["image"][0])
    prob_u8 = np.clip(target_prob * 255.0, 0, 255).astype(np.uint8)
    prob_heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_TURBO)

    cv2.imwrite(str(output_dir / "raw_image.png"), raw_bgr)
    cv2.imwrite(str(output_dir / "patched_image.png"), patched_bgr)
    cv2.imwrite(str(output_dir / "target_prob_heatmap.png"), prob_heatmap)

    _write_mask_bundle(
        output_dir=output_dir,
        prefix="raw_gt",
        bgr_image=raw_bgr,
        mask=raw_mask,
        label_mode=config.segmentation_label_mode,
    )
    _write_mask_bundle(
        output_dir=output_dir,
        prefix="patched_gt",
        bgr_image=patched_bgr,
        mask=patched_mask,
        label_mode=config.segmentation_label_mode,
    )
    _write_mask_bundle(
        output_dir=output_dir,
        prefix="pred",
        bgr_image=patched_bgr,
        mask=pred_mask,
        label_mode=config.segmentation_label_mode,
    )

    summary = {
        "sample_index": int(args.sample_index),
        "sample_dir": sample.sample_dir,
        "active_view": sample.active_view,
        "active_target_area_metadata": int(sample.active_target_area),
        "checkpoint": str(args.checkpoint),
        "label_mode": config.segmentation_label_mode,
        "target_class_id": target_class_id,
        "raw_gt_target_area": int((raw_mask == target_class_id).sum()),
        "patched_gt_target_area": int((patched_mask == target_class_id).sum()),
        "pred_target_area": int((pred_mask == target_class_id).sum()),
        "target_prob_mean": float(target_prob.mean()),
        "target_prob_on_raw_gt_mean": float(target_prob[raw_mask == target_class_id].mean()) if (raw_mask == target_class_id).any() else 0.0,
        "target_prob_on_patched_gt_mean": float(target_prob[patched_mask == target_class_id].mean()) if (patched_mask == target_class_id).any() else 0.0,
        "target_prob_on_patched_bg_mean": float(target_prob[patched_mask != target_class_id].mean()) if (patched_mask != target_class_id).any() else 0.0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
