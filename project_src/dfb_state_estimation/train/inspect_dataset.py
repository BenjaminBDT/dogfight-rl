from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import cv2
import numpy as np

from dfb_state_estimation.datasets import DatasetContract
from dfb_state_estimation.datasets import StepDataset
from dfb_state_estimation.datasets import WindowDataset
from dfb_state_estimation.datasets import load_dataset_contract

SEGMENTATION_COLORS = {
    0: np.array([0, 0, 0], dtype=np.uint8),
    1: np.array([40, 220, 40], dtype=np.uint8),
    2: np.array([40, 80, 255], dtype=np.uint8),
}
KEYPOINT_VISIBLE_COLOR = (60, 255, 60)
KEYPOINT_HIDDEN_COLOR = (60, 60, 255)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect DFB state estimation dataset views.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to packed dataset root containing meta.json and schema.json.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index to inspect.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Window index to inspect.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Window length used for inspection.",
    )
    parser.add_argument(
        "--max-stride-steps",
        type=int,
        default=1,
        help="Maximum backward stride for sampled window inspection.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Deterministic seed for non-uniform sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory for exporting inspected step assets.",
    )
    return parser


def _print_contract(contract: DatasetContract) -> None:
    print("dataset_root:", contract.dataset_root)
    print("schema_id:", contract.schema.schema_id)
    print("schema_version:", contract.schema.schema_version)
    print("dataset_id:", contract.meta.dataset_id)
    print("dataset_version:", contract.meta.dataset_version)
    print("episodes:", len(contract.meta.episodes))
    print("chunks:", len(contract.meta.chunks))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_rgba_png(path: Path, image: list[list[list[int]]]) -> None:
    rgba = np.asarray(image, dtype=np.uint8)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgra)


def _write_gray_png(path: Path, image: list[list[int]]) -> None:
    array = np.asarray(image, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


def _segmentation_to_color(mask: list[list[int]]) -> np.ndarray:
    class_ids = np.asarray(mask, dtype=np.uint8)
    color = np.zeros((class_ids.shape[0], class_ids.shape[1], 3), dtype=np.uint8)
    for class_id, class_color in SEGMENTATION_COLORS.items():
        color[class_ids == class_id] = class_color
    return color


def _write_segmentation_color_png(path: Path, mask: list[list[int]]) -> None:
    color = _segmentation_to_color(mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color)


def _write_segmentation_overlay(path: Path, image: list[list[list[int]]], mask: list[list[int]]) -> None:
    rgba = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    color = _segmentation_to_color(mask)
    overlay = cv2.addWeighted(bgr, 0.7, color, 0.3, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def _write_keypoint_overlay(
    path: Path,
    image: list[list[list[int]]],
    keypoints_2d: list[list[float]],
    visibility: list[int],
) -> None:
    rgba = np.asarray(image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    height, width = bgr.shape[:2]
    for index, (point, visible) in enumerate(zip(keypoints_2d, visibility, strict=False)):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        color = KEYPOINT_VISIBLE_COLOR if int(visible) != 0 else KEYPOINT_HIDDEN_COLOR
        cv2.circle(bgr, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, (x, y), 7, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def _write_binaural_wav(path: Path, audio_window: list[list[float]]) -> None:
    array = np.asarray(audio_window, dtype=np.float32)
    clipped = np.clip(array, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        wav_file.writeframes(pcm16.tobytes())


def _infer_shape(value):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _infer_shape(value[0])


def _step_summary(step_sample) -> dict:
    front_visibility = [int(value) for value in step_sample.vision_labels["keypoint_visibility_front"]]
    rear_visibility = [int(value) for value in step_sample.vision_labels["keypoint_visibility_rear"]]
    return {
        "ref": {
            "episode_id": step_sample.ref.episode_id,
            "chunk_id": step_sample.ref.chunk_id,
            "chunk_step_offset": step_sample.ref.chunk_step_offset,
            "chunk_index": step_sample.ref.chunk_index,
            "global_model_step_index": step_sample.ref.global_model_step_index,
            "simulation_step_index": step_sample.ref.simulation_step_index,
        },
        "core": {
            "audio_window_shape": [
                len(step_sample.core["audio_window_binaural"]),
                len(step_sample.core["audio_window_binaural"][0]),
            ],
            "gt_relative_position": step_sample.core["gt_relative_position"],
            "gt_relative_orientation": step_sample.core["gt_relative_orientation"],
            "gt_linear_velocity": step_sample.core["gt_linear_velocity"],
            "gt_angular_velocity": step_sample.core["gt_angular_velocity"],
            "ego_position_world": step_sample.core["ego_position_world"],
            "ego_orientation_world": step_sample.core["ego_orientation_world"],
            "ego_linear_velocity_world": step_sample.core["ego_linear_velocity_world"],
            "ego_angular_velocity_body": step_sample.core["ego_angular_velocity_body"],
        },
        "vision_labels": {
            "front_segmentation_classes": sorted(
                {int(value) for row in step_sample.vision_labels["segmentation_mask_front"] for value in row}
            ),
            "rear_segmentation_classes": sorted(
                {int(value) for row in step_sample.vision_labels["segmentation_mask_rear"] for value in row}
            ),
            "front_keypoints_2d": step_sample.vision_labels["keypoints_2d_front"],
            "rear_keypoints_2d": step_sample.vision_labels["keypoints_2d_rear"],
            "front_keypoint_visibility": front_visibility,
            "rear_keypoint_visibility": rear_visibility,
            "front_keypoint_visible_count": int(sum(front_visibility)),
            "rear_keypoint_visible_count": int(sum(rear_visibility)),
            "front_keypoint_voting_pixels_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_pixels_front"]
            )
            if "keypoint_voting_pixels_front" in step_sample.vision_labels
            else None,
            "rear_keypoint_voting_pixels_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_pixels_rear"]
            )
            if "keypoint_voting_pixels_rear" in step_sample.vision_labels
            else None,
            "front_keypoint_voting_unit_vectors_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_unit_vectors_front"]
            )
            if "keypoint_voting_unit_vectors_front" in step_sample.vision_labels
            else None,
            "rear_keypoint_voting_unit_vectors_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_unit_vectors_rear"]
            )
            if "keypoint_voting_unit_vectors_rear" in step_sample.vision_labels
            else None,
            "front_keypoint_voting_mask_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_mask_front"]
            )
            if "keypoint_voting_mask_front" in step_sample.vision_labels
            else None,
            "rear_keypoint_voting_mask_shape": _infer_shape(
                step_sample.vision_labels["keypoint_voting_mask_rear"]
            )
            if "keypoint_voting_mask_rear" in step_sample.vision_labels
            else None,
        },
        "audio_features": {
            "binaural_energy_t": step_sample.audio_features["binaural_energy_t"],
            "binaural_cue_vector_t": step_sample.audio_features["binaural_cue_vector_t"],
        },
        "rule_targets": {
            "gt_doa_unit_vector_body": step_sample.rule_targets["gt_doa_unit_vector_body"],
            "gt_log_distance_scalar": step_sample.rule_targets["gt_log_distance_scalar"],
            "target_pos_conf": step_sample.rule_targets["target_pos_conf"],
            "target_ori_conf": step_sample.rule_targets["target_ori_conf"],
        },
    }


def _export_step(output_dir: Path, step_index: int, step_sample) -> None:
    step_dir = output_dir / f"step_{step_index:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_rgba_png(step_dir / "front_camera_image.png", step_sample.core["front_camera_image"])
    _write_rgba_png(step_dir / "rear_camera_image.png", step_sample.core["rear_camera_image"])
    _write_gray_png(step_dir / "front_segmentation_mask.png", step_sample.vision_labels["segmentation_mask_front"])
    _write_gray_png(step_dir / "rear_segmentation_mask.png", step_sample.vision_labels["segmentation_mask_rear"])
    _write_segmentation_color_png(
        step_dir / "front_segmentation_color.png",
        step_sample.vision_labels["segmentation_mask_front"],
    )
    _write_segmentation_color_png(
        step_dir / "rear_segmentation_color.png",
        step_sample.vision_labels["segmentation_mask_rear"],
    )
    _write_segmentation_overlay(
        step_dir / "front_segmentation_overlay.png",
        step_sample.core["front_camera_image"],
        step_sample.vision_labels["segmentation_mask_front"],
    )
    _write_segmentation_overlay(
        step_dir / "rear_segmentation_overlay.png",
        step_sample.core["rear_camera_image"],
        step_sample.vision_labels["segmentation_mask_rear"],
    )
    _write_keypoint_overlay(
        step_dir / "front_keypoints_overlay.png",
        step_sample.core["front_camera_image"],
        step_sample.vision_labels["keypoints_2d_front"],
        step_sample.vision_labels["keypoint_visibility_front"],
    )
    _write_keypoint_overlay(
        step_dir / "rear_keypoints_overlay.png",
        step_sample.core["rear_camera_image"],
        step_sample.vision_labels["keypoints_2d_rear"],
        step_sample.vision_labels["keypoint_visibility_rear"],
    )
    _write_binaural_wav(step_dir / "audio_window_binaural.wav", step_sample.core["audio_window_binaural"])
    _write_json(step_dir / "step_summary.json", _step_summary(step_sample))


def _window_summary(window_sample) -> dict:
    return {
        "ref": {
            "episode_id": window_sample.ref.episode_id,
            "window_end_model_step_index": window_sample.ref.window_end_model_step_index,
            "window_end_simulation_step_index": window_sample.ref.window_end_simulation_step_index,
            "window_length_steps": window_sample.ref.window_length_steps,
            "max_steps": window_sample.ref.max_steps,
        },
        "step_refs": [ref.global_model_step_index for ref in window_sample.step_refs],
        "window_valid_mask": window_sample.window_valid_mask,
        "dt_to_prev": window_sample.dt_to_prev,
        "time_from_now": window_sample.time_from_now,
        "has_prev_step": window_sample.has_prev_step,
        "audio_window_shape": _shape_of_nested(window_sample.core["audio_window_binaural"]),
        "binaural_energy_shape": _shape_of_nested(window_sample.audio_features["binaural_energy_t"]),
        "binaural_cue_shape": _shape_of_nested(window_sample.audio_features["binaural_cue_vector_t"]),
        "gt_relative_position_last": window_sample.core["gt_relative_position"][-1],
        "gt_relative_orientation_last": window_sample.core["gt_relative_orientation"][-1],
        "gt_doa_unit_vector_body_last": window_sample.rule_targets["gt_doa_unit_vector_body"][-1],
        "gt_log_distance_scalar_last": window_sample.rule_targets["gt_log_distance_scalar"][-1],
        "target_pos_conf_last": window_sample.rule_targets["target_pos_conf"][-1],
        "target_ori_conf_last": window_sample.rule_targets["target_ori_conf"][-1],
    }


def _shape_of_nested(value) -> list[int]:
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def _export_window(output_dir: Path, name: str, window_sample) -> None:
    window_dir = output_dir / name
    window_dir.mkdir(parents=True, exist_ok=True)
    _write_json(window_dir / "window_summary.json", _window_summary(window_sample))


def main() -> None:
    args = _build_parser().parse_args()
    contract = load_dataset_contract(args.dataset_root)
    _print_contract(contract)

    step_dataset = StepDataset(args.dataset_root, contract=contract)
    step_sample = step_dataset[args.step_index]
    print("step_index:", args.step_index)
    print("step_ref:", step_sample.ref)
    print(
        "front_camera_image_shape:",
        len(step_sample.core["front_camera_image"]),
        len(step_sample.core["front_camera_image"][0]),
        len(step_sample.core["front_camera_image"][0][0]),
    )
    print(
        "audio_window_shape:",
        len(step_sample.core["audio_window_binaural"]),
        len(step_sample.core["audio_window_binaural"][0]),
    )
    print("binaural_energy_t:", step_sample.audio_features["binaural_energy_t"])
    print("binaural_cue_vector_t:", step_sample.audio_features["binaural_cue_vector_t"])
    print("gt_relative_position:", step_sample.core["gt_relative_position"])
    print("gt_doa_unit_vector_body:", step_sample.rule_targets["gt_doa_unit_vector_body"])
    print("gt_log_distance_scalar:", step_sample.rule_targets["gt_log_distance_scalar"])
    print("target_pos_conf:", step_sample.rule_targets["target_pos_conf"])
    print("target_ori_conf:", step_sample.rule_targets["target_ori_conf"])

    fixed_window_dataset = WindowDataset(
        args.dataset_root,
        max_steps=args.max_steps,
        contract=contract,
    )
    fixed_window = fixed_window_dataset[args.window_index]
    print("fixed_window_step_refs:", [ref.global_model_step_index for ref in fixed_window.step_refs])
    print("fixed_window_mask:", fixed_window.window_valid_mask)
    print("fixed_window_dt_to_prev:", fixed_window.dt_to_prev)

    sampled_window_dataset = WindowDataset(
        args.dataset_root,
        max_steps=args.max_steps,
        min_stride_steps=1,
        max_stride_steps=args.max_stride_steps,
        random_seed=args.random_seed,
        contract=contract,
    )
    sampled_window = sampled_window_dataset[args.window_index]
    print(
        "sampled_window_step_refs:",
        [ref.global_model_step_index for ref in sampled_window.step_refs],
    )
    print("sampled_window_mask:", sampled_window.window_valid_mask)
    print("sampled_window_dt_to_prev:", sampled_window.dt_to_prev)
    print("sampled_window_time_from_now:", sampled_window.time_from_now)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        _export_step(output_dir, args.step_index, step_sample)
        _export_window(output_dir, f"window_fixed_{args.window_index:06d}", fixed_window)
        _export_window(output_dir, f"window_sampled_{args.window_index:06d}", sampled_window)
        print("export_dir:", output_dir)


if __name__ == "__main__":
    main()
