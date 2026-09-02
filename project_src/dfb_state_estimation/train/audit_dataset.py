from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from dfb_state_estimation.datasets import StepDataset, WindowDataset, load_dataset_contract


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit packed DFB state estimation dataset integrity."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to packed dataset root containing meta.json and schema.json.",
    )
    parser.add_argument(
        "--max-step-samples",
        type=int,
        default=16,
        help="Number of step samples to inspect through StepDataset.",
    )
    parser.add_argument(
        "--max-window-samples",
        type=int,
        default=8,
        help="Number of window samples to inspect through WindowDataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Window length used for WindowDataset checks.",
    )
    parser.add_argument(
        "--max-stride-steps",
        type=int,
        default=1,
        help="Maximum stride used for sampled window checks.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Deterministic seed for non-uniform window sampling.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=50,
        help="Maximum number of issues to print before exiting.",
    )
    return parser


def _sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted({round(i * (total - 1) / (limit - 1)) for i in range(limit)})


def _infer_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _infer_shape(value[0])


def _all_finite(value: Any) -> bool:
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _in_range_0_1(value: Any) -> bool:
    if isinstance(value, list):
        return all(_in_range_0_1(item) for item in value)
    if isinstance(value, (float, int)):
        return 0.0 <= float(value) <= 1.0
    return False


def _record_issue(issues: list[str], message: str, max_issues: int) -> None:
    if len(issues) < max_issues:
        issues.append(message)


def _validate_float_array(
    issues: list[str],
    label: str,
    array: np.ndarray,
    max_issues: int,
) -> None:
    if not np.isfinite(array).all():
        _record_issue(issues, f"{label}: contains non-finite values", max_issues)


def _validate_storage_layer(dataset_root: Path, max_issues: int) -> list[str]:
    contract = load_dataset_contract(dataset_root)
    issues: list[str] = []
    schema_groups = contract.schema.storage_schema["groups"]

    for chunk in contract.meta.chunks:
        for group_name, relative_path in chunk.group_files.items():
            group_path = dataset_root / relative_path
            if not group_path.exists():
                _record_issue(
                    issues,
                    f"chunk {chunk.chunk_id} group {group_name}: missing file {relative_path}",
                    max_issues,
                )
                continue

            with np.load(group_path) as archive:
                expected_fields = set(schema_groups[group_name]["fields"].keys())
                actual_fields = set(archive.files)
                if expected_fields != actual_fields:
                    missing = sorted(expected_fields - actual_fields)
                    extra = sorted(actual_fields - expected_fields)
                    _record_issue(
                        issues,
                        f"chunk {chunk.chunk_id} group {group_name}: field mismatch "
                        f"missing={missing} extra={extra}",
                        max_issues,
                    )

                for field_name in expected_fields & actual_fields:
                    array = archive[field_name]
                    if array.shape[0] != chunk.step_count:
                        _record_issue(
                            issues,
                            f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                            f"leading dim {array.shape[0]} != chunk.step_count {chunk.step_count}",
                            max_issues,
                        )

                    if np.issubdtype(array.dtype, np.floating):
                        _validate_float_array(
                            issues,
                            f"chunk {chunk.chunk_id} field {group_name}.{field_name}",
                            array,
                            max_issues,
                        )

                    if group_name == "core":
                        if field_name in {"front_camera_image", "rear_camera_image"}:
                            if array.ndim != 4 or array.shape[-1] != 4:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,H,W,4], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name == "audio_window_binaural":
                            if array.ndim != 3 or array.shape[-1] != 2:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,S,2], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name in {
                            "ego_position_world",
                            "ego_linear_velocity_world",
                            "ego_angular_velocity_body",
                            "gt_relative_position",
                            "gt_linear_velocity",
                            "gt_angular_velocity",
                        }:
                            if array.ndim != 2 or array.shape[-1] != 3:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,3], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name in {"ego_orientation_world", "gt_relative_orientation"}:
                            if array.ndim != 2 or array.shape[-1] != 6:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,6], got {tuple(array.shape)}",
                                    max_issues,
                                )
                    elif group_name == "vision_labels":
                        if field_name.startswith("segmentation_mask_"):
                            values = np.unique(array)
                            if not np.isin(values, [0, 1, 2]).all():
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"unexpected class ids {values.tolist()}",
                                    max_issues,
                                )
                        elif field_name.startswith("keypoints_2d_"):
                            if array.ndim != 3 or array.shape[-1] != 2:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,K,2], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name.startswith("keypoint_visibility_"):
                            values = np.unique(array)
                            if not np.isin(values, [0, 1]).all():
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"unexpected visibility values {values.tolist()}",
                                    max_issues,
                                )
                        elif field_name.startswith("keypoint_voting_pixels_"):
                            if array.ndim != 3 or array.shape[-1] != 2:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,P,2], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name.startswith("keypoint_voting_unit_vectors_"):
                            if array.ndim != 4 or array.shape[-1] != 2:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,P,K,2], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name.startswith("keypoint_voting_mask_"):
                            values = np.unique(array)
                            if array.ndim != 2:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,P], got {tuple(array.shape)}",
                                    max_issues,
                                )
                            if not np.isin(values, [0, 1]).all():
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"unexpected voting mask values {values.tolist()}",
                                    max_issues,
                                )
                    elif group_name == "audio_features":
                        if field_name == "binaural_energy_t":
                            if array.ndim != 2 or array.shape[-1] != 4:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,4], got {tuple(array.shape)}",
                                    max_issues,
                                )
                        elif field_name == "binaural_cue_vector_t":
                            if array.ndim != 2 or array.shape[-1] != 10:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,10], got {tuple(array.shape)}",
                                    max_issues,
                                )
                    elif group_name == "rule_targets":
                        if field_name == "gt_doa_unit_vector_body":
                            if array.ndim != 2 or array.shape[-1] != 3:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N,3], got {tuple(array.shape)}",
                                    max_issues,
                                )
                            elif not np.isfinite(array).all():
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    "contains non-finite values",
                                    max_issues,
                                )
                        elif field_name == "gt_log_distance_scalar":
                            if array.ndim != 1:
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected [N], got {tuple(array.shape)}",
                                    max_issues,
                                )
                            elif not np.isfinite(array).all():
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    "contains non-finite values",
                                    max_issues,
                                )
                        else:
                            if (
                                not np.isfinite(array).all()
                                or np.min(array) < 0.0
                                or np.max(array) > 1.0
                            ):
                                _record_issue(
                                    issues,
                                    f"chunk {chunk.chunk_id} field {group_name}.{field_name}: "
                                    f"expected range [0,1], got min={float(np.min(array)):.4f} "
                                    f"max={float(np.max(array)):.4f}",
                                    max_issues,
                                )

    return issues


def _validate_step_sample(step_sample: Any, max_issues: int) -> list[str]:
    issues: list[str] = []
    core = step_sample.core
    vision = step_sample.vision_labels
    audio = step_sample.audio_features
    rules = step_sample.rule_targets

    if _infer_shape(core["front_camera_image"])[-1:] != (4,):
        _record_issue(issues, "step front_camera_image shape is not [...,4]", max_issues)
    if _infer_shape(core["rear_camera_image"])[-1:] != (4,):
        _record_issue(issues, "step rear_camera_image shape is not [...,4]", max_issues)
    if _infer_shape(core["audio_window_binaural"])[-1:] != (2,):
        _record_issue(issues, "step audio_window_binaural shape is not [S,2]", max_issues)
    if len(audio["binaural_energy_t"]) != 4:
        _record_issue(issues, "step binaural_energy_t len != 4", max_issues)
    if len(audio["binaural_cue_vector_t"]) != 10:
        _record_issue(issues, "step binaural_cue_vector_t len != 10", max_issues)
    if len(core["gt_relative_position"]) != 3:
        _record_issue(issues, "step gt_relative_position len != 3", max_issues)
    if len(rules["gt_doa_unit_vector_body"]) != 3:
        _record_issue(issues, "step gt_doa_unit_vector_body len != 3", max_issues)
    if len(core["gt_relative_orientation"]) != 6:
        _record_issue(issues, "step gt_relative_orientation len != 6", max_issues)
    if not _all_finite(core["audio_window_binaural"]):
        _record_issue(issues, "step audio_window_binaural contains non-finite values", max_issues)
    if not _all_finite(audio["binaural_energy_t"]):
        _record_issue(issues, "step binaural_energy_t contains non-finite values", max_issues)
    if not _all_finite(audio["binaural_cue_vector_t"]):
        _record_issue(issues, "step binaural_cue_vector_t contains non-finite values", max_issues)
    if not _all_finite(rules["gt_doa_unit_vector_body"]):
        _record_issue(issues, "step gt_doa_unit_vector_body contains non-finite values", max_issues)
    if not _all_finite(rules["gt_log_distance_scalar"]):
        _record_issue(issues, "step gt_log_distance_scalar is non-finite", max_issues)
    if not _in_range_0_1(rules["target_pos_conf"]):
        _record_issue(issues, "step target_pos_conf out of [0,1]", max_issues)
    if not _in_range_0_1(rules["target_ori_conf"]):
        _record_issue(issues, "step target_ori_conf out of [0,1]", max_issues)

    front_mask_values = {
        int(value)
        for row in vision["segmentation_mask_front"]
        for value in row
    }
    rear_mask_values = {
        int(value)
        for row in vision["segmentation_mask_rear"]
        for value in row
    }
    if not front_mask_values.issubset({0, 1, 2}):
        _record_issue(issues, f"step front segmentation has invalid classes {sorted(front_mask_values)}", max_issues)
    if not rear_mask_values.issubset({0, 1, 2}):
        _record_issue(issues, f"step rear segmentation has invalid classes {sorted(rear_mask_values)}", max_issues)

    if any(value not in (0, 1) for value in vision["keypoint_visibility_front"]):
        _record_issue(issues, "step keypoint_visibility_front contains values outside {0,1}", max_issues)
    if any(value not in (0, 1) for value in vision["keypoint_visibility_rear"]):
        _record_issue(issues, "step keypoint_visibility_rear contains values outside {0,1}", max_issues)
    if "keypoint_voting_pixels_front" in vision:
        if _infer_shape(vision["keypoint_voting_pixels_front"])[-1:] != (2,):
            _record_issue(
                issues,
                "step keypoint_voting_pixels_front shape is not [P,2]",
                max_issues,
            )
    if "keypoint_voting_pixels_rear" in vision:
        if _infer_shape(vision["keypoint_voting_pixels_rear"])[-1:] != (2,):
            _record_issue(
                issues,
                "step keypoint_voting_pixels_rear shape is not [P,2]",
                max_issues,
            )
    if "keypoint_voting_unit_vectors_front" in vision:
        if _infer_shape(vision["keypoint_voting_unit_vectors_front"])[-1:] != (2,):
            _record_issue(
                issues,
                "step keypoint_voting_unit_vectors_front shape is not [P,K,2]",
                max_issues,
            )
        if not _all_finite(vision["keypoint_voting_unit_vectors_front"]):
            _record_issue(
                issues,
                "step keypoint_voting_unit_vectors_front contains non-finite values",
                max_issues,
            )
    if "keypoint_voting_unit_vectors_rear" in vision:
        if _infer_shape(vision["keypoint_voting_unit_vectors_rear"])[-1:] != (2,):
            _record_issue(
                issues,
                "step keypoint_voting_unit_vectors_rear shape is not [P,K,2]",
                max_issues,
            )
        if not _all_finite(vision["keypoint_voting_unit_vectors_rear"]):
            _record_issue(
                issues,
                "step keypoint_voting_unit_vectors_rear contains non-finite values",
                max_issues,
            )
    if "keypoint_voting_mask_front" in vision:
        values = {int(value) for value in vision["keypoint_voting_mask_front"]}
        if not values.issubset({0, 1}):
            _record_issue(
                issues,
                f"step keypoint_voting_mask_front has invalid values {sorted(values)}",
                max_issues,
            )
    if "keypoint_voting_mask_rear" in vision:
        values = {int(value) for value in vision["keypoint_voting_mask_rear"]}
        if not values.issubset({0, 1}):
            _record_issue(
                issues,
                f"step keypoint_voting_mask_rear has invalid values {sorted(values)}",
                max_issues,
            )

    return issues


def _validate_window_sample(window_sample: Any, max_issues: int) -> list[str]:
    issues: list[str] = []
    max_steps = window_sample.ref.max_steps

    for field_name in ("window_valid_mask", "dt_to_prev", "time_from_now", "has_prev_step"):
        value = getattr(window_sample, field_name)
        if len(value) != max_steps:
            _record_issue(
                issues,
                f"window {field_name} len {len(value)} != max_steps {max_steps}",
                max_issues,
            )

    valid_mask = window_sample.window_valid_mask
    valid_count = sum(valid_mask)
    if valid_count != len(window_sample.step_refs):
        _record_issue(
            issues,
            f"window valid_count {valid_count} != step_ref_count {len(window_sample.step_refs)}",
            max_issues,
        )
    if any(value not in (0, 1) for value in valid_mask):
        _record_issue(issues, "window_valid_mask contains values outside {0,1}", max_issues)
    if valid_mask != sorted(valid_mask):
        _record_issue(issues, "window_valid_mask is not zero-prefix then one-suffix", max_issues)

    valid_dt = [value for value, flag in zip(window_sample.dt_to_prev, valid_mask) if flag]
    valid_t = [value for value, flag in zip(window_sample.time_from_now, valid_mask) if flag]
    if valid_dt:
        if abs(valid_dt[0]) > 1e-6:
            _record_issue(issues, f"window first valid dt_to_prev is not zero: {valid_dt[0]}", max_issues)
        if any(value < 0.0 for value in valid_dt[1:]):
            _record_issue(issues, "window valid dt_to_prev contains negative values", max_issues)
    if valid_t:
        if abs(valid_t[-1]) > 1e-6:
            _record_issue(issues, f"window last valid time_from_now is not zero: {valid_t[-1]}", max_issues)
        if any(left < right for left, right in zip(valid_t, valid_t[1:])):
            _record_issue(issues, "window valid time_from_now is not monotonically non-increasing", max_issues)

    if _infer_shape(window_sample.core["audio_window_binaural"])[-1:] != (2,):
        _record_issue(issues, "window audio_window_binaural shape is not [T,S,2]", max_issues)
    if _infer_shape(window_sample.audio_features["binaural_energy_t"])[-1:] != (4,):
        _record_issue(issues, "window binaural_energy_t shape is not [T,4]", max_issues)
    if _infer_shape(window_sample.audio_features["binaural_cue_vector_t"])[-1:] != (10,):
        _record_issue(issues, "window binaural_cue_vector_t shape is not [T,10]", max_issues)
    if _infer_shape(window_sample.rule_targets["gt_doa_unit_vector_body"])[-1:] != (3,):
        _record_issue(issues, "window gt_doa_unit_vector_body shape is not [T,3]", max_issues)
    if "keypoint_voting_pixels_front" in window_sample.vision_labels:
        if _infer_shape(window_sample.vision_labels["keypoint_voting_pixels_front"])[-1:] != (2,):
            _record_issue(
                issues,
                "window keypoint_voting_pixels_front shape is not [T,P,2]",
                max_issues,
            )
    if "keypoint_voting_pixels_rear" in window_sample.vision_labels:
        if _infer_shape(window_sample.vision_labels["keypoint_voting_pixels_rear"])[-1:] != (2,):
            _record_issue(
                issues,
                "window keypoint_voting_pixels_rear shape is not [T,P,2]",
                max_issues,
            )
    if "keypoint_voting_unit_vectors_front" in window_sample.vision_labels:
        if _infer_shape(window_sample.vision_labels["keypoint_voting_unit_vectors_front"])[-1:] != (2,):
            _record_issue(
                issues,
                "window keypoint_voting_unit_vectors_front shape is not [T,P,K,2]",
                max_issues,
            )
    if "keypoint_voting_unit_vectors_rear" in window_sample.vision_labels:
        if _infer_shape(window_sample.vision_labels["keypoint_voting_unit_vectors_rear"])[-1:] != (2,):
            _record_issue(
                issues,
                "window keypoint_voting_unit_vectors_rear shape is not [T,P,K,2]",
                max_issues,
            )

    return issues


def main() -> None:
    args = _build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    contract = load_dataset_contract(dataset_root)

    storage_issues = _validate_storage_layer(dataset_root, args.max_issues)

    step_dataset = StepDataset(dataset_root, contract=contract)
    step_issues: list[str] = []
    for index in _sample_indices(len(step_dataset), args.max_step_samples):
        for issue in _validate_step_sample(step_dataset[index], args.max_issues):
            _record_issue(step_issues, f"step[{index}]: {issue}", args.max_issues)

    window_dataset = WindowDataset(
        dataset_root,
        max_steps=args.max_steps,
        max_stride_steps=args.max_stride_steps,
        random_seed=args.random_seed,
        contract=contract,
    )
    window_issues: list[str] = []
    for index in _sample_indices(len(window_dataset), args.max_window_samples):
        for issue in _validate_window_sample(window_dataset[index], args.max_issues):
            _record_issue(window_issues, f"window[{index}]: {issue}", args.max_issues)

    print("dataset audit:")
    print("  dataset_root:", dataset_root)
    print("  schema_id:", contract.schema.schema_id)
    print("  chunks:", len(contract.meta.chunks))
    print("  episodes:", len(contract.meta.episodes))
    print("  step_samples_checked:", len(_sample_indices(len(step_dataset), args.max_step_samples)))
    print("  window_samples_checked:", len(_sample_indices(len(window_dataset), args.max_window_samples)))
    print("  storage_issues:", len(storage_issues))
    print("  step_issues:", len(step_issues))
    print("  window_issues:", len(window_issues))

    all_issues = storage_issues + step_issues + window_issues
    if all_issues:
        print("dataset audit failed:")
        for issue in all_issues[: args.max_issues]:
            print("  -", issue)
        raise SystemExit(1)

    print("dataset audit passed")


if __name__ == "__main__":
    main()
