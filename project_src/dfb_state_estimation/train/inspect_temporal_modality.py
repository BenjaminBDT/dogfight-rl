from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dfb_state_estimation.datasets import WindowDataset
from dfb_state_estimation.models.temporal import (
    TemporalModalityInputs,
    TemporalModalityCalibrationStage,
    compute_delta_binaural_cue_t,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-validate temporal modality token projection."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _tensorize_sequence(values, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype).unsqueeze(0)


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    dataset = WindowDataset(args.dataset_root, max_steps=args.max_steps)
    sample = dataset[args.window_index]
    height = len(sample.core["front_camera_image"][0])
    width = len(sample.core["front_camera_image"][0][0])

    inputs = TemporalModalityInputs(
        relative_position=_tensorize_sequence(sample.core["gt_relative_position"], dtype=torch.float32),
        relative_orientation=_tensorize_sequence(sample.core["gt_relative_orientation"], dtype=torch.float32),
        position_confidence=_tensorize_sequence(sample.rule_targets["target_pos_conf"], dtype=torch.float32),
        orientation_confidence=_tensorize_sequence(sample.rule_targets["target_ori_conf"], dtype=torch.float32),
        pos_valid=torch.ones(1, args.max_steps, dtype=torch.float32),
        ori_valid=torch.ones(1, args.max_steps, dtype=torch.float32),
        visual_embedding=torch.randn(1, args.max_steps, 128),
        audio_embedding=torch.randn(1, args.max_steps, 64),
        raw_visual_evidence_strength=torch.rand(1, args.max_steps),
        view_valid=torch.ones(1, args.max_steps, dtype=torch.float32),
        selected_segmentation_difference_t=torch.zeros(
            1, args.max_steps, 2, height, width, dtype=torch.float32
        ),
        selected_segmentation_diff_valid_t=torch.zeros(1, args.max_steps, dtype=torch.float32),
        selected_keypoint_delta_t=torch.zeros(1, args.max_steps, 9, 2, dtype=torch.float32),
        selected_keypoint_delta_support_summary_t=torch.zeros(1, args.max_steps, dtype=torch.float32),
        selected_keypoint_delta_valid_t=torch.zeros(1, args.max_steps, dtype=torch.float32),
        raw_audio_evidence_strength=torch.rand(1, args.max_steps),
        binaural_energy_t=_tensorize_sequence(sample.audio_features["binaural_energy_t"], dtype=torch.float32),
        binaural_cue_vector_t=_tensorize_sequence(
            sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32
        ),
        delta_binaural_cue_t=compute_delta_binaural_cue_t(
            _tensorize_sequence(sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32)
        ),
        dt_to_prev=torch.tensor(sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
        time_from_now=torch.tensor(sample.time_from_now, dtype=torch.float32).unsqueeze(0),
    )
    model = TemporalModalityCalibrationStage()
    output = model(inputs)
    print("state_tokens:", tuple(output.projected_tokens.state_tokens.shape))
    print("visual_tokens:", tuple(output.projected_tokens.visual_tokens.shape))
    print("audio_tokens:", tuple(output.projected_tokens.audio_tokens.shape))
    print(
        "selected_segmentation_difference_t:",
        tuple(inputs.selected_segmentation_difference_t.shape),
    )
    print(
        "selected_segmentation_diff_valid_t:",
        tuple(inputs.selected_segmentation_diff_valid_t.shape),
    )
    print(
        "selected_keypoint_delta_t:",
        tuple(inputs.selected_keypoint_delta_t.shape),
    )
    print(
        "selected_keypoint_delta_support_summary_t:",
        tuple(inputs.selected_keypoint_delta_support_summary_t.shape),
    )
    print(
        "selected_keypoint_delta_valid_t:",
        tuple(inputs.selected_keypoint_delta_valid_t.shape),
    )
    print("stacked_tokens:", tuple(output.projected_tokens.stacked_tokens.shape))
    print("hidden_tokens:", tuple(output.backbone.hidden_tokens.shape))
    print("state_hidden:", tuple(output.backbone.state_hidden.shape))
    print("visual_hidden:", tuple(output.backbone.visual_hidden.shape))
    print("audio_hidden:", tuple(output.backbone.audio_hidden.shape))
    print("coarse_relative_position:", tuple(output.coarse_state.relative_position.shape))
    print(
        "coarse_relative_orientation:",
        tuple(output.coarse_state.relative_orientation.shape),
    )
    print(
        "position_confidence:",
        tuple(output.coarse_state.position_confidence.shape),
    )
    print(
        "orientation_confidence:",
        tuple(output.coarse_state.orientation_confidence.shape),
    )
    print("visual_evidence_strength:", tuple(output.visual_evidence_strength.shape))
    print("audio_evidence_strength:", tuple(output.audio_evidence_strength.shape))


if __name__ == "__main__":
    main()
