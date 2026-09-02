from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dfb_state_estimation.datasets import WindowDataset
from dfb_state_estimation.models.temporal import (
    BeliefUpdateInputs,
    PolicyViewAdapter,
    TemporalBeliefUpdateStage,
    TemporalModalityCalibrationStage,
    TemporalModalityInputs,
    compute_delta_binaural_cue_t,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-validate belief update token construction."
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
    binaural_cue_vector_t = _tensorize_sequence(
        sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32
    )
    modality_inputs = TemporalModalityInputs(
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
        binaural_cue_vector_t=binaural_cue_vector_t,
        delta_binaural_cue_t=compute_delta_binaural_cue_t(binaural_cue_vector_t),
        dt_to_prev=torch.tensor(sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
        time_from_now=torch.tensor(sample.time_from_now, dtype=torch.float32).unsqueeze(0),
    )
    stage1 = TemporalModalityCalibrationStage()
    stage1_output = stage1(modality_inputs)

    belief_inputs = BeliefUpdateInputs(
        coarse_state_t=stage1_output.coarse_state,
        context_relative_position=_tensorize_sequence(
            sample.core["gt_relative_position"], dtype=torch.float32
        ),
        context_relative_orientation=_tensorize_sequence(
            sample.core["gt_relative_orientation"], dtype=torch.float32
        ),
        context_position_confidence=_tensorize_sequence(
            sample.rule_targets["target_pos_conf"], dtype=torch.float32
        ),
        context_orientation_confidence=_tensorize_sequence(
            sample.rule_targets["target_ori_conf"], dtype=torch.float32
        ),
        visual_evidence_strength_t=stage1_output.visual_evidence_strength,
        audio_evidence_strength_t=stage1_output.audio_evidence_strength,
        linear_velocity=_tensorize_sequence(sample.core["gt_linear_velocity"], dtype=torch.float32),
        angular_velocity=_tensorize_sequence(sample.core["gt_angular_velocity"], dtype=torch.float32),
        dt_to_prev=torch.tensor(sample.dt_to_prev, dtype=torch.float32).unsqueeze(0),
        time_from_now=torch.tensor(sample.time_from_now, dtype=torch.float32).unsqueeze(0),
    )
    stage = TemporalBeliefUpdateStage()
    output = stage(belief_inputs)
    policy_view = PolicyViewAdapter()(output.belief_state)
    print("delta_position:", tuple(output.token_output.delta_position.shape))
    print("delta_orientation:", tuple(output.token_output.delta_orientation.shape))
    print("belief_update_tokens:", tuple(output.token_output.belief_update_tokens.shape))
    print("belief_hidden_states:", tuple(output.backbone.hidden_states.shape))
    print("belief_relative_position:", tuple(output.belief_state.relative_position.shape))
    print("belief_relative_orientation:", tuple(output.belief_state.relative_orientation.shape))
    print("belief_linear_velocity:", tuple(output.belief_state.linear_velocity.shape))
    print("belief_angular_velocity:", tuple(output.belief_state.angular_velocity.shape))
    print("belief_track_confidence:", tuple(output.belief_state.track_confidence.shape))
    print("policy_relative_position:", tuple(policy_view.relative_position.shape))
    print("policy_relative_orientation:", tuple(policy_view.relative_orientation.shape))
    print("policy_position_confidence:", tuple(policy_view.position_confidence.shape))
    print("policy_orientation_confidence:", tuple(policy_view.orientation_confidence.shape))
    print("policy_linear_velocity:", tuple(policy_view.linear_velocity.shape))
    print("policy_angular_velocity:", tuple(policy_view.angular_velocity.shape))
    print("policy_track_confidence:", tuple(policy_view.track_confidence.shape))


if __name__ == "__main__":
    main()
