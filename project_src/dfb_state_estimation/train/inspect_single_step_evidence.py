from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dfb_state_estimation.datasets import StepDataset, target_segmentation_class_id
from dfb_state_estimation.losses import (
    EvidenceSupervisionTargets,
    compute_single_step_evidence_loss,
)
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.evidence import SingleStepEvidenceModule
from dfb_state_estimation.models.vision import SingleStepVisionModule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-validate the single-step evidence module."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--step-index", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _rgba_image_to_tensor(image: list[list[list[int]]]) -> torch.Tensor:
    tensor = torch.tensor(image, dtype=torch.float32)[..., :3].permute(2, 0, 1)
    return tensor / 255.0


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    dataset = StepDataset(args.dataset_root)
    sample = dataset[args.step_index]

    front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0)
    rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0)
    audio_window = torch.tensor(sample.core["audio_window_binaural"], dtype=torch.float32).unsqueeze(0)
    binaural_energy_t = torch.tensor(sample.audio_features["binaural_energy_t"], dtype=torch.float32).unsqueeze(0)
    binaural_cue_vector_t = torch.tensor(
        sample.audio_features["binaural_cue_vector_t"],
        dtype=torch.float32,
    ).unsqueeze(0)

    vision = SingleStepVisionModule()
    audio = SingleStepAudioModule()
    evidence = SingleStepEvidenceModule()
    vision.train()
    audio.train()
    evidence.train()

    target_class_ids = torch.tensor(
        [target_segmentation_class_id(sample.ref.observed_role)],
        dtype=torch.long,
    )
    vision_output = vision(front, rear, target_class_ids)
    audio_output = audio(
        audio_window,
        binaural_energy_t,
        binaural_cue_vector_t,
    )
    evidence_output = evidence(vision_output, audio_output)
    print("audio_relative_position:", evidence_output.evidence.audio_relative_position.tolist())
    print(
        "audio_position_confidence:",
        evidence_output.evidence.audio_position_confidence.tolist(),
    )
    print("visual_relative_position:", evidence_output.evidence.visual_relative_position.tolist())
    print(
        "visual_position_confidence:",
        evidence_output.evidence.visual_position_confidence.tolist(),
    )
    print("relative_position:", tuple(evidence_output.evidence_state.relative_position.shape))
    print(
        "relative_orientation:",
        tuple(evidence_output.evidence_state.relative_orientation.shape),
    )
    print(
        "position_confidence:",
        evidence_output.evidence_state.position_confidence.tolist(),
    )
    print(
        "orientation_confidence:",
        evidence_output.evidence_state.orientation_confidence.tolist(),
    )
    print(
        "raw_visual_evidence_strength:",
        evidence_output.evidence.raw_visual_evidence_strength.tolist(),
    )
    print(
        "raw_audio_evidence_strength:",
        evidence_output.evidence.raw_audio_evidence_strength.tolist(),
    )

    targets = EvidenceSupervisionTargets(
        gt_relative_position=torch.tensor(
            sample.core["gt_relative_position"],
            dtype=torch.float32,
        ).unsqueeze(0),
        gt_relative_orientation=torch.tensor(
            sample.core["gt_relative_orientation"],
            dtype=torch.float32,
        ).unsqueeze(0),
        target_pos_conf=torch.tensor(
            sample.rule_targets["target_pos_conf"],
            dtype=torch.float32,
        ).unsqueeze(0),
        target_ori_conf=torch.tensor(
            sample.rule_targets["target_ori_conf"],
            dtype=torch.float32,
        ).unsqueeze(0),
        view_valid_target=torch.tensor(
            [
                float(
                    any(int(value) == 2 for row in sample.vision_labels["segmentation_mask_front"] for value in row)
                    or any(int(value) == 2 for row in sample.vision_labels["segmentation_mask_rear"] for value in row)
                )
            ],
            dtype=torch.float32,
        ),
        pos_valid_target=torch.tensor(
            [float(sample.rule_targets["target_pos_conf"] > 0.0)],
            dtype=torch.float32,
        ),
        ori_valid_target=torch.tensor(
            [float(sample.rule_targets["target_ori_conf"] > 0.0)],
            dtype=torch.float32,
        ),
    )
    losses = compute_single_step_evidence_loss(evidence_output, targets)
    losses["total"].backward()
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")
    print("backward: ok")


if __name__ == "__main__":
    main()
