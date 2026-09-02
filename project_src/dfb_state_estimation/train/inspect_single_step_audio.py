from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dfb_state_estimation.datasets import StepDataset
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.audio.module import SingleStepAudioConfig, compute_audio_evidence_terms


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-validate the single-step audio module."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--step-index", type=int, default=10)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    model = SingleStepAudioModule()
    model.train()

    if args.dataset_root is not None:
        step_dataset = StepDataset(args.dataset_root)
        sample = step_dataset[args.step_index]
        audio_window = torch.tensor(
            sample.core["audio_window_binaural"],
            dtype=torch.float32,
        ).unsqueeze(0)
        binaural_energy_t = torch.tensor(
            sample.audio_features["binaural_energy_t"],
            dtype=torch.float32,
        ).unsqueeze(0)
        binaural_cue_vector_t = torch.tensor(
            sample.audio_features["binaural_cue_vector_t"],
            dtype=torch.float32,
        ).unsqueeze(0)
    else:
        audio_window = torch.rand(args.batch_size, args.window_samples, 2)
        binaural_energy_t = torch.rand(args.batch_size, 4)
        binaural_cue_vector_t = torch.rand(args.batch_size, 10)

    output = model(
        audio_window,
        binaural_energy_t,
        binaural_cue_vector_t,
    )
    a_energy, a_cue, _ = compute_audio_evidence_terms(
        binaural_energy_t=binaural_energy_t,
        binaural_cue_vector_t=binaural_cue_vector_t,
        config=SingleStepAudioConfig(),
    )
    print("audio_embedding:", tuple(output.audio_embedding.shape))
    print("doa_unit_vector_body:", tuple(output.doa_unit_vector_body.shape))
    print("doa_conf:", output.doa_conf.tolist())
    print("log_distance_scalar:", output.log_distance_scalar.tolist())
    print("dist_conf:", output.dist_conf.tolist())
    print("a_energy:", a_energy.tolist())
    print("a_cue:", a_cue.tolist())
    print(
        "raw_audio_evidence_strength:",
        output.raw_audio_evidence_strength.tolist(),
    )
    total = (
        output.audio_embedding.pow(2).mean()
        + output.doa_conf.mean()
        + output.dist_conf.mean()
        + output.raw_audio_evidence_strength.mean()
    )
    total.backward()
    print(f"total: {total.item():.6f}")
    print("backward: ok")


if __name__ == "__main__":
    main()
