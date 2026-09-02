from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dfb_state_estimation.datasets import StepDataset, target_segmentation_class_id
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.models.evidence import SingleStepEvidenceConfig, SingleStepEvidenceModule
from dfb_state_estimation.models.vision import SingleStepVisionModule
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import _build_vision_module_for_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose single-step evidence position chain against GT."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def _rgba_image_to_tensor(image: list[list[list[int]]]) -> torch.Tensor:
    tensor = torch.tensor(image, dtype=torch.float32)[..., :3].permute(2, 0, 1)
    return tensor / 255.0


def _build_evidence_module(config_path: Path) -> SingleStepEvidenceModule:
    config = load_train_config(config_path)
    evidence_config = SingleStepEvidenceConfig(
        position_refine_scale=config.evidence.position_refine_scale
    )
    return SingleStepEvidenceModule(evidence_config)


def _load_modules(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[SingleStepVisionModule, SingleStepAudioModule, SingleStepEvidenceModule]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    module_states = payload["modules"]
    config = load_train_config(config_path)
    vision = _build_vision_module_for_config(config).to(device).eval()
    audio = SingleStepAudioModule().to(device).eval()
    evidence = _build_evidence_module(config_path).to(device).eval()
    vision.load_state_dict(module_states["vision"], strict=False)
    current_audio = audio.state_dict()
    filtered_audio = {
        key: value
        for key, value in module_states["audio"].items()
        if key in current_audio and current_audio[key].shape == value.shape
    }
    audio.load_state_dict(filtered_audio, strict=False)
    evidence.load_state_dict(module_states["evidence"])
    return vision, audio, evidence


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


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor).detach().cpu().item())


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = StepDataset(args.dataset_root)
    vision, audio, evidence = _load_modules(args.checkpoint, args.config, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_entries: list[dict[str, Any]] = []
    visual_errors: list[float] = []
    audio_errors: list[float] = []
    fused_errors: list[float] = []
    visual_confidences: list[float] = []
    audio_confidences: list[float] = []
    fused_confidences: list[float] = []

    with torch.no_grad():
        for index in _sample_indices(len(dataset), args.num_samples):
            sample = dataset[index]
            front = _rgba_image_to_tensor(sample.core["front_camera_image"]).unsqueeze(0).to(device)
            rear = _rgba_image_to_tensor(sample.core["rear_camera_image"]).unsqueeze(0).to(device)
            audio_window = torch.tensor(sample.core["audio_window_binaural"], dtype=torch.float32).unsqueeze(0).to(device)
            binaural_energy_t = torch.tensor(sample.audio_features["binaural_energy_t"], dtype=torch.float32).unsqueeze(0).to(device)
            binaural_cue_vector_t = torch.tensor(sample.audio_features["binaural_cue_vector_t"], dtype=torch.float32).unsqueeze(0).to(device)
            gt_position = torch.tensor(sample.core["gt_relative_position"], dtype=torch.float32).to(device)

            target_class_ids = torch.tensor(
                [target_segmentation_class_id(sample.ref.observed_role)],
                dtype=torch.long,
                device=device,
            )
            vision_output = vision(front, rear, target_class_ids)
            audio_output = audio(audio_window, binaural_energy_t, binaural_cue_vector_t)
            evidence_output = evidence(vision_output, audio_output)

            visual_position = evidence_output.evidence.visual_relative_position[0]
            audio_position = evidence_output.evidence.audio_relative_position[0]
            fused_position = evidence_output.evidence_state.relative_position[0]

            visual_error = _norm(visual_position - gt_position)
            audio_error = _norm(audio_position - gt_position)
            fused_error = _norm(fused_position - gt_position)
            visual_conf = float(evidence_output.evidence.visual_position_confidence[0].detach().cpu().item())
            audio_conf = float(evidence_output.evidence.audio_position_confidence[0].detach().cpu().item())
            fused_conf = float(evidence_output.evidence_state.position_confidence[0].detach().cpu().item())

            visual_errors.append(visual_error)
            audio_errors.append(audio_error)
            fused_errors.append(fused_error)
            visual_confidences.append(visual_conf)
            audio_confidences.append(audio_conf)
            fused_confidences.append(fused_conf)

            sample_entries.append(
                {
                    "index": index,
                    "episode_id": sample.ref.episode_id,
                    "observed_role": sample.ref.observed_role,
                    "gt_relative_position": gt_position.detach().cpu().tolist(),
                    "visual_relative_position": visual_position.detach().cpu().tolist(),
                    "audio_relative_position": audio_position.detach().cpu().tolist(),
                    "fused_relative_position": fused_position.detach().cpu().tolist(),
                    "visual_position_error_norm": visual_error,
                    "audio_position_error_norm": audio_error,
                    "fused_position_error_norm": fused_error,
                    "visual_position_confidence": visual_conf,
                    "audio_position_confidence": audio_conf,
                    "fused_position_confidence": fused_conf,
                    "selected_view_valid": float(vision_output.selected_candidate.view_valid[0].detach().cpu().item()),
                    "selected_pos_valid": float(vision_output.selected_candidate.pos_valid[0].detach().cpu().item()),
                    "selected_ori_valid": float(vision_output.selected_candidate.ori_valid[0].detach().cpu().item()),
                    "raw_visual_evidence_strength": float(vision_output.selected_candidate.raw_visual_evidence_strength[0].detach().cpu().item()),
                    "raw_audio_evidence_strength": float(audio_output.raw_audio_evidence_strength[0].detach().cpu().item()),
                }
            )

    sample_entries.sort(key=lambda entry: entry["fused_position_error_norm"], reverse=True)
    summary = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_samples": len(sample_entries),
        "mean_visual_position_error_norm": _mean(visual_errors),
        "mean_audio_position_error_norm": _mean(audio_errors),
        "mean_fused_position_error_norm": _mean(fused_errors),
        "mean_visual_position_confidence": _mean(visual_confidences),
        "mean_audio_position_confidence": _mean(audio_confidences),
        "mean_fused_position_confidence": _mean(fused_confidences),
        "top_fused_failures": sample_entries[:8],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "samples.json").write_text(json.dumps(sample_entries, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
