from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dfb_state_estimation.datasets import AudioOnlyRecordingDataset
from dfb_state_estimation.losses.audio_supervision import (
    AudioConfidenceConfig,
    compute_audio_confidence_targets,
)
from dfb_state_estimation.models.audio import SingleStepAudioModule
from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import (
    _collate_audio_only_batch,
    _move_audio_only_batch_to_device,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect audio-only samples from a training checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0,512,1024,1535,2047",
        help="Comma-separated sample indices. Use 'auto:N' for evenly spaced N samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference, usually 'cpu' or 'cuda'.",
    )
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_sample_indices(spec: str, total: int) -> list[int]:
    spec = spec.strip()
    if spec.startswith("auto:"):
        count = int(spec.split(":", 1)[1])
        if count <= 0:
            raise ValueError("auto sample count must be positive")
        if total <= count:
            return list(range(total))
        return [int(round(i * (total - 1) / (count - 1))) for i in range(count)]
    indices = [int(value) for value in spec.split(",") if value.strip()]
    if not indices:
        raise ValueError("sample-indices must not be empty")
    return indices


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return 0.0
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(x_var * y_var)


def _audio_confidence_from_loaded_config(config) -> AudioConfidenceConfig:
    return AudioConfidenceConfig(
        doa_half_angle=math.radians(config.audio_confidence.doa_half_angle_degrees),
        dist_half_error=config.audio_confidence.dist_half_error,
    )


def main() -> None:
    args = _build_parser().parse_args()
    config = load_train_config(args.config)
    dataset = AudioOnlyRecordingDataset(config.dataset_root)
    sample_indices = _parse_sample_indices(args.sample_indices, len(dataset))
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    audio = SingleStepAudioModule().to(device).eval()
    state_dict = checkpoint["modules"]["audio"]
    current_state = audio.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    audio.load_state_dict(filtered_state_dict, strict=False)
    confidence_config = _audio_confidence_from_loaded_config(config)

    aggregate: list[dict[str, object]] = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        batch = _collate_audio_only_batch([sample])
        batch = _move_audio_only_batch_to_device(batch, device)
        with torch.no_grad():
            output = audio(
                batch["audio_window"],
                batch["binaural_energy_t"],
                batch["binaural_cue_vector_t"],
            )
        confidence_targets = compute_audio_confidence_targets(
            output,
            batch["audio_targets"],
            config=confidence_config,
        )
        energy = batch["binaural_energy_t"][0].detach().cpu().tolist()
        cues = batch["binaural_cue_vector_t"][0].detach().cpu().tolist()
        doa_angle_error_rad = float(confidence_targets["doa_angle_error"][0].item())
        summary = {
            "index": sample_index,
            "episode_id": sample.ref.episode_id,
            "observed_role": sample.ref.observed_role,
            "simulation_step_index": sample.ref.simulation_step_index,
            "pred_doa_unit_vector_body": output.doa_unit_vector_body[0].detach().cpu().tolist(),
            "gt_doa_unit_vector_body": batch["audio_targets"].gt_doa_unit_vector_body[0].detach().cpu().tolist(),
            "pred_log_distance_scalar": float(output.log_distance_scalar[0].item()),
            "gt_log_distance_scalar": float(batch["audio_targets"].gt_log_distance_scalar[0].item()),
            "doa_angle_error_rad": doa_angle_error_rad,
            "doa_angle_error_deg": math.degrees(doa_angle_error_rad),
            "log_distance_error": float(confidence_targets["log_distance_error"][0].item()),
            "doa_conf": float(output.doa_conf[0].item()),
            "dist_conf": float(output.dist_conf[0].item()),
            "raw_audio_evidence_strength": float(output.raw_audio_evidence_strength[0].item()),
            "target_doa_conf": float(confidence_targets["target_doa_conf"][0].item()),
            "target_dist_conf": float(confidence_targets["target_dist_conf"][0].item()),
            "binaural_energy_t": energy,
            "binaural_cue_vector_t": cues,
            "energy_sum": float(energy[2]),
            "energy_diff_norm": float(energy[3]),
            "gcc_peak": float(cues[1]),
            "ild_db": float(cues[2]),
            "ipd_low": float(cues[3]),
            "ipd_high": float(cues[4]),
            "interaural_coherence": float(cues[5]),
            "reverb_proxy": float(cues[6]),
            "directness_proxy": float(cues[7]),
            "ild_low_band_db": float(cues[8]),
            "ild_high_band_db": float(cues[9]),
        }
        sample_dir = args.output_dir / f"sample_{sample_index:06d}"
        _write_json(sample_dir / "summary.json", summary)
        aggregate.append(summary)

    doa_errors_deg = [float(item["doa_angle_error_deg"]) for item in aggregate]
    logdist_errors = [float(item["log_distance_error"]) for item in aggregate]
    doa_conf = [float(item["doa_conf"]) for item in aggregate]
    dist_conf = [float(item["dist_conf"]) for item in aggregate]
    raw_strength = [float(item["raw_audio_evidence_strength"]) for item in aggregate]
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(config.dataset_root),
        "sample_indices": sample_indices,
        "sample_count": len(aggregate),
        "mean_doa_angle_error_deg": sum(doa_errors_deg) / len(doa_errors_deg),
        "mean_log_distance_error": sum(logdist_errors) / len(logdist_errors),
        "mean_doa_conf": sum(doa_conf) / len(doa_conf),
        "mean_dist_conf": sum(dist_conf) / len(dist_conf),
        "mean_raw_audio_evidence_strength": sum(raw_strength) / len(raw_strength),
        "doa_conf_vs_error_corr": _corr(doa_conf, doa_errors_deg),
        "dist_conf_vs_error_corr": _corr(dist_conf, logdist_errors),
    }
    _write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "sample_indices.txt").write_text(
        ", ".join(str(index) for index in sample_indices) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
