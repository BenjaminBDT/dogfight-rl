from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bucket audio-only inspection failures by cue features.")
    parser.add_argument("--inspect-dir", type=Path, required=True)
    parser.add_argument("--failure-angle-deg", type=float, default=60.0)
    return parser


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_samples(inspect_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for sample_dir in sorted(inspect_dir.glob("sample_*")):
        summary_path = sample_dir / "summary.json"
        if summary_path.exists():
            samples.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not samples:
        raise FileNotFoundError(f"no sample summaries found under {inspect_dir}")
    return samples


def _quantile_thresholds(values: list[float]) -> tuple[float, float]:
    if len(values) < 3:
        median = statistics.median(values)
        return median, median
    ordered = sorted(values)
    low = ordered[len(ordered) // 3]
    high = ordered[(2 * len(ordered)) // 3]
    return low, high


def _bucket_name(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def main() -> None:
    args = _build_parser().parse_args()
    samples = _load_samples(args.inspect_dir)
    fields = {
        "energy_sum": [float(sample["energy_sum"]) for sample in samples],
        "gcc_peak": [float(sample["gcc_peak"]) for sample in samples],
        "ild_abs": [abs(float(sample["ild_db"])) for sample in samples],
        "ild_low_abs": [abs(float(sample["ild_low_band_db"])) for sample in samples],
        "ild_high_abs": [abs(float(sample["ild_high_band_db"])) for sample in samples],
        "interaural_coherence": [float(sample["interaural_coherence"]) for sample in samples],
        "raw_audio_evidence_strength": [float(sample["raw_audio_evidence_strength"]) for sample in samples],
        "doa_conf": [float(sample["doa_conf"]) for sample in samples],
        "dist_conf": [float(sample["dist_conf"]) for sample in samples],
    }
    thresholds = {name: _quantile_thresholds(values) for name, values in fields.items()}

    bucket_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    failures: list[dict] = []
    for sample in samples:
        is_failure = float(sample["doa_angle_error_deg"]) >= args.failure_angle_deg
        if is_failure:
            failures.append(
                {
                    "index": sample["index"],
                    "episode_id": sample["episode_id"],
                    "observed_role": sample["observed_role"],
                    "simulation_step_index": sample["simulation_step_index"],
                    "doa_angle_error_deg": sample["doa_angle_error_deg"],
                    "log_distance_error": sample["log_distance_error"],
                    "doa_conf": sample["doa_conf"],
                    "dist_conf": sample["dist_conf"],
                    "raw_audio_evidence_strength": sample["raw_audio_evidence_strength"],
                    "energy_sum": sample["energy_sum"],
                    "gcc_peak": sample["gcc_peak"],
                    "ild_db": sample["ild_db"],
                    "ild_low_band_db": sample["ild_low_band_db"],
                    "ild_high_band_db": sample["ild_high_band_db"],
                    "interaural_coherence": sample["interaural_coherence"],
                    "reverb_proxy": sample["reverb_proxy"],
                    "directness_proxy": sample["directness_proxy"],
                }
            )
        for field_name, values in fields.items():
            low, high = thresholds[field_name]
            if field_name == "ild_abs":
                value = abs(float(sample["ild_db"]))
            elif field_name == "ild_low_abs":
                value = abs(float(sample["ild_low_band_db"]))
            elif field_name == "ild_high_abs":
                value = abs(float(sample["ild_high_band_db"]))
            else:
                value = float(sample[field_name])
            bucket = _bucket_name(value, low, high)
            stats = bucket_stats.setdefault(field_name, {}).setdefault(
                bucket,
                {"count": 0, "failure_count": 0, "mean_error_deg": 0.0},
            )
            stats["count"] = int(stats["count"]) + 1
            stats["failure_count"] = int(stats["failure_count"]) + int(is_failure)
            stats["mean_error_deg"] = float(stats["mean_error_deg"]) + float(sample["doa_angle_error_deg"])

    for bucket_groups in bucket_stats.values():
        for stats in bucket_groups.values():
            count = max(int(stats["count"]), 1)
            stats["failure_rate"] = float(stats["failure_count"]) / float(count)
            stats["mean_error_deg"] = float(stats["mean_error_deg"]) / float(count)

    summary = {
        "inspect_dir": str(args.inspect_dir),
        "sample_count": len(samples),
        "failure_angle_deg": args.failure_angle_deg,
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(samples),
        "thresholds": {
            name: {"low": low, "high": high} for name, (low, high) in thresholds.items()
        },
        "bucket_stats": bucket_stats,
        "failures": failures,
    }
    _write_json(args.inspect_dir / "failure_buckets.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
