from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


TIMING_FIELDS = (
    "rollout_collection_seconds",
    "policy_forward_seconds",
    "opponent_action_seconds",
    "env_step_seconds",
    "reward_compute_seconds",
    "diagnostics_seconds",
    "env_reset_seconds",
    "ppo_update_seconds",
    "checkpoint_seconds",
    "update_total_seconds",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ordinary-update PPO throughput from metrics.jsonl."
    )
    parser.add_argument("metrics", type=Path, help="Path to a PPO metrics.jsonl file")
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=2,
        help="Discard this many eligible ordinary updates before aggregation",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Aggregate at most this many updates after warm-up (0 = all)",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Include updates whose eval_seconds is greater than zero",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: metric record must be an object")
            records.append(payload)
    return records


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _finite_float(record: dict[str, Any], field: str) -> float:
    if field not in record:
        raise ValueError(f"metric record is missing required field {field!r}")
    value = float(record[field])
    if not math.isfinite(value):
        raise ValueError(f"metric field {field!r} must be finite, got {value}")
    return value


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    warmup_updates: int,
    max_updates: int = 0,
    include_eval: bool = False,
) -> dict[str, Any]:
    if warmup_updates < 0:
        raise ValueError("warmup_updates must be non-negative")
    if max_updates < 0:
        raise ValueError("max_updates must be non-negative")

    eligible = [
        record
        for record in records
        if include_eval or float(record.get("eval_seconds", 0.0)) <= 0.0
    ]
    selected = eligible[warmup_updates:]
    if max_updates:
        selected = selected[:max_updates]
    if not selected:
        raise ValueError("no PPO updates remain after filtering and warm-up")

    summary: dict[str, Any] = {
        "sample_count": len(selected),
        "first_update": int(selected[0]["update"]),
        "last_update": int(selected[-1]["update"]),
        "excluded_eval_count": len(records) - len(eligible),
        "discarded_warmup_count": min(warmup_updates, len(eligible)),
        "omitted_fields": [],
        "metrics": {},
    }
    fields = ("rollout_steps_per_second", *TIMING_FIELDS)
    for field in fields:
        if not all(field in record for record in selected):
            summary["omitted_fields"].append(field)
            continue
        values = [_finite_float(record, field) for record in selected]
        summary["metrics"][field] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _print_text(path: Path, summary: dict[str, Any]) -> None:
    print(f"metrics: {path}")
    print(
        "updates: "
        f"{summary['first_update']}..{summary['last_update']} "
        f"(n={summary['sample_count']}, "
        f"warmup={summary['discarded_warmup_count']}, "
        f"eval_excluded={summary['excluded_eval_count']})"
    )
    if summary["omitted_fields"]:
        print(f"omitted fields: {', '.join(summary['omitted_fields'])}")
    print(f"{'metric':34s} {'mean':>10s} {'p50':>10s} {'p95':>10s}")
    for field, values in summary["metrics"].items():
        print(
            f"{field:34s} "
            f"{values['mean']:10.3f} "
            f"{values['p50']:10.3f} "
            f"{values['p95']:10.3f}"
        )


def main() -> None:
    args = _parse_args()
    summary = summarize_records(
        load_metrics(args.metrics),
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        include_eval=args.include_eval,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_text(args.metrics, summary)


if __name__ == "__main__":
    main()
