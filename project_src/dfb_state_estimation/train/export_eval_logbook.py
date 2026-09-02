from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dfb_state_estimation.train.eval_runner import STAGES, _format_summary_text, run_eval_stage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export multi-stage eval snapshots and append logbook entries."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages or 'all'.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    return parser


def _resolve_stages(spec: str) -> list[str]:
    if spec == "all":
        return list(STAGES)
    stages = [item.strip() for item in spec.split(",") if item.strip()]
    invalid = [item for item in stages if item not in STAGES]
    if invalid:
        raise ValueError(f"unsupported stages: {', '.join(invalid)}")
    return stages


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = _build_parser().parse_args()
    stages = _resolve_stages(args.stages)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_rows: list[dict[str, Any]] = []
    comparison_payload: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "num_samples": args.num_samples,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "tag": args.tag,
        "stages": {},
    }
    timestamp = datetime.now(timezone.utc).isoformat()

    for stage in stages:
        result = run_eval_stage(
            dataset_root=args.dataset_root,
            stage=stage,
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        stage_dir = args.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_text(stage_dir / "metrics.json", metrics_json)
        _write_text(stage_dir / "summary.txt", _format_summary_text(result))
        comparison_payload["stages"][stage] = result

        row: dict[str, Any] = {
            "timestamp_utc": timestamp,
            "tag": args.tag,
            "stage": stage,
            "dataset_root": str(args.dataset_root),
            "num_samples": result["num_samples"],
            "max_steps": args.max_steps,
            "seed": args.seed,
        }
        row.update(result["metrics"])
        snapshot_rows.append(row)

    _write_text(
        args.output_dir / "comparison.json",
        json.dumps(comparison_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(args.output_dir / "comparison.csv", snapshot_rows)

    log_jsonl = args.output_dir / "metrics_log.jsonl"
    _append_jsonl(log_jsonl, snapshot_rows)
    _write_csv(args.output_dir / "metrics_log.csv", _load_jsonl(log_jsonl))

    print(json.dumps({"output_dir": str(args.output_dir), "stages": stages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
