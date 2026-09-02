#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create small audio-only subset indexes from an existing authoritative "
            "audio-only recordings dataset without copying episode artifacts."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("runs/dfb_state_estimation/audio_only_recordings_dual_observer_min20s_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 2048],
    )
    return parser.parse_args()


def _select_evenly(entries: list[str], subset_size: int) -> list[str]:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    if subset_size >= len(entries):
        return list(entries)
    if subset_size == 1:
        return [entries[0]]
    indices = []
    last_index = len(entries) - 1
    for i in range(subset_size):
        index = round(i * last_index / (subset_size - 1))
        indices.append(index)
    return [entries[index] for index in indices]


def main() -> None:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else dataset_root.with_name(dataset_root.name + "_subsets")
    )
    manifest_path = dataset_root / "manifest.json"
    entries_path = dataset_root / "entries.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = entries_path.read_text(encoding="utf-8").splitlines()

    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "source_dataset_root": str(dataset_root),
        "entry_count": len(entries),
        "subsets": [],
    }

    for subset_size in args.subset_sizes:
        subset_name = f"overfit_{subset_size:04d}"
        subset_root = output_root / subset_name
        subset_root.mkdir(parents=True, exist_ok=True)
        subset_entries = _select_evenly(entries, subset_size)
        subset_manifest = dict(manifest)
        subset_manifest["dataset_root"] = str(subset_root)
        subset_manifest["entries_jsonl"] = str(subset_root / "entries.jsonl")
        subset_manifest["entry_count"] = len(subset_entries)
        subset_manifest["subset_of"] = str(dataset_root)
        subset_manifest["subset_name"] = subset_name
        (subset_root / "entries.jsonl").write_text(
            "\n".join(subset_entries) + "\n",
            encoding="utf-8",
        )
        (subset_root / "manifest.json").write_text(
            json.dumps(subset_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["subsets"].append(
            {
                "name": subset_name,
                "root": str(subset_root),
                "entry_count": len(subset_entries),
            }
        )

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
