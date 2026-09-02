from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Candidate:
    entry: dict
    active_view: str
    active_target_area: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_candidates(dataset_root: Path, manifest: dict) -> tuple[list[_Candidate], list[_Candidate]]:
    front: list[_Candidate] = []
    rear: list[_Candidate] = []
    labels_root = dataset_root / "labels"
    for entry in manifest["entries"]:
        label_path = dataset_root / entry["labels_path"]
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        front_area = int(payload.get("front_target_area", 0))
        rear_area = int(payload.get("rear_target_area", 0))
        if front_area <= 0 and rear_area <= 0:
            continue
        active_view = "front" if front_area >= rear_area else "rear"
        active_target_area = front_area if active_view == "front" else rear_area
        candidate = _Candidate(
            entry={
                "sample_dir": str(entry["sample_dir"]),
                "labels_path": str(entry["labels_path"]),
                "observed_role": str(entry["observed_role"]),
                "target_role": str(entry["target_role"]),
            },
            active_view=active_view,
            active_target_area=active_target_area,
        )
        if active_view == "front":
            front.append(candidate)
        else:
            rear.append(candidate)
    front.sort(key=lambda item: (-item.active_target_area, item.entry["sample_dir"]))
    rear.sort(key=lambda item: (-item.active_target_area, item.entry["sample_dir"]))
    return front, rear


def _select_candidates(
    front: list[_Candidate],
    rear: list[_Candidate],
    subset_size: int,
) -> list[_Candidate]:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    per_view = subset_size // 2
    selected_front = front[:per_view]
    selected_rear = rear[:per_view]
    selected = selected_front + selected_rear
    if len(selected) < subset_size:
        remainder = subset_size - len(selected)
        merged = front[per_view:] + rear[per_view:]
        merged.sort(key=lambda item: (-item.active_target_area, item.entry["sample_dir"]))
        selected.extend(merged[:remainder])
    if len(selected) != subset_size:
        raise ValueError(
            f"failed to select {subset_size} candidates; got {len(selected)} "
            f"(front={len(front)}, rear={len(rear)})"
        )
    return selected


def _copy_subset(
    *,
    source_root: Path,
    source_manifest: dict,
    destination_root: Path,
    selected: list[_Candidate],
) -> dict:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    (destination_root / "labels").mkdir(parents=True, exist_ok=True)
    (destination_root / "voting").mkdir(parents=True, exist_ok=True)

    for bundle_name in ("front.bin", "rear.bin"):
        shutil.copy2(source_root / "voting" / bundle_name, destination_root / "voting" / bundle_name)

    entries: list[dict] = []
    for candidate in selected:
        shutil.copy2(
            source_root / candidate.entry["labels_path"],
            destination_root / candidate.entry["labels_path"],
        )
        entries.append(dict(candidate.entry))

    subset_manifest = dict(source_manifest)
    subset_manifest["source_dataset_root"] = str(source_manifest["source_dataset_root"])
    subset_manifest["entries"] = entries
    subset_manifest["entries_count"] = len(entries)
    subset_manifest["subset_strategy"] = {
        "type": "active_view_balanced_top_target_area",
        "front_entries": sum(1 for item in selected if item.active_view == "front"),
        "rear_entries": sum(1 for item in selected if item.active_view == "rear"),
    }
    (destination_root / "manifest.json").write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "entries": len(entries),
        "front_entries": sum(1 for item in selected if item.active_view == "front"),
        "rear_entries": sum(1 for item in selected if item.active_view == "rear"),
        "active_target_area_min": min(item.active_target_area for item in selected),
        "active_target_area_max": max(item.active_target_area for item in selected),
    }
    (destination_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic single-step overfit subsets")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    args = parser.parse_args()

    source_root = args.source_root
    source_manifest = _load_manifest(source_root / "manifest.json")
    front, rear = _build_candidates(source_root, source_manifest)

    results: dict[str, dict] = {}
    for subset_size in args.subset_sizes:
        subset_name = f"overfit_{subset_size:03d}"
        selected = _select_candidates(front, rear, subset_size)
        results[subset_name] = _copy_subset(
            source_root=source_root,
            source_manifest=source_manifest,
            destination_root=args.output_root / subset_name,
            selected=selected,
        )

    (args.output_root / "single_step_overfit_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
