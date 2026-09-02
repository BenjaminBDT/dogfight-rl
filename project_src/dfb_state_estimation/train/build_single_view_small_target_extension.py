from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy_sample_tree(source_root: Path, sample_dir: str, destination_root: Path, new_sample_dir: str) -> dict:
    src = source_root / sample_dir
    dst = destination_root / new_sample_dir
    shutil.copytree(src, dst)
    metadata = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
    metadata["sample_dir"] = new_sample_dir
    (dst / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _infer_active_view_and_area(clean_root: Path, entry: dict) -> tuple[str, int]:
    metadata = json.loads((clean_root / entry["sample_dir"] / "metadata.json").read_text(encoding="utf-8"))
    front_area = int(metadata.get("front_target_area", entry.get("front_target_area", 0)))
    rear_area = int(metadata.get("rear_target_area", entry.get("rear_target_area", 0)))
    if front_area > rear_area and front_area > 0:
        return "front", front_area
    if rear_area > front_area and rear_area > 0:
        return "rear", rear_area
    raise ValueError(f"failed to infer active view for {entry['sample_dir']}")


def _pick_small_entries(clean_root: Path, manifest: dict, *, per_view: int) -> list[dict]:
    front: list[tuple[dict, int]] = []
    rear: list[tuple[dict, int]] = []
    for entry in manifest["entries"]:
        if entry.get("actual_area_bucket") != "px10_to19":
            continue
        active_view, active_area = _infer_active_view_and_area(clean_root, entry)
        if active_view == "front":
            front.append((entry, active_area))
        else:
            rear.append((entry, active_area))
    front.sort(key=lambda item: item[1], reverse=True)
    rear.sort(key=lambda item: item[1], reverse=True)
    selected = front[:per_view] + rear[:per_view]
    return [entry for entry, _ in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend a single-view overfit subset with a very small number of small-target samples.")
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--small-per-view", type=int, default=2)
    args = parser.parse_args()

    base_manifest = json.loads((args.base_root / "manifest.json").read_text(encoding="utf-8"))
    clean_manifest = json.loads((args.clean_root / "manifest.json").read_text(encoding="utf-8"))
    selected_small_entries = _pick_small_entries(
        args.clean_root,
        clean_manifest,
        per_view=int(args.small_per_view),
    )

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    new_entries: list[dict] = []
    sample_index = 0

    for entry in base_manifest["entries"]:
        sample_dir = f"sample_{sample_index:06d}"
        metadata = _copy_sample_tree(args.base_root, entry["sample_dir"], args.output_root, sample_dir)
        new_entry = dict(entry)
        new_entry["sample_dir"] = sample_dir
        new_entry["active_view"] = metadata["active_view"]
        new_entry["active_target_area"] = int(metadata["active_target_area"])
        new_entries.append(new_entry)
        sample_index += 1

    for entry in selected_small_entries:
        sample_dir = f"sample_{sample_index:06d}"
        metadata = _copy_sample_tree(args.clean_root, entry["sample_dir"], args.output_root, sample_dir)
        active_view, active_area = _infer_active_view_and_area(args.clean_root, entry)
        new_entry = dict(entry)
        new_entry["sample_dir"] = sample_dir
        new_entry["active_view"] = active_view
        new_entry["active_target_area"] = active_area
        new_entries.append(new_entry)
        sample_index += 1

    manifest = dict(base_manifest)
    manifest["entries"] = new_entries
    manifest["entries_count"] = len(new_entries)
    manifest["source_dataset_root"] = str(args.base_root.resolve())
    manifest["small_extension_source_root"] = str(args.clean_root.resolve())
    manifest["small_extension_per_view"] = int(args.small_per_view)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "entries": len(new_entries),
        "base_entries": len(base_manifest["entries"]),
        "small_extension_entries": len(selected_small_entries),
        "front_entries": sum(1 for entry in new_entries if entry["active_view"] == "front"),
        "rear_entries": sum(1 for entry in new_entries if entry["active_view"] == "rear"),
        "active_target_area_min": min(int(entry["active_target_area"]) for entry in new_entries),
        "active_target_area_max": max(int(entry["active_target_area"]) for entry in new_entries),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
