from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


STAGE_DEFINITIONS: dict[str, dict[str, object]] = {
    "stage1_large_bootstrap": {
        "positive_buckets": ["px50_to99", "px100_to199", "px200_plus"],
        "max_none_samples": 64,
    },
    "stage2_medium_main": {
        "positive_buckets": ["px20_to49", "px50_to99", "px100_to199", "px200_plus"],
        "max_none_samples": 96,
    },
    "stage3_small_introduce": {
        "positive_buckets": [
            "px10_to19",
            "px20_to49",
            "px50_to99",
            "px100_to199",
            "px200_plus",
        ],
        "max_none_samples": 128,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build three curriculum subsets from a merged synthetic segmentation dataset."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy-mode",
        choices=["symlink", "copy"],
        default="symlink",
    )
    return parser


def _materialize_path(src: Path, dst: Path, *, copy_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy_mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _write_subset(
    *,
    input_root: Path,
    output_root: Path,
    payload: dict,
    entries: list[dict[str, object]],
    stage_name: str,
    stage_spec: dict[str, object],
    copy_mode: str,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir_name in ("rgb", "seg_color", "metadata"):
        (output_root / subdir_name).mkdir(parents=True, exist_ok=True)

    remapped_entries: list[dict[str, object]] = []
    for new_index, entry in enumerate(entries):
        sample_dir = str(entry["sample_dir"])
        dst_sample_dir = f"sample_{new_index:06d}"
        src_sample_root = input_root / sample_dir
        dst_sample_root = output_root / dst_sample_dir
        _materialize_path(src_sample_root, dst_sample_root, copy_mode=copy_mode)

        for stem in ("front", "rear"):
            rgb_src = input_root / "rgb" / f"{stem}_{sample_dir}.ppm"
            seg_src = input_root / "seg_color" / f"{stem}_{sample_dir}.ppm"
            metadata_src = input_root / "metadata" / f"{sample_dir}.json"
            rgb_dst = output_root / "rgb" / f"{stem}_{dst_sample_dir}.ppm"
            seg_dst = output_root / "seg_color" / f"{stem}_{dst_sample_dir}.ppm"
            metadata_dst = output_root / "metadata" / f"{dst_sample_dir}.json"
            if stem == "front":
                _materialize_path(metadata_src, metadata_dst, copy_mode=copy_mode)
            _materialize_path(rgb_src, rgb_dst, copy_mode=copy_mode)
            _materialize_path(seg_src, seg_dst, copy_mode=copy_mode)

        remapped_entry = copy.deepcopy(entry)
        remapped_entry["sample_dir"] = dst_sample_dir
        remapped_entries.append(remapped_entry)

    output_payload = dict(payload)
    output_payload["entries"] = remapped_entries
    output_payload["curriculum_stage"] = stage_name
    output_payload["curriculum_stage_spec"] = stage_spec
    output_payload["subset_from"] = str(input_root)
    output_payload["copy_mode"] = copy_mode
    (output_root / "manifest.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    positives = [
        entry for entry in remapped_entries if str(entry["actual_visibility_bucket"]) != "None"
    ]
    negatives = [
        entry for entry in remapped_entries if str(entry["actual_visibility_bucket"]) == "None"
    ]
    area_counts: dict[str, int] = {}
    for entry in remapped_entries:
        bucket = str(entry.get("actual_area_bucket", "None"))
        area_counts[bucket] = area_counts.get(bucket, 0) + 1

    summary = {
        "stage_name": stage_name,
        "entries": len(remapped_entries),
        "positive_entries": len(positives),
        "none_entries": len(negatives),
        "area_bucket_counts": area_counts,
    }
    (output_root / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    payload = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = payload["entries"]

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "input_root": str(input_root),
        "stages": {},
    }

    for stage_name, stage_spec in STAGE_DEFINITIONS.items():
        positive_buckets = set(stage_spec["positive_buckets"])
        max_none_samples = int(stage_spec["max_none_samples"])
        positive_entries = [
            entry
            for entry in entries
            if str(entry["actual_visibility_bucket"]) != "None"
            and str(entry["actual_area_bucket"]) in positive_buckets
        ]
        none_entries = [
            entry for entry in entries if str(entry["actual_visibility_bucket"]) == "None"
        ][:max_none_samples]
        stage_entries = positive_entries + none_entries
        stage_output_root = output_root / stage_name
        _write_subset(
            input_root=input_root,
            output_root=stage_output_root,
            payload=payload,
            entries=stage_entries,
            stage_name=stage_name,
            stage_spec=stage_spec,
            copy_mode=args.copy_mode,
        )
        summary["stages"][stage_name] = {
            "entries": len(stage_entries),
            "positive_entries": len(positive_entries),
            "none_entries": len(none_entries),
            "positive_buckets": list(stage_spec["positive_buckets"]),
        }

    (output_root / "curriculum_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
