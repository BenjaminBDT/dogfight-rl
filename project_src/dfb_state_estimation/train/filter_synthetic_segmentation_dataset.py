from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter a synthetic segmentation dataset into a target-positive subset."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-selected-target-area", type=int, default=5)
    parser.add_argument("--max-none-samples", type=int, default=24)
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


def main() -> None:
    args = _build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = input_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[dict[str, object]] = payload["entries"]

    positive_entries = [
        entry
        for entry in entries
        if str(entry["actual_visibility_bucket"]) != "None"
        and int(entry["selected_target_area"]) >= args.min_selected_target_area
    ]
    none_entries = [
        entry for entry in entries if str(entry["actual_visibility_bucket"]) == "None"
    ]
    kept_none_entries = none_entries[: args.max_none_samples]
    kept_entries = positive_entries + kept_none_entries

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir_name in ("rgb", "seg_color", "metadata"):
        (output_root / subdir_name).mkdir(parents=True, exist_ok=True)

    for new_index, entry in enumerate(kept_entries):
        sample_dir = str(entry["sample_dir"])
        src_sample_root = input_root / sample_dir
        dst_sample_dir = f"sample_{new_index:06d}"
        dst_sample_root = output_root / dst_sample_dir
        _materialize_path(src_sample_root, dst_sample_root, copy_mode=args.copy_mode)

        for stem in ("front", "rear"):
            rgb_src = input_root / "rgb" / f"{stem}_{sample_dir}.ppm"
            seg_src = input_root / "seg_color" / f"{stem}_{sample_dir}.ppm"
            metadata_src = input_root / "metadata" / f"{sample_dir}.json"
            rgb_dst = output_root / "rgb" / f"{stem}_{dst_sample_dir}.ppm"
            seg_dst = output_root / "seg_color" / f"{stem}_{dst_sample_dir}.ppm"
            metadata_dst = output_root / "metadata" / f"{dst_sample_dir}.json"
            if stem == "front":
                _materialize_path(metadata_src, metadata_dst, copy_mode=args.copy_mode)
            _materialize_path(rgb_src, rgb_dst, copy_mode=args.copy_mode)
            _materialize_path(seg_src, seg_dst, copy_mode=args.copy_mode)

        entry["sample_dir"] = dst_sample_dir

    output_payload = dict(payload)
    output_payload["entries"] = kept_entries
    output_payload["filtered_from"] = str(input_root)
    output_payload["filter"] = {
        "min_selected_target_area": args.min_selected_target_area,
        "max_none_samples": args.max_none_samples,
        "copy_mode": args.copy_mode,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_entries": len(entries),
        "kept_entries": len(kept_entries),
        "kept_positive_entries": len(positive_entries),
        "kept_none_entries": len(kept_none_entries),
    }
    (output_root / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
