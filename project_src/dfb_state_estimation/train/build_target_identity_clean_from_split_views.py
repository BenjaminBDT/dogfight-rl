from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


ROLE_TO_TARGET = {
    "fighter1": "fighter2",
    "fighter2": "fighter1",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build target-identity clean datasets from split-view synthetic roots."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    return parser


def _materialize(src: Path, dst: Path, *, copy_mode: str) -> None:
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


def _load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_clean_dataset(
    *,
    split_roots: list[Path],
    output_root: Path,
    observed_role: str,
    target_role: str,
    copy_mode: str,
) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "seg_color", "metadata"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    base_manifest = _load_manifest(split_roots[0])
    merged_entries: list[dict] = []
    next_index = 0

    for split_root in split_roots:
        payload = _load_manifest(split_root)
        for entry in payload["entries"]:
            src_sample_dir = str(entry["sample_dir"])
            dst_sample_dir = f"sample_{next_index:06d}"
            next_index += 1

            src_sample_root = split_root / src_sample_dir
            dst_sample_root = output_root / dst_sample_dir
            _materialize(src_sample_root, dst_sample_root, copy_mode=copy_mode)

            for stem in ("front", "rear"):
                rgb_src = split_root / "rgb" / f"{stem}_{src_sample_dir}.ppm"
                seg_src = split_root / "seg_color" / f"{stem}_{src_sample_dir}.ppm"
                metadata_src = split_root / "metadata" / f"{src_sample_dir}.json"
                if not rgb_src.exists():
                    rgb_src = src_sample_root / f"{stem}_rgb.ppm"
                if not seg_src.exists():
                    seg_src = src_sample_root / f"{stem}_segmentation_color.ppm"
                if not metadata_src.exists():
                    metadata_src = src_sample_root / "metadata.json"
                _materialize(rgb_src, output_root / "rgb" / f"{stem}_{dst_sample_dir}.ppm", copy_mode=copy_mode)
                _materialize(seg_src, output_root / "seg_color" / f"{stem}_{dst_sample_dir}.ppm", copy_mode=copy_mode)
                if stem == "front":
                    _materialize(
                        metadata_src,
                        output_root / "metadata" / f"{dst_sample_dir}.json",
                        copy_mode=copy_mode,
                    )

            remapped_entry = copy.deepcopy(entry)
            remapped_entry["sample_dir"] = dst_sample_dir
            remapped_entry["observed_role"] = observed_role
            remapped_entry["target_role"] = target_role
            merged_entries.append(remapped_entry)

    manifest = {
        "observed_role": observed_role,
        "target_role": target_role,
        "width": base_manifest["width"],
        "height": base_manifest["height"],
        "band_positions_per_bucket": base_manifest["band_positions_per_bucket"],
        "uniform_positions_per_bucket": base_manifest["uniform_positions_per_bucket"],
        "orientations_per_position": base_manifest["orientations_per_position"],
        "seed": base_manifest["seed"],
        "semantic_label_mode": base_manifest["semantic_label_mode"],
        "dataset_kind": "target_identity_clean",
        "source_dataset_root": str(split_roots[0].parent.parent.resolve()),
        "entries": merged_entries,
        "entries_count": len(merged_entries),
        "copy_mode": copy_mode,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "observed_role": observed_role,
        "target_role": target_role,
        "entries": len(merged_entries),
    }


def main() -> None:
    args = _build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for observed_role, target_role in ROLE_TO_TARGET.items():
        split_roots = [
            input_root / observed_role / "front",
            input_root / observed_role / "rear",
        ]
        clean_root = output_root / f"target_{target_role}" / "clean"
        clean_root.parent.mkdir(parents=True, exist_ok=True)
        summary[target_role] = _write_clean_dataset(
            split_roots=split_roots,
            output_root=clean_root,
            observed_role=observed_role,
            target_role=target_role,
            copy_mode=args.copy_mode,
        )

    (output_root / "target_identity_clean_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
