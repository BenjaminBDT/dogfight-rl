from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge synthetic segmentation datasets into a single mixed-role dataset."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("input_roots", nargs="+", type=Path)
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


def _load_manifest(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_netpbm_payload_size(path: Path) -> tuple[bytes, int, int, int, int]:
    data = path.read_bytes()
    length = len(data)
    cursor = 0

    def _next_token() -> bytes:
        nonlocal cursor
        while cursor < length:
            byte = data[cursor]
            if byte == 35:
                while cursor < length and data[cursor] not in (10, 13):
                    cursor += 1
            elif chr(byte).isspace():
                cursor += 1
            else:
                break
        start = cursor
        while cursor < length and not chr(data[cursor]).isspace():
            cursor += 1
        if start == cursor:
            raise ValueError(f"failed to parse netpbm token from {path}")
        return data[start:cursor]

    magic = _next_token()
    width = int(_next_token())
    height = int(_next_token())
    max_value = int(_next_token())
    while cursor < length and chr(data[cursor]).isspace():
        cursor += 1
    return magic, width, height, max_value, len(data[cursor:])


def _sample_is_valid(sample_root: Path) -> bool:
    checks = (
        ("front_rgb.ppm", b"P6", 3),
        ("rear_rgb.ppm", b"P6", 3),
        ("front_segmentation.pgm", b"P5", 1),
        ("rear_segmentation.pgm", b"P5", 1),
    )
    for name, expected_magic, channels in checks:
        path = sample_root / name
        magic, width, height, max_value, payload_size = _read_netpbm_payload_size(path)
        if magic != expected_magic:
            return False
        if max_value != 255:
            return False
        if payload_size != width * height * channels:
            return False
    return True


def main() -> None:
    args = _build_parser().parse_args()
    output_root = args.output_root.resolve()
    input_roots = [root.resolve() for root in args.input_roots]

    manifests = [(root, _load_manifest(root)) for root in input_roots]
    first_payload = manifests[0][1]
    width = int(first_payload["width"])
    height = int(first_payload["height"])
    semantic_label_mode = str(first_payload.get("semantic_label_mode", "strict"))

    for root, payload in manifests[1:]:
        if int(payload["width"]) != width or int(payload["height"]) != height:
            raise ValueError(f"incompatible resolution in {root}")
        if str(payload.get("semantic_label_mode", "strict")) != semantic_label_mode:
            raise ValueError(f"incompatible semantic_label_mode in {root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir_name in ("rgb", "seg_color", "metadata"):
        (output_root / subdir_name).mkdir(parents=True, exist_ok=True)

    merged_entries: list[dict[str, object]] = []
    merged_sources: list[str] = []
    skipped_invalid_samples = 0
    sample_index = 0

    for root, payload in manifests:
        observed_role = str(payload["observed_role"])
        target_role = str(payload["target_role"])
        merged_sources.append(str(root))
        for entry in payload["entries"]:
            src_sample_dir = str(entry["sample_dir"])
            src_sample_root = root / src_sample_dir
            if not _sample_is_valid(src_sample_root):
                skipped_invalid_samples += 1
                continue
            dst_sample_dir = f"sample_{sample_index:06d}"
            dst_sample_root = output_root / dst_sample_dir
            _materialize_path(src_sample_root, dst_sample_root, copy_mode=args.copy_mode)

            for stem in ("front", "rear"):
                rgb_src = root / "rgb" / f"{stem}_{src_sample_dir}.ppm"
                seg_src = root / "seg_color" / f"{stem}_{src_sample_dir}.ppm"
                metadata_src = root / "metadata" / f"{src_sample_dir}.json"
                rgb_dst = output_root / "rgb" / f"{stem}_{dst_sample_dir}.ppm"
                seg_dst = output_root / "seg_color" / f"{stem}_{dst_sample_dir}.ppm"
                metadata_dst = output_root / "metadata" / f"{dst_sample_dir}.json"
                if stem == "front":
                    _materialize_path(metadata_src, metadata_dst, copy_mode=args.copy_mode)
                _materialize_path(rgb_src, rgb_dst, copy_mode=args.copy_mode)
                _materialize_path(seg_src, seg_dst, copy_mode=args.copy_mode)

            merged_entry = dict(entry)
            merged_entry["sample_dir"] = dst_sample_dir
            merged_entry["observed_role"] = str(entry.get("observed_role", observed_role))
            merged_entry["target_role"] = str(entry.get("target_role", target_role))
            merged_entry["source_dataset_root"] = str(root)
            merged_entries.append(merged_entry)
            sample_index += 1

    merged_payload = {
        "observed_role": "mixed",
        "target_role": "mixed",
        "width": width,
        "height": height,
        "semantic_label_mode": semantic_label_mode,
        "merged_from": merged_sources,
        "copy_mode": args.copy_mode,
        "entries": merged_entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "input_roots": merged_sources,
        "entries": len(merged_entries),
        "skipped_invalid_samples": skipped_invalid_samples,
    }
    (output_root / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
