from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


def _read_netpbm(path: Path, *, expected_magic: bytes) -> tuple[int, int, int, bytes]:
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
    if magic != expected_magic:
        raise ValueError(f"unexpected netpbm magic for {path}: {magic!r}")
    width = int(_next_token())
    height = int(_next_token())
    max_value = int(_next_token())
    while cursor < length and chr(data[cursor]).isspace():
        cursor += 1
    payload = data[cursor:]
    return width, height, max_value, payload


def _ppm_size(path: Path) -> tuple[int, int]:
    width, height, max_value, payload = _read_netpbm(path, expected_magic=b"P6")
    if max_value != 255:
        raise ValueError(f"unsupported ppm max value for {path}: {max_value}")
    expected_size = width * height * 3
    if len(payload) != expected_size:
        raise ValueError(
            f"unexpected ppm payload size for {path}: got {len(payload)}, expected {expected_size}"
        )
    return width, height


def _write_zero_ppm(path: Path, *, width: int, height: int) -> None:
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    payload = bytes(width * height * 3)
    path.write_bytes(header + payload)


def _write_zero_pgm(path: Path, *, width: int, height: int) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    payload = bytes(width * height)
    path.write_bytes(header + payload)


@dataclass(frozen=True)
class _NegativeCandidate:
    entry: dict
    active_view: str
    opposite_target_area: int


def _build_negative_candidates(entries: list[dict]) -> tuple[list[_NegativeCandidate], list[_NegativeCandidate]]:
    front: list[_NegativeCandidate] = []
    rear: list[_NegativeCandidate] = []
    for entry in entries:
        front_area = int(entry.get("front_target_area", 0))
        rear_area = int(entry.get("rear_target_area", 0))
        if front_area == 0 and rear_area > 0:
            front.append(
                _NegativeCandidate(entry=entry, active_view="front", opposite_target_area=rear_area)
            )
        if rear_area == 0 and front_area > 0:
            rear.append(
                _NegativeCandidate(entry=entry, active_view="rear", opposite_target_area=front_area)
            )
    front.sort(key=lambda item: item.opposite_target_area, reverse=True)
    rear.sort(key=lambda item: item.opposite_target_area, reverse=True)
    return front, rear


def _copy_positive_sample(
    *,
    source_root: Path,
    entry: dict,
    destination_root: Path,
    sample_index: int,
) -> dict:
    sample_dir_name = f"sample_{sample_index:06d}"
    sample_root = destination_root / sample_dir_name
    sample_root.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / entry["sample_dir"]

    for filename in ("front_rgb.ppm", "rear_rgb.ppm", "front_segmentation.pgm", "rear_segmentation.pgm"):
        shutil.copy2(source_dir / filename, sample_root / filename)
    shutil.copy2(source_dir / "metadata.json", sample_root / "metadata.json")

    new_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    new_meta["sample_dir"] = sample_dir_name
    (sample_root / "metadata.json").write_text(json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination_root / "metadata" / f"{sample_dir_name}.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for view_name in ("front", "rear"):
        shutil.copy2(sample_root / f"{view_name}_rgb.ppm", destination_root / "rgb" / f"{view_name}_{sample_dir_name}.ppm")
        shutil.copy2(sample_root / f"{view_name}_segmentation.pgm", destination_root / "seg" / f"{view_name}_{sample_dir_name}.pgm")

    new_entry = dict(entry)
    new_entry["sample_dir"] = sample_dir_name
    new_entry["replay_kind"] = "positive"
    return new_entry


def _copy_negative_sample(
    *,
    source_root: Path,
    candidate: _NegativeCandidate,
    destination_root: Path,
    sample_index: int,
    width: int,
    height: int,
) -> dict:
    sample_dir_name = f"sample_{sample_index:06d}"
    sample_root = destination_root / sample_dir_name
    sample_root.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / candidate.entry["sample_dir"]

    active_view = candidate.active_view
    inactive_view = "rear" if active_view == "front" else "front"

    shutil.copy2(source_dir / f"{active_view}_rgb.ppm", sample_root / f"{active_view}_rgb.ppm")
    shutil.copy2(source_dir / f"{active_view}_segmentation.pgm", sample_root / f"{active_view}_segmentation.pgm")
    _write_zero_ppm(sample_root / f"{inactive_view}_rgb.ppm", width=width, height=height)
    _write_zero_pgm(sample_root / f"{inactive_view}_segmentation.pgm", width=width, height=height)

    for view_name in ("front", "rear"):
        shutil.copy2(sample_root / f"{view_name}_rgb.ppm", destination_root / "rgb" / f"{view_name}_{sample_dir_name}.ppm")
        shutil.copy2(sample_root / f"{view_name}_segmentation.pgm", destination_root / "seg" / f"{view_name}_{sample_dir_name}.pgm")

    source_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    new_meta = dict(source_meta)
    new_meta["sample_dir"] = sample_dir_name
    new_meta["active_view"] = active_view
    new_meta["inactive_view"] = inactive_view
    new_meta["active_target_area"] = 0
    new_meta["hard_negative_replay"] = True
    if active_view == "front":
        new_meta["rear_target_area"] = 0
    else:
        new_meta["front_target_area"] = 0
    (sample_root / "metadata.json").write_text(json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (destination_root / "metadata" / f"{sample_dir_name}.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    new_entry = dict(candidate.entry)
    new_entry["sample_dir"] = sample_dir_name
    new_entry["active_view"] = active_view
    new_entry["active_target_area"] = 0
    new_entry["selected_target_area"] = 0
    new_entry["hard_negative_replay"] = True
    new_entry["replay_kind"] = "hard_negative"
    new_entry["actual_visibility_bucket"] = "none"
    if active_view == "front":
        new_entry["rear_target_area"] = 0
    else:
        new_entry["front_target_area"] = 0
    return new_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-view hard-negative replay subset")
    parser.add_argument("--positive-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--negative-count", type=int, default=8)
    args = parser.parse_args()

    positive_root: Path = args.positive_root
    clean_root: Path = args.clean_root
    output_root: Path = args.output_root

    positive_manifest = json.loads((positive_root / "manifest.json").read_text(encoding="utf-8"))
    clean_manifest = json.loads((clean_root / "manifest.json").read_text(encoding="utf-8"))
    width, height = _ppm_size(positive_root / "rgb" / f"front_{positive_manifest['entries'][0]['sample_dir']}.ppm")
    front_neg, rear_neg = _build_negative_candidates(clean_manifest["entries"])

    if args.negative_count % 2 != 0:
        raise ValueError("negative-count must be even")
    per_view_neg = args.negative_count // 2
    selected_negatives = front_neg[:per_view_neg] + rear_neg[:per_view_neg]

    if output_root.exists():
        shutil.rmtree(output_root)
    for subdir in ("rgb", "seg", "metadata"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for index, entry in enumerate(positive_manifest["entries"]):
        entries.append(
            _copy_positive_sample(
                source_root=positive_root,
                entry=entry,
                destination_root=output_root,
                sample_index=index,
            )
        )
    for offset, candidate in enumerate(selected_negatives, start=len(entries)):
        entries.append(
            _copy_negative_sample(
                source_root=clean_root,
                candidate=candidate,
                destination_root=output_root,
                sample_index=offset,
                width=width,
                height=height,
            )
        )

    manifest = dict(positive_manifest)
    manifest["entries"] = entries
    manifest["entries_count"] = len(entries)
    manifest["hard_negative_replay"] = {
        "enabled": True,
        "negative_entries": len(selected_negatives),
        "front_negative_entries": sum(1 for item in selected_negatives if item.active_view == "front"),
        "rear_negative_entries": sum(1 for item in selected_negatives if item.active_view == "rear"),
        "clean_source_root": str(clean_root.resolve()),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "positive_entries": len(positive_manifest["entries"]),
        "negative_entries": len(selected_negatives),
        "entries": len(entries),
        "front_negative_entries": sum(1 for item in selected_negatives if item.active_view == "front"),
        "rear_negative_entries": sum(1 for item in selected_negatives if item.active_view == "rear"),
        "negative_opposite_target_area_min": min(item.opposite_target_area for item in selected_negatives),
        "negative_opposite_target_area_max": max(item.opposite_target_area for item in selected_negatives),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
