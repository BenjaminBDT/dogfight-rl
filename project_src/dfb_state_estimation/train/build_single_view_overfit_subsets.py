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
class _ViewCandidate:
    entry: dict
    active_view: str
    target_area: int


def _build_candidates(entries: list[dict]) -> tuple[list[_ViewCandidate], list[_ViewCandidate]]:
    front: list[_ViewCandidate] = []
    rear: list[_ViewCandidate] = []
    for entry in entries:
        front_area = int(entry.get("front_target_area", 0))
        rear_area = int(entry.get("rear_target_area", 0))
        if front_area > 0:
            front.append(_ViewCandidate(entry=entry, active_view="front", target_area=front_area))
        if rear_area > 0:
            rear.append(_ViewCandidate(entry=entry, active_view="rear", target_area=rear_area))
    front.sort(key=lambda item: item.target_area, reverse=True)
    rear.sort(key=lambda item: item.target_area, reverse=True)
    return front, rear


def _copy_single_view_sample(
    *,
    source_root: Path,
    candidate: _ViewCandidate,
    destination_root: Path,
    sample_index: int,
    width: int,
    height: int,
) -> dict:
    sample_dir_name = f"sample_{sample_index:06d}"
    sample_root = destination_root / sample_dir_name
    sample_root.mkdir(parents=True, exist_ok=True)

    source_dir = source_root / candidate.entry["sample_dir"]
    source_meta = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    active_view = candidate.active_view
    inactive_view = "rear" if active_view == "front" else "front"

    shutil.copy2(source_dir / f"{active_view}_rgb.ppm", sample_root / f"{active_view}_rgb.ppm")
    shutil.copy2(
        source_dir / f"{active_view}_segmentation.pgm",
        sample_root / f"{active_view}_segmentation.pgm",
    )
    _write_zero_ppm(sample_root / f"{inactive_view}_rgb.ppm", width=width, height=height)
    _write_zero_pgm(sample_root / f"{inactive_view}_segmentation.pgm", width=width, height=height)

    for view_name in ("front", "rear"):
        shutil.copy2(
            sample_root / f"{view_name}_rgb.ppm",
            destination_root / "rgb" / f"{view_name}_{sample_dir_name}.ppm",
        )
        shutil.copy2(
            sample_root / f"{view_name}_segmentation.pgm",
            destination_root / "seg" / f"{view_name}_{sample_dir_name}.pgm",
        )

    new_meta = dict(source_meta)
    new_meta["sample_dir"] = sample_dir_name
    new_meta["active_view"] = active_view
    new_meta["active_target_area"] = candidate.target_area
    new_meta["inactive_view"] = inactive_view
    if active_view == "front":
        new_meta["rear_target_area"] = 0
    else:
        new_meta["front_target_area"] = 0
    (sample_root / "metadata.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination_root / "metadata" / f"{sample_dir_name}.json").write_text(
        json.dumps(new_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    new_entry = dict(candidate.entry)
    new_entry["sample_dir"] = sample_dir_name
    new_entry["active_view"] = active_view
    new_entry["active_target_area"] = candidate.target_area
    if active_view == "front":
        new_entry["rear_target_area"] = 0
        new_entry["actual_visibility_bucket"] = "front_only"
        new_entry["selected_target_area"] = int(candidate.entry.get("front_target_area", 0))
    else:
        new_entry["front_target_area"] = 0
        new_entry["actual_visibility_bucket"] = "rear_only"
        new_entry["selected_target_area"] = int(candidate.entry.get("rear_target_area", 0))
    return new_entry


def _build_subset(
    *,
    source_root: Path,
    base_manifest: dict,
    front_candidates: list[_ViewCandidate],
    rear_candidates: list[_ViewCandidate],
    destination_root: Path,
    subset_size: int,
    width: int,
    height: int,
) -> dict:
    if subset_size % 2 != 0:
        raise ValueError("subset_size must be even")
    if destination_root.exists():
        shutil.rmtree(destination_root)
    for subdir in ("rgb", "seg", "metadata"):
        (destination_root / subdir).mkdir(parents=True, exist_ok=True)

    per_view = subset_size // 2
    selected = front_candidates[:per_view] + rear_candidates[:per_view]
    selected.sort(key=lambda item: (item.active_view, -item.target_area, item.entry["sample_dir"]))

    entries: list[dict] = []
    for index, candidate in enumerate(selected):
        entries.append(
            _copy_single_view_sample(
                source_root=source_root,
                candidate=candidate,
                destination_root=destination_root,
                sample_index=index,
                width=width,
                height=height,
            )
        )

    manifest = dict(base_manifest)
    manifest["entries"] = entries
    manifest["entries_count"] = len(entries)
    manifest["single_view_mode"] = True
    manifest["source_dataset_root"] = str(source_root.resolve())
    (destination_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "entries": len(entries),
        "front_entries": sum(1 for entry in entries if entry["active_view"] == "front"),
        "rear_entries": sum(1 for entry in entries if entry["active_view"] == "rear"),
        "active_target_area_min": min(entry["active_target_area"] for entry in entries),
        "active_target_area_max": max(entry["active_target_area"] for entry in entries),
    }
    (destination_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-view synthetic overfit subsets")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256, 512],
    )
    args = parser.parse_args()

    source_root: Path = args.source_root
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    width, height = _ppm_size(source_root / "rgb" / f"front_{manifest['entries'][0]['sample_dir']}.ppm")
    front_candidates, rear_candidates = _build_candidates(manifest["entries"])

    results: dict[str, dict] = {}
    for subset_size in args.subset_sizes:
        name = f"overfit_{subset_size:03d}"
        destination_root = args.output_root / name
        results[name] = _build_subset(
            source_root=source_root,
            base_manifest=manifest,
            front_candidates=front_candidates,
            rear_candidates=rear_candidates,
            destination_root=destination_root,
            subset_size=subset_size,
            width=width,
            height=height,
        )

    (args.output_root / "single_view_overfit_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
