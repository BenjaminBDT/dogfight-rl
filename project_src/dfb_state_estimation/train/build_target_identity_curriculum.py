from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict, deque
from pathlib import Path


STAGE_DEFINITIONS: dict[str, dict[str, object]] = {
    "stage1_seed_large": {
        "positive_buckets": ["px100_to199", "px200_plus"],
        "max_samples": 512,
    },
    "stage2_large_bootstrap": {
        "positive_buckets": ["px50_to99", "px100_to199", "px200_plus"],
    },
    "stage3_medium_main": {
        "positive_buckets": ["px20_to49", "px50_to99", "px100_to199", "px200_plus"],
    },
    "stage4_small_introduce": {
        "positive_buckets": ["px10_to19", "px20_to49", "px50_to99", "px100_to199", "px200_plus"],
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean target-identity synthetic datasets and curriculum subsets."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
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


def _write_dataset(
    *,
    input_root: Path,
    output_root: Path,
    payload: dict,
    entries: list[dict[str, object]],
    copy_mode: str,
    extra_fields: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir_name in ("rgb", "seg_color", "metadata"):
        (output_root / subdir_name).mkdir(parents=True, exist_ok=True)

    remapped_entries: list[dict[str, object]] = []
    for new_index, entry in enumerate(entries):
        src_sample_dir = str(entry["sample_dir"])
        dst_sample_dir = f"sample_{new_index:06d}"
        src_sample_root = input_root / src_sample_dir
        dst_sample_root = output_root / dst_sample_dir
        _materialize_path(src_sample_root, dst_sample_root, copy_mode=copy_mode)

        for stem in ("front", "rear"):
            rgb_src = input_root / "rgb" / f"{stem}_{src_sample_dir}.ppm"
            seg_src = input_root / "seg_color" / f"{stem}_{src_sample_dir}.ppm"
            if not rgb_src.exists():
                rgb_src = src_sample_root / f"{stem}_rgb.ppm"
            if not seg_src.exists():
                seg_src = src_sample_root / f"{stem}_segmentation_color.ppm"
            metadata_src = input_root / "metadata" / f"{src_sample_dir}.json"
            if not metadata_src.exists():
                metadata_src = src_sample_root / "metadata.json"
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
    output_payload["copy_mode"] = copy_mode
    if extra_fields:
        output_payload.update(extra_fields)
    (output_root / "manifest.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return remapped_entries


def _area_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        bucket = str(entry.get("actual_area_bucket", "None"))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _visibility_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        bucket = str(entry.get("actual_visibility_bucket", "None"))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _balanced_take(entries: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], deque[dict[str, object]]] = defaultdict(deque)
    for entry in entries:
        key = (
            str(entry.get("actual_area_bucket", "None")),
            str(entry.get("actual_visibility_bucket", "None")),
        )
        groups[key].append(entry)
    ordered_keys = sorted(groups.keys())
    selected: list[dict[str, object]] = []
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            queue = groups[key]
            if not queue:
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def main() -> None:
    args = _build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    payload = _load_manifest(input_root)
    entries: list[dict[str, object]] = payload["entries"]

    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        by_target[str(entry["target_role"])].append(entry)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, object] = {
        "input_root": str(input_root),
        "copy_mode": args.copy_mode,
        "targets": {},
    }

    for target_role, target_entries in sorted(by_target.items()):
        observed_roles = {str(entry["observed_role"]) for entry in target_entries}
        if len(observed_roles) != 1:
            raise ValueError(
                f"expected exactly one observed_role for target_role={target_role}, got {observed_roles}"
            )
        observed_role = next(iter(observed_roles))
        target_root = output_root / f"target_{target_role}"
        clean_root = target_root / "clean"
        curriculum_root = target_root / "curriculum"
        curriculum_root.mkdir(parents=True, exist_ok=True)

        clean_payload = dict(payload)
        clean_payload["observed_role"] = observed_role
        clean_payload["target_role"] = target_role
        clean_payload["source_dataset_root"] = str(input_root)
        clean_entries = _write_dataset(
            input_root=input_root,
            output_root=clean_root,
            payload=clean_payload,
            entries=target_entries,
            copy_mode=args.copy_mode,
            extra_fields={"dataset_kind": "target_identity_clean"},
        )

        target_summary: dict[str, object] = {
            "observed_role": observed_role,
            "target_role": target_role,
            "clean_entries": len(clean_entries),
            "clean_area_counts": _area_counts(clean_entries),
            "clean_visibility_counts": _visibility_counts(clean_entries),
            "stages": {},
        }

        for stage_name, stage_spec in STAGE_DEFINITIONS.items():
            positive_buckets = set(stage_spec["positive_buckets"])
            stage_entries = [
                entry
                for entry in clean_entries
                if str(entry.get("actual_area_bucket", "None")) in positive_buckets
            ]
            max_samples = stage_spec.get("max_samples")
            if isinstance(max_samples, int):
                stage_entries = _balanced_take(stage_entries, max_samples)

            stage_root = curriculum_root / stage_name
            stage_payload = dict(clean_payload)
            stage_entries = _write_dataset(
                input_root=clean_root,
                output_root=stage_root,
                payload=stage_payload,
                entries=stage_entries,
                copy_mode=args.copy_mode,
                extra_fields={
                    "dataset_kind": "target_identity_curriculum",
                    "curriculum_stage": stage_name,
                    "curriculum_stage_spec": stage_spec,
                    "subset_from": str(clean_root),
                },
            )
            stage_summary = {
                "entries": len(stage_entries),
                "area_counts": _area_counts(stage_entries),
                "visibility_counts": _visibility_counts(stage_entries),
            }
            (stage_root / "subset_summary.json").write_text(
                json.dumps(stage_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            target_summary["stages"][stage_name] = stage_summary

        global_summary["targets"][target_role] = target_summary

    (output_root / "target_identity_curriculum_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(global_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
