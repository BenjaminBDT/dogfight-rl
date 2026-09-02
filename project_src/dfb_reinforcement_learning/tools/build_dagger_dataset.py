from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from dfb_reinforcement_learning.policy_assets import (
    load_and_validate_policy_dataset,
)

CHUNK_SIZE = 256
AUXILIARY_KEYS = (
    "done",
    "did_hit",
    "got_hit",
    "did_fire",
    "self_out_of_bounds_seconds",
    "self_ceiling_recovery_seconds",
    "self_repair_elapsed_seconds",
    "self_destroyed",
    "enemy_destroyed",
    "self_health_state_norm",
    "enemy_health_state_norm",
    "self_gun_overheated",
    "self_gun_heat_norm",
    "winner_label",
    "target_distance",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append DAgger teacher labels to a parent policy dataset."
    )
    parser.add_argument("--base-dataset-root", required=True)
    parser.add_argument(
        "--dagger-collection-root",
        action="append",
        required=True,
        help="May be repeated to aggregate multiple DAgger collections.",
    )
    parser.add_argument("--dagger-source-name", default="dagger_teacher")
    parser.add_argument("--dagger-sample-weight", type=float, default=1.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_episode_arrays(
    episode_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    obs_dim: int,
) -> int:
    required_shapes = {
        "obs": (obs_dim,),
        "teacher_action_cont": (4,),
        "teacher_action_bin": (3,),
        "simulation_step_index": (),
        "timestamp": (),
    }
    if "obs" not in arrays:
        raise ValueError(f"DAgger episode missing obs: {episode_path}")
    step_count = int(arrays["obs"].shape[0])
    if step_count < 1:
        raise ValueError(f"DAgger episode is empty: {episode_path}")
    for key, trailing_shape in required_shapes.items():
        if key not in arrays:
            raise ValueError(f"DAgger episode missing {key}: {episode_path}")
        if arrays[key].shape != (step_count, *trailing_shape):
            raise ValueError(
                f"DAgger episode {key} shape mismatch in {episode_path}: "
                f"{arrays[key].shape}"
            )
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"DAgger episode {key} contains non-finite values: {episode_path}")
    for key in AUXILIARY_KEYS:
        if key not in arrays or arrays[key].shape[0] != step_count:
            raise ValueError(f"DAgger episode auxiliary field {key} is invalid: {episode_path}")
    return step_count


def build_dagger_dataset(
    *,
    base_dataset_root: Path,
    collection_roots: list[Path],
    output_dir: Path,
    dagger_source_name: str,
    dagger_sample_weight: float,
    force: bool,
) -> dict[str, Any]:
    dagger_source_name = dagger_source_name.strip()
    if not dagger_source_name:
        raise ValueError("dagger_source_name must be non-empty")
    if not np.isfinite(dagger_sample_weight) or dagger_sample_weight <= 0.0:
        raise ValueError("dagger_sample_weight must be finite and positive")
    base_contract, _ = load_and_validate_policy_dataset(base_dataset_root)
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(base_dataset_root, output_dir)

    collection_records: list[tuple[Path, dict[str, Any], str]] = []
    lineage_digest = hashlib.sha256(base_contract.dataset_id.encode("utf-8"))
    for collection_root in collection_roots:
        manifest_path = collection_root / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("schema") != "dfb.part3.dagger_collection.v1":
            raise ValueError(f"unsupported DAgger collection schema: {manifest_path}")
        if manifest.get("dataset_id") != base_contract.dataset_id:
            raise ValueError(
                f"DAgger collection {manifest_path} was generated from dataset "
                f"{manifest.get('dataset_id')!r}, expected {base_contract.dataset_id!r}"
            )
        manifest_hash = _sha256(manifest_path)
        lineage_digest.update(bytes.fromhex(manifest_hash))
        collection_records.append((collection_root, manifest, manifest_hash))

    dataset_id = f"dfb_part3_policy_dataset_dagger_{lineage_digest.hexdigest()[:16]}"
    meta_path = output_dir / "meta.json"
    meta = _read_json(meta_path)
    normalizer_path = output_dir / str(meta["obs_normalizer_path"])
    normalizer = _read_json(normalizer_path)
    normalizer["source_dataset_id"] = dataset_id
    constants = meta["constants"]
    conventions = constants.get("bc_demonstration_conventions")
    if conventions is None:
        sources = constants.pop("demonstration_sources")
        weights = constants.pop("bc_sample_weights")
        conventions = {
            role: {str(source): float(weights[role])}
            for role, source in sources.items()
        }
        constants["bc_demonstration_conventions"] = conventions
    if not isinstance(conventions, dict):
        raise ValueError("bc_demonstration_conventions must be an object")

    next_chunk_index = max(
        (int(chunk["chunk_index"]) for chunk in meta["chunks"]),
        default=-1,
    ) + 1
    fighter1_conventions = conventions.setdefault("fighter1", {})
    if not isinstance(fighter1_conventions, dict):
        raise ValueError("fighter1 BC demonstration conventions must be an object")
    existing_weight = fighter1_conventions.get(dagger_source_name)
    if existing_weight is not None and float(existing_weight) != dagger_sample_weight:
        raise ValueError(
            f"DAgger source {dagger_source_name!r} already has weight {existing_weight}"
        )
    fighter1_conventions[dagger_source_name] = dagger_sample_weight
    added_steps = 0
    added_episode_ids: list[str] = []
    dagger_episode_provenance: list[dict[str, Any]] = []

    for collection_root, manifest, manifest_hash in collection_records:
        for episode in manifest.get("episodes", []):
            if not isinstance(episode, dict):
                raise ValueError("DAgger collection episode entries must be objects")
            episode_id = str(episode["episode_id"])
            if episode_id in added_episode_ids or any(
                existing["episode_id"] == episode_id for existing in meta["episodes"]
            ):
                raise ValueError(f"duplicate DAgger episode ID: {episode_id}")
            episode_path = collection_root / str(episode["episode_file"])
            with np.load(episode_path) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
            step_count = _validate_episode_arrays(
                episode_path,
                arrays,
                obs_dim=base_contract.obs_dim,
            )
            if step_count != int(episode["step_count"]):
                raise ValueError(f"DAgger episode manifest step count mismatch: {episode_id}")

            for start in range(0, step_count, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, step_count)
                chunk_id = f"chunk_{next_chunk_index:06d}"
                policy_relative = Path("policy_input") / f"{chunk_id}.npz"
                action_relative = Path("action_targets") / f"{chunk_id}.npz"
                auxiliary_relative = Path("auxiliary") / f"{chunk_id}.npz"
                np.savez_compressed(
                    output_dir / policy_relative,
                    simulation_step_index=np.asarray(
                        arrays["simulation_step_index"][start:end],
                        dtype=np.int32,
                    ),
                    timestamp=np.asarray(arrays["timestamp"][start:end], dtype=np.float64),
                    obs=np.asarray(arrays["obs"][start:end], dtype=np.float32),
                )
                np.savez_compressed(
                    output_dir / action_relative,
                    action_cont=np.asarray(
                        arrays["teacher_action_cont"][start:end],
                        dtype=np.float32,
                    ),
                    action_bin=np.asarray(
                        arrays["teacher_action_bin"][start:end],
                        dtype=np.float32,
                    ),
                )
                np.savez_compressed(
                    output_dir / auxiliary_relative,
                    **{
                        key: arrays[key][start:end]
                        for key in AUXILIARY_KEYS
                    },
                )
                meta["chunks"].append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": next_chunk_index,
                        "demonstration_source": dagger_source_name,
                        "episode_id": episode_id,
                        "group_files": {
                            "action_targets": str(action_relative),
                            "aux": str(auxiliary_relative),
                            "policy_input": str(policy_relative),
                        },
                        "observed_role": str(manifest["ego_role"]),
                        "sample_weight": dagger_sample_weight,
                        "simulation_step_index_start": int(
                            arrays["simulation_step_index"][start]
                        ),
                        "simulation_step_index_end_exclusive": int(
                            arrays["simulation_step_index"][end - 1]
                        )
                        + 1,
                        "split": "train",
                        "step_count": end - start,
                    }
                )
                next_chunk_index += 1

            added_steps += step_count
            added_episode_ids.append(episode_id)
            meta["episodes"].append(
                {
                    "authoritative_source": True,
                    "episode_id": episode_id,
                    "scene_name": str(episode["scene"]["label"]),
                    "source_episode_root": str(episode_path),
                    "split": "train",
                    "termination_reason": (
                        "terminated"
                        if bool(episode["terminated"])
                        else "truncated"
                        if bool(episode["truncated"])
                        else "collection_limit"
                    ),
                    "total_steps": step_count,
                    "winner": episode.get("winner"),
                }
            )
            dagger_episode_provenance.append(
                {
                    "episode_id": episode_id,
                    "collection_manifest_sha256": manifest_hash,
                    "student_checkpoint": manifest["student_checkpoint"],
                    "teacher_execution_probability": manifest[
                        "teacher_execution_probability"
                    ],
                    "mean_absolute_continuous_error": episode[
                        "mean_absolute_continuous_error"
                    ],
                    "binary_mismatch_rate": episode["binary_mismatch_rate"],
                }
            )

    meta["dataset_id"] = dataset_id
    meta["statistics"]["total_model_steps"] = (
        int(meta["statistics"]["total_model_steps"]) + added_steps
    )
    meta["statistics"]["total_simulation_steps"] = (
        int(meta["statistics"]["total_simulation_steps"]) + added_steps
    )
    meta["splits"]["train"].extend(added_episode_ids)
    meta["lineage"] = {
        "initialization_parent_dataset_ids": [base_contract.dataset_id],
        "normalizer_reused_from_dataset_id": base_contract.dataset_id,
    }
    meta["dagger"] = {
        "schema": "dfb.part3.dagger_dataset_lineage.v1",
        "added_episode_count": len(added_episode_ids),
        "added_step_count": added_steps,
        "demonstration_source": dagger_source_name,
        "sample_weight": dagger_sample_weight,
        "collections": [
            {
                "root": str(root),
                "manifest_sha256": manifest_hash,
            }
            for root, _, manifest_hash in collection_records
        ],
        "episodes": dagger_episode_provenance,
    }
    meta["source_episode_manifest_sha256"] = lineage_digest.hexdigest()
    _write_json(normalizer_path, normalizer)
    _write_json(meta_path, meta)
    load_and_validate_policy_dataset(output_dir)
    return {
        "dataset_id": dataset_id,
        "base_dataset_id": base_contract.dataset_id,
        "added_episode_count": len(added_episode_ids),
        "added_step_count": added_steps,
        "dagger_source_name": dagger_source_name,
        "dagger_sample_weight": dagger_sample_weight,
        "total_model_steps": meta["statistics"]["total_model_steps"],
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = _parse_args()
    summary = build_dagger_dataset(
        base_dataset_root=Path(args.base_dataset_root).resolve(),
        collection_roots=[
            Path(value).resolve() for value in args.dagger_collection_root
        ],
        output_dir=Path(args.output_dir).resolve(),
        dagger_source_name=args.dagger_source_name,
        dagger_sample_weight=args.dagger_sample_weight,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
