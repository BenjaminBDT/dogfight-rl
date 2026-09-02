from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_assets import load_and_validate_policy_dataset
from dfb_reinforcement_learning.policy_contract import (
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge selected roles from compatible Part 3 policy datasets."
    )
    parser.add_argument(
        "--input-dataset",
        action="append",
        required=True,
        help="Parent policy dataset. May be repeated.",
    )
    parser.add_argument(
        "--include-role",
        action="append",
        choices=("fighter1", "fighter2"),
        default=None,
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        required=True,
        metavar="SOURCE=WEIGHT",
    )
    parser.add_argument("--dagger-source-name", default="dagger_teacher")
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


def _parse_source_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        source, separator, raw_weight = value.partition("=")
        source = source.strip()
        if not separator or not source:
            raise ValueError(f"source weight must use SOURCE=WEIGHT syntax: {value!r}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"invalid source weight: {value!r}") from exc
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"source weight must be finite and positive: {value!r}")
        if source in weights:
            raise ValueError(f"duplicate source weight: {source}")
        weights[source] = weight
    return weights


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _dagger_episode_ids(meta: dict[str, Any]) -> frozenset[str]:
    dagger = meta.get("dagger")
    if not isinstance(dagger, dict):
        return frozenset()
    episodes = dagger.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError("dataset dagger.episodes must be an array")
    result: set[str] = set()
    for entry in episodes:
        if not isinstance(entry, dict):
            raise ValueError("dataset dagger episode entry must be an object")
        episode_id = str(entry.get("episode_id", "")).strip()
        if not episode_id or episode_id in result:
            raise ValueError("dataset dagger episode IDs must be unique and non-empty")
        result.add(episode_id)
    return frozenset(result)


def _normalizer_payload(
    *,
    dataset_id: str,
    obs_sum: np.ndarray,
    obs_sum_sq: np.ndarray,
    row_count: int,
    epsilon: float,
) -> dict[str, Any]:
    if row_count <= 0:
        raise ValueError("merged train split has no observation rows")
    mean = obs_sum / row_count
    variance = np.maximum(obs_sum_sq / row_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), epsilon)
    for index in POLICY_OBSERVATION_SCHEMA.binary_indices:
        mean[index] = 0.0
        std[index] = 1.0
    return {
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "obs_dim": POLICY_OBSERVATION_SCHEMA.dim,
        "epsilon": epsilon,
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "train_row_count": row_count,
        "source_dataset_id": dataset_id,
    }


def merge_policy_datasets(
    *,
    input_datasets: list[Path],
    output_dir: Path,
    included_roles: tuple[str, ...],
    source_weights: dict[str, float],
    dagger_source_name: str,
    force: bool,
) -> dict[str, Any]:
    if len(input_datasets) < 2:
        raise ValueError("at least two input datasets are required")
    roles = frozenset(included_roles)
    if not roles or not roles.issubset({"fighter1", "fighter2"}):
        raise ValueError("included_roles must contain fighter1 and/or fighter2")
    dagger_source_name = dagger_source_name.strip()
    if not dagger_source_name:
        raise ValueError("dagger_source_name must be non-empty")
    if dagger_source_name not in source_weights:
        raise ValueError(f"missing source weight for {dagger_source_name}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    parents: list[tuple[Path, Any, dict[str, Any], frozenset[str]]] = []
    lineage_digest = hashlib.sha256(b"dfb.part3.policy-dataset-merge.v1")
    for raw_root in input_datasets:
        root = raw_root.resolve()
        contract, _ = load_and_validate_policy_dataset(root)
        meta = _read_json(root / "meta.json")
        lineage_digest.update(contract.dataset_id.encode("utf-8"))
        parents.append((root, contract, meta, _dagger_episode_ids(meta)))
    lineage_digest.update(",".join(sorted(roles)).encode("utf-8"))
    for source, weight in sorted(source_weights.items()):
        lineage_digest.update(f"{source}={weight:.17g}".encode("utf-8"))
    lineage_digest.update(dagger_source_name.encode("utf-8"))
    dataset_id = f"dfb_part3_policy_dataset_merge_{lineage_digest.hexdigest()[:16]}"

    output_dir.mkdir(parents=True)
    first_root, first_contract, first_meta, _ = parents[0]
    shutil.copy2(first_root / str(first_meta["schema_path"]), output_dir / "schema.json")

    chunks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    split_episode_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    selected_episode_ids: set[str] = set()
    conventions: dict[str, dict[str, float]] = {}
    obs_sum = np.zeros((first_contract.obs_dim,), dtype=np.float64)
    obs_sum_sq = np.zeros((first_contract.obs_dim,), dtype=np.float64)
    train_row_count = 0
    total_model_steps = 0
    source_step_counts: dict[str, int] = {}
    next_chunk_index = 0

    for parent_root, contract, meta, dagger_ids in parents:
        episode_by_id = {
            str(entry["episode_id"]): entry
            for entry in meta.get("episodes", [])
            if isinstance(entry, dict) and entry.get("episode_id")
        }
        parent_selected_ids: set[str] = set()
        for raw_chunk in meta.get("chunks", []):
            if not isinstance(raw_chunk, dict):
                raise ValueError(f"dataset chunk entry must be an object: {parent_root}")
            role = str(raw_chunk.get("observed_role", ""))
            if role not in roles:
                continue
            episode_id = str(raw_chunk["episode_id"])
            source = (
                dagger_source_name
                if episode_id in dagger_ids
                else str(raw_chunk.get("demonstration_source", "")).strip()
            )
            if source not in source_weights:
                raise ValueError(
                    f"no explicit source weight for {source!r} from {parent_root}"
                )
            split = str(raw_chunk["split"])
            if split not in split_episode_ids:
                raise ValueError(f"unsupported split {split!r} in {parent_root}")
            group_files = raw_chunk.get("group_files")
            if not isinstance(group_files, dict):
                raise ValueError(f"chunk group_files must be an object in {parent_root}")

            chunk_id = f"chunk_{next_chunk_index:06d}"
            destination_groups: dict[str, str] = {}
            for group_name, raw_relative in group_files.items():
                source_path = parent_root / str(raw_relative)
                suffix = source_path.suffix or ".npz"
                destination_relative = Path(source_path.parent.name) / f"{chunk_id}{suffix}"
                _link_or_copy(source_path, output_dir / destination_relative)
                destination_groups[str(group_name)] = destination_relative.as_posix()

            step_count = int(raw_chunk["step_count"])
            chunk = dict(raw_chunk)
            chunk.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": next_chunk_index,
                    "demonstration_source": source,
                    "sample_weight": source_weights[source],
                    "group_files": destination_groups,
                    "source_dataset_id": contract.dataset_id,
                }
            )
            chunks.append(chunk)
            conventions.setdefault(role, {})[source] = source_weights[source]
            parent_selected_ids.add(episode_id)
            total_model_steps += step_count
            source_step_counts[source] = source_step_counts.get(source, 0) + step_count

            if split == "train":
                with np.load(output_dir / destination_groups["policy_input"]) as loaded:
                    obs = np.asarray(loaded["obs"], dtype=np.float64)
                if obs.shape != (step_count, first_contract.obs_dim):
                    raise ValueError(f"observation shape mismatch in merged chunk {chunk_id}")
                if not np.isfinite(obs).all():
                    raise ValueError(f"non-finite observation in merged chunk {chunk_id}")
                obs_sum += obs.sum(axis=0)
                obs_sum_sq += np.square(obs).sum(axis=0)
                train_row_count += step_count
            next_chunk_index += 1

        for episode_id in sorted(parent_selected_ids):
            if episode_id in selected_episode_ids:
                raise ValueError(f"duplicate episode ID across parent datasets: {episode_id}")
            if episode_id not in episode_by_id:
                raise ValueError(f"chunk references missing episode metadata: {episode_id}")
            episode = dict(episode_by_id[episode_id])
            split = str(episode["split"])
            episode["source_dataset_id"] = contract.dataset_id
            episodes.append(episode)
            split_episode_ids[split].append(episode_id)
            selected_episode_ids.add(episode_id)

    if not chunks:
        raise ValueError("no chunks matched the requested roles")

    normalizer_epsilon = float(
        _read_json(first_contract.normalizer_path).get("epsilon", 1e-6)
    )
    normalizer = _normalizer_payload(
        dataset_id=dataset_id,
        obs_sum=obs_sum,
        obs_sum_sq=obs_sum_sq,
        row_count=train_row_count,
        epsilon=normalizer_epsilon,
    )
    base_constants = dict(first_meta.get("constants", {}))
    base_constants.pop("demonstration_sources", None)
    base_constants.pop("bc_sample_weights", None)
    base_constants["bc_demonstration_conventions"] = conventions

    parent_ids = [contract.dataset_id for _, contract, _, _ in parents]
    merge_record = {
        "schema": "dfb.part3.policy_dataset_merge.v1",
        "included_roles": sorted(roles),
        "dagger_source_name": dagger_source_name,
        "source_weights": dict(sorted(source_weights.items())),
        "source_step_counts": dict(sorted(source_step_counts.items())),
        "parents": [
            {"dataset_id": contract.dataset_id, "root": str(root)}
            for root, contract, _, _ in parents
        ],
    }
    meta = {
        "dataset_id": dataset_id,
        "dataset_schema_id": first_contract.dataset_schema_id,
        "dataset_version": "1.2.0",
        "schema_version": "1.2.0",
        "policy_contract_id": first_contract.policy_contract_id,
        "observation_schema_id": first_contract.observation_schema_id,
        "action_schema_id": first_contract.action_schema_id,
        "normalizer_schema_id": first_contract.normalizer_schema_id,
        "contract_sha256": first_contract.contract_sha256,
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "source_episode_manifest_sha256": lineage_digest.hexdigest(),
        "split_strategy": "preserve_parent_episode_splits_v1",
        "storage_layout": first_meta["storage_layout"],
        "constants": base_constants,
        "statistics": {
            "obs_dim": first_contract.obs_dim,
            "total_model_steps": total_model_steps,
            "total_simulation_steps": total_model_steps,
        },
        "splits": split_episode_ids,
        "episodes": episodes,
        "chunks": chunks,
        "lineage": {
            "initialization_parent_dataset_ids": parent_ids,
        },
        "merge": merge_record,
    }
    _write_json(output_dir / "obs_normalizer.json", normalizer)
    _write_json(output_dir / "meta.json", meta)
    load_and_validate_policy_dataset(output_dir)
    return {
        "dataset_id": dataset_id,
        "output_dir": str(output_dir),
        "parent_dataset_ids": parent_ids,
        "episode_count": len(episodes),
        "chunk_count": len(chunks),
        "train_row_count": train_row_count,
        "total_model_steps": total_model_steps,
        "source_step_counts": dict(sorted(source_step_counts.items())),
    }


def main() -> None:
    args = _parse_args()
    summary = merge_policy_datasets(
        input_datasets=[Path(value) for value in args.input_dataset],
        output_dir=Path(args.output_dir).resolve(),
        included_roles=tuple(args.include_role or ("fighter1",)),
        source_weights=_parse_source_weights(args.source_weight),
        dagger_source_name=args.dagger_source_name,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
