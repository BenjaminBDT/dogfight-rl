from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dfb_reinforcement_learning.policy_assets import load_and_validate_policy_dataset

from .normalizer import ObservationNormalizer


@dataclass(frozen=True)
class BcSplitData:
    obs: np.ndarray
    action_cont: np.ndarray
    action_bin: np.ndarray
    episode_ids: tuple[str, ...]
    observed_roles: tuple[str, ...]
    demonstration_sources: tuple[str, ...]
    sample_weights: np.ndarray
    step_indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


def filter_bc_split_by_roles(
    split_data: BcSplitData,
    included_roles: tuple[str, ...] | list[str],
) -> BcSplitData:
    roles = frozenset(included_roles)
    unsupported = roles.difference(("fighter1", "fighter2"))
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported BC observed roles: {names}")
    if not roles:
        raise ValueError("at least one BC observed role must be included")

    mask = np.fromiter(
        (role in roles for role in split_data.observed_roles),
        dtype=np.bool_,
        count=split_data.size,
    )
    selected = np.flatnonzero(mask)
    return BcSplitData(
        obs=split_data.obs[selected],
        action_cont=split_data.action_cont[selected],
        action_bin=split_data.action_bin[selected],
        episode_ids=tuple(split_data.episode_ids[index] for index in selected),
        observed_roles=tuple(split_data.observed_roles[index] for index in selected),
        demonstration_sources=tuple(
            split_data.demonstration_sources[index] for index in selected
        ),
        sample_weights=split_data.sample_weights[selected],
        step_indices=split_data.step_indices[selected],
    )


def with_episode_balanced_weights(split_data: BcSplitData) -> BcSplitData:
    if split_data.size == 0:
        return split_data

    group_keys = tuple(zip(split_data.episode_ids, split_data.observed_roles, strict=True))
    group_counts = Counter(group_keys)
    mean_group_size = split_data.size / len(group_counts)
    balanced_weights = np.asarray(
        [
            float(base_weight) * mean_group_size / group_counts[group_key]
            for base_weight, group_key in zip(split_data.sample_weights, group_keys, strict=True)
        ],
        dtype=np.float32,
    )
    return BcSplitData(
        obs=split_data.obs,
        action_cont=split_data.action_cont,
        action_bin=split_data.action_bin,
        episode_ids=split_data.episode_ids,
        observed_roles=split_data.observed_roles,
        demonstration_sources=split_data.demonstration_sources,
        sample_weights=balanced_weights,
        step_indices=split_data.step_indices,
    )


def _load_meta(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))


def _valid_sample_weight(value: Any, *, context: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"{context} must be finite and positive")
    return weight


def _load_demonstration_conventions(meta: dict[str, Any]) -> dict[str, dict[str, float]]:
    try:
        constants = meta["constants"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset metadata is missing BC demonstration conventions") from exc
    if not isinstance(constants, dict):
        raise ValueError("dataset metadata constants must be an object")

    multi_source = constants.get("bc_demonstration_conventions")
    if multi_source is not None:
        if not isinstance(multi_source, dict) or not multi_source:
            raise ValueError("bc_demonstration_conventions must be a non-empty object")
        conventions: dict[str, dict[str, float]] = {}
        for raw_role, raw_sources in multi_source.items():
            role = str(raw_role).strip()
            if role not in {"fighter1", "fighter2"}:
                raise ValueError(f"unsupported BC demonstration role: {role!r}")
            if not isinstance(raw_sources, dict) or not raw_sources:
                raise ValueError(f"BC demonstration sources for {role} must be a non-empty object")
            role_sources: dict[str, float] = {}
            for raw_source, raw_weight in raw_sources.items():
                source = str(raw_source).strip()
                if not source:
                    raise ValueError(f"dataset metadata has empty demonstration source for {role}")
                role_sources[source] = _valid_sample_weight(
                    raw_weight,
                    context=f"BC sample weight for {role}/{source}",
                )
            conventions[role] = role_sources
        return conventions

    try:
        sources = constants["demonstration_sources"]
        weights = constants["bc_sample_weights"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset metadata is missing BC demonstration conventions") from exc
    if not isinstance(sources, dict) or not isinstance(weights, dict):
        raise ValueError("legacy BC demonstration conventions must be objects")

    conventions = {}
    for role in ("fighter1", "fighter2"):
        source = str(sources.get(role, "")).strip()
        if not source:
            raise ValueError(f"dataset metadata has empty demonstration source for {role}")
        if role not in weights:
            raise ValueError(f"dataset metadata has no sample weight for {role}")
        conventions[role] = {
            source: _valid_sample_weight(
                weights[role],
                context=f"BC sample weight for {role}",
            )
        }
    return conventions


def load_bc_split(dataset_root: str | Path, split: str) -> tuple[BcSplitData, ObservationNormalizer]:
    root = Path(dataset_root)
    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(root)
    meta = _load_meta(root)
    demonstration_conventions = _load_demonstration_conventions(meta)
    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    split_chunks = [entry for entry in meta["chunks"] if entry["split"] == split]
    policy_obs_parts: list[np.ndarray] = []
    action_cont_parts: list[np.ndarray] = []
    action_bin_parts: list[np.ndarray] = []
    episode_ids: list[str] = []
    observed_roles: list[str] = []
    demonstration_sources: list[str] = []
    sample_weights: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    for entry in split_chunks:
        observed_role = str(entry["observed_role"])
        if observed_role not in demonstration_conventions:
            raise ValueError(f"chunk {entry['chunk_id']} has unsupported observed_role {observed_role}")
        expected_sources = demonstration_conventions[observed_role]
        demonstration_source = str(entry.get("demonstration_source", "")).strip()
        if not demonstration_source:
            raise ValueError(f"chunk {entry['chunk_id']} has empty demonstration_source")
        if demonstration_source not in expected_sources:
            raise ValueError(
                f"chunk {entry['chunk_id']} has unsupported demonstration source "
                f"{demonstration_source!r} for {observed_role}"
            )
        sample_weight = _valid_sample_weight(
            entry.get("sample_weight"),
            context=f"chunk {entry['chunk_id']} sample_weight",
        )
        expected_weight = expected_sources[demonstration_source]
        if sample_weight != expected_weight:
            raise ValueError(f"chunk {entry['chunk_id']} demonstration metadata disagrees with dataset constants")
        group_files = entry["group_files"]
        with np.load(root / group_files["policy_input"]) as policy_npz:
            obs = np.asarray(policy_npz["obs"], dtype=np.float32)
            sim_step = np.asarray(policy_npz["simulation_step_index"], dtype=np.int32)
        with np.load(root / group_files["action_targets"]) as action_npz:
            action_cont = np.asarray(action_npz["action_cont"], dtype=np.float32)
            action_bin = np.asarray(action_npz["action_bin"], dtype=np.float32)
        step_count = int(entry["step_count"])
        if obs.shape[0] != step_count or action_cont.shape[0] != step_count or action_bin.shape[0] != step_count:
            raise ValueError(f"chunk {entry['chunk_id']} shape/step_count mismatch")
        if obs.shape != (step_count, dataset_contract.obs_dim):
            raise ValueError(f"chunk {entry['chunk_id']} observation shape mismatch")
        if action_cont.shape != (step_count, 4) or action_bin.shape != (step_count, 3):
            raise ValueError(f"chunk {entry['chunk_id']} action shape mismatch")
        if not np.isfinite(obs).all() or not np.isfinite(action_cont).all() or not np.isfinite(action_bin).all():
            raise ValueError(f"chunk {entry['chunk_id']} contains non-finite values")
        policy_obs_parts.append(obs)
        action_cont_parts.append(action_cont)
        action_bin_parts.append(action_bin)
        episode_ids.extend([str(entry["episode_id"])] * step_count)
        observed_roles.extend([observed_role] * step_count)
        demonstration_sources.extend([demonstration_source] * step_count)
        sample_weights.append(np.full((step_count,), sample_weight, dtype=np.float32))
        step_indices.append(sim_step)
    if not policy_obs_parts:
        empty_obs = np.zeros((0, normalizer.obs_dim), dtype=np.float32)
        return (
            BcSplitData(
                obs=empty_obs,
                action_cont=np.zeros((0, 4), dtype=np.float32),
                action_bin=np.zeros((0, 3), dtype=np.float32),
                episode_ids=(),
                observed_roles=(),
                demonstration_sources=(),
                sample_weights=np.zeros((0,), dtype=np.float32),
                step_indices=np.zeros((0,), dtype=np.int32),
            ),
            normalizer,
        )
    return (
        BcSplitData(
            obs=np.concatenate(policy_obs_parts, axis=0, dtype=np.float32),
            action_cont=np.concatenate(action_cont_parts, axis=0, dtype=np.float32),
            action_bin=np.concatenate(action_bin_parts, axis=0, dtype=np.float32),
            episode_ids=tuple(episode_ids),
            observed_roles=tuple(observed_roles),
            demonstration_sources=tuple(demonstration_sources),
            sample_weights=np.concatenate(sample_weights, axis=0, dtype=np.float32),
            step_indices=np.concatenate(step_indices, axis=0, dtype=np.int32),
        ),
        normalizer,
    )


class BcDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, split_data: BcSplitData, *, normalizer: ObservationNormalizer, normalize_obs: bool = True) -> None:
        if split_data.obs.shape[1] != normalizer.obs_dim:
            raise ValueError("observation dim mismatch between split data and normalizer")
        obs = split_data.obs
        if normalize_obs and obs.size > 0:
            obs = normalizer.normalize_np(obs).astype(np.float32, copy=False)
        self._obs = torch.from_numpy(obs)
        self._action_cont = torch.from_numpy(split_data.action_cont.astype(np.float32, copy=False))
        self._action_bin = torch.from_numpy(split_data.action_bin.astype(np.float32, copy=False))
        self._sample_weight = torch.from_numpy(split_data.sample_weights.astype(np.float32, copy=False))
        if self._sample_weight.shape != (self._obs.shape[0],):
            raise ValueError("sample weight shape mismatch")

    def __len__(self) -> int:
        return int(self._obs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self._obs[index],
            "action_cont": self._action_cont[index],
            "action_bin": self._action_bin[index],
            "sample_weight": self._sample_weight[index],
        }
