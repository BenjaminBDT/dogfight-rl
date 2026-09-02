from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.models import (
    StatelessHybridActorCritic,
    model_architecture_kwargs,
)
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.policy_assets import (
    load_and_validate_policy_dataset,
    validate_policy_checkpoint_payload,
)
from dfb_reinforcement_learning.policy_inference import deterministic_policy_output


@dataclass(frozen=True)
class OpponentPoolEntrySpec:
    name: str
    kind: str
    weight: float = 1.0
    mode: str | None = None
    checkpoint: str | None = None
    checkpoints: tuple[str, ...] = ()
    dataset_root: str | None = None


@dataclass(frozen=True)
class SampledOpponent:
    label: str
    env_mode: str
    runtime_kind: str
    checkpoint_path: str | None = None
    dataset_root: str | None = None

    def needs_external_action(self) -> bool:
        return self.env_mode == "external"


@dataclass(frozen=True)
class OpponentPoolSpec:
    entries: tuple[OpponentPoolEntrySpec, ...]

    @classmethod
    def from_json(cls, path: Path) -> "OpponentPoolSpec":
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_dir = path.parent
        entries = []
        for item in payload.get("entries", []):
            checkpoint = item.get("checkpoint")
            checkpoints = item.get("checkpoints", [])
            dataset_root = item.get("dataset_root")

            def _resolve_optional(value: str | None) -> str | None:
                if not value:
                    return None
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = (base_dir / candidate).resolve()
                return str(candidate)

            resolved_checkpoint = _resolve_optional(checkpoint)
            resolved_checkpoints = tuple(
                _resolve_optional(value) for value in checkpoints if _resolve_optional(value) is not None
            )
            resolved_dataset_root = _resolve_optional(dataset_root)
            entries.append(
                OpponentPoolEntrySpec(
                    name=str(item["name"]),
                    kind=str(item["kind"]),
                    weight=float(item.get("weight", 1.0)),
                    mode=None if item.get("mode") is None else str(item.get("mode")),
                    checkpoint=resolved_checkpoint,
                    checkpoints=resolved_checkpoints,
                    dataset_root=resolved_dataset_root,
                )
            )
        return cls(entries=tuple(entries))


class PreparedOpponentPool:
    def __init__(self, entries: tuple[OpponentPoolEntrySpec, ...], *, rng_seed: int) -> None:
        if not entries:
            raise ValueError("opponent pool must not be empty")
        self.entries = entries
        self._rng = random.Random(rng_seed)

    @property
    def has_self_play(self) -> bool:
        return any(entry.kind == "self_play" for entry in self.entries)

    def sample_episode_opponent(self) -> SampledOpponent:
        weights = [max(item.weight, 0.0) for item in self.entries]
        spec = self._rng.choices(self.entries, weights=weights, k=1)[0]
        if spec.kind == "built_in_ai":
            env_mode = spec.mode or "built_in_ai"
            return SampledOpponent(label=spec.name, env_mode=env_mode, runtime_kind="built_in")
        if spec.kind == "external_neutral":
            return SampledOpponent(label=spec.name, env_mode="external", runtime_kind="neutral")
        if spec.kind == "self_play":
            return SampledOpponent(label=spec.name, env_mode="model", runtime_kind="self_play")
        if spec.kind == "fixed_checkpoint":
            if spec.checkpoint is None:
                raise ValueError(f"fixed_checkpoint entry '{spec.name}' missing checkpoint")
            return SampledOpponent(
                label=spec.name,
                env_mode="external",
                runtime_kind="checkpoint",
                checkpoint_path=spec.checkpoint,
                dataset_root=spec.dataset_root,
            )
        if spec.kind == "sampled_checkpoints":
            if not spec.checkpoints:
                raise ValueError(f"sampled_checkpoints entry '{spec.name}' missing checkpoints")
            checkpoint_path = self._rng.choice(spec.checkpoints)
            return SampledOpponent(
                label=f"{spec.name}:{Path(checkpoint_path).stem}",
                env_mode="external",
                runtime_kind="checkpoint",
                checkpoint_path=checkpoint_path,
                dataset_root=spec.dataset_root,
            )
        raise ValueError(f"unsupported opponent pool kind: {spec.kind}")


def materialize_opponent_pool(spec: OpponentPoolSpec) -> PreparedOpponentPool:
    for entry in spec.entries:
        checkpoint_paths = (
            (entry.checkpoint,) if entry.kind == "fixed_checkpoint" else entry.checkpoints
        )
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path is None:
                continue
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            dataset_root = _resolve_checkpoint_dataset_root(
                checkpoint_path,
                entry.dataset_root,
                payload,
            )
            dataset, _ = load_and_validate_policy_dataset(dataset_root)
            validate_policy_checkpoint_payload(
                payload,
                dataset=dataset,
                context=f"opponent pool checkpoint {checkpoint_path}",
            )
    return PreparedOpponentPool(spec.entries, rng_seed=29)


def _resolve_checkpoint_dataset_root(
    checkpoint_path: str,
    dataset_root: str | None,
    payload: dict[str, Any],
) -> Path:
    if dataset_root:
        return Path(dataset_root)
    args = payload.get("args")
    if isinstance(args, dict):
        resolved = args.get("dataset_root")
        if isinstance(resolved, str) and resolved:
            return Path(resolved)
    raise ValueError(f"dataset root missing for checkpoint opponent: {checkpoint_path}")


@dataclass
class _LoadedCheckpointOpponent:
    normalizer: ObservationNormalizer
    model: StatelessHybridActorCritic
    obs_adapter: PolicyObservationAdapter


class OpponentActionProvider:
    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._checkpoint_cache: dict[tuple[str, str | None], _LoadedCheckpointOpponent] = {}

    def _resolve_dataset_root(self, checkpoint_path: str, dataset_root: str | None) -> Path:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return _resolve_checkpoint_dataset_root(checkpoint_path, dataset_root, payload)

    def _load_checkpoint_opponent(
        self,
        checkpoint_path: str,
        dataset_root: str | None,
    ) -> _LoadedCheckpointOpponent:
        cache_key = (checkpoint_path, dataset_root)
        cached = self._checkpoint_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved_dataset_root = self._resolve_dataset_root(checkpoint_path, dataset_root)
        dataset, normalizer_payload = load_and_validate_policy_dataset(resolved_dataset_root)
        normalizer = ObservationNormalizer.from_payload(
            normalizer_payload,
            dataset=dataset,
        )
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        validate_policy_checkpoint_payload(
            payload,
            dataset=dataset,
            context="opponent checkpoint",
        )
        hyperparameters = payload["model_hyperparameters"]
        model = StatelessHybridActorCritic(
            obs_dim=normalizer.obs_dim,
            **model_architecture_kwargs(hyperparameters),
        ).to(self.device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        cached = _LoadedCheckpointOpponent(
            normalizer=normalizer,
            model=model,
            obs_adapter=PolicyObservationAdapter(),
        )
        self._checkpoint_cache[cache_key] = cached
        return cached

    def action_arrays_for_state(
        self,
        sampled: SampledOpponent,
        *,
        state: dict[str, Any],
        role: str,
        episode_start_sim_time_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if sampled.runtime_kind == "built_in":
            return None
        if sampled.runtime_kind == "neutral":
            return (
                np.zeros((4,), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
            )
        if sampled.runtime_kind == "checkpoint":
            if episode_start_sim_time_seconds is None:
                raise ValueError("checkpoint opponent missing episode_start_sim_time_seconds")
            if sampled.checkpoint_path is None:
                raise ValueError("checkpoint opponent missing checkpoint_path")
            loaded = self._load_checkpoint_opponent(sampled.checkpoint_path, sampled.dataset_root)
            obs = loaded.obs_adapter.build(
                state,
                role,
                episode_start_sim_time_seconds=episode_start_sim_time_seconds,
            )["vector"]
            output = deterministic_policy_output(
                model=loaded.model,
                normalizer=loaded.normalizer,
                obs=obs,
                device=self.device,
            )
            return output.action_cont, output.binary_actions()
        raise ValueError(f"unsupported opponent runtime kind: {sampled.runtime_kind}")
