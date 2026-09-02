from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.envs import (
    PolicyDogfightEnvConfig,
    ResetRequest,
    StepRequest,
    SubprocPolicyVecEnv,
)
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    load_and_validate_policy_dataset,
)
from dfb_reinforcement_learning.policy_inference import (
    deterministic_policy_output,
    load_policy_model,
    policy_output_batch,
)
from dfb_reinforcement_learning.train.ppo_diagnostics import (
    BINARY_ACTION_NAMES,
    CONTINUOUS_ACTION_NAMES,
)


GATE_SCHEMA_ID = "dfb_part3_policy_capability_gate_v1"
_FIELD_INDICES = {
    field.name: field.value_slice.start
    for field in POLICY_OBSERVATION_SCHEMA.fields
    if field.value_slice.stop - field.value_slice.start == 1
}


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    path: Path
    expected_sha256: str | None


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    kind: str
    modes: tuple[str, ...]
    checkpoint: Path | None = None
    continuous_std: float = 0.25
    ego_mode: str = "external"
    comparison_role: str | None = None


@dataclass(frozen=True)
class GateManifest:
    source_path: Path
    dataset_root: Path
    scenes: tuple[SceneSpec, ...]
    policies: tuple[PolicySpec, ...]
    roles: tuple[str, ...]
    seeds: tuple[int, ...]
    opponent_mode: str
    num_envs: int
    ticks_per_step: int
    max_sim_seconds: float
    max_steps: int
    shot_window_threshold: float
    boundary_threat_seconds: float
    parity_stride_steps: int


@dataclass
class EpisodeAccumulator:
    seed: int
    start_sim_time_seconds: float
    previous_sim_time_seconds: float
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = "incomplete"
    self_destroyed: bool = False
    enemy_destroyed: bool = False
    self_hit_count: int = 0
    enemy_hit_count: int = 0
    first_shot_window_seconds: float | None = None
    shot_window_seconds: float = 0.0
    enemy_shot_window_seconds: float = 0.0
    fire_seconds: float = 0.0
    effective_fire_seconds: float = 0.0
    tracking_integral: float = 0.0
    tail_integral: float = 0.0
    enemy_tail_integral: float = 0.0
    advantage_integral: float = 0.0
    speed_integral: float = 0.0
    stall_integral: float = 0.0
    boundary_threat_seconds: float = 0.0
    out_of_bounds_seconds: float = 0.0
    max_out_of_bounds_seconds: float = 0.0
    requested_cont_sum: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float64))
    requested_cont_sq_sum: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float64))
    requested_cont_saturated: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float64))
    requested_cont_delta_sum: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float64))
    requested_cont_delta_count: int = 0
    previous_requested_cont: np.ndarray | None = None
    binary_probability_sum: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    actual_binary_on_sum: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    action_sample_count: int = 0
    action_source_observable: bool = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Part 3 fixed-seed closed-loop capability gate.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project-root", default=None)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"manifest field {key} must be a non-empty array")
    return value


def _resolve_path(value: Any, *, base: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {field_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() and field_name != "dataset_root":
        raise ValueError(f"manifest field {field_name} does not exist: {path}")
    return path


def load_manifest(path: str | Path) -> GateManifest:
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("capability gate manifest root must be an object")
    if payload.get("schema_id") != GATE_SCHEMA_ID:
        raise ValueError(f"manifest schema_id must be {GATE_SCHEMA_ID}")
    base = source_path.parent
    dataset_root_raw = payload.get("dataset_root")
    if not isinstance(dataset_root_raw, str) or not dataset_root_raw:
        raise ValueError("manifest field dataset_root must be a non-empty path")
    dataset_root = Path(dataset_root_raw)
    if not dataset_root.is_absolute():
        dataset_root = base / dataset_root
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root does not exist: {dataset_root}")

    scenes: list[SceneSpec] = []
    scene_ids: set[str] = set()
    for index, raw in enumerate(_required_list(payload, "scenes")):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest scenes[{index}] must be an object")
        scene_id = str(raw.get("id", ""))
        if not scene_id or scene_id in scene_ids:
            raise ValueError(f"manifest scenes[{index}].id must be non-empty and unique")
        scene_ids.add(scene_id)
        expected_sha = raw.get("sha256")
        if expected_sha is not None and (
            not isinstance(expected_sha, str) or len(expected_sha) != 64
        ):
            raise ValueError(f"manifest scenes[{index}].sha256 must be a SHA-256 string")
        scenes.append(
            SceneSpec(
                scene_id=scene_id,
                path=_resolve_path(raw.get("path"), base=base, field_name=f"scenes[{index}].path"),
                expected_sha256=expected_sha,
            )
        )

    policies: list[PolicySpec] = []
    policy_ids: set[str] = set()
    for index, raw in enumerate(_required_list(payload, "policies")):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest policies[{index}] must be an object")
        policy_id = str(raw.get("id", ""))
        kind = str(raw.get("type", ""))
        if not policy_id or policy_id in policy_ids:
            raise ValueError(f"manifest policies[{index}].id must be non-empty and unique")
        if kind not in {"random", "built_in_ai", "checkpoint"}:
            raise ValueError(f"unsupported policy type for {policy_id}: {kind}")
        policy_ids.add(policy_id)
        modes = tuple(str(item) for item in raw.get("modes", []))
        expected_modes = {
            "random": {"sampled"},
            "built_in_ai": {"native"},
            "checkpoint": {"deterministic", "sampled"},
        }[kind]
        if not modes or any(mode not in expected_modes for mode in modes):
            raise ValueError(f"invalid inference modes for policy {policy_id}: {modes}")
        checkpoint = None
        if kind == "checkpoint":
            checkpoint = _resolve_path(
                raw.get("checkpoint"),
                base=base,
                field_name=f"policies[{index}].checkpoint",
            )
        continuous_std = float(raw.get("continuous_std", 0.25))
        if not np.isfinite(continuous_std) or continuous_std < 0.0:
            raise ValueError(f"policy {policy_id} continuous_std must be finite and non-negative")
        ego_mode = str(
            raw.get(
                "ego_mode",
                "built_in_ai_imperfect" if kind == "built_in_ai" else "external",
            )
        )
        comparison_role = raw.get("comparison_role")
        if comparison_role is not None:
            comparison_role = str(comparison_role)
            if comparison_role not in {"random_reference", "bc_reference"}:
                raise ValueError(f"unsupported comparison_role for policy {policy_id}")
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                kind=kind,
                modes=modes,
                checkpoint=checkpoint,
                continuous_std=continuous_std,
                ego_mode=ego_mode,
                comparison_role=comparison_role,
            )
        )

    roles = tuple(str(item) for item in _required_list(payload, "roles"))
    if any(role not in {"fighter1", "fighter2"} for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("manifest roles must contain unique fighter1/fighter2 values")
    raw_seeds = _required_list(payload, "seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in raw_seeds):
        raise ValueError("manifest seeds must be non-negative integers")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("manifest seeds must be unique")

    num_envs = int(payload.get("num_envs", 16))
    ticks_per_step = int(payload.get("ticks_per_step", 1))
    max_sim_seconds = float(payload.get("max_sim_seconds", 120.0))
    max_steps = int(payload.get("max_steps", 0))
    shot_window_threshold = float(payload.get("shot_window_threshold", 0.5))
    boundary_threat_seconds = float(payload.get("boundary_threat_seconds", 2.0))
    parity_stride_steps = int(payload.get("parity_stride_steps", 60))
    if num_envs <= 0 or ticks_per_step <= 0:
        raise ValueError("num_envs and ticks_per_step must be positive")
    if max_sim_seconds <= 0.0 and max_steps <= 0:
        raise ValueError("at least one of max_sim_seconds or max_steps must be positive")
    if not 0.0 <= shot_window_threshold <= 1.0:
        raise ValueError("shot_window_threshold must be in [0, 1]")
    if boundary_threat_seconds <= 0.0 or parity_stride_steps <= 0:
        raise ValueError("boundary_threat_seconds and parity_stride_steps must be positive")
    return GateManifest(
        source_path=source_path,
        dataset_root=dataset_root,
        scenes=tuple(scenes),
        policies=tuple(policies),
        roles=roles,
        seeds=seeds,
        opponent_mode=str(payload.get("opponent_mode", "built_in_ai_imperfect")),
        num_envs=num_envs,
        ticks_per_step=ticks_per_step,
        max_sim_seconds=max_sim_seconds,
        max_steps=max_steps,
        shot_window_threshold=shot_window_threshold,
        boundary_threat_seconds=boundary_threat_seconds,
        parity_stride_steps=parity_stride_steps,
    )


def _materialize_scenes(manifest: GateManifest, output_dir: Path) -> tuple[SceneSpec, ...]:
    scene_dir = output_dir / "assets" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    materialized: list[SceneSpec] = []
    for scene in manifest.scenes:
        actual_sha = _sha256_file(scene.path)
        if scene.expected_sha256 is not None and scene.expected_sha256 != actual_sha:
            raise ValueError(
                f"scene {scene.scene_id} SHA-256 mismatch: {actual_sha}, expected {scene.expected_sha256}"
            )
        suffix = scene.path.suffix or ".ron"
        target = scene_dir / f"{scene.scene_id}-{actual_sha[:12]}{suffix}"
        shutil.copyfile(scene.path, target)
        materialized.append(SceneSpec(scene.scene_id, target.resolve(), actual_sha))
    return tuple(materialized)


def _load_checkpoint_candidate(
    policy: PolicySpec,
    *,
    normalizer: ObservationNormalizer,
    dataset_contract: PolicyDatasetContract,
    device: torch.device,
) -> StatelessHybridActorCritic | None:
    if policy.kind != "checkpoint":
        return None
    if policy.checkpoint is None:
        raise ValueError(f"checkpoint path missing for policy {policy.policy_id}")
    return load_policy_model(
        policy.checkpoint,
        normalizer=normalizer,
        dataset_contract=dataset_contract,
        device=device,
        context=f"capability gate policy {policy.policy_id}",
    )


def _count_hits(
    events: list[dict[str, Any]],
    *,
    ego_role: str,
    enemy_role: str,
) -> tuple[int, int]:
    self_hits = 0
    enemy_hits = 0
    for event in events:
        if str(event.get("kind")) not in {
            "Hit",
            "Damage",
            "SubsystemHit",
            "SubsystemDestroyed",
            "Destroy",
            "Kill",
        }:
            continue
        subject = event.get("subject")
        if subject == ego_role:
            self_hits += 1
        elif subject == enemy_role:
            enemy_hits += 1
    return self_hits, enemy_hits


def _termination_reason(
    *,
    info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    limit_reason: str | None,
) -> str:
    self_state = info["aircraft_by_role"][info["ego_role"]]
    enemy_state = info["aircraft_by_role"][info["enemy_role"]]
    self_destroyed = bool(self_state["destroyed"])
    enemy_destroyed = bool(enemy_state["destroyed"])
    if self_destroyed and enemy_destroyed:
        return "mutual_destroyed"
    if self_destroyed and float(self_state.get("out_of_bounds_seconds", 0.0)) > 0.0:
        return "self_out_of_bounds_destroyed"
    if self_destroyed:
        return "self_destroyed"
    if enemy_destroyed:
        return "enemy_destroyed"
    if truncated:
        return limit_reason or "environment_truncated"
    if terminated:
        return "unknown_terminated"
    return "incomplete"


def _boundary_threat(state: dict[str, Any], *, threshold_seconds: float) -> bool:
    if float(state.get("out_of_bounds_seconds", 0.0)) > 0.0:
        return True
    for key in (
        "time_to_ground_impact_s",
        "time_to_ceiling_impact_s",
        "time_to_horizontal_boundary_impact_s",
    ):
        value = state.get(key)
        if value is not None and 0.0 <= float(value) <= threshold_seconds:
            return True
    return False


def _actual_binary_from_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            obs[_FIELD_INDICES["self_brake_active"]],
            obs[_FIELD_INDICES["self_fire_gun_active"]],
            obs[_FIELD_INDICES["self_repair_active"]],
        ],
        dtype=np.float32,
    )


def _update_accumulator(
    accumulator: EpisodeAccumulator,
    *,
    obs: np.ndarray,
    info: dict[str, Any],
    requested_cont: np.ndarray,
    binary_probabilities: np.ndarray,
    action_source_observable: bool,
    shot_window_threshold: float,
    boundary_threat_seconds: float,
) -> None:
    sim_time = float(info["sim_time_seconds"])
    dt = max(sim_time - accumulator.previous_sim_time_seconds, 0.0)
    elapsed = max(sim_time - accumulator.start_sim_time_seconds, 0.0)
    accumulator.previous_sim_time_seconds = sim_time
    accumulator.steps += 1

    self_state = info["aircraft_by_role"][info["ego_role"]]
    self_shot = float(obs[_FIELD_INDICES["self_shot_feasibility"]])
    enemy_shot = float(obs[_FIELD_INDICES["enemy_shot_feasibility"]])
    tracking = float(obs[_FIELD_INDICES["self_tracking_quality"]])
    self_tail = float(obs[_FIELD_INDICES["self_tail_hold_score"]])
    enemy_tail = float(obs[_FIELD_INDICES["enemy_tail_hold_score"]])
    actual_binary = _actual_binary_from_obs(obs)
    firing = actual_binary[1] >= 0.5
    if self_shot >= shot_window_threshold:
        if accumulator.first_shot_window_seconds is None:
            accumulator.first_shot_window_seconds = elapsed
        accumulator.shot_window_seconds += dt
    if enemy_shot >= shot_window_threshold:
        accumulator.enemy_shot_window_seconds += dt
    if firing:
        accumulator.fire_seconds += dt
        if self_shot >= shot_window_threshold:
            accumulator.effective_fire_seconds += dt
    accumulator.tracking_integral += tracking * dt
    accumulator.tail_integral += self_tail * dt
    accumulator.enemy_tail_integral += enemy_tail * dt
    accumulator.advantage_integral += (self_tail - enemy_tail) * dt
    accumulator.speed_integral += float(self_state["speed"]) * dt
    accumulator.stall_integral += float(self_state["stall_factor"]) * dt
    if _boundary_threat(self_state, threshold_seconds=boundary_threat_seconds):
        accumulator.boundary_threat_seconds += dt
    out_of_bounds_seconds = float(self_state.get("out_of_bounds_seconds", 0.0))
    if out_of_bounds_seconds > 0.0:
        accumulator.out_of_bounds_seconds += dt
    accumulator.max_out_of_bounds_seconds = max(
        accumulator.max_out_of_bounds_seconds,
        out_of_bounds_seconds,
    )
    self_hits, enemy_hits = _count_hits(
        info.get("events_since_last_step", []),
        ego_role=info["ego_role"],
        enemy_role=info["enemy_role"],
    )
    accumulator.self_hit_count += self_hits
    accumulator.enemy_hit_count += enemy_hits
    accumulator.actual_binary_on_sum += actual_binary
    accumulator.binary_probability_sum += binary_probabilities
    accumulator.action_sample_count += 1
    accumulator.action_source_observable = action_source_observable
    if action_source_observable:
        requested = np.asarray(requested_cont, dtype=np.float64)
        accumulator.requested_cont_sum += requested
        accumulator.requested_cont_sq_sum += np.square(requested)
        accumulator.requested_cont_saturated += np.abs(requested) >= 0.95
        if accumulator.previous_requested_cont is not None:
            accumulator.requested_cont_delta_sum += np.abs(
                requested - accumulator.previous_requested_cont
            )
            accumulator.requested_cont_delta_count += 1
        accumulator.previous_requested_cont = requested.copy()


def _finalize_episode(accumulator: EpisodeAccumulator, info: dict[str, Any]) -> dict[str, Any]:
    duration = max(
        float(info["sim_time_seconds"]) - accumulator.start_sim_time_seconds,
        0.0,
    )
    safe_duration = max(duration, 1e-9)
    count = max(accumulator.action_sample_count, 1)
    requested_action: dict[str, Any] | None = None
    if accumulator.action_source_observable:
        mean = accumulator.requested_cont_sum / count
        variance = np.maximum(accumulator.requested_cont_sq_sum / count - np.square(mean), 0.0)
        requested_action = {
            "mean": dict(zip(CONTINUOUS_ACTION_NAMES, mean.tolist(), strict=True)),
            "std": dict(
                zip(CONTINUOUS_ACTION_NAMES, np.sqrt(variance).tolist(), strict=True)
            ),
            "saturation_fraction": dict(
                zip(
                    CONTINUOUS_ACTION_NAMES,
                    (accumulator.requested_cont_saturated / count).tolist(),
                    strict=True,
                )
            ),
            "mean_abs_step_delta": dict(
                zip(
                    CONTINUOUS_ACTION_NAMES,
                    (
                        accumulator.requested_cont_delta_sum
                        / max(accumulator.requested_cont_delta_count, 1)
                    ).tolist(),
                    strict=True,
                )
            ),
        }
    return {
        "seed": accumulator.seed,
        "steps": accumulator.steps,
        "duration_seconds": duration,
        "terminated": accumulator.terminated,
        "truncated": accumulator.truncated,
        "termination_reason": accumulator.termination_reason,
        "self_destroyed": accumulator.self_destroyed,
        "enemy_destroyed": accumulator.enemy_destroyed,
        "self_hit_count": accumulator.self_hit_count,
        "enemy_hit_count": accumulator.enemy_hit_count,
        "first_shot_window_seconds": accumulator.first_shot_window_seconds,
        "shot_window_seconds": accumulator.shot_window_seconds,
        "shot_window_fraction": accumulator.shot_window_seconds / safe_duration,
        "enemy_shot_window_fraction": accumulator.enemy_shot_window_seconds / safe_duration,
        "fire_fraction": accumulator.fire_seconds / safe_duration,
        "effective_fire_fraction": (
            accumulator.effective_fire_seconds / max(accumulator.fire_seconds, 1e-9)
        ),
        "shot_window_utilization": (
            accumulator.effective_fire_seconds / max(accumulator.shot_window_seconds, 1e-9)
        ),
        "tracking_quality_mean": accumulator.tracking_integral / safe_duration,
        "tail_hold_score_mean": accumulator.tail_integral / safe_duration,
        "enemy_tail_hold_score_mean": accumulator.enemy_tail_integral / safe_duration,
        "tail_advantage_mean": accumulator.advantage_integral / safe_duration,
        "speed_mean_mps": accumulator.speed_integral / safe_duration,
        "stall_factor_mean": accumulator.stall_integral / safe_duration,
        "boundary_threat_fraction": accumulator.boundary_threat_seconds / safe_duration,
        "out_of_bounds_fraction": accumulator.out_of_bounds_seconds / safe_duration,
        "max_out_of_bounds_seconds": accumulator.max_out_of_bounds_seconds,
        "requested_continuous_action": requested_action,
        "binary_probability_mean": (
            dict(
                zip(
                    BINARY_ACTION_NAMES,
                    (accumulator.binary_probability_sum / count).tolist(),
                    strict=True,
                )
            )
            if accumulator.action_source_observable
            else None
        ),
        "actual_binary_on_fraction": dict(
            zip(
                BINARY_ACTION_NAMES,
                (accumulator.actual_binary_on_sum / count).tolist(),
                strict=True,
            )
        ),
    }


def _aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode list")

    def mean(key: str) -> float:
        return float(np.mean([float(item[key]) for item in episodes]))

    durations = np.asarray([item["duration_seconds"] for item in episodes], dtype=np.float64)
    reasons: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    first_windows = [
        float(item["first_shot_window_seconds"])
        for item in episodes
        if item["first_shot_window_seconds"] is not None
    ]
    action_available = all(item["requested_continuous_action"] is not None for item in episodes)
    action_summary = None
    if action_available:
        action_summary = {
            section: {
                name: float(
                    np.mean(
                        [
                            item["requested_continuous_action"][section][name]
                            for item in episodes
                        ]
                    )
                )
                for name in CONTINUOUS_ACTION_NAMES
            }
            for section in ("mean", "std", "saturation_fraction", "mean_abs_step_delta")
        }
    return {
        "episode_count": len(episodes),
        "enemy_destroy_rate": mean("enemy_destroyed"),
        "self_destroy_rate": mean("self_destroyed"),
        "mutual_destroy_rate": float(
            np.mean(
                [
                    float(item["enemy_destroyed"] and item["self_destroyed"])
                    for item in episodes
                ]
            )
        ),
        "timeout_rate": float(
            np.mean(
                [
                    float(
                        item["termination_reason"]
                        in {"capability_step_limit", "capability_time_limit"}
                    )
                    for item in episodes
                ]
            )
        ),
        "termination_reason_counts": reasons,
        "duration_seconds": {
            "mean": float(np.mean(durations)),
            "p10": float(np.quantile(durations, 0.10)),
            "p50": float(np.quantile(durations, 0.50)),
            "p90": float(np.quantile(durations, 0.90)),
            "max": float(np.max(durations)),
        },
        "shot_window_episode_rate": float(
            np.mean([float(item["first_shot_window_seconds"] is not None) for item in episodes])
        ),
        "first_shot_window_seconds_mean": (
            float(np.mean(first_windows)) if first_windows else None
        ),
        "shot_window_fraction_mean": mean("shot_window_fraction"),
        "enemy_shot_window_fraction_mean": mean("enemy_shot_window_fraction"),
        "effective_fire_fraction_mean": mean("effective_fire_fraction"),
        "shot_window_utilization_mean": mean("shot_window_utilization"),
        "tracking_quality_mean": mean("tracking_quality_mean"),
        "tail_hold_score_mean": mean("tail_hold_score_mean"),
        "enemy_tail_hold_score_mean": mean("enemy_tail_hold_score_mean"),
        "tail_advantage_mean": mean("tail_advantage_mean"),
        "speed_mean_mps": mean("speed_mean_mps"),
        "stall_factor_mean": mean("stall_factor_mean"),
        "boundary_threat_fraction_mean": mean("boundary_threat_fraction"),
        "out_of_bounds_fraction_mean": mean("out_of_bounds_fraction"),
        "self_hit_count_mean": mean("self_hit_count"),
        "enemy_hit_count_mean": mean("enemy_hit_count"),
        "requested_continuous_action": action_summary,
        "binary_probability_mean": (
            {
                name: float(
                    np.mean(
                        [item["binary_probability_mean"][name] for item in episodes]
                    )
                )
                for name in BINARY_ACTION_NAMES
            }
            if all(item["binary_probability_mean"] is not None for item in episodes)
            else None
        ),
        "actual_binary_on_fraction": {
            name: float(
                np.mean([item["actual_binary_on_fraction"][name] for item in episodes])
            )
            for name in BINARY_ACTION_NAMES
        },
    }


def _checkpoint_actions(
    *,
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    obs: np.ndarray,
    device: torch.device,
    mode: str,
    continuous_std: float,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = policy_output_batch(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=device,
        mode=mode,
        continuous_std=continuous_std,
        generator=generator,
    )
    return output.action_cont, output.action_bin, output.action_bin_prob


def _run_seed_batch(
    *,
    manifest: GateManifest,
    scene: SceneSpec,
    policy: PolicySpec,
    role: str,
    mode: str,
    seeds: tuple[int, ...],
    model: StatelessHybridActorCritic | None,
    normalizer: ObservationNormalizer,
    device: torch.device,
    project_root: str | None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    num_envs = len(seeds)
    ego_mode = policy.ego_mode if policy.kind == "built_in_ai" else "external"
    config = PolicyDogfightEnvConfig(
        project_root=project_root,
        scene_path=str(scene.path),
        ego_role=role,
        ego_mode=ego_mode,
        opponent_mode=manifest.opponent_mode,
        seed=seeds[0],
        ticks_per_step=manifest.ticks_per_step,
    )
    accumulators: list[EpisodeAccumulator] = []
    parity_max_cont = 0.0
    parity_max_prob = 0.0
    parity_binary_mismatches = 0
    parity_samples = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(_stable_seed(scene.scene_id, role, mode, seeds))
    random_generators = [
        np.random.default_rng(_stable_seed("random", scene.scene_id, role, seed))
        for seed in seeds
    ]
    with SubprocPolicyVecEnv(config, num_envs=num_envs, reward_mode="main") as envs:
        reset_results = envs.reset_many(
            [
                ResetRequest(seed=seed, scene_path=str(scene.path))
                for seed in seeds
            ]
        )
        observations = np.stack([result.obs for result in reset_results]).astype(
            np.float32,
            copy=False,
        )
        for seed, reset_result in zip(seeds, reset_results, strict=True):
            start_time = float(reset_result.info["sim_time_seconds"])
            accumulators.append(
                EpisodeAccumulator(
                    seed=seed,
                    start_sim_time_seconds=start_time,
                    previous_sim_time_seconds=start_time,
                    action_source_observable=policy.kind != "built_in_ai",
                )
            )
        done = np.zeros((num_envs,), dtype=np.bool_)
        global_step = 0
        last_infos = [result.info for result in reset_results]
        while not bool(done.all()):
            if policy.kind == "checkpoint":
                if model is None:
                    raise RuntimeError(f"model not loaded for policy {policy.policy_id}")
                action_cont, action_bin, action_bin_prob = _checkpoint_actions(
                    model=model,
                    normalizer=normalizer,
                    obs=observations,
                    device=device,
                    mode=mode,
                    continuous_std=policy.continuous_std,
                    generator=generator,
                )
                if mode == "deterministic" and global_step % manifest.parity_stride_steps == 0:
                    active_indices = np.flatnonzero(~done)
                    if active_indices.size:
                        index = int(active_indices[0])
                        single = deterministic_policy_output(
                            model=model,
                            normalizer=normalizer,
                            obs=observations[index],
                            device=device,
                        )
                        parity_max_cont = max(
                            parity_max_cont,
                            float(np.max(np.abs(single.action_cont - action_cont[index]))),
                        )
                        parity_max_prob = max(
                            parity_max_prob,
                            float(
                                np.max(
                                    np.abs(single.action_bin_prob - action_bin_prob[index])
                                )
                            ),
                        )
                        parity_binary_mismatches += int(
                            np.any(single.binary_actions() != action_bin[index])
                        )
                        parity_samples += 1
            elif policy.kind == "random":
                action_cont = np.stack(
                    [
                        generator.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
                        for generator in random_generators
                    ]
                )
                action_bin_prob = np.full((num_envs, 3), 0.5, dtype=np.float32)
                action_bin = np.stack(
                    [
                        generator.integers(0, 2, size=(3,)).astype(np.float32)
                        for generator in random_generators
                    ]
                )
            else:
                action_cont = np.zeros((num_envs, 4), dtype=np.float32)
                action_bin = np.zeros((num_envs, 3), dtype=np.float32)
                action_bin_prob = np.zeros((num_envs, 3), dtype=np.float32)
            action_cont[done] = 0.0
            action_bin[done] = 0.0
            action_bin_prob[done] = 0.0
            step_results = envs.step_many(
                [
                    StepRequest(action_cont[index], action_bin[index])
                    for index in range(num_envs)
                ]
            )
            for index, result in enumerate(step_results):
                if done[index]:
                    continue
                observations[index] = result.obs
                last_infos[index] = result.info
                accumulator = accumulators[index]
                _update_accumulator(
                    accumulator,
                    obs=result.obs,
                    info=result.info,
                    requested_cont=action_cont[index],
                    binary_probabilities=action_bin_prob[index],
                    action_source_observable=policy.kind != "built_in_ai",
                    shot_window_threshold=manifest.shot_window_threshold,
                    boundary_threat_seconds=manifest.boundary_threat_seconds,
                )
                elapsed = (
                    float(result.info["sim_time_seconds"])
                    - accumulator.start_sim_time_seconds
                )
                limit_reason = None
                if manifest.max_steps > 0 and accumulator.steps >= manifest.max_steps:
                    limit_reason = "capability_step_limit"
                if (
                    limit_reason is None
                    and manifest.max_sim_seconds > 0.0
                    and elapsed >= manifest.max_sim_seconds
                ):
                    limit_reason = "capability_time_limit"
                terminated = bool(result.terminated)
                truncated = bool(result.truncated or (limit_reason is not None and not terminated))
                if terminated or truncated:
                    accumulator.terminated = terminated
                    accumulator.truncated = truncated
                    self_state = result.info["aircraft_by_role"][result.info["ego_role"]]
                    enemy_state = result.info["aircraft_by_role"][result.info["enemy_role"]]
                    accumulator.self_destroyed = bool(self_state["destroyed"])
                    accumulator.enemy_destroyed = bool(enemy_state["destroyed"])
                    accumulator.termination_reason = _termination_reason(
                        info=result.info,
                        terminated=terminated,
                        truncated=truncated,
                        limit_reason=limit_reason,
                    )
                    done[index] = True
            global_step += 1
    return (
        [
            _finalize_episode(accumulator, info)
            for accumulator, info in zip(accumulators, last_infos, strict=True)
        ],
        {
            "sample_count": parity_samples,
            "max_abs_continuous_difference": parity_max_cont,
            "max_abs_binary_probability_difference": parity_max_prob,
            "binary_action_mismatch_count": parity_binary_mismatches,
        },
    )


def _run_experiment(
    *,
    manifest: GateManifest,
    scene: SceneSpec,
    policy: PolicySpec,
    role: str,
    mode: str,
    model: StatelessHybridActorCritic | None,
    normalizer: ObservationNormalizer,
    device: torch.device,
    project_root: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    parity = {
        "sample_count": 0,
        "max_abs_continuous_difference": 0.0,
        "max_abs_binary_probability_difference": 0.0,
        "binary_action_mismatch_count": 0,
    }
    for offset in range(0, len(manifest.seeds), manifest.num_envs):
        seed_batch = manifest.seeds[offset : offset + manifest.num_envs]
        batch_episodes, batch_parity = _run_seed_batch(
            manifest=manifest,
            scene=scene,
            policy=policy,
            role=role,
            mode=mode,
            seeds=seed_batch,
            model=model,
            normalizer=normalizer,
            device=device,
            project_root=project_root,
        )
        episodes.extend(batch_episodes)
        parity["sample_count"] += int(batch_parity["sample_count"])
        parity["max_abs_continuous_difference"] = max(
            float(parity["max_abs_continuous_difference"]),
            float(batch_parity["max_abs_continuous_difference"]),
        )
        parity["max_abs_binary_probability_difference"] = max(
            float(parity["max_abs_binary_probability_difference"]),
            float(batch_parity["max_abs_binary_probability_difference"]),
        )
        parity["binary_action_mismatch_count"] += int(
            batch_parity["binary_action_mismatch_count"]
        )
    aggregate = _aggregate_episodes(episodes)
    aggregate["wall_time_seconds"] = time.perf_counter() - started
    aggregate["simulation_steps_per_second"] = (
        sum(int(item["steps"]) for item in episodes)
        / max(float(aggregate["wall_time_seconds"]), 1e-9)
    )
    aggregate["inference_parity"] = parity
    return episodes, aggregate


def _comparison_key(experiment: dict[str, Any]) -> tuple[str, str]:
    return str(experiment["scene_id"]), str(experiment["ego_role"])


def _build_bc_gate(
    experiments: list[dict[str, Any]],
    policies: tuple[PolicySpec, ...],
) -> dict[str, Any]:
    random_ids = {
        policy.policy_id
        for policy in policies
        if policy.comparison_role == "random_reference"
    }
    bc_ids = {
        policy.policy_id
        for policy in policies
        if policy.comparison_role == "bc_reference"
    }
    random_by_key = {
        _comparison_key(experiment): experiment
        for experiment in experiments
        if experiment["policy_id"] in random_ids
    }
    bc_by_key = {
        _comparison_key(experiment): experiment
        for experiment in experiments
        if experiment["policy_id"] in bc_ids
        and experiment["inference_mode"] == "deterministic"
    }
    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(random_by_key) & set(bc_by_key)):
        random_metrics = random_by_key[key]["aggregate"]
        bc_metrics = bc_by_key[key]["aggregate"]
        conditions = {
            "enemy_destroy_rate_not_worse": (
                bc_metrics["enemy_destroy_rate"] >= random_metrics["enemy_destroy_rate"]
            ),
            "self_destroy_rate_not_worse": (
                bc_metrics["self_destroy_rate"] <= random_metrics["self_destroy_rate"]
            ),
            "shot_window_rate_better": (
                bc_metrics["shot_window_episode_rate"]
                > random_metrics["shot_window_episode_rate"]
            ),
            "boundary_threat_not_worse": (
                bc_metrics["boundary_threat_fraction_mean"]
                <= random_metrics["boundary_threat_fraction_mean"]
            ),
        }
        comparisons.append(
            {
                "scene_id": key[0],
                "ego_role": key[1],
                "passed": all(conditions.values()),
                "conditions": conditions,
                "delta": {
                    metric: float(bc_metrics[metric] - random_metrics[metric])
                    for metric in (
                        "enemy_destroy_rate",
                        "self_destroy_rate",
                        "shot_window_episode_rate",
                        "boundary_threat_fraction_mean",
                    )
                },
            }
        )
    return {
        "available": bool(comparisons),
        "passed": bool(comparisons) and all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }


def run_capability_gate(
    *,
    manifest: GateManifest,
    output_dir: Path,
    device: torch.device,
    project_root: str | None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"capability gate output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_scenes = _materialize_scenes(manifest, output_dir)
    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(
        manifest.dataset_root
    )
    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    checkpoint_hashes = {
        policy.policy_id: (
            None if policy.checkpoint is None else _sha256_file(policy.checkpoint)
        )
        for policy in manifest.policies
    }
    episode_path = output_dir / "episodes.jsonl"
    experiments: list[dict[str, Any]] = []
    experiment_cache: dict[
        tuple[str, float, str, str, str],
        tuple[str, list[dict[str, Any]], dict[str, Any]],
    ] = {}
    with episode_path.open("w", encoding="utf-8") as episode_file:
        for policy in manifest.policies:
            model = _load_checkpoint_candidate(
                policy,
                normalizer=normalizer,
                dataset_contract=dataset_contract,
                device=device,
            )
            for scene in materialized_scenes:
                for role in manifest.roles:
                    for mode in policy.modes:
                        print(
                            f"[CapabilityGate] policy={policy.policy_id} mode={mode} "
                            f"scene={scene.scene_id} role={role}",
                            flush=True,
                        )
                        checkpoint_hash = checkpoint_hashes[policy.policy_id]
                        cache_key = None
                        if checkpoint_hash is not None:
                            cache_key = (
                                checkpoint_hash,
                                policy.continuous_std,
                                mode,
                                scene.expected_sha256 or "",
                                role,
                            )
                        cached = None if cache_key is None else experiment_cache.get(cache_key)
                        if cached is None:
                            episodes, aggregate = _run_experiment(
                                manifest=manifest,
                                scene=scene,
                                policy=policy,
                                role=role,
                                mode=mode,
                                model=model,
                                normalizer=normalizer,
                                device=device,
                                project_root=project_root,
                            )
                            if cache_key is not None:
                                experiment_cache[cache_key] = (
                                    policy.policy_id,
                                    copy.deepcopy(episodes),
                                    copy.deepcopy(aggregate),
                                )
                        else:
                            reused_policy_id, cached_episodes, cached_aggregate = cached
                            episodes = copy.deepcopy(cached_episodes)
                            aggregate = copy.deepcopy(cached_aggregate)
                            aggregate["reused_from_policy_id"] = reused_policy_id
                        identity = {
                            "policy_id": policy.policy_id,
                            "policy_type": policy.kind,
                            "inference_mode": mode,
                            "scene_id": scene.scene_id,
                            "scene_sha256": scene.expected_sha256,
                            "ego_role": role,
                        }
                        for episode in episodes:
                            episode_file.write(
                                json.dumps({**identity, **episode}, ensure_ascii=False)
                                + "\n"
                            )
                        experiment = {**identity, "aggregate": aggregate}
                        experiments.append(experiment)
                        print(
                            f"[CapabilityGate] enemy_destroy={aggregate['enemy_destroy_rate']:.3f} "
                            f"self_destroy={aggregate['self_destroy_rate']:.3f} "
                            f"timeout={aggregate['timeout_rate']:.3f} "
                            f"shot_window={aggregate['shot_window_episode_rate']:.3f} "
                            f"sps={aggregate['simulation_steps_per_second']:.1f}",
                            flush=True,
                        )

    resolved_manifest = {
        "schema_id": GATE_SCHEMA_ID,
        "source_manifest": str(manifest.source_path),
        "dataset_root": str(manifest.dataset_root),
        "dataset_id": dataset_contract.dataset_id,
        "contract_sha256": dataset_contract.contract_sha256,
        "scenes": [
            {
                "id": scene.scene_id,
                "path": str(scene.path),
                "sha256": scene.expected_sha256,
            }
            for scene in materialized_scenes
        ],
        "policies": [
            {
                **asdict(policy),
                "checkpoint": None if policy.checkpoint is None else str(policy.checkpoint),
                "checkpoint_sha256": (
                    checkpoint_hashes[policy.policy_id]
                ),
            }
            for policy in manifest.policies
        ],
        "roles": list(manifest.roles),
        "seeds": list(manifest.seeds),
        "opponent_mode": manifest.opponent_mode,
        "num_envs": manifest.num_envs,
        "ticks_per_step": manifest.ticks_per_step,
        "max_sim_seconds": manifest.max_sim_seconds,
        "max_steps": manifest.max_steps,
        "shot_window_threshold": manifest.shot_window_threshold,
        "boundary_threat_seconds": manifest.boundary_threat_seconds,
        "parity_stride_steps": manifest.parity_stride_steps,
        "device": str(device),
    }
    summary = {
        "schema_id": GATE_SCHEMA_ID,
        "manifest": resolved_manifest,
        "experiments": experiments,
        "bc_vs_random_gate": _build_bc_gate(experiments, manifest.policies),
    }
    (output_dir / "resolved_manifest.json").write_text(
        json.dumps(resolved_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = _parse_args()
    manifest = load_manifest(args.manifest)
    summary = run_capability_gate(
        manifest=manifest,
        output_dir=Path(args.output_dir).resolve(),
        device=_resolve_device(args.device),
        project_root=args.project_root,
    )
    gate = summary["bc_vs_random_gate"]
    print(
        f"[CapabilityGate] completed experiments={len(summary['experiments'])} "
        f"bc_vs_random_available={int(gate['available'])} "
        f"bc_vs_random_passed={int(gate['passed'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
