from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.envs import (
    PolicyDogfightEnv,
    PolicyDogfightEnvConfig,
    ResetRequest,
    ResetResult,
    StepRequest,
    StepResult,
    SubprocPolicyVecEnv,
    WorkerRewardStateMachine,
)
from dfb_reinforcement_learning.models.stateless_hybrid_actor_critic import (
    StatelessHybridActorCritic,
    model_architecture_kwargs,
)
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    load_and_validate_policy_dataset,
    validate_policy_checkpoint_payload,
)
from dfb_reinforcement_learning.policy_inference import (
    deterministic_policy_output,
    normalize_policy_observation,
)
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig
from dfb_reinforcement_learning.rewards.reward_diagnostics import (
    REWARD_COMPONENT_FIELDS,
    RewardDiagnosticsAccumulator,
)
from dfb_reinforcement_learning.opponents import (
    OpponentActionProvider,
    OpponentPoolSpec,
    PreparedOpponentPool,
    SampledOpponent,
    materialize_opponent_pool,
)
from dfb_reinforcement_learning.scenes import MaterializedScene, PreparedScenePool, ScenePoolSpec, materialize_scene_pool
from dfb_reinforcement_learning.train.truncation import build_truncation_policy
from dfb_reinforcement_learning.train.ppo_diagnostics import (
    summarize_actions,
    summarize_values,
)

PPO_TRAINING_SEMANTICS_SCHEMA_ID = "dfb.part3.ppo-training-semantics.v2"
PPO_CONTINUOUS_DISTRIBUTION_ID = "dfb.part3.clipped-normal-action.v1"


def _count_hits_from_events(
    events: list[dict[str, Any]],
    *,
    self_role: str,
    enemy_role: str,
) -> tuple[int, int]:
    """Count the same authoritative Hit events used by the reward."""
    self_hits = 0
    enemy_hits = 0
    self_prefix = f"{self_role}:"
    enemy_prefix = f"{enemy_role}:"
    for event in events:
        if event.get("kind") != "Hit":
            continue
        subject = event.get("subject")
        if not isinstance(subject, str):
            continue
        if subject == self_role or subject.startswith(self_prefix):
            self_hits += 1
        elif subject == enemy_role or subject.startswith(enemy_prefix):
            enemy_hits += 1
    return self_hits, enemy_hits


@dataclass(frozen=True)
class EpisodeRecord:
    episode_return: float
    episode_length: int
    episode_survival_seconds: float
    self_destroyed: bool
    enemy_destroyed: bool
    terminated: bool
    truncated: bool
    max_out_of_bounds_seconds: float
    self_hit_count: int = 0
    enemy_hit_count: int = 0
    termination_reason: str = "unknown"
    reward_diagnostics: dict[str, Any] | None = None
    opponent_label: str = "unknown"
    opponent_kind: str = "unknown"
    scene_label: str = "unknown"


@dataclass(frozen=True)
class FrozenSceneAsset:
    scene_name: str
    scene_path: str
    manifest_path: str


@dataclass(frozen=True)
class PpoEvalMetrics:
    mean_return: float
    mean_length: float
    mean_survival_seconds: float
    self_destroy_rate: float
    enemy_destroy_rate: float
    mean_max_out_of_bounds_seconds: float
    episode_count: int
    self_mean_hit_count: float = 0.0
    enemy_mean_hit_count: float = 0.0
    mutual_destroy_rate: float = 0.0
    truncated_rate: float = 0.0
    timeout_rate: float = 0.0
    termination_reason_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class EpisodeAggregateMetrics:
    episode_count: int
    mean_return: float
    mean_length: float
    mean_duration_seconds: float
    self_destroy_rate: float
    enemy_destroy_rate: float
    mutual_destroy_rate: float
    out_of_bounds_destroy_rate: float
    truncated_rate: float
    mean_max_out_of_bounds_seconds: float
    self_mean_hit_count: float
    enemy_mean_hit_count: float
    termination_reason_counts: dict[str, int]


@dataclass(frozen=True)
class PpoTrainMetrics:
    update: int
    global_step: int
    rollout_return_mean: float
    rollout_length_mean: float
    rollout_survival_seconds_mean: float
    rollout_episode_duration_seconds_mean: float
    rollout_self_destroy_rate: float
    rollout_enemy_destroy_rate: float
    rollout_mutual_destroy_rate: float
    rollout_out_of_bounds_destroy_rate: float
    rollout_truncated_rate: float
    rollout_mean_max_out_of_bounds_seconds: float
    rollout_self_mean_hit_count: float
    rollout_enemy_mean_hit_count: float
    rollout_episode_count: int
    rollout_termination_reason_counts: dict[str, int]
    rollout_window: EpisodeAggregateMetrics
    policy_loss: float
    value_loss: float
    continuous_entropy: float
    binary_entropy: float
    approx_kl: float
    max_approx_kl: float
    clip_fraction: float
    target_kl: float
    target_kl_triggered: bool
    ppo_epochs_completed: int
    ppo_minibatches_completed: int
    shared_learning_rate: float
    actor_learning_rate: float
    critic_learning_rate: float
    reward_mean: float
    advantage_mean: float
    action_diagnostics: dict[str, Any]
    reward_diagnostics: dict[str, Any]
    reward_diagnostics_by_outcome: dict[str, Any]
    value_diagnostics: dict[str, float]
    policy_reference_diagnostics: dict[str, Any] | None
    rollout_collection_seconds: float
    rollout_steps_per_second: float
    policy_forward_seconds: float
    opponent_action_seconds: float
    env_step_seconds: float
    reward_compute_seconds: float
    diagnostics_seconds: float
    env_reset_seconds: float
    ppo_update_seconds: float
    eval_seconds: float
    checkpoint_seconds: float
    update_compute_seconds: float
    update_total_seconds: float
    eval: PpoEvalMetrics | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO under the active Part 3 policy contract.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="open_head_on_200m")
    parser.add_argument("--scene-path", default=None)
    parser.add_argument("--scene-pool-json", default=None)
    parser.add_argument("--opponent-pool-json", default=None)
    parser.add_argument("--ego-role", default="fighter1", choices=["fighter1", "fighter2"])
    parser.add_argument(
        "--opponent-mode",
        default="built_in_ai",
        choices=[
            "built_in_ai",
            "built_in_ai_precise",
            "built_in_ai_imperfect",
            "built_in_ai_passive_bounce",
            "self_play",
        ],
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--reset-critic-on-init", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--env-backend", default="serial", choices=["serial", "subproc"])
    parser.add_argument("--subproc-request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--worker-reward-mode", default="main", choices=["main", "worker"])
    parser.add_argument("--ticks-per-step", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Stop the current PPO update when minibatch approximate KL reaches this value (0 = disabled)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--shared-learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--value-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--binary-entropy-coef",
        type=float,
        default=0.0,
        help=(
            "Entropy regularization for brake/fire/repair Bernoulli actions; "
            "continuous exploration is controlled only by --continuous-action-std"
        ),
    )
    parser.add_argument("--popart-beta", type=float, default=0.999)
    parser.add_argument("--popart-min-std", type=float, default=1e-2)
    parser.add_argument("--continuous-action-std", type=float, default=0.18)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--shared-extension-blocks", type=int, default=0)
    parser.add_argument("--actor-extension-blocks", type=int, default=0)
    parser.add_argument("--critic-extension-blocks", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument(
        "--eval-max-seconds",
        type=float,
        default=120.0,
        help="Truncate each evaluation episode after this many simulation seconds (0 = disabled)",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=0,
        help="Truncate each evaluation episode after this many policy steps (0 = disabled)",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--episode-metrics-window",
        type=int,
        default=100,
        help="Number of completed training episodes retained for rolling outcome metrics",
    )
    parser.add_argument(
        "--policy-drift-reference-checkpoint",
        default=None,
        help="Reference checkpoint for actor drift diagnostics (defaults to the PPO init checkpoint)",
    )
    parser.add_argument(
        "--policy-drift-interval",
        type=int,
        default=10,
        help="Compute policy-reference drift every N updates (0 = disabled)",
    )
    parser.add_argument("--policy-drift-sample-size", type=int, default=2048)
    parser.add_argument("--opening-shot-window-mode", default="none", choices=["none", "truncate"])
    parser.add_argument("--opening-shot-window-max-seconds", type=float, default=2.5)
    parser.add_argument("--opening-shot-window-activate-threshold", type=float, default=0.08)
    parser.add_argument("--opening-shot-window-keep-threshold", type=float, default=0.04)
    parser.add_argument("--opening-shot-window-loss-seconds", type=float, default=0.35)
    parser.add_argument("--max-episode-seconds", type=float, default=0.0,
                        help="Hard truncate episodes after this many seconds (0 = disabled)")
    parser.add_argument(
        "--tactical-advantage-loss-threshold",
        type=float,
        default=None,
        help="Truncate after an established, smoothed advantage margin remains below this value",
    )
    parser.add_argument("--tactical-advantage-activate-threshold", type=float, default=0.5)
    parser.add_argument("--tactical-advantage-ema-seconds", type=float, default=0.35)
    parser.add_argument("--tactical-advantage-activate-seconds", type=float, default=0.5)
    parser.add_argument("--tactical-advantage-loss-seconds", type=float, default=0.75)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.dropout != 0.0:
        raise ValueError("PPO requires --dropout 0 so rollout and update log-probabilities remain comparable")
    if args.opening_shot_window_max_seconds <= 0.0:
        raise ValueError("--opening-shot-window-max-seconds must be positive")
    if args.opening_shot_window_loss_seconds < 0.0:
        raise ValueError("--opening-shot-window-loss-seconds must be non-negative")
    if args.opening_shot_window_keep_threshold < 0.0:
        raise ValueError("--opening-shot-window-keep-threshold must be non-negative")
    if args.opening_shot_window_activate_threshold < 0.0:
        raise ValueError("--opening-shot-window-activate-threshold must be non-negative")
    if not np.isfinite(args.max_episode_seconds) or args.max_episode_seconds < 0.0:
        raise ValueError("--max-episode-seconds must be non-negative")
    tactical_values = (
        args.tactical_advantage_activate_threshold,
        args.tactical_advantage_ema_seconds,
        args.tactical_advantage_activate_seconds,
        args.tactical_advantage_loss_seconds,
    )
    if args.tactical_advantage_loss_threshold is not None:
        tactical_values = (
            *tactical_values,
            args.tactical_advantage_loss_threshold,
        )
    if not all(np.isfinite(value) for value in tactical_values):
        raise ValueError("tactical-advantage parameters must be finite")
    if args.tactical_advantage_ema_seconds < 0.0:
        raise ValueError("--tactical-advantage-ema-seconds must be non-negative")
    if args.tactical_advantage_activate_seconds < 0.0:
        raise ValueError("--tactical-advantage-activate-seconds must be non-negative")
    if args.tactical_advantage_loss_seconds <= 0.0:
        raise ValueError("--tactical-advantage-loss-seconds must be positive")
    if (
        args.tactical_advantage_loss_threshold is not None
        and args.tactical_advantage_activate_threshold
        <= args.tactical_advantage_loss_threshold
    ):
        raise ValueError(
            "--tactical-advantage-activate-threshold must be greater than "
            "--tactical-advantage-loss-threshold"
        )
    if args.eval_max_seconds < 0.0:
        raise ValueError("--eval-max-seconds must be non-negative")
    if args.eval_max_steps < 0:
        raise ValueError("--eval-max-steps must be non-negative")
    if args.episode_metrics_window <= 0:
        raise ValueError("--episode-metrics-window must be positive")
    if args.policy_drift_interval < 0:
        raise ValueError("--policy-drift-interval must be non-negative")
    if args.policy_drift_sample_size <= 0:
        raise ValueError("--policy-drift-sample-size must be positive")
    if args.target_kl < 0.0:
        raise ValueError("--target-kl must be non-negative")
    if args.binary_entropy_coef < 0.0:
        raise ValueError("--binary-entropy-coef must be non-negative")
    for name in (
        "shared_extension_blocks",
        "actor_extension_blocks",
        "critic_extension_blocks",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path_value: str | None) -> dict[str, str] | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training semantic input does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
    }


def _resolve_project_root(path_value: str | None) -> Path:
    if path_value is None:
        return Path.cwd().resolve()
    return Path(path_value).expanduser().resolve()


def _requested_scene_path(args: argparse.Namespace) -> Path:
    if args.scene_path is not None:
        path = Path(args.scene_path).expanduser().resolve()
    else:
        path = (
            _resolve_project_root(args.project_root)
            / "config"
            / "dfb_game"
            / "scenes"
            / f"{args.scene_name}.ron"
        )
    if not path.is_file():
        raise FileNotFoundError(f"requested PPO scene does not exist: {path}")
    return path


def _materialize_base_scene_asset(
    args: argparse.Namespace,
    output_dir: Path,
) -> FrozenSceneAsset:
    asset_dir = output_dir / "scene_assets"
    manifest_path = asset_dir / "base_scene_manifest.json"
    requested_path = (
        None
        if args.scene_path is None
        else str(Path(args.scene_path).expanduser().resolve())
    )

    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "cannot resume PPO because the frozen base-scene manifest is missing; "
                "branch from the checkpoint with --init-checkpoint into a new output directory"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("requested_scene_name") != args.scene_name:
            raise ValueError(
                "PPO resume scene name differs from the frozen run "
                f"({args.scene_name!r} != {payload.get('requested_scene_name')!r})"
            )
        if payload.get("requested_scene_path") != requested_path:
            raise ValueError(
                "PPO resume scene path differs from the frozen run "
                f"({requested_path!r} != {payload.get('requested_scene_path')!r})"
            )
        scene_path = Path(str(payload["snapshot_path"]))
        expected_sha = str(payload["snapshot_sha256"])
        actual_identity = _file_identity(str(scene_path))
        if actual_identity is None or actual_identity["sha256"] != expected_sha:
            raise ValueError(f"frozen PPO base-scene hash mismatch: {scene_path}")
        return FrozenSceneAsset(
            scene_name=str(payload["scene_name"]),
            scene_path=str(scene_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
        )

    source_path = _requested_scene_path(args)
    source_identity = _file_identity(str(source_path))
    if source_identity is None:
        raise RuntimeError("resolved PPO scene identity is unexpectedly missing")
    scene_name = args.scene_name if args.scene_path is None else source_path.stem
    asset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = asset_dir / f"{scene_name}.ron"
    shutil.copyfile(source_path, snapshot_path)
    snapshot_identity = _file_identity(str(snapshot_path))
    if snapshot_identity is None:
        raise RuntimeError("frozen PPO scene identity is unexpectedly missing")
    payload = {
        "scene_name": scene_name,
        "requested_scene_name": args.scene_name,
        "requested_scene_path": requested_path,
        "source_path": source_identity["path"],
        "source_sha256": source_identity["sha256"],
        "snapshot_path": snapshot_identity["path"],
        "snapshot_sha256": snapshot_identity["sha256"],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return FrozenSceneAsset(
        scene_name=scene_name,
        scene_path=snapshot_identity["path"],
        manifest_path=str(manifest_path.resolve()),
    )


def _build_training_semantics(
    *,
    args: argparse.Namespace,
    dataset_contract: PolicyDatasetContract,
    reward_config: PolicyRewardConfig,
    base_scene_asset: FrozenSceneAsset,
    scene_pool_manifest_path: Path | None,
) -> dict[str, Any]:
    reward_payload = asdict(reward_config)
    payload = {
        "schema_id": PPO_TRAINING_SEMANTICS_SCHEMA_ID,
        "dataset_id": dataset_contract.dataset_id,
        "policy_contract_sha256": dataset_contract.contract_sha256,
        "continuous_distribution_id": PPO_CONTINUOUS_DISTRIBUTION_ID,
        "reward_config": reward_payload,
        "reward_config_sha256": _canonical_payload_sha256(reward_payload),
        "scene": {
            "requested_name": args.scene_name,
            "requested_path": (
                None
                if args.scene_path is None
                else str(Path(args.scene_path).expanduser().resolve())
            ),
            "base_snapshot": _file_identity(base_scene_asset.scene_path),
            "base_snapshot_manifest": _file_identity(base_scene_asset.manifest_path),
            "pool_spec": _file_identity(args.scene_pool_json),
            "pool_snapshot_manifest": _file_identity(
                None if scene_pool_manifest_path is None else str(scene_pool_manifest_path)
            ),
        },
        "opponent": {
            "mode": args.opponent_mode,
            "pool": _file_identity(args.opponent_pool_json),
        },
        "ego_role": args.ego_role,
        "ticks_per_step": args.ticks_per_step,
        "truncation": {
            "opening_shot_window_mode": args.opening_shot_window_mode,
            "opening_shot_window_max_seconds": args.opening_shot_window_max_seconds,
            "opening_shot_window_activate_threshold": args.opening_shot_window_activate_threshold,
            "opening_shot_window_keep_threshold": args.opening_shot_window_keep_threshold,
            "opening_shot_window_loss_seconds": args.opening_shot_window_loss_seconds,
            "max_episode_seconds": args.max_episode_seconds,
            "tactical_advantage_loss_threshold": args.tactical_advantage_loss_threshold,
            "tactical_advantage_activate_threshold": (
                args.tactical_advantage_activate_threshold
            ),
            "tactical_advantage_ema_seconds": args.tactical_advantage_ema_seconds,
            "tactical_advantage_activate_seconds": (
                args.tactical_advantage_activate_seconds
            ),
            "tactical_advantage_loss_seconds": args.tactical_advantage_loss_seconds,
        },
    }
    return {
        **payload,
        "sha256": _canonical_payload_sha256(payload),
    }


def _metrics_update_sequence(metrics_path: Path) -> list[int]:
    if not metrics_path.exists():
        return []
    updates: list[int] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                update = int(payload["update"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid PPO metrics row at {metrics_path}:{line_number}"
                ) from exc
            if updates and update <= updates[-1]:
                raise ValueError(
                    f"PPO metrics must be strictly increasing, found update {update} "
                    f"after {updates[-1]} at {metrics_path}:{line_number}"
                )
            updates.append(update)
    return updates


def _validate_output_dir_usage(args: argparse.Namespace, output_dir: Path) -> None:
    if args.resume:
        expected_latest = (output_dir / "checkpoints" / "latest.pt").resolve()
        actual_resume = Path(args.resume).expanduser().resolve()
        if actual_resume != expected_latest:
            raise ValueError(
                "--resume is only valid for the same output directory's checkpoints/latest.pt; "
                "use --init-checkpoint with a new empty --output-dir to create a branch"
            )
        if not expected_latest.is_file():
            raise FileNotFoundError(f"PPO latest checkpoint does not exist: {expected_latest}")
        payload = torch.load(expected_latest, map_location="cpu", weights_only=False)
        checkpoint_update = int(payload["update"])
        updates = _metrics_update_sequence(output_dir / "metrics.jsonl")
        if not updates:
            raise ValueError("cannot resume PPO because metrics.jsonl is missing or empty")
        if updates[-1] != checkpoint_update:
            raise ValueError(
                f"PPO resume mismatch: metrics ends at update {updates[-1]}, "
                f"but latest.pt is update {checkpoint_update}"
            )
        return

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "new PPO training requires an empty --output-dir; use --resume for an exact "
            "continuation or choose a new directory for --init-checkpoint"
        )


def _validate_resume_training_semantics(
    *,
    checkpoint_path: Path,
    expected: dict[str, Any],
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload.get("ppo_training_semantics")
    if not isinstance(saved, dict):
        raise ValueError(
            "resume checkpoint predates PPO training-semantics tracking; start a new output "
            "directory with --init-checkpoint instead of resuming it"
        )
    if saved.get("schema_id") != PPO_TRAINING_SEMANTICS_SCHEMA_ID:
        raise ValueError(f"unsupported PPO training semantics schema: {saved.get('schema_id')!r}")
    if saved.get("sha256") != expected.get("sha256"):
        raise ValueError(
            "PPO training semantics changed since the checkpoint "
            f"(saved={saved.get('sha256')}, current={expected.get('sha256')}); "
            "use --init-checkpoint with a new output directory"
        )


def _resolve_scene_override(
    scene: MaterializedScene | None,
    base_scene_asset: FrozenSceneAsset,
) -> tuple[str | None, str | None]:
    if scene is None:
        return base_scene_asset.scene_name, base_scene_asset.scene_path
    return scene.scene_name, scene.scene_path


def _opponent_mode_for_env(args: argparse.Namespace, sampled_opponent: Any) -> str:
    """Return the opponent_mode string to pass to the Rust environment.

    Translates ``self_play`` → ``model`` (which both Rust and the Python env
    understand), and falls back to the sampled opponent's env_mode when one is
    provided.
    """
    if sampled_opponent is not None:
        return sampled_opponent.env_mode
    return "model" if args.opponent_mode == "self_play" else args.opponent_mode


def _prepare_scene_pool(args: argparse.Namespace, output_dir: Path) -> PreparedScenePool | None:
    if not args.scene_pool_json:
        return None
    spec = ScenePoolSpec.from_json(Path(args.scene_pool_json))
    return materialize_scene_pool(
        spec=spec,
        output_dir=output_dir / "scene_pool",
        project_root=_resolve_project_root(args.project_root),
        reuse_existing=bool(args.resume),
    )


def _prepare_opponent_pool(args: argparse.Namespace) -> PreparedOpponentPool | None:
    if not args.opponent_pool_json:
        return None
    spec = OpponentPoolSpec.from_json(Path(args.opponent_pool_json))
    return materialize_opponent_pool(spec)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _normalize_obs(normalizer: ObservationNormalizer, obs: np.ndarray) -> np.ndarray:
    return normalize_policy_observation(normalizer, obs)


def _opposing_role(role: str) -> str:
    if role == "fighter1":
        return "fighter2"
    if role == "fighter2":
        return "fighter1"
    raise ValueError(f"unsupported fighter role: {role}")


def _info_for_role(info: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        **info,
        "ego_role": role,
        "enemy_role": _opposing_role(role),
    }


def _build_rollout_policy_observation_batch(
    *,
    ego_obs_normalized: np.ndarray,
    self_play_env_indices: list[int],
    dual_policy_slots: bool,
    opponent_obs_raw_by_env: list[np.ndarray | None],
    normalizer: ObservationNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    num_envs = ego_obs_normalized.shape[0]
    if not dual_policy_slots:
        if self_play_env_indices:
            raise ValueError("self-play episodes require dual policy slots")
        return ego_obs_normalized, np.arange(num_envs, dtype=np.int64)
    if len(opponent_obs_raw_by_env) != num_envs:
        raise ValueError("opponent observation count must match environment count")

    policy_observations = [ego_obs_normalized]
    policy_slots = [np.arange(num_envs, dtype=np.int64) * 2]
    if self_play_env_indices:
        enemy_observations: list[np.ndarray] = []
        for env_index in self_play_env_indices:
            opponent_obs_raw = opponent_obs_raw_by_env[env_index]
            if opponent_obs_raw is None:
                raise RuntimeError(
                    f"self-play environment {env_index} is missing opponent observation"
                )
            enemy_observations.append(
                _normalize_obs(normalizer, opponent_obs_raw)
            )
        policy_observations.append(np.stack(enemy_observations, axis=0))
        policy_slots.append(
            np.asarray(self_play_env_indices, dtype=np.int64) * 2 + 1
        )
    return (
        np.concatenate(policy_observations, axis=0).astype(np.float32, copy=False),
        np.concatenate(policy_slots, axis=0),
    )


@torch.no_grad()
def _compute_rollout_bootstrap_values(
    *,
    model: StatelessHybridActorCritic,
    ego_obs_normalized: np.ndarray,
    device: torch.device,
    self_play: bool,
    self_play_env_indices: list[int] | None = None,
    dual_policy_slots: bool | None = None,
    opponent_obs_raw_by_env: list[np.ndarray | None],
    normalizer: ObservationNormalizer,
) -> np.ndarray:
    resolved_self_play_indices = (
        list(range(ego_obs_normalized.shape[0]))
        if self_play_env_indices is None and self_play
        else list(self_play_env_indices or [])
    )
    resolved_dual_policy_slots = (
        self_play if dual_policy_slots is None else dual_policy_slots
    )
    bootstrap_obs, policy_slots = _build_rollout_policy_observation_batch(
        ego_obs_normalized=ego_obs_normalized,
        self_play_env_indices=resolved_self_play_indices,
        dual_policy_slots=resolved_dual_policy_slots,
        opponent_obs_raw_by_env=opponent_obs_raw_by_env,
        normalizer=normalizer,
    )
    policy_values = (
        model(torch.from_numpy(bootstrap_obs).to(device))
        .value.cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if not resolved_dual_policy_slots:
        return policy_values
    values = np.zeros((ego_obs_normalized.shape[0] * 2,), dtype=np.float32)
    values[policy_slots] = policy_values
    return values


def _new_reward_runtime(
    composer: PolicyRewardComposer,
    *,
    initial_info: dict[str, Any],
) -> WorkerRewardStateMachine:
    runtime = WorkerRewardStateMachine(composer)
    runtime.reset(initial_info=initial_info)
    return runtime


def _normal_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    variance = std.square()
    return (-0.5 * (((action - mean).square() / variance) + 2.0 * torch.log(std) + np.log(2.0 * np.pi))).sum(
        dim=-1
    )


def _clipped_normal_log_prob(
    action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    standard_lower = (-1.0 - mean) / std
    standard_upper_survival = (mean - 1.0) / std
    interior = -0.5 * (
        ((action - mean).square() / std.square())
        + 2.0 * torch.log(std)
        + np.log(2.0 * np.pi)
    )
    lower_mass = torch.special.log_ndtr(standard_lower)
    upper_mass = torch.special.log_ndtr(standard_upper_survival)
    per_axis = torch.where(
        action <= -1.0,
        lower_mass,
        torch.where(action >= 1.0, upper_mass, interior),
    )
    return per_axis.sum(dim=-1)


def _normal_entropy(std: torch.Tensor) -> torch.Tensor:
    return (0.5 + 0.5 * np.log(2.0 * np.pi) + torch.log(std)).sum(dim=-1)


def _bernoulli_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return -nn.functional.binary_cross_entropy_with_logits(logits, actions, reduction="none").sum(dim=-1)


def _bernoulli_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (
        -probs * torch.log(probs.clamp_min(1e-8))
        - (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-8))
    ).sum(dim=-1)


def _sample_policy(
    model: StatelessHybridActorCritic,
    obs: torch.Tensor,
    *,
    continuous_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(obs)
    std = torch.full_like(output.action_cont_mean, continuous_std)
    raw_action_cont = output.action_cont_mean + std * torch.randn_like(output.action_cont_mean)
    env_action_cont = raw_action_cont.clamp(-1.0, 1.0)
    action_bin_prob = torch.sigmoid(output.action_bin_logits)
    action_bin = torch.bernoulli(action_bin_prob)
    log_prob = _clipped_normal_log_prob(env_action_cont, output.action_cont_mean, std) + _bernoulli_log_prob(
        output.action_bin_logits, action_bin
    )
    return env_action_cont, env_action_cont, action_bin, action_bin_prob, log_prob, output.value


def _evaluate_policy(
    model: StatelessHybridActorCritic,
    obs: torch.Tensor,
    action_cont: torch.Tensor,
    action_bin: torch.Tensor,
    *,
    continuous_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(obs)
    std = torch.full_like(output.action_cont_mean, continuous_std)
    log_prob = _clipped_normal_log_prob(action_cont, output.action_cont_mean, std) + _bernoulli_log_prob(
        output.action_bin_logits, action_bin
    )
    continuous_entropy = _normal_entropy(std)
    binary_entropy = _bernoulli_entropy(output.action_bin_logits)
    return log_prob, continuous_entropy, binary_entropy, output.value


@torch.no_grad()
def _assert_rollout_log_prob_consistency(
    *,
    model: StatelessHybridActorCritic,
    observations: torch.Tensor,
    action_cont: torch.Tensor,
    action_bin: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    continuous_std: float,
    absolute_tolerance: float = 2e-4,
    relative_tolerance: float = 2e-5,
) -> float:
    recomputed_log_probs, _, _, _ = _evaluate_policy(
        model,
        observations,
        action_cont,
        action_bin,
        continuous_std=continuous_std,
    )
    error = (recomputed_log_probs - sampled_log_probs).abs()
    max_error = float(error.max().item()) if error.numel() else 0.0
    log_prob_scale = (
        max(1.0, float(sampled_log_probs.abs().max().item()))
        if sampled_log_probs.numel()
        else 1.0
    )
    allowed_error = absolute_tolerance + relative_tolerance * log_prob_scale
    if not np.isfinite(max_error) or max_error > allowed_error:
        raise RuntimeError(
            "rollout log-probability mismatch before PPO update: "
            f"max_abs_error={max_error:.6g}, allowed_error={allowed_error:.6g}"
        )
    return max_error


def _build_env(config: PolicyDogfightEnvConfig, *, obs_normalizer: ObservationNormalizer) -> PolicyDogfightEnv:
    _ = obs_normalizer
    return PolicyDogfightEnv(config)


def _build_train_env_backend(
    *,
    base_config: PolicyDogfightEnvConfig,
    args: argparse.Namespace,
    obs_normalizer: ObservationNormalizer,
) -> list[PolicyDogfightEnv] | SubprocPolicyVecEnv:
    if args.env_backend == "serial":
        return [
            _build_env(
                PolicyDogfightEnvConfig(
                    project_root=base_config.project_root,
                    scene_name=base_config.scene_name,
                    scene_path=base_config.scene_path,
                    ego_role=base_config.ego_role,
                    ego_mode=base_config.ego_mode,
                    opponent_mode=base_config.opponent_mode,
                    seed=args.seed + index,
                    ticks_per_step=base_config.ticks_per_step,
                ),
                obs_normalizer=obs_normalizer,
            )
            for index in range(args.num_envs)
        ]
    if args.env_backend == "subproc":
        return SubprocPolicyVecEnv(
            PolicyDogfightEnvConfig(
                project_root=base_config.project_root,
                scene_name=base_config.scene_name,
                scene_path=base_config.scene_path,
                ego_role=base_config.ego_role,
                ego_mode=base_config.ego_mode,
                opponent_mode=base_config.opponent_mode,
                seed=base_config.seed,
                ticks_per_step=base_config.ticks_per_step,
            ),
            num_envs=args.num_envs,
            reward_mode=args.worker_reward_mode,
            request_timeout_seconds=args.subproc_request_timeout_seconds,
        )
    raise ValueError(f"unsupported env backend: {args.env_backend}")


def _reset_train_envs(
    env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv,
    requests: list[ResetRequest],
) -> list[ResetResult]:
    if isinstance(env_backend, SubprocPolicyVecEnv):
        return env_backend.reset_many(requests)
    results: list[ResetResult] = []
    for env, request in zip(env_backend, requests, strict=True):
        obs, info = env.reset(
            seed=request.seed,
            scene_name=request.scene_name,
            scene_path=request.scene_path,
            opponent_mode=request.opponent_mode,
        )
        results.append(
            ResetResult(
                obs=obs,
                info=info,
                state=env.latest_state() if request.include_state else None,
                opponent_obs=(
                    env.observation_for_role(env.enemy_role)
                    if request.self_play
                    else None
                ),
            )
        )
    return results


def _reset_train_env_at(
    env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv,
    *,
    index: int,
    request: ResetRequest,
) -> ResetResult:
    if isinstance(env_backend, SubprocPolicyVecEnv):
        return env_backend.reset_at(index, request)
    env = env_backend[index]
    obs, info = env.reset(
        seed=request.seed,
        scene_name=request.scene_name,
        scene_path=request.scene_path,
        opponent_mode=request.opponent_mode,
    )
    return ResetResult(
        obs=obs,
        info=info,
        state=env.latest_state() if request.include_state else None,
        opponent_obs=(
            env.observation_for_role(env.enemy_role)
            if request.self_play
            else None
        ),
    )


def _step_train_envs(
    env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv,
    requests: list[StepRequest],
) -> list[StepResult]:
    if isinstance(env_backend, SubprocPolicyVecEnv):
        return env_backend.step_many(requests)
    results: list[StepResult] = []
    for env, request in zip(env_backend, requests, strict=True):
        obs, reward, terminated, truncated, info = env.step_arrays(
            request.continuous,
            request.binary,
            binary_threshold=request.binary_threshold,
            opponent_continuous=request.opponent_continuous,
            opponent_binary=request.opponent_binary,
        )
        results.append(
            StepResult(
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                state=env.latest_state() if request.include_state else None,
                opponent_obs=(
                    env.observation_for_role(env.enemy_role)
                    if request.self_play
                    else None
                ),
            )
        )
    return results


def _close_train_env_backend(env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv) -> None:
    if isinstance(env_backend, SubprocPolicyVecEnv):
        env_backend.close()
        return
    for env in env_backend:
        env.shutdown()


def _sample_episode_opponent(
    prepared_opponent_pool: PreparedOpponentPool | None,
) -> SampledOpponent | None:
    if prepared_opponent_pool is None:
        return None
    return prepared_opponent_pool.sample_episode_opponent()


def _sampled_opponent_requires_state(sampled: SampledOpponent | None) -> bool:
    return sampled is not None and sampled.runtime_kind == "checkpoint"


def _episode_uses_self_play(
    *,
    configured_opponent_mode: str,
    sampled_opponent: SampledOpponent | None,
) -> bool:
    if sampled_opponent is not None:
        return sampled_opponent.runtime_kind == "self_play"
    return configured_opponent_mode == "self_play"


def _validate_worker_reward_support(
    *,
    worker_reward_mode: str,
    env_backend: str,
    sampled_opponent: SampledOpponent | None,
) -> None:
    if worker_reward_mode != "worker":
        return
    if env_backend != "subproc":
        raise ValueError("--worker-reward-mode worker requires --env-backend subproc")
    if sampled_opponent is not None and sampled_opponent.runtime_kind == "checkpoint":
        raise ValueError("worker reward mode does not yet support checkpoint opponents")


def _load_model(
    *,
    obs_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> StatelessHybridActorCritic:
    return StatelessHybridActorCritic(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        shared_extension_blocks=args.shared_extension_blocks,
        actor_extension_blocks=args.actor_extension_blocks,
        critic_extension_blocks=args.critic_extension_blocks,
    ).to(device)


def _resolve_policy_reference_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.policy_drift_interval == 0:
        return None
    if args.policy_drift_reference_checkpoint:
        return Path(args.policy_drift_reference_checkpoint).resolve()
    if args.init_checkpoint:
        return Path(args.init_checkpoint).resolve()
    if not args.resume:
        return None
    payload = torch.load(args.resume, map_location="cpu", weights_only=False)
    checkpoint_args = payload.get("args", {})
    if not isinstance(checkpoint_args, dict):
        return None
    reference_path = checkpoint_args.get("policy_drift_reference_checkpoint")
    if not reference_path:
        reference_path = checkpoint_args.get("init_checkpoint")
    return None if not reference_path else Path(str(reference_path)).resolve()


def _load_policy_reference_model(
    *,
    path: Path | None,
    obs_dim: int,
    device: torch.device,
    dataset_contract: PolicyDatasetContract,
) -> StatelessHybridActorCritic | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"policy drift reference checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_policy_checkpoint_payload(
        payload,
        dataset=dataset_contract,
        context="PPO policy drift reference checkpoint",
    )
    reference_model = StatelessHybridActorCritic(
        obs_dim=obs_dim,
        **model_architecture_kwargs(payload["model_hyperparameters"]),
    ).to(device)
    reference_model.load_state_dict(payload["model_state_dict"], strict=True)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return reference_model


@torch.no_grad()
def _policy_reference_diagnostics(
    *,
    model: StatelessHybridActorCritic,
    reference_model: StatelessHybridActorCritic,
    observations: torch.Tensor,
    sample_size: int,
) -> dict[str, Any]:
    if observations.shape[0] <= 0:
        return {"sample_count": 0}
    actual_size = min(int(sample_size), int(observations.shape[0]))
    sample_indices = torch.linspace(
        0,
        observations.shape[0] - 1,
        steps=actual_size,
        device=observations.device,
    ).to(torch.long)
    sampled_observations = observations[sample_indices]
    was_training = model.training
    model.eval()
    current = model(sampled_observations)
    reference = reference_model(sampled_observations)
    model.train(was_training)

    continuous_delta = current.action_cont_mean - reference.action_cont_mean
    current_probabilities = torch.sigmoid(current.action_bin_logits).clamp(1e-6, 1.0 - 1e-6)
    reference_probabilities = torch.sigmoid(reference.action_bin_logits).clamp(1e-6, 1.0 - 1e-6)
    bernoulli_kl = (
        reference_probabilities
        * (reference_probabilities.log() - current_probabilities.log())
        + (1.0 - reference_probabilities)
        * ((1.0 - reference_probabilities).log() - (1.0 - current_probabilities).log())
    )
    return {
        "sample_count": actual_size,
        "continuous_mean_abs_drift": float(continuous_delta.abs().mean().item()),
        "continuous_rmse": float(continuous_delta.square().mean().sqrt().item()),
        "continuous_mean_abs_drift_by_action": [
            float(value) for value in continuous_delta.abs().mean(dim=0).cpu().tolist()
        ],
        "binary_probability_mean_abs_drift": float(
            (current_probabilities - reference_probabilities).abs().mean().item()
        ),
        "binary_reference_to_current_kl": float(bernoulli_kl.sum(dim=-1).mean().item()),
        "current_binary_probability_mean": [
            float(value) for value in current_probabilities.mean(dim=0).cpu().tolist()
        ],
        "reference_binary_probability_mean": [
            float(value) for value in reference_probabilities.mean(dim=0).cpu().tolist()
        ],
    }


def _build_optimizer(
    model: StatelessHybridActorCritic,
    *,
    learning_rate: float,
    weight_decay: float,
    shared_learning_rate: float | None,
    actor_learning_rate: float | None,
    critic_learning_rate: float | None,
) -> torch.optim.Optimizer:
    if shared_learning_rate is None and actor_learning_rate is None and critic_learning_rate is None:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    shared_lr = learning_rate if shared_learning_rate is None else shared_learning_rate
    actor_lr = learning_rate if actor_learning_rate is None else actor_learning_rate
    critic_lr = learning_rate if critic_learning_rate is None else critic_learning_rate

    param_groups = [
        {
            "params": list(model.shared_stem.parameters())
            + list(model.shared_extension_tower.parameters()),
            "lr": shared_lr,
            "weight_decay": weight_decay,
            "group_name": "shared",
        },
        {
            "params": list(model.actor_tower.parameters())
            + list(model.actor_extension_tower.parameters())
            + list(model.action_cont_head.parameters())
            + list(model.action_bin_head.parameters()),
            "lr": actor_lr,
            "weight_decay": weight_decay,
            "group_name": "actor",
        },
        {
            "params": list(model.critic_tower.parameters())
            + list(model.critic_extension_tower.parameters())
            + list(model.value_head.parameters()),
            "lr": critic_lr,
            "weight_decay": weight_decay,
            "group_name": "critic",
        },
    ]
    return torch.optim.AdamW(param_groups)


def _apply_optimizer_hparams(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
    shared_learning_rate: float | None,
    actor_learning_rate: float | None,
    critic_learning_rate: float | None,
) -> None:
    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.param_groups[0]["weight_decay"] = weight_decay
        return

    shared_lr = learning_rate if shared_learning_rate is None else shared_learning_rate
    actor_lr = learning_rate if actor_learning_rate is None else actor_learning_rate
    critic_lr = learning_rate if critic_learning_rate is None else critic_learning_rate
    lrs_by_group_name = {
        "shared": shared_lr,
        "actor": actor_lr,
        "critic": critic_lr,
    }
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        if group_name not in lrs_by_group_name:
            raise ValueError(f"optimizer param group is missing supported group_name, got {group_name!r}")
        group["lr"] = lrs_by_group_name[group_name]
        group["weight_decay"] = weight_decay


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float, float]:
    if len(optimizer.param_groups) == 1:
        lr = float(optimizer.param_groups[0]["lr"])
        return lr, lr, lr
    groups = {str(group.get("group_name")): group for group in optimizer.param_groups}
    return (
        float(groups["shared"]["lr"]),
        float(groups["actor"]["lr"]),
        float(groups["critic"]["lr"]),
    )


def _compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    terminated: np.ndarray,
    truncated_bootstrap_values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Compute GAE without treating time-limit truncations as true terminals."""
    expected_shape = rewards.shape
    for name, array in (
        ("values", values),
        ("dones", dones),
        ("terminated", terminated),
        ("truncated_bootstrap_values", truncated_bootstrap_values),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} does not match rewards shape {expected_shape}")
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [steps, envs], got {rewards.shape}")
    if next_values.shape != (rewards.shape[1],):
        raise ValueError(
            f"next_values shape {next_values.shape} does not match env count {rewards.shape[1]}"
        )
    if np.any(terminated > dones):
        raise ValueError("terminated transitions must also be marked done")

    advantages = np.zeros_like(rewards)
    last_gae = np.zeros((rewards.shape[1],), dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        chronological_next_value = (
            next_values if step == rewards.shape[0] - 1 else values[step + 1]
        )
        truncated_mask = np.clip(dones[step] - terminated[step], 0.0, 1.0)
        bootstrap_value = np.where(
            truncated_mask > 0.5,
            truncated_bootstrap_values[step],
            chronological_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminated[step]
        episode_continues = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * bootstrap_value * bootstrap_nonterminal
            - values[step]
        )
        last_gae = delta + gamma * gae_lambda * episode_continues * last_gae
        advantages[step] = last_gae
    return advantages


def _apply_checkpoint_arch_args(
    args: argparse.Namespace,
    *,
    dataset_contract: PolicyDatasetContract,
) -> None:
    """Override --hidden-dim and --num-layers from --resume checkpoint if present."""
    checkpoint_path = args.resume or args.init_checkpoint
    if checkpoint_path is None:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_policy_checkpoint_payload(
        payload,
        dataset=dataset_contract,
        context="PPO architecture checkpoint",
    )
    architecture = model_architecture_kwargs(payload["model_hyperparameters"])
    for key, value in architecture.items():
        setattr(args, key, value)


def _load_warm_start(
    model: StatelessHybridActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    resume: str | None,
    init_checkpoint: str | None,
    reset_critic_on_init: bool,
    device: torch.device,
    dataset_contract: PolicyDatasetContract,
) -> tuple[int, int, float]:
    start_update = 0
    global_step = 0
    best_eval_return = float("-inf")
    if reset_critic_on_init and resume:
        raise ValueError("--reset-critic-on-init can only be used with --init-checkpoint, not --resume")
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        validate_policy_checkpoint_payload(
            payload,
            dataset=dataset_contract,
            context="PPO resume checkpoint",
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_update = int(payload["update"]) + 1
        global_step = int(payload["global_step"])
        best_eval_return = float(payload.get("best_eval_return", float("-inf")))
    elif init_checkpoint:
        payload = torch.load(init_checkpoint, map_location=device, weights_only=False)
        validate_policy_checkpoint_payload(
            payload,
            dataset=dataset_contract,
            context="PPO init checkpoint",
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if reset_critic_on_init:
            model.value_head.reset_parameters()
    return start_update, global_step, best_eval_return


def _load_best_eval_return_from_dir(
    checkpoints_dir: Path,
    *,
    dataset_contract: PolicyDatasetContract,
) -> float:
    best_path = checkpoints_dir / "best.pt"
    if not best_path.exists():
        return float("-inf")
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_policy_checkpoint_payload(
        payload,
        dataset=dataset_contract,
        context="PPO best checkpoint",
    )
    return float(payload.get("best_eval_return", float("-inf")))


def _save_checkpoint(
    *,
    path: Path,
    model: StatelessHybridActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    update: int,
    global_step: int,
    best_eval_return: float,
    metrics: PpoTrainMetrics,
    dataset_contract: PolicyDatasetContract,
    training_semantics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(
            {
                **checkpoint_contract_metadata(dataset_contract),
                "model_hyperparameters": checkpoint_model_hyperparameters(
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    shared_extension_blocks=args.shared_extension_blocks,
                    actor_extension_blocks=args.actor_extension_blocks,
                    critic_extension_blocks=args.critic_extension_blocks,
                    continuous_action_std=args.continuous_action_std,
                    popart_beta=args.popart_beta,
                    popart_min_std=args.popart_min_std,
                ),
                "training_stage": "ppo",
                "update_index": update,
                "update": update,
                "global_step": global_step,
                "best_eval_return": best_eval_return,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": asdict(metrics),
                "ppo_training_semantics": training_semantics,
            },
            temporary_path,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _should_save_periodic_checkpoint(*, update: int, target_update: int, checkpoint_interval: int) -> bool:
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be >= 1")
    return ((update + 1) % checkpoint_interval == 0) or (update == target_update - 1)


def _empty_episode_stats(num_envs: int) -> list[dict[str, float | int]]:
    return [
        {
            "return": 0.0,
            "length": 0,
            "start_sim_time_seconds": 0.0,
            "max_out_of_bounds_seconds": 0.0,
            "self_hit_count": 0,
            "enemy_hit_count": 0,
        }
        for _ in range(num_envs)
    ]


def _finalize_episode_record(
    stat: dict[str, float | int],
    info: dict[str, Any],
    *,
    terminated: bool,
    truncated: bool,
    truncation_reason: str | None = None,
    reward_diagnostics: dict[str, Any] | None = None,
    opponent_label: str = "unknown",
    opponent_kind: str = "unknown",
    scene_label: str = "unknown",
) -> EpisodeRecord:
    self_state = info["aircraft_by_role"][info["ego_role"]]
    enemy_state = info["aircraft_by_role"][info["enemy_role"]]
    current_sim_time_seconds = float(info.get("sim_time_seconds", 0.0))
    start_sim_time_seconds = float(stat.get("start_sim_time_seconds", 0.0))
    self_destroyed = bool(self_state["destroyed"])
    enemy_destroyed = bool(enemy_state["destroyed"])
    if self_destroyed and enemy_destroyed:
        termination_reason = "mutual_destroyed"
    elif self_destroyed and float(self_state.get("out_of_bounds_seconds", 0.0)) > 0.0:
        termination_reason = "self_out_of_bounds_destroyed"
    elif self_destroyed:
        termination_reason = "self_destroyed"
    elif enemy_destroyed:
        termination_reason = "enemy_destroyed"
    elif truncated:
        termination_reason = truncation_reason or "environment_truncated"
    elif terminated:
        termination_reason = "unknown_terminated"
    else:
        termination_reason = "incomplete"
    return EpisodeRecord(
        episode_return=float(stat["return"]),
        episode_length=int(stat["length"]),
        episode_survival_seconds=max(0.0, current_sim_time_seconds - start_sim_time_seconds),
        self_destroyed=self_destroyed,
        enemy_destroyed=enemy_destroyed,
        terminated=terminated,
        truncated=truncated,
        max_out_of_bounds_seconds=float(stat["max_out_of_bounds_seconds"]),
        self_hit_count=int(stat.get("self_hit_count", 0)),
        enemy_hit_count=int(stat.get("enemy_hit_count", 0)),
        termination_reason=termination_reason,
        reward_diagnostics=reward_diagnostics,
        opponent_label=opponent_label,
        opponent_kind=opponent_kind,
        scene_label=scene_label,
    )


def _episode_sampling_diagnostics(records: list[EpisodeRecord]) -> dict[str, Any]:
    opponent_kind_counts: dict[str, int] = {}
    opponent_label_counts: dict[str, int] = {}
    scene_label_counts: dict[str, int] = {}
    for record in records:
        opponent_kind_counts[record.opponent_kind] = (
            opponent_kind_counts.get(record.opponent_kind, 0) + 1
        )
        opponent_label_counts[record.opponent_label] = (
            opponent_label_counts.get(record.opponent_label, 0) + 1
        )
        scene_label_counts[record.scene_label] = (
            scene_label_counts.get(record.scene_label, 0) + 1
        )
    episode_count = len(records)
    return {
        "episode_count": episode_count,
        "self_play_episode_fraction": (
            float(opponent_kind_counts.get("self_play", 0)) / float(episode_count)
            if episode_count > 0
            else 0.0
        ),
        "opponent_kind_counts": opponent_kind_counts,
        "opponent_label_counts": opponent_label_counts,
        "scene_label_counts": scene_label_counts,
    }


def _aggregate_episode_records(records: list[EpisodeRecord]) -> tuple[float, float, float]:
    if not records:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean([item.episode_return for item in records])),
        float(np.mean([item.episode_length for item in records])),
        float(np.mean([item.episode_survival_seconds for item in records])),
    )


def _episode_rate(records: list[EpisodeRecord], predicate: Callable[[EpisodeRecord], bool]) -> float:
    if not records:
        return 0.0
    return float(np.mean([1.0 if predicate(item) else 0.0 for item in records]))


def _mean_max_out_of_bounds_seconds(records: list[EpisodeRecord]) -> float:
    if not records:
        return 0.0
    return float(np.mean([item.max_out_of_bounds_seconds for item in records]))


def _termination_reason_counts(records: list[EpisodeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.termination_reason] = counts.get(record.termination_reason, 0) + 1
    return counts


def _reward_diagnostics_by_outcome(
    records: list[EpisodeRecord],
) -> dict[str, Any]:
    grouped_records: dict[str, list[EpisodeRecord]] = {"all": records}
    for record in records:
        grouped_records.setdefault(record.termination_reason, []).append(record)

    result: dict[str, Any] = {}
    for outcome, outcome_records in grouped_records.items():
        episode_count = len(outcome_records)
        transition_count = 0
        sums = {name: 0.0 for name in REWARD_COMPONENT_FIELDS}
        absolute_sums = {name: 0.0 for name in REWARD_COMPONENT_FIELDS}
        for record in outcome_records:
            payload = record.reward_diagnostics
            if payload is None:
                continue
            transition_count += int(payload.get("sample_count", 0))
            component_sums = payload.get("sums", {})
            component_absolute_sums = payload.get("absolute_sums", {})
            if not isinstance(component_sums, dict) or not isinstance(
                component_absolute_sums,
                dict,
            ):
                raise TypeError("episode reward diagnostics components must be mappings")
            for name in REWARD_COMPONENT_FIELDS:
                sums[name] += float(component_sums.get(name, 0.0))
                absolute_sums[name] += float(component_absolute_sums.get(name, 0.0))

        episode_denominator = max(episode_count, 1)
        transition_denominator = max(transition_count, 1)
        result[outcome] = {
            "episode_count": episode_count,
            "transition_count": transition_count,
            "components": {
                name: {
                    "mean_episode_sum": sums[name] / episode_denominator,
                    "mean_episode_abs_sum": (
                        absolute_sums[name] / episode_denominator
                    ),
                    "mean_step": sums[name] / transition_denominator,
                }
                for name in REWARD_COMPONENT_FIELDS
            },
        }
    return result


def _aggregate_episode_metrics(records: list[EpisodeRecord]) -> EpisodeAggregateMetrics:
    mean_return, mean_length, mean_duration_seconds = _aggregate_episode_records(records)
    return EpisodeAggregateMetrics(
        episode_count=len(records),
        mean_return=mean_return,
        mean_length=mean_length,
        mean_duration_seconds=mean_duration_seconds,
        self_destroy_rate=_episode_rate(records, lambda item: item.self_destroyed),
        enemy_destroy_rate=_episode_rate(records, lambda item: item.enemy_destroyed),
        mutual_destroy_rate=_episode_rate(
            records,
            lambda item: item.self_destroyed and item.enemy_destroyed,
        ),
        out_of_bounds_destroy_rate=_episode_rate(
            records,
            lambda item: item.termination_reason == "self_out_of_bounds_destroyed",
        ),
        truncated_rate=_episode_rate(records, lambda item: item.truncated),
        mean_max_out_of_bounds_seconds=_mean_max_out_of_bounds_seconds(records),
        self_mean_hit_count=(
            float(np.mean([item.self_hit_count for item in records])) if records else 0.0
        ),
        enemy_mean_hit_count=(
            float(np.mean([item.enemy_hit_count for item in records])) if records else 0.0
        ),
        termination_reason_counts=_termination_reason_counts(records),
    )


def _drain_reward_diagnostics(
    *,
    env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv,
    worker_reward_mode: str,
    reward_runtimes: list[WorkerRewardStateMachine | None],
) -> dict[str, Any]:
    accumulator = RewardDiagnosticsAccumulator()
    if worker_reward_mode == "worker":
        if not isinstance(env_backend, SubprocPolicyVecEnv):
            raise RuntimeError("worker reward diagnostics require the subproc backend")
        for payload in env_backend.drain_reward_diagnostics():
            accumulator.merge_payload(payload)
    for runtime in reward_runtimes:
        if runtime is not None:
            accumulator.merge_payload(runtime.drain_reward_diagnostics())
    return accumulator.summary()


def _drain_episode_reward_diagnostics(
    *,
    env_backend: list[PolicyDogfightEnv] | SubprocPolicyVecEnv,
    worker_reward_mode: str,
    env_index: int,
    reward_runtime: WorkerRewardStateMachine | None,
) -> dict[str, Any]:
    if worker_reward_mode == "worker":
        if not isinstance(env_backend, SubprocPolicyVecEnv):
            raise RuntimeError("worker episode reward diagnostics require subproc backend")
        return env_backend.drain_episode_reward_diagnostics_at(env_index)
    if reward_runtime is None:
        raise RuntimeError("main-process episode reward runtime is not initialized")
    return reward_runtime.drain_episode_reward_diagnostics()


def _evaluation_limit_reason(
    *,
    episode_steps: int,
    elapsed_sim_seconds: float,
    max_steps: int,
    max_seconds: float,
) -> str | None:
    if max_steps > 0 and episode_steps >= max_steps:
        return "eval_step_limit"
    if max_seconds > 0.0 and elapsed_sim_seconds >= max_seconds:
        return "eval_time_limit"
    return None


def _is_new_best_eval(
    eval_metrics: PpoEvalMetrics | None,
    *,
    best_eval_return: float,
) -> bool:
    return eval_metrics is not None and eval_metrics.mean_return > best_eval_return


def _target_kl_reached(*, approximate_kl: float, target_kl: float) -> bool:
    return target_kl > 0.0 and approximate_kl >= target_kl


@torch.no_grad()
def _run_eval(
    *,
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    reward_composer: PolicyRewardComposer,
    config: PolicyDogfightEnvConfig,
    eval_scenes: list[MaterializedScene] | None,
    prepared_opponent_pool: PreparedOpponentPool | None,
    episodes: int,
    max_steps: int,
    max_seconds: float,
    device: torch.device,
) -> PpoEvalMetrics:
    env = _build_env(config, obs_normalizer=normalizer)
    opponent_provider = OpponentActionProvider(device=device)
    records: list[EpisodeRecord] = []
    try:
        for episode_index in range(episodes):
            eval_scene = None if not eval_scenes else eval_scenes[episode_index % len(eval_scenes)]
            scene_name = config.scene_name if eval_scene is None else eval_scene.scene_name
            scene_path = config.scene_path if eval_scene is None else eval_scene.scene_path
            episode_opponent = _sample_episode_opponent(prepared_opponent_pool)
            obs_raw, info = env.reset(
                seed=(config.seed or 0) + 10_000 + episode_index,
                scene_name=scene_name,
                scene_path=scene_path,
                opponent_mode=config.opponent_mode if episode_opponent is None else episode_opponent.env_mode,
            )
            reward_runtime = _new_reward_runtime(
                reward_composer,
                initial_info=info,
            )
            stat = {
                "return": 0.0,
                "length": 0,
                "start_sim_time_seconds": float(info.get("sim_time_seconds", 0.0)),
                "max_out_of_bounds_seconds": 0.0,
                "self_hit_count": 0,
                "enemy_hit_count": 0,
            }
            terminated = False
            truncated = False
            truncation_reason: str | None = None
            while not (terminated or truncated):
                policy_output = deterministic_policy_output(
                    model=model,
                    normalizer=normalizer,
                    obs=obs_raw,
                    device=device,
                )
                action_cont = policy_output.action_cont
                action_bin = policy_output.binary_actions()
                opponent_action = None
                if (
                    episode_opponent is not None
                    and episode_opponent.runtime_kind == "self_play"
                ):
                    enemy_obs_raw = env.observation_for_role(env.enemy_role)
                    enemy_output = deterministic_policy_output(
                        model=model,
                        normalizer=normalizer,
                        obs=enemy_obs_raw,
                        device=device,
                    )
                    opponent_action = (
                        enemy_output.action_cont,
                        enemy_output.binary_actions(),
                    )
                elif (
                    episode_opponent is not None
                    and episode_opponent.needs_external_action()
                ):
                    opponent_action = opponent_provider.action_arrays_for_state(
                        episode_opponent,
                        state=env.latest_state(),
                        role=env.enemy_role,
                        episode_start_sim_time_seconds=env.episode_start_sim_time_seconds,
                    )
                next_obs_raw, _, terminated, truncated, info = env.step_arrays(
                    action_cont,
                    action_bin,
                    opponent_continuous=None if opponent_action is None else opponent_action[0],
                    opponent_binary=None if opponent_action is None else opponent_action[1],
                )
                reward = reward_runtime.compute(
                    current_info=info,
                    current_obs=next_obs_raw,
                    action_cont=action_cont,
                    action_bin=action_bin,
                )
                stat["return"] = float(stat["return"]) + reward.total
                stat["length"] = int(stat["length"]) + 1
                stat["max_out_of_bounds_seconds"] = max(
                    float(stat["max_out_of_bounds_seconds"]),
                    float(info["aircraft_by_role"][info["ego_role"]]["out_of_bounds_seconds"]),
                )
                s_hits, e_hits = _count_hits_from_events(
                    info.get("events_since_last_step", []),
                    self_role=info["ego_role"],
                    enemy_role=info["enemy_role"],
                )
                stat["self_hit_count"] = int(stat["self_hit_count"]) + s_hits
                stat["enemy_hit_count"] = int(stat["enemy_hit_count"]) + e_hits
                obs_raw = next_obs_raw
                reward_runtime.advance(
                    info=info,
                    action_cont=action_cont,
                    action_bin=action_bin,
                    terminated_or_truncated=bool(terminated or truncated),
                )
                elapsed_sim_seconds = max(
                    0.0,
                    float(info.get("sim_time_seconds", 0.0))
                    - float(stat["start_sim_time_seconds"]),
                )
                truncation_reason = _evaluation_limit_reason(
                    episode_steps=int(stat["length"]),
                    elapsed_sim_seconds=elapsed_sim_seconds,
                    max_steps=max_steps,
                    max_seconds=max_seconds,
                )
                if truncation_reason is not None and not terminated:
                    truncated = True
            records.append(
                _finalize_episode_record(
                    stat,
                    info,
                    terminated=terminated,
                    truncated=truncated,
                    truncation_reason=truncation_reason,
                    opponent_label=(
                        episode_opponent.label
                        if episode_opponent is not None
                        else config.opponent_mode
                    ),
                    opponent_kind=(
                        episode_opponent.runtime_kind
                        if episode_opponent is not None
                        else config.opponent_mode
                    ),
                    scene_label=(
                        eval_scene.label
                        if eval_scene is not None
                        else str(scene_name)
                    ),
                )
            )
    finally:
        env.shutdown()
    if not records:
        return PpoEvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    return PpoEvalMetrics(
        mean_return=float(np.mean([item.episode_return for item in records])),
        mean_length=float(np.mean([item.episode_length for item in records])),
        mean_survival_seconds=float(np.mean([item.episode_survival_seconds for item in records])),
        self_destroy_rate=float(np.mean([float(item.self_destroyed) for item in records])),
        enemy_destroy_rate=float(np.mean([float(item.enemy_destroyed) for item in records])),
        mean_max_out_of_bounds_seconds=float(np.mean([item.max_out_of_bounds_seconds for item in records])),
        episode_count=len(records),
        self_mean_hit_count=float(np.mean([item.self_hit_count for item in records])),
        enemy_mean_hit_count=float(np.mean([item.enemy_hit_count for item in records])),
        mutual_destroy_rate=float(
            np.mean([float(item.self_destroyed and item.enemy_destroyed) for item in records])
        ),
        truncated_rate=float(np.mean([float(item.truncated) for item in records])),
        timeout_rate=float(
            np.mean(
                [
                    float(item.termination_reason in {"eval_step_limit", "eval_time_limit"})
                    for item in records
                ]
            )
        ),
        termination_reason_counts=_termination_reason_counts(records),
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _validate_output_dir_usage(args, output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "eval"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(args.dataset_root)
    reward_config = PolicyRewardConfig()
    base_scene_asset = _materialize_base_scene_asset(args, output_dir)
    prepared_scene_pool = _prepare_scene_pool(args, output_dir)
    scene_pool_manifest_path = (
        None
        if prepared_scene_pool is None
        else output_dir / "scene_pool" / "manifests" / "scene_pool_manifest.json"
    )
    training_semantics = _build_training_semantics(
        args=args,
        dataset_contract=dataset_contract,
        reward_config=reward_config,
        base_scene_asset=base_scene_asset,
        scene_pool_manifest_path=scene_pool_manifest_path,
    )
    if args.resume:
        _validate_resume_training_semantics(
            checkpoint_path=Path(args.resume).expanduser().resolve(),
            expected=training_semantics,
        )
    prepared_opponent_pool = _prepare_opponent_pool(args)

    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    _apply_checkpoint_arch_args(args, dataset_contract=dataset_contract)
    model = _load_model(obs_dim=normalizer.obs_dim, device=device, args=args)
    policy_reference_path = _resolve_policy_reference_checkpoint(args)
    policy_reference_model = _load_policy_reference_model(
        path=policy_reference_path,
        obs_dim=normalizer.obs_dim,
        device=device,
        dataset_contract=dataset_contract,
    )
    optimizer = _build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        shared_learning_rate=args.shared_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
    )
    start_update, global_step, best_eval_return = _load_warm_start(
        model,
        optimizer,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        reset_critic_on_init=args.reset_critic_on_init,
        device=device,
        dataset_contract=dataset_contract,
    )
    if args.resume:
        _apply_optimizer_hparams(
            optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            shared_learning_rate=args.shared_learning_rate,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
        )
    if args.resume:
        best_eval_return = _load_best_eval_return_from_dir(
            checkpoints_dir,
            dataset_contract=dataset_contract,
        )
    target_update = start_update + args.updates if args.resume else args.updates

    reward_composer = PolicyRewardComposer(reward_config)
    truncation_policy = build_truncation_policy(args, reward_composer=reward_composer)
    base_env_config = PolicyDogfightEnvConfig(
        project_root=args.project_root,
        scene_name=base_scene_asset.scene_name,
        scene_path=base_scene_asset.scene_path,
        ego_role=args.ego_role,
        opponent_mode="model" if args.opponent_mode == "self_play" else args.opponent_mode,
        seed=args.seed,
        ticks_per_step=args.ticks_per_step,
    )
    env_backend = _build_train_env_backend(
        base_config=base_env_config,
        args=args,
        obs_normalizer=normalizer,
    )
    try:
        opponent_provider = OpponentActionProvider(device=device)
        reset_requests: list[ResetRequest] = []
        episode_opponents: list[SampledOpponent | None] = []
        episode_scene_labels: list[str] = []
        for env_index in range(args.num_envs):
            initial_scene = None if prepared_scene_pool is None else prepared_scene_pool.sample_train_scene()
            scene_name, scene_path = _resolve_scene_override(initial_scene, base_scene_asset)
            initial_opponent = _sample_episode_opponent(prepared_opponent_pool)
            _validate_worker_reward_support(
                worker_reward_mode=args.worker_reward_mode,
                env_backend=args.env_backend,
                sampled_opponent=initial_opponent,
            )
            initial_is_self_play = _episode_uses_self_play(
                configured_opponent_mode=args.opponent_mode,
                sampled_opponent=initial_opponent,
            )
            reset_requests.append(
                ResetRequest(
                    seed=args.seed + env_index,
                    scene_name=scene_name,
                    scene_path=scene_path,
                    opponent_mode=_opponent_mode_for_env(args, initial_opponent),
                    include_state=_sampled_opponent_requires_state(initial_opponent),
                    self_play=initial_is_self_play,
                )
            )
            episode_opponents.append(initial_opponent)
            episode_scene_labels.append(
                initial_scene.label
                if initial_scene is not None
                else base_scene_asset.scene_name
            )
        reset_results = _reset_train_envs(env_backend, reset_requests)
        obs_raw = np.stack([result.obs for result in reset_results], axis=0).astype(np.float32, copy=False)
        infos = [result.info for result in reset_results]
        states_by_env = [result.state for result in reset_results]
        opponent_obs_raw_by_env = [result.opponent_obs for result in reset_results]
        episode_start_sim_times_by_env = [
            float(info["episode_start_sim_time_seconds"]) for info in infos
        ]
        obs = np.stack([_normalize_obs(normalizer, item) for item in obs_raw], axis=0).astype(np.float32, copy=False)
        episode_stats = _empty_episode_stats(args.num_envs)
        for env_index, info in enumerate(infos):
            episode_stats[env_index]["start_sim_time_seconds"] = float(info.get("sim_time_seconds", 0.0))
        truncation_runtime_by_env = [truncation_policy.initial_runtime(info) for info in infos]
        recent_records: list[EpisodeRecord] = []
        rolling_records: deque[EpisodeRecord] = deque(maxlen=args.episode_metrics_window)
        self_play = args.opponent_mode == "self_play" and prepared_opponent_pool is None
        dual_policy_slots = self_play or bool(
            prepared_opponent_pool is not None
            and prepared_opponent_pool.has_self_play
        )
        effective_envs = args.num_envs * 2 if dual_policy_slots else args.num_envs
        reward_runtime_by_slot: list[WorkerRewardStateMachine | None] = [
            None for _ in range(effective_envs)
        ]
        for env_index, info in enumerate(infos):
            ego_slot = env_index * 2 if dual_policy_slots else env_index
            if args.worker_reward_mode != "worker":
                reward_runtime_by_slot[ego_slot] = _new_reward_runtime(
                    reward_composer,
                    initial_info=_info_for_role(info, args.ego_role),
                )
            if _episode_uses_self_play(
                configured_opponent_mode=args.opponent_mode,
                sampled_opponent=episode_opponents[env_index],
            ) and args.worker_reward_mode != "worker":
                reward_runtime_by_slot[ego_slot + 1] = _new_reward_runtime(
                    reward_composer,
                    initial_info=_info_for_role(info, _opposing_role(args.ego_role)),
                )

        summary_path = output_dir / "train_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dataset_root": str(Path(args.dataset_root).resolve()),
                    "obs_dim": normalizer.obs_dim,
                    **checkpoint_contract_metadata(dataset_contract),
                    "scene_pool_json": str(Path(args.scene_pool_json).resolve()) if args.scene_pool_json else None,
                    "opponent_pool_json": str(Path(args.opponent_pool_json).resolve()) if args.opponent_pool_json else None,
                    "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
                    "policy_drift_reference_checkpoint": (
                        str(policy_reference_path) if policy_reference_path is not None else None
                    ),
                    "start_update": start_update,
                    "target_update_exclusive": target_update,
                    "ppo_training_semantics": training_semantics,
                    "args": vars(args),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        metrics_path = output_dir / "metrics.jsonl"
        for update in range(start_update, target_update):
            update_start_time = time.perf_counter()
            buffer_obs = np.zeros((args.rollout_steps, effective_envs, normalizer.obs_dim), dtype=np.float32)
            buffer_action_cont_policy = np.zeros((args.rollout_steps, effective_envs, 4), dtype=np.float32)
            buffer_action_cont_env = np.zeros((args.rollout_steps, effective_envs, 4), dtype=np.float32)
            buffer_action_bin = np.zeros((args.rollout_steps, effective_envs, 3), dtype=np.float32)
            buffer_action_bin_prob = np.zeros(
                (args.rollout_steps, effective_envs, 3),
                dtype=np.float32,
            )
            buffer_rewards = np.zeros((args.rollout_steps, effective_envs), dtype=np.float32)
            buffer_dones = np.zeros((args.rollout_steps, effective_envs), dtype=np.float32)
            buffer_terminated = np.zeros((args.rollout_steps, effective_envs), dtype=np.float32)
            buffer_truncated_bootstrap_values = np.zeros(
                (args.rollout_steps, effective_envs),
                dtype=np.float32,
            )
            buffer_values = np.zeros((args.rollout_steps, effective_envs), dtype=np.float32)
            buffer_log_probs = np.zeros((args.rollout_steps, effective_envs), dtype=np.float32)
            buffer_valid = np.zeros((args.rollout_steps, effective_envs), dtype=np.bool_)
            buffer_self_play_episode = np.zeros(
                (args.rollout_steps, effective_envs),
                dtype=np.bool_,
            )
            truncated_bootstrap_observations: list[tuple[int, int, np.ndarray]] = []

            reward_totals: list[float] = []
            policy_forward_seconds = 0.0
            opponent_action_seconds = 0.0
            env_step_seconds = 0.0
            reward_compute_seconds = 0.0
            diagnostics_seconds = 0.0
            env_reset_seconds = 0.0
            rollout_collection_start_time = time.perf_counter()
            for step in range(args.rollout_steps):
                step_requests: list[StepRequest] = []
                next_obs_raw_list: list[np.ndarray] = []
                next_state_list: list[dict[str, Any] | None] = []
                next_opponent_obs_raw_list: list[np.ndarray | None] = []
                done_mask = np.zeros((effective_envs,), dtype=np.float32)
                terminated_mask = np.zeros((effective_envs,), dtype=np.float32)
                self_play_env_indices = [
                    env_index
                    for env_index, sampled_opponent in enumerate(episode_opponents)
                    if _episode_uses_self_play(
                        configured_opponent_mode=args.opponent_mode,
                        sampled_opponent=sampled_opponent,
                    )
                ]
                policy_obs, policy_slots = _build_rollout_policy_observation_batch(
                    ego_obs_normalized=obs,
                    self_play_env_indices=self_play_env_indices,
                    dual_policy_slots=dual_policy_slots,
                    opponent_obs_raw_by_env=opponent_obs_raw_by_env,
                    normalizer=normalizer,
                )
                policy_forward_start_time = time.perf_counter()
                with torch.no_grad():
                    policy_outputs = _sample_policy(
                        model,
                        torch.from_numpy(policy_obs).to(device),
                        continuous_std=args.continuous_action_std,
                    )
                policy_forward_seconds += time.perf_counter() - policy_forward_start_time
                policy_arrays = [
                    item.cpu().numpy().astype(np.float32, copy=False)
                    for item in policy_outputs
                ]
                (
                    policy_action_cont,
                    env_action_cont,
                    action_bin,
                    action_bin_prob,
                    log_prob,
                    values,
                ) = policy_arrays
                buffer_obs[step, policy_slots] = policy_obs
                buffer_action_cont_policy[step, policy_slots] = policy_action_cont
                buffer_action_cont_env[step, policy_slots] = env_action_cont
                buffer_action_bin[step, policy_slots] = action_bin
                buffer_action_bin_prob[step, policy_slots] = action_bin_prob
                buffer_values[step, policy_slots] = values
                buffer_log_probs[step, policy_slots] = log_prob
                buffer_valid[step, policy_slots] = True
                for env_index in self_play_env_indices:
                    ego_slot = env_index * 2
                    buffer_self_play_episode[step, ego_slot : ego_slot + 2] = True

                slot_to_policy_row = {
                    int(slot): row for row, slot in enumerate(policy_slots)
                }
                for env_index in range(args.num_envs):
                    ego_slot = env_index * 2 if dual_policy_slots else env_index
                    ego_row = slot_to_policy_row[ego_slot]
                    episode_opponent = episode_opponents[env_index]
                    episode_is_self_play = env_index in self_play_env_indices
                    opponent_action = None
                    if episode_is_self_play:
                        enemy_row = slot_to_policy_row[ego_slot + 1]
                        opponent_action = (
                            env_action_cont[enemy_row],
                            action_bin[enemy_row],
                        )
                    elif (
                        episode_opponent is not None
                        and episode_opponent.needs_external_action()
                    ):
                        opponent_action_start_time = time.perf_counter()
                        opponent_action = opponent_provider.action_arrays_for_state(
                            episode_opponent,
                            state=(
                                {}
                                if states_by_env[env_index] is None
                                else states_by_env[env_index]
                            ),
                            role=_opposing_role(args.ego_role),
                            episode_start_sim_time_seconds=(
                                episode_start_sim_times_by_env[env_index]
                            ),
                        )
                        opponent_action_seconds += (
                            time.perf_counter() - opponent_action_start_time
                        )
                    step_requests.append(
                        StepRequest(
                            continuous=env_action_cont[ego_row],
                            binary=action_bin[ego_row],
                            opponent_continuous=(
                                None if opponent_action is None else opponent_action[0]
                            ),
                            opponent_binary=(
                                None if opponent_action is None else opponent_action[1]
                            ),
                            include_state=_sampled_opponent_requires_state(
                                episode_opponent
                            ),
                            self_play=episode_is_self_play,
                        )
                    )
                env_step_start_time = time.perf_counter()
                step_results = _step_train_envs(env_backend, step_requests)
                env_step_seconds += time.perf_counter() - env_step_start_time
                for env_index, step_result in enumerate(step_results):
                    next_obs_raw = step_result.obs
                    terminated = step_result.terminated
                    truncated = step_result.truncated
                    info = _info_for_role(step_result.info, args.ego_role)
                    next_state = step_result.state
                    next_opponent_obs_raw = step_result.opponent_obs
                    episode_is_self_play = env_index in self_play_env_indices
                    buf_idx = env_index * 2 if dual_policy_slots else env_index
                    ego_action_cont = np.asarray(
                        step_requests[env_index].continuous,
                        dtype=np.float32,
                    )
                    ego_action_bin = np.asarray(
                        step_requests[env_index].binary,
                        dtype=np.float32,
                    )
                    reward_compute_start_time = time.perf_counter()
                    if args.worker_reward_mode == "worker":
                        reward_value = float(step_result.reward)
                    else:
                        ego_reward_runtime = reward_runtime_by_slot[buf_idx]
                        if ego_reward_runtime is None:
                            raise RuntimeError("ego reward runtime is not initialized")
                        reward = ego_reward_runtime.compute(
                            current_info=info,
                            current_obs=next_obs_raw,
                            action_cont=ego_action_cont,
                            action_bin=ego_action_bin,
                        )
                        reward_value = reward.total
                    enemy_reward_runtime: WorkerRewardStateMachine | None = None
                    enemy_action_cont: np.ndarray | None = None
                    enemy_action_bin: np.ndarray | None = None
                    info_enemy: dict[str, Any] | None = None
                    next_obs_enemy_raw: np.ndarray | None = None
                    if episode_is_self_play:
                        if next_opponent_obs_raw is None:
                            raise RuntimeError(
                                "self-play step must return the opponent observation"
                            )
                        enemy_role = _opposing_role(args.ego_role)
                        info_enemy = _info_for_role(info, enemy_role)
                        next_obs_enemy_raw = next_opponent_obs_raw
                        opponent_continuous = step_requests[env_index].opponent_continuous
                        opponent_binary = step_requests[env_index].opponent_binary
                        if opponent_continuous is None or opponent_binary is None:
                            raise RuntimeError("self-play step is missing the enemy action")
                        enemy_action_cont = np.asarray(opponent_continuous, dtype=np.float32)
                        enemy_action_bin = np.asarray(opponent_binary, dtype=np.float32)
                        if args.worker_reward_mode == "worker":
                            if step_result.opponent_reward is None:
                                raise RuntimeError(
                                    "worker self-play step is missing opponent reward"
                                )
                            enemy_reward_value = float(step_result.opponent_reward)
                        else:
                            enemy_reward_runtime = reward_runtime_by_slot[buf_idx + 1]
                            if enemy_reward_runtime is None:
                                raise RuntimeError(
                                    "enemy reward runtime is not initialized"
                                )
                            reward_enemy = enemy_reward_runtime.compute(
                                current_info=info_enemy,
                                current_obs=next_obs_enemy_raw,
                                action_cont=enemy_action_cont,
                                action_bin=enemy_action_bin,
                            )
                            enemy_reward_value = float(reward_enemy.total)
                        buffer_rewards[step, buf_idx + 1] = enemy_reward_value
                        reward_totals.append(enemy_reward_value)
                    if args.worker_reward_mode != "worker":
                        reward_compute_seconds += time.perf_counter() - reward_compute_start_time
                    early_truncated, next_trunc_runtime, truncation_reason = (
                        truncation_policy.check_with_reason(
                            info=info,
                            episode_stats=episode_stats[env_index],
                            runtime=truncation_runtime_by_env[env_index],
                        )
                    )
                    if early_truncated:
                        truncated = True
                    transition_done = bool(terminated or truncated)
                    if truncated and not terminated:
                        truncated_bootstrap_observations.append(
                            (
                                step,
                                buf_idx,
                                _normalize_obs(normalizer, next_obs_raw),
                            )
                        )
                        if episode_is_self_play:
                            if next_obs_enemy_raw is None:
                                raise RuntimeError(
                                    "self-play truncated transition is missing the enemy observation"
                                )
                            truncated_bootstrap_observations.append(
                                (
                                    step,
                                    buf_idx + 1,
                                    _normalize_obs(normalizer, next_obs_enemy_raw),
                                )
                            )
                    ego_reward_runtime = reward_runtime_by_slot[buf_idx]
                    if ego_reward_runtime is not None:
                        ego_reward_runtime.advance(
                            info=info,
                            action_cont=ego_action_cont,
                            action_bin=ego_action_bin,
                            terminated_or_truncated=transition_done,
                        )
                    if enemy_reward_runtime is not None:
                        if (
                            info_enemy is None
                            or enemy_action_cont is None
                            or enemy_action_bin is None
                        ):
                            raise RuntimeError("enemy reward transition is incomplete")
                        enemy_reward_runtime.advance(
                            info=info_enemy,
                            action_cont=enemy_action_cont,
                            action_bin=enemy_action_bin,
                            terminated_or_truncated=transition_done,
                        )
                    buffer_rewards[step, buf_idx] = reward_value
                    reward_totals.append(reward_value)
                    episode_stats[env_index]["return"] = float(episode_stats[env_index]["return"]) + reward_value
                    episode_stats[env_index]["length"] = int(episode_stats[env_index]["length"]) + 1
                    episode_stats[env_index]["max_out_of_bounds_seconds"] = max(
                        float(episode_stats[env_index]["max_out_of_bounds_seconds"]),
                        float(info["aircraft_by_role"][args.ego_role]["out_of_bounds_seconds"]),
                    )
                    s_hits, e_hits = _count_hits_from_events(
                        info.get("events_since_last_step", []),
                        self_role=args.ego_role,
                        enemy_role="fighter2" if args.ego_role == "fighter1" else "fighter1",
                    )
                    episode_stats[env_index]["self_hit_count"] = int(episode_stats[env_index]["self_hit_count"]) + s_hits
                    episode_stats[env_index]["enemy_hit_count"] = int(episode_stats[env_index]["enemy_hit_count"]) + e_hits
                    if terminated or truncated:
                        done_mask[buf_idx] = 1.0
                        terminated_mask[buf_idx] = 1.0 if terminated else 0.0
                        if episode_is_self_play:
                            done_mask[buf_idx + 1] = 1.0
                            terminated_mask[buf_idx + 1] = 1.0 if terminated else 0.0
                        episode_reward_diagnostics = _drain_episode_reward_diagnostics(
                            env_backend=env_backend,
                            worker_reward_mode=args.worker_reward_mode,
                            env_index=env_index,
                            reward_runtime=ego_reward_runtime,
                        )
                        recent_records.append(
                            _finalize_episode_record(
                                episode_stats[env_index],
                                info,
                                terminated=terminated,
                                truncated=truncated,
                                truncation_reason=truncation_reason,
                                reward_diagnostics=episode_reward_diagnostics,
                                opponent_label=(
                                    "current_policy_self_play"
                                    if episode_opponents[env_index] is None
                                    and episode_is_self_play
                                    else (
                                        episode_opponents[env_index].label
                                        if episode_opponents[env_index] is not None
                                        else args.opponent_mode
                                    )
                                ),
                                opponent_kind=(
                                    "self_play"
                                    if episode_is_self_play
                                    else (
                                        episode_opponents[env_index].runtime_kind
                                        if episode_opponents[env_index] is not None
                                        else args.opponent_mode
                                    )
                                ),
                                scene_label=episode_scene_labels[env_index],
                            )
                        )
                        reset_scene = None if prepared_scene_pool is None else prepared_scene_pool.sample_train_scene()
                        scene_name, scene_path = _resolve_scene_override(
                            reset_scene,
                            base_scene_asset,
                        )
                        reset_opponent = _sample_episode_opponent(prepared_opponent_pool)
                        _validate_worker_reward_support(
                            worker_reward_mode=args.worker_reward_mode,
                            env_backend=args.env_backend,
                            sampled_opponent=reset_opponent,
                        )
                        reset_is_self_play = _episode_uses_self_play(
                            configured_opponent_mode=args.opponent_mode,
                            sampled_opponent=reset_opponent,
                        )
                        env_reset_start_time = time.perf_counter()
                        reset_result = _reset_train_env_at(
                            env_backend,
                            index=env_index,
                            request=ResetRequest(
                                seed=args.seed + update * args.num_envs + env_index + 1000,
                                scene_name=scene_name,
                                scene_path=scene_path,
                                opponent_mode=_opponent_mode_for_env(args, reset_opponent),
                                include_state=_sampled_opponent_requires_state(
                                    reset_opponent
                                ),
                                self_play=reset_is_self_play,
                            ),
                        )
                        env_reset_seconds += time.perf_counter() - env_reset_start_time
                        next_obs_raw = reset_result.obs
                        next_opponent_obs_raw = reset_result.opponent_obs
                        info = _info_for_role(reset_result.info, args.ego_role)
                        episode_opponents[env_index] = reset_opponent
                        episode_scene_labels[env_index] = (
                            reset_scene.label
                            if reset_scene is not None
                            else base_scene_asset.scene_name
                        )
                        episode_start_sim_times_by_env[env_index] = float(
                            info["episode_start_sim_time_seconds"]
                        )
                        episode_stats[env_index] = {
                            "return": 0.0,
                            "length": 0,
                            "start_sim_time_seconds": float(info.get("sim_time_seconds", 0.0)),
                            "max_out_of_bounds_seconds": 0.0,
                            "self_hit_count": 0,
                            "enemy_hit_count": 0,
                        }
                        truncation_runtime_by_env[env_index] = truncation_policy.initial_runtime(info)
                        ego_reward_runtime = reward_runtime_by_slot[buf_idx]
                        if ego_reward_runtime is not None:
                            ego_reward_runtime.reset(initial_info=info)
                        if (
                            dual_policy_slots
                            and reset_is_self_play
                            and args.worker_reward_mode != "worker"
                        ):
                            reward_runtime_by_slot[buf_idx + 1] = _new_reward_runtime(
                                reward_composer,
                                initial_info=_info_for_role(
                                    info,
                                    _opposing_role(args.ego_role),
                                ),
                            )
                        elif dual_policy_slots:
                            reward_runtime_by_slot[buf_idx + 1] = None
                        next_state = reset_result.state
                    else:
                        truncation_runtime_by_env[env_index] = next_trunc_runtime
                    next_state_list.append(next_state)
                    next_obs_raw_list.append(next_obs_raw)
                    next_opponent_obs_raw_list.append(next_opponent_obs_raw)
                    global_step += 1
                buffer_dones[step] = done_mask
                buffer_terminated[step] = terminated_mask
                obs_raw = np.stack(next_obs_raw_list, axis=0).astype(np.float32, copy=False)
                obs = np.stack([_normalize_obs(normalizer, item) for item in obs_raw], axis=0).astype(
                    np.float32, copy=False
                )
                states_by_env = next_state_list
                opponent_obs_raw_by_env = next_opponent_obs_raw_list
            rollout_collection_seconds = time.perf_counter() - rollout_collection_start_time
            diagnostics_start_time = time.perf_counter()
            action_diagnostics = summarize_actions(
                continuous=buffer_action_cont_env,
                binary=buffer_action_bin,
                binary_probabilities=buffer_action_bin_prob,
                dones=buffer_dones,
                valid_mask=buffer_valid,
            )
            valid_transition_count = int(buffer_valid.sum())
            self_play_transition_count = int(buffer_self_play_episode.sum())
            self_play_opponent_transition_count = int(
                buffer_self_play_episode[:, 1::2].sum()
            ) if dual_policy_slots else 0
            action_diagnostics["valid_transition_count"] = valid_transition_count
            action_diagnostics["self_play_transition_count"] = (
                self_play_transition_count
            )
            action_diagnostics["self_play_opponent_transition_count"] = (
                self_play_opponent_transition_count
            )
            action_diagnostics["self_play_transition_fraction"] = (
                float(self_play_transition_count) / float(valid_transition_count)
                if valid_transition_count > 0
                else 0.0
            )
            action_diagnostics["episode_sampling"] = (
                _episode_sampling_diagnostics(recent_records)
            )
            reward_diagnostics = _drain_reward_diagnostics(
                env_backend=env_backend,
                worker_reward_mode=args.worker_reward_mode,
                reward_runtimes=reward_runtime_by_slot,
            )
            diagnostics_seconds += time.perf_counter() - diagnostics_start_time

            bootstrap_self_play_env_indices = [
                env_index
                for env_index, sampled_opponent in enumerate(episode_opponents)
                if _episode_uses_self_play(
                    configured_opponent_mode=args.opponent_mode,
                    sampled_opponent=sampled_opponent,
                )
            ]
            next_values = _compute_rollout_bootstrap_values(
                model=model,
                ego_obs_normalized=obs,
                device=device,
                self_play=self_play,
                self_play_env_indices=bootstrap_self_play_env_indices,
                dual_policy_slots=dual_policy_slots,
                opponent_obs_raw_by_env=opponent_obs_raw_by_env,
                normalizer=normalizer,
            )

            if truncated_bootstrap_observations:
                bootstrap_obs = np.stack(
                    [entry[2] for entry in truncated_bootstrap_observations],
                    axis=0,
                ).astype(np.float32, copy=False)
                with torch.no_grad():
                    bootstrap_values = (
                        model(torch.from_numpy(bootstrap_obs).to(device))
                        .value.cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                for (bootstrap_step, bootstrap_slot, _), bootstrap_value in zip(
                    truncated_bootstrap_observations,
                    bootstrap_values,
                    strict=True,
                ):
                    buffer_truncated_bootstrap_values[
                        bootstrap_step,
                        bootstrap_slot,
                    ] = bootstrap_value

            advantages = _compute_gae(
                rewards=buffer_rewards,
                values=buffer_values,
                dones=buffer_dones,
                terminated=buffer_terminated,
                truncated_bootstrap_values=buffer_truncated_bootstrap_values,
                next_values=next_values,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            returns = advantages + buffer_values
            valid_advantage_mean = float(advantages[buffer_valid].mean())

            flat_valid = buffer_valid.reshape(-1)
            flat_obs = torch.from_numpy(
                buffer_obs.reshape(-1, normalizer.obs_dim)[flat_valid]
            ).to(device)
            flat_action_cont_policy = torch.from_numpy(
                buffer_action_cont_policy.reshape(-1, 4)[flat_valid]
            ).to(device)
            flat_action_bin = torch.from_numpy(
                buffer_action_bin.reshape(-1, 3)[flat_valid]
            ).to(device)
            flat_log_probs = torch.from_numpy(
                buffer_log_probs.reshape(-1)[flat_valid]
            ).to(device)
            flat_advantages = torch.from_numpy(
                advantages.reshape(-1)[flat_valid]
            ).to(device)
            flat_returns = torch.from_numpy(returns.reshape(-1)[flat_valid]).to(device)
            flat_values = torch.from_numpy(
                buffer_values.reshape(-1)[flat_valid]
            ).to(device)
            action_diagnostics["rollout_log_prob_max_abs_error"] = (
                _assert_rollout_log_prob_consistency(
                    model=model,
                    observations=flat_obs,
                    action_cont=flat_action_cont_policy,
                    action_bin=flat_action_bin,
                    sampled_log_probs=flat_log_probs,
                    continuous_std=args.continuous_action_std,
                )
            )

            popart_mean_before = float(model.value_head.mean.item())
            popart_std_before = float(model.value_head.std.item())
            model.update_value_normalizer(
                flat_returns,
                beta=args.popart_beta,
                min_std=args.popart_min_std,
            )
            value_diagnostics = summarize_values(
                values=buffer_values[buffer_valid],
                returns=returns[buffer_valid],
                popart_mean_before=popart_mean_before,
                popart_std_before=popart_std_before,
                popart_mean_after=float(model.value_head.mean.item()),
                popart_std_after=float(model.value_head.std.item()),
            )
            flat_returns_norm = model.normalize_values(flat_returns)
            flat_values_norm = model.normalize_values(flat_values)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / flat_advantages.std().clamp_min(1e-8)
            batch_size = flat_obs.shape[0]
            indices = np.arange(batch_size)
            policy_loss_total = 0.0
            value_loss_total = 0.0
            continuous_entropy_total = 0.0
            binary_entropy_total = 0.0
            approx_kl_total = 0.0
            max_approx_kl = 0.0
            clip_fraction_total = 0.0
            batch_count = 0
            ppo_epochs_completed = 0
            target_kl_triggered = False

            ppo_update_start_time = time.perf_counter()
            for _ in range(args.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, args.minibatch_size):
                    end = min(start + args.minibatch_size, batch_size)
                    mb_idx = indices[start:end]
                    mb_obs = flat_obs[mb_idx]
                    mb_action_cont_policy = flat_action_cont_policy[mb_idx]
                    mb_action_bin = flat_action_bin[mb_idx]
                    mb_old_log_probs = flat_log_probs[mb_idx]
                    mb_advantages = flat_advantages[mb_idx]
                    mb_returns_norm = flat_returns_norm[mb_idx]
                    mb_old_values_norm = flat_values_norm[mb_idx]

                    (
                        new_log_probs,
                        continuous_entropy,
                        binary_entropy,
                        new_values,
                    ) = _evaluate_policy(
                        model,
                        mb_obs,
                        mb_action_cont_policy,
                        mb_action_bin,
                        continuous_std=args.continuous_action_std,
                    )
                    new_values_norm = model.normalize_values(new_values)
                    log_ratio = new_log_probs - mb_old_log_probs
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                        approximate_kl_value = float(approximate_kl.item())
                        max_approx_kl = max(max_approx_kl, approximate_kl_value)
                    if _target_kl_reached(
                        approximate_kl=approximate_kl_value,
                        target_kl=args.target_kl,
                    ):
                        target_kl_triggered = True
                        break
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                    pg_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                    value_pred_clipped = mb_old_values_norm + (new_values_norm - mb_old_values_norm).clamp(
                        -args.clip_eps, args.clip_eps
                    )
                    value_loss_unclipped = F.smooth_l1_loss(
                        new_values_norm,
                        mb_returns_norm,
                        reduction="none",
                        beta=args.value_huber_delta,
                    )
                    value_loss_clipped = F.smooth_l1_loss(
                        value_pred_clipped,
                        mb_returns_norm,
                        reduction="none",
                        beta=args.value_huber_delta,
                    )
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    continuous_entropy_mean = continuous_entropy.mean()
                    binary_entropy_mean = binary_entropy.mean()
                    loss = (
                        pg_loss
                        + args.value_coef * value_loss
                        - args.binary_entropy_coef * binary_entropy_mean
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        clip_fraction = ((ratio - 1.0).abs() > args.clip_eps).to(torch.float32).mean()
                    policy_loss_total += float(pg_loss.item())
                    value_loss_total += float(value_loss.item())
                    continuous_entropy_total += float(
                        continuous_entropy_mean.item()
                    )
                    binary_entropy_total += float(binary_entropy_mean.item())
                    approx_kl_total += approximate_kl_value
                    clip_fraction_total += float(clip_fraction.item())
                    batch_count += 1
                if target_kl_triggered:
                    break
                ppo_epochs_completed += 1
            ppo_update_seconds = time.perf_counter() - ppo_update_start_time
            policy_reference_metrics = None
            should_measure_policy_drift = (
                policy_reference_model is not None
                and (
                    (update + 1) % args.policy_drift_interval == 0
                    or update == target_update - 1
                )
            )
            if should_measure_policy_drift:
                diagnostics_start_time = time.perf_counter()
                policy_reference_metrics = _policy_reference_diagnostics(
                    model=model,
                    reference_model=policy_reference_model,
                    observations=flat_obs,
                    sample_size=args.policy_drift_sample_size,
                )
                diagnostics_seconds += time.perf_counter() - diagnostics_start_time

            eval_metrics = None
            eval_seconds = 0.0
            if (update + 1) % args.eval_interval == 0 or update == target_update - 1:
                eval_start_time = time.perf_counter()
                eval_metrics = _run_eval(
                    model=model,
                    normalizer=normalizer,
                    reward_composer=reward_composer,
                    config=base_env_config,
                    eval_scenes=None if prepared_scene_pool is None else prepared_scene_pool.eval_scenes,
                    prepared_opponent_pool=prepared_opponent_pool,
                    episodes=args.eval_episodes,
                    max_steps=args.eval_max_steps,
                    max_seconds=args.eval_max_seconds,
                    device=device,
                )
                eval_seconds = time.perf_counter() - eval_start_time

            update_episode_metrics = _aggregate_episode_metrics(recent_records)
            rolling_records.extend(recent_records)
            rollout_window = _aggregate_episode_metrics(list(rolling_records))
            reward_diagnostics_by_outcome = _reward_diagnostics_by_outcome(
                list(rolling_records)
            )
            rollout_transitions = args.rollout_steps * args.num_envs
            rollout_steps_per_second = (
                float(rollout_transitions) / rollout_collection_seconds if rollout_collection_seconds > 0.0 else 0.0
            )
            shared_lr, actor_lr, critic_lr = _optimizer_learning_rates(optimizer)
            update_compute_seconds = time.perf_counter() - update_start_time
            metrics = PpoTrainMetrics(
                update=update,
                global_step=global_step,
                rollout_return_mean=update_episode_metrics.mean_return,
                rollout_length_mean=update_episode_metrics.mean_length,
                rollout_survival_seconds_mean=update_episode_metrics.mean_duration_seconds,
                rollout_episode_duration_seconds_mean=update_episode_metrics.mean_duration_seconds,
                rollout_self_destroy_rate=update_episode_metrics.self_destroy_rate,
                rollout_enemy_destroy_rate=update_episode_metrics.enemy_destroy_rate,
                rollout_mutual_destroy_rate=update_episode_metrics.mutual_destroy_rate,
                rollout_out_of_bounds_destroy_rate=(
                    update_episode_metrics.out_of_bounds_destroy_rate
                ),
                rollout_truncated_rate=update_episode_metrics.truncated_rate,
                rollout_mean_max_out_of_bounds_seconds=(
                    update_episode_metrics.mean_max_out_of_bounds_seconds
                ),
                rollout_self_mean_hit_count=update_episode_metrics.self_mean_hit_count,
                rollout_enemy_mean_hit_count=update_episode_metrics.enemy_mean_hit_count,
                rollout_episode_count=update_episode_metrics.episode_count,
                rollout_termination_reason_counts=(
                    update_episode_metrics.termination_reason_counts
                ),
                rollout_window=rollout_window,
                policy_loss=policy_loss_total / max(batch_count, 1),
                value_loss=value_loss_total / max(batch_count, 1),
                continuous_entropy=(
                    continuous_entropy_total / max(batch_count, 1)
                ),
                binary_entropy=binary_entropy_total / max(batch_count, 1),
                approx_kl=approx_kl_total / max(batch_count, 1),
                max_approx_kl=max_approx_kl,
                clip_fraction=clip_fraction_total / max(batch_count, 1),
                target_kl=args.target_kl,
                target_kl_triggered=target_kl_triggered,
                ppo_epochs_completed=ppo_epochs_completed,
                ppo_minibatches_completed=batch_count,
                shared_learning_rate=shared_lr,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                reward_mean=float(np.mean(reward_totals)) if reward_totals else 0.0,
                advantage_mean=valid_advantage_mean,
                action_diagnostics=action_diagnostics,
                reward_diagnostics=reward_diagnostics,
                reward_diagnostics_by_outcome=reward_diagnostics_by_outcome,
                value_diagnostics=value_diagnostics,
                policy_reference_diagnostics=policy_reference_metrics,
                rollout_collection_seconds=rollout_collection_seconds,
                rollout_steps_per_second=rollout_steps_per_second,
                policy_forward_seconds=policy_forward_seconds,
                opponent_action_seconds=opponent_action_seconds,
                env_step_seconds=env_step_seconds,
                reward_compute_seconds=reward_compute_seconds,
                diagnostics_seconds=diagnostics_seconds,
                env_reset_seconds=env_reset_seconds,
                ppo_update_seconds=ppo_update_seconds,
                eval_seconds=eval_seconds,
                checkpoint_seconds=0.0,
                update_compute_seconds=update_compute_seconds,
                update_total_seconds=update_compute_seconds,
                eval=eval_metrics,
            )
            recent_records.clear()

            is_new_best = _is_new_best_eval(
                eval_metrics,
                best_eval_return=best_eval_return,
            )
            if is_new_best:
                best_eval_return = eval_metrics.mean_return

            checkpoint_start_time = time.perf_counter()
            if is_new_best:
                _save_checkpoint(
                    path=checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    update=update,
                    global_step=global_step,
                    best_eval_return=best_eval_return,
                    metrics=metrics,
                    dataset_contract=dataset_contract,
                    training_semantics=training_semantics,
                )

            if _should_save_periodic_checkpoint(
                update=update,
                target_update=target_update,
                checkpoint_interval=args.checkpoint_interval,
            ):
                _save_checkpoint(
                    path=checkpoints_dir / f"update_{update:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    update=update,
                    global_step=global_step,
                    best_eval_return=best_eval_return,
                    metrics=metrics,
                    dataset_contract=dataset_contract,
                    training_semantics=training_semantics,
                )

            _save_checkpoint(
                path=checkpoints_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                update=update,
                global_step=global_step,
                best_eval_return=best_eval_return,
                metrics=metrics,
                dataset_contract=dataset_contract,
                training_semantics=training_semantics,
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_start_time
            metrics = replace(
                metrics,
                checkpoint_seconds=checkpoint_seconds,
                update_total_seconds=time.perf_counter() - update_start_time,
            )

            metrics_payload = asdict(metrics)
            metrics_payload["step_timing_label"] = (
                "step_bundle_seconds" if args.worker_reward_mode == "worker" else "env_step_seconds"
            )
            metrics_payload["step_timing_seconds"] = metrics.env_step_seconds
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics_payload, ensure_ascii=False) + "\n")
            (eval_dir / f"update_{update:04d}.json").write_text(
                json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            step_timing_label = "t_step_bundle" if args.worker_reward_mode == "worker" else "t_env"
            # --- Overview ---
            print(
                f"[Update] #{update:04d}  step={global_step:07d}  "
                f"reward_mean={metrics.reward_mean:.4f}"
            )
            # --- Rollout ---
            print(
                f"[Rollout] return={metrics.rollout_return_mean:.4f}  "
                f"eps={metrics.rollout_episode_count}  "
                f"len={metrics.rollout_length_mean:.1f}  "
                f"dur={metrics.rollout_episode_duration_seconds_mean:.2f}s  "
                f"self_destroy={metrics.rollout_self_destroy_rate:.3f}  "
                f"enemy_destroy={metrics.rollout_enemy_destroy_rate:.3f}  "
                f"trunc={metrics.rollout_truncated_rate:.3f}  "
                f"hit_self={metrics.rollout_self_mean_hit_count:.2f}  "
                f"hit_enemy={metrics.rollout_enemy_mean_hit_count:.2f}"
            )
            print(
                f"[Window] eps={metrics.rollout_window.episode_count}  "
                f"return={metrics.rollout_window.mean_return:.4f}  "
                f"dur={metrics.rollout_window.mean_duration_seconds:.2f}s  "
                f"self_destroy={metrics.rollout_window.self_destroy_rate:.3f}  "
                f"enemy_destroy={metrics.rollout_window.enemy_destroy_rate:.3f}  "
                f"oob_destroy={metrics.rollout_window.out_of_bounds_destroy_rate:.3f}  "
                f"trunc={metrics.rollout_window.truncated_rate:.3f}"
            )
            # --- Losses ---
            print(
                f"[Loss] policy={metrics.policy_loss:.4f}  "
                f"value={metrics.value_loss:.4f}  "
                f"cont_entropy={metrics.continuous_entropy:.4f}  "
                f"bin_entropy={metrics.binary_entropy:.4f}  "
                f"approx_kl={metrics.approx_kl:.4f}  "
                f"max_kl={metrics.max_approx_kl:.4f}  "
                f"clip_frac={metrics.clip_fraction:.4f}  "
                f"epochs={metrics.ppo_epochs_completed}/{args.ppo_epochs}  "
                f"kl_stop={int(metrics.target_kl_triggered)}"
            )
            # --- Learning rates ---
            print(
                f"[LR] shared={metrics.shared_learning_rate:.3g}  "
                f"actor={metrics.actor_learning_rate:.3g}  "
                f"critic={metrics.critic_learning_rate:.3g}"
            )
            continuous_diagnostics = metrics.action_diagnostics["continuous"]
            binary_diagnostics = metrics.action_diagnostics["binary"]
            print(
                "[Action] "
                f"roll_sat={continuous_diagnostics['saturation_fraction']['roll']:.3f}  "
                f"yaw_sat={continuous_diagnostics['saturation_fraction']['yaw']:.3f}  "
                f"brake_on={binary_diagnostics['on_fraction']['brake']:.3f}  "
                f"fire_on={binary_diagnostics['on_fraction']['fire_gun']:.3f}  "
                f"repair_on={binary_diagnostics['on_fraction']['repair']:.3f}"
            )
            episode_sampling = metrics.action_diagnostics["episode_sampling"]
            print(
                "[Pool] "
                f"eps={episode_sampling['episode_count']}  "
                f"self_play_eps={episode_sampling['self_play_episode_fraction']:.3f}  "
                f"self_play_samples={metrics.action_diagnostics['self_play_transition_fraction']:.3f}  "
                f"valid_samples={metrics.action_diagnostics['valid_transition_count']}"
            )
            print(
                f"[Value] ev={metrics.value_diagnostics['explained_variance']:.4f}  "
                f"mae={metrics.value_diagnostics['raw_mae']:.3f}  "
                f"rmse={metrics.value_diagnostics['raw_rmse']:.3f}  "
                f"popart_mean={metrics.value_diagnostics['popart_mean_after']:.3f}  "
                f"popart_std={metrics.value_diagnostics['popart_std_after']:.3f}"
            )
            if metrics.policy_reference_diagnostics is not None:
                print(
                    "[Drift] "
                    f"cont_abs={metrics.policy_reference_diagnostics['continuous_mean_abs_drift']:.4f}  "
                    f"bin_abs={metrics.policy_reference_diagnostics['binary_probability_mean_abs_drift']:.4f}  "
                    f"bin_kl={metrics.policy_reference_diagnostics['binary_reference_to_current_kl']:.4f}"
                )
            # --- Timing ---
            print(
                f"[Time] rollout={metrics.rollout_collection_seconds:.2f}s  "
                f"policy={metrics.policy_forward_seconds:.2f}s  "
                f"opp={metrics.opponent_action_seconds:.2f}s  "
                f"{step_timing_label}={metrics.env_step_seconds:.2f}s  "
                f"reward={metrics.reward_compute_seconds:.2f}s  "
                f"diag={metrics.diagnostics_seconds:.2f}s  "
                f"reset={metrics.env_reset_seconds:.2f}s  "
                f"ppo={metrics.ppo_update_seconds:.2f}s  "
                f"eval={metrics.eval_seconds:.2f}s  "
                f"ckpt={metrics.checkpoint_seconds:.2f}s  "
                f"total={metrics.update_total_seconds:.2f}s  "
                f"sps={metrics.rollout_steps_per_second:.1f}"
            )
            # --- Eval (if any) ---
            if metrics.eval is not None:
                print(
                    f"[Eval] return={metrics.eval.mean_return:.4f}  "
                    f"surv={metrics.eval.mean_survival_seconds:.2f}s  "
                    f"self_destroy={metrics.eval.self_destroy_rate:.4f}  "
                    f"enemy_destroy={metrics.eval.enemy_destroy_rate:.4f}  "
                    f"timeout={metrics.eval.timeout_rate:.4f}  "
                    f"hit_self={metrics.eval.self_mean_hit_count:.2f}  "
                    f"hit_enemy={metrics.eval.enemy_mean_hit_count:.2f}"
                )
            # --- blank line between updates ---
            print()
    finally:
        _close_train_env_backend(env_backend)


if __name__ == "__main__":
    main()
