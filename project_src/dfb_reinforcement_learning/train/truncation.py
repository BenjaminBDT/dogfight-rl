"""Episode truncation policies for PPO training.

Each policy decides whether to force-end an episode early based on
observable state, independent of environment termination (crash, kill).
Policies are composable via :class:`CompositePolicy`.

Usage in training loop::

    policy = build_truncation_policy(args, reward_composer=composer)
    runtime = policy.initial_runtime(initial_info)

    for step in rollout:
        ...
        early_truncated, runtime, reason = policy.check_with_reason(
            info=info,
            episode_stats=stats[env_index],
            runtime=runtime,
        )
        if early_truncated:
            truncated = True
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

from dfb_reinforcement_learning.rewards.policy_reward import PolicyRewardComposer


class TruncationPolicy:
    """Abstract truncation policy.

    Subclasses override :meth:`_check` and :meth:`_initial_runtime`.
    """

    def check(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> tuple[bool, object]:
        """Return ``(should_truncate, new_runtime)``."""
        return self._check(info=info, episode_stats=episode_stats, runtime=runtime)

    def check_with_reason(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> tuple[bool, object, str | None]:
        should_truncate, next_runtime = self._check(
            info=info,
            episode_stats=episode_stats,
            runtime=runtime,
        )
        reason = (
            self._truncation_reason(
                info=info,
                episode_stats=episode_stats,
                runtime=next_runtime,
            )
            if should_truncate
            else None
        )
        return should_truncate, next_runtime, reason

    def initial_runtime(self, info: dict[str, Any]) -> object:
        """Return the runtime state for a fresh episode."""
        return self._initial_runtime(info)

    # ── override points ──

    def _check(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> tuple[bool, object]:
        return False, runtime

    def _initial_runtime(self, info: dict[str, Any]) -> object:
        return None

    def _truncation_reason(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> str:
        return "training_truncation"


class NoTruncation(TruncationPolicy):
    """Never truncate (default)."""


# ── Built-in policies ────────────────────────────────────────────────────────


class MaxEpisodeSeconds(TruncationPolicy):
    """Truncate after *max_seconds* of sim time."""

    def __init__(self, max_seconds: float) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        self._max_seconds = max_seconds

    def _check(self, *, info, episode_stats, runtime):
        start = float(episode_stats.get("start_sim_time_seconds", 0.0))
        elapsed = float(info.get("sim_time_seconds", 0.0)) - start
        return elapsed >= self._max_seconds, runtime

    def _initial_runtime(self, info):
        return None

    def _truncation_reason(self, *, info, episode_stats, runtime):
        return "training_time_limit"


@dataclass(frozen=True)
class OpeningShotWindowRuntime:
    episode_start_sim_time: float
    previous_sim_time: float
    window_seen: bool = False
    window_lost_seconds: float = 0.0


class OpeningShotWindow(TruncationPolicy):
    """Truncate after the *opening shot window* closes.

    The window is "seen" once shot feasibility crosses *activate_threshold*.
    After that, if feasibility stays below *keep_threshold* for *loss_seconds*
    (cumulative), the episode is truncated.  Also truncated after
    *max_seconds* total elapsed sim time regardless.
    """

    def __init__(
        self,
        *,
        max_seconds: float,
        activate_threshold: float,
        keep_threshold: float,
        loss_seconds: float,
        reward_composer: PolicyRewardComposer,
        ego_role: str,
    ) -> None:
        self._max_seconds = max_seconds
        self._activate_threshold = activate_threshold
        self._keep_threshold = keep_threshold
        self._loss_seconds = loss_seconds
        self._reward_composer = reward_composer
        self._ego_role = ego_role

    def _initial_runtime(self, info: dict[str, Any]) -> OpeningShotWindowRuntime:
        sim_time = float(info["sim_time_seconds"])
        return OpeningShotWindowRuntime(
            episode_start_sim_time=sim_time,
            previous_sim_time=sim_time,
        )

    def _check(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> tuple[bool, object]:
        if not isinstance(runtime, OpeningShotWindowRuntime):
            raise TypeError("opening-shot-window runtime has an invalid type")
        current_sim_time = float(info["sim_time_seconds"])
        elapsed = max(current_sim_time - runtime.episode_start_sim_time, 0.0)
        dt = max(current_sim_time - runtime.previous_sim_time, 0.0)
        shot_feasibility, attack_advantage = self._compute_attack_metrics(info)

        window_seen = runtime.window_seen
        window_lost_seconds = runtime.window_lost_seconds
        if shot_feasibility >= self._activate_threshold:
            window_seen = True
            window_lost_seconds = 0.0
        elif window_seen:
            if shot_feasibility < self._keep_threshold:
                window_lost_seconds += dt
            else:
                window_lost_seconds = 0.0

        next_runtime = OpeningShotWindowRuntime(
            episode_start_sim_time=runtime.episode_start_sim_time,
            previous_sim_time=current_sim_time,
            window_seen=window_seen,
            window_lost_seconds=window_lost_seconds,
        )
        info["opening_shot_window"] = {
            "shot_feasibility": shot_feasibility,
            "attack_advantage": attack_advantage,
            "window_seen": window_seen,
            "window_lost_seconds": window_lost_seconds,
            "elapsed_seconds": elapsed,
        }

        should_truncate = elapsed >= self._max_seconds or (
            window_seen and window_lost_seconds >= self._loss_seconds
        )
        return should_truncate, next_runtime

    def _compute_attack_metrics(self, info: dict[str, Any]) -> tuple[float, float]:
        enemy_role = "fighter2" if self._ego_role == "fighter1" else "fighter1"
        components = self._reward_composer._compute_attack_advantage(
            attacker_state=info["aircraft_by_role"][self._ego_role],
            defender_state=info["aircraft_by_role"][enemy_role],
        )
        return float(components.shot_feasibility), float(components.attack_advantage)

    def _truncation_reason(self, *, info, episode_stats, runtime):
        elapsed = float(info["sim_time_seconds"]) - float(
            episode_stats.get("start_sim_time_seconds", 0.0)
        )
        if elapsed >= self._max_seconds:
            return "opening_shot_window_time_limit"
        return "opening_shot_window_lost"


@dataclass(frozen=True)
class TacticalAdvantageRuntime:
    previous_sim_time: float
    smoothed_margin: float | None = None
    activation_seconds: float = 0.0
    loss_seconds: float = 0.0
    armed: bool = False


class TacticalAdvantageTruncation(TruncationPolicy):
    """Crop a rollout after an established tactical advantage is persistently lost."""

    def __init__(
        self,
        *,
        loss_threshold: float,
        activate_threshold: float,
        ema_seconds: float,
        activate_seconds: float,
        loss_seconds: float,
        reward_composer: PolicyRewardComposer,
        ego_role: str = "fighter1",
    ) -> None:
        for name, value in (
            ("loss_threshold", loss_threshold),
            ("activate_threshold", activate_threshold),
            ("ema_seconds", ema_seconds),
            ("activate_seconds", activate_seconds),
            ("loss_seconds", loss_seconds),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if activate_threshold <= loss_threshold:
            raise ValueError("activate_threshold must be greater than loss_threshold")
        if ema_seconds < 0.0:
            raise ValueError("ema_seconds must be non-negative")
        if activate_seconds < 0.0:
            raise ValueError("activate_seconds must be non-negative")
        if loss_seconds <= 0.0:
            raise ValueError("loss_seconds must be positive")
        self._loss_threshold = loss_threshold
        self._activate_threshold = activate_threshold
        self._ema_seconds = ema_seconds
        self._activate_seconds = activate_seconds
        self._loss_seconds = loss_seconds
        self._reward_composer = reward_composer
        self._ego_role = ego_role

    def _initial_runtime(self, info: dict[str, Any]) -> TacticalAdvantageRuntime:
        return TacticalAdvantageRuntime(previous_sim_time=float(info["sim_time_seconds"]))

    def _check(
        self,
        *,
        info: dict[str, Any],
        episode_stats: dict[str, float | int],
        runtime: object,
    ) -> tuple[bool, object]:
        del episode_stats
        if not isinstance(runtime, TacticalAdvantageRuntime):
            raise TypeError("tactical-advantage runtime has an invalid type")
        info_ego_role = str(info.get("ego_role", self._ego_role))
        if info_ego_role != self._ego_role:
            raise ValueError(
                f"tactical-advantage role mismatch: expected {self._ego_role}, got {info_ego_role}"
            )

        current_sim_time = float(info["sim_time_seconds"])
        dt = max(current_sim_time - runtime.previous_sim_time, 0.0)
        cache = self._reward_composer.build_attack_history_cache(info)
        self_attack = float(cache["self_attack"].attack_advantage)
        enemy_attack = float(cache["opponent_attack"].attack_advantage)
        raw_margin = self_attack - enemy_attack

        if runtime.smoothed_margin is None or self._ema_seconds == 0.0:
            smoothed_margin = raw_margin
        else:
            alpha = -math.expm1(-dt / self._ema_seconds)
            smoothed_margin = runtime.smoothed_margin + alpha * (
                raw_margin - runtime.smoothed_margin
            )

        armed = runtime.armed
        activation_seconds = runtime.activation_seconds
        loss_seconds = runtime.loss_seconds
        if not armed:
            if smoothed_margin >= self._activate_threshold:
                activation_seconds += dt
            else:
                activation_seconds = 0.0
            armed = activation_seconds >= self._activate_seconds

        if armed:
            if smoothed_margin < self._loss_threshold:
                loss_seconds += dt
            else:
                loss_seconds = 0.0

        next_runtime = TacticalAdvantageRuntime(
            previous_sim_time=current_sim_time,
            smoothed_margin=smoothed_margin,
            activation_seconds=activation_seconds,
            loss_seconds=loss_seconds,
            armed=armed,
        )
        info["tactical_advantage_window"] = {
            "self_attack_advantage": self_attack,
            "enemy_attack_advantage": enemy_attack,
            "raw_margin": raw_margin,
            "smoothed_margin": smoothed_margin,
            "armed": armed,
            "activation_seconds": activation_seconds,
            "loss_seconds": loss_seconds,
        }
        return armed and loss_seconds >= self._loss_seconds, next_runtime

    def _truncation_reason(self, *, info, episode_stats, runtime):
        return "tactical_advantage_lost"


class CompositePolicy(TruncationPolicy):
    """OR-combination: truncates if *any* sub-policy triggers."""

    def __init__(self, *policies: TruncationPolicy) -> None:
        self._policies = policies

    def _check(self, *, info, episode_stats, runtime):
        should_truncate, next_runtime, _ = self.check_with_reason(
            info=info,
            episode_stats=episode_stats,
            runtime=runtime,
        )
        return should_truncate, next_runtime

    def check_with_reason(self, *, info, episode_stats, runtime):
        runtimes = runtime if isinstance(runtime, tuple) else (runtime,) * len(self._policies)
        next_runtimes: list[object] = []
        should_truncate = False
        first_reason: str | None = None
        for policy, rt in zip(self._policies, runtimes):
            trunc, next_rt, reason = policy.check_with_reason(
                info=info,
                episode_stats=episode_stats,
                runtime=rt,
            )
            next_runtimes.append(next_rt)
            should_truncate = should_truncate or trunc
            if trunc and first_reason is None:
                first_reason = reason
        return should_truncate, tuple(next_runtimes), first_reason

    def _initial_runtime(self, info):
        return tuple(p.initial_runtime(info) for p in self._policies)


# ── Factory ──────────────────────────────────────────────────────────────────


def build_truncation_policy(
    args: argparse.Namespace,
    *,
    reward_composer: PolicyRewardComposer | None = None,
) -> TruncationPolicy:
    """Build a truncation policy from CLI args."""
    policies: list[TruncationPolicy] = []

    if args.max_episode_seconds > 0.0:
        policies.append(MaxEpisodeSeconds(args.max_episode_seconds))

    if args.tactical_advantage_loss_threshold is not None:
        if reward_composer is None:
            raise ValueError("reward_composer is required for tactical-advantage truncation")
        policies.append(
            TacticalAdvantageTruncation(
                loss_threshold=args.tactical_advantage_loss_threshold,
                activate_threshold=args.tactical_advantage_activate_threshold,
                ema_seconds=args.tactical_advantage_ema_seconds,
                activate_seconds=args.tactical_advantage_activate_seconds,
                loss_seconds=args.tactical_advantage_loss_seconds,
                reward_composer=reward_composer,
                ego_role=args.ego_role,
            )
        )

    if args.opening_shot_window_mode == "truncate":
        if reward_composer is None:
            raise ValueError("reward_composer is required for opening-shot-window truncation")
        policies.append(
            OpeningShotWindow(
                max_seconds=args.opening_shot_window_max_seconds,
                activate_threshold=args.opening_shot_window_activate_threshold,
                keep_threshold=args.opening_shot_window_keep_threshold,
                loss_seconds=args.opening_shot_window_loss_seconds,
                reward_composer=reward_composer,
                ego_role=args.ego_role,
            )
        )

    if not policies:
        return NoTruncation()
    if len(policies) == 1:
        return policies[0]
    return CompositePolicy(*policies)
