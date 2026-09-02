from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.connection import Connection
import time
import traceback
from typing import Any, Sequence

import numpy as np

from .policy_dogfight_env import PolicyDogfightEnv, PolicyDogfightEnvConfig
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig
from dfb_reinforcement_learning.rewards.reward_diagnostics import RewardDiagnosticsAccumulator


@dataclass(frozen=True)
class ResetRequest:
    seed: int | None = None
    scene_name: str | None = None
    scene_path: str | None = None
    opponent_mode: str | None = None
    include_state: bool = False
    self_play: bool = False


@dataclass(frozen=True)
class StepRequest:
    continuous: np.ndarray
    binary: np.ndarray
    binary_threshold: float = 0.5
    opponent_continuous: np.ndarray | None = None
    opponent_binary: np.ndarray | None = None
    include_state: bool = False
    self_play: bool = False


@dataclass(frozen=True)
class ResetResult:
    obs: np.ndarray
    info: dict[str, Any]
    state: dict[str, Any] | None
    opponent_obs: np.ndarray | None = None


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    state: dict[str, Any] | None
    opponent_obs: np.ndarray | None = None
    opponent_reward: float | None = None


def _append_worker_reward_history_frame(
    *,
    composer: Any,
    reward_history_frames: list[dict[str, Any]],
    info: dict[str, Any],
    action_cont: np.ndarray,
    action_bin: np.ndarray,
    max_length: int,
) -> None:
    frame: dict[str, Any] = {
        "info": info,
        "action_cont": np.asarray(action_cont, dtype=np.float32).copy(),
        "action_bin": np.asarray(action_bin, dtype=np.float32).copy(),
    }
    build_attack_history_cache = getattr(composer, "build_attack_history_cache", None)
    if callable(build_attack_history_cache):
        frame["attack_history_cache"] = build_attack_history_cache(info)
    build_boundary_phi_cache = getattr(composer, "build_boundary_phi_cache", None)
    if callable(build_boundary_phi_cache):
        frame["boundary_phi_cache"] = build_boundary_phi_cache(info)
    reward_history_frames.append(frame)
    if len(reward_history_frames) > max_length:
        del reward_history_frames[:-max_length]


@dataclass
class WorkerRewardStateMachine:
    """Per-worker reward runtime state for reward-in-worker rollout execution."""

    composer: Any
    previous_info: dict[str, Any] | None = None
    previous_previous_action_cont: np.ndarray | None = None
    previous_action_cont: np.ndarray | None = None
    reward_history_frames: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: RewardDiagnosticsAccumulator = field(default_factory=RewardDiagnosticsAccumulator)
    episode_diagnostics: RewardDiagnosticsAccumulator = field(
        default_factory=RewardDiagnosticsAccumulator
    )
    reward_history_frame_length: int = field(init=False)

    def __post_init__(self) -> None:
        reward_history_frame_length = getattr(self.composer, "reward_history_frame_length", None)
        if not callable(reward_history_frame_length):
            raise TypeError("composer must provide reward_history_frame_length()")
        self.reward_history_frame_length = int(reward_history_frame_length())

    def reset(
        self,
        *,
        initial_info: dict[str, Any] | None = None,
        clear_episode_diagnostics: bool = True,
    ) -> None:
        self.previous_info = initial_info
        self.previous_previous_action_cont = None
        self.previous_action_cont = None
        self.reward_history_frames.clear()
        if clear_episode_diagnostics:
            self.episode_diagnostics.clear()

    def compute(
        self,
        *,
        current_info: dict[str, Any],
        current_obs: np.ndarray,
        action_cont: np.ndarray,
        action_bin: np.ndarray,
    ) -> Any:
        breakdown = self.composer.compute(
            previous_info=self.previous_info,
            previous_previous_action_cont=self.previous_previous_action_cont,
            previous_action_cont=self.previous_action_cont,
            reward_history={"frames": self.reward_history_frames},
            current_info=current_info,
            current_obs=current_obs,
            action_cont=action_cont,
            action_bin=action_bin,
        )
        self.diagnostics.add(breakdown)
        self.episode_diagnostics.add(breakdown)
        return breakdown

    def drain_reward_diagnostics(self) -> dict[str, Any]:
        return self.diagnostics.drain_raw_payload()

    def drain_episode_reward_diagnostics(self) -> dict[str, Any]:
        return self.episode_diagnostics.drain_raw_payload()

    def advance(
        self,
        *,
        info: dict[str, Any],
        action_cont: np.ndarray,
        action_bin: np.ndarray,
        terminated_or_truncated: bool,
    ) -> None:
        if terminated_or_truncated:
            self.reset(clear_episode_diagnostics=False)
            return
        _append_worker_reward_history_frame(
            composer=self.composer,
            reward_history_frames=self.reward_history_frames,
            info=info,
            action_cont=action_cont,
            action_bin=action_bin,
            max_length=self.reward_history_frame_length,
        )
        self.previous_info = info
        self.previous_previous_action_cont = self.previous_action_cont
        self.previous_action_cont = np.asarray(action_cont, dtype=np.float32).copy()


def _worker_handle_command(
    env: PolicyDogfightEnv,
    reward_runtime: WorkerRewardStateMachine | None,
    opponent_reward_runtime: WorkerRewardStateMachine | None = None,
    *,
    command: str,
    payload: Any,
) -> Any:
    if command == "reset":
        if payload is None:
            payload = ResetRequest()
        if not isinstance(payload, ResetRequest):
            raise TypeError(f"reset payload must be ResetRequest, got {type(payload)!r}")
        obs, info = env.reset(
            seed=payload.seed,
            scene_name=payload.scene_name,
            scene_path=payload.scene_path,
            opponent_mode=payload.opponent_mode,
        )
        if reward_runtime is not None:
            reward_runtime.reset(initial_info=info)
        if opponent_reward_runtime is not None:
            opponent_reward_runtime.reset(
                initial_info=(
                    {
                        **info,
                        "ego_role": env.enemy_role,
                        "enemy_role": env.config.ego_role,
                    }
                    if payload.self_play
                    else None
                )
            )
        return ResetResult(
            obs=obs,
            info=info,
            state=env.latest_state() if payload.include_state else None,
            opponent_obs=(
                env.observation_for_role(env.enemy_role)
                if payload.self_play
                else None
            ),
        )
    if command == "step":
        if not isinstance(payload, StepRequest):
            raise TypeError(f"step payload must be StepRequest, got {type(payload)!r}")
        obs, reward, terminated, truncated, info = env.step_arrays(
            payload.continuous,
            payload.binary,
            binary_threshold=payload.binary_threshold,
            opponent_continuous=payload.opponent_continuous,
            opponent_binary=payload.opponent_binary,
        )
        reward_value = reward
        opponent_obs = (
            env.observation_for_role(env.enemy_role)
            if payload.self_play
            else None
        )
        opponent_reward_value: float | None = None
        if reward_runtime is not None:
            reward_breakdown = reward_runtime.compute(
                current_info=info,
                current_obs=obs,
                action_cont=payload.continuous,
                action_bin=payload.binary,
            )
            reward_value = float(reward_breakdown.total)
            reward_runtime.advance(
                info=info,
                action_cont=payload.continuous,
                action_bin=payload.binary,
                terminated_or_truncated=bool(terminated or truncated),
            )
        if payload.self_play and opponent_reward_runtime is not None:
            if opponent_obs is None:
                raise RuntimeError("self-play opponent observation is missing")
            if (
                payload.opponent_continuous is None
                or payload.opponent_binary is None
            ):
                raise ValueError("self-play opponent reward requires opponent action")
            opponent_info = {
                **info,
                "ego_role": env.enemy_role,
                "enemy_role": env.config.ego_role,
            }
            opponent_breakdown = opponent_reward_runtime.compute(
                current_info=opponent_info,
                current_obs=opponent_obs,
                action_cont=payload.opponent_continuous,
                action_bin=payload.opponent_binary,
            )
            opponent_reward_value = float(opponent_breakdown.total)
            opponent_reward_runtime.advance(
                info=opponent_info,
                action_cont=payload.opponent_continuous,
                action_bin=payload.opponent_binary,
                terminated_or_truncated=bool(terminated or truncated),
            )
        return StepResult(
            obs=obs,
            reward=reward_value,
            terminated=terminated,
            truncated=truncated,
            info=info,
            state=env.latest_state() if payload.include_state else None,
            opponent_obs=opponent_obs,
            opponent_reward=opponent_reward_value,
        )
    if command == "drain_reward_diagnostics":
        if reward_runtime is None:
            return RewardDiagnosticsAccumulator().raw_payload()
        accumulator = RewardDiagnosticsAccumulator()
        accumulator.merge_payload(reward_runtime.drain_reward_diagnostics())
        if opponent_reward_runtime is not None:
            accumulator.merge_payload(
                opponent_reward_runtime.drain_reward_diagnostics()
            )
        return accumulator.raw_payload()
    if command == "drain_episode_reward_diagnostics":
        if reward_runtime is None:
            return RewardDiagnosticsAccumulator().raw_payload()
        return reward_runtime.drain_episode_reward_diagnostics()
    if command == "close":
        env.shutdown()
        return None
    raise ValueError(f"unsupported subproc env command: {command}")


def _subproc_env_worker(
    connection: Connection,
    *,
    config: PolicyDogfightEnvConfig,
    reward_mode: str,
) -> None:
    env: PolicyDogfightEnv | None = None
    reward_runtime: WorkerRewardStateMachine | None = None
    opponent_reward_runtime: WorkerRewardStateMachine | None = None
    try:
        env = PolicyDogfightEnv(config)
        if reward_mode == "worker":
            reward_runtime = WorkerRewardStateMachine(PolicyRewardComposer(PolicyRewardConfig()))
            opponent_reward_runtime = WorkerRewardStateMachine(
                PolicyRewardComposer(PolicyRewardConfig())
            )
        elif reward_mode != "main":
            raise ValueError(f"unsupported worker reward mode: {reward_mode}")
        while True:
            command, payload = connection.recv()
            if command == "close":
                _worker_handle_command(
                    env,
                    reward_runtime,
                    opponent_reward_runtime,
                    command=command,
                    payload=payload,
                )
                connection.send(("ok", None))
                break
            result = _worker_handle_command(
                env,
                reward_runtime,
                opponent_reward_runtime,
                command=command,
                payload=payload,
            )
            connection.send(("ok", result))
    except EOFError:
        if env is not None:
            env.shutdown()
    except BaseException as exc:  # pragma: no cover - exercised by parent-side error handling
        if env is not None:
            try:
                env.shutdown()
            except Exception:
                pass
        connection.send(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        connection.close()


def _validate_request_count(name: str, requests: Sequence[Any], *, expected: int) -> None:
    if len(requests) != expected:
        raise ValueError(f"{name} expects {expected} requests, got {len(requests)}")


class SubprocPolicyVecEnv:
    """Synchronous multi-process vector environment for the Part 3 policy."""

    def __init__(
        self,
        config: PolicyDogfightEnvConfig,
        *,
        num_envs: int,
        reward_mode: str = "main",
        start_method: str = "spawn",
        request_timeout_seconds: float | None = 300.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._context = mp.get_context(start_method)
        self._parent_connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        self.num_envs = num_envs
        self._config = config
        self.reward_mode = reward_mode
        self.request_timeout_seconds = request_timeout_seconds
        for env_index in range(num_envs):
            parent_conn, child_conn = self._context.Pipe()
            process = self._context.Process(
                target=_subproc_env_worker,
                kwargs={"connection": child_conn, "config": config, "reward_mode": reward_mode},
                name=f"PolicyEnvWorker-{env_index}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._parent_connections.append(parent_conn)
            self._processes.append(process)

    def _worker_status_summary(self, indices: Sequence[int]) -> str:
        parts: list[str] = []
        for index in indices:
            process = self._processes[index]
            parts.append(
                f"{index}:pid={process.pid},alive={process.is_alive()},exitcode={process.exitcode}"
            )
        return "; ".join(parts)

    def _raise_worker_error(self, payload: Any) -> None:
        message = payload.get("message", "unknown worker error") if isinstance(payload, dict) else str(payload)
        tb = payload.get("traceback") if isinstance(payload, dict) else None
        suffix = f"\n{tb}" if tb else ""
        raise RuntimeError(f"subproc env worker failed: {message}{suffix}")

    def _timeout_label(self) -> str:
        if self.request_timeout_seconds is None:
            return "without timeout"
        return f"after {self.request_timeout_seconds:.1f}s"

    def _request_many(self, command: str, requests: Sequence[Any]) -> list[Any]:
        _validate_request_count(command, requests, expected=self.num_envs)
        for connection, request in zip(self._parent_connections, requests, strict=True):
            connection.send((command, request))
        deadline = None
        if self.request_timeout_seconds is not None:
            deadline = time.monotonic() + self.request_timeout_seconds
        results: list[Any] = []
        for index, connection in enumerate(self._parent_connections):
            timeout = None
            if deadline is not None:
                timeout = max(deadline - time.monotonic(), 0.0)
                if not connection.poll(timeout):
                    pending_indices = list(range(index, self.num_envs))
                    raise TimeoutError(
                        f"subproc env command '{command}' timed out {self._timeout_label()}; pending workers: "
                        f"{self._worker_status_summary(pending_indices)}"
                    )
            try:
                status, payload = connection.recv()
            except EOFError as exc:
                raise RuntimeError(
                    f"subproc env worker {index} closed pipe during '{command}'; "
                    f"{self._worker_status_summary([index])}"
                ) from exc
            if status != "ok":
                self._raise_worker_error(payload)
            results.append(payload)
        return results

    def _recv_one(self, index: int, command: str, connection: Connection) -> Any:
        if self.request_timeout_seconds is not None and not connection.poll(self.request_timeout_seconds):
            raise TimeoutError(
                f"subproc env command '{command}' for worker {index} timed out "
                f"{self._timeout_label()}; worker: {self._worker_status_summary([index])}"
            )
        try:
            status, payload = connection.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"subproc env worker {index} closed pipe during '{command}'; "
                f"{self._worker_status_summary([index])}"
            ) from exc
        if status != "ok":
            self._raise_worker_error(payload)
        return payload

    def _request_one(self, index: int, command: str, request: Any) -> Any:
        if index < 0 or index >= self.num_envs:
            raise IndexError(f"env index out of range: {index}")
        connection = self._parent_connections[index]
        connection.send((command, request))
        return self._recv_one(index, command, connection)

    def reset_many(self, requests: Sequence[ResetRequest | None]) -> list[ResetResult]:
        return self._request_many("reset", requests)

    def reset_at(self, index: int, request: ResetRequest | None) -> ResetResult:
        return self._request_one(index, "reset", request)

    def step_many(
        self,
        requests: Sequence[StepRequest],
    ) -> list[StepResult]:
        return self._request_many("step", requests)

    def drain_reward_diagnostics(self) -> list[dict[str, Any]]:
        return self._request_many(
            "drain_reward_diagnostics",
            [None for _ in range(self.num_envs)],
        )

    def drain_episode_reward_diagnostics_at(self, index: int) -> dict[str, Any]:
        return self._request_one(
            index,
            "drain_episode_reward_diagnostics",
            None,
        )

    def close(self) -> None:
        if self._closed:
            return
        for connection in self._parent_connections:
            connection.send(("close", None))
        for connection in self._parent_connections:
            try:
                connection.recv()
            except EOFError:
                pass
            connection.close()
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._closed = True

    def __enter__(self) -> "SubprocPolicyVecEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
