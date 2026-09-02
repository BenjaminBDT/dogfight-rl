from __future__ import annotations

import numpy as np
from multiprocessing import Pipe
from types import SimpleNamespace

from dfb_reinforcement_learning.envs.policy_dogfight_env import (
    _agent_modes_for_ego_role,
    _extract_episode_info,
    _extract_world_state,
)
from dfb_reinforcement_learning.envs.subproc_policy_vec_env import (
    ResetRequest,
    ResetResult,
    StepRequest,
    StepResult,
    SubprocPolicyVecEnv,
    WorkerRewardStateMachine,
    _validate_request_count,
    _worker_handle_command,
)


def test_agent_modes_for_ego_role() -> None:
    assert _agent_modes_for_ego_role("fighter1") == ("external", "built_in_ai")
    assert _agent_modes_for_ego_role("fighter2") == ("built_in_ai", "external")
    assert _agent_modes_for_ego_role("fighter1", "built_in_ai_precise") == (
        "external",
        "built_in_ai_precise",
    )
    assert _agent_modes_for_ego_role("fighter2", "built_in_ai_imperfect") == (
        "built_in_ai_imperfect",
        "external",
    )
    assert _agent_modes_for_ego_role(
        "fighter1",
        "built_in_ai_imperfect",
        ego_mode="built_in_ai_precise",
    ) == ("built_in_ai_precise", "built_in_ai_imperfect")
    assert _agent_modes_for_ego_role(
        "fighter2",
        "built_in_ai_imperfect",
        ego_mode="built_in_ai_precise",
    ) == ("built_in_ai_imperfect", "built_in_ai_precise")


def test_extract_world_state() -> None:
    state = {"tick": 0, "aircraft": [], "sim_time_seconds": 0.0}
    bundle = {"state": state}
    assert _extract_world_state(bundle) is state


def test_extract_episode_info_includes_target_distance() -> None:
    state = {
        "tick": 7,
        "sim_time_seconds": 0.0,
        "aircraft": [
            {
                "role": "fighter1",
                "position": [0.0, 0.0, 0.0],
                "linear_velocity": [1.0, 2.0, 2.0],
                "stall_factor": 0.25,
                "destroyed": False,
                "out_of_bounds_seconds": 0.0,
                "gun_heat": 0.25,
                "gun_overheated": False,
                "repairing": False,
                "velocity_turn_rate_rad_s": 0.8,
                "pullup_turn_radius_m": 75.0,
                "max_level_speed_mps": 80.0,
                "time_to_ground_impact_s": 1.5,
                "time_to_ceiling_impact_s": None,
                "time_to_horizontal_boundary_impact_s": 2.0,
                "time_to_reenter_arena_s": None,
            },
            {
                "role": "fighter2",
                "position": [3.0, 4.0, 0.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "destroyed": False,
                "out_of_bounds_seconds": 0.0,
                "gun_heat": 0.0,
                "gun_overheated": False,
                "repairing": False,
            },
        ],
        "events_since_last_step": [],
    }
    info = _extract_episode_info(state, ego_role="fighter1")
    assert info["tick"] == 7
    assert info["target_distance"] == 5.0
    assert info["speeds_by_role"]["fighter1"] == 3.0
    assert info["aircraft_by_role"]["fighter1"]["stall_factor"] == 0.25
    assert info["aircraft_by_role"]["fighter1"]["velocity_turn_rate_rad_s"] == 0.8
    assert info["aircraft_by_role"]["fighter1"]["pullup_turn_radius_m"] == 75.0
    assert info["aircraft_by_role"]["fighter1"]["max_level_speed_mps"] == 80.0
    assert info["aircraft_by_role"]["fighter1"]["time_to_ground_impact_s"] == 1.5
    assert info["aircraft_by_role"]["fighter1"]["time_to_horizontal_boundary_impact_s"] == 2.0
    assert info["ego_role"] == "fighter1"
    assert info["enemy_role"] == "fighter2"


def test_numpy_import_sanity() -> None:
    assert float(np.zeros((1,), dtype=np.float32)[0]) == 0.0


class _FakeEnv:
    def __init__(self) -> None:
        self.reset_calls: list[dict[str, object | None]] = []
        self.step_calls: list[dict[str, object]] = []
        self.shutdown_calls = 0
        self.enemy_role = "fighter2"
        self.config = SimpleNamespace(ego_role="fighter1")

    def reset(
        self,
        *,
        seed: int | None = None,
        scene_name: str | None = None,
        scene_path: str | None = None,
        opponent_mode: str | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        self.reset_calls.append(
            {
                "seed": seed,
                "scene_name": scene_name,
                "scene_path": scene_path,
                "opponent_mode": opponent_mode,
            }
        )
        return np.zeros((4,), dtype=np.float32), {"seed": seed}

    def step_arrays(
        self,
        continuous: np.ndarray,
        binary: np.ndarray,
        *,
        binary_threshold: float = 0.5,
        opponent_continuous: np.ndarray | None = None,
        opponent_binary: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_calls.append(
            {
                "continuous": continuous,
                "binary": binary,
                "binary_threshold": binary_threshold,
                "opponent_continuous": opponent_continuous,
                "opponent_binary": opponent_binary,
            }
        )
        return np.ones((4,), dtype=np.float32), 1.0, False, False, {"ok": True}

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def latest_state(self) -> dict[str, object]:
        return {"fake": True}

    def observation_for_role(self, role: str) -> np.ndarray:
        assert role == self.enemy_role
        return np.full((4,), 2.0, dtype=np.float32)


class _FakeRewardRuntime:
    def __init__(self) -> None:
        self.reset_calls: list[dict[str, object] | None] = []
        self.compute_calls: list[dict[str, object]] = []
        self.advance_calls: list[dict[str, object]] = []

    def reset(self, *, initial_info=None) -> None:
        self.reset_calls.append(initial_info)

    def compute(self, *, current_info, current_obs, action_cont, action_bin):
        self.compute_calls.append(
            {
                "current_info": current_info,
                "current_obs_shape": tuple(np.asarray(current_obs).shape),
                "action_cont": np.asarray(action_cont, dtype=np.float32).copy(),
                "action_bin": np.asarray(action_bin, dtype=np.float32).copy(),
            }
        )
        return type("Breakdown", (), {"total": 7.5})()

    def advance(self, *, info, action_cont, action_bin, terminated_or_truncated) -> None:
        self.advance_calls.append(
            {
                "info": info,
                "action_cont": np.asarray(action_cont, dtype=np.float32).copy(),
                "action_bin": np.asarray(action_bin, dtype=np.float32).copy(),
                "terminated_or_truncated": terminated_or_truncated,
            }
        )


def test_worker_handle_command_reset_dispatches_to_env() -> None:
    env = _FakeEnv()
    reward_runtime = _FakeRewardRuntime()
    result = _worker_handle_command(
        env,
        reward_runtime,
        command="reset",
        payload=ResetRequest(seed=123, scene_name="scene_a", opponent_mode="external", include_state=True),
    )
    assert isinstance(result, ResetResult)
    assert result.obs.shape == (4,)
    assert result.info["seed"] == 123
    assert result.state == {"fake": True}
    assert env.reset_calls == [
        {
            "seed": 123,
            "scene_name": "scene_a",
            "scene_path": None,
            "opponent_mode": "external",
        }
    ]
    assert reward_runtime.reset_calls == [{"seed": 123}]


def test_worker_handle_command_step_dispatches_to_env() -> None:
    env = _FakeEnv()
    reward_runtime = _FakeRewardRuntime()
    request = StepRequest(
        continuous=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        binary=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        binary_threshold=0.25,
        opponent_continuous=np.array([-0.1, -0.2, -0.3, -0.4], dtype=np.float32),
        opponent_binary=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        include_state=True,
    )
    result = _worker_handle_command(env, reward_runtime, command="step", payload=request)
    assert isinstance(result, StepResult)
    assert result.obs.shape == (4,)
    assert result.reward == 7.5
    assert not result.terminated
    assert not result.truncated
    assert result.info["ok"] is True
    assert result.state == {"fake": True}
    assert len(env.step_calls) == 1
    assert env.step_calls[0]["binary_threshold"] == 0.25
    assert reward_runtime.compute_calls[0]["current_info"] == {"ok": True}
    assert reward_runtime.advance_calls[0]["terminated_or_truncated"] is False


def test_worker_handle_command_computes_self_play_opponent_payload() -> None:
    env = _FakeEnv()
    ego_reward_runtime = _FakeRewardRuntime()
    opponent_reward_runtime = _FakeRewardRuntime()
    reset_result = _worker_handle_command(
        env,
        ego_reward_runtime,
        opponent_reward_runtime,
        command="reset",
        payload=ResetRequest(seed=123, opponent_mode="model", self_play=True),
    )
    np.testing.assert_allclose(reset_result.opponent_obs, 2.0)
    assert opponent_reward_runtime.reset_calls == [
        {"seed": 123, "ego_role": "fighter2", "enemy_role": "fighter1"}
    ]

    request = StepRequest(
        continuous=np.zeros((4,), dtype=np.float32),
        binary=np.zeros((3,), dtype=np.float32),
        opponent_continuous=np.ones((4,), dtype=np.float32),
        opponent_binary=np.ones((3,), dtype=np.float32),
        self_play=True,
    )
    step_result = _worker_handle_command(
        env,
        ego_reward_runtime,
        opponent_reward_runtime,
        command="step",
        payload=request,
    )
    np.testing.assert_allclose(step_result.opponent_obs, 2.0)
    assert step_result.opponent_reward == 7.5
    assert opponent_reward_runtime.compute_calls[0]["current_info"] == {
        "ok": True,
        "ego_role": "fighter2",
        "enemy_role": "fighter1",
    }


def test_worker_handle_command_close_dispatches_shutdown() -> None:
    env = _FakeEnv()
    result = _worker_handle_command(env, None, command="close", payload=None)
    assert result is None
    assert env.shutdown_calls == 1


def test_worker_handle_command_can_skip_state_return() -> None:
    env = _FakeEnv()
    reset_result = _worker_handle_command(
        env,
        None,
        command="reset",
        payload=ResetRequest(seed=1, include_state=False),
    )
    assert isinstance(reset_result, ResetResult)
    assert reset_result.state is None

    step_result = _worker_handle_command(
        env,
        None,
        command="step",
        payload=StepRequest(
            continuous=np.zeros((4,), dtype=np.float32),
            binary=np.zeros((3,), dtype=np.float32),
            include_state=False,
        ),
    )
    assert isinstance(step_result, StepResult)
    assert step_result.state is None


def test_validate_request_count_rejects_length_mismatch() -> None:
    try:
        _validate_request_count("step", [object()], expected=2)
    except ValueError as exc:
        assert "expects 2 requests" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class _FakeProcess:
    pid = 12345
    exitcode = None

    def is_alive(self) -> bool:
        return True


def test_subproc_vec_env_request_many_times_out_instead_of_blocking() -> None:
    parent_conn, child_conn = Pipe()
    vec = SubprocPolicyVecEnv.__new__(SubprocPolicyVecEnv)
    vec.num_envs = 1
    vec._parent_connections = [parent_conn]
    vec._processes = [_FakeProcess()]
    vec.request_timeout_seconds = 0.01
    try:
        try:
            vec._request_many(
                "step",
                [StepRequest(np.zeros((4,), dtype=np.float32), np.zeros((3,), dtype=np.float32))],
            )
        except TimeoutError as exc:
            assert "timed out" in str(exc)
            assert "pending workers" in str(exc)
            assert "pid=12345" in str(exc)
        else:
            raise AssertionError("expected TimeoutError")
    finally:
        parent_conn.close()
        child_conn.close()


class _FakeRewardComposer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reward_history_frame_length(self) -> int:
        return 2

    def compute(
        self,
        *,
        previous_info,
        previous_previous_action_cont,
        previous_action_cont,
        reward_history,
        current_info,
        current_obs,
        action_cont,
        action_bin,
    ) -> dict[str, object]:
        payload = {
            "previous_info": previous_info,
            "previous_previous_action_cont": previous_previous_action_cont,
            "previous_action_cont": previous_action_cont,
            "reward_history_length": len(reward_history["frames"]),
            "current_info": current_info,
            "current_obs_shape": tuple(np.asarray(current_obs).shape),
            "action_cont": np.asarray(action_cont, dtype=np.float32).copy(),
            "action_bin": np.asarray(action_bin, dtype=np.float32).copy(),
        }
        self.calls.append(payload)
        return payload

    def build_attack_history_cache(self, info: dict[str, object]) -> dict[str, object]:
        return {"cached_attack": info["tick"]}

    def build_boundary_phi_cache(self, info: dict[str, object]) -> dict[str, float]:
        return {"ground": float(info["tick"])}


def test_worker_reward_state_machine_tracks_history_and_previous_actions() -> None:
    composer = _FakeRewardComposer()
    runtime = WorkerRewardStateMachine(composer)
    initial_info = {"tick": 0}
    runtime.reset(initial_info=initial_info)
    first_info = {"tick": 1}
    first_action_cont = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    first_action_bin = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first_breakdown = runtime.compute(
        current_info=first_info,
        current_obs=np.zeros((5,), dtype=np.float32),
        action_cont=first_action_cont,
        action_bin=first_action_bin,
    )
    assert first_breakdown["previous_info"] is initial_info
    assert first_breakdown["reward_history_length"] == 0

    runtime.advance(
        info=first_info,
        action_cont=first_action_cont,
        action_bin=first_action_bin,
        terminated_or_truncated=False,
    )
    assert runtime.previous_info is first_info
    assert runtime.previous_previous_action_cont is None
    assert np.allclose(runtime.previous_action_cont, first_action_cont)
    assert len(runtime.reward_history_frames) == 1
    assert runtime.reward_history_frames[0]["attack_history_cache"] == {"cached_attack": 1}
    assert runtime.reward_history_frames[0]["boundary_phi_cache"] == {"ground": 1.0}

    second_info = {"tick": 2}
    second_action_cont = np.asarray([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    second_breakdown = runtime.compute(
        current_info=second_info,
        current_obs=np.ones((5,), dtype=np.float32),
        action_cont=second_action_cont,
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert second_breakdown["previous_info"] is first_info
    assert second_breakdown["reward_history_length"] == 1
    assert np.allclose(second_breakdown["previous_action_cont"], first_action_cont)

    runtime.advance(
        info=second_info,
        action_cont=second_action_cont,
        action_bin=np.zeros((3,), dtype=np.float32),
        terminated_or_truncated=False,
    )
    assert np.allclose(runtime.previous_previous_action_cont, first_action_cont)
    assert np.allclose(runtime.previous_action_cont, second_action_cont)
    assert len(runtime.reward_history_frames) == 2
    diagnostics = runtime.drain_reward_diagnostics()
    assert diagnostics["sample_count"] == 2
    assert runtime.drain_reward_diagnostics()["sample_count"] == 0
    episode_diagnostics = runtime.drain_episode_reward_diagnostics()
    assert episode_diagnostics["sample_count"] == 2
    assert runtime.drain_episode_reward_diagnostics()["sample_count"] == 0


def test_worker_handle_command_drains_reward_diagnostics() -> None:
    env = _FakeEnv()
    runtime = WorkerRewardStateMachine(_FakeRewardComposer())
    runtime.compute(
        current_info={"tick": 1},
        current_obs=np.zeros((5,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )

    payload = _worker_handle_command(
        env,
        runtime,
        command="drain_reward_diagnostics",
        payload=None,
    )

    assert payload["sample_count"] == 1
    assert runtime.drain_reward_diagnostics()["sample_count"] == 0


def test_worker_handle_command_drains_episode_reward_diagnostics() -> None:
    env = _FakeEnv()
    runtime = WorkerRewardStateMachine(_FakeRewardComposer())
    runtime.compute(
        current_info={"tick": 1},
        current_obs=np.zeros((5,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )

    payload = _worker_handle_command(
        env,
        runtime,
        command="drain_episode_reward_diagnostics",
        payload=None,
    )

    assert payload["sample_count"] == 1
    assert runtime.drain_episode_reward_diagnostics()["sample_count"] == 0


def test_worker_reward_state_machine_resets_on_terminal_transition() -> None:
    composer = _FakeRewardComposer()
    runtime = WorkerRewardStateMachine(composer)
    runtime.reset(initial_info={"tick": 0})
    runtime.advance(
        info={"tick": 1},
        action_cont=np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
        terminated_or_truncated=False,
    )
    assert runtime.previous_info is not None
    assert len(runtime.reward_history_frames) == 1

    runtime.compute(
        current_info={"tick": 2},
        current_obs=np.zeros((5,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    runtime.advance(
        info={"tick": 2},
        action_cont=np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
        terminated_or_truncated=True,
    )
    assert runtime.previous_info is None
    assert runtime.previous_previous_action_cont is None
    assert runtime.previous_action_cont is None
    assert runtime.reward_history_frames == []
    assert runtime.drain_episode_reward_diagnostics()["sample_count"] == 1
