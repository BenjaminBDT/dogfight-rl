from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from dfb_game_py import Environment

from dfb_reinforcement_learning.actions import ActionAdapter, HybridAction
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter


@dataclass(frozen=True)
class PolicyDogfightEnvConfig:
    project_root: str | None = None
    scene_name: str = "open_head_on_200m"
    scene_path: str | None = None
    ego_role: str = "fighter1"
    ego_mode: str = "external"
    opponent_mode: str = "built_in_ai"
    seed: int | None = None
    ticks_per_step: int = 1


def _agent_modes_for_ego_role(
    ego_role: str,
    opponent_mode: str = "built_in_ai",
    *,
    ego_mode: str = "external",
) -> tuple[str, str]:
    if opponent_mode == "model":
        opponent_mode = "external"
    if ego_role == "fighter1":
        return ego_mode, opponent_mode
    if ego_role == "fighter2":
        return opponent_mode, ego_mode
    raise ValueError(f"unsupported ego role: {ego_role}")


def _extract_world_state(observation_bundle: dict[str, Any]) -> dict[str, Any]:
    if "state" not in observation_bundle:
        raise ValueError("observation bundle missing 'state'")
    state = observation_bundle["state"]
    if not isinstance(state, dict):
        raise ValueError("observation bundle field 'state' must be an object")
    return state


def _extract_episode_info(state: dict[str, Any], *, ego_role: str) -> dict[str, Any]:
    aircraft = state.get("aircraft", [])
    arena = state.get("arena", {})
    enemy_role = "fighter2" if ego_role == "fighter1" else "fighter1"
    positions_by_role: dict[str, np.ndarray] = {}
    speeds_by_role: dict[str, float] = {}
    aircraft_by_role = {
        item["role"]: {
            "destroyed": bool(item.get("destroyed", False)),
            "out_of_bounds_seconds": float(item.get("out_of_bounds_seconds", 0.0)),
            "brake": bool(item.get("brake", False)),
            "gun_heat": float(item.get("gun_heat", 0.0)),
            "gun_overheated": bool(item.get("gun_overheated", False)),
            "repairing": bool(item.get("repairing", False)),
            "repair_elapsed_seconds": float(item.get("repair_elapsed_seconds", 0.0)),
            "repair_progress": float(item.get("repair_progress", 0.0)),
            "stall_factor": float(item.get("stall_factor", 0.0)),
            "position": list(item.get("position", [0.0, 0.0, 0.0])),
            "orientation_quat": list(item.get("orientation_quat", [0.0, 0.0, 0.0, 1.0])),
            "linear_velocity": list(item.get("linear_velocity", [0.0, 0.0, 0.0])),
            "forward": list(item.get("forward", [0.0, 0.0, 1.0])),
            "subsystems": list(item.get("subsystems", [])),
            "velocity_turn_rate_rad_s": item.get("velocity_turn_rate_rad_s"),
            "pullup_turn_radius_m": item.get("pullup_turn_radius_m"),
            "max_level_speed_mps": item.get("max_level_speed_mps"),
            "time_to_ground_impact_s": item.get("time_to_ground_impact_s"),
            "time_to_ceiling_impact_s": item.get("time_to_ceiling_impact_s"),
            "time_to_horizontal_boundary_impact_s": item.get("time_to_horizontal_boundary_impact_s"),
            "time_to_reenter_arena_s": item.get("time_to_reenter_arena_s"),
            "speed": float(np.linalg.norm(np.asarray(item.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32))),
        }
        for item in aircraft
    }
    for item in aircraft:
        role = item["role"]
        positions_by_role[role] = np.asarray(item.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        speeds_by_role[role] = aircraft_by_role[role]["speed"]
    target_distance = None
    if "fighter1" in positions_by_role and "fighter2" in positions_by_role:
        target_distance = float(np.linalg.norm(positions_by_role["fighter2"] - positions_by_role["fighter1"]))
    return {
        "tick": int(state["tick"]),
        "sim_time_seconds": float(state["sim_time_seconds"]),
        "scene_name": str(state.get("scene_name", "unknown")),
        "ego_role": ego_role,
        "enemy_role": enemy_role,
        "events_since_last_step": list(state.get("events_since_last_step", [])),
        "arena": {
            "ground_height": float(arena.get("ground_height", 0.0)),
            "arena_radius": float(arena.get("arena_radius", 0.0)),
            "flight_ceiling_height": float(arena.get("flight_ceiling_height", 0.0)),
            "ceiling_falloff_range": float(arena.get("ceiling_falloff_range", 0.0)),
        },
        "aircraft_by_role": aircraft_by_role,
        "target_distance": target_distance,
        "positions_by_role": {role: value.tolist() for role, value in positions_by_role.items()},
        "speeds_by_role": speeds_by_role,
    }


class PolicyDogfightEnv:
    """Gym-like wrapper around dfb_game using the active Part 3 policy contract."""

    def __init__(
        self,
        config: PolicyDogfightEnvConfig,
        *,
        obs_adapter: PolicyObservationAdapter | None = None,
    ) -> None:
        self.config = config
        self.obs_adapter = obs_adapter or PolicyObservationAdapter()
        fighter1_mode, fighter2_mode = _agent_modes_for_ego_role(
            config.ego_role,
            config.opponent_mode,
            ego_mode=config.ego_mode,
        )
        self._env = Environment(
            project_root=config.project_root,
            scene_name=None if config.scene_path else config.scene_name,
            scene_path=config.scene_path,
            seed=config.seed,
            enable_visual=False,
            enable_audio=False,
            audio_window_seconds=0.25,
            ticks_per_step=config.ticks_per_step,
            self_play=False,
            fighter1_mode=fighter1_mode,
            fighter2_mode=fighter2_mode,
        )
        self._last_state: dict[str, Any] | None = None
        self._episode_start_sim_time_seconds: float | None = None

    @property
    def ego_role(self) -> str:
        return self.config.ego_role

    @property
    def enemy_role(self) -> str:
        return "fighter2" if self.config.ego_role == "fighter1" else "fighter1"

    @property
    def episode_start_sim_time_seconds(self) -> float:
        if self._episode_start_sim_time_seconds is None:
            raise RuntimeError("environment has not been reset yet")
        return self._episode_start_sim_time_seconds

    def reset(
        self,
        *,
        seed: int | None = None,
        scene_name: str | None = None,
        scene_path: str | None = None,
        opponent_mode: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        effective_opponent_mode = self.config.opponent_mode if opponent_mode is None else opponent_mode
        observation_bundle = json.loads(
            self._env.reset_json(
                scene_name=None if scene_path else (scene_name or self.config.scene_name),
                scene_path=scene_path or self.config.scene_path,
                seed=seed if seed is not None else self.config.seed,
                enable_visual=False,
                enable_audio=False,
                audio_window_seconds=0.25,
                ticks_per_step=self.config.ticks_per_step,
                self_play=False,
                fighter1_mode=_agent_modes_for_ego_role(
                    self.config.ego_role,
                    effective_opponent_mode,
                    ego_mode=self.config.ego_mode,
                )[0],
                fighter2_mode=_agent_modes_for_ego_role(
                    self.config.ego_role,
                    effective_opponent_mode,
                    ego_mode=self.config.ego_mode,
                )[1],
            )
        )
        state = _extract_world_state(observation_bundle)
        self._last_state = state
        self._episode_start_sim_time_seconds = float(state["sim_time_seconds"])
        obs = self.obs_adapter.build(
            state,
            self.config.ego_role,
            episode_start_sim_time_seconds=self.episode_start_sim_time_seconds,
        )["vector"]
        info = _extract_episode_info(state, ego_role=self.config.ego_role)
        info["episode_start_sim_time_seconds"] = self.episode_start_sim_time_seconds
        return obs, info

    def step(
        self,
        action: HybridAction,
        *,
        opponent_action: HybridAction | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        env_action = ActionAdapter.to_environment_action(action)
        payload = [{"role": self.config.ego_role, "action": json.loads(env_action.json())}]
        if opponent_action is not None:
            opponent_env_action = ActionAdapter.to_environment_action(opponent_action)
            payload.append({"role": self.enemy_role, "action": json.loads(opponent_env_action.json())})
        step_result = json.loads(self._env.step_targeted_json(json.dumps(payload)))
        observation_bundle = step_result["observation"]
        state = _extract_world_state(observation_bundle)
        self._last_state = state
        obs = self.obs_adapter.build(
            state,
            self.config.ego_role,
            episode_start_sim_time_seconds=self.episode_start_sim_time_seconds,
        )["vector"]
        reward_raw = step_result.get("reward")
        reward = 0.0 if reward_raw is None else float(reward_raw)
        terminated = bool(step_result.get("terminated", False))
        truncated = bool(step_result.get("truncated", False))
        info = _extract_episode_info(state, ego_role=self.config.ego_role)
        info["episode_start_sim_time_seconds"] = self.episode_start_sim_time_seconds
        info["winner"] = step_result.get("info", {}).get("winner")
        info["step_events"] = step_result.get("info", {}).get("events", [])
        return obs, reward, terminated, truncated, info

    def step_arrays(
        self,
        continuous: np.ndarray,
        binary: np.ndarray,
        *,
        binary_threshold: float = 0.5,
        opponent_continuous: np.ndarray | None = None,
        opponent_binary: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = ActionAdapter.from_arrays(continuous, binary, binary_threshold=binary_threshold)
        opponent_action = None
        if opponent_continuous is not None and opponent_binary is not None:
            opponent_action = ActionAdapter.from_arrays(
                opponent_continuous,
                opponent_binary,
                binary_threshold=binary_threshold,
            )
        return self.step(action, opponent_action=opponent_action)

    def teacher_action_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        payload = json.loads(self._env.teacher_action_json(self.config.ego_role))
        action = HybridAction(
            throttle=float(payload["throttle"]),
            pitch=float(payload["pitch"]),
            roll=float(payload["roll"]),
            yaw=float(payload["yaw"]),
            brake=bool(payload["brake"]),
            fire_gun=bool(payload["fire_gun"]),
            repair=bool(payload["repair"]),
        )
        return action.continuous_array(), action.binary_array()

    def latest_state(self) -> dict[str, Any]:
        if self._last_state is None:
            raise RuntimeError("environment has not been reset yet")
        return self._last_state

    def observation_for_role(self, role: str) -> np.ndarray:
        return self.obs_adapter.build(
            self.latest_state(),
            role,
            episode_start_sim_time_seconds=self.episode_start_sim_time_seconds,
        )["vector"]

    def shutdown(self) -> None:
        self._env.shutdown()

    def __enter__(self) -> "PolicyDogfightEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown()
