from __future__ import annotations

from typing import Any

import numpy as np

from dfb_reinforcement_learning.policy_contract import (
    POLICY_CONTRACT,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)

from .combat_geometry import (
    compute_attack_geometry,
    finite_vec3,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_6d,
)
from .policy_schema import POLICY_OBSERVATION_SCHEMA


_OBSERVATION_CONFIG = POLICY_CONTRACT["observation"]
_SCALES = _OBSERVATION_CONFIG["scales"]
_HEALTH_SUBSYSTEMS = ("LeftWing", "RightWing", "PitchTail", "YawTail", "Engine")


def _finite_scalar(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{field_name} must be a finite number")
    return scalar


def _required_bool(aircraft: dict[str, Any], key: str, *, role: str) -> bool:
    if key not in aircraft or not isinstance(aircraft[key], bool):
        raise ValueError(f"{role}.{key} must be a bool")
    return aircraft[key]


def _required_scalar(aircraft: dict[str, Any], key: str, *, role: str) -> float:
    if key not in aircraft:
        raise ValueError(f"{role} missing {key}")
    return _finite_scalar(aircraft[key], field_name=f"{role}.{key}")


def _health_state(aircraft: dict[str, Any], *, role: str) -> np.ndarray:
    total_hit_points = _required_scalar(aircraft, "hit_points", role=role)
    max_total_hit_points = float(_SCALES["total_hit_points"])

    raw_subsystems = aircraft.get("subsystems")
    if not isinstance(raw_subsystems, list):
        raise ValueError(f"{role}.subsystems must be an array")
    subsystem_by_name: dict[str, dict[str, Any]] = {}
    for raw_subsystem in raw_subsystems:
        if not isinstance(raw_subsystem, dict) or "name" not in raw_subsystem:
            raise ValueError(f"{role} subsystem must be an object with a name")
        name = str(raw_subsystem["name"])
        if name in subsystem_by_name:
            raise ValueError(f"{role} has duplicate subsystem {name}")
        subsystem_by_name[name] = raw_subsystem

    values = [float(np.clip(total_hit_points / max_total_hit_points, 0.0, 1.0))]
    for name in _HEALTH_SUBSYSTEMS:
        if name not in subsystem_by_name:
            raise ValueError(f"{role} missing subsystem {name}")
        subsystem = subsystem_by_name[name]
        hit_points = _finite_scalar(
            subsystem.get("hit_points"),
            field_name=f"{role}.subsystems.{name}.hit_points",
        )
        max_hit_points = _finite_scalar(
            subsystem.get("max_hit_points"),
            field_name=f"{role}.subsystems.{name}.max_hit_points",
        )
        if max_hit_points <= 0.0:
            raise ValueError(f"{role}.subsystems.{name}.max_hit_points must be positive")
        values.append(float(np.clip(hit_points / max_hit_points, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float32)


def _normalized_duration(*, active: bool, seconds: float, scale: float) -> np.ndarray:
    value = float(np.clip(seconds / scale, 0.0, 4.0)) if active else 0.0
    return np.asarray([value], dtype=np.float32)


def _select_aircraft(world_state: dict[str, Any], role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    aircraft = world_state.get("aircraft")
    if not isinstance(aircraft, list) or len(aircraft) != 2:
        raise ValueError("world_state.aircraft must contain exactly two aircraft")
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    by_role: dict[str, dict[str, Any]] = {}
    for item in aircraft:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ValueError("each aircraft must be an object with a string role")
        item_role = item["role"]
        if item_role in by_role:
            raise ValueError(f"duplicate aircraft role: {item_role}")
        by_role[item_role] = item
    if role not in by_role:
        raise ValueError(f"requested role not found: {role}")
    enemy_roles = [item_role for item_role in by_role if item_role != role]
    if len(enemy_roles) != 1:
        raise ValueError(f"unable to resolve enemy for role: {role}")
    return by_role[role], by_role[enemy_roles[0]]


def _validate_aircraft_state(aircraft: dict[str, Any], *, role: str) -> None:
    finite_vec3(aircraft.get("position"), field_name=f"{role}.position")
    finite_vec3(aircraft.get("linear_velocity"), field_name=f"{role}.linear_velocity")
    finite_vec3(aircraft.get("angular_velocity_deg"), field_name=f"{role}.angular_velocity_deg")
    quaternion_xyzw_to_rotation_matrix(
        aircraft.get("orientation_quat"),
        field_name=f"{role}.orientation_quat",
    )
    _health_state(aircraft, role=role)
    for key in (
        "throttle",
        "stall_factor",
        "gun_heat",
        "repair_elapsed_seconds",
        "out_of_bounds_seconds",
    ):
        _required_scalar(aircraft, key, role=role)
    for key in ("brake", "gun_overheated", "is_firing", "repairing"):
        _required_bool(aircraft, key, role=role)


class PolicyObservationAdapter:
    schema = POLICY_OBSERVATION_SCHEMA

    def build(
        self,
        world_state: dict[str, Any],
        role: str,
        *,
        episode_start_sim_time_seconds: float,
    ) -> dict[str, Any]:
        if not isinstance(world_state, dict):
            raise ValueError("world_state must be an object")
        ego, enemy = _select_aircraft(world_state, role)
        _validate_aircraft_state(ego, role=role)
        enemy_role = str(enemy["role"])
        _validate_aircraft_state(enemy, role=enemy_role)

        sim_time = _finite_scalar(world_state.get("sim_time_seconds"), field_name="sim_time_seconds")
        episode_start = _finite_scalar(
            episode_start_sim_time_seconds,
            field_name="episode_start_sim_time_seconds",
        )
        arena = world_state.get("arena")
        if not isinstance(arena, dict):
            raise ValueError("world_state.arena must be an object")
        arena_radius = _finite_scalar(arena.get("arena_radius"), field_name="arena.arena_radius")
        if arena_radius <= 0.0:
            raise ValueError("arena.arena_radius must be positive")

        ego_position = finite_vec3(ego["position"], field_name=f"{role}.position")
        enemy_position = finite_vec3(enemy["position"], field_name=f"{enemy_role}.position")
        ego_velocity = finite_vec3(ego["linear_velocity"], field_name=f"{role}.linear_velocity")
        enemy_velocity = finite_vec3(
            enemy["linear_velocity"],
            field_name=f"{enemy_role}.linear_velocity",
        )
        ego_angular_velocity = finite_vec3(
            ego["angular_velocity_deg"],
            field_name=f"{role}.angular_velocity_deg",
        )
        enemy_angular_velocity = finite_vec3(
            enemy["angular_velocity_deg"],
            field_name=f"{enemy_role}.angular_velocity_deg",
        )
        ego_rotation = quaternion_xyzw_to_rotation_matrix(
            ego["orientation_quat"],
            field_name=f"{role}.orientation_quat",
        )
        enemy_rotation = quaternion_xyzw_to_rotation_matrix(
            enemy["orientation_quat"],
            field_name=f"{enemy_role}.orientation_quat",
        )

        self_attack = compute_attack_geometry(attacker_state=ego, defender_state=enemy)
        enemy_attack = compute_attack_geometry(attacker_state=enemy, defender_state=ego)

        position_scale = float(_SCALES["relative_position_m"])
        world_position_scale = float(_SCALES["world_position_m"])
        velocity_scale = float(_SCALES["linear_velocity_mps"])
        angular_velocity_scale = float(_SCALES["angular_velocity_rad_s"])
        gun_heat_scale = float(_SCALES["gun_heat"])
        repair_scale = float(_SCALES["repair_seconds"])
        out_of_bounds_scale = float(_SCALES["out_of_bounds_seconds"])
        episode_scale = float(_SCALES["episode_seconds"])

        enemy_oob_seconds = _required_scalar(enemy, "out_of_bounds_seconds", role=enemy_role)
        self_oob_seconds = _required_scalar(ego, "out_of_bounds_seconds", role=role)
        enemy_oob_active = (
            enemy_oob_seconds > 0.0
            or float(np.linalg.norm(enemy_position[[0, 2]])) >= arena_radius
        )
        self_oob_active = (
            self_oob_seconds > 0.0
            or float(np.linalg.norm(ego_position[[0, 2]])) >= arena_radius
        )

        components: dict[str, np.ndarray] = {
            "enemy_relative_position_body": (
                ego_rotation.T @ (enemy_position - ego_position) / position_scale
            ).astype(np.float32),
            "enemy_relative_orientation_6d": rotation_matrix_to_6d(
                ego_rotation.T @ enemy_rotation
            ),
            "enemy_linear_velocity_body": (
                enemy_rotation.T @ enemy_velocity / velocity_scale
            ).astype(np.float32),
            "enemy_angular_velocity_body": (
                np.deg2rad(enemy_angular_velocity) / angular_velocity_scale
            ).astype(np.float32),
            "enemy_health_state_norm": _health_state(enemy, role=enemy_role),
            "enemy_throttle_norm": np.asarray(
                [np.clip(_required_scalar(enemy, "throttle", role=enemy_role), 0.0, 1.0)],
                dtype=np.float32,
            ),
            "enemy_brake_active": np.asarray(
                [float(_required_bool(enemy, "brake", role=enemy_role))],
                dtype=np.float32,
            ),
            "enemy_stall_factor": np.asarray(
                [np.clip(_required_scalar(enemy, "stall_factor", role=enemy_role), 0.0, 1.0)],
                dtype=np.float32,
            ),
            "enemy_gun_overheated": np.asarray(
                [float(_required_bool(enemy, "gun_overheated", role=enemy_role))],
                dtype=np.float32,
            ),
            "enemy_gun_heat_norm": np.asarray(
                [np.clip(_required_scalar(enemy, "gun_heat", role=enemy_role) / gun_heat_scale, 0.0, 4.0)],
                dtype=np.float32,
            ),
            "enemy_fire_gun_active": np.asarray(
                [float(_required_bool(enemy, "is_firing", role=enemy_role))],
                dtype=np.float32,
            ),
            "enemy_repair_active": np.asarray(
                [float(_required_bool(enemy, "repairing", role=enemy_role))],
                dtype=np.float32,
            ),
            "enemy_repair_seconds_norm": _normalized_duration(
                active=_required_bool(enemy, "repairing", role=enemy_role),
                seconds=_required_scalar(enemy, "repair_elapsed_seconds", role=enemy_role),
                scale=repair_scale,
            ),
            "enemy_out_of_bounds_active": np.asarray(
                [float(enemy_oob_active)],
                dtype=np.float32,
            ),
            "enemy_out_of_bounds_seconds_norm": _normalized_duration(
                active=enemy_oob_active,
                seconds=enemy_oob_seconds,
                scale=out_of_bounds_scale,
            ),
            "enemy_tracking_quality": np.asarray([enemy_attack.tracking_quality], dtype=np.float32),
            "enemy_tail_hold_score": np.asarray([enemy_attack.tail_hold_score], dtype=np.float32),
            "enemy_shot_feasibility": np.asarray([enemy_attack.shot_feasibility], dtype=np.float32),
            "episode_time_norm": np.asarray(
                [np.clip(max(sim_time - episode_start, 0.0) / episode_scale, 0.0, 4.0)],
                dtype=np.float32,
            ),
            "self_position_world_norm": (ego_position / world_position_scale).astype(np.float32),
            "self_orientation_world_6d": rotation_matrix_to_6d(ego_rotation),
            "self_throttle_norm": np.asarray(
                [np.clip(_required_scalar(ego, "throttle", role=role), 0.0, 1.0)],
                dtype=np.float32,
            ),
            "self_brake_active": np.asarray(
                [float(_required_bool(ego, "brake", role=role))],
                dtype=np.float32,
            ),
            "self_stall_factor": np.asarray(
                [np.clip(_required_scalar(ego, "stall_factor", role=role), 0.0, 1.0)],
                dtype=np.float32,
            ),
            "self_linear_velocity_body": (
                ego_rotation.T @ ego_velocity / velocity_scale
            ).astype(np.float32),
            "self_angular_velocity_body": (
                np.deg2rad(ego_angular_velocity) / angular_velocity_scale
            ).astype(np.float32),
            "self_health_state_norm": _health_state(ego, role=role),
            "self_gun_overheated": np.asarray(
                [float(_required_bool(ego, "gun_overheated", role=role))],
                dtype=np.float32,
            ),
            "self_gun_heat_norm": np.asarray(
                [np.clip(_required_scalar(ego, "gun_heat", role=role) / gun_heat_scale, 0.0, 4.0)],
                dtype=np.float32,
            ),
            "self_fire_gun_active": np.asarray(
                [float(_required_bool(ego, "is_firing", role=role))],
                dtype=np.float32,
            ),
            "self_repair_active": np.asarray(
                [float(_required_bool(ego, "repairing", role=role))],
                dtype=np.float32,
            ),
            "self_repair_seconds_norm": _normalized_duration(
                active=_required_bool(ego, "repairing", role=role),
                seconds=_required_scalar(ego, "repair_elapsed_seconds", role=role),
                scale=repair_scale,
            ),
            "self_out_of_bounds_active": np.asarray(
                [float(self_oob_active)],
                dtype=np.float32,
            ),
            "self_out_of_bounds_seconds_norm": _normalized_duration(
                active=self_oob_active,
                seconds=self_oob_seconds,
                scale=out_of_bounds_scale,
            ),
            "self_tracking_quality": np.asarray([self_attack.tracking_quality], dtype=np.float32),
            "self_tail_hold_score": np.asarray([self_attack.tail_hold_score], dtype=np.float32),
            "self_shot_feasibility": np.asarray([self_attack.shot_feasibility], dtype=np.float32),
        }

        vector = np.empty((self.schema.dim,), dtype=np.float32)
        for field in self.schema.fields:
            value = np.asarray(components[field.name], dtype=np.float32)
            if value.shape != (field.size,):
                raise ValueError(
                    f"field {field.name} expected shape {(field.size,)}, got {value.shape}"
                )
            vector[field.value_slice] = value
        if not np.isfinite(vector).all():
            raise ValueError("observation contains non-finite values")

        return {
            "policy_contract_id": POLICY_CONTRACT_ID,
            "contract_sha256": POLICY_CONTRACT_SHA256,
            "schema_id": self.schema.schema_id,
            "role": role,
            "tick": int(world_state["tick"]),
            "sim_time_seconds": sim_time,
            "episode_start_sim_time_seconds": episode_start,
            "scene_name": str(world_state.get("scene_name", "unknown")),
            "components": components,
            "vector": vector,
        }
