"""Canonical Part 3 combat geometry shared by observations and rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dfb_reinforcement_learning.policy_contract import POLICY_CONTRACT


_GEOMETRY = POLICY_CONTRACT["geometry"]

PROJECTILE_SPEED_MPS = float(_GEOMETRY["projectile_speed_mps"])
PROJECTILE_MAX_RANGE_M = float(_GEOMETRY["projectile_max_range_m"])
PROJECTILE_MUZZLE_FORWARD_OFFSET_M = float(_GEOMETRY["muzzle_forward_offset_m"])
PROJECTILE_AIRCRAFT_HIT_RADIUS_M = float(_GEOMETRY["projectile_aircraft_hit_radius_m"])
PROJECTILE_SUBSYSTEM_HIT_RADIUS_M = float(_GEOMETRY["projectile_subsystem_hit_radius_m"])
ATTACK_TAU_REFERENCE_SECONDS = float(_GEOMETRY["attack_tau_reference_seconds"])
FIRE_ALIGNMENT_THRESHOLD_COS = float(_GEOMETRY["fire_alignment_threshold_cos"])
SHOT_OUTER_RADIUS_M = float(_GEOMETRY["shot_outer_radius_m"])
SHOT_CORE_RADIUS_M = float(_GEOMETRY["shot_core_radius_m"])
SHOT_OUTER_WEIGHT = float(_GEOMETRY["shot_outer_weight"])
SHOT_CORE_WEIGHT = float(_GEOMETRY["shot_core_weight"])

SHOT_BROADPHASE_EPSILON = 1e-5
AIRCRAFT_COLLISION_BROADPHASE_FINE_THRESHOLD = 0.01

# Retained until the old adapter is removed. It is not part of policy contract v1.
COMBAT_TURN_RADIUS_METERS = 200.0


@dataclass(frozen=True)
class LocalCollisionBox:
    name: str
    center: np.ndarray
    half_extents: np.ndarray
    is_subsystem: bool


@dataclass(frozen=True)
class WorldCollisionBox:
    name: str
    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray
    is_subsystem: bool


@dataclass(frozen=True)
class AttackGeometryMetrics:
    tau_seconds: float
    min_distance_meters: float
    tau_gate: float
    fire_alignment: float
    shot_outer_score: float
    shot_core_score: float
    shot_feasibility: float
    tracking_quality: float
    tail_hold_score: float


def _load_local_collision_boxes() -> tuple[LocalCollisionBox, ...]:
    boxes: list[LocalCollisionBox] = []
    for raw_box in _GEOMETRY["collision_boxes"]:
        boxes.append(
            LocalCollisionBox(
                name=str(raw_box["name"]),
                center=np.asarray(raw_box["center"], dtype=np.float32),
                half_extents=np.asarray(raw_box["half_extents"], dtype=np.float32),
                is_subsystem=bool(raw_box["subsystem"]),
            )
        )
    return tuple(boxes)


POLICY_LOCAL_COLLISION_BOXES = _load_local_collision_boxes()

# Temporary tuple API for old collision/reward code. Values still originate in the contract JSON.
LOCAL_COLLISION_BOXES: tuple[tuple[str, np.ndarray, np.ndarray], ...] = tuple(
    (box.name, box.center, box.half_extents) for box in POLICY_LOCAL_COLLISION_BOXES
)

AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS = max(
    float(np.linalg.norm(box.center) + np.linalg.norm(box.half_extents))
    for box in POLICY_LOCAL_COLLISION_BOXES
)
SHOT_BROADPHASE_RADIUS_METERS = AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS


def finite_vec3(value: Any, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a finite vec3")
    return vector


def quaternion_xyzw_to_rotation_matrix(value: Any, *, field_name: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"{field_name} must be a finite xyzw quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{field_name} quaternion has zero length")
    x, y, z, w = quaternion / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def quat_to_rotation_matrix(quat_xyzw: Any) -> np.ndarray:
    """Compatibility name backed by the strict policy-contract conversion."""
    return quaternion_xyzw_to_rotation_matrix(quat_xyzw, field_name="orientation_quat")


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    return np.concatenate((matrix[:, 0], matrix[:, 1])).astype(np.float32)


def closest_approach(
    *,
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    horizon_seconds: float,
) -> tuple[float, float]:
    if not np.isfinite(horizon_seconds) or horizon_seconds < 0.0:
        raise ValueError("horizon_seconds must be finite and non-negative")
    relative_position = finite_vec3(relative_position, field_name="relative_position")
    relative_velocity = finite_vec3(relative_velocity, field_name="relative_velocity")
    relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
    if relative_speed_sq <= 1e-6:
        tau_star = 0.0
    else:
        tau_star = -float(np.dot(relative_position, relative_velocity)) / relative_speed_sq
    tau_star = float(np.clip(tau_star, 0.0, horizon_seconds))
    closest_delta = relative_position + relative_velocity * tau_star
    return tau_star, float(np.linalg.norm(closest_delta))


def box_support_radius(
    rotation: np.ndarray,
    half_extents: np.ndarray,
    direction: np.ndarray,
) -> float:
    rotation = np.asarray(rotation, dtype=np.float32)
    half_extents = finite_vec3(half_extents, field_name="half_extents")
    direction = finite_vec3(direction, field_name="direction")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return float(np.max(half_extents))
    direction = direction / norm
    return float(
        sum(
            float(half_extents[index]) * abs(float(np.dot(rotation[:, index], direction)))
            for index in range(3)
        )
    )


def _destroyed_subsystem_names(aircraft_state: dict[str, Any]) -> frozenset[str]:
    raw_subsystems = aircraft_state.get("subsystems")
    if not isinstance(raw_subsystems, list):
        raise ValueError("aircraft subsystems must be an array")
    names: set[str] = set()
    destroyed: set[str] = set()
    for entry in raw_subsystems:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("aircraft subsystem must be an object with a name")
        name = str(entry["name"])
        if name in names:
            raise ValueError(f"duplicate aircraft subsystem: {name}")
        names.add(name)
        if str(entry.get("stage", "")) == "Destroyed":
            destroyed.add(name)
    return frozenset(destroyed)


def world_collision_boxes(aircraft_state: dict[str, Any]) -> tuple[WorldCollisionBox, ...]:
    position = finite_vec3(aircraft_state.get("position"), field_name="position")
    rotation = quaternion_xyzw_to_rotation_matrix(
        aircraft_state.get("orientation_quat"),
        field_name="orientation_quat",
    )
    destroyed = _destroyed_subsystem_names(aircraft_state)
    return tuple(
        WorldCollisionBox(
            name=box.name,
            center=(position + rotation @ box.center).astype(np.float32),
            rotation=rotation,
            half_extents=box.half_extents,
            is_subsystem=box.is_subsystem,
        )
        for box in POLICY_LOCAL_COLLISION_BOXES
        if box.name not in destroyed
    )


def active_collision_boxes(
    aircraft_state: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Temporary tuple API used by old reward collision code."""
    return [
        (box.center, box.rotation, box.half_extents)
        for box in world_collision_boxes(aircraft_state)
    ]


def _box_hit_score(
    *,
    box: WorldCollisionBox,
    muzzle_position: np.ndarray,
    relative_velocity: np.ndarray,
    tau_max: float,
    hit_radius: float,
    score_radius: float,
) -> float:
    relative_position = box.center - muzzle_position
    tau_star, center_distance = closest_approach(
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        horizon_seconds=tau_max,
    )
    closest_delta = relative_position + relative_velocity * tau_star
    closest_norm = float(np.linalg.norm(closest_delta))
    if closest_norm <= 1e-6:
        closest_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        closest_direction = closest_delta / closest_norm
    support = box_support_radius(box.rotation, box.half_extents, -closest_direction)
    clearance = max(center_distance - support - hit_radius, 0.0)
    tau_gate = float(np.exp(-tau_star / ATTACK_TAU_REFERENCE_SECONDS))
    return tau_gate * float(np.exp(-((clearance / max(score_radius, hit_radius)) ** 2)))


def projectile_box_hit_scores(
    *,
    muzzle_position: np.ndarray,
    bullet_velocity: np.ndarray,
    defender_state: dict[str, Any],
    dynamic_outer_radius: float = SHOT_OUTER_RADIUS_M,
) -> tuple[float, float]:
    muzzle_position = finite_vec3(muzzle_position, field_name="muzzle_position")
    bullet_velocity = finite_vec3(bullet_velocity, field_name="bullet_velocity")
    defender_velocity = finite_vec3(
        defender_state.get("linear_velocity"),
        field_name="defender.linear_velocity",
    )
    bullet_speed = float(np.linalg.norm(bullet_velocity))
    if bullet_speed <= 1e-6:
        return 0.0, 0.0
    tau_max = PROJECTILE_MAX_RANGE_M / bullet_speed
    relative_velocity = defender_velocity - bullet_velocity

    outer_best = 0.0
    core_best = 0.0
    for box in world_collision_boxes(defender_state):
        outer_best = max(
            outer_best,
            _box_hit_score(
                box=box,
                muzzle_position=muzzle_position,
                relative_velocity=relative_velocity,
                tau_max=tau_max,
                hit_radius=PROJECTILE_AIRCRAFT_HIT_RADIUS_M,
                score_radius=dynamic_outer_radius,
            ),
        )
        if box.is_subsystem:
            core_best = max(
                core_best,
                _box_hit_score(
                    box=box,
                    muzzle_position=muzzle_position,
                    relative_velocity=relative_velocity,
                    tau_max=tau_max,
                    hit_radius=PROJECTILE_SUBSYSTEM_HIT_RADIUS_M,
                    score_radius=SHOT_CORE_RADIUS_M,
                ),
            )
    return float(np.clip(outer_best, 0.0, 1.0)), float(np.clip(core_best, 0.0, 1.0))


def centered_cone_score(cos_value: float, threshold_cos: float = FIRE_ALIGNMENT_THRESHOLD_COS) -> float:
    if not np.isfinite(cos_value):
        raise ValueError("cos_value must be finite")
    cos_value = float(np.clip(cos_value, -1.0, 1.0))
    if cos_value <= threshold_cos:
        return 0.0
    alpha = (cos_value - threshold_cos) / max(1.0 - threshold_cos, 1e-6)
    return float(alpha * alpha)


def compute_attack_geometry(
    *,
    attacker_state: dict[str, Any],
    defender_state: dict[str, Any],
) -> AttackGeometryMetrics:
    attacker_position = finite_vec3(attacker_state.get("position"), field_name="attacker.position")
    defender_position = finite_vec3(defender_state.get("position"), field_name="defender.position")
    attacker_velocity = finite_vec3(
        attacker_state.get("linear_velocity"),
        field_name="attacker.linear_velocity",
    )
    defender_velocity = finite_vec3(
        defender_state.get("linear_velocity"),
        field_name="defender.linear_velocity",
    )
    attacker_rotation = quaternion_xyzw_to_rotation_matrix(
        attacker_state.get("orientation_quat"),
        field_name="attacker.orientation_quat",
    )
    defender_rotation = quaternion_xyzw_to_rotation_matrix(
        defender_state.get("orientation_quat"),
        field_name="defender.orientation_quat",
    )
    attacker_forward = attacker_rotation[:, 2]
    defender_forward = defender_rotation[:, 2]

    relative_position = defender_position - attacker_position
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-6:
        raise ValueError("attacker and defender positions must differ")
    line_of_sight = relative_position / distance
    aim_cos = float(np.clip(np.dot(attacker_forward, line_of_sight), -1.0, 1.0))
    tail_cos = float(np.clip(np.dot(defender_forward, line_of_sight), -1.0, 1.0))
    heading_cos = float(np.clip(np.dot(attacker_forward, defender_forward), -1.0, 1.0))

    tracking_quality = float(np.clip(0.5 * (aim_cos + 1.0), 0.0, 1.0))
    tail_exposure = float(np.clip(0.5 * (tail_cos + 1.0), 0.0, 1.0))
    heading_alignment = float(np.clip(0.5 * (heading_cos + 1.0), 0.0, 1.0))
    tail_hold_score = tail_exposure * heading_alignment

    muzzle_position = attacker_position + attacker_forward * PROJECTILE_MUZZLE_FORWARD_OFFSET_M
    bullet_velocity = attacker_velocity + attacker_forward * PROJECTILE_SPEED_MPS
    bullet_speed = float(np.linalg.norm(bullet_velocity))
    if bullet_speed <= 1e-6:
        tau_seconds = 0.0
        min_distance = float(np.linalg.norm(defender_position - muzzle_position))
        tau_gate = 0.0
        outer_score = 0.0
        core_score = 0.0
    else:
        tau_max = PROJECTILE_MAX_RANGE_M / bullet_speed
        tau_seconds, min_distance = closest_approach(
            relative_position=defender_position - muzzle_position,
            relative_velocity=defender_velocity - bullet_velocity,
            horizon_seconds=tau_max,
        )
        tau_gate = float(np.exp(-tau_seconds / ATTACK_TAU_REFERENCE_SECONDS))
        outer_score, core_score = projectile_box_hit_scores(
            muzzle_position=muzzle_position,
            bullet_velocity=bullet_velocity,
            defender_state=defender_state,
        )

    fire_alignment = centered_cone_score(aim_cos)
    shot_feasibility = float(
        np.clip(
            fire_alignment * (SHOT_OUTER_WEIGHT * outer_score + SHOT_CORE_WEIGHT * core_score),
            0.0,
            1.0,
        )
    )
    return AttackGeometryMetrics(
        tau_seconds=tau_seconds,
        min_distance_meters=min_distance,
        tau_gate=tau_gate,
        fire_alignment=fire_alignment,
        shot_outer_score=outer_score,
        shot_core_score=core_score,
        shot_feasibility=shot_feasibility,
        tracking_quality=tracking_quality,
        tail_hold_score=tail_hold_score,
    )
