from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dfb_reinforcement_learning.obs.combat_geometry import (
    AIRCRAFT_COLLISION_BROADPHASE_FINE_THRESHOLD,
    AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS,
    LOCAL_COLLISION_BOXES,
    SHOT_BROADPHASE_RADIUS_METERS,
    compute_attack_geometry,
    quat_to_rotation_matrix,
)
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA


def _field_slices() -> dict[str, slice]:
    result: dict[str, slice] = {}
    for field in POLICY_OBSERVATION_SCHEMA.fields:
        result[field.name] = field.value_slice
    return result


FIELD_SLICES = _field_slices()
ATTACK_HISTORY_CACHE_KEY = "_reward_attack_history_cache"
BOUNDARY_PHI_CACHE_KEY = "_reward_boundary_phi_cache"

BOUNDARY_REFERENCE_SPEED_MPS = 80.0
BOUNDARY_REFERENCE_PITCH_TURN_RATE_RAD_S = float(np.deg2rad(45.0))
BOUNDARY_REFERENCE_TURN_RADIUS_METERS = (
    BOUNDARY_REFERENCE_SPEED_MPS / BOUNDARY_REFERENCE_PITCH_TURN_RATE_RAD_S
)
BOUNDARY_REFERENCE_OOB_GRACE_SECONDS = 20.0
BOUNDARY_REFERENCE_HALF_TURN_SECONDS = (0.5 * np.pi) / BOUNDARY_REFERENCE_PITCH_TURN_RATE_RAD_S
BOUNDARY_DEFAULT_WARNING_CLEARANCE_METERS = 2.0 * BOUNDARY_REFERENCE_TURN_RADIUS_METERS
BOUNDARY_DEFAULT_GROUND_WARNING_CLEARANCE_METERS = BOUNDARY_DEFAULT_WARNING_CLEARANCE_METERS
BOUNDARY_DEFAULT_CEILING_WARNING_CLEARANCE_METERS = BOUNDARY_DEFAULT_WARNING_CLEARANCE_METERS
BOUNDARY_DEFAULT_HORIZONTAL_WARNING_DISTANCE_METERS = BOUNDARY_DEFAULT_WARNING_CLEARANCE_METERS
BOUNDARY_REFERENCE_OUTBOUND_RECOVERY_SECONDS = max(
    (BOUNDARY_REFERENCE_OOB_GRACE_SECONDS - BOUNDARY_REFERENCE_HALF_TURN_SECONDS) / 2.0,
    0.0,
)
BOUNDARY_DEFAULT_HORIZONTAL_HARD_BOUNDARY_EXTRA_METERS = (
    BOUNDARY_REFERENCE_TURN_RADIUS_METERS
    + BOUNDARY_REFERENCE_SPEED_MPS * BOUNDARY_REFERENCE_OUTBOUND_RECOVERY_SECONDS
)


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _flat_roll_score_from_rotation(rotation: np.ndarray | None) -> float:
    if rotation is None or rotation.shape != (3, 3):
        return 1.0
    local_up_world = rotation[:, 1]
    up_y = float(np.clip(local_up_world[1], -1.0, 1.0))
    abs_roll_radians = float(np.arccos(up_y))
    return float(np.clip(1.0 - abs_roll_radians / np.pi, 0.0, 1.0))


@dataclass(frozen=True)
class PolicyRewardConfig:
    # ---- 交战距离基带 ----
    combat_turn_radius_meters: float = 80.0
    dynamic_turn_recovery_speed_mps: float = 40.0
    dynamic_turn_cruise_speed_mps: float = 60.0
    dynamic_turn_maneuver_speed_mps: float = 72.0
    dynamic_turn_high_speed_mps: float = 80.0
    dynamic_turn_low_speed_rate_scale: float = 0.94
    dynamic_turn_maneuver_rate_scale: float = 1.08
    dynamic_turn_high_speed_rate_scale: float = 1.10
    dynamic_turn_stall_authority_loss_scale: float = 0.78
    dynamic_turn_radius_min_scale: float = 0.45
    dynamic_turn_radius_max_scale: float = 2.40
    combat_range_meters: float = 320.0
    combat_range_tolerance_meters: float = 480.0
    combat_range_reward: float = 0.00
    too_far_penalty: float = 0.00

    # ---- 碰撞安全 ----
    aircraft_collision_radius_meters: float = 5.4
    aircraft_collision_threat_weight: float = 5.0
    aircraft_collision_outer_sigma_meters: float = 6.0
    aircraft_collision_core_sigma_meters: float = 1.5
    aircraft_collision_outer_weight: float = 0.2
    aircraft_collision_core_weight: float = 0.80
    aircraft_collision_tau_reference_seconds: float = 1.00
    aircraft_collision_horizon_seconds: float = 3.0
    aircraft_collision_dynamic_sigma_power: float = 0.5
    aircraft_collision_penalty_weight: float = 200.0
    surface_collision_penalty_weight: float = 200.0

    # ---- 时间压力 ----
    time_pressure_initial_bonus_per_second: float = 0.1
    time_pressure_rate_per_second: float = 2.0
    time_pressure_ramp_reference_seconds: float = 600.0
    positive_reward_decay_reference_seconds: float = 600.0
    positive_reward_decay_min_scale: float = 0.10

    # ---- 统一进攻 / 防御几何 ----
    attack_tau_reference_seconds: float = 0.75
    fire_alignment_threshold_cos: float = 0.25
    shot_outer_radius_meters: float = 2.4
    shot_core_radius_meters: float = 0.9
    shot_outer_weight: float = 0.2
    shot_core_weight: float = 0.8
    tracking_quality_weight: float = 2.5
    shot_feasibility_weight: float = 5.0
    tail_hold_weight: float = 1.0
    attack_advantage_weight: float = 1.0
    threat_advantage_weight: float = 1.0

    tactical_component_delta_history_length: int = 8
    tracking_delta_self_improve_weight: float = 0.00
    tracking_delta_self_worsen_weight: float = 0.00
    tracking_delta_enemy_improve_weight: float = 0.00
    tracking_delta_enemy_worsen_weight: float = 0.00
    tracking_delta_scale: float = 0.01
    tracking_delta_shape_power: float = 0.75
    tracking_delta_deadzone: float = 0.00
    shot_delta_self_improve_weight: float = 0.00
    shot_delta_self_worsen_weight: float = 0.00
    shot_delta_enemy_improve_weight: float = 0.00
    shot_delta_enemy_worsen_weight: float = 0.00
    shot_delta_scale: float = 0.01
    shot_delta_shape_power: float = 0.75
    shot_delta_deadzone: float = 0.00
    tail_delta_self_improve_weight: float = 0.00
    tail_delta_self_worsen_weight: float = 0.00
    tail_delta_enemy_improve_weight: float = 0.00
    tail_delta_enemy_worsen_weight: float = 0.00
    tail_delta_scale: float = 0.01
    tail_delta_shape_power: float = 0.75
    tail_delta_deadzone: float = 0.00

    # ---- 动作质量 ----
    pitch_up_tracking_weight: float = 0.00
    maneuver_activity_weight: float = 0.00
    flat_roll_bonus_weight: float = 0.0
    pitch_jitter_penalty_weight: float = 0.0
    roll_jitter_penalty_weight: float = 0.0
    yaw_jitter_penalty_weight: float = 0.0
    action_jitter_history_length: int = 8
    action_jitter_delta_threshold: float = 0.08
    action_jitter_path_reference: float = 1.0
    speed_jitter_penalty_weight: float = 0.00
    speed_jitter_history_length: int = 8
    speed_jitter_control_delta_threshold: float = 0.08
    speed_jitter_speed_delta_threshold_mps: float = 0.35
    speed_jitter_speed_delta_reference_mps: float = 2.5
    brake_penalty_weight: float = 0.30
    throttle_change_bonus_weight: float = 0.00
    throttle_change_reference: float = 0.10
    throttle_low_penalty_weight: float = 1.00
    throttle_low_penalty_threshold: float = 1.00
    low_speed_penalty_weight: float = 1.00
    low_speed_relief_speed_mps: float = 80.0
    one_circle_target_speed_mps: float = 45.0
    one_circle_speed_band_mps: float = 20.0
    two_circle_speed_reward_weight: float = 1.0
    one_circle_speed_reward_weight: float = 1.0
    stall_penalty_weight: float = 3.00
    overheat_penalty_weight: float = 1.00

    # ---- 修理相关 ----
    repair_static_penalty_weight: float = 3.00
    repair_twitch_penalty_weight: float = 1.00
    repair_high_health_penalty_weight: float = 3.00
    repair_low_health_bonus_weight: float = 0.10
    repair_destroyed_subsystem_bonus_weight: float = 0.5
    repair_under_threat_penalty_weight: float = 3.00
    repair_threat_aim_weight: float = 0.75
    repair_threat_distance_weight: float = 0.25
    repair_distance_reference_meters: float = 250.0
    repair_boundary_reference: float = 0.1
    repair_stall_reference: float = 0.1
    repair_duration_seconds: float = 10.0
    repair_completion_margin_seconds: float = 0.75
    repair_collision_threat_reference: float = 0.1

    # ---- 射击与命中 ----
    projectile_speed_meters_per_second: float = 1200.0
    projectile_max_range_meters: float = 1400.0
    projectile_aircraft_hit_radius_meters: float = 0.8
    projectile_subsystem_hit_radius_meters: float = 0.4
    projectile_muzzle_forward_offset_meters: float = 11.4
    shot_broadphase_epsilon: float = 1e-5
    fire_alignment_weight: float = 0.30
    fire_shot_weight: float = 0.60
    fire_tail_weight: float = 0.10
    fire_command_bonus_weight: float = 1.0
    fire_window_bonus_weight: float = 1.0
    predictive_fire_hit_bonus_weight: float = 1.0
    fire_hesitation_penalty_weight: float = 3.0
    hit_enemy_bonus_weight: float = 20.0
    got_hit_penalty_weight: float = 20.0

    # ---- 边界 ----
    boundary_history_length: int = 8
    boundary_warning_penalty: float = 0.25
    boundary_critical_penalty: float = 1.0
    boundary_delta_improve_weight: float = 2.00
    boundary_delta_worsen_weight: float = 2.00
    boundary_recovery_bonus_weight: float = 1.00
    ground_boundary_severity_weight: float = 5.00
    ceiling_boundary_severity_weight: float = 3.00
    horizontal_boundary_severity_weight: float = 2.00
    ground_warning_clearance_meters: float = BOUNDARY_DEFAULT_GROUND_WARNING_CLEARANCE_METERS
    ceiling_warning_clearance_meters: float = BOUNDARY_DEFAULT_CEILING_WARNING_CLEARANCE_METERS
    horizontal_warning_distance_meters: float = BOUNDARY_DEFAULT_HORIZONTAL_WARNING_DISTANCE_METERS
    horizontal_hard_boundary_extra_meters: float = BOUNDARY_DEFAULT_HORIZONTAL_HARD_BOUNDARY_EXTRA_METERS
    out_of_bounds_time_penalty_base: float = 1.00
    out_of_bounds_time_penalty_extra: float = 2.00
    out_of_bounds_time_penalty_reference_seconds: float = 20.0
    boundary_combat_suppression_reference: float = 6.0

    # ---- 终局事件 ----
    self_destroy_penalty: float = 600.0
    enemy_destroy_bonus: float = 600.0

    def __post_init__(self) -> None:
        if self.positive_reward_decay_reference_seconds <= 0.0:
            raise ValueError("positive_reward_decay_reference_seconds must be positive")
        if not 0.0 <= self.positive_reward_decay_min_scale <= 1.0:
            raise ValueError("positive_reward_decay_min_scale must be within [0, 1]")


@dataclass(frozen=True)
class AttackAdvantageComponents:
    tau_seconds: float
    min_distance_meters: float
    tau_gate: float
    fire_alignment_score: float
    shot_coarse_upper_bound: float
    shot_outer_score: float
    shot_core_score: float
    shot_feasibility: float
    tracking_quality: float
    tail_hold_score: float
    attack_advantage: float


@dataclass
class AircraftGeometryCache:
    position: np.ndarray
    velocity: np.ndarray
    forward: np.ndarray
    speed_mps: float
    stall_factor: float
    pullup_turn_radius_m: float
    rotation: np.ndarray | None = None
    destroyed_subsystems: frozenset[str] | None = None
    active_collision_boxes: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] | None = None


@dataclass(frozen=True)
class PolicyRewardBreakdown:
    total: float
    time_pressure: float
    time_pressure_scale: float
    positive_reward_decay_scale: float
    negative_reward_decay_scale: float
    distance_band: float
    attack_advantage: float
    threat_advantage: float
    tracking_delta_bonus: float
    shot_delta_bonus: float
    tail_delta_bonus: float

    fire_alignment_score: float
    shot_coarse_upper_bound: float
    shot_feasibility: float
    shot_outer_score: float
    shot_core_score: float
    tracking_quality: float
    tail_hold_score: float
    attack_tau_seconds: float
    opponent_attack_advantage: float
    opponent_fire_alignment_score: float
    opponent_shot_coarse_upper_bound: float
    opponent_shot_feasibility: float
    opponent_shot_outer_score: float
    opponent_shot_core_score: float
    opponent_tracking_quality: float
    opponent_tail_hold_score: float
    opponent_attack_tau_seconds: float

    pitch_up_tracking: float
    maneuver_activity: float
    flat_roll_bonus: float
    brake_penalty: float
    throttle_change_bonus: float
    throttle_low_penalty: float
    low_speed_penalty: float
    speed_jitter_penalty: float
    one_circle_gate: float
    two_circle_gate: float
    one_circle_speed_reward: float
    two_circle_speed_reward: float
    self_speed_mps: float
    speed_delta_mps: float
    closing_speed_mps: float
    pitch_jitter_penalty: float
    roll_jitter_penalty: float
    yaw_jitter_penalty: float
    stall_penalty: float
    overheat_penalty: float

    repair_twitch_penalty: float
    repair_static_penalty: float
    repair_high_health_penalty: float
    repair_under_threat_penalty: float
    repair_low_health_bonus: float
    repair_destroyed_subsystem_bonus: float

    repair_opportunity_gate: float
    repair_completion_safety_gate: float

    ground_boundary_threat: float
    ceiling_boundary_threat: float
    horizontal_boundary_threat: float
    ground_boundary_penalty: float
    ceiling_boundary_penalty: float
    horizontal_boundary_penalty: float
    boundary_recovery_bonus: float
    out_of_bounds_time_penalty: float
    boundary_combat_gate: float

    predictive_fire_hit_bonus: float
    fire_command_bonus: float
    fire_window_bonus: float
    fire_hesitation_penalty: float
    hit_enemy_bonus: float
    got_hit_penalty: float
    aircraft_collision_threat: float
    aircraft_collision_penalty: float
    surface_collision_penalty: float

    self_destroy_penalty: float
    enemy_destroy_bonus: float

    def asdict(self) -> dict[str, float]:
        return asdict(self)


class PolicyRewardComposer:
    def __init__(self, config: PolicyRewardConfig | None = None) -> None:
        self.config = config or PolicyRewardConfig()

    def reward_history_frame_length(self) -> int:
        cfg = self.config
        return max(
            cfg.speed_jitter_history_length,
            cfg.action_jitter_history_length,
            cfg.boundary_history_length,
            cfg.tactical_component_delta_history_length,
        )

    def _compute_line_of_sight_body(
        self,
        self_state: dict[str, Any],
        enemy_state: dict[str, Any],
    ) -> np.ndarray:
        ego_pos = np.asarray(self_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        enemy_pos = np.asarray(enemy_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        ego_rot = quat_to_rotation_matrix(self_state.get("orientation_quat", [0.0, 0.0, 0.0, 1.0]))
        world_to_ego = ego_rot.T
        rel_pos_body = world_to_ego @ (enemy_pos - ego_pos)
        norm = float(np.linalg.norm(rel_pos_body))
        if norm <= 1e-6:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return rel_pos_body / norm

    def _aircraft_geometry_cache(
        self,
        aircraft_state: dict[str, Any],
        cache_by_id: dict[int, AircraftGeometryCache],
    ) -> AircraftGeometryCache:
        state_id = id(aircraft_state)
        cached = cache_by_id.get(state_id)
        if cached is not None:
            return cached

        position = np.asarray(aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        velocity = np.asarray(aircraft_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        speed_mps = float(np.linalg.norm(velocity))
        forward = np.asarray(aircraft_state.get("forward", [0.0, 0.0, 1.0]), dtype=np.float32)
        forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
        stall_factor = float(aircraft_state.get("stall_factor", 0.0))
        pullup_turn_radius_m = self._pullup_turn_radius_from_primitives(
            aircraft_state=aircraft_state,
            speed_mps=speed_mps,
            stall_factor=stall_factor,
        )
        cached = AircraftGeometryCache(
            position=position,
            velocity=velocity,
            forward=forward,
            speed_mps=speed_mps,
            stall_factor=stall_factor,
            pullup_turn_radius_m=pullup_turn_radius_m,
        )
        cache_by_id[state_id] = cached
        return cached

    def _ensure_aircraft_fine_geometry(
        self,
        aircraft_state: dict[str, Any],
        geometry_cache: AircraftGeometryCache,
    ) -> None:
        if (
            geometry_cache.rotation is not None
            and geometry_cache.destroyed_subsystems is not None
            and geometry_cache.active_collision_boxes is not None
        ):
            return

        rotation = quat_to_rotation_matrix(aircraft_state.get("orientation_quat", [0.0, 0.0, 0.0, 1.0])).astype(
            np.float32
        )
        destroyed_subsystems = frozenset(
            str(entry.get("name"))
            for entry in aircraft_state.get("subsystems", [])
            if str(entry.get("stage", "")) == "Destroyed"
        )
        active_collision_boxes = tuple(
            (
                (geometry_cache.position + rotation @ local_center).astype(np.float32),
                rotation,
                half_extents.astype(np.float32),
            )
            for name, local_center, half_extents in LOCAL_COLLISION_BOXES
            if name not in destroyed_subsystems
        )
        geometry_cache.rotation = rotation
        geometry_cache.destroyed_subsystems = destroyed_subsystems
        geometry_cache.active_collision_boxes = active_collision_boxes

    def _cached_attack_history_components_from_info(
        self,
        info: dict[str, Any] | None,
    ) -> tuple[AttackAdvantageComponents, AttackAdvantageComponents] | None:
        if not isinstance(info, dict):
            return None
        attack_history_cache = info.get(ATTACK_HISTORY_CACHE_KEY)
        if not isinstance(attack_history_cache, dict):
            return None
        self_attack = attack_history_cache.get("self_attack")
        opponent_attack = attack_history_cache.get("opponent_attack")
        if isinstance(self_attack, AttackAdvantageComponents) and isinstance(opponent_attack, AttackAdvantageComponents):
            return self_attack, opponent_attack
        return None

    def _cached_boundary_phi_from_info(self, info: dict[str, Any] | None) -> dict[str, float] | None:
        if not isinstance(info, dict):
            return None
        boundary_phi_cache = info.get(BOUNDARY_PHI_CACHE_KEY)
        if not isinstance(boundary_phi_cache, dict):
            return None
        try:
            return {
                "ground": float(boundary_phi_cache["ground"]),
                "ceiling": float(boundary_phi_cache["ceiling"]),
                "horizontal": float(boundary_phi_cache["horizontal"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _scaled_positive_delta_gain(
        self,
        delta_value: float,
        *,
        scale: float,
        shape_power: float,
        deadzone: float,
    ) -> float:
        positive_delta = max(float(delta_value) - max(deadzone, 0.0), 0.0)
        if positive_delta <= 0.0:
            return 0.0
        safe_scale = max(scale, 1e-6)
        safe_power = max(shape_power, 1e-6)
        normalized = positive_delta / safe_scale
        return float(np.tanh(normalized**safe_power))

    def _positive_continuous_reward_decay_scale(self, elapsed_time_seconds: float | None) -> float:
        cfg = self.config
        if elapsed_time_seconds is None:
            return 1.0
        safe_elapsed = max(float(elapsed_time_seconds), 0.0)
        safe_reference = max(cfg.positive_reward_decay_reference_seconds, 1e-6)
        decay = 1.0 / (1.0 + (safe_elapsed / safe_reference))
        return float(max(decay, cfg.positive_reward_decay_min_scale))

    def _negative_continuous_reward_decay_scale(self, elapsed_time_seconds: float | None) -> float:
        _ = elapsed_time_seconds
        return 1.0

    def _time_pressure_scale(self, elapsed_time_seconds: float | None) -> float:
        cfg = self.config
        if elapsed_time_seconds is None:
            return 1.0
        safe_elapsed = max(float(elapsed_time_seconds), 0.0)
        safe_reference = max(cfg.time_pressure_ramp_reference_seconds, 1e-6)
        return float(_smoothstep01(safe_elapsed / safe_reference))

    def _distance_band_reward(self, distance_meters: float) -> float:
        cfg = self.config
        if distance_meters <= cfg.combat_range_meters:
            return cfg.combat_range_reward
        if distance_meters >= cfg.combat_range_tolerance_meters:
            return -cfg.too_far_penalty
        alpha = (distance_meters - cfg.combat_range_meters) / max(
            cfg.combat_range_tolerance_meters - cfg.combat_range_meters,
            1e-6,
        )
        return float(cfg.combat_range_reward + alpha * (-cfg.too_far_penalty - cfg.combat_range_reward))

    def _estimated_non_stall_rate_scale(self, speed_mps: float) -> float:
        cfg = self.config
        recovery = max(cfg.dynamic_turn_recovery_speed_mps, 0.1)
        cruise = max(cfg.dynamic_turn_cruise_speed_mps, recovery + 1.0)
        maneuver = max(cfg.dynamic_turn_maneuver_speed_mps, cruise + 1.0)
        high_speed = max(cfg.dynamic_turn_high_speed_mps, maneuver + 1.0)
        low = cfg.dynamic_turn_low_speed_rate_scale
        maneuver_scale = cfg.dynamic_turn_maneuver_rate_scale
        high = cfg.dynamic_turn_high_speed_rate_scale

        if speed_mps <= cruise:
            t = _smoothstep01((speed_mps - recovery) / max(cruise - recovery, 1e-6))
            return float(low + (1.0 - low) * t)
        if speed_mps <= maneuver:
            t = _smoothstep01((speed_mps - cruise) / max(maneuver - cruise, 1e-6))
            return float(1.0 + (maneuver_scale - 1.0) * t)
        t = _smoothstep01((speed_mps - maneuver) / max(high_speed - maneuver, 1e-6))
        return float(maneuver_scale + (high - maneuver_scale) * t)

    def _instantaneous_turn_radius(self, speed_mps: float, stall_factor: float) -> float:
        cfg = self.config
        speed = max(speed_mps, 1e-3)
        rate_scale = self._estimated_non_stall_rate_scale(speed)
        control_authority = max(1.0 - stall_factor * cfg.dynamic_turn_stall_authority_loss_scale, 0.2)
        radius_scale = ((speed / max(cfg.dynamic_turn_cruise_speed_mps, 1e-6)) ** 2) / max(
            rate_scale * control_authority,
            1e-3,
        )
        radius_scale = float(
            np.clip(radius_scale, cfg.dynamic_turn_radius_min_scale, cfg.dynamic_turn_radius_max_scale)
        )
        return cfg.combat_turn_radius_meters * radius_scale

    def _pullup_turn_radius_from_primitives(
        self,
        *,
        aircraft_state: dict[str, Any],
        speed_mps: float,
        stall_factor: float,
    ) -> float:
        pullup_turn_radius_m = aircraft_state.get("pullup_turn_radius_m")
        if pullup_turn_radius_m is not None:
            try:
                value = float(pullup_turn_radius_m)
                if np.isfinite(value) and value > 1e-3:
                    return value
            except (TypeError, ValueError):
                pass
        return self._instantaneous_turn_radius(speed_mps, stall_factor)

    def _state_pullup_turn_radius(self, aircraft_state: dict[str, Any]) -> float:
        speed_mps = float(
            np.linalg.norm(np.asarray(aircraft_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32))
        )
        stall_factor = float(aircraft_state.get("stall_factor", 0.0))
        return self._pullup_turn_radius_from_primitives(
            aircraft_state=aircraft_state,
            speed_mps=speed_mps,
            stall_factor=stall_factor,
        )

    def _aircraft_speed_mps(self, aircraft_state: dict[str, Any]) -> float:
        return float(
            np.linalg.norm(np.asarray(aircraft_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32))
        )

    def _max_level_speed_mps(self, aircraft_state: dict[str, Any]) -> float:
        value = aircraft_state.get("max_level_speed_mps")
        if value is not None:
            try:
                value_f = float(value)
                if np.isfinite(value_f) and value_f > 1e-3:
                    return value_f
            except (TypeError, ValueError):
                pass
        return max(self.config.dynamic_turn_high_speed_mps, 1.0)

    def _closing_speed_mps(
        self,
        *,
        attacker_position: np.ndarray,
        attacker_velocity: np.ndarray,
        defender_position: np.ndarray,
        defender_velocity: np.ndarray,
    ) -> float:
        relative_position = defender_position - attacker_position
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-6:
            return 0.0
        line_of_sight = relative_position / distance
        return float(np.dot(attacker_velocity - defender_velocity, line_of_sight))

    def _aircraft_up_vector(self, aircraft_state: dict[str, Any]) -> np.ndarray:
        rotation = quat_to_rotation_matrix(aircraft_state.get("orientation_quat", [0.0, 0.0, 0.0, 1.0]))
        up = rotation[:, 1].astype(np.float32)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-6:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return up / up_norm

    def _merge_circle_gates(
        self,
        *,
        self_state: dict[str, Any],
        enemy_state: dict[str, Any],
    ) -> tuple[float, float]:
        cfg = self.config
        self_forward = np.asarray(self_state.get("forward", [0.0, 0.0, 1.0]), dtype=np.float32)
        enemy_forward = np.asarray(enemy_state.get("forward", [0.0, 0.0, -1.0]), dtype=np.float32)
        self_forward = self_forward / max(float(np.linalg.norm(self_forward)), 1e-6)
        enemy_forward = enemy_forward / max(float(np.linalg.norm(enemy_forward)), 1e-6)
        self_up = self._aircraft_up_vector(self_state)
        enemy_up = self._aircraft_up_vector(enemy_state)
        cos_front = float(np.clip(np.dot(enemy_forward, self_forward), -1.0, 1.0))
        cos_up = float(np.clip(np.dot(enemy_up, self_up), -1.0, 1.0))
        circle_product = float(np.clip(cos_front * cos_up, -1.0, 1.0))
        two_circle_gate = 1.0 if circle_product >= 0.0 else 0.0
        one_circle_gate = 1.0 if circle_product < 0.0 else 0.0
        return one_circle_gate, two_circle_gate

    def _merge_speed_rewards(
        self,
        *,
        self_state: dict[str, Any],
        self_speed_mps: float,
    ) -> tuple[float, float, float, float]:
        cfg = self.config
        max_level_speed = self._max_level_speed_mps(self_state)
        speed_ratio = float(np.clip(self_speed_mps / max(max_level_speed, 1e-6), 0.0, 1.0))
        one_circle_target_ratio = float(
            np.clip(cfg.one_circle_target_speed_mps / max(max_level_speed, 1e-6), 0.0, 1.0)
        )
        one_circle_band = max(cfg.one_circle_speed_band_mps / max(max_level_speed, 1e-6), 1e-3)
        one_circle_score = float(
            np.exp(-(((speed_ratio - one_circle_target_ratio) / one_circle_band) ** 2))
        )
        two_circle_score = speed_ratio * speed_ratio
        return speed_ratio, one_circle_target_ratio, one_circle_score, two_circle_score

    def _low_speed_penalty(self, speed_mps: float) -> float:
        cfg = self.config
        relief_speed = max(cfg.low_speed_relief_speed_mps, 1.0)
        alpha = float(np.clip((relief_speed - speed_mps) / relief_speed, 0.0, 1.0))
        return cfg.low_speed_penalty_weight * alpha

    def _repair_completion_safety_gate(
        self,
        *,
        aircraft_state: dict[str, Any],
        geometry_cache: AircraftGeometryCache | None,
        arena: dict[str, Any],
        repair_engaged: float,
        repair_active: bool,
        repair_elapsed_seconds: float,
        aircraft_collision_threat: float,
    ) -> float:
        cfg = self.config
        if repair_engaged <= 0.0:
            return 1.0

        repair_duration = max(cfg.repair_duration_seconds, 0.1)
        elapsed_seconds = float(np.clip(repair_elapsed_seconds, 0.0, repair_duration))
        remaining_seconds = repair_duration if not repair_active else max(repair_duration - elapsed_seconds, 0.0)
        if remaining_seconds <= 1e-6:
            return 1.0

        completion_window_seconds = remaining_seconds + max(cfg.repair_completion_margin_seconds, 0.0)
        candidate_times: list[float] = []
        ground_height = float(arena.get("ground_height", 0.0))
        ceiling_height = float(arena.get("flight_ceiling_height", 0.0))
        arena_radius = float(arena.get("arena_radius", 0.0))
        position = (
            geometry_cache.position
            if geometry_cache is not None
            else np.asarray(aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        velocity = (
            geometry_cache.velocity
            if geometry_cache is not None
            else np.asarray(aircraft_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        sphere_radius = AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS

        ground_time = self._time_to_boundary(
            float(position[1] - ground_height - sphere_radius),
            max(float(-velocity[1]), 0.0),
        )
        if ground_time is not None:
            candidate_times.append(ground_time)

        ceiling_time = self._time_to_boundary(
            float(ceiling_height - position[1] - sphere_radius),
            max(float(velocity[1]), 0.0),
        )
        if ceiling_time is not None:
            candidate_times.append(ceiling_time)

        if arena_radius > 1e-6:
            horizontal_position = np.asarray([position[0], position[2]], dtype=np.float32)
            horizontal_velocity = np.asarray([velocity[0], velocity[2]], dtype=np.float32)
            radial_distance = float(np.linalg.norm(horizontal_position))
            if radial_distance <= 1e-6:
                outward_speed = 0.0
            else:
                radial_direction = horizontal_position / radial_distance
                outward_speed = max(float(np.dot(horizontal_velocity, radial_direction)), 0.0)
            horizontal_time = self._time_to_boundary(
                float(self._horizontal_hard_radius(arena_radius) - radial_distance - sphere_radius),
                outward_speed,
            )
            if horizontal_time is not None:
                candidate_times.append(horizontal_time)
        if candidate_times:
            earliest_impact_seconds = min(candidate_times)
            boundary_gate = float(
                np.clip(
                    (earliest_impact_seconds - cfg.repair_completion_margin_seconds)
                    / max(completion_window_seconds, 1e-6),
                    0.0,
                    1.0,
                )
            )
        else:
            boundary_gate = 1.0

        collision_gate = float(
            np.exp(-aircraft_collision_threat / max(cfg.repair_collision_threat_reference, 1e-6))
        )
        return boundary_gate * collision_gate

    def _repair_threat_gate(
        self,
        *,
        distance_meters: float,
        opponent_tracking_quality: float,
    ) -> float:
        cfg = self.config
        aim_threat = float(np.clip(opponent_tracking_quality, 0.0, 1.0))
        distance_threat = float(
            np.exp(-max(distance_meters, 0.0) / max(cfg.repair_distance_reference_meters, 1e-6))
        )
        threat_score = float(
            np.clip(
                cfg.repair_threat_aim_weight * aim_threat
                + cfg.repair_threat_distance_weight * distance_threat,
                0.0,
                1.0,
            )
        )
        return float(1.0 - threat_score)

    def _speed_jitter_penalty(
        self,
        *,
        current_throttle: float,
        previous_throttle: float,
        previous_previous_throttle: float,
        current_speed_mps: float,
        previous_speed_mps: float,
        opportunity_relief_gate: float,
    ) -> float:
        cfg = self.config
        delta_now = current_throttle - previous_throttle
        delta_prev = previous_throttle - previous_previous_throttle
        if delta_now * delta_prev >= 0.0:
            return 0.0

        throttle_path = abs(delta_now) + abs(delta_prev)
        if throttle_path <= 1e-6:
            return 0.0
        throttle_net = abs(current_throttle - previous_previous_throttle)
        throttle_inefficiency = 1.0 - float(np.clip(throttle_net / throttle_path, 0.0, 1.0))
        reversal_strength = min(abs(delta_now), abs(delta_prev))
        speed_delta = abs(current_speed_mps - previous_speed_mps)
        speed_inefficiency = 1.0 - float(
            np.clip(speed_delta / max(cfg.speed_jitter_speed_delta_reference_mps, 1e-6), 0.0, 1.0)
        )
        return (
            cfg.speed_jitter_penalty_weight
            * reversal_strength
            * throttle_inefficiency
            * speed_inefficiency
            * (1.0 - 0.85 * opportunity_relief_gate)
        )

    def _reversal_rate(self, deltas: np.ndarray, threshold: float) -> float:
        if deltas.size < 2:
            return 0.0
        signs = np.zeros_like(deltas, dtype=np.int8)
        signs[deltas > threshold] = 1
        signs[deltas < -threshold] = -1
        reversal_count = 0
        opportunity_count = 0
        for prev_sign, next_sign in zip(signs[:-1], signs[1:]):
            if prev_sign == 0 or next_sign == 0:
                continue
            opportunity_count += 1
            if prev_sign != next_sign:
                reversal_count += 1
        if opportunity_count <= 0:
            return 0.0
        return float(reversal_count / opportunity_count)

    def _control_jitter_score_from_values(
        self,
        values: list[float],
        *,
        delta_threshold: float,
        path_reference: float,
        opportunity_relief_gate: float,
    ) -> float:
        if len(values) < 3:
            return 0.0
        value_array = np.asarray(values, dtype=np.float32)
        value_deltas = np.diff(value_array)
        reversal_rate = self._reversal_rate(value_deltas, delta_threshold)
        oscillation_score = reversal_rate * reversal_rate

        control_path = float(np.sum(np.abs(value_deltas)))
        if control_path <= 1e-6:
            return 0.0
        control_net = float(abs(value_array[-1] - value_array[0]))
        control_inefficiency = 1.0 - float(np.clip(control_net / control_path, 0.0, 1.0))
        path_activity = float(np.clip(control_path / max(path_reference, 1e-6), 0.0, 1.0))
        return float(
            oscillation_score
            * control_inefficiency
            * path_activity
            * (1.0 - 0.85 * opportunity_relief_gate)
        )

    def _speed_jitter_penalty_from_history(
        self,
        *,
        reward_history: dict[str, Any],
        ego_role: str,
        current_throttle: float,
        current_brake: float,
        current_speed_mps: float,
        opportunity_relief_gate: float,
    ) -> float:
        cfg = self.config
        frames = reward_history.get("frames", [])
        if not isinstance(frames, list) or len(frames) < 2:
            return 0.0

        controls: list[float] = []
        speeds: list[float] = []
        for frame in frames[-cfg.speed_jitter_history_length :]:
            if not isinstance(frame, dict):
                continue
            frame_action_cont = frame.get("action_cont")
            frame_action_bin = frame.get("action_bin")
            frame_info = frame.get("info")
            if frame_action_cont is None or not isinstance(frame_info, dict):
                continue
            try:
                throttle = float(np.asarray(frame_action_cont, dtype=np.float32)[0])
            except (IndexError, TypeError, ValueError):
                continue
            brake = 0.0
            if frame_action_bin is not None:
                try:
                    brake = float(np.asarray(frame_action_bin, dtype=np.float32)[0])
                except (IndexError, TypeError, ValueError):
                    brake = 0.0
            aircraft_state = frame_info.get("aircraft_by_role", {}).get(ego_role)
            if not isinstance(aircraft_state, dict):
                continue
            controls.append(throttle - brake)
            speeds.append(self._aircraft_speed_mps(aircraft_state))

        controls.append(current_throttle - current_brake)
        speeds.append(current_speed_mps)
        if len(controls) < 3 or len(speeds) < 3:
            return 0.0

        control_score = self._control_jitter_score_from_values(
            controls,
            delta_threshold=cfg.speed_jitter_control_delta_threshold,
            path_reference=1.0,
            opportunity_relief_gate=0.0,
        )
        if control_score <= 0.0:
            return 0.0

        speed_array = np.asarray(speeds, dtype=np.float32)
        speed_deltas = np.diff(speed_array)

        speed_reversal_rate = self._reversal_rate(
            speed_deltas,
            cfg.speed_jitter_speed_delta_threshold_mps,
        )
        speed_reversal_score = speed_reversal_rate * speed_reversal_rate

        speed_path = float(np.sum(np.abs(speed_deltas)))
        speed_net = float(abs(speed_array[-1] - speed_array[0]))
        if speed_path <= 1e-6:
            speed_inefficiency = 1.0
        else:
            speed_inefficiency = 1.0 - float(np.clip(speed_net / speed_path, 0.0, 1.0))

        return (
            cfg.speed_jitter_penalty_weight
            * (0.65 * control_score + 0.35 * speed_reversal_score * speed_inefficiency)
            * (1.0 - 0.85 * opportunity_relief_gate)
        )

    def _action_axis_jitter_penalty_from_history(
        self,
        *,
        reward_history: dict[str, Any] | None,
        current_action_cont: np.ndarray,
        axis_index: int,
        weight: float,
        opportunity_relief_gate: float,
        previous_previous_action_cont: np.ndarray | None = None,
        previous_action_cont: np.ndarray | None = None,
    ) -> float:
        cfg = self.config
        values: list[float] = []
        if reward_history is not None:
            frames = reward_history.get("frames", [])
            if isinstance(frames, list):
                for frame in frames[-cfg.action_jitter_history_length :]:
                    if not isinstance(frame, dict):
                        continue
                    frame_action_cont = frame.get("action_cont")
                    if frame_action_cont is None:
                        continue
                    try:
                        values.append(float(np.asarray(frame_action_cont, dtype=np.float32)[axis_index]))
                    except (IndexError, TypeError, ValueError):
                        continue
        elif previous_previous_action_cont is not None and previous_action_cont is not None:
            values.extend(
                [
                    float(np.asarray(previous_previous_action_cont, dtype=np.float32)[axis_index]),
                    float(np.asarray(previous_action_cont, dtype=np.float32)[axis_index]),
                ]
            )

        try:
            values.append(float(np.asarray(current_action_cont, dtype=np.float32)[axis_index]))
        except (IndexError, TypeError, ValueError):
            return 0.0
        score = self._control_jitter_score_from_values(
            values,
            delta_threshold=cfg.action_jitter_delta_threshold,
            path_reference=cfg.action_jitter_path_reference,
            opportunity_relief_gate=opportunity_relief_gate,
        )
        return float(weight * score)

    def _boundary_clearance_risk(
        self,
        clearance_meters: float,
        *,
        zero_risk_at_meters: float,
        full_risk_at_meters: float,
    ) -> float:
        if clearance_meters >= zero_risk_at_meters:
            return 0.0
        if clearance_meters <= full_risk_at_meters:
            return 1.0
        alpha = (zero_risk_at_meters - clearance_meters) / max(
            zero_risk_at_meters - full_risk_at_meters,
            1e-6,
        )
        return _smoothstep01(alpha)

    def _time_to_boundary(self, clearance_meters: float, inward_speed_meters_per_second: float) -> float | None:
        if inward_speed_meters_per_second <= 1e-6:
            return None
        return max(clearance_meters, 0.0) / inward_speed_meters_per_second

    def _clearance_zone_phi(
        self,
        *,
        clearance_meters: float,
        reference_clearance_meters: float,
    ) -> float:
        alpha = 1.0 - (clearance_meters / max(reference_clearance_meters, 1e-6))
        return _smoothstep01(alpha)

    def _ground_zone_phi(
        self,
        aircraft_state: dict[str, Any],
        ground_height: float,
        *,
        geometry_cache: AircraftGeometryCache | None = None,
    ) -> float:
        position = geometry_cache.position if geometry_cache is not None else np.asarray(
            aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        clearance = float(position[1] - ground_height - AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS)
        return self._clearance_zone_phi(
            clearance_meters=clearance,
            reference_clearance_meters=self.config.ground_warning_clearance_meters,
        )

    def _ceiling_zone_phi(
        self,
        aircraft_state: dict[str, Any],
        ceiling_height: float,
        *,
        geometry_cache: AircraftGeometryCache | None = None,
    ) -> float:
        position = geometry_cache.position if geometry_cache is not None else np.asarray(
            aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        clearance = float(ceiling_height - position[1] - AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS)
        return self._clearance_zone_phi(
            clearance_meters=clearance,
            reference_clearance_meters=self.config.ceiling_warning_clearance_meters,
        )

    def _horizontal_hard_radius(self, arena_radius: float) -> float:
        return float(arena_radius + self.config.horizontal_hard_boundary_extra_meters)

    def _horizontal_zone_phi(
        self,
        aircraft_state: dict[str, Any],
        arena_radius: float,
        *,
        geometry_cache: AircraftGeometryCache | None = None,
    ) -> float:
        if arena_radius <= 1e-6:
            return 0.0
        position = geometry_cache.position if geometry_cache is not None else np.asarray(
            aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        radial_distance = float(np.linalg.norm(position[[0, 2]]))
        occupied_radius = radial_distance + AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS
        soft_clearance = arena_radius - occupied_radius
        phi_soft = 0.5 * self._clearance_zone_phi(
            clearance_meters=soft_clearance,
            reference_clearance_meters=self.config.horizontal_warning_distance_meters,
        )
        hard_radius = self._horizontal_hard_radius(arena_radius)
        if occupied_radius <= arena_radius:
            phi_hard = 0.0
        else:
            alpha = (occupied_radius - arena_radius) / max(hard_radius - arena_radius, 1e-6)
            phi_hard = 0.5 + (0.5 * _smoothstep01(alpha))
        return float(max(phi_soft, phi_hard))

    def _phi_history_values(
        self,
        *,
        current_info: dict[str, Any],
        previous_info: dict[str, Any] | None,
        previous_aircraft_state: dict[str, Any] | None,
        reward_history: dict[str, Any] | None,
        cache_key: str,
        phi_fn: Any,
    ) -> list[float]:
        cfg = self.config
        history_values: list[float] = []
        if reward_history is not None:
            frames = reward_history.get("frames", [])
            ego_role = current_info.get("ego_role", "fighter1")
            for frame in frames[-cfg.boundary_history_length :]:
                info = frame.get("info")
                if not isinstance(info, dict):
                    continue
                boundary_phi_cache = frame.get("boundary_phi_cache")
                if isinstance(boundary_phi_cache, dict) and cache_key in boundary_phi_cache:
                    try:
                        history_values.append(float(boundary_phi_cache[cache_key]))
                        continue
                    except (TypeError, ValueError):
                        pass
                frame_state = info.get("aircraft_by_role", {}).get(ego_role)
                if not isinstance(frame_state, dict):
                    continue
                history_values.append(float(phi_fn(info, frame_state)))
        elif previous_aircraft_state is not None and previous_info is not None:
            history_values.append(float(phi_fn(previous_info, previous_aircraft_state)))
        return history_values

    def _window_delta_scores(
        self,
        *,
        history_values: list[float],
        current_value: float,
    ) -> tuple[float, float, float]:
        window = [float(v) for v in history_values if np.isfinite(v)]
        window.append(float(current_value))
        if len(window) < 2:
            return 0.0, 0.0, 0.0

        oldest = window[0]
        newest = window[-1]
        improve = max(oldest - newest, 0.0)
        worsen = max(newest - oldest, 0.0)
        path_change = 0.0
        for left, right in zip(window[:-1], window[1:], strict=False):
            path_change += abs(right - left)
        if path_change <= 1e-6:
            anti_oscillation = 0.0
        else:
            anti_oscillation = float(np.clip(abs(newest - oldest) / path_change, 0.0, 1.0))
        return float(improve), float(worsen), anti_oscillation

    def build_attack_history_cache(self, info: dict[str, Any]) -> dict[str, AttackAdvantageComponents]:
        cached = self._cached_attack_history_components_from_info(info)
        if cached is not None:
            self_attack, opponent_attack = cached
            return {
                "self_attack": self_attack,
                "opponent_attack": opponent_attack,
            }
        ego_role = info.get("ego_role", "fighter1")
        enemy_role = info.get("enemy_role", "fighter2")
        aircraft_by_role = info.get("aircraft_by_role", {})
        attacker_state = aircraft_by_role.get(ego_role)
        defender_state = aircraft_by_role.get(enemy_role)
        if not isinstance(attacker_state, dict) or not isinstance(defender_state, dict):
            raise ValueError("info missing aircraft states for attack history cache")
        cache = {
            "self_attack": self._compute_attack_advantage(
                attacker_state=attacker_state,
                defender_state=defender_state,
            ),
            "opponent_attack": self._compute_attack_advantage(
                attacker_state=defender_state,
                defender_state=attacker_state,
            ),
        }
        info[ATTACK_HISTORY_CACHE_KEY] = cache
        return cache

    def build_boundary_phi_cache(self, info: dict[str, Any]) -> dict[str, float]:
        cached = self._cached_boundary_phi_from_info(info)
        if cached is not None:
            return cached
        ego_role = info.get("ego_role", "fighter1")
        aircraft_state = info.get("aircraft_by_role", {}).get(ego_role)
        if not isinstance(aircraft_state, dict):
            raise ValueError("info missing ego aircraft state for boundary phi cache")
        arena = info.get("arena", {})
        cache = {
            "ground": self._ground_zone_phi(
                aircraft_state,
                float(arena.get("ground_height", 0.0)),
            ),
            "ceiling": self._ceiling_zone_phi(
                aircraft_state,
                float(arena.get("flight_ceiling_height", 0.0)),
            ),
            "horizontal": self._horizontal_zone_phi(
                aircraft_state,
                float(arena.get("arena_radius", 0.0)),
            ),
        }
        info[BOUNDARY_PHI_CACHE_KEY] = cache
        return cache

    def _history_attack_components_from_frame(
        self,
        *,
        frame: dict[str, Any],
        current_info: dict[str, Any],
    ) -> tuple[AttackAdvantageComponents, AttackAdvantageComponents] | None:
        attack_history_cache = frame.get("attack_history_cache")
        if isinstance(attack_history_cache, dict):
            self_attack = attack_history_cache.get("self_attack")
            opponent_attack = attack_history_cache.get("opponent_attack")
            if isinstance(self_attack, AttackAdvantageComponents) and isinstance(
                opponent_attack, AttackAdvantageComponents
            ):
                return self_attack, opponent_attack

        info = frame.get("info")
        if not isinstance(info, dict):
            return None
        ego_role = current_info.get("ego_role", "fighter1")
        enemy_role = current_info.get("enemy_role", "fighter2")
        aircraft_by_role = info.get("aircraft_by_role", {})
        attacker_state = aircraft_by_role.get(ego_role)
        defender_state = aircraft_by_role.get(enemy_role)
        if not isinstance(attacker_state, dict) or not isinstance(defender_state, dict):
            return None
        return (
            self._compute_attack_advantage(attacker_state=attacker_state, defender_state=defender_state),
            self._compute_attack_advantage(attacker_state=defender_state, defender_state=attacker_state),
        )

    def _attack_component_histories(
        self,
        *,
        reward_history: dict[str, Any] | None,
        current_info: dict[str, Any],
        previous_self_attack: AttackAdvantageComponents | None,
        previous_opponent_attack: AttackAdvantageComponents | None,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
        cfg = self.config
        tracking_history: list[float] = []
        opponent_tracking_history: list[float] = []
        shot_history: list[float] = []
        opponent_shot_history: list[float] = []
        tail_history: list[float] = []
        opponent_tail_history: list[float] = []
        if reward_history is not None:
            frames = reward_history.get("frames", [])
            for frame in frames[-cfg.tactical_component_delta_history_length :]:
                if not isinstance(frame, dict):
                    continue
                attack_components = self._history_attack_components_from_frame(
                    frame=frame,
                    current_info=current_info,
                )
                if attack_components is None:
                    continue
                self_attack, opponent_attack = attack_components
                tracking_history.append(float(self_attack.tracking_quality))
                opponent_tracking_history.append(float(opponent_attack.tracking_quality))
                shot_history.append(float(self_attack.shot_feasibility))
                opponent_shot_history.append(float(opponent_attack.shot_feasibility))
                tail_history.append(float(self_attack.tail_hold_score))
                opponent_tail_history.append(float(opponent_attack.tail_hold_score))
        if not tracking_history and previous_self_attack is not None:
            tracking_history.append(float(previous_self_attack.tracking_quality))
            shot_history.append(float(previous_self_attack.shot_feasibility))
            tail_history.append(float(previous_self_attack.tail_hold_score))
        if not opponent_tracking_history and previous_opponent_attack is not None:
            opponent_tracking_history.append(float(previous_opponent_attack.tracking_quality))
            opponent_shot_history.append(float(previous_opponent_attack.shot_feasibility))
            opponent_tail_history.append(float(previous_opponent_attack.tail_hold_score))
        return (
            tracking_history,
            opponent_tracking_history,
            shot_history,
            opponent_shot_history,
            tail_history,
            opponent_tail_history,
        )

    def _component_delta_bonus(
        self,
        *,
        self_history: list[float],
        self_current: float,
        opponent_history: list[float],
        opponent_current: float,
        self_improve_weight: float,
        self_worsen_weight: float,
        enemy_improve_weight: float,
        enemy_worsen_weight: float,
        scale: float,
        shape_power: float,
        deadzone: float,
    ) -> float:
        def directional_terms(history: list[float], current: float) -> tuple[float, float]:
            window = [float(v) for v in history if np.isfinite(v)]
            window.append(float(current))
            if len(window) < 2:
                return 0.0, 0.0
            baseline = window[0]
            path_change = 0.0
            for left, right in zip(window[:-1], window[1:], strict=False):
                path_change += abs(right - left)
            if path_change <= 1e-6:
                anti_oscillation = 0.0
            else:
                anti_oscillation = float(np.clip(abs(current - baseline) / path_change, 0.0, 1.0))
            improve = self._scaled_positive_delta_gain(
                current - baseline,
                scale=scale,
                shape_power=shape_power,
                deadzone=deadzone,
            ) * anti_oscillation
            worsen = self._scaled_positive_delta_gain(
                baseline - current,
                scale=scale,
                shape_power=shape_power,
                deadzone=deadzone,
            ) * anti_oscillation
            return improve, worsen

        self_improve, self_worsen = directional_terms(self_history, self_current)
        enemy_improve, enemy_worsen = directional_terms(opponent_history, opponent_current)
        return (
            self_improve_weight * self_improve
            - self_worsen_weight * self_worsen
            - enemy_improve_weight * enemy_improve
            + enemy_worsen_weight * enemy_worsen
        )

    def _zone_boundary_terms(
        self,
        *,
        current_info: dict[str, Any],
        current_phi: float,
        severity_weight: float,
        history_phis: list[float],
    ) -> tuple[float, float, float]:
        cfg = self.config
        improve_score, worsen_score, anti_oscillation = self._window_delta_scores(
            history_values=history_phis,
            current_value=current_phi,
        )
        severity_penalty = (
            cfg.boundary_warning_penalty
            + (cfg.boundary_critical_penalty - cfg.boundary_warning_penalty) * current_phi
        )
        penalty = severity_weight * (
            severity_penalty * current_phi
            + cfg.boundary_delta_worsen_weight * worsen_score
        )
        recovery_bonus = (
            severity_weight
            * cfg.boundary_recovery_bonus_weight
            * cfg.boundary_delta_improve_weight
            * improve_score
            * anti_oscillation
        )
        return float(current_phi), float(penalty), float(recovery_bonus)

    def _ground_boundary_terms(
        self,
        *,
        current_info: dict[str, Any],
        aircraft_state: dict[str, Any],
        geometry_cache: AircraftGeometryCache | None = None,
        severity_weight: float,
        previous_info: dict[str, Any] | None = None,
        previous_aircraft_state: dict[str, Any] | None = None,
        reward_history: dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        arena = current_info.get("arena", {})
        ground_height = float(arena.get("ground_height", 0.0))
        phi_now = self._ground_zone_phi(aircraft_state, ground_height, geometry_cache=geometry_cache)
        history_phis = self._phi_history_values(
            current_info=current_info,
            previous_info=previous_info,
            previous_aircraft_state=previous_aircraft_state,
            reward_history=reward_history,
            cache_key="ground",
            phi_fn=lambda info, state: self._ground_zone_phi(state, float(info.get("arena", {}).get("ground_height", 0.0))),
        )
        return self._zone_boundary_terms(
            current_info=current_info,
            current_phi=phi_now,
            severity_weight=severity_weight,
            history_phis=history_phis,
        )

    def _ceiling_boundary_terms(
        self,
        *,
        current_info: dict[str, Any],
        aircraft_state: dict[str, Any],
        geometry_cache: AircraftGeometryCache | None = None,
        severity_weight: float,
        previous_info: dict[str, Any] | None = None,
        previous_aircraft_state: dict[str, Any] | None = None,
        reward_history: dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        arena = current_info.get("arena", {})
        ceiling_height = float(arena.get("flight_ceiling_height", 0.0))
        phi_now = self._ceiling_zone_phi(aircraft_state, ceiling_height, geometry_cache=geometry_cache)
        history_phis = self._phi_history_values(
            current_info=current_info,
            previous_info=previous_info,
            previous_aircraft_state=previous_aircraft_state,
            reward_history=reward_history,
            cache_key="ceiling",
            phi_fn=lambda info, state: self._ceiling_zone_phi(
                state,
                float(info.get("arena", {}).get("flight_ceiling_height", 0.0)),
            ),
        )
        return self._zone_boundary_terms(
            current_info=current_info,
            current_phi=phi_now,
            severity_weight=severity_weight,
            history_phis=history_phis,
        )

    def _horizontal_boundary_terms(
        self,
        *,
        current_info: dict[str, Any],
        aircraft_state: dict[str, Any],
        geometry_cache: AircraftGeometryCache | None = None,
        severity_weight: float,
        previous_info: dict[str, Any] | None = None,
        previous_aircraft_state: dict[str, Any] | None = None,
        reward_history: dict[str, Any] | None = None,
    ) -> tuple[float, float, float]:
        arena = current_info.get("arena", {})
        arena_radius = float(arena.get("arena_radius", 0.0))
        phi_now = self._horizontal_zone_phi(aircraft_state, arena_radius, geometry_cache=geometry_cache)
        history_phis = self._phi_history_values(
            current_info=current_info,
            previous_info=previous_info,
            previous_aircraft_state=previous_aircraft_state,
            reward_history=reward_history,
            cache_key="horizontal",
            phi_fn=lambda info, state: self._horizontal_zone_phi(
                state,
                float(info.get("arena", {}).get("arena_radius", 0.0)),
            ),
        )
        return self._zone_boundary_terms(
            current_info=current_info,
            current_phi=phi_now,
            severity_weight=severity_weight,
            history_phis=history_phis,
        )

    def _out_of_bounds_time_penalty(self, out_of_bounds_seconds: float) -> float:
        cfg = self.config
        if out_of_bounds_seconds <= 0.0:
            return 0.0
        alpha = _smoothstep01(
            np.clip(
                out_of_bounds_seconds / max(cfg.out_of_bounds_time_penalty_reference_seconds, 1e-6),
                0.0,
                1.0,
            )
        )
        return float(
            cfg.out_of_bounds_time_penalty_base
            + cfg.out_of_bounds_time_penalty_extra * alpha
        )

    def _hit_enemy_bonus(self, current_info: dict[str, Any], enemy_role: str) -> float:
        events = current_info.get("events_since_last_step")
        if not isinstance(events, list):
            events = current_info.get("step_events")
        if not isinstance(events, list):
            return 0.0
        hit_count = 0
        enemy_prefix = f"{enemy_role}:"
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("kind") != "Hit":
                continue
            subject = event.get("subject")
            if not isinstance(subject, str):
                continue
            if subject == enemy_role or subject.startswith(enemy_prefix):
                hit_count += 1
        return self.config.hit_enemy_bonus_weight * float(hit_count)

    def _got_hit_penalty(self, current_info: dict[str, Any], ego_role: str) -> float:
        events = current_info.get("events_since_last_step")
        if not isinstance(events, list):
            events = current_info.get("step_events")
        if not isinstance(events, list):
            return 0.0
        hit_count = 0
        ego_prefix = f"{ego_role}:"
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("kind") != "Hit":
                continue
            subject = event.get("subject")
            if not isinstance(subject, str):
                continue
            if subject == ego_role or subject.startswith(ego_prefix):
                hit_count += 1
        return self.config.got_hit_penalty_weight * float(hit_count)

    def _collision_penalties(
        self,
        current_info: dict[str, Any],
        ego_role: str,
        target_distance_meters: float,
    ) -> tuple[float, float]:
        events = current_info.get("events_since_last_step")
        if not isinstance(events, list):
            events = current_info.get("step_events")
        if not isinstance(events, list):
            return 0.0, 0.0

        ego_prefix = f"{ego_role}:"
        aircraft_collision_detected = False
        surface_collision_detected = False
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("kind") != "Collision":
                continue
            subject = event.get("subject")
            if not isinstance(subject, str):
                continue
            if subject == ego_role or subject.startswith(ego_prefix):
                detail = event.get("event_detail")
                other_subject = event.get("other_subject")
                if detail == "aircraft" or isinstance(other_subject, str):
                    aircraft_collision_detected = True
                else:
                    surface_collision_detected = True

        if not aircraft_collision_detected and not surface_collision_detected:
            return 0.0, 0.0
        if aircraft_collision_detected:
            return self.config.aircraft_collision_penalty_weight, 0.0
        if surface_collision_detected:
            return 0.0, self.config.surface_collision_penalty_weight
        if target_distance_meters <= 2.0 * self.config.aircraft_collision_radius_meters:
            return self.config.aircraft_collision_penalty_weight, 0.0
        return 0.0, self.config.surface_collision_penalty_weight

    def _active_collision_boxes(self, aircraft_state: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        position = np.asarray(aircraft_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        rotation = quat_to_rotation_matrix(aircraft_state.get("orientation_quat", [0.0, 0.0, 0.0, 1.0]))
        destroyed_subsystems = {
            str(entry.get("name"))
            for entry in aircraft_state.get("subsystems", [])
            if str(entry.get("stage", "")) == "Destroyed"
        }
        boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for name, local_center, half_extents in LOCAL_COLLISION_BOXES:
            if name in destroyed_subsystems:
                continue
            world_center = position + rotation @ local_center
            boxes.append((world_center.astype(np.float32), rotation.astype(np.float32), half_extents.astype(np.float32)))
        return boxes

    def _active_collision_boxes_from_cache(
        self,
        geometry_cache: AircraftGeometryCache,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        return geometry_cache.active_collision_boxes or ()

    def _box_support_radius(
        self,
        rotation: np.ndarray,
        half_extents: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        direction = np.asarray(direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return float(np.max(half_extents))
        direction = direction / norm
        axes = (rotation[:, 0], rotation[:, 1], rotation[:, 2])
        return float(sum(float(half_extents[idx]) * abs(float(np.dot(axis, direction))) for idx, axis in enumerate(axes)))

    def _closest_approach(
        self,
        *,
        relative_position: np.ndarray,
        relative_velocity: np.ndarray,
        horizon_seconds: float,
    ) -> tuple[float, float]:
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        if relative_speed_sq <= 1e-6:
            tau_star = 0.0
        else:
            tau_star = -float(np.dot(relative_position, relative_velocity)) / relative_speed_sq
        tau_star = float(np.clip(tau_star, 0.0, horizon_seconds))
        closest_delta = relative_position + relative_velocity * tau_star
        return tau_star, float(np.linalg.norm(closest_delta))

    def _closest_approach_threat(
        self,
        *,
        tau_seconds: float,
        min_clearance_meters: float,
        tau_reference_seconds: float,
        outer_sigma_meters: float,
        core_sigma_meters: float,
        outer_weight: float,
        core_weight: float,
    ) -> float:
        tau_gate = float(np.exp(-tau_seconds / max(tau_reference_seconds, 1e-6)))
        outer_score = float(np.exp(-((min_clearance_meters / max(outer_sigma_meters, 1e-6)) ** 2)))
        core_score = float(np.exp(-((min_clearance_meters / max(core_sigma_meters, 1e-6)) ** 2)))
        return tau_gate * (outer_weight * outer_score + core_weight * core_score)

    def _aircraft_collision_broadphase_threat(
        self,
        *,
        attacker_position: np.ndarray,
        defender_position: np.ndarray,
        relative_velocity: np.ndarray,
        sigma_scale: float,
    ) -> float:
        cfg = self.config
        tau_star, min_center_distance = self._closest_approach(
            relative_position=defender_position - attacker_position,
            relative_velocity=relative_velocity,
            horizon_seconds=cfg.aircraft_collision_horizon_seconds,
        )
        min_clearance = max(
            min_center_distance - (2.0 * AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS),
            0.0,
        )
        return self._closest_approach_threat(
            tau_seconds=tau_star,
            min_clearance_meters=min_clearance,
            tau_reference_seconds=cfg.aircraft_collision_tau_reference_seconds,
            outer_sigma_meters=cfg.aircraft_collision_outer_sigma_meters * sigma_scale,
            core_sigma_meters=cfg.aircraft_collision_core_sigma_meters * sigma_scale,
            outer_weight=cfg.aircraft_collision_outer_weight,
            core_weight=cfg.aircraft_collision_core_weight,
        )

    def _projectile_box_hit_scores(
        self,
        *,
        muzzle_position: np.ndarray,
        bullet_velocity: np.ndarray,
        defender_state: dict[str, Any],
        dynamic_outer_radius: float,
        defender_geometry: AircraftGeometryCache | None = None,
    ) -> tuple[float, float]:
        cfg = self.config
        defender_velocity = (
            defender_geometry.velocity
            if defender_geometry is not None
            else np.asarray(defender_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        relative_velocity = defender_velocity - bullet_velocity
        bullet_speed = float(np.linalg.norm(bullet_velocity))
        if bullet_speed <= 1e-6:
            return 0.0, 0.0

        tau_max = cfg.projectile_max_range_meters / bullet_speed
        outer_best = 0.0
        core_best = 0.0
        subsystem_names = {"LeftWing", "RightWing", "PitchTail", "YawTail", "Engine"}
        if defender_geometry is not None:
            self._ensure_aircraft_fine_geometry(defender_state, defender_geometry)
            destroyed_subsystems = defender_geometry.destroyed_subsystems or frozenset()
            active_collision_boxes = self._active_collision_boxes_from_cache(defender_geometry)
            defender_position = defender_geometry.position
            defender_rotation = (
                defender_geometry.rotation if defender_geometry.rotation is not None else np.eye(3, dtype=np.float32)
            )
        else:
            destroyed_subsystems = {
                str(entry.get("name"))
                for entry in defender_state.get("subsystems", [])
                if str(entry.get("stage", "")) == "Destroyed"
            }
            active_collision_boxes = self._active_collision_boxes(defender_state)
            defender_position = np.asarray(defender_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            defender_rotation = quat_to_rotation_matrix(defender_state.get("orientation_quat", [0.0, 0.0, 0.0, 1.0]))
        for center, rotation, half_extents in active_collision_boxes:
            tau_star, min_center_distance = self._closest_approach(
                relative_position=center - muzzle_position,
                relative_velocity=relative_velocity,
                horizon_seconds=tau_max,
            )
            closest_delta = (center - muzzle_position) + relative_velocity * tau_star
            delta_norm = float(np.linalg.norm(closest_delta))
            if delta_norm <= 1e-6:
                closest_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                closest_direction = closest_delta / delta_norm
            support = self._box_support_radius(rotation, half_extents, -closest_direction)
            aircraft_clearance = max(
                min_center_distance - support - cfg.projectile_aircraft_hit_radius_meters,
                0.0,
            )
            tau_gate = float(np.exp(-tau_star / max(cfg.attack_tau_reference_seconds, 1e-6)))
            outer_score = tau_gate * float(
                np.exp(-((aircraft_clearance / max(dynamic_outer_radius, cfg.projectile_aircraft_hit_radius_meters)) ** 2))
            )
            outer_best = max(outer_best, outer_score)

        for name, local_center, half_extents in LOCAL_COLLISION_BOXES:
            if name not in subsystem_names:
                continue
            if name in destroyed_subsystems:
                continue
            center = defender_position + defender_rotation @ local_center
            tau_star, min_center_distance = self._closest_approach(
                relative_position=center - muzzle_position,
                relative_velocity=relative_velocity,
                horizon_seconds=tau_max,
            )
            closest_delta = (center - muzzle_position) + relative_velocity * tau_star
            delta_norm = float(np.linalg.norm(closest_delta))
            if delta_norm <= 1e-6:
                closest_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                closest_direction = closest_delta / delta_norm
            support = self._box_support_radius(defender_rotation, half_extents, -closest_direction)
            subsystem_clearance = max(
                min_center_distance - support - cfg.projectile_subsystem_hit_radius_meters,
                0.0,
            )
            tau_gate = float(np.exp(-tau_star / max(cfg.attack_tau_reference_seconds, 1e-6)))
            core_score = tau_gate * float(
                np.exp(
                    -(
                        (
                            subsystem_clearance
                            / max(cfg.shot_core_radius_meters, cfg.projectile_subsystem_hit_radius_meters)
                        )
                        ** 2
                    )
                )
            )
            core_best = max(core_best, core_score)
        return outer_best, core_best

    def _projectile_coarse_outer_upper_bound(
        self,
        *,
        muzzle_position: np.ndarray,
        bullet_velocity: np.ndarray,
        defender_position: np.ndarray,
        defender_velocity: np.ndarray,
        dynamic_outer_radius: float,
    ) -> float:
        cfg = self.config
        relative_velocity = defender_velocity - bullet_velocity
        bullet_speed = float(np.linalg.norm(bullet_velocity))
        if bullet_speed <= 1e-6:
            return 0.0
        tau_max = cfg.projectile_max_range_meters / bullet_speed
        tau_star, min_center_distance = self._closest_approach(
            relative_position=defender_position - muzzle_position,
            relative_velocity=relative_velocity,
            horizon_seconds=tau_max,
        )
        clearance = max(
            min_center_distance - SHOT_BROADPHASE_RADIUS_METERS - cfg.projectile_aircraft_hit_radius_meters,
            0.0,
        )
        tau_gate = float(np.exp(-tau_star / max(cfg.attack_tau_reference_seconds, 1e-6)))
        return tau_gate * float(
            np.exp(-((clearance / max(dynamic_outer_radius, cfg.projectile_aircraft_hit_radius_meters)) ** 2))
        )

    def _aircraft_collision_threat(
        self,
        *,
        attacker_state: dict[str, Any],
        defender_state: dict[str, Any],
        attacker_geometry: AircraftGeometryCache | None = None,
        defender_geometry: AircraftGeometryCache | None = None,
    ) -> float:
        cfg = self.config
        attacker_position = (
            attacker_geometry.position
            if attacker_geometry is not None
            else np.asarray(attacker_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        defender_position = (
            defender_geometry.position
            if defender_geometry is not None
            else np.asarray(defender_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        attacker_velocity = (
            attacker_geometry.velocity
            if attacker_geometry is not None
            else np.asarray(attacker_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        defender_velocity = (
            defender_geometry.velocity
            if defender_geometry is not None
            else np.asarray(defender_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        )
        relative_velocity = defender_velocity - attacker_velocity
        attacker_turn_radius = (
            attacker_geometry.pullup_turn_radius_m
            if attacker_geometry is not None
            else self._state_pullup_turn_radius(attacker_state)
        )
        defender_turn_radius = (
            defender_geometry.pullup_turn_radius_m
            if defender_geometry is not None
            else self._state_pullup_turn_radius(defender_state)
        )
        sigma_scale = (
            ((attacker_turn_radius + defender_turn_radius) * 0.5)
            / max(cfg.combat_turn_radius_meters, 1e-6)
        ) ** cfg.aircraft_collision_dynamic_sigma_power
        sigma_scale = float(np.clip(sigma_scale, 0.75, 1.60))
        broadphase_threat = self._aircraft_collision_broadphase_threat(
            attacker_position=attacker_position,
            defender_position=defender_position,
            relative_velocity=relative_velocity,
            sigma_scale=sigma_scale,
        )
        if broadphase_threat <= AIRCRAFT_COLLISION_BROADPHASE_FINE_THRESHOLD:
            return cfg.aircraft_collision_threat_weight * broadphase_threat
        max_threat = 0.0
        if attacker_geometry is not None:
            self._ensure_aircraft_fine_geometry(attacker_state, attacker_geometry)
            attacker_boxes = self._active_collision_boxes_from_cache(attacker_geometry)
        else:
            attacker_boxes = self._active_collision_boxes(attacker_state)
        if defender_geometry is not None:
            self._ensure_aircraft_fine_geometry(defender_state, defender_geometry)
            defender_boxes = self._active_collision_boxes_from_cache(defender_geometry)
        else:
            defender_boxes = self._active_collision_boxes(defender_state)
        for center_a, rotation_a, half_extents_a in attacker_boxes:
            for center_b, rotation_b, half_extents_b in defender_boxes:
                tau_star, min_center_distance = self._closest_approach(
                    relative_position=center_b - center_a,
                    relative_velocity=relative_velocity,
                    horizon_seconds=cfg.aircraft_collision_horizon_seconds,
                )
                closest_delta = (center_b - center_a) + relative_velocity * tau_star
                if float(np.linalg.norm(closest_delta)) <= 1e-6:
                    closest_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    closest_direction = closest_delta / max(float(np.linalg.norm(closest_delta)), 1e-6)
                support_a = self._box_support_radius(rotation_a, half_extents_a, closest_direction)
                support_b = self._box_support_radius(rotation_b, half_extents_b, -closest_direction)
                min_clearance = max(min_center_distance - (support_a + support_b), 0.0)
                threat = self._closest_approach_threat(
                    tau_seconds=tau_star,
                    min_clearance_meters=min_clearance,
                    tau_reference_seconds=cfg.aircraft_collision_tau_reference_seconds,
                    outer_sigma_meters=cfg.aircraft_collision_outer_sigma_meters * sigma_scale,
                    core_sigma_meters=cfg.aircraft_collision_core_sigma_meters * sigma_scale,
                    outer_weight=cfg.aircraft_collision_outer_weight,
                    core_weight=cfg.aircraft_collision_core_weight,
                )
                max_threat = max(max_threat, threat)
        return cfg.aircraft_collision_threat_weight * max_threat

    def _centered_cone_score(self, cos_value: float, threshold_cos: float) -> float:
        if cos_value <= threshold_cos:
            return 0.0
        alpha = (cos_value - threshold_cos) / max(1.0 - threshold_cos, 1e-6)
        return float(alpha * alpha)

    def _compute_attack_advantage(
        self,
        *,
        attacker_state: dict[str, Any],
        defender_state: dict[str, Any],
        attacker_geometry: AircraftGeometryCache | None = None,
        defender_geometry: AircraftGeometryCache | None = None,
    ) -> AttackAdvantageComponents:
        cfg = self.config
        del attacker_geometry, defender_geometry
        metrics = compute_attack_geometry(
            attacker_state=attacker_state,
            defender_state=defender_state,
        )
        attack_advantage = (
            cfg.shot_feasibility_weight * metrics.shot_feasibility
            + cfg.tail_hold_weight * metrics.tail_hold_score
            + cfg.tracking_quality_weight * metrics.tracking_quality
        )

        return AttackAdvantageComponents(
            tau_seconds=metrics.tau_seconds,
            min_distance_meters=metrics.min_distance_meters,
            tau_gate=metrics.tau_gate,
            fire_alignment_score=metrics.fire_alignment,
            shot_coarse_upper_bound=max(metrics.shot_outer_score, metrics.shot_core_score),
            shot_outer_score=metrics.shot_outer_score,
            shot_core_score=metrics.shot_core_score,
            shot_feasibility=metrics.shot_feasibility,
            tracking_quality=metrics.tracking_quality,
            tail_hold_score=metrics.tail_hold_score,
            attack_advantage=float(attack_advantage),
        )

    def compute(
        self,
        *,
        previous_info: dict[str, Any] | None,
        previous_previous_action_cont: np.ndarray | None,
        previous_action_cont: np.ndarray | None,
        reward_history: dict[str, Any] | None = None,
        current_info: dict[str, Any],
        current_obs: np.ndarray,
        action_cont: np.ndarray,
        action_bin: np.ndarray,
    ) -> PolicyRewardBreakdown:
        cfg = self.config

        current_distance_raw = current_info.get("target_distance")
        current_distance_meters = (
            float(current_distance_raw) if current_distance_raw is not None else cfg.combat_range_tolerance_meters
        )
        distance_band = self._distance_band_reward(current_distance_meters)

        stall_factor = float(current_obs[FIELD_SLICES["self_stall_factor"]][0])
        out_of_bounds_active = float(current_obs[FIELD_SLICES["self_out_of_bounds_active"]][0])
        repair_active_state = float(current_obs[FIELD_SLICES["self_repair_active"]][0])
        gun_overheated = float(current_obs[FIELD_SLICES["self_gun_overheated"]][0])
        gun_heat = float(current_obs[FIELD_SLICES["self_gun_heat_norm"]][0])
        self_throttle_norm = float(current_obs[FIELD_SLICES["self_throttle_norm"]][0])
        self_health_state = np.asarray(current_obs[FIELD_SLICES["self_health_state_norm"]], dtype=np.float32)

        if float(np.max(np.abs(self_health_state))) <= 1e-6:
            self_total_health = 1.0
            self_subsystem_health = np.ones((5,), dtype=np.float32)
        else:
            self_total_health = float(self_health_state[0])
            self_subsystem_health = np.asarray(self_health_state[1:], dtype=np.float32)
        destroyed_subsystem_count = float(np.count_nonzero(self_subsystem_health <= 1e-3))

        self_state = current_info["aircraft_by_role"][current_info["ego_role"]]
        enemy_state = current_info["aircraft_by_role"][current_info["enemy_role"]]
        line_of_sight_body = self._compute_line_of_sight_body(self_state, enemy_state)
        out_of_bounds_seconds = float(self_state.get("out_of_bounds_seconds", 0.0))
        previous_self_state = None if previous_info is None else previous_info.get("aircraft_by_role", {}).get(current_info["ego_role"])
        geometry_cache_by_id: dict[int, AircraftGeometryCache] = {}
        self_geometry = self._aircraft_geometry_cache(self_state, geometry_cache_by_id)
        enemy_geometry = self._aircraft_geometry_cache(enemy_state, geometry_cache_by_id)

        self_forward = self_geometry.forward
        self_speed_mps = self_geometry.speed_mps

        self_attack = self._compute_attack_advantage(
            attacker_state=self_state,
            defender_state=enemy_state,
            attacker_geometry=self_geometry,
            defender_geometry=enemy_geometry,
        )
        opponent_attack = self._compute_attack_advantage(
            attacker_state=enemy_state,
            defender_state=self_state,
            attacker_geometry=enemy_geometry,
            defender_geometry=self_geometry,
        )
        current_info[ATTACK_HISTORY_CACHE_KEY] = {
            "self_attack": self_attack,
            "opponent_attack": opponent_attack,
        }
        previous_self_attack: AttackAdvantageComponents | None = None
        previous_opponent_attack: AttackAdvantageComponents | None = None
        previous_attack_cache = self._cached_attack_history_components_from_info(previous_info)
        if previous_attack_cache is not None:
            previous_self_attack, previous_opponent_attack = previous_attack_cache
        elif previous_info is not None:
            previous_aircraft_by_role = previous_info.get("aircraft_by_role", {})
            previous_self_role_state = previous_aircraft_by_role.get(current_info["ego_role"])
            previous_enemy_role_state = previous_aircraft_by_role.get(current_info["enemy_role"])
            if isinstance(previous_self_role_state, dict) and isinstance(previous_enemy_role_state, dict):
                previous_self_attack = self._compute_attack_advantage(
                    attacker_state=previous_self_role_state,
                    defender_state=previous_enemy_role_state,
                )
                previous_opponent_attack = self._compute_attack_advantage(
                    attacker_state=previous_enemy_role_state,
                    defender_state=previous_self_role_state,
                )
        attack_advantage = cfg.attack_advantage_weight * self_attack.attack_advantage
        threat_advantage = cfg.threat_advantage_weight * opponent_attack.attack_advantage
        (
            tracking_history,
            opponent_tracking_history,
            shot_history,
            opponent_shot_history,
            tail_history,
            opponent_tail_history,
        ) = self._attack_component_histories(
            reward_history=reward_history,
            current_info=current_info,
            previous_self_attack=previous_self_attack,
            previous_opponent_attack=previous_opponent_attack,
        )
        tracking_delta_bonus = self._component_delta_bonus(
            self_history=tracking_history,
            self_current=self_attack.tracking_quality,
            opponent_history=opponent_tracking_history,
            opponent_current=opponent_attack.tracking_quality,
            self_improve_weight=cfg.tracking_delta_self_improve_weight,
            self_worsen_weight=cfg.tracking_delta_self_worsen_weight,
            enemy_improve_weight=cfg.tracking_delta_enemy_improve_weight,
            enemy_worsen_weight=cfg.tracking_delta_enemy_worsen_weight,
            scale=cfg.tracking_delta_scale,
            shape_power=cfg.tracking_delta_shape_power,
            deadzone=cfg.tracking_delta_deadzone,
        )
        shot_delta_bonus = self._component_delta_bonus(
            self_history=shot_history,
            self_current=self_attack.shot_feasibility,
            opponent_history=opponent_shot_history,
            opponent_current=opponent_attack.shot_feasibility,
            self_improve_weight=cfg.shot_delta_self_improve_weight,
            self_worsen_weight=cfg.shot_delta_self_worsen_weight,
            enemy_improve_weight=cfg.shot_delta_enemy_improve_weight,
            enemy_worsen_weight=cfg.shot_delta_enemy_worsen_weight,
            scale=cfg.shot_delta_scale,
            shape_power=cfg.shot_delta_shape_power,
            deadzone=cfg.shot_delta_deadzone,
        )
        tail_delta_bonus = self._component_delta_bonus(
            self_history=tail_history,
            self_current=self_attack.tail_hold_score,
            opponent_history=opponent_tail_history,
            opponent_current=opponent_attack.tail_hold_score,
            self_improve_weight=cfg.tail_delta_self_improve_weight,
            self_worsen_weight=cfg.tail_delta_self_worsen_weight,
            enemy_improve_weight=cfg.tail_delta_enemy_improve_weight,
            enemy_worsen_weight=cfg.tail_delta_enemy_worsen_weight,
            scale=cfg.tail_delta_scale,
            shape_power=cfg.tail_delta_shape_power,
            deadzone=cfg.tail_delta_deadzone,
        )
        aircraft_collision_threat = self._aircraft_collision_threat(
            attacker_state=self_state,
            defender_state=enemy_state,
            attacker_geometry=self_geometry,
            defender_geometry=enemy_geometry,
        )
        one_circle_gate, two_circle_gate = self._merge_circle_gates(
            self_state=self_state,
            enemy_state=enemy_state,
        )

        maneuver_command = np.asarray(action_cont, dtype=np.float32)[1:4]
        pitch_command = float(maneuver_command[0])
        los_forward_cos = float(line_of_sight_body[2])
        maneuver_alignment_score = self._centered_cone_score(los_forward_cos, cfg.fire_alignment_threshold_cos)
        desired_pitch = -float(line_of_sight_body[1])
        desired_lateral = float(line_of_sight_body[0])
        pitch_alignment = max(pitch_command * desired_pitch, 0.0)
        yaw_alignment = max(float(maneuver_command[2]) * desired_lateral, 0.0)
        roll_alignment = max(-float(maneuver_command[1]) * desired_lateral, 0.0)
        maneuver_activity_raw = (
            0.45 * abs(float(maneuver_command[0]))
            + 0.40 * abs(float(maneuver_command[2]))
            + 0.15 * abs(float(maneuver_command[1]))
        )
        maneuver_alignment = float(0.45 * pitch_alignment + 0.40 * yaw_alignment + 0.15 * roll_alignment)
        maneuver_activity = cfg.maneuver_activity_weight * maneuver_alignment_score * maneuver_alignment * maneuver_activity_raw
        upward_tracking_gate = float(np.clip(float(line_of_sight_body[1]) + 0.15, 0.0, 1.0))
        pitch_up_tracking = cfg.pitch_up_tracking_weight * upward_tracking_gate * (-pitch_command)
        self._ensure_aircraft_fine_geometry(self_state, self_geometry)
        flat_roll_bonus = cfg.flat_roll_bonus_weight * _flat_roll_score_from_rotation(self_geometry.rotation)

        fire_command = float(action_bin[1])
        fire_command_bonus = cfg.fire_command_bonus_weight * fire_command
        brake_command = float(action_bin[0])
        throttle_command = float(np.asarray(action_cont, dtype=np.float32)[0])
        brake_penalty = cfg.brake_penalty_weight * brake_command
        previous_throttle_for_bonus = throttle_command
        if previous_action_cont is not None:
            previous_throttle_for_bonus = float(np.asarray(previous_action_cont, dtype=np.float32)[0])
        throttle_change_alpha = _smoothstep01(
            abs(throttle_command - previous_throttle_for_bonus) / max(cfg.throttle_change_reference, 1e-6)
        )
        throttle_change_bonus = cfg.throttle_change_bonus_weight * throttle_change_alpha
        throttle_low_penalty_alpha = float(
            np.clip(
                (cfg.throttle_low_penalty_threshold - self_throttle_norm)
                / max(cfg.throttle_low_penalty_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
        throttle_low_penalty = cfg.throttle_low_penalty_weight * (throttle_low_penalty_alpha**3)
        low_speed_penalty = self._low_speed_penalty(self_speed_mps)
        previous_speed_mps = (
            self_speed_mps
            if not isinstance(previous_self_state, dict)
            else self._aircraft_speed_mps(previous_self_state)
        )
        self_position_world = self_geometry.position
        enemy_position_world = enemy_geometry.position
        self_velocity_world = self_geometry.velocity
        enemy_velocity_world = enemy_geometry.velocity
        speed_delta_mps = self_speed_mps - previous_speed_mps
        closing_speed_mps = self._closing_speed_mps(
            attacker_position=self_position_world,
            attacker_velocity=self_velocity_world,
            defender_position=enemy_position_world,
            defender_velocity=enemy_velocity_world,
        )
        _, _, one_circle_speed_score, two_circle_speed_score = self._merge_speed_rewards(
            self_state=self_state,
            self_speed_mps=self_speed_mps,
        )
        one_circle_speed_reward = cfg.one_circle_speed_reward_weight * one_circle_gate * one_circle_speed_score
        two_circle_speed_reward = cfg.two_circle_speed_reward_weight * two_circle_gate * two_circle_speed_score
        close_distance_gate = float(
            np.clip(
                1.0 - current_distance_meters / max(2.0 * cfg.combat_turn_radius_meters, 1e-6),
                0.0,
                1.0,
            )
        )
        opportunity_relief_gate = max(
            self_attack.tail_hold_score * close_distance_gate,
            opponent_attack.tail_hold_score * close_distance_gate,
            float(np.clip(out_of_bounds_active, 0.0, 1.0)),
        )
        speed_jitter_penalty = 0.0
        if reward_history is not None:
            speed_jitter_penalty = self._speed_jitter_penalty_from_history(
                reward_history=reward_history,
                ego_role=current_info["ego_role"],
                current_throttle=throttle_command,
                current_brake=brake_command,
                current_speed_mps=self_speed_mps,
                opportunity_relief_gate=opportunity_relief_gate,
            )
        elif previous_action_cont is not None and previous_previous_action_cont is not None:
            previous_throttle = float(np.asarray(previous_action_cont, dtype=np.float32)[0])
            previous_previous_throttle = float(np.asarray(previous_previous_action_cont, dtype=np.float32)[0])
            speed_jitter_penalty = self._speed_jitter_penalty(
                current_throttle=throttle_command,
                previous_throttle=previous_throttle,
                previous_previous_throttle=previous_previous_throttle,
                current_speed_mps=self_speed_mps,
                previous_speed_mps=previous_speed_mps,
                opportunity_relief_gate=opportunity_relief_gate,
            )

        pitch_jitter_penalty = self._action_axis_jitter_penalty_from_history(
            reward_history=reward_history,
            current_action_cont=action_cont,
            axis_index=1,
            weight=cfg.pitch_jitter_penalty_weight,
            opportunity_relief_gate=opportunity_relief_gate,
            previous_previous_action_cont=previous_previous_action_cont,
            previous_action_cont=previous_action_cont,
        )
        roll_jitter_penalty = self._action_axis_jitter_penalty_from_history(
            reward_history=reward_history,
            current_action_cont=action_cont,
            axis_index=2,
            weight=cfg.roll_jitter_penalty_weight,
            opportunity_relief_gate=opportunity_relief_gate,
            previous_previous_action_cont=previous_previous_action_cont,
            previous_action_cont=previous_action_cont,
        )
        yaw_jitter_penalty = self._action_axis_jitter_penalty_from_history(
            reward_history=reward_history,
            current_action_cont=action_cont,
            axis_index=3,
            weight=cfg.yaw_jitter_penalty_weight,
            opportunity_relief_gate=opportunity_relief_gate,
            previous_previous_action_cont=previous_previous_action_cont,
            previous_action_cont=previous_action_cont,
        )

        stall_penalty = cfg.stall_penalty_weight * stall_factor
        overheat_penalty = cfg.overheat_penalty_weight * (0.5 * gun_heat + gun_overheated)

        repair_command = float(action_bin[2])
        repair_active = repair_active_state >= 0.5
        repair_engaged = max(repair_command, 1.0 if repair_active else 0.0)
        repair_elapsed_seconds = float(self_state.get("repair_elapsed_seconds", 0.0))

        repair_static_penalty = cfg.repair_static_penalty_weight * repair_command
        repair_twitch_penalty = cfg.repair_twitch_penalty_weight * repair_command * (1.0 if not repair_active else 0.0)

        repair_high_health_penalty = 0.0
        if repair_engaged > 0.0 and self_total_health > 0.7:
            high_health_alpha = (self_total_health - 0.7) / 0.3
            repair_high_health_penalty = (
                cfg.repair_high_health_penalty_weight * repair_engaged * float(high_health_alpha * high_health_alpha)
            )
        arena = current_info.get("arena", {})
        self_position = self_geometry.position
        self_velocity = self_geometry.velocity
        self_stall_factor = self_geometry.stall_factor
        previous_self_position = None
        previous_self_velocity = None
        if isinstance(previous_self_state, dict):
            previous_self_position = np.asarray(previous_self_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            previous_self_velocity = np.asarray(
                previous_self_state.get("linear_velocity", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
        ground_height = float(arena.get("ground_height", 0.0))
        ceiling_height = float(arena.get("flight_ceiling_height", 0.0))
        arena_radius = float(arena.get("arena_radius", 0.0))
        ground_clearance = float(self_position[1] - ground_height)
        ceiling_clearance = float(ceiling_height - self_position[1])
        ground_boundary_threat, ground_boundary_penalty, ground_recovery_bonus = self._ground_boundary_terms(
            current_info=current_info,
            aircraft_state=self_state,
            geometry_cache=self_geometry,
            severity_weight=cfg.ground_boundary_severity_weight,
            previous_info=previous_info if isinstance(previous_info, dict) else None,
            previous_aircraft_state=previous_self_state if isinstance(previous_self_state, dict) else None,
            reward_history=reward_history,
        )
        ceiling_boundary_threat, ceiling_boundary_penalty, ceiling_recovery_bonus = self._ceiling_boundary_terms(
            current_info=current_info,
            aircraft_state=self_state,
            geometry_cache=self_geometry,
            severity_weight=cfg.ceiling_boundary_severity_weight,
            previous_info=previous_info if isinstance(previous_info, dict) else None,
            previous_aircraft_state=previous_self_state if isinstance(previous_self_state, dict) else None,
            reward_history=reward_history,
        )
        horizontal_boundary_threat = 0.0
        horizontal_boundary_penalty = 0.0
        horizontal_recovery_bonus = 0.0
        if arena_radius > 1e-6:
            (
                horizontal_boundary_threat,
                horizontal_boundary_penalty,
                horizontal_recovery_bonus,
            ) = self._horizontal_boundary_terms(
                current_info=current_info,
                aircraft_state=self_state,
                geometry_cache=self_geometry,
                severity_weight=cfg.horizontal_boundary_severity_weight,
                previous_info=previous_info if isinstance(previous_info, dict) else None,
                previous_aircraft_state=previous_self_state if isinstance(previous_self_state, dict) else None,
                reward_history=reward_history,
            )
        boundary_recovery_bonus = ground_recovery_bonus + ceiling_recovery_bonus + horizontal_recovery_bonus
        out_of_bounds_time_penalty = self._out_of_bounds_time_penalty(out_of_bounds_seconds)
        current_info[BOUNDARY_PHI_CACHE_KEY] = {
            "ground": float(ground_boundary_threat),
            "ceiling": float(ceiling_boundary_threat),
            "horizontal": float(horizontal_boundary_threat),
        }
        boundary_threat_total = (
            ground_boundary_threat + ceiling_boundary_threat + horizontal_boundary_threat
        )
        boundary_penalty_total = (
            ground_boundary_penalty + ceiling_boundary_penalty + horizontal_boundary_penalty
        )
        boundary_combat_gate = float(
            np.exp(-boundary_threat_total / max(cfg.boundary_combat_suppression_reference, 1e-6))
        )

        repair_low_health_bonus = 0.0
        if repair_engaged > 0.0 and self_total_health < 0.6:
            low_health_alpha = (0.6 - self_total_health) / 0.6
            repair_low_health_bonus = (
                cfg.repair_low_health_bonus_weight * repair_engaged * float(low_health_alpha * low_health_alpha)
            )
        repair_destroyed_subsystem_bonus = (
            cfg.repair_destroyed_subsystem_bonus_weight * repair_engaged * destroyed_subsystem_count
        )
        repair_threat_gate = self._repair_threat_gate(
            distance_meters=current_distance_meters,
            opponent_tracking_quality=opponent_attack.tracking_quality,
        )
        repair_boundary_gate = float(np.exp(-boundary_penalty_total / max(cfg.repair_boundary_reference, 1e-6)))
        repair_stall_gate = float(np.exp(-stall_factor / max(cfg.repair_stall_reference, 1e-6)))
        repair_completion_safety_gate = self._repair_completion_safety_gate(
            aircraft_state=self_state,
            geometry_cache=self_geometry,
            arena=arena,
            repair_engaged=repair_engaged,
            repair_active=repair_active,
            repair_elapsed_seconds=repair_elapsed_seconds,
            aircraft_collision_threat=aircraft_collision_threat,
        )
        repair_opportunity_gate = (
            repair_threat_gate
            * repair_boundary_gate
            * repair_stall_gate
            * repair_completion_safety_gate
        )
        repair_low_health_bonus *= repair_opportunity_gate
        repair_destroyed_subsystem_bonus *= repair_opportunity_gate
        repair_under_threat_penalty = (
            cfg.repair_under_threat_penalty_weight * repair_engaged * (1.0 - repair_opportunity_gate)
        )

        if bool(self_state["destroyed"]):
            repair_twitch_penalty = 0.0
            repair_static_penalty = 0.0
            repair_high_health_penalty = 0.0
            repair_under_threat_penalty = 0.0
            repair_low_health_bonus = 0.0
            repair_destroyed_subsystem_bonus = 0.0
            repair_opportunity_gate = 0.0
            repair_completion_safety_gate = 0.0

        attack_advantage *= boundary_combat_gate
        maneuver_activity *= boundary_combat_gate
        predictive_fire_hit_bonus = cfg.predictive_fire_hit_bonus_weight * fire_command * self_attack.shot_feasibility
        predictive_fire_hit_bonus *= boundary_combat_gate
        fire_opportunity_score = float(
            cfg.fire_shot_weight * self_attack.shot_feasibility
            + cfg.fire_alignment_weight * self_attack.fire_alignment_score
            + cfg.fire_tail_weight * self_attack.tail_hold_score
        )
        fire_window_bonus = cfg.fire_window_bonus_weight * fire_command * fire_opportunity_score
        fire_window_bonus *= boundary_combat_gate
        fire_hesitation_penalty = cfg.fire_hesitation_penalty_weight * (1.0 - fire_command) * fire_opportunity_score
        fire_hesitation_penalty *= boundary_combat_gate
        hit_enemy_bonus = self._hit_enemy_bonus(current_info, current_info["enemy_role"])
        hit_enemy_bonus *= boundary_combat_gate
        got_hit_penalty = self._got_hit_penalty(current_info, current_info["ego_role"])
        aircraft_collision_penalty, surface_collision_penalty = self._collision_penalties(
            current_info,
            current_info["ego_role"],
            current_distance_meters,
        )

        prev_sim_time = None if previous_info is None else previous_info.get("sim_time_seconds")
        cur_sim_time = current_info.get("sim_time_seconds")
        dt_seconds = 1.0 / 60.0
        if prev_sim_time is not None and cur_sim_time is not None:
            measured_dt = float(cur_sim_time) - float(prev_sim_time)
            if measured_dt > 0.0:
                dt_seconds = measured_dt
        positive_reward_decay_scale = self._positive_continuous_reward_decay_scale(cur_sim_time)
        negative_reward_decay_scale = self._negative_continuous_reward_decay_scale(cur_sim_time)
        time_pressure_scale = self._time_pressure_scale(cur_sim_time)
        time_pressure = 0.0
        if not bool(self_state["destroyed"]):
            time_pressure_rate = (
                (1.0 - time_pressure_scale) * cfg.time_pressure_initial_bonus_per_second
                - time_pressure_scale * cfg.time_pressure_rate_per_second
            )
            time_pressure = time_pressure_rate * dt_seconds

        positive_continuous_scale = dt_seconds * positive_reward_decay_scale
        negative_continuous_scale = dt_seconds * negative_reward_decay_scale
        distance_band *= positive_continuous_scale
        attack_advantage *= positive_continuous_scale
        tracking_delta_bonus *= positive_continuous_scale
        shot_delta_bonus *= positive_continuous_scale
        tail_delta_bonus *= positive_continuous_scale
        pitch_up_tracking *= positive_continuous_scale
        maneuver_activity *= positive_continuous_scale
        flat_roll_bonus *= positive_continuous_scale
        throttle_change_bonus *= positive_continuous_scale
        one_circle_speed_reward *= positive_continuous_scale
        two_circle_speed_reward *= positive_continuous_scale
        repair_low_health_bonus *= positive_continuous_scale
        repair_destroyed_subsystem_bonus *= positive_continuous_scale
        boundary_recovery_bonus *= positive_continuous_scale
        predictive_fire_hit_bonus *= positive_continuous_scale
        fire_command_bonus *= positive_continuous_scale
        fire_window_bonus *= positive_continuous_scale

        threat_advantage *= negative_continuous_scale
        brake_penalty *= negative_continuous_scale
        throttle_low_penalty *= negative_continuous_scale
        low_speed_penalty *= negative_continuous_scale
        speed_jitter_penalty *= negative_continuous_scale
        pitch_jitter_penalty *= negative_continuous_scale
        roll_jitter_penalty *= negative_continuous_scale
        yaw_jitter_penalty *= negative_continuous_scale
        stall_penalty *= negative_continuous_scale
        overheat_penalty *= negative_continuous_scale
        repair_twitch_penalty *= negative_continuous_scale
        repair_static_penalty *= negative_continuous_scale
        repair_high_health_penalty *= negative_continuous_scale
        repair_under_threat_penalty *= negative_continuous_scale
        ground_boundary_penalty *= negative_continuous_scale
        ceiling_boundary_penalty *= negative_continuous_scale
        horizontal_boundary_penalty *= negative_continuous_scale
        out_of_bounds_time_penalty *= negative_continuous_scale
        fire_hesitation_penalty *= negative_continuous_scale
        aircraft_collision_threat *= negative_continuous_scale

        self_destroy_penalty = cfg.self_destroy_penalty if bool(self_state["destroyed"]) else 0.0
        enemy_destroy_bonus = cfg.enemy_destroy_bonus if bool(enemy_state["destroyed"]) else 0.0

        total = (
            distance_band
            + attack_advantage
            + tracking_delta_bonus
            + shot_delta_bonus
            + tail_delta_bonus
            + pitch_up_tracking
            + maneuver_activity
            + flat_roll_bonus
            + repair_low_health_bonus
            + repair_destroyed_subsystem_bonus
            + boundary_recovery_bonus
            + predictive_fire_hit_bonus
            + fire_command_bonus
            + fire_window_bonus
            + hit_enemy_bonus
            + throttle_change_bonus
            + one_circle_speed_reward
            + two_circle_speed_reward
            + time_pressure
            - threat_advantage
            - got_hit_penalty
            - fire_hesitation_penalty
            - aircraft_collision_threat
            - aircraft_collision_penalty
            - surface_collision_penalty
            - brake_penalty
            - throttle_low_penalty
            - low_speed_penalty
            - speed_jitter_penalty
            - pitch_jitter_penalty
            - roll_jitter_penalty
            - yaw_jitter_penalty
            - repair_twitch_penalty
            - repair_static_penalty
            - repair_high_health_penalty
            - repair_under_threat_penalty
            - ground_boundary_penalty
            - ceiling_boundary_penalty
            - horizontal_boundary_penalty
            - out_of_bounds_time_penalty
            - stall_penalty
            - overheat_penalty
            - self_destroy_penalty
            + enemy_destroy_bonus
        )

        return PolicyRewardBreakdown(
            total=float(total),
            time_pressure=time_pressure,
            time_pressure_scale=time_pressure_scale,
            positive_reward_decay_scale=positive_reward_decay_scale,
            negative_reward_decay_scale=negative_reward_decay_scale,
            distance_band=distance_band,
            attack_advantage=attack_advantage,
            threat_advantage=threat_advantage,
            tracking_delta_bonus=tracking_delta_bonus,
            shot_delta_bonus=shot_delta_bonus,
            tail_delta_bonus=tail_delta_bonus,
            fire_alignment_score=self_attack.fire_alignment_score,
            shot_coarse_upper_bound=self_attack.shot_coarse_upper_bound,
            shot_feasibility=self_attack.shot_feasibility,
            shot_outer_score=self_attack.shot_outer_score,
            shot_core_score=self_attack.shot_core_score,
            tracking_quality=self_attack.tracking_quality,
            tail_hold_score=self_attack.tail_hold_score,
            attack_tau_seconds=self_attack.tau_seconds,
            opponent_attack_advantage=opponent_attack.attack_advantage,
            opponent_fire_alignment_score=opponent_attack.fire_alignment_score,
            opponent_shot_coarse_upper_bound=opponent_attack.shot_coarse_upper_bound,
            opponent_shot_feasibility=opponent_attack.shot_feasibility,
            opponent_shot_outer_score=opponent_attack.shot_outer_score,
            opponent_shot_core_score=opponent_attack.shot_core_score,
            opponent_tracking_quality=opponent_attack.tracking_quality,
            opponent_tail_hold_score=opponent_attack.tail_hold_score,
            opponent_attack_tau_seconds=opponent_attack.tau_seconds,
            pitch_up_tracking=pitch_up_tracking,
            maneuver_activity=maneuver_activity,
            flat_roll_bonus=flat_roll_bonus,
            brake_penalty=brake_penalty,
            throttle_change_bonus=throttle_change_bonus,
            throttle_low_penalty=throttle_low_penalty,
            low_speed_penalty=low_speed_penalty,
            speed_jitter_penalty=speed_jitter_penalty,
            one_circle_gate=one_circle_gate,
            two_circle_gate=two_circle_gate,
            one_circle_speed_reward=one_circle_speed_reward,
            two_circle_speed_reward=two_circle_speed_reward,
            self_speed_mps=self_speed_mps,
            speed_delta_mps=speed_delta_mps,
            closing_speed_mps=closing_speed_mps,
            pitch_jitter_penalty=pitch_jitter_penalty,
            roll_jitter_penalty=roll_jitter_penalty,
            yaw_jitter_penalty=yaw_jitter_penalty,
            stall_penalty=stall_penalty,
            overheat_penalty=overheat_penalty,
            repair_twitch_penalty=repair_twitch_penalty,
            repair_static_penalty=repair_static_penalty,
            repair_high_health_penalty=repair_high_health_penalty,
            repair_under_threat_penalty=repair_under_threat_penalty,
            repair_low_health_bonus=repair_low_health_bonus,
            repair_destroyed_subsystem_bonus=repair_destroyed_subsystem_bonus,
            repair_opportunity_gate=repair_opportunity_gate,
            repair_completion_safety_gate=repair_completion_safety_gate,
            ground_boundary_threat=ground_boundary_threat,
            ceiling_boundary_threat=ceiling_boundary_threat,
            horizontal_boundary_threat=horizontal_boundary_threat,
            ground_boundary_penalty=ground_boundary_penalty,
            ceiling_boundary_penalty=ceiling_boundary_penalty,
            horizontal_boundary_penalty=horizontal_boundary_penalty,
            boundary_recovery_bonus=boundary_recovery_bonus,
            out_of_bounds_time_penalty=out_of_bounds_time_penalty,
            boundary_combat_gate=boundary_combat_gate,
            predictive_fire_hit_bonus=predictive_fire_hit_bonus,
            fire_command_bonus=fire_command_bonus,
            fire_window_bonus=fire_window_bonus,
            fire_hesitation_penalty=fire_hesitation_penalty,
            hit_enemy_bonus=hit_enemy_bonus,
            got_hit_penalty=got_hit_penalty,
            aircraft_collision_threat=aircraft_collision_threat,
            aircraft_collision_penalty=aircraft_collision_penalty,
            surface_collision_penalty=surface_collision_penalty,
            self_destroy_penalty=self_destroy_penalty,
            enemy_destroy_bonus=enemy_destroy_bonus,
        )
