from __future__ import annotations

import dataclasses

import numpy as np

from dfb_reinforcement_learning.obs.policy_schema import (
    POLICY_OBSERVATION_SCHEMA,
)
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig
from dfb_reinforcement_learning.rewards.policy_reward import AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS


def _orientation_for_forward(forward: list[float]) -> list[float]:
    direction = np.asarray(forward, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, direction), -1.0, 1.0))
    if dot <= -1.0 + 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    xyz = np.cross(source, direction)
    quaternion = np.asarray([xyz[0], xyz[1], xyz[2], 1.0 + dot], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def _field_slices() -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for field in POLICY_OBSERVATION_SCHEMA.fields:
        result[field.name] = slice(offset, offset + field.size)
        offset += field.size
    return result


def _info(
    *,
    target_distance: float,
    sim_time_seconds: float | None = None,
    self_destroyed: bool = False,
    enemy_destroyed: bool = False,
    self_position: list[float] | None = None,
    self_velocity: list[float] | None = None,
    self_forward: list[float] | None = None,
    enemy_position: list[float] | None = None,
    enemy_velocity: list[float] | None = None,
    enemy_forward: list[float] | None = None,
    self_stall_factor: float = 0.0,
    enemy_stall_factor: float = 0.0,
    self_orientation_quat: list[float] | None = None,
    enemy_orientation_quat: list[float] | None = None,
    flight_ceiling_height: float = 300.0,
    self_pullup_turn_radius_m: float | None = None,
    enemy_pullup_turn_radius_m: float | None = None,
    self_max_level_speed_mps: float | None = 80.0,
    enemy_max_level_speed_mps: float | None = 80.0,
    self_time_to_ground_impact_s: float | None = None,
    self_time_to_ceiling_impact_s: float | None = None,
    self_time_to_horizontal_boundary_impact_s: float | None = None,
    self_time_to_reenter_arena_s: float | None = None,
    self_repair_elapsed_seconds: float = 0.0,
    self_out_of_bounds_seconds: float = 0.0,
) -> dict[str, object]:
    return {
        "target_distance": target_distance,
        "sim_time_seconds": 0.0 if sim_time_seconds is None else sim_time_seconds,
        "ego_role": "fighter1",
        "enemy_role": "fighter2",
        "arena": {
            "ground_height": 0.0,
            "arena_radius": 5000.0,
            "flight_ceiling_height": flight_ceiling_height,
            "ceiling_falloff_range": 50.0,
        },
        "aircraft_by_role": {
            "fighter1": {
                "destroyed": self_destroyed,
                "position": self_position or [0.0, 150.0, 0.0],
                "stall_factor": self_stall_factor,
                "orientation_quat": self_orientation_quat
                or _orientation_for_forward(self_forward or [0.0, 0.0, 1.0]),
                "linear_velocity": self_velocity or [0.0, 0.0, 0.0],
                "forward": self_forward or [0.0, 0.0, 1.0],
                "pullup_turn_radius_m": self_pullup_turn_radius_m,
                "max_level_speed_mps": self_max_level_speed_mps,
                "time_to_ground_impact_s": self_time_to_ground_impact_s,
                "time_to_ceiling_impact_s": self_time_to_ceiling_impact_s,
                "time_to_horizontal_boundary_impact_s": self_time_to_horizontal_boundary_impact_s,
                "time_to_reenter_arena_s": self_time_to_reenter_arena_s,
                "repair_elapsed_seconds": self_repair_elapsed_seconds,
                "out_of_bounds_seconds": self_out_of_bounds_seconds,
                "subsystems": [
                    {"name": "LeftWing", "stage": "Intact"},
                    {"name": "RightWing", "stage": "Intact"},
                    {"name": "PitchTail", "stage": "Intact"},
                    {"name": "YawTail", "stage": "Intact"},
                    {"name": "Engine", "stage": "Intact"},
                ],
            },
            "fighter2": {
                "destroyed": enemy_destroyed,
                "position": enemy_position or [0.0, 150.0, target_distance],
                "stall_factor": enemy_stall_factor,
                "orientation_quat": enemy_orientation_quat
                or _orientation_for_forward(enemy_forward or [0.0, 0.0, -1.0]),
                "linear_velocity": enemy_velocity or [0.0, 0.0, 0.0],
                "forward": enemy_forward or [0.0, 0.0, -1.0],
                "pullup_turn_radius_m": enemy_pullup_turn_radius_m,
                "max_level_speed_mps": enemy_max_level_speed_mps,
                "subsystems": [
                    {"name": "LeftWing", "stage": "Intact"},
                    {"name": "RightWing", "stage": "Intact"},
                    {"name": "PitchTail", "stage": "Intact"},
                    {"name": "YawTail", "stage": "Intact"},
                    {"name": "Engine", "stage": "Intact"},
                ],
            },
        },
    }


def test_reward_increases_when_closing_and_aiming() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            pitch_up_tracking_weight=0.5,
            combat_range_reward=0.2,
            time_pressure_initial_bonus_per_second=1.0,
            threat_advantage_weight=0.0,
            throttle_low_penalty_weight=0.0,
            low_speed_penalty_weight=0.0,
            boundary_warning_penalty=0.0,
            boundary_critical_penalty=0.0,
        )
    )
    reward = composer.compute(
        previous_info=_info(target_distance=500.0, sim_time_seconds=10.0),
        previous_previous_action_cont=np.zeros((4,), dtype=np.float32),
        previous_action_cont=np.zeros((4,), dtype=np.float32),
        current_info=_info(
            target_distance=200.0,
            sim_time_seconds=10.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 200.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.asarray([0.0, -0.6, 0.0, 0.0], dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert reward.total > 0.0
    assert reward.time_pressure > 0.0
    assert reward.distance_band > 0.0
    assert reward.attack_advantage > 0.0
    assert reward.shot_feasibility > 0.0
    assert reward.tracking_quality > 0.0
    assert reward.pitch_up_tracking > 0.0


def test_reward_penalizes_self_destroy_and_oob() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_out_of_bounds_active"]] = 1.0
    obs[slices["self_out_of_bounds_seconds_norm"]] = 1.0
    obs[slices["self_stall_factor"]] = 0.9
    obs[slices["self_gun_overheated"]] = 1.0
    obs[slices["self_gun_heat_norm"]] = 0.8
    obs[slices["self_repair_active"]] = 1.0
    obs[slices["self_repair_seconds_norm"]] = 0.5
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=150.0, self_destroyed=True),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.total < -1.0
    assert reward.self_destroy_penalty > 0.0
    assert reward.repair_twitch_penalty == 0.0
    assert reward.repair_high_health_penalty == 0.0
    assert reward.repair_low_health_bonus == 0.0
    assert reward.stall_penalty > 0.0
    assert reward.overheat_penalty > 0.0
    assert reward.repair_static_penalty == 0.0


def test_flat_roll_bonus_prefers_level_attitude() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(flat_roll_bonus_weight=1.0))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)

    level = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    bank_90 = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            self_orientation_quat=[0.0, 0.0, float(np.sin(np.pi / 4.0)), float(np.cos(np.pi / 4.0))],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    inverted = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            self_orientation_quat=[0.0, 0.0, 1.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )

    assert level.flat_roll_bonus > bank_90.flat_roll_bonus > inverted.flat_roll_bonus
    assert np.isclose(inverted.flat_roll_bonus, 0.0, atol=1e-8)


def test_shot_broadphase_diagnostics_rejects_far_geometry() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    reward = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=1400.0,
            sim_time_seconds=2.0,
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[600.0, 150.0, 1200.0],
            enemy_velocity=[0.0, 0.0, -20.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.shot_coarse_upper_bound < composer.config.shot_broadphase_epsilon
    assert reward.shot_outer_score == 0.0
    assert reward.shot_core_score == 0.0
    assert reward.shot_feasibility == 0.0


def test_out_of_bounds_time_penalty_is_zero_when_inside_arena() -> None:
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=200.0, self_out_of_bounds_seconds=0.0),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.out_of_bounds_time_penalty == 0.0


def test_out_of_bounds_time_penalty_grows_with_elapsed_seconds() -> None:
    composer = PolicyRewardComposer()
    short = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=200.0, self_out_of_bounds_seconds=1.0),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    long = composer.compute(
        previous_info=None,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=200.0, self_out_of_bounds_seconds=20.0),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert short.out_of_bounds_time_penalty > 0.0
    assert long.out_of_bounds_time_penalty > short.out_of_bounds_time_penalty


def test_reward_prefers_pitch_up_when_target_not_below() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    # line_of_sight_body now computed from state positions; enemy at [0, 210, 200] gives y>0
    composer = PolicyRewardComposer(PolicyRewardConfig(pitch_up_tracking_weight=0.5))
    common_info = _info(target_distance=220.0, sim_time_seconds=3.0)
    up = composer.compute(
        previous_info=common_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=3.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.asarray([0.0, -0.8, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    down = composer.compute(
        previous_info=common_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=3.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.8, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert up.pitch_up_tracking > 0.0
    assert down.pitch_up_tracking < 0.0
    assert up.total > down.total


def test_reward_disables_pitch_up_bias_when_target_is_below() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    # line_of_sight_body now computed from state positions; use enemy below ego
    composer = PolicyRewardComposer()
    up = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=3.0, enemy_position=[0.0, 80.0, 220.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=3.0 + (1.0 / 60.0), enemy_position=[0.0, 80.0, 220.0]),
        current_obs=obs,
        action_cont=np.asarray([0.0, -0.8, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    down = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=3.0, enemy_position=[0.0, 80.0, 220.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=3.0 + (1.0 / 60.0), enemy_position=[0.0, 80.0, 220.0]),
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.8, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert up.pitch_up_tracking == 0.0
    assert down.pitch_up_tracking == 0.0


def test_reward_penalizes_rear_threat() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -180.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -180.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.threat_advantage > 0.0


def test_tracking_delta_bonus_rewards_improving_tracking() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tracking_delta_self_improve_weight=0.5,
            tracking_delta_self_worsen_weight=0.0,
            tracking_delta_enemy_improve_weight=0.0,
            tracking_delta_enemy_worsen_weight=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.5, 0.0, 0.8660254],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tracking_delta_bonus > 0.0


def test_tracking_delta_shape_power_amplifies_small_improvements() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.04, 0.0, 0.9991997],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.02, 0.0, 0.99979997],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    base_cfg = PolicyRewardConfig(
        tracking_delta_self_improve_weight=1.0,
        tracking_delta_self_worsen_weight=0.0,
        tracking_delta_enemy_improve_weight=0.0,
        tracking_delta_enemy_worsen_weight=0.0,
        tracking_delta_scale=0.05,
        tracking_delta_deadzone=0.0,
    )
    linearish_reward = PolicyRewardComposer(
        dataclasses.replace(base_cfg, tracking_delta_shape_power=1.0)
    ).compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    shaped_reward = PolicyRewardComposer(
        dataclasses.replace(base_cfg, tracking_delta_shape_power=0.75)
    ).compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert shaped_reward.tracking_delta_bonus > linearish_reward.tracking_delta_bonus > 0.0


def test_tracking_delta_bonus_penalizes_losing_tracking() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tracking_delta_self_improve_weight=0.0,
            tracking_delta_self_worsen_weight=0.7,
            tracking_delta_enemy_improve_weight=0.0,
            tracking_delta_enemy_worsen_weight=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.5, 0.0, 0.8660254],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tracking_delta_bonus < 0.0


def test_tracking_quality_remains_continuous_in_rear_hemisphere() -> None:
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(
            target_distance=float(np.linalg.norm(np.asarray([60.0, 0.0, -120.0], dtype=np.float32))),
            sim_time_seconds=6.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[60.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=float(np.linalg.norm(np.asarray([60.0, 0.0, -120.0], dtype=np.float32))),
            sim_time_seconds=6.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[60.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert 0.0 < reward.tracking_quality < 0.5
    assert reward.fire_alignment_score == 0.0
    assert reward.fire_window_bonus == 0.0


def test_tracking_delta_bonus_rewards_recovering_target_from_rear_hemisphere() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tactical_component_delta_history_length=1,
            tracking_delta_self_improve_weight=0.7,
            tracking_delta_self_worsen_weight=0.0,
            tracking_delta_enemy_improve_weight=0.0,
            tracking_delta_enemy_worsen_weight=0.0,
            tracking_delta_scale=0.1,
            tracking_delta_deadzone=0.0,
        )
    )
    previous_info = _info(
        target_distance=160.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -160.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=float(np.linalg.norm(np.asarray([90.0, 0.0, -120.0], dtype=np.float32))),
        sim_time_seconds=6.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[90.0, 150.0, -120.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=np.zeros((4,), dtype=np.float32),
        previous_action_cont=np.zeros((4,), dtype=np.float32),
        current_info=current_info,
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tracking_quality > 0.0
    assert reward.tracking_quality < 0.5
    assert reward.tracking_delta_bonus > 0.0


def test_shot_delta_bonus_rewards_improving_shot_window() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            shot_delta_self_improve_weight=0.35,
            shot_delta_self_worsen_weight=0.0,
            shot_delta_enemy_improve_weight=0.0,
            shot_delta_enemy_worsen_weight=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=7.0,
        self_position=[0.0, 150.0, 0.0],
        self_velocity=[0.0, 0.0, 60.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[30.0, 150.0, 220.0],
        enemy_velocity=[0.0, 0.0, -60.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=7.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_velocity=[0.0, 0.0, 60.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 220.0],
        enemy_velocity=[0.0, 0.0, -60.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.shot_delta_bonus > 0.0


def test_tail_delta_bonus_rewards_improving_tail_hold() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tail_delta_self_improve_weight=0.25,
            tail_delta_self_worsen_weight=0.0,
            tail_delta_enemy_improve_weight=0.0,
            tail_delta_enemy_worsen_weight=0.0,
        )
    )
    previous_info = _info(
        target_distance=160.0,
        sim_time_seconds=8.0,
        self_position=[0.0, 150.0, -20.0],
        self_forward=[0.6, 0.0, 0.8],
        enemy_position=[0.0, 150.0, 140.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=160.0,
        sim_time_seconds=8.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, -20.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 140.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tail_delta_bonus > 0.0


def test_tracking_delta_bonus_penalizes_enemy_improving_tracking() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tracking_delta_self_improve_weight=0.0,
            tracking_delta_self_worsen_weight=0.0,
            tracking_delta_enemy_improve_weight=0.6,
            tracking_delta_enemy_worsen_weight=0.0,
            tracking_delta_scale=0.1,
            tracking_delta_deadzone=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=9.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[110.0, 150.0, -220.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=9.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -220.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tracking_delta_bonus < 0.0


def test_tracking_delta_bonus_rewards_enemy_losing_tracking() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tracking_delta_self_improve_weight=0.0,
            tracking_delta_self_worsen_weight=0.0,
            tracking_delta_enemy_improve_weight=0.0,
            tracking_delta_enemy_worsen_weight=0.6,
            tracking_delta_scale=0.1,
            tracking_delta_deadzone=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=9.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -220.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=9.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[110.0, 150.0, -220.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tracking_delta_bonus > 0.0


def test_shot_delta_bonus_penalizes_enemy_improving_shot_window() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            shot_delta_self_improve_weight=0.0,
            shot_delta_self_worsen_weight=0.0,
            shot_delta_enemy_improve_weight=0.5,
            shot_delta_enemy_worsen_weight=0.0,
        )
    )
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=10.0,
        self_position=[30.0, 150.0, 220.0],
        self_velocity=[0.0, 0.0, -60.0],
        self_forward=[0.0, 0.0, -1.0],
        enemy_position=[0.0, 150.0, 0.0],
        enemy_velocity=[0.0, 0.0, 60.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=10.0 + (1.0 / 60.0),
        self_position=[0.0, 150.0, 220.0],
        self_velocity=[0.0, 0.0, -60.0],
        self_forward=[0.0, 0.0, -1.0],
        enemy_position=[0.0, 150.0, 0.0],
        enemy_velocity=[0.0, 0.0, 60.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.shot_delta_bonus < 0.0


def test_tail_delta_bonus_rewards_enemy_losing_tail_hold() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            tail_delta_self_improve_weight=0.0,
            tail_delta_self_worsen_weight=0.0,
            tail_delta_enemy_improve_weight=0.0,
            tail_delta_enemy_worsen_weight=0.4,
        )
    )
    previous_info = _info(
        target_distance=160.0,
        sim_time_seconds=11.0,
        self_position=[0.0, 150.0, 140.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -20.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    current_info = _info(
        target_distance=160.0,
        sim_time_seconds=11.0 + (1.0 / 60.0),
        self_position=[60.0, 150.0, 140.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -20.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    reward = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.tail_delta_bonus > 0.0


def test_tail_hold_score_is_continuous_across_nonideal_tail_geometries() -> None:
    composer = PolicyRewardComposer()
    ideal = composer.compute(
        previous_info=_info(
            target_distance=160.0,
            sim_time_seconds=8.0,
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=160.0,
            sim_time_seconds=8.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    offset = composer.compute(
        previous_info=_info(
            target_distance=float(np.linalg.norm(np.asarray([50.0, 0.0, 140.0], dtype=np.float32))),
            sim_time_seconds=8.0,
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[50.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=float(np.linalg.norm(np.asarray([50.0, 0.0, 140.0], dtype=np.float32))),
            sim_time_seconds=8.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[50.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    opposite_heading = composer.compute(
        previous_info=_info(
            target_distance=160.0,
            sim_time_seconds=8.0,
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=160.0,
            sim_time_seconds=8.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, -20.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 140.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert ideal.tail_hold_score > offset.tail_hold_score > 0.0
    assert opposite_heading.tail_hold_score < ideal.tail_hold_score


def test_reward_front_target_stronger_than_side_target() -> None:
    composer = PolicyRewardComposer()
    common_front = _info(
        target_distance=180.0,
        sim_time_seconds=4.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 180.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    common_side = _info(
        target_distance=180.0,
        sim_time_seconds=4.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[180.0, 150.0, 0.0],
        enemy_forward=[-1.0, 0.0, 0.0],
    )
    front_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    side_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    front = composer.compute(
        previous_info=common_front,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**common_front, "sim_time_seconds": 4.0 + (1.0 / 60.0)},
        current_obs=front_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    side = composer.compute(
        previous_info=common_side,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**common_side, "sim_time_seconds": 4.0 + (1.0 / 60.0)},
        current_obs=side_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert front.attack_advantage > side.attack_advantage
    assert front.total > side.total


def test_reward_prefers_stronger_fire_alignment_for_more_forward_target() -> None:
    composer = PolicyRewardComposer()
    core = composer.compute(
        previous_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 180.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 180.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    outer = composer.compute(
        previous_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[55.0, 150.0, 180.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=180.0,
            sim_time_seconds=4.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[55.0, 150.0, 180.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert core.fire_alignment_score > outer.fire_alignment_score
    assert core.tracking_quality > outer.tracking_quality
    assert core.attack_advantage > outer.attack_advantage


def test_reward_mirror_threat_guides_escape_from_enemy_axis() -> None:
    composer = PolicyRewardComposer()
    core_threat = composer.compute(
        previous_info=_info(
            target_distance=120.0,
            sim_time_seconds=4.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_velocity=[0.0, 0.0, 120.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=120.0,
            sim_time_seconds=4.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_velocity=[0.0, 0.0, 120.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    escaped = composer.compute(
        previous_info=_info(
            target_distance=138.0,
            sim_time_seconds=4.0,
            self_position=[55.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_velocity=[0.0, 0.0, 120.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=138.0,
            sim_time_seconds=4.0 + (1.0 / 60.0),
            self_position=[55.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_velocity=[0.0, 0.0, 120.0],
        ),
        current_obs=np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32),
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert core_threat.opponent_fire_alignment_score > escaped.opponent_fire_alignment_score
    assert core_threat.threat_advantage > escaped.threat_advantage
    assert escaped.total > core_threat.total


def test_reward_does_not_penalize_sustained_full_deflection() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    action = np.asarray([1.0, 1.0, -1.0, 0.5], dtype=np.float32)
    reward = composer.compute(
        previous_info=None,
        previous_previous_action_cont=action,
        previous_action_cont=action,
        current_info=_info(target_distance=150.0),
        current_obs=obs,
        action_cont=action,
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.pitch_jitter_penalty == 0.0
    assert reward.roll_jitter_penalty == 0.0
    assert reward.yaw_jitter_penalty == 0.0


def test_reward_penalizes_action_jitter() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            pitch_jitter_penalty_weight=0.1,
            roll_jitter_penalty_weight=0.1,
            yaw_jitter_penalty_weight=0.1,
        )
    )
    reward = composer.compute(
        previous_info=None,
        previous_previous_action_cont=np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32),
        previous_action_cont=np.asarray([0.0, -1.0, 1.0, -0.5], dtype=np.float32),
        current_info=_info(target_distance=150.0),
        current_obs=obs,
        action_cont=np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.pitch_jitter_penalty > 0.0
    assert reward.roll_jitter_penalty > 0.0
    assert reward.yaw_jitter_penalty > 0.0


def test_ground_boundary_penalty_is_high_when_diving_toward_ground() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 25.0, 0.0],
            self_velocity=[0.0, -60.0, 0.0],
            self_forward=[0.0, -0.8, 0.6],
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    safe = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 180.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.ground_boundary_penalty > 0.0
    assert reward.ground_boundary_threat > 0.0
    assert reward.ground_boundary_penalty > safe.ground_boundary_penalty


def test_ground_boundary_threat_grows_as_clearance_shrinks() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    higher_clearance = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_stall_factor=0.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 120.0, 0.0],
            self_velocity=[0.0, -60.0, 0.0],
            self_forward=[0.0, -0.8, 0.6],
            self_stall_factor=0.0,
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    lower_clearance = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_stall_factor=0.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 40.0, 0.0],
            self_velocity=[0.0, -60.0, 0.0],
            self_forward=[0.0, -0.8, 0.6],
            self_stall_factor=0.0,
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert lower_clearance.ground_boundary_threat > higher_clearance.ground_boundary_threat
    assert lower_clearance.ground_boundary_penalty > higher_clearance.ground_boundary_penalty


def test_ground_boundary_threat_uses_broadphase_sphere_clearance() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    reward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS * 0.5, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.ground_boundary_threat > 0.0
    assert reward.ground_boundary_penalty > 0.0


def test_ground_boundary_threat_suppresses_combat_rewards() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    safe = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    diving = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 25.0, 0.0],
            self_velocity=[0.0, -60.0, 0.0],
            self_forward=[0.0, -0.8, 0.6],
            enemy_position=[0.0, 25.0, 220.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert diving.ground_boundary_threat > 0.0
    assert diving.boundary_combat_gate < safe.boundary_combat_gate
    # predictive_fire_hit_bonus_weight=0 by default; skip
    assert diving.attack_advantage < safe.attack_advantage


def test_ground_boundary_recovery_bonus_requires_windowed_clearance_improvement() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    recovering = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_position=[0.0, 20.0, 0.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        reward_history={
            "frames": [
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.90, self_position=[0.0, 18.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.95, self_position=[0.0, 19.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
            ]
        },
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 25.0, 0.0],
            self_velocity=[0.0, 60.0, 0.0],
            self_forward=[0.0, 0.8, 0.6],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    oscillating = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_position=[0.0, 19.0, 0.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        reward_history={
            "frames": [
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.90, self_position=[0.0, 18.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.92, self_position=[0.0, 24.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.94, self_position=[0.0, 18.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.96, self_position=[0.0, 24.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
            ]
        },
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 25.0, 0.0],
            self_velocity=[0.0, 60.0, 0.0],
            self_forward=[0.0, 0.8, 0.6],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert recovering.boundary_recovery_bonus > 0.0
    assert oscillating.boundary_recovery_bonus < recovering.boundary_recovery_bonus


def test_ground_boundary_recovery_bonus_does_not_reward_direction_flip_without_clearance_gain() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_position=[0.0, 8.0, 0.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        reward_history={
            "frames": [
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.90, self_position=[0.0, 8.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.95, self_position=[0.0, 8.1, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
            ]
        },
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 8.0, 0.0],
            self_velocity=[0.0, 60.0, 0.0],
            self_forward=[0.0, 0.8, 0.6],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.boundary_recovery_bonus == 0.0


def test_horizontal_boundary_penalty_prefers_turning_back_inward() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    outward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[4950.0, 150.0, 0.0],
            self_velocity=[120.0, 0.0, 0.0],
            self_forward=[1.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    inward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[4950.0, 150.0, 0.0],
            self_velocity=[-120.0, 0.0, 0.0],
            self_forward=[-1.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert outward.horizontal_boundary_penalty > 0.0
    assert abs(inward.horizontal_boundary_penalty - outward.horizontal_boundary_penalty) < 1e-9
    assert inward.total > outward.total


def test_horizontal_boundary_threat_uses_broadphase_sphere_clearance() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    reward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[5000.0 - (AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS * 0.5), 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.horizontal_boundary_threat > 0.0
    assert reward.horizontal_boundary_penalty > 0.0


def test_horizontal_boundary_recovery_bonus_rewards_windowed_return_from_soft_zone() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0, self_position=[4970.0, 150.0, 0.0]),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        reward_history={
            "frames": [
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.90, self_position=[4990.0, 150.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.95, self_position=[4980.0, 150.0, 0.0]),
                    "action_cont": np.zeros((4,), dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
            ]
        },
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[4960.0, 150.0, 0.0],
            self_velocity=[-120.0, 0.0, 0.0],
            self_forward=[-1.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.horizontal_boundary_penalty > 0.0
    assert reward.boundary_recovery_bonus > 0.0


def test_ceiling_boundary_penalty_is_high_when_climbing_toward_ceiling() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            flight_ceiling_height=300.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 275.0, 0.0],
            self_velocity=[0.0, 60.0, 0.0],
            self_forward=[0.0, 0.8, 0.6],
            flight_ceiling_height=300.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    safe = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            flight_ceiling_height=300.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            flight_ceiling_height=300.0,
            enemy_position=[0.0, 150.0, -220.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.ceiling_boundary_penalty > 0.0
    assert reward.ceiling_boundary_threat > 0.0
    assert reward.ceiling_boundary_penalty > safe.ceiling_boundary_penalty


def test_boundary_recovery_bonus_is_zero_when_far_from_all_boundaries() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    composer = PolicyRewardComposer()
    reward = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            self_position=[0.0, 600.0, 0.0],
            flight_ceiling_height=2000.0,
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 600.0, 0.0],
            self_velocity=[0.0, 20.0, 0.0],
            self_forward=[0.0, 0.3, 0.95],
            flight_ceiling_height=2000.0,
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.ground_boundary_threat < 1e-12
    assert reward.ceiling_boundary_threat < 1e-12
    assert reward.horizontal_boundary_threat < 1e-12
    assert reward.boundary_recovery_bonus == 0.0


def test_reward_adds_predictive_aircraft_collision_threat_at_close_distance() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(aircraft_collision_threat_weight=0.5))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    reward = composer.compute(
        previous_info=_info(
            target_distance=18.0,
            sim_time_seconds=2.0,
            self_position=[0.0, 150.0, 0.0],
            enemy_position=[0.0, 150.0, 18.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=6.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            enemy_position=[0.0, 150.0, 6.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    safe = composer.compute(
        previous_info=_info(
            target_distance=40.0,
            sim_time_seconds=2.0,
            self_position=[0.0, 150.0, 0.0],
            enemy_position=[0.0, 150.0, 40.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=40.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            enemy_position=[0.0, 150.0, 40.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.aircraft_collision_threat > 0.0
    assert reward.aircraft_collision_threat > safe.aircraft_collision_threat


def test_reward_adds_predictive_aircraft_collision_threat_before_contact() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(aircraft_collision_threat_weight=0.5))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    threat = composer.compute(
        previous_info=_info(target_distance=60.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=60.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, -30.0],
            self_velocity=[0.0, 0.0, 60.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 30.0],
            enemy_velocity=[0.0, 0.0, -60.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    safe = composer.compute(
        previous_info=_info(target_distance=60.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=60.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, -30.0],
            self_velocity=[60.0, 0.0, 0.0],
            self_forward=[1.0, 0.0, 0.0],
            enemy_position=[0.0, 150.0, 30.0],
            enemy_velocity=[-60.0, 0.0, 0.0],
            enemy_forward=[-1.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert threat.aircraft_collision_threat > 0.0
    assert threat.aircraft_collision_threat > safe.aircraft_collision_threat


def test_dynamic_turn_radius_grows_with_speed_and_stall() -> None:
    composer = PolicyRewardComposer()
    low_speed = composer._instantaneous_turn_radius(40.0, 0.0)
    cruise = composer._instantaneous_turn_radius(60.0, 0.0)
    high_speed = composer._instantaneous_turn_radius(80.0, 0.0)
    stalled = composer._instantaneous_turn_radius(60.0, 0.8)
    assert low_speed < cruise < high_speed
    assert stalled > cruise


def test_state_pullup_turn_radius_prefers_environment_value() -> None:
    composer = PolicyRewardComposer()
    radius = composer._state_pullup_turn_radius(
        {
            "pullup_turn_radius_m": 123.0,
            "linear_velocity": [80.0, 0.0, 0.0],
            "stall_factor": 0.9,
        }
    )
    assert radius == 123.0


def test_reward_penalizes_aircraft_collision_event_heavily() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    info = _info(target_distance=8.0, sim_time_seconds=2.0)
    collided = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(target_distance=8.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
            "events_since_last_step": [
                {
                    "kind": "Collision",
                    "subject": "fighter1",
                    "other_subject": "fighter2",
                    "event_detail": "aircraft",
                    "magnitude": 6.0,
                },
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    clean = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=8.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert collided.aircraft_collision_penalty == composer.config.aircraft_collision_penalty_weight
    assert collided.surface_collision_penalty == 0.0
    assert collided.total < clean.total


def test_reward_penalizes_surface_collision_event_heavily() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    info = _info(target_distance=120.0, sim_time_seconds=2.0)
    collided = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(target_distance=120.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
            "events_since_last_step": [
                {
                    "kind": "Collision",
                    "subject": "fighter1",
                    "event_detail": "ground",
                    "magnitude": 6.0,
                },
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    clean = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=120.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert collided.aircraft_collision_penalty == 0.0
    assert collided.surface_collision_penalty == composer.config.surface_collision_penalty_weight
    assert collided.total < clean.total


def test_collision_event_context_overrides_distance_heuristic() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    info = _info(target_distance=120.0, sim_time_seconds=2.0)
    collided = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(target_distance=120.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
            "events_since_last_step": [
                {
                    "kind": "Collision",
                    "subject": "fighter1",
                    "other_subject": "fighter2",
                    "event_detail": "aircraft",
                    "magnitude": 6.0,
                },
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert collided.aircraft_collision_penalty == composer.config.aircraft_collision_penalty_weight
    assert collided.surface_collision_penalty == 0.0


def test_reward_prefers_better_attack_solution_geometry() -> None:
    composer = PolicyRewardComposer()
    common_info = _info(target_distance=220.0, sim_time_seconds=2.0)
    good_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    poor_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    high = composer.compute(
        previous_info=common_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=good_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    low = composer.compute(
        previous_info=common_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[150.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=poor_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert high.shot_feasibility > low.shot_feasibility
    assert high.attack_advantage > low.attack_advantage
    # total comparison removed: with correct geometry, opponent threat advantage
    # makes head-on positions more dangerous, offsetting the attack advantage


def test_reward_adds_bonus_when_hitting_enemy() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    info = _info(
        target_distance=150.0,
        sim_time_seconds=2.0,
        self_position=[0.0, 300.0, 0.0],
        flight_ceiling_height=2000.0,
    )
    hit = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(
                target_distance=150.0,
                sim_time_seconds=2.0 + (1.0 / 60.0),
                self_position=[0.0, 300.0, 0.0],
                flight_ceiling_height=2000.0,
            ),
            "events_since_last_step": [
                {"kind": "Hit", "subject": "fighter2", "magnitude": 4.0},
                {"kind": "Hit", "subject": "fighter2:LeftWing", "magnitude": 4.0},
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    miss = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(
                target_distance=150.0,
                sim_time_seconds=2.0 + (1.0 / 60.0),
                self_position=[0.0, 300.0, 0.0],
                flight_ceiling_height=2000.0,
            ),
            "events_since_last_step": [
                {"kind": "Hit", "subject": "fighter1", "magnitude": 4.0},
                {"kind": "Damage", "subject": "fighter2", "magnitude": 4.0},
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert hit.hit_enemy_bonus == composer.config.hit_enemy_bonus_weight * 2.0
    assert miss.hit_enemy_bonus == 0.0
    assert hit.total > miss.total


def test_reward_penalizes_getting_hit() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    info = _info(target_distance=150.0, sim_time_seconds=2.0)
    got_hit = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={
            **_info(target_distance=150.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
            "events_since_last_step": [
                {"kind": "Hit", "subject": "fighter1", "magnitude": 4.0},
                {"kind": "Hit", "subject": "fighter1:LeftWing", "magnitude": 4.0},
            ],
        },
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    clean = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=150.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert got_hit.got_hit_penalty == composer.config.got_hit_penalty_weight * 2.0
    assert got_hit.total < clean.total


def test_reward_adds_predictive_fire_bonus_for_future_intercept() -> None:
    slices = _field_slices()
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    # gun_window_score removed from obs schema
    hit = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_velocity=[0.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    miss = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[120.0, 150.0, 220.0],
            enemy_velocity=[0.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    # predictive_fire_hit_bonus_weight=0 by default; gun_window_score removed from obs
    # total comparison no longer valid without pre-set obs values


def test_reward_predictive_fire_bonus_requires_fire_command() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    reward = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_velocity=[0.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.predictive_fire_hit_bonus == 0.0


def test_reward_applies_light_brake_penalty() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(brake_penalty_weight=1.0))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    free = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    braking = composer.compute(
        previous_info=_info(target_distance=220.0, sim_time_seconds=2.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=220.0, sim_time_seconds=2.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert braking.brake_penalty > 0.0
    assert braking.total < free.total


def test_reward_gives_moderate_bonus_for_throttle_change() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(throttle_change_bonus_weight=0.5))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    previous_info = _info(target_distance=220.0, sim_time_seconds=2.0)
    current_info = _info(target_distance=220.0, sim_time_seconds=2.0 + (1.0 / 60.0))
    steady = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=np.asarray([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
        current_info=current_info,
        current_obs=obs,
        action_cont=np.asarray([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    changing = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=np.asarray([0.20, 0.0, 0.0, 0.0], dtype=np.float32),
        current_info=current_info,
        current_obs=obs,
        action_cont=np.asarray([0.70, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert steady.throttle_change_bonus == 0.0
    assert changing.throttle_change_bonus > 0.0
    assert changing.total > steady.total


def test_reward_penalizes_too_low_throttle_more_steeply() -> None:
    composer = PolicyRewardComposer()
    slices = _field_slices()
    high_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    mid_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    low_obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    high_obs[slices["self_throttle_norm"]] = np.asarray([0.9], dtype=np.float32)
    mid_obs[slices["self_throttle_norm"]] = np.asarray([0.35], dtype=np.float32)
    low_obs[slices["self_throttle_norm"]] = np.asarray([0.1], dtype=np.float32)
    common_previous = _info(target_distance=220.0, sim_time_seconds=2.0)
    common_current = _info(target_distance=220.0, sim_time_seconds=2.0 + (1.0 / 60.0))
    high = composer.compute(
        previous_info=common_previous,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=common_current,
        current_obs=high_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    mid = composer.compute(
        previous_info=common_previous,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=common_current,
        current_obs=mid_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    low = composer.compute(
        previous_info=common_previous,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=common_current,
        current_obs=low_obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert high.throttle_low_penalty > 0.0
    assert mid.throttle_low_penalty > high.throttle_low_penalty
    assert low.throttle_low_penalty > mid.throttle_low_penalty
    assert (low.throttle_low_penalty - mid.throttle_low_penalty) > (
        mid.throttle_low_penalty - high.throttle_low_penalty
    )
    assert low.total < mid.total < high.total


def test_speed_jitter_penalty_hits_reversing_throttle_without_speed_change() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(speed_jitter_penalty_weight=0.5))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=2.0,
        self_velocity=[0.0, 0.0, 60.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=2.0 + (1.0 / 60.0),
        self_velocity=[0.0, 0.0, 60.0],
    )
    jittery = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        previous_action_cont=np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        current_info=current_info,
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert jittery.speed_jitter_penalty > 0.0


def test_speed_jitter_penalty_is_zero_for_monotonic_throttle_change() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    previous_info = _info(
        target_distance=220.0,
        sim_time_seconds=2.0,
        self_velocity=[0.0, 0.0, 54.0],
    )
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=2.0 + (1.0 / 60.0),
        self_velocity=[0.0, 0.0, 57.0],
    )
    monotonic = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
        previous_action_cont=np.asarray([0.4, 0.0, 0.0, 0.0], dtype=np.float32),
        current_info=current_info,
        current_obs=obs,
        action_cont=np.asarray([0.7, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert monotonic.speed_jitter_penalty == 0.0


def test_speed_jitter_penalty_uses_history_window_for_high_frequency_oscillation() -> None:
    composer = PolicyRewardComposer(PolicyRewardConfig(speed_jitter_penalty_weight=0.5))
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    current_info = _info(
        target_distance=220.0,
        sim_time_seconds=2.0 + (1.0 / 60.0),
        self_velocity=[0.0, 0.0, 60.2],
    )
    reward = composer.compute(
        previous_info=_info(
            target_distance=220.0,
            sim_time_seconds=2.0,
            self_velocity=[0.0, 0.0, 60.0],
        ),
        previous_previous_action_cont=np.asarray([0.7, 0.0, 0.0, 0.0], dtype=np.float32),
        previous_action_cont=np.asarray([0.3, 0.0, 0.0, 0.0], dtype=np.float32),
        reward_history={
            "frames": [
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.90, self_velocity=[0.0, 0.0, 60.0]),
                    "action_cont": np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.92, self_velocity=[0.0, 0.0, 60.1]),
                    "action_cont": np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.94, self_velocity=[0.0, 0.0, 60.0]),
                    "action_cont": np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
                {
                    "info": _info(target_distance=220.0, sim_time_seconds=1.96, self_velocity=[0.0, 0.0, 60.1]),
                    "action_cont": np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
                    "action_bin": np.zeros((3,), dtype=np.float32),
                },
            ]
        },
        current_info=current_info,
        current_obs=obs,
        action_cont=np.asarray([0.7, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert reward.speed_jitter_penalty > 0.0


def test_time_pressure_ramps_up_over_episode_time() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            time_pressure_initial_bonus_per_second=0.0,
            time_pressure_rate_per_second=1.0,
            time_pressure_ramp_reference_seconds=180.0,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    early = composer.compute(
        previous_info=_info(target_distance=250.0, sim_time_seconds=1.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=250.0, sim_time_seconds=1.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    late = composer.compute(
        previous_info=_info(target_distance=250.0, sim_time_seconds=121.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=250.0, sim_time_seconds=121.0 + (1.0 / 60.0)),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert early.time_pressure <= 0.0
    assert late.time_pressure < 0.0
    assert early.time_pressure_scale < late.time_pressure_scale
    assert early.time_pressure > late.time_pressure


def test_time_pressure_starts_near_configured_survival_bonus() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            time_pressure_initial_bonus_per_second=0.5,
            time_pressure_rate_per_second=1.0,
            time_pressure_ramp_reference_seconds=120.0,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    start = composer.compute(
        previous_info=_info(target_distance=250.0, sim_time_seconds=0.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(target_distance=250.0, sim_time_seconds=1.0 / 60.0),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert start.time_pressure_scale < 1e-6
    assert np.isclose(start.time_pressure, 0.5 / 60.0, rtol=1e-3, atol=1e-8)


def test_positive_continuous_rewards_decay_over_episode_time() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            fire_window_bonus_weight=1.0,
            positive_reward_decay_reference_seconds=180.0,
            positive_reward_decay_min_scale=0.1,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    early = composer.compute(
        previous_info=_info(target_distance=200.0, sim_time_seconds=1.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            sim_time_seconds=1.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 200.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    late = composer.compute(
        previous_info=_info(target_distance=200.0, sim_time_seconds=121.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            sim_time_seconds=121.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 200.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    assert early.positive_reward_decay_scale > late.positive_reward_decay_scale
    assert early.attack_advantage > late.attack_advantage
    assert early.fire_window_bonus > late.fire_window_bonus


def test_default_positive_continuous_reward_decay_is_not_identity() -> None:
    composer = PolicyRewardComposer()

    assert composer._positive_continuous_reward_decay_scale(0.0) == 1.0
    assert composer._positive_continuous_reward_decay_scale(600.0) == 0.5
    assert composer._positive_continuous_reward_decay_scale(100_000.0) == 0.1


def test_positive_continuous_reward_decay_rejects_invalid_config() -> None:
    for invalid_scale in (-0.1, 1.1):
        try:
            PolicyRewardConfig(positive_reward_decay_min_scale=invalid_scale)
        except ValueError as exc:
            assert "within [0, 1]" in str(exc)
        else:
            raise AssertionError("expected invalid positive reward decay scale to be rejected")


def test_negative_continuous_rewards_remain_identity_scaled_for_now() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    early = composer.compute(
        previous_info=_info(target_distance=200.0, sim_time_seconds=1.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            sim_time_seconds=1.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -200.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    late = composer.compute(
        previous_info=_info(target_distance=200.0, sim_time_seconds=121.0),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=200.0,
            sim_time_seconds=121.0 + (1.0 / 60.0),
            self_position=[0.0, 150.0, 0.0],
            self_velocity=[0.0, 0.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, -200.0],
            enemy_velocity=[0.0, 0.0, 0.0],
            enemy_forward=[0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert np.isclose(early.negative_reward_decay_scale, 1.0)
    assert np.isclose(late.negative_reward_decay_scale, 1.0)
    assert np.isclose(early.threat_advantage, late.threat_advantage)


def test_reward_encourages_target_aligned_maneuver_activity_in_engagement() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    # line_of_sight_body now computed from state positions
    composer = PolicyRewardComposer(PolicyRewardConfig(maneuver_activity_weight=0.05))
    info = _info(
        target_distance=134.4,
        sim_time_seconds=5.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[55.0, 192.0, 120.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    flat = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 5.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    aligned = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 5.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, -0.6, 0.5, -0.4], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    misaligned = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 5.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.6, -0.5, -0.4], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert flat.maneuver_activity == 0.0
    assert aligned.tracking_quality > 0.0
    assert aligned.maneuver_activity > 0.0
    assert misaligned.maneuver_activity < 1e-4
    assert aligned.total > flat.total
    assert aligned.total > misaligned.total


def test_low_speed_penalty_is_linear_and_monotonic() -> None:
    composer = PolicyRewardComposer()
    high = composer._low_speed_penalty(78.0)
    mid = composer._low_speed_penalty(58.0)
    low = composer._low_speed_penalty(38.0)
    very_low = composer._low_speed_penalty(18.0)
    assert high < mid < low <= very_low
    assert np.isclose(high, composer.config.low_speed_penalty_weight * (2.0 / 80.0), atol=1e-6)
    assert np.isclose((mid - high), (low - mid), atol=1e-6)
    assert very_low > low


def test_reward_prefers_higher_speed_in_two_circle_merge() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    base_kwargs = dict(
        target_distance=float(np.linalg.norm(np.asarray([80.0, 0.0, 220.0], dtype=np.float32))),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
        enemy_position=[80.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
        enemy_orientation_quat=[1.0, 0.0, 0.0, 0.0],
        self_max_level_speed_mps=140.0,
        enemy_max_level_speed_mps=140.0,
    )
    slow = composer.compute(
        previous_info=_info(sim_time_seconds=7.0, self_velocity=[0.0, 0.0, 70.0], enemy_velocity=[0.0, 0.0, -70.0], **base_kwargs),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(sim_time_seconds=7.0 + (1.0 / 60.0), self_velocity=[0.0, 0.0, 70.0], enemy_velocity=[0.0, 0.0, -70.0], **base_kwargs),
        current_obs=obs,
        action_cont=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    fast = composer.compute(
        previous_info=_info(sim_time_seconds=7.0, self_velocity=[0.0, 0.0, 120.0], enemy_velocity=[0.0, 0.0, -120.0], **base_kwargs),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(sim_time_seconds=7.0 + (1.0 / 60.0), self_velocity=[0.0, 0.0, 120.0], enemy_velocity=[0.0, 0.0, -120.0], **base_kwargs),
        current_obs=obs,
        action_cont=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert fast.two_circle_gate > fast.one_circle_gate
    assert fast.two_circle_speed_reward > slow.two_circle_speed_reward


def test_reward_prefers_low_non_stall_target_speed_in_one_circle_merge() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    base_kwargs = dict(
        target_distance=float(np.linalg.norm(np.asarray([80.0, 0.0, 220.0], dtype=np.float32))),
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
        enemy_position=[80.0, 150.0, 220.0],
        enemy_forward=[0.0, 0.0, -1.0],
        enemy_orientation_quat=[0.0, 1.0, 0.0, 0.0],
        self_max_level_speed_mps=140.0,
        enemy_max_level_speed_mps=140.0,
    )
    target_speed = composer.compute(
        previous_info=_info(sim_time_seconds=7.0, self_velocity=[0.0, 0.0, 45.0], enemy_velocity=[0.0, 0.0, -45.0], **base_kwargs),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(sim_time_seconds=7.0 + (1.0 / 60.0), self_velocity=[0.0, 0.0, 45.0], enemy_velocity=[0.0, 0.0, -45.0], **base_kwargs),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    higher_speed = composer.compute(
        previous_info=_info(sim_time_seconds=7.0, self_velocity=[0.0, 0.0, 75.0], enemy_velocity=[0.0, 0.0, -75.0], **base_kwargs),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(sim_time_seconds=7.0 + (1.0 / 60.0), self_velocity=[0.0, 0.0, 75.0], enemy_velocity=[0.0, 0.0, -75.0], **base_kwargs),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    too_slow = composer.compute(
        previous_info=_info(sim_time_seconds=7.0, self_velocity=[0.0, 0.0, 25.0], enemy_velocity=[0.0, 0.0, -25.0], **base_kwargs),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(sim_time_seconds=7.0 + (1.0 / 60.0), self_velocity=[0.0, 0.0, 25.0], enemy_velocity=[0.0, 0.0, -25.0], **base_kwargs),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert target_speed.one_circle_gate > target_speed.two_circle_gate
    assert target_speed.one_circle_speed_reward > higher_speed.one_circle_speed_reward
    assert target_speed.one_circle_speed_reward > too_slow.one_circle_speed_reward
    assert too_slow.one_circle_speed_reward > 0.0


def test_two_circle_gate_covers_head_on_and_tail_chase() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    head_on = composer.compute(
        previous_info=_info(
            sim_time_seconds=7.0,
            target_distance=220.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
            enemy_orientation_quat=[1.0, 0.0, 0.0, 0.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            sim_time_seconds=7.0 + (1.0 / 60.0),
            target_distance=220.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
            enemy_orientation_quat=[1.0, 0.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    tail_chase = composer.compute(
        previous_info=_info(
            sim_time_seconds=7.0,
            target_distance=160.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 160.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_orientation_quat=[0.0, 0.0, 0.0, 1.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            sim_time_seconds=7.0 + (1.0 / 60.0),
            target_distance=160.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 160.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_orientation_quat=[0.0, 0.0, 0.0, 1.0],
        ),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert head_on.two_circle_gate > head_on.one_circle_gate
    assert tail_chase.two_circle_gate > tail_chase.one_circle_gate


def test_one_circle_gate_covers_crossing_and_mid_circle() -> None:
    composer = PolicyRewardComposer()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    crossing = composer.compute(
        previous_info=_info(
            sim_time_seconds=7.0,
            target_distance=220.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
            enemy_orientation_quat=[0.0, 1.0, 0.0, 0.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            sim_time_seconds=7.0 + (1.0 / 60.0),
            target_distance=220.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 220.0],
            enemy_forward=[0.0, 0.0, -1.0],
            enemy_orientation_quat=[0.0, 1.0, 0.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    mid_circle = composer.compute(
        previous_info=_info(
            sim_time_seconds=7.0,
            target_distance=160.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 160.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_orientation_quat=[0.0, 0.0, 1.0, 0.0],
        ),
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            sim_time_seconds=7.0 + (1.0 / 60.0),
            target_distance=160.0,
            self_position=[0.0, 150.0, 0.0],
            self_forward=[0.0, 0.0, 1.0],
            self_orientation_quat=[0.0, 0.0, 0.0, 1.0],
            enemy_position=[0.0, 150.0, 160.0],
            enemy_forward=[0.0, 0.0, 1.0],
            enemy_orientation_quat=[0.0, 0.0, 1.0, 0.0],
        ),
        current_obs=obs,
        action_cont=np.asarray([0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )
    assert crossing.one_circle_gate > crossing.two_circle_gate
    assert mid_circle.one_circle_gate > mid_circle.two_circle_gate


def test_reward_encourages_firing_in_shot_window_and_penalizes_hesitation() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            fire_window_bonus_weight=1.0,
            fire_hesitation_penalty_weight=3.0,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    base_info = _info(
        target_distance=120.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_velocity=[0.0, 0.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 120.0],
        enemy_velocity=[0.0, 0.0, -20.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    firing = composer.compute(
        previous_info=base_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**base_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    hesitating = composer.compute(
        previous_info=base_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**base_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert firing.shot_feasibility > 0.0
    assert firing.shot_coarse_upper_bound >= composer.config.shot_broadphase_epsilon
    assert firing.fire_alignment_score > 0.0
    assert firing.fire_window_bonus > 0.0
    assert hesitating.fire_hesitation_penalty > 0.0
    assert firing.total > hesitating.total


def test_reward_adds_configurable_fire_bonus_without_tactical_gates() -> None:
    weight = 6.0
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            fire_command_bonus_weight=weight,
            fire_window_bonus_weight=0.0,
            predictive_fire_hit_bonus_weight=0.0,
            fire_hesitation_penalty_weight=0.0,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    previous_info = _info(
        target_distance=120.0,
        sim_time_seconds=10.0,
        enemy_position=[0.0, 150.0, -120.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    current_info = _info(
        target_distance=120.0,
        sim_time_seconds=10.0 + (1.0 / 30.0),
        enemy_position=[0.0, 150.0, -120.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )

    firing = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=current_info,
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    idle = composer.compute(
        previous_info=previous_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info=_info(
            target_distance=120.0,
            sim_time_seconds=10.0 + (1.0 / 30.0),
            enemy_position=[0.0, 150.0, -120.0],
            enemy_forward=[0.0, 0.0, -1.0],
        ),
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.zeros((3,), dtype=np.float32),
    )

    expected = weight * (1.0 / 30.0) * firing.positive_reward_decay_scale
    assert firing.fire_alignment_score == 0.0
    assert firing.fire_window_bonus == 0.0
    assert firing.predictive_fire_hit_bonus == 0.0
    assert np.isclose(firing.fire_command_bonus, expected)
    assert idle.fire_command_bonus == 0.0
    assert np.isclose(firing.total - idle.total, firing.fire_command_bonus)


def test_reward_adds_fire_window_terms_for_broad_forward_window() -> None:
    composer = PolicyRewardComposer(
        PolicyRewardConfig(
            fire_window_bonus_weight=1.0,
            fire_hesitation_penalty_weight=3.0,
        )
    )
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    base_info = _info(
        target_distance=220.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_velocity=[0.0, 0.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[35.0, 150.0, 220.0],
        enemy_velocity=[0.0, 0.0, -10.0],
        enemy_forward=[0.0, 0.0, -1.0],
    )
    firing = composer.compute(
        previous_info=base_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**base_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    hesitating = composer.compute(
        previous_info=base_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**base_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert firing.fire_alignment_score > 0.0
    assert firing.fire_window_bonus > 0.0
    assert hesitating.fire_hesitation_penalty > 0.0
    assert firing.total > hesitating.total


def test_reward_applies_static_repair_cost_independent_of_ignored_axes() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_repair_active"]] = 1.0
    obs[slices["self_repair_seconds_norm"]] = 0.4
    composer = PolicyRewardComposer()
    info = _info(target_distance=180.0, sim_time_seconds=6.0)
    static = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    maneuver = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.5, 0.5, 0.5], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert static.repair_static_penalty > 0.0
    assert np.isclose(
        static.repair_static_penalty,
        maneuver.repair_static_penalty,
    )


def test_reward_penalizes_high_health_repair_attempt() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_health_state_norm"]] = np.asarray([0.95, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    composer = PolicyRewardComposer()
    info = _info(target_distance=180.0, sim_time_seconds=6.0)
    repair = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    no_repair = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.zeros((4,), dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert repair.repair_twitch_penalty > 0.0
    assert repair.repair_high_health_penalty > 0.0
    assert repair.total < no_repair.total


def test_reward_encourages_low_health_repair() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_health_state_norm"]] = np.asarray([0.25, 0.3, 0.2, 0.2, 0.25, 0.3], dtype=np.float32)
    composer = PolicyRewardComposer()
    info = _info(
        target_distance=360.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 220.0, 0.0],
        enemy_position=[360.0, 220.0, 0.0],
        enemy_velocity=[40.0, 0.0, 0.0],
        enemy_forward=[1.0, 0.0, 0.0],
        flight_ceiling_height=1200.0,
    )
    repair = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    no_repair = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert repair.repair_low_health_bonus > 0.0
    assert repair.repair_opportunity_gate > 0.0
    assert repair.repair_low_health_bonus > no_repair.repair_low_health_bonus


def test_reward_strongly_encourages_repair_for_destroyed_subsystems() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_health_state_norm"]] = np.asarray([0.55, 0.0, 0.4, 0.0, 0.5, 0.8], dtype=np.float32)
    composer = PolicyRewardComposer()
    info = _info(
        target_distance=360.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 220.0, 0.0],
        enemy_position=[360.0, 220.0, 0.0],
        enemy_velocity=[40.0, 0.0, 0.0],
        enemy_forward=[1.0, 0.0, 0.0],
        flight_ceiling_height=1200.0,
    )
    repair = composer.compute(
        previous_info=info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert repair.repair_destroyed_subsystem_bonus > 0.0


def test_reward_suppresses_repair_under_high_threat() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_health_state_norm"]] = np.asarray([0.25, 0.3, 0.2, 0.2, 0.25, 0.3], dtype=np.float32)
    composer = PolicyRewardComposer()
    threat_info = _info(
        target_distance=120.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, -120.0],
        enemy_velocity=[0.0, 0.0, 120.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    safe_info = _info(
        target_distance=120.0,
        sim_time_seconds=6.0,
        self_position=[0.0, 150.0, 0.0],
        self_forward=[0.0, 0.0, 1.0],
        enemy_position=[0.0, 150.0, 120.0],
        enemy_velocity=[0.0, 0.0, 40.0],
        enemy_forward=[0.0, 0.0, 1.0],
    )
    threatened = composer.compute(
        previous_info=threat_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**threat_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    safe = composer.compute(
        previous_info=safe_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**safe_info, "sim_time_seconds": 6.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert threatened.repair_opportunity_gate < safe.repair_opportunity_gate
    assert threatened.repair_under_threat_penalty >= safe.repair_under_threat_penalty
    assert threatened.repair_low_health_bonus < safe.repair_low_health_bonus


def test_reward_suppresses_repair_when_boundary_impact_precedes_completion() -> None:
    slices = _field_slices()
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[slices["self_health_state_norm"]] = np.asarray([0.25, 0.3, 0.2, 0.2, 0.25, 0.3], dtype=np.float32)
    composer = PolicyRewardComposer()
    safe_info = _info(
        target_distance=360.0,
        sim_time_seconds=7.0,
        self_position=[0.0, 220.0, 0.0],
        self_velocity=[0.0, 0.0, 0.0],
        enemy_position=[360.0, 220.0, 0.0],
        enemy_velocity=[40.0, 0.0, 0.0],
        enemy_forward=[1.0, 0.0, 0.0],
        self_repair_elapsed_seconds=0.0,
    )
    unsafe_info = _info(
        target_distance=360.0,
        sim_time_seconds=7.0,
        self_position=[0.0, 18.0, 0.0],
        self_velocity=[0.0, -20.0, 0.0],
        enemy_position=[360.0, 220.0, 0.0],
        enemy_velocity=[40.0, 0.0, 0.0],
        enemy_forward=[1.0, 0.0, 0.0],
        self_repair_elapsed_seconds=0.0,
    )
    safe = composer.compute(
        previous_info=safe_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**safe_info, "sim_time_seconds": 7.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    unsafe = composer.compute(
        previous_info=unsafe_info,
        previous_previous_action_cont=None,
        previous_action_cont=None,
        current_info={**unsafe_info, "sim_time_seconds": 7.0 + (1.0 / 60.0)},
        current_obs=obs,
        action_cont=np.asarray([0.0, 0.4, 0.4, 0.4], dtype=np.float32),
        action_bin=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert unsafe.repair_completion_safety_gate < safe.repair_completion_safety_gate
    assert unsafe.repair_opportunity_gate < safe.repair_opportunity_gate
    assert unsafe.repair_under_threat_penalty > safe.repair_under_threat_penalty
    assert unsafe.repair_low_health_bonus < safe.repair_low_health_bonus
