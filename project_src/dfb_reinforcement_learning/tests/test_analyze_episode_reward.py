from __future__ import annotations

from typing import Any

import numpy as np

from dfb_reinforcement_learning.tools.analyze_episode_reward import (
    _validate_policy_recording_manifest,
    build_head_on_crossing_report,
    detect_base_events,
    detect_head_on_crossing_windows,
    replay_episode_reward_frames,
)
from dfb_reinforcement_learning.policy_contract import ACTION_SCHEMA_ID, POLICY_CONTRACT_ID
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig


def _aircraft(
    *,
    role: str,
    position: list[float],
    linear_velocity: list[float],
    throttle: float,
    forward: list[float],
    destroyed: bool = False,
) -> dict[str, Any]:
    return {
        "role": role,
        "destroyed": destroyed,
        "position": position,
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity": linear_velocity,
        "angular_velocity_deg": [0.0, 0.0, 0.0],
        "forward": forward,
        "stall_factor": 0.0,
        "throttle": throttle,
        "brake": False,
        "gun_overheated": False,
        "gun_heat": 0.0,
        "is_firing": False,
        "repairing": False,
        "repair_elapsed_seconds": 0.0,
        "repair_progress": 0.0,
        "out_of_bounds_seconds": 0.0,
        "hit_points": 100.0,
        "max_hit_points": 100.0,
        "subsystems": [
            {"name": "LeftWing", "hit_points": 25.0, "max_hit_points": 25.0},
            {"name": "RightWing", "hit_points": 25.0, "max_hit_points": 25.0},
            {"name": "PitchTail", "hit_points": 20.0, "max_hit_points": 20.0},
            {"name": "YawTail", "hit_points": 20.0, "max_hit_points": 20.0},
            {"name": "Engine", "hit_points": 30.0, "max_hit_points": 30.0},
        ],
    }


def _state(
    *,
    tick: int,
    sim_time_seconds: float,
    fighter1_position: list[float],
    fighter2_position: list[float],
    fighter1_velocity: list[float] | None = None,
    fighter2_velocity: list[float] | None = None,
    fighter1_throttle: float = 0.2,
    fighter2_throttle: float = 0.2,
) -> dict[str, Any]:
    return {
        "tick": tick,
        "sim_time_seconds": sim_time_seconds,
        "scene_name": "test_scene",
        "arena": {
            "ground_height": 0.0,
            "arena_radius": 5000.0,
            "flight_ceiling_height": 2000.0,
            "ceiling_falloff_range": 250.0,
        },
        "events_since_last_step": [],
        "aircraft": [
            _aircraft(
                role="fighter1",
                position=fighter1_position,
                linear_velocity=fighter1_velocity or [0.0, 0.0, 80.0],
                throttle=fighter1_throttle,
                forward=[0.0, 0.0, 1.0],
            ),
            _aircraft(
                role="fighter2",
                position=fighter2_position,
                linear_velocity=fighter2_velocity or [0.0, 0.0, -80.0],
                throttle=fighter2_throttle,
                forward=[0.0, 0.0, -1.0],
            ),
        ],
    }


def test_reward_diagnosis_requires_native_policy_recording() -> None:
    _validate_policy_recording_manifest(
        {
            "policy_contract_id": POLICY_CONTRACT_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "authoritative_source": True,
        }
    )

    for invalid in (
        {},
        {
            "policy_contract_id": POLICY_CONTRACT_ID,
            "action_schema_id": "wrong",
            "authoritative_source": True,
        },
        {
            "policy_contract_id": POLICY_CONTRACT_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "authoritative_source": False,
        },
    ):
        try:
            _validate_policy_recording_manifest(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid manifest accepted: {invalid}")


def test_replay_episode_reward_frames_replays_recorded_actions() -> None:
    initial_state = _state(
        tick=0,
        sim_time_seconds=0.0,
        fighter1_position=[0.0, 150.0, 0.0],
        fighter2_position=[0.0, 150.0, 240.0],
    )
    steps = [
        {
            "index": 0,
            "tick": 1,
            "sim_time_seconds": 1.0 / 60.0,
            "fighter1_command": {
                "throttle": 0.8,
                "pitch": -0.2,
                "roll": 0.0,
                "yaw": 0.1,
                "brake": False,
                "fire_gun": True,
                "repair": False,
            },
            "fighter2_command": {
                "throttle": 0.2,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "brake": False,
                "fire_gun": False,
                "repair": False,
            },
            "state": _state(
                tick=1,
                sim_time_seconds=1.0 / 60.0,
                fighter1_position=[0.0, 150.0, 5.0],
                fighter2_position=[0.0, 150.0, 220.0],
                fighter1_throttle=0.8,
            ),
        },
        {
            "index": 1,
            "tick": 2,
            "sim_time_seconds": 2.0 / 60.0,
            "fighter1_command": {
                "throttle": 0.1,
                "pitch": -0.1,
                "roll": 0.0,
                "yaw": 0.0,
                "brake": True,
                "fire_gun": False,
                "repair": False,
            },
            "fighter2_command": {
                "throttle": 0.2,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "brake": False,
                "fire_gun": False,
                "repair": False,
            },
            "state": _state(
                tick=2,
                sim_time_seconds=2.0 / 60.0,
                fighter1_position=[0.0, 150.0, 10.0],
                fighter2_position=[0.0, 150.0, 205.0],
                fighter1_throttle=0.1,
            ),
        },
    ]

    records = replay_episode_reward_frames(
        initial_state=initial_state,
        steps=steps,
        ego_role="fighter1",
        reward_composer=PolicyRewardComposer(
            PolicyRewardConfig(brake_penalty_weight=1.0)
        ),
    )

    assert len(records) == 2
    assert records[0]["step_index"] == 0
    assert records[0]["action_cont"] == [0.800000011920929, -0.20000000298023224, 0.0, 0.10000000149011612]
    assert records[0]["action_bin"] == [0.0, 1.0, 0.0]
    assert records[0]["action_named"]["fire_gun"] is True
    assert records[1]["action_bin"] == [1.0, 0.0, 0.0]
    assert records[1]["reward"]["brake_penalty"] > 0.0


def test_replay_episode_reward_frames_keeps_reward_history_alignment() -> None:
    initial_state = _state(
        tick=0,
        sim_time_seconds=0.0,
        fighter1_position=[0.0, 150.0, 0.0],
        fighter2_position=[0.0, 150.0, 200.0],
        fighter1_throttle=0.0,
    )
    steps = []
    for index, throttle in enumerate((0.0, 0.8, 0.0), start=1):
        steps.append(
            {
                "index": index - 1,
                "tick": index,
                "sim_time_seconds": index / 60.0,
                "fighter1_command": {
                    "throttle": throttle,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "yaw": 0.0,
                    "brake": False,
                    "fire_gun": False,
                    "repair": False,
                },
                "fighter2_command": {
                    "throttle": 0.2,
                    "pitch": 0.0,
                    "roll": 0.0,
                    "yaw": 0.0,
                    "brake": False,
                    "fire_gun": False,
                    "repair": False,
                },
                "state": _state(
                    tick=index,
                    sim_time_seconds=index / 60.0,
                    fighter1_position=[0.0, 150.0, float(index * 5)],
                    fighter2_position=[0.0, 150.0, 200.0 - float(index * 5)],
                    fighter1_throttle=throttle,
                ),
            }
        )

    records = replay_episode_reward_frames(
        initial_state=initial_state,
        steps=steps,
        ego_role="fighter1",
        reward_composer=PolicyRewardComposer(
            PolicyRewardConfig(throttle_change_bonus_weight=0.5)
        ),
    )

    assert len(records) == 3
    assert records[1]["reward"]["throttle_change_bonus"] > 0.0
    assert records[2]["reward"]["speed_jitter_penalty"] >= 0.0
    assert np.isfinite(records[2]["reward"]["total"])


def test_detect_base_events_finds_onsets_and_collapses() -> None:
    records = [
        {
            "step_index": 0,
            "tick": 1,
            "sim_time_seconds": 1.0 / 60.0,
            "ego_role": "fighter1",
            "action_bin": [0.0, 0.0, 0.0],
            "reward": {
                "attack_advantage": 2.8,
                "aircraft_collision_threat": 0.10,
                "ground_boundary_threat": 0.0,
                "ceiling_boundary_threat": 0.0,
                "horizontal_boundary_threat": 0.0,
                "shot_feasibility": 0.55,
                "fire_window_bonus": 0.0,
                "closing_speed_mps": 110.0,
            },
            "info": {
                "aircraft_by_role": {
                    "fighter1": {
                        "out_of_bounds_seconds": 0.0,
                    }
                }
            },
        },
        {
            "step_index": 1,
            "tick": 2,
            "sim_time_seconds": 2.0 / 60.0,
            "ego_role": "fighter1",
            "action_bin": [1.0, 1.0, 0.0],
            "reward": {
                "attack_advantage": 1.4,
                "aircraft_collision_threat": 0.92,
                "ground_boundary_threat": 0.0,
                "ceiling_boundary_threat": 0.0,
                "horizontal_boundary_threat": 0.62,
                "shot_feasibility": 0.72,
                "fire_window_bonus": 0.31,
                "closing_speed_mps": 115.0,
            },
            "info": {
                "aircraft_by_role": {
                    "fighter1": {
                        "out_of_bounds_seconds": 0.8,
                    }
                }
            },
        },
    ]

    events = detect_base_events(records)
    event_kinds = [event["kind"] for event in events]
    assert "brake_onset" in event_kinds
    assert "fire_onset" in event_kinds
    assert "attack_advantage_collapse" in event_kinds
    assert "collision_threat_spike" in event_kinds
    assert "boundary_threat_spike" in event_kinds
    assert "out_of_bounds_entry" in event_kinds


def test_detect_head_on_crossing_windows_and_report() -> None:
    records = [
        {
            "step_index": 10,
            "tick": 11,
            "sim_time_seconds": 11.0 / 60.0,
            "ego_role": "fighter1",
            "action_cont": [0.6, 0.0, 0.0, 0.0],
            "action_bin": [0.0, 0.0, 0.0],
            "reward": {
                "attack_advantage": 2.6,
                "threat_advantage": 0.9,
                "shot_feasibility": 0.48,
                "fire_window_bonus": 0.12,
                "aircraft_collision_threat": 0.15,
                "two_circle_gate": 0.72,
                "closing_speed_mps": 105.0,
                "speed_delta_mps": 0.0,
                "predictive_fire_hit_bonus": 0.0,
                "fire_hesitation_penalty": 0.0,
                "brake_penalty": 0.0,
                "ground_boundary_threat": 0.0,
                "ceiling_boundary_threat": 0.0,
                "horizontal_boundary_threat": 0.0,
            },
            "info": {"aircraft_by_role": {"fighter1": {"out_of_bounds_seconds": 0.0}}},
        },
        {
            "step_index": 11,
            "tick": 12,
            "sim_time_seconds": 12.0 / 60.0,
            "ego_role": "fighter1",
            "action_cont": [0.4, 0.0, 0.0, 0.0],
            "action_bin": [1.0, 0.0, 0.0],
            "reward": {
                "attack_advantage": 1.2,
                "threat_advantage": 1.1,
                "shot_feasibility": 0.40,
                "fire_window_bonus": 0.08,
                "aircraft_collision_threat": 0.92,
                "two_circle_gate": 0.80,
                "closing_speed_mps": 112.0,
                "speed_delta_mps": -5.0,
                "predictive_fire_hit_bonus": 0.0,
                "fire_hesitation_penalty": 0.03,
                "brake_penalty": 0.05,
                "ground_boundary_threat": 0.0,
                "ceiling_boundary_threat": 0.0,
                "horizontal_boundary_threat": 0.0,
            },
            "info": {"aircraft_by_role": {"fighter1": {"out_of_bounds_seconds": 0.0}}},
        },
        {
            "step_index": 12,
            "tick": 13,
            "sim_time_seconds": 13.0 / 60.0,
            "ego_role": "fighter1",
            "action_cont": [0.3, 0.0, 0.0, 0.0],
            "action_bin": [1.0, 0.0, 0.0],
            "reward": {
                "attack_advantage": 0.7,
                "threat_advantage": 1.5,
                "shot_feasibility": 0.20,
                "fire_window_bonus": 0.02,
                "aircraft_collision_threat": 0.60,
                "two_circle_gate": 0.65,
                "closing_speed_mps": 95.0,
                "speed_delta_mps": -7.0,
                "predictive_fire_hit_bonus": 0.0,
                "fire_hesitation_penalty": 0.01,
                "brake_penalty": 0.05,
                "ground_boundary_threat": 0.0,
                "ceiling_boundary_threat": 0.0,
                "horizontal_boundary_threat": 0.0,
            },
            "info": {"aircraft_by_role": {"fighter1": {"out_of_bounds_seconds": 0.0}}},
        },
    ]
    events = detect_base_events(records)
    windows = detect_head_on_crossing_windows(records, events, window_radius=1, merge_step_gap=2)
    assert len(windows) == 1
    window = windows[0]
    assert window["anchor_kind"] in {"brake_onset", "attack_advantage_collapse", "collision_threat_spike"}
    assert window["anchor"]["two_circle_gate"] > 0.5
    report = build_head_on_crossing_report(
        summary={
            "episode_id": "test-episode",
            "ego_role": "fighter1",
            "frame_count": len(records),
        },
        windows=windows,
    )
    assert "Head-on Crossing Diagnosis" in report
    assert "Window 1" in report
    assert "Top Negative Terms" in report
