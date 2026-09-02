from __future__ import annotations

import copy

import numpy as np
import pytest

from dfb_reinforcement_learning.obs.combat_geometry import compute_attack_geometry
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_contract import POLICY_CONTRACT_ID, POLICY_CONTRACT_SHA256


def _subsystems() -> list[dict[str, object]]:
    return [
        {"name": "LeftWing", "hit_points": 40.0, "max_hit_points": 50.0, "stage": "Intact"},
        {"name": "RightWing", "hit_points": 50.0, "max_hit_points": 50.0, "stage": "Intact"},
        {"name": "PitchTail", "hit_points": 45.0, "max_hit_points": 50.0, "stage": "Intact"},
        {"name": "YawTail", "hit_points": 50.0, "max_hit_points": 50.0, "stage": "Intact"},
        {"name": "Engine", "hit_points": 30.0, "max_hit_points": 50.0, "stage": "Intact"},
    ]


def _aircraft(
    *,
    role: str,
    position: list[float],
    orientation_quat: list[float],
    velocity: list[float],
) -> dict[str, object]:
    return {
        "role": role,
        "position": position,
        "orientation_quat": orientation_quat,
        "linear_velocity": velocity,
        "angular_velocity_deg": [180.0, -90.0, 45.0],
        "hit_points": 80.0,
        "throttle": 0.6,
        "brake": role == "fighter1",
        "stall_factor": 0.2,
        "gun_overheated": role == "fighter2",
        "gun_heat": 0.3,
        "is_firing": role == "fighter1",
        "repairing": False,
        "repair_elapsed_seconds": 0.0,
        "out_of_bounds_seconds": 0.0,
        "subsystems": _subsystems(),
    }


def _world_state() -> dict[str, object]:
    half_sqrt = float(np.sqrt(0.5))
    return {
        "tick": 1200,
        "sim_time_seconds": 190.0,
        "scene_name": "open_ho",
        "arena": {"arena_radius": 5000.0},
        "aircraft": [
            _aircraft(
                role="fighter1",
                position=[0.0, 600.0, 0.0],
                orientation_quat=[0.0, 0.0, 0.0, 1.0],
                velocity=[0.0, 0.0, 60.0],
            ),
            _aircraft(
                role="fighter2",
                position=[0.0, 600.0, 300.0],
                orientation_quat=[0.0, half_sqrt, 0.0, half_sqrt],
                velocity=[30.0, 0.0, 0.0],
            ),
        ],
    }


def test_policy_observation_builds_exact_contract_vector() -> None:
    result = PolicyObservationAdapter().build(
        _world_state(),
        "fighter1",
        episode_start_sim_time_seconds=100.0,
    )
    assert result["policy_contract_id"] == POLICY_CONTRACT_ID
    assert result["contract_sha256"] == POLICY_CONTRACT_SHA256
    assert result["schema_id"] == POLICY_OBSERVATION_SCHEMA.schema_id
    assert result["vector"].shape == (69,)
    assert result["vector"].dtype == np.float32
    assert np.isfinite(result["vector"]).all()

    components = result["components"]
    np.testing.assert_allclose(
        components["self_orientation_world_6d"],
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        components["self_angular_velocity_body"],
        np.asarray([1.0, -0.5, 0.25], dtype=np.float32),
        atol=1e-6,
    )
    assert float(components["episode_time_norm"][0]) == pytest.approx(0.5)
    assert float(components["self_brake_active"][0]) == 1.0
    assert float(components["self_fire_gun_active"][0]) == 1.0
    assert float(components["enemy_gun_overheated"][0]) == 1.0
    assert float(components["self_repair_active"][0]) == 0.0
    assert float(components["self_repair_seconds_norm"][0]) == 0.0
    assert float(components["self_out_of_bounds_active"][0]) == 0.0

    for field in POLICY_OBSERVATION_SCHEMA.fields:
        np.testing.assert_array_equal(
            result["vector"][field.value_slice],
            components[field.name],
        )


def test_policy_observation_tactical_fields_match_shared_geometry_for_both_roles() -> None:
    world = _world_state()
    fighter1 = world["aircraft"][0]
    fighter2 = world["aircraft"][1]
    f1_metrics = compute_attack_geometry(attacker_state=fighter1, defender_state=fighter2)
    f2_metrics = compute_attack_geometry(attacker_state=fighter2, defender_state=fighter1)

    f1 = PolicyObservationAdapter().build(
        world,
        "fighter1",
        episode_start_sim_time_seconds=100.0,
    )["components"]
    f2 = PolicyObservationAdapter().build(
        world,
        "fighter2",
        episode_start_sim_time_seconds=100.0,
    )["components"]

    assert float(f1["self_tracking_quality"][0]) == pytest.approx(f1_metrics.tracking_quality)
    assert float(f1["enemy_tracking_quality"][0]) == pytest.approx(f2_metrics.tracking_quality)
    assert float(f1["self_tail_hold_score"][0]) == pytest.approx(f1_metrics.tail_hold_score)
    assert float(f1["enemy_tail_hold_score"][0]) == pytest.approx(f2_metrics.tail_hold_score)
    assert float(f1["self_shot_feasibility"][0]) == pytest.approx(f1_metrics.shot_feasibility)
    assert float(f1["enemy_shot_feasibility"][0]) == pytest.approx(f2_metrics.shot_feasibility)
    assert float(f2["self_tracking_quality"][0]) == pytest.approx(float(f1["enemy_tracking_quality"][0]))
    assert float(f2["enemy_tracking_quality"][0]) == pytest.approx(float(f1["self_tracking_quality"][0]))


def test_episode_time_is_relative_and_clamped_at_zero() -> None:
    result = PolicyObservationAdapter().build(
        _world_state(),
        "fighter1",
        episode_start_sim_time_seconds=200.0,
    )
    assert float(result["components"]["episode_time_norm"][0]) == 0.0


def test_out_of_bounds_state_detects_boundary_entry_at_zero_elapsed_time() -> None:
    world = _world_state()
    world["aircraft"][0]["position"] = [5000.0, 600.0, 0.0]
    result = PolicyObservationAdapter().build(
        world,
        "fighter1",
        episode_start_sim_time_seconds=100.0,
    )
    assert float(result["components"]["self_out_of_bounds_active"][0]) == 1.0
    assert float(result["components"]["self_out_of_bounds_seconds_norm"][0]) == 0.0


def test_repair_state_separates_activation_from_zero_elapsed_time() -> None:
    world = _world_state()
    world["aircraft"][0]["repairing"] = True
    result = PolicyObservationAdapter().build(
        world,
        "fighter1",
        episode_start_sim_time_seconds=100.0,
    )
    assert float(result["components"]["self_repair_active"][0]) == 1.0
    assert float(result["components"]["self_repair_seconds_norm"][0]) == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda world: world["aircraft"][0].pop("brake"), "fighter1.brake"),
        (lambda world: world["aircraft"][0].pop("orientation_quat"), "orientation_quat"),
        (lambda world: world["aircraft"][0]["subsystems"].pop(), "missing subsystem Engine"),
        (lambda world: world["aircraft"][1].update(role="fighter1"), "duplicate aircraft role"),
    ],
)
def test_policy_observation_fails_closed_on_missing_or_ambiguous_state(mutation, message: str) -> None:
    world = copy.deepcopy(_world_state())
    mutation(world)
    with pytest.raises(ValueError, match=message):
        PolicyObservationAdapter().build(
            world,
            "fighter1",
            episode_start_sim_time_seconds=100.0,
        )


def test_policy_observation_rejects_coincident_aircraft() -> None:
    world = _world_state()
    world["aircraft"][1]["position"] = list(world["aircraft"][0]["position"])
    with pytest.raises(ValueError, match="positions must differ"):
        PolicyObservationAdapter().build(
            world,
            "fighter1",
            episode_start_sim_time_seconds=100.0,
        )
