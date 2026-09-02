from __future__ import annotations

import numpy as np
import pytest

from dfb_reinforcement_learning.obs.combat_geometry import (
    SHOT_CORE_WEIGHT,
    SHOT_OUTER_WEIGHT,
    compute_attack_geometry,
    quaternion_xyzw_to_rotation_matrix,
    rotation_matrix_to_6d,
)
from dfb_reinforcement_learning.rewards import PolicyRewardComposer


def _aircraft(
    *,
    position: list[float],
    orientation_quat: list[float],
    velocity: list[float] | None = None,
) -> dict[str, object]:
    return {
        "position": position,
        "orientation_quat": orientation_quat,
        "linear_velocity": velocity or [0.0, 0.0, 0.0],
        "subsystems": [
            {"name": "LeftWing", "stage": "Intact"},
            {"name": "RightWing", "stage": "Intact"},
            {"name": "PitchTail", "stage": "Intact"},
            {"name": "YawTail", "stage": "Intact"},
            {"name": "Engine", "stage": "Intact"},
        ],
    }


def test_quaternion_conversion_normalizes_and_rejects_invalid_values() -> None:
    identity = quaternion_xyzw_to_rotation_matrix(
        [0.0, 0.0, 0.0, 2.0],
        field_name="orientation",
    )
    np.testing.assert_allclose(identity, np.eye(3, dtype=np.float32), atol=1e-6)
    with pytest.raises(ValueError, match="zero length"):
        quaternion_xyzw_to_rotation_matrix([0.0, 0.0, 0.0, 0.0], field_name="orientation")
    with pytest.raises(ValueError, match="finite xyzw"):
        quaternion_xyzw_to_rotation_matrix([0.0, np.nan, 0.0, 1.0], field_name="orientation")


def test_rotation_6d_uses_first_two_columns_in_column_major_order() -> None:
    rotation = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        rotation_matrix_to_6d(rotation),
        np.asarray([1.0, 4.0, 7.0, 2.0, 5.0, 8.0], dtype=np.float32),
    )


def test_attack_geometry_uses_each_aircraft_forward_for_tracking() -> None:
    self_state = _aircraft(
        position=[0.0, 0.0, 0.0],
        orientation_quat=[0.0, 0.0, 0.0, 1.0],
    )
    half_sqrt = float(np.sqrt(0.5))
    enemy_state = _aircraft(
        position=[0.0, 0.0, 200.0],
        orientation_quat=[0.0, half_sqrt, 0.0, half_sqrt],
    )

    self_metrics = compute_attack_geometry(attacker_state=self_state, defender_state=enemy_state)
    enemy_metrics = compute_attack_geometry(attacker_state=enemy_state, defender_state=self_state)

    assert self_metrics.tracking_quality == pytest.approx(1.0)
    assert enemy_metrics.tracking_quality == pytest.approx(0.5)
    assert self_metrics.tail_hold_score == pytest.approx(0.25)
    assert enemy_metrics.tail_hold_score == pytest.approx(0.0)


def test_shot_feasibility_applies_only_box_level_time_gate() -> None:
    attacker = _aircraft(
        position=[0.0, 0.0, 0.0],
        orientation_quat=[0.0, 0.0, 0.0, 1.0],
    )
    defender = _aircraft(
        position=[0.0, 0.0, 200.0],
        orientation_quat=[0.0, 0.0, 0.0, 1.0],
    )
    metrics = compute_attack_geometry(attacker_state=attacker, defender_state=defender)
    weighted_box_score = (
        SHOT_OUTER_WEIGHT * metrics.shot_outer_score
        + SHOT_CORE_WEIGHT * metrics.shot_core_score
    )
    assert 0.0 < metrics.tau_gate < 1.0
    assert metrics.fire_alignment == pytest.approx(1.0)
    assert metrics.shot_feasibility == pytest.approx(weighted_box_score)
    assert metrics.shot_feasibility > metrics.tau_gate * weighted_box_score


def test_reward_attack_components_use_canonical_geometry() -> None:
    attacker = _aircraft(
        position=[0.0, 0.0, 0.0],
        orientation_quat=[0.0, 0.0, 0.0, 1.0],
    )
    defender = _aircraft(
        position=[10.0, 0.0, 250.0],
        orientation_quat=[0.0, 0.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 20.0],
    )
    expected = compute_attack_geometry(attacker_state=attacker, defender_state=defender)
    actual = PolicyRewardComposer()._compute_attack_advantage(
        attacker_state=attacker,
        defender_state=defender,
    )
    assert actual.tracking_quality == pytest.approx(expected.tracking_quality)
    assert actual.tail_hold_score == pytest.approx(expected.tail_hold_score)
    assert actual.shot_feasibility == pytest.approx(expected.shot_feasibility)
