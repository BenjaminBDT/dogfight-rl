from __future__ import annotations

from dfb_reinforcement_learning.rewards import PolicyRewardComposer
from dfb_reinforcement_learning.tools.audit_attack_component_path import build_path_audit_report


def _state(
    *,
    position: list[float],
    velocity: list[float],
    forward: list[float],
) -> dict[str, object]:
    return {
        "position": position,
        "linear_velocity": velocity,
        "forward": forward,
        "orientation_quat": [0.0, 0.0, 0.0, 1.0],
        "subsystems": [],
    }


def test_path_report_contains_reference_speed() -> None:
    audit = {
        "reference_step_index": 10,
        "reference_speed_mps": 72.5,
        "start_step_index": 20,
        "end_step_index": 22,
        "rows": [
            {
                "step_index": 20,
                "actual_speed_mps": 40.0,
                "counterfactual_speed_mps": 72.5,
                "shot_contribution_delta": 0.1,
                "lead_contribution_delta": -0.2,
                "attack_advantage_weighted_delta": -0.5,
            }
        ],
    }
    report = build_path_audit_report(episode_id="demo", ego_role="fighter1", audit=audit)
    assert "Reference Speed: `72.500 m/s`" in report
    assert "| 20 | 40.000 | 72.500 | 0.100 | -0.200 | -0.500 |" in report


def test_counterfactual_higher_speed_can_raise_shot_component() -> None:
    composer = PolicyRewardComposer()
    attacker = _state(position=[0.0, 150.0, 0.0], velocity=[0.0, 0.0, 40.0], forward=[0.0, 0.0, 1.0])
    defender = _state(position=[0.0, 150.0, 200.0], velocity=[0.0, 0.0, -60.0], forward=[0.0, 0.0, -1.0])
    from dfb_reinforcement_learning.tools.audit_attack_component_path import _row_for_step

    row = _row_for_step(
        composer,
        attacker_state=attacker,
        defender_state=defender,
        counterfactual_speed_mps=80.0,
    )
    assert row["counterfactual"]["shot_contribution"] >= row["actual"]["shot_contribution"]
