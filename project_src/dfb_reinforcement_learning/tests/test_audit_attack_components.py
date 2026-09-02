from __future__ import annotations

import numpy as np

from dfb_reinforcement_learning.rewards import PolicyRewardComposer
from dfb_reinforcement_learning.tools.audit_attack_components import (
    build_attack_component_report,
    sweep_attack_components,
)


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


def test_sweep_attack_components_preserves_requested_scales() -> None:
    composer = PolicyRewardComposer()
    attacker = _state(position=[0.0, 150.0, 0.0], velocity=[0.0, 0.0, 80.0], forward=[0.0, 0.0, 1.0])
    defender = _state(position=[0.0, 150.0, 200.0], velocity=[0.0, 0.0, -60.0], forward=[0.0, 0.0, -1.0])
    rows = sweep_attack_components(
        composer=composer,
        attacker_state=attacker,
        defender_state=defender,
        speed_scales=[0.5, 1.0, 1.2],
    )
    assert [round(row["speed_scale"], 2) for row in rows] == [0.5, 1.0, 1.2]
    assert np.isclose(rows[0]["attacker_speed_mps"], 40.0)
    assert np.isclose(rows[1]["attacker_speed_mps"], 80.0)
    assert np.isclose(rows[2]["attacker_speed_mps"], 96.0)


def test_attack_component_report_mentions_main_contributions() -> None:
    rows = [
        {
            "speed_scale": 1.0,
            "attacker_speed_mps": 80.0,
            "tau_seconds": 0.12,
            "tracking_contribution": 0.4,
            "shot_contribution": 0.6,
            "tail_contribution": 0.0,
            "attack_advantage_unweighted": 1.2,
            "attack_advantage_weighted": 6.0,
        }
    ]
    report = build_attack_component_report(
        episode_id="demo",
        ego_role="fighter1",
        step_index=42,
        rows=rows,
    )
    assert "Attack Component Sweep Audit" in report
    assert "`demo`" in report
    assert "0.600" in report
    assert "6.000" in report
