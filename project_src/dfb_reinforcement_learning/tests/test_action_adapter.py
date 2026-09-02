from __future__ import annotations

import json

from dfb_reinforcement_learning.actions import ActionAdapter


def test_action_adapter_clamps_and_converts() -> None:
    action = ActionAdapter.from_arrays(
        [2.0, -2.0, 0.25, -0.5],
        [0.2, 0.7, 1.0],
    )
    env_action = ActionAdapter.to_environment_action(action)
    payload = json.loads(env_action.json())
    assert payload["throttle"] == 1.0
    assert payload["pitch"] == -1.0
    assert payload["roll"] == 0.25
    assert payload["yaw"] == -0.5
    assert payload["brake"] is False
    assert payload["fire_gun"] is True
    assert payload["repair"] is True
