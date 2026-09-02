from __future__ import annotations

import numpy as np

from dfb_reinforcement_learning.actions import (
    ActionAdapter,
    HYBRID_ACTION_DIM_BINARY,
    HYBRID_ACTION_DIM_CONTINUOUS,
)
from dfb_reinforcement_learning.policy_contract import ACTION_BIN_DIM, ACTION_CONT_DIM


def test_policy_action_dimensions_match_contract() -> None:
    assert HYBRID_ACTION_DIM_CONTINUOUS == ACTION_CONT_DIM
    assert HYBRID_ACTION_DIM_BINARY == ACTION_BIN_DIM


def test_policy_action_adapter_uses_contract_order() -> None:
    action = ActionAdapter.from_arrays(
        np.asarray([0.25, -0.5, 0.75, -1.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert action.continuous_array().tolist() == [0.25, -0.5, 0.75, -1.0]
    assert action.binary_array().tolist() == [1.0, 0.0, 1.0]
