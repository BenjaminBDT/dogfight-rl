from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dfb_reinforcement_learning.rewards.reward_diagnostics import (
    RewardDiagnosticsAccumulator,
)
from dfb_reinforcement_learning.train.ppo_diagnostics import (
    summarize_actions,
    summarize_values,
)


def test_reward_diagnostics_accumulates_and_merges_compact_payloads() -> None:
    first = RewardDiagnosticsAccumulator()
    first.add(
        SimpleNamespace(
            total=2.0,
            attack_advantage=1.5,
            fire_command_bonus=0.5,
            self_destroy_penalty=0.0,
        )
    )
    first.add(
        SimpleNamespace(
            total=-4.0,
            attack_advantage=-0.5,
            fire_command_bonus=0.0,
            self_destroy_penalty=-3.0,
        )
    )

    second = RewardDiagnosticsAccumulator()
    second.merge_payload(first.drain_raw_payload())
    summary = second.summary()

    assert summary["sample_count"] == 2
    assert summary["components"]["total"]["mean"] == -1.0
    assert summary["components"]["total"]["mean_abs"] == 3.0
    assert summary["components"]["total"]["positive_fraction"] == 0.5
    assert summary["components"]["total"]["negative_fraction"] == 0.5
    assert summary["components"]["self_destroy_penalty"]["mean"] == -1.5
    assert summary["components"]["fire_command_bonus"]["mean"] == 0.25
    assert first.summary()["sample_count"] == 0


def test_action_diagnostics_excludes_episode_boundaries_from_deltas() -> None:
    continuous = np.zeros((3, 1, 4), dtype=np.float32)
    continuous[1, 0, 2] = 1.0
    continuous[2, 0, 2] = -1.0
    binary = np.zeros((3, 1, 3), dtype=np.float32)
    binary[:2, 0, 1] = 1.0
    probabilities = np.full_like(binary, 0.25)
    dones = np.zeros((3, 1), dtype=np.float32)
    dones[1, 0] = 1.0

    summary = summarize_actions(
        continuous=continuous,
        binary=binary,
        binary_probabilities=probabilities,
        dones=dones,
    )

    assert summary["continuous"]["mean_abs_step_delta"]["roll"] == 1.0
    assert summary["binary"]["probability_mean"]["fire_gun"] == 0.25
    assert np.isclose(summary["binary"]["on_fraction"]["fire_gun"], 2.0 / 3.0)
    assert summary["binary"]["mean_on_run_steps"]["fire_gun"] == 2.0


def test_action_diagnostics_excludes_inactive_policy_slots() -> None:
    continuous = np.zeros((2, 2, 4), dtype=np.float32)
    continuous[:, 1, :] = 1.0
    binary = np.zeros((2, 2, 3), dtype=np.float32)
    binary[:, 1, :] = 1.0
    probabilities = np.zeros_like(binary)
    probabilities[:, 1, :] = 1.0
    dones = np.zeros((2, 2), dtype=np.float32)
    valid_mask = np.asarray([[True, False], [True, False]])

    summary = summarize_actions(
        continuous=continuous,
        binary=binary,
        binary_probabilities=probabilities,
        dones=dones,
        valid_mask=valid_mask,
    )

    assert summary["sample_count"] == 2
    assert summary["continuous"]["mean"]["roll"] == 0.0
    assert summary["binary"]["on_fraction"]["fire_gun"] == 0.0


def test_value_diagnostics_reports_raw_error_and_explained_variance() -> None:
    summary = summarize_values(
        values=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        returns=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
        popart_mean_before=0.0,
        popart_std_before=1.0,
        popart_mean_after=0.5,
        popart_std_after=2.0,
    )

    assert summary["raw_mae"] == 2.0
    assert np.isclose(summary["raw_rmse"], np.sqrt(14.0 / 3.0))
    assert summary["explained_variance"] == 0.75
    assert summary["popart_mean_after"] == 0.5
