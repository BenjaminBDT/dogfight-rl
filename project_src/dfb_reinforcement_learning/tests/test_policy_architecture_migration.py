from __future__ import annotations

import torch

from dfb_reinforcement_learning.models import (
    StatelessHybridActorCritic,
    measure_policy_output_migration_error,
    migrate_policy_parameters,
)
from dfb_reinforcement_learning.policy_contract import OBS_DIM


def _extension_counts(*, shared: int = 0, actor: int = 0, critic: int = 0) -> dict[str, int]:
    return {
        "shared_extension_blocks": shared,
        "actor_extension_blocks": actor,
        "critic_extension_blocks": critic,
    }


def test_depth_migration_preserves_policy_outputs() -> None:
    torch.manual_seed(31)
    source = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    target = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=64,
        num_layers=2,
        shared_extension_blocks=1,
        actor_extension_blocks=2,
    )
    migrate_policy_parameters(
        source=source,
        target=target,
        source_hidden_dim=64,
        target_hidden_dim=64,
        source_extension_counts=_extension_counts(),
    )

    error = measure_policy_output_migration_error(
        source=source,
        target=target,
        observations=torch.randn((128, OBS_DIM)),
    )

    assert error.maximum == 0.0


def test_twofold_width_migration_preserves_policy_outputs_and_breaks_weight_symmetry() -> None:
    torch.manual_seed(37)
    source = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    target = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=128, num_layers=2)
    migrate_policy_parameters(
        source=source,
        target=target,
        source_hidden_dim=64,
        target_hidden_dim=128,
        source_extension_counts=_extension_counts(),
    )

    error = measure_policy_output_migration_error(
        source=source,
        target=target,
        observations=torch.randn((256, OBS_DIM)),
    )

    assert error.maximum < 1e-5
    widened_head = target.action_cont_head.weight.detach()
    assert not torch.equal(widened_head[:, 0], widened_head[:, 1])
    torch.testing.assert_close(
        widened_head[:, 0] + widened_head[:, 1],
        source.action_cont_head.weight.detach()[:, 0],
    )


def test_width_and_depth_migration_can_be_combined() -> None:
    torch.manual_seed(41)
    source = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=32,
        num_layers=1,
        actor_extension_blocks=1,
    )
    with torch.no_grad():
        source.actor_extension_tower[0].gate.fill_(0.05)
    target = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=64,
        num_layers=1,
        shared_extension_blocks=1,
        actor_extension_blocks=2,
        critic_extension_blocks=1,
    )
    migrate_policy_parameters(
        source=source,
        target=target,
        source_hidden_dim=32,
        target_hidden_dim=64,
        source_extension_counts=_extension_counts(actor=1),
    )

    error = measure_policy_output_migration_error(
        source=source,
        target=target,
        observations=torch.randn((128, OBS_DIM)),
    )

    assert error.maximum < 1e-5
    assert torch.count_nonzero(target.actor_extension_tower[1].gate).item() == 0


def test_width_migration_rejects_non_integer_multiplier() -> None:
    source = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=32, num_layers=1)
    target = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=48, num_layers=1)

    try:
        migrate_policy_parameters(
            source=source,
            target=target,
            source_hidden_dim=32,
            target_hidden_dim=48,
            source_extension_counts=_extension_counts(),
        )
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("expected non-integer width migration to fail")
