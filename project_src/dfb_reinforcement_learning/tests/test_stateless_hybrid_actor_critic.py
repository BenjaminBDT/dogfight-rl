from __future__ import annotations

import torch

from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_contract import OBS_DIM


def test_default_architecture_uses_two_residual_blocks_per_tower() -> None:
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM)
    assert len(model.actor_tower) == 2
    assert len(model.critic_tower) == 2


def test_stateless_hybrid_actor_critic_output_shapes() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    obs = torch.zeros((8, obs_dim), dtype=torch.float32)
    output = model(obs)
    assert tuple(output.action_cont_mean.shape) == (8, 4)
    assert tuple(output.action_bin_logits.shape) == (8, 3)
    assert tuple(output.value.shape) == (8,)


def test_stateless_hybrid_actor_critic_uses_split_actor_critic_towers() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=4)
    assert len(model.actor_tower) == 4
    assert len(model.critic_tower) == 4
    actor_param_ids = {id(param) for param in model.actor_tower.parameters()}
    critic_param_ids = {id(param) for param in model.critic_tower.parameters()}
    assert actor_param_ids
    assert critic_param_ids
    assert actor_param_ids.isdisjoint(critic_param_ids)


def test_gated_extension_blocks_are_identity_at_initialization() -> None:
    torch.manual_seed(17)
    baseline = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    expanded = StatelessHybridActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=64,
        num_layers=2,
        shared_extension_blocks=1,
        actor_extension_blocks=2,
        critic_extension_blocks=1,
    )
    expanded.load_state_dict(baseline.state_dict(), strict=False)
    observations = torch.randn((32, OBS_DIM))

    baseline_output = baseline(observations)
    expanded_output = expanded(observations)

    torch.testing.assert_close(expanded_output.action_cont_mean, baseline_output.action_cont_mean)
    torch.testing.assert_close(expanded_output.action_bin_logits, baseline_output.action_bin_logits)
    torch.testing.assert_close(expanded_output.value, baseline_output.value)
    assert torch.count_nonzero(expanded.shared_extension_tower[0].gate).item() == 0
    assert torch.count_nonzero(expanded.actor_extension_tower[0].gate).item() == 0
    assert torch.count_nonzero(expanded.critic_extension_tower[0].gate).item() == 0


def test_popart_value_head_preserves_unnormalized_outputs_when_stats_update() -> None:
    obs_dim = OBS_DIM
    model = StatelessHybridActorCritic(obs_dim=obs_dim, hidden_dim=64, num_layers=2)
    obs = torch.randn((6, obs_dim), dtype=torch.float32)
    before = model(obs).value.detach().clone()

    targets = torch.tensor([2000.0, -3000.0, 1500.0, -2500.0, 300.0, -120.0], dtype=torch.float32)
    model.update_value_normalizer(targets, beta=0.5, min_std=1e-2)
    after = model(obs).value.detach()

    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)
    normalized_targets = model.normalize_values(targets)
    assert float(normalized_targets.std(unbiased=False)) > 0.0
