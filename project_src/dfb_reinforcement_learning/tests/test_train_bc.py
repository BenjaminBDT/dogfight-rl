from __future__ import annotations

import torch
from torch.nn import functional as F

from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_contract import OBS_DIM
from dfb_reinforcement_learning.train.train_bc import (
    BcMetrics,
    _compute_batch_losses,
    _monitored_bc_loss,
    _resolve_monitored_bc_loss,
)


def test_bc_losses_apply_per_sample_demonstration_weights() -> None:
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=16, num_layers=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_bin_head.bias.fill_(2.0)

    batch = {
        "obs": torch.zeros((2, OBS_DIM), dtype=torch.float32),
        "action_cont": torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.float32),
        "action_bin": torch.tensor([[1.0] * 3, [0.0] * 3], dtype=torch.float32),
        "sample_weight": torch.tensor([3.0, 1.0], dtype=torch.float32),
    }
    total, continuous, binary, accuracy, _ = _compute_batch_losses(
        model,
        batch,
        continuous_loss_weight=1.0,
        binary_loss_weight=1.0,
    )

    cont_per_sample = F.smooth_l1_loss(
        torch.zeros_like(batch["action_cont"]),
        batch["action_cont"],
        reduction="none",
    ).mean(dim=1)
    bin_per_sample = F.binary_cross_entropy_with_logits(
        torch.full_like(batch["action_bin"], 2.0),
        batch["action_bin"],
        reduction="none",
    ).mean(dim=1)
    expected_cont = (cont_per_sample * batch["sample_weight"]).sum() / 4.0
    expected_bin = (bin_per_sample * batch["sample_weight"]).sum() / 4.0

    assert torch.allclose(continuous, expected_cont)
    assert torch.allclose(binary, expected_bin)
    assert torch.allclose(total, expected_cont + expected_bin)
    assert float(accuracy) == 0.75


def test_bc_checkpoint_monitor_selects_requested_metric() -> None:
    metrics = BcMetrics(
        total_loss=0.8,
        continuous_loss=0.2,
        binary_loss=0.6,
        binary_accuracy=0.9,
        fire_positive_rate=0.1,
        repair_positive_rate=0.2,
        brake_positive_rate=0.3,
        sample_count=10,
        sample_weight_sum=10.0,
    )

    assert _monitored_bc_loss(metrics, "total_loss") == 0.8
    assert _monitored_bc_loss(metrics, "continuous_loss") == 0.2
    assert _monitored_bc_loss(metrics, "binary_loss") == 0.6


def test_bc_checkpoint_payload_monitor_uses_validation_before_training() -> None:
    payload = {
        "train_metrics": {"continuous_loss": 0.1},
        "val_metrics": {"continuous_loss": 0.3},
    }

    assert (
        _resolve_monitored_bc_loss(payload, monitor="continuous_loss")
        == 0.3
    )
