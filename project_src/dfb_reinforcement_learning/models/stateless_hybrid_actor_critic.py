from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActorCriticOutput:
    action_cont_mean: torch.Tensor
    action_bin_logits: torch.Tensor
    value: torch.Tensor


class MlpResidualBlock(nn.Module):
    def __init__(self, *, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.linear1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        hidden = self.norm2(hidden)
        hidden = residual + hidden
        hidden = self.activation(hidden)
        return self.dropout(hidden)


class GatedResidualExtensionBlock(nn.Module):
    """A residual MLP extension that is exactly identity at initialization."""

    def __init__(self, *, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gate = nn.Parameter(torch.zeros((hidden_dim,), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        hidden = self.norm2(hidden)
        return x + self.gate * self.dropout(hidden)


class PopArtValueHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.weight = nn.Parameter(torch.empty((1, hidden_dim)))
        self.bias = nn.Parameter(torch.empty((1,)))
        self.register_buffer("mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("mean_sq", torch.ones((), dtype=torch.float32))
        self.register_buffer("std", torch.ones((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.mean.zero_()
        self.mean_sq.fill_(1.0)
        self.std.fill_(1.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.linear(hidden, self.weight, self.bias).squeeze(-1)
        return self.denormalize(normalized)

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        return normalized_values * self.std + self.mean

    def normalize(self, unnormalized_values: torch.Tensor) -> torch.Tensor:
        return (unnormalized_values - self.mean) / self.std.clamp_min(1e-8)

    @torch.no_grad()
    def update_stats(
        self,
        targets: torch.Tensor,
        *,
        beta: float,
        min_std: float,
    ) -> None:
        if targets.numel() <= 0:
            return
        safe_beta = float(min(max(beta, 0.0), 0.999999))
        safe_min_std = max(float(min_std), 1e-6)

        old_mean = self.mean.clone()
        old_std = self.std.clone()

        batch_mean = targets.mean()
        batch_mean_sq = targets.square().mean()
        new_mean = safe_beta * self.mean + (1.0 - safe_beta) * batch_mean
        new_mean_sq = safe_beta * self.mean_sq + (1.0 - safe_beta) * batch_mean_sq
        new_var = (new_mean_sq - new_mean.square()).clamp_min(safe_min_std * safe_min_std)
        new_std = new_var.sqrt()

        scale = (old_std / new_std).to(self.weight.dtype)
        shift = ((old_mean - new_mean) / new_std).to(self.bias.dtype)
        self.weight.data.mul_(scale)
        self.bias.data.mul_(scale).add_(shift)

        self.mean.copy_(new_mean)
        self.mean_sq.copy_(new_mean_sq)
        self.std.copy_(new_std)


class StatelessHybridActorCritic(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        shared_extension_blocks: int = 0,
        actor_extension_blocks: int = 0,
        critic_extension_blocks: int = 0,
    ) -> None:
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        extension_counts = {
            "shared_extension_blocks": shared_extension_blocks,
            "actor_extension_blocks": actor_extension_blocks,
            "critic_extension_blocks": critic_extension_blocks,
        }
        for name, count in extension_counts.items():
            if count < 0:
                raise ValueError(f"{name} must be non-negative")

        self.shared_stem = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.actor_tower = nn.ModuleList(
            [MlpResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.critic_tower = nn.ModuleList(
            [MlpResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.shared_extension_tower = nn.ModuleList(
            [
                GatedResidualExtensionBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(shared_extension_blocks)
            ]
        )
        self.actor_extension_tower = nn.ModuleList(
            [
                GatedResidualExtensionBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(actor_extension_blocks)
            ]
        )
        self.critic_extension_tower = nn.ModuleList(
            [
                GatedResidualExtensionBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(critic_extension_blocks)
            ]
        )
        self.action_cont_head = nn.Linear(hidden_dim, 4)
        self.action_bin_head = nn.Linear(hidden_dim, 3)
        self.value_head = PopArtValueHead(hidden_dim)

    def forward(self, obs: torch.Tensor) -> ActorCriticOutput:
        shared = self.shared_stem(obs)
        for block in self.shared_extension_tower:
            shared = block(shared)

        actor_hidden = shared
        for block in self.actor_tower:
            actor_hidden = block(actor_hidden)
        for block in self.actor_extension_tower:
            actor_hidden = block(actor_hidden)

        critic_hidden = shared
        for block in self.critic_tower:
            critic_hidden = block(critic_hidden)
        for block in self.critic_extension_tower:
            critic_hidden = block(critic_hidden)

        action_cont_mean = torch.tanh(self.action_cont_head(actor_hidden))
        action_bin_logits = self.action_bin_head(actor_hidden)
        value = self.value_head(critic_hidden)
        return ActorCriticOutput(
            action_cont_mean=action_cont_mean,
            action_bin_logits=action_bin_logits,
            value=value,
        )

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.value_head.normalize(values)

    @torch.no_grad()
    def update_value_normalizer(
        self,
        targets: torch.Tensor,
        *,
        beta: float,
        min_std: float,
    ) -> None:
        self.value_head.update_stats(targets, beta=beta, min_std=min_std)


def model_architecture_kwargs(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return constructor kwargs, treating absent extension fields as legacy zeros."""

    return {
        "hidden_dim": int(hyperparameters["hidden_dim"]),
        "num_layers": int(hyperparameters["num_layers"]),
        "dropout": float(hyperparameters["dropout"]),
        "shared_extension_blocks": int(hyperparameters.get("shared_extension_blocks", 0)),
        "actor_extension_blocks": int(hyperparameters.get("actor_extension_blocks", 0)),
        "critic_extension_blocks": int(hyperparameters.get("critic_extension_blocks", 0)),
    }
