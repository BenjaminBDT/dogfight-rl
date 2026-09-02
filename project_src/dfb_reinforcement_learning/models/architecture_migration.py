from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .stateless_hybrid_actor_critic import StatelessHybridActorCritic


@dataclass(frozen=True)
class PolicyOutputMigrationError:
    action_cont_mean_max_abs: float
    action_bin_logits_max_abs: float
    action_bin_probability_max_abs: float
    value_max_abs: float
    value_normalized_max_abs: float

    @property
    def maximum(self) -> float:
        return max(
            self.action_cont_mean_max_abs,
            self.action_bin_probability_max_abs,
            self.value_normalized_max_abs,
        )


def _split_coefficients(*, multiplier: int, tensor: torch.Tensor) -> torch.Tensor:
    if multiplier == 2:
        return torch.tensor((0.25, 0.75), dtype=tensor.dtype, device=tensor.device)
    raw = torch.arange(1, multiplier + 1, dtype=tensor.dtype, device=tensor.device)
    return raw / raw.sum()


def _expand_hidden_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_hidden_dim: int,
    target_hidden_dim: int,
) -> torch.Tensor:
    if source.shape == target.shape:
        return source.detach().clone()
    if target_hidden_dim % source_hidden_dim != 0:
        raise ValueError("target hidden dimension must be an integer multiple of source hidden dimension")
    multiplier = target_hidden_dim // source_hidden_dim
    if multiplier < 1:
        raise ValueError("target hidden dimension must not be smaller than source hidden dimension")

    if source.ndim == 1:
        if source.shape != (source_hidden_dim,) or target.shape != (target_hidden_dim,):
            raise ValueError(f"unsupported one-dimensional migration shape {source.shape} -> {target.shape}")
        return source.repeat_interleave(multiplier, dim=0)

    if source.ndim != 2:
        raise ValueError(f"unsupported migration rank {source.ndim} for {source.shape} -> {target.shape}")

    source_rows, source_columns = source.shape
    target_rows, target_columns = target.shape
    expanded = source
    if source_rows == source_hidden_dim and target_rows == target_hidden_dim:
        expanded = expanded.repeat_interleave(multiplier, dim=0)
    elif source_rows != target_rows:
        raise ValueError(f"unsupported output migration shape {source.shape} -> {target.shape}")

    if source_columns == source_hidden_dim and target_columns == target_hidden_dim:
        expanded = expanded.repeat_interleave(multiplier, dim=1)
        coefficients = _split_coefficients(multiplier=multiplier, tensor=expanded)
        expanded = expanded * coefficients.repeat(source_hidden_dim).reshape(1, -1)
    elif source_columns != target_columns:
        raise ValueError(f"unsupported input migration shape {source.shape} -> {target.shape}")

    if expanded.shape != target.shape:
        raise ValueError(f"migration produced shape {expanded.shape}, expected {target.shape}")
    return expanded.detach().clone()


def _is_new_extension_parameter(
    name: str,
    *,
    source_extension_counts: Mapping[str, int],
) -> bool:
    prefixes = {
        "shared_extension_tower": "shared_extension_blocks",
        "actor_extension_tower": "actor_extension_blocks",
        "critic_extension_tower": "critic_extension_blocks",
    }
    for prefix, count_key in prefixes.items():
        marker = f"{prefix}."
        if not name.startswith(marker):
            continue
        remainder = name[len(marker) :]
        raw_index, separator, _ = remainder.partition(".")
        if not separator or not raw_index.isdigit():
            return False
        return int(raw_index) >= int(source_extension_counts[count_key])
    return False


@torch.no_grad()
def migrate_policy_parameters(
    *,
    source: StatelessHybridActorCritic,
    target: StatelessHybridActorCritic,
    source_hidden_dim: int,
    target_hidden_dim: int,
    source_extension_counts: Mapping[str, int],
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    migrated_state: dict[str, torch.Tensor] = {}
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            if not _is_new_extension_parameter(
                name,
                source_extension_counts=source_extension_counts,
            ):
                raise ValueError(f"target parameter {name!r} has no source migration mapping")
            migrated_state[name] = target_tensor
            continue
        migrated_state[name] = _expand_hidden_tensor(
            source_tensor,
            target_tensor,
            source_hidden_dim=source_hidden_dim,
            target_hidden_dim=target_hidden_dim,
        )

    unexpected_source_names = sorted(set(source_state).difference(target_state))
    if unexpected_source_names:
        raise ValueError(
            "target architecture dropped source parameters: " + ", ".join(unexpected_source_names)
        )
    target.load_state_dict(migrated_state, strict=True)


@torch.no_grad()
def measure_policy_output_migration_error(
    *,
    source: StatelessHybridActorCritic,
    target: StatelessHybridActorCritic,
    observations: torch.Tensor,
) -> PolicyOutputMigrationError:
    source_was_training = source.training
    target_was_training = target.training
    source.eval()
    target.eval()
    try:
        source_output = source(observations)
        target_output = target(observations)
    finally:
        source.train(source_was_training)
        target.train(target_was_training)
    return PolicyOutputMigrationError(
        action_cont_mean_max_abs=float(
            (target_output.action_cont_mean - source_output.action_cont_mean).abs().max().item()
        ),
        action_bin_logits_max_abs=float(
            (target_output.action_bin_logits - source_output.action_bin_logits).abs().max().item()
        ),
        action_bin_probability_max_abs=float(
            (
                torch.sigmoid(target_output.action_bin_logits)
                - torch.sigmoid(source_output.action_bin_logits)
            )
            .abs()
            .max()
            .item()
        ),
        value_max_abs=float((target_output.value - source_output.value).abs().max().item()),
        value_normalized_max_abs=float(
            (
                target.normalize_values(target_output.value)
                - source.normalize_values(source_output.value)
            )
            .abs()
            .max()
            .item()
        ),
    )
