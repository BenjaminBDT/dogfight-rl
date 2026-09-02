from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.models import (
    StatelessHybridActorCritic,
    model_architecture_kwargs,
)
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    validate_policy_checkpoint_payload,
)


@dataclass(frozen=True)
class DeterministicPolicyOutput:
    action_cont: np.ndarray
    action_bin_prob: np.ndarray

    def binary_actions(self, *, threshold: float = 0.5) -> np.ndarray:
        return (self.action_bin_prob >= threshold).astype(np.float32)


@dataclass(frozen=True)
class PolicyBatchOutput:
    action_cont_mean: np.ndarray
    action_cont: np.ndarray
    action_bin_prob: np.ndarray
    action_bin: np.ndarray


def load_policy_model(
    checkpoint_path: str | Path,
    *,
    normalizer: ObservationNormalizer,
    dataset_contract: PolicyDatasetContract,
    device: torch.device,
    context: str,
) -> StatelessHybridActorCritic:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_policy_checkpoint_payload(payload, dataset=dataset_contract, context=context)
    hyperparameters = payload["model_hyperparameters"]
    model = StatelessHybridActorCritic(
        obs_dim=normalizer.obs_dim,
        **model_architecture_kwargs(hyperparameters),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def normalize_policy_observation(
    normalizer: ObservationNormalizer,
    obs: np.ndarray,
) -> np.ndarray:
    if obs.ndim == 0 or obs.shape[-1] != normalizer.obs_dim:
        raise ValueError(
            f"unexpected observation shape {obs.shape}; last dimension must be {normalizer.obs_dim}"
        )
    return normalizer.normalize_np(obs).astype(np.float32, copy=False)


@torch.no_grad()
def policy_output_batch(
    *,
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    obs: np.ndarray,
    device: torch.device,
    mode: str,
    continuous_std: float = 0.0,
    binary_threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> PolicyBatchOutput:
    if obs.ndim != 2 or obs.shape[1] != normalizer.obs_dim:
        raise ValueError(
            f"batch policy expects observations with shape [N, {normalizer.obs_dim}], got {obs.shape}"
        )
    if mode not in {"deterministic", "sampled"}:
        raise ValueError(f"unsupported policy inference mode: {mode}")
    if continuous_std < 0.0:
        raise ValueError("continuous_std must be non-negative")
    normalized = normalize_policy_observation(normalizer, obs)
    output = model(torch.from_numpy(normalized).to(device))
    action_cont_mean = output.action_cont_mean
    action_bin_prob = torch.sigmoid(output.action_bin_logits)
    if mode == "deterministic":
        action_cont = action_cont_mean
        action_bin = action_bin_prob >= binary_threshold
    else:
        noise = torch.randn(
            action_cont_mean.shape,
            dtype=action_cont_mean.dtype,
            device=action_cont_mean.device,
            generator=generator,
        )
        action_cont = (action_cont_mean + continuous_std * noise).clamp(-1.0, 1.0)
        action_bin = torch.bernoulli(action_bin_prob, generator=generator)
    return PolicyBatchOutput(
        action_cont_mean=action_cont_mean.cpu().numpy().astype(np.float32, copy=False),
        action_cont=action_cont.cpu().numpy().astype(np.float32, copy=False),
        action_bin_prob=action_bin_prob.cpu().numpy().astype(np.float32, copy=False),
        action_bin=action_bin.cpu().numpy().astype(np.float32, copy=False),
    )


@torch.no_grad()
def deterministic_policy_output(
    *,
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    obs: np.ndarray,
    device: torch.device,
) -> DeterministicPolicyOutput:
    if obs.shape != (normalizer.obs_dim,):
        raise ValueError(
            f"deterministic policy expects one observation with shape "
            f"({normalizer.obs_dim},), got {obs.shape}"
        )
    output = policy_output_batch(
        model=model,
        normalizer=normalizer,
        obs=obs.reshape(1, -1),
        device=device,
        mode="deterministic",
    )
    return DeterministicPolicyOutput(
        action_cont=output.action_cont[0],
        action_bin_prob=output.action_bin_prob[0],
    )
