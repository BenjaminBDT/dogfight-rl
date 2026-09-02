from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    validate_policy_normalizer_payload,
)


@dataclass(frozen=True)
class ObservationNormalizer:
    normalizer_schema_id: str
    policy_contract_id: str
    observation_schema_id: str
    contract_sha256: str
    obs_dim: int
    epsilon: float
    mean: np.ndarray
    std: np.ndarray
    train_row_count: int
    source_dataset_id: str

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        dataset: PolicyDatasetContract | None = None,
    ) -> "ObservationNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("observation normalizer root must be an object")
        return cls.from_payload(payload, dataset=dataset)

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        dataset: PolicyDatasetContract | None = None,
    ) -> "ObservationNormalizer":
        validate_policy_normalizer_payload(payload, dataset=dataset)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        obs_dim = int(payload["obs_dim"])
        if mean.shape != (obs_dim,) or std.shape != (obs_dim,):
            raise ValueError("obs_normalizer mean/std shape mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("obs_normalizer mean/std must be finite")
        if np.any(std <= 0.0):
            raise ValueError("obs_normalizer std must be positive")
        epsilon = float(payload["epsilon"])
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("obs_normalizer epsilon must be finite and positive")
        train_row_count = int(payload["train_row_count"])
        if train_row_count < 0:
            raise ValueError("obs_normalizer train_row_count must be non-negative")
        for index in POLICY_OBSERVATION_SCHEMA.binary_indices:
            if float(mean[index]) != 0.0 or float(std[index]) != 1.0:
                raise ValueError(
                    f"obs_normalizer binary index {index} must use mean=0 and std=1"
                )
        return cls(
            normalizer_schema_id=str(payload["normalizer_schema_id"]),
            policy_contract_id=str(payload["policy_contract_id"]),
            observation_schema_id=str(payload["observation_schema_id"]),
            contract_sha256=str(payload["contract_sha256"]),
            obs_dim=obs_dim,
            epsilon=epsilon,
            mean=mean,
            std=std,
            train_row_count=train_row_count,
            source_dataset_id=str(payload["source_dataset_id"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "normalizer_schema_id": self.normalizer_schema_id,
            "policy_contract_id": self.policy_contract_id,
            "observation_schema_id": self.observation_schema_id,
            "contract_sha256": self.contract_sha256,
            "obs_dim": self.obs_dim,
            "epsilon": self.epsilon,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "train_row_count": self.train_row_count,
            "source_dataset_id": self.source_dataset_id,
        }

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / np.maximum(self.std, self.epsilon)

    def normalize_torch(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=obs.dtype, device=obs.device)
        std = torch.as_tensor(self.std, dtype=obs.dtype, device=obs.device)
        eps = torch.as_tensor(self.epsilon, dtype=obs.dtype, device=obs.device)
        return (obs - mean) / torch.maximum(std, eps)
