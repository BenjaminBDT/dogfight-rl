from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.opponents import OpponentActionProvider, OpponentPoolSpec, materialize_opponent_pool
from dfb_reinforcement_learning.policy_assets import (
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    load_policy_dataset_contract,
)
from dfb_reinforcement_learning.policy_contract import (
    ACTION_SCHEMA_ID,
    DATASET_SCHEMA_ID,
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    OBS_DIM,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)


def _write_dataset(root: Path) -> None:
    identity = {
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
    }
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                **identity,
                "dataset_id": "dataset-fixture",
                "schema_path": "schema.json",
                "obs_normalizer_path": "obs_normalizer.json",
                "statistics": {"obs_dim": OBS_DIM},
            }
        ),
        encoding="utf-8",
    )
    (root / "schema.json").write_text(json.dumps(identity), encoding="utf-8")
    (root / "obs_normalizer.json").write_text(
        json.dumps(
            {
                "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
                "policy_contract_id": POLICY_CONTRACT_ID,
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "contract_sha256": POLICY_CONTRACT_SHA256,
                "obs_dim": OBS_DIM,
                "epsilon": 1e-6,
                "mean": [0.0] * OBS_DIM,
                "std": [1.0] * OBS_DIM,
                "train_row_count": 1,
                "source_dataset_id": "dataset-fixture",
            }
        ),
        encoding="utf-8",
    )


def _checkpoint_metadata(dataset) -> dict[str, object]:
    return {
        **checkpoint_contract_metadata(dataset),
        "model_hyperparameters": checkpoint_model_hyperparameters(
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            continuous_action_std=None,
            popart_beta=None,
            popart_min_std=None,
        ),
        "training_stage": "test",
        "update_index": 0,
        "global_step": 0,
    }


def test_opponent_pool_resolves_relative_checkpoint_paths(tmp_path: Path) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"placeholder")
    checkpoint_b.write_bytes(b"placeholder")
    pool_json = tmp_path / "opponents.json"
    pool_json.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "sampled",
                        "kind": "sampled_checkpoints",
                        "checkpoints": ["a.pt", "b.pt"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = OpponentPoolSpec.from_json(pool_json)
    assert spec.entries[0].checkpoints == (str(checkpoint_a.resolve()), str(checkpoint_b.resolve()))


def test_materialized_opponent_pool_samples_external_neutral(tmp_path: Path) -> None:
    pool_json = tmp_path / "neutral.json"
    pool_json.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "neutral",
                        "kind": "external_neutral",
                        "weight": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = OpponentPoolSpec.from_json(pool_json)
    prepared = materialize_opponent_pool(spec)
    sampled = prepared.sample_episode_opponent()
    assert sampled.env_mode == "external"
    assert sampled.runtime_kind == "neutral"


def test_materialized_opponent_pool_samples_current_policy_self_play(tmp_path: Path) -> None:
    pool_json = tmp_path / "self_play.json"
    pool_json.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "current_policy",
                        "kind": "self_play",
                        "weight": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prepared = materialize_opponent_pool(OpponentPoolSpec.from_json(pool_json))
    sampled = prepared.sample_episode_opponent()
    assert prepared.has_self_play
    assert sampled.env_mode == "model"
    assert sampled.runtime_kind == "self_play"


def test_opponent_action_provider_neutral_returns_zero_actions() -> None:
    from dfb_reinforcement_learning.opponents.opponent_pool import SampledOpponent

    provider = OpponentActionProvider(device=torch.device("cpu"))
    actions = provider.action_arrays_for_state(
        SampledOpponent(label="neutral", env_mode="external", runtime_kind="neutral"),
        state={},
        role="fighter2",
    )
    assert actions is not None
    cont, binary = actions
    assert np.allclose(cont, 0.0)
    assert np.allclose(binary, 0.0)


def test_opponent_action_provider_loads_contract_checkpoint(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    dataset = load_policy_dataset_contract(dataset_root)
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    checkpoint_path = tmp_path / "opponent.pt"
    torch.save(
        {
            **_checkpoint_metadata(dataset),
            "args": {
                "dataset_root": str(dataset_root),
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    provider = OpponentActionProvider(device=torch.device("cpu"))
    loaded = provider._load_checkpoint_opponent(str(checkpoint_path), None)
    assert loaded.model.shared_stem[0].weight.shape == (64, OBS_DIM)


def test_materialized_pool_rejects_checkpoint_from_another_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    dataset = load_policy_dataset_contract(dataset_root)
    checkpoint_path = tmp_path / "opponent.pt"
    model = StatelessHybridActorCritic(obs_dim=OBS_DIM, hidden_dim=64, num_layers=2)
    metadata = _checkpoint_metadata(dataset)
    metadata["dataset_id"] = "wrong-dataset"
    torch.save(
        {
            **metadata,
            "args": {
                "dataset_root": str(dataset_root),
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "name": "bad",
                        "kind": "fixed_checkpoint",
                        "checkpoint": str(checkpoint_path),
                        "dataset_root": str(dataset_root),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_id"):
        materialize_opponent_pool(OpponentPoolSpec.from_json(pool_path))
