from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dfb_reinforcement_learning.data import (
    BcDataset,
    BcSplitData,
    filter_bc_split_by_roles,
    load_bc_split,
    with_episode_balanced_weights,
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


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_load_bc_split_and_dataset(tmp_path: Path) -> None:
    obs_dim = OBS_DIM
    dataset_root = tmp_path / "dataset"
    identity = {
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
    }
    meta = {
        **identity,
        "dataset_id": "dataset-fixture",
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "statistics": {"obs_dim": obs_dim},
        "constants": {
            "demonstration_sources": {"fighter1": "human", "fighter2": "non_human"},
            "bc_sample_weights": {"fighter1": 2.0, "fighter2": 1.0},
        },
        "chunks": [
            {
                "chunk_id": "chunk_000000",
                "episode_id": "ep1",
                "observed_role": "fighter1",
                "demonstration_source": "human",
                "sample_weight": 2.0,
                "split": "train",
                "step_count": 2,
                "group_files": {
                    "policy_input": "policy_input/chunk_000000.npz",
                    "action_targets": "action_targets/chunk_000000.npz",
                    "aux": "aux/chunk_000000.npz",
                },
            }
        ],
    }
    schema = identity
    normalizer = {
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "obs_dim": obs_dim,
        "epsilon": 1e-6,
        "mean": [0.0] * obs_dim,
        "std": [1.0] * obs_dim,
        "train_row_count": 2,
        "source_dataset_id": "dataset-fixture",
    }
    (dataset_root / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (dataset_root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (dataset_root / "obs_normalizer.json").write_text(json.dumps(normalizer), encoding="utf-8")
    _write_npz(
        dataset_root / "policy_input/chunk_000000.npz",
        simulation_step_index=np.asarray([0, 1], dtype=np.int32),
        timestamp=np.asarray([0.0, 0.1], dtype=np.float64),
        obs=np.zeros((2, obs_dim), dtype=np.float32),
    )
    _write_npz(
        dataset_root / "action_targets/chunk_000000.npz",
        action_cont=np.ones((2, 4), dtype=np.float32),
        action_bin=np.zeros((2, 3), dtype=np.float32),
    )
    _write_npz(dataset_root / "aux/chunk_000000.npz", done=np.zeros((2,), dtype=np.uint8))
    split_data, obs_normalizer = load_bc_split(dataset_root, "train")
    assert split_data.obs.shape == (2, obs_dim)
    assert split_data.action_cont.shape == (2, 4)
    assert split_data.action_bin.shape == (2, 3)
    assert split_data.demonstration_sources == ("human", "human")
    np.testing.assert_array_equal(split_data.sample_weights, np.asarray([2.0, 2.0], dtype=np.float32))
    dataset = BcDataset(split_data, normalizer=obs_normalizer, normalize_obs=True)
    sample = dataset[0]
    assert tuple(sample["obs"].shape) == (obs_dim,)
    assert tuple(sample["action_cont"].shape) == (4,)
    assert tuple(sample["action_bin"].shape) == (3,)
    assert float(sample["sample_weight"]) == 2.0


def test_load_bc_split_rejects_missing_policy_contract_id(tmp_path: Path) -> None:
    obs_dim = OBS_DIM
    dataset_root = tmp_path / "dataset"
    meta = {
        "dataset_id": "dataset-fixture",
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "statistics": {"obs_dim": obs_dim},
        "chunks": [],
    }
    normalizer = {
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "obs_dim": obs_dim,
        "epsilon": 1e-6,
        "mean": [0.0] * obs_dim,
        "std": [1.0] * obs_dim,
        "train_row_count": 0,
        "source_dataset_id": "dataset-fixture",
    }
    dataset_root.mkdir(parents=True)
    (dataset_root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (dataset_root / "schema.json").write_text(
        json.dumps(
            {
                "dataset_schema_id": DATASET_SCHEMA_ID,
                "policy_contract_id": POLICY_CONTRACT_ID,
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "action_schema_id": ACTION_SCHEMA_ID,
                "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
                "contract_sha256": POLICY_CONTRACT_SHA256,
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "obs_normalizer.json").write_text(json.dumps(normalizer), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field policy_contract_id"):
        load_bc_split(dataset_root, "train")


def test_episode_balanced_weights_preserve_source_ratio_and_equalize_episode_roles() -> None:
    split_data = BcSplitData(
        obs=np.zeros((6, OBS_DIM), dtype=np.float32),
        action_cont=np.zeros((6, 4), dtype=np.float32),
        action_bin=np.zeros((6, 3), dtype=np.float32),
        episode_ids=("long", "long", "short", "long", "long", "short"),
        observed_roles=("fighter1", "fighter1", "fighter1", "fighter2", "fighter2", "fighter2"),
        demonstration_sources=("human", "human", "human", "non_human", "non_human", "non_human"),
        sample_weights=np.asarray([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32),
        step_indices=np.asarray([0, 1, 0, 0, 1, 0], dtype=np.int32),
    )

    balanced = with_episode_balanced_weights(split_data)

    np.testing.assert_allclose(
        balanced.sample_weights,
        np.asarray([1.5, 1.5, 3.0, 0.75, 0.75, 1.5], dtype=np.float32),
    )
    assert float(balanced.sample_weights[:2].sum()) == pytest.approx(
        float(balanced.sample_weights[2])
    )
    assert float(balanced.sample_weights[3:5].sum()) == pytest.approx(
        float(balanced.sample_weights[5])
    )
    assert float(balanced.sample_weights[:3].sum()) == pytest.approx(
        2.0 * float(balanced.sample_weights[3:].sum())
    )


def test_filter_bc_split_by_roles_keeps_only_selected_action_labels() -> None:
    split = BcSplitData(
        obs=np.arange(12, dtype=np.float32).reshape(3, 4),
        action_cont=np.arange(12, dtype=np.float32).reshape(3, 4),
        action_bin=np.arange(9, dtype=np.float32).reshape(3, 3),
        episode_ids=("a", "a", "b"),
        observed_roles=("fighter1", "fighter2", "fighter1"),
        demonstration_sources=("teacher", "imperfect", "teacher"),
        sample_weights=np.asarray([1.0, 0.5, 1.0], dtype=np.float32),
        step_indices=np.asarray([1, 2, 3], dtype=np.int32),
    )

    filtered = filter_bc_split_by_roles(split, ("fighter1",))

    assert filtered.size == 2
    assert filtered.observed_roles == ("fighter1", "fighter1")
    assert filtered.demonstration_sources == ("teacher", "teacher")
    np.testing.assert_array_equal(filtered.step_indices, np.asarray([1, 3], dtype=np.int32))
