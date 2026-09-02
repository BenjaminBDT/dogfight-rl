from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dfb_reinforcement_learning.policy_assets import (
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    load_and_validate_policy_dataset,
    validate_policy_checkpoint_payload,
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


def _write_asset_set(root: Path) -> None:
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


def _checkpoint_payload(dataset) -> dict[str, object]:
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
        "model_state_dict": {},
    }


def test_asset_bundle_and_checkpoint_share_exact_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_asset_set(root)
    dataset, _ = load_and_validate_policy_dataset(root)
    checkpoint = _checkpoint_payload(dataset)
    validate_policy_checkpoint_payload(checkpoint, dataset=dataset)


@pytest.mark.parametrize(
    ("file_name", "field"),
    [
        ("meta.json", "policy_contract_id"),
        ("schema.json", "action_schema_id"),
        ("obs_normalizer.json", "contract_sha256"),
    ],
)
def test_asset_bundle_rejects_missing_contract_fields(
    tmp_path: Path,
    file_name: str,
    field: str,
) -> None:
    root = tmp_path / "dataset"
    _write_asset_set(root)
    path = root / file_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(field)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"missing required field {field}"):
        load_and_validate_policy_dataset(root)


def test_same_dimension_different_contract_hash_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_asset_set(root)
    path = root / "obs_normalizer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract_sha256"):
        load_and_validate_policy_dataset(root)


def test_checkpoint_from_another_dataset_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_asset_set(root)
    dataset, _ = load_and_validate_policy_dataset(root)
    checkpoint = _checkpoint_payload(dataset)
    mismatched = copy.deepcopy(checkpoint)
    mismatched["dataset_id"] = "different-dataset"
    with pytest.raises(ValueError, match="dataset_id"):
        validate_policy_checkpoint_payload(mismatched, dataset=dataset)


@pytest.mark.parametrize(
    "field",
    [
        "action_cont_dim",
        "action_bin_dim",
        "model_hyperparameters",
        "training_stage",
        "update_index",
        "global_step",
    ],
)
def test_checkpoint_rejects_missing_frozen_metadata(tmp_path: Path, field: str) -> None:
    root = tmp_path / "dataset"
    _write_asset_set(root)
    dataset, _ = load_and_validate_policy_dataset(root)
    checkpoint = _checkpoint_payload(dataset)
    checkpoint.pop(field)
    with pytest.raises(ValueError, match=field):
        validate_policy_checkpoint_payload(checkpoint, dataset=dataset)
