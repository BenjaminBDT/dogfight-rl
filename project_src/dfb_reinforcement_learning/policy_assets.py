from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

from dfb_reinforcement_learning.policy_contract import (
    ACTION_BIN_DIM,
    ACTION_CONT_DIM,
    ACTION_SCHEMA_ID,
    CHECKPOINT_SCHEMA_ID,
    DATASET_SCHEMA_ID,
    MODEL_FAMILY_ID,
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    OBS_DIM,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)


@dataclass(frozen=True)
class PolicyDatasetContract:
    root: Path
    dataset_id: str
    dataset_schema_id: str
    policy_contract_id: str
    observation_schema_id: str
    action_schema_id: str
    normalizer_schema_id: str
    contract_sha256: str
    obs_dim: int
    normalizer_path: Path
    initialization_parent_dataset_ids: tuple[str, ...] = ()


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{context}: unable to read {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: invalid JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: root must be an object")
    return payload


def _required_string(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in payload:
        raise ValueError(f"{context}: missing required field {key}")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    if key not in payload:
        raise ValueError(f"{context}: missing required field {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {key} must be an integer")
    return value


def _expect(value: Any, expected: Any, *, field: str, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context}: {field}={value!r}, expected {expected!r}")


def _validate_component_identity(payload: Mapping[str, Any], *, context: str) -> None:
    expected = {
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
    }
    for key, expected_value in expected.items():
        _expect(
            _required_string(payload, key, context=context),
            expected_value,
            field=key,
            context=context,
        )


def load_policy_dataset_contract(dataset_root: str | Path) -> PolicyDatasetContract:
    root = Path(dataset_root).resolve()
    meta = _read_json_object(root / "meta.json", context="dataset meta")
    schema_path_value = _required_string(meta, "schema_path", context="dataset meta")
    normalizer_path_value = _required_string(meta, "obs_normalizer_path", context="dataset meta")
    schema = _read_json_object(root / schema_path_value, context="dataset schema")

    _validate_component_identity(meta, context="dataset meta")
    _validate_component_identity(schema, context="dataset schema")
    _expect(
        _required_string(meta, "dataset_schema_id", context="dataset meta"),
        DATASET_SCHEMA_ID,
        field="dataset_schema_id",
        context="dataset meta",
    )
    _expect(
        _required_string(schema, "dataset_schema_id", context="dataset schema"),
        DATASET_SCHEMA_ID,
        field="dataset_schema_id",
        context="dataset schema",
    )
    dataset_id = _required_string(meta, "dataset_id", context="dataset meta")

    for key in (
        "dataset_schema_id",
        "policy_contract_id",
        "observation_schema_id",
        "action_schema_id",
        "normalizer_schema_id",
        "contract_sha256",
    ):
        _expect(schema[key], meta[key], field=key, context="dataset meta/schema")

    statistics = meta.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("dataset meta: statistics must be an object")
    obs_dim = _required_int(statistics, "obs_dim", context="dataset meta.statistics")
    _expect(obs_dim, OBS_DIM, field="obs_dim", context="dataset meta.statistics")

    normalizer_path = (root / normalizer_path_value).resolve()
    if root not in normalizer_path.parents:
        raise ValueError("dataset meta: obs_normalizer_path escapes dataset root")
    if not normalizer_path.is_file():
        raise ValueError(f"dataset meta: normalizer does not exist at {normalizer_path}")
    lineage = meta.get("lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("dataset meta: lineage must be an object")
    parent_values = lineage.get("initialization_parent_dataset_ids", [])
    if not isinstance(parent_values, list):
        raise ValueError(
            "dataset meta: lineage.initialization_parent_dataset_ids must be an array"
        )
    parent_dataset_ids: list[str] = []
    for value in parent_values:
        if not isinstance(value, str) or not value:
            raise ValueError(
                "dataset meta: initialization parent dataset IDs must be non-empty strings"
            )
        if value == dataset_id or value in parent_dataset_ids:
            raise ValueError(
                "dataset meta: initialization parent dataset IDs must be unique and differ from dataset_id"
            )
        parent_dataset_ids.append(value)
    return PolicyDatasetContract(
        root=root,
        dataset_id=dataset_id,
        dataset_schema_id=DATASET_SCHEMA_ID,
        policy_contract_id=POLICY_CONTRACT_ID,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        action_schema_id=ACTION_SCHEMA_ID,
        normalizer_schema_id=NORMALIZER_SCHEMA_ID,
        contract_sha256=POLICY_CONTRACT_SHA256,
        obs_dim=obs_dim,
        normalizer_path=normalizer_path,
        initialization_parent_dataset_ids=tuple(parent_dataset_ids),
    )


def validate_policy_normalizer_payload(
    payload: Mapping[str, Any],
    *,
    dataset: PolicyDatasetContract | None = None,
    context: str = "observation normalizer",
) -> None:
    expected = {
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
    }
    for key, expected_value in expected.items():
        _expect(
            _required_string(payload, key, context=context),
            expected_value,
            field=key,
            context=context,
        )
    obs_dim = _required_int(payload, "obs_dim", context=context)
    _expect(obs_dim, OBS_DIM, field="obs_dim", context=context)
    source_dataset_id = _required_string(payload, "source_dataset_id", context=context)
    if dataset is not None:
        _expect(source_dataset_id, dataset.dataset_id, field="source_dataset_id", context=context)


def checkpoint_contract_metadata(dataset: PolicyDatasetContract) -> dict[str, Any]:
    return {
        "checkpoint_schema_id": CHECKPOINT_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "model_family_id": MODEL_FAMILY_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "obs_dim": dataset.obs_dim,
        "action_cont_dim": ACTION_CONT_DIM,
        "action_bin_dim": ACTION_BIN_DIM,
        "dataset_id": dataset.dataset_id,
    }


def checkpoint_model_hyperparameters(
    *,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    continuous_action_std: float | None,
    popart_beta: float | None,
    popart_min_std: float | None,
    shared_extension_blocks: int = 0,
    actor_extension_blocks: int = 0,
    critic_extension_blocks: int = 0,
) -> dict[str, Any]:
    return {
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "shared_extension_blocks": int(shared_extension_blocks),
        "actor_extension_blocks": int(actor_extension_blocks),
        "critic_extension_blocks": int(critic_extension_blocks),
        "continuous_action_std": (
            None if continuous_action_std is None else float(continuous_action_std)
        ),
        "popart_beta": None if popart_beta is None else float(popart_beta),
        "popart_min_std": None if popart_min_std is None else float(popart_min_std),
    }


def validate_policy_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    dataset: PolicyDatasetContract,
    context: str = "policy checkpoint",
) -> None:
    expected = checkpoint_contract_metadata(dataset)
    for key, expected_value in expected.items():
        if isinstance(expected_value, int):
            value = _required_int(payload, key, context=context)
        else:
            value = _required_string(payload, key, context=context)
        _expect(value, expected_value, field=key, context=context)
    hyperparameters = payload.get("model_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{context}: model_hyperparameters must be an object")
    for key in ("hidden_dim", "num_layers"):
        value = _required_int(hyperparameters, key, context=f"{context}.model_hyperparameters")
        if value <= 0:
            raise ValueError(f"{context}: model_hyperparameters.{key} must be positive")
    for key in (
        "shared_extension_blocks",
        "actor_extension_blocks",
        "critic_extension_blocks",
    ):
        value = hyperparameters.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{context}: model_hyperparameters.{key} must be a non-negative integer"
            )
    dropout = hyperparameters.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError(f"{context}: model_hyperparameters.dropout must be numeric")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"{context}: model_hyperparameters.dropout must be in [0, 1)")
    for key in ("continuous_action_std", "popart_beta", "popart_min_std"):
        if key not in hyperparameters:
            raise ValueError(f"{context}: missing required field model_hyperparameters.{key}")
        value = hyperparameters[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{context}: model_hyperparameters.{key} must be numeric or null")
    _required_string(payload, "training_stage", context=context)
    for key in ("update_index", "global_step"):
        value = _required_int(payload, key, context=context)
        if value < 0:
            raise ValueError(f"{context}: {key} must be non-negative")
    if "model_state_dict" not in payload:
        raise ValueError(f"{context}: missing required field model_state_dict")


def validate_policy_checkpoint_initialization_payload(
    payload: Mapping[str, Any],
    *,
    dataset: PolicyDatasetContract,
    context: str = "policy initialization checkpoint",
) -> None:
    checkpoint_dataset_id = _required_string(payload, "dataset_id", context=context)
    allowed_dataset_ids = {
        dataset.dataset_id,
        *dataset.initialization_parent_dataset_ids,
    }
    if checkpoint_dataset_id not in allowed_dataset_ids:
        expected = ", ".join(sorted(allowed_dataset_ids))
        raise ValueError(
            f"{context}: dataset_id={checkpoint_dataset_id!r} is not an allowed "
            f"initialization source; expected one of {expected}"
        )
    validate_policy_checkpoint_payload(
        payload,
        dataset=replace(dataset, dataset_id=checkpoint_dataset_id),
        context=context,
    )


def load_and_validate_policy_dataset(
    dataset_root: str | Path,
) -> tuple[PolicyDatasetContract, dict[str, Any]]:
    dataset = load_policy_dataset_contract(dataset_root)
    normalizer_payload = _read_json_object(dataset.normalizer_path, context="observation normalizer")
    validate_policy_normalizer_payload(normalizer_payload, dataset=dataset)
    return dataset, normalizer_payload
