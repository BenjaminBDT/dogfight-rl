from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfb_reinforcement_learning.data import load_bc_split
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_contract import (
    ACTION_SCHEMA_ID,
    DATASET_SCHEMA_ID,
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    OBS_DIM,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)
from dfb_reinforcement_learning.tools.merge_policy_datasets import merge_policy_datasets


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset(
    root: Path,
    *,
    dataset_id: str,
    episode_id: str,
    source: str,
    observed_role: str,
    obs_value: float,
    dagger: bool,
) -> None:
    identity = {
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
    }
    for directory in ("policy_input", "action_targets", "auxiliary"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    obs = np.full((2, OBS_DIM), obs_value, dtype=np.float32)
    np.savez(
        root / "policy_input/chunk_000000.npz",
        simulation_step_index=np.asarray([0, 1], dtype=np.int32),
        timestamp=np.asarray([0.0, 1.0 / 60.0], dtype=np.float64),
        obs=obs,
    )
    np.savez(
        root / "action_targets/chunk_000000.npz",
        action_cont=np.zeros((2, 4), dtype=np.float32),
        action_bin=np.zeros((2, 3), dtype=np.float32),
    )
    np.savez(
        root / "auxiliary/chunk_000000.npz",
        done=np.zeros((2,), dtype=np.uint8),
    )
    meta: dict[str, object] = {
        **identity,
        "dataset_id": dataset_id,
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "source_episode_manifest_sha256": "0" * 64,
        "storage_layout": {
            "format": "chunked_npz",
            "groups": ["policy_input", "action_targets", "aux"],
        },
        "constants": {
            "demonstration_sources": {
                "fighter1": source if observed_role == "fighter1" else "other",
                "fighter2": source if observed_role == "fighter2" else "other",
            },
            "bc_sample_weights": {"fighter1": 1.0, "fighter2": 1.0},
        },
        "statistics": {
            "obs_dim": OBS_DIM,
            "total_model_steps": 2,
            "total_simulation_steps": 2,
        },
        "chunks": [
            {
                "chunk_id": "chunk_000000",
                "chunk_index": 0,
                "episode_id": episode_id,
                "observed_role": observed_role,
                "demonstration_source": source,
                "sample_weight": 1.0,
                "split": "train",
                "step_count": 2,
                "simulation_step_index_start": 0,
                "simulation_step_index_end_exclusive": 2,
                "group_files": {
                    "policy_input": "policy_input/chunk_000000.npz",
                    "action_targets": "action_targets/chunk_000000.npz",
                    "aux": "auxiliary/chunk_000000.npz",
                },
            }
        ],
        "episodes": [
            {
                "episode_id": episode_id,
                "scene_name": "fixture",
                "split": "train",
                "total_steps": 2,
            }
        ],
        "splits": {"train": [episode_id], "val": [], "test": []},
    }
    if dagger:
        meta["dagger"] = {"episodes": [{"episode_id": episode_id}]}
    _write_json(root / "meta.json", meta)
    _write_json(root / "schema.json", identity)
    _write_json(
        root / "obs_normalizer.json",
        {
            "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
            "policy_contract_id": POLICY_CONTRACT_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "contract_sha256": POLICY_CONTRACT_SHA256,
            "obs_dim": OBS_DIM,
            "epsilon": 1e-6,
            "mean": [0.0] * OBS_DIM,
            "std": [1.0] * OBS_DIM,
            "train_row_count": 2,
            "source_dataset_id": dataset_id,
        },
    )


def test_merge_policy_datasets_keeps_sources_and_recomputes_normalizer(
    tmp_path: Path,
) -> None:
    human = tmp_path / "human"
    dagger = tmp_path / "dagger"
    output = tmp_path / "merged"
    _make_dataset(
        human,
        dataset_id="human-dataset",
        episode_id="human-episode",
        source="human",
        observed_role="fighter1",
        obs_value=1.0,
        dagger=False,
    )
    _make_dataset(
        dagger,
        dataset_id="dagger-dataset",
        episode_id="dagger-episode",
        source="rule_teacher",
        observed_role="fighter1",
        obs_value=3.0,
        dagger=True,
    )

    summary = merge_policy_datasets(
        input_datasets=[human, dagger],
        output_dir=output,
        included_roles=("fighter1",),
        source_weights={"human": 2.0, "dagger_teacher": 1.5},
        dagger_source_name="dagger_teacher",
        force=False,
    )

    assert summary["total_model_steps"] == 4
    assert summary["source_step_counts"] == {"dagger_teacher": 2, "human": 2}
    split, normalizer = load_bc_split(output, "train")
    assert split.demonstration_sources == (
        "human",
        "human",
        "dagger_teacher",
        "dagger_teacher",
    )
    np.testing.assert_array_equal(
        split.sample_weights,
        np.asarray([2.0, 2.0, 1.5, 1.5], dtype=np.float32),
    )
    expected_mean = np.full((OBS_DIM,), 2.0, dtype=np.float32)
    expected_std = np.full((OBS_DIM,), 1.0, dtype=np.float32)
    expected_mean[list(POLICY_OBSERVATION_SCHEMA.binary_indices)] = 0.0
    np.testing.assert_allclose(normalizer.mean, expected_mean)
    np.testing.assert_allclose(normalizer.std, expected_std)

    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["lineage"]["initialization_parent_dataset_ids"] == [
        "human-dataset",
        "dagger-dataset",
    ]
    assert meta["constants"]["bc_demonstration_conventions"] == {
        "fighter1": {"human": 2.0, "dagger_teacher": 1.5}
    }
