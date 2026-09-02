from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfb_reinforcement_learning.data import load_bc_split
from dfb_reinforcement_learning.policy_contract import (
    ACTION_SCHEMA_ID,
    DATASET_SCHEMA_ID,
    NORMALIZER_SCHEMA_ID,
    OBSERVATION_SCHEMA_ID,
    OBS_DIM,
    POLICY_CONTRACT_ID,
    POLICY_CONTRACT_SHA256,
)
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.tools.build_dagger_dataset import (
    AUXILIARY_KEYS,
    build_dagger_dataset,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_base_dataset(root: Path) -> None:
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
    np.savez(
        root / "policy_input/chunk_000000.npz",
        simulation_step_index=np.asarray([0, 1], dtype=np.int32),
        timestamp=np.asarray([0.0, 1.0 / 60.0], dtype=np.float64),
        obs=np.zeros((2, OBS_DIM), dtype=np.float32),
    )
    np.savez(
        root / "action_targets/chunk_000000.npz",
        action_cont=np.zeros((2, 4), dtype=np.float32),
        action_bin=np.zeros((2, 3), dtype=np.float32),
    )
    np.savez(root / "auxiliary/chunk_000000.npz", done=np.zeros((2,), dtype=np.uint8))
    meta = {
        **identity,
        "dataset_id": "dataset-parent",
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "source_episode_manifest_sha256": "0" * 64,
        "constants": {
            "demonstration_sources": {
                "fighter1": "rule_teacher",
                "fighter2": "rule_opponent_imperfect",
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
                "episode_id": "base-episode",
                "observed_role": "fighter1",
                "demonstration_source": "rule_teacher",
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
                "episode_id": "base-episode",
                "split": "train",
            }
        ],
        "splits": {"train": ["base-episode"], "val": [], "test": []},
    }
    mean = [0.25] * OBS_DIM
    std = [2.0] * OBS_DIM
    for index in POLICY_OBSERVATION_SCHEMA.binary_indices:
        mean[index] = 0.0
        std[index] = 1.0
    normalizer = {
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "contract_sha256": POLICY_CONTRACT_SHA256,
        "obs_dim": OBS_DIM,
        "epsilon": 1e-6,
        "mean": mean,
        "std": std,
        "train_row_count": 2,
        "source_dataset_id": "dataset-parent",
    }
    _write_json(root / "meta.json", meta)
    _write_json(root / "schema.json", identity)
    _write_json(root / "obs_normalizer.json", normalizer)


def _make_collection(root: Path) -> None:
    episode_dir = root / "episodes"
    episode_dir.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {
        "obs": np.ones((3, OBS_DIM), dtype=np.float32),
        "teacher_action_cont": np.full((3, 4), 0.5, dtype=np.float32),
        "teacher_action_bin": np.zeros((3, 3), dtype=np.float32),
        "simulation_step_index": np.arange(3, dtype=np.int32),
        "timestamp": np.arange(3, dtype=np.float64) / 60.0,
    }
    for key in AUXILIARY_KEYS:
        if key in {"self_health_state_norm", "enemy_health_state_norm"}:
            arrays[key] = np.ones((3, 6), dtype=np.float32)
        elif key == "winner_label":
            arrays[key] = np.zeros((3,), dtype=np.int32)
        elif key in {
            "self_out_of_bounds_seconds",
            "self_ceiling_recovery_seconds",
            "self_repair_elapsed_seconds",
            "self_gun_heat_norm",
            "target_distance",
        }:
            arrays[key] = np.zeros((3,), dtype=np.float32)
        else:
            arrays[key] = np.zeros((3,), dtype=np.uint8)
    np.savez(episode_dir / "dagger-episode.npz", **arrays)
    _write_json(
        root / "manifest.json",
        {
            "schema": "dfb.part3.dagger_collection.v1",
            "dataset_id": "dataset-parent",
            "ego_role": "fighter1",
            "student_checkpoint": "/tmp/student.pt",
            "teacher_execution_probability": 0.0,
            "episodes": [
                {
                    "episode_id": "dagger-episode",
                    "episode_file": "episodes/dagger-episode.npz",
                    "scene": {"label": "scene-a"},
                    "step_count": 3,
                    "terminated": False,
                    "truncated": False,
                    "winner": None,
                    "mean_absolute_continuous_error": [0.1] * 4,
                    "binary_mismatch_rate": [0.0] * 3,
                }
            ],
        },
    )


def test_build_dagger_dataset_preserves_normalizer_and_appends_training_rows(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    collection = tmp_path / "collection"
    output = tmp_path / "output"
    _make_base_dataset(base)
    _make_collection(collection)

    summary = build_dagger_dataset(
        base_dataset_root=base,
        collection_roots=[collection],
        output_dir=output,
        dagger_source_name="dagger_teacher",
        dagger_sample_weight=1.5,
        force=False,
    )

    assert summary["added_step_count"] == 3
    assert summary["dagger_sample_weight"] == 1.5
    split, normalizer = load_bc_split(output, "train")
    assert split.size == 5
    np.testing.assert_array_equal(split.action_cont[-3:], np.full((3, 4), 0.5))
    np.testing.assert_array_equal(
        split.sample_weights,
        np.asarray([1.0, 1.0, 1.5, 1.5, 1.5], dtype=np.float32),
    )
    expected_mean = np.full((OBS_DIM,), 0.25, dtype=np.float32)
    expected_mean[list(POLICY_OBSERVATION_SCHEMA.binary_indices)] = 0.0
    np.testing.assert_array_equal(normalizer.mean, expected_mean)
    meta = json.loads((output / "meta.json").read_text(encoding="utf-8"))
    assert meta["lineage"]["initialization_parent_dataset_ids"] == ["dataset-parent"]
    assert meta["splits"]["train"][-1] == "dagger-episode"
    assert meta["constants"]["bc_demonstration_conventions"] == {
        "fighter1": {
            "rule_teacher": 1.0,
            "dagger_teacher": 1.5,
        },
        "fighter2": {"rule_opponent_imperfect": 1.0},
    }
    normalizer_payload = json.loads(
        (output / "obs_normalizer.json").read_text(encoding="utf-8")
    )
    assert normalizer_payload["source_dataset_id"] == summary["dataset_id"]
