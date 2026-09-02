from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dfb_reinforcement_learning.eval.policy_capability_gate import (
    GATE_SCHEMA_ID,
    EpisodeAccumulator,
    PolicySpec,
    _aggregate_episodes,
    _build_bc_gate,
    _count_hits,
    _finalize_episode,
    _update_accumulator,
    load_manifest,
    run_capability_gate,
)
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA


def _field_index(name: str) -> int:
    for field in POLICY_OBSERVATION_SCHEMA.fields:
        if field.name == name:
            return field.value_slice.start
    raise KeyError(name)


def _info(sim_time_seconds: float) -> dict[str, object]:
    return {
        "sim_time_seconds": sim_time_seconds,
        "ego_role": "fighter1",
        "enemy_role": "fighter2",
        "events_since_last_step": [
            {"kind": "Hit", "subject": "fighter2", "other_subject": "fighter1"}
        ],
        "aircraft_by_role": {
            "fighter1": {
                "destroyed": False,
                "out_of_bounds_seconds": 0.0,
                "speed": 70.0,
                "stall_factor": 0.2,
                "time_to_ground_impact_s": None,
                "time_to_ceiling_impact_s": None,
                "time_to_horizontal_boundary_impact_s": 1.0,
            },
            "fighter2": {
                "destroyed": False,
                "out_of_bounds_seconds": 0.0,
                "speed": 65.0,
                "stall_factor": 0.1,
            },
        },
    }


def _episode(
    *,
    enemy_destroyed: bool,
    self_destroyed: bool,
    shot_window_episode: bool,
    boundary_threat_fraction: float,
) -> dict[str, object]:
    return {
        "duration_seconds": 10.0,
        "termination_reason": "enemy_destroyed" if enemy_destroyed else "self_destroyed",
        "enemy_destroyed": enemy_destroyed,
        "self_destroyed": self_destroyed,
        "first_shot_window_seconds": 2.0 if shot_window_episode else None,
        "shot_window_fraction": 0.2 if shot_window_episode else 0.0,
        "enemy_shot_window_fraction": 0.1,
        "effective_fire_fraction": 0.5,
        "shot_window_utilization": 0.4,
        "tracking_quality_mean": 0.5,
        "tail_hold_score_mean": 0.4,
        "enemy_tail_hold_score_mean": 0.2,
        "tail_advantage_mean": 0.2,
        "speed_mean_mps": 70.0,
        "stall_factor_mean": 0.2,
        "boundary_threat_fraction": boundary_threat_fraction,
        "out_of_bounds_fraction": 0.0,
        "self_hit_count": int(self_destroyed),
        "enemy_hit_count": int(enemy_destroyed),
        "requested_continuous_action": None,
        "binary_probability_mean": None,
        "actual_binary_on_fraction": {
            "brake": 0.0,
            "fire_gun": 0.1,
            "repair": 0.0,
        },
    }


def test_manifest_loads_relative_assets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scene = tmp_path / "scene.ron"
    scene.write_text("()", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "gate.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": GATE_SCHEMA_ID,
                "dataset_root": "dataset",
                "scenes": [{"id": "scene_a", "path": "scene.ron"}],
                "policies": [
                    {
                        "id": "checkpoint_a",
                        "type": "checkpoint",
                        "checkpoint": "checkpoint.pt",
                        "modes": ["deterministic", "sampled"],
                    }
                ],
                "roles": ["fighter1", "fighter2"],
                "seeds": [10, 11],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    assert manifest.dataset_root == dataset.resolve()
    assert manifest.scenes[0].path == scene.resolve()
    assert manifest.policies[0].checkpoint == checkpoint.resolve()
    assert manifest.max_sim_seconds == 120.0


def test_manifest_rejects_duplicate_seeds(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scene = tmp_path / "scene.ron"
    scene.write_text("()", encoding="utf-8")
    manifest_path = tmp_path / "gate.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": GATE_SCHEMA_ID,
                "dataset_root": str(dataset),
                "scenes": [{"id": "scene_a", "path": str(scene)}],
                "policies": [{"id": "random", "type": "random", "modes": ["sampled"]}],
                "roles": ["fighter1"],
                "seeds": [10, 10],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seeds must be unique"):
        load_manifest(manifest_path)


def test_accumulator_tracks_tactical_and_action_metrics() -> None:
    obs = np.zeros((POLICY_OBSERVATION_SCHEMA.dim,), dtype=np.float32)
    obs[_field_index("self_shot_feasibility")] = 0.8
    obs[_field_index("enemy_shot_feasibility")] = 0.1
    obs[_field_index("self_tracking_quality")] = 0.6
    obs[_field_index("self_tail_hold_score")] = 0.7
    obs[_field_index("enemy_tail_hold_score")] = 0.2
    obs[_field_index("self_fire_gun_active")] = 1.0
    obs[_field_index("self_repair_active")] = 0.0
    accumulator = EpisodeAccumulator(
        seed=5,
        start_sim_time_seconds=0.0,
        previous_sim_time_seconds=0.0,
    )
    info = _info(1.0)
    _update_accumulator(
        accumulator,
        obs=obs,
        info=info,
        requested_cont=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        binary_probabilities=np.asarray([0.2, 0.8, 0.1], dtype=np.float32),
        action_source_observable=True,
        shot_window_threshold=0.5,
        boundary_threat_seconds=2.0,
    )
    result = _finalize_episode(accumulator, info)
    assert result["first_shot_window_seconds"] == 1.0
    assert result["shot_window_fraction"] == 1.0
    assert result["effective_fire_fraction"] == 1.0
    assert result["tail_advantage_mean"] == pytest.approx(0.5)
    assert result["boundary_threat_fraction"] == 1.0
    assert result["enemy_hit_count"] == 1
    assert result["requested_continuous_action"] is not None


def test_hit_count_uses_subject_as_damaged_role() -> None:
    events = [
        {"kind": "Hit", "subject": "fighter1", "other_subject": "fighter2"},
        {"kind": "SubsystemHit", "subject": "fighter2", "other_subject": "fighter1"},
        {"kind": "GunFired", "subject": "fighter1"},
    ]
    assert _count_hits(events, ego_role="fighter1", enemy_role="fighter2") == (1, 1)


def test_aggregate_reports_duration_quantiles_and_unavailable_probabilities() -> None:
    aggregate = _aggregate_episodes(
        [
            _episode(
                enemy_destroyed=True,
                self_destroyed=False,
                shot_window_episode=True,
                boundary_threat_fraction=0.1,
            ),
            _episode(
                enemy_destroyed=False,
                self_destroyed=True,
                shot_window_episode=False,
                boundary_threat_fraction=0.3,
            ),
        ]
    )
    assert aggregate["enemy_destroy_rate"] == 0.5
    assert aggregate["duration_seconds"]["p50"] == 10.0
    assert aggregate["shot_window_episode_rate"] == 0.5
    assert aggregate["binary_probability_mean"] is None


def test_bc_gate_requires_all_fixed_scene_role_comparisons() -> None:
    random_aggregate = _aggregate_episodes(
        [
            _episode(
                enemy_destroyed=False,
                self_destroyed=True,
                shot_window_episode=False,
                boundary_threat_fraction=0.4,
            )
        ]
    )
    bc_aggregate = _aggregate_episodes(
        [
            _episode(
                enemy_destroyed=True,
                self_destroyed=False,
                shot_window_episode=True,
                boundary_threat_fraction=0.1,
            )
        ]
    )
    experiments = [
        {
            "policy_id": "random",
            "inference_mode": "sampled",
            "scene_id": "scene_a",
            "ego_role": "fighter1",
            "aggregate": random_aggregate,
        },
        {
            "policy_id": "bc",
            "inference_mode": "deterministic",
            "scene_id": "scene_a",
            "ego_role": "fighter1",
            "aggregate": bc_aggregate,
        },
    ]
    policies = (
        PolicySpec(
            policy_id="random",
            kind="random",
            modes=("sampled",),
            comparison_role="random_reference",
        ),
        PolicySpec(
            policy_id="bc",
            kind="checkpoint",
            modes=("deterministic",),
            comparison_role="bc_reference",
        ),
    )
    gate = _build_bc_gate(experiments, policies)
    assert gate["available"]
    assert gate["passed"]


def test_gate_refuses_non_empty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        run_capability_gate(
            manifest=None,  # type: ignore[arg-type]
            output_dir=output_dir,
            device=None,  # type: ignore[arg-type]
            project_root=None,
        )
