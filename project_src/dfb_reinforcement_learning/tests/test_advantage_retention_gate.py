from __future__ import annotations

from pathlib import Path

from dfb_reinforcement_learning.eval.advantage_retention_gate import (
    _checkpoint_candidates,
    _selection_summary,
)


def test_checkpoint_candidates_select_completed_interval_updates(
    tmp_path: Path,
) -> None:
    for name in (
        "update_0009.pt",
        "update_0099.pt",
        "update_0199.pt",
        "update_0509.pt",
        "latest.pt",
    ):
        (tmp_path / name).touch()

    candidates = _checkpoint_candidates(
        tmp_path,
        interval_updates=100,
        max_update=500,
    )

    assert [(update, path.name) for update, path in candidates] == [
        (100, "update_0099.pt"),
        (200, "update_0199.pt"),
    ]


def test_selection_summary_requires_retention_improvement_and_safe_outcome() -> None:
    def experiment(
        policy_id: str,
        scene_id: str,
        *,
        enemy: float,
        mutual: float,
    ) -> dict[str, object]:
        return {
            "policy_id": policy_id,
            "scene_id": scene_id,
            "inference_mode": "deterministic",
            "ego_role": "fighter1",
            "aggregate": {
                "enemy_destroy_rate": enemy,
                "mutual_destroy_rate": mutual,
            },
        }

    experiments = [
        experiment("reference", "head", enemy=0.30, mutual=0.10),
        experiment("reference", "tail", enemy=0.95, mutual=0.00),
        experiment("update_0100", "head", enemy=0.40, mutual=0.05),
        experiment("update_0100", "tail", enemy=0.90, mutual=0.00),
        experiment("update_0200", "head", enemy=0.50, mutual=0.20),
        experiment("update_0200", "tail", enemy=0.95, mutual=0.00),
    ]

    summary = _selection_summary(
        experiments,
        reference_policy_id="reference",
        head_scene_id="head",
        tail_scene_id="tail",
        tail_clean_destroy_min=0.8,
    )

    assert summary["eligible_policy_ids"] == ["update_0100"]
    assert summary["recommended_policy_id"] == "update_0100"
