from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch

from dfb_reinforcement_learning.eval.policy_capability_gate import (
    load_manifest,
    run_capability_gate,
)


_UPDATE_CHECKPOINT_PATTERN = re.compile(r"^update_(\d+)\.pt$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed 100-update checkpoints on frozen fair head-on and "
            "tail-chase capability gates."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=100)
    parser.add_argument("--max-update", type=int, default=500)
    parser.add_argument("--tail-clean-destroy-min", type=float, default=0.8)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-sim-seconds", type=float, default=60.0)
    parser.add_argument("--continuous-std", type=float, default=0.08)
    parser.add_argument("--opponent-mode", default="built_in_ai_imperfect")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--head-on-scene", default=None)
    parser.add_argument("--tail-chase-scene", default=None)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _checkpoint_candidates(
    checkpoint_dir: Path,
    *,
    interval_updates: int,
    max_update: int,
) -> list[tuple[int, Path]]:
    if interval_updates <= 0:
        raise ValueError("--checkpoint-interval-updates must be positive")
    if max_update <= 0:
        raise ValueError("--max-update must be positive")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("update_*.pt"):
        match = _UPDATE_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        zero_based_update = int(match.group(1))
        completed_updates = zero_based_update + 1
        if (
            completed_updates <= max_update
            and completed_updates % interval_updates == 0
        ):
            candidates.append((completed_updates, path.resolve()))
    return sorted(candidates)


def _clean_enemy_destroy_rate(aggregate: dict[str, Any]) -> float:
    return max(
        0.0,
        float(aggregate["enemy_destroy_rate"])
        - float(aggregate["mutual_destroy_rate"]),
    )


def _selection_summary(
    experiments: list[dict[str, Any]],
    *,
    reference_policy_id: str,
    head_scene_id: str,
    tail_scene_id: str,
    tail_clean_destroy_min: float,
) -> dict[str, Any]:
    by_policy_scene: dict[tuple[str, str], dict[str, Any]] = {}
    for experiment in experiments:
        if (
            experiment["inference_mode"] == "deterministic"
            and experiment["ego_role"] == "fighter1"
        ):
            by_policy_scene[
                (str(experiment["policy_id"]), str(experiment["scene_id"]))
            ] = experiment["aggregate"]

    reference_head = by_policy_scene[(reference_policy_id, head_scene_id)]
    reference_head_clean = _clean_enemy_destroy_rate(reference_head)
    reference_head_mutual = float(reference_head["mutual_destroy_rate"])
    policy_ids = sorted(
        {
            policy_id
            for policy_id, scene_id in by_policy_scene
            if scene_id in {head_scene_id, tail_scene_id}
        }
    )
    candidates: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        if (
            (policy_id, head_scene_id) not in by_policy_scene
            or (policy_id, tail_scene_id) not in by_policy_scene
        ):
            continue
        head = by_policy_scene[(policy_id, head_scene_id)]
        tail = by_policy_scene[(policy_id, tail_scene_id)]
        head_clean = _clean_enemy_destroy_rate(head)
        tail_clean = _clean_enemy_destroy_rate(tail)
        head_mutual = float(head["mutual_destroy_rate"])
        eligible = (
            policy_id != reference_policy_id
            and tail_clean >= tail_clean_destroy_min
            and head_clean > reference_head_clean
            and head_mutual <= reference_head_mutual
        )
        candidates.append(
            {
                "policy_id": policy_id,
                "eligible": eligible,
                "head_on_clean_enemy_destroy_rate": head_clean,
                "head_on_mutual_destroy_rate": head_mutual,
                "tail_chase_clean_enemy_destroy_rate": tail_clean,
                "head_on_clean_enemy_delta": head_clean - reference_head_clean,
                "head_on_mutual_delta": head_mutual - reference_head_mutual,
            }
        )
    eligible_candidates = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (
            -item["head_on_clean_enemy_destroy_rate"],
            item["head_on_mutual_destroy_rate"],
            -item["tail_chase_clean_enemy_destroy_rate"],
        ),
    )
    return {
        "reference_policy_id": reference_policy_id,
        "constraints": {
            "tail_chase_clean_enemy_destroy_rate_min": tail_clean_destroy_min,
            "head_on_clean_enemy_destroy_rate_must_exceed_reference": (
                reference_head_clean
            ),
            "head_on_mutual_destroy_rate_must_not_exceed_reference": (
                reference_head_mutual
            ),
        },
        "candidates": candidates,
        "eligible_policy_ids": [
            item["policy_id"] for item in eligible_candidates
        ],
        "recommended_policy_id": (
            None
            if not eligible_candidates
            else eligible_candidates[0]["policy_id"]
        ),
    }


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    candidates = _checkpoint_candidates(
        checkpoint_dir,
        interval_updates=args.checkpoint_interval_updates,
        max_update=args.max_update,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no interval checkpoints found under {checkpoint_dir}"
        )

    frozen_existing_dir = run_dir / "scene_pool" / "existing"
    head_scene = Path(
        args.head_on_scene or frozen_existing_dir / "open_ho.ron"
    ).expanduser().resolve()
    tail_scene = Path(
        args.tail_chase_scene or frozen_existing_dir / "open_tc_f1.ron"
    ).expanduser().resolve()
    for scene in (head_scene, tail_scene):
        if not scene.is_file():
            raise FileNotFoundError(f"frozen capability-gate scene missing: {scene}")

    policies = [
        {
            "id": "reference",
            "type": "checkpoint",
            "modes": ["deterministic"],
            "checkpoint": str(
                Path(args.reference_checkpoint).expanduser().resolve()
            ),
            "continuous_std": args.continuous_std,
        }
    ]
    policies.extend(
        {
            "id": f"update_{completed_updates:04d}",
            "type": "checkpoint",
            "modes": ["deterministic"],
            "checkpoint": str(checkpoint),
            "continuous_std": args.continuous_std,
        }
        for completed_updates, checkpoint in candidates
    )
    manifest_payload = {
        "schema_id": "dfb_part3_policy_capability_gate_v1",
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "scenes": [
            {"id": "fair_head_on", "path": str(head_scene)},
            {"id": "tail_chase_retention", "path": str(tail_scene)},
        ],
        "policies": policies,
        "roles": ["fighter1"],
        "seeds": list(range(args.seed_start, args.seed_start + args.seed_count)),
        "opponent_mode": args.opponent_mode,
        "num_envs": args.num_envs,
        "ticks_per_step": 1,
        "max_sim_seconds": args.max_sim_seconds,
        "max_steps": 0,
        "shot_window_threshold": 0.5,
        "boundary_threat_seconds": 2.0,
        "parity_stride_steps": 60,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gate_summary = run_capability_gate(
        manifest=load_manifest(manifest_path),
        output_dir=output_dir / "capability_gate",
        device=_resolve_device(args.device),
        project_root=args.project_root,
    )
    selection = _selection_summary(
        gate_summary["experiments"],
        reference_policy_id="reference",
        head_scene_id="fair_head_on",
        tail_scene_id="tail_chase_retention",
        tail_clean_destroy_min=args.tail_clean_destroy_min,
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[AdvantageRetentionGate] "
        f"checkpoints={len(candidates)} "
        f"eligible={len(selection['eligible_policy_ids'])} "
        f"recommended={selection['recommended_policy_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
