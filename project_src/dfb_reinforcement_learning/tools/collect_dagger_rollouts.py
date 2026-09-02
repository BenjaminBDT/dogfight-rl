from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.envs import PolicyDogfightEnv, PolicyDogfightEnvConfig
from dfb_reinforcement_learning.obs.policy_schema import POLICY_OBSERVATION_SCHEMA
from dfb_reinforcement_learning.policy_assets import load_and_validate_policy_dataset
from dfb_reinforcement_learning.policy_inference import (
    deterministic_policy_output,
    load_policy_model,
)
from dfb_reinforcement_learning.scenes import ScenePoolSpec, materialize_scene_pool
from dfb_reinforcement_learning.tools.collect_teacher_recordings import (
    build_collection_jobs,
    summarize_collection_jobs,
)

FIELD_SLICES = POLICY_OBSERVATION_SCHEMA.field_slices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect student-state DAgger rollouts with read-only teacher labels."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--scene-pool-json", required=True)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ego-role", choices=("fighter1", "fighter2"), default="fighter1")
    parser.add_argument("--opponent-mode", default="built_in_ai_imperfect")
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument(
        "--sampling-strategy",
        choices=("weighted_with_replacement", "exhaustive"),
        default="weighted_with_replacement",
    )
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--ticks-per-step", type=int, default=1)
    parser.add_argument("--teacher-execution-probability", type=float, default=0.0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _prepare_empty_directory(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _scene_reset_kwargs(scene: Any) -> dict[str, str]:
    if scene.scene_path:
        return {"scene_path": scene.scene_path}
    if scene.scene_name:
        return {"scene_name": scene.scene_name}
    raise ValueError(f"materialized scene {scene.label!r} has no scene path or name")


def _event_count(events: list[dict[str, Any]], *, kind: str, subject_prefix: str) -> int:
    return sum(
        1
        for event in events
        if event.get("kind") == kind
        and isinstance(event.get("subject"), str)
        and (
            event["subject"] == subject_prefix
            or event["subject"].startswith(f"{subject_prefix}:")
        )
    )


def _auxiliary_row(
    *,
    obs: np.ndarray,
    info: dict[str, Any],
    executed_action_bin: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    ego_role = str(info["ego_role"])
    enemy_role = str(info["enemy_role"])
    ego = info["aircraft_by_role"][ego_role]
    enemy = info["aircraft_by_role"][enemy_role]
    events = [event for event in info.get("step_events", []) if isinstance(event, dict)]
    winner = info.get("winner")
    winner_label = 1 if winner == ego_role else -1 if winner == enemy_role else 0
    target_distance = info.get("target_distance")
    return {
        "done": int(terminated or truncated),
        "did_hit": int(_event_count(events, kind="Hit", subject_prefix=enemy_role) > 0),
        "got_hit": int(_event_count(events, kind="Hit", subject_prefix=ego_role) > 0),
        "did_fire": int(float(executed_action_bin[1]) >= 0.5),
        "self_out_of_bounds_seconds": float(ego["out_of_bounds_seconds"]),
        "self_ceiling_recovery_seconds": float(ego.get("ceiling_recovery_seconds", 0.0)),
        "self_repair_elapsed_seconds": float(ego["repair_elapsed_seconds"]),
        "self_destroyed": int(bool(ego["destroyed"])),
        "enemy_destroyed": int(bool(enemy["destroyed"])),
        "self_health_state_norm": np.asarray(
            obs[FIELD_SLICES["self_health_state_norm"]],
            dtype=np.float32,
        ),
        "enemy_health_state_norm": np.asarray(
            obs[FIELD_SLICES["enemy_health_state_norm"]],
            dtype=np.float32,
        ),
        "self_gun_overheated": int(
            float(obs[FIELD_SLICES["self_gun_overheated"]][0]) >= 0.5
        ),
        "self_gun_heat_norm": float(obs[FIELD_SLICES["self_gun_heat_norm"]][0]),
        "winner_label": winner_label,
        "target_distance": 0.0 if target_distance is None else float(target_distance),
    }


def _stack_auxiliary(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    scalar_dtypes = {
        "done": np.uint8,
        "did_hit": np.uint8,
        "got_hit": np.uint8,
        "did_fire": np.uint8,
        "self_out_of_bounds_seconds": np.float32,
        "self_ceiling_recovery_seconds": np.float32,
        "self_repair_elapsed_seconds": np.float32,
        "self_destroyed": np.uint8,
        "enemy_destroyed": np.uint8,
        "self_gun_overheated": np.uint8,
        "self_gun_heat_norm": np.float32,
        "winner_label": np.int32,
        "target_distance": np.float32,
    }
    result = {
        key: np.asarray([row[key] for row in rows], dtype=dtype)
        for key, dtype in scalar_dtypes.items()
    }
    for key in ("self_health_state_norm", "enemy_health_state_norm"):
        result[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
    return result


def main() -> None:
    args = _parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("max-seconds must be positive")
    if args.ticks_per_step < 1:
        raise ValueError("ticks-per-step must be positive")
    if not 0.0 <= args.teacher_execution_probability <= 1.0:
        raise ValueError("teacher-execution-probability must be in [0, 1]")
    if not 0.0 < args.binary_threshold < 1.0:
        raise ValueError("binary-threshold must be in (0, 1)")

    project_root = Path(args.project_root).resolve()
    workspace_dir = Path(args.workspace_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _prepare_empty_directory(workspace_dir, force=args.force)
    _prepare_empty_directory(output_dir, force=args.force)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir()

    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(args.dataset_root)
    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    device = _resolve_device(args.device)
    model = load_policy_model(
        args.student_checkpoint,
        normalizer=normalizer,
        dataset_contract=dataset_contract,
        device=device,
        context="DAgger student checkpoint",
    )

    scene_spec = ScenePoolSpec.from_json(Path(args.scene_pool_json))
    prepared = materialize_scene_pool(
        spec=scene_spec,
        output_dir=workspace_dir / "scene_pool",
    )
    jobs = build_collection_jobs(
        prepared,
        episode_count=args.episode_count,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
    )
    if not jobs:
        raise ValueError("DAgger collection scene pool is empty")

    rng = np.random.default_rng(args.seed)
    manifest: dict[str, Any] = {
        "schema": "dfb.part3.dagger_collection.v1",
        "dataset_id": dataset_contract.dataset_id,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "scene_pool_json": str(Path(args.scene_pool_json).resolve()),
        "project_root": str(project_root),
        "workspace_dir": str(workspace_dir),
        "ego_role": args.ego_role,
        "opponent_mode": args.opponent_mode,
        "episode_count": args.episode_count,
        "sampling": {
            "strategy": args.sampling_strategy,
            "train_tactical_ratio": prepared.train_tactical_ratio,
            "train_recovery_ratio": prepared.train_recovery_ratio,
            **summarize_collection_jobs(jobs),
        },
        "max_seconds": args.max_seconds,
        "ticks_per_step": args.ticks_per_step,
        "teacher_execution_probability": args.teacher_execution_probability,
        "binary_threshold": args.binary_threshold,
        "seed": args.seed,
        "episodes": [],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    first = jobs[0]
    with PolicyDogfightEnv(
        PolicyDogfightEnvConfig(
            project_root=str(project_root),
            scene_name=first.scene.scene_name or "open",
            scene_path=first.scene.scene_path,
            ego_role=args.ego_role,
            opponent_mode=args.opponent_mode,
            seed=first.environment_seed,
            ticks_per_step=args.ticks_per_step,
        )
    ) as env:
        for job_index, job in enumerate(jobs, start=1):
            obs, info = env.reset(
                seed=job.environment_seed,
                opponent_mode=args.opponent_mode,
                **_scene_reset_kwargs(job.scene),
            )
            episode_start = float(info["sim_time_seconds"])
            obs_rows: list[np.ndarray] = []
            teacher_cont_rows: list[np.ndarray] = []
            teacher_bin_rows: list[np.ndarray] = []
            student_cont_rows: list[np.ndarray] = []
            student_bin_rows: list[np.ndarray] = []
            student_bin_prob_rows: list[np.ndarray] = []
            executed_teacher_rows: list[int] = []
            step_indices: list[int] = []
            timestamps: list[float] = []
            auxiliary_rows: list[dict[str, Any]] = []
            terminated = False
            truncated = False
            step_index = 0

            while float(info["sim_time_seconds"]) - episode_start < args.max_seconds:
                teacher_cont, teacher_bin = env.teacher_action_arrays()
                student_output = deterministic_policy_output(
                    model=model,
                    normalizer=normalizer,
                    obs=obs,
                    device=device,
                )
                student_cont = student_output.action_cont
                student_bin = student_output.binary_actions(
                    threshold=args.binary_threshold
                )
                execute_teacher = (
                    float(rng.random()) < args.teacher_execution_probability
                )
                executed_cont = teacher_cont if execute_teacher else student_cont
                executed_bin = teacher_bin if execute_teacher else student_bin

                obs_rows.append(obs.copy())
                teacher_cont_rows.append(teacher_cont.copy())
                teacher_bin_rows.append(teacher_bin.copy())
                student_cont_rows.append(student_cont.copy())
                student_bin_rows.append(student_bin.copy())
                student_bin_prob_rows.append(student_output.action_bin_prob.copy())
                executed_teacher_rows.append(int(execute_teacher))
                step_indices.append(step_index)
                timestamps.append(float(info["sim_time_seconds"]))

                next_obs, _, terminated, truncated, next_info = env.step_arrays(
                    executed_cont,
                    executed_bin,
                    binary_threshold=args.binary_threshold,
                )
                auxiliary_rows.append(
                    _auxiliary_row(
                        obs=next_obs,
                        info=next_info,
                        executed_action_bin=executed_bin,
                        terminated=terminated,
                        truncated=truncated,
                    )
                )
                obs = next_obs
                info = next_info
                step_index += 1
                if terminated or truncated:
                    break

            if not obs_rows:
                raise RuntimeError(f"DAgger episode {job_index} produced no rows")
            episode_id = (
                f"dagger_{job.scene.label}_{job.environment_seed:08d}_{job.repetition:02d}"
            )
            episode_path = episodes_dir / f"{episode_id}.npz"
            student_cont_array = np.asarray(student_cont_rows, dtype=np.float32)
            teacher_cont_array = np.asarray(teacher_cont_rows, dtype=np.float32)
            student_bin_array = np.asarray(student_bin_rows, dtype=np.float32)
            teacher_bin_array = np.asarray(teacher_bin_rows, dtype=np.float32)
            np.savez_compressed(
                episode_path,
                obs=np.asarray(obs_rows, dtype=np.float32),
                teacher_action_cont=teacher_cont_array,
                teacher_action_bin=teacher_bin_array,
                student_action_cont=student_cont_array,
                student_action_bin=student_bin_array,
                student_action_bin_prob=np.asarray(
                    student_bin_prob_rows,
                    dtype=np.float32,
                ),
                executed_teacher=np.asarray(executed_teacher_rows, dtype=np.uint8),
                simulation_step_index=np.asarray(step_indices, dtype=np.int32),
                timestamp=np.asarray(timestamps, dtype=np.float64),
                **_stack_auxiliary(auxiliary_rows),
            )
            cont_error = np.abs(student_cont_array - teacher_cont_array)
            bin_mismatch = np.not_equal(student_bin_array, teacher_bin_array)
            episode_summary = {
                "episode_index": job_index - 1,
                "episode_id": episode_id,
                "episode_file": str(episode_path.relative_to(output_dir)),
                "scene": asdict(job.scene),
                "scene_category": job.category,
                "repetition": job.repetition,
                "environment_seed": job.environment_seed,
                "step_count": len(obs_rows),
                "terminated": terminated,
                "truncated": truncated,
                "winner": info.get("winner"),
                "final_sim_time_seconds": float(info["sim_time_seconds"]),
                "mean_absolute_continuous_error": cont_error.mean(axis=0).tolist(),
                "binary_mismatch_rate": bin_mismatch.mean(axis=0).tolist(),
                "teacher_execution_rate": float(np.mean(executed_teacher_rows)),
            }
            manifest["episodes"].append(episode_summary)
            _write_json(manifest_path, manifest)
            print(
                f"[{job_index}/{len(jobs)}] {job.scene.label} "
                f"seed={job.environment_seed} steps={len(obs_rows)} "
                f"cont_mae={float(cont_error.mean()):.4f} "
                f"bin_mismatch={float(bin_mismatch.mean()):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
