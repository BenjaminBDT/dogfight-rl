from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Any

from dfb_reinforcement_learning.scenes import (
    MaterializedScene,
    PreparedScenePool,
    ScenePoolSpec,
    materialize_scene_pool,
)


@dataclass(frozen=True)
class CollectionJob:
    scene: MaterializedScene
    category: str
    repetition: int
    environment_seed: int


def build_collection_jobs(
    prepared: PreparedScenePool,
    *,
    episode_count: int,
    seed: int,
    sampling_strategy: str = "weighted_with_replacement",
) -> list[CollectionJob]:
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    if sampling_strategy not in {"weighted_with_replacement", "exhaustive"}:
        raise ValueError(f"unsupported collection sampling strategy: {sampling_strategy!r}")

    tactical_scenes = prepared.train_tactical_scenes
    recovery_scenes = prepared.train_recovery_scenes
    if not tactical_scenes and not recovery_scenes:
        raise ValueError("collection scene pool is empty")

    rng = random.Random(seed)
    if sampling_strategy == "exhaustive":
        selected = [
            *((scene, "tactical") for scene in tactical_scenes),
            *((scene, "recovery") for scene in recovery_scenes),
        ]
        if episode_count != len(selected):
            raise ValueError(
                "exhaustive collection requires episode_count to equal the "
                f"materialized scene count ({len(selected)})"
            )
        rng.shuffle(selected)
    elif tactical_scenes and recovery_scenes:
        ratio_total = prepared.train_tactical_ratio + prepared.train_recovery_ratio
        tactical_probability = (
            0.5
            if ratio_total <= 1e-12
            else prepared.train_tactical_ratio / ratio_total
        )
        tactical_count = min(
            episode_count,
            max(0, int(episode_count * tactical_probability + 0.5)),
        )
        recovery_count = episode_count - tactical_count

        def sample(
            scenes: list[MaterializedScene],
            *,
            count: int,
            category: str,
        ) -> list[tuple[MaterializedScene, str]]:
            if count == 0:
                return []
            weights = [max(scene.weight, 0.0) for scene in scenes]
            if sum(weights) <= 1e-12:
                weights = [1.0] * len(scenes)
            return [
                (scene, category)
                for scene in rng.choices(scenes, weights=weights, k=count)
            ]

        selected = sample(
            tactical_scenes,
            count=tactical_count,
            category="tactical",
        )
        selected.extend(
            sample(
                recovery_scenes,
                count=recovery_count,
                category="recovery",
            )
        )
        rng.shuffle(selected)
    else:
        scenes = tactical_scenes or recovery_scenes
        category = "tactical" if tactical_scenes else "recovery"
        weights = [max(scene.weight, 0.0) for scene in scenes]
        if sum(weights) <= 1e-12:
            weights = [1.0] * len(scenes)
        selected = [
            (scene, category)
            for scene in rng.choices(scenes, weights=weights, k=episode_count)
        ]

    repetitions: Counter[str] = Counter()
    jobs: list[CollectionJob] = []
    for index, (scene, category) in enumerate(selected):
        repetition = repetitions[scene.label]
        repetitions[scene.label] += 1
        jobs.append(
            CollectionJob(
                scene=scene,
                category=category,
                repetition=repetition,
                environment_seed=seed + index,
            )
        )
    return jobs


def summarize_collection_jobs(jobs: list[CollectionJob]) -> dict[str, Any]:
    return {
        "episode_count": len(jobs),
        "category_counts": dict(sorted(Counter(job.category for job in jobs).items())),
        "scene_counts": dict(sorted(Counter(job.scene.label for job in jobs).items())),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record deterministic teacher-vs-imperfect episodes for BC distillation."
    )
    parser.add_argument("--scene-pool-json", required=True)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--recordings-output-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--episode-count", type=int, default=60)
    parser.add_argument(
        "--sampling-strategy",
        choices=("weighted_with_replacement", "exhaustive"),
        default="weighted_with_replacement",
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _prepare_empty_directory(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _scene_reset_kwargs(scene: MaterializedScene) -> dict[str, str]:
    if scene.scene_path:
        return {"scene_path": scene.scene_path}
    if scene.scene_name:
        return {"scene_name": scene.scene_name}
    raise ValueError(f"materialized scene {scene.label!r} has no scene path or name")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _advance_until_running(environment: Any, noop: Any) -> dict[str, Any]:
    for _ in range(4):
        status = json.loads(environment.episode_status_json())
        if (
            status["match_phase"] == "Running"
            and not status["terminated"]
            and not status["truncated"]
        ):
            return status
        environment.step_json(noop)
    raise RuntimeError("environment did not enter Running after reset")


def main() -> None:
    args = _parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("max_seconds must be positive")

    project_root = Path(args.project_root).resolve()
    workspace_dir = Path(args.workspace_dir).resolve()
    recordings_output_dir = Path(args.recordings_output_dir).resolve()
    _prepare_empty_directory(workspace_dir, force=args.force)
    _prepare_empty_directory(recordings_output_dir, force=args.force)

    spec = ScenePoolSpec.from_json(Path(args.scene_pool_json))
    prepared = materialize_scene_pool(spec=spec, output_dir=workspace_dir / "scene_pool")
    jobs = build_collection_jobs(
        prepared,
        episode_count=args.episode_count,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
    )
    if not jobs:
        raise ValueError("teacher collection scene pool is empty")

    from dfb_game_py import Environment, EnvironmentAction

    noop = EnvironmentAction()
    manifest_path = recordings_output_dir / "teacher_collection_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "dfb.teacher_collection.v1",
        "project_root": str(project_root),
        "scene_pool_json": str(Path(args.scene_pool_json).resolve()),
        "workspace_dir": str(workspace_dir),
        "fighter1_profile": "built_in_ai_teacher",
        "fighter2_profile": "built_in_ai_imperfect",
        "episode_count": args.episode_count,
        "sampling": {
            "strategy": args.sampling_strategy,
            "train_tactical_ratio": prepared.train_tactical_ratio,
            "train_recovery_ratio": prepared.train_recovery_ratio,
            **summarize_collection_jobs(jobs),
        },
        "max_seconds": args.max_seconds,
        "seed": args.seed,
        "episodes": [],
    }
    _write_manifest(manifest_path, manifest)

    first = jobs[0]
    environment = Environment(
        project_root=str(project_root),
        seed=first.environment_seed,
        enable_visual=False,
        enable_audio=False,
        ticks_per_step=1,
        fighter1_mode="built_in_ai_teacher",
        fighter2_mode="built_in_ai_imperfect",
        **_scene_reset_kwargs(first.scene),
    )
    try:
        for index, job in enumerate(jobs, start=1):
            if index > 1:
                environment.reset_json(
                    seed=job.environment_seed,
                    enable_visual=False,
                    enable_audio=False,
                    ticks_per_step=1,
                    fighter1_mode="built_in_ai_teacher",
                    fighter2_mode="built_in_ai_imperfect",
                    **_scene_reset_kwargs(job.scene),
                )
            _advance_until_running(environment, noop)
            environment.start_recording(enable_visual=False, enable_audio=False)

            status = json.loads(environment.episode_status_json())
            while (
                not status["terminated"]
                and not status["truncated"]
                and float(status["sim_time_seconds"]) < args.max_seconds
            ):
                environment.step_json(noop)
                status = json.loads(environment.episode_status_json())

            recording_status = json.loads(environment.recording_status_json())
            if recording_status["active"] or recording_status["pending_start"]:
                environment.stop_recording()
                environment.step_json(noop)
                recording_status = json.loads(environment.recording_status_json())

            saved_path_value = recording_status.get("last_saved_path")
            if not saved_path_value:
                raise RuntimeError(f"recording {index} did not produce a saved episode")
            saved_manifest_path = Path(saved_path_value)
            if not saved_manifest_path.is_absolute():
                saved_manifest_path = project_root / saved_manifest_path
            saved_episode_root = saved_manifest_path.parent
            destination = recordings_output_dir / saved_episode_root.name
            if destination.exists():
                raise FileExistsError(f"recording destination already exists: {destination}")
            shutil.move(str(saved_episode_root), destination)

            manifest["episodes"].append(
                {
                    "episode_index": index - 1,
                    "episode_id": destination.name,
                    "recording_path": str(destination),
                    "scene": asdict(job.scene),
                    "scene_category": job.category,
                    "repetition": job.repetition,
                    "environment_seed": job.environment_seed,
                    "terminal_status": status,
                    "recorded_frames": recording_status["last_saved_frame_count"],
                }
            )
            _write_manifest(manifest_path, manifest)
            print(
                f"[{index}/{len(jobs)}] {job.scene.label} seed={job.environment_seed} "
                f"frames={recording_status['last_saved_frame_count']} "
                f"terminated={status['terminated']} truncated={status['truncated']}",
                flush=True,
            )
    finally:
        environment.shutdown()


if __name__ == "__main__":
    main()
