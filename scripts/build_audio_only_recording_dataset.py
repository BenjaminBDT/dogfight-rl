#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EpisodeInfo:
    name: str
    path: str
    scene_name: str
    total_steps: int
    fixed_time_step_seconds: float
    duration_seconds: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an audio-only dual-observer recording dataset from authoritative "
            "recordings, filtering out short episodes and writing a mixed step index."
        )
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("datasets/dfb_game/recordings"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dfb_state_estimation/audio_only_recordings_dual_observer_min20s_v1"),
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Optional explicit path for the episode selection audit JSON.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--audio-window-seconds",
        type=float,
        default=1.0 / 60.0,
    )
    parser.add_argument(
        "--force-derive",
        action="store_true",
        help="Force regeneration of existing audio-only derived artifacts.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume the target tool binary already exists.",
    )
    parser.add_argument(
        "--build-profile",
        choices=["release", "debug"],
        default="release",
        help="Rust build profile used for dfb_tool_dataset. Default uses release for reconstruction speed.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) // 2 or 1)),
        help="Number of episode-level extract jobs to run in parallel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write audit/manifest files and print planned commands.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tool_path(project_root: Path, *, build_profile: str) -> Path:
    profile_dir = "release" if build_profile == "release" else "debug"
    return project_root / "target" / profile_dir / "dfb_tool_dataset"


def _episode_info(episode_root: Path) -> EpisodeInfo:
    episode_text = (episode_root / "episode.ron").read_text(encoding="utf-8", errors="ignore")
    scene_match = re.search(r'scene_name:\s*"([^"]+)"', episode_text)
    steps_match = re.search(r"total_steps:\s*(\d+)", episode_text)
    dt_match = re.search(r"fixed_time_step_seconds:\s*([0-9.]+)", episode_text)
    scene_name = scene_match.group(1) if scene_match else "unknown"
    total_steps = int(steps_match.group(1)) if steps_match else 0
    fixed_time_step_seconds = float(dt_match.group(1)) if dt_match else (1.0 / 60.0)
    duration_seconds = total_steps * fixed_time_step_seconds
    return EpisodeInfo(
        name=episode_root.name,
        path=str(episode_root),
        scene_name=scene_name,
        total_steps=total_steps,
        fixed_time_step_seconds=fixed_time_step_seconds,
        duration_seconds=duration_seconds,
    )


def _scan_episodes(recordings_root: Path) -> list[EpisodeInfo]:
    roots = sorted(
        path
        for path in recordings_root.iterdir()
        if path.is_dir() and (path / "episode.ron").exists()
    )
    return [_episode_info(path) for path in roots]


def _run(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_capture(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )


def _role_output_root(dataset_root: Path, episode_name: str, role: str) -> Path:
    return dataset_root / "episodes" / episode_name / role


def _write_entries_jsonl(dataset_root: Path, eligible: list[EpisodeInfo]) -> tuple[Path, int]:
    entries_path = dataset_root / "entries.jsonl"
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    entry_count = 0
    with entries_path.open("w", encoding="utf-8") as handle:
        for episode in eligible:
            for role in ("fighter1", "fighter2"):
                derived_role_root = _role_output_root(dataset_root, episode.name, role)
                for step_index in range(episode.total_steps):
                    handle.write(
                        json.dumps(
                            {
                                "episode_id": episode.name,
                                "scene_name": episode.scene_name,
                                "source_episode_root": episode.path,
                                "observed_role": role,
                                "step_index": step_index,
                                "total_steps": episode.total_steps,
                                "fixed_time_step_seconds": episode.fixed_time_step_seconds,
                                "duration_seconds": episode.duration_seconds,
                                "derived_role_root": str(derived_role_root),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    entry_count += 1
    return entries_path, entry_count


def _extract_episode(
    *,
    tool_path: Path,
    project_root: Path,
    dataset_root: Path,
    episode: EpisodeInfo,
    audio_window_seconds: float,
    force_derive: bool,
) -> tuple[str, float]:
    episode_root = Path(episode.path)
    role_output_root = dataset_root / "episodes" / episode.name
    cmd = [
        str(tool_path),
        "extract",
        "--episode",
        str(episode_root),
        "--observed-role",
        "all",
        "--no-visual",
        "--output-dir",
        str(role_output_root),
        "--audio-window",
        str(audio_window_seconds),
        *(["--force"] if force_derive else []),
    ]
    started = time.perf_counter()
    result = _run_capture(cmd, cwd=project_root)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"extract failed for {episode.name} with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return episode.name, elapsed


def main() -> None:
    args = _parse_args()
    project_root = _project_root()
    recordings_root = (project_root / args.recordings_root).resolve()
    dataset_root = (project_root / args.output_dir).resolve()
    selection_json = (
        args.selection_json.resolve()
        if args.selection_json is not None
        else dataset_root.with_name(dataset_root.name + "_selection.json")
    )

    episodes = _scan_episodes(recordings_root)
    eligible = [item for item in episodes if item.duration_seconds >= args.min_duration_seconds]
    excluded = [item for item in episodes if item.duration_seconds < args.min_duration_seconds]

    selection_payload = {
        "recordings_root": str(recordings_root),
        "output_dir": str(dataset_root),
        "min_duration_seconds": args.min_duration_seconds,
        "audio_window_seconds": args.audio_window_seconds,
        "eligible_episode_count": len(eligible),
        "excluded_episode_count": len(excluded),
        "eligible_total_steps": sum(item.total_steps for item in eligible),
        "eligible_dual_observer_steps": 2 * sum(item.total_steps for item in eligible),
        "eligible": [asdict(item) for item in eligible],
        "excluded": [asdict(item) for item in excluded],
    }
    selection_json.parent.mkdir(parents=True, exist_ok=True)
    selection_json.write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {selection_json}")

    if not eligible:
        raise SystemExit("no eligible episodes after duration filtering")

    tool_path = _tool_path(project_root, build_profile=args.build_profile)
    if not args.skip_build and not args.dry_run:
        build_cmd = ["cargo", "build", "--bin", "dfb_tool_dataset"]
        if args.build_profile == "release":
            build_cmd.insert(2, "--release")
        _run(build_cmd, cwd=project_root, dry_run=False)
    elif not tool_path.exists():
        raise SystemExit(
            f"missing tool binary at {tool_path}; rerun without --skip-build or build it first"
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    pending: list[EpisodeInfo] = []
    for episode in eligible:
        if not args.force_derive:
            fighter1_done = (_role_output_root(dataset_root, episode.name, "fighter1") / "derived_modalities.ron").exists()
            fighter2_done = (_role_output_root(dataset_root, episode.name, "fighter2") / "derived_modalities.ron").exists()
            if fighter1_done and fighter2_done:
                continue
        pending.append(episode)

    if args.dry_run:
        for episode in pending:
            role_output_root = dataset_root / "episodes" / episode.name
            _run(
                [
                    str(tool_path),
                    "extract",
                    "--episode",
                    str(Path(episode.path)),
                    "--observed-role",
                    "all",
                    "--no-visual",
                    "--output-dir",
                    str(role_output_root),
                    "--audio-window",
                    str(args.audio_window_seconds),
                    *(["--force"] if args.force_derive else []),
                ],
                cwd=project_root,
                dry_run=True,
            )
    elif pending:
        print(
            f"reconstructing {len(pending)} episodes with profile={args.build_profile} jobs={args.jobs}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
            futures = {
                executor.submit(
                    _extract_episode,
                    tool_path=tool_path,
                    project_root=project_root,
                    dataset_root=dataset_root,
                    episode=episode,
                    audio_window_seconds=args.audio_window_seconds,
                    force_derive=args.force_derive,
                ): episode
                for episode in pending
            }
            for future in concurrent.futures.as_completed(futures):
                episode = futures[future]
                name, elapsed = future.result()
                print(f"finished {name} in {elapsed:.1f}s")

    entries_path, entry_count = _write_entries_jsonl(dataset_root, eligible)
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_kind": "audio_only_recordings_index_v1",
                "recordings_root": str(recordings_root),
                "dataset_root": str(dataset_root),
                "entries_jsonl": str(entries_path),
                "min_duration_seconds": args.min_duration_seconds,
                "audio_window_seconds": args.audio_window_seconds,
                "episode_count": len(eligible),
                "entry_count": entry_count,
                "episodes_dir": str(dataset_root / "episodes"),
                "episodes": [
                    {
                        "episode_id": episode.name,
                        "scene_name": episode.scene_name,
                        "source_episode_root": episode.path,
                        "total_steps": episode.total_steps,
                        "duration_seconds": episode.duration_seconds,
                        "fighter1_root": str(_role_output_root(dataset_root, episode.name, "fighter1")),
                        "fighter2_root": str(_role_output_root(dataset_root, episode.name, "fighter2")),
                    }
                    for episode in eligible
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
