#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import threading
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


class RuntimeLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **payload: object) -> None:
        record = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _parse_visual_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", value)
    if match is None:
        raise ValueError(f"invalid visual resolution {value!r}; expected WIDTHxHEIGHT")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"visual resolution must be positive, got {value!r}")
    return width, height


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full authoritative multimodal recording pack by filtering episodes, "
            "running extract/label for both observers, then packing into the standard dataset schema."
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
        default=Path("runs/dfb_state_estimation/authoritative_multimodal_recording_pack_v1"),
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Optional explicit path for the episode selection audit JSON.",
    )
    parser.add_argument(
        "--exclude-episode",
        action="append",
        default=[],
        help="Episode name to exclude from this build. Repeatable.",
    )
    parser.add_argument(
        "--runtime-log-jsonl",
        type=Path,
        default=None,
        help="Optional explicit path for runtime jsonl logs.",
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
        "--visual-resolution",
        type=str,
        default="400x300",
    )
    parser.add_argument(
        "--force-derive",
        action="store_true",
        help="Force regeneration of existing derived visual/audio artifacts.",
    )
    parser.add_argument(
        "--force-label",
        action="store_true",
        help="Force regeneration of existing derived labels.",
    )
    parser.add_argument(
        "--force-pack",
        action="store_true",
        help="Overwrite the output packed dataset if it already exists.",
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
        "--label-jobs",
        type=int,
        default=1,
        help="Number of episode-level label jobs to run in parallel. Default is 1 for stability.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Pass --profile to extract/label/pack for timing output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write audit files and print planned commands.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval for long-running extract/label stages.",
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


def _run_stage(
    *,
    stage_name: str,
    episodes: list[EpisodeInfo],
    max_workers: int,
    submit_fn,
    logger: RuntimeLogger,
    heartbeat_seconds: float,
) -> None:
    if not episodes:
        return
    stage_started = time.perf_counter()
    logger.log(
        "stage_start",
        stage=stage_name,
        episode_count=len(episodes),
        max_workers=max_workers,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {}
        for episode in episodes:
            logger.log("task_submitted", stage=stage_name, episode=episode.name)
            future = submit_fn(executor, episode)
            future_to_episode[future] = episode.name
        pending = set(future_to_episode.keys())
        completed_count = 0
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=max(1.0, heartbeat_seconds),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.perf_counter() - stage_started
                active = [future_to_episode[future] for future in list(pending)[:8]]
                print(
                    f"[heartbeat] stage={stage_name} elapsed={elapsed:.1f}s "
                    f"completed={completed_count}/{len(episodes)} pending={len(pending)} "
                    f"active={active}"
                )
                logger.log(
                    "stage_heartbeat",
                    stage=stage_name,
                    elapsed_seconds=elapsed,
                    completed=completed_count,
                    total=len(episodes),
                    pending=len(pending),
                    active_episodes=active,
                )
                continue
            for future in done:
                episode_name = future_to_episode[future]
                episode_result, elapsed = future.result()
                completed_count += 1
                print(
                    f"{stage_name} ok: {episode_result} ({elapsed:.2f}s) "
                    f"[{completed_count}/{len(episodes)}]"
                )
                logger.log(
                    "task_completed",
                    stage=stage_name,
                    episode=episode_name,
                    elapsed_seconds=elapsed,
                    completed=completed_count,
                    total=len(episodes),
                )
    logger.log(
        "stage_completed",
        stage=stage_name,
        elapsed_seconds=time.perf_counter() - stage_started,
        episode_count=len(episodes),
    )


def _derived_complete(episode_root: Path) -> bool:
    return all(
        (episode_root / "derived" / role / "derived_modalities.ron").exists()
        for role in ("fighter1", "fighter2")
    )


def _derived_manifest_matches_resolution(
    manifest_path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> bool:
    if not manifest_path.exists():
        return False
    text = manifest_path.read_text(encoding="utf-8", errors="ignore")
    width_match = re.search(r"\bkind:\s*Front,\s*width:\s*(\d+)", text, re.S)
    height_match = re.search(r"\bkind:\s*Front,\s*width:\s*\d+,\s*height:\s*(\d+)", text, re.S)
    if width_match is None or height_match is None:
        return False
    return (
        int(width_match.group(1)) == expected_width
        and int(height_match.group(1)) == expected_height
    )


def _derived_current(
    episode_root: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> bool:
    if not _derived_complete(episode_root):
        return False
    return all(
        _derived_manifest_matches_resolution(
            episode_root / "derived" / role / "derived_modalities.ron",
            expected_width=expected_width,
            expected_height=expected_height,
        )
        for role in ("fighter1", "fighter2")
    )


def _derived_requires_force_reextract(
    episode_root: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> bool:
    if _derived_incomplete_nonempty(episode_root):
        return True
    if not _derived_complete(episode_root):
        return False
    return not _derived_current(
        episode_root,
        expected_width=expected_width,
        expected_height=expected_height,
    )


def _derived_incomplete_nonempty(episode_root: Path) -> bool:
    for role in ("fighter1", "fighter2"):
        role_root = episode_root / "derived" / role
        if not role_root.exists():
            continue
        if (role_root / "derived_modalities.ron").exists():
            continue
        try:
            next(role_root.iterdir())
        except StopIteration:
            continue
        return True
    return False


def _labels_complete(episode_root: Path) -> bool:
    return all(
        (episode_root / "derived" / role / "derived_labels.ron").exists()
        for role in ("fighter1", "fighter2")
    )


def _label_manifest_is_current(role_root: Path) -> bool:
    manifest_path = role_root / "derived_labels.ron"
    if not manifest_path.exists():
        return False
    text = manifest_path.read_text(encoding="utf-8", errors="ignore")
    version_match = re.search(r"\bschema_version\s*:\s*(\d+)", text)
    if version_match is None or int(version_match.group(1)) < 2:
        return False
    required_fields = (
        "keypoint_projectable_front",
        "keypoint_projectable_rear",
    )
    return all(field in text for field in required_fields)


def _labels_current(episode_root: Path) -> bool:
    return all(
        _label_manifest_is_current(episode_root / "derived" / role)
        for role in ("fighter1", "fighter2")
    )


def _labels_incomplete_nonempty(episode_root: Path) -> bool:
    for role in ("fighter1", "fighter2"):
        role_root = episode_root / "derived" / role
        if not role_root.exists():
            continue
        label_path = role_root / "derived_labels.ron"
        if label_path.exists():
            sibling_artifacts_exist = any(
                path.exists()
                for path in (
                    role_root / "derived_modalities.ron",
                    role_root / "visual",
                    role_root / "segmentation",
                    role_root / "vision_voting",
                    role_root / "audio",
                )
            )
            if sibling_artifacts_exist:
                opposite_role = "fighter2" if role == "fighter1" else "fighter1"
                if not (episode_root / "derived" / opposite_role / "derived_labels.ron").exists():
                    return True
            continue
        try:
            next(role_root.iterdir())
        except StopIteration:
            continue
        return True
    return False


def _roles_missing_labels(episode_root: Path) -> list[str]:
    missing: list[str] = []
    for role in ("fighter1", "fighter2"):
        role_root = episode_root / "derived" / role
        if not _label_manifest_is_current(role_root):
            missing.append(role)
    return missing


def _clear_partial_label_outputs(episode_root: Path, role: str, *, logger: RuntimeLogger | None = None) -> None:
    role_root = episode_root / "derived" / role
    if not role_root.exists():
        return
    removed: list[str] = []
    label_manifest = role_root / "derived_labels.ron"
    if label_manifest.exists():
        label_manifest.unlink()
        removed.append(str(label_manifest))
    voting_root = role_root / "vision_voting"
    if voting_root.exists():
        shutil.rmtree(voting_root)
        removed.append(str(voting_root))
    if removed:
        print(f"cleared partial label outputs: role={role} removed={removed}")
        if logger is not None:
            logger.log(
                "label_partial_cleanup",
                episode=episode_root.name,
                observed_role=role,
                removed=removed,
            )


def _extract_episode(
    *,
    tool_path: Path,
    project_root: Path,
    episode: EpisodeInfo,
    audio_window_seconds: float,
    visual_resolution: str,
    expected_width: int,
    expected_height: int,
    force_derive: bool,
    profile: bool,
) -> tuple[str, float]:
    episode_root = Path(episode.path)
    force_this_episode = force_derive or _derived_requires_force_reextract(
        episode_root,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    cmd = [
        str(tool_path),
        "extract",
        "--episode",
        str(episode_root),
        "--observed-role",
        "all",
        "--visual-resolution",
        visual_resolution,
        "--audio-window",
        str(audio_window_seconds),
        *(["--force"] if force_this_episode else []),
        *(["--profile"] if profile else []),
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


def _label_episode(
    *,
    tool_path: Path,
    project_root: Path,
    episode: EpisodeInfo,
    force_label: bool,
    profile: bool,
    logger: RuntimeLogger | None = None,
) -> tuple[str, float]:
    episode_root = Path(episode.path)
    missing_roles = _roles_missing_labels(episode_root)
    if force_label or not missing_roles:
        roles_to_label = ["fighter1", "fighter2"]
    else:
        roles_to_label = missing_roles
    started = time.perf_counter()
    result = None
    for role in roles_to_label:
        if role in missing_roles:
            _clear_partial_label_outputs(episode_root, role, logger=logger)
        cmd = [
            str(tool_path),
            "label",
            "--episode",
            str(episode_root),
            "--observed-role",
            role,
            *(["--force"] if force_label else []),
            *(["--profile"] if profile else []),
        ]
        result = _run_capture(cmd, cwd=project_root)
        if result.returncode == -11:
            retry_cmd = [
                str(tool_path),
                "label",
                "--episode",
                str(episode_root),
                "--observed-role",
                role,
                "--force",
                *(["--profile"] if profile else []),
            ]
            result = _run_capture(retry_cmd, cwd=project_root)
        if result.returncode != 0:
            break
    elapsed = time.perf_counter() - started
    assert result is not None
    if result.returncode != 0:
        raise RuntimeError(
            f"label failed for {episode.name} with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return episode.name, elapsed


def main() -> None:
    args = _parse_args()
    project_root = _project_root()
    recordings_root = (project_root / args.recordings_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    expected_width, expected_height = _parse_visual_resolution(args.visual_resolution)
    selection_json = (
        args.selection_json.resolve()
        if args.selection_json is not None
        else output_dir.with_name(output_dir.name + "_selection.json")
    )
    runtime_log_jsonl = (
        args.runtime_log_jsonl.resolve()
        if args.runtime_log_jsonl is not None
        else output_dir.with_name(output_dir.name + "_runtime_log.jsonl")
    )
    logger = RuntimeLogger(runtime_log_jsonl)

    episodes = _scan_episodes(recordings_root)
    excluded_names = {name.strip() for name in args.exclude_episode if name.strip()}
    eligible = [
        item
        for item in episodes
        if item.duration_seconds >= args.min_duration_seconds and item.name not in excluded_names
    ]
    excluded = [
        item
        for item in episodes
        if item.duration_seconds < args.min_duration_seconds or item.name in excluded_names
    ]
    eligible_roots = [Path(item.path) for item in eligible]

    selection_payload = {
        "recordings_root": str(recordings_root),
        "output_dir": str(output_dir),
        "min_duration_seconds": args.min_duration_seconds,
        "excluded_episode_names": sorted(excluded_names),
        "audio_window_seconds": args.audio_window_seconds,
        "visual_resolution": args.visual_resolution,
        "label_jobs": args.label_jobs,
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
    print(f"runtime log: {runtime_log_jsonl}")
    logger.log(
        "run_start",
        recordings_root=str(recordings_root),
        output_dir=str(output_dir),
        selection_json=str(selection_json),
        runtime_log_jsonl=str(runtime_log_jsonl),
        min_duration_seconds=args.min_duration_seconds,
        audio_window_seconds=args.audio_window_seconds,
        visual_resolution=args.visual_resolution,
        jobs=args.jobs,
        label_jobs=args.label_jobs,
    )

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

    extract_candidates = (
        eligible
        if args.force_derive
        else [
            item
            for item in eligible
            if not _derived_current(
                Path(item.path),
                expected_width=expected_width,
                expected_height=expected_height,
            )
        ]
    )
    extract_candidate_names = {item.name for item in extract_candidates}
    label_candidates = (
        eligible
        if args.force_label
        else [
            item
            for item in eligible
            if item.name in extract_candidate_names or not _labels_current(Path(item.path))
        ]
    )

    print(
        f"eligible episodes: {len(eligible)} | "
        f"extract pending: {len(extract_candidates)} | "
        f"label pending: {len(label_candidates)}"
    )
    logger.log(
        "selection_summary",
        eligible_episodes=len(eligible),
        extract_pending=len(extract_candidates),
        label_pending=len(label_candidates),
    )

    if args.dry_run:
        for episode in extract_candidates:
            print(f"[dry-run] would extract {episode.name}")
        for episode in label_candidates:
            print(f"[dry-run] would label {episode.name}")
    else:
        if extract_candidates:
            _run_stage(
                stage_name="extract",
                episodes=extract_candidates,
                max_workers=args.jobs,
                submit_fn=lambda executor, episode: executor.submit(
                        _extract_episode,
                        tool_path=tool_path,
                        project_root=project_root,
                        episode=episode,
                        audio_window_seconds=args.audio_window_seconds,
                        visual_resolution=args.visual_resolution,
                        expected_width=expected_width,
                        expected_height=expected_height,
                        force_derive=args.force_derive,
                        profile=args.profile,
                    ),
                logger=logger,
                heartbeat_seconds=args.heartbeat_seconds,
            )
        if label_candidates:
            _run_stage(
                stage_name="label",
                episodes=label_candidates,
                max_workers=args.label_jobs,
                submit_fn=lambda executor, episode: executor.submit(
                        _label_episode,
                        tool_path=tool_path,
                        project_root=project_root,
                        episode=episode,
                        force_label=args.force_label,
                        profile=args.profile,
                        logger=logger,
                    ),
                logger=logger,
                heartbeat_seconds=args.heartbeat_seconds,
            )

    pack_cmd = [
        str(tool_path),
        "pack",
        "--output-dir",
        str(output_dir),
        "--observed-role",
        "fighter1",
        "--observed-role",
        "fighter2",
        *(["--force"] if args.force_pack else []),
        *(["--profile"] if args.profile else []),
    ]
    for episode_root in eligible_roots:
        pack_cmd.extend(["--episode", str(episode_root)])
    _run(pack_cmd, cwd=project_root, dry_run=args.dry_run)

    print("multimodal pack build plan complete")
    logger.log("run_completed", pack_output_dir=str(output_dir))


if __name__ == "__main__":
    main()
