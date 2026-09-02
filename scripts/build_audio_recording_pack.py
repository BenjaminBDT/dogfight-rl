#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
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
    existing_derived_roles: list[str]
    existing_label_roles: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed-observer audio training pack from authoritative recordings, "
            "filtering out low-duration episodes and backfilling missing extract/label artifacts."
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
        default=Path(
            "runs/dfb_state_estimation/audio_recording_dual_observer_min20s_pack_160x100_v1"
        ),
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
        "--visual-resolution",
        default="160x100",
        help="Resolution used during reconstruction for derived modalities.",
    )
    parser.add_argument(
        "--audio-window-seconds",
        type=float,
        default=1.0 / 60.0,
    )
    parser.add_argument(
        "--force-pack",
        action="store_true",
        help="Overwrite the output pack if it already exists.",
    )
    parser.add_argument(
        "--force-derive",
        action="store_true",
        help="Force regeneration of existing derived extract/label artifacts.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume target/debug/dfb_tool_dataset already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the selection audit JSON and print planned commands.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tool_path(project_root: Path) -> Path:
    return project_root / "target" / "debug" / "dfb_tool_dataset"


def _episode_info(episode_root: Path) -> EpisodeInfo:
    episode_text = (episode_root / "episode.ron").read_text(encoding="utf-8", errors="ignore")
    scene_match = re.search(r'scene_name:\s*"([^"]+)"', episode_text)
    steps_match = re.search(r"total_steps:\s*(\d+)", episode_text)
    dt_match = re.search(r"fixed_time_step_seconds:\s*([0-9.]+)", episode_text)
    scene_name = scene_match.group(1) if scene_match else "unknown"
    total_steps = int(steps_match.group(1)) if steps_match else 0
    fixed_time_step_seconds = float(dt_match.group(1)) if dt_match else (1.0 / 60.0)
    duration_seconds = total_steps * fixed_time_step_seconds
    existing_derived_roles: list[str] = []
    existing_label_roles: list[str] = []
    derived_root = episode_root / "derived"
    if derived_root.exists():
        for role_root in sorted([path for path in derived_root.iterdir() if path.is_dir()]):
            role_name = role_root.name
            if (role_root / "derived_modalities.ron").exists():
                existing_derived_roles.append(role_name)
            if (role_root / "derived_labels.ron").exists():
                existing_label_roles.append(role_name)
    return EpisodeInfo(
        name=episode_root.name,
        path=str(episode_root),
        scene_name=scene_name,
        total_steps=total_steps,
        fixed_time_step_seconds=fixed_time_step_seconds,
        duration_seconds=duration_seconds,
        existing_derived_roles=existing_derived_roles,
        existing_label_roles=existing_label_roles,
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


def main() -> None:
    args = _parse_args()
    project_root = _project_root()
    recordings_root = (project_root / args.recordings_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    selection_json = (
        args.selection_json.resolve()
        if args.selection_json is not None
        else output_dir.with_name(output_dir.name + "_selection.json")
    )

    episodes = _scan_episodes(recordings_root)
    eligible = [item for item in episodes if item.duration_seconds >= args.min_duration_seconds]
    excluded = [item for item in episodes if item.duration_seconds < args.min_duration_seconds]

    selection_payload = {
        "recordings_root": str(recordings_root),
        "output_dir": str(output_dir),
        "min_duration_seconds": args.min_duration_seconds,
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

    tool_path = _tool_path(project_root)
    if not args.skip_build and not args.dry_run:
        _run(
            ["cargo", "build", "--bin", "dfb_tool_dataset"],
            cwd=project_root,
            dry_run=False,
        )
    elif not tool_path.exists():
        raise SystemExit(
            f"missing tool binary at {tool_path}; rerun without --skip-build or build it first"
        )

    for episode in eligible:
        episode_root = Path(episode.path)
        required_roles = ("fighter1", "fighter2")
        missing_derived = [
            role
            for role in required_roles
            if args.force_derive or role not in episode.existing_derived_roles
        ]
        if missing_derived:
            observed_role_arg = (
                "all" if len(missing_derived) == 2 else missing_derived[0]
            )
            _run(
                [
                    str(tool_path),
                    "extract",
                    "--episode",
                    str(episode_root),
                    "--observed-role",
                    observed_role_arg,
                    "--visual-resolution",
                    args.visual_resolution,
                    "--audio-window",
                    str(args.audio_window_seconds),
                    *(["--force"] if args.force_derive else []),
                ],
                cwd=project_root,
                dry_run=args.dry_run,
            )

        missing_labels = [
            role
            for role in required_roles
            if args.force_derive or role not in episode.existing_label_roles
        ]
        if missing_labels:
            observed_role_arg = (
                "all" if len(missing_labels) == 2 else ",".join(missing_labels)
            )
            _run(
                [
                    str(tool_path),
                    "label",
                    "--episode",
                    str(episode_root),
                    "--observed-role",
                    observed_role_arg,
                    *(["--force"] if args.force_derive else []),
                ],
                cwd=project_root,
                dry_run=args.dry_run,
            )

    pack_cmd = [str(tool_path), "pack", "--output-dir", str(output_dir)]
    if args.force_pack:
        pack_cmd.append("--force")
    for episode in eligible:
        pack_cmd.extend(["--episode", episode.path])
    _run(pack_cmd, cwd=project_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
