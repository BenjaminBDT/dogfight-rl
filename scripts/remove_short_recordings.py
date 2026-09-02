#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    episode_root: str
    scene_name: str
    duration_seconds: float
    total_steps: int
    fixed_time_step_seconds: float
    duration_source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove recorded episodes shorter than a minimum duration. "
            "Defaults to dry-run; pass --apply to delete directories."
        )
    )
    parser.add_argument(
        "--recordings-root",
        default="datasets/dfb_game/recordings",
        help="Root directory containing episode subdirectories.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=30.0,
        help="Delete episodes whose duration is strictly less than this threshold.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete episode directories. Without this flag, only print matches.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a JSON report of matched episodes.",
    )
    return parser.parse_args()


def _extract_str(field: str, text: str) -> str | None:
    match = re.search(rf"{re.escape(field)}:\s*\"([^\"]*)\"", text)
    return match.group(1) if match else None


def _extract_float(field: str, text: str) -> float | None:
    match = re.search(rf"{re.escape(field)}:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    return float(match.group(1)) if match else None


def _extract_int(field: str, text: str) -> int | None:
    match = re.search(rf"{re.escape(field)}:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _read_episode_summary(episode_manifest_path: Path) -> EpisodeSummary:
    text = episode_manifest_path.read_text(encoding="utf-8")
    episode_id = _extract_str("episode_id", text) or episode_manifest_path.parent.name
    scene_name = _extract_str("scene_name", text) or "unknown"
    total_steps = _extract_int("total_steps", text)
    fixed_dt = _extract_float("fixed_time_step_seconds", text)
    started_sim = _extract_float("started_sim_time_seconds", text)
    final_sim = _extract_float("final_sim_time_seconds", text)
    if total_steps is None or fixed_dt is None:
        raise ValueError(f"missing total_steps or fixed_time_step_seconds in {episode_manifest_path}")
    if started_sim is not None and final_sim is not None and final_sim >= started_sim:
        duration_seconds = final_sim - started_sim
        duration_source = "sim_time_range"
    else:
        duration_seconds = total_steps * fixed_dt
        duration_source = "total_steps_x_fixed_dt"
    return EpisodeSummary(
        episode_id=episode_id,
        episode_root=str(episode_manifest_path.parent),
        scene_name=scene_name,
        duration_seconds=float(duration_seconds),
        total_steps=total_steps,
        fixed_time_step_seconds=float(fixed_dt),
        duration_source=duration_source,
    )


def _iter_episode_manifests(recordings_root: Path) -> list[Path]:
    return sorted(path for path in recordings_root.glob("*/episode.ron") if path.is_file())


def main() -> None:
    args = _parse_args()
    recordings_root = Path(args.recordings_root).resolve()
    if not recordings_root.is_dir():
        raise SystemExit(f"recordings root does not exist: {recordings_root}")
    matched: list[EpisodeSummary] = []
    skipped: list[str] = []
    for manifest_path in _iter_episode_manifests(recordings_root):
        try:
            episode = _read_episode_summary(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            skipped.append(f"{manifest_path}: {exc}")
            continue
        if episode.duration_seconds < args.min_duration_seconds:
            matched.append(episode)

    matched.sort(key=lambda item: (item.duration_seconds, item.episode_id))
    total_duration = sum(item.duration_seconds for item in matched)
    print(
        f"recordings_root={recordings_root}\n"
        f"min_duration_seconds={args.min_duration_seconds:.3f}\n"
        f"matched_episodes={len(matched)}\n"
        f"matched_total_duration_seconds={total_duration:.3f}\n"
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    for item in matched:
        print(
            f"{item.episode_id} | scene={item.scene_name} | "
            f"duration={item.duration_seconds:.3f}s | steps={item.total_steps} | "
            f"path={item.episode_root}"
        )

    if skipped:
        print(f"skipped_manifests={len(skipped)}")
        for line in skipped:
            print(f"skip: {line}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "recordings_root": str(recordings_root),
            "min_duration_seconds": float(args.min_duration_seconds),
            "apply": bool(args.apply),
            "matched_episode_count": len(matched),
            "matched_total_duration_seconds": total_duration,
            "matched_episodes": [asdict(item) for item in matched],
            "skipped_manifests": skipped,
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote report: {report_path}")

    if not args.apply:
        return

    removed_count = 0
    for item in matched:
        episode_root = Path(item.episode_root)
        shutil.rmtree(episode_root)
        removed_count += 1
    print(f"removed_episodes={removed_count}")


if __name__ == "__main__":
    main()
