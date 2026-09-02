from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dfb_game_py import EpisodeRecording

from dfb_reinforcement_learning.envs.policy_dogfight_env import _extract_episode_info
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig


def _package_version() -> str:
    return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()


def _episode_states(episode_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    recording = EpisodeRecording(str(episode_root))
    manifest = json.loads(recording.manifest_json())
    initial_snapshot = json.loads(recording.initial_snapshot_json())
    steps = json.loads(recording.steps_json())
    initial_state = initial_snapshot["state"]
    return manifest, steps, initial_state


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _scale_speed(state: dict[str, Any], scale: float) -> dict[str, Any]:
    next_state = copy.deepcopy(state)
    velocity = np.asarray(next_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-6:
        forward = _normalize(np.asarray(next_state.get("forward", [0.0, 0.0, 1.0]), dtype=np.float32))
        velocity = forward * scale
    else:
        velocity = velocity * scale
    next_state["linear_velocity"] = velocity.tolist()
    return next_state


def _attack_component_breakdown(
    composer: PolicyRewardComposer,
    *,
    attacker_state: dict[str, Any],
    defender_state: dict[str, Any],
) -> dict[str, float]:
    cfg = composer.config
    components = composer._compute_attack_advantage(attacker_state=attacker_state, defender_state=defender_state)
    tracking_contrib = components.tau_gate * cfg.tracking_quality_weight * components.tracking_quality
    shot_mix = cfg.shot_outer_weight * components.shot_outer_score + cfg.shot_core_weight * components.shot_core_score
    shot_contrib = components.tau_gate * cfg.shot_feasibility_weight * shot_mix
    tail_contrib = components.tau_gate * cfg.tail_hold_weight * components.tail_hold_score
    internal_sum = tracking_contrib + shot_contrib + tail_contrib
    return {
        "tau_seconds": float(components.tau_seconds),
        "tau_gate": float(components.tau_gate),
        "tracking_quality": float(components.tracking_quality),
        "fire_alignment_score": float(components.fire_alignment_score),
        "shot_outer_score": float(components.shot_outer_score),
        "shot_core_score": float(components.shot_core_score),
        "shot_feasibility": float(components.shot_feasibility),
        "tail_hold_score": float(components.tail_hold_score),
        "tracking_contribution": float(tracking_contrib),
        "shot_contribution": float(shot_contrib),
        "tail_contribution": float(tail_contrib),
        "attack_internal_sum": float(internal_sum),
        "attack_advantage_unweighted": float(components.attack_advantage),
        "attack_advantage_weighted": float(cfg.attack_advantage_weight * components.attack_advantage),
    }


def sweep_attack_components(
    *,
    composer: PolicyRewardComposer,
    attacker_state: dict[str, Any],
    defender_state: dict[str, Any],
    speed_scales: Sequence[float],
) -> list[dict[str, Any]]:
    base_velocity = np.asarray(attacker_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
    base_speed = float(np.linalg.norm(base_velocity))
    rows: list[dict[str, Any]] = []
    for scale in speed_scales:
        scaled_attacker = _scale_speed(attacker_state, float(scale))
        scaled_speed = float(np.linalg.norm(np.asarray(scaled_attacker["linear_velocity"], dtype=np.float32)))
        row = {
            "speed_scale": float(scale),
            "attacker_speed_mps": float(scaled_speed),
            "base_attacker_speed_mps": float(base_speed),
        }
        row.update(
            _attack_component_breakdown(
                composer,
                attacker_state=scaled_attacker,
                defender_state=defender_state,
            )
        )
        rows.append(row)
    return rows


def build_attack_component_report(
    *,
    episode_id: str,
    ego_role: str,
    step_index: int,
    rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Attack Component Sweep Audit",
        "",
        f"- Episode: `{episode_id}`",
        f"- Ego Role: `{ego_role}`",
        f"- Step: `{step_index}`",
        f"- Samples: `{len(rows)}`",
        "",
        "| speed_scale | speed_mps | tau | tracking | shot | tail | attack_unweighted | attack_weighted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['speed_scale']:.3f} | "
            f"{row['attacker_speed_mps']:.3f} | "
            f"{row['tau_seconds']:.3f} | "
            f"{row['tracking_contribution']:.3f} | "
            f"{row['shot_contribution']:.3f} | "
            f"{row['tail_contribution']:.3f} | "
            f"{row['attack_advantage_unweighted']:.3f} | "
            f"{row['attack_advantage_weighted']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _default_speed_scales() -> list[float]:
    return [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit tracking/shot/tail contributions for a chosen episode frame under attacker-speed sweeps."
    )
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--ego-role", default="fighter1")
    parser.add_argument("--step-index", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speed-scales", type=float, nargs="*", default=_default_speed_scales())
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episode_root = Path(args.episode_root)
    output_dir = Path(args.output_dir)
    manifest, steps, initial_state = _episode_states(episode_root)
    step_index = int(args.step_index)
    if step_index < 0 or step_index >= len(steps):
        raise SystemExit(f"step-index {step_index} out of range for {len(steps)} recorded steps")
    state = steps[step_index]["state"]
    info = _extract_episode_info(state, ego_role=args.ego_role)
    aircraft_by_role = info["aircraft_by_role"]
    ego_role = args.ego_role
    enemy_role = "fighter2" if ego_role == "fighter1" else "fighter1"
    composer = PolicyRewardComposer(PolicyRewardConfig())
    rows = sweep_attack_components(
        composer=composer,
        attacker_state=aircraft_by_role[ego_role],
        defender_state=aircraft_by_role[enemy_role],
        speed_scales=args.speed_scales,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_attack_component_report(
        episode_id=str(manifest.get("episode_id")),
        ego_role=ego_role,
        step_index=step_index,
        rows=rows,
    )
    json_path = output_dir / "attack_component_sweep.json"
    report_path = output_dir / "attack_component_sweep.md"
    payload = {
        "tool_version": _package_version(),
        "episode_root": str(episode_root),
        "episode_id": manifest.get("episode_id"),
        "ego_role": ego_role,
        "step_index": step_index,
        "speed_scales": list(args.speed_scales),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "attack_component_sweep_path": str(json_path),
                "attack_component_report_path": str(report_path),
                "episode_id": manifest.get("episode_id"),
                "step_index": step_index,
                "row_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
