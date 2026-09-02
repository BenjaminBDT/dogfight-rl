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
from dfb_reinforcement_learning.tools.audit_attack_components import _attack_component_breakdown


def _package_version() -> str:
    return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()


def _episode_steps(episode_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recording = EpisodeRecording(str(episode_root))
    manifest = json.loads(recording.manifest_json())
    steps = json.loads(recording.steps_json())
    return manifest, steps


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def _set_speed_magnitude(state: dict[str, Any], speed_mps: float) -> dict[str, Any]:
    next_state = copy.deepcopy(state)
    velocity = np.asarray(next_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
    direction = _normalize(velocity)
    if float(np.linalg.norm(velocity)) <= 1e-6:
        direction = _normalize(np.asarray(next_state.get("forward", [0.0, 0.0, 1.0]), dtype=np.float32))
    next_state["linear_velocity"] = (direction * speed_mps).tolist()
    return next_state


def _row_for_step(
    composer: PolicyRewardComposer,
    *,
    attacker_state: dict[str, Any],
    defender_state: dict[str, Any],
    counterfactual_speed_mps: float,
) -> dict[str, Any]:
    actual = _attack_component_breakdown(composer, attacker_state=attacker_state, defender_state=defender_state)
    counterfactual_state = _set_speed_magnitude(attacker_state, counterfactual_speed_mps)
    counterfactual = _attack_component_breakdown(
        composer,
        attacker_state=counterfactual_state,
        defender_state=defender_state,
    )
    actual_speed = float(np.linalg.norm(np.asarray(attacker_state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)))
    delta = {f"{key}_delta": float(counterfactual.get(key, 0.0) - actual.get(key, 0.0)) for key in (
        "tracking_contribution",
        "shot_contribution",
        "lead_contribution",
        "tail_contribution",
        "attack_advantage_unweighted",
        "attack_advantage_weighted",
    )}
    return {
        "actual_speed_mps": actual_speed,
        "counterfactual_speed_mps": float(counterfactual_speed_mps),
        "actual": actual,
        "counterfactual": counterfactual,
        **delta,
    }


def audit_attack_path(
    *,
    composer: PolicyRewardComposer,
    steps: Sequence[dict[str, Any]],
    ego_role: str,
    start_step_index: int,
    end_step_index: int,
    reference_step_index: int,
) -> dict[str, Any]:
    if start_step_index > end_step_index:
        raise ValueError("start_step_index must be <= end_step_index")
    if reference_step_index < 0 or reference_step_index >= len(steps):
        raise ValueError("reference_step_index out of range")
    reference_info = _extract_episode_info(steps[reference_step_index]["state"], ego_role=ego_role)
    enemy_role = "fighter2" if ego_role == "fighter1" else "fighter1"
    reference_speed_mps = float(
        np.linalg.norm(
            np.asarray(
                reference_info["aircraft_by_role"][ego_role].get("linear_velocity", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
        )
    )
    rows: list[dict[str, Any]] = []
    for step_index in range(start_step_index, end_step_index + 1):
        info = _extract_episode_info(steps[step_index]["state"], ego_role=ego_role)
        aircraft_by_role = info["aircraft_by_role"]
        row = _row_for_step(
            composer,
            attacker_state=aircraft_by_role[ego_role],
            defender_state=aircraft_by_role[enemy_role],
            counterfactual_speed_mps=reference_speed_mps,
        )
        row["step_index"] = step_index
        row["tick"] = int(info["tick"])
        row["sim_time_seconds"] = float(info["sim_time_seconds"])
        rows.append(row)
    return {
        "reference_step_index": reference_step_index,
        "reference_speed_mps": reference_speed_mps,
        "start_step_index": start_step_index,
        "end_step_index": end_step_index,
        "rows": rows,
    }


def build_path_audit_report(
    *,
    episode_id: str,
    ego_role: str,
    audit: dict[str, Any],
) -> str:
    rows = audit["rows"]
    lines = [
        "# Counterfactual Attack Path Audit",
        "",
        f"- Episode: `{episode_id}`",
        f"- Ego Role: `{ego_role}`",
        f"- Range: `{audit['start_step_index']} .. {audit['end_step_index']}`",
        f"- Reference Step: `{audit['reference_step_index']}`",
        f"- Reference Speed: `{audit['reference_speed_mps']:.3f} m/s`",
        "",
        "| step | actual_speed | counterfactual_speed | shot_delta | lead_delta | attack_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['step_index']} | "
            f"{row['actual_speed_mps']:.3f} | "
            f"{row['counterfactual_speed_mps']:.3f} | "
            f"{row['shot_contribution_delta']:.3f} | "
            f"{row['lead_contribution_delta']:.3f} | "
            f"{row['attack_advantage_weighted_delta']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare actual attack-component growth against a counterfactual fixed-speed path."
    )
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--ego-role", default="fighter1")
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--reference-step", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episode_root = Path(args.episode_root)
    manifest, steps = _episode_steps(episode_root)
    composer = PolicyRewardComposer(PolicyRewardConfig())
    audit = audit_attack_path(
        composer=composer,
        steps=steps,
        ego_role=args.ego_role,
        start_step_index=args.start_step,
        end_step_index=args.end_step,
        reference_step_index=args.reference_step,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "attack_component_path_audit.json"
    report_path = output_dir / "attack_component_path_audit.md"
    payload = {
        "tool_version": _package_version(),
        "episode_root": str(episode_root),
        "episode_id": manifest.get("episode_id"),
        "ego_role": args.ego_role,
        **audit,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_path_audit_report(
            episode_id=str(manifest.get("episode_id")),
            ego_role=args.ego_role,
            audit=audit,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "attack_component_path_audit_path": str(json_path),
                "attack_component_path_report_path": str(report_path),
                "episode_id": manifest.get("episode_id"),
                "start_step_index": args.start_step,
                "end_step_index": args.end_step,
                "reference_step_index": args.reference_step,
                "row_count": len(audit["rows"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
