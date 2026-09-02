from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from dfb_game_py import EpisodeRecording

from dfb_reinforcement_learning.envs.policy_dogfight_env import _extract_episode_info
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.policy_contract import ACTION_SCHEMA_ID, POLICY_CONTRACT_ID
from dfb_reinforcement_learning.rewards import PolicyRewardComposer, PolicyRewardConfig


def _append_reward_history_frame(
    history: list[dict[str, Any]],
    *,
    info: dict[str, Any],
    action_cont: np.ndarray,
    action_bin: np.ndarray,
    attack_history_cache: dict[str, Any] | None,
    boundary_phi_cache: dict[str, float] | None,
    max_length: int,
) -> None:
    frame: dict[str, Any] = {
        "info": info,
        "action_cont": np.asarray(action_cont, dtype=np.float32).copy(),
        "action_bin": np.asarray(action_bin, dtype=np.float32).copy(),
    }
    if attack_history_cache is not None:
        frame["attack_history_cache"] = attack_history_cache
    if boundary_phi_cache is not None:
        frame["boundary_phi_cache"] = boundary_phi_cache
    history.append(frame)
    if len(history) > max_length:
        del history[:-max_length]


def _command_for_role(step: dict[str, Any], ego_role: str) -> dict[str, Any]:
    command_key = f"{ego_role}_command"
    command = step.get(command_key)
    if not isinstance(command, dict):
        raise ValueError(f"recorded step missing {command_key}")
    return command


def _recorded_action_arrays(step: dict[str, Any], ego_role: str) -> tuple[np.ndarray, np.ndarray]:
    command = _command_for_role(step, ego_role)
    action_cont = np.asarray(
        [
            float(command.get("throttle", 0.0)),
            float(command.get("pitch", 0.0)),
            float(command.get("roll", 0.0)),
            float(command.get("yaw", 0.0)),
        ],
        dtype=np.float32,
    )
    action_bin = np.asarray(
        [
            1.0 if bool(command.get("brake", False)) else 0.0,
            1.0 if bool(command.get("fire_gun", False)) else 0.0,
            1.0 if bool(command.get("repair", False)) else 0.0,
        ],
        dtype=np.float32,
    )
    return action_cont, action_bin


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _package_version() -> str:
    return (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()


def _validate_policy_recording_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("policy_contract_id") != POLICY_CONTRACT_ID:
        raise ValueError(
            "recording policy_contract_id="
            f"{manifest.get('policy_contract_id')!r}, expected {POLICY_CONTRACT_ID!r}"
        )
    if manifest.get("action_schema_id") != ACTION_SCHEMA_ID:
        raise ValueError(
            "recording action_schema_id="
            f"{manifest.get('action_schema_id')!r}, expected {ACTION_SCHEMA_ID!r}"
        )
    if manifest.get("authoritative_source") is not True:
        raise ValueError("reward diagnosis requires an authoritative recording")


def replay_episode_reward_frames(
    *,
    initial_state: dict[str, Any],
    steps: Sequence[dict[str, Any]],
    ego_role: str,
    reward_composer: PolicyRewardComposer | None = None,
    obs_adapter: PolicyObservationAdapter | None = None,
) -> list[dict[str, Any]]:
    composer = reward_composer or PolicyRewardComposer(PolicyRewardConfig())
    adapter = obs_adapter or PolicyObservationAdapter()
    episode_start_sim_time_seconds = float(initial_state["sim_time_seconds"])
    previous_info = _extract_episode_info(initial_state, ego_role=ego_role)
    previous_previous_action_cont: np.ndarray | None = None
    previous_action_cont: np.ndarray | None = None
    reward_history_frames: list[dict[str, Any]] = []
    reward_history_frame_length = composer.reward_history_frame_length()
    records: list[dict[str, Any]] = []

    for step in steps:
        state = step.get("state")
        if not isinstance(state, dict):
            raise ValueError("recorded step missing 'state'")
        obs_payload = adapter.build(
            state,
            ego_role,
            episode_start_sim_time_seconds=episode_start_sim_time_seconds,
        )
        current_obs = np.asarray(obs_payload["vector"], dtype=np.float32)
        current_info = _extract_episode_info(state, ego_role=ego_role)
        action_cont, action_bin = _recorded_action_arrays(step, ego_role)
        reward = composer.compute(
            previous_info=previous_info,
            previous_previous_action_cont=previous_previous_action_cont,
            previous_action_cont=previous_action_cont,
            reward_history={"frames": reward_history_frames},
            current_info=current_info,
            current_obs=current_obs,
            action_cont=action_cont,
            action_bin=action_bin,
        )
        records.append(
            {
                "step_index": int(step.get("index", len(records))),
                "tick": int(step.get("tick", current_info["tick"])),
                "sim_time_seconds": float(step.get("sim_time_seconds", current_info["sim_time_seconds"])),
                "ego_role": ego_role,
                "action_cont": action_cont.tolist(),
                "action_bin": action_bin.tolist(),
                "action_named": _jsonable(_command_for_role(step, ego_role)),
                "reward": reward.asdict(),
                "obs_components": _jsonable(obs_payload["components"]),
                "info": _jsonable(current_info),
            }
        )
        _append_reward_history_frame(
            reward_history_frames,
            info=current_info,
            action_cont=action_cont,
            action_bin=action_bin,
            attack_history_cache=composer.build_attack_history_cache(current_info),
            boundary_phi_cache=composer.build_boundary_phi_cache(current_info),
            max_length=reward_history_frame_length,
        )
        previous_info = current_info
        previous_previous_action_cont = previous_action_cont
        previous_action_cont = action_cont

    return records


def detect_base_events(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_record: dict[str, Any] | None = None
    for record in records:
        reward = record["reward"]
        current_step = int(record["step_index"])
        base_payload = {
            "step_index": current_step,
            "tick": int(record["tick"]),
            "sim_time_seconds": float(record["sim_time_seconds"]),
        }
        brake_now = float(record["action_bin"][0]) > 0.5
        fire_now = float(record["action_bin"][1]) > 0.5
        out_of_bounds_now = float(record["info"]["aircraft_by_role"][record["ego_role"]]["out_of_bounds_seconds"]) > 0.0
        boundary_threat_now = float(
            reward["ground_boundary_threat"]
            + reward["ceiling_boundary_threat"]
            + reward["horizontal_boundary_threat"]
        )
        if previous_record is None:
            previous_record = record
            continue

        previous_reward = previous_record["reward"]
        brake_prev = float(previous_record["action_bin"][0]) > 0.5
        fire_prev = float(previous_record["action_bin"][1]) > 0.5
        out_of_bounds_prev = (
            float(previous_record["info"]["aircraft_by_role"][record["ego_role"]]["out_of_bounds_seconds"]) > 0.0
        )
        boundary_threat_prev = float(
            previous_reward["ground_boundary_threat"]
            + previous_reward["ceiling_boundary_threat"]
            + previous_reward["horizontal_boundary_threat"]
        )

        if brake_now and not brake_prev:
            events.append(
                {
                    "kind": "brake_onset",
                    **base_payload,
                    "brake_command": float(record["action_bin"][0]),
                    "closing_speed_mps": float(reward["closing_speed_mps"]),
                }
            )
        if fire_now and not fire_prev:
            events.append(
                {
                    "kind": "fire_onset",
                    **base_payload,
                    "shot_feasibility": float(reward["shot_feasibility"]),
                    "fire_command_bonus": float(reward.get("fire_command_bonus", 0.0)),
                    "fire_window_bonus": float(reward["fire_window_bonus"]),
                }
            )

        attack_drop = float(previous_reward["attack_advantage"]) - float(reward["attack_advantage"])
        if attack_drop >= 1.0 and float(previous_reward["attack_advantage"]) >= 1.0:
            events.append(
                {
                    "kind": "attack_advantage_collapse",
                    **base_payload,
                    "attack_advantage_prev": float(previous_reward["attack_advantage"]),
                    "attack_advantage_now": float(reward["attack_advantage"]),
                    "attack_drop": attack_drop,
                }
            )

        collision_rise = float(reward["aircraft_collision_threat"]) - float(previous_reward["aircraft_collision_threat"])
        if collision_rise >= 0.5 or (
            float(reward["aircraft_collision_threat"]) >= 0.75
            and float(previous_reward["aircraft_collision_threat"]) < 0.75
        ):
            events.append(
                {
                    "kind": "collision_threat_spike",
                    **base_payload,
                    "collision_threat_prev": float(previous_reward["aircraft_collision_threat"]),
                    "collision_threat_now": float(reward["aircraft_collision_threat"]),
                    "collision_rise": collision_rise,
                }
            )

        boundary_rise = boundary_threat_now - boundary_threat_prev
        if boundary_rise >= 0.35 or (boundary_threat_now >= 0.5 and boundary_threat_prev < 0.5):
            events.append(
                {
                    "kind": "boundary_threat_spike",
                    **base_payload,
                    "boundary_threat_prev": boundary_threat_prev,
                    "boundary_threat_now": boundary_threat_now,
                    "boundary_rise": boundary_rise,
                }
            )

        if out_of_bounds_now and not out_of_bounds_prev:
            events.append(
                {
                    "kind": "out_of_bounds_entry",
                    **base_payload,
                    "out_of_bounds_seconds": float(
                        record["info"]["aircraft_by_role"][record["ego_role"]]["out_of_bounds_seconds"]
                    ),
                    "boundary_threat_now": boundary_threat_now,
                }
            )

        previous_record = record

    return events


def _record_by_step_index(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["step_index"]): record for record in records}


def _window_mean(records: Sequence[dict[str, Any]], start: int, end: int, field: str) -> float:
    if start > end:
        return 0.0
    values = [float(record["reward"][field]) for record in records[start : end + 1]]
    if not values:
        return 0.0
    return float(np.mean(values))


def _reward_value(record: dict[str, Any], field: str) -> float:
    return float(record["reward"].get(field, 0.0))


def _candidate_head_on_event_score(record: dict[str, Any], event: dict[str, Any]) -> float:
    reward = record["reward"]
    event_kind = str(event["kind"])
    event_kind_bonus = {
        "brake_onset": 0.30,
        "attack_advantage_collapse": 0.25,
        "collision_threat_spike": 0.20,
    }.get(event_kind, 0.0)
    return float(
        1.2 * float(reward["two_circle_gate"])
        + 0.7 * min(float(reward["closing_speed_mps"]) / 120.0, 1.5)
        + 0.6 * float(reward["shot_feasibility"])
        + 0.3 * min(float(reward["attack_advantage"]) / 5.0, 1.5)
        + event_kind_bonus
    )


def detect_head_on_crossing_windows(
    records: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    *,
    window_radius: int = 20,
    merge_step_gap: int = 10,
) -> list[dict[str, Any]]:
    if not records:
        return []
    by_step = _record_by_step_index(records)
    indexed_records = {int(record["step_index"]): idx for idx, record in enumerate(records)}
    candidate_events: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] not in {"brake_onset", "attack_advantage_collapse", "collision_threat_spike"}:
            continue
        step_index = int(event["step_index"])
        record = by_step.get(step_index)
        if record is None:
            continue
        reward = record["reward"]
        if float(reward["two_circle_gate"]) < 0.15:
            continue
        if float(reward["closing_speed_mps"]) < 40.0:
            continue
        if (
            float(reward["shot_feasibility"]) < 0.08
            and float(reward["fire_window_bonus"]) < 0.05
            and float(reward["attack_advantage"]) < 0.8
        ):
            continue
        candidate_events.append(
            {
                "event": event,
                "record": record,
                "record_index": indexed_records[step_index],
                "score": _candidate_head_on_event_score(record, event),
            }
        )

    candidate_events.sort(key=lambda item: (item["record_index"], -item["score"]))
    selected: list[dict[str, Any]] = []
    for candidate in candidate_events:
        if selected and candidate["record_index"] - selected[-1]["record_index"] <= merge_step_gap:
            if candidate["score"] > selected[-1]["score"]:
                selected[-1] = candidate
            continue
        selected.append(candidate)

    windows: list[dict[str, Any]] = []
    for candidate in selected:
        anchor_index = int(candidate["record_index"])
        anchor_record = candidate["record"]
        anchor_step = int(anchor_record["step_index"])
        start_index = max(anchor_index - window_radius, 0)
        end_index = min(anchor_index + window_radius, len(records) - 1)
        reward = anchor_record["reward"]
        negative_fields = (
            "threat_advantage",
            "aircraft_collision_threat",
            "fire_hesitation_penalty",
            "brake_penalty",
        )
        positive_fields = (
            "attack_advantage",
            "shot_feasibility",
            "predictive_fire_hit_bonus",
            "fire_command_bonus",
            "fire_window_bonus",
            "two_circle_speed_reward",
        )
        top_negative = sorted(
            ((field, float(reward.get(field, 0.0))) for field in negative_fields),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_positive = sorted(
            ((field, float(reward.get(field, 0.0))) for field in positive_fields),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        windows.append(
            {
                "anchor_kind": str(candidate["event"]["kind"]),
                "anchor_step_index": anchor_step,
                "anchor_tick": int(anchor_record["tick"]),
                "anchor_sim_time_seconds": float(anchor_record["sim_time_seconds"]),
                "window_start_step_index": int(records[start_index]["step_index"]),
                "window_end_step_index": int(records[end_index]["step_index"]),
                "score": float(candidate["score"]),
                "anchor": {
                    "attack_advantage": float(reward.get("attack_advantage", 0.0)),
                    "threat_advantage": float(reward.get("threat_advantage", 0.0)),
                    "shot_feasibility": float(reward.get("shot_feasibility", 0.0)),
                    "fire_window_bonus": float(reward.get("fire_window_bonus", 0.0)),
                    "aircraft_collision_threat": float(reward.get("aircraft_collision_threat", 0.0)),
                    "two_circle_gate": float(reward.get("two_circle_gate", 0.0)),
                    "closing_speed_mps": float(reward.get("closing_speed_mps", 0.0)),
                    "speed_delta_mps": float(reward.get("speed_delta_mps", 0.0)),
                    "brake_command": float(anchor_record["action_bin"][0]),
                    "fire_command": float(anchor_record["action_bin"][1]),
                    "throttle_command": float(anchor_record["action_cont"][0]),
                },
                "pre_means": {
                    "attack_advantage": _window_mean(records, start_index, max(anchor_index - 1, start_index), "attack_advantage"),
                    "threat_advantage": _window_mean(records, start_index, max(anchor_index - 1, start_index), "threat_advantage"),
                    "shot_feasibility": _window_mean(records, start_index, max(anchor_index - 1, start_index), "shot_feasibility"),
                    "fire_window_bonus": _window_mean(records, start_index, max(anchor_index - 1, start_index), "fire_window_bonus"),
                    "aircraft_collision_threat": _window_mean(records, start_index, max(anchor_index - 1, start_index), "aircraft_collision_threat"),
                },
                "post_means": {
                    "attack_advantage": _window_mean(records, min(anchor_index + 1, end_index), end_index, "attack_advantage"),
                    "threat_advantage": _window_mean(records, min(anchor_index + 1, end_index), end_index, "threat_advantage"),
                    "shot_feasibility": _window_mean(records, min(anchor_index + 1, end_index), end_index, "shot_feasibility"),
                    "fire_window_bonus": _window_mean(records, min(anchor_index + 1, end_index), end_index, "fire_window_bonus"),
                    "aircraft_collision_threat": _window_mean(records, min(anchor_index + 1, end_index), end_index, "aircraft_collision_threat"),
                },
                "top_negative_terms": [
                    {"field": field, "value": value} for field, value in top_negative
                ],
                "top_positive_terms": [
                    {"field": field, "value": value} for field, value in top_positive
                ],
            }
        )
    return windows


def build_head_on_crossing_report(
    *,
    summary: dict[str, Any],
    windows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Head-on Crossing Diagnosis",
        "",
        f"- Episode: `{summary.get('episode_id')}`",
        f"- Ego Role: `{summary.get('ego_role')}`",
        f"- Frame Count: `{summary.get('frame_count')}`",
        f"- Candidate Windows: `{len(windows)}`",
        "",
    ]
    if not windows:
        lines.extend(
            [
                "## Result",
                "",
                "No head-on crossing candidate window was detected under the current fixed thresholds.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Windows",
            "",
            "Each window is anchored on a base event that also satisfies the current two-circle / closing-speed / shot-opportunity filters.",
            "",
        ]
    )
    for index, window in enumerate(windows, start=1):
        anchor = window["anchor"]
        pre = window["pre_means"]
        post = window["post_means"]
        lines.extend(
            [
                f"### Window {index}",
                "",
                f"- Anchor: `{window['anchor_kind']}` at step `{window['anchor_step_index']}` tick `{window['anchor_tick']}`",
                f"- Range: `{window['window_start_step_index']} .. {window['window_end_step_index']}`",
                f"- Candidate Score: `{window['score']:.3f}`",
                f"- Anchor State: `two_circle_gate={anchor['two_circle_gate']:.3f}`, `closing_speed_mps={anchor['closing_speed_mps']:.3f}`, `brake={anchor['brake_command']:.0f}`, `fire={anchor['fire_command']:.0f}`, `throttle={anchor['throttle_command']:.3f}`",
                f"- Anchor Reward: `attack={anchor['attack_advantage']:.3f}`, `threat={anchor['threat_advantage']:.3f}`, `shot={anchor['shot_feasibility']:.3f}`, `fire_window={anchor['fire_window_bonus']:.3f}`, `collision={anchor['aircraft_collision_threat']:.3f}`",
                f"- Pre Mean: `attack={pre['attack_advantage']:.3f}`, `threat={pre['threat_advantage']:.3f}`, `shot={pre['shot_feasibility']:.3f}`, `fire_window={pre['fire_window_bonus']:.3f}`, `collision={pre['aircraft_collision_threat']:.3f}`",
                f"- Post Mean: `attack={post['attack_advantage']:.3f}`, `threat={post['threat_advantage']:.3f}`, `shot={post['shot_feasibility']:.3f}`, `fire_window={post['fire_window_bonus']:.3f}`, `collision={post['aircraft_collision_threat']:.3f}`",
                "- Top Positive Terms: "
                + ", ".join(f"`{item['field']}={item['value']:.3f}`" for item in window["top_positive_terms"]),
                "- Top Negative Terms: "
                + ", ".join(f"`{item['field']}={item['value']:.3f}`" for item in window["top_negative_terms"]),
                "",
            ]
        )
    return "\n".join(lines)


def _summarize_records(
    *,
    episode_root: str,
    ego_role: str,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    first_brake_step = next(
        (int(record["step_index"]) for record in records if float(record["action_bin"][0]) > 0.5),
        None,
    )
    first_fire_step = next(
        (int(record["step_index"]) for record in records if float(record["action_bin"][1]) > 0.5),
        None,
    )
    total_reward_sum = float(sum(float(record["reward"]["total"]) for record in records))
    max_attack_advantage = max((float(record["reward"]["attack_advantage"]) for record in records), default=0.0)
    max_threat_advantage = max((float(record["reward"]["threat_advantage"]) for record in records), default=0.0)
    max_collision_threat = max((float(record["reward"]["aircraft_collision_threat"]) for record in records), default=0.0)
    max_shot_feasibility = max((float(record["reward"]["shot_feasibility"]) for record in records), default=0.0)
    return {
        "tool_version": _package_version(),
        "episode_root": episode_root,
        "episode_id": manifest.get("episode_id"),
        "schema_version": manifest.get("schema_version"),
        "ego_role": ego_role,
        "frame_count": len(records),
        "first_brake_step": first_brake_step,
        "first_fire_step": first_fire_step,
        "total_reward_sum": total_reward_sum,
        "max_attack_advantage": max_attack_advantage,
        "max_threat_advantage": max_threat_advantage,
        "max_collision_threat": max_collision_threat,
        "max_shot_feasibility": max_shot_feasibility,
    }


def _summarize_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "event_count": len(events),
        "event_counts_by_kind": counts,
    }


def _write_frames_jsonl(records: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an authoritative episode and recompute the policy reward breakdown frame by frame."
    )
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--ego-role", default="fighter1")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episode_root = Path(args.episode_root)
    output_dir = Path(args.output_dir)
    recording = EpisodeRecording(str(episode_root))
    manifest = json.loads(recording.manifest_json())
    _validate_policy_recording_manifest(manifest)
    initial_snapshot = json.loads(recording.initial_snapshot_json())
    steps = json.loads(recording.steps_json())
    initial_state = initial_snapshot["state"]
    records = replay_episode_reward_frames(
        initial_state=initial_state,
        steps=steps,
        ego_role=args.ego_role,
    )
    events = detect_base_events(records)
    head_on_windows = detect_head_on_crossing_windows(records, events)
    summary = _summarize_records(
        episode_root=str(episode_root),
        ego_role=args.ego_role,
        manifest=manifest,
        records=records,
    )
    summary.update(_summarize_events(events))
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "frames.jsonl"
    events_path = output_dir / "events.json"
    head_on_report_path = output_dir / "head_on_crossing.md"
    summary_path = output_dir / "summary.json"
    _write_frames_jsonl(records, frames_path)
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    head_on_report_path.write_text(
        build_head_on_crossing_report(summary=summary, windows=head_on_windows),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "frames_path": str(frames_path),
                "events_path": str(events_path),
                "head_on_crossing_report_path": str(head_on_report_path),
                "summary_path": str(summary_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
