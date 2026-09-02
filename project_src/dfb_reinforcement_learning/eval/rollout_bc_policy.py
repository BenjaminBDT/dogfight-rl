from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.actions import ActionAdapter
from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.envs import PolicyDogfightEnv, PolicyDogfightEnvConfig
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_inference import (
    deterministic_policy_output,
    load_policy_model,
)
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    load_and_validate_policy_dataset,
)


@dataclass(frozen=True)
class EpisodeRolloutSummary:
    episode_index: int
    ego_role: str
    steps: int
    sim_time_seconds: float
    terminated: bool
    truncated: bool
    winner: str | None
    self_destroyed: bool
    enemy_destroyed: bool
    max_out_of_bounds_seconds: float
    fired_step_count: int
    repair_step_count: int
    gun_overheated_step_count: int
    distance_start: float | None
    distance_end: float | None
    distance_min: float | None
    distance_mean: float | None
    mean_action_cont: list[float]
    mean_abs_action_cont: list[float]
    std_action_cont: list[float]
    binary_probability_mean: list[float]
    binary_on_rate: list[float]
    nontrivial_control_rate: float
    trace_sample_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BC policy rollouts in the active policy environment.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-name", default="open_head_on_200m")
    parser.add_argument("--ego-role", default="fighter1", choices=["fighter1", "fighter2"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--ticks-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--trace-stride", type=int, default=20)
    parser.add_argument("--trace-limit", type=int, default=200)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_policy(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    normalizer: ObservationNormalizer,
    dataset_contract: PolicyDatasetContract,
) -> StatelessHybridActorCritic:
    return load_policy_model(
        checkpoint_path,
        normalizer=normalizer,
        dataset_contract=dataset_contract,
        device=device,
        context="eval checkpoint",
    )


def _policy_action(
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    obs: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = deterministic_policy_output(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=device,
    )
    return output.action_cont, output.binary_actions(), output.action_bin_prob


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(args.dataset_root)
    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    model = _load_policy(
        args.checkpoint,
        device,
        normalizer=normalizer,
        dataset_contract=dataset_contract,
    )

    episode_summaries: list[EpisodeRolloutSummary] = []
    episode_traces: list[dict[str, Any]] = []
    with PolicyDogfightEnv(
        PolicyDogfightEnvConfig(
            project_root=args.project_root,
            scene_name=args.scene_name,
            ego_role=args.ego_role,
            seed=args.seed,
            ticks_per_step=args.ticks_per_step,
        )
    ) as env:
        for episode_index in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_index, scene_name=args.scene_name)
            terminated = False
            truncated = False
            fired_step_count = 0
            repair_step_count = 0
            gun_overheated_step_count = 0
            max_oob = 0.0
            last_info = info
            distance_history: list[float] = []
            action_cont_history: list[np.ndarray] = []
            action_bin_history: list[np.ndarray] = []
            action_bin_prob_history: list[np.ndarray] = []
            trace: list[dict[str, Any]] = []
            for step_index in range(args.max_steps):
                action_cont, action_bin, action_bin_prob = _policy_action(model, normalizer, obs, device)
                obs, _, terminated, truncated, info = env.step_arrays(action_cont, action_bin)
                last_info = info
                actor = info["aircraft_by_role"][args.ego_role]
                max_oob = max(max_oob, float(actor["out_of_bounds_seconds"]))
                distance = info.get("target_distance")
                if distance is not None:
                    distance_history.append(float(distance))
                action_cont_history.append(action_cont.copy())
                action_bin_history.append(action_bin.copy())
                action_bin_prob_history.append(action_bin_prob.copy())
                if bool(action_bin[1] >= 0.5):
                    fired_step_count += 1
                if bool(action_bin[2] >= 0.5):
                    repair_step_count += 1
                if bool(actor["gun_overheated"]):
                    gun_overheated_step_count += 1
                if step_index % args.trace_stride == 0 and len(trace) < args.trace_limit:
                    trace.append(
                        {
                            "step_index": step_index,
                            "tick": int(info["tick"]),
                            "sim_time_seconds": float(info["sim_time_seconds"]),
                            "target_distance": distance,
                            "self_out_of_bounds_seconds": float(actor["out_of_bounds_seconds"]),
                            "self_gun_heat": float(actor["gun_heat"]),
                            "self_gun_overheated": bool(actor["gun_overheated"]),
                            "action_cont": action_cont.tolist(),
                            "action_bin": action_bin.astype(float).tolist(),
                            "action_bin_prob": action_bin_prob.tolist(),
                        }
                    )
                if terminated or truncated:
                    break

            self_state = last_info["aircraft_by_role"][args.ego_role]
            enemy_state = last_info["aircraft_by_role"]["fighter2" if args.ego_role == "fighter1" else "fighter1"]
            action_cont_array = np.asarray(action_cont_history, dtype=np.float32) if action_cont_history else np.zeros((0, 4), dtype=np.float32)
            action_bin_array = np.asarray(action_bin_history, dtype=np.float32) if action_bin_history else np.zeros((0, 3), dtype=np.float32)
            action_bin_prob_array = np.asarray(action_bin_prob_history, dtype=np.float32) if action_bin_prob_history else np.zeros((0, 3), dtype=np.float32)
            if action_cont_array.size > 0:
                mean_action_cont = action_cont_array.mean(axis=0)
                mean_abs_action_cont = np.abs(action_cont_array).mean(axis=0)
                std_action_cont = action_cont_array.std(axis=0)
                nontrivial_control_rate = float((np.max(np.abs(action_cont_array), axis=1) >= 0.1).mean())
            else:
                mean_action_cont = np.zeros((4,), dtype=np.float32)
                mean_abs_action_cont = np.zeros((4,), dtype=np.float32)
                std_action_cont = np.zeros((4,), dtype=np.float32)
                nontrivial_control_rate = 0.0
            if action_bin_prob_array.size > 0:
                binary_probability_mean = action_bin_prob_array.mean(axis=0)
                binary_on_rate = action_bin_array.mean(axis=0)
            else:
                binary_probability_mean = np.zeros((3,), dtype=np.float32)
                binary_on_rate = np.zeros((3,), dtype=np.float32)
            episode_summaries.append(
                EpisodeRolloutSummary(
                    episode_index=episode_index,
                    ego_role=args.ego_role,
                    steps=step_index + 1,
                    sim_time_seconds=float(last_info["sim_time_seconds"]),
                    terminated=terminated,
                    truncated=truncated,
                    winner=last_info.get("winner"),
                    self_destroyed=bool(self_state["destroyed"]),
                    enemy_destroyed=bool(enemy_state["destroyed"]),
                    max_out_of_bounds_seconds=max_oob,
                    fired_step_count=fired_step_count,
                    repair_step_count=repair_step_count,
                    gun_overheated_step_count=gun_overheated_step_count,
                    distance_start=distance_history[0] if distance_history else None,
                    distance_end=distance_history[-1] if distance_history else None,
                    distance_min=float(np.min(distance_history)) if distance_history else None,
                    distance_mean=float(np.mean(distance_history)) if distance_history else None,
                    mean_action_cont=mean_action_cont.tolist(),
                    mean_abs_action_cont=mean_abs_action_cont.tolist(),
                    std_action_cont=std_action_cont.tolist(),
                    binary_probability_mean=binary_probability_mean.tolist(),
                    binary_on_rate=binary_on_rate.tolist(),
                    nontrivial_control_rate=nontrivial_control_rate,
                    trace_sample_count=len(trace),
                )
            )
            episode_traces.append(
                {
                    "episode_index": episode_index,
                    "ego_role": args.ego_role,
                    "trace": trace,
                }
            )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "scene_name": args.scene_name,
        "ego_role": args.ego_role,
        "episodes": [asdict(item) for item in episode_summaries],
        "aggregate": {
            "episode_count": len(episode_summaries),
            "mean_steps": float(np.mean([item.steps for item in episode_summaries])) if episode_summaries else 0.0,
            "mean_sim_time_seconds": float(np.mean([item.sim_time_seconds for item in episode_summaries]))
            if episode_summaries
            else 0.0,
            "self_destroy_rate": float(np.mean([float(item.self_destroyed) for item in episode_summaries]))
            if episode_summaries
            else 0.0,
            "enemy_destroy_rate": float(np.mean([float(item.enemy_destroyed) for item in episode_summaries]))
            if episode_summaries
            else 0.0,
            "mean_max_out_of_bounds_seconds": float(
                np.mean([item.max_out_of_bounds_seconds for item in episode_summaries])
            )
            if episode_summaries
            else 0.0,
            "mean_fired_step_count": float(np.mean([item.fired_step_count for item in episode_summaries]))
            if episode_summaries
            else 0.0,
            "mean_repair_step_count": float(np.mean([item.repair_step_count for item in episode_summaries]))
            if episode_summaries
            else 0.0,
            "mean_distance_start": float(
                np.mean([item.distance_start for item in episode_summaries if item.distance_start is not None])
            )
            if any(item.distance_start is not None for item in episode_summaries)
            else 0.0,
            "mean_distance_end": float(
                np.mean([item.distance_end for item in episode_summaries if item.distance_end is not None])
            )
            if any(item.distance_end is not None for item in episode_summaries)
            else 0.0,
            "mean_nontrivial_control_rate": float(
                np.mean([item.nontrivial_control_rate for item in episode_summaries])
            )
            if episode_summaries
            else 0.0,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "trace.json").write_text(
        json.dumps({"episodes": episode_traces}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
