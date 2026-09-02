from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.actions import ActionAdapter
from dfb_reinforcement_learning.data import ObservationNormalizer
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.obs.policy_adapter import PolicyObservationAdapter
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    load_and_validate_policy_dataset,
)
from dfb_reinforcement_learning.policy_inference import deterministic_policy_output
from dfb_reinforcement_learning.policy_inference import load_policy_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BC model pilot over stdio.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dataset_root(
    checkpoint_path: str | Path,
    dataset_root: str | None,
) -> Path:
    if dataset_root:
        return Path(dataset_root)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args")
    if isinstance(args, dict):
        resolved = args.get("dataset_root")
        if isinstance(resolved, str) and resolved:
            return Path(resolved)
    raise ValueError(
        "dataset root is missing; pass --dataset-root or use a checkpoint whose args contain dataset_root"
    )


def _load_model(
    checkpoint_path: str | Path,
    normalizer: ObservationNormalizer,
    dataset_contract: PolicyDatasetContract,
    device: torch.device,
) -> StatelessHybridActorCritic:
    return load_policy_model(
        checkpoint_path,
        normalizer=normalizer,
        dataset_contract=dataset_contract,
        device=device,
        context="live checkpoint",
    )


def _predict_action_response(
    *,
    model: StatelessHybridActorCritic,
    normalizer: ObservationNormalizer,
    obs_adapter: PolicyObservationAdapter,
    state: dict[str, Any],
    role: str,
    episode_start_sim_time_seconds: float,
    device: torch.device,
    binary_threshold: float,
) -> dict[str, Any]:
    obs = obs_adapter.build(
        state,
        role,
        episode_start_sim_time_seconds=episode_start_sim_time_seconds,
    )["vector"]
    output = deterministic_policy_output(
        model=model,
        normalizer=normalizer,
        obs=obs,
        device=device,
    )
    action_cont = output.action_cont
    action_bin_prob = output.action_bin_prob
    action = ActionAdapter.from_arrays(action_cont, action_bin_prob, binary_threshold=binary_threshold)
    return {
        "action": {
            "throttle": action.throttle,
            "brake": action.brake,
            "pitch": action.pitch,
            "roll": action.roll,
            "yaw": action.yaw,
            "fire_gun": action.fire_gun,
            "repair": action.repair,
        },
        "action_cont": action_cont.tolist(),
        "action_bin_prob": action_bin_prob.tolist(),
        "error": None,
    }


def _neutral_response(error: str) -> dict[str, Any]:
    neutral = ActionAdapter.neutral()
    return {
        "action": {
            "throttle": neutral.throttle,
            "brake": neutral.brake,
            "pitch": neutral.pitch,
            "roll": neutral.roll,
            "yaw": neutral.yaw,
            "fire_gun": neutral.fire_gun,
            "repair": neutral.repair,
        },
        "action_cont": [0.0, 0.0, 0.0, 0.0],
        "action_bin_prob": [0.0, 0.0, 0.0],
        "error": error,
    }


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    dataset_root = _resolve_dataset_root(args.checkpoint, args.dataset_root)
    dataset_contract, normalizer_payload = load_and_validate_policy_dataset(dataset_root)
    normalizer = ObservationNormalizer.from_payload(
        normalizer_payload,
        dataset=dataset_contract,
    )
    model = _load_model(args.checkpoint, normalizer, dataset_contract, device)
    obs_adapter = PolicyObservationAdapter()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            state = payload["state"]
            role = str(payload["role"])
            episode_start_sim_time_seconds = float(payload["episode_start_sim_time_seconds"])
            response = _predict_action_response(
                model=model,
                normalizer=normalizer,
                obs_adapter=obs_adapter,
                state=state,
                role=role,
                episode_start_sim_time_seconds=episode_start_sim_time_seconds,
                device=device,
                binary_threshold=args.binary_threshold,
            )
        except Exception as exc:  # pragma: no cover - live guard path
            print(json.dumps(_neutral_response(str(exc))), flush=True)
            continue
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
