from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dfb_reinforcement_learning.data import load_bc_split
from dfb_reinforcement_learning.models import (
    StatelessHybridActorCritic,
    measure_policy_output_migration_error,
    migrate_policy_parameters,
    model_architecture_kwargs,
)
from dfb_reinforcement_learning.policy_assets import (
    checkpoint_model_hyperparameters,
    load_policy_dataset_contract,
    validate_policy_checkpoint_payload,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate a policy checkpoint to a wider and/or identity-deepened architecture."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--target-hidden-dim", type=int, default=None)
    parser.add_argument("--shared-extension-blocks", type=int, default=None)
    parser.add_argument("--actor-extension-blocks", type=int, default=None)
    parser.add_argument("--critic-extension-blocks", type=int, default=None)
    parser.add_argument("--verification-split", default="val")
    parser.add_argument("--verification-samples", type=int, default=10_000)
    parser.add_argument("--verification-tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_extension_count(
    requested: int | None,
    *,
    source: int,
    name: str,
) -> int:
    target = source if requested is None else requested
    if target < source:
        raise ValueError(f"{name} cannot decrease from {source} to {target}")
    return target


def _checkpoint_args(payload: dict[str, Any]) -> dict[str, Any]:
    raw_args = payload.get("args", {})
    if isinstance(raw_args, argparse.Namespace):
        return dict(vars(raw_args))
    if isinstance(raw_args, dict):
        return dict(raw_args)
    return {}


def _verification_observations(
    *,
    dataset_root: Path,
    split: str,
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    if sample_count <= 0:
        raise ValueError("--verification-samples must be positive")
    split_data, normalizer = load_bc_split(dataset_root, split)
    if split_data.size <= 0:
        raise ValueError(f"verification split {split!r} has no samples")
    actual_count = min(sample_count, split_data.size)
    indices = np.linspace(0, split_data.size - 1, num=actual_count, dtype=np.int64)
    normalized = normalizer.normalize_np(split_data.obs[indices]).astype(np.float32, copy=False)
    return torch.from_numpy(normalized).to(device)


def _migrated_payload(
    *,
    source_payload: dict[str, Any],
    target_model: StatelessHybridActorCritic,
    target_architecture: dict[str, Any],
    source_checkpoint: Path,
    output_checkpoint: Path,
    source_parameter_count: int,
    target_parameter_count: int,
    verification_error: dict[str, float],
    verification_sample_count: int,
) -> dict[str, Any]:
    source_hyperparameters = source_payload["model_hyperparameters"]
    migrated = dict(source_payload)
    migrated.pop("optimizer_state_dict", None)
    migrated["training_stage"] = "architecture_migration"
    migrated["model_hyperparameters"] = checkpoint_model_hyperparameters(
        hidden_dim=int(target_architecture["hidden_dim"]),
        num_layers=int(target_architecture["num_layers"]),
        dropout=float(target_architecture["dropout"]),
        shared_extension_blocks=int(target_architecture["shared_extension_blocks"]),
        actor_extension_blocks=int(target_architecture["actor_extension_blocks"]),
        critic_extension_blocks=int(target_architecture["critic_extension_blocks"]),
        continuous_action_std=source_hyperparameters.get("continuous_action_std"),
        popart_beta=source_hyperparameters.get("popart_beta"),
        popart_min_std=source_hyperparameters.get("popart_min_std"),
    )
    migrated["model_state_dict"] = target_model.state_dict()
    checkpoint_args = _checkpoint_args(source_payload)
    checkpoint_args.update(target_architecture)
    checkpoint_args["resume"] = None
    checkpoint_args["init_checkpoint"] = str(source_checkpoint)
    migrated["args"] = checkpoint_args
    migrated["architecture_migration"] = {
        "method": "identity_gated_depth_and_integer_net2wider_v1",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "source_architecture": model_architecture_kwargs(source_hyperparameters),
        "target_architecture": target_architecture,
        "source_parameter_count": source_parameter_count,
        "target_parameter_count": target_parameter_count,
        "optimizer_state_migrated": False,
        "verification_sample_count": verification_sample_count,
        "verification_error": verification_error,
    }
    return migrated


def main() -> None:
    args = _parse_args()
    if args.verification_tolerance <= 0.0:
        raise ValueError("--verification-tolerance must be positive")
    source_checkpoint = Path(args.source_checkpoint).expanduser().resolve()
    output_checkpoint = Path(args.output_checkpoint).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_checkpoint}")
    if output_checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"output checkpoint already exists: {output_checkpoint}")

    dataset_contract = load_policy_dataset_contract(args.dataset_root)
    device = torch.device(args.device)
    source_payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
    validate_policy_checkpoint_payload(
        source_payload,
        dataset=dataset_contract,
        context="architecture migration source checkpoint",
    )
    source_architecture = model_architecture_kwargs(source_payload["model_hyperparameters"])
    source_hidden_dim = int(source_architecture["hidden_dim"])
    target_hidden_dim = (
        source_hidden_dim if args.target_hidden_dim is None else int(args.target_hidden_dim)
    )
    if target_hidden_dim < source_hidden_dim or target_hidden_dim % source_hidden_dim != 0:
        raise ValueError("--target-hidden-dim must be an integer multiple of the source width")
    target_architecture = {
        **source_architecture,
        "hidden_dim": target_hidden_dim,
        "shared_extension_blocks": _resolved_extension_count(
            args.shared_extension_blocks,
            source=int(source_architecture["shared_extension_blocks"]),
            name="shared extension blocks",
        ),
        "actor_extension_blocks": _resolved_extension_count(
            args.actor_extension_blocks,
            source=int(source_architecture["actor_extension_blocks"]),
            name="actor extension blocks",
        ),
        "critic_extension_blocks": _resolved_extension_count(
            args.critic_extension_blocks,
            source=int(source_architecture["critic_extension_blocks"]),
            name="critic extension blocks",
        ),
    }

    torch.manual_seed(args.seed)
    source_model = StatelessHybridActorCritic(
        obs_dim=dataset_contract.obs_dim,
        **source_architecture,
    ).to(device)
    source_model.load_state_dict(source_payload["model_state_dict"], strict=True)
    target_model = StatelessHybridActorCritic(
        obs_dim=dataset_contract.obs_dim,
        **target_architecture,
    ).to(device)
    source_extension_counts = {
        key: int(source_architecture[key])
        for key in (
            "shared_extension_blocks",
            "actor_extension_blocks",
            "critic_extension_blocks",
        )
    }
    migrate_policy_parameters(
        source=source_model,
        target=target_model,
        source_hidden_dim=source_hidden_dim,
        target_hidden_dim=target_hidden_dim,
        source_extension_counts=source_extension_counts,
    )

    observations = _verification_observations(
        dataset_root=Path(args.dataset_root).expanduser().resolve(),
        split=args.verification_split,
        sample_count=args.verification_samples,
        device=device,
    )
    error = measure_policy_output_migration_error(
        source=source_model,
        target=target_model,
        observations=observations,
    )
    if error.maximum > args.verification_tolerance:
        raise RuntimeError(
            f"migration verification error {error.maximum:.9g} exceeds "
            f"tolerance {args.verification_tolerance:.9g}"
        )

    verification_error = asdict(error)
    source_parameter_count = sum(parameter.numel() for parameter in source_model.parameters())
    target_parameter_count = sum(parameter.numel() for parameter in target_model.parameters())
    payload = _migrated_payload(
        source_payload=source_payload,
        target_model=target_model,
        target_architecture=target_architecture,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        source_parameter_count=source_parameter_count,
        target_parameter_count=target_parameter_count,
        verification_error=verification_error,
        verification_sample_count=int(observations.shape[0]),
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_checkpoint.with_name(f".{output_checkpoint.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(output_checkpoint)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(
        json.dumps(
            {
                "output_checkpoint": str(output_checkpoint),
                "source_architecture": source_architecture,
                "target_architecture": target_architecture,
                "source_parameter_count": source_parameter_count,
                "target_parameter_count": target_parameter_count,
                "verification_sample_count": int(observations.shape[0]),
                "verification_error": verification_error,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
