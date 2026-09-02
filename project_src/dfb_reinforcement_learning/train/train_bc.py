from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dfb_reinforcement_learning.data import (
    BcDataset,
    filter_bc_split_by_roles,
    load_bc_split,
    with_episode_balanced_weights,
)
from dfb_reinforcement_learning.models import StatelessHybridActorCritic
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    load_policy_dataset_contract,
    validate_policy_checkpoint_initialization_payload,
    validate_policy_checkpoint_payload,
)
from dfb_reinforcement_learning.train.weighted_losses import weighted_mean


@dataclass(frozen=True)
class BcMetrics:
    total_loss: float
    continuous_loss: float
    binary_loss: float
    binary_accuracy: float
    fire_positive_rate: float
    repair_positive_rate: float
    brake_positive_rate: float
    sample_count: int
    sample_weight_sum: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC under the active Part 3 policy contract.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--include-roles",
        nargs="+",
        choices=("fighter1", "fighter2"),
        default=("fighter1", "fighter2"),
        help="Only use action labels from the selected recorded roles.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--continuous-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-monitor",
        choices=("total_loss", "continuous_loss", "binary_loss"),
        default="total_loss",
        help="Validation metric used to select best.pt.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save numbered checkpoints every N epochs; latest and best are unaffected.",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("step", "episode"),
        default="step",
        help="Apply source weights per step or equalize total weight per episode-role view.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _compute_batch_losses(
    model: StatelessHybridActorCritic,
    batch: dict[str, torch.Tensor],
    *,
    continuous_loss_weight: float,
    binary_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(batch["obs"])
    sample_weight = batch["sample_weight"]
    cont_per_sample = nn.functional.smooth_l1_loss(
        output.action_cont_mean,
        batch["action_cont"],
        reduction="none",
    ).mean(dim=1)
    bin_per_sample = nn.functional.binary_cross_entropy_with_logits(
        output.action_bin_logits,
        batch["action_bin"],
        reduction="none",
    ).mean(dim=1)
    cont_loss = weighted_mean(cont_per_sample, sample_weight)
    bin_loss = weighted_mean(bin_per_sample, sample_weight)
    total = continuous_loss_weight * cont_loss + binary_loss_weight * bin_loss
    action_bin_probs = torch.sigmoid(output.action_bin_logits)
    pred_bin = (action_bin_probs >= 0.5).to(batch["action_bin"].dtype)
    bin_acc = weighted_mean(
        (pred_bin == batch["action_bin"]).to(torch.float32).mean(dim=1),
        sample_weight,
    )
    return total, cont_loss, bin_loss, bin_acc, action_bin_probs


def _positive_action_rates(
    action_bin_probs: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[float, float, float]:
    predictions = (action_bin_probs >= 0.5).to(torch.float32)
    brake = float(weighted_mean(predictions[:, 0], sample_weight).item())
    fire = float(weighted_mean(predictions[:, 1], sample_weight).item())
    repair = float(weighted_mean(predictions[:, 2], sample_weight).item())
    return brake, fire, repair


@torch.no_grad()
def _evaluate(
    model: StatelessHybridActorCritic,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    continuous_loss_weight: float,
    binary_loss_weight: float,
) -> BcMetrics:
    model.eval()
    total_loss = 0.0
    continuous_loss = 0.0
    binary_loss = 0.0
    binary_accuracy = 0.0
    fire_rate = 0.0
    repair_rate = 0.0
    brake_rate = 0.0
    sample_count = 0
    sample_weight_sum = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        total, cont, binary, bin_acc, probs = _compute_batch_losses(
            model,
            batch,
            continuous_loss_weight=continuous_loss_weight,
            binary_loss_weight=binary_loss_weight,
        )
        batch_size = int(batch["obs"].shape[0])
        batch_weight_sum = float(batch["sample_weight"].sum().item())
        total_loss += float(total.item()) * batch_weight_sum
        continuous_loss += float(cont.item()) * batch_weight_sum
        binary_loss += float(binary.item()) * batch_weight_sum
        binary_accuracy += float(bin_acc.item()) * batch_weight_sum
        brake, fire, repair = _positive_action_rates(probs, batch["sample_weight"])
        brake_rate += brake * batch_weight_sum
        fire_rate += fire * batch_weight_sum
        repair_rate += repair * batch_weight_sum
        sample_count += batch_size
        sample_weight_sum += batch_weight_sum
    if sample_count == 0:
        return BcMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    inv = 1.0 / sample_weight_sum
    return BcMetrics(
        total_loss=total_loss * inv,
        continuous_loss=continuous_loss * inv,
        binary_loss=binary_loss * inv,
        binary_accuracy=binary_accuracy * inv,
        fire_positive_rate=fire_rate * inv,
        repair_positive_rate=repair_rate * inv,
        brake_positive_rate=brake_rate * inv,
        sample_count=sample_count,
        sample_weight_sum=sample_weight_sum,
    )


def _save_checkpoint(
    *,
    path: Path,
    model: StatelessHybridActorCritic,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    train_metrics: BcMetrics,
    val_metrics: BcMetrics | None,
    dataset_contract: PolicyDatasetContract,
) -> None:
    payload = {
        **checkpoint_contract_metadata(dataset_contract),
        "model_hyperparameters": checkpoint_model_hyperparameters(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            continuous_action_std=None,
            popart_beta=None,
            popart_min_std=None,
        ),
        "training_stage": "behavior_cloning",
        "update_index": epoch,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "train_metrics": asdict(train_metrics),
        "val_metrics": asdict(val_metrics) if val_metrics is not None else None,
    }
    torch.save(payload, path)


def _monitored_bc_loss(metrics: BcMetrics, monitor: str) -> float:
    if monitor not in {"total_loss", "continuous_loss", "binary_loss"}:
        raise ValueError(f"unsupported BC checkpoint monitor: {monitor!r}")
    return float(getattr(metrics, monitor))


def _resolve_monitored_bc_loss(payload: dict[str, Any], *, monitor: str) -> float:
    val_metrics = payload.get("val_metrics")
    if val_metrics is not None:
        return float(val_metrics[monitor])
    train_metrics = payload.get("train_metrics")
    if train_metrics is not None:
        return float(train_metrics[monitor])
    return float("inf")


def main() -> None:
    args = _parse_args()
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    if args.continuous_loss_weight <= 0.0 or args.binary_loss_weight <= 0.0:
        raise ValueError("BC loss weights must be positive")
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "eval"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_contract = load_policy_dataset_contract(args.dataset_root)
    train_split, normalizer = load_bc_split(args.dataset_root, args.train_split)
    val_split, _ = load_bc_split(args.dataset_root, args.val_split)
    train_split = filter_bc_split_by_roles(train_split, args.include_roles)
    val_split = filter_bc_split_by_roles(val_split, args.include_roles)
    if args.loss_weighting == "episode":
        train_split = with_episode_balanced_weights(train_split)
        val_split = with_episode_balanced_weights(val_split)
    train_dataset = BcDataset(train_split, normalizer=normalizer, normalize_obs=True)
    val_dataset = BcDataset(val_split, normalizer=normalizer, normalize_obs=True)
    if len(train_dataset) == 0:
        raise ValueError("train split is empty")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = StatelessHybridActorCritic(
        obs_dim=normalizer.obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        validate_policy_checkpoint_payload(
            payload,
            dataset=dataset_contract,
            context="BC resume checkpoint",
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        best_path = checkpoints_dir / "best.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            validate_policy_checkpoint_payload(
                best_payload,
                dataset=dataset_contract,
                context="BC best checkpoint",
            )
            best_val_loss = _resolve_monitored_bc_loss(
                best_payload,
                monitor=args.checkpoint_monitor,
            )
        else:
            best_val_loss = _resolve_monitored_bc_loss(
                payload,
                monitor=args.checkpoint_monitor,
            )
    elif args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        validate_policy_checkpoint_initialization_payload(
            payload,
            dataset=dataset_contract,
            context="BC init checkpoint",
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
    target_epoch = start_epoch + args.epochs if args.resume else args.epochs

    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "obs_dim": normalizer.obs_dim,
                **checkpoint_contract_metadata(dataset_contract),
                "train_rows": train_split.size,
                "val_rows": val_split.size,
                "init_checkpoint": str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
                "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
                "start_epoch": start_epoch,
                "target_epoch_exclusive": target_epoch,
                "args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_log_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, target_epoch):
        model.train()
        train_total = 0.0
        train_cont = 0.0
        train_bin = 0.0
        train_bin_acc = 0.0
        train_brake_rate = 0.0
        train_fire_rate = 0.0
        train_repair_rate = 0.0
        train_count = 0
        train_weight_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            total, cont, binary, bin_acc, probs = _compute_batch_losses(
                model,
                batch,
                continuous_loss_weight=args.continuous_loss_weight,
                binary_loss_weight=args.binary_loss_weight,
            )
            total.backward()
            optimizer.step()
            batch_size = int(batch["obs"].shape[0])
            batch_weight_sum = float(batch["sample_weight"].sum().item())
            brake, fire, repair = _positive_action_rates(probs, batch["sample_weight"])
            train_total += float(total.item()) * batch_weight_sum
            train_cont += float(cont.item()) * batch_weight_sum
            train_bin += float(binary.item()) * batch_weight_sum
            train_bin_acc += float(bin_acc.item()) * batch_weight_sum
            train_brake_rate += brake * batch_weight_sum
            train_fire_rate += fire * batch_weight_sum
            train_repair_rate += repair * batch_weight_sum
            train_count += batch_size
            train_weight_sum += batch_weight_sum
            global_step += 1
        train_metrics = BcMetrics(
            total_loss=train_total / train_weight_sum,
            continuous_loss=train_cont / train_weight_sum,
            binary_loss=train_bin / train_weight_sum,
            binary_accuracy=train_bin_acc / train_weight_sum,
            fire_positive_rate=train_fire_rate / train_weight_sum,
            repair_positive_rate=train_repair_rate / train_weight_sum,
            brake_positive_rate=train_brake_rate / train_weight_sum,
            sample_count=train_count,
            sample_weight_sum=train_weight_sum,
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            continuous_loss_weight=args.continuous_loss_weight,
            binary_loss_weight=args.binary_loss_weight,
        ) if len(val_dataset) > 0 else None

        metrics_payload: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics) if val_metrics is not None else None,
        }
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_payload, ensure_ascii=False) + "\n")
        (eval_dir / f"epoch_{epoch:04d}.json").write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == target_epoch:
            _save_checkpoint(
                path=checkpoints_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset_contract=dataset_contract,
            )
        _save_checkpoint(
            path=checkpoints_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            dataset_contract=dataset_contract,
        )
        monitored_metrics = val_metrics if val_metrics is not None else train_metrics
        monitored_val = _monitored_bc_loss(
            monitored_metrics,
            args.checkpoint_monitor,
        )
        if monitored_val < best_val_loss:
            best_val_loss = monitored_val
            _save_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset_contract=dataset_contract,
            )
        print(
            f"epoch={epoch:04d} step={global_step:06d} "
            f"train_total={train_metrics.total_loss:.6f} "
            f"train_cont={train_metrics.continuous_loss:.6f} "
            f"train_bin={train_metrics.binary_loss:.6f} "
            f"train_bin_acc={train_metrics.binary_accuracy:.4f} "
            f"train_fire_rate={train_metrics.fire_positive_rate:.4f} "
            f"train_repair_rate={train_metrics.repair_positive_rate:.4f}"
            + (
                f" val_total={val_metrics.total_loss:.6f}"
                f" val_cont={val_metrics.continuous_loss:.6f}"
                f" val_bin_acc={val_metrics.binary_accuracy:.4f}"
                f" val_fire_rate={val_metrics.fire_positive_rate:.4f}"
                f" val_repair_rate={val_metrics.repair_positive_rate:.4f}"
                if val_metrics is not None
                else ""
            )
            + f" monitor={args.checkpoint_monitor}:{monitored_val:.6f}"
        )


if __name__ == "__main__":
    main()
