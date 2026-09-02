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

from dfb_reinforcement_learning.data import BcDataset, load_bc_split
from dfb_reinforcement_learning.models import (
    StatelessHybridActorCritic,
    model_architecture_kwargs,
)
from dfb_reinforcement_learning.policy_assets import (
    PolicyDatasetContract,
    checkpoint_contract_metadata,
    checkpoint_model_hyperparameters,
    load_policy_dataset_contract,
    validate_policy_checkpoint_payload,
)
from dfb_reinforcement_learning.train.weighted_losses import weighted_mean


@dataclass(frozen=True)
class DistillMetrics:
    total_loss: float
    continuous_loss: float
    binary_loss: float
    value_loss: float
    sample_count: int
    sample_weight_sum: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a larger BC student from an existing teacher checkpoint.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--teacher-dropout", type=float, default=0.0)
    parser.add_argument("--continuous-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--disable-overlap-init",
        action="store_true",
        help="Do not copy overlapping teacher weights into the larger student before distillation.",
    )
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


def _load_teacher_arch(payload: dict[str, Any]) -> dict[str, Any]:
    return model_architecture_kwargs(payload["model_hyperparameters"])


def copy_overlapping_parameters(
    student: StatelessHybridActorCritic,
    teacher_state_dict: dict[str, torch.Tensor],
) -> int:
    copied = 0
    student_state = student.state_dict()
    patched_state: dict[str, torch.Tensor] = {}
    for name, target in student_state.items():
        source = teacher_state_dict.get(name)
        if source is None or source.dtype != target.dtype or source.ndim != target.ndim:
            patched_state[name] = target
            continue
        result = target.clone()
        slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target.shape))
        result[slices] = source[slices]
        patched_state[name] = result
        if result.numel() > 0:
            copied += int(np.prod([sl.stop - sl.start for sl in slices], dtype=np.int64))
    student.load_state_dict(patched_state, strict=True)
    return copied


def _compute_distill_losses(
    student: StatelessHybridActorCritic,
    teacher: StatelessHybridActorCritic,
    batch: dict[str, torch.Tensor],
    *,
    continuous_loss_weight: float,
    binary_loss_weight: float,
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_output = teacher(batch["obs"])
        teacher_bin_prob = torch.sigmoid(teacher_output.action_bin_logits)

    student_output = student(batch["obs"])
    sample_weight = batch["sample_weight"]
    cont_loss = weighted_mean(
        nn.functional.mse_loss(
            student_output.action_cont_mean,
            teacher_output.action_cont_mean,
            reduction="none",
        ).mean(dim=1),
        sample_weight,
    )
    bin_loss = weighted_mean(
        nn.functional.binary_cross_entropy_with_logits(
            student_output.action_bin_logits,
            teacher_bin_prob,
            reduction="none",
        ).mean(dim=1),
        sample_weight,
    )
    value_loss = weighted_mean(
        nn.functional.mse_loss(
            student_output.value,
            teacher_output.value,
            reduction="none",
        ),
        sample_weight,
    )
    total = (
        continuous_loss_weight * cont_loss
        + binary_loss_weight * bin_loss
        + value_loss_weight * value_loss
    )
    return total, cont_loss, bin_loss, value_loss


@torch.no_grad()
def _evaluate(
    student: StatelessHybridActorCritic,
    teacher: StatelessHybridActorCritic,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    continuous_loss_weight: float,
    binary_loss_weight: float,
    value_loss_weight: float,
) -> DistillMetrics:
    student.eval()
    teacher.eval()
    total_loss = 0.0
    continuous_loss = 0.0
    binary_loss = 0.0
    value_loss = 0.0
    sample_count = 0
    sample_weight_sum = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        total, cont, binary, value = _compute_distill_losses(
            student,
            teacher,
            batch,
            continuous_loss_weight=continuous_loss_weight,
            binary_loss_weight=binary_loss_weight,
            value_loss_weight=value_loss_weight,
        )
        batch_size = int(batch["obs"].shape[0])
        batch_weight_sum = float(batch["sample_weight"].sum().item())
        total_loss += float(total.item()) * batch_weight_sum
        continuous_loss += float(cont.item()) * batch_weight_sum
        binary_loss += float(binary.item()) * batch_weight_sum
        value_loss += float(value.item()) * batch_weight_sum
        sample_count += batch_size
        sample_weight_sum += batch_weight_sum
    if sample_count == 0:
        return DistillMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)
    inv = 1.0 / sample_weight_sum
    return DistillMetrics(
        total_loss=total_loss * inv,
        continuous_loss=continuous_loss * inv,
        binary_loss=binary_loss * inv,
        value_loss=value_loss * inv,
        sample_count=sample_count,
        sample_weight_sum=sample_weight_sum,
    )


def _save_checkpoint(
    *,
    path: Path,
    student: StatelessHybridActorCritic,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    train_metrics: DistillMetrics,
    val_metrics: DistillMetrics | None,
    teacher_checkpoint: str,
    copied_parameter_count: int,
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
        "training_stage": "behavior_cloning_distillation",
        "update_index": epoch,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "teacher_checkpoint": str(Path(teacher_checkpoint).resolve()),
        "copied_parameter_count": copied_parameter_count,
        "train_metrics": asdict(train_metrics),
        "val_metrics": asdict(val_metrics) if val_metrics is not None else None,
    }
    torch.save(payload, path)


def _resolve_monitored_loss(payload: dict[str, Any]) -> float:
    val_metrics = payload.get("val_metrics")
    if val_metrics is not None:
        return float(val_metrics["total_loss"])
    train_metrics = payload.get("train_metrics")
    if train_metrics is not None:
        return float(train_metrics["total_loss"])
    return float("inf")


def main() -> None:
    args = _parse_args()
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

    teacher_payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    validate_policy_checkpoint_payload(
        teacher_payload,
        dataset=dataset_contract,
        context="distillation teacher checkpoint",
    )
    teacher_architecture = _load_teacher_arch(teacher_payload)
    if args.teacher_dropout >= 0.0:
        teacher_architecture["dropout"] = args.teacher_dropout
    teacher = StatelessHybridActorCritic(
        obs_dim=normalizer.obs_dim,
        **teacher_architecture,
    ).to(device)
    teacher.load_state_dict(teacher_payload["model_state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = StatelessHybridActorCritic(
        obs_dim=normalizer.obs_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    copied_parameter_count = 0
    if not args.disable_overlap_init:
        copied_parameter_count = copy_overlapping_parameters(student, teacher.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        validate_policy_checkpoint_payload(
            payload,
            dataset=dataset_contract,
            context="distillation resume checkpoint",
        )
        student.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        copied_parameter_count = int(payload.get("copied_parameter_count", copied_parameter_count))
        best_path = checkpoints_dir / "best.pt"
        if best_path.exists():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            validate_policy_checkpoint_payload(
                best_payload,
                dataset=dataset_contract,
                context="distillation best checkpoint",
            )
            best_val_loss = _resolve_monitored_loss(best_payload)
        else:
            best_val_loss = _resolve_monitored_loss(payload)
    target_epoch = start_epoch + args.epochs if args.resume else args.epochs

    summary_path = output_dir / "distill_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_root": str(Path(args.dataset_root).resolve()),
                "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
                "obs_dim": normalizer.obs_dim,
                "train_rows": train_split.size,
                "val_rows": val_split.size,
                "copied_parameter_count": copied_parameter_count,
                "resume_from": str(Path(args.resume).resolve()) if args.resume else None,
                "start_epoch": start_epoch,
                "target_epoch_exclusive": target_epoch,
                "student_args": vars(args),
                "teacher_arch": {
                    "hidden_dim": teacher_hidden_dim,
                    "num_layers": teacher_num_layers,
                    "dropout": teacher_dropout,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_log_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, target_epoch):
        student.train()
        train_total = 0.0
        train_cont = 0.0
        train_bin = 0.0
        train_value = 0.0
        train_count = 0
        train_weight_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            total, cont, binary, value = _compute_distill_losses(
                student,
                teacher,
                batch,
                continuous_loss_weight=args.continuous_loss_weight,
                binary_loss_weight=args.binary_loss_weight,
                value_loss_weight=args.value_loss_weight,
            )
            total.backward()
            optimizer.step()
            batch_size = int(batch["obs"].shape[0])
            batch_weight_sum = float(batch["sample_weight"].sum().item())
            train_total += float(total.item()) * batch_weight_sum
            train_cont += float(cont.item()) * batch_weight_sum
            train_bin += float(binary.item()) * batch_weight_sum
            train_value += float(value.item()) * batch_weight_sum
            train_count += batch_size
            train_weight_sum += batch_weight_sum
            global_step += batch_size

        inv = 1.0 / max(train_weight_sum, 1.0)
        train_metrics = DistillMetrics(
            total_loss=train_total * inv,
            continuous_loss=train_cont * inv,
            binary_loss=train_bin * inv,
            value_loss=train_value * inv,
            sample_count=train_count,
            sample_weight_sum=train_weight_sum,
        )
        val_metrics = _evaluate(
            student,
            teacher,
            val_loader,
            device=device,
            continuous_loss_weight=args.continuous_loss_weight,
            binary_loss_weight=args.binary_loss_weight,
            value_loss_weight=args.value_loss_weight,
        )
        latest_path = checkpoints_dir / "latest.pt"
        _save_checkpoint(
            path=latest_path,
            student=student,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            teacher_checkpoint=args.teacher_checkpoint,
            copied_parameter_count=copied_parameter_count,
            dataset_contract=dataset_contract,
        )
        if val_metrics.total_loss <= best_val_loss:
            best_val_loss = val_metrics.total_loss
            _save_checkpoint(
                path=checkpoints_dir / "best.pt",
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                teacher_checkpoint=args.teacher_checkpoint,
                copied_parameter_count=copied_parameter_count,
                dataset_contract=dataset_contract,
            )

        (eval_dir / f"epoch_{epoch:04d}.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_metrics": asdict(train_metrics),
                    "val_metrics": asdict(val_metrics),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_total": train_metrics.total_loss,
                        "train_cont": train_metrics.continuous_loss,
                        "train_bin": train_metrics.binary_loss,
                        "train_value": train_metrics.value_loss,
                        "val_total": val_metrics.total_loss,
                        "val_cont": val_metrics.continuous_loss,
                        "val_bin": val_metrics.binary_loss,
                        "val_value": val_metrics.value_loss,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            "epoch="
            f"{epoch:04d} step={global_step:06d} "
            f"train_total={train_metrics.total_loss:.6f} "
            f"train_cont={train_metrics.continuous_loss:.6f} "
            f"train_bin={train_metrics.binary_loss:.6f} "
            f"train_value={train_metrics.value_loss:.6f} "
            f"val_total={val_metrics.total_loss:.6f} "
            f"val_cont={val_metrics.continuous_loss:.6f} "
            f"val_bin={val_metrics.binary_loss:.6f} "
            f"val_value={val_metrics.value_loss:.6f}"
        )


if __name__ == "__main__":
    main()
