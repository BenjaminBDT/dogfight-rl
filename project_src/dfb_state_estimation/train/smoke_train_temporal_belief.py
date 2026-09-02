from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dfb_state_estimation.train.config import load_train_config
from dfb_state_estimation.train.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick smoke wrapper around the unified trainer for temporal belief stage."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--window-index", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dfb_state_estimation/train/default_train_config.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/dfb_state_estimation/train/smoke_temporal_belief"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_train_config(args.config)
    config = replace(
        config,
        dataset_root=args.dataset_root.resolve(),
        output_root=args.output_root.resolve(),
        seed=args.seed,
        optimizer=replace(config.optimizer, lr=args.lr),
        temporal_belief=replace(
            config.temporal_belief,
            window_index=args.window_index,
            max_steps=args.max_steps,
        ),
    )
    summary = run_training(
        config,
        stage_override="temporal_belief",
        num_steps_override=args.steps,
        output_root_override=config.output_root,
    )
    first_loss = summary["first_loss"]
    last_loss = summary["last_loss"]
    print(f"first_loss={first_loss:.6f}")
    print(f"last_loss={last_loss:.6f}")
    if last_loss >= first_loss:
        raise SystemExit(
            f"smoke train did not improve: first_loss={first_loss:.6f}, last_loss={last_loss:.6f}"
        )
    print("loss_decreased: ok")


if __name__ == "__main__":
    main()
