from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate grouped training/eval curves for a Part 2 training run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_eval_snapshots(eval_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not eval_dir.exists():
        return rows
    for path in sorted(eval_dir.glob("eval_step_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "path": path.name,
                "step": int(path.stem.split("_")[-1]),
                "stage": payload.get("stage", "unknown"),
                "dataset_root": payload.get("dataset_root"),
                "num_samples": payload.get("num_samples"),
                "metrics": metrics,
            }
        )
    return rows


def _render_multi_curve(
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    *,
    title: str,
    width: int = 960,
    height: int = 360,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    plot_left = 60
    plot_right = width - 20
    plot_top = 48
    plot_bottom = height - 56
    cv2.putText(
        canvas,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    if not x_values or not series:
        return canvas

    all_y = np.concatenate([np.asarray(values, dtype=np.float32) for _, values in series])
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0

    for frac in np.linspace(0.0, 1.0, 5):
        y = int(round(plot_bottom - frac * (plot_bottom - plot_top)))
        cv2.line(canvas, (plot_left, y), (plot_right, y), (225, 225, 225), 1, cv2.LINE_AA)
        value = y_min + frac * (y_max - y_min)
        cv2.putText(
            canvas,
            f"{value:.3f}",
            (8, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    colors = [
        (30, 90, 220),
        (220, 90, 30),
        (30, 170, 80),
        (180, 60, 180),
        (40, 160, 180),
        (160, 120, 30),
    ]
    for idx, (name, values) in enumerate(series):
        values_arr = np.asarray(values, dtype=np.float32)
        points = []
        for x, y in zip(x_values, values_arr, strict=True):
            px = plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)
            py = plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            points.append((int(round(px)), int(round(py))))
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                False,
                colors[idx % len(colors)],
                2,
                cv2.LINE_AA,
            )
        elif points:
            cv2.circle(canvas, points[0], 2, colors[idx % len(colors)], -1, cv2.LINE_AA)
        legend_y = plot_top + 18 * idx
        cv2.rectangle(
            canvas,
            (plot_right - 220, legend_y - 10),
            (plot_right - 204, legend_y + 2),
            colors[idx % len(colors)],
            -1,
        )
        cv2.putText(
            canvas,
            name,
            (plot_right - 196, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"step: {int(x_min)} -> {int(x_max)}",
        (plot_left, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _collect_train_series(train_rows: list[dict[str, Any]]) -> dict[str, tuple[list[float], dict[str, list[float]]]]:
    steps = [float(row["step"]) for row in train_rows if "step" in row]
    metrics: dict[str, list[float]] = {}
    for row in train_rows:
        if "step" not in row:
            continue
        for key, value in row.items():
            if key in {"timestamp_utc", "stage", "step"}:
                continue
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))
    return {"train": (steps, metrics)}


def _collect_eval_series(eval_rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, list[float]]]:
    steps: list[float] = []
    metrics: dict[str, list[float]] = {}
    for row in eval_rows:
        step = float(row["step"])
        steps.append(step)
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))
    return steps, metrics


def _write_group_curves(
    output_dir: Path,
    prefix: str,
    x_values: list[float],
    metrics: dict[str, list[float]],
    groups: dict[str, list[str]],
) -> list[str]:
    written: list[str] = []
    for group_name, metric_names in groups.items():
        present = [(name, metrics[name]) for name in metric_names if name in metrics]
        if not present:
            continue
        image = _render_multi_curve(x_values, present, title=f"{prefix}: {group_name}")
        filename = f"{prefix}_{group_name}.png"
        cv2.imwrite(str(output_dir / filename), image)
        written.append(filename)
    return written


def _build_index_html(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    train_images: list[str],
    eval_images: list[str],
) -> str:
    train_cards = "\n".join(
        f'<div class="card"><div>{name}</div><img src="{name}" alt="{name}"></div>'
        for name in train_images
    )
    eval_cards = "\n".join(
        f'<div class="card"><div>{name}</div><img src="{name}" alt="{name}"></div>'
        for name in eval_images
    )
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Training Run Visuals</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #111; background: #f7f7f7; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ margin: 0 0 20px; color: #444; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .card img {{ width: 100%; height: auto; display: block; background: #eee; }}
    pre {{ background: #111; color: #f2f2f2; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Training Run Visuals</h1>
  <div class="meta">run_dir={run_dir}</div>
  <h2>Train Curves</h2>
  <div class="grid">{train_cards or "<div class='card'>No train curves</div>"}</div>
  <h2>Eval Curves</h2>
  <div class="grid">{eval_cards or "<div class='card'>No eval curves</div>"}</div>
  <h2>Summary</h2>
  <pre>{json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)}</pre>
</body>
</html>
"""


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "visuals")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_jsonl(run_dir / "logs" / "train_log.jsonl")
    eval_rows = _load_eval_snapshots(run_dir / "eval")
    resolved_config_path = run_dir / "resolved_train_config.json"
    resolved_config = (
        json.loads(resolved_config_path.read_text(encoding="utf-8"))
        if resolved_config_path.exists()
        else None
    )
    train_steps, train_metrics = _collect_train_series(train_rows)["train"]
    eval_steps, eval_metrics = _collect_eval_series(eval_rows)

    train_groups = {
        "loss_main": [
            "total_loss",
            "weighted_vision_loss",
            "weighted_audio_loss",
            "weighted_evidence_loss",
            "vision_loss",
            "audio_loss",
            "evidence_loss",
        ],
        "loss_audio": [
            "audio_doa_loss",
            "audio_distance_loss",
            "audio_doa_conf_loss",
            "audio_dist_conf_loss",
        ],
        "loss_state": [
            "position_loss",
            "orientation_loss",
            "linear_velocity_loss",
            "angular_velocity_loss",
            "position_confidence_loss",
            "orientation_confidence_loss",
        ],
        "audio_runtime": [
            "doa_conf_mean",
            "dist_conf_mean",
            "audio_position_confidence_mean",
            "raw_audio_evidence_mean",
        ],
    }
    eval_groups = {
        "loss_main": [
            "total_loss",
            "weighted_vision_loss",
            "weighted_audio_loss",
            "weighted_evidence_loss",
            "vision_total_loss",
            "audio_total_loss",
            "evidence_total_loss",
        ],
        "loss_audio": [
            "audio_doa_loss",
            "audio_distance_loss",
            "audio_doa_conf_loss",
            "audio_dist_conf_loss",
            "audio_doa_conf_mse",
            "audio_dist_conf_mse",
        ],
        "audio_geometry": [
            "audio_position_l1",
            "audio_doa_angle_error",
            "audio_log_distance_error",
            "log_distance_mean",
        ],
        "audio_confidence": [
            "doa_conf_mean",
            "dist_conf_mean",
            "audio_position_confidence_mean",
            "audio_doa_conf_target_mean",
            "audio_dist_conf_target_mean",
        ],
        "state_primary": [
            "evidence_position_l1",
            "evidence_orientation_l1",
            "evidence_pos_conf_mse",
            "evidence_ori_conf_mse",
            "belief_position_l1",
            "belief_orientation_l1",
            "belief_linear_velocity_l1",
            "belief_angular_velocity_l1",
        ],
        "evidence_strength": [
            "raw_visual_evidence_mean",
            "raw_audio_evidence_mean",
            "a_energy_mean",
            "a_cue_mean",
            "visual_evidence_mean",
            "audio_evidence_mean",
            "track_confidence_mean",
        ],
    }

    train_images = _write_group_curves(output_dir, "train", train_steps, train_metrics, train_groups)
    eval_images = _write_group_curves(output_dir, "eval", eval_steps, eval_metrics, eval_groups)

    summary = {
        "run_dir": str(run_dir),
        "stage": resolved_config.get("stage") if resolved_config else None,
        "train_points": len(train_rows),
        "eval_points": len(eval_rows),
        "train_metrics": sorted(train_metrics.keys()),
        "eval_metrics": sorted(eval_metrics.keys()),
        "resolved_config_path": str(resolved_config_path) if resolved_config_path.exists() else None,
        "train_images": train_images,
        "eval_images": eval_images,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        _build_index_html(
            run_dir=run_dir,
            summary=summary,
            train_images=train_images,
            eval_images=eval_images,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
