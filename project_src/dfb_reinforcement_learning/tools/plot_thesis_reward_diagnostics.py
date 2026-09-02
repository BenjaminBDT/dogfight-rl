from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dfb_reinforcement_learning.rewards.policy_reward import (
    AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS,
)

SVG_FONT_FAMILY = '"Noto Sans CJK SC", "Microsoft YaHei", "WenQuanYi Zen Hei", "Source Han Sans SC", sans-serif'


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _pad_range(values: list[float], *, min_pad: float = 0.0, max_pad: float = 0.0) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        delta = 1.0 if low == 0.0 else abs(low) * 0.1
        low -= delta
        high += delta
    span = high - low
    return low - span * min_pad, high + span * max_pad


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", weight: str = "normal") -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family={SVG_FONT_FAMILY!r} '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{safe}</text>'
    )


def _polyline(
    xs: list[float],
    ys: list[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    color: str,
    stroke_width: float = 2.0,
) -> str:
    if not xs:
        return ""
    x_min = xs[0]
    x_max = xs[-1]
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    points: list[str] = []
    for x, y in zip(xs, ys, strict=True):
        px = x0 + (x - x_min) / x_span * width
        py = y0 + height - (y - y_min) / y_span * height
        points.append(f"{px:.2f},{py:.2f}")
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'points="{" ".join(points)}" />'
    )


def _panel_grid(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    x_label: str | None = None,
    left_label: str | None = None,
    right_label: str | None = None,
    right_min: float | None = None,
    right_max: float | None = None,
) -> str:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="white" stroke="#222" stroke-width="1.2" />'
    ]
    for frac in (0.25, 0.5, 0.75):
        py = y + height * frac
        parts.append(
            f'<line x1="{x:.1f}" y1="{py:.1f}" x2="{x + width:.1f}" y2="{py:.1f}" stroke="#ddd" stroke-width="1" />'
        )
    for frac in (0.25, 0.5, 0.75):
        px = x + width * frac
        parts.append(
            f'<line x1="{px:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{y + height:.1f}" stroke="#eee" stroke-width="1" />'
        )
    parts.append(_svg_text(x - 12, y + 4, f"{y_max:.3f}", size=11, anchor="end"))
    parts.append(_svg_text(x - 12, y + height + 4, f"{y_min:.3f}", size=11, anchor="end"))
    if left_label:
        parts.append(_svg_text(x - 58, y + height / 2, left_label, size=12, anchor="middle"))
    if right_label and right_min is not None and right_max is not None:
        parts.append(_svg_text(x + width + 12, y + 4, f"{right_max:.2f}", size=11, anchor="start"))
        parts.append(_svg_text(x + width + 12, y + height + 4, f"{right_min:.2f}", size=11, anchor="start"))
        parts.append(_svg_text(x + width + 56, y + height / 2, right_label, size=12, anchor="middle"))
    if x_label:
        parts.append(_svg_text(x + width / 2, y + height + 28, x_label, size=12, anchor="middle"))
    return "\n".join(parts)


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    cursor = x
    for label, color in items:
        parts.append(f'<line x1="{cursor:.1f}" y1="{y:.1f}" x2="{cursor + 18:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" />')
        parts.append(_svg_text(cursor + 24, y + 4, label, size=12))
        cursor += 24 + len(label) * 7.2 + 18
    return "\n".join(parts)


def build_figure(records: list[dict[str, Any]], *, start_step: int, end_step: int, output_path: Path) -> None:
    window = [record for record in records if start_step <= int(record["step_index"]) <= end_step]
    if not window:
        raise ValueError("selected window is empty")

    times = [float(record["sim_time_seconds"]) - float(window[0]["sim_time_seconds"]) for record in window]
    heights = [
        float(record["info"]["aircraft_by_role"][record["ego_role"]]["position"][1])
        for record in window
    ]
    ground_heights = [float(record["info"]["arena"]["ground_height"]) for record in window]
    ground_clearances = [
        height - ground_height - AIRCRAFT_COLLISION_BROADPHASE_RADIUS_METERS
        for height, ground_height in zip(heights, ground_heights, strict=True)
    ]
    ground_threat = [float(record["reward"]["ground_boundary_threat"]) for record in window]
    attack_adv = [float(record["reward"]["attack_advantage"]) for record in window]
    threat_adv = [float(record["reward"]["threat_advantage"]) for record in window]
    time_pressure = [float(record["reward"]["time_pressure"]) for record in window]
    pitch = [float(record["action_cont"][1]) for record in window]
    roll = [float(record["action_cont"][2]) for record in window]
    yaw = [float(record["action_cont"][3]) for record in window]
    throttle = [float(record["action_cont"][0]) for record in window]
    ground_penalty = [float(record["reward"]["ground_boundary_penalty"]) for record in window]
    boundary_recovery = [float(record["reward"]["boundary_recovery_bonus"]) for record in window]
    total = [float(record["reward"]["total"]) for record in window]

    width = 1400
    height = 1120
    left_margin = 120
    right_margin = 120
    top_margin = 90
    panel_gap = 46
    panel_height = 200
    plot_width = width - left_margin - right_margin

    clearance_ymin, clearance_ymax = _pad_range(ground_clearances, min_pad=0.10, max_pad=0.10)
    adv_ymin, adv_ymax = _pad_range(attack_adv + threat_adv + time_pressure, min_pad=0.15, max_pad=0.15)
    reward_ymin, reward_ymax = _pad_range(ground_penalty + boundary_recovery + time_pressure + total, min_pad=0.15, max_pad=0.15)

    panels_y = [
        top_margin,
        top_margin + panel_height + panel_gap,
        top_margin + 2 * (panel_height + panel_gap),
        top_margin + 3 * (panel_height + panel_gap),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        _svg_text(
            width / 2,
            34,
            "低空无助滚转案例的 reward diagnostics 时序图",
            size=22,
            anchor="middle",
            weight="bold",
        ),
        _svg_text(
            width / 2,
            58,
            f"录制：open_fighter2_tail_chase-20260514-123421 | 窗口：step {start_step}-{end_step}",
            size=13,
            anchor="middle",
        ),
    ]

    parts.append(_svg_text(left_margin, panels_y[0] - 16, "A. 边界状态", size=16, weight="bold"))
    parts.append(
        _panel_grid(
            x=left_margin,
            y=panels_y[0],
            width=plot_width,
            height=panel_height,
            y_min=clearance_ymin,
            y_max=clearance_ymax,
            left_label="净空 (m)",
            right_label="威胁",
            right_min=0.0,
            right_max=1.0,
        )
    )
    parts.append(_legend(left_margin + 8, panels_y[0] + 16, [("ground_clearance", "#1f77b4"), ("ground_boundary_threat", "#d62728")]))
    parts.append(_polyline(times, ground_clearances, x0=left_margin, y0=panels_y[0], width=plot_width, height=panel_height, y_min=clearance_ymin, y_max=clearance_ymax, color="#1f77b4"))
    parts.append(_polyline(times, ground_threat, x0=left_margin, y0=panels_y[0], width=plot_width, height=panel_height, y_min=0.0, y_max=1.0, color="#d62728"))

    parts.append(_svg_text(left_margin, panels_y[1] - 16, "B. 攻防态势", size=16, weight="bold"))
    parts.append(
        _panel_grid(
            x=left_margin,
            y=panels_y[1],
            width=plot_width,
            height=panel_height,
            y_min=adv_ymin,
            y_max=adv_ymax,
            left_label="数值",
        )
    )
    parts.append(_legend(left_margin + 8, panels_y[1] + 16, [("attack_advantage", "#2ca02c"), ("threat_advantage", "#ff7f0e"), ("time_pressure", "#9467bd")]))
    parts.append(_polyline(times, attack_adv, x0=left_margin, y0=panels_y[1], width=plot_width, height=panel_height, y_min=adv_ymin, y_max=adv_ymax, color="#2ca02c"))
    parts.append(_polyline(times, threat_adv, x0=left_margin, y0=panels_y[1], width=plot_width, height=panel_height, y_min=adv_ymin, y_max=adv_ymax, color="#ff7f0e"))
    parts.append(_polyline(times, time_pressure, x0=left_margin, y0=panels_y[1], width=plot_width, height=panel_height, y_min=adv_ymin, y_max=adv_ymax, color="#9467bd"))

    parts.append(_svg_text(left_margin, panels_y[2] - 16, "C. 动作输出", size=16, weight="bold"))
    parts.append(
        _panel_grid(
            x=left_margin,
            y=panels_y[2],
            width=plot_width,
            height=panel_height,
            y_min=-1.05,
            y_max=1.05,
            left_label="动作",
        )
    )
    parts.append(_legend(left_margin + 8, panels_y[2] + 16, [("pitch", "#d62728"), ("roll", "#1f77b4"), ("yaw", "#2ca02c"), ("throttle", "#7f7f7f")]))
    parts.append(_polyline(times, pitch, x0=left_margin, y0=panels_y[2], width=plot_width, height=panel_height, y_min=-1.05, y_max=1.05, color="#d62728"))
    parts.append(_polyline(times, roll, x0=left_margin, y0=panels_y[2], width=plot_width, height=panel_height, y_min=-1.05, y_max=1.05, color="#1f77b4"))
    parts.append(_polyline(times, yaw, x0=left_margin, y0=panels_y[2], width=plot_width, height=panel_height, y_min=-1.05, y_max=1.05, color="#2ca02c"))
    parts.append(_polyline(times, throttle, x0=left_margin, y0=panels_y[2], width=plot_width, height=panel_height, y_min=-1.05, y_max=1.05, color="#7f7f7f"))

    parts.append(_svg_text(left_margin, panels_y[3] - 16, "D. 局部奖励项", size=16, weight="bold"))
    parts.append(
        _panel_grid(
            x=left_margin,
            y=panels_y[3],
            width=plot_width,
            height=panel_height,
            y_min=reward_ymin,
            y_max=reward_ymax,
            x_label="局部时间 (s)",
            left_label="奖励",
        )
    )
    parts.append(_legend(left_margin + 8, panels_y[3] + 16, [("ground_boundary_penalty", "#d62728"), ("boundary_recovery_bonus", "#1f77b4"), ("time_pressure", "#9467bd"), ("total", "#111111")]))
    parts.append(_polyline(times, ground_penalty, x0=left_margin, y0=panels_y[3], width=plot_width, height=panel_height, y_min=reward_ymin, y_max=reward_ymax, color="#d62728"))
    parts.append(_polyline(times, boundary_recovery, x0=left_margin, y0=panels_y[3], width=plot_width, height=panel_height, y_min=reward_ymin, y_max=reward_ymax, color="#1f77b4"))
    parts.append(_polyline(times, time_pressure, x0=left_margin, y0=panels_y[3], width=plot_width, height=panel_height, y_min=reward_ymin, y_max=reward_ymax, color="#9467bd"))
    parts.append(_polyline(times, total, x0=left_margin, y0=panels_y[3], width=plot_width, height=panel_height, y_min=reward_ymin, y_max=reward_ymax, color="#111111", stroke_width=2.4))
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis reward diagnostics SVG from analyzed episode frames.")
    parser.add_argument("--frames-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    args = parser.parse_args()

    records = _load_records(args.frames_jsonl)
    build_figure(records, start_step=args.start_step, end_step=args.end_step, output_path=args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
