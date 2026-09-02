from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", weight: str = "normal") -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family={SVG_FONT_FAMILY!r} '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{safe}</text>'
    )


def _pad_range(values: list[float], *, min_pad: float = 0.0, max_pad: float = 0.0) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        delta = 1.0 if low == 0.0 else abs(low) * 0.1
        low -= delta
        high += delta
    span = high - low
    return low - span * min_pad, high + span * max_pad


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


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    cursor = x
    for label, color in items:
        parts.append(f'<line x1="{cursor:.1f}" y1="{y:.1f}" x2="{cursor + 18:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" />')
        parts.append(_svg_text(cursor + 24, y + 4, label, size=12))
        cursor += 24 + len(label) * 7.2 + 18
    return "\n".join(parts)


def _panel_grid(*, x: float, y: float, width: float, height: float, y_min: float, y_max: float, x_label: str, left_label: str, right_label: str, right_min: float, right_max: float) -> str:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="white" stroke="#222" stroke-width="1.2" />'
    ]
    for frac in (0.25, 0.5, 0.75):
        py = y + height * frac
        px = x + width * frac
        parts.append(f'<line x1="{x:.1f}" y1="{py:.1f}" x2="{x + width:.1f}" y2="{py:.1f}" stroke="#ddd" stroke-width="1" />')
        parts.append(f'<line x1="{px:.1f}" y1="{y:.1f}" x2="{px:.1f}" y2="{y + height:.1f}" stroke="#eee" stroke-width="1" />')
    parts.append(_svg_text(x - 12, y + 4, f"{y_max:.2f}", size=11, anchor="end"))
    parts.append(_svg_text(x - 12, y + height + 4, f"{y_min:.2f}", size=11, anchor="end"))
    parts.append(_svg_text(x - 48, y + height / 2, left_label, size=12, anchor="middle"))
    parts.append(_svg_text(x + width + 12, y + 4, f"{right_max:.3f}", size=11, anchor="start"))
    parts.append(_svg_text(x + width + 12, y + height + 4, f"{right_min:.3f}", size=11, anchor="start"))
    parts.append(_svg_text(x + width + 46, y + height / 2, right_label, size=12, anchor="middle"))
    parts.append(_svg_text(x + width / 2, y + height + 28, x_label, size=12, anchor="middle"))
    return "\n".join(parts)


def _slice(records: list[dict[str, Any]], start_step: int, end_step: int) -> list[dict[str, Any]]:
    return [record for record in records if start_step <= int(record["step_index"]) <= end_step]


def _series(records: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for record in records:
        if key in {"pitch", "roll", "yaw", "throttle"}:
            index = {"throttle": 0, "pitch": 1, "roll": 2, "yaw": 3}[key]
            out.append(float(record["action_cont"][index]))
        else:
            out.append(float(record["reward"][key]))
    return out


def build_figure(
    *,
    headon_records: list[dict[str, Any]],
    tail_records: list[dict[str, Any]],
    recovery_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    headon = _slice(headon_records, 80, 180)
    tail = _slice(tail_records, 0, 240)
    recovery = _slice(recovery_records, 0, 2400)

    width = 1800
    height = 700
    margin_x = 90
    top = 90
    panel_width = 470
    panel_height = 360
    gap = 95
    panel_y = 180
    panel_xs = [margin_x, margin_x + panel_width + gap, margin_x + 2 * (panel_width + gap)]

    panels = [
        {
            "title": "A. 对头交汇与快速终结",
            "subtitle": "bc_refresh_6000_0014 | open_head_on_200m-20260507-143148 | step 80-180",
            "records": headon,
            "left": [("shot_feasibility", "#d62728"), ("tracking_quality", "#1f77b4")],
            "right": [("attack_advantage", "#2ca02c"), ("threat_advantage", "#ff7f0e")],
            "caption": "在短窗口内迅速形成高 shot feasibility 与较高攻击优势，体现对头终结能力。",
        },
        {
            "title": "B. 咬尾保持与追击压制",
            "subtitle": "bc_new_3000 | open_fighter2_tail_chase-20260512-091454 | step 0-240",
            "records": tail,
            "left": [("tail_hold_score", "#d62728"), ("tracking_quality", "#1f77b4")],
            "right": [("attack_advantage", "#2ca02c")],
            "caption": "在开局尾追窗口内持续维持较高 tail hold 与 tracking quality，体现稳定追击保持能力。",
        },
        {
            "title": "C. 被咬尾恢复与重回对头",
            "subtitle": "bc_refresh_12416 | open_fighter1_tail_chase-20260509-152617 | step 0-2400",
            "records": recovery,
            "left": [("tracking_quality", "#1f77b4")],
            "right": [("attack_advantage", "#2ca02c"), ("threat_advantage", "#ff7f0e")],
            "caption": "初始阶段 threat 较高，后续 tracking 恢复并多次重新形成攻击窗口，体现 defensive recovery 能力。",
        },
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        _svg_text(width / 2, 34, "代表性专项策略行为图", size=24, anchor="middle", weight="bold"),
        _svg_text(width / 2, 60, "三份代表性 checkpoint 在专项录制中的局部行为时序", size=14, anchor="middle"),
    ]

    for panel_x, panel in zip(panel_xs, panels, strict=True):
        recs = panel["records"]
        times = [float(r["sim_time_seconds"]) - float(recs[0]["sim_time_seconds"]) for r in recs]
        left_values = [v for key, _ in panel["left"] for v in _series(recs, key)]
        right_values = [v for key, _ in panel["right"] for v in _series(recs, key)]
        left_min, left_max = _pad_range(left_values, min_pad=0.08, max_pad=0.08)
        right_min, right_max = _pad_range(right_values, min_pad=0.12, max_pad=0.12)

        parts.append(_svg_text(panel_x, 112, panel["title"], size=17, weight="bold"))
        parts.append(_svg_text(panel_x, 136, panel["subtitle"], size=11))
        parts.append(
            _panel_grid(
                x=panel_x,
                y=panel_y,
                width=panel_width,
                height=panel_height,
                y_min=left_min,
                y_max=left_max,
                x_label="局部时间 (s)",
                left_label="状态量",
                right_label="优势量",
                right_min=right_min,
                right_max=right_max,
            )
        )
        parts.append(_legend(panel_x + 8, panel_y + 18, panel["left"] + panel["right"]))
        for key, color in panel["left"]:
            parts.append(_polyline(times, _series(recs, key), x0=panel_x, y0=panel_y, width=panel_width, height=panel_height, y_min=left_min, y_max=left_max, color=color))
        for key, color in panel["right"]:
            parts.append(_polyline(times, _series(recs, key), x0=panel_x, y0=panel_y, width=panel_width, height=panel_height, y_min=right_min, y_max=right_max, color=color))
        parts.append(_svg_text(panel_x, panel_y + panel_height + 54, panel["caption"], size=12))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis specialist triptych SVG.")
    parser.add_argument("--headon-frames", type=Path, required=True)
    parser.add_argument("--tail-frames", type=Path, required=True)
    parser.add_argument("--recovery-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    build_figure(
        headon_records=_load_records(args.headon_frames),
        tail_records=_load_records(args.tail_frames),
        recovery_records=_load_records(args.recovery_frames),
        output_path=args.output,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
