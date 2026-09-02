"""DFB state estimation package."""

from __future__ import annotations

from pathlib import Path

__version__ = (
    Path(__file__).resolve().parent.joinpath("VERSION").read_text(encoding="utf-8").strip()
)
