"""Audio models for Part 2."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AudioBackboneConfig",
    "AudioHeadConfig",
    "SingleStepAudioConfig",
    "SingleStepAudioModule",
    "SingleStepAudioOutput",
]


def __getattr__(name: str) -> Any:
    if name == "AudioBackboneConfig":
        module = import_module(".backbone", __name__)
        return getattr(module, name)
    if name == "AudioHeadConfig":
        module = import_module(".heads", __name__)
        return getattr(module, name)
    if name in {
        "SingleStepAudioConfig",
        "SingleStepAudioModule",
        "SingleStepAudioOutput",
    }:
        module = import_module(".module", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
