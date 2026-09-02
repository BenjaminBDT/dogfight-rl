"""Single-step evidence extraction models for Part 2."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "EvidenceFeaturesOutput",
    "EvidenceStateOutput",
    "SingleStepEvidenceConfig",
    "SingleStepEvidenceModule",
    "SingleStepEvidenceOutput",
]


def __getattr__(name: str) -> Any:
    if name in {
        "EvidenceFeaturesOutput",
        "EvidenceStateOutput",
        "SingleStepEvidenceConfig",
        "SingleStepEvidenceModule",
        "SingleStepEvidenceOutput",
    }:
        module = import_module(".module", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
