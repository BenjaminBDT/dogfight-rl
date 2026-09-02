"""Vision models for Part 2."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DeepLabSingleStepVisionConfig",
    "DeepLabSingleStepVisionModule",
    "DeepLabSingleViewSegmentationConfig",
    "DeepLabSingleViewSegmentationModule",
    "GeometryValidationConfig",
    "SingleViewSegmentationConfig",
    "SingleViewSegmentationModule",
    "SingleViewSegmentationOutput",
    "SingleStepVisionConfig",
    "SelectedVisualCandidateOutput",
    "SingleStepVisionCandidateOutput",
    "SingleStepVisionModule",
    "SingleStepVisionOutput",
    "VisionBackboneConfig",
    "VisionHeadConfig",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DeepLabSingleStepVisionConfig",
        "DeepLabSingleStepVisionModule",
        "DeepLabSingleViewSegmentationConfig",
        "DeepLabSingleViewSegmentationModule",
        "SingleViewSegmentationConfig",
        "SingleViewSegmentationModule",
        "SingleViewSegmentationOutput",
            "SingleStepVisionConfig",
            "SelectedVisualCandidateOutput",
            "SingleStepVisionCandidateOutput",
            "SingleStepVisionModule",
            "SingleStepVisionOutput",
    }:
        if name in {
            "DeepLabSingleStepVisionConfig",
            "DeepLabSingleStepVisionModule",
        }:
            module = import_module(".deeplab_module", __name__)
        elif name in {
            "DeepLabSingleViewSegmentationConfig",
            "DeepLabSingleViewSegmentationModule",
            "SingleViewSegmentationConfig",
            "SingleViewSegmentationModule",
            "SingleViewSegmentationOutput",
        }:
            module = import_module(".single_view_segmentation_module", __name__)
        else:
            module = import_module(".module", __name__)
        return getattr(module, name)
    if name == "GeometryValidationConfig":
        module = import_module(".geometry", __name__)
        return getattr(module, name)
    if name == "VisionBackboneConfig":
        module = import_module(".backbone", __name__)
        return getattr(module, name)
    if name == "VisionHeadConfig":
        module = import_module(".heads", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
