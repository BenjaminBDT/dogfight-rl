"""Temporal models for Part 2."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BeliefUpdateBackboneOutput",
    "BeliefUpdateConfig",
    "BeliefUpdateInputs",
    "BeliefStateHeads",
    "BeliefStateOutput",
    "BeliefUpdateTokenBuilder",
    "BeliefUpdateTokenOutput",
    "PolicyViewAdapter",
    "PolicyViewOutput",
    "TemporalBeliefUpdateStage",
    "TemporalBeliefUpdateStageOutput",
    "TemporalBeliefUpdateTransformer",
    "CoarseStateOutput",
    "TemporalModalityCalibrationHeads",
    "TemporalModalityCalibrationStage",
    "TemporalModalityBackboneOutput",
    "TemporalModalityConfig",
    "TemporalModalityInputs",
    "TemporalModalityProjection",
    "TemporalModalityProjectionOutput",
    "TemporalModalityStageOutput",
    "TemporalModalityTransformer",
    "compute_selected_segmentation_difference_t",
    "compute_selected_keypoint_delta_t",
    "compute_delta_binaural_cue_t",
    "TemporalVisualRoutingOutput",
    "route_selected_view_with_inertia_t",
    "reroute_vision_output_with_inertia",
    "select_view_tensor",
    "select_view_target_probability",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BeliefUpdateBackboneOutput",
        "BeliefUpdateConfig",
        "BeliefUpdateInputs",
        "BeliefStateHeads",
        "BeliefStateOutput",
        "BeliefUpdateTokenBuilder",
        "BeliefUpdateTokenOutput",
        "PolicyViewAdapter",
        "PolicyViewOutput",
        "TemporalBeliefUpdateStage",
        "TemporalBeliefUpdateStageOutput",
        "TemporalBeliefUpdateTransformer",
    }:
        module = import_module(".belief", __name__)
        return getattr(module, name)
    if name in {
        "CoarseStateOutput",
        "TemporalModalityCalibrationHeads",
        "TemporalModalityCalibrationStage",
        "TemporalModalityBackboneOutput",
        "TemporalModalityConfig",
        "TemporalModalityInputs",
        "TemporalModalityProjection",
        "TemporalModalityProjectionOutput",
        "TemporalModalityStageOutput",
        "TemporalModalityTransformer",
        "compute_selected_segmentation_difference_t",
        "compute_selected_keypoint_delta_t",
        "compute_delta_binaural_cue_t",
        "select_view_tensor",
        "select_view_target_probability",
    }:
        module = import_module(".modality", __name__)
        return getattr(module, name)
    if name in {
        "TemporalVisualRoutingOutput",
        "route_selected_view_with_inertia_t",
        "reroute_vision_output_with_inertia",
    }:
        module = import_module(".router", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
