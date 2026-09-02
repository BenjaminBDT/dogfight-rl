"""Loss definitions for Part 2."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AudioConfidenceConfig",
    "AudioLossWeights",
    "AudioSupervisionTargets",
    "BeliefLossWeights",
    "BeliefSupervisionTargets",
    "EvidenceLossWeights",
    "EvidenceSupervisionTargets",
    "SingleViewSegmentationTargets",
    "TemporalLossWeights",
    "TemporalSupervisionTargets",
    "VisionLossWeights",
    "VisionSupervisionTargets",
    "compute_single_view_segmentation_loss",
    "compute_audio_confidence_targets",
    "compute_single_step_audio_loss",
    "compute_temporal_belief_loss",
    "compute_single_step_evidence_loss",
    "compute_temporal_modality_loss",
    "compute_single_step_vision_loss",
]


def __getattr__(name: str) -> Any:
    if name in {
        "AudioConfidenceConfig",
        "AudioLossWeights",
        "AudioSupervisionTargets",
        "compute_audio_confidence_targets",
        "compute_single_step_audio_loss",
    }:
        module = import_module(".audio_supervision", __name__)
        return getattr(module, name)
    if name in {
        "BeliefLossWeights",
        "BeliefSupervisionTargets",
        "compute_temporal_belief_loss",
    }:
        module = import_module(".belief_supervision", __name__)
        return getattr(module, name)
    if name in {
        "EvidenceLossWeights",
        "EvidenceSupervisionTargets",
        "compute_single_step_evidence_loss",
    }:
        module = import_module(".evidence_supervision", __name__)
        return getattr(module, name)
    if name in {
        "TemporalLossWeights",
        "TemporalSupervisionTargets",
        "compute_temporal_modality_loss",
    }:
        module = import_module(".temporal_supervision", __name__)
        return getattr(module, name)
    if name in {
        "SingleViewSegmentationTargets",
        "VisionLossWeights",
        "VisionSupervisionTargets",
        "compute_single_view_segmentation_loss",
        "compute_single_step_vision_loss",
    }:
        module = import_module(".vision_supervision", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
