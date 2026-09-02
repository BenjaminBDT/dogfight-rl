from __future__ import annotations

import torch


def weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or sample_weight.shape != values.shape:
        raise ValueError("weighted mean expects matching one-dimensional tensors")
    if not torch.isfinite(sample_weight).all() or bool(torch.any(sample_weight <= 0.0)):
        raise ValueError("sample weights must be finite and positive")
    return (values * sample_weight).sum() / sample_weight.sum()
