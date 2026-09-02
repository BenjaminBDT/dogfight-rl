from __future__ import annotations

from typing import Any

import numpy as np


CONTINUOUS_ACTION_NAMES: tuple[str, ...] = (
    "throttle_delta",
    "pitch",
    "roll",
    "yaw",
)
BINARY_ACTION_NAMES: tuple[str, ...] = (
    "brake",
    "fire_gun",
    "repair",
)


def _named_values(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _mean_on_run_steps(
    actions: np.ndarray,
    dones: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    steps, envs, dimensions = actions.shape
    runs: list[list[int]] = [[] for _ in range(dimensions)]
    for env_index in range(envs):
        current = np.zeros((dimensions,), dtype=np.int64)
        for step in range(steps):
            if not valid_mask[step, env_index]:
                for dim in range(dimensions):
                    if current[dim] > 0:
                        runs[dim].append(int(current[dim]))
                        current[dim] = 0
                continue
            active = actions[step, env_index] >= 0.5
            for dim in range(dimensions):
                if active[dim]:
                    current[dim] += 1
                elif current[dim] > 0:
                    runs[dim].append(int(current[dim]))
                    current[dim] = 0
            if dones[step, env_index] >= 0.5:
                for dim in range(dimensions):
                    if current[dim] > 0:
                        runs[dim].append(int(current[dim]))
                        current[dim] = 0
        for dim in range(dimensions):
            if current[dim] > 0:
                runs[dim].append(int(current[dim]))
    return np.asarray(
        [float(np.mean(dim_runs)) if dim_runs else 0.0 for dim_runs in runs],
        dtype=np.float64,
    )


def summarize_actions(
    *,
    continuous: np.ndarray,
    binary: np.ndarray,
    binary_probabilities: np.ndarray,
    dones: np.ndarray,
    valid_mask: np.ndarray | None = None,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    if continuous.ndim != 3 or continuous.shape[-1] != len(CONTINUOUS_ACTION_NAMES):
        raise ValueError(f"continuous actions must have shape [T, E, {len(CONTINUOUS_ACTION_NAMES)}]")
    if binary.ndim != 3 or binary.shape[-1] != len(BINARY_ACTION_NAMES):
        raise ValueError(f"binary actions must have shape [T, E, {len(BINARY_ACTION_NAMES)}]")
    if binary_probabilities.shape != binary.shape:
        raise ValueError("binary probabilities must match binary action shape")
    if dones.shape != continuous.shape[:2]:
        raise ValueError("done mask must match action time and environment dimensions")
    if valid_mask is None:
        valid_mask = np.ones(continuous.shape[:2], dtype=np.bool_)
    if valid_mask.shape != continuous.shape[:2]:
        raise ValueError("valid mask must match action time and environment dimensions")
    if not np.any(valid_mask):
        raise ValueError("action diagnostics require at least one valid sample")

    continuous_flat = continuous[valid_mask]
    binary_flat = binary[valid_mask]
    probability_flat = binary_probabilities[valid_mask]
    valid_continuity = (
        (dones[:-1] < 0.5)
        & valid_mask[:-1]
        & valid_mask[1:]
    )
    if continuous.shape[0] > 1 and np.any(valid_continuity):
        continuous_delta = np.abs(continuous[1:] - continuous[:-1])
        mean_abs_delta = continuous_delta[valid_continuity].mean(axis=0)
        previous_on = binary[:-1] >= 0.5
        current_on = binary[1:] >= 0.5
        persistence = np.zeros((binary.shape[-1],), dtype=np.float64)
        for dim in range(binary.shape[-1]):
            denominator_mask = valid_continuity & previous_on[:, :, dim]
            persistence[dim] = (
                float(current_on[:, :, dim][denominator_mask].mean())
                if np.any(denominator_mask)
                else 0.0
            )
    else:
        mean_abs_delta = np.zeros((continuous.shape[-1],), dtype=np.float64)
        persistence = np.zeros((binary.shape[-1],), dtype=np.float64)

    return {
        "sample_count": int(continuous_flat.shape[0]),
        "continuous": {
            "mean": _named_values(CONTINUOUS_ACTION_NAMES, continuous_flat.mean(axis=0)),
            "std": _named_values(CONTINUOUS_ACTION_NAMES, continuous_flat.std(axis=0)),
            "saturation_fraction": _named_values(
                CONTINUOUS_ACTION_NAMES,
                (np.abs(continuous_flat) >= saturation_threshold).mean(axis=0),
            ),
            "mean_abs_step_delta": _named_values(CONTINUOUS_ACTION_NAMES, mean_abs_delta),
        },
        "binary": {
            "probability_mean": _named_values(
                BINARY_ACTION_NAMES,
                probability_flat.mean(axis=0),
            ),
            "on_fraction": _named_values(BINARY_ACTION_NAMES, binary_flat.mean(axis=0)),
            "on_persistence": _named_values(BINARY_ACTION_NAMES, persistence),
            "mean_on_run_steps": _named_values(
                BINARY_ACTION_NAMES,
                _mean_on_run_steps(binary, dones, valid_mask),
            ),
        },
    }


def summarize_values(
    *,
    values: np.ndarray,
    returns: np.ndarray,
    popart_mean_before: float,
    popart_std_before: float,
    popart_mean_after: float,
    popart_std_after: float,
) -> dict[str, float]:
    values_flat = np.asarray(values, dtype=np.float64).reshape(-1)
    returns_flat = np.asarray(returns, dtype=np.float64).reshape(-1)
    if values_flat.shape != returns_flat.shape:
        raise ValueError("values and returns must have the same flattened shape")
    errors = returns_flat - values_flat
    return_variance = float(np.var(returns_flat))
    explained_variance = (
        1.0 - float(np.var(errors)) / return_variance
        if return_variance > 1e-12
        else 0.0
    )
    return {
        "sample_count": float(values_flat.size),
        "explained_variance": explained_variance,
        "raw_mae": float(np.mean(np.abs(errors))),
        "raw_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "value_mean": float(np.mean(values_flat)),
        "value_std": float(np.std(values_flat)),
        "return_mean": float(np.mean(returns_flat)),
        "return_std": float(np.std(returns_flat)),
        "popart_mean_before": float(popart_mean_before),
        "popart_std_before": float(popart_std_before),
        "popart_mean_after": float(popart_mean_after),
        "popart_std_after": float(popart_std_after),
    }
