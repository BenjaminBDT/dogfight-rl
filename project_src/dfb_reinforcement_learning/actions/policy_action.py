from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from dfb_game_py import EnvironmentAction
from dfb_reinforcement_learning.policy_contract import ACTION_BIN_DIM, ACTION_CONT_DIM

HYBRID_ACTION_DIM_CONTINUOUS = ACTION_CONT_DIM
HYBRID_ACTION_DIM_BINARY = ACTION_BIN_DIM


@dataclass(frozen=True)
class HybridAction:
    throttle: float
    pitch: float
    roll: float
    yaw: float
    brake: bool
    fire_gun: bool
    repair: bool

    def continuous_array(self) -> np.ndarray:
        return np.asarray(
            [self.throttle, self.pitch, self.roll, self.yaw],
            dtype=np.float32,
        )

    def binary_array(self) -> np.ndarray:
        return np.asarray(
            [self.brake, self.fire_gun, self.repair],
            dtype=np.float32,
        )


class ActionAdapter:
    """Adapter between Part 3 hybrid actions and dfb_game EnvironmentAction."""

    @staticmethod
    def from_arrays(
        continuous: Sequence[float] | np.ndarray,
        binary: Sequence[float | bool] | np.ndarray,
        *,
        binary_threshold: float = 0.5,
    ) -> HybridAction:
        continuous_np = np.asarray(list(continuous), dtype=np.float32)
        binary_np = np.asarray(list(binary), dtype=np.float32)
        if continuous_np.shape != (HYBRID_ACTION_DIM_CONTINUOUS,):
            raise ValueError("continuous action must have shape (4,)")
        if binary_np.shape != (HYBRID_ACTION_DIM_BINARY,):
            raise ValueError("binary action must have shape (3,)")
        clipped = np.clip(continuous_np, -1.0, 1.0)
        return HybridAction(
            throttle=float(clipped[0]),
            pitch=float(clipped[1]),
            roll=float(clipped[2]),
            yaw=float(clipped[3]),
            brake=bool(binary_np[0] >= binary_threshold),
            fire_gun=bool(binary_np[1] >= binary_threshold),
            repair=bool(binary_np[2] >= binary_threshold),
        )

    @staticmethod
    def to_environment_action(action: HybridAction | dict[str, object]) -> EnvironmentAction:
        if isinstance(action, dict):
            action = HybridAction(
                throttle=float(action["throttle"]),
                pitch=float(action["pitch"]),
                roll=float(action["roll"]),
                yaw=float(action["yaw"]),
                brake=bool(action["brake"]),
                fire_gun=bool(action["fire_gun"]),
                repair=bool(action["repair"]),
            )
        return EnvironmentAction(
            float(np.clip(action.throttle, -1.0, 1.0)),
            bool(action.brake),
            float(np.clip(action.pitch, -1.0, 1.0)),
            float(np.clip(action.roll, -1.0, 1.0)),
            float(np.clip(action.yaw, -1.0, 1.0)),
            bool(action.fire_gun),
            bool(action.repair),
        )

    @staticmethod
    def neutral() -> HybridAction:
        return HybridAction(
            throttle=0.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            brake=False,
            fire_gun=False,
            repair=False,
        )

    @staticmethod
    def batch_to_arrays(actions: Iterable[HybridAction]) -> tuple[np.ndarray, np.ndarray]:
        actions = list(actions)
        return (
            np.asarray([action.continuous_array() for action in actions], dtype=np.float32),
            np.asarray([action.binary_array() for action in actions], dtype=np.float32),
        )
