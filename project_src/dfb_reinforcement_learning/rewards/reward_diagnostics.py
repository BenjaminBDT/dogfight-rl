from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


REWARD_COMPONENT_FIELDS: tuple[str, ...] = (
    "total",
    "time_pressure",
    "distance_band",
    "attack_advantage",
    "threat_advantage",
    "tracking_delta_bonus",
    "shot_delta_bonus",
    "tail_delta_bonus",
    "pitch_up_tracking",
    "maneuver_activity",
    "flat_roll_bonus",
    "brake_penalty",
    "throttle_change_bonus",
    "throttle_low_penalty",
    "low_speed_penalty",
    "speed_jitter_penalty",
    "one_circle_speed_reward",
    "two_circle_speed_reward",
    "pitch_jitter_penalty",
    "roll_jitter_penalty",
    "yaw_jitter_penalty",
    "stall_penalty",
    "overheat_penalty",
    "repair_twitch_penalty",
    "repair_static_penalty",
    "repair_high_health_penalty",
    "repair_under_threat_penalty",
    "repair_low_health_bonus",
    "repair_destroyed_subsystem_bonus",
    "ground_boundary_penalty",
    "ceiling_boundary_penalty",
    "horizontal_boundary_penalty",
    "boundary_recovery_bonus",
    "out_of_bounds_time_penalty",
    "predictive_fire_hit_bonus",
    "fire_command_bonus",
    "fire_window_bonus",
    "fire_hesitation_penalty",
    "hit_enemy_bonus",
    "got_hit_penalty",
    "aircraft_collision_threat",
    "aircraft_collision_penalty",
    "surface_collision_penalty",
    "self_destroy_penalty",
    "enemy_destroy_bonus",
)


def _empty_values() -> dict[str, float]:
    return {name: 0.0 for name in REWARD_COMPONENT_FIELDS}


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in REWARD_COMPONENT_FIELDS}


@dataclass
class RewardDiagnosticsAccumulator:
    sample_count: int = 0
    sums: dict[str, float] = field(default_factory=_empty_values)
    absolute_sums: dict[str, float] = field(default_factory=_empty_values)
    positive_counts: dict[str, int] = field(default_factory=_empty_counts)
    negative_counts: dict[str, int] = field(default_factory=_empty_counts)

    def add(self, breakdown: Any) -> None:
        if hasattr(breakdown, "asdict"):
            values = breakdown.asdict()
        elif isinstance(breakdown, Mapping):
            values = breakdown
        else:
            values = vars(breakdown)
        self.sample_count += 1
        for name in REWARD_COMPONENT_FIELDS:
            value = float(values.get(name, 0.0))
            self.sums[name] += value
            self.absolute_sums[name] += abs(value)
            if value > 0.0:
                self.positive_counts[name] += 1
            elif value < 0.0:
                self.negative_counts[name] += 1

    def merge_payload(self, payload: Mapping[str, Any]) -> None:
        self.sample_count += int(payload.get("sample_count", 0))
        for target, key in (
            (self.sums, "sums"),
            (self.absolute_sums, "absolute_sums"),
            (self.positive_counts, "positive_counts"),
            (self.negative_counts, "negative_counts"),
        ):
            source = payload.get(key, {})
            if not isinstance(source, Mapping):
                raise TypeError(f"reward diagnostics field {key!r} must be a mapping")
            for name in REWARD_COMPONENT_FIELDS:
                target[name] += source.get(name, 0)

    def raw_payload(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "sums": dict(self.sums),
            "absolute_sums": dict(self.absolute_sums),
            "positive_counts": dict(self.positive_counts),
            "negative_counts": dict(self.negative_counts),
        }

    def summary(self) -> dict[str, Any]:
        denominator = max(self.sample_count, 1)
        return {
            "sample_count": self.sample_count,
            "components": {
                name: {
                    "mean": self.sums[name] / denominator,
                    "mean_abs": self.absolute_sums[name] / denominator,
                    "positive_fraction": self.positive_counts[name] / denominator,
                    "negative_fraction": self.negative_counts[name] / denominator,
                }
                for name in REWARD_COMPONENT_FIELDS
            },
        }

    def drain_raw_payload(self) -> dict[str, Any]:
        payload = self.raw_payload()
        self.clear()
        return payload

    def clear(self) -> None:
        self.sample_count = 0
        self.sums = _empty_values()
        self.absolute_sums = _empty_values()
        self.positive_counts = _empty_counts()
        self.negative_counts = _empty_counts()
