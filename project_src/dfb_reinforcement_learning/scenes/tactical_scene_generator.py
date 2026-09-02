from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np


TURN_RADIUS_METERS = 80.0
RECOVERY_BOUNDARY_MARGIN_METERS = TURN_RADIUS_METERS * 4.0
RECOVERY_OPPONENT_MAX_DISTANCE_METERS = 500.0


@dataclass(frozen=True)
class ArenaPreset:
    name: str
    arena_radius: float
    ground_height: float
    flight_ceiling_height: float
    ceiling_falloff_range: float
    out_of_bounds_grace_seconds: float


ARENA_PRESETS: dict[str, ArenaPreset] = {
    "standard_open": ArenaPreset(
        name="standard_open",
        arena_radius=3000.0,
        ground_height=0.0,
        flight_ceiling_height=2000.0,
        ceiling_falloff_range=250.0,
        out_of_bounds_grace_seconds=20.0,
    ),
    "low_ceiling": ArenaPreset(
        name="low_ceiling",
        arena_radius=3000.0,
        ground_height=0.0,
        flight_ceiling_height=2000.0,
        ceiling_falloff_range=220.0,
        out_of_bounds_grace_seconds=20.0,
    ),
    "boundary_pressure": ArenaPreset(
        name="boundary_pressure",
        arena_radius=1800.0,
        ground_height=0.0,
        flight_ceiling_height=2000.0,
        ceiling_falloff_range=220.0,
        out_of_bounds_grace_seconds=14.0,
    ),
    "wide_open": ArenaPreset(
        name="wide_open",
        arena_radius=4500.0,
        ground_height=0.0,
        flight_ceiling_height=2000.0,
        ceiling_falloff_range=280.0,
        out_of_bounds_grace_seconds=24.0,
    ),
}


@dataclass(frozen=True)
class SpawnState:
    position: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]
    initial_speed: float
    initial_throttle: float
    initial_damage: SpawnDamage | None = None


@dataclass(frozen=True)
class SpawnDamage:
    total_hit_points_fraction: float | None = None
    left_wing_fraction: float | None = None
    right_wing_fraction: float | None = None
    pitch_tail_fraction: float | None = None
    yaw_tail_fraction: float | None = None
    engine_fraction: float | None = None


@dataclass(frozen=True)
class GeneratedScene:
    name: str
    arena: ArenaPreset
    fighter1_spawn: SpawnState
    fighter2_spawn: SpawnState
    metadata: dict[str, Any]

    def to_ron(self) -> str:
        def fmt_vec3(value: tuple[float, float, float]) -> str:
            return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"

        def fmt_spawn(spawn: SpawnState) -> str:
            if spawn.initial_damage is None:
                damage_text = "None"
            else:
                damage = spawn.initial_damage
                damage_lines = []
                if damage.total_hit_points_fraction is not None:
                    damage_lines.append(
                        f"            total_hit_points_fraction: Some({damage.total_hit_points_fraction:.3f}),\n"
                    )
                if damage.left_wing_fraction is not None:
                    damage_lines.append(
                        f"            left_wing_fraction: Some({damage.left_wing_fraction:.3f}),\n"
                    )
                if damage.right_wing_fraction is not None:
                    damage_lines.append(
                        f"            right_wing_fraction: Some({damage.right_wing_fraction:.3f}),\n"
                    )
                if damage.pitch_tail_fraction is not None:
                    damage_lines.append(
                        f"            pitch_tail_fraction: Some({damage.pitch_tail_fraction:.3f}),\n"
                    )
                if damage.yaw_tail_fraction is not None:
                    damage_lines.append(
                        f"            yaw_tail_fraction: Some({damage.yaw_tail_fraction:.3f}),\n"
                    )
                if damage.engine_fraction is not None:
                    damage_lines.append(
                        f"            engine_fraction: Some({damage.engine_fraction:.3f}),\n"
                    )
                damage_text = "Some((\n" + "".join(damage_lines) + "        ))"
            return (
                "(\n"
                f"        position: {fmt_vec3(spawn.position)},\n"
                f"        rotation_degrees: {fmt_vec3(spawn.rotation_degrees)},\n"
                f"        initial_speed: {spawn.initial_speed:.3f},\n"
                f"        initial_throttle: {spawn.initial_throttle:.3f},\n"
                f"        initial_damage: {damage_text},\n"
                "    )"
            )

        return (
            "(\n"
            f"    arena_radius: {self.arena.arena_radius:.3f},\n"
            f"    ground_height: {self.arena.ground_height:.3f},\n"
            f"    flight_ceiling_height: {self.arena.flight_ceiling_height:.3f},\n"
            f"    ceiling_falloff_range: {self.arena.ceiling_falloff_range:.3f},\n"
            f"    out_of_bounds_grace_seconds: {self.arena.out_of_bounds_grace_seconds:.3f},\n"
            f"    fighter1_spawn: {fmt_spawn(self.fighter1_spawn)},\n"
            f"    fighter2_spawn: {fmt_spawn(self.fighter2_spawn)},\n"
            "    obstacles: [],\n"
            "    ground_clutter: [],\n"
            "    ground_accents: [],\n"
            "    sky_markers: [],\n"
            ")\n"
        )


@dataclass(frozen=True)
class GeneratedSceneSpec:
    name: str
    template: str
    weight: float = 1.0
    count: int = 1
    seed: int = 0
    arena_preset: str = "standard_open"
    altitude_profile: str = "mid"
    boundary_profile: str = "center"
    advantaged_role: str = "fighter1"
    center_role: str = "fighter1"
    damage_profile: str = "none"


@dataclass(frozen=True)
class ExistingSceneSpec:
    scene_name: str
    weight: float = 1.0


@dataclass(frozen=True)
class MaterializedScene:
    label: str
    weight: float
    scene_name: str | None = None
    scene_path: str | None = None
    generated: bool = False
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScenePoolSpec:
    train_tactical_generated: list[GeneratedSceneSpec]
    train_tactical_existing: list[ExistingSceneSpec]
    train_recovery_generated: list[GeneratedSceneSpec]
    train_recovery_existing: list[ExistingSceneSpec]
    train_tactical_ratio: float
    train_recovery_ratio: float
    eval_generated: list[GeneratedSceneSpec]
    eval_existing: list[ExistingSceneSpec]

    @classmethod
    def from_json(cls, path: Path) -> "ScenePoolSpec":
        payload = json.loads(path.read_text(encoding="utf-8"))

        def parse_generated(items: list[dict[str, Any]]) -> list[GeneratedSceneSpec]:
            return [
                GeneratedSceneSpec(
                    name=str(item["name"]),
                    template=str(item["template"]),
                    weight=float(item.get("weight", 1.0)),
                    count=int(item.get("count", 1)),
                    seed=int(item.get("seed", 0)),
                    arena_preset=str(item.get("arena_preset", "standard_open")),
                    altitude_profile=str(item.get("altitude_profile", "mid")),
                    boundary_profile=str(item.get("boundary_profile", "center")),
                    advantaged_role=str(item.get("advantaged_role", "fighter1")),
                    center_role=str(item.get("center_role", "fighter1")),
                    damage_profile=str(item.get("damage_profile", "none")),
                )
                for item in items
            ]

        def parse_existing(items: list[dict[str, Any]]) -> list[ExistingSceneSpec]:
            return [
                ExistingSceneSpec(
                    scene_name=str(item["scene_name"]),
                    weight=float(item.get("weight", 1.0)),
                )
                for item in items
            ]

        train = payload.get("train", {})
        train_tactical = train.get("tactical", {})
        train_recovery = train.get("recovery", {})
        eval_spec = payload.get("eval", {})
        return cls(
            train_tactical_generated=parse_generated(train_tactical.get("generated", [])),
            train_tactical_existing=parse_existing(train_tactical.get("existing", [])),
            train_recovery_generated=parse_generated(train_recovery.get("generated", [])),
            train_recovery_existing=parse_existing(train_recovery.get("existing", [])),
            train_tactical_ratio=float(train.get("tactical_ratio", 0.7)),
            train_recovery_ratio=float(train.get("recovery_ratio", 0.3)),
            eval_generated=parse_generated(eval_spec.get("generated", [])),
            eval_existing=parse_existing(eval_spec.get("existing", [])),
        )


class PreparedScenePool:
    def __init__(
        self,
        *,
        train_tactical_scenes: list[MaterializedScene],
        train_recovery_scenes: list[MaterializedScene],
        train_tactical_ratio: float,
        train_recovery_ratio: float,
        eval_scenes: list[MaterializedScene],
        rng_seed: int,
    ) -> None:
        if not train_tactical_scenes and not train_recovery_scenes:
            raise ValueError("train scene pool must not be empty")
        self.train_tactical_scenes = train_tactical_scenes
        self.train_recovery_scenes = train_recovery_scenes
        self.train_tactical_ratio = max(train_tactical_ratio, 0.0)
        self.train_recovery_ratio = max(train_recovery_ratio, 0.0)
        self.eval_scenes = eval_scenes
        self._rng = random.Random(rng_seed)

    def sample_train_scene(self) -> MaterializedScene:
        choose_tactical = False
        if self.train_tactical_scenes and self.train_recovery_scenes:
            total = self.train_tactical_ratio + self.train_recovery_ratio
            tactical_prob = 0.5 if total <= 1e-6 else (self.train_tactical_ratio / total)
            choose_tactical = self._rng.random() < tactical_prob
        elif self.train_tactical_scenes:
            choose_tactical = True
        scenes = self.train_tactical_scenes if choose_tactical else self.train_recovery_scenes
        weights = [max(scene.weight, 0.0) for scene in scenes]
        return self._rng.choices(scenes, weights=weights, k=1)[0]

    def eval_scene_for_episode(self, episode_index: int) -> MaterializedScene:
        if not self.eval_scenes:
            raise ValueError("eval scene pool must not be empty")
        return self.eval_scenes[episode_index % len(self.eval_scenes)]


def _yaw_forward(yaw_degrees: float) -> np.ndarray:
    radians = np.deg2rad(yaw_degrees)
    return np.asarray([np.sin(radians), 0.0, np.cos(radians)], dtype=np.float32)


def _yaw_left(yaw_degrees: float) -> np.ndarray:
    radians = np.deg2rad(yaw_degrees)
    return np.asarray([np.cos(radians), 0.0, -np.sin(radians)], dtype=np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def _yaw_from_direction(direction: np.ndarray) -> float:
    direction = _normalize(np.asarray([direction[0], 0.0, direction[2]], dtype=np.float32))
    return float(np.rad2deg(np.arctan2(direction[0], direction[2])))


def _sample_altitude(rng: random.Random, arena: ArenaPreset, profile: str) -> float:
    if profile == "low":
        low = arena.ground_height + 60.0
        high = min(arena.ground_height + 180.0, arena.flight_ceiling_height - 120.0)
    elif profile == "ceiling":
        low = arena.flight_ceiling_height - 180.0
        high = arena.flight_ceiling_height - 80.0
    elif profile == "high":
        low = arena.ground_height + 780.0
        high = arena.flight_ceiling_height - 140.0
    else:
        low = arena.ground_height + 350.0
        high = min(arena.ground_height + 750.0, arena.flight_ceiling_height - 180.0)
    if high <= low:
        high = low + 40.0
    return rng.uniform(low, high)


def _sample_uniform_orientation(rng: random.Random) -> tuple[float, float, float]:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    x, y, z, w = qx, qy, qz, qw
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    pitch_x = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        yaw_y = np.sign(sinp) * (np.pi / 2.0)
    else:
        yaw_y = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    roll_z = np.arctan2(siny_cosp, cosy_cosp)
    return (
        float(np.rad2deg(pitch_x)),
        float(np.rad2deg(yaw_y)),
        float(np.rad2deg(roll_z)),
    )


def _rotation_from_forward_with_random_roll(
    rng: random.Random,
    forward: np.ndarray,
) -> tuple[float, float, float]:
    forward = _normalize(forward)
    pitch = float(-np.rad2deg(np.arcsin(np.clip(float(forward[1]), -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(float(forward[0]), float(forward[2]))))
    roll = rng.uniform(-180.0, 180.0)
    return pitch, yaw, roll


def _random_speed_and_throttle(rng: random.Random) -> tuple[float, float]:
    return rng.uniform(0.0, 80.0), rng.uniform(0.0, 1.0)


def _forward_from_rotation(rotation_degrees: tuple[float, float, float]) -> np.ndarray:
    pitch = np.deg2rad(rotation_degrees[0])
    yaw = np.deg2rad(rotation_degrees[1])
    return _normalize(
        np.asarray(
            [
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
                np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )
    )


def _sample_anchor_xy(rng: random.Random, arena: ArenaPreset, boundary_profile: str) -> np.ndarray:
    if boundary_profile == "edge":
        radius = rng.uniform(0.68, 0.84) * arena.arena_radius
    elif boundary_profile == "near_edge":
        radius = rng.uniform(0.78, 0.9) * arena.arena_radius
    else:
        radius = rng.uniform(0.0, 0.35) * arena.arena_radius
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.asarray([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)


def _sample_front_hemisphere_offset(
    rng: random.Random,
    *,
    forward: np.ndarray,
    radius: float,
    min_distance: float = 80.0,
) -> np.ndarray:
    forward = _normalize(forward)
    low = max(min_distance, 0.0)
    high = max(radius, low + 1e-3)
    distance = ((high**3 - low**3) * rng.random() + low**3) ** (1.0 / 3.0)
    while True:
        direction = np.asarray(
            [rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)],
            dtype=np.float32,
        )
        direction = _normalize(direction)
        if float(np.dot(direction, forward)) < 0.0:
            direction = -direction
        if float(np.dot(direction, forward)) > 1e-4:
            return direction * float(distance)


def _sample_front_hemisphere_forward(
    rng: random.Random,
    *,
    center_direction: np.ndarray,
    min_dot: float = 0.15,
) -> np.ndarray:
    center_direction = _normalize(center_direction)
    for _ in range(256):
        candidate = _sample_relative_direction(rng)
        if float(np.dot(candidate, center_direction)) >= min_dot:
            return candidate
    return center_direction


def _is_position_inside_arena(
    position: np.ndarray,
    arena: ArenaPreset,
    *,
    collision_radius: float = 5.4,
) -> bool:
    floor = arena.ground_height + collision_radius
    ceiling = arena.flight_ceiling_height - collision_radius
    if float(position[1]) < floor or float(position[1]) > ceiling:
        return False
    xz = np.asarray([position[0], position[2]], dtype=np.float32)
    return float(np.linalg.norm(xz)) <= arena.arena_radius - collision_radius


def _clamp_inside_radius(position: np.ndarray, *, radius_limit: float, margin: float) -> np.ndarray:
    xz = np.asarray([position[0], position[2]], dtype=np.float32)
    radius = float(np.linalg.norm(xz))
    max_radius = max(radius_limit - margin, 0.0)
    if radius > max_radius and radius > 1e-6:
        xz = xz * (max_radius / radius)
        position = position.copy()
        position[0] = float(xz[0])
        position[2] = float(xz[1])
    return position


def _clamp_inside_arena(position: np.ndarray, arena: ArenaPreset, margin: float = 120.0) -> np.ndarray:
    return _clamp_inside_radius(position, radius_limit=arena.arena_radius, margin=margin)


def _recovery_boundary_radius(arena: ArenaPreset) -> float:
    return arena.arena_radius + RECOVERY_BOUNDARY_MARGIN_METERS


def _is_spawn_safe_linear(
    *,
    position: np.ndarray,
    rotation_degrees: tuple[float, float, float],
    speed: float,
    arena: ArenaPreset,
    horizon_seconds: float,
    collision_radius: float = 5.4,
    horizontal_radius_override: float | None = None,
) -> bool:
    forward = _forward_from_rotation(rotation_degrees)
    future = position + forward * speed * horizon_seconds
    floor = arena.ground_height + collision_radius
    ceiling = arena.flight_ceiling_height - collision_radius
    if float(future[1]) < floor or float(future[1]) > ceiling:
        return False
    future_xz = np.asarray([future[0], future[2]], dtype=np.float32)
    horizontal_radius = arena.arena_radius if horizontal_radius_override is None else horizontal_radius_override
    if float(np.linalg.norm(future_xz)) > horizontal_radius - collision_radius:
        return False
    return True


def _positions_collide_linear(
    *,
    position_a: np.ndarray,
    rotation_a: tuple[float, float, float],
    speed_a: float,
    position_b: np.ndarray,
    rotation_b: tuple[float, float, float],
    speed_b: float,
    horizon_seconds: float,
    collision_radius: float = 5.4,
) -> bool:
    forward_a = _forward_from_rotation(rotation_a)
    forward_b = _forward_from_rotation(rotation_b)
    rel_pos = position_b - position_a
    rel_vel = forward_b * speed_b - forward_a * speed_a
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq <= 1e-6:
        tau = 0.0
    else:
        tau = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon_seconds))
    closest = rel_pos + rel_vel * tau
    return float(np.linalg.norm(closest)) <= collision_radius * 2.0


def _sample_relative_direction(rng: random.Random) -> np.ndarray:
    return _normalize(np.asarray([rng.normalvariate(0.0, 1.0) for _ in range(3)], dtype=np.float32))


def _sample_enemy_near_self(
    *,
    rng: random.Random,
    self_position: np.ndarray,
    arena: ArenaPreset,
    altitude_profile: str,
    horizontal_radius_limit: float,
    collision_radius: float,
    tactical_radius: float,
    horizon_seconds: float,
) -> SpawnState:
    for _ in range(256):
        direction = _sample_relative_direction(rng)
        radius = tactical_radius * np.cbrt(rng.random()) * rng.uniform(0.6, 1.1)
        radius = min(radius, tactical_radius)
        enemy_position = self_position + direction * radius
        enemy_position[1] = np.clip(
            enemy_position[1],
            arena.ground_height + 60.0,
            arena.flight_ceiling_height - 60.0,
        )
        enemy_position = _clamp_inside_radius(
            enemy_position,
            radius_limit=horizontal_radius_limit,
            margin=40.0,
        )
        if float(np.linalg.norm(enemy_position - self_position)) <= collision_radius * 3.0:
            continue
        enemy_rotation = _sample_uniform_orientation(rng)
        enemy_throttle = rng.uniform(0.0, 1.0)
        enemy_speed = float(np.interp(enemy_throttle, [0.0, 1.0], [32.0, 78.0]).item())
        if altitude_profile == "low":
            enemy_rotation = (
                float(np.clip(enemy_rotation[0], -35.0, 35.0)),
                enemy_rotation[1],
                enemy_rotation[2],
            )
        if not _is_spawn_safe_linear(
            position=enemy_position,
            rotation_degrees=enemy_rotation,
            speed=enemy_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
            horizontal_radius_override=horizontal_radius_limit,
        ):
            continue
        return _spawn_state(
            position=enemy_position,
            yaw=enemy_rotation[1],
            pitch=enemy_rotation[0],
            roll=enemy_rotation[2],
            speed=enemy_speed,
            throttle=float(enemy_throttle),
        )
    raise RuntimeError("failed to sample enemy recovery spawn")


def _sample_uniform_full_range_damage(rng: random.Random) -> SpawnDamage:
    return SpawnDamage(
        left_wing_fraction=rng.uniform(0.0, 1.0),
        right_wing_fraction=rng.uniform(0.0, 1.0),
        pitch_tail_fraction=rng.uniform(0.0, 1.0),
        yaw_tail_fraction=rng.uniform(0.0, 1.0),
        engine_fraction=rng.uniform(0.0, 1.0),
    )


def _sample_repairable_damage(rng: random.Random) -> SpawnDamage:
    subsystem_fractions = [rng.uniform(0.72, 1.0) for _ in range(5)]
    subsystem_fractions[rng.randrange(len(subsystem_fractions))] = rng.uniform(0.30, 0.48)
    return SpawnDamage(
        total_hit_points_fraction=rng.uniform(0.65, 0.90),
        left_wing_fraction=subsystem_fractions[0],
        right_wing_fraction=subsystem_fractions[1],
        pitch_tail_fraction=subsystem_fractions[2],
        yaw_tail_fraction=subsystem_fractions[3],
        engine_fraction=subsystem_fractions[4],
    )


def _sample_combat_wear_damage(rng: random.Random) -> SpawnDamage | None:
    total_roll = rng.random()
    if total_roll < 0.75:
        total_fraction: float | None = None
    elif total_roll < 0.95:
        total_fraction = rng.uniform(0.60, 0.90)
    else:
        total_fraction = rng.uniform(0.35, 0.60)

    subsystem_fractions: list[float | None] = [None] * 5
    if rng.random() < 0.02:
        subsystem_fractions[rng.randrange(len(subsystem_fractions))] = rng.uniform(0.50, 0.85)
        if total_fraction is None:
            total_fraction = rng.uniform(0.75, 0.95)

    if total_fraction is None:
        return None
    return SpawnDamage(
        total_hit_points_fraction=total_fraction,
        left_wing_fraction=subsystem_fractions[0],
        right_wing_fraction=subsystem_fractions[1],
        pitch_tail_fraction=subsystem_fractions[2],
        yaw_tail_fraction=subsystem_fractions[3],
        engine_fraction=subsystem_fractions[4],
    )


def _spawn_damage_for_profile(rng: random.Random, damage_profile: str) -> SpawnDamage | None:
    if damage_profile == "none":
        return None
    if damage_profile == "uniform_full_range":
        return _sample_uniform_full_range_damage(rng)
    if damage_profile == "repairable_pair":
        return _sample_repairable_damage(rng)
    if damage_profile == "combat_wear":
        return _sample_combat_wear_damage(rng)
    raise ValueError(f"unsupported damage profile: {damage_profile}")


def _apply_damage_profile_to_scene(
    scene: GeneratedScene,
    *,
    spec: GeneratedSceneSpec,
    rng: random.Random,
) -> GeneratedScene:
    fighter1_damage = _spawn_damage_for_profile(rng, spec.damage_profile)
    fighter2_damage = _spawn_damage_for_profile(rng, spec.damage_profile)
    metadata = dict(scene.metadata)
    metadata["damage_profile"] = spec.damage_profile
    return replace(
        scene,
        fighter1_spawn=replace(scene.fighter1_spawn, initial_damage=fighter1_damage),
        fighter2_spawn=replace(scene.fighter2_spawn, initial_damage=fighter2_damage),
        metadata=metadata,
    )


def _generate_recovery_pair(
    *,
    name: str,
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    self_spawn: SpawnState,
    enemy_spawn: SpawnState,
    metadata: dict[str, Any],
) -> GeneratedScene:
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=self_spawn,
        fighter2_spawn=enemy_spawn,
        metadata=metadata,
    )


def _generate_mild_oob_recoverable(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    tactical_radius = RECOVERY_OPPONENT_MAX_DISTANCE_METERS
    recovery_boundary_radius = _recovery_boundary_radius(arena)
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    for _ in range(256):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(arena.arena_radius + 25.0, min(arena.arena_radius + 220.0, recovery_boundary_radius - 30.0))
        self_position = np.asarray([np.cos(angle) * radius, altitude, np.sin(angle) * radius], dtype=np.float32)
        inward = _normalize(np.asarray([-self_position[0], 0.0, -self_position[2]], dtype=np.float32))
        tangent = np.asarray([inward[2], 0.0, -inward[0]], dtype=np.float32)
        heading = _normalize(inward + tangent * rng.uniform(-0.35, 0.35))
        self_yaw = _yaw_from_direction(heading)
        self_pitch = rng.uniform(-10.0, 10.0)
        self_roll = rng.uniform(-20.0, 20.0)
        self_throttle = rng.uniform(0.42, 0.78)
        self_speed = float(np.interp(self_throttle, [0.0, 1.0], [32.0, 78.0]).item())
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=(self_pitch, self_yaw, self_roll),
            speed=self_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
            horizontal_radius_override=recovery_boundary_radius,
        ):
            continue
        center_distance = radius - arena.arena_radius
        if center_distance / max(self_speed, 1e-3) > 10.0:
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=self_yaw,
            pitch=self_pitch,
            roll=self_roll,
            speed=self_speed,
            throttle=self_throttle,
        )
        enemy_spawn = _sample_enemy_near_self(
            rng=rng,
            self_position=self_position,
            arena=arena,
            altitude_profile=spec.altitude_profile,
            horizontal_radius_limit=recovery_boundary_radius,
            collision_radius=collision_radius,
            tactical_radius=tactical_radius,
            horizon_seconds=horizon_seconds,
        )
        if _positions_collide_linear(
            position_a=np.asarray(self_spawn.position, dtype=np.float32),
            rotation_a=self_spawn.rotation_degrees,
            speed_a=self_spawn.initial_speed,
            position_b=np.asarray(enemy_spawn.position, dtype=np.float32),
            rotation_b=enemy_spawn.rotation_degrees,
            speed_b=enemy_spawn.initial_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        return _generate_recovery_pair(
            name=name,
            spec=spec,
            arena=arena,
            self_spawn=self_spawn,
            enemy_spawn=enemy_spawn,
            metadata={
                "template": spec.template,
                "altitude_profile": spec.altitude_profile,
                "boundary_profile": spec.boundary_profile,
                "sampling_horizon_seconds": horizon_seconds,
                "recovery_boundary_radius": recovery_boundary_radius,
                "outside_arena_distance": center_distance,
            },
        )
    raise RuntimeError(f"failed to generate mild out-of-bounds recovery scene for {name}")


def _generate_low_altitude_recoverable(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    tactical_radius = RECOVERY_OPPONENT_MAX_DISTANCE_METERS
    recovery_boundary_radius = _recovery_boundary_radius(arena)
    altitude = rng.uniform(arena.ground_height + 45.0, arena.ground_height + 120.0)
    for _ in range(256):
        self_anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
        self_position = np.asarray([self_anchor_xy[0], altitude, self_anchor_xy[1]], dtype=np.float32)
        pitch = rng.uniform(-6.0, 18.0)
        yaw = rng.uniform(0.0, 360.0)
        roll = rng.uniform(-45.0, 45.0)
        throttle = rng.uniform(0.45, 0.82)
        speed = float(np.interp(throttle, [0.0, 1.0], [32.0, 78.0]).item())
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=(pitch, yaw, roll),
            speed=speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
            horizontal_radius_override=recovery_boundary_radius,
        ):
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            speed=speed,
            throttle=throttle,
        )
        enemy_spawn = _sample_enemy_near_self(
            rng=rng,
            self_position=self_position,
            arena=arena,
            altitude_profile="low",
            horizontal_radius_limit=recovery_boundary_radius,
            collision_radius=collision_radius,
            tactical_radius=tactical_radius,
            horizon_seconds=horizon_seconds,
        )
        if _positions_collide_linear(
            position_a=np.asarray(self_spawn.position, dtype=np.float32),
            rotation_a=self_spawn.rotation_degrees,
            speed_a=self_spawn.initial_speed,
            position_b=np.asarray(enemy_spawn.position, dtype=np.float32),
            rotation_b=enemy_spawn.rotation_degrees,
            speed_b=enemy_spawn.initial_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        return _generate_recovery_pair(
            name=name,
            spec=spec,
            arena=arena,
            self_spawn=self_spawn,
            enemy_spawn=enemy_spawn,
            metadata={
                "template": spec.template,
                "altitude_profile": "low",
                "boundary_profile": spec.boundary_profile,
                "sampling_horizon_seconds": horizon_seconds,
                "recovery_boundary_radius": recovery_boundary_radius,
            },
        )
    raise RuntimeError(f"failed to generate low-altitude recovery scene for {name}")


def _generate_near_boundary_recoverable(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    tactical_radius = RECOVERY_OPPONENT_MAX_DISTANCE_METERS
    recovery_boundary_radius = _recovery_boundary_radius(arena)
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    for _ in range(256):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(arena.arena_radius - 160.0, arena.arena_radius + 90.0)
        self_position = np.asarray([np.cos(angle) * radius, altitude, np.sin(angle) * radius], dtype=np.float32)
        inward = _normalize(np.asarray([-self_position[0], 0.0, -self_position[2]], dtype=np.float32))
        tangent = np.asarray([inward[2], 0.0, -inward[0]], dtype=np.float32)
        heading = _normalize(inward * rng.uniform(0.55, 0.95) + tangent * rng.uniform(-0.7, 0.7))
        self_yaw = _yaw_from_direction(heading)
        self_pitch = rng.uniform(-8.0, 10.0)
        self_roll = rng.uniform(-35.0, 35.0)
        self_throttle = rng.uniform(0.44, 0.82)
        self_speed = float(np.interp(self_throttle, [0.0, 1.0], [32.0, 78.0]).item())
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=(self_pitch, self_yaw, self_roll),
            speed=self_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
            horizontal_radius_override=recovery_boundary_radius,
        ):
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=self_yaw,
            pitch=self_pitch,
            roll=self_roll,
            speed=self_speed,
            throttle=self_throttle,
        )
        enemy_spawn = _sample_enemy_near_self(
            rng=rng,
            self_position=self_position,
            arena=arena,
            altitude_profile=spec.altitude_profile,
            horizontal_radius_limit=recovery_boundary_radius,
            collision_radius=collision_radius,
            tactical_radius=tactical_radius,
            horizon_seconds=horizon_seconds,
        )
        if _positions_collide_linear(
            position_a=np.asarray(self_spawn.position, dtype=np.float32),
            rotation_a=self_spawn.rotation_degrees,
            speed_a=self_spawn.initial_speed,
            position_b=np.asarray(enemy_spawn.position, dtype=np.float32),
            rotation_b=enemy_spawn.rotation_degrees,
            speed_b=enemy_spawn.initial_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        return _generate_recovery_pair(
            name=name,
            spec=spec,
            arena=arena,
            self_spawn=self_spawn,
            enemy_spawn=enemy_spawn,
            metadata={
                "template": spec.template,
                "altitude_profile": spec.altitude_profile,
                "boundary_profile": "near_edge",
                "sampling_horizon_seconds": horizon_seconds,
                "recovery_boundary_radius": recovery_boundary_radius,
            },
        )
    raise RuntimeError(f"failed to generate near-boundary recovery scene for {name}")


def _spawn_state(
    *,
    position: np.ndarray,
    yaw: float,
    speed: float,
    throttle: float,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> SpawnState:
    return SpawnState(
        position=(float(position[0]), float(position[1]), float(position[2])),
        rotation_degrees=(float(pitch), float(yaw), float(roll)),
        initial_speed=float(speed),
        initial_throttle=float(throttle),
    )


def _generate_head_on(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
    midpoint = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
    base_yaw = rng.uniform(0.0, 360.0)
    forward = _yaw_forward(base_yaw)
    left = _yaw_left(base_yaw)
    separation = rng.uniform(180.0, 420.0)
    lateral_offset = rng.uniform(-35.0, 35.0)
    vertical_offset = rng.uniform(-20.0, 20.0)
    fighter1_pos = midpoint - forward * (separation * 0.5) + left * lateral_offset
    fighter2_pos = midpoint + forward * (separation * 0.5) - left * lateral_offset
    fighter2_pos[1] += vertical_offset
    fighter1_pos = _clamp_inside_arena(fighter1_pos, arena)
    fighter2_pos = _clamp_inside_arena(fighter2_pos, arena)
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=_spawn_state(
            position=fighter1_pos,
            yaw=base_yaw + rng.uniform(-12.0, 12.0),
            speed=rng.uniform(56.0, 66.0),
            throttle=rng.uniform(0.56, 0.66),
        ),
        fighter2_spawn=_spawn_state(
            position=fighter2_pos,
            yaw=base_yaw + 180.0 + rng.uniform(-12.0, 12.0),
            speed=rng.uniform(56.0, 66.0),
            throttle=rng.uniform(0.56, 0.66),
        ),
        metadata={
            "template": spec.template,
            "altitude_profile": spec.altitude_profile,
            "boundary_profile": spec.boundary_profile,
            "base_yaw": base_yaw,
            "separation": separation,
        },
    )


def _generate_safe_repair_pair(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, "mid")
    anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile="center")
    midpoint = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
    base_yaw = rng.uniform(0.0, 360.0)
    left = _yaw_left(base_yaw)
    separation = rng.uniform(1_000.0, 1_400.0)
    fighter1_position = _clamp_inside_arena(midpoint + left * (separation * 0.5), arena)
    fighter2_position = _clamp_inside_arena(midpoint - left * (separation * 0.5), arena)
    speed = rng.uniform(60.0, 68.0)
    throttle = rng.uniform(0.60, 0.68)
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=_spawn_state(
            position=fighter1_position,
            yaw=base_yaw,
            speed=speed,
            throttle=throttle,
        ),
        fighter2_spawn=_spawn_state(
            position=fighter2_position,
            yaw=base_yaw,
            speed=speed,
            throttle=throttle,
        ),
        metadata={
            "template": spec.template,
            "altitude_profile": "mid",
            "boundary_profile": "center",
            "base_yaw": base_yaw,
            "separation": separation,
            "repair_safety_horizon_seconds": 11.0,
        },
    )


def _generate_close_head_on(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    min_front_dot = 0.15
    for _ in range(2048):
        altitude = _sample_altitude(rng, arena, spec.altitude_profile)
        anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
        midpoint = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
        separation_direction = _sample_relative_direction(rng)
        separation_distance = rng.uniform(10.0, 80.0)
        fighter1_position = midpoint - separation_direction * (separation_distance * 0.5)
        fighter2_position = midpoint + separation_direction * (separation_distance * 0.5)
        if not _is_position_inside_arena(fighter1_position, arena, collision_radius=collision_radius):
            continue
        if not _is_position_inside_arena(fighter2_position, arena, collision_radius=collision_radius):
            continue

        fighter1_forward = _sample_front_hemisphere_forward(
            rng,
            center_direction=separation_direction,
            min_dot=min_front_dot,
        )
        fighter2_forward = _sample_front_hemisphere_forward(
            rng,
            center_direction=-separation_direction,
            min_dot=min_front_dot,
        )
        fighter1_rotation = _rotation_from_forward_with_random_roll(rng, fighter1_forward)
        fighter2_rotation = _rotation_from_forward_with_random_roll(rng, fighter2_forward)
        fighter1_speed, fighter1_throttle = _random_speed_and_throttle(rng)
        fighter2_speed, fighter2_throttle = _random_speed_and_throttle(rng)
        if not _is_spawn_safe_linear(
            position=fighter1_position,
            rotation_degrees=fighter1_rotation,
            speed=fighter1_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        if not _is_spawn_safe_linear(
            position=fighter2_position,
            rotation_degrees=fighter2_rotation,
            speed=fighter2_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        if _positions_collide_linear(
            position_a=fighter1_position,
            rotation_a=fighter1_rotation,
            speed_a=fighter1_speed,
            position_b=fighter2_position,
            rotation_b=fighter2_rotation,
            speed_b=fighter2_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue

        return GeneratedScene(
            name=name,
            arena=arena,
            fighter1_spawn=_spawn_state(
                position=fighter1_position,
                yaw=fighter1_rotation[1],
                pitch=fighter1_rotation[0],
                roll=fighter1_rotation[2],
                speed=fighter1_speed,
                throttle=fighter1_throttle,
            ),
            fighter2_spawn=_spawn_state(
                position=fighter2_position,
                yaw=fighter2_rotation[1],
                pitch=fighter2_rotation[0],
                roll=fighter2_rotation[2],
                speed=fighter2_speed,
                throttle=fighter2_throttle,
            ),
            metadata={
                "template": spec.template,
                "altitude_profile": spec.altitude_profile,
                "boundary_profile": spec.boundary_profile,
                "separation": separation_distance,
                "minimum_front_dot": min_front_dot,
                "self_safe_horizon_seconds": horizon_seconds,
                "enemy_safe_horizon_seconds": horizon_seconds,
                "pair_collision_horizon_seconds": horizon_seconds,
            },
        )
    raise RuntimeError(f"failed to generate close head-on scene for {name}")


def _generate_tail_chase(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    self_spawn: SpawnState | None = None
    self_forward: np.ndarray | None = None
    for _ in range(256):
        altitude = _sample_altitude(rng, arena, spec.altitude_profile)
        anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
        self_position = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
        self_pitch, self_yaw, self_roll = _sample_uniform_orientation(rng)
        self_speed = rng.uniform(56.0, 76.0)
        self_throttle = rng.uniform(0.60, 0.78)
        self_rotation = (self_pitch, self_yaw, self_roll)
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=self_rotation,
            speed=self_speed,
            arena=arena,
            horizon_seconds=2.0,
        ):
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=self_yaw,
            pitch=self_pitch,
            roll=self_roll,
            speed=self_speed,
            throttle=self_throttle,
        )
        self_forward = _forward_from_rotation(self_rotation)
        break
    if self_spawn is None or self_forward is None:
        raise RuntimeError("failed to sample safe self tail-chase spawn")

    enemy_spawn: SpawnState | None = None
    front_distance: float | None = None
    for _ in range(512):
        offset = _sample_front_hemisphere_offset(rng, forward=self_forward, radius=400.0, min_distance=80.0)
        enemy_position = np.asarray(self_spawn.position, dtype=np.float32) + offset
        if not _is_position_inside_arena(enemy_position, arena):
            continue
        enemy_pitch, enemy_yaw, enemy_roll = _sample_uniform_orientation(rng)
        enemy_rotation = (enemy_pitch, enemy_yaw, enemy_roll)
        enemy_forward = _forward_from_rotation(enemy_rotation)
        to_self = np.asarray(self_spawn.position, dtype=np.float32) - enemy_position
        if float(np.dot(enemy_forward, to_self)) >= 0.0:
            continue
        enemy_speed = rng.uniform(54.0, 68.0)
        enemy_throttle = rng.uniform(0.54, 0.68)
        if not _is_spawn_safe_linear(
            position=enemy_position,
            rotation_degrees=enemy_rotation,
            speed=enemy_speed,
            arena=arena,
            horizon_seconds=2.0,
        ):
            continue
        enemy_spawn = _spawn_state(
            position=enemy_position,
            yaw=enemy_yaw,
            pitch=enemy_pitch,
            roll=enemy_roll,
            speed=enemy_speed,
            throttle=enemy_throttle,
        )
        front_distance = float(np.linalg.norm(offset))
        break
    if enemy_spawn is None or front_distance is None:
        raise RuntimeError("failed to sample enemy front-hemisphere tail-chase spawn")

    if spec.advantaged_role == "fighter2":
        fighter1_spawn = enemy_spawn
        fighter2_spawn = self_spawn
    else:
        fighter1_spawn = self_spawn
        fighter2_spawn = enemy_spawn
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=fighter1_spawn,
        fighter2_spawn=fighter2_spawn,
        metadata={
            "template": spec.template,
            "advantaged_role": spec.advantaged_role,
            "altitude_profile": spec.altitude_profile,
            "boundary_profile": spec.boundary_profile,
            "front_distance": front_distance,
            "self_safe_horizon_seconds": 2.0,
            "enemy_safe_horizon_seconds": 2.0,
        },
    )


def _generate_aim_fire(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    for _ in range(256):
        altitude = _sample_altitude(rng, arena, spec.altitude_profile)
        anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
        self_position = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
        pitch, yaw, roll = _sample_uniform_orientation(rng)
        speed, throttle = _random_speed_and_throttle(rng)
        rotation = (pitch, yaw, roll)
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=rotation,
            speed=speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            speed=speed,
            throttle=throttle,
        )
        self_forward = _forward_from_rotation(rotation)

        enemy_spawn: SpawnState | None = None
        target_distance: float | None = None
        for _ in range(1024):
            distance = rng.uniform(20.0, 80.0)
            enemy_position = self_position + self_forward * distance
            if not _is_position_inside_arena(enemy_position, arena, collision_radius=collision_radius):
                continue
            enemy_pitch, enemy_yaw, enemy_roll = _sample_uniform_orientation(rng)
            enemy_speed, enemy_throttle = _random_speed_and_throttle(rng)
            enemy_rotation = (enemy_pitch, enemy_yaw, enemy_roll)
            if not _is_spawn_safe_linear(
                position=enemy_position,
                rotation_degrees=enemy_rotation,
                speed=enemy_speed,
                arena=arena,
                horizon_seconds=horizon_seconds,
                collision_radius=collision_radius,
            ):
                continue
            if _positions_collide_linear(
                position_a=self_position,
                rotation_a=rotation,
                speed_a=speed,
                position_b=enemy_position,
                rotation_b=enemy_rotation,
                speed_b=enemy_speed,
                horizon_seconds=horizon_seconds,
                collision_radius=collision_radius,
            ):
                continue
            enemy_spawn = _spawn_state(
                position=enemy_position,
                yaw=enemy_yaw,
                pitch=enemy_pitch,
                roll=enemy_roll,
                speed=enemy_speed,
                throttle=enemy_throttle,
            )
            target_distance = distance
            break
        if enemy_spawn is None or target_distance is None:
            continue

        if spec.advantaged_role == "fighter2":
            fighter1_spawn = enemy_spawn
            fighter2_spawn = self_spawn
        else:
            fighter1_spawn = self_spawn
            fighter2_spawn = enemy_spawn
        return GeneratedScene(
            name=name,
            arena=arena,
            fighter1_spawn=fighter1_spawn,
            fighter2_spawn=fighter2_spawn,
            metadata={
                "template": spec.template,
                "advantaged_role": spec.advantaged_role,
                "altitude_profile": spec.altitude_profile,
                "boundary_profile": spec.boundary_profile,
                "target_distance": target_distance,
                "self_safe_horizon_seconds": horizon_seconds,
                "enemy_safe_horizon_seconds": horizon_seconds,
                "pair_collision_horizon_seconds": horizon_seconds,
            },
        )
    raise RuntimeError("failed to sample safe aim-fire scene")


def _generate_centered_tail_chase(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    center_position = np.asarray([0.0, altitude, 0.0], dtype=np.float32)
    center_yaw = rng.uniform(0.0, 360.0)
    center_forward = _yaw_forward(center_yaw)
    center_left = _yaw_left(center_yaw)
    forward_distance = rng.uniform(80.0, 450.0)
    lateral_offset = rng.uniform(-120.0, 120.0)
    vertical_offset = rng.uniform(-80.0, 80.0)
    enemy_position = center_position + center_forward * forward_distance + center_left * lateral_offset
    enemy_position[1] = float(np.clip(enemy_position[1] + vertical_offset, arena.ground_height + 60.0, arena.flight_ceiling_height - 60.0))
    enemy_position = _clamp_inside_arena(enemy_position, arena)
    center_speed, center_throttle = _random_speed_and_throttle(rng)
    enemy_speed, enemy_throttle = _random_speed_and_throttle(rng)
    center_spawn = _spawn_state(
        position=center_position,
        yaw=center_yaw,
        pitch=0.0,
        roll=0.0,
        speed=center_speed,
        throttle=center_throttle,
    )
    enemy_spawn = _spawn_state(
        position=enemy_position,
        yaw=center_yaw + rng.uniform(-12.0, 12.0),
        pitch=0.0,
        roll=0.0,
        speed=enemy_speed,
        throttle=enemy_throttle,
    )
    if spec.center_role == "fighter2":
        fighter1_spawn = enemy_spawn
        fighter2_spawn = center_spawn
    else:
        fighter1_spawn = center_spawn
        fighter2_spawn = enemy_spawn
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=fighter1_spawn,
        fighter2_spawn=fighter2_spawn,
        metadata={
            "template": spec.template,
            "center_role": spec.center_role,
            "altitude_profile": spec.altitude_profile,
            "front_distance": forward_distance,
            "lateral_offset": lateral_offset,
        },
    )


def _generate_centered_being_tailed(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    center_position = np.asarray([0.0, altitude, 0.0], dtype=np.float32)
    center_yaw = rng.uniform(0.0, 360.0)
    center_forward = _yaw_forward(center_yaw)
    center_left = _yaw_left(center_yaw)
    back_distance = rng.uniform(80.0, 450.0)
    lateral_offset = rng.uniform(-120.0, 120.0)
    vertical_offset = rng.uniform(-80.0, 80.0)
    enemy_position = center_position - center_forward * back_distance + center_left * lateral_offset
    enemy_position[1] = float(np.clip(enemy_position[1] + vertical_offset, arena.ground_height + 60.0, arena.flight_ceiling_height - 60.0))
    enemy_position = _clamp_inside_arena(enemy_position, arena)
    center_speed, center_throttle = _random_speed_and_throttle(rng)
    enemy_speed, enemy_throttle = _random_speed_and_throttle(rng)
    center_spawn = _spawn_state(
        position=center_position,
        yaw=center_yaw,
        pitch=0.0,
        roll=0.0,
        speed=center_speed,
        throttle=center_throttle,
    )
    enemy_spawn = _spawn_state(
        position=enemy_position,
        yaw=center_yaw + rng.uniform(-12.0, 12.0),
        pitch=0.0,
        roll=0.0,
        speed=enemy_speed,
        throttle=enemy_throttle,
    )
    if spec.center_role == "fighter2":
        fighter1_spawn = enemy_spawn
        fighter2_spawn = center_spawn
    else:
        fighter1_spawn = center_spawn
        fighter2_spawn = enemy_spawn
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=fighter1_spawn,
        fighter2_spawn=fighter2_spawn,
        metadata={
            "template": spec.template,
            "center_role": spec.center_role,
            "altitude_profile": spec.altitude_profile,
            "back_distance": back_distance,
            "lateral_offset": lateral_offset,
        },
    )


def _generate_side_cut_in(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
    defender_pos = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
    defender_yaw = rng.uniform(0.0, 360.0)
    forward = _yaw_forward(defender_yaw)
    left = _yaw_left(defender_yaw)
    side_sign = -1.0 if rng.random() < 0.5 else 1.0
    back_offset = rng.uniform(80.0, 220.0)
    side_offset = rng.uniform(120.0, 260.0) * side_sign
    attacker_pos = defender_pos - forward * back_offset + left * side_offset
    attacker_pos[1] += rng.uniform(-30.0, 30.0)
    aim_point = defender_pos + forward * rng.uniform(60.0, 140.0)
    attacker_yaw = _yaw_from_direction(aim_point - attacker_pos) + rng.uniform(-12.0, 12.0)
    attacker_speed = rng.uniform(58.0, 72.0)
    defender_speed = rng.uniform(54.0, 66.0)
    attacker = _spawn_state(
        position=_clamp_inside_arena(attacker_pos, arena),
        yaw=attacker_yaw,
        speed=attacker_speed,
        throttle=rng.uniform(0.58, 0.76),
    )
    defender = _spawn_state(
        position=_clamp_inside_arena(defender_pos, arena),
        yaw=defender_yaw + rng.uniform(-8.0, 8.0),
        speed=defender_speed,
        throttle=rng.uniform(0.54, 0.68),
    )
    if spec.advantaged_role == "fighter2":
        fighter1_spawn = defender
        fighter2_spawn = attacker
    else:
        fighter1_spawn = attacker
        fighter2_spawn = defender
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=fighter1_spawn,
        fighter2_spawn=fighter2_spawn,
        metadata={
            "template": spec.template,
            "advantaged_role": spec.advantaged_role,
            "altitude_profile": spec.altitude_profile,
            "boundary_profile": spec.boundary_profile,
            "side_offset": side_offset,
            "back_offset": back_offset,
        },
    )


def _generate_offset_merge(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
    midpoint = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
    base_yaw = rng.uniform(0.0, 360.0)
    forward = _yaw_forward(base_yaw)
    left = _yaw_left(base_yaw)
    separation = rng.uniform(220.0, 520.0)
    cross_offset = rng.uniform(60.0, 180.0) * (-1.0 if rng.random() < 0.5 else 1.0)
    fighter1_pos = midpoint - forward * (separation * 0.5) + left * cross_offset
    fighter2_pos = midpoint + forward * (separation * 0.5) - left * cross_offset
    fighter1_yaw = base_yaw + rng.uniform(-25.0, 20.0)
    fighter2_yaw = base_yaw + 180.0 + rng.uniform(-20.0, 25.0)
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=_spawn_state(
            position=_clamp_inside_arena(fighter1_pos, arena),
            yaw=fighter1_yaw,
            speed=rng.uniform(56.0, 66.0),
            throttle=rng.uniform(0.56, 0.68),
        ),
        fighter2_spawn=_spawn_state(
            position=_clamp_inside_arena(fighter2_pos, arena),
            yaw=fighter2_yaw,
            speed=rng.uniform(56.0, 66.0),
            throttle=rng.uniform(0.56, 0.68),
        ),
        metadata={
            "template": spec.template,
            "altitude_profile": spec.altitude_profile,
            "boundary_profile": spec.boundary_profile,
            "cross_offset": cross_offset,
            "separation": separation,
        },
    )


def _generate_boundary_return(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    for _ in range(512):
        altitude = _sample_altitude(rng, arena, spec.altitude_profile)
        anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile="near_edge")
        radial = _normalize(np.asarray([anchor_xy[0], 0.0, anchor_xy[1]], dtype=np.float32))
        inward = -radial
        tangent = np.asarray([inward[2], 0.0, -inward[0]], dtype=np.float32)
        defender_pos = np.asarray([anchor_xy[0], altitude, anchor_xy[1]], dtype=np.float32)
        defender_yaw = _yaw_from_direction(-inward * 0.75 + tangent * rng.uniform(-0.4, 0.4))
        attacker_pos = (
            defender_pos
            - inward * rng.uniform(80.0, 180.0)
            + tangent * rng.uniform(-50.0, 50.0)
        )
        attacker_yaw = _yaw_from_direction(defender_pos - attacker_pos + inward * 40.0)
        defender = _spawn_state(
            position=_clamp_inside_arena(defender_pos, arena, margin=40.0),
            yaw=defender_yaw,
            speed=rng.uniform(54.0, 64.0),
            throttle=rng.uniform(0.52, 0.66),
        )
        attacker = _spawn_state(
            position=_clamp_inside_arena(attacker_pos, arena, margin=40.0),
            yaw=attacker_yaw + rng.uniform(-10.0, 10.0),
            speed=rng.uniform(58.0, 72.0),
            throttle=rng.uniform(0.58, 0.74),
        )
        if _positions_collide_linear(
            position_a=np.asarray(attacker.position, dtype=np.float32),
            rotation_a=attacker.rotation_degrees,
            speed_a=attacker.initial_speed,
            position_b=np.asarray(defender.position, dtype=np.float32),
            rotation_b=defender.rotation_degrees,
            speed_b=defender.initial_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        if spec.advantaged_role == "fighter2":
            fighter1_spawn = defender
            fighter2_spawn = attacker
        else:
            fighter1_spawn = attacker
            fighter2_spawn = defender
        return GeneratedScene(
            name=name,
            arena=arena,
            fighter1_spawn=fighter1_spawn,
            fighter2_spawn=fighter2_spawn,
            metadata={
                "template": spec.template,
                "advantaged_role": spec.advantaged_role,
                "altitude_profile": spec.altitude_profile,
                "boundary_profile": "near_edge",
                "pair_collision_horizon_seconds": horizon_seconds,
            },
        )
    raise RuntimeError(f"failed to generate safe boundary-return scene for {name}")


def _generate_constrained_random_dogfight(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    horizon_seconds = 3.0
    collision_radius = 5.4
    minimum_distance = 240.0
    maximum_distance = 1_200.0
    immediate_fire_distance = 360.0
    immediate_fire_alignment_cos = 0.97
    horizontal_limit = arena.arena_radius - 180.0
    altitude_low = arena.ground_height + 180.0
    altitude_high = arena.flight_ceiling_height - 180.0

    def sample_speed_and_throttle() -> tuple[float, float]:
        throttle = rng.uniform(0.15, 1.0)
        speed = float(np.interp(throttle, [0.15, 1.0], [32.0, 78.0]).item())
        return speed, throttle

    def has_immediate_fire_geometry(
        shooter_position: np.ndarray,
        shooter_rotation: tuple[float, float, float],
        target_position: np.ndarray,
    ) -> bool:
        offset = target_position - shooter_position
        distance = float(np.linalg.norm(offset))
        if distance > immediate_fire_distance or distance <= 1e-6:
            return False
        alignment = float(np.dot(_forward_from_rotation(shooter_rotation), offset / distance))
        return alignment >= immediate_fire_alignment_cos

    for _ in range(4096):
        radius = horizontal_limit * np.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * np.pi)
        fighter1_position = np.asarray(
            [
                np.cos(angle) * radius,
                rng.uniform(altitude_low, altitude_high),
                np.sin(angle) * radius,
            ],
            dtype=np.float32,
        )
        fighter1_rotation = _sample_uniform_orientation(rng)
        fighter1_speed, fighter1_throttle = sample_speed_and_throttle()
        if not _is_spawn_safe_linear(
            position=fighter1_position,
            rotation_degrees=fighter1_rotation,
            speed=fighter1_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue

        distance = rng.uniform(minimum_distance, maximum_distance)
        fighter2_position = fighter1_position + _sample_relative_direction(rng) * distance
        if not _is_position_inside_arena(
            fighter2_position,
            arena,
            collision_radius=collision_radius,
        ):
            continue
        fighter2_rotation = _sample_uniform_orientation(rng)
        fighter2_speed, fighter2_throttle = sample_speed_and_throttle()
        if not _is_spawn_safe_linear(
            position=fighter2_position,
            rotation_degrees=fighter2_rotation,
            speed=fighter2_speed,
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        if _positions_collide_linear(
            position_a=fighter1_position,
            rotation_a=fighter1_rotation,
            speed_a=fighter1_speed,
            position_b=fighter2_position,
            rotation_b=fighter2_rotation,
            speed_b=fighter2_speed,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        if has_immediate_fire_geometry(
            fighter1_position,
            fighter1_rotation,
            fighter2_position,
        ) or has_immediate_fire_geometry(
            fighter2_position,
            fighter2_rotation,
            fighter1_position,
        ):
            continue

        return GeneratedScene(
            name=name,
            arena=arena,
            fighter1_spawn=_spawn_state(
                position=fighter1_position,
                yaw=fighter1_rotation[1],
                pitch=fighter1_rotation[0],
                roll=fighter1_rotation[2],
                speed=fighter1_speed,
                throttle=fighter1_throttle,
            ),
            fighter2_spawn=_spawn_state(
                position=fighter2_position,
                yaw=fighter2_rotation[1],
                pitch=fighter2_rotation[0],
                roll=fighter2_rotation[2],
                speed=fighter2_speed,
                throttle=fighter2_throttle,
            ),
            metadata={
                "template": spec.template,
                "separation": distance,
                "sampling_horizon_seconds": horizon_seconds,
                "minimum_distance": minimum_distance,
                "maximum_distance": maximum_distance,
                "immediate_fire_distance": immediate_fire_distance,
                "immediate_fire_alignment_cos": immediate_fire_alignment_cos,
            },
        )
    raise RuntimeError(f"failed to generate constrained random dogfight scene for {name}")


def _generate_collision_course(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    collision_radius = 5.4
    combined_collision_radius = collision_radius * 2.0
    min_contact_seconds = 2.0
    max_contact_seconds = 5.0
    horizontal_limit = arena.arena_radius - collision_radius
    altitude_low = arena.ground_height + collision_radius
    altitude_high = arena.flight_ceiling_height - collision_radius

    for _ in range(8192):
        collision_point_radius = horizontal_limit * np.sqrt(rng.random())
        collision_point_angle = rng.uniform(0.0, 2.0 * np.pi)
        collision_point = np.asarray(
            [
                np.cos(collision_point_angle) * collision_point_radius,
                rng.uniform(altitude_low, altitude_high),
                np.sin(collision_point_angle) * collision_point_radius,
            ],
            dtype=np.float32,
        )
        fighter1_forward = _sample_relative_direction(rng)
        fighter2_forward = _sample_relative_direction(rng)
        fighter1_speed, fighter1_throttle = _random_speed_and_throttle(rng)
        fighter2_speed, fighter2_throttle = _random_speed_and_throttle(rng)
        fighter1_velocity = fighter1_forward * fighter1_speed
        fighter2_velocity = fighter2_forward * fighter2_speed
        relative_speed = float(np.linalg.norm(fighter2_velocity - fighter1_velocity))
        if relative_speed <= 1e-3:
            continue

        contact_seconds = rng.uniform(min_contact_seconds, max_contact_seconds)
        center_crossing_seconds = contact_seconds + combined_collision_radius / relative_speed
        fighter1_position = collision_point - fighter1_velocity * center_crossing_seconds
        fighter2_position = collision_point - fighter2_velocity * center_crossing_seconds
        if not _is_position_inside_arena(
            fighter1_position,
            arena,
            collision_radius=collision_radius,
        ):
            continue
        if not _is_position_inside_arena(
            fighter2_position,
            arena,
            collision_radius=collision_radius,
        ):
            continue

        fighter1_rotation = _rotation_from_forward_with_random_roll(rng, fighter1_forward)
        fighter2_rotation = _rotation_from_forward_with_random_roll(rng, fighter2_forward)
        return GeneratedScene(
            name=name,
            arena=arena,
            fighter1_spawn=_spawn_state(
                position=fighter1_position,
                yaw=fighter1_rotation[1],
                pitch=fighter1_rotation[0],
                roll=fighter1_rotation[2],
                speed=fighter1_speed,
                throttle=fighter1_throttle,
            ),
            fighter2_spawn=_spawn_state(
                position=fighter2_position,
                yaw=fighter2_rotation[1],
                pitch=fighter2_rotation[0],
                roll=fighter2_rotation[2],
                speed=fighter2_speed,
                throttle=fighter2_throttle,
            ),
            metadata={
                "template": spec.template,
                "collision_point": [float(value) for value in collision_point],
                "collision_radius_meters": collision_radius,
                "combined_collision_radius_meters": combined_collision_radius,
                "contact_seconds": contact_seconds,
                "center_crossing_seconds": center_crossing_seconds,
                "relative_speed_mps": relative_speed,
            },
        )
    raise RuntimeError(f"failed to generate collision-course scene for {name}")


def _generate_random_recovery(spec: GeneratedSceneSpec, arena: ArenaPreset, rng: random.Random, name: str) -> GeneratedScene:
    horizon_seconds = 2.0
    collision_radius = 5.4
    tactical_radius = RECOVERY_OPPONENT_MAX_DISTANCE_METERS
    recovery_boundary_radius = _recovery_boundary_radius(arena)
    self_altitude = _sample_altitude(rng, arena, spec.altitude_profile)
    for _ in range(256):
        self_anchor_xy = _sample_anchor_xy(rng, arena, boundary_profile=spec.boundary_profile)
        self_position = np.asarray([self_anchor_xy[0], self_altitude, self_anchor_xy[1]], dtype=np.float32)
        self_rotation = _sample_uniform_orientation(rng)
        self_throttle = rng.uniform(0.0, 1.0)
        self_speed = np.interp(self_throttle, [0.0, 1.0], [32.0, 78.0]).item()
        if not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=self_rotation,
            speed=float(self_speed),
            arena=arena,
            horizon_seconds=horizon_seconds,
            collision_radius=collision_radius,
            horizontal_radius_override=recovery_boundary_radius,
        ):
            continue

        self_spawn = _spawn_state(
            position=self_position,
            yaw=self_rotation[1],
            pitch=self_rotation[0],
            roll=self_rotation[2],
            speed=float(self_speed),
            throttle=float(self_throttle),
        )
        for _ in range(256):
            enemy_spawn = _sample_enemy_near_self(
                rng=rng,
                self_position=self_position,
                arena=arena,
                altitude_profile=spec.altitude_profile,
                horizontal_radius_limit=recovery_boundary_radius,
                collision_radius=collision_radius,
                tactical_radius=tactical_radius,
                horizon_seconds=horizon_seconds,
            )
            if _positions_collide_linear(
                position_a=self_position,
                rotation_a=self_rotation,
                speed_a=float(self_speed),
                position_b=np.asarray(enemy_spawn.position, dtype=np.float32),
                rotation_b=enemy_spawn.rotation_degrees,
                speed_b=enemy_spawn.initial_speed,
                horizon_seconds=horizon_seconds,
                collision_radius=collision_radius,
            ):
                continue
            return GeneratedScene(
                name=name,
                arena=arena,
                fighter1_spawn=self_spawn,
                fighter2_spawn=enemy_spawn,
                metadata={
                    "template": spec.template,
                    "altitude_profile": spec.altitude_profile,
                    "boundary_profile": spec.boundary_profile,
                    "sampling_horizon_seconds": horizon_seconds,
                    "recovery_boundary_radius": recovery_boundary_radius,
                },
            )
    raise RuntimeError(f"failed to generate random recovery scene for {name}")


def _generate_ground_recovery_variant(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
    *,
    max_relative_altitude_meters: float,
    safety_horizon_seconds: float | None,
) -> GeneratedScene:
    collision_radius = 5.4
    max_self_altitude = arena.ground_height + max_relative_altitude_meters
    horizontal_margin = collision_radius + 5.0
    max_radius = max(arena.arena_radius - horizontal_margin, 0.0)
    for _ in range(512):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = max_radius * np.sqrt(rng.random())
        self_position = np.asarray(
            [
                np.cos(angle) * radius,
                rng.uniform(arena.ground_height + collision_radius, max_self_altitude),
                np.sin(angle) * radius,
            ],
            dtype=np.float32,
        )
        self_rotation = _sample_uniform_orientation(rng)
        self_speed = rng.uniform(0.0, 80.0)
        self_throttle = rng.uniform(0.0, 1.0)
        if safety_horizon_seconds is not None and not _is_spawn_safe_linear(
            position=self_position,
            rotation_degrees=self_rotation,
            speed=self_speed,
            arena=arena,
            horizon_seconds=safety_horizon_seconds,
            collision_radius=collision_radius,
        ):
            continue
        self_spawn = _spawn_state(
            position=self_position,
            yaw=self_rotation[1],
            pitch=self_rotation[0],
            roll=self_rotation[2],
            speed=self_speed,
            throttle=self_throttle,
        )
        break
    else:
        raise RuntimeError(f"failed to generate ground recovery scene for {name}")

    enemy_spawn = SpawnState(
        position=(220.0, 600.0, 1500.0),
        rotation_degrees=(-90.0, 0.0, 0.0),
        initial_speed=50.0,
        initial_throttle=0.5,
    )
    return GeneratedScene(
        name=name,
        arena=arena,
        fighter1_spawn=self_spawn,
        fighter2_spawn=enemy_spawn,
        metadata={
            "template": spec.template,
            "self_altitude_max_meters": max_relative_altitude_meters,
            "self_collision_radius_meters": collision_radius,
            "enemy_fixed_scene_reference": "open_gr3_f1.ron/fighter2_spawn",
            "safety_horizon_seconds": safety_horizon_seconds,
        },
    )


def _generate_ground_recovery(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    return _generate_ground_recovery_variant(
        spec,
        arena,
        rng,
        name,
        max_relative_altitude_meters=200.0,
        safety_horizon_seconds=None,
    )


def _generate_ground_recovery_extreme(
    spec: GeneratedSceneSpec,
    arena: ArenaPreset,
    rng: random.Random,
    name: str,
) -> GeneratedScene:
    return _generate_ground_recovery_variant(
        spec,
        arena,
        rng,
        name,
        max_relative_altitude_meters=20.0,
        safety_horizon_seconds=2.0,
    )


def generate_scene_from_spec(spec: GeneratedSceneSpec, *, index: int) -> GeneratedScene:
    arena = ARENA_PRESETS[spec.arena_preset]
    rng = random.Random(spec.seed + index * 9973)
    scene_name = f"{spec.name}_{index:03d}"
    if spec.template == "head_on":
        scene = _generate_head_on(spec, arena, rng, scene_name)
    elif spec.template == "safe_repair_pair":
        scene = _generate_safe_repair_pair(spec, arena, rng, scene_name)
    elif spec.template == "close_head_on":
        scene = _generate_close_head_on(spec, arena, rng, scene_name)
    elif spec.template == "aim_fire":
        scene = _generate_aim_fire(spec, arena, rng, scene_name)
    elif spec.template == "tail_chase":
        scene = _generate_tail_chase(spec, arena, rng, scene_name)
    elif spec.template == "centered_tail_chase":
        scene = _generate_centered_tail_chase(spec, arena, rng, scene_name)
    elif spec.template == "centered_being_tailed":
        scene = _generate_centered_being_tailed(spec, arena, rng, scene_name)
    elif spec.template == "side_cut_in":
        scene = _generate_side_cut_in(spec, arena, rng, scene_name)
    elif spec.template == "offset_merge":
        scene = _generate_offset_merge(spec, arena, rng, scene_name)
    elif spec.template == "boundary_return":
        scene = _generate_boundary_return(spec, arena, rng, scene_name)
    elif spec.template == "constrained_random_dogfight":
        scene = _generate_constrained_random_dogfight(spec, arena, rng, scene_name)
    elif spec.template == "collision_course":
        scene = _generate_collision_course(spec, arena, rng, scene_name)
    elif spec.template == "random_recovery":
        scene = _generate_random_recovery(spec, arena, rng, scene_name)
    elif spec.template == "mild_oob_recoverable":
        scene = _generate_mild_oob_recoverable(spec, arena, rng, scene_name)
    elif spec.template == "low_altitude_recoverable":
        scene = _generate_low_altitude_recoverable(spec, arena, rng, scene_name)
    elif spec.template == "near_boundary_recoverable":
        scene = _generate_near_boundary_recoverable(spec, arena, rng, scene_name)
    elif spec.template == "ground_recovery":
        scene = _generate_ground_recovery(spec, arena, rng, scene_name)
    elif spec.template == "ground_recovery_extreme":
        scene = _generate_ground_recovery_extreme(spec, arena, rng, scene_name)
    else:
        raise ValueError(f"unsupported tactical scene template: {spec.template}")
    return _apply_damage_profile_to_scene(scene, spec=spec, rng=rng)


def materialize_scene_pool(
    *,
    spec: ScenePoolSpec,
    output_dir: Path,
    project_root: Path | None = None,
    reuse_existing: bool = False,
) -> PreparedScenePool:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    generated_dir = output_dir / "generated"
    existing_dir = output_dir / "existing"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    existing_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "scene_pool_manifest.json"

    def prepared_from_manifest() -> PreparedScenePool:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        def load_scenes(key: str) -> list[MaterializedScene]:
            scenes = [MaterializedScene(**item) for item in payload[key]]
            for scene in scenes:
                if scene.scene_path is None:
                    raise ValueError(
                        f"frozen scene pool entry has no scene_path: {scene.label}"
                    )
                path = Path(scene.scene_path)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"frozen scene pool asset does not exist: {path}"
                    )
                expected_sha = (scene.metadata or {}).get("scene_sha256")
                if expected_sha is not None and _sha256_file(path) != expected_sha:
                    raise ValueError(f"frozen scene pool asset hash mismatch: {path}")
            return scenes

        return PreparedScenePool(
            train_tactical_scenes=load_scenes("train_tactical_scenes"),
            train_recovery_scenes=load_scenes("train_recovery_scenes"),
            train_tactical_ratio=float(payload["train_tactical_ratio"]),
            train_recovery_ratio=float(payload["train_recovery_ratio"]),
            eval_scenes=load_scenes("eval_scenes"),
            rng_seed=17,
        )

    if reuse_existing:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "cannot resume PPO because the frozen scene-pool manifest is missing: "
                f"{manifest_path}"
            )
        return prepared_from_manifest()

    resolved_project_root = (project_root or Path.cwd()).expanduser().resolve()

    def scene_metadata(
        metadata: dict[str, Any] | None,
        *,
        path: Path,
        source_path: Path | None = None,
    ) -> dict[str, Any]:
        payload = {
            **(metadata or {}),
            "scene_sha256": _sha256_file(path),
        }
        if source_path is not None:
            payload["source_path"] = str(source_path.resolve())
        return payload

    def materialize_generated(items: list[GeneratedSceneSpec]) -> list[MaterializedScene]:
        materialized: list[MaterializedScene] = []
        for item in items:
            per_scene_weight = item.weight / max(item.count, 1)
            for index in range(item.count):
                scene = generate_scene_from_spec(item, index=index)
                scene_path = generated_dir / f"{scene.name}.ron"
                scene_path.write_text(scene.to_ron(), encoding="utf-8")
                materialized.append(
                    MaterializedScene(
                        label=scene.name,
                        weight=per_scene_weight,
                        scene_name=scene.name,
                        scene_path=str(scene_path.resolve()),
                        generated=True,
                        metadata=scene_metadata(scene.metadata, path=scene_path),
                    )
                )
        return materialized

    def materialize_existing(items: list[ExistingSceneSpec]) -> list[MaterializedScene]:
        materialized: list[MaterializedScene] = []
        for item in items:
            source_path = (
                resolved_project_root
                / "config"
                / "dfb_game"
                / "scenes"
                / f"{item.scene_name}.ron"
            )
            if not source_path.is_file():
                raise FileNotFoundError(f"named scene does not exist: {source_path}")
            scene_path = existing_dir / f"{item.scene_name}.ron"
            shutil.copyfile(source_path, scene_path)
            materialized.append(
                MaterializedScene(
                    label=item.scene_name,
                    weight=item.weight,
                    scene_name=item.scene_name,
                    scene_path=str(scene_path.resolve()),
                    generated=False,
                    metadata=scene_metadata(
                        None,
                        path=scene_path,
                        source_path=source_path,
                    ),
                )
            )
        return materialized

    train_tactical_scenes = materialize_existing(spec.train_tactical_existing)
    train_tactical_scenes.extend(materialize_generated(spec.train_tactical_generated))
    train_recovery_scenes = materialize_existing(spec.train_recovery_existing)
    train_recovery_scenes.extend(materialize_generated(spec.train_recovery_generated))
    eval_scenes = materialize_existing(spec.eval_existing)
    eval_scenes.extend(materialize_generated(spec.eval_generated))
    prepared = PreparedScenePool(
        train_tactical_scenes=train_tactical_scenes,
        train_recovery_scenes=train_recovery_scenes,
        train_tactical_ratio=spec.train_tactical_ratio,
        train_recovery_ratio=spec.train_recovery_ratio,
        eval_scenes=eval_scenes or train_tactical_scenes or train_recovery_scenes,
        rng_seed=17,
    )
    manifest_payload = {
        "train_tactical_scenes": [asdict(scene) for scene in prepared.train_tactical_scenes],
        "train_recovery_scenes": [asdict(scene) for scene in prepared.train_recovery_scenes],
        "train_tactical_ratio": prepared.train_tactical_ratio,
        "train_recovery_ratio": prepared.train_recovery_ratio,
        "eval_scenes": [asdict(scene) for scene in prepared.eval_scenes],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return prepared


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
