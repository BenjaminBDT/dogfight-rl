from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dfb_reinforcement_learning.scenes import ScenePoolSpec, materialize_scene_pool
from dfb_reinforcement_learning.scenes.tactical_scene_generator import (
    ExistingSceneSpec,
    GeneratedSceneSpec,
    _forward_from_rotation,
    generate_scene_from_spec,
    _is_spawn_safe_linear,
    _positions_collide_linear,
)


def _forward_from_spawn_rotation(spawn) -> tuple[float, float, float]:
    forward = _forward_from_rotation(spawn.rotation_degrees)
    return (float(forward[0]), float(forward[1]), float(forward[2]))


def test_generate_head_on_scene_renders_ron() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="headon_test",
            template="head_on",
            count=1,
            seed=7,
        ),
        index=0,
    )
    ron = scene.to_ron()
    assert "fighter1_spawn" in ron
    assert "fighter2_spawn" in ron
    assert "obstacles: []" in ron
    assert "initial_damage: None" in ron


def test_generate_scene_with_uniform_damage_profile_emits_spawn_damage() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="damaged_headon_test",
            template="head_on",
            count=1,
            seed=13,
            damage_profile="uniform_full_range",
        ),
        index=0,
    )
    ron = scene.to_ron()
    assert "initial_damage: Some((" in ron
    assert "left_wing_fraction: Some(" in ron
    assert scene.fighter1_spawn.initial_damage is not None
    assert scene.fighter2_spawn.initial_damage is not None
    for damage in (scene.fighter1_spawn.initial_damage, scene.fighter2_spawn.initial_damage):
        assert damage is not None
        for value in (
            damage.left_wing_fraction,
            damage.right_wing_fraction,
            damage.pitch_tail_fraction,
            damage.yaw_tail_fraction,
            damage.engine_fraction,
        ):
            assert value is not None
            assert 0.0 <= value <= 1.0


def test_combat_wear_prioritizes_total_hp_and_keeps_component_damage_sparse() -> None:
    damaged_aircraft = 0
    component_damaged_aircraft = 0
    low_health_aircraft = 0
    for index in range(400):
        scene = generate_scene_from_spec(
            GeneratedSceneSpec(
                name="combat_wear_test",
                template="head_on",
                count=400,
                seed=101,
                damage_profile="combat_wear",
            ),
            index=index,
        )
        for spawn in (scene.fighter1_spawn, scene.fighter2_spawn):
            damage = spawn.initial_damage
            if damage is None:
                continue
            damaged_aircraft += 1
            assert damage.total_hit_points_fraction is not None
            assert 0.35 <= damage.total_hit_points_fraction <= 0.95
            if damage.total_hit_points_fraction < 0.60:
                low_health_aircraft += 1
            subsystem_values = (
                damage.left_wing_fraction,
                damage.right_wing_fraction,
                damage.pitch_tail_fraction,
                damage.yaw_tail_fraction,
                damage.engine_fraction,
            )
            present = [value for value in subsystem_values if value is not None]
            assert len(present) <= 1
            if present:
                component_damaged_aircraft += 1
                assert 0.50 <= present[0] <= 0.85

    assert 150 <= damaged_aircraft <= 250
    assert 20 <= low_health_aircraft <= 70
    assert component_damaged_aircraft <= 35


def test_safe_repair_pair_is_damaged_low_threat_and_boundary_safe() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="safe_repair_test",
            template="safe_repair_pair",
            count=1,
            seed=17,
            damage_profile="repairable_pair",
        ),
        index=0,
    )
    fighter1 = scene.fighter1_spawn
    fighter2 = scene.fighter2_spawn
    assert "total_hit_points_fraction: Some(" in scene.to_ron()
    separation = np.linalg.norm(np.asarray(fighter1.position) - np.asarray(fighter2.position))
    assert 1_000.0 <= separation <= 1_400.0
    assert np.allclose(
        _forward_from_rotation(fighter1.rotation_degrees),
        _forward_from_rotation(fighter2.rotation_degrees),
    )
    for spawn in (fighter1, fighter2):
        damage = spawn.initial_damage
        assert damage is not None
        subsystem_fractions = (
            damage.left_wing_fraction,
            damage.right_wing_fraction,
            damage.pitch_tail_fraction,
            damage.yaw_tail_fraction,
            damage.engine_fraction,
        )
        assert sum(value is not None and value <= 0.5 for value in subsystem_fractions) == 1
        assert _is_spawn_safe_linear(
            position=np.asarray(spawn.position, dtype=np.float32),
            rotation_degrees=spawn.rotation_degrees,
            speed=spawn.initial_speed,
            arena=scene.arena,
            horizon_seconds=11.0,
            collision_radius=5.4,
        )


def test_materialize_scene_pool_writes_generated_scene_files(tmp_path: Path) -> None:
    pool_json = tmp_path / "scene_pool.json"
    pool_json.write_text(
        json.dumps(
            {
                "train": {
                    "tactical_ratio": 0.75,
                    "recovery_ratio": 0.25,
                    "tactical": {
                        "existing": [{"scene_name": "open_head_on_200m", "weight": 0.5}],
                        "generated": [
                            {
                                "name": "tail_pool",
                                "template": "tail_chase",
                                "count": 2,
                                "weight": 1.0,
                                "seed": 11,
                                "advantaged_role": "fighter1"
                            }
                        ]
                    },
                    "recovery": {
                        "generated": [
                            {
                                "name": "recovery_pool",
                                "template": "random_recovery",
                                "count": 1,
                                "weight": 1.0,
                                "seed": 23,
                                "altitude_profile": "low",
                                "boundary_profile": "near_edge",
                                "damage_profile": "uniform_full_range"
                            }
                        ]
                    }
                },
                "eval": {
                    "generated": [
                        {
                            "name": "eval_headon",
                            "template": "head_on",
                            "count": 1,
                            "weight": 1.0,
                            "seed": 19,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    spec = ScenePoolSpec.from_json(pool_json)
    prepared = materialize_scene_pool(
        spec=spec,
        output_dir=tmp_path / "materialized",
        project_root=Path.cwd(),
    )

    assert len(prepared.train_tactical_scenes) == 3
    assert len(prepared.train_recovery_scenes) == 1
    assert any(scene.scene_name == "open_head_on_200m" for scene in prepared.train_tactical_scenes)
    existing_scene = next(scene for scene in prepared.train_tactical_scenes if not scene.generated)
    assert existing_scene.scene_path is not None
    assert Path(existing_scene.scene_path).is_file()
    assert existing_scene.metadata is not None
    assert len(existing_scene.metadata["scene_sha256"]) == 64
    generated_paths = [scene.scene_path for scene in prepared.train_tactical_scenes if scene.generated]
    assert len(generated_paths) == 2
    assert all(path is not None and Path(path).is_file() for path in generated_paths)
    assert prepared.train_recovery_scenes[0].generated is True
    assert prepared.eval_scenes[0].generated is True
    assert prepared.train_tactical_ratio == 0.75
    assert prepared.train_recovery_ratio == 0.25
    recovery_scene_path = Path(prepared.train_recovery_scenes[0].scene_path or "")
    assert "initial_damage: Some((" in recovery_scene_path.read_text(encoding="utf-8")


def test_materialize_scene_pool_reuses_existing_snapshot(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "config" / "dfb_game" / "scenes"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "fixture.ron"
    source_path.write_text("(revision: 1)\n", encoding="utf-8")
    spec = ScenePoolSpec(
        train_tactical_generated=[],
        train_tactical_existing=[ExistingSceneSpec(scene_name="fixture")],
        train_recovery_generated=[],
        train_recovery_existing=[],
        train_tactical_ratio=1.0,
        train_recovery_ratio=0.0,
        eval_generated=[],
        eval_existing=[],
    )
    output_dir = tmp_path / "materialized"

    prepared = materialize_scene_pool(
        spec=spec,
        output_dir=output_dir,
        project_root=project_root,
    )
    snapshot_path = Path(prepared.train_tactical_scenes[0].scene_path or "")
    source_path.write_text("(revision: 2)\n", encoding="utf-8")
    resumed = materialize_scene_pool(
        spec=spec,
        output_dir=output_dir,
        project_root=project_root,
        reuse_existing=True,
    )

    assert Path(resumed.train_tactical_scenes[0].scene_path or "") == snapshot_path
    assert snapshot_path.read_text(encoding="utf-8") == "(revision: 1)\n"


def test_general_scene_pool_v2_has_stable_weights_and_disjoint_generation_seeds() -> None:
    pool_path = Path("config/dfb_reinforcement_learning/scene_pools/part3_train_scene_pool_v2.json")
    spec = ScenePoolSpec.from_json(pool_path)

    tactical_weight = sum(item.weight for item in spec.train_tactical_existing) + sum(
        item.weight for item in spec.train_tactical_generated
    )
    recovery_weight = sum(item.weight for item in spec.train_recovery_existing) + sum(
        item.weight for item in spec.train_recovery_generated
    )
    assert tactical_weight == 75.0
    assert recovery_weight == 25.0
    assert spec.train_tactical_ratio == 0.75
    assert spec.train_recovery_ratio == 0.25

    required_anchors = {
        "open_ho",
        "open_tc_f1",
        "open_tc_f2",
        "open_gr_f1",
    }
    train_existing = {
        item.scene_name
        for item in spec.train_tactical_existing + spec.train_recovery_existing
    }
    eval_existing = {item.scene_name for item in spec.eval_existing}
    assert required_anchors <= train_existing
    assert required_anchors == eval_existing

    train_generated = spec.train_tactical_generated + spec.train_recovery_generated
    train_seeds = {item.seed for item in train_generated}
    eval_seeds = {item.seed for item in spec.eval_generated}
    assert len(train_seeds) == len(train_generated)
    assert len(eval_seeds) == len(spec.eval_generated)
    assert train_seeds.isdisjoint(eval_seeds)
    assert sum(item.count for item in train_generated) == 1_520
    assert sum(item.count for item in spec.eval_generated) == 30

    constrained_random = [
        item for item in train_generated if item.template == "constrained_random_dogfight"
    ]
    assert len(constrained_random) == 1
    assert constrained_random[0].weight == 5.0
    assert constrained_random[0].count == 192


def test_general_scene_pool_v3_reserves_ten_percent_for_collision_courses() -> None:
    pool_path = Path("config/dfb_reinforcement_learning/scene_pools/part3_train_scene_pool_v3.json")
    spec = ScenePoolSpec.from_json(pool_path)

    tactical_weight = sum(item.weight for item in spec.train_tactical_existing) + sum(
        item.weight for item in spec.train_tactical_generated
    )
    recovery_weight = sum(item.weight for item in spec.train_recovery_existing) + sum(
        item.weight for item in spec.train_recovery_generated
    )
    collision_courses = [
        item for item in spec.train_tactical_generated if item.template == "collision_course"
    ]
    assert tactical_weight == 75.0
    assert recovery_weight == 25.0
    assert spec.train_tactical_ratio == 0.5
    assert spec.train_recovery_ratio == 0.5
    assert len(collision_courses) == 1
    assert collision_courses[0].weight == 15.0
    assert collision_courses[0].count == 192
    assert spec.train_tactical_ratio * collision_courses[0].weight / tactical_weight == 0.10

    train_generated = spec.train_tactical_generated + spec.train_recovery_generated
    train_seeds = {item.seed for item in train_generated}
    eval_seeds = {item.seed for item in spec.eval_generated}
    assert len(train_seeds) == len(train_generated)
    assert len(eval_seeds) == len(spec.eval_generated)
    assert train_seeds.isdisjoint(eval_seeds)
    assert any(item.template == "collision_course" for item in spec.eval_generated)


def test_random_recovery_scene_stays_within_ceiling_and_arena() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="recovery_test",
            template="random_recovery",
            count=1,
            seed=29,
            altitude_profile="low",
            boundary_profile="near_edge",
        ),
        index=0,
    )
    assert scene.arena.flight_ceiling_height == 2000.0
    for spawn in (scene.fighter1_spawn, scene.fighter2_spawn):
        assert scene.arena.ground_height + 5.4 < spawn.position[1] < scene.arena.flight_ceiling_height - 5.4
        assert (spawn.position[0] ** 2 + spawn.position[2] ** 2) ** 0.5 < scene.metadata["recovery_boundary_radius"]
    distance = (
        (scene.fighter1_spawn.position[0] - scene.fighter2_spawn.position[0]) ** 2
        + (scene.fighter1_spawn.position[1] - scene.fighter2_spawn.position[1]) ** 2
        + (scene.fighter1_spawn.position[2] - scene.fighter2_spawn.position[2]) ** 2
    ) ** 0.5
    assert distance <= 500.0 + 1e-6


def test_mild_oob_recovery_scene_allows_spawn_outside_battle_arena_but_inside_recovery_boundary() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="mild_oob_test",
            template="mild_oob_recoverable",
            count=1,
            seed=41,
            altitude_profile="mid",
            boundary_profile="near_edge",
        ),
        index=0,
    )
    fighter1_radius = (scene.fighter1_spawn.position[0] ** 2 + scene.fighter1_spawn.position[2] ** 2) ** 0.5
    assert fighter1_radius > scene.arena.arena_radius
    assert fighter1_radius < scene.metadata["recovery_boundary_radius"]
    assert scene.metadata["outside_arena_distance"] > 0.0


def test_near_boundary_recovery_scene_stays_within_recovery_boundary() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="near_boundary_test",
            template="near_boundary_recoverable",
            count=1,
            seed=53,
            altitude_profile="mid",
            boundary_profile="near_edge",
            arena_preset="boundary_pressure",
        ),
        index=0,
    )
    for spawn in (scene.fighter1_spawn, scene.fighter2_spawn):
        radius = (spawn.position[0] ** 2 + spawn.position[2] ** 2) ** 0.5
        assert radius < scene.metadata["recovery_boundary_radius"]


def test_centered_tail_chase_places_center_role_at_world_center_and_enemy_ahead() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="centered_tail_test",
            template="centered_tail_chase",
            count=1,
            seed=73,
            center_role="fighter2",
            altitude_profile="mid",
        ),
        index=0,
    )
    center = scene.fighter2_spawn
    enemy = scene.fighter1_spawn
    assert center.position[0] == 0.0
    assert center.position[2] == 0.0
    center_forward = _forward_from_spawn_rotation(center)
    rel = (
        enemy.position[0] - center.position[0],
        enemy.position[1] - center.position[1],
        enemy.position[2] - center.position[2],
    )
    assert center_forward[0] * rel[0] + center_forward[2] * rel[2] > 0.0
    enemy_forward = _forward_from_spawn_rotation(enemy)
    enemy_to_center = (
        center.position[0] - enemy.position[0],
        center.position[1] - enemy.position[1],
        center.position[2] - enemy.position[2],
    )
    assert enemy_forward[0] * enemy_to_center[0] + enemy_forward[2] * enemy_to_center[2] < 0.0
    assert (rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) ** 0.5 <= 500.0


def test_centered_being_tailed_places_enemy_behind_and_facing_center() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="centered_defense_test",
            template="centered_being_tailed",
            count=1,
            seed=79,
            center_role="fighter1",
            altitude_profile="mid",
        ),
        index=0,
    )
    center = scene.fighter1_spawn
    enemy = scene.fighter2_spawn
    assert center.position[0] == 0.0
    assert center.position[2] == 0.0
    center_forward = _forward_from_spawn_rotation(center)
    rel = (
        enemy.position[0] - center.position[0],
        enemy.position[1] - center.position[1],
        enemy.position[2] - center.position[2],
    )
    assert center_forward[0] * rel[0] + center_forward[2] * rel[2] < 0.0
    enemy_forward = _forward_from_spawn_rotation(enemy)
    enemy_to_center = (
        center.position[0] - enemy.position[0],
        center.position[1] - enemy.position[1],
        center.position[2] - enemy.position[2],
    )
    assert enemy_forward[0] * enemy_to_center[0] + enemy_forward[2] * enemy_to_center[2] > 0.0
    assert (rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) ** 0.5 <= 500.0


def test_tail_chase_places_enemy_in_self_front_hemisphere_and_self_in_enemy_rear_hemisphere() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="tail_chase_test",
            template="tail_chase",
            count=1,
            seed=181,
            advantaged_role="fighter1",
            altitude_profile="mid",
            boundary_profile="center",
        ),
        index=0,
    )
    self_spawn = scene.fighter1_spawn
    enemy_spawn = scene.fighter2_spawn
    self_forward = _forward_from_spawn_rotation(self_spawn)
    rel = (
        enemy_spawn.position[0] - self_spawn.position[0],
        enemy_spawn.position[1] - self_spawn.position[1],
        enemy_spawn.position[2] - self_spawn.position[2],
    )
    assert self_forward[0] * rel[0] + self_forward[1] * rel[1] + self_forward[2] * rel[2] > 0.0
    distance = (rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2) ** 0.5
    assert 80.0 <= distance <= 400.0

    enemy_forward = _forward_from_spawn_rotation(enemy_spawn)
    enemy_to_self = (
        self_spawn.position[0] - enemy_spawn.position[0],
        self_spawn.position[1] - enemy_spawn.position[1],
        self_spawn.position[2] - enemy_spawn.position[2],
    )
    assert (
        enemy_forward[0] * enemy_to_self[0]
        + enemy_forward[1] * enemy_to_self[1]
        + enemy_forward[2] * enemy_to_self[2]
        < 0.0
    )
    assert scene.metadata["self_safe_horizon_seconds"] == 2.0
    assert scene.metadata["enemy_safe_horizon_seconds"] == 2.0
    assert _is_spawn_safe_linear(
        position=np.asarray(self_spawn.position, dtype=np.float32),
        rotation_degrees=self_spawn.rotation_degrees,
        speed=self_spawn.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
    assert _is_spawn_safe_linear(
        position=np.asarray(enemy_spawn.position, dtype=np.float32),
        rotation_degrees=enemy_spawn.rotation_degrees,
        speed=enemy_spawn.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )


def test_boundary_return_rejects_two_second_linear_pair_collisions() -> None:
    for index in range(64):
        scene = generate_scene_from_spec(
            GeneratedSceneSpec(
                name="boundary_return_test",
                template="boundary_return",
                count=64,
                seed=301,
                altitude_profile="mid",
                boundary_profile="near_edge",
                arena_preset="boundary_pressure",
                advantaged_role="fighter1",
            ),
            index=index,
        )
        fighter1 = scene.fighter1_spawn
        fighter2 = scene.fighter2_spawn
        assert not _positions_collide_linear(
            position_a=np.asarray(fighter1.position, dtype=np.float32),
            rotation_a=fighter1.rotation_degrees,
            speed_a=fighter1.initial_speed,
            position_b=np.asarray(fighter2.position, dtype=np.float32),
            rotation_b=fighter2.rotation_degrees,
            speed_b=fighter2.initial_speed,
            horizon_seconds=2.0,
            collision_radius=5.4,
        )


def test_constrained_random_dogfight_is_safe_and_not_an_immediate_fire_window() -> None:
    for index in range(96):
        scene = generate_scene_from_spec(
            GeneratedSceneSpec(
                name="constrained_random_test",
                template="constrained_random_dogfight",
                count=96,
                seed=17001,
                arena_preset="standard_open",
                damage_profile="combat_wear",
            ),
            index=index,
        )
        fighter1 = scene.fighter1_spawn
        fighter2 = scene.fighter2_spawn
        fighter1_position = np.asarray(fighter1.position, dtype=np.float32)
        fighter2_position = np.asarray(fighter2.position, dtype=np.float32)
        separation = float(np.linalg.norm(fighter2_position - fighter1_position))
        assert 240.0 <= separation <= 1_200.0
        for spawn, position in (
            (fighter1, fighter1_position),
            (fighter2, fighter2_position),
        ):
            assert _is_spawn_safe_linear(
                position=position,
                rotation_degrees=spawn.rotation_degrees,
                speed=spawn.initial_speed,
                arena=scene.arena,
                horizon_seconds=3.0,
                collision_radius=5.4,
            )
        assert not _positions_collide_linear(
            position_a=fighter1_position,
            rotation_a=fighter1.rotation_degrees,
            speed_a=fighter1.initial_speed,
            position_b=fighter2_position,
            rotation_b=fighter2.rotation_degrees,
            speed_b=fighter2.initial_speed,
            horizon_seconds=3.0,
            collision_radius=5.4,
        )
        if separation <= 360.0:
            direction_1_to_2 = (fighter2_position - fighter1_position) / separation
            direction_2_to_1 = -direction_1_to_2
            assert (
                float(np.dot(_forward_from_rotation(fighter1.rotation_degrees), direction_1_to_2))
                < 0.97
            )
            assert (
                float(np.dot(_forward_from_rotation(fighter2.rotation_degrees), direction_2_to_1))
                < 0.97
            )


def test_collision_course_reaches_combined_collision_radius_at_sampled_time() -> None:
    for index in range(96):
        scene = generate_scene_from_spec(
            GeneratedSceneSpec(
                name="collision_course_test",
                template="collision_course",
                count=96,
                seed=26001,
                arena_preset="standard_open",
            ),
            index=index,
        )
        fighter1 = scene.fighter1_spawn
        fighter2 = scene.fighter2_spawn
        fighter1_position = np.asarray(fighter1.position, dtype=np.float32)
        fighter2_position = np.asarray(fighter2.position, dtype=np.float32)
        fighter1_velocity = (
            _forward_from_rotation(fighter1.rotation_degrees) * fighter1.initial_speed
        )
        fighter2_velocity = (
            _forward_from_rotation(fighter2.rotation_degrees) * fighter2.initial_speed
        )
        contact_seconds = float(scene.metadata["contact_seconds"])
        fighter1_at_contact = fighter1_position + fighter1_velocity * contact_seconds
        fighter2_at_contact = fighter2_position + fighter2_velocity * contact_seconds
        contact_distance = float(np.linalg.norm(fighter2_at_contact - fighter1_at_contact))

        assert 2.0 <= contact_seconds <= 5.0
        assert np.isclose(
            contact_distance,
            float(scene.metadata["combined_collision_radius_meters"]),
            atol=2e-3,
        )
        assert _positions_collide_linear(
            position_a=fighter1_position,
            rotation_a=fighter1.rotation_degrees,
            speed_a=fighter1.initial_speed,
            position_b=fighter2_position,
            rotation_b=fighter2.rotation_degrees,
            speed_b=fighter2.initial_speed,
            horizon_seconds=contact_seconds + 1e-3,
            collision_radius=float(scene.metadata["collision_radius_meters"]),
        )
        for spawn, position in (
            (fighter1, fighter1_position),
            (fighter2, fighter2_position),
        ):
            assert scene.arena.ground_height + 5.4 <= position[1]
            assert position[1] <= scene.arena.flight_ceiling_height - 5.4
            assert float(np.linalg.norm(position[[0, 2]])) <= scene.arena.arena_radius - 5.4


def test_close_head_on_keeps_both_aircraft_in_front_hemispheres_and_safe() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="close_head_on_test",
            template="close_head_on",
            count=1,
            seed=191,
            altitude_profile="mid",
            boundary_profile="center",
        ),
        index=0,
    )
    fighter1 = scene.fighter1_spawn
    fighter2 = scene.fighter2_spawn
    f1_position = np.asarray(fighter1.position, dtype=np.float32)
    f2_position = np.asarray(fighter2.position, dtype=np.float32)
    f1_to_f2 = f2_position - f1_position
    separation = float(np.linalg.norm(f1_to_f2))
    assert 10.0 <= separation <= 80.0
    f1_to_f2_dir = f1_to_f2 / max(separation, 1e-6)
    f2_to_f1_dir = -f1_to_f2_dir
    assert float(np.dot(_forward_from_rotation(fighter1.rotation_degrees), f1_to_f2_dir)) >= 0.15
    assert float(np.dot(_forward_from_rotation(fighter2.rotation_degrees), f2_to_f1_dir)) >= 0.15
    assert scene.metadata["pair_collision_horizon_seconds"] == 2.0
    assert _is_spawn_safe_linear(
        position=f1_position,
        rotation_degrees=fighter1.rotation_degrees,
        speed=fighter1.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
    assert _is_spawn_safe_linear(
        position=f2_position,
        rotation_degrees=fighter2.rotation_degrees,
        speed=fighter2.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
    assert not _positions_collide_linear(
        position_a=f1_position,
        rotation_a=fighter1.rotation_degrees,
        speed_a=fighter1.initial_speed,
        position_b=f2_position,
        rotation_b=fighter2.rotation_degrees,
        speed_b=fighter2.initial_speed,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )


def test_aim_fire_places_enemy_on_self_fire_line_and_keeps_both_safe() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="aim_fire_test",
            template="aim_fire",
            count=1,
            seed=223,
            advantaged_role="fighter1",
            altitude_profile="mid",
            boundary_profile="center",
        ),
        index=0,
    )
    self_spawn = scene.fighter1_spawn
    enemy_spawn = scene.fighter2_spawn
    self_forward = _forward_from_spawn_rotation(self_spawn)
    rel = np.asarray(
        [
            enemy_spawn.position[0] - self_spawn.position[0],
            enemy_spawn.position[1] - self_spawn.position[1],
            enemy_spawn.position[2] - self_spawn.position[2],
        ],
        dtype=np.float32,
    )
    distance = float(np.linalg.norm(rel))
    assert 10.0 <= distance <= 500.0
    rel_dir = rel / max(distance, 1e-6)
    assert float(np.dot(self_forward, rel_dir)) > 0.999
    assert scene.metadata["self_safe_horizon_seconds"] == 2.0
    assert scene.metadata["enemy_safe_horizon_seconds"] == 2.0
    assert _is_spawn_safe_linear(
        position=np.asarray(self_spawn.position, dtype=np.float32),
        rotation_degrees=self_spawn.rotation_degrees,
        speed=self_spawn.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
    assert _is_spawn_safe_linear(
        position=np.asarray(enemy_spawn.position, dtype=np.float32),
        rotation_degrees=enemy_spawn.rotation_degrees,
        speed=enemy_spawn.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
    assert not _positions_collide_linear(
        position_a=np.asarray(self_spawn.position, dtype=np.float32),
        rotation_a=self_spawn.rotation_degrees,
        speed_a=self_spawn.initial_speed,
        position_b=np.asarray(enemy_spawn.position, dtype=np.float32),
        rotation_b=enemy_spawn.rotation_degrees,
        speed_b=enemy_spawn.initial_speed,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )


def test_ground_recovery_places_fighter1_low_and_fighter2_at_fixed_vertical_spawn() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="ground_recovery_test",
            template="ground_recovery",
            count=1,
            seed=97,
        ),
        index=0,
    )
    assert scene.fighter2_spawn.position == (220.0, 600.0, 1500.0)
    assert scene.fighter2_spawn.rotation_degrees == (-90.0, 0.0, 0.0)
    assert scene.fighter2_spawn.initial_speed == 50.0
    assert scene.fighter2_spawn.initial_throttle == 0.5
    fighter1 = scene.fighter1_spawn
    assert scene.arena.ground_height + 5.4 <= fighter1.position[1] <= scene.arena.ground_height + 200.0
    assert (fighter1.position[0] ** 2 + fighter1.position[2] ** 2) ** 0.5 < scene.arena.arena_radius
    assert 0.0 <= fighter1.initial_speed <= 80.0
    assert 0.0 <= fighter1.initial_throttle <= 1.0


def test_ground_recovery_extreme_keeps_fighter1_within_20m_and_safe_for_2_seconds() -> None:
    scene = generate_scene_from_spec(
        GeneratedSceneSpec(
            name="ground_recovery_extreme_test",
            template="ground_recovery_extreme",
            count=1,
            seed=197,
        ),
        index=0,
    )
    fighter1 = scene.fighter1_spawn
    assert scene.arena.ground_height + 5.4 <= fighter1.position[1] <= scene.arena.ground_height + 20.0
    assert scene.metadata["safety_horizon_seconds"] == 2.0
    assert _is_spawn_safe_linear(
        position=np.asarray(fighter1.position, dtype=np.float32),
        rotation_degrees=fighter1.rotation_degrees,
        speed=fighter1.initial_speed,
        arena=scene.arena,
        horizon_seconds=2.0,
        collision_radius=5.4,
    )
