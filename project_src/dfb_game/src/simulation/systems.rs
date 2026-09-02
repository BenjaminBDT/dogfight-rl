use bevy::camera::visibility::RenderLayers;
use bevy::{gltf::GltfAssetLabel, prelude::*};

use crate::api::vision::{SEMANTIC_RENDER_LAYER, SemanticCaptureMode};
use crate::app::{AppMode, HeadlessMode};
use crate::audio::{AudioEventKind, AudioEventQueue};
use crate::bridge::protocol::BridgeRole;
use crate::bridge::{
    AssignedControlRole, BridgeClientInbox, BridgeEnabled, BridgeLinkState, BridgeMode,
    BridgeRemoteInterpolationState, BridgeServerSessions, RequestedControlRole,
    bridge_slot_aircraft_role, players_ready, sample_remote_aircraft_snapshot,
};
use crate::core::config::RepositoryConfig;
use crate::gameplay::damage::{
    AircraftDamageState, DamageStage, apply_spawn_damage_overrides, damage_flight_modifiers,
    effective_aircraft_performance,
};
use crate::input::actions::ControlInput;
use crate::presentation::hud::DamageIndicatorQueue;
use crate::simulation::collision::ObstacleCollider;
use crate::simulation::collision::{
    aircraft_world_collision_boxes, apply_world_bounds, obb_aabb_penetration,
    obb_plane_y_penetration,
};
use crate::simulation::components::{
    AircraftPerformance, AircraftRole, AircraftState, ControlAuthority, GunState, SpawnTransform,
};
use crate::simulation::flight_model::step_aircraft;
use crate::simulation::resources::SimulationDebugState;

const LOCAL_GROUND_BOUNCE_MIN_SPEED: f32 = 20.0;
const LOCAL_OBSTACLE_BOUNCE_MIN_SPEED: f32 = 20.0;

#[derive(Default)]
pub struct RenderStateDebugLog {
    next_log_at_seconds: f64,
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AircraftVisualPart {
    Body,
    Cockpit,
    Engine,
    LeftWing,
    RightWing,
    PitchTail,
    YawTail,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct DamageVisualStyle {
    intact: [f32; 4],
    damaged: [f32; 4],
    destroyed: [f32; 4],
}

#[derive(Component, Debug, Clone, Copy)]
pub struct AircraftVisualPalette {
    fuselage: [f32; 4],
    damaged_fuselage: [f32; 4],
    accent: [f32; 4],
    damaged_accent: [f32; 4],
}

#[derive(Component)]
pub struct AircraftVisualSceneRoot;

#[derive(Component)]
pub struct AircraftVisualPartsInitialized;

#[derive(Component, Clone)]
pub struct AircraftVisualSceneHandle(pub Handle<Scene>);

#[derive(Component)]
pub struct AircraftSemanticSceneRoot;

#[derive(Component)]
pub struct AircraftSemanticPartsInitialized;

#[derive(Component, Debug, Clone, Copy)]
pub struct AircraftSemanticClassColor {
    color: [f32; 4],
}

pub fn spawn_placeholder_aircraft(
    mut commands: Commands,
    config: Res<RepositoryConfig>,
    headless_mode: Res<HeadlessMode>,
    bridge_mode: Option<Res<BridgeMode>>,
    semantic_capture_mode: Option<Res<SemanticCaptureMode>>,
    asset_server: Res<AssetServer>,
) {
    spawn_aircraft_entities(
        &mut commands,
        &config,
        headless_mode.0,
        bridge_mode.as_deref(),
        semantic_capture_mode.is_some_and(|mode| mode.0),
        Some(&asset_server),
    );
}

pub fn spawn_scene_obstacle_colliders(mut commands: Commands, config: Res<RepositoryConfig>) {
    for obstacle in &config.scene.obstacles {
        let position = Vec3::from_array(obstacle.position);
        let scale = Vec3::from_array(obstacle.size);
        commands.spawn((
            Transform::from_translation(position),
            GlobalTransform::default(),
            ObstacleCollider {
                half_extents: scale * 0.5,
                damage_scale: obstacle.damage_scale,
            },
        ));
    }
}

pub fn spawn_headless_aircraft(
    mut commands: Commands,
    config: Res<RepositoryConfig>,
    headless_mode: Res<HeadlessMode>,
    bridge_mode: Option<Res<BridgeMode>>,
    semantic_capture_mode: Option<Res<SemanticCaptureMode>>,
) {
    spawn_aircraft_entities(
        &mut commands,
        &config,
        headless_mode.0,
        bridge_mode.as_deref(),
        semantic_capture_mode.is_some_and(|mode| mode.0),
        None,
    );
}

fn spawn_aircraft_entities(
    commands: &mut Commands,
    config: &RepositoryConfig,
    headless_mode: bool,
    bridge_mode: Option<&BridgeMode>,
    semantic_capture_mode: bool,
    asset_server: Option<&AssetServer>,
) {
    let fighter1 = &config.fighter1_aircraft;
    let fighter1_spawn = &config.scene.fighter1_spawn;
    let fighter1_performance = aircraft_performance_from_config(fighter1);
    info!(
        "Spawning fighter1 aircraft with initial_speed={} initial_throttle={} cruise_ref_throttle={} max_thrust={} cruise_ref_speed={} maneuver_reference_speed={} trim_pitch_deg={} pitch+={} pitch-={} roll+={} roll-={} yaw+={} yaw-={}",
        fighter1_spawn.initial_speed,
        fighter1_spawn.initial_throttle,
        fighter1.cruise_reference_throttle,
        fighter1.max_thrust,
        fighter1_performance.cruise_reference_speed,
        fighter1_performance.maneuver_reference_speed,
        fighter1_performance.trim_pitch_degrees,
        fighter1.pitch_positive_rate_limit_deg,
        fighter1.pitch_negative_rate_limit_deg,
        fighter1.roll_positive_rate_limit_deg,
        fighter1.roll_negative_rate_limit_deg,
        fighter1.yaw_positive_rate_limit_deg,
        fighter1.yaw_negative_rate_limit_deg
    );
    let fighter1_position = Vec3::from_array(fighter1_spawn.position);
    let fighter1_flight_path_orientation =
        scene_flight_path_orientation(fighter1_spawn.rotation_degrees);
    let fighter1_orientation = trimmed_spawn_orientation(
        fighter1_flight_path_orientation,
        fighter1_performance.trim_pitch_degrees,
    );
    let fighter1_authority = if matches!(bridge_mode.map(|mode| mode.0), Some(BridgeRole::Server)) {
        ControlAuthority::BuiltInAi
    } else if headless_mode {
        ControlAuthority::ExternalAgent
    } else {
        ControlAuthority::Human
    };
    let mut fighter1_state = AircraftState {
        position: fighter1_position,
        velocity: fighter1_flight_path_orientation
            * Vec3::new(0.0, 0.0, fighter1_spawn.initial_speed),
        orientation: fighter1_orientation,
        throttle: fighter1_spawn.initial_throttle.clamp(0.0, 1.0),
        hit_points: fighter1.hit_points,
        ..Default::default()
    };
    let mut fighter1_damage = AircraftDamageState::new(fighter1.hit_points, &fighter1_performance);
    apply_spawn_damage_overrides(
        &mut fighter1_state,
        &mut fighter1_damage,
        &fighter1_performance,
        fighter1_spawn.initial_damage.as_ref(),
    );
    let mut fighter1_entity = commands.spawn((
        AircraftRole::Fighter1,
        fighter1_authority,
        fighter1_state,
        SpawnTransform {
            position: fighter1_position,
            orientation: fighter1_orientation,
        },
        Transform::from_translation(fighter1_position).with_rotation(fighter1_orientation),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        fighter1_performance.clone(),
        fighter1_damage,
        ControlInput::default(),
        GunState::default(),
    ));
    if let Some(asset_server) = asset_server {
        fighter1_entity.with_children(|parent| {
            spawn_aircraft_visual(
                parent,
                asset_server,
                Color::srgb(0.96, 0.62, 0.16),
                Color::srgb(1.0, 0.95, 0.82),
            );
            if semantic_capture_mode {
                spawn_aircraft_semantic_visual(
                    parent,
                    asset_server,
                    semantic_class_color(AircraftRole::Fighter1),
                );
            }
        });
    }

    let fighter2 = &config.fighter2_aircraft;
    let fighter2_spawn = &config.scene.fighter2_spawn;
    let fighter2_performance = aircraft_performance_from_config(fighter2);
    info!(
        "Spawning fighter2 aircraft with initial_speed={} initial_throttle={} cruise_ref_throttle={} max_thrust={} cruise_ref_speed={} maneuver_reference_speed={} trim_pitch_deg={} pitch+={} pitch-={} roll+={} roll-={} yaw+={} yaw-={}",
        fighter2_spawn.initial_speed,
        fighter2_spawn.initial_throttle,
        fighter2.cruise_reference_throttle,
        fighter2.max_thrust,
        fighter2_performance.cruise_reference_speed,
        fighter2_performance.maneuver_reference_speed,
        fighter2_performance.trim_pitch_degrees,
        fighter2.pitch_positive_rate_limit_deg,
        fighter2.pitch_negative_rate_limit_deg,
        fighter2.roll_positive_rate_limit_deg,
        fighter2.roll_negative_rate_limit_deg,
        fighter2.yaw_positive_rate_limit_deg,
        fighter2.yaw_negative_rate_limit_deg
    );
    let fighter2_position = Vec3::from_array(fighter2_spawn.position);
    let fighter2_flight_path_orientation =
        scene_flight_path_orientation(fighter2_spawn.rotation_degrees);
    let fighter2_orientation = trimmed_spawn_orientation(
        fighter2_flight_path_orientation,
        fighter2_performance.trim_pitch_degrees,
    );
    let mut fighter2_state = AircraftState {
        position: fighter2_position,
        velocity: fighter2_flight_path_orientation
            * Vec3::new(0.0, 0.0, fighter2_spawn.initial_speed),
        orientation: fighter2_orientation,
        throttle: fighter2_spawn.initial_throttle.clamp(0.0, 1.0),
        hit_points: fighter2.hit_points,
        ..Default::default()
    };
    let mut fighter2_damage = AircraftDamageState::new(fighter2.hit_points, &fighter2_performance);
    apply_spawn_damage_overrides(
        &mut fighter2_state,
        &mut fighter2_damage,
        &fighter2_performance,
        fighter2_spawn.initial_damage.as_ref(),
    );
    let mut fighter2_entity = commands.spawn((
        AircraftRole::Fighter2,
        ControlAuthority::BuiltInAi,
        fighter2_state,
        SpawnTransform {
            position: fighter2_position,
            orientation: fighter2_orientation,
        },
        Transform::from_translation(fighter2_position).with_rotation(fighter2_orientation),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        fighter2_performance.clone(),
        fighter2_damage,
        ControlInput::default(),
        GunState::default(),
    ));
    if let Some(asset_server) = asset_server {
        fighter2_entity.with_children(|parent| {
            spawn_aircraft_visual(
                parent,
                asset_server,
                Color::srgb(0.88, 0.24, 0.24),
                Color::srgb(1.0, 0.9, 0.84),
            );
            if semantic_capture_mode {
                spawn_aircraft_semantic_visual(
                    parent,
                    asset_server,
                    semantic_class_color(AircraftRole::Fighter2),
                );
            }
        });
    }
}

fn semantic_class_color(role: AircraftRole) -> Color {
    match role {
        AircraftRole::Fighter1 => Color::linear_rgba(0.0, 1.0, 0.0, 1.0),
        AircraftRole::Fighter2 => Color::linear_rgba(1.0, 0.0, 0.0, 1.0),
    }
}

pub fn aircraft_performance_from_config(
    aircraft: &crate::core::config::AircraftConfig,
) -> AircraftPerformance {
    let cruise_reference_speed = aircraft.cruise_reference_speed.max(1.0);
    let maneuver_reference_speed =
        reference_speed_from_throttle(aircraft, aircraft.maneuver_reference_throttle)
            .max(cruise_reference_speed);

    AircraftPerformance {
        max_level_speed: aircraft.max_level_speed,
        cruise_reference_throttle: aircraft.cruise_reference_throttle,
        cruise_reference_speed,
        maneuver_reference_speed,
        max_thrust: aircraft.max_thrust,
        trim_pitch_degrees: aircraft.trim_pitch_degrees,
        trim_angle_of_attack_radians: aircraft.trim_angle_of_attack_radians,
        reference_level_lift_factor: aircraft.reference_level_lift_factor,
        throttle_response: aircraft.throttle_response,
        lift_coefficient: aircraft.lift_coefficient,
        induced_drag_coefficient: aircraft.induced_drag_coefficient,
        side_drag_coefficient: aircraft.side_drag_coefficient,
        gravity_scale: aircraft.gravity_scale,
        linear_drag: aircraft.linear_drag,
        brake_drag: aircraft.brake_drag,
        weight_kg: aircraft.weight_kg,
        stall_speed: aircraft.stall_speed,
        stall_recovery_speed: aircraft.stall_recovery_speed,
        stall_reference_dynamic_pressure: aircraft.stall_reference_dynamic_pressure,
        stall_recovery_dynamic_pressure: aircraft.stall_recovery_dynamic_pressure,
        pitch_response: aircraft.pitch_response,
        yaw_response: aircraft.yaw_response,
        roll_response: aircraft.roll_response,
        angular_damping: aircraft.angular_damping,
        pitch_positive_rate_limit_deg: aircraft.pitch_positive_rate_limit_deg,
        pitch_negative_rate_limit_deg: aircraft.pitch_negative_rate_limit_deg,
        roll_positive_rate_limit_deg: aircraft.roll_positive_rate_limit_deg,
        roll_negative_rate_limit_deg: aircraft.roll_negative_rate_limit_deg,
        yaw_positive_rate_limit_deg: aircraft.yaw_positive_rate_limit_deg,
        yaw_negative_rate_limit_deg: aircraft.yaw_negative_rate_limit_deg,
        pitch_maneuver_scale: aircraft.pitch_maneuver_scale,
        roll_maneuver_scale: aircraft.roll_maneuver_scale,
        yaw_maneuver_scale: aircraft.yaw_maneuver_scale,
        pitch_low_speed_scale: aircraft.pitch_low_speed_scale,
        roll_low_speed_scale: aircraft.roll_low_speed_scale,
        yaw_low_speed_scale: aircraft.yaw_low_speed_scale,
        pitch_high_speed_max_scale: aircraft.pitch_high_speed_max_scale,
        roll_high_speed_max_scale: aircraft.roll_high_speed_max_scale,
        yaw_high_speed_max_scale: aircraft.yaw_high_speed_max_scale,
        left_wing_hit_points: aircraft.left_wing_hit_points,
        right_wing_hit_points: aircraft.right_wing_hit_points,
        pitch_tail_hit_points: aircraft.pitch_tail_hit_points,
        yaw_tail_hit_points: aircraft.yaw_tail_hit_points,
        engine_hit_points: aircraft.engine_hit_points,
        damaged_control_surface_scale: aircraft.damaged_control_surface_scale,
        destroyed_control_surface_scale: aircraft.destroyed_control_surface_scale,
        damaged_engine_thrust_scale: aircraft.damaged_engine_thrust_scale,
        destroyed_engine_thrust_scale: aircraft.destroyed_engine_thrust_scale,
        damaged_engine_throttle_response_min: aircraft.damaged_engine_throttle_response_min,
        damaged_wing_lift_scale: aircraft.damaged_wing_lift_scale,
        destroyed_wing_lift_scale: aircraft.destroyed_wing_lift_scale,
        damage_roll_trim_base_deg: aircraft.damage_roll_trim_base_deg,
        damage_roll_trim_asymmetry_deg: aircraft.damage_roll_trim_asymmetry_deg,
        damage_yaw_trim_base_deg: aircraft.damage_yaw_trim_base_deg,
        damage_yaw_trim_asymmetry_deg: aircraft.damage_yaw_trim_asymmetry_deg,
        damage_extra_drag_per_surface: aircraft.damage_extra_drag_per_surface,
        damage_extra_drag_asymmetry: aircraft.damage_extra_drag_asymmetry,
        gun: aircraft.gun.clone(),
    }
}

pub fn scene_flight_path_orientation(rotation_degrees: [f32; 3]) -> Quat {
    Quat::from_euler(
        EulerRot::XYZ,
        rotation_degrees[0].to_radians(),
        rotation_degrees[1].to_radians(),
        rotation_degrees[2].to_radians(),
    )
}

pub fn trimmed_spawn_orientation(flight_path_orientation: Quat, trim_pitch_degrees: f32) -> Quat {
    // Positive trim pitch means the body is pitched up relative to the level flight path.
    flight_path_orientation * Quat::from_rotation_x(-trim_pitch_degrees.to_radians())
}

fn reference_speed_from_throttle(
    aircraft: &crate::core::config::AircraftConfig,
    throttle: f32,
) -> f32 {
    let thrust = (aircraft.max_thrust * throttle.clamp(0.05, 1.0)).max(0.1);
    let induced_drag_baseline =
        (aircraft.lift_coefficient * 0.55 * aircraft.induced_drag_coefficient).max(0.0);
    let available_linear_thrust = (thrust - induced_drag_baseline).max(0.1);
    (available_linear_thrust / aircraft.linear_drag.max(0.0001)).sqrt()
}

pub fn apply_control_inputs(mut debug: ResMut<SimulationDebugState>) {
    debug.tick_count += 1;
}

pub fn integrate_flight_model(
    time: Res<Time<Fixed>>,
    app_mode: Res<AppMode>,
    config: Res<RepositoryConfig>,
    bridge_mode: Option<Res<BridgeMode>>,
    bridge_enabled: Option<Res<BridgeEnabled>>,
    bridge_link: Option<Res<BridgeLinkState>>,
    requested_role: Option<Res<RequestedControlRole>>,
    assigned_role: Option<Res<AssignedControlRole>>,
    server_sessions: Option<Res<BridgeServerSessions>>,
    mut query: Query<(
        &AircraftRole,
        &mut AircraftState,
        &AircraftPerformance,
        &AircraftDamageState,
        &ControlInput,
    )>,
) {
    if *app_mode == AppMode::Observer {
        return;
    }

    let waiting_for_players = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Server)
    ) && bridge_enabled
        .as_deref()
        .map(|enabled| enabled.0)
        .unwrap_or(true)
        && !server_sessions
            .as_deref()
            .map(players_ready)
            .unwrap_or(false);
    if waiting_for_players {
        return;
    }

    let live_spectator_client = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Client)
    ) && bridge_enabled
        .as_deref()
        .map(|enabled| enabled.0)
        .unwrap_or(true)
        && matches!(
            requested_role.as_deref().map(|role| role.0),
            Some(crate::bridge::protocol::BridgeControlSlot::Spectator)
        );
    if live_spectator_client {
        return;
    }

    let remote_authority_active = bridge_link
        .as_deref()
        .map(|link| link.remote_authority_active)
        .unwrap_or(false);
    let locally_controlled_role = assigned_role
        .as_deref()
        .and_then(|role| bridge_slot_aircraft_role(role.0));
    let delta_seconds = time.delta_secs();
    for (role, mut state, performance, damage, input) in &mut query {
        if matches!(
            bridge_mode.as_deref().map(|mode| mode.0),
            Some(BridgeRole::Client)
        ) && remote_authority_active
            && Some(*role) != locally_controlled_role
        {
            continue;
        }
        if state.is_destroyed {
            continue;
        }
        step_predicted_aircraft_state(
            &config,
            &mut state,
            performance,
            damage,
            input,
            delta_seconds,
        );
    }
}

pub fn step_predicted_aircraft_state(
    config: &RepositoryConfig,
    state: &mut AircraftState,
    performance: &AircraftPerformance,
    damage: &AircraftDamageState,
    input: &ControlInput,
    delta_seconds: f32,
) {
    let mut effective = effective_aircraft_performance(performance, damage, input);
    let ceiling_recovery = apply_ceiling_penalty(
        config,
        state.position.y,
        state.velocity.y,
        state.stall_factor,
        &mut effective,
    );
    if ceiling_recovery > 0.0 {
        if state.ceiling_recovery_seconds <= 0.0 {
            state.ceiling_recovery_target_pitch_deg =
                compute_ceiling_recovery_target_pitch_deg(config, state);
        }
        state.ceiling_recovery_seconds = state
            .ceiling_recovery_seconds
            .max(config.game.ceiling_recovery_duration_seconds);
    }
    let recovery_active = state.ceiling_recovery_seconds > 0.0;
    let modifiers = damage_flight_modifiers(damage, performance);
    let mut effective_input = *input;
    if recovery_active {
        effective_input.pitch = 0.0;
        effective_input.roll = 0.0;
        effective_input.yaw = 0.0;
    }
    step_aircraft(
        state,
        &effective,
        &modifiers,
        &effective_input,
        delta_seconds,
    );
    if recovery_active {
        apply_forced_ceiling_recovery(config, state, delta_seconds);
        state.ceiling_recovery_seconds = (state.ceiling_recovery_seconds - delta_seconds).max(0.0);
        let ceiling = config.scene.flight_ceiling_height;
        if state.position.y <= ceiling
            && state.stall_factor <= config.game.ceiling_recovery_release_stall_factor
        {
            state.ceiling_recovery_seconds = 0.0;
        }
    }
    update_out_of_bounds_timer(config, state, delta_seconds);
    apply_world_bounds(state, config);
}

pub fn predict_local_environment_collisions(
    app_mode: Res<AppMode>,
    config: Res<RepositoryConfig>,
    bridge_mode: Option<Res<BridgeMode>>,
    bridge_enabled: Option<Res<BridgeEnabled>>,
    bridge_link: Option<Res<BridgeLinkState>>,
    requested_role: Option<Res<RequestedControlRole>>,
    assigned_role: Option<Res<AssignedControlRole>>,
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut damage_indicators: Option<ResMut<DamageIndicatorQueue>>,
    obstacle_query: Query<(&Transform, &ObstacleCollider)>,
    mut query: Query<(&AircraftRole, &mut AircraftState, &AircraftDamageState)>,
) {
    if *app_mode == AppMode::Observer {
        return;
    }
    if !matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Client)
    ) {
        return;
    }
    if !bridge_enabled
        .as_deref()
        .map(|enabled| enabled.0)
        .unwrap_or(true)
    {
        return;
    }
    if matches!(
        requested_role.as_deref().map(|role| role.0),
        Some(crate::bridge::protocol::BridgeControlSlot::Spectator)
    ) {
        return;
    }
    if !bridge_link
        .as_deref()
        .map(|link| link.remote_authority_active)
        .unwrap_or(false)
    {
        return;
    }
    let Some(locally_controlled_role) = assigned_role
        .as_deref()
        .and_then(|role| bridge_slot_aircraft_role(role.0))
    else {
        return;
    };

    let ground_y = config.scene.ground_height;
    for (role, mut state, damage) in &mut query {
        if *role != locally_controlled_role {
            continue;
        }
        if state.is_destroyed {
            continue;
        }

        if let Some((contact, penetration)) = lowest_ground_contact(&state, damage, ground_y) {
            let vertical_impact = (-state.velocity.y).max(0.0);
            let impact_speed = punitive_collision_speed(
                vertical_impact * 0.95 + state.velocity.xz().length() * 0.45,
                24.0,
            );
            emit_local_collision_feedback(
                audio_events.as_deref_mut(),
                damage_indicators.as_deref_mut(),
                contact,
                impact_speed,
            );
            state.position.y += penetration.max(0.0);
            if state.velocity.y < 0.0 {
                state.velocity.y =
                    (-state.velocity.y * 0.45).max(LOCAL_GROUND_BOUNCE_MIN_SPEED * 0.35);
            }
            state.velocity.x *= 0.52;
            state.velocity.z *= 0.52;
        }

        for (transform, collider) in &obstacle_query {
            let center = transform.translation;
            let Some((contact, normal, penetration)) =
                aircraft_aabb_penetration(&state, damage, center, collider.half_extents)
            else {
                continue;
            };

            let relative_speed = punitive_collision_speed(state.velocity.length(), 22.0);
            emit_local_collision_feedback(
                audio_events.as_deref_mut(),
                damage_indicators.as_deref_mut(),
                contact,
                relative_speed * 1.75 * collider.damage_scale,
            );
            state.position += normal * penetration.max(0.0);
            let normal_speed = state.velocity.dot(normal);
            if normal_speed < 0.0 {
                state.velocity -= normal * (normal_speed * 1.35);
            }
            let tangential_velocity = state.velocity - normal * state.velocity.dot(normal);
            let bounce_speed = (relative_speed * 0.82).max(LOCAL_OBSTACLE_BOUNCE_MIN_SPEED);
            state.velocity = tangential_velocity * 0.26 + normal * bounce_speed;
            break;
        }
    }
}

fn emit_local_collision_feedback(
    audio_events: Option<&mut AudioEventQueue>,
    damage_indicators: Option<&mut DamageIndicatorQueue>,
    position: Vec3,
    impact_strength: f32,
) {
    let intensity = (impact_strength / 60.0).clamp(0.25, 1.0);
    if let Some(audio_events) = audio_events {
        audio_events.push(AudioEventKind::Hit, position, intensity);
    }
    if let Some(indicators) = damage_indicators {
        indicators.push(position, intensity);
    }
}

fn lowest_ground_contact(
    state: &AircraftState,
    damage: &AircraftDamageState,
    ground_y: f32,
) -> Option<(Vec3, f32)> {
    aircraft_world_collision_boxes(state, damage)
        .into_iter()
        .filter_map(|collision_box| obb_plane_y_penetration(&collision_box, ground_y, Vec3::Y))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

fn aircraft_aabb_penetration(
    state: &AircraftState,
    damage: &AircraftDamageState,
    aabb_center: Vec3,
    aabb_half_extents: Vec3,
) -> Option<(Vec3, Vec3, f32)> {
    aircraft_world_collision_boxes(state, damage)
        .into_iter()
        .filter_map(|collision_box| {
            obb_aabb_penetration(&collision_box, aabb_center, aabb_half_extents)
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
}

fn punitive_collision_speed(measured_speed: f32, minimum_damage_speed: f32) -> f32 {
    measured_speed.max(minimum_damage_speed)
}

fn update_out_of_bounds_timer(
    config: &RepositoryConfig,
    state: &mut AircraftState,
    delta_seconds: f32,
) {
    let radial_distance = Vec2::new(state.position.x, state.position.z).length();
    if radial_distance >= config.scene.arena_radius {
        state.out_of_bounds_seconds = (state.out_of_bounds_seconds + delta_seconds)
            .min(config.scene.out_of_bounds_grace_seconds);
    } else {
        state.out_of_bounds_seconds = 0.0;
    }
}

fn apply_ceiling_penalty(
    config: &RepositoryConfig,
    altitude: f32,
    vertical_speed: f32,
    stall_factor: f32,
    performance: &mut AircraftPerformance,
) -> f32 {
    let ceiling = config.scene.flight_ceiling_height;
    let falloff = config.scene.ceiling_falloff_range.max(1.0);
    let start = ceiling - falloff;
    if altitude <= start {
        return 0.0;
    }

    let thrust_penalty = ((altitude - start) / falloff).clamp(0.0, 1.0);
    performance.max_thrust *= 1.0 - thrust_penalty;

    let soft_recovery_window = (config.game.ceiling_soft_recovery_stall_full_factor
        - config.game.ceiling_soft_recovery_stall_threshold)
        .max(0.01);
    let soft_recovery = if altitude <= ceiling
        && stall_factor >= config.game.ceiling_soft_recovery_stall_threshold
        && vertical_speed > config.game.ceiling_soft_recovery_vertical_speed_threshold
    {
        ((stall_factor - config.game.ceiling_soft_recovery_stall_threshold) / soft_recovery_window)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    if altitude <= ceiling {
        return soft_recovery;
    }

    let overshoot = (altitude - ceiling) / falloff;
    let overshoot_penalty = overshoot.clamp(0.0, 1.0);
    let descent_relief =
        (-vertical_speed / config.game.ceiling_descent_relief_speed.max(1.0)).clamp(0.0, 1.0);
    (overshoot_penalty * (1.0 - descent_relief * config.game.ceiling_descent_relief_factor))
        .clamp(0.0, 1.0)
        .max(soft_recovery)
}

fn compute_ceiling_recovery_target_pitch_deg(
    config: &RepositoryConfig,
    state: &AircraftState,
) -> f32 {
    let current_forward = state.forward.normalize_or_zero();
    if current_forward == Vec3::ZERO {
        return -config.game.ceiling_recovery_min_dive_angle_deg;
    }

    let current_pitch_deg = current_forward.y.clamp(-1.0, 1.0).asin().to_degrees();
    if current_pitch_deg > 0.0 {
        -current_pitch_deg
            .abs()
            .max(config.game.ceiling_recovery_min_dive_angle_deg)
    } else {
        current_pitch_deg.min(-config.game.ceiling_recovery_min_dive_angle_deg)
    }
}

fn apply_forced_ceiling_recovery(
    config: &RepositoryConfig,
    state: &mut AircraftState,
    delta_seconds: f32,
) {
    let current_forward = state.forward.normalize_or_zero();
    if current_forward == Vec3::ZERO {
        return;
    }

    let target_pitch = state.ceiling_recovery_target_pitch_deg.to_radians();

    let mut horizontal_heading = Vec3::new(current_forward.x, 0.0, current_forward.z);
    if horizontal_heading.length_squared() <= f32::EPSILON {
        let fallback_forward = state.orientation * Vec3::Z;
        horizontal_heading = Vec3::new(fallback_forward.x, 0.0, fallback_forward.z);
    }
    horizontal_heading = horizontal_heading.normalize_or_zero();
    if horizontal_heading == Vec3::ZERO {
        horizontal_heading = Vec3::Z;
    }

    let target_forward =
        (horizontal_heading * target_pitch.cos() + Vec3::Y * target_pitch.sin()).normalize();
    let target_rotation =
        Quat::from_rotation_arc(current_forward, target_forward) * state.orientation;
    let remaining = state.ceiling_recovery_seconds.max(delta_seconds).min(
        config
            .game
            .ceiling_recovery_duration_seconds
            .max(delta_seconds),
    );
    let blend = (delta_seconds / remaining).clamp(0.0, 1.0);
    state.orientation = state
        .orientation
        .slerp(target_rotation.normalize(), blend)
        .normalize();
    state.forward = (state.orientation * Vec3::Z).normalize_or_zero();
    if state.forward == Vec3::ZERO {
        state.forward = Vec3::Z;
        state.orientation = Quat::IDENTITY;
    }
    state.angular_rates_deg *= config.game.ceiling_recovery_angular_damping_factor;
}

pub fn update_aircraft_state(
    time: Res<Time>,
    real_time: Res<Time<Real>>,
    bridge_mode: Option<Res<BridgeMode>>,
    bridge_link: Option<Res<BridgeLinkState>>,
    bridge_inbox: Option<Res<BridgeClientInbox>>,
    remote_interpolation: Option<Res<BridgeRemoteInterpolationState>>,
    assigned_role: Option<Res<AssignedControlRole>>,
    mut debug_log: Local<RenderStateDebugLog>,
    mut query: Query<(&AircraftRole, &mut Transform, &AircraftState)>,
) {
    let remote_authority_active = bridge_link
        .as_deref()
        .map(|link| link.remote_authority_active)
        .unwrap_or(false);
    let locally_controlled_role = assigned_role
        .as_deref()
        .and_then(|role| bridge_slot_aircraft_role(role.0));
    let client_remote = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Client)
    ) && remote_authority_active;
    let local_alpha = (time.delta_secs() * 28.0).clamp(0.0, 1.0);
    let render_sim_time_seconds = bridge_inbox.as_deref().and_then(|inbox| {
        inbox.latest_snapshot.as_ref().map(|snapshot| {
            let fixed_dt = inbox
                .server_hello
                .as_ref()
                .map(|hello| hello.fixed_time_step_seconds.max(1.0 / 240.0))
                .unwrap_or(1.0 / 60.0);
            snapshot.observation.state.sim_time_seconds
                - fixed_dt * (inbox.target_buffer_len.max(1) as f32)
        })
    });
    let should_log = real_time.elapsed_secs_f64() >= debug_log.next_log_at_seconds;
    let mut saw_debug_signal = false;

    for (role, mut transform, state) in &mut query {
        if client_remote {
            if Some(*role) == locally_controlled_role {
                let render_pos_err = transform.translation.distance(state.position);
                let render_orient_dot = transform.rotation.dot(state.orientation).abs();
                if should_log && render_pos_err > 0.35 {
                    info!(
                        "bridge render local role={role:?} pos_err={:.3} orient_dot={:.5} local_alpha={:.3}",
                        render_pos_err, render_orient_dot, local_alpha,
                    );
                    saw_debug_signal = true;
                }
                transform.translation = transform.translation.lerp(state.position, local_alpha);
                transform.rotation = transform
                    .rotation
                    .slerp(state.orientation, local_alpha)
                    .normalize();
            } else {
                let previous_translation = transform.translation;
                let previous_rotation = transform.rotation;
                if let (Some(remote_interpolation), Some(render_sim_time_seconds)) =
                    (remote_interpolation.as_deref(), render_sim_time_seconds)
                    && let Some((position, orientation)) = sample_remote_aircraft_snapshot(
                        remote_interpolation,
                        *role,
                        render_sim_time_seconds,
                    )
                {
                    if should_log {
                        let jump_distance = previous_translation.distance(position);
                        let jump_orientation_dot = previous_rotation.dot(orientation).abs();
                        if jump_distance > 2.5 || jump_orientation_dot < 0.995 {
                            info!(
                                "bridge render remote role={role:?} source=interpolated jump_dist={:.3} jump_orient_dot={:.5} render_sim_time={:.3}",
                                jump_distance, jump_orientation_dot, render_sim_time_seconds,
                            );
                            saw_debug_signal = true;
                        }
                    }
                    transform.translation = position;
                    transform.rotation = orientation;
                    continue;
                }
                if should_log {
                    let jump_distance = previous_translation.distance(state.position);
                    let jump_orientation_dot = previous_rotation.dot(state.orientation).abs();
                    if jump_distance > 2.5 || jump_orientation_dot < 0.995 {
                        info!(
                            "bridge render remote role={role:?} source=authoritative_fallback jump_dist={:.3} jump_orient_dot={:.5}",
                            jump_distance, jump_orientation_dot,
                        );
                        saw_debug_signal = true;
                    }
                }
                transform.translation = state.position;
                transform.rotation = state.orientation;
            }
        } else {
            transform.translation = state.position;
            transform.rotation = state.orientation;
        }
    }
    if should_log {
        debug_log.next_log_at_seconds =
            real_time.elapsed_secs_f64() + if saw_debug_signal { 0.5 } else { 0.2 };
    }
}

pub fn update_damage_visuals(
    aircraft_query: Query<(&Children, &AircraftState, &AircraftDamageState)>,
    children_query: Query<&Children>,
    mut part_query: Query<(
        &AircraftVisualPart,
        &MeshMaterial3d<StandardMaterial>,
        &DamageVisualStyle,
        &mut Visibility,
    )>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (children, state, damage) in &aircraft_query {
        update_damage_visuals_recursive(
            children,
            &children_query,
            &mut part_query,
            &mut materials,
            state,
            damage,
        );
    }
}

fn spawn_aircraft_visual(
    parent: &mut ChildSpawnerCommands,
    asset_server: &AssetServer,
    fuselage_color: Color,
    accent_color: Color,
) {
    let damaged_fuselage = mix_color(fuselage_color, Color::srgb(0.28, 0.18, 0.18), 0.55);
    let damaged_accent = mix_color(accent_color, Color::srgb(0.36, 0.12, 0.12), 0.6);
    let scene_handle =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/fighter_plane.gltf"));
    parent.spawn((
        Name::new("FighterPlaneScene"),
        SceneRoot(scene_handle.clone()),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        AircraftVisualSceneRoot,
        AircraftVisualSceneHandle(scene_handle),
        AircraftVisualPalette {
            fuselage: color_to_array(fuselage_color),
            damaged_fuselage: color_to_array(damaged_fuselage),
            accent: color_to_array(accent_color),
            damaged_accent: color_to_array(damaged_accent),
        },
    ));
}

fn spawn_aircraft_semantic_visual(
    parent: &mut ChildSpawnerCommands,
    asset_server: &AssetServer,
    semantic_color: Color,
) {
    let scene_handle =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/fighter_plane.gltf"));
    parent.spawn((
        Name::new("FighterPlaneSemanticScene"),
        SceneRoot(scene_handle.clone()),
        Transform::default(),
        GlobalTransform::default(),
        Visibility::default(),
        InheritedVisibility::default(),
        ViewVisibility::default(),
        AircraftSemanticSceneRoot,
        AircraftVisualSceneHandle(scene_handle),
        AircraftSemanticClassColor {
            color: color_to_array(semantic_color),
        },
        RenderLayers::layer(SEMANTIC_RENDER_LAYER),
    ));
}

fn visual_stage_for_part(
    part: AircraftVisualPart,
    _state: &AircraftState,
    damage: &AircraftDamageState,
) -> DamageStage {
    match part {
        AircraftVisualPart::Body | AircraftVisualPart::Cockpit => DamageStage::Undamaged,
        AircraftVisualPart::Engine => damage.engine.stage(),
        AircraftVisualPart::LeftWing => damage.left_wing.stage(),
        AircraftVisualPart::RightWing => damage.right_wing.stage(),
        AircraftVisualPart::PitchTail => damage.pitch_tail.stage(),
        AircraftVisualPart::YawTail => damage.yaw_tail.stage(),
    }
}

fn damage_visual_blend(fraction: f32) -> f32 {
    let loss = (1.0 - fraction).clamp(0.0, 1.0);
    visual_smoothstep(0.05, 0.45, loss)
}

fn visual_smoothstep(start: f32, end: f32, value: f32) -> f32 {
    let span = (end - start).abs().max(f32::EPSILON);
    let t = if end >= start {
        ((value - start) / span).clamp(0.0, 1.0)
    } else {
        ((start - value) / span).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

fn color_to_array(color: Color) -> [f32; 4] {
    let srgba = color.to_srgba();
    [srgba.red, srgba.green, srgba.blue, srgba.alpha]
}

fn mix_color(a: Color, b: Color, t: f32) -> Color {
    let a = a.to_srgba();
    let b = b.to_srgba();
    let mix = |x: f32, y: f32| x + (y - x) * t;
    Color::srgba(
        mix(a.red, b.red),
        mix(a.green, b.green),
        mix(a.blue, b.blue),
        mix(a.alpha, b.alpha),
    )
}

fn update_damage_visuals_recursive(
    children: &Children,
    children_query: &Query<&Children>,
    part_query: &mut Query<(
        &AircraftVisualPart,
        &MeshMaterial3d<StandardMaterial>,
        &DamageVisualStyle,
        &mut Visibility,
    )>,
    materials: &mut Assets<StandardMaterial>,
    state: &AircraftState,
    damage: &AircraftDamageState,
) {
    for child in children.iter() {
        if let Ok((part, material, style, mut visibility)) = part_query.get_mut(child) {
            let stage = visual_stage_for_part(*part, state, damage);
            *visibility = Visibility::Visible;

            let fraction = match *part {
                AircraftVisualPart::Body | AircraftVisualPart::Cockpit => 1.0,
                AircraftVisualPart::Engine => damage.engine.fraction(),
                AircraftVisualPart::LeftWing => damage.left_wing.fraction(),
                AircraftVisualPart::RightWing => damage.right_wing.fraction(),
                AircraftVisualPart::PitchTail => damage.pitch_tail.fraction(),
                AircraftVisualPart::YawTail => damage.yaw_tail.fraction(),
            };

            if let Some(material_asset) = materials.get_mut(&material.0) {
                material_asset.base_color = match stage {
                    DamageStage::Destroyed => Color::srgba(
                        style.destroyed[0],
                        style.destroyed[1],
                        style.destroyed[2],
                        style.destroyed[3],
                    ),
                    _ => mix_color(
                        Color::srgba(
                            style.intact[0],
                            style.intact[1],
                            style.intact[2],
                            style.intact[3],
                        ),
                        Color::srgba(
                            style.damaged[0],
                            style.damaged[1],
                            style.damaged[2],
                            style.damaged[3],
                        ),
                        damage_visual_blend(fraction),
                    ),
                };
                material_asset.alpha_mode = if material_asset.base_color.alpha() < 0.999 {
                    AlphaMode::Blend
                } else {
                    AlphaMode::Opaque
                };
            }
        }
        if let Ok(grandchildren) = children_query.get(child) {
            update_damage_visuals_recursive(
                grandchildren,
                children_query,
                part_query,
                materials,
                state,
                damage,
            );
        }
    }
}

pub fn attach_aircraft_visual_parts(
    mut commands: Commands,
    root_query: Query<
        (Entity, &Children, &AircraftVisualPalette),
        (
            With<AircraftVisualSceneRoot>,
            Without<AircraftVisualPartsInitialized>,
        ),
    >,
    children_query: Query<&Children>,
    mut mesh_query: Query<(Entity, &Name, &mut MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (root_entity, children, palette) in &root_query {
        let mut tagged_count = 0;
        tag_visual_parts_recursive(
            &mut commands,
            children,
            &children_query,
            &mut mesh_query,
            &mut materials,
            palette,
            &mut tagged_count,
        );
        if tagged_count >= 7 {
            commands
                .entity(root_entity)
                .insert(AircraftVisualPartsInitialized);
        }
    }
}

pub fn attach_aircraft_semantic_parts(
    mut commands: Commands,
    root_query: Query<
        (Entity, &Children, &AircraftSemanticClassColor),
        (
            With<AircraftSemanticSceneRoot>,
            Without<AircraftSemanticPartsInitialized>,
        ),
    >,
    children_query: Query<&Children>,
    mut mesh_query: Query<(Entity, &Name, &mut MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (root_entity, children, semantic_class) in &root_query {
        let mut tagged_count = 0;
        tag_semantic_parts_recursive(
            &mut commands,
            children,
            &children_query,
            &mut mesh_query,
            &mut materials,
            *semantic_class,
            &mut tagged_count,
        );
        if tagged_count > 0 {
            commands
                .entity(root_entity)
                .insert(AircraftSemanticPartsInitialized);
        }
    }
}

fn tag_visual_parts_recursive(
    commands: &mut Commands,
    children: &Children,
    children_query: &Query<&Children>,
    mesh_query: &mut Query<(Entity, &Name, &mut MeshMaterial3d<StandardMaterial>)>,
    materials: &mut Assets<StandardMaterial>,
    palette: &AircraftVisualPalette,
    tagged_count: &mut usize,
) {
    for child in children.iter() {
        if let Ok((entity, name, mut material)) = mesh_query.get_mut(child)
            && let Some((part, style, roughness, metallic)) =
                visual_metadata_for_name(name.as_str(), palette)
        {
            let handle = materials.add(StandardMaterial {
                base_color: Color::srgba(
                    style.intact[0],
                    style.intact[1],
                    style.intact[2],
                    style.intact[3],
                ),
                perceptual_roughness: roughness,
                metallic,
                double_sided: true,
                alpha_mode: AlphaMode::Blend,
                ..default()
            });
            material.0 = handle;
            commands.entity(entity).insert((part, style));
            *tagged_count += 1;
        }
        if let Ok(grandchildren) = children_query.get(child) {
            tag_visual_parts_recursive(
                commands,
                grandchildren,
                children_query,
                mesh_query,
                materials,
                palette,
                tagged_count,
            );
        }
    }
}

fn tag_semantic_parts_recursive(
    commands: &mut Commands,
    children: &Children,
    children_query: &Query<&Children>,
    mesh_query: &mut Query<(Entity, &Name, &mut MeshMaterial3d<StandardMaterial>)>,
    materials: &mut Assets<StandardMaterial>,
    semantic_class: AircraftSemanticClassColor,
    tagged_count: &mut usize,
) {
    for child in children.iter() {
        if let Ok((entity, name, mut material)) = mesh_query.get_mut(child)
            && is_aircraft_visual_part_name(name.as_str())
        {
            let base_color = Color::srgba(
                semantic_class.color[0],
                semantic_class.color[1],
                semantic_class.color[2],
                semantic_class.color[3],
            );
            let handle = materials.add(StandardMaterial {
                base_color,
                emissive: LinearRgba::rgb(
                    semantic_class.color[0],
                    semantic_class.color[1],
                    semantic_class.color[2],
                ),
                unlit: true,
                double_sided: true,
                alpha_mode: AlphaMode::Opaque,
                ..default()
            });
            material.0 = handle;
            commands
                .entity(entity)
                .insert(RenderLayers::layer(SEMANTIC_RENDER_LAYER));
            *tagged_count += 1;
        }
        if let Ok(grandchildren) = children_query.get(child) {
            tag_semantic_parts_recursive(
                commands,
                grandchildren,
                children_query,
                mesh_query,
                materials,
                semantic_class,
                tagged_count,
            );
        }
    }
}

fn visual_metadata_for_name(
    name: &str,
    palette: &AircraftVisualPalette,
) -> Option<(AircraftVisualPart, DamageVisualStyle, f32, f32)> {
    let name = name.to_ascii_lowercase();
    let cockpit_style = DamageVisualStyle {
        intact: color_to_array(Color::srgb(0.08, 0.1, 0.13)),
        damaged: color_to_array(Color::srgb(0.16, 0.08, 0.08)),
        destroyed: color_to_array(Color::srgba(0.12, 0.08, 0.08, 0.50)),
    };
    let fuselage_style = DamageVisualStyle {
        intact: palette.fuselage,
        damaged: palette.damaged_fuselage,
        destroyed: color_to_array(Color::srgba(0.16, 0.12, 0.12, 0.52)),
    };
    let engine_style = DamageVisualStyle {
        intact: palette.accent,
        damaged: palette.damaged_accent,
        destroyed: color_to_array(Color::srgba(0.18, 0.10, 0.10, 0.48)),
    };

    if name.contains("body") {
        Some((AircraftVisualPart::Body, fuselage_style, 0.62, 0.06))
    } else if name.contains("engine") {
        Some((AircraftVisualPart::Engine, engine_style, 0.42, 0.14))
    } else if name.contains("cockpit") {
        Some((AircraftVisualPart::Cockpit, cockpit_style, 0.3, 0.2))
    } else if name.contains("leftwing") || name.contains("left_wing") {
        Some((AircraftVisualPart::LeftWing, fuselage_style, 0.58, 0.08))
    } else if name.contains("rightwing") || name.contains("right_wing") {
        Some((AircraftVisualPart::RightWing, fuselage_style, 0.58, 0.08))
    } else if name.contains("pitchtail")
        || name.contains("tailplane")
        || name.contains("horizontaltail")
    {
        Some((AircraftVisualPart::PitchTail, fuselage_style, 0.58, 0.08))
    } else if name.contains("yawtail") || name.contains("verticaltail") || name.contains("fin") {
        Some((AircraftVisualPart::YawTail, fuselage_style, 0.58, 0.08))
    } else {
        None
    }
}

fn is_aircraft_visual_part_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("body")
        || name.contains("engine")
        || name.contains("cockpit")
        || name.contains("leftwing")
        || name.contains("left_wing")
        || name.contains("rightwing")
        || name.contains("right_wing")
        || name.contains("pitchtail")
        || name.contains("tailplane")
        || name.contains("horizontaltail")
        || name.contains("yawtail")
        || name.contains("verticaltail")
        || name.contains("fin")
}
