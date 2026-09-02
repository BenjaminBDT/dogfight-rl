use bevy::prelude::*;

use crate::api::events::{
    PendingEnvironmentEvents, push_event, push_event_with_context, role_subject, subsystem_subject,
};
use crate::api::types::EnvironmentEventKind;
use crate::audio::{AudioEventKind, AudioEventQueue, FrameAudioState};
use crate::core::config::RepositoryConfig;
use crate::gameplay::damage::{AircraftDamageState, DamageStage, subsystem_hitboxes};
use crate::input::actions::ControlInput;
use crate::presentation::hud::DamageIndicatorQueue;
use crate::presentation::tracers::TracerLifetime;
use crate::simulation::collision::{
    AircraftCollisionBox, ObstacleCollider, aircraft_world_collision_boxes_for_pose,
    segment_obb_hit_fraction,
};
use crate::simulation::components::{
    AircraftPerformance, AircraftRole, AircraftState, ControlAuthority, GunState,
};
use crate::simulation::resources::SimulationDebugState;
use crate::{
    bridge::protocol::BridgeRole,
    bridge::{
        AssignedControlRole, BridgeEnabled, BridgeLinkState, BridgeMode,
        BridgeServerLagCompensationState, BridgeServerSessions, bridge_slot_aircraft_role,
        sample_server_historical_aircraft_snapshot,
    },
};

#[derive(Debug, Clone, Resource, Default)]
pub struct CombatState {
    pub hits_landed: u64,
    pub next_projectile_id: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct AircraftDestroyedEvent {
    pub role: AircraftRole,
    pub position: Vec3,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct CombatPresentationQueue {
    pub destroyed: Vec<AircraftDestroyedEvent>,
}

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct LocalFireVisualState {
    pub fighter1_cooldown_seconds: f32,
    pub fighter2_cooldown_seconds: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Projectile {
    pub id: u64,
    pub shooter_role: AircraftRole,
    pub velocity: Vec3,
    pub damage: f32,
    pub remaining_distance: f32,
    pub hit_radius: f32,
    pub flyby_emitted: bool,
    pub lag_compensation_ticks: u16,
}

#[derive(Component, Debug, Clone)]
pub struct LocalVisualProjectile {
    pub velocity: Vec3,
    pub remaining_distance: f32,
}

fn simulation_muzzle_pose(state: &AircraftState) -> (Vec3, Vec3) {
    let muzzle_position = state.position + state.orientation * Vec3::new(0.0, 0.15, 11.4);
    let muzzle_forward = (state.orientation * Vec3::Z).normalize_or_zero();
    (muzzle_position, muzzle_forward)
}

pub(crate) fn presentation_muzzle_pose(transform: &Transform) -> (Vec3, Vec3) {
    let muzzle_position = transform.translation + transform.rotation * Vec3::new(0.0, 0.15, 11.4);
    let muzzle_forward = (transform.rotation * Vec3::Z).normalize_or_zero();
    (muzzle_position, muzzle_forward)
}

fn aircraft_role_bridge_slot(role: AircraftRole) -> crate::bridge::protocol::BridgeControlSlot {
    match role {
        AircraftRole::Fighter1 => crate::bridge::protocol::BridgeControlSlot::Fighter1,
        AircraftRole::Fighter2 => crate::bridge::protocol::BridgeControlSlot::Fighter2,
    }
}

const MAX_SERVER_LAG_COMP_TICKS: u64 = 6;
const PROJECTILE_HIT_RADIUS_METERS: f32 = 0.8;
const PROJECTILE_SUBSYSTEM_HIT_RADIUS_METERS: f32 = 0.4;

fn update_best_target_hit(best: &mut Option<(f32, Vec3)>, hit_t: f32, hit_point: Vec3) {
    match best {
        Some((best_t, _)) if *best_t <= hit_t => {}
        _ => *best = Some((hit_t, hit_point)),
    }
}

fn update_best_subsystem_hit(
    best: &mut Option<(f32, crate::gameplay::damage::AircraftSubsystem, Vec3)>,
    hit_t: f32,
    subsystem: crate::gameplay::damage::AircraftSubsystem,
    hit_point: Vec3,
) {
    match best {
        Some((best_t, _, _)) if *best_t <= hit_t => {}
        _ => *best = Some((hit_t, subsystem, hit_point)),
    }
}

fn subsystem_assignment_t_slack(
    travel_distance: f32,
    aircraft_hit_radius: f32,
    subsystem_hit_radius: f32,
) -> f32 {
    let distance_slack =
        (aircraft_hit_radius - subsystem_hit_radius).max(0.0) + subsystem_hit_radius * 0.25;
    distance_slack / travel_distance.max(0.001)
}

pub fn spawn_projectile_visual_entity(
    commands: &mut Commands,
    meshes: Option<&mut Assets<Mesh>>,
    materials: Option<&mut Assets<StandardMaterial>>,
    projectile: Projectile,
    transform: Transform,
) {
    let bullet_color = match projectile.shooter_role {
        AircraftRole::Fighter1 => Color::srgb(1.0, 0.88, 0.44),
        AircraftRole::Fighter2 => Color::srgb(1.0, 0.42, 0.22),
    };

    let mut entity = commands.spawn((projectile, transform, Visibility::default()));

    if let (Some(meshes), Some(materials)) = (meshes, materials) {
        let outer_mesh = meshes.add(Cuboid::new(0.62, 0.62, 24.0));
        let outer_material = materials.add(StandardMaterial {
            base_color: bullet_color,
            emissive: LinearRgba::rgb(
                bullet_color.to_linear().red * 40.0,
                bullet_color.to_linear().green * 40.0,
                bullet_color.to_linear().blue * 40.0,
            ),
            unlit: true,
            alpha_mode: AlphaMode::Add,
            ..default()
        });
        let inner_mesh = meshes.add(Cuboid::new(0.24, 0.24, 14.0));
        let inner_material = materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.98, 0.94),
            emissive: LinearRgba::rgb(52.0, 48.0, 42.0),
            alpha_mode: AlphaMode::Add,
            unlit: true,
            ..default()
        });

        entity.with_children(|parent| {
            parent.spawn((
                Mesh3d(outer_mesh),
                MeshMaterial3d(outer_material),
                Transform::default(),
            ));
            parent.spawn((
                Mesh3d(inner_mesh),
                MeshMaterial3d(inner_material),
                Transform::from_xyz(0.0, 0.0, 1.2),
            ));
        });
    }
}

pub fn spawn_tracer_visual_entity(
    commands: &mut Commands,
    meshes: Option<&mut Assets<Mesh>>,
    materials: Option<&mut Assets<StandardMaterial>>,
    position: Vec3,
    remaining_seconds: f32,
) {
    match (meshes, materials) {
        (Some(meshes), Some(materials)) => {
            let color = Color::srgb(1.0, 0.55, 0.18);
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(6.0).mesh().uv(24, 16))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    emissive: color.into(),
                    unlit: true,
                    ..default()
                })),
                Transform::from_translation(position),
                TracerLifetime { remaining_seconds },
            ));
        }
        _ => {
            commands.spawn((
                Transform::from_translation(position),
                Visibility::default(),
                TracerLifetime { remaining_seconds },
            ));
        }
    }
}

pub fn tick_weapon_cooldowns(time: Res<Time<Fixed>>, mut query: Query<&mut GunState>) {
    let dt = time.delta_secs();
    for mut gun in &mut query {
        gun.cooldown_seconds = (gun.cooldown_seconds - dt).max(0.0);
        gun.is_firing = false;
    }
}

pub fn tick_local_fire_visual_cooldowns(
    time: Res<Time<Fixed>>,
    mut visuals: ResMut<LocalFireVisualState>,
) {
    let dt = time.delta_secs();
    visuals.fighter1_cooldown_seconds = (visuals.fighter1_cooldown_seconds - dt).max(0.0);
    visuals.fighter2_cooldown_seconds = (visuals.fighter2_cooldown_seconds - dt).max(0.0);
}

pub fn tick_weapon_heat(
    time: Res<Time<Fixed>>,
    debug: Res<SimulationDebugState>,
    mut env_events: Option<ResMut<PendingEnvironmentEvents>>,
    mut query: Query<(
        &AircraftRole,
        &AircraftState,
        &AircraftDamageState,
        &ControlInput,
        &AircraftPerformance,
        &mut GunState,
    )>,
) {
    let dt = time.delta_secs();
    for (role, state, damage_state, input, performance, mut gun) in &mut query {
        let actively_firing = !state.is_destroyed
            && !damage_state.is_repairing
            && !input.repair
            && input.fire_gun
            && !gun.overheated;
        if actively_firing {
            continue;
        }

        let cool_time = performance.gun.overheat_cool_seconds.max(0.1);
        gun.heat = (gun.heat - dt / cool_time).max(0.0);
        if gun.overheated && gun.heat <= performance.gun.overheat_resume_fraction.clamp(0.0, 1.0) {
            gun.overheated = false;
            if let Some(env_events) = env_events.as_deref_mut() {
                push_event(
                    env_events,
                    debug.tick_count,
                    EnvironmentEventKind::GunCooled,
                    Some(role_subject(*role)),
                    Some(state.position),
                    Some(gun.heat),
                );
            }
        }
    }
}

pub fn resolve_gunfire(
    debug: Res<SimulationDebugState>,
    mode: Res<BridgeMode>,
    config: Res<RepositoryConfig>,
    server_sessions: Option<Res<BridgeServerSessions>>,
    mut combat_state: ResMut<CombatState>,
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut env_events: Option<ResMut<PendingEnvironmentEvents>>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
    mut query: Query<(
        &AircraftRole,
        &AircraftState,
        &ControlAuthority,
        &AircraftPerformance,
        &AircraftDamageState,
        &ControlInput,
        &mut GunState,
    )>,
) {
    for (
        shooter_role,
        shooter_state,
        shooter_authority,
        performance,
        damage_state,
        shooter_input,
        mut gun_state,
    ) in &mut query
    {
        if shooter_state.is_destroyed
            || damage_state.is_repairing
            || shooter_input.repair
            || !shooter_input.fire_gun
            || gun_state.cooldown_seconds > 0.0
            || gun_state.overheated
        {
            continue;
        }

        let cooldown = 1.0 / performance.gun.rounds_per_second.max(0.1);
        gun_state.cooldown_seconds = cooldown;
        gun_state.is_firing = true;
        if let Some(env_events) = env_events.as_deref_mut() {
            push_event(
                env_events,
                debug.tick_count,
                EnvironmentEventKind::GunFired,
                Some(role_subject(*shooter_role)),
                Some(shooter_state.position),
                Some(1.0),
            );
        }
        gun_state.heat = (gun_state.heat
            + cooldown / performance.gun.overheat_fire_seconds.max(cooldown))
        .clamp(0.0, 1.0);
        if gun_state.heat >= 1.0 {
            gun_state.heat = 1.0;
            gun_state.overheated = true;
            if let Some(env_events) = env_events.as_deref_mut() {
                push_event(
                    env_events,
                    debug.tick_count,
                    EnvironmentEventKind::GunOverheated,
                    Some(role_subject(*shooter_role)),
                    Some(shooter_state.position),
                    Some(gun_state.heat),
                );
            }
        }
        // Authoritative fire / hit logic must follow simulation state, not
        // presentation transforms, otherwise prediction smoothing leaks into hit resolution.
        let (mut muzzle_position, muzzle_forward) = simulation_muzzle_pose(shooter_state);
        let projectile_velocity =
            shooter_state.velocity + muzzle_forward * performance.gun.projectile_speed;
        let mut lag_compensation_ticks = 0_u16;
        if mode.0 == BridgeRole::Server
            && matches!(shooter_authority, ControlAuthority::ExternalAgent)
            && let Some(session) = server_sessions.as_deref().and_then(|sessions| {
                let slot = aircraft_role_bridge_slot(*shooter_role);
                sessions
                    .clients
                    .values()
                    .find(|session| session.assigned_role == slot)
            })
            && let (Some(client_tick), Some(offset)) = (
                session.last_input_tick,
                session.estimated_server_tick_offset,
            )
        {
            let estimated_fire_server_tick = (client_tick as f64 + offset).round();
            let lag_ticks = debug
                .tick_count
                .saturating_sub(estimated_fire_server_tick.max(0.0) as u64)
                .min(MAX_SERVER_LAG_COMP_TICKS);
            lag_compensation_ticks = lag_ticks as u16;
            if lag_ticks > 0 {
                muzzle_position +=
                    projectile_velocity * (config.game.fixed_time_step_seconds * lag_ticks as f32);
            }
        }
        let tracer_rotation = Quat::from_rotation_arc(Vec3::Z, projectile_velocity.normalize());
        spawn_projectile_visual_entity(
            &mut commands,
            meshes.as_deref_mut(),
            materials.as_deref_mut(),
            Projectile {
                id: combat_state.next_projectile_id,
                shooter_role: *shooter_role,
                velocity: projectile_velocity,
                damage: performance.gun.damage_per_hit,
                remaining_distance: performance.gun.max_range,
                hit_radius: PROJECTILE_HIT_RADIUS_METERS,
                flyby_emitted: false,
                lag_compensation_ticks,
            },
            Transform::from_translation(muzzle_position).with_rotation(tracer_rotation),
        );
        combat_state.next_projectile_id = combat_state.next_projectile_id.saturating_add(1);

        if let Some(audio_events) = audio_events.as_deref_mut() {
            audio_events.push(AudioEventKind::GunFire, muzzle_position, 1.0);
        }
    }
}

pub fn spawn_local_fire_visuals(
    bridge_enabled: Res<BridgeEnabled>,
    bridge_mode: Res<BridgeMode>,
    bridge_link: Res<BridgeLinkState>,
    assigned_role: Res<AssignedControlRole>,
    mut visual_state: ResMut<LocalFireVisualState>,
    mut commands: Commands,
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
    mut query: Query<(
        &AircraftRole,
        &AircraftState,
        &Transform,
        &AircraftPerformance,
        &AircraftDamageState,
        &ControlInput,
        &GunState,
    )>,
) {
    if !bridge_enabled.0
        || bridge_mode.0 != BridgeRole::Client
        || !bridge_link.remote_authority_active
    {
        return;
    }
    let Some(local_role) = bridge_slot_aircraft_role(assigned_role.0) else {
        return;
    };
    for (role, state, transform, performance, damage_state, input, gun_state) in &mut query {
        let local_cooldown = match role {
            AircraftRole::Fighter1 => &mut visual_state.fighter1_cooldown_seconds,
            AircraftRole::Fighter2 => &mut visual_state.fighter2_cooldown_seconds,
        };
        if *role != local_role
            || state.is_destroyed
            || damage_state.is_repairing
            || input.repair
            || !input.fire_gun
            || *local_cooldown > 0.0
            || gun_state.overheated
        {
            continue;
        }

        let cooldown = 1.0 / performance.gun.rounds_per_second.max(0.1);
        *local_cooldown = cooldown;
        // Local immediate visuals should follow the presented aircraft pose the
        // player currently sees on screen.
        let (muzzle_position, muzzle_forward) = presentation_muzzle_pose(transform);
        let projectile_velocity =
            state.velocity + muzzle_forward * performance.gun.projectile_speed;
        let projectile_rotation =
            Quat::from_rotation_arc(Vec3::Z, projectile_velocity.normalize_or_zero());
        let mut projectile_entity = commands.spawn((
            LocalVisualProjectile {
                velocity: projectile_velocity,
                remaining_distance: performance.gun.max_range,
            },
            Transform::from_translation(muzzle_position).with_rotation(projectile_rotation),
            Visibility::default(),
        ));

        if let (Some(meshes), Some(materials)) = (meshes.as_deref_mut(), materials.as_deref_mut()) {
            let bullet_color = match role {
                AircraftRole::Fighter1 => Color::srgb(1.0, 0.88, 0.44),
                AircraftRole::Fighter2 => Color::srgb(1.0, 0.42, 0.22),
            };
            projectile_entity.with_children(|parent| {
                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.62, 0.62, 24.0))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: bullet_color,
                        emissive: LinearRgba::rgb(
                            bullet_color.to_linear().red * 40.0,
                            bullet_color.to_linear().green * 40.0,
                            bullet_color.to_linear().blue * 40.0,
                        ),
                        unlit: true,
                        alpha_mode: AlphaMode::Add,
                        ..default()
                    })),
                    Transform::default(),
                ));

                parent.spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.24, 0.24, 14.0))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(1.0, 0.98, 0.94),
                        emissive: LinearRgba::rgb(52.0, 48.0, 42.0),
                        alpha_mode: AlphaMode::Add,
                        unlit: true,
                        ..default()
                    })),
                    Transform::from_xyz(0.0, 0.0, 1.2),
                ));
            });
        }
        break;
    }
}

pub fn update_local_visual_projectiles(
    mut commands: Commands,
    time: Res<Time<Fixed>>,
    mut query: Query<(Entity, &mut LocalVisualProjectile, &mut Transform)>,
) {
    let delta_seconds = time.delta_secs();
    for (entity, mut projectile, mut transform) in &mut query {
        let travel = projectile.velocity * delta_seconds;
        projectile.remaining_distance -= travel.length();
        transform.translation += travel;
        if projectile.remaining_distance <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

pub fn update_projectiles(
    config: Res<RepositoryConfig>,
    mode: Res<BridgeMode>,
    lag_comp_state: Option<Res<BridgeServerLagCompensationState>>,
    debug: Res<SimulationDebugState>,
    frame_audio: Option<Res<FrameAudioState>>,
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut env_events: Option<ResMut<PendingEnvironmentEvents>>,
    mut damage_indicators: Option<ResMut<DamageIndicatorQueue>>,
    mut commands: Commands,
    time: Res<Time<Fixed>>,
    mut combat_state: ResMut<CombatState>,
    mut presentation_queue: Option<ResMut<CombatPresentationQueue>>,
    mut projectile_query: Query<(Entity, &mut Projectile, &mut Transform)>,
    obstacle_query: Query<(&Transform, &ObstacleCollider), Without<Projectile>>,
    mut aircraft_query: Query<(
        Entity,
        &AircraftRole,
        &mut AircraftState,
        &mut AircraftDamageState,
    )>,
) {
    let delta_seconds = time.delta_secs();
    let ground_y = config.scene.ground_height;

    for (projectile_entity, mut projectile, mut transform) in &mut projectile_query {
        let start = transform.translation;
        let travel = projectile.velocity * delta_seconds;
        let distance = travel.length();
        if distance <= f32::EPSILON {
            continue;
        }

        let end = start + travel;
        let mut hit_target = None;
        let mut shooter_position = None;
        let mut closest_hit_t = 1.1;

        if !projectile.flyby_emitted
            && projectile.shooter_role != AircraftRole::Fighter1
            && let Some(listener_position) = frame_audio
                .as_ref()
                .and_then(|frame| frame.listener_position)
            && let Some(flyby_t) = segment_sphere_hit_fraction(start, end, listener_position, 22.0)
        {
            let flyby_point = start.lerp(end, flyby_t);
            if let Some(audio_events) = audio_events.as_deref_mut() {
                audio_events.push(AudioEventKind::BulletFlyBy, flyby_point, 0.9);
            }
            projectile.flyby_emitted = true;
        }

        for (target_entity, target_role, target_state, target_damage) in &mut aircraft_query {
            if *target_role == projectile.shooter_role {
                shooter_position = Some(target_state.position);
                continue;
            }
            if target_state.is_destroyed {
                continue;
            }

            let (target_position, target_orientation) =
                if mode.0 == BridgeRole::Server && projectile.lag_compensation_ticks > 0 {
                    lag_comp_state
                        .as_deref()
                        .and_then(|lag_comp_state| {
                            sample_server_historical_aircraft_snapshot(
                                lag_comp_state,
                                *target_role,
                                debug
                                    .tick_count
                                    .saturating_sub(projectile.lag_compensation_ticks as u64),
                            )
                        })
                        .map(|snapshot| (snapshot.position, snapshot.orientation))
                        .unwrap_or((target_state.position, target_state.orientation))
                } else {
                    (target_state.position, target_state.orientation)
                };

            let mut best_target_hit = None;
            let mut best_subsystem_hit = None;
            let collision_boxes = aircraft_world_collision_boxes_for_pose(
                target_position,
                target_orientation,
                &target_damage,
            );
            for collision_box in collision_boxes {
                if let Some(hit_t) =
                    segment_obb_hit_fraction(start, end, &collision_box, projectile.hit_radius)
                {
                    if hit_t >= closest_hit_t {
                        continue;
                    }
                    let hit_point = start.lerp(end, hit_t);
                    update_best_target_hit(&mut best_target_hit, hit_t, hit_point);
                }
            }

            for hitbox in subsystem_hitboxes() {
                if target_damage.subsystem(hitbox.subsystem).stage() == DamageStage::Destroyed {
                    continue;
                }
                let collision_box = AircraftCollisionBox {
                    name: hitbox.name,
                    center: hitbox.center,
                    half_extents: hitbox.half_extents,
                }
                .to_world(target_position, target_orientation);
                if let Some(hit_t) = segment_obb_hit_fraction(
                    start,
                    end,
                    &collision_box,
                    PROJECTILE_SUBSYSTEM_HIT_RADIUS_METERS,
                ) {
                    if hit_t >= closest_hit_t {
                        continue;
                    }
                    let hit_point = start.lerp(end, hit_t);
                    update_best_subsystem_hit(
                        &mut best_subsystem_hit,
                        hit_t,
                        hitbox.subsystem,
                        hit_point,
                    );
                }
            }

            if let Some((hit_t, hit_point)) = best_target_hit {
                let subsystem_hit = best_subsystem_hit.and_then(
                    |(subsystem_hit_t, subsystem, subsystem_hit_point)| {
                        let hit_slack = subsystem_assignment_t_slack(
                            distance,
                            projectile.hit_radius,
                            PROJECTILE_SUBSYSTEM_HIT_RADIUS_METERS,
                        );
                        (subsystem_hit_t <= hit_t + hit_slack)
                            .then_some((subsystem, subsystem_hit_point))
                    },
                );
                closest_hit_t = hit_t;
                hit_target = Some((target_entity, *target_role, hit_point, subsystem_hit));
            }
        }

        let obstacle_hit_t = obstacle_query
            .iter()
            .filter_map(|(transform, collider)| {
                segment_aabb_hit_fraction(start, end, transform.translation, collider.half_extents)
            })
            .fold(None, |best, hit_t| match best {
                Some(best_t) if best_t <= hit_t => Some(best_t),
                _ => Some(hit_t),
            });
        let ground_hit_t = segment_plane_y_hit_fraction(start, end, ground_y);
        let environment_hit_t = match (obstacle_hit_t, ground_hit_t) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        if let Some(hit_t) = environment_hit_t
            && hit_t <= closest_hit_t
        {
            commands.entity(projectile_entity).despawn();
            continue;
        }

        if let Some((target_entity, target_role, hit_point, subsystem_hit)) = hit_target {
            if let Ok((_, _, mut target_state, mut damage_state)) =
                aircraft_query.get_mut(target_entity)
            {
                damage_state.interrupt_repair();
                let damage_hit_point = subsystem_hit.map(|(_, point)| point).unwrap_or(hit_point);
                if let Some((subsystem, subsystem_hit_point)) = subsystem_hit {
                    let was_destroyed =
                        damage_state.subsystem(subsystem).stage() == DamageStage::Destroyed;
                    damage_state.apply_subsystem_damage(subsystem, projectile.damage * 0.95);
                    let is_destroyed =
                        damage_state.subsystem(subsystem).stage() == DamageStage::Destroyed;
                    if let Some(env_events) = env_events.as_deref_mut() {
                        let subsystem_name = format!("{subsystem:?}");
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::Hit,
                            Some(subsystem_subject(target_role, &subsystem_name)),
                            Some(role_subject(projectile.shooter_role)),
                            Some(subsystem_hit_point),
                            Some(projectile.damage),
                            Some("gun"),
                            Some(&subsystem_name),
                        );
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::SubsystemHit,
                            Some(subsystem_subject(target_role, &subsystem_name)),
                            Some(role_subject(projectile.shooter_role)),
                            Some(subsystem_hit_point),
                            Some(projectile.damage),
                            Some("gun"),
                            Some(&subsystem_name),
                        );
                        if !was_destroyed && is_destroyed {
                            push_event_with_context(
                                env_events,
                                debug.tick_count,
                                EnvironmentEventKind::SubsystemDestroyed,
                                Some(subsystem_subject(target_role, &subsystem_name)),
                                Some(role_subject(projectile.shooter_role)),
                                Some(subsystem_hit_point),
                                Some(0.0),
                                Some("gun"),
                                Some(&subsystem_name),
                            );
                        }
                    }
                } else if let Some(env_events) = env_events.as_deref_mut() {
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::Hit,
                        Some(role_subject(target_role)),
                        Some(role_subject(projectile.shooter_role)),
                        Some(hit_point),
                        Some(projectile.damage),
                        Some("gun"),
                        None,
                    );
                }
                if let Some(env_events) = env_events.as_deref_mut() {
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::Damage,
                        Some(role_subject(target_role)),
                        Some(role_subject(projectile.shooter_role)),
                        Some(damage_hit_point),
                        Some(projectile.damage),
                        Some("gun"),
                        None,
                    );
                }
                target_state.hit_points = (target_state.hit_points - projectile.damage).max(0.0);
                if let Some(audio_events) = audio_events.as_deref_mut() {
                    audio_events.push(AudioEventKind::Hit, damage_hit_point, 1.0);
                }
                if target_role == AircraftRole::Fighter1
                    && let Some(damage_indicators) = damage_indicators.as_deref_mut()
                {
                    damage_indicators.push(shooter_position.unwrap_or(start), 1.0);
                }
                if target_state.hit_points <= 0.0 {
                    target_state.is_destroyed = true;
                    if let Some(presentation_queue) = presentation_queue.as_deref_mut() {
                        presentation_queue.destroyed.push(AircraftDestroyedEvent {
                            role: target_role,
                            position: target_state.position,
                        });
                    }
                }
                combat_state.hits_landed += 1;
            }
            commands.entity(projectile_entity).despawn();
            continue;
        }

        projectile.remaining_distance -= distance;
        transform.translation = end;
        if projectile.remaining_distance <= 0.0 {
            commands.entity(projectile_entity).despawn();
        }
    }
}

pub fn segment_intersects_sphere(
    segment_start: Vec3,
    segment_end: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> bool {
    segment_sphere_hit_fraction(segment_start, segment_end, sphere_center, sphere_radius).is_some()
}

fn segment_sphere_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<f32> {
    let direction = segment_end - segment_start;
    let a = direction.length_squared();
    if a <= f32::EPSILON {
        return (segment_start.distance(sphere_center) <= sphere_radius).then_some(0.0);
    }

    let offset = segment_start - sphere_center;
    let b = 2.0 * offset.dot(direction);
    let c = offset.length_squared() - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_discriminant = discriminant.sqrt();
    let near_t = (-b - sqrt_discriminant) / (2.0 * a);
    let far_t = (-b + sqrt_discriminant) / (2.0 * a);

    if (0.0..=1.0).contains(&near_t) {
        Some(near_t)
    } else if (0.0..=1.0).contains(&far_t) {
        Some(far_t)
    } else {
        None
    }
}

pub fn segment_aabb_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    aabb_center: Vec3,
    aabb_half_extents: Vec3,
) -> Option<f32> {
    let direction = segment_end - segment_start;
    let min = aabb_center - aabb_half_extents;
    let max = aabb_center + aabb_half_extents;

    let mut t_min: f32 = 0.0;
    let mut t_max: f32 = 1.0;

    for axis in 0..3 {
        let start = segment_start[axis];
        let delta = direction[axis];
        let slab_min = min[axis];
        let slab_max = max[axis];

        if delta.abs() <= f32::EPSILON {
            if start < slab_min || start > slab_max {
                return None;
            }
            continue;
        }

        let inv_delta = 1.0 / delta;
        let mut t1 = (slab_min - start) * inv_delta;
        let mut t2 = (slab_max - start) * inv_delta;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }

        t_min = t_min.max(t1);
        t_max = t_max.min(t2);
        if t_min > t_max {
            return None;
        }
    }

    Some(t_min.clamp(0.0, 1.0))
}

fn segment_plane_y_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    plane_y: f32,
) -> Option<f32> {
    if segment_start.y <= plane_y {
        return Some(0.0);
    }

    if segment_end.y > plane_y {
        return None;
    }

    let delta_y = segment_end.y - segment_start.y;
    if delta_y.abs() <= f32::EPSILON {
        return None;
    }

    Some(((plane_y - segment_start.y) / delta_y).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::{subsystem_assignment_t_slack, update_best_subsystem_hit, update_best_target_hit};
    use crate::gameplay::damage::AircraftSubsystem;
    use bevy::prelude::Vec3;

    #[test]
    fn best_target_hit_prefers_smallest_hit_fraction() {
        let mut best = None;
        update_best_target_hit(&mut best, 0.42, Vec3::X);
        update_best_target_hit(&mut best, 0.37, Vec3::Y);
        update_best_target_hit(&mut best, 0.55, Vec3::Z);
        let best = best.expect("expected best hit");
        assert!((best.0 - 0.37).abs() <= f32::EPSILON);
        assert_eq!(best.1, Vec3::Y);
    }

    #[test]
    fn best_subsystem_hit_prefers_smallest_hit_fraction() {
        let mut best = None;
        update_best_subsystem_hit(
            &mut best,
            0.42,
            AircraftSubsystem::Engine,
            Vec3::new(0.0, 0.0, 1.0),
        );
        update_best_subsystem_hit(
            &mut best,
            0.31,
            AircraftSubsystem::LeftWing,
            Vec3::new(1.0, 0.0, 0.0),
        );
        update_best_subsystem_hit(
            &mut best,
            0.55,
            AircraftSubsystem::RightWing,
            Vec3::new(-1.0, 0.0, 0.0),
        );
        let best = best.expect("expected best subsystem hit");
        assert!((best.0 - 0.31).abs() <= f32::EPSILON);
        assert_eq!(best.1, AircraftSubsystem::LeftWing);
        assert_eq!(best.2, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn subsystem_assignment_slack_stays_small_for_deep_hits() {
        let slack = subsystem_assignment_t_slack(24.0, 0.75, 0.35);
        assert!(slack > 0.0);
        assert!(slack < 0.03);
    }
}
