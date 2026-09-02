use bevy::prelude::*;

use crate::api::events::{
    PendingEnvironmentEvents, push_event_with_context, role_subject, subsystem_subject,
};
use crate::api::types::EnvironmentEventKind;
use crate::audio::{AudioEventKind, AudioEventQueue};
use crate::gameplay::combat::{AircraftDestroyedEvent, CombatPresentationQueue};
use crate::gameplay::damage::{AircraftDamageState, apply_collision_damage};
use crate::presentation::hud::DamageIndicatorQueue;
use crate::simulation::collision::{
    ObstacleCollider, aircraft_world_collision_boxes, aircraft_world_collision_boxes_for_pose,
    obb_aabb_penetration, obb_obb_penetration, obb_plane_y_penetration,
};
use crate::simulation::components::{AircraftRole, AircraftState};
use crate::simulation::resources::SimulationDebugState;

const GROUND_DAMAGE_SCALE: f32 = 1.9;
const OBSTACLE_DAMAGE_SCALE: f32 = 1.75;
const AIRCRAFT_DAMAGE_SCALE: f32 = 1.1;
const OBSTACLE_BOUNCE_MIN_SPEED: f32 = 20.0;
const AIRCRAFT_BOUNCE_MIN_SPEED: f32 = 14.0;
const GROUND_MIN_DAMAGE_SPEED: f32 = 24.0;
const OBSTACLE_MIN_DAMAGE_SPEED: f32 = 22.0;
const AIRCRAFT_MIN_DAMAGE_SPEED: f32 = 20.0;

#[derive(Debug, Clone)]
struct AircraftSnapshot {
    entity: Entity,
    position: Vec3,
    velocity: Vec3,
    orientation: Quat,
    damage: AircraftDamageState,
    is_destroyed: bool,
}

pub fn resolve_environment_collisions(
    ground_height: Res<crate::core::config::RepositoryConfig>,
    debug: Res<SimulationDebugState>,
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut env_events: Option<ResMut<PendingEnvironmentEvents>>,
    mut damage_indicators: Option<ResMut<DamageIndicatorQueue>>,
    mut presentation_queue: Option<ResMut<CombatPresentationQueue>>,
    obstacle_query: Query<(&Transform, &ObstacleCollider)>,
    mut aircraft_query: Query<(
        Entity,
        &AircraftRole,
        &mut AircraftState,
        &mut AircraftDamageState,
    )>,
) {
    let ground_y = ground_height.scene.ground_height;

    for (_, role, mut state, mut damage) in &mut aircraft_query {
        if state.is_destroyed {
            continue;
        }

        let mut destroyed = false;
        if let Some((contact, penetration)) = lowest_ground_contact(&state, &damage, ground_y) {
            let vertical_impact = (-state.velocity.y).max(0.0);
            let impact_speed = punitive_collision_speed(
                vertical_impact * 0.95 + state.velocity.xz().length() * 0.45,
                GROUND_MIN_DAMAGE_SPEED,
            );
            let local_hit = state.orientation.inverse() * (contact - state.position);
            let (total_damage, hit_subsystem, subsystem_destroyed) = apply_collision_damage(
                &mut state,
                &mut damage,
                local_hit,
                impact_speed,
                GROUND_DAMAGE_SCALE,
            );
            if let Some(env_events) = env_events.as_deref_mut() {
                push_event_with_context(
                    env_events,
                    debug.tick_count,
                    EnvironmentEventKind::Collision,
                    Some(role_subject(*role)),
                    None,
                    Some(contact),
                    Some(impact_speed),
                    Some("ground"),
                    None,
                );
                if total_damage > 0.0 {
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::Damage,
                        Some(role_subject(*role)),
                        None,
                        Some(contact),
                        Some(total_damage),
                        Some("collision"),
                        None,
                    );
                    if let Some(subsystem) = hit_subsystem {
                        let subsystem_name = format!("{subsystem:?}");
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::SubsystemHit,
                            Some(subsystem_subject(*role, &subsystem_name)),
                            None,
                            Some(contact),
                            Some(total_damage),
                            Some("collision"),
                            Some(&subsystem_name),
                        );
                        if subsystem_destroyed {
                            push_event_with_context(
                                env_events,
                                debug.tick_count,
                                EnvironmentEventKind::SubsystemDestroyed,
                                Some(subsystem_subject(*role, &subsystem_name)),
                                None,
                                Some(contact),
                                Some(0.0),
                                Some("collision"),
                                Some(&subsystem_name),
                            );
                        }
                    }
                }
            }
            if let Some(audio_events) = audio_events.as_deref_mut() {
                audio_events.push(
                    AudioEventKind::Hit,
                    contact,
                    collision_audio_volume(impact_speed, 38.0),
                );
            }
            if *role == AircraftRole::Fighter1
                && let Some(damage_indicators) = damage_indicators.as_deref_mut()
            {
                damage_indicators.push(contact, collision_audio_volume(impact_speed, 38.0));
            }
            state.position.y += penetration.max(0.0);
            if state.velocity.y < 0.0 {
                state.velocity.y = (-state.velocity.y * 0.45).max(OBSTACLE_BOUNCE_MIN_SPEED * 0.35);
            }
            state.velocity.x *= 0.52;
            state.velocity.z *= 0.52;
            destroyed = state.is_destroyed;
        }

        if state.is_destroyed {
            destroyed = true;
        } else {
            for (transform, collider) in &obstacle_query {
                let center = transform.translation;
                let Some((contact, normal, penetration)) =
                    aircraft_aabb_penetration(&state, &damage, center, collider.half_extents)
                else {
                    continue;
                };

                let relative_speed =
                    punitive_collision_speed(state.velocity.length(), OBSTACLE_MIN_DAMAGE_SPEED);
                let local_hit = state.orientation.inverse() * (contact - state.position);
                let (total_damage, hit_subsystem, subsystem_destroyed) = apply_collision_damage(
                    &mut state,
                    &mut damage,
                    local_hit,
                    relative_speed,
                    OBSTACLE_DAMAGE_SCALE * collider.damage_scale,
                );
                if let Some(env_events) = env_events.as_deref_mut() {
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::Collision,
                        Some(role_subject(*role)),
                        None,
                        Some(contact),
                        Some(relative_speed),
                        Some("obstacle"),
                        None,
                    );
                    if total_damage > 0.0 {
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::Damage,
                            Some(role_subject(*role)),
                            None,
                            Some(contact),
                            Some(total_damage),
                            Some("collision"),
                            None,
                        );
                        if let Some(subsystem) = hit_subsystem {
                            let subsystem_name = format!("{subsystem:?}");
                            push_event_with_context(
                                env_events,
                                debug.tick_count,
                                EnvironmentEventKind::SubsystemHit,
                                Some(subsystem_subject(*role, &subsystem_name)),
                                None,
                                Some(contact),
                                Some(total_damage),
                                Some("collision"),
                                Some(&subsystem_name),
                            );
                            if subsystem_destroyed {
                                push_event_with_context(
                                    env_events,
                                    debug.tick_count,
                                    EnvironmentEventKind::SubsystemDestroyed,
                                    Some(subsystem_subject(*role, &subsystem_name)),
                                    None,
                                    Some(contact),
                                    Some(0.0),
                                    Some("collision"),
                                    Some(&subsystem_name),
                                );
                            }
                        }
                    }
                }
                if let Some(audio_events) = audio_events.as_deref_mut() {
                    audio_events.push(
                        AudioEventKind::Hit,
                        contact,
                        collision_audio_volume(relative_speed, 55.0),
                    );
                }
                if *role == AircraftRole::Fighter1
                    && let Some(damage_indicators) = damage_indicators.as_deref_mut()
                {
                    damage_indicators.push(contact, collision_audio_volume(relative_speed, 55.0));
                }
                state.position += normal * penetration.max(0.0);
                let normal_speed = state.velocity.dot(normal);
                if normal_speed < 0.0 {
                    state.velocity -= normal * (normal_speed * 1.35);
                }
                let tangential_velocity = state.velocity - normal * state.velocity.dot(normal);
                let bounce_speed = (relative_speed * 0.82).max(OBSTACLE_BOUNCE_MIN_SPEED);
                state.velocity = tangential_velocity * 0.26 + normal * bounce_speed;
                destroyed = state.is_destroyed;
                break;
            }
        }

        if destroyed && let Some(presentation_queue) = presentation_queue.as_deref_mut() {
            presentation_queue.destroyed.push(AircraftDestroyedEvent {
                role: *role,
                position: state.position,
            });
        }
    }
}

pub fn resolve_aircraft_collisions(
    debug: Res<SimulationDebugState>,
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut env_events: Option<ResMut<PendingEnvironmentEvents>>,
    mut damage_indicators: Option<ResMut<DamageIndicatorQueue>>,
    mut presentation_queue: Option<ResMut<CombatPresentationQueue>>,
    mut aircraft_query: ParamSet<(
        Query<(Entity, &AircraftRole, &AircraftState, &AircraftDamageState)>,
        Query<(
            Entity,
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
        )>,
    )>,
) {
    let snapshots: Vec<_> = aircraft_query
        .p0()
        .iter()
        .map(|(entity, _role, state, damage)| AircraftSnapshot {
            entity,
            position: state.position,
            velocity: state.velocity,
            orientation: state.orientation,
            damage: damage.clone(),
            is_destroyed: state.is_destroyed,
        })
        .collect();

    let mut collisions = Vec::new();
    for i in 0..snapshots.len() {
        for j in (i + 1)..snapshots.len() {
            let a = &snapshots[i];
            let b = &snapshots[j];
            if a.is_destroyed || b.is_destroyed {
                continue;
            }
            let Some((normal, penetration, contact_a, contact_b)) = aircraft_aircraft_contact(a, b)
            else {
                continue;
            };
            let relative_speed = punitive_collision_speed(
                (a.velocity - b.velocity).length(),
                AIRCRAFT_MIN_DAMAGE_SPEED,
            );
            collisions.push((
                a,
                b,
                normal,
                penetration,
                relative_speed,
                contact_a,
                contact_b,
            ));
        }
    }

    let mut query = aircraft_query.p1();
    for (a, b, normal, penetration, relative_speed, contact_a, contact_b) in collisions {
        let Ok(
            [
                (_, role_a, mut state_a, mut damage_a),
                (_, role_b, mut state_b, mut damage_b),
            ],
        ) = query.get_many_mut([a.entity, b.entity])
        else {
            continue;
        };
        if state_a.is_destroyed || state_b.is_destroyed {
            continue;
        }

        let local_hit_a = state_a.orientation.inverse() * (contact_a - state_a.position);
        let local_hit_b = state_b.orientation.inverse() * (contact_b - state_b.position);
        let (damage_amount_a, hit_subsystem_a, subsystem_destroyed_a) = apply_collision_damage(
            &mut state_a,
            &mut damage_a,
            local_hit_a,
            relative_speed,
            AIRCRAFT_DAMAGE_SCALE,
        );
        let (damage_amount_b, hit_subsystem_b, subsystem_destroyed_b) = apply_collision_damage(
            &mut state_b,
            &mut damage_b,
            local_hit_b,
            relative_speed,
            AIRCRAFT_DAMAGE_SCALE,
        );
        let impact_center = (contact_a + contact_b) * 0.5;
        if let Some(env_events) = env_events.as_deref_mut() {
            push_event_with_context(
                env_events,
                debug.tick_count,
                EnvironmentEventKind::Collision,
                Some(role_subject(*role_a)),
                Some(role_subject(*role_b)),
                Some(contact_a),
                Some(relative_speed),
                Some("aircraft"),
                None,
            );
            push_event_with_context(
                env_events,
                debug.tick_count,
                EnvironmentEventKind::Collision,
                Some(role_subject(*role_b)),
                Some(role_subject(*role_a)),
                Some(contact_b),
                Some(relative_speed),
                Some("aircraft"),
                None,
            );
            if damage_amount_a > 0.0 {
                push_event_with_context(
                    env_events,
                    debug.tick_count,
                    EnvironmentEventKind::Damage,
                    Some(role_subject(*role_a)),
                    Some(role_subject(*role_b)),
                    Some(contact_a),
                    Some(damage_amount_a),
                    Some("collision"),
                    None,
                );
                if let Some(subsystem) = hit_subsystem_a {
                    let subsystem_name = format!("{subsystem:?}");
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::SubsystemHit,
                        Some(subsystem_subject(*role_a, &subsystem_name)),
                        Some(role_subject(*role_b)),
                        Some(contact_a),
                        Some(damage_amount_a),
                        Some("collision"),
                        Some(&subsystem_name),
                    );
                    if subsystem_destroyed_a {
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::SubsystemDestroyed,
                            Some(subsystem_subject(*role_a, &subsystem_name)),
                            Some(role_subject(*role_b)),
                            Some(contact_a),
                            Some(0.0),
                            Some("collision"),
                            Some(&subsystem_name),
                        );
                    }
                }
            }
            if damage_amount_b > 0.0 {
                push_event_with_context(
                    env_events,
                    debug.tick_count,
                    EnvironmentEventKind::Damage,
                    Some(role_subject(*role_b)),
                    Some(role_subject(*role_a)),
                    Some(contact_b),
                    Some(damage_amount_b),
                    Some("collision"),
                    None,
                );
                if let Some(subsystem) = hit_subsystem_b {
                    let subsystem_name = format!("{subsystem:?}");
                    push_event_with_context(
                        env_events,
                        debug.tick_count,
                        EnvironmentEventKind::SubsystemHit,
                        Some(subsystem_subject(*role_b, &subsystem_name)),
                        Some(role_subject(*role_a)),
                        Some(contact_b),
                        Some(damage_amount_b),
                        Some("collision"),
                        Some(&subsystem_name),
                    );
                    if subsystem_destroyed_b {
                        push_event_with_context(
                            env_events,
                            debug.tick_count,
                            EnvironmentEventKind::SubsystemDestroyed,
                            Some(subsystem_subject(*role_b, &subsystem_name)),
                            Some(role_subject(*role_a)),
                            Some(contact_b),
                            Some(0.0),
                            Some("collision"),
                            Some(&subsystem_name),
                        );
                    }
                }
            }
        }
        let hit_volume = collision_audio_volume(relative_speed, 60.0);
        if let Some(audio_events) = audio_events.as_deref_mut() {
            audio_events.push(AudioEventKind::Hit, impact_center, hit_volume);
        }
        if *role_a == AircraftRole::Fighter1
            && let Some(damage_indicators) = damage_indicators.as_deref_mut()
        {
            damage_indicators.push(contact_a, hit_volume);
        }
        if *role_b == AircraftRole::Fighter1
            && let Some(damage_indicators) = damage_indicators.as_deref_mut()
        {
            damage_indicators.push(contact_b, hit_volume);
        }

        state_a.position -= normal * (penetration * 0.5);
        state_b.position += normal * (penetration * 0.5);
        let tangential_a = state_a.velocity - normal * state_a.velocity.dot(normal);
        let tangential_b = state_b.velocity + normal * (-state_b.velocity.dot(normal));
        let bounce_speed = (relative_speed * 0.45).max(AIRCRAFT_BOUNCE_MIN_SPEED);
        state_a.velocity = tangential_a * 0.36 - normal * bounce_speed;
        state_b.velocity = tangential_b * 0.36 + normal * bounce_speed;

        if state_a.is_destroyed
            && let Some(presentation_queue) = presentation_queue.as_deref_mut()
        {
            presentation_queue.destroyed.push(AircraftDestroyedEvent {
                role: *role_a,
                position: state_a.position,
            });
        }
        if state_b.is_destroyed
            && let Some(presentation_queue) = presentation_queue.as_deref_mut()
        {
            presentation_queue.destroyed.push(AircraftDestroyedEvent {
                role: *role_b,
                position: state_b.position,
            });
        }
    }
}

fn collision_audio_volume(relative_speed: f32, reference_speed: f32) -> f32 {
    (relative_speed / reference_speed.max(1.0)).clamp(0.25, 1.2)
}

fn punitive_collision_speed(measured_speed: f32, minimum_damage_speed: f32) -> f32 {
    measured_speed.max(minimum_damage_speed)
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

fn aircraft_aircraft_contact(
    a: &AircraftSnapshot,
    b: &AircraftSnapshot,
) -> Option<(Vec3, f32, Vec3, Vec3)> {
    let mut best_contact = None;
    let mut best_penetration = 0.0;
    let boxes_a = aircraft_world_collision_boxes_for_pose(a.position, a.orientation, &a.damage);
    let boxes_b = aircraft_world_collision_boxes_for_pose(b.position, b.orientation, &b.damage);
    for box_a in &boxes_a {
        for box_b in &boxes_b {
            let Some((_, normal, penetration)) = obb_obb_penetration(box_a, box_b) else {
                continue;
            };
            if penetration <= best_penetration {
                continue;
            }
            best_penetration = penetration;
            let normal = normal.normalize_or_zero();
            best_contact = Some((
                normal,
                penetration,
                box_a.support_point(normal),
                box_b.support_point(-normal),
            ));
        }
    }
    best_contact
}
