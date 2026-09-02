use bevy::ecs::system::SystemState;
use bevy::prelude::*;

use crate::api::environment::{DEFAULT_ENVIRONMENT_SEED, EnvironmentSeed};
use crate::api::events::PendingEnvironmentEvents;
use crate::api::types::{
    AircraftObservation, ArenaObservation, AudioObservation, DynamicWorldObservation,
    ObservationBundle, ObservationCaptureConfig, ProjectileObservation, StateObservation,
    SubsystemObservation, TracerObservation, VisualCaptureVariant, VisualObservation,
    VisualResolutionMode,
};
use crate::api::vision::{VisualCaptureFrames, VisualCaptureKey};
use crate::audio::collect_audio_observation;
use crate::core::config::{ConfigPaths, RepositoryConfig};
use crate::gameplay::combat::Projectile;
use crate::gameplay::damage::{
    AircraftDamageState, DamageStage, damage_flight_modifiers, effective_aircraft_performance,
};
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::input::actions::ControlInput;
use crate::presentation::tracers::TracerLifetime;
use crate::simulation::collision::{aircraft_world_collision_boxes, obb_plane_y_penetration};
use crate::simulation::components::{AircraftPerformance, AircraftRole, AircraftState, GunState};
use crate::simulation::flight_model::step_aircraft;
use crate::simulation::resources::SimulationDebugState;
use crate::simulation::systems::step_predicted_aircraft_state;
use bevy::window::{PrimaryWindow, Window};

const PULLUP_REFERENCE_HORIZON_SECONDS: f32 = 0.1;
const PULLUP_REFERENCE_SUBSTEPS: usize = 5;
const RECOVERY_REFERENCE_HORIZON_SECONDS: f32 = 12.0;
const RECOVERY_REFERENCE_SUBSTEPS: usize = 120;
const MIN_REFERENCE_SPEED_MPS: f32 = 1.0;
const MIN_TURN_RATE_RAD_S: f32 = 1e-4;

#[derive(Debug, Clone, Default, Resource)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub observation: ObservationBundle,
}

pub fn collect_observation(world: &mut World) -> ObservationBundle {
    let mut system_state: SystemState<(
        Res<SimulationDebugState>,
        Res<MatchClock>,
        Res<State<MatchPhase>>,
        Res<RepositoryConfig>,
        Res<ConfigPaths>,
        Res<ObservationCaptureConfig>,
        Option<Res<EnvironmentSeed>>,
        ResMut<PendingEnvironmentEvents>,
        Option<Res<VisualCaptureFrames>>,
        Query<(
            &AircraftRole,
            &AircraftState,
            &AircraftPerformance,
            &ControlInput,
            Option<&AircraftDamageState>,
            Option<&GunState>,
        )>,
        Query<(&Projectile, &Transform)>,
        Query<(&TracerLifetime, &Transform)>,
        Query<&Window, With<PrimaryWindow>>,
    )> = SystemState::new(world);

    let (
        tick,
        sim_time_seconds,
        match_phase_name,
        scene_name,
        seed,
        arena,
        aircraft,
        capture_config,
        visual,
        dynamic,
        events_since_last_step,
    ) = {
        let (
            debug,
            clock,
            match_phase,
            config,
            config_paths,
            capture_config,
            environment_seed,
            mut pending_events,
            capture_frames,
            query,
            projectile_query,
            tracer_query,
            window_query,
        ) = system_state.get_mut(world);

        let aircraft = query
            .iter()
            .map(|(role, state, performance, input, damage, gun)| {
                let (velocity_turn_rate_rad_s, pullup_turn_radius_m) =
                    estimate_pullup_turn_metrics(state, performance, damage);
                let boundary_times =
                    estimate_recovery_time_metrics(&config, state, performance, input, damage);
                AircraftObservation {
                    role: match role {
                        AircraftRole::Fighter1 => "fighter1".to_string(),
                        AircraftRole::Fighter2 => "fighter2".to_string(),
                    },
                    position: state.position.to_array(),
                    orientation_quat: state.orientation.to_array(),
                    linear_velocity: state.velocity.to_array(),
                    angular_velocity_deg: state.angular_rates_deg.to_array(),
                    forward: state.forward.to_array(),
                    throttle: state.throttle,
                    brake: input.brake,
                    stall_factor: state.stall_factor,
                    hit_points: state.hit_points,
                    destroyed: state.is_destroyed,
                    out_of_bounds_seconds: state.out_of_bounds_seconds,
                    ceiling_recovery_seconds: state.ceiling_recovery_seconds,
                    gun_heat: gun.map(|gun| gun.heat).unwrap_or(0.0),
                    gun_overheated: gun.map(|gun| gun.overheated).unwrap_or(false),
                    is_firing: gun.map(|gun| gun.is_firing).unwrap_or(false),
                    repairing: damage.map(|damage| damage.is_repairing).unwrap_or(false),
                    repair_elapsed_seconds: damage
                        .map(|damage| damage.repair_elapsed_seconds)
                        .unwrap_or(0.0),
                    repair_progress: damage
                        .map(|damage| {
                            (damage.repair_elapsed_seconds
                                / config.game.repair_duration_seconds.max(0.1))
                            .clamp(0.0, 1.0)
                        })
                        .unwrap_or(0.0),
                    velocity_turn_rate_rad_s,
                    pullup_turn_radius_m,
                    max_level_speed_mps: Some(performance.max_level_speed),
                    time_to_ground_impact_s: boundary_times.time_to_ground_impact_s,
                    time_to_ceiling_impact_s: boundary_times.time_to_ceiling_impact_s,
                    time_to_horizontal_boundary_impact_s: boundary_times
                        .time_to_horizontal_boundary_impact_s,
                    time_to_reenter_arena_s: boundary_times.time_to_reenter_arena_s,
                    subsystems: damage
                        .map(|damage| {
                            damage
                                .all_subsystems()
                                .into_iter()
                                .map(|(subsystem, health)| SubsystemObservation {
                                    name: format!("{subsystem:?}"),
                                    hit_points: health.current,
                                    max_hit_points: health.max,
                                    stage: match health.stage() {
                                        DamageStage::Undamaged => "undamaged".to_string(),
                                        DamageStage::Damaged => "damaged".to_string(),
                                        DamageStage::Destroyed => "destroyed".to_string(),
                                    },
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();

        let runtime_window_size = window_query
            .iter()
            .next()
            .map(|window| (window.physical_width(), window.physical_height()));

        let visual = resolve_visual_observations(
            &capture_config,
            runtime_window_size,
            capture_frames.as_deref(),
            VisualCaptureVariant::Rgb,
        );
        let dynamic = DynamicWorldObservation {
            projectiles: projectile_query
                .iter()
                .map(|(projectile, transform)| ProjectileObservation {
                    id: projectile.id,
                    shooter_role: format!("{:?}", projectile.shooter_role),
                    position: transform.translation.to_array(),
                    velocity: projectile.velocity.to_array(),
                    remaining_distance: projectile.remaining_distance,
                    damage: projectile.damage,
                    hit_radius: projectile.hit_radius,
                })
                .collect(),
            tracers: tracer_query
                .iter()
                .map(|(tracer, transform)| TracerObservation {
                    position: transform.translation.to_array(),
                    remaining_seconds: tracer.remaining_seconds,
                })
                .collect(),
        };

        (
            debug.tick_count,
            clock.elapsed_seconds,
            format!("{:?}", match_phase.get()),
            config_paths
                .scene_override
                .clone()
                .unwrap_or_else(|| config.game.active_scene.clone()),
            environment_seed
                .as_ref()
                .map(|seed| seed.effective)
                .unwrap_or(DEFAULT_ENVIRONMENT_SEED),
            ArenaObservation {
                ground_height: config.scene.ground_height,
                arena_radius: config.scene.arena_radius,
                flight_ceiling_height: config.scene.flight_ceiling_height,
                ceiling_falloff_range: config.scene.ceiling_falloff_range,
            },
            aircraft,
            capture_config.clone(),
            visual,
            dynamic,
            std::mem::take(&mut pending_events.events),
        )
    };

    ObservationBundle {
        state: StateObservation {
            tick,
            seed,
            sim_time_seconds,
            match_phase: match_phase_name,
            scene_name,
            aircraft,
            arena,
            events_since_last_step,
        },
        dynamic,
        visual,
        audio: collect_audio_observation(world, &capture_config),
    }
}

fn estimate_pullup_turn_metrics(
    state: &AircraftState,
    performance: &AircraftPerformance,
    damage: Option<&AircraftDamageState>,
) -> (Option<f32>, Option<f32>) {
    if state.is_destroyed {
        return (None, None);
    }

    let current_speed = state.velocity.length();
    if current_speed < MIN_REFERENCE_SPEED_MPS {
        return (None, None);
    }

    let current_velocity_hat = state.velocity.normalize_or_zero();
    if current_velocity_hat == Vec3::ZERO {
        return (None, None);
    }

    let default_input = ControlInput::default();
    let effective_performance = damage
        .map(|damage| effective_aircraft_performance(performance, damage, &default_input))
        .unwrap_or_else(|| performance.clone());
    let modifiers = damage
        .map(|damage| damage_flight_modifiers(damage, performance))
        .unwrap_or_default();

    let mut reference_state = state.clone();
    let mut reference_input = ControlInput::default();
    reference_input.pitch = -1.0;

    let substep_seconds = PULLUP_REFERENCE_HORIZON_SECONDS / PULLUP_REFERENCE_SUBSTEPS as f32;
    for _ in 0..PULLUP_REFERENCE_SUBSTEPS {
        step_aircraft(
            &mut reference_state,
            &effective_performance,
            &modifiers,
            &reference_input,
            substep_seconds,
        );
    }

    let future_speed = reference_state.velocity.length();
    if future_speed < MIN_REFERENCE_SPEED_MPS {
        return (None, None);
    }
    let future_velocity_hat = reference_state.velocity.normalize_or_zero();
    if future_velocity_hat == Vec3::ZERO {
        return (None, None);
    }

    let dot = current_velocity_hat
        .dot(future_velocity_hat)
        .clamp(-1.0, 1.0);
    let delta_theta = dot.acos();
    let turn_rate = delta_theta / PULLUP_REFERENCE_HORIZON_SECONDS.max(f32::EPSILON);
    if !turn_rate.is_finite() || turn_rate < MIN_TURN_RATE_RAD_S {
        return (Some(0.0), None);
    }

    let pullup_turn_radius = current_speed / turn_rate;
    (
        Some(turn_rate),
        if pullup_turn_radius.is_finite() {
            Some(pullup_turn_radius)
        } else {
            None
        },
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct RecoveryTimeMetrics {
    time_to_ground_impact_s: Option<f32>,
    time_to_ceiling_impact_s: Option<f32>,
    time_to_horizontal_boundary_impact_s: Option<f32>,
    time_to_reenter_arena_s: Option<f32>,
}

fn estimate_recovery_time_metrics(
    config: &RepositoryConfig,
    state: &AircraftState,
    performance: &AircraftPerformance,
    input: &ControlInput,
    damage: Option<&AircraftDamageState>,
) -> RecoveryTimeMetrics {
    if state.is_destroyed {
        return RecoveryTimeMetrics::default();
    }

    let Some(damage) = damage else {
        return RecoveryTimeMetrics::default();
    };

    let mut metrics = RecoveryTimeMetrics::default();
    if has_ground_impact(state, damage, config.scene.ground_height) {
        metrics.time_to_ground_impact_s = Some(0.0);
    }
    if has_ceiling_impact(state, damage, config.scene.flight_ceiling_height) {
        metrics.time_to_ceiling_impact_s = Some(0.0);
    }
    if has_horizontal_boundary_impact(state, config.scene.arena_radius) {
        metrics.time_to_horizontal_boundary_impact_s = Some(0.0);
    }
    let started_outside_arena = !is_inside_horizontal_arena(state, config.scene.arena_radius);

    let mut rollout_state = state.clone();
    let step_seconds = RECOVERY_REFERENCE_HORIZON_SECONDS / RECOVERY_REFERENCE_SUBSTEPS as f32;
    for step_idx in 1..=RECOVERY_REFERENCE_SUBSTEPS {
        step_predicted_aircraft_state(
            config,
            &mut rollout_state,
            performance,
            damage,
            input,
            step_seconds,
        );
        let sim_time = step_idx as f32 * step_seconds;

        if metrics.time_to_ground_impact_s.is_none()
            && has_ground_impact(&rollout_state, damage, config.scene.ground_height)
        {
            metrics.time_to_ground_impact_s = Some(sim_time);
        }
        if metrics.time_to_ceiling_impact_s.is_none()
            && has_ceiling_impact(&rollout_state, damage, config.scene.flight_ceiling_height)
        {
            metrics.time_to_ceiling_impact_s = Some(sim_time);
        }
        if metrics.time_to_horizontal_boundary_impact_s.is_none()
            && has_horizontal_boundary_impact(&rollout_state, config.scene.arena_radius)
        {
            metrics.time_to_horizontal_boundary_impact_s = Some(sim_time);
        }
        if started_outside_arena
            && metrics.time_to_reenter_arena_s.is_none()
            && is_inside_horizontal_arena(&rollout_state, config.scene.arena_radius)
        {
            metrics.time_to_reenter_arena_s = Some(sim_time);
        }
    }

    metrics
}

fn has_ground_impact(
    state: &AircraftState,
    damage: &AircraftDamageState,
    ground_height: f32,
) -> bool {
    aircraft_world_collision_boxes(state, damage)
        .into_iter()
        .any(|collision_box| {
            obb_plane_y_penetration(&collision_box, ground_height, Vec3::Y).is_some()
        })
}

fn has_ceiling_impact(
    state: &AircraftState,
    damage: &AircraftDamageState,
    ceiling_height: f32,
) -> bool {
    aircraft_world_collision_boxes(state, damage)
        .into_iter()
        .any(|collision_box| {
            obb_plane_y_penetration(&collision_box, ceiling_height, -Vec3::Y).is_some()
        })
}

fn has_horizontal_boundary_impact(state: &AircraftState, arena_radius: f32) -> bool {
    Vec2::new(state.position.x, state.position.z).length() >= arena_radius
}

fn is_inside_horizontal_arena(state: &AircraftState, arena_radius: f32) -> bool {
    !has_horizontal_boundary_impact(state, arena_radius)
}

pub fn collect_visual_observations_only(world: &mut World) -> Vec<VisualObservation> {
    collect_visual_observations_for_variant(world, VisualCaptureVariant::Rgb)
}

pub fn collect_visual_observations_for_variant(
    world: &mut World,
    variant: VisualCaptureVariant,
) -> Vec<VisualObservation> {
    let mut system_state: SystemState<(
        Res<ObservationCaptureConfig>,
        Option<Res<VisualCaptureFrames>>,
        Query<&Window, With<PrimaryWindow>>,
    )> = SystemState::new(world);

    let (capture_config, capture_frames, window_query) = system_state.get_mut(world);
    let runtime_window_size = window_query
        .iter()
        .next()
        .map(|window| (window.physical_width(), window.physical_height()));
    resolve_visual_observations(
        &capture_config,
        runtime_window_size,
        capture_frames.as_deref(),
        variant,
    )
}

pub fn collect_audio_observation_only(world: &mut World) -> Option<AudioObservation> {
    let mut system_state: SystemState<Res<ObservationCaptureConfig>> = SystemState::new(world);
    let capture_config = system_state.get_mut(world).clone();
    collect_audio_observation(world, &capture_config)
}

pub fn capture_world_snapshot(world: &mut World) {
    let snapshot_state = collect_observation(world);
    let tick = snapshot_state.state.tick;
    {
        let mut observation = world.resource_mut::<ObservationBundle>();
        *observation = snapshot_state.clone();
    }
    {
        let mut snapshot = world.resource_mut::<WorldSnapshot>();
        snapshot.tick = tick;
        snapshot.observation = snapshot_state;
    }
}

fn resolve_visual_observations(
    capture_config: &ObservationCaptureConfig,
    runtime_window_size: Option<(u32, u32)>,
    capture_frames: Option<&VisualCaptureFrames>,
    variant: VisualCaptureVariant,
) -> Vec<VisualObservation> {
    if !capture_config.enable_visual {
        return Vec::new();
    }

    capture_config
        .visual_sensors
        .iter()
        .map(|sensor| {
            let (width, height) = match sensor.resolution_mode {
                VisualResolutionMode::Fixed => (sensor.width, sensor.height),
                VisualResolutionMode::RuntimeWindow => {
                    runtime_window_size.unwrap_or((sensor.width, sensor.height))
                }
            };
            let bytes = capture_frames
                .and_then(|frames| {
                    frames.frames.get(&VisualCaptureKey {
                        kind: sensor.kind,
                        variant,
                    })
                })
                .filter(|frame| frame.width == width && frame.height == height)
                .map(|frame| frame.bytes.clone())
                .unwrap_or_default();
            let bytes_ready = !bytes.is_empty();
            let (format, bytes) = match variant {
                VisualCaptureVariant::Semantic => (
                    crate::api::types::PixelFormat::Gray8,
                    semantic_rgba_to_class_ids(&bytes),
                ),
                VisualCaptureVariant::Rgb => {
                    let bytes = match sensor.format {
                        crate::api::types::PixelFormat::Rgba8 => bytes,
                        crate::api::types::PixelFormat::Rgb8 => rgba_to_rgb(&bytes),
                        crate::api::types::PixelFormat::Gray8 => rgba_to_gray(&bytes),
                    };
                    (sensor.format, bytes)
                }
            };
            VisualObservation {
                camera: sensor.kind,
                width,
                height,
                format,
                resolution_mode: sensor.resolution_mode,
                include_hud: sensor.include_hud,
                bytes_ready,
                bytes,
            }
        })
        .collect()
}

fn rgba_to_rgb(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
        .collect()
}

fn rgba_to_gray(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .map(|rgba| {
            let r = rgba[0] as f32;
            let g = rgba[1] as f32;
            let b = rgba[2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

pub(crate) fn semantic_rgba_to_class_ids(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .map(class_id_from_semantic_rgba)
        .collect()
}

pub(crate) fn class_id_from_semantic_rgba(rgba: &[u8]) -> u8 {
    let r = rgba[0];
    let g = rgba[1];
    let b = rgba[2];
    if r == 0 && g == 0 && b == 0 {
        return 0;
    }
    if g > r && g > b {
        return 1;
    }
    if r > g && r > b {
        return 2;
    }

    let bg = [0u8, 0u8, 0u8];
    let fighter1 = [0u8, 255u8, 0u8];
    let fighter2 = [255u8, 0u8, 0u8];
    let d0 = semantic_rgb_distance_sq([r, g, b], bg);
    let d1 = semantic_rgb_distance_sq([r, g, b], fighter1);
    let d2 = semantic_rgb_distance_sq([r, g, b], fighter2);
    if d0 <= d1 && d0 <= d2 {
        0
    } else if d1 <= d2 {
        1
    } else {
        2
    }
}

fn semantic_rgb_distance_sq(lhs: [u8; 3], rhs: [u8; 3]) -> u32 {
    let dr = lhs[0] as i32 - rhs[0] as i32;
    let dg = lhs[1] as i32 - rhs[1] as i32;
    let db = lhs[2] as i32 - rhs[2] as i32;
    (dr * dr + dg * dg + db * db) as u32
}

#[cfg(test)]
mod tests {
    use super::{class_id_from_semantic_rgba, semantic_rgba_to_class_ids};

    #[test]
    fn semantic_rgba_maps_to_nearest_valid_class_id() {
        assert_eq!(class_id_from_semantic_rgba(&[0, 0, 0, 255]), 0);
        assert_eq!(class_id_from_semantic_rgba(&[0, 255, 0, 255]), 1);
        assert_eq!(class_id_from_semantic_rgba(&[255, 0, 0, 255]), 2);
        assert_eq!(class_id_from_semantic_rgba(&[0, 32, 0, 255]), 1);
        assert_eq!(class_id_from_semantic_rgba(&[32, 0, 0, 255]), 2);
        assert_eq!(class_id_from_semantic_rgba(&[255, 255, 0, 255]), 1);

        assert_eq!(
            semantic_rgba_to_class_ids(&[
                0, 0, 0, 255, //
                0, 255, 0, 255, //
                255, 0, 0, 255, //
                16, 0, 0, 255,
            ]),
            vec![0, 1, 2, 2]
        );
    }
}
