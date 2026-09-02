use bevy::prelude::*;
use rand::{RngExt, seq::SliceRandom};

use crate::api::environment::DeterministicRng;
use crate::api::types::EnvironmentAction;
use crate::app::schedules::SimulationSet;
use crate::bridge::protocol::BridgeRole;
use crate::bridge::{
    AssignedControlRole, BridgeClientConnectionStatus, BridgeLinkState, BridgeMode, LocalPilotMode,
    bridge_slot_aircraft_role,
};
use crate::core::config::RepositoryConfig;
use crate::gameplay::damage::{
    AircraftDamageState, damage_flight_modifiers, effective_aircraft_performance,
};
use crate::gameplay::match_state::MatchPhase;
use crate::input::actions::ControlInput;
use crate::simulation::components::{
    AircraftPerformance, AircraftRole, AircraftState, ControlAuthority, GunState,
};
use crate::simulation::flight_model::step_aircraft;

const BOUNDARY_COLLISION_RADIUS_METERS: f32 = 5.4;
const BOUNDARY_PREDICTION_HORIZON_SECONDS: f32 = 3.0;
const BOUNDARY_PREDICTION_STEP_SECONDS: f32 = 0.05;
const BOUNDARY_RECOVERY_HOLD_SECONDS: f32 = 2.0;
const BOUNDARY_RECOVERY_TARGET_NORMAL_COS: f32 = 0.5;
const BOUNDARY_RECOVERY_PITCH_GAIN: f32 = 3.0;
const BOUNDARY_RECOVERY_YAW_GAIN: f32 = 1.1;
const BOUNDARY_RECOVERY_ROLL_GAIN: f32 = 3.0;
const TEACHER_BOUNDARY_RECOVERY_MIN_BLEND: f32 = 0.65;
const FOCUS_FIRE_TRIGGER_ALIGNMENT_COS: f32 = 0.95;
const FOCUS_FIRE_TRIGGER_SIDE_COS: f32 = 0.6;
const FOCUS_FIRE_TRIGGER_RATE_PER_SECOND: f32 = 1.35;
const FOCUS_FIRE_MIN_SECONDS: f32 = 2.0;
const FOCUS_FIRE_MAX_SECONDS: f32 = 5.0;
const IMPERFECTION_TRIGGER_RATE_PER_SECOND: f32 = 0.14;
const IMPERFECTION_MIN_SECONDS: f32 = 0.3;
const IMPERFECTION_MAX_SECONDS: f32 = 2.0;
const CLOSE_SPEED_CONTROL_DISTANCE_METERS: f32 = 20.0;
const GENERAL_FOLLOW_MIN_THROTTLE: f32 = 0.80;
const GENERAL_FOLLOW_MAX_THROTTLE: f32 = 1.00;
const TAIL_SPEED_MIN_THROTTLE: f32 = 0.0;
const TACTICAL_BRAKE_DISTANCE_METERS: f32 = 50.0;
const TACTICAL_BRAKE_THROTTLE_DELTA_THRESHOLD: f32 = -0.05;
const TACTICAL_BRAKE_MAX_STALL_FACTOR: f32 = 0.2;
const TACTICAL_BRAKE_MIN_RECOVERY_SPEED_SCALE: f32 = 1.05;
const AI_FIRE_ALIGNMENT_COS: f32 = 0.866_025_4;
const AI_FIRE_MAX_RANGE_FRACTION: f32 = 0.95;
const AI_FIRE_HEAT_LIMIT: f32 = 0.8;
const REPAIR_DAMAGE_FRACTION_THRESHOLD: f32 = 0.5;
const REPAIR_MAX_STALL_FACTOR: f32 = 0.1;
const REPAIR_COMPLETION_MARGIN_SECONDS: f32 = 0.75;
const REPAIR_PREDICTION_STEP_SECONDS: f32 = 0.1;
const REPAIR_MIN_AIRCRAFT_SEPARATION_METERS: f32 = 80.0;
const REPAIR_THREAT_ALIGNMENT_COS: f32 = std::f32::consts::FRAC_1_SQRT_2;
const REPAIR_THREAT_DISTANCE_REFERENCE_METERS: f32 = 250.0;
const REPAIR_MAX_THREAT_SCORE: f32 = 0.15;

pub struct BasicFighterAiPlugin;

impl Plugin for BasicFighterAiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BuiltInAiProfileOverrides>();
        app.add_systems(
            FixedUpdate,
            update_basic_fighter_controls.in_set(SimulationSet::GatherInput),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryKind {
    Ground,
    Ceiling,
    Horizontal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiDriveProfile {
    PreciseFollow,
    ImperfectBuiltIn,
    Teacher,
    PassiveBounce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInAiProfile {
    PreciseFollow,
    Imperfect,
    Teacher,
    PassiveBounce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Resource)]
pub struct BuiltInAiProfileOverrides {
    pub fighter1: Option<BuiltInAiProfile>,
    pub fighter2: Option<BuiltInAiProfile>,
}

impl BuiltInAiProfileOverrides {
    fn get(self, role: AircraftRole) -> Option<BuiltInAiProfile> {
        match role {
            AircraftRole::Fighter1 => self.fighter1,
            AircraftRole::Fighter2 => self.fighter2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BoundaryRecoveryState {
    kind: BoundaryKind,
    inward_normal_world: Vec3,
    hold_remaining_seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FocusFireState {
    hold_remaining_seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FollowAiImperfectionState {
    hold_remaining_seconds: f32,
    target_bias_local: Vec3,
    distance_scale: f32,
    pitch_multiplier: f32,
    yaw_multiplier: f32,
    roll_multiplier: f32,
    throttle_multiplier: f32,
    fire_inhibited: bool,
    focus_fire_inhibited: bool,
}

#[derive(Debug, Clone)]
struct RoleMemory<T: Copy> {
    fighter1: Option<T>,
    fighter2: Option<T>,
}

impl<T: Copy> Default for RoleMemory<T> {
    fn default() -> Self {
        Self {
            fighter1: None,
            fighter2: None,
        }
    }
}

impl<T: Copy> RoleMemory<T> {
    fn get(&self, role: AircraftRole) -> Option<T> {
        match role {
            AircraftRole::Fighter1 => self.fighter1,
            AircraftRole::Fighter2 => self.fighter2,
        }
    }

    fn set(&mut self, role: AircraftRole, state: Option<T>) {
        match role {
            AircraftRole::Fighter1 => self.fighter1 = state,
            AircraftRole::Fighter2 => self.fighter2 = state,
        }
    }
}

type BoundaryRecoveryMemory = RoleMemory<BoundaryRecoveryState>;
type FocusFireMemory = RoleMemory<FocusFireState>;
type ImperfectionMemory = RoleMemory<FollowAiImperfectionState>;

fn predicted_boundary_hit_kind(
    config: &RepositoryConfig,
    state: &AircraftState,
    performance: &AircraftPerformance,
    damage: &AircraftDamageState,
    input: &ControlInput,
) -> Option<BoundaryRecoveryState> {
    predicted_boundary_hit(config, state, performance, damage, input).map(|(recovery, _)| recovery)
}

fn predicted_boundary_hit(
    config: &RepositoryConfig,
    state: &AircraftState,
    performance: &AircraftPerformance,
    damage: &AircraftDamageState,
    input: &ControlInput,
) -> Option<(BoundaryRecoveryState, f32)> {
    let mut predicted = state.clone();
    let effective = effective_aircraft_performance(performance, damage, input);
    let modifiers = damage_flight_modifiers(damage, &effective);
    let mut elapsed = 0.0;
    while elapsed < BOUNDARY_PREDICTION_HORIZON_SECONDS {
        step_aircraft(
            &mut predicted,
            &effective,
            &modifiers,
            input,
            BOUNDARY_PREDICTION_STEP_SECONDS,
        );
        elapsed += BOUNDARY_PREDICTION_STEP_SECONDS;
        if predicted.position.y - BOUNDARY_COLLISION_RADIUS_METERS <= config.scene.ground_height {
            return Some((
                BoundaryRecoveryState {
                    kind: BoundaryKind::Ground,
                    inward_normal_world: Vec3::Y,
                    hold_remaining_seconds: BOUNDARY_RECOVERY_HOLD_SECONDS,
                },
                elapsed,
            ));
        }
        if predicted.position.y + BOUNDARY_COLLISION_RADIUS_METERS
            >= config.scene.flight_ceiling_height
        {
            return Some((
                BoundaryRecoveryState {
                    kind: BoundaryKind::Ceiling,
                    inward_normal_world: -Vec3::Y,
                    hold_remaining_seconds: BOUNDARY_RECOVERY_HOLD_SECONDS,
                },
                elapsed,
            ));
        }
        let radial = Vec2::new(predicted.position.x, predicted.position.z);
        let radial_distance = radial.length();
        if radial_distance + BOUNDARY_COLLISION_RADIUS_METERS >= config.scene.arena_radius
            && radial_distance > 1e-3
        {
            return Some((
                BoundaryRecoveryState {
                    kind: BoundaryKind::Horizontal,
                    inward_normal_world: Vec3::new(
                        -predicted.position.x,
                        0.0,
                        -predicted.position.z,
                    )
                    .normalize(),
                    hold_remaining_seconds: BOUNDARY_RECOVERY_HOLD_SECONDS,
                },
                elapsed,
            ));
        }
    }
    None
}

fn recovery_throttle(kind: BoundaryKind) -> f32 {
    match kind {
        BoundaryKind::Ground => 1.0,
        BoundaryKind::Ceiling => 0.25,
        BoundaryKind::Horizontal => 1.0,
    }
}

fn recovery_brake(kind: BoundaryKind) -> bool {
    matches!(kind, BoundaryKind::Ceiling)
}

fn nearest_safe_velocity_direction(state: &AircraftState, inward_normal_world: Vec3) -> Vec3 {
    let normal = inward_normal_world.normalize_or_zero();
    let velocity_hat = state.velocity.normalize_or_zero();
    if velocity_hat == Vec3::ZERO {
        return normal;
    }

    let alignment = velocity_hat.dot(normal);
    if alignment >= BOUNDARY_RECOVERY_TARGET_NORMAL_COS {
        return velocity_hat;
    }

    let tangent = velocity_hat - normal * alignment;
    let tangent_dir = if tangent.length_squared() > 1e-6 {
        tangent.normalize()
    } else {
        let fallback = state.forward - normal * state.forward.dot(normal);
        if fallback.length_squared() > 1e-6 {
            fallback.normalize()
        } else {
            Vec3::X
        }
    };

    let tangent_scale = (1.0 - BOUNDARY_RECOVERY_TARGET_NORMAL_COS.powi(2)).sqrt();
    (normal * BOUNDARY_RECOVERY_TARGET_NORMAL_COS + tangent_dir * tangent_scale).normalize()
}

fn apply_boundary_recovery(
    state: &AircraftState,
    input: &mut ControlInput,
    recovery: BoundaryRecoveryState,
) {
    let inward_normal_world = recovery.inward_normal_world.normalize_or_zero();
    let current_alignment = state.velocity.normalize_or_zero().dot(inward_normal_world);
    let deficit = (BOUNDARY_RECOVERY_TARGET_NORMAL_COS - current_alignment).max(0.0)
        / BOUNDARY_RECOVERY_TARGET_NORMAL_COS.max(1e-6);
    let strength = (0.45 + 0.55 * deficit).clamp(0.45, 1.0);
    let target_velocity_dir = nearest_safe_velocity_direction(state, inward_normal_world);
    let target_local = (state.orientation.inverse() * target_velocity_dir).normalize_or_zero();
    let target_pitch = (-target_local.y * BOUNDARY_RECOVERY_PITCH_GAIN).clamp(-1.0, 1.0);
    let target_yaw = (target_local.x * BOUNDARY_RECOVERY_YAW_GAIN).clamp(-1.0, 1.0);
    let target_roll = (-target_local.x * BOUNDARY_RECOVERY_ROLL_GAIN).clamp(-1.0, 1.0);

    input.pitch = input.pitch + (target_pitch - input.pitch) * strength;
    input.yaw = input.yaw + (target_yaw - input.yaw) * strength;
    input.roll = input.roll + (target_roll - input.roll) * strength;

    let desired_throttle = recovery_throttle(recovery.kind);
    input.throttle_delta = if state.throttle + 0.02 < desired_throttle {
        1.0
    } else if state.throttle - 0.02 > desired_throttle {
        -1.0
    } else {
        0.0
    };
    input.brake = recovery_brake(recovery.kind);
    input.fire_gun = false;
}

fn apply_teacher_boundary_recovery(
    state: &AircraftState,
    input: &mut ControlInput,
    recovery: BoundaryRecoveryState,
    collision_seconds: f32,
) {
    let mut recovery_input = *input;
    apply_boundary_recovery(state, &mut recovery_input, recovery);
    let collision_urgency = smoothstep01(
        1.0 - collision_seconds / BOUNDARY_PREDICTION_HORIZON_SECONDS.max(f32::EPSILON),
    );
    let urgency = lerp(TEACHER_BOUNDARY_RECOVERY_MIN_BLEND, 1.0, collision_urgency);
    input.throttle_delta = lerp(input.throttle_delta, recovery_input.throttle_delta, urgency);
    input.pitch = lerp(input.pitch, recovery_input.pitch, urgency);
    input.yaw = lerp(input.yaw, recovery_input.yaw, urgency);
    input.roll = lerp(input.roll, recovery_input.roll, urgency);
    if urgency >= 0.5 {
        input.brake = recovery_input.brake;
    }
    input.fire_gun = false;
}

fn pilot_profile(
    role: AircraftRole,
    authority: ControlAuthority,
    local_pilot_mode: Option<LocalPilotMode>,
    built_in_profile_overrides: BuiltInAiProfileOverrides,
) -> AiDriveProfile {
    match authority {
        ControlAuthority::BuiltInAi => match built_in_profile_overrides.get(role) {
            Some(BuiltInAiProfile::PreciseFollow) => AiDriveProfile::PreciseFollow,
            Some(BuiltInAiProfile::Teacher) => AiDriveProfile::Teacher,
            Some(BuiltInAiProfile::PassiveBounce) => AiDriveProfile::PassiveBounce,
            Some(BuiltInAiProfile::Imperfect) | None => AiDriveProfile::ImperfectBuiltIn,
        },
        ControlAuthority::ExternalAgent => match local_pilot_mode {
            Some(LocalPilotMode::ImperfectFollowAi) => AiDriveProfile::ImperfectBuiltIn,
            Some(LocalPilotMode::TeacherFollowAi) => AiDriveProfile::Teacher,
            _ => AiDriveProfile::PreciseFollow,
        },
        _ => AiDriveProfile::PreciseFollow,
    }
}

fn should_drive_aircraft(
    server_mode: bool,
    client_remote_authority_active: bool,
    role: AircraftRole,
    authority: ControlAuthority,
    follow_ai_controlled_role: Option<AircraftRole>,
) -> bool {
    if server_mode {
        return matches!(authority, ControlAuthority::BuiltInAi);
    }
    if client_remote_authority_active {
        return matches!(authority, ControlAuthority::ExternalAgent)
            && Some(role) == follow_ai_controlled_role;
    }
    matches!(authority, ControlAuthority::BuiltInAi)
        || (matches!(authority, ControlAuthority::ExternalAgent)
            && Some(role) == follow_ai_controlled_role)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smooth_gate(value: f32, threshold: f32) -> f32 {
    ((value - threshold) / (1.0 - threshold).max(1e-6)).clamp(0.0, 1.0)
}

fn smoothstep01(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn should_fire_gun(
    target_forward_local_z: f32,
    distance: f32,
    max_range: f32,
    gun_state: &GunState,
) -> bool {
    target_forward_local_z.is_finite()
        && distance.is_finite()
        && max_range.is_finite()
        && gun_state.heat.is_finite()
        && target_forward_local_z >= AI_FIRE_ALIGNMENT_COS
        && distance <= max_range.max(0.0) * AI_FIRE_MAX_RANGE_FRACTION
        && !gun_state.overheated
        && gun_state.heat < AI_FIRE_HEAT_LIMIT
}

fn tactical_brake(
    state: &AircraftState,
    performance: &AircraftPerformance,
    distance: f32,
    throttle_delta: f32,
    requested_by_follow_control: bool,
) -> bool {
    let minimum_speed = performance.stall_recovery_speed * TACTICAL_BRAKE_MIN_RECOVERY_SPEED_SCALE;
    let safe_to_brake = state.stall_factor <= TACTICAL_BRAKE_MAX_STALL_FACTOR
        && state.velocity.length() >= minimum_speed;
    safe_to_brake
        && (requested_by_follow_control
            || throttle_delta <= TACTICAL_BRAKE_THROTTLE_DELTA_THRESHOLD
            || distance < TACTICAL_BRAKE_DISTANCE_METERS)
}

fn has_severe_repairable_damage(damage: &AircraftDamageState) -> bool {
    damage
        .all_subsystems()
        .iter()
        .any(|(_, subsystem)| subsystem.fraction() <= REPAIR_DAMAGE_FRACTION_THRESHOLD)
}

fn predicted_closest_aircraft_separation(
    state: &AircraftState,
    target_state: &AircraftState,
    horizon_seconds: f32,
) -> f32 {
    let relative_position = target_state.position - state.position;
    let relative_velocity = target_state.velocity - state.velocity;
    let relative_speed_squared = relative_velocity.length_squared();
    let closest_time = if relative_speed_squared <= 1e-6 {
        0.0
    } else {
        (-relative_position.dot(relative_velocity) / relative_speed_squared)
            .clamp(0.0, horizon_seconds.max(0.0))
    };
    (relative_position + relative_velocity * closest_time).length()
}

fn repair_threat_score(
    state: &AircraftState,
    target_state: &AircraftState,
    predicted_closest_distance: f32,
) -> f32 {
    let target_to_self = (state.position - target_state.position).normalize_or_zero();
    let target_alignment = target_state.forward.dot(target_to_self);
    let aim_score = smooth_gate(target_alignment, REPAIR_THREAT_ALIGNMENT_COS);
    let proximity_score =
        (-predicted_closest_distance.max(0.0) / REPAIR_THREAT_DISTANCE_REFERENCE_METERS).exp();
    (0.75 * aim_score + 0.25 * proximity_score).clamp(0.0, 1.0)
}

fn repair_boundary_safe(
    config: &RepositoryConfig,
    state: &AircraftState,
    damage: &AircraftDamageState,
    performance: &AircraftPerformance,
    horizon_seconds: f32,
) -> bool {
    let mut predicted = state.clone();
    let mut repair_damage = damage.clone();
    repair_damage.is_repairing = true;
    let repair_input = ControlInput {
        repair: true,
        ..Default::default()
    };
    let effective = effective_aircraft_performance(performance, &repair_damage, &repair_input);
    let modifiers = damage_flight_modifiers(&repair_damage, &effective);
    let mut elapsed = 0.0;
    while elapsed < horizon_seconds {
        let dt = REPAIR_PREDICTION_STEP_SECONDS.min(horizon_seconds - elapsed);
        step_aircraft(&mut predicted, &effective, &modifiers, &repair_input, dt);
        elapsed += dt;

        if predicted.position.y - BOUNDARY_COLLISION_RADIUS_METERS <= config.scene.ground_height
            || predicted.position.y + BOUNDARY_COLLISION_RADIUS_METERS
                >= config.scene.flight_ceiling_height
            || Vec2::new(predicted.position.x, predicted.position.z).length()
                + BOUNDARY_COLLISION_RADIUS_METERS
                >= config.scene.arena_radius
        {
            return false;
        }
    }
    true
}

fn should_repair(
    config: &RepositoryConfig,
    state: &AircraftState,
    damage: &AircraftDamageState,
    performance: &AircraftPerformance,
    target_state: &AircraftState,
) -> bool {
    if !has_severe_repairable_damage(damage) || state.stall_factor > REPAIR_MAX_STALL_FACTOR {
        return false;
    }

    let repair_duration = config.game.repair_duration_seconds.max(0.1);
    let remaining_seconds = if damage.is_repairing {
        (repair_duration - damage.repair_elapsed_seconds).max(0.0)
    } else {
        repair_duration
    };
    let safety_horizon = remaining_seconds + REPAIR_COMPLETION_MARGIN_SECONDS;
    let closest_distance =
        predicted_closest_aircraft_separation(state, target_state, safety_horizon);
    if closest_distance < REPAIR_MIN_AIRCRAFT_SEPARATION_METERS
        || repair_threat_score(state, target_state, closest_distance) > REPAIR_MAX_THREAT_SCORE
    {
        return false;
    }

    repair_boundary_safe(config, state, damage, performance, safety_horizon)
}

fn apply_repair_command(input: &mut ControlInput) {
    *input = ControlInput {
        repair: true,
        ..Default::default()
    };
}

fn tail_chase_quality(
    target_forward_local_z: f32,
    heading_alignment: f32,
    target_tail_exposure: f32,
) -> f32 {
    let front_score = smooth_gate(target_forward_local_z, 0.7);
    let heading_score = smooth_gate(heading_alignment, 0.35);
    let tail_score = smooth_gate(target_tail_exposure, 0.2);
    front_score * heading_score * tail_score
}

fn compute_follow_throttle(
    distance: f32,
    max_range: f32,
    target_forward_local_z: f32,
    tail_chase_quality: f32,
    defensive_tail_quality: f32,
    focus_fire_active: bool,
) -> (f32, bool) {
    let range_ratio = (distance / max_range.max(1.0)).clamp(0.0, 1.25);
    let range_bias = smoothstep01(range_ratio.sqrt());
    let mut desired = if target_forward_local_z < -0.1 {
        1.0
    } else {
        lerp(
            GENERAL_FOLLOW_MIN_THROTTLE,
            GENERAL_FOLLOW_MAX_THROTTLE,
            range_bias,
        )
    };
    let tactical_close =
        smoothstep01(1.0 - (distance / (max_range * 0.42).max(1.0)).clamp(0.0, 1.0));
    let tactical_tail_capture = tail_chase_quality * tactical_close;
    if tactical_tail_capture > 0.0 {
        desired = lerp(desired, 0.60, 0.45 * tactical_tail_capture);
        if focus_fire_active {
            desired = lerp(desired, 0.40, 0.35 * tactical_tail_capture);
        }
    }
    let close_control =
        smoothstep01(1.0 - (distance / CLOSE_SPEED_CONTROL_DISTANCE_METERS).clamp(0.0, 1.0));
    let offensive_control = tail_chase_quality * close_control;
    let defensive_control = defensive_tail_quality * close_control;
    let speed_control = offensive_control
        .sqrt()
        .max((defensive_control * 1.4).clamp(0.0, 1.0).sqrt());
    if speed_control > 0.0 {
        desired = lerp(desired, TAIL_SPEED_MIN_THROTTLE, speed_control);
        if focus_fire_active {
            desired = lerp(
                desired,
                TAIL_SPEED_MIN_THROTTLE,
                0.55 * offensive_control.sqrt(),
            );
        }
    }
    let brake = distance < CLOSE_SPEED_CONTROL_DISTANCE_METERS
        && (offensive_control > 0.65 || defensive_control > 0.30)
        && target_forward_local_z > 0.25;
    (desired.clamp(0.0, 1.0), brake)
}

fn compute_tracking_controls(
    perceived_target_local: Vec3,
    focus_fire_active: bool,
) -> (f32, f32, f32) {
    let pitch_gain = if focus_fire_active { 4.2 } else { 3.0 };
    let yaw_gain = if focus_fire_active { 0.55 } else { 0.32 };
    let behind_pressure = (-perceived_target_local.z).clamp(0.0, 1.0);
    let roll_gain = lerp(1.15, 1.6, behind_pressure);

    let pitch = (-perceived_target_local.y * pitch_gain).clamp(-1.0, 1.0);
    let yaw = (perceived_target_local.x * yaw_gain).clamp(-1.0, 1.0);
    let roll = (-perceived_target_local.x * roll_gain - yaw * 0.08).clamp(-1.0, 1.0);
    (pitch, yaw, roll)
}

fn maybe_trigger_focus_fire(
    rng: &mut DeterministicRng,
    dt: f32,
    forward_alignment: f32,
    target_forward_local_z: f32,
    distance: f32,
    max_range: f32,
) -> bool {
    if forward_alignment < FOCUS_FIRE_TRIGGER_ALIGNMENT_COS
        || target_forward_local_z < FOCUS_FIRE_TRIGGER_SIDE_COS
        || distance > max_range * 0.95
    {
        return false;
    }
    rng.0.random::<f32>() < FOCUS_FIRE_TRIGGER_RATE_PER_SECOND * dt
}

fn random_control_multiplier(rng: &mut DeterministicRng) -> f32 {
    let magnitude = rng.0.random_range(0.0..2.0);
    let sign = if rng.0.random::<f32>() < 0.22 {
        -1.0
    } else {
        1.0
    };
    sign * magnitude
}

fn sample_imperfection_state(
    rng: &mut DeterministicRng,
    distance: f32,
) -> FollowAiImperfectionState {
    let mut state = FollowAiImperfectionState {
        hold_remaining_seconds: rng
            .0
            .random_range(IMPERFECTION_MIN_SECONDS..IMPERFECTION_MAX_SECONDS),
        target_bias_local: Vec3::ZERO,
        distance_scale: 1.0,
        pitch_multiplier: 1.0,
        yaw_multiplier: 1.0,
        roll_multiplier: 1.0,
        throttle_multiplier: 1.0,
        fire_inhibited: false,
        focus_fire_inhibited: false,
    };
    let mut symptom_ids = [0_u8, 1, 2, 3];
    symptom_ids.shuffle(&mut rng.0);
    let symptom_count = rng.0.random_range(1..=3);
    for symptom in symptom_ids.into_iter().take(symptom_count) {
        match symptom {
            0 => {
                let bias_scale = distance.clamp(35.0, 180.0) * 0.35;
                state.target_bias_local = Vec3::new(
                    rng.0.random_range(-1.0..1.0) * bias_scale,
                    rng.0.random_range(-0.75..0.75) * bias_scale,
                    rng.0.random_range(-0.5..0.8) * bias_scale,
                );
            }
            1 => {
                state.distance_scale = rng.0.random_range(0.55..1.7);
            }
            2 => {
                state.pitch_multiplier = random_control_multiplier(rng);
                state.yaw_multiplier = random_control_multiplier(rng);
                state.roll_multiplier = random_control_multiplier(rng);
                state.throttle_multiplier = random_control_multiplier(rng);
            }
            3 => {
                state.fire_inhibited = rng.0.random::<f32>() < 0.65;
                state.focus_fire_inhibited = rng.0.random::<f32>() < 0.75;
            }
            _ => {}
        }
    }
    state
}

fn apply_imperfection_perception(
    target_offset_local: Vec3,
    imperfection: Option<FollowAiImperfectionState>,
) -> (Vec3, f32) {
    let mut perceived = target_offset_local;
    if let Some(active) = imperfection {
        perceived += active.target_bias_local;
        let magnitude = perceived.length().max(1.0) * active.distance_scale.max(0.1);
        perceived = perceived.normalize_or_zero() * magnitude;
    }
    let distance = perceived.length().max(1.0);
    (perceived, distance)
}

fn apply_imperfection_controls(
    input: &mut ControlInput,
    imperfection: Option<FollowAiImperfectionState>,
) {
    let Some(active) = imperfection else {
        return;
    };
    input.pitch = (input.pitch * active.pitch_multiplier).clamp(-1.0, 1.0);
    input.yaw = (input.yaw * active.yaw_multiplier).clamp(-1.0, 1.0);
    input.roll = (input.roll * active.roll_multiplier).clamp(-1.0, 1.0);
    input.throttle_delta = (input.throttle_delta * active.throttle_multiplier).clamp(-1.0, 1.0);
    if active.fire_inhibited {
        input.fire_gun = false;
    }
}

fn compute_teacher_control_input(
    config: &RepositoryConfig,
    role: AircraftRole,
    state: &AircraftState,
    damage: &AircraftDamageState,
    performance: &AircraftPerformance,
    gun_state: &GunState,
    target_state: &AircraftState,
) -> ControlInput {
    let target_offset = target_state.position - state.position;
    let distance = target_offset.length();
    let to_target_world = target_offset.normalize_or_zero();
    let target_offset_local = state.orientation.inverse() * target_offset;
    let target_local = target_offset_local.normalize_or_zero();
    let heading_alignment = state.forward.dot(target_state.forward);
    let target_tail_exposure = target_state.forward.dot(to_target_world);
    let self_tail_exposure = state.forward.dot(-to_target_world);
    let tail_quality = tail_chase_quality(target_local.z, heading_alignment, target_tail_exposure);
    let defensive_tail_quality =
        tail_chase_quality(-target_local.z, heading_alignment, self_tail_exposure);
    let (pitch, yaw, roll) = compute_tracking_controls(target_local, false);
    let gun = match role {
        AircraftRole::Fighter1 => &config.fighter1_aircraft.gun,
        AircraftRole::Fighter2 => &config.fighter2_aircraft.gun,
    };
    let (desired_throttle, brake) = compute_follow_throttle(
        distance,
        gun.max_range,
        target_local.z,
        tail_quality,
        defensive_tail_quality,
        false,
    );
    let throttle_delta = ((desired_throttle - state.throttle) / 0.18).clamp(-1.0, 1.0);
    let mut input = ControlInput {
        throttle_delta,
        brake: tactical_brake(state, performance, distance, throttle_delta, brake),
        pitch,
        yaw,
        roll,
        fire_gun: should_fire_gun(target_local.z, distance, gun.max_range, gun_state),
        ..Default::default()
    };
    let recovering = if let Some((predicted_recovery, collision_seconds)) =
        predicted_boundary_hit(config, state, performance, damage, &input)
    {
        apply_teacher_boundary_recovery(state, &mut input, predicted_recovery, collision_seconds);
        true
    } else {
        false
    };
    if !recovering && should_repair(config, state, damage, performance, target_state) {
        apply_repair_command(&mut input);
    }
    input
}

fn compute_passive_bounce_control_input(
    config: &RepositoryConfig,
    state: &AircraftState,
    damage: &AircraftDamageState,
    performance: &AircraftPerformance,
    previous_recovery: Option<BoundaryRecoveryState>,
    dt_seconds: f32,
) -> (ControlInput, Option<BoundaryRecoveryState>) {
    let mut input = ControlInput::default();
    let mut recovery = previous_recovery;
    if let Some(predicted_recovery) =
        predicted_boundary_hit_kind(config, state, performance, damage, &input)
    {
        recovery = Some(predicted_recovery);
    }
    let Some(mut active) = recovery else {
        return (input, None);
    };

    apply_boundary_recovery(state, &mut input, active);
    input.fire_gun = false;
    input.repair = false;
    active.hold_remaining_seconds = (active.hold_remaining_seconds - dt_seconds).max(0.0);
    let next_recovery = (active.hold_remaining_seconds > 0.0).then_some(active);
    (input, next_recovery)
}

pub fn query_teacher_action(world: &mut World, role: AircraftRole) -> Option<EnvironmentAction> {
    let config = world.get_resource::<RepositoryConfig>()?.clone();
    let mut query = world.query::<(
        &AircraftRole,
        &AircraftState,
        &AircraftDamageState,
        &AircraftPerformance,
        &GunState,
    )>();
    let mut controlled = None;
    let mut target = None;
    for (aircraft_role, state, damage, performance, gun_state) in query.iter(world) {
        if *aircraft_role == role {
            controlled = Some((
                state.clone(),
                damage.clone(),
                performance.clone(),
                gun_state.clone(),
            ));
        } else {
            target = Some(state.clone());
        }
    }
    let (state, damage, performance, gun_state) = controlled?;
    let target_state = target?;
    if state.is_destroyed || target_state.is_destroyed {
        return None;
    }
    Some(EnvironmentAction::from(compute_teacher_control_input(
        &config,
        role,
        &state,
        &damage,
        &performance,
        &gun_state,
        &target_state,
    )))
}

fn update_basic_fighter_controls(
    config: Res<RepositoryConfig>,
    mut deterministic_rng: ResMut<DeterministicRng>,
    time: Res<Time<Fixed>>,
    bridge_mode: Option<Res<BridgeMode>>,
    bridge_link: Option<Res<BridgeLinkState>>,
    assigned_role: Option<Res<AssignedControlRole>>,
    local_pilot_mode: Option<Res<LocalPilotMode>>,
    built_in_profile_overrides: Res<BuiltInAiProfileOverrides>,
    match_phase: Res<State<MatchPhase>>,
    mut boundary_recovery_memory: Local<BoundaryRecoveryMemory>,
    mut focus_fire_memory: Local<FocusFireMemory>,
    mut imperfection_memory: Local<ImperfectionMemory>,
    mut controlled_query: Query<(
        &AircraftRole,
        &ControlAuthority,
        &AircraftState,
        &AircraftDamageState,
        &AircraftPerformance,
        &GunState,
        &mut ControlInput,
    )>,
    aircraft_query: Query<(&AircraftRole, &AircraftState)>,
) {
    let dt = time.delta_secs();
    let fighter1_state = aircraft_query.iter().find_map(|(role, state)| {
        (*role == AircraftRole::Fighter1 && !state.is_destroyed).then_some(state.clone())
    });
    let fighter2_state = aircraft_query.iter().find_map(|(role, state)| {
        (*role == AircraftRole::Fighter2 && !state.is_destroyed).then_some(state.clone())
    });
    let server_mode = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Server)
    );
    let client_remote_authority_active = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Client)
    ) && bridge_link
        .as_deref()
        .map(|link| {
            link.remote_authority_active
                || link.client_status == BridgeClientConnectionStatus::Connected
        })
        .unwrap_or(false);
    let locally_controlled_role = assigned_role
        .as_deref()
        .and_then(|role| bridge_slot_aircraft_role(role.0))
        .or(Some(AircraftRole::Fighter1));
    let follow_ai_controlled_role = matches!(
        local_pilot_mode.as_deref().copied(),
        Some(
            LocalPilotMode::FollowAi
                | LocalPilotMode::ImperfectFollowAi
                | LocalPilotMode::TeacherFollowAi
        )
    )
    .then_some(locally_controlled_role)
    .flatten();

    for (role, authority, state, damage, performance, gun_state, mut input) in &mut controlled_query
    {
        let should_drive = should_drive_aircraft(
            server_mode,
            client_remote_authority_active,
            *role,
            *authority,
            follow_ai_controlled_role,
        );
        if !should_drive {
            continue;
        }

        if *match_phase.get() != MatchPhase::Running || state.is_destroyed {
            (*boundary_recovery_memory).set(*role, None);
            (*focus_fire_memory).set(*role, None);
            (*imperfection_memory).set(*role, None);
            *input = ControlInput::default();
            continue;
        }

        let profile = pilot_profile(
            *role,
            *authority,
            local_pilot_mode.as_deref().copied(),
            *built_in_profile_overrides,
        );
        if matches!(profile, AiDriveProfile::PassiveBounce) {
            let (next_input, next_recovery) = compute_passive_bounce_control_input(
                &config,
                state,
                damage,
                performance,
                (*boundary_recovery_memory).get(*role),
                dt,
            );
            (*boundary_recovery_memory).set(*role, next_recovery);
            (*focus_fire_memory).set(*role, None);
            (*imperfection_memory).set(*role, None);
            *input = next_input;
            continue;
        }

        let target_state = match role {
            AircraftRole::Fighter1 => fighter2_state.as_ref(),
            AircraftRole::Fighter2 => fighter1_state.as_ref(),
        };

        let Some(player_state) = target_state else {
            (*focus_fire_memory).set(*role, None);
            (*imperfection_memory).set(*role, None);
            *input = ControlInput::default();
            continue;
        };
        if matches!(profile, AiDriveProfile::Teacher) {
            *input = compute_teacher_control_input(
                &config,
                *role,
                state,
                damage,
                performance,
                gun_state,
                player_state,
            );
            (*boundary_recovery_memory).set(*role, None);
            (*focus_fire_memory).set(*role, None);
            (*imperfection_memory).set(*role, None);
            continue;
        }

        let target_offset = player_state.position - state.position;
        let distance = target_offset.length();
        let to_target_world = target_offset.normalize_or_zero();
        let target_offset_local = state.orientation.inverse() * target_offset;

        let gun = match role {
            AircraftRole::Fighter1 => &config.fighter1_aircraft.gun,
            AircraftRole::Fighter2 => &config.fighter2_aircraft.gun,
        };

        let forward_alignment = state.forward.dot(to_target_world);
        let mut imperfection = (*imperfection_memory).get(*role);
        if let Some(mut active) = imperfection {
            active.hold_remaining_seconds = (active.hold_remaining_seconds - dt).max(0.0);
            imperfection = if active.hold_remaining_seconds > 0.0 {
                Some(active)
            } else {
                None
            };
        }
        if imperfection.is_none()
            && matches!(profile, AiDriveProfile::ImperfectBuiltIn)
            && deterministic_rng.0.random::<f32>() < IMPERFECTION_TRIGGER_RATE_PER_SECOND * dt
        {
            imperfection = Some(sample_imperfection_state(&mut deterministic_rng, distance));
        }
        (*imperfection_memory).set(*role, imperfection);

        let mut focus_fire = (*focus_fire_memory).get(*role);
        if let Some(mut active) = focus_fire {
            active.hold_remaining_seconds = (active.hold_remaining_seconds - dt).max(0.0);
            focus_fire = if active.hold_remaining_seconds > 0.0 {
                Some(active)
            } else {
                None
            };
        }
        let can_trigger_focus_fire = imperfection
            .map(|active| !active.focus_fire_inhibited)
            .unwrap_or(true);
        if focus_fire.is_none()
            && can_trigger_focus_fire
            && maybe_trigger_focus_fire(
                &mut deterministic_rng,
                dt,
                forward_alignment,
                target_offset_local.normalize_or_zero().z,
                distance,
                gun.max_range,
            )
        {
            focus_fire = Some(FocusFireState {
                hold_remaining_seconds: deterministic_rng
                    .0
                    .random_range(FOCUS_FIRE_MIN_SECONDS..FOCUS_FIRE_MAX_SECONDS),
            });
        }
        (*focus_fire_memory).set(*role, focus_fire);

        let (perceived_target_offset_local, perceived_distance) =
            apply_imperfection_perception(target_offset_local, imperfection);
        let perceived_target_local = perceived_target_offset_local.normalize_or_zero();
        let focus_fire_active = focus_fire.is_some();
        let heading_alignment = state.forward.dot(player_state.forward);
        let target_tail_exposure = player_state.forward.dot(to_target_world);
        let self_tail_exposure = state.forward.dot(-to_target_world);
        let tail_quality = tail_chase_quality(
            perceived_target_local.z,
            heading_alignment,
            target_tail_exposure,
        );
        let defensive_tail_quality = tail_chase_quality(
            -perceived_target_local.z,
            heading_alignment,
            self_tail_exposure,
        );
        let (pitch, yaw, roll) =
            compute_tracking_controls(perceived_target_local, focus_fire_active);

        let (desired_throttle, brake) = compute_follow_throttle(
            perceived_distance,
            gun.max_range,
            perceived_target_local.z,
            tail_quality,
            defensive_tail_quality,
            focus_fire_active,
        );
        let throttle_delta = ((desired_throttle - state.throttle) / 0.18).clamp(-1.0, 1.0);
        let mut next_input = ControlInput {
            throttle_delta,
            brake: tactical_brake(
                state,
                performance,
                perceived_distance,
                throttle_delta,
                brake,
            ),
            pitch,
            roll,
            yaw,
            fire_gun: should_fire_gun(
                perceived_target_local.z,
                perceived_distance,
                gun.max_range,
                gun_state,
            ),
            ..Default::default()
        };
        apply_imperfection_controls(&mut next_input, imperfection);

        let predicted_hit =
            predicted_boundary_hit_kind(&config, state, performance, damage, &next_input);
        let mut recovery = (*boundary_recovery_memory).get(*role);
        if let Some(predicted_recovery) = predicted_hit {
            recovery = Some(predicted_recovery);
            focus_fire = None;
            imperfection = None;
            (*focus_fire_memory).set(*role, None);
            (*imperfection_memory).set(*role, None);
        }
        let recovering = if let Some(mut active) = recovery {
            apply_boundary_recovery(state, &mut next_input, active);
            active.hold_remaining_seconds = (active.hold_remaining_seconds - dt).max(0.0);
            (*boundary_recovery_memory).set(
                *role,
                if active.hold_remaining_seconds > 0.0 {
                    Some(active)
                } else {
                    None
                },
            );
            true
        } else {
            (*boundary_recovery_memory).set(*role, None);
            false
        };
        if !recovering && should_repair(&config, state, damage, performance, player_state) {
            apply_repair_command(&mut next_input);
        }
        *input = next_input;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BOUNDARY_RECOVERY_TARGET_NORMAL_COS, BoundaryKind, BoundaryRecoveryState, BuiltInAiProfile,
        BuiltInAiProfileOverrides, FollowAiImperfectionState, apply_boundary_recovery,
        apply_imperfection_controls, apply_imperfection_perception, apply_repair_command,
        apply_teacher_boundary_recovery, compute_follow_throttle,
        compute_passive_bounce_control_input, compute_teacher_control_input,
        compute_tracking_controls, nearest_safe_velocity_direction, predicted_boundary_hit_kind,
        should_drive_aircraft, should_fire_gun, should_repair, tactical_brake, tail_chase_quality,
        update_basic_fighter_controls,
    };
    use crate::api::environment::DeterministicRng;
    use crate::core::config::AircraftConfig;
    use crate::core::config::RepositoryConfig;
    use crate::gameplay::damage::AircraftDamageState;
    use crate::gameplay::match_state::MatchPhase;
    use crate::input::actions::ControlInput;
    use crate::simulation::components::{AircraftRole, AircraftState, ControlAuthority, GunState};
    use crate::simulation::systems::aircraft_performance_from_config;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::{Fixed, State, Time, Vec3, World};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn built_in_ai_drives_either_role_in_authoritative_worlds() {
        for role in [AircraftRole::Fighter1, AircraftRole::Fighter2] {
            assert!(should_drive_aircraft(
                false,
                false,
                role,
                ControlAuthority::BuiltInAi,
                None,
            ));
            assert!(should_drive_aircraft(
                true,
                false,
                role,
                ControlAuthority::BuiltInAi,
                None,
            ));
            assert!(!should_drive_aircraft(
                false,
                true,
                role,
                ControlAuthority::BuiltInAi,
                None,
            ));
        }
    }

    #[test]
    fn passive_bounce_is_neutral_until_boundary_recovery_is_needed() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let safe_state = AircraftState {
            position: Vec3::new(0.0, 1_000.0, 0.0),
            velocity: Vec3::Z * 60.0,
            forward: Vec3::Z,
            throttle: 0.6,
            ..Default::default()
        };
        let damage = AircraftDamageState::new(100.0, &performance);
        let (safe_input, safe_recovery) = compute_passive_bounce_control_input(
            &config,
            &safe_state,
            &damage,
            &performance,
            None,
            1.0 / 60.0,
        );
        assert_eq!(safe_input.throttle_delta, 0.0);
        assert_eq!(safe_input.pitch, 0.0);
        assert_eq!(safe_input.roll, 0.0);
        assert_eq!(safe_input.yaw, 0.0);
        assert!(!safe_input.brake);
        assert!(!safe_input.fire_gun);
        assert!(!safe_input.repair);
        assert!(safe_recovery.is_none());

        let diving_state = AircraftState {
            position: Vec3::new(0.0, 80.0, 0.0),
            velocity: -Vec3::Y * 60.0,
            forward: -Vec3::Y,
            throttle: 0.6,
            ..Default::default()
        };
        let (recovery_input, recovery) = compute_passive_bounce_control_input(
            &config,
            &diving_state,
            &damage,
            &performance,
            None,
            1.0 / 60.0,
        );
        assert!(recovery.is_some());
        assert!(
            recovery_input.pitch.abs() > 0.0
                || recovery_input.roll.abs() > 0.0
                || recovery_input.yaw.abs() > 0.0
        );
        assert!(!recovery_input.fire_gun);
        assert!(!recovery_input.repair);
    }

    #[test]
    fn precise_follow_replaces_stale_repair_input_after_recovery() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let state = AircraftState {
            position: Vec3::new(0.0, 1_000.0, 0.0),
            velocity: Vec3::Z * 80.0,
            forward: Vec3::Z,
            throttle: 0.8,
            hit_points: 40.0,
            ..Default::default()
        };
        let damage = AircraftDamageState::new(100.0, &performance);
        let target = AircraftState {
            position: state.position + Vec3::Z * 40.0,
            velocity: Vec3::Z * 60.0,
            forward: Vec3::Z,
            ..Default::default()
        };

        let mut world = World::new();
        world.insert_resource(config);
        world.insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(7)));
        world.insert_resource(Time::<Fixed>::from_hz(60.0));
        world.insert_resource(State::new(MatchPhase::Running));
        world.insert_resource(BuiltInAiProfileOverrides {
            fighter1: Some(BuiltInAiProfile::PreciseFollow),
            fighter2: None,
        });
        let controlled = world
            .spawn((
                AircraftRole::Fighter1,
                ControlAuthority::BuiltInAi,
                state,
                damage,
                performance,
                GunState::default(),
                ControlInput {
                    repair: true,
                    ..Default::default()
                },
            ))
            .id();
        world.spawn((AircraftRole::Fighter2, target));

        world
            .run_system_once(update_basic_fighter_controls)
            .expect("rule AI system should run");

        let input = world
            .get::<ControlInput>(controlled)
            .expect("controlled aircraft should retain an input");
        assert!(!input.repair);
        assert!(input.fire_gun);
    }

    #[test]
    fn fire_gate_uses_wide_cone_and_heat_headroom() {
        let ready = GunState {
            heat: 0.79,
            ..Default::default()
        };
        assert!(should_fire_gun(
            29.0_f32.to_radians().cos(),
            900.0,
            1_000.0,
            &ready,
        ));
        assert!(!should_fire_gun(
            31.0_f32.to_radians().cos(),
            900.0,
            1_000.0,
            &ready,
        ));

        let at_limit = GunState {
            heat: 0.8,
            ..Default::default()
        };
        assert!(!should_fire_gun(1.0, 500.0, 1_000.0, &at_limit));
        let overheated = GunState {
            heat: 0.1,
            overheated: true,
            ..Default::default()
        };
        assert!(!should_fire_gun(1.0, 500.0, 1_000.0, &overheated));
    }

    #[test]
    fn tactical_brake_follows_slowdown_and_close_range_but_respects_low_speed() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let cruising = AircraftState {
            velocity: Vec3::Z * 80.0,
            stall_factor: 0.0,
            ..Default::default()
        };
        assert!(tactical_brake(&cruising, &performance, 500.0, -0.1, false,));
        assert!(tactical_brake(&cruising, &performance, 49.0, 0.0, false,));

        let stalled = AircraftState {
            velocity: Vec3::Z * 25.0,
            stall_factor: 0.8,
            ..Default::default()
        };
        assert!(!tactical_brake(&stalled, &performance, 20.0, -1.0, true,));
    }

    #[test]
    fn teacher_control_emits_fire_and_brake_from_shared_gates() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let state = AircraftState {
            position: Vec3::new(0.0, 1_000.0, 0.0),
            velocity: Vec3::Z * 80.0,
            forward: Vec3::Z,
            throttle: 1.0,
            ..Default::default()
        };
        let damage = AircraftDamageState::new(100.0, &performance);
        let gun_state = GunState {
            heat: 0.2,
            ..Default::default()
        };
        let target = AircraftState {
            position: Vec3::new(0.0, 1_000.0, 40.0),
            velocity: Vec3::Z * 60.0,
            forward: Vec3::Z,
            ..Default::default()
        };

        let input = compute_teacher_control_input(
            &config,
            AircraftRole::Fighter1,
            &state,
            &damage,
            &performance,
            &gun_state,
            &target,
        );
        assert!(input.fire_gun);
        assert!(input.brake);
        assert!(!input.repair);
    }

    fn repair_test_fixture() -> (
        RepositoryConfig,
        crate::simulation::components::AircraftPerformance,
        AircraftState,
        AircraftDamageState,
        AircraftState,
    ) {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let state = AircraftState {
            position: Vec3::new(0.0, 1_000.0, 0.0),
            velocity: Vec3::Z * 60.0,
            forward: Vec3::Z,
            throttle: 0.6,
            hit_points: 100.0,
            ..Default::default()
        };
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.left_wing.current = damage.left_wing.max * 0.5;
        let target = AircraftState {
            position: Vec3::new(1_000.0, 1_000.0, 0.0),
            velocity: Vec3::Z * 60.0,
            forward: Vec3::Z,
            ..Default::default()
        };
        (config, performance, state, damage, target)
    }

    #[test]
    fn repair_gate_accepts_severe_damage_in_safe_low_threat_state() {
        let (config, performance, state, damage, target) = repair_test_fixture();
        assert!(should_repair(
            &config,
            &state,
            &damage,
            &performance,
            &target,
        ));

        let mut input = ControlInput {
            throttle_delta: -1.0,
            brake: true,
            pitch: 0.5,
            roll: -0.5,
            yaw: 0.25,
            fire_gun: true,
            ..Default::default()
        };
        apply_repair_command(&mut input);
        assert!(input.repair);
        assert_eq!(input.throttle_delta, 0.0);
        assert!(!input.brake);
        assert_eq!(input.pitch, 0.0);
        assert_eq!(input.roll, 0.0);
        assert_eq!(input.yaw, 0.0);
        assert!(!input.fire_gun);
    }

    #[test]
    fn repair_gate_rejects_enemy_aim_and_predicted_close_pass() {
        let (config, performance, state, damage, mut target) = repair_test_fixture();
        target.position = Vec3::new(0.0, 1_000.0, -500.0);
        target.forward = Vec3::Z;
        assert!(!should_repair(
            &config,
            &state,
            &damage,
            &performance,
            &target,
        ));

        target.position = Vec3::new(120.0, 1_000.0, 0.0);
        target.velocity = Vec3::new(-30.0, 0.0, 60.0);
        target.forward = Vec3::Z;
        assert!(!should_repair(
            &config,
            &state,
            &damage,
            &performance,
            &target,
        ));
    }

    #[test]
    fn repair_gate_rejects_incomplete_damage_and_boundary_risk() {
        let (config, performance, mut state, mut damage, target) = repair_test_fixture();
        damage.left_wing.current = damage.left_wing.max * 0.51;
        state.hit_points = damage.total_max_hit_points * 0.4;
        assert!(!should_repair(
            &config,
            &state,
            &damage,
            &performance,
            &target,
        ));

        damage.left_wing.current = damage.left_wing.max * 0.5;
        state.position.y = config.scene.ground_height + 20.0;
        state.velocity = Vec3::new(0.0, -20.0, 60.0);
        assert!(!should_repair(
            &config,
            &state,
            &damage,
            &performance,
            &target,
        ));
    }

    #[test]
    fn teacher_resumes_fire_after_subsystems_are_repaired() {
        let (config, performance, mut state, damage, mut target) = repair_test_fixture();
        state.hit_points = damage.total_max_hit_points * 0.4;
        target.position = state.position + Vec3::Z * 40.0;
        target.velocity = state.velocity;
        target.forward = Vec3::Z;
        let gun_state = GunState::default();
        let mut repaired_damage = damage;
        for (_, subsystem) in repaired_damage.all_subsystems_mut() {
            subsystem.restore_full();
        }

        let input = compute_teacher_control_input(
            &config,
            AircraftRole::Fighter1,
            &state,
            &repaired_damage,
            &performance,
            &gun_state,
            &target,
        );

        assert!(!input.repair);
        assert!(input.fire_gun);
    }

    #[test]
    fn teacher_boundary_response_scales_with_collision_urgency() {
        let state = AircraftState {
            throttle: 0.5,
            velocity: Vec3::new(0.0, -20.0, 40.0),
            ..Default::default()
        };
        let recovery = BoundaryRecoveryState {
            kind: BoundaryKind::Ground,
            inward_normal_world: Vec3::Y,
            hold_remaining_seconds: 0.0,
        };
        let base = ControlInput {
            pitch: 0.4,
            roll: 0.2,
            fire_gun: true,
            ..Default::default()
        };
        let mut early = base;
        let mut imminent = base;

        apply_teacher_boundary_recovery(&state, &mut early, recovery, 2.9);
        apply_teacher_boundary_recovery(&state, &mut imminent, recovery, 0.1);

        assert!((imminent.pitch - base.pitch).abs() > (early.pitch - base.pitch).abs());
        assert!(!early.fire_gun);
        assert!(!imminent.fire_gun);
    }

    #[test]
    fn predicted_dive_hits_ground_within_horizon() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let state = AircraftState {
            position: Vec3::new(0.0, 40.0, 0.0),
            velocity: Vec3::new(0.0, -50.0, 20.0),
            forward: Vec3::new(0.0, -0.7, 0.7).normalize(),
            ..Default::default()
        };
        let hit = predicted_boundary_hit_kind(
            &config,
            &state,
            &performance,
            &damage,
            &ControlInput::default(),
        );
        assert!(matches!(
            hit,
            Some(BoundaryRecoveryState {
                kind: BoundaryKind::Ground,
                ..
            })
        ));
    }

    #[test]
    fn predicted_climb_hits_ceiling_within_horizon() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let state = AircraftState {
            position: Vec3::new(0.0, config.scene.flight_ceiling_height - 40.0, 0.0),
            velocity: Vec3::new(0.0, 50.0, 20.0),
            forward: Vec3::new(0.0, 0.7, 0.7).normalize(),
            ..Default::default()
        };
        let hit = predicted_boundary_hit_kind(
            &config,
            &state,
            &performance,
            &damage,
            &ControlInput::default(),
        );
        assert!(matches!(
            hit,
            Some(BoundaryRecoveryState {
                kind: BoundaryKind::Ceiling,
                ..
            })
        ));
    }

    #[test]
    fn safe_mid_altitude_has_no_predicted_boundary_hit() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let state = AircraftState {
            position: Vec3::new(0.0, config.scene.flight_ceiling_height * 0.5, 0.0),
            velocity: Vec3::new(0.0, 0.0, 60.0),
            forward: Vec3::Z,
            ..Default::default()
        };
        let hit = predicted_boundary_hit_kind(
            &config,
            &state,
            &performance,
            &damage,
            &ControlInput::default(),
        );
        assert_eq!(hit, None);
    }

    #[test]
    fn predicted_horizontal_escape_hits_boundary_within_horizon() {
        let config = RepositoryConfig::default();
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let state = AircraftState {
            position: Vec3::new(config.scene.arena_radius - 20.0, 400.0, 0.0),
            velocity: Vec3::new(80.0, 0.0, 0.0),
            forward: Vec3::X,
            ..Default::default()
        };
        let hit = predicted_boundary_hit_kind(
            &config,
            &state,
            &performance,
            &damage,
            &ControlInput::default(),
        );
        assert!(matches!(
            hit,
            Some(BoundaryRecoveryState {
                kind: BoundaryKind::Horizontal,
                ..
            })
        ));
    }

    #[test]
    fn ground_recovery_drives_nose_up_and_disables_fire() {
        let state = AircraftState {
            velocity: Vec3::new(0.0, -40.0, 20.0),
            ..Default::default()
        };
        let mut input = ControlInput {
            pitch: 0.4,
            roll: 0.9,
            yaw: 0.8,
            throttle_delta: 0.0,
            fire_gun: true,
            ..Default::default()
        };
        apply_boundary_recovery(
            &state,
            &mut input,
            BoundaryRecoveryState {
                kind: BoundaryKind::Ground,
                inward_normal_world: Vec3::Y,
                hold_remaining_seconds: 2.0,
            },
        );
        assert!(input.pitch < 0.0);
        assert!(!input.fire_gun);
    }

    #[test]
    fn nearest_safe_velocity_direction_reaches_required_ground_alignment() {
        let state = AircraftState {
            velocity: Vec3::new(0.0, -40.0, 20.0),
            forward: Vec3::new(0.0, -0.4, 0.9).normalize(),
            ..Default::default()
        };
        let target = nearest_safe_velocity_direction(&state, Vec3::Y);
        assert!(target.dot(Vec3::Y) >= BOUNDARY_RECOVERY_TARGET_NORMAL_COS - 1e-4);
        assert!((target.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ceiling_recovery_drives_nose_down_and_disables_fire() {
        let state = AircraftState {
            velocity: Vec3::new(0.0, 40.0, 20.0),
            ..Default::default()
        };
        let mut input = ControlInput {
            pitch: -0.4,
            roll: 0.9,
            yaw: 0.8,
            throttle_delta: 0.0,
            fire_gun: true,
            ..Default::default()
        };
        apply_boundary_recovery(
            &state,
            &mut input,
            BoundaryRecoveryState {
                kind: BoundaryKind::Ceiling,
                inward_normal_world: -Vec3::Y,
                hold_remaining_seconds: 2.0,
            },
        );
        assert!(input.pitch > 0.0);
        assert!(!input.fire_gun);
    }

    #[test]
    fn nearest_safe_velocity_direction_reaches_required_ceiling_alignment() {
        let state = AircraftState {
            velocity: Vec3::new(0.0, 40.0, 20.0),
            forward: Vec3::new(0.0, 0.4, 0.9).normalize(),
            ..Default::default()
        };
        let target = nearest_safe_velocity_direction(&state, -Vec3::Y);
        assert!(target.dot(-Vec3::Y) >= BOUNDARY_RECOVERY_TARGET_NORMAL_COS - 1e-4);
        assert!((target.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn follow_throttle_decreases_when_target_is_close() {
        let (far_throttle, far_brake) =
            compute_follow_throttle(900.0, 1200.0, 0.95, 0.0, 0.0, false);
        let (near_throttle, near_brake) =
            compute_follow_throttle(120.0, 1200.0, 0.95, 0.0, 0.0, false);
        assert!(far_throttle > near_throttle);
        assert!(far_throttle >= 0.99);
        assert!(near_throttle >= 0.8);
        assert!(!far_brake);
        assert!(!near_brake);
    }

    #[test]
    fn focus_fire_throttle_prefers_lower_closure() {
        let (normal_throttle, _) = compute_follow_throttle(220.0, 1200.0, 0.99, 1.0, 0.0, false);
        let (focus_throttle, _) = compute_follow_throttle(220.0, 1200.0, 0.99, 1.0, 0.0, true);
        assert!(focus_throttle < normal_throttle);
        assert!(focus_throttle >= 0.0);
    }

    #[test]
    fn tail_chase_quality_requires_tail_position() {
        let perfect = tail_chase_quality(0.98, 0.95, 0.95);
        let head_on = tail_chase_quality(0.98, -0.95, -0.95);
        let side = tail_chase_quality(0.45, 0.95, 0.95);
        assert!(perfect > 0.8);
        assert!(head_on < 0.01);
        assert!(side < perfect);
    }

    #[test]
    fn tail_chase_throttle_control_scales_continuously() {
        let (no_tail, _) = compute_follow_throttle(120.0, 1200.0, 0.98, 0.0, 0.0, false);
        let (partial_tail, _) = compute_follow_throttle(120.0, 1200.0, 0.98, 0.5, 0.0, false);
        let (perfect_tail, _) = compute_follow_throttle(120.0, 1200.0, 0.98, 1.0, 0.0, false);
        assert!(no_tail > partial_tail);
        assert!(partial_tail > perfect_tail);
        assert!(perfect_tail >= 0.0);
    }

    #[test]
    fn non_tail_close_follow_keeps_throttle_high() {
        let (throttle, brake) = compute_follow_throttle(160.0, 1200.0, 0.96, 0.15, 0.0, false);
        assert!(throttle > 0.8);
        assert!(!brake);
    }

    #[test]
    fn defensive_tail_throttle_control_allows_large_slowdown() {
        let (normal, _) = compute_follow_throttle(12.0, 1200.0, -0.7, 0.0, 0.0, false);
        let (defensive, brake) = compute_follow_throttle(12.0, 1200.0, 0.5, 0.0, 1.0, false);
        assert!(defensive < normal);
        assert!(defensive < 0.3);
        assert!(brake);
    }

    #[test]
    fn imperfection_can_shift_target_perception_and_invert_controls() {
        let (perceived_offset, perceived_distance) = apply_imperfection_perception(
            Vec3::new(0.0, 0.0, 100.0),
            Some(FollowAiImperfectionState {
                hold_remaining_seconds: 1.0,
                target_bias_local: Vec3::new(50.0, -10.0, 0.0),
                distance_scale: 1.5,
                pitch_multiplier: -1.0,
                yaw_multiplier: 0.5,
                roll_multiplier: -0.25,
                throttle_multiplier: -1.0,
                fire_inhibited: true,
                focus_fire_inhibited: false,
            }),
        );
        assert!(perceived_offset.x > 0.0);
        assert!(perceived_distance > 120.0);

        let mut input = ControlInput {
            throttle_delta: 0.5,
            pitch: 0.4,
            yaw: 0.6,
            roll: -0.8,
            fire_gun: true,
            ..Default::default()
        };
        apply_imperfection_controls(
            &mut input,
            Some(FollowAiImperfectionState {
                hold_remaining_seconds: 1.0,
                target_bias_local: Vec3::ZERO,
                distance_scale: 1.0,
                pitch_multiplier: -1.0,
                yaw_multiplier: 0.5,
                roll_multiplier: -0.25,
                throttle_multiplier: -1.0,
                fire_inhibited: true,
                focus_fire_inhibited: false,
            }),
        );
        assert!(input.pitch < 0.0);
        assert!(input.yaw > 0.0 && input.yaw < 0.6);
        assert!(input.roll > 0.0);
        assert!(input.throttle_delta < 0.0);
        assert!(!input.fire_gun);
    }

    #[test]
    fn tracking_controls_bias_pitch_and_yaw_over_roll() {
        let (pitch, yaw, roll) =
            compute_tracking_controls(Vec3::new(0.35, -0.25, 0.9).normalize(), false);
        assert!(pitch > 0.0);
        assert!(yaw > 0.0);
        assert!(roll < 0.0);
        assert!(roll.abs() > yaw.abs());
    }
}
