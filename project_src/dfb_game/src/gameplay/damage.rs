use bevy::prelude::*;

use crate::api::events::{PendingEnvironmentEvents, push_event, role_subject};
use crate::api::types::EnvironmentEventKind;
use crate::core::config::RepositoryConfig;
use crate::core::config::SpawnDamageConfig;
use crate::input::actions::ControlInput;
use crate::simulation::components::{AircraftPerformance, AircraftState};
use crate::simulation::resources::SimulationDebugState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageStage {
    Undamaged,
    Damaged,
    Destroyed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AircraftSubsystem {
    LeftWing,
    RightWing,
    PitchTail,
    YawTail,
    Engine,
}

#[derive(Debug, Clone, Copy)]
pub struct SubsystemHitbox {
    pub name: &'static str,
    pub subsystem: AircraftSubsystem,
    pub center: Vec3,
    pub half_extents: Vec3,
}

#[derive(Debug, Clone)]
pub struct SubsystemHealth {
    pub current: f32,
    pub max: f32,
}

impl SubsystemHealth {
    pub fn new(max: f32) -> Self {
        Self { current: max, max }
    }

    pub fn apply_damage(&mut self, amount: f32) {
        self.current = (self.current - amount).max(0.0);
    }

    pub fn restore_full(&mut self) {
        self.current = self.max;
    }

    pub fn fraction(&self) -> f32 {
        if self.max <= f32::EPSILON {
            0.0
        } else {
            (self.current / self.max).clamp(0.0, 1.0)
        }
    }

    pub fn stage(&self) -> DamageStage {
        if self.current <= f32::EPSILON {
            DamageStage::Destroyed
        } else if self.fraction() < 0.65 {
            DamageStage::Damaged
        } else {
            DamageStage::Undamaged
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct AircraftDamageState {
    pub total_max_hit_points: f32,
    pub left_wing: SubsystemHealth,
    pub right_wing: SubsystemHealth,
    pub pitch_tail: SubsystemHealth,
    pub yaw_tail: SubsystemHealth,
    pub engine: SubsystemHealth,
    pub repair_elapsed_seconds: f32,
    pub is_repairing: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DamageFlightModifiers {
    pub left_wing_lift_scale: f32,
    pub right_wing_lift_scale: f32,
    pub roll_trim_rate_deg: f32,
    pub yaw_trim_rate_deg: f32,
    pub extra_drag: f32,
}

const DAMAGE_STAGE_THRESHOLD: f32 = 0.5;
const DAMAGE_TRIM_SOFTEN_EXPONENT: f32 = 1.35;
const DAMAGE_GRACE_EXPONENT: f32 = 3.0;
const DAMAGE_FAILURE_EXPONENT: f32 = 1.2;
const SPAWN_DAMAGE_TOTAL_FRACTION_FLOOR: f32 = 0.2;

impl AircraftDamageState {
    pub fn new(total_max_hit_points: f32, performance: &AircraftPerformance) -> Self {
        Self {
            total_max_hit_points,
            left_wing: SubsystemHealth::new(performance.left_wing_hit_points),
            right_wing: SubsystemHealth::new(performance.right_wing_hit_points),
            pitch_tail: SubsystemHealth::new(performance.pitch_tail_hit_points),
            yaw_tail: SubsystemHealth::new(performance.yaw_tail_hit_points),
            engine: SubsystemHealth::new(performance.engine_hit_points),
            repair_elapsed_seconds: 0.0,
            is_repairing: false,
        }
    }

    pub fn subsystem(&self, subsystem: AircraftSubsystem) -> &SubsystemHealth {
        match subsystem {
            AircraftSubsystem::LeftWing => &self.left_wing,
            AircraftSubsystem::RightWing => &self.right_wing,
            AircraftSubsystem::PitchTail => &self.pitch_tail,
            AircraftSubsystem::YawTail => &self.yaw_tail,
            AircraftSubsystem::Engine => &self.engine,
        }
    }

    pub fn subsystem_mut(&mut self, subsystem: AircraftSubsystem) -> &mut SubsystemHealth {
        match subsystem {
            AircraftSubsystem::LeftWing => &mut self.left_wing,
            AircraftSubsystem::RightWing => &mut self.right_wing,
            AircraftSubsystem::PitchTail => &mut self.pitch_tail,
            AircraftSubsystem::YawTail => &mut self.yaw_tail,
            AircraftSubsystem::Engine => &mut self.engine,
        }
    }

    pub fn apply_subsystem_damage(&mut self, subsystem: AircraftSubsystem, amount: f32) {
        self.subsystem_mut(subsystem).apply_damage(amount);
    }

    pub fn has_repairable_damage(&self, aircraft: &AircraftState) -> bool {
        aircraft.hit_points + 0.5 < self.total_max_hit_points
            || self
                .all_subsystems()
                .iter()
                .any(|(_, subsystem)| subsystem.stage() != DamageStage::Undamaged)
    }

    pub fn complete_repair_cycle(&mut self, aircraft: &mut AircraftState, restore_hit_points: f32) {
        aircraft.hit_points =
            (aircraft.hit_points + restore_hit_points).min(self.total_max_hit_points);
        for (_, subsystem) in self.all_subsystems_mut() {
            subsystem.restore_full();
        }
        self.interrupt_repair();
    }

    pub fn interrupt_repair(&mut self) {
        self.is_repairing = false;
        self.repair_elapsed_seconds = 0.0;
    }

    pub fn reset(&mut self, aircraft: &mut AircraftState) {
        aircraft.hit_points = self.total_max_hit_points;
        self.interrupt_repair();
        for (_, subsystem) in self.all_subsystems_mut() {
            subsystem.restore_full();
        }
    }

    pub fn update_repair_progress(
        &mut self,
        wants_repair: bool,
        dt: f32,
        repair_duration: f32,
    ) -> bool {
        if !wants_repair {
            self.interrupt_repair();
            return false;
        }

        self.is_repairing = true;
        self.repair_elapsed_seconds += dt;
        if self.repair_elapsed_seconds >= repair_duration {
            self.repair_elapsed_seconds = 0.0;
            true
        } else {
            false
        }
    }

    pub fn all_subsystems(&self) -> [(AircraftSubsystem, &SubsystemHealth); 5] {
        [
            (AircraftSubsystem::LeftWing, &self.left_wing),
            (AircraftSubsystem::RightWing, &self.right_wing),
            (AircraftSubsystem::PitchTail, &self.pitch_tail),
            (AircraftSubsystem::YawTail, &self.yaw_tail),
            (AircraftSubsystem::Engine, &self.engine),
        ]
    }

    pub fn all_subsystems_mut(&mut self) -> [(AircraftSubsystem, &mut SubsystemHealth); 5] {
        [
            (AircraftSubsystem::LeftWing, &mut self.left_wing),
            (AircraftSubsystem::RightWing, &mut self.right_wing),
            (AircraftSubsystem::PitchTail, &mut self.pitch_tail),
            (AircraftSubsystem::YawTail, &mut self.yaw_tail),
            (AircraftSubsystem::Engine, &mut self.engine),
        ]
    }
}

pub fn apply_spawn_damage_overrides(
    aircraft: &mut AircraftState,
    damage: &mut AircraftDamageState,
    performance: &AircraftPerformance,
    spawn_damage: Option<&SpawnDamageConfig>,
) {
    let Some(spawn_damage) = spawn_damage else {
        return;
    };

    let mut any_subsystem_override = false;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    let apply_fraction =
        |fraction: Option<f32>, subsystem: &mut SubsystemHealth, any: &mut bool| -> f32 {
            let resolved = fraction.unwrap_or(1.0).clamp(0.0, 1.0);
            if fraction.is_some() {
                *any = true;
            }
            subsystem.current = subsystem.max * resolved;
            resolved
        };

    let left_fraction = apply_fraction(
        spawn_damage.left_wing_fraction,
        &mut damage.left_wing,
        &mut any_subsystem_override,
    );
    weighted_sum += left_fraction * performance.left_wing_hit_points;
    total_weight += performance.left_wing_hit_points;

    let right_fraction = apply_fraction(
        spawn_damage.right_wing_fraction,
        &mut damage.right_wing,
        &mut any_subsystem_override,
    );
    weighted_sum += right_fraction * performance.right_wing_hit_points;
    total_weight += performance.right_wing_hit_points;

    let pitch_tail_fraction = apply_fraction(
        spawn_damage.pitch_tail_fraction,
        &mut damage.pitch_tail,
        &mut any_subsystem_override,
    );
    weighted_sum += pitch_tail_fraction * performance.pitch_tail_hit_points;
    total_weight += performance.pitch_tail_hit_points;

    let yaw_tail_fraction = apply_fraction(
        spawn_damage.yaw_tail_fraction,
        &mut damage.yaw_tail,
        &mut any_subsystem_override,
    );
    weighted_sum += yaw_tail_fraction * performance.yaw_tail_hit_points;
    total_weight += performance.yaw_tail_hit_points;

    let engine_fraction = apply_fraction(
        spawn_damage.engine_fraction,
        &mut damage.engine,
        &mut any_subsystem_override,
    );
    weighted_sum += engine_fraction * performance.engine_hit_points;
    total_weight += performance.engine_hit_points;

    let total_hit_points_fraction =
        if let Some(explicit_fraction) = spawn_damage.total_hit_points_fraction {
            explicit_fraction.clamp(SPAWN_DAMAGE_TOTAL_FRACTION_FLOOR, 1.0)
        } else if any_subsystem_override && total_weight > f32::EPSILON {
            (weighted_sum / total_weight).clamp(SPAWN_DAMAGE_TOTAL_FRACTION_FLOOR, 1.0)
        } else {
            1.0
        };

    aircraft.hit_points = damage.total_max_hit_points * total_hit_points_fraction;
    aircraft.is_destroyed = aircraft.hit_points <= f32::EPSILON;
}

pub fn update_repair_state(
    time: Res<Time<Fixed>>,
    debug: Res<SimulationDebugState>,
    config: Res<RepositoryConfig>,
    mut events: Option<ResMut<PendingEnvironmentEvents>>,
    mut query: Query<(
        &crate::simulation::components::AircraftRole,
        &mut AircraftState,
        &mut AircraftDamageState,
        &mut ControlInput,
    )>,
) {
    let dt = time.delta_secs();
    let repair_duration = config.game.repair_duration_seconds.max(0.1);

    for (role, mut aircraft, mut damage, mut input) in &mut query {
        if aircraft.is_destroyed {
            damage.is_repairing = false;
            damage.repair_elapsed_seconds = 0.0;
            continue;
        }

        let wants_repair = input.repair && damage.has_repairable_damage(&aircraft);
        let was_repairing = damage.is_repairing;
        let repair_complete = damage.update_repair_progress(wants_repair, dt, repair_duration);

        if !was_repairing
            && damage.is_repairing
            && let Some(events) = events.as_deref_mut()
        {
            push_event(
                events,
                debug.tick_count,
                EnvironmentEventKind::RepairStarted,
                Some(role_subject(*role)),
                Some(aircraft.position),
                Some(0.0),
            );
        }

        if damage.is_repairing {
            input.throttle_delta = 0.0;
            input.brake = false;
            input.pitch = 0.0;
            input.roll = 0.0;
            input.yaw = 0.0;
            input.fire_gun = false;
        }

        if repair_complete {
            let restore_hp = damage.total_max_hit_points * config.game.repair_heal_fraction;
            damage.complete_repair_cycle(&mut aircraft, restore_hp);
            if let Some(events) = events.as_deref_mut() {
                push_event(
                    events,
                    debug.tick_count,
                    EnvironmentEventKind::RepairCompleted,
                    Some(role_subject(*role)),
                    Some(aircraft.position),
                    Some(restore_hp),
                );
            }
        }
    }
}

pub fn effective_aircraft_performance(
    base: &AircraftPerformance,
    damage: &AircraftDamageState,
    _input: &ControlInput,
) -> AircraftPerformance {
    let mut effective = base.clone();
    if damage.is_repairing {
        effective.pitch_positive_rate_limit_deg = 0.0;
        effective.pitch_negative_rate_limit_deg = 0.0;
        effective.roll_positive_rate_limit_deg = 0.0;
        effective.roll_negative_rate_limit_deg = 0.0;
        effective.yaw_positive_rate_limit_deg = 0.0;
        effective.yaw_negative_rate_limit_deg = 0.0;
        return effective;
    }

    let physical_left_scale = control_surface_scale(damage.left_wing.fraction(), base);
    let physical_right_scale = control_surface_scale(damage.right_wing.fraction(), base);
    let pitch_tail_scale = control_surface_scale(damage.pitch_tail.fraction(), base);
    let yaw_tail_scale = control_surface_scale(damage.yaw_tail.fraction(), base);
    let engine_scale = engine_scale(damage.engine.fraction(), base);

    effective.roll_positive_rate_limit_deg *= physical_right_scale;
    effective.roll_negative_rate_limit_deg *= physical_left_scale;
    effective.pitch_positive_rate_limit_deg *= pitch_tail_scale;
    effective.pitch_negative_rate_limit_deg *= pitch_tail_scale;
    effective.yaw_positive_rate_limit_deg *= yaw_tail_scale;
    effective.yaw_negative_rate_limit_deg *= yaw_tail_scale;
    effective.max_thrust *= engine_scale;
    effective.throttle_response *= engine_scale.max(base.damaged_engine_throttle_response_min);
    effective
}

pub fn damage_flight_modifiers(
    damage: &AircraftDamageState,
    performance: &AircraftPerformance,
) -> DamageFlightModifiers {
    let physical_left_scale = wing_lift_scale(damage.left_wing.fraction(), performance);
    let physical_right_scale = wing_lift_scale(damage.right_wing.fraction(), performance);
    let lift_difference = physical_right_scale - physical_left_scale;
    let roll_difference = physical_left_scale - physical_right_scale;
    let asymmetry = lift_difference.abs();
    let left_loss = 1.0 - physical_left_scale;
    let right_loss = 1.0 - physical_right_scale;
    let damaged_surface_severity = left_loss + right_loss;
    let softened_roll_difference = soften_trim_difference(roll_difference);
    let softened_lift_difference = soften_trim_difference(lift_difference);

    DamageFlightModifiers {
        left_wing_lift_scale: physical_left_scale,
        right_wing_lift_scale: physical_right_scale,
        roll_trim_rate_deg: softened_roll_difference
            * (performance.damage_roll_trim_base_deg
                + asymmetry * performance.damage_roll_trim_asymmetry_deg),
        yaw_trim_rate_deg: softened_lift_difference
            * (performance.damage_yaw_trim_base_deg
                + asymmetry * performance.damage_yaw_trim_asymmetry_deg),
        extra_drag: damaged_surface_severity * performance.damage_extra_drag_per_surface
            + asymmetry * performance.damage_extra_drag_asymmetry,
    }
}

pub fn select_hit_subsystem(
    local_hit_point: Vec3,
    damage: &AircraftDamageState,
) -> Option<AircraftSubsystem> {
    subsystem_hitboxes()
        .into_iter()
        .filter(|hitbox| damage.subsystem(hitbox.subsystem).stage() != DamageStage::Destroyed)
        .filter_map(|hitbox| {
            let local = local_hit_point - hitbox.center;
            if local.x.abs() <= hitbox.half_extents.x
                && local.y.abs() <= hitbox.half_extents.y
                && local.z.abs() <= hitbox.half_extents.z
            {
                Some((hitbox.subsystem, local.length_squared()))
            } else {
                None
            }
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(subsystem, _)| subsystem)
}

pub fn apply_collision_damage(
    aircraft: &mut AircraftState,
    damage: &mut AircraftDamageState,
    local_hit_point: Vec3,
    impact_speed: f32,
    damage_scale: f32,
) -> (f32, Option<AircraftSubsystem>, bool) {
    damage.interrupt_repair();

    let severity = (impact_speed - 8.0).max(0.0);
    if severity <= f32::EPSILON {
        return (0.0, None, false);
    }

    let total_damage = severity * damage_scale;
    let mut hit_subsystem = None;
    let mut subsystem_destroyed = false;
    if let Some(subsystem) = select_hit_subsystem(local_hit_point, damage) {
        let was_destroyed = damage.subsystem(subsystem).stage() == DamageStage::Destroyed;
        damage.apply_subsystem_damage(subsystem, total_damage * 1.2);
        let is_destroyed = damage.subsystem(subsystem).stage() == DamageStage::Destroyed;
        hit_subsystem = Some(subsystem);
        subsystem_destroyed = !was_destroyed && is_destroyed;
    }

    aircraft.hit_points = (aircraft.hit_points - total_damage).max(0.0);
    if aircraft.hit_points <= f32::EPSILON {
        aircraft.is_destroyed = true;
    }

    (total_damage, hit_subsystem, subsystem_destroyed)
}

pub fn subsystem_hitboxes() -> [SubsystemHitbox; 5] {
    [
        SubsystemHitbox {
            name: "LeftWing",
            subsystem: AircraftSubsystem::LeftWing,
            center: Vec3::new(5.6, 0.0, 0.0),
            half_extents: Vec3::new(5.8, 0.17, 1.2),
        },
        SubsystemHitbox {
            name: "RightWing",
            subsystem: AircraftSubsystem::RightWing,
            center: Vec3::new(-5.6, 0.0, 0.0),
            half_extents: Vec3::new(5.8, 0.17, 1.2),
        },
        SubsystemHitbox {
            name: "PitchTail",
            subsystem: AircraftSubsystem::PitchTail,
            center: Vec3::new(0.0, 0.0, -7.0),
            half_extents: Vec3::new(5.2, 0.15, 0.9),
        },
        SubsystemHitbox {
            name: "YawTail",
            subsystem: AircraftSubsystem::YawTail,
            center: Vec3::new(0.0, 1.3, -7.0),
            half_extents: Vec3::new(0.25, 1.5, 0.9),
        },
        SubsystemHitbox {
            name: "Engine",
            subsystem: AircraftSubsystem::Engine,
            center: Vec3::new(0.0, 0.0, 9.9),
            half_extents: Vec3::new(0.8, 0.5, 1.4),
        },
    ]
}

fn control_surface_scale(health_fraction: f32, performance: &AircraftPerformance) -> f32 {
    blended_damage_scale(
        health_fraction,
        performance.damaged_control_surface_scale,
        performance.destroyed_control_surface_scale,
    )
}

fn engine_scale(health_fraction: f32, performance: &AircraftPerformance) -> f32 {
    blended_damage_scale(
        health_fraction,
        performance.damaged_engine_thrust_scale,
        performance.destroyed_engine_thrust_scale,
    )
}

fn wing_lift_scale(health_fraction: f32, performance: &AircraftPerformance) -> f32 {
    blended_damage_scale(
        health_fraction,
        performance.damaged_wing_lift_scale,
        performance.destroyed_wing_lift_scale,
    )
}

fn blended_damage_scale(health_fraction: f32, damaged_scale: f32, destroyed_scale: f32) -> f32 {
    let health_fraction = health_fraction.clamp(0.0, 1.0);
    if health_fraction >= DAMAGE_STAGE_THRESHOLD {
        let t = ((1.0 - health_fraction) / (1.0 - DAMAGE_STAGE_THRESHOLD))
            .clamp(0.0, 1.0)
            .powf(DAMAGE_GRACE_EXPONENT);
        let t = smoothstep(t);
        1.0 + (damaged_scale - 1.0) * t
    } else {
        let t = ((DAMAGE_STAGE_THRESHOLD - health_fraction) / DAMAGE_STAGE_THRESHOLD)
            .clamp(0.0, 1.0)
            .powf(DAMAGE_FAILURE_EXPONENT);
        let t = smoothstep(t);
        damaged_scale + (destroyed_scale - damaged_scale) * t
    }
}

fn soften_trim_difference(value: f32) -> f32 {
    value.signum() * value.abs().powf(DAMAGE_TRIM_SOFTEN_EXPONENT)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::{
        AircraftDamageState, AircraftSubsystem, apply_spawn_damage_overrides,
        damage_flight_modifiers, effective_aircraft_performance, select_hit_subsystem,
        soften_trim_difference,
    };
    use crate::core::config::{AircraftConfig, SpawnDamageConfig};
    use crate::input::actions::ControlInput;
    use crate::simulation::components::AircraftState;
    use crate::simulation::systems::aircraft_performance_from_config;
    use bevy::prelude::Vec3;

    #[test]
    fn softened_trim_difference_reduces_moderate_asymmetry() {
        let raw = 0.4;
        let softened = soften_trim_difference(raw);
        assert!(softened > 0.0);
        assert!(softened < raw);
    }

    #[test]
    fn left_wing_damage_produces_negative_roll_trim() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.left_wing.current = 0.0;
        let modifiers = damage_flight_modifiers(&damage, &performance);
        assert!(modifiers.roll_trim_rate_deg < 0.0);
    }

    #[test]
    fn right_wing_damage_produces_positive_roll_trim() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.right_wing.current = 0.0;
        let modifiers = damage_flight_modifiers(&damage, &performance);
        assert!(modifiers.roll_trim_rate_deg > 0.0);
    }

    #[test]
    fn left_wing_damage_reduces_negative_roll_rate_limit() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.left_wing.current = 0.0;
        let effective =
            effective_aircraft_performance(&performance, &damage, &ControlInput::default());
        assert_eq!(
            effective.roll_positive_rate_limit_deg,
            performance.roll_positive_rate_limit_deg
        );
        assert!(effective.roll_negative_rate_limit_deg < performance.roll_negative_rate_limit_deg);
    }

    #[test]
    fn right_wing_damage_reduces_positive_roll_rate_limit() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.right_wing.current = 0.0;
        let effective =
            effective_aircraft_performance(&performance, &damage, &ControlInput::default());
        assert_eq!(
            effective.roll_negative_rate_limit_deg,
            performance.roll_negative_rate_limit_deg
        );
        assert!(effective.roll_positive_rate_limit_deg < performance.roll_positive_rate_limit_deg);
    }

    #[test]
    fn subsystem_selection_hits_left_wing_at_wing_center() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let subsystem = select_hit_subsystem(Vec3::new(9.5, 0.0, 0.0), &damage);
        assert_eq!(subsystem, Some(AircraftSubsystem::LeftWing));
    }

    #[test]
    fn subsystem_selection_ignores_body_point() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let subsystem = select_hit_subsystem(Vec3::new(0.0, 0.0, 4.0), &damage);
        assert_eq!(subsystem, None);
    }

    #[test]
    fn spawn_damage_overrides_floor_total_hit_points_fraction() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut aircraft = AircraftState {
            hit_points: 100.0,
            ..AircraftState::default()
        };
        let mut damage = AircraftDamageState::new(100.0, &performance);
        let spawn_damage = SpawnDamageConfig {
            total_hit_points_fraction: None,
            left_wing_fraction: Some(0.0),
            right_wing_fraction: Some(0.0),
            pitch_tail_fraction: Some(0.0),
            yaw_tail_fraction: Some(0.0),
            engine_fraction: Some(0.0),
        };
        apply_spawn_damage_overrides(
            &mut aircraft,
            &mut damage,
            &performance,
            Some(&spawn_damage),
        );
        assert!((aircraft.hit_points - 20.0).abs() <= 1e-4);
        assert_eq!(damage.left_wing.current, 0.0);
        assert_eq!(damage.engine.current, 0.0);
    }

    #[test]
    fn completed_repair_restores_every_subsystem_in_one_cycle() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut aircraft = AircraftState {
            hit_points: 40.0,
            ..AircraftState::default()
        };
        let mut damage = AircraftDamageState::new(100.0, &performance);
        damage.is_repairing = true;
        damage.repair_elapsed_seconds = 10.0;
        damage.left_wing.current = damage.left_wing.max * 0.2;
        damage.right_wing.current = damage.right_wing.max * 0.9;
        damage.pitch_tail.current = damage.pitch_tail.max * 0.4;
        damage.yaw_tail.current = damage.yaw_tail.max * 0.8;
        damage.engine.current = damage.engine.max * 0.6;

        damage.complete_repair_cycle(&mut aircraft, 25.0);

        assert_eq!(aircraft.hit_points, 65.0);
        for (_, subsystem) in damage.all_subsystems() {
            assert_eq!(subsystem.current, subsystem.max);
        }
        assert!(!damage.is_repairing);
        assert_eq!(damage.repair_elapsed_seconds, 0.0);
    }
}
