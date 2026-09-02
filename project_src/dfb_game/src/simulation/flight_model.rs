use bevy::prelude::*;

use crate::core::config::STANDARD_WEIGHT_KG;
use crate::core::math::clamp_unit;
use crate::gameplay::damage::DamageFlightModifiers;
use crate::input::actions::ControlInput;
use crate::simulation::components::{AircraftPerformance, AircraftState};

#[derive(Debug, Clone, Copy)]
struct AirframeState {
    airspeed: f32,
    dynamic_pressure: f32,
    local_velocity: Vec3,
    forward_speed: f32,
    angle_of_attack: f32,
    airflow_direction: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct AerodynamicForces {
    thrust: Vec3,
    lift: Vec3,
    drag: Vec3,
    gravity: Vec3,
}

#[derive(Debug, Clone, Copy)]
struct AngularAccelerationComponents {
    control: Vec3,
    damping: Vec3,
}

pub fn step_aircraft(
    state: &mut AircraftState,
    performance: &AircraftPerformance,
    modifiers: &DamageFlightModifiers,
    input: &ControlInput,
    delta_seconds: f32,
) {
    let throttle_step = performance.throttle_response * delta_seconds;
    state.throttle = (state.throttle + input.throttle_delta * throttle_step).clamp(0.0, 1.0);

    let airframe = sample_airframe_state(state, performance);
    let stall_factor = compute_stall_factor(&airframe, performance, modifiers, input);
    state.stall_factor = stall_factor;
    integrate_rotation(
        state,
        performance,
        modifiers,
        input,
        &airframe,
        stall_factor,
        delta_seconds,
    );
    let airframe = sample_airframe_state(state, performance);
    let forces = compute_aerodynamic_forces(
        state,
        performance,
        modifiers,
        input,
        &airframe,
        stall_factor,
    );

    state.velocity += (forces.thrust + forces.lift + forces.drag + forces.gravity) * delta_seconds;
    let forward_component = state.velocity.dot(state.forward);
    if forward_component < 0.0 {
        state.velocity -= state.forward * forward_component;
    }
    state.position += state.velocity * delta_seconds;
}

fn sample_airframe_state(
    state: &AircraftState,
    performance: &AircraftPerformance,
) -> AirframeState {
    let airspeed = state.velocity.length().max(0.001);
    let local_velocity = state.orientation.inverse() * state.velocity;
    let forward_speed = local_velocity.z.max(0.0);
    let airflow_direction = if airspeed > f32::EPSILON {
        state.velocity / airspeed
    } else {
        state.forward
    };

    AirframeState {
        airspeed,
        dynamic_pressure: (airspeed / performance.cruise_reference_speed.max(1.0)).powi(2),
        local_velocity,
        forward_speed,
        angle_of_attack: (-local_velocity.y).atan2(local_velocity.z.abs().max(0.1)),
        airflow_direction,
    }
}

fn integrate_rotation(
    state: &mut AircraftState,
    performance: &AircraftPerformance,
    modifiers: &DamageFlightModifiers,
    input: &ControlInput,
    airframe: &AirframeState,
    stall_factor: f32,
    delta_seconds: f32,
) {
    let control_authority = control_authority(airframe, performance, stall_factor);
    let angular_acceleration = compute_angular_acceleration(
        state,
        performance,
        modifiers,
        input,
        airframe,
        control_authority,
    );
    let total_angular_acceleration = angular_acceleration.control + angular_acceleration.damping;
    state.angular_rates_deg += total_angular_acceleration * delta_seconds;

    let pitch_delta = Quat::from_axis_angle(
        state.orientation * Vec3::X,
        state.angular_rates_deg.x.to_radians() * delta_seconds,
    );
    let yaw_delta = Quat::from_axis_angle(
        state.orientation * Vec3::Y,
        state.angular_rates_deg.y.to_radians() * delta_seconds,
    );
    let roll_delta = Quat::from_axis_angle(
        state.orientation * Vec3::Z,
        state.angular_rates_deg.z.to_radians() * delta_seconds,
    );
    state.orientation = (yaw_delta * pitch_delta * roll_delta * state.orientation).normalize();
    state.forward = (state.orientation * Vec3::Z).normalize_or_zero();
    if state.forward == Vec3::ZERO {
        state.forward = Vec3::Z;
        state.orientation = Quat::IDENTITY;
    }
}

fn compute_angular_acceleration(
    state: &AircraftState,
    performance: &AircraftPerformance,
    modifiers: &DamageFlightModifiers,
    input: &ControlInput,
    airframe: &AirframeState,
    control_authority: f32,
) -> AngularAccelerationComponents {
    let disturbance_scale = airframe.dynamic_pressure * (0.7 + control_authority * 0.3);
    let disturbance_rates = Vec3::new(
        0.0,
        modifiers.yaw_trim_rate_deg * disturbance_scale,
        modifiers.roll_trim_rate_deg * disturbance_scale,
    );
    let desired_rates =
        desired_angular_rates(input, airframe, performance, control_authority) + disturbance_rates;
    let rate_response = Vec3::new(
        performance.pitch_response,
        performance.yaw_response,
        performance.roll_response,
    );
    let rate_error = desired_rates - state.angular_rates_deg;
    let control = Vec3::new(
        rate_error.x * rate_response.x,
        rate_error.y * rate_response.y,
        rate_error.z * rate_response.z,
    );

    let damping_pressure_scale = 0.35 + 0.65 * smoothstep(airframe.dynamic_pressure.min(1.0));
    let damping = -state.angular_rates_deg * performance.angular_damping * damping_pressure_scale;

    AngularAccelerationComponents { control, damping }
}

fn desired_angular_rates(
    input: &ControlInput,
    airframe: &AirframeState,
    performance: &AircraftPerformance,
    control_authority: f32,
) -> Vec3 {
    let pitch_input = clamp_unit(input.pitch);
    let yaw_input = clamp_unit(input.yaw);
    let roll_input = clamp_unit(input.roll);
    let rate_scale = non_stall_rate_scale(airframe.forward_speed, performance);

    let pitch_limit = if pitch_input >= 0.0 {
        performance.pitch_positive_rate_limit_deg * rate_scale.x
    } else {
        performance.pitch_negative_rate_limit_deg * rate_scale.x
    };
    let yaw_limit = if yaw_input >= 0.0 {
        performance.yaw_positive_rate_limit_deg * rate_scale.y
    } else {
        performance.yaw_negative_rate_limit_deg * rate_scale.y
    };
    let roll_limit = if roll_input >= 0.0 {
        performance.roll_positive_rate_limit_deg * rate_scale.z
    } else {
        performance.roll_negative_rate_limit_deg * rate_scale.z
    };

    Vec3::new(
        pitch_input * pitch_limit * control_authority,
        yaw_input * yaw_limit * control_authority,
        roll_input * roll_limit * control_authority,
    )
}

fn non_stall_rate_scale(airspeed: f32, performance: &AircraftPerformance) -> Vec3 {
    let recovery = performance.stall_recovery_speed.max(0.1);
    let cruise = performance.cruise_reference_speed.max(recovery + 1.0);
    let maneuver = performance.maneuver_reference_speed.max(cruise + 1.0);
    let high_speed = performance.max_level_speed.max(maneuver + 1.0);

    let low = Vec3::new(
        performance.pitch_low_speed_scale,
        performance.yaw_low_speed_scale,
        performance.roll_low_speed_scale,
    );
    let unit = Vec3::ONE;
    let maneuver_scale = Vec3::new(
        performance.pitch_maneuver_scale,
        performance.yaw_maneuver_scale,
        performance.roll_maneuver_scale,
    );
    let high = Vec3::new(
        performance.pitch_high_speed_max_scale,
        performance.yaw_high_speed_max_scale,
        performance.roll_high_speed_max_scale,
    );

    if airspeed <= cruise {
        let t = smoothstep(((airspeed - recovery) / (cruise - recovery)).clamp(0.0, 1.0));
        return low.lerp(unit, t);
    }

    let standard_t = smoothstep(((airspeed - cruise) / (maneuver - cruise)).clamp(0.0, 1.0));
    if airspeed <= maneuver {
        return unit.lerp(maneuver_scale, standard_t);
    }

    let high_t = smoothstep(((airspeed - maneuver) / (high_speed - maneuver)).clamp(0.0, 1.0));
    maneuver_scale.lerp(high, high_t)
}

fn control_authority(
    _airframe: &AirframeState,
    _performance: &AircraftPerformance,
    stall_factor: f32,
) -> f32 {
    1.0 - stall_factor * 0.78
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn compute_aerodynamic_forces(
    state: &AircraftState,
    performance: &AircraftPerformance,
    modifiers: &DamageFlightModifiers,
    input: &ControlInput,
    airframe: &AirframeState,
    stall_factor: f32,
) -> AerodynamicForces {
    let local_up = state.orientation * Vec3::Y;
    let lift_direction = (local_up
        - airframe.airflow_direction * local_up.dot(airframe.airflow_direction))
    .normalize_or_zero();
    let lift_factor = (performance.reference_level_lift_factor
        + (airframe.angle_of_attack - performance.trim_angle_of_attack_radians) * 2.4)
        .clamp(0.0, 1.5);
    let wing_lift_average =
        (modifiers.left_wing_lift_scale + modifiers.right_wing_lift_scale) * 0.5;
    let lift_strength = performance.lift_coefficient
        * airframe.dynamic_pressure
        * lift_factor
        * (1.0 - stall_factor).max(0.0)
        * wing_lift_average
        * (airframe.forward_speed / airframe.airspeed.max(0.1)).clamp(0.0, 1.0);
    let lift = if lift_direction == Vec3::ZERO {
        Vec3::ZERO
    } else {
        lift_direction * lift_strength
    };

    let mut local_drag = Vec3::new(
        -airframe.local_velocity.x
            * airframe.local_velocity.x.abs()
            * performance.side_drag_coefficient,
        -airframe.local_velocity.y
            * airframe.local_velocity.y.abs()
            * performance.side_drag_coefficient
            * 0.75,
        -airframe.local_velocity.z * airframe.local_velocity.z.abs() * performance.linear_drag,
    );
    local_drag.z -= lift_strength * performance.induced_drag_coefficient;
    local_drag.z -=
        airframe.local_velocity.z * airframe.local_velocity.z.abs() * modifiers.extra_drag;
    if input.brake {
        local_drag.z -=
            airframe.local_velocity.z * airframe.local_velocity.z.abs() * performance.brake_drag;
    }
    let drag = state.orientation * local_drag;

    AerodynamicForces {
        thrust: state.forward * performance.max_thrust * state.throttle,
        lift,
        drag,
        gravity: Vec3::NEG_Y * 9.81 * performance.gravity_scale,
    }
}

fn compute_stall_factor(
    airframe: &AirframeState,
    performance: &AircraftPerformance,
    modifiers: &DamageFlightModifiers,
    input: &ControlInput,
) -> f32 {
    // First-pass capacity/demand model:
    // - capacity uses normalized dynamic pressure and current wing health
    // - demand comes from fixed aircraft weight plus current maneuver intent
    // - stall thresholds are expressed directly in capacity-ratio space, not raw speed
    let wing_capacity_scale =
        ((modifiers.left_wing_lift_scale + modifiers.right_wing_lift_scale) * 0.5).max(0.05);
    let pitch_demand = clamp_unit(input.pitch).abs();
    let roll_demand = clamp_unit(input.roll).abs();
    let yaw_demand = clamp_unit(input.yaw).abs();
    let maneuver_demand = 1.0 + pitch_demand * 0.30 + roll_demand * 0.18 + yaw_demand * 0.08;
    let weight_demand =
        (performance.weight_kg / STANDARD_WEIGHT_KG).max(0.1) * performance.gravity_scale;
    let capacity_ratio =
        airframe.dynamic_pressure * wing_capacity_scale.powi(2) / (weight_demand * maneuver_demand);
    let full_stall_ratio = performance.stall_reference_dynamic_pressure.max(0.001);
    let clear_ratio = performance
        .stall_recovery_dynamic_pressure
        .max(full_stall_ratio + 1e-6);

    if capacity_ratio >= clear_ratio {
        0.0
    } else if capacity_ratio <= full_stall_ratio {
        1.0
    } else {
        1.0 - (capacity_ratio - full_stall_ratio) / (clear_ratio - full_stall_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AirframeState, compute_angular_acceleration, compute_stall_factor, non_stall_rate_scale,
        sample_airframe_state, step_aircraft,
    };
    use crate::core::config::{AircraftConfig, AircraftSpecConfig, STANDARD_WEIGHT_KG};
    use crate::gameplay::damage::DamageFlightModifiers;
    use crate::input::actions::ControlInput;
    use crate::simulation::components::{AircraftPerformance, AircraftState};
    use crate::simulation::systems::aircraft_performance_from_config;
    use bevy::prelude::{Quat, Vec3};

    #[test]
    fn non_stall_rate_scale_is_unity_below_recovery_speed() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let scale = non_stall_rate_scale(performance.stall_recovery_speed - 5.0, &performance);
        assert!((scale.x - performance.pitch_low_speed_scale).abs() < 1e-6);
        assert!((scale.y - performance.yaw_low_speed_scale).abs() < 1e-6);
        assert!((scale.z - performance.roll_low_speed_scale).abs() < 1e-6);
    }

    #[test]
    fn non_stall_rate_scale_reaches_configured_caps_asymptotically() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let scale = non_stall_rate_scale(performance.max_level_speed + 500.0, &performance);
        assert!((scale.x - performance.pitch_high_speed_max_scale).abs() < 1e-6);
        assert!((scale.y - performance.yaw_high_speed_max_scale).abs() < 1e-6);
        assert!((scale.z - performance.roll_high_speed_max_scale).abs() < 1e-6);
    }

    #[test]
    fn non_stall_rate_scale_is_unity_at_cruise_reference_speed() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let scale = non_stall_rate_scale(performance.cruise_reference_speed, &performance);
        assert!((scale.x - 1.0).abs() < 1e-6);
        assert!((scale.y - 1.0).abs() < 1e-6);
        assert!((scale.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_stall_rate_scale_reaches_maneuver_soft_cap_at_reference_speed() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let scale = non_stall_rate_scale(performance.maneuver_reference_speed, &performance);
        assert!((scale.x - performance.pitch_maneuver_scale).abs() < 1e-6);
        assert!((scale.y - performance.yaw_maneuver_scale).abs() < 1e-6);
        assert!((scale.z - performance.roll_maneuver_scale).abs() < 1e-6);
    }

    fn sample_airframe(performance: &AircraftPerformance, forward_speed: f32) -> AirframeState {
        AirframeState {
            airspeed: forward_speed.max(0.001),
            dynamic_pressure: derived_dynamic_pressure_from_speed(
                forward_speed.max(0.001),
                performance.cruise_reference_speed,
            ),
            local_velocity: Vec3::new(0.0, 0.0, forward_speed),
            forward_speed,
            angle_of_attack: 0.0,
            airflow_direction: Vec3::Z,
        }
    }

    fn derived_dynamic_pressure_from_speed(speed: f32, reference_speed: f32) -> f32 {
        (speed / reference_speed.max(0.1)).powi(2)
    }

    fn sample_trimmed_level_state(performance: &AircraftPerformance) -> AircraftState {
        let orientation = Quat::from_rotation_x(-performance.trim_pitch_degrees.to_radians());
        AircraftState {
            position: Vec3::ZERO,
            velocity: Vec3::Z * performance.cruise_reference_speed,
            orientation,
            forward: orientation * Vec3::Z,
            throttle: performance.cruise_reference_throttle,
            ..Default::default()
        }
    }

    fn performance_with_weight(weight_kg: f32) -> AircraftPerformance {
        let mut spec = AircraftSpecConfig::default_standard();
        spec.weight_kg = weight_kg;
        aircraft_performance_from_config(&AircraftConfig::from_spec(&spec))
    }

    fn clean_damage_modifiers() -> DamageFlightModifiers {
        DamageFlightModifiers {
            left_wing_lift_scale: 1.0,
            right_wing_lift_scale: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn stall_factor_is_full_at_stall_reference_dynamic_pressure() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let speed = performance.cruise_reference_speed
            * performance.stall_reference_dynamic_pressure.sqrt();
        let airframe = sample_airframe(&performance, speed);
        let factor = compute_stall_factor(
            &airframe,
            &performance,
            &DamageFlightModifiers::default(),
            &ControlInput::default(),
        );
        assert!((factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weight_variants_keep_full_stall_at_their_reference_stall_speed() {
        let lighter = performance_with_weight(STANDARD_WEIGHT_KG * 0.8);
        let heavier = performance_with_weight(STANDARD_WEIGHT_KG * 1.2);

        let lighter_factor = compute_stall_factor(
            &sample_airframe(&lighter, lighter.stall_speed),
            &lighter,
            &clean_damage_modifiers(),
            &ControlInput::default(),
        );
        let heavier_factor = compute_stall_factor(
            &sample_airframe(&heavier, heavier.stall_speed),
            &heavier,
            &clean_damage_modifiers(),
            &ControlInput::default(),
        );

        assert!((lighter_factor - 1.0).abs() < 1e-6);
        assert!((heavier_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weight_variants_clear_stall_at_their_recovery_speed() {
        let lighter = performance_with_weight(STANDARD_WEIGHT_KG * 0.8);
        let heavier = performance_with_weight(STANDARD_WEIGHT_KG * 1.2);

        let lighter_factor = compute_stall_factor(
            &sample_airframe(&lighter, lighter.stall_recovery_speed),
            &lighter,
            &clean_damage_modifiers(),
            &ControlInput::default(),
        );
        let heavier_factor = compute_stall_factor(
            &sample_airframe(&heavier, heavier.stall_recovery_speed),
            &heavier,
            &clean_damage_modifiers(),
            &ControlInput::default(),
        );

        assert!(lighter_factor <= 1e-6);
        assert!(heavier_factor <= 1e-6);
    }

    #[test]
    fn recovery_speed_keeps_full_control_authority_but_low_speed_rate_floor() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let airframe = sample_airframe(&performance, performance.stall_recovery_speed);
        let authority = super::control_authority(&airframe, &performance, 0.0);
        let scale = non_stall_rate_scale(performance.stall_recovery_speed, &performance);

        assert!((authority - 1.0).abs() < 1e-6);
        assert!((scale.x - performance.pitch_low_speed_scale).abs() < 1e-6);
        assert!((scale.y - performance.yaw_low_speed_scale).abs() < 1e-6);
        assert!((scale.z - performance.roll_low_speed_scale).abs() < 1e-6);
    }

    #[test]
    fn trimmed_level_state_samples_expected_angle_of_attack() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let state = sample_trimmed_level_state(&performance);
        let airframe = sample_airframe_state(&state, &performance);
        assert!((airframe.angle_of_attack - performance.trim_angle_of_attack_radians).abs() < 1e-4);
    }

    #[test]
    fn trimmed_level_state_does_not_generate_spurious_pitch_rate() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let mut state = sample_trimmed_level_state(&performance);
        step_aircraft(
            &mut state,
            &performance,
            &DamageFlightModifiers::default(),
            &ControlInput::default(),
            1.0 / 60.0,
        );
        assert!(state.angular_rates_deg.x.abs() < 1e-4);
    }

    #[test]
    fn trimmed_level_state_has_near_zero_total_pitch_angular_acceleration() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let state = sample_trimmed_level_state(&performance);
        let airframe = sample_airframe_state(&state, &performance);
        let angular_acceleration = compute_angular_acceleration(
            &state,
            &performance,
            &DamageFlightModifiers::default(),
            &ControlInput::default(),
            &airframe,
            super::control_authority(&airframe, &performance, 0.0),
        );
        let total_pitch = angular_acceleration.control.x + angular_acceleration.damping.x;
        assert!(total_pitch.abs() < 1e-4);
    }

    #[test]
    fn control_authority_is_full_when_not_stalled() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let airframe = sample_airframe(&performance, performance.cruise_reference_speed);
        let authority = super::control_authority(&airframe, &performance, 0.0);
        assert!((authority - 1.0).abs() < 1e-6);
    }

    #[test]
    fn control_authority_only_drops_with_stall_factor() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let airframe = sample_airframe(&performance, performance.stall_recovery_speed);
        let low = super::control_authority(&airframe, &performance, 0.2);
        let high = super::control_authority(&airframe, &performance, 0.8);
        assert!(low > high);
        assert!(high >= 0.0);
    }

    #[test]
    fn positive_pitch_rate_drops_nose() {
        let mut state = AircraftState {
            orientation: Quat::IDENTITY,
            forward: Vec3::Z,
            angular_rates_deg: Vec3::new(30.0, 0.0, 0.0),
            ..Default::default()
        };
        let nose_before = state.orientation * Vec3::Z;
        let pitch_delta = Quat::from_axis_angle(
            state.orientation * Vec3::X,
            state.angular_rates_deg.x.to_radians() * (1.0 / 60.0),
        );
        state.orientation = (pitch_delta * state.orientation).normalize();
        let nose_after = state.orientation * Vec3::Z;
        assert!(nose_before.y.abs() < 1e-6);
        assert!(nose_after.y < 0.0);
    }

    #[test]
    fn positive_yaw_rate_turns_nose_left() {
        let mut state = AircraftState {
            orientation: Quat::IDENTITY,
            forward: Vec3::Z,
            angular_rates_deg: Vec3::new(0.0, 30.0, 0.0),
            ..Default::default()
        };
        let nose_before = state.orientation * Vec3::Z;
        let yaw_delta = Quat::from_axis_angle(
            state.orientation * Vec3::Y,
            state.angular_rates_deg.y.to_radians() * (1.0 / 60.0),
        );
        state.orientation = (yaw_delta * state.orientation).normalize();
        let nose_after = state.orientation * Vec3::Z;
        assert!(nose_before.x.abs() < 1e-6);
        assert!(nose_after.x > 0.0);
    }

    #[test]
    fn positive_roll_rate_drops_right_wing() {
        let mut state = AircraftState {
            orientation: Quat::IDENTITY,
            forward: Vec3::Z,
            angular_rates_deg: Vec3::new(0.0, 0.0, 30.0),
            ..Default::default()
        };
        let right_wing_before = state.orientation * -Vec3::X;
        let roll_delta = Quat::from_axis_angle(
            state.orientation * Vec3::Z,
            state.angular_rates_deg.z.to_radians() * (1.0 / 60.0),
        );
        state.orientation = (roll_delta * state.orientation).normalize();
        let right_wing_after = state.orientation * -Vec3::X;
        assert!(right_wing_before.y.abs() < 1e-6);
        assert!(right_wing_after.y < 0.0);
    }
}
