use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::config::GunConfig;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AircraftRole {
    Fighter1,
    Fighter2,
}

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlAuthority {
    Human,
    BuiltInAi,
    ExternalAgent,
    Replay,
}

#[derive(Component, Debug, Clone)]
pub struct AircraftState {
    pub position: Vec3,
    pub velocity: Vec3,
    pub orientation: Quat,
    pub forward: Vec3,
    pub angular_rates_deg: Vec3,
    pub throttle: f32,
    pub hit_points: f32,
    pub stall_factor: f32,
    pub out_of_bounds_seconds: f32,
    pub ceiling_recovery_seconds: f32,
    pub ceiling_recovery_target_pitch_deg: f32,
    pub is_destroyed: bool,
}

impl Default for AircraftState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            forward: Vec3::Z,
            angular_rates_deg: Vec3::ZERO,
            throttle: 0.5,
            hit_points: 100.0,
            stall_factor: 0.0,
            out_of_bounds_seconds: 0.0,
            ceiling_recovery_seconds: 0.0,
            ceiling_recovery_target_pitch_deg: -30.0,
            is_destroyed: false,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct AircraftPerformance {
    pub max_level_speed: f32,
    pub cruise_reference_throttle: f32,
    pub cruise_reference_speed: f32,
    pub maneuver_reference_speed: f32,
    pub max_thrust: f32,
    pub trim_pitch_degrees: f32,
    pub trim_angle_of_attack_radians: f32,
    pub reference_level_lift_factor: f32,
    pub throttle_response: f32,
    pub lift_coefficient: f32,
    pub induced_drag_coefficient: f32,
    pub side_drag_coefficient: f32,
    pub gravity_scale: f32,
    pub linear_drag: f32,
    pub brake_drag: f32,
    pub weight_kg: f32,
    pub stall_speed: f32,
    pub stall_recovery_speed: f32,
    pub stall_reference_dynamic_pressure: f32,
    pub stall_recovery_dynamic_pressure: f32,
    pub pitch_response: f32,
    pub yaw_response: f32,
    pub roll_response: f32,
    pub angular_damping: f32,
    pub pitch_positive_rate_limit_deg: f32,
    pub pitch_negative_rate_limit_deg: f32,
    pub roll_positive_rate_limit_deg: f32,
    pub roll_negative_rate_limit_deg: f32,
    pub yaw_positive_rate_limit_deg: f32,
    pub yaw_negative_rate_limit_deg: f32,
    pub pitch_maneuver_scale: f32,
    pub roll_maneuver_scale: f32,
    pub yaw_maneuver_scale: f32,
    pub pitch_low_speed_scale: f32,
    pub roll_low_speed_scale: f32,
    pub yaw_low_speed_scale: f32,
    pub pitch_high_speed_max_scale: f32,
    pub roll_high_speed_max_scale: f32,
    pub yaw_high_speed_max_scale: f32,
    pub left_wing_hit_points: f32,
    pub right_wing_hit_points: f32,
    pub pitch_tail_hit_points: f32,
    pub yaw_tail_hit_points: f32,
    pub engine_hit_points: f32,
    pub damaged_control_surface_scale: f32,
    pub destroyed_control_surface_scale: f32,
    pub damaged_engine_thrust_scale: f32,
    pub destroyed_engine_thrust_scale: f32,
    pub damaged_engine_throttle_response_min: f32,
    pub damaged_wing_lift_scale: f32,
    pub destroyed_wing_lift_scale: f32,
    pub damage_roll_trim_base_deg: f32,
    pub damage_roll_trim_asymmetry_deg: f32,
    pub damage_yaw_trim_base_deg: f32,
    pub damage_yaw_trim_asymmetry_deg: f32,
    pub damage_extra_drag_per_surface: f32,
    pub damage_extra_drag_asymmetry: f32,
    pub gun: GunConfig,
}

#[derive(Component, Debug, Clone)]
pub struct SpawnTransform {
    pub position: Vec3,
    pub orientation: Quat,
}

#[derive(Component, Debug, Clone, Default)]
pub struct GunState {
    pub cooldown_seconds: f32,
    pub heat: f32,
    pub overheated: bool,
    pub is_firing: bool,
}
