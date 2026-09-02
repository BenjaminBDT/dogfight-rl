use bevy::prelude::*;

use crate::{gameplay::combat::presentation_muzzle_pose, simulation::components::AircraftState};

const BODY_AXIS_LENGTH_METERS: f32 = 24.0;
const BODY_AXIS_TIP_LENGTH_METERS: f32 = 2.4;
const AIM_RAY_LENGTH_METERS: f32 = 100_000.0;

const BODY_LEFT_COLOR: Color = Color::srgb(0.95, 0.12, 0.12);
const BODY_UP_COLOR: Color = Color::srgb(0.15, 0.95, 0.25);
const BODY_FORWARD_COLOR: Color = Color::srgb(0.15, 0.45, 1.0);
const AIM_RAY_COLOR: Color = Color::srgb(1.0, 0.78, 0.08);

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct FlightAssistOverlayState {
    pub visible: bool,
}

#[derive(Debug, Clone, Copy)]
struct FlightAssistGeometry {
    origin: Vec3,
    body_left_end: Vec3,
    body_up_end: Vec3,
    body_forward_end: Vec3,
    aim_origin: Vec3,
    aim_end: Vec3,
}

pub fn toggle_flight_assist_overlay(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<FlightAssistOverlayState>,
) {
    if keyboard.just_pressed(KeyCode::F4) {
        state.visible = !state.visible;
    }
}

pub fn draw_flight_assist_overlay(
    state: Res<FlightAssistOverlayState>,
    aircraft_query: Query<(&AircraftState, &Transform)>,
    mut gizmos: Gizmos,
) {
    if !state.visible {
        return;
    }

    for (aircraft, transform) in &aircraft_query {
        if aircraft.is_destroyed {
            continue;
        }

        let geometry = flight_assist_geometry(transform);
        gizmos
            .arrow(geometry.origin, geometry.body_left_end, BODY_LEFT_COLOR)
            .with_tip_length(BODY_AXIS_TIP_LENGTH_METERS);
        gizmos
            .arrow(geometry.origin, geometry.body_up_end, BODY_UP_COLOR)
            .with_tip_length(BODY_AXIS_TIP_LENGTH_METERS);
        gizmos
            .arrow(
                geometry.origin,
                geometry.body_forward_end,
                BODY_FORWARD_COLOR,
            )
            .with_tip_length(BODY_AXIS_TIP_LENGTH_METERS);
        gizmos.line(geometry.aim_origin, geometry.aim_end, AIM_RAY_COLOR);
    }
}

fn flight_assist_geometry(transform: &Transform) -> FlightAssistGeometry {
    let origin = transform.translation;
    let body_left = transform.rotation * Vec3::X;
    let body_up = transform.rotation * Vec3::Y;
    let body_forward = transform.rotation * Vec3::Z;
    let (aim_origin, aim_forward) = presentation_muzzle_pose(transform);

    FlightAssistGeometry {
        origin,
        body_left_end: origin + body_left * BODY_AXIS_LENGTH_METERS,
        body_up_end: origin + body_up * BODY_AXIS_LENGTH_METERS,
        body_forward_end: origin + body_forward * BODY_AXIS_LENGTH_METERS,
        aim_origin,
        aim_end: aim_origin + aim_forward * AIM_RAY_LENGTH_METERS,
    }
}

#[cfg(test)]
mod tests {
    use super::{AIM_RAY_LENGTH_METERS, BODY_AXIS_LENGTH_METERS, flight_assist_geometry};
    use bevy::prelude::*;

    const EPSILON: f32 = 0.01;

    fn assert_vec3_approx_eq(actual: Vec3, expected: Vec3) {
        assert!(
            actual.distance(expected) <= EPSILON,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn identity_pose_uses_frozen_aircraft_body_axes() {
        let geometry =
            flight_assist_geometry(&Transform::from_translation(Vec3::new(10.0, 20.0, 30.0)));

        assert_vec3_approx_eq(
            geometry.body_left_end,
            geometry.origin + Vec3::X * BODY_AXIS_LENGTH_METERS,
        );
        assert_vec3_approx_eq(
            geometry.body_up_end,
            geometry.origin + Vec3::Y * BODY_AXIS_LENGTH_METERS,
        );
        assert_vec3_approx_eq(
            geometry.body_forward_end,
            geometry.origin + Vec3::Z * BODY_AXIS_LENGTH_METERS,
        );
        assert_vec3_approx_eq(
            geometry.aim_end,
            geometry.aim_origin + Vec3::Z * AIM_RAY_LENGTH_METERS,
        );
    }

    #[test]
    fn rotated_pose_keeps_aim_ray_on_body_forward_axis() {
        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let geometry = flight_assist_geometry(&Transform::from_rotation(rotation));

        assert_vec3_approx_eq(
            geometry.aim_end,
            geometry.aim_origin + Vec3::X * AIM_RAY_LENGTH_METERS,
        );
    }
}
