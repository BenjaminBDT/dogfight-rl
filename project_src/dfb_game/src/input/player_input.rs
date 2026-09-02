use bevy::input::ButtonInput;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::bridge::protocol::BridgeControlSlot;
use crate::bridge::{AssignedControlRole, LocalPilotMode};
use crate::core::config::{InputConfig, MouseAxisConfig, MouseFlightAxisTarget, RepositoryConfig};
use crate::input::actions::{
    ControlBindings, ControlInput, ControlIntent, MouseControlState, MouseFlightInput,
    MouseSmoothingState, NormalizedControlCommand,
};
use crate::simulation::components::{AircraftRole, ControlAuthority};

pub fn initialize_mouse_capture(
    config: Res<RepositoryConfig>,
    cursor_options: Option<Single<&mut CursorOptions, With<PrimaryWindow>>>,
    mut mouse_control_state: ResMut<MouseControlState>,
) {
    let Some(mut cursor_options) = cursor_options else {
        return;
    };
    set_mouse_capture(
        config.input.capture_mouse_on_start,
        &mut cursor_options,
        &mut mouse_control_state,
    );
}

pub fn toggle_mouse_capture(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    cursor_options: Option<Single<&mut CursorOptions, With<PrimaryWindow>>>,
    mut mouse_control_state: ResMut<MouseControlState>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings), Some(mut cursor_options)) =
        (keyboard, mouse_buttons, bindings, cursor_options)
    else {
        return;
    };
    if !bindings
        .toggle_mouse_capture
        .just_pressed(&keyboard, &mouse_buttons)
    {
        return;
    }

    let next_captured = !mouse_control_state.captured;
    set_mouse_capture(next_captured, &mut cursor_options, &mut mouse_control_state);
}

pub fn toggle_local_pilot_mode(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    controlled_role: Option<Res<AssignedControlRole>>,
    mut local_pilot_mode: ResMut<LocalPilotMode>,
    mut query: Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings)) = (keyboard, mouse_buttons, bindings)
    else {
        return;
    };
    if !bindings
        .toggle_local_pilot_mode
        .just_pressed(&keyboard, &mouse_buttons)
    {
        return;
    }

    *local_pilot_mode = match *local_pilot_mode {
        LocalPilotMode::Human => LocalPilotMode::FollowAi,
        LocalPilotMode::FollowAi => LocalPilotMode::ImperfectFollowAi,
        LocalPilotMode::ImperfectFollowAi => LocalPilotMode::TeacherFollowAi,
        LocalPilotMode::TeacherFollowAi => LocalPilotMode::Model,
        LocalPilotMode::Model => LocalPilotMode::Human,
    };

    let controlled_role = resolved_controlled_role(controlled_role.as_deref());
    apply_local_pilot_mode(controlled_role, *local_pilot_mode, &mut query);
    info!("Switched local pilot mode to {:?}", *local_pilot_mode);
}

pub fn sample_mouse_flight_input(
    time: Res<Time>,
    accumulated_mouse_motion: Option<Res<AccumulatedMouseMotion>>,
    config: Res<RepositoryConfig>,
    mouse_control_state: Res<MouseControlState>,
    mut smoothing_state: ResMut<MouseSmoothingState>,
    mut mouse_input: ResMut<MouseFlightInput>,
) {
    let input_config = &config.input;
    let mouse_delta = accumulated_mouse_motion
        .as_ref()
        .map(|motion| motion.delta)
        .unwrap_or(Vec2::ZERO);
    let (target_pitch, target_roll, target_yaw) = if mouse_control_state.captured {
        let mut pitch = 0.0;
        let mut roll = 0.0;
        let mut yaw = 0.0;
        apply_mouse_axis(
            mouse_delta.x,
            &input_config.mouse_x_axis,
            &mut pitch,
            &mut roll,
            &mut yaw,
        );
        apply_mouse_axis(
            mouse_delta.y,
            &input_config.mouse_y_axis,
            &mut pitch,
            &mut roll,
            &mut yaw,
        );
        (
            pitch.clamp(-1.0, 1.0),
            roll.clamp(-1.0, 1.0),
            yaw.clamp(-1.0, 1.0),
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    let smoothing_alpha = (input_config.mouse_smoothing * time.delta_secs()).clamp(0.0, 1.0);
    smoothing_state.pitch += (target_pitch - smoothing_state.pitch) * smoothing_alpha;
    smoothing_state.roll += (target_roll - smoothing_state.roll) * smoothing_alpha;
    smoothing_state.yaw += (target_yaw - smoothing_state.yaw) * smoothing_alpha;

    mouse_input.pitch = smoothing_state.pitch;
    mouse_input.roll = smoothing_state.roll;
    mouse_input.yaw = smoothing_state.yaw;
}

pub fn update_player_controls(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    mouse_input: Res<MouseFlightInput>,
    config: Res<RepositoryConfig>,
    controlled_role: Option<Res<AssignedControlRole>>,
    mut query: Query<(&AircraftRole, &ControlAuthority, &mut ControlInput)>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings)) = (keyboard, mouse_buttons, bindings)
    else {
        return;
    };
    let controlled_role = resolved_controlled_role(controlled_role.as_deref());
    let Some(controlled_role) = controlled_role else {
        return;
    };
    let input_config = &config.input;
    for (role, authority, mut input) in &mut query {
        if *role != controlled_role || *authority != ControlAuthority::Human {
            continue;
        }

        let intent = ControlIntent {
            throttle: axis(
                bindings.throttle_up.pressed(&keyboard, &mouse_buttons),
                bindings.throttle_down.pressed(&keyboard, &mouse_buttons),
            ) as i8,
            brake: bindings.brake.pressed(&keyboard, &mouse_buttons),
            pitch: axis(
                bindings.pitch_positive.pressed(&keyboard, &mouse_buttons),
                bindings.pitch_negative.pressed(&keyboard, &mouse_buttons),
            ) as i8,
            roll: axis(
                bindings.roll_positive.pressed(&keyboard, &mouse_buttons),
                bindings.roll_negative.pressed(&keyboard, &mouse_buttons),
            ) as i8,
            yaw: axis(
                bindings.yaw_positive.pressed(&keyboard, &mouse_buttons),
                bindings.yaw_negative.pressed(&keyboard, &mouse_buttons),
            ) as i8,
            fire_gun: bindings.fire_gun.pressed(&keyboard, &mouse_buttons),
            repair: bindings.repair_aircraft.pressed(&keyboard, &mouse_buttons),
        };
        let command = compose_human_control_command(intent, *mouse_input, input_config);
        *input = ControlInput::from(command);
    }
}

fn resolved_controlled_role(controlled_role: Option<&AssignedControlRole>) -> Option<AircraftRole> {
    controlled_role
        .map(|role| match role.0 {
            BridgeControlSlot::Fighter1 => Some(AircraftRole::Fighter1),
            BridgeControlSlot::Fighter2 => Some(AircraftRole::Fighter2),
            BridgeControlSlot::Spectator => None,
        })
        .unwrap_or(Some(AircraftRole::Fighter1))
}

fn apply_local_pilot_mode(
    controlled_role: Option<AircraftRole>,
    local_pilot_mode: LocalPilotMode,
    query: &mut Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    for (role, mut authority) in query.iter_mut() {
        *authority = if Some(*role) == controlled_role {
            match local_pilot_mode {
                LocalPilotMode::Human => ControlAuthority::Human,
                LocalPilotMode::FollowAi
                | LocalPilotMode::ImperfectFollowAi
                | LocalPilotMode::TeacherFollowAi
                | LocalPilotMode::Model => ControlAuthority::ExternalAgent,
            }
        } else {
            ControlAuthority::BuiltInAi
        };
    }
}

fn compose_human_control_command(
    intent: ControlIntent,
    mouse_input: MouseFlightInput,
    input_config: &InputConfig,
) -> NormalizedControlCommand {
    let intent = intent.clamped();
    NormalizedControlCommand {
        throttle: (intent.throttle as f32) * input_config.keyboard_throttle_weight,
        brake: intent.brake,
        pitch: (intent.pitch as f32) * input_config.keyboard_pitch_weight + mouse_input.pitch,
        roll: (intent.roll as f32) * input_config.keyboard_roll_weight + mouse_input.roll,
        yaw: (intent.yaw as f32) + mouse_input.yaw,
        fire_gun: intent.fire_gun,
        repair: intent.repair,
    }
    .clamped()
}

fn apply_mouse_axis(
    delta: f32,
    axis_config: &MouseAxisConfig,
    pitch: &mut f32,
    roll: &mut f32,
    yaw: &mut f32,
) {
    let sign = if axis_config.invert { -1.0 } else { 1.0 };
    let value = delta * axis_config.sensitivity * axis_config.weight * sign;
    match axis_config.target {
        MouseFlightAxisTarget::Pitch => *pitch += value,
        MouseFlightAxisTarget::Roll => *roll += value,
        MouseFlightAxisTarget::Yaw => *yaw += value,
    }
}

fn axis(positive: bool, negative: bool) -> f32 {
    match (positive, negative) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    }
}

fn set_mouse_capture(
    captured: bool,
    cursor_options: &mut CursorOptions,
    mouse_control_state: &mut MouseControlState,
) {
    cursor_options.visible = !captured;
    cursor_options.grab_mode = if captured {
        CursorGrabMode::Locked
    } else {
        CursorGrabMode::None
    };
    mouse_control_state.captured = captured;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_human_control_command_applies_keyboard_weights() {
        let input_config = InputConfig {
            keyboard_throttle_weight: 0.5,
            keyboard_pitch_weight: 0.25,
            keyboard_roll_weight: 0.75,
            ..InputConfig::default()
        };

        let command = compose_human_control_command(
            ControlIntent {
                throttle: 1,
                brake: true,
                pitch: -1,
                roll: 1,
                yaw: -1,
                fire_gun: true,
                repair: false,
            },
            MouseFlightInput::default(),
            &input_config,
        );

        assert_eq!(command.throttle, 0.5);
        assert_eq!(command.pitch, -0.25);
        assert_eq!(command.roll, 0.75);
        assert_eq!(command.yaw, -1.0);
        assert!(command.brake);
        assert!(command.fire_gun);
    }

    #[test]
    fn compose_human_control_command_mixes_mouse_and_keyboard_then_clamps() {
        let input_config = InputConfig {
            keyboard_throttle_weight: 1.0,
            keyboard_pitch_weight: 0.6,
            keyboard_roll_weight: 0.5,
            ..InputConfig::default()
        };

        let command = compose_human_control_command(
            ControlIntent {
                throttle: 1,
                brake: false,
                pitch: 1,
                roll: -1,
                yaw: 1,
                fire_gun: false,
                repair: true,
            },
            MouseFlightInput {
                pitch: 0.8,
                roll: -0.75,
                yaw: 0.5,
            },
            &input_config,
        );

        assert_eq!(command.throttle, 1.0);
        assert_eq!(command.pitch, 1.0);
        assert_eq!(command.roll, -1.0);
        assert_eq!(command.yaw, 1.0);
        assert!(command.repair);
    }

    #[test]
    fn resolved_controlled_role_maps_bridge_slots() {
        assert_eq!(
            resolved_controlled_role(Some(&AssignedControlRole(BridgeControlSlot::Fighter1))),
            Some(AircraftRole::Fighter1)
        );
        assert_eq!(
            resolved_controlled_role(Some(&AssignedControlRole(BridgeControlSlot::Fighter2))),
            Some(AircraftRole::Fighter2)
        );
        assert_eq!(
            resolved_controlled_role(Some(&AssignedControlRole(BridgeControlSlot::Spectator))),
            None
        );
        assert_eq!(resolved_controlled_role(None), Some(AircraftRole::Fighter1));
    }
}
