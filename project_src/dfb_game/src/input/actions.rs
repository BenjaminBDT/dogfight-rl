use bevy::prelude::{Component, KeyCode, MouseButton, Resource, warn};
use serde::{Deserialize, Serialize};

use crate::core::config::{ActionBindingsConfig, InputBindingConfig, InputBindingsConfig};

fn clamp_unit_step(value: i8) -> i8 {
    value.clamp(-1, 1)
}

fn clamp_command_axis(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ControlIntent {
    pub throttle: i8,
    #[serde(default)]
    pub brake: bool,
    pub pitch: i8,
    pub roll: i8,
    pub yaw: i8,
    pub fire_gun: bool,
    pub repair: bool,
}

impl ControlIntent {
    pub fn clamped(self) -> Self {
        Self {
            throttle: clamp_unit_step(self.throttle),
            brake: self.brake,
            pitch: clamp_unit_step(self.pitch),
            roll: clamp_unit_step(self.roll),
            yaw: clamp_unit_step(self.yaw),
            fire_gun: self.fire_gun,
            repair: self.repair,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct NormalizedControlCommand {
    pub throttle: f32,
    #[serde(default)]
    pub brake: bool,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub fire_gun: bool,
    pub repair: bool,
}

impl NormalizedControlCommand {
    pub fn clamped(self) -> Self {
        Self {
            throttle: clamp_command_axis(self.throttle),
            brake: self.brake,
            pitch: clamp_command_axis(self.pitch),
            roll: clamp_command_axis(self.roll),
            yaw: clamp_command_axis(self.yaw),
            fire_gun: self.fire_gun,
            repair: self.repair,
        }
    }
}

impl From<ControlIntent> for NormalizedControlCommand {
    fn from(value: ControlIntent) -> Self {
        let value = value.clamped();
        Self {
            throttle: value.throttle as f32,
            brake: value.brake,
            pitch: value.pitch as f32,
            roll: value.roll as f32,
            yaw: value.yaw as f32,
            fire_gun: value.fire_gun,
            repair: value.repair,
        }
    }
}

#[derive(Component, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ControlInput {
    pub throttle_delta: f32,
    #[serde(default)]
    pub brake: bool,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub fire_gun: bool,
    pub repair: bool,
}

impl Default for ControlInput {
    fn default() -> Self {
        Self {
            throttle_delta: 0.0,
            brake: false,
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            fire_gun: false,
            repair: false,
        }
    }
}

impl From<NormalizedControlCommand> for ControlInput {
    fn from(value: NormalizedControlCommand) -> Self {
        let value = value.clamped();
        Self {
            throttle_delta: value.throttle,
            brake: value.brake,
            pitch: value.pitch,
            roll: value.roll,
            yaw: value.yaw,
            fire_gun: value.fire_gun,
            repair: value.repair,
        }
    }
}

impl From<ControlInput> for NormalizedControlCommand {
    fn from(value: ControlInput) -> Self {
        Self {
            throttle: value.throttle_delta,
            brake: value.brake,
            pitch: value.pitch,
            roll: value.roll,
            yaw: value.yaw,
            fire_gun: value.fire_gun,
            repair: value.repair,
        }
        .clamped()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InputBinding {
    Keyboard(KeyCode),
    Mouse(MouseButton),
}

impl InputBinding {
    pub fn pressed(
        &self,
        keyboard: &bevy::input::ButtonInput<KeyCode>,
        mouse: &bevy::input::ButtonInput<MouseButton>,
    ) -> bool {
        match self {
            Self::Keyboard(key) => keyboard.pressed(*key),
            Self::Mouse(button) => mouse.pressed(*button),
        }
    }

    pub fn just_pressed(
        &self,
        keyboard: &bevy::input::ButtonInput<KeyCode>,
        mouse: &bevy::input::ButtonInput<MouseButton>,
    ) -> bool {
        match self {
            Self::Keyboard(key) => keyboard.just_pressed(*key),
            Self::Mouse(button) => mouse.just_pressed(*button),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ActionBindings {
    pub bindings: Vec<InputBinding>,
}

impl ActionBindings {
    pub fn pressed(
        &self,
        keyboard: &bevy::input::ButtonInput<KeyCode>,
        mouse: &bevy::input::ButtonInput<MouseButton>,
    ) -> bool {
        self.bindings
            .iter()
            .any(|binding| binding.pressed(keyboard, mouse))
    }

    pub fn just_pressed(
        &self,
        keyboard: &bevy::input::ButtonInput<KeyCode>,
        mouse: &bevy::input::ButtonInput<MouseButton>,
    ) -> bool {
        self.bindings
            .iter()
            .any(|binding| binding.just_pressed(keyboard, mouse))
    }
}

#[derive(Debug, Clone, Resource)]
pub struct ControlBindings {
    pub throttle_up: ActionBindings,
    pub throttle_down: ActionBindings,
    pub brake: ActionBindings,
    pub pitch_positive: ActionBindings,
    pub pitch_negative: ActionBindings,
    pub roll_positive: ActionBindings,
    pub roll_negative: ActionBindings,
    pub yaw_positive: ActionBindings,
    pub yaw_negative: ActionBindings,
    pub fire_gun: ActionBindings,
    pub repair_aircraft: ActionBindings,
    pub toggle_controls_guide: ActionBindings,
    pub reset_match: ActionBindings,
    pub rear_view: ActionBindings,
    pub toggle_local_pilot_mode: ActionBindings,
    pub toggle_audio_mute: ActionBindings,
    pub toggle_mouse_capture: ActionBindings,
}

impl Default for ControlBindings {
    fn default() -> Self {
        Self::from_config(&InputBindingsConfig::default())
    }
}

impl ControlBindings {
    pub fn from_config(config: &InputBindingsConfig) -> Self {
        Self {
            throttle_up: parse_action_bindings(&config.throttle_up),
            throttle_down: parse_action_bindings(&config.throttle_down),
            brake: parse_action_bindings(&config.brake),
            pitch_positive: parse_action_bindings(&config.pitch_positive),
            pitch_negative: parse_action_bindings(&config.pitch_negative),
            roll_positive: parse_action_bindings(&config.roll_positive),
            roll_negative: parse_action_bindings(&config.roll_negative),
            yaw_positive: parse_action_bindings(&config.yaw_positive),
            yaw_negative: parse_action_bindings(&config.yaw_negative),
            fire_gun: parse_action_bindings(&config.fire_gun),
            repair_aircraft: parse_action_bindings(&config.repair_aircraft),
            toggle_controls_guide: parse_action_bindings(&config.toggle_controls_guide),
            reset_match: parse_action_bindings(&config.reset_match),
            rear_view: parse_action_bindings(&config.rear_view),
            toggle_local_pilot_mode: parse_action_bindings(&config.toggle_local_pilot_mode),
            toggle_audio_mute: parse_action_bindings(&config.toggle_audio_mute),
            toggle_mouse_capture: parse_action_bindings(&config.toggle_mouse_capture),
        }
    }
}

fn parse_action_bindings(config: &ActionBindingsConfig) -> ActionBindings {
    let mut bindings = Vec::new();

    if let Some(name) = &config.keyboard_primary {
        bindings.push(parse_input_binding(&InputBindingConfig::Keyboard(
            name.clone(),
        )));
    }
    if let Some(name) = &config.keyboard_secondary {
        bindings.push(parse_input_binding(&InputBindingConfig::Keyboard(
            name.clone(),
        )));
    }
    if let Some(name) = &config.mouse_primary {
        bindings.push(parse_input_binding(&InputBindingConfig::Mouse(
            name.clone(),
        )));
    }
    if let Some(name) = &config.mouse_secondary {
        bindings.push(parse_input_binding(&InputBindingConfig::Mouse(
            name.clone(),
        )));
    }

    ActionBindings { bindings }
}

fn parse_input_binding(config: &InputBindingConfig) -> InputBinding {
    match config {
        InputBindingConfig::Keyboard(name) => InputBinding::Keyboard(parse_key_code(name)),
        InputBindingConfig::Mouse(name) => InputBinding::Mouse(parse_mouse_button(name)),
    }
}

fn parse_key_code(name: &str) -> KeyCode {
    match name {
        "ShiftLeft" => KeyCode::ShiftLeft,
        "ShiftRight" => KeyCode::ShiftRight,
        "ControlLeft" => KeyCode::ControlLeft,
        "ControlRight" => KeyCode::ControlRight,
        "AltLeft" => KeyCode::AltLeft,
        "AltRight" => KeyCode::AltRight,
        "Space" => KeyCode::Space,
        "Tab" => KeyCode::Tab,
        "Enter" => KeyCode::Enter,
        "Backspace" => KeyCode::Backspace,
        "Escape" => KeyCode::Escape,
        "KeyA" => KeyCode::KeyA,
        "KeyB" => KeyCode::KeyB,
        "KeyC" => KeyCode::KeyC,
        "KeyD" => KeyCode::KeyD,
        "KeyE" => KeyCode::KeyE,
        "KeyF" => KeyCode::KeyF,
        "KeyG" => KeyCode::KeyG,
        "KeyH" => KeyCode::KeyH,
        "KeyI" => KeyCode::KeyI,
        "KeyJ" => KeyCode::KeyJ,
        "KeyK" => KeyCode::KeyK,
        "KeyL" => KeyCode::KeyL,
        "KeyM" => KeyCode::KeyM,
        "KeyN" => KeyCode::KeyN,
        "KeyO" => KeyCode::KeyO,
        "KeyP" => KeyCode::KeyP,
        "KeyQ" => KeyCode::KeyQ,
        "KeyR" => KeyCode::KeyR,
        "KeyS" => KeyCode::KeyS,
        "KeyT" => KeyCode::KeyT,
        "KeyU" => KeyCode::KeyU,
        "KeyV" => KeyCode::KeyV,
        "KeyW" => KeyCode::KeyW,
        "KeyX" => KeyCode::KeyX,
        "KeyY" => KeyCode::KeyY,
        "KeyZ" => KeyCode::KeyZ,
        "Digit0" => KeyCode::Digit0,
        "Digit1" => KeyCode::Digit1,
        "Digit2" => KeyCode::Digit2,
        "Digit3" => KeyCode::Digit3,
        "Digit4" => KeyCode::Digit4,
        "Digit5" => KeyCode::Digit5,
        "Digit6" => KeyCode::Digit6,
        "Digit7" => KeyCode::Digit7,
        "Digit8" => KeyCode::Digit8,
        "Digit9" => KeyCode::Digit9,
        "ArrowUp" => KeyCode::ArrowUp,
        "ArrowDown" => KeyCode::ArrowDown,
        "ArrowLeft" => KeyCode::ArrowLeft,
        "ArrowRight" => KeyCode::ArrowRight,
        "F1" => KeyCode::F1,
        "F2" => KeyCode::F2,
        "F3" => KeyCode::F3,
        "F4" => KeyCode::F4,
        "F5" => KeyCode::F5,
        "F6" => KeyCode::F6,
        "F7" => KeyCode::F7,
        "F8" => KeyCode::F8,
        "F9" => KeyCode::F9,
        "F10" => KeyCode::F10,
        "F11" => KeyCode::F11,
        "F12" => KeyCode::F12,
        other => {
            warn!(
                "Unknown key binding {:?}; falling back to Space. Check config/dfb_game/input.ron",
                other
            );
            KeyCode::Space
        }
    }
}

fn parse_mouse_button(name: &str) -> MouseButton {
    match name {
        "Left" => MouseButton::Left,
        "Right" => MouseButton::Right,
        "Middle" => MouseButton::Middle,
        "Back" => MouseButton::Back,
        "Forward" => MouseButton::Forward,
        other => {
            warn!(
                "Unknown mouse binding {:?}; falling back to Left. Check config/dfb_game/input.ron",
                other
            );
            MouseButton::Left
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Resource)]
pub struct MouseFlightInput {
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
}

#[derive(Debug, Clone, Copy, Default, Resource)]
pub struct MouseControlState {
    pub captured: bool,
}

#[derive(Debug, Clone, Copy, Default, Resource)]
pub struct MouseSmoothingState {
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_intent_clamps_all_axes_to_unit_steps() {
        let intent = ControlIntent {
            throttle: 3,
            brake: true,
            pitch: -5,
            roll: 2,
            yaw: -9,
            fire_gun: true,
            repair: false,
        }
        .clamped();

        assert_eq!(intent.throttle, 1);
        assert_eq!(intent.pitch, -1);
        assert_eq!(intent.roll, 1);
        assert_eq!(intent.yaw, -1);
        assert!(intent.brake);
        assert!(intent.fire_gun);
    }

    #[test]
    fn normalized_control_command_clamps_axes() {
        let command = NormalizedControlCommand {
            throttle: 1.8,
            brake: true,
            pitch: -1.4,
            roll: 0.4,
            yaw: 9.0,
            fire_gun: false,
            repair: true,
        }
        .clamped();

        assert_eq!(command.throttle, 1.0);
        assert_eq!(command.pitch, -1.0);
        assert_eq!(command.roll, 0.4);
        assert_eq!(command.yaw, 1.0);
        assert!(command.brake);
        assert!(command.repair);
    }

    #[test]
    fn control_intent_maps_to_normalized_control_command() {
        let command = NormalizedControlCommand::from(ControlIntent {
            throttle: 1,
            brake: false,
            pitch: -1,
            roll: 0,
            yaw: 1,
            fire_gun: true,
            repair: false,
        });

        assert_eq!(command.throttle, 1.0);
        assert_eq!(command.pitch, -1.0);
        assert_eq!(command.roll, 0.0);
        assert_eq!(command.yaw, 1.0);
        assert!(command.fire_gun);
    }

    #[test]
    fn normalized_control_command_round_trips_through_control_input() {
        let command = NormalizedControlCommand {
            throttle: 0.75,
            brake: true,
            pitch: -0.25,
            roll: 0.5,
            yaw: -0.125,
            fire_gun: true,
            repair: false,
        };

        let input = ControlInput::from(command);
        let reconstructed = NormalizedControlCommand::from(input);

        assert_eq!(reconstructed, command);
    }
}
