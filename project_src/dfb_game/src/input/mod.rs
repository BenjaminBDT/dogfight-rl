pub mod actions;
pub mod control_adapter;
pub mod player_input;

use bevy::prelude::*;

use crate::app::schedules::SimulationSet;
use crate::core::config::RepositoryConfig;
use crate::input::actions::{
    ControlBindings, MouseControlState, MouseFlightInput, MouseSmoothingState,
};

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ControlBindings>()
            .init_resource::<MouseFlightInput>()
            .init_resource::<MouseControlState>()
            .init_resource::<MouseSmoothingState>()
            .add_systems(Startup, load_control_bindings)
            .add_systems(PostStartup, player_input::initialize_mouse_capture)
            .add_systems(Update, player_input::toggle_mouse_capture)
            .add_systems(Update, player_input::toggle_local_pilot_mode)
            .add_systems(Update, player_input::sample_mouse_flight_input)
            .add_systems(
                FixedUpdate,
                player_input::update_player_controls.in_set(SimulationSet::GatherInput),
            );
    }
}

fn load_control_bindings(mut bindings: ResMut<ControlBindings>, config: Res<RepositoryConfig>) {
    *bindings = ControlBindings::from_config(&config.input.bindings);
    info!("Loaded input bindings from config/dfb_game/input.ron");
}
