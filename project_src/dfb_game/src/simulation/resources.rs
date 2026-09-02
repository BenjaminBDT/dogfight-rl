use bevy::prelude::*;

#[derive(Debug, Default, Resource)]
pub struct SimulationDebugState {
    pub tick_count: u64,
}
