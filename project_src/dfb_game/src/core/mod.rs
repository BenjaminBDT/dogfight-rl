pub mod config;
pub mod math;
pub mod time;

use bevy::prelude::*;

use crate::core::config::{ConfigPlugin, RepositoryConfig};
use crate::core::time::SimulationTimePlugin;

pub struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RepositoryConfig>()
            .add_plugins((ConfigPlugin, SimulationTimePlugin));
    }
}
