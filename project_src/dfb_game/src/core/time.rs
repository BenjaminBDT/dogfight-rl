use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

use crate::core::config::RepositoryConfig;

pub struct SimulationTimePlugin;

impl Plugin for SimulationTimePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, configure_fixed_timestep);
    }
}

fn configure_fixed_timestep(mut time: ResMut<Time<Fixed>>, config: Res<RepositoryConfig>) {
    time.set_timestep_seconds(config.game.fixed_time_step_seconds as f64);
}

#[allow(dead_code)]
pub fn headless_time_strategy() -> TimeUpdateStrategy {
    TimeUpdateStrategy::ManualDuration(std::time::Duration::from_secs_f64(1.0 / 60.0))
}
