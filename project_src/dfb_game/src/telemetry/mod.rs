pub mod logging;
pub mod replay;

use bevy::prelude::*;

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, _app: &mut App) {}
}
