pub mod commands;
pub mod environment;
pub mod events;
#[cfg(feature = "python-bindings")]
pub mod python;
pub mod snapshot;
pub mod types;
pub mod vision;

use bevy::prelude::*;

use crate::app::schedules::SimulationSet;

pub struct ApiPlugin;

impl Plugin for ApiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(events::EnvironmentEventPlugin)
            .init_resource::<commands::ExternalCommandBuffer>()
            .init_resource::<environment::EnvironmentFacade>()
            .init_resource::<snapshot::WorldSnapshot>()
            .init_resource::<types::ObservationBundle>()
            .init_resource::<types::ObservationCaptureConfig>()
            .init_resource::<vision::VisualCaptureBackend>()
            .init_resource::<vision::VisualCaptureGeneration>()
            .init_resource::<vision::VisualCaptureFrames>()
            .init_resource::<vision::SyntheticCaptureSessions>()
            .init_resource::<vision::PendingVisualCaptures>()
            .init_resource::<vision::OffscreenVisualCaptureBridge>()
            .add_observer(vision::store_visual_capture_screenshot)
            .add_systems(
                FixedUpdate,
                (
                    commands::apply_external_commands.in_set(SimulationSet::GatherInput),
                    (
                        snapshot::capture_world_snapshot,
                        environment::sync_environment_facade,
                    )
                        .chain()
                        .in_set(SimulationSet::ProduceSnapshot),
                ),
            )
            .add_systems(
                Update,
                (
                    vision::ensure_capture_cameras,
                    vision::sync_capture_cameras,
                    vision::attach_hud_to_capture_camera,
                    vision::ensure_offscreen_image_copiers,
                    vision::request_visual_captures,
                    vision::drain_offscreen_visual_frames,
                )
                    .chain(),
            );
        vision::install_offscreen_visual_capture(app);
    }
}
