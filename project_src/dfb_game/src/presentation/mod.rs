pub mod camera;
pub mod debug_draw;
pub mod hud;
pub mod hud_table;
pub mod scene;
pub mod tracers;

use bevy::prelude::*;

pub struct PresentationPlugin;
pub struct CaptureHudPlugin;

impl Plugin for PresentationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<hud::ControlsGuideState>()
            .init_resource::<debug_draw::FlightAssistOverlayState>()
            .init_resource::<hud::ObservedAircraftRole>()
            .init_resource::<hud::CompassReferenceState>()
            .init_resource::<hud::SpeedGaugeState>()
            .init_resource::<hud::UiFontHandles>()
            .init_resource::<hud::ObserverViewContext>()
            .init_resource::<hud::ObserverFeedStatus>()
            .init_resource::<hud::ObserverFeedProviderState>()
            .init_resource::<hud::DamageIndicatorQueue>()
            .init_resource::<hud::DamageIndicatorState>();
        app.add_systems(
            Startup,
            (
                hud::load_ui_fonts,
                scene::setup_scene,
                camera::spawn_camera,
                hud::spawn_hud,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                camera::toggle_fullscreen,
                hud::attach_hud_to_main_camera,
                hud::toggle_controls_guide,
                hud::sync_observer_view_context,
                hud::sync_live_observer_feed_status,
                hud::update_live_spectator_observed_role,
                hud::update_controls_guide_text,
                hud::update_controls_guide_visibility,
            ),
        )
        .add_systems(
            Update,
            (
                debug_draw::toggle_flight_assist_overlay,
                debug_draw::draw_flight_assist_overlay,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                hud::update_damage_indicators,
                hud::update_damage_indicator_hud,
                camera::sync_camera_projection,
                camera::update_follow_camera,
                hud::update_hud,
                tracers::drain_presentation_queue,
                tracers::update_tracer_lifetimes,
            ),
        );
    }
}

impl Plugin for CaptureHudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<hud::ControlsGuideState>()
            .init_resource::<hud::ObservedAircraftRole>()
            .init_resource::<hud::CompassReferenceState>()
            .init_resource::<hud::SpeedGaugeState>()
            .init_resource::<hud::UiFontHandles>()
            .init_resource::<hud::ObserverViewContext>()
            .init_resource::<hud::ObserverFeedStatus>()
            .init_resource::<hud::ObserverFeedProviderState>()
            .init_resource::<hud::DamageIndicatorQueue>()
            .init_resource::<hud::DamageIndicatorState>()
            .add_systems(Startup, (hud::load_ui_fonts, hud::spawn_hud).chain())
            .add_systems(
                Update,
                (
                    hud::toggle_controls_guide,
                    hud::sync_observer_view_context,
                    hud::sync_live_observer_feed_status,
                    hud::update_live_spectator_observed_role,
                    hud::update_controls_guide_text,
                    hud::update_controls_guide_visibility,
                ),
            )
            .add_systems(
                Update,
                (
                    hud::update_damage_indicators,
                    hud::update_damage_indicator_hud,
                    hud::update_hud,
                ),
            );
    }
}
