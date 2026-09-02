use std::time::Duration;

use bevy::app::ScheduleRunnerPlugin;
use bevy::asset::AssetPlugin;
use bevy::audio::AudioPlugin;
use bevy::ecs::error::{DefaultErrorHandler, error as bevy_error_log};
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::{
    RenderPlugin,
    settings::{InstanceFlags, RenderCreation, WgpuSettings},
};
use bevy::state::app::StatesPlugin;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use crate::ai::BasicFighterAiPlugin;
use crate::api::ApiPlugin;
use crate::api::environment::{DEFAULT_ENVIRONMENT_SEED, DeterministicRng, EnvironmentSeed};
use crate::api::vision::VisualCaptureBackend;
use crate::app::{AppMode, HeadlessMode, ObserverClientMode, RenderEnabled};
use crate::audio::{AudioObservationPlugin, CpalAudioPlugin};
use crate::bridge::protocol::BridgeRole;
use crate::bridge::{BridgeEnabled, BridgePlugin};
use crate::core::CorePlugin;
use crate::core::config::ConfigPaths;
use crate::core::time::headless_time_strategy;
use crate::gameplay::server_admin::{
    self, ServerAdminInterfaceMode, ServerAdminTimeControl, ServerAdminTimeMode,
};
use crate::gameplay::{GameplayPlugin, SharedGameplayStatePlugin};
use crate::input::InputPlugin;
use crate::model_control::ModelControlPlugin;
use crate::presentation::PresentationPlugin;
use crate::recording::ActionRecordingPlugin;
use crate::simulation::SimulationPlugin;
use crate::telemetry::TelemetryPlugin;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerTimeMode {
    Realtime,
    Simulated,
}

fn shared_asset_plugin(config_paths: &ConfigPaths) -> AssetPlugin {
    AssetPlugin {
        file_path: config_paths
            .project_root
            .join("assets")
            .display()
            .to_string(),
        ..default()
    }
}

pub fn build_game_app() -> App {
    build_game_app_with_paths(ConfigPaths::default())
}

pub fn build_game_app_with_paths(config_paths: ConfigPaths) -> App {
    build_client_app_with_paths(config_paths)
}

pub fn build_client_app_with_paths(config_paths: ConfigPaths) -> App {
    let mut app = App::new();
    let asset_plugin = shared_asset_plugin(&config_paths);
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(false))
        .insert_resource(RenderEnabled(true))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Client))
        .insert_resource(EnvironmentSeed::from_request(Some(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .add_plugins(
            DefaultPlugins
                .build()
                .set(asset_plugin)
                .disable::<AudioPlugin>(),
        )
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            SharedGameplayStatePlugin,
            GameplayPlugin,
            InputPlugin,
            ModelControlPlugin,
            BasicFighterAiPlugin,
            ActionRecordingPlugin,
            PresentationPlugin,
            AudioObservationPlugin,
            CpalAudioPlugin,
            ApiPlugin,
            BridgePlugin,
            TelemetryPlugin,
        ));
    app
}

pub fn build_observer_app_with_paths(config_paths: ConfigPaths) -> App {
    build_observer_app_with_mode(config_paths, ObserverClientMode::RecordedEpisode)
}

pub fn build_observer_app_with_mode(
    config_paths: ConfigPaths,
    observer_mode: ObserverClientMode,
) -> App {
    let mut app = App::new();
    let bridge_enabled = observer_mode == ObserverClientMode::LiveServer;
    let asset_plugin = shared_asset_plugin(&config_paths);
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Observer)
        .insert_resource(observer_mode)
        .insert_resource(BridgeEnabled(bridge_enabled))
        .insert_resource(HeadlessMode(false))
        .insert_resource(RenderEnabled(true))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Client))
        .insert_resource(EnvironmentSeed::from_request(Some(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .add_plugins(
            DefaultPlugins
                .build()
                .set(asset_plugin)
                .disable::<AudioPlugin>(),
        )
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            InputPlugin,
            PresentationPlugin,
            AudioObservationPlugin,
            CpalAudioPlugin,
            ApiPlugin,
            BridgePlugin,
            TelemetryPlugin,
            SharedGameplayStatePlugin,
        ));
    app
}

pub fn build_server_app_with_paths(config_paths: ConfigPaths, time_mode: ServerTimeMode) -> App {
    build_server_app_with_paths_and_admin(config_paths, time_mode, ServerAdminInterfaceMode::Plain)
}

pub fn build_server_app_with_paths_and_admin(
    config_paths: ConfigPaths,
    time_mode: ServerTimeMode,
    admin_interface: ServerAdminInterfaceMode,
) -> App {
    let mut app = App::new();
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(true))
        .insert_resource(RenderEnabled(false))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Server))
        .insert_resource(admin_interface)
        .insert_resource(ServerAdminTimeControl {
            current_mode: match time_mode {
                ServerTimeMode::Realtime => ServerAdminTimeMode::Realtime,
                ServerTimeMode::Simulated => ServerAdminTimeMode::Simulated,
            },
            requested_mode: match time_mode {
                ServerTimeMode::Realtime => ServerAdminTimeMode::Realtime,
                ServerTimeMode::Simulated => ServerAdminTimeMode::Simulated,
            },
        })
        .insert_resource(EnvironmentSeed::from_request(Some(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(headless_time_strategy());
    match time_mode {
        ServerTimeMode::Realtime => {
            app.add_plugins((
                MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                    1.0 / 60.0,
                ))),
                server_log_plugin(admin_interface),
                StatesPlugin,
            ));
        }
        ServerTimeMode::Simulated => {
            app.add_plugins((
                MinimalPlugins,
                server_log_plugin(admin_interface),
                StatesPlugin,
            ));
        }
    }
    app.add_plugins((
        CorePlugin,
        SimulationPlugin,
        SharedGameplayStatePlugin,
        GameplayPlugin,
        BasicFighterAiPlugin,
        ActionRecordingPlugin,
        AudioObservationPlugin,
        ApiPlugin,
        BridgePlugin,
        TelemetryPlugin,
    ));
    app
}

fn server_log_plugin(admin_interface: ServerAdminInterfaceMode) -> LogPlugin {
    match admin_interface {
        ServerAdminInterfaceMode::Tui => LogPlugin {
            custom_layer: server_admin::create_log_capture_layer,
            fmt_layer: server_admin::create_silent_fmt_layer,
            ..default()
        },
        _ => LogPlugin::default(),
    }
}

pub fn build_headless_app_with_paths(config_paths: ConfigPaths) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(StatesPlugin);
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(true))
        .insert_resource(RenderEnabled(false))
        .insert_resource(BridgeEnabled(false))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Server))
        .insert_resource(server_admin::ServerAdminInterfaceMode::Off)
        .insert_resource(EnvironmentSeed::from_request(Some(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(
            DEFAULT_ENVIRONMENT_SEED,
        )))
        .insert_resource(headless_time_strategy())
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            SharedGameplayStatePlugin,
            GameplayPlugin,
            BasicFighterAiPlugin,
            ActionRecordingPlugin,
            AudioObservationPlugin,
            ApiPlugin,
            BridgePlugin,
            TelemetryPlugin,
        ));
    app
}

pub fn build_headless_audio_reconstruction_app_with_paths(config_paths: ConfigPaths) -> App {
    let mut app = App::new();
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(true))
        .insert_resource(RenderEnabled(false))
        .insert_resource(BridgeEnabled(false))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Server))
        .insert_resource(server_admin::ServerAdminInterfaceMode::Off)
        .insert_resource(headless_time_strategy())
        .add_plugins(MinimalPlugins)
        .add_plugins(StatesPlugin)
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            SharedGameplayStatePlugin,
            GameplayPlugin,
            AudioObservationPlugin,
            ApiPlugin,
        ));
    app
}

pub fn build_headless_capture_app_with_paths(config_paths: ConfigPaths, _include_hud: bool) -> App {
    let mut app = App::new();
    let asset_plugin = shared_asset_plugin(&config_paths);
    let default_plugins = DefaultPlugins
        .build()
        .set(asset_plugin)
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                instance_flags: InstanceFlags::empty(),
                ..default()
            }),
            ..default()
        })
        .set(WindowPlugin {
            primary_window: Some(Window {
                visible: false,
                title: "dogfight_capture".to_string(),
                ..default()
            }),
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .disable::<LogPlugin>()
        .disable::<AudioPlugin>();
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(true))
        .insert_resource(RenderEnabled(true))
        .insert_resource(BridgeEnabled(false))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Server))
        .insert_resource(server_admin::ServerAdminInterfaceMode::Off)
        .insert_resource(DefaultErrorHandler(bevy_error_log))
        .insert_resource(headless_time_strategy())
        .add_plugins(default_plugins)
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            SharedGameplayStatePlugin,
            GameplayPlugin,
            InputPlugin,
            BasicFighterAiPlugin,
            ActionRecordingPlugin,
            PresentationPlugin,
            AudioObservationPlugin,
            ApiPlugin,
            BridgePlugin,
            TelemetryPlugin,
        ));
    app
}

pub fn build_headless_offscreen_capture_app_with_paths(
    config_paths: ConfigPaths,
    _include_hud: bool,
) -> App {
    let mut app = App::new();
    let asset_plugin = shared_asset_plugin(&config_paths);
    let default_plugins = DefaultPlugins
        .build()
        .set(asset_plugin)
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                instance_flags: InstanceFlags::empty(),
                ..default()
            }),
            ..default()
        })
        .set(WindowPlugin {
            primary_window: None,
            exit_condition: ExitCondition::DontExit,
            ..default()
        })
        .disable::<LogPlugin>()
        .disable::<WinitPlugin>()
        .disable::<AudioPlugin>();
    app.insert_resource(config_paths)
        .insert_resource(AppMode::Game)
        .insert_resource(HeadlessMode(true))
        .insert_resource(RenderEnabled(true))
        .insert_resource(BridgeEnabled(false))
        .insert_resource(crate::bridge::BridgeMode(BridgeRole::Server))
        .insert_resource(server_admin::ServerAdminInterfaceMode::Off)
        .insert_resource(DefaultErrorHandler(bevy_error_log))
        .insert_resource(headless_time_strategy())
        .insert_resource(VisualCaptureBackend::Offscreen)
        .add_plugins(default_plugins)
        .add_plugins((
            CorePlugin,
            SimulationPlugin,
            SharedGameplayStatePlugin,
            GameplayPlugin,
            BasicFighterAiPlugin,
            ActionRecordingPlugin,
            PresentationPlugin,
            AudioObservationPlugin,
            ApiPlugin,
            BridgePlugin,
            TelemetryPlugin,
        ));
    app
}

pub fn run_game_app() {
    build_client_app_with_paths(ConfigPaths::default()).run();
}

pub fn run_game_app_with_paths(config_paths: ConfigPaths) {
    build_client_app_with_paths(config_paths).run();
}

pub fn run_server_app_with_paths(config_paths: ConfigPaths) {
    build_server_app_with_paths(config_paths, ServerTimeMode::Realtime).run();
}
