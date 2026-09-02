use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::pbr::StandardMaterial;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport};
use bevy::render::view::NoIndirectDrawing;
use bevy::time::Virtual;
use dfb_game::api::commands::TargetedEnvironmentAction;
use dfb_game::api::environment::EnvironmentInstance;
use dfb_game::api::types::{EnvironmentResetOptions, StepResult, SubsystemObservation};
use dfb_game::app::ObserverClientMode;
use dfb_game::app::game_app::{build_observer_app_with_mode, build_observer_app_with_paths};
use dfb_game::app::schedules::SimulationSet;
use dfb_game::audio::{AudioEventQueue, ObserverAudioListenerOverride};
use dfb_game::bridge::ipc_stub::{BridgeEndpointConfig, BridgeTransport};
use dfb_game::bridge::protocol::{BridgeControlSlot, LobbyClientKind};
use dfb_game::bridge::{BridgeLaunchBinding, RequestedControlRole};
use dfb_game::core::config::{ConfigPaths, resolve_project_root};
use dfb_game::gameplay::combat::{
    Projectile, spawn_projectile_visual_entity, spawn_tracer_visual_entity,
};
use dfb_game::gameplay::damage::{AircraftDamageState, AircraftSubsystem};
use dfb_game::gameplay::match_state::{MatchClock, MatchPhase};
use dfb_game::input::actions::MouseControlState;
use dfb_game::input::actions::{ControlBindings, ControlInput};
use dfb_game::presentation::camera::{FollowPlayerCamera, MainViewCamera};
use dfb_game::presentation::hud::{
    CompassReferenceState, ObservedAircraftRole, ObserverFeedProviderState, ObserverFeedStatus,
    RecordedObserverFeedSample, ReplayProgressBarFill, ReplayProgressBarText,
    ReplayProgressHudText, UiFontHandles, apply_recorded_observer_feed_status,
    spawn_replay_progress_hud,
};
use dfb_game::presentation::hud_table::{Cell, render_rows_with_min_widths};
use dfb_game::presentation::tracers::TracerLifetime;
use dfb_game::recording::reconstruct::RecordingAccess;
use dfb_game::recording::{
    InitialWorldSnapshot, RecordedDynamicWorldState, RecordedEpisodeManifest, RecordedStep,
    queue_recorded_audio_for_playback,
};
use dfb_game::simulation::components::{AircraftRole, AircraftState, ControlAuthority, GunState};
use serde::Serialize;

#[derive(bevy::ecs::system::SystemParam)]
struct ReplayDynamicContext<'w, 's> {
    commands: Commands<'w, 's>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    projectile_query: Query<'w, 's, Entity, With<Projectile>>,
    tracer_query: Query<'w, 's, Entity, With<TracerLifetime>>,
}

#[derive(Debug, Serialize)]
struct ReplaySummary {
    episode_root: String,
    total_steps: u32,
    final_tick: u64,
    final_match_phase: String,
    player_position: Option<[f32; 3]>,
    enemy_position: Option<[f32; 3]>,
    recorded_last_tick: u64,
}

#[derive(Resource)]
struct ReplayEpisodeResource {
    manifest: RecordedEpisodeManifest,
    initial: InitialWorldSnapshot,
    steps: Vec<RecordedStep>,
    has_recorded_fighter2_commands: bool,
    next_index: usize,
    finished: bool,
}

const REPLAY_SPEED_LEVELS: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

fn command_is_neutral(command: &dfb_game::api::types::EnvironmentAction) -> bool {
    command.throttle.abs() <= f32::EPSILON
        && !command.brake
        && command.pitch.abs() <= f32::EPSILON
        && command.roll.abs() <= f32::EPSILON
        && command.yaw.abs() <= f32::EPSILON
        && !command.fire_gun
        && !command.repair
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayCameraMode {
    Chase,
    Free,
}

#[derive(Debug, Resource)]
struct ReplayViewerState {
    mode: ReplayCameraMode,
    target_role: AircraftRole,
    rear_view: bool,
    free_translation: Vec3,
    free_yaw: f32,
    free_pitch: f32,
    enemy_control_source: &'static str,
}

impl Default for ReplayViewerState {
    fn default() -> Self {
        Self::new(AircraftRole::Fighter1)
    }
}

impl ReplayViewerState {
    fn new(target_role: AircraftRole) -> Self {
        Self {
            mode: ReplayCameraMode::Chase,
            target_role,
            rear_view: false,
            free_translation: Vec3::new(-20.0, 6.0, 20.0),
            free_yaw: 0.0,
            free_pitch: -0.15,
            enemy_control_source: "AI",
        }
    }
}

#[derive(Debug, Resource)]
struct ReplayTimelineState {
    paused: bool,
    speed_index: usize,
}

impl Default for ReplayTimelineState {
    fn default() -> Self {
        Self {
            paused: true,
            speed_index: 2,
        }
    }
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let mut headless = false;
    let mut episode_arg: Option<PathBuf> = None;
    let mut scene_override = None;
    let mut project_root_override: Option<PathBuf> = None;
    let mut endpoint = BridgeEndpointConfig::default();
    let mut observer_source = ObserverClientMode::RecordedEpisode;
    let mut observed_role = AircraftRole::Fighter1;
    let mut launcher_session_id: Option<String> = None;
    let mut launch_token: Option<String> = None;
    let mut child_kind: Option<LobbyClientKind> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                let Some(path) = args.next() else {
                    bail!("missing value for --episode");
                };
                episode_arg = Some(PathBuf::from(path));
            }
            "--headless" => headless = true,
            "--scene" => {
                let Some(scene_name) = args.next() else {
                    bail!("missing value for --scene");
                };
                scene_override = Some(scene_name);
            }
            "--project-root" => {
                let Some(project_root) = args.next() else {
                    bail!("missing value for --project-root");
                };
                project_root_override = Some(PathBuf::from(project_root));
            }
            "--live" => observer_source = ObserverClientMode::LiveServer,
            "--bridge" => {
                let Some(mode) = args.next() else {
                    bail!("missing value for --bridge");
                };
                endpoint.transport = match mode.as_str() {
                    "in_process" => BridgeTransport::InProcess,
                    "tcp" => BridgeTransport::Tcp,
                    other => bail!("unsupported bridge transport: {other}"),
                };
                observer_source = ObserverClientMode::LiveServer;
            }
            "--bridge-addr" => {
                let Some(address) = args.next() else {
                    bail!("missing value for --bridge-addr");
                };
                endpoint.address = address;
                observer_source = ObserverClientMode::LiveServer;
            }
            "--bridge-session" => {
                let Some(session) = args.next() else {
                    bail!("missing value for --bridge-session");
                };
                endpoint.session = session;
                observer_source = ObserverClientMode::LiveServer;
            }
            "--observed-role" => {
                let Some(role) = args.next() else {
                    bail!("missing value for --observed-role");
                };
                observed_role = parse_aircraft_role_arg(&role)?;
            }
            "--launcher-session-id" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --launcher-session-id");
                };
                launcher_session_id = Some(value);
            }
            "--launch-token" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --launch-token");
                };
                launch_token = Some(value);
            }
            "--child-kind" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --child-kind");
                };
                child_kind = match value.as_str() {
                    "gameplay" => Some(LobbyClientKind::Gameplay),
                    "observer" => Some(LobbyClientKind::Observer),
                    "launcher" => Some(LobbyClientKind::Launcher),
                    other => bail!("unsupported child kind: {other}"),
                };
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    match observer_source {
        ObserverClientMode::RecordedEpisode => {
            let recordings_root = ConfigPaths::default().recordings_root();
            let episode_root = episode_arg.unwrap_or(find_latest_episode(&recordings_root)?);
            if headless {
                run_headless_summary(&episode_root)
            } else {
                run_headed_replay(
                    &episode_root,
                    observed_role,
                    project_root_override.unwrap_or_else(resolve_project_root),
                )
            }
        }
        ObserverClientMode::LiveServer => {
            if headless {
                bail!("--headless is only supported for recorded episodes");
            }
            run_live_observer(
                scene_override,
                project_root_override.unwrap_or_else(resolve_project_root),
                endpoint,
                observed_role,
                BridgeLaunchBinding {
                    launcher_session_id,
                    launch_token,
                    child_kind,
                },
            )
        }
    }
}

fn run_live_observer(
    scene_override: Option<String>,
    project_root: PathBuf,
    endpoint: BridgeEndpointConfig,
    observed_role: AircraftRole,
    launch_binding: BridgeLaunchBinding,
) -> Result<()> {
    info!(
        "starting live spectator observer via {:?} at {} session='{}'",
        endpoint.transport, endpoint.address, endpoint.session
    );
    let mut app = build_observer_app_with_mode(
        ConfigPaths {
            project_root,
            scene_override,
            scene_override_path: None,
        },
        ObserverClientMode::LiveServer,
    );
    app.insert_resource(endpoint);
    app.insert_resource(RequestedControlRole(BridgeControlSlot::Spectator));
    app.insert_resource(launch_binding);
    app.insert_resource(ObservedAircraftRole(observed_role));
    app.insert_resource(ReplayViewerState::new(observed_role));
    app.add_systems(
        PostStartup,
        (
            configure_replay_camera,
            initialize_replay_viewer,
            log_live_observer_controls,
        )
            .chain(),
    )
    .add_systems(Update, sync_observer_compass_reference)
    .add_systems(Update, update_live_observer_camera_input)
    .add_systems(
        Update,
        (
            update_live_observer_free_camera
                .after(dfb_game::presentation::camera::update_follow_camera),
            sync_observer_audio_listener_override.after(update_live_observer_free_camera),
        ),
    );
    app.run();
    Ok(())
}

fn run_headed_replay(
    episode_root: &Path,
    observed_role: AircraftRole,
    project_root: PathBuf,
) -> Result<()> {
    let (manifest, initial, steps) = load_episode(episode_root)?;
    let has_recorded_fighter2_commands = steps
        .iter()
        .any(|step| !command_is_neutral(&step.fighter2_command));
    let mut app = build_observer_app_with_paths(ConfigPaths {
        project_root,
        scene_override: Some(manifest.scene_name.clone()),
        scene_override_path: None,
    });
    app.sub_app_mut(RenderApp)
        .insert_resource(GpuPreprocessingSupport {
            max_supported_mode: GpuPreprocessingMode::None,
        });
    app.insert_resource(ReplayEpisodeResource {
        manifest,
        initial,
        steps,
        has_recorded_fighter2_commands,
        next_index: 0,
        finished: false,
    })
    .insert_resource(ReplayViewerState::new(observed_role))
    .insert_resource(ReplayTimelineState::default())
    .add_systems(
        PostStartup,
        (
            initialize_replay_world,
            configure_replay_camera,
            initialize_replay_viewer,
            initialize_replay_timeline,
            spawn_replay_overlay,
            log_replay_viewer_controls,
        )
            .chain(),
    )
    .add_systems(Update, sync_observer_compass_reference)
    .add_systems(Update, (update_replay_viewer_input, update_replay_overlay))
    .add_systems(Update, sync_recorded_observer_feed_status)
    .add_systems(
        Update,
        (
            update_replay_camera.after(dfb_game::presentation::camera::update_follow_camera),
            sync_observer_audio_listener_override.after(update_replay_camera),
        ),
    )
    .add_systems(
        FixedUpdate,
        drive_replay_controls.in_set(SimulationSet::GatherInput),
    );
    app.run();
    Ok(())
}

fn parse_aircraft_role_arg(role: &str) -> Result<AircraftRole> {
    match role {
        "fighter1" => Ok(AircraftRole::Fighter1),
        "fighter2" => Ok(AircraftRole::Fighter2),
        other => bail!("unsupported observed role: {other} (expected fighter1|fighter2)"),
    }
}

fn initialize_replay_world(
    replay: Res<ReplayEpisodeResource>,
    mut clock: ResMut<MatchClock>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut viewer: ResMut<ReplayViewerState>,
    mut dynamic_ctx: ReplayDynamicContext,
    mut query: Query<
        (
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut GunState,
            &mut ControlInput,
            &mut ControlAuthority,
            &mut Transform,
        ),
        Without<MainViewCamera>,
    >,
) {
    restore_replay_world(
        &mut dynamic_ctx.commands,
        &replay,
        &mut clock,
        &mut next_phase,
        &mut viewer,
        &mut dynamic_ctx.meshes,
        &mut dynamic_ctx.materials,
        &mut query,
        &dynamic_ctx.projectile_query,
        &dynamic_ctx.tracer_query,
        replay.has_recorded_fighter2_commands,
    );
}

fn restore_replay_world(
    commands: &mut Commands,
    replay: &ReplayEpisodeResource,
    clock: &mut ResMut<MatchClock>,
    next_phase: &mut ResMut<NextState<MatchPhase>>,
    viewer: &mut ResMut<ReplayViewerState>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    query: &mut Query<
        (
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut GunState,
            &mut ControlInput,
            &mut ControlAuthority,
            &mut Transform,
        ),
        Without<MainViewCamera>,
    >,
    projectile_query: &Query<Entity, With<Projectile>>,
    tracer_query: &Query<Entity, With<TracerLifetime>>,
    has_recorded_fighter2_commands: bool,
) {
    clock.elapsed_seconds = replay.initial.state.sim_time_seconds;
    next_phase.set(MatchPhase::Running);
    viewer.enemy_control_source = if has_recorded_fighter2_commands {
        "REPLAY"
    } else {
        "AI"
    };

    for (role, mut state, mut damage, mut gun, mut input, mut authority, mut transform) in
        query.iter_mut()
    {
        let role_name = match role {
            AircraftRole::Fighter1 => "fighter1",
            AircraftRole::Fighter2 => "fighter2",
        };
        let Some(recorded) = replay
            .initial
            .state
            .aircraft
            .iter()
            .find(|aircraft| aircraft.role == role_name)
        else {
            continue;
        };

        state.position = Vec3::from_array(recorded.position);
        state.orientation = Quat::from_array(recorded.orientation_quat);
        state.velocity = Vec3::from_array(recorded.linear_velocity);
        state.angular_rates_deg = Vec3::from_array(recorded.angular_velocity_deg);
        state.forward = (state.orientation * Vec3::Z).normalize_or_zero();
        state.throttle = recorded.throttle;
        state.hit_points = recorded.hit_points;
        state.stall_factor = recorded.stall_factor;
        state.out_of_bounds_seconds = recorded.out_of_bounds_seconds;
        state.ceiling_recovery_seconds = recorded.ceiling_recovery_seconds;
        state.is_destroyed = recorded.destroyed;
        transform.translation = state.position;
        transform.rotation = state.orientation;

        gun.heat = recorded.gun_heat;
        gun.overheated = recorded.gun_overheated;
        input.throttle_delta = 0.0;
        input.brake = false;
        input.pitch = 0.0;
        input.roll = 0.0;
        input.yaw = 0.0;
        input.fire_gun = false;
        input.repair = false;

        damage.is_repairing = recorded.repairing;
        damage.repair_elapsed_seconds = recorded.repair_elapsed_seconds;
        apply_subsystem_snapshot(&mut damage, &recorded.subsystems);

        *authority = match role {
            AircraftRole::Fighter1 => ControlAuthority::Replay,
            AircraftRole::Fighter2 => {
                if has_recorded_fighter2_commands {
                    ControlAuthority::Replay
                } else {
                    ControlAuthority::BuiltInAi
                }
            }
        };
    }

    respawn_replay_dynamic_world(
        commands,
        meshes,
        materials,
        projectile_query,
        tracer_query,
        &replay.initial.dynamic,
    );

    info!(
        "initialized replay for episode {} with {} steps",
        replay.manifest.episode_id,
        replay.steps.len()
    );
}

fn configure_replay_camera(
    mut commands: Commands,
    query: Query<Entity, (With<MainViewCamera>, Without<NoIndirectDrawing>)>,
) {
    for entity in &query {
        commands.entity(entity).insert(NoIndirectDrawing);
    }
}

fn initialize_replay_viewer(
    mut viewer: ResMut<ReplayViewerState>,
    mut observed_role: ResMut<ObservedAircraftRole>,
    camera_query: Query<&Transform, With<MainViewCamera>>,
) {
    let Some(transform) = camera_query.iter().next() else {
        return;
    };
    viewer.free_translation = transform.translation;
    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
    viewer.free_yaw = yaw;
    viewer.free_pitch = pitch;
    observed_role.0 = viewer.target_role;
}

fn initialize_replay_timeline(
    mut virtual_time: ResMut<Time<Virtual>>,
    timeline: Res<ReplayTimelineState>,
) {
    virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
    if timeline.paused {
        virtual_time.pause();
    } else {
        virtual_time.unpause();
    }
}

fn log_replay_viewer_controls() {
    info!(
        "recorded spectator controls: Space play/pause, Left/Right seek, Up/Down speed, Home restart, F1 chase/free, F2 switch target, C front/rear, Tab mouse capture, WASD/QE move, Shift accelerate"
    );
}

fn log_live_observer_controls() {
    info!(
        "live spectator controls: F1 chase/free, F2 switch target, C front/rear, Tab mouse capture, WASD/QE move, Shift accelerate"
    );
}

fn update_live_observer_camera_input(
    real_time: Res<Time<Real>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    bindings: Res<ControlBindings>,
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_control_state: Res<MouseControlState>,
    camera_query: Query<&Transform, With<MainViewCamera>>,
    observed_role: Res<ObservedAircraftRole>,
    mut viewer: ResMut<ReplayViewerState>,
) {
    viewer.target_role = observed_role.0;
    viewer.rear_view = bindings.rear_view.pressed(&keyboard, &mouse_buttons);

    if keyboard.just_pressed(KeyCode::F1) {
        viewer.mode = match viewer.mode {
            ReplayCameraMode::Chase => {
                if let Some(transform) = camera_query.iter().next() {
                    viewer.free_translation = transform.translation;
                    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
                    viewer.free_yaw = yaw;
                    viewer.free_pitch = pitch;
                }
                ReplayCameraMode::Free
            }
            ReplayCameraMode::Free => ReplayCameraMode::Chase,
        };
    }

    if viewer.mode != ReplayCameraMode::Free || !mouse_control_state.captured {
        return;
    }

    viewer.free_yaw -= accumulated_mouse_motion.delta.x * 0.0035;
    viewer.free_pitch -= accumulated_mouse_motion.delta.y * 0.0035;
    viewer.free_pitch = viewer.free_pitch.clamp(-1.54, 1.54);

    let mut move_axis = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        move_axis.z += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        move_axis.z -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        move_axis.x += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        move_axis.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyE) {
        move_axis.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        move_axis.y -= 1.0;
    }

    if move_axis == Vec3::ZERO {
        return;
    }

    let speed = if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
        220.0
    } else {
        80.0
    };
    let rotation = Quat::from_euler(EulerRot::YXZ, viewer.free_yaw, viewer.free_pitch, 0.0);
    let forward = rotation * -Vec3::Z;
    let right = rotation * Vec3::X;
    let up = Vec3::Y;
    let displacement =
        (right * move_axis.x + up * move_axis.y + forward * move_axis.z).normalize_or_zero();
    viewer.free_translation += displacement * speed * real_time.delta_secs();
}

fn update_live_observer_free_camera(
    viewer: Res<ReplayViewerState>,
    mut camera_query: Query<&mut Transform, With<MainViewCamera>>,
) {
    if viewer.mode != ReplayCameraMode::Free {
        return;
    }
    let Some(mut camera_transform) = camera_query.iter_mut().next() else {
        return;
    };
    let rotation = Quat::from_euler(EulerRot::YXZ, viewer.free_yaw, viewer.free_pitch, 0.0);
    camera_transform.translation = viewer.free_translation;
    camera_transform.rotation = rotation;
}

fn sync_observer_compass_reference(
    viewer: Res<ReplayViewerState>,
    mut compass_reference: ResMut<CompassReferenceState>,
) {
    compass_reference.use_camera_forward = viewer.mode == ReplayCameraMode::Free;
}

fn sync_observer_audio_listener_override(
    viewer: Res<ReplayViewerState>,
    camera_query: Query<&Transform, With<MainViewCamera>>,
    mut listener_override: ResMut<ObserverAudioListenerOverride>,
) {
    if viewer.mode != ReplayCameraMode::Free {
        listener_override.use_camera_listener = false;
        listener_override.suspend_audio = false;
        return;
    }

    let Some(transform) = camera_query.iter().next() else {
        listener_override.use_camera_listener = false;
        listener_override.suspend_audio = true;
        return;
    };

    listener_override.use_camera_listener = false;
    listener_override.suspend_audio = true;
    listener_override.position = transform.translation;
    listener_override.forward = transform.forward().as_vec3();
    listener_override.right = transform.right().as_vec3();
}

fn update_replay_viewer_input(
    real_time: Res<Time<Real>>,
    mut virtual_time: ResMut<Time<Virtual>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    bindings: Res<ControlBindings>,
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_control_state: Res<MouseControlState>,
    camera_query: Query<&Transform, With<MainViewCamera>>,
    mut observed_role: ResMut<ObservedAircraftRole>,
    mut clock: ResMut<MatchClock>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut dynamic_ctx: ReplayDynamicContext,
    mut aircraft_query: Query<
        (
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut GunState,
            &mut ControlInput,
            &mut ControlAuthority,
            &mut Transform,
        ),
        Without<MainViewCamera>,
    >,
    mut timeline: ResMut<ReplayTimelineState>,
    mut viewer: ResMut<ReplayViewerState>,
    mut replay_resource: ResMut<ReplayEpisodeResource>,
) {
    viewer.rear_view = bindings.rear_view.pressed(&keyboard, &mouse_buttons);

    if keyboard.just_pressed(KeyCode::Space) {
        timeline.paused = !timeline.paused;
        if timeline.paused {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }
    if keyboard.just_pressed(KeyCode::ArrowUp) {
        timeline.speed_index = (timeline.speed_index + 1).min(REPLAY_SPEED_LEVELS.len() - 1);
        if !timeline.paused {
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }
    if keyboard.just_pressed(KeyCode::ArrowDown) {
        timeline.speed_index = timeline.speed_index.saturating_sub(1);
        if !timeline.paused {
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }
    let seek_steps = 30usize;
    if keyboard.just_pressed(KeyCode::ArrowLeft) {
        let target_next_index = replay_resource.next_index.saturating_sub(seek_steps);
        replay_resource.next_index = target_next_index;
        replay_resource.finished = false;
        if target_next_index == 0 {
            restore_replay_world(
                &mut dynamic_ctx.commands,
                &replay_resource,
                &mut clock,
                &mut next_phase,
                &mut viewer,
                &mut dynamic_ctx.meshes,
                &mut dynamic_ctx.materials,
                &mut aircraft_query,
                &dynamic_ctx.projectile_query,
                &dynamic_ctx.tracer_query,
                replay_resource.has_recorded_fighter2_commands,
            );
        } else if let Some(snapshot) = replay_resource.steps.get(target_next_index - 1) {
            restore_replay_state(
                &mut dynamic_ctx.commands,
                &snapshot.state,
                &snapshot.dynamic,
                &mut clock,
                &mut next_phase,
                &mut viewer,
                &mut dynamic_ctx.meshes,
                &mut dynamic_ctx.materials,
                &mut aircraft_query,
                &dynamic_ctx.projectile_query,
                &dynamic_ctx.tracer_query,
                replay_resource.has_recorded_fighter2_commands,
            );
        }
        if timeline.paused {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }
    if keyboard.just_pressed(KeyCode::ArrowRight) {
        let target_next_index =
            (replay_resource.next_index + seek_steps).min(replay_resource.steps.len());
        replay_resource.next_index = target_next_index;
        replay_resource.finished = target_next_index >= replay_resource.steps.len();
        if target_next_index == 0 {
            restore_replay_world(
                &mut dynamic_ctx.commands,
                &replay_resource,
                &mut clock,
                &mut next_phase,
                &mut viewer,
                &mut dynamic_ctx.meshes,
                &mut dynamic_ctx.materials,
                &mut aircraft_query,
                &dynamic_ctx.projectile_query,
                &dynamic_ctx.tracer_query,
                replay_resource.has_recorded_fighter2_commands,
            );
        } else if let Some(snapshot) = replay_resource.steps.get(target_next_index - 1) {
            restore_replay_state(
                &mut dynamic_ctx.commands,
                &snapshot.state,
                &snapshot.dynamic,
                &mut clock,
                &mut next_phase,
                &mut viewer,
                &mut dynamic_ctx.meshes,
                &mut dynamic_ctx.materials,
                &mut aircraft_query,
                &dynamic_ctx.projectile_query,
                &dynamic_ctx.tracer_query,
                replay_resource.has_recorded_fighter2_commands,
            );
        }
        if timeline.paused || replay_resource.finished {
            timeline.paused = true;
            virtual_time.pause();
        } else {
            virtual_time.unpause();
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }
    if keyboard.just_pressed(KeyCode::Home) {
        replay_resource.next_index = 0;
        replay_resource.finished = false;
        restore_replay_world(
            &mut dynamic_ctx.commands,
            &replay_resource,
            &mut clock,
            &mut next_phase,
            &mut viewer,
            &mut dynamic_ctx.meshes,
            &mut dynamic_ctx.materials,
            &mut aircraft_query,
            &dynamic_ctx.projectile_query,
            &dynamic_ctx.tracer_query,
            replay_resource.has_recorded_fighter2_commands,
        );
        if timeline.paused {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
            virtual_time.set_relative_speed(REPLAY_SPEED_LEVELS[timeline.speed_index]);
        }
    }

    if keyboard.just_pressed(KeyCode::F1) {
        viewer.mode = match viewer.mode {
            ReplayCameraMode::Chase => {
                if let Some(transform) = camera_query.iter().next() {
                    viewer.free_translation = transform.translation;
                    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
                    viewer.free_yaw = yaw;
                    viewer.free_pitch = pitch;
                }
                ReplayCameraMode::Free
            }
            ReplayCameraMode::Free => ReplayCameraMode::Chase,
        };
    }
    if keyboard.just_pressed(KeyCode::F2) {
        viewer.target_role = match viewer.target_role {
            AircraftRole::Fighter1 => AircraftRole::Fighter2,
            AircraftRole::Fighter2 => AircraftRole::Fighter1,
        };
    }
    observed_role.0 = viewer.target_role;

    if viewer.mode != ReplayCameraMode::Free || !mouse_control_state.captured {
        return;
    }

    viewer.free_yaw -= accumulated_mouse_motion.delta.x * 0.0035;
    viewer.free_pitch -= accumulated_mouse_motion.delta.y * 0.0035;
    viewer.free_pitch = viewer.free_pitch.clamp(-1.54, 1.54);

    let mut move_axis = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        move_axis.z += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        move_axis.z -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        move_axis.x += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        move_axis.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyE) {
        move_axis.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyQ) {
        move_axis.y -= 1.0;
    }

    if move_axis == Vec3::ZERO {
        return;
    }

    let speed = if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
        220.0
    } else {
        80.0
    };
    let rotation = Quat::from_euler(EulerRot::YXZ, viewer.free_yaw, viewer.free_pitch, 0.0);
    let forward = rotation * -Vec3::Z;
    let right = rotation * Vec3::X;
    let up = Vec3::Y;
    let displacement =
        (right * move_axis.x + up * move_axis.y + forward * move_axis.z).normalize_or_zero();
    viewer.free_translation += displacement * speed * real_time.delta_secs();
}

fn update_replay_camera(
    viewer: Res<ReplayViewerState>,
    aircraft_query: Query<(&AircraftRole, &AircraftState, &Transform), Without<MainViewCamera>>,
    mut camera_query: Query<(&FollowPlayerCamera, &mut Transform), With<MainViewCamera>>,
) {
    let Some((follow, mut camera_transform)) = camera_query.iter_mut().next() else {
        return;
    };

    match viewer.mode {
        ReplayCameraMode::Chase => {
            let Some((_, _aircraft, aircraft_transform)) = aircraft_query
                .iter()
                .find(|(role, state, _)| **role == viewer.target_role && !state.is_destroyed)
            else {
                return;
            };
            let offset = if viewer.rear_view {
                follow.rear_view_offset
            } else {
                follow.offset
            };
            let forward = aircraft_transform.rotation * Vec3::Z;
            let up = aircraft_transform.rotation * Vec3::Y;
            camera_transform.translation =
                aircraft_transform.translation + aircraft_transform.rotation * offset;
            let look_direction = if viewer.rear_view { -forward } else { forward };
            camera_transform.look_to(look_direction, up);
        }
        ReplayCameraMode::Free => {
            let rotation = Quat::from_euler(EulerRot::YXZ, viewer.free_yaw, viewer.free_pitch, 0.0);
            camera_transform.translation = viewer.free_translation;
            camera_transform.rotation = rotation;
        }
    }
}

fn drive_replay_controls(
    mut commands: Commands,
    mut replay: ResMut<ReplayEpisodeResource>,
    mut virtual_time: ResMut<Time<Virtual>>,
    mut timeline: ResMut<ReplayTimelineState>,
    mut clock: ResMut<MatchClock>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut audio_events: ResMut<AudioEventQueue>,
    mut viewer: ResMut<ReplayViewerState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<
        (
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut GunState,
            &mut ControlInput,
            &mut ControlAuthority,
            &mut Transform,
        ),
        Without<MainViewCamera>,
    >,
    projectile_query: Query<Entity, With<Projectile>>,
    tracer_query: Query<Entity, With<TracerLifetime>>,
) {
    if timeline.paused || replay.finished {
        return;
    }

    if let Some(step) = replay.steps.get(replay.next_index) {
        restore_replay_state(
            &mut commands,
            &step.state,
            &step.dynamic,
            &mut clock,
            &mut next_phase,
            &mut viewer,
            &mut meshes,
            &mut materials,
            &mut query,
            &projectile_query,
            &tracer_query,
            replay.has_recorded_fighter2_commands,
        );
        queue_recorded_audio_for_playback(
            &mut audio_events,
            step.audio_semantics.as_ref(),
            &step.state.events_since_last_step,
        );
        replay.next_index += 1;
    } else {
        replay.finished = true;
        timeline.paused = true;
        virtual_time.pause();
        info!("replay finished");
    }
}

fn spawn_replay_overlay(mut commands: Commands, fonts: Res<UiFontHandles>) {
    spawn_replay_progress_hud(&mut commands, &fonts);
}

fn update_replay_overlay(
    viewer: Res<ReplayViewerState>,
    observer_feed: Res<ObserverFeedStatus>,
    mut text_query: Query<&mut Text, With<ReplayProgressHudText>>,
    mut fill_query: Query<&mut Node, With<ReplayProgressBarFill>>,
    mut bar_text_query: Query<
        &mut Text,
        (With<ReplayProgressBarText>, Without<ReplayProgressHudText>),
    >,
) {
    let Some(mut text) = text_query.iter_mut().next() else {
        return;
    };
    let total_steps = observer_feed.total_steps.unwrap_or(0);
    let current_step = observer_feed.current_step.unwrap_or(0);
    let progress_ratio = if total_steps > 0 {
        (current_step as f32 / total_steps as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mode = match viewer.mode {
        ReplayCameraMode::Chase => "CHASE",
        ReplayCameraMode::Free => "FREE",
    };
    let target = match viewer.target_role {
        AircraftRole::Fighter1 => "FIGHTER1",
        AircraftRole::Fighter2 => "FIGHTER2",
    };
    let facing = if viewer.rear_view { "REAR" } else { "FRONT" };
    let playback = if observer_feed.playback_finished == Some(true) {
        "FINISHED"
    } else if observer_feed.playback_paused == Some(true) {
        "PAUSED"
    } else {
        "PLAYING"
    };
    let speed = observer_feed.playback_speed.unwrap_or(1.0);
    let rows = vec![
        vec![
            Cell::left("STATE"),
            Cell::right(playback),
            Cell::left("SPD"),
            Cell::right(format!("{speed:.2}x")),
            Cell::left("STEP"),
            Cell::right(format!("{current_step:>4}/{total_steps:<4}")),
        ],
        vec![
            Cell::left("CAM"),
            Cell::right(mode),
            Cell::left("VIEW"),
            Cell::right(facing),
            Cell::left("TGT"),
            Cell::right(target),
        ],
        vec![
            Cell::left("ENMY"),
            Cell::right(viewer.enemy_control_source),
            Cell::left("CTRL"),
            Cell::right("Space/Arrows/Home"),
            Cell::left("MODE"),
            Cell::right("F1/F2/C"),
        ],
    ];
    text.0 = format!(
        "OBSERVER REPLAY\n{}",
        render_rows_with_min_widths(&rows, 2, &[5, 10, 4, 8, 4, 12]),
    );
    if let Ok(mut fill) = fill_query.single_mut() {
        fill.width = percent(progress_ratio * 100.0);
    }
    if let Ok(mut bar_text) = bar_text_query.single_mut() {
        bar_text.0 = format!("{:>5.1}%", progress_ratio * 100.0);
    }
}

fn sync_recorded_observer_feed_status(
    replay: Res<ReplayEpisodeResource>,
    clock: Res<MatchClock>,
    timeline: Res<ReplayTimelineState>,
    mut observer_provider: ResMut<ObserverFeedProviderState>,
    mut observer_feed: ResMut<ObserverFeedStatus>,
) {
    apply_recorded_observer_feed_status(
        &mut observer_provider,
        &mut observer_feed,
        RecordedObserverFeedSample {
            sim_time_seconds: clock.elapsed_seconds,
            current_tick: if replay.next_index == 0 {
                Some(0)
            } else {
                replay
                    .steps
                    .get(replay.next_index - 1)
                    .map(|step| step.tick)
            },
            current_step: replay.next_index.min(replay.steps.len()),
            total_steps: replay.steps.len(),
            playback_finished: replay.finished,
            playback_paused: timeline.paused,
            playback_speed: REPLAY_SPEED_LEVELS[timeline.speed_index],
        },
    );
}

fn restore_replay_state(
    commands: &mut Commands,
    snapshot: &dfb_game::api::types::StateObservation,
    dynamic: &RecordedDynamicWorldState,
    clock: &mut ResMut<MatchClock>,
    next_phase: &mut ResMut<NextState<MatchPhase>>,
    viewer: &mut ResMut<ReplayViewerState>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    query: &mut Query<
        (
            &AircraftRole,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut GunState,
            &mut ControlInput,
            &mut ControlAuthority,
            &mut Transform,
        ),
        Without<MainViewCamera>,
    >,
    projectile_query: &Query<Entity, With<Projectile>>,
    tracer_query: &Query<Entity, With<TracerLifetime>>,
    has_recorded_fighter2_commands: bool,
) {
    clock.elapsed_seconds = snapshot.sim_time_seconds;
    next_phase.set(MatchPhase::Running);
    viewer.enemy_control_source = if has_recorded_fighter2_commands {
        "REPLAY"
    } else {
        "AI"
    };

    for (role, mut state, mut damage, mut gun, mut input, mut authority, mut transform) in
        query.iter_mut()
    {
        let role_name = match role {
            AircraftRole::Fighter1 => "fighter1",
            AircraftRole::Fighter2 => "fighter2",
        };
        let Some(recorded) = snapshot
            .aircraft
            .iter()
            .find(|aircraft| aircraft.role == role_name)
        else {
            continue;
        };

        state.position = Vec3::from_array(recorded.position);
        state.orientation = Quat::from_array(recorded.orientation_quat);
        state.velocity = Vec3::from_array(recorded.linear_velocity);
        state.angular_rates_deg = Vec3::from_array(recorded.angular_velocity_deg);
        state.forward = (state.orientation * Vec3::Z).normalize_or_zero();
        state.throttle = recorded.throttle;
        state.hit_points = recorded.hit_points;
        state.stall_factor = recorded.stall_factor;
        state.out_of_bounds_seconds = recorded.out_of_bounds_seconds;
        state.ceiling_recovery_seconds = recorded.ceiling_recovery_seconds;
        state.is_destroyed = recorded.destroyed;
        transform.translation = state.position;
        transform.rotation = state.orientation;

        gun.heat = recorded.gun_heat;
        gun.overheated = recorded.gun_overheated;
        input.throttle_delta = 0.0;
        input.brake = false;
        input.pitch = 0.0;
        input.roll = 0.0;
        input.yaw = 0.0;
        input.fire_gun = false;
        input.repair = false;

        damage.is_repairing = recorded.repairing;
        damage.repair_elapsed_seconds = recorded.repair_elapsed_seconds;
        apply_subsystem_snapshot(&mut damage, &recorded.subsystems);

        *authority = match role {
            AircraftRole::Fighter1 => ControlAuthority::Replay,
            AircraftRole::Fighter2 => {
                if has_recorded_fighter2_commands {
                    ControlAuthority::Replay
                } else {
                    ControlAuthority::BuiltInAi
                }
            }
        };
    }

    respawn_replay_dynamic_world(
        commands,
        meshes,
        materials,
        projectile_query,
        tracer_query,
        dynamic,
    );
}

fn respawn_replay_dynamic_world(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    projectile_query: &Query<Entity, With<Projectile>>,
    tracer_query: &Query<Entity, With<TracerLifetime>>,
    dynamic: &RecordedDynamicWorldState,
) {
    for entity in projectile_query.iter() {
        commands.entity(entity).despawn();
    }
    for entity in tracer_query.iter() {
        commands.entity(entity).despawn();
    }

    for projectile in &dynamic.projectiles {
        let shooter_role = match projectile.shooter_role.as_str() {
            "Fighter2" | "fighter2" => AircraftRole::Fighter2,
            _ => AircraftRole::Fighter1,
        };
        let velocity = Vec3::from_array(projectile.velocity);
        let rotation = Quat::from_rotation_arc(Vec3::Z, velocity.normalize_or_zero());
        spawn_projectile_visual_entity(
            commands,
            Some(meshes),
            Some(materials),
            Projectile {
                id: projectile.id,
                shooter_role,
                velocity,
                damage: projectile.damage,
                remaining_distance: projectile.remaining_distance,
                hit_radius: projectile.hit_radius,
                flyby_emitted: true,
                lag_compensation_ticks: 0,
            },
            Transform::from_translation(Vec3::from_array(projectile.position))
                .with_rotation(rotation),
        );
    }

    for tracer in &dynamic.tracers {
        spawn_tracer_visual_entity(
            commands,
            Some(meshes),
            Some(materials),
            Vec3::from_array(tracer.position),
            tracer.remaining_seconds,
        );
    }
}

fn apply_subsystem_snapshot(damage: &mut AircraftDamageState, subsystems: &[SubsystemObservation]) {
    for subsystem in subsystems {
        let Some(kind) = parse_subsystem_name(&subsystem.name) else {
            continue;
        };
        let target = damage.subsystem_mut(kind);
        target.current = subsystem.hit_points.clamp(0.0, target.max);
    }
}

fn parse_subsystem_name(name: &str) -> Option<AircraftSubsystem> {
    match name {
        "LeftWing" => Some(AircraftSubsystem::LeftWing),
        "RightWing" => Some(AircraftSubsystem::RightWing),
        "PitchTail" => Some(AircraftSubsystem::PitchTail),
        "YawTail" => Some(AircraftSubsystem::YawTail),
        "Engine" => Some(AircraftSubsystem::Engine),
        _ => None,
    }
}

fn run_headless_summary(episode_root: &Path) -> Result<()> {
    let (manifest, _initial, _steps) = load_episode(episode_root)?;
    let mut env = EnvironmentInstance::new_headless(
        ConfigPaths::default(),
        EnvironmentResetOptions {
            scene_name: Some(manifest.scene_name.clone()),
            seed: manifest.seed,
            enable_visual: false,
            enable_audio: false,
            visual_sensors: Vec::new(),
            audio_window_seconds: manifest.fixed_time_step_seconds,
            ticks_per_step: 1,
            ..EnvironmentResetOptions::default()
        },
    );
    env.reset(&EnvironmentResetOptions {
        scene_name: Some(manifest.scene_name.clone()),
        seed: manifest.seed,
        enable_visual: false,
        enable_audio: false,
        visual_sensors: Vec::new(),
        audio_window_seconds: manifest.fixed_time_step_seconds,
        ticks_per_step: 1,
        ..EnvironmentResetOptions::default()
    });
    env.set_control_authority(AircraftRole::Fighter1, ControlAuthority::ExternalAgent);

    let mut final_result: Option<StepResult> = None;
    let mut recorded_last_tick = 0;
    for step in RecordingAccess::new(episode_root).steps()? {
        recorded_last_tick = step.tick;
        let mut actions = vec![TargetedEnvironmentAction {
            role: AircraftRole::Fighter1,
            action: step.fighter1_command,
        }];
        if !command_is_neutral(&step.fighter2_command) {
            env.set_control_authority(AircraftRole::Fighter2, ControlAuthority::ExternalAgent);
            actions.push(TargetedEnvironmentAction {
                role: AircraftRole::Fighter2,
                action: step.fighter2_command,
            });
        }
        final_result = Some(env.step_targeted(actions));
    }

    let final_result = final_result.unwrap_or_else(|| env.step_targeted(std::iter::empty()));
    let player_position = final_result
        .observation
        .state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter1")
        .map(|aircraft| aircraft.position);
    let enemy_position = final_result
        .observation
        .state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter2")
        .map(|aircraft| aircraft.position);

    let summary = ReplaySummary {
        episode_root: episode_root.display().to_string(),
        total_steps: manifest.total_steps,
        final_tick: final_result.observation.state.tick,
        final_match_phase: final_result.observation.state.match_phase.clone(),
        player_position,
        enemy_position,
        recorded_last_tick,
    };
    let output_path = PathBuf::from("tmp/replay_recorded_episode_summary.ron");
    fs::write(
        &output_path,
        ron::ser::to_string_pretty(&summary, ron::ser::PrettyConfig::default())?,
    )?;
    println!("wrote {}", output_path.display());
    Ok(())
}

fn load_episode(
    episode_root: &Path,
) -> Result<(
    RecordedEpisodeManifest,
    InitialWorldSnapshot,
    Vec<RecordedStep>,
)> {
    let manifest: RecordedEpisodeManifest = ron::from_str(
        &fs::read_to_string(episode_root.join("episode.ron")).with_context(|| {
            format!(
                "failed to read {}",
                episode_root.join("episode.ron").display()
            )
        })?,
    )?;
    let initial: InitialWorldSnapshot = ron::from_str(
        &fs::read_to_string(episode_root.join(&manifest.artifact_convention.initial_state_path))
            .with_context(|| {
                format!(
                    "failed to read {}",
                    episode_root
                        .join(&manifest.artifact_convention.initial_state_path)
                        .display()
                )
            })?,
    )?;
    let steps = load_steps(episode_root, &manifest)?;
    Ok((manifest, initial, steps))
}

fn load_steps(
    episode_root: &Path,
    manifest: &RecordedEpisodeManifest,
) -> Result<Vec<RecordedStep>> {
    let _ = (episode_root, manifest);
    RecordingAccess::new(episode_root).steps()
}

fn find_latest_episode(recordings_root: &Path) -> Result<PathBuf> {
    let mut candidates = fs::read_dir(recordings_root)
        .with_context(|| format!("failed to read {}", recordings_root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("episode.ron").exists())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop().with_context(|| {
        format!(
            "no recorded episodes found in {}",
            recordings_root.display()
        )
    })
}
