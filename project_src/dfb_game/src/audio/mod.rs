use std::cell::RefCell;
use std::collections::VecDeque;
use std::env;
use std::f32::consts::TAU;
#[cfg(target_os = "linux")]
use std::fs::OpenOptions;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use bevy::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig, SupportedStreamConfigRange};
use serde::{Deserialize, Serialize};

use crate::api::types::EnvironmentEvent;
use crate::api::types::{AudioFeatureVector, AudioObservation, ObservationCaptureConfig};
use crate::bridge::{BridgeClientConnectionStatus, BridgeLinkState};
use crate::core::config::RepositoryConfig;
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::input::actions::ControlBindings;
use crate::presentation::hud::ObservedAircraftRole;
use crate::simulation::components::{AircraftRole, AircraftState};

const FRONT_LEFT: usize = 0;
const FRONT_RIGHT: usize = 1;
const FRONT_CENTER: usize = 2;
const LFE: usize = 3;
#[cfg(test)]
const REAR_LEFT: usize = 4;
#[cfg(test)]
const REAR_RIGHT: usize = 5;
#[cfg(test)]
const SIDE_LEFT: usize = 6;
#[cfg(test)]
const SIDE_RIGHT: usize = 7;

pub struct CpalAudioPlugin;
pub struct AudioObservationPlugin;

const AUDIO_OBSERVATION_CHANNELS: u16 = 2;
const AUDIO_OBSERVATION_MAX_SECONDS: f32 = 5.0;
const AUDIO_OUTPUT_RETRY_SECONDS: f64 = 2.0;
impl Plugin for AudioObservationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AudioEventQueue>()
            .init_resource::<FrameAudioState>()
            .init_resource::<ObserverAudioListenerOverride>()
            .init_resource::<AudioObservationState>()
            .init_resource::<AudioMuteState>()
            .init_resource::<AudioAnomalyDiagnostics>()
            .add_systems(PreUpdate, begin_audio_frame)
            .add_systems(FixedUpdate, reset_frame_audio_state)
            .add_systems(Update, toggle_audio_mute)
            .add_systems(
                Update,
                reset_audio_runtime_lifecycle.before(sync_audio_scene_state),
            )
            .add_systems(
                FixedUpdate,
                reset_audio_runtime_lifecycle_capture.before(capture_observed_audio_frame),
            )
            .add_systems(
                FixedUpdate,
                (
                    capture_observed_audio_frame,
                    synthesize_audio_observation.after(capture_observed_audio_frame),
                ),
            )
            .add_systems(
                Update,
                (
                    sync_audio_scene_state,
                    diagnose_audio_runtime.after(sync_audio_scene_state),
                    self_heal_audio_runtime.after(diagnose_audio_runtime),
                ),
            );
    }
}

impl Plugin for CpalAudioPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AudioOutputRetryState>()
            .add_systems(Startup, initialize_audio_output)
            .add_systems(Update, ensure_audio_output.before(sync_audio_scene_state));
    }
}

#[derive(Resource, Clone)]
pub(crate) struct AudioOutputResource {
    state: Arc<Mutex<RuntimeAudioState>>,
    stream_error_count: Arc<AtomicU32>,
}

#[derive(Resource, Debug, Clone, Copy, Default)]
struct AudioOutputRetryState {
    next_retry_at_seconds: f64,
    pending_stream_rebuild: bool,
}

#[derive(Debug, Default)]
struct AudioLifecycleTracker {
    last_clock_seconds: f32,
    last_phase: Option<MatchPhase>,
    last_connected: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AudioEmitterAnomalySnapshot {
    fighter1_missing: bool,
    fighter2_missing: bool,
    stream_error_count: u32,
}

#[derive(Resource, Debug, Default, Clone, Copy)]
struct AudioAnomalyDiagnostics {
    last_logged: Option<AudioEmitterAnomalySnapshot>,
    active_snapshot: Option<AudioEmitterAnomalySnapshot>,
    active_seconds: f32,
}

#[derive(Resource, Debug, Default, Clone, Copy)]
struct AudioMuteState {
    muted: bool,
}

thread_local! {
    static AUDIO_STREAM: RefCell<Option<cpal::Stream>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioEventKind {
    GunFire,
    BulletFlyBy,
    Hit,
}

#[derive(Debug, Clone, Copy)]
pub struct QueuedAudioEvent {
    pub kind: AudioEventKind,
    pub position: Vec3,
    pub volume: f32,
}

#[derive(Resource, Debug, Default, Clone)]
pub struct AudioEventQueue {
    pub events: Vec<QueuedAudioEvent>,
}

impl AudioEventQueue {
    pub fn push(&mut self, kind: AudioEventKind, position: Vec3, volume: f32) {
        self.events.push(QueuedAudioEvent {
            kind,
            position,
            volume,
        });
    }
}

#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct FrameAudioState {
    pub listener_position: Option<Vec3>,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct ObserverAudioListenerOverride {
    pub use_camera_listener: bool,
    pub suspend_audio: bool,
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
}

impl Default for ObserverAudioListenerOverride {
    fn default() -> Self {
        Self {
            use_camera_listener: false,
            suspend_audio: false,
            position: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AudioFeatureSample {
    left_right_energy: f32,
    front_back_energy: f32,
    engine_energy: f32,
    gunfire_energy: f32,
    hit_energy: f32,
    flyby_energy: f32,
}

#[derive(Debug, Clone, Resource)]
pub struct AudioObservationState {
    sample_rate: f32,
    listener: ListenerState,
    observed_role: AircraftRole,
    external_listener_mix: bool,
    voices: EngineVoicePair<ObservedEngineVoiceState>,
    one_shots: Vec<OneShotEmitterState>,
    recent_samples: VecDeque<f32>,
    recent_features: VecDeque<AudioFeatureSample>,
    max_seconds: f32,
}

impl Default for AudioObservationState {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            listener: ListenerState::default(),
            observed_role: AircraftRole::Fighter1,
            external_listener_mix: false,
            voices: EngineVoicePair::default(),
            one_shots: Vec::new(),
            recent_samples: VecDeque::new(),
            recent_features: VecDeque::new(),
            max_seconds: AUDIO_OBSERVATION_MAX_SECONDS,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EngineVoicePair<T> {
    fighter1: T,
    fighter2: T,
}

impl<T> EngineVoicePair<T> {
    fn get(&self, role: AircraftRole) -> &T {
        match role {
            AircraftRole::Fighter1 => &self.fighter1,
            AircraftRole::Fighter2 => &self.fighter2,
        }
    }

    fn get_mut(&mut self, role: AircraftRole) -> &mut T {
        match role {
            AircraftRole::Fighter1 => &mut self.fighter1,
            AircraftRole::Fighter2 => &mut self.fighter2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ListenerState {
    position: Vec3,
    forward: Vec3,
    right: Vec3,
}

impl Default for ListenerState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EngineEmitterState {
    position: Vec3,
    throttle: f32,
    alive: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ObservedEngineVoiceState {
    emitter: EngineEmitterState,
    phase: f32,
    missing_seconds: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct RuntimeEngineVoiceState {
    emitter: EngineEmitterState,
    phase: f32,
    missing_seconds: f32,
}

#[derive(Debug, Clone, Copy)]
struct OneShotEmitterState {
    kind: AudioEventKind,
    position: Vec3,
    gain: f32,
    age: f32,
    duration: f32,
    phase: f32,
}

#[derive(Debug)]
struct RuntimeAudioState {
    sample_rate: f32,
    listener: ListenerState,
    observed_role: AircraftRole,
    external_listener_mix: bool,
    suspend_audio: bool,
    voices: EngineVoicePair<RuntimeEngineVoiceState>,
    master_volume: f32,
    player_engine_volume: f32,
    enemy_engine_volume: f32,
    gun_fire_volume: f32,
    bullet_flyby_volume: f32,
    hit_volume: f32,
    spatial_near_distance: f32,
    spatial_far_distance: f32,
    emitter_presence_grace: f32,
    one_shots: Vec<OneShotEmitterState>,
}

impl RuntimeAudioState {
    fn new(config: &RepositoryConfig, sample_rate: f32) -> Self {
        Self {
            sample_rate,
            listener: ListenerState::default(),
            observed_role: AircraftRole::Fighter1,
            external_listener_mix: false,
            suspend_audio: false,
            voices: EngineVoicePair::default(),
            master_volume: config.game.audio.master_volume,
            player_engine_volume: config.game.audio.player_engine_volume,
            enemy_engine_volume: config.game.audio.enemy_engine_volume,
            gun_fire_volume: config.game.audio.gun_fire_volume,
            bullet_flyby_volume: config.game.audio.bullet_flyby_volume,
            hit_volume: config.game.audio.hit_volume,
            spatial_near_distance: config.game.audio.spatial_near_distance.max(1.0),
            spatial_far_distance: config.game.audio.spatial_far_distance.max(10.0),
            emitter_presence_grace: 0.35,
            one_shots: Vec::new(),
        }
    }
}

fn begin_audio_frame(mut frame_audio: ResMut<FrameAudioState>) {
    frame_audio.listener_position = None;
}

fn reset_frame_audio_state(mut frame_audio: ResMut<FrameAudioState>) {
    frame_audio.listener_position = None;
}

fn reset_audio_runtime_lifecycle(
    config: Res<RepositoryConfig>,
    match_phase: Option<Res<State<MatchPhase>>>,
    match_clock: Option<Res<MatchClock>>,
    link_state: Option<Res<BridgeLinkState>>,
    mut observation_audio: ResMut<AudioObservationState>,
    audio: Option<Res<AudioOutputResource>>,
    mut tracker: Local<AudioLifecycleTracker>,
) {
    let phase = match_phase.as_deref().map(|phase| *phase.get());
    let clock_seconds = match_clock
        .as_deref()
        .map(|clock| clock.elapsed_seconds)
        .unwrap_or(tracker.last_clock_seconds);
    let connected = link_state
        .as_deref()
        .map(|state| state.client_status == BridgeClientConnectionStatus::Connected);

    let round_rewound = clock_seconds + 0.001 < tracker.last_clock_seconds;
    let phase_reset = matches!(
        tracker.last_phase,
        Some(MatchPhase::Running | MatchPhase::Finished)
    ) && phase == Some(MatchPhase::Loading);
    let reconnected = tracker.last_connected == Some(false) && connected == Some(true);
    let disconnected = tracker.last_connected == Some(true) && connected == Some(false);

    if round_rewound || phase_reset || reconnected || disconnected {
        reset_audio_states(&config, &mut observation_audio, audio.as_deref());
    }

    tracker.last_clock_seconds = clock_seconds;
    tracker.last_phase = phase;
    tracker.last_connected = connected;
}

fn reset_audio_runtime_lifecycle_capture(
    match_phase: Option<Res<State<MatchPhase>>>,
    match_clock: Option<Res<MatchClock>>,
    mut observation_audio: ResMut<AudioObservationState>,
    mut tracker: Local<AudioLifecycleTracker>,
) {
    let phase = match_phase.as_deref().map(|phase| *phase.get());
    let clock_seconds = match_clock
        .as_deref()
        .map(|clock| clock.elapsed_seconds)
        .unwrap_or(tracker.last_clock_seconds);

    let round_rewound = clock_seconds + 0.001 < tracker.last_clock_seconds;
    let phase_reset = matches!(
        tracker.last_phase,
        Some(MatchPhase::Running | MatchPhase::Finished)
    ) && phase == Some(MatchPhase::Loading);

    if round_rewound || phase_reset {
        reset_observation_audio_state(&mut observation_audio);
    }

    tracker.last_clock_seconds = clock_seconds;
    tracker.last_phase = phase;
}

fn reset_audio_states(
    config: &RepositoryConfig,
    observation_audio: &mut AudioObservationState,
    audio: Option<&AudioOutputResource>,
) {
    reset_observation_audio_state(observation_audio);
    if let Some(audio) = audio
        && let Ok(mut runtime) = audio.state.lock()
    {
        let sample_rate = runtime.sample_rate;
        *runtime = RuntimeAudioState::new(config, sample_rate);
    }
}

fn reset_observation_audio_state(observation_audio: &mut AudioObservationState) {
    let sample_rate = observation_audio.sample_rate;
    let max_seconds = observation_audio.max_seconds;
    *observation_audio = AudioObservationState::default();
    observation_audio.sample_rate = sample_rate;
    observation_audio.max_seconds = max_seconds;
}

pub fn reset_capture_audio_observation(world: &mut World) {
    if let Some(mut observation_audio) = world.get_resource_mut::<AudioObservationState>() {
        reset_observation_audio_state(&mut observation_audio);
    }
}

fn suspend_observer_audio_state(observation_audio: &mut AudioObservationState) {
    observation_audio.one_shots.clear();
    observation_audio.recent_samples.clear();
    observation_audio.recent_features.clear();
    observation_audio.external_listener_mix = false;
    observation_audio.voices.fighter1.emitter.alive = false;
    observation_audio.voices.fighter2.emitter.alive = false;
}

fn capture_observed_audio_frame(
    aircraft_query: Query<(&AircraftRole, &AircraftState)>,
    observed_role: Option<Res<ObservedAircraftRole>>,
    listener_override: Option<Res<ObserverAudioListenerOverride>>,
    mut frame_audio: ResMut<FrameAudioState>,
) {
    if listener_override
        .as_deref()
        .is_some_and(|override_state| override_state.suspend_audio)
    {
        frame_audio.listener_position = None;
        return;
    }

    if let Some(listener_override) = listener_override
        .as_deref()
        .filter(|override_state| override_state.use_camera_listener)
    {
        frame_audio.listener_position = Some(listener_override.position);
        return;
    }

    let observed_role = observed_role
        .as_deref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    frame_audio.listener_position = aircraft_query
        .iter()
        .find(|(role, state)| **role == observed_role && !state.is_destroyed)
        .map(|(_, state)| state.position);
}

fn initialize_audio_output(mut commands: Commands, config: Res<RepositoryConfig>) {
    if !config.game.audio.enabled {
        info!("cpal audio disabled by configuration");
        return;
    }

    match build_audio_output(&config) {
        Ok((resource, stream)) => {
            AUDIO_STREAM.with(|slot| {
                *slot.borrow_mut() = Some(stream);
            });
            commands.insert_resource(resource);
        }
        Err(error) => warn!("failed to initialize cpal audio output: {error:#}"),
    }
}

fn ensure_audio_output(
    mut commands: Commands,
    config: Res<RepositoryConfig>,
    time: Res<Time<Real>>,
    audio: Option<Res<AudioOutputResource>>,
    mut retry_state: ResMut<AudioOutputRetryState>,
) {
    if !config.game.audio.enabled {
        return;
    }

    if let Some(audio) = audio.as_ref() {
        let stream_error_count = audio.stream_error_count.load(Ordering::Relaxed);
        if stream_error_count > 0 {
            retry_state.pending_stream_rebuild = true;
        }
        if retry_state.pending_stream_rebuild {
            AUDIO_STREAM.with(|slot| {
                *slot.borrow_mut() = None;
            });
            commands.remove_resource::<AudioOutputResource>();
            retry_state.pending_stream_rebuild = false;
            retry_state.next_retry_at_seconds =
                time.elapsed_secs_f64() + AUDIO_OUTPUT_RETRY_SECONDS;
            warn!(
                "rebuilding cpal audio output after stream failure count={}",
                stream_error_count
            );
        }
        return;
    }

    let now = time.elapsed_secs_f64();
    if now < retry_state.next_retry_at_seconds {
        return;
    }

    match build_audio_output(&config) {
        Ok((resource, stream)) => {
            AUDIO_STREAM.with(|slot| {
                *slot.borrow_mut() = Some(stream);
            });
            commands.insert_resource(resource);
            retry_state.next_retry_at_seconds = 0.0;
        }
        Err(error) => {
            retry_state.next_retry_at_seconds = now + AUDIO_OUTPUT_RETRY_SECONDS;
            warn!("retrying cpal audio initialization later: {error:#}");
        }
    }
}

fn build_audio_output(config: &RepositoryConfig) -> Result<(AudioOutputResource, cpal::Stream)> {
    let host = cpal::default_host();
    let default_device = with_suppressed_stderr(|| {
        host.default_output_device()
            .context("no default output device available")
    })?;
    let output_devices = with_suppressed_stderr(|| {
        host.output_devices()
            .context("failed to enumerate output devices")
            .map(|devices| devices.collect::<Vec<_>>())
    })?;
    let device = select_preferred_device(
        output_devices,
        Some(&default_device),
        config.game.audio.preferred_output_device.as_deref(),
    )
    .context("failed to select an output device")?;
    let device_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

    let supported = with_suppressed_stderr(|| {
        device
            .supported_output_configs()
            .context("failed to query supported output configs")
            .map(|configs| configs.collect::<Vec<_>>())
    })?;
    let preferred = pick_surround_or_stereo_config(&supported)
        .context("failed to find a usable stereo-or-better output config")?;
    let sample_format = preferred.sample_format();
    let stream_config = StreamConfig {
        channels: preferred.channels(),
        sample_rate: pick_reasonable_sample_rate(preferred),
        buffer_size: cpal::BufferSize::Default,
    };

    let runtime = Arc::new(Mutex::new(RuntimeAudioState::new(
        config,
        stream_config.sample_rate.0 as f32,
    )));
    let runtime_for_stream = runtime.clone();
    let stream_error_count = Arc::new(AtomicU32::new(0));
    let stream_error_counter = stream_error_count.clone();
    let err_fn = move |err| {
        stream_error_counter.fetch_add(1, Ordering::Relaxed);
        warn!("audio stream error: {err}");
    };
    let stream = match sample_format {
        SampleFormat::F32 => {
            build_stream::<f32>(&device, &stream_config, runtime_for_stream, err_fn)?
        }
        SampleFormat::I16 => {
            build_stream::<i16>(&device, &stream_config, runtime_for_stream, err_fn)?
        }
        SampleFormat::U16 => {
            build_stream::<u16>(&device, &stream_config, runtime_for_stream, err_fn)?
        }
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    };
    stream.play()?;

    info!(
        "initialized cpal audio on device '{}' channels={} sample_rate={}",
        device_name, stream_config.channels, stream_config.sample_rate.0
    );

    Ok((
        AudioOutputResource {
            state: runtime,
            stream_error_count,
        },
        stream,
    ))
}

#[cfg(target_os = "linux")]
fn with_suppressed_stderr<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    let devnull = OpenOptions::new()
        .write(true)
        .open("/dev/null")
        .context("failed to open /dev/null for stderr suppression")?;
    let stderr_fd = std::io::stderr().as_raw_fd();
    // SAFETY: dup/dup2/close are called with valid file descriptors owned by this process.
    let saved_stderr = unsafe { libc::dup(stderr_fd) };
    if saved_stderr < 0 {
        return f();
    }

    // SAFETY: redirecting process stderr to /dev/null for the duration of the probe.
    unsafe {
        libc::dup2(devnull.as_raw_fd(), stderr_fd);
    }
    let result = f();
    // SAFETY: restore original stderr and close the duplicated fd after the probe.
    unsafe {
        libc::dup2(saved_stderr, stderr_fd);
        libc::close(saved_stderr);
    }
    result
}

#[cfg(not(target_os = "linux"))]
fn with_suppressed_stderr<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    f()
}

fn sync_audio_scene_state(
    audio: Option<Res<AudioOutputResource>>,
    config: Res<RepositoryConfig>,
    time: Res<Time>,
    audio_mute: Res<AudioMuteState>,
    observed_role: Option<Res<ObservedAircraftRole>>,
    listener_override: Option<Res<ObserverAudioListenerOverride>>,
    mut events: ResMut<AudioEventQueue>,
    aircraft_query: Query<(&AircraftRole, &AircraftState)>,
    mut observation_audio: ResMut<AudioObservationState>,
) {
    observation_audio.sample_rate = 48_000.0;
    let observer_audio_suspended = listener_override
        .as_deref()
        .is_some_and(|override_state| override_state.suspend_audio);
    let camera_listener_override = listener_override
        .as_deref()
        .filter(|override_state| {
            override_state.use_camera_listener && !override_state.suspend_audio
        })
        .copied();

    if let Some(audio) = audio.as_ref()
        && let Ok(mut runtime) = audio.state.lock()
    {
        runtime.observed_role = observed_role
            .as_deref()
            .map(|role| role.0)
            .unwrap_or(AircraftRole::Fighter1);
        runtime.external_listener_mix = camera_listener_override.is_some();
        runtime.suspend_audio = observer_audio_suspended;
        runtime.master_volume = if audio_mute.muted {
            0.0
        } else {
            config.game.audio.master_volume
        };
        runtime.player_engine_volume = config.game.audio.player_engine_volume;
        runtime.enemy_engine_volume = config.game.audio.enemy_engine_volume;
        runtime.gun_fire_volume = config.game.audio.gun_fire_volume;
        runtime.bullet_flyby_volume = config.game.audio.bullet_flyby_volume;
        runtime.hit_volume = config.game.audio.hit_volume;
        runtime.spatial_near_distance = config.game.audio.spatial_near_distance.max(1.0);
        runtime.spatial_far_distance = config
            .game
            .audio
            .spatial_far_distance
            .max(runtime.spatial_near_distance + 1.0);
    }

    if observer_audio_suspended {
        suspend_observer_audio_state(&mut observation_audio);
        return;
    }

    for event in events.events.drain(..) {
        let base_gain = match event.kind {
            AudioEventKind::GunFire => config.game.audio.gun_fire_volume,
            AudioEventKind::BulletFlyBy => config.game.audio.bullet_flyby_volume,
            AudioEventKind::Hit => config.game.audio.hit_volume,
        } * event.volume.max(0.0);
        let shot = OneShotEmitterState {
            kind: event.kind,
            position: event.position,
            gain: base_gain,
            age: 0.0,
            duration: oneshot_duration(event.kind),
            phase: 0.0,
        };
        observation_audio.one_shots.push(shot);
        if let Some(audio) = audio.as_ref()
            && let Ok(mut runtime) = audio.state.lock()
        {
            runtime.one_shots.push(shot);
        }
    }

    let alpha = smoothing_alpha(config.game.audio.spatial_smoothing, time.delta_secs());

    let observed = observed_role
        .as_deref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    observation_audio.observed_role = observed;
    observation_audio.external_listener_mix = camera_listener_override.is_some();
    if let Some(listener_override) = camera_listener_override {
        let target_listener = listener_from_override(&listener_override);
        observation_audio.listener =
            smooth_listener(observation_audio.listener, target_listener, alpha);
        if let Some(audio) = audio.as_deref()
            && let Ok(mut runtime) = audio.state.lock()
        {
            runtime.listener = smooth_listener(runtime.listener, target_listener, alpha);
        }
    }
    for (role, state) in &aircraft_query {
        apply_engine_voice_sample(
            *role,
            state,
            observed,
            camera_listener_override.is_none(),
            alpha,
            &mut observation_audio,
            audio.as_deref(),
        );
    }
    decay_missing_engine_voices(
        time.delta_secs(),
        &aircraft_query,
        &mut observation_audio,
        audio.as_deref(),
    );
}

fn toggle_audio_mute(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    mut audio_mute: ResMut<AudioMuteState>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings)) = (keyboard, mouse_buttons, bindings)
    else {
        return;
    };
    if bindings
        .toggle_audio_mute
        .just_pressed(&keyboard, &mouse_buttons)
    {
        audio_mute.muted = !audio_mute.muted;
    }
}

fn diagnose_audio_runtime(
    link_state: Option<Res<BridgeLinkState>>,
    match_phase: Option<Res<State<MatchPhase>>>,
    audio: Option<Res<AudioOutputResource>>,
    aircraft_query: Query<(&AircraftRole, &AircraftState)>,
    observation_audio: Res<AudioObservationState>,
    mut diagnostics: ResMut<AudioAnomalyDiagnostics>,
) {
    if link_state
        .as_deref()
        .is_some_and(|state| state.client_status != BridgeClientConnectionStatus::Connected)
    {
        diagnostics.last_logged = None;
        return;
    }

    if match_phase
        .as_deref()
        .is_some_and(|phase| *phase.get() != MatchPhase::Running)
    {
        diagnostics.last_logged = None;
        return;
    }

    let mut fighter1_state = None;
    let mut fighter2_state = None;
    for (role, state) in &aircraft_query {
        match role {
            AircraftRole::Fighter1 => fighter1_state = Some(state.clone()),
            AircraftRole::Fighter2 => fighter2_state = Some(state.clone()),
        }
    }
    if fighter1_state.is_none() && fighter2_state.is_none() {
        diagnostics.last_logged = None;
        return;
    }

    let fighter1_voice = observation_audio.voices.get(AircraftRole::Fighter1);
    let fighter2_voice = observation_audio.voices.get(AircraftRole::Fighter2);
    let fighter1_missing = fighter1_state
        .as_ref()
        .is_some_and(|state| !state.is_destroyed && state.throttle > 0.05)
        && !fighter1_voice.emitter.alive;
    let fighter2_missing = fighter2_state
        .as_ref()
        .is_some_and(|state| !state.is_destroyed && state.throttle > 0.05)
        && !fighter2_voice.emitter.alive;
    let stream_error_count = audio
        .as_deref()
        .map(|resource| resource.stream_error_count.load(Ordering::Relaxed))
        .unwrap_or(0);
    let has_anomaly = fighter1_missing || fighter2_missing;
    let snapshot = AudioEmitterAnomalySnapshot {
        fighter1_missing,
        fighter2_missing,
        stream_error_count,
    };
    if !has_anomaly {
        diagnostics.last_logged = None;
        diagnostics.active_snapshot = None;
        diagnostics.active_seconds = 0.0;
        return;
    }
    if diagnostics.active_snapshot == Some(snapshot) {
        // timer advanced in self_heal_audio_runtime
    } else {
        diagnostics.active_snapshot = Some(snapshot);
        diagnostics.active_seconds = 0.0;
    }
    if diagnostics.last_logged == Some(snapshot) {
        return;
    }

    let fighter1_status = fighter1_state
        .map(|state| {
            format!(
                "alive={} throttle={:.2}",
                !state.is_destroyed, state.throttle
            )
        })
        .unwrap_or_else(|| "missing".to_string());
    let fighter2_status = fighter2_state
        .map(|state| {
            format!(
                "alive={} throttle={:.2}",
                !state.is_destroyed, state.throttle
            )
        })
        .unwrap_or_else(|| "missing".to_string());
    warn!(
        "audio anomaly: fighter1_missing={} fighter2_missing={} stream_errors={} fighter1_state=({}) fighter2_state=({}) fighter1_voice=(alive={} throttle={:.2} missing={:.2}) fighter2_voice=(alive={} throttle={:.2} missing={:.2})",
        fighter1_missing,
        fighter2_missing,
        stream_error_count,
        fighter1_status,
        fighter2_status,
        fighter1_voice.emitter.alive,
        fighter1_voice.emitter.throttle,
        fighter1_voice.missing_seconds,
        fighter2_voice.emitter.alive,
        fighter2_voice.emitter.throttle,
        fighter2_voice.missing_seconds,
    );
    diagnostics.last_logged = Some(snapshot);
}

fn self_heal_audio_runtime(
    time: Res<Time>,
    link_state: Option<Res<BridgeLinkState>>,
    match_phase: Option<Res<State<MatchPhase>>>,
    mut observation_audio: ResMut<AudioObservationState>,
    audio: Option<Res<AudioOutputResource>>,
    mut diagnostics: ResMut<AudioAnomalyDiagnostics>,
) {
    if link_state
        .as_deref()
        .is_some_and(|state| state.client_status != BridgeClientConnectionStatus::Connected)
        || match_phase
            .as_deref()
            .is_some_and(|phase| *phase.get() != MatchPhase::Running)
    {
        diagnostics.active_snapshot = None;
        diagnostics.active_seconds = 0.0;
        return;
    }

    if diagnostics.active_snapshot.is_none() {
        diagnostics.active_seconds = 0.0;
        return;
    }

    diagnostics.active_seconds += time.delta_secs();
    if diagnostics.active_seconds < 1.0 {
        return;
    }

    warn!(
        "self-healing audio runtime after persistent anomaly for {:.2}s",
        diagnostics.active_seconds
    );
    if let Some(snapshot) = diagnostics.active_snapshot {
        if snapshot.fighter1_missing {
            reset_voice_runtime(
                AircraftRole::Fighter1,
                &mut observation_audio,
                audio.as_deref(),
            );
        }
        if snapshot.fighter2_missing {
            reset_voice_runtime(
                AircraftRole::Fighter2,
                &mut observation_audio,
                audio.as_deref(),
            );
        }
    }
    diagnostics.last_logged = None;
    diagnostics.active_snapshot = None;
    diagnostics.active_seconds = 0.0;
}

pub fn queue_capture_audio_events(world: &mut World, events: &[EnvironmentEvent]) {
    let mut audio_events = world.resource_mut::<AudioEventQueue>();
    for event in events {
        let Some(position) = event.position.map(Vec3::from_array) else {
            continue;
        };
        let magnitude = event.magnitude.unwrap_or(1.0).max(0.0);
        match event.kind.as_str() {
            "GunFired" => audio_events.push(AudioEventKind::GunFire, position, magnitude.max(0.4)),
            "Hit" | "Damage" | "Collision" | "SubsystemHit" | "SubsystemDestroyed" => {
                audio_events.push(AudioEventKind::Hit, position, magnitude.max(0.35));
            }
            "Destroy" => audio_events.push(AudioEventKind::Hit, position, 1.0),
            _ => {}
        }
    }
}

pub fn accumulate_audio_capture_step(world: &mut World) -> Result<()> {
    let mut sync_state: bevy::ecs::system::SystemState<(
        Res<RepositoryConfig>,
        Option<Res<ObservedAircraftRole>>,
        ResMut<AudioEventQueue>,
        Query<(&AircraftRole, &AircraftState)>,
        ResMut<AudioObservationState>,
    )> = bevy::ecs::system::SystemState::new(world);
    let (config, observed_role, mut events, aircraft_query, mut observation_audio) =
        sync_state.get_mut(world);

    let observed = observed_role
        .as_deref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    observation_audio.observed_role = observed;
    observation_audio.external_listener_mix = false;
    let alpha = smoothing_alpha(
        config.game.audio.spatial_smoothing,
        config.game.fixed_time_step_seconds,
    );

    for event in events.events.drain(..) {
        let base_gain = match event.kind {
            AudioEventKind::GunFire => config.game.audio.gun_fire_volume,
            AudioEventKind::BulletFlyBy => config.game.audio.bullet_flyby_volume,
            AudioEventKind::Hit => config.game.audio.hit_volume,
        } * event.volume.max(0.0);
        observation_audio.one_shots.push(OneShotEmitterState {
            kind: event.kind,
            position: event.position,
            gain: base_gain,
            age: 0.0,
            duration: oneshot_duration(event.kind),
            phase: 0.0,
        });
    }

    for (role, state) in &aircraft_query {
        apply_engine_voice_sample_capture(
            *role,
            state,
            observed,
            true,
            alpha,
            &mut observation_audio,
        );
    }
    decay_missing_engine_voices_capture(
        config.game.fixed_time_step_seconds,
        &aircraft_query,
        &mut observation_audio,
    );
    sync_state.apply(world);

    let mut synth_state: bevy::ecs::system::SystemState<(
        Res<RepositoryConfig>,
        Res<ObservationCaptureConfig>,
        ResMut<AudioObservationState>,
    )> = bevy::ecs::system::SystemState::new(world);
    let (config, capture_config, mut observation_audio) = synth_state.get_mut(world);
    if !capture_config.enable_audio {
        return Ok(());
    }

    observation_audio.max_seconds = AUDIO_OBSERVATION_MAX_SECONDS;
    let sample_rate = observation_audio.sample_rate.max(8_000.0);
    let frames =
        (sample_rate * config.game.fixed_time_step_seconds.max(1.0 / 240.0)).round() as usize;
    let near_distance = config.game.audio.spatial_near_distance.max(1.0);
    let far_distance = config
        .game
        .audio
        .spatial_far_distance
        .max(near_distance + 1.0);
    let max_samples = (sample_rate * observation_audio.max_seconds) as usize
        * AUDIO_OBSERVATION_CHANNELS as usize;
    let max_feature_frames = (sample_rate * observation_audio.max_seconds) as usize;

    for _ in 0..frames.max(1) {
        append_runtime_style_audio_frame(
            &config,
            &mut observation_audio,
            sample_rate,
            near_distance,
            far_distance,
            max_samples,
            max_feature_frames,
        );
    }
    Ok(())
}

fn synthesize_audio_observation(
    config: Res<RepositoryConfig>,
    capture_config: Res<ObservationCaptureConfig>,
    mut observation_audio: ResMut<AudioObservationState>,
) {
    if !capture_config.enable_audio {
        return;
    }

    observation_audio.max_seconds = AUDIO_OBSERVATION_MAX_SECONDS;
    let sample_rate = observation_audio.sample_rate.max(8_000.0);
    let frames =
        (sample_rate * config.game.fixed_time_step_seconds.max(1.0 / 240.0)).round() as usize;
    let near_distance = config.game.audio.spatial_near_distance.max(1.0);
    let far_distance = config
        .game
        .audio
        .spatial_far_distance
        .max(near_distance + 1.0);
    let max_samples = (sample_rate * observation_audio.max_seconds) as usize
        * AUDIO_OBSERVATION_CHANNELS as usize;
    let max_feature_frames = (sample_rate * observation_audio.max_seconds) as usize;

    for _ in 0..frames.max(1) {
        append_runtime_style_audio_frame(
            &config,
            &mut observation_audio,
            sample_rate,
            near_distance,
            far_distance,
            max_samples,
            max_feature_frames,
        );
    }
}

pub fn collect_audio_observation(
    world: &World,
    capture_config: &ObservationCaptureConfig,
) -> Option<AudioObservation> {
    if !capture_config.enable_audio {
        return None;
    }

    let observation_audio = world.get_resource::<AudioObservationState>()?;
    let requested_frames = (observation_audio.sample_rate
        * capture_config.audio_window_seconds.max(0.01))
    .round() as usize;
    let requested_samples = requested_frames * AUDIO_OBSERVATION_CHANNELS as usize;
    let available_samples = observation_audio.recent_samples.len();
    let start = available_samples.saturating_sub(requested_samples);
    let samples = observation_audio
        .recent_samples
        .iter()
        .skip(start)
        .copied()
        .collect::<Vec<_>>();

    let feature_start = observation_audio
        .recent_features
        .len()
        .saturating_sub(requested_frames);
    let features =
        aggregate_audio_features(observation_audio.recent_features.iter().skip(feature_start));

    Some(AudioObservation {
        sample_rate: observation_audio.sample_rate.round() as u32,
        channels: AUDIO_OBSERVATION_CHANNELS,
        window_seconds: capture_config.audio_window_seconds,
        samples,
        features,
    })
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    state: Arc<Mutex<RuntimeAudioState>>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = config.channels as usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let Ok(mut state) = state.lock() else {
                return;
            };
            if state.suspend_audio {
                for sample in data.iter_mut() {
                    *sample = T::from_sample(0.0);
                }
                return;
            }
            let sample_rate = state.sample_rate;
            let listener = state.listener;
            let near_distance = state.spatial_near_distance;
            let far_distance = state.spatial_far_distance;
            let master_volume = state.master_volume;

            for frame in data.chunks_mut(channels) {
                let mut output = [0.0_f32; 8];
                mix_runtime_engine_voices(
                    &mut output,
                    &mut state,
                    sample_rate,
                    listener,
                    near_distance,
                    far_distance,
                );
                mix_one_shots(
                    &mut output,
                    &mut state.one_shots,
                    sample_rate,
                    listener,
                    near_distance,
                    far_distance,
                    master_volume,
                );

                for sample in frame.iter_mut() {
                    *sample = T::from_sample(0.0);
                }

                for (index, sample) in frame.iter_mut().enumerate() {
                    let channel_value = if index < output.len() {
                        output[index]
                    } else {
                        0.0
                    };
                    *sample = T::from_sample(channel_value);
                }
            }
        },
        err_fn,
        None,
    )?;
    Ok(stream)
}

fn mix_one_shots(
    output: &mut [f32; 8],
    one_shots: &mut Vec<OneShotEmitterState>,
    sample_rate: f32,
    listener: ListenerState,
    near_distance: f32,
    far_distance: f32,
    master_gain: f32,
) {
    one_shots.retain_mut(|emitter| {
        if emitter.age >= emitter.duration {
            return false;
        }

        let sample = next_oneshot_sample(emitter, sample_rate);
        let relative = emitter.position - listener.position;
        let distance = relative.length();
        if distance <= f32::EPSILON {
            for (index, gain) in coincident_oneshot_gains().into_iter().enumerate() {
                output[index] += sample * gain * master_gain;
            }
        } else {
            let direction = relative / distance;
            let right = listener.right.normalize_or_zero();
            let forward = listener.forward.normalize_or_zero();
            let azimuth = direction.dot(right).atan2(direction.dot(forward));
            let attenuation = distance_attenuation(distance, near_distance, far_distance);
            let gains = surround_gains(azimuth, attenuation);
            for (index, gain) in gains.into_iter().enumerate() {
                output[index] += sample * gain * master_gain;
            }
        }

        emitter.age += 1.0 / sample_rate.max(1.0);
        emitter.age < emitter.duration
    });
}

fn mix_one_shots_runtime_split(
    gun_output: &mut [f32; 8],
    flyby_output: &mut [f32; 8],
    hit_output: &mut [f32; 8],
    one_shots: &mut Vec<OneShotEmitterState>,
    sample_rate: f32,
    listener: ListenerState,
    near_distance: f32,
    far_distance: f32,
    master_gain: f32,
) {
    one_shots.retain_mut(|emitter| {
        if emitter.age >= emitter.duration {
            return false;
        }

        let sample = next_oneshot_sample(emitter, sample_rate);
        let relative = emitter.position - listener.position;
        let distance = relative.length();
        if distance <= f32::EPSILON {
            for (index, gain) in coincident_oneshot_gains().into_iter().enumerate() {
                let value = sample * gain * master_gain;
                match emitter.kind {
                    AudioEventKind::GunFire => gun_output[index] += value,
                    AudioEventKind::BulletFlyBy => flyby_output[index] += value,
                    AudioEventKind::Hit => hit_output[index] += value,
                }
            }
        } else {
            let direction = relative / distance;
            let right = listener.right.normalize_or_zero();
            let forward = listener.forward.normalize_or_zero();
            let azimuth = direction.dot(right).atan2(direction.dot(forward));
            let attenuation = distance_attenuation(distance, near_distance, far_distance);
            let gains = surround_gains(azimuth, attenuation);
            for (index, gain) in gains.into_iter().enumerate() {
                let value = sample * gain * master_gain;
                match emitter.kind {
                    AudioEventKind::GunFire => gun_output[index] += value,
                    AudioEventKind::BulletFlyBy => flyby_output[index] += value,
                    AudioEventKind::Hit => hit_output[index] += value,
                }
            }
        }

        emitter.age += 1.0 / sample_rate.max(1.0);
        emitter.age < emitter.duration
    });
}

fn coincident_oneshot_gains() -> [f32; 8] {
    [0.7, 0.7, 1.0, 0.0, 0.15, 0.15, 0.2, 0.2]
}

fn mix_engine_emitter(
    output: &mut [f32; 8],
    phase: &mut f32,
    sample_rate: f32,
    emitter: EngineEmitterState,
    listener: ListenerState,
    base_gain: f32,
    spatialize: bool,
    near_distance: f32,
    far_distance: f32,
) {
    if !emitter.alive {
        return;
    }

    let throttle = emitter.throttle.clamp(0.0, 1.0);
    let engine_sample = next_engine_sample(phase, sample_rate, throttle) * base_gain;
    if !spatialize {
        output[FRONT_LEFT] += engine_sample * 0.55;
        output[FRONT_RIGHT] += engine_sample * 0.55;
        output[FRONT_CENTER] += engine_sample * 0.65;
        return;
    }

    let relative = emitter.position - listener.position;
    let distance = relative.length();
    if distance <= f32::EPSILON {
        return;
    }

    let direction = relative / distance;
    let right = listener.right.normalize_or_zero();
    let forward = listener.forward.normalize_or_zero();
    let azimuth = direction.dot(right).atan2(direction.dot(forward));
    let attenuation = distance_attenuation(distance, near_distance, far_distance);
    let gains = surround_gains(azimuth, attenuation);

    for (index, gain) in gains.into_iter().enumerate() {
        output[index] += engine_sample * gain;
    }
}

fn next_engine_sample(phase: &mut f32, sample_rate: f32, throttle: f32) -> f32 {
    let rpm = 55.0 + throttle * 40.0;
    *phase = (*phase + TAU * rpm / sample_rate).rem_euclid(TAU);
    let harmonic = *phase * 2.03;
    let overtone = *phase * 3.91;
    ((*phase).sin() * 0.58 + harmonic.sin() * 0.28 + overtone.sin() * 0.14).tanh() * 0.36
}

fn oneshot_duration(kind: AudioEventKind) -> f32 {
    match kind {
        AudioEventKind::GunFire => 0.11,
        AudioEventKind::BulletFlyBy => 0.18,
        AudioEventKind::Hit => 0.16,
    }
}

fn next_oneshot_sample(emitter: &mut OneShotEmitterState, sample_rate: f32) -> f32 {
    let t = emitter.age / emitter.duration.max(0.001);
    let envelope = (1.0 - t).clamp(0.0, 1.0);
    let freq = match emitter.kind {
        AudioEventKind::GunFire => 420.0 + (1.0 - t) * 900.0,
        AudioEventKind::BulletFlyBy => 1200.0 - t * 700.0,
        AudioEventKind::Hit => 220.0 + (1.0 - t) * 260.0,
    };
    emitter.phase += TAU * freq / sample_rate.max(1.0);
    let base = match emitter.kind {
        AudioEventKind::GunFire => emitter.phase.sin() * 0.7 + (emitter.phase * 2.7).sin() * 0.3,
        AudioEventKind::BulletFlyBy => {
            (emitter.phase * 1.8).sin() * 0.35 + emitter.phase.sin() * 0.65
        }
        AudioEventKind::Hit => emitter.phase.sin() * 0.5 + (emitter.phase * 0.5).sin() * 0.5,
    };
    base.tanh() * envelope * emitter.gain
}

fn surround_gains(azimuth: f32, attenuation: f32) -> [f32; 8] {
    let speaker_angles = [
        30.0_f32.to_radians(),
        -30.0_f32.to_radians(),
        0.0,
        0.0,
        150.0_f32.to_radians(),
        -150.0_f32.to_radians(),
        90.0_f32.to_radians(),
        -90.0_f32.to_radians(),
    ];

    let mut gains = [0.0; 8];
    let mut sum = 0.0;
    for (index, speaker_angle) in speaker_angles.into_iter().enumerate() {
        if index == LFE {
            continue;
        }
        let wrapped =
            (azimuth - speaker_angle + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
        let gain = wrapped.cos().max(0.0).powi(2);
        gains[index] = gain;
        sum += gain;
    }
    if sum > f32::EPSILON {
        let normalization = attenuation / sum.sqrt();
        for (index, gain) in gains.iter_mut().enumerate() {
            if index != LFE {
                *gain *= normalization;
            }
        }
    }
    gains[LFE] = 0.0;
    gains
}

fn distance_attenuation(distance: f32, near_distance: f32, far_distance: f32) -> f32 {
    if distance <= near_distance {
        return 1.0;
    }

    let normalized = ((distance - near_distance) / (far_distance - near_distance)).clamp(0.0, 1.0);
    (1.0 - normalized).powi(2)
}

fn smoothing_alpha(smoothing: f32, dt: f32) -> f32 {
    1.0 - (-smoothing.max(0.0) * dt.max(0.0)).exp()
}

fn smooth_listener(current: ListenerState, target: ListenerState, alpha: f32) -> ListenerState {
    ListenerState {
        position: current.position.lerp(target.position, alpha),
        forward: current
            .forward
            .lerp(target.forward, alpha)
            .normalize_or_zero(),
        right: current.right.lerp(target.right, alpha).normalize_or_zero(),
    }
}

fn smooth_emitter(
    current: EngineEmitterState,
    target: EngineEmitterState,
    alpha: f32,
) -> EngineEmitterState {
    EngineEmitterState {
        position: current.position.lerp(target.position, alpha),
        throttle: current.throttle.lerp(target.throttle, alpha),
        alive: target.alive,
    }
}

fn opposing_role(role: AircraftRole) -> AircraftRole {
    match role {
        AircraftRole::Fighter1 => AircraftRole::Fighter2,
        AircraftRole::Fighter2 => AircraftRole::Fighter1,
    }
}

fn listener_from_aircraft_state(state: &AircraftState) -> ListenerState {
    ListenerState {
        position: state.position,
        forward: (state.orientation * Vec3::Z).normalize_or_zero(),
        right: (state.orientation * Vec3::X).normalize_or_zero(),
    }
}

fn listener_from_override(listener_override: &ObserverAudioListenerOverride) -> ListenerState {
    ListenerState {
        position: listener_override.position,
        forward: listener_override.forward.normalize_or_zero(),
        right: listener_override.right.normalize_or_zero(),
    }
}

fn emitter_from_aircraft_state(state: &AircraftState) -> EngineEmitterState {
    EngineEmitterState {
        position: state.position,
        throttle: state.throttle.clamp(0.0, 1.0),
        alive: !state.is_destroyed,
    }
}

fn reset_voice_runtime(
    role: AircraftRole,
    observation_audio: &mut AudioObservationState,
    audio: Option<&AudioOutputResource>,
) {
    *observation_audio.voices.get_mut(role) = ObservedEngineVoiceState::default();
    if let Some(audio) = audio
        && let Ok(mut runtime) = audio.state.lock()
    {
        *runtime.voices.get_mut(role) = RuntimeEngineVoiceState::default();
    }
}

fn apply_engine_voice_sample(
    role: AircraftRole,
    state: &AircraftState,
    observed: AircraftRole,
    lock_listener_to_observed: bool,
    alpha: f32,
    observation_audio: &mut AudioObservationState,
    audio: Option<&AudioOutputResource>,
) {
    if lock_listener_to_observed && role == observed {
        let target_listener = listener_from_aircraft_state(state);
        observation_audio.listener =
            smooth_listener(observation_audio.listener, target_listener, alpha);
        if let Some(audio) = audio
            && let Ok(mut runtime) = audio.state.lock()
        {
            runtime.listener = smooth_listener(runtime.listener, target_listener, alpha);
        }
    }
    let target_emitter = emitter_from_aircraft_state(state);
    let voice = observation_audio.voices.get_mut(role);
    voice.missing_seconds = 0.0;
    voice.emitter = smooth_emitter(voice.emitter, target_emitter, alpha);
    if let Some(audio) = audio
        && let Ok(mut runtime) = audio.state.lock()
    {
        let voice = runtime.voices.get_mut(role);
        voice.missing_seconds = 0.0;
        voice.emitter = smooth_emitter(voice.emitter, target_emitter, alpha);
    }
}

fn apply_engine_voice_sample_capture(
    role: AircraftRole,
    state: &AircraftState,
    observed: AircraftRole,
    lock_listener_to_observed: bool,
    alpha: f32,
    observation_audio: &mut AudioObservationState,
) {
    if lock_listener_to_observed && role == observed {
        let target_listener = listener_from_aircraft_state(state);
        observation_audio.listener =
            smooth_listener(observation_audio.listener, target_listener, alpha);
    }
    let target_emitter = emitter_from_aircraft_state(state);
    let voice = observation_audio.voices.get_mut(role);
    voice.missing_seconds = 0.0;
    voice.emitter = smooth_emitter(voice.emitter, target_emitter, alpha);
}

fn decay_missing_engine_voices(
    delta_seconds: f32,
    aircraft_query: &Query<(&AircraftRole, &AircraftState)>,
    observation_audio: &mut AudioObservationState,
    audio: Option<&AudioOutputResource>,
) {
    for (role, seen) in [
        (
            AircraftRole::Fighter1,
            aircraft_query
                .iter()
                .any(|(aircraft_role, _)| *aircraft_role == AircraftRole::Fighter1),
        ),
        (
            AircraftRole::Fighter2,
            aircraft_query
                .iter()
                .any(|(aircraft_role, _)| *aircraft_role == AircraftRole::Fighter2),
        ),
    ] {
        if !seen {
            let voice = observation_audio.voices.get_mut(role);
            voice.missing_seconds += delta_seconds;
            if voice.missing_seconds > 0.35 {
                voice.emitter.alive = false;
            }
            if let Some(audio) = audio
                && let Ok(mut runtime) = audio.state.lock()
            {
                let emitter_presence_grace = runtime.emitter_presence_grace;
                let voice = runtime.voices.get_mut(role);
                voice.missing_seconds += delta_seconds;
                if voice.missing_seconds > emitter_presence_grace {
                    voice.emitter.alive = false;
                }
            }
        }
    }
}

fn decay_missing_engine_voices_capture(
    delta_seconds: f32,
    aircraft_query: &Query<(&AircraftRole, &AircraftState)>,
    observation_audio: &mut AudioObservationState,
) {
    for (role, seen) in [
        (
            AircraftRole::Fighter1,
            aircraft_query
                .iter()
                .any(|(aircraft_role, _)| *aircraft_role == AircraftRole::Fighter1),
        ),
        (
            AircraftRole::Fighter2,
            aircraft_query
                .iter()
                .any(|(aircraft_role, _)| *aircraft_role == AircraftRole::Fighter2),
        ),
    ] {
        if !seen {
            let voice = observation_audio.voices.get_mut(role);
            voice.missing_seconds += delta_seconds;
            if voice.missing_seconds > 0.35 {
                voice.emitter.alive = false;
            }
        }
    }
}

fn mix_observation_engine_voices(
    output: &mut [f32; 8],
    observation_audio: &mut AudioObservationState,
    sample_rate: f32,
    listener: ListenerState,
    config: &RepositoryConfig,
    near_distance: f32,
    far_distance: f32,
) {
    if observation_audio.external_listener_mix {
        let external_gain = config.game.audio.master_volume * config.game.audio.enemy_engine_volume;
        mix_engine_emitter(
            output,
            &mut observation_audio.voices.fighter1.phase,
            sample_rate,
            observation_audio.voices.fighter1.emitter,
            listener,
            external_gain,
            true,
            near_distance,
            far_distance,
        );
        mix_engine_emitter(
            output,
            &mut observation_audio.voices.fighter2.phase,
            sample_rate,
            observation_audio.voices.fighter2.emitter,
            listener,
            external_gain,
            true,
            near_distance,
            far_distance,
        );
        return;
    }

    let observed = observation_audio.observed_role;
    let opponent = opposing_role(observed);
    let player = observation_audio.voices.get(observed).emitter;
    let enemy = observation_audio.voices.get(opponent).emitter;
    match observed {
        AircraftRole::Fighter1 => {
            mix_engine_emitter(
                output,
                &mut observation_audio.voices.fighter1.phase,
                sample_rate,
                player,
                listener,
                config.game.audio.master_volume * config.game.audio.player_engine_volume,
                false,
                near_distance,
                far_distance,
            );
            mix_engine_emitter(
                output,
                &mut observation_audio.voices.fighter2.phase,
                sample_rate,
                enemy,
                listener,
                config.game.audio.master_volume * config.game.audio.enemy_engine_volume,
                true,
                near_distance,
                far_distance,
            );
        }
        AircraftRole::Fighter2 => {
            mix_engine_emitter(
                output,
                &mut observation_audio.voices.fighter2.phase,
                sample_rate,
                player,
                listener,
                config.game.audio.master_volume * config.game.audio.player_engine_volume,
                false,
                near_distance,
                far_distance,
            );
            mix_engine_emitter(
                output,
                &mut observation_audio.voices.fighter1.phase,
                sample_rate,
                enemy,
                listener,
                config.game.audio.master_volume * config.game.audio.enemy_engine_volume,
                true,
                near_distance,
                far_distance,
            );
        }
    }
}

fn mix_runtime_engine_voices(
    output: &mut [f32; 8],
    state: &mut RuntimeAudioState,
    sample_rate: f32,
    listener: ListenerState,
    near_distance: f32,
    far_distance: f32,
) {
    if state.external_listener_mix {
        let external_gain = state.master_volume * state.enemy_engine_volume;
        mix_engine_emitter(
            output,
            &mut state.voices.fighter1.phase,
            sample_rate,
            state.voices.fighter1.emitter,
            listener,
            external_gain,
            true,
            near_distance,
            far_distance,
        );
        mix_engine_emitter(
            output,
            &mut state.voices.fighter2.phase,
            sample_rate,
            state.voices.fighter2.emitter,
            listener,
            external_gain,
            true,
            near_distance,
            far_distance,
        );
        return;
    }

    let observed = state.observed_role;
    let opponent = opposing_role(observed);
    let player = state.voices.get(observed).emitter;
    let enemy = state.voices.get(opponent).emitter;
    let player_gain = state.master_volume * state.player_engine_volume;
    let enemy_gain = state.master_volume * state.enemy_engine_volume;
    match observed {
        AircraftRole::Fighter1 => {
            mix_engine_emitter(
                output,
                &mut state.voices.fighter1.phase,
                sample_rate,
                player,
                listener,
                player_gain,
                false,
                near_distance,
                far_distance,
            );
            mix_engine_emitter(
                output,
                &mut state.voices.fighter2.phase,
                sample_rate,
                enemy,
                listener,
                enemy_gain,
                true,
                near_distance,
                far_distance,
            );
        }
        AircraftRole::Fighter2 => {
            mix_engine_emitter(
                output,
                &mut state.voices.fighter2.phase,
                sample_rate,
                player,
                listener,
                player_gain,
                false,
                near_distance,
                far_distance,
            );
            mix_engine_emitter(
                output,
                &mut state.voices.fighter1.phase,
                sample_rate,
                enemy,
                listener,
                enemy_gain,
                true,
                near_distance,
                far_distance,
            );
        }
    }
}

fn append_runtime_style_audio_frame(
    config: &RepositoryConfig,
    observation_audio: &mut AudioObservationState,
    sample_rate: f32,
    near_distance: f32,
    far_distance: f32,
    max_samples: usize,
    max_feature_frames: usize,
) {
    let mut full_mix = [0.0_f32; 8];
    let mut engine_mix = [0.0_f32; 8];
    let mut gun_mix = [0.0_f32; 8];
    let mut flyby_mix = [0.0_f32; 8];
    let mut hit_mix = [0.0_f32; 8];
    let listener = observation_audio.listener;
    mix_observation_engine_voices(
        &mut engine_mix,
        observation_audio,
        sample_rate,
        listener,
        config,
        near_distance,
        far_distance,
    );
    mix_one_shots_runtime_split(
        &mut gun_mix,
        &mut flyby_mix,
        &mut hit_mix,
        &mut observation_audio.one_shots,
        sample_rate,
        listener,
        near_distance,
        far_distance,
        config.game.audio.master_volume,
    );

    for index in 0..full_mix.len() {
        full_mix[index] = engine_mix[index] + gun_mix[index] + flyby_mix[index] + hit_mix[index];
    }

    let (left, right) = render_two_channel_audio_head(full_mix);
    observation_audio.recent_samples.push_back(left);
    observation_audio.recent_samples.push_back(right);
    while observation_audio.recent_samples.len() > max_samples {
        observation_audio.recent_samples.pop_front();
    }

    let feature_sample = AudioFeatureSample {
        left_right_energy: (left - right).abs(),
        front_back_energy: front_back_metric(full_mix),
        engine_energy: two_channel_energy(engine_mix),
        gunfire_energy: two_channel_energy(gun_mix),
        hit_energy: two_channel_energy(hit_mix),
        flyby_energy: two_channel_energy(flyby_mix),
    };
    observation_audio.recent_features.push_back(feature_sample);
    while observation_audio.recent_features.len() > max_feature_frames {
        observation_audio.recent_features.pop_front();
    }
}

fn render_two_channel_audio_head(channels: [f32; 8]) -> (f32, f32) {
    // First-pass shared 2-channel render head for both observation export and
    // ordinary 2-channel playback. The internal audio scene remains object-
    // based and 7.1-capable; this head folds the scene into a listener-aware
    // 2-channel output path until a richer HRTF renderer replaces it.
    let center = channels[FRONT_CENTER] * 0.707_106_77;
    let rear_left = channels[4] * 0.707_106_77;
    let rear_right = channels[5] * 0.707_106_77;
    let side_left = channels[6] * 0.707_106_77;
    let side_right = channels[7] * 0.707_106_77;
    (
        channels[FRONT_LEFT] + center + rear_left + side_left,
        channels[FRONT_RIGHT] + center + rear_right + side_right,
    )
}

fn two_channel_energy(channels: [f32; 8]) -> f32 {
    let (left, right) = render_two_channel_audio_head(channels);
    ((left * left + right * right) * 0.5).sqrt()
}

fn front_back_metric(channels: [f32; 8]) -> f32 {
    let front =
        (channels[FRONT_LEFT].abs() + channels[FRONT_RIGHT].abs() + channels[FRONT_CENTER].abs())
            / 3.0;
    let back =
        (channels[4].abs() + channels[5].abs() + channels[6].abs() + channels[7].abs()) / 4.0;
    (front - back).abs()
}

fn aggregate_audio_features<'a>(
    samples: impl Iterator<Item = &'a AudioFeatureSample>,
) -> AudioFeatureVector {
    let mut count = 0.0_f32;
    let mut features = AudioFeatureVector::default();
    for sample in samples {
        count += 1.0;
        features.left_right_energy += sample.left_right_energy;
        features.front_back_energy += sample.front_back_energy;
        features.engine_energy += sample.engine_energy;
        features.gunfire_energy += sample.gunfire_energy;
        features.hit_energy += sample.hit_energy;
        features.flyby_energy += sample.flyby_energy;
    }
    if count > 0.0 {
        features.left_right_energy /= count;
        features.front_back_energy /= count;
        features.engine_energy /= count;
        features.gunfire_energy /= count;
        features.hit_energy /= count;
        features.flyby_energy /= count;
    }
    features
}

fn pick_surround_or_stereo_config(
    supported: &[SupportedStreamConfigRange],
) -> Result<&SupportedStreamConfigRange> {
    supported
        .iter()
        .find(|config| {
            config.channels() >= 8 && matches!(config.sample_format(), SampleFormat::F32)
        })
        .or_else(|| {
            supported.iter().find(|config| {
                config.channels() >= 2 && matches!(config.sample_format(), SampleFormat::F32)
            })
        })
        .or_else(|| supported.iter().find(|config| config.channels() >= 2))
        .context("no stereo-or-better output config found")
}

fn pick_reasonable_sample_rate(range: &SupportedStreamConfigRange) -> cpal::SampleRate {
    let min = range.min_sample_rate().0;
    let max = range.max_sample_rate().0.min(192_000);
    for preferred in [48_000, 44_100, 96_000] {
        if preferred >= min && preferred <= max {
            return cpal::SampleRate(preferred);
        }
    }
    cpal::SampleRate(min.max(8_000).min(max))
}

fn select_preferred_device(
    devices: Vec<cpal::Device>,
    default_device: Option<&cpal::Device>,
    config_preference: Option<&str>,
) -> Result<cpal::Device> {
    let explicit = env::var("DOGFIGHT_AUDIO_DEVICE").ok();
    let requested = explicit.as_deref().or(config_preference);
    let mut named_devices = devices
        .into_iter()
        .filter_map(|device| {
            let name = device.name().ok()?;
            Some((device, name))
        })
        .collect::<Vec<_>>();

    if let Some(requested) = requested
        && let Some(index) = named_devices
            .iter()
            .position(|(_, name)| name.eq_ignore_ascii_case(requested))
    {
        return Ok(named_devices.swap_remove(index).0);
    }

    if let Some(default_device) = default_device {
        let default_name = default_device
            .name()
            .unwrap_or_else(|_| "<unknown>".to_string());
        if let Some(index) = named_devices
            .iter()
            .position(|(_, name)| name == &default_name)
        {
            return Ok(named_devices.swap_remove(index).0);
        }
    }

    let preferred_names = [
        "pipewire",
        "pulse",
        "default:CARD=PCH",
        "sysdefault:CARD=PCH",
        "front:CARD=PCH,DEV=0",
    ];
    for preferred in preferred_names {
        if let Some(index) = named_devices
            .iter()
            .position(|(_, name)| name.eq_ignore_ascii_case(preferred))
        {
            return Ok(named_devices.swap_remove(index).0);
        }
    }

    named_devices
        .into_iter()
        .next()
        .map(|(device, _)| device)
        .context("no output devices available after filtering")
}

#[cfg(test)]
mod tests {
    use super::{
        AudioEventKind, FRONT_CENTER, FRONT_LEFT, FRONT_RIGHT, ListenerState, OneShotEmitterState,
        REAR_LEFT, REAR_RIGHT, SIDE_LEFT, SIDE_RIGHT, distance_attenuation, mix_one_shots,
        oneshot_duration, render_two_channel_audio_head, surround_gains,
    };
    use bevy::prelude::Vec3;
    use std::f32::consts::{FRAC_PI_2, PI};

    #[test]
    fn coincident_gunfire_one_shot_is_mixed() {
        let mut output = [0.0_f32; 8];
        let mut one_shots = vec![OneShotEmitterState {
            kind: AudioEventKind::GunFire,
            position: Vec3::ZERO,
            gain: 1.0,
            age: 0.0,
            duration: oneshot_duration(AudioEventKind::GunFire),
            phase: 0.0,
        }];

        mix_one_shots(
            &mut output,
            &mut one_shots,
            48_000.0,
            ListenerState::default(),
            8.0,
            250.0,
            1.0,
        );

        assert!(output.iter().any(|sample| sample.abs() > f32::EPSILON));
    }

    #[test]
    fn front_target_favors_front_speakers() {
        let gains = surround_gains(0.0, 1.0);
        assert!(gains[FRONT_LEFT] > gains[REAR_LEFT]);
        assert!(gains[FRONT_RIGHT] > gains[REAR_RIGHT]);
    }

    #[test]
    fn right_side_target_favors_right_speakers() {
        let gains = surround_gains(-FRAC_PI_2, 1.0);
        assert!(gains[SIDE_RIGHT] > gains[SIDE_LEFT]);
    }

    #[test]
    fn rear_target_favors_rear_speakers() {
        let gains = surround_gains(PI, 1.0);
        assert!(gains[REAR_LEFT] > gains[FRONT_LEFT]);
        assert!(gains[REAR_RIGHT] > gains[FRONT_RIGHT]);
    }

    #[test]
    fn distance_attenuation_falls_off_with_range() {
        assert_eq!(distance_attenuation(20.0, 80.0, 1200.0), 1.0);
        assert!(distance_attenuation(600.0, 80.0, 1200.0) < 1.0);
        assert_eq!(distance_attenuation(1_500.0, 80.0, 1200.0), 0.0);
    }

    #[test]
    fn two_channel_render_head_preserves_surround_energy() {
        let mut channels = [0.0_f32; 8];
        channels[REAR_LEFT] = 1.0;
        channels[SIDE_RIGHT] = 0.5;
        channels[FRONT_CENTER] = 0.25;

        let (left, right) = render_two_channel_audio_head(channels);

        assert!(
            left > 0.0,
            "rear-left energy should survive two-channel render head"
        );
        assert!(
            right > 0.0,
            "side-right energy should survive two-channel render head"
        );
    }
}
