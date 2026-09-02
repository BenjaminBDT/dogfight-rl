use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
pub mod reconstruct;

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::macros::format_description;

use crate::api::environment::EnvironmentSeed;
use crate::api::types::{
    AircraftObservation, AudioObservation, EnvironmentAction, EnvironmentEvent, ObservationBundle,
    ObservationCaptureConfig, PixelFormat, StateObservation, VisualObservation, VisualSensorKind,
};
use crate::app::schedules::SimulationSet;
use crate::audio::{AudioEventKind, AudioEventQueue};
use crate::bridge::ipc_stub::{BridgeEndpointConfig, BridgeTransport};
use crate::bridge::protocol::BridgeRole;
use crate::bridge::{BridgeMode, BridgeServerSessions, players_ready};
use crate::core::config::{ConfigPaths, RepositoryConfig};
use crate::input::actions::ControlInput;
use crate::policy_contract::{ACTION_SCHEMA_ID, POLICY_CONTRACT_ID};
use crate::presentation::tracers::TracerLifetime;
use crate::simulation::components::AircraftRole;
use crate::simulation::resources::SimulationDebugState;
use crate::{gameplay::combat::Projectile, gameplay::match_state::MatchPhase};

pub const RECORDING_SCHEMA_VERSION: u32 = 12;
pub const UNSPECIFIED_POLICY_CONTRACT_ID: &str = "unspecified_legacy_policy_contract";
const DEFAULT_RECORDED_AUDIO_ROLE: &str = "fighter1";
const RECORDED_STEP_CHUNK_SIZE: usize = 256;

fn unspecified_policy_contract_id() -> String {
    UNSPECIFIED_POLICY_CONTRACT_ID.to_string()
}

pub fn queue_recorded_audio_for_playback(
    queue: &mut AudioEventQueue,
    audio_semantics: Option<&RecordedAudioFrame>,
    events: &[EnvironmentEvent],
) {
    if let Some(audio_semantics) = audio_semantics {
        for one_shot in &audio_semantics.one_shots {
            queue.push(
                one_shot.kind,
                Vec3::from_array(one_shot.position),
                one_shot.gain.max(0.0),
            );
        }
        return;
    }

    queue_recorded_audio_from_environment_events(queue, events);
}

fn queue_recorded_audio_from_environment_events(
    queue: &mut AudioEventQueue,
    events: &[EnvironmentEvent],
) {
    for event in events {
        let Some(position) = event.position.map(Vec3::from_array) else {
            continue;
        };
        let magnitude = event.magnitude.unwrap_or(1.0).max(0.0);
        match event.kind.as_str() {
            "GunFired" => queue.push(AudioEventKind::GunFire, position, magnitude.max(0.4)),
            "Hit" | "Damage" | "Collision" | "SubsystemHit" | "SubsystemDestroyed" => {
                queue.push(AudioEventKind::Hit, position, magnitude.max(0.35));
            }
            "Destroy" => queue.push(AudioEventKind::Hit, position, 1.0),
            _ => {}
        }
    }
}

pub struct ActionRecordingPlugin;

impl Plugin for ActionRecordingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActionRecordingState>()
            .init_resource::<ServerAuthoritativeRecording>()
            .add_systems(Update, drive_server_authoritative_recording)
            .add_systems(
                FixedUpdate,
                record_episode_step
                    .after(SimulationSet::ProduceSnapshot)
                    .run_if(resource_exists::<ObservationBundle>),
            );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedProjectileState {
    pub id: u64,
    pub shooter_role: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub remaining_distance: f32,
    pub damage: f32,
    pub hit_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedTracerState {
    pub position: [f32; 3],
    pub remaining_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecordedDynamicWorldState {
    pub projectiles: Vec<RecordedProjectileState>,
    pub tracers: Vec<RecordedTracerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualArtifactRef {
    pub camera: VisualSensorKind,
    pub file_path: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub format: Option<PixelFormat>,
    #[serde(default)]
    pub byte_offset: Option<u64>,
    #[serde(default)]
    pub byte_length: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioArtifactRef {
    pub file_path: Option<String>,
    #[serde(default)]
    pub byte_offset: Option<u64>,
    #[serde(default)]
    pub byte_length: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioArtifactMetadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub window_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecordedEngineAudioState {
    pub position: [f32; 3],
    pub throttle: f32,
    pub alive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedAudioOneShot {
    pub kind: AudioEventKind,
    pub position: [f32; 3],
    pub gain: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecordedAudioFrame {
    pub observed_role: String,
    pub fighter1_engine: RecordedEngineAudioState,
    pub fighter2_engine: RecordedEngineAudioState,
    #[serde(default)]
    pub one_shots: Vec<RecordedAudioOneShot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialWorldSnapshot {
    pub state: StateObservation,
    pub dynamic: RecordedDynamicWorldState,
    #[serde(default)]
    pub audio_semantics: Option<RecordedAudioFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedStep {
    pub index: u32,
    pub tick: u64,
    pub sim_time_seconds: f32,
    pub fighter1_command: EnvironmentAction,
    #[serde(default)]
    pub fighter2_command: EnvironmentAction,
    pub state: StateObservation,
    pub dynamic: RecordedDynamicWorldState,
    #[serde(default)]
    pub audio_semantics: Option<RecordedAudioFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedStepArtifacts {
    pub index: u32,
    pub tick: u64,
    pub visual: Vec<VisualArtifactRef>,
    pub audio: Option<AudioArtifactRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedStepChunk {
    pub chunk_index: u32,
    pub start_step_index: u32,
    pub steps: Vec<RecordedStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedStepChunkIndexEntry {
    pub chunk_index: u32,
    pub start_step_index: u32,
    pub step_count: u32,
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedStepArtifacts {
    pub index: u32,
    pub tick: u64,
    pub visual: Vec<VisualArtifactRef>,
    #[serde(default)]
    pub segmentation: Vec<VisualArtifactRef>,
    pub audio: Option<AudioArtifactRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingArtifactConvention {
    pub initial_state_path: String,
    pub step_chunk_pattern: String,
    pub visual_artifact_pattern: String,
    pub audio_artifact_pattern: String,
    pub derived_role_dir_pattern: String,
    pub validation_role_dir_pattern: String,
}

impl Default for RecordingArtifactConvention {
    fn default() -> Self {
        Self {
            initial_state_path: "initial_state.ron".to_string(),
            step_chunk_pattern: "steps/chunk_{chunk:06}.ron".to_string(),
            visual_artifact_pattern: "visual/{camera}/step_{index:06}.{ext}".to_string(),
            audio_artifact_pattern: "audio/steps/step_{index:06}.wav".to_string(),
            derived_role_dir_pattern: "derived/{role}".to_string(),
            validation_role_dir_pattern: "validations/{role}".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedArtifactConvention {
    pub manifest_file: String,
    pub visual_artifact_pattern: String,
    pub visual_bundle_pattern: String,
    pub segmentation_artifact_pattern: String,
    pub segmentation_bundle_pattern: String,
    pub audio_artifact_pattern: String,
    pub audio_bundle_file: String,
    pub validation_audio_file: String,
    pub validation_video_pattern: String,
}

impl Default for DerivedArtifactConvention {
    fn default() -> Self {
        Self {
            manifest_file: "derived_modalities.ron".to_string(),
            visual_artifact_pattern: "visual/{camera}/{frame_key}.{ext}".to_string(),
            visual_bundle_pattern: "visual/{camera}.frames".to_string(),
            segmentation_artifact_pattern: "segmentation/{camera}/{frame_key}.{ext}".to_string(),
            segmentation_bundle_pattern: "segmentation/{camera}.frames".to_string(),
            audio_artifact_pattern: "audio/{frame_key}.wav".to_string(),
            audio_bundle_file: "audio/steps.wavs".to_string(),
            validation_audio_file: "validation_audio.wav".to_string(),
            validation_video_pattern: "validation_{camera}.mp4".to_string(),
        }
    }
}

impl From<&AudioObservation> for AudioArtifactMetadata {
    fn from(value: &AudioObservation) -> Self {
        Self {
            sample_rate: value.sample_rate,
            channels: value.channels,
            window_seconds: value.window_seconds,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedEpisodeManifest {
    pub schema_version: u32,
    pub source_episode_id: String,
    pub source_episode_root: String,
    pub observed_role: String,
    pub capture_config: ObservationCaptureConfig,
    pub audio_artifact_metadata: Option<AudioArtifactMetadata>,
    pub initial_tick: u64,
    pub total_steps: u32,
    pub initial_visual: Vec<VisualArtifactRef>,
    #[serde(default)]
    pub initial_segmentation: Vec<VisualArtifactRef>,
    pub initial_audio: Option<AudioArtifactRef>,
    pub steps: Vec<DerivedStepArtifacts>,
    pub artifact_convention: DerivedArtifactConvention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedEpisodeManifest {
    pub schema_version: u32,
    #[serde(default = "unspecified_policy_contract_id")]
    pub policy_contract_id: String,
    pub action_schema_id: String,
    pub episode_id: String,
    pub scene_name: String,
    pub seed: Option<u64>,
    pub fixed_time_step_seconds: f32,
    pub capture_config: ObservationCaptureConfig,
    pub audio_artifact_metadata: Option<AudioArtifactMetadata>,
    pub started_tick: u64,
    pub started_sim_time_seconds: f32,
    pub final_tick: u64,
    pub final_sim_time_seconds: f32,
    pub total_steps: u32,
    pub step_chunks: Vec<RecordedStepChunkIndexEntry>,
    pub step_artifacts: Vec<RecordedStepArtifacts>,
    pub termination_reason: String,
    pub winner: Option<String>,
    pub bridge_role: Option<String>,
    pub bridge_session: Option<String>,
    pub bridge_transport: Option<String>,
    pub authoritative_source: bool,
    pub initial_state_path: String,
    pub artifact_convention: RecordingArtifactConvention,
}

#[derive(Debug, Clone)]
struct ActiveEpisodeRecording {
    episode_id: String,
    episode_root: PathBuf,
    scene_name: String,
    seed: Option<u64>,
    fixed_time_step_seconds: f32,
    capture_config: ObservationCaptureConfig,
    bridge_role: Option<String>,
    bridge_session: Option<String>,
    bridge_transport: Option<String>,
    authoritative_source: bool,
    started_tick: u64,
    started_sim_time_seconds: f32,
    next_step_index: u32,
    next_chunk_index: u32,
    initial_snapshot_written: bool,
    audio_artifact_metadata: Option<AudioArtifactMetadata>,
    step_chunk_buffer: Vec<RecordedStep>,
    step_chunks: Vec<RecordedStepChunkIndexEntry>,
    step_artifacts: Vec<RecordedStepArtifacts>,
}

#[derive(Debug, Resource, Default)]
pub struct ActionRecordingState {
    pub active: bool,
    pub pending_start: bool,
    pub pending_stop: bool,
    pub last_saved_path: Option<String>,
    pub last_saved_frame_count: usize,
    pub active_step_count: usize,
    previous_capture_config: Option<ObservationCaptureConfig>,
    session: Option<ActiveEpisodeRecording>,
}

#[derive(Debug, Resource, Clone, Copy, Default)]
pub struct ServerAuthoritativeRecording {
    pub enabled: bool,
}

impl ActionRecordingState {
    pub fn status_label(&self) -> String {
        if self.active || self.pending_start {
            format!("REC {}f", self.active_step_count)
        } else if let Some(path) = &self.last_saved_path {
            let short = PathBuf::from(path)
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.to_string())
                .unwrap_or_else(|| path.clone());
            format!("SAVED {} ({})", self.last_saved_frame_count, short)
        } else {
            "OFF".to_string()
        }
    }
}

pub fn request_manual_recording_start(
    world: &mut World,
    desired_capture: Option<ObservationCaptureConfig>,
) -> Result<()> {
    request_recording_start(world, desired_capture, false)
}

pub fn request_authoritative_recording_start(
    world: &mut World,
    desired_capture: Option<ObservationCaptureConfig>,
) -> Result<()> {
    request_recording_start(world, desired_capture, true)
}

fn request_recording_start(
    world: &mut World,
    desired_capture: Option<ObservationCaptureConfig>,
    authoritative_source: bool,
) -> Result<()> {
    let config_paths = world
        .get_resource::<ConfigPaths>()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing ConfigPaths resource"))?;
    let config = world
        .get_resource::<RepositoryConfig>()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing RepositoryConfig resource"))?;
    let started_tick = world
        .get_resource::<SimulationDebugState>()
        .map(|debug| debug.tick_count)
        .ok_or_else(|| anyhow::anyhow!("missing SimulationDebugState resource"))?;
    let bridge_mode = world.get_resource::<BridgeMode>().map(|mode| mode.0);
    let bridge_endpoint = world.get_resource::<BridgeEndpointConfig>().cloned();
    let environment_seed = world.get_resource::<EnvironmentSeed>().copied();
    let capture = desired_capture.unwrap_or_else(|| {
        world
            .get_resource::<ObservationCaptureConfig>()
            .cloned()
            .unwrap_or_default()
    });

    world.resource_scope(|world, mut capture_config: Mut<ObservationCaptureConfig>| {
        world.resource_scope(|_world, mut recording: Mut<ActionRecordingState>| {
            if recording.active || recording.pending_start {
                bail!("recording is already active or pending");
            }

            arm_recording_session(
                &config_paths,
                &config,
                started_tick,
                environment_seed.as_ref(),
                bridge_mode,
                bridge_endpoint
                    .as_ref()
                    .map(|endpoint| endpoint.session.as_str()),
                bridge_endpoint.as_ref().map(|endpoint| endpoint.transport),
                authoritative_source,
                capture,
                &mut capture_config,
                &mut recording,
            );
            Ok(())
        })
    })?;
    Ok(())
}

pub fn request_manual_recording_stop(world: &mut World) -> bool {
    let Some(mut recording) = world.get_resource_mut::<ActionRecordingState>() else {
        return false;
    };
    if !(recording.active || recording.pending_start || recording.pending_stop) {
        return false;
    }
    recording.pending_start = false;
    recording.pending_stop = true;
    true
}

#[allow(clippy::too_many_arguments)]
fn arm_recording_session(
    config_paths: &ConfigPaths,
    config: &RepositoryConfig,
    started_tick: u64,
    environment_seed: Option<&EnvironmentSeed>,
    bridge_mode: Option<BridgeRole>,
    bridge_session: Option<&str>,
    bridge_transport: Option<BridgeTransport>,
    authoritative_source: bool,
    desired_capture: ObservationCaptureConfig,
    capture_config: &mut ObservationCaptureConfig,
    recording: &mut ActionRecordingState,
) {
    let scene_name = config_paths
        .scene_override
        .clone()
        .unwrap_or_else(|| config.game.active_scene.clone());
    let scene_slug = scene_name.replace(['/', '\\', ' '], "_");
    let recordings_root = config_paths.recordings_root();
    let episode_id_base = format!("{scene_slug}-{}", recording_timestamp_slug());
    let (episode_id, episode_root) =
        allocate_recording_episode_path(&recordings_root, &episode_id_base);

    recording.previous_capture_config = Some(capture_config.clone());
    *capture_config = desired_capture.clone();

    recording.active = false;
    recording.pending_start = true;
    recording.pending_stop = false;
    recording.last_saved_path = None;
    recording.last_saved_frame_count = 0;
    recording.active_step_count = 0;
    recording.session = Some(ActiveEpisodeRecording {
        episode_id,
        episode_root,
        scene_name,
        seed: environment_seed.map(|seed| seed.effective),
        fixed_time_step_seconds: config.game.fixed_time_step_seconds,
        capture_config: desired_capture,
        bridge_role: bridge_mode.map(|mode| format!("{mode:?}")),
        bridge_session: bridge_session.map(ToOwned::to_owned),
        bridge_transport: bridge_transport.map(|transport| format!("{transport:?}")),
        authoritative_source,
        started_tick,
        started_sim_time_seconds: 0.0,
        next_step_index: 0,
        next_chunk_index: 0,
        initial_snapshot_written: false,
        audio_artifact_metadata: None,
        step_chunk_buffer: Vec::with_capacity(RECORDED_STEP_CHUNK_SIZE),
        step_chunks: Vec::new(),
        step_artifacts: Vec::new(),
    });
}

fn drive_server_authoritative_recording(
    mode: Option<Res<BridgeMode>>,
    server_policy: Res<ServerAuthoritativeRecording>,
    sessions: Option<Res<BridgeServerSessions>>,
    endpoint: Option<Res<BridgeEndpointConfig>>,
    config_paths: Res<ConfigPaths>,
    config: Res<RepositoryConfig>,
    debug: Res<SimulationDebugState>,
    match_phase: Res<State<MatchPhase>>,
    environment_seed: Option<Res<EnvironmentSeed>>,
    mut capture_config: ResMut<ObservationCaptureConfig>,
    mut recording: ResMut<ActionRecordingState>,
) {
    if !server_policy.enabled
        || mode.as_deref().map(|mode| mode.0) != Some(BridgeRole::Server)
        || recording.active
        || recording.pending_start
        || !matches!(match_phase.get(), MatchPhase::Running)
        || !sessions.as_deref().map(players_ready).unwrap_or(false)
    {
        return;
    }

    let endpoint = endpoint.as_deref();
    arm_recording_session(
        &config_paths,
        &config,
        debug.tick_count,
        environment_seed.as_deref(),
        Some(BridgeRole::Server),
        endpoint.map(|endpoint| endpoint.session.as_str()),
        endpoint.map(|endpoint| endpoint.transport),
        true,
        authoritative_server_recording_capture_config(config.game.fixed_time_step_seconds),
        &mut capture_config,
        &mut recording,
    );
    info!("arming authoritative episode recording");
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_episode_step(
    match_phase: Res<State<MatchPhase>>,
    observation: Res<ObservationBundle>,
    mut capture_config: ResMut<ObservationCaptureConfig>,
    mut recording: ResMut<ActionRecordingState>,
    control_query: Query<(&AircraftRole, &ControlInput)>,
    projectile_query: Query<(&Projectile, &Transform)>,
    tracer_query: Query<(&TracerLifetime, &Transform)>,
) {
    let Some(mut session) = recording.session.take() else {
        return;
    };

    let player_input = control_query
        .iter()
        .find(|(role, _)| **role == AircraftRole::Fighter1)
        .map(|(_, input)| *input);
    let enemy_input = control_query
        .iter()
        .find(|(role, _)| **role == AircraftRole::Fighter2)
        .map(|(_, input)| *input);

    let (Some(player_input), Some(enemy_input)) = (player_input, enemy_input) else {
        return;
    };

    let pending_start = recording.pending_start;
    let pending_stop = recording.pending_stop;

    if pending_start {
        if let Err(error) = prepare_episode_dirs(&session.episode_root, &session.capture_config) {
            error!("failed to create recording directories: {error}");
            restore_capture_config(&mut capture_config, &mut recording);
            return;
        }
        session.started_sim_time_seconds = observation.state.sim_time_seconds;
        let initial_snapshot = InitialWorldSnapshot {
            state: observation.state.clone(),
            dynamic: capture_dynamic_world(&projectile_query, &tracer_query),
            audio_semantics: Some(capture_recorded_audio_frame(&observation.state)),
        };
        if let Err(error) = write_initial_snapshot(&session.episode_root, &initial_snapshot) {
            error!("failed to write initial world snapshot: {error}");
            restore_capture_config(&mut capture_config, &mut recording);
            return;
        }
        recording.pending_start = false;
        recording.active = true;
        session.initial_snapshot_written = true;
        info!(
            "started authoritative episode recording in {}",
            session.episode_root.display()
        );
    }

    if !recording.active {
        recording.session = Some(session);
        return;
    }

    let step_index = session.next_step_index;
    let dynamic = capture_dynamic_world(&projectile_query, &tracer_query);
    let visual = observation
        .visual
        .iter()
        .map(|frame| write_visual_artifact(&session.episode_root, step_index, frame))
        .collect::<anyhow::Result<Vec<_>>>();
    let audio = observation
        .audio
        .as_ref()
        .map(|audio| write_audio_artifact(&session.episode_root, step_index, audio))
        .transpose();

    let (visual, audio) = match (visual, audio) {
        (Ok(visual), Ok(audio)) => (visual, audio),
        (Err(error), _) | (_, Err(error)) => {
            error!("failed to write recording artifacts: {error}");
            restore_capture_config(&mut capture_config, &mut recording);
            return;
        }
    };
    if session.audio_artifact_metadata.is_none() {
        session.audio_artifact_metadata =
            observation.audio.as_ref().map(AudioArtifactMetadata::from);
    }

    let step = RecordedStep {
        index: step_index,
        tick: observation.state.tick,
        sim_time_seconds: observation.state.sim_time_seconds,
        fighter1_command: EnvironmentAction::from(player_input),
        fighter2_command: EnvironmentAction::from(enemy_input),
        state: observation.state.clone(),
        dynamic,
        audio_semantics: Some(capture_recorded_audio_frame(&observation.state)),
    };
    session.step_chunk_buffer.push(step);
    session.step_artifacts.push(RecordedStepArtifacts {
        index: step_index,
        tick: observation.state.tick,
        visual,
        audio,
    });
    if let Err(error) = maybe_flush_recorded_step_chunk(&mut session) {
        error!("failed to write step record: {error}");
        restore_capture_config(&mut capture_config, &mut recording);
        return;
    }

    session.next_step_index += 1;
    recording.active_step_count = session.next_step_index as usize;

    if pending_stop || *match_phase.get() == MatchPhase::Finished {
        if let Err(error) = flush_recorded_step_chunk(&mut session) {
            error!("failed to flush recorded step chunk: {error}");
            restore_capture_config(&mut capture_config, &mut recording);
            return;
        }
        let termination_reason = if pending_stop {
            "manual_stop".to_string()
        } else {
            "match_finished".to_string()
        };
        let winner = infer_winner(&observation.state);
        let manifest = RecordedEpisodeManifest {
            schema_version: RECORDING_SCHEMA_VERSION,
            policy_contract_id: POLICY_CONTRACT_ID.to_string(),
            action_schema_id: ACTION_SCHEMA_ID.to_string(),
            episode_id: session.episode_id.clone(),
            scene_name: session.scene_name.clone(),
            seed: session.seed,
            fixed_time_step_seconds: session.fixed_time_step_seconds,
            capture_config: session.capture_config.clone(),
            audio_artifact_metadata: session.audio_artifact_metadata.clone(),
            started_tick: session.started_tick,
            started_sim_time_seconds: session.started_sim_time_seconds,
            final_tick: observation.state.tick,
            final_sim_time_seconds: observation.state.sim_time_seconds,
            total_steps: session.next_step_index,
            step_chunks: session.step_chunks.clone(),
            step_artifacts: session.step_artifacts.clone(),
            termination_reason,
            winner,
            bridge_role: session.bridge_role.clone(),
            bridge_session: session.bridge_session.clone(),
            bridge_transport: session.bridge_transport.clone(),
            authoritative_source: session.authoritative_source,
            initial_state_path: "initial_state.ron".to_string(),
            artifact_convention: RecordingArtifactConvention::default(),
        };

        match write_episode_manifest(&session.episode_root, &manifest) {
            Ok(path) => {
                recording.last_saved_path = Some(path.display().to_string());
                recording.last_saved_frame_count = session.next_step_index as usize;
                info!(
                    "saved authoritative episode recording to {}",
                    path.display()
                );
            }
            Err(error) => {
                error!("failed to finalize episode recording: {error}");
                recording.last_saved_path = None;
            }
        }

        restore_capture_config(&mut capture_config, &mut recording);
        return;
    }

    recording.session = Some(session);
}

pub fn render_indexed_artifact_pattern(pattern: &str, index: u32) -> String {
    pattern
        .replace("{index:06}", &format!("{index:06}"))
        .replace("{index}", &index.to_string())
}

pub fn render_chunk_artifact_pattern(pattern: &str, chunk: u32) -> String {
    pattern
        .replace("{chunk:06}", &format!("{chunk:06}"))
        .replace("{chunk}", &chunk.to_string())
}

pub fn render_named_artifact_pattern(
    pattern: &str,
    camera: Option<&str>,
    frame_key: Option<&str>,
    ext: Option<&str>,
    role: Option<&str>,
) -> String {
    let mut rendered = pattern.to_string();
    if let Some(camera) = camera {
        rendered = rendered.replace("{camera}", camera);
    }
    if let Some(frame_key) = frame_key {
        rendered = rendered.replace("{frame_key}", frame_key);
    }
    if let Some(ext) = ext {
        rendered = rendered.replace("{ext}", ext);
    }
    if let Some(role) = role {
        rendered = rendered.replace("{role}", role);
    }
    rendered
}

fn restore_capture_config(
    capture_config: &mut ObservationCaptureConfig,
    recording: &mut ActionRecordingState,
) {
    if let Some(previous) = recording.previous_capture_config.take() {
        *capture_config = previous;
    }
    recording.active = false;
    recording.pending_start = false;
    recording.pending_stop = false;
    recording.session = None;
}

fn authoritative_server_recording_capture_config(
    fixed_time_step_seconds: f32,
) -> ObservationCaptureConfig {
    ObservationCaptureConfig {
        enable_visual: false,
        enable_audio: false,
        visual_sensors: Vec::new(),
        audio_window_seconds: fixed_time_step_seconds.max(1.0 / 240.0),
    }
}

fn recording_timestamp_slug() -> String {
    let now = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let base = now
        .format(format_description!(
            "[year][month][day]-[hour][minute][second]"
        ))
        .unwrap_or_else(|_| "unknown-time".to_string());
    format!("{base}-{:03}", now.millisecond())
}

fn allocate_recording_episode_path(recordings_root: &Path, base_id: &str) -> (String, PathBuf) {
    for suffix in 0_u32.. {
        let episode_id = if suffix == 0 {
            base_id.to_string()
        } else {
            format!("{base_id}-{suffix:03}")
        };
        let episode_root = recordings_root.join(&episode_id);
        if !episode_root.exists() {
            return (episode_id, episode_root);
        }
    }
    unreachable!("u32 recording suffix space is exhausted")
}

fn prepare_episode_dirs(root: &Path, capture: &ObservationCaptureConfig) -> anyhow::Result<()> {
    if root.exists() {
        anyhow::bail!(
            "refusing to overwrite existing recording {}",
            root.display()
        );
    }
    fs::create_dir_all(root.join("steps"))?;
    if capture.enable_visual {
        for sensor in &capture.visual_sensors {
            match sensor.kind {
                VisualSensorKind::Front => fs::create_dir_all(root.join("visual/front"))?,
                VisualSensorKind::Rear => fs::create_dir_all(root.join("visual/rear"))?,
            }
        }
    }
    if capture.enable_audio {
        fs::create_dir_all(root.join("audio/steps"))?;
    }
    Ok(())
}

fn capture_dynamic_world(
    projectile_query: &Query<(&Projectile, &Transform)>,
    tracer_query: &Query<(&TracerLifetime, &Transform)>,
) -> RecordedDynamicWorldState {
    let projectiles = projectile_query
        .iter()
        .map(|(projectile, transform)| RecordedProjectileState {
            id: projectile.id,
            shooter_role: format!("{:?}", projectile.shooter_role),
            position: transform.translation.to_array(),
            velocity: projectile.velocity.to_array(),
            remaining_distance: projectile.remaining_distance,
            damage: projectile.damage,
            hit_radius: projectile.hit_radius,
        })
        .collect();
    let tracers = tracer_query
        .iter()
        .map(|(tracer, transform)| RecordedTracerState {
            position: transform.translation.to_array(),
            remaining_seconds: tracer.remaining_seconds,
        })
        .collect();
    RecordedDynamicWorldState {
        projectiles,
        tracers,
    }
}

fn write_initial_snapshot(root: &Path, snapshot: &InitialWorldSnapshot) -> anyhow::Result<()> {
    let payload = ron::ser::to_string_pretty(snapshot, ron::ser::PrettyConfig::default())?;
    fs::write(
        root.join(&RecordingArtifactConvention::default().initial_state_path),
        payload,
    )?;
    Ok(())
}

fn write_step_chunk(root: &Path, chunk: &RecordedStepChunk) -> anyhow::Result<PathBuf> {
    let payload = ron::ser::to_string_pretty(chunk, ron::ser::PrettyConfig::default())?;
    let relative = PathBuf::from(render_chunk_artifact_pattern(
        &RecordingArtifactConvention::default().step_chunk_pattern,
        chunk.chunk_index,
    ));
    fs::write(root.join(&relative), payload)?;
    Ok(relative)
}

fn flush_recorded_step_chunk(session: &mut ActiveEpisodeRecording) -> anyhow::Result<()> {
    if session.step_chunk_buffer.is_empty() {
        return Ok(());
    }
    let start_step_index = session
        .step_chunk_buffer
        .first()
        .map(|step| step.index)
        .unwrap_or(session.next_step_index);
    let steps = std::mem::take(&mut session.step_chunk_buffer);
    let chunk = RecordedStepChunk {
        chunk_index: session.next_chunk_index,
        start_step_index,
        steps,
    };
    let file_path = write_step_chunk(&session.episode_root, &chunk)?;
    session.step_chunks.push(RecordedStepChunkIndexEntry {
        chunk_index: chunk.chunk_index,
        start_step_index: chunk.start_step_index,
        step_count: chunk.steps.len() as u32,
        file_path: file_path.display().to_string(),
    });
    session.next_chunk_index += 1;
    Ok(())
}

fn maybe_flush_recorded_step_chunk(session: &mut ActiveEpisodeRecording) -> anyhow::Result<()> {
    if session.step_chunk_buffer.len() >= RECORDED_STEP_CHUNK_SIZE {
        flush_recorded_step_chunk(session)?;
    }
    Ok(())
}

fn write_episode_manifest(
    root: &Path,
    manifest: &RecordedEpisodeManifest,
) -> anyhow::Result<PathBuf> {
    let output_path = root.join("episode.ron");
    let payload = ron::ser::to_string_pretty(manifest, ron::ser::PrettyConfig::default())?;
    fs::write(&output_path, payload)?;
    Ok(output_path)
}

fn write_visual_artifact(
    root: &Path,
    step_index: u32,
    frame: &VisualObservation,
) -> anyhow::Result<VisualArtifactRef> {
    let file_path = if frame.bytes_ready && !frame.bytes.is_empty() {
        let subdir = match frame.camera {
            VisualSensorKind::Front => "front",
            VisualSensorKind::Rear => "rear",
        };
        let extension = match frame.format {
            PixelFormat::Rgb8 => "ppm",
            PixelFormat::Gray8 => "pgm",
            PixelFormat::Rgba8 => "raw",
        };
        let relative = PathBuf::from(render_named_artifact_pattern(
            &render_indexed_artifact_pattern(
                &RecordingArtifactConvention::default().visual_artifact_pattern,
                step_index,
            ),
            Some(subdir),
            None,
            Some(extension),
            None,
        ));
        let full_path = root.join(&relative);
        match frame.format {
            PixelFormat::Rgb8 => write_ppm(&full_path, frame.width, frame.height, &frame.bytes)?,
            PixelFormat::Gray8 => write_pgm(&full_path, frame.width, frame.height, &frame.bytes)?,
            PixelFormat::Rgba8 => fs::write(&full_path, &frame.bytes)?,
        }
        Some(relative.display().to_string())
    } else {
        None
    };

    Ok(VisualArtifactRef {
        camera: frame.camera,
        file_path,
        width: Some(frame.width),
        height: Some(frame.height),
        format: Some(frame.format),
        byte_offset: None,
        byte_length: None,
    })
}

fn write_audio_artifact(
    root: &Path,
    step_index: u32,
    audio: &AudioObservation,
) -> anyhow::Result<AudioArtifactRef> {
    let relative = PathBuf::from(render_indexed_artifact_pattern(
        &RecordingArtifactConvention::default().audio_artifact_pattern,
        step_index,
    ));
    let full_path = root.join(&relative);
    write_wav_i16(
        &full_path,
        audio.sample_rate,
        audio.channels,
        &audio.samples,
    )?;
    Ok(AudioArtifactRef {
        file_path: Some(relative.display().to_string()),
        byte_offset: None,
        byte_length: None,
    })
}

fn capture_recorded_audio_frame(state: &StateObservation) -> RecordedAudioFrame {
    RecordedAudioFrame {
        observed_role: DEFAULT_RECORDED_AUDIO_ROLE.to_string(),
        fighter1_engine: capture_recorded_engine_audio_state(state, AircraftRole::Fighter1),
        fighter2_engine: capture_recorded_engine_audio_state(state, AircraftRole::Fighter2),
        one_shots: capture_recorded_audio_one_shots(&state.events_since_last_step, &state.aircraft),
    }
}

fn capture_recorded_engine_audio_state(
    state: &StateObservation,
    role: AircraftRole,
) -> RecordedEngineAudioState {
    let Some(aircraft) = state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == recorded_role_name(role))
    else {
        return RecordedEngineAudioState::default();
    };

    RecordedEngineAudioState {
        position: aircraft.position,
        throttle: aircraft.throttle,
        alive: !aircraft.destroyed,
    }
}

fn capture_recorded_audio_one_shots(
    events: &[EnvironmentEvent],
    aircraft: &[AircraftObservation],
) -> Vec<RecordedAudioOneShot> {
    events
        .iter()
        .filter_map(|event| recorded_audio_one_shot_from_event(event, aircraft))
        .collect()
}

fn recorded_audio_one_shot_from_event(
    event: &EnvironmentEvent,
    aircraft: &[AircraftObservation],
) -> Option<RecordedAudioOneShot> {
    let kind = match event.kind.as_str() {
        "GunFired" => AudioEventKind::GunFire,
        "Hit" | "Damage" | "SubsystemHit" | "Collision" | "Destroy" => AudioEventKind::Hit,
        _ => return None,
    };

    let position = event
        .position
        .or_else(|| {
            event.subject.as_deref().and_then(|subject| {
                aircraft
                    .iter()
                    .find(|aircraft| aircraft.role == subject)
                    .map(|aircraft| aircraft.position)
            })
        })
        .unwrap_or([0.0, 0.0, 0.0]);

    Some(RecordedAudioOneShot {
        kind,
        position,
        gain: event.magnitude.unwrap_or(1.0).clamp(0.0, 4.0),
    })
}

fn recorded_role_name(role: AircraftRole) -> &'static str {
    match role {
        AircraftRole::Fighter1 => "fighter1",
        AircraftRole::Fighter2 => "fighter2",
    }
}

fn write_wav_i16(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    samples: &[f32],
) -> anyhow::Result<()> {
    let mut file = fs::File::create(path)?;
    let bytes_per_sample = 2u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (samples.len() * bytes_per_sample as usize) as u32;
    let riff_len = 36 + data_len;

    file.write_all(b"RIFF")?;
    file.write_all(&riff_len.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&(bytes_per_sample * 8).to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

fn write_ppm(path: &Path, width: u32, height: u32, bytes: &[u8]) -> anyhow::Result<()> {
    let expected = width as usize * height as usize * 3;
    anyhow::ensure!(
        bytes.len() == expected,
        "unexpected RGB byte length for {}: got {}, expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    file.write_all(bytes)?;
    Ok(())
}

fn write_pgm(path: &Path, width: u32, height: u32, bytes: &[u8]) -> anyhow::Result<()> {
    let expected = width as usize * height as usize;
    anyhow::ensure!(
        bytes.len() == expected,
        "unexpected Gray byte length for {}: got {}, expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{} {}\n255\n", width, height)?;
    file.write_all(bytes)?;
    Ok(())
}

fn infer_winner(state: &StateObservation) -> Option<String> {
    let fighter1_destroyed = state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter1" || aircraft.role == "Fighter1")
        .map(|aircraft| aircraft.destroyed)
        .unwrap_or(false);
    let fighter2_destroyed = state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter2" || aircraft.role == "Fighter2")
        .map(|aircraft| aircraft.destroyed)
        .unwrap_or(false);
    match (fighter1_destroyed, fighter2_destroyed) {
        (false, true) => Some("fighter1".to_string()),
        (true, false) => Some("fighter2".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::allocate_recording_episode_path;
    use std::fs;

    #[test]
    fn recording_path_allocation_never_reuses_existing_episode() {
        let root =
            std::env::temp_dir().join(format!("dfb-recording-path-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("episode")).expect("create occupied episode path");
        fs::create_dir_all(root.join("episode-001")).expect("create occupied suffixed path");

        let (episode_id, episode_root) = allocate_recording_episode_path(&root, "episode");

        assert_eq!(episode_id, "episode-002");
        assert_eq!(episode_root, root.join("episode-002"));
        fs::remove_dir_all(root).expect("remove test directory");
    }
}
