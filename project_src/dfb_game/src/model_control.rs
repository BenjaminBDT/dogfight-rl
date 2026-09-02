use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::api::commands::{ExternalCommandBuffer, TargetedEnvironmentAction};
use crate::api::snapshot::WorldSnapshot;
use crate::api::types::{EnvironmentAction, StateObservation};
use crate::app::schedules::SimulationSet;
use crate::bridge::protocol::BridgeControlSlot;
use crate::bridge::{AssignedControlRole, BridgeClientInbox, BridgeEnabled, LocalPilotMode};
use crate::core::config::ConfigPaths;
use crate::simulation::components::AircraftRole;

#[derive(Debug, Clone, Resource, Default)]
pub struct ModelControlConfig {
    pub checkpoint: Option<String>,
    pub dataset_root: Option<String>,
    pub python_executable: Option<String>,
    pub device: Option<String>,
    pub observation_source: Option<String>,
}

#[derive(Debug, Default, Resource)]
struct ModelControlState {
    runtime: Mutex<ModelControlRuntime>,
}

#[derive(Debug, Default)]
struct ModelControlRuntime {
    worker: Option<ModelPilotWorker>,
    warned_missing_config: bool,
    last_action: EnvironmentAction,
    request_in_flight: bool,
    episode_cursor: Option<ModelEpisodeCursor>,
    episode_start_sim_time_seconds: Option<f32>,
}

#[derive(Debug, Clone)]
struct ModelEpisodeCursor {
    scene_name: String,
    tick: u64,
    sim_time_seconds: f32,
    all_aircraft_alive: bool,
}

#[derive(Debug)]
struct ModelPilotSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug)]
struct ModelPilotWorker {
    request_tx: SyncSender<ModelPilotRequest>,
    response_rx: Receiver<ModelPilotResponse>,
    _thread: JoinHandle<()>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelPilotRequest {
    role: String,
    episode_start_sim_time_seconds: f32,
    state: StateObservation,
}

#[derive(Debug, Deserialize)]
struct ModelPilotResponse {
    action: EnvironmentAction,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelObservationSource {
    AuthoritativeInBridge,
    LocalPredicted,
}

pub struct ModelControlPlugin;

impl Plugin for ModelControlPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ModelControlConfig>()
            .init_resource::<ModelControlState>()
            .add_systems(
                FixedUpdate,
                drive_model_control
                    .in_set(SimulationSet::GatherInput)
                    .before(crate::api::commands::apply_external_commands),
            );
    }
}

fn drive_model_control(
    local_pilot_mode: Option<Res<LocalPilotMode>>,
    controlled_role: Option<Res<AssignedControlRole>>,
    config_paths: Res<ConfigPaths>,
    config: Res<ModelControlConfig>,
    snapshot: Res<WorldSnapshot>,
    bridge_enabled: Option<Res<BridgeEnabled>>,
    bridge_inbox: Option<Res<BridgeClientInbox>>,
    mut command_buffer: ResMut<ExternalCommandBuffer>,
    state: Res<ModelControlState>,
) {
    let Some(local_pilot_mode) = local_pilot_mode else {
        return;
    };
    if *local_pilot_mode != LocalPilotMode::Model {
        return;
    }
    let Some(role) = resolved_controlled_role(controlled_role.as_deref()) else {
        return;
    };
    if snapshot.observation.state.aircraft.is_empty() {
        return;
    }
    let authoritative_state = resolve_live_state(
        &snapshot.observation.state,
        bridge_enabled.as_deref(),
        bridge_inbox.as_deref(),
        resolved_observation_source(&config),
    );

    let mut runtime = state.runtime.lock().expect("model control mutex poisoned");
    let Some((checkpoint, dataset_root)) = resolved_required_paths(&config) else {
        if !runtime.warned_missing_config {
            warn!(
                "model control enabled but missing checkpoint; set --model-checkpoint or DFB_MODEL_CHECKPOINT"
            );
            runtime.warned_missing_config = true;
        }
        command_buffer
            .targeted_actions
            .push(TargetedEnvironmentAction {
                role,
                action: runtime.last_action,
            });
        return;
    };
    runtime.warned_missing_config = false;

    if runtime.worker.is_none() {
        match spawn_model_pilot_worker(
            &config_paths.project_root,
            &config,
            &checkpoint,
            dataset_root.as_deref(),
        ) {
            Ok(session) => {
                info!(
                    "started live model-control subprocess checkpoint={} dataset_root={}",
                    checkpoint.display(),
                    dataset_root
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "<from checkpoint>".to_string())
                );
                runtime.worker = Some(session);
                runtime.request_in_flight = false;
            }
            Err(error) => {
                warn!("failed to start live model-control subprocess: {error:#}");
                command_buffer
                    .targeted_actions
                    .push(TargetedEnvironmentAction {
                        role,
                        action: runtime.last_action,
                    });
                return;
            }
        }
    }

    if runtime.worker.is_none() {
        command_buffer
            .targeted_actions
            .push(TargetedEnvironmentAction {
                role,
                action: runtime.last_action,
            });
        return;
    }

    loop {
        let recv_result = {
            let worker = runtime
                .worker
                .as_mut()
                .expect("worker checked before polling response");
            worker.response_rx.try_recv()
        };
        match recv_result {
            Ok(response) => {
                runtime.request_in_flight = false;
                if let Some(error) = response.error.as_deref() {
                    warn!("model-control subprocess returned error: {error}");
                    terminate_worker(&mut runtime.worker);
                } else {
                    runtime.last_action = response.action;
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("live model-control worker disconnected");
                terminate_worker(&mut runtime.worker);
                runtime.request_in_flight = false;
                break;
            }
        }
    }

    if runtime.worker.is_none() {
        command_buffer
            .targeted_actions
            .push(TargetedEnvironmentAction {
                role,
                action: runtime.last_action,
            });
        return;
    }

    if !runtime.request_in_flight {
        let episode_start_sim_time_seconds =
            update_episode_time_context(&mut runtime, &authoritative_state);
        let request = ModelPilotRequest {
            role: role_name(role).to_string(),
            episode_start_sim_time_seconds,
            state: authoritative_state,
        };
        let send_result = {
            let worker = runtime
                .worker
                .as_mut()
                .expect("worker checked before sending request");
            worker.request_tx.try_send(request)
        };
        match send_result {
            Ok(()) => {
                runtime.request_in_flight = true;
            }
            Err(mpsc::TrySendError::Full(_)) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => {
                warn!("live model-control request channel disconnected");
                terminate_worker(&mut runtime.worker);
                runtime.request_in_flight = false;
            }
        }
    }

    command_buffer
        .targeted_actions
        .push(TargetedEnvironmentAction {
            role,
            action: runtime.last_action,
        });
}

fn update_episode_time_context(runtime: &mut ModelControlRuntime, state: &StateObservation) -> f32 {
    let all_aircraft_alive = state.aircraft.iter().all(|aircraft| !aircraft.destroyed);
    let starts_new_episode = runtime.episode_cursor.as_ref().is_none_or(|cursor| {
        cursor.scene_name != state.scene_name
            || state.tick < cursor.tick
            || state.sim_time_seconds < cursor.sim_time_seconds
            || (!cursor.all_aircraft_alive && all_aircraft_alive)
    });
    if starts_new_episode {
        runtime.episode_start_sim_time_seconds = Some(state.sim_time_seconds);
    }
    runtime.episode_cursor = Some(ModelEpisodeCursor {
        scene_name: state.scene_name.clone(),
        tick: state.tick,
        sim_time_seconds: state.sim_time_seconds,
        all_aircraft_alive,
    });
    runtime
        .episode_start_sim_time_seconds
        .expect("episode start initialized with cursor")
}

fn resolve_live_state(
    local_state: &StateObservation,
    bridge_enabled: Option<&BridgeEnabled>,
    bridge_inbox: Option<&BridgeClientInbox>,
    observation_source: ModelObservationSource,
) -> StateObservation {
    if matches!(observation_source, ModelObservationSource::LocalPredicted) {
        return local_state.clone();
    }
    if matches!(bridge_enabled, Some(BridgeEnabled(true)))
        && let Some(inbox) = bridge_inbox
        && let Some(snapshot) = inbox.latest_snapshot.as_ref()
        && !snapshot.observation.state.aircraft.is_empty()
    {
        return snapshot.observation.state.clone();
    }
    local_state.clone()
}

fn resolved_observation_source(config: &ModelControlConfig) -> ModelObservationSource {
    match config.observation_source.as_deref() {
        Some("authoritative") => ModelObservationSource::AuthoritativeInBridge,
        _ => ModelObservationSource::LocalPredicted,
    }
}

fn resolved_controlled_role(controlled_role: Option<&AssignedControlRole>) -> Option<AircraftRole> {
    controlled_role
        .map(|role| match role.0 {
            BridgeControlSlot::Fighter1 => Some(AircraftRole::Fighter1),
            BridgeControlSlot::Fighter2 => Some(AircraftRole::Fighter2),
            BridgeControlSlot::Spectator => None,
        })
        .unwrap_or(Some(AircraftRole::Fighter1))
}

fn resolved_required_paths(config: &ModelControlConfig) -> Option<(PathBuf, Option<PathBuf>)> {
    let checkpoint = config.checkpoint.as_ref().map(PathBuf::from)?;
    let dataset_root = config.dataset_root.as_ref().map(PathBuf::from);
    Some((checkpoint, dataset_root))
}

fn role_name(role: AircraftRole) -> &'static str {
    match role {
        AircraftRole::Fighter1 => "fighter1",
        AircraftRole::Fighter2 => "fighter2",
    }
}

fn spawn_model_pilot_session(
    project_root: &Path,
    config: &ModelControlConfig,
    checkpoint: &Path,
    dataset_root: Option<&Path>,
) -> anyhow::Result<ModelPilotSession> {
    let python_executable = config
        .python_executable
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join(".venv/bin/python"));
    let python_executable = if python_executable.exists() {
        python_executable
    } else {
        PathBuf::from("python3")
    };

    let mut command = Command::new(&python_executable);
    command
        .current_dir(project_root)
        .arg("-u")
        .arg("-m")
        .arg("dfb_reinforcement_learning.live.model_pilot_stdio")
        .arg("--checkpoint")
        .arg(checkpoint);
    if let Some(dataset_root) = dataset_root {
        command.arg("--dataset-root").arg(dataset_root);
    }
    if let Some(device) = config.device.as_deref() {
        command.arg("--device").arg(device);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .env("PYTHONPATH", build_pythonpath(project_root));

    let mut child = command
        .spawn()
        .map_err(|error| anyhow::anyhow!("failed to spawn {:?}: {error}", python_executable))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow::anyhow!("model-control child stdin not available"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("model-control child stdout not available"))?;
    Ok(ModelPilotSession {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

fn spawn_model_pilot_worker(
    project_root: &Path,
    config: &ModelControlConfig,
    checkpoint: &Path,
    dataset_root: Option<&Path>,
) -> anyhow::Result<ModelPilotWorker> {
    let session = spawn_model_pilot_session(project_root, config, checkpoint, dataset_root)?;
    let (request_tx, request_rx) = mpsc::sync_channel::<ModelPilotRequest>(1);
    let (response_tx, response_rx) = mpsc::channel::<ModelPilotResponse>();
    let thread = thread::spawn(move || worker_loop(session, request_rx, response_tx));
    Ok(ModelPilotWorker {
        request_tx,
        response_rx,
        _thread: thread,
    })
}

fn build_pythonpath(project_root: &Path) -> OsString {
    let project_src = project_root.join("project_src");
    let mut path = project_src.into_os_string();
    if let Some(existing) = std::env::var_os("PYTHONPATH")
        && !existing.is_empty()
    {
        path.push(":");
        path.push(existing);
    }
    path
}

fn request_action(
    session: &mut ModelPilotSession,
    request: &ModelPilotRequest,
) -> anyhow::Result<ModelPilotResponse> {
    if let Some(status) = session.child.try_wait()? {
        anyhow::bail!("model-control subprocess already exited with status {status}");
    }
    let request_json = serde_json::to_string(request)?;
    session.stdin.write_all(request_json.as_bytes())?;
    session.stdin.write_all(b"\n")?;
    session.stdin.flush()?;

    let mut response_line = String::new();
    let bytes_read = session.stdout.read_line(&mut response_line)?;
    if bytes_read == 0 {
        anyhow::bail!("model-control subprocess closed stdout");
    }
    let response: ModelPilotResponse = serde_json::from_str(response_line.trim_end())?;
    Ok(response)
}

fn terminate_worker(worker: &mut Option<ModelPilotWorker>) {
    let Some(worker) = worker.take() else {
        return;
    };
    drop(worker.request_tx);
    let _ = worker._thread.join();
}

fn worker_loop(
    mut session: ModelPilotSession,
    request_rx: Receiver<ModelPilotRequest>,
    response_tx: mpsc::Sender<ModelPilotResponse>,
) {
    for request in request_rx {
        let response = match request_action(&mut session, &request) {
            Ok(response) => response,
            Err(error) => ModelPilotResponse {
                action: EnvironmentAction::default(),
                error: Some(error.to_string()),
            },
        };
        if response_tx.send(response).is_err() {
            break;
        }
    }
    let _ = session.child.kill();
    let _ = session.child.wait();
}
