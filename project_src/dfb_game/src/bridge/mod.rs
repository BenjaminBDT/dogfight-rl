pub mod ipc_stub;
pub mod protocol;
pub mod transport;

use std::collections::{HashMap, HashSet, VecDeque};

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::api::commands::ExternalCommandBuffer;
use crate::api::commands::TargetedEnvironmentAction;
use crate::api::types::ObservationBundle;
use crate::api::types::{AircraftObservation, EnvironmentAction, EnvironmentEvent};
use crate::app::AppMode;
use crate::app::schedules::SimulationSet;
use crate::audio::{AudioEventKind, AudioEventQueue};
use crate::bridge::protocol::BridgeRole;
use crate::bridge::protocol::{
    BRIDGE_PROTOCOL_VERSION, BridgeControlSlot, BridgeMessage, BridgePing, BridgePong, ClientHello,
    ClientInputFrame, LobbyClientKind, LobbySessionPhase, LobbySlotState, ProcessedInputTick,
    ServerHello, ServerLobbyStart, ServerLobbyState, ServerSnapshotFrame,
};
use crate::bridge::transport::BridgeDispatchTarget;
use crate::core::config::RepositoryConfig;
use crate::gameplay::combat::{
    AircraftDestroyedEvent, CombatPresentationQueue, Projectile, spawn_projectile_visual_entity,
    spawn_tracer_visual_entity,
};
use crate::gameplay::damage::{AircraftDamageState, AircraftSubsystem};
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::gameplay::reset::PendingMatchReset;
use crate::input::actions::ControlInput;
use crate::presentation::hud::DamageIndicatorQueue;
use crate::presentation::hud::ObservedAircraftRole;
use crate::presentation::tracers::TracerLifetime;
use crate::recording::ActionRecordingState;
use crate::simulation::components::{AircraftRole, AircraftState, ControlAuthority, GunState};
use crate::simulation::resources::SimulationDebugState;
use crate::simulation::systems::step_predicted_aircraft_state;

const MAX_PENDING_SNAPSHOTS: usize = 6;

#[derive(Debug, Clone, Resource)]
pub struct BridgeSmoothingTuning {
    pub snapshot_buffer_fast_consume_lead_ticks: f64,
    pub snapshot_buffer_slow_consume_lead_ticks: f64,
    pub local_correction_max_ack_lag_ticks: u64,
    pub local_snapshot_snap_position_error: f32,
    pub local_snapshot_snap_orientation_dot: f32,
    pub remote_snapshot_snap_position_error: f32,
    pub remote_snapshot_snap_orientation_dot: f32,
    pub remote_snapshot_position_blend: f32,
    pub remote_snapshot_rotation_blend: f32,
    pub remote_snapshot_velocity_blend: f32,
    pub remote_snapshot_angular_blend: f32,
    pub local_correction_acked_factor: f32,
    pub local_correction_unacked_factor: f32,
    pub local_correction_position_base_speed: f32,
    pub local_correction_position_error_scale: f32,
    pub local_correction_rotation_base_speed: f32,
    pub local_correction_rotation_error_scale: f32,
    pub local_correction_velocity_base_speed: f32,
    pub local_correction_velocity_error_scale: f32,
    pub local_correction_alignment_scale: f32,
    pub local_correction_angular_base_speed: f32,
    pub local_correction_angular_error_scale: f32,
    pub local_correction_rotation_alpha_acked_max: f32,
    pub local_correction_rotation_alpha_unacked_max: f32,
    pub local_correction_velocity_alpha_acked_max: f32,
    pub local_correction_velocity_alpha_unacked_max: f32,
    pub local_correction_angular_alpha_acked_max: f32,
    pub local_correction_angular_alpha_unacked_max: f32,
    pub local_correction_close_position_error: f32,
    pub local_correction_close_velocity_error: f32,
    pub local_correction_close_orientation_dot: f32,
}

impl Default for BridgeSmoothingTuning {
    fn default() -> Self {
        Self {
            snapshot_buffer_fast_consume_lead_ticks: 3.0,
            snapshot_buffer_slow_consume_lead_ticks: 1.2,
            local_correction_max_ack_lag_ticks: 3,
            local_snapshot_snap_position_error: 40.0,
            local_snapshot_snap_orientation_dot: 0.8,
            remote_snapshot_snap_position_error: 14.0,
            remote_snapshot_snap_orientation_dot: 0.9,
            remote_snapshot_position_blend: 0.36,
            remote_snapshot_rotation_blend: 0.4,
            remote_snapshot_velocity_blend: 0.42,
            remote_snapshot_angular_blend: 0.42,
            local_correction_acked_factor: 0.24,
            local_correction_unacked_factor: 0.1,
            local_correction_position_base_speed: 2.2,
            local_correction_position_error_scale: 0.2,
            local_correction_rotation_base_speed: 1.8,
            local_correction_rotation_error_scale: 0.02,
            local_correction_velocity_base_speed: 1.3,
            local_correction_velocity_error_scale: 0.018,
            local_correction_alignment_scale: 3.0,
            local_correction_angular_base_speed: 1.5,
            local_correction_angular_error_scale: 0.01,
            local_correction_rotation_alpha_acked_max: 0.08,
            local_correction_rotation_alpha_unacked_max: 0.045,
            local_correction_velocity_alpha_acked_max: 0.08,
            local_correction_velocity_alpha_unacked_max: 0.045,
            local_correction_angular_alpha_acked_max: 0.1,
            local_correction_angular_alpha_unacked_max: 0.05,
            local_correction_close_position_error: 3.5,
            local_correction_close_velocity_error: 8.0,
            local_correction_close_orientation_dot: 0.985,
        }
    }
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct BridgeMode(pub BridgeRole);

impl Default for BridgeMode {
    fn default() -> Self {
        Self(BridgeRole::Client)
    }
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct BridgeEnabled(pub bool);

impl Default for BridgeEnabled {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BridgeClientConnectionStatus {
    Disabled,
    #[default]
    Connecting,
    Reconnecting,
    Timeout,
    SessionMismatch,
    ProtocolMismatch,
    Connected,
}

#[derive(Debug, Resource)]
pub struct BridgeClientInbox {
    pub server_hello: Option<ServerHello>,
    pub latest_snapshot: Option<ServerSnapshotFrame>,
    pub pending_snapshots: VecDeque<ServerSnapshotFrame>,
    pub last_applied_snapshot_tick: Option<u64>,
    pub target_buffer_len: usize,
    pub last_sent_input_tick: Option<u64>,
    pub local_acked_input_tick: Option<u64>,
}

impl Default for BridgeClientInbox {
    fn default() -> Self {
        Self {
            server_hello: None,
            latest_snapshot: None,
            pending_snapshots: VecDeque::new(),
            last_applied_snapshot_tick: None,
            // Keep a small nonzero snapshot buffer even before we have enough
            // transport timing data to estimate lead safely.
            target_buffer_len: 2,
            last_sent_input_tick: None,
            local_acked_input_tick: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalPredictedInputFrame {
    pub tick: u64,
    pub actions: Vec<TargetedEnvironmentAction>,
}

#[derive(Debug, Clone)]
struct LocalPredictedStateFrame {
    tick: u64,
    state: AircraftState,
}

#[derive(Debug, Clone)]
struct LocalPredictionBaseline {
    authoritative_tick: u64,
    state: AircraftState,
}

#[derive(Debug, Default, Resource)]
pub struct LocalPredictionState {
    pub sent_inputs: VecDeque<LocalPredictedInputFrame>,
    pub last_acked_tick: Option<u64>,
    fighter1_baseline: Option<LocalPredictionBaseline>,
    fighter2_baseline: Option<LocalPredictionBaseline>,
    fighter1_history: VecDeque<LocalPredictedStateFrame>,
    fighter2_history: VecDeque<LocalPredictedStateFrame>,
}

#[derive(Debug, Default, Resource)]
pub struct BridgeJitterDiagnostics {
    pub connected_at_client_tick: Option<u64>,
    pub first_baseline_tick: Option<u64>,
    pub local_snap_count: u32,
    pub last_local_snap_tick: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct SnapshotProjectileAudioEntry {
    last_position: Vec3,
    flyby_emitted: bool,
}

#[derive(Debug, Clone, Copy)]
struct RemoteAircraftSnapshot {
    sim_time_seconds: f32,
    position: Vec3,
    orientation: Quat,
}

#[derive(Debug, Clone, Copy)]
pub struct HistoricalAircraftSnapshot {
    pub tick: u64,
    pub position: Vec3,
    pub orientation: Quat,
}

#[derive(Debug, Default, Resource)]
struct SnapshotProjectileAudioState {
    entries: HashMap<u64, SnapshotProjectileAudioEntry>,
}

#[derive(Debug, Default, Resource)]
pub struct BridgeRemoteInterpolationState {
    fighter1_history: VecDeque<RemoteAircraftSnapshot>,
    fighter2_history: VecDeque<RemoteAircraftSnapshot>,
}

#[derive(Debug, Default, Resource)]
pub struct BridgeServerLagCompensationState {
    fighter1_history: VecDeque<HistoricalAircraftSnapshot>,
    fighter2_history: VecDeque<HistoricalAircraftSnapshot>,
}

impl LocalPredictionState {
    pub fn pending_input_count(&self) -> usize {
        self.sent_inputs.len()
    }

    pub fn baseline_tick(&self, role: AircraftRole) -> Option<u64> {
        prediction_baseline(self, role).map(|baseline| baseline.authoritative_tick)
    }
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct RequestedControlRole(pub BridgeControlSlot);

impl Default for RequestedControlRole {
    fn default() -> Self {
        Self(BridgeControlSlot::Fighter1)
    }
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct AssignedControlRole(pub BridgeControlSlot);

impl Default for AssignedControlRole {
    fn default() -> Self {
        Self(BridgeControlSlot::Fighter1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource, Default)]
pub enum LocalPilotMode {
    #[default]
    Human,
    FollowAi,
    ImperfectFollowAi,
    TeacherFollowAi,
    Model,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct BridgeLaunchBinding {
    pub launcher_session_id: Option<String>,
    pub launch_token: Option<String>,
    pub child_kind: Option<LobbyClientKind>,
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct BridgeServerClientSession {
    pub assigned_role: BridgeControlSlot,
    pub last_input_tick: Option<u64>,
    pub estimated_server_tick_offset: Option<f64>,
    pub ready: bool,
    pub client_kind: LobbyClientKind,
    pub launched: bool,
    pub launcher_session_id: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct BridgeLauncherOwnerState {
    pub launcher_client_id: String,
    pub launcher_session_id: u64,
    pub selected_role: BridgeControlSlot,
    pub ready: bool,
    pub launched: bool,
    pub active_child_client_id: Option<String>,
    pub active_child_kind: Option<LobbyClientKind>,
    pub pending_launch_token: Option<String>,
}

#[derive(Debug, Clone, Resource, Default)]
pub struct BridgeServerSessions {
    pub clients: HashMap<String, BridgeServerClientSession>,
    pub owners: HashMap<u64, BridgeLauncherOwnerState>,
    pub next_launcher_session_id: u64,
    pub next_launch_token_id: u64,
}

pub fn connected_player_count(sessions: &BridgeServerSessions) -> usize {
    sessions
        .clients
        .values()
        .filter(|session| {
            session.client_kind != LobbyClientKind::Launcher
                && matches!(
                    session.assigned_role,
                    BridgeControlSlot::Fighter1 | BridgeControlSlot::Fighter2
                )
        })
        .count()
}

pub fn players_ready(sessions: &BridgeServerSessions) -> bool {
    (!sessions.owners.is_empty())
        .then(|| {
            let fighter1_ready = sessions
                .owners
                .values()
                .any(|owner| owner.selected_role == BridgeControlSlot::Fighter1 && owner.ready);
            let fighter2_ready = sessions
                .owners
                .values()
                .any(|owner| owner.selected_role == BridgeControlSlot::Fighter2 && owner.ready);
            fighter1_ready && fighter2_ready
        })
        .unwrap_or_else(|| connected_player_count(sessions) >= 2)
}

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct BridgeLinkState {
    pub remote_authority_active: bool,
    pub client_status: BridgeClientConnectionStatus,
}

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct BridgeTimingState {
    pub last_snapshot_receive_seconds: Option<f64>,
    pub average_snapshot_interval_seconds: Option<f64>,
    pub last_ping_sent_seconds: Option<f64>,
    pub average_rtt_seconds: Option<f64>,
    pub estimated_snapshot_lead_ticks: Option<f64>,
}

#[derive(SystemParam)]
struct BridgeServerDisconnectContext<'w> {
    recording: Option<ResMut<'w, ActionRecordingState>>,
    pending_reset: Option<ResMut<'w, PendingMatchReset>>,
}

#[derive(SystemParam)]
struct BridgeClientRuntimeContext<'w> {
    link_state: ResMut<'w, BridgeLinkState>,
    handshake: ResMut<'w, BridgeClientHandshakeState>,
    timing_state: ResMut<'w, BridgeTimingState>,
    remote_interpolation: ResMut<'w, BridgeRemoteInterpolationState>,
}

#[derive(Debug, Clone, Copy, Resource, Default)]
struct BridgeServerMetricsState {
    next_log_at_seconds: f64,
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct BridgeServerMetricsControl {
    pub enabled: bool,
    pub interval_seconds: f64,
}

impl Default for BridgeServerMetricsControl {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 5.0,
        }
    }
}

#[derive(Debug, Default, Resource)]
pub struct BridgeClientHandshakeState {
    pub connect_started_tick: Option<u64>,
    pub last_hello_tick: Option<u64>,
    pub timeout_count: u32,
    pub next_retry_tick: Option<u64>,
}

const CLIENT_HELLO_RESEND_TICKS: u64 = 30;
const CLIENT_HANDSHAKE_TIMEOUT_TICKS: u64 = 180;
const CLIENT_HANDSHAKE_RETRY_BASE_TICKS: u64 = 60;
const CLIENT_HANDSHAKE_RETRY_MAX_TICKS: u64 = 300;

#[derive(Debug, Default)]
struct LocalCorrectionLogState {
    next_log_at_seconds: f64,
}

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<transport::IpcBridgeState>()
            .init_resource::<transport::BridgeEndpointConfig>()
            .init_resource::<BridgeClientInbox>()
            .init_resource::<LocalPredictionState>()
            .init_resource::<BridgeJitterDiagnostics>()
            .init_resource::<BridgeRemoteInterpolationState>()
            .init_resource::<BridgeServerLagCompensationState>()
            .init_resource::<SnapshotProjectileAudioState>()
            .init_resource::<RequestedControlRole>()
            .init_resource::<AssignedControlRole>()
            .init_resource::<LocalPilotMode>()
            .init_resource::<BridgeLaunchBinding>()
            .init_resource::<BridgeServerSessions>()
            .init_resource::<BridgeLinkState>()
            .init_resource::<BridgeTimingState>()
            .init_resource::<BridgeServerMetricsState>()
            .init_resource::<BridgeServerMetricsControl>()
            .init_resource::<BridgeClientHandshakeState>()
            .init_resource::<BridgeSmoothingTuning>()
            .init_resource::<BridgeEnabled>()
            .init_resource::<BridgeMode>();

        app.add_systems(Startup, connect_bridge_endpoint)
            .add_systems(
                Update,
                (
                    resend_client_hello,
                    send_periodic_bridge_ping.after(resend_client_hello),
                    track_client_connection_epoch.after(send_periodic_bridge_ping),
                    log_server_bridge_metrics.after(send_periodic_bridge_ping),
                    apply_assigned_control_role.after(connect_bridge_endpoint),
                ),
            )
            .add_systems(
                FixedUpdate,
                (
                    drain_bridge_messages
                        .in_set(SimulationSet::GatherInput)
                        .before(crate::api::commands::apply_external_commands),
                    apply_server_snapshot_to_world
                        .in_set(SimulationSet::GatherInput)
                        .after(drain_bridge_messages),
                    record_local_prediction_history
                        .in_set(SimulationSet::ResolveGameplay)
                        .after(crate::simulation::systems::predict_local_environment_collisions),
                    record_server_lag_comp_history
                        .in_set(SimulationSet::ResolveGameplay)
                        .after(crate::simulation::systems::predict_local_environment_collisions),
                    publish_client_input_frame.in_set(SimulationSet::ResolveGameplay),
                    publish_server_snapshot
                        .in_set(SimulationSet::ProduceSnapshot)
                        .after(crate::api::snapshot::capture_world_snapshot),
                ),
            );
    }
}

fn connect_bridge_endpoint(
    bridge_enabled: Res<BridgeEnabled>,
    config: Res<RepositoryConfig>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    requested_role: Res<RequestedControlRole>,
    launch_binding: Res<BridgeLaunchBinding>,
    mut link_state: ResMut<BridgeLinkState>,
    mut handshake: ResMut<BridgeClientHandshakeState>,
    mut state: ResMut<transport::IpcBridgeState>,
) {
    if !bridge_enabled.0 {
        link_state.client_status = BridgeClientConnectionStatus::Disabled;
        return;
    }
    transport::connect_endpoint(&endpoint, mode.0, &mut state);

    match mode.0 {
        BridgeRole::Client => {
            link_state.client_status = BridgeClientConnectionStatus::Connecting;
            handshake.connect_started_tick = None;
            handshake.last_hello_tick = None;
            handshake.timeout_count = 0;
            handshake.next_retry_tick = None;
            send_client_hello(
                &config,
                &endpoint,
                requested_role.0,
                &launch_binding,
                &mut state,
            );
        }
        BridgeRole::Server => {
            info!(
                "bridge server starting session '{}' via {:?} at {}",
                endpoint.session, endpoint.transport, endpoint.address
            );
        }
    }
}

fn track_client_connection_epoch(
    mode: Res<BridgeMode>,
    bridge_state: Res<transport::IpcBridgeState>,
    link_state: Res<BridgeLinkState>,
    mut diagnostics: ResMut<BridgeJitterDiagnostics>,
    mut previous_status: Local<Option<BridgeClientConnectionStatus>>,
) {
    if mode.0 != BridgeRole::Client {
        return;
    }

    let status = link_state.client_status;
    if *previous_status != Some(status) {
        match status {
            BridgeClientConnectionStatus::Connected => {
                diagnostics.connected_at_client_tick = bridge_state.last_client_tick.or(Some(0));
                diagnostics.first_baseline_tick = None;
                diagnostics.local_snap_count = 0;
                diagnostics.last_local_snap_tick = None;
            }
            BridgeClientConnectionStatus::Connecting
            | BridgeClientConnectionStatus::Reconnecting
            | BridgeClientConnectionStatus::Timeout
            | BridgeClientConnectionStatus::SessionMismatch
            | BridgeClientConnectionStatus::ProtocolMismatch
            | BridgeClientConnectionStatus::Disabled => {
                diagnostics.connected_at_client_tick = None;
                diagnostics.first_baseline_tick = None;
                diagnostics.local_snap_count = 0;
                diagnostics.last_local_snap_tick = None;
            }
        }
        *previous_status = Some(status);
    }
}

fn resend_client_hello(
    bridge_enabled: Res<BridgeEnabled>,
    config: Res<RepositoryConfig>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    requested_role: Res<RequestedControlRole>,
    launch_binding: Res<BridgeLaunchBinding>,
    mut link_state: ResMut<BridgeLinkState>,
    inbox: Res<BridgeClientInbox>,
    debug: Res<SimulationDebugState>,
    mut handshake: ResMut<BridgeClientHandshakeState>,
    mut state: ResMut<transport::IpcBridgeState>,
) {
    if !bridge_enabled.0 {
        return;
    }
    if mode.0 != BridgeRole::Client || inbox.server_hello.is_some() {
        return;
    }
    if !matches!(
        link_state.client_status,
        BridgeClientConnectionStatus::Connecting
            | BridgeClientConnectionStatus::Reconnecting
            | BridgeClientConnectionStatus::Timeout
    ) {
        return;
    }

    let tick = debug.tick_count;
    if let Some(next_retry_tick) = handshake.next_retry_tick {
        if tick < next_retry_tick {
            return;
        }
        let reconnecting = link_state.client_status == BridgeClientConnectionStatus::Reconnecting;
        handshake.next_retry_tick = None;
        handshake.connect_started_tick = None;
        link_state.client_status = if reconnecting {
            BridgeClientConnectionStatus::Reconnecting
        } else {
            BridgeClientConnectionStatus::Connecting
        };
    }
    handshake.connect_started_tick.get_or_insert(tick);
    if handshake
        .connect_started_tick
        .is_some_and(|start_tick| tick.saturating_sub(start_tick) >= CLIENT_HANDSHAKE_TIMEOUT_TICKS)
    {
        link_state.client_status = BridgeClientConnectionStatus::Timeout;
        handshake.timeout_count = handshake.timeout_count.saturating_add(1);
        let backoff = (CLIENT_HANDSHAKE_RETRY_BASE_TICKS
            .saturating_mul(1_u64 << handshake.timeout_count.min(4)))
        .min(CLIENT_HANDSHAKE_RETRY_MAX_TICKS);
        handshake.next_retry_tick = Some(tick.saturating_add(backoff));
        handshake.connect_started_tick = None;
        handshake.last_hello_tick = None;
        return;
    }
    if handshake
        .last_hello_tick
        .is_some_and(|last_tick| tick.saturating_sub(last_tick) < CLIENT_HELLO_RESEND_TICKS)
    {
        return;
    }

    send_client_hello(
        &config,
        &endpoint,
        requested_role.0,
        &launch_binding,
        &mut state,
    );
    handshake.last_hello_tick = Some(tick);
}

fn send_periodic_bridge_ping(
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    link_state: Res<BridgeLinkState>,
    inbox: Res<BridgeClientInbox>,
    time: Res<Time<Real>>,
    mut timing_state: ResMut<BridgeTimingState>,
    mut state: ResMut<transport::IpcBridgeState>,
) {
    if !bridge_enabled.0 {
        return;
    }
    if mode.0 != BridgeRole::Client
        || inbox.server_hello.is_none()
        || !link_state.remote_authority_active
    {
        return;
    }

    let now = time.elapsed_secs_f64();
    if timing_state
        .last_ping_sent_seconds
        .is_some_and(|last| now - last < 1.0)
    {
        return;
    }

    transport::send_message(
        &endpoint,
        BridgeRole::Client,
        &mut state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::Ping(BridgePing {
            sent_at_seconds: now,
        }),
    );
    timing_state.last_ping_sent_seconds = Some(now);
}

fn record_local_prediction_history(
    mode: Res<BridgeMode>,
    bridge_enabled: Res<BridgeEnabled>,
    bridge_link: Res<BridgeLinkState>,
    requested_role: Res<RequestedControlRole>,
    assigned_role: Res<AssignedControlRole>,
    debug: Res<SimulationDebugState>,
    mut prediction: ResMut<LocalPredictionState>,
    query: Query<(&AircraftRole, &AircraftState)>,
) {
    if !bridge_enabled.0
        || mode.0 != BridgeRole::Client
        || !bridge_link.remote_authority_active
        || matches!(requested_role.0, BridgeControlSlot::Spectator)
    {
        return;
    }
    let Some(local_role) = bridge_slot_aircraft_role(assigned_role.0) else {
        return;
    };
    for (role, state) in &query {
        if *role == local_role {
            push_predicted_state_history(&mut prediction, *role, debug.tick_count, state);
            break;
        }
    }
}

const SERVER_LAG_COMP_HISTORY_LIMIT: usize = 128;

fn server_lag_comp_history_mut(
    state: &mut BridgeServerLagCompensationState,
    role: AircraftRole,
) -> &mut VecDeque<HistoricalAircraftSnapshot> {
    match role {
        AircraftRole::Fighter1 => &mut state.fighter1_history,
        AircraftRole::Fighter2 => &mut state.fighter2_history,
    }
}

pub fn sample_server_historical_aircraft_snapshot(
    state: &BridgeServerLagCompensationState,
    role: AircraftRole,
    tick: u64,
) -> Option<HistoricalAircraftSnapshot> {
    let history = match role {
        AircraftRole::Fighter1 => &state.fighter1_history,
        AircraftRole::Fighter2 => &state.fighter2_history,
    };
    history
        .iter()
        .rev()
        .find(|entry| entry.tick <= tick)
        .copied()
        .or_else(|| history.front().copied())
}

fn record_server_lag_comp_history(
    mode: Res<BridgeMode>,
    debug: Res<SimulationDebugState>,
    mut lag_comp: ResMut<BridgeServerLagCompensationState>,
    query: Query<(&AircraftRole, &AircraftState)>,
) {
    if mode.0 != BridgeRole::Server {
        return;
    }
    for (role, state) in &query {
        let history = server_lag_comp_history_mut(&mut lag_comp, *role);
        if history
            .back()
            .is_some_and(|entry| entry.tick == debug.tick_count)
        {
            history.pop_back();
        }
        history.push_back(HistoricalAircraftSnapshot {
            tick: debug.tick_count,
            position: state.position,
            orientation: state.orientation,
        });
        while history.len() > SERVER_LAG_COMP_HISTORY_LIMIT {
            history.pop_front();
        }
    }
}

fn drain_bridge_messages(
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    config: Res<RepositoryConfig>,
    current_phase: Res<State<MatchPhase>>,
    debug: Res<SimulationDebugState>,
    time: Res<Time<Real>>,
    mut bridge_state: ResMut<transport::IpcBridgeState>,
    mut commands: ResMut<ExternalCommandBuffer>,
    mut inbox: ResMut<BridgeClientInbox>,
    mut prediction: ResMut<LocalPredictionState>,
    mut server_sessions: ResMut<BridgeServerSessions>,
    mut client_ctx: BridgeClientRuntimeContext,
    mut disconnect_ctx: BridgeServerDisconnectContext,
    smoothing: Res<BridgeSmoothingTuning>,
    mut authority_query: Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    if !bridge_enabled.0 {
        return;
    }
    let messages = transport::drain_messages(&endpoint, mode.0, &mut bridge_state);
    let transport_snapshot = transport::snapshot(&bridge_state);
    let link_state = &mut client_ctx.link_state;
    let handshake = &mut client_ctx.handshake;
    let timing_state = &mut client_ctx.timing_state;

    if mode.0 == BridgeRole::Server {
        let active_client_ids = transport_snapshot.connected_client_ids.clone();
        let previous_sessions = server_sessions.clients.clone();
        server_sessions
            .clients
            .retain(|client_id, _| active_client_ids.iter().any(|active| active == client_id));
        for (client_id, released) in &previous_sessions {
            if !server_sessions.clients.contains_key(client_id) {
                info!(
                    "bridge session released client_id={} role={:?}",
                    client_id, released.assigned_role
                );
                if released.client_kind == LobbyClientKind::Launcher {
                    if let Some(launcher_session_id) = released.launcher_session_id {
                        if let Some(owner) = server_sessions.owners.get(&launcher_session_id)
                            && let Some(active_child_client_id) =
                                owner.active_child_client_id.clone()
                        {
                            transport::force_disconnect_client(
                                &mut bridge_state,
                                &active_child_client_id,
                            );
                            server_sessions.clients.remove(&active_child_client_id);
                        }
                        server_sessions.owners.remove(&launcher_session_id);
                    }
                } else if let Some(launcher_session_id) = released.launcher_session_id
                    && let Some(owner) = server_sessions.owners.get_mut(&launcher_session_id)
                {
                    owner.active_child_client_id = None;
                    owner.active_child_kind = None;
                    owner.launched = false;
                    owner.ready = false;
                    owner.pending_launch_token = None;
                }
            }
        }
        if server_sessions.clients.len() != previous_sessions.len() {
            let player_left_during_match = *current_phase.get() == MatchPhase::Running
                && previous_sessions.iter().any(|(client_id, released)| {
                    !server_sessions.clients.contains_key(client_id)
                        && matches!(
                            released.assigned_role,
                            BridgeControlSlot::Fighter1 | BridgeControlSlot::Fighter2
                        )
                });
            if player_left_during_match {
                close_lobby_after_player_leave(
                    &endpoint,
                    &mut bridge_state,
                    &mut server_sessions,
                    disconnect_ctx.recording.as_deref_mut(),
                    disconnect_ctx.pending_reset.as_deref_mut(),
                    "player-left",
                );
            }
            apply_server_authorities(&server_sessions, &mut authority_query);
            broadcast_lobby_state(
                &endpoint,
                &mut bridge_state,
                &server_sessions,
                &config.game.active_scene,
                *current_phase.get(),
                None,
            );
        }
    } else if mode.0 == BridgeRole::Client && transport::client_should_reconnect(&bridge_state) {
        let was_connected = link_state.remote_authority_active
            || inbox.server_hello.is_some()
            || link_state.client_status == BridgeClientConnectionStatus::Connected;
        link_state.remote_authority_active = false;
        if link_state.client_status != BridgeClientConnectionStatus::Disabled {
            link_state.client_status = if was_connected {
                BridgeClientConnectionStatus::Reconnecting
            } else {
                BridgeClientConnectionStatus::Connecting
            };
        }
        inbox.server_hello = None;
        inbox.latest_snapshot = None;
        inbox.pending_snapshots.clear();
        inbox.local_acked_input_tick = None;
        prediction.last_acked_tick = None;
        prediction.sent_inputs.clear();
        prediction.fighter1_baseline = None;
        prediction.fighter2_baseline = None;
        prediction.fighter1_history.clear();
        prediction.fighter2_history.clear();
        client_ctx.remote_interpolation.fighter1_history.clear();
        client_ctx.remote_interpolation.fighter2_history.clear();
        handshake.connect_started_tick = None;
        handshake.last_hello_tick = None;
        handshake.next_retry_tick = None;
    }

    for inbound in messages {
        match (mode.0, inbound.client_id, inbound.message) {
            (BridgeRole::Server, Some(client_id), BridgeMessage::ClientHello(hello)) => {
                if hello.protocol_version != BRIDGE_PROTOCOL_VERSION {
                    warn!(
                        "bridge protocol mismatch client_id={} client_version={} server_version={}",
                        client_id, hello.protocol_version, BRIDGE_PROTOCOL_VERSION
                    );
                    continue;
                }
                if hello.requested_session != endpoint.session {
                    warn!(
                        "bridge session mismatch client_id={} requested_session='{}' server_session='{}'",
                        client_id, hello.requested_session, endpoint.session
                    );
                    transport::send_message(
                        &endpoint,
                        BridgeRole::Server,
                        &mut bridge_state,
                        BridgeDispatchTarget::Client(client_id),
                        BridgeMessage::ServerHello(ServerHello {
                            protocol_version: BRIDGE_PROTOCOL_VERSION,
                            accepted_session: endpoint.session.clone(),
                            accepted_scene: config.game.active_scene.clone(),
                            assigned_role: BridgeControlSlot::Spectator,
                            fixed_time_step_seconds: config.game.fixed_time_step_seconds,
                        }),
                    );
                    continue;
                }
                let assigned_role = if hello.launcher_session_id.is_some()
                    || hello.launch_token.is_some()
                    || hello.child_kind.is_some()
                {
                    bind_child_to_owner(&client_id, &hello, &mut server_sessions).unwrap_or_else(
                        || {
                            warn!(
                                "bridge child binding rejected client_id={} launcher_session_id={:?} child_kind={:?}",
                                client_id, hello.launcher_session_id, hello.child_kind
                            );
                            assign_unbound_child_spectator(
                                &client_id,
                                hello.child_kind.unwrap_or(LobbyClientKind::Observer),
                                &mut server_sessions,
                            )
                        },
                    )
                } else {
                    allocate_role(&client_id, hello.requested_role, &mut server_sessions)
                };
                apply_server_authorities(&server_sessions, &mut authority_query);
                info!(
                    "bridge session assigned client_id={} requested={:?} assigned={:?}",
                    client_id, hello.requested_role, assigned_role
                );
                transport::send_message(
                    &endpoint,
                    BridgeRole::Server,
                    &mut bridge_state,
                    BridgeDispatchTarget::Client(client_id.clone()),
                    BridgeMessage::ServerHello(ServerHello {
                        protocol_version: BRIDGE_PROTOCOL_VERSION,
                        accepted_session: endpoint.session.clone(),
                        accepted_scene: config.game.active_scene.clone(),
                        assigned_role,
                        fixed_time_step_seconds: config.game.fixed_time_step_seconds,
                    }),
                );
                broadcast_lobby_state(
                    &endpoint,
                    &mut bridge_state,
                    &server_sessions,
                    &config.game.active_scene,
                    *current_phase.get(),
                    Some(client_id.as_str()),
                );
            }
            (BridgeRole::Server, Some(client_id), BridgeMessage::ClientLobbyHello(hello)) => {
                if hello.protocol_version != BRIDGE_PROTOCOL_VERSION {
                    warn!(
                        "bridge lobby protocol mismatch client_id={} client_version={} server_version={}",
                        client_id, hello.protocol_version, BRIDGE_PROTOCOL_VERSION
                    );
                    continue;
                }
                if hello.requested_session != endpoint.session {
                    continue;
                }
                let launcher_session_id = hello
                    .launcher_session_id
                    .as_deref()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or_else(|| {
                        server_sessions.next_launcher_session_id += 1;
                        server_sessions.next_launcher_session_id
                    });
                server_sessions.clients.insert(
                    client_id.clone(),
                    BridgeServerClientSession {
                        assigned_role: BridgeControlSlot::Spectator,
                        last_input_tick: None,
                        estimated_server_tick_offset: None,
                        ready: false,
                        client_kind: LobbyClientKind::Launcher,
                        launched: false,
                        launcher_session_id: Some(launcher_session_id),
                    },
                );
                server_sessions
                    .owners
                    .entry(launcher_session_id)
                    .and_modify(|owner| {
                        owner.launcher_client_id = client_id.clone();
                    })
                    .or_insert(BridgeLauncherOwnerState {
                        launcher_client_id: client_id.clone(),
                        launcher_session_id,
                        selected_role: BridgeControlSlot::Spectator,
                        ready: false,
                        launched: false,
                        active_child_client_id: None,
                        active_child_kind: None,
                        pending_launch_token: None,
                    });
                broadcast_lobby_state(
                    &endpoint,
                    &mut bridge_state,
                    &server_sessions,
                    &config.game.active_scene,
                    *current_phase.get(),
                    Some(client_id.as_str()),
                );
            }
            (
                BridgeRole::Server,
                Some(client_id),
                BridgeMessage::ClientLobbySelectRole(message),
            ) => {
                set_or_allocate_role(
                    &client_id,
                    message.requested_role,
                    &mut server_sessions,
                    LobbyClientKind::Launcher,
                );
                apply_server_authorities(&server_sessions, &mut authority_query);
                broadcast_lobby_state(
                    &endpoint,
                    &mut bridge_state,
                    &server_sessions,
                    &config.game.active_scene,
                    *current_phase.get(),
                    Some(client_id.as_str()),
                );
            }
            (BridgeRole::Server, Some(client_id), BridgeMessage::ClientLobbyReady(message)) => {
                if let Some(session) = server_sessions.clients.get_mut(&client_id) {
                    session.ready = message.ready;
                    if !message.ready {
                        session.launched = false;
                    }
                    if let Some(launcher_session_id) = session.launcher_session_id
                        && let Some(owner) = server_sessions.owners.get_mut(&launcher_session_id)
                    {
                        owner.ready = message.ready;
                        if !message.ready {
                            owner.launched = false;
                            owner.pending_launch_token = None;
                        }
                    }
                }
                broadcast_lobby_state(
                    &endpoint,
                    &mut bridge_state,
                    &server_sessions,
                    &config.game.active_scene,
                    *current_phase.get(),
                    Some(client_id.as_str()),
                );
            }
            (BridgeRole::Server, Some(client_id), BridgeMessage::ClientInputFrame(frame)) => {
                let assigned_role = server_sessions
                    .clients
                    .get(&client_id)
                    .map(|session| session.assigned_role)
                    .unwrap_or(BridgeControlSlot::Spectator);
                if let Some(session) = server_sessions.clients.get_mut(&client_id) {
                    session.last_input_tick = Some(frame.tick);
                    let sample_offset = debug.tick_count as f64 - frame.tick as f64;
                    session.estimated_server_tick_offset = Some(
                        session
                            .estimated_server_tick_offset
                            .map(|previous| previous * 0.9 + sample_offset * 0.1)
                            .unwrap_or(sample_offset),
                    );
                }
                commands
                    .targeted_actions
                    .extend(frame.actions.into_iter().filter(|action| {
                        Some(action.role) == bridge_slot_aircraft_role(assigned_role)
                    }));
            }
            (BridgeRole::Server, Some(client_id), BridgeMessage::Ping(ping)) => {
                transport::send_message(
                    &endpoint,
                    BridgeRole::Server,
                    &mut bridge_state,
                    BridgeDispatchTarget::Client(client_id),
                    BridgeMessage::Pong(BridgePong {
                        sent_at_seconds: ping.sent_at_seconds,
                    }),
                );
            }
            (BridgeRole::Client, _, BridgeMessage::ServerHello(hello)) => {
                if hello.protocol_version != BRIDGE_PROTOCOL_VERSION {
                    link_state.client_status = BridgeClientConnectionStatus::ProtocolMismatch;
                    warn!(
                        "ignoring server hello with incompatible protocol_version={} (expected {})",
                        hello.protocol_version, BRIDGE_PROTOCOL_VERSION
                    );
                    continue;
                }
                if hello.accepted_session != endpoint.session {
                    link_state.client_status = BridgeClientConnectionStatus::SessionMismatch;
                    warn!(
                        "ignoring server hello from mismatched session '{}' (expected '{}')",
                        hello.accepted_session, endpoint.session
                    );
                    continue;
                }
                link_state.remote_authority_active = true;
                link_state.client_status = BridgeClientConnectionStatus::Connected;
                handshake.connect_started_tick = None;
                handshake.last_hello_tick = None;
                handshake.timeout_count = 0;
                handshake.next_retry_tick = None;
                info!(
                    "bridge client accepted session='{}' scene='{}' assigned_role={:?} dt={:.5}",
                    hello.accepted_session,
                    hello.accepted_scene,
                    hello.assigned_role,
                    hello.fixed_time_step_seconds
                );
                inbox.server_hello = Some(hello);
            }
            (BridgeRole::Client, _, BridgeMessage::ServerLobbyState(_)) => {}
            (BridgeRole::Client, _, BridgeMessage::ServerLobbyStart(_)) => {}
            (BridgeRole::Client, _, BridgeMessage::ServerLobbyClose(_)) => {}
            (BridgeRole::Client, _, BridgeMessage::ServerSnapshotFrame(frame)) => {
                if frame.session != endpoint.session {
                    continue;
                }
                link_state.remote_authority_active = true;
                link_state.client_status = BridgeClientConnectionStatus::Connected;
                handshake.connect_started_tick = None;
                handshake.last_hello_tick = None;
                handshake.timeout_count = 0;
                handshake.next_retry_tick = None;
                let now = time.elapsed_secs_f64();
                if let Some(previous) = timing_state.last_snapshot_receive_seconds {
                    let interval = (now - previous).max(0.0);
                    timing_state.average_snapshot_interval_seconds = Some(
                        timing_state
                            .average_snapshot_interval_seconds
                            .map(|average| average * 0.85 + interval * 0.15)
                            .unwrap_or(interval),
                    );
                }
                timing_state.last_snapshot_receive_seconds = Some(now);
                if let Some(client_tick) = transport_snapshot.last_client_tick {
                    let lead_ticks = frame.tick as f64 - client_tick as f64;
                    let estimated_lead = timing_state
                        .estimated_snapshot_lead_ticks
                        .map(|previous| previous * 0.85 + lead_ticks * 0.15)
                        .unwrap_or(lead_ticks);
                    timing_state.estimated_snapshot_lead_ticks = Some(estimated_lead);
                    inbox.target_buffer_len = if estimated_lead
                        > smoothing.snapshot_buffer_fast_consume_lead_ticks
                    {
                        1
                    } else if estimated_lead < smoothing.snapshot_buffer_slow_consume_lead_ticks {
                        3
                    } else {
                        2
                    };
                } else {
                    inbox.target_buffer_len = inbox.target_buffer_len.max(2);
                }
                inbox.local_acked_input_tick = inbox.server_hello.as_ref().and_then(|hello| {
                    frame
                        .processed_input_ticks
                        .iter()
                        .find(|processed| processed.slot == hello.assigned_role)
                        .map(|processed| processed.tick)
                });
                prediction.last_acked_tick = inbox.local_acked_input_tick;
                if let Some(acked_tick) = prediction.last_acked_tick {
                    while prediction
                        .sent_inputs
                        .front()
                        .is_some_and(|input| input.tick <= acked_tick)
                    {
                        prediction.sent_inputs.pop_front();
                    }
                }
                for aircraft in &frame.observation.state.aircraft {
                    if let Some(role) = aircraft_role_name(aircraft.role.as_str()) {
                        push_remote_aircraft_snapshot(
                            &mut client_ctx.remote_interpolation,
                            role,
                            RemoteAircraftSnapshot {
                                sim_time_seconds: frame.observation.state.sim_time_seconds,
                                position: Vec3::from_array(aircraft.position),
                                orientation: Quat::from_array(aircraft.orientation_quat)
                                    .normalize(),
                            },
                        );
                    }
                }
                inbox.latest_snapshot = Some(frame.clone());
                inbox.pending_snapshots.push_back(frame);
                while inbox.pending_snapshots.len() > MAX_PENDING_SNAPSHOTS {
                    inbox.pending_snapshots.pop_front();
                }
            }
            (BridgeRole::Client, _, BridgeMessage::Pong(pong)) => {
                let now = time.elapsed_secs_f64();
                let rtt = (now - pong.sent_at_seconds).max(0.0);
                timing_state.average_rtt_seconds = Some(
                    timing_state
                        .average_rtt_seconds
                        .map(|average| average * 0.8 + rtt * 0.2)
                        .unwrap_or(rtt),
                );
            }
            _ => {}
        }
    }
    if mode.0 == BridgeRole::Server {
        dispatch_lobby_start_messages(
            &endpoint,
            &mut bridge_state,
            &mut server_sessions,
            &config.game.active_scene,
            *current_phase.get(),
        );
    }
}

fn close_lobby_after_player_leave(
    endpoint: &transport::BridgeEndpointConfig,
    bridge_state: &mut transport::IpcBridgeState,
    sessions: &mut BridgeServerSessions,
    recording: Option<&mut ActionRecordingState>,
    pending_reset: Option<&mut PendingMatchReset>,
    reason: &str,
) {
    for (client_id, session) in sessions.clients.iter_mut() {
        if session.client_kind == LobbyClientKind::Launcher {
            transport::send_message(
                endpoint,
                BridgeRole::Server,
                bridge_state,
                BridgeDispatchTarget::Client(client_id.clone()),
                BridgeMessage::ServerLobbyClose(crate::bridge::protocol::ServerLobbyClose {
                    accepted_session: endpoint.session.clone(),
                    reason: reason.to_string(),
                }),
            );
            session.ready = false;
            session.launched = false;
            if let Some(launcher_session_id) = session.launcher_session_id
                && let Some(owner) = sessions.owners.get_mut(&launcher_session_id)
            {
                owner.ready = false;
                owner.launched = false;
                owner.active_child_client_id = None;
                owner.active_child_kind = None;
                owner.pending_launch_token = None;
            }
        }
    }
    if let Some(recording) = recording
        && (recording.active || recording.pending_start || recording.pending_stop)
    {
        recording.pending_start = false;
        recording.pending_stop = true;
    }
    if let Some(pending_reset) = pending_reset {
        pending_reset.requested = true;
    }
}

fn publish_server_snapshot(
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    server_sessions: Res<BridgeServerSessions>,
    mut bridge_state: ResMut<transport::IpcBridgeState>,
    observation: Res<ObservationBundle>,
) {
    if !bridge_enabled.0 {
        return;
    }
    if mode.0 != BridgeRole::Server {
        return;
    }

    transport::send_message(
        &endpoint,
        BridgeRole::Server,
        &mut bridge_state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ServerSnapshotFrame(ServerSnapshotFrame {
            tick: observation.state.tick,
            session: endpoint.session.clone(),
            connected_clients: server_sessions.clients.len(),
            occupied_slots: occupied_slots(&server_sessions),
            processed_input_ticks: processed_input_ticks(&server_sessions),
            observation: observation.clone(),
        }),
    );
}

fn publish_client_input_frame(
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Res<transport::BridgeEndpointConfig>,
    mode: Res<BridgeMode>,
    mut bridge_state: ResMut<transport::IpcBridgeState>,
    mut inbox: ResMut<BridgeClientInbox>,
    mut prediction: ResMut<LocalPredictionState>,
    debug: Res<SimulationDebugState>,
    query: Query<(&AircraftRole, &ControlAuthority, &ControlInput)>,
) {
    if !bridge_enabled.0 {
        return;
    }
    if mode.0 != BridgeRole::Client {
        return;
    }

    let mut actions = Vec::new();
    for (role, authority, input) in &query {
        if matches!(
            authority,
            ControlAuthority::Human | ControlAuthority::ExternalAgent
        ) {
            actions.push(TargetedEnvironmentAction {
                role: *role,
                action: EnvironmentAction::from(*input),
            });
        }
    }

    if actions.is_empty() {
        return;
    }

    inbox.last_sent_input_tick = Some(debug.tick_count);
    prediction.sent_inputs.push_back(LocalPredictedInputFrame {
        tick: debug.tick_count,
        actions: actions.clone(),
    });

    transport::send_message(
        &endpoint,
        BridgeRole::Client,
        &mut bridge_state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ClientInputFrame(ClientInputFrame {
            tick: debug.tick_count,
            actions,
        }),
    );
}

fn apply_server_snapshot_to_world(
    mut commands: Commands,
    mode: Res<BridgeMode>,
    config: Res<RepositoryConfig>,
    mut inbox: ResMut<BridgeClientInbox>,
    smoothing: Res<BridgeSmoothingTuning>,
    diagnostics_time: (Res<Time<Real>>, Local<LocalCorrectionLogState>),
    time_state: (
        ResMut<MatchClock>,
        Res<State<MatchPhase>>,
        ResMut<NextState<MatchPhase>>,
    ),
    mut meshes: Option<ResMut<Assets<Mesh>>>,
    mut materials: Option<ResMut<Assets<StandardMaterial>>>,
    role_state: (
        Option<Res<ObservedAircraftRole>>,
        Option<Res<AssignedControlRole>>,
    ),
    mut audio_events: Option<ResMut<AudioEventQueue>>,
    mut damage_indicators: Option<ResMut<DamageIndicatorQueue>>,
    mut presentation_queue: Option<ResMut<CombatPresentationQueue>>,
    prediction_state: (
        ResMut<LocalPredictionState>,
        ResMut<BridgeJitterDiagnostics>,
        ResMut<SnapshotProjectileAudioState>,
    ),
    mut query: Query<(
        &AircraftRole,
        &mut AircraftState,
        &crate::simulation::components::AircraftPerformance,
        &mut AircraftDamageState,
        &mut GunState,
    )>,
    mut dynamic_entities: ParamSet<(
        Query<Entity, With<Projectile>>,
        Query<Entity, With<TracerLifetime>>,
    )>,
) {
    if mode.0 != BridgeRole::Client {
        return;
    }

    let snapshot = if inbox.pending_snapshots.len() > inbox.target_buffer_len
        || (inbox.last_applied_snapshot_tick.is_none() && inbox.pending_snapshots.len() == 1)
    {
        inbox.pending_snapshots.pop_front()
    } else {
        None
    };
    let Some(snapshot) = snapshot else {
        return;
    };
    if inbox.last_applied_snapshot_tick == Some(snapshot.tick) {
        return;
    }

    let (mut clock, current_phase, mut next_phase) = time_state;
    let (mut prediction, mut jitter_diagnostics, mut projectile_audio) = prediction_state;
    let (observed_role, assigned_role) = role_state;
    let (time, mut correction_log) = diagnostics_time;
    clock.elapsed_seconds = snapshot.observation.state.sim_time_seconds;
    if let Some(snapshot_phase) = match_phase_name(&snapshot.observation.state.match_phase)
        && *current_phase.get() != snapshot_phase
    {
        next_phase.set(snapshot_phase);
    }
    let locally_controlled_role = assigned_role
        .as_deref()
        .and_then(|role| bridge_slot_aircraft_role(role.0));
    for aircraft in &snapshot.observation.state.aircraft {
        let role = aircraft_role_name(aircraft.role.as_str());
        let Some(role) = role else {
            continue;
        };

        for (entity_role, mut state, performance, mut damage, mut gun) in &mut query {
            if *entity_role != role {
                continue;
            }

            let target_position = Vec3::from_array(aircraft.position);
            let target_orientation = Quat::from_array(aircraft.orientation_quat).normalize();
            let target_velocity = Vec3::from_array(aircraft.linear_velocity);
            let target_angular_rates = Vec3::from_array(aircraft.angular_velocity_deg);
            let is_locally_controlled = Some(role) == locally_controlled_role;
            let position_error = state.position.distance(target_position);
            let orientation_error = state.orientation.dot(target_orientation).abs();
            let should_snap = if is_locally_controlled {
                position_error > smoothing.local_snapshot_snap_position_error
                    || orientation_error < smoothing.local_snapshot_snap_orientation_dot
            } else {
                position_error > smoothing.remote_snapshot_snap_position_error
                    || orientation_error < smoothing.remote_snapshot_snap_orientation_dot
            };
            let authoritative_state = AircraftState {
                position: target_position,
                velocity: target_velocity,
                orientation: target_orientation,
                forward: Vec3::from_array(aircraft.forward).normalize_or_zero(),
                angular_rates_deg: target_angular_rates,
                throttle: aircraft.throttle,
                hit_points: aircraft.hit_points,
                stall_factor: aircraft.stall_factor,
                out_of_bounds_seconds: aircraft.out_of_bounds_seconds,
                ceiling_recovery_seconds: aircraft.ceiling_recovery_seconds,
                ceiling_recovery_target_pitch_deg: state.ceiling_recovery_target_pitch_deg,
                is_destroyed: aircraft.destroyed,
            };
            if is_locally_controlled {
                let acked_tick = prediction.last_acked_tick.or(inbox.local_acked_input_tick);
                if should_snap || acked_tick.is_none() {
                    jitter_diagnostics.local_snap_count =
                        jitter_diagnostics.local_snap_count.saturating_add(1);
                    jitter_diagnostics.last_local_snap_tick = Some(snapshot.tick);
                    *state = authoritative_state.clone();
                } else if let Some(acked_tick) = acked_tick {
                    set_prediction_baseline(
                        &mut prediction,
                        role,
                        LocalPredictionBaseline {
                            authoritative_tick: acked_tick,
                            state: authoritative_state.clone(),
                        },
                    );
                    jitter_diagnostics
                        .first_baseline_tick
                        .get_or_insert(acked_tick);
                    let Some(baseline) = prediction_baseline(&prediction, role) else {
                        *state = authoritative_state.clone();
                        break;
                    };
                    let mut replayed_state = baseline.state.clone();
                    for frame in prediction
                        .sent_inputs
                        .iter()
                        .filter(|frame| frame.tick > baseline.authoritative_tick)
                    {
                        let replay_input = replay_control_input(&frame.actions, role);
                        step_predicted_aircraft_state(
                            &config,
                            &mut replayed_state,
                            performance,
                            &damage,
                            &replay_input,
                            config.game.fixed_time_step_seconds,
                        );
                    }
                    let latest_sent_tick = prediction
                        .sent_inputs
                        .back()
                        .map(|frame| frame.tick)
                        .unwrap_or(acked_tick);
                    let ack_lag_ticks = latest_sent_tick.saturating_sub(acked_tick);
                    let historical_position_error =
                        predicted_state_at_tick(&prediction, role, acked_tick)
                            .map(|predicted| {
                                predicted.position.distance(authoritative_state.position)
                            })
                            .unwrap_or_else(|| {
                                state.position.distance(authoritative_state.position)
                            });
                    let historical_velocity_error =
                        predicted_state_at_tick(&prediction, role, acked_tick)
                            .map(|predicted| {
                                predicted.velocity.distance(authoritative_state.velocity)
                            })
                            .unwrap_or_else(|| {
                                state.velocity.distance(authoritative_state.velocity)
                            });
                    let historical_orientation_dot =
                        predicted_state_at_tick(&prediction, role, acked_tick)
                            .map(|predicted| {
                                predicted
                                    .orientation
                                    .dot(authoritative_state.orientation)
                                    .abs()
                            })
                            .unwrap_or_else(|| {
                                state.orientation.dot(authoritative_state.orientation).abs()
                            });
                    *state = replayed_state;
                    let now = time.elapsed_secs_f64();
                    if now >= correction_log.next_log_at_seconds {
                        correction_log.next_log_at_seconds = now + 0.5;
                        info!(
                            "bridge local correction role={role:?} snapshot_tick={} acked_tick={} latest_sent_tick={} ack_lag={} pending_inputs={} hist_pos_err={:.3} hist_vel_err={:.3} hist_orient_dot={:.5} target_buffer_len={} pending_snapshots={}",
                            snapshot.tick,
                            acked_tick,
                            latest_sent_tick,
                            ack_lag_ticks,
                            prediction.sent_inputs.len(),
                            historical_position_error,
                            historical_velocity_error,
                            historical_orientation_dot,
                            inbox.target_buffer_len,
                            inbox.pending_snapshots.len(),
                        );
                    }
                }
            } else {
                *state = authoritative_state;
            }
            state.forward = (state.orientation * Vec3::Z).normalize_or_zero();
            if state.forward == Vec3::ZERO {
                state.forward = Vec3::from_array(aircraft.forward).normalize_or_zero();
            }
            state.throttle = aircraft.throttle;
            state.hit_points = aircraft.hit_points;
            state.stall_factor = aircraft.stall_factor;
            state.out_of_bounds_seconds = aircraft.out_of_bounds_seconds;
            state.ceiling_recovery_seconds = aircraft.ceiling_recovery_seconds;
            state.is_destroyed = aircraft.destroyed;

            gun.heat = aircraft.gun_heat;
            gun.overheated = aircraft.gun_overheated;

            damage.is_repairing = aircraft.repairing;
            damage.repair_elapsed_seconds = aircraft.repair_elapsed_seconds;
            for subsystem in &aircraft.subsystems {
                if let Some(target) = subsystem_name(subsystem.name.as_str()) {
                    let health = damage.subsystem_mut(target);
                    health.current = subsystem.hit_points;
                    health.max = subsystem.max_hit_points;
                }
            }
        }
    }

    for entity in &dynamic_entities.p0() {
        commands.entity(entity).despawn();
    }
    for entity in &dynamic_entities.p1() {
        commands.entity(entity).despawn();
    }

    let listener_role = observed_role
        .as_deref()
        .map(|role| role.0)
        .or(locally_controlled_role);
    let listener_position = listener_role.and_then(|role| {
        snapshot
            .observation
            .state
            .aircraft
            .iter()
            .find(|aircraft| aircraft_role_name(aircraft.role.as_str()) == Some(role))
            .map(|aircraft| Vec3::from_array(aircraft.position))
    });
    let mut seen_projectiles = HashSet::new();

    for projectile in &snapshot.observation.dynamic.projectiles {
        let Some(shooter_role) = aircraft_role_name(&projectile.shooter_role) else {
            continue;
        };
        if Some(shooter_role) == locally_controlled_role {
            continue;
        }
        let projectile_position = Vec3::from_array(projectile.position);
        let entry =
            projectile_audio
                .entries
                .entry(projectile.id)
                .or_insert(SnapshotProjectileAudioEntry {
                    last_position: projectile_position,
                    flyby_emitted: false,
                });
        if !entry.flyby_emitted
            && Some(shooter_role) != listener_role
            && let Some(listener_position) = listener_position
            && let Some(flyby_t) = segment_sphere_hit_fraction(
                entry.last_position,
                projectile_position,
                listener_position,
                22.0,
            )
        {
            let flyby_point = entry.last_position.lerp(projectile_position, flyby_t);
            if let Some(audio_events) = audio_events.as_deref_mut() {
                audio_events.push(AudioEventKind::BulletFlyBy, flyby_point, 0.9);
            }
            entry.flyby_emitted = true;
        }
        entry.last_position = projectile_position;
        seen_projectiles.insert(projectile.id);
        spawn_snapshot_projectile(
            &mut commands,
            meshes.as_deref_mut(),
            materials.as_deref_mut(),
            Projectile {
                id: projectile.id,
                shooter_role,
                velocity: Vec3::from_array(projectile.velocity),
                damage: projectile.damage,
                remaining_distance: projectile.remaining_distance,
                hit_radius: projectile.hit_radius,
                flyby_emitted: entry.flyby_emitted,
                lag_compensation_ticks: 0,
            },
            Transform::from_translation(projectile_position).with_rotation(
                Quat::from_rotation_arc(
                    Vec3::Z,
                    Vec3::from_array(projectile.velocity).normalize_or_zero(),
                ),
            ),
        );
    }
    projectile_audio
        .entries
        .retain(|id, _| seen_projectiles.contains(id));

    for tracer in &snapshot.observation.dynamic.tracers {
        spawn_snapshot_tracer(
            &mut commands,
            meshes.as_deref_mut(),
            materials.as_deref_mut(),
            Vec3::from_array(tracer.position),
            tracer.remaining_seconds,
        );
    }

    let observed_role = observed_role
        .as_ref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    for event in &snapshot.observation.state.events_since_last_step {
        let source_position =
            authoritative_event_source_position(event, &snapshot.observation.state.aircraft);
        apply_authoritative_event_feedback(
            event,
            observed_role,
            source_position,
            audio_events.as_deref_mut(),
            damage_indicators.as_deref_mut(),
            presentation_queue.as_deref_mut(),
        );
    }

    inbox.last_applied_snapshot_tick = Some(snapshot.tick);
}

fn log_server_bridge_metrics(
    mode: Res<BridgeMode>,
    enabled: Res<BridgeEnabled>,
    time: Res<Time<Real>>,
    sim_debug: Res<SimulationDebugState>,
    match_phase: Res<State<MatchPhase>>,
    sessions: Res<BridgeServerSessions>,
    control: Res<BridgeServerMetricsControl>,
    mut metrics: ResMut<BridgeServerMetricsState>,
) {
    if mode.0 != BridgeRole::Server || !enabled.0 || !control.enabled {
        return;
    }

    let now = time.elapsed_secs_f64();
    if now < metrics.next_log_at_seconds {
        return;
    }
    metrics.next_log_at_seconds = now + control.interval_seconds.max(0.25);

    let connected_clients = sessions.clients.len();
    let connected_players = connected_player_count(&sessions);
    let latest_client_tick = sessions
        .clients
        .values()
        .filter_map(|session| session.last_input_tick)
        .max();
    let latest_input_lag = latest_client_tick.map(|tick| sim_debug.tick_count.saturating_sub(tick));
    let slot_summary = occupied_slots(&sessions)
        .into_iter()
        .map(|slot| format!("{slot:?}"))
        .collect::<Vec<_>>()
        .join(",");

    info!(
        "bridge server metrics phase={:?} snapshot_tick={} clients={} players={}/2 latest_client_tick={:?} input_lag_ticks={:?} slots=[{}]",
        match_phase.get(),
        sim_debug.tick_count,
        connected_clients,
        connected_players,
        latest_client_tick,
        latest_input_lag,
        slot_summary,
    );
}

fn prediction_baseline(
    state: &LocalPredictionState,
    role: AircraftRole,
) -> Option<&LocalPredictionBaseline> {
    match role {
        AircraftRole::Fighter1 => state.fighter1_baseline.as_ref(),
        AircraftRole::Fighter2 => state.fighter2_baseline.as_ref(),
    }
}

const LOCAL_PREDICTION_HISTORY_LIMIT: usize = 128;

fn prediction_history_mut(
    state: &mut LocalPredictionState,
    role: AircraftRole,
) -> &mut VecDeque<LocalPredictedStateFrame> {
    match role {
        AircraftRole::Fighter1 => &mut state.fighter1_history,
        AircraftRole::Fighter2 => &mut state.fighter2_history,
    }
}

fn prediction_history(
    state: &LocalPredictionState,
    role: AircraftRole,
) -> &VecDeque<LocalPredictedStateFrame> {
    match role {
        AircraftRole::Fighter1 => &state.fighter1_history,
        AircraftRole::Fighter2 => &state.fighter2_history,
    }
}

fn push_predicted_state_history(
    state: &mut LocalPredictionState,
    role: AircraftRole,
    tick: u64,
    predicted_state: &AircraftState,
) {
    let history = prediction_history_mut(state, role);
    if history.back().is_some_and(|entry| entry.tick == tick) {
        history.pop_back();
    }
    history.push_back(LocalPredictedStateFrame {
        tick,
        state: predicted_state.clone(),
    });
    while history.len() > LOCAL_PREDICTION_HISTORY_LIMIT {
        history.pop_front();
    }
}

fn predicted_state_at_tick(
    state: &LocalPredictionState,
    role: AircraftRole,
    tick: u64,
) -> Option<&AircraftState> {
    prediction_history(state, role)
        .iter()
        .find(|entry| entry.tick == tick)
        .map(|entry| &entry.state)
}

fn set_prediction_baseline(
    state: &mut LocalPredictionState,
    role: AircraftRole,
    baseline: LocalPredictionBaseline,
) {
    match role {
        AircraftRole::Fighter1 => state.fighter1_baseline = Some(baseline),
        AircraftRole::Fighter2 => state.fighter2_baseline = Some(baseline),
    }
}

const REMOTE_INTERPOLATION_HISTORY_LIMIT: usize = 64;

fn remote_aircraft_history_mut(
    state: &mut BridgeRemoteInterpolationState,
    role: AircraftRole,
) -> &mut VecDeque<RemoteAircraftSnapshot> {
    match role {
        AircraftRole::Fighter1 => &mut state.fighter1_history,
        AircraftRole::Fighter2 => &mut state.fighter2_history,
    }
}

fn remote_aircraft_history(
    state: &BridgeRemoteInterpolationState,
    role: AircraftRole,
) -> &VecDeque<RemoteAircraftSnapshot> {
    match role {
        AircraftRole::Fighter1 => &state.fighter1_history,
        AircraftRole::Fighter2 => &state.fighter2_history,
    }
}

fn push_remote_aircraft_snapshot(
    state: &mut BridgeRemoteInterpolationState,
    role: AircraftRole,
    snapshot: RemoteAircraftSnapshot,
) {
    let history = remote_aircraft_history_mut(state, role);
    if history
        .back()
        .is_some_and(|previous| previous.sim_time_seconds >= snapshot.sim_time_seconds)
    {
        history.retain(|entry| entry.sim_time_seconds < snapshot.sim_time_seconds);
    }
    history.push_back(snapshot);
    while history.len() > REMOTE_INTERPOLATION_HISTORY_LIMIT {
        history.pop_front();
    }
}

pub fn sample_remote_aircraft_snapshot(
    state: &BridgeRemoteInterpolationState,
    role: AircraftRole,
    render_sim_time_seconds: f32,
) -> Option<(Vec3, Quat)> {
    let history = remote_aircraft_history(state, role);
    let newest = history.back()?;
    if history.len() == 1 || render_sim_time_seconds >= newest.sim_time_seconds {
        return Some((newest.position, newest.orientation));
    }

    let mut previous = history.front()?;
    for next in history.iter().skip(1) {
        if render_sim_time_seconds <= next.sim_time_seconds {
            let span = (next.sim_time_seconds - previous.sim_time_seconds).max(1.0 / 240.0);
            let alpha =
                ((render_sim_time_seconds - previous.sim_time_seconds) / span).clamp(0.0, 1.0);
            return Some((
                previous.position.lerp(next.position, alpha),
                previous
                    .orientation
                    .slerp(next.orientation, alpha)
                    .normalize(),
            ));
        }
        previous = next;
    }

    Some((newest.position, newest.orientation))
}

fn replay_control_input(actions: &[TargetedEnvironmentAction], role: AircraftRole) -> ControlInput {
    actions
        .iter()
        .find(|action| action.role == role)
        .map(|action| ControlInput::from(action.action))
        .unwrap_or_default()
}

fn segment_sphere_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<f32> {
    let direction = segment_end - segment_start;
    let a = direction.length_squared();
    if a <= f32::EPSILON {
        return (segment_start.distance(sphere_center) <= sphere_radius).then_some(0.0);
    }

    let offset = segment_start - sphere_center;
    let b = 2.0 * offset.dot(direction);
    let c = offset.length_squared() - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrt_discriminant = discriminant.sqrt();
    let near_t = (-b - sqrt_discriminant) / (2.0 * a);
    let far_t = (-b + sqrt_discriminant) / (2.0 * a);

    if (0.0..=1.0).contains(&near_t) {
        Some(near_t)
    } else if (0.0..=1.0).contains(&far_t) {
        Some(far_t)
    } else {
        None
    }
}

fn send_client_hello(
    _config: &RepositoryConfig,
    endpoint: &transport::BridgeEndpointConfig,
    requested_role: BridgeControlSlot,
    launch_binding: &BridgeLaunchBinding,
    state: &mut transport::IpcBridgeState,
) {
    transport::send_message(
        endpoint,
        BridgeRole::Client,
        state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ClientHello(ClientHello {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            requested_session: endpoint.session.clone(),
            requested_scene: None,
            requested_role,
            launcher_session_id: launch_binding.launcher_session_id.clone(),
            launch_token: launch_binding.launch_token.clone(),
            child_kind: launch_binding.child_kind,
        }),
    );
}

fn apply_assigned_control_role(
    mode: Res<BridgeMode>,
    inbox: Res<BridgeClientInbox>,
    local_pilot_mode: Res<LocalPilotMode>,
    mut assigned_role: ResMut<AssignedControlRole>,
    mut observed_role: Option<ResMut<ObservedAircraftRole>>,
    mut authority_query: Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    if mode.0 != BridgeRole::Client {
        return;
    }

    let Some(server_hello) = inbox.server_hello.as_ref() else {
        return;
    };
    if assigned_role.0 != server_hello.assigned_role {
        assigned_role.0 = server_hello.assigned_role;
        if let (Some(observed_role), Some(role)) = (
            observed_role.as_deref_mut(),
            bridge_slot_aircraft_role(server_hello.assigned_role),
        ) {
            observed_role.0 = role;
        }
    }
    apply_control_authority_assignment(
        server_hello.assigned_role,
        *local_pilot_mode,
        &mut authority_query,
    );
}

pub fn should_run_local_authority(
    app_mode: Res<AppMode>,
    mode: Res<BridgeMode>,
    bridge_enabled: Res<BridgeEnabled>,
    requested_role: Res<RequestedControlRole>,
    launch_binding: Res<BridgeLaunchBinding>,
    link_state: Option<Res<BridgeLinkState>>,
) -> bool {
    if *app_mode == AppMode::Observer {
        return false;
    }
    if mode.0 == BridgeRole::Server {
        return true;
    }
    if !bridge_enabled.0 {
        return true;
    }
    if launch_binding.launch_token.is_some()
        && !link_state
            .as_ref()
            .map(|state| state.remote_authority_active)
            .unwrap_or(false)
    {
        return false;
    }
    if requested_role.0 == BridgeControlSlot::Spectator {
        return false;
    }
    !link_state
        .as_ref()
        .map(|state| state.remote_authority_active)
        .unwrap_or(false)
}

pub fn should_run_local_gameplay_authority(
    app_mode: Res<AppMode>,
    mode: Res<BridgeMode>,
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Option<Res<transport::BridgeEndpointConfig>>,
    requested_role: Option<Res<RequestedControlRole>>,
    launch_binding: Option<Res<BridgeLaunchBinding>>,
    link_state: Option<Res<BridgeLinkState>>,
) -> bool {
    let requested_role = requested_role
        .as_ref()
        .map(|role| role.0)
        .unwrap_or(BridgeControlSlot::Fighter1);
    let has_launch_token = launch_binding
        .as_ref()
        .and_then(|binding| binding.launch_token.as_ref())
        .is_some();
    if *app_mode == AppMode::Observer {
        return false;
    }
    if mode.0 == BridgeRole::Server {
        return true;
    }
    if !bridge_enabled.0 {
        return true;
    }
    if has_launch_token
        && !link_state
            .as_ref()
            .map(|state| state.remote_authority_active)
            .unwrap_or(false)
    {
        return false;
    }
    if requested_role == BridgeControlSlot::Spectator {
        return false;
    }
    if endpoint
        .as_ref()
        .is_some_and(|endpoint| endpoint.transport == transport::BridgeTransport::Tcp)
    {
        return false;
    }
    !link_state
        .as_ref()
        .map(|state| state.remote_authority_active)
        .unwrap_or(false)
}

pub fn should_run_local_fire_visuals(
    app_mode: Res<AppMode>,
    mode: Res<BridgeMode>,
    bridge_enabled: Res<BridgeEnabled>,
    endpoint: Option<Res<transport::BridgeEndpointConfig>>,
    requested_role: Option<Res<RequestedControlRole>>,
    link_state: Option<Res<BridgeLinkState>>,
) -> bool {
    let requested_role = requested_role
        .as_ref()
        .map(|role| role.0)
        .unwrap_or(BridgeControlSlot::Fighter1);
    if *app_mode != AppMode::Game
        || mode.0 != BridgeRole::Client
        || !bridge_enabled.0
        || requested_role == BridgeControlSlot::Spectator
    {
        return false;
    }
    endpoint
        .as_ref()
        .is_some_and(|endpoint| endpoint.transport == transport::BridgeTransport::Tcp)
        && link_state
            .as_ref()
            .map(|state| state.remote_authority_active)
            .unwrap_or(false)
}

fn match_phase_name(name: &str) -> Option<MatchPhase> {
    match name {
        "Loading" => Some(MatchPhase::Loading),
        "Running" => Some(MatchPhase::Running),
        "Finished" => Some(MatchPhase::Finished),
        _ => None,
    }
}

fn aircraft_role_name(name: &str) -> Option<AircraftRole> {
    match name {
        "Fighter1" | "fighter1" | "player1" => Some(AircraftRole::Fighter1),
        "Fighter2" | "fighter2" | "player2" => Some(AircraftRole::Fighter2),
        _ => None,
    }
}

fn apply_control_authority_assignment(
    assigned_role: BridgeControlSlot,
    local_pilot_mode: LocalPilotMode,
    authority_query: &mut Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    let assigned_aircraft_role = bridge_slot_aircraft_role(assigned_role);
    for (role, mut authority) in authority_query.iter_mut() {
        *authority = match assigned_aircraft_role {
            Some(target_role) if *role == target_role => match local_pilot_mode {
                LocalPilotMode::Human => ControlAuthority::Human,
                LocalPilotMode::FollowAi
                | LocalPilotMode::ImperfectFollowAi
                | LocalPilotMode::TeacherFollowAi
                | LocalPilotMode::Model => ControlAuthority::ExternalAgent,
            },
            _ => ControlAuthority::BuiltInAi,
        };
    }
}

fn allocate_role(
    client_id: &str,
    requested_role: BridgeControlSlot,
    sessions: &mut BridgeServerSessions,
) -> BridgeControlSlot {
    if let Some(existing) = sessions.clients.get(client_id) {
        return existing.assigned_role;
    }

    let player_taken = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .any(|session| session.assigned_role == BridgeControlSlot::Fighter1);
    let enemy_taken = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .any(|session| session.assigned_role == BridgeControlSlot::Fighter2);
    let assigned_role = match requested_role {
        BridgeControlSlot::Fighter1 if !player_taken => BridgeControlSlot::Fighter1,
        BridgeControlSlot::Fighter2 if !enemy_taken => BridgeControlSlot::Fighter2,
        BridgeControlSlot::Spectator => BridgeControlSlot::Spectator,
        BridgeControlSlot::Fighter1 if !enemy_taken => BridgeControlSlot::Fighter2,
        BridgeControlSlot::Fighter2 if !player_taken => BridgeControlSlot::Fighter1,
        _ => BridgeControlSlot::Spectator,
    };

    sessions.clients.insert(
        client_id.to_string(),
        BridgeServerClientSession {
            assigned_role,
            last_input_tick: None,
            estimated_server_tick_offset: None,
            ready: true,
            client_kind: if assigned_role == BridgeControlSlot::Spectator {
                LobbyClientKind::Observer
            } else {
                LobbyClientKind::Gameplay
            },
            launched: false,
            launcher_session_id: None,
        },
    );
    assigned_role
}

fn assign_unbound_child_spectator(
    client_id: &str,
    client_kind: LobbyClientKind,
    sessions: &mut BridgeServerSessions,
) -> BridgeControlSlot {
    let normalized_kind = match client_kind {
        LobbyClientKind::Launcher => LobbyClientKind::Observer,
        other => other,
    };
    sessions.clients.insert(
        client_id.to_string(),
        BridgeServerClientSession {
            assigned_role: BridgeControlSlot::Spectator,
            last_input_tick: None,
            estimated_server_tick_offset: None,
            ready: true,
            client_kind: normalized_kind,
            launched: false,
            launcher_session_id: None,
        },
    );
    BridgeControlSlot::Spectator
}

fn bind_child_to_owner(
    client_id: &str,
    hello: &ClientHello,
    sessions: &mut BridgeServerSessions,
) -> Option<BridgeControlSlot> {
    let launcher_session_id = hello
        .launcher_session_id
        .as_deref()
        .and_then(|value| value.parse::<u64>().ok())?;
    let launch_token = hello.launch_token.as_deref()?;
    let owner = sessions.owners.get_mut(&launcher_session_id)?;
    if owner.pending_launch_token.as_deref()? != launch_token {
        return None;
    }
    if owner
        .active_child_client_id
        .as_deref()
        .is_some_and(|active| active != client_id)
    {
        return None;
    }

    let child_kind = hello.child_kind.unwrap_or(match owner.selected_role {
        BridgeControlSlot::Spectator => LobbyClientKind::Observer,
        BridgeControlSlot::Fighter1 | BridgeControlSlot::Fighter2 => LobbyClientKind::Gameplay,
    });
    let child_kind = match child_kind {
        LobbyClientKind::Launcher => return None,
        other => other,
    };
    let assigned_role = owner.selected_role;

    sessions.clients.insert(
        client_id.to_string(),
        BridgeServerClientSession {
            assigned_role,
            last_input_tick: None,
            estimated_server_tick_offset: None,
            ready: true,
            client_kind: child_kind,
            launched: false,
            launcher_session_id: Some(launcher_session_id),
        },
    );
    owner.active_child_client_id = Some(client_id.to_string());
    owner.active_child_kind = Some(child_kind);
    owner.pending_launch_token = None;
    Some(assigned_role)
}

fn set_or_allocate_role(
    client_id: &str,
    requested_role: BridgeControlSlot,
    sessions: &mut BridgeServerSessions,
    client_kind: LobbyClientKind,
) -> BridgeControlSlot {
    let assigned_role = {
        if client_kind == LobbyClientKind::Launcher {
            requested_role
        } else {
            let mut other_sessions = sessions.clients.clone();
            other_sessions.remove(client_id);
            let fighter1_taken = other_sessions
                .values()
                .filter(|session| session.client_kind != LobbyClientKind::Launcher)
                .any(|session| session.assigned_role == BridgeControlSlot::Fighter1);
            let fighter2_taken = other_sessions
                .values()
                .filter(|session| session.client_kind != LobbyClientKind::Launcher)
                .any(|session| session.assigned_role == BridgeControlSlot::Fighter2);
            match requested_role {
                BridgeControlSlot::Fighter1 if !fighter1_taken => BridgeControlSlot::Fighter1,
                BridgeControlSlot::Fighter2 if !fighter2_taken => BridgeControlSlot::Fighter2,
                BridgeControlSlot::Spectator => BridgeControlSlot::Spectator,
                BridgeControlSlot::Fighter1 if !fighter2_taken => BridgeControlSlot::Fighter2,
                BridgeControlSlot::Fighter2 if !fighter1_taken => BridgeControlSlot::Fighter1,
                _ => BridgeControlSlot::Spectator,
            }
        }
    };

    sessions
        .clients
        .entry(client_id.to_string())
        .and_modify(|session| {
            session.assigned_role = assigned_role;
            session.client_kind = client_kind;
            session.ready = false;
            session.launched = false;
            if client_kind != LobbyClientKind::Launcher {
                session.launcher_session_id = None;
            }
        })
        .or_insert(BridgeServerClientSession {
            assigned_role,
            last_input_tick: None,
            estimated_server_tick_offset: None,
            ready: false,
            client_kind,
            launched: false,
            launcher_session_id: None,
        });
    if client_kind == LobbyClientKind::Launcher {
        let launcher_session_id = sessions
            .clients
            .get(client_id)
            .and_then(|session| session.launcher_session_id);
        if let Some(launcher_session_id) = launcher_session_id
            && let Some(owner) = sessions.owners.get_mut(&launcher_session_id)
        {
            owner.selected_role = assigned_role;
            owner.ready = false;
            owner.launched = false;
            owner.pending_launch_token = None;
        }
    }
    assigned_role
}

fn current_lobby_phase(match_phase: MatchPhase) -> LobbySessionPhase {
    match match_phase {
        MatchPhase::Loading => LobbySessionPhase::Lobby,
        MatchPhase::Running => LobbySessionPhase::Running,
        MatchPhase::Finished => LobbySessionPhase::Ending,
    }
}

fn dispatch_lobby_start_messages(
    endpoint: &transport::BridgeEndpointConfig,
    bridge_state: &mut transport::IpcBridgeState,
    sessions: &mut BridgeServerSessions,
    scene_name: &str,
    match_phase: MatchPhase,
) {
    if current_lobby_phase(match_phase) != LobbySessionPhase::Running {
        return;
    }
    let owner_ids = sessions.owners.keys().copied().collect::<Vec<_>>();
    for launcher_session_id in owner_ids {
        let Some(owner) = sessions.owners.get_mut(&launcher_session_id) else {
            continue;
        };
        if !owner.ready || owner.launched {
            continue;
        }
        let child_kind = match owner.selected_role {
            BridgeControlSlot::Spectator => LobbyClientKind::Observer,
            BridgeControlSlot::Fighter1 | BridgeControlSlot::Fighter2 => LobbyClientKind::Gameplay,
        };
        sessions.next_launch_token_id += 1;
        let launch_token = format!("launch-{}", sessions.next_launch_token_id);
        transport::send_message(
            endpoint,
            BridgeRole::Server,
            bridge_state,
            BridgeDispatchTarget::Client(owner.launcher_client_id.clone()),
            BridgeMessage::ServerLobbyStart(ServerLobbyStart {
                accepted_session: endpoint.session.clone(),
                accepted_scene: scene_name.to_string(),
                assigned_role: owner.selected_role,
                launcher_session_id: Some(owner.launcher_session_id.to_string()),
                launch_token: Some(launch_token.clone()),
                child_kind: Some(child_kind),
            }),
        );
        owner.launched = true;
        owner.pending_launch_token = Some(launch_token);
        if let Some(session) = sessions.clients.get_mut(&owner.launcher_client_id) {
            session.launched = true;
        }
    }
}

fn broadcast_lobby_state(
    endpoint: &transport::BridgeEndpointConfig,
    bridge_state: &mut transport::IpcBridgeState,
    sessions: &BridgeServerSessions,
    scene_name: &str,
    match_phase: MatchPhase,
    focus_client_id: Option<&str>,
) {
    let phase = current_lobby_phase(match_phase);
    let slot_entries = sessions
        .clients
        .iter()
        .map(|(client_id, session)| LobbySlotState {
            client_id: client_id.clone(),
            assigned_role: session.assigned_role,
            ready: session.ready,
            client_kind: session.client_kind,
        })
        .collect::<Vec<_>>();
    for client_id in sessions.clients.keys() {
        let assigned_role = sessions.clients.get(client_id).and_then(|session| {
            if session.client_kind == LobbyClientKind::Launcher {
                session
                    .launcher_session_id
                    .and_then(|id| sessions.owners.get(&id).map(|owner| owner.selected_role))
            } else {
                Some(session.assigned_role)
            }
        });
        let ready = sessions.clients.get(client_id).is_some_and(|session| {
            if session.client_kind == LobbyClientKind::Launcher {
                session
                    .launcher_session_id
                    .and_then(|id| sessions.owners.get(&id).map(|owner| owner.ready))
                    .unwrap_or(false)
            } else {
                session.ready
            }
        });
        let launcher_session_id = sessions
            .clients
            .get(client_id)
            .and_then(|session| session.launcher_session_id.map(|id| id.to_string()));
        let message = BridgeMessage::ServerLobbyState(ServerLobbyState {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            accepted_session: endpoint.session.clone(),
            accepted_scene: scene_name.to_string(),
            phase,
            slots: slot_entries.clone(),
            assigned_role,
            ready,
            launcher_session_id,
        });
        transport::send_message(
            endpoint,
            BridgeRole::Server,
            bridge_state,
            BridgeDispatchTarget::Client(client_id.clone()),
            message,
        );
    }
    if let Some(client_id) = focus_client_id
        && !sessions.clients.contains_key(client_id)
    {}
}

pub(crate) fn apply_server_authorities(
    sessions: &BridgeServerSessions,
    authority_query: &mut Query<(&AircraftRole, &mut ControlAuthority)>,
) {
    let player_claimed = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .any(|session| session.assigned_role == BridgeControlSlot::Fighter1);
    let enemy_claimed = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .any(|session| session.assigned_role == BridgeControlSlot::Fighter2);

    for (role, mut authority) in authority_query.iter_mut() {
        *authority = match *role {
            AircraftRole::Fighter1 if player_claimed => ControlAuthority::ExternalAgent,
            AircraftRole::Fighter2 if enemy_claimed => ControlAuthority::ExternalAgent,
            AircraftRole::Fighter2 => ControlAuthority::BuiltInAi,
            AircraftRole::Fighter1 => ControlAuthority::BuiltInAi,
        };
    }
}

pub fn bridge_slot_aircraft_role(slot: BridgeControlSlot) -> Option<AircraftRole> {
    match slot {
        BridgeControlSlot::Fighter1 => Some(AircraftRole::Fighter1),
        BridgeControlSlot::Fighter2 => Some(AircraftRole::Fighter2),
        BridgeControlSlot::Spectator => None,
    }
}

fn occupied_slots(sessions: &BridgeServerSessions) -> Vec<BridgeControlSlot> {
    let mut slots = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .map(|session| session.assigned_role)
        .collect::<Vec<_>>();
    slots.sort_by_key(|slot| match slot {
        BridgeControlSlot::Fighter1 => 0,
        BridgeControlSlot::Fighter2 => 1,
        BridgeControlSlot::Spectator => 2,
    });
    slots.dedup();
    slots
}

fn processed_input_ticks(sessions: &BridgeServerSessions) -> Vec<ProcessedInputTick> {
    let mut ticks = sessions
        .clients
        .values()
        .filter(|session| session.client_kind != LobbyClientKind::Launcher)
        .filter_map(|session| {
            session.last_input_tick.map(|tick| ProcessedInputTick {
                slot: session.assigned_role,
                tick,
            })
        })
        .collect::<Vec<_>>();
    ticks.sort_by_key(|entry| match entry.slot {
        BridgeControlSlot::Fighter1 => 0,
        BridgeControlSlot::Fighter2 => 1,
        BridgeControlSlot::Spectator => 2,
    });
    ticks
}

fn spawn_snapshot_projectile(
    commands: &mut Commands,
    meshes: Option<&mut Assets<Mesh>>,
    materials: Option<&mut Assets<StandardMaterial>>,
    projectile: Projectile,
    transform: Transform,
) {
    spawn_projectile_visual_entity(commands, meshes, materials, projectile, transform);
}

fn spawn_snapshot_tracer(
    commands: &mut Commands,
    meshes: Option<&mut Assets<Mesh>>,
    materials: Option<&mut Assets<StandardMaterial>>,
    position: Vec3,
    remaining_seconds: f32,
) {
    spawn_tracer_visual_entity(commands, meshes, materials, position, remaining_seconds);
}

fn apply_authoritative_event_feedback(
    event: &EnvironmentEvent,
    observed_role: AircraftRole,
    source_position: Option<Vec3>,
    audio_events: Option<&mut AudioEventQueue>,
    damage_indicators: Option<&mut DamageIndicatorQueue>,
    presentation_queue: Option<&mut CombatPresentationQueue>,
) {
    let event_role = event
        .subject
        .as_deref()
        .and_then(subject_role_name)
        .unwrap_or(observed_role);
    let position = event.position.map(Vec3::from_array);
    let magnitude = event.magnitude.unwrap_or(1.0).abs();

    match event.kind.as_str() {
        "GunFired" => {
            if let (Some(audio_events), Some(position)) = (audio_events, position) {
                audio_events.push(AudioEventKind::GunFire, position, magnitude.max(0.35));
            }
        }
        "Hit" | "Damage" | "Collision" | "SubsystemHit" | "SubsystemDestroyed" => {
            if let (Some(audio_events), Some(position)) = (audio_events, position) {
                audio_events.push(AudioEventKind::Hit, position, magnitude.max(0.35));
            }
            if event_role == observed_role
                && let (Some(damage_indicators), Some(indicator_position)) =
                    (damage_indicators, source_position.or(position))
            {
                damage_indicators.push(indicator_position, magnitude.max(0.35));
            }
        }
        "Destroy" => {
            if let (Some(audio_events), Some(position)) = (audio_events, position) {
                audio_events.push(AudioEventKind::Hit, position, 1.0);
            }
            if let (Some(presentation_queue), Some(position)) = (presentation_queue, position) {
                presentation_queue.destroyed.push(AircraftDestroyedEvent {
                    role: event_role,
                    position,
                });
            }
        }
        _ => {}
    }
}

fn authoritative_event_source_position(
    event: &EnvironmentEvent,
    aircraft: &[AircraftObservation],
) -> Option<Vec3> {
    let source_role = event.other_subject.as_deref().and_then(subject_role_name)?;
    aircraft
        .iter()
        .find(|aircraft| aircraft_role_name(&aircraft.role) == Some(source_role))
        .map(|aircraft| Vec3::from_array(aircraft.position))
}

fn subject_role_name(subject: &str) -> Option<AircraftRole> {
    let role_name = subject.split(':').next()?;
    aircraft_role_name(role_name)
}

fn subsystem_name(name: &str) -> Option<AircraftSubsystem> {
    match name {
        "LeftWing" => Some(AircraftSubsystem::LeftWing),
        "RightWing" => Some(AircraftSubsystem::RightWing),
        "PitchTail" => Some(AircraftSubsystem::PitchTail),
        "YawTail" => Some(AircraftSubsystem::YawTail),
        "Engine" => Some(AircraftSubsystem::Engine),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_authoritative_event_feedback, authoritative_event_source_position};
    use crate::{
        api::types::{AircraftObservation, EnvironmentEvent, EnvironmentEventKind},
        presentation::hud::DamageIndicatorQueue,
        simulation::components::AircraftRole,
    };
    use bevy::prelude::Vec3;

    fn gun_damage_event() -> EnvironmentEvent {
        EnvironmentEvent::with_context(
            42,
            EnvironmentEventKind::Damage,
            Some("fighter1".to_string()),
            Some("fighter2".to_string()),
            Some([1.0, 2.0, 3.0]),
            Some(8.0),
            Some("gun".to_string()),
            None,
        )
    }

    #[test]
    fn authoritative_feedback_uses_attacker_position_for_damage_direction() {
        let event = gun_damage_event();
        let aircraft = [AircraftObservation {
            role: "fighter2".to_string(),
            position: [120.0, 80.0, -40.0],
            ..Default::default()
        }];
        let source_position = authoritative_event_source_position(&event, &aircraft);
        let mut indicators = DamageIndicatorQueue::default();

        apply_authoritative_event_feedback(
            &event,
            AircraftRole::Fighter1,
            source_position,
            None,
            Some(&mut indicators),
            None,
        );

        assert_eq!(indicators.events.len(), 1);
        assert_eq!(
            indicators.events[0].source_position,
            Vec3::new(120.0, 80.0, -40.0)
        );
    }

    #[test]
    fn authoritative_feedback_falls_back_to_contact_position_without_source() {
        let event = gun_damage_event();
        let mut indicators = DamageIndicatorQueue::default();

        apply_authoritative_event_feedback(
            &event,
            AircraftRole::Fighter1,
            None,
            None,
            Some(&mut indicators),
            None,
        );

        assert_eq!(indicators.events.len(), 1);
        assert_eq!(
            indicators.events[0].source_position,
            Vec3::new(1.0, 2.0, 3.0)
        );
    }
}
