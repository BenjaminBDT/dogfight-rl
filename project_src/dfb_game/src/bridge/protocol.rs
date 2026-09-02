use serde::{Deserialize, Serialize};

use crate::api::commands::TargetedEnvironmentAction;
use crate::api::types::ObservationBundle;

pub const BRIDGE_PROTOCOL_VERSION: u32 = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BridgeRole {
    Server,
    Client,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BridgeControlSlot {
    Fighter1,
    Fighter2,
    Spectator,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LobbySessionPhase {
    Lobby,
    Starting,
    Running,
    Ending,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LobbyClientKind {
    Launcher,
    Gameplay,
    Observer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHello {
    pub protocol_version: u32,
    pub requested_session: String,
    pub requested_scene: Option<String>,
    pub requested_role: BridgeControlSlot,
    pub launcher_session_id: Option<String>,
    pub launch_token: Option<String>,
    pub child_kind: Option<LobbyClientKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHello {
    pub protocol_version: u32,
    pub accepted_session: String,
    pub accepted_scene: String,
    pub assigned_role: BridgeControlSlot,
    pub fixed_time_step_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientLobbyHello {
    pub protocol_version: u32,
    pub requested_session: String,
    pub requested_scene: Option<String>,
    pub client_kind: LobbyClientKind,
    pub launcher_session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientLobbySelectRole {
    pub requested_role: BridgeControlSlot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientLobbyReady {
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LobbySlotState {
    pub client_id: String,
    pub assigned_role: BridgeControlSlot,
    pub ready: bool,
    pub client_kind: LobbyClientKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLobbyState {
    pub protocol_version: u32,
    pub accepted_session: String,
    pub accepted_scene: String,
    pub phase: LobbySessionPhase,
    pub slots: Vec<LobbySlotState>,
    pub assigned_role: Option<BridgeControlSlot>,
    pub ready: bool,
    pub launcher_session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLobbyStart {
    pub accepted_session: String,
    pub accepted_scene: String,
    pub assigned_role: BridgeControlSlot,
    pub launcher_session_id: Option<String>,
    pub launch_token: Option<String>,
    pub child_kind: Option<LobbyClientKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLobbyClose {
    pub accepted_session: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInputFrame {
    pub tick: u64,
    pub actions: Vec<TargetedEnvironmentAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedInputTick {
    pub slot: BridgeControlSlot,
    pub tick: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePing {
    pub sent_at_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePong {
    pub sent_at_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSnapshotFrame {
    pub tick: u64,
    pub session: String,
    pub connected_clients: usize,
    pub occupied_slots: Vec<BridgeControlSlot>,
    pub processed_input_ticks: Vec<ProcessedInputTick>,
    pub observation: ObservationBundle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BridgeMessage {
    ClientHello(ClientHello),
    ServerHello(ServerHello),
    ClientLobbyHello(ClientLobbyHello),
    ClientLobbySelectRole(ClientLobbySelectRole),
    ClientLobbyReady(ClientLobbyReady),
    ServerLobbyState(ServerLobbyState),
    ServerLobbyStart(ServerLobbyStart),
    ServerLobbyClose(ServerLobbyClose),
    ClientInputFrame(ClientInputFrame),
    Ping(BridgePing),
    Pong(BridgePong),
    ServerSnapshotFrame(ServerSnapshotFrame),
}
