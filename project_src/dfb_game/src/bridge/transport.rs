use crate::bridge::ipc_stub;
use crate::bridge::protocol::{BridgeMessage, BridgeRole};

pub use crate::bridge::ipc_stub::{
    BridgeDispatchTarget, BridgeEndpointConfig, BridgeInboundMessage, BridgeTransport,
    IpcBridgeState,
};

pub trait TransportBackend {
    fn connect_endpoint(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
    );

    fn send_message(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
        target: BridgeDispatchTarget,
        message: BridgeMessage,
    );

    fn drain_messages(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
    ) -> Vec<BridgeInboundMessage>;

    fn force_disconnect_client(&self, state: &mut IpcBridgeState, client_id: &str);

    fn snapshot(&self, state: &IpcBridgeState) -> TransportRuntimeSnapshot;

    fn client_should_reconnect(&self, state: &IpcBridgeState) -> bool {
        !self.snapshot(state).server_connected
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IpcTransportBackend;

impl TransportBackend for IpcTransportBackend {
    fn connect_endpoint(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
    ) {
        ipc_stub::connect_endpoint(endpoint, role, state);
    }

    fn send_message(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
        target: BridgeDispatchTarget,
        message: BridgeMessage,
    ) {
        ipc_stub::send_message(endpoint, role, state, target, message);
    }

    fn drain_messages(
        &self,
        endpoint: &BridgeEndpointConfig,
        role: BridgeRole,
        state: &mut IpcBridgeState,
    ) -> Vec<BridgeInboundMessage> {
        ipc_stub::drain_messages(endpoint, role, state)
    }

    fn force_disconnect_client(&self, state: &mut IpcBridgeState, client_id: &str) {
        ipc_stub::force_disconnect_client(state, client_id);
    }

    fn snapshot(&self, state: &IpcBridgeState) -> TransportRuntimeSnapshot {
        TransportRuntimeSnapshot {
            connected_clients: state.connected_clients,
            server_connected: state.server_connected,
            client_connected: state.client_connected,
            last_client_tick: state.last_client_tick,
            last_server_tick: state.last_server_tick,
            connected_client_ids: state.connected_client_ids.clone(),
        }
    }
}

const IPC_BACKEND: IpcTransportBackend = IpcTransportBackend;

#[derive(Debug, Clone)]
pub struct TransportRuntimeSnapshot {
    pub connected_clients: usize,
    pub server_connected: bool,
    pub client_connected: bool,
    pub last_client_tick: Option<u64>,
    pub last_server_tick: Option<u64>,
    pub connected_client_ids: Vec<String>,
}

pub fn connect_endpoint(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) {
    IPC_BACKEND.connect_endpoint(endpoint, role, state);
}

pub fn send_message(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
    target: BridgeDispatchTarget,
    message: BridgeMessage,
) {
    IPC_BACKEND.send_message(endpoint, role, state, target, message);
}

pub fn drain_messages(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) -> Vec<BridgeInboundMessage> {
    IPC_BACKEND.drain_messages(endpoint, role, state)
}

pub fn force_disconnect_client(state: &mut IpcBridgeState, client_id: &str) {
    IPC_BACKEND.force_disconnect_client(state, client_id);
}

pub fn snapshot(state: &IpcBridgeState) -> TransportRuntimeSnapshot {
    IPC_BACKEND.snapshot(state)
}

pub fn client_should_reconnect(state: &IpcBridgeState) -> bool {
    IPC_BACKEND.client_should_reconnect(state)
}
