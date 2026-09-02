use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::Duration;

use bevy::prelude::{Resource, error, info};

use crate::bridge::protocol::{BridgeMessage, BridgeRole, ClientInputFrame, ServerSnapshotFrame};

static IN_PROCESS_BRIDGES: LazyLock<Mutex<HashMap<String, Arc<Mutex<SharedBridge>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

const LOCAL_CLIENT_ID: &str = "local-client";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeTransport {
    InProcess,
    Tcp,
}

#[derive(Debug, Clone, Resource)]
pub struct BridgeEndpointConfig {
    pub session: String,
    pub transport: BridgeTransport,
    pub address: String,
}

impl Default for BridgeEndpointConfig {
    fn default() -> Self {
        Self {
            session: "default".to_string(),
            transport: BridgeTransport::InProcess,
            address: "127.0.0.1:50051".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BridgeInboundMessage {
    pub client_id: Option<String>,
    pub message: BridgeMessage,
}

#[derive(Debug, Clone)]
pub enum BridgeDispatchTarget {
    Broadcast,
    Client(String),
}

#[derive(Resource)]
pub struct IpcBridgeState {
    pub session: String,
    pub transport: BridgeTransport,
    pub address: String,
    pub connected_clients: usize,
    pub server_connected: bool,
    pub client_connected: bool,
    pub last_client_tick: Option<u64>,
    pub last_server_tick: Option<u64>,
    pub local_client_id: Option<String>,
    pub connected_client_ids: Vec<String>,
    tcp: TcpBridgeRuntime,
}

impl Default for IpcBridgeState {
    fn default() -> Self {
        Self {
            session: String::new(),
            transport: BridgeTransport::InProcess,
            address: "127.0.0.1:50051".to_string(),
            connected_clients: 0,
            server_connected: false,
            client_connected: false,
            last_client_tick: None,
            last_server_tick: None,
            local_client_id: None,
            connected_client_ids: Vec::new(),
            tcp: TcpBridgeRuntime::default(),
        }
    }
}

#[derive(Default)]
struct TcpBridgeRuntime {
    shared: Arc<Mutex<TcpSharedBridge>>,
    server_thread_started: bool,
    client_thread_started: bool,
}

#[derive(Default)]
struct TcpSharedBridge {
    client_inbox: VecDeque<BridgeInboundMessage>,
    server_inbox: VecDeque<BridgeInboundMessage>,
    client_sender: Option<Sender<BridgeMessage>>,
    server_client_senders: HashMap<String, Sender<BridgeMessage>>,
    next_client_id: u64,
}

#[derive(Debug, Default)]
struct SharedBridge {
    connected_clients: usize,
    server_connected: bool,
    client_to_server: VecDeque<BridgeInboundMessage>,
    server_to_client: VecDeque<BridgeInboundMessage>,
}

fn get_or_create_bridge(session: &str) -> Arc<Mutex<SharedBridge>> {
    let mut registry = IN_PROCESS_BRIDGES.lock().expect("bridge registry poisoned");
    registry
        .entry(session.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(SharedBridge::default())))
        .clone()
}

pub fn connect_endpoint(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) {
    state.session = endpoint.session.clone();
    state.transport = endpoint.transport;
    state.address = endpoint.address.clone();

    match endpoint.transport {
        BridgeTransport::InProcess => connect_local_endpoint(endpoint, role, state),
        BridgeTransport::Tcp => connect_tcp_endpoint(endpoint, role, state),
    }
}

pub fn send_message(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
    target: BridgeDispatchTarget,
    message: BridgeMessage,
) {
    match endpoint.transport {
        BridgeTransport::InProcess => send_local_message(
            &endpoint.session,
            role,
            state.local_client_id.clone(),
            target,
            message,
        ),
        BridgeTransport::Tcp => send_tcp_message(role, state, target, message),
    }
}

pub fn drain_messages(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) -> Vec<BridgeInboundMessage> {
    match endpoint.transport {
        BridgeTransport::InProcess => drain_local_messages(&endpoint.session, role, state),
        BridgeTransport::Tcp => drain_tcp_messages(role, state),
    }
}

pub fn force_disconnect_client(state: &mut IpcBridgeState, client_id: &str) {
    match state.transport {
        BridgeTransport::InProcess => {
            if client_id_matches_local(client_id) {
                state.connected_clients = 0;
                state.connected_client_ids.clear();
            }
        }
        BridgeTransport::Tcp => {
            if let Ok(mut shared) = state.tcp.shared.lock() {
                shared.server_client_senders.remove(client_id);
                state.connected_client_ids = shared.server_client_senders.keys().cloned().collect();
                state.connected_clients = state.connected_client_ids.len();
            }
        }
    }
}

fn connect_local_endpoint(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) {
    let bridge = get_or_create_bridge(&endpoint.session);
    let mut bridge = bridge.lock().expect("bridge state poisoned");

    match role {
        BridgeRole::Server => {
            bridge.server_connected = true;
        }
        BridgeRole::Client => {
            bridge.connected_clients += 1;
            state.local_client_id = Some(LOCAL_CLIENT_ID.to_string());
        }
    }

    state.connected_clients = bridge.connected_clients;
    state.server_connected = bridge.server_connected;
    state.client_connected = matches!(role, BridgeRole::Client);
    state.connected_client_ids =
        if matches!(role, BridgeRole::Server) && bridge.connected_clients > 0 {
            vec![LOCAL_CLIENT_ID.to_string()]
        } else {
            Vec::new()
        };
}

fn connect_tcp_endpoint(
    endpoint: &BridgeEndpointConfig,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) {
    match role {
        BridgeRole::Server if !state.tcp.server_thread_started => {
            state.server_connected = true;
            state.tcp.server_thread_started = true;
            let address = endpoint.address.clone();
            let shared = Arc::clone(&state.tcp.shared);
            thread::spawn(move || run_server_listener(address, shared));
        }
        BridgeRole::Client if !state.tcp.client_thread_started => {
            state.client_connected = true;
            state.tcp.client_thread_started = true;
            let address = endpoint.address.clone();
            let shared = Arc::clone(&state.tcp.shared);
            thread::spawn(move || run_client_connector(address, shared));
        }
        _ => {}
    }
}

fn send_local_message(
    session: &str,
    role: BridgeRole,
    client_id: Option<String>,
    target: BridgeDispatchTarget,
    message: BridgeMessage,
) {
    let bridge = get_or_create_bridge(session);
    let mut bridge = bridge.lock().expect("bridge state poisoned");

    match role {
        BridgeRole::Client => bridge
            .client_to_server
            .push_back(BridgeInboundMessage { client_id, message }),
        BridgeRole::Server => match target {
            BridgeDispatchTarget::Broadcast => {
                bridge.server_to_client.push_back(BridgeInboundMessage {
                    client_id: None,
                    message,
                });
            }
            BridgeDispatchTarget::Client(target_id) => {
                if client_id_matches_local(&target_id) {
                    bridge.server_to_client.push_back(BridgeInboundMessage {
                        client_id: None,
                        message,
                    });
                }
            }
        },
    }
}

fn send_tcp_message(
    role: BridgeRole,
    state: &mut IpcBridgeState,
    target: BridgeDispatchTarget,
    message: BridgeMessage,
) {
    let shared = state.tcp.shared.lock().expect("tcp bridge poisoned");
    match role {
        BridgeRole::Client => {
            if let Some(sender) = shared.client_sender.clone() {
                let _ = sender.send(message);
            }
        }
        BridgeRole::Server => match target {
            BridgeDispatchTarget::Broadcast => {
                for sender in shared.server_client_senders.values() {
                    let _ = sender.send(message.clone());
                }
            }
            BridgeDispatchTarget::Client(client_id) => {
                if let Some(sender) = shared.server_client_senders.get(&client_id) {
                    let _ = sender.send(message);
                }
            }
        },
    }
}

fn drain_local_messages(
    session: &str,
    role: BridgeRole,
    state: &mut IpcBridgeState,
) -> Vec<BridgeInboundMessage> {
    let bridge = get_or_create_bridge(session);
    let mut bridge = bridge.lock().expect("bridge state poisoned");

    state.connected_clients = bridge.connected_clients;
    state.server_connected = bridge.server_connected;
    state.client_connected = matches!(role, BridgeRole::Client);

    let queue = match role {
        BridgeRole::Server => &mut bridge.client_to_server,
        BridgeRole::Client => &mut bridge.server_to_client,
    };

    let drained: Vec<_> = queue.drain(..).collect();
    update_last_ticks(state, &drained);
    drained
}

fn drain_tcp_messages(role: BridgeRole, state: &mut IpcBridgeState) -> Vec<BridgeInboundMessage> {
    let mut shared = state.tcp.shared.lock().expect("tcp bridge poisoned");
    let drained: Vec<_> = match role {
        BridgeRole::Server => shared.server_inbox.drain(..).collect(),
        BridgeRole::Client => shared.client_inbox.drain(..).collect(),
    };

    state.connected_clients = shared.server_client_senders.len();
    state.connected_client_ids = shared.server_client_senders.keys().cloned().collect();
    state.server_connected = matches!(role, BridgeRole::Server) || shared.client_sender.is_some();
    state.client_connected =
        matches!(role, BridgeRole::Client) || !shared.server_client_senders.is_empty();
    drop(shared);

    update_last_ticks(state, &drained);
    drained
}

fn update_last_ticks(state: &mut IpcBridgeState, messages: &[BridgeInboundMessage]) {
    for inbound in messages {
        match &inbound.message {
            BridgeMessage::ClientInputFrame(ClientInputFrame { tick, .. }) => {
                state.last_client_tick = Some(*tick)
            }
            BridgeMessage::ServerSnapshotFrame(ServerSnapshotFrame { tick, .. }) => {
                state.last_server_tick = Some(*tick)
            }
            _ => {}
        }
    }
}

fn run_server_listener(address: String, shared: Arc<Mutex<TcpSharedBridge>>) {
    let listener = match TcpListener::bind(&address) {
        Ok(listener) => listener,
        Err(error) => {
            error!("failed to bind bridge server on {address}: {error}");
            return;
        }
    };

    info!("bridge tcp server listening on {address}");
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let client_id = {
                    let mut shared = shared.lock().expect("tcp bridge poisoned");
                    shared.next_client_id += 1;
                    format!("tcp-client-{}", shared.next_client_id)
                };
                info!("bridge tcp server accepted client {client_id} on {address}");
                let shared_clone = Arc::clone(&shared);
                thread::spawn(move || run_server_connection(stream, shared_clone, client_id));
            }
            Err(error) => {
                error!("bridge tcp accept failed on {address}: {error}");
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

fn run_client_connector(address: String, shared: Arc<Mutex<TcpSharedBridge>>) {
    loop {
        match TcpStream::connect(&address) {
            Ok(stream) => {
                info!("bridge tcp client connected to {address}");
                run_client_connection(stream, Arc::clone(&shared));
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

fn run_server_connection(
    stream: TcpStream,
    shared: Arc<Mutex<TcpSharedBridge>>,
    client_id: String,
) {
    run_tcp_connection(stream, shared, Some(client_id));
}

fn run_client_connection(stream: TcpStream, shared: Arc<Mutex<TcpSharedBridge>>) {
    run_tcp_connection(stream, shared, None);
}

fn run_tcp_connection(
    stream: TcpStream,
    shared: Arc<Mutex<TcpSharedBridge>>,
    server_side_client_id: Option<String>,
) {
    let writer_stream = match stream.try_clone() {
        Ok(clone) => clone,
        Err(error) => {
            error!("failed to clone tcp bridge stream: {error}");
            return;
        }
    };
    let (tx, rx) = mpsc::channel::<BridgeMessage>();

    {
        let mut shared = shared.lock().expect("tcp bridge poisoned");
        match server_side_client_id.as_ref() {
            Some(client_id) => {
                shared.server_client_senders.insert(client_id.clone(), tx);
            }
            None => {
                shared.client_sender = Some(tx);
            }
        }
    }

    let writer_handle = thread::spawn(move || {
        let mut writer = writer_stream;
        while let Ok(message) = rx.recv() {
            if serde_json::to_writer(&mut writer, &message).is_err() {
                break;
            }
            if writer.write_all(b"\n").is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                match serde_json::from_str::<BridgeMessage>(trimmed) {
                    Ok(message) => {
                        let mut shared = shared.lock().expect("tcp bridge poisoned");
                        match server_side_client_id.as_ref() {
                            Some(client_id) => {
                                shared.server_inbox.push_back(BridgeInboundMessage {
                                    client_id: Some(client_id.clone()),
                                    message,
                                });
                            }
                            None => {
                                shared.client_inbox.push_back(BridgeInboundMessage {
                                    client_id: None,
                                    message,
                                });
                            }
                        }
                    }
                    Err(error) => error!("failed to decode bridge message: {error}"),
                }
            }
            Err(error) => {
                error!("bridge tcp read failed: {error}");
                break;
            }
        }
    }

    {
        let mut shared = shared.lock().expect("tcp bridge poisoned");
        match server_side_client_id {
            Some(client_id) => {
                shared.server_client_senders.remove(&client_id);
            }
            None => {
                shared.client_sender = None;
            }
        }
    }

    let _ = writer_handle.join();
}

fn client_id_matches_local(client_id: &str) -> bool {
    client_id == LOCAL_CLIENT_ID
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::commands::TargetedEnvironmentAction;
    use crate::api::types::ObservationBundle;
    use crate::simulation::components::AircraftRole;

    #[test]
    fn local_bridge_round_trips_messages_between_roles() {
        let session = format!("test_session_{}", std::process::id());
        let endpoint = BridgeEndpointConfig {
            session: session.clone(),
            ..Default::default()
        };
        let mut server_state = IpcBridgeState::default();
        let mut client_state = IpcBridgeState::default();

        connect_endpoint(&endpoint, BridgeRole::Server, &mut server_state);
        connect_endpoint(&endpoint, BridgeRole::Client, &mut client_state);

        send_message(
            &endpoint,
            BridgeRole::Client,
            &mut client_state,
            BridgeDispatchTarget::Broadcast,
            BridgeMessage::ClientInputFrame(ClientInputFrame {
                tick: 7,
                actions: vec![TargetedEnvironmentAction {
                    role: AircraftRole::Fighter1,
                    action: Default::default(),
                }],
            }),
        );
        let server_messages = drain_messages(&endpoint, BridgeRole::Server, &mut server_state);
        assert_eq!(server_messages.len(), 1);
        assert_eq!(
            server_messages[0].client_id.as_deref(),
            Some(LOCAL_CLIENT_ID)
        );
        assert_eq!(server_state.last_client_tick, Some(7));

        send_message(
            &endpoint,
            BridgeRole::Server,
            &mut server_state,
            BridgeDispatchTarget::Client(LOCAL_CLIENT_ID.to_string()),
            BridgeMessage::ServerSnapshotFrame(ServerSnapshotFrame {
                tick: 9,
                session: endpoint.session.clone(),
                connected_clients: 1,
                occupied_slots: vec![crate::bridge::protocol::BridgeControlSlot::Fighter1],
                processed_input_ticks: Vec::new(),
                observation: ObservationBundle::default(),
            }),
        );
        let client_messages = drain_messages(&endpoint, BridgeRole::Client, &mut client_state);
        assert_eq!(client_messages.len(), 1);
        assert_eq!(client_state.last_server_tick, Some(9));
    }
}
