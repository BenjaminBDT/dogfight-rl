use dfb_game::api::types::ObservationBundle;
use dfb_game::bridge::protocol::{
    BRIDGE_PROTOCOL_VERSION, BridgeControlSlot, BridgeMessage, BridgeRole, ClientHello,
    ServerHello, ServerSnapshotFrame,
};
use dfb_game::bridge::transport::{
    self, BridgeDispatchTarget, BridgeEndpointConfig, BridgeTransport, IpcBridgeState,
};
use std::net::TcpListener;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn test_endpoint(session: &str) -> BridgeEndpointConfig {
    BridgeEndpointConfig {
        session: session.to_string(),
        transport: BridgeTransport::InProcess,
        address: "127.0.0.1:0".to_string(),
    }
}

#[test]
fn in_process_bridge_supports_basic_handshake_and_snapshot_flow() {
    let session = format!(
        "bridge-smoke-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let endpoint = test_endpoint(&session);

    let mut server_state = IpcBridgeState::default();
    let mut client_state = IpcBridgeState::default();

    transport::connect_endpoint(&endpoint, BridgeRole::Server, &mut server_state);
    transport::connect_endpoint(&endpoint, BridgeRole::Client, &mut client_state);

    let server_runtime = transport::snapshot(&server_state);
    let client_runtime = transport::snapshot(&client_state);
    assert!(server_runtime.server_connected);
    assert!(client_runtime.client_connected);
    assert_eq!(server_runtime.connected_clients, 0);

    transport::send_message(
        &endpoint,
        BridgeRole::Client,
        &mut client_state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ClientHello(ClientHello {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            requested_session: endpoint.session.clone(),
            requested_scene: Some("default".to_string()),
            requested_role: BridgeControlSlot::Fighter1,
            launcher_session_id: None,
            launch_token: None,
            child_kind: None,
        }),
    );

    let server_messages =
        transport::drain_messages(&endpoint, BridgeRole::Server, &mut server_state);
    assert_eq!(server_messages.len(), 1);
    let server_runtime = transport::snapshot(&server_state);
    assert_eq!(server_runtime.connected_clients, 1);
    match &server_messages[0].message {
        BridgeMessage::ClientHello(hello) => {
            assert_eq!(hello.protocol_version, BRIDGE_PROTOCOL_VERSION);
            assert_eq!(hello.requested_session, endpoint.session);
            assert_eq!(hello.requested_role, BridgeControlSlot::Fighter1);
        }
        other => panic!("expected ClientHello, got {other:?}"),
    }
    let Some(client_id) = server_messages[0].client_id.clone() else {
        panic!("server should see client id for inbound client message");
    };

    transport::send_message(
        &endpoint,
        BridgeRole::Server,
        &mut server_state,
        BridgeDispatchTarget::Client(client_id),
        BridgeMessage::ServerHello(ServerHello {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            accepted_session: endpoint.session.clone(),
            accepted_scene: "default".to_string(),
            assigned_role: BridgeControlSlot::Fighter1,
            fixed_time_step_seconds: 1.0 / 60.0,
        }),
    );
    transport::send_message(
        &endpoint,
        BridgeRole::Server,
        &mut server_state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ServerSnapshotFrame(ServerSnapshotFrame {
            tick: 12,
            session: endpoint.session.clone(),
            connected_clients: 1,
            occupied_slots: vec![BridgeControlSlot::Fighter1],
            processed_input_ticks: Vec::new(),
            observation: ObservationBundle::default(),
        }),
    );

    let client_messages =
        transport::drain_messages(&endpoint, BridgeRole::Client, &mut client_state);
    assert_eq!(client_messages.len(), 2);
    assert!(matches!(
        &client_messages[0].message,
        BridgeMessage::ServerHello(ServerHello {
            assigned_role: BridgeControlSlot::Fighter1,
            ..
        })
    ));
    assert!(matches!(
        &client_messages[1].message,
        BridgeMessage::ServerSnapshotFrame(ServerSnapshotFrame { tick: 12, .. })
    ));

    let client_runtime = transport::snapshot(&client_state);
    let server_runtime = transport::snapshot(&server_state);
    assert_eq!(client_runtime.last_server_tick, Some(12));
    assert_eq!(server_runtime.last_client_tick, None);
}

#[test]
fn tcp_bridge_supports_basic_handshake_flow() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let session = format!(
        "bridge-smoke-tcp-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let endpoint = BridgeEndpointConfig {
        session: session.clone(),
        transport: BridgeTransport::Tcp,
        address: format!("127.0.0.1:{port}"),
    };

    let mut server_state = IpcBridgeState::default();
    let mut client_state = IpcBridgeState::default();

    transport::connect_endpoint(&endpoint, BridgeRole::Server, &mut server_state);
    transport::connect_endpoint(&endpoint, BridgeRole::Client, &mut client_state);

    thread::sleep(Duration::from_millis(100));

    transport::send_message(
        &endpoint,
        BridgeRole::Client,
        &mut client_state,
        BridgeDispatchTarget::Broadcast,
        BridgeMessage::ClientHello(ClientHello {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            requested_session: endpoint.session.clone(),
            requested_scene: Some("default".to_string()),
            requested_role: BridgeControlSlot::Fighter2,
            launcher_session_id: None,
            launch_token: None,
            child_kind: None,
        }),
    );

    let mut server_messages = Vec::new();
    for _ in 0..20 {
        server_messages =
            transport::drain_messages(&endpoint, BridgeRole::Server, &mut server_state);
        if !server_messages.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(server_messages.len(), 1);
    let client_id = server_messages[0]
        .client_id
        .clone()
        .expect("server should see tcp client id");
    assert!(matches!(
        &server_messages[0].message,
        BridgeMessage::ClientHello(ClientHello {
            requested_role: BridgeControlSlot::Fighter2,
            ..
        })
    ));

    transport::send_message(
        &endpoint,
        BridgeRole::Server,
        &mut server_state,
        BridgeDispatchTarget::Client(client_id),
        BridgeMessage::ServerHello(ServerHello {
            protocol_version: BRIDGE_PROTOCOL_VERSION,
            accepted_session: endpoint.session.clone(),
            accepted_scene: "default".to_string(),
            assigned_role: BridgeControlSlot::Fighter2,
            fixed_time_step_seconds: 1.0 / 60.0,
        }),
    );

    let mut client_messages = Vec::new();
    for _ in 0..20 {
        client_messages =
            transport::drain_messages(&endpoint, BridgeRole::Client, &mut client_state);
        if !client_messages.is_empty() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(client_messages.len(), 1);
    assert!(matches!(
        &client_messages[0].message,
        BridgeMessage::ServerHello(ServerHello {
            assigned_role: BridgeControlSlot::Fighter2,
            ..
        })
    ));
}
