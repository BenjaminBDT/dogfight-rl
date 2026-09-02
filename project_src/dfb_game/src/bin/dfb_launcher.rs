use std::io::{self, BufRead, BufReader, Stdout, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use dfb_game::bridge::protocol::{
    BRIDGE_PROTOCOL_VERSION, BridgeControlSlot, BridgeMessage, ClientLobbyHello, ClientLobbyReady,
    ClientLobbySelectRole, LobbyClientKind, LobbySessionPhase, ServerLobbyClose, ServerLobbyStart,
    ServerLobbyState,
};
use dfb_game::core::config::resolve_project_root;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherFocus {
    Address,
    Session,
    Role,
    Pilot,
    Ready,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherPilotMode {
    Human,
    FollowAi,
    ImperfectFollowAi,
    TeacherFollowAi,
    Model,
}

#[derive(Debug)]
struct LauncherState {
    address: String,
    session: String,
    launcher_session_id: Option<String>,
    selected_role: BridgeControlSlot,
    pilot_mode: LauncherPilotMode,
    ready_locked: bool,
    focus: LauncherFocus,
    status_line: String,
    lobby_state: Option<ServerLobbyState>,
    pending_start: Option<ServerLobbyStart>,
    pending_close: Option<ServerLobbyClose>,
    connected: bool,
    launched_role: Option<BridgeControlSlot>,
    child_running: bool,
}

impl Default for LauncherState {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:50051".to_string(),
            session: "default".to_string(),
            launcher_session_id: None,
            selected_role: BridgeControlSlot::Fighter1,
            pilot_mode: LauncherPilotMode::Human,
            ready_locked: false,
            focus: LauncherFocus::Address,
            status_line: "Edit address/session, choose role, then mark ready.".to_string(),
            lobby_state: None,
            pending_start: None,
            pending_close: None,
            connected: false,
            launched_role: None,
            child_running: false,
        }
    }
}

struct LauncherTerminal {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Drop for LauncherTerminal {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[derive(Debug)]
enum LauncherInbound {
    Connected,
    Disconnected(String),
    LobbyState(ServerLobbyState),
    LobbyStart(ServerLobbyStart),
    LobbyClose(ServerLobbyClose),
    Ignored(String),
    Error(String),
}

fn main() -> Result<()> {
    let mut terminal = setup_terminal()?;
    let mut state = LauncherState::default();
    let mut network: Option<Sender<BridgeMessage>> = None;
    let mut inbound_rx: Option<Receiver<LauncherInbound>> = None;
    let mut child_process: Option<Child> = None;

    loop {
        if let Some(rx) = inbound_rx.as_ref() {
            drain_network_events(rx, &mut state);
        }
        handle_lobby_close(&mut child_process, &mut state)?;
        poll_child_process(&mut child_process, &mut state);
        maybe_launch_child(&mut child_process, &mut state);

        terminal
            .terminal
            .draw(|frame| render_launcher(frame, &state))?;

        if event::poll(Duration::from_millis(50))? {
            let CrosstermEvent::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => break,
                KeyCode::Up => {
                    state.focus = match state.focus {
                        LauncherFocus::Address => LauncherFocus::Ready,
                        LauncherFocus::Session => LauncherFocus::Address,
                        LauncherFocus::Role => LauncherFocus::Session,
                        LauncherFocus::Pilot => LauncherFocus::Role,
                        LauncherFocus::Ready => LauncherFocus::Pilot,
                    };
                }
                KeyCode::Down => {
                    state.focus = match state.focus {
                        LauncherFocus::Address => LauncherFocus::Session,
                        LauncherFocus::Session => LauncherFocus::Role,
                        LauncherFocus::Role => LauncherFocus::Pilot,
                        LauncherFocus::Pilot => LauncherFocus::Ready,
                        LauncherFocus::Ready => LauncherFocus::Address,
                    };
                }
                KeyCode::Left => match state.focus {
                    LauncherFocus::Role if !state.ready_locked && !state.child_running => {
                        cycle_role(&mut state.selected_role, false);
                        send_role_selection(network.as_ref(), &mut state);
                    }
                    LauncherFocus::Pilot if !state.child_running => {
                        cycle_pilot_mode(&mut state.pilot_mode, false);
                        state.status_line =
                            format!("Selected pilot mode {}", pilot_mode_label(state.pilot_mode));
                    }
                    LauncherFocus::Ready if !state.child_running => {
                        set_ready(network.as_ref(), &mut state, false);
                    }
                    _ => {}
                },
                KeyCode::Right => match state.focus {
                    LauncherFocus::Role if !state.ready_locked && !state.child_running => {
                        cycle_role(&mut state.selected_role, true);
                        send_role_selection(network.as_ref(), &mut state);
                    }
                    LauncherFocus::Pilot if !state.child_running => {
                        cycle_pilot_mode(&mut state.pilot_mode, true);
                        state.status_line =
                            format!("Selected pilot mode {}", pilot_mode_label(state.pilot_mode));
                    }
                    LauncherFocus::Ready if !state.child_running => {
                        set_ready(network.as_ref(), &mut state, true);
                    }
                    _ => {}
                },
                KeyCode::Enter => {
                    connect_launcher(&mut network, &mut inbound_rx, &mut state);
                }
                KeyCode::Backspace => match state.focus {
                    LauncherFocus::Address => {
                        state.address.pop();
                    }
                    LauncherFocus::Session => {
                        state.session.pop();
                    }
                    LauncherFocus::Role | LauncherFocus::Pilot | LauncherFocus::Ready => {}
                },
                KeyCode::Char(ch) => match state.focus {
                    LauncherFocus::Address => state.address.push(ch),
                    LauncherFocus::Session => state.session.push(ch),
                    LauncherFocus::Role | LauncherFocus::Pilot | LauncherFocus::Ready => {}
                },
                _ => {}
            }
        }

        if let Some(tx) = network.as_ref() {
            let _ = tx;
        }
    }

    Ok(())
}

fn setup_terminal() -> Result<LauncherTerminal> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    terminal.hide_cursor().ok();
    Ok(LauncherTerminal { terminal })
}

fn spawn_lobby_connection(
    address: &str,
    session: &str,
) -> Result<(Sender<BridgeMessage>, Receiver<LauncherInbound>)> {
    let stream =
        TcpStream::connect(address).with_context(|| format!("failed to connect to {address}"))?;
    stream
        .set_nodelay(true)
        .with_context(|| format!("failed to set TCP_NODELAY on {address}"))?;
    let writer_stream = stream
        .try_clone()
        .with_context(|| format!("failed to clone TCP stream for {address}"))?;

    let (outgoing_tx, outgoing_rx) = mpsc::channel::<BridgeMessage>();
    let (inbound_tx, inbound_rx) = mpsc::channel::<LauncherInbound>();

    let hello = BridgeMessage::ClientLobbyHello(ClientLobbyHello {
        protocol_version: BRIDGE_PROTOCOL_VERSION,
        requested_session: session.to_string(),
        requested_scene: None,
        client_kind: LobbyClientKind::Launcher,
        launcher_session_id: None,
    });
    outgoing_tx
        .send(hello)
        .context("failed to enqueue ClientLobbyHello")?;

    let writer_inbound_tx = inbound_tx.clone();
    thread::spawn(move || {
        let mut writer = writer_stream;
        while let Ok(message) = outgoing_rx.recv() {
            if serde_json::to_writer(&mut writer, &message).is_err()
                || writer.write_all(b"\n").is_err()
                || writer.flush().is_err()
            {
                let _ = writer_inbound_tx.send(LauncherInbound::Disconnected(
                    "connection closed while writing".to_string(),
                ));
                break;
            }
        }
    });

    let inbound_tx_clone = inbound_tx.clone();
    thread::spawn(move || {
        let _ = inbound_tx_clone.send(LauncherInbound::Connected);
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = inbound_tx_clone.send(LauncherInbound::Disconnected(
                        "server closed the connection".to_string(),
                    ));
                    break;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<BridgeMessage>(trimmed) {
                        Ok(BridgeMessage::ServerLobbyState(state)) => {
                            let _ = inbound_tx_clone.send(LauncherInbound::LobbyState(state));
                        }
                        Ok(BridgeMessage::ServerLobbyStart(start)) => {
                            let _ = inbound_tx_clone.send(LauncherInbound::LobbyStart(start));
                        }
                        Ok(BridgeMessage::ServerLobbyClose(close)) => {
                            let _ = inbound_tx_clone.send(LauncherInbound::LobbyClose(close));
                        }
                        Ok(other) => {
                            let _ = inbound_tx_clone.send(LauncherInbound::Ignored(format!(
                                "ignored message: {other:?}"
                            )));
                        }
                        Err(error) => {
                            let _ = inbound_tx_clone
                                .send(LauncherInbound::Error(format!("decode error: {error}")));
                        }
                    }
                }
                Err(error) => {
                    let _ = inbound_tx_clone.send(LauncherInbound::Disconnected(format!(
                        "socket read failed: {error}"
                    )));
                    break;
                }
            }
        }
    });

    Ok((outgoing_tx, inbound_rx))
}

fn drain_network_events(rx: &Receiver<LauncherInbound>, state: &mut LauncherState) {
    loop {
        match rx.try_recv() {
            Ok(LauncherInbound::Connected) => {
                state.connected = true;
                state.status_line = "Connected. Waiting for lobby state...".to_string();
            }
            Ok(LauncherInbound::Disconnected(reason)) => {
                state.connected = false;
                state.status_line = format!("Disconnected: {reason}");
            }
            Ok(LauncherInbound::LobbyState(lobby)) => {
                state.launcher_session_id = lobby.launcher_session_id.clone();
                if let Some(role) = lobby.assigned_role {
                    state.selected_role = role;
                }
                state.ready_locked = lobby.ready;
                state.status_line = format!(
                    "Lobby connected: phase={:?}, assigned={:?}, ready={}",
                    lobby.phase, lobby.assigned_role, lobby.ready
                );
                state.lobby_state = Some(lobby);
            }
            Ok(LauncherInbound::LobbyStart(start)) => {
                state.status_line =
                    format!("Server requested match start as {:?}", start.assigned_role);
                state.pending_start = Some(start);
            }
            Ok(LauncherInbound::LobbyClose(close)) => {
                state.status_line = format!("Session closed: {}", close.reason);
                state.pending_close = Some(close);
                state.child_running = false;
                state.launched_role = None;
            }
            Ok(LauncherInbound::Ignored(message)) => {
                state.status_line = message;
            }
            Ok(LauncherInbound::Error(message)) => {
                state.status_line = message;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                state.connected = false;
                state.status_line = "Network worker terminated".to_string();
                break;
            }
        }
    }
}

fn render_launcher(frame: &mut ratatui::Frame<'_>, state: &LauncherState) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Min(6),
            Constraint::Length(4),
        ])
        .split(area);

    let block = Block::default()
        .title(" DFB Launcher ")
        .borders(Borders::ALL);
    frame.render_widget(block, rows[0]);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Address: ",
                    focus_style(state.focus == LauncherFocus::Address),
                ),
                Span::raw(&state.address),
            ]),
            Line::from(vec![
                Span::styled(
                    "Session: ",
                    focus_style(state.focus == LauncherFocus::Session),
                ),
                Span::raw(&state.session),
            ]),
            Line::from(vec![
                Span::styled("Role: ", focus_style(state.focus == LauncherFocus::Role)),
                role_option_span(
                    state.selected_role == BridgeControlSlot::Fighter1,
                    "fighter1",
                ),
                Span::raw(" "),
                role_option_span(
                    state.selected_role == BridgeControlSlot::Fighter2,
                    "fighter2",
                ),
                Span::raw(" "),
                role_option_span(
                    state.selected_role == BridgeControlSlot::Spectator,
                    "observer",
                ),
            ]),
            Line::from(vec![
                Span::styled("Pilot: ", focus_style(state.focus == LauncherFocus::Pilot)),
                pilot_option_span(state.pilot_mode == LauncherPilotMode::Human, "human"),
                Span::raw(" "),
                pilot_option_span(state.pilot_mode == LauncherPilotMode::FollowAi, "follow-ai"),
                Span::raw(" "),
                pilot_option_span(
                    state.pilot_mode == LauncherPilotMode::ImperfectFollowAi,
                    "follow-ai*",
                ),
                Span::raw(" "),
                pilot_option_span(
                    state.pilot_mode == LauncherPilotMode::TeacherFollowAi,
                    "teacher",
                ),
                Span::raw(" "),
                pilot_option_span(state.pilot_mode == LauncherPilotMode::Model, "model"),
            ]),
            Line::from(vec![
                Span::styled("Ready: ", focus_style(state.focus == LauncherFocus::Ready)),
                ready_option_span(!state.ready_locked, "unready"),
                Span::raw(" "),
                ready_option_span(state.ready_locked, "ready"),
            ]),
        ]),
        rows[0].inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        }),
    );

    let status_block = Block::default().title(" Status ").borders(Borders::ALL);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(state.status_line.as_str()),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    if state.connected {
                        "CONNECTED"
                    } else {
                        "DISCONNECTED"
                    },
                    Style::default()
                        .fg(if state.connected {
                            Color::Green
                        } else {
                            Color::Yellow
                        })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(
                    match state
                        .lobby_state
                        .as_ref()
                        .map(|lobby| lobby.phase)
                        .unwrap_or(LobbySessionPhase::Lobby)
                    {
                        LobbySessionPhase::Lobby => "Waiting in lobby",
                        LobbySessionPhase::Starting => "Starting",
                        LobbySessionPhase::Running => "Running",
                        LobbySessionPhase::Ending => "Ending",
                    },
                ),
                Span::raw("  "),
                Span::raw(if state.child_running {
                    "Child running"
                } else {
                    "No child running"
                }),
            ]),
        ])
        .block(status_block),
        rows[1],
    );

    let lobby_block = Block::default().title(" Lobby ").borders(Borders::ALL);
    frame.render_widget(lobby_block, rows[2]);

    let lobby_items = if let Some(lobby) = &state.lobby_state {
        let mut items = vec![
            ListItem::new(format!("phase: {:?}", lobby.phase)),
            ListItem::new(format!("assigned_role: {:?}", lobby.assigned_role)),
            ListItem::new(format!("ready: {}", lobby.ready)),
            ListItem::new(format!("pilot: {}", pilot_mode_label(state.pilot_mode))),
            ListItem::new(format!("scene: {}", lobby.accepted_scene)),
            ListItem::new(format!(
                "child: {}{}",
                if state.child_running {
                    "running"
                } else {
                    "idle"
                },
                state
                    .launched_role
                    .map(|role| format!(" as {}", role_label(role)))
                    .unwrap_or_default()
            )),
            ListItem::new("slots:".to_string()),
        ];
        for slot in &lobby.slots {
            items.push(ListItem::new(format!(
                "  {}  role={:?} ready={} kind={:?}",
                slot.client_id, slot.assigned_role, slot.ready, slot.client_kind
            )));
        }
        items
    } else {
        vec![ListItem::new("No lobby state received yet.")]
    };
    frame.render_widget(
        List::new(lobby_items),
        rows[2].inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        }),
    );

    let footer = Paragraph::new(vec![
        Line::from("[Up/Down] Focus  [Left/Right] Change current field"),
        Line::from("[Enter] Connect  [q/Esc] Quit"),
    ])
    .block(Block::default().title(" Controls ").borders(Borders::ALL));
    frame.render_widget(footer, rows[3]);
}

fn focus_style(focused: bool) -> Style {
    if focused {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    }
}

fn role_option_span(selected: bool, label: &str) -> Span<'static> {
    Span::styled(
        format!("[{}]", label),
        if selected {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        },
    )
}

fn ready_option_span(selected: bool, label: &str) -> Span<'static> {
    Span::styled(
        format!("[{}]", label),
        if selected {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        },
    )
}

fn pilot_option_span(selected: bool, label: &str) -> Span<'static> {
    Span::styled(
        format!("[{}]", label),
        if selected {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        },
    )
}

fn role_label(role: BridgeControlSlot) -> &'static str {
    match role {
        BridgeControlSlot::Fighter1 => "fighter1",
        BridgeControlSlot::Fighter2 => "fighter2",
        BridgeControlSlot::Spectator => "observer",
    }
}

fn pilot_mode_label(mode: LauncherPilotMode) -> &'static str {
    match mode {
        LauncherPilotMode::Human => "human",
        LauncherPilotMode::FollowAi => "follow-ai",
        LauncherPilotMode::ImperfectFollowAi => "imperfect-follow-ai",
        LauncherPilotMode::TeacherFollowAi => "teacher-follow-ai",
        LauncherPilotMode::Model => "model",
    }
}

fn cycle_role(role: &mut BridgeControlSlot, forward: bool) {
    *role = match (*role, forward) {
        (BridgeControlSlot::Fighter1, true) => BridgeControlSlot::Fighter2,
        (BridgeControlSlot::Fighter2, true) => BridgeControlSlot::Spectator,
        (BridgeControlSlot::Spectator, true) => BridgeControlSlot::Fighter1,
        (BridgeControlSlot::Fighter1, false) => BridgeControlSlot::Spectator,
        (BridgeControlSlot::Fighter2, false) => BridgeControlSlot::Fighter1,
        (BridgeControlSlot::Spectator, false) => BridgeControlSlot::Fighter2,
    };
}

fn cycle_pilot_mode(mode: &mut LauncherPilotMode, forward: bool) {
    *mode = match (*mode, forward) {
        (LauncherPilotMode::Human, true) => LauncherPilotMode::FollowAi,
        (LauncherPilotMode::FollowAi, true) => LauncherPilotMode::ImperfectFollowAi,
        (LauncherPilotMode::ImperfectFollowAi, true) => LauncherPilotMode::TeacherFollowAi,
        (LauncherPilotMode::TeacherFollowAi, true) => LauncherPilotMode::Model,
        (LauncherPilotMode::Model, true) => LauncherPilotMode::Human,
        (LauncherPilotMode::Human, false) => LauncherPilotMode::Model,
        (LauncherPilotMode::FollowAi, false) => LauncherPilotMode::Human,
        (LauncherPilotMode::ImperfectFollowAi, false) => LauncherPilotMode::FollowAi,
        (LauncherPilotMode::TeacherFollowAi, false) => LauncherPilotMode::ImperfectFollowAi,
        (LauncherPilotMode::Model, false) => LauncherPilotMode::TeacherFollowAi,
    };
}

fn send_role_selection(network: Option<&Sender<BridgeMessage>>, state: &mut LauncherState) {
    let Some(tx) = network else {
        state.status_line = "Connect to server before selecting a role.".to_string();
        return;
    };
    if tx
        .send(BridgeMessage::ClientLobbySelectRole(
            ClientLobbySelectRole {
                requested_role: state.selected_role,
            },
        ))
        .is_ok()
    {
        state.status_line = format!("Requested role {}", role_label(state.selected_role));
    } else {
        state.status_line = "Failed to send role selection to server.".to_string();
    }
}

fn set_ready(network: Option<&Sender<BridgeMessage>>, state: &mut LauncherState, ready: bool) {
    let Some(tx) = network else {
        state.status_line = "Connect to server before changing ready state.".to_string();
        return;
    };
    if tx
        .send(BridgeMessage::ClientLobbyReady(ClientLobbyReady { ready }))
        .is_ok()
    {
        state.ready_locked = ready;
        state.status_line = if ready {
            format!("Ready locked as {}", role_label(state.selected_role))
        } else {
            "Ready cleared.".to_string()
        };
    } else {
        state.status_line = "Failed to send ready state to server.".to_string();
    }
}

fn connect_launcher(
    network: &mut Option<Sender<BridgeMessage>>,
    inbound_rx: &mut Option<Receiver<LauncherInbound>>,
    state: &mut LauncherState,
) {
    match spawn_lobby_connection(&state.address, &state.session) {
        Ok((tx, rx)) => {
            *network = Some(tx);
            *inbound_rx = Some(rx);
            state.connected = false;
            state.launcher_session_id = None;
            state.lobby_state = None;
            state.pending_start = None;
            state.pending_close = None;
            state.ready_locked = false;
            state.launched_role = None;
            state.child_running = false;
            state.status_line = format!(
                "Connecting to {} / session {}",
                state.address, state.session
            );
        }
        Err(error) => {
            *network = None;
            *inbound_rx = None;
            state.connected = false;
            state.status_line = error.to_string();
        }
    }
}

fn maybe_launch_child(child_process: &mut Option<Child>, state: &mut LauncherState) {
    let Some(start) = state.pending_start.take() else {
        return;
    };
    if child_process.is_some() {
        state.status_line = "Child client already running; ignoring duplicate start.".to_string();
        return;
    }
    let child = match spawn_client_process(
        &state.address,
        &state.session,
        &start.accepted_scene,
        start.assigned_role,
        state.pilot_mode,
        start.launcher_session_id.as_deref(),
        start.launch_token.as_deref(),
        start.child_kind,
    ) {
        Ok(child) => child,
        Err(error) => {
            state.status_line = format!(
                "Failed to launch {}: {error}",
                role_label(start.assigned_role)
            );
            return;
        }
    };
    state.child_running = true;
    state.launched_role = Some(start.assigned_role);
    state.status_line = format!(
        "Launched {} for session {}.",
        role_label(start.assigned_role),
        start.accepted_session
    );
    *child_process = Some(child);
}

fn poll_child_process(child_process: &mut Option<Child>, state: &mut LauncherState) {
    let Some(child) = child_process.as_mut() else {
        return;
    };
    match child.try_wait() {
        Ok(Some(status)) => {
            state.child_running = false;
            state.status_line = format!("Child client exited with status {status}.");
            *child_process = None;
        }
        Ok(None) => {}
        Err(error) => {
            state.child_running = false;
            state.status_line = format!("Failed to poll child client: {error}");
            *child_process = None;
        }
    }
}

fn handle_lobby_close(child_process: &mut Option<Child>, state: &mut LauncherState) -> Result<()> {
    let Some(close) = state.pending_close.take() else {
        return Ok(());
    };
    if let Some(mut child) = child_process.take() {
        child
            .kill()
            .with_context(|| "failed to terminate launched child client")?;
        let _ = child.wait();
    }
    state.child_running = false;
    state.launched_role = None;
    state.ready_locked = false;
    state.status_line = format!("Session closed: {}. Returned to lobby.", close.reason);
    Ok(())
}

fn spawn_client_process(
    address: &str,
    session: &str,
    scene: &str,
    role: BridgeControlSlot,
    pilot_mode: LauncherPilotMode,
    launcher_session_id: Option<&str>,
    launch_token: Option<&str>,
    child_kind: Option<LobbyClientKind>,
) -> Result<Child> {
    let distribution_root = resolve_distribution_root();
    let project_root = distribution_root
        .clone()
        .unwrap_or_else(resolve_project_root);
    let model_checkpoint = std::env::var("DFB_MODEL_CHECKPOINT").ok();
    let model_dataset_root = std::env::var("DFB_MODEL_DATASET_ROOT").ok();
    let model_python = std::env::var("DFB_MODEL_PYTHON").ok();
    let model_device = std::env::var("DFB_MODEL_DEVICE").ok();
    let model_observation_source = std::env::var("DFB_MODEL_OBSERVATION_SOURCE").ok();
    let (binary_name, mut args) = match role {
        BridgeControlSlot::Fighter1 => (
            "dfb_client_gameplay",
            vec![
                "--bridge".to_string(),
                "tcp".to_string(),
                "--bridge-addr".to_string(),
                address.to_string(),
                "--bridge-session".to_string(),
                session.to_string(),
                "--scene".to_string(),
                scene.to_string(),
                "--control-role".to_string(),
                "fighter1".to_string(),
            ],
        ),
        BridgeControlSlot::Fighter2 => (
            "dfb_client_gameplay",
            vec![
                "--bridge".to_string(),
                "tcp".to_string(),
                "--bridge-addr".to_string(),
                address.to_string(),
                "--bridge-session".to_string(),
                session.to_string(),
                "--scene".to_string(),
                scene.to_string(),
                "--control-role".to_string(),
                "fighter2".to_string(),
            ],
        ),
        BridgeControlSlot::Spectator => (
            "dfb_client_observer",
            vec![
                "--live".to_string(),
                "--bridge".to_string(),
                "tcp".to_string(),
                "--bridge-addr".to_string(),
                address.to_string(),
                "--bridge-session".to_string(),
                session.to_string(),
                "--observed-role".to_string(),
                "fighter1".to_string(),
            ],
        ),
    };

    if let Some(launcher_session_id) = launcher_session_id {
        args.extend([
            "--launcher-session-id".to_string(),
            launcher_session_id.to_string(),
        ]);
    }
    if let Some(launch_token) = launch_token {
        args.extend(["--launch-token".to_string(), launch_token.to_string()]);
    }
    if let Some(child_kind) = child_kind {
        let child_kind = match child_kind {
            LobbyClientKind::Launcher => "launcher",
            LobbyClientKind::Gameplay => "gameplay",
            LobbyClientKind::Observer => "observer",
        };
        args.extend(["--child-kind".to_string(), child_kind.to_string()]);
    }
    args.extend([
        "--project-root".to_string(),
        project_root.display().to_string(),
    ]);
    if matches!(pilot_mode, LauncherPilotMode::FollowAi)
        && !matches!(role, BridgeControlSlot::Spectator)
    {
        args.push("--follow-ai".to_string());
    }
    if matches!(pilot_mode, LauncherPilotMode::ImperfectFollowAi)
        && !matches!(role, BridgeControlSlot::Spectator)
    {
        args.push("--imperfect-follow-ai".to_string());
    }
    if matches!(pilot_mode, LauncherPilotMode::TeacherFollowAi)
        && !matches!(role, BridgeControlSlot::Spectator)
    {
        args.push("--teacher-follow-ai".to_string());
    }
    if matches!(pilot_mode, LauncherPilotMode::Model)
        && !matches!(role, BridgeControlSlot::Spectator)
    {
        args.push("--model-control".to_string());
        if let Some(value) = model_checkpoint {
            args.extend(["--model-checkpoint".to_string(), value]);
        }
        if let Some(value) = model_dataset_root {
            args.extend(["--model-dataset-root".to_string(), value]);
        }
        if let Some(value) = model_python {
            args.extend(["--model-python".to_string(), value]);
        }
        if let Some(value) = model_device {
            args.extend(["--model-device".to_string(), value]);
        }
        if let Some(value) = model_observation_source {
            args.extend(["--model-observation-source".to_string(), value]);
        }
    }

    if let Some(path) = resolve_distribution_binary(distribution_root.as_deref(), binary_name) {
        return Command::new(path)
            .args(&args)
            .current_dir(distribution_root.as_deref().unwrap())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to launch {binary_name}"));
    }

    let mut cargo_args = vec!["run".to_string()];
    if running_release_profile() {
        cargo_args.push("--release".to_string());
    }
    cargo_args.extend([
        "--bin".to_string(),
        binary_name.to_string(),
        "--".to_string(),
    ]);
    cargo_args.append(&mut args);

    let cargo_workdir = distribution_root
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    Command::new("cargo")
        .args(&cargo_args)
        .current_dir(cargo_workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch {binary_name} via cargo"))
}

fn resolve_distribution_binary(
    root: Option<&std::path::Path>,
    binary_name: &str,
) -> Option<PathBuf> {
    let root = root?;
    let candidate = root.join(format!("{binary_name}{}", std::env::consts::EXE_SUFFIX));
    candidate.is_file().then_some(candidate)
}

fn resolve_distribution_root() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.to_path_buf()));
    if let Some(dir) = exe_dir
        .as_ref()
        .filter(|dir| looks_like_distribution_root(dir))
    {
        return Some(dir.clone());
    }
    let cwd = std::env::current_dir().ok();
    cwd.filter(|dir| looks_like_distribution_root(dir))
}

fn looks_like_distribution_root(path: &std::path::Path) -> bool {
    path.join("config").is_dir() && path.join("assets").is_dir()
}

fn running_release_profile() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|dir| dir.ends_with("release")))
        .unwrap_or(false)
}
