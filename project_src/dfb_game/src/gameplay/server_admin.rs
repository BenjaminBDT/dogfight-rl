use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Stdout};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use bevy::app::AppExit;
use bevy::ecs::system::SystemParam;
use bevy::log::tracing::field::{Field, Visit};
use bevy::log::tracing::{Event, Subscriber};
use bevy::log::tracing_subscriber::Layer;
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::{BoxedFmtLayer, BoxedLayer};
use bevy::prelude::*;
use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::api::environment::{DeterministicRng, EnvironmentSeed};
use crate::api::snapshot::WorldSnapshot;
use crate::bridge::protocol::BridgeControlSlot;
use crate::bridge::transport;
use crate::bridge::{
    BridgeServerMetricsControl, BridgeServerSessions, apply_server_authorities,
    connected_player_count,
};
use crate::core::config::{ConfigPaths, RepositoryConfig};
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::gameplay::reset::PendingMatchReset;
use crate::recording::{ActionRecordingState, ServerAuthoritativeRecording};
use crate::simulation::components::{AircraftRole, ControlAuthority};
use crate::simulation::resources::SimulationDebugState;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SERVER_ADMIN_BUFFER_LINES: usize = 400;
const SERVER_ADMIN_SCROLL_PAGE: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource)]
pub enum ServerAdminTimeMode {
    Realtime,
    Simulated,
}

#[derive(Debug, Clone, Copy, Resource)]
pub struct ServerAdminTimeControl {
    pub current_mode: ServerAdminTimeMode,
    pub requested_mode: ServerAdminTimeMode,
}

impl ServerAdminTimeControl {
    pub fn status_label(&self) -> String {
        if self.current_mode == self.requested_mode {
            format!("current={:?}", self.current_mode)
        } else {
            format!(
                "current={:?} requested={:?} (restart required)",
                self.current_mode, self.requested_mode
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource, Default)]
pub enum ServerAdminInterfaceMode {
    Off,
    #[default]
    Plain,
    Tui,
}

#[derive(Resource)]
pub struct ServerAdminCommandReceiver(pub Mutex<Receiver<String>>);

#[derive(SystemParam)]
pub struct ServerAdminCommandContext<'w, 's> {
    phase: Res<'w, State<MatchPhase>>,
    clock: Res<'w, MatchClock>,
    sim_debug: Res<'w, SimulationDebugState>,
    recording: Option<ResMut<'w, ActionRecordingState>>,
    server_recording: Option<ResMut<'w, ServerAuthoritativeRecording>>,
    metrics_control: Option<ResMut<'w, BridgeServerMetricsControl>>,
    bridge_state: Option<ResMut<'w, transport::IpcBridgeState>>,
    server_sessions: Option<ResMut<'w, BridgeServerSessions>>,
    time_control: ResMut<'w, ServerAdminTimeControl>,
    config_paths: ResMut<'w, ConfigPaths>,
    config: ResMut<'w, RepositoryConfig>,
    environment_seed: Option<ResMut<'w, EnvironmentSeed>>,
    deterministic_rng: Option<ResMut<'w, DeterministicRng>>,
    snapshot: Res<'w, WorldSnapshot>,
    authority_query: Query<'w, 's, (&'static AircraftRole, &'static mut ControlAuthority)>,
    admin: ResMut<'w, ServerAdminState>,
    pending_reset: ResMut<'w, PendingMatchReset>,
    pending_shutdown: ResMut<'w, PendingServerShutdown>,
    app_exit: MessageWriter<'w, AppExit>,
}

#[derive(Debug, Default, Clone, Copy, Resource)]
pub struct ServerAdminState {
    pub hold_match: bool,
}

#[derive(Debug, Default, Clone, Copy, Resource)]
pub struct PendingServerShutdown {
    pub requested: bool,
}

#[derive(Resource, Clone)]
pub struct ServerAdminLogBuffer(pub Arc<Mutex<VecDeque<String>>>);

#[derive(Resource, Default)]
pub struct ServerAdminUiState {
    pub logs: VecDeque<String>,
    pub output: VecDeque<String>,
    pub input: String,
    pub focused_pane: ServerAdminPane,
    pub logs_scroll_from_bottom: usize,
    pub output_scroll_from_bottom: usize,
    pub logs_scroll_x: usize,
    pub output_scroll_x: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ServerAdminPane {
    #[default]
    Logs,
    Output,
}

pub struct ServerAdminTerminal {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Drop for ServerAdminTerminal {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[derive(Debug, Default)]
struct AdminLogVisitor {
    message: Option<String>,
    extras: Vec<String>,
}

impl Visit for AdminLogVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.record_value(field, format!("{value:?}"));
    }
}

impl AdminLogVisitor {
    fn record_value(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
        } else {
            self.extras.push(format!("{}={}", field.name(), value));
        }
    }
}

struct ServerAdminLogLayer {
    lines: Arc<Mutex<VecDeque<String>>>,
}

impl<S> Layer<S> for ServerAdminLogLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = AdminLogVisitor::default();
        event.record(&mut visitor);
        let level = event.metadata().level();
        let target = event.metadata().target();
        let mut line = visitor
            .message
            .unwrap_or_else(|| event.metadata().name().to_string());
        if !visitor.extras.is_empty() {
            line.push(' ');
            line.push_str(&visitor.extras.join(" "));
        }
        push_buffer_line(
            &self.lines,
            format!("[{level}] {target}: {line}"),
            SERVER_ADMIN_BUFFER_LINES,
        );
    }
}

pub fn default_interface_mode() -> ServerAdminInterfaceMode {
    if io::stdin().is_terminal() && io::stdout().is_terminal() {
        ServerAdminInterfaceMode::Tui
    } else {
        ServerAdminInterfaceMode::Plain
    }
}

pub fn create_log_capture_layer(app: &mut App) -> Option<BoxedLayer> {
    let lines = Arc::new(Mutex::new(VecDeque::with_capacity(
        SERVER_ADMIN_BUFFER_LINES,
    )));
    app.insert_resource(ServerAdminLogBuffer(lines.clone()));
    Some(Box::new(ServerAdminLogLayer { lines }))
}

pub fn create_silent_fmt_layer(_app: &mut App) -> Option<BoxedFmtLayer> {
    Some(Box::new(
        bevy::log::tracing_subscriber::fmt::Layer::default().with_writer(std::io::sink),
    ))
}

pub fn start_plain_server_admin_console(mut commands: Commands) {
    let (sender, receiver) = mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut locked = stdin.lock();
        let mut line = String::new();
        loop {
            line.clear();
            match locked.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let command = line.trim().to_string();
                    if !command.is_empty() && sender.send(command).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    commands.insert_resource(ServerAdminCommandReceiver(Mutex::new(receiver)));
    for line in server_admin_help_lines("server admin console ready:") {
        info!("{line}");
    }
}

pub fn start_tui_server_admin(world: &mut World) {
    if let Err(error) = enable_raw_mode() {
        error!("failed to enable raw mode for server admin TUI: {error:#}");
        return;
    }
    let mut stdout = io::stdout();
    if let Err(error) = execute!(stdout, EnterAlternateScreen) {
        error!("failed to enter alternate screen for server admin TUI: {error:#}");
        let _ = disable_raw_mode();
        return;
    }
    let backend = CrosstermBackend::new(stdout);
    let Ok(terminal) = Terminal::new(backend) else {
        error!("failed to create server admin terminal");
        let _ = disable_raw_mode();
        return;
    };

    world.insert_non_send_resource(ServerAdminTerminal { terminal });
    world.init_resource::<ServerAdminUiState>();
    if world.get_resource::<ServerAdminLogBuffer>().is_none() {
        let lines = Arc::new(Mutex::new(VecDeque::with_capacity(
            SERVER_ADMIN_BUFFER_LINES,
        )));
        world.insert_resource(ServerAdminLogBuffer(lines));
    }
    for line in server_admin_help_lines("server admin TUI ready: Enter execute, Esc clear.") {
        info!("{line}");
    }
}

pub fn drive_plain_server_admin_commands(
    receiver: Res<ServerAdminCommandReceiver>,
    mut ctx: ServerAdminCommandContext,
) {
    let Ok(receiver) = receiver.0.lock() else {
        return;
    };
    loop {
        match receiver.try_recv() {
            Ok(command) => {
                for line in execute_server_admin_command(&command, &mut ctx) {
                    info!("{line}");
                }
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

pub fn drain_server_admin_logs(
    log_buffer: Res<ServerAdminLogBuffer>,
    mut ui: ResMut<ServerAdminUiState>,
) {
    let Ok(mut shared) = log_buffer.0.lock() else {
        return;
    };
    while let Some(line) = shared.pop_front() {
        push_line(&mut ui.logs, line, SERVER_ADMIN_BUFFER_LINES);
    }
}

pub fn drive_tui_server_admin_input(
    mut ctx: ServerAdminCommandContext,
    mut ui: ResMut<ServerAdminUiState>,
) {
    while event::poll(Duration::from_millis(0)).unwrap_or(false) {
        let Ok(crossterm_event) = event::read() else {
            break;
        };
        let CrosstermEvent::Key(key) = crossterm_event else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Tab => {
                ui.focused_pane = match ui.focused_pane {
                    ServerAdminPane::Logs => ServerAdminPane::Output,
                    ServerAdminPane::Output => ServerAdminPane::Logs,
                };
            }
            KeyCode::Left => scroll_active_pane_x(&mut ui, -1),
            KeyCode::Right => scroll_active_pane_x(&mut ui, 1),
            KeyCode::Up => scroll_active_pane(&mut ui, 1),
            KeyCode::Down => scroll_active_pane(&mut ui, -1),
            KeyCode::PageUp => scroll_active_pane(&mut ui, SERVER_ADMIN_SCROLL_PAGE as isize),
            KeyCode::PageDown => scroll_active_pane(&mut ui, -(SERVER_ADMIN_SCROLL_PAGE as isize)),
            KeyCode::Home => jump_active_pane_to_oldest(&mut ui),
            KeyCode::End => jump_active_pane_to_latest(&mut ui),
            KeyCode::Char(ch) => ui.input.push(ch),
            KeyCode::Backspace => {
                ui.input.pop();
            }
            KeyCode::Enter => {
                let command = ui.input.trim().to_string();
                if !command.is_empty() {
                    push_line(
                        &mut ui.output,
                        format!("> {command}"),
                        SERVER_ADMIN_BUFFER_LINES,
                    );
                    for line in execute_server_admin_command(&command, &mut ctx) {
                        push_line(&mut ui.output, line, SERVER_ADMIN_BUFFER_LINES);
                    }
                }
                ui.input.clear();
            }
            KeyCode::Esc => ui.input.clear(),
            _ => {}
        }
    }
}

pub fn process_pending_server_shutdown(
    recording: Option<Res<ActionRecordingState>>,
    mut pending_shutdown: ResMut<PendingServerShutdown>,
    mut app_exit: MessageWriter<AppExit>,
) {
    if !pending_shutdown.requested {
        return;
    }

    if let Some(recording) = recording.as_deref()
        && (recording.active || recording.pending_start || recording.pending_stop)
    {
        return;
    }

    pending_shutdown.requested = false;
    app_exit.write(AppExit::Success);
}

pub fn render_server_admin_tui(
    mut terminal: NonSendMut<ServerAdminTerminal>,
    ui: Res<ServerAdminUiState>,
) {
    if let Err(error) = terminal.terminal.draw(|frame| {
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(58),
                Constraint::Percentage(30),
                Constraint::Length(3),
            ])
            .split(frame.area());

        let log_height = areas[0].height.saturating_sub(2) as usize;
        let output_height = areas[1].height.saturating_sub(2) as usize;
        let log_width = areas[0].width.saturating_sub(2) as usize;
        let output_width = areas[1].width.saturating_sub(2) as usize;
        let log_lines = visible_lines(
            &ui.logs,
            ui.logs_scroll_from_bottom,
            ui.logs_scroll_x,
            log_height,
            log_width,
        );
        let output_lines = visible_lines(
            &ui.output,
            ui.output_scroll_from_bottom,
            ui.output_scroll_x,
            output_height,
            output_width,
        );
        let logs_title = pane_title(
            "Logs",
            ui.focused_pane == ServerAdminPane::Logs,
            ui.logs_scroll_from_bottom,
            ui.logs_scroll_x,
            ui.logs.len(),
            log_height,
        );
        let output_title = pane_title(
            "Command Output",
            ui.focused_pane == ServerAdminPane::Output,
            ui.output_scroll_from_bottom,
            ui.output_scroll_x,
            ui.output.len(),
            output_height,
        );

        frame.render_widget(
            Paragraph::new(log_lines)
                .block(Block::default().title(logs_title).borders(Borders::ALL)),
            areas[0],
        );
        frame.render_widget(
            Paragraph::new(output_lines)
                .block(Block::default().title(output_title).borders(Borders::ALL)),
            areas[1],
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    "> ",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(ui.input.as_str()),
            ]))
            .block(
                Block::default()
                    .title("Command Input [Tab switch pane, Arrows scroll, PageUp/PageDown]")
                    .borders(Borders::ALL),
            ),
            areas[2],
        );
    }) {
        error!("failed to draw server admin TUI: {error:#}");
    }
}

fn execute_server_admin_command(command: &str, ctx: &mut ServerAdminCommandContext) -> Vec<String> {
    match command {
        "help" => server_admin_help_lines("commands:"),
        "status" => {
            let (clients, players) = ctx
                .server_sessions
                .as_deref()
                .map(|sessions| (sessions.clients.len(), connected_player_count(sessions)))
                .unwrap_or((0, 0));
            let recording_status = ctx
                .recording
                .as_deref()
                .map(|recording| recording.status_label())
                .unwrap_or_else(|| "OFF".to_string());
            let server_recording_enabled = ctx
                .server_recording
                .as_ref()
                .map(|recording| recording.enabled)
                .unwrap_or(false);
            let metrics_enabled = ctx
                .metrics_control
                .as_ref()
                .map(|control| control.enabled)
                .unwrap_or(false);
            let metrics_interval = ctx
                .metrics_control
                .as_ref()
                .map(|control| control.interval_seconds)
                .unwrap_or(0.0);
            let scene_name = ctx
                .config_paths
                .scene_override
                .clone()
                .unwrap_or_else(|| ctx.config.game.active_scene.clone());
            let seed_value = ctx
                .environment_seed
                .as_ref()
                .map(|seed| seed.effective.to_string())
                .unwrap_or_else(|| "UNAVAILABLE".to_string());
            let time_status = ctx.time_control.status_label();
            vec![format!(
                "phase={:?} tick={} sim_time={:.2}s players={}/2 clients={} hold={} recording={} record_policy={} metrics={} interval={:.2}s scene={} seed={} time={}",
                ctx.phase.get(),
                ctx.sim_debug.tick_count,
                ctx.clock.elapsed_seconds,
                players,
                clients,
                ctx.admin.hold_match,
                recording_status,
                if server_recording_enabled {
                    "ON"
                } else {
                    "OFF"
                },
                if metrics_enabled { "ON" } else { "OFF" },
                metrics_interval,
                scene_name,
                seed_value,
                time_status,
            )]
        }
        "reset" => {
            if let Some(recording) = ctx.recording.as_deref_mut()
                && (recording.active || recording.pending_start || recording.pending_stop)
            {
                recording.pending_start = false;
                recording.pending_stop = true;
            }
            ctx.pending_reset.requested = true;
            vec!["requested match reset".to_string()]
        }
        "hold" => {
            ctx.admin.hold_match = true;
            vec!["enabled hold mode".to_string()]
        }
        "resume" => {
            ctx.admin.hold_match = false;
            vec!["resumed match flow".to_string()]
        }
        "record" | "record status" => vec![format!(
            "record policy={}",
            match ctx.server_recording.as_ref() {
                Some(recording) if recording.enabled => "ON",
                Some(_) => "OFF",
                None => "UNAVAILABLE",
            }
        )],
        "record on" => set_recording_policy(ctx.server_recording.as_deref_mut(), true),
        "record off" => set_recording_policy(ctx.server_recording.as_deref_mut(), false),
        "record toggle" => {
            let enabled = ctx
                .server_recording
                .as_ref()
                .map(|recording| !recording.enabled)
                .unwrap_or(false);
            set_recording_policy(ctx.server_recording.as_deref_mut(), enabled)
        }
        "metrics" | "metrics status" => vec![format!(
            "metrics policy={}",
            match ctx.metrics_control.as_ref() {
                Some(control) if control.enabled => "ON",
                Some(_) => "OFF",
                None => "UNAVAILABLE",
            }
        )],
        "metrics on" => set_metrics_policy(ctx.metrics_control.as_deref_mut(), true),
        "metrics off" => set_metrics_policy(ctx.metrics_control.as_deref_mut(), false),
        other if other.starts_with("metrics interval ") => set_metrics_interval(
            ctx.metrics_control.as_deref_mut(),
            other.trim_start_matches("metrics interval ").trim(),
        ),
        other if other.starts_with("disconnect ") => disconnect_session(
            other.trim_start_matches("disconnect ").trim(),
            ctx.bridge_state.as_deref_mut(),
            ctx.server_sessions.as_deref_mut(),
            &mut ctx.authority_query,
        ),
        other if other.starts_with("assign ") => assign_session(
            other.trim_start_matches("assign ").trim(),
            ctx.server_sessions.as_deref_mut(),
            &mut ctx.authority_query,
        ),
        other if other.starts_with("scene ") => set_scene_override(
            other.trim_start_matches("scene ").trim(),
            &mut ctx.config_paths,
            &mut ctx.config,
            &mut ctx.pending_reset,
        ),
        other if other.starts_with("seed ") => set_seed_override(
            other.trim_start_matches("seed ").trim(),
            ctx.environment_seed.as_deref_mut(),
            ctx.deterministic_rng.as_deref_mut(),
            &mut ctx.pending_reset,
        ),
        "time" | "time status" => vec![format!("time {}", ctx.time_control.status_label())],
        "time realtime" => set_time_mode(&mut ctx.time_control, ServerAdminTimeMode::Realtime),
        "time simulated" => set_time_mode(&mut ctx.time_control, ServerAdminTimeMode::Simulated),
        "snapshot" => save_snapshot(&ctx.config_paths, &ctx.snapshot),
        "quit" | "exit" => {
            if let Some(server_recording) = ctx.server_recording.as_deref_mut() {
                server_recording.enabled = false;
            }
            let should_defer = if let Some(recording) = ctx.recording.as_deref_mut() {
                if recording.active || recording.pending_start || recording.pending_stop {
                    recording.pending_start = false;
                    recording.pending_stop = true;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if should_defer {
                ctx.pending_shutdown.requested = true;
                vec!["stopping recording, disabling record policy, then shutting down".to_string()]
            } else {
                ctx.app_exit.write(AppExit::Success);
                vec!["requested shutdown".to_string()]
            }
        }
        other => vec![format!("unknown command: {other}")],
    }
}

fn server_admin_help_lines(header: &str) -> Vec<String> {
    vec![
        header.to_string(),
        "  help".to_string(),
        "  status".to_string(),
        "  reset".to_string(),
        "  hold".to_string(),
        "  resume".to_string(),
        "  record status".to_string(),
        "  record on".to_string(),
        "  record off".to_string(),
        "  record toggle".to_string(),
        "  metrics status".to_string(),
        "  metrics on".to_string(),
        "  metrics off".to_string(),
        "  metrics interval <seconds>".to_string(),
        "  disconnect <fighter1|fighter2|spectator|all|client_id>".to_string(),
        "  assign <client_id> <fighter1|fighter2|spectator>".to_string(),
        "  scene <name>".to_string(),
        "  seed <value>".to_string(),
        "  time status".to_string(),
        "  time realtime".to_string(),
        "  time simulated".to_string(),
        "  snapshot".to_string(),
        "  quit".to_string(),
    ]
}

fn push_line(lines: &mut VecDeque<String>, line: String, limit: usize) {
    lines.push_back(line);
    while lines.len() > limit {
        lines.pop_front();
    }
}

fn push_buffer_line(buffer: &Arc<Mutex<VecDeque<String>>>, line: String, limit: usize) {
    if let Ok(mut lines) = buffer.lock() {
        push_line(&mut lines, line, limit);
    }
}

fn visible_lines(
    lines: &VecDeque<String>,
    scroll_from_bottom: usize,
    scroll_x: usize,
    height: usize,
    width: usize,
) -> Vec<Line<'static>> {
    if height == 0 || width == 0 || lines.is_empty() {
        return Vec::new();
    }
    let len = lines.len();
    let max_scroll = len.saturating_sub(height);
    let scroll = scroll_from_bottom.min(max_scroll);
    let end = len.saturating_sub(scroll);
    let start = end.saturating_sub(height);
    lines
        .iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .map(|line| Line::from(truncate_for_width_and_offset(line, width, scroll_x)))
        .collect()
}

fn pane_title(
    base: &str,
    focused: bool,
    scroll_from_bottom: usize,
    scroll_x: usize,
    len: usize,
    height: usize,
) -> String {
    let max_scroll = len.saturating_sub(height);
    let scroll = scroll_from_bottom.min(max_scroll);
    let focus_marker = if focused { "*" } else { " " };
    format!("{focus_marker} {base} (y:{scroll}/{max_scroll}, x:{scroll_x})")
}

fn scroll_active_pane(ui: &mut ServerAdminUiState, delta: isize) {
    match ui.focused_pane {
        ServerAdminPane::Logs => {
            adjust_scroll(&mut ui.logs_scroll_from_bottom, ui.logs.len(), delta);
        }
        ServerAdminPane::Output => {
            adjust_scroll(&mut ui.output_scroll_from_bottom, ui.output.len(), delta);
        }
    }
}

fn jump_active_pane_to_oldest(ui: &mut ServerAdminUiState) {
    match ui.focused_pane {
        ServerAdminPane::Logs => ui.logs_scroll_from_bottom = ui.logs.len(),
        ServerAdminPane::Output => ui.output_scroll_from_bottom = ui.output.len(),
    }
}

fn jump_active_pane_to_latest(ui: &mut ServerAdminUiState) {
    match ui.focused_pane {
        ServerAdminPane::Logs => ui.logs_scroll_from_bottom = 0,
        ServerAdminPane::Output => ui.output_scroll_from_bottom = 0,
    }
}

fn adjust_scroll(scroll_from_bottom: &mut usize, len: usize, delta: isize) {
    if delta >= 0 {
        *scroll_from_bottom = scroll_from_bottom.saturating_add(delta as usize).min(len);
    } else {
        *scroll_from_bottom = scroll_from_bottom.saturating_sub((-delta) as usize);
    }
}

fn scroll_active_pane_x(ui: &mut ServerAdminUiState, delta: isize) {
    match ui.focused_pane {
        ServerAdminPane::Logs => adjust_scroll_x(&mut ui.logs_scroll_x, delta),
        ServerAdminPane::Output => adjust_scroll_x(&mut ui.output_scroll_x, delta),
    }
}

fn adjust_scroll_x(scroll_x: &mut usize, delta: isize) {
    if delta >= 0 {
        *scroll_x = scroll_x.saturating_add(delta as usize);
    } else {
        *scroll_x = scroll_x.saturating_sub((-delta) as usize);
    }
}

fn truncate_for_width_and_offset(line: &str, width: usize, offset: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars = line.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return String::new();
    }
    let start = offset.min(chars.len());
    let visible = &chars[start..];
    if visible.len() <= width {
        return visible.iter().collect();
    }
    if width == 1 {
        return "…".to_string();
    }
    let mut truncated = visible
        .iter()
        .take(width.saturating_sub(1))
        .collect::<String>();
    truncated.push('…');
    truncated
}

fn set_recording_policy(
    server_recording: Option<&mut ServerAuthoritativeRecording>,
    enabled: bool,
) -> Vec<String> {
    match server_recording {
        Some(server_recording) => {
            server_recording.enabled = enabled;
            vec![format!(
                "authoritative recording policy set to {}",
                if enabled { "ON" } else { "OFF" }
            )]
        }
        None => vec!["authoritative recording policy unavailable".to_string()],
    }
}

fn set_metrics_policy(
    metrics_control: Option<&mut BridgeServerMetricsControl>,
    enabled: bool,
) -> Vec<String> {
    match metrics_control {
        Some(metrics_control) => {
            metrics_control.enabled = enabled;
            vec![format!(
                "bridge server metrics set to {}",
                if enabled { "ON" } else { "OFF" }
            )]
        }
        None => vec!["bridge server metrics unavailable".to_string()],
    }
}

fn set_metrics_interval(
    metrics_control: Option<&mut BridgeServerMetricsControl>,
    value: &str,
) -> Vec<String> {
    let Ok(interval_seconds) = value.parse::<f64>() else {
        return vec![format!("invalid metrics interval: {value}")];
    };
    if interval_seconds <= 0.0 {
        return vec!["metrics interval must be > 0".to_string()];
    }
    match metrics_control {
        Some(metrics_control) => {
            metrics_control.interval_seconds = interval_seconds;
            vec![format!(
                "bridge server metrics interval set to {:.2}s",
                interval_seconds
            )]
        }
        None => vec!["bridge server metrics unavailable".to_string()],
    }
}

fn disconnect_session(
    target: &str,
    mut bridge_state: Option<&mut transport::IpcBridgeState>,
    server_sessions: Option<&mut BridgeServerSessions>,
    authority_query: &mut Query<(&AircraftRole, &mut ControlAuthority)>,
) -> Vec<String> {
    let Some(server_sessions) = server_sessions else {
        return vec!["server sessions unavailable".to_string()];
    };
    if target == "all" {
        let client_ids = server_sessions.clients.keys().cloned().collect::<Vec<_>>();
        for client_id in &client_ids {
            if let Some(bridge_state) = bridge_state.as_deref_mut() {
                transport::force_disconnect_client(bridge_state, client_id);
            }
        }
        let disconnected = client_ids.len();
        server_sessions.clients.clear();
        apply_server_authorities(server_sessions, authority_query);
        return vec![format!("disconnected all clients ({disconnected})")];
    }

    let (client_id, role) = if let Some(slot) = parse_disconnect_target(target) {
        let Some(client_id) = server_sessions
            .clients
            .iter()
            .find(|(_, session)| session.assigned_role == slot)
            .map(|(client_id, _)| client_id.clone())
        else {
            return vec![format!("no active client for {slot:?}")];
        };
        (client_id, slot)
    } else if let Some(session) = server_sessions.clients.get(target) {
        (target.to_string(), session.assigned_role)
    } else {
        return vec![format!(
            "unsupported disconnect target: {target} (expected fighter1|fighter2|spectator|all|client_id)"
        )];
    };

    server_sessions.clients.remove(&client_id);
    if let Some(bridge_state) = bridge_state {
        transport::force_disconnect_client(bridge_state, &client_id);
    }
    apply_server_authorities(server_sessions, authority_query);
    vec![format!("disconnected client_id={client_id} role={role:?}")]
}

fn parse_disconnect_target(target: &str) -> Option<BridgeControlSlot> {
    match target {
        "fighter1" => Some(BridgeControlSlot::Fighter1),
        "fighter2" => Some(BridgeControlSlot::Fighter2),
        "spectator" => Some(BridgeControlSlot::Spectator),
        _ => None,
    }
}

fn set_scene_override(
    scene_name: &str,
    config_paths: &mut ConfigPaths,
    config: &mut RepositoryConfig,
    pending_reset: &mut PendingMatchReset,
) -> Vec<String> {
    if scene_name.is_empty() {
        return vec!["usage: scene <name>".to_string()];
    }
    match RepositoryConfig::load_from_root_with_scene(&config_paths.project_root, Some(scene_name))
    {
        Ok(loaded) => {
            config_paths.scene_override = Some(scene_name.to_string());
            *config = loaded;
            pending_reset.requested = true;
            vec![format!(
                "scene override set to '{}' and next round reset requested",
                scene_name
            )]
        }
        Err(error) => vec![format!("failed to load scene '{scene_name}': {error:#}")],
    }
}

fn set_seed_override(
    seed_value: &str,
    environment_seed: Option<&mut EnvironmentSeed>,
    deterministic_rng: Option<&mut DeterministicRng>,
    pending_reset: &mut PendingMatchReset,
) -> Vec<String> {
    let Ok(seed) = seed_value.parse::<u64>() else {
        return vec![format!("invalid seed value: {seed_value}")];
    };
    match (environment_seed, deterministic_rng) {
        (Some(environment_seed), Some(deterministic_rng)) => {
            *environment_seed = EnvironmentSeed::from_request(Some(seed));
            deterministic_rng.0 = ChaCha8Rng::seed_from_u64(seed);
            pending_reset.requested = true;
            vec![format!("seed set to {seed} and next round reset requested")]
        }
        _ => vec!["seed control unavailable".to_string()],
    }
}

fn set_time_mode(
    time_control: &mut ServerAdminTimeControl,
    requested_mode: ServerAdminTimeMode,
) -> Vec<String> {
    time_control.requested_mode = requested_mode;
    vec![format!(
        "server time mode request set to {:?} (restart required; current={:?})",
        requested_mode, time_control.current_mode
    )]
}

fn save_snapshot(config_paths: &ConfigPaths, snapshot: &WorldSnapshot) -> Vec<String> {
    let snapshot_dir = config_paths.admin_snapshots_root();
    if let Err(error) = fs::create_dir_all(&snapshot_dir) {
        return vec![format!("failed to create snapshot directory: {error:#}")];
    }

    let file_path = snapshot_dir.join(format!("snapshot_tick_{:06}.ron", snapshot.tick));
    let Ok(text) =
        ron::ser::to_string_pretty(&snapshot.observation, ron::ser::PrettyConfig::default())
    else {
        return vec!["failed to serialize snapshot".to_string()];
    };
    if let Err(error) = fs::write(&file_path, text) {
        return vec![format!("failed to write snapshot: {error:#}")];
    }

    vec![format!(
        "saved snapshot tick={} phase={} path={}",
        snapshot.tick,
        snapshot.observation.state.match_phase,
        file_path.display()
    )]
}

fn assign_session(
    args: &str,
    server_sessions: Option<&mut BridgeServerSessions>,
    authority_query: &mut Query<(&AircraftRole, &mut ControlAuthority)>,
) -> Vec<String> {
    let Some(server_sessions) = server_sessions else {
        return vec!["server sessions unavailable".to_string()];
    };
    let mut parts = args.split_whitespace();
    let Some(client_id) = parts.next() else {
        return vec!["usage: assign <client_id> <fighter1|fighter2|spectator>".to_string()];
    };
    let Some(slot_name) = parts.next() else {
        return vec!["usage: assign <client_id> <fighter1|fighter2|spectator>".to_string()];
    };
    if parts.next().is_some() {
        return vec!["usage: assign <client_id> <fighter1|fighter2|spectator>".to_string()];
    }
    let Some(slot) = parse_disconnect_target(slot_name) else {
        return vec![format!(
            "unsupported assignment slot: {slot_name} (expected fighter1|fighter2|spectator)"
        )];
    };
    let Some(previous) = server_sessions
        .clients
        .get(client_id)
        .map(|session| session.assigned_role)
    else {
        return vec![format!("unknown client_id: {client_id}")];
    };
    if slot != BridgeControlSlot::Spectator {
        for (other_client_id, other_session) in &mut server_sessions.clients {
            if other_client_id != client_id && other_session.assigned_role == slot {
                other_session.assigned_role = BridgeControlSlot::Spectator;
            }
        }
    }
    if let Some(session) = server_sessions.clients.get_mut(client_id) {
        session.assigned_role = slot;
    }
    apply_server_authorities(server_sessions, authority_query);
    vec![format!(
        "assigned client_id={client_id} role={previous:?}->{slot:?}"
    )]
}
