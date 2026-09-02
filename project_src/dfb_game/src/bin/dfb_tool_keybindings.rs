use std::fs;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, MouseButton,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use dfb_game::core::config::{
    ActionBindingsConfig, InputBindingsConfig, InputConfig, MouseFlightAxisTarget,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table, Wrap,
};

const INPUT_CONFIG_PATH: &str = "config/dfb_game/input.ron";
const SLOT_COUNT: usize = 4;
const PARAM_COUNT: usize = 13;

#[derive(Clone, Copy, PartialEq, Eq)]
enum FocusArea {
    Bindings,
    Parameters,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BindingSlot {
    KeyboardPrimary,
    KeyboardSecondary,
    MousePrimary,
    MouseSecondary,
}

impl BindingSlot {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::KeyboardPrimary,
            1 => Self::KeyboardSecondary,
            2 => Self::MousePrimary,
            _ => Self::MouseSecondary,
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::KeyboardPrimary => "K1",
            Self::KeyboardSecondary => "K2",
            Self::MousePrimary => "M1",
            Self::MouseSecondary => "M2",
        }
    }

    fn accepts_keyboard(self) -> bool {
        matches!(self, Self::KeyboardPrimary | Self::KeyboardSecondary)
    }

    fn accepts_mouse(self) -> bool {
        matches!(self, Self::MousePrimary | Self::MouseSecondary)
    }
}

#[derive(Clone, Copy)]
struct ActionRowMeta {
    name: &'static str,
    label: &'static str,
}

const ACTION_ROWS: [ActionRowMeta; 17] = [
    ActionRowMeta {
        name: "throttle_up",
        label: "Throttle Up",
    },
    ActionRowMeta {
        name: "throttle_down",
        label: "Throttle Down",
    },
    ActionRowMeta {
        name: "brake",
        label: "Brake",
    },
    ActionRowMeta {
        name: "pitch_positive",
        label: "Pitch Up",
    },
    ActionRowMeta {
        name: "pitch_negative",
        label: "Pitch Down",
    },
    ActionRowMeta {
        name: "roll_positive",
        label: "Roll Left",
    },
    ActionRowMeta {
        name: "roll_negative",
        label: "Roll Right",
    },
    ActionRowMeta {
        name: "yaw_positive",
        label: "Yaw Left",
    },
    ActionRowMeta {
        name: "yaw_negative",
        label: "Yaw Right",
    },
    ActionRowMeta {
        name: "fire_gun",
        label: "Fire Gun",
    },
    ActionRowMeta {
        name: "repair_aircraft",
        label: "Repair",
    },
    ActionRowMeta {
        name: "toggle_controls_guide",
        label: "Toggle Guide",
    },
    ActionRowMeta {
        name: "reset_match",
        label: "Reset Match",
    },
    ActionRowMeta {
        name: "rear_view",
        label: "Rear View",
    },
    ActionRowMeta {
        name: "toggle_local_pilot_mode",
        label: "Pilot Mode",
    },
    ActionRowMeta {
        name: "toggle_audio_mute",
        label: "Toggle Mute",
    },
    ActionRowMeta {
        name: "toggle_mouse_capture",
        label: "Mouse Capture",
    },
];

const PICKABLE_BINDINGS: &[&str] = &[
    "ShiftLeft",
    "ShiftRight",
    "ControlLeft",
    "ControlRight",
    "AltLeft",
    "AltRight",
    "Space",
    "Tab",
    "Enter",
    "Backspace",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "Digit0",
    "Digit1",
    "Digit2",
    "Digit3",
    "Digit4",
    "Digit5",
    "Digit6",
    "Digit7",
    "Digit8",
    "Digit9",
    "KeyA",
    "KeyB",
    "KeyC",
    "KeyD",
    "KeyE",
    "KeyF",
    "KeyG",
    "KeyH",
    "KeyI",
    "KeyJ",
    "KeyK",
    "KeyL",
    "KeyM",
    "KeyN",
    "KeyO",
    "KeyP",
    "KeyQ",
    "KeyR",
    "KeyS",
    "KeyT",
    "KeyU",
    "KeyV",
    "KeyW",
    "KeyX",
    "KeyY",
    "KeyZ",
    "Left",
    "Right",
    "Middle",
    "Back",
    "Forward",
];

const PICKABLE_MOUSE_AXIS_TARGETS: &[MouseFlightAxisTarget; 3] = &[
    MouseFlightAxisTarget::Pitch,
    MouseFlightAxisTarget::Roll,
    MouseFlightAxisTarget::Yaw,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverlayMode {
    None,
    Capture,
    Picker,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PickerMode {
    Binding,
    MouseAxisTarget,
}

#[derive(Clone, Copy)]
struct ParameterMeta {
    label: &'static str,
}

const PARAM_ROWS: [ParameterMeta; PARAM_COUNT] = [
    ParameterMeta {
        label: "Mouse X Target",
    },
    ParameterMeta {
        label: "Mouse X Sensitivity",
    },
    ParameterMeta {
        label: "Mouse X Weight",
    },
    ParameterMeta {
        label: "Mouse X Invert",
    },
    ParameterMeta {
        label: "Mouse Y Target",
    },
    ParameterMeta {
        label: "Mouse Y Sensitivity",
    },
    ParameterMeta {
        label: "Mouse Y Weight",
    },
    ParameterMeta {
        label: "Mouse Y Invert",
    },
    ParameterMeta {
        label: "Throttle Weight",
    },
    ParameterMeta {
        label: "Pitch Weight",
    },
    ParameterMeta {
        label: "Roll Weight",
    },
    ParameterMeta {
        label: "Mouse Smoothing",
    },
    ParameterMeta {
        label: "Capture Mouse On Start",
    },
];

struct App {
    config_path: PathBuf,
    config: InputConfig,
    focus: FocusArea,
    selected_action: usize,
    selected_slot: usize,
    selected_param: usize,
    overlay_mode: OverlayMode,
    picker_mode: PickerMode,
    picker_state: ListState,
    status: String,
    dirty: bool,
    should_quit: bool,
}

impl App {
    fn new(config_path: PathBuf, config: InputConfig) -> Self {
        let mut picker_state = ListState::default();
        picker_state.select(Some(0));
        Self {
            config_path,
            config,
            focus: FocusArea::Bindings,
            selected_action: 0,
            selected_slot: 0,
            selected_param: 0,
            overlay_mode: OverlayMode::None,
            picker_mode: PickerMode::Binding,
            picker_state,
            status: format!("Loaded {}", INPUT_CONFIG_PATH),
            dirty: false,
            should_quit: false,
        }
    }

    fn selected_meta(&self) -> ActionRowMeta {
        ACTION_ROWS[self.selected_action]
    }

    fn selected_param_meta(&self) -> ParameterMeta {
        PARAM_ROWS[self.selected_param]
    }

    fn selected_slot_kind(&self) -> BindingSlot {
        BindingSlot::from_index(self.selected_slot)
    }

    fn filtered_pickable_bindings(&self) -> Vec<&'static str> {
        let slot = self.selected_slot_kind();
        PICKABLE_BINDINGS
            .iter()
            .copied()
            .filter(|binding| {
                if slot.accepts_keyboard() {
                    is_keyboard_binding_name(binding)
                } else {
                    is_mouse_binding_name(binding)
                }
            })
            .collect()
    }

    fn selected_axis_target(&self) -> Option<MouseFlightAxisTarget> {
        match self.selected_param {
            0 => Some(self.config.mouse_x_axis.target),
            4 => Some(self.config.mouse_y_axis.target),
            _ => None,
        }
    }

    fn current_binding(&self) -> Option<&str> {
        let action = action_bindings(&self.config.bindings, self.selected_action);
        slot_value(action, self.selected_slot_kind())
    }

    fn set_current_binding(&mut self, value: Option<String>) {
        let slot = self.selected_slot_kind();
        let action = action_bindings_mut(&mut self.config.bindings, self.selected_action);
        set_slot_value(action, slot, value);
        self.dirty = true;
    }

    fn save(&mut self) -> Result<()> {
        let pretty = ron::ser::PrettyConfig::new()
            .depth_limit(4)
            .enumerate_arrays(true)
            .separate_tuple_members(true);
        let serialized = ron::ser::to_string_pretty(&self.config, pretty)?;
        fs::write(&self.config_path, serialized)
            .with_context(|| format!("failed to write {}", self.config_path.display()))?;
        self.dirty = false;
        self.status = format!("Saved {}", self.config_path.display());
        Ok(())
    }

    fn restore_defaults(&mut self) {
        self.config = InputConfig::default();
        self.dirty = true;
        self.status = "Restored default bindings".to_string();
    }

    fn adjust_selected_param(&mut self, delta: f32) {
        match self.selected_param {
            1 => {
                self.config.mouse_x_axis.sensitivity =
                    (self.config.mouse_x_axis.sensitivity + delta).max(0.0)
            }
            2 => {
                self.config.mouse_x_axis.weight =
                    (self.config.mouse_x_axis.weight + delta).clamp(0.0, 5.0)
            }
            5 => {
                self.config.mouse_y_axis.sensitivity =
                    (self.config.mouse_y_axis.sensitivity + delta).max(0.0)
            }
            6 => {
                self.config.mouse_y_axis.weight =
                    (self.config.mouse_y_axis.weight + delta).clamp(0.0, 5.0)
            }
            8 => {
                self.config.keyboard_throttle_weight =
                    (self.config.keyboard_throttle_weight + delta).clamp(0.0, 5.0)
            }
            9 => {
                self.config.keyboard_pitch_weight =
                    (self.config.keyboard_pitch_weight + delta).clamp(0.0, 5.0)
            }
            10 => {
                self.config.keyboard_roll_weight =
                    (self.config.keyboard_roll_weight + delta).clamp(0.0, 5.0)
            }
            11 => self.config.mouse_smoothing = (self.config.mouse_smoothing + delta).max(0.0),
            12 => {}
            3 | 7 => {}
            0 | 4 => {
                self.cycle_selected_axis_target(delta.signum() as isize);
                return;
            }
            13.. => {}
        }
        self.dirty = true;
        self.status = format!("Adjusted {}", self.selected_param_meta().label);
    }

    fn cycle_selected_axis_target(&mut self, delta: isize) {
        let Some(current) = self.selected_axis_target() else {
            return;
        };
        let current_index = PICKABLE_MOUSE_AXIS_TARGETS
            .iter()
            .position(|target| *target == current)
            .unwrap_or(0) as isize;
        let next_index =
            (current_index + delta).rem_euclid(PICKABLE_MOUSE_AXIS_TARGETS.len() as isize) as usize;
        let next_target = PICKABLE_MOUSE_AXIS_TARGETS[next_index];
        match self.selected_param {
            0 => self.config.mouse_x_axis.target = next_target,
            4 => self.config.mouse_y_axis.target = next_target,
            _ => return,
        }
        self.dirty = true;
        self.status = format!("Adjusted {}", self.selected_param_meta().label);
    }

    fn toggle_selected_bool(&mut self) {
        match self.selected_param {
            3 => {
                self.config.mouse_x_axis.invert = !self.config.mouse_x_axis.invert;
                self.dirty = true;
            }
            7 => {
                self.config.mouse_y_axis.invert = !self.config.mouse_y_axis.invert;
                self.dirty = true;
            }
            12 => {
                self.config.capture_mouse_on_start = !self.config.capture_mouse_on_start;
                self.dirty = true;
            }
            _ => return,
        }
        self.status = format!("Adjusted {}", self.selected_param_meta().label);
    }

    fn open_binding_picker(&mut self) {
        self.overlay_mode = OverlayMode::Picker;
        self.picker_mode = PickerMode::Binding;
        let filtered = self.filtered_pickable_bindings();
        let target = self
            .current_binding()
            .and_then(|current| filtered.iter().position(|binding| *binding == current))
            .unwrap_or(0);
        self.picker_state.select(Some(target));
        self.status = "Picker opened".to_string();
    }

    fn open_axis_target_picker(&mut self) {
        self.overlay_mode = OverlayMode::Picker;
        self.picker_mode = PickerMode::MouseAxisTarget;
        let target = self
            .selected_axis_target()
            .and_then(|current| {
                PICKABLE_MOUSE_AXIS_TARGETS
                    .iter()
                    .position(|candidate| *candidate == current)
            })
            .unwrap_or(0);
        self.picker_state.select(Some(target));
        self.status = "Picker opened".to_string();
    }
}

fn main() -> Result<()> {
    let config_path = PathBuf::from(INPUT_CONFIG_PATH);
    let config = load_input_config(&config_path)?;
    let mut terminal = init_terminal()?;
    let result = run_app(&mut terminal, App::new(config_path, config));
    restore_terminal(&mut terminal)?;
    result
}

fn load_input_config(path: &Path) -> Result<InputConfig> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    ron::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: App) -> Result<()> {
    loop {
        terminal.draw(|frame| draw_ui(frame, &mut app))?;
        if app.should_quit {
            break;
        }
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind == KeyEventKind::Press => match app.overlay_mode {
                OverlayMode::None => handle_normal_key(&mut app, key.code)?,
                OverlayMode::Capture => handle_capture_key(&mut app, key.code),
                OverlayMode::Picker => handle_picker_key(&mut app, key.code),
            },
            Event::Mouse(mouse) => match app.overlay_mode {
                OverlayMode::Capture => handle_capture_mouse(&mut app, mouse.kind),
                OverlayMode::Picker => handle_picker_mouse(&mut app, mouse.kind),
                OverlayMode::None => {}
            },
            _ => {}
        }
    }
    Ok(())
}

fn handle_normal_key(app: &mut App, code: KeyCode) -> Result<()> {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Tab => {
            app.focus = match app.focus {
                FocusArea::Bindings => FocusArea::Parameters,
                FocusArea::Parameters => FocusArea::Bindings,
            };
        }
        KeyCode::Up => match app.focus {
            FocusArea::Bindings => app.selected_action = app.selected_action.saturating_sub(1),
            FocusArea::Parameters => app.selected_param = app.selected_param.saturating_sub(1),
        },
        KeyCode::Down => match app.focus {
            FocusArea::Bindings => {
                app.selected_action = (app.selected_action + 1).min(ACTION_ROWS.len() - 1);
            }
            FocusArea::Parameters => {
                app.selected_param = (app.selected_param + 1).min(PARAM_COUNT - 1);
            }
        },
        KeyCode::Left => match app.focus {
            FocusArea::Bindings => app.selected_slot = app.selected_slot.saturating_sub(1),
            FocusArea::Parameters => {
                if matches!(app.selected_param, 3 | 7 | 12) {
                    app.toggle_selected_bool();
                } else {
                    app.adjust_selected_param(-parameter_step(app.selected_param));
                }
            }
        },
        KeyCode::Right => match app.focus {
            FocusArea::Bindings => app.selected_slot = (app.selected_slot + 1).min(SLOT_COUNT - 1),
            FocusArea::Parameters => {
                if matches!(app.selected_param, 3 | 7 | 12) {
                    app.toggle_selected_bool();
                } else {
                    app.adjust_selected_param(parameter_step(app.selected_param));
                }
            }
        },
        KeyCode::Enter => match app.focus {
            FocusArea::Bindings => app.open_binding_picker(),
            FocusArea::Parameters => {
                if matches!(app.selected_param, 0 | 4) {
                    app.open_axis_target_picker();
                } else if matches!(app.selected_param, 3 | 7 | 12) {
                    app.toggle_selected_bool();
                }
            }
        },
        KeyCode::Char(' ') => {
            if app.focus == FocusArea::Bindings {
                app.overlay_mode = OverlayMode::Capture;
                app.status = format!(
                    "Capturing {} {}. Press a key or mouse button, Esc to cancel, Enter for picker",
                    app.selected_meta().label,
                    app.selected_slot_kind().title()
                );
            }
        }
        KeyCode::Char('p') => match app.focus {
            FocusArea::Bindings => app.open_binding_picker(),
            FocusArea::Parameters if matches!(app.selected_param, 0 | 4) => {
                app.open_axis_target_picker();
            }
            FocusArea::Parameters => return Ok(()),
        },
        KeyCode::Backspace | KeyCode::Delete => {
            app.set_current_binding(None);
            app.status = format!(
                "Cleared {} {}",
                app.selected_meta().label,
                app.selected_slot_kind().title()
            );
        }
        KeyCode::Char('s') => app.save()?,
        KeyCode::Char('r') => app.restore_defaults(),
        _ => {}
    }
    Ok(())
}

fn handle_capture_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.overlay_mode = OverlayMode::None;
            app.status = "Capture cancelled".to_string();
        }
        KeyCode::Char('p') => {
            app.open_binding_picker();
        }
        _ => match key_code_to_binding_name(code) {
            Some(name) => {
                if !app.selected_slot_kind().accepts_keyboard() {
                    app.status = format!(
                        "{} only accepts mouse buttons. Press a mouse button or use P.",
                        app.selected_slot_kind().title()
                    );
                    return;
                }
                app.set_current_binding(Some(name.to_string()));
                app.overlay_mode = OverlayMode::None;
                app.status = format!(
                    "Bound {} {} to {}",
                    app.selected_meta().label,
                    app.selected_slot_kind().title(),
                    name
                );
            }
            None => {
                app.status =
                    "This terminal key cannot be captured directly. Press P for the picker."
                        .to_string();
            }
        },
    }
}

fn handle_capture_mouse(app: &mut App, kind: MouseEventKind) {
    let Some(name) = mouse_kind_to_binding_name(kind) else {
        return;
    };
    if !app.selected_slot_kind().accepts_mouse() {
        app.status = format!(
            "{} only accepts keyboard bindings. Press a key or use P.",
            app.selected_slot_kind().title()
        );
        return;
    }
    app.set_current_binding(Some(name.to_string()));
    app.overlay_mode = OverlayMode::None;
    app.status = format!(
        "Bound {} {} to mouse {}",
        app.selected_meta().label,
        app.selected_slot_kind().title(),
        name
    );
}

fn handle_picker_key(app: &mut App, code: KeyCode) {
    let selected = app.picker_state.selected().unwrap_or(0);
    match code {
        KeyCode::Esc => {
            app.overlay_mode = OverlayMode::None;
            app.status = "Picker cancelled".to_string();
        }
        KeyCode::Up => {
            app.picker_state.select(Some(selected.saturating_sub(1)));
        }
        KeyCode::Down => {
            let max_index = match app.picker_mode {
                PickerMode::Binding => app.filtered_pickable_bindings().len().saturating_sub(1),
                PickerMode::MouseAxisTarget => PICKABLE_MOUSE_AXIS_TARGETS.len().saturating_sub(1),
            };
            app.picker_state.select(Some((selected + 1).min(max_index)));
        }
        KeyCode::Enter => match app.picker_mode {
            PickerMode::Binding => {
                let bindings = app.filtered_pickable_bindings();
                let Some(binding) = bindings.get(selected).copied() else {
                    return;
                };
                app.set_current_binding(Some(binding.to_string()));
                app.overlay_mode = OverlayMode::None;
                app.status = format!(
                    "Bound {} {} to {}",
                    app.selected_meta().label,
                    app.selected_slot_kind().title(),
                    binding
                );
            }
            PickerMode::MouseAxisTarget => {
                let Some(target) = PICKABLE_MOUSE_AXIS_TARGETS.get(selected).copied() else {
                    return;
                };
                match app.selected_param {
                    0 => app.config.mouse_x_axis.target = target,
                    4 => app.config.mouse_y_axis.target = target,
                    _ => return,
                }
                app.dirty = true;
                app.overlay_mode = OverlayMode::None;
                app.status = format!(
                    "Set {} to {}",
                    app.selected_param_meta().label,
                    mouse_axis_target_name(target)
                );
            }
        },
        _ => {}
    }
}

fn handle_picker_mouse(app: &mut App, kind: MouseEventKind) {
    if let Some(name) = mouse_kind_to_binding_name(kind) {
        app.set_current_binding(Some(name.to_string()));
        app.overlay_mode = OverlayMode::None;
        app.status = format!(
            "Bound {} {} to mouse {}",
            app.selected_meta().label,
            app.selected_slot_kind().title(),
            name
        );
    }
}

fn draw_ui(frame: &mut ratatui::Frame<'_>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(10),
            Constraint::Length(6),
        ])
        .split(frame.area());

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "Input Binding Configurator",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(if app.dirty { "[unsaved]" } else { "[saved]" }),
    ]))
    .block(Block::default().borders(Borders::ALL).title("Status"));
    frame.render_widget(title, chunks[0]);

    let header = Row::new(["Action", "K1", "K2", "M1", "M2"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let selected_slot = app.selected_slot_kind();
    let rows = ACTION_ROWS.iter().enumerate().map(|(row_index, meta)| {
        let bindings = action_bindings(&app.config.bindings, row_index);
        let values = [
            slot_value(bindings, BindingSlot::KeyboardPrimary).unwrap_or("-"),
            slot_value(bindings, BindingSlot::KeyboardSecondary).unwrap_or("-"),
            slot_value(bindings, BindingSlot::MousePrimary).unwrap_or("-"),
            slot_value(bindings, BindingSlot::MouseSecondary).unwrap_or("-"),
        ];
        let mut cells = Vec::with_capacity(5);
        cells.push(Cell::from(meta.label));
        for (slot_index, value) in values.into_iter().enumerate() {
            let slot = BindingSlot::from_index(slot_index);
            let mut style = binding_style(&app.config.bindings, row_index, slot, value);
            if row_index == app.selected_action && slot == selected_slot {
                style = style
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD);
            }
            cells.push(Cell::from(value).style(style));
        }
        let row_style = if row_index == app.selected_action && app.focus == FocusArea::Bindings {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        Row::new(cells).style(row_style)
    });
    let bindings_title = if app.focus == FocusArea::Bindings {
        "Bindings [active]"
    } else {
        "Bindings"
    };
    let table = Table::new(
        rows,
        [
            Constraint::Length(22),
            Constraint::Length(18),
            Constraint::Length(18),
            Constraint::Length(14),
            Constraint::Length(14),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(bindings_title))
    .column_spacing(1);
    frame.render_widget(table, chunks[1]);

    let params_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[2]);

    let param_rows = PARAM_ROWS.iter().enumerate().map(|(index, meta)| {
        let value = parameter_value_string(&app.config, index);
        let style = if index == app.selected_param && app.focus == FocusArea::Parameters {
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        Row::new(vec![Cell::from(meta.label), Cell::from(value)]).style(style)
    });
    let params_title = if app.focus == FocusArea::Parameters {
        "Parameters [active]"
    } else {
        "Parameters"
    };
    let params_table = Table::new(
        param_rows,
        [Constraint::Percentage(65), Constraint::Percentage(35)],
    )
    .block(Block::default().borders(Borders::ALL).title(params_title))
    .column_spacing(1);
    frame.render_widget(params_table, params_chunks[0]);

    let conflict_lines = selected_conflict_lines(app);
    let conflicts = Paragraph::new(conflict_lines)
        .wrap(Wrap { trim: true })
        .block(Block::default().borders(Borders::ALL).title("Conflicts"));
    frame.render_widget(conflicts, params_chunks[1]);

    let help = Paragraph::new(vec![
        Line::from("Tab switches Bindings/Parameters. In Bindings, Space captures and Enter opens the picker."),
        Line::from("Arrow keys move selection; in Parameters, Left/Right adjusts values. Enter opens target pickers or toggles booleans."),
        Line::from("Backspace/Delete clears the selected binding slot. S saves. R restores defaults. Q quits."),
        Line::from(app.status.as_str()),
    ])
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Help"));
    frame.render_widget(help, chunks[3]);

    match app.overlay_mode {
        OverlayMode::Capture => draw_capture_overlay(frame),
        OverlayMode::Picker => draw_picker_overlay(frame, app),
        OverlayMode::None => {}
    }
}

fn draw_capture_overlay(frame: &mut ratatui::Frame<'_>) {
    let area = centered_rect(70, 24, frame.area());
    frame.render_widget(Clear, area);
    let widget = Paragraph::new(vec![
        Line::from("Waiting for input"),
        Line::from(""),
        Line::from("Press any supported key, or click a mouse button."),
        Line::from("Esc cancels. Enter opens the full picker for special bindings."),
    ])
    .wrap(Wrap { trim: true })
    .block(Block::default().borders(Borders::ALL).title("Capture"));
    frame.render_widget(widget, area);
}

fn draw_picker_overlay(frame: &mut ratatui::Frame<'_>, app: &mut App) {
    let area = centered_rect(50, 70, frame.area());
    frame.render_widget(Clear, area);
    let (title, items) = match app.picker_mode {
        PickerMode::Binding => (
            "Binding Picker",
            app.filtered_pickable_bindings()
                .into_iter()
                .map(ListItem::new)
                .collect::<Vec<_>>(),
        ),
        PickerMode::MouseAxisTarget => (
            "Mouse Axis Target",
            PICKABLE_MOUSE_AXIS_TARGETS
                .iter()
                .copied()
                .map(mouse_axis_target_name)
                .map(ListItem::new)
                .collect::<Vec<_>>(),
        ),
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_style(
            Style::default()
                .bg(Color::Blue)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");
    frame.render_stateful_widget(list, area, &mut app.picker_state);
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn parameter_step(index: usize) -> f32 {
    match index {
        1 | 5 => 0.05,
        2 | 6 | 8 | 9 | 10 => 0.05,
        11 => 1.0,
        _ => 0.0,
    }
}

fn parameter_value_string(config: &InputConfig, index: usize) -> String {
    match index {
        0 => mouse_axis_target_name(config.mouse_x_axis.target).to_string(),
        1 => format!("{:.2}", config.mouse_x_axis.sensitivity),
        2 => format!("{:.2}", config.mouse_x_axis.weight),
        3 => config.mouse_x_axis.invert.to_string(),
        4 => mouse_axis_target_name(config.mouse_y_axis.target).to_string(),
        5 => format!("{:.2}", config.mouse_y_axis.sensitivity),
        6 => format!("{:.2}", config.mouse_y_axis.weight),
        7 => config.mouse_y_axis.invert.to_string(),
        8 => format!("{:.2}", config.keyboard_throttle_weight),
        9 => format!("{:.2}", config.keyboard_pitch_weight),
        10 => format!("{:.2}", config.keyboard_roll_weight),
        11 => format!("{:.1}", config.mouse_smoothing),
        12 => {
            if config.capture_mouse_on_start {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        _ => "-".to_string(),
    }
}

fn selected_conflict_lines(app: &App) -> Vec<Line<'static>> {
    if app.focus != FocusArea::Bindings {
        return vec![
            Line::from("Parameters do not participate in binding conflicts."),
            Line::from("Use Left/Right to adjust the selected value."),
        ];
    }

    let Some(value) = app.current_binding() else {
        return vec![Line::from("Selected slot is empty.")];
    };
    let conflicts = conflicts_for_binding(
        &app.config.bindings,
        app.selected_action,
        app.selected_slot_kind(),
        value,
    );
    if conflicts.is_empty() {
        vec![
            Line::from(format!("{} has no conflicts.", value)),
            Line::from("Yellow slots in the table indicate bindings reused elsewhere."),
        ]
    } else {
        let mut lines = vec![Line::from(format!("{} is also used by:", value))];
        lines.extend(conflicts.into_iter().map(Line::from));
        lines
    }
}

fn conflicts_for_binding(
    bindings: &InputBindingsConfig,
    selected_action_index: usize,
    selected_slot: BindingSlot,
    value: &str,
) -> Vec<String> {
    let mut conflicts = Vec::new();
    for action_index in 0..ACTION_ROWS.len() {
        let action = action_bindings(bindings, action_index);
        for slot_index in 0..SLOT_COUNT {
            let slot = BindingSlot::from_index(slot_index);
            if action_index == selected_action_index && slot == selected_slot {
                continue;
            }
            if slot_value(action, slot) == Some(value) {
                conflicts.push(format!(
                    "{} {}",
                    ACTION_ROWS[action_index].label,
                    slot.title()
                ));
            }
        }
    }
    conflicts
}

fn binding_style(
    bindings: &InputBindingsConfig,
    selected_action_index: usize,
    selected_slot: BindingSlot,
    value: &str,
) -> Style {
    if value == "-" {
        return Style::default().fg(Color::DarkGray);
    }
    if !conflicts_for_binding(bindings, selected_action_index, selected_slot, value).is_empty() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    }
}

fn action_bindings(bindings: &InputBindingsConfig, index: usize) -> &ActionBindingsConfig {
    match ACTION_ROWS[index].name {
        "throttle_up" => &bindings.throttle_up,
        "throttle_down" => &bindings.throttle_down,
        "brake" => &bindings.brake,
        "pitch_positive" => &bindings.pitch_positive,
        "pitch_negative" => &bindings.pitch_negative,
        "roll_positive" => &bindings.roll_positive,
        "roll_negative" => &bindings.roll_negative,
        "yaw_positive" => &bindings.yaw_positive,
        "yaw_negative" => &bindings.yaw_negative,
        "fire_gun" => &bindings.fire_gun,
        "repair_aircraft" => &bindings.repair_aircraft,
        "toggle_controls_guide" => &bindings.toggle_controls_guide,
        "reset_match" => &bindings.reset_match,
        "rear_view" => &bindings.rear_view,
        "toggle_local_pilot_mode" => &bindings.toggle_local_pilot_mode,
        "toggle_audio_mute" => &bindings.toggle_audio_mute,
        "toggle_mouse_capture" => &bindings.toggle_mouse_capture,
        _ => unreachable!(),
    }
}

fn action_bindings_mut(
    bindings: &mut InputBindingsConfig,
    index: usize,
) -> &mut ActionBindingsConfig {
    match ACTION_ROWS[index].name {
        "throttle_up" => &mut bindings.throttle_up,
        "throttle_down" => &mut bindings.throttle_down,
        "brake" => &mut bindings.brake,
        "pitch_positive" => &mut bindings.pitch_positive,
        "pitch_negative" => &mut bindings.pitch_negative,
        "roll_positive" => &mut bindings.roll_positive,
        "roll_negative" => &mut bindings.roll_negative,
        "yaw_positive" => &mut bindings.yaw_positive,
        "yaw_negative" => &mut bindings.yaw_negative,
        "fire_gun" => &mut bindings.fire_gun,
        "repair_aircraft" => &mut bindings.repair_aircraft,
        "toggle_controls_guide" => &mut bindings.toggle_controls_guide,
        "reset_match" => &mut bindings.reset_match,
        "rear_view" => &mut bindings.rear_view,
        "toggle_local_pilot_mode" => &mut bindings.toggle_local_pilot_mode,
        "toggle_audio_mute" => &mut bindings.toggle_audio_mute,
        "toggle_mouse_capture" => &mut bindings.toggle_mouse_capture,
        _ => unreachable!(),
    }
}

fn slot_value(bindings: &ActionBindingsConfig, slot: BindingSlot) -> Option<&str> {
    match slot {
        BindingSlot::KeyboardPrimary => bindings.keyboard_primary.as_deref(),
        BindingSlot::KeyboardSecondary => bindings.keyboard_secondary.as_deref(),
        BindingSlot::MousePrimary => bindings.mouse_primary.as_deref(),
        BindingSlot::MouseSecondary => bindings.mouse_secondary.as_deref(),
    }
}

fn set_slot_value(bindings: &mut ActionBindingsConfig, slot: BindingSlot, value: Option<String>) {
    match slot {
        BindingSlot::KeyboardPrimary => bindings.keyboard_primary = value,
        BindingSlot::KeyboardSecondary => bindings.keyboard_secondary = value,
        BindingSlot::MousePrimary => bindings.mouse_primary = value,
        BindingSlot::MouseSecondary => bindings.mouse_secondary = value,
    }
}

fn key_code_to_binding_name(code: KeyCode) -> Option<&'static str> {
    match code {
        KeyCode::Backspace => Some("Backspace"),
        KeyCode::Enter => Some("Enter"),
        KeyCode::Left => Some("ArrowLeft"),
        KeyCode::Right => Some("ArrowRight"),
        KeyCode::Up => Some("ArrowUp"),
        KeyCode::Down => Some("ArrowDown"),
        KeyCode::Esc => Some("Escape"),
        KeyCode::Tab => Some("Tab"),
        KeyCode::Char(' ') => Some("Space"),
        KeyCode::Char('0') => Some("Digit0"),
        KeyCode::Char('1') => Some("Digit1"),
        KeyCode::Char('2') => Some("Digit2"),
        KeyCode::Char('3') => Some("Digit3"),
        KeyCode::Char('4') => Some("Digit4"),
        KeyCode::Char('5') => Some("Digit5"),
        KeyCode::Char('6') => Some("Digit6"),
        KeyCode::Char('7') => Some("Digit7"),
        KeyCode::Char('8') => Some("Digit8"),
        KeyCode::Char('9') => Some("Digit9"),
        KeyCode::Char('a') | KeyCode::Char('A') => Some("KeyA"),
        KeyCode::Char('b') | KeyCode::Char('B') => Some("KeyB"),
        KeyCode::Char('c') | KeyCode::Char('C') => Some("KeyC"),
        KeyCode::Char('d') | KeyCode::Char('D') => Some("KeyD"),
        KeyCode::Char('e') | KeyCode::Char('E') => Some("KeyE"),
        KeyCode::Char('f') | KeyCode::Char('F') => Some("KeyF"),
        KeyCode::Char('g') | KeyCode::Char('G') => Some("KeyG"),
        KeyCode::Char('h') | KeyCode::Char('H') => Some("KeyH"),
        KeyCode::Char('i') | KeyCode::Char('I') => Some("KeyI"),
        KeyCode::Char('j') | KeyCode::Char('J') => Some("KeyJ"),
        KeyCode::Char('k') | KeyCode::Char('K') => Some("KeyK"),
        KeyCode::Char('l') | KeyCode::Char('L') => Some("KeyL"),
        KeyCode::Char('m') | KeyCode::Char('M') => Some("KeyM"),
        KeyCode::Char('n') | KeyCode::Char('N') => Some("KeyN"),
        KeyCode::Char('o') | KeyCode::Char('O') => Some("KeyO"),
        KeyCode::Char('p') | KeyCode::Char('P') => Some("KeyP"),
        KeyCode::Char('q') | KeyCode::Char('Q') => Some("KeyQ"),
        KeyCode::Char('r') | KeyCode::Char('R') => Some("KeyR"),
        KeyCode::Char('s') | KeyCode::Char('S') => Some("KeyS"),
        KeyCode::Char('t') | KeyCode::Char('T') => Some("KeyT"),
        KeyCode::Char('u') | KeyCode::Char('U') => Some("KeyU"),
        KeyCode::Char('v') | KeyCode::Char('V') => Some("KeyV"),
        KeyCode::Char('w') | KeyCode::Char('W') => Some("KeyW"),
        KeyCode::Char('x') | KeyCode::Char('X') => Some("KeyX"),
        KeyCode::Char('y') | KeyCode::Char('Y') => Some("KeyY"),
        KeyCode::Char('z') | KeyCode::Char('Z') => Some("KeyZ"),
        KeyCode::F(1) => Some("F1"),
        KeyCode::F(2) => Some("F2"),
        KeyCode::F(3) => Some("F3"),
        KeyCode::F(4) => Some("F4"),
        KeyCode::F(5) => Some("F5"),
        KeyCode::F(6) => Some("F6"),
        KeyCode::F(7) => Some("F7"),
        KeyCode::F(8) => Some("F8"),
        KeyCode::F(9) => Some("F9"),
        KeyCode::F(10) => Some("F10"),
        KeyCode::F(11) => Some("F11"),
        KeyCode::F(12) => Some("F12"),
        _ => None,
    }
}

fn mouse_kind_to_binding_name(kind: MouseEventKind) -> Option<&'static str> {
    match kind {
        MouseEventKind::Down(MouseButton::Left) => Some("Left"),
        MouseEventKind::Down(MouseButton::Right) => Some("Right"),
        MouseEventKind::Down(MouseButton::Middle) => Some("Middle"),
        _ => None,
    }
}

fn is_keyboard_binding_name(name: &str) -> bool {
    !is_mouse_binding_name(name)
}

fn is_mouse_binding_name(name: &str) -> bool {
    matches!(name, "Left" | "Right" | "Middle" | "Back" | "Forward")
}

fn mouse_axis_target_name(target: MouseFlightAxisTarget) -> &'static str {
    match target {
        MouseFlightAxisTarget::Pitch => "Pitch",
        MouseFlightAxisTarget::Roll => "Roll",
        MouseFlightAxisTarget::Yaw => "Yaw",
    }
}
