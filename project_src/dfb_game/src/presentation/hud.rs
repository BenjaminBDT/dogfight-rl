use bevy::input::ButtonInput;
use bevy::prelude::*;

use crate::app::{AppMode, ObserverClientMode};
use crate::bridge::protocol::BridgeControlSlot;
use crate::bridge::transport::IpcBridgeState;
use crate::bridge::{
    AssignedControlRole, BridgeClientConnectionStatus, BridgeClientHandshakeState,
    BridgeClientInbox, BridgeJitterDiagnostics, BridgeLinkState, BridgeTimingState,
    LocalPredictionState,
};
use crate::core::config::{ActionBindingsConfig, MouseFlightAxisTarget, RepositoryConfig};
use crate::gameplay::damage::AircraftDamageState;
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::input::actions::ControlBindings;
use crate::presentation::camera::MainViewCamera;
use crate::presentation::hud_table::{Cell, render_rows_with_min_widths};
use crate::simulation::components::{AircraftPerformance, AircraftRole, AircraftState, GunState};

const COMPASS_TICK_SPACING: f32 = 24.0;
const COMPASS_TICK_WIDTH: f32 = 20.0;
const COMPASS_VISIBLE_TICKS: i32 = 8;
const COMPASS_WIDTH: f32 = 440.0;
const TAPE_GAUGE_HEIGHT: f32 = 220.0;
const SPEED_TAPE_TICK_SPACING: f32 = 44.0;
const ALTITUDE_TAPE_TICK_SPACING: f32 = 44.0;
const THROTTLE_TAPE_TICK_SPACING: f32 = 22.0;
const STALL_TAPE_TICK_SPACING: f32 = 22.0;
const SPEED_TAPE_TICK_STEP: f32 = 10.0;
const SPEED_TAPE_VISIBLE_TICKS: i32 = 5;
const TAPE_GAUGE_PANEL_WIDTH: f32 = 72.0;
const TAPE_GAUGE_PANEL_HEIGHT: f32 = 280.0;
const TAPE_GAUGE_PANEL_TOP_OFFSET: f32 = -140.0;
const TAPE_GAUGE_WINDOW_LEFT: f32 = 8.0;
const TAPE_GAUGE_WINDOW_TOP: f32 = 44.0;
const TAPE_GAUGE_WINDOW_WIDTH: f32 = 56.0;
const TAPE_GAUGE_LEFT_OUTER_X: f32 = 8.0;
const TAPE_GAUGE_LEFT_INNER_X: f32 = 88.0;
const TAPE_GAUGE_RIGHT_INNER_X: f32 = 88.0;
const TAPE_GAUGE_RIGHT_OUTER_X: f32 = 8.0;
const VERTICAL_TAPE_VISIBLE_TICKS: i32 = 5;
const ALTITUDE_TAPE_TICK_STEP: f32 = 100.0;
const THROTTLE_TAPE_TICK_STEP: f32 = 0.1;
const STALL_TAPE_TICK_STEP: f32 = 0.1;
const ALTITUDE_WARNING_RANGE_FALLBACK: f32 = 200.0;
const MATCH_PANEL_WIDTH: f32 = 300.0;
const NETWORK_PANEL_WIDTH: f32 = 300.0;
const STATUS_PANEL_WIDTH: f32 = 320.0;
const DAMAGE_INDICATOR_RADIUS: f32 = 118.0;
const DAMAGE_INDICATOR_LIFETIME: f32 = 1.15;
const DAMAGE_INDICATOR_SLOTS: usize = 6;
const DAMAGE_INDICATOR_MERGE_ANGLE_RADIANS: f32 = 24.0_f32.to_radians();

#[derive(Resource, Clone, Default)]
pub struct UiFontHandles {
    pub text: Handle<Font>,
    pub table: Handle<Font>,
    pub symbols: Handle<Font>,
}

#[derive(Resource, Default)]
pub struct ControlsGuideState {
    pub visible: bool,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct ObservedAircraftRole(pub AircraftRole);

impl Default for ObservedAircraftRole {
    fn default() -> Self {
        Self(AircraftRole::Fighter1)
    }
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct CompassReferenceState {
    pub use_camera_forward: bool,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct SpeedGaugeState {
    pub display_center_speed: f32,
    pub display_center_altitude: f32,
}

impl Default for SpeedGaugeState {
    fn default() -> Self {
        Self {
            display_center_speed: 80.0,
            display_center_altitude: 500.0,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObserverSessionKind {
    #[default]
    PlayerLocal,
    PlayerLive,
    SpectatorLive,
    SpectatorReplay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ObserverDataSourceKind {
    #[default]
    LocalSimulation,
    LiveServer,
    RecordedEpisode,
}

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObserverViewContext {
    pub session: ObserverSessionKind,
    pub source: ObserverDataSourceKind,
}

impl Default for ObserverViewContext {
    fn default() -> Self {
        Self {
            session: ObserverSessionKind::PlayerLocal,
            source: ObserverDataSourceKind::LocalSimulation,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct ObserverFeedStatus {
    pub phase: Option<MatchPhase>,
    pub sim_time_seconds: f32,
    pub current_tick: Option<u64>,
    pub current_step: Option<usize>,
    pub total_steps: Option<usize>,
    pub playback_finished: Option<bool>,
    pub playback_paused: Option<bool>,
    pub playback_speed: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserverFeedProviderKind {
    LiveBridge,
    RecordedTimeline,
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct ObserverFeedProviderState {
    pub provider: Option<ObserverFeedProviderKind>,
    pub ready: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct DamageIndicatorEvent {
    pub source_position: Vec3,
    pub intensity: f32,
}

#[derive(Resource, Default)]
pub struct DamageIndicatorQueue {
    pub events: Vec<DamageIndicatorEvent>,
}

impl DamageIndicatorQueue {
    pub fn push(&mut self, source_position: Vec3, intensity: f32) {
        self.events.push(DamageIndicatorEvent {
            source_position,
            intensity,
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveDamageIndicator {
    angle_radians: f32,
    ttl_seconds: f32,
    intensity: f32,
}

#[derive(Resource, Default)]
pub struct DamageIndicatorState {
    indicators: Vec<ActiveDamageIndicator>,
}

#[derive(Component)]
pub struct HudLayer;

#[derive(Component)]
pub struct MatchHudText;

#[derive(Component)]
pub struct EnemyHudText;

#[derive(Component)]
pub struct AttitudeText;

#[derive(Component)]
pub struct NetworkHudText;

#[derive(Component)]
pub struct AircraftStatusText;

#[derive(Component)]
pub struct ControlsGuideRoot;

#[derive(Component)]
pub struct ControlsGuideText;

#[derive(Component)]
pub struct ReplayProgressHudRoot;

#[derive(Component)]
pub struct ReplayProgressHudText;

#[derive(Component)]
pub struct ReplayProgressBarFill;

#[derive(Component)]
pub struct ReplayProgressBarText;

#[derive(Component)]
pub struct HeatGaugeFill;

#[derive(Component)]
pub struct HeatGaugeText;

#[derive(Component)]
pub struct HpGaugeFill;

#[derive(Component)]
pub struct HpGaugeText;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum VerticalTapeKind {
    Altitude,
    Throttle,
    Stall,
}

fn vertical_tape_tick_spacing(kind: VerticalTapeKind) -> f32 {
    match kind {
        VerticalTapeKind::Altitude => ALTITUDE_TAPE_TICK_SPACING,
        VerticalTapeKind::Throttle => THROTTLE_TAPE_TICK_SPACING,
        VerticalTapeKind::Stall => STALL_TAPE_TICK_SPACING,
    }
}

fn altitude_warning_range(performance: &AircraftPerformance) -> f32 {
    let pitch_rate = performance.pitch_positive_rate_limit_deg.to_radians();
    if performance.max_level_speed.is_finite() && pitch_rate.is_finite() && pitch_rate > 0.01 {
        (performance.max_level_speed.max(1.0) / pitch_rate * 2.0).max(1.0)
    } else {
        ALTITUDE_WARNING_RANGE_FALLBACK
    }
}

#[derive(Component, Clone, Copy)]
pub struct VerticalTapeCurrentText {
    pub kind: VerticalTapeKind,
}

#[derive(Component, Clone, Copy)]
pub struct VerticalTapePointer {
    pub kind: VerticalTapeKind,
}

#[derive(Component, Clone, Copy)]
pub struct VerticalTapeTickMark {
    pub kind: VerticalTapeKind,
    pub offset: i32,
}

#[derive(Component, Clone, Copy)]
pub struct VerticalTapeTickLabel {
    pub kind: VerticalTapeKind,
    pub offset: i32,
}

#[derive(Component)]
pub struct SpeedGaugeCurrentText;

#[derive(Component)]
pub struct SpeedGaugePointer;

#[derive(Component, Clone, Copy)]
pub struct SpeedGaugeTick {
    pub offset: i32,
}

#[derive(Component, Clone, Copy)]
pub struct SpeedGaugeTickLabel {
    pub offset: i32,
}

#[derive(Component)]
pub struct OutOfBoundsOverlay;

#[derive(Component)]
pub struct OutOfBoundsWarningText;

#[derive(Component, Clone, Copy)]
pub struct CompassTick {
    pub index: i32,
}

#[derive(Component, Clone, Copy)]
pub struct CompassTickLabel {
    pub index: i32,
}

#[derive(Component, Clone, Copy)]
pub struct DamageIndicatorSlot {
    pub index: usize,
}

pub fn load_ui_fonts(asset_server: Res<AssetServer>, mut fonts: ResMut<UiFontHandles>) {
    let iosevka = asset_server.load("fonts/IosevkaNerdFont-Regular.ttf");
    fonts.text = iosevka.clone();
    fonts.table = iosevka.clone();
    fonts.symbols = iosevka;
}

pub fn spawn_hud(mut commands: Commands, fonts: Res<UiFontHandles>) {
    spawn_match_panel(&mut commands, &fonts);
    spawn_network_panel(&mut commands, &fonts);
    spawn_compass(&mut commands, &fonts);
    spawn_altitude_gauge(&mut commands, &fonts);
    spawn_speed_gauge(&mut commands, &fonts);
    spawn_stall_gauge(&mut commands, &fonts);
    spawn_thrust_gauge(&mut commands, &fonts);
    spawn_attitude_panel(&mut commands, &fonts);
    spawn_aircraft_status_panel(&mut commands, &fonts);
    spawn_damage_indicator_ring(&mut commands, &fonts);
    spawn_controls_guide(&mut commands, &fonts);
    spawn_out_of_bounds_overlay(&mut commands, &fonts);
}

fn ui_text_font(fonts: &UiFontHandles, font_size: f32) -> TextFont {
    TextFont {
        font: fonts.text.clone(),
        font_size,
        ..default()
    }
}

fn ui_table_font(fonts: &UiFontHandles, font_size: f32) -> TextFont {
    TextFont {
        font: fonts.table.clone(),
        font_size,
        ..default()
    }
}

fn ui_symbol_font(fonts: &UiFontHandles, font_size: f32) -> TextFont {
    TextFont {
        font: fonts.symbols.clone(),
        font_size,
        ..default()
    }
}

fn spawn_match_panel(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn(hud_panel(
            Val::Px(16.0),
            Val::Px(16.0),
            Val::Auto,
            Val::Auto,
            MATCH_PANEL_WIDTH,
        ))
        .insert(HudLayer)
        .with_children(|parent| {
            parent.spawn((
                Text::new("MATCH"),
                ui_table_font(fonts, 12.0),
                TextColor(Color::srgb(0.86, 0.93, 0.97)),
                MatchHudText,
            ));
        });
}

fn spawn_network_panel(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn(hud_panel(
            Val::Px(16.0),
            Val::Auto,
            Val::Px(16.0),
            Val::Auto,
            NETWORK_PANEL_WIDTH,
        ))
        .insert(HudLayer)
        .with_children(|parent| {
            parent.spawn((
                Text::new("NETWORK"),
                ui_table_font(fonts, 12.0),
                TextColor(Color::srgb(0.86, 0.93, 0.97)),
                NetworkHudText,
            ));
        });
}

fn spawn_compass(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: px(12.0),
                left: percent(50.0),
                margin: UiRect::left(px(-(COMPASS_WIDTH * 0.5))),
                width: px(COMPASS_WIDTH),
                height: px(60.0),
                padding: UiRect::axes(px(10.0), px(8.0)),
                border: UiRect::all(px(1.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.09, 0.12, 0.74)),
            BorderColor::all(Color::srgba(0.74, 0.82, 0.90, 0.35)),
            HudLayer,
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    top: px(3.0),
                    left: percent(50.0),
                    margin: UiRect::left(px(-1.0)),
                    width: px(2.0),
                    height: px(48.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.98, 0.84, 0.42)),
            ));

            for index in -COMPASS_VISIBLE_TICKS..=COMPASS_VISIBLE_TICKS {
                parent
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            top: px(5.0),
                            left: px(0.0),
                            width: px(COMPASS_TICK_WIDTH),
                            height: px(48.0),
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::SpaceBetween,
                            ..default()
                        },
                        Visibility::Inherited,
                        CompassTick { index },
                    ))
                    .with_children(|tick| {
                        tick.spawn((
                            Text::new(""),
                            ui_table_font(fonts, 9.0),
                            TextColor(Color::srgb(0.88, 0.94, 0.97)),
                            CompassTickLabel { index },
                        ));
                        tick.spawn((
                            Node {
                                width: px(2.0),
                                height: px(16.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.88, 0.94, 0.97)),
                        ));
                    });
            }
        });
}

fn spawn_attitude_panel(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn(hud_panel(
            Val::Auto,
            Val::Px(16.0),
            Val::Auto,
            Val::Px(16.0),
            STATUS_PANEL_WIDTH,
        ))
        .insert(HudLayer)
        .with_children(|parent| {
            parent.spawn((
                Text::new("POSE"),
                ui_table_font(fonts, 11.0),
                TextColor(Color::srgb(0.88, 0.94, 0.97)),
                AttitudeText,
            ));
        });
}

fn spawn_altitude_gauge(commands: &mut Commands, fonts: &UiFontHandles) {
    spawn_vertical_tape_gauge(
        commands,
        fonts,
        VerticalTapeKind::Altitude,
        "ALT",
        TAPE_GAUGE_LEFT_OUTER_X,
        true,
        Color::srgb(0.88, 0.94, 0.97),
        Color::srgba(0.70, 0.82, 0.90, 0.35),
        Color::srgb(0.54, 0.80, 0.97),
    );
}

fn spawn_thrust_gauge(commands: &mut Commands, fonts: &UiFontHandles) {
    spawn_vertical_tape_gauge(
        commands,
        fonts,
        VerticalTapeKind::Throttle,
        "THR",
        TAPE_GAUGE_RIGHT_OUTER_X,
        false,
        Color::srgb(0.97, 0.90, 0.80),
        Color::srgba(0.97, 0.78, 0.42, 0.36),
        Color::srgb(0.98, 0.78, 0.42),
    );
}

fn gauge_panel_node(horizontal: Node) -> Node {
    Node {
        position_type: PositionType::Absolute,
        top: percent(50.0),
        margin: UiRect::top(px(TAPE_GAUGE_PANEL_TOP_OFFSET)),
        width: px(TAPE_GAUGE_PANEL_WIDTH),
        height: px(TAPE_GAUGE_PANEL_HEIGHT),
        padding: UiRect::all(px(6.0)),
        border: UiRect::all(px(1.0)),
        ..horizontal
    }
}

fn gauge_tape_window_node(height: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: px(TAPE_GAUGE_WINDOW_LEFT),
        top: px(TAPE_GAUGE_WINDOW_TOP),
        width: px(TAPE_GAUGE_WINDOW_WIDTH),
        height: px(height),
        ..default()
    }
}

fn spawn_speed_gauge(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            gauge_panel_node(Node {
                left: px(TAPE_GAUGE_LEFT_INNER_X),
                ..default()
            }),
            BackgroundColor(Color::srgba(0.04, 0.08, 0.11, 0.72)),
            BorderColor::all(Color::srgba(0.60, 0.88, 0.78, 0.40)),
            HudLayer,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("SPD"),
                ui_text_font(fonts, 10.0),
                TextColor(Color::srgb(0.74, 0.96, 0.44)),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(20.0),
                    top: px(6.0),
                    ..default()
                },
            ));
            parent.spawn((
                Text::new("0"),
                ui_text_font(fonts, 10.0),
                TextColor(Color::srgb(0.74, 0.96, 0.44)),
                Node {
                    position_type: PositionType::Absolute,
                    right: px(10.0),
                    top: px(20.0),
                    ..default()
                },
                SpeedGaugeCurrentText,
            ));
            parent
                .spawn(gauge_tape_window_node(TAPE_GAUGE_HEIGHT))
                .with_children(|tape| {
                    tape.spawn((
                        Text::new("▶"),
                        ui_symbol_font(fonts, 12.0),
                        TextColor(Color::srgb(0.92, 1.0, 0.54)),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.0),
                            top: px((TAPE_GAUGE_HEIGHT * 0.5) - 8.0),
                            ..default()
                        },
                        SpeedGaugePointer,
                    ));
                    for offset in -SPEED_TAPE_VISIBLE_TICKS..=SPEED_TAPE_VISIBLE_TICKS {
                        tape.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                right: px(0.0),
                                top: px(0.0),
                                width: px(44.0),
                                height: px(16.0),
                                ..default()
                            },
                            Visibility::Inherited,
                            SpeedGaugeTick { offset },
                        ))
                        .with_children(|tick| {
                            tick.spawn((
                                Node {
                                    position_type: PositionType::Absolute,
                                    right: px(0.0),
                                    top: px(7.0),
                                    width: px(12.0),
                                    height: px(2.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.66, 0.92, 0.34)),
                            ));
                            tick.spawn((
                                Text::new(""),
                                ui_table_font(fonts, 9.0),
                                TextColor(Color::srgb(0.74, 0.96, 0.44)),
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(0.0),
                                    top: px(0.0),
                                    ..default()
                                },
                                SpeedGaugeTickLabel { offset },
                            ));
                        });
                    }
                });
        });
}

fn spawn_vertical_tape_gauge(
    commands: &mut Commands,
    fonts: &UiFontHandles,
    kind: VerticalTapeKind,
    title: &str,
    horizontal_margin: f32,
    align_left: bool,
    text_color: Color,
    _frame_border: Color,
    accent_color: Color,
) {
    let horizontal = if align_left {
        Node {
            left: px(horizontal_margin),
            ..default()
        }
    } else {
        Node {
            right: px(horizontal_margin),
            ..default()
        }
    };
    commands
        .spawn((
            gauge_panel_node(horizontal),
            BackgroundColor(Color::srgba(0.04, 0.08, 0.11, 0.72)),
            HudLayer,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(title),
                ui_text_font(fonts, 10.0),
                TextColor(text_color),
                Node {
                    position_type: PositionType::Absolute,
                    left: px(18.0),
                    top: px(6.0),
                    ..default()
                },
            ));
            parent.spawn((
                Text::new(""),
                ui_text_font(fonts, 10.0),
                TextColor(text_color),
                Node {
                    position_type: PositionType::Absolute,
                    right: px(8.0),
                    top: px(20.0),
                    ..default()
                },
                VerticalTapeCurrentText { kind },
            ));
            parent
                .spawn((
                    gauge_tape_window_node(TAPE_GAUGE_HEIGHT),
                    BackgroundColor(Color::NONE),
                ))
                .with_children(|tape| {
                    tape.spawn((
                        Text::new("▶"),
                        ui_symbol_font(fonts, 12.0),
                        TextColor(accent_color),
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.0),
                            top: px((TAPE_GAUGE_HEIGHT * 0.5) - 8.0),
                            ..default()
                        },
                        VerticalTapePointer { kind },
                    ));
                    for offset in -VERTICAL_TAPE_VISIBLE_TICKS..=VERTICAL_TAPE_VISIBLE_TICKS {
                        tape.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                right: px(0.0),
                                top: px(0.0),
                                width: px(12.0),
                                height: px(2.0),
                                ..default()
                            },
                            BackgroundColor(accent_color),
                            Visibility::Inherited,
                            VerticalTapeTickMark { kind, offset },
                        ));
                        tape.spawn((
                            Text::new(""),
                            ui_table_font(fonts, 9.0),
                            TextColor(text_color),
                            Node {
                                position_type: PositionType::Absolute,
                                left: px(0.0),
                                top: px(0.0),
                                ..default()
                            },
                            Visibility::Inherited,
                            VerticalTapeTickLabel { kind, offset },
                        ));
                    }
                });
        });
}

fn spawn_stall_gauge(commands: &mut Commands, fonts: &UiFontHandles) {
    spawn_vertical_tape_gauge(
        commands,
        fonts,
        VerticalTapeKind::Stall,
        "STL",
        TAPE_GAUGE_RIGHT_INNER_X,
        false,
        Color::srgb(0.98, 0.78, 0.78),
        Color::srgba(0.88, 0.36, 0.36, 0.45),
        Color::srgb(0.92, 0.22, 0.22),
    );
}

fn spawn_aircraft_status_panel(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn(hud_panel(
            Val::Auto,
            Val::Auto,
            Val::Px(16.0),
            Val::Px(16.0),
            STATUS_PANEL_WIDTH,
        ))
        .insert(HudLayer)
        .with_children(|parent| {
            parent.spawn((
                Text::new("AIRCRAFT"),
                ui_table_font(fonts, 11.0),
                TextColor(Color::srgb(0.94, 0.92, 0.84)),
                AircraftStatusText,
            ));

            parent
                .spawn((
                    Node {
                        width: percent(100.0),
                        height: px(14.0),
                        margin: UiRect::top(px(4.0)),
                        border: UiRect::all(px(1.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.08, 0.10, 0.12, 0.92)),
                    BorderColor::all(Color::srgba(0.92, 0.92, 0.92, 0.52)),
                ))
                .with_children(|bar| {
                    bar.spawn((
                        Node {
                            width: percent(100.0),
                            height: percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.92, 0.92, 0.92)),
                        HpGaugeFill,
                    ));
                });
            parent.spawn((
                Text::new("HP 100/100"),
                ui_table_font(fonts, 10.0),
                TextColor(Color::srgb(0.94, 0.92, 0.92)),
                HpGaugeText,
            ));

            parent
                .spawn((
                    Node {
                        width: percent(100.0),
                        height: px(14.0),
                        margin: UiRect::top(px(6.0)),
                        border: UiRect::all(px(1.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.08, 0.10, 0.12, 0.92)),
                    BorderColor::all(Color::srgba(0.97, 0.78, 0.42, 0.52)),
                ))
                .with_children(|bar| {
                    bar.spawn((
                        Node {
                            width: percent(0.0),
                            height: percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.98, 0.78, 0.42)),
                        HeatGaugeFill,
                    ));
                });

            parent.spawn((
                Text::new("GUN HEAT 0%"),
                ui_table_font(fonts, 10.0),
                TextColor(Color::srgb(0.97, 0.90, 0.80)),
                HeatGaugeText,
            ));
        });
}

fn spawn_damage_indicator_ring(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: percent(50.0),
                top: percent(50.0),
                width: px(0.0),
                height: px(0.0),
                ..default()
            },
            Visibility::Inherited,
            HudLayer,
        ))
        .with_children(|parent| {
            for index in 0..DAMAGE_INDICATOR_SLOTS {
                parent.spawn((
                    Text::new("↑"),
                    ui_symbol_font(fonts, 28.0),
                    TextColor(Color::srgba(1.0, 0.25, 0.16, 0.0)),
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(-8.0),
                        top: px(-8.0),
                        ..default()
                    },
                    Visibility::Hidden,
                    DamageIndicatorSlot { index },
                ));
            }
        });
}

fn spawn_controls_guide(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: percent(50.0),
                left: percent(50.0),
                margin: UiRect {
                    left: px(-210.0),
                    top: px(-100.0),
                    ..default()
                },
                width: px(420.0),
                padding: UiRect::axes(px(14.0), px(12.0)),
                border: UiRect::all(px(1.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.03, 0.05, 0.07, 0.88)),
            BorderColor::all(Color::srgba(0.88, 0.90, 0.92, 0.38)),
            Visibility::Hidden,
            HudLayer,
            ControlsGuideRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                ui_text_font(fonts, 12.0),
                TextColor(Color::srgb(0.94, 0.92, 0.84)),
                ControlsGuideText,
            ));
        });
}

pub fn spawn_replay_progress_hud(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: px(12.0),
                left: percent(50.0),
                margin: UiRect::left(px(-230.0)),
                width: px(460.0),
                padding: UiRect::axes(px(12.0), px(10.0)),
                border: UiRect::all(px(1.0)),
                flex_direction: FlexDirection::Column,
                row_gap: px(8.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.08, 0.11, 0.72)),
            BorderColor::all(Color::srgba(0.70, 0.82, 0.90, 0.35)),
            HudLayer,
            ReplayProgressHudRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                ui_table_font(fonts, 12.0),
                TextColor(Color::srgb(0.94, 0.92, 0.84)),
                ReplayProgressHudText,
            ));
            parent
                .spawn((
                    Node {
                        width: percent(100.0),
                        height: px(18.0),
                        border: UiRect::all(px(1.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        overflow: Overflow::clip_x(),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.08, 0.10, 0.13, 0.96)),
                    BorderColor::all(Color::srgba(0.70, 0.82, 0.90, 0.35)),
                ))
                .with_children(|bar| {
                    bar.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.0),
                            top: px(0.0),
                            bottom: px(0.0),
                            width: percent(0.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.74, 0.88, 0.98)),
                        ReplayProgressBarFill,
                    ));
                    bar.spawn((
                        Text::new(""),
                        ui_table_font(fonts, 11.0),
                        TextColor(Color::srgb(0.12, 0.18, 0.22)),
                        ReplayProgressBarText,
                    ));
                });
        });
}

fn spawn_out_of_bounds_overlay(commands: &mut Commands, fonts: &UiFontHandles) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: percent(100.0),
                height: percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::FlexStart,
                padding: UiRect::top(px(48.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.92, 0.93, 0.95, 0.0)),
            Visibility::Hidden,
            HudLayer,
            OutOfBoundsOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                ui_text_font(fonts, 20.0),
                TextColor(Color::srgb(0.12, 0.13, 0.15)),
                OutOfBoundsWarningText,
            ));
        });
}

fn hud_panel(top: Val, left: Val, right: Val, bottom: Val, width: f32) -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            top,
            left,
            right,
            bottom,
            width: px(width),
            padding: UiRect::axes(px(12.0), px(10.0)),
            border: UiRect::all(px(1.0)),
            flex_direction: FlexDirection::Column,
            row_gap: px(6.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.04, 0.08, 0.11, 0.72)),
        BorderColor::all(Color::srgba(0.70, 0.82, 0.90, 0.35)),
    )
}

pub fn attach_hud_to_main_camera(
    camera_query: Query<Entity, With<MainViewCamera>>,
    mut hud_query: Query<(Entity, Option<&UiTargetCamera>), With<HudLayer>>,
    mut commands: Commands,
) {
    let Ok(camera_entity) = camera_query.single() else {
        return;
    };

    for (hud_entity, target_camera) in &mut hud_query {
        if target_camera.is_none() {
            commands
                .entity(hud_entity)
                .insert(UiTargetCamera(camera_entity));
        }
    }
}

pub fn toggle_controls_guide(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    mut guide_state: ResMut<ControlsGuideState>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings)) = (keyboard, mouse_buttons, bindings)
    else {
        return;
    };

    if bindings
        .toggle_controls_guide
        .just_pressed(&keyboard, &mouse_buttons)
    {
        guide_state.visible = !guide_state.visible;
    }
}

pub fn update_live_spectator_observed_role(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    observer_view: Res<ObserverViewContext>,
    mut observed_role: ResMut<ObservedAircraftRole>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    if observer_view.session != ObserverSessionKind::SpectatorLive {
        return;
    }
    if keyboard.just_pressed(KeyCode::F2) {
        observed_role.0 = match observed_role.0 {
            AircraftRole::Fighter1 => AircraftRole::Fighter2,
            AircraftRole::Fighter2 => AircraftRole::Fighter1,
        };
    }
}

pub fn update_damage_indicators(
    time: Res<Time>,
    observed_role: Res<ObservedAircraftRole>,
    mut indicator_queue: ResMut<DamageIndicatorQueue>,
    mut indicator_state: ResMut<DamageIndicatorState>,
    aircraft_query: Query<(&AircraftRole, &AircraftState)>,
) {
    let dt = time.delta_secs();
    for indicator in &mut indicator_state.indicators {
        indicator.ttl_seconds -= dt;
    }
    indicator_state
        .indicators
        .retain(|indicator| indicator.ttl_seconds > 0.0);

    let Some((_, observed)) = aircraft_query
        .iter()
        .find(|(role, state)| **role == observed_role.0 && !state.is_destroyed)
    else {
        indicator_queue.events.clear();
        indicator_state.indicators.clear();
        return;
    };

    for event in indicator_queue.events.drain(..) {
        let Some(angle_radians) = damage_indicator_angle_radians(
            observed.position,
            observed.orientation,
            event.source_position,
        ) else {
            continue;
        };
        let intensity = event.intensity.clamp(0.2, 1.4);

        if let Some(existing) = indicator_state.indicators.iter_mut().find(|existing| {
            shortest_angle_distance(existing.angle_radians, angle_radians).abs()
                <= DAMAGE_INDICATOR_MERGE_ANGLE_RADIANS
        }) {
            existing.angle_radians = blend_angle(existing.angle_radians, angle_radians, 0.35);
            existing.intensity = (existing.intensity + intensity).clamp(0.2, 1.8);
            existing.ttl_seconds = DAMAGE_INDICATOR_LIFETIME;
        } else {
            indicator_state.indicators.push(ActiveDamageIndicator {
                angle_radians,
                ttl_seconds: DAMAGE_INDICATOR_LIFETIME,
                intensity,
            });
        }
    }

    indicator_state.indicators.sort_by(|a, b| {
        b.intensity
            .partial_cmp(&a.intensity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indicator_state.indicators.truncate(DAMAGE_INDICATOR_SLOTS);
}

fn damage_indicator_angle_radians(
    observed_position: Vec3,
    observed_orientation: Quat,
    source_position: Vec3,
) -> Option<f32> {
    let to_source_world = source_position - observed_position;
    if to_source_world.length_squared() <= f32::EPSILON {
        return None;
    }

    let to_source_local = observed_orientation.inverse() * to_source_world.normalize();
    Some((-to_source_local.x).atan2(to_source_local.z))
}

fn shortest_angle_distance(from: f32, to: f32) -> f32 {
    let mut delta = to - from;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta
}

fn blend_angle(current: f32, target: f32, weight: f32) -> f32 {
    current + shortest_angle_distance(current, target) * weight.clamp(0.0, 1.0)
}

fn directional_arrow(angle_radians: f32) -> &'static str {
    let angle = angle_radians.rem_euclid(std::f32::consts::TAU);
    let sector = ((angle / (std::f32::consts::TAU / 8.0)).round() as i32).rem_euclid(8);
    match sector {
        0 => "↑",
        1 => "↗",
        2 => "→",
        3 => "↘",
        4 => "↓",
        5 => "↙",
        6 => "←",
        _ => "↖",
    }
}

fn compact_metric_u64(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:>4.1}m", value as f64 / 1_000_000.0)
    } else if value >= 10_000 {
        format!("{:>4.1}k", value as f64 / 1_000.0)
    } else {
        format!("{value:>4}")
    }
}

fn compact_metric_u32(value: u32) -> String {
    compact_metric_u64(value as u64)
}

pub fn update_damage_indicator_hud(
    indicator_state: Res<DamageIndicatorState>,
    mut slot_query: Query<
        (
            &DamageIndicatorSlot,
            &mut Node,
            &mut Visibility,
            &mut TextColor,
            &mut Text,
        ),
        Without<CompassTick>,
    >,
) {
    for (slot, mut node, mut visibility, mut text_color, mut text) in &mut slot_query {
        let Some(indicator) = indicator_state.indicators.get(slot.index) else {
            *visibility = Visibility::Hidden;
            continue;
        };

        let fade = (indicator.ttl_seconds / DAMAGE_INDICATOR_LIFETIME).clamp(0.0, 1.0);
        let x = indicator.angle_radians.sin() * DAMAGE_INDICATOR_RADIUS;
        let y = -indicator.angle_radians.cos() * DAMAGE_INDICATOR_RADIUS;
        node.left = px(x - 10.0);
        node.top = px(y - 12.0);
        text.0 = directional_arrow(indicator.angle_radians).to_string();
        *text_color = TextColor(Color::srgba(
            1.0,
            0.28 + indicator.intensity * 0.12,
            0.18,
            0.18 + fade * 0.72,
        ));
        *visibility = Visibility::Inherited;
    }
}

pub fn update_hud(
    time: Res<Time<Real>>,
    config: Res<RepositoryConfig>,
    observed_role: Res<ObservedAircraftRole>,
    compass_reference: Res<CompassReferenceState>,
    observer_view: Res<ObserverViewContext>,
    observer_feed: Res<ObserverFeedStatus>,
    observer_provider: Res<ObserverFeedProviderState>,
    mut speed_gauge: ResMut<SpeedGaugeState>,
    bridge_resources: (
        Option<Res<IpcBridgeState>>,
        Option<Res<BridgeLinkState>>,
        Option<Res<BridgeClientHandshakeState>>,
        Option<Res<BridgeTimingState>>,
        Option<Res<AssignedControlRole>>,
        Option<Res<BridgeClientInbox>>,
        Option<Res<LocalPredictionState>>,
        Option<Res<BridgeJitterDiagnostics>>,
    ),
    mut text_queries: ParamSet<(
        Query<&mut Text, With<MatchHudText>>,
        Query<&mut Text, With<NetworkHudText>>,
        Query<&mut Text, With<AttitudeText>>,
        Query<&mut Text, With<AircraftStatusText>>,
        Query<(
            Option<&HpGaugeText>,
            Option<&HeatGaugeText>,
            Option<&VerticalTapeCurrentText>,
            Option<&SpeedGaugeCurrentText>,
            Option<&OutOfBoundsWarningText>,
            Option<&CompassTickLabel>,
            Option<&SpeedGaugeTickLabel>,
            &mut Text,
        )>,
        Query<
            (
                &VerticalTapeTickLabel,
                &mut Text,
                &mut TextColor,
                &mut Node,
                &mut Visibility,
            ),
            (
                Without<HpGaugeFill>,
                Without<HeatGaugeFill>,
                Without<VerticalTapePointer>,
                Without<SpeedGaugePointer>,
                Without<CompassTick>,
                Without<SpeedGaugeTick>,
                Without<VerticalTapeTickMark>,
                Without<OutOfBoundsOverlay>,
            ),
        >,
    )>,
    mut node_queries: ParamSet<(
        Query<
            (&mut Node, &mut BackgroundColor),
            (
                With<HpGaugeFill>,
                Without<HeatGaugeFill>,
                Without<VerticalTapePointer>,
                Without<SpeedGaugePointer>,
                Without<CompassTick>,
                Without<SpeedGaugeTick>,
                Without<VerticalTapeTickMark>,
                Without<OutOfBoundsOverlay>,
            ),
        >,
        Query<&mut Node, With<HeatGaugeFill>>,
        Query<(&VerticalTapePointer, &mut Node)>,
        Query<&mut Node, With<SpeedGaugePointer>>,
        Query<(&CompassTick, &mut Node, &mut Visibility)>,
        Query<(&SpeedGaugeTick, &mut Node, &mut Visibility)>,
        Query<
            (
                &VerticalTapeTickMark,
                &mut Node,
                &mut BackgroundColor,
                &mut Visibility,
            ),
            (Without<OutOfBoundsOverlay>,),
        >,
    )>,
    mut overlay_query: Query<
        (&mut BackgroundColor, &mut Visibility),
        (
            With<OutOfBoundsOverlay>,
            Without<CompassTick>,
            Without<SpeedGaugeTick>,
            Without<VerticalTapeTickMark>,
            Without<VerticalTapeTickLabel>,
        ),
    >,
    aircraft_query: Query<(
        &AircraftRole,
        &AircraftState,
        &AircraftPerformance,
        Option<&AircraftDamageState>,
        Option<&GunState>,
    )>,
    camera_query: Query<&Transform, With<MainViewCamera>>,
) {
    let (
        bridge_state,
        bridge_link,
        bridge_handshake,
        bridge_timing,
        _assigned_role,
        bridge_inbox,
        bridge_prediction,
        bridge_jitter,
    ) = bridge_resources;
    let Some((_, player, player_performance, player_damage, player_gun)) = aircraft_query
        .iter()
        .find(|(role, _, _, _, _)| **role == observed_role.0)
    else {
        return;
    };

    let speed = player.velocity.length();
    let vertical_speed = player.velocity.y;
    let local_velocity = player.orientation.inverse() * player.velocity;
    let pitch_deg = player.forward.y.clamp(-1.0, 1.0).asin().to_degrees();
    let roll_deg = compute_roll_degrees(player.orientation);
    let yaw_forward = if compass_reference.use_camera_forward {
        camera_query
            .iter()
            .next()
            .map(|transform| transform.forward().as_vec3())
            .unwrap_or(player.forward)
    } else {
        player.forward
    };
    let yaw_deg = compute_heading_degrees(yaw_forward);
    let forward_speed = (-local_velocity.z).abs().max(0.1);
    let aoa_deg = (-local_velocity.y).atan2(forward_speed).to_degrees();
    let sideslip_deg = local_velocity.x.atan2(forward_speed).to_degrees();
    let gun_heat = player_gun.map(|gun| gun.heat).unwrap_or(0.0);
    let overheated = player_gun.map(|gun| gun.overheated).unwrap_or(false);
    let altitude = player.position.y - config.scene.ground_height;
    let stall_ratio = player.stall_factor.clamp(0.0, 1.0);
    let speed_alpha = 1.0 - (-time.delta_secs() * 2.4).exp();
    speed_gauge.display_center_speed += (speed - speed_gauge.display_center_speed) * speed_alpha;
    let altitude_alpha = 1.0 - (-time.delta_secs() * 2.0).exp();
    speed_gauge.display_center_altitude +=
        (altitude - speed_gauge.display_center_altitude) * altitude_alpha;
    let out_of_bounds_remaining =
        (config.scene.out_of_bounds_grace_seconds - player.out_of_bounds_seconds).max(0.0);
    let repair_remaining = player_damage
        .map(|damage| {
            (config.game.repair_duration_seconds - damage.repair_elapsed_seconds).max(0.0)
        })
        .unwrap_or(0.0);
    let player_count_line = bridge_inbox
        .as_deref()
        .and_then(|inbox| inbox.latest_snapshot.as_ref())
        .map(|snapshot| {
            snapshot
                .occupied_slots
                .iter()
                .filter(|slot| {
                    matches!(
                        slot,
                        BridgeControlSlot::Fighter1 | BridgeControlSlot::Fighter2
                    )
                })
                .count()
        })
        .map(|count| format!("{count:>1}/2"))
        .unwrap_or_else(|| {
            if bridge_link
                .as_deref()
                .map(|link| link.remote_authority_active)
                .unwrap_or(false)
            {
                "0/2".to_string()
            } else {
                "-/2".to_string()
            }
        });
    let server_status_line = bridge_link
        .as_deref()
        .map(|state| {
            let base = bridge_connection_status_label(state.client_status).to_string();
            match state.client_status {
                BridgeClientConnectionStatus::Timeout => bridge_handshake
                    .as_deref()
                    .and_then(|handshake| handshake.next_retry_tick)
                    .map(|retry_tick| {
                        let current_tick = bridge_state
                            .as_deref()
                            .and_then(|state| state.last_client_tick)
                            .unwrap_or(0);
                        let remaining = retry_tick.saturating_sub(current_tick);
                        format!("{base} retry~{remaining}t")
                    })
                    .unwrap_or(base),
                BridgeClientConnectionStatus::SessionMismatch => {
                    format!("{base} check-session")
                }
                BridgeClientConnectionStatus::ProtocolMismatch => {
                    format!("{base} rebuild")
                }
                _ => base,
            }
        })
        .unwrap_or_else(|| {
            if bridge_link
                .as_deref()
                .map(|link| link.remote_authority_active)
                .unwrap_or(false)
            {
                "CONNECTED".to_string()
            } else {
                "CONNECTING".to_string()
            }
        });
    let wait_line = if observer_feed.phase == Some(MatchPhase::Loading) {
        format!("WAITING FOR PLAYERS {players}", players = player_count_line)
    } else {
        "-".to_string()
    };
    let viewer_target = match observed_role.0 {
        AircraftRole::Fighter1 => "FIGHTER1",
        AircraftRole::Fighter2 => "FIGHTER2",
    };
    let viewer_line = format!(
        "{} {viewer_target}",
        observer_session_kind_label(observer_view.session),
    );
    let net_rows = bridge_timing
        .as_deref()
        .map(|timing| {
            let age_ms = timing
                .last_snapshot_receive_seconds
                .map(|received| ((time.elapsed_secs_f64() - received).max(0.0) * 1000.0) as u32)
                .map(compact_metric_u32)
                .unwrap_or_else(|| "-".to_string());
            let interval_ms = timing
                .average_snapshot_interval_seconds
                .map(|interval| (interval * 1000.0) as u32)
                .map(compact_metric_u32)
                .unwrap_or_else(|| "-".to_string());
            let ping_ms = timing
                .average_rtt_seconds
                .map(|rtt| (rtt * 1000.0) as u32)
                .map(compact_metric_u32)
                .unwrap_or_else(|| "-".to_string());
            let ack = bridge_prediction
                .as_deref()
                .and_then(|prediction| prediction.last_acked_tick)
                .map(compact_metric_u64)
                .unwrap_or_else(|| "-".to_string());
            let base = bridge_prediction
                .as_deref()
                .and_then(|prediction| prediction.baseline_tick(observed_role.0))
                .map(compact_metric_u64)
                .unwrap_or_else(|| "-".to_string());
            let replay = bridge_prediction
                .as_deref()
                .map(|prediction| compact_metric_u64(prediction.pending_input_count() as u64))
                .unwrap_or_else(|| "-".to_string());
            let warmup = bridge_jitter
                .as_deref()
                .and_then(|diag| {
                    diag.connected_at_client_tick.and_then(|connected_tick| {
                        bridge_state
                            .as_deref()
                            .and_then(|state| state.last_client_tick)
                            .map(|current_tick| current_tick.saturating_sub(connected_tick))
                    })
                })
                .map(compact_metric_u64)
                .unwrap_or_else(|| "-".to_string());
            let snaps = bridge_jitter
                .as_deref()
                .map(|diag| compact_metric_u32(diag.local_snap_count))
                .unwrap_or_else(|| "-".to_string());
            let baseline_init = bridge_jitter
                .as_deref()
                .and_then(|diag| diag.first_baseline_tick)
                .map(compact_metric_u64)
                .unwrap_or_else(|| "-".to_string());
            vec![
                vec![
                    Cell::left("AGE"),
                    Cell::right(age_ms),
                    Cell::left("GAP"),
                    Cell::right(interval_ms),
                    Cell::left("PNG"),
                    Cell::right(ping_ms),
                ],
                vec![
                    Cell::left("ACK"),
                    Cell::right(ack),
                    Cell::left("BASE"),
                    Cell::right(base),
                    Cell::left("RPY"),
                    Cell::right(replay),
                ],
                vec![
                    Cell::left("WARM"),
                    Cell::right(warmup),
                    Cell::left("SNAP"),
                    Cell::right(snaps),
                    Cell::left("INIT"),
                    Cell::right(baseline_init),
                ],
            ]
        })
        .unwrap_or_else(|| {
            vec![
                vec![
                    Cell::left("AGE"),
                    Cell::right("-"),
                    Cell::left("GAP"),
                    Cell::right("-"),
                    Cell::left("PNG"),
                    Cell::right("-"),
                ],
                vec![
                    Cell::left("ACK"),
                    Cell::right("-"),
                    Cell::left("BASE"),
                    Cell::right("-"),
                    Cell::left("RPY"),
                    Cell::right("-"),
                ],
                vec![
                    Cell::left("WARM"),
                    Cell::right("-"),
                    Cell::left("SNAP"),
                    Cell::right("-"),
                    Cell::left("INIT"),
                    Cell::right("-"),
                ],
            ]
        });
    let playback_suffix = match (
        observer_feed.playback_finished,
        observer_feed.playback_paused,
        observer_feed.playback_speed,
    ) {
        (Some(true), _, Some(speed)) => format!(" done@{speed:.2}x"),
        (_, Some(paused), Some(speed)) => {
            let state = if paused { "pause" } else { "play" };
            format!(" {state}@{speed:.2}x")
        }
        _ => String::new(),
    };
    let feed_line = observer_provider.provider.map(|provider| {
        let provider = observer_feed_provider_label(provider);
        observer_feed.current_tick.map_or_else(
            || provider.to_string(),
            |tick| match (observer_feed.current_step, observer_feed.total_steps) {
                (Some(current), Some(total)) => {
                    format!("{provider} t={tick:>6} s={current:>4}/{total:<4}{playback_suffix}")
                }
                _ => format!("{provider} t={tick:>6}{playback_suffix}"),
            },
        )
    });

    if let Ok(mut text) = text_queries.p0().single_mut() {
        let rows = vec![
            vec![
                Cell::left("PHASE"),
                Cell::right(short_phase_label(
                    observer_feed.phase.unwrap_or(MatchPhase::Loading),
                )),
            ],
            vec![
                Cell::left("TIME"),
                Cell::right(format!("{:>6.1}s", observer_feed.sim_time_seconds)),
            ],
            vec![
                Cell::left("STATE"),
                Cell::right(if player.is_destroyed {
                    "DESTROYED".to_string()
                } else {
                    "ACTIVE".to_string()
                }),
            ],
            vec![Cell::left("PILOTS"), Cell::right(player_count_line.clone())],
            vec![Cell::left("WAIT"), Cell::right(wait_line.clone())],
            vec![
                Cell::left("BOUNDS"),
                Cell::right(if player.out_of_bounds_seconds > 0.0 {
                    format!("{out_of_bounds_remaining:>6.1}s")
                } else {
                    format!("{:>6}", "IN")
                }),
            ],
        ];
        text.0 = format!("MATCH\n{}", render_rows_with_min_widths(&rows, 2, &[6, 10]));
    }

    if let Ok(mut text) = text_queries.p1().single_mut() {
        let summary_rows = vec![
            vec![Cell::left("VIEW"), Cell::left(viewer_line)],
            vec![Cell::left("LINK"), Cell::left(server_status_line.clone())],
            vec![
                Cell::left("FEED"),
                Cell::left(feed_line.unwrap_or_else(|| "-".to_string())),
            ],
        ];
        let summary = render_rows_with_min_widths(&summary_rows, 2, &[5, 26]);
        let metrics = render_rows_with_min_widths(&net_rows, 2, &[4, 6, 4, 6, 4, 6]);
        text.0 = format!(
            "NETWORK\n{summary}\n{metrics}",
            summary = summary,
            metrics = metrics,
        );
    }

    if let Ok(mut text) = text_queries.p2().single_mut() {
        let rows = vec![
            vec![
                Cell::left("POS"),
                Cell::left("X"),
                Cell::right(format!("{:+7.0}", player.position.x)),
                Cell::left("Y"),
                Cell::right(format!("{:+7.0}", player.position.y)),
                Cell::left("Z"),
                Cell::right(format!("{:+7.0}", player.position.z)),
            ],
            vec![
                Cell::left("ATT"),
                Cell::left("YAW"),
                Cell::right(format!("{:+7.1}", yaw_deg)),
                Cell::left("PIT"),
                Cell::right(format!("{:+7.1}", pitch_deg)),
                Cell::left("ROL"),
                Cell::right(format!("{:+7.1}", roll_deg)),
            ],
            vec![
                Cell::left("VEL"),
                Cell::left("SPD"),
                Cell::right(format!("{:>7.1}", speed)),
                Cell::left("VSI"),
                Cell::right(format!("{:+7.1}", vertical_speed)),
                Cell::left("THR"),
                Cell::right(format!("{:>6.0}%", player.throttle * 100.0)),
            ],
            vec![
                Cell::left("AIR"),
                Cell::left("AOA"),
                Cell::right(format!("{:+7.1}", aoa_deg)),
                Cell::left("SSL"),
                Cell::right(format!("{:+7.1}", sideslip_deg)),
                Cell::left("STL"),
                Cell::right(format!("{:>7.2}", player.stall_factor)),
            ],
        ];
        text.0 = format!(
            "POSE\n{}",
            render_rows_with_min_widths(&rows, 2, &[4, 3, 6, 3, 6, 3, 6])
        );
    }

    if let Ok(mut text) = text_queries.p3().single_mut() {
        let rows = if let Some(damage) = player_damage {
            vec![
                vec![
                    Cell::left("STALL"),
                    Cell::right(format!("{:>7.2}", player.stall_factor)),
                ],
                vec![
                    Cell::left("REPAIR"),
                    Cell::right(if damage.is_repairing {
                        format!("{repair_remaining:>7.1}s")
                    } else {
                        format!("{:>7}", "READY")
                    }),
                ],
                vec![
                    Cell::left("GUN"),
                    Cell::right(if overheated {
                        format!("{:>8}", "OVERHEAT")
                    } else {
                        format!("{:>8}", "READY")
                    }),
                ],
                vec![
                    Cell::left("LW"),
                    Cell::right(short_health(&damage.left_wing)),
                    Cell::left("RW"),
                    Cell::right(short_health(&damage.right_wing)),
                ],
                vec![
                    Cell::left("PT"),
                    Cell::right(short_health(&damage.pitch_tail)),
                    Cell::left("YT"),
                    Cell::right(short_health(&damage.yaw_tail)),
                ],
                vec![Cell::left("ENG"), Cell::right(short_health(&damage.engine))],
            ]
        } else {
            vec![
                vec![
                    Cell::left("STALL"),
                    Cell::right(format!("{:>7.2}", player.stall_factor)),
                ],
                vec![Cell::left("REPAIR"), Cell::right(format!("{:>7}", "?"))],
                vec![
                    Cell::left("GUN"),
                    Cell::right(format!(
                        "{:>8}",
                        if overheated { "OVERHEAT" } else { "READY" }
                    )),
                ],
                vec![
                    Cell::left("LW"),
                    Cell::right(format!("{:>6}", "?")),
                    Cell::left("RW"),
                    Cell::right(format!("{:>6}", "?")),
                ],
                vec![
                    Cell::left("PT"),
                    Cell::right(format!("{:>6}", "?")),
                    Cell::left("YT"),
                    Cell::right(format!("{:>6}", "?")),
                ],
                vec![Cell::left("ENG"), Cell::right(format!("{:>6}", "?"))],
            ]
        };
        text.0 = format!(
            "AIRCRAFT\n{}",
            render_rows_with_min_widths(&rows, 2, &[6, 8, 3, 8])
        );
    }

    if let Ok((mut node, mut color)) = node_queries.p0().single_mut() {
        let max_hp = player_damage
            .map(|damage| damage.total_max_hit_points)
            .unwrap_or(100.0)
            .max(1.0);
        let hp_ratio = (player.hit_points / max_hp).clamp(0.0, 1.0);
        node.width = percent(hp_ratio * 100.0);
        let hp_color = Color::srgb(1.0 - hp_ratio * 0.08, hp_ratio, hp_ratio);
        *color = BackgroundColor(hp_color);
    }
    if let Ok(mut node) = node_queries.p1().single_mut() {
        node.width = percent(gun_heat.clamp(0.0, 1.0) * 100.0);
    }
    let tape_center_y = (TAPE_GAUGE_HEIGHT * 0.5) - 8.0;
    let throttle_pointer_offset = (-(player.throttle - 0.5) / THROTTLE_TAPE_TICK_STEP)
        * vertical_tape_tick_spacing(VerticalTapeKind::Throttle);
    let stall_pointer_offset = (-(player.stall_factor - 0.5) / STALL_TAPE_TICK_STEP)
        * vertical_tape_tick_spacing(VerticalTapeKind::Stall);
    let altitude_pointer_offset = (-(altitude - speed_gauge.display_center_altitude)
        / ALTITUDE_TAPE_TICK_STEP)
        * vertical_tape_tick_spacing(VerticalTapeKind::Altitude);
    for (pointer, mut node) in &mut node_queries.p2() {
        let top = match pointer.kind {
            VerticalTapeKind::Altitude => tape_center_y + altitude_pointer_offset,
            VerticalTapeKind::Throttle => tape_center_y + throttle_pointer_offset,
            VerticalTapeKind::Stall => tape_center_y + stall_pointer_offset,
        };
        node.top = px(top.clamp(-8.0, TAPE_GAUGE_HEIGHT - 8.0));
    }
    if let Ok(mut node) = node_queries.p3().single_mut() {
        let speed_delta = speed - speed_gauge.display_center_speed;
        let pointer_offset = (-speed_delta / SPEED_TAPE_TICK_STEP) * SPEED_TAPE_TICK_SPACING;
        let center_y = (TAPE_GAUGE_HEIGHT * 0.5) - 8.0;
        node.top = px((center_y + pointer_offset).clamp(-8.0, TAPE_GAUGE_HEIGHT - 8.0));
    }
    let heading = yaw_deg.rem_euclid(360.0);
    let base_tick = (heading / 10.0).floor() as i32;
    let fractional = (heading / 10.0) - base_tick as f32;
    let center_x = COMPASS_WIDTH * 0.5;

    for (tick, mut node, mut visibility) in &mut node_queries.p4() {
        let x = center_x - ((tick.index as f32 - fractional) * COMPASS_TICK_SPACING);
        node.left = px(x - (COMPASS_TICK_WIDTH * 0.5));
        *visibility = if (-COMPASS_TICK_WIDTH..=COMPASS_WIDTH + COMPASS_TICK_WIDTH).contains(&x) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }

    let speed_ticks = speed_gauge.display_center_speed / SPEED_TAPE_TICK_STEP;
    let speed_base_tick = speed_ticks.floor() as i32;
    let speed_fractional = speed_ticks - speed_base_tick as f32;
    let speed_center_y = TAPE_GAUGE_HEIGHT * 0.5;
    for (tick, mut node, mut visibility) in &mut node_queries.p5() {
        let y = speed_center_y
            - (((tick.offset as f32) - speed_fractional) * SPEED_TAPE_TICK_SPACING)
            - 8.0;
        node.top = px(y);
        let tick_speed = (speed_base_tick + tick.offset) as f32 * SPEED_TAPE_TICK_STEP;
        *visibility = if tick_speed < 0.0 || !(-8.0..=TAPE_GAUGE_HEIGHT - 8.0).contains(&y) {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }

    let altitude_ticks = speed_gauge.display_center_altitude / ALTITUDE_TAPE_TICK_STEP;
    let altitude_tick_base = altitude_ticks.floor() as i32;
    let altitude_tick_fractional = altitude_ticks - altitude_tick_base as f32;
    let throttle_tick_base = 5;
    let throttle_tick_fractional = 0.0;
    let stall_tick_base = 5;
    let stall_tick_fractional = 0.0;
    let vertical_center_y = TAPE_GAUGE_HEIGHT * 0.5;
    let ceiling_height = config.scene.flight_ceiling_height.max(1.0);
    let altitude_warning_range = altitude_warning_range(player_performance);
    let warning_flash = ((time.elapsed_secs() * 7.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    for (tick, mut node, mut color, mut visibility) in &mut node_queries.p6() {
        let (base_tick, fractional, step, value, visible, bg_color) = match tick.kind {
            VerticalTapeKind::Altitude => {
                let tick_value =
                    (altitude_tick_base + tick.offset) as f32 * ALTITUDE_TAPE_TICK_STEP;
                let in_visible_range = tick_value >= 0.0 && tick_value <= ceiling_height;
                let dark = Color::srgba(0.44, 0.48, 0.52, 0.72);
                let bright = Color::srgb(0.54, 0.80, 0.97);
                (
                    altitude_tick_base,
                    altitude_tick_fractional,
                    ALTITUDE_TAPE_TICK_STEP,
                    tick_value,
                    true,
                    if in_visible_range { bright } else { dark },
                )
            }
            VerticalTapeKind::Throttle => {
                let tick_value =
                    (throttle_tick_base + tick.offset) as f32 * THROTTLE_TAPE_TICK_STEP;
                (
                    throttle_tick_base,
                    throttle_tick_fractional,
                    THROTTLE_TAPE_TICK_STEP,
                    tick_value,
                    (0.0..=1.0).contains(&tick_value),
                    Color::srgb(0.98, 0.78, 0.42),
                )
            }
            VerticalTapeKind::Stall => {
                let tick_value = (stall_tick_base + tick.offset) as f32 * STALL_TAPE_TICK_STEP;
                (
                    stall_tick_base,
                    stall_tick_fractional,
                    STALL_TAPE_TICK_STEP,
                    tick_value,
                    (0.0..=1.0).contains(&tick_value),
                    Color::srgb(0.92, 0.22, 0.22),
                )
            }
        };
        let y = vertical_center_y
            - ((((value / step) - base_tick as f32) - fractional)
                * vertical_tape_tick_spacing(tick.kind))
            - 1.0;
        node.top = px(y);
        let flash_alpha = match tick.kind {
            VerticalTapeKind::Altitude => {
                let near_ground =
                    (1.0 - (altitude / altitude_warning_range).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                let remaining = (ceiling_height - altitude).max(0.0);
                let near_ceiling =
                    (1.0 - (remaining / altitude_warning_range).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                near_ground.max(near_ceiling) * warning_flash
            }
            VerticalTapeKind::Stall => stall_ratio * warning_flash,
            VerticalTapeKind::Throttle => 0.0,
        };
        *color = BackgroundColor(if flash_alpha > 0.05 {
            Color::srgba(0.96, 0.18, 0.18, 0.25 + flash_alpha * 0.55)
        } else {
            bg_color
        });
        *visibility = if visible && (-1.0..=TAPE_GAUGE_HEIGHT - 1.0).contains(&y) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }

    for (
        hp_text,
        heat_text,
        current_text,
        speed_text,
        warning_text,
        tick_label,
        speed_tick_label,
        mut text,
    ) in &mut text_queries.p4()
    {
        if hp_text.is_some() {
            let max_hp = player_damage
                .map(|damage| damage.total_max_hit_points)
                .unwrap_or(100.0)
                .max(1.0);
            text.0 = format!("HP {:>3.0}/{:<3.0}", player.hit_points, max_hp);
        } else if heat_text.is_some() {
            text.0 = format!(
                "GUN HEAT {:>3.0}%{}",
                gun_heat * 100.0,
                if overheated { "  OVERHEAT" } else { "" },
            );
        } else if let Some(current_text) = current_text {
            text.0 = match current_text.kind {
                VerticalTapeKind::Altitude => format!("{altitude:>4.0} m"),
                VerticalTapeKind::Throttle => format!("{:>3.0}%", player.throttle * 100.0),
                VerticalTapeKind::Stall => format!("{:>3.0}%", player.stall_factor * 100.0),
            };
        } else if speed_text.is_some() {
            text.0 = format!("{:>3.0}", speed);
        } else if warning_text.is_some() {
            text.0 = if player.out_of_bounds_seconds > 0.0 {
                format!("RETURN TO BATTLE AREA\n{out_of_bounds_remaining:>4.1}s BEFORE DESTRUCTION")
            } else {
                String::new()
            };
        } else if let Some(label) = tick_label {
            let tick_heading = (base_tick + label.index).rem_euclid(36) * 10;
            text.0 = compass_label(tick_heading);
        } else if let Some(label) = speed_tick_label {
            let tick_speed = (speed_base_tick + label.offset) as f32 * SPEED_TAPE_TICK_STEP;
            text.0 = if tick_speed < 0.0 {
                String::new()
            } else {
                format!("{:>3.0}", tick_speed)
            };
        }
    }

    for (label, mut text, mut color, mut node, mut visibility) in &mut text_queries.p5() {
        let (tick_value, label_text, label_color) = match label.kind {
            VerticalTapeKind::Altitude => {
                let tick_value =
                    (altitude_tick_base + label.offset) as f32 * ALTITUDE_TAPE_TICK_STEP;
                let in_visible_range = tick_value >= 0.0 && tick_value <= ceiling_height;
                let color = if in_visible_range {
                    Color::srgb(0.88, 0.94, 0.97)
                } else {
                    Color::srgba(0.56, 0.60, 0.64, 0.82)
                };
                (tick_value, format!("{:>4.0}", tick_value), color)
            }
            VerticalTapeKind::Throttle => {
                let tick_value =
                    (throttle_tick_base + label.offset) as f32 * THROTTLE_TAPE_TICK_STEP;
                (
                    tick_value,
                    format!("{:>3.0}", tick_value * 100.0),
                    Color::srgb(0.97, 0.90, 0.80),
                )
            }
            VerticalTapeKind::Stall => {
                let tick_value = (stall_tick_base + label.offset) as f32 * STALL_TAPE_TICK_STEP;
                (
                    tick_value,
                    format!("{:>3.0}", tick_value * 100.0),
                    Color::srgb(0.98, 0.78, 0.78),
                )
            }
        };
        let (base, fractional, step) = match label.kind {
            VerticalTapeKind::Altitude => (
                altitude_tick_base as f32,
                altitude_tick_fractional,
                ALTITUDE_TAPE_TICK_STEP,
            ),
            VerticalTapeKind::Throttle => (
                throttle_tick_base as f32,
                throttle_tick_fractional,
                THROTTLE_TAPE_TICK_STEP,
            ),
            VerticalTapeKind::Stall => (
                stall_tick_base as f32,
                stall_tick_fractional,
                STALL_TAPE_TICK_STEP,
            ),
        };
        let y = vertical_center_y
            - ((((tick_value / step) - base) - fractional)
                * vertical_tape_tick_spacing(label.kind))
            - 8.0;
        node.top = px(y);
        text.0 = label_text;
        let flash_alpha = match label.kind {
            VerticalTapeKind::Altitude => {
                let near_ground =
                    (1.0 - (altitude / altitude_warning_range).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                let remaining = (ceiling_height - altitude).max(0.0);
                let near_ceiling =
                    (1.0 - (remaining / altitude_warning_range).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                near_ground.max(near_ceiling) * warning_flash
            }
            VerticalTapeKind::Stall => stall_ratio * warning_flash,
            VerticalTapeKind::Throttle => 0.0,
        };
        *color = TextColor(if flash_alpha > 0.05 {
            Color::srgba(1.0, 0.40, 0.40, 0.45 + flash_alpha * 0.45)
        } else {
            label_color
        });
        let in_range = match label.kind {
            VerticalTapeKind::Altitude => true,
            VerticalTapeKind::Throttle | VerticalTapeKind::Stall => {
                (0.0..=1.0).contains(&tick_value)
            }
        };
        *visibility = if in_range && (-8.0..=TAPE_GAUGE_HEIGHT - 8.0).contains(&y) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }

    if let Ok((mut overlay_color, mut overlay_visibility)) = overlay_query.single_mut() {
        let warning_alpha = (player.out_of_bounds_seconds
            / config.scene.out_of_bounds_grace_seconds)
            .clamp(0.0, 1.0)
            * 0.72;
        *overlay_color = BackgroundColor(Color::srgba(0.92, 0.93, 0.95, warning_alpha));
        *overlay_visibility = if player.out_of_bounds_seconds > 0.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

pub fn update_controls_guide_text(
    config: Res<RepositoryConfig>,
    observer_view: Res<ObserverViewContext>,
    mut query: Query<&mut Text, With<ControlsGuideText>>,
) {
    let bindings = &config.input.bindings;
    let guide = match observer_view.session {
        ObserverSessionKind::PlayerLocal | ObserverSessionKind::PlayerLive => {
            let rows = vec![
                vec![
                    Cell::left("Mouse X"),
                    Cell::right(describe_mouse_axis(
                        "",
                        config.input.mouse_x_axis.target,
                        config.input.mouse_x_axis.invert,
                    )),
                    Cell::left("Mouse Y"),
                    Cell::right(describe_mouse_axis(
                        "",
                        config.input.mouse_y_axis.target,
                        config.input.mouse_y_axis.invert,
                    )),
                ],
                vec![
                    Cell::left("Pitch"),
                    Cell::right(describe_axis_pair(
                        &bindings.pitch_positive,
                        &bindings.pitch_negative,
                    )),
                    Cell::left("Roll"),
                    Cell::right(describe_axis_pair(
                        &bindings.roll_positive,
                        &bindings.roll_negative,
                    )),
                ],
                vec![
                    Cell::left("Yaw"),
                    Cell::right(describe_axis_pair(
                        &bindings.yaw_positive,
                        &bindings.yaw_negative,
                    )),
                    Cell::left("Throttle"),
                    Cell::right(describe_axis_pair(
                        &bindings.throttle_up,
                        &bindings.throttle_down,
                    )),
                ],
                vec![
                    Cell::left("Brake"),
                    Cell::right(describe_binding_group(&bindings.brake)),
                    Cell::left("Fire"),
                    Cell::right(describe_binding_group(&bindings.fire_gun)),
                ],
                vec![
                    Cell::left("Repair"),
                    Cell::right(describe_binding_group(&bindings.repair_aircraft)),
                    Cell::left("Rear"),
                    Cell::right(format!(
                        "Hold {}",
                        describe_binding_group(&bindings.rear_view)
                    )),
                ],
                vec![
                    Cell::left("Pilot"),
                    Cell::right(describe_binding_group(&bindings.toggle_local_pilot_mode)),
                    Cell::left("Guide"),
                    Cell::right(describe_binding_group(&bindings.toggle_controls_guide)),
                ],
                vec![
                    Cell::left("Mode"),
                    Cell::right("Human/AI"),
                    Cell::left("Capture"),
                    Cell::right(describe_binding_group(&bindings.toggle_mouse_capture)),
                ],
                vec![
                    Cell::left("F4"),
                    Cell::right("Flight Assist"),
                    Cell::left(""),
                    Cell::right(""),
                ],
            ];
            format!(
                "CONTROLS\n{}",
                render_rows_with_min_widths(&rows, 2, &[8, 18, 8, 18]),
            )
        }
        ObserverSessionKind::SpectatorLive => {
            let rows = vec![
                vec![
                    Cell::left("F1"),
                    Cell::right("Chase/Free"),
                    Cell::left("F2"),
                    Cell::right("Switch Target"),
                ],
                vec![
                    Cell::left("Rear"),
                    Cell::right(format!(
                        "Hold {}",
                        describe_binding_group(&bindings.rear_view)
                    )),
                    Cell::left("Capture"),
                    Cell::right(describe_binding_group(&bindings.toggle_mouse_capture)),
                ],
                vec![
                    Cell::left("Mouse"),
                    Cell::right("Look in Free"),
                    Cell::left("Move"),
                    Cell::right("WASD/QE"),
                ],
                vec![
                    Cell::left("Shift"),
                    Cell::right("Accelerate"),
                    Cell::left("Guide"),
                    Cell::right(describe_binding_group(&bindings.toggle_controls_guide)),
                ],
                vec![
                    Cell::left("F4"),
                    Cell::right("Flight Assist"),
                    Cell::left(""),
                    Cell::right(""),
                ],
            ];
            format!(
                "OBSERVER\n{}",
                render_rows_with_min_widths(&rows, 2, &[8, 18, 8, 18]),
            )
        }
        ObserverSessionKind::SpectatorReplay => {
            let rows = vec![
                vec![
                    Cell::left("Space"),
                    Cell::right("Play/Pause"),
                    Cell::left("L/R"),
                    Cell::right("Seek"),
                ],
                vec![
                    Cell::left("Up/Down"),
                    Cell::right("Speed"),
                    Cell::left("Home"),
                    Cell::right("Restart"),
                ],
                vec![
                    Cell::left("F1"),
                    Cell::right("Chase/Free"),
                    Cell::left("F2"),
                    Cell::right("Switch Target"),
                ],
                vec![
                    Cell::left("Rear"),
                    Cell::right(format!(
                        "Hold {}",
                        describe_binding_group(&bindings.rear_view)
                    )),
                    Cell::left("Capture"),
                    Cell::right(describe_binding_group(&bindings.toggle_mouse_capture)),
                ],
                vec![
                    Cell::left("Mouse"),
                    Cell::right("Look in Free"),
                    Cell::left("Move"),
                    Cell::right("WASD/QE"),
                ],
                vec![
                    Cell::left("Shift"),
                    Cell::right("Accelerate"),
                    Cell::left("Guide"),
                    Cell::right(describe_binding_group(&bindings.toggle_controls_guide)),
                ],
                vec![
                    Cell::left("F4"),
                    Cell::right("Flight Assist"),
                    Cell::left(""),
                    Cell::right(""),
                ],
            ];
            format!(
                "OBSERVER REPLAY\n{}",
                render_rows_with_min_widths(&rows, 2, &[8, 18, 8, 18]),
            )
        }
    };

    if let Ok(mut text) = query.single_mut() {
        text.0 = guide;
    }
}

fn describe_axis_pair(positive: &ActionBindingsConfig, negative: &ActionBindingsConfig) -> String {
    format!(
        "{}/{}",
        primary_binding_label(positive),
        primary_binding_label(negative)
    )
}

fn describe_binding_group(bindings: &ActionBindingsConfig) -> String {
    let mut labels = Vec::new();
    if let Some(value) = bindings.keyboard_primary.as_deref() {
        labels.push(binding_label(value));
    }
    if let Some(value) = bindings.keyboard_secondary.as_deref() {
        labels.push(binding_label(value));
    }
    if let Some(value) = bindings.mouse_primary.as_deref() {
        labels.push(binding_label(value));
    }
    if let Some(value) = bindings.mouse_secondary.as_deref() {
        labels.push(binding_label(value));
    }
    if labels.is_empty() {
        "-".to_string()
    } else {
        labels.join("/")
    }
}

fn primary_binding_label(bindings: &ActionBindingsConfig) -> String {
    bindings
        .keyboard_primary
        .as_deref()
        .or(bindings.mouse_primary.as_deref())
        .map(binding_label)
        .unwrap_or_else(|| "-".to_string())
}

fn describe_mouse_axis(axis_name: &str, target: MouseFlightAxisTarget, invert: bool) -> String {
    let direction = if invert { "inv" } else { "norm" };
    let target = match target {
        MouseFlightAxisTarget::Pitch => "pitch",
        MouseFlightAxisTarget::Roll => "roll",
        MouseFlightAxisTarget::Yaw => "yaw",
    };
    if axis_name.is_empty() {
        format!("{target} ({direction})")
    } else {
        format!("{axis_name} {target} ({direction})")
    }
}

fn binding_label(name: &str) -> String {
    match name {
        "ShiftLeft" => "LShift".to_string(),
        "ShiftRight" => "RShift".to_string(),
        "ControlLeft" => "LCtrl".to_string(),
        "ControlRight" => "RCtrl".to_string(),
        "AltLeft" => "LAlt".to_string(),
        "AltRight" => "RAlt".to_string(),
        "Space" => "Space".to_string(),
        "Tab" => "Tab".to_string(),
        "Enter" => "Enter".to_string(),
        "Backspace" => "Backspace".to_string(),
        "Escape" => "Esc".to_string(),
        "ArrowUp" => "Up".to_string(),
        "ArrowDown" => "Down".to_string(),
        "ArrowLeft" => "Left".to_string(),
        "ArrowRight" => "Right".to_string(),
        "Middle" => "MMB".to_string(),
        "Left" => "LMB".to_string(),
        "Right" => "RMB".to_string(),
        "Forward" => "MouseFwd".to_string(),
        "Back" => "MouseBack".to_string(),
        _ if name.starts_with("Key") && name.len() == 4 => name[3..].to_string(),
        _ if name.starts_with("Digit") && name.len() == 6 => name[5..].to_string(),
        _ => name.to_string(),
    }
}

pub fn update_controls_guide_visibility(
    guide_state: Res<ControlsGuideState>,
    mut query: Query<&mut Visibility, With<ControlsGuideRoot>>,
) {
    if let Ok(mut visibility) = query.single_mut() {
        *visibility = if guide_state.visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

pub fn sync_observer_view_context(
    app_mode: Res<AppMode>,
    observer_mode: Option<Res<ObserverClientMode>>,
    bridge_link: Option<Res<BridgeLinkState>>,
    assigned_role: Option<Res<AssignedControlRole>>,
    mut observer_view: ResMut<ObserverViewContext>,
) {
    observer_view.session = match *app_mode {
        AppMode::Observer => match observer_mode
            .as_deref()
            .copied()
            .unwrap_or(ObserverClientMode::RecordedEpisode)
        {
            ObserverClientMode::RecordedEpisode => ObserverSessionKind::SpectatorReplay,
            ObserverClientMode::LiveServer => ObserverSessionKind::SpectatorLive,
        },
        AppMode::Game => match assigned_role.as_deref().map(|role| role.0) {
            Some(BridgeControlSlot::Spectator) => ObserverSessionKind::SpectatorLive,
            _ if bridge_link
                .as_deref()
                .map(|link| link.remote_authority_active)
                .unwrap_or(false) =>
            {
                ObserverSessionKind::PlayerLive
            }
            _ => ObserverSessionKind::PlayerLocal,
        },
    };
    observer_view.source = match *app_mode {
        AppMode::Observer => match observer_mode
            .as_deref()
            .copied()
            .unwrap_or(ObserverClientMode::RecordedEpisode)
        {
            ObserverClientMode::RecordedEpisode => ObserverDataSourceKind::RecordedEpisode,
            ObserverClientMode::LiveServer => ObserverDataSourceKind::LiveServer,
        },
        AppMode::Game => {
            if bridge_link
                .as_deref()
                .map(|link| link.remote_authority_active)
                .unwrap_or(false)
            {
                ObserverDataSourceKind::LiveServer
            } else {
                ObserverDataSourceKind::LocalSimulation
            }
        }
    };
}

pub fn sync_live_observer_feed_status(
    observer_view: Res<ObserverViewContext>,
    clock: Res<MatchClock>,
    match_phase: Res<State<MatchPhase>>,
    bridge_inbox: Option<Res<BridgeClientInbox>>,
    bridge_state: Option<Res<IpcBridgeState>>,
    mut observer_provider: ResMut<ObserverFeedProviderState>,
    mut observer_feed: ResMut<ObserverFeedStatus>,
) {
    if observer_view.source == ObserverDataSourceKind::RecordedEpisode {
        return;
    }

    let current_tick = bridge_inbox
        .as_deref()
        .and_then(|inbox| inbox.latest_snapshot.as_ref().map(|snapshot| snapshot.tick))
        .or(bridge_state
            .as_deref()
            .and_then(|state| state.last_server_tick));
    apply_live_observer_feed_status(
        &mut observer_provider,
        &mut observer_feed,
        *match_phase.get(),
        clock.elapsed_seconds,
        current_tick,
    );
}

pub fn apply_live_observer_feed_status(
    observer_provider: &mut ObserverFeedProviderState,
    observer_feed: &mut ObserverFeedStatus,
    phase: MatchPhase,
    sim_time_seconds: f32,
    current_tick: Option<u64>,
) {
    observer_provider.provider = Some(ObserverFeedProviderKind::LiveBridge);
    observer_provider.ready = current_tick.is_some();
    observer_feed.phase = Some(phase);
    observer_feed.sim_time_seconds = sim_time_seconds;
    observer_feed.current_step = None;
    observer_feed.total_steps = None;
    observer_feed.playback_finished = None;
    observer_feed.playback_paused = None;
    observer_feed.playback_speed = None;
    observer_feed.current_tick = current_tick;
}

#[derive(Debug, Clone, Copy)]
pub struct RecordedObserverFeedSample {
    pub sim_time_seconds: f32,
    pub current_tick: Option<u64>,
    pub current_step: usize,
    pub total_steps: usize,
    pub playback_finished: bool,
    pub playback_paused: bool,
    pub playback_speed: f32,
}

pub fn apply_recorded_observer_feed_status(
    observer_provider: &mut ObserverFeedProviderState,
    observer_feed: &mut ObserverFeedStatus,
    sample: RecordedObserverFeedSample,
) {
    observer_provider.provider = Some(ObserverFeedProviderKind::RecordedTimeline);
    observer_provider.ready = sample.total_steps > 0;
    observer_feed.phase = Some(MatchPhase::Running);
    observer_feed.sim_time_seconds = sample.sim_time_seconds;
    observer_feed.current_tick = sample.current_tick;
    observer_feed.current_step = Some(sample.current_step);
    observer_feed.total_steps = Some(sample.total_steps);
    observer_feed.playback_finished = Some(sample.playback_finished);
    observer_feed.playback_paused = Some(sample.playback_paused);
    observer_feed.playback_speed = Some(sample.playback_speed);
}

fn observer_session_kind_label(kind: ObserverSessionKind) -> &'static str {
    match kind {
        ObserverSessionKind::PlayerLocal => "LOCAL PILOT",
        ObserverSessionKind::PlayerLive => "LIVE PILOT",
        ObserverSessionKind::SpectatorLive => "LIVE OBS",
        ObserverSessionKind::SpectatorReplay => "REPLAY OBS",
    }
}

fn observer_feed_provider_label(kind: ObserverFeedProviderKind) -> &'static str {
    match kind {
        ObserverFeedProviderKind::LiveBridge => "LIVE",
        ObserverFeedProviderKind::RecordedTimeline => "REPLAY",
    }
}

fn bridge_connection_status_label(status: BridgeClientConnectionStatus) -> &'static str {
    match status {
        BridgeClientConnectionStatus::Disabled => "OFFLINE",
        BridgeClientConnectionStatus::Connecting => "CONNECTING",
        BridgeClientConnectionStatus::Reconnecting => "RETRYING",
        BridgeClientConnectionStatus::Timeout => "TIMEOUT",
        BridgeClientConnectionStatus::SessionMismatch => "BAD SESSION",
        BridgeClientConnectionStatus::ProtocolMismatch => "BAD BUILD",
        BridgeClientConnectionStatus::Connected => "CONNECTED",
    }
}

fn short_phase_label(phase: MatchPhase) -> &'static str {
    match phase {
        MatchPhase::Loading => "LOAD",
        MatchPhase::Running => "RUN",
        MatchPhase::Finished => "DONE",
    }
}

fn compute_heading_degrees(forward: Vec3) -> f32 {
    forward.x.atan2(forward.z).to_degrees()
}

fn compass_label(degrees: i32) -> String {
    match degrees.rem_euclid(360) {
        0 => "+Z".to_string(),
        90 => "+X".to_string(),
        180 => "-Z".to_string(),
        270 => "-X".to_string(),
        value if value % 30 == 0 => format!("{value:03}"),
        _ => String::new(),
    }
}

fn compute_roll_degrees(orientation: Quat) -> f32 {
    let (_, _, roll) = orientation.to_euler(EulerRot::YXZ);
    roll.to_degrees()
}

fn short_health(subsystem: &crate::gameplay::damage::SubsystemHealth) -> String {
    if subsystem.current <= f32::EPSILON {
        return "OUT".to_string();
    }

    let current = subsystem.current.round() as i32;
    let max = subsystem.max.round() as i32;
    if current >= max {
        "OK".to_string()
    } else {
        format!("{current}/{max}")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compass_label, compute_heading_degrees, damage_indicator_angle_radians, directional_arrow,
    };
    use bevy::prelude::{Quat, Vec3};

    #[test]
    fn heading_uses_clockwise_positive_convention() {
        assert!((compute_heading_degrees(Vec3::Z) - 0.0).abs() < 0.001);
        assert!((compute_heading_degrees(Vec3::X) - 90.0).abs() < 0.001);
        assert!((compute_heading_degrees(Vec3::NEG_X) + 90.0).abs() < 0.001);
    }

    #[test]
    fn compass_axis_labels_match_clockwise_heading() {
        assert_eq!(compass_label(0), "+Z");
        assert_eq!(compass_label(90), "+X");
        assert_eq!(compass_label(180), "-Z");
        assert_eq!(compass_label(270), "-X");
    }

    #[test]
    fn damage_indicator_maps_body_left_to_screen_left() {
        let angle = damage_indicator_angle_radians(Vec3::ZERO, Quat::IDENTITY, Vec3::X)
            .expect("left source should have a direction");

        assert!((angle + std::f32::consts::FRAC_PI_2).abs() < 0.001);
        assert_eq!(directional_arrow(angle), "←");
    }

    #[test]
    fn damage_indicator_maps_body_right_to_screen_right() {
        let angle = damage_indicator_angle_radians(Vec3::ZERO, Quat::IDENTITY, Vec3::NEG_X)
            .expect("right source should have a direction");

        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 0.001);
        assert_eq!(directional_arrow(angle), "→");
    }
}
