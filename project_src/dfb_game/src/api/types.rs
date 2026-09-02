use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::input::actions::{ControlInput, NormalizedControlCommand};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EnvironmentAgentMode {
    #[default]
    External,
    Model,
    BuiltInAi,
    BuiltInAiPrecise,
    BuiltInAiImperfect,
    BuiltInAiTeacher,
    BuiltInAiPassiveBounce,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvironmentAgentControlConfig {
    pub fighter1: EnvironmentAgentMode,
    pub fighter2: EnvironmentAgentMode,
}

impl EnvironmentAgentControlConfig {
    pub fn single_agent_vs_ai() -> Self {
        Self {
            fighter1: EnvironmentAgentMode::External,
            fighter2: EnvironmentAgentMode::BuiltInAi,
        }
    }

    pub fn self_play() -> Self {
        Self {
            fighter1: EnvironmentAgentMode::External,
            fighter2: EnvironmentAgentMode::External,
        }
    }
}

impl Default for EnvironmentAgentControlConfig {
    fn default() -> Self {
        Self::single_agent_vs_ai()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct EnvironmentAction {
    pub throttle: f32,
    #[serde(default)]
    pub brake: bool,
    pub pitch: f32,
    pub roll: f32,
    pub yaw: f32,
    pub fire_gun: bool,
    pub repair: bool,
}

impl From<EnvironmentAction> for ControlInput {
    fn from(value: EnvironmentAction) -> Self {
        ControlInput::from(NormalizedControlCommand::from(value))
    }
}

impl From<ControlInput> for EnvironmentAction {
    fn from(value: ControlInput) -> Self {
        EnvironmentAction::from(NormalizedControlCommand::from(value))
    }
}

impl From<EnvironmentAction> for NormalizedControlCommand {
    fn from(value: EnvironmentAction) -> Self {
        Self {
            throttle: value.throttle,
            brake: value.brake,
            pitch: value.pitch,
            roll: value.roll,
            yaw: value.yaw,
            fire_gun: value.fire_gun,
            repair: value.repair,
        }
        .clamped()
    }
}

impl From<NormalizedControlCommand> for EnvironmentAction {
    fn from(value: NormalizedControlCommand) -> Self {
        let value = value.clamped();
        Self {
            throttle: value.throttle,
            brake: value.brake,
            pitch: value.pitch,
            roll: value.roll,
            yaw: value.yaw,
            fire_gun: value.fire_gun,
            repair: value.repair,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum VisualSensorKind {
    #[default]
    Front,
    Rear,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum PixelFormat {
    #[default]
    Rgb8,
    Rgba8,
    Gray8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum VisualResolutionMode {
    #[default]
    Fixed,
    RuntimeWindow,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub enum VisualCaptureVariant {
    #[default]
    Rgb,
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualSensorConfig {
    pub kind: VisualSensorKind,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub resolution_mode: VisualResolutionMode,
    pub include_hud: bool,
    #[serde(default)]
    pub capture_variants: Vec<VisualCaptureVariant>,
}

impl Default for VisualSensorConfig {
    fn default() -> Self {
        Self {
            kind: VisualSensorKind::Front,
            width: 96,
            height: 96,
            format: PixelFormat::Rgb8,
            resolution_mode: VisualResolutionMode::Fixed,
            include_hud: false,
            capture_variants: Vec::new(),
        }
    }
}

impl VisualSensorConfig {
    pub fn requested_capture_variants(&self) -> Vec<VisualCaptureVariant> {
        if self.capture_variants.is_empty() {
            vec![VisualCaptureVariant::Rgb]
        } else {
            self.capture_variants.clone()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioFeatureVector {
    pub left_right_energy: f32,
    pub front_back_energy: f32,
    pub engine_energy: f32,
    pub gunfire_energy: f32,
    pub hit_energy: f32,
    pub flyby_energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AudioObservation {
    pub sample_rate: u32,
    pub channels: u16,
    pub window_seconds: f32,
    pub samples: Vec<f32>,
    pub features: AudioFeatureVector,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisualObservation {
    pub camera: VisualSensorKind,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub resolution_mode: VisualResolutionMode,
    pub include_hud: bool,
    pub bytes_ready: bool,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default, Resource, Serialize, Deserialize)]
pub struct ObservationCaptureConfig {
    pub enable_visual: bool,
    pub enable_audio: bool,
    pub visual_sensors: Vec<VisualSensorConfig>,
    pub audio_window_seconds: f32,
}

impl From<&EnvironmentResetOptions> for ObservationCaptureConfig {
    fn from(value: &EnvironmentResetOptions) -> Self {
        Self {
            enable_visual: value.enable_visual,
            enable_audio: value.enable_audio,
            visual_sensors: value.visual_sensors.clone(),
            audio_window_seconds: value.audio_window_seconds,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubsystemObservation {
    pub name: String,
    pub hit_points: f32,
    pub max_hit_points: f32,
    pub stage: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AircraftObservation {
    pub role: String,
    pub position: [f32; 3],
    pub orientation_quat: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity_deg: [f32; 3],
    pub forward: [f32; 3],
    pub throttle: f32,
    #[serde(default)]
    pub brake: bool,
    pub stall_factor: f32,
    pub hit_points: f32,
    pub destroyed: bool,
    pub out_of_bounds_seconds: f32,
    pub ceiling_recovery_seconds: f32,
    pub gun_heat: f32,
    pub gun_overheated: bool,
    #[serde(default)]
    pub is_firing: bool,
    pub repairing: bool,
    #[serde(default)]
    pub repair_elapsed_seconds: f32,
    pub repair_progress: f32,
    #[serde(default)]
    pub velocity_turn_rate_rad_s: Option<f32>,
    #[serde(default)]
    pub pullup_turn_radius_m: Option<f32>,
    #[serde(default)]
    pub max_level_speed_mps: Option<f32>,
    #[serde(default)]
    pub time_to_ground_impact_s: Option<f32>,
    #[serde(default)]
    pub time_to_ceiling_impact_s: Option<f32>,
    #[serde(default)]
    pub time_to_horizontal_boundary_impact_s: Option<f32>,
    #[serde(default)]
    pub time_to_reenter_arena_s: Option<f32>,
    pub subsystems: Vec<SubsystemObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArenaObservation {
    pub ground_height: f32,
    pub arena_radius: f32,
    pub flight_ceiling_height: f32,
    pub ceiling_falloff_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectileObservation {
    pub id: u64,
    pub shooter_role: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub remaining_distance: f32,
    pub damage: f32,
    pub hit_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TracerObservation {
    pub position: [f32; 3],
    pub remaining_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicWorldObservation {
    pub projectiles: Vec<ProjectileObservation>,
    pub tracers: Vec<TracerObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StateObservation {
    pub tick: u64,
    pub seed: u64,
    pub sim_time_seconds: f32,
    pub match_phase: String,
    pub scene_name: String,
    pub aircraft: Vec<AircraftObservation>,
    pub arena: ArenaObservation,
    pub events_since_last_step: Vec<EnvironmentEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, Resource)]
pub struct ObservationBundle {
    pub state: StateObservation,
    pub dynamic: DynamicWorldObservation,
    pub visual: Vec<VisualObservation>,
    pub audio: Option<AudioObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnvironmentEventKind {
    Hit,
    Damage,
    SubsystemHit,
    SubsystemDestroyed,
    Collision,
    Destroy,
    OutOfBounds,
    RepairStarted,
    RepairCompleted,
    GunFired,
    GunOverheated,
    GunCooled,
    StallEntered,
    StallRecovered,
    Kill,
    Win,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentEvent {
    pub tick: u64,
    pub kind: String,
    pub subject: Option<String>,
    #[serde(default)]
    pub other_subject: Option<String>,
    pub position: Option<[f32; 3]>,
    pub magnitude: Option<f32>,
    #[serde(default)]
    pub event_detail: Option<String>,
    #[serde(default)]
    pub subsystem: Option<String>,
}

impl EnvironmentEvent {
    pub fn new(
        tick: u64,
        kind: EnvironmentEventKind,
        subject: Option<String>,
        position: Option<[f32; 3]>,
        magnitude: Option<f32>,
    ) -> Self {
        Self {
            tick,
            kind: format!("{kind:?}"),
            subject,
            other_subject: None,
            position,
            magnitude,
            event_detail: None,
            subsystem: None,
        }
    }

    pub fn with_context(
        tick: u64,
        kind: EnvironmentEventKind,
        subject: Option<String>,
        other_subject: Option<String>,
        position: Option<[f32; 3]>,
        magnitude: Option<f32>,
        event_detail: Option<String>,
        subsystem: Option<String>,
    ) -> Self {
        Self {
            tick,
            kind: format!("{kind:?}"),
            subject,
            other_subject,
            position,
            magnitude,
            event_detail,
            subsystem,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EnvironmentEvent, EnvironmentEventKind};

    #[test]
    fn environment_event_with_context_populates_optional_fields() {
        let event = EnvironmentEvent::with_context(
            42,
            EnvironmentEventKind::Collision,
            Some("fighter1".to_string()),
            Some("fighter2".to_string()),
            Some([1.0, 2.0, 3.0]),
            Some(5.0),
            Some("aircraft".to_string()),
            Some("engine".to_string()),
        );

        assert_eq!(event.tick, 42);
        assert_eq!(event.kind, "Collision");
        assert_eq!(event.subject.as_deref(), Some("fighter1"));
        assert_eq!(event.other_subject.as_deref(), Some("fighter2"));
        assert_eq!(event.event_detail.as_deref(), Some("aircraft"));
        assert_eq!(event.subsystem.as_deref(), Some("engine"));
        assert_eq!(event.position, Some([1.0, 2.0, 3.0]));
        assert_eq!(event.magnitude, Some(5.0));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentResetOptions {
    pub scene_name: Option<String>,
    #[serde(default)]
    pub scene_path: Option<String>,
    pub seed: Option<u64>,
    pub enable_visual: bool,
    pub enable_audio: bool,
    pub visual_sensors: Vec<VisualSensorConfig>,
    pub audio_window_seconds: f32,
    pub ticks_per_step: u32,
    #[serde(default)]
    pub agent_control: EnvironmentAgentControlConfig,
}

impl Default for EnvironmentResetOptions {
    fn default() -> Self {
        Self {
            scene_name: None,
            scene_path: None,
            seed: None,
            enable_visual: true,
            enable_audio: true,
            visual_sensors: vec![
                VisualSensorConfig::default(),
                VisualSensorConfig {
                    kind: VisualSensorKind::Rear,
                    ..VisualSensorConfig::default()
                },
            ],
            audio_window_seconds: 0.25,
            ticks_per_step: 1,
            agent_control: EnvironmentAgentControlConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentEpisodeStatus {
    pub tick: u64,
    pub sim_time_seconds: f32,
    pub match_phase: String,
    pub scene_name: String,
    pub terminated: bool,
    pub truncated: bool,
    pub winner: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentRecordingStatus {
    pub active: bool,
    pub pending_start: bool,
    pub pending_stop: bool,
    pub active_step_count: usize,
    pub last_saved_path: Option<String>,
    pub last_saved_frame_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepInfo {
    pub tick: u64,
    pub sim_time_seconds: f32,
    pub winner: Option<String>,
    pub events: Vec<EnvironmentEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepResult {
    pub observation: ObservationBundle,
    pub reward: Option<f32>,
    pub terminated: bool,
    pub truncated: bool,
    pub info: StepInfo,
}
