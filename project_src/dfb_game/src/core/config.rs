use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) const STANDARD_WEIGHT_KG: f32 = 9000.0;

#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct RepositoryConfig {
    pub game: GameConfig,
    pub input: InputConfig,
    pub scene: SceneConfig,
    pub fighter1_aircraft_spec: AircraftSpecConfig,
    pub fighter2_aircraft_spec: AircraftSpecConfig,
    pub fighter1_aircraft: AircraftConfig,
    pub fighter2_aircraft: AircraftConfig,
}

impl Default for RepositoryConfig {
    fn default() -> Self {
        Self {
            game: GameConfig::default(),
            input: InputConfig::default(),
            scene: SceneConfig::default(),
            fighter1_aircraft_spec: AircraftSpecConfig::default_fighter1(),
            fighter2_aircraft_spec: AircraftSpecConfig::default_fighter2(),
            fighter1_aircraft: AircraftConfig::from_spec(&AircraftSpecConfig::default_fighter1()),
            fighter2_aircraft: AircraftConfig::from_spec(&AircraftSpecConfig::default_fighter2()),
        }
    }
}

impl RepositoryConfig {
    pub fn load_from_root(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_from_root_with_scene_spec(root, None, None)
    }

    pub fn load_from_root_with_scene(
        root: impl AsRef<Path>,
        scene_override: Option<&str>,
    ) -> Result<Self> {
        Self::load_from_root_with_scene_spec(root, scene_override, None)
    }

    pub fn load_from_root_with_scene_path(
        root: impl AsRef<Path>,
        scene_path_override: Option<&Path>,
    ) -> Result<Self> {
        Self::load_from_root_with_scene_spec(root, None, scene_path_override)
    }

    pub fn load_from_root_with_scene_spec(
        root: impl AsRef<Path>,
        scene_override: Option<&str>,
        scene_path_override: Option<&Path>,
    ) -> Result<Self> {
        let root = root.as_ref();
        let mut game: GameConfig = load_ron(root.join("config/dfb_game/game.ron"))?;
        let scene_name = scene_override
            .map(ToOwned::to_owned)
            .or_else(|| {
                scene_path_override.and_then(|path| {
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(ToOwned::to_owned)
                })
            })
            .unwrap_or_else(|| game.active_scene.clone());
        game.active_scene = scene_name.clone();
        let standard_aircraft_spec = load_aircraft_spec(root, "standard.ron")?;
        let scene_path = match scene_path_override {
            Some(path) => path.to_path_buf(),
            None => root.join(format!("config/dfb_game/scenes/{scene_name}.ron")),
        };
        Ok(Self {
            game,
            input: load_ron(root.join("config/dfb_game/input.ron"))?,
            scene: load_ron(scene_path)?,
            fighter1_aircraft: AircraftConfig::from_spec(&standard_aircraft_spec),
            fighter2_aircraft: AircraftConfig::from_spec(&standard_aircraft_spec),
            fighter1_aircraft_spec: standard_aircraft_spec.clone(),
            fighter2_aircraft_spec: standard_aircraft_spec,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneConfig {
    pub arena_radius: f32,
    pub ground_height: f32,
    pub flight_ceiling_height: f32,
    pub ceiling_falloff_range: f32,
    #[serde(default = "default_out_of_bounds_grace_seconds")]
    pub out_of_bounds_grace_seconds: f32,
    pub fighter1_spawn: SpawnPointConfig,
    pub fighter2_spawn: SpawnPointConfig,
    #[serde(default)]
    pub obstacles: Vec<BoxObstacleConfig>,
    #[serde(default)]
    pub ground_clutter: Vec<GroundClutterConfig>,
    #[serde(default)]
    pub ground_accents: Vec<GroundAccentConfig>,
    #[serde(default)]
    pub sky_markers: Vec<SkyMarkerConfig>,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            arena_radius: 12_000.0,
            ground_height: 0.0,
            flight_ceiling_height: 1_200.0,
            ceiling_falloff_range: 250.0,
            out_of_bounds_grace_seconds: default_out_of_bounds_grace_seconds(),
            fighter1_spawn: SpawnPointConfig {
                position: [0.0, 600.0, -400.0],
                rotation_degrees: [0.0, 0.0, 0.0],
                initial_speed: default_spawn_initial_speed(),
                initial_throttle: default_spawn_initial_throttle(),
                initial_damage: None,
            },
            fighter2_spawn: SpawnPointConfig {
                position: [220.0, 600.0, 1500.0],
                rotation_degrees: [0.0, 180.0, 0.0],
                initial_speed: default_spawn_initial_speed(),
                initial_throttle: default_spawn_initial_throttle(),
                initial_damage: None,
            },
            obstacles: vec![
                BoxObstacleConfig {
                    position: [-1750.0, 170.0, -1100.0],
                    size: [120.0, 340.0, 120.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [-1200.0, 210.0, 900.0],
                    size: [160.0, 420.0, 160.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [-350.0, 180.0, 1500.0],
                    size: [150.0, 360.0, 150.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [650.0, 145.0, -1450.0],
                    size: [140.0, 290.0, 140.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [1250.0, 195.0, 1100.0],
                    size: [170.0, 390.0, 170.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [1850.0, 150.0, -250.0],
                    size: [130.0, 300.0, 130.0],
                    color: [0.42, 0.38, 0.33, 1.0],
                    damage_scale: 1.05,
                },
                BoxObstacleConfig {
                    position: [-250.0, 115.0, 300.0],
                    size: [520.0, 230.0, 360.0],
                    color: [0.55, 0.48, 0.39, 1.0],
                    damage_scale: 0.9,
                },
                BoxObstacleConfig {
                    position: [1500.0, 105.0, 1800.0],
                    size: [460.0, 210.0, 340.0],
                    color: [0.55, 0.48, 0.39, 1.0],
                    damage_scale: 0.9,
                },
                BoxObstacleConfig {
                    position: [-1650.0, 95.0, 1700.0],
                    size: [420.0, 190.0, 300.0],
                    color: [0.55, 0.48, 0.39, 1.0],
                    damage_scale: 0.9,
                },
            ],
            ground_clutter: vec![
                GroundClutterConfig {
                    position: [-1900.0, 0.0, -1600.0],
                    footprint: [140.0, 100.0],
                    height: 18.0,
                    color: [0.36, 0.48, 0.25, 1.0],
                    kind: GroundClutterKind::GrassPatch,
                },
                GroundClutterConfig {
                    position: [-1400.0, 0.0, -500.0],
                    footprint: [110.0, 80.0],
                    height: 26.0,
                    color: [0.34, 0.42, 0.22, 1.0],
                    kind: GroundClutterKind::ShrubCluster,
                },
                GroundClutterConfig {
                    position: [-900.0, 0.0, 1400.0],
                    footprint: [120.0, 90.0],
                    height: 34.0,
                    color: [0.30, 0.39, 0.20, 1.0],
                    kind: GroundClutterKind::TreeStand,
                },
                GroundClutterConfig {
                    position: [-250.0, 0.0, -1200.0],
                    footprint: [150.0, 110.0],
                    height: 20.0,
                    color: [0.38, 0.52, 0.28, 1.0],
                    kind: GroundClutterKind::GrassPatch,
                },
                GroundClutterConfig {
                    position: [250.0, 0.0, 900.0],
                    footprint: [100.0, 70.0],
                    height: 30.0,
                    color: [0.30, 0.40, 0.22, 1.0],
                    kind: GroundClutterKind::ShrubCluster,
                },
                GroundClutterConfig {
                    position: [650.0, 0.0, -350.0],
                    footprint: [140.0, 100.0],
                    height: 38.0,
                    color: [0.28, 0.37, 0.19, 1.0],
                    kind: GroundClutterKind::TreeStand,
                },
                GroundClutterConfig {
                    position: [1200.0, 0.0, 1550.0],
                    footprint: [130.0, 90.0],
                    height: 24.0,
                    color: [0.36, 0.48, 0.25, 1.0],
                    kind: GroundClutterKind::GrassPatch,
                },
                GroundClutterConfig {
                    position: [1800.0, 0.0, 250.0],
                    footprint: [110.0, 75.0],
                    height: 28.0,
                    color: [0.31, 0.41, 0.22, 1.0],
                    kind: GroundClutterKind::ShrubCluster,
                },
                GroundClutterConfig {
                    position: [2100.0, 0.0, -1500.0],
                    footprint: [150.0, 110.0],
                    height: 36.0,
                    color: [0.27, 0.35, 0.19, 1.0],
                    kind: GroundClutterKind::TreeStand,
                },
            ],
            ground_accents: vec![
                GroundAccentConfig {
                    position: [-1650.0, 0.0, -1350.0],
                    size: [240.0, 12.0, 170.0],
                    color: [0.42, 0.31, 0.19, 1.0],
                    kind: GroundAccentKind::SoilPatch,
                },
                GroundAccentConfig {
                    position: [-850.0, 0.0, 200.0],
                    size: [170.0, 20.0, 140.0],
                    color: [0.46, 0.34, 0.22, 1.0],
                    kind: GroundAccentKind::RockCluster,
                },
                GroundAccentConfig {
                    position: [150.0, 0.0, -1650.0],
                    size: [260.0, 12.0, 190.0],
                    color: [0.41, 0.30, 0.18, 1.0],
                    kind: GroundAccentKind::SoilPatch,
                },
                GroundAccentConfig {
                    position: [720.0, 0.0, 520.0],
                    size: [180.0, 22.0, 130.0],
                    color: [0.48, 0.37, 0.24, 1.0],
                    kind: GroundAccentKind::RockCluster,
                },
                GroundAccentConfig {
                    position: [1380.0, 0.0, 1650.0],
                    size: [250.0, 12.0, 180.0],
                    color: [0.40, 0.29, 0.18, 1.0],
                    kind: GroundAccentKind::SoilPatch,
                },
                GroundAccentConfig {
                    position: [1920.0, 0.0, -980.0],
                    size: [160.0, 18.0, 120.0],
                    color: [0.47, 0.36, 0.24, 1.0],
                    kind: GroundAccentKind::RockCluster,
                },
            ],
            sky_markers: vec![
                SkyMarkerConfig {
                    position: [-1000.0, 1500.0, 900.0],
                    radius: 45.0,
                },
                SkyMarkerConfig {
                    position: [900.0, 1850.0, 1150.0],
                    radius: 45.0,
                },
                SkyMarkerConfig {
                    position: [-600.0, 2100.0, 2100.0],
                    radius: 45.0,
                },
                SkyMarkerConfig {
                    position: [1450.0, 1950.0, -300.0],
                    radius: 45.0,
                },
                SkyMarkerConfig {
                    position: [-1750.0, 1750.0, 300.0],
                    radius: 45.0,
                },
            ],
        }
    }
}

fn default_out_of_bounds_grace_seconds() -> f32 {
    20.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPointConfig {
    pub position: [f32; 3],
    pub rotation_degrees: [f32; 3],
    #[serde(default = "default_spawn_initial_speed")]
    pub initial_speed: f32,
    #[serde(default = "default_spawn_initial_throttle")]
    pub initial_throttle: f32,
    #[serde(default)]
    pub initial_damage: Option<SpawnDamageConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpawnDamageConfig {
    #[serde(default)]
    pub total_hit_points_fraction: Option<f32>,
    #[serde(default)]
    pub left_wing_fraction: Option<f32>,
    #[serde(default)]
    pub right_wing_fraction: Option<f32>,
    #[serde(default)]
    pub pitch_tail_fraction: Option<f32>,
    #[serde(default)]
    pub yaw_tail_fraction: Option<f32>,
    #[serde(default)]
    pub engine_fraction: Option<f32>,
}

fn default_spawn_initial_speed() -> f32 {
    75.0
}

fn default_spawn_initial_throttle() -> f32 {
    0.6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxObstacleConfig {
    pub position: [f32; 3],
    pub size: [f32; 3],
    pub color: [f32; 4],
    pub damage_scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundClutterConfig {
    pub position: [f32; 3],
    pub footprint: [f32; 2],
    pub height: f32,
    pub color: [f32; 4],
    pub kind: GroundClutterKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GroundClutterKind {
    GrassPatch,
    ShrubCluster,
    TreeStand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundAccentConfig {
    pub position: [f32; 3],
    pub size: [f32; 3],
    pub color: [f32; 4],
    pub kind: GroundAccentKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GroundAccentKind {
    SoilPatch,
    RockCluster,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkyMarkerConfig {
    pub position: [f32; 3],
    pub radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub fixed_time_step_seconds: f32,
    #[serde(default = "default_match_time_limit_seconds")]
    pub match_time_limit_seconds: Option<f32>,
    #[serde(default = "default_active_scene")]
    pub active_scene: String,
    #[serde(default = "default_ceiling_recovery_duration_seconds")]
    pub ceiling_recovery_duration_seconds: f32,
    #[serde(default = "default_ceiling_recovery_min_dive_angle_deg")]
    pub ceiling_recovery_min_dive_angle_deg: f32,
    #[serde(default = "default_ceiling_recovery_release_stall_factor")]
    pub ceiling_recovery_release_stall_factor: f32,
    #[serde(default = "default_ceiling_soft_recovery_stall_threshold")]
    pub ceiling_soft_recovery_stall_threshold: f32,
    #[serde(default = "default_ceiling_soft_recovery_stall_full_factor")]
    pub ceiling_soft_recovery_stall_full_factor: f32,
    #[serde(default = "default_ceiling_soft_recovery_vertical_speed_threshold")]
    pub ceiling_soft_recovery_vertical_speed_threshold: f32,
    #[serde(default = "default_ceiling_descent_relief_speed")]
    pub ceiling_descent_relief_speed: f32,
    #[serde(default = "default_ceiling_descent_relief_factor")]
    pub ceiling_descent_relief_factor: f32,
    #[serde(default = "default_ceiling_recovery_angular_damping_factor")]
    pub ceiling_recovery_angular_damping_factor: f32,
    pub repair_duration_seconds: f32,
    pub repair_heal_fraction: f32,
    #[serde(default)]
    pub audio: AudioConfig,
    pub camera: CameraConfig,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            fixed_time_step_seconds: 1.0 / 60.0,
            match_time_limit_seconds: default_match_time_limit_seconds(),
            active_scene: default_active_scene(),
            ceiling_recovery_duration_seconds: 5.0,
            ceiling_recovery_min_dive_angle_deg: 30.0,
            ceiling_recovery_release_stall_factor: 0.2,
            ceiling_soft_recovery_stall_threshold: 0.7,
            ceiling_soft_recovery_stall_full_factor: 1.0,
            ceiling_soft_recovery_vertical_speed_threshold: -40.0,
            ceiling_descent_relief_speed: 120.0,
            ceiling_descent_relief_factor: 0.85,
            ceiling_recovery_angular_damping_factor: 0.35,
            repair_duration_seconds: 10.0,
            repair_heal_fraction: 0.25,
            audio: AudioConfig::default(),
            camera: CameraConfig::default(),
        }
    }
}

fn default_match_time_limit_seconds() -> Option<f32> {
    None
}

fn default_active_scene() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "default_audio_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub preferred_output_device: Option<String>,
    #[serde(default = "default_audio_master_volume")]
    pub master_volume: f32,
    #[serde(default = "default_player_engine_volume")]
    pub player_engine_volume: f32,
    #[serde(default = "default_enemy_engine_volume")]
    pub enemy_engine_volume: f32,
    #[serde(default = "default_gun_fire_volume")]
    pub gun_fire_volume: f32,
    #[serde(default = "default_bullet_flyby_volume")]
    pub bullet_flyby_volume: f32,
    #[serde(default = "default_hit_volume")]
    pub hit_volume: f32,
    #[serde(default = "default_audio_spatial_near_distance")]
    pub spatial_near_distance: f32,
    #[serde(default = "default_audio_spatial_far_distance")]
    pub spatial_far_distance: f32,
    #[serde(default = "default_audio_spatial_smoothing")]
    pub spatial_smoothing: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: default_audio_enabled(),
            preferred_output_device: None,
            master_volume: default_audio_master_volume(),
            player_engine_volume: default_player_engine_volume(),
            enemy_engine_volume: default_enemy_engine_volume(),
            gun_fire_volume: default_gun_fire_volume(),
            bullet_flyby_volume: default_bullet_flyby_volume(),
            hit_volume: default_hit_volume(),
            spatial_near_distance: default_audio_spatial_near_distance(),
            spatial_far_distance: default_audio_spatial_far_distance(),
            spatial_smoothing: default_audio_spatial_smoothing(),
        }
    }
}

fn default_audio_enabled() -> bool {
    true
}

fn default_audio_master_volume() -> f32 {
    0.7
}

fn default_player_engine_volume() -> f32 {
    0.18
}

fn default_enemy_engine_volume() -> f32 {
    0.9
}

fn default_gun_fire_volume() -> f32 {
    0.42
}

fn default_bullet_flyby_volume() -> f32 {
    0.34
}

fn default_hit_volume() -> f32 {
    0.40
}

fn default_audio_spatial_near_distance() -> f32 {
    80.0
}

fn default_audio_spatial_far_distance() -> f32 {
    1200.0
}

fn default_audio_spatial_smoothing() -> f32 {
    10.0
}

fn default_ceiling_recovery_duration_seconds() -> f32 {
    5.0
}

fn default_ceiling_recovery_min_dive_angle_deg() -> f32 {
    30.0
}

fn default_ceiling_recovery_release_stall_factor() -> f32 {
    0.2
}

fn default_ceiling_soft_recovery_stall_threshold() -> f32 {
    0.7
}

fn default_ceiling_soft_recovery_stall_full_factor() -> f32 {
    1.0
}

fn default_ceiling_soft_recovery_vertical_speed_threshold() -> f32 {
    -40.0
}

fn default_ceiling_descent_relief_speed() -> f32 {
    120.0
}

fn default_ceiling_descent_relief_factor() -> f32 {
    0.85
}

fn default_ceiling_recovery_angular_damping_factor() -> f32 {
    0.35
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub fov_x_degrees: f32,
    pub fov_y_degrees: f32,
    pub aspect_width: u32,
    pub aspect_height: u32,
    pub follow_offset: [f32; 3],
    pub rear_view_offset: [f32; 3],
    pub sky_color: [f32; 4],
    pub fog_color: [f32; 4],
    pub fog_visibility: f32,
    pub fog_light_color: [f32; 4],
    pub fog_light_exponent: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            fov_x_degrees: 105.0,
            fov_y_degrees: 105.0,
            aspect_width: 4,
            aspect_height: 3,
            follow_offset: [0.0, 6.0, -18.0],
            rear_view_offset: [0.0, 9.0, 24.0],
            sky_color: [0.57, 0.72, 0.92, 1.0],
            fog_color: [0.66, 0.77, 0.92, 0.82],
            fog_visibility: 12_000.0,
            fog_light_color: [1.0, 0.94, 0.84, 0.28],
            fog_light_exponent: 14.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    pub bindings: InputBindingsConfig,
    pub mouse_x_axis: MouseAxisConfig,
    pub mouse_y_axis: MouseAxisConfig,
    pub keyboard_throttle_weight: f32,
    pub keyboard_pitch_weight: f32,
    pub keyboard_roll_weight: f32,
    pub mouse_smoothing: f32,
    pub capture_mouse_on_start: bool,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            bindings: InputBindingsConfig::default(),
            mouse_x_axis: MouseAxisConfig {
                target: MouseFlightAxisTarget::Roll,
                sensitivity: 0.055,
                weight: 1.0,
                invert: false,
            },
            mouse_y_axis: MouseAxisConfig {
                target: MouseFlightAxisTarget::Pitch,
                sensitivity: 0.05,
                weight: 1.0,
                invert: false,
            },
            keyboard_throttle_weight: 1.0,
            keyboard_pitch_weight: 0.45,
            keyboard_roll_weight: 0.55,
            mouse_smoothing: 18.0,
            capture_mouse_on_start: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MouseFlightAxisTarget {
    Pitch,
    Roll,
    Yaw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouseAxisConfig {
    pub target: MouseFlightAxisTarget,
    pub sensitivity: f32,
    pub weight: f32,
    pub invert: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputBindingsConfig {
    pub throttle_up: ActionBindingsConfig,
    pub throttle_down: ActionBindingsConfig,
    pub brake: ActionBindingsConfig,
    pub pitch_positive: ActionBindingsConfig,
    pub pitch_negative: ActionBindingsConfig,
    pub roll_positive: ActionBindingsConfig,
    pub roll_negative: ActionBindingsConfig,
    pub yaw_positive: ActionBindingsConfig,
    pub yaw_negative: ActionBindingsConfig,
    pub fire_gun: ActionBindingsConfig,
    pub repair_aircraft: ActionBindingsConfig,
    pub toggle_controls_guide: ActionBindingsConfig,
    pub reset_match: ActionBindingsConfig,
    pub rear_view: ActionBindingsConfig,
    pub toggle_local_pilot_mode: ActionBindingsConfig,
    pub toggle_audio_mute: ActionBindingsConfig,
    pub toggle_mouse_capture: ActionBindingsConfig,
}

impl Default for InputBindingsConfig {
    fn default() -> Self {
        Self {
            throttle_up: ActionBindingsConfig::keyboard("ShiftLeft"),
            throttle_down: ActionBindingsConfig::keyboard("ControlLeft"),
            brake: ActionBindingsConfig::keyboard("Space"),
            pitch_positive: ActionBindingsConfig::keyboard("KeyS"),
            pitch_negative: ActionBindingsConfig::keyboard("KeyW"),
            roll_positive: ActionBindingsConfig::keyboard("KeyE"),
            roll_negative: ActionBindingsConfig::keyboard("KeyQ"),
            yaw_positive: ActionBindingsConfig::keyboard("KeyA"),
            yaw_negative: ActionBindingsConfig::keyboard("KeyD"),
            fire_gun: ActionBindingsConfig::mouse("Left"),
            repair_aircraft: ActionBindingsConfig::keyboard("KeyX"),
            toggle_controls_guide: ActionBindingsConfig::keyboard("KeyH"),
            reset_match: ActionBindingsConfig::keyboard("KeyR"),
            rear_view: ActionBindingsConfig::keyboard("KeyC"),
            toggle_local_pilot_mode: ActionBindingsConfig::keyboard("F3"),
            toggle_audio_mute: ActionBindingsConfig::keyboard("KeyM"),
            toggle_mouse_capture: ActionBindingsConfig::keyboard("Tab"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputBindingConfig {
    Keyboard(String),
    Mouse(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionBindingsConfig {
    pub keyboard_primary: Option<String>,
    pub keyboard_secondary: Option<String>,
    pub mouse_primary: Option<String>,
    pub mouse_secondary: Option<String>,
}

impl ActionBindingsConfig {
    fn keyboard(primary: &str) -> Self {
        Self {
            keyboard_primary: Some(primary.to_string()),
            ..default()
        }
    }

    fn mouse(primary: &str) -> Self {
        Self {
            mouse_primary: Some(primary.to_string()),
            ..default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AircraftSpecConfig {
    pub display_name: String,
    #[serde(
        default = "default_max_level_speed",
        alias = "max_level_speed_mps",
        alias = "max_speed"
    )]
    pub max_level_speed: f32,
    #[serde(
        default = "default_cruise_reference_throttle",
        alias = "cruise_throttle"
    )]
    pub cruise_reference_throttle: f32,
    #[serde(default = "default_maneuver_reference_throttle")]
    pub maneuver_reference_throttle: f32,
    pub throttle_response: f32,
    pub lift_coefficient: f32,
    pub induced_drag_coefficient: f32,
    pub side_drag_coefficient: f32,
    pub gravity_scale: f32,
    pub linear_drag: f32,
    pub brake_drag: f32,
    #[serde(default = "default_weight_kg")]
    pub weight_kg: f32,
    pub stall_speed: f32,
    pub stall_recovery_speed: f32,
    pub pitch_response: f32,
    pub yaw_response: f32,
    pub roll_response: f32,
    pub angular_damping: f32,
    pub pitch_positive_rate_limit_deg: f32,
    pub pitch_negative_rate_limit_deg: f32,
    pub roll_positive_rate_limit_deg: f32,
    pub roll_negative_rate_limit_deg: f32,
    pub yaw_positive_rate_limit_deg: f32,
    pub yaw_negative_rate_limit_deg: f32,
    #[serde(default = "default_pitch_maneuver_scale")]
    pub pitch_maneuver_scale: f32,
    #[serde(default = "default_roll_maneuver_scale")]
    pub roll_maneuver_scale: f32,
    #[serde(default = "default_yaw_maneuver_scale")]
    pub yaw_maneuver_scale: f32,
    #[serde(default = "default_pitch_low_speed_scale")]
    pub pitch_low_speed_scale: f32,
    #[serde(default = "default_roll_low_speed_scale")]
    pub roll_low_speed_scale: f32,
    #[serde(default = "default_yaw_low_speed_scale")]
    pub yaw_low_speed_scale: f32,
    #[serde(default = "default_pitch_high_speed_max_scale")]
    pub pitch_high_speed_max_scale: f32,
    #[serde(default = "default_roll_high_speed_max_scale")]
    pub roll_high_speed_max_scale: f32,
    #[serde(default = "default_yaw_high_speed_max_scale")]
    pub yaw_high_speed_max_scale: f32,
    pub left_wing_hit_points: f32,
    pub right_wing_hit_points: f32,
    pub pitch_tail_hit_points: f32,
    pub yaw_tail_hit_points: f32,
    pub engine_hit_points: f32,
    pub damaged_control_surface_scale: f32,
    pub destroyed_control_surface_scale: f32,
    pub damaged_engine_thrust_scale: f32,
    pub destroyed_engine_thrust_scale: f32,
    pub damaged_engine_throttle_response_min: f32,
    pub damaged_wing_lift_scale: f32,
    pub destroyed_wing_lift_scale: f32,
    pub damage_roll_trim_base_deg: f32,
    pub damage_roll_trim_asymmetry_deg: f32,
    pub damage_yaw_trim_base_deg: f32,
    pub damage_yaw_trim_asymmetry_deg: f32,
    pub damage_extra_drag_per_surface: f32,
    pub damage_extra_drag_asymmetry: f32,
    #[serde(default)]
    pub gun: GunConfig,
    pub hit_points: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AircraftConfig {
    pub display_name: String,
    pub max_level_speed: f32,
    pub cruise_reference_throttle: f32,
    pub max_thrust: f32,
    pub trim_pitch_degrees: f32,
    pub trim_angle_of_attack_radians: f32,
    pub reference_level_lift_factor: f32,
    pub throttle_response: f32,
    pub lift_coefficient: f32,
    pub induced_drag_coefficient: f32,
    pub side_drag_coefficient: f32,
    pub gravity_scale: f32,
    pub linear_drag: f32,
    pub brake_drag: f32,
    pub weight_kg: f32,
    pub stall_speed: f32,
    pub stall_recovery_speed: f32,
    #[serde(default = "default_stall_reference_dynamic_pressure")]
    pub stall_reference_dynamic_pressure: f32,
    #[serde(default = "default_stall_recovery_dynamic_pressure")]
    pub stall_recovery_dynamic_pressure: f32,
    pub pitch_response: f32,
    pub yaw_response: f32,
    pub roll_response: f32,
    pub angular_damping: f32,
    pub pitch_positive_rate_limit_deg: f32,
    pub pitch_negative_rate_limit_deg: f32,
    pub roll_positive_rate_limit_deg: f32,
    pub roll_negative_rate_limit_deg: f32,
    pub yaw_positive_rate_limit_deg: f32,
    pub yaw_negative_rate_limit_deg: f32,
    pub cruise_reference_speed: f32,
    pub maneuver_reference_throttle: f32,
    pub pitch_maneuver_scale: f32,
    pub roll_maneuver_scale: f32,
    pub yaw_maneuver_scale: f32,
    pub pitch_low_speed_scale: f32,
    pub roll_low_speed_scale: f32,
    pub yaw_low_speed_scale: f32,
    pub pitch_high_speed_max_scale: f32,
    pub roll_high_speed_max_scale: f32,
    pub yaw_high_speed_max_scale: f32,
    pub left_wing_hit_points: f32,
    pub right_wing_hit_points: f32,
    pub pitch_tail_hit_points: f32,
    pub yaw_tail_hit_points: f32,
    pub engine_hit_points: f32,
    pub damaged_control_surface_scale: f32,
    pub destroyed_control_surface_scale: f32,
    pub damaged_engine_thrust_scale: f32,
    pub destroyed_engine_thrust_scale: f32,
    pub damaged_engine_throttle_response_min: f32,
    pub damaged_wing_lift_scale: f32,
    pub destroyed_wing_lift_scale: f32,
    pub damage_roll_trim_base_deg: f32,
    pub damage_roll_trim_asymmetry_deg: f32,
    pub damage_yaw_trim_base_deg: f32,
    pub damage_yaw_trim_asymmetry_deg: f32,
    pub damage_extra_drag_per_surface: f32,
    pub damage_extra_drag_asymmetry: f32,
    #[serde(default)]
    pub gun: GunConfig,
    pub hit_points: f32,
}

impl AircraftSpecConfig {
    pub fn default_standard() -> Self {
        Self {
            display_name: "Fighter".to_string(),
            max_level_speed: 80.0,
            cruise_reference_throttle: 0.6,
            throttle_response: 0.8,
            lift_coefficient: 13.6,
            induced_drag_coefficient: 0.05,
            side_drag_coefficient: 0.018,
            gravity_scale: 0.92,
            linear_drag: 0.012,
            brake_drag: 0.035,
            weight_kg: default_weight_kg(),
            stall_speed: 30.0,
            stall_recovery_speed: 40.0,
            pitch_response: 7.5,
            yaw_response: 5.0,
            roll_response: 8.5,
            angular_damping: 2.0,
            pitch_positive_rate_limit_deg: 45.0,
            pitch_negative_rate_limit_deg: 45.0,
            roll_positive_rate_limit_deg: 240.0,
            roll_negative_rate_limit_deg: 240.0,
            yaw_positive_rate_limit_deg: 10.0,
            yaw_negative_rate_limit_deg: 10.0,
            maneuver_reference_throttle: default_maneuver_reference_throttle(),
            pitch_maneuver_scale: default_pitch_maneuver_scale(),
            roll_maneuver_scale: default_roll_maneuver_scale(),
            yaw_maneuver_scale: default_yaw_maneuver_scale(),
            pitch_low_speed_scale: default_pitch_low_speed_scale(),
            roll_low_speed_scale: default_roll_low_speed_scale(),
            yaw_low_speed_scale: default_yaw_low_speed_scale(),
            pitch_high_speed_max_scale: default_pitch_high_speed_max_scale(),
            roll_high_speed_max_scale: default_roll_high_speed_max_scale(),
            yaw_high_speed_max_scale: default_yaw_high_speed_max_scale(),
            left_wing_hit_points: 40.0,
            right_wing_hit_points: 40.0,
            pitch_tail_hit_points: 36.0,
            yaw_tail_hit_points: 36.0,
            engine_hit_points: 48.0,
            damaged_control_surface_scale: 0.55,
            destroyed_control_surface_scale: 0.0,
            damaged_engine_thrust_scale: 0.6,
            destroyed_engine_thrust_scale: 0.0,
            damaged_engine_throttle_response_min: 0.3,
            damaged_wing_lift_scale: 0.82,
            destroyed_wing_lift_scale: 0.05,
            damage_roll_trim_base_deg: 42.0,
            damage_roll_trim_asymmetry_deg: 72.0,
            damage_yaw_trim_base_deg: 5.0,
            damage_yaw_trim_asymmetry_deg: 8.0,
            damage_extra_drag_per_surface: 0.008,
            damage_extra_drag_asymmetry: 0.012,
            gun: GunConfig {
                rounds_per_second: 15.0,
                projectile_speed: 1200.0,
                damage_per_hit: 3.0,
                max_range: 1400.0,
                overheat_fire_seconds: 3.0,
                overheat_cool_seconds: 3.0,
                overheat_resume_fraction: 0.2,
            },
            hit_points: 100.0,
        }
    }

    pub fn default_fighter1() -> Self {
        Self::default_standard()
    }

    pub fn default_fighter2() -> Self {
        Self::default_standard()
    }
}

impl AircraftConfig {
    pub fn default_fighter1() -> Self {
        Self::from_spec(&AircraftSpecConfig::default_fighter1())
    }

    pub fn default_fighter2() -> Self {
        Self::from_spec(&AircraftSpecConfig::default_fighter2())
    }

    pub fn from_spec(spec: &AircraftSpecConfig) -> Self {
        let reference_level_lift_factor =
            reference_level_lift_factor(spec.gravity_scale, spec.lift_coefficient, spec.weight_kg);
        let max_thrust = max_thrust_from_level_speed(
            spec.max_level_speed,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            reference_level_lift_factor,
        );
        let cruise_reference_throttle = spec.cruise_reference_throttle.clamp(0.0, 1.0);
        let cruise_reference_speed = reference_speed_from_parameters(
            max_thrust,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            cruise_reference_throttle,
            reference_level_lift_factor,
        );
        let trim_angle_of_attack_radians =
            trim_angle_of_attack_radians_from_reference_lift_factor(reference_level_lift_factor);
        let trim_pitch_degrees = trim_angle_of_attack_radians.to_degrees();
        let stall_reference_dynamic_pressure = stall_capacity_ratio_from_speed(
            spec.stall_speed,
            cruise_reference_speed,
            spec.gravity_scale,
            spec.weight_kg,
        );
        let stall_recovery_dynamic_pressure = stall_capacity_ratio_from_speed(
            spec.stall_recovery_speed,
            cruise_reference_speed,
            spec.gravity_scale,
            spec.weight_kg,
        );
        Self {
            display_name: spec.display_name.clone(),
            max_level_speed: spec.max_level_speed,
            cruise_reference_throttle,
            max_thrust,
            trim_pitch_degrees,
            trim_angle_of_attack_radians,
            reference_level_lift_factor,
            throttle_response: spec.throttle_response,
            lift_coefficient: spec.lift_coefficient,
            induced_drag_coefficient: spec.induced_drag_coefficient,
            side_drag_coefficient: spec.side_drag_coefficient,
            gravity_scale: spec.gravity_scale,
            linear_drag: spec.linear_drag,
            brake_drag: spec.brake_drag,
            weight_kg: spec.weight_kg,
            stall_speed: spec.stall_speed,
            stall_recovery_speed: spec.stall_recovery_speed,
            stall_reference_dynamic_pressure,
            stall_recovery_dynamic_pressure,
            pitch_response: spec.pitch_response,
            yaw_response: spec.yaw_response,
            roll_response: spec.roll_response,
            angular_damping: spec.angular_damping,
            pitch_positive_rate_limit_deg: spec.pitch_positive_rate_limit_deg,
            pitch_negative_rate_limit_deg: spec.pitch_negative_rate_limit_deg,
            roll_positive_rate_limit_deg: spec.roll_positive_rate_limit_deg,
            roll_negative_rate_limit_deg: spec.roll_negative_rate_limit_deg,
            yaw_positive_rate_limit_deg: spec.yaw_positive_rate_limit_deg,
            yaw_negative_rate_limit_deg: spec.yaw_negative_rate_limit_deg,
            cruise_reference_speed,
            maneuver_reference_throttle: spec.maneuver_reference_throttle,
            pitch_maneuver_scale: spec.pitch_maneuver_scale,
            roll_maneuver_scale: spec.roll_maneuver_scale,
            yaw_maneuver_scale: spec.yaw_maneuver_scale,
            pitch_low_speed_scale: spec.pitch_low_speed_scale,
            roll_low_speed_scale: spec.roll_low_speed_scale,
            yaw_low_speed_scale: spec.yaw_low_speed_scale,
            pitch_high_speed_max_scale: spec.pitch_high_speed_max_scale,
            roll_high_speed_max_scale: spec.roll_high_speed_max_scale,
            yaw_high_speed_max_scale: spec.yaw_high_speed_max_scale,
            left_wing_hit_points: spec.left_wing_hit_points,
            right_wing_hit_points: spec.right_wing_hit_points,
            pitch_tail_hit_points: spec.pitch_tail_hit_points,
            yaw_tail_hit_points: spec.yaw_tail_hit_points,
            engine_hit_points: spec.engine_hit_points,
            damaged_control_surface_scale: spec.damaged_control_surface_scale,
            destroyed_control_surface_scale: spec.destroyed_control_surface_scale,
            damaged_engine_thrust_scale: spec.damaged_engine_thrust_scale,
            destroyed_engine_thrust_scale: spec.destroyed_engine_thrust_scale,
            damaged_engine_throttle_response_min: spec.damaged_engine_throttle_response_min,
            damaged_wing_lift_scale: spec.damaged_wing_lift_scale,
            destroyed_wing_lift_scale: spec.destroyed_wing_lift_scale,
            damage_roll_trim_base_deg: spec.damage_roll_trim_base_deg,
            damage_roll_trim_asymmetry_deg: spec.damage_roll_trim_asymmetry_deg,
            damage_yaw_trim_base_deg: spec.damage_yaw_trim_base_deg,
            damage_yaw_trim_asymmetry_deg: spec.damage_yaw_trim_asymmetry_deg,
            damage_extra_drag_per_surface: spec.damage_extra_drag_per_surface,
            damage_extra_drag_asymmetry: spec.damage_extra_drag_asymmetry,
            gun: spec.gun.clone(),
            hit_points: spec.hit_points,
        }
    }
}

fn default_pitch_low_speed_scale() -> f32 {
    0.94
}

fn default_roll_low_speed_scale() -> f32 {
    0.92
}

fn default_yaw_low_speed_scale() -> f32 {
    0.97
}

fn default_weight_kg() -> f32 {
    STANDARD_WEIGHT_KG
}

fn default_pitch_high_speed_max_scale() -> f32 {
    1.10
}

fn default_roll_high_speed_max_scale() -> f32 {
    1.20
}

fn default_yaw_high_speed_max_scale() -> f32 {
    1.05
}

fn default_pitch_maneuver_scale() -> f32 {
    1.08
}

fn default_roll_maneuver_scale() -> f32 {
    1.16
}

fn default_yaw_maneuver_scale() -> f32 {
    1.04
}

fn default_cruise_reference_throttle() -> f32 {
    0.6
}

fn default_maneuver_reference_throttle() -> f32 {
    0.8
}

fn default_max_level_speed() -> f32 {
    100.0
}

fn default_stall_reference_dynamic_pressure() -> f32 {
    0.066_666_67
}

fn default_stall_recovery_dynamic_pressure() -> f32 {
    0.266_666_68
}

fn reference_speed_from_parameters(
    max_thrust: f32,
    lift_coefficient: f32,
    induced_drag_coefficient: f32,
    linear_drag: f32,
    throttle: f32,
    reference_level_lift_factor: f32,
) -> f32 {
    let thrust = (max_thrust * throttle.clamp(0.05, 1.0)).max(0.1);
    let induced_drag_baseline = reference_induced_drag_baseline(
        lift_coefficient,
        induced_drag_coefficient,
        reference_level_lift_factor,
    );
    let available_linear_thrust = (thrust - induced_drag_baseline).max(0.1);
    (available_linear_thrust / linear_drag.max(0.0001)).sqrt()
}

fn max_thrust_from_level_speed(
    max_level_speed: f32,
    lift_coefficient: f32,
    induced_drag_coefficient: f32,
    linear_drag: f32,
    reference_level_lift_factor: f32,
) -> f32 {
    let induced_drag_baseline = reference_induced_drag_baseline(
        lift_coefficient,
        induced_drag_coefficient,
        reference_level_lift_factor,
    );
    linear_drag.max(0.0001) * max_level_speed.max(1.0).powi(2) + induced_drag_baseline
}

fn dynamic_pressure_from_speed(speed: f32, reference_speed: f32) -> f32 {
    (speed / reference_speed.max(1.0)).powi(2)
}

fn reference_weight_scale(weight_kg: f32) -> f32 {
    (weight_kg / STANDARD_WEIGHT_KG).max(0.1)
}

fn reference_weight_demand(gravity_scale: f32, weight_kg: f32) -> f32 {
    reference_weight_scale(weight_kg) * gravity_scale
}

fn stall_capacity_ratio_from_speed(
    speed: f32,
    reference_speed: f32,
    gravity_scale: f32,
    weight_kg: f32,
) -> f32 {
    dynamic_pressure_from_speed(speed, reference_speed)
        / reference_weight_demand(gravity_scale, weight_kg).max(0.1)
}

fn reference_level_lift_factor(gravity_scale: f32, lift_coefficient: f32, weight_kg: f32) -> f32 {
    (9.81 * gravity_scale * reference_weight_scale(weight_kg) / lift_coefficient.max(0.1))
        .clamp(0.0, 1.5)
}

fn reference_induced_drag_baseline(
    lift_coefficient: f32,
    induced_drag_coefficient: f32,
    reference_level_lift_factor: f32,
) -> f32 {
    (lift_coefficient * reference_level_lift_factor * induced_drag_coefficient).max(0.0)
}

fn trim_angle_of_attack_radians_from_reference_lift_factor(
    reference_level_lift_factor: f32,
) -> f32 {
    ((reference_level_lift_factor - 0.55) / 2.4).clamp(-0.2, 0.2)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GunConfig {
    pub rounds_per_second: f32,
    pub projectile_speed: f32,
    pub damage_per_hit: f32,
    pub max_range: f32,
    pub overheat_fire_seconds: f32,
    pub overheat_cool_seconds: f32,
    pub overheat_resume_fraction: f32,
}

impl Default for GunConfig {
    fn default() -> Self {
        Self {
            rounds_per_second: 12.0,
            projectile_speed: 700.0,
            damage_per_hit: 8.0,
            max_range: 900.0,
            overheat_fire_seconds: 3.0,
            overheat_cool_seconds: 3.0,
            overheat_resume_fraction: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AircraftConfig, AircraftSpecConfig, InputConfig, MouseFlightAxisTarget, STANDARD_WEIGHT_KG,
        max_thrust_from_level_speed, reference_induced_drag_baseline, reference_level_lift_factor,
        reference_speed_from_parameters, stall_capacity_ratio_from_speed,
        trim_angle_of_attack_radians_from_reference_lift_factor,
    };

    #[test]
    fn default_fighter1_trim_pitch_is_positive() {
        let config = AircraftConfig::default_fighter1();
        assert!(config.trim_pitch_degrees > 0.0);
    }

    #[test]
    fn default_input_bindings_preserve_human_roll_intent_under_body_rh_actions() {
        let input = InputConfig::default();
        assert_eq!(
            input.bindings.roll_positive.keyboard_primary.as_deref(),
            Some("KeyE")
        );
        assert_eq!(
            input.bindings.roll_negative.keyboard_primary.as_deref(),
            Some("KeyQ")
        );
        assert_eq!(input.mouse_x_axis.target, MouseFlightAxisTarget::Roll);
        assert!(!input.mouse_x_axis.invert);
    }

    #[test]
    fn default_fighter1_trim_aoa_matches_reference_lift_factor() {
        let config = AircraftConfig::default_fighter1();
        let lift_factor = (0.55 + config.trim_angle_of_attack_radians * 2.4).clamp(0.0, 1.5);
        assert!((lift_factor - config.reference_level_lift_factor).abs() < 1e-6);
    }

    #[test]
    fn standard_weight_reference_lift_factor_matches_legacy_baseline() {
        let config = AircraftConfig::default_fighter1();
        let legacy_baseline =
            (9.81 * config.gravity_scale / config.lift_coefficient.max(0.1)).clamp(0.0, 1.5);
        assert!((config.reference_level_lift_factor - legacy_baseline).abs() < 1e-6);
    }

    #[test]
    fn heavier_weight_increases_reference_lift_factor() {
        let mut heavier = AircraftSpecConfig::default_standard();
        heavier.weight_kg = STANDARD_WEIGHT_KG * 1.2;
        let heavier_config = AircraftConfig::from_spec(&heavier);

        let mut lighter = AircraftSpecConfig::default_standard();
        lighter.weight_kg = STANDARD_WEIGHT_KG * 0.8;
        let lighter_config = AircraftConfig::from_spec(&lighter);

        assert!(
            heavier_config.reference_level_lift_factor > lighter_config.reference_level_lift_factor
        );
    }

    #[test]
    fn reference_lift_factor_scales_with_weight_ratio() {
        let gravity_scale = 1.0;
        let lift_coefficient = 10.8;
        let standard =
            reference_level_lift_factor(gravity_scale, lift_coefficient, STANDARD_WEIGHT_KG);
        let heavier =
            reference_level_lift_factor(gravity_scale, lift_coefficient, STANDARD_WEIGHT_KG * 1.1);
        let lighter =
            reference_level_lift_factor(gravity_scale, lift_coefficient, STANDARD_WEIGHT_KG * 0.9);

        assert!(heavier > standard);
        assert!(lighter < standard);
    }

    #[test]
    fn heavier_weight_increases_trim_angle_of_attack_and_pitch() {
        let mut heavier = AircraftSpecConfig::default_standard();
        heavier.weight_kg = STANDARD_WEIGHT_KG * 1.2;
        let heavier_config = AircraftConfig::from_spec(&heavier);

        let mut lighter = AircraftSpecConfig::default_standard();
        lighter.weight_kg = STANDARD_WEIGHT_KG * 0.8;
        let lighter_config = AircraftConfig::from_spec(&lighter);

        assert!(
            heavier_config.trim_angle_of_attack_radians
                > lighter_config.trim_angle_of_attack_radians
        );
        assert!(heavier_config.trim_pitch_degrees > lighter_config.trim_pitch_degrees);
    }

    #[test]
    fn trim_angle_of_attack_is_explicit_mapping_from_reference_lift_factor() {
        let config = AircraftConfig::default_fighter1();
        let mapped = trim_angle_of_attack_radians_from_reference_lift_factor(
            config.reference_level_lift_factor,
        );
        assert!((mapped - config.trim_angle_of_attack_radians).abs() < 1e-6);
    }

    #[test]
    fn standard_weight_reference_speed_and_thrust_match_legacy_baseline() {
        let spec = AircraftSpecConfig::default_standard();
        let legacy_reference_lift_factor =
            (9.81 * spec.gravity_scale / spec.lift_coefficient.max(0.1)).clamp(0.0, 1.5);
        let legacy_induced_drag = reference_induced_drag_baseline(
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            legacy_reference_lift_factor,
        );
        let legacy_max_thrust = spec.linear_drag.max(0.0001)
            * spec.max_level_speed.max(1.0).powi(2)
            + legacy_induced_drag;
        let legacy_cruise_reference_speed = (((legacy_max_thrust
            * spec.cruise_reference_throttle.clamp(0.05, 1.0))
            - legacy_induced_drag)
            .max(0.1)
            / spec.linear_drag.max(0.0001))
        .sqrt();

        let config = AircraftConfig::from_spec(&spec);
        assert!((config.max_thrust - legacy_max_thrust).abs() < 1e-6);
        assert!((config.cruise_reference_speed - legacy_cruise_reference_speed).abs() < 1e-6);
    }

    #[test]
    fn standard_weight_stall_capacity_thresholds_match_weight_adjusted_baseline() {
        let spec = AircraftSpecConfig::default_standard();
        let config = AircraftConfig::from_spec(&spec);
        let legacy_stall_reference = (spec.stall_speed / config.cruise_reference_speed.max(1.0))
            .powi(2)
            / spec.gravity_scale.max(0.1);
        let legacy_stall_recovery =
            (spec.stall_recovery_speed / config.cruise_reference_speed.max(1.0)).powi(2)
                / spec.gravity_scale.max(0.1);

        assert!((config.stall_reference_dynamic_pressure - legacy_stall_reference).abs() < 1e-6);
        assert!((config.stall_recovery_dynamic_pressure - legacy_stall_recovery).abs() < 1e-6);
    }

    #[test]
    fn heavier_weight_increases_reference_thrust_requirement() {
        let mut heavier = AircraftSpecConfig::default_standard();
        heavier.weight_kg = STANDARD_WEIGHT_KG * 1.2;
        let heavier_config = AircraftConfig::from_spec(&heavier);

        let mut lighter = AircraftSpecConfig::default_standard();
        lighter.weight_kg = STANDARD_WEIGHT_KG * 0.8;
        let lighter_config = AircraftConfig::from_spec(&lighter);

        assert!(heavier_config.max_thrust > lighter_config.max_thrust);
    }

    #[test]
    fn heavier_weight_reduces_cruise_reference_speed_at_same_throttle_target() {
        let mut heavier = AircraftSpecConfig::default_standard();
        heavier.weight_kg = STANDARD_WEIGHT_KG * 1.2;
        let heavier_config = AircraftConfig::from_spec(&heavier);

        let mut lighter = AircraftSpecConfig::default_standard();
        lighter.weight_kg = STANDARD_WEIGHT_KG * 0.8;
        let lighter_config = AircraftConfig::from_spec(&lighter);

        assert!(heavier_config.cruise_reference_speed < lighter_config.cruise_reference_speed);
    }

    #[test]
    fn reference_speed_and_thrust_share_reference_lift_factor_semantics() {
        let spec = AircraftSpecConfig::default_standard();
        let lighter_reference_lift_factor = reference_level_lift_factor(
            spec.gravity_scale,
            spec.lift_coefficient,
            STANDARD_WEIGHT_KG * 0.8,
        );
        let heavier_reference_lift_factor = reference_level_lift_factor(
            spec.gravity_scale,
            spec.lift_coefficient,
            STANDARD_WEIGHT_KG * 1.2,
        );

        let lighter_max_thrust = max_thrust_from_level_speed(
            spec.max_level_speed,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            lighter_reference_lift_factor,
        );
        let heavier_max_thrust = max_thrust_from_level_speed(
            spec.max_level_speed,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            heavier_reference_lift_factor,
        );

        let lighter_cruise_reference_speed = reference_speed_from_parameters(
            lighter_max_thrust,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            spec.cruise_reference_throttle,
            lighter_reference_lift_factor,
        );
        let heavier_cruise_reference_speed = reference_speed_from_parameters(
            heavier_max_thrust,
            spec.lift_coefficient,
            spec.induced_drag_coefficient,
            spec.linear_drag,
            spec.cruise_reference_throttle,
            heavier_reference_lift_factor,
        );

        assert!(heavier_max_thrust > lighter_max_thrust);
        assert!(heavier_cruise_reference_speed < lighter_cruise_reference_speed);
    }

    #[test]
    fn heavier_weight_lowers_stall_capacity_ratio_thresholds_for_same_physical_speeds() {
        let spec = AircraftSpecConfig::default_standard();
        let lighter_weight = STANDARD_WEIGHT_KG * 0.8;
        let heavier_weight = STANDARD_WEIGHT_KG * 1.2;
        let lighter_cruise_reference_speed = AircraftConfig::from_spec(&AircraftSpecConfig {
            weight_kg: lighter_weight,
            ..spec.clone()
        })
        .cruise_reference_speed;
        let heavier_cruise_reference_speed = AircraftConfig::from_spec(&AircraftSpecConfig {
            weight_kg: heavier_weight,
            ..spec.clone()
        })
        .cruise_reference_speed;

        let lighter = stall_capacity_ratio_from_speed(
            spec.stall_recovery_speed,
            lighter_cruise_reference_speed,
            spec.gravity_scale,
            lighter_weight,
        );
        let heavier = stall_capacity_ratio_from_speed(
            spec.stall_recovery_speed,
            heavier_cruise_reference_speed,
            spec.gravity_scale,
            heavier_weight,
        );

        assert!(heavier < lighter);
    }
}

#[derive(Debug, Clone, Resource, PartialEq, Eq)]
pub struct ConfigPaths {
    pub project_root: PathBuf,
    pub scene_override: Option<String>,
    pub scene_override_path: Option<PathBuf>,
}

impl ConfigPaths {
    pub fn is_dev_environment(&self) -> bool {
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(".dfb_dev_root").is_file())
            .unwrap_or(false)
    }

    pub fn recordings_root(&self) -> PathBuf {
        if self.is_dev_environment() {
            self.project_root.join("datasets/dfb_game/recordings")
        } else {
            self.project_root.join("recordings")
        }
    }

    pub fn reconstruct_root(&self) -> PathBuf {
        if self.is_dev_environment() {
            self.project_root.join("datasets/dfb_game/reconstruct")
        } else {
            self.project_root.join("reconstruct")
        }
    }

    pub fn runs_dfb_game_root(&self) -> PathBuf {
        if self.is_dev_environment() {
            self.project_root.join("runs/dfb_game")
        } else {
            self.project_root.clone()
        }
    }

    pub fn admin_snapshots_root(&self) -> PathBuf {
        self.runs_dfb_game_root().join("admin_snapshots")
    }
}

impl Default for ConfigPaths {
    fn default() -> Self {
        Self {
            project_root: resolve_project_root(),
            scene_override: None,
            scene_override_path: None,
        }
    }
}

pub fn resolve_project_root() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        candidates.push(exe_dir.to_path_buf());
    }

    for candidate in candidates {
        if let Some(root) = find_project_root(&candidate) {
            return root;
        }
    }

    PathBuf::from(".")
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|path| path.join("config").is_dir() && path.join("assets").is_dir())
        .map(Path::to_path_buf)
}

pub struct ConfigPlugin;

impl Plugin for ConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConfigPaths>()
            .add_systems(PreStartup, load_repository_config);
    }
}

fn load_repository_config(mut config: ResMut<RepositoryConfig>, paths: Res<ConfigPaths>) {
    match RepositoryConfig::load_from_root_with_scene_spec(
        &paths.project_root,
        paths.scene_override.as_deref(),
        paths.scene_override_path.as_deref(),
    ) {
        Ok(loaded) => {
            let active_scene = paths
                .scene_override
                .clone()
                .or_else(|| {
                    paths.scene_override_path.as_ref().and_then(|path| {
                        path.file_stem()
                            .and_then(|stem| stem.to_str())
                            .map(ToOwned::to_owned)
                    })
                })
                .unwrap_or_else(|| loaded.game.active_scene.clone());
            info!(
                "Config summary: standard aircraft max_level_speed={} cruise_ref_thr={} maneuver_ref_thr={} max_thrust={} pitch+={} pitch-={} roll+={} roll-={} yaw+={} yaw-={}",
                loaded.fighter1_aircraft_spec.max_level_speed,
                loaded.fighter1_aircraft_spec.cruise_reference_throttle,
                loaded.fighter1_aircraft_spec.maneuver_reference_throttle,
                loaded.fighter1_aircraft.max_thrust,
                loaded.fighter1_aircraft_spec.pitch_positive_rate_limit_deg,
                loaded.fighter1_aircraft_spec.pitch_negative_rate_limit_deg,
                loaded.fighter1_aircraft_spec.roll_positive_rate_limit_deg,
                loaded.fighter1_aircraft_spec.roll_negative_rate_limit_deg,
                loaded.fighter1_aircraft_spec.yaw_positive_rate_limit_deg,
                loaded.fighter1_aircraft_spec.yaw_negative_rate_limit_deg
            );
            *config = loaded;
            info!(
                "Loaded repository configuration from {:?} using scene {:?}",
                paths.project_root, active_scene
            );
        }
        Err(error) => {
            panic!("failed to load repository configuration: {error:#}");
        }
    }
}

fn load_ron<T>(path: PathBuf) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("failed to parse config file {}", path.display()))
}

fn load_aircraft_spec(root: &Path, file_name: &str) -> Result<AircraftSpecConfig> {
    load_ron(root.join("config/dfb_game/aircraft_specs").join(file_name))
}
