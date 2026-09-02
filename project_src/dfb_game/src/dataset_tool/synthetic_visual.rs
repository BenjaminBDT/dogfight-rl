use std::f32::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use crate::api::snapshot::class_id_from_semantic_rgba;
use crate::api::types::{
    ObservationCaptureConfig, PixelFormat, VisualCaptureVariant, VisualObservation,
    VisualResolutionMode, VisualSensorConfig, VisualSensorKind,
};
use crate::api::vision::{
    CapturedVisualFrame, SemanticCaptureMode, SemanticRenderConfig, SyntheticCaptureSessionId,
    VisualCaptureKey, begin_synthetic_capture_session, clear_synthetic_capture_session,
    request_synthetic_capture_variant, synthetic_capture_session_ready,
    take_synthetic_capture_frame,
};
use crate::app::game_app::build_headless_offscreen_capture_app_with_paths;
use crate::core::config::{ConfigPaths, RepositoryConfig, resolve_project_root};
use crate::input::actions::ControlInput;
use crate::presentation::camera::{FollowPlayerCamera, resolve_follow_camera_pose};
use crate::presentation::hud::ObservedAircraftRole;
use crate::simulation::collision::AIRCRAFT_COLLISION_RADIUS;
use crate::simulation::components::{AircraftRole, AircraftState, ControlAuthority, GunState};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum VisibilityBucket {
    FrontOnly,
    RearOnly,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum AreaBucket {
    Px1To4,
    Px5To9,
    Px10To19,
    Px20To49,
    Px50To99,
    Px100To199,
    Px200Plus,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum SemanticLabelMode {
    Strict,
    MsaaTrial,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum SamplingRegime {
    BandGuaranteed,
    UniformSupplement,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct DistanceBin {
    index: usize,
    min_meters: f32,
    max_meters: f32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum GridView {
    Front,
    Rear,
    Both,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct GridCell {
    view: GridView,
    cols: usize,
    rows: usize,
    col: usize,
    row: usize,
}

impl AreaBucket {
    fn from_pixels(px: usize) -> Option<Self> {
        match px {
            1..=4 => Some(Self::Px1To4),
            5..=9 => Some(Self::Px5To9),
            10..=19 => Some(Self::Px10To19),
            20..=49 => Some(Self::Px20To49),
            50..=99 => Some(Self::Px50To99),
            100..=199 => Some(Self::Px100To199),
            200..=usize::MAX => Some(Self::Px200Plus),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct CliArgs {
    output_dir: PathBuf,
    observed_role_selection: ObservedRoleSelection,
    width: u32,
    height: u32,
    seed: u64,
    band_positions_per_bucket: usize,
    uniform_positions_per_bucket: usize,
    orientations_per_position: usize,
    max_position_attempts_per_bucket: usize,
    target_radius_min: f32,
    target_radius_max: f32,
    target_radius_band_width: f32,
    min_selected_target_area: usize,
    semantic_label_mode: SemanticLabelMode,
    export_comparison_artifacts: bool,
    export_overlay_artifacts: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ObservedRoleSelection {
    Fighter1,
    Fighter2,
    Both,
}

#[derive(Debug, Clone, Serialize)]
struct SampleMetadata {
    sample_index: usize,
    observed_role: &'static str,
    target_role: &'static str,
    sampling_regime: SamplingRegime,
    requested_visibility_bucket: VisibilityBucket,
    requested_distance_bin: Option<DistanceBin>,
    requested_area_bucket: Option<AreaBucket>,
    sampled_grid_cell: GridCell,
    semantic_label_mode: SemanticLabelMode,
    actual_visibility_bucket: Option<VisibilityBucket>,
    actual_area_bucket: Option<AreaBucket>,
    observer_position_world: [f32; 3],
    observer_orientation_world_xyzw: [f32; 4],
    target_position_world: [f32; 3],
    target_orientation_world_xyzw: [f32; 4],
    front_target_area_strict: usize,
    rear_target_area_strict: usize,
    front_target_area: usize,
    rear_target_area: usize,
    selected_target_area: usize,
}

#[derive(Debug, Serialize)]
struct ManifestEntry {
    sample_dir: String,
    sampling_regime: SamplingRegime,
    requested_visibility_bucket: VisibilityBucket,
    requested_distance_bin: Option<DistanceBin>,
    requested_area_bucket: Option<AreaBucket>,
    sampled_grid_cell: GridCell,
    matched_requested_bucket: bool,
    actual_visibility_bucket: Option<VisibilityBucket>,
    actual_area_bucket: Option<AreaBucket>,
    front_target_area: usize,
    rear_target_area: usize,
    selected_target_area: usize,
}

#[derive(Debug, Serialize)]
struct Manifest {
    observed_role: &'static str,
    target_role: &'static str,
    width: u32,
    height: u32,
    band_positions_per_bucket: usize,
    uniform_positions_per_bucket: usize,
    orientations_per_position: usize,
    seed: u64,
    semantic_label_mode: SemanticLabelMode,
    coverage_summary: CoverageSummary,
    entries: Vec<ManifestEntry>,
}

#[derive(Debug, Serialize, Default)]
struct CoverageSummary {
    cells: Vec<GridCoverageCount>,
}

#[derive(Debug, Serialize)]
struct GridCoverageCount {
    sampling_regime: SamplingRegime,
    requested_visibility_bucket: VisibilityBucket,
    requested_distance_bin: Option<DistanceBin>,
    view: GridView,
    cols: usize,
    rows: usize,
    col: usize,
    row: usize,
    count: usize,
}

#[derive(Debug, Clone, Copy)]
struct SampledTargetPosition {
    position: Vec3,
    grid_cell: GridCell,
}

#[derive(Clone)]
struct SemanticLabels {
    active: Vec<VisualObservation>,
    strict: Vec<VisualObservation>,
    trial: Option<Vec<VisualObservation>>,
}

struct SyntheticCaptureSession {
    app: App,
    observer_entity: Entity,
    target_entity: Entity,
}

pub fn run_from_args(args: impl Iterator<Item = String>) -> Result<()> {
    let args = parse_args(args)?;
    fs::create_dir_all(&args.output_dir)?;

    let roles: &[AircraftRole] = match args.observed_role_selection {
        ObservedRoleSelection::Fighter1 => &[AircraftRole::Fighter1],
        ObservedRoleSelection::Fighter2 => &[AircraftRole::Fighter2],
        ObservedRoleSelection::Both => &[AircraftRole::Fighter1, AircraftRole::Fighter2],
    };

    if matches!(args.observed_role_selection, ObservedRoleSelection::Both) {
        fs::create_dir_all(args.output_dir.join("fighter1"))?;
        fs::create_dir_all(args.output_dir.join("fighter2"))?;
    }

    let visibility_buckets = [VisibilityBucket::FrontOnly, VisibilityBucket::RearOnly];
    let grid_cells = enumerate_grid_cells(6, 4);
    let radius_bands = build_radius_bands(&args)?;
    let uniform_distance_bin = DistanceBin {
        index: radius_bands.len(),
        min_meters: args.target_radius_min,
        max_meters: args.target_radius_max,
    };

    for (role_offset, &observed_role) in roles.iter().enumerate() {
        let mut role_args = args.clone();
        role_args.observed_role_selection = match observed_role {
            AircraftRole::Fighter1 => ObservedRoleSelection::Fighter1,
            AircraftRole::Fighter2 => ObservedRoleSelection::Fighter2,
        };
        if matches!(args.observed_role_selection, ObservedRoleSelection::Both) {
            role_args.output_dir = args.output_dir.join(role_name(observed_role));
        }
        fs::create_dir_all(&role_args.output_dir)?;

        let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(role_offset as u64));
        let mut session = SyntheticCaptureSession::new(observed_role, &role_args)?;
        let mut sample_index = 0usize;

        for visibility_bucket in visibility_buckets {
            let view_root = role_args
                .output_dir
                .join(requested_visibility_prefix(visibility_bucket));
            fs::create_dir_all(&view_root)?;
            let mut manifest_entries = Vec::new();

            for requested_radius_band in &radius_bands {
                generate_samples_for_regime(
                    &mut rng,
                    &mut session,
                    &role_args,
                    observed_role,
                    visibility_bucket,
                    SamplingRegime::BandGuaranteed,
                    Some(*requested_radius_band),
                    role_args.band_positions_per_bucket,
                    &grid_cells,
                    &view_root,
                    &mut sample_index,
                    &mut manifest_entries,
                )?;
            }
            if role_args.uniform_positions_per_bucket > 0 {
                generate_samples_for_regime(
                    &mut rng,
                    &mut session,
                    &role_args,
                    observed_role,
                    visibility_bucket,
                    SamplingRegime::UniformSupplement,
                    Some(uniform_distance_bin),
                    role_args.uniform_positions_per_bucket,
                    &grid_cells,
                    &view_root,
                    &mut sample_index,
                    &mut manifest_entries,
                )?;
            }
            let manifest = Manifest {
                observed_role: role_name(observed_role),
                target_role: role_name(target_role(observed_role)),
                width: role_args.width,
                height: role_args.height,
                band_positions_per_bucket: role_args.band_positions_per_bucket,
                uniform_positions_per_bucket: role_args.uniform_positions_per_bucket,
                orientations_per_position: role_args.orientations_per_position,
                seed: args.seed.wrapping_add(role_offset as u64),
                semantic_label_mode: role_args.semantic_label_mode,
                coverage_summary: build_coverage_summary(&manifest_entries),
                entries: manifest_entries,
            };
            fs::write(
                view_root.join("manifest.json"),
                serde_json::to_string_pretty(&manifest)?,
            )?;
        }
    }
    Ok(())
}

struct CapturedSample {
    capture_rgb: Vec<VisualObservation>,
    capture_seg: SemanticLabels,
    meta: SampleMetadata,
}

impl SyntheticCaptureSession {
    fn new(observed_role: AircraftRole, args: &CliArgs) -> Result<Self> {
        let mut app = build_headless_offscreen_capture_app_with_paths(
            ConfigPaths {
                project_root: resolve_project_root(),
                scene_override: None,
                scene_override_path: None,
            },
            false,
        );
        app.insert_resource(SemanticCaptureMode(true));
        app.insert_resource(SemanticRenderConfig {
            msaa_enabled: matches!(args.semantic_label_mode, SemanticLabelMode::MsaaTrial),
            remove_distance_fog: true,
        });
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
        app.finish();
        app.cleanup();
        app.update();
        app.update();
        if let Some(mut capture_config) = app
            .world_mut()
            .get_resource_mut::<ObservationCaptureConfig>()
        {
            *capture_config = ObservationCaptureConfig {
                enable_visual: true,
                enable_audio: false,
                visual_sensors: vec![
                    VisualSensorConfig {
                        kind: VisualSensorKind::Front,
                        width: args.width,
                        height: args.height,
                        format: PixelFormat::Rgb8,
                        resolution_mode: VisualResolutionMode::Fixed,
                        include_hud: false,
                        capture_variants: vec![
                            VisualCaptureVariant::Rgb,
                            VisualCaptureVariant::Semantic,
                        ],
                    },
                    VisualSensorConfig {
                        kind: VisualSensorKind::Rear,
                        width: args.width,
                        height: args.height,
                        format: PixelFormat::Rgb8,
                        resolution_mode: VisualResolutionMode::Fixed,
                        include_hud: false,
                        capture_variants: vec![
                            VisualCaptureVariant::Rgb,
                            VisualCaptureVariant::Semantic,
                        ],
                    },
                ],
                audio_window_seconds: 0.0,
            };
        }
        app.world_mut()
            .insert_resource(ObservedAircraftRole(observed_role));
        warm_up_visual_pipeline(&mut app);
        let (observer_entity, target_entity) =
            resolve_aircraft_entities(app.world_mut(), observed_role)?;
        zero_aircraft_controls(app.world_mut(), observer_entity)?;
        zero_aircraft_controls(app.world_mut(), target_entity)?;
        Ok(Self {
            app,
            observer_entity,
            target_entity,
        })
    }

    fn capture_sample(
        &mut self,
        args: &CliArgs,
        observed_role: AircraftRole,
        sampling_regime: SamplingRegime,
        sample_index: usize,
        requested_visibility_bucket: VisibilityBucket,
        requested_distance_bin: Option<DistanceBin>,
        requested_area_bucket: Option<AreaBucket>,
        sampled_grid_cell: GridCell,
        observer_position: Vec3,
        observer_orientation: Quat,
        target_position: Vec3,
        target_orientation: Quat,
    ) -> Result<Option<CapturedSample>> {
        apply_aircraft_pose(
            self.app.world_mut(),
            self.observer_entity,
            observer_position,
            observer_orientation,
        )?;
        apply_aircraft_pose(
            self.app.world_mut(),
            self.target_entity,
            target_position,
            target_orientation,
        )?;
        let (rgb, seg) = collect_rgb_then_semantic_from_frozen_offscreen_state(
            &mut self.app,
            observed_role,
            args.semantic_label_mode,
        )?;
        let front_target_area_strict =
            target_area_for_sensor(&seg.strict, VisualSensorKind::Front, observed_role)?;
        let rear_target_area_strict =
            target_area_for_sensor(&seg.strict, VisualSensorKind::Rear, observed_role)?;
        let front_target_area =
            target_area_for_sensor(&seg.active, VisualSensorKind::Front, observed_role)?;
        let rear_target_area =
            target_area_for_sensor(&seg.active, VisualSensorKind::Rear, observed_role)?;
        let actual_visibility_bucket =
            classify_visibility_bucket(front_target_area, rear_target_area);
        let selected_target_area = front_target_area.max(rear_target_area);
        let actual_area_bucket = AreaBucket::from_pixels(selected_target_area);
        Ok(Some(CapturedSample {
            capture_rgb: rgb,
            capture_seg: seg,
            meta: SampleMetadata {
                sample_index,
                observed_role: role_name(observed_role),
                target_role: role_name(target_role(observed_role)),
                sampling_regime,
                requested_visibility_bucket,
                requested_distance_bin,
                requested_area_bucket,
                sampled_grid_cell,
                semantic_label_mode: args.semantic_label_mode,
                actual_visibility_bucket,
                actual_area_bucket,
                observer_position_world: observer_position.to_array(),
                observer_orientation_world_xyzw: observer_orientation.to_array(),
                target_position_world: target_position.to_array(),
                target_orientation_world_xyzw: target_orientation.to_array(),
                front_target_area_strict,
                rear_target_area_strict,
                front_target_area,
                rear_target_area,
                selected_target_area,
            },
        }))
    }
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<CliArgs> {
    let mut output_dir = None;
    let mut observed_role_selection = ObservedRoleSelection::Fighter1;
    let mut width = 400u32;
    let mut height = 300u32;
    let mut seed = 0u64;
    let mut band_positions_per_bucket = 2usize;
    let mut uniform_positions_per_bucket = 0usize;
    let mut orientations_per_position = 4usize;
    let mut max_position_attempts_per_bucket = 64usize;
    let mut target_radius_min = 2.0 * AIRCRAFT_COLLISION_RADIUS;
    let mut target_radius_max = 500.0f32;
    let mut target_radius_band_width = 50.0f32;
    let mut min_selected_target_area = 1usize;
    let mut semantic_label_mode = SemanticLabelMode::Strict;
    let mut export_comparison_artifacts = false;
    let mut export_overlay_artifacts = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    args.next().context("missing value for --output-dir")?,
                ))
            }
            "--observed-role" => {
                observed_role_selection = parse_role_selection(
                    &args.next().context("missing value for --observed-role")?,
                )?
            }
            "--visual-resolution" => {
                let value = args
                    .next()
                    .context("missing value for --visual-resolution")?;
                let (w, h) = value
                    .split_once('x')
                    .context("expected WIDTHxHEIGHT for --visual-resolution")?;
                width = w.parse()?;
                height = h.parse()?;
            }
            "--seed" => seed = args.next().context("missing value for --seed")?.parse()?,
            "--band-positions-per-bucket" => {
                band_positions_per_bucket = args
                    .next()
                    .context("missing value for --band-positions-per-bucket")?
                    .parse()?
            }
            "--uniform-positions-per-bucket" => {
                uniform_positions_per_bucket = args
                    .next()
                    .context("missing value for --uniform-positions-per-bucket")?
                    .parse()?
            }
            "--orientations-per-position" => {
                orientations_per_position = args
                    .next()
                    .context("missing value for --orientations-per-position")?
                    .parse()?
            }
            "--max-position-attempts-per-bucket" => {
                max_position_attempts_per_bucket = args
                    .next()
                    .context("missing value for --max-position-attempts-per-bucket")?
                    .parse()?
            }
            "--target-radius-min" => {
                target_radius_min = args
                    .next()
                    .context("missing value for --target-radius-min")?
                    .parse()?
            }
            "--target-radius-max" => {
                target_radius_max = args
                    .next()
                    .context("missing value for --target-radius-max")?
                    .parse()?
            }
            "--target-radius-band-width" => {
                target_radius_band_width = args
                    .next()
                    .context("missing value for --target-radius-band-width")?
                    .parse()?
            }
            "--min-selected-target-area" => {
                min_selected_target_area = args
                    .next()
                    .context("missing value for --min-selected-target-area")?
                    .parse()?
            }
            "--semantic-label-mode" => {
                let value = args
                    .next()
                    .context("missing value for --semantic-label-mode")?;
                semantic_label_mode = match value.as_str() {
                    "strict" => SemanticLabelMode::Strict,
                    "msaa_trial" => SemanticLabelMode::MsaaTrial,
                    other => bail!("unknown semantic label mode: {other}"),
                };
            }
            "--export-comparison-artifacts" => {
                export_comparison_artifacts = true;
            }
            "--export-overlay-artifacts" => {
                export_overlay_artifacts = true;
            }
            other => bail!("unknown synthetic-visual arg: {other}"),
        }
    }

    Ok(CliArgs {
        output_dir: output_dir.context("--output-dir is required")?,
        observed_role_selection,
        width,
        height,
        seed,
        band_positions_per_bucket,
        uniform_positions_per_bucket,
        orientations_per_position,
        max_position_attempts_per_bucket,
        target_radius_min,
        target_radius_max,
        target_radius_band_width,
        min_selected_target_area,
        semantic_label_mode,
        export_comparison_artifacts,
        export_overlay_artifacts,
    })
}

fn parse_role_selection(value: &str) -> Result<ObservedRoleSelection> {
    match value {
        "fighter1" | "Fighter1" | "f1" => Ok(ObservedRoleSelection::Fighter1),
        "fighter2" | "Fighter2" | "f2" => Ok(ObservedRoleSelection::Fighter2),
        "both" => Ok(ObservedRoleSelection::Both),
        other => bail!("unknown observed role: {other}"),
    }
}

fn role_name(role: AircraftRole) -> &'static str {
    match role {
        AircraftRole::Fighter1 => "fighter1",
        AircraftRole::Fighter2 => "fighter2",
    }
}

fn target_role(observed_role: AircraftRole) -> AircraftRole {
    match observed_role {
        AircraftRole::Fighter1 => AircraftRole::Fighter2,
        AircraftRole::Fighter2 => AircraftRole::Fighter1,
    }
}

fn resolve_aircraft_entities(
    world: &mut World,
    observed_role: AircraftRole,
) -> Result<(Entity, Entity)> {
    let mut observer = None;
    let mut target = None;
    let target_role = target_role(observed_role);
    let mut query = world.query::<(Entity, &AircraftRole)>();
    for (entity, role) in query.iter(world) {
        if *role == observed_role {
            observer = Some(entity);
        } else if *role == target_role {
            target = Some(entity);
        }
    }
    Ok((
        observer.context("missing observer aircraft entity")?,
        target.context("missing target aircraft entity")?,
    ))
}

fn zero_aircraft_controls(world: &mut World, entity: Entity) -> Result<()> {
    let mut entity_mut = world.entity_mut(entity);
    if let Some(mut authority) = entity_mut.get_mut::<ControlAuthority>() {
        *authority = ControlAuthority::ExternalAgent;
    }
    if let Some(mut input) = entity_mut.get_mut::<ControlInput>() {
        *input = ControlInput::default();
    }
    if let Some(mut gun) = entity_mut.get_mut::<GunState>() {
        gun.cooldown_seconds = 0.0;
        gun.heat = 0.0;
        gun.overheated = false;
    }
    Ok(())
}

fn apply_aircraft_pose(
    world: &mut World,
    entity: Entity,
    position: Vec3,
    orientation: Quat,
) -> Result<()> {
    let mut entity_mut = world.entity_mut(entity);
    {
        let mut state = entity_mut
            .get_mut::<AircraftState>()
            .context("synthetic aircraft missing AircraftState")?;
        state.position = position;
        state.orientation = orientation;
        state.velocity = Vec3::ZERO;
        state.angular_rates_deg = Vec3::ZERO;
        state.forward = (orientation * Vec3::Z).normalize_or_zero();
        state.throttle = 0.0;
        state.stall_factor = 0.0;
        state.out_of_bounds_seconds = 0.0;
        state.ceiling_recovery_seconds = 0.0;
        state.is_destroyed = false;
    }
    {
        let mut transform = entity_mut
            .get_mut::<Transform>()
            .context("synthetic aircraft missing Transform")?;
        transform.translation = position;
        transform.rotation = orientation;
    }
    Ok(())
}

fn sample_observer_pose(rng: &mut ChaCha8Rng, app: &App) -> Result<(Vec3, Quat)> {
    const MAX_ATTEMPTS: usize = 128;
    let config = app.world().resource::<RepositoryConfig>();
    let min_altitude = config.scene.ground_height + 120.0;
    let max_altitude = config.scene.flight_ceiling_height - 120.0;
    let arena_radius = config.scene.arena_radius - 120.0;
    for _ in 0..MAX_ATTEMPTS {
        let theta = rng.random_range(-PI..PI);
        let radius = arena_radius * rng.random::<f32>().sqrt();
        let position = Vec3::new(
            radius * theta.cos(),
            rng.random_range(min_altitude..max_altitude),
            radius * theta.sin(),
        );
        let orientation = sample_uniform_so3(rng);
        if is_valid_aircraft_position(app, position, 120.0) {
            return Ok((position, orientation));
        }
    }
    bail!("failed to sample valid observer pose in arena cylinder")
}

fn sample_uniform_so3(rng: &mut ChaCha8Rng) -> Quat {
    let u1 = rng.random::<f32>();
    let u2 = rng.random::<f32>();
    let u3 = rng.random::<f32>();
    let sqrt1_minus_u1 = (1.0 - u1).sqrt();
    let sqrt_u1 = u1.sqrt();
    Quat::from_xyzw(
        sqrt1_minus_u1 * (2.0 * PI * u2).sin(),
        sqrt1_minus_u1 * (2.0 * PI * u2).cos(),
        sqrt_u1 * (2.0 * PI * u3).sin(),
        sqrt_u1 * (2.0 * PI * u3).cos(),
    )
    .normalize()
}

fn sample_target_position(
    rng: &mut ChaCha8Rng,
    visibility_bucket: VisibilityBucket,
    sampling_regime: SamplingRegime,
    requested_grid_cell: GridCell,
    requested_radius_band: DistanceBin,
    app: &App,
    observer_position: Vec3,
    observer_orientation: Quat,
) -> Result<SampledTargetPosition> {
    const MAX_ATTEMPTS: usize = 64;
    const GRID_COLS: usize = 6;
    const GRID_ROWS: usize = 4;
    let config = app.world().resource::<RepositoryConfig>();
    let fov_y = config.game.camera.fov_y_degrees.to_radians();
    let aspect = (config.game.camera.aspect_width.max(1) as f32)
        / (config.game.camera.aspect_height.max(1) as f32);
    let fov_x = 2.0 * ((fov_y * 0.5).tan() * aspect).atan();
    let cone_half_angle = ((fov_x * 0.5).tan().powi(2) + (fov_y * 0.5).tan().powi(2))
        .sqrt()
        .atan();
    let observer_transform =
        Transform::from_translation(observer_position).with_rotation(observer_orientation);
    let follow = FollowPlayerCamera {
        offset: Vec3::from_array(config.game.camera.follow_offset),
        rear_view_offset: Vec3::from_array(config.game.camera.rear_view_offset),
    };
    let (front_camera, front_look, front_up) =
        resolve_follow_camera_pose(&observer_transform, &follow, false);
    let (rear_camera, rear_look, rear_up) =
        resolve_follow_camera_pose(&observer_transform, &follow, true);
    let _ = sampling_regime;
    for _ in 0..MAX_ATTEMPTS {
        let (camera, look, up, expected_view) = match visibility_bucket {
            VisibilityBucket::FrontOnly => (front_camera, front_look, front_up, GridView::Front),
            VisibilityBucket::RearOnly => (rear_camera, rear_look, rear_up, GridView::Rear),
            VisibilityBucket::Both => bail!("both is not active in current synthetic mainline"),
        };
        let position = sample_point_in_spherical_cone_shell(
            rng,
            camera,
            look,
            cone_half_angle,
            requested_radius_band.min_meters,
            requested_radius_band.max_meters,
        );
        let Some(projected_cell) = project_point_to_grid_cell(
            position,
            camera,
            look,
            up,
            fov_x,
            fov_y,
            GRID_COLS,
            GRID_ROWS,
            expected_view,
        ) else {
            continue;
        };
        if projected_cell == requested_grid_cell
            && position.distance(observer_position) >= 2.0 * AIRCRAFT_COLLISION_RADIUS
            && position.distance(camera) >= minimum_camera_target_center_distance()
            && is_valid_aircraft_position(app, position, 90.0)
        {
            return Ok(SampledTargetPosition {
                position,
                grid_cell: projected_cell,
            });
        }
    }

    bail!("failed to sample valid target position for synthetic visual sample")
}

fn enumerate_grid_cells(cols: usize, rows: usize) -> Vec<GridCell> {
    let mut cells = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            cells.push(GridCell {
                view: GridView::Front,
                cols,
                rows,
                col,
                row,
            });
        }
    }
    cells
}

fn sample_quota_for_cell(cell_index: usize, total_cells: usize, total_samples: usize) -> usize {
    let base = total_samples / total_cells.max(1);
    let extra = total_samples % total_cells.max(1);
    base + usize::from(cell_index < extra)
}

fn requested_visibility_prefix(bucket: VisibilityBucket) -> &'static str {
    match bucket {
        VisibilityBucket::FrontOnly => "front",
        VisibilityBucket::RearOnly => "rear",
        VisibilityBucket::Both => "both",
    }
}

fn requested_view_is_visible(
    requested: VisibilityBucket,
    actual: Option<VisibilityBucket>,
) -> bool {
    matches!(
        (requested, actual),
        (
            VisibilityBucket::FrontOnly,
            Some(VisibilityBucket::FrontOnly | VisibilityBucket::Both)
        ) | (
            VisibilityBucket::RearOnly,
            Some(VisibilityBucket::RearOnly | VisibilityBucket::Both)
        ) | (VisibilityBucket::Both, Some(VisibilityBucket::Both))
    )
}

fn sample_point_in_spherical_cone_shell(
    rng: &mut ChaCha8Rng,
    origin: Vec3,
    forward: Vec3,
    cone_half_angle: f32,
    min_radius: f32,
    max_radius: f32,
) -> Vec3 {
    let u1 = rng.random::<f32>();
    let u2 = rng.random::<f32>();
    let u3 = rng.random::<f32>();
    let min_r3 = min_radius.powi(3);
    let max_r3 = max_radius.powi(3);
    let radius = (min_r3 + (max_r3 - min_r3) * u1).cbrt();
    let cos_alpha = cone_half_angle.cos();
    let cos_theta = 1.0 + (cos_alpha - 1.0) * u2;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = 2.0 * PI * u3;
    let local_dir = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

    let w = forward.normalize_or_zero();
    let a = if w.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let u = a.cross(w).normalize_or_zero();
    let v = w.cross(u);
    let world_dir = (local_dir.x * u + local_dir.y * v + local_dir.z * w).normalize_or_zero();
    origin + world_dir * radius
}

fn project_point_to_grid_cell(
    world_point: Vec3,
    camera: Vec3,
    look_direction: Vec3,
    up: Vec3,
    fov_x: f32,
    fov_y: f32,
    cols: usize,
    rows: usize,
    view: GridView,
) -> Option<GridCell> {
    let forward = look_direction.normalize_or_zero();
    let camera_up = up.normalize_or_zero();
    let right = camera_up.cross(forward).normalize_or_zero();
    let rel = world_point - camera;
    let z = rel.dot(forward);
    if z <= 0.0 {
        return None;
    }
    let x = rel.dot(right);
    let y = rel.dot(camera_up);
    let ndc_x = x / (z * (fov_x * 0.5).tan());
    let ndc_y = y / (z * (fov_y * 0.5).tan());
    if !(-1.0..=1.0).contains(&ndc_x) || !(-1.0..=1.0).contains(&ndc_y) {
        return None;
    }
    let u = ((ndc_x + 1.0) * 0.5).clamp(0.0, 0.999_999);
    let v = ((1.0 - ndc_y) * 0.5).clamp(0.0, 0.999_999);
    Some(GridCell {
        view,
        cols,
        rows,
        col: (u * cols as f32).floor() as usize,
        row: (v * rows as f32).floor() as usize,
    })
}

fn sample_target_position_noise(
    rng: &mut ChaCha8Rng,
    observer_position: Vec3,
    target_position: Vec3,
) -> Vec3 {
    let distance = target_position.distance(observer_position);
    let radius = (distance * 0.008).clamp(0.0, 2.0);
    if radius <= f32::EPSILON {
        return Vec3::ZERO;
    }
    sample_uniform_direction(rng) * (radius * rng.random::<f32>().cbrt())
}

fn minimum_camera_target_center_distance() -> f32 {
    PerspectiveProjection::default().near + AIRCRAFT_COLLISION_RADIUS
}

fn build_radius_bands(args: &CliArgs) -> Result<Vec<DistanceBin>> {
    anyhow::ensure!(
        args.target_radius_max > args.target_radius_min,
        "target radius max must be greater than min"
    );
    anyhow::ensure!(
        args.target_radius_band_width > 0.0,
        "target radius band width must be positive"
    );
    let mut bands = Vec::new();
    let mut min = args.target_radius_min;
    let mut index = 0usize;
    while min < args.target_radius_max - 1e-4 {
        let next_boundary = if min < args.target_radius_band_width {
            args.target_radius_band_width
        } else {
            let band_index = (min / args.target_radius_band_width).floor() + 1.0;
            band_index * args.target_radius_band_width
        };
        let max = next_boundary.min(args.target_radius_max);
        bands.push(DistanceBin {
            index,
            min_meters: min,
            max_meters: max,
        });
        min = max;
        index += 1;
    }
    anyhow::ensure!(!bands.is_empty(), "no radius bands were generated");
    Ok(bands)
}

#[allow(clippy::too_many_arguments)]
fn generate_samples_for_regime(
    rng: &mut ChaCha8Rng,
    session: &mut SyntheticCaptureSession,
    role_args: &CliArgs,
    observed_role: AircraftRole,
    visibility_bucket: VisibilityBucket,
    sampling_regime: SamplingRegime,
    requested_distance_bin: Option<DistanceBin>,
    total_samples: usize,
    grid_cells: &[GridCell],
    view_root: &Path,
    sample_index: &mut usize,
    manifest_entries: &mut Vec<ManifestEntry>,
) -> Result<()> {
    for (cell_index, base_grid_cell) in grid_cells.iter().copied().enumerate() {
        let grid_cell = GridCell {
            view: match visibility_bucket {
                VisibilityBucket::FrontOnly => GridView::Front,
                VisibilityBucket::RearOnly => GridView::Rear,
                VisibilityBucket::Both => GridView::Both,
            },
            ..base_grid_cell
        };
        let quota = sample_quota_for_cell(cell_index, grid_cells.len(), total_samples);
        let mut accepted_positions = 0usize;
        let mut attempts = 0usize;
        let max_attempts = role_args.max_position_attempts_per_bucket.max(1) * quota.max(1);
        while accepted_positions < quota && attempts < max_attempts {
            attempts += 1;
            let (observer_position, observer_orientation) =
                sample_observer_pose(rng, &session.app)?;
            let Ok(target_sample) = sample_target_position(
                rng,
                visibility_bucket,
                sampling_regime,
                grid_cell,
                requested_distance_bin.unwrap_or(DistanceBin {
                    index: usize::MAX,
                    min_meters: role_args.target_radius_min,
                    max_meters: role_args.target_radius_max,
                }),
                &session.app,
                observer_position,
                observer_orientation,
            ) else {
                continue;
            };
            let mut accepted = false;
            for _ in 0..role_args.orientations_per_position {
                let target_orientation = sample_target_orientation(rng);
                let target_position = target_sample.position
                    + sample_target_position_noise(rng, observer_position, target_sample.position);
                if !is_valid_aircraft_position(&session.app, target_position, 90.0) {
                    continue;
                }
                if target_position.distance(observer_position) < 2.0 * AIRCRAFT_COLLISION_RADIUS {
                    continue;
                }
                let camera_distance_ok = camera_distance_ok(
                    &session.app,
                    visibility_bucket,
                    observer_position,
                    observer_orientation,
                    target_position,
                );
                if !camera_distance_ok {
                    continue;
                }
                let Some(sample) = session.capture_sample(
                    role_args,
                    observed_role,
                    sampling_regime,
                    *sample_index,
                    visibility_bucket,
                    requested_distance_bin,
                    None,
                    target_sample.grid_cell,
                    observer_position,
                    observer_orientation,
                    target_position,
                    target_orientation,
                )?
                else {
                    continue;
                };
                if !requested_view_is_visible(
                    visibility_bucket,
                    sample.meta.actual_visibility_bucket,
                ) {
                    continue;
                }
                if sample.meta.selected_target_area < role_args.min_selected_target_area {
                    continue;
                }
                let sample_dir = write_sample(
                    role_args,
                    view_root,
                    *sample_index,
                    &sample.capture_rgb,
                    &sample.capture_seg,
                    &sample.meta,
                )?;
                manifest_entries.push(ManifestEntry {
                    sample_dir,
                    sampling_regime,
                    requested_visibility_bucket: visibility_bucket,
                    requested_distance_bin,
                    requested_area_bucket: None,
                    sampled_grid_cell: sample.meta.sampled_grid_cell,
                    matched_requested_bucket: true,
                    actual_visibility_bucket: sample.meta.actual_visibility_bucket,
                    actual_area_bucket: sample.meta.actual_area_bucket,
                    front_target_area: sample.meta.front_target_area,
                    rear_target_area: sample.meta.rear_target_area,
                    selected_target_area: sample.meta.selected_target_area,
                });
                *sample_index += 1;
                accepted_positions += 1;
                accepted = true;
                break;
            }
            if !accepted {
                continue;
            }
        }
    }
    Ok(())
}

fn camera_distance_ok(
    app: &App,
    visibility_bucket: VisibilityBucket,
    observer_position: Vec3,
    observer_orientation: Quat,
    target_position: Vec3,
) -> bool {
    let config = app.world().resource::<RepositoryConfig>();
    let observer_transform =
        Transform::from_translation(observer_position).with_rotation(observer_orientation);
    let follow = FollowPlayerCamera {
        offset: Vec3::from_array(config.game.camera.follow_offset),
        rear_view_offset: Vec3::from_array(config.game.camera.rear_view_offset),
    };
    let (camera, _, _) = match visibility_bucket {
        VisibilityBucket::FrontOnly => {
            resolve_follow_camera_pose(&observer_transform, &follow, false)
        }
        VisibilityBucket::RearOnly => {
            resolve_follow_camera_pose(&observer_transform, &follow, true)
        }
        VisibilityBucket::Both => return false,
    };
    target_position.distance(camera) >= minimum_camera_target_center_distance()
}

fn sample_uniform_direction(rng: &mut ChaCha8Rng) -> Vec3 {
    let z = rng.random_range(-1.0..1.0);
    let phi = rng.random_range(-PI..PI);
    let xy = (1.0f32 - z * z).sqrt();
    Vec3::new(xy * phi.cos(), xy * phi.sin(), z)
}

fn sample_target_orientation(rng: &mut ChaCha8Rng) -> Quat {
    let yaw = rng.random_range(-PI..PI);
    let pitch = rng.random_range((-0.5 * PI)..(0.5 * PI));
    let roll = rng.random_range(-PI..PI);
    Quat::from_euler(EulerRot::YXZ, yaw, pitch, roll)
}

fn is_valid_aircraft_position(app: &App, position: Vec3, clearance_margin: f32) -> bool {
    let config = app.world().resource::<RepositoryConfig>();
    let min_altitude = config.scene.ground_height + 120.0;
    let max_altitude = config.scene.flight_ceiling_height - 120.0;
    if position.y < min_altitude || position.y > max_altitude {
        return false;
    }
    if position.x.hypot(position.z) >= config.scene.arena_radius - 120.0 {
        return false;
    }
    !config.scene.obstacles.iter().any(|obstacle| {
        let center = Vec3::from_array(obstacle.position);
        let half_extents =
            Vec3::from_array(obstacle.size) * 0.5 + Vec3::splat(clearance_margin.max(0.0));
        let min = center - half_extents;
        let max = center + half_extents;
        position.x >= min.x
            && position.x <= max.x
            && position.y >= min.y
            && position.y <= max.y
            && position.z >= min.z
            && position.z <= max.z
    })
}

fn collect_rgb_then_semantic_from_frozen_offscreen_state(
    app: &mut App,
    observed_role: AircraftRole,
    label_mode: SemanticLabelMode,
) -> Result<(Vec<VisualObservation>, SemanticLabels)> {
    // Freeze world progression for this sample and capture RGB / semantic in two phases
    // from the same logical state. The guarantee here is "same sample state", not
    // "same GPU frame index".
    app.world_mut()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    let rgb = collect_offscreen_rgb_after_flush(app)?;
    let seg = collect_semantic_labels_after_offscreen_flush(app, observed_role, label_mode)?;
    Ok((rgb, seg))
}

fn collect_offscreen_rgb_after_flush(app: &mut App) -> Result<Vec<VisualObservation>> {
    const MAX_CAPTURE_UPDATES: usize = 24;
    let session_id = begin_synthetic_capture_session(app.world_mut());
    let keys =
        request_synthetic_capture_variant(app.world_mut(), session_id, VisualCaptureVariant::Rgb);
    for _ in 0..MAX_CAPTURE_UPDATES {
        app.update();
        if synthetic_capture_session_ready(app.world(), session_id, &keys) {
            let rgb = collect_rgb_observations_from_session(app.world_mut(), session_id)?;
            clear_synthetic_capture_session(app.world_mut(), session_id);
            return Ok(rgb);
        }
    }
    clear_synthetic_capture_session(app.world_mut(), session_id);
    bail!(
        "synthetic RGB capture did not become ready within {MAX_CAPTURE_UPDATES} updates: {}",
        describe_session_capture_state(app.world(), session_id, &keys)
    )
}

fn collect_semantic_labels_after_offscreen_flush(
    app: &mut App,
    observed_role: AircraftRole,
    label_mode: SemanticLabelMode,
) -> Result<SemanticLabels> {
    const MAX_CAPTURE_UPDATES: usize = 24;
    let session_id = begin_synthetic_capture_session(app.world_mut());
    let keys = request_synthetic_capture_variant(
        app.world_mut(),
        session_id,
        VisualCaptureVariant::Semantic,
    );
    for _ in 0..MAX_CAPTURE_UPDATES {
        app.update();
        if synthetic_capture_session_ready(app.world(), session_id, &keys) {
            let seg = collect_semantic_labels_from_capture_session(
                app,
                session_id,
                observed_role,
                label_mode,
            );
            clear_synthetic_capture_session(app.world_mut(), session_id);
            return seg;
        }
    }
    clear_synthetic_capture_session(app.world_mut(), session_id);
    bail!(
        "synthetic semantic capture did not become ready within {MAX_CAPTURE_UPDATES} updates: {}",
        describe_session_capture_state(app.world(), session_id, &keys)
    )
}

fn describe_session_capture_state(
    world: &World,
    session_id: SyntheticCaptureSessionId,
    keys: &[VisualCaptureKey],
) -> String {
    let mut parts = Vec::new();
    for key in keys {
        let part = if synthetic_capture_session_ready(world, session_id, &[*key]) {
            format!("{:?}/{:?}=present", key.kind, key.variant)
        } else {
            format!("{:?}/{:?}=missing", key.kind, key.variant)
        };
        parts.push(part);
    }
    format!("session={}, {}", session_id.0, parts.join(", "))
}

fn collect_semantic_labels_from_capture_session(
    app: &mut App,
    session_id: SyntheticCaptureSessionId,
    observed_role: AircraftRole,
    label_mode: SemanticLabelMode,
) -> Result<SemanticLabels> {
    let raw_frames = collect_raw_semantic_frames_from_capture_session(app, session_id)?;
    let strict = raw_frames
        .iter()
        .map(|(camera, frame)| {
            raw_semantic_frame_to_observation(*camera, frame, observed_role, false)
        })
        .collect::<Result<Vec<_>>>()?;
    let trial = if matches!(label_mode, SemanticLabelMode::MsaaTrial) {
        Some(
            raw_frames
                .iter()
                .map(|(camera, frame)| {
                    raw_semantic_frame_to_observation(*camera, frame, observed_role, true)
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let active = trial.clone().unwrap_or_else(|| strict.clone());
    Ok(SemanticLabels {
        active,
        strict,
        trial,
    })
}

fn collect_raw_semantic_frames_from_capture_session(
    app: &mut App,
    session_id: SyntheticCaptureSessionId,
) -> Result<Vec<(VisualSensorKind, CapturedVisualFrame)>> {
    let mut result = Vec::new();
    for kind in [VisualSensorKind::Front, VisualSensorKind::Rear] {
        let key = VisualCaptureKey {
            kind,
            variant: VisualCaptureVariant::Semantic,
        };
        let frame = take_synthetic_capture_frame(app.world_mut(), session_id, key)
            .context("missing raw semantic frame from synthetic capture session")?;
        result.push((kind, frame));
    }
    Ok(result)
}

fn collect_rgb_observations_from_session(
    world: &mut World,
    session_id: SyntheticCaptureSessionId,
) -> Result<Vec<VisualObservation>> {
    let capture_config = world.resource::<ObservationCaptureConfig>().clone();
    let mut result = Vec::new();
    for sensor in &capture_config.visual_sensors {
        if !sensor
            .requested_capture_variants()
            .contains(&VisualCaptureVariant::Rgb)
        {
            continue;
        }
        let key = VisualCaptureKey {
            kind: sensor.kind,
            variant: VisualCaptureVariant::Rgb,
        };
        let frame = take_synthetic_capture_frame(world, session_id, key)
            .context("missing RGB frame from synthetic capture session")?;
        let bytes = match sensor.format {
            PixelFormat::Rgba8 => frame.bytes,
            PixelFormat::Rgb8 => rgba_to_rgb(&frame.bytes),
            PixelFormat::Gray8 => rgba_to_gray(&frame.bytes),
        };
        result.push(VisualObservation {
            camera: sensor.kind,
            width: frame.width,
            height: frame.height,
            format: sensor.format,
            resolution_mode: sensor.resolution_mode,
            include_hud: sensor.include_hud,
            bytes_ready: true,
            bytes,
        });
    }
    Ok(result)
}

fn raw_semantic_frame_to_observation(
    camera: VisualSensorKind,
    frame: &CapturedVisualFrame,
    observed_role: AircraftRole,
    msaa_trial: bool,
) -> Result<VisualObservation> {
    let expected = frame.width as usize * frame.height as usize * 4;
    anyhow::ensure!(
        frame.bytes.len() == expected,
        "unexpected semantic RGBA byte length: got {}, expected {}",
        frame.bytes.len(),
        expected
    );
    let strict = semantic_rgba_to_strict_class_ids(&frame.bytes);
    let bytes = if msaa_trial {
        semantic_rgba_to_msaa_trial_class_ids(&frame.bytes, target_class_id(observed_role))
    } else {
        strict
    };
    Ok(VisualObservation {
        camera,
        width: frame.width,
        height: frame.height,
        format: PixelFormat::Gray8,
        resolution_mode: VisualResolutionMode::Fixed,
        include_hud: false,
        bytes_ready: true,
        bytes,
    })
}

fn semantic_rgba_to_strict_class_ids(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .map(class_id_from_semantic_rgba)
        .collect()
}

fn rgba_to_rgb(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
        .collect()
}

fn rgba_to_gray(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .map(|rgba| {
            let r = rgba[0] as f32;
            let g = rgba[1] as f32;
            let b = rgba[2] as f32;
            (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn semantic_rgba_to_msaa_trial_class_ids(bytes: &[u8], target_class: u8) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .map(|rgba| semantic_trial_class_id_from_rgba(rgba, target_class))
        .collect()
}

fn semantic_trial_class_id_from_rgba(rgba: &[u8], target_class: u8) -> u8 {
    let r = rgba[0];
    let g = rgba[1];
    let b = rgba[2];
    if r == 0 && g == 0 && b == 0 {
        return 0;
    }
    match target_class {
        1 => {
            if g > 0 && g >= r && g >= b {
                1
            } else if r > 0 {
                2
            } else {
                0
            }
        }
        2 => {
            if r > 0 && r >= g && r >= b {
                2
            } else if g > 0 {
                1
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn warm_up_visual_pipeline(app: &mut App) {
    const MAX_WARMUP_FRAMES: usize = 240;
    for _ in 0..MAX_WARMUP_FRAMES {
        app.update();
    }
}

fn target_class_id(observed_role: AircraftRole) -> u8 {
    match observed_role {
        AircraftRole::Fighter1 => 2,
        AircraftRole::Fighter2 => 1,
    }
}

fn target_area_for_sensor(
    observations: &[VisualObservation],
    sensor: VisualSensorKind,
    observed_role: AircraftRole,
) -> Result<usize> {
    let frame = observations
        .iter()
        .find(|obs| obs.camera == sensor)
        .context("missing observation for requested sensor")?;
    if frame.format != PixelFormat::Gray8 {
        bail!("expected Gray8 semantic frame");
    }
    let class_id = target_class_id(observed_role);
    Ok(frame.bytes.iter().filter(|&&v| v == class_id).count())
}

fn classify_visibility_bucket(
    front_target_area: usize,
    rear_target_area: usize,
) -> Option<VisibilityBucket> {
    match (front_target_area > 0, rear_target_area > 0) {
        (true, false) => Some(VisibilityBucket::FrontOnly),
        (false, true) => Some(VisibilityBucket::RearOnly),
        (true, true) => Some(VisibilityBucket::Both),
        (false, false) => None,
    }
}

fn write_sample(
    args: &CliArgs,
    root: &Path,
    sample_index: usize,
    rgb: &[VisualObservation],
    seg: &SemanticLabels,
    meta: &SampleMetadata,
) -> Result<String> {
    let sample_dir_name = format!("sample_{sample_index:06}");
    let sample_dir = root.join(&sample_dir_name);
    fs::create_dir_all(&sample_dir)?;
    fs::create_dir_all(root.join("rgb"))?;
    fs::create_dir_all(root.join("seg_color"))?;
    fs::create_dir_all(root.join("metadata"))?;
    if args.export_overlay_artifacts {
        fs::create_dir_all(root.join("overlay"))?;
    }
    if args.export_comparison_artifacts {
        fs::create_dir_all(root.join("seg_color_strict"))?;
        fs::create_dir_all(root.join("seg_color_trial"))?;
    }

    for frame in rgb {
        let stem = match frame.camera {
            VisualSensorKind::Front => "front_rgb.ppm",
            VisualSensorKind::Rear => "rear_rgb.ppm",
        };
        write_ppm(
            &sample_dir.join(stem),
            frame.width,
            frame.height,
            &frame.bytes,
        )?;
        write_ppm(
            &root.join("rgb").join(format!(
                "{}_step_{sample_index:06}.ppm",
                sensor_prefix(frame.camera)
            )),
            frame.width,
            frame.height,
            &frame.bytes,
        )?;
    }
    for frame in &seg.active {
        let stem = match frame.camera {
            VisualSensorKind::Front => "front_segmentation.pgm",
            VisualSensorKind::Rear => "rear_segmentation.pgm",
        };
        let color_stem = match frame.camera {
            VisualSensorKind::Front => "front_segmentation_color.ppm",
            VisualSensorKind::Rear => "rear_segmentation_color.ppm",
        };
        write_pgm(
            &sample_dir.join(stem),
            frame.width,
            frame.height,
            &frame.bytes,
        )?;
        write_segmentation_color_ppm(
            &sample_dir.join(color_stem),
            frame.width,
            frame.height,
            &frame.bytes,
        )?;
        write_segmentation_color_ppm(
            &root.join("seg_color").join(format!(
                "{}_step_{sample_index:06}.ppm",
                sensor_prefix(frame.camera)
            )),
            frame.width,
            frame.height,
            &frame.bytes,
        )?;
        if args.export_overlay_artifacts
            && let Some(rgb_frame) = rgb.iter().find(|obs| obs.camera == frame.camera)
        {
            let overlay_stem = match frame.camera {
                VisualSensorKind::Front => "front_overlay.ppm",
                VisualSensorKind::Rear => "rear_overlay.ppm",
            };
            write_segmentation_overlay_ppm(
                &sample_dir.join(overlay_stem),
                rgb_frame.width,
                rgb_frame.height,
                &rgb_frame.bytes,
                &frame.bytes,
            )?;
            write_segmentation_overlay_ppm(
                &root.join("overlay").join(format!(
                    "{}_step_{sample_index:06}.ppm",
                    sensor_prefix(frame.camera)
                )),
                rgb_frame.width,
                rgb_frame.height,
                &rgb_frame.bytes,
                &frame.bytes,
            )?;
        }
    }
    if args.export_comparison_artifacts {
        for frame in &seg.strict {
            let stem = match frame.camera {
                VisualSensorKind::Front => "front_segmentation_strict.pgm",
                VisualSensorKind::Rear => "rear_segmentation_strict.pgm",
            };
            let color_stem = match frame.camera {
                VisualSensorKind::Front => "front_segmentation_strict_color.ppm",
                VisualSensorKind::Rear => "rear_segmentation_strict_color.ppm",
            };
            write_pgm(
                &sample_dir.join(stem),
                frame.width,
                frame.height,
                &frame.bytes,
            )?;
            write_segmentation_color_ppm(
                &sample_dir.join(color_stem),
                frame.width,
                frame.height,
                &frame.bytes,
            )?;
            write_segmentation_color_ppm(
                &root.join("seg_color_strict").join(format!(
                    "{}_step_{sample_index:06}.ppm",
                    sensor_prefix(frame.camera)
                )),
                frame.width,
                frame.height,
                &frame.bytes,
            )?;
        }
        if let Some(trial) = &seg.trial {
            for frame in trial {
                let stem = match frame.camera {
                    VisualSensorKind::Front => "front_segmentation_trial.pgm",
                    VisualSensorKind::Rear => "rear_segmentation_trial.pgm",
                };
                let color_stem = match frame.camera {
                    VisualSensorKind::Front => "front_segmentation_trial_color.ppm",
                    VisualSensorKind::Rear => "rear_segmentation_trial_color.ppm",
                };
                write_pgm(
                    &sample_dir.join(stem),
                    frame.width,
                    frame.height,
                    &frame.bytes,
                )?;
                write_segmentation_color_ppm(
                    &sample_dir.join(color_stem),
                    frame.width,
                    frame.height,
                    &frame.bytes,
                )?;
                write_segmentation_color_ppm(
                    &root.join("seg_color_trial").join(format!(
                        "{}_step_{sample_index:06}.ppm",
                        sensor_prefix(frame.camera)
                    )),
                    frame.width,
                    frame.height,
                    &frame.bytes,
                )?;
            }
        }
    }
    fs::write(
        sample_dir.join("metadata.json"),
        serde_json::to_string_pretty(meta)?,
    )?;
    fs::write(
        root.join("metadata")
            .join(format!("step_{sample_index:06}.json")),
        serde_json::to_string_pretty(meta)?,
    )?;
    Ok(sample_dir_name)
}

fn build_coverage_summary(entries: &[ManifestEntry]) -> CoverageSummary {
    use std::collections::BTreeMap;

    let mut counts: BTreeMap<
        (
            SamplingRegime,
            VisibilityBucket,
            Option<usize>,
            GridView,
            usize,
            usize,
            usize,
            usize,
        ),
        usize,
    > = BTreeMap::new();
    for entry in entries {
        let cell = entry.sampled_grid_cell;
        *counts
            .entry((
                entry.sampling_regime,
                entry.requested_visibility_bucket,
                entry.requested_distance_bin.map(|bin| bin.index),
                cell.view,
                cell.cols,
                cell.rows,
                cell.col,
                cell.row,
            ))
            .or_default() += 1;
    }
    CoverageSummary {
        cells: counts
            .into_iter()
            .map(
                |(
                    (
                        sampling_regime,
                        requested_visibility_bucket,
                        requested_distance_bin_index,
                        view,
                        cols,
                        rows,
                        col,
                        row,
                    ),
                    count,
                )| GridCoverageCount {
                    sampling_regime,
                    requested_visibility_bucket,
                    requested_distance_bin: entry_distance_bin_from_index(
                        entries,
                        sampling_regime,
                        requested_visibility_bucket,
                        requested_distance_bin_index,
                    ),
                    view,
                    cols,
                    rows,
                    col,
                    row,
                    count,
                },
            )
            .collect(),
    }
}

fn entry_distance_bin_from_index(
    entries: &[ManifestEntry],
    sampling_regime: SamplingRegime,
    requested_visibility_bucket: VisibilityBucket,
    requested_distance_bin_index: Option<usize>,
) -> Option<DistanceBin> {
    entries
        .iter()
        .find(|entry| {
            entry.sampling_regime == sampling_regime
                && entry.requested_visibility_bucket == requested_visibility_bucket
                && entry.requested_distance_bin.map(|bin| bin.index) == requested_distance_bin_index
        })
        .and_then(|entry| entry.requested_distance_bin)
}

fn sensor_prefix(sensor: VisualSensorKind) -> &'static str {
    match sensor {
        VisualSensorKind::Front => "front",
        VisualSensorKind::Rear => "rear",
    }
}

fn write_ppm(path: &Path, width: u32, height: u32, bytes: &[u8]) -> Result<()> {
    let expected = width as usize * height as usize * 3;
    anyhow::ensure!(
        bytes.len() == expected,
        "unexpected RGB byte length for {}: got {}, expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut file = fs::File::create(path)?;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    file.write_all(bytes)?;
    Ok(())
}

fn write_pgm(path: &Path, width: u32, height: u32, bytes: &[u8]) -> Result<()> {
    let expected = width as usize * height as usize;
    anyhow::ensure!(
        bytes.len() == expected,
        "unexpected grayscale byte length for {}: got {}, expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut file = fs::File::create(path)?;
    write!(file, "P5\n{} {}\n255\n", width, height)?;
    file.write_all(bytes)?;
    Ok(())
}

fn write_segmentation_color_ppm(path: &Path, width: u32, height: u32, bytes: &[u8]) -> Result<()> {
    let expected = width as usize * height as usize;
    anyhow::ensure!(
        bytes.len() == expected,
        "unexpected grayscale byte length for {}: got {}, expected {}",
        path.display(),
        bytes.len(),
        expected
    );
    let mut rgb = Vec::with_capacity(expected * 3);
    for &class_id in bytes {
        let color = segmentation_color(class_id);
        rgb.extend_from_slice(&color);
    }
    write_ppm(path, width, height, &rgb)
}

fn write_segmentation_overlay_ppm(
    path: &Path,
    width: u32,
    height: u32,
    rgb_bytes: &[u8],
    class_ids: &[u8],
) -> Result<()> {
    let expected_rgb = width as usize * height as usize * 3;
    let expected_mask = width as usize * height as usize;
    anyhow::ensure!(
        rgb_bytes.len() == expected_rgb,
        "unexpected RGB byte length for {}: got {}, expected {}",
        path.display(),
        rgb_bytes.len(),
        expected_rgb
    );
    anyhow::ensure!(
        class_ids.len() == expected_mask,
        "unexpected mask byte length for {}: got {}, expected {}",
        path.display(),
        class_ids.len(),
        expected_mask
    );
    let mut overlay = Vec::with_capacity(expected_rgb);
    for (rgb, &class_id) in rgb_bytes.chunks_exact(3).zip(class_ids.iter()) {
        let base = [rgb[0], rgb[1], rgb[2]];
        let color = segmentation_color(class_id);
        let alpha = match class_id {
            0 => 0.0f32,
            1 => 0.35f32,
            2 => 0.70f32,
            _ => 0.0f32,
        };
        for i in 0..3 {
            let blended = (base[i] as f32 * (1.0 - alpha) + color[i] as f32 * alpha).round() as u8;
            overlay.push(blended);
        }
    }
    write_ppm(path, width, height, &overlay)
}

fn segmentation_color(class_id: u8) -> [u8; 3] {
    match class_id {
        1 => [40, 220, 40],
        2 => [255, 80, 40],
        _ => [0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AreaBucket, semantic_rgba_to_msaa_trial_class_ids, semantic_rgba_to_strict_class_ids,
    };

    #[test]
    fn area_bucket_from_pixels_matches_frozen_ranges() {
        assert_eq!(AreaBucket::from_pixels(1), Some(AreaBucket::Px1To4));
        assert_eq!(AreaBucket::from_pixels(4), Some(AreaBucket::Px1To4));
        assert_eq!(AreaBucket::from_pixels(5), Some(AreaBucket::Px5To9));
        assert_eq!(AreaBucket::from_pixels(19), Some(AreaBucket::Px10To19));
        assert_eq!(AreaBucket::from_pixels(49), Some(AreaBucket::Px20To49));
        assert_eq!(AreaBucket::from_pixels(99), Some(AreaBucket::Px50To99));
        assert_eq!(AreaBucket::from_pixels(199), Some(AreaBucket::Px100To199));
        assert_eq!(AreaBucket::from_pixels(200), Some(AreaBucket::Px200Plus));
        assert_eq!(AreaBucket::from_pixels(0), None);
    }

    #[test]
    fn msaa_trial_prefers_target_on_red_green_ties() {
        let rgba = vec![
            0, 0, 0, 255, //
            255, 255, 0, 255, //
            255, 0, 0, 255, //
        ];
        let strict = semantic_rgba_to_strict_class_ids(&rgba);
        assert_eq!(strict, vec![0, 1, 2]);
        let trial = semantic_rgba_to_msaa_trial_class_ids(&rgba, 2);
        assert_eq!(trial, vec![0, 2, 2]);
    }

    #[test]
    fn msaa_trial_keeps_source_when_target_channel_absent() {
        let rgba = vec![
            0, 255, 0, 255, //
            0, 128, 0, 255, //
            32, 64, 0, 255, //
        ];
        let strict = semantic_rgba_to_strict_class_ids(&rgba);
        assert_eq!(strict, vec![1, 1, 1]);
        let trial = semantic_rgba_to_msaa_trial_class_ids(&rgba, 2);
        assert_eq!(trial, vec![1, 1, 1]);
    }
}
