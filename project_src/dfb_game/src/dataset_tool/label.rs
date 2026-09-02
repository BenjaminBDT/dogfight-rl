use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use bevy::math::{Quat, Vec3};
use bevy::prelude::Transform;
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::types::{AircraftObservation, StateObservation, VisualSensorKind};
use crate::core::config::{ConfigPaths, RepositoryConfig};
use crate::presentation::camera::{FollowPlayerCamera, resolve_follow_camera_pose};
use crate::recording::reconstruct::RecordingAccess;

const LABEL_SCHEMA_VERSION: u32 = 2;
const POS_FLOOR: f32 = 0.3;
const ORI_FLOOR: f32 = 0.1;
const K_V: f32 = 1.0;
const K_R: f32 = 1.0;
const KEYPOINT_DEPTH_EPSILON_MIN: f32 = 0.05;
const KEYPOINT_DEPTH_EPSILON_RELATIVE: f32 = 0.005;
const KEYPOINT_SEGMENTATION_NEIGHBORHOOD_RADIUS: usize = 1;
const AUDIO_DISTANCE_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Deserialize)]
struct AssetIdsManifest {
    coordinate_conventions: Vec<NamedAssetRef>,
    keypoint_schemas: Vec<NamedAssetRef>,
    audio_cue_schemas: Vec<NamedAssetRef>,
}

#[derive(Debug, Deserialize)]
struct NamedAssetRef {
    id: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    default: bool,
}

#[derive(Debug, Deserialize)]
struct AudioFeatureField {
    order: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AudioCueSchema {
    schema_id: String,
    layout: String,
    channel_order: Vec<String>,
    sample_rate_hz: u32,
    energy_features: AudioFeatureField,
    cue_features: AudioFeatureField,
}

#[derive(Debug, Deserialize)]
struct KeypointSchema {
    schema_id: String,
    coordinate_convention_id: String,
    point_labels: Vec<String>,
    points_3d_object: HashMap<String, [f32; 3]>,
}

#[derive(Debug)]
struct CliArgs {
    episode_roots: Vec<PathBuf>,
    recordings_root: PathBuf,
    observed_roles: Vec<String>,
    force: bool,
    profile: bool,
}

#[derive(Debug)]
struct VisibilityAuditArgs {
    episode_root: PathBuf,
    observed_role: String,
    max_steps: Option<usize>,
    output_path: Option<PathBuf>,
}

#[derive(Debug)]
pub struct SyntheticSingleStepArgs {
    source_root: PathBuf,
    output_dir: PathBuf,
    force: bool,
    max_samples: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SyntheticCleanManifestEntry {
    sample_dir: String,
    #[serde(default)]
    observed_role: Option<String>,
    #[serde(default)]
    target_role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SyntheticCleanManifest {
    width: u32,
    height: u32,
    #[serde(default)]
    observed_role: Option<String>,
    #[serde(default)]
    target_role: Option<String>,
    entries: Vec<SyntheticCleanManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyntheticSingleStepLabelEntry {
    pub sample_dir: String,
    pub observed_role: String,
    pub target_role: String,
    pub gt_relative_position: [f32; 3],
    pub gt_relative_orientation: [f32; 6],
    pub gt_doa_unit_vector_body: [f32; 3],
    pub gt_log_distance_scalar: f32,
    pub target_pos_conf: f32,
    pub target_ori_conf: f32,
    pub keypoints_2d_front: Vec<[f32; 2]>,
    pub keypoints_2d_rear: Vec<[f32; 2]>,
    pub keypoint_visibility_front: Vec<u8>,
    pub keypoint_visibility_rear: Vec<u8>,
    pub keypoint_projectable_front: Vec<u8>,
    pub keypoint_projectable_rear: Vec<u8>,
    pub keypoint_voting_front: SparseVotingArtifactRef,
    pub keypoint_voting_rear: SparseVotingArtifactRef,
    pub front_target_area: u32,
    pub rear_target_area: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyntheticSingleStepManifestEntry {
    pub sample_dir: String,
    pub labels_path: String,
    pub observed_role: String,
    pub target_role: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyntheticSingleStepManifest {
    pub schema_version: u32,
    pub dataset_format: String,
    pub source_dataset_root: String,
    pub width: u32,
    pub height: u32,
    pub observed_role: String,
    pub target_role: String,
    pub coordinate_convention_id: String,
    pub keypoint_schema_id: String,
    pub point_labels: Vec<String>,
    pub max_front_voting_pixels: u32,
    pub max_rear_voting_pixels: u32,
    pub entries: Vec<SyntheticSingleStepManifestEntry>,
    pub notes: Vec<String>,
}

struct CleanCameraProjectionInput<'a> {
    observer: &'a AircraftObservation,
    target: &'a AircraftObservation,
    camera_kind: VisualSensorKind,
    width: usize,
    height: usize,
    segmentation_mask: &'a [u8],
}

#[derive(Debug, Clone)]
struct CameraVisualLabels {
    keypoints_2d: Vec<[f32; 2]>,
    keypoint_visibility: Vec<u8>,
    keypoint_projectable: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVotingArtifactRef {
    pub file_path: String,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub width: u32,
    pub height: u32,
    pub keypoint_count: u16,
    pub pixel_count: u32,
    pub coord_dtype: String,
    pub vector_dtype: String,
}

#[derive(Debug, Clone)]
struct CameraVisibilityDiagnostics {
    keypoints_2d: Vec<[f32; 2]>,
    keypoint_visibility: Vec<u8>,
    keypoint_projectable: Vec<u8>,
    projected_mask: Vec<u8>,
    segmentation_matches: Vec<u8>,
    depth_matches: Vec<u8>,
    segmentation_mask: Vec<u8>,
    width: usize,
    height: usize,
    target_class_id: u8,
}

#[derive(Debug, Clone, Copy)]
struct KeypointNeighborhoodMatch {
    segmentation_match: bool,
    depth_match: bool,
    final_visibility: bool,
}

#[derive(Debug, Serialize)]
pub struct CameraVisibilityAuditSummary {
    total_keypoints: u64,
    projected_keypoints: u64,
    segmentation_match_count: u64,
    depth_match_count: u64,
    final_visibility_count: u64,
    steps_with_any_segmentation_match: u64,
    steps_with_any_depth_match: u64,
    steps_with_any_final_visibility: u64,
}

#[derive(Debug, Serialize)]
pub struct StepCameraVisibilityAudit {
    segmentation_match_count: u32,
    depth_match_count: u32,
    final_visibility_count: u32,
}

#[derive(Debug, Serialize)]
pub struct StepVisibilityAudit {
    step_index: u32,
    sim_time_seconds: f32,
    front: StepCameraVisibilityAudit,
    rear: StepCameraVisibilityAudit,
}

#[derive(Debug, Serialize)]
pub struct VisibilityAuditReport {
    source_episode_id: String,
    source_episode_root: String,
    observed_role: String,
    max_steps: Option<usize>,
    front: CameraVisibilityAuditSummary,
    rear: CameraVisibilityAuditSummary,
    steps: Vec<StepVisibilityAudit>,
}

#[derive(Debug, Clone, Copy)]
struct RawAudioFeatures {
    binaural_energy_t: [f32; 4],
    binaural_cue_vector_t: [f32; 10],
}

struct SparseVotingBundleWriter {
    relative_path: String,
    file: File,
    next_offset: u64,
}

impl SparseVotingBundleWriter {
    fn create(derived_root: &Path, relative_path: &str) -> Result<Self> {
        let full_path = derived_root.join(relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let file = File::create(&full_path)
            .with_context(|| format!("failed to create {}", full_path.display()))?;
        Ok(Self {
            relative_path: relative_path.to_string(),
            file,
            next_offset: 0,
        })
    }

    fn write_step(
        &mut self,
        width: usize,
        height: usize,
        keypoints_2d: &[[f32; 2]],
        keypoint_projectable: &[u8],
        segmentation_mask: &[u8],
        target_class_id: u8,
    ) -> Result<SparseVotingArtifactRef> {
        let payload = build_sparse_voting_payload(
            width,
            height,
            keypoints_2d,
            keypoint_projectable,
            segmentation_mask,
            target_class_id,
        )?;
        use std::io::Write as _;
        self.file
            .write_all(&payload.bytes)
            .with_context(|| format!("failed to append {}", self.relative_path))?;
        let artifact = SparseVotingArtifactRef {
            file_path: self.relative_path.clone(),
            byte_offset: self.next_offset,
            byte_length: payload.bytes.len() as u64,
            width: width as u32,
            height: height as u32,
            keypoint_count: keypoints_2d.len() as u16,
            pixel_count: payload.pixel_count,
            coord_dtype: "u16".to_string(),
            vector_dtype: "float16".to_string(),
        };
        self.next_offset += payload.bytes.len() as u64;
        Ok(artifact)
    }
}

struct SparseVotingPayload {
    bytes: Vec<u8>,
    pixel_count: u32,
}

#[derive(Debug, Clone)]
struct AircraftMesh {
    triangles: Vec<[Vec3; 3]>,
}

#[derive(Debug, Default)]
struct LabelProfile {
    load_assets: Duration,
    load_episode_and_manifests: Duration,
    raw_audio_feature_preload: Duration,
    per_step_visual_decode: Duration,
    per_step_geometry: Duration,
    manifest_serialize: Duration,
    step_count: u64,
}

#[derive(Debug, Default)]
struct VisualArtifactBundleCache {
    bytes_by_relative_path: HashMap<PathBuf, Vec<u8>>,
}

impl VisualArtifactBundleCache {
    fn read_slice<'a>(
        &'a mut self,
        derived_root: &Path,
        artifact: &crate::recording::VisualArtifactRef,
    ) -> Result<&'a [u8]> {
        let Some(relative_path) = &artifact.file_path else {
            return Ok(&[]);
        };
        let relative_path = PathBuf::from(relative_path);
        if !self.bytes_by_relative_path.contains_key(&relative_path) {
            let path = derived_root.join(&relative_path);
            let bytes =
                fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
            self.bytes_by_relative_path
                .insert(relative_path.clone(), bytes);
        }
        let bundle = self
            .bytes_by_relative_path
            .get(&relative_path)
            .ok_or_else(|| anyhow!("missing visual artifact bundle cache entry"))?;
        if let Some(offset) = artifact.byte_offset {
            let length = artifact
                .byte_length
                .with_context(|| "visual artifact has byte_offset but missing byte_length")?;
            let start = offset as usize;
            let end = start
                .checked_add(length as usize)
                .with_context(|| "visual artifact byte range overflowed usize")?;
            return bundle.get(start..end).with_context(|| {
                format!(
                    "visual artifact byte range [{start}..{end}) is out of bounds for {} bytes",
                    bundle.len()
                )
            });
        }
        Ok(bundle.as_slice())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedStepLabels {
    pub index: u32,
    pub tick: u64,
    pub sim_time_seconds: f32,
    pub gt_relative_position_body: [f32; 3],
    pub gt_doa_unit_vector_body: [f32; 3],
    pub gt_log_distance_scalar: f32,
    pub gt_relative_orientation_quat: [f32; 4],
    pub gt_relative_linear_velocity_body: [f32; 3],
    pub gt_relative_angular_velocity_body: [f32; 3],
    pub keypoints_2d_front: Vec<[f32; 2]>,
    pub keypoints_2d_rear: Vec<[f32; 2]>,
    pub keypoint_visibility_front: Vec<u8>,
    pub keypoint_visibility_rear: Vec<u8>,
    pub keypoint_projectable_front: Vec<u8>,
    pub keypoint_projectable_rear: Vec<u8>,
    pub keypoint_voting_front: SparseVotingArtifactRef,
    pub keypoint_voting_rear: SparseVotingArtifactRef,
    pub binaural_energy_t: [f32; 4],
    pub binaural_cue_vector_t: [f32; 10],
    pub target_pos_conf: f32,
    pub target_ori_conf: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedLabelsManifest {
    pub schema_version: u32,
    pub source_episode_id: String,
    pub source_episode_root: String,
    pub observed_role: String,
    pub derived_manifest_path: String,
    pub coordinate_convention_id: String,
    pub keypoint_schema_id: String,
    pub audio_cue_schema_id: String,
    pub visual_label_mode: String,
    pub notes: Vec<String>,
    pub steps: Vec<DerivedStepLabels>,
}

pub fn run_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_args(args)?;
    let config_paths = ConfigPaths::default();
    let project_root = config_paths.project_root.clone();
    let mut profile = LabelProfile::default();
    let assets_started_at = Instant::now();
    let assets = load_assets(&project_root)?;
    profile.load_assets += assets_started_at.elapsed();

    let episode_roots = if args.episode_roots.is_empty() {
        find_episode_roots(&args.recordings_root)?
    } else {
        args.episode_roots
    };

    if episode_roots.is_empty() {
        bail!("no recorded episodes found");
    }

    for episode_root in episode_roots {
        let episode_load_started_at = Instant::now();
        let access = RecordingAccess::new(&episode_root);
        let manifest = access.manifest()?;
        let repository_config =
            RepositoryConfig::load_from_root_with_scene(&project_root, Some(&manifest.scene_name))?;
        profile.load_episode_and_manifests += episode_load_started_at.elapsed();

        let requested_roles = if args.observed_roles.is_empty() {
            access.available_derived_roles()?
        } else {
            args.observed_roles.clone()
        };

        for role in requested_roles {
            write_labels_for_role(
                &access,
                &manifest.episode_id,
                &repository_config,
                &assets,
                &role,
                args.force,
                &mut profile,
            )?;
        }
    }

    if args.profile {
        print_label_profile_report(&profile);
    }

    Ok(())
}

pub fn run_visibility_audit_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_visibility_audit_args(args)?;
    let config_paths = ConfigPaths::default();
    let project_root = config_paths.project_root.clone();
    let assets = load_assets(&project_root)?;
    let access = RecordingAccess::new(&args.episode_root);
    let manifest = access.manifest()?;
    let repository_config =
        RepositoryConfig::load_from_root_with_scene(&project_root, Some(&manifest.scene_name))?;
    let derived_manifest = access
        .derived_manifest(&args.observed_role)
        .with_context(|| format!("missing derived modalities for role {}", args.observed_role))?;
    let derived_root = access.derived_root_for_role(&args.observed_role);
    let episode = access.load_episode()?;
    let step_limit = args.max_steps.unwrap_or(usize::MAX);
    let mut visual_bundle_cache = VisualArtifactBundleCache::default();
    let mut audit_profile = LabelProfile::default();
    let mut report = VisibilityAuditReport {
        source_episode_id: manifest.episode_id.clone(),
        source_episode_root: access.episode_root().display().to_string(),
        observed_role: args.observed_role.clone(),
        max_steps: args.max_steps,
        front: CameraVisibilityAuditSummary {
            total_keypoints: 0,
            projected_keypoints: 0,
            segmentation_match_count: 0,
            depth_match_count: 0,
            final_visibility_count: 0,
            steps_with_any_segmentation_match: 0,
            steps_with_any_depth_match: 0,
            steps_with_any_final_visibility: 0,
        },
        rear: CameraVisibilityAuditSummary {
            total_keypoints: 0,
            projected_keypoints: 0,
            segmentation_match_count: 0,
            depth_match_count: 0,
            final_visibility_count: 0,
            steps_with_any_segmentation_match: 0,
            steps_with_any_depth_match: 0,
            steps_with_any_final_visibility: 0,
        },
        steps: Vec::new(),
    };

    for (step, artifacts) in episode
        .steps
        .iter()
        .zip(&derived_manifest.steps)
        .take(step_limit)
    {
        let observer = find_aircraft(&step.state, &args.observed_role)?;
        let target = find_aircraft(&step.state, opposite_role_name(&args.observed_role))?;
        let front = project_keypoints_visibility_diagnostics_for_camera(
            observer,
            target,
            &repository_config,
            &derived_manifest.capture_config.visual_sensors,
            &assets.keypoint_schema,
            &assets.aircraft_mesh,
            &derived_root,
            &artifacts.segmentation,
            VisualSensorKind::Front,
            &mut audit_profile,
            &mut visual_bundle_cache,
        )?;
        let rear = project_keypoints_visibility_diagnostics_for_camera(
            observer,
            target,
            &repository_config,
            &derived_manifest.capture_config.visual_sensors,
            &assets.keypoint_schema,
            &assets.aircraft_mesh,
            &derived_root,
            &artifacts.segmentation,
            VisualSensorKind::Rear,
            &mut audit_profile,
            &mut visual_bundle_cache,
        )?;
        update_visibility_summary(&mut report.front, &front);
        update_visibility_summary(&mut report.rear, &rear);
        report.steps.push(StepVisibilityAudit {
            step_index: step.index,
            sim_time_seconds: step.state.sim_time_seconds,
            front: build_step_camera_visibility_audit(&front),
            rear: build_step_camera_visibility_audit(&rear),
        });
    }

    let payload = serde_json::to_string_pretty(&report)?;
    if let Some(output_path) = args.output_path {
        fs::write(&output_path, format!("{payload}\n"))
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        println!("wrote {}", output_path.display());
    } else {
        println!("{payload}");
    }
    Ok(())
}

pub fn run_synthetic_single_step_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_synthetic_single_step_args(args)?;
    let config_paths = ConfigPaths::default();
    let project_root = config_paths.project_root.clone();
    let assets = load_assets(&project_root)?;
    let repository_config = RepositoryConfig::load_from_root(&project_root)?;
    build_synthetic_single_step_dataset(&args, &repository_config, &assets)
}

fn write_labels_for_role(
    access: &RecordingAccess,
    episode_id: &str,
    repository_config: &RepositoryConfig,
    assets: &ResolvedAssets,
    observed_role: &str,
    force: bool,
    profile: &mut LabelProfile,
) -> Result<()> {
    let manifest_load_started_at = Instant::now();
    let derived_manifest = access
        .derived_manifest(observed_role)
        .with_context(|| format!("missing derived modalities for role {observed_role}"))?;
    let derived_root = access.derived_root_for_role(observed_role);
    let episode = access.load_episode()?;
    profile.load_episode_and_manifests += manifest_load_started_at.elapsed();
    let label_path = access
        .derived_root_for_role(observed_role)
        .join("derived_labels.ron");

    if label_path.exists() && !force {
        bail!(
            "label manifest already exists at {}; rerun with --force",
            label_path.display()
        );
    }

    let raw_audio_started_at = Instant::now();
    let raw_audio_features = episode
        .steps
        .iter()
        .zip(&derived_manifest.steps)
        .map(|(step, artifacts)| {
            raw_audio_features_for_step(
                &derived_root,
                &assets.audio_schema,
                artifacts.audio.as_ref(),
            )
            .map(|audio| (step, audio))
        })
        .collect::<Result<Vec<_>>>()?;
    profile.raw_audio_feature_preload += raw_audio_started_at.elapsed();
    let mut visual_bundle_cache = VisualArtifactBundleCache::default();
    let mut front_voting_writer =
        SparseVotingBundleWriter::create(&derived_root, "vision_voting/front.bin")?;
    let mut rear_voting_writer =
        SparseVotingBundleWriter::create(&derived_root, "vision_voting/rear.bin")?;
    let mut previous_labels: Option<DerivedStepLabels> = None;
    let mut steps = Vec::with_capacity(raw_audio_features.len());
    for ((step, raw_audio), artifacts) in raw_audio_features.iter().zip(&derived_manifest.steps) {
        let labels = derive_step_labels(
            step.index,
            &step.state,
            repository_config,
            &derived_manifest.capture_config.visual_sensors,
            observed_role,
            &assets.keypoint_schema,
            &assets.aircraft_mesh,
            &derived_root,
            artifacts,
            raw_audio,
            previous_labels.as_ref(),
            profile,
            &mut visual_bundle_cache,
            &mut front_voting_writer,
            &mut rear_voting_writer,
        )?;
        previous_labels = Some(labels.clone());
        steps.push(labels);
        profile.step_count += 1;
        anyhow::ensure!(
            artifacts.index == step.index,
            "derived artifact step alignment mismatch for role {observed_role}: {} != {}",
            artifacts.index,
            step.index
        );
    }

    let derived_manifest_path = access
        .derived_root_for_role(observed_role)
        .join(&derived_manifest.artifact_convention.manifest_file);
    let output = DerivedLabelsManifest {
        schema_version: LABEL_SCHEMA_VERSION,
        source_episode_id: episode_id.to_string(),
        source_episode_root: access.episode_root().display().to_string(),
        observed_role: observed_role.to_string(),
        derived_manifest_path: derived_manifest_path.display().to_string(),
        coordinate_convention_id: assets.coordinate_convention_id.clone(),
        keypoint_schema_id: assets.keypoint_schema.schema_id.clone(),
        audio_cue_schema_id: assets.audio_schema.schema_id.clone(),
        visual_label_mode: "segmentation_and_keypoint_rules_v1".to_string(),
        notes: vec![
            "This v1 label manifest exports GT relative state, projected keypoint labels, segmentation-driven keypoint visibility, and binaural audio structure features.".to_string(),
            "Visual/audio evidence strength terms are runtime-only intermediates and are intentionally excluded from the offline label contract.".to_string(),
            "Sparse dense-voting supervision is stored as separate foreground-only binary bundles and referenced per step from this manifest.".to_string(),
        ],
        steps,
    };
    let serialize_started_at = Instant::now();
    fs::write(
        &label_path,
        ron::ser::to_string_pretty(&output, ron::ser::PrettyConfig::default())?,
    )?;
    profile.manifest_serialize += serialize_started_at.elapsed();
    println!("wrote {}", label_path.display());
    Ok(())
}

fn build_synthetic_single_step_dataset(
    args: &SyntheticSingleStepArgs,
    repository_config: &RepositoryConfig,
    assets: &ResolvedAssets,
) -> Result<()> {
    let manifest_path = args.source_root.join("manifest.json");
    let clean_manifest: SyntheticCleanManifest = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    if args.output_dir.exists() {
        if !args.force {
            bail!(
                "synthetic single-step output already exists at {}; rerun with --force",
                args.output_dir.display()
            );
        }
        fs::remove_dir_all(&args.output_dir)
            .with_context(|| format!("failed to remove {}", args.output_dir.display()))?;
    }
    fs::create_dir_all(args.output_dir.join("labels")).with_context(|| {
        format!(
            "failed to create {}",
            args.output_dir.join("labels").display()
        )
    })?;
    let mut front_voting_writer =
        SparseVotingBundleWriter::create(&args.output_dir, "voting/front.bin")?;
    let mut rear_voting_writer =
        SparseVotingBundleWriter::create(&args.output_dir, "voting/rear.bin")?;
    let default_observed_role = clean_manifest
        .observed_role
        .clone()
        .unwrap_or_else(|| "fighter1".to_string());
    let default_target_role = clean_manifest
        .target_role
        .clone()
        .unwrap_or_else(|| "fighter2".to_string());
    let sample_limit = args.max_samples.unwrap_or(clean_manifest.entries.len());
    let mut max_front_voting_pixels = 0_u32;
    let mut max_rear_voting_pixels = 0_u32;
    let mut entries = Vec::with_capacity(sample_limit.min(clean_manifest.entries.len()));

    for entry in clean_manifest.entries.iter().take(sample_limit) {
        let observed_role = entry
            .observed_role
            .clone()
            .unwrap_or_else(|| default_observed_role.clone());
        let target_role = entry
            .target_role
            .clone()
            .unwrap_or_else(|| default_target_role.clone());
        let sample_root = args.source_root.join(&entry.sample_dir);
        let metadata: Value = serde_json::from_slice(
            &fs::read(sample_root.join("metadata.json")).with_context(|| {
                format!(
                    "failed to read {}",
                    sample_root.join("metadata.json").display()
                )
            })?,
        )
        .with_context(|| {
            format!(
                "failed to parse {}",
                sample_root.join("metadata.json").display()
            )
        })?;
        let observer = synthetic_aircraft_from_metadata(&metadata, &observed_role, "observer")?;
        let target = synthetic_aircraft_from_metadata(&metadata, &target_role, "target")?;
        let (
            gt_relative_position,
            gt_relative_orientation,
            gt_doa_unit_vector_body,
            gt_log_distance_scalar,
        ) = synthetic_relative_pose_labels(&observer, &target);
        let front_segmentation = read_clean_segmentation_mask(
            &sample_root.join("front_segmentation.pgm"),
            clean_manifest.width as usize,
            clean_manifest.height as usize,
        )?;
        let rear_segmentation = read_clean_segmentation_mask(
            &sample_root.join("rear_segmentation.pgm"),
            clean_manifest.width as usize,
            clean_manifest.height as usize,
        )?;
        let front_diagnostics = project_keypoints_visibility_diagnostics_for_clean_camera(
            repository_config,
            &assets.keypoint_schema,
            &assets.aircraft_mesh,
            CleanCameraProjectionInput {
                observer: &observer,
                target: &target,
                camera_kind: VisualSensorKind::Front,
                width: clean_manifest.width as usize,
                height: clean_manifest.height as usize,
                segmentation_mask: &front_segmentation,
            },
        )?;
        let rear_diagnostics = project_keypoints_visibility_diagnostics_for_clean_camera(
            repository_config,
            &assets.keypoint_schema,
            &assets.aircraft_mesh,
            CleanCameraProjectionInput {
                observer: &observer,
                target: &target,
                camera_kind: VisualSensorKind::Rear,
                width: clean_manifest.width as usize,
                height: clean_manifest.height as usize,
                segmentation_mask: &rear_segmentation,
            },
        )?;
        let front_voting_artifact = front_voting_writer.write_step(
            front_diagnostics.width,
            front_diagnostics.height,
            front_diagnostics.keypoints_2d.as_slice(),
            front_diagnostics.keypoint_projectable.as_slice(),
            front_diagnostics.segmentation_mask.as_slice(),
            front_diagnostics.target_class_id,
        )?;
        let rear_voting_artifact = rear_voting_writer.write_step(
            rear_diagnostics.width,
            rear_diagnostics.height,
            rear_diagnostics.keypoints_2d.as_slice(),
            rear_diagnostics.keypoint_projectable.as_slice(),
            rear_diagnostics.segmentation_mask.as_slice(),
            rear_diagnostics.target_class_id,
        )?;
        max_front_voting_pixels = max_front_voting_pixels.max(front_voting_artifact.pixel_count);
        max_rear_voting_pixels = max_rear_voting_pixels.max(rear_voting_artifact.pixel_count);
        let label_entry = SyntheticSingleStepLabelEntry {
            sample_dir: entry.sample_dir.clone(),
            observed_role: observed_role.clone(),
            target_role: target_role.clone(),
            gt_relative_position,
            gt_relative_orientation,
            gt_doa_unit_vector_body,
            gt_log_distance_scalar,
            target_pos_conf: 1.0,
            target_ori_conf: 1.0,
            keypoints_2d_front: front_diagnostics.keypoints_2d.clone(),
            keypoints_2d_rear: rear_diagnostics.keypoints_2d.clone(),
            keypoint_visibility_front: front_diagnostics.keypoint_visibility.clone(),
            keypoint_visibility_rear: rear_diagnostics.keypoint_visibility.clone(),
            keypoint_projectable_front: front_diagnostics.keypoint_projectable.clone(),
            keypoint_projectable_rear: rear_diagnostics.keypoint_projectable.clone(),
            keypoint_voting_front: front_voting_artifact,
            keypoint_voting_rear: rear_voting_artifact,
            front_target_area: front_diagnostics
                .segmentation_mask
                .iter()
                .filter(|&&value| value == front_diagnostics.target_class_id)
                .count() as u32,
            rear_target_area: rear_diagnostics
                .segmentation_mask
                .iter()
                .filter(|&&value| value == rear_diagnostics.target_class_id)
                .count() as u32,
        };
        let labels_relative_path = format!("labels/{}.json", entry.sample_dir);
        fs::write(
            args.output_dir.join(&labels_relative_path),
            format!("{}\n", serde_json::to_string_pretty(&label_entry)?),
        )
        .with_context(|| {
            format!(
                "failed to write {}",
                args.output_dir.join(&labels_relative_path).display()
            )
        })?;
        entries.push(SyntheticSingleStepManifestEntry {
            sample_dir: entry.sample_dir.clone(),
            labels_path: labels_relative_path,
            observed_role,
            target_role,
        });
    }

    let output_manifest = SyntheticSingleStepManifest {
        schema_version: 1,
        dataset_format: "synthetic_single_step_v1".to_string(),
        source_dataset_root: args.source_root.display().to_string(),
        width: clean_manifest.width,
        height: clean_manifest.height,
        observed_role: default_observed_role,
        target_role: default_target_role,
        coordinate_convention_id: assets.coordinate_convention_id.clone(),
        keypoint_schema_id: assets.keypoint_schema.schema_id.clone(),
        point_labels: assets.keypoint_schema.point_labels.clone(),
        max_front_voting_pixels,
        max_rear_voting_pixels,
        entries,
        notes: vec![
            "Synthetic single-step v1 keeps RGB/segmentation in the source clean root and stores only keypoint/voting/geometry supervision in this derived dataset root.".to_string(),
            "Audio window features remain zero-filled placeholders in v1; geometry/rule targets are derived from synthetic observer/target poses.".to_string(),
        ],
    };
    fs::write(
        args.output_dir.join("manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&output_manifest)?),
    )
    .with_context(|| {
        format!(
            "failed to write {}",
            args.output_dir.join("manifest.json").display()
        )
    })?;
    println!("wrote {}", args.output_dir.display());
    Ok(())
}

fn build_step_camera_visibility_audit(
    diagnostics: &CameraVisibilityDiagnostics,
) -> StepCameraVisibilityAudit {
    StepCameraVisibilityAudit {
        segmentation_match_count: diagnostics
            .segmentation_matches
            .iter()
            .map(|&value| u32::from(value))
            .sum(),
        depth_match_count: diagnostics
            .depth_matches
            .iter()
            .map(|&value| u32::from(value))
            .sum(),
        final_visibility_count: diagnostics
            .keypoint_visibility
            .iter()
            .map(|&value| u32::from(value))
            .sum(),
    }
}

fn update_visibility_summary(
    summary: &mut CameraVisibilityAuditSummary,
    diagnostics: &CameraVisibilityDiagnostics,
) {
    summary.total_keypoints += diagnostics.keypoint_visibility.len() as u64;
    summary.projected_keypoints += diagnostics
        .projected_mask
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>();
    let segmentation_match_count = diagnostics
        .segmentation_matches
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>();
    let depth_match_count = diagnostics
        .depth_matches
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>();
    let final_visibility_count = diagnostics
        .keypoint_visibility
        .iter()
        .map(|&value| u64::from(value))
        .sum::<u64>();
    summary.segmentation_match_count += segmentation_match_count;
    summary.depth_match_count += depth_match_count;
    summary.final_visibility_count += final_visibility_count;
    summary.steps_with_any_segmentation_match += u64::from(segmentation_match_count > 0);
    summary.steps_with_any_depth_match += u64::from(depth_match_count > 0);
    summary.steps_with_any_final_visibility += u64::from(final_visibility_count > 0);
}

fn derive_step_labels(
    step_index: u32,
    state: &StateObservation,
    repository_config: &RepositoryConfig,
    visual_sensors: &[crate::api::types::VisualSensorConfig],
    observed_role: &str,
    keypoint_schema: &KeypointSchema,
    aircraft_mesh: &AircraftMesh,
    derived_root: &Path,
    artifacts: &crate::recording::DerivedStepArtifacts,
    raw_audio: &RawAudioFeatures,
    previous: Option<&DerivedStepLabels>,
    profile: &mut LabelProfile,
    visual_bundle_cache: &mut VisualArtifactBundleCache,
    front_voting_writer: &mut SparseVotingBundleWriter,
    rear_voting_writer: &mut SparseVotingBundleWriter,
) -> Result<DerivedStepLabels> {
    let observer = find_aircraft(state, observed_role)?;
    let target = find_aircraft(state, opposite_role_name(observed_role))?;

    let observer_position = vec3(observer.position);
    let observer_orientation = quat(observer.orientation_quat).normalize();
    let target_position = vec3(target.position);
    let target_orientation = quat(target.orientation_quat).normalize();
    let relative_position = observer_orientation.inverse() * (target_position - observer_position);
    let relative_distance = relative_position.length().max(AUDIO_DISTANCE_EPSILON);
    let gt_doa_unit_vector_body =
        if relative_position.length_squared() > AUDIO_DISTANCE_EPSILON.powi(2) {
            (relative_position / relative_distance).to_array()
        } else {
            Vec3::Z.to_array()
        };
    let gt_log_distance_scalar = relative_distance.ln();
    let relative_orientation = (observer_orientation.inverse() * target_orientation).normalize();
    let front_camera_diagnostics = project_keypoints_visibility_diagnostics_for_camera(
        observer,
        target,
        repository_config,
        visual_sensors,
        keypoint_schema,
        aircraft_mesh,
        derived_root,
        &artifacts.segmentation,
        VisualSensorKind::Front,
        profile,
        visual_bundle_cache,
    )?;
    let rear_camera_diagnostics = project_keypoints_visibility_diagnostics_for_camera(
        observer,
        target,
        repository_config,
        visual_sensors,
        keypoint_schema,
        aircraft_mesh,
        derived_root,
        &artifacts.segmentation,
        VisualSensorKind::Rear,
        profile,
        visual_bundle_cache,
    )?;
    let front_camera_labels = CameraVisualLabels {
        keypoints_2d: front_camera_diagnostics.keypoints_2d.clone(),
        keypoint_visibility: front_camera_diagnostics.keypoint_visibility.clone(),
        keypoint_projectable: front_camera_diagnostics.keypoint_projectable.clone(),
    };
    let rear_camera_labels = CameraVisualLabels {
        keypoints_2d: rear_camera_diagnostics.keypoints_2d.clone(),
        keypoint_visibility: rear_camera_diagnostics.keypoint_visibility.clone(),
        keypoint_projectable: rear_camera_diagnostics.keypoint_projectable.clone(),
    };
    let front_voting_artifact = front_voting_writer.write_step(
        front_camera_diagnostics.width,
        front_camera_diagnostics.height,
        front_camera_diagnostics.keypoints_2d.as_slice(),
        front_camera_diagnostics.keypoint_projectable.as_slice(),
        front_camera_diagnostics.segmentation_mask.as_slice(),
        front_camera_diagnostics.target_class_id,
    )?;
    let rear_voting_artifact = rear_voting_writer.write_step(
        rear_camera_diagnostics.width,
        rear_camera_diagnostics.height,
        rear_camera_diagnostics.keypoints_2d.as_slice(),
        rear_camera_diagnostics.keypoint_projectable.as_slice(),
        rear_camera_diagnostics.segmentation_mask.as_slice(),
        rear_camera_diagnostics.target_class_id,
    )?;

    let relative_linear_velocity = if let Some(previous) = previous {
        let dt = (state.sim_time_seconds - previous.sim_time_seconds).max(1e-3);
        (relative_position - vec3(previous.gt_relative_position_body)) / dt
    } else {
        observer_orientation.inverse()
            * (vec3(target.linear_velocity) - vec3(observer.linear_velocity))
    };

    let (relative_angular_velocity, delta_orientation_quat) = if let Some(previous) = previous {
        let dt = (state.sim_time_seconds - previous.sim_time_seconds).max(1e-3);
        let previous_orientation = quat(previous.gt_relative_orientation_quat).normalize();
        let delta = (relative_orientation * previous_orientation.inverse()).normalize();
        let angular_velocity = so3_log(delta) / dt;
        (angular_velocity, delta)
    } else {
        (Vec3::ZERO, Quat::IDENTITY)
    };

    let (target_pos_conf, target_ori_conf) = if let Some(previous) = previous {
        let dt = (state.sim_time_seconds - previous.sim_time_seconds).max(1e-3);
        let previous_relative_position = vec3(previous.gt_relative_position_body);
        let predicted_position =
            previous_relative_position + vec3(previous.gt_relative_linear_velocity_body) * dt;
        let e_pos = (predicted_position - relative_position).length();
        let previous_relative_orientation = quat(previous.gt_relative_orientation_quat).normalize();
        let e_ori = rotation_angle_between(previous_relative_orientation, relative_orientation);
        let pos_scale = (dt * K_V * relative_linear_velocity.length()).max(POS_FLOOR);
        let ori_scale = (dt * K_R * relative_angular_velocity.length()).max(ORI_FLOOR);
        let pos_conf = (-std::f32::consts::LN_2 * (e_pos / pos_scale).powi(2)).exp();
        let ori_conf = (-std::f32::consts::LN_2 * (e_ori / ori_scale).powi(2)).exp();
        (pos_conf.clamp(0.0, 1.0), ori_conf.clamp(0.0, 1.0))
    } else {
        (1.0, 1.0)
    };

    let _ = delta_orientation_quat;

    Ok(DerivedStepLabels {
        index: step_index,
        tick: state.tick,
        sim_time_seconds: state.sim_time_seconds,
        gt_relative_position_body: relative_position.to_array(),
        gt_doa_unit_vector_body,
        gt_log_distance_scalar,
        gt_relative_orientation_quat: quat_to_array(relative_orientation),
        gt_relative_linear_velocity_body: relative_linear_velocity.to_array(),
        gt_relative_angular_velocity_body: relative_angular_velocity.to_array(),
        keypoints_2d_front: front_camera_labels.keypoints_2d,
        keypoints_2d_rear: rear_camera_labels.keypoints_2d,
        keypoint_visibility_front: front_camera_labels.keypoint_visibility,
        keypoint_visibility_rear: rear_camera_labels.keypoint_visibility,
        keypoint_projectable_front: front_camera_labels.keypoint_projectable,
        keypoint_projectable_rear: rear_camera_labels.keypoint_projectable,
        keypoint_voting_front: front_voting_artifact,
        keypoint_voting_rear: rear_voting_artifact,
        binaural_energy_t: raw_audio.binaural_energy_t,
        binaural_cue_vector_t: raw_audio.binaural_cue_vector_t,
        target_pos_conf,
        target_ori_conf,
    })
}

fn project_keypoints_visibility_diagnostics_for_camera(
    observer: &AircraftObservation,
    target: &AircraftObservation,
    repository_config: &RepositoryConfig,
    visual_sensors: &[crate::api::types::VisualSensorConfig],
    keypoint_schema: &KeypointSchema,
    aircraft_mesh: &AircraftMesh,
    derived_root: &Path,
    segmentation_artifacts: &[crate::recording::VisualArtifactRef],
    camera_kind: VisualSensorKind,
    profile: &mut LabelProfile,
    visual_bundle_cache: &mut VisualArtifactBundleCache,
) -> Result<CameraVisibilityDiagnostics> {
    let Some(sensor) = visual_sensors
        .iter()
        .find(|sensor| sensor.kind == camera_kind)
    else {
        return Ok(CameraVisibilityDiagnostics {
            keypoints_2d: vec![[0.0, 0.0]; keypoint_schema.point_labels.len()],
            keypoint_visibility: vec![0; keypoint_schema.point_labels.len()],
            keypoint_projectable: vec![0; keypoint_schema.point_labels.len()],
            projected_mask: vec![0; keypoint_schema.point_labels.len()],
            segmentation_matches: vec![0; keypoint_schema.point_labels.len()],
            depth_matches: vec![0; keypoint_schema.point_labels.len()],
            segmentation_mask: Vec::new(),
            width: 0,
            height: 0,
            target_class_id: 0,
        });
    };
    let observer_transform = aircraft_transform(observer);
    let follow = FollowPlayerCamera {
        offset: Vec3::from_array(repository_config.game.camera.follow_offset),
        rear_view_offset: Vec3::from_array(repository_config.game.camera.rear_view_offset),
    };
    let rear_view = matches!(camera_kind, VisualSensorKind::Rear);
    let (camera_position, look_direction, up) =
        resolve_follow_camera_pose(&observer_transform, &follow, rear_view);
    let mut camera_transform = Transform::from_translation(camera_position);
    camera_transform.look_to(look_direction, up);

    let width = sensor.width as usize;
    let height = sensor.height as usize;
    let decode_started_at = Instant::now();
    let segmentation_mask = decode_segmentation_by_camera(
        derived_root,
        segmentation_artifacts,
        camera_kind,
        width,
        height,
        visual_bundle_cache,
    )?;
    profile.per_step_visual_decode += decode_started_at.elapsed();
    let geometry_started_at = Instant::now();
    let aspect_ratio = repository_config.game.camera.aspect_width.max(1) as f32
        / repository_config.game.camera.aspect_height.max(1) as f32;
    let fov_y = repository_config.game.camera.fov_y_degrees.to_radians();
    let target_transform = aircraft_transform(target);
    let depth_buffer = rasterize_aircraft_depth_buffer(
        aircraft_mesh,
        &target_transform,
        &camera_transform,
        width,
        height,
        fov_y,
        aspect_ratio,
    );
    let target_class_id = segmentation_class_id_for_role(&target.role);
    let mut keypoints_2d = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut visibility = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut projectable = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut projected_mask = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut segmentation_matches_list = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut depth_matches_list = Vec::with_capacity(keypoint_schema.point_labels.len());
    for label in &keypoint_schema.point_labels {
        let Some(point_object) = keypoint_schema.points_3d_object.get(label) else {
            keypoints_2d.push([0.0, 0.0]);
            visibility.push(0);
            projectable.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        };
        let world = target_transform.transform_point(Vec3::from_array(*point_object));
        let Some(projected) = project_world_to_image_with_depth(
            &camera_transform,
            width,
            height,
            fov_y,
            aspect_ratio,
            world,
        ) else {
            keypoints_2d.push([0.0, 0.0]);
            visibility.push(0);
            projectable.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        };
        projectable.push(1);
        keypoints_2d.push(projected.pixel);
        if !projected.in_frame {
            visibility.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        }
        projected_mask.push(1);
        let pixel_x = projected.pixel[0]
            .round()
            .clamp(0.0, width.saturating_sub(1) as f32) as usize;
        let pixel_y = projected.pixel[1]
            .round()
            .clamp(0.0, height.saturating_sub(1) as f32) as usize;
        let neighborhood = evaluate_keypoint_neighborhood_match(
            segmentation_mask.as_slice(),
            depth_buffer.as_slice(),
            width,
            height,
            pixel_x,
            pixel_y,
            target_class_id,
            projected.depth,
        );
        segmentation_matches_list.push(u8::from(neighborhood.segmentation_match));
        depth_matches_list.push(u8::from(neighborhood.depth_match));
        visibility.push(u8::from(neighborhood.final_visibility));
    }
    let labels = CameraVisibilityDiagnostics {
        keypoints_2d,
        keypoint_visibility: visibility,
        keypoint_projectable: projectable,
        projected_mask,
        segmentation_matches: segmentation_matches_list,
        depth_matches: depth_matches_list,
        segmentation_mask,
        width,
        height,
        target_class_id,
    };
    profile.per_step_geometry += geometry_started_at.elapsed();
    Ok(labels)
}

fn project_keypoints_visibility_diagnostics_for_clean_camera(
    repository_config: &RepositoryConfig,
    keypoint_schema: &KeypointSchema,
    aircraft_mesh: &AircraftMesh,
    input: CleanCameraProjectionInput<'_>,
) -> Result<CameraVisibilityDiagnostics> {
    let observer_transform = aircraft_transform(input.observer);
    let follow = FollowPlayerCamera {
        offset: Vec3::from_array(repository_config.game.camera.follow_offset),
        rear_view_offset: Vec3::from_array(repository_config.game.camera.rear_view_offset),
    };
    let rear_view = matches!(input.camera_kind, VisualSensorKind::Rear);
    let (camera_position, look_direction, up) =
        resolve_follow_camera_pose(&observer_transform, &follow, rear_view);
    let mut camera_transform = Transform::from_translation(camera_position);
    camera_transform.look_to(look_direction, up);
    let aspect_ratio = repository_config.game.camera.aspect_width.max(1) as f32
        / repository_config.game.camera.aspect_height.max(1) as f32;
    let fov_y = repository_config.game.camera.fov_y_degrees.to_radians();
    let target_transform = aircraft_transform(input.target);
    let depth_buffer = rasterize_aircraft_depth_buffer(
        aircraft_mesh,
        &target_transform,
        &camera_transform,
        input.width,
        input.height,
        fov_y,
        aspect_ratio,
    );
    let target_class_id = segmentation_class_id_for_role(&input.target.role);
    let mut keypoints_2d = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut visibility = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut projectable = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut projected_mask = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut segmentation_matches_list = Vec::with_capacity(keypoint_schema.point_labels.len());
    let mut depth_matches_list = Vec::with_capacity(keypoint_schema.point_labels.len());
    for label in &keypoint_schema.point_labels {
        let Some(point_object) = keypoint_schema.points_3d_object.get(label) else {
            keypoints_2d.push([0.0, 0.0]);
            visibility.push(0);
            projectable.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        };
        let world = target_transform.transform_point(Vec3::from_array(*point_object));
        let Some(projected) = project_world_to_image_with_depth(
            &camera_transform,
            input.width,
            input.height,
            fov_y,
            aspect_ratio,
            world,
        ) else {
            keypoints_2d.push([0.0, 0.0]);
            visibility.push(0);
            projectable.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        };
        projectable.push(1);
        keypoints_2d.push(projected.pixel);
        if !projected.in_frame {
            visibility.push(0);
            projected_mask.push(0);
            segmentation_matches_list.push(0);
            depth_matches_list.push(0);
            continue;
        }
        projected_mask.push(1);
        let pixel_x = projected.pixel[0]
            .round()
            .clamp(0.0, input.width.saturating_sub(1) as f32) as usize;
        let pixel_y = projected.pixel[1]
            .round()
            .clamp(0.0, input.height.saturating_sub(1) as f32) as usize;
        let neighborhood = evaluate_keypoint_neighborhood_match(
            input.segmentation_mask,
            depth_buffer.as_slice(),
            input.width,
            input.height,
            pixel_x,
            pixel_y,
            target_class_id,
            projected.depth,
        );
        segmentation_matches_list.push(u8::from(neighborhood.segmentation_match));
        depth_matches_list.push(u8::from(neighborhood.depth_match));
        visibility.push(u8::from(neighborhood.final_visibility));
    }
    Ok(CameraVisibilityDiagnostics {
        keypoints_2d,
        keypoint_visibility: visibility,
        keypoint_projectable: projectable,
        projected_mask,
        segmentation_matches: segmentation_matches_list,
        depth_matches: depth_matches_list,
        segmentation_mask: input.segmentation_mask.to_vec(),
        width: input.width,
        height: input.height,
        target_class_id,
    })
}

fn synthetic_aircraft_from_metadata(
    metadata: &Value,
    role: &str,
    prefix: &str,
) -> Result<AircraftObservation> {
    let position = metadata_vec3(metadata, &format!("{prefix}_position_world"))?;
    let orientation = metadata_quat_xyzw(metadata, &format!("{prefix}_orientation_world_xyzw"))?;
    let orientation_quat = quat(orientation).normalize();
    Ok(AircraftObservation {
        role: role.to_string(),
        position,
        orientation_quat: orientation_quat.to_array(),
        linear_velocity: [0.0; 3],
        angular_velocity_deg: [0.0; 3],
        forward: (orientation_quat * Vec3::Z).normalize_or_zero().to_array(),
        throttle: 0.0,
        brake: false,
        stall_factor: 0.0,
        hit_points: 1.0,
        destroyed: false,
        out_of_bounds_seconds: 0.0,
        ceiling_recovery_seconds: 0.0,
        gun_heat: 0.0,
        gun_overheated: false,
        is_firing: false,
        repairing: false,
        repair_elapsed_seconds: 0.0,
        repair_progress: 0.0,
        velocity_turn_rate_rad_s: None,
        pullup_turn_radius_m: None,
        max_level_speed_mps: None,
        time_to_ground_impact_s: None,
        time_to_ceiling_impact_s: None,
        time_to_horizontal_boundary_impact_s: None,
        time_to_reenter_arena_s: None,
        subsystems: Vec::new(),
    })
}

fn synthetic_relative_pose_labels(
    observer: &AircraftObservation,
    target: &AircraftObservation,
) -> ([f32; 3], [f32; 6], [f32; 3], f32) {
    let observer_position = vec3(observer.position);
    let observer_orientation = quat(observer.orientation_quat).normalize();
    let target_position = vec3(target.position);
    let target_orientation = quat(target.orientation_quat).normalize();
    let relative_position = observer_orientation.inverse() * (target_position - observer_position);
    let relative_distance = relative_position.length().max(AUDIO_DISTANCE_EPSILON);
    let gt_doa_unit_vector_body =
        if relative_position.length_squared() > AUDIO_DISTANCE_EPSILON.powi(2) {
            (relative_position / relative_distance).to_array()
        } else {
            Vec3::Z.to_array()
        };
    let gt_log_distance_scalar = relative_distance.ln();
    let relative_orientation = (observer_orientation.inverse() * target_orientation).normalize();
    (
        relative_position.to_array(),
        rotation6d_from_quat(relative_orientation),
        gt_doa_unit_vector_body,
        gt_log_distance_scalar,
    )
}

fn read_clean_segmentation_mask(path: &Path, width: usize, height: usize) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    decode_pgm_gray8_payload(&bytes, width, height)
}

fn metadata_vec3(metadata: &Value, key: &str) -> Result<[f32; 3]> {
    let values = metadata
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("metadata missing array field {key}"))?;
    anyhow::ensure!(values.len() == 3, "metadata field {key} must have len 3");
    Ok([
        values[0]
            .as_f64()
            .with_context(|| format!("invalid {key}[0]"))? as f32,
        values[1]
            .as_f64()
            .with_context(|| format!("invalid {key}[1]"))? as f32,
        values[2]
            .as_f64()
            .with_context(|| format!("invalid {key}[2]"))? as f32,
    ])
}

fn metadata_quat_xyzw(metadata: &Value, key: &str) -> Result<[f32; 4]> {
    let values = metadata
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("metadata missing array field {key}"))?;
    anyhow::ensure!(values.len() == 4, "metadata field {key} must have len 4");
    Ok([
        values[0]
            .as_f64()
            .with_context(|| format!("invalid {key}[0]"))? as f32,
        values[1]
            .as_f64()
            .with_context(|| format!("invalid {key}[1]"))? as f32,
        values[2]
            .as_f64()
            .with_context(|| format!("invalid {key}[2]"))? as f32,
        values[3]
            .as_f64()
            .with_context(|| format!("invalid {key}[3]"))? as f32,
    ])
}

fn keypoint_depth_tolerance(depth: f32) -> f32 {
    KEYPOINT_DEPTH_EPSILON_MIN.max(depth.abs() * KEYPOINT_DEPTH_EPSILON_RELATIVE)
}

fn evaluate_keypoint_neighborhood_match(
    segmentation_mask: &[u8],
    depth_buffer: &[f32],
    width: usize,
    height: usize,
    pixel_x: usize,
    pixel_y: usize,
    target_class_id: u8,
    projected_depth: f32,
) -> KeypointNeighborhoodMatch {
    if width == 0 || height == 0 {
        return KeypointNeighborhoodMatch {
            segmentation_match: false,
            depth_match: false,
            final_visibility: false,
        };
    }

    let radius = KEYPOINT_SEGMENTATION_NEIGHBORHOOD_RADIUS;
    let x0 = pixel_x.saturating_sub(radius);
    let y0 = pixel_y.saturating_sub(radius);
    let x1 = (pixel_x + radius).min(width - 1);
    let y1 = (pixel_y + radius).min(height - 1);

    let mut segmentation_match = false;
    let mut depth_match = false;
    let mut final_visibility = false;

    for ny in y0..=y1 {
        for nx in x0..=x1 {
            let index = ny * width + nx;
            let class_matches = segmentation_mask
                .get(index)
                .copied()
                .map(|class_id| class_id == target_class_id)
                .unwrap_or(false);
            let surface_depth = depth_buffer.get(index).copied().unwrap_or(f32::INFINITY);
            let current_depth_match = if surface_depth.is_finite() {
                let tolerance = keypoint_depth_tolerance(projected_depth.max(surface_depth));
                (projected_depth - surface_depth).abs() <= tolerance
            } else {
                false
            };
            segmentation_match |= class_matches;
            depth_match |= current_depth_match;
            final_visibility |= class_matches && current_depth_match;
        }
    }

    KeypointNeighborhoodMatch {
        segmentation_match,
        depth_match,
        final_visibility,
    }
}

fn build_sparse_voting_payload(
    width: usize,
    height: usize,
    keypoints_2d: &[[f32; 2]],
    keypoint_projectable: &[u8],
    segmentation_mask: &[u8],
    target_class_id: u8,
) -> Result<SparseVotingPayload> {
    let expected_len = width
        .checked_mul(height)
        .ok_or_else(|| anyhow!("voting payload dimensions overflowed usize"))?;
    anyhow::ensure!(
        segmentation_mask.len() == expected_len,
        "segmentation mask len {} does not match {}x{}",
        segmentation_mask.len(),
        width,
        height,
    );
    anyhow::ensure!(
        keypoint_projectable.len() == keypoints_2d.len(),
        "keypoint projectable len {} does not match keypoint count {}",
        keypoint_projectable.len(),
        keypoints_2d.len(),
    );
    let pixel_count = segmentation_mask
        .iter()
        .filter(|&&value| value == target_class_id)
        .count();
    let keypoint_count = keypoints_2d.len();
    let bytes_per_pixel =
        2 * std::mem::size_of::<u16>() + keypoint_count * 2 * std::mem::size_of::<u16>();
    let mut bytes = Vec::with_capacity(pixel_count * bytes_per_pixel);
    let width_scale = width.saturating_sub(1).max(1) as f32;
    let height_scale = height.saturating_sub(1).max(1) as f32;

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if segmentation_mask[index] != target_class_id {
                continue;
            }
            let x_u16 = u16::try_from(x).with_context(|| format!("pixel x {x} exceeds u16"))?;
            let y_u16 = u16::try_from(y).with_context(|| format!("pixel y {y} exceeds u16"))?;
            bytes.extend_from_slice(&x_u16.to_le_bytes());
            bytes.extend_from_slice(&y_u16.to_le_bytes());

            let pixel_x = x as f32 / width_scale;
            let pixel_y = y as f32 / height_scale;
            for (keypoint, projectable) in keypoints_2d.iter().zip(keypoint_projectable.iter()) {
                let unit = if *projectable == 0 {
                    [0.0, 0.0]
                } else {
                    let keypoint_x = keypoint[0] / width_scale;
                    let keypoint_y = keypoint[1] / height_scale;
                    let dx = keypoint_x - pixel_x;
                    let dy = keypoint_y - pixel_y;
                    let norm = (dx * dx + dy * dy).sqrt();
                    if norm > 1.0e-6 {
                        [dx / norm, dy / norm]
                    } else {
                        [0.0, 0.0]
                    }
                };
                bytes.extend_from_slice(&f16::from_f32(unit[0]).to_bits().to_le_bytes());
                bytes.extend_from_slice(&f16::from_f32(unit[1]).to_bits().to_le_bytes());
            }
        }
    }

    Ok(SparseVotingPayload {
        bytes,
        pixel_count: pixel_count as u32,
    })
}

#[derive(Debug, Clone, Copy)]
struct ProjectedPoint {
    pixel: [f32; 2],
    depth: f32,
    in_frame: bool,
}

fn aircraft_transform(aircraft: &AircraftObservation) -> Transform {
    Transform {
        translation: Vec3::from_array(aircraft.position),
        rotation: quat(aircraft.orientation_quat).normalize(),
        ..Default::default()
    }
}

fn project_world_to_image_with_depth(
    camera_transform: &Transform,
    width: usize,
    height: usize,
    fov_y: f32,
    aspect_ratio: f32,
    world: Vec3,
) -> Option<ProjectedPoint> {
    let view = camera_transform.to_matrix().inverse();
    let local = view.transform_point3(world);
    let depth = -local.z;
    if depth <= 1e-3 {
        return None;
    }
    let tan_half_y = (fov_y * 0.5).tan().max(1e-6);
    let tan_half_x = tan_half_y * aspect_ratio.max(1e-6);
    let x_ndc = local.x / (depth * tan_half_x);
    let y_ndc = local.y / (depth * tan_half_y);
    let pixel_x = (x_ndc * 0.5 + 0.5) * (width.saturating_sub(1)) as f32;
    let pixel_y = (1.0 - (y_ndc * 0.5 + 0.5)) * (height.saturating_sub(1)) as f32;
    Some(ProjectedPoint {
        pixel: [pixel_x, pixel_y],
        depth,
        in_frame: (-1.0..=1.0).contains(&x_ndc) && (-1.0..=1.0).contains(&y_ndc),
    })
}

fn rasterize_aircraft_depth_buffer(
    mesh: &AircraftMesh,
    target_transform: &Transform,
    camera_transform: &Transform,
    width: usize,
    height: usize,
    fov_y: f32,
    aspect_ratio: f32,
) -> Vec<f32> {
    let mut depth_buffer = vec![f32::INFINITY; width * height];
    if width == 0 || height == 0 {
        return depth_buffer;
    }
    let view = camera_transform.to_matrix().inverse();
    let tan_half_y = (fov_y * 0.5).tan().max(1e-6);
    let tan_half_x = tan_half_y * aspect_ratio.max(1e-6);

    for triangle in &mesh.triangles {
        let mut projected = [[0.0_f32; 2]; 3];
        let mut depths = [0.0_f32; 3];
        let mut valid = true;
        for (index, vertex) in triangle.iter().enumerate() {
            let world = target_transform.transform_point(*vertex);
            let local = view.transform_point3(world);
            let depth = -local.z;
            if depth <= 1e-3 {
                valid = false;
                break;
            }
            let x_ndc = local.x / (depth * tan_half_x);
            let y_ndc = local.y / (depth * tan_half_y);
            let pixel_x = (x_ndc * 0.5 + 0.5) * (width.saturating_sub(1)) as f32;
            let pixel_y = (1.0 - (y_ndc * 0.5 + 0.5)) * (height.saturating_sub(1)) as f32;
            projected[index] = [pixel_x, pixel_y];
            depths[index] = depth;
        }
        if !valid {
            continue;
        }
        rasterize_triangle_depth(&projected, &depths, width, height, &mut depth_buffer);
    }
    depth_buffer
}

fn rasterize_triangle_depth(
    projected: &[[f32; 2]; 3],
    depths: &[f32; 3],
    width: usize,
    height: usize,
    depth_buffer: &mut [f32],
) {
    let min_x = projected
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let max_x = projected
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let min_y = projected
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, height.saturating_sub(1) as f32) as usize;
    let max_y = projected
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, height.saturating_sub(1) as f32) as usize;
    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_function(projected[0], projected[1], projected[2]);
    if area.abs() <= 1e-6 {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge_function(projected[1], projected[2], sample);
            let w1 = edge_function(projected[2], projected[0], sample);
            let w2 = edge_function(projected[0], projected[1], sample);
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if !inside {
                continue;
            }
            let bary0 = w0 / area;
            let bary1 = w1 / area;
            let bary2 = w2 / area;
            let depth = bary0 * depths[0] + bary1 * depths[1] + bary2 * depths[2];
            let index = y * width + x;
            if depth < depth_buffer[index] {
                depth_buffer[index] = depth;
            }
        }
    }
}

fn edge_function(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn segmentation_class_id_for_role(role: &str) -> u8 {
    match role {
        "fighter1" => 1,
        "fighter2" => 2,
        _ => 0,
    }
}

fn decode_segmentation_by_camera(
    derived_root: &Path,
    artifacts: &[crate::recording::VisualArtifactRef],
    camera: VisualSensorKind,
    fallback_width: usize,
    fallback_height: usize,
    visual_bundle_cache: &mut VisualArtifactBundleCache,
) -> Result<Vec<u8>> {
    let Some(artifact) = artifacts.iter().find(|artifact| artifact.camera == camera) else {
        bail!("missing {:?} segmentation artifact", camera);
    };
    let width = artifact.width.unwrap_or(fallback_width as u32) as usize;
    let height = artifact.height.unwrap_or(fallback_height as u32) as usize;
    let bytes = visual_bundle_cache.read_slice(derived_root, artifact)?;
    match artifact
        .format
        .unwrap_or(crate::api::types::PixelFormat::Gray8)
    {
        crate::api::types::PixelFormat::Gray8 => decode_gray8_to_segmentation(bytes, width, height),
        crate::api::types::PixelFormat::Rgb8 => decode_rgb8_to_segmentation(bytes, width, height),
        crate::api::types::PixelFormat::Rgba8 => {
            let expected = width * height * 4;
            anyhow::ensure!(
                bytes.len() == expected,
                "unexpected RGBA segmentation length: got {}, expected {}",
                bytes.len(),
                expected
            );
            Ok(sanitize_segmentation_classes(
                bytes.chunks_exact(4).map(|chunk| chunk[0]).collect(),
            ))
        }
    }
}

fn decode_gray8_to_segmentation(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    if bytes.len() == width * height {
        return Ok(sanitize_segmentation_classes(bytes.to_vec()));
    }
    Ok(sanitize_segmentation_classes(decode_pgm_gray8_payload(
        bytes, width, height,
    )?))
}

fn decode_rgb8_to_segmentation(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let payload = if bytes.len() == width * height * 3 {
        bytes.to_vec()
    } else {
        decode_ppm_rgb8_payload(bytes, width, height)?
    };
    Ok(sanitize_segmentation_classes(
        payload.chunks_exact(3).map(|chunk| chunk[0]).collect(),
    ))
}

fn sanitize_segmentation_classes(values: Vec<u8>) -> Vec<u8> {
    values
        .into_iter()
        .map(|value| match value {
            0..=2 => value,
            _ => 0,
        })
        .collect()
}

fn decode_pgm_gray8_payload(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let magic = b"P5";
    anyhow::ensure!(
        bytes.starts_with(magic),
        "unsupported PGM payload: missing P5 header"
    );
    let mut index = 2_usize;
    let mut tokens = Vec::new();
    while tokens.len() < 3 {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            bail!("truncated PGM header");
        }
        if bytes[index] == b'#' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        let start = index;
        while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        tokens.push(std::str::from_utf8(&bytes[start..index])?.to_string());
    }
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let header_width = tokens[0].parse::<usize>()?;
    let header_height = tokens[1].parse::<usize>()?;
    let max_value = tokens[2].parse::<u32>()?;
    anyhow::ensure!(
        header_width == width && header_height == height,
        "PGM dimension mismatch"
    );
    anyhow::ensure!(max_value == 255, "unsupported PGM max value {max_value}");
    let expected = width * height;
    anyhow::ensure!(
        bytes.len().saturating_sub(index) == expected,
        "PGM payload length mismatch: got {}, expected {}",
        bytes.len().saturating_sub(index),
        expected
    );
    Ok(bytes[index..].to_vec())
}

fn decode_ppm_rgb8_payload(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let magic = b"P6";
    anyhow::ensure!(
        bytes.starts_with(magic),
        "unsupported PPM payload: missing P6 header"
    );
    let mut index = 2_usize;
    let mut tokens = Vec::new();
    while tokens.len() < 3 {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() {
            bail!("truncated PPM header");
        }
        if bytes[index] == b'#' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        let start = index;
        while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        tokens.push(std::str::from_utf8(&bytes[start..index])?.to_string());
    }
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let header_width = tokens[0].parse::<usize>()?;
    let header_height = tokens[1].parse::<usize>()?;
    let max_value = tokens[2].parse::<u32>()?;
    anyhow::ensure!(
        header_width == width && header_height == height,
        "PPM dimension mismatch"
    );
    anyhow::ensure!(max_value == 255, "unsupported PPM max value {max_value}");
    let expected = width * height * 3;
    anyhow::ensure!(
        bytes.len().saturating_sub(index) == expected,
        "PPM payload length mismatch: got {}, expected {}",
        bytes.len().saturating_sub(index),
        expected
    );
    Ok(bytes[index..].to_vec())
}

fn raw_audio_features_for_step(
    derived_root: &Path,
    audio_schema: &AudioCueSchema,
    artifact: Option<&crate::recording::AudioArtifactRef>,
) -> Result<RawAudioFeatures> {
    let Some(artifact) = artifact else {
        return Ok(RawAudioFeatures {
            binaural_energy_t: [0.0; 4],
            binaural_cue_vector_t: [0.0; 10],
        });
    };
    let bytes = read_relative_audio_artifact_bytes(derived_root, artifact)?;
    if bytes.is_empty() {
        return Ok(RawAudioFeatures {
            binaural_energy_t: [0.0; 4],
            binaural_cue_vector_t: [0.0; 10],
        });
    }
    let wav = parse_wav_i16(&bytes)?;
    anyhow::ensure!(
        audio_schema.layout == "binaural",
        "unsupported audio cue layout {}; expected binaural",
        audio_schema.layout
    );
    anyhow::ensure!(
        audio_schema.sample_rate_hz == 0 || wav.sample_rate == audio_schema.sample_rate_hz,
        "unexpected audio sample rate {}; expected {}",
        wav.sample_rate,
        audio_schema.sample_rate_hz
    );
    anyhow::ensure!(
        audio_schema.channel_order == ["L", "R"],
        "unexpected binaural channel order {:?}",
        audio_schema.channel_order
    );
    anyhow::ensure!(
        audio_schema.energy_features.order == ["E_L", "E_R", "E_sum", "E_diff_norm"],
        "unexpected binaural energy feature order {:?}",
        audio_schema.energy_features.order
    );
    anyhow::ensure!(
        audio_schema.cue_features.order
            == [
                "gcc_peak_lag",
                "gcc_peak_value",
                "ild",
                "ipd_low",
                "ipd_mid",
                "interaural_coherence",
                "reverb_ratio_proxy",
                "directness_proxy",
                "ild_low_band_db",
                "ild_high_band_db"
            ],
        "unexpected binaural cue feature order {:?}",
        audio_schema.cue_features.order
    );
    if wav.channels < 2 {
        return Ok(RawAudioFeatures {
            binaural_energy_t: [0.0; 4],
            binaural_cue_vector_t: [0.0; 10],
        });
    }
    let frame_count = wav.samples.len() / wav.channels as usize;
    if frame_count == 0 {
        return Ok(RawAudioFeatures {
            binaural_energy_t: [0.0; 4],
            binaural_cue_vector_t: [0.0; 10],
        });
    }

    let mut left = Vec::with_capacity(frame_count);
    let mut right = Vec::with_capacity(frame_count);
    for frame in wav.samples.chunks_exact(wav.channels as usize) {
        left.push(frame[0]);
        right.push(frame[1]);
    }
    let energy = compute_binaural_energy(&left, &right);
    let cues = compute_binaural_cues(&left, &right, wav.sample_rate, energy[2]);
    Ok(RawAudioFeatures {
        binaural_energy_t: energy,
        binaural_cue_vector_t: cues,
    })
}

fn read_relative_audio_artifact_bytes(
    derived_root: &Path,
    artifact: &crate::recording::AudioArtifactRef,
) -> Result<Vec<u8>> {
    let Some(relative_path) = &artifact.file_path else {
        return Ok(Vec::new());
    };
    let path = derived_root.join(relative_path);
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let Some(offset) = artifact.byte_offset else {
        return Ok(bytes);
    };
    let length = artifact
        .byte_length
        .with_context(|| "audio artifact has byte_offset but missing byte_length")?;
    let start = offset as usize;
    let end = start
        .checked_add(length as usize)
        .with_context(|| "audio artifact byte range overflowed usize")?;
    bytes
        .get(start..end)
        .map(|slice| slice.to_vec())
        .with_context(|| {
            format!(
                "audio artifact byte range [{start}..{end}) is out of bounds for {} bytes",
                bytes.len()
            )
        })
}

#[derive(Debug)]
struct ParsedWav {
    channels: u16,
    sample_rate: u32,
    samples: Vec<f32>,
}

fn parse_wav_i16(bytes: &[u8]) -> Result<ParsedWav> {
    anyhow::ensure!(bytes.len() >= 44, "wav payload too small");
    anyhow::ensure!(&bytes[0..4] == b"RIFF", "missing RIFF header");
    anyhow::ensure!(&bytes[8..12] == b"WAVE", "missing WAVE header");
    anyhow::ensure!(&bytes[12..16] == b"fmt ", "unsupported wav layout");
    let audio_format = u16::from_le_bytes([bytes[20], bytes[21]]);
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits_per_sample = u16::from_le_bytes([bytes[34], bytes[35]]);
    anyhow::ensure!(audio_format == 1, "only PCM wav is supported");
    anyhow::ensure!(bits_per_sample == 16, "only 16-bit wav is supported");

    let mut cursor = 12usize;
    let mut data_range = None;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        let chunk_data_start = cursor + 8;
        let chunk_data_end = chunk_data_start
            .checked_add(chunk_len)
            .with_context(|| "wav chunk length overflowed")?;
        anyhow::ensure!(chunk_data_end <= bytes.len(), "wav chunk exceeds payload");
        if chunk_id == b"data" {
            data_range = Some(chunk_data_start..chunk_data_end);
            break;
        }
        cursor = chunk_data_end + (chunk_len % 2);
    }
    let data_range = data_range.ok_or_else(|| anyhow!("wav payload is missing data chunk"))?;
    let mut samples = Vec::with_capacity((data_range.len() / 2).max(1));
    for chunk in bytes[data_range].chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32;
        samples.push(sample);
    }
    Ok(ParsedWav {
        channels,
        sample_rate,
        samples,
    })
}

fn find_episode_roots(recordings_root: &Path) -> Result<Vec<PathBuf>> {
    if !recordings_root.exists() {
        return Ok(Vec::new());
    }
    let mut episodes = fs::read_dir(recordings_root)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    episodes.sort();
    Ok(episodes)
}

fn parse_args<I>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut episode_roots = Vec::new();
    let mut recordings_root = ConfigPaths::default().recordings_root();
    let mut observed_roles = Vec::new();
    let mut force = false;
    let mut profile = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --episode");
                };
                episode_roots.push(PathBuf::from(value));
            }
            "--recordings-root" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --recordings-root");
                };
                recordings_root = PathBuf::from(value);
            }
            "--observed-role" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --observed-role");
                };
                observed_roles = parse_roles(&value)?;
            }
            "--force" => force = true,
            "--profile" => profile = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(CliArgs {
        episode_roots,
        recordings_root,
        observed_roles,
        force,
        profile,
    })
}

fn parse_visibility_audit_args<I>(args: I) -> Result<VisibilityAuditArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut episode_root: Option<PathBuf> = None;
    let mut observed_role: Option<String> = None;
    let mut max_steps: Option<usize> = None;
    let mut output_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --episode");
                };
                episode_root = Some(PathBuf::from(value));
            }
            "--observed-role" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --observed-role");
                };
                let mut roles = parse_roles(&value)?;
                if roles.len() != 1 {
                    bail!("audit-visibility requires exactly one observed role");
                }
                observed_role = Some(roles.remove(0));
            }
            "--max-steps" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --max-steps");
                };
                max_steps = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("invalid --max-steps value: {value}"))?,
                );
            }
            "--output" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --output");
                };
                output_path = Some(PathBuf::from(value));
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let episode_root = episode_root.ok_or_else(|| anyhow!("missing required --episode"))?;
    let observed_role = observed_role.ok_or_else(|| anyhow!("missing required --observed-role"))?;
    Ok(VisibilityAuditArgs {
        episode_root,
        observed_role,
        max_steps,
        output_path,
    })
}

fn parse_synthetic_single_step_args<I>(args: I) -> Result<SyntheticSingleStepArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut source_root: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut force = false;
    let mut max_samples: Option<usize> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--source-root" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --source-root");
                };
                source_root = Some(PathBuf::from(value));
            }
            "--output-dir" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --output-dir");
                };
                output_dir = Some(PathBuf::from(value));
            }
            "--max-samples" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --max-samples");
                };
                max_samples = Some(
                    value
                        .parse::<usize>()
                        .with_context(|| format!("invalid --max-samples value: {value}"))?,
                );
            }
            "--force" => force = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(SyntheticSingleStepArgs {
        source_root: source_root.ok_or_else(|| anyhow!("missing required --source-root"))?,
        output_dir: output_dir.ok_or_else(|| anyhow!("missing required --output-dir"))?,
        force,
        max_samples,
    })
}

fn print_label_profile_report(profile: &LabelProfile) {
    let total = profile.load_assets
        + profile.load_episode_and_manifests
        + profile.raw_audio_feature_preload
        + profile.per_step_visual_decode
        + profile.per_step_geometry
        + profile.manifest_serialize;
    println!("label profile:");
    print_profile_stage("  load_assets", profile.load_assets, total);
    print_profile_stage(
        "  load_episode_and_manifests",
        profile.load_episode_and_manifests,
        total,
    );
    print_profile_stage(
        "  raw_audio_feature_preload",
        profile.raw_audio_feature_preload,
        total,
    );
    print_profile_stage(
        "  per_step_visual_decode",
        profile.per_step_visual_decode,
        total,
    );
    print_profile_stage("  per_step_geometry", profile.per_step_geometry, total);
    print_profile_stage("  manifest_serialize", profile.manifest_serialize, total);
    if profile.step_count > 0 {
        println!("  step_count: {}", profile.step_count);
        println!(
            "  per_step_visual_decode_avg_ms: {:.3}",
            duration_ms(profile.per_step_visual_decode) / profile.step_count as f64
        );
        println!(
            "  per_step_geometry_avg_ms: {:.3}",
            duration_ms(profile.per_step_geometry) / profile.step_count as f64
        );
    }
}

fn print_profile_stage(label: &str, duration: Duration, total: Duration) {
    let total_ms = duration_ms(total);
    let duration_ms = duration_ms(duration);
    let share = if total_ms > 0.0 {
        duration_ms / total_ms * 100.0
    } else {
        0.0
    };
    println!("{label}: {duration_ms:.3} ms ({share:.1}%)");
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn parse_roles(value: &str) -> Result<Vec<String>> {
    if value.eq_ignore_ascii_case("all") {
        return Ok(Vec::new());
    }

    value
        .split(',')
        .map(|role| role.trim().to_ascii_lowercase())
        .map(|role| match role.as_str() {
            "fighter1" | "fighter2" => Ok(role),
            other => bail!("unsupported observed role: {other}"),
        })
        .collect()
}

struct ResolvedAssets {
    coordinate_convention_id: String,
    keypoint_schema: KeypointSchema,
    audio_schema: AudioCueSchema,
    aircraft_mesh: AircraftMesh,
}

fn load_assets(project_root: &Path) -> Result<ResolvedAssets> {
    let asset_ids_path = project_root.join("config/dfb_state_estimation/ids/asset_ids.json");
    let asset_ids: AssetIdsManifest = serde_json::from_str(
        &fs::read_to_string(&asset_ids_path)
            .with_context(|| format!("failed to read {}", asset_ids_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", asset_ids_path.display()))?;

    let coordinate_convention_id = asset_ids
        .coordinate_conventions
        .iter()
        .find(|item| item.default)
        .or_else(|| asset_ids.coordinate_conventions.first())
        .map(|item| item.id.clone())
        .ok_or_else(|| anyhow!("missing default coordinate convention in asset_ids.json"))?;
    let keypoint_schema_path = asset_ids
        .keypoint_schemas
        .iter()
        .find(|item| item.default)
        .or_else(|| asset_ids.keypoint_schemas.first())
        .and_then(|item| item.path.clone())
        .ok_or_else(|| anyhow!("missing default keypoint schema path in asset_ids.json"))?;
    let keypoint_schema_path = project_root.join(keypoint_schema_path);
    let keypoint_schema: KeypointSchema = serde_json::from_str(
        &fs::read_to_string(&keypoint_schema_path)
            .with_context(|| format!("failed to read {}", keypoint_schema_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", keypoint_schema_path.display()))?;
    anyhow::ensure!(
        keypoint_schema.coordinate_convention_id == coordinate_convention_id,
        "keypoint schema convention {} does not match canonical convention {}",
        keypoint_schema.coordinate_convention_id,
        coordinate_convention_id
    );
    let audio_schema_path = asset_ids
        .audio_cue_schemas
        .iter()
        .find(|item| item.default)
        .or_else(|| asset_ids.audio_cue_schemas.first())
        .and_then(|item| item.path.clone())
        .ok_or_else(|| anyhow!("missing default audio cue schema path in asset_ids.json"))?;
    let audio_schema_path = project_root.join(audio_schema_path);
    let audio_schema: AudioCueSchema = serde_json::from_str(
        &fs::read_to_string(&audio_schema_path)
            .with_context(|| format!("failed to read {}", audio_schema_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", audio_schema_path.display()))?;
    let aircraft_mesh = load_aircraft_mesh(project_root)?;

    Ok(ResolvedAssets {
        coordinate_convention_id,
        keypoint_schema,
        audio_schema,
        aircraft_mesh,
    })
}

fn compute_binaural_energy(left: &[f32], right: &[f32]) -> [f32; 4] {
    let frame_count = left.len().min(right.len()).max(1) as f32;
    let e_l = left.iter().map(|sample| sample * sample).sum::<f32>() / frame_count;
    let e_r = right.iter().map(|sample| sample * sample).sum::<f32>() / frame_count;
    let e_sum = e_l + e_r;
    let e_diff_norm = (e_l - e_r) / (e_sum + 1e-6);
    [e_l, e_r, e_sum, e_diff_norm]
}

fn compute_binaural_cues(
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
    total_energy: f32,
) -> [f32; 10] {
    let (gcc_peak_lag, gcc_peak_value) = dominant_gcc_peak(left, right, sample_rate);
    let ild = 10.0 * ((mean_square(left) + 1e-6) / (mean_square(right) + 1e-6)).log10();
    let ild_low = band_ild_db(left, right, sample_rate, 200.0, 1_500.0);
    let ild_high = band_ild_db(left, right, sample_rate, 1_500.0, 6_000.0);
    let ipd_low = normalized_band_ipd(left, right, sample_rate, 200.0, 1_500.0);
    let ipd_mid = normalized_band_ipd(left, right, sample_rate, 1_500.0, 6_000.0);
    let interaural_coherence = interaural_coherence(left, right);
    let directness_proxy = directness_proxy(left, right);
    let reverb_ratio_proxy = ((1.0 - directness_proxy).max(0.0)) / (directness_proxy + 1e-6);
    let cue_floor = if total_energy > 1e-6 { 1.0 } else { 0.0 };
    [
        gcc_peak_lag * cue_floor,
        gcc_peak_value * cue_floor,
        ild * cue_floor,
        ipd_low * cue_floor,
        ipd_mid * cue_floor,
        interaural_coherence * cue_floor,
        reverb_ratio_proxy * cue_floor,
        directness_proxy * cue_floor,
        ild_low * cue_floor,
        ild_high * cue_floor,
    ]
}

fn band_ild_db(left: &[f32], right: &[f32], sample_rate: u32, low_hz: f32, high_hz: f32) -> f32 {
    let frames = left.len().min(right.len());
    if frames < 2 || sample_rate == 0 {
        return 0.0;
    }
    let mut left_power = 0.0_f32;
    let mut right_power = 0.0_f32;
    let mut bin_count = 0_u32;
    let nyquist_bins = frames / 2;
    for bin in 1..nyquist_bins {
        let freq_hz = bin as f32 * sample_rate as f32 / frames as f32;
        if freq_hz < low_hz || freq_hz >= high_hz {
            continue;
        }
        let mut l_re = 0.0_f32;
        let mut l_im = 0.0_f32;
        let mut r_re = 0.0_f32;
        let mut r_im = 0.0_f32;
        for index in 0..frames {
            let angle = 2.0 * PI * bin as f32 * index as f32 / frames as f32;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            l_re += left[index] * cos_angle;
            l_im -= left[index] * sin_angle;
            r_re += right[index] * cos_angle;
            r_im -= right[index] * sin_angle;
        }
        let left_mag_sq = l_re * l_re + l_im * l_im;
        let right_mag_sq = r_re * r_re + r_im * r_im;
        left_power += left_mag_sq;
        right_power += right_mag_sq;
        bin_count += 1;
    }
    if bin_count == 0 {
        return 0.0;
    }
    left_power /= bin_count as f32;
    right_power /= bin_count as f32;
    10.0 * ((left_power + 1e-6) / (right_power + 1e-6)).log10()
}

fn mean_square(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    signal.iter().map(|sample| sample * sample).sum::<f32>() / signal.len() as f32
}

fn dominant_gcc_peak(left: &[f32], right: &[f32], sample_rate: u32) -> (f32, f32) {
    let frames = left.len().min(right.len());
    if frames == 0 {
        return (0.0, 0.0);
    }
    let norm = (left.iter().map(|v| v * v).sum::<f32>() * right.iter().map(|v| v * v).sum::<f32>())
        .sqrt()
        .max(1e-6);
    let max_lag = ((sample_rate as f32) * 0.001).round().max(1.0) as isize;
    let mut best_lag = 0_isize;
    let mut best_value = 0.0_f32;
    for lag in -max_lag..=max_lag {
        let mut corr = 0.0_f32;
        for index in 0..frames {
            let right_index = index as isize + lag;
            if !(0..frames as isize).contains(&right_index) {
                continue;
            }
            corr += left[index] * right[right_index as usize];
        }
        let value = (corr / norm).abs();
        if value > best_value {
            best_value = value;
            best_lag = lag;
        }
    }
    (
        best_lag as f32 / sample_rate.max(1) as f32,
        best_value.clamp(0.0, 1.0),
    )
}

fn normalized_band_ipd(
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
    low_hz: f32,
    high_hz: f32,
) -> f32 {
    let frames = left.len().min(right.len());
    if frames < 2 || sample_rate == 0 {
        return 0.0;
    }
    let mut re_sum = 0.0_f32;
    let mut im_sum = 0.0_f32;
    let nyquist_bins = frames / 2;
    for bin in 1..nyquist_bins {
        let freq_hz = bin as f32 * sample_rate as f32 / frames as f32;
        if freq_hz < low_hz || freq_hz >= high_hz {
            continue;
        }
        let mut l_re = 0.0_f32;
        let mut l_im = 0.0_f32;
        let mut r_re = 0.0_f32;
        let mut r_im = 0.0_f32;
        for index in 0..frames {
            let angle = 2.0 * PI * bin as f32 * index as f32 / frames as f32;
            let cos_angle = angle.cos();
            let sin_angle = angle.sin();
            l_re += left[index] * cos_angle;
            l_im -= left[index] * sin_angle;
            r_re += right[index] * cos_angle;
            r_im -= right[index] * sin_angle;
        }
        let cross_re = l_re * r_re + l_im * r_im;
        let cross_im = l_im * r_re - l_re * r_im;
        let magnitude = (cross_re * cross_re + cross_im * cross_im).sqrt().max(1e-6);
        re_sum += cross_re / magnitude;
        im_sum += cross_im / magnitude;
    }
    if re_sum == 0.0 && im_sum == 0.0 {
        0.0
    } else {
        im_sum.atan2(re_sum) / PI
    }
}

fn interaural_coherence(left: &[f32], right: &[f32]) -> f32 {
    let frames = left.len().min(right.len());
    if frames == 0 {
        return 0.0;
    }
    let cross = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum::<f32>()
        .abs();
    let norm = (left.iter().map(|v| v * v).sum::<f32>() * right.iter().map(|v| v * v).sum::<f32>())
        .sqrt()
        .max(1e-6);
    (cross / norm).clamp(0.0, 1.0)
}

fn directness_proxy(left: &[f32], right: &[f32]) -> f32 {
    let frames = left.len().min(right.len());
    if frames == 0 {
        return 0.0;
    }
    let mono = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| 0.5 * (l + r))
        .collect::<Vec<_>>();
    let segment_count = 8_usize.min(frames).max(1);
    let segment_len = (frames / segment_count).max(1);
    let mut total = 0.0_f32;
    let mut peak = 0.0_f32;
    for segment in mono.chunks(segment_len).take(segment_count) {
        let energy = segment.iter().map(|sample| sample * sample).sum::<f32>();
        total += energy;
        peak = peak.max(energy);
    }
    if total <= 1e-6 {
        0.0
    } else {
        (peak / total).clamp(0.0, 1.0)
    }
}

fn load_aircraft_mesh(project_root: &Path) -> Result<AircraftMesh> {
    let gltf_path = project_root.join("assets/models/fighter_plane.gltf");
    let gltf_json: Value = serde_json::from_slice(
        &fs::read(&gltf_path).with_context(|| format!("failed to read {}", gltf_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", gltf_path.display()))?;
    let Some(buffer_uri) = gltf_json["buffers"][0]["uri"].as_str() else {
        bail!("fighter_plane.gltf missing primary buffer uri");
    };
    let buffer_path = gltf_path
        .parent()
        .ok_or_else(|| anyhow!("fighter_plane.gltf has no parent directory"))?
        .join(buffer_uri);
    let buffer = fs::read(&buffer_path)
        .with_context(|| format!("failed to read {}", buffer_path.display()))?;

    let nodes = gltf_json["nodes"]
        .as_array()
        .ok_or_else(|| anyhow!("fighter_plane.gltf missing nodes"))?;
    let meshes = gltf_json["meshes"]
        .as_array()
        .ok_or_else(|| anyhow!("fighter_plane.gltf missing meshes"))?;
    let accessors = gltf_json["accessors"]
        .as_array()
        .ok_or_else(|| anyhow!("fighter_plane.gltf missing accessors"))?;
    let buffer_views = gltf_json["bufferViews"]
        .as_array()
        .ok_or_else(|| anyhow!("fighter_plane.gltf missing bufferViews"))?;

    let mut triangles = Vec::new();
    for node in nodes {
        let Some(mesh_index) = node["mesh"].as_u64() else {
            continue;
        };
        let translation = node["translation"]
            .as_array()
            .map(|values| parse_vec3_json(values))
            .transpose()?
            .unwrap_or(Vec3::ZERO);
        let mesh = meshes
            .get(mesh_index as usize)
            .ok_or_else(|| anyhow!("mesh index {} out of bounds", mesh_index))?;
        let primitives = mesh["primitives"]
            .as_array()
            .ok_or_else(|| anyhow!("mesh {} missing primitives", mesh_index))?;
        for primitive in primitives {
            let position_accessor_index = primitive["attributes"]["POSITION"]
                .as_u64()
                .ok_or_else(|| anyhow!("primitive missing POSITION accessor"))?
                as usize;
            let indices_accessor_index = primitive["indices"]
                .as_u64()
                .ok_or_else(|| anyhow!("primitive missing indices accessor"))?
                as usize;
            let positions =
                read_accessor_positions(&buffer, accessors, buffer_views, position_accessor_index)?;
            let indices = read_accessor_indices_u32(
                &buffer,
                accessors,
                buffer_views,
                indices_accessor_index,
            )?;
            anyhow::ensure!(
                indices.len() % 3 == 0,
                "mesh indices are not triangle-aligned: {}",
                indices.len()
            );
            for triangle in indices.chunks_exact(3) {
                let a = positions
                    .get(triangle[0] as usize)
                    .copied()
                    .ok_or_else(|| anyhow!("triangle index out of bounds"))?
                    + translation;
                let b = positions
                    .get(triangle[1] as usize)
                    .copied()
                    .ok_or_else(|| anyhow!("triangle index out of bounds"))?
                    + translation;
                let c = positions
                    .get(triangle[2] as usize)
                    .copied()
                    .ok_or_else(|| anyhow!("triangle index out of bounds"))?
                    + translation;
                triangles.push([a, b, c]);
            }
        }
    }
    anyhow::ensure!(
        !triangles.is_empty(),
        "failed to load any aircraft mesh triangles from fighter_plane.gltf"
    );
    Ok(AircraftMesh { triangles })
}

fn parse_vec3_json(values: &[Value]) -> Result<Vec3> {
    anyhow::ensure!(values.len() == 3, "expected vec3 JSON array");
    Ok(Vec3::new(
        values[0]
            .as_f64()
            .ok_or_else(|| anyhow!("invalid vec3 x"))? as f32,
        values[1]
            .as_f64()
            .ok_or_else(|| anyhow!("invalid vec3 y"))? as f32,
        values[2]
            .as_f64()
            .ok_or_else(|| anyhow!("invalid vec3 z"))? as f32,
    ))
}

fn read_accessor_positions(
    buffer: &[u8],
    accessors: &[Value],
    buffer_views: &[Value],
    accessor_index: usize,
) -> Result<Vec<Vec3>> {
    let accessor = accessors
        .get(accessor_index)
        .ok_or_else(|| anyhow!("accessor {} out of bounds", accessor_index))?;
    anyhow::ensure!(
        accessor["componentType"].as_u64() == Some(5126),
        "POSITION accessor must use float32"
    );
    anyhow::ensure!(
        accessor["type"].as_str() == Some("VEC3"),
        "POSITION accessor must be VEC3"
    );
    let count = accessor["count"]
        .as_u64()
        .ok_or_else(|| anyhow!("POSITION accessor missing count"))? as usize;
    let bytes = accessor_bytes(buffer, accessor, buffer_views)?;
    anyhow::ensure!(
        bytes.len() >= count * 12,
        "POSITION accessor byte length too small: got {}, expected at least {}",
        bytes.len(),
        count * 12
    );
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * 12;
        let x = f32::from_le_bytes(bytes[start..start + 4].try_into()?);
        let y = f32::from_le_bytes(bytes[start + 4..start + 8].try_into()?);
        let z = f32::from_le_bytes(bytes[start + 8..start + 12].try_into()?);
        out.push(Vec3::new(x, y, z));
    }
    Ok(out)
}

fn read_accessor_indices_u32(
    buffer: &[u8],
    accessors: &[Value],
    buffer_views: &[Value],
    accessor_index: usize,
) -> Result<Vec<u32>> {
    let accessor = accessors
        .get(accessor_index)
        .ok_or_else(|| anyhow!("accessor {} out of bounds", accessor_index))?;
    let component_type = accessor["componentType"]
        .as_u64()
        .ok_or_else(|| anyhow!("index accessor missing componentType"))?;
    anyhow::ensure!(
        accessor["type"].as_str() == Some("SCALAR"),
        "index accessor must be scalar"
    );
    let count = accessor["count"]
        .as_u64()
        .ok_or_else(|| anyhow!("index accessor missing count"))? as usize;
    let bytes = accessor_bytes(buffer, accessor, buffer_views)?;
    match component_type {
        5125 => {
            anyhow::ensure!(
                bytes.len() >= count * 4,
                "u32 index accessor byte length too small"
            );
            let mut out = Vec::with_capacity(count);
            for index in 0..count {
                let start = index * 4;
                out.push(u32::from_le_bytes(bytes[start..start + 4].try_into()?));
            }
            Ok(out)
        }
        5123 => {
            anyhow::ensure!(
                bytes.len() >= count * 2,
                "u16 index accessor byte length too small"
            );
            let mut out = Vec::with_capacity(count);
            for index in 0..count {
                let start = index * 2;
                out.push(u16::from_le_bytes(bytes[start..start + 2].try_into()?) as u32);
            }
            Ok(out)
        }
        other => bail!("unsupported index accessor component type {}", other),
    }
}

fn accessor_bytes<'a>(
    buffer: &'a [u8],
    accessor: &Value,
    buffer_views: &[Value],
) -> Result<&'a [u8]> {
    let buffer_view_index = accessor["bufferView"]
        .as_u64()
        .ok_or_else(|| anyhow!("accessor missing bufferView"))?
        as usize;
    let buffer_view = buffer_views
        .get(buffer_view_index)
        .ok_or_else(|| anyhow!("bufferView {} out of bounds", buffer_view_index))?;
    let view_offset = buffer_view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let view_length = buffer_view["byteLength"]
        .as_u64()
        .ok_or_else(|| anyhow!("bufferView missing byteLength"))? as usize;
    let accessor_offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
    let start = view_offset + accessor_offset;
    let end = view_offset + view_length;
    buffer
        .get(start..end)
        .ok_or_else(|| anyhow!("accessor byte range out of bounds"))
}

fn find_aircraft<'a>(state: &'a StateObservation, role: &str) -> Result<&'a AircraftObservation> {
    state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role.eq_ignore_ascii_case(role))
        .with_context(|| format!("missing aircraft observation for role {role}"))
}

fn opposite_role_name(role: &str) -> &'static str {
    if role.eq_ignore_ascii_case("fighter1") {
        "fighter2"
    } else {
        "fighter1"
    }
}

fn vec3(value: [f32; 3]) -> Vec3 {
    Vec3::from_array(value)
}

fn quat(value: [f32; 4]) -> Quat {
    Quat::from_xyzw(value[0], value[1], value[2], value[3])
}

fn quat_to_array(value: Quat) -> [f32; 4] {
    [value.x, value.y, value.z, value.w]
}

fn rotation6d_from_quat(quat: Quat) -> [f32; 6] {
    let mat = bevy::math::Mat3::from_quat(quat.normalize());
    let x = mat.x_axis;
    let y = mat.y_axis;
    [x.x, x.y, x.z, y.x, y.y, y.z]
}

fn so3_log(rotation: Quat) -> Vec3 {
    let rotation = if rotation.w < 0.0 {
        Quat::from_xyzw(-rotation.x, -rotation.y, -rotation.z, -rotation.w)
    } else {
        rotation
    }
    .normalize();
    let angle = 2.0 * rotation.w.clamp(-1.0, 1.0).acos();
    let sin_half = (1.0 - rotation.w * rotation.w).sqrt();
    if angle <= 1e-6 || sin_half <= 1e-6 {
        Vec3::ZERO
    } else {
        Vec3::new(rotation.x, rotation.y, rotation.z) / sin_half * angle
    }
}

fn rotation_angle_between(a: Quat, b: Quat) -> f32 {
    let delta = (a * b.inverse()).normalize();
    2.0 * delta.w.abs().clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_segmentation_classes_clamps_invalid_values_to_background() {
        let values = vec![0, 1, 2, 6, 7, 8, 255];
        let sanitized = sanitize_segmentation_classes(values);
        assert_eq!(sanitized, vec![0, 1, 2, 0, 0, 0, 0]);
    }

    #[test]
    fn rasterized_depth_buffer_prefers_front_surface() {
        let mesh = AircraftMesh {
            triangles: vec![[
                Vec3::new(-1.0, -1.0, -5.0),
                Vec3::new(1.0, -1.0, -5.0),
                Vec3::new(0.0, 1.0, -5.0),
            ]],
        };
        let target_transform = Transform::IDENTITY;
        let camera_transform = Transform::IDENTITY;
        let width = 32;
        let height = 32;
        let fov_y = 90.0_f32.to_radians();
        let aspect_ratio = 1.0;

        let depth_buffer = rasterize_aircraft_depth_buffer(
            &mesh,
            &target_transform,
            &camera_transform,
            width,
            height,
            fov_y,
            aspect_ratio,
        );
        let front_point = project_world_to_image_with_depth(
            &camera_transform,
            width,
            height,
            fov_y,
            aspect_ratio,
            Vec3::new(0.0, 0.0, -5.0),
        )
        .expect("surface point should project");
        let hidden_point = project_world_to_image_with_depth(
            &camera_transform,
            width,
            height,
            fov_y,
            aspect_ratio,
            Vec3::new(0.0, 0.0, -6.0),
        )
        .expect("occluded point should still project");

        let pixel_x = front_point.pixel[0].round() as usize;
        let pixel_y = front_point.pixel[1].round() as usize;
        let depth = depth_buffer[pixel_y * width + pixel_x];
        assert!((front_point.depth - depth).abs() <= 1e-3);
        assert!(hidden_point.depth - depth > keypoint_depth_tolerance(depth));
    }

    #[test]
    fn keypoint_depth_tolerance_grows_with_distance() {
        assert!((keypoint_depth_tolerance(1.0) - KEYPOINT_DEPTH_EPSILON_MIN).abs() <= 1e-6);
        assert!((keypoint_depth_tolerance(100.0) - 0.5).abs() <= 1e-6);
        assert!((keypoint_depth_tolerance(1000.0) - 5.0).abs() <= 1e-6);
    }

    #[test]
    fn keypoint_neighborhood_match_accepts_one_pixel_segmentation_offset() {
        let width = 5;
        let height = 5;
        let mut segmentation = vec![0_u8; width * height];
        let mut depth = vec![f32::INFINITY; width * height];
        let target_index = 2 * width + 3;
        segmentation[target_index] = 1;
        depth[target_index] = 100.0;

        let matched = evaluate_keypoint_neighborhood_match(
            &segmentation,
            &depth,
            width,
            height,
            2,
            2,
            1,
            100.0,
        );

        assert!(matched.segmentation_match);
        assert!(matched.depth_match);
        assert!(matched.final_visibility);
    }

    #[test]
    fn keypoint_neighborhood_match_keeps_joint_visibility_strict() {
        let width = 5;
        let height = 5;
        let mut segmentation = vec![0_u8; width * height];
        let mut depth = vec![f32::INFINITY; width * height];
        segmentation[2 * width + 3] = 1;
        depth[1 * width + 2] = 100.0;

        let matched = evaluate_keypoint_neighborhood_match(
            &segmentation,
            &depth,
            width,
            height,
            2,
            2,
            1,
            100.0,
        );

        assert!(matched.segmentation_match);
        assert!(matched.depth_match);
        assert!(!matched.final_visibility);
    }

    #[test]
    fn binaural_energy_tracks_left_right_sum_and_balance() {
        let left = [1.0_f32, 1.0, 1.0, 1.0];
        let right = [0.0_f32, 0.0, 0.0, 0.0];

        let energy = compute_binaural_energy(&left, &right);

        assert!((energy[0] - 1.0).abs() <= 1e-6);
        assert!((energy[1] - 0.0).abs() <= 1e-6);
        assert!((energy[2] - 1.0).abs() <= 1e-6);
        assert!((energy[3] - 1.0).abs() <= 1e-6);
    }

    #[test]
    fn binaural_cues_zero_out_for_silence() {
        let left = [0.0_f32; 16];
        let right = [0.0_f32; 16];

        let cues = compute_binaural_cues(&left, &right, 48_000, 0.0);

        assert!(cues.into_iter().all(|value| value == 0.0));
    }

    #[test]
    fn binaural_cues_capture_clear_leading_channel_difference() {
        let left = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let right = [0.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let energy = compute_binaural_energy(&left, &right);
        let cues = compute_binaural_cues(&left, &right, 48_000, energy[2]);

        assert!(
            cues[0].abs() > 0.0,
            "gcc lag should reflect interaural offset"
        );
        assert!(
            cues[1] > 0.0,
            "gcc peak should be positive for a clear dominant lag"
        );
        assert!(
            cues[5] >= 0.0 && cues[5] <= 1.0,
            "coherence should stay normalized"
        );
        assert!(
            cues[7] >= 0.0 && cues[7] <= 1.0,
            "directness proxy should stay normalized"
        );
    }
}
