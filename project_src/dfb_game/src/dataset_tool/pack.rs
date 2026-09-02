use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use bevy::math::{Mat3, Quat, Vec3};
use half::f16;
use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_npy::WriteNpyExt;
use serde::{Deserialize, Serialize};
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

use crate::api::types::{AircraftObservation, PixelFormat, StateObservation, VisualSensorKind};
use crate::core::config::{ConfigPaths, RepositoryConfig};
use crate::dataset_tool::label::{DerivedLabelsManifest, SparseVotingArtifactRef};
use crate::recording::reconstruct::RecordingAccess;
use crate::recording::{DerivedEpisodeManifest, VisualArtifactRef};

const MAX_STEPS_PER_CHUNK: usize = 32;
const MAX_CHUNK_TARGET_BYTES: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone)]
struct CliArgs {
    episode_paths: Vec<PathBuf>,
    recordings_root: PathBuf,
    output_dir: Option<PathBuf>,
    observed_roles: Vec<String>,
    force: bool,
    profile: bool,
}

#[derive(Debug, Default)]
struct PackProfile {
    schema_bootstrap: Duration,
    episode_and_label_load: Duration,
    per_step_row_build: Duration,
    chunk_group_write_core: Duration,
    chunk_group_write_vision: Duration,
    chunk_group_write_audio: Duration,
    chunk_group_write_rule: Duration,
    meta_serialize: Duration,
    row_count: u64,
}

#[derive(Debug, Default)]
struct ArtifactBundleCache {
    bytes_by_relative_path: HashMap<PathBuf, Vec<u8>>,
}

impl ArtifactBundleCache {
    fn read_slice<'a>(
        &'a mut self,
        derived_root: &Path,
        relative_path: &str,
        byte_offset: Option<u64>,
        byte_length: Option<u64>,
        artifact_kind: &str,
    ) -> Result<&'a [u8]> {
        let relative_path = PathBuf::from(relative_path);
        if !self.bytes_by_relative_path.contains_key(&relative_path) {
            let derived_path = derived_root.join(&relative_path);
            let bytes = fs::read(&derived_path)
                .with_context(|| format!("failed to read {}", derived_path.display()))?;
            self.bytes_by_relative_path
                .insert(relative_path.clone(), bytes);
        }
        let bundle = self
            .bytes_by_relative_path
            .get(&relative_path)
            .ok_or_else(|| anyhow!("missing artifact bundle cache entry"))?;
        let Some(offset) = byte_offset else {
            return Ok(bundle.as_slice());
        };
        let length = byte_length.with_context(|| {
            format!("{artifact_kind} artifact has byte_offset but missing byte_length")
        })?;
        let start = offset as usize;
        let end = start
            .checked_add(length as usize)
            .with_context(|| format!("{artifact_kind} artifact byte range overflowed usize"))?;
        bundle.get(start..end).with_context(|| {
            format!(
                "{artifact_kind} artifact byte range [{start}..{end}) is out of bounds for {} bytes",
                bundle.len()
            )
        })
    }
}

#[derive(Debug, Deserialize)]
struct KeypointSchema {
    schema_id: String,
    point_labels: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DatasetChunkEntry {
    chunk_id: String,
    episode_id: String,
    observed_role: String,
    chunk_index: u32,
    step_count: u32,
    simulation_step_index_start: u32,
    simulation_step_index_end_exclusive: u32,
    group_files: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
struct DatasetEpisodeEntry {
    episode_id: String,
    scene_name: String,
    source_episode_root: String,
    observed_roles: Vec<String>,
    total_steps: u32,
}

#[derive(Debug, Serialize)]
struct DatasetStatistics {
    total_simulation_steps: u64,
    total_model_steps: u64,
}

#[derive(Debug, Serialize)]
struct DatasetStorageLayout {
    format: String,
    groups: Vec<String>,
    runtime_only_groups: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DatasetVisualConfig {
    camera_ids: Vec<String>,
    resolution: DatasetResolution,
    segmentation_classes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DatasetResolution {
    width: u32,
    height: u32,
}

#[derive(Debug, Serialize)]
struct DatasetAudioConfig {
    layout: String,
    channel_order: Vec<String>,
    window_samples: usize,
    sample_rate_hz: u32,
    energy_feature_order: Vec<String>,
    cue_feature_order: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DatasetMeta {
    dataset_id: String,
    dataset_version: String,
    schema_id: String,
    schema_version: String,
    schema_path: String,
    asset_ids_path: String,
    coordinate_convention_id: String,
    keypoint_schema_id: String,
    keypoint_schema_path: String,
    audio_cue_schema_id: String,
    audio_cue_schema_path: String,
    time_semantics: serde_json::Value,
    storage_layout: DatasetStorageLayout,
    visual_config: DatasetVisualConfig,
    audio_config: DatasetAudioConfig,
    splits: BTreeMap<String, Vec<String>>,
    episodes: Vec<DatasetEpisodeEntry>,
    chunks: Vec<DatasetChunkEntry>,
    statistics: DatasetStatistics,
}

#[derive(Debug)]
struct PackedStepRow {
    simulation_step_index: i32,
    timestamp: f64,
    ego_position_world: [f32; 3],
    ego_orientation_world_6d: [f32; 6],
    ego_linear_velocity_world: [f32; 3],
    ego_angular_velocity_body: [f32; 3],
    camera_extrinsics_front: [f32; 9],
    camera_extrinsics_rear: [f32; 9],
    gt_relative_position: [f32; 3],
    gt_doa_unit_vector_body: [f32; 3],
    gt_log_distance_scalar: f32,
    gt_relative_orientation_6d: [f32; 6],
    gt_linear_velocity: [f32; 3],
    gt_angular_velocity: [f32; 3],
    binaural_energy_t: [f32; 4],
    binaural_cue_vector_t: [f32; 10],
    target_pos_conf: f32,
    target_ori_conf: f32,
    segmentation_mask_front: Vec<u8>,
    segmentation_mask_rear: Vec<u8>,
    keypoints_2d_front: Vec<f32>,
    keypoints_2d_rear: Vec<f32>,
    keypoint_visibility_front: Vec<u8>,
    keypoint_visibility_rear: Vec<u8>,
    keypoint_projectable_front: Vec<u8>,
    keypoint_projectable_rear: Vec<u8>,
    keypoint_voting_pixels_front: Vec<u16>,
    keypoint_voting_pixels_rear: Vec<u16>,
    keypoint_voting_unit_vectors_front: Vec<f32>,
    keypoint_voting_unit_vectors_rear: Vec<f32>,
    keypoint_voting_mask_front: Vec<u8>,
    keypoint_voting_mask_rear: Vec<u8>,
    keypoint_voting_front_pixel_count: usize,
    keypoint_voting_rear_pixel_count: usize,
    front_camera_image: Vec<u8>,
    rear_camera_image: Vec<u8>,
    audio_window_binaural: Vec<f32>,
}

#[derive(Debug)]
struct DecodedSparseVotingStep {
    pixel_count: usize,
    pixels: Vec<u16>,
    unit_vectors: Vec<f32>,
    valid_mask: Vec<u8>,
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
    energy_features: AudioFeatureField,
    cue_features: AudioFeatureField,
}

pub fn run_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_args(args)?;
    let config_paths = ConfigPaths::default();
    let project_root = config_paths.project_root.clone();
    let mut profile = PackProfile::default();

    let schema_bootstrap_started_at = Instant::now();
    let schema_source = project_root.join("config/dfb_state_estimation/schema.json");
    let schema_json: serde_json::Value = serde_json::from_slice(
        &fs::read(&schema_source)
            .with_context(|| format!("failed to read {}", schema_source.display()))?,
    )
    .with_context(|| format!("failed to parse {}", schema_source.display()))?;
    let schema_id = schema_json["schema_id"]
        .as_str()
        .ok_or_else(|| anyhow!("schema.json missing schema_id"))?
        .to_string();
    let schema_version = schema_json["schema_version"]
        .as_str()
        .ok_or_else(|| anyhow!("schema.json missing schema_version"))?
        .to_string();
    let coordinate_convention_id = schema_json["conventions"]["coordinate_convention_id"]
        .as_str()
        .ok_or_else(|| anyhow!("schema.json missing coordinate_convention_id"))?
        .to_string();
    let time_semantics = schema_json["conventions"]["time_semantics"].clone();

    let keypoint_schema_path = project_root.join(
        schema_json["asset_refs"]["default_keypoint_schema_path"]
            .as_str()
            .ok_or_else(|| anyhow!("schema.json missing default_keypoint_schema_path"))?,
    );
    let keypoint_schema: KeypointSchema = serde_json::from_slice(
        &fs::read(&keypoint_schema_path)
            .with_context(|| format!("failed to read {}", keypoint_schema_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", keypoint_schema_path.display()))?;
    let keypoint_count = keypoint_schema.point_labels.len();

    let audio_cue_schema_rel = schema_json["asset_refs"]["default_audio_cue_schema_path"]
        .as_str()
        .ok_or_else(|| anyhow!("schema.json missing default_audio_cue_schema_path"))?
        .to_string();
    let audio_cue_schema: AudioCueSchema = serde_json::from_slice(
        &fs::read(project_root.join(&audio_cue_schema_rel)).with_context(|| {
            format!(
                "failed to read {}",
                project_root.join(&audio_cue_schema_rel).display()
            )
        })?,
    )?;
    let asset_ids_path = schema_json["asset_refs"]["ids_manifest_path"]
        .as_str()
        .ok_or_else(|| anyhow!("schema.json missing ids_manifest_path"))?
        .to_string();
    profile.schema_bootstrap += schema_bootstrap_started_at.elapsed();

    let episode_paths = if args.episode_paths.is_empty() {
        find_episode_roots(&args.recordings_root)?
    } else {
        args.episode_paths
    };
    if episode_paths.is_empty() {
        bail!("no recorded episodes found");
    }

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let dataset_id = format!("dfb_state_estimation_dataset_{created_at}");
    let output_root = args.output_dir.unwrap_or_else(|| {
        project_root
            .join("datasets/dfb_state_estimation")
            .join(&dataset_id)
    });

    if output_root.exists() {
        if !args.force {
            bail!(
                "output dir already exists at {}; rerun with --force",
                output_root.display()
            );
        }
        fs::remove_dir_all(&output_root)
            .with_context(|| format!("failed to clear {}", output_root.display()))?;
    }

    for group in ["core", "vision_labels", "audio_features", "rule_targets"] {
        fs::create_dir_all(output_root.join(group))?;
    }

    fs::copy(&schema_source, output_root.join("schema.json")).with_context(|| {
        format!(
            "failed to copy schema {} -> {}",
            schema_source.display(),
            output_root.join("schema.json").display()
        )
    })?;

    let mut chunk_entries = Vec::new();
    let mut episode_entries = Vec::new();
    let mut total_simulation_steps = 0_u64;
    let mut visual_width = 0_u32;
    let mut visual_height = 0_u32;
    let mut audio_window_samples = 0_usize;
    let mut audio_sample_rate = 0_u32;
    let mut chunk_index = 0_u32;

    for episode_root in episode_paths {
        let access = RecordingAccess::new(&episode_root);
        let manifest = access.manifest()?;
        let repository_config =
            RepositoryConfig::load_from_root_with_scene(&project_root, Some(&manifest.scene_name))?;
        let observed_roles = if args.observed_roles.is_empty() {
            access.available_derived_roles()?
        } else {
            args.observed_roles.clone()
        };
        if observed_roles.is_empty() {
            bail!(
                "episode {} has no derived roles; run `dfb_tool_dataset extract` and `label` first",
                manifest.episode_id
            );
        }

        let mut exported_episode_steps = 0_u32;

        for observed_role in &observed_roles {
            let load_started_at = Instant::now();
            let derived_manifest = access
                .derived_manifest(&observed_role)
                .with_context(|| format!("missing derived manifest for role {observed_role}"))?;
            let labels_path = access
                .derived_root_for_role(&observed_role)
                .join("derived_labels.ron");
            let labels_manifest: DerivedLabelsManifest = ron::from_str(
                &fs::read_to_string(&labels_path)
                    .with_context(|| format!("failed to read {}", labels_path.display()))?,
            )
            .with_context(|| format!("failed to parse {}", labels_path.display()))?;
            profile.episode_and_label_load += load_started_at.elapsed();

            let packed_chunks = pack_episode_role(
                &output_root,
                &mut chunk_index,
                &access,
                &manifest,
                &derived_manifest,
                &labels_manifest,
                &repository_config,
                &observed_role,
                keypoint_count,
                &mut profile,
            )?;

            let role_step_count: u32 = packed_chunks.iter().map(|packed| packed.step_count).sum();
            exported_episode_steps = exported_episode_steps.max(role_step_count);
            for packed in packed_chunks {
                visual_width = visual_width.max(packed.visual_width);
                visual_height = visual_height.max(packed.visual_height);
                audio_window_samples = audio_window_samples.max(packed.audio_window_samples);
                audio_sample_rate = audio_sample_rate.max(packed.audio_sample_rate);
                total_simulation_steps += packed.step_count as u64;
                chunk_entries.push(DatasetChunkEntry {
                    chunk_id: format!("chunk_{:06}", packed.chunk_index),
                    episode_id: manifest.episode_id.clone(),
                    observed_role: observed_role.clone(),
                    chunk_index: packed.chunk_index,
                    step_count: packed.step_count,
                    simulation_step_index_start: packed.simulation_step_index_start,
                    simulation_step_index_end_exclusive: packed.simulation_step_index_end_exclusive,
                    group_files: packed.group_files,
                });
            }
        }

        episode_entries.push(DatasetEpisodeEntry {
            episode_id: manifest.episode_id.clone(),
            scene_name: manifest.scene_name.clone(),
            source_episode_root: episode_root.display().to_string(),
            observed_roles,
            total_steps: exported_episode_steps,
        });
    }

    let meta = DatasetMeta {
        dataset_id,
        dataset_version: "1.0.0".to_string(),
        schema_id,
        schema_version,
        schema_path: "schema.json".to_string(),
        asset_ids_path,
        coordinate_convention_id,
        keypoint_schema_id: keypoint_schema.schema_id,
        keypoint_schema_path: keypoint_schema_path.display().to_string(),
        audio_cue_schema_id: audio_cue_schema.schema_id.clone(),
        audio_cue_schema_path: audio_cue_schema_rel,
        time_semantics,
        storage_layout: DatasetStorageLayout {
            format: "chunked_npz".to_string(),
            groups: vec![
                "core".to_string(),
                "vision_labels".to_string(),
                "audio_features".to_string(),
                "rule_targets".to_string(),
            ],
            runtime_only_groups: vec!["runtime_only".to_string()],
        },
        visual_config: DatasetVisualConfig {
            camera_ids: vec!["front".to_string(), "rear".to_string()],
            resolution: DatasetResolution {
                width: visual_width,
                height: visual_height,
            },
            segmentation_classes: vec![
                "background".to_string(),
                "fighter1".to_string(),
                "fighter2".to_string(),
            ],
        },
        audio_config: DatasetAudioConfig {
            layout: audio_cue_schema.layout,
            channel_order: audio_cue_schema.channel_order,
            window_samples: audio_window_samples,
            sample_rate_hz: audio_sample_rate,
            energy_feature_order: audio_cue_schema.energy_features.order,
            cue_feature_order: audio_cue_schema.cue_features.order,
        },
        splits: BTreeMap::from([
            ("train".to_string(), Vec::new()),
            ("val".to_string(), Vec::new()),
            ("test".to_string(), Vec::new()),
        ]),
        episodes: episode_entries,
        chunks: chunk_entries,
        statistics: DatasetStatistics {
            total_simulation_steps,
            total_model_steps: 0,
        },
    };

    let meta_started_at = Instant::now();
    fs::write(
        output_root.join("meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    profile.meta_serialize += meta_started_at.elapsed();
    println!("wrote {}", output_root.join("meta.json").display());
    if args.profile {
        print_pack_profile_report(&profile);
    }
    Ok(())
}

struct PackedChunkResult {
    chunk_index: u32,
    group_files: BTreeMap<String, String>,
    step_count: u32,
    simulation_step_index_start: u32,
    simulation_step_index_end_exclusive: u32,
    visual_width: u32,
    visual_height: u32,
    audio_window_samples: usize,
    audio_sample_rate: u32,
}

struct LargeNpzWriter<W: std::io::Write + std::io::Seek> {
    zip: zip::ZipWriter<W>,
    options: SimpleFileOptions,
}

impl<W: std::io::Write + std::io::Seek> LargeNpzWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            zip: zip::ZipWriter::new(writer).set_auto_large_file(),
            options: SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored)
                .large_file(true),
        }
    }

    fn add_array<N, S, D>(&mut self, name: N, array: &ndarray::ArrayBase<S, D>) -> Result<()>
    where
        N: Into<String>,
        S: ndarray::Data,
        S::Elem: ndarray_npy::WritableElement,
        D: ndarray::Dimension,
    {
        self.zip.start_file(name.into() + ".npy", self.options)?;
        array.write_npy(std::io::BufWriter::new(&mut self.zip))?;
        Ok(())
    }

    fn finish(self) -> Result<W> {
        let mut writer = self.zip.finish()?;
        std::io::Write::flush(&mut writer)?;
        Ok(writer)
    }
}

fn pack_episode_role(
    output_root: &Path,
    next_chunk_index: &mut u32,
    access: &RecordingAccess,
    manifest: &crate::recording::RecordedEpisodeManifest,
    derived_manifest: &DerivedEpisodeManifest,
    labels_manifest: &DerivedLabelsManifest,
    repository_config: &RepositoryConfig,
    observed_role: &str,
    keypoint_count: usize,
    profile: &mut PackProfile,
) -> Result<Vec<PackedChunkResult>> {
    let episode_load_started_at = Instant::now();
    let episode = access.load_episode()?;
    profile.episode_and_label_load += episode_load_started_at.elapsed();
    anyhow::ensure!(
        derived_manifest.steps.len() == labels_manifest.steps.len(),
        "step count mismatch for derived labels in episode {} role {}: derived={} labels={}",
        manifest.episode_id,
        observed_role,
        derived_manifest.steps.len(),
        labels_manifest.steps.len()
    );
    anyhow::ensure!(
        derived_manifest.steps.len() <= episode.steps.len(),
        "derived step count exceeds source episode length for episode {} role {}: episode={} derived={}",
        manifest.episode_id,
        observed_role,
        episode.steps.len(),
        derived_manifest.steps.len()
    );

    let visual_cfg_by_kind = derived_manifest
        .capture_config
        .visual_sensors
        .iter()
        .map(|cfg| (cfg.kind, cfg))
        .collect::<HashMap<_, _>>();
    let front_cfg = visual_cfg_by_kind.get(&VisualSensorKind::Front);
    let rear_cfg = visual_cfg_by_kind.get(&VisualSensorKind::Rear);
    let visual_width = front_cfg
        .map(|cfg| cfg.width)
        .or_else(|| rear_cfg.map(|cfg| cfg.width))
        .unwrap_or(0);
    let visual_height = front_cfg
        .map(|cfg| cfg.height)
        .or_else(|| rear_cfg.map(|cfg| cfg.height))
        .unwrap_or(0);
    let audio_sample_rate = derived_manifest
        .audio_artifact_metadata
        .as_ref()
        .map(|meta| meta.sample_rate)
        .unwrap_or(0);
    let audio_window_samples = derived_manifest
        .audio_artifact_metadata
        .as_ref()
        .map(|meta| (meta.sample_rate as f32 * meta.window_seconds).round() as usize)
        .unwrap_or(0);
    let derived_root = access.derived_root_for_role(observed_role);
    let mut bundle_cache = ArtifactBundleCache::default();

    let exported_steps = &episode.steps[..derived_manifest.steps.len()];
    let mut rows = Vec::with_capacity(exported_steps.len());
    let row_build_started_at = Instant::now();
    for ((step, artifacts), labels) in exported_steps
        .iter()
        .zip(&derived_manifest.steps)
        .zip(&labels_manifest.steps)
    {
        anyhow::ensure!(
            step.index == artifacts.index && step.index == labels.index,
            "step alignment mismatch in episode {} role {} at step {}",
            manifest.episode_id,
            observed_role,
            step.index
        );
        rows.push(pack_step_row(
            &derived_root,
            repository_config,
            &step.state,
            artifacts,
            labels,
            observed_role,
            keypoint_count,
            visual_width,
            visual_height,
            audio_window_samples,
            audio_sample_rate,
            &mut bundle_cache,
        )?);
    }
    profile.per_step_row_build += row_build_started_at.elapsed();
    profile.row_count += rows.len() as u64;

    let mut packed_chunks = Vec::new();
    let mut chunk_start = 0_usize;
    let mut chunk_len = 0_usize;
    let mut max_front_voting_pixels = 0_usize;
    let mut max_rear_voting_pixels = 0_usize;

    let mut flush_chunk =
        |start: usize, len: usize, max_front_pixels: usize, max_rear_pixels: usize| -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            let rows_chunk = &rows[start..start + len];
            let chunk_index = *next_chunk_index;
            let group_files = write_chunk_group_files(
                output_root,
                chunk_index,
                rows_chunk,
                keypoint_count,
                visual_width as usize,
                visual_height as usize,
                audio_window_samples,
                profile,
            )?;
            let simulation_step_index_start = rows_chunk
                .first()
                .and_then(|row| u32::try_from(row.simulation_step_index).ok())
                .unwrap_or(0);
            let simulation_step_index_end_exclusive = rows_chunk
                .last()
                .and_then(|row| row.simulation_step_index.checked_add(1))
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(simulation_step_index_start);
            packed_chunks.push(PackedChunkResult {
                chunk_index,
                group_files,
                step_count: rows_chunk.len() as u32,
                simulation_step_index_start,
                simulation_step_index_end_exclusive,
                visual_width,
                visual_height,
                audio_window_samples,
                audio_sample_rate,
            });
            *next_chunk_index += 1;
            let _ = (max_front_pixels, max_rear_pixels);
            Ok(())
        };

    for (row_index, row) in rows.iter().enumerate() {
        let projected_len = chunk_len + 1;
        let projected_max_front =
            max_front_voting_pixels.max(row.keypoint_voting_front_pixel_count);
        let projected_max_rear = max_rear_voting_pixels.max(row.keypoint_voting_rear_pixel_count);
        let projected_bytes = estimate_chunk_bytes(
            projected_len,
            projected_max_front,
            projected_max_rear,
            keypoint_count,
            visual_width as usize,
            visual_height as usize,
            audio_window_samples,
        );
        if chunk_len > 0
            && (projected_len > MAX_STEPS_PER_CHUNK || projected_bytes > MAX_CHUNK_TARGET_BYTES)
        {
            flush_chunk(
                chunk_start,
                chunk_len,
                max_front_voting_pixels,
                max_rear_voting_pixels,
            )?;
            chunk_start = row_index;
            chunk_len = 1;
            max_front_voting_pixels = row.keypoint_voting_front_pixel_count;
            max_rear_voting_pixels = row.keypoint_voting_rear_pixel_count;
            continue;
        }
        chunk_len = projected_len;
        max_front_voting_pixels = projected_max_front;
        max_rear_voting_pixels = projected_max_rear;
    }

    flush_chunk(
        chunk_start,
        chunk_len,
        max_front_voting_pixels,
        max_rear_voting_pixels,
    )?;

    Ok(packed_chunks)
}

fn estimate_chunk_bytes(
    step_count: usize,
    max_front_voting_pixels: usize,
    max_rear_voting_pixels: usize,
    keypoint_count: usize,
    visual_width: usize,
    visual_height: usize,
    audio_window_samples: usize,
) -> usize {
    let pixel_count = visual_width * visual_height;
    let front_image_bytes = pixel_count * 4;
    let rear_image_bytes = pixel_count * 4;
    let audio_window_bytes = audio_window_samples * 2 * std::mem::size_of::<f32>();
    let core_per_step_bytes = std::mem::size_of::<i32>()
        + std::mem::size_of::<f64>()
        + front_image_bytes
        + rear_image_bytes
        + audio_window_bytes
        + (3 + 6 + 3 + 3 + 9 + 9 + 3 + 6 + 3 + 3) * std::mem::size_of::<f32>();
    let vision_fixed_per_step_bytes = pixel_count * 2
        + keypoint_count * 2 * 2 * std::mem::size_of::<f32>()
        + keypoint_count * 4 * std::mem::size_of::<u8>();
    let front_sparse_per_step_bytes = max_front_voting_pixels
        * (2 * std::mem::size_of::<u16>()
            + keypoint_count * 2 * std::mem::size_of::<f32>()
            + std::mem::size_of::<u8>());
    let rear_sparse_per_step_bytes = max_rear_voting_pixels
        * (2 * std::mem::size_of::<u16>()
            + keypoint_count * 2 * std::mem::size_of::<f32>()
            + std::mem::size_of::<u8>());
    let audio_feature_per_step_bytes = (4 + 10) * std::mem::size_of::<f32>();
    let rule_per_step_bytes = (3 + 1 + 1 + 1) * std::mem::size_of::<f32>();
    step_count
        * (core_per_step_bytes
            + vision_fixed_per_step_bytes
            + front_sparse_per_step_bytes
            + rear_sparse_per_step_bytes
            + audio_feature_per_step_bytes
            + rule_per_step_bytes)
}

#[allow(clippy::too_many_arguments)]
fn pack_step_row(
    derived_root: &Path,
    repository_config: &RepositoryConfig,
    state: &StateObservation,
    artifacts: &crate::recording::DerivedStepArtifacts,
    labels: &crate::dataset_tool::label::DerivedStepLabels,
    observed_role: &str,
    keypoint_count: usize,
    visual_width: u32,
    visual_height: u32,
    audio_window_samples: usize,
    audio_sample_rate: u32,
    bundle_cache: &mut ArtifactBundleCache,
) -> Result<PackedStepRow> {
    let observer = find_aircraft(state, observed_role)?;
    let observer_quat = quat(observer.orientation_quat).normalize();
    let front_image = decode_visual_by_camera(
        derived_root,
        &artifacts.visual,
        VisualSensorKind::Front,
        visual_width,
        visual_height,
        bundle_cache,
    )?;
    let rear_image = decode_visual_by_camera(
        derived_root,
        &artifacts.visual,
        VisualSensorKind::Rear,
        visual_width,
        visual_height,
        bundle_cache,
    )?;
    let audio_window = decode_audio_window(
        derived_root,
        artifacts.audio.as_ref(),
        audio_window_samples,
        audio_sample_rate,
        bundle_cache,
    )?;
    let front_camera_extrinsics = camera_extrinsics_6d(
        Vec3::from_array(repository_config.game.camera.follow_offset),
        true,
    );
    let rear_camera_extrinsics = camera_extrinsics_6d(
        Vec3::from_array(repository_config.game.camera.rear_view_offset),
        false,
    );
    let segmentation_mask_front = decode_segmentation_by_camera(
        derived_root,
        &artifacts.segmentation,
        VisualSensorKind::Front,
        visual_width,
        visual_height,
        bundle_cache,
    )?;
    let segmentation_mask_rear = decode_segmentation_by_camera(
        derived_root,
        &artifacts.segmentation,
        VisualSensorKind::Rear,
        visual_width,
        visual_height,
        bundle_cache,
    )?;
    let keypoints_2d_front = flatten_keypoints_2d(&labels.keypoints_2d_front, keypoint_count)?;
    let keypoints_2d_rear = flatten_keypoints_2d(&labels.keypoints_2d_rear, keypoint_count)?;
    let keypoint_visibility_front =
        flatten_keypoint_visibility(&labels.keypoint_visibility_front, keypoint_count)?;
    let keypoint_visibility_rear =
        flatten_keypoint_visibility(&labels.keypoint_visibility_rear, keypoint_count)?;
    let keypoint_projectable_front =
        flatten_keypoint_visibility(&labels.keypoint_projectable_front, keypoint_count)?;
    let keypoint_projectable_rear =
        flatten_keypoint_visibility(&labels.keypoint_projectable_rear, keypoint_count)?;
    let keypoint_voting_front = decode_sparse_voting_artifact(
        derived_root,
        &labels.keypoint_voting_front,
        keypoint_count,
        bundle_cache,
    )?;
    let keypoint_voting_rear = decode_sparse_voting_artifact(
        derived_root,
        &labels.keypoint_voting_rear,
        keypoint_count,
        bundle_cache,
    )?;

    Ok(PackedStepRow {
        simulation_step_index: labels.index as i32,
        timestamp: labels.sim_time_seconds as f64,
        ego_position_world: observer.position,
        ego_orientation_world_6d: rotation6d_from_quat(observer_quat),
        ego_linear_velocity_world: observer.linear_velocity,
        ego_angular_velocity_body: degrees_to_radians_vec3(observer.angular_velocity_deg)
            .to_array(),
        camera_extrinsics_front: front_camera_extrinsics,
        camera_extrinsics_rear: rear_camera_extrinsics,
        gt_relative_position: labels.gt_relative_position_body,
        gt_doa_unit_vector_body: labels.gt_doa_unit_vector_body,
        gt_log_distance_scalar: labels.gt_log_distance_scalar,
        gt_relative_orientation_6d: rotation6d_from_quat(quat(labels.gt_relative_orientation_quat)),
        gt_linear_velocity: labels.gt_relative_linear_velocity_body,
        gt_angular_velocity: labels.gt_relative_angular_velocity_body,
        binaural_energy_t: labels.binaural_energy_t,
        binaural_cue_vector_t: labels.binaural_cue_vector_t,
        target_pos_conf: labels.target_pos_conf,
        target_ori_conf: labels.target_ori_conf,
        segmentation_mask_front,
        segmentation_mask_rear,
        keypoints_2d_front,
        keypoints_2d_rear,
        keypoint_visibility_front,
        keypoint_visibility_rear,
        keypoint_projectable_front,
        keypoint_projectable_rear,
        keypoint_voting_pixels_front: keypoint_voting_front.pixels,
        keypoint_voting_pixels_rear: keypoint_voting_rear.pixels,
        keypoint_voting_unit_vectors_front: keypoint_voting_front.unit_vectors,
        keypoint_voting_unit_vectors_rear: keypoint_voting_rear.unit_vectors,
        keypoint_voting_mask_front: keypoint_voting_front.valid_mask,
        keypoint_voting_mask_rear: keypoint_voting_rear.valid_mask,
        keypoint_voting_front_pixel_count: keypoint_voting_front.pixel_count,
        keypoint_voting_rear_pixel_count: keypoint_voting_rear.pixel_count,
        front_camera_image: front_image,
        rear_camera_image: rear_image,
        audio_window_binaural: audio_window,
    })
}

fn write_chunk_group_files(
    output_root: &Path,
    chunk_index: u32,
    rows: &[PackedStepRow],
    keypoint_count: usize,
    visual_width: usize,
    visual_height: usize,
    audio_window_samples: usize,
    profile: &mut PackProfile,
) -> Result<BTreeMap<String, String>> {
    let n = rows.len();
    let chunk_name = format!("chunk_{chunk_index:06}.npz");
    let mut group_files = BTreeMap::new();

    let mut simulation_step_index = Vec::with_capacity(n);
    let mut timestamp = Vec::with_capacity(n);
    let mut front_camera_image = Vec::with_capacity(n * visual_width * visual_height * 4);
    let mut rear_camera_image = Vec::with_capacity(n * visual_width * visual_height * 4);
    let mut audio_window = Vec::with_capacity(n * audio_window_samples * 2);
    let mut ego_position_world = Vec::with_capacity(n * 3);
    let mut ego_orientation_world = Vec::with_capacity(n * 6);
    let mut ego_linear_velocity_world = Vec::with_capacity(n * 3);
    let mut ego_angular_velocity_body = Vec::with_capacity(n * 3);
    let mut camera_extrinsics_front = Vec::with_capacity(n * 9);
    let mut camera_extrinsics_rear = Vec::with_capacity(n * 9);
    let mut gt_relative_position = Vec::with_capacity(n * 3);
    let mut gt_doa_unit_vector_body = Vec::with_capacity(n * 3);
    let mut gt_log_distance_scalar = Vec::with_capacity(n);
    let mut gt_relative_orientation = Vec::with_capacity(n * 6);
    let mut gt_linear_velocity = Vec::with_capacity(n * 3);
    let mut gt_angular_velocity = Vec::with_capacity(n * 3);
    let mut binaural_energy_t = Vec::with_capacity(n * 4);
    let mut binaural_cue_vector_t = Vec::with_capacity(n * 8);
    let mut target_pos_conf = Vec::with_capacity(n);
    let mut target_ori_conf = Vec::with_capacity(n);
    let mut segmentation_mask_front = Vec::with_capacity(n * visual_height * visual_width);
    let mut segmentation_mask_rear = Vec::with_capacity(n * visual_height * visual_width);
    let mut keypoints_2d_front = Vec::with_capacity(n * keypoint_count * 2);
    let mut keypoints_2d_rear = Vec::with_capacity(n * keypoint_count * 2);
    let mut keypoint_visibility_front = Vec::with_capacity(n * keypoint_count);
    let mut keypoint_visibility_rear = Vec::with_capacity(n * keypoint_count);
    let mut keypoint_projectable_front = Vec::with_capacity(n * keypoint_count);
    let mut keypoint_projectable_rear = Vec::with_capacity(n * keypoint_count);
    let max_front_voting_pixels = rows
        .iter()
        .map(|row| row.keypoint_voting_front_pixel_count)
        .max()
        .unwrap_or(0);
    let max_rear_voting_pixels = rows
        .iter()
        .map(|row| row.keypoint_voting_rear_pixel_count)
        .max()
        .unwrap_or(0);
    let mut keypoint_voting_pixels_front = Vec::with_capacity(n * max_front_voting_pixels * 2);
    let mut keypoint_voting_pixels_rear = Vec::with_capacity(n * max_rear_voting_pixels * 2);
    let mut keypoint_voting_unit_vectors_front =
        Vec::with_capacity(n * max_front_voting_pixels * keypoint_count * 2);
    let mut keypoint_voting_unit_vectors_rear =
        Vec::with_capacity(n * max_rear_voting_pixels * keypoint_count * 2);
    let mut keypoint_voting_mask_front = Vec::with_capacity(n * max_front_voting_pixels);
    let mut keypoint_voting_mask_rear = Vec::with_capacity(n * max_rear_voting_pixels);

    for row in rows {
        simulation_step_index.push(row.simulation_step_index);
        timestamp.push(row.timestamp);
        front_camera_image.extend_from_slice(&row.front_camera_image);
        rear_camera_image.extend_from_slice(&row.rear_camera_image);
        audio_window.extend_from_slice(&row.audio_window_binaural);
        ego_position_world.extend_from_slice(&row.ego_position_world);
        ego_orientation_world.extend_from_slice(&row.ego_orientation_world_6d);
        ego_linear_velocity_world.extend_from_slice(&row.ego_linear_velocity_world);
        ego_angular_velocity_body.extend_from_slice(&row.ego_angular_velocity_body);
        camera_extrinsics_front.extend_from_slice(&row.camera_extrinsics_front);
        camera_extrinsics_rear.extend_from_slice(&row.camera_extrinsics_rear);
        gt_relative_position.extend_from_slice(&row.gt_relative_position);
        gt_doa_unit_vector_body.extend_from_slice(&row.gt_doa_unit_vector_body);
        gt_log_distance_scalar.push(row.gt_log_distance_scalar);
        gt_relative_orientation.extend_from_slice(&row.gt_relative_orientation_6d);
        gt_linear_velocity.extend_from_slice(&row.gt_linear_velocity);
        gt_angular_velocity.extend_from_slice(&row.gt_angular_velocity);
        binaural_energy_t.extend_from_slice(&row.binaural_energy_t);
        binaural_cue_vector_t.extend_from_slice(&row.binaural_cue_vector_t);
        target_pos_conf.push(row.target_pos_conf);
        target_ori_conf.push(row.target_ori_conf);
        segmentation_mask_front.extend_from_slice(&row.segmentation_mask_front);
        segmentation_mask_rear.extend_from_slice(&row.segmentation_mask_rear);
        keypoints_2d_front.extend_from_slice(&row.keypoints_2d_front);
        keypoints_2d_rear.extend_from_slice(&row.keypoints_2d_rear);
        keypoint_visibility_front.extend_from_slice(&row.keypoint_visibility_front);
        keypoint_visibility_rear.extend_from_slice(&row.keypoint_visibility_rear);
        keypoint_projectable_front.extend_from_slice(&row.keypoint_projectable_front);
        keypoint_projectable_rear.extend_from_slice(&row.keypoint_projectable_rear);
        append_padded_sparse_voting_row(
            &mut keypoint_voting_pixels_front,
            &mut keypoint_voting_unit_vectors_front,
            &mut keypoint_voting_mask_front,
            &row.keypoint_voting_pixels_front,
            &row.keypoint_voting_unit_vectors_front,
            &row.keypoint_voting_mask_front,
            row.keypoint_voting_front_pixel_count,
            max_front_voting_pixels,
            keypoint_count,
        );
        append_padded_sparse_voting_row(
            &mut keypoint_voting_pixels_rear,
            &mut keypoint_voting_unit_vectors_rear,
            &mut keypoint_voting_mask_rear,
            &row.keypoint_voting_pixels_rear,
            &row.keypoint_voting_unit_vectors_rear,
            &row.keypoint_voting_mask_rear,
            row.keypoint_voting_rear_pixel_count,
            max_rear_voting_pixels,
            keypoint_count,
        );
    }

    let core_path = output_root.join("core").join(&chunk_name);
    let core_started_at = Instant::now();
    let mut core = open_npz_writer(&core_path)?;
    core.add_array(
        "simulation_step_index",
        &Array1::from_vec(simulation_step_index),
    )?;
    core.add_array("timestamp", &Array1::from_vec(timestamp))?;
    core.add_array(
        "front_camera_image",
        &Array4::from_shape_vec((n, visual_height, visual_width, 4), front_camera_image)?,
    )?;
    core.add_array(
        "rear_camera_image",
        &Array4::from_shape_vec((n, visual_height, visual_width, 4), rear_camera_image)?,
    )?;
    core.add_array(
        "audio_window_binaural",
        &Array3::from_shape_vec((n, audio_window_samples, 2), audio_window)?,
    )?;
    core.add_array(
        "ego_position_world",
        &Array2::from_shape_vec((n, 3), ego_position_world)?,
    )?;
    core.add_array(
        "ego_orientation_world",
        &Array2::from_shape_vec((n, 6), ego_orientation_world)?,
    )?;
    core.add_array(
        "ego_linear_velocity_world",
        &Array2::from_shape_vec((n, 3), ego_linear_velocity_world)?,
    )?;
    core.add_array(
        "ego_angular_velocity_body",
        &Array2::from_shape_vec((n, 3), ego_angular_velocity_body)?,
    )?;
    core.add_array(
        "camera_extrinsics_front",
        &Array2::from_shape_vec((n, 9), camera_extrinsics_front)?,
    )?;
    core.add_array(
        "camera_extrinsics_rear",
        &Array2::from_shape_vec((n, 9), camera_extrinsics_rear)?,
    )?;
    core.add_array(
        "gt_relative_position",
        &Array2::from_shape_vec((n, 3), gt_relative_position)?,
    )?;
    core.add_array(
        "gt_relative_orientation",
        &Array2::from_shape_vec((n, 6), gt_relative_orientation)?,
    )?;
    core.add_array(
        "gt_linear_velocity",
        &Array2::from_shape_vec((n, 3), gt_linear_velocity)?,
    )?;
    core.add_array(
        "gt_angular_velocity",
        &Array2::from_shape_vec((n, 3), gt_angular_velocity)?,
    )?;
    core.finish()?;
    profile.chunk_group_write_core += core_started_at.elapsed();
    group_files.insert("core".to_string(), format!("core/{chunk_name}"));

    let vision_path = output_root.join("vision_labels").join(&chunk_name);
    let vision_started_at = Instant::now();
    let mut vision = open_npz_writer(&vision_path)?;
    vision.add_array(
        "segmentation_mask_front",
        &Array3::from_shape_vec((n, visual_height, visual_width), segmentation_mask_front)?,
    )?;
    vision.add_array(
        "segmentation_mask_rear",
        &Array3::from_shape_vec((n, visual_height, visual_width), segmentation_mask_rear)?,
    )?;
    vision.add_array(
        "keypoints_2d_front",
        &Array3::from_shape_vec((n, keypoint_count, 2), keypoints_2d_front)?,
    )?;
    vision.add_array(
        "keypoints_2d_rear",
        &Array3::from_shape_vec((n, keypoint_count, 2), keypoints_2d_rear)?,
    )?;
    vision.add_array(
        "keypoint_visibility_front",
        &Array2::from_shape_vec((n, keypoint_count), keypoint_visibility_front)?,
    )?;
    vision.add_array(
        "keypoint_visibility_rear",
        &Array2::from_shape_vec((n, keypoint_count), keypoint_visibility_rear)?,
    )?;
    vision.add_array(
        "keypoint_projectable_front",
        &Array2::from_shape_vec((n, keypoint_count), keypoint_projectable_front)?,
    )?;
    vision.add_array(
        "keypoint_projectable_rear",
        &Array2::from_shape_vec((n, keypoint_count), keypoint_projectable_rear)?,
    )?;
    vision.add_array(
        "keypoint_voting_pixels_front",
        &Array3::from_shape_vec(
            (n, max_front_voting_pixels, 2),
            keypoint_voting_pixels_front,
        )?,
    )?;
    vision.add_array(
        "keypoint_voting_pixels_rear",
        &Array3::from_shape_vec((n, max_rear_voting_pixels, 2), keypoint_voting_pixels_rear)?,
    )?;
    vision.add_array(
        "keypoint_voting_unit_vectors_front",
        &Array4::from_shape_vec(
            (n, max_front_voting_pixels, keypoint_count, 2),
            keypoint_voting_unit_vectors_front,
        )?,
    )?;
    vision.add_array(
        "keypoint_voting_unit_vectors_rear",
        &Array4::from_shape_vec(
            (n, max_rear_voting_pixels, keypoint_count, 2),
            keypoint_voting_unit_vectors_rear,
        )?,
    )?;
    vision.add_array(
        "keypoint_voting_mask_front",
        &Array2::from_shape_vec((n, max_front_voting_pixels), keypoint_voting_mask_front)?,
    )?;
    vision.add_array(
        "keypoint_voting_mask_rear",
        &Array2::from_shape_vec((n, max_rear_voting_pixels), keypoint_voting_mask_rear)?,
    )?;
    vision.finish()?;
    profile.chunk_group_write_vision += vision_started_at.elapsed();
    group_files.insert(
        "vision_labels".to_string(),
        format!("vision_labels/{chunk_name}"),
    );

    let audio_path = output_root.join("audio_features").join(&chunk_name);
    let audio_started_at = Instant::now();
    let mut audio = open_npz_writer(&audio_path)?;
    audio.add_array(
        "binaural_energy_t",
        &Array2::from_shape_vec((n, 4), binaural_energy_t)?,
    )?;
    audio.add_array(
        "binaural_cue_vector_t",
        &Array2::from_shape_vec((n, 10), binaural_cue_vector_t)?,
    )?;
    audio.finish()?;
    profile.chunk_group_write_audio += audio_started_at.elapsed();
    group_files.insert(
        "audio_features".to_string(),
        format!("audio_features/{chunk_name}"),
    );

    let rule_path = output_root.join("rule_targets").join(&chunk_name);
    let rule_started_at = Instant::now();
    let mut rules = open_npz_writer(&rule_path)?;
    rules.add_array(
        "gt_doa_unit_vector_body",
        &Array2::from_shape_vec((n, 3), gt_doa_unit_vector_body)?,
    )?;
    rules.add_array(
        "gt_log_distance_scalar",
        &Array1::from_vec(gt_log_distance_scalar),
    )?;
    rules.add_array("target_pos_conf", &Array1::from_vec(target_pos_conf))?;
    rules.add_array("target_ori_conf", &Array1::from_vec(target_ori_conf))?;
    rules.finish()?;
    profile.chunk_group_write_rule += rule_started_at.elapsed();
    group_files.insert(
        "rule_targets".to_string(),
        format!("rule_targets/{chunk_name}"),
    );

    Ok(group_files)
}

fn open_npz_writer(path: &Path) -> Result<LargeNpzWriter<File>> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(LargeNpzWriter::new(file))
}

fn decode_visual_by_camera(
    derived_root: &Path,
    artifacts: &[VisualArtifactRef],
    camera: VisualSensorKind,
    fallback_width: u32,
    fallback_height: u32,
    bundle_cache: &mut ArtifactBundleCache,
) -> Result<Vec<u8>> {
    let Some(artifact) = artifacts.iter().find(|artifact| artifact.camera == camera) else {
        return Ok(vec![
            0;
            fallback_width as usize * fallback_height as usize * 4
        ]);
    };
    let width = artifact.width.unwrap_or(fallback_width) as usize;
    let height = artifact.height.unwrap_or(fallback_height) as usize;
    let Some(format) = artifact.format else {
        return Ok(vec![0; width * height * 4]);
    };
    let Some(relative_path) = &artifact.file_path else {
        return Ok(vec![0; width * height * 4]);
    };
    let bytes = bundle_cache.read_slice(
        derived_root,
        relative_path,
        artifact.byte_offset,
        artifact.byte_length,
        "visual",
    )?;
    match format {
        PixelFormat::Rgb8 => decode_rgb8_to_rgba(bytes, width, height),
        PixelFormat::Rgba8 => {
            let expected = width * height * 4;
            if bytes.len() != expected {
                bail!(
                    "unexpected RGBA artifact byte length: got {}, expected {}",
                    bytes.len(),
                    expected
                );
            }
            Ok(bytes.to_vec())
        }
        PixelFormat::Gray8 => {
            let payload = if bytes.len() == width * height {
                bytes.to_vec()
            } else {
                decode_pgm_gray8_payload(bytes, width, height)?
            };
            let mut rgba = Vec::with_capacity(width * height * 4);
            for gray in payload {
                rgba.extend_from_slice(&[gray, gray, gray, 255]);
            }
            Ok(rgba)
        }
    }
}

fn decode_segmentation_by_camera(
    derived_root: &Path,
    artifacts: &[VisualArtifactRef],
    camera: VisualSensorKind,
    fallback_width: u32,
    fallback_height: u32,
    bundle_cache: &mut ArtifactBundleCache,
) -> Result<Vec<u8>> {
    let Some(artifact) = artifacts.iter().find(|artifact| artifact.camera == camera) else {
        bail!("missing {:?} segmentation artifact", camera);
    };
    let width = artifact.width.unwrap_or(fallback_width) as usize;
    let height = artifact.height.unwrap_or(fallback_height) as usize;
    let Some(relative_path) = &artifact.file_path else {
        bail!("missing {:?} segmentation artifact file path", camera);
    };
    let bytes = bundle_cache.read_slice(
        derived_root,
        relative_path,
        artifact.byte_offset,
        artifact.byte_length,
        "visual",
    )?;
    match artifact.format.unwrap_or(PixelFormat::Gray8) {
        PixelFormat::Gray8 => decode_gray8_to_segmentation(bytes, width, height),
        PixelFormat::Rgb8 => decode_rgb8_to_segmentation(bytes, width, height),
        PixelFormat::Rgba8 => {
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

fn decode_rgb8_to_rgba(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let expected_rgb = width * height * 3;
    if bytes.len() == expected_rgb {
        let mut rgba = Vec::with_capacity(width * height * 4);
        for chunk in bytes.chunks_exact(3) {
            rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
        }
        return Ok(rgba);
    }
    decode_ppm_rgb8_to_rgba(bytes, width, height)
}

fn decode_ppm_rgb8_to_rgba(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let payload = decode_ppm_rgb8_payload(bytes, width, height)?;
    let mut rgba = Vec::with_capacity(width * height * 4);
    for chunk in payload.chunks_exact(3) {
        rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
    }
    Ok(rgba)
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
    let expected_rgb = width * height * 3;
    anyhow::ensure!(
        bytes.len().saturating_sub(index) == expected_rgb,
        "PPM payload length mismatch: got {}, expected {}",
        bytes.len().saturating_sub(index),
        expected_rgb
    );
    Ok(bytes[index..].to_vec())
}

fn decode_audio_window(
    derived_root: &Path,
    artifact: Option<&crate::recording::AudioArtifactRef>,
    expected_frames: usize,
    expected_sample_rate: u32,
    bundle_cache: &mut ArtifactBundleCache,
) -> Result<Vec<f32>> {
    let expected_samples = expected_frames * 2;
    let Some(artifact) = artifact else {
        return Ok(vec![0.0; expected_samples]);
    };
    let Some(relative_path) = &artifact.file_path else {
        return Ok(vec![0.0; expected_samples]);
    };
    let bytes = bundle_cache.read_slice(
        derived_root,
        relative_path,
        artifact.byte_offset,
        artifact.byte_length,
        "audio",
    )?;
    if bytes.len() < 44 {
        return Ok(vec![0.0; expected_samples]);
    }
    anyhow::ensure!(
        &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "invalid WAV header"
    );
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    anyhow::ensure!(
        expected_sample_rate == 0 || sample_rate == expected_sample_rate,
        "audio sample rate mismatch: got {}, expected {}",
        sample_rate,
        expected_sample_rate
    );
    anyhow::ensure!(channels > 0, "invalid WAV channel count");
    let data = &bytes[44..];
    let pcm_samples = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
        .collect::<Vec<_>>();
    let frames = pcm_samples.len() / channels;
    let copy_frames = expected_frames.min(frames);
    let mut out = vec![0.0_f32; expected_samples];
    for frame in 0..copy_frames {
        for channel in 0..2 {
            let source = if channel < channels {
                pcm_samples[frame * channels + channel]
            } else {
                0.0
            };
            out[frame * 2 + channel] = source;
        }
    }
    Ok(out)
}

fn camera_extrinsics_6d(position_body: Vec3, front_camera: bool) -> [f32; 9] {
    let look_direction_body = if front_camera { Vec3::Z } else { Vec3::NEG_Z };
    let orientation_body = Quat::from_rotation_arc(Vec3::NEG_Z, look_direction_body).normalize();
    let orientation_6d = rotation6d_from_quat(orientation_body);
    [
        position_body.x,
        position_body.y,
        position_body.z,
        orientation_6d[0],
        orientation_6d[1],
        orientation_6d[2],
        orientation_6d[3],
        orientation_6d[4],
        orientation_6d[5],
    ]
}

fn flatten_keypoints_2d(keypoints: &[[f32; 2]], expected_count: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(
        keypoints.len() == expected_count,
        "keypoint count mismatch: got {}, expected {}",
        keypoints.len(),
        expected_count
    );
    let mut out = Vec::with_capacity(expected_count * 2);
    for point in keypoints {
        out.extend_from_slice(point);
    }
    Ok(out)
}

fn flatten_keypoint_visibility(visibility: &[u8], expected_count: usize) -> Result<Vec<u8>> {
    anyhow::ensure!(
        visibility.len() == expected_count,
        "keypoint visibility length mismatch: got {}, expected {}",
        visibility.len(),
        expected_count
    );
    Ok(visibility.to_vec())
}

fn decode_sparse_voting_artifact(
    derived_root: &Path,
    artifact: &SparseVotingArtifactRef,
    expected_keypoint_count: usize,
    bundle_cache: &mut ArtifactBundleCache,
) -> Result<DecodedSparseVotingStep> {
    anyhow::ensure!(
        artifact.coord_dtype == "u16",
        "unsupported sparse voting coord dtype {}",
        artifact.coord_dtype
    );
    anyhow::ensure!(
        artifact.vector_dtype == "float16",
        "unsupported sparse voting vector dtype {}",
        artifact.vector_dtype
    );
    anyhow::ensure!(
        artifact.keypoint_count as usize == expected_keypoint_count,
        "sparse voting keypoint count mismatch: got {}, expected {}",
        artifact.keypoint_count,
        expected_keypoint_count
    );
    let bytes = bundle_cache.read_slice(
        derived_root,
        &artifact.file_path,
        Some(artifact.byte_offset),
        Some(artifact.byte_length),
        "sparse voting",
    )?;
    let pixel_count = artifact.pixel_count as usize;
    let bytes_per_pixel =
        2 * std::mem::size_of::<u16>() + expected_keypoint_count * 2 * std::mem::size_of::<u16>();
    anyhow::ensure!(
        bytes.len() == pixel_count * bytes_per_pixel,
        "sparse voting artifact byte length mismatch: got {}, expected {}",
        bytes.len(),
        pixel_count * bytes_per_pixel
    );
    let mut pixels = Vec::with_capacity(pixel_count * 2);
    let mut unit_vectors = Vec::with_capacity(pixel_count * expected_keypoint_count * 2);
    let mut valid_mask = Vec::with_capacity(pixel_count);
    let mut cursor = 0usize;
    for _ in 0..pixel_count {
        let x = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        cursor += 2;
        let y = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
        cursor += 2;
        pixels.extend_from_slice(&[x, y]);
        for _ in 0..expected_keypoint_count {
            let ux = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            cursor += 2;
            let uy = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            cursor += 2;
            unit_vectors.push(f16::from_bits(ux).to_f32());
            unit_vectors.push(f16::from_bits(uy).to_f32());
        }
        valid_mask.push(1);
    }
    Ok(DecodedSparseVotingStep {
        pixel_count,
        pixels,
        unit_vectors,
        valid_mask,
    })
}

fn append_padded_sparse_voting_row(
    pixels_out: &mut Vec<u16>,
    vectors_out: &mut Vec<f32>,
    mask_out: &mut Vec<u8>,
    row_pixels: &[u16],
    row_vectors: &[f32],
    row_mask: &[u8],
    row_pixel_count: usize,
    padded_pixel_count: usize,
    keypoint_count: usize,
) {
    pixels_out.extend_from_slice(row_pixels);
    vectors_out.extend_from_slice(row_vectors);
    mask_out.extend_from_slice(row_mask);
    let pad_pixels = padded_pixel_count.saturating_sub(row_pixel_count);
    pixels_out.extend(std::iter::repeat_n(0_u16, pad_pixels * 2));
    vectors_out.extend(std::iter::repeat_n(
        0.0_f32,
        pad_pixels * keypoint_count * 2,
    ));
    mask_out.extend(std::iter::repeat_n(0_u8, pad_pixels));
}

fn rotation6d_from_quat(quat: Quat) -> [f32; 6] {
    let mat = Mat3::from_quat(quat.normalize());
    let x = mat.x_axis;
    let y = mat.y_axis;
    [x.x, x.y, x.z, y.x, y.y, y.z]
}

fn degrees_to_radians_vec3(value: [f32; 3]) -> Vec3 {
    Vec3::new(
        value[0].to_radians(),
        value[1].to_radians(),
        value[2].to_radians(),
    )
}

fn quat(value: [f32; 4]) -> Quat {
    Quat::from_xyzw(value[0], value[1], value[2], value[3])
}

fn find_aircraft<'a>(state: &'a StateObservation, role: &str) -> Result<&'a AircraftObservation> {
    state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role.eq_ignore_ascii_case(role))
        .ok_or_else(|| anyhow!("missing aircraft role {role}"))
}

fn find_episode_roots(recordings_root: &Path) -> Result<Vec<PathBuf>> {
    if !recordings_root.exists() {
        return Ok(Vec::new());
    }
    let mut episodes = fs::read_dir(recordings_root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("episode.ron").exists())
        .collect::<Vec<_>>();
    episodes.sort();
    Ok(episodes)
}

fn parse_args<I>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut episode_paths = Vec::new();
    let mut recordings_root = ConfigPaths::default().recordings_root();
    let mut output_dir = None;
    let mut observed_roles = Vec::new();
    let mut force = false;
    let mut profile = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                let Some(path) = args.next() else {
                    bail!("missing value for --episode");
                };
                episode_paths.push(PathBuf::from(path));
            }
            "--recordings-root" => {
                let Some(path) = args.next() else {
                    bail!("missing value for --recordings-root");
                };
                recordings_root = PathBuf::from(path);
            }
            "--output-dir" => {
                let Some(path) = args.next() else {
                    bail!("missing value for --output-dir");
                };
                output_dir = Some(PathBuf::from(path));
            }
            "--observed-role" => {
                let Some(role) = args.next() else {
                    bail!("missing value for --observed-role");
                };
                observed_roles.push(role);
            }
            "--force" => force = true,
            "--profile" => profile = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(CliArgs {
        episode_paths,
        recordings_root,
        output_dir,
        observed_roles,
        force,
        profile,
    })
}

fn print_pack_profile_report(profile: &PackProfile) {
    let total = profile.schema_bootstrap
        + profile.episode_and_label_load
        + profile.per_step_row_build
        + profile.chunk_group_write_core
        + profile.chunk_group_write_vision
        + profile.chunk_group_write_audio
        + profile.chunk_group_write_rule
        + profile.meta_serialize;
    println!("pack profile:");
    print_profile_stage("  schema_bootstrap", profile.schema_bootstrap, total);
    print_profile_stage(
        "  episode_and_label_load",
        profile.episode_and_label_load,
        total,
    );
    print_profile_stage("  per_step_row_build", profile.per_step_row_build, total);
    print_profile_stage(
        "  chunk_group_write_core",
        profile.chunk_group_write_core,
        total,
    );
    print_profile_stage(
        "  chunk_group_write_vision",
        profile.chunk_group_write_vision,
        total,
    );
    print_profile_stage(
        "  chunk_group_write_audio",
        profile.chunk_group_write_audio,
        total,
    );
    print_profile_stage(
        "  chunk_group_write_rule",
        profile.chunk_group_write_rule,
        total,
    );
    print_profile_stage("  meta_serialize", profile.meta_serialize, total);
    if profile.row_count > 0 {
        println!("  row_count: {}", profile.row_count);
        println!(
            "  per_step_row_build_avg_ms: {:.3}",
            duration_ms(profile.per_step_row_build) / profile.row_count as f64
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
