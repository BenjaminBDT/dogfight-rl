use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail, ensure};
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

use crate::api::types::{AircraftObservation, StateObservation};
use crate::core::config::ConfigPaths;
use crate::dataset_tool::part3_policy::{
    ACTION_BIN_DIM, ACTION_CONT_DIM, ACTION_SCHEMA_ID, BINARY_OBS_INDICES, DATASET_SCHEMA_ID,
    NORMALIZER_SCHEMA_ID, OBS_DIM, OBSERVATION_SCHEMA_ID, POLICY_CONTRACT_ID,
    build_policy_observation, policy_contract_sha256,
};
use crate::recording::reconstruct::RecordingAccess;
use crate::recording::{RECORDING_SCHEMA_VERSION, RecordedEpisodeManifest, RecordedStep};

const MAX_STEPS_PER_CHUNK: usize = 256;
const HEALTH_STATE_DIM: usize = 6;
const OBS_STD_EPSILON: f32 = 1e-6;
const AUX_STORAGE_DIR: &str = "auxiliary";
const HEALTH_LABELS: [&str; HEALTH_STATE_DIM] = [
    "total",
    "left_wing",
    "right_wing",
    "pitch_tail",
    "yaw_tail",
    "engine",
];

#[derive(Debug, Clone)]
struct CliArgs {
    episode_paths: Vec<PathBuf>,
    recordings_root: PathBuf,
    output_dir: Option<PathBuf>,
    force: bool,
    profile: bool,
    seed: u64,
    train_ratio: f32,
    val_ratio: f32,
    split_strategy: SplitStrategy,
    fighter1_demonstration_source: String,
    fighter2_demonstration_source: String,
    fighter1_sample_weight: f32,
    fighter2_sample_weight: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetSplit {
    Train,
    Val,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitStrategy {
    SceneStratified,
    EpisodeRandom,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DemonstrationSettings<'a> {
    source: &'a str,
    sample_weight: f32,
}

impl DatasetSplit {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

impl SplitStrategy {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "scene_stratified" => Ok(Self::SceneStratified),
            "episode_random" => Ok(Self::EpisodeRandom),
            _ => bail!("unsupported Part 3 BC split strategy: {value}"),
        }
    }

    fn schema_id(self) -> &'static str {
        match self {
            Self::SceneStratified => "scene_stratified_episode_v1",
            Self::EpisodeRandom => "episode_random_v1",
        }
    }
}

#[derive(Debug)]
struct EpisodeWorkItem {
    root: PathBuf,
    manifest: RecordedEpisodeManifest,
    split: DatasetSplit,
}

#[derive(Debug, Clone)]
struct PackedRow {
    simulation_step_index: i32,
    timestamp: f64,
    obs: [f32; OBS_DIM],
    action_cont: [f32; ACTION_CONT_DIM],
    action_bin: [f32; ACTION_BIN_DIM],
    done: u8,
    did_hit: u8,
    got_hit: u8,
    did_fire: u8,
    self_out_of_bounds_seconds: f32,
    self_ceiling_recovery_seconds: f32,
    self_repair_elapsed_seconds: f32,
    self_destroyed: u8,
    enemy_destroyed: u8,
    self_health_state_norm: [f32; HEALTH_STATE_DIM],
    enemy_health_state_norm: [f32; HEALTH_STATE_DIM],
    self_gun_overheated: u8,
    self_gun_heat_norm: f32,
    winner_label: i8,
    target_distance: f32,
}

#[derive(Debug, Serialize)]
struct DatasetChunkEntry {
    chunk_id: String,
    episode_id: String,
    observed_role: String,
    demonstration_source: String,
    sample_weight: f32,
    split: String,
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
    total_steps: u32,
    authoritative_source: bool,
    winner: Option<String>,
    termination_reason: String,
    split: String,
}

#[derive(Debug, Serialize)]
struct ObsNormalizer {
    normalizer_schema_id: String,
    policy_contract_id: String,
    observation_schema_id: String,
    contract_sha256: String,
    obs_dim: usize,
    epsilon: f32,
    mean: Vec<f32>,
    std: Vec<f32>,
    train_row_count: u64,
    source_dataset_id: String,
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

    fn add_array<S, D>(&mut self, name: &str, array: &ndarray::ArrayBase<S, D>) -> Result<()>
    where
        S: ndarray::Data,
        D: ndarray::Dimension,
        S::Elem: ndarray_npy::WritableElement,
    {
        self.zip
            .start_file(format!("{name}.npy"), self.options)
            .with_context(|| format!("failed to start {name}.npy"))?;
        array
            .write_npy(std::io::BufWriter::new(&mut self.zip))
            .with_context(|| format!("failed to write {name}.npy"))?;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        let mut writer = self.zip.finish().context("failed to finish npz writer")?;
        std::io::Write::flush(&mut writer).context("failed to flush npz writer")
    }
}

pub fn run_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_args(args)?;
    ensure!(
        args.train_ratio > 0.0 && args.val_ratio >= 0.0 && args.train_ratio + args.val_ratio < 1.0,
        "train/val ratios must satisfy train>0, val>=0, train+val<1"
    );
    validate_demonstration_settings(&args)?;

    let episode_paths = if args.episode_paths.is_empty() {
        find_episode_roots(&args.recordings_root)?
    } else {
        args.episode_paths.clone()
    };
    ensure!(!episode_paths.is_empty(), "no recorded episodes found");

    let mut items = load_work_items(episode_paths)?;
    items.sort_by(|a, b| a.manifest.episode_id.cmp(&b.manifest.episode_id));
    let source_manifest_sha256 = hash_source_episode_manifests(&items)?;
    assign_splits(
        &mut items,
        args.seed,
        args.train_ratio,
        args.val_ratio,
        args.split_strategy,
    );

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let dataset_id = format!("dfb_part3_policy_dataset_{created_at}");
    let output_root = args.output_dir.clone().unwrap_or_else(|| {
        ConfigPaths::default()
            .project_root
            .join("datasets/dfb_reinforcement_learning")
            .join(&dataset_id)
    });
    prepare_output_root(&output_root, args.force)?;

    let contract_sha256 = policy_contract_sha256();
    let source_recording_schema_versions = unique_recording_schema_versions(&items);
    let source_recording_policy_contract_ids = unique_recording_policy_contract_ids(&items);
    let source_recording_action_schema_ids = unique_recording_action_schema_ids(&items);
    let mut chunks = Vec::new();
    let mut episodes = Vec::new();
    let mut split_episode_ids = BTreeMap::from([
        ("train".to_string(), Vec::new()),
        ("val".to_string(), Vec::new()),
        ("test".to_string(), Vec::new()),
    ]);
    let mut obs_sum = [0.0_f64; OBS_DIM];
    let mut obs_sumsq = [0.0_f64; OBS_DIM];
    let mut train_rows = 0_u64;
    let mut total_rows = 0_u64;
    let mut total_simulation_steps = 0_u64;
    let mut chunk_index = 0_u32;

    for item in items {
        split_episode_ids
            .get_mut(item.split.as_str())
            .expect("known split")
            .push(item.manifest.episode_id.clone());
        let access = RecordingAccess::new(&item.root);
        let initial = access.initial_snapshot()?;
        let steps = access.steps()?;
        total_simulation_steps += steps.len() as u64;
        let episode_start = initial.state.sim_time_seconds;

        for role in ["fighter1", "fighter2"] {
            let demonstration = demonstration_settings(&args, role)?;
            let mut rows = Vec::with_capacity(steps.len());
            for (offset, step) in steps.iter().enumerate() {
                let observation_state = if offset == 0 {
                    &initial.state
                } else {
                    &steps[offset - 1].state
                };
                let row = build_row(
                    &item.manifest,
                    observation_state,
                    &step.state,
                    step,
                    role,
                    offset + 1 == steps.len(),
                    episode_start,
                )?;
                if item.split == DatasetSplit::Train {
                    accumulate_observation(&row.obs, &mut obs_sum, &mut obs_sumsq);
                    train_rows += 1;
                }
                rows.push(row);
            }
            total_rows += rows.len() as u64;
            for chunk_rows in rows.chunks(MAX_STEPS_PER_CHUNK) {
                chunks.push(write_chunk(
                    &output_root,
                    chunk_index,
                    &item.manifest.episode_id,
                    role,
                    demonstration,
                    item.split,
                    chunk_rows,
                )?);
                chunk_index += 1;
            }
        }

        episodes.push(DatasetEpisodeEntry {
            episode_id: item.manifest.episode_id.clone(),
            scene_name: item.manifest.scene_name.clone(),
            source_episode_root: item.root.display().to_string(),
            total_steps: steps.len() as u32,
            authoritative_source: item.manifest.authoritative_source,
            winner: item.manifest.winner.clone(),
            termination_reason: item.manifest.termination_reason.clone(),
            split: item.split.as_str().to_string(),
        });
    }

    ensure!(train_rows > 0, "train split produced zero rows");
    let normalizer = build_normalizer(
        &dataset_id,
        &contract_sha256,
        train_rows,
        &obs_sum,
        &obs_sumsq,
    );
    let schema = build_schema_json(&contract_sha256);
    let meta = json!({
        "dataset_id": dataset_id,
        "dataset_version": "1.1.0",
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": contract_sha256,
        "source_recording_schema_versions": source_recording_schema_versions,
        "source_recording_policy_contract_ids": source_recording_policy_contract_ids,
        "source_recording_action_schema_ids": source_recording_action_schema_ids,
        "source_episode_manifest_sha256": source_manifest_sha256,
        "split_seed": args.seed,
        "split_strategy": args.split_strategy.schema_id(),
        "schema_version": "1.1.0",
        "schema_path": "schema.json",
        "obs_normalizer_path": "obs_normalizer.json",
        "storage_layout": {
            "format": "chunked_npz",
            "groups": ["policy_input", "action_targets", "aux"],
        },
        "splits": split_episode_ids,
        "episodes": episodes,
        "chunks": chunks,
        "statistics": {
            "total_simulation_steps": total_simulation_steps,
            "total_model_steps": total_rows,
            "obs_dim": OBS_DIM,
        },
        "constants": {
            "health_state_labels": HEALTH_LABELS,
            "winner_label_convention": {"1": "ego_win", "0": "draw_or_unknown", "-1": "ego_lose"},
            "demonstration_sources": {
                "fighter1": args.fighter1_demonstration_source,
                "fighter2": args.fighter2_demonstration_source,
            },
            "bc_sample_weights": {
                "fighter1": args.fighter1_sample_weight,
                "fighter2": args.fighter2_sample_weight,
            },
        },
    });

    write_json(&output_root.join("schema.json"), &schema)?;
    write_json(&output_root.join("obs_normalizer.json"), &normalizer)?;
    write_json(&output_root.join("meta.json"), &meta)?;
    println!("wrote {}", output_root.join("meta.json").display());
    if args.profile {
        println!("part3 policy dataset rows: {total_rows}");
    }
    Ok(())
}

fn load_work_items(episode_paths: Vec<PathBuf>) -> Result<Vec<EpisodeWorkItem>> {
    episode_paths
        .into_iter()
        .map(|root| {
            let manifest = RecordingAccess::new(&root).manifest()?;
            ensure_policy_action_schema(&manifest, &root)?;
            ensure!(
                manifest.authoritative_source,
                "recording {} is not authoritative",
                root.display()
            );
            Ok(EpisodeWorkItem {
                root,
                manifest,
                split: DatasetSplit::Train,
            })
        })
        .collect()
}

fn assign_splits(
    items: &mut [EpisodeWorkItem],
    seed: u64,
    train_ratio: f32,
    val_ratio: f32,
    strategy: SplitStrategy,
) {
    match strategy {
        SplitStrategy::SceneStratified => {
            assign_scene_stratified_splits(items, seed, train_ratio, val_ratio)
        }
        SplitStrategy::EpisodeRandom => {
            assign_episode_random_splits(items, seed, train_ratio, val_ratio)
        }
    }
}

fn assign_scene_stratified_splits(
    items: &mut [EpisodeWorkItem],
    seed: u64,
    train_ratio: f32,
    val_ratio: f32,
) {
    let mut indices_by_scene = BTreeMap::<String, Vec<usize>>::new();
    for (index, item) in items.iter().enumerate() {
        indices_by_scene
            .entry(item.manifest.scene_name.clone())
            .or_default()
            .push(index);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for indices in indices_by_scene.values_mut() {
        indices.shuffle(&mut rng);
        let (train_count, val_count, _) = split_counts(indices.len(), train_ratio, val_ratio);
        for (rank, index) in indices.iter().copied().enumerate() {
            items[index].split = if rank < train_count {
                DatasetSplit::Train
            } else if rank < train_count + val_count {
                DatasetSplit::Val
            } else {
                DatasetSplit::Test
            };
        }
    }
}

fn assign_episode_random_splits(
    items: &mut [EpisodeWorkItem],
    seed: u64,
    train_ratio: f32,
    val_ratio: f32,
) {
    let mut indices = (0..items.len()).collect::<Vec<_>>();
    indices.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let (train_count, val_count, _) = split_counts(items.len(), train_ratio, val_ratio);
    for (rank, index) in indices.into_iter().enumerate() {
        items[index].split = if rank < train_count {
            DatasetSplit::Train
        } else if rank < train_count + val_count {
            DatasetSplit::Val
        } else {
            DatasetSplit::Test
        };
    }
}

fn split_counts(item_count: usize, train_ratio: f32, val_ratio: f32) -> (usize, usize, usize) {
    if item_count == 0 {
        return (0, 0, 0);
    }

    let test_ratio = 1.0 - train_ratio - val_ratio;
    let reserve_val = usize::from(val_ratio > 0.0 && item_count >= 3);
    let reserve_test = usize::from(test_ratio > 0.0 && item_count >= 2 + reserve_val);
    let max_train = item_count.saturating_sub(reserve_val + reserve_test).max(1);
    let train_count = (((item_count as f32) * train_ratio).round() as usize).clamp(1, max_train);
    let remaining = item_count - train_count;
    let max_val = remaining.saturating_sub(reserve_test);
    let val_count =
        (((item_count as f32) * val_ratio).round() as usize).clamp(reserve_val, max_val);
    let test_count = item_count - train_count - val_count;
    (train_count, val_count, test_count)
}

fn prepare_output_root(output_root: &Path, force: bool) -> Result<()> {
    if output_root.exists() {
        ensure!(
            force,
            "output dir already exists at {}; rerun with --force",
            output_root.display()
        );
        fs::remove_dir_all(output_root)
            .with_context(|| format!("failed to clear {}", output_root.display()))?;
    }
    for group in ["policy_input", "action_targets", AUX_STORAGE_DIR] {
        fs::create_dir_all(output_root.join(group))?;
    }
    Ok(())
}

fn build_row(
    manifest: &RecordedEpisodeManifest,
    observation_state: &StateObservation,
    result_state: &StateObservation,
    step: &RecordedStep,
    role: &str,
    done: bool,
    episode_start: f32,
) -> Result<PackedRow> {
    let enemy_role = opposite_role(role)?;
    let ego = find_aircraft(observation_state, role)?;
    let enemy = find_aircraft(observation_state, enemy_role)?;
    let result_ego = find_aircraft(result_state, role)?;
    let result_enemy = find_aircraft(result_state, enemy_role)?;
    let action = command_for_role(step, role)?;
    let self_health_state_norm = subsystem_health_state_norm(ego)?;
    let enemy_health_state_norm = subsystem_health_state_norm(enemy)?;
    let relative = [
        enemy.position[0] - ego.position[0],
        enemy.position[1] - ego.position[1],
        enemy.position[2] - ego.position[2],
    ];

    Ok(PackedRow {
        simulation_step_index: step.index as i32,
        timestamp: observation_state.sim_time_seconds as f64,
        obs: build_policy_observation(observation_state, role, episode_start)?,
        action_cont: [action.throttle, action.pitch, action.roll, action.yaw],
        action_bin: [
            bool_f32(action.brake),
            bool_f32(action.fire_gun),
            bool_f32(action.repair),
        ],
        done: u8::from(done),
        did_hit: u8::from(event_matches(
            result_state,
            enemy_role,
            damage_event_kinds(),
        )),
        got_hit: u8::from(event_matches(result_state, role, damage_event_kinds())),
        did_fire: u8::from(event_matches(result_state, role, &["GunFired"])),
        self_out_of_bounds_seconds: ego.out_of_bounds_seconds,
        self_ceiling_recovery_seconds: ego.ceiling_recovery_seconds,
        self_repair_elapsed_seconds: ego.repair_elapsed_seconds,
        self_destroyed: u8::from(result_ego.destroyed),
        enemy_destroyed: u8::from(result_enemy.destroyed),
        self_health_state_norm,
        enemy_health_state_norm,
        self_gun_overheated: u8::from(ego.gun_overheated),
        self_gun_heat_norm: ego.gun_heat,
        winner_label: winner_label_for_role(role, manifest.winner.as_deref()),
        target_distance: relative
            .into_iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt(),
    })
}

fn write_chunk(
    output_root: &Path,
    chunk_index: u32,
    episode_id: &str,
    role: &str,
    demonstration: DemonstrationSettings<'_>,
    split: DatasetSplit,
    rows: &[PackedRow],
) -> Result<DatasetChunkEntry> {
    ensure!(!rows.is_empty(), "cannot write empty chunk");
    let chunk_name = format!("chunk_{chunk_index:06}.npz");
    let n = rows.len();

    let simulation_step_index = rows.iter().map(|row| row.simulation_step_index).collect();
    let timestamp = rows.iter().map(|row| row.timestamp).collect();
    let obs = rows.iter().flat_map(|row| row.obs).collect();
    let action_cont = rows.iter().flat_map(|row| row.action_cont).collect();
    let action_bin = rows.iter().flat_map(|row| row.action_bin).collect();

    let mut policy = open_npz_writer(&output_root.join("policy_input").join(&chunk_name))?;
    policy.add_array(
        "simulation_step_index",
        &Array1::from_vec(simulation_step_index),
    )?;
    policy.add_array("timestamp", &Array1::from_vec(timestamp))?;
    policy.add_array("obs", &Array2::from_shape_vec((n, OBS_DIM), obs)?)?;
    policy.finish()?;

    let mut action = open_npz_writer(&output_root.join("action_targets").join(&chunk_name))?;
    action.add_array(
        "action_cont",
        &Array2::from_shape_vec((n, ACTION_CONT_DIM), action_cont)?,
    )?;
    action.add_array(
        "action_bin",
        &Array2::from_shape_vec((n, ACTION_BIN_DIM), action_bin)?,
    )?;
    action.finish()?;

    write_auxiliary_chunk(output_root, &chunk_name, rows)?;
    let first = rows.first().expect("non-empty rows");
    let last = rows.last().expect("non-empty rows");
    Ok(DatasetChunkEntry {
        chunk_id: format!("chunk_{chunk_index:06}"),
        episode_id: episode_id.to_string(),
        observed_role: role.to_string(),
        demonstration_source: demonstration.source.to_string(),
        sample_weight: demonstration.sample_weight,
        split: split.as_str().to_string(),
        chunk_index,
        step_count: n as u32,
        simulation_step_index_start: first.simulation_step_index.max(0) as u32,
        simulation_step_index_end_exclusive: last.simulation_step_index.max(0) as u32 + 1,
        group_files: BTreeMap::from([
            (
                "policy_input".to_string(),
                format!("policy_input/{chunk_name}"),
            ),
            (
                "action_targets".to_string(),
                format!("action_targets/{chunk_name}"),
            ),
            ("aux".to_string(), format!("{AUX_STORAGE_DIR}/{chunk_name}")),
        ]),
    })
}

fn write_auxiliary_chunk(output_root: &Path, chunk_name: &str, rows: &[PackedRow]) -> Result<()> {
    let n = rows.len();
    let mut aux = open_npz_writer(&output_root.join(AUX_STORAGE_DIR).join(chunk_name))?;
    aux.add_array("done", &Array1::from_iter(rows.iter().map(|row| row.done)))?;
    aux.add_array(
        "did_hit",
        &Array1::from_iter(rows.iter().map(|row| row.did_hit)),
    )?;
    aux.add_array(
        "got_hit",
        &Array1::from_iter(rows.iter().map(|row| row.got_hit)),
    )?;
    aux.add_array(
        "did_fire",
        &Array1::from_iter(rows.iter().map(|row| row.did_fire)),
    )?;
    aux.add_array(
        "self_out_of_bounds_seconds",
        &Array1::from_iter(rows.iter().map(|row| row.self_out_of_bounds_seconds)),
    )?;
    aux.add_array(
        "self_ceiling_recovery_seconds",
        &Array1::from_iter(rows.iter().map(|row| row.self_ceiling_recovery_seconds)),
    )?;
    aux.add_array(
        "self_repair_elapsed_seconds",
        &Array1::from_iter(rows.iter().map(|row| row.self_repair_elapsed_seconds)),
    )?;
    aux.add_array(
        "self_destroyed",
        &Array1::from_iter(rows.iter().map(|row| row.self_destroyed)),
    )?;
    aux.add_array(
        "enemy_destroyed",
        &Array1::from_iter(rows.iter().map(|row| row.enemy_destroyed)),
    )?;
    aux.add_array(
        "self_health_state_norm",
        &Array2::from_shape_vec(
            (n, HEALTH_STATE_DIM),
            rows.iter()
                .flat_map(|row| row.self_health_state_norm)
                .collect(),
        )?,
    )?;
    aux.add_array(
        "enemy_health_state_norm",
        &Array2::from_shape_vec(
            (n, HEALTH_STATE_DIM),
            rows.iter()
                .flat_map(|row| row.enemy_health_state_norm)
                .collect(),
        )?,
    )?;
    aux.add_array(
        "self_gun_overheated",
        &Array1::from_iter(rows.iter().map(|row| row.self_gun_overheated)),
    )?;
    aux.add_array(
        "self_gun_heat_norm",
        &Array1::from_iter(rows.iter().map(|row| row.self_gun_heat_norm)),
    )?;
    aux.add_array(
        "winner_label",
        &Array1::from_iter(rows.iter().map(|row| row.winner_label as i32)),
    )?;
    aux.add_array(
        "target_distance",
        &Array1::from_iter(rows.iter().map(|row| row.target_distance)),
    )?;
    aux.finish()
}

fn build_normalizer(
    dataset_id: &str,
    contract_sha256: &str,
    train_rows: u64,
    sum: &[f64; OBS_DIM],
    sumsq: &[f64; OBS_DIM],
) -> ObsNormalizer {
    let mut mean = Vec::with_capacity(OBS_DIM);
    let mut std = Vec::with_capacity(OBS_DIM);
    for index in 0..OBS_DIM {
        if BINARY_OBS_INDICES.contains(&index) {
            mean.push(0.0);
            std.push(1.0);
            continue;
        }
        let value_mean = sum[index] / train_rows as f64;
        let variance = (sumsq[index] / train_rows as f64 - value_mean * value_mean).max(0.0);
        mean.push(value_mean as f32);
        std.push(variance.sqrt().max(OBS_STD_EPSILON as f64) as f32);
    }
    ObsNormalizer {
        normalizer_schema_id: NORMALIZER_SCHEMA_ID.to_string(),
        policy_contract_id: POLICY_CONTRACT_ID.to_string(),
        observation_schema_id: OBSERVATION_SCHEMA_ID.to_string(),
        contract_sha256: contract_sha256.to_string(),
        obs_dim: OBS_DIM,
        epsilon: OBS_STD_EPSILON,
        mean,
        std,
        train_row_count: train_rows,
        source_dataset_id: dataset_id.to_string(),
    }
}

fn build_schema_json(contract_sha256: &str) -> serde_json::Value {
    json!({
        "schema_version": "1.1.0",
        "dataset_schema_id": DATASET_SCHEMA_ID,
        "policy_contract_id": POLICY_CONTRACT_ID,
        "observation_schema_id": OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "normalizer_schema_id": NORMALIZER_SCHEMA_ID,
        "contract_sha256": contract_sha256,
        "description": "Part 3 policy-contract behavior-cloning dataset schema.",
        "conventions": {
            "coordinate_convention_id": "dfb_aircraft_body_rhs_forward_pos_z_v1",
            "orientation_representation": "rotation_6d_column_major",
            "transition_alignment": "observation_before_command_then_result_state",
        },
        "storage_schema": {
            "storage_unit": "chunked_npz",
            "chunk_step_level": "simulation_step",
            "chunk_metadata": {
                "demonstration_source": {"dtype": "string"},
                "sample_weight": {"dtype": "float32", "constraint": "finite_and_positive"},
            },
            "group_order": ["policy_input", "action_targets", "aux"],
            "policy_input": {
                "simulation_step_index": {"dtype": "int32", "shape": ["N"]},
                "timestamp": {"dtype": "float64", "shape": ["N"]},
                "obs": {"dtype": "float32", "shape": ["N", OBS_DIM]},
            },
            "action_targets": {
                "action_cont": {"dtype": "float32", "shape": ["N", ACTION_CONT_DIM]},
                "action_bin": {"dtype": "float32", "shape": ["N", ACTION_BIN_DIM]},
            },
        },
    })
}

fn accumulate_observation(
    obs: &[f32; OBS_DIM],
    sum: &mut [f64; OBS_DIM],
    sumsq: &mut [f64; OBS_DIM],
) {
    for (index, value) in obs.iter().enumerate() {
        let value = f64::from(*value);
        sum[index] += value;
        sumsq[index] += value * value;
    }
}

fn subsystem_health_state_norm(aircraft: &AircraftObservation) -> Result<[f32; HEALTH_STATE_DIM]> {
    let mut values = [0.0; HEALTH_STATE_DIM];
    values[0] = (aircraft.hit_points / 100.0).clamp(0.0, 1.0);
    for (offset, name) in ["LeftWing", "RightWing", "PitchTail", "YawTail", "Engine"]
        .into_iter()
        .enumerate()
    {
        let subsystem = aircraft
            .subsystems
            .iter()
            .find(|subsystem| subsystem.name == name)
            .ok_or_else(|| anyhow!("aircraft {} missing subsystem {name}", aircraft.role))?;
        ensure!(
            subsystem.max_hit_points > 0.0,
            "subsystem {name} has invalid max hit points"
        );
        values[offset + 1] = (subsystem.hit_points / subsystem.max_hit_points).clamp(0.0, 1.0);
    }
    Ok(values)
}

fn find_aircraft<'a>(state: &'a StateObservation, role: &str) -> Result<&'a AircraftObservation> {
    state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role.eq_ignore_ascii_case(role))
        .ok_or_else(|| anyhow!("missing aircraft role {role}"))
}

fn command_for_role(
    step: &RecordedStep,
    role: &str,
) -> Result<crate::api::types::EnvironmentAction> {
    match role {
        "fighter1" => Ok(step.fighter1_command),
        "fighter2" => Ok(step.fighter2_command),
        _ => bail!("unsupported role {role}"),
    }
}

fn opposite_role(role: &str) -> Result<&'static str> {
    match role {
        "fighter1" => Ok("fighter2"),
        "fighter2" => Ok("fighter1"),
        _ => bail!("unsupported role {role}"),
    }
}

fn damage_event_kinds() -> &'static [&'static str] {
    &[
        "Hit",
        "Damage",
        "SubsystemHit",
        "SubsystemDestroyed",
        "Destroy",
        "Kill",
    ]
}

fn event_matches(state: &StateObservation, subject_role: &str, kinds: &[&str]) -> bool {
    state.events_since_last_step.iter().any(|event| {
        event
            .subject
            .as_deref()
            .is_some_and(|subject| subject.eq_ignore_ascii_case(subject_role))
            && kinds
                .iter()
                .any(|kind| event.kind.eq_ignore_ascii_case(kind))
    })
}

fn winner_label_for_role(role: &str, winner: Option<&str>) -> i8 {
    match winner {
        Some(winner_role) if winner_role.eq_ignore_ascii_case(role) => 1,
        Some(_) => -1,
        None => 0,
    }
}

fn ensure_policy_action_schema(
    manifest: &RecordedEpisodeManifest,
    episode_root: &Path,
) -> Result<()> {
    ensure!(
        manifest.schema_version == RECORDING_SCHEMA_VERSION,
        "recording {} uses schema {}; expected {}",
        episode_root.display(),
        manifest.schema_version,
        RECORDING_SCHEMA_VERSION
    );
    ensure!(
        manifest.policy_contract_id == POLICY_CONTRACT_ID,
        "recording {} uses policy contract {}; expected {}",
        episode_root.display(),
        manifest.policy_contract_id,
        POLICY_CONTRACT_ID
    );
    ensure!(
        manifest.action_schema_id == ACTION_SCHEMA_ID,
        "recording {} uses action schema {}; expected {}",
        episode_root.display(),
        manifest.action_schema_id,
        ACTION_SCHEMA_ID
    );
    Ok(())
}

fn hash_source_episode_manifests(items: &[EpisodeWorkItem]) -> Result<String> {
    let mut hash = Sha256::new();
    for item in items {
        let bytes = fs::read(item.root.join("episode.ron"))?;
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(hex::encode(hash.finalize()))
}

fn unique_recording_schema_versions(items: &[EpisodeWorkItem]) -> Vec<u32> {
    items
        .iter()
        .map(|item| item.manifest.schema_version)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn unique_recording_policy_contract_ids(items: &[EpisodeWorkItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.manifest.policy_contract_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn unique_recording_action_schema_ids(items: &[EpisodeWorkItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.manifest.action_schema_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn open_npz_writer(path: &Path) -> Result<LargeNpzWriter<File>> {
    Ok(LargeNpzWriter::new(File::create(path).with_context(
        || format!("failed to create {}", path.display()),
    )?))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn find_episode_roots(recordings_root: &Path) -> Result<Vec<PathBuf>> {
    if !recordings_root.exists() {
        return Ok(Vec::new());
    }
    let mut roots = fs::read_dir(recordings_root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("episode.ron").exists())
        .collect::<Vec<_>>();
    roots.sort();
    Ok(roots)
}

fn bool_f32(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}

fn parse_args<I>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut parsed = CliArgs {
        episode_paths: Vec::new(),
        recordings_root: ConfigPaths::default().recordings_root(),
        output_dir: None,
        force: false,
        profile: false,
        seed: 42,
        train_ratio: 0.8,
        val_ratio: 0.1,
        split_strategy: SplitStrategy::SceneStratified,
        fighter1_demonstration_source: "human".to_string(),
        fighter2_demonstration_source: "non_human".to_string(),
        fighter1_sample_weight: 2.0,
        fighter2_sample_weight: 1.0,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => parsed
                .episode_paths
                .push(PathBuf::from(next_arg(&mut args, "--episode")?)),
            "--recordings-root" => {
                parsed.recordings_root = PathBuf::from(next_arg(&mut args, "--recordings-root")?)
            }
            "--output-dir" => {
                parsed.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?))
            }
            "--seed" => parsed.seed = next_arg(&mut args, "--seed")?.parse()?,
            "--train-ratio" => {
                parsed.train_ratio = next_arg(&mut args, "--train-ratio")?.parse()?
            }
            "--val-ratio" => parsed.val_ratio = next_arg(&mut args, "--val-ratio")?.parse()?,
            "--split-strategy" => {
                parsed.split_strategy =
                    SplitStrategy::parse(&next_arg(&mut args, "--split-strategy")?)?
            }
            "--fighter1-demonstration-source" => {
                parsed.fighter1_demonstration_source =
                    next_arg(&mut args, "--fighter1-demonstration-source")?
            }
            "--fighter2-demonstration-source" => {
                parsed.fighter2_demonstration_source =
                    next_arg(&mut args, "--fighter2-demonstration-source")?
            }
            "--fighter1-sample-weight" => {
                parsed.fighter1_sample_weight =
                    next_arg(&mut args, "--fighter1-sample-weight")?.parse()?
            }
            "--fighter2-sample-weight" => {
                parsed.fighter2_sample_weight =
                    next_arg(&mut args, "--fighter2-sample-weight")?.parse()?
            }
            "--force" => parsed.force = true,
            "--profile" => parsed.profile = true,
            other => bail!("unknown argument: {other}"),
        }
    }
    Ok(parsed)
}

fn validate_demonstration_settings(args: &CliArgs) -> Result<()> {
    for (role, source, weight) in [
        (
            "fighter1",
            args.fighter1_demonstration_source.as_str(),
            args.fighter1_sample_weight,
        ),
        (
            "fighter2",
            args.fighter2_demonstration_source.as_str(),
            args.fighter2_sample_weight,
        ),
    ] {
        ensure!(
            !source.trim().is_empty(),
            "{role} demonstration source must not be empty"
        );
        ensure!(
            weight.is_finite() && weight > 0.0,
            "{role} sample weight must be finite and positive"
        );
    }
    Ok(())
}

fn demonstration_settings<'a>(args: &'a CliArgs, role: &str) -> Result<DemonstrationSettings<'a>> {
    match role {
        "fighter1" => Ok(DemonstrationSettings {
            source: args.fighter1_demonstration_source.as_str(),
            sample_weight: args.fighter1_sample_weight,
        }),
        "fighter2" => Ok(DemonstrationSettings {
            source: args.fighter2_demonstration_source.as_str(),
            sample_weight: args.fighter2_sample_weight,
        }),
        _ => bail!("unsupported role {role}"),
    }
}

fn next_arg<I>(args: &mut I, flag: &str) -> Result<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| anyhow!("missing value for {flag}"))
}

#[cfg(test)]
mod tests {
    use super::{
        BINARY_OBS_INDICES, DemonstrationSettings, OBS_DIM, SplitStrategy, build_normalizer,
        demonstration_settings, parse_args, split_counts, validate_demonstration_settings,
        winner_label_for_role,
    };

    #[test]
    fn winner_label_is_ego_relative() {
        assert_eq!(winner_label_for_role("fighter1", Some("fighter1")), 1);
        assert_eq!(winner_label_for_role("fighter1", Some("fighter2")), -1);
        assert_eq!(winner_label_for_role("fighter1", None), 0);
    }

    #[test]
    fn binary_observation_normalizer_is_identity() {
        let normalizer = build_normalizer("dataset", "hash", 2, &[2.0; OBS_DIM], &[4.0; OBS_DIM]);
        for index in BINARY_OBS_INDICES {
            assert_eq!(normalizer.mean[index], 0.0);
            assert_eq!(normalizer.std[index], 1.0);
        }
    }

    #[test]
    fn demonstration_defaults_prioritize_human_fighter1() {
        let args = parse_args(Vec::<String>::new()).expect("default arguments must parse");
        validate_demonstration_settings(&args).expect("default settings must be valid");
        assert_eq!(
            demonstration_settings(&args, "fighter1").unwrap(),
            DemonstrationSettings {
                source: "human",
                sample_weight: 2.0,
            }
        );
        assert_eq!(
            demonstration_settings(&args, "fighter2").unwrap(),
            DemonstrationSettings {
                source: "non_human",
                sample_weight: 1.0,
            }
        );
    }

    #[test]
    fn split_counts_reserve_validation_and_test_episodes_per_scene() {
        assert_eq!(split_counts(14, 0.8, 0.1), (11, 1, 2));
        assert_eq!(split_counts(9, 0.8, 0.1), (7, 1, 1));
        assert_eq!(split_counts(6, 0.8, 0.1), (4, 1, 1));
        assert_eq!(split_counts(500, 0.8, 0.1), (400, 50, 50));
    }

    #[test]
    fn episode_random_split_strategy_is_explicit() {
        let args = parse_args(["--split-strategy".to_string(), "episode_random".to_string()])
            .expect("episode random split strategy must parse");

        assert_eq!(args.split_strategy, SplitStrategy::EpisodeRandom);
    }
}
