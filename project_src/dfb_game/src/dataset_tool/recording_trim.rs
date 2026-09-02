use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::recording::reconstruct::RecordingAccess;
use crate::recording::{RecordedEpisodeManifest, RecordedStepChunk, RecordedStepChunkIndexEntry};

#[derive(Debug)]
struct CliArgs {
    episode_root: PathBuf,
    end_step: u32,
    reason: String,
}

#[derive(Debug, Serialize)]
pub struct RecordingTrimResult {
    pub episode_root: String,
    pub backup_root: String,
    pub original_total_steps: u32,
    pub trimmed_total_steps: u32,
    pub end_step: u32,
    pub removed_steps: u32,
    pub termination_reason: String,
}

pub fn run_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_args(args)?;
    let result = trim_episode(&args.episode_root, args.end_step, &args.reason)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

pub fn trim_episode(
    episode_root: &Path,
    end_step: u32,
    termination_reason: &str,
) -> Result<RecordingTrimResult> {
    ensure!(
        !termination_reason.trim().is_empty(),
        "termination reason must not be empty"
    );

    let access = RecordingAccess::new(episode_root);
    let original_manifest = access.manifest()?;
    ensure!(original_manifest.total_steps > 0, "recording has no steps");
    ensure!(
        end_step + 1 < original_manifest.total_steps,
        "end step {end_step} must remove at least one step from recording with {} steps",
        original_manifest.total_steps
    );

    let boundary_entry = original_manifest
        .step_chunks
        .iter()
        .find(|entry| {
            end_step >= entry.start_step_index
                && end_step < entry.start_step_index + entry.step_count
        })
        .cloned()
        .with_context(|| format!("step {end_step} is not covered by the manifest chunk index"))?;
    let final_step = access.step(end_step)?;

    let boundary_path = episode_root.join(&boundary_entry.file_path);
    let mut boundary_chunk: RecordedStepChunk = ron::from_str(
        &fs::read_to_string(&boundary_path)
            .with_context(|| format!("failed to read {}", boundary_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", boundary_path.display()))?;
    boundary_chunk.steps.retain(|step| step.index <= end_step);
    ensure!(
        boundary_chunk
            .steps
            .last()
            .is_some_and(|step| step.index == end_step),
        "boundary chunk does not end at requested step {end_step}"
    );

    let backup_root = create_backup(
        episode_root,
        &original_manifest,
        boundary_entry.chunk_index,
        end_step,
    )?;

    let boundary_relative = PathBuf::from(format!(
        "steps/chunk_{:06}_trim_{end_step:09}.ron",
        boundary_entry.chunk_index
    ));
    let boundary_output = episode_root.join(&boundary_relative);
    write_ron_atomic(&boundary_output, &boundary_chunk)?;

    let mut trimmed_manifest = original_manifest.clone();
    trimmed_manifest.total_steps = end_step + 1;
    trimmed_manifest.final_tick = final_step.tick;
    trimmed_manifest.final_sim_time_seconds = final_step.sim_time_seconds;
    trimmed_manifest.termination_reason = termination_reason.to_string();
    trimmed_manifest.winner = None;
    trimmed_manifest
        .step_artifacts
        .retain(|artifacts| artifacts.index <= end_step);
    trimmed_manifest
        .step_chunks
        .retain(|entry| entry.chunk_index < boundary_entry.chunk_index);
    trimmed_manifest
        .step_chunks
        .push(RecordedStepChunkIndexEntry {
            chunk_index: boundary_entry.chunk_index,
            start_step_index: boundary_chunk.start_step_index,
            step_count: boundary_chunk.steps.len() as u32,
            file_path: boundary_relative.display().to_string(),
        });

    write_ron_atomic(&episode_root.join("episode.ron"), &trimmed_manifest)?;

    for entry in &original_manifest.step_chunks {
        if entry.chunk_index >= boundary_entry.chunk_index {
            let path = episode_root.join(&entry.file_path);
            if path != boundary_output && path.exists() {
                fs::remove_file(&path)
                    .with_context(|| format!("failed to remove {}", path.display()))?;
            }
        }
    }

    let result = RecordingTrimResult {
        episode_root: episode_root.display().to_string(),
        backup_root: backup_root.display().to_string(),
        original_total_steps: original_manifest.total_steps,
        trimmed_total_steps: trimmed_manifest.total_steps,
        end_step,
        removed_steps: original_manifest.total_steps - trimmed_manifest.total_steps,
        termination_reason: termination_reason.to_string(),
    };
    write_json_atomic(
        &backup_root.join("trim_result.json"),
        &serde_json::to_string_pretty(&result)?,
    )?;
    Ok(result)
}

fn create_backup(
    episode_root: &Path,
    manifest: &RecordedEpisodeManifest,
    boundary_chunk_index: u32,
    end_step: u32,
) -> Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_secs();
    let backup_root = episode_root
        .join(".trim_backups")
        .join(format!("end_step_{end_step:09}_{timestamp}"));
    ensure!(
        !backup_root.exists(),
        "trim backup already exists: {}",
        backup_root.display()
    );
    fs::create_dir_all(backup_root.join("steps"))
        .with_context(|| format!("failed to create {}", backup_root.display()))?;
    fs::copy(
        episode_root.join("episode.ron"),
        backup_root.join("episode.ron"),
    )
    .context("failed to back up episode manifest")?;

    for entry in &manifest.step_chunks {
        if entry.chunk_index < boundary_chunk_index {
            continue;
        }
        let source = episode_root.join(&entry.file_path);
        let destination = backup_root.join(&entry.file_path);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to back up {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(backup_root)
}

fn write_ron_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let payload = ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default())?;
    write_atomic(path, payload.as_bytes())
}

fn write_json_atomic(path: &Path, payload: &str) -> Result<()> {
    write_atomic(path, format!("{payload}\n").as_bytes())
}

fn write_atomic(path: &Path, payload: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("output path has no UTF-8 file name")?;
    let temporary_path = path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()));
    fs::write(&temporary_path, payload)
        .with_context(|| format!("failed to write {}", temporary_path.display()))?;
    fs::rename(&temporary_path, path).with_context(|| {
        format!(
            "failed to replace {} with {}",
            path.display(),
            temporary_path.display()
        )
    })
}

fn parse_args<I>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut episode_root = None;
    let mut end_step = None;
    let mut reason = "curated_trim".to_string();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                episode_root = Some(PathBuf::from(
                    args.next().context("missing value for --episode")?,
                ));
            }
            "--end-step" => {
                end_step = Some(
                    args.next()
                        .context("missing value for --end-step")?
                        .parse()
                        .context("invalid --end-step")?,
                );
            }
            "--reason" => {
                reason = args.next().context("missing value for --reason")?;
            }
            "--help" | "-h" => {
                print_help();
                bail!("help requested");
            }
            other => bail!("unknown trim-recording argument: {other}"),
        }
    }

    Ok(CliArgs {
        episode_root: episode_root.context("missing --episode")?,
        end_step: end_step.context("missing --end-step")?,
        reason,
    })
}

fn print_help() {
    eprintln!(
        "Usage:\n  dfb_tool_dataset trim-recording --episode <episode-dir> --end-step <inclusive-step> [--reason <reason>]"
    );
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::api::types::{EnvironmentAction, ObservationCaptureConfig, StateObservation};
    use crate::recording::{
        RecordedDynamicWorldState, RecordedEpisodeManifest, RecordedStep, RecordedStepArtifacts,
        RecordedStepChunk, RecordedStepChunkIndexEntry, RecordingArtifactConvention,
    };

    use super::*;

    fn step(index: u32) -> RecordedStep {
        RecordedStep {
            index,
            tick: 100 + u64::from(index),
            sim_time_seconds: index as f32 / 60.0,
            fighter1_command: EnvironmentAction::default(),
            fighter2_command: EnvironmentAction::default(),
            state: StateObservation {
                tick: 100 + u64::from(index),
                sim_time_seconds: index as f32 / 60.0,
                ..Default::default()
            },
            dynamic: RecordedDynamicWorldState::default(),
            audio_semantics: None,
        }
    }

    fn write_ron<T: Serialize>(path: &Path, value: &T) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(
            path,
            ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default()).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn trim_episode_rewrites_boundary_chunk_and_preserves_backup() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("dfb-recording-trim-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("steps")).unwrap();

        let chunks = [
            RecordedStepChunk {
                chunk_index: 0,
                start_step_index: 0,
                steps: vec![step(0), step(1)],
            },
            RecordedStepChunk {
                chunk_index: 1,
                start_step_index: 2,
                steps: vec![step(2), step(3)],
            },
        ];
        for chunk in &chunks {
            write_ron(
                &root.join(format!("steps/chunk_{:06}.ron", chunk.chunk_index)),
                chunk,
            );
        }

        let manifest = RecordedEpisodeManifest {
            schema_version: 12,
            policy_contract_id: "test_policy".to_string(),
            action_schema_id: "test_action".to_string(),
            episode_id: "trim_fixture".to_string(),
            scene_name: "default".to_string(),
            seed: Some(42),
            fixed_time_step_seconds: 1.0 / 60.0,
            capture_config: ObservationCaptureConfig::default(),
            audio_artifact_metadata: None,
            started_tick: 100,
            started_sim_time_seconds: 0.0,
            final_tick: 103,
            final_sim_time_seconds: 3.0 / 60.0,
            total_steps: 4,
            step_chunks: vec![
                RecordedStepChunkIndexEntry {
                    chunk_index: 0,
                    start_step_index: 0,
                    step_count: 2,
                    file_path: "steps/chunk_000000.ron".to_string(),
                },
                RecordedStepChunkIndexEntry {
                    chunk_index: 1,
                    start_step_index: 2,
                    step_count: 2,
                    file_path: "steps/chunk_000001.ron".to_string(),
                },
            ],
            step_artifacts: (0..4)
                .map(|index| RecordedStepArtifacts {
                    index,
                    tick: 100 + u64::from(index),
                    visual: Vec::new(),
                    audio: None,
                })
                .collect(),
            termination_reason: "match_finished".to_string(),
            winner: Some("fighter2".to_string()),
            bridge_role: None,
            bridge_session: None,
            bridge_transport: None,
            authoritative_source: true,
            initial_state_path: "initial_state.ron".to_string(),
            artifact_convention: RecordingArtifactConvention::default(),
        };
        write_ron(&root.join("episode.ron"), &manifest);

        let result = trim_episode(&root, 2, "curated_trim").unwrap();
        let access = RecordingAccess::new(&root);
        let trimmed = access.manifest().unwrap();
        let steps = access.steps().unwrap();

        assert_eq!(result.original_total_steps, 4);
        assert_eq!(result.trimmed_total_steps, 3);
        assert_eq!(trimmed.total_steps, 3);
        assert_eq!(trimmed.final_tick, 102);
        assert_eq!(trimmed.termination_reason, "curated_trim");
        assert_eq!(trimmed.winner, None);
        assert_eq!(trimmed.step_artifacts.len(), 3);
        assert_eq!(
            steps.iter().map(|step| step.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(Path::new(&result.backup_root).join("episode.ron").exists());
        assert!(
            Path::new(&result.backup_root)
                .join("steps/chunk_000001.ron")
                .exists()
        );
        assert!(!root.join("steps/chunk_000001.ron").exists());

        fs::remove_dir_all(root).unwrap();
    }
}
