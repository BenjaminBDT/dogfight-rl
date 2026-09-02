use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use std::{collections::HashMap, fs::File};

use crate::api::snapshot::{
    collect_audio_observation_only, collect_visual_observations_for_variant,
    collect_visual_observations_only,
};
use crate::api::types::{
    AudioObservation, EnvironmentEvent, ObservationCaptureConfig, PixelFormat,
    VisualCaptureVariant, VisualObservation, VisualResolutionMode, VisualSensorConfig,
    VisualSensorKind,
};
use crate::api::vision::{
    PendingVisualCaptureRequest, PendingVisualCaptures, SemanticCaptureMode,
    clear_offscreen_visual_frames, drain_offscreen_visual_frames_now,
    offscreen_visual_frames_ready, shutdown_visual_capture,
};
use crate::app::game_app::build_headless_offscreen_capture_app_with_paths;
use crate::audio::{
    AudioEventQueue, accumulate_audio_capture_step, reset_capture_audio_observation,
};
use crate::core::config::{ConfigPaths, resolve_project_root};
use crate::presentation::hud::ObservedAircraftRole;
use crate::recording::reconstruct::{LoadedEpisode, load_episode, restore_recorded_world};
use crate::recording::{
    AudioArtifactMetadata, AudioArtifactRef, DerivedArtifactConvention, DerivedEpisodeManifest,
    DerivedStepArtifacts, RecordedAudioFrame, RecordedDynamicWorldState,
    RecordingArtifactConvention, VisualArtifactRef, queue_recorded_audio_for_playback,
    render_named_artifact_pattern,
};
use crate::simulation::components::AircraftRole;
use crate::simulation::systems::{AircraftVisualPartsInitialized, AircraftVisualSceneRoot};
use anyhow::{Context, Result, bail};
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

struct VisualBundleWriter {
    relative_path: String,
    file: File,
    next_offset: u64,
}

enum VisualArtifactStorage {
    IndividualFiles,
    Bundles(HashMap<VisualSensorKind, VisualBundleWriter>),
}

type SegmentationArtifactStorage = VisualArtifactStorage;

struct AudioBundleWriter {
    relative_path: String,
    file: File,
    next_offset: u64,
}

enum AudioArtifactStorage {
    IndividualFiles,
    Bundle(AudioBundleWriter),
}

struct AudioCaptureResourceSnapshot {
    observation: crate::audio::AudioObservationState,
    queue: AudioEventQueue,
}

#[derive(Debug)]
struct CliArgs {
    episode_root: PathBuf,
    output_dir: Option<PathBuf>,
    observed_roles: Vec<AircraftRole>,
    capture_config: ObservationCaptureConfig,
    max_steps: Option<usize>,
    profile: bool,
    validation_video: bool,
    validation_video_path: Option<PathBuf>,
    force: bool,
}

#[derive(Debug, Default, Clone)]
struct ExtractStageProfile {
    restore_world_for_visual: Duration,
    take_audio_snapshot: Duration,
    visual_capture: Duration,
    restore_world_for_audio: Duration,
    restore_audio_resources: Duration,
    audio_capture: Duration,
    collect_rgb: Duration,
    collect_semantic: Duration,
    collect_audio_observation: Duration,
    write_visual: Duration,
    write_segmentation: Duration,
    write_audio: Duration,
    validate: Duration,
    visual_capture_updates: u64,
}

impl ExtractStageProfile {
    fn total_measured(&self) -> Duration {
        self.restore_world_for_visual
            + self.take_audio_snapshot
            + self.visual_capture
            + self.restore_world_for_audio
            + self.restore_audio_resources
            + self.audio_capture
            + self.collect_rgb
            + self.collect_semantic
            + self.collect_audio_observation
            + self.write_visual
            + self.write_segmentation
            + self.write_audio
            + self.validate
    }

    fn merge_assign(&mut self, other: &Self) {
        self.restore_world_for_visual += other.restore_world_for_visual;
        self.take_audio_snapshot += other.take_audio_snapshot;
        self.visual_capture += other.visual_capture;
        self.restore_world_for_audio += other.restore_world_for_audio;
        self.restore_audio_resources += other.restore_audio_resources;
        self.audio_capture += other.audio_capture;
        self.collect_rgb += other.collect_rgb;
        self.collect_semantic += other.collect_semantic;
        self.collect_audio_observation += other.collect_audio_observation;
        self.write_visual += other.write_visual;
        self.write_segmentation += other.write_segmentation;
        self.write_audio += other.write_audio;
        self.validate += other.validate;
        self.visual_capture_updates += other.visual_capture_updates;
    }
}

#[derive(Debug, Default)]
struct ExtractProfileReport {
    initial: ExtractStageProfile,
    per_step_total: ExtractStageProfile,
    processed_steps: usize,
}

impl ExtractProfileReport {
    fn record_initial(&mut self, profile: &ExtractStageProfile) {
        self.initial = profile.clone();
    }

    fn record_step(&mut self, profile: &ExtractStageProfile) {
        self.per_step_total.merge_assign(profile);
        self.processed_steps += 1;
    }
}

fn profile_stage<T, F>(accumulator: &mut Duration, op: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let started_at = Instant::now();
    let result = op();
    *accumulator += started_at.elapsed();
    result
}

fn format_stage_ms(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1000.0)
}

fn print_profile_report(role: AircraftRole, wall_elapsed: Duration, report: &ExtractProfileReport) {
    println!(
        "extract profile role={} wall_ms={} initial_measured_ms={} step_measured_ms={} steps={}",
        observed_role_name(role),
        format_stage_ms(wall_elapsed),
        format_stage_ms(report.initial.total_measured()),
        format_stage_ms(report.per_step_total.total_measured()),
        report.processed_steps,
    );

    let mut rows = vec![
        (
            "restore_world_for_visual",
            report.initial.restore_world_for_visual,
            report.per_step_total.restore_world_for_visual,
        ),
        (
            "take_audio_snapshot",
            report.initial.take_audio_snapshot,
            report.per_step_total.take_audio_snapshot,
        ),
        (
            "visual_capture",
            report.initial.visual_capture,
            report.per_step_total.visual_capture,
        ),
        (
            "restore_world_for_audio",
            report.initial.restore_world_for_audio,
            report.per_step_total.restore_world_for_audio,
        ),
        (
            "restore_audio_resources",
            report.initial.restore_audio_resources,
            report.per_step_total.restore_audio_resources,
        ),
        (
            "audio_capture",
            report.initial.audio_capture,
            report.per_step_total.audio_capture,
        ),
        (
            "collect_rgb",
            report.initial.collect_rgb,
            report.per_step_total.collect_rgb,
        ),
        (
            "collect_semantic",
            report.initial.collect_semantic,
            report.per_step_total.collect_semantic,
        ),
        (
            "collect_audio_observation",
            report.initial.collect_audio_observation,
            report.per_step_total.collect_audio_observation,
        ),
        (
            "write_visual",
            report.initial.write_visual,
            report.per_step_total.write_visual,
        ),
        (
            "write_segmentation",
            report.initial.write_segmentation,
            report.per_step_total.write_segmentation,
        ),
        (
            "write_audio",
            report.initial.write_audio,
            report.per_step_total.write_audio,
        ),
        (
            "validate",
            report.initial.validate,
            report.per_step_total.validate,
        ),
    ];

    let total = report.per_step_total.total_measured();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    for (name, initial, step_total) in rows {
        let step_avg = if report.processed_steps == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(step_total.as_secs_f64() / report.processed_steps as f64)
        };
        let share = if total.is_zero() {
            0.0
        } else {
            step_total.as_secs_f64() / total.as_secs_f64() * 100.0
        };
        println!(
            "extract profile role={} stage={} initial_ms={} step_total_ms={} step_avg_ms={} share={:.1}%",
            observed_role_name(role),
            name,
            format_stage_ms(initial),
            format_stage_ms(step_total),
            format_stage_ms(step_avg),
            share,
        );
    }
    if report.processed_steps > 0 {
        println!(
            "extract profile role={} visual_capture_updates initial={} step_total={} step_avg={:.3}",
            observed_role_name(role),
            report.initial.visual_capture_updates,
            report.per_step_total.visual_capture_updates,
            report.per_step_total.visual_capture_updates as f64 / report.processed_steps as f64,
        );
    }
}

fn derived_visual_capture_config(base: &ObservationCaptureConfig) -> ObservationCaptureConfig {
    ObservationCaptureConfig {
        enable_visual: base.enable_visual,
        enable_audio: base.enable_audio,
        visual_sensors: base
            .visual_sensors
            .iter()
            .map(|sensor| {
                let mut sensor = sensor.clone();
                sensor.capture_variants =
                    vec![VisualCaptureVariant::Rgb, VisualCaptureVariant::Semantic];
                sensor
            })
            .collect(),
        audio_window_seconds: base.audio_window_seconds,
    }
}

pub fn run_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let args = parse_args(args)?;
    let capture_config = derived_visual_capture_config(&args.capture_config);
    let episode = load_episode(&args.episode_root)?;
    let fixed_step =
        Duration::from_secs_f32(episode.manifest.fixed_time_step_seconds.max(1.0 / 240.0));

    let mut app = build_headless_offscreen_capture_app_with_paths(
        ConfigPaths {
            project_root: resolve_project_root(),
            scene_override: Some(episode.manifest.scene_name.clone()),
            scene_override_path: None,
        },
        false,
    );
    app.insert_resource(SemanticCaptureMode(
        capture_config.enable_visual
            && capture_config.visual_sensors.iter().any(|sensor| {
                sensor
                    .requested_capture_variants()
                    .contains(&VisualCaptureVariant::Semantic)
            }),
    ));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(fixed_step));
    app.finish();
    app.cleanup();
    app.update();
    app.update();
    warm_up_visual_pipeline(&mut app);
    reset_capture_audio_observation(app.world_mut());

    if args.validation_video_path.is_some()
        && (args.observed_roles.len() > 1 || args.capture_config.visual_sensors.len() > 1)
    {
        bail!("--validation-video <path> only supports a single observed role and a single camera");
    }

    for observed_role in &args.observed_roles {
        let mut profile_report = ExtractProfileReport::default();
        if let Some(mut app_capture_config) = app
            .world_mut()
            .get_resource_mut::<ObservationCaptureConfig>()
        {
            *app_capture_config = capture_config.clone();
        }
        app.world_mut()
            .insert_resource(ObservedAircraftRole(*observed_role));
        reset_capture_audio_observation(app.world_mut());

        let output_root = output_root_for_role(&args, *observed_role);
        prepare_output_dirs(&output_root, &args.capture_config, args.force)?;
        let mut visual_storage = create_visual_artifact_storage(
            &output_root,
            &args.capture_config,
            args.validation_video,
        )?;
        let mut segmentation_storage =
            create_segmentation_artifact_storage(&output_root, &args.capture_config)?;
        let mut audio_storage = create_audio_artifact_storage(
            &output_root,
            &args.capture_config,
            args.validation_video,
        )?;
        let mut audio_artifact_metadata: Option<AudioArtifactMetadata> = None;

        let (initial_visual, initial_segmentation, initial_audio, initial_audio_observation) =
            reconstruct_initial_frame(
                &mut app,
                &episode,
                &output_root,
                &mut visual_storage,
                &mut segmentation_storage,
                &mut audio_storage,
                &args.capture_config,
                *observed_role,
                args.profile,
            )?;
        if args.profile {
            profile_report.record_initial(&initial_audio_observation.1);
        }
        let initial_audio_observation = initial_audio_observation.0;
        if audio_artifact_metadata.is_none() {
            audio_artifact_metadata = initial_audio_observation
                .as_ref()
                .map(AudioArtifactMetadata::from);
        }

        let mut validation_audio = if args.validation_video {
            Some(ContinuousAudioTrack::default())
        } else {
            None
        };

        let total_source_steps = episode.steps.len();
        let selected_step_count = args
            .max_steps
            .map(|max_steps| max_steps.min(total_source_steps))
            .unwrap_or(total_source_steps);
        println!(
            "extract role={} total_steps={} selected_steps={} visual={} audio={}",
            observed_role_name(*observed_role),
            total_source_steps,
            selected_step_count,
            capture_config.enable_visual,
            capture_config.enable_audio,
        );
        let started_at = Instant::now();
        let mut steps = Vec::with_capacity(selected_step_count);
        for (processed_index, step) in episode.steps.iter().take(selected_step_count).enumerate() {
            let (visual, segmentation, audio, audio_observation) = reconstruct_step_frame(
                &mut app,
                step.index,
                episode.manifest.fixed_time_step_seconds,
                &step.state,
                &step.dynamic,
                step.audio_semantics.as_ref(),
                &step.state.events_since_last_step,
                &output_root,
                &mut visual_storage,
                &mut segmentation_storage,
                &mut audio_storage,
                &args.capture_config,
                *observed_role,
                args.profile,
            )?;
            if args.profile {
                profile_report.record_step(&audio_observation.1);
            }
            let audio_observation = audio_observation.0;
            steps.push(DerivedStepArtifacts {
                index: step.index,
                tick: step.tick,
                visual,
                segmentation,
                audio: audio.clone(),
            });
            if let (Some(track), Some(audio_observation)) =
                (&mut validation_audio, audio_observation.as_ref())
            {
                track.append_step_observation(
                    audio_observation,
                    episode.manifest.fixed_time_step_seconds,
                );
            }
            if audio_artifact_metadata.is_none() {
                audio_artifact_metadata =
                    audio_observation.as_ref().map(AudioArtifactMetadata::from);
            }
            let completed = processed_index + 1;
            if completed == 1 || completed == selected_step_count || completed % 100 == 0 {
                println!(
                    "extract role={} progress={}/{} elapsed={:.1}s",
                    observed_role_name(*observed_role),
                    completed,
                    selected_step_count,
                    started_at.elapsed().as_secs_f32(),
                );
            }
        }
        if args.profile {
            print_profile_report(*observed_role, started_at.elapsed(), &profile_report);
        }

        let derived_manifest = DerivedEpisodeManifest {
            schema_version: 1,
            source_episode_id: episode.manifest.episode_id.clone(),
            source_episode_root: args.episode_root.display().to_string(),
            observed_role: observed_role_name(*observed_role).to_string(),
            capture_config: capture_config.clone(),
            audio_artifact_metadata,
            initial_tick: episode.initial.state.tick,
            total_steps: u32::try_from(steps.len())
                .context("derived episode step count exceeded u32 range")?,
            initial_visual,
            initial_segmentation,
            initial_audio,
            steps,
            artifact_convention: DerivedArtifactConvention::default(),
        };
        let derived_manifest_path =
            output_root.join(&DerivedArtifactConvention::default().manifest_file);
        fs::write(
            &derived_manifest_path,
            ron::ser::to_string_pretty(&derived_manifest, ron::ser::PrettyConfig::default())?,
        )?;
        println!("wrote {}", derived_manifest_path.display());

        if args.validation_video {
            let validation_root = validation_root_for_role(&args, *observed_role)?;
            for camera in &args.capture_config.visual_sensors {
                let video_path = validation_video_path_for_role(
                    &args,
                    &validation_root,
                    &output_root,
                    camera.kind,
                );
                render_validation_video(
                    &output_root,
                    &validation_root,
                    &derived_manifest,
                    validation_audio.as_ref(),
                    camera.kind,
                    &video_path,
                )?;
                println!("wrote {}", video_path.display());
            }
        }
    }
    drain_visual_capture_shutdown(&mut app);
    Ok(())
}

#[derive(Debug, Default)]
struct ContinuousAudioTrack {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

impl ContinuousAudioTrack {
    fn append_step_observation(&mut self, audio: &AudioObservation, step_seconds: f32) {
        let sample_rate = audio.sample_rate;
        let channels = audio.channels;
        if self.samples.is_empty() {
            self.sample_rate = sample_rate;
            self.channels = channels;
        } else if self.sample_rate != sample_rate || self.channels != channels {
            return;
        }
        let step_frames = (sample_rate as f32 * step_seconds.max(1.0 / 240.0)).round() as usize;
        let step_samples = step_frames * channels as usize;
        let start = audio.samples.len().saturating_sub(step_samples);
        self.samples.extend_from_slice(&audio.samples[start..]);
    }
}

fn reconstruct_initial_frame(
    app: &mut App,
    episode: &LoadedEpisode,
    output_root: &Path,
    visual_storage: &mut VisualArtifactStorage,
    segmentation_storage: &mut SegmentationArtifactStorage,
    audio_storage: &mut AudioArtifactStorage,
    capture_config: &ObservationCaptureConfig,
    observed_role: AircraftRole,
    profile_enabled: bool,
) -> Result<(
    Vec<VisualArtifactRef>,
    Vec<VisualArtifactRef>,
    Option<AudioArtifactRef>,
    (Option<AudioObservation>, ExtractStageProfile),
)> {
    let mut profile = ExtractStageProfile::default();
    if profile_enabled {
        profile_stage(&mut profile.restore_world_for_visual, || {
            restore_recorded_world(
                app.world_mut(),
                &episode.initial.state,
                &episode.initial.dynamic,
                observed_role,
            );
            Ok(())
        })?;
    } else {
        restore_recorded_world(
            app.world_mut(),
            &episode.initial.state,
            &episode.initial.dynamic,
            observed_role,
        );
    }
    let audio_snapshot = if profile_enabled {
        profile_stage(&mut profile.take_audio_snapshot, || {
            take_audio_capture_resources(app.world_mut())
        })?
    } else {
        take_audio_capture_resources(app.world_mut())?
    };
    if profile_enabled {
        profile.visual_capture_updates = profile_stage(&mut profile.visual_capture, || {
            run_visual_capture_updates(
                app,
                episode.manifest.fixed_time_step_seconds,
                &episode.initial.state,
                &episode.initial.dynamic,
            )
        })?;
        profile_stage(&mut profile.restore_world_for_audio, || {
            restore_recorded_world(
                app.world_mut(),
                &episode.initial.state,
                &episode.initial.dynamic,
                observed_role,
            );
            Ok(())
        })?;
        profile_stage(&mut profile.restore_audio_resources, || {
            restore_audio_capture_resources(app.world_mut(), audio_snapshot);
            Ok(())
        })?;
        profile_stage(&mut profile.audio_capture, || {
            run_audio_capture_updates(
                app,
                episode.manifest.fixed_time_step_seconds,
                episode.initial.audio_semantics.as_ref(),
                &episode.initial.state.events_since_last_step,
            )
        })?;
    } else {
        profile.visual_capture_updates = run_visual_capture_updates(
            app,
            episode.manifest.fixed_time_step_seconds,
            &episode.initial.state,
            &episode.initial.dynamic,
        )?;
        restore_recorded_world(
            app.world_mut(),
            &episode.initial.state,
            &episode.initial.dynamic,
            observed_role,
        );
        restore_audio_capture_resources(app.world_mut(), audio_snapshot);
        run_audio_capture_updates(
            app,
            episode.manifest.fixed_time_step_seconds,
            episode.initial.audio_semantics.as_ref(),
            &episode.initial.state.events_since_last_step,
        )?;
    }
    let visual_observation = if profile_enabled {
        profile_stage(&mut profile.collect_rgb, || {
            Ok(collect_visual_observations_only(app.world_mut()))
        })?
    } else {
        collect_visual_observations_only(app.world_mut())
    };
    let segmentation_observation = if profile_enabled {
        profile_stage(&mut profile.collect_semantic, || {
            Ok(collect_visual_observations_for_variant(
                app.world_mut(),
                VisualCaptureVariant::Semantic,
            ))
        })?
    } else {
        collect_visual_observations_for_variant(app.world_mut(), VisualCaptureVariant::Semantic)
    };
    let audio_observation = if profile_enabled {
        profile_stage(&mut profile.collect_audio_observation, || {
            Ok(collect_audio_observation_only(app.world_mut()))
        })?
    } else {
        collect_audio_observation_only(app.world_mut())
    };
    let visual = visual_observation.iter();
    let visual = if profile_enabled {
        profile_stage(&mut profile.write_visual, || {
            visual
                .map(|frame| write_visual_artifact(output_root, visual_storage, "initial", frame))
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        visual
            .map(|frame| write_visual_artifact(output_root, visual_storage, "initial", frame))
            .collect::<Result<Vec<_>>>()?
    };
    let segmentation = segmentation_observation.iter();
    let segmentation = if profile_enabled {
        profile_stage(&mut profile.write_segmentation, || {
            segmentation
                .map(|frame| {
                    write_segmentation_artifact(output_root, segmentation_storage, "initial", frame)
                })
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        segmentation
            .map(|frame| {
                write_segmentation_artifact(output_root, segmentation_storage, "initial", frame)
            })
            .collect::<Result<Vec<_>>>()?
    };
    let audio = if profile_enabled {
        profile_stage(&mut profile.write_audio, || {
            audio_observation
                .as_ref()
                .map(|audio| write_audio_artifact(output_root, audio_storage, "initial", audio))
                .transpose()
        })?
    } else {
        audio_observation
            .as_ref()
            .map(|audio| write_audio_artifact(output_root, audio_storage, "initial", audio))
            .transpose()?
    };
    if profile_enabled {
        profile_stage(&mut profile.validate, || {
            validate_capture_config(
                capture_config,
                &visual_observation,
                audio_observation.as_ref(),
            )
        })?;
    } else {
        validate_capture_config(
            capture_config,
            &visual_observation,
            audio_observation.as_ref(),
        )?;
    }
    Ok((visual, segmentation, audio, (audio_observation, profile)))
}

fn warm_up_visual_pipeline(app: &mut App) {
    const MAX_WARMUP_FRAMES: usize = 240;
    for _ in 0..MAX_WARMUP_FRAMES {
        let root_count = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<Entity, With<AircraftVisualSceneRoot>>();
            query.iter(world).count()
        };
        let initialized_count = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<Entity, With<AircraftVisualPartsInitialized>>();
            query.iter(world).count()
        };
        if root_count > 0 && initialized_count >= root_count {
            let semantic_ready = {
                let world = app.world_mut();
                let semantic_mode = world
                    .get_resource::<SemanticCaptureMode>()
                    .is_some_and(|mode| mode.0);
                if !semantic_mode {
                    true
                } else {
                    let mut semantic_root_query = world.query_filtered::<Entity, With<crate::simulation::systems::AircraftSemanticSceneRoot>>();
                    let semantic_root_count = semantic_root_query.iter(world).count();
                    let mut semantic_initialized_query = world.query_filtered::<Entity, With<crate::simulation::systems::AircraftSemanticPartsInitialized>>();
                    let semantic_initialized_count = semantic_initialized_query.iter(world).count();
                    semantic_root_count == 0 || semantic_initialized_count >= semantic_root_count
                }
            };
            if semantic_ready {
                break;
            }
        }
        app.update();
    }
}

fn flush_offscreen_visual_capture(app: &mut App, step_seconds: f32) -> Result<u64> {
    const MAX_CAPTURE_UPDATES: usize = 8;
    clear_offscreen_visual_frames(app.world_mut());
    app.world_mut()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            step_seconds.max(1.0 / 240.0),
        )));
    app.update();
    drain_offscreen_visual_frames_now(app.world_mut());
    if offscreen_visual_frames_ready(app.world()) {
        return Ok(1);
    }
    for update_index in 1..MAX_CAPTURE_UPDATES {
        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
        app.update();
        drain_offscreen_visual_frames_now(app.world_mut());
        if offscreen_visual_frames_ready(app.world()) {
            return Ok((update_index + 1) as u64);
        }
    }
    bail!("offscreen visual capture did not become ready within {MAX_CAPTURE_UPDATES} updates");
}

fn reconstruct_step_frame(
    app: &mut App,
    step_index: u32,
    step_seconds: f32,
    state: &crate::api::types::StateObservation,
    dynamic: &crate::recording::RecordedDynamicWorldState,
    audio_semantics: Option<&RecordedAudioFrame>,
    events: &[EnvironmentEvent],
    output_root: &Path,
    visual_storage: &mut VisualArtifactStorage,
    segmentation_storage: &mut SegmentationArtifactStorage,
    audio_storage: &mut AudioArtifactStorage,
    capture_config: &ObservationCaptureConfig,
    observed_role: AircraftRole,
    profile_enabled: bool,
) -> Result<(
    Vec<VisualArtifactRef>,
    Vec<VisualArtifactRef>,
    Option<AudioArtifactRef>,
    (Option<AudioObservation>, ExtractStageProfile),
)> {
    let mut profile = ExtractStageProfile::default();
    if profile_enabled {
        profile_stage(&mut profile.restore_world_for_visual, || {
            restore_recorded_world(app.world_mut(), state, dynamic, observed_role);
            Ok(())
        })?;
    } else {
        restore_recorded_world(app.world_mut(), state, dynamic, observed_role);
    }
    let audio_snapshot = if profile_enabled {
        profile_stage(&mut profile.take_audio_snapshot, || {
            take_audio_capture_resources(app.world_mut())
        })?
    } else {
        take_audio_capture_resources(app.world_mut())?
    };
    if profile_enabled {
        profile.visual_capture_updates = profile_stage(&mut profile.visual_capture, || {
            run_visual_capture_updates(app, step_seconds, state, dynamic)
        })?;
        profile_stage(&mut profile.restore_world_for_audio, || {
            restore_recorded_world(app.world_mut(), state, dynamic, observed_role);
            Ok(())
        })?;
        profile_stage(&mut profile.restore_audio_resources, || {
            restore_audio_capture_resources(app.world_mut(), audio_snapshot);
            Ok(())
        })?;
        profile_stage(&mut profile.audio_capture, || {
            run_audio_capture_updates(app, step_seconds, audio_semantics, events)
        })?;
    } else {
        profile.visual_capture_updates =
            run_visual_capture_updates(app, step_seconds, state, dynamic)?;
        restore_recorded_world(app.world_mut(), state, dynamic, observed_role);
        restore_audio_capture_resources(app.world_mut(), audio_snapshot);
        run_audio_capture_updates(app, step_seconds, audio_semantics, events)?;
    }
    let visual_observation = if profile_enabled {
        profile_stage(&mut profile.collect_rgb, || {
            Ok(collect_visual_observations_only(app.world_mut()))
        })?
    } else {
        collect_visual_observations_only(app.world_mut())
    };
    let segmentation_observation = if profile_enabled {
        profile_stage(&mut profile.collect_semantic, || {
            Ok(collect_visual_observations_for_variant(
                app.world_mut(),
                VisualCaptureVariant::Semantic,
            ))
        })?
    } else {
        collect_visual_observations_for_variant(app.world_mut(), VisualCaptureVariant::Semantic)
    };
    let audio_observation = if profile_enabled {
        profile_stage(&mut profile.collect_audio_observation, || {
            Ok(collect_audio_observation_only(app.world_mut()))
        })?
    } else {
        collect_audio_observation_only(app.world_mut())
    };
    let frame_key = format!("step_{step_index:06}");
    let visual = if profile_enabled {
        profile_stage(&mut profile.write_visual, || {
            visual_observation
                .iter()
                .map(|frame| write_visual_artifact(output_root, visual_storage, &frame_key, frame))
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        visual_observation
            .iter()
            .map(|frame| write_visual_artifact(output_root, visual_storage, &frame_key, frame))
            .collect::<Result<Vec<_>>>()?
    };
    let segmentation = if profile_enabled {
        profile_stage(&mut profile.write_segmentation, || {
            segmentation_observation
                .iter()
                .map(|frame| {
                    write_segmentation_artifact(
                        output_root,
                        segmentation_storage,
                        &frame_key,
                        frame,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })?
    } else {
        segmentation_observation
            .iter()
            .map(|frame| {
                write_segmentation_artifact(output_root, segmentation_storage, &frame_key, frame)
            })
            .collect::<Result<Vec<_>>>()?
    };
    let audio = if profile_enabled {
        profile_stage(&mut profile.write_audio, || {
            audio_observation
                .as_ref()
                .map(|audio| write_audio_artifact(output_root, audio_storage, &frame_key, audio))
                .transpose()
        })?
    } else {
        audio_observation
            .as_ref()
            .map(|audio| write_audio_artifact(output_root, audio_storage, &frame_key, audio))
            .transpose()?
    };
    if profile_enabled {
        profile_stage(&mut profile.validate, || {
            validate_capture_config(
                capture_config,
                &visual_observation,
                audio_observation.as_ref(),
            )
        })?;
    } else {
        validate_capture_config(
            capture_config,
            &visual_observation,
            audio_observation.as_ref(),
        )?;
    }
    Ok((visual, segmentation, audio, (audio_observation, profile)))
}

fn run_visual_capture_updates(
    app: &mut App,
    step_seconds: f32,
    _state: &crate::api::types::StateObservation,
    _dynamic: &RecordedDynamicWorldState,
) -> Result<u64> {
    let (original_enable_visual, original_enable_audio) = app
        .world()
        .get_resource::<ObservationCaptureConfig>()
        .map(|config| (config.enable_visual, config.enable_audio))
        .unwrap_or((false, false));
    if original_enable_audio
        && let Some(mut capture_config) = app
            .world_mut()
            .get_resource_mut::<ObservationCaptureConfig>()
    {
        capture_config.enable_audio = false;
    }
    let updates = if original_enable_visual {
        flush_offscreen_visual_capture(app, step_seconds)?
    } else {
        app.world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                step_seconds.max(1.0 / 240.0),
            )));
        app.update();
        1
    };
    if original_enable_visual {
        // no-op: updates already captured above
    }
    if original_enable_audio
        && let Some(mut capture_config) = app
            .world_mut()
            .get_resource_mut::<ObservationCaptureConfig>()
    {
        capture_config.enable_audio = true;
    }
    Ok(updates)
}

fn take_audio_capture_resources(world: &mut World) -> Result<AudioCaptureResourceSnapshot> {
    let observation = {
        let mut observation = world
            .get_resource_mut::<crate::audio::AudioObservationState>()
            .with_context(|| "missing AudioObservationState in modalities app")?;
        std::mem::take(&mut *observation)
    };
    let queue = {
        let mut queue = world
            .get_resource_mut::<AudioEventQueue>()
            .with_context(|| "missing AudioEventQueue in modalities app")?;
        std::mem::take(&mut *queue)
    };
    Ok(AudioCaptureResourceSnapshot { observation, queue })
}

fn restore_audio_capture_resources(world: &mut World, snapshot: AudioCaptureResourceSnapshot) {
    world.insert_resource(snapshot.observation);
    world.insert_resource(snapshot.queue);
}

fn run_audio_capture_updates(
    app: &mut App,
    step_seconds: f32,
    audio_semantics: Option<&RecordedAudioFrame>,
    events: &[EnvironmentEvent],
) -> Result<()> {
    app.world_mut()
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            step_seconds.max(1.0 / 240.0),
        )));
    if let Some(mut queue) = app.world_mut().get_resource_mut::<AudioEventQueue>() {
        queue_recorded_audio_for_playback(&mut queue, audio_semantics, events);
    }
    accumulate_audio_capture_step(app.world_mut())?;
    Ok(())
}

fn validate_capture_config(
    capture_config: &ObservationCaptureConfig,
    visual: &[VisualObservation],
    audio: Option<&AudioObservation>,
) -> Result<()> {
    if capture_config.enable_visual && visual.is_empty() {
        bail!("visual reconstruction produced no frames");
    }
    if capture_config.enable_audio && audio.is_none() {
        bail!("audio reconstruction produced no observation window");
    }
    Ok(())
}

fn parse_args<I>(args: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter().peekable();
    let mut episode_root: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut observed_roles = vec![AircraftRole::Fighter1, AircraftRole::Fighter2];
    let mut enable_visual = true;
    let mut enable_audio = true;
    let mut include_hud = false;
    let mut width = 160;
    let mut height = 100;
    let mut audio_window_seconds = 1.0 / 60.0;
    let mut max_steps: Option<usize> = None;
    let mut profile = false;
    let mut visual_sensors = vec![VisualSensorKind::Front, VisualSensorKind::Rear];
    let mut validation_video = false;
    let mut validation_video_path: Option<PathBuf> = None;
    let mut force = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--episode" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --episode");
                };
                episode_root = Some(PathBuf::from(value));
            }
            "--output-dir" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --output-dir");
                };
                output_dir = Some(PathBuf::from(value));
            }
            "--observed-role" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --observed-role");
                };
                observed_roles = parse_roles(&value)?;
            }
            "--camera" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --camera");
                };
                visual_sensors = parse_cameras(&value)?;
            }
            "--no-visual" => enable_visual = false,
            "--no-audio" => enable_audio = false,
            "--include-hud" => include_hud = true,
            "--force" => force = true,
            "--visual-resolution" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --visual-resolution");
                };
                (width, height) = parse_visual_resolution(&value)?;
            }
            "--width" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --width");
                };
                width = value.parse().context("invalid --width")?;
            }
            "--height" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --height");
                };
                height = value.parse().context("invalid --height")?;
            }
            "--audio-window" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --audio-window");
                };
                audio_window_seconds = value.parse().context("invalid --audio-window")?;
            }
            "--max-steps" => {
                let Some(value) = args.next() else {
                    bail!("missing value for --max-steps");
                };
                let parsed = value.parse::<usize>().context("invalid --max-steps")?;
                anyhow::ensure!(parsed > 0, "--max-steps must be positive");
                max_steps = Some(parsed);
            }
            "--profile" => profile = true,
            "--validation-video" => {
                validation_video = true;
                if let Some(value) = args.peek().cloned()
                    && !value.starts_with("--")
                {
                    validation_video_path = Some(PathBuf::from(args.next().unwrap()));
                };
            }
            "--front-only" => visual_sensors = vec![VisualSensorKind::Front],
            "--rear-only" => visual_sensors = vec![VisualSensorKind::Rear],
            other => bail!("unknown argument: {other}"),
        }
    }

    let Some(episode_root) = episode_root else {
        bail!("missing required --episode <path>");
    };
    let capture_config = ObservationCaptureConfig {
        enable_visual,
        enable_audio,
        visual_sensors: if enable_visual {
            visual_sensors
                .into_iter()
                .map(|kind| VisualSensorConfig {
                    kind,
                    width,
                    height,
                    format: PixelFormat::Rgb8,
                    resolution_mode: VisualResolutionMode::Fixed,
                    include_hud,
                    capture_variants: Vec::new(),
                })
                .collect()
        } else {
            Vec::new()
        },
        audio_window_seconds,
    };

    Ok(CliArgs {
        episode_root,
        output_dir,
        observed_roles,
        capture_config,
        max_steps,
        profile,
        validation_video,
        validation_video_path,
        force,
    })
}

fn parse_visual_resolution(value: &str) -> Result<(u32, u32)> {
    let Some((width, height)) = value.split_once('x').or_else(|| value.split_once('X')) else {
        bail!("invalid --visual-resolution, expected WIDTHxHEIGHT");
    };
    let width = width
        .parse::<u32>()
        .with_context(|| format!("invalid width in --visual-resolution: {value}"))?;
    let height = height
        .parse::<u32>()
        .with_context(|| format!("invalid height in --visual-resolution: {value}"))?;
    anyhow::ensure!(
        width > 0 && height > 0,
        "visual resolution must be non-zero"
    );
    Ok((width, height))
}

fn parse_role(value: &str) -> Result<AircraftRole> {
    match value {
        "fighter2" | "Fighter2" | "f2" => Ok(AircraftRole::Fighter2),
        "fighter1" | "Fighter1" | "f1" => Ok(AircraftRole::Fighter1),
        other => bail!("unknown observed role: {other}"),
    }
}

fn parse_roles(value: &str) -> Result<Vec<AircraftRole>> {
    match value {
        "all" | "All" => Ok(vec![AircraftRole::Fighter1, AircraftRole::Fighter2]),
        other => Ok(vec![parse_role(other)?]),
    }
}

fn parse_cameras(value: &str) -> Result<Vec<VisualSensorKind>> {
    match value {
        "front" | "Front" => Ok(vec![VisualSensorKind::Front]),
        "rear" | "Rear" => Ok(vec![VisualSensorKind::Rear]),
        "all" | "All" => Ok(vec![VisualSensorKind::Front, VisualSensorKind::Rear]),
        other => bail!("unknown camera selector: {other}"),
    }
}

fn default_output_dir(episode_root: &Path, observed_role: AircraftRole) -> PathBuf {
    episode_root
        .join("derived")
        .join(observed_role_name(observed_role))
}

fn validation_root_for_role(args: &CliArgs, observed_role: AircraftRole) -> Result<PathBuf> {
    if let Some(path) = &args.validation_video_path
        && let Some(parent) = path.parent()
    {
        fs::create_dir_all(parent)?;
        return Ok(parent.to_path_buf());
    }
    let root = args.episode_root.join(render_named_artifact_pattern(
        &RecordingArtifactConvention::default().validation_role_dir_pattern,
        None,
        None,
        None,
        Some(observed_role_name(observed_role)),
    ));
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn output_root_for_role(args: &CliArgs, observed_role: AircraftRole) -> PathBuf {
    match &args.output_dir {
        Some(root) if args.observed_roles.len() == 1 => root.clone(),
        Some(root) => root.join(observed_role_name(observed_role)),
        None => default_output_dir(&args.episode_root, observed_role),
    }
}

fn observed_role_name(role: AircraftRole) -> &'static str {
    match role {
        AircraftRole::Fighter1 => "fighter1",
        AircraftRole::Fighter2 => "fighter2",
    }
}

fn prepare_output_dirs(
    root: &Path,
    capture_config: &ObservationCaptureConfig,
    force: bool,
) -> Result<()> {
    if root.exists() {
        if force {
            remove_path_forcefully(root)?;
        } else {
            let is_non_empty = fs::read_dir(root)?.next().transpose()?.is_some();
            if is_non_empty {
                bail!(
                    "output directory {} is not empty; rerun with --force to overwrite",
                    root.display()
                );
            }
        }
    }
    fs::create_dir_all(root)?;
    if capture_config.enable_audio {
        fs::create_dir_all(root.join("audio"))?;
    }
    Ok(())
}

fn create_visual_artifact_storage(
    root: &Path,
    capture_config: &ObservationCaptureConfig,
    validation_video: bool,
) -> Result<VisualArtifactStorage> {
    if !capture_config.enable_visual || validation_video {
        return Ok(VisualArtifactStorage::IndividualFiles);
    }

    let mut bundles = HashMap::new();
    for sensor in &capture_config.visual_sensors {
        let camera = match sensor.kind {
            VisualSensorKind::Front => "front",
            VisualSensorKind::Rear => "rear",
        };
        let relative_path = render_named_artifact_pattern(
            &DerivedArtifactConvention::default().visual_bundle_pattern,
            Some(camera),
            None,
            None,
            None,
        );
        let full_path = root.join(&relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        bundles.insert(
            sensor.kind,
            VisualBundleWriter {
                relative_path,
                file: File::create(&full_path)?,
                next_offset: 0,
            },
        );
    }
    Ok(VisualArtifactStorage::Bundles(bundles))
}

fn create_segmentation_artifact_storage(
    root: &Path,
    capture_config: &ObservationCaptureConfig,
) -> Result<SegmentationArtifactStorage> {
    if !capture_config.enable_visual {
        return Ok(SegmentationArtifactStorage::IndividualFiles);
    }
    let mut bundles = HashMap::new();
    for sensor in &capture_config.visual_sensors {
        let camera = match sensor.kind {
            VisualSensorKind::Front => "front",
            VisualSensorKind::Rear => "rear",
        };
        let relative_path = render_named_artifact_pattern(
            &DerivedArtifactConvention::default().segmentation_bundle_pattern,
            Some(camera),
            None,
            None,
            None,
        );
        let full_path = root.join(&relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        bundles.insert(
            sensor.kind,
            VisualBundleWriter {
                relative_path,
                file: File::create(&full_path)?,
                next_offset: 0,
            },
        );
    }
    Ok(SegmentationArtifactStorage::Bundles(bundles))
}

fn create_audio_artifact_storage(
    root: &Path,
    capture_config: &ObservationCaptureConfig,
    _validation_video: bool,
) -> Result<AudioArtifactStorage> {
    if !capture_config.enable_audio {
        return Ok(AudioArtifactStorage::IndividualFiles);
    }

    let relative_path = DerivedArtifactConvention::default().audio_bundle_file;
    let full_path = root.join(&relative_path);
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(AudioArtifactStorage::Bundle(AudioBundleWriter {
        relative_path,
        file: File::create(&full_path)?,
        next_offset: 0,
    }))
}

fn remove_path_forcefully(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
                if let Err(error) = fs::remove_dir_all(path) {
                    if error.kind() != std::io::ErrorKind::DirectoryNotEmpty {
                        return Err(error.into());
                    }
                    remove_dir_contents_forcefully(path)?;
                    fs::remove_dir(path)?;
                }
            } else {
                fs::remove_file(path)?;
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_dir_contents_forcefully(path: &Path) -> Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        remove_path_forcefully(&entry.path())?;
    }
    Ok(())
}

fn write_visual_artifact(
    root: &Path,
    storage: &mut VisualArtifactStorage,
    frame_key: &str,
    frame: &VisualObservation,
) -> Result<VisualArtifactRef> {
    if !frame.bytes_ready || frame.bytes.is_empty() {
        return Ok(VisualArtifactRef {
            camera: frame.camera,
            file_path: None,
            width: Some(frame.width),
            height: Some(frame.height),
            format: Some(frame.format),
            byte_offset: None,
            byte_length: None,
        });
    }

    match storage {
        VisualArtifactStorage::IndividualFiles => {
            let subdir = match frame.camera {
                VisualSensorKind::Front => "front",
                VisualSensorKind::Rear => "rear",
            };
            let extension = match frame.format {
                PixelFormat::Rgb8 => "ppm",
                PixelFormat::Gray8 => "pgm",
                PixelFormat::Rgba8 => "raw",
            };
            let relative = PathBuf::from(render_named_artifact_pattern(
                &DerivedArtifactConvention::default().visual_artifact_pattern,
                Some(subdir),
                Some(frame_key),
                Some(extension),
                None,
            ));
            let full_path = root.join(&relative);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }
            match frame.format {
                PixelFormat::Rgb8 => {
                    write_ppm(&full_path, frame.width, frame.height, &frame.bytes)?
                }
                PixelFormat::Gray8 => {
                    write_pgm(&full_path, frame.width, frame.height, &frame.bytes)?
                }
                PixelFormat::Rgba8 => fs::write(&full_path, &frame.bytes)?,
            }
            Ok(VisualArtifactRef {
                camera: frame.camera,
                file_path: Some(relative.display().to_string()),
                width: Some(frame.width),
                height: Some(frame.height),
                format: Some(frame.format),
                byte_offset: None,
                byte_length: None,
            })
        }
        VisualArtifactStorage::Bundles(bundles) => {
            let writer = bundles
                .get_mut(&frame.camera)
                .with_context(|| format!("missing visual bundle writer for {:?}", frame.camera))?;
            let offset = writer.next_offset;
            writer.file.write_all(&frame.bytes)?;
            writer.next_offset += frame.bytes.len() as u64;
            Ok(VisualArtifactRef {
                camera: frame.camera,
                file_path: Some(writer.relative_path.clone()),
                width: Some(frame.width),
                height: Some(frame.height),
                format: Some(frame.format),
                byte_offset: Some(offset),
                byte_length: Some(frame.bytes.len() as u64),
            })
        }
    }
}

fn write_segmentation_artifact(
    root: &Path,
    storage: &mut SegmentationArtifactStorage,
    frame_key: &str,
    frame: &VisualObservation,
) -> Result<VisualArtifactRef> {
    if !frame.bytes_ready || frame.bytes.is_empty() {
        return Ok(VisualArtifactRef {
            camera: frame.camera,
            file_path: None,
            width: Some(frame.width),
            height: Some(frame.height),
            format: Some(frame.format),
            byte_offset: None,
            byte_length: None,
        });
    }

    match storage {
        SegmentationArtifactStorage::IndividualFiles => {
            let subdir = match frame.camera {
                VisualSensorKind::Front => "front",
                VisualSensorKind::Rear => "rear",
            };
            let extension = match frame.format {
                PixelFormat::Gray8 => "pgm",
                PixelFormat::Rgb8 => "ppm",
                PixelFormat::Rgba8 => "raw",
            };
            let relative = PathBuf::from(render_named_artifact_pattern(
                &DerivedArtifactConvention::default().segmentation_artifact_pattern,
                Some(subdir),
                Some(frame_key),
                Some(extension),
                None,
            ));
            let full_path = root.join(&relative);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }
            match frame.format {
                PixelFormat::Gray8 => {
                    write_pgm(&full_path, frame.width, frame.height, &frame.bytes)?
                }
                PixelFormat::Rgb8 => {
                    write_ppm(&full_path, frame.width, frame.height, &frame.bytes)?
                }
                PixelFormat::Rgba8 => fs::write(&full_path, &frame.bytes)?,
            }
            Ok(VisualArtifactRef {
                camera: frame.camera,
                file_path: Some(relative.display().to_string()),
                width: Some(frame.width),
                height: Some(frame.height),
                format: Some(frame.format),
                byte_offset: None,
                byte_length: None,
            })
        }
        SegmentationArtifactStorage::Bundles(bundles) => {
            let writer = bundles.get_mut(&frame.camera).with_context(|| {
                format!("missing segmentation bundle writer for {:?}", frame.camera)
            })?;
            let offset = writer.next_offset;
            writer.file.write_all(&frame.bytes)?;
            writer.next_offset += frame.bytes.len() as u64;
            Ok(VisualArtifactRef {
                camera: frame.camera,
                file_path: Some(writer.relative_path.clone()),
                width: Some(frame.width),
                height: Some(frame.height),
                format: Some(frame.format),
                byte_offset: Some(offset),
                byte_length: Some(frame.bytes.len() as u64),
            })
        }
    }
}

fn write_audio_artifact(
    root: &Path,
    storage: &mut AudioArtifactStorage,
    frame_key: &str,
    audio: &AudioObservation,
) -> Result<AudioArtifactRef> {
    match storage {
        AudioArtifactStorage::IndividualFiles => {
            let relative = PathBuf::from(render_named_artifact_pattern(
                &DerivedArtifactConvention::default().audio_artifact_pattern,
                None,
                Some(frame_key),
                None,
                None,
            ));
            let full_path = root.join(&relative);
            write_wav_i16(
                &full_path,
                audio.sample_rate,
                audio.channels,
                &audio.samples,
            )?;
            Ok(AudioArtifactRef {
                file_path: Some(relative.display().to_string()),
                byte_offset: None,
                byte_length: None,
            })
        }
        AudioArtifactStorage::Bundle(writer) => {
            let payload = encode_wav_i16_bytes(audio.sample_rate, audio.channels, &audio.samples);
            let offset = writer.next_offset;
            writer.file.write_all(&payload)?;
            let length = payload.len() as u64;
            writer.next_offset += length;
            Ok(AudioArtifactRef {
                file_path: Some(writer.relative_path.clone()),
                byte_offset: Some(offset),
                byte_length: Some(length),
            })
        }
    }
}

fn write_wav_i16(path: &Path, sample_rate: u32, channels: u16, samples: &[f32]) -> Result<()> {
    let payload = encode_wav_i16_bytes(sample_rate, channels, samples);
    let mut file = fs::File::create(path)?;
    file.write_all(&payload)?;
    Ok(())
}

fn encode_wav_i16_bytes(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let bytes_per_sample = 2u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (samples.len() * bytes_per_sample as usize) as u32;
    let riff_len = 36 + data_len;

    let mut payload = Vec::with_capacity(44 + data_len as usize);
    payload.extend_from_slice(b"RIFF");
    payload.extend_from_slice(&riff_len.to_le_bytes());
    payload.extend_from_slice(b"WAVE");
    payload.extend_from_slice(b"fmt ");
    payload.extend_from_slice(&16u32.to_le_bytes());
    payload.extend_from_slice(&1u16.to_le_bytes());
    payload.extend_from_slice(&channels.to_le_bytes());
    payload.extend_from_slice(&sample_rate.to_le_bytes());
    payload.extend_from_slice(&byte_rate.to_le_bytes());
    payload.extend_from_slice(&block_align.to_le_bytes());
    payload.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes());
    payload.extend_from_slice(b"data");
    payload.extend_from_slice(&data_len.to_le_bytes());

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        payload.extend_from_slice(&pcm.to_le_bytes());
    }
    payload
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

fn render_validation_video(
    output_root: &Path,
    validation_root: &Path,
    manifest: &DerivedEpisodeManifest,
    audio_track: Option<&ContinuousAudioTrack>,
    camera: VisualSensorKind,
    video_path: &Path,
) -> Result<()> {
    let camera_dir = match camera {
        VisualSensorKind::Front => "front",
        VisualSensorKind::Rear => "rear",
    };
    anyhow::ensure!(
        manifest
            .capture_config
            .visual_sensors
            .iter()
            .any(|sensor| sensor.kind == camera),
        "validation video requested for a camera that was not reconstructed"
    );

    let image_pattern = output_root
        .join("visual")
        .join(camera_dir)
        .join("step_%06d.ppm");
    anyhow::ensure!(
        image_pattern
            .parent()
            .map(|dir| dir.exists())
            .unwrap_or(false),
        "missing visual frames for validation video"
    );

    let temp_audio_path = if let Some(track) = audio_track {
        if !track.samples.is_empty() {
            let path = validation_root.join(&manifest.artifact_convention.validation_audio_file);
            let normalized = normalize_validation_audio_samples(&track.samples);
            write_wav_i16(&path, track.sample_rate, track.channels, &normalized)?;
            Some(path)
        } else {
            None
        }
    } else {
        None
    };

    let fps = 1.0
        / manifest
            .capture_config
            .audio_window_seconds
            .max(1.0 / 240.0);
    let mut command = Command::new("ffmpeg");
    command
        .arg("-y")
        .arg("-framerate")
        .arg(format!("{fps:.3}"))
        .arg("-i")
        .arg(image_pattern);
    if let Some(audio_path) = &temp_audio_path {
        command.arg("-i").arg(audio_path);
    }
    command
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("ultrafast")
        .arg("-crf")
        .arg("30")
        .arg("-pix_fmt")
        .arg("yuv420p");
    if temp_audio_path.is_some() {
        command
            .arg("-c:a")
            .arg("aac")
            .arg("-ac")
            .arg("2")
            .arg("-channel_layout")
            .arg("stereo")
            .arg("-shortest");
    }
    command.arg(video_path);
    let output = command
        .output()
        .with_context(|| "failed to spawn ffmpeg for validation video")?;
    if !output.status.success() {
        bail!(
            "ffmpeg failed to render validation video: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn drain_visual_capture_shutdown(app: &mut App) {
    shutdown_visual_capture(app.world_mut());
    for _ in 0..8 {
        app.update();
        if !has_pending_visual_capture_requests(app.world_mut()) {
            break;
        }
    }
}

fn has_pending_visual_capture_requests(world: &mut World) -> bool {
    if world
        .get_resource::<PendingVisualCaptures>()
        .is_some_and(|pending| !pending.keys.is_empty())
    {
        return true;
    }
    let mut query = world.query_filtered::<Entity, With<PendingVisualCaptureRequest>>();
    query.iter(world).next().is_some()
}

fn normalize_validation_audio_samples(samples: &[f32]) -> Vec<f32> {
    let peak = samples
        .iter()
        .fold(0.0_f32, |current, sample| current.max(sample.abs()));
    if peak <= f32::EPSILON {
        return samples.to_vec();
    }

    let target_peak = 0.92_f32;
    let gain = (target_peak / peak).min(1.0);
    samples.iter().map(|sample| sample * gain).collect()
}

fn validation_video_path_for_role(
    args: &CliArgs,
    validation_root: &Path,
    output_root: &Path,
    camera: VisualSensorKind,
) -> PathBuf {
    if let Some(path) = &args.validation_video_path {
        return path.clone();
    }
    if args.capture_config.visual_sensors.len() == 1 {
        validation_root.join("validation.mp4")
    } else {
        let camera = match camera {
            VisualSensorKind::Front => "front",
            VisualSensorKind::Rear => "rear",
        };
        let _ = output_root;
        validation_root.join(render_named_artifact_pattern(
            &DerivedArtifactConvention::default().validation_video_pattern,
            Some(camera),
            None,
            None,
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_path_forcefully_removes_non_empty_directory_tree() {
        let temp_root = std::env::temp_dir().join(format!(
            "dfb_tool_dataset_extract_remove_test_{}",
            std::process::id()
        ));
        let nested = temp_root.join("nested").join("deeper");
        fs::create_dir_all(&nested).unwrap();
        fs::write(temp_root.join("root.txt"), b"root").unwrap();
        fs::write(nested.join("child.txt"), b"child").unwrap();

        remove_path_forcefully(&temp_root).unwrap();

        assert!(!temp_root.exists());
    }

    #[test]
    fn normalize_validation_audio_samples_scales_down_clipping_track() {
        let normalized = normalize_validation_audio_samples(&[0.2, -1.5, 0.9]);
        let peak = normalized
            .iter()
            .fold(0.0_f32, |current, sample| current.max(sample.abs()));

        assert!(peak <= 0.92 + 1e-5);
        assert!(peak >= 0.9);
    }
}
