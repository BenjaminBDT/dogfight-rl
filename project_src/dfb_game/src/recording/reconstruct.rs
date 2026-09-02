use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Context, Result};
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

use crate::api::snapshot::collect_observation;
use crate::api::types::{
    ObservationBundle, ObservationCaptureConfig, StateObservation, SubsystemObservation,
};
use crate::api::vision::{clear_offscreen_visual_frames, offscreen_visual_frames_ready};
use crate::app::game_app::build_headless_offscreen_capture_app_with_paths;
use crate::audio::{
    AudioEventQueue, accumulate_audio_capture_step, reset_capture_audio_observation,
};
use crate::core::config::{ConfigPaths, resolve_project_root};
use crate::gameplay::combat::{
    Projectile, spawn_projectile_visual_entity, spawn_tracer_visual_entity,
};
use crate::gameplay::damage::{AircraftDamageState, AircraftSubsystem};
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::input::actions::ControlInput;
use crate::presentation::hud::ObservedAircraftRole;
use crate::presentation::tracers::TracerLifetime;
use crate::recording::{
    DerivedArtifactConvention, DerivedEpisodeManifest, InitialWorldSnapshot,
    RecordedDynamicWorldState, RecordedEpisodeManifest, RecordedStep, RecordedStepArtifacts,
    RecordedStepChunk, queue_recorded_audio_for_playback, render_named_artifact_pattern,
};
use crate::simulation::components::{AircraftRole, AircraftState, ControlAuthority, GunState};
use crate::simulation::resources::SimulationDebugState;
use crate::simulation::systems::{AircraftVisualPartsInitialized, AircraftVisualSceneRoot};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LoadedEpisode {
    pub manifest: RecordedEpisodeManifest,
    pub initial: InitialWorldSnapshot,
    pub steps: Vec<RecordedStep>,
}

#[derive(Debug)]
pub struct RecordingAccess {
    episode_root: PathBuf,
    manifest_cache: Mutex<Option<RecordedEpisodeManifest>>,
    step_chunk_cache: Mutex<Option<CachedStepChunk>>,
}

#[derive(Debug, Clone)]
struct CachedStepChunk {
    chunk_index: u32,
    steps: Vec<RecordedStep>,
}

pub struct RecordingReconstructionSession {
    access: RecordingAccess,
    manifest: RecordedEpisodeManifest,
    initial: InitialWorldSnapshot,
    app: App,
    observed_role: AircraftRole,
}

#[derive(Debug, Default, Clone, Copy, Resource)]
struct RecordedAircraftEntityCache {
    fighter1: Option<Entity>,
    fighter2: Option<Entity>,
}

impl RecordingAccess {
    pub fn new(episode_root: impl Into<PathBuf>) -> Self {
        Self {
            episode_root: episode_root.into(),
            manifest_cache: Mutex::new(None),
            step_chunk_cache: Mutex::new(None),
        }
    }

    pub fn episode_root(&self) -> &Path {
        &self.episode_root
    }

    pub fn manifest(&self) -> Result<RecordedEpisodeManifest> {
        let mut guard = self
            .manifest_cache
            .lock()
            .expect("recording manifest cache poisoned");
        if let Some(manifest) = guard.as_ref() {
            return Ok(manifest.clone());
        }
        let manifest = load_manifest(&self.episode_root)?;
        *guard = Some(manifest.clone());
        Ok(manifest)
    }

    pub fn load_episode(&self) -> Result<LoadedEpisode> {
        load_episode(&self.episode_root)
    }

    pub fn initial_snapshot(&self) -> Result<InitialWorldSnapshot> {
        let manifest = self.manifest()?;
        load_initial_snapshot(&self.episode_root, &manifest)
    }

    pub fn step(&self, index: u32) -> Result<RecordedStep> {
        let manifest = self.manifest()?;
        load_step_at_cached(&self.episode_root, &manifest, index, &self.step_chunk_cache)
    }

    pub fn steps(&self) -> Result<Vec<RecordedStep>> {
        let manifest = self.manifest()?;
        load_steps(&self.episode_root, &manifest)
    }

    pub fn step_artifacts(&self) -> Result<Vec<RecordedStepArtifacts>> {
        Ok(self.manifest()?.step_artifacts)
    }

    pub fn step_artifacts_at(&self, index: u32) -> Result<RecordedStepArtifacts> {
        self.manifest()?
            .step_artifacts
            .into_iter()
            .find(|artifacts| artifacts.index == index)
            .with_context(|| format!("missing step artifacts for step {index}"))
    }

    pub fn derived_root_for_role(&self, role: &str) -> PathBuf {
        self.episode_root.join("derived").join(role)
    }

    pub fn validation_root_for_role(&self, role: &str) -> Result<PathBuf> {
        let manifest = self.manifest()?;
        Ok(self.episode_root.join(render_named_artifact_pattern(
            &manifest.artifact_convention.validation_role_dir_pattern,
            None,
            None,
            None,
            Some(role),
        )))
    }

    pub fn available_derived_roles(&self) -> Result<Vec<String>> {
        let derived_root = self.episode_root.join("derived");
        if !derived_root.is_dir() {
            return Ok(Vec::new());
        }

        let mut roles = fs::read_dir(&derived_root)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect::<Vec<_>>();
        roles.sort();
        Ok(roles)
    }

    pub fn derived_manifest(&self, role: &str) -> Result<DerivedEpisodeManifest> {
        let path = self
            .derived_root_for_role(role)
            .join(&DerivedArtifactConvention::default().manifest_file);
        ron::from_str(
            &fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn validation_audio_path(&self, role: &str) -> Result<PathBuf> {
        let validation_root = self.validation_root_for_role(role)?;
        let manifest = self.derived_manifest(role)?;
        Ok(validation_root.join(manifest.artifact_convention.validation_audio_file))
    }

    pub fn validation_video_path(&self, role: &str, camera: &str) -> Result<PathBuf> {
        let validation_root = self.validation_root_for_role(role)?;
        let manifest = self.derived_manifest(role)?;
        Ok(validation_root.join(render_named_artifact_pattern(
            &manifest.artifact_convention.validation_video_pattern,
            Some(camera),
            None,
            None,
            None,
        )))
    }

    pub fn read_relative_bytes(&self, relative_path: impl AsRef<Path>) -> Result<Vec<u8>> {
        let path = self.episode_root.join(relative_path.as_ref());
        fs::read(&path).with_context(|| format!("failed to read {}", path.display()))
    }

    pub fn read_visual_artifact_bytes(
        &self,
        artifact: &crate::recording::VisualArtifactRef,
    ) -> Result<Vec<u8>> {
        let Some(relative_path) = &artifact.file_path else {
            return Ok(Vec::new());
        };
        let bytes = self.read_relative_bytes(relative_path)?;
        let Some(offset) = artifact.byte_offset else {
            return Ok(bytes);
        };
        let length = artifact
            .byte_length
            .with_context(|| "visual artifact has byte_offset but missing byte_length")?;
        let start = offset as usize;
        let end = start
            .checked_add(length as usize)
            .with_context(|| "visual artifact byte range overflowed usize")?;
        bytes
            .get(start..end)
            .map(|slice| slice.to_vec())
            .with_context(|| {
                format!(
                    "visual artifact byte range [{start}..{end}) is out of bounds for {} bytes",
                    bytes.len()
                )
            })
    }

    pub fn read_audio_artifact_bytes(
        &self,
        artifact: &crate::recording::AudioArtifactRef,
    ) -> Result<Vec<u8>> {
        let Some(relative_path) = &artifact.file_path else {
            return Ok(Vec::new());
        };
        let bytes = self.read_relative_bytes(relative_path)?;
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
}

impl RecordingReconstructionSession {
    pub fn new(
        episode_root: impl Into<PathBuf>,
        observed_role: AircraftRole,
        capture_config: ObservationCaptureConfig,
    ) -> Result<Self> {
        let access = RecordingAccess::new(episode_root);
        let manifest = access.manifest()?;
        let initial = access.initial_snapshot()?;

        let mut app = build_headless_offscreen_capture_app_with_paths(
            ConfigPaths {
                project_root: resolve_project_root(),
                scene_override: Some(manifest.scene_name.clone()),
                scene_override_path: None,
            },
            false,
        );
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            manifest.fixed_time_step_seconds.max(1.0 / 240.0),
        )));
        app.finish();
        app.cleanup();
        app.update();
        app.update();
        if let Some(mut current_capture_config) = app
            .world_mut()
            .get_resource_mut::<ObservationCaptureConfig>()
        {
            *current_capture_config = capture_config;
        }
        warm_up_visual_pipeline(&mut app);
        reset_capture_audio_observation(app.world_mut());

        Ok(Self {
            access,
            manifest,
            initial,
            app,
            observed_role,
        })
    }

    pub fn manifest(&self) -> &RecordedEpisodeManifest {
        &self.manifest
    }

    pub fn reconstruct_initial_observation(&mut self) -> Result<ObservationBundle> {
        let state = self.initial.state.clone();
        let dynamic = self.initial.dynamic.clone();
        let audio_semantics = self.initial.audio_semantics.clone();
        let events = state.events_since_last_step.clone();
        restore_recorded_world(self.app.world_mut(), &state, &dynamic, self.observed_role);
        self.run_capture_updates(&state, &dynamic, audio_semantics.as_ref(), &events)?;
        Ok(collect_observation(self.app.world_mut()))
    }

    pub fn reconstruct_step_observation(&mut self, index: u32) -> Result<ObservationBundle> {
        let step = self.access.step(index)?;
        restore_recorded_world(
            self.app.world_mut(),
            &step.state,
            &step.dynamic,
            self.observed_role,
        );
        self.run_capture_updates(
            &step.state,
            &step.dynamic,
            step.audio_semantics.as_ref(),
            &step.state.events_since_last_step,
        )?;
        Ok(collect_observation(self.app.world_mut()))
    }

    fn run_capture_updates(
        &mut self,
        state: &StateObservation,
        dynamic: &RecordedDynamicWorldState,
        audio_semantics: Option<&crate::recording::RecordedAudioFrame>,
        events: &[crate::api::types::EnvironmentEvent],
    ) -> Result<()> {
        let _ = state;
        let _ = dynamic;
        self.app
            .world_mut()
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
                self.manifest.fixed_time_step_seconds.max(1.0 / 240.0),
            )));
        let original_enable_audio = self
            .app
            .world()
            .get_resource::<ObservationCaptureConfig>()
            .map(|config| config.enable_audio)
            .unwrap_or(false);
        let original_enable_visual = self
            .app
            .world()
            .get_resource::<ObservationCaptureConfig>()
            .map(|config| config.enable_visual)
            .unwrap_or(false);
        if original_enable_audio
            && let Some(mut capture_config) = self
                .app
                .world_mut()
                .get_resource_mut::<ObservationCaptureConfig>()
        {
            capture_config.enable_audio = false;
        }
        if original_enable_visual {
            flush_offscreen_visual_capture(&mut self.app)?;
        } else {
            self.app.update();
        }
        if original_enable_audio
            && let Some(mut capture_config) = self
                .app
                .world_mut()
                .get_resource_mut::<ObservationCaptureConfig>()
        {
            capture_config.enable_audio = true;
        }
        if let Some(mut queue) = self.app.world_mut().get_resource_mut::<AudioEventQueue>() {
            queue_recorded_audio_for_playback(&mut queue, audio_semantics, events);
        }
        accumulate_audio_capture_step(self.app.world_mut())?;
        Ok(())
    }
}

fn flush_offscreen_visual_capture(app: &mut App) -> Result<()> {
    const MAX_CAPTURE_UPDATES: usize = 8;
    clear_offscreen_visual_frames(app.world_mut());
    for _ in 0..MAX_CAPTURE_UPDATES {
        app.update();
        if offscreen_visual_frames_ready(app.world()) {
            return Ok(());
        }
    }
    anyhow::bail!(
        "offscreen visual capture did not become ready within {MAX_CAPTURE_UPDATES} updates"
    );
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
            break;
        }
        app.update();
    }
}

pub fn load_episode(episode_root: &Path) -> Result<LoadedEpisode> {
    let manifest = load_manifest(episode_root)?;
    let initial = load_initial_snapshot(episode_root, &manifest)?;
    let steps = load_steps(episode_root, &manifest)?;
    Ok(LoadedEpisode {
        manifest,
        initial,
        steps,
    })
}

pub fn load_steps(
    episode_root: &Path,
    manifest: &RecordedEpisodeManifest,
) -> Result<Vec<RecordedStep>> {
    let mut steps = Vec::with_capacity(manifest.total_steps as usize);
    for chunk in &manifest.step_chunks {
        let path = episode_root.join(&chunk.file_path);
        let bundle: RecordedStepChunk = ron::from_str(
            &fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        steps.extend(bundle.steps);
    }
    Ok(steps)
}

fn load_manifest(episode_root: &Path) -> Result<RecordedEpisodeManifest> {
    let path = episode_root.join("episode.ron");
    ron::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn load_initial_snapshot(
    episode_root: &Path,
    manifest: &RecordedEpisodeManifest,
) -> Result<InitialWorldSnapshot> {
    let path = episode_root.join(&manifest.artifact_convention.initial_state_path);
    ron::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))
}

fn load_step_at_cached(
    episode_root: &Path,
    manifest: &RecordedEpisodeManifest,
    step_index: u32,
    cache: &Mutex<Option<CachedStepChunk>>,
) -> Result<RecordedStep> {
    let chunk = manifest
        .step_chunks
        .iter()
        .find(|chunk| {
            step_index >= chunk.start_step_index
                && step_index < chunk.start_step_index + chunk.step_count
        })
        .with_context(|| format!("missing step chunk for step {step_index}"))?;

    {
        let guard = cache.lock().expect("recording step chunk cache poisoned");
        if let Some(cached) = guard
            .as_ref()
            .filter(|cached| cached.chunk_index == chunk.chunk_index)
        {
            return cached
                .steps
                .iter()
                .find(|step| step.index == step_index)
                .cloned()
                .with_context(|| {
                    format!(
                        "missing step {step_index} inside cached chunk {}",
                        chunk.chunk_index
                    )
                });
        }
    }

    let path = episode_root.join(&chunk.file_path);
    let bundle: RecordedStepChunk = ron::from_str(
        &fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let step = bundle
        .steps
        .iter()
        .find(|step| step.index == step_index)
        .cloned()
        .with_context(|| format!("missing step {step_index} inside {}", path.display()))?;
    let mut guard = cache.lock().expect("recording step chunk cache poisoned");
    *guard = Some(CachedStepChunk {
        chunk_index: chunk.chunk_index,
        steps: bundle.steps,
    });
    Ok(step)
}

pub fn restore_recorded_world(
    world: &mut World,
    snapshot: &StateObservation,
    dynamic: &RecordedDynamicWorldState,
    observed_role: AircraftRole,
) {
    world.insert_resource(ObservedAircraftRole(observed_role));
    if let Some(mut debug) = world.get_resource_mut::<SimulationDebugState>() {
        debug.tick_count = snapshot.tick;
    }
    if let Some(mut clock) = world.get_resource_mut::<MatchClock>() {
        clock.elapsed_seconds = snapshot.sim_time_seconds;
    }
    if let Some(mut next_phase) = world.get_resource_mut::<NextState<MatchPhase>>() {
        next_phase.set(parse_match_phase(&snapshot.match_phase));
    }

    let entity_cache = resolve_recorded_aircraft_entity_cache(world);
    let fighter1 = snapshot
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter1");
    let fighter2 = snapshot
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role == "fighter2");
    if let (Some(entity), Some(recorded)) = (entity_cache.fighter1, fighter1) {
        apply_recorded_aircraft_state(world, entity, recorded);
    }
    if let (Some(entity), Some(recorded)) = (entity_cache.fighter2, fighter2) {
        apply_recorded_aircraft_state(world, entity, recorded);
    }

    respawn_dynamic_world(world, dynamic);
}

fn resolve_recorded_aircraft_entity_cache(world: &mut World) -> RecordedAircraftEntityCache {
    let mut cache = world
        .remove_resource::<RecordedAircraftEntityCache>()
        .unwrap_or_default();
    let fighter1_valid = cache
        .fighter1
        .is_some_and(|entity| world.entities().contains(entity));
    let fighter2_valid = cache
        .fighter2
        .is_some_and(|entity| world.entities().contains(entity));
    if !fighter1_valid || !fighter2_valid {
        cache = RecordedAircraftEntityCache::default();
        let mut query = world.query::<(Entity, &AircraftRole)>();
        for (entity, role) in query.iter(world) {
            match role {
                AircraftRole::Fighter1 => cache.fighter1 = Some(entity),
                AircraftRole::Fighter2 => cache.fighter2 = Some(entity),
            }
        }
    }
    world.insert_resource(cache);
    cache
}

fn apply_recorded_aircraft_state(
    world: &mut World,
    entity: Entity,
    recorded: &crate::api::types::AircraftObservation,
) {
    let mut entity_mut = world.entity_mut(entity);

    let position = Vec3::from_array(recorded.position);
    let orientation = Quat::from_array(recorded.orientation_quat);
    let velocity = Vec3::from_array(recorded.linear_velocity);
    let angular_rates_deg = Vec3::from_array(recorded.angular_velocity_deg);
    let forward = (orientation * Vec3::Z).normalize_or_zero();

    {
        let mut state = entity_mut
            .get_mut::<AircraftState>()
            .expect("recorded aircraft entity missing AircraftState");
        if state.position != position {
            state.position = position;
        }
        if state.orientation != orientation {
            state.orientation = orientation;
        }
        if state.velocity != velocity {
            state.velocity = velocity;
        }
        if state.angular_rates_deg != angular_rates_deg {
            state.angular_rates_deg = angular_rates_deg;
        }
        if state.forward != forward {
            state.forward = forward;
        }
        if state.throttle != recorded.throttle {
            state.throttle = recorded.throttle;
        }
        if state.hit_points != recorded.hit_points {
            state.hit_points = recorded.hit_points;
        }
        if state.stall_factor != recorded.stall_factor {
            state.stall_factor = recorded.stall_factor;
        }
        if state.out_of_bounds_seconds != recorded.out_of_bounds_seconds {
            state.out_of_bounds_seconds = recorded.out_of_bounds_seconds;
        }
        if state.ceiling_recovery_seconds != recorded.ceiling_recovery_seconds {
            state.ceiling_recovery_seconds = recorded.ceiling_recovery_seconds;
        }
        if state.is_destroyed != recorded.destroyed {
            state.is_destroyed = recorded.destroyed;
        }
    }

    {
        let mut transform = entity_mut
            .get_mut::<Transform>()
            .expect("recorded aircraft entity missing Transform");
        if transform.translation != position {
            transform.translation = position;
        }
        if transform.rotation != orientation {
            transform.rotation = orientation;
        }
    }

    {
        let mut gun = entity_mut
            .get_mut::<GunState>()
            .expect("recorded aircraft entity missing GunState");
        if gun.heat != recorded.gun_heat {
            gun.heat = recorded.gun_heat;
        }
        if gun.overheated != recorded.gun_overheated {
            gun.overheated = recorded.gun_overheated;
        }
    }

    {
        let mut input = entity_mut
            .get_mut::<ControlInput>()
            .expect("recorded aircraft entity missing ControlInput");
        if input.throttle_delta != 0.0 {
            input.throttle_delta = 0.0;
        }
        if input.brake {
            input.brake = false;
        }
        if input.pitch != 0.0 {
            input.pitch = 0.0;
        }
        if input.roll != 0.0 {
            input.roll = 0.0;
        }
        if input.yaw != 0.0 {
            input.yaw = 0.0;
        }
        if input.fire_gun {
            input.fire_gun = false;
        }
        if input.repair {
            input.repair = false;
        }
    }

    {
        let mut damage = entity_mut
            .get_mut::<AircraftDamageState>()
            .expect("recorded aircraft entity missing AircraftDamageState");
        if damage.is_repairing != recorded.repairing {
            damage.is_repairing = recorded.repairing;
        }
        if damage.repair_elapsed_seconds != recorded.repair_elapsed_seconds {
            damage.repair_elapsed_seconds = recorded.repair_elapsed_seconds;
        }
        apply_subsystem_snapshot(&mut damage, &recorded.subsystems);
    }

    {
        let mut authority = entity_mut
            .get_mut::<ControlAuthority>()
            .expect("recorded aircraft entity missing ControlAuthority");
        if *authority != ControlAuthority::Replay {
            *authority = ControlAuthority::Replay;
        }
    }
}

fn respawn_dynamic_world(world: &mut World, dynamic: &RecordedDynamicWorldState) {
    let has_visual_assets = world.contains_resource::<Assets<Mesh>>()
        && world.contains_resource::<Assets<StandardMaterial>>();
    sync_projectiles(world, dynamic, has_visual_assets);
    sync_tracers(world, dynamic, has_visual_assets);
}

fn sync_projectiles(
    world: &mut World,
    dynamic: &RecordedDynamicWorldState,
    has_visual_assets: bool,
) {
    let existing = {
        let mut query = world.query::<(Entity, &Projectile)>();
        query
            .iter(world)
            .map(|(entity, projectile)| (projectile.id, entity))
            .collect::<std::collections::HashMap<_, _>>()
    };
    let desired_ids = dynamic
        .projectiles
        .iter()
        .map(|projectile| projectile.id)
        .collect::<std::collections::HashSet<_>>();

    for projectile in &dynamic.projectiles {
        if let Some(entity) = existing.get(&projectile.id).copied() {
            update_projectile_entity(world, entity, projectile);
        } else {
            spawn_recorded_projectile(world, projectile, has_visual_assets);
        }
    }

    for (id, entity) in existing {
        if !desired_ids.contains(&id) {
            let _ = world.despawn(entity);
        }
    }
}

fn sync_tracers(world: &mut World, dynamic: &RecordedDynamicWorldState, has_visual_assets: bool) {
    let existing = {
        let mut query = world.query::<(Entity, &TracerLifetime)>();
        query
            .iter(world)
            .map(|(entity, _)| entity)
            .collect::<Vec<_>>()
    };

    for (entity, tracer) in existing.iter().copied().zip(dynamic.tracers.iter()) {
        let mut entity_mut = world.entity_mut(entity);
        if let Some(mut transform) = entity_mut.get_mut::<Transform>() {
            let position = Vec3::from_array(tracer.position);
            if transform.translation != position {
                transform.translation = position;
            }
        }
        let mut lifetime = entity_mut
            .get_mut::<TracerLifetime>()
            .expect("recorded tracer entity missing TracerLifetime");
        if lifetime.remaining_seconds != tracer.remaining_seconds {
            lifetime.remaining_seconds = tracer.remaining_seconds;
        }
    }

    if dynamic.tracers.len() > existing.len() {
        for tracer in &dynamic.tracers[existing.len()..] {
            spawn_recorded_tracer(world, tracer, has_visual_assets);
        }
    } else if existing.len() > dynamic.tracers.len() {
        for entity in &existing[dynamic.tracers.len()..] {
            let _ = world.despawn(*entity);
        }
    }
}

fn update_projectile_entity(
    world: &mut World,
    entity: Entity,
    projectile: &crate::recording::RecordedProjectileState,
) {
    let shooter_role = parse_recorded_projectile_role(&projectile.shooter_role);
    let velocity = Vec3::from_array(projectile.velocity);
    let rotation = recorded_projectile_rotation(velocity);
    let position = Vec3::from_array(projectile.position);

    let mut entity_mut = world.entity_mut(entity);
    {
        let mut projectile_component = entity_mut
            .get_mut::<Projectile>()
            .expect("recorded projectile entity missing Projectile");
        if projectile_component.shooter_role != shooter_role {
            projectile_component.shooter_role = shooter_role;
        }
        if projectile_component.velocity != velocity {
            projectile_component.velocity = velocity;
        }
        if projectile_component.damage != projectile.damage {
            projectile_component.damage = projectile.damage;
        }
        if projectile_component.remaining_distance != projectile.remaining_distance {
            projectile_component.remaining_distance = projectile.remaining_distance;
        }
        if projectile_component.hit_radius != projectile.hit_radius {
            projectile_component.hit_radius = projectile.hit_radius;
        }
        projectile_component.flyby_emitted = false;
    }
    if let Some(mut transform) = entity_mut.get_mut::<Transform>() {
        if transform.translation != position {
            transform.translation = position;
        }
        if transform.rotation != rotation {
            transform.rotation = rotation;
        }
    }
}

fn spawn_recorded_projectile(
    world: &mut World,
    projectile: &crate::recording::RecordedProjectileState,
    has_visual_assets: bool,
) {
    let shooter_role = parse_recorded_projectile_role(&projectile.shooter_role);
    let velocity = Vec3::from_array(projectile.velocity);
    let rotation = recorded_projectile_rotation(velocity);
    let projectile_component = Projectile {
        id: projectile.id,
        shooter_role,
        velocity,
        damage: projectile.damage,
        remaining_distance: projectile.remaining_distance,
        hit_radius: projectile.hit_radius,
        flyby_emitted: false,
        lag_compensation_ticks: 0,
    };
    let transform =
        Transform::from_translation(Vec3::from_array(projectile.position)).with_rotation(rotation);

    if has_visual_assets {
        world.resource_scope(|world, mut meshes: Mut<Assets<Mesh>>| {
            world.resource_scope(|world, mut materials: Mut<Assets<StandardMaterial>>| {
                let mut commands = world.commands();
                spawn_projectile_visual_entity(
                    &mut commands,
                    Some(&mut meshes),
                    Some(&mut materials),
                    projectile_component,
                    transform,
                );
            });
        });
    } else {
        let mut commands = world.commands();
        spawn_projectile_visual_entity(&mut commands, None, None, projectile_component, transform);
    }
}

fn spawn_recorded_tracer(
    world: &mut World,
    tracer: &crate::recording::RecordedTracerState,
    has_visual_assets: bool,
) {
    let position = Vec3::from_array(tracer.position);
    if has_visual_assets {
        world.resource_scope(|world, mut meshes: Mut<Assets<Mesh>>| {
            world.resource_scope(|world, mut materials: Mut<Assets<StandardMaterial>>| {
                let mut commands = world.commands();
                spawn_tracer_visual_entity(
                    &mut commands,
                    Some(&mut meshes),
                    Some(&mut materials),
                    position,
                    tracer.remaining_seconds,
                );
            });
        });
    } else {
        let mut commands = world.commands();
        spawn_tracer_visual_entity(
            &mut commands,
            None,
            None,
            position,
            tracer.remaining_seconds,
        );
    }
}

fn parse_recorded_projectile_role(role: &str) -> AircraftRole {
    match role {
        "Fighter2" | "fighter2" => AircraftRole::Fighter2,
        _ => AircraftRole::Fighter1,
    }
}

fn recorded_projectile_rotation(velocity: Vec3) -> Quat {
    if velocity.length_squared() > f32::EPSILON {
        Quat::from_rotation_arc(Vec3::Z, velocity.normalize())
    } else {
        Quat::IDENTITY
    }
}

fn apply_subsystem_snapshot(damage: &mut AircraftDamageState, subsystems: &[SubsystemObservation]) {
    for subsystem in subsystems {
        let Some(kind) = parse_subsystem_name(&subsystem.name) else {
            continue;
        };
        let target = damage.subsystem_mut(kind);
        target.current = subsystem.hit_points.clamp(0.0, target.max);
    }
}

fn parse_subsystem_name(name: &str) -> Option<AircraftSubsystem> {
    match name {
        "LeftWing" => Some(AircraftSubsystem::LeftWing),
        "RightWing" => Some(AircraftSubsystem::RightWing),
        "PitchTail" => Some(AircraftSubsystem::PitchTail),
        "YawTail" => Some(AircraftSubsystem::YawTail),
        "Engine" => Some(AircraftSubsystem::Engine),
        _ => None,
    }
}

fn parse_match_phase(name: &str) -> MatchPhase {
    match name {
        "Finished" => MatchPhase::Finished,
        _ => MatchPhase::Running,
    }
}
