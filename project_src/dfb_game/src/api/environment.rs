use anyhow::Result;
use bevy::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::ai::basic_fighter::{BuiltInAiProfile, BuiltInAiProfileOverrides, query_teacher_action};
use crate::api::commands::{ExternalCommandBuffer, TargetedEnvironmentAction};
use crate::api::events::{AircraftEventTracker, PendingEnvironmentEvents};
use crate::api::snapshot::collect_observation;
use crate::api::types::{
    EnvironmentAction, EnvironmentAgentControlConfig, EnvironmentAgentMode,
    EnvironmentEpisodeStatus, EnvironmentRecordingStatus, EnvironmentResetOptions,
    ObservationBundle, ObservationCaptureConfig, StepInfo, StepResult,
};
use crate::api::vision::shutdown_visual_capture;
use crate::app::game_app::{build_headless_app_with_paths, build_headless_capture_app_with_paths};
use crate::audio::{AudioEventQueue, AudioObservationState, FrameAudioState};
use crate::core::config::ConfigPaths;
use crate::gameplay::combat::{CombatPresentationQueue, CombatState, Projectile};
use crate::gameplay::damage::AircraftDamageState;
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::gameplay::reset::reset_match_world;
use crate::input::actions::ControlInput;
use crate::presentation::hud::{ControlsGuideState, DamageIndicatorQueue, DamageIndicatorState};
use crate::presentation::tracers::TracerLifetime;
use crate::recording::{
    ActionRecordingState, request_authoritative_recording_start, request_manual_recording_stop,
};
use crate::simulation::components::{
    AircraftPerformance, AircraftRole, AircraftState, ControlAuthority, GunState, SpawnTransform,
};

pub const DEFAULT_ENVIRONMENT_SEED: u64 = 0xD06F_1A77;

#[derive(Debug, Clone, Copy, Resource)]
pub struct EnvironmentSeed {
    pub requested: Option<u64>,
    pub effective: u64,
}

impl EnvironmentSeed {
    pub fn from_request(seed: Option<u64>) -> Self {
        Self {
            requested: seed,
            effective: seed.unwrap_or(DEFAULT_ENVIRONMENT_SEED),
        }
    }
}

#[derive(Resource)]
pub struct DeterministicRng(pub ChaCha8Rng);

#[derive(Debug, Resource, Default)]
pub struct EnvironmentFacade {
    pub latest_observation: ObservationBundle,
    pub pending_commands: ExternalCommandBuffer,
}

pub fn sync_environment_facade(
    observation: Res<ObservationBundle>,
    pending_commands: Res<ExternalCommandBuffer>,
    mut facade: ResMut<EnvironmentFacade>,
) {
    facade.latest_observation = observation.clone();
    facade.pending_commands = pending_commands.clone();
}

pub struct EnvironmentInstance {
    config_paths: ConfigPaths,
    app: App,
    ticks_per_step: u32,
    visual_enabled: bool,
}

impl EnvironmentInstance {
    pub fn new_headless(config_paths: ConfigPaths, options: EnvironmentResetOptions) -> Self {
        let config_paths = merged_config_paths(
            &config_paths,
            options.scene_name.as_deref(),
            options.scene_path.as_deref(),
        );
        let mut instance = Self {
            config_paths: config_paths.clone(),
            app: build_app_for_options(config_paths, &options),
            ticks_per_step: options.ticks_per_step.max(1),
            visual_enabled: options.enable_visual && !cfg!(test),
        };
        apply_capture_options(instance.app.world_mut(), &options);
        instance.bootstrap();
        apply_agent_control_config(instance.app.world_mut(), &options.agent_control);
        instance
    }

    pub fn latest_observation(&mut self) -> ObservationBundle {
        let cached = self
            .app
            .world()
            .get_resource::<EnvironmentFacade>()
            .map(|facade| facade.latest_observation.clone())
            .unwrap_or_default();
        if !cached.state.scene_name.is_empty() || !cached.state.aircraft.is_empty() {
            cached
        } else {
            collect_observation(self.app.world_mut())
        }
    }

    pub fn reset(&mut self, options: &EnvironmentResetOptions) -> ObservationBundle {
        let next_config_paths = merged_config_paths(
            &self.config_paths,
            options.scene_name.as_deref(),
            options.scene_path.as_deref(),
        );
        let next_visual_enabled = options.enable_visual && !cfg!(test);
        let can_reset_in_place =
            self.visual_enabled == next_visual_enabled && self.config_paths == next_config_paths;

        self.config_paths = next_config_paths;
        self.ticks_per_step = options.ticks_per_step.max(1);

        if can_reset_in_place {
            soft_reset_world(self.app.world_mut(), options);
        } else {
            self.app = build_app_for_options(self.config_paths.clone(), options);
            self.visual_enabled = next_visual_enabled;
            apply_capture_options(self.app.world_mut(), options);
            self.bootstrap();
        }

        apply_agent_control_config(self.app.world_mut(), &options.agent_control);

        self.latest_observation()
    }

    pub fn step(&mut self, action: EnvironmentAction) -> StepResult {
        self.step_targeted([TargetedEnvironmentAction {
            role: AircraftRole::Fighter1,
            action,
        }])
    }

    pub fn step_targeted<I>(&mut self, actions: I) -> StepResult
    where
        I: IntoIterator<Item = TargetedEnvironmentAction>,
    {
        if let Some(mut commands) = self
            .app
            .world_mut()
            .get_resource_mut::<ExternalCommandBuffer>()
        {
            commands.targeted_actions.extend(actions);
        }

        for _ in 0..self.ticks_per_step {
            self.app.update();
        }

        build_step_result(self.latest_observation())
    }

    pub fn step_self_play(
        &mut self,
        fighter1_action: EnvironmentAction,
        fighter2_action: EnvironmentAction,
    ) -> StepResult {
        self.step_targeted([
            TargetedEnvironmentAction {
                role: AircraftRole::Fighter1,
                action: fighter1_action,
            },
            TargetedEnvironmentAction {
                role: AircraftRole::Fighter2,
                action: fighter2_action,
            },
        ])
    }

    pub fn set_control_authority(&mut self, role: AircraftRole, authority: ControlAuthority) {
        let mut query = self
            .app
            .world_mut()
            .query::<(&AircraftRole, &mut ControlAuthority)>();
        for (aircraft_role, mut control_authority) in query.iter_mut(self.app.world_mut()) {
            if *aircraft_role == role {
                *control_authority = authority;
            }
        }
    }

    pub fn teacher_action(&mut self, role: AircraftRole) -> Result<EnvironmentAction> {
        query_teacher_action(self.app.world_mut(), role)
            .ok_or_else(|| anyhow::anyhow!("teacher action unavailable for {role:?}"))
    }

    pub fn episode_status(&mut self) -> EnvironmentEpisodeStatus {
        build_episode_status(&self.latest_observation())
    }

    pub fn recording_status(&self) -> EnvironmentRecordingStatus {
        self.app
            .world()
            .get_resource::<ActionRecordingState>()
            .map(|recording| EnvironmentRecordingStatus {
                active: recording.active,
                pending_start: recording.pending_start,
                pending_stop: recording.pending_stop,
                active_step_count: recording.active_step_count,
                last_saved_path: recording.last_saved_path.clone(),
                last_saved_frame_count: recording.last_saved_frame_count,
            })
            .unwrap_or_default()
    }

    pub fn start_recording(
        &mut self,
        capture_config: Option<ObservationCaptureConfig>,
    ) -> Result<()> {
        request_authoritative_recording_start(self.app.world_mut(), capture_config)
    }

    pub fn stop_recording(&mut self) -> bool {
        request_manual_recording_stop(self.app.world_mut())
    }

    fn bootstrap(&mut self) {
        self.app.finish();
        self.app.cleanup();
        self.app.update();
        self.app.update();
    }

    pub fn shutdown(&mut self) {
        shutdown_visual_capture(self.app.world_mut());
        self.app.update();
    }
}

fn build_episode_status(observation: &ObservationBundle) -> EnvironmentEpisodeStatus {
    let fighter1_destroyed = aircraft_destroyed(observation, "fighter1");
    let fighter2_destroyed = aircraft_destroyed(observation, "fighter2");
    let terminated = fighter1_destroyed || fighter2_destroyed;
    let truncated = observation.state.match_phase == "Finished" && !terminated;
    let winner = match (fighter1_destroyed, fighter2_destroyed) {
        (false, true) => Some("fighter1".to_string()),
        (true, false) => Some("fighter2".to_string()),
        _ => None,
    };

    EnvironmentEpisodeStatus {
        tick: observation.state.tick,
        sim_time_seconds: observation.state.sim_time_seconds,
        match_phase: observation.state.match_phase.clone(),
        scene_name: observation.state.scene_name.clone(),
        terminated,
        truncated,
        winner,
    }
}

fn build_step_result(observation: ObservationBundle) -> StepResult {
    let status = build_episode_status(&observation);

    StepResult {
        info: StepInfo {
            tick: observation.state.tick,
            sim_time_seconds: observation.state.sim_time_seconds,
            winner: status.winner.clone(),
            events: observation.state.events_since_last_step.clone(),
        },
        observation,
        reward: None,
        terminated: status.terminated,
        truncated: status.truncated,
    }
}

fn aircraft_destroyed(observation: &ObservationBundle, role: &str) -> bool {
    observation
        .state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role.eq_ignore_ascii_case(role))
        .map(|aircraft| aircraft.destroyed)
        .unwrap_or(false)
}

fn build_app_for_options(config_paths: ConfigPaths, options: &EnvironmentResetOptions) -> App {
    if options.enable_visual && !cfg!(test) {
        let include_hud = options
            .visual_sensors
            .iter()
            .any(|sensor| sensor.include_hud);
        build_headless_capture_app_with_paths(config_paths, include_hud)
    } else {
        build_headless_app_with_paths(config_paths)
    }
}

fn merged_config_paths(
    base: &ConfigPaths,
    scene_override: Option<&str>,
    scene_path_override: Option<&str>,
) -> ConfigPaths {
    ConfigPaths {
        project_root: base.project_root.clone(),
        scene_override: scene_override
            .map(ToOwned::to_owned)
            .or_else(|| base.scene_override.clone()),
        scene_override_path: scene_path_override
            .map(std::path::PathBuf::from)
            .or_else(|| base.scene_override_path.clone()),
    }
}

fn apply_capture_options(world: &mut World, options: &EnvironmentResetOptions) {
    if let Some(mut capture_config) = world.get_resource_mut::<ObservationCaptureConfig>() {
        *capture_config = ObservationCaptureConfig::from(options);
    }
    let seed = EnvironmentSeed::from_request(options.seed);
    world.insert_resource(seed);
    world.insert_resource(DeterministicRng(ChaCha8Rng::seed_from_u64(seed.effective)));
}

fn apply_agent_control_config(world: &mut World, config: &EnvironmentAgentControlConfig) {
    world.insert_resource(BuiltInAiProfileOverrides {
        fighter1: built_in_profile_for_mode(config.fighter1),
        fighter2: built_in_profile_for_mode(config.fighter2),
    });
    let mut query = world.query::<(&AircraftRole, &mut ControlAuthority)>();
    for (role, mut authority) in query.iter_mut(world) {
        *authority = match role {
            AircraftRole::Fighter1 => control_authority_for_mode(config.fighter1),
            AircraftRole::Fighter2 => control_authority_for_mode(config.fighter2),
        };
    }
}

fn control_authority_for_mode(mode: EnvironmentAgentMode) -> ControlAuthority {
    match mode {
        EnvironmentAgentMode::External | EnvironmentAgentMode::Model => {
            ControlAuthority::ExternalAgent
        }
        EnvironmentAgentMode::BuiltInAi
        | EnvironmentAgentMode::BuiltInAiPrecise
        | EnvironmentAgentMode::BuiltInAiImperfect
        | EnvironmentAgentMode::BuiltInAiTeacher
        | EnvironmentAgentMode::BuiltInAiPassiveBounce => ControlAuthority::BuiltInAi,
    }
}

fn built_in_profile_for_mode(mode: EnvironmentAgentMode) -> Option<BuiltInAiProfile> {
    match mode {
        EnvironmentAgentMode::BuiltInAi => Some(BuiltInAiProfile::Imperfect),
        EnvironmentAgentMode::BuiltInAiPrecise => Some(BuiltInAiProfile::PreciseFollow),
        EnvironmentAgentMode::BuiltInAiImperfect => Some(BuiltInAiProfile::Imperfect),
        EnvironmentAgentMode::BuiltInAiTeacher => Some(BuiltInAiProfile::Teacher),
        EnvironmentAgentMode::BuiltInAiPassiveBounce => Some(BuiltInAiProfile::PassiveBounce),
        EnvironmentAgentMode::External | EnvironmentAgentMode::Model => None,
    }
}

fn soft_reset_world(world: &mut World, options: &EnvironmentResetOptions) {
    apply_capture_options(world, options);

    if let Some(mut pending) = world.get_resource_mut::<PendingEnvironmentEvents>() {
        pending.events.clear();
    }
    if let Some(mut tracker) = world.get_resource_mut::<AircraftEventTracker>() {
        *tracker = AircraftEventTracker::default();
    }
    if let Some(mut commands) = world.get_resource_mut::<ExternalCommandBuffer>() {
        *commands = ExternalCommandBuffer::default();
    }
    if let Some(mut facade) = world.get_resource_mut::<EnvironmentFacade>() {
        *facade = EnvironmentFacade::default();
    }
    shutdown_visual_capture(world);
    if let Some(mut frame_audio) = world.get_resource_mut::<FrameAudioState>() {
        *frame_audio = FrameAudioState::default();
    }
    if let Some(mut audio_events) = world.get_resource_mut::<AudioEventQueue>() {
        audio_events.events.clear();
    }
    if let Some(mut observation_audio) = world.get_resource_mut::<AudioObservationState>() {
        *observation_audio = AudioObservationState::default();
    }
    if let Some(mut combat_state) = world.get_resource_mut::<CombatState>() {
        *combat_state = CombatState::default();
    }
    if let Some(mut presentation_queue) = world.get_resource_mut::<CombatPresentationQueue>() {
        *presentation_queue = CombatPresentationQueue::default();
    }
    if let Some(mut guide_state) = world.get_resource_mut::<ControlsGuideState>() {
        guide_state.visible = false;
    }
    if let Some(mut indicators) = world.get_resource_mut::<DamageIndicatorQueue>() {
        indicators.events.clear();
    }
    if let Some(mut indicator_state) = world.get_resource_mut::<DamageIndicatorState>() {
        *indicator_state = DamageIndicatorState::default();
    }

    let projectile_entities: Vec<Entity> = {
        let mut query = world.query_filtered::<Entity, With<Projectile>>();
        query.iter(world).collect()
    };
    let tracer_entities: Vec<Entity> = {
        let mut query = world.query_filtered::<Entity, With<TracerLifetime>>();
        query.iter(world).collect()
    };
    for entity in projectile_entities.into_iter().chain(tracer_entities) {
        let _ = world.despawn(entity);
    }

    let Some(config) = world
        .get_resource::<crate::core::config::RepositoryConfig>()
        .cloned()
    else {
        return;
    };

    let mut system_state: bevy::ecs::system::SystemState<(
        ResMut<MatchClock>,
        ResMut<CombatState>,
        Query<(
            &AircraftRole,
            &mut SpawnTransform,
            &mut AircraftPerformance,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut ControlInput,
            &mut GunState,
            &mut Transform,
        )>,
        Option<ResMut<NextState<MatchPhase>>>,
    )> = bevy::ecs::system::SystemState::new(world);

    let (mut clock, mut combat_state, mut aircraft_query, mut next_phase) =
        system_state.get_mut(world);
    reset_match_world(&config, &mut clock, &mut combat_state, &mut aircraft_query);
    if let Some(next_phase) = next_phase.as_deref_mut() {
        next_phase.set(MatchPhase::Running);
    }
    system_state.apply(world);

    world.flush();
    world
        .resource_mut::<ObservationBundle>()
        .clone_from(&ObservationBundle::default());
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::api::types::EnvironmentAgentControlConfig;
    use crate::recording::RECORDING_SCHEMA_VERSION;
    use crate::recording::reconstruct::RecordingAccess;
    use crate::simulation::components::{AircraftRole, AircraftState, ControlAuthority};

    #[test]
    fn reset_rebuilds_environment_with_scene_override() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions::default(),
        );

        let observation = env.reset(&EnvironmentResetOptions {
            scene_name: Some("open".to_string()),
            ..EnvironmentResetOptions::default()
        });

        assert_eq!(observation.state.scene_name, "open");
        assert!(
            observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter1")
        );
        assert!(
            observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter2")
        );
        assert_eq!(observation.state.seed, DEFAULT_ENVIRONMENT_SEED);
    }

    #[test]
    fn reset_rebuilds_environment_with_scene_path_override() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions::default(),
        );

        let scene_path = PathBuf::from("../../config/dfb_game/scenes/open_head_on_200m.ron");
        let observation = env.reset(&EnvironmentResetOptions {
            scene_path: Some(scene_path.display().to_string()),
            ..EnvironmentResetOptions::default()
        });

        assert_eq!(observation.state.scene_name, "open_head_on_200m");
        assert!(
            observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter1")
        );
        assert!(
            observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter2")
        );
    }

    #[test]
    #[should_panic(expected = "failed to load repository configuration")]
    fn reset_rejects_missing_scene_override() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions::default(),
        );

        let _ = env.reset(&EnvironmentResetOptions {
            scene_name: Some("definitely_missing_scene_xyz".to_string()),
            ..EnvironmentResetOptions::default()
        });
    }

    #[test]
    fn step_advances_exact_number_of_ticks() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                ticks_per_step: 3,
                ..EnvironmentResetOptions::default()
            },
        );

        let before = env.reset(&EnvironmentResetOptions {
            ticks_per_step: 3,
            ..EnvironmentResetOptions::default()
        });
        let result = env.step(EnvironmentAction::default());

        assert_eq!(result.info.tick, before.state.tick + 3);
        assert_eq!(result.observation.state.tick, before.state.tick + 3);
    }

    #[test]
    fn teacher_action_query_is_deterministic_and_read_only() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions::default(),
        );
        let before = env.reset(&EnvironmentResetOptions::default());

        let first = env
            .teacher_action(AircraftRole::Fighter1)
            .expect("teacher action");
        let second = env
            .teacher_action(AircraftRole::Fighter1)
            .expect("teacher action");
        let after = env.latest_observation();

        assert_eq!(first, second);
        assert_eq!(before.state.tick, after.state.tick);
        assert_eq!(before.state.sim_time_seconds, after.state.sim_time_seconds);
        assert_eq!(before.state.aircraft.len(), after.state.aircraft.len());
        for value in [first.throttle, first.pitch, first.roll, first.yaw] {
            assert!(value.is_finite());
            assert!((-1.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn headless_step_advances_sim_time_and_aircraft_position() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                ticks_per_step: 2,
                ..EnvironmentResetOptions::default()
            },
        );

        let before = env.reset(&EnvironmentResetOptions {
            ticks_per_step: 2,
            ..EnvironmentResetOptions::default()
        });
        let before_fighter1 = before
            .state
            .aircraft
            .iter()
            .find(|aircraft| aircraft.role == "fighter1")
            .expect("fighter1 should exist");

        let result = env.step(EnvironmentAction {
            throttle: 0.1,
            pitch: 0.1,
            ..EnvironmentAction::default()
        });
        let after_fighter1 = result
            .observation
            .state
            .aircraft
            .iter()
            .find(|aircraft| aircraft.role == "fighter1")
            .expect("fighter1 should exist");

        assert!(
            result.info.sim_time_seconds > before.state.sim_time_seconds,
            "expected sim time to advance: before={} after={}",
            before.state.sim_time_seconds,
            result.info.sim_time_seconds
        );
        assert_ne!(
            after_fighter1.position, before_fighter1.position,
            "expected fighter1 position to change after stepping"
        );
    }

    #[test]
    fn step_exposes_audio_observation_window() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                enable_visual: false,
                enable_audio: true,
                ticks_per_step: 4,
                audio_window_seconds: 0.1,
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            enable_visual: false,
            enable_audio: true,
            ticks_per_step: 4,
            audio_window_seconds: 0.1,
            ..EnvironmentResetOptions::default()
        });
        let result = env.step(EnvironmentAction::default());

        let audio = result
            .observation
            .audio
            .as_ref()
            .expect("audio observation should be present");
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.window_seconds, 0.1);
        assert!(!audio.samples.is_empty());
    }

    #[test]
    fn step_result_uses_fighter_role_labels_for_termination_and_winner() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                enable_visual: false,
                enable_audio: false,
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            enable_visual: false,
            enable_audio: false,
            ..EnvironmentResetOptions::default()
        });

        {
            let mut query = env
                .app
                .world_mut()
                .query::<(&AircraftRole, &mut AircraftState)>();
            for (role, mut state) in query.iter_mut(env.app.world_mut()) {
                if *role == AircraftRole::Fighter2 {
                    state.is_destroyed = true;
                }
            }
        }

        let result = env.step(EnvironmentAction::default());
        assert!(result.terminated);
        assert!(!result.truncated);
        assert_eq!(result.info.winner.as_deref(), Some("fighter1"));
    }

    #[test]
    fn reset_applies_self_play_agent_control_to_both_roles() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                agent_control: EnvironmentAgentControlConfig::self_play(),
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            agent_control: EnvironmentAgentControlConfig::self_play(),
            ..EnvironmentResetOptions::default()
        });

        let mut fighter1 = None;
        let mut fighter2 = None;
        let mut query = env
            .app
            .world_mut()
            .query::<(&AircraftRole, &ControlAuthority)>();
        for (role, authority) in query.iter(env.app.world()) {
            match role {
                AircraftRole::Fighter1 => fighter1 = Some(*authority),
                AircraftRole::Fighter2 => fighter2 = Some(*authority),
            }
        }

        assert_eq!(fighter1, Some(ControlAuthority::ExternalAgent));
        assert_eq!(fighter2, Some(ControlAuthority::ExternalAgent));
    }

    #[test]
    fn reset_maps_model_agent_control_to_external_authority() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                agent_control: EnvironmentAgentControlConfig {
                    fighter1: EnvironmentAgentMode::Model,
                    fighter2: EnvironmentAgentMode::BuiltInAi,
                },
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            agent_control: EnvironmentAgentControlConfig {
                fighter1: EnvironmentAgentMode::Model,
                fighter2: EnvironmentAgentMode::BuiltInAi,
            },
            ..EnvironmentResetOptions::default()
        });

        let mut fighter1 = None;
        let mut fighter2 = None;
        let mut query = env
            .app
            .world_mut()
            .query::<(&AircraftRole, &ControlAuthority)>();
        for (role, authority) in query.iter(env.app.world()) {
            match role {
                AircraftRole::Fighter1 => fighter1 = Some(*authority),
                AircraftRole::Fighter2 => fighter2 = Some(*authority),
            }
        }

        assert_eq!(fighter1, Some(ControlAuthority::ExternalAgent));
        assert_eq!(fighter2, Some(ControlAuthority::BuiltInAi));
    }

    #[test]
    fn reset_applies_built_in_ai_profile_overrides() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                agent_control: EnvironmentAgentControlConfig {
                    fighter1: EnvironmentAgentMode::External,
                    fighter2: EnvironmentAgentMode::BuiltInAiPrecise,
                },
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            agent_control: EnvironmentAgentControlConfig {
                fighter1: EnvironmentAgentMode::BuiltInAiTeacher,
                fighter2: EnvironmentAgentMode::BuiltInAiPassiveBounce,
            },
            ..EnvironmentResetOptions::default()
        });

        let overrides = env
            .app
            .world()
            .get_resource::<BuiltInAiProfileOverrides>()
            .copied()
            .expect("BuiltInAiProfileOverrides should be inserted");
        assert_eq!(overrides.fighter1, Some(BuiltInAiProfile::Teacher));
        assert_eq!(overrides.fighter2, Some(BuiltInAiProfile::PassiveBounce));
    }

    #[test]
    fn episode_status_matches_step_result_terminal_fields() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                enable_visual: false,
                enable_audio: false,
                ..EnvironmentResetOptions::default()
            },
        );

        let _ = env.reset(&EnvironmentResetOptions {
            enable_visual: false,
            enable_audio: false,
            ..EnvironmentResetOptions::default()
        });

        let result = env.step(EnvironmentAction::default());
        let status = env.episode_status();

        assert_eq!(status.tick, result.info.tick);
        assert_eq!(status.sim_time_seconds, result.info.sim_time_seconds);
        assert_eq!(status.terminated, result.terminated);
        assert_eq!(status.truncated, result.truncated);
        assert_eq!(status.winner, result.info.winner);
    }

    #[test]
    fn manual_recording_hooks_update_recording_status() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                enable_visual: false,
                enable_audio: false,
                ..EnvironmentResetOptions::default()
            },
        );

        env.start_recording(Some(ObservationCaptureConfig {
            enable_visual: false,
            enable_audio: false,
            visual_sensors: Vec::new(),
            audio_window_seconds: 0.25,
        }))
        .expect("recording should arm successfully");
        let armed = env.recording_status();
        assert!(armed.pending_start);
        assert!(!armed.active);

        assert!(env.stop_recording());
        let stopped = env.recording_status();
        assert!(stopped.pending_stop);
    }

    #[test]
    fn self_play_step_and_recording_access_work_end_to_end() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                enable_visual: false,
                enable_audio: true,
                audio_window_seconds: 0.05,
                ticks_per_step: 2,
                agent_control: EnvironmentAgentControlConfig::self_play(),
                ..EnvironmentResetOptions::default()
            },
        );

        let reset = env.reset(&EnvironmentResetOptions {
            enable_visual: false,
            enable_audio: true,
            audio_window_seconds: 0.05,
            ticks_per_step: 2,
            agent_control: EnvironmentAgentControlConfig::self_play(),
            ..EnvironmentResetOptions::default()
        });
        assert!(
            reset
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter1")
        );
        assert!(
            reset
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter2")
        );

        env.start_recording(Some(ObservationCaptureConfig {
            enable_visual: false,
            enable_audio: true,
            visual_sensors: Vec::new(),
            audio_window_seconds: 0.05,
        }))
        .expect("recording should arm successfully");

        let fighter1_action = EnvironmentAction {
            throttle: 0.05,
            pitch: 0.1,
            fire_gun: true,
            ..EnvironmentAction::default()
        };
        let fighter2_action = EnvironmentAction {
            throttle: -0.02,
            yaw: 0.15,
            ..EnvironmentAction::default()
        };
        let first = env.step_self_play(fighter1_action, fighter2_action);
        assert_eq!(first.reward, None);
        assert!(!first.terminated);
        assert!(!first.truncated);
        assert!(first.observation.audio.is_some());
        assert!(
            first
                .observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter1")
        );
        assert!(
            first
                .observation
                .state
                .aircraft
                .iter()
                .any(|aircraft| aircraft.role == "fighter2")
        );

        let _second =
            env.step_self_play(EnvironmentAction::default(), EnvironmentAction::default());
        assert!(env.stop_recording());
        let final_step =
            env.step_self_play(EnvironmentAction::default(), EnvironmentAction::default());
        let status = env.episode_status();
        assert_eq!(status.terminated, final_step.terminated);
        assert_eq!(status.truncated, final_step.truncated);

        let recording_status = env.recording_status();
        let manifest_path = PathBuf::from(
            recording_status
                .last_saved_path
                .clone()
                .expect("recording should have written an episode manifest"),
        );
        let episode_root = manifest_path
            .parent()
            .expect("manifest path should have an episode root")
            .to_path_buf();

        let access = RecordingAccess::new(&episode_root);
        let manifest = access.manifest().expect("manifest should load");
        let initial = access
            .initial_snapshot()
            .expect("initial snapshot should load");
        let steps = access.steps().expect("steps should load");
        let first_recorded_step = access.step(0).expect("step 0 should load");

        assert_eq!(manifest.schema_version, RECORDING_SCHEMA_VERSION);
        assert!(manifest.authoritative_source);
        assert_eq!(manifest.bridge_role.as_deref(), Some("Server"));
        assert!(manifest.total_steps >= 3);
        assert_eq!(steps.len(), manifest.total_steps as usize);
        assert_eq!(
            manifest.artifact_convention.initial_state_path,
            "initial_state.ron"
        );
        assert_eq!(
            manifest.artifact_convention.step_chunk_pattern,
            "steps/chunk_{chunk:06}.ron"
        );
        assert!(!manifest.step_chunks.is_empty());
        assert_eq!(
            manifest
                .audio_artifact_metadata
                .as_ref()
                .expect("audio artifact metadata should exist")
                .channels,
            2
        );
        assert_eq!(initial.state.scene_name, reset.state.scene_name);
        assert_eq!(first_recorded_step.fighter1_command.throttle, 0.05);
        assert_eq!(first_recorded_step.fighter1_command.pitch, 0.1);
        assert!(first_recorded_step.fighter1_command.fire_gun);
        assert_eq!(first_recorded_step.fighter2_command.throttle, -0.02);
        assert_eq!(first_recorded_step.fighter2_command.yaw, 0.15);
        assert!(
            manifest
                .step_artifacts
                .first()
                .expect("step artifacts should exist")
                .audio
                .is_some()
        );

        fs::remove_dir_all(&episode_root).ok();
    }

    #[test]
    fn reset_with_same_seed_is_repeatable() {
        let mut env = EnvironmentInstance::new_headless(
            ConfigPaths::default(),
            EnvironmentResetOptions {
                seed: Some(12345),
                ..EnvironmentResetOptions::default()
            },
        );

        let first = env.reset(&EnvironmentResetOptions {
            seed: Some(12345),
            ..EnvironmentResetOptions::default()
        });
        let second = env.reset(&EnvironmentResetOptions {
            seed: Some(12345),
            ..EnvironmentResetOptions::default()
        });

        assert_eq!(first.state.seed, 12345);
        assert_eq!(second.state.seed, 12345);
        assert_eq!(first.state.scene_name, second.state.scene_name);
        assert_eq!(first.state.aircraft.len(), second.state.aircraft.len());
        assert_eq!(
            first.state.aircraft[0].position,
            second.state.aircraft[0].position
        );
        assert_eq!(
            first.state.aircraft[0].orientation_quat,
            second.state.aircraft[0].orientation_quat
        );
        assert_eq!(
            first.state.aircraft[1].position,
            second.state.aircraft[1].position
        );
        assert_eq!(
            first.state.aircraft[1].orientation_quat,
            second.state.aircraft[1].orientation_quat
        );
    }
}
