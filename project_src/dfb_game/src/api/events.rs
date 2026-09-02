use bevy::prelude::*;

use crate::api::types::{EnvironmentEvent, EnvironmentEventKind};
use crate::app::schedules::SimulationSet;
use crate::core::config::RepositoryConfig;
use crate::gameplay::damage::AircraftDamageState;
use crate::gameplay::match_state::MatchPhase;
use crate::simulation::components::{AircraftRole, AircraftState};
use crate::simulation::resources::SimulationDebugState;

const STALL_EVENT_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone, Default, Resource)]
pub struct PendingEnvironmentEvents {
    pub events: Vec<EnvironmentEvent>,
}

#[derive(Debug, Clone, Copy, Default)]
struct AircraftEventSnapshot {
    initialized: bool,
    is_destroyed: bool,
    out_of_bounds: bool,
    stalled: bool,
}

#[derive(Debug, Clone, Default, Resource)]
pub struct AircraftEventTracker {
    player: AircraftEventSnapshot,
    enemy: AircraftEventSnapshot,
    previous_match_phase: Option<MatchPhase>,
}

pub struct EnvironmentEventPlugin;

impl Plugin for EnvironmentEventPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingEnvironmentEvents>()
            .init_resource::<AircraftEventTracker>()
            .add_systems(
                FixedUpdate,
                track_environment_events
                    .before(crate::api::snapshot::capture_world_snapshot)
                    .in_set(SimulationSet::ProduceSnapshot),
            );
    }
}

pub fn push_event(
    queue: &mut PendingEnvironmentEvents,
    tick: u64,
    kind: EnvironmentEventKind,
    subject: Option<String>,
    position: Option<Vec3>,
    magnitude: Option<f32>,
) {
    queue.events.push(EnvironmentEvent::new(
        tick,
        kind,
        subject,
        position.map(|value| value.to_array()),
        magnitude,
    ));
}

pub fn push_event_with_context(
    queue: &mut PendingEnvironmentEvents,
    tick: u64,
    kind: EnvironmentEventKind,
    subject: Option<String>,
    other_subject: Option<String>,
    position: Option<Vec3>,
    magnitude: Option<f32>,
    event_detail: Option<&str>,
    subsystem: Option<&str>,
) {
    queue.events.push(EnvironmentEvent::with_context(
        tick,
        kind,
        subject,
        other_subject,
        position.map(|value| value.to_array()),
        magnitude,
        event_detail.map(ToOwned::to_owned),
        subsystem.map(ToOwned::to_owned),
    ));
}

pub fn role_subject(role: AircraftRole) -> String {
    match role {
        AircraftRole::Fighter1 => "fighter1",
        AircraftRole::Fighter2 => "fighter2",
    }
    .to_string()
}

pub fn subsystem_subject(role: AircraftRole, subsystem: &str) -> String {
    format!("{}:{subsystem}", role_subject(role))
}

fn track_environment_events(
    config: Res<RepositoryConfig>,
    debug: Res<SimulationDebugState>,
    match_phase: Res<State<MatchPhase>>,
    mut tracker: ResMut<AircraftEventTracker>,
    mut events: ResMut<PendingEnvironmentEvents>,
    query: Query<(&AircraftRole, &AircraftState, &AircraftDamageState)>,
) {
    let previous_player_destroyed = tracker.player.is_destroyed;
    let previous_enemy_destroyed = tracker.enemy.is_destroyed;
    let mut player_destroyed = None;
    let mut enemy_destroyed = None;
    let mut player_position = None;
    let mut enemy_position = None;

    for (role, state, _damage) in &query {
        let snapshot = match role {
            AircraftRole::Fighter1 => &mut tracker.player,
            AircraftRole::Fighter2 => &mut tracker.enemy,
        };

        let subject = Some(role_subject(*role));
        let position = Some(state.position);
        let radial_distance = Vec2::new(state.position.x, state.position.z).length();
        let out_of_bounds = radial_distance >= config.scene.arena_radius;
        let stalled = state.stall_factor >= STALL_EVENT_THRESHOLD;

        if snapshot.initialized {
            if !snapshot.is_destroyed && state.is_destroyed {
                push_event(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::Destroy,
                    subject.clone(),
                    position,
                    Some(state.hit_points),
                );
            }

            if !snapshot.out_of_bounds && out_of_bounds {
                push_event_with_context(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::OutOfBounds,
                    subject.clone(),
                    None,
                    position,
                    Some(state.out_of_bounds_seconds),
                    Some("horizontal"),
                    None,
                );
            }

            if !snapshot.stalled && stalled {
                push_event(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::StallEntered,
                    subject.clone(),
                    position,
                    Some(state.stall_factor),
                );
            } else if snapshot.stalled && !stalled {
                push_event(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::StallRecovered,
                    subject.clone(),
                    position,
                    Some(state.stall_factor),
                );
            }
        } else {
            if out_of_bounds {
                push_event_with_context(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::OutOfBounds,
                    subject.clone(),
                    None,
                    position,
                    Some(state.out_of_bounds_seconds),
                    Some("horizontal"),
                    None,
                );
            }
            if stalled {
                push_event(
                    &mut events,
                    debug.tick_count,
                    EnvironmentEventKind::StallEntered,
                    subject.clone(),
                    position,
                    Some(state.stall_factor),
                );
            }
        }

        snapshot.initialized = true;
        snapshot.is_destroyed = state.is_destroyed;
        snapshot.out_of_bounds = out_of_bounds;
        snapshot.stalled = stalled;

        match role {
            AircraftRole::Fighter1 => {
                player_destroyed = Some(state.is_destroyed);
                player_position = Some(state.position);
            }
            AircraftRole::Fighter2 => {
                enemy_destroyed = Some(state.is_destroyed);
                enemy_position = Some(state.position);
            }
        }
    }

    if let (Some(player_destroyed), Some(enemy_destroyed)) = (player_destroyed, enemy_destroyed) {
        if tracker.player.initialized
            && !previous_enemy_destroyed
            && player_destroyed
            && !enemy_destroyed
        {
            push_event(
                &mut events,
                debug.tick_count,
                EnvironmentEventKind::Kill,
                Some(role_subject(AircraftRole::Fighter2)),
                enemy_position,
                Some(1.0),
            );
        }

        if tracker.enemy.initialized
            && !previous_player_destroyed
            && enemy_destroyed
            && !player_destroyed
        {
            push_event(
                &mut events,
                debug.tick_count,
                EnvironmentEventKind::Kill,
                Some(role_subject(AircraftRole::Fighter1)),
                player_position,
                Some(1.0),
            );
        }
    }

    let current_phase = *match_phase.get();
    let previous_phase = tracker.previous_match_phase;
    if previous_phase != Some(current_phase) && current_phase == MatchPhase::Finished {
        match (player_destroyed, enemy_destroyed) {
            (Some(false), Some(true)) => push_event(
                &mut events,
                debug.tick_count,
                EnvironmentEventKind::Win,
                Some(role_subject(AircraftRole::Fighter1)),
                player_position,
                Some(1.0),
            ),
            (Some(true), Some(false)) => push_event(
                &mut events,
                debug.tick_count,
                EnvironmentEventKind::Win,
                Some(role_subject(AircraftRole::Fighter2)),
                enemy_position,
                Some(1.0),
            ),
            _ => {}
        }
    }

    tracker.previous_match_phase = Some(current_phase);
}
