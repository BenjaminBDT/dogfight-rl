use bevy::prelude::*;

use crate::bridge::protocol::BridgeRole;
use crate::bridge::{BridgeEnabled, BridgeMode, BridgeServerSessions, players_ready};
use crate::core::config::RepositoryConfig;
use crate::gameplay::server_admin::ServerAdminState;
use crate::simulation::components::{AircraftRole, AircraftState};

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatchPhase {
    #[default]
    Loading,
    Running,
    Finished,
}

#[derive(Debug, Default, Resource)]
pub struct MatchClock {
    pub elapsed_seconds: f32,
}

pub fn tick_match_clock(
    time: Res<Time>,
    config: Res<RepositoryConfig>,
    mut clock: ResMut<MatchClock>,
    current_phase: Res<State<MatchPhase>>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    bridge_mode: Option<Res<BridgeMode>>,
    bridge_enabled: Option<Res<BridgeEnabled>>,
    server_sessions: Option<Res<BridgeServerSessions>>,
    admin: Option<Res<ServerAdminState>>,
) {
    let waiting_for_players = matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Server)
    ) && bridge_enabled
        .as_deref()
        .map(|enabled| enabled.0)
        .unwrap_or(true)
        && !server_sessions
            .as_deref()
            .map(players_ready)
            .unwrap_or(false);
    let hold_match = admin
        .as_deref()
        .map(|state| state.hold_match)
        .unwrap_or(false);

    match current_phase.get() {
        MatchPhase::Loading => {
            if !waiting_for_players && !hold_match {
                next_phase.set(MatchPhase::Running);
            }
        }
        MatchPhase::Running => {
            if waiting_for_players || hold_match {
                next_phase.set(MatchPhase::Loading);
                return;
            }
            clock.elapsed_seconds += time.delta_secs();
            if match_time_limit_reached(clock.elapsed_seconds, config.game.match_time_limit_seconds)
            {
                next_phase.set(MatchPhase::Finished);
            }
        }
        MatchPhase::Finished => {}
    }
}

fn match_time_limit_reached(elapsed_seconds: f32, limit_seconds: Option<f32>) -> bool {
    limit_seconds
        .filter(|limit| limit.is_finite() && *limit > 0.0)
        .is_some_and(|limit| elapsed_seconds >= limit)
}

pub fn update_match_phase_from_aircraft(
    current_phase: Res<State<MatchPhase>>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    query: Query<(&AircraftRole, &AircraftState)>,
) {
    if *current_phase.get() != MatchPhase::Running {
        return;
    }

    let mut player_destroyed = false;
    let mut enemy_destroyed = false;

    for (role, state) in &query {
        match role {
            AircraftRole::Fighter1 => player_destroyed |= state.is_destroyed,
            AircraftRole::Fighter2 => enemy_destroyed |= state.is_destroyed,
        }
    }

    if player_destroyed || enemy_destroyed {
        next_phase.set(MatchPhase::Finished);
    }
}

#[cfg(test)]
mod tests {
    use super::match_time_limit_reached;

    #[test]
    fn none_match_time_limit_means_unbounded_match() {
        assert!(!match_time_limit_reached(10_000.0, None));
    }

    #[test]
    fn finite_positive_match_time_limit_still_finishes_match() {
        assert!(!match_time_limit_reached(59.9, Some(60.0)));
        assert!(match_time_limit_reached(60.0, Some(60.0)));
        assert!(match_time_limit_reached(60.1, Some(60.0)));
    }

    #[test]
    fn non_positive_or_non_finite_limits_are_treated_as_unbounded() {
        assert!(!match_time_limit_reached(10_000.0, Some(0.0)));
        assert!(!match_time_limit_reached(10_000.0, Some(-1.0)));
        assert!(!match_time_limit_reached(10_000.0, Some(f32::INFINITY)));
    }
}
