use bevy::input::ButtonInput;
use bevy::prelude::*;

use crate::bridge::BridgeMode;
use crate::bridge::protocol::BridgeRole;
use crate::core::config::RepositoryConfig;
use crate::gameplay::combat::{CombatState, LocalVisualProjectile, Projectile};
use crate::gameplay::damage::{AircraftDamageState, apply_spawn_damage_overrides};
use crate::gameplay::match_state::{MatchClock, MatchPhase};
use crate::gameplay::server_admin::ServerAdminState;
use crate::input::actions::{ControlBindings, ControlInput};
use crate::presentation::tracers::TracerLifetime;
use crate::recording::ActionRecordingState;
use crate::simulation::components::{
    AircraftPerformance, AircraftRole, AircraftState, GunState, SpawnTransform,
};
use crate::simulation::systems::{
    aircraft_performance_from_config, scene_flight_path_orientation, trimmed_spawn_orientation,
};

#[derive(Debug, Default, Resource)]
pub struct PendingMatchReset {
    pub requested: bool,
}

type TransientCombatEntityFilter = Or<(
    With<Projectile>,
    With<LocalVisualProjectile>,
    With<TracerLifetime>,
)>;

fn despawn_transient_combat_entities(
    commands: &mut Commands,
    query: &Query<'_, '_, Entity, TransientCombatEntityFilter>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

pub fn reset_match_world(
    config: &RepositoryConfig,
    clock: &mut MatchClock,
    combat_state: &mut CombatState,
    query: &mut Query<
        '_,
        '_,
        (
            &AircraftRole,
            &mut SpawnTransform,
            &mut AircraftPerformance,
            &mut AircraftState,
            &mut AircraftDamageState,
            &mut ControlInput,
            &mut GunState,
            &mut Transform,
        ),
    >,
) {
    clock.elapsed_seconds = 0.0;
    *combat_state = CombatState::default();

    for (
        role,
        mut spawn,
        mut performance,
        mut state,
        mut damage,
        mut input,
        mut gun,
        mut transform,
    ) in query.iter_mut()
    {
        let (
            spawn_position,
            spawn_orientation,
            spawn_velocity_orientation,
            initial_speed,
            initial_throttle,
            next_performance,
            hit_points,
        ) = match *role {
            AircraftRole::Fighter1 => {
                let spawn_cfg = &config.scene.fighter1_spawn;
                let next_performance = aircraft_performance_from_config(&config.fighter1_aircraft);
                let flight_path_orientation =
                    scene_flight_path_orientation(spawn_cfg.rotation_degrees);
                (
                    Vec3::from_array(spawn_cfg.position),
                    trimmed_spawn_orientation(
                        flight_path_orientation,
                        next_performance.trim_pitch_degrees,
                    ),
                    flight_path_orientation,
                    spawn_cfg.initial_speed,
                    spawn_cfg.initial_throttle,
                    next_performance,
                    config.fighter1_aircraft.hit_points,
                )
            }
            AircraftRole::Fighter2 => {
                let spawn_cfg = &config.scene.fighter2_spawn;
                let next_performance = aircraft_performance_from_config(&config.fighter2_aircraft);
                let flight_path_orientation =
                    scene_flight_path_orientation(spawn_cfg.rotation_degrees);
                (
                    Vec3::from_array(spawn_cfg.position),
                    trimmed_spawn_orientation(
                        flight_path_orientation,
                        next_performance.trim_pitch_degrees,
                    ),
                    flight_path_orientation,
                    spawn_cfg.initial_speed,
                    spawn_cfg.initial_throttle,
                    next_performance,
                    config.fighter2_aircraft.hit_points,
                )
            }
        };
        spawn.position = spawn_position;
        spawn.orientation = spawn_orientation;
        *performance = next_performance.clone();
        *state = AircraftState {
            position: spawn.position,
            velocity: spawn_velocity_orientation * Vec3::new(0.0, 0.0, initial_speed),
            orientation: spawn.orientation,
            forward: spawn.orientation * Vec3::Z,
            throttle: initial_throttle.clamp(0.0, 1.0),
            hit_points,
            stall_factor: 0.0,
            out_of_bounds_seconds: 0.0,
            ceiling_recovery_seconds: 0.0,
            ceiling_recovery_target_pitch_deg: -30.0,
            is_destroyed: false,
            ..Default::default()
        };
        *damage = AircraftDamageState::new(hit_points, &performance);
        let spawn_damage = match role {
            AircraftRole::Fighter1 => config.scene.fighter1_spawn.initial_damage.as_ref(),
            AircraftRole::Fighter2 => config.scene.fighter2_spawn.initial_damage.as_ref(),
        };
        apply_spawn_damage_overrides(
            state.as_mut(),
            damage.as_mut(),
            performance.as_ref(),
            spawn_damage,
        );
        *input = ControlInput::default();
        *gun = GunState::default();
        transform.translation = spawn.position;
        transform.rotation = spawn.orientation;
    }
}

pub fn handle_reset_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    config: Res<RepositoryConfig>,
    bindings: Res<ControlBindings>,
    mut recording: Option<ResMut<ActionRecordingState>>,
    admin: Option<Res<ServerAdminState>>,
    mut pending_reset: ResMut<PendingMatchReset>,
    mut clock: ResMut<MatchClock>,
    mut combat_state: ResMut<CombatState>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut commands: Commands,
    transient_entities: Query<Entity, TransientCombatEntityFilter>,
    mut query: Query<(
        &AircraftRole,
        &mut SpawnTransform,
        &mut AircraftPerformance,
        &mut AircraftState,
        &mut AircraftDamageState,
        &mut ControlInput,
        &mut GunState,
        &mut Transform,
    )>,
) {
    if !bindings.reset_match.just_pressed(&keyboard, &mouse_buttons) {
        return;
    }

    if let Some(recording) = recording.as_mut()
        && (recording.active || recording.pending_start || recording.pending_stop)
    {
        recording.pending_stop = true;
        pending_reset.requested = true;
        info!("deferring match reset until active episode recording is saved");
        return;
    }

    despawn_transient_combat_entities(&mut commands, &transient_entities);
    reset_match_world(&config, &mut clock, &mut combat_state, &mut query);
    next_phase.set(
        if admin
            .as_deref()
            .map(|state| state.hold_match)
            .unwrap_or(false)
        {
            MatchPhase::Loading
        } else {
            MatchPhase::Running
        },
    );
}

pub fn apply_deferred_match_reset(
    config: Res<RepositoryConfig>,
    recording: Option<Res<ActionRecordingState>>,
    admin: Option<Res<ServerAdminState>>,
    mut pending_reset: ResMut<PendingMatchReset>,
    mut clock: ResMut<MatchClock>,
    mut combat_state: ResMut<CombatState>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut commands: Commands,
    transient_entities: Query<Entity, TransientCombatEntityFilter>,
    mut query: Query<(
        &AircraftRole,
        &mut SpawnTransform,
        &mut AircraftPerformance,
        &mut AircraftState,
        &mut AircraftDamageState,
        &mut ControlInput,
        &mut GunState,
        &mut Transform,
    )>,
) {
    if !pending_reset.requested {
        return;
    }

    if let Some(recording) = recording.as_ref()
        && (recording.active || recording.pending_start || recording.pending_stop)
    {
        return;
    }

    despawn_transient_combat_entities(&mut commands, &transient_entities);
    reset_match_world(&config, &mut clock, &mut combat_state, &mut query);
    next_phase.set(
        if admin
            .as_deref()
            .map(|state| state.hold_match)
            .unwrap_or(false)
        {
            MatchPhase::Loading
        } else {
            MatchPhase::Running
        },
    );
    pending_reset.requested = false;
}

pub fn auto_reset_finished_server_match(
    config: Res<RepositoryConfig>,
    bridge_mode: Option<Res<BridgeMode>>,
    admin: Option<Res<ServerAdminState>>,
    current_phase: Res<State<MatchPhase>>,
    mut clock: ResMut<MatchClock>,
    mut combat_state: ResMut<CombatState>,
    mut next_phase: ResMut<NextState<MatchPhase>>,
    mut commands: Commands,
    transient_entities: Query<Entity, TransientCombatEntityFilter>,
    mut query: Query<(
        &AircraftRole,
        &mut SpawnTransform,
        &mut AircraftPerformance,
        &mut AircraftState,
        &mut AircraftDamageState,
        &mut ControlInput,
        &mut GunState,
        &mut Transform,
    )>,
) {
    if !matches!(
        bridge_mode.as_deref().map(|mode| mode.0),
        Some(BridgeRole::Server)
    ) {
        return;
    }
    if *current_phase.get() != MatchPhase::Finished {
        return;
    }

    despawn_transient_combat_entities(&mut commands, &transient_entities);
    reset_match_world(&config, &mut clock, &mut combat_state, &mut query);
    next_phase.set(
        if admin
            .as_deref()
            .map(|state| state.hold_match)
            .unwrap_or(false)
        {
            MatchPhase::Loading
        } else {
            MatchPhase::Running
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_transient_combat_entities_for_test(
        mut commands: Commands,
        query: Query<Entity, TransientCombatEntityFilter>,
    ) {
        despawn_transient_combat_entities(&mut commands, &query);
    }

    #[test]
    fn reset_cleanup_despawns_all_transient_combat_entities() {
        let mut app = App::new();
        app.add_systems(Update, clear_transient_combat_entities_for_test);

        app.world_mut().spawn(Projectile {
            id: 1,
            shooter_role: AircraftRole::Fighter1,
            velocity: Vec3::Z,
            damage: 3.0,
            remaining_distance: 100.0,
            hit_radius: 0.8,
            flyby_emitted: false,
            lag_compensation_ticks: 0,
        });
        app.world_mut().spawn(LocalVisualProjectile {
            velocity: Vec3::Z,
            remaining_distance: 100.0,
        });
        app.world_mut().spawn(TracerLifetime {
            remaining_seconds: 0.1,
        });

        app.update();

        assert_eq!(
            app.world_mut()
                .query::<&Projectile>()
                .iter(app.world())
                .count(),
            0
        );
        assert_eq!(
            app.world_mut()
                .query::<&LocalVisualProjectile>()
                .iter(app.world())
                .count(),
            0
        );
        assert_eq!(
            app.world_mut()
                .query::<&TracerLifetime>()
                .iter(app.world())
                .count(),
            0
        );
    }
}
