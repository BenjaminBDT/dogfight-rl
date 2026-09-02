pub mod collision;
pub mod combat;
pub mod damage;
pub mod match_state;
pub mod reset;
pub mod server_admin;
pub mod spawning;

use crate::app::schedules::SimulationSet;
use crate::app::{AppMode, HeadlessMode};
use crate::bridge::protocol::BridgeRole;
use crate::bridge::{should_run_local_fire_visuals, should_run_local_gameplay_authority};
use bevy::prelude::*;

pub struct SharedGameplayStatePlugin;
pub struct GameplayPlugin;

impl Plugin for SharedGameplayStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<match_state::MatchPhase>()
            .init_resource::<match_state::MatchClock>()
            .init_resource::<combat::CombatState>()
            .init_resource::<combat::CombatPresentationQueue>()
            .init_resource::<combat::LocalFireVisualState>()
            .init_resource::<reset::PendingMatchReset>()
            .init_resource::<server_admin::ServerAdminState>()
            .init_resource::<server_admin::PendingServerShutdown>();
    }
}

impl Plugin for GameplayPlugin {
    fn build(&self, app: &mut App) {
        let headless = app
            .world()
            .get_resource::<HeadlessMode>()
            .map(|mode| mode.0)
            .unwrap_or(false);
        let app_mode = app
            .world()
            .get_resource::<AppMode>()
            .copied()
            .unwrap_or(AppMode::Game);
        let bridge_role = app
            .world()
            .get_resource::<crate::bridge::BridgeMode>()
            .map(|mode| mode.0);
        let admin_interface = app
            .world()
            .get_resource::<server_admin::ServerAdminInterfaceMode>()
            .copied()
            .unwrap_or(server_admin::ServerAdminInterfaceMode::Plain);
        app.add_systems(
            FixedUpdate,
            damage::update_repair_state
                .in_set(SimulationSet::StepSimulation)
                .run_if(should_run_local_gameplay_authority),
        )
        .add_systems(
            FixedUpdate,
            (
                combat::tick_weapon_cooldowns,
                combat::tick_weapon_heat,
                combat::resolve_gunfire,
                combat::update_projectiles,
                collision::resolve_environment_collisions,
                collision::resolve_aircraft_collisions,
                match_state::update_match_phase_from_aircraft,
            )
                .chain()
                .in_set(SimulationSet::ResolveGameplay)
                .run_if(should_run_local_gameplay_authority),
        )
        .add_systems(
            FixedUpdate,
            (
                combat::tick_local_fire_visual_cooldowns,
                combat::spawn_local_fire_visuals,
                combat::update_local_visual_projectiles,
            )
                .chain()
                .in_set(SimulationSet::ResolveGameplay)
                .run_if(should_run_local_fire_visuals),
        )
        .add_systems(
            Update,
            match_state::tick_match_clock.run_if(should_run_local_gameplay_authority),
        )
        .add_systems(
            FixedUpdate,
            reset::apply_deferred_match_reset
                .after(crate::recording::record_episode_step)
                .run_if(should_run_local_gameplay_authority),
        )
        .add_systems(
            Update,
            reset::auto_reset_finished_server_match.run_if(should_run_local_gameplay_authority),
        );
        if !headless && app_mode == AppMode::Game {
            app.add_systems(
                Update,
                reset::handle_reset_input.run_if(should_run_local_gameplay_authority),
            );
        }
        if headless && app_mode == AppMode::Game && bridge_role == Some(BridgeRole::Server) {
            app.add_systems(Update, server_admin::process_pending_server_shutdown);
            match admin_interface {
                server_admin::ServerAdminInterfaceMode::Off => {}
                server_admin::ServerAdminInterfaceMode::Plain => {
                    app.add_systems(Startup, server_admin::start_plain_server_admin_console)
                        .add_systems(Update, server_admin::drive_plain_server_admin_commands);
                }
                server_admin::ServerAdminInterfaceMode::Tui => {
                    app.add_systems(Startup, server_admin::start_tui_server_admin)
                        .add_systems(
                            Update,
                            (
                                server_admin::drain_server_admin_logs,
                                server_admin::drive_tui_server_admin_input,
                                server_admin::render_server_admin_tui,
                            )
                                .chain(),
                        );
                }
            }
        }
    }
}
