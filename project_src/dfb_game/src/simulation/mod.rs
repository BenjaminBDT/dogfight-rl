pub mod collision;
pub mod components;
pub mod flight_model;
pub mod resources;
pub mod systems;

use bevy::prelude::*;

use crate::app::RenderEnabled;
use crate::app::schedules::SimulationSet;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        let render_enabled = app
            .world()
            .get_resource::<RenderEnabled>()
            .map(|mode| mode.0)
            .unwrap_or(true);
        app.init_resource::<resources::SimulationDebugState>()
            .configure_sets(
                FixedUpdate,
                (
                    SimulationSet::GatherInput,
                    SimulationSet::StepSimulation,
                    SimulationSet::ResolveGameplay,
                    SimulationSet::ProduceSnapshot,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                (
                    systems::apply_control_inputs.in_set(SimulationSet::GatherInput),
                    systems::integrate_flight_model.in_set(SimulationSet::StepSimulation),
                    systems::predict_local_environment_collisions
                        .in_set(SimulationSet::StepSimulation)
                        .after(systems::integrate_flight_model),
                ),
            )
            .add_systems(Update, systems::update_aircraft_state);
        if render_enabled {
            app.add_systems(
                Startup,
                (
                    systems::spawn_scene_obstacle_colliders,
                    systems::spawn_placeholder_aircraft,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    systems::attach_aircraft_visual_parts,
                    systems::attach_aircraft_semantic_parts,
                    systems::update_damage_visuals,
                )
                    .chain(),
            );
        } else {
            app.add_systems(
                Startup,
                (
                    systems::spawn_scene_obstacle_colliders,
                    systems::spawn_headless_aircraft,
                )
                    .chain(),
            );
        }
    }
}
