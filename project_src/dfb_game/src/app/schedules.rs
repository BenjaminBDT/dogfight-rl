use bevy::prelude::*;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimulationSet {
    GatherInput,
    StepSimulation,
    ResolveGameplay,
    ProduceSnapshot,
}
