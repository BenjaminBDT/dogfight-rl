use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::api::types::EnvironmentAction;
use crate::input::actions::ControlInput;
use crate::simulation::components::{AircraftRole, ControlAuthority};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetedEnvironmentAction {
    pub role: AircraftRole,
    pub action: EnvironmentAction,
}

#[derive(Debug, Clone, Serialize, Deserialize, Resource, Default)]
pub struct ExternalCommandBuffer {
    pub fighter1_input: Option<ControlInput>,
    pub fighter1_command: Option<EnvironmentAction>,
    pub targeted_actions: Vec<TargetedEnvironmentAction>,
}

pub fn apply_external_commands(
    mut buffer: bevy::prelude::ResMut<ExternalCommandBuffer>,
    mut query: bevy::prelude::Query<(&AircraftRole, &ControlAuthority, &mut ControlInput)>,
) {
    if let Some(action) = buffer.fighter1_command.take() {
        buffer.targeted_actions.push(TargetedEnvironmentAction {
            role: AircraftRole::Fighter1,
            action,
        });
    } else if let Some(input) = buffer.fighter1_input.take() {
        buffer.targeted_actions.push(TargetedEnvironmentAction {
            role: AircraftRole::Fighter1,
            action: EnvironmentAction::from(input),
        });
    }

    if buffer.targeted_actions.is_empty() {
        return;
    }

    let targeted_actions = std::mem::take(&mut buffer.targeted_actions);
    for targeted in targeted_actions {
        for (role, authority, mut input) in &mut query {
            if *role == targeted.role
                && matches!(
                    authority,
                    ControlAuthority::ExternalAgent | ControlAuthority::Replay
                )
            {
                *input = ControlInput::from(targeted.action);
            }
        }
    }
}
