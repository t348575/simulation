use bevy::prelude::*;

use crate::InspectWindowState;

use self::{resources::*, systems::*};

pub mod resources;
mod systems;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum IwSet {
    Events,
    Window,
}

pub struct NeuralNetPlugin;

impl Plugin for NeuralNetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InspectInfo>()
            .init_resource::<WindowInfo>()
            .init_resource::<ControlPanel>()
            .init_state::<InspectNodeState>()
            .add_systems(OnEnter(InspectWindowState::Display), setup)
            .add_systems(OnExit(InspectWindowState::Display), exit_inspector)
            .add_systems(
                Update,
                get_inspect_net.run_if(in_state(InspectWindowState::Display)),
            )
            .configure_sets(Update, IwSet::Events.before(IwSet::Window))
            .add_systems(
                Update,
                toggle_inspect_window
                    .run_if(in_state(InspectWindowState::Display))
                    .run_if(resource_exists::<Nn>)
                    .in_set(IwSet::Events),
            )
            .add_systems(
                Update,
                inspect_window
                    .run_if(in_state(InspectWindowState::Display))
                    .in_set(IwSet::Window)
                    .run_if(resource_exists::<Nn>)
                    .run_if(in_state(InspectNodeState::Display)),
            );
    }
}
