use bevy::prelude::*;
use engine::{activations::Sigmoid, nn::Node as NnNode};

use crate::{inputs::*, InspectWindowState};

use self::{resources::*, systems::*};

mod resources;
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
            .add_systems(OnEnter(InspectWindowState::Display), setup)
            .add_systems(OnExit(InspectWindowState::Display), exit_inspector)
            // .add_systems(Update, exit_inspector.run_if(resource_exists::<InspectorWindowId>))
            .add_systems(Update, test.run_if(in_state(InspectWindowState::Display)));
            // .configure_sets(Update, IwSet::Events.before(IwSet::Window))
            // .add_systems(
            //     Update,
            //     toggle_inspect_window
            //         .run_if(in_state(TabState::Net))
            //         .in_set(IwSet::Events),
            // )
            // .add_systems(Update, control_panel.run_if(in_state(TabState::Net)))
            // .add_systems(
            //     Update,
            //     inspect_window
            //         .run_if(in_state(TabState::Net))
            //         .in_set(IwSet::Window)
            //         .run_if(in_state(InspectWindowState::Display)),
            // );
    }
}
