use bevy::prelude::*;
use bevy_vector_shapes::Shape2dPlugin;

use crate::{InspectWindowState, TabState};

use self::{resources::*, systems::*};

mod resources;
mod systems;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ControlPanel>()
            .init_resource::<Simulation>()
            .init_state::<SimulationState>()
            .add_plugins(Shape2dPlugin::default())
            .add_systems(OnEnter(TabState::Simulation), setup)
            .add_systems(Update, control_panel.run_if(in_state(TabState::Simulation)))
            .add_systems(
                OnTransition {
                    from: SimulationState::None,
                    to: SimulationState::Paused,
                },
                initialize_world,
            )
            .add_systems(
                Update,
                run_simulation.run_if(in_state(SimulationState::Running)),
            )
            .add_systems(
                Update,
                inspect_creature
                    .run_if(in_state(InspectWindowState::Display))
                    .run_if(in_state(TabState::Simulation)),
            );
    }
}
