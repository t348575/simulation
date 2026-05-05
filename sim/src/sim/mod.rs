use bevy::prelude::*;
use bevy_vector_shapes::Shape2dPlugin;

use self::{resources::*, systems::*};

mod gpu;
mod resources;
mod sim;
mod spatial;
mod systems;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ControlPanel>()
            .init_resource::<Simulation>()
            .init_resource::<SimulationStats>()
            .insert_resource(load_cached_config())
            .init_resource::<SystemWindow>()
            .init_resource::<PendingNet>()
            .init_resource::<CreatureExplorer>()
            .init_resource::<ViewState>()
            .init_state::<SimulationState>()
            .add_plugins(Shape2dPlugin::default())
            .add_systems(Startup, (init_runner, setup))
            .add_systems(Update, poll_generated_world)
            .add_systems(Update, on_creature_explorer_closed)
            .add_systems(SystemWindowContextPass, system_window)
            .add_systems(CreatureExplorerContextPass, creature_explorer_window)
            .add_systems(
                Update,
                run_simulation.run_if(not(in_state(SimulationState::None))),
            )
            .add_systems(Update, inspect_creature)
            .add_systems(Update, deliver_inspect_net)
            .add_systems(Update, camera_controls);
    }
}
