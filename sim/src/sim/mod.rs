use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;
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
        let cached = load_cached_settings();
        app.insert_resource(ControlPanel::from_cached_settings(&cached))
            .init_resource::<Simulation>()
            .init_resource::<SimulationStats>()
            .insert_resource(cached.config.clone())
            .insert_resource(cached)
            .init_resource::<SystemWindow>()
            .init_resource::<PendingNet>()
            .init_resource::<CreatureExplorer>()
            .init_resource::<ViewState>()
            .init_resource::<WorldBrush>()
            .init_resource::<SpawnArea>()
            .init_state::<SimulationState>()
            .add_plugins(Shape2dPlugin::default())
            .add_systems(Startup, (init_runner, setup))
            .add_systems(Update, poll_generated_world)
            .add_systems(Update, on_creature_explorer_closed)
            .add_systems(Update, persist_window_settings)
            .add_systems(SystemWindowContextPass, system_window)
            .add_systems(EguiPrimaryContextPass, simulation_window_overlay)
            .add_systems(CreatureExplorerContextPass, creature_explorer_window)
            .add_systems(
                Update,
                run_simulation.run_if(not(in_state(SimulationState::None))),
            )
            .add_systems(Update, inspect_creature)
            .add_systems(Update, paint_world)
            .add_systems(Update, deliver_inspect_net)
            .add_systems(Update, camera_controls);
    }
}
