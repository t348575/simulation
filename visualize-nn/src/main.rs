use bevy::{prelude::*, window::WindowMode};
use net::NeuralNetPlugin;

pub mod inputs;
mod net;
pub mod outputs;
mod sim;

fn main() {
    let mut window_plugin = WindowPlugin {
        primary_window: Some(Window {
            title: "Visualize NN".into(),
            mode: WindowMode::Windowed,
            ..default()
        }),
        ..default()
    };
    window_plugin
        .primary_window
        .as_mut()
        .unwrap()
        .set_maximized(true);

    App::new()
        .add_plugins(DefaultPlugins.set(window_plugin))
        .add_plugins(NeuralNetPlugin)
        // .add_state::<InspectWindowState>()
        // .init_resource::<InspectInfo>()
        // .insert_resource(Nn {
        //     net,
        //     node_positions: Vec::new(),
        //     input_nodes,
        //     output_nodes,
        // })
        // .add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::WHITE))
        // .add_plugins(Shape2dPlugin::default())
        // .add_systems(Startup, setup)
        // .add_systems(Update, gen_new_graph)
        // .configure_sets(Update, IwSet::Events.before(IwSet::Window))
        // .add_systems(Update, toggle_inspect_window.in_set(IwSet::Events))
        // .add_systems(
        //     Update,
        //     inspect_window
        //         .in_set(IwSet::Window)
        //         .run_if(in_state(InspectWindowState::Display)),
        // )
        .run();
}
