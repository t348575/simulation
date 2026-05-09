use bevy::{
    prelude::*,
    window::{ExitCondition, PresentMode, WindowMode},
};
use bevy_egui::EguiPlugin;
use engine::{activations::Sigmoid, nn::Node};
use inputs::*;
use net::{resources::InspectNet, NeuralNetPlugin};
use sim::SimulationPlugin;

// Re-export lib's `inputs` so the typetag registrations live in a single
// crate. If we declared `mod inputs;` here, inputs.rs would be compiled twice
// (once in lib, once in bin) and typetag would see duplicate "Hunger" tags.
mod inputs {
    pub use ::sim::inputs::*;
}
mod net;
mod sim;

#[derive(Resource, Debug)]
pub struct BaseNodes {
    pub input_nodes: Vec<Node>,
    pub output_nodes: Vec<Node>,
}

fn main() {
    let mut window_plugin = WindowPlugin {
        primary_window: Some(Window {
            title: "Visualize NN".into(),
            present_mode: PresentMode::AutoVsync,
            mode: WindowMode::Windowed,
            ..default()
        }),
        exit_condition: ExitCondition::OnPrimaryClosed,
        ..default()
    };
    window_plugin
        .primary_window
        .as_mut()
        .unwrap()
        .set_maximized(true);

    let input_nodes = vec![
        Node::Input(Hunger::new(0.0, 0)),
        Node::Input(Health::new(100.0, 1)),
        Node::Input(BlankInput::new(0.0, 2, "prev_thrust")),
        Node::Input(BlankInput::new(0.0, 3, "prev_turn")),
        Node::Input(BlankInput::new(1.0, 4, "heading_cos")),
        Node::Input(BlankInput::new(0.0, 5, "heading_sin")),
        Node::Input(BlankInput::new(0.0, 6, "touch")),
        Node::Input(BlankInput::new(0.0, 7, "damage")),
        Node::Input(BlankInput::new(0.0, 8, "osc")),
        Node::Input(BlankInput::new(0.0, 9, "mem_in")),
        Node::Input(Age::new(0, 10)),
        Node::Input(BlankInput::new(0.0, 11, "size")),
        Node::Input(BlankInput::new(0.0, 12, "wall_left")),
        Node::Input(BlankInput::new(0.0, 13, "wall_right")),
        Node::Input(BlankInput::new(0.0, 14, "wall_btm")),
        Node::Input(BlankInput::new(0.0, 15, "wall_top")),
        Node::Input(BlankInput::new(0.0, 16, "food_s0")),
        Node::Input(BlankInput::new(0.0, 17, "food_s1")),
        Node::Input(BlankInput::new(0.0, 18, "food_s2")),
        Node::Input(BlankInput::new(0.0, 19, "food_s3")),
        Node::Input(BlankInput::new(0.0, 20, "poison_s0")),
        Node::Input(BlankInput::new(0.0, 21, "poison_s1")),
        Node::Input(BlankInput::new(0.0, 22, "poison_s2")),
        Node::Input(BlankInput::new(0.0, 23, "poison_s3")),
        Node::Input(BlankInput::new(0.0, 24, "obstacle_s0")),
        Node::Input(BlankInput::new(0.0, 25, "obstacle_s1")),
        Node::Input(BlankInput::new(0.0, 26, "obstacle_s2")),
        Node::Input(BlankInput::new(0.0, 27, "obstacle_s3")),
        Node::Input(BlankInput::new(0.0, 28, "creature_s0")),
        Node::Input(BlankInput::new(0.0, 29, "creature_s1")),
        Node::Input(BlankInput::new(0.0, 30, "creature_s2")),
        Node::Input(BlankInput::new(0.0, 31, "creature_s3")),
        Node::Input(BlankInput::new(0.0, 32, "terrain_elevation")),
        Node::Input(BlankInput::new(1.0, 33, "terrain_speed")),
        Node::Input(BlankInput::new(0.0, 34, "terrain_cost")),
        Node::Input(BlankInput::new(0.0, 35, "terrain_hazard")),
        Node::Input(BlankInput::new(0.0, 36, "food_odor")),
        Node::Input(BlankInput::new(0.0, 37, "food_color")),
        Node::Input(BlankInput::new(0.0, 38, "poison_odor")),
        Node::Input(BlankInput::new(0.0, 39, "poison_color")),
        Node::Input(BlankInput::new(0.0, 40, "terrain_cost_s0")),
        Node::Input(BlankInput::new(0.0, 41, "terrain_cost_s1")),
        Node::Input(BlankInput::new(0.0, 42, "terrain_cost_s2")),
        Node::Input(BlankInput::new(0.0, 43, "terrain_cost_s3")),
        Node::Input(BlankInput::new(0.0, 44, "terrain_haz_s0")),
        Node::Input(BlankInput::new(0.0, 45, "terrain_haz_s1")),
        Node::Input(BlankInput::new(0.0, 46, "terrain_haz_s2")),
        Node::Input(BlankInput::new(0.0, 47, "terrain_haz_s3")),
        Node::Input(BlankInput::new(0.0, 48, "signal_s0")),
        Node::Input(BlankInput::new(0.0, 49, "signal_s1")),
        Node::Input(BlankInput::new(0.0, 50, "signal_s2")),
        Node::Input(BlankInput::new(0.0, 51, "signal_s3")),
    ];

    let output_nodes = vec![
        Node::Output(Sigmoid::new(0.0, 52, "thrust".to_string())),
        Node::Output(Sigmoid::new(0.0, 53, "turn_left".to_string())),
        Node::Output(Sigmoid::new(0.0, 54, "turn_right".to_string())),
        Node::Output(Sigmoid::new(0.0, 55, "speed".to_string())),
        Node::Output(Sigmoid::new(0.0, 56, "mate".to_string())),
        Node::Output(Sigmoid::new(0.0, 57, "eat".to_string())),
        Node::Output(Sigmoid::new(0.0, 58, "attack".to_string())),
        Node::Output(Sigmoid::new(0.0, 59, "mem_out".to_string())),
        Node::Output(Sigmoid::new(0.0, 60, "speak".to_string())),
        Node::Output(Sigmoid::new(0.0, 61, "voice".to_string())),
    ];

    App::new()
        .add_plugins(DefaultPlugins.set(window_plugin))
        .add_message::<InspectNet>()
        .insert_resource(BaseNodes {
            input_nodes,
            output_nodes,
        })
        .add_plugins(EguiPlugin::default())
        .add_plugins(SimulationPlugin)
        .add_plugins(NeuralNetPlugin)
        .insert_resource(ClearColor(Color::WHITE))
        .run();
}
