use bevy::{
    prelude::*,
    window::{ExitCondition, PresentMode, WindowMode},
};
use bevy_egui::EguiPlugin;
use engine::{activations::Sigmoid, nn::Node};
use inputs::*;
use net::{resources::InspectNet, NeuralNetPlugin};
use sim::SimulationPlugin;

pub mod inputs;
mod net;
pub mod outputs;
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
        Node::Input(Age::new(0, 1)),
        Node::Input(Health::new(100.0, 2)),
        Node::Input(Speed::new(0.0, 3)),
        Node::Input(BlankInput::new(0.0, 4)),  // food_dx
        Node::Input(BlankInput::new(0.0, 5)),  // food_dy
        Node::Input(BlankInput::new(1.0, 6)),  // food_distance
        Node::Input(BlankInput::new(0.0, 7)),  // mate_dx
        Node::Input(BlankInput::new(0.0, 8)),  // mate_dy
        Node::Input(BlankInput::new(1.0, 9)),  // mate_distance
        Node::Input(BlankInput::new(0.0, 10)), // wall_left
        Node::Input(BlankInput::new(0.0, 11)), // wall_right
        Node::Input(BlankInput::new(0.0, 12)), // wall_bottom
        Node::Input(BlankInput::new(0.0, 13)), // wall_top
        Node::Input(BlankInput::new(0.0, 14)), // poison_dx
        Node::Input(BlankInput::new(0.0, 15)), // poison_dy
        Node::Input(BlankInput::new(1.0, 16)), // poison_distance
    ];

    let output_nodes = vec![
        Node::Output(Sigmoid::new(0.0, 17, "forward".to_string())), // ::<Forward>
        Node::Output(Sigmoid::new(0.0, 18, "backward".to_string())), // ::<Backward>
        Node::Output(Sigmoid::new(0.0, 19, "left".to_string())),    // ::<Left>
        Node::Output(Sigmoid::new(0.0, 20, "right".to_string())),   // ::<Right>
        Node::Output(Sigmoid::new(0.0, 21, "output_speed".to_string())), // ::<OutputSpeed>
        Node::Output(Sigmoid::new(0.0, 22, "mate".to_string())),    // ::<Mate>
        Node::Output(Sigmoid::new(0.0, 23, "eat".to_string())),     // ::<Eat>
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
