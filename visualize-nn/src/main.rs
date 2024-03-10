use bevy::{prelude::*, render::{camera::RenderTarget, view::RenderLayers}, window::{PrimaryWindow, WindowMode, WindowRef}};
use bevy_egui::EguiPlugin;
use engine::{activations::Sigmoid, nn::Node};
use inputs::*;
use main_menu::MainMenuPlugin;
use net::NeuralNetPlugin;
use sim::SimulationPlugin;

pub mod inputs;
mod main_menu;
mod net;
pub mod outputs;
mod sim;

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TabState {
    #[default]
    MainMenu,
    Simulation,
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InspectWindowState {
    #[default]
    None,
    Display,
}

#[derive(Resource, Debug)]
pub struct BaseNodes {
    pub input_nodes: Vec<Node>,
    pub output_nodes: Vec<Node>,
}

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

    let input_nodes = vec![
        Node::Input(Hunger::new(0.0, 0)),
        Node::Input(Age::new(0, 1)),
        Node::Input(Health::new(100.0, 2)),
        Node::Input(Speed::new(0.0, 3)),
    ];

    let output_nodes = vec![
        Node::Output(Sigmoid::new(0.0, 4, "forward".to_string())), // ::<Forward>
        Node::Output(Sigmoid::new(0.0, 5, "backward".to_string())), // ::<Backward>
        Node::Output(Sigmoid::new(0.0, 6, "left".to_string())), // ::<Left>
        Node::Output(Sigmoid::new(0.0, 7, "right".to_string())), // ::<Right>
        Node::Output(Sigmoid::new(0.0, 8, "output_speed".to_string())), // ::<OutputSpeed>
    ];

    App::new()
        .init_state::<TabState>()
        .init_state::<InspectWindowState>()
        .insert_resource(BaseNodes {
            input_nodes,
            output_nodes,
        })
        .add_plugins(DefaultPlugins.set(window_plugin))
        .add_systems(Startup, setup)
        .add_plugins(EguiPlugin)
        .add_plugins(MainMenuPlugin)
        .add_plugins(SimulationPlugin)
        .add_plugins(NeuralNetPlugin)
        .insert_resource(ClearColor(Color::WHITE))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle {
        camera: Camera {
            target: RenderTarget::Window(WindowRef::Primary),
            ..Default::default()
        },
        ..Default::default()
    });
}
