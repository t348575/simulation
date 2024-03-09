use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use bevy_vector_shapes::prelude::*;
use engine::{activations::Sigmoid, nn::Node as NnNode};

use crate::{inputs::*, outputs::*};

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
        let input_nodes = vec![
            NnNode::Input(Hunger::new(0.0, 0)),
            NnNode::Input(Age::new(0.0, 1)),
            NnNode::Input(Health::new(100.0, 2)),
            NnNode::Input(Speed::new(0.0, 3)),
        ];

        let output_nodes = vec![
            NnNode::Output(Sigmoid::new(0.0, 4)), // ::<Forward>
            NnNode::Output(Sigmoid::new(0.0, 5)), // ::<Backward>
            NnNode::Output(Sigmoid::new(0.0, 6)), // ::<Left>
            NnNode::Output(Sigmoid::new(0.0, 7)), // ::<Right>
            NnNode::Output(Sigmoid::new(0.0, 8)), // ::<OutputSpeed>
        ];

        // let nets = (0..2).map(|_| Nn {
        //     net: Net::gen(&input_nodes, &output_nodes).unwrap(),
        //     node_positions: Vec::new(),
        //     input_nodes: input_nodes.clone(),
        //     output_nodes: output_nodes.clone(),
        // }).collect::<Vec<_>>();

        app.add_state::<InspectWindowState>()
            .init_resource::<InspectInfo>()
            .init_resource::<WindowInfo>()
            .init_resource::<ControlPanel>()
            .insert_resource(Simulation {
                nets: vec![],
                input_nodes,
                output_nodes,
            })
            .add_plugins(EguiPlugin)
            .insert_resource(ClearColor(Color::WHITE))
            .add_plugins(Shape2dPlugin::default())
            .add_systems(Startup, setup)
            // .add_systems(Update, gen_new_graph)
            .configure_sets(Update, IwSet::Events.before(IwSet::Window))
            .add_systems(Update, toggle_inspect_window.in_set(IwSet::Events))
            .add_systems(Update, control_panel)
            .add_systems(
                Update,
                inspect_window
                    .in_set(IwSet::Window)
                    .run_if(in_state(InspectWindowState::Display)),
            );
    }
}
