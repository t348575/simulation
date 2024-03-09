use engine::{
    activations::Sigmoid,
    nn::{
        reproduce::{Crossover, DefaultIterator},
        GraphLocation, Net, Node,
    },
};
use inputs::*;
use outputs::*;
use serde::{Deserialize, Serialize};

mod inputs;
mod outputs;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Simulation {
    pub nets: Vec<Nn>,
    pub input_nodes: Vec<Node>,
    pub output_nodes: Vec<Node>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Nn {
    pub net: Net,
    pub node_positions: Vec<(GraphLocation, NodePosition)>,
}

fn main() {
    // let input_nodes = [
    //     Node::Input(Hunger::new(0.0)),
    //     Node::Input(Age::new(0)),
    //     Node::Input(Health::new(100)),
    //     Node::Input(Speed::new(0.0)),
    // ];

    // let output_nodes = vec![
    //     Node::Output(Sigmoid::new(0.0)), // ::<Forward>
    //     Node::Output(Sigmoid::new(0.0)), // ::<Backward>
    //     Node::Output(Sigmoid::new(0.0)), // ::<Left>
    //     Node::Output(Sigmoid::new(0.0)), // ::<Right>
    //     Node::Output(Sigmoid::new(0.0)), // ::<OutputSpeed>
    // ];

    // let mut nn = Net::gen(&input_nodes, &output_nodes).unwrap();

    let sim: Simulation =
        bincode::deserialize(&std::fs::read("../visualize-nn/nets.bin").unwrap()).unwrap();

    let nn_a = sim.nets[6].net.clone();
    let nn_b = sim.nets[7].net.clone();
    // let r = Net::reproduce(&nn_a, &nn_b, &[Crossover::new()], DefaultIterator::new()).unwrap();
    let aligned = engine::nn::util::intersection(&nn_a, &nn_b);
    println!("{:#?}", aligned);
    // r.print_graph();
    // nn.print_graph();
    // for _ in 0..100 {
    //     nn.tick();
    // }
    // let last = nn.graph.layers.iter().last().unwrap();
    // for item in last {
    //     println!("{item:?}");
    // }
}
