use engine::{nn::{Net, Node}, activations::Sigmoid};
use inputs::*;
use outputs::*;

mod inputs;
mod outputs;

fn main() {
    let input_nodes = [
        Node::Input(Hunger::new(0.0)),
        Node::Input(Age::new(0)),
        Node::Input(Health::new(100)),
        Node::Input(Speed::new(0.0)),
    ];

    let output_nodes = vec![
        Node::Output(Sigmoid::<Forward>::new(0.0)),
        Node::Output(Sigmoid::<Backward>::new(0.0)),
        Node::Output(Sigmoid::<Left>::new(0.0)),
        Node::Output(Sigmoid::<Right>::new(0.0)),
        Node::Output(Sigmoid::<OutputSpeed>::new(0.0)),
    ];

    let mut nn = Net::gen(&input_nodes, &output_nodes).unwrap();
    nn.print_graph();
    nn.tick();
}