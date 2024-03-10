use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    util::{self, ConnectionInfo},
    Net, NeuralGraph, NeuralGraphError,
};
use crate::nn::GraphLocation;

pub trait Reproducer {
    fn reproduce(&self, a: &Net, b: &Net, output: &mut Net) -> Result<(), ReproduceError>;
}

#[derive(Debug, Error)]
pub enum ReproduceError {
    #[error("Generator error")]
    GeneratorError(GeneratorError),
    #[error("Child net could not be created")]
    ChildNetError(NeuralGraphError),
}

impl From<GeneratorError> for ReproduceError {
    fn from(value: GeneratorError) -> Self {
        ReproduceError::GeneratorError(value)
    }
}

impl From<NeuralGraphError> for ReproduceError {
    fn from(value: NeuralGraphError) -> Self {
        ReproduceError::ChildNetError(value)
    }
}

pub trait Generator {
    /// Returns (index, done)
    fn generate(
        &mut self,
        a: &NeuralGraph,
        b: &NeuralGraph,
        _types: &[impl Reproducer],
    ) -> Result<(usize, bool), GeneratorError>;
}

#[derive(Debug, Error)]
pub enum GeneratorError {}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Crossover;

#[derive(Debug)]
struct NodeReplacement {
    from: GraphLocation,
    to: GraphLocation,
}

impl Reproducer for Crossover {
    /// Crossover between two networks
    fn reproduce(&self, a: &Net, b: &Net, output: &mut Net) -> Result<(), ReproduceError> {
        let common_elements = util::intersection(a, b);
        let mut node_replacements = Vec::new();

        for item in common_elements.clone() {
            match item {
                util::AlignedItem::Node(a_node, b_node) => {
                    let mut rng = rand::thread_rng();
                    let (g, loc) = if rng.gen::<f32>() < 0.5 {
                        node_replacements.push(NodeReplacement {
                            from: b_node.clone(),
                            to: a_node.clone(),
                        });
                        (&a.graph, a_node)
                    } else {
                        node_replacements.push(NodeReplacement {
                            from: a_node.clone(),
                            to: b_node.clone(),
                        });
                        (&b.graph, b_node)
                    };

                    output
                        .graph
                        .create_node_at(&loc, g.get_node(&loc).unwrap().value.clone());
                }
                _ => {}
            }
        }

        for item in common_elements {
            match item {
                util::AlignedItem::Edge { data, _type } => {
                    let mut rng = rand::thread_rng();

                    let mut choose_graph = |a_conn: ConnectionInfo,
                                            b_conn: ConnectionInfo|
                     -> (ConnectionInfo, &NeuralGraph) {
                        if rng.gen::<f32>() < 0.5 {
                            (a_conn, &a.graph)
                        } else {
                            (b_conn, &b.graph)
                        }
                    };

                    let (conn, g) = choose_graph(data.0, data.1);
                    let (graph_specific_conn, replaced_conn) = {
                        let (conn, from, to) = match _type {
                            util::EdgeType::Incoming => {
                                match node_replacements.iter().find(|x| x.from == conn.to) {
                                    Some(f) => (conn.clone(), conn.from, f.to.clone()),
                                    None => (conn.clone(), conn.from, conn.to),
                                }
                            }
                            util::EdgeType::Outgoing => {
                                match node_replacements.iter().find(|x| x.from == conn.from) {
                                    Some(f) => (conn.clone(), f.to.clone(), conn.to),
                                    None => (conn.clone(), conn.from, conn.to),
                                }
                            }
                        };

                        (conn, (from, to))
                    };

                    output.graph.add_edge(
                        replaced_conn.0.clone(),
                        replaced_conn.1.clone(),
                        g.get_edge(&graph_specific_conn.from, &graph_specific_conn.to)
                            .unwrap()
                            .value
                            .clone(),
                    )?;
                }
                _ => {}
            }
        }

        output.graph.prune();
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DefaultIterator(usize);

impl Generator for DefaultIterator {
    fn generate(
        &mut self,
        _: &NeuralGraph,
        _: &NeuralGraph,
        _types: &[impl Reproducer],
    ) -> Result<(usize, bool), GeneratorError> {
        self.0 += 1;
        Ok((self.0 - 1, self.0 >= _types.len()))
    }
}

impl DefaultIterator {
    pub fn new() -> DefaultIterator {
        DefaultIterator(0)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        activations::Sigmoid,
        nn::{
            reproduce::{Crossover, Reproducer},
            BasicNeuron, Edge, GraphLocation, GraphNode, Net, Node,
        },
    };

    use crate::nn::test_requirements::*;

    #[test]
    #[rustfmt::skip]
    fn basic_test() {
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(0.0, 0))),
            Node::Input(Box::new(BlankInput::new(0.0, 1))),
            Node::Input(Box::new(BlankInput::new(0.0, 2))),
        ];

        let output_nodes = [
            Node::Output(Sigmoid::new(0.0, 3)),
            Node::Output(Sigmoid::new(0.0, 4)),
        ];

        let mut graph_a = create_graph(&input_nodes, &output_nodes);

        graph_a.add_layer(1);
        graph_a.add_node(1,GraphNode::new(Node::Neuron(Box::new(BasicNeuron { bias: 0.0, id: 5 }))),).unwrap();
        graph_a.add_node(1,GraphNode::new(Node::Neuron(Box::new(TestNeuronA { value: 0.0, id: 6 }))),).unwrap();

        graph_a.add_edge(GraphLocation::new(0, 0),GraphLocation::new(1, 0),Edge::default(),).unwrap();
        graph_a.add_edge(GraphLocation::new(0, 1),GraphLocation::new(1, 0),Edge::default(),).unwrap();
        graph_a.add_edge(GraphLocation::new(0, 2),GraphLocation::new(1, 0),Edge::default(),).unwrap();
        graph_a.add_edge(GraphLocation::new(0, 2),GraphLocation::new(1, 1),Edge::default(),).unwrap();

        graph_a.add_edge(GraphLocation::new(1, 0),GraphLocation::new(2, 0),Edge::default(),).unwrap();
        graph_a.add_edge(GraphLocation::new(1, 0),GraphLocation::new(2, 1),Edge::default(),).unwrap();

        graph_a.add_edge(GraphLocation::new(1, 1),GraphLocation::new(2, 1),Edge::default(),).unwrap();

        let a = Net {
            graph: graph_a,
            input_layer: 0,
            output_layer: 2,
        };

        let mut graph_b = create_graph(&input_nodes, &output_nodes);

        graph_b.add_layer(1);
        graph_b.add_node(1,GraphNode::new(Node::Neuron(Box::new(TestNeuronB { value: 0.0, id: 6 }))),).unwrap();
        graph_b.add_node(1,GraphNode::new(Node::Neuron(Box::new(BasicNeuron { bias: 0.0, id: 5 }))),).unwrap();

        graph_b.add_edge(GraphLocation::new(0, 0),GraphLocation::new(1, 1),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(0, 1),GraphLocation::new(1, 1),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(0, 2),GraphLocation::new(1, 1),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(0, 0),GraphLocation::new(1, 0),Edge::default(),).unwrap();

        graph_b.add_edge(GraphLocation::new(1, 1),GraphLocation::new(2, 0),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(1, 1),GraphLocation::new(2, 1),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(1, 0),GraphLocation::new(2, 0),Edge::default(),).unwrap();
        graph_b.add_edge(GraphLocation::new(1, 0),GraphLocation::new(2, 1),Edge::default(),).unwrap();

        let b = Net {
            graph: graph_b,
            input_layer: 0,
            output_layer: 2,
        };

        let mut output = Net::from_preserving_basic(&a).expect("Could not create neural net from A");
        Crossover::default().reproduce(&a, &b, &mut output).expect("Could not crossover");

        let mut compose_output = create_graph(&input_nodes, &output_nodes);

        compose_output.add_layer(1);
        compose_output.add_node(1, GraphNode::new(Node::Neuron(Box::new(BasicNeuron { bias: 0.0, id: 5 })))).unwrap();

        compose_output.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();
        compose_output.add_edge(GraphLocation::new(0, 1), GraphLocation::new(1, 0), Edge::default()).unwrap();
        compose_output.add_edge(GraphLocation::new(0, 2), GraphLocation::new(1, 0), Edge::default()).unwrap();

        compose_output.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 0), Edge::default()).unwrap();
        compose_output.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 1), Edge::default()).unwrap();

        // let sim= Simulation {
        //     nets: vec![Nn { net: output.clone(), node_positions: vec![] }],
        //     input_nodes: input_nodes.to_vec(),
        //     output_nodes: output_nodes.to_vec(),
        // };

        // std::fs::write("reproduce.bin", bincode::serialize(&sim).unwrap()).unwrap();

        assert_eq!(bincode::serialize(&compose_output).unwrap(), bincode::serialize(&output.graph).unwrap());
    }
}
