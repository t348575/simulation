use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{Edge, GraphEdge, GraphLocation, Net, NeuralGraph, NeuralGraphError, Neuron, Node};

pub trait LinkMutator {
    fn mutate(&self, net: &mut Net) -> Result<bool, MutateError>;
}

pub type MutationSelector = dyn Fn(usize) -> usize;
pub trait NeuronMutator {
    fn mutate(
        &self,
        net: &mut Net,
        neurons: &[Box<dyn Neuron>],
        selector: &MutationSelector,
    ) -> Result<bool, MutateError>;
}

pub trait Generator {
    /// Returns (is_link, index, done)
    fn generate(
        &mut self,
        g: &NeuralGraph,
        link_mutators: &[Box<dyn LinkMutator>],
        neuron_mutators: &[Box<dyn NeuronMutator>],
    ) -> Result<(bool, usize, bool), GeneratorError>;
}

impl From<GeneratorError> for MutateError {
    fn from(value: GeneratorError) -> Self {
        MutateError::GeneratorError(value)
    }
}

#[derive(Debug, Error)]
pub enum GeneratorError {}

#[derive(Debug, Error)]
pub enum MutateError {
    #[error("Generator error")]
    GeneratorError(GeneratorError),
    #[error("Error adding link")]
    AddLinkError(NeuralGraphError),
    #[error("Neuron selector function returned outside of size: {0}")]
    NeuronSelectorOutOfRange(usize),
    #[error("Error removing link")]
    RemoveLinkError(NeuralGraphError),
    #[error("Error adding neuron")]
    AddNeuronError(NeuralGraphError),
    #[error("Error removing neuron")]
    RemoveNeuronError(NeuralGraphError),
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AddEdge;

impl LinkMutator for AddEdge {
    fn mutate(&self, net: &mut Net) -> Result<bool, MutateError> {
        let input = match net.graph.random_input_or_hidden() {
            Some(s) => s,
            None => return Ok(false),
        };
        let output = match net.graph.random_output_or_hidden(Some(input.layer)) {
            Some(s) => s,
            None => return Ok(false),
        };

        if let Some(e) = net.graph.get_edge_mut(&input, &output) {
            e.value.enabled = true;
            return Ok(true);
        }

        net.graph
            .add_edge(input, output, Edge::random())
            .map_err(|err| MutateError::AddLinkError(err))?;
        if net.graph.has_cycle(Some(input)) {
            net.graph.remove_edge(&input, &output);
            return Ok(false);
        }

        Ok(true)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RemoveEdge;

impl LinkMutator for RemoveEdge {
    fn mutate(&self, net: &mut Net) -> Result<bool, MutateError> {
        if let Some((pos, e)) = net.graph.random_edge_mut() {
            let to = e.to.clone();
            return Ok(net.graph.remove_edge(&pos, &to));
        }
        Ok(false)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AddNeuron;

impl AddNeuron {
    fn run(
        &self,
        from: GraphLocation,
        link_to_split: GraphEdge,
        net: &mut Net,
        neuron: Box<dyn Neuron>,
    ) -> Result<(), MutateError> {
        net.graph.remove_edge(&from, &link_to_split.to);
        let layer = from.layer + 1;
        let mut to_layer_new = link_to_split.to.layer;
        if link_to_split.to.layer == from.layer + 1 {
            net.graph.add_layer(from.layer + 1);
            to_layer_new = layer + 1;
        }

        let new_node_pos = net.graph.push_node_at(layer, Node::Neuron(neuron));
        net.graph
            .add_edge(from, new_node_pos, link_to_split.value)
            .map_err(|err| MutateError::AddNeuronError(err))?;
        net.graph
            .add_edge(
                new_node_pos,
                GraphLocation::new(to_layer_new, link_to_split.to.node),
                Edge::default(),
            )
            .map_err(|err| MutateError::AddNeuronError(err))?;
        Ok(())
    }
}

impl NeuronMutator for AddNeuron {
    fn mutate(
        &self,
        net: &mut Net,
        neurons: &[Box<dyn Neuron>],
        selector: &MutationSelector,
    ) -> Result<bool, MutateError> {
        let (from, link_to_split) = match net.graph.random_edge_mut() {
            Some(s) => s,
            None => return Ok(false),
        };

        let neuron = selector(neurons.len());
        if neuron >= neurons.len() {
            return Err(MutateError::NeuronSelectorOutOfRange(neurons.len()));
        }

        self.run(from, link_to_split.clone(), net, neurons[neuron].clone())?;

        Ok(true)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RemoveNeuron;

impl NeuronMutator for RemoveNeuron {
    fn mutate(
        &self,
        net: &mut Net,
        _: &[Box<dyn Neuron>],
        _: &MutationSelector,
    ) -> Result<bool, MutateError> {
        if net.graph.layers.len() == 2 {
            return Ok(false);
        }

        match net.graph.random_hidden() {
            Some(n) => {
                net.graph.removed_node(n);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        activations::Sigmoid,
        nn::{
            test_requirements::{create_graph, BlankInput},
            BasicNeuron, Edge, GraphEdge, GraphLocation, GraphNode, Net, Node,
        },
    };

    use super::AddNeuron;

    #[test]
    #[rustfmt::skip]
    fn add_neuron() {
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(0.0, 0))),
            Node::Input(Box::new(BlankInput::new(0.0, 1))),
        ];

        let output_nodes = [
            Node::Output(Sigmoid::new(0.0, 2, "a".to_owned())),
            Node::Output(Sigmoid::new(0.0, 3, "b".to_owned())),
        ];

        let mut graph_mutation = create_graph(&input_nodes, &output_nodes);
        let mut graph_verify = create_graph(&input_nodes, &output_nodes);

        graph_mutation.add_layer(1);
        graph_verify.add_layer(1);
        graph_verify.add_layer(1);
        graph_mutation.add_node(1, GraphNode::new(Node::Neuron(Box::new(BasicNeuron::default())))).unwrap();
        graph_verify.add_node(1, GraphNode::new(Node::Neuron(Box::new(BasicNeuron::default())))).unwrap();
        graph_verify.add_node(2, GraphNode::new(Node::Neuron(Box::new(BasicNeuron::default())))).unwrap();

        let e = Edge {
            weight: 1.3,
            enabled: true,
        };
        graph_verify.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), e.clone()).unwrap();
        graph_verify.add_edge(GraphLocation::new(0, 1), GraphLocation::new(2, 0), Edge::default()).unwrap();

        graph_verify.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 0), Edge::default()).unwrap();

        graph_verify.add_edge(GraphLocation::new(2, 0), GraphLocation::new(3, 0), Edge::default()).unwrap();
        graph_verify.add_edge(GraphLocation::new(2, 0), GraphLocation::new(3, 1), Edge::default()).unwrap();

        graph_mutation.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();
        graph_mutation.add_edge(GraphLocation::new(0, 1), GraphLocation::new(1, 0), Edge::default()).unwrap();

        graph_mutation.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 0), Edge::default()).unwrap();
        graph_mutation.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 1), Edge::default()).unwrap();

        let mut m = Net {
            graph: graph_mutation,
            input_layer: 0,
            output_layer: 2,
        };

        let v = Net {
            graph: graph_verify,
            input_layer: 0,
            output_layer: 3
        };

        AddNeuron{}.run(GraphLocation::new(0, 0), GraphEdge {
            to: GraphLocation::new(1, 0),
            value: e.clone(),
        }, &mut m, Box::new(BasicNeuron::default())).unwrap();

        // let sim = Simulation {
        //     nets: vec![Nn { net: m, node_positions: vec![] }, Nn {net: v, node_positions: vec![]}],
        //     input_nodes: input_nodes.to_vec(),
        //     output_nodes: output_nodes.to_vec(),
        // };

        // std::fs::write("mutate.bin", bincode::serialize(&sim).unwrap()).unwrap();

        assert_eq!(bincode::serialize(&m.graph).unwrap(), bincode::serialize(&v.graph).unwrap());
    }
}
