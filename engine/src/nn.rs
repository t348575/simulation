use std::{collections::LinkedList, fmt::Debug};

use dyn_clone::{clone_trait_object, DynClone};
use erased_serde::serialize_trait_object;
use indexmap::IndexMap;
use rand::{random, Rng};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Serialize, Clone)]
pub enum Node {
    Input(Box<dyn InputNeuron>),
    Output(Box<dyn OutputNeuron>),
    Neuron(Box<dyn Neuron>),
}

#[derive(Debug, Default, Serialize, PartialEq, Clone)]
pub struct Edge(pub f32);

impl Edge {
    pub fn random() -> Edge {
        Edge(random())
    }
}

serialize_trait_object!(InputNeuron);
clone_trait_object!(InputNeuron);
pub trait InputNeuron: Debug + DynClone + erased_serde::Serialize {
    fn as_standard(&self) -> f32;
}

serialize_trait_object!(OutputNeuron);
clone_trait_object!(OutputNeuron);
pub trait OutputNeuron: Debug + DynClone + erased_serde::Serialize {
    fn step(&self, edge: &Edge, input: f32) -> f32;
    fn finish_and_save(&mut self, partial: f32) -> f32;
}

serialize_trait_object!(Neuron);
clone_trait_object!(Neuron);
pub trait Neuron: Debug + DynClone + erased_serde::Serialize {
    fn step(&self, edge: &Edge, input: f32) -> f32;
}

type GraphSize = u16;
pub struct NeuralGraph {
    layers: LinkedList<LinkedList<GraphNode>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphLocation {
    layer: GraphSize,
    node: GraphSize,
}

impl GraphLocation {
    pub fn new(layer: GraphSize, node: GraphSize) -> GraphLocation {
        GraphLocation { layer, node }
    }
}

#[derive(Debug, PartialEq, Serialize, Clone)]
pub struct GraphEdge {
    to: GraphLocation,
    value: Edge,
}

#[derive(Debug, Serialize, Clone)]
pub struct GraphNode {
    value: Node,
    connections: Vec<GraphEdge>,
}

impl GraphNode {
    pub fn blank(value: Node) -> GraphNode {
        GraphNode {
            value,
            connections: Vec::new(),
        }
    }
}

#[derive(Error, Debug)]
pub enum NeuralGraphError {
    #[error("Invalid layer number, last layer {0}")]
    LayerNotFound(GraphSize),
    #[error("Node not found at layer layer: {0:?}")]
    NodeNotFound(GraphLocation),
    #[error("Connection already exists from {from:?} to {to:?}")]
    ConnectionExists {
        from: GraphLocation,
        to: GraphLocation,
    },
}

impl NeuralGraph {
    pub fn new() -> NeuralGraph {
        NeuralGraph {
            layers: LinkedList::new(),
        }
    }

    pub fn add_layer_to_end(&mut self) -> GraphSize {
        self.layers.push_back(LinkedList::new());
        self.layers.len() as GraphSize - 1
    }

    pub fn add_node(
        &mut self,
        layer_num: GraphSize,
        graph_node: GraphNode,
    ) -> Result<GraphLocation, NeuralGraphError> {
        if layer_num > self.layers.len() as GraphSize {
            return Err(NeuralGraphError::LayerNotFound(
                self.layers.len() as GraphSize
            ));
        }

        let layer = self
            .layers
            .iter_mut()
            .skip(layer_num as usize)
            .next()
            .unwrap();
        layer.push_back(graph_node);
        Ok(GraphLocation {
            layer: layer_num,
            node: layer.len() as GraphSize - 1,
        })
    }

    pub fn get_node_mut(&mut self, location: &GraphLocation) -> Option<&mut GraphNode> {
        match self.layers.iter_mut().skip(location.layer as usize).next() {
            Some(l) => l.iter_mut().skip(location.node as usize).next(),
            None => None,
        }
    }

    pub fn get_node(&self, location: &GraphLocation) -> Option<&GraphNode> {
        match self.layers.iter().skip(location.layer as usize).next() {
            Some(l) => l.iter().skip(location.node as usize).next(),
            None => None,
        }
    }

    pub fn add_edge(
        &mut self,
        from: GraphLocation,
        to: GraphLocation,
        value: Edge,
    ) -> Result<(), NeuralGraphError> {
        _ = self
            .get_node(&to)
            .ok_or(NeuralGraphError::NodeNotFound(to.clone()))?;
        let from_node = self
            .get_node_mut(&from)
            .ok_or(NeuralGraphError::NodeNotFound(from.clone()))?;

        if from_node
            .connections
            .iter()
            .filter(|x| x.to.eq(&to))
            .count()
            != 0
        {
            return Err(NeuralGraphError::ConnectionExists { from, to });
        }

        from_node.connections.push(GraphEdge { to, value });
        Ok(())
    }
}

pub struct Net {
    graph: NeuralGraph,
    input_layer: GraphSize,
    output_layer: GraphSize,
}

impl Net {
    pub fn gen(input_nodes: &[Node], output_nodes: &[Node]) -> Result<Net, NeuralGraphError> {
        let mut g = NeuralGraph::new();
        let input_layer = g.add_layer_to_end();
        for value in input_nodes {
            g.add_node(input_layer, GraphNode::blank(value.clone()))?;
        }

        let output_layer = g.add_layer_to_end();
        for value in output_nodes {
            g.add_node(output_layer, GraphNode::blank(value.clone()))?;
        }

        let mut rng = rand::thread_rng();
        let num_connections = rng.gen_range(0..=8);
        let mut connection_pairs = Vec::new();
        for _ in 0..num_connections {
            let from: GraphSize = rng.gen_range(0..4);
            let to: GraphSize = rng.gen_range(0..5);

            if connection_pair_exists(&connection_pairs, from, to) {
                continue;
            }

            connection_pairs.push((from, to));
            g.add_edge(
                GraphLocation::new(input_layer, from),
                GraphLocation::new(output_layer, to),
                Edge::random(),
            )?;
        }

        Ok(Net {
            graph: g,
            input_layer,
            output_layer,
        })
    }

    pub fn print_graph(&self) {
        for (num, layer) in self.graph.layers.iter().enumerate() {
            println!("Layer {num}:");
            for (node_num, node) in layer.iter().enumerate() {
                println!("Node {node_num} {:?}", node.value);

                if node.connections.len() > 0 {
                    println!("Connections: ")
                }

                for c in node.connections.iter() {
                    println!("{node_num} -> {c:?}");
                }
                println!("");
            }
        }
    }

    pub fn tick(&mut self) {
        let mut next_layer_inputs: IndexMap<GraphSize, Vec<(GraphEdge, f32)>> = IndexMap::new();
        let input_layer = self.graph.layers.front().unwrap();
        for node in input_layer.iter() {
            for c in node.connections.iter() {
                if let Node::Input(i) = &node.value {
                    let entry = (c.clone(), i.as_standard());
                    match next_layer_inputs.get_mut(&c.to.layer) {
                        Some(s) => s.push(entry),
                        None => {
                            next_layer_inputs.insert(c.to.layer, vec![entry]);
                        }
                    }
                }
            }
        }

        while next_layer_inputs.len() != 0 {
            next_layer_inputs.sort_keys();
            let mut partials: Vec<(GraphLocation, f32)> = Vec::new();
            let layer = {
                let (got_layer, compute_step) = next_layer_inputs.first_mut().unwrap();
                for (edge, input) in compute_step {
                    let node = self.graph.get_node(&edge.to).unwrap();

                    let ele = match partials.iter().position(|x| x.0.eq(&edge.to)) {
                        Some(s) => &mut partials[s].1,
                        None => {
                            partials.push((edge.to.clone(), 0.0));
                            &mut partials.last_mut().unwrap().1
                        }
                    };
                    match &node.value {
                        Node::Output(o) => *ele += o.step(&edge.value, *input),
                        Node::Neuron(n) => *ele += n.step(&edge.value, *input),
                        _ => unreachable!(),
                    }
                }
                *got_layer
            };

            for (node, partial) in partials {
                let node = self.graph.get_node_mut(&node).unwrap();
                match &mut node.value {
                    Node::Output(o) => {
                        o.finish_and_save(partial);
                    }
                    _ => unreachable!(),
                }
            }
            next_layer_inputs.remove(&layer);
        }
    }
}

fn connection_pair_exists(
    pairs: &[(GraphSize, GraphSize)],
    from: GraphSize,
    to: GraphSize,
) -> bool {
    pairs
        .iter()
        .position(|x| (x.0 == from && x.1 == to) || (x.0 == to && x.1 == from))
        .map_or_else(|| false, |_| true)
}
