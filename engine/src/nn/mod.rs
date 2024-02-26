use std::fmt::Debug;

use dyn_clone::{clone_trait_object, DynClone};
use indexmap::IndexMap;
use log::debug;
use macros::{Name, SubTraits};
use rand::{random, Rng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{nn::util::connection_pair_exists, NeuronInfo};

pub mod reproduce;
pub mod util;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub enum Node {
    #[default]
    None,
    Input(Box<dyn InputNeuron>),
    Output(Box<dyn OutputNeuron>),
    Neuron(Box<dyn Neuron>),
}

impl NeuronInfo for Node {
    fn _type(&self) -> &'static str {
        match self {
            Node::None => unreachable!(),
            Node::Input(n) => n._type(),
            Node::Output(n) => n._type(),
            Node::Neuron(n) => n._type(),
        }
    }

    fn id(&self) -> usize {
        match self {
            Node::None => unreachable!(),
            Node::Input(n) => n.id(),
            Node::Output(n) => n.id(),
            Node::Neuron(n) => n.id(),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq, Clone, Copy)]
pub struct Edge {
    pub weight: f32,
    pub enabled: bool,
}

impl Edge {
    pub fn random() -> Edge {
        Edge {
            weight: random(),
            enabled: random(),
        }
    }
}

#[typetag::serde]
pub trait NeuronSubTraits: Debug + DynClone + NeuronInfo + Sync + Send {}

clone_trait_object!(InputNeuron);
#[typetag::serde]
pub trait InputNeuron: NeuronSubTraits {
    fn as_standard(&self) -> f32;
}

clone_trait_object!(OutputNeuron);
#[typetag::serde]
pub trait OutputNeuron: NeuronSubTraits {
    fn step(&self, edge: &Edge, input: f32) -> f32;
    fn finish_and_save(&mut self, partial: f32) -> f32;
}

clone_trait_object!(Neuron);
#[typetag::serde]
pub trait Neuron: NeuronSubTraits {
    fn step(&self, edge: &Edge, input: f32) -> f32;
    fn finish(&self, partial: f32) -> f32;
}

pub type GraphSize = u16;
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuralGraph {
    pub layers: Vec<Vec<GraphNode>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, Eq, Hash, Copy)]
pub struct GraphLocation {
    pub layer: GraphSize,
    pub node: GraphSize,
}

impl GraphLocation {
    pub fn new(layer: GraphSize, node: GraphSize) -> GraphLocation {
        GraphLocation { layer, node }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub struct GraphEdge {
    pub to: GraphLocation,
    pub value: Edge,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GraphNode {
    pub value: Node,
    pub connections: Vec<GraphEdge>,
}

impl GraphNode {
    pub fn new(value: Node) -> GraphNode {
        GraphNode {
            value,
            connections: Vec::new(),
        }
    }

    pub fn blank() -> GraphNode {
        GraphNode {
            value: Node::None,
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
        NeuralGraph { layers: Vec::new() }
    }

    pub fn add_layer_to_end(&mut self) -> GraphSize {
        self.layers.push(Vec::new());
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
        layer.push(graph_node);
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

    pub fn get_edge(&self, from: &GraphLocation, to: &GraphLocation) -> Option<&GraphEdge> {
        self.layers[from.layer as usize][from.node as usize]
            .connections
            .iter()
            .find(|x| x.to.eq(to))
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

    pub fn add_layer(&mut self, idx: GraphSize) {
        while idx as usize > self.layers.len() - 2 {
            self.layers.insert(idx as usize, Vec::new());
            for layer in self.layers.iter_mut() {
                for node in layer.iter_mut() {
                    for c in node.connections.iter_mut() {
                        if c.to.layer >= idx {
                            c.to.layer += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn create_node_at(&mut self, location: &GraphLocation, value: Node) {
        self.add_layer(location.layer);
        while location.node as usize > self.layers[location.layer as usize].len() {
            self.layers[location.layer as usize].push(GraphNode::blank());
        }
        self.layers[location.layer as usize].insert(location.node as usize, GraphNode::new(value));
    }

    pub fn prune(&mut self) {
        let mut to_delete = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (node_idx, node) in layer.iter().enumerate() {
                if let Node::None = node.value {
                    to_delete.push(GraphLocation::new(
                        layer_idx as GraphSize,
                        node_idx as GraphSize,
                    ))
                }
            }
        }

        to_delete.drain(..).for_each(|loc| {
            self.layers[loc.layer as usize].remove(loc.node as usize);
            self.layers.iter_mut().for_each(|layer| {
                layer.iter_mut().for_each(|node| {
                    let t = node
                        .connections
                        .drain(..)
                        .filter_map(|mut c| {
                            if c.to.layer == loc.layer && c.to.node > loc.node {
                                if c.to.node == 0 {
                                    None
                                } else {
                                    c.to.node -= 1;
                                    Some(c)
                                }
                            } else {
                                Some(c)
                            }
                        })
                        .collect::<Vec<_>>();
                    node.connections = t;
                })
            })
        });
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Net {
    pub graph: NeuralGraph,
    pub input_layer: GraphSize,
    pub output_layer: GraphSize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct BasicNeuron {
    bias: f32,
    id: usize,
}

#[typetag::serde]
impl Neuron for BasicNeuron {
    fn step(&self, edge: &Edge, input: f32) -> f32 {
        edge.weight * input
    }

    fn finish(&self, partial: f32) -> f32 {
        partial + self.bias
    }
}

impl Net {
    pub fn from_preserving_basic(from: &Net) -> Result<Net, NeuralGraphError> {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        for node in &from.graph.layers[from.input_layer as usize] {
            g.add_node(0, GraphNode::new(node.value.clone()))?;
        }
        g.add_layer_to_end();
        for node in &from.graph.layers[from.output_layer as usize] {
            g.add_node(1, GraphNode::new(node.value.clone()))?;
        }
        Ok(Net {
            graph: g,
            input_layer: 0,
            output_layer: 1,
        })
    }

    pub fn gen(input_nodes: &[Node], output_nodes: &[Node]) -> Result<Net, NeuralGraphError> {
        let mut rng = rand::thread_rng();
        let mut g = NeuralGraph::new();
        let input_layer = g.add_layer_to_end();
        for value in input_nodes {
            g.add_node(input_layer, GraphNode::new(value.clone()))?;
        }

        let mut num_internal_layers = rng.gen_range(0..5);
        let mut remove_layers = 0;
        let mut id = 0;
        for _ in 0..num_internal_layers {
            let num_nodes = rng.gen_range(0..10);
            if num_nodes == 0 {
                remove_layers += 1;
                continue;
            }
            let l = g.add_layer_to_end();
            for _ in 0..num_nodes {
                g.add_node(
                    l,
                    GraphNode::new(Node::Neuron(Box::new(BasicNeuron { bias: random(), id }))),
                )?;
                id += 1;
            }
        }
        num_internal_layers -= remove_layers;
        debug!("{num_internal_layers}");

        let output_layer = g.add_layer_to_end();
        for value in output_nodes {
            g.add_node(output_layer, GraphNode::new(value.clone()))?;
        }

        let num_connections = rng.gen_range(0..(4 * (num_internal_layers + 1)));
        let mut actual_connections = 0;
        let mut connection_pairs = Vec::new();
        while actual_connections != num_connections {
            let from_layer: GraphSize = rng.gen_range(0..=num_internal_layers + 1);
            let to_layer: GraphSize = rng.gen_range(0..=num_internal_layers + 1);
            if from_layer >= to_layer {
                continue;
            }

            let from_node = rng.gen_range(
                0..g.layers
                    .iter()
                    .skip(from_layer as usize)
                    .next()
                    .unwrap()
                    .len() as u16,
            );
            let to_node = rng.gen_range(
                0..g.layers
                    .iter()
                    .skip(to_layer as usize)
                    .next()
                    .unwrap()
                    .len() as u16,
            );
            let from = GraphLocation {
                layer: from_layer,
                node: from_node,
            };
            let to = GraphLocation {
                layer: to_layer,
                node: to_node,
            };
            debug!("{from:?}, {to:?}");
            if connection_pair_exists(&connection_pairs, &from, &to) {
                continue;
            }

            connection_pairs.push((from.clone(), to.clone()));
            g.add_edge(from, to, Edge::random())?;
            actual_connections += 1;
        }

        Ok(Net {
            graph: g,
            input_layer,
            output_layer,
        })
    }

    pub fn print_graph(&self) {
        for (num, layer) in self.graph.layers.iter().enumerate() {
            debug!("Layer {num}:");
            for (node_num, node) in layer.iter().enumerate() {
                debug!("Node {node_num} {:?}", node.value);

                if node.connections.len() > 0 {
                    debug!("Connections: ")
                }

                for c in node.connections.iter() {
                    debug!("{node_num} -> {c:?}");
                }
                debug!("");
            }
        }
    }

    pub fn tick(&mut self) {
        let mut next_layer_inputs: IndexMap<GraphSize, Vec<(GraphEdge, f32)>> = IndexMap::new();
        let input_layer = self.graph.layers.first().unwrap();
        for node in input_layer.iter() {
            for c in node.connections.iter() {
                if !c.value.enabled {
                    continue;
                }

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
                    Node::Neuron(n) => {
                        let step_value = n.finish(partial);
                        for c in node.connections.iter() {
                            if !c.value.enabled {
                                continue;
                            }

                            let entry = (c.clone(), step_value);
                            match next_layer_inputs.get_mut(&c.to.layer) {
                                Some(s) => s.push(entry),
                                None => {
                                    next_layer_inputs.insert(c.to.layer, vec![entry]);
                                }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            next_layer_inputs.shift_remove(&layer);
        }
    }

    pub fn reproduce(
        a: &Net,
        b: &Net,
        _types: &[impl reproduce::Reproducer],
        mut generator: impl reproduce::Generator,
    ) -> Result<Net, reproduce::ReproduceError> {
        let mut net = Net::from_preserving_basic(&a)?;
        loop {
            let (idx, done) = generator.generate(&a.graph, &b.graph, _types)?;

            _types[idx].reproduce(&a, &b, &mut net)?;

            if done {
                break;
            }
        }
        Ok(net)
    }
}
