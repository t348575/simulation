use std::fmt::Debug;

use dyn_clone::{clone_trait_object, DynClone};
use hashbrown::HashSet;
use indexmap::IndexMap;
use log::debug;
use macros::{DNeuronInfo, SubTraits};
use rand::{random, seq::IteratorRandom, Rng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{nn::util::connection_pair_exists, NeuronInfo};

use self::mutate::MutationSelector;

pub mod mutate;
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
    fn set_value(&mut self, value: f32);
}

clone_trait_object!(OutputNeuron);
#[typetag::serde]
pub trait OutputNeuron: NeuronSubTraits {
    fn step(&self, edge: &Edge, input: f32) -> f32;
    fn finish_and_save(&mut self, partial: f32) -> f32;
    fn value(&self) -> f32;
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

    pub fn get_edge_mut(
        &mut self,
        from: &GraphLocation,
        to: &GraphLocation,
    ) -> Option<&mut GraphEdge> {
        self.layers[from.layer as usize][from.node as usize]
            .connections
            .iter_mut()
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

    pub fn remove_edge(&mut self, from: &GraphLocation, to: &GraphLocation) -> bool {
        let node = &mut self.layers[from.layer as usize][from.node as usize];
        let before = node.connections.len();
        node.connections.retain(|x| !x.to.eq(to));
        node.connections.len() == before
    }

    pub fn removed_node(&mut self, remove: GraphLocation) {
        self.layers[remove.layer as usize].remove(remove.node as usize);

        self.layers.iter_mut().for_each(|layer| {
            layer
                .iter_mut()
                .for_each(|node| node.connections.retain(|c| c.to.ne(&remove)))
        })
    }

    pub fn add_layer(&mut self, idx: GraphSize) {
        let mut first = true;
        while idx as usize > self.layers.len() - 2 || first {
            first = false;
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

    /// Grows the graph by inserting empty layers just before the output until `idx` is a
    /// valid hidden-layer address. Unlike `add_layer`, this never inserts at `idx` itself —
    /// it only ensures the slot exists so a node can be placed there.
    pub fn ensure_layer_exists(&mut self, idx: GraphSize) {
        while idx as usize >= self.layers.len().saturating_sub(1) {
            let before_output = self.layers.len().saturating_sub(1).max(1);
            let shifted = before_output as GraphSize;
            self.layers.insert(before_output, Vec::new());
            for layer in self.layers.iter_mut() {
                for node in layer.iter_mut() {
                    for c in node.connections.iter_mut() {
                        if c.to.layer >= shifted {
                            c.to.layer += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn push_node_at(&mut self, layer: GraphSize, value: Node) -> GraphLocation {
        self.layers[layer as usize].push(GraphNode::new(value));
        GraphLocation::new(layer, (self.layers[layer as usize].len() - 1) as GraphSize)
    }

    pub fn create_node_at(&mut self, location: &GraphLocation, value: Node) {
        self.ensure_layer_exists(location.layer);
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

        // Remove higher node indices first within each layer so earlier removals
        // don't shift indices and invalidate subsequent removals in the same layer.
        to_delete.sort_by(|a, b| a.layer.cmp(&b.layer).then(b.node.cmp(&a.node)));

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

    fn random_from(
        &self,
        from: GraphSize,
        subtract_from_end: Option<usize>,
    ) -> Option<GraphLocation> {
        if self.layers.len() == 0 {
            return None;
        }

        let subtract_from_end = subtract_from_end.unwrap_or_default();
        let mut rng = rand::thread_rng();
        let layer_idx = rng.gen_range(from as usize..self.layers.len() - subtract_from_end);
        let layer = &self.layers[layer_idx];

        if layer.len() == 0 {
            return None;
        }
        Some(GraphLocation::new(
            layer_idx as GraphSize,
            rng.gen_range(0..layer.len()) as GraphSize,
        ))
    }

    pub fn random_input_or_hidden(&self) -> Option<GraphLocation> {
        self.random_from(0, Some(0))
    }

    pub fn random_output_or_hidden(
        &self,
        greater_than: Option<GraphSize>,
    ) -> Option<GraphLocation> {
        self.random_from(greater_than.unwrap_or_default() + 1, None)
    }

    pub fn random_hidden(&self) -> Option<GraphLocation> {
        self.random_from(1, Some(1))
    }

    pub fn random_edge_mut(&mut self) -> Option<(GraphLocation, &mut GraphEdge)> {
        self.layers
            .iter_mut()
            .enumerate()
            .flat_map(|l| {
                l.1.iter_mut()
                    .enumerate()
                    .flat_map(|n| n.1.connections.iter_mut().map(move |e| (n.0, e)))
                    .map(move |(node_pos, e)| {
                        (
                            GraphLocation::new(l.0 as GraphSize, node_pos as GraphSize),
                            e,
                        )
                    })
            })
            .choose_stable(&mut rand::thread_rng())
    }

    pub fn has_cycle(&self, start_from: Option<GraphLocation>) -> bool {
        if self.layers.len() == 0 {
            return false;
        }

        let start = start_from.unwrap_or(GraphLocation::default());

        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        stack.push(start);

        while let Some(loc) = stack.pop() {
            if visited.contains(&loc) {
                return true;
            }
            visited.insert(loc);

            for node in &self.layers[loc.layer as usize][loc.node as usize].connections {
                stack.push(node.to.clone());
            }
        }
        false
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Net {
    pub graph: NeuralGraph,
    pub input_layer: GraphSize,
    pub output_layer: GraphSize,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
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

        // Always generate at least 1 hidden layer with meaningful node counts.
        let num_hidden_layers = rng.gen_range(1..=6usize);
        let mut id = 0;
        for _ in 0..num_hidden_layers {
            let num_nodes = rng.gen_range(2..=14usize);
            let l = g.add_layer_to_end();
            for _ in 0..num_nodes {
                g.add_node(
                    l,
                    GraphNode::new(Node::Neuron(Box::new(BasicNeuron {
                        bias: rng.gen_range(-1.0..1.0),
                        id,
                    }))),
                )?;
                id += 1;
            }
        }
        debug!("{num_hidden_layers}");

        let output_layer = g.add_layer_to_end();
        for value in output_nodes {
            g.add_node(output_layer, GraphNode::new(value.clone()))?;
        }

        // More connections, better weight range, and mostly-enabled edges.
        let total_layers = (num_hidden_layers + 2) as GraphSize;
        let num_connections =
            rng.gen_range((4 * (num_hidden_layers + 1))..(12 * (num_hidden_layers + 1)));
        let mut actual_connections = 0;
        let mut connection_pairs = Vec::new();
        let mut attempts = 0;
        while actual_connections < num_connections && attempts < num_connections * 10 {
            attempts += 1;
            let from_layer: GraphSize = rng.gen_range(0..total_layers - 1);
            let to_layer: GraphSize = rng.gen_range(from_layer + 1..total_layers);

            let from_len = g.layers[from_layer as usize].len();
            let to_len = g.layers[to_layer as usize].len();
            if from_len == 0 || to_len == 0 {
                continue;
            }

            let from_node = rng.gen_range(0..from_len) as GraphSize;
            let to_node = rng.gen_range(0..to_len) as GraphSize;
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
            g.add_edge(
                from,
                to,
                Edge {
                    weight: rng.gen_range(-3.0..3.0),
                    enabled: rng.gen_bool(0.85),
                },
            )?;
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

    pub fn mutate(
        &mut self,
        link_mutators: &[Box<dyn mutate::LinkMutator>],
        neuron_mutators: &[Box<dyn mutate::NeuronMutator>],
        neurons: &[Box<dyn Neuron>],
        neuron_selector: &MutationSelector,
        mut generator: impl mutate::Generator,
    ) -> Result<(), mutate::MutateError> {
        loop {
            let (is_link, idx, done) =
                generator.generate(&self.graph, link_mutators, &neuron_mutators)?;

            if is_link {
                link_mutators[idx].mutate(self)?;
            } else {
                neuron_mutators[idx].mutate(self, neurons, neuron_selector)?;
            }

            if done {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test_requirements {
    use macros::{DNeuronInfo, SubTraits};
    use serde::{Deserialize, Serialize};

    use super::{GraphLocation, GraphNode, Net, NeuralGraph, NeuronInfo, NeuronSubTraits, Node};

    #[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
    pub struct BlankInput {
        pub value: f32,
        pub id: usize,
    }

    #[typetag::serde]
    impl super::InputNeuron for BlankInput {
        fn as_standard(&self) -> f32 {
            self.value
        }

        fn set_value(&mut self, value: f32) {
            self.value = value;
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
    pub struct TestNeuronA {
        pub value: f32,
        pub id: usize,
    }

    #[typetag::serde]
    impl super::Neuron for TestNeuronA {
        fn step(&self, edge: &super::Edge, input: f32) -> f32 {
            edge.weight * input
        }

        fn finish(&self, partial: f32) -> f32 {
            partial + self.value
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone, DNeuronInfo, SubTraits)]
    pub struct TestNeuronB {
        pub value: f32,
        pub id: usize,
    }

    #[typetag::serde]
    impl super::Neuron for TestNeuronB {
        fn step(&self, edge: &super::Edge, input: f32) -> f32 {
            edge.weight * input
        }

        fn finish(&self, partial: f32) -> f32 {
            partial + self.value
        }
    }

    impl BlankInput {
        pub fn new(value: f32, id: usize) -> Self {
            BlankInput { value, id }
        }
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

    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct Simulation {
        pub nets: Vec<Nn>,
        pub input_nodes: Vec<Node>,
        pub output_nodes: Vec<Node>,
    }

    pub fn create_graph(inputs: &[Node], outputs: &[Node]) -> NeuralGraph {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        for item in inputs.to_vec() {
            g.add_node(0, GraphNode::new(item)).unwrap();
        }

        g.add_layer_to_end();
        for item in outputs.to_vec() {
            g.add_node(1, GraphNode::new(item)).unwrap();
        }
        g
    }
}

#[cfg(test)]
mod test {
    use crate::{activations::Sigmoid, nn::Edge};

    use super::{
        test_requirements::*, BasicNeuron, GraphLocation, GraphNode, NeuralGraph, NeuralGraphError,
        Net, Node,
    };

    #[test]
    #[rustfmt::skip]
    fn has_cycle() {
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(0.0, 0))),
            Node::Input(Box::new(BlankInput::new(0.0, 1))),
            Node::Input(Box::new(BlankInput::new(0.0, 2))),
        ];

        let output_nodes = [
            Node::Output(Sigmoid::new(0.0, 3, "a".to_owned())),
            Node::Output(Sigmoid::new(0.0, 4, "b".to_owned())),
        ];

        let mut g = create_graph(&input_nodes, &output_nodes);
        g.add_layer(1);
        g.add_node(1, GraphNode::blank()).unwrap();
        g.add_node(1, GraphNode::blank()).unwrap();

        g.add_layer(2);
        g.add_node(2, GraphNode::blank()).unwrap();
        g.add_node(2, GraphNode::blank()).unwrap();

        g.add_edge(GraphLocation::new(0, 1), GraphLocation::new(1, 0), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(0, 2), GraphLocation::new(1, 1), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(1, 0), GraphLocation::new(2, 1), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(1, 1), GraphLocation::new(2, 0), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(1, 1), GraphLocation::new(2, 1), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(2, 0), GraphLocation::new(3, 0), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(2, 1), GraphLocation::new(3, 0), Edge::default()).unwrap();

        // should not be a cycle since 0,0 is not connected to anything
        assert!(!g.has_cycle(None));
        assert!(!g.has_cycle(Some(GraphLocation::new(0, 1))));

        g.add_edge(GraphLocation::new(3, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();

        assert!(g.has_cycle(Some(GraphLocation::new(0, 1))));
    }

    #[test]
    fn add_node_out_of_bounds_returns_layer_not_found() {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end(); // only layer 0 exists
        let result = g.add_node(5, GraphNode::blank());
        assert!(
            matches!(result, Err(NeuralGraphError::LayerNotFound(_))),
            "Expected LayerNotFound for layer index 5, got {result:?}"
        );
    }

    #[test]
    fn add_edge_to_missing_node_returns_node_not_found() {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        g.add_layer_to_end();
        g.add_node(0, GraphNode::new(Node::Input(Box::new(BlankInput::new(0.0, 0))))).unwrap();
        // layer 1 is empty – no node at (1, 0)
        let result = g.add_edge(
            GraphLocation::new(0, 0),
            GraphLocation::new(1, 0),
            Edge::default(),
        );
        assert!(
            matches!(result, Err(NeuralGraphError::NodeNotFound(_))),
            "Expected NodeNotFound, got {result:?}"
        );
    }

    #[test]
    fn add_edge_duplicate_returns_connection_exists() {
        let input_nodes = [Node::Input(Box::new(BlankInput::new(0.0, 0)))];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 1, "a".to_owned()))];
        let mut g = create_graph(&input_nodes, &output_nodes);
        g.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();
        let result = g.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default());
        assert!(
            matches!(result, Err(NeuralGraphError::ConnectionExists { .. })),
            "Expected ConnectionExists for duplicate edge, got {result:?}"
        );
    }

    #[test]
    fn remove_edge_returns_false_when_edge_absent() {
        let input_nodes = [Node::Input(Box::new(BlankInput::new(0.0, 0)))];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 1, "a".to_owned()))];
        let mut g = create_graph(&input_nodes, &output_nodes);
        // No edge added yet – the lengths are equal → returns true (unchanged)
        let result = g.remove_edge(&GraphLocation::new(0, 0), &GraphLocation::new(1, 0));
        assert!(result, "remove_edge should return true (unchanged) when no matching edge exists");
    }

    #[test]
    fn remove_edge_returns_false_after_successful_removal() {
        let input_nodes = [Node::Input(Box::new(BlankInput::new(0.0, 0)))];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 1, "a".to_owned()))];
        let mut g = create_graph(&input_nodes, &output_nodes);
        g.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();
        assert_eq!(g.layers[0][0].connections.len(), 1, "Should have 1 edge before removal");
        let result = g.remove_edge(&GraphLocation::new(0, 0), &GraphLocation::new(1, 0));
        // After removal lengths differ → returns false
        assert!(!result, "remove_edge should return false when an edge was actually removed");
        assert_eq!(g.layers[0][0].connections.len(), 0, "Edge should be gone after removal");
    }

    #[test]
    fn removed_node_deletes_node_and_clears_incoming_edges() {
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(0.0, 0))),
            Node::Input(Box::new(BlankInput::new(0.0, 1))),
        ];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 2, "a".to_owned()))];
        let mut g = create_graph(&input_nodes, &output_nodes);
        g.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();
        g.add_edge(GraphLocation::new(0, 1), GraphLocation::new(1, 0), Edge::default()).unwrap();

        assert_eq!(g.layers[1].len(), 1);
        g.removed_node(GraphLocation::new(1, 0));

        assert_eq!(g.layers[1].len(), 0, "Output node should be removed");
        assert_eq!(g.layers[0][0].connections.len(), 0, "Edge from input 0 should be cleaned up");
        assert_eq!(g.layers[0][1].connections.len(), 0, "Edge from input 1 should be cleaned up");
    }

    #[test]
    fn add_layer_shifts_existing_edge_targets() {
        let input_nodes = [Node::Input(Box::new(BlankInput::new(0.0, 0)))];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 1, "a".to_owned()))];
        let mut g = create_graph(&input_nodes, &output_nodes);
        g.add_edge(GraphLocation::new(0, 0), GraphLocation::new(1, 0), Edge::default()).unwrap();

        assert_eq!(g.layers[0][0].connections[0].to.layer, 1);
        g.add_layer(1); // inserts empty hidden layer between input and output
        assert_eq!(g.layers.len(), 3, "Should now have 3 layers");
        assert_eq!(
            g.layers[0][0].connections[0].to.layer, 2,
            "Edge target should have shifted from layer 1 to layer 2"
        );
    }

    #[test]
    fn prune_removes_none_nodes_and_keeps_real_nodes() {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        g.add_node(0, GraphNode::new(Node::Input(Box::new(BlankInput::new(0.0, 0))))).unwrap();
        g.add_node(0, GraphNode::blank()).unwrap(); // Node::None – should be pruned
        g.add_layer_to_end();
        g.add_node(1, GraphNode::new(Node::Output(Sigmoid::new(0.0, 1, "a".to_owned())))).unwrap();

        assert_eq!(g.layers[0].len(), 2, "Before prune: 2 nodes in layer 0");
        g.prune();
        assert_eq!(g.layers[0].len(), 1, "After prune: only the real input node should remain");
        assert!(
            matches!(g.layers[0][0].value, Node::Input(_)),
            "Remaining node should be the Input node, not the blank placeholder"
        );
    }

    #[test]
    fn get_node_returns_none_for_nonexistent_layer() {
        let g = NeuralGraph::new();
        assert!(g.get_node(&GraphLocation::new(99, 0)).is_none());
    }

    #[test]
    fn get_node_returns_none_for_nonexistent_node_in_layer() {
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        // Layer 0 is empty
        assert!(g.get_node(&GraphLocation::new(0, 5)).is_none());
    }

    #[test]
    fn from_preserving_basic_copies_input_and_output_nodes() {
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(1.0, 0))),
            Node::Input(Box::new(BlankInput::new(2.0, 1))),
        ];
        let output_nodes = [Node::Output(Sigmoid::new(0.0, 2, "Sigmoid".to_owned()))];
        let g = create_graph(&input_nodes, &output_nodes);
        let original = Net { graph: g, input_layer: 0, output_layer: 1 };

        let preserved = Net::from_preserving_basic(&original).expect("from_preserving_basic failed");

        assert_eq!(preserved.graph.layers.len(), 2, "Preserved net should have exactly 2 layers");
        assert_eq!(preserved.graph.layers[0].len(), 2, "Should preserve 2 input nodes");
        assert_eq!(preserved.graph.layers[1].len(), 1, "Should preserve 1 output node");
        // No edges should be carried over
        assert_eq!(
            preserved.graph.layers[0][0].connections.len(), 0,
            "Preserved net should have no edges"
        );
    }

    #[test]
    fn tick_direct_input_to_output_sigmoid_zero() {
        // Input value 0.0 → direct edge (weight 1.0) → Sigmoid → sigmoid(0.0) = 0.5
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        g.add_node(0, GraphNode::new(Node::Input(Box::new(BlankInput::new(0.0, 0))))).unwrap();
        g.add_layer_to_end();
        g.add_node(1, GraphNode::new(Node::Output(Sigmoid::new(0.0, 1, "Sigmoid".to_owned())))).unwrap();
        g.add_edge(
            GraphLocation::new(0, 0),
            GraphLocation::new(1, 0),
            Edge { weight: 1.0, enabled: true },
        ).unwrap();

        let mut net = Net { graph: g, input_layer: 0, output_layer: 1 };
        net.tick();

        if let Node::Output(o) = &net.graph.layers[1][0].value {
            assert!(
                (o.value() - 0.5).abs() < 1e-6,
                "sigmoid(0.0) should be 0.5, got {}",
                o.value()
            );
        } else {
            panic!("Expected Output node at layer 1, node 0");
        }
    }

    #[test]
    fn tick_through_hidden_layer_with_known_weights() {
        // Input(2.0) → [weight=0.5] → BasicNeuron(bias=1.0) → [weight=1.0] → Sigmoid
        // hidden step : 0.5 × 2.0 = 1.0
        // hidden finish: 1.0 + 1.0(bias) = 2.0
        // output step : 1.0 × 2.0 = 2.0
        // output finish: sigmoid(2.0) ≈ 0.8807970
        let mut g = NeuralGraph::new();
        g.add_layer_to_end(); // layer 0
        g.add_node(0, GraphNode::new(Node::Input(Box::new(BlankInput::new(2.0, 0))))).unwrap();

        g.add_layer_to_end(); // layer 1
        g.add_node(1, GraphNode::new(Node::Neuron(Box::new(BasicNeuron { bias: 1.0, id: 1 })))).unwrap();

        g.add_layer_to_end(); // layer 2
        g.add_node(2, GraphNode::new(Node::Output(Sigmoid::new(0.0, 2, "Sigmoid".to_owned())))).unwrap();

        g.add_edge(
            GraphLocation::new(0, 0),
            GraphLocation::new(1, 0),
            Edge { weight: 0.5, enabled: true },
        ).unwrap();
        g.add_edge(
            GraphLocation::new(1, 0),
            GraphLocation::new(2, 0),
            Edge { weight: 1.0, enabled: true },
        ).unwrap();

        let mut net = Net { graph: g, input_layer: 0, output_layer: 2 };
        net.tick();

        use std::f32::consts::E;
        let expected = 1.0 / (1.0 + E.powf(-2.0f32));
        if let Node::Output(o) = &net.graph.layers[2][0].value {
            assert!(
                (o.value() - expected).abs() < 1e-6,
                "Expected sigmoid(2.0) ≈ {expected}, got {}",
                o.value()
            );
        } else {
            panic!("Expected Output node at layer 2, node 0");
        }
    }

    #[test]
    fn tick_disabled_edge_does_not_propagate() {
        // Input(5.0) → disabled edge → Sigmoid. Output should stay at its initial 0.0.
        let mut g = NeuralGraph::new();
        g.add_layer_to_end();
        g.add_node(0, GraphNode::new(Node::Input(Box::new(BlankInput::new(5.0, 0))))).unwrap();
        g.add_layer_to_end();
        g.add_node(1, GraphNode::new(Node::Output(Sigmoid::new(0.0, 1, "Sigmoid".to_owned())))).unwrap();
        g.add_edge(
            GraphLocation::new(0, 0),
            GraphLocation::new(1, 0),
            Edge { weight: 1.0, enabled: false }, // disabled!
        ).unwrap();

        let mut net = Net { graph: g, input_layer: 0, output_layer: 1 };
        net.tick();

        if let Node::Output(o) = &net.graph.layers[1][0].value {
            assert_eq!(
                o.value(), 0.0,
                "Disabled edge should not propagate; output should remain at initial 0.0, got {}",
                o.value()
            );
        } else {
            panic!("Expected Output node");
        }
    }
}
