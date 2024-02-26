use serde::{Deserialize, Serialize};

use crate::{
    nn::{GraphLocation, Net},
    NeuronInfo,
};

use super::{GraphNode, GraphSize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub enum AlignedItem {
    Node(GraphLocation, GraphLocation),
    Edge {
        data: (ConnectionInfo, ConnectionInfo),
        _type: EdgeType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub enum EdgeType {
    Incoming,
    Outgoing,
}

pub fn intersection(a: &Net, b: &Net) -> Vec<AlignedItem> {
    let mut res = Vec::new();

    if a.graph.layers.len() == 2 || b.graph.layers.len() == 2 {
        return res;
    }

    let mut a_iter = a.graph.layers.iter().enumerate();
    let mut b_iter = b.graph.layers.iter().enumerate();

    let (mut a_connections, mut b_connections) =
        extend_prev_connections(a_iter.next().unwrap(), b_iter.next().unwrap());

    for _ in 1..a.graph.layers.len() - 1 {
        let a_layer = match a_iter.next() {
            Some(a) => a,
            None => break,
        };
        let b_layer = match b_iter.next() {
            Some(b) => b,
            None => break,
        };

        for (a_idx, a_node) in a_layer.1.iter().enumerate() {
            for (b_idx, b_node) in b_layer.1.iter().enumerate() {
                let are_nodes_same = (b_node.value._type() == a_node.value._type())
                    && (a_node.value.id() == b_node.value.id());
                if !are_nodes_same {
                    continue;
                }

                let a_conn_incoming = a_connections
                    .iter()
                    .filter(|x| {
                        x.to.layer == a_layer.0 as GraphSize && x.to.node == a_idx as GraphSize
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                let b_conn_incoming = b_connections
                    .iter()
                    .filter(|x| {
                        x.to.layer == b_layer.0 as GraphSize && x.to.node == b_idx as GraphSize
                    })
                    .cloned()
                    .collect::<Vec<_>>();

                let a_conn_outgoing = a_node
                    .connections
                    .iter()
                    .map(|x| ConnectionInfo {
                        from: GraphLocation::new(a_layer.0 as GraphSize, a_idx as GraphSize),
                        to: x.to.clone(),
                        _type: a.graph.get_node(&x.to).unwrap().value._type(),
                        id: a.graph.get_node(&x.to).unwrap().value.id(),
                    })
                    .collect::<Vec<_>>();
                let b_conn_outgoing = b_node
                    .connections
                    .iter()
                    .map(|x| ConnectionInfo {
                        from: GraphLocation::new(b_layer.0 as GraphSize, b_idx as GraphSize),
                        to: x.to.clone(),
                        _type: b.graph.get_node(&x.to).unwrap().value._type(),
                        id: b.graph.get_node(&x.to).unwrap().value.id(),
                    })
                    .collect::<Vec<_>>();

                let common_edges_incoming =
                    connection_intersection(a_conn_incoming, b_conn_incoming);
                let common_edges_outgoing =
                    connection_intersection(a_conn_outgoing, b_conn_outgoing);

                if common_edges_incoming.len() > 0 || common_edges_outgoing.len() > 0 {
                    res.push(AlignedItem::Node(
                        GraphLocation::new(a_layer.0 as GraphSize, a_idx as GraphSize),
                        GraphLocation::new(b_layer.0 as GraphSize, b_idx as GraphSize),
                    ));
                }

                res.extend(
                    common_edges_incoming
                        .into_iter()
                        .map(|edge| AlignedItem::Edge {
                            data: (edge.0, edge.1),
                            _type: EdgeType::Incoming,
                        }),
                );
                res.extend(
                    common_edges_outgoing
                        .into_iter()
                        .map(|edge| AlignedItem::Edge {
                            data: (edge.0, edge.1),
                            _type: EdgeType::Outgoing,
                        }),
                );
            }
        }

        (a_connections, b_connections) = extend_prev_connections(a_layer, b_layer);
    }

    res
}

fn connection_intersection(
    a: Vec<ConnectionInfo>,
    b: Vec<ConnectionInfo>,
) -> Vec<(ConnectionInfo, ConnectionInfo)> {
    let mut res = Vec::new();
    for a_item in a {
        for b_item in b.clone() {
            if a_item._type == b_item._type && a_item.id == b_item.id {
                res.push((a_item, b_item));
                break;
            }
        }
    }
    res
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub from: GraphLocation,
    pub to: GraphLocation,
    pub _type: &'static str,
    pub id: usize,
}

fn extend_prev_connections(
    a: (usize, &Vec<GraphNode>),
    b: (usize, &Vec<GraphNode>),
) -> (Vec<ConnectionInfo>, Vec<ConnectionInfo>) {
    let a_c =
        a.1.iter()
            .enumerate()
            .map(|item| {
                item.1.connections.iter().map(move |x| ConnectionInfo {
                    from: GraphLocation::new(a.0 as GraphSize, item.0 as GraphSize),
                    to: x.to.clone(),
                    _type: item.1.value._type(),
                    id: item.1.value.id(),
                })
            })
            .flatten()
            .collect();
    let b_c =
        b.1.iter()
            .enumerate()
            .map(|item| {
                item.1.connections.iter().map(move |x| ConnectionInfo {
                    from: GraphLocation::new(b.0 as GraphSize, item.0 as GraphSize),
                    to: x.to.clone(),
                    _type: item.1.value._type(),
                    id: item.1.value.id(),
                })
            })
            .flatten()
            .collect();
    (a_c, b_c)
}

/// Pairs as (From, To)
pub fn connection_pair_exists(
    pairs: &[(GraphLocation, GraphLocation)],
    from: &GraphLocation,
    to: &GraphLocation,
) -> bool {
    pairs
        .iter()
        .position(|x| (x.0 == *from && x.1 == *to) || (x.0 == *to && x.1 == *from))
        .map_or_else(|| false, |_| true)
}

#[cfg(test)]
mod test {
    use macros::{Name, SubTraits};
    use serde::{Deserialize, Serialize};

    use crate::{
        activations::Sigmoid,
        nn::{
            reproduce::{Crossover, Reproducer},
            BasicNeuron, Edge, GraphLocation, GraphNode, InputNeuron, Net, NeuralGraph, Neuron,
            NeuronSubTraits, Node,
        },
        NeuronInfo,
    };

    #[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
    pub struct BlankInput {
        value: f32,
        id: usize,
    }

    #[typetag::serde]
    impl InputNeuron for BlankInput {
        fn as_standard(&self) -> f32 {
            self.value
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
    pub struct TestNeuronA {
        value: f32,
        id: usize,
    }

    #[typetag::serde]
    impl Neuron for TestNeuronA {
        fn step(&self, edge: &Edge, input: f32) -> f32 {
            edge.weight * input
        }

        fn finish(&self, partial: f32) -> f32 {
            partial + self.value
        }
    }

    #[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
    pub struct TestNeuronB {
        value: f32,
        id: usize,
    }

    #[typetag::serde]
    impl Neuron for TestNeuronB {
        fn step(&self, edge: &Edge, input: f32) -> f32 {
            edge.weight * input
        }

        fn finish(&self, partial: f32) -> f32 {
            partial + self.value
        }
    }

    impl BlankInput {
        fn new(value: f32, id: usize) -> Self {
            BlankInput { value, id }
        }
    }

    // #[derive(Debug, Default, Serialize, Deserialize)]
    // pub struct NodePosition {
    //     pub x: f32,
    //     pub y: f32,
    // }

    // #[derive(Debug, Default, Serialize, Deserialize)]
    // pub struct Nn {
    //     pub net: Net,
    //     pub node_positions: Vec<(GraphLocation, NodePosition)>,
    // }

    // #[derive(Debug, Default, Serialize, Deserialize)]
    // pub struct Simulation {
    //     pub nets: Vec<Nn>,
    //     pub input_nodes: Vec<Node>,
    //     pub output_nodes: Vec<Node>,
    // }

    #[test]
    #[rustfmt::skip]
    fn basic_test() {
        env_logger::init();
        let input_nodes = [
            Node::Input(Box::new(BlankInput::new(0.0, 0))),
            Node::Input(Box::new(BlankInput::new(0.0, 1))),
            Node::Input(Box::new(BlankInput::new(0.0, 2))),
        ];

        let output_nodes = [
            Node::Output(Sigmoid::new(0.0, 3)),
            Node::Output(Sigmoid::new(0.0, 4)),
        ];

        let mut graph_a = NeuralGraph::new();
        graph_a.add_layer_to_end();
        for item in input_nodes.clone() {
            graph_a.add_node(0, GraphNode::new(item)).unwrap();
        }

        graph_a.add_layer_to_end();
        for item in output_nodes.clone() {
            graph_a.add_node(1, GraphNode::new(item)).unwrap();
        }

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

        let mut graph_b = NeuralGraph::new();
        graph_b.add_layer_to_end();
        for item in input_nodes.clone() {
            graph_b.add_node(0, GraphNode::new(item)).unwrap();
        }

        graph_b.add_layer_to_end();
        for item in output_nodes.clone() {
            graph_b.add_node(1, GraphNode::new(item)).unwrap();
        }

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
        Crossover::new().reproduce(&a, &b, &mut output).expect("Could not crossover");

        let mut compose_output = NeuralGraph::new();
        compose_output.add_layer_to_end();
        for item in input_nodes.clone() {
            compose_output.add_node(0, GraphNode::new(item)).unwrap();
        }

        compose_output.add_layer_to_end();
        for item in output_nodes.clone() {
            compose_output.add_node(1, GraphNode::new(item)).unwrap();
        }

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
