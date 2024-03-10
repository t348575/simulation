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
