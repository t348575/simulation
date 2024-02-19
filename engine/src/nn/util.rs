use hashbrown::HashSet;

use crate::{
    nn::{GraphLocation, Net},
    NeuronName,
};

use super::GraphSize;

pub fn intersection(a: &Net, b: &Net) -> Vec<AlignedItem> {
    let mut res = Vec::new();

    if a.graph.layers.len() == 2 || b.graph.layers.len() == 2 {
        return res;
    }

    let mut a_iter = a.graph.layers.iter().enumerate();
    let mut b_iter = b.graph.layers.iter().enumerate();

    let mut a_connections = Vec::new();
    let mut b_connections = Vec::new();
    a_iter
        .next()
        .unwrap()
        .1
        .iter()
        .for_each(|item| a_connections.extend(item.connections.iter().map(|x| x.to.clone())));
    b_iter
        .next()
        .unwrap()
        .1
        .iter()
        .for_each(|item| b_connections.extend(item.connections.iter().map(|x| x.to.clone())));

    for a_layer in a_iter {
        let b_layer = match b_iter.next() {
            Some(b) => b,
            None => break,
        };

        for (a_idx, a_node) in a_layer.1.iter().enumerate() {
            for (b_idx, b_node) in b_layer.1.iter().enumerate() {
                let are_nodes_same = b_node.value.name() == a_node.value.name();
                if !are_nodes_same {
                    continue;
                }

                let a_conn_incoming = a_connections
                    .iter()
                    .filter(|x| x.layer == a_layer.0 as GraphSize && x.node == a_idx as GraphSize)
                    .collect::<HashSet<_>>();
                let b_conn_incoming = b_connections
                    .iter()
                    .filter(|x| x.layer == b_layer.0 as GraphSize && x.node == b_idx as GraphSize)
                    .collect::<HashSet<_>>();

                let a_conn_outgoing = a_node
                    .connections
                    .iter()
                    .map(|x| &x.to)
                    .collect::<HashSet<_>>();
                let b_conn_outgoing = b_node
                    .connections
                    .iter()
                    .map(|x| &x.to)
                    .collect::<HashSet<_>>();

                let common_edges_incoming = &a_conn_incoming - &b_conn_incoming;
                let common_edges_outgoing = &a_conn_outgoing - &b_conn_outgoing;
                let common_node = GraphLocation::new(a_layer.0 as GraphSize, a_idx as GraphSize);
                if common_edges_incoming.len() > 0 || common_edges_outgoing.len() > 0 {
                    res.push(AlignedItem::Node(common_node.clone()))
                }

                common_edges_incoming.iter().for_each(|edge| {
                    res.push(AlignedItem::Edge(((*edge).clone(), common_node.clone())))
                });
                common_edges_outgoing.iter().for_each(|edge| {
                    res.push(AlignedItem::Edge((common_node.clone(), (*edge).clone())))
                });
            }
        }
    }

    res
}

#[derive(Debug)]
pub enum AlignedItem {
    Node(GraphLocation),
    /// (From, To)
    Edge((GraphLocation, GraphLocation)),
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
