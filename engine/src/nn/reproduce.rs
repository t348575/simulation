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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Crossover;

impl Crossover {
    pub fn new() -> Crossover {
        Crossover {}
    }
}

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DefaultIterator(usize);

impl DefaultIterator {
    pub fn new() -> DefaultIterator {
        DefaultIterator(0)
    }
}

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
