use dyn_clone::clone_trait_object;
use serde::{Deserialize, Serialize};
use macros::{Name, SubTraits};
use rand::Rng;
use thiserror::Error;

use super::{util, Net, NeuralGraph, NeuralGraphError, NeuronSubTraits};
use crate::NeuronName;

clone_trait_object!(NeuronSubTraits);
pub trait Reproducer: NeuronSubTraits {
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

pub trait Generator: NeuronSubTraits {
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

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
pub struct Crossover;

impl Crossover {
    pub fn new() -> Crossover {
        Crossover{}
    }
}

impl Reproducer for Crossover {
    /// Crossover between two networks
    fn reproduce(&self, a: &Net, b: &Net, output: &mut Net) -> Result<(), ReproduceError> {
        let common_elements = util::intersection(a, b);
        println!("{common_elements:#?}");
        for item in common_elements {
            match item {
                util::AlignedItem::Node(loc) => {
                    let mut rng = rand::thread_rng();
                    let n = if rng.gen::<f32>() < 0.5 {
                        a.graph.get_node(&loc)
                    } else {
                        b.graph.get_node(&loc)
                    }
                    .unwrap();

                    output.graph.add_node_at(loc, n.value.clone());
                }
                util::AlignedItem::Edge((from, to)) => {
                    let mut rng = rand::thread_rng();
                    let n = if rng.gen::<f32>() < 0.5 {
                        a.graph.get_edge(&from, &to)
                    } else {
                        b.graph.get_edge(&from, &to)
                    }
                    .unwrap();

                    output.graph.add_edge(from, to, n.value.clone())?;
                }
            }
        }

        output.graph.prune();
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Name, SubTraits)]
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