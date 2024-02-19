use thiserror::Error;

use crate::nn::{Net, NeuralGraphError, Node};

#[derive(Debug)]
pub struct Simulation {
    world_dim: (f32, f32),
    creatures: Vec<Creature>,
}

#[derive(Debug)]
pub struct Creature {
    brain: Net,
}

#[derive(Debug, Error)]
pub enum CreateSimError {
    #[error("Could not create brain {0}")]
    BrainCreateError(NeuralGraphError),
}

impl Simulation {
    pub fn new(
        world_dim: (f32, f32),
        num_creatures: usize,
        input_nodes: &[Node],
        output_nodes: &[Node],
    ) -> Result<Simulation, CreateSimError> {
        let mut creatures = Vec::new();

        for _ in 0..num_creatures {
            let brain = Net::gen(input_nodes, output_nodes)
                .map_err(|err| CreateSimError::BrainCreateError(err))?;
            creatures.push(Creature { brain });
        }

        Ok(Simulation {
            world_dim,
            creatures,
        })
    }
}
