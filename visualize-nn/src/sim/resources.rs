use bevy::ecs::{schedule::States, system::Resource};
use engine::nn::Net;
use serde::{Deserialize, Serialize};

#[derive(Resource, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Simulation {
    pub world_dim: (f32, f32),
    pub creatures: Vec<Creature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Creature {
    pub brain: Net,
    pub position: (f32, f32)
}

#[derive(Resource, Debug)]
pub struct ControlPanel {
    pub initial_num_creatures: String,
    pub width: String,
    pub height: String,
    pub can_create_sim: bool
}

impl Default for ControlPanel {
    fn default() -> Self {
        Self {
            initial_num_creatures: "50".to_owned(),
            width: "0".to_owned(),
            height: "0".to_owned(),
            can_create_sim: true
        }
    }
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimulationState {
    #[default]
    None,
    Paused,
    Running
}