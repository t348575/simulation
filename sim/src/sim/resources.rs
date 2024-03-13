use std::thread::JoinHandle;

use bevy::ecs::{schedule::States, system::Resource};
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};

use super::sim::{BasicCreature, RunnerReq, RunnerRes};

#[derive(Resource)]
pub struct RunnerResource {
    pub tx: Sender<RunnerReq>,
    pub rx: Receiver<RunnerRes>,
    pub thread: JoinHandle<()>,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize, Default)]
pub struct Simulation {
    pub world_dim: (f32, f32),
    pub window_dims: (f32, f32),
    pub creatures: Vec<BasicCreature>,
    pub food: Vec<(f32, f32)>,
    pub ticks: usize,
}

#[derive(Resource, Debug)]
pub struct ControlPanel {
    pub initial_num_creatures: String,
    pub width: String,
    pub height: String,
    pub can_create_sim: bool,
}

impl Default for ControlPanel {
    fn default() -> Self {
        Self {
            initial_num_creatures: "500".to_owned(),
            width: "0".to_owned(),
            height: "0".to_owned(),
            can_create_sim: true,
        }
    }
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimulationState {
    #[default]
    None,
    Paused,
    Running,
}
