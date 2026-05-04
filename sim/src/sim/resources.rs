use std::thread::JoinHandle;

use bevy::prelude::*;
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};

use super::sim::{BasicCreature, BasicFood, RunnerReq, RunnerRes};

#[derive(Resource)]
#[allow(dead_code)]
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
    pub food: Vec<BasicFood>,
    pub poison: Vec<(f32, f32)>,
    pub ticks: usize,
}

#[derive(Resource, Debug)]
pub struct ControlPanel {
    pub initial_num_creatures: String,
    pub width: String,
    pub height: String,
    pub can_create_sim: bool,
}

#[derive(Resource, Debug, Default)]
pub struct SystemWindow(pub Option<Entity>);

#[derive(Resource, Debug, Default)]
pub struct PendingNet(pub Option<engine::nn::Net>);

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

#[derive(Component)]
pub struct SystemWindowCamera;

#[derive(Component)]
pub struct SimCamera;

#[derive(Resource, Debug, Clone)]
pub struct ViewState {
    pub zoom: f32,
    pub pan: Vec2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self { zoom: 1.0, pan: Vec2::ZERO }
    }
}

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SimulationState {
    #[default]
    None,
    Paused,
    Running,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationStats {
    pub current_population: usize,
    pub target_population: usize,
    pub births_this_tick: usize,
    pub deaths_this_tick: usize,
    pub total_spawned: usize,
    pub avg_energy: f32,
    pub avg_age: f32,
    pub avg_size: f32,
    pub food_count: usize,
    pub poison_count: usize,
    pub survival_rate: f32,
    pub selected_creature_id: Option<usize>,
    pub food_eaten_this_tick: usize,
    pub total_food_eaten: usize,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub start_energy: f32,
    pub max_energy: f32,
    pub base_energy_cost: f32,
    pub move_energy_cost: f32,
    pub min_move_energy: f32,
    pub full_speed_energy: f32,
    pub mate_attempt_cost: f32,
    pub eat_attempt_cost: f32,
    pub eat_action_cost: f32,
    pub eat_action_base_ticks: u16,
    pub mate_energy_cost: f32,
    pub min_mate_energy: f32,
    pub child_energy: f32,
    pub max_age: u32,
    pub min_creature_size: f32,
    pub max_creature_size: f32,
    pub creature_dim: f32,
    pub mate_threshold: f32,
    pub mate_cooldown_ticks: u16,
    pub mutation_rate: f32,
    pub mutation_amount: f32,
    pub max_births_per_tick: usize,
    pub food_per_creature: usize,
    pub min_food_size: f32,
    pub max_food_size: f32,
    pub food_energy_per_size: f32,
    pub food_eat_ticks_per_size: f32,
    pub poison_per_creature: usize,
    pub poison_damage: f32,
    pub creature_spawning_enabled: bool,
    pub food_spawn_multiplier: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            start_energy: 80.0,
            max_energy: 120.0,
            base_energy_cost: 0.02,
            move_energy_cost: 0.08,
            min_move_energy: 5.0,
            full_speed_energy: 45.0,
            mate_attempt_cost: 0.03,
            eat_attempt_cost: 0.02,
            eat_action_cost: 0.5,
            eat_action_base_ticks: 8,
            mate_energy_cost: 25.0,
            min_mate_energy: 55.0,
            child_energy: 45.0,
            max_age: 8_000,
            min_creature_size: 3.0,
            max_creature_size: 14.0,
            creature_dim: 5.0,
            mate_threshold: 0.5,
            mate_cooldown_ticks: 120,
            mutation_rate: 0.05,
            mutation_amount: 0.35,
            max_births_per_tick: 10,
            food_per_creature: 4,
            min_food_size: 2.0,
            max_food_size: 10.0,
            food_energy_per_size: 7.0,
            food_eat_ticks_per_size: 4.0,
            poison_per_creature: 12,
            poison_damage: 45.0,
            creature_spawning_enabled: true,
            food_spawn_multiplier: 1.0,
        }
    }
}
