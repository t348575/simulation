use std::path::PathBuf;
use std::thread::JoinHandle;

use bevy::prelude::*;
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};

use super::sim::{BasicCreature, BasicFood, BasicObstacle, RunnerReq, RunnerRes};

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
    pub obstacles: Vec<BasicObstacle>,
    pub ticks: usize,
}

#[derive(Resource, Debug)]
pub struct ControlPanel {
    pub initial_num_creatures: String,
    pub width: String,
    pub height: String,
    pub can_create_sim: bool,
    pub save_path: String,
    pub save_top_percent: String,
    pub config_path: String,
    pub status: String,
}

#[derive(Resource, Debug, Default)]
pub struct SystemWindow(pub Option<Entity>);

#[derive(Resource, Debug, Default)]
pub struct PendingNet(pub Option<engine::nn::Net>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExplorerSort {
    #[default]
    Id,
    Age,
    Energy,
    Size,
    MateCd,
    ActionLock,
}

#[derive(Resource, Debug, Default)]
pub struct CreatureExplorer {
    pub open: bool,
    pub window: Option<Entity>,
    pub snapshot: Vec<BasicCreature>,
    pub sort_by: ExplorerSort,
    pub sort_desc: bool,
    pub filter_id: String,
}

impl Default for ControlPanel {
    fn default() -> Self {
        Self {
            initial_num_creatures: "2000".to_owned(),
            width: "2000".to_owned(),
            height: "2000".to_owned(),
            can_create_sim: true,
            save_path: "sim.bin".to_owned(),
            save_top_percent: "100".to_owned(),
            config_path: "config.json".to_owned(),
            status: String::new(),
        }
    }
}

#[derive(Component)]
pub struct SystemWindowCamera;

#[derive(Component)]
pub struct CreatureExplorerCamera;

#[derive(Component)]
pub struct SimCamera;

#[derive(Resource, Debug, Clone)]
pub struct ViewState {
    pub zoom: f32,
    pub pan: Vec2,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
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
    pub total_births: usize,
    pub total_deaths: usize,
    pub eat_rate_avg: f32,
    pub birth_rate_avg: f32,
    pub death_rate_avg: f32,
    pub ticks: usize,
    pub generation: usize,
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
    pub ignore_max_age: bool,
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
    pub extinction_restart_enabled: bool,
    pub extinction_threshold: f32,
    pub fov_angle: f32,
    pub vision_distance: f32,
    pub num_obstacles: usize,
    pub min_obstacle_size: f32,
    pub max_obstacle_size: f32,
    pub max_turn_rate: f32,
    pub turn_energy_cost: f32,
    pub attack_range: f32,
    pub attack_damage: f32,
    pub attack_cost: f32,
    pub attack_steal_ratio: f32,
    pub size_speed_penalty: f32,
    pub size_move_cost_factor: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            start_energy: 80.0,
            max_energy: 140.0,
            base_energy_cost: 0.005,
            move_energy_cost: 0.015,
            min_move_energy: 1.0,
            full_speed_energy: 30.0,
            mate_attempt_cost: 0.005,
            eat_attempt_cost: 0.005,
            eat_action_cost: 0.1,
            eat_action_base_ticks: 8,
            mate_energy_cost: 18.0,
            min_mate_energy: 55.0,
            child_energy: 50.0,
            ignore_max_age: true,
            max_age: 60_000,
            min_creature_size: 3.0,
            max_creature_size: 14.0,
            creature_dim: 5.0,
            mate_threshold: 0.45,
            mate_cooldown_ticks: 180,
            mutation_rate: 0.05,
            mutation_amount: 0.35,
            max_births_per_tick: 25,
            food_per_creature: 2,
            min_food_size: 2.0,
            max_food_size: 10.0,
            food_energy_per_size: 10.0,
            food_eat_ticks_per_size: 4.0,
            poison_per_creature: 12,
            poison_damage: 45.0,
            creature_spawning_enabled: false,
            food_spawn_multiplier: 2.0,
            extinction_restart_enabled: false,
            extinction_threshold: 0.05,
            fov_angle: std::f32::consts::PI * 1.5, // 270°
            vision_distance: 250.0,
            num_obstacles: 30,
            min_obstacle_size: 30.0,
            max_obstacle_size: 120.0,
            max_turn_rate: 0.18,
            turn_energy_cost: 0.002,
            attack_range: 18.0,
            attack_damage: 12.0,
            attack_cost: 0.4,
            attack_steal_ratio: 0.6,
            size_speed_penalty: 0.5,
            size_move_cost_factor: 1.0,
        }
    }
}

pub fn config_cache_path() -> PathBuf {
    PathBuf::from(".sim.config")
}

pub fn load_cached_config() -> SimulationConfig {
    let path = config_cache_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str::<SimulationConfig>(&s).unwrap_or_default(),
        Err(_) => SimulationConfig::default(),
    }
}

pub fn save_cached_config(config: &SimulationConfig) {
    let path = config_cache_path();
    if let Ok(s) = serde_json::to_string_pretty(config) {
        let _ = std::fs::write(path, s);
    }
}
