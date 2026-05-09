use std::path::PathBuf;
use std::thread::JoinHandle;

use bevy::prelude::*;
use flume::{Receiver, Sender};
use serde::{Deserialize, Serialize};

use super::sim::{
    BasicCreature, BasicFood, BasicObstacle, BasicPoison, BasicTerrainTile, RunnerReq, RunnerRes,
    TerrainMaterial,
};

pub const FOOD_KIND_COUNT: usize = 7;
pub const POISON_KIND_COUNT: usize = 5;
pub const TERRAIN_MATERIAL_COUNT: usize = 13;

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
    pub poison: Vec<BasicPoison>,
    pub obstacles: Vec<BasicObstacle>,
    pub terrain: Vec<BasicTerrainTile>,
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
    // Last `initial_num_creatures` we pushed to the runner. Used to dedupe
    // SetTargetPopulation messages when the text field changes.
    pub last_pushed_target_population: Option<usize>,
    pub show_signal_field: bool,
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
            last_pushed_target_population: None,
            show_signal_field: true,
        }
    }
}

impl ControlPanel {
    pub fn from_cached_settings(settings: &CachedSettings) -> Self {
        let mut panel = Self::default();
        panel.initial_num_creatures = settings.initial_num_creatures.to_string();
        panel.width = settings.world_width.to_string();
        panel.height = settings.world_height.to_string();
        panel
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

#[derive(Resource, Debug, Clone)]
pub struct WorldBrush {
    pub mode: BrushMode,
    pub radius: f32,
    pub material: TerrainMaterial,
    pub elevation: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrushMode {
    #[default]
    Off,
    Terrain,
    Obstacle,
    EraseObstacle,
    // Click on the world to move the next-generation spawn area's center.
    // Radius is set in the spawn area UI panel.
    SpawnArea,
}

#[derive(Resource, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpawnArea {
    pub center: (f32, f32),
    pub radius: f32,
    pub enabled: bool,
}

impl Default for SpawnArea {
    fn default() -> Self {
        Self {
            center: (1000.0, 1000.0),
            radius: 1000.0,
            enabled: true,
        }
    }
}

impl Default for WorldBrush {
    fn default() -> Self {
        Self {
            mode: BrushMode::Off,
            radius: 80.0,
            material: TerrainMaterial::Grass,
            elevation: 0.45,
        }
    }
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
    #[serde(default)]
    pub initial_genome: Genome,
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
    pub min_food_size: f32,
    pub max_food_size: f32,
    pub food_energy_per_size: f32,
    pub food_eat_ticks_per_size: f32,
    pub poison_damage: f32,
    pub creature_spawning_enabled: bool,
    pub extinction_restart_enabled: bool,
    pub extinction_threshold: f32,
    // When restarting a generation, keep only the top fraction of survivors
    // ranked by (current energy, age). 1.0 = no culling. Lower = stronger
    // selection pressure.
    #[serde(default = "default_next_gen_top_fraction")]
    pub next_gen_top_fraction: f32,
    // When true, only survivors that mated at least once are eligible to seed
    // the next generation (falls back to all survivors if filter empties pool).
    #[serde(default)]
    pub next_gen_only_mated: bool,
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
    #[serde(default = "default_signal_action_cost")]
    pub signal_action_cost: f32,
    // Energy gained per tick from hearing nearby creatures. Falloff is linear with
    // distance over vision range. Net positive incentive to be in earshot of speakers.
    #[serde(default = "default_signal_listener_bonus")]
    pub signal_listener_bonus: f32,
    // When both candidate parents are currently speaking, multiply their effective
    // mate-threshold by (1 - this). 0 = no bonus, 0.9 = strong bias toward
    // talkative pairs mating earlier.
    #[serde(default = "default_mate_voice_bonus")]
    pub mate_voice_bonus: f32,
    #[serde(default = "default_food_biome_rates")]
    pub food_biome_rates: [f32; TERRAIN_MATERIAL_COUNT],
    #[serde(default = "default_poison_biome_rates")]
    pub poison_biome_rates: [f32; TERRAIN_MATERIAL_COUNT],
}

pub fn default_next_gen_top_fraction() -> f32 {
    0.5
}

pub fn default_signal_action_cost() -> f32 {
    0.005
}

pub fn default_signal_listener_bonus() -> f32 {
    0.01
}

pub fn default_mate_voice_bonus() -> f32 {
    0.4
}

pub fn default_food_biome_rates() -> [f32; TERRAIN_MATERIAL_COUNT] {
    [
        0.3, 0.9, 1.8, 0.5, 0.35, 1.8, 2.0, 2.5, 3.2, 1.7, 0.7, 0.35, 0.25,
    ]
}

pub fn default_poison_biome_rates() -> [f32; TERRAIN_MATERIAL_COUNT] {
    [
        0.12, 0.09, 0.13, 0.09, 0.18, 0.11, 0.13, 0.23, 0.32, 0.34, 0.09, 0.08, 0.05,
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub body_size: f32,
    pub reserve_density: f32,
    pub metabolism: f32,
    pub muscle_power: f32,
    pub move_efficiency: f32,
    pub turn_agility: f32,
    pub vision_distance: f32,
    pub fov_angle: f32,
    pub sensory_cost: f32,
    pub bite_size: f32,
    pub digestion_rate: f32,
    pub digest_efficiency: [f32; FOOD_KIND_COUNT],
    pub poison_resist: [f32; POISON_KIND_COUNT],
    pub armor: f32,
    pub cold_tolerance: f32,
    pub heat_tolerance: f32,
    pub water_adaptation: f32,
    pub rough_terrain_adaptation: f32,
    pub mate_threshold_frac: f32,
    pub offspring_energy_frac: f32,
    pub gestation_ticks: u16,
    pub mutation_rate: f32,
    pub mutation_scale: f32,
    pub max_age: u32,
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            body_size: 8.0,
            reserve_density: 2.2,
            metabolism: 0.005,
            muscle_power: 1.0,
            move_efficiency: 1.0,
            turn_agility: 1.0,
            vision_distance: 250.0,
            fov_angle: std::f32::consts::PI * 1.5,
            sensory_cost: 0.000001,
            bite_size: 8.0,
            digestion_rate: 1.0,
            digest_efficiency: [1.0; FOOD_KIND_COUNT],
            poison_resist: [1.0; POISON_KIND_COUNT],
            armor: 1.0,
            cold_tolerance: 1.0,
            heat_tolerance: 1.0,
            water_adaptation: 1.0,
            rough_terrain_adaptation: 1.0,
            mate_threshold_frac: 0.45,
            offspring_energy_frac: 0.25,
            gestation_ticks: 180,
            mutation_rate: 0.05,
            mutation_scale: 0.35,
            max_age: 60_000,
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            initial_genome: Genome::default(),
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
            min_food_size: 2.0,
            max_food_size: 10.0,
            food_energy_per_size: 10.0,
            food_eat_ticks_per_size: 4.0,
            poison_damage: 45.0,
            creature_spawning_enabled: false,
            extinction_restart_enabled: false,
            extinction_threshold: 0.05,
            next_gen_top_fraction: default_next_gen_top_fraction(),
            next_gen_only_mated: false,
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
            signal_action_cost: default_signal_action_cost(),
            signal_listener_bonus: default_signal_listener_bonus(),
            mate_voice_bonus: default_mate_voice_bonus(),
            food_biome_rates: default_food_biome_rates(),
            poison_biome_rates: default_poison_biome_rates(),
        }
    }
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct CachedSettings {
    pub config: SimulationConfig,
    pub world_width: f32,
    pub world_height: f32,
    pub initial_num_creatures: usize,
    #[serde(default)]
    pub primary_window: Option<WindowGeometry>,
    #[serde(default)]
    pub system_window: Option<WindowGeometry>,
    #[serde(default)]
    pub creature_explorer_window: Option<WindowGeometry>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: f32,
    pub height: f32,
}

impl Default for CachedSettings {
    fn default() -> Self {
        let panel = ControlPanel::default();
        Self {
            config: SimulationConfig::default(),
            world_width: panel.width.parse().unwrap_or(2000.0),
            world_height: panel.height.parse().unwrap_or(2000.0),
            initial_num_creatures: panel.initial_num_creatures.parse().unwrap_or(2000),
            primary_window: None,
            system_window: None,
            creature_explorer_window: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum CachedSettingsFile {
    Current(CachedSettings),
    ConfigOnly(SimulationConfig),
}

pub fn config_cache_path() -> PathBuf {
    PathBuf::from(".sim.config")
}

pub fn load_cached_settings() -> CachedSettings {
    let path = config_cache_path();
    match std::fs::read_to_string(&path) {
        Ok(s) => match serde_json::from_str::<CachedSettingsFile>(&s) {
            Ok(CachedSettingsFile::Current(settings)) => settings,
            Ok(CachedSettingsFile::ConfigOnly(config)) => CachedSettings {
                config,
                ..Default::default()
            },
            Err(_) => CachedSettings::default(),
        },
        Err(_) => CachedSettings::default(),
    }
}

pub fn save_cached_settings(config: &SimulationConfig, control_panel: Option<&ControlPanel>) {
    let mut settings = load_cached_settings();
    let default_panel;
    let panel = match control_panel {
        Some(panel) => panel,
        None => {
            default_panel = ControlPanel::default();
            &default_panel
        }
    };
    settings.config = config.clone();
    settings.world_width = panel.width.trim().parse().unwrap_or(2000.0);
    settings.world_height = panel.height.trim().parse().unwrap_or(2000.0);
    settings.initial_num_creatures = panel.initial_num_creatures.trim().parse().unwrap_or(2000);
    save_cached_settings_data(&settings);
}

pub fn save_cached_settings_data(settings: &CachedSettings) {
    if let Ok(s) = serde_json::to_string_pretty(settings) {
        let _ = std::fs::write(config_cache_path(), s);
    }
}
