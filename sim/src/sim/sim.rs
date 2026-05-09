use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicUsize, Ordering},
    thread::sleep,
    time::{Duration, Instant},
};

const RATE_WINDOW_TICKS: usize = 120;

use bevy::{math::Vec2, prelude::info};
use dashmap::{DashMap, DashSet};
use engine::nn::{
    mutate::{AddNeuron, NeuronMutator, RemoveEdge, RemoveNeuron, LinkMutator},
    BasicNeuron, Edge, GraphLocation, GraphNode, Net, Neuron, Node,
};
use flume::{unbounded, Receiver, Sender};
use rand::{Rng, RngExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::gpu::{
    GpuBrainCompute, GpuCreatureSensor, GpuFoodSensor, GpuObstacleSensor, GpuPoisonSensor,
    GpuSensorWorld, GPU_INPUTS, GPU_OUTPUTS,
};
use super::resources::{
    Genome, SimulationConfig, SimulationStats, SpawnArea, FOOD_KIND_COUNT, POISON_KIND_COUNT,
    TERRAIN_MATERIAL_COUNT,
};
use super::spatial::Grid;

const GRID_CELL: f32 = 32.0;
const TERRAIN_TILE_SIZE: f32 = 64.0;
// Terrain expansion + resource refill are idempotent at steady state. Running
// them every tick is wasted work once the world is mostly explored.
const TERRAIN_EXPAND_INTERVAL: usize = 30;
const RESOURCE_REFILL_INTERVAL: usize = 30;
const BASE_BODY_SIZE: f32 = 8.0;
const BASE_MOVE_SPEED: f32 = 4.0;
const STARVATION_STRESS_COST: f32 = 0.02;
static DEFAULT_TERRAIN_TILE: TerrainTile = TerrainTile {
    position: (0.0, 0.0),
    size: (1.0, 1.0),
    elevation: 0.5,
    material: TerrainMaterial::Grass,
};

pub const CREATURE_DIM: f32 = 5.0;
pub const CREATURE_DIM_HALF: f32 = CREATURE_DIM / 2.0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Simulation {
    world_dim: (f32, f32),
    creatures: DashMap<usize, Creature>,
    food: Vec<Food>,
    poison: Vec<Poison>,
    obstacles: Vec<Obstacle>,
    #[serde(default)]
    terrain: Vec<TerrainTile>,
    #[serde(default)]
    terrain_seed: f32,
    #[serde(default)]
    generated_terrain_tiles: HashSet<(i32, i32)>,
    // Spatial index over `terrain`, keyed by tile coord. Maintained alongside
    // `generated_terrain_tiles`; rebuilt whenever `terrain` mutates. Hot-path
    // sensor lookups use this instead of an O(N) Vec scan.
    #[serde(skip)]
    terrain_index: HashMap<(i32, i32), usize>,
    // Per-biome bucket of terrain indices. Lets `random_terrain_tile_for_biome`
    // pick uniformly from a biome in O(1) after an O(T) rebuild, instead of
    // scanning all terrain on every food/poison spawn.
    #[serde(skip)]
    terrain_by_biome: Vec<Vec<usize>>,
    #[serde(default)]
    spawn_area: SpawnArea,
    last_id: usize,
    ticks: usize,
    target_population: usize,
    input_nodes: Vec<Node>,
    output_nodes: Vec<Node>,
    previous_ids: HashSet<usize>,
    total_spawned: usize,
    config: SimulationConfig,
    food_eaten_this_tick: usize,
    total_food_eaten: usize,
    births_this_tick: usize,
    deaths_this_tick: usize,
    total_births: usize,
    total_deaths: usize,
    recent_eats: VecDeque<usize>,
    recent_births: VecDeque<usize>,
    recent_deaths: VecDeque<usize>,
    generation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Creature {
    brain: Net,
    genome: Genome,
    position: (f32, f32),
    angle: f32,
    energy: f32,
    age: u32,
    mate_cooldown: u16,
    action_lock: u16,
    mem: f32,
    prev_energy: f32,
    touched: f32,
    #[serde(default)]
    signal: f32,
    // Lifetime mate count. Used by restart_generation to gate eligibility
    // when `next_gen_only_mated` is set.
    #[serde(default)]
    times_mated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Obstacle {
    position: (f32, f32),
    half_width: f32,
    half_height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TerrainMaterial {
    DeepWater,
    Water,
    ShallowWater,
    Sand,
    Desert,
    Savanna,
    Grass,
    Forest,
    Rainforest,
    Marsh,
    Tundra,
    Rock,
    #[default]
    Snow,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct TerrainTile {
    position: (f32, f32),
    size: (f32, f32),
    elevation: f32,
    material: TerrainMaterial,
}

#[derive(Debug, Clone, Copy, Default)]
struct TerrainEffects {
    elevation: f32,
    speed: f32,
    energy_cost: f32,
    hazard: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicCreature {
    pub position: (f32, f32),
    pub angle: f32,
    pub id: usize,
    pub size: f32,
    pub energy: f32,
    pub age: u32,
    pub mate_cooldown: u16,
    pub action_lock: u16,
    #[serde(default)]
    pub signal: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicObstacle {
    pub position: (f32, f32),
    pub half_width: f32,
    pub half_height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicTerrainTile {
    pub position: (f32, f32),
    pub size: (f32, f32),
    pub elevation: f32,
    pub material: TerrainMaterial,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicFood {
    pub position: (f32, f32),
    pub size: f32,
    #[serde(default)]
    pub kind: FoodKind,
    #[serde(default)]
    pub energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicPoison {
    pub position: (f32, f32),
    #[serde(default)]
    pub kind: PoisonKind,
    #[serde(default)]
    pub damage: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FoodKind {
    #[default]
    Berries,
    Fruit,
    Fungus,
    Kelp,
    CactusFruit,
    Lichen,
    Seeds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoisonKind {
    #[default]
    Nightshade,
    ToxicMushroom,
    BrineBloom,
    ThornPatch,
    BitterLichen,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Food {
    position: (f32, f32),
    size: f32,
    #[serde(default)]
    kind: FoodKind,
    #[serde(default)]
    energy: f32,
    #[serde(default)]
    eat_ticks: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Poison {
    position: (f32, f32),
    #[serde(default)]
    kind: PoisonKind,
    #[serde(default)]
    damage: f32,
}

pub struct Runner {
    rx: Receiver<RunnerReq>,
    tx: Sender<RunnerRes>,
    sim: Simulation,
    paused: bool,
    gpu: GpuBrainCompute,
}

#[derive(Debug, Clone)]
pub enum RunnerReq {
    Generate(Generate),
    Resume,
    Pause,
    GetNet(usize),
    UpdateConfig(SimulationConfig),
    SaveSim(std::path::PathBuf),
    LoadSim(std::path::PathBuf, Vec<Node>, Vec<Node>),
    SaveCreatures(std::path::PathBuf, f32),
    LoadCreatures(
        std::path::PathBuf,
        (f32, f32),
        SimulationConfig,
        Vec<Node>,
        Vec<Node>,
    ),
    NewGeneration,
    RecreateWorld((f32, f32)),
    PaintWorld(WorldPaint),
    SetSpawnArea(SpawnArea),
    SetTargetPopulation(usize),
}

#[derive(Debug, Clone)]
pub struct WorldPaint {
    pub position: (f32, f32),
    pub radius: f32,
    pub action: WorldPaintAction,
}

#[derive(Debug, Clone)]
pub enum WorldPaintAction {
    Terrain {
        material: TerrainMaterial,
        elevation: f32,
    },
    AddObstacle,
    EraseObstacle,
}

#[derive(Debug, Clone)]
pub enum RunnerRes {
    Positions(Positions),
    Net(Option<Net>),
    SaveResult(Result<std::path::PathBuf, String>),
    LoadResult(Result<std::path::PathBuf, String>),
}

#[derive(Debug, Clone)]
pub struct Positions {
    pub creatures: Vec<BasicCreature>,
    pub food: Vec<BasicFood>,
    pub poison: Vec<BasicPoison>,
    pub obstacles: Vec<BasicObstacle>,
    pub terrain: Vec<BasicTerrainTile>,
    pub stats: SimulationStats,
}

#[derive(Debug, Clone)]
pub struct Generate {
    pub num_creatures: usize,
    pub input_nodes: Vec<Node>,
    pub output_nodes: Vec<Node>,
    pub dims: (f32, f32),
    pub config: SimulationConfig,
}

impl Runner {
    pub fn new() -> (Self, Sender<RunnerReq>, Receiver<RunnerRes>) {
        let (tx_req, rx_req) = unbounded();
        let (tx_res, rx_res) = unbounded();
        let gpu = GpuBrainCompute::new().expect(
            "GPU compute is required (no CPU fallback). Could not initialize wgpu adapter/device.",
        );
        let r = Runner {
            rx: rx_req,
            tx: tx_res,
            sim: Simulation::default(),
            paused: true,
            gpu,
        };
        (r, tx_req, rx_res)
    }

    pub fn run(mut self) {
        loop {
            while let Ok(msg) = self.rx.try_recv() {
                match msg {
                    RunnerReq::Generate(g) => {
                        let mut rng = rand::rng();
                        let width = g.dims.0;
                        let height = g.dims.1;
                        self.sim = Simulation::default();
                        self.sim.world_dim = g.dims;
                        self.sim.target_population = g.num_creatures;
                        self.sim.input_nodes = g.input_nodes;
                        self.sim.output_nodes = g.output_nodes;
                        self.sim.config = g.config;
                        self.sim.terrain_seed = rng.random_range(0.0..1000.0);
                        self.sim.terrain =
                            generate_terrain_with_seed(self.sim.terrain_seed, g.dims);
                        self.sim.rebuild_generated_terrain_tiles();
                        self.sim.obstacles = generate_obstacles(&mut rng, g.dims, &self.sim.config);

                        self.gpu.clear_all();
                        self.sim.creatures = (0..g.num_creatures)
                            .map(|_i| {
                                self.sim.last_id += 1;
                                let creature = self.sim.random_creature(&mut rng, (width, height));
                                self.gpu
                                    .assign_brain(self.sim.last_id, &creature.brain)
                                    .expect("preset/random brain must compile for GPU");
                                (self.sim.last_id, creature)
                            })
                            .collect();
                        self.sim.total_spawned = g.num_creatures;
                        self.sim.previous_ids =
                            self.sim.creatures.iter().map(|c| *c.key()).collect();
                        self.sim.refill_food_and_poison(&mut rng);
                        self.tx
                            .send(RunnerRes::Positions(self.positions()))
                            .expect("send positions");
                    }
                    RunnerReq::Resume => self.paused = false,
                    RunnerReq::Pause => self.paused = true,
                    RunnerReq::GetNet(id) => {
                        match self.sim.creatures.iter().find(|c| *c.key() == id) {
                            Some(c) => self.tx.send(RunnerRes::Net(Some(c.brain.clone()))),
                            None => self.tx.send(RunnerRes::Net(None)),
                        }
                        .expect("send net")
                    }
                    RunnerReq::UpdateConfig(config) => {
                        self.sim.config = config;
                    }
                    RunnerReq::SaveSim(path) => {
                        self.paused = true;
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            let cfg = bincode::config::standard();
                            let bytes = bincode::serde::encode_to_vec(&self.sim, cfg)
                                .map_err(|e| format!("serialize: {e}"))?;
                            std::fs::write(&path, &bytes).map_err(|e| format!("write: {e}"))?;
                            info!("Saved sim ({} bytes) → {}", bytes.len(), path.display());
                            Ok(path)
                        })();
                        let _ = self.tx.send(RunnerRes::SaveResult(result));
                    }
                    RunnerReq::SaveCreatures(path, top_percent) => {
                        self.paused = true;
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            let pct = top_percent.clamp(0.0, 100.0);
                            let total = self.sim.creatures.len();
                            let keep_n = ((total as f32) * (pct / 100.0)).ceil() as usize;
                            let keep_n = keep_n.min(total).max(if total > 0 { 1 } else { 0 });
                            let mut by_age: Vec<(usize, u32)> = self
                                .sim
                                .creatures
                                .iter()
                                .map(|c| (*c.key(), c.value().age))
                                .collect();
                            by_age.sort_by(|a, b| b.1.cmp(&a.1));
                            by_age.truncate(keep_n);
                            let creatures: Vec<Creature> = by_age
                                .iter()
                                .filter_map(|(id, _)| {
                                    self.sim.creatures.get(id).map(|c| c.value().clone())
                                })
                                .collect();
                            let cfg = bincode::config::standard();
                            let bytes = bincode::serde::encode_to_vec(&creatures, cfg)
                                .map_err(|e| format!("serialize: {e}"))?;
                            std::fs::write(&path, &bytes).map_err(|e| format!("write: {e}"))?;
                            info!(
                                "Saved {}/{} creatures (top {:.1}%, {} bytes) → {}",
                                keep_n,
                                total,
                                pct,
                                bytes.len(),
                                path.display()
                            );
                            Ok(path)
                        })();
                        let _ = self.tx.send(RunnerRes::SaveResult(result));
                    }
                    RunnerReq::LoadCreatures(path, dims, config, input_nodes, output_nodes) => {
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            let bytes = std::fs::read(&path).map_err(|e| format!("read: {e}"))?;
                            let cfg = bincode::config::standard();
                            let (loaded, _read): (Vec<Creature>, usize) =
                                bincode::serde::decode_from_slice(&bytes, cfg)
                                    .map_err(|e| format!("deserialize: {e}"))?;

                            let mut rng = rand::rng();
                            self.sim.world_dim = dims;
                            self.sim.config = config;
                            self.sim.input_nodes = input_nodes;
                            self.sim.output_nodes = output_nodes;
                            self.sim.target_population = loaded.len();
                            self.sim.terrain_seed = rng.random_range(0.0..1000.0);
                            self.sim.terrain =
                                generate_terrain_with_seed(self.sim.terrain_seed, dims);
                            self.sim.rebuild_generated_terrain_tiles();
                            self.sim.obstacles =
                                generate_obstacles(&mut rng, dims, &self.sim.config);
                            self.sim.food.clear();
                            self.sim.poison.clear();

                            // Drop existing creatures, free GPU slots.
                            let old_ids: Vec<usize> =
                                self.sim.creatures.iter().map(|c| *c.key()).collect();
                            for id in old_ids {
                                self.gpu.release(id);
                            }
                            self.sim.creatures.clear();

                            let obstacles = self.sim.obstacles.clone();
                            let count = loaded.len();
                            for mut c in loaded {
                                ensure_brain_io(
                                    &mut c.brain,
                                    &self.sim.input_nodes,
                                    &self.sim.output_nodes,
                                );
                                c.position =
                                    random_pos_outside_obstacles(&mut rng, dims, &obstacles);
                                c.angle =
                                    rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                                c.age = 0;
                                c.energy =
                                    self.sim.config.start_energy.min(max_energy_for(&c.genome));
                                c.prev_energy = c.energy;
                                c.mate_cooldown = 0;
                                c.action_lock = 0;
                                c.touched = 0.0;
                                c.mem = 0.0;
                                self.sim.last_id += 1;
                                self.gpu
                                    .assign_brain(self.sim.last_id, &c.brain)
                                    .map_err(|e| {
                                        format!(
                                            "creature {} brain not GPU-compatible: {e}",
                                            self.sim.last_id
                                        )
                                    })?;
                                self.sim.creatures.insert(self.sim.last_id, c);
                                self.sim.total_spawned += 1;
                            }
                            self.sim.refill_food_and_poison(&mut rng);
                            self.sim.previous_ids =
                                self.sim.creatures.iter().map(|c| *c.key()).collect();
                            info!("Loaded {} creatures ← {}", count, path.display());
                            Ok(path)
                        })();
                        if result.is_ok() {
                            let _ = self.tx.send(RunnerRes::Positions(self.positions()));
                        }
                        let _ = self.tx.send(RunnerRes::LoadResult(result));
                    }
                    RunnerReq::LoadSim(path, input_nodes, output_nodes) => {
                        self.paused = true;
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            let bytes = std::fs::read(&path).map_err(|e| format!("read: {e}"))?;
                            let cfg = bincode::config::standard();
                            let (loaded, _read): (Simulation, usize) =
                                bincode::serde::decode_from_slice(&bytes, cfg)
                                    .map_err(|e| format!("deserialize: {e}"))?;
                            self.sim = loaded;
                            self.sim.input_nodes = input_nodes;
                            self.sim.output_nodes = output_nodes;
                            if self.sim.terrain.is_empty() {
                                let mut rng = rand::rng();
                                self.sim.terrain_seed = rng.random_range(0.0..1000.0);
                                self.sim.terrain = generate_terrain_with_seed(
                                    self.sim.terrain_seed,
                                    self.sim.world_dim,
                                );
                            }
                            self.sim.rebuild_generated_terrain_tiles();

                            // Rebuild GPU weight slots from each creature's brain.
                            self.gpu.clear_all();
                            for mut entry in self.sim.creatures.iter_mut() {
                                let id = *entry.key();
                                ensure_brain_io(
                                    &mut entry.value_mut().brain,
                                    &self.sim.input_nodes,
                                    &self.sim.output_nodes,
                                );
                                self.gpu
                                    .assign_brain(id, &entry.value().brain)
                                    .map_err(|e| {
                                        format!("creature {id} brain not GPU-compatible: {e}")
                                    })?;
                            }
                            self.sim.previous_ids =
                                self.sim.creatures.iter().map(|c| *c.key()).collect();
                            info!(
                                "Loaded sim ({} creatures) ← {}",
                                self.sim.creatures.len(),
                                path.display()
                            );
                            Ok(path)
                        })();
                        if result.is_ok() {
                            let _ = self.tx.send(RunnerRes::Positions(self.positions()));
                        }
                        let _ = self.tx.send(RunnerRes::LoadResult(result));
                    }
                    RunnerReq::NewGeneration => {
                        if !self.sim.creatures.is_empty() {
                            self.sim.restart_generation(&mut self.gpu);
                            let _ = self.tx.send(RunnerRes::Positions(self.positions()));
                        }
                    }
                    RunnerReq::RecreateWorld(dims) => {
                        let mut rng = rand::rng();
                        self.sim.world_dim = dims;
                        self.sim.terrain_seed = rng.random_range(0.0..1000.0);
                        self.sim.terrain = generate_terrain_with_seed(self.sim.terrain_seed, dims);
                        self.sim.rebuild_generated_terrain_tiles();
                        self.sim.obstacles = generate_obstacles(&mut rng, dims, &self.sim.config);
                        self.sim.food.clear();
                        self.sim.poison.clear();
                        self.sim.refill_food_and_poison(&mut rng);
                        let obstacles = self.sim.obstacles.clone();
                        for mut entry in self.sim.creatures.iter_mut() {
                            let c = entry.value_mut();
                            c.position = random_pos_outside_obstacles(&mut rng, dims, &obstacles);
                            c.angle = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
                            c.touched = 0.0;
                        }
                        let _ = self.tx.send(RunnerRes::Positions(self.positions()));
                    }
                    RunnerReq::PaintWorld(paint) => {
                        self.sim.paint_world(paint);
                        let _ = self.tx.send(RunnerRes::Positions(self.positions()));
                    }
                    RunnerReq::SetSpawnArea(area) => {
                        self.sim.spawn_area = area;
                    }
                    RunnerReq::SetTargetPopulation(n) => {
                        self.sim.target_population = n.max(1);
                    }
                }
            }

            if self.paused {
                sleep(Duration::from_millis(100));
                continue;
            }

            self.sim.run(&mut self.gpu);
            self.sim.ticks += 1;

            if self.tx.len() == 0 {
                _ = self.tx.send(RunnerRes::Positions(self.positions()));
                self.sim.previous_ids = self.sim.creatures.iter().map(|c| *c.key()).collect();
            }
        }
    }

    fn positions(&self) -> Positions {
        let creatures: Vec<BasicCreature> = self
            .sim
            .creatures
            .par_iter()
            .map(|x| BasicCreature {
                position: x.value().position,
                angle: x.value().angle,
                id: *x.key(),
                size: creature_size(&x.value().genome),
                energy: x.value().energy,
                age: x.value().age,
                mate_cooldown: x.value().mate_cooldown,
                action_lock: x.value().action_lock,
                signal: x.value().signal,
            })
            .collect();

        let (total_energy, total_age, total_size) =
            self.sim
                .creatures
                .iter()
                .fold((0.0f32, 0u32, 0.0f32), |acc, c| {
                    let creature = c.value();
                    (
                        acc.0 + creature.energy,
                        acc.1 + creature.age,
                        acc.2 + creature_size(&creature.genome),
                    )
                });

        let count = self.sim.creatures.len();
        let avg_energy = if count > 0 {
            total_energy / count as f32
        } else {
            0.0
        };
        let avg_age = if count > 0 {
            total_age as f32 / count as f32
        } else {
            0.0
        };
        let avg_size = if count > 0 {
            total_size / count as f32
        } else {
            0.0
        };

        let survival_rate = if self.sim.total_spawned > 0 {
            self.sim.creatures.len() as f32 / self.sim.total_spawned as f32
        } else {
            0.0
        };

        Positions {
            creatures,
            food: self
                .sim
                .food
                .iter()
                .map(|food| BasicFood {
                    position: food.position,
                    size: food.size,
                    kind: food.kind,
                    energy: food.energy,
                })
                .collect(),
            poison: self
                .sim
                .poison
                .iter()
                .map(|poison| BasicPoison {
                    position: poison.position,
                    kind: poison.kind,
                    damage: poison.damage,
                })
                .collect(),
            obstacles: self
                .sim
                .obstacles
                .iter()
                .map(|o| BasicObstacle {
                    position: o.position,
                    half_width: o.half_width,
                    half_height: o.half_height,
                })
                .collect(),
            terrain: self
                .sim
                .terrain
                .iter()
                .map(|tile| BasicTerrainTile {
                    position: tile.position,
                    size: tile.size,
                    elevation: tile.elevation,
                    material: tile.material,
                })
                .collect(),
            stats: SimulationStats {
                current_population: self.sim.creatures.len(),
                target_population: self.sim.target_population,
                births_this_tick: self.sim.births_this_tick,
                deaths_this_tick: self.sim.deaths_this_tick,
                total_spawned: self.sim.total_spawned,
                avg_energy,
                avg_age,
                avg_size,
                food_count: self.sim.food.len(),
                poison_count: self.sim.poison.len(),
                survival_rate,
                selected_creature_id: None,
                food_eaten_this_tick: self.sim.food_eaten_this_tick,
                total_food_eaten: self.sim.total_food_eaten,
                total_births: self.sim.total_births,
                total_deaths: self.sim.total_deaths,
                eat_rate_avg: window_avg(&self.sim.recent_eats),
                birth_rate_avg: window_avg(&self.sim.recent_births),
                death_rate_avg: window_avg(&self.sim.recent_deaths),
                ticks: self.sim.ticks,
                generation: self.sim.generation,
            },
        }
    }
}

fn push_window(window: &mut VecDeque<usize>, value: usize) {
    window.push_back(value);
    while window.len() > RATE_WINDOW_TICKS {
        window.pop_front();
    }
}

fn window_avg(window: &VecDeque<usize>) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    window.iter().sum::<usize>() as f32 / window.len() as f32
}

fn get_output_value(value: &Node) -> f32 {
    if let Node::Output(o) = value {
        o.value()
    } else {
        0.0
    }
}

fn do_squares_collide(a: (f32, f32), b: (f32, f32)) -> bool {
    let a_min_x = a.0 - CREATURE_DIM_HALF;
    let a_max_x = a.0 + CREATURE_DIM_HALF;
    let a_min_y = a.1 - CREATURE_DIM_HALF;
    let a_max_y = a.1 + CREATURE_DIM_HALF;

    let b_min_x = b.0 - CREATURE_DIM_HALF;
    let b_max_x = b.0 + CREATURE_DIM_HALF;
    let b_min_y = b.1 - CREATURE_DIM_HALF;
    let b_max_y = b.1 + CREATURE_DIM_HALF;

    a_min_x < b_max_x && a_max_x > b_min_x && a_min_y < b_max_y && a_max_y > b_min_y
}

fn do_sized_squares_collide(a: (f32, f32), a_size: f32, b: (f32, f32), b_size: f32) -> bool {
    let a_half = a_size / 2.0;
    let b_half = b_size / 2.0;
    let a_min_x = a.0 - a_half;
    let a_max_x = a.0 + a_half;
    let a_min_y = a.1 - a_half;
    let a_max_y = a.1 + a_half;

    let b_min_x = b.0 - b_half;
    let b_max_x = b.0 + b_half;
    let b_min_y = b.1 - b_half;
    let b_max_y = b.1 + b_half;

    a_min_x < b_max_x && a_max_x > b_min_x && a_min_y < b_max_y && a_max_y > b_min_y
}

fn creature_size(genome: &Genome) -> f32 {
    genome.body_size
}

fn body_mass(genome: &Genome) -> f32 {
    genome.body_size.max(0.5).powi(2)
}

fn max_energy_for(genome: &Genome) -> f32 {
    (genome.reserve_density * body_mass(genome)).max(1.0)
}

fn energy_frac(energy: f32, genome: &Genome) -> f32 {
    (energy / max_energy_for(genome)).clamp(0.0, 1.0)
}

fn body_size_speed_scale(genome: &Genome) -> f32 {
    (BASE_BODY_SIZE / genome.body_size.max(1.0)).clamp(0.25, 3.0)
}

fn food_kind_index(kind: FoodKind) -> usize {
    match kind {
        FoodKind::Berries => 0,
        FoodKind::Fruit => 1,
        FoodKind::Fungus => 2,
        FoodKind::Kelp => 3,
        FoodKind::CactusFruit => 4,
        FoodKind::Lichen => 5,
        FoodKind::Seeds => 6,
    }
}

fn poison_kind_index(kind: PoisonKind) -> usize {
    match kind {
        PoisonKind::Nightshade => 0,
        PoisonKind::ToxicMushroom => 1,
        PoisonKind::BrineBloom => 2,
        PoisonKind::ThornPatch => 3,
        PoisonKind::BitterLichen => 4,
    }
}

fn terrain_material_index(material: TerrainMaterial) -> usize {
    match material {
        TerrainMaterial::DeepWater => 0,
        TerrainMaterial::Water => 1,
        TerrainMaterial::ShallowWater => 2,
        TerrainMaterial::Sand => 3,
        TerrainMaterial::Desert => 4,
        TerrainMaterial::Savanna => 5,
        TerrainMaterial::Grass => 6,
        TerrainMaterial::Forest => 7,
        TerrainMaterial::Rainforest => 8,
        TerrainMaterial::Marsh => 9,
        TerrainMaterial::Tundra => 10,
        TerrainMaterial::Rock => 11,
        TerrainMaterial::Snow => 12,
    }
}

fn biome_modifier(genome: &Genome, material: TerrainMaterial) -> (f32, f32, f32) {
    let cold = genome.cold_tolerance.max(0.1);
    let heat = genome.heat_tolerance.max(0.1);
    let water = genome.water_adaptation.max(0.1);
    let rough = genome.rough_terrain_adaptation.max(0.1);
    match material {
        TerrainMaterial::DeepWater | TerrainMaterial::Water | TerrainMaterial::ShallowWater => {
            (water.sqrt(), 1.0 / water, 1.0 / water)
        }
        TerrainMaterial::Marsh => (water.sqrt(), 1.0 / water, 1.0 / cold.sqrt()),
        TerrainMaterial::Tundra | TerrainMaterial::Snow => (cold.sqrt(), 1.0 / cold, 1.0 / cold),
        TerrainMaterial::Sand | TerrainMaterial::Desert => (heat.sqrt(), 1.0 / heat, 1.0 / heat),
        TerrainMaterial::Forest | TerrainMaterial::Rainforest | TerrainMaterial::Rock => {
            (rough.sqrt(), 1.0 / rough, 1.0)
        }
        _ => (1.0, 1.0, 1.0),
    }
}

fn terrain_effects_for(genome: &Genome, tile: &TerrainTile) -> TerrainEffects {
    let base = terrain_effects(tile.material, tile.elevation);
    let (speed_mult, cost_mult, hazard_mult) = biome_modifier(genome, tile.material);
    TerrainEffects {
        elevation: base.elevation,
        speed: (base.speed * speed_mult).clamp(0.05, 3.0),
        energy_cost: (base.energy_cost * cost_mult).clamp(0.05, 5.0),
        hazard: (base.hazard * hazard_mult).max(0.0),
    }
}

fn generate_terrain_with_seed(seed: f32, dims: (f32, f32)) -> Vec<TerrainTile> {
    let cols = (dims.0 / TERRAIN_TILE_SIZE).ceil().max(1.0) as usize;
    let rows = (dims.1 / TERRAIN_TILE_SIZE).ceil().max(1.0) as usize;
    let mut tiles = Vec::with_capacity(cols * rows);

    for row in 0..rows {
        for col in 0..cols {
            tiles.push(terrain_tile_for_coord(
                col as i32,
                row as i32,
                seed,
                (TERRAIN_TILE_SIZE, TERRAIN_TILE_SIZE),
            ));
        }
    }

    tiles
}

fn terrain_tile_for_coord(tx: i32, ty: i32, seed: f32, size: (f32, f32)) -> TerrainTile {
    let x = tx as f32 * TERRAIN_TILE_SIZE;
    let y = ty as f32 * TERRAIN_TILE_SIZE;
    let nx = (x + size.0 * 0.5) / TERRAIN_TILE_SIZE;
    let ny = (y + size.1 * 0.5) / TERRAIN_TILE_SIZE;
    let elevation = terrain_elevation(nx, ny, seed);
    let moisture = terrain_moisture(nx, ny, elevation, seed);
    let temperature = terrain_temperature(nx, ny, elevation, seed);
    let material = terrain_material(elevation, moisture, temperature);
    TerrainTile {
        position: (x + size.0 * 0.5, y + size.1 * 0.5),
        size,
        elevation,
        material,
    }
}

fn terrain_elevation(nx: f32, ny: f32, seed: f32) -> f32 {
    let continents = fbm(nx * 0.055 + 3.1, ny * 0.055 - 7.2, seed, 5);
    let ridges = 1.0 - (fbm(nx * 0.18 - 13.0, ny * 0.18 + 2.0, seed + 71.0, 4) - 0.5).abs() * 2.0;
    let detail = fbm(nx * 0.55 + 19.0, ny * 0.55 - 23.0, seed + 137.0, 3);
    (continents * 0.72 + ridges * ridges * 0.2 + detail * 0.08).clamp(0.0, 1.0)
}

fn terrain_moisture(nx: f32, ny: f32, elevation: f32, seed: f32) -> f32 {
    let weather = fbm(nx * 0.075 + 29.0, ny * 0.075 - 41.0, seed + 211.0, 5);
    let coastal = (1.0 - ((elevation - 0.28).max(0.0) / 0.45)).clamp(0.0, 1.0);
    let mountain_shadow = if elevation > 0.68 { 0.2 } else { 0.0 };
    (weather * 0.82 + coastal * 0.18 - mountain_shadow).clamp(0.0, 1.0)
}

fn terrain_temperature(nx: f32, ny: f32, elevation: f32, seed: f32) -> f32 {
    let latitude = 0.5 + (ny * 0.025).sin() * 0.45;
    let climate = fbm(nx * 0.055 - 17.0, ny * 0.055 + 31.0, seed + 307.0, 4) - 0.5;
    (latitude * 0.8 + climate * 0.25 - elevation * 0.35).clamp(0.0, 1.0)
}

fn fbm(mut x: f32, mut y: f32, seed: f32, octaves: usize) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 0.5;
    let mut total = 0.0;
    for octave in 0..octaves {
        value += value_noise(x, y, seed + octave as f32 * 17.0) * amplitude;
        total += amplitude;
        x *= 2.0;
        y *= 2.0;
        amplitude *= 0.5;
    }
    (value / total.max(f32::EPSILON)).clamp(0.0, 1.0)
}

fn value_noise(x: f32, y: f32, seed: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = smoothstep(x - x0);
    let ty = smoothstep(y - y0);
    let a = lattice_noise(x0, y0, seed);
    let b = lattice_noise(x0 + 1.0, y0, seed);
    let c = lattice_noise(x0, y0 + 1.0, seed);
    let d = lattice_noise(x0 + 1.0, y0 + 1.0, seed);
    lerp(lerp(a, b, tx), lerp(c, d, tx), ty)
}

fn lattice_noise(x: f32, y: f32, seed: f32) -> f32 {
    let n = (x * 127.1 + y * 311.7 + seed * 74.7).sin() * 43_758.547;
    n.fract().abs()
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn terrain_material(elevation: f32, moisture: f32, temperature: f32) -> TerrainMaterial {
    if elevation < 0.18 {
        TerrainMaterial::DeepWater
    } else if elevation < 0.25 {
        TerrainMaterial::Water
    } else if elevation < 0.29 {
        TerrainMaterial::ShallowWater
    } else if elevation < 0.31 {
        TerrainMaterial::Sand
    } else if elevation < 0.36 && moisture > 0.62 {
        TerrainMaterial::Marsh
    } else if elevation > 0.82 {
        TerrainMaterial::Snow
    } else if elevation > 0.68 {
        if temperature < 0.28 || elevation > 0.78 {
            TerrainMaterial::Snow
        } else {
            TerrainMaterial::Rock
        }
    } else if temperature < 0.22 {
        TerrainMaterial::Tundra
    } else if moisture < 0.22 && temperature > 0.55 {
        TerrainMaterial::Desert
    } else if moisture < 0.35 && temperature > 0.42 {
        TerrainMaterial::Savanna
    } else if moisture > 0.72 && temperature > 0.55 {
        TerrainMaterial::Rainforest
    } else if moisture > 0.5 {
        TerrainMaterial::Forest
    } else {
        TerrainMaterial::Grass
    }
}

fn terrain_coord(position: (f32, f32)) -> (i32, i32) {
    (
        (position.0 / TERRAIN_TILE_SIZE).floor() as i32,
        (position.1 / TERRAIN_TILE_SIZE).floor() as i32,
    )
}

fn terrain_effects(material: TerrainMaterial, elevation: f32) -> TerrainEffects {
    let slope_cost = if elevation > 0.6 {
        1.0 + (elevation - 0.6) * 0.8
    } else {
        1.0
    };
    let (speed, energy_cost, hazard) = match material {
        TerrainMaterial::DeepWater => (0.25, 2.3, 0.025),
        TerrainMaterial::Water => (0.4, 1.9, 0.01),
        TerrainMaterial::ShallowWater => (0.55, 1.6, 0.004),
        TerrainMaterial::Sand => (0.75, 1.25, 0.0),
        TerrainMaterial::Desert => (0.82, 1.45, 0.003),
        TerrainMaterial::Savanna => (0.95, 1.05, 0.0),
        TerrainMaterial::Grass => (1.0, 1.0, 0.0),
        TerrainMaterial::Forest => (0.78, 1.2, 0.0),
        TerrainMaterial::Rainforest => (0.62, 1.45, 0.0),
        TerrainMaterial::Marsh => (0.42, 1.9, 0.002),
        TerrainMaterial::Tundra => (0.72, 1.35, 0.002),
        TerrainMaterial::Rock => (0.58, 1.65, 0.0),
        TerrainMaterial::Snow => (0.48, 1.9, 0.004),
    };
    TerrainEffects {
        elevation,
        speed,
        energy_cost: energy_cost * slope_cost,
        hazard,
    }
}

// 4-sector look-ahead for terrain. Each sector returns (cost_norm, hazard_norm)
// using the same normalization as inputs[34]/[35]: cost = (raw-1)/1.5 clamped,
// hazard = raw/0.05 clamped. Sectors split the FOV cone into four equal angular
// bins (matching food_s0..3 layout).
fn sample_terrain_sectors(
    terrain: &[TerrainTile],
    terrain_index: &HashMap<(i32, i32), usize>,
    _dims: (f32, f32),
    pos: (f32, f32),
    angle: f32,
    half_fov: f32,
    vision: f32,
    genome: &Genome,
) -> [(f32, f32); 4] {
    const RAYS_PER_SECTOR: usize = 3;
    const STEPS: usize = 6;
    let mut out = [(0.0f32, 0.0f32); 4];
    if vision <= 0.0 || terrain.is_empty() {
        return out;
    }
    let span = (half_fov * 2.0).max(1e-3);
    for sector in 0..4 {
        let mut max_cost = 0.0f32;
        let mut max_haz = 0.0f32;
        for ray in 0..RAYS_PER_SECTOR {
            let t = (sector as f32 + (ray as f32 + 0.5) / RAYS_PER_SECTOR as f32) / 4.0;
            let rel = t * span - half_fov;
            let theta = angle + rel;
            let dx = theta.cos();
            let dy = theta.sin();
            for step in 1..=STEPS {
                let d = vision * (step as f32) / (STEPS as f32);
                let p = (pos.0 + dx * d, pos.1 + dy * d);
                let tile = terrain_index
                    .get(&terrain_coord(p))
                    .and_then(|&i| terrain.get(i));
                if let Some(tile) = tile {
                    let eff = terrain_effects_for(genome, tile);
                    if eff.energy_cost > max_cost {
                        max_cost = eff.energy_cost;
                    }
                    if eff.hazard > max_haz {
                        max_haz = eff.hazard;
                    }
                }
            }
        }
        let cost_n = ((max_cost - 1.0) / 1.5).clamp(0.0, 1.0);
        let haz_n = (max_haz / 0.05).clamp(0.0, 1.0);
        out[sector] = (cost_n, haz_n);
    }
    out
}

fn random_range_ordered(rng: &mut impl Rng, a: f32, b: f32) -> f32 {
    if a <= b {
        rng.random_range(a..=b)
    } else {
        rng.random_range(b..=a)
    }
}

impl Genome {
    fn mutated_from(&self, rng: &mut impl Rng) -> Self {
        let mut out = self.clone();
        let rate = self.mutation_rate.clamp(0.0, 0.5);
        let scale = self.mutation_scale.clamp(0.001, 1.5);

        fn jitter(value: &mut f32, rng: &mut impl Rng, rate: f32, scale: f32, min: f32, max: f32) {
            if rng.random::<f32>() < rate {
                let factor = 1.0 + rng.random_range(-scale..=scale);
                *value = (*value * factor).clamp(min, max);
            }
        }

        jitter(&mut out.body_size, rng, rate, scale, 3.0, 18.0);
        jitter(&mut out.reserve_density, rng, rate, scale, 0.8, 8.0);
        jitter(&mut out.metabolism, rng, rate, scale, 0.001, 0.05);
        jitter(&mut out.muscle_power, rng, rate, scale, 0.2, 3.0);
        jitter(&mut out.move_efficiency, rng, rate, scale, 0.2, 4.0);
        jitter(&mut out.turn_agility, rng, rate, scale, 0.2, 4.0);
        jitter(&mut out.vision_distance, rng, rate, scale, 40.0, 500.0);
        jitter(
            &mut out.fov_angle,
            rng,
            rate,
            scale,
            0.35,
            std::f32::consts::TAU,
        );
        jitter(&mut out.sensory_cost, rng, rate, scale, 0.0000001, 0.00002);
        jitter(&mut out.bite_size, rng, rate, scale, 1.0, 25.0);
        jitter(&mut out.digestion_rate, rng, rate, scale, 0.2, 5.0);
        jitter(&mut out.armor, rng, rate, scale, 0.25, 2.0);
        jitter(&mut out.cold_tolerance, rng, rate, scale, 0.25, 4.0);
        jitter(&mut out.heat_tolerance, rng, rate, scale, 0.25, 4.0);
        jitter(&mut out.water_adaptation, rng, rate, scale, 0.25, 4.0);
        jitter(
            &mut out.rough_terrain_adaptation,
            rng,
            rate,
            scale,
            0.25,
            4.0,
        );
        jitter(&mut out.mate_threshold_frac, rng, rate, scale, 0.15, 0.9);
        jitter(&mut out.offspring_energy_frac, rng, rate, scale, 0.05, 0.45);
        jitter(&mut out.mutation_rate, rng, rate, scale, 0.001, 0.25);
        jitter(&mut out.mutation_scale, rng, rate, scale, 0.02, 1.0);

        if rng.random::<f32>() < rate {
            let factor = 1.0 + rng.random_range(-scale..=scale);
            out.gestation_ticks =
                ((out.gestation_ticks as f32 * factor).round() as u16).clamp(20, 1200);
        }
        if rng.random::<f32>() < rate {
            let factor = 1.0 + rng.random_range(-scale..=scale);
            out.max_age = ((out.max_age as f32 * factor).round() as u32).clamp(2_000, 200_000);
        }
        for x in &mut out.digest_efficiency {
            jitter(x, rng, rate, scale, 0.15, 2.5);
        }
        for x in &mut out.poison_resist {
            jitter(x, rng, rate, scale, 0.15, 2.5);
        }
        out
    }

    fn crossover(a: &Self, b: &Self, rng: &mut impl Rng) -> Self {
        let mut out = if rng.random_bool(0.5) {
            a.clone()
        } else {
            b.clone()
        };
        macro_rules! pick {
            ($field:ident) => {
                if rng.random_bool(0.5) {
                    out.$field = b.$field;
                }
            };
        }
        pick!(body_size);
        pick!(reserve_density);
        pick!(metabolism);
        pick!(muscle_power);
        pick!(move_efficiency);
        pick!(turn_agility);
        pick!(vision_distance);
        pick!(fov_angle);
        pick!(sensory_cost);
        pick!(bite_size);
        pick!(digestion_rate);
        pick!(armor);
        pick!(cold_tolerance);
        pick!(heat_tolerance);
        pick!(water_adaptation);
        pick!(rough_terrain_adaptation);
        pick!(mate_threshold_frac);
        pick!(offspring_energy_frac);
        pick!(gestation_ticks);
        pick!(mutation_rate);
        pick!(mutation_scale);
        pick!(max_age);
        for i in 0..FOOD_KIND_COUNT {
            if rng.random_bool(0.5) {
                out.digest_efficiency[i] = b.digest_efficiency[i];
            }
        }
        for i in 0..POISON_KIND_COUNT {
            if rng.random_bool(0.5) {
                out.poison_resist[i] = b.poison_resist[i];
            }
        }
        out.mutated_from(rng)
    }
}

fn random_pos_in_tile(rng: &mut impl Rng, tile: &TerrainTile, _dims: (f32, f32)) -> (f32, f32) {
    let half_w = tile.size.0 * 0.5;
    let half_h = tile.size.1 * 0.5;
    (
        rng.random_range((tile.position.0 - half_w)..=(tile.position.0 + half_w)),
        rng.random_range((tile.position.1 - half_h)..=(tile.position.1 + half_h)),
    )
}

fn make_food(kind: FoodKind, position: (f32, f32), size: f32, config: &SimulationConfig) -> Food {
    let energy = size * config.food_energy_per_size * food_energy_multiplier(kind);
    let eat_ticks = config.eat_action_base_ticks
        + (size * config.food_eat_ticks_per_size * food_eat_multiplier(kind)) as u16;
    Food {
        position,
        size,
        kind,
        energy,
        eat_ticks,
    }
}

fn make_poison(kind: PoisonKind, position: (f32, f32), config: &SimulationConfig) -> Poison {
    Poison {
        position,
        kind,
        damage: config.poison_damage * poison_damage_multiplier(kind),
    }
}

fn choose_food_kind(rng: &mut impl Rng, material: TerrainMaterial) -> FoodKind {
    match material {
        TerrainMaterial::DeepWater | TerrainMaterial::Water | TerrainMaterial::ShallowWater => {
            FoodKind::Kelp
        }
        TerrainMaterial::Desert | TerrainMaterial::Sand => FoodKind::CactusFruit,
        TerrainMaterial::Forest => {
            if rng.random_bool(0.65) {
                FoodKind::Fruit
            } else {
                FoodKind::Fungus
            }
        }
        TerrainMaterial::Rainforest => {
            if rng.random_bool(0.7) {
                FoodKind::Fruit
            } else {
                FoodKind::Fungus
            }
        }
        TerrainMaterial::Marsh => {
            if rng.random_bool(0.6) {
                FoodKind::Fungus
            } else {
                FoodKind::Kelp
            }
        }
        TerrainMaterial::Tundra | TerrainMaterial::Rock | TerrainMaterial::Snow => FoodKind::Lichen,
        TerrainMaterial::Savanna => {
            if rng.random_bool(0.55) {
                FoodKind::Seeds
            } else {
                FoodKind::Berries
            }
        }
        TerrainMaterial::Grass => {
            if rng.random_bool(0.7) {
                FoodKind::Berries
            } else {
                FoodKind::Seeds
            }
        }
    }
}

fn choose_poison_kind(rng: &mut impl Rng, material: TerrainMaterial) -> PoisonKind {
    match material {
        TerrainMaterial::DeepWater
        | TerrainMaterial::Water
        | TerrainMaterial::ShallowWater
        | TerrainMaterial::Marsh => PoisonKind::BrineBloom,
        TerrainMaterial::Forest | TerrainMaterial::Rainforest => {
            if rng.random_bool(0.6) {
                PoisonKind::ToxicMushroom
            } else {
                PoisonKind::Nightshade
            }
        }
        TerrainMaterial::Desert | TerrainMaterial::Sand | TerrainMaterial::Savanna => {
            PoisonKind::ThornPatch
        }
        TerrainMaterial::Tundra | TerrainMaterial::Rock | TerrainMaterial::Snow => {
            PoisonKind::BitterLichen
        }
        TerrainMaterial::Grass => {
            if rng.random_bool(0.55) {
                PoisonKind::Nightshade
            } else {
                PoisonKind::ThornPatch
            }
        }
    }
}

fn food_energy_multiplier(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Berries => 1.0,
        FoodKind::Fruit => 1.45,
        FoodKind::Fungus => 0.8,
        FoodKind::Kelp => 0.9,
        FoodKind::CactusFruit => 1.25,
        FoodKind::Lichen => 0.55,
        FoodKind::Seeds => 0.75,
    }
}

fn food_eat_multiplier(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Fruit => 1.25,
        FoodKind::Fungus => 0.8,
        FoodKind::Kelp => 1.1,
        FoodKind::CactusFruit => 1.4,
        FoodKind::Lichen => 1.6,
        _ => 1.0,
    }
}

fn poison_damage_multiplier(kind: PoisonKind) -> f32 {
    match kind {
        PoisonKind::Nightshade => 1.0,
        PoisonKind::ToxicMushroom => 1.25,
        PoisonKind::BrineBloom => 0.85,
        PoisonKind::ThornPatch => 0.7,
        PoisonKind::BitterLichen => 0.55,
    }
}

fn food_hue(kind: FoodKind) -> f32 {
    match kind {
        FoodKind::Berries => 0.92,
        FoodKind::Fruit => 0.1,
        FoodKind::Fungus => 0.78,
        FoodKind::Kelp => 0.42,
        FoodKind::CactusFruit => 0.02,
        FoodKind::Lichen => 0.23,
        FoodKind::Seeds => 0.16,
    }
}

fn poison_hue(kind: PoisonKind) -> f32 {
    match kind {
        PoisonKind::Nightshade => 0.78,
        PoisonKind::ToxicMushroom => 0.0,
        PoisonKind::BrineBloom => 0.55,
        PoisonKind::ThornPatch => 0.08,
        PoisonKind::BitterLichen => 0.2,
    }
}

fn food_odor(kind: FoodKind, energy: f32, max_energy: f32) -> f32 {
    ((energy / max_energy.max(1.0)).clamp(0.0, 1.0) * 0.65 + food_energy_multiplier(kind) / 2.0)
        .clamp(0.0, 1.0)
}

fn poison_odor(kind: PoisonKind, damage: f32, base_damage: f32) -> f32 {
    ((damage / base_damage.max(1.0)).clamp(0.0, 1.0) * 0.7 + poison_damage_multiplier(kind) / 2.5)
        .clamp(0.0, 1.0)
}

/// Generate random rectangular obstacles. Allows light overlap; rejects placements that touch world edges.
fn generate_obstacles(
    rng: &mut impl Rng,
    dims: (f32, f32),
    config: &SimulationConfig,
) -> Vec<Obstacle> {
    let mut out = Vec::with_capacity(config.num_obstacles);
    let min_s = config.min_obstacle_size.max(1.0);
    let max_s = config.max_obstacle_size.max(min_s + 1.0);
    let margin = 10.0;
    for _ in 0..config.num_obstacles {
        let w = rng.random_range(min_s..=max_s);
        let h = rng.random_range(min_s..=max_s);
        let hw = w * 0.5;
        let hh = h * 0.5;
        if dims.0 - 2.0 * (hw + margin) <= 0.0 || dims.1 - 2.0 * (hh + margin) <= 0.0 {
            continue;
        }
        let cx = rng.random_range((hw + margin)..=(dims.0 - hw - margin));
        let cy = rng.random_range((hh + margin)..=(dims.1 - hh - margin));
        out.push(Obstacle {
            position: (cx, cy),
            half_width: hw,
            half_height: hh,
        });
    }
    out
}

fn point_in_any_obstacle(pos: (f32, f32), creature_size: f32, obstacles: &[Obstacle]) -> bool {
    let half = creature_size * 0.5;
    obstacles.iter().any(|o| {
        let dx = (pos.0 - o.position.0).abs();
        let dy = (pos.1 - o.position.1).abs();
        dx < o.half_width + half && dy < o.half_height + half
    })
}

fn random_pos_in_disc(
    rng: &mut impl Rng,
    area: &SpawnArea,
    obstacles: &[Obstacle],
) -> (f32, f32) {
    let r = area.radius.max(1.0);
    for _ in 0..32 {
        let theta = rng.random_range(0.0..std::f32::consts::TAU);
        let radial = r * rng.random::<f32>().sqrt();
        let pos = (
            area.center.0 + radial * theta.cos(),
            area.center.1 + radial * theta.sin(),
        );
        if !point_in_any_obstacle(pos, CREATURE_DIM, obstacles) {
            return pos;
        }
    }
    area.center
}

fn random_pos_outside_obstacles(
    rng: &mut impl Rng,
    dims: (f32, f32),
    obstacles: &[Obstacle],
) -> (f32, f32) {
    random_pos_outside_obstacles_sized(rng, dims, obstacles, CREATURE_DIM)
}

fn random_pos_outside_obstacles_sized(
    rng: &mut impl Rng,
    dims: (f32, f32),
    obstacles: &[Obstacle],
    size: f32,
) -> (f32, f32) {
    for _ in 0..20 {
        let p = (rng.random_range(0.0..dims.0), rng.random_range(0.0..dims.1));
        if !point_in_any_obstacle(p, size, obstacles) {
            return p;
        }
    }
    (rng.random_range(0.0..dims.0), rng.random_range(0.0..dims.1))
}

impl Simulation {
    fn rebuild_generated_terrain_tiles(&mut self) {
        self.generated_terrain_tiles = self
            .terrain
            .iter()
            .map(|tile| terrain_coord(tile.position))
            .collect();
        self.rebuild_terrain_index();
    }

    // Group creatures into spatially-connected components ("islands"). Two
    // creatures land in the same cluster iff their cluster cells are
    // 8-connected. Cell size is sized to the largest creature's vision so any
    // two creatures that could possibly interact this tick stay in one cluster.
    fn cluster_creature_ids(&self) -> Vec<Vec<usize>> {
        if self.creatures.is_empty() {
            return Vec::new();
        }
        let mut max_vision = 0.0f32;
        let mut entries: Vec<(usize, (f32, f32))> = Vec::with_capacity(self.creatures.len());
        for entry in self.creatures.iter() {
            let c = entry.value();
            max_vision = max_vision.max(c.genome.vision_distance);
            entries.push((*entry.key(), c.position));
        }
        let cluster_cell = (max_vision * 2.0).max(GRID_CELL * 4.0);
        let mut buckets: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        let mut id_cell: HashMap<usize, (i32, i32)> = HashMap::with_capacity(entries.len());
        for (id, pos) in &entries {
            let cell = (
                (pos.0 / cluster_cell).floor() as i32,
                (pos.1 / cluster_cell).floor() as i32,
            );
            buckets.entry(cell).or_default().push(*id);
            id_cell.insert(*id, cell);
        }
        let mut visited: HashSet<usize> = HashSet::with_capacity(entries.len());
        let mut clusters: Vec<Vec<usize>> = Vec::new();
        for (id, _) in &entries {
            if visited.contains(id) {
                continue;
            }
            let mut cluster: Vec<usize> = Vec::new();
            let mut cell_visited: HashSet<(i32, i32)> = HashSet::new();
            let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
            let start = id_cell[id];
            queue.push_back(start);
            cell_visited.insert(start);
            while let Some(cell) = queue.pop_front() {
                if let Some(bucket) = buckets.get(&cell) {
                    for &cid in bucket {
                        if visited.insert(cid) {
                            cluster.push(cid);
                        }
                    }
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let n = (cell.0 + dx, cell.1 + dy);
                        if buckets.contains_key(&n) && cell_visited.insert(n) {
                            queue.push_back(n);
                        }
                    }
                }
            }
            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }
        clusters
    }

    fn rebuild_terrain_index(&mut self) {
        self.terrain_index.clear();
        self.terrain_index.reserve(self.terrain.len());
        if self.terrain_by_biome.len() != TERRAIN_MATERIAL_COUNT {
            self.terrain_by_biome = vec![Vec::new(); TERRAIN_MATERIAL_COUNT];
        }
        for bucket in &mut self.terrain_by_biome {
            bucket.clear();
        }
        for (i, tile) in self.terrain.iter().enumerate() {
            self.terrain_index.insert(terrain_coord(tile.position), i);
            self.terrain_by_biome[terrain_material_index(tile.material)].push(i);
        }
    }

    fn terrain_tile_at_indexed(&self, position: (f32, f32)) -> Option<&TerrainTile> {
        if self.terrain.is_empty() {
            return None;
        }
        self.terrain_index
            .get(&terrain_coord(position))
            .and_then(|&i| self.terrain.get(i))
    }

    fn ensure_terrain_around_creatures(&mut self) {
        if self.terrain_seed == 0.0 {
            self.terrain_seed = 1.0;
        }
        if self.generated_terrain_tiles.is_empty() && !self.terrain.is_empty() {
            self.rebuild_generated_terrain_tiles();
        }
        let positions: Vec<((f32, f32), f32)> = self
            .creatures
            .iter()
            .map(|entry| {
                let c = entry.value();
                (c.position, c.genome.vision_distance)
            })
            .collect();
        for (position, vision) in positions {
            let radius = (vision + TERRAIN_TILE_SIZE * 3.0).max(TERRAIN_TILE_SIZE * 4.0);
            let min = terrain_coord((position.0 - radius, position.1 - radius));
            let max = terrain_coord((position.0 + radius, position.1 + radius));
            for ty in min.1..=max.1 {
                for tx in min.0..=max.0 {
                    if self.generated_terrain_tiles.insert((tx, ty)) {
                        self.terrain.push(terrain_tile_for_coord(
                            tx,
                            ty,
                            self.terrain_seed,
                            (TERRAIN_TILE_SIZE, TERRAIN_TILE_SIZE),
                        ));
                    }
                }
            }
        }
    }

    // Despawn world (terrain/food/poison) outside the initial spawn box that
    // is also far from every creature. The initial spawn box [(0,0)..world_dim]
    // plus a generous margin is preserved unconditionally so the boundary of
    // the home world doesn't show up as a visible discontinuity. Anything past
    // that is reclaimed when no creature is nearby (using a larger radius than
    // `ensure_terrain_around_creatures` so we don't churn tiles at the edge).
    fn prune_world_far_from_creatures(&mut self) {
        let centers: Vec<((f32, f32), f32)> = self
            .creatures
            .iter()
            .map(|entry| {
                let c = entry.value();
                let r = (c.genome.vision_distance * 3.0 + TERRAIN_TILE_SIZE * 16.0).max(2048.0);
                (c.position, r)
            })
            .collect();
        if centers.is_empty() {
            return;
        }
        let world_w = self.world_dim.0;
        let world_h = self.world_dim.1;
        // Hide the initial-world discontinuity behind an 8-tile buffer.
        let margin = TERRAIN_TILE_SIZE * 8.0;
        let near_any = |position: (f32, f32)| {
            centers.iter().any(|(center, radius)| {
                let dx = position.0 - center.0;
                let dy = position.1 - center.1;
                dx * dx + dy * dy <= radius * radius
            })
        };
        let inside_initial = |position: (f32, f32)| {
            position.0 >= -margin
                && position.1 >= -margin
                && position.0 <= world_w + margin
                && position.1 <= world_h + margin
        };
        let keep = |position: (f32, f32)| inside_initial(position) || near_any(position);
        let before = self.terrain.len();
        self.terrain.retain(|tile| keep(tile.position));
        if self.terrain.len() != before {
            self.rebuild_generated_terrain_tiles();
        }
        self.food.retain(|food| keep(food.position));
        self.poison.retain(|poison| keep(poison.position));
    }

    fn terrain_effects_at_for(&self, position: (f32, f32), genome: &Genome) -> TerrainEffects {
        self.terrain_tile_at_indexed(position)
            .map(|tile| terrain_effects_for(genome, tile))
            .unwrap_or(TerrainEffects {
                elevation: 0.5,
                speed: 1.0,
                energy_cost: 1.0,
                hazard: 0.0,
            })
    }

    fn sensor_bounds(&self) -> ((f32, f32), (f32, f32)) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut include = |x: f32, y: f32, radius: f32| {
            min_x = min_x.min(x - radius);
            min_y = min_y.min(y - radius);
            max_x = max_x.max(x + radius);
            max_y = max_y.max(y + radius);
        };
        for entry in self.creatures.iter() {
            let c = entry.value();
            include(
                c.position.0,
                c.position.1,
                c.genome.vision_distance + GRID_CELL * 2.0,
            );
        }
        for f in &self.food {
            include(f.position.0, f.position.1, f.size + GRID_CELL);
        }
        for p in &self.poison {
            include(p.position.0, p.position.1, GRID_CELL);
        }
        if !min_x.is_finite() {
            return ((0.0, 0.0), self.world_dim);
        }
        let width = (max_x - min_x).max(GRID_CELL);
        let height = (max_y - min_y).max(GRID_CELL);
        ((min_x, min_y), (width, height))
    }

    fn run(&mut self, gpu: &mut GpuBrainCompute) {
        let start_all = Instant::now();
        self.food_eaten_this_tick = 0;
        self.births_this_tick = 0;
        self.deaths_this_tick = 0;
        // Terrain only grows on a fixed cadence. Cheap when steady-state but
        // also force a run on first-tick / empty terrain so the world boots.
        let needs_terrain_expand =
            self.terrain.is_empty() || self.ticks % TERRAIN_EXPAND_INTERVAL == 0;
        if needs_terrain_expand {
            self.ensure_terrain_around_creatures();
            self.prune_world_far_from_creatures();
            // Refresh the spatial index whenever terrain may have changed.
            self.rebuild_terrain_index();
        }

        let (sensor_origin, dims) = self.sensor_bounds();

        // Drain previous tick's GPU work first. Pipelined readback: brain
        // outputs computed from tick N-1's state are applied to creatures here
        // at the start of tick N, so the GPU dispatch we submit below runs
        // concurrently with this tick's CPU update. Adds 1-tick reaction lag.
        let prev_outputs = gpu.take_pending_outputs();
        let prev_outputs_by_id: HashMap<usize, [f32; GPU_OUTPUTS]> = match prev_outputs {
            Some((ids, outs)) => ids.into_iter().zip(outs.into_iter()).collect(),
            None => HashMap::new(),
        };

        // Ordered creature sensor rows are the GPU dispatch order. Terrain is
        // packed once here and reused below to avoid a second
        // `terrain_effects_at` lookup during the parallel update.
        let t_start_inputs = Instant::now();
        let creature_rows: Vec<(usize, GpuCreatureSensor, (f32, f32), TerrainEffects)> = self
            .creatures
            .par_iter()
            .map(|entry| {
                let id = *entry.key();
                let c = entry.value();
                let output_layer_idx = c.brain.output_layer as usize;
                let output_layer = &c.brain.graph.layers[output_layer_idx];
                let terrain = self.terrain_effects_at_for(c.position, &c.genome);
                let sectors = sample_terrain_sectors(
                    &self.terrain,
                    &self.terrain_index,
                    dims,
                    c.position,
                    c.angle,
                    c.genome.fov_angle * 0.5,
                    c.genome.vision_distance,
                    &c.genome,
                );
                let prev_thrust = output_layer
                    .get(0)
                    .map(|n| get_output_value(&n.value))
                    .unwrap_or_default();
                let prev_turn_left = output_layer
                    .get(1)
                    .map(|n| get_output_value(&n.value))
                    .unwrap_or_default();
                let prev_turn_right = output_layer
                    .get(2)
                    .map(|n| get_output_value(&n.value))
                    .unwrap_or_default();
                (
                    id,
                    GpuCreatureSensor {
                        pos_x: c.position.0,
                        pos_y: c.position.1,
                        angle: c.angle,
                        energy: c.energy,
                        prev_energy: c.prev_energy,
                        touched: c.touched,
                        mem: c.mem,
                        age: (c.age + 1) as f32,
                        max_energy: max_energy_for(&c.genome),
                        body_size: c.genome.body_size,
                        max_age: c.genome.max_age as f32,
                        half_fov: c.genome.fov_angle * 0.5,
                        vision: c.genome.vision_distance,
                        prev_thrust,
                        prev_turn_left,
                        prev_turn_right,
                        terrain_elevation: terrain.elevation,
                        terrain_speed: terrain.speed,
                        terrain_energy_cost: terrain.energy_cost,
                        terrain_hazard: terrain.hazard,
                        terrain_cost_s0: sectors[0].0,
                        terrain_cost_s1: sectors[1].0,
                        terrain_cost_s2: sectors[2].0,
                        terrain_cost_s3: sectors[3].0,
                        terrain_haz_s0: sectors[0].1,
                        terrain_haz_s1: sectors[1].1,
                        terrain_haz_s2: sectors[2].1,
                        terrain_haz_s3: sectors[3].1,
                        signal: c.signal,
                        _pad0: 0.0,
                        _pad1: 0.0,
                        _pad2: 0.0,
                    },
                    c.position,
                    terrain,
                )
            })
            .collect();
        let creature_ids: Vec<usize> = creature_rows.iter().map(|(id, _, _, _)| *id).collect();
        let creature_sensors: Vec<GpuCreatureSensor> = creature_rows
            .iter()
            .map(|(_, sensor, _, _)| *sensor)
            .collect();
        let max_creature_vision = creature_sensors
            .iter()
            .map(|c| c.vision)
            .fold(1.0f32, f32::max);
        let max_creature_energy = creature_sensors
            .iter()
            .map(|c| c.max_energy)
            .fold(1.0f32, f32::max);
        let creature_positions: Vec<Vec2> = creature_rows
            .iter()
            .map(|(_, _, p, _)| Vec2::new(p.0, p.1))
            .collect();
        // id -> creature_grid index (also used to look up the cached terrain).
        let mut creature_idx_by_id: HashMap<usize, usize> =
            HashMap::with_capacity(creature_ids.len());
        for (i, id) in creature_ids.iter().enumerate() {
            creature_idx_by_id.insert(*id, i);
        }
        let creature_terrains: Vec<TerrainEffects> =
            creature_rows.iter().map(|(_, _, _, t)| *t).collect();
        let food_sensors: Vec<GpuFoodSensor> = self
            .food
            .par_iter()
            .map(|f| GpuFoodSensor {
                x: f.position.0,
                y: f.position.1,
                size: f.size,
                energy: f.energy,
                hue: food_hue(f.kind),
                odor: food_odor(f.kind, f.energy, self.config.max_energy),
                _pad0: 0.0,
                _pad1: 0.0,
            })
            .collect();
        let poison_sensors: Vec<GpuPoisonSensor> = self
            .poison
            .par_iter()
            .map(|p| GpuPoisonSensor {
                x: p.position.0,
                y: p.position.1,
                damage: p.damage,
                hue: poison_hue(p.kind),
                odor: poison_odor(p.kind, p.damage, self.config.poison_damage),
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            })
            .collect();
        let obstacle_sensors: Vec<GpuObstacleSensor> = self
            .obstacles
            .par_iter()
            .map(|o| GpuObstacleSensor {
                x: o.position.0,
                y: o.position.1,
                half_w: o.half_width,
                half_h: o.half_height,
            })
            .collect();
        let t_inputs = t_start_inputs.elapsed();

        // CPU grids still serve collision/eating/attack resolution until those passes move to GPU.
        let food_positions: Vec<Vec2> = self
            .food
            .iter()
            .map(|f| Vec2::new(f.position.0, f.position.1))
            .collect();
        let poison_positions: Vec<Vec2> = self
            .poison
            .iter()
            .map(|p| Vec2::new(p.position.0, p.position.1))
            .collect();
        let food_grid = Grid::build(food_positions, GRID_CELL, dims);
        let poison_grid = Grid::build(poison_positions, GRID_CELL, dims);
        let creature_grid = Grid::build(creature_positions, GRID_CELL, dims);
        // Snapshot of last-tick voice per creature, aligned with creature_grid indices.
        // Used by the listener-bonus pass below.
        let creature_signals_snapshot: Vec<f32> =
            creature_sensors.iter().map(|s| s.signal).collect();

        let n = creature_ids.len();

        // Submit current tick's sensor + brain compute. Non-blocking — runs in
        // parallel with the CPU update below. We will drain it at the start of
        // the next tick.
        let t_start_gpu = Instant::now();
        if n > 0 {
            gpu.enqueue_with_sensors(
                &creature_ids,
                GpuSensorWorld {
                    creatures: &creature_sensors,
                    foods: &food_sensors,
                    poisons: &poison_sensors,
                    obstacles: &obstacle_sensors,
                    origin: sensor_origin,
                    dims,
                    cell: GRID_CELL,
                    half_fov: std::f32::consts::PI,
                    vision: max_creature_vision,
                    max_energy: max_creature_energy,
                    min_creature_size: 3.0,
                    max_creature_size: 18.0,
                    max_age: 1.0,
                },
            );
        }
        let t_gpu = t_start_gpu.elapsed();

        let t_start_update = Instant::now();
        let bitten_food_indices = DashSet::<usize>::new();
        let food_bites: DashMap<usize, f32> = DashMap::new();
        let food_eaten_count = AtomicUsize::new(0);
        let hit_poison = DashSet::<usize>::new();
        let attack_intents: DashMap<usize, (usize, f32)> = DashMap::new(); // attacker_id -> (target_id, damage)

        // Partition creatures into spatially-disjoint islands so each cluster's
        // updates are independent. Per-cluster sequential mutation also gives
        // good cache locality and avoids cross-island contention on the same
        // DashMap shards. Inter-island writes only happen through the shared
        // DashSet/DashMap accumulators (food_bites/hit_poison/attack_intents),
        // which are intrinsically concurrent.
        let clusters = self.cluster_creature_ids();
        clusters.par_iter().for_each(|cluster| {
        for &id in cluster {
            let Some(mut accessor) = self.creatures.get_mut(&id) else {
                continue;
            };
            let c = accessor.value_mut();
            c.mate_cooldown = c.mate_cooldown.saturating_sub(1);
            c.action_lock = c.action_lock.saturating_sub(1);
            c.age += 1;
            c.prev_energy = c.energy;
            c.touched = 0.0;

            // Apply GPU partials (sigmoid + save) from PREVIOUS tick's compute
            // (pipelined readback). Creatures with no prior compute (just spawned)
            // skip this and use whatever is already on their output layer.
            if let Some(partials) = prev_outputs_by_id.get(&id) {
                let output_layer_idx = c.brain.output_layer as usize;
                for (idx, partial) in partials.iter().copied().enumerate() {
                    if let Some(node) = c.brain.graph.layers[output_layer_idx].get_mut(idx) {
                        if let Node::Output(output) = &mut node.value {
                            output.finish_and_save(partial);
                        }
                    }
                }
            }

            let output_layer_idx = c.brain.output_layer as usize;
            let output_layer = &c.brain.graph.layers[output_layer_idx];
            // Sigmoid(0) = 0.5, so an unwired output node yields 0.5 by default.
            // Rescale "drive" outputs through (x - 0.5) * 2 clamped to [0, 1] so a
            // creature with no wiring stays still instead of drifting at half thrust.
            let drive = |raw: f32| ((raw - 0.5) * 2.0).clamp(0.0, 1.0);
            let thrust_raw = output_layer
                .get(0)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let thrust = drive(thrust_raw);
            let turn_left = output_layer
                .get(1)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let turn_right = output_layer
                .get(2)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let speed_out_raw = output_layer
                .get(3)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let speed_out = drive(speed_out_raw);
            let mate = output_layer
                .get(4)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let eat = output_layer
                .get(5)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let attack = output_layer
                .get(6)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let mem_out = output_layer
                .get(7)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let speak = output_layer
                .get(8)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let voice = output_layer
                .get(9)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();

            // Memory feedback: rescale [0,1] sigmoid to [-1,1] for richer signal next tick.
            c.mem = mem_out * 2.0 - 1.0;
            // Voice is an action: speak > 0.5 emits a pulse, content = voice [-1,1].
            // Costs energy proportional to loudness so silence is the default and
            // selection has to favor signaling for it to spread.
            if speak > 0.5 && c.energy > self.config.signal_action_cost {
                let content = voice * 2.0 - 1.0;
                c.signal = content;
                c.energy -= self.config.signal_action_cost * content.abs().max(0.1);
            } else {
                c.signal = 0.0;
            }

            // Turn + thrust movement.
            let mass = body_mass(&c.genome);
            let max_energy = max_energy_for(&c.genome);
            let energy_level = energy_frac(c.energy, &c.genome);
            let size_speed_scale = body_size_speed_scale(&c.genome);
            let turn =
                (turn_left - turn_right) * 0.18 * c.genome.turn_agility.max(0.1) * size_speed_scale;
            c.angle += turn;
            // Wrap angle.
            if c.angle > std::f32::consts::PI {
                c.angle -= std::f32::consts::TAU;
            }
            if c.angle < -std::f32::consts::PI {
                c.angle += std::f32::consts::TAU;
            }

            // Reuse terrain captured during sensor pack (computed pre-move,
            // matches the pre-move position the GPU brain saw). Falls back to
            // a fresh lookup for newly-spawned creatures absent from the cache.
            let terrain = creature_idx_by_id
                .get(&id)
                .and_then(|i| creature_terrains.get(*i).copied())
                .unwrap_or_else(|| self.terrain_effects_at_for(c.position, &c.genome));
            let speed_cap = energy_level;
            let c_size_now = creature_size(&c.genome);
            let speed_factor =
                BASE_MOVE_SPEED * terrain.speed * c.genome.muscle_power.max(0.1) * size_speed_scale;
            let cost_factor = terrain.energy_cost / c.genome.move_efficiency.max(0.1);
            let desired = if c.action_lock == 0 {
                thrust * speed_out * speed_cap
            } else {
                0.0
            };
            let move_amount = desired * speed_factor;

            let adaptation_load = (c.genome.cold_tolerance
                + c.genome.heat_tolerance
                + c.genome.water_adaptation
                + c.genome.rough_terrain_adaptation
                - 4.0)
                .max(0.0)
                * 0.001;
            let sensory_load =
                c.genome.vision_distance * c.genome.fov_angle * c.genome.sensory_cost;
            c.energy -= c.genome.metabolism * mass.sqrt()
                + adaptation_load
                + sensory_load
                + (move_amount.abs() * mass * 0.002 * cost_factor)
                + (turn.abs() * mass * 0.001 / c.genome.turn_agility.max(0.1))
                + (mate * c.genome.metabolism * mass.sqrt() * 0.2)
                + (eat * c.genome.metabolism * mass.sqrt() * 0.1)
                + (attack * 0.03 * mass.sqrt() * c.genome.muscle_power.max(0.1))
                + terrain.hazard;

            let dx = c.angle.cos() * move_amount;
            let dy = c.angle.sin() * move_amount;
            let new_x = c.position.0 + dx;
            if !point_in_any_obstacle((new_x, c.position.1), c_size_now, &self.obstacles) {
                c.position.0 = new_x;
            } else {
                c.touched = 1.0;
            }
            let new_y = c.position.1 + dy;
            if !point_in_any_obstacle((c.position.0, new_y), c_size_now, &self.obstacles) {
                c.position.1 = new_y;
            } else {
                c.touched = 1.0;
            }

            let pos = Vec2::new(c.position.0, c.position.1);
            if energy_level < 0.2 {
                c.energy -= STARVATION_STRESS_COST * (0.2 - energy_level) / 0.2;
            }

            // Poison hits: grid AABB query around creature.
            let poison_half = c_size_now * 0.5;
            let pmin = pos - Vec2::splat(poison_half + GRID_CELL);
            let pmax = pos + Vec2::splat(poison_half + GRID_CELL);
            poison_grid.query_aabb(pmin, pmax, |p_idx| {
                let p = &self.poison[p_idx as usize];
                if do_squares_collide(c.position, p.position) {
                    let resist = c.genome.poison_resist[poison_kind_index(p.kind)].max(0.1);
                    c.energy -= p.damage / resist;
                    hit_poison.insert(p_idx as usize);
                }
            });

            // Eat food: grid AABB query.
            if c.action_lock == 0 && eat > 0.5 {
                let c_size = creature_size(&c.genome);
                let half = (c_size + self.config.max_food_size) * 0.5;
                let fmin = pos - Vec2::splat(half + GRID_CELL);
                let fmax = pos + Vec2::splat(half + GRID_CELL);
                let mut ate = false;
                food_grid.query_aabb(fmin, fmax, |f_idx| {
                    if ate {
                        return;
                    }
                    let f_idx_us = f_idx as usize;
                    if bitten_food_indices.contains(&f_idx_us) {
                        return;
                    }
                    let f = &self.food[f_idx_us];
                    if do_sized_squares_collide(c.position, c_size, f.position, f.size)
                        && bitten_food_indices.insert(f_idx_us)
                    {
                        let bite_size = c.genome.bite_size.max(0.1).min(f.size);
                        let bite_frac = if f.size > 0.0 {
                            bite_size / f.size
                        } else {
                            0.0
                        };
                        let digestion = c.genome.digestion_rate.max(0.1);
                        c.energy -= mass.sqrt() * 0.01;
                        c.action_lock =
                            ((f.eat_ticks as f32 * bite_frac / digestion).round() as u16).max(1);
                        if bite_size >= f.size {
                            food_eaten_count.fetch_add(1, Ordering::Relaxed);
                        }
                        food_bites.insert(f_idx_us, bite_size);
                        let efficiency =
                            c.genome.digest_efficiency[food_kind_index(f.kind)].max(0.0);
                        c.energy = (c.energy + f.energy * bite_frac * efficiency).min(max_energy);
                        ate = true;
                    }
                });
            }

            // Attack: pick nearest creature within attack_range in front (FOV).
            if c.action_lock == 0 && attack > 0.5 {
                let r = self.config.attack_range;
                let r2 = r * r;
                let amin = pos - Vec2::splat(r);
                let amax = pos + Vec2::splat(r);
                let mut best: Option<(usize, f32)> = None;
                let cf = c.angle.cos();
                let sf = c.angle.sin();
                creature_grid.query_aabb(amin, amax, |idx| {
                    let other_id = creature_ids[idx as usize];
                    if other_id == id {
                        return;
                    }
                    let tp = creature_grid.position(idx);
                    let ddx = tp.x - pos.x;
                    let ddy = tp.y - pos.y;
                    let d2 = ddx * ddx + ddy * ddy;
                    if d2 > r2 || d2 < 1e-6 {
                        return;
                    }
                    // Front half check: dot with facing > 0
                    if ddx * cf + ddy * sf <= 0.0 {
                        return;
                    }
                    match best {
                        Some((_, bd)) if bd <= d2 => {}
                        _ => best = Some((other_id, d2)),
                    }
                });
                if let Some((target_id, _)) = best {
                    let damage = 4.0 * c.genome.muscle_power.max(0.1) * mass.sqrt();
                    attack_intents.insert(id, (target_id, damage));
                }
            }

            // Listener bonus: reward energy for being near speakers. Linear falloff
            // over vision range. Drives voice to spread (speakers help neighbors,
            // neighbors gain fitness; pairs that congregate around chatter outcompete
            // mute loners).
            if self.config.signal_listener_bonus > 0.0 {
                let r = c.genome.vision_distance.max(1.0);
                let inv_r = 1.0 / r;
                let r2 = r * r;
                let lmin = pos - Vec2::splat(r);
                let lmax = pos + Vec2::splat(r);
                let mut sum = 0.0f32;
                creature_grid.query_aabb(lmin, lmax, |cidx| {
                    let i = cidx as usize;
                    if creature_ids[i] == id {
                        return;
                    }
                    let tp = creature_grid.position(cidx);
                    let dx = tp.x - pos.x;
                    let dy = tp.y - pos.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 > r2 {
                        return;
                    }
                    let d = d2.sqrt();
                    let mag = creature_signals_snapshot[i].abs();
                    sum += mag * (1.0 - d * inv_r).max(0.0);
                });
                if sum > 0.0 {
                    let cap = max_energy_for(&c.genome);
                    c.energy = (c.energy + sum * self.config.signal_listener_bonus).min(cap);
                }
            }
        }
        });

        // Apply attacks: deduct from target, transfer ratio to attacker.
        for entry in attack_intents.iter() {
            let attacker_id = *entry.key();
            let (target_id, dmg) = *entry.value();
            let actual_damage = if let Some(mut t) = self.creatures.get_mut(&target_id) {
                let before = t.energy;
                t.energy -= dmg / t.genome.armor.max(0.1);
                (before - t.energy).max(0.0)
            } else {
                0.0
            };
            if actual_damage > 0.0 {
                if let Some(mut a) = self.creatures.get_mut(&attacker_id) {
                    let cap = max_energy_for(&a.genome);
                    a.energy = (a.energy + actual_damage * 0.35).min(cap);
                }
            }
        }

        let food_eaten_this_tick_val = food_eaten_count.load(Ordering::Relaxed);
        self.total_food_eaten += food_eaten_this_tick_val;
        self.food_eaten_this_tick += food_eaten_this_tick_val;
        let t_update = t_start_update.elapsed();

        let t_start_mate = Instant::now();
        self.mate_creatures(gpu);
        let t_mate = t_start_mate.elapsed();

        // Collect dead ids before retain so we can release GPU slots.
        let mut dead_ids: Vec<usize> = Vec::new();
        self.creatures.retain(|id, creature| {
            let alive = creature.energy > 0.0
                && (self.config.ignore_max_age || creature.age <= creature.genome.max_age);
            if !alive {
                dead_ids.push(*id);
            }
            alive
        });
        self.deaths_this_tick += dead_ids.len();
        self.total_deaths += dead_ids.len();
        for id in dead_ids {
            gpu.release(id);
        }

        // Extinction restart: when population drops below threshold, cross-breed survivors into a full new generation.
        let extinction_count =
            ((self.target_population as f32 * self.config.extinction_threshold).max(1.0)) as usize;
        if self.config.extinction_restart_enabled
            && !self.creatures.is_empty()
            && self.creatures.len() < extinction_count
        {
            self.restart_generation(gpu);
        }

        for bite in food_bites.iter() {
            if let Some(food) = self.food.get_mut(*bite.key()) {
                let bite_size = (*bite.value()).min(food.size);
                let bite_frac = if food.size > 0.0 {
                    bite_size / food.size
                } else {
                    1.0
                };
                food.size -= bite_size;
                food.energy *= 1.0 - bite_frac;
                food.eat_ticks =
                    ((food.eat_ticks as f32 * (1.0 - bite_frac)).round() as u16).max(1);
            }
        }
        self.food
            .retain(|food| food.size > 0.01 && food.energy > 0.01);
        let mut i = 0;
        self.poison.retain(|_| {
            let alive = !hit_poison.contains(&i);
            i += 1;
            alive
        });

        let mut rng = rand::rng();
        if self.ticks % RESOURCE_REFILL_INTERVAL == 0 {
            self.refill_food_and_poison(&mut rng);
        }
        if self.config.creature_spawning_enabled {
            while self.creatures.len() < self.target_population {
                self.last_id += 1;
                let creature = self.random_creature(&mut rng, self.world_dim);
                gpu.assign_brain(self.last_id, &creature.brain)
                    .expect("random spawn brain must compile for GPU");
                self.creatures.insert(self.last_id, creature);
                self.total_spawned += 1;
                self.births_this_tick += 1;
                self.total_births += 1;
            }
        }

        // Roll rate windows.
        push_window(&mut self.recent_eats, food_eaten_this_tick_val);
        push_window(&mut self.recent_births, self.births_this_tick);
        push_window(&mut self.recent_deaths, self.deaths_this_tick);

        let t_total = start_all.elapsed();
        if self.ticks % 1000 == 0 {
            info!(
                "Tick {}: total={:?}, mate={:?}, inputs={:?}, gpu={:?}, update={:?} [creatures={}]",
                self.ticks,
                t_total,
                t_mate,
                t_inputs,
                t_gpu,
                t_update,
                self.creatures.len()
            );
        }
    }

    fn mate_creatures(&mut self, gpu: &mut GpuBrainCompute) {
        // Hard cap on births — once we're 25% over target, no new mating.
        let cap = self.target_population + self.target_population / 4;
        if self.creatures.len() >= cap {
            return;
        }
        let candidates: Vec<(usize, (f32, f32))> = self
            .creatures
            .par_iter()
            .filter_map(|c| {
                let creature = c.value();
                let voice_relax = if creature.signal != 0.0 {
                    (1.0 - self.config.mate_voice_bonus.clamp(0.0, 0.95) * 0.5).max(0.05)
                } else {
                    1.0
                };
                let threshold = max_energy_for(&creature.genome)
                    * creature.genome.mate_threshold_frac
                    * voice_relax;
                if creature.action_lock != 0
                    || creature.mate_cooldown != 0
                    || creature.energy < threshold
                {
                    return None;
                }
                let output_layer =
                    &creature.brain.graph.layers[creature.brain.output_layer as usize];
                // > 0.5 matches eat/attack convention. Using mate_threshold_frac
                // (0.45) caused unwired brains (sigmoid default 0.5) to be flagged
                // as actively trying to mate every tick.
                let is_mating = output_layer
                    .get(4)
                    .map(|n| get_output_value(&n.value) > 0.5)
                    .unwrap_or(false);
                if !is_mating {
                    return None;
                }
                Some((*c.key(), creature.position))
            })
            .collect();

        if candidates.is_empty() {
            return;
        }

        // Spatial grid over candidates so we don't do O(n²) collision search.
        let cand_positions: Vec<Vec2> = candidates
            .iter()
            .map(|(_, p)| Vec2::new(p.0, p.1))
            .collect();
        let cand_grid = Grid::build(cand_positions, GRID_CELL, self.world_dim);

        let mut used_parents = HashSet::new();
        let mut parent_pairs = Vec::new();
        for (idx, (a_id, a_position)) in candidates.iter().enumerate() {
            if used_parents.contains(a_id) {
                continue;
            }
            let a_pos = Vec2::new(a_position.0, a_position.1);
            let half = CREATURE_DIM_HALF + GRID_CELL;
            let amin = a_pos - Vec2::splat(half);
            let amax = a_pos + Vec2::splat(half);

            let mut chosen: Option<(usize, (f32, f32))> = None;
            cand_grid.query_aabb(amin, amax, |b_grid_idx| {
                if chosen.is_some() {
                    return;
                }
                let b_idx = b_grid_idx as usize;
                if b_idx <= idx {
                    return;
                }
                let (b_id, b_position) = candidates[b_idx];
                if used_parents.contains(&b_id) {
                    return;
                }
                if do_squares_collide(*a_position, b_position) {
                    chosen = Some((b_id, b_position));
                }
            });

            let Some((b_id, b_position)) = chosen else {
                continue;
            };

            let pair_data = {
                let Some(a) = self.creatures.get(a_id) else {
                    continue;
                };
                let Some(b) = self.creatures.get(&b_id) else {
                    continue;
                };
                let voice_relax = if a.signal != 0.0 && b.signal != 0.0 {
                    (1.0 - self.config.mate_voice_bonus.clamp(0.0, 0.95)).max(0.05)
                } else {
                    1.0
                };
                let a_threshold =
                    max_energy_for(&a.genome) * a.genome.mate_threshold_frac * voice_relax;
                let b_threshold =
                    max_energy_for(&b.genome) * b.genome.mate_threshold_frac * voice_relax;
                if a.energy < a_threshold || b.energy < b_threshold {
                    continue;
                }
                if a.action_lock != 0 || b.action_lock != 0 {
                    continue;
                }
                Some((
                    a.brain.clone(),
                    b.brain.clone(),
                    a.genome.clone(),
                    b.genome.clone(),
                    (
                        ((a_position.0 + b_position.0) / 2.0).clamp(0.0, self.world_dim.0),
                        ((a_position.1 + b_position.1) / 2.0).clamp(0.0, self.world_dim.1),
                    ),
                ))
            };

            if let Some((a_brain, b_brain, a_genome, b_genome, child_pos)) = pair_data {
                parent_pairs.push((*a_id, b_id, a_brain, b_brain, a_genome, b_genome, child_pos));
                used_parents.insert(*a_id);
                used_parents.insert(b_id);
                if parent_pairs.len() >= self.config.max_births_per_tick {
                    break;
                }
            }
        }

        if parent_pairs.is_empty() {
            return;
        }
        let input_nodes = self.input_nodes.clone();
        let output_nodes = self.output_nodes.clone();
        let births: Vec<_> = parent_pairs
            .into_par_iter()
            .map(
                |(a_id, b_id, a_brain, b_brain, a_genome, b_genome, child_pos)| {
                    let mut rng = rand::rng();
                    let mut child_brain = crossover_brains(&a_brain, &b_brain, &mut rng);
                    ensure_brain_io(&mut child_brain, &input_nodes, &output_nodes);
                    let child_genome = Genome::crossover(&a_genome, &b_genome, &mut rng);
                    Self::mutate_child_brain(
                        &mut child_brain,
                        child_genome.mutation_rate,
                        child_genome.mutation_scale,
                    );
                    (a_id, b_id, child_brain, child_genome, child_pos)
                },
            )
            .collect();

        for (a_id, b_id, child_brain, child_genome, child_pos) in births {
            let mut child_energy = 0.0;
            if let Some(mut a) = self.creatures.get_mut(&a_id) {
                let transfer = (a.energy * a.genome.offspring_energy_frac).min(a.energy * 0.45);
                a.energy -= transfer;
                child_energy += transfer;
                a.mate_cooldown = a.genome.gestation_ticks;
                a.action_lock = (a.genome.gestation_ticks / 8).max(1);
                a.times_mated = a.times_mated.saturating_add(1);
            }
            if let Some(mut b) = self.creatures.get_mut(&b_id) {
                let transfer = (b.energy * b.genome.offspring_energy_frac).min(b.energy * 0.45);
                b.energy -= transfer;
                child_energy += transfer;
                b.mate_cooldown = b.genome.gestation_ticks;
                b.action_lock = (b.genome.gestation_ticks / 8).max(1);
                b.times_mated = b.times_mated.saturating_add(1);
            }
            child_energy = child_energy.min(max_energy_for(&child_genome));
            self.last_id += 1;
            self.total_spawned += 1;
            self.births_this_tick += 1;
            self.total_births += 1;
            gpu.assign_brain(self.last_id, &child_brain)
                .expect("child brain must compile for GPU");
            self.creatures.insert(
                self.last_id,
                Creature {
                    brain: child_brain,
                    genome: child_genome,
                    position: child_pos,
                    angle: 0.0,
                    energy: child_energy,
                    age: 0,
                    mate_cooldown: 0,
                    action_lock: 0,
                    mem: 0.0,
                    prev_energy: child_energy,
                    touched: 0.0,
                    signal: 0.0,
                    times_mated: 0,
                },
            );
        }
    }

    fn restart_generation(&mut self, gpu: &mut GpuBrainCompute) {
        let mut rng = rand::rng();

        // Snapshot every survivor with its fitness signal. Score combines
        // current energy with age so longer-lived, higher-energy creatures
        // dominate the gene pool.
        struct Survivor {
            brain: Net,
            genome: Genome,
            score: f32,
            mated: bool,
        }
        let mut all_survivors: Vec<Survivor> = self
            .creatures
            .iter()
            .map(|c| {
                let v = c.value();
                let max_e = max_energy_for(&v.genome).max(1.0);
                // Normalize so energy and age contribute on similar scales.
                let energy_n = (v.energy / max_e).clamp(0.0, 1.0);
                let age_n = if v.genome.max_age > 0 {
                    (v.age as f32 / v.genome.max_age as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                Survivor {
                    brain: v.brain.clone(),
                    genome: v.genome.clone(),
                    score: energy_n + age_n,
                    mated: v.times_mated > 0,
                }
            })
            .collect();
        if all_survivors.is_empty() {
            return;
        }

        // Optional gate: only those that mated at least once. If filter would
        // empty the pool, fall back to using every survivor so the sim doesn't
        // deadlock.
        if self.config.next_gen_only_mated {
            let mated_count = all_survivors.iter().filter(|s| s.mated).count();
            if mated_count > 0 {
                all_survivors.retain(|s| s.mated);
            }
        }

        // Sort descending by score (highest energy + age first).
        all_survivors
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top fraction. Always keep at least one to seed the next gen.
        let frac = self.config.next_gen_top_fraction.clamp(0.0, 1.0);
        let keep = ((all_survivors.len() as f32 * frac).ceil() as usize).max(1);
        all_survivors.truncate(keep);

        let survivors: Vec<(Net, Genome)> = all_survivors
            .into_iter()
            .map(|s| (s.brain, s.genome))
            .collect();

        self.generation += 1;
        info!(
            "Extinction restart — generation {} (survivors: {})",
            self.generation,
            survivors.len()
        );

        let all_ids: Vec<usize> = self.creatures.iter().map(|c| *c.key()).collect();
        for id in &all_ids {
            gpu.release(*id);
        }
        self.creatures.clear();

        // Choose the spawn region:
        //  - If a movable spawn area is enabled, sample uniformly inside that disc.
        //  - Otherwise spawn anywhere within the generated terrain footprint.
        let (origin, gen_dims) = self.generated_world_bounds();
        let spawn_area = self.spawn_area.clone();
        let input_nodes = self.input_nodes.clone();
        let output_nodes = self.output_nodes.clone();

        for _ in 0..self.target_population {
            self.last_id += 1;
            let (a_brain, a_genome) = &survivors[rng.random_range(0..survivors.len())];
            let (b_brain, b_genome) = &survivors[rng.random_range(0..survivors.len())];
            let mut child_brain = crossover_brains(a_brain, b_brain, &mut rng);
            let child_genome = Genome::crossover(a_genome, b_genome, &mut rng);
            ensure_brain_io(&mut child_brain, &input_nodes, &output_nodes);
            Self::mutate_child_brain(
                &mut child_brain,
                child_genome.mutation_rate,
                child_genome.mutation_scale,
            );
            let start_energy = self.config.start_energy.min(max_energy_for(&child_genome));
            gpu.assign_brain(self.last_id, &child_brain)
                .expect("restart child brain must compile for GPU");
            let position = if spawn_area.enabled && spawn_area.radius > 0.0 {
                random_pos_in_disc(&mut rng, &spawn_area, &self.obstacles)
            } else {
                let local_pos = random_pos_outside_obstacles(&mut rng, gen_dims, &self.obstacles);
                (origin.0 + local_pos.0, origin.1 + local_pos.1)
            };
            self.creatures.insert(
                self.last_id,
                Creature {
                    brain: child_brain,
                    genome: child_genome,
                    position,
                    angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                    energy: start_energy,
                    age: 0,
                    mate_cooldown: 0,
                    action_lock: 0,
                    mem: 0.0,
                    prev_energy: start_energy,
                    touched: 0.0,
                    signal: 0.0,
                    times_mated: 0,
                },
            );
            self.total_spawned += 1;
        }

        // Recompute the world footprint for the new generation. Children only
        // spawn inside the spawn-area disc (or the generated bounds when the
        // disc is disabled), so terrain past that region serves no purpose.
        // Drop it so per-tick costs and the GPU sensor grid shrink to match.
        // The initial spawn box [0..world_dim] (plus a small buffer to hide
        // the boundary) is always preserved as the curated home world.
        let world_w = self.world_dim.0;
        let world_h = self.world_dim.1;
        let margin = TERRAIN_TILE_SIZE * 8.0;
        let inside_initial = |position: (f32, f32)| {
            position.0 >= -margin
                && position.1 >= -margin
                && position.0 <= world_w + margin
                && position.1 <= world_h + margin
        };
        let inside_spawn = |position: (f32, f32)| {
            if !spawn_area.enabled || spawn_area.radius <= 0.0 {
                return false;
            }
            let dx = position.0 - spawn_area.center.0;
            let dy = position.1 - spawn_area.center.1;
            // Tile-sized buffer so the circle edge isn't ragged.
            let r = spawn_area.radius + TERRAIN_TILE_SIZE;
            dx * dx + dy * dy <= r * r
        };
        let keep = |position: (f32, f32)| inside_initial(position) || inside_spawn(position);
        self.terrain.retain(|tile| keep(tile.position));
        self.rebuild_generated_terrain_tiles();
        self.rebuild_terrain_index();

        // Fresh generation gets fresh food and poison. Wipe and respawn from
        // biome targets so the next gen evaluates against a reset resource
        // landscape instead of inheriting stale leftovers.
        self.food.clear();
        self.poison.clear();
        self.refill_food_and_poison(&mut rng);
    }

    // Bounding box of all generated terrain tiles. Returns ((origin_x, origin_y), (width, height)).
    // Falls back to ((0,0), world_dim) when no terrain has been generated yet.
    fn generated_world_bounds(&self) -> ((f32, f32), (f32, f32)) {
        if self.terrain.is_empty() {
            return ((0.0, 0.0), self.world_dim);
        }
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for tile in &self.terrain {
            min_x = min_x.min(tile.position.0);
            min_y = min_y.min(tile.position.1);
            max_x = max_x.max(tile.position.0 + tile.size.0);
            max_y = max_y.max(tile.position.1 + tile.size.1);
        }
        let width = (max_x - min_x).max(TERRAIN_TILE_SIZE);
        let height = (max_y - min_y).max(TERRAIN_TILE_SIZE);
        ((min_x, min_y), (width, height))
    }

    fn random_creature(&self, rng: &mut impl Rng, dims: (f32, f32)) -> Creature {
        let genome = self.config.initial_genome.mutated_from(rng);
        let start_energy = self.config.start_energy.min(max_energy_for(&genome));
        Creature {
            brain: random_brain(&self.input_nodes, &self.output_nodes),
            genome,
            position: random_pos_outside_obstacles(rng, dims, &self.obstacles),
            angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            energy: start_energy,
            age: 0,
            mate_cooldown: 0,
            action_lock: 0,
            mem: 0.0,
            prev_energy: start_energy,
            touched: 0.0,
            signal: 0.0,
            times_mated: 0,
        }
    }

    fn refill_food_and_poison(&mut self, rng: &mut impl Rng) {
        let food_targets = self.biome_resource_targets(&self.config.food_biome_rates);
        let poison_targets = self.biome_resource_targets(&self.config.poison_biome_rates);
        let mut food_counts = self.food_biome_counts();
        let mut poison_counts = self.poison_biome_counts();
        for biome_idx in 0..TERRAIN_MATERIAL_COUNT {
            while food_counts[biome_idx] < food_targets[biome_idx] {
                self.food.push(self.random_food(rng, biome_idx));
                food_counts[biome_idx] += 1;
            }
            while poison_counts[biome_idx] < poison_targets[biome_idx] {
                self.poison.push(self.random_poison(rng, biome_idx));
                poison_counts[biome_idx] += 1;
            }
        }
    }

    fn biome_resource_targets(
        &self,
        rates: &[f32; TERRAIN_MATERIAL_COUNT],
    ) -> [usize; TERRAIN_MATERIAL_COUNT] {
        let mut targets = [0.0f32; TERRAIN_MATERIAL_COUNT];
        for tile in &self.terrain {
            let idx = terrain_material_index(tile.material);
            targets[idx] += rates[idx].max(0.0);
        }
        targets.map(|x| x.round() as usize)
    }

    fn food_biome_counts(&self) -> [usize; TERRAIN_MATERIAL_COUNT] {
        let mut counts = [0usize; TERRAIN_MATERIAL_COUNT];
        for food in &self.food {
            counts[self.biome_index_at(food.position)] += 1;
        }
        counts
    }

    fn poison_biome_counts(&self) -> [usize; TERRAIN_MATERIAL_COUNT] {
        let mut counts = [0usize; TERRAIN_MATERIAL_COUNT];
        for poison in &self.poison {
            counts[self.biome_index_at(poison.position)] += 1;
        }
        counts
    }

    fn biome_index_at(&self, position: (f32, f32)) -> usize {
        self.terrain_tile_at_indexed(position)
            .map(|tile| terrain_material_index(tile.material))
            .unwrap_or(terrain_material_index(TerrainMaterial::Grass))
    }

    fn random_food(&self, rng: &mut impl Rng, biome_idx: usize) -> Food {
        let size = random_range_ordered(rng, self.config.min_food_size, self.config.max_food_size);
        for _ in 0..32 {
            let tile = self.random_terrain_tile_for_biome(rng, biome_idx);
            let pos = random_pos_in_tile(rng, tile, self.world_dim);
            if !point_in_any_obstacle(pos, size, &self.obstacles) {
                let kind = choose_food_kind(rng, tile.material);
                return make_food(kind, pos, size, &self.config);
            }
        }
        let tile = self.random_terrain_tile_for_biome(rng, biome_idx);
        let pos = random_pos_in_tile(rng, tile, self.world_dim);
        make_food(
            choose_food_kind(rng, tile.material),
            pos,
            size,
            &self.config,
        )
    }

    fn random_poison(&self, rng: &mut impl Rng, biome_idx: usize) -> Poison {
        for _ in 0..32 {
            let tile = self.random_terrain_tile_for_biome(rng, biome_idx);
            let pos = random_pos_in_tile(rng, tile, self.world_dim);
            if !point_in_any_obstacle(pos, CREATURE_DIM, &self.obstacles) {
                let kind = choose_poison_kind(rng, tile.material);
                return make_poison(kind, pos, &self.config);
            }
        }
        let tile = self.random_terrain_tile_for_biome(rng, biome_idx);
        let pos = random_pos_in_tile(rng, tile, self.world_dim);
        make_poison(choose_poison_kind(rng, tile.material), pos, &self.config)
    }

    fn random_terrain_tile_for_biome(&self, rng: &mut impl Rng, biome_idx: usize) -> &TerrainTile {
        if let Some(bucket) = self.terrain_by_biome.get(biome_idx) {
            if !bucket.is_empty() {
                let i = bucket[rng.random_range(0..bucket.len())];
                if let Some(tile) = self.terrain.get(i) {
                    return tile;
                }
            }
        }
        self.terrain.first().unwrap_or(&DEFAULT_TERRAIN_TILE)
    }

    fn paint_world(&mut self, paint: WorldPaint) {
        match paint.action {
            WorldPaintAction::Terrain {
                material,
                elevation,
            } => {
                let radius2 = paint.radius * paint.radius;
                for tile in &mut self.terrain {
                    let dx = tile.position.0 - paint.position.0;
                    let dy = tile.position.1 - paint.position.1;
                    if dx * dx + dy * dy <= radius2 {
                        tile.material = material;
                        tile.elevation = elevation.clamp(0.0, 1.0);
                    }
                }
            }
            WorldPaintAction::AddObstacle => {
                let half = (paint.radius * 0.5).max(2.0);
                self.obstacles.push(Obstacle {
                    position: paint.position,
                    half_width: half,
                    half_height: half,
                });
            }
            WorldPaintAction::EraseObstacle => {
                let radius = paint.radius.max(1.0);
                self.obstacles.retain(|o| {
                    let dx = (paint.position.0 - o.position.0).abs() - o.half_width;
                    let dy = (paint.position.1 - o.position.1).abs() - o.half_height;
                    let outside = Vec2::new(dx.max(0.0), dy.max(0.0));
                    outside.length_squared() > radius * radius
                });
            }
        }
    }

    fn mutate_child_brain(net: &mut Net, mutation_rate: f32, mutation_amount: f32) {
        let mut rng = rand::rng();
        let mutation_rate = mutation_rate.clamp(0.0, 0.5);
        let mutation_amount = mutation_amount.clamp(0.001, 2.0);
        // Per-edge weight + per-neuron bias jitter, each at the same rate.
        for layer in &mut net.graph.layers {
            for node in layer {
                if let Node::Neuron(neuron) = &mut node.value {
                    if rng.random::<f32>() < mutation_rate {
                        let new_bias =
                            neuron.bias() + rng.random_range(-mutation_amount..mutation_amount);
                        neuron.set_bias(new_bias);
                    }
                }
                for edge in &mut node.connections {
                    if rng.random::<f32>() < mutation_rate {
                        edge.value.weight += rng.random_range(-mutation_amount..mutation_amount);
                    }
                }
            }
        }

        // Topology mutations: AddEdge, AddNeuron, RemoveEdge, RemoveNeuron all
        // fire with the same probability. No bias toward any of them.
        if rng.random::<f32>() < mutation_rate {
            let _ = add_random_feed_forward_edge(net, &mut rng, mutation_amount);
        }
        if rng.random::<f32>() < mutation_rate {
            let neurons: Vec<Box<dyn Neuron>> = vec![Box::new(BasicNeuron::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(0..usize::MAX),
            ))];
            let _ = AddNeuron.mutate(net, &neurons, &|_| 0);
        }
        if rng.random::<f32>() < mutation_rate {
            let _ = RemoveEdge.mutate(net);
        }
        if rng.random::<f32>() < mutation_rate {
            let _ = RemoveNeuron.mutate(net, &[], &|_| 0);
        }
    }
}

fn random_brain(input_nodes: &[Node], output_nodes: &[Node]) -> Net {
    assert_eq!(
        input_nodes.len(),
        GPU_INPUTS,
        "input nodes must equal GPU_INPUTS"
    );
    assert_eq!(
        output_nodes.len(),
        GPU_OUTPUTS,
        "output nodes must equal GPU_OUTPUTS"
    );
    Net::gen(input_nodes, output_nodes).expect("generated brain must be valid")
}

fn crossover_brains(a: &Net, b: &Net, rng: &mut impl Rng) -> Net {
    // Clone one parent's full topology. The previous Crossover reproducer
    // started from an empty input+output skeleton and refilled only the
    // intersection of parent graphs, which collapsed structure across
    // generations. Direct inheritance keeps topology intact; mutation
    // handles all variation.
    if rng.random_bool(0.5) {
        a.clone()
    } else {
        b.clone()
    }
}

fn add_random_feed_forward_edge(net: &mut Net, rng: &mut impl Rng, mutation_amount: f32) -> bool {
    let len = net.graph.layers.len();
    if len < 2 {
        return false;
    }

    for _ in 0..32 {
        let from_layer = rng.random_range(0..len - 1);
        if net.graph.layers[from_layer].is_empty() {
            continue;
        }
        let to_layer = rng.random_range(from_layer + 1..len);
        if net.graph.layers[to_layer].is_empty() {
            continue;
        }
        let from = GraphLocation::new(
            from_layer as u16,
            rng.random_range(0..net.graph.layers[from_layer].len()) as u16,
        );
        let to = GraphLocation::new(
            to_layer as u16,
            rng.random_range(0..net.graph.layers[to_layer].len()) as u16,
        );
        if net.graph.get_edge(&from, &to).is_some() {
            continue;
        }
        if net
            .graph
            .add_edge(
                from,
                to,
                Edge {
                    weight: rng.random_range(-mutation_amount..mutation_amount),
                    enabled: true,
                },
            )
            .is_ok()
        {
            return true;
        }
    }
    false
}

fn ensure_brain_io(net: &mut Net, input_nodes: &[Node], output_nodes: &[Node]) {
    net.input_layer = 0;
    net.output_layer = net.graph.layers.len().saturating_sub(1) as u16;
    if let Some(layer) = net.graph.layers.get_mut(net.input_layer as usize) {
        for (idx, node) in input_nodes.iter().enumerate() {
            if let Some(existing) = layer.get_mut(idx) {
                existing.value = node.clone();
            } else {
                layer.push(GraphNode::new(node.clone()));
            }
        }
    }
    if let Some(layer) = net.graph.layers.get_mut(net.output_layer as usize) {
        for (idx, node) in output_nodes.iter().enumerate() {
            if let Some(existing) = layer.get_mut(idx) {
                existing.value = node.clone();
            } else {
                layer.push(GraphNode::new(node.clone()));
            }
        }
    }
}


#[cfg(test)]
mod inspect_save {
    use super::*;
    use std::collections::BTreeMap;

    // `cargo test -p sim --lib inspect_save_bin -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn inspect_save_bin() {
        let path = std::env::var("SIM_SAVE")
            .unwrap_or_else(|_| "sim.bin".to_string());
        let bytes = std::fs::read(&path).expect("read sim.bin");
        let cfg = bincode::config::standard();
        let (sim, _consumed): (Simulation, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).expect("decode sim");

        eprintln!("world_dim={:?}", sim.world_dim);
        eprintln!("ticks={}, generation={}", sim.ticks, sim.generation);
        eprintln!("creatures={}", sim.creatures.len());
        eprintln!("food={}, poison={}, obstacles={}", sim.food.len(), sim.poison.len(), sim.obstacles.len());
        eprintln!("terrain tiles={}", sim.terrain.len());
        eprintln!("generated_terrain_tiles set size={}", sim.generated_terrain_tiles.len());

        // Tile bounding box.
        if !sim.terrain.is_empty() {
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            let mut sizes = BTreeMap::<(u32, u32), usize>::new();
            let mut coord_counts = BTreeMap::<(i32, i32), usize>::new();
            for tile in &sim.terrain {
                min_x = min_x.min(tile.position.0);
                min_y = min_y.min(tile.position.1);
                max_x = max_x.max(tile.position.0 + tile.size.0);
                max_y = max_y.max(tile.position.1 + tile.size.1);
                let key = (tile.size.0.to_bits(), tile.size.1.to_bits());
                *sizes.entry(key).or_default() += 1;
                let coord = (
                    (tile.position.0 / TERRAIN_TILE_SIZE).floor() as i32,
                    (tile.position.1 / TERRAIN_TILE_SIZE).floor() as i32,
                );
                *coord_counts.entry(coord).or_default() += 1;
            }
            eprintln!("terrain bbox: x=[{:.1}..{:.1}] y=[{:.1}..{:.1}]", min_x, max_x, min_y, max_y);
            for (k, n) in sizes {
                eprintln!("  size {:?} -> {} tiles", (f32::from_bits(k.0), f32::from_bits(k.1)), n);
            }
            // Duplicate coords (same coord, more than one tile) — would produce overlapping rects.
            let dup: Vec<((i32, i32), usize)> =
                coord_counts.into_iter().filter(|(_, n)| *n > 1).collect();
            eprintln!("duplicate tile coords: {}", dup.len());
            for entry in dup.iter().take(20) {
                eprintln!("  coord {:?} -> {} tiles", entry.0, entry.1);
            }

            // Holes inside the inner ring (initial spawn box [0, world_dim]):
            // tile coord present iff terrain has tile at (tx,ty) inside box.
            let world_w = sim.world_dim.0;
            let world_h = sim.world_dim.1;
            let cols = (world_w / TERRAIN_TILE_SIZE).ceil() as i32;
            let rows = (world_h / TERRAIN_TILE_SIZE).ceil() as i32;
            let mut have = std::collections::HashSet::<(i32, i32)>::new();
            for tile in &sim.terrain {
                let coord = (
                    (tile.position.0 / TERRAIN_TILE_SIZE).floor() as i32,
                    (tile.position.1 / TERRAIN_TILE_SIZE).floor() as i32,
                );
                have.insert(coord);
            }
            let mut holes = 0usize;
            let mut sample = Vec::new();
            for ty in 0..rows {
                for tx in 0..cols {
                    if !have.contains(&(tx, ty)) {
                        holes += 1;
                        if sample.len() < 20 {
                            sample.push((tx, ty));
                        }
                    }
                }
            }
            eprintln!("inner-ring holes: {} of {} cells", holes, cols * rows);
            for s in &sample {
                eprintln!("  hole at coord {:?} (sim pos ~{:?})", s, (s.0 as f32 * TERRAIN_TILE_SIZE, s.1 as f32 * TERRAIN_TILE_SIZE));
            }

            // Tiles just outside the inner ring (border ring): how many remain.
            let mut outer = 0usize;
            let mut outer_min = (i32::MAX, i32::MAX);
            let mut outer_max = (i32::MIN, i32::MIN);
            for tile in &sim.terrain {
                let coord = (
                    (tile.position.0 / TERRAIN_TILE_SIZE).floor() as i32,
                    (tile.position.1 / TERRAIN_TILE_SIZE).floor() as i32,
                );
                if coord.0 < 0 || coord.1 < 0 || coord.0 >= cols || coord.1 >= rows {
                    outer += 1;
                    outer_min.0 = outer_min.0.min(coord.0);
                    outer_min.1 = outer_min.1.min(coord.1);
                    outer_max.0 = outer_max.0.max(coord.0);
                    outer_max.1 = outer_max.1.max(coord.1);
                }
            }
            eprintln!(
                "tiles outside initial box: {} (coord range x=[{}..{}] y=[{}..{}])",
                outer, outer_min.0, outer_max.0, outer_min.1, outer_max.1
            );
        }

        // Creature position bounds.
        if !sim.creatures.is_empty() {
            let mut cmin_x = f32::INFINITY;
            let mut cmin_y = f32::INFINITY;
            let mut cmax_x = f32::NEG_INFINITY;
            let mut cmax_y = f32::NEG_INFINITY;
            for c in sim.creatures.iter() {
                let p = c.value().position;
                cmin_x = cmin_x.min(p.0);
                cmin_y = cmin_y.min(p.1);
                cmax_x = cmax_x.max(p.0);
                cmax_y = cmax_y.max(p.1);
            }
            eprintln!("creature bbox: x=[{:.1}..{:.1}] y=[{:.1}..{:.1}]", cmin_x, cmax_x, cmin_y, cmax_y);
        }
    }
}
