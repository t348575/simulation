use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicUsize, Ordering},
    thread::sleep,
    time::{Duration, Instant},
};

const RATE_WINDOW_TICKS: usize = 120;

use bevy::{math::Vec2, prelude::info};
use dashmap::{DashMap, DashSet};
use engine::nn::{Edge, GraphLocation, GraphNode, Net, NeuralGraph, Node};
use flume::{unbounded, Receiver, Sender};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::gpu::{direct_brain_weights, GpuBrainCompute, GPU_INPUTS, GPU_OUTPUTS};
use super::resources::{SimulationConfig, SimulationStats};
use super::spatial::Grid;

const GRID_CELL: f32 = 32.0;

pub const CREATURE_DIM: f32 = 5.0;
pub const CREATURE_DIM_HALF: f32 = CREATURE_DIM / 2.0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Simulation {
    world_dim: (f32, f32),
    creatures: DashMap<usize, Creature>,
    food: Vec<Food>,
    poison: Vec<(f32, f32)>,
    obstacles: Vec<Obstacle>,
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
    position: (f32, f32),
    angle: f32,
    energy: f32,
    age: u32,
    mate_cooldown: u16,
    action_lock: u16,
    mem: f32,
    prev_energy: f32,
    touched: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Obstacle {
    position: (f32, f32),
    half_width: f32,
    half_height: f32,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicObstacle {
    pub position: (f32, f32),
    pub half_width: f32,
    pub half_height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicFood {
    pub position: (f32, f32),
    pub size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Food {
    position: (f32, f32),
    size: f32,
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
    LoadSim(std::path::PathBuf),
    SaveCreatures(std::path::PathBuf, f32),
    LoadCreatures(std::path::PathBuf),
    NewGeneration,
    RecreateWorld((f32, f32)),
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
    pub poison: Vec<(f32, f32)>,
    pub obstacles: Vec<BasicObstacle>,
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
                        let mut rng = rand::thread_rng();
                        let width = g.dims.0;
                        let height = g.dims.1;
                        self.sim = Simulation::default();
                        self.sim.world_dim = g.dims;
                        self.sim.target_population = g.num_creatures;
                        self.sim.input_nodes = g.input_nodes;
                        self.sim.output_nodes = g.output_nodes;
                        self.sim.config = g.config;
                        self.sim.obstacles = generate_obstacles(&mut rng, g.dims, &self.sim.config);

                        self.gpu.clear_all();
                        self.sim.creatures = (0..g.num_creatures)
                            .map(|_i| {
                                self.sim.last_id += 1;
                                let creature = self.sim.random_creature(&mut rng, (width, height));
                                let weights = direct_brain_weights(&creature.brain)
                                    .expect("preset/random brain must be direct 2-layer net");
                                self.gpu.assign_slot(self.sim.last_id, &weights);
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
                    RunnerReq::LoadCreatures(path) => {
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            if self.sim.world_dim.0 <= 0.0 || self.sim.world_dim.1 <= 0.0 {
                                return Err("No active world. Create or load a sim first.".into());
                            }
                            let bytes = std::fs::read(&path).map_err(|e| format!("read: {e}"))?;
                            let cfg = bincode::config::standard();
                            let (loaded, _read): (Vec<Creature>, usize) =
                                bincode::serde::decode_from_slice(&bytes, cfg)
                                    .map_err(|e| format!("deserialize: {e}"))?;

                            // Drop existing creatures, free GPU slots.
                            let old_ids: Vec<usize> =
                                self.sim.creatures.iter().map(|c| *c.key()).collect();
                            for id in old_ids {
                                self.gpu.release(id);
                            }
                            self.sim.creatures.clear();

                            let mut rng = rand::thread_rng();
                            let dims = self.sim.world_dim;
                            let obstacles = self.sim.obstacles.clone();
                            let count = loaded.len();
                            for mut c in loaded {
                                c.position =
                                    random_pos_outside_obstacles(&mut rng, dims, &obstacles);
                                c.angle =
                                    rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
                                c.age = 0;
                                c.energy = self.sim.config.start_energy;
                                c.prev_energy = c.energy;
                                c.mate_cooldown = 0;
                                c.action_lock = 0;
                                c.touched = 0.0;
                                c.mem = 0.0;
                                self.sim.last_id += 1;
                                let weights = direct_brain_weights(&c.brain).ok_or_else(|| {
                                    format!(
                                        "creature {} brain not direct 2-layer",
                                        self.sim.last_id
                                    )
                                })?;
                                self.gpu.assign_slot(self.sim.last_id, &weights);
                                self.sim.creatures.insert(self.sim.last_id, c);
                                self.sim.total_spawned += 1;
                            }
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
                    RunnerReq::LoadSim(path) => {
                        self.paused = true;
                        let result = (|| -> Result<std::path::PathBuf, String> {
                            let bytes = std::fs::read(&path).map_err(|e| format!("read: {e}"))?;
                            let cfg = bincode::config::standard();
                            let (loaded, _read): (Simulation, usize) =
                                bincode::serde::decode_from_slice(&bytes, cfg)
                                    .map_err(|e| format!("deserialize: {e}"))?;
                            self.sim = loaded;

                            // Rebuild GPU weight slots from each creature's brain.
                            self.gpu.clear_all();
                            for entry in self.sim.creatures.iter() {
                                let id = *entry.key();
                                let weights = direct_brain_weights(&entry.value().brain)
                                    .ok_or_else(|| {
                                        format!("creature {id} brain not direct 2-layer")
                                    })?;
                                self.gpu.assign_slot(id, &weights);
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
                        let mut rng = rand::thread_rng();
                        self.sim.world_dim = dims;
                        self.sim.obstacles = generate_obstacles(&mut rng, dims, &self.sim.config);
                        self.sim.food.clear();
                        self.sim.poison.clear();
                        self.sim.refill_food_and_poison(&mut rng);
                        let obstacles = self.sim.obstacles.clone();
                        for mut entry in self.sim.creatures.iter_mut() {
                            let c = entry.value_mut();
                            c.position = random_pos_outside_obstacles(&mut rng, dims, &obstacles);
                            c.angle = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
                            c.touched = 0.0;
                        }
                        let _ = self.tx.send(RunnerRes::Positions(self.positions()));
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
                size: creature_size(x.value().energy, &self.sim.config),
                energy: x.value().energy,
                age: x.value().age,
                mate_cooldown: x.value().mate_cooldown,
                action_lock: x.value().action_lock,
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
                        acc.2 + creature_size(creature.energy, &self.sim.config),
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
                })
                .collect(),
            poison: self.sim.poison.clone(),
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

fn dirs_to_vec(forward: f32, backward: f32, left: f32, right: f32) -> Vec2 {
    let x = Vec2::new(right, 0.0) + Vec2::new(left * -1.0, 0.0);
    let y = Vec2::new(0.0, forward) + Vec2::new(0.0, backward * -1.0);
    let direction = x + y;
    if direction.length_squared() > 1.0 {
        direction.normalize()
    } else {
        direction
    }
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

fn creature_size(energy: f32, config: &SimulationConfig) -> f32 {
    config.min_creature_size
        + (energy / config.max_energy).clamp(0.0, 1.0)
            * (config.max_creature_size - config.min_creature_size)
}

fn set_input(inputs: &mut [engine::nn::GraphNode], idx: usize, value: f32) {
    if let Some(node) = inputs.get_mut(idx) {
        if let Node::Input(input) = &mut node.value {
            input.set_value(value);
        }
    }
}

fn speed_cap_for_energy(energy: f32, config: &SimulationConfig) -> f32 {
    ((energy - config.min_move_energy) / (config.full_speed_energy - config.min_move_energy))
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
        let w = rng.gen_range(min_s..=max_s);
        let h = rng.gen_range(min_s..=max_s);
        let hw = w * 0.5;
        let hh = h * 0.5;
        if dims.0 - 2.0 * (hw + margin) <= 0.0 || dims.1 - 2.0 * (hh + margin) <= 0.0 {
            continue;
        }
        let cx = rng.gen_range((hw + margin)..=(dims.0 - hw - margin));
        let cy = rng.gen_range((hh + margin)..=(dims.1 - hh - margin));
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
        let p = (rng.gen_range(0.0..dims.0), rng.gen_range(0.0..dims.1));
        if !point_in_any_obstacle(p, size, obstacles) {
            return p;
        }
    }
    (rng.gen_range(0.0..dims.0), rng.gen_range(0.0..dims.1))
}

/// Nearest grid item that is (a) within `vision_distance` and (b) within FOV cone of `angle`.
/// `extra_keep` is an additional per-index filter (used for excluding self in mate sensing).
fn nearest_in_fov(
    grid: &Grid,
    from_pos: Vec2,
    angle: f32,
    half_fov: f32,
    vision_distance: f32,
    mut extra_keep: impl FnMut(u32) -> bool,
) -> Option<(u32, f32)> {
    let cos_facing = angle.cos();
    let sin_facing = angle.sin();
    let cos_half = half_fov.cos();
    let vd2 = vision_distance * vision_distance;
    grid.nearest_filtered(from_pos, |idx| {
        if !extra_keep(idx) {
            return false;
        }
        let target = grid.position(idx);
        let dx = target.x - from_pos.x;
        let dy = target.y - from_pos.y;
        let d2 = dx * dx + dy * dy;
        if d2 > vd2 {
            return false;
        }
        if d2 < 1e-4 {
            return true;
        }
        let inv = 1.0 / d2.sqrt();
        let dot = (dx * cos_facing + dy * sin_facing) * inv;
        // half_fov >= π means full circle (cos becomes -1 or less); accept all.
        if half_fov >= std::f32::consts::PI {
            return true;
        }
        dot >= cos_half
    })
}

/// Find nearest obstacle whose center is in FOV + vision range. Returns world-frame normalized signal.
fn nearest_obstacle_signal(
    from_pos: Vec2,
    angle: f32,
    half_fov: f32,
    vision_distance: f32,
    obstacles: &[Obstacle],
    dims: (f32, f32),
) -> (f32, f32, f32) {
    if obstacles.is_empty() {
        return (0.0, 0.0, 1.0);
    }
    let cos_facing = angle.cos();
    let sin_facing = angle.sin();
    let cos_half = half_fov.cos();
    let vd2 = vision_distance * vision_distance;
    let full_circle = half_fov >= std::f32::consts::PI;

    let mut best: Option<(usize, f32)> = None;
    for (i, o) in obstacles.iter().enumerate() {
        let dx = o.position.0 - from_pos.x;
        let dy = o.position.1 - from_pos.y;
        let d2 = dx * dx + dy * dy;
        if d2 > vd2 {
            continue;
        }
        if !full_circle && d2 > 1e-4 {
            let inv = 1.0 / d2.sqrt();
            let dot = (dx * cos_facing + dy * sin_facing) * inv;
            if dot < cos_half {
                continue;
            }
        }
        match best {
            Some((_, bd)) if bd <= d2 => {}
            _ => best = Some((i, d2)),
        }
    }
    let Some((i, d2)) = best else {
        return (0.0, 0.0, 1.0);
    };
    let o = &obstacles[i];
    let dx = o.position.0 - from_pos.x;
    let dy = o.position.1 - from_pos.y;
    let diagonal = (dims.0.powi(2) + dims.1.powi(2)).sqrt().max(1.0);
    (
        dx / dims.0.max(1.0),
        dy / dims.1.max(1.0),
        d2.sqrt() / diagonal,
    )
}

/// 4-sector vision: split FOV cone into 4 angular bins. Each bin returns closeness in [0,1]
/// (1 = right at creature, 0 = nothing). Iterates targets via grid AABB query.
fn sector_vision_grid(
    grid: &Grid,
    from_pos: Vec2,
    angle: f32,
    half_fov: f32,
    vision_distance: f32,
    mut keep: impl FnMut(u32) -> bool,
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    if vision_distance <= 0.0 {
        return out;
    }
    let cos_facing = angle.cos();
    let sin_facing = angle.sin();
    let cos_half = half_fov.cos();
    let full_circle = half_fov >= std::f32::consts::PI;
    let vd2 = vision_distance * vision_distance;
    let inv_v = 1.0 / vision_distance;
    let pmin = from_pos - Vec2::splat(vision_distance);
    let pmax = from_pos + Vec2::splat(vision_distance);
    grid.query_aabb(pmin, pmax, |idx| {
        if !keep(idx) {
            return;
        }
        let target = grid.position(idx);
        let dx = target.x - from_pos.x;
        let dy = target.y - from_pos.y;
        let d2 = dx * dx + dy * dy;
        if d2 > vd2 || d2 < 1e-6 {
            return;
        }
        let d = d2.sqrt();
        let inv = 1.0 / d;
        let dot = (dx * cos_facing + dy * sin_facing) * inv;
        if !full_circle && dot < cos_half {
            return;
        }
        // Signed angle from facing → target (cross sign).
        let cross = cos_facing * dy - sin_facing * dx;
        let rel = cross.atan2((dx * cos_facing + dy * sin_facing).max(-1e9));
        // rel in [-half_fov, half_fov]. Map to sector 0..3.
        let span = (half_fov * 2.0).max(1e-3);
        let t = ((rel + half_fov) / span).clamp(0.0, 0.9999);
        let s = (t * 4.0) as usize;
        let closeness = (1.0 - d * inv_v).clamp(0.0, 1.0);
        if closeness > out[s] {
            out[s] = closeness;
        }
    });
    out
}

/// Same as sector_vision_grid but operates on obstacles (linear scan over centers).
fn sector_vision_obstacles(
    obstacles: &[Obstacle],
    from_pos: Vec2,
    angle: f32,
    half_fov: f32,
    vision_distance: f32,
) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    if obstacles.is_empty() || vision_distance <= 0.0 {
        return out;
    }
    let cos_facing = angle.cos();
    let sin_facing = angle.sin();
    let cos_half = half_fov.cos();
    let full_circle = half_fov >= std::f32::consts::PI;
    let vd2 = vision_distance * vision_distance;
    let inv_v = 1.0 / vision_distance;
    for o in obstacles {
        let dx = o.position.0 - from_pos.x;
        let dy = o.position.1 - from_pos.y;
        let d2 = dx * dx + dy * dy;
        if d2 > vd2 || d2 < 1e-6 {
            continue;
        }
        let d = d2.sqrt();
        let inv = 1.0 / d;
        let dot = (dx * cos_facing + dy * sin_facing) * inv;
        if !full_circle && dot < cos_half {
            continue;
        }
        let cross = cos_facing * dy - sin_facing * dx;
        let rel = cross.atan2((dx * cos_facing + dy * sin_facing).max(-1e9));
        let span = (half_fov * 2.0).max(1e-3);
        let t = ((rel + half_fov) / span).clamp(0.0, 0.9999);
        let s = (t * 4.0) as usize;
        let closeness = (1.0 - d * inv_v).clamp(0.0, 1.0);
        if closeness > out[s] {
            out[s] = closeness;
        }
    }
    out
}

fn nearest_signal_from(
    from_pos: Vec2,
    target: Option<(u32, f32)>,
    grid: &Grid,
    dims: (f32, f32),
) -> (f32, f32, f32) {
    let Some((idx, d2)) = target else {
        return (0.0, 0.0, 1.0);
    };
    let target_pos = grid.position(idx);
    let dx = target_pos.x - from_pos.x;
    let dy = target_pos.y - from_pos.y;
    let diagonal = (dims.0.powi(2) + dims.1.powi(2)).sqrt().max(1.0);
    (
        dx / dims.0.max(1.0),
        dy / dims.1.max(1.0),
        d2.sqrt() / diagonal,
    )
}

impl Simulation {
    fn run(&mut self, gpu: &mut GpuBrainCompute) {
        let start_all = Instant::now();
        self.food_eaten_this_tick = 0;
        self.births_this_tick = 0;
        self.deaths_this_tick = 0;

        let dims = self.world_dim;

        // Build spatial grids for food, poison, and creatures.
        let food_positions: Vec<Vec2> = self
            .food
            .iter()
            .map(|f| Vec2::new(f.position.0, f.position.1))
            .collect();
        let poison_positions: Vec<Vec2> = self.poison.iter().map(|p| Vec2::new(p.0, p.1)).collect();

        let creature_snapshot: Vec<(usize, (f32, f32))> = self
            .creatures
            .iter()
            .map(|c| (*c.key(), c.value().position))
            .collect();
        let creature_positions: Vec<Vec2> = creature_snapshot
            .iter()
            .map(|(_, p)| Vec2::new(p.0, p.1))
            .collect();
        let creature_grid_ids: Vec<usize> = creature_snapshot.iter().map(|(id, _)| *id).collect();

        let food_grid = Grid::build(food_positions, GRID_CELL, dims);
        let poison_grid = Grid::build(poison_positions, GRID_CELL, dims);
        let creature_grid = Grid::build(creature_positions, GRID_CELL, dims);

        // Ordered list of creature ids, used as the parallel index for inputs/outputs.
        let creature_ids: Vec<usize> = self.creatures.iter().map(|c| *c.key()).collect();
        let n = creature_ids.len();

        // Build flat inputs vector in parallel; outputs preserve order of creature_ids.
        let t_start_inputs = Instant::now();
        let mut inputs_flat: Vec<f32> = vec![0.0; n * GPU_INPUTS];
        creature_ids
            .par_iter()
            .zip(inputs_flat.par_chunks_mut(GPU_INPUTS))
            .for_each(|(id, slot)| {
                let Some(c) = self.creatures.get(id) else {
                    return;
                };
                let arr = self.input_values_for(
                    *id,
                    c.value(),
                    &food_grid,
                    &poison_grid,
                    &creature_grid,
                    &creature_grid_ids,
                    dims,
                    c.age + 1,
                );
                slot.copy_from_slice(&arr);
            });
        let t_inputs = t_start_inputs.elapsed();

        // GPU dispatch (synchronous). Order matches creature_ids.
        let t_start_gpu = Instant::now();
        let outputs: Vec<[f32; GPU_OUTPUTS]> = if n == 0 {
            Vec::new()
        } else {
            gpu.compute(&creature_ids, &inputs_flat)
        };
        let t_gpu = t_start_gpu.elapsed();

        // Map output index → id for parallel update lookup.
        let mut id_to_idx: HashMap<usize, usize> = HashMap::with_capacity(n);
        for (i, id) in creature_ids.iter().enumerate() {
            id_to_idx.insert(*id, i);
        }

        let t_start_update = Instant::now();
        let eaten_food_indices = DashSet::<usize>::new();
        let food_eaten_count = AtomicUsize::new(0);
        let hit_poison = DashSet::<usize>::new();
        let attack_intents: DashMap<usize, (usize, f32)> = DashMap::new(); // attacker_id -> (target_id, damage)

        self.creatures.par_iter_mut().for_each(|mut accessor| {
            let id = *accessor.key();
            let c = accessor.value_mut();
            c.mate_cooldown = c.mate_cooldown.saturating_sub(1);
            c.action_lock = c.action_lock.saturating_sub(1);
            c.age += 1;
            c.prev_energy = c.energy;
            c.touched = 0.0;

            // Write inputs back into brain graph (for inspector / preserved semantics).
            if let Some(&i) = id_to_idx.get(&id) {
                let base = i * GPU_INPUTS;
                let inputs = &mut c.brain.graph.layers[c.brain.input_layer as usize];
                for k in 0..GPU_INPUTS {
                    set_input(inputs, k, inputs_flat[base + k]);
                }

                // Apply GPU partials (sigmoid + save).
                if i < outputs.len() {
                    let partials = &outputs[i];
                    let output_layer_idx = c.brain.output_layer as usize;
                    for (idx, partial) in partials.iter().copied().enumerate() {
                        if let Some(node) = c.brain.graph.layers[output_layer_idx].get_mut(idx) {
                            if let Node::Output(output) = &mut node.value {
                                output.finish_and_save(partial);
                            }
                        }
                    }
                }
            }

            let output_layer_idx = c.brain.output_layer as usize;
            let output_layer = &c.brain.graph.layers[output_layer_idx];
            let thrust = output_layer
                .get(0)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let turn_left = output_layer
                .get(1)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let turn_right = output_layer
                .get(2)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
            let speed_out = output_layer
                .get(3)
                .map(|n| get_output_value(&n.value))
                .unwrap_or_default();
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

            // Memory feedback: rescale [0,1] sigmoid to [-1,1] for richer signal next tick.
            c.mem = mem_out * 2.0 - 1.0;

            // Turn + thrust movement.
            let turn = (turn_left - turn_right) * self.config.max_turn_rate;
            c.angle += turn;
            // Wrap angle.
            if c.angle > std::f32::consts::PI {
                c.angle -= std::f32::consts::TAU;
            }
            if c.angle < -std::f32::consts::PI {
                c.angle += std::f32::consts::TAU;
            }

            let speed_cap = speed_cap_for_energy(c.energy, &self.config);
            let c_size_now = creature_size(c.energy, &self.config);
            let size_n = ((c_size_now - self.config.min_creature_size)
                / (self.config.max_creature_size - self.config.min_creature_size).max(1e-3))
            .clamp(0.0, 1.0);
            let speed_factor = (1.0 - self.config.size_speed_penalty * size_n).max(0.0);
            let cost_factor = 1.0 + self.config.size_move_cost_factor * size_n;
            let desired = if c.action_lock == 0 {
                thrust * speed_out * speed_cap
            } else {
                0.0
            };
            let move_amount = desired * speed_factor;

            c.energy -= self.config.base_energy_cost
                + (move_amount * self.config.move_energy_cost * cost_factor)
                + (turn.abs() * self.config.turn_energy_cost)
                + (mate * self.config.mate_attempt_cost)
                + (eat * self.config.eat_attempt_cost)
                + (attack * self.config.attack_cost);

            let dx = c.angle.cos() * move_amount;
            let dy = c.angle.sin() * move_amount;
            let new_x = (c.position.0 + dx).clamp(0.0, self.world_dim.0);
            if !point_in_any_obstacle((new_x, c.position.1), c_size_now, &self.obstacles) {
                c.position.0 = new_x;
            } else {
                c.touched = 1.0;
            }
            let new_y = (c.position.1 + dy).clamp(0.0, self.world_dim.1);
            if !point_in_any_obstacle((c.position.0, new_y), c_size_now, &self.obstacles) {
                c.position.1 = new_y;
            } else {
                c.touched = 1.0;
            }

            // Poison hits: grid AABB query around creature.
            let pos = Vec2::new(c.position.0, c.position.1);
            let poison_half = CREATURE_DIM_HALF;
            let pmin = pos - Vec2::splat(poison_half + GRID_CELL);
            let pmax = pos + Vec2::splat(poison_half + GRID_CELL);
            poison_grid.query_aabb(pmin, pmax, |p_idx| {
                let p = self.poison[p_idx as usize];
                if do_squares_collide(c.position, p) {
                    c.energy -= self.config.poison_damage;
                    hit_poison.insert(p_idx as usize);
                }
            });

            // Eat food: grid AABB query.
            if c.action_lock == 0 && eat > 0.5 {
                let c_size = creature_size(c.energy, &self.config);
                let half = (c_size + self.config.max_food_size) * 0.5;
                let fmin = pos - Vec2::splat(half + GRID_CELL);
                let fmax = pos + Vec2::splat(half + GRID_CELL);
                let mut ate = false;
                food_grid.query_aabb(fmin, fmax, |f_idx| {
                    if ate {
                        return;
                    }
                    let f_idx_us = f_idx as usize;
                    if eaten_food_indices.contains(&f_idx_us) {
                        return;
                    }
                    let f = &self.food[f_idx_us];
                    if c_size >= f.size
                        && do_sized_squares_collide(c.position, c_size, f.position, f.size)
                        && eaten_food_indices.insert(f_idx_us)
                    {
                        c.energy -= self.config.eat_action_cost;
                        c.action_lock = self.config.eat_action_base_ticks
                            + (f.size * self.config.food_eat_ticks_per_size) as u16;
                        food_eaten_count.fetch_add(1, Ordering::Relaxed);
                        c.energy = (c.energy + f.size * self.config.food_energy_per_size)
                            .min(self.config.max_energy);
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
                    let other_id = creature_grid_ids[idx as usize];
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
                    attack_intents.insert(id, (target_id, self.config.attack_damage));
                }
            }
        });

        // Apply attacks: deduct from target, transfer ratio to attacker.
        for entry in attack_intents.iter() {
            let attacker_id = *entry.key();
            let (target_id, dmg) = *entry.value();
            let actual_damage = if let Some(mut t) = self.creatures.get_mut(&target_id) {
                let before = t.energy;
                t.energy -= dmg;
                (before - t.energy).max(0.0)
            } else {
                0.0
            };
            if actual_damage > 0.0 {
                if let Some(mut a) = self.creatures.get_mut(&attacker_id) {
                    a.energy = (a.energy + actual_damage * self.config.attack_steal_ratio)
                        .min(self.config.max_energy);
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
                && (self.config.ignore_max_age || creature.age <= self.config.max_age);
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

        let mut i = 0;
        self.food.retain(|_| {
            let alive = !eaten_food_indices.contains(&i);
            i += 1;
            alive
        });
        let mut i = 0;
        self.poison.retain(|_| {
            let alive = !hit_poison.contains(&i);
            i += 1;
            alive
        });

        let mut rng = rand::thread_rng();
        self.refill_food_and_poison(&mut rng);
        if self.config.creature_spawning_enabled {
            while self.creatures.len() < self.target_population {
                self.last_id += 1;
                let creature = self.random_creature(&mut rng, self.world_dim);
                let weights = direct_brain_weights(&creature.brain)
                    .expect("random spawn brain must be direct 2-layer net");
                gpu.assign_slot(self.last_id, &weights);
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

    fn input_values_for(
        &self,
        id: usize,
        c: &Creature,
        food_grid: &Grid,
        poison_grid: &Grid,
        creature_grid: &Grid,
        creature_ids: &[usize],
        dims: (f32, f32),
        age: u32,
    ) -> [f32; GPU_INPUTS] {
        let output_layer_idx = c.brain.output_layer as usize;
        let layer = c.brain.graph.layers.get(output_layer_idx);
        let prev_thrust = layer
            .and_then(|l| l.get(0))
            .map(|n| get_output_value(&n.value))
            .unwrap_or(0.0);
        let prev_tl = layer
            .and_then(|l| l.get(1))
            .map(|n| get_output_value(&n.value))
            .unwrap_or(0.0);
        let prev_tr = layer
            .and_then(|l| l.get(2))
            .map(|n| get_output_value(&n.value))
            .unwrap_or(0.0);

        let pos = Vec2::new(c.position.0, c.position.1);
        let half_fov = self.config.fov_angle * 0.5;
        let vision = self.config.vision_distance;

        let food_sect = sector_vision_grid(food_grid, pos, c.angle, half_fov, vision, |_| true);
        let poison_sect = sector_vision_grid(poison_grid, pos, c.angle, half_fov, vision, |_| true);
        let obstacle_sect =
            sector_vision_obstacles(&self.obstacles, pos, c.angle, half_fov, vision);
        let creature_sect =
            sector_vision_grid(creature_grid, pos, c.angle, half_fov, vision, |idx| {
                creature_ids[idx as usize] != id
            });

        let max_e = self.config.max_energy.max(1.0);
        let hunger = 1.0 - (c.energy / max_e).clamp(0.0, 1.0);
        let health = (c.energy / max_e).clamp(0.0, 1.0);
        let damage = ((c.prev_energy - c.energy).max(0.0) / max_e).clamp(0.0, 1.0);
        let osc = (age as f32 * 0.1).sin();
        let size_n = ((creature_size(c.energy, &self.config) - self.config.min_creature_size)
            / (self.config.max_creature_size - self.config.min_creature_size).max(1e-3))
        .clamp(0.0, 1.0);

        [
            hunger,
            health,
            prev_thrust,
            prev_tl - prev_tr,
            c.angle.cos(),
            c.angle.sin(),
            c.touched,
            damage,
            osc,
            c.mem,
            (age as f32 / self.config.max_age as f32).clamp(0.0, 1.0),
            size_n,
            1.0 - (c.position.0 / dims.0.max(1.0)).clamp(0.0, 1.0),
            (c.position.0 / dims.0.max(1.0)).clamp(0.0, 1.0),
            1.0 - (c.position.1 / dims.1.max(1.0)).clamp(0.0, 1.0),
            (c.position.1 / dims.1.max(1.0)).clamp(0.0, 1.0),
            food_sect[0],
            food_sect[1],
            food_sect[2],
            food_sect[3],
            poison_sect[0],
            poison_sect[1],
            poison_sect[2],
            poison_sect[3],
            obstacle_sect[0],
            obstacle_sect[1],
            obstacle_sect[2],
            obstacle_sect[3],
            creature_sect[0],
            creature_sect[1],
            creature_sect[2],
            creature_sect[3],
        ]
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
                if creature.action_lock != 0
                    || creature.mate_cooldown != 0
                    || creature.energy < self.config.min_mate_energy
                {
                    return None;
                }
                let output_layer =
                    &creature.brain.graph.layers[creature.brain.output_layer as usize];
                let is_mating = output_layer
                    .get(4)
                    .map(|n| get_output_value(&n.value) > self.config.mate_threshold)
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
                if a.energy < self.config.min_mate_energy || b.energy < self.config.min_mate_energy
                {
                    continue;
                }
                if a.action_lock != 0 || b.action_lock != 0 {
                    continue;
                }
                Some((
                    a.brain.clone(),
                    b.brain.clone(),
                    (
                        ((a_position.0 + b_position.0) / 2.0).clamp(0.0, self.world_dim.0),
                        ((a_position.1 + b_position.1) / 2.0).clamp(0.0, self.world_dim.1),
                    ),
                ))
            };

            if let Some((a_brain, b_brain, child_pos)) = pair_data {
                parent_pairs.push((*a_id, b_id, a_brain, b_brain, child_pos));
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
        let config = self.config.clone();
        let births: Vec<_> = parent_pairs
            .into_par_iter()
            .map(|(a_id, b_id, a_brain, b_brain, child_pos)| {
                let mut rng = rand::thread_rng();
                let mut child_brain =
                    direct_crossover(&a_brain, &b_brain, &input_nodes, &output_nodes, &mut rng);
                Self::mutate_child_brain(&mut child_brain, &config);
                (a_id, b_id, child_brain, child_pos)
            })
            .collect();

        for (a_id, b_id, child_brain, child_pos) in births {
            if let Some(mut a) = self.creatures.get_mut(&a_id) {
                a.energy -= self.config.mate_energy_cost;
                a.mate_cooldown = self.config.mate_cooldown_ticks;
                a.action_lock = self.config.eat_action_base_ticks;
            }
            if let Some(mut b) = self.creatures.get_mut(&b_id) {
                b.energy -= self.config.mate_energy_cost;
                b.mate_cooldown = self.config.mate_cooldown_ticks;
                b.action_lock = self.config.eat_action_base_ticks;
            }
            self.last_id += 1;
            self.total_spawned += 1;
            self.births_this_tick += 1;
            self.total_births += 1;
            let weights =
                direct_brain_weights(&child_brain).expect("child brain must be direct 2-layer net");
            gpu.assign_slot(self.last_id, &weights);
            self.creatures.insert(
                self.last_id,
                Creature {
                    brain: child_brain,
                    position: child_pos,
                    angle: 0.0,
                    energy: self.config.child_energy,
                    age: 0,
                    mate_cooldown: self.config.mate_cooldown_ticks,
                    action_lock: self.config.eat_action_base_ticks,
                    mem: 0.0,
                    prev_energy: self.config.child_energy,
                    touched: 0.0,
                },
            );
        }
    }

    fn restart_generation(&mut self, gpu: &mut GpuBrainCompute) {
        let mut rng = rand::thread_rng();

        let survivor_brains: Vec<Net> = self
            .creatures
            .iter()
            .map(|c| c.value().brain.clone())
            .collect();
        if survivor_brains.is_empty() {
            return;
        }

        self.generation += 1;
        info!(
            "Extinction restart — generation {} (survivors: {})",
            self.generation,
            survivor_brains.len()
        );

        let all_ids: Vec<usize> = self.creatures.iter().map(|c| *c.key()).collect();
        for id in &all_ids {
            gpu.release(*id);
        }
        self.creatures.clear();

        let width = self.world_dim.0;
        let height = self.world_dim.1;
        let input_nodes = self.input_nodes.clone();
        let output_nodes = self.output_nodes.clone();

        for _ in 0..self.target_population {
            self.last_id += 1;
            let a_brain = &survivor_brains[rng.gen_range(0..survivor_brains.len())];
            let b_brain = &survivor_brains[rng.gen_range(0..survivor_brains.len())];
            let mut child_brain =
                direct_crossover(a_brain, b_brain, &input_nodes, &output_nodes, &mut rng);
            Self::mutate_child_brain(&mut child_brain, &self.config);
            let weights = direct_brain_weights(&child_brain)
                .expect("restart child brain must be direct 2-layer net");
            gpu.assign_slot(self.last_id, &weights);
            self.creatures.insert(
                self.last_id,
                Creature {
                    brain: child_brain,
                    position: random_pos_outside_obstacles(
                        &mut rng,
                        (width, height),
                        &self.obstacles,
                    ),
                    angle: rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
                    energy: self.config.start_energy,
                    age: 0,
                    mate_cooldown: 0,
                    action_lock: 0,
                    mem: 0.0,
                    prev_energy: self.config.start_energy,
                    touched: 0.0,
                },
            );
            self.total_spawned += 1;
        }

        self.food.clear();
        self.poison.clear();
        self.refill_food_and_poison(&mut rng);
    }

    fn random_creature(&self, rng: &mut impl Rng, dims: (f32, f32)) -> Creature {
        Creature {
            brain: random_direct_net(&self.input_nodes, &self.output_nodes, rng),
            position: random_pos_outside_obstacles(rng, dims, &self.obstacles),
            angle: rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
            energy: self.config.start_energy,
            age: 0,
            mate_cooldown: 0,
            action_lock: 0,
            mem: 0.0,
            prev_energy: self.config.start_energy,
            touched: 0.0,
        }
    }

    fn refill_food_and_poison(&mut self, rng: &mut impl Rng) {
        let pop = self.creatures.len().max(1);
        let food_target = ((pop as f32 * self.config.food_spawn_multiplier) as usize
            / self.config.food_per_creature)
            .max(1);
        let poison_target = (pop / self.config.poison_per_creature).max(1);
        while self.food.len() < food_target {
            let size = rng.gen_range(self.config.min_food_size..=self.config.max_food_size);
            let pos =
                random_pos_outside_obstacles_sized(rng, self.world_dim, &self.obstacles, size);
            self.food.push(Food {
                position: pos,
                size,
            });
        }
        while self.poison.len() < poison_target {
            let pos = random_pos_outside_obstacles_sized(
                rng,
                self.world_dim,
                &self.obstacles,
                CREATURE_DIM,
            );
            self.poison.push(pos);
        }
    }

    fn mutate_child_brain(net: &mut Net, config: &SimulationConfig) {
        let mut rng = rand::thread_rng();
        let il = net.input_layer;
        let ol = net.output_layer;
        for i in 0..GPU_INPUTS as u16 {
            for o in 0..GPU_OUTPUTS as u16 {
                let from = GraphLocation::new(il, i);
                let to = GraphLocation::new(ol, o);
                if let Some(edge) = net.graph.get_edge_mut(&from, &to) {
                    if rng.gen::<f32>() < config.mutation_rate {
                        edge.value.weight +=
                            rng.gen_range(-config.mutation_amount..config.mutation_amount);
                    }
                    if rng.gen::<f32>() < config.mutation_rate / 4.0 {
                        edge.value.enabled = !edge.value.enabled;
                    }
                } else if rng.gen::<f32>() < config.mutation_rate / 4.0 {
                    let _ = net.graph.add_edge(
                        from,
                        to,
                        Edge {
                            weight: rng.gen_range(-config.mutation_amount..config.mutation_amount),
                            enabled: true,
                        },
                    );
                }
            }
        }
    }
}

fn random_direct_net(input_nodes: &[Node], output_nodes: &[Node], rng: &mut impl Rng) -> Net {
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
    let mut g = NeuralGraph::new();
    let il = g.add_layer_to_end();
    for n in input_nodes {
        g.add_node(il, GraphNode::new(n.clone())).unwrap();
    }
    let ol = g.add_layer_to_end();
    for n in output_nodes {
        g.add_node(ol, GraphNode::new(n.clone())).unwrap();
    }
    // Sparse, low-weight starter brains. Roughly 2-4 connections per creature average,
    // small magnitudes so behavior is gentle rather than chaotic.
    for i in 0..GPU_INPUTS as u16 {
        for o in 0..GPU_OUTPUTS as u16 {
            if rng.gen_bool(0.15) {
                let _ = g.add_edge(
                    GraphLocation::new(il, i),
                    GraphLocation::new(ol, o),
                    Edge {
                        weight: rng.gen_range(-1.5..1.5),
                        enabled: true,
                    },
                );
            }
        }
    }
    Net {
        graph: g,
        input_layer: il,
        output_layer: ol,
    }
}

fn direct_crossover(
    a: &Net,
    b: &Net,
    input_nodes: &[Node],
    output_nodes: &[Node],
    rng: &mut impl Rng,
) -> Net {
    let wa = direct_brain_weights(a).expect("crossover parent A must be direct");
    let wb = direct_brain_weights(b).expect("crossover parent B must be direct");
    let mut g = NeuralGraph::new();
    let il = g.add_layer_to_end();
    for n in input_nodes {
        g.add_node(il, GraphNode::new(n.clone())).unwrap();
    }
    let ol = g.add_layer_to_end();
    for n in output_nodes {
        g.add_node(ol, GraphNode::new(n.clone())).unwrap();
    }
    for i in 0..GPU_INPUTS {
        for o in 0..GPU_OUTPUTS {
            let idx = i * GPU_OUTPUTS + o;
            let weight = if rng.gen_bool(0.5) { wa[idx] } else { wb[idx] };
            if weight != 0.0 {
                let _ = g.add_edge(
                    GraphLocation::new(il, i as u16),
                    GraphLocation::new(ol, o as u16),
                    Edge {
                        weight,
                        enabled: true,
                    },
                );
            }
        }
    }
    Net {
        graph: g,
        input_layer: il,
        output_layer: ol,
    }
}
