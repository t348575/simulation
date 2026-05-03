use std::{thread::sleep, time::Duration};

use bevy::math::Vec2;
use dashmap::DashMap;
use engine::nn::{
    reproduce::{Crossover, DefaultIterator},
    Edge, GraphLocation, GraphNode, NeuralGraph, Net, Node,
};
use flume::{unbounded, Receiver, Sender};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const CREATURE_DIM: f32 = 5.0;
pub const CREATURE_DIM_HALF: f32 = CREATURE_DIM / 2.0;
const MIN_CREATURE_SIZE: f32 = 3.0;
const MAX_CREATURE_SIZE: f32 = 14.0;
const MIN_FOOD_SIZE: f32 = 2.0;
const MAX_FOOD_SIZE: f32 = 10.0;
const FOOD_ENERGY_PER_SIZE: f32 = 7.0;
const FOOD_EAT_TICKS_PER_SIZE: f32 = 4.0;
const MATE_THRESHOLD: f32 = 0.5;
const MATE_COOLDOWN_TICKS: u16 = 120;
const MAX_BIRTHS_PER_TICK: usize = 10;
const START_ENERGY: f32 = 80.0;
const MAX_ENERGY: f32 = 120.0;
const BASE_ENERGY_COST: f32 = 0.02;
const MOVE_ENERGY_COST: f32 = 0.08;
const MIN_MOVE_ENERGY: f32 = 5.0;
const FULL_SPEED_ENERGY: f32 = 45.0;
const MATE_ATTEMPT_COST: f32 = 0.03;
const EAT_ATTEMPT_COST: f32 = 0.02;
const EAT_ACTION_COST: f32 = 0.5;
const EAT_ACTION_BASE_TICKS: u16 = 8;
const MATE_ACTION_TICKS: u16 = 60;
const POISON_DAMAGE: f32 = 45.0;
const MATE_ENERGY_COST: f32 = 25.0;
const MIN_MATE_ENERGY: f32 = 55.0;
const CHILD_ENERGY: f32 = 45.0;
const MAX_AGE: u32 = 8_000;
const FOOD_PER_CREATURE: usize = 4;
const POISON_PER_CREATURE: usize = 12;
const MUTATION_RATE: f32 = 0.05;
const MUTATION_AMOUNT: f32 = 0.35;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Simulation {
    world_dim: (f32, f32),
    creatures: DashMap<usize, Creature>,
    food: Vec<Food>,
    poison: Vec<(f32, f32)>,
    last_id: usize,
    ticks: usize,
    target_population: usize,
    input_nodes: Vec<Node>,
    output_nodes: Vec<Node>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Creature {
    brain: Net,
    position: (f32, f32),
    energy: f32,
    age: u32,
    mate_cooldown: u16,
    action_lock: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicCreature {
    pub position: (f32, f32),
    pub id: usize,
    pub size: f32,
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

/// Adds a small random offset to every edge weight so cloned preset brains differ slightly.
fn jitter_brain(mut net: Net, rng: &mut impl Rng) -> Net {
    for layer in &mut net.graph.layers {
        for node in layer {
            for edge in &mut node.connections {
                edge.value.weight += rng.gen_range(-0.4..0.4f32);
            }
        }
    }
    net
}

/// Build a two-layer (input → output) net and wire the given (input_idx, output_idx, weight) triples.
fn build_direct_net(
    input_nodes: &[Node],
    output_nodes: &[Node],
    wires: &[(u16, u16, f32)],
) -> Net {
    let mut g = NeuralGraph::new();
    let il = g.add_layer_to_end();
    for node in input_nodes {
        g.add_node(il, GraphNode::new(node.clone())).unwrap();
    }
    let ol = g.add_layer_to_end();
    for node in output_nodes {
        g.add_node(ol, GraphNode::new(node.clone())).unwrap();
    }
    for &(from_node, to_node, weight) in wires {
        let _ = g.add_edge(
            GraphLocation::new(il, from_node),
            GraphLocation::new(ol, to_node),
            Edge { weight, enabled: true },
        );
    }
    Net { graph: g, input_layer: il, output_layer: ol }
}

// Input indices:  0=hunger 1=age 2=health 3=speed 4=food_dx 5=food_dy 6=food_dist
//                 7=mate_dx 8=mate_dy 9=mate_dist 10=wall_left 11=wall_right
//                 12=wall_bottom 13=wall_top 14=poison_dx 15=poison_dy 16=poison_dist
// Output indices: 0=forward 1=backward 2=left 3=right 4=speed 5=mate 6=eat

/// Seeks food directly, eats when hungry, mates when healthy.
fn preset_brain_forager(input_nodes: &[Node], output_nodes: &[Node], rng: &mut impl Rng) -> Net {
    let net = build_direct_net(input_nodes, output_nodes, &[
        (4, 3, 8.0),  // food_dx → right
        (4, 2, -8.0), // food_dx → left
        (5, 0, 8.0),  // food_dy → forward
        (5, 1, -8.0), // food_dy → backward
        (2, 4, 5.0),  // health  → speed
        (0, 6, 7.0),  // hunger  → eat
        (2, 5, 3.0),  // health  → mate
    ]);
    jitter_brain(net, rng)
}

/// Flees poison, also steers toward food at lower priority.
fn preset_brain_avoider(input_nodes: &[Node], output_nodes: &[Node], rng: &mut impl Rng) -> Net {
    let net = build_direct_net(input_nodes, output_nodes, &[
        (14, 2,  8.0), // poison_dx → left   (flee right-side poison)
        (14, 3, -8.0), // poison_dx → right  (don't go toward right-side poison)
        (15, 1,  8.0), // poison_dy → backward
        (15, 0, -8.0), // poison_dy → forward
        (4, 3,  4.0),  // food_dx → right  (lower priority than poison flee)
        (4, 2, -4.0),  // food_dx → left
        (5, 0,  4.0),  // food_dy → forward
        (5, 1, -4.0),  // food_dy → backward
        (2, 4,  5.0),  // health  → speed
        (0, 6,  7.0),  // hunger  → eat
        (2, 5,  3.0),  // health  → mate
    ]);
    jitter_brain(net, rng)
}

/// Bounces off walls, always moving; eats and mates aggressively.
fn preset_brain_wanderer(input_nodes: &[Node], output_nodes: &[Node], rng: &mut impl Rng) -> Net {
    let net = build_direct_net(input_nodes, output_nodes, &[
        (10, 3, 8.0), // wall_left   → right
        (11, 2, 8.0), // wall_right  → left
        (12, 0, 8.0), // wall_bottom → forward
        (13, 1, 8.0), // wall_top    → backward
        (4,  3, 3.0), // food_dx     → right  (gentle food attraction)
        (4,  2,-3.0), // food_dx     → left
        (5,  0, 3.0), // food_dy     → forward
        (5,  1,-3.0), // food_dy     → backward
        (2,  4, 7.0), // health      → speed  (stay fast)
        (0,  6, 8.0), // hunger      → eat
        (2,  5, 5.0), // health      → mate   (mate aggressively)
    ]);
    jitter_brain(net, rng)
}

pub struct Runner {
    rx: Receiver<RunnerReq>,
    tx: Sender<RunnerRes>,
    sim: Simulation,
    paused: bool,
}

#[derive(Debug, Clone)]
pub enum RunnerReq {
    Generate(Generate),
    Resume,
    Pause,
    GetNet(usize),
}

#[derive(Debug, Clone)]
pub enum RunnerRes {
    Positions(Positions),
    Net(Option<Net>),
}

#[derive(Debug, Clone)]
pub struct Positions {
    pub creatures: Vec<BasicCreature>,
    pub food: Vec<BasicFood>,
    pub poison: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
pub struct Generate {
    pub num_creatures: usize,
    pub input_nodes: Vec<Node>,
    pub output_nodes: Vec<Node>,
    pub dims: (f32, f32),
}

impl Runner {
    pub fn new() -> (Self, Sender<RunnerReq>, Receiver<RunnerRes>) {
        let (tx_req, rx_req) = unbounded();
        let (tx_res, rx_res) = unbounded();
        let r = Runner {
            rx: rx_req,
            tx: tx_res,
            sim: Simulation::default(),
            paused: true,
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

                        // Seed ~12% of the initial population with hand-crafted brains so
                        // the world never stalls while purely random creatures find their footing.
                        let preset_count = (g.num_creatures / 8).max(2);
                        let preset_brains = [
                            preset_brain_forager(
                                &self.sim.input_nodes,
                                &self.sim.output_nodes,
                                &mut rng,
                            ),
                            preset_brain_avoider(
                                &self.sim.input_nodes,
                                &self.sim.output_nodes,
                                &mut rng,
                            ),
                            preset_brain_wanderer(
                                &self.sim.input_nodes,
                                &self.sim.output_nodes,
                                &mut rng,
                            ),
                        ];
                        self.sim.creatures = (0..g.num_creatures)
                            .map(|i| {
                                self.sim.last_id += 1;
                                let creature = if i < preset_count {
                                    // Each preset clone gets its own jitter via the stored brain.
                                    Creature {
                                        brain: preset_brains[i % preset_brains.len()].clone(),
                                        position: (
                                            rng.gen_range(0.0..width),
                                            rng.gen_range(0.0..height),
                                        ),
                                        energy: START_ENERGY,
                                        age: 0,
                                        mate_cooldown: 0,
                                        action_lock: 0,
                                    }
                                } else {
                                    self.sim.random_creature(&mut rng, (width, height))
                                };
                                (self.sim.last_id, creature)
                            })
                            .collect();
                        self.sim.refill_food_and_poison(&mut rng);
                        self.tx
                            .send(RunnerRes::Positions(self.positions()))
                            .expect("Could not send generated positions");
                    }
                    RunnerReq::Resume => self.paused = false,
                    RunnerReq::Pause => self.paused = true,
                    RunnerReq::GetNet(id) => {
                        match self.sim.creatures.iter().find(|c| *c.key() == id) {
                            Some(c) => self.tx.send(RunnerRes::Net(Some(c.brain.clone()))),
                            None => self.tx.send(RunnerRes::Net(None)),
                        }
                        .expect("Could not send net")
                    }
                }
            }

            if self.paused {
                sleep(Duration::from_millis(100));
                continue;
            }

            self.sim.run();
            self.sim.ticks += 1;
            if self.tx.len() == 0 {
                _ = self.tx.send(RunnerRes::Positions(self.positions()));
            }
        }
    }

    fn positions(&self) -> Positions {
        Positions {
            creatures: self
                .sim
                .creatures
                .par_iter()
                .map(|x| BasicCreature {
                    position: x.value().position,
                    id: *x.key(),
                    size: creature_size(x.value().energy),
                })
                .collect(),
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
        }
    }
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

fn creature_size(energy: f32) -> f32 {
    MIN_CREATURE_SIZE
        + (energy / MAX_ENERGY).clamp(0.0, 1.0) * (MAX_CREATURE_SIZE - MIN_CREATURE_SIZE)
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn set_input(inputs: &mut [engine::nn::GraphNode], idx: usize, value: f32) {
    if let Some(node) = inputs.get_mut(idx) {
        if let Node::Input(input) = &mut node.value {
            input.set_value(value);
        }
    }
}

fn nearest_signal(from: (f32, f32), targets: &[(f32, f32)], dims: (f32, f32)) -> (f32, f32, f32) {
    let Some(target) = targets
        .iter()
        .min_by(|a, b| distance(from, **a).total_cmp(&distance(from, **b)))
    else {
        return (0.0, 0.0, 1.0);
    };

    let dx = target.0 - from.0;
    let dy = target.1 - from.1;
    let diagonal = (dims.0.powi(2) + dims.1.powi(2)).sqrt().max(1.0);
    (
        dx / dims.0.max(1.0),
        dy / dims.1.max(1.0),
        distance(from, *target) / diagonal,
    )
}

fn speed_cap_for_energy(energy: f32) -> f32 {
    ((energy - MIN_MOVE_ENERGY) / (FULL_SPEED_ENERGY - MIN_MOVE_ENERGY)).clamp(0.0, 1.0)
}

impl Simulation {
    fn run(&mut self) {
        let food_snapshot = self.food.clone();
        let poison_snapshot = self.poison.clone();
        let mate_snapshot = self
            .creatures
            .iter()
            .filter_map(|c| {
                let creature = c.value();
                let output_layer =
                    &creature.brain.graph.layers[creature.brain.graph.layers.len() - 1];
                (creature.mate_cooldown == 0
                    && creature.energy >= MIN_MATE_ENERGY
                    && get_output_value(&output_layer[5].value) > MATE_THRESHOLD)
                    .then_some((*c.key(), creature.position))
            })
            .collect::<Vec<_>>();
        let dims = self.world_dim;

        self.creatures.par_iter_mut().for_each(|mut accessor| {
            let id = *accessor.key();
            let c = accessor.value_mut();
            c.mate_cooldown = c.mate_cooldown.saturating_sub(1);
            c.action_lock = c.action_lock.saturating_sub(1);
            c.age += 1;
            let speed = c.brain.graph.layers[c.brain.graph.layers.len() - 1][4]
                .value
                .clone();
            let inputs = &mut c.brain.graph.layers[c.brain.input_layer as usize];
            let food_positions = food_snapshot
                .iter()
                .map(|food| food.position)
                .collect::<Vec<_>>();
            let (food_dx, food_dy, food_dist) = nearest_signal(c.position, &food_positions, dims);
            let mate_targets = mate_snapshot
                .iter()
                .filter_map(|(mate_id, position)| (*mate_id != id).then_some(*position))
                .collect::<Vec<_>>();
            let (mate_dx, mate_dy, mate_dist) = nearest_signal(c.position, &mate_targets, dims);
            let (poison_dx, poison_dy, poison_dist) =
                nearest_signal(c.position, &poison_snapshot, dims);

            set_input(inputs, 0, 1.0 - (c.energy / MAX_ENERGY).clamp(0.0, 1.0));
            set_input(inputs, 1, c.age as f32 / MAX_AGE as f32);
            set_input(inputs, 2, (c.energy / MAX_ENERGY).clamp(0.0, 1.0));
            set_input(
                inputs,
                3,
                if let Node::Output(o) = speed {
                    o.value()
                } else {
                    0.0
                },
            );
            set_input(inputs, 4, food_dx);
            set_input(inputs, 5, food_dy);
            set_input(inputs, 6, food_dist);
            set_input(inputs, 7, mate_dx);
            set_input(inputs, 8, mate_dy);
            set_input(inputs, 9, mate_dist);
            set_input(
                inputs,
                10,
                1.0 - (c.position.0 / dims.0.max(1.0)).clamp(0.0, 1.0),
            );
            set_input(inputs, 11, (c.position.0 / dims.0.max(1.0)).clamp(0.0, 1.0));
            set_input(
                inputs,
                12,
                1.0 - (c.position.1 / dims.1.max(1.0)).clamp(0.0, 1.0),
            );
            set_input(inputs, 13, (c.position.1 / dims.1.max(1.0)).clamp(0.0, 1.0));
            set_input(inputs, 14, poison_dx);
            set_input(inputs, 15, poison_dy);
            set_input(inputs, 16, poison_dist);

            c.brain.tick();

            let output_layer = &c.brain.graph.layers[c.brain.graph.layers.len() - 1];
            let forward = get_output_value(&output_layer[0].value);
            let backward = get_output_value(&output_layer[1].value);
            let left = get_output_value(&output_layer[2].value);
            let right = get_output_value(&output_layer[3].value);

            let movement_vec = dirs_to_vec(forward, backward, left, right);
            let speed = if c.action_lock == 0 {
                get_output_value(&output_layer[4].value).min(speed_cap_for_energy(c.energy))
            } else {
                0.0
            };
            let mate = get_output_value(&output_layer[5].value);
            let eat = get_output_value(&output_layer[6].value);

            let t = movement_vec * speed; // add time diff here if needed
            c.energy -= BASE_ENERGY_COST
                + (speed * MOVE_ENERGY_COST)
                + (mate * MATE_ATTEMPT_COST)
                + (eat * EAT_ATTEMPT_COST);

            let next_position = (c.position.0 + t.x, c.position.1 + t.y);
            if next_position.0 >= 0.0
                && next_position.0 <= self.world_dim.0
                && next_position.1 >= 0.0
                && next_position.1 <= self.world_dim.1
            {
                c.position = next_position;
            } else {
                c.position.0 = c.position.0.clamp(0.0, self.world_dim.0);
                c.position.1 = c.position.1.clamp(0.0, self.world_dim.1);
            }
        });

        self.mate_creatures();

        self.creatures
            .retain(|_, creature| creature.energy > 0.0 && creature.age <= MAX_AGE);

        self.food.retain(|f| {
            let Some(mut accessor) = self.creatures.iter_mut().find(|x| {
                do_sized_squares_collide(x.position, creature_size(x.energy), f.position, f.size)
            }) else {
                return true;
            };

            let c_meet = accessor.value();
            let output_layer = &c_meet.brain.graph.layers[c_meet.brain.graph.layers.len() - 1];
            if get_output_value(&output_layer[6].value) <= 0.5 {
                return true;
            }
            if c_meet.action_lock != 0 || creature_size(c_meet.energy) < f.size {
                return true;
            }

            let c_meet = accessor.value_mut();
            c_meet.energy -= EAT_ACTION_COST;
            c_meet.action_lock = EAT_ACTION_BASE_TICKS + (f.size * FOOD_EAT_TICKS_PER_SIZE) as u16;
            if let Node::Input(n) = &mut c_meet.brain.graph.layers[0][0].value {
                let v = n.as_standard() - 1.0;
                n.set_value(if v < 0.0 { 0.0 } else { v });
            }

            c_meet.energy = (c_meet.energy + f.size * FOOD_ENERGY_PER_SIZE).min(MAX_ENERGY);

            if let Node::Input(n) = &mut c_meet.brain.graph.layers[0][3].value {
                n.set_value(0.0);
            }

            false
        });

        self.poison.retain(|p| {
            let Some(mut accessor) = self
                .creatures
                .iter_mut()
                .find(|x| do_squares_collide(x.position, *p))
            else {
                return true;
            };

            accessor.value_mut().energy -= POISON_DAMAGE;
            false
        });

        let mut rng = rand::thread_rng();
        self.refill_food_and_poison(&mut rng);
        while self.creatures.len() < self.target_population {
            self.last_id += 1;
            let creature = self.random_creature(&mut rng, self.world_dim);
            self.creatures.insert(self.last_id, creature);
        }
    }

    fn mate_creatures(&mut self) {
        if self.creatures.len() >= self.target_population * 2 {
            return;
        }

        let candidates = self
            .creatures
            .iter()
            .filter_map(|c| {
                let creature = c.value();
                if creature.action_lock != 0 {
                    return None;
                }
                if creature.mate_cooldown != 0 {
                    return None;
                }
                if creature.energy < MIN_MATE_ENERGY {
                    return None;
                }

                let output_layer =
                    &creature.brain.graph.layers[creature.brain.graph.layers.len() - 1];
                if get_output_value(&output_layer[5].value) <= MATE_THRESHOLD {
                    return None;
                }

                Some((*c.key(), creature.position))
            })
            .collect::<Vec<_>>();

        let mut used_parents = Vec::new();
        let mut births = Vec::new();
        for (idx, (a_id, a_position)) in candidates.iter().enumerate() {
            if used_parents.contains(a_id) {
                continue;
            }

            let Some((b_id, b_position)) =
                candidates.iter().skip(idx + 1).find(|(b_id, b_position)| {
                    !used_parents.contains(b_id) && do_squares_collide(*a_position, *b_position)
                })
            else {
                continue;
            };

            let Some(a) = self.creatures.get(a_id) else {
                continue;
            };
            let Some(b) = self.creatures.get(b_id) else {
                continue;
            };
            if a.energy < MIN_MATE_ENERGY || b.energy < MIN_MATE_ENERGY {
                continue;
            }
            if a.action_lock != 0 || b.action_lock != 0 {
                continue;
            }

            let mut child_brain = Net::reproduce(
                &a.brain,
                &b.brain,
                &[Crossover::default()],
                DefaultIterator::new(),
            )
            .unwrap_or_else(|_| a.brain.clone());
            Self::mutate_child_brain(&mut child_brain);

            births.push((
                *a_id,
                *b_id,
                Creature {
                    brain: child_brain,
                    position: (
                        ((a_position.0 + b_position.0) / 2.0).clamp(0.0, self.world_dim.0),
                        ((a_position.1 + b_position.1) / 2.0).clamp(0.0, self.world_dim.1),
                    ),
                    energy: CHILD_ENERGY,
                    age: 0,
                    mate_cooldown: MATE_COOLDOWN_TICKS,
                    action_lock: MATE_ACTION_TICKS,
                },
            ));
            used_parents.push(*a_id);
            used_parents.push(*b_id);

            if births.len() >= MAX_BIRTHS_PER_TICK {
                break;
            }
            if self.creatures.len() + births.len() >= self.target_population * 2 {
                break;
            }
        }

        for (a_id, b_id, child) in births {
            if let Some(mut a) = self.creatures.get_mut(&a_id) {
                a.energy -= MATE_ENERGY_COST;
                a.mate_cooldown = MATE_COOLDOWN_TICKS;
                a.action_lock = MATE_ACTION_TICKS;
            }
            if let Some(mut b) = self.creatures.get_mut(&b_id) {
                b.energy -= MATE_ENERGY_COST;
                b.mate_cooldown = MATE_COOLDOWN_TICKS;
                b.action_lock = MATE_ACTION_TICKS;
            }

            self.last_id += 1;
            self.creatures.insert(self.last_id, child);
        }
    }

    fn random_creature(&self, rng: &mut impl Rng, dims: (f32, f32)) -> Creature {
        Creature {
            brain: Net::gen(&self.input_nodes, &self.output_nodes).unwrap(),
            position: (rng.gen_range(0.0..dims.0), rng.gen_range(0.0..dims.1)),
            energy: START_ENERGY,
            age: 0,
            mate_cooldown: 0,
            action_lock: 0,
        }
    }

    fn refill_food_and_poison(&mut self, rng: &mut impl Rng) {
        let food_target = (self.target_population / FOOD_PER_CREATURE).max(1);
        let poison_target = (self.target_population / POISON_PER_CREATURE).max(1);
        while self.food.len() < food_target {
            self.food.push(Food {
                position: (
                    rng.gen_range(0.0..self.world_dim.0),
                    rng.gen_range(0.0..self.world_dim.1),
                ),
                size: rng.gen_range(MIN_FOOD_SIZE..=MAX_FOOD_SIZE),
            });
        }
        while self.poison.len() < poison_target {
            self.poison.push((
                rng.gen_range(0.0..self.world_dim.0),
                rng.gen_range(0.0..self.world_dim.1),
            ));
        }
    }

    fn mutate_child_brain(net: &mut Net) {
        let mut rng = rand::thread_rng();
        for layer in &mut net.graph.layers {
            for node in layer {
                for edge in &mut node.connections {
                    if rng.gen::<f32>() < MUTATION_RATE {
                        edge.value.weight += rng.gen_range(-MUTATION_AMOUNT..MUTATION_AMOUNT);
                    }
                    if rng.gen::<f32>() < MUTATION_RATE / 4.0 {
                        edge.value.enabled = !edge.value.enabled;
                    }
                }
            }
        }
    }
}
