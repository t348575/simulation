use std::{thread::sleep, time::Duration};

use dashmap::DashMap;
use engine::nn::{Net, Node};
use flume::{unbounded, Receiver, Sender};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use bevy::math::Vec2;

pub const CREATURE_DIM: f32 = 5.0;
pub const CREATURE_DIM_HALF: f32 = CREATURE_DIM / 2.0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Simulation {
    world_dim: (f32, f32),
    creatures: DashMap<usize, Creature>,
    food: Vec<(f32, f32)>,
    last_id: usize,
    ticks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Creature {
    brain: Net,
    position: (f32, f32),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicCreature {
    pub position: (f32, f32),
    pub id: usize,
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
    GetPositions,
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
    pub food: Vec<(f32, f32)>,
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
            if self.rx.len() > 0 {
                self.rx.drain().for_each(|msg| match msg {
                    RunnerReq::Generate(g) => {
                        let mut rng = rand::thread_rng();
                        let width = g.dims.0;
                        let height = g.dims.1;
                        self.sim.creatures = (0..g.num_creatures)
                            .map(|_| {
                                self.sim.last_id += 1;
                                (
                                    self.sim.last_id,
                                    Creature {
                                        brain: Net::gen(&g.input_nodes, &g.output_nodes).unwrap(),
                                        position: (
                                            rng.gen_range(0.0..width),
                                            rng.gen_range(0.0..height),
                                        ),
                                    },
                                )
                            })
                            .collect();
                        self.sim.food = (0..(g.num_creatures / 4))
                            .map(|_| (rng.gen_range(0.0..width), rng.gen_range(0.0..height)))
                            .collect();
                        self.sim.world_dim = g.dims;
                    }
                    RunnerReq::Resume => self.paused = false,
                    RunnerReq::Pause => self.paused = true,
                    RunnerReq::GetPositions => self
                        .tx
                        .send(RunnerRes::Positions(Positions {
                            creatures: self
                                .sim
                                .creatures
                                .par_iter()
                                .map(|x| BasicCreature {
                                    position: x.value().position,
                                    id: *x.key(),
                                })
                                .collect(),
                            food: self.sim.food.clone(),
                        }))
                        .expect("Could not send positions"),
                    RunnerReq::GetNet(id) => {
                        match self.sim.creatures.iter().find(|c| *c.key() == id) {
                            Some(c) => self.tx.send(RunnerRes::Net(Some(c.brain.clone()))),
                            None => self.tx.send(RunnerRes::Net(None)),
                        }
                        .expect("Could not send net")
                    }
                });
            }

            if self.paused {
                sleep(Duration::from_millis(100));
                continue;
            }

            self.sim.run();
            self.sim.ticks += 1;
        }
    }
}

fn dirs_to_vec(forward: f32, backward: f32, left: f32, right: f32) -> Vec2 {
    let x = Vec2::new(right, 0.0) + Vec2::new(left * -1.0, 0.0);
    let y = Vec2::new(0.0, forward) + Vec2::new(0.0, backward * -1.0);
    x + y
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

impl Simulation {
    fn run(&mut self) {
        self.creatures.par_iter_mut().for_each(|mut accessor| {
            let c = accessor.value_mut();
            let speed = c.brain.graph.layers[c.brain.graph.layers.len() - 1][4]
                .value
                .clone();
            let inputs = &mut c.brain.graph.layers[c.brain.input_layer as usize];
            if let Node::Input(i) = &mut inputs[0].value {
                i.set_value(i.as_standard() + 0.01); // hunger
            }

            if let Node::Input(i) = &mut inputs[0].value {
                i.set_value(i.as_standard() + 1.0); // age
            }

            if let Node::Input(i) = &mut inputs[0].value {
                i.set_value(i.as_standard() - 0.00001); // health
            }

            if let Node::Input(i) = &mut inputs[0].value {
                if let Node::Output(o) = speed {
                    i.set_value(o.value()); // speed
                }
            }

            c.brain.tick();

            let output_layer = &c.brain.graph.layers[c.brain.graph.layers.len() - 1];
            let forward = get_output_value(&output_layer[0].value);
            let backward = get_output_value(&output_layer[1].value);
            let left = get_output_value(&output_layer[2].value);
            let right = get_output_value(&output_layer[3].value);

            let movement_vec = dirs_to_vec(forward, backward, left, right);
            let speed = get_output_value(&output_layer[4].value);

            let t = movement_vec * speed; // add time diff here if needed

            c.position.0 += t.x;
            c.position.1 += t.y;

            // wall collisions
            if c.position.0 <= 0.0 {
                c.position.0 = 0.0;
            } else if c.position.0 >= self.world_dim.0 {
                c.position.0 = self.world_dim.0;
            }

            if c.position.1 <= 0.0 {
                c.position.1 = 0.0;
            } else if c.position.1 >= self.world_dim.1 {
                c.position.1 = self.world_dim.1;
            }
        });

        self.creatures.par_iter().for_each(|c| {
            let output_layer = &c.brain.graph.layers[c.brain.graph.layers.len() - 1];
            if let Some(c_meet) = self
                .creatures
                .par_iter()
                .find_first(|x| do_squares_collide(x.position, c.position))
            {
                if get_output_value(&output_layer[5].value) > 0.5
                    && get_output_value(
                        &c_meet.brain.graph.layers[c_meet.brain.graph.layers.len() - 1][5].value,
                    ) > 0.5
                {
                    // TODO: mate
                }
            }
        });

        self.food = self
            .food
            .par_iter()
            .map(|f| {
                if let Some(mut accessor) = self
                    .creatures
                    .par_iter_mut()
                    .find_first(|x| do_squares_collide(x.position, *f))
                {
                    let c_meet = accessor.value();
                    let output_layer =
                        &c_meet.brain.graph.layers[c_meet.brain.graph.layers.len() - 1];
                    if get_output_value(&output_layer[6].value) > 0.5 {
                        let c_meet = accessor.value_mut();
                        if let Node::Input(n) = &mut c_meet.brain.graph.layers[0][0].value {
                            let v = n.as_standard() - 1.0;
                            n.set_value(if v < 0.0 { 0.0 } else { v });
                        }

                        if let Node::Input(n) = &mut c_meet.brain.graph.layers[0][3].value {
                            n.set_value(0.0);
                        }
                        return Err(());
                    }
                }
                return Ok(*f);
            })
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();
    }
}
