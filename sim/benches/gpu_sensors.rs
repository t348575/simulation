use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use engine::{
    activations::Sigmoid,
    nn::{BasicNeuron, Edge, GraphLocation, GraphNode, Net, NeuralGraph, Node},
};
use rand::{rngs::StdRng, Rng, RngExt, SeedableRng};
use sim::{
    gpu::{
        GpuBrainCompute, GpuCreatureSensor, GpuFoodSensor, GpuObstacleSensor, GpuPoisonSensor,
        GpuSensorWorld, GPU_INPUTS, GPU_OUTPUTS,
    },
    inputs::BlankInput,
};

const GRID_CELL: f32 = 32.0;

struct BenchWorld {
    ids: Vec<usize>,
    creatures: Vec<GpuCreatureSensor>,
    foods: Vec<GpuFoodSensor>,
    poisons: Vec<GpuPoisonSensor>,
    obstacles: Vec<GpuObstacleSensor>,
    dims: (f32, f32),
    half_fov: f32,
    vision: f32,
    max_energy: f32,
    min_creature_size: f32,
    max_creature_size: f32,
    max_age: f32,
}

fn gpu_sensor_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_sensor_pipeline");
    group.sample_size(10);

    // Scaling sweep: spans the regime where each stage's cost dominates.
    let sizes: &[usize] = &[500, 1_000, 2_000, 5_000, 10_000];

    for &creature_count in sizes {
        let world = make_world(creature_count);

        // Baseline: CPU sensor + CPU brain forward pass. Lets us compute the
        // full speed-up of GPU vs single-machine CPU.
        group.bench_with_input(
            BenchmarkId::new("cpu_inputs_then_cpu_brain", creature_count),
            &creature_count,
            |b, _| {
                let mut nets: Vec<Net> = world.ids.iter().map(|&id| make_brain(id)).collect();
                b.iter(|| {
                    let _inputs = cpu_generate_inputs(&world);
                    for net in nets.iter_mut() {
                        net.tick();
                    }
                    black_box(&nets);
                });
            },
        );

        // Pipelined: submits this tick's work and drains the previous tick's
        // result inside the same `enqueue` call. CPU "stub work" runs after
        // submit, in parallel with the GPU. This is the harness that mirrors
        // the live simulator's tick loop. The first iter pays a cold-start
        // cost (no prior tick to drain); criterion's warm-up smooths it.
        group.bench_with_input(
            BenchmarkId::new("gpu_pipelined_full_tick", creature_count),
            &creature_count,
            |b, _| {
                let mut gpu = make_gpu(&world.ids);
                // Prime the pipeline so the first measured iter has prev work to drain.
                gpu.enqueue_with_sensors(
                    &world.ids,
                    GpuSensorWorld {
                        creatures: &world.creatures,
                        foods: &world.foods,
                        poisons: &world.poisons,
                        obstacles: &world.obstacles,
                        origin: (0.0, 0.0),
                        dims: world.dims,
                        cell: GRID_CELL,
                        half_fov: world.half_fov,
                        vision: world.vision,
                        max_energy: world.max_energy,
                        min_creature_size: world.min_creature_size,
                        max_creature_size: world.max_creature_size,
                        max_age: world.max_age,
                    },
                );
                b.iter(|| {
                    let prev = gpu.take_pending_outputs();
                    gpu.enqueue_with_sensors(
                        &world.ids,
                        GpuSensorWorld {
                            creatures: &world.creatures,
                            foods: &world.foods,
                            poisons: &world.poisons,
                            obstacles: &world.obstacles,
                            origin: (0.0, 0.0),
                            dims: world.dims,
                            cell: GRID_CELL,
                            half_fov: world.half_fov,
                            vision: world.vision,
                            max_energy: world.max_energy,
                            min_creature_size: world.min_creature_size,
                            max_creature_size: world.max_creature_size,
                            max_age: world.max_age,
                        },
                    );
                    // Stub CPU work: in production this overlaps with the GPU
                    // running the tick we just submitted. We simply touch the
                    // prev outputs so the optimizer can't elide them.
                    if let Some((_ids, outs)) = prev {
                        black_box(outs.first().copied());
                    }
                });
                // Final drain to leave GPU clean.
                let _ = gpu.take_pending_outputs();
            },
        );
    }

    group.finish();
}

fn make_gpu(ids: &[usize]) -> GpuBrainCompute {
    let mut gpu = GpuBrainCompute::new().expect("GPU benchmark requires a wgpu adapter");
    for &id in ids {
        gpu.assign_brain(id, &make_brain(id))
            .expect("benchmark brain must compile for GPU");
    }
    gpu
}

fn make_brain(seed: usize) -> Net {
    let mut graph = NeuralGraph::new();
    let input_layer = graph.add_layer_to_end();
    for id in 0..GPU_INPUTS {
        graph
            .add_node(
                input_layer,
                GraphNode::new(Node::Input(BlankInput::new(0.0, id, "bench"))),
            )
            .unwrap();
    }

    let hidden_layer = graph.add_layer_to_end();
    for h in 0..8u16 {
        graph
            .add_node(
                hidden_layer,
                GraphNode::new(Node::Neuron(Box::new(BasicNeuron::new(
                    ((seed + h as usize) as f32 * 0.001).sin() * 0.1,
                    1_000 + h as usize,
                )))),
            )
            .unwrap();
    }

    let output_layer = graph.add_layer_to_end();
    for id in 0..GPU_OUTPUTS {
        graph
            .add_node(
                output_layer,
                GraphNode::new(Node::Output(Sigmoid::new(0.0, id, "out".to_string()))),
            )
            .unwrap();
    }

    for input in 0..GPU_INPUTS as u16 {
        for hidden in 0..8u16 {
            if (input + hidden + seed as u16) % 3 == 0 {
                add_edge(&mut graph, input_layer, input, hidden_layer, hidden, seed);
            }
        }
    }
    for hidden in 0..8u16 {
        for output in 0..GPU_OUTPUTS as u16 {
            add_edge(
                &mut graph,
                hidden_layer,
                hidden,
                output_layer,
                output,
                seed + 17,
            );
        }
    }

    Net {
        graph,
        input_layer,
        output_layer,
    }
}

fn add_edge(
    graph: &mut NeuralGraph,
    from_layer: u16,
    from_node: u16,
    to_layer: u16,
    to_node: u16,
    seed: usize,
) {
    let weight = (((from_node as usize * 31 + to_node as usize * 17 + seed) as f32) * 0.01).sin();
    graph
        .add_edge(
            GraphLocation::new(from_layer, from_node),
            GraphLocation::new(to_layer, to_node),
            Edge {
                weight,
                enabled: true,
            },
        )
        .unwrap();
}

fn make_world(creature_count: usize) -> BenchWorld {
    let mut rng = StdRng::seed_from_u64(42);
    let dims = (2_000.0, 2_000.0);
    let max_energy = 140.0;
    let min_creature_size = 3.0;
    let max_creature_size = 14.0;

    let ids = (1..=creature_count).collect();
    let creatures = (0..creature_count)
        .map(|_| GpuCreatureSensor {
            pos_x: rng.random_range(0.0..dims.0),
            pos_y: rng.random_range(0.0..dims.1),
            angle: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            energy: rng.random_range(1.0..max_energy),
            prev_energy: rng.random_range(1.0..max_energy),
            touched: rng.random_range(0.0..1.0),
            mem: rng.random_range(-1.0..1.0),
            age: rng.random_range(1.0..60_000.0),
            max_energy,
            body_size: 8.0,
            max_age: 60_000.0,
            half_fov: std::f32::consts::PI * 0.75,
            vision: 250.0,
            prev_thrust: rng.random_range(0.0..1.0),
            prev_turn_left: rng.random_range(0.0..1.0),
            prev_turn_right: rng.random_range(0.0..1.0),
            terrain_elevation: 0.5,
            terrain_speed: 1.0,
            terrain_energy_cost: 1.0,
            terrain_hazard: 0.0,
            terrain_cost_s0: 0.0,
            terrain_cost_s1: 0.0,
            terrain_cost_s2: 0.0,
            terrain_cost_s3: 0.0,
            terrain_haz_s0: 0.0,
            terrain_haz_s1: 0.0,
            terrain_haz_s2: 0.0,
            terrain_haz_s3: 0.0,
            signal: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        })
        .collect();
    let foods = (0..creature_count * 2)
        .map(|_| GpuFoodSensor {
            x: rng.random_range(0.0..dims.0),
            y: rng.random_range(0.0..dims.1),
            size: rng.random_range(2.0..10.0),
            energy: rng.random_range(20.0..100.0),
            hue: rng.random_range(0.0..1.0),
            odor: rng.random_range(0.0..1.0),
            _pad0: 0.0,
            _pad1: 0.0,
        })
        .collect();
    let poisons = (0..creature_count * 12)
        .map(|_| GpuPoisonSensor {
            x: rng.random_range(0.0..dims.0),
            y: rng.random_range(0.0..dims.1),
            damage: rng.random_range(10.0..80.0),
            hue: rng.random_range(0.0..1.0),
            odor: rng.random_range(0.0..1.0),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        })
        .collect();
    let obstacles = (0..30)
        .map(|_| GpuObstacleSensor {
            x: rng.random_range(0.0..dims.0),
            y: rng.random_range(0.0..dims.1),
            half_w: rng.random_range(15.0..60.0),
            half_h: rng.random_range(15.0..60.0),
        })
        .collect();

    BenchWorld {
        ids,
        creatures,
        foods,
        poisons,
        obstacles,
        dims,
        half_fov: std::f32::consts::PI * 0.75,
        vision: 250.0,
        max_energy,
        min_creature_size,
        max_creature_size,
        max_age: 60_000.0,
    }
}

fn cpu_generate_inputs(world: &BenchWorld) -> Vec<f32> {
    let food_grid = CpuGrid::build(
        world
            .foods
            .iter()
            .enumerate()
            .map(|(idx, f)| (idx, f.x, f.y)),
        world.dims,
        GRID_CELL,
    );
    let poison_grid = CpuGrid::build(
        world
            .poisons
            .iter()
            .enumerate()
            .map(|(idx, p)| (idx, p.x, p.y)),
        world.dims,
        GRID_CELL,
    );
    let creature_grid = CpuGrid::build(
        world
            .creatures
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, c.pos_x, c.pos_y)),
        world.dims,
        GRID_CELL,
    );

    let mut out = vec![0.0; world.creatures.len() * GPU_INPUTS];
    for (idx, creature) in world.creatures.iter().enumerate() {
        let food = scan_grid_cpu(creature, &food_grid, world.half_fov, world.vision, |_| true);
        let poison = scan_grid_cpu(creature, &poison_grid, world.half_fov, world.vision, |_| {
            true
        });
        let obstacle = scan_cpu(
            creature,
            world.half_fov,
            world.vision,
            world.obstacles.iter().map(|o| (o.x, o.y)),
        );
        let creature_scan = scan_grid_cpu(
            creature,
            &creature_grid,
            world.half_fov,
            world.vision,
            |other_idx| other_idx != idx,
        );

        let base = idx * GPU_INPUTS;
        let max_e = world.max_energy.max(1.0);
        let size = world.min_creature_size
            + (creature.energy / max_e).clamp(0.0, 1.0)
                * (world.max_creature_size - world.min_creature_size);
        let size_n = ((size - world.min_creature_size)
            / (world.max_creature_size - world.min_creature_size).max(0.001))
        .clamp(0.0, 1.0);
        out[base] = 1.0 - (creature.energy / max_e).clamp(0.0, 1.0);
        out[base + 1] = (creature.energy / max_e).clamp(0.0, 1.0);
        out[base + 2] = creature.prev_thrust;
        out[base + 3] = creature.prev_turn_left - creature.prev_turn_right;
        out[base + 4] = creature.angle.cos();
        out[base + 5] = creature.angle.sin();
        out[base + 6] = creature.touched;
        out[base + 7] = ((creature.prev_energy - creature.energy).max(0.0) / max_e).clamp(0.0, 1.0);
        out[base + 8] = (creature.age * 0.1).sin();
        out[base + 9] = creature.mem;
        out[base + 10] = (creature.age / world.max_age).clamp(0.0, 1.0);
        out[base + 11] = size_n;
        out[base + 12] = 1.0 - (creature.pos_x / world.dims.0.max(1.0)).clamp(0.0, 1.0);
        out[base + 13] = (creature.pos_x / world.dims.0.max(1.0)).clamp(0.0, 1.0);
        out[base + 14] = 1.0 - (creature.pos_y / world.dims.1.max(1.0)).clamp(0.0, 1.0);
        out[base + 15] = (creature.pos_y / world.dims.1.max(1.0)).clamp(0.0, 1.0);
        out[base + 16..base + 20].copy_from_slice(&food);
        out[base + 20..base + 24].copy_from_slice(&poison);
        out[base + 24..base + 28].copy_from_slice(&obstacle);
        out[base + 28..base + 32].copy_from_slice(&creature_scan);
        out[base + 32] = creature.terrain_elevation;
        out[base + 33] = creature.terrain_speed;
        out[base + 34] = ((creature.terrain_energy_cost - 1.0) / 1.5).clamp(0.0, 1.0);
        out[base + 35] = (creature.terrain_hazard / 0.05).clamp(0.0, 1.0);
    }
    out
}

struct CpuGrid {
    cols: usize,
    rows: usize,
    cell: f32,
    buckets: Vec<Vec<usize>>,
    positions: Vec<(f32, f32)>,
}

impl CpuGrid {
    fn build(
        entries: impl Iterator<Item = (usize, f32, f32)>,
        dims: (f32, f32),
        cell: f32,
    ) -> Self {
        let cell = cell.max(1.0);
        let cols = ((dims.0 / cell).ceil() as usize).max(1);
        let rows = ((dims.1 / cell).ceil() as usize).max(1);
        let mut buckets = vec![Vec::new(); cols * rows];
        let mut positions = Vec::new();
        for (idx, x, y) in entries {
            if idx >= positions.len() {
                positions.resize(idx + 1, (0.0, 0.0));
            }
            positions[idx] = (x, y);
            let cell_idx = Self::cell_index_for(cols, rows, cell, x, y);
            buckets[cell_idx].push(idx);
        }
        Self {
            cols,
            rows,
            cell,
            buckets,
            positions,
        }
    }

    fn cell_index_for(cols: usize, rows: usize, cell: f32, x: f32, y: f32) -> usize {
        let cx = ((x / cell) as isize).clamp(0, cols as isize - 1) as usize;
        let cy = ((y / cell) as isize).clamp(0, rows as isize - 1) as usize;
        cy * cols + cx
    }

    fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let cx = ((x / self.cell) as isize).clamp(0, self.cols as isize - 1) as usize;
        let cy = ((y / self.cell) as isize).clamp(0, self.rows as isize - 1) as usize;
        (cx, cy)
    }
}

fn scan_grid_cpu(
    creature: &GpuCreatureSensor,
    grid: &CpuGrid,
    half_fov: f32,
    vision: f32,
    mut keep: impl FnMut(usize) -> bool,
) -> [f32; 4] {
    let mut out = [0.0; 4];
    let (min_x, min_y) = grid.cell_of(creature.pos_x - vision, creature.pos_y - vision);
    let (max_x, max_y) = grid.cell_of(creature.pos_x + vision, creature.pos_y + vision);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            for &idx in &grid.buckets[y * grid.cols + x] {
                if keep(idx) {
                    let (target_x, target_y) = grid.positions[idx];
                    update_cpu_sector(&mut out, creature, half_fov, vision, target_x, target_y);
                }
            }
        }
    }
    out
}

fn scan_cpu(
    creature: &GpuCreatureSensor,
    half_fov: f32,
    vision: f32,
    targets: impl Iterator<Item = (f32, f32)>,
) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (x, y) in targets {
        update_cpu_sector(&mut out, creature, half_fov, vision, x, y);
    }
    out
}

fn update_cpu_sector(
    out: &mut [f32; 4],
    creature: &GpuCreatureSensor,
    half_fov: f32,
    vision: f32,
    x: f32,
    y: f32,
) {
    let facing_x = creature.angle.cos();
    let facing_y = creature.angle.sin();
    let cos_half = half_fov.cos();
    let full_circle = half_fov >= std::f32::consts::PI;
    let vision2 = vision * vision;
    let dx = x - creature.pos_x;
    let dy = y - creature.pos_y;
    let d2 = dx * dx + dy * dy;
    if d2 > vision2 || d2 < 0.000001 {
        return;
    }
    let d = d2.sqrt();
    let dot = (dx * facing_x + dy * facing_y) / d;
    if !full_circle && dot < cos_half {
        return;
    }
    let cross = facing_x * dy - facing_y * dx;
    let rel = cross.atan2((dx * facing_x + dy * facing_y).max(-1_000_000_000.0));
    let span = (half_fov * 2.0).max(0.001);
    let t = ((rel + half_fov) / span).clamp(0.0, 0.9999);
    let sector = (t * 4.0) as usize;
    let closeness = (1.0 - d / vision).clamp(0.0, 1.0);
    if closeness > out[sector] {
        out[sector] = closeness;
    }
}

criterion_group!(benches, gpu_sensor_bench);
criterion_main!(benches);
