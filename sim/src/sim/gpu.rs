use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine::nn::{Net, Node};

pub const GPU_INPUTS: usize = 52;
pub const GPU_OUTPUTS: usize = 10;
pub const MAX_NODES_PER_BRAIN: usize = 256;
const BRAIN_WORKGROUP_SIZE: u32 = 128;
const SENSOR_WORKGROUP_SIZE: u32 = 128;
const SENSOR_GRID_BUCKET_CAPACITY: u32 = 64;
const MAX_STORAGE_BINDING_BYTES: usize = 128 * 1024 * 1024;
// Upper bound on sensor grid cell count. Buffer size scales as
// cells * 3 (kinds) * SENSOR_GRID_BUCKET_CAPACITY (slots) * 4 (u32 bytes),
// and must fit both the bind binding limit (128MB) and the device max buffer
// limit (typically 256MB). 131_072 cells → 96MB grid_items buffer, with
// generous headroom on both. Beyond this we widen the per-cell size so the
// world stays scannable without exceeding GPU buffer caps.
const MAX_SENSOR_GRID_CELLS: u32 = 131_072;

const INITIAL_BRAIN_CAPACITY: u32 = 16_384;
const INITIAL_NODE_CAPACITY: u32 = INITIAL_BRAIN_CAPACITY * 16;
const INITIAL_EDGE_CAPACITY: u32 = INITIAL_BRAIN_CAPACITY * 32;
const INITIAL_LAYER_CAPACITY: u32 = INITIAL_BRAIN_CAPACITY * 4;
// Initial per-creature value scratch slots. Per-creature actual usage is the
// owning brain's node_count; this is just the starting allocation across all
// dispatch slots. Grown on demand.
const INITIAL_VALUE_SLOTS_PER_CREATURE: u32 = 64;

const ACTIVATION_INPUT: u32 = 0;
const ACTIVATION_LINEAR: u32 = 1;
const ACTIVATION_OUTPUT_PARTIAL: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    creature_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuSensorParams {
    creature_count: u32,
    food_count: u32,
    poison_count: u32,
    obstacle_count: u32,
    cols: u32,
    rows: u32,
    bucket_capacity: u32,
    max_entity_count: u32,
    world_w: f32,
    world_h: f32,
    cell: f32,
    half_fov: f32,
    vision: f32,
    max_energy: f32,
    min_creature_size: f32,
    max_creature_size: f32,
    max_age: f32,
    origin_x: f32,
    origin_y: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuCreatureSensor {
    pub pos_x: f32,
    pub pos_y: f32,
    pub angle: f32,
    pub energy: f32,
    pub prev_energy: f32,
    pub touched: f32,
    pub mem: f32,
    pub age: f32,
    pub max_energy: f32,
    pub body_size: f32,
    pub max_age: f32,
    pub half_fov: f32,
    pub vision: f32,
    pub prev_thrust: f32,
    pub prev_turn_left: f32,
    pub prev_turn_right: f32,
    pub terrain_elevation: f32,
    pub terrain_speed: f32,
    pub terrain_energy_cost: f32,
    pub terrain_hazard: f32,
    pub terrain_cost_s0: f32,
    pub terrain_cost_s1: f32,
    pub terrain_cost_s2: f32,
    pub terrain_cost_s3: f32,
    pub terrain_haz_s0: f32,
    pub terrain_haz_s1: f32,
    pub terrain_haz_s2: f32,
    pub terrain_haz_s3: f32,
    pub signal: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuFoodSensor {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub energy: f32,
    pub hue: f32,
    pub odor: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuPoisonSensor {
    pub x: f32,
    pub y: f32,
    pub damage: f32,
    pub hue: f32,
    pub odor: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuObstacleSensor {
    pub x: f32,
    pub y: f32,
    pub half_w: f32,
    pub half_h: f32,
}

pub struct GpuSensorWorld<'a> {
    pub creatures: &'a [GpuCreatureSensor],
    pub foods: &'a [GpuFoodSensor],
    pub poisons: &'a [GpuPoisonSensor],
    pub obstacles: &'a [GpuObstacleSensor],
    pub origin: (f32, f32),
    pub dims: (f32, f32),
    pub cell: f32,
    pub half_fov: f32,
    pub vision: f32,
    pub max_energy: f32,
    pub min_creature_size: f32,
    pub max_creature_size: f32,
    pub max_age: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuBrainMeta {
    node_offset: u32,
    node_count: u32,
    edge_offset: u32,
    edge_count: u32,
    layer_offset: u32,
    layer_count: u32,
    output_offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuNode {
    incoming_offset: u32,
    incoming_count: u32,
    activation: u32,
    _pad: u32,
    bias: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuEdge {
    from_node: u32,
    weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GpuLayer {
    node_offset: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct CompiledBrain {
    meta: GpuBrainMeta,
    nodes: Vec<GpuNode>,
    edges: Vec<GpuEdge>,
    layers: Vec<GpuLayer>,
}

// Holds a submitted-but-not-drained GPU compute. Kept inside GpuBrainCompute so
// callers can submit work, do CPU work in parallel, then drain.
struct PendingDispatch {
    ids: Vec<usize>,
    n: usize,
    output_bytes: u64,
    mapped: bool,
}

pub struct GpuBrainCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    sensor_food_pipeline: wgpu::ComputePipeline,
    sensor_poison_pipeline: wgpu::ComputePipeline,
    sensor_creature_pipeline: wgpu::ComputePipeline,
    sensor_input_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sensor_bind_group_layout: wgpu::BindGroupLayout,

    params_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    brain_index_buffer: wgpu::Buffer,
    brain_meta_buffer: wgpu::Buffer,
    node_buffer: wgpu::Buffer,
    edge_buffer: wgpu::Buffer,
    layer_buffer: wgpu::Buffer,
    value_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    value_offset_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    sensor_params_buffer: wgpu::Buffer,
    sensor_creature_buffer: wgpu::Buffer,
    sensor_food_buffer: wgpu::Buffer,
    sensor_poison_buffer: wgpu::Buffer,
    sensor_obstacle_buffer: wgpu::Buffer,
    sensor_grid_count_buffer: wgpu::Buffer,
    sensor_grid_item_buffer: wgpu::Buffer,
    sensor_overflow_buffer: wgpu::Buffer,
    sensor_bind_group: wgpu::BindGroup,

    brain_capacity: u32,
    node_capacity: u32,
    edge_capacity: u32,
    layer_capacity: u32,
    dispatch_capacity: u32,
    value_capacity: u32,
    sensor_food_capacity: u32,
    sensor_poison_capacity: u32,
    sensor_obstacle_capacity: u32,
    sensor_grid_cell_capacity: u32,

    brain_of_id: HashMap<usize, u32>,
    free_brains: Vec<u32>,
    brain_metas: Vec<GpuBrainMeta>,
    nodes: Vec<GpuNode>,
    edges: Vec<GpuEdge>,
    layers: Vec<GpuLayer>,

    pending: Option<PendingDispatch>,
}

impl GpuBrainCompute {
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async()).ok()
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|err| format!("No GPU adapter available: {err}"))?;

        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(10);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("brain-compute-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: Default::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|err| format!("Could not create GPU device: {err}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("brain_compute.wgsl").into()),
        });
        let sensor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sensor-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sensor_compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain-compute-bind-group-layout"),
            entries: &[
                storage_entry(0, true),  // params
                storage_entry(1, true),  // inputs
                storage_entry(2, true),  // brain indices
                storage_entry(3, true),  // brain metas
                storage_entry(4, true),  // nodes
                storage_entry(5, true),  // incoming edges
                storage_entry(6, true),  // layers
                storage_entry(7, false), // per-dispatch node values
                storage_entry(8, false), // outputs
                storage_entry(9, true),  // value offsets
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain-compute-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brain-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sensor_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sensor-compute-bind-group-layout"),
                entries: &[
                    storage_entry(0, true),  // sensor params
                    storage_entry(1, true),  // creatures
                    storage_entry(2, true),  // food
                    storage_entry(3, true),  // poison
                    storage_entry(4, true),  // obstacles
                    storage_entry(5, false), // grid counts
                    storage_entry(6, false), // grid items
                    storage_entry(7, false), // generated brain inputs
                    storage_entry(8, false), // overflow counter
                ],
            });
        let sensor_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sensor-compute-pipeline-layout"),
                bind_group_layouts: &[Some(&sensor_bind_group_layout)],
                immediate_size: 0,
            });
        let make_sensor_pipeline = |entry: &'static str, label: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&sensor_pipeline_layout),
                module: &sensor_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sensor_food_pipeline = make_sensor_pipeline("build_food_grid", "sensor-food-grid");
        let sensor_poison_pipeline =
            make_sensor_pipeline("build_poison_grid", "sensor-poison-grid");
        let sensor_creature_pipeline =
            make_sensor_pipeline("build_creature_grid", "sensor-creature-grid");
        let sensor_input_pipeline = make_sensor_pipeline("generate_inputs", "sensor-input");

        let brain_capacity = INITIAL_BRAIN_CAPACITY;
        let node_capacity = INITIAL_NODE_CAPACITY;
        let edge_capacity = INITIAL_EDGE_CAPACITY;
        let layer_capacity = INITIAL_LAYER_CAPACITY;
        let dispatch_capacity = INITIAL_BRAIN_CAPACITY;
        let value_capacity = dispatch_capacity * INITIAL_VALUE_SLOTS_PER_CREATURE;

        let params_buffer = create_buffer::<Params>(&device, 1, "brain-compute-params", true);
        let input_buffer = create_input_buffer(&device, dispatch_capacity);
        let brain_index_buffer = create_buffer::<u32>(
            &device,
            dispatch_capacity,
            "brain-compute-dispatch-brain-indices",
            true,
        );
        let brain_meta_buffer = create_buffer::<GpuBrainMeta>(
            &device,
            brain_capacity,
            "brain-compute-brain-metas",
            true,
        );
        let node_buffer =
            create_buffer::<GpuNode>(&device, node_capacity, "brain-compute-nodes", true);
        let edge_buffer =
            create_buffer::<GpuEdge>(&device, edge_capacity, "brain-compute-edges", true);
        let layer_buffer =
            create_buffer::<GpuLayer>(&device, layer_capacity, "brain-compute-layers", true);
        let value_buffer = create_value_buffer(&device, value_capacity);
        let output_buffer = create_output_buffer(&device, dispatch_capacity);
        let readback_buffer = create_readback_buffer(&device, dispatch_capacity);
        let value_offset_buffer = create_buffer::<u32>(
            &device,
            dispatch_capacity,
            "brain-compute-value-offsets",
            true,
        );

        let sensor_params_buffer =
            create_buffer::<GpuSensorParams>(&device, 1, "sensor-params", true);
        let sensor_creature_buffer = create_buffer::<GpuCreatureSensor>(
            &device,
            dispatch_capacity,
            "sensor-creatures",
            true,
        );
        let sensor_food_capacity = dispatch_capacity * 2;
        let sensor_poison_capacity = dispatch_capacity * 12;
        let sensor_obstacle_capacity = 128;
        let sensor_grid_cell_capacity = 4096;
        let sensor_food_buffer =
            create_buffer::<GpuFoodSensor>(&device, sensor_food_capacity, "sensor-food", true);
        let sensor_poison_buffer = create_buffer::<GpuPoisonSensor>(
            &device,
            sensor_poison_capacity,
            "sensor-poison",
            true,
        );
        let sensor_obstacle_buffer = create_buffer::<GpuObstacleSensor>(
            &device,
            sensor_obstacle_capacity,
            "sensor-obstacles",
            true,
        );
        let sensor_grid_count_buffer = create_buffer::<u32>(
            &device,
            sensor_grid_cell_capacity * 3,
            "sensor-grid-counts",
            true,
        );
        let sensor_grid_item_buffer = create_buffer::<u32>(
            &device,
            sensor_grid_cell_capacity * 3 * SENSOR_GRID_BUCKET_CAPACITY,
            "sensor-grid-items",
            false,
        );
        let sensor_overflow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sensor-overflow"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = make_bind_group(
            &device,
            &bind_group_layout,
            &params_buffer,
            &input_buffer,
            &brain_index_buffer,
            &brain_meta_buffer,
            &node_buffer,
            &edge_buffer,
            &layer_buffer,
            &value_buffer,
            &output_buffer,
            &value_offset_buffer,
        );

        let sensor_bind_group = make_sensor_bind_group(
            &device,
            &sensor_bind_group_layout,
            &sensor_params_buffer,
            &sensor_creature_buffer,
            &sensor_food_buffer,
            &sensor_poison_buffer,
            &sensor_obstacle_buffer,
            &sensor_grid_count_buffer,
            &sensor_grid_item_buffer,
            &input_buffer,
            &sensor_overflow_buffer,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            sensor_food_pipeline,
            sensor_poison_pipeline,
            sensor_creature_pipeline,
            sensor_input_pipeline,
            bind_group_layout,
            sensor_bind_group_layout,
            params_buffer,
            input_buffer,
            brain_index_buffer,
            brain_meta_buffer,
            node_buffer,
            edge_buffer,
            layer_buffer,
            value_buffer,
            output_buffer,
            readback_buffer,
            value_offset_buffer,
            bind_group,
            sensor_params_buffer,
            sensor_creature_buffer,
            sensor_food_buffer,
            sensor_poison_buffer,
            sensor_obstacle_buffer,
            sensor_grid_count_buffer,
            sensor_grid_item_buffer,
            sensor_overflow_buffer,
            sensor_bind_group,
            brain_capacity,
            node_capacity,
            edge_capacity,
            layer_capacity,
            dispatch_capacity,
            value_capacity,
            sensor_food_capacity,
            sensor_poison_capacity,
            sensor_obstacle_capacity,
            sensor_grid_cell_capacity,
            brain_of_id: HashMap::new(),
            free_brains: Vec::new(),
            brain_metas: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            layers: Vec::new(),
            pending: None,
        })
    }

    pub fn clear_all(&mut self) {
        // Drain any in-flight work first so we don't leave the readback mapped.
        let _ = self.take_pending_outputs();
        self.brain_of_id.clear();
        self.free_brains.clear();
        self.brain_metas.clear();
        self.nodes.clear();
        self.edges.clear();
        self.layers.clear();
    }

    pub fn release(&mut self, id: usize) {
        if let Some(brain_idx) = self.brain_of_id.remove(&id) {
            self.free_brains.push(brain_idx);
        }
    }

    pub fn assign_brain(&mut self, id: usize, net: &Net) -> Result<(), String> {
        let compiled = compile_brain(net)?;

        let nodes_needed_after_append = self.nodes.len() + compiled.nodes.len();
        let edges_needed_after_append = self.edges.len() + compiled.edges.len();
        let layers_needed_after_append = self.layers.len() + compiled.layers.len();
        if self.should_compact_graph_storage(
            nodes_needed_after_append,
            edges_needed_after_append,
            layers_needed_after_append,
        ) {
            self.compact_graph_storage();
        }

        let node_offset = self.nodes.len() as u32;
        let edge_offset = self.edges.len() as u32;
        let layer_offset = self.layers.len() as u32;

        let mut meta = compiled.meta;
        meta.node_offset = node_offset;
        meta.edge_offset = edge_offset;
        meta.layer_offset = layer_offset;
        meta.output_offset += node_offset;

        self.check_graph_binding_limits(
            self.brain_metas.len() + 1,
            node_offset as usize + compiled.nodes.len(),
            edge_offset as usize + compiled.edges.len(),
            layer_offset as usize + compiled.layers.len(),
        )?;

        let rebuilt = self.ensure_graph_capacity(
            self.brain_metas.len() as u32 + 1,
            node_offset + compiled.nodes.len() as u32,
            edge_offset + compiled.edges.len() as u32,
            layer_offset + compiled.layers.len() as u32,
        );

        let brain_idx = if let Some(existing) = self.brain_of_id.get(&id).copied() {
            existing
        } else if let Some(free) = self.free_brains.pop() {
            self.brain_of_id.insert(id, free);
            free
        } else {
            let idx = self.brain_metas.len() as u32;
            self.brain_metas.push(GpuBrainMeta::default());
            self.brain_of_id.insert(id, idx);
            idx
        };

        self.nodes.extend(compiled.nodes);
        self.edges.extend(compiled.edges);
        self.layers.extend(compiled.layers);
        self.brain_metas[brain_idx as usize] = meta;

        if rebuilt {
            self.upload_graph_buffers();
        } else {
            self.upload_brain_meta(brain_idx);
            self.upload_graph_slice(
                &self.node_buffer,
                node_offset,
                &self.nodes[node_offset as usize..],
            );
            self.upload_graph_slice(
                &self.edge_buffer,
                edge_offset,
                &self.edges[edge_offset as usize..],
            );
            self.upload_graph_slice(
                &self.layer_buffer,
                layer_offset,
                &self.layers[layer_offset as usize..],
            );
        }
        Ok(())
    }

    fn should_compact_graph_storage(
        &self,
        nodes_needed_after_append: usize,
        edges_needed_after_append: usize,
        layers_needed_after_append: usize,
    ) -> bool {
        let max_nodes = max_elements_for_binding::<GpuNode>();
        let max_edges = max_elements_for_binding::<GpuEdge>();
        let max_layers = max_elements_for_binding::<GpuLayer>();
        if nodes_needed_after_append > max_nodes
            || edges_needed_after_append > max_edges
            || layers_needed_after_append > max_layers
        {
            return true;
        }

        let (active_nodes, active_edges, active_layers) = self.active_graph_usage();
        self.nodes.len()
            > active_nodes
                .saturating_mul(2)
                .max(INITIAL_NODE_CAPACITY as usize)
            || self.edges.len()
                > active_edges
                    .saturating_mul(2)
                    .max(INITIAL_EDGE_CAPACITY as usize)
            || self.layers.len()
                > active_layers
                    .saturating_mul(2)
                    .max(INITIAL_LAYER_CAPACITY as usize)
    }

    fn active_graph_usage(&self) -> (usize, usize, usize) {
        let mut nodes = 0usize;
        let mut edges = 0usize;
        let mut layers = 0usize;
        for &brain_idx in self.brain_of_id.values() {
            let brain = self.brain_metas[brain_idx as usize];
            nodes += brain.node_count as usize;
            edges += brain.edge_count as usize;
            layers += brain.layer_count as usize;
        }
        (nodes, edges, layers)
    }

    fn compact_graph_storage(&mut self) {
        let old_metas = self.brain_metas.clone();
        let old_nodes = std::mem::take(&mut self.nodes);
        let old_edges = std::mem::take(&mut self.edges);
        let old_layers = std::mem::take(&mut self.layers);

        self.nodes = Vec::new();
        self.edges = Vec::new();
        self.layers = Vec::new();
        self.brain_metas = vec![GpuBrainMeta::default(); old_metas.len()];

        let mut active_brains: Vec<u32> = self.brain_of_id.values().copied().collect();
        active_brains.sort_unstable();
        active_brains.dedup();

        for brain_idx in active_brains {
            let old = old_metas[brain_idx as usize];
            let node_offset = self.nodes.len() as u32;
            let edge_offset = self.edges.len() as u32;
            let layer_offset = self.layers.len() as u32;
            let output_local = old.output_offset - old.node_offset;

            self.nodes.extend_from_slice(
                &old_nodes[old.node_offset as usize..(old.node_offset + old.node_count) as usize],
            );
            self.edges.extend_from_slice(
                &old_edges[old.edge_offset as usize..(old.edge_offset + old.edge_count) as usize],
            );
            self.layers.extend_from_slice(
                &old_layers
                    [old.layer_offset as usize..(old.layer_offset + old.layer_count) as usize],
            );

            self.brain_metas[brain_idx as usize] = GpuBrainMeta {
                node_offset,
                node_count: old.node_count,
                edge_offset,
                edge_count: old.edge_count,
                layer_offset,
                layer_count: old.layer_count,
                output_offset: node_offset + output_local,
                _pad: 0,
            };
        }

        self.shrink_graph_capacity_to_fit();
        self.rebuild_bind_group();
        self.upload_graph_buffers();
    }

    fn shrink_graph_capacity_to_fit(&mut self) {
        self.brain_capacity = (self.brain_metas.len() as u32).max(INITIAL_BRAIN_CAPACITY);
        self.node_capacity = (self.nodes.len() as u32).max(INITIAL_NODE_CAPACITY);
        self.edge_capacity = (self.edges.len() as u32).max(INITIAL_EDGE_CAPACITY);
        self.layer_capacity = (self.layers.len() as u32).max(INITIAL_LAYER_CAPACITY);
        self.node_capacity = self
            .node_capacity
            .min(max_elements_for_binding::<GpuNode>() as u32);
        self.edge_capacity = self
            .edge_capacity
            .min(max_elements_for_binding::<GpuEdge>() as u32);
        self.layer_capacity = self
            .layer_capacity
            .min(max_elements_for_binding::<GpuLayer>() as u32);
        self.brain_meta_buffer = create_buffer::<GpuBrainMeta>(
            &self.device,
            self.brain_capacity,
            "brain-compute-brain-metas",
            true,
        );
        self.node_buffer = create_buffer::<GpuNode>(
            &self.device,
            self.node_capacity,
            "brain-compute-nodes",
            true,
        );
        self.edge_buffer = create_buffer::<GpuEdge>(
            &self.device,
            self.edge_capacity,
            "brain-compute-edges",
            true,
        );
        self.layer_buffer = create_buffer::<GpuLayer>(
            &self.device,
            self.layer_capacity,
            "brain-compute-layers",
            true,
        );
    }

    fn check_graph_binding_limits(
        &self,
        brains_needed: usize,
        nodes_needed: usize,
        edges_needed: usize,
        layers_needed: usize,
    ) -> Result<(), String> {
        if brains_needed > max_elements_for_binding::<GpuBrainMeta>() {
            return Err(format!(
                "active brain metadata exceeds GPU storage binding limit: {brains_needed}"
            ));
        }
        if nodes_needed > max_elements_for_binding::<GpuNode>() {
            return Err(format!(
                "active brain nodes exceed GPU storage binding limit: {nodes_needed}"
            ));
        }
        if edges_needed > max_elements_for_binding::<GpuEdge>() {
            return Err(format!(
                "active brain edges exceed GPU storage binding limit: {edges_needed}"
            ));
        }
        if layers_needed > max_elements_for_binding::<GpuLayer>() {
            return Err(format!(
                "active brain layers exceed GPU storage binding limit: {layers_needed}"
            ));
        }
        Ok(())
    }

    fn ensure_graph_capacity(
        &mut self,
        brains_needed: u32,
        nodes_needed: u32,
        edges_needed: u32,
        layers_needed: u32,
    ) -> bool {
        let mut rebuilt = false;
        if brains_needed > self.brain_capacity {
            self.brain_capacity = brains_needed.max(self.brain_capacity * 2);
            self.brain_meta_buffer = create_buffer::<GpuBrainMeta>(
                &self.device,
                self.brain_capacity,
                "brain-compute-brain-metas",
                true,
            );
            rebuilt = true;
        }
        if nodes_needed > self.node_capacity {
            self.node_capacity = nodes_needed
                .max(self.node_capacity * 2)
                .min(max_elements_for_binding::<GpuNode>() as u32);
            self.node_buffer = create_buffer::<GpuNode>(
                &self.device,
                self.node_capacity,
                "brain-compute-nodes",
                true,
            );
            rebuilt = true;
        }
        if edges_needed > self.edge_capacity {
            self.edge_capacity = edges_needed
                .max(self.edge_capacity * 2)
                .min(max_elements_for_binding::<GpuEdge>() as u32);
            self.edge_buffer = create_buffer::<GpuEdge>(
                &self.device,
                self.edge_capacity,
                "brain-compute-edges",
                true,
            );
            rebuilt = true;
        }
        if layers_needed > self.layer_capacity {
            self.layer_capacity = layers_needed
                .max(self.layer_capacity * 2)
                .min(max_elements_for_binding::<GpuLayer>() as u32);
            self.layer_buffer = create_buffer::<GpuLayer>(
                &self.device,
                self.layer_capacity,
                "brain-compute-layers",
                true,
            );
            rebuilt = true;
        }

        if rebuilt {
            self.rebuild_bind_group();
        }
        rebuilt
    }

    fn ensure_dispatch_capacity(&mut self, n: u32) {
        if n <= self.dispatch_capacity {
            return;
        }
        let new_cap = n.max(self.dispatch_capacity * 2);
        self.input_buffer = create_input_buffer(&self.device, new_cap);
        self.brain_index_buffer = create_buffer::<u32>(
            &self.device,
            new_cap,
            "brain-compute-dispatch-brain-indices",
            true,
        );
        self.output_buffer = create_output_buffer(&self.device, new_cap);
        self.readback_buffer = create_readback_buffer(&self.device, new_cap);
        self.value_offset_buffer =
            create_buffer::<u32>(&self.device, new_cap, "brain-compute-value-offsets", true);
        self.sensor_creature_buffer =
            create_buffer::<GpuCreatureSensor>(&self.device, new_cap, "sensor-creatures", true);
        self.dispatch_capacity = new_cap;
        self.rebuild_bind_group();
        self.rebuild_sensor_bind_group();
    }

    fn ensure_value_capacity(&mut self, total_slots: u32) {
        if total_slots <= self.value_capacity {
            return;
        }
        let new_cap = total_slots.max(self.value_capacity * 2);
        self.value_capacity = new_cap;
        self.value_buffer = create_value_buffer(&self.device, new_cap);
        self.rebuild_bind_group();
    }

    fn ensure_sensor_capacity(
        &mut self,
        food_count: u32,
        poison_count: u32,
        obstacle_count: u32,
        grid_cells: u32,
    ) {
        let mut rebuilt = false;
        if food_count.max(1) > self.sensor_food_capacity {
            self.sensor_food_capacity = food_count.max(self.sensor_food_capacity * 2).max(1);
            self.sensor_food_buffer = create_buffer::<GpuFoodSensor>(
                &self.device,
                self.sensor_food_capacity,
                "sensor-food",
                true,
            );
            rebuilt = true;
        }
        if poison_count.max(1) > self.sensor_poison_capacity {
            self.sensor_poison_capacity = poison_count.max(self.sensor_poison_capacity * 2).max(1);
            self.sensor_poison_buffer = create_buffer::<GpuPoisonSensor>(
                &self.device,
                self.sensor_poison_capacity,
                "sensor-poison",
                true,
            );
            rebuilt = true;
        }
        if obstacle_count.max(1) > self.sensor_obstacle_capacity {
            self.sensor_obstacle_capacity =
                obstacle_count.max(self.sensor_obstacle_capacity * 2).max(1);
            self.sensor_obstacle_buffer = create_buffer::<GpuObstacleSensor>(
                &self.device,
                self.sensor_obstacle_capacity,
                "sensor-obstacles",
                true,
            );
            rebuilt = true;
        }
        let grid_cells = grid_cells.min(MAX_SENSOR_GRID_CELLS);
        if grid_cells.max(1) > self.sensor_grid_cell_capacity {
            self.sensor_grid_cell_capacity = grid_cells
                .max(self.sensor_grid_cell_capacity * 2)
                .max(1)
                .min(MAX_SENSOR_GRID_CELLS);
            self.sensor_grid_count_buffer = create_buffer::<u32>(
                &self.device,
                self.sensor_grid_cell_capacity * 3,
                "sensor-grid-counts",
                true,
            );
            self.sensor_grid_item_buffer = create_buffer::<u32>(
                &self.device,
                self.sensor_grid_cell_capacity * 3 * SENSOR_GRID_BUCKET_CAPACITY,
                "sensor-grid-items",
                false,
            );
            rebuilt = true;
        }
        if rebuilt {
            self.rebuild_sensor_bind_group();
        }
    }

    fn rebuild_bind_group(&mut self) {
        self.bind_group = make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.params_buffer,
            &self.input_buffer,
            &self.brain_index_buffer,
            &self.brain_meta_buffer,
            &self.node_buffer,
            &self.edge_buffer,
            &self.layer_buffer,
            &self.value_buffer,
            &self.output_buffer,
            &self.value_offset_buffer,
        );
    }

    fn rebuild_sensor_bind_group(&mut self) {
        self.sensor_bind_group = make_sensor_bind_group(
            &self.device,
            &self.sensor_bind_group_layout,
            &self.sensor_params_buffer,
            &self.sensor_creature_buffer,
            &self.sensor_food_buffer,
            &self.sensor_poison_buffer,
            &self.sensor_obstacle_buffer,
            &self.sensor_grid_count_buffer,
            &self.sensor_grid_item_buffer,
            &self.input_buffer,
            &self.sensor_overflow_buffer,
        );
    }

    fn upload_graph_buffers(&self) {
        if !self.brain_metas.is_empty() {
            self.queue.write_buffer(
                &self.brain_meta_buffer,
                0,
                bytemuck::cast_slice(&self.brain_metas),
            );
        }
        if !self.nodes.is_empty() {
            self.queue
                .write_buffer(&self.node_buffer, 0, bytemuck::cast_slice(&self.nodes));
        }
        if !self.edges.is_empty() {
            self.queue
                .write_buffer(&self.edge_buffer, 0, bytemuck::cast_slice(&self.edges));
        }
        if !self.layers.is_empty() {
            self.queue
                .write_buffer(&self.layer_buffer, 0, bytemuck::cast_slice(&self.layers));
        }
    }

    fn upload_brain_meta(&self, brain_idx: u32) {
        self.queue.write_buffer(
            &self.brain_meta_buffer,
            brain_idx as u64 * std::mem::size_of::<GpuBrainMeta>() as u64,
            bytemuck::bytes_of(&self.brain_metas[brain_idx as usize]),
        );
    }

    fn upload_graph_slice<T: Pod>(&self, buffer: &wgpu::Buffer, element_offset: u32, data: &[T]) {
        if data.is_empty() {
            return;
        }
        self.queue.write_buffer(
            buffer,
            element_offset as u64 * std::mem::size_of::<T>() as u64,
            bytemuck::cast_slice(data),
        );
    }

    // Build per-creature value-buffer offsets via prefix sum over each brain's
    // node_count. Total slots returned is the size needed for value_buffer.
    fn build_value_offsets(&self, brain_indices: &[u32]) -> (Vec<u32>, u32) {
        let mut offsets = Vec::with_capacity(brain_indices.len());
        let mut acc: u32 = 0;
        for &b in brain_indices {
            offsets.push(acc);
            let nc = self.brain_metas[b as usize].node_count;
            acc = acc.saturating_add(nc);
        }
        (offsets, acc)
    }

    /// Submit a sensor + brain compute without blocking on readback. The caller
    /// must call `take_pending_outputs` to consume the result; subsequent
    /// `enqueue_*` calls will drain implicitly. Returns only after work is
    /// queued — GPU runs concurrently with subsequent CPU work.
    pub fn enqueue_with_sensors(&mut self, ids: &[usize], world: GpuSensorWorld<'_>) {
        // Drain previous before resubmitting (single readback buffer).
        let _ = self.take_pending_outputs();
        let n = ids.len();
        if n == 0 {
            return;
        }
        let output_bytes = self.submit_with_sensors(ids, world);
        let slice = self.readback_buffer.slice(0..output_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.pending = Some(PendingDispatch {
            ids: ids.to_vec(),
            n,
            output_bytes,
            mapped: true,
        });
    }

    /// Block until pending GPU work finishes and return its (ids, outputs).
    /// Returns `None` if nothing is pending.
    pub fn take_pending_outputs(&mut self) -> Option<(Vec<usize>, Vec<[f32; GPU_OUTPUTS]>)> {
        let pending = self.pending.take()?;
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let outputs = if pending.mapped {
            let slice = self.readback_buffer.slice(0..pending.output_bytes);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
            drop(mapped);
            self.readback_buffer.unmap();
            self.rows_from_flat(pending.n, &values)
        } else {
            Vec::new()
        };
        Some((pending.ids, outputs))
    }

    fn rows_from_flat(&self, n: usize, values: &[f32]) -> Vec<[f32; GPU_OUTPUTS]> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0; GPU_OUTPUTS];
            row.copy_from_slice(&values[i * GPU_OUTPUTS..(i + 1) * GPU_OUTPUTS]);
            out.push(row);
        }
        out
    }

    /// Encodes and submits the sensor + brain pipeline. Does not map readback.
    /// Returns the byte size of the output region.
    fn submit_with_sensors(&mut self, ids: &[usize], world: GpuSensorWorld<'_>) -> u64 {
        let n = ids.len();
        debug_assert_eq!(world.creatures.len(), n);

        self.ensure_dispatch_capacity(n as u32);
        // Pick a cell size that keeps total cells <= MAX_SENSOR_GRID_CELLS.
        // Doubles the cell as needed; coarser cells just mean each per-creature
        // vision rect touches fewer cells (cheaper) at the cost of more
        // entities per cell to filter. Wins big when the world grows.
        let mut cell = world.cell.max(1.0);
        let (cols, rows) = loop {
            let c = ((world.dims.0 / cell).ceil() as u32).max(1);
            let r = ((world.dims.1 / cell).ceil() as u32).max(1);
            let total = (c as u64).saturating_mul(r as u64);
            if total <= MAX_SENSOR_GRID_CELLS as u64 {
                break (c, r);
            }
            cell *= 2.0;
        };
        let grid_cells = cols * rows;
        self.ensure_sensor_capacity(
            world.foods.len() as u32,
            world.poisons.len() as u32,
            world.obstacles.len() as u32,
            grid_cells,
        );

        let brain_indices: Vec<u32> = ids
            .iter()
            .map(|id| {
                *self
                    .brain_of_id
                    .get(id)
                    .unwrap_or_else(|| panic!("GPU compute: id {id} has no brain assigned"))
            })
            .collect();
        let (value_offsets, total_value_slots) = self.build_value_offsets(&brain_indices);
        self.ensure_value_capacity(total_value_slots.max(1));

        let params = Params {
            creature_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let max_entity_count = (n as u32)
            .max(world.foods.len() as u32)
            .max(world.poisons.len() as u32);
        let sensor_params = GpuSensorParams {
            creature_count: n as u32,
            food_count: world.foods.len() as u32,
            poison_count: world.poisons.len() as u32,
            obstacle_count: world.obstacles.len() as u32,
            cols,
            rows,
            bucket_capacity: SENSOR_GRID_BUCKET_CAPACITY,
            max_entity_count,
            world_w: world.dims.0,
            world_h: world.dims.1,
            cell,
            half_fov: world.half_fov,
            vision: world.vision,
            max_energy: world.max_energy,
            min_creature_size: world.min_creature_size,
            max_creature_size: world.max_creature_size,
            max_age: world.max_age.max(1.0),
            origin_x: world.origin.0,
            origin_y: world.origin.1,
            _pad2: 0.0,
        };

        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(
            &self.brain_index_buffer,
            0,
            bytemuck::cast_slice(&brain_indices),
        );
        self.queue.write_buffer(
            &self.value_offset_buffer,
            0,
            bytemuck::cast_slice(&value_offsets),
        );
        self.queue.write_buffer(
            &self.sensor_params_buffer,
            0,
            bytemuck::bytes_of(&sensor_params),
        );
        self.queue.write_buffer(
            &self.sensor_creature_buffer,
            0,
            bytemuck::cast_slice(world.creatures),
        );
        if !world.foods.is_empty() {
            self.queue.write_buffer(
                &self.sensor_food_buffer,
                0,
                bytemuck::cast_slice(world.foods),
            );
        }
        if !world.poisons.is_empty() {
            self.queue.write_buffer(
                &self.sensor_poison_buffer,
                0,
                bytemuck::cast_slice(world.poisons),
            );
        }
        if !world.obstacles.is_empty() {
            self.queue.write_buffer(
                &self.sensor_obstacle_buffer,
                0,
                bytemuck::cast_slice(world.obstacles),
            );
        }

        let output_bytes = (n * GPU_OUTPUTS * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sensor-brain-compute-encoder"),
            });

        // Reset grid_counts and overflow counter via on-encoder clear (no CPU
        // alloc per tick). clear_buffer requires multiples of 4 bytes — buffers
        // already aligned.
        let count_bytes = (grid_cells as u64 * 3) * std::mem::size_of::<u32>() as u64;
        encoder.clear_buffer(&self.sensor_grid_count_buffer, 0, Some(count_bytes));
        encoder.clear_buffer(&self.sensor_overflow_buffer, 0, None);

        // Build three grids in separate dispatches sized to actual entity count.
        if !world.foods.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-food-grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sensor_food_pipeline);
            pass.set_bind_group(0, &self.sensor_bind_group, &[]);
            let groups = (world.foods.len() as u32).div_ceil(SENSOR_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if !world.poisons.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-poison-grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sensor_poison_pipeline);
            pass.set_bind_group(0, &self.sensor_bind_group, &[]);
            let groups = (world.poisons.len() as u32).div_ceil(SENSOR_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-creature-grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sensor_creature_pipeline);
            pass.set_bind_group(0, &self.sensor_bind_group, &[]);
            let groups = (n as u32).div_ceil(SENSOR_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sensor-input-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sensor_input_pipeline);
            pass.set_bind_group(0, &self.sensor_bind_group, &[]);
            let groups = (n as u32).div_ceil(SENSOR_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brain-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (n as u32).div_ceil(BRAIN_WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            output_bytes,
        );
        self.queue.submit(Some(encoder.finish()));
        output_bytes
    }
}

fn compile_brain(net: &Net) -> Result<CompiledBrain, String> {
    if net.input_layer != 0 {
        return Err("brain input layer must be layer 0".into());
    }
    if net.graph.layers.len() < 2 {
        return Err("brain must have input and output layers".into());
    }
    if net.output_layer as usize != net.graph.layers.len() - 1 {
        return Err("brain output layer must be the final layer".into());
    }
    if net.graph.layers[0].len() != GPU_INPUTS {
        return Err(format!("brain must have {GPU_INPUTS} inputs"));
    }
    if net.graph.layers[net.output_layer as usize].len() != GPU_OUTPUTS {
        return Err(format!("brain must have {GPU_OUTPUTS} outputs"));
    }

    let node_count: usize = net.graph.layers.iter().map(Vec::len).sum();
    if node_count > MAX_NODES_PER_BRAIN {
        return Err(format!(
            "brain has {node_count} nodes, max supported is {MAX_NODES_PER_BRAIN}"
        ));
    }

    let mut local_of_graph = Vec::with_capacity(net.graph.layers.len());
    let mut next_node = 0u32;
    let mut layers = Vec::with_capacity(net.graph.layers.len());
    for layer in &net.graph.layers {
        let start = next_node;
        let mut layer_map = Vec::with_capacity(layer.len());
        for _ in layer {
            layer_map.push(next_node);
            next_node += 1;
        }
        layers.push(GpuLayer {
            node_offset: start,
            node_count: layer.len() as u32,
            _pad0: 0,
            _pad1: 0,
        });
        local_of_graph.push(layer_map);
    }

    for node in &net.graph.layers[0] {
        if !matches!(node.value, Node::Input(_)) {
            return Err("input layer contains a non-input node".into());
        }
    }
    for node in &net.graph.layers[net.output_layer as usize] {
        match &node.value {
            Node::Output(output) if output._type() == "Sigmoid" => {}
            _ => return Err("output layer must contain Sigmoid output nodes".into()),
        }
    }

    let mut incoming: Vec<Vec<GpuEdge>> = vec![Vec::new(); node_count];
    for (from_layer_idx, layer) in net.graph.layers.iter().enumerate() {
        for (from_node_idx, node) in layer.iter().enumerate() {
            let from_local = local_of_graph[from_layer_idx][from_node_idx];
            for edge in &node.connections {
                if !edge.value.enabled {
                    continue;
                }
                if edge.to.layer as usize >= net.graph.layers.len() {
                    return Err("edge points outside graph layers".into());
                }
                if edge.to.node as usize >= net.graph.layers[edge.to.layer as usize].len() {
                    return Err("edge points outside target layer".into());
                }
                if edge.to.layer as usize <= from_layer_idx {
                    return Err("GPU brains must be feed-forward by layer".into());
                }
                let to_local = local_of_graph[edge.to.layer as usize][edge.to.node as usize];
                incoming[to_local as usize].push(GpuEdge {
                    from_node: from_local,
                    weight: edge.value.weight,
                });
            }
        }
    }

    let mut nodes = Vec::with_capacity(node_count);
    let mut edges = Vec::new();
    for (layer_idx, layer) in net.graph.layers.iter().enumerate() {
        for node in layer {
            let activation = if layer_idx == 0 {
                ACTIVATION_INPUT
            } else if layer_idx == net.output_layer as usize {
                ACTIVATION_OUTPUT_PARTIAL
            } else {
                match &node.value {
                    Node::Neuron(neuron) if neuron._type() == "BasicNeuron" => ACTIVATION_LINEAR,
                    Node::Neuron(neuron) => {
                        return Err(format!("unsupported hidden neuron type {}", neuron._type()))
                    }
                    _ => return Err("hidden layers must contain neuron nodes".into()),
                }
            };

            let bias = match &node.value {
                Node::Neuron(neuron) => neuron.bias(),
                _ => 0.0,
            };
            let incoming_offset = edges.len() as u32;
            let node_incoming = &incoming[nodes.len()];
            edges.extend(node_incoming.iter().copied());
            nodes.push(GpuNode {
                incoming_offset,
                incoming_count: node_incoming.len() as u32,
                activation,
                _pad: 0,
                bias,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            });
        }
    }

    Ok(CompiledBrain {
        meta: GpuBrainMeta {
            node_offset: 0,
            node_count: node_count as u32,
            edge_offset: 0,
            edge_count: edges.len() as u32,
            layer_offset: 0,
            layer_count: layers.len() as u32,
            output_offset: layers[net.output_layer as usize].node_offset,
            _pad: 0,
        },
        nodes,
        edges,
        layers,
    })
}

fn create_buffer<T: Pod>(
    device: &wgpu::Device,
    cap: u32,
    label: &'static str,
    copy_dst: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (cap as usize * std::mem::size_of::<T>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn max_elements_for_binding<T>() -> usize {
    MAX_STORAGE_BINDING_BYTES / std::mem::size_of::<T>()
}

fn create_input_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    create_buffer::<f32>(
        device,
        cap * GPU_INPUTS as u32,
        "brain-compute-inputs",
        true,
    )
}

fn create_value_buffer(device: &wgpu::Device, total_slots: u32) -> wgpu::Buffer {
    create_buffer::<f32>(device, total_slots.max(1), "brain-compute-values", false)
}

fn create_output_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    usage |= wgpu::BufferUsages::COPY_DST;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain-compute-outputs"),
        size: (cap as usize * GPU_OUTPUTS * std::mem::size_of::<f32>()) as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn create_readback_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain-compute-readback"),
        size: (cap as usize * GPU_OUTPUTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    inputs: &wgpu::Buffer,
    brain_indices: &wgpu::Buffer,
    brain_metas: &wgpu::Buffer,
    nodes: &wgpu::Buffer,
    edges: &wgpu::Buffer,
    layers: &wgpu::Buffer,
    values: &wgpu::Buffer,
    outputs: &wgpu::Buffer,
    value_offsets: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("brain-compute-bind-group"),
        layout,
        entries: &[
            bind_entry(0, params),
            bind_entry(1, inputs),
            bind_entry(2, brain_indices),
            bind_entry(3, brain_metas),
            bind_entry(4, nodes),
            bind_entry(5, edges),
            bind_entry(6, layers),
            bind_entry(7, values),
            bind_entry(8, outputs),
            bind_entry(9, value_offsets),
        ],
    })
}

fn make_sensor_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    creatures: &wgpu::Buffer,
    foods: &wgpu::Buffer,
    poisons: &wgpu::Buffer,
    obstacles: &wgpu::Buffer,
    grid_counts: &wgpu::Buffer,
    grid_items: &wgpu::Buffer,
    inputs: &wgpu::Buffer,
    overflow: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sensor-compute-bind-group"),
        layout,
        entries: &[
            bind_entry(0, params),
            bind_entry(1, creatures),
            bind_entry(2, foods),
            bind_entry(3, poisons),
            bind_entry(4, obstacles),
            bind_entry(5, grid_counts),
            bind_entry(6, grid_items),
            bind_entry(7, inputs),
            bind_entry(8, overflow),
        ],
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    let size = buffer.size().min(MAX_STORAGE_BINDING_BYTES as u64);
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(size),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::BlankInput;
    use engine::{
        activations::Sigmoid,
        nn::{BasicNeuron, Edge, GraphLocation, GraphNode, NeuralGraph},
    };

    #[test]
    fn compiles_hidden_layer_brain() {
        let mut graph = NeuralGraph::new();
        let input_layer = graph.add_layer_to_end();
        for id in 0..GPU_INPUTS {
            graph
                .add_node(
                    input_layer,
                    GraphNode::new(Node::Input(BlankInput::new(0.0, id, "test"))),
                )
                .unwrap();
        }

        let hidden_layer = graph.add_layer_to_end();
        graph
            .add_node(
                hidden_layer,
                GraphNode::new(Node::Neuron(Box::new(BasicNeuron::new(0.25, 1000)))),
            )
            .unwrap();

        let output_layer = graph.add_layer_to_end();
        for id in 0..GPU_OUTPUTS {
            graph
                .add_node(
                    output_layer,
                    GraphNode::new(Node::Output(Sigmoid::new(0.0, id, "out".to_string()))),
                )
                .unwrap();
        }

        graph
            .add_edge(
                GraphLocation::new(input_layer, 0),
                GraphLocation::new(hidden_layer, 0),
                Edge {
                    weight: 2.0,
                    enabled: true,
                },
            )
            .unwrap();
        graph
            .add_edge(
                GraphLocation::new(hidden_layer, 0),
                GraphLocation::new(output_layer, 0),
                Edge {
                    weight: 3.0,
                    enabled: true,
                },
            )
            .unwrap();

        let net = Net {
            graph,
            input_layer,
            output_layer,
        };

        let compiled = compile_brain(&net).unwrap();
        assert_eq!(
            compiled.meta.node_count as usize,
            GPU_INPUTS + 1 + GPU_OUTPUTS
        );
        assert_eq!(compiled.meta.layer_count, 3);
        assert_eq!(compiled.meta.edge_count, 2);
        assert_eq!(compiled.nodes[GPU_INPUTS].activation, ACTIVATION_LINEAR);
        assert_eq!(compiled.nodes[GPU_INPUTS].bias, 0.25);
    }
}
