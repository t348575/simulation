use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine::nn::{Net, Node};

pub const GPU_INPUTS: usize = 32;
pub const GPU_OUTPUTS: usize = 8;
pub const WEIGHTS_PER_BRAIN: usize = GPU_INPUTS * GPU_OUTPUTS;

const INITIAL_CAPACITY: u32 = 16384;
const TILE: u32 = 8; // creatures per workgroup (must match shader)

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    creature_count: u32,
    input_count: u32,
    output_count: u32,
    _pad: u32,
}

pub struct GpuBrainCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // Persistent weight slab (capacity * WEIGHTS_PER_BRAIN floats), indexed by slot.
    weight_buffer: wgpu::Buffer,
    // Per-tick buffers
    params_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    slot_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    // Slot allocation
    capacity: u32,
    slot_of_id: HashMap<usize, u32>,
    free_slots: Vec<u32>,
    next_slot: u32,

    // Per-tick scratch capacity (for inputs/slots/outputs)
    dispatch_capacity: u32,
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
            limits.max_storage_buffers_per_shader_stage.max(5);
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain-compute-bind-group-layout"),
            entries: &[
                storage_entry(0, true),  // params
                storage_entry(1, true),  // inputs
                storage_entry(2, true),  // weights (persistent)
                storage_entry(3, true),  // slots (per-tick)
                storage_entry(4, false), // outputs
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brain-compute-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brain-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let capacity = INITIAL_CAPACITY;
        let dispatch_capacity = INITIAL_CAPACITY;

        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-compute-weights-persistent"),
            size: (capacity as usize * WEIGHTS_PER_BRAIN * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-compute-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let input_buffer = create_input_buffer(&device, dispatch_capacity);
        let slot_buffer = create_slot_buffer(&device, dispatch_capacity);
        let output_buffer = create_output_buffer(&device, dispatch_capacity);
        let readback_buffer = create_readback_buffer(&device, dispatch_capacity);

        let bind_group = make_bind_group(
            &device,
            &bind_group_layout,
            &params_buffer,
            &input_buffer,
            &weight_buffer,
            &slot_buffer,
            &output_buffer,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            weight_buffer,
            params_buffer,
            input_buffer,
            slot_buffer,
            output_buffer,
            readback_buffer,
            bind_group,
            capacity,
            slot_of_id: HashMap::new(),
            free_slots: Vec::new(),
            next_slot: 0,
            dispatch_capacity,
        })
    }

    pub fn clear_all(&mut self) {
        self.slot_of_id.clear();
        self.free_slots.clear();
        self.next_slot = 0;
    }

    pub fn release(&mut self, id: usize) {
        if let Some(slot) = self.slot_of_id.remove(&id) {
            self.free_slots.push(slot);
        }
    }

    /// Allocate (or reuse) a slot for `id` and write its weights into the persistent slab.
    pub fn assign_slot(&mut self, id: usize, weights: &[f32; WEIGHTS_PER_BRAIN]) {
        let slot = if let Some(&existing) = self.slot_of_id.get(&id) {
            existing
        } else if let Some(s) = self.free_slots.pop() {
            self.slot_of_id.insert(id, s);
            s
        } else {
            let s = self.next_slot;
            self.next_slot += 1;
            if s >= self.capacity {
                self.grow_weight_capacity((s + 1).max(self.capacity * 2));
            }
            self.slot_of_id.insert(id, s);
            s
        };

        let offset =
            (slot as u64) * (WEIGHTS_PER_BRAIN as u64) * (std::mem::size_of::<f32>() as u64);
        self.queue
            .write_buffer(&self.weight_buffer, offset, bytemuck::cast_slice(weights));
    }

    fn grow_weight_capacity(&mut self, new_capacity: u32) {
        // Re-create persistent weight buffer larger. All slots must be re-uploaded by caller
        // afterwards — but to avoid that, we copy the old contents over first.
        let new_size =
            (new_capacity as usize * WEIGHTS_PER_BRAIN * std::mem::size_of::<f32>()) as u64;
        let new_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-compute-weights-persistent"),
            size: new_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let old_size =
            (self.capacity as usize * WEIGHTS_PER_BRAIN * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("weight-grow-copy"),
            });
        encoder.copy_buffer_to_buffer(&self.weight_buffer, 0, &new_buf, 0, old_size);
        self.queue.submit(Some(encoder.finish()));

        self.weight_buffer = new_buf;
        self.capacity = new_capacity;
        self.rebuild_bind_group();
    }

    fn ensure_dispatch_capacity(&mut self, n: u32) {
        if n <= self.dispatch_capacity {
            return;
        }
        let new_cap = n.max(self.dispatch_capacity * 2);
        self.input_buffer = create_input_buffer(&self.device, new_cap);
        self.slot_buffer = create_slot_buffer(&self.device, new_cap);
        self.output_buffer = create_output_buffer(&self.device, new_cap);
        self.readback_buffer = create_readback_buffer(&self.device, new_cap);
        self.dispatch_capacity = new_cap;
        self.rebuild_bind_group();
    }

    fn rebuild_bind_group(&mut self) {
        self.bind_group = make_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.params_buffer,
            &self.input_buffer,
            &self.weight_buffer,
            &self.slot_buffer,
            &self.output_buffer,
        );
    }

    /// Synchronous dispatch. `ids` and `inputs` are parallel: inputs.len() == ids.len() * GPU_INPUTS.
    /// All ids must already have a slot assigned via `assign_slot`.
    pub fn compute(&mut self, ids: &[usize], inputs: &[f32]) -> Vec<[f32; GPU_OUTPUTS]> {
        let n = ids.len();
        if n == 0 {
            return Vec::new();
        }
        debug_assert_eq!(inputs.len(), n * GPU_INPUTS);

        self.ensure_dispatch_capacity(n as u32);

        // Build slot index array
        let mut slots: Vec<u32> = Vec::with_capacity(n);
        for &id in ids {
            let slot = *self
                .slot_of_id
                .get(&id)
                .unwrap_or_else(|| panic!("GPU compute: id {id} has no slot assigned"));
            slots.push(slot);
        }

        let params = Params {
            creature_count: n as u32,
            input_count: GPU_INPUTS as u32,
            output_count: GPU_OUTPUTS as u32,
            _pad: 0,
        };

        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(inputs));
        self.queue
            .write_buffer(&self.slot_buffer, 0, bytemuck::cast_slice(&slots));

        let output_bytes = (n * GPU_OUTPUTS * std::mem::size_of::<f32>()) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("brain-compute-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brain-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (n as u32 + TILE - 1) / TILE;
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

        let slice = self.readback_buffer.slice(0..output_bytes);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("readback channel").expect("readback map");

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = [0.0; GPU_OUTPUTS];
            row.copy_from_slice(&values[i * GPU_OUTPUTS..(i + 1) * GPU_OUTPUTS]);
            out.push(row);
        }
        out
    }
}

fn create_input_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain-compute-inputs"),
        size: (cap as usize * GPU_INPUTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_slot_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain-compute-slots"),
        size: (cap as usize * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_output_buffer(device: &wgpu::Device, cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brain-compute-outputs"),
        size: (cap as usize * GPU_OUTPUTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
    weights: &wgpu::Buffer,
    slots: &wgpu::Buffer,
    outputs: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("brain-compute-bind-group"),
        layout,
        entries: &[
            bind_entry(0, params),
            bind_entry(1, inputs),
            bind_entry(2, weights),
            bind_entry(3, slots),
            bind_entry(4, outputs),
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
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub fn direct_brain_weights(net: &Net) -> Option<[f32; GPU_INPUTS * GPU_OUTPUTS]> {
    if net.input_layer != 0 || net.output_layer != 1 || net.graph.layers.len() != 2 {
        return None;
    }
    if net.graph.layers[0].len() != GPU_INPUTS || net.graph.layers[1].len() != GPU_OUTPUTS {
        return None;
    }
    for node in &net.graph.layers[0] {
        if !matches!(node.value, Node::Input(_)) {
            return None;
        }
    }
    for node in &net.graph.layers[1] {
        match &node.value {
            Node::Output(output) if output._type() == "Sigmoid" => {}
            _ => return None,
        }
    }

    let mut weights = [0.0; GPU_INPUTS * GPU_OUTPUTS];
    for (input_idx, node) in net.graph.layers[0].iter().enumerate() {
        for edge in &node.connections {
            if edge.to.layer as usize != 1 || edge.to.node as usize >= GPU_OUTPUTS {
                return None;
            }
            if edge.value.enabled {
                weights[input_idx * GPU_OUTPUTS + edge.to.node as usize] = edge.value.weight;
            }
        }
    }
    Some(weights)
}
