struct Params {
    creature_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct BrainMeta {
    node_offset: u32,
    node_count: u32,
    edge_offset: u32,
    edge_count: u32,
    layer_offset: u32,
    layer_count: u32,
    output_offset: u32,
    _pad: u32,
};

struct Node {
    incoming_offset: u32,
    incoming_count: u32,
    activation: u32,
    _pad: u32,
    bias: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

struct Edge {
    from_node: u32,
    weight: f32,
};

struct Layer {
    node_offset: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> brain_indices: array<u32>;
@group(0) @binding(3) var<storage, read> brain_metas: array<BrainMeta>;
@group(0) @binding(4) var<storage, read> nodes: array<Node>;
@group(0) @binding(5) var<storage, read> edges: array<Edge>;
@group(0) @binding(6) var<storage, read> layers: array<Layer>;
@group(0) @binding(7) var<storage, read_write> values: array<f32>;
@group(0) @binding(8) var<storage, read_write> outputs: array<f32>;
// Per-creature start index into `values`. Lets us right-size scratch instead of
// reserving MAX_NODES_PER_BRAIN slots per creature.
@group(0) @binding(9) var<storage, read> value_offsets: array<u32>;

const INPUTS: u32 = 52u;
const OUTPUTS: u32 = 10u;
const WORKGROUP_SIZE: u32 = 128u;

const ACTIVATION_INPUT: u32 = 0u;
const ACTIVATION_LINEAR: u32 = 1u;
const ACTIVATION_OUTPUT_PARTIAL: u32 = 2u;

fn activate(kind: u32, partial: f32, bias: f32) -> f32 {
    if (kind == ACTIVATION_LINEAR) {
        return partial + bias;
    }
    if (kind == ACTIVATION_OUTPUT_PARTIAL) {
        return partial;
    }
    return partial;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let creature_idx = global_id.x;
    if (creature_idx >= params.creature_count) {
        return;
    }

    let brain_idx = brain_indices[creature_idx];
    let brain = brain_metas[brain_idx];
    let value_base = value_offsets[creature_idx];

    for (var i = 0u; i < INPUTS; i = i + 1u) {
        values[value_base + i] = inputs[creature_idx * INPUTS + i];
    }

    // Layer 0 is the input layer and has already been written.
    for (var layer_i = 1u; layer_i < brain.layer_count; layer_i = layer_i + 1u) {
        let layer = layers[brain.layer_offset + layer_i];
        for (var layer_node_i = 0u; layer_node_i < layer.node_count; layer_node_i = layer_node_i + 1u) {
            let local_node = layer.node_offset + layer_node_i;
            let node = nodes[brain.node_offset + local_node];

            var partial = 0.0;
            for (var edge_i = 0u; edge_i < node.incoming_count; edge_i = edge_i + 1u) {
                let edge = edges[brain.edge_offset + node.incoming_offset + edge_i];
                partial = partial + values[value_base + edge.from_node] * edge.weight;
            }

            values[value_base + local_node] = activate(node.activation, partial, node.bias);
        }
    }

    for (var o = 0u; o < OUTPUTS; o = o + 1u) {
        outputs[creature_idx * OUTPUTS + o] = values[value_base + (brain.output_offset - brain.node_offset) + o];
    }
}
