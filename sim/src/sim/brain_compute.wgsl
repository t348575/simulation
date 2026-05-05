struct Params {
    creature_count: u32,
    input_count: u32,
    output_count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;

const TILE: u32 = 8u;
const INPUTS: u32 = 32u;
const OUTPUTS: u32 = 8u;

// 8 creatures per workgroup × 32 inputs each = 256 floats shared.
var<workgroup> shared_inputs: array<f32, 256>;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let lane_x = local_id.x; // 0..8 — output index within creature (only 0..7 used)
    let lane_y = local_id.y; // 0..8 — creature within tile
    let creature_idx = wg_id.x * TILE + lane_y;

    // Cooperative input load: 64 lanes load 8 creatures × 32 inputs = 256 floats.
    // Linear lane id 0..63, each loads ceil(256/64) = 4 entries.
    let lane = lane_y * 8u + lane_x; // 0..63
    for (var k = 0u; k < 4u; k = k + 1u) {
        let slot_idx = lane + k * 64u;
        if (slot_idx < TILE * INPUTS) {
            let tile_creature = slot_idx / INPUTS;
            let input_i = slot_idx % INPUTS;
            let global_creature = wg_id.x * TILE + tile_creature;
            if (global_creature < params.creature_count) {
                shared_inputs[slot_idx] = inputs[global_creature * INPUTS + input_i];
            }
        }
    }

    workgroupBarrier();

    if (creature_idx >= params.creature_count) {
        return;
    }
    if (lane_x >= OUTPUTS) {
        return;
    }

    let weight_slot = slots[creature_idx];
    let weight_offset = weight_slot * INPUTS * OUTPUTS;
    let input_base = lane_y * INPUTS;

    var sum = 0.0;
    for (var i = 0u; i < INPUTS; i = i + 1u) {
        sum = sum + shared_inputs[input_base + i] * weights[weight_offset + i * OUTPUTS + lane_x];
    }

    partials[creature_idx * OUTPUTS + lane_x] = sum;
}
