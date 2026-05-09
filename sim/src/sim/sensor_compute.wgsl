struct SensorParams {
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
};

struct CreatureSensor {
    pos_x: f32,
    pos_y: f32,
    angle: f32,
    energy: f32,
    prev_energy: f32,
    touched: f32,
    mem: f32,
    age: f32,
    max_energy: f32,
    body_size: f32,
    max_age: f32,
    half_fov: f32,
    vision: f32,
    prev_thrust: f32,
    prev_turn_left: f32,
    prev_turn_right: f32,
    terrain_elevation: f32,
    terrain_speed: f32,
    terrain_energy_cost: f32,
    terrain_hazard: f32,
    terrain_cost_s0: f32,
    terrain_cost_s1: f32,
    terrain_cost_s2: f32,
    terrain_cost_s3: f32,
    terrain_haz_s0: f32,
    terrain_haz_s1: f32,
    terrain_haz_s2: f32,
    terrain_haz_s3: f32,
    signal: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct FoodSensor {
    x: f32,
    y: f32,
    size: f32,
    energy: f32,
    hue: f32,
    odor: f32,
    _pad0: f32,
    _pad1: f32,
};

struct PoisonSensor {
    x: f32,
    y: f32,
    damage: f32,
    hue: f32,
    odor: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct ObstacleSensor {
    x: f32,
    y: f32,
    half_w: f32,
    half_h: f32,
};

@group(0) @binding(0) var<storage, read> params: SensorParams;
@group(0) @binding(1) var<storage, read> creatures: array<CreatureSensor>;
@group(0) @binding(2) var<storage, read> foods: array<FoodSensor>;
@group(0) @binding(3) var<storage, read> poisons: array<PoisonSensor>;
@group(0) @binding(4) var<storage, read> obstacles: array<ObstacleSensor>;
@group(0) @binding(5) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> grid_items: array<u32>;
@group(0) @binding(7) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(8) var<storage, read_write> overflow: array<atomic<u32>>;

const INPUTS: u32 = 52u;
const WORKGROUP_SIZE: u32 = 128u;
const GRID_FOOD: u32 = 0u;
const GRID_POISON: u32 = 1u;
const GRID_CREATURE: u32 = 2u;

fn clamp_cell_x(x: f32) -> u32 {
    return u32(clamp(i32((x - params.origin_x) / params.cell), 0, i32(params.cols) - 1));
}

fn clamp_cell_y(y: f32) -> u32 {
    return u32(clamp(i32((y - params.origin_y) / params.cell), 0, i32(params.rows) - 1));
}

fn cell_index(x: f32, y: f32) -> u32 {
    return clamp_cell_y(y) * params.cols + clamp_cell_x(x);
}

fn grid_count_offset(kind: u32, cell: u32) -> u32 {
    return kind * params.cols * params.rows + cell;
}

fn grid_item_offset(kind: u32, cell: u32, slot: u32) -> u32 {
    return (kind * params.cols * params.rows + cell) * params.bucket_capacity + slot;
}

fn scatter(kind: u32, idx: u32, x: f32, y: f32) {
    let cell = cell_index(x, y);
    let count_idx = grid_count_offset(kind, cell);
    let slot = atomicAdd(&grid_counts[count_idx], 1u);
    if (slot < params.bucket_capacity) {
        grid_items[grid_item_offset(kind, cell, slot)] = idx;
    } else {
        atomicAdd(&overflow[0], 1u);
    }
}

// View context precomputed once per creature. Saves recomputing trig per target.
struct View {
    from_x: f32,
    from_y: f32,
    facing_x: f32,
    facing_y: f32,
    cos_half: f32,     // cos(half_fov)
    cos_quarter: f32,  // cos(half_fov / 2) — sector midpoint
    vision: f32,
    vision2: f32,
    inv_vision: f32,
    full_circle: u32,
};

fn make_view(c_x: f32, c_y: f32, angle: f32, half_fov: f32, vision: f32) -> View {
    var v: View;
    v.from_x = c_x;
    v.from_y = c_y;
    v.facing_x = cos(angle);
    v.facing_y = sin(angle);
    v.cos_half = cos(half_fov);
    v.cos_quarter = cos(half_fov * 0.5);
    v.vision = vision;
    v.vision2 = vision * vision;
    v.inv_vision = select(1.0 / max(vision, 0.000001), 0.0, vision <= 0.0);
    v.full_circle = select(0u, 1u, half_fov >= 3.14159265359);
    return v;
}

// Returns: x = 1 if target hit (in vision and FOV); y = closeness; z = sector index (0..3); w = signed-dot.
// Sector binning without atan2: sign(cross) splits left/right halves, |dot/d| vs cos(quarter) splits sub-halves.
// Layout matches old code (rel mapping): rel in [-h, -h/2)=0, [-h/2, 0)=1, [0, h/2)=2, [h/2, h]=3.
// rel = atan2(cross, alongFacing). cross > 0 → rel > 0 → sector >= 2.
// Within rel >= 0: sector 2 iff rel < h/2 iff cos(rel) > cos(h/2). cos(rel) = dot/d.
// Within rel <  0: sector 1 iff rel >= -h/2 iff cos(rel) > cos(h/2).
fn classify(v: ptr<function, View>, target_x: f32, target_y: f32) -> vec4<f32> {
    let dx = target_x - (*v).from_x;
    let dy = target_y - (*v).from_y;
    let d2 = dx * dx + dy * dy;
    if (d2 > (*v).vision2 || d2 < 0.000001) {
        return vec4<f32>(0.0);
    }
    let d = sqrt(d2);
    let inv_d = 1.0 / d;
    let along = dx * (*v).facing_x + dy * (*v).facing_y;
    let cross = (*v).facing_x * dy - (*v).facing_y * dx;
    let cos_rel = along * inv_d;
    if ((*v).full_circle == 0u && cos_rel < (*v).cos_half) {
        return vec4<f32>(0.0);
    }
    // Determine sector without atan2.
    var sector: f32;
    if (cross >= 0.0) {
        // rel >= 0 → sector 2 or 3.
        if (cos_rel > (*v).cos_quarter) {
            sector = 2.0;
        } else {
            sector = 3.0;
        }
    } else {
        if (cos_rel > (*v).cos_quarter) {
            sector = 1.0;
        } else {
            sector = 0.0;
        }
    }
    let closeness = clamp(1.0 - d * (*v).inv_vision, 0.0, 1.0);
    return vec4<f32>(1.0, closeness, sector, 0.0);
}

fn fold_sector(sect: ptr<function, vec4<f32>>, hit: vec4<f32>) {
    if (hit.x == 0.0) { return; }
    let s = u32(hit.z);
    let c = hit.y;
    if (c > (*sect)[s]) { (*sect)[s] = c; }
}

fn fold_visible(vis: ptr<function, vec2<f32>>, hit: vec4<f32>, strength: f32, hue: f32) {
    if (hit.x == 0.0) { return; }
    let signal = hit.y * clamp(strength, 0.0, 1.0);
    if (signal > (*vis).x) {
        *vis = vec2<f32>(signal, hue);
    }
}

@compute @workgroup_size(128)
fn build_food_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.food_count) { return; }
    let f = foods[idx];
    scatter(GRID_FOOD, idx, f.x, f.y);
}

@compute @workgroup_size(128)
fn build_poison_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.poison_count) { return; }
    let p = poisons[idx];
    scatter(GRID_POISON, idx, p.x, p.y);
}

@compute @workgroup_size(128)
fn build_creature_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.creature_count) { return; }
    let c = creatures[idx];
    scatter(GRID_CREATURE, idx, c.pos_x, c.pos_y);
}

@compute @workgroup_size(128)
fn generate_inputs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.creature_count) { return; }

    let c = creatures[idx];
    var view = make_view(c.pos_x, c.pos_y, c.angle, c.half_fov, c.vision);
    // Signal hearing is omnidirectional but uses the same sector mapping so
    // the brain can tell which side a voice is coming from.
    var signal_view = make_view(c.pos_x, c.pos_y, c.angle, 3.14159265359, c.vision);

    var food_sect = vec4<f32>(0.0);
    var poison_sect = vec4<f32>(0.0);
    var creature_sect = vec4<f32>(0.0);
    var obstacle_sect = vec4<f32>(0.0);
    var food_visible = vec2<f32>(0.0);
    var poison_visible = vec2<f32>(0.0);
    var signal_sect = vec4<f32>(0.0);

    if (c.vision > 0.0) {
        let min_x = clamp_cell_x(c.pos_x - c.vision);
        let max_x = clamp_cell_x(c.pos_x + c.vision);
        let min_y = clamp_cell_y(c.pos_y - c.vision);
        let max_y = clamp_cell_y(c.pos_y + c.vision);
        let cols = params.cols;
        let cap = params.bucket_capacity;
        let cells_per_kind = params.cols * params.rows;

        for (var y = min_y; y <= max_y; y = y + 1u) {
            let row_base = y * cols;
            for (var x = min_x; x <= max_x; x = x + 1u) {
                let cell = row_base + x;

                // Fused food: sector + visible in one pass.
                let fcount = min(atomicLoad(&grid_counts[GRID_FOOD * cells_per_kind + cell]), cap);
                let fbase = (GRID_FOOD * cells_per_kind + cell) * cap;
                for (var s = 0u; s < fcount; s = s + 1u) {
                    let i = grid_items[fbase + s];
                    let f = foods[i];
                    let hit = classify(&view, f.x, f.y);
                    fold_sector(&food_sect, hit);
                    fold_visible(&food_visible, hit, f.odor, f.hue);
                }

                // Fused poison.
                let pcount = min(atomicLoad(&grid_counts[GRID_POISON * cells_per_kind + cell]), cap);
                let pbase = (GRID_POISON * cells_per_kind + cell) * cap;
                for (var s = 0u; s < pcount; s = s + 1u) {
                    let i = grid_items[pbase + s];
                    let p = poisons[i];
                    let hit = classify(&view, p.x, p.y);
                    fold_sector(&poison_sect, hit);
                    fold_visible(&poison_visible, hit, p.odor, p.hue);
                }

                // Creature sector + signal accumulation.
                let ccount = min(atomicLoad(&grid_counts[GRID_CREATURE * cells_per_kind + cell]), cap);
                let cbase = (GRID_CREATURE * cells_per_kind + cell) * cap;
                for (var s = 0u; s < ccount; s = s + 1u) {
                    let i = grid_items[cbase + s];
                    if (i == idx) { continue; }
                    let other = creatures[i];
                    let hit = classify(&view, other.pos_x, other.pos_y);
                    fold_sector(&creature_sect, hit);
                    let shit = classify(&signal_view, other.pos_x, other.pos_y);
                    if (shit.x != 0.0) {
                        let sect_idx = u32(shit.z);
                        signal_sect[sect_idx] = signal_sect[sect_idx] + other.signal * shit.y;
                    }
                }
            }
        }
    }

    // Obstacles aren't gridded — small list scan.
    for (var i = 0u; i < params.obstacle_count; i = i + 1u) {
        let o = obstacles[i];
        let hit = classify(&view, o.x, o.y);
        fold_sector(&obstacle_sect, hit);
    }

    let max_e = max(c.max_energy, 1.0);
    let hunger = 1.0 - clamp(c.energy / max_e, 0.0, 1.0);
    let health = clamp(c.energy / max_e, 0.0, 1.0);
    let damage = clamp(max(c.prev_energy - c.energy, 0.0) / max_e, 0.0, 1.0);
    let osc = sin(c.age * 0.1);
    let size_n = clamp((c.body_size - params.min_creature_size) / max(params.max_creature_size - params.min_creature_size, 0.001), 0.0, 1.0);

    let base = idx * INPUTS;
    inputs[base + 0u] = hunger;
    inputs[base + 1u] = health;
    inputs[base + 2u] = c.prev_thrust;
    inputs[base + 3u] = c.prev_turn_left - c.prev_turn_right;
    inputs[base + 4u] = view.facing_x;
    inputs[base + 5u] = view.facing_y;
    inputs[base + 6u] = c.touched;
    inputs[base + 7u] = damage;
    inputs[base + 8u] = osc;
    inputs[base + 9u] = c.mem;
    inputs[base + 10u] = clamp(c.age / max(c.max_age, 1.0), 0.0, 1.0);
    inputs[base + 11u] = size_n;
    let local_x = c.pos_x - params.origin_x;
    let local_y = c.pos_y - params.origin_y;
    inputs[base + 12u] = 1.0 - clamp(local_x / max(params.world_w, 1.0), 0.0, 1.0);
    inputs[base + 13u] = clamp(local_x / max(params.world_w, 1.0), 0.0, 1.0);
    inputs[base + 14u] = 1.0 - clamp(local_y / max(params.world_h, 1.0), 0.0, 1.0);
    inputs[base + 15u] = clamp(local_y / max(params.world_h, 1.0), 0.0, 1.0);
    inputs[base + 16u] = food_sect[0];
    inputs[base + 17u] = food_sect[1];
    inputs[base + 18u] = food_sect[2];
    inputs[base + 19u] = food_sect[3];
    inputs[base + 20u] = poison_sect[0];
    inputs[base + 21u] = poison_sect[1];
    inputs[base + 22u] = poison_sect[2];
    inputs[base + 23u] = poison_sect[3];
    inputs[base + 24u] = obstacle_sect[0];
    inputs[base + 25u] = obstacle_sect[1];
    inputs[base + 26u] = obstacle_sect[2];
    inputs[base + 27u] = obstacle_sect[3];
    inputs[base + 28u] = creature_sect[0];
    inputs[base + 29u] = creature_sect[1];
    inputs[base + 30u] = creature_sect[2];
    inputs[base + 31u] = creature_sect[3];
    inputs[base + 32u] = clamp(c.terrain_elevation, 0.0, 1.0);
    inputs[base + 33u] = clamp(c.terrain_speed, 0.0, 1.0);
    inputs[base + 34u] = clamp((c.terrain_energy_cost - 1.0) / 1.5, 0.0, 1.0);
    inputs[base + 35u] = clamp(c.terrain_hazard / 0.05, 0.0, 1.0);
    inputs[base + 36u] = food_visible.x;
    inputs[base + 37u] = food_visible.y;
    inputs[base + 38u] = poison_visible.x;
    inputs[base + 39u] = poison_visible.y;
    inputs[base + 40u] = c.terrain_cost_s0;
    inputs[base + 41u] = c.terrain_cost_s1;
    inputs[base + 42u] = c.terrain_cost_s2;
    inputs[base + 43u] = c.terrain_cost_s3;
    inputs[base + 44u] = c.terrain_haz_s0;
    inputs[base + 45u] = c.terrain_haz_s1;
    inputs[base + 46u] = c.terrain_haz_s2;
    inputs[base + 47u] = c.terrain_haz_s3;
    inputs[base + 48u] = clamp(signal_sect[0], -1.0, 1.0);
    inputs[base + 49u] = clamp(signal_sect[1], -1.0, 1.0);
    inputs[base + 50u] = clamp(signal_sect[2], -1.0, 1.0);
    inputs[base + 51u] = clamp(signal_sect[3], -1.0, 1.0);
}
