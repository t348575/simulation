use std::collections::HashMap;

use bevy::math::Vec2;

pub struct Grid {
    cell: f32,
    buckets: HashMap<(i32, i32), Vec<u32>>,
    positions: Vec<Vec2>,
}

impl Grid {
    pub fn build(positions: Vec<Vec2>, cell: f32, _dims: (f32, f32)) -> Self {
        let cell = cell.max(1.0);
        let mut buckets = HashMap::<(i32, i32), Vec<u32>>::new();
        for (i, p) in positions.iter().enumerate() {
            buckets.entry(cell_of(*p, cell)).or_default().push(i as u32);
        }
        Self {
            cell,
            buckets,
            positions,
        }
    }

    pub fn position(&self, idx: u32) -> Vec2 {
        self.positions[idx as usize]
    }

    /// Visit every index whose bucket overlaps the AABB.
    pub fn query_aabb(&self, min: Vec2, max: Vec2, mut f: impl FnMut(u32)) {
        let (lx, ly) = cell_of(min, self.cell);
        let (hx, hy) = cell_of(max, self.cell);
        for y in ly.min(hy)..=ly.max(hy) {
            for x in lx.min(hx)..=lx.max(hx) {
                if let Some(bucket) = self.buckets.get(&(x, y)) {
                    for &idx in bucket {
                        f(idx);
                    }
                }
            }
        }
    }
}

fn cell_of(p: Vec2, cell: f32) -> (i32, i32) {
    ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32)
}
