use bevy::math::Vec2;

pub struct Grid {
    cell: f32,
    cols: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
    positions: Vec<Vec2>,
}

impl Grid {
    pub fn build(positions: Vec<Vec2>, cell: f32, dims: (f32, f32)) -> Self {
        let cell = cell.max(1.0);
        let cols = ((dims.0 / cell).ceil() as usize).max(1);
        let rows = ((dims.1 / cell).ceil() as usize).max(1);
        let mut buckets = vec![Vec::<u32>::new(); cols * rows];
        for (i, p) in positions.iter().enumerate() {
            let cx = ((p.x / cell) as isize).clamp(0, cols as isize - 1) as usize;
            let cy = ((p.y / cell) as isize).clamp(0, rows as isize - 1) as usize;
            buckets[cy * cols + cx].push(i as u32);
        }
        Self {
            cell,
            cols,
            rows,
            buckets,
            positions,
        }
    }

    pub fn position(&self, idx: u32) -> Vec2 {
        self.positions[idx as usize]
    }

    fn cell_of(&self, p: Vec2) -> (isize, isize) {
        let cx = (p.x / self.cell) as isize;
        let cy = (p.y / self.cell) as isize;
        (
            cx.clamp(0, self.cols as isize - 1),
            cy.clamp(0, self.rows as isize - 1),
        )
    }

    /// Find nearest position (squared distance). Returns None if grid empty.
    pub fn nearest(&self, from: Vec2) -> Option<(u32, f32)> {
        if self.positions.is_empty() {
            return None;
        }
        let (fx, fy) = self.cell_of(from);
        let max_radius = self.cols.max(self.rows) as isize;
        let mut best: Option<(u32, f32)> = None;
        let mut r: isize = 0;
        loop {
            let lo_x = (fx - r).max(0) as usize;
            let hi_x = (fx + r).min(self.cols as isize - 1) as usize;
            let lo_y = (fy - r).max(0) as usize;
            let hi_y = (fy + r).min(self.rows as isize - 1) as usize;

            for y in lo_y..=hi_y {
                for x in lo_x..=hi_x {
                    if r > 0
                        && (x as isize) > fx - r
                        && (x as isize) < fx + r
                        && (y as isize) > fy - r
                        && (y as isize) < fy + r
                    {
                        continue; // interior — already searched in prior ring
                    }
                    if x >= self.cols || y >= self.rows {
                        continue;
                    }
                    for &idx in &self.buckets[y * self.cols + x] {
                        let d2 = (self.positions[idx as usize] - from).length_squared();
                        match best {
                            Some((_, b)) if b <= d2 => {}
                            _ => best = Some((idx, d2)),
                        }
                    }
                }
            }

            if let Some((_, b2)) = best {
                // Expand one extra ring beyond the radius that contains the candidate.
                let safe_dist = (r as f32) * self.cell;
                if b2.sqrt() <= safe_dist {
                    return best;
                }
            }
            r += 1;
            if r > max_radius {
                break;
            }
        }
        best
    }

    pub fn nearest_filtered(
        &self,
        from: Vec2,
        mut keep: impl FnMut(u32) -> bool,
    ) -> Option<(u32, f32)> {
        if self.positions.is_empty() {
            return None;
        }
        let (fx, fy) = self.cell_of(from);
        let max_radius = self.cols.max(self.rows) as isize;
        let mut best: Option<(u32, f32)> = None;
        let mut r: isize = 0;
        loop {
            let lo_x = (fx - r).max(0) as usize;
            let hi_x = (fx + r).min(self.cols as isize - 1) as usize;
            let lo_y = (fy - r).max(0) as usize;
            let hi_y = (fy + r).min(self.rows as isize - 1) as usize;

            for y in lo_y..=hi_y {
                for x in lo_x..=hi_x {
                    if r > 0
                        && (x as isize) > fx - r
                        && (x as isize) < fx + r
                        && (y as isize) > fy - r
                        && (y as isize) < fy + r
                    {
                        continue;
                    }
                    if x >= self.cols || y >= self.rows {
                        continue;
                    }
                    for &idx in &self.buckets[y * self.cols + x] {
                        if !keep(idx) {
                            continue;
                        }
                        let d2 = (self.positions[idx as usize] - from).length_squared();
                        match best {
                            Some((_, b)) if b <= d2 => {}
                            _ => best = Some((idx, d2)),
                        }
                    }
                }
            }

            if let Some((_, b2)) = best {
                let safe_dist = (r as f32) * self.cell;
                if b2.sqrt() <= safe_dist {
                    return best;
                }
            }
            r += 1;
            if r > max_radius {
                break;
            }
        }
        best
    }

    /// Visit every index whose bucket overlaps the AABB.
    pub fn query_aabb(&self, min: Vec2, max: Vec2, mut f: impl FnMut(u32)) {
        let (lx, ly) = self.cell_of(min);
        let (hx, hy) = self.cell_of(max);
        let lx = lx as usize;
        let ly = ly as usize;
        let hx = hx as usize;
        let hy = hy as usize;
        for y in ly..=hy {
            for x in lx..=hx {
                for &idx in &self.buckets[y * self.cols + x] {
                    f(idx);
                }
            }
        }
    }
}
