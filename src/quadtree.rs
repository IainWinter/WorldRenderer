use crate::camera::Camera;
use crate::gpu::{Gpu, UploadBudget};
use crate::math::{horizon_visible, Frustum};
use crate::stream::{JobKind, WorkerPool};
use crate::terrain_gpu::{TerrainRenderer, TileInstance, MAX_DRAWN_TILES};
use crate::tiling::{
    imagery_max_zoom, imagery_url, max_tile_zoom, terrain_source, tile_at, TileKey, MAX_TILE_HEIGHT,
};
use glam::DVec3;
use std::collections::HashMap;

pub const ROOT_LEVEL: u8 = 3;
pub const TILE_PIXEL_TARGET: f64 = 512.0;
pub const SPLIT_HYSTERESIS: f64 = 0.75;
pub const MAX_RETRIES: u8 = 3;
pub const RETRY_DELAY_FRAMES: u64 = 120;
pub const IMAGERY_FADE_FRAMES: f64 = 15.0;
pub const GEOMORPH_FRAMES: f64 = 18.0;
pub const EVICT_PROTECT_FRAMES: u64 = 30;
pub const MAX_DEFERRED: usize = 512;
pub const MAX_LEVELS: usize = 24;
pub const HEIGHT_PAD_SLACK: f64 = 500.0;
pub const ARENA_PRESSURE_LIMIT: f32 = 0.95;
pub const RETIRE_AFTER_FRAMES: u64 = 240;
pub const RETIRE_JUMP_FRAMES: u64 = 15;
pub const RETIRE_PRESSURE_FRAMES: u64 = 60;
pub const ARENA_TRIM_HIGH: f32 = 0.85;
pub const CAMERA_JUMP_FRACTION: f64 = 0.5;
pub const DEBUG_TINT: f32 = 0.6;

const LEVEL_COLORS: [[f32; 3]; 12] = [
    [0.25, 0.35, 1.0],
    [0.2, 0.7, 1.0],
    [0.2, 1.0, 0.85],
    [0.3, 1.0, 0.35],
    [0.75, 1.0, 0.2],
    [1.0, 0.9, 0.2],
    [1.0, 0.65, 0.15],
    [1.0, 0.4, 0.2],
    [1.0, 0.25, 0.45],
    [1.0, 0.3, 0.8],
    [0.75, 0.4, 1.0],
    [0.55, 0.55, 0.65],
];

struct TileShading {
    layer: f32,
    prev_layer: f32,
    uvxf: [f32; 4],
    prev_uvxf: [f32; 4],
    fade: f32,
}

impl Default for TileShading {
    fn default() -> Self {
        Self {
            layer: -1.0,
            prev_layer: -1.0,
            uvxf: [1.0, 1.0, 0.0, 0.0],
            prev_uvxf: [1.0, 1.0, 0.0, 0.0],
            fade: 0.0,
        }
    }
}

struct MeshRes {
    slot: u32,
    center: DVec3,
    min_height: f32,
    max_height: f32,
}

#[derive(Default)]
struct Tile {
    mesh: Option<MeshRes>,
    imagery: Option<u32>,
    mesh_inbound: bool,
    imagery_inbound: bool,
    imagery_at: u64,
    mesh_fails: u8,
    imagery_fails: u8,
    mesh_retry_at: u64,
    imagery_retry_at: u64,
    last_used: u64,
    last_drawn: u64,
    unfold_at: u64,
    merge_at: u64,
    split: bool,
}

fn ramp(at: u64, frame: u64) -> f32 {
    if at == 0 {
        return 1.0;
    }
    (frame.saturating_sub(at) as f64 / GEOMORPH_FRAMES).clamp(0.0, 1.0) as f32
}

#[derive(Default)]
pub struct LevelStat {
    pub meshes: u32,
    pub imagery: u32,
    pub drawn: u32,
    pub pending: u32,
}

#[derive(Default)]
pub struct TreeDebug {
    pub tiles: usize,
    pub meshes: usize,
    pub imagery: usize,
    pub mesh_inbound: usize,
    pub imagery_inbound: usize,
    pub mesh_failed: usize,
    pub imagery_failed: usize,
    pub split: usize,
    pub protected: usize,
    pub evictable_mesh: usize,
    pub evictable_imagery: usize,
    pub levels: Vec<(u8, LevelStat)>,
}

pub struct TileTree {
    tiles: HashMap<TileKey, Tile>,
    frame: u64,
    pub drawn: Vec<u32>,
    pub drawn_keys: Vec<TileKey>,
    pub instances: Vec<TileInstance>,
    pub max_level_drawn: u8,
    pub min_level_drawn: u8,
    pub uploaded_bytes: usize,
    pub deferred_uploads: usize,
    pub mesh_evictions: u64,
    pub imagery_evictions: u64,
    pub upload_stalls: u64,
    pub dropped_uploads: u64,
    pub splits: u32,
    pub starved_splits: u32,
    pub visited: u32,
    pub culled: u32,
    pub blocked_splits: u32,
    pub mesh_pressure: f32,
    pub layer_pressure: f32,
    pub retired_meshes: u64,
    pub retired_imagery: u64,
    pub camera_jumped: bool,
    pub last_frustum: Option<Frustum>,
    pub last_cull_eye: DVec3,
    pub frozen: bool,
    pub debug_mode: u32,
    last_anchor: DVec3,
    pub models: Vec<crate::stream::Incoming>,
    pub icons: Vec<crate::stream::Incoming>,
}

impl TileTree {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            frame: 0,
            drawn: Vec::new(),
            drawn_keys: Vec::new(),
            instances: Vec::new(),
            max_level_drawn: 0,
            min_level_drawn: 0,
            uploaded_bytes: 0,
            deferred_uploads: 0,
            mesh_evictions: 0,
            imagery_evictions: 0,
            upload_stalls: 0,
            dropped_uploads: 0,
            splits: 0,
            starved_splits: 0,
            visited: 0,
            culled: 0,
            blocked_splits: 0,
            mesh_pressure: 0.0,
            layer_pressure: 0.0,
            retired_meshes: 0,
            retired_imagery: 0,
            camera_jumped: false,
            last_frustum: None,
            last_cull_eye: DVec3::ZERO,
            frozen: false,
            debug_mode: 0,
            last_anchor: DVec3::ZERO,
            models: Vec::new(),
            icons: Vec::new(),
        }
    }

    pub fn debug(&self) -> TreeDebug {
        let frame = self.frame;
        let mut out = TreeDebug {
            tiles: self.tiles.len(),
            ..Default::default()
        };
        let mut levels: Vec<LevelStat> = (0..MAX_LEVELS).map(|_| LevelStat::default()).collect();
        for (key, tile) in self.tiles.iter() {
            let slot = &mut levels[(key.z as usize).min(MAX_LEVELS - 1)];
            if tile.mesh.is_some() {
                out.meshes += 1;
                slot.meshes += 1;
            }
            if tile.imagery.is_some() {
                out.imagery += 1;
                slot.imagery += 1;
            }
            if tile.mesh_inbound {
                out.mesh_inbound += 1;
                slot.pending += 1;
            }
            if tile.imagery_inbound {
                out.imagery_inbound += 1;
                slot.pending += 1;
            }
            if tile.mesh_fails >= MAX_RETRIES {
                out.mesh_failed += 1;
            }
            if tile.imagery_fails >= MAX_RETRIES {
                out.imagery_failed += 1;
            }
            if tile.split {
                out.split += 1;
            }
            if tile.last_drawn == frame {
                slot.drawn += 1;
            }
            let protected =
                tile.last_drawn + EVICT_PROTECT_FRAMES > frame || tile.last_used >= frame;
            if protected && (tile.mesh.is_some() || tile.imagery.is_some()) {
                out.protected += 1;
            }
            if !protected {
                if tile.mesh.is_some() {
                    out.evictable_mesh += 1;
                }
                if tile.imagery.is_some() {
                    out.evictable_imagery += 1;
                }
            }
        }
        out.levels = levels
            .into_iter()
            .enumerate()
            .filter(|(_, s)| s.meshes > 0 || s.imagery > 0 || s.drawn > 0 || s.pending > 0)
            .map(|(z, s)| (z as u8, s))
            .collect();
        out
    }

    pub fn ground_height(&self, lon: f64, lat: f64) -> f64 {
        let mut z = max_tile_zoom();
        loop {
            let key = tile_at(lon, lat, z);
            if let Some(h) = self.tiles.get(&key).and_then(|t| t.mesh.as_ref()) {
                return h.max_height as f64;
            }
            if z == 0 {
                return 0.0;
            }
            z -= 1;
        }
    }

    pub fn resident_tiles(&self) -> usize {
        self.tiles.values().filter(|t| t.mesh.is_some()).count()
    }

    pub fn resident_imagery(&self) -> usize {
        self.tiles.values().filter(|t| t.imagery.is_some()).count()
    }

    pub fn clear(&mut self, renderer: &mut TerrainRenderer) {
        for tile in self.tiles.values_mut() {
            if let Some(mesh) = tile.mesh.take() {
                renderer.slots.free(mesh.slot);
            }
            if let Some(layer) = tile.imagery.take() {
                renderer.layers.free(layer);
            }
        }
        self.tiles.clear();
        self.drawn.clear();
        self.drawn_keys.clear();
        self.instances.clear();
    }

    pub fn clear_imagery(&mut self, renderer: &mut TerrainRenderer) {
        for tile in self.tiles.values_mut() {
            if let Some(layer) = tile.imagery.take() {
                renderer.layers.free(layer);
            }
            tile.imagery_fails = 0;
            tile.imagery_retry_at = 0;
            tile.imagery_at = 0;
        }
    }

    pub fn select(&mut self, camera: &Camera, eye: DVec3, screen_h: f32, pool: &mut WorkerPool) {
        if !self.frozen {
            self.frame += 1;
        }
        self.drawn.clear();
        self.drawn_keys.clear();
        self.instances.clear();
        self.max_level_drawn = 0;
        self.min_level_drawn = u8::MAX;
        self.splits = 0;
        self.starved_splits = 0;
        self.visited = 0;
        self.culled = 0;
        self.blocked_splits = 0;

        let anchor_now = camera.target();
        let moved = (anchor_now - self.last_anchor).length();
        self.camera_jumped =
            self.last_anchor != DVec3::ZERO && moved > camera.distance * CAMERA_JUMP_FRACTION;
        self.last_anchor = anchor_now;

        let frustum = Frustum::from_camera(camera);
        self.last_frustum = Some(Frustum::from_camera(camera));
        self.last_cull_eye = camera.eye;
        let k = screen_h as f64 / (2.0 * (camera.fov_y as f64 * 0.5).tan());
        if !self.frozen {
            self.prefetch_focus(camera, k, pool);
        }
        let roots = 1u32 << ROOT_LEVEL;
        for y in 0..roots {
            for x in 0..roots {
                self.visit(
                    TileKey {
                        z: ROOT_LEVEL,
                        x,
                        y,
                    },
                    camera,
                    eye,
                    &frustum,
                    k,
                    pool,
                );
            }
        }
        if self.min_level_drawn == u8::MAX {
            self.min_level_drawn = 0;
        }
    }

    fn prefetch_focus(&mut self, camera: &Camera, k: f64, pool: &mut WorkerPool) {
        let (lon, lat) = camera.lon_lat();
        let deepest = max_tile_zoom();
        for z in ROOT_LEVEL..=deepest {
            let key = tile_at(lon, lat, z);
            let (center, radius) = key.bounding_sphere();
            let dist = ((center - camera.eye).length() - radius).max(1.0);
            let n = 1u32 << z;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = key.x as i32 + dx;
                    let ny = key.y as i32 + dy;
                    if nx < 0 || ny < 0 || nx as u32 >= n || ny as u32 >= n {
                        continue;
                    }
                    let nkey = TileKey {
                        z,
                        x: nx as u32,
                        y: ny as u32,
                    };
                    self.request_mesh(nkey, pool, -1000.0 + z as f32);
                    self.request_imagery(nkey, pool, -1000.0 + z as f32);
                }
            }
            if key.ground_extent() / dist * k <= TILE_PIXEL_TARGET {
                break;
            }
        }
    }

    fn height_range(&self, key: TileKey) -> (f64, f64) {
        let mut levels = 0;
        loop {
            let anc = key.ancestor(levels);
            if let Some(mesh) = self.tiles.get(&anc).and_then(|t| t.mesh.as_ref()) {
                return (
                    mesh.min_height as f64 - HEIGHT_PAD_SLACK,
                    mesh.max_height as f64 + HEIGHT_PAD_SLACK,
                );
            }
            if anc.z == 0 {
                return (-HEIGHT_PAD_SLACK, MAX_TILE_HEIGHT);
            }
            levels += 1;
        }
    }

    fn visible(&self, key: TileKey, camera: &Camera, frustum: &Frustum) -> Option<f64> {
        let (center, radius) = key.bounding_sphere();
        let (h_min, h_max) = self.height_range(key);
        if !horizon_visible(camera.eye, center, radius + h_max.max(0.0)) {
            return None;
        }
        let (bc, baxes, bhalf) = key.bounding_box(h_min, h_max);
        let rel = (bc - camera.eye).as_vec3();
        let axes = [baxes[0].as_vec3(), baxes[1].as_vec3(), baxes[2].as_vec3()];
        let half = [bhalf[0] as f32, bhalf[1] as f32, bhalf[2] as f32];
        if !frustum.box_visible(rel, axes, half) {
            return None;
        }
        Some(((center - camera.eye).length() - radius).max(1.0))
    }

    pub fn surface_in_frustum(&self, key: TileKey) -> bool {
        let Some(frustum) = self.last_frustum.as_ref() else {
            return true;
        };
        let (lon0, lat0, lon1, lat1) = key.lon_lat_bounds();
        let (h_min, h_max) = self.height_range(key);
        let n = 6;
        for i in 0..=n {
            let lat = lat0 + (lat1 - lat0) * i as f64 / n as f64;
            for j in 0..=n {
                let lon = lon0 + (lon1 - lon0) * j as f64 / n as f64;
                for h in [h_min, h_max] {
                    let p = crate::math::geodetic_to_ecef(lon, lat, h);
                    if frustum.contains_point((p - self.last_cull_eye).as_vec3()) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn bound_segments(&self) -> Vec<(DVec3, DVec3, u32)> {
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        let mut out = Vec::with_capacity(self.drawn_keys.len() * EDGES.len());
        for key in self.drawn_keys.iter() {
            let (h_min, h_max) = self.height_range(*key);
            let (c, axes, half) = key.bounding_box(h_min, h_max);
            let rgb = LEVEL_COLORS[key.z as usize % LEVEL_COLORS.len()];
            let color = if self.surface_in_frustum(*key) {
                ((rgb[0] * 255.0) as u32) << 24
                    | ((rgb[1] * 255.0) as u32) << 16
                    | ((rgb[2] * 255.0) as u32) << 8
                    | 0xff
            } else {
                0xff0000ff
            };
            let mut corners = [DVec3::ZERO; 8];
            for (i, corner) in corners.iter_mut().enumerate() {
                let sx = if i & 1 == 0 { -1.0 } else { 1.0 };
                let sy = if i & 2 == 0 { -1.0 } else { 1.0 };
                let sz = if i & 4 == 0 { -1.0 } else { 1.0 };
                *corner =
                    c + axes[0] * half[0] * sx + axes[1] * half[1] * sy + axes[2] * half[2] * sz;
            }
            for (a, b) in EDGES {
                out.push((corners[a], corners[b], color));
            }
        }
        out
    }

    fn renderable(&self, key: TileKey) -> bool {
        self.tiles.get(&key).is_some_and(|t| t.mesh.is_some())
    }

    fn exhausted(&self, key: TileKey) -> bool {
        self.tiles
            .get(&key)
            .is_some_and(|t| t.mesh_fails >= MAX_RETRIES)
    }

    fn visit(
        &mut self,
        key: TileKey,
        camera: &Camera,
        eye: DVec3,
        frustum: &Frustum,
        k: f64,
        pool: &mut WorkerPool,
    ) {
        let Some(dist) = self.visible(key, camera, frustum) else {
            self.culled += 1;
            return;
        };
        self.visited += 1;

        let extent = key.ground_extent();
        let screen_px = extent / dist * k;
        let was_split = self.tiles.get(&key).is_some_and(|t| t.split);
        let prev_drawn = self.tiles.get(&key).map(|t| t.last_drawn).unwrap_or(0);
        let threshold = if was_split {
            TILE_PIXEL_TARGET * SPLIT_HYSTERESIS
        } else {
            TILE_PIXEL_TARGET
        };
        let want_children = key.z < max_tile_zoom() && screen_px > threshold;

        if want_children {
            let children: [TileKey; 4] = [key.child(0), key.child(1), key.child(2), key.child(3)];
            let mut any_visible = false;
            let mut all_visible_ready = true;
            for c in children.iter() {
                if self.visible(*c, camera, frustum).is_some() {
                    any_visible = true;
                    if !self.renderable(*c) && !self.exhausted(*c) {
                        all_visible_ready = false;
                    }
                }
            }
            if any_visible && all_visible_ready {
                self.splits += 1;
                let frame = self.frame;
                let parent = self.tiles.entry(key).or_default();
                parent.split = true;
                parent.last_used = frame;
                parent.last_drawn = frame;
                if self.mesh_pressure < ARENA_TRIM_HIGH {
                    for c in children {
                        self.request_mesh(c, pool, dist as f32 * 2.0);
                    }
                }
                for c in children {
                    self.visit(c, camera, eye, frustum, k, pool);
                }
                return;
            }
            if any_visible {
                self.starved_splits += 1;
                if self.mesh_pressure < ARENA_PRESSURE_LIMIT {
                    for c in children {
                        if self.visible(c, camera, frustum).is_some() {
                            self.request_mesh(c, pool, dist as f32 * 0.5);
                        }
                    }
                } else {
                    self.blocked_splits += 1;
                }
            }
        }

        self.tiles.entry(key).or_default().split = false;
        self.request_mesh(key, pool, dist as f32);

        if self.drawn.len() as u32 >= MAX_DRAWN_TILES {
            return;
        }
        let frame = self.frame;
        let resident = match self.tiles.get_mut(&key) {
            Some(tile) => {
                tile.last_used = frame;
                tile.mesh.as_ref().map(|m| (m.slot, m.center))
            }
            None => None,
        };
        let Some((slot, center)) = resident else {
            return;
        };
        let mut morph_floor = 0.0f32;
        let mut morph_ceil = 1.0f32;
        if let Some(tile) = self.tiles.get_mut(&key) {
            tile.last_drawn = frame;
            if was_split {
                tile.merge_at = frame;
                tile.unfold_at = 0;
            } else if prev_drawn + 1 < frame {
                tile.unfold_at = frame;
                tile.merge_at = 0;
            }
            morph_floor = 1.0 - ramp(tile.unfold_at, frame);
            morph_ceil = ramp(tile.merge_at, frame).max(morph_floor);
        }
        self.request_imagery(key, pool, dist as f32);
        let origin = (center - eye).as_vec3();
        let shade = self.resolve_imagery(key);
        let dbg = self.debug_color(key, slot);
        let morph_lo = extent * k / TILE_PIXEL_TARGET;
        self.max_level_drawn = self.max_level_drawn.max(key.z);
        self.min_level_drawn = self.min_level_drawn.min(key.z);
        self.drawn.push(slot);
        self.drawn_keys.push(key);
        self.instances.push(TileInstance {
            origin: origin.to_array(),
            morph_lo: morph_lo as f32,
            uvxf: shade.uvxf,
            prev_uvxf: shade.prev_uvxf,
            layers: [shade.layer, shade.prev_layer],
            blend: [shade.fade, (morph_lo * 2.0) as f32, morph_floor, morph_ceil],
            dbg,
        });
    }

    fn imagery_depth(&self, key: TileKey) -> i32 {
        let mut levels = 0u8;
        loop {
            let anc = key.ancestor(levels);
            if self.tiles.get(&anc).is_some_and(|t| t.imagery.is_some()) {
                return levels as i32;
            }
            if anc.z == 0 {
                return -1;
            }
            levels += 1;
        }
    }

    fn debug_color(&self, key: TileKey, slot: u32) -> [f32; 4] {
        if self.debug_mode == 0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let a = DEBUG_TINT;
        let rgb = match self.debug_mode {
            1 => LEVEL_COLORS[key.z as usize % LEVEL_COLORS.len()],
            2 => {
                let tile = self.tiles.get(&key);
                let failed = tile
                    .is_some_and(|t| t.mesh_fails >= MAX_RETRIES || t.imagery_fails >= MAX_RETRIES);
                let inbound = tile.is_some_and(|t| t.mesh_inbound || t.imagery_inbound);
                let own = tile.is_some_and(|t| t.imagery.is_some());
                if failed {
                    [1.0, 0.15, 0.15]
                } else if inbound {
                    [0.3, 0.55, 1.0]
                } else if own {
                    [0.25, 1.0, 0.4]
                } else {
                    [1.0, 0.85, 0.2]
                }
            }
            3 => match self.imagery_depth(key) {
                0 => [0.25, 1.0, 0.4],
                1 => [1.0, 0.85, 0.2],
                2 => [1.0, 0.5, 0.15],
                d if d > 2 => [1.0, 0.2, 0.2],
                _ => [1.0, 0.2, 1.0],
            },
            4 => {
                let h = slot.wrapping_mul(2_654_435_761);
                [
                    (0.25 + ((h >> 16) & 0xff) as f32 / 340.0).min(1.0),
                    (0.25 + ((h >> 8) & 0xff) as f32 / 340.0).min(1.0),
                    (0.25 + (h & 0xff) as f32 / 340.0).min(1.0),
                ]
            }
            _ => return [0.0, 0.0, 0.0, 0.0],
        };
        [rgb[0], rgb[1], rgb[2], a]
    }

    fn uv_transform(key: TileKey, levels: u8) -> [f32; 4] {
        let scale = 1.0 / (1u32 << levels) as f32;
        let ox = (key.x & ((1u32 << levels) - 1)) as f32 * scale;
        let oy = (key.y & ((1u32 << levels) - 1)) as f32 * scale;
        [scale, scale, ox, oy]
    }

    fn find_imagery(&mut self, key: TileKey, from: u8) -> Option<(f32, [f32; 4])> {
        let mut levels = from;
        loop {
            let anc = key.ancestor(levels);
            if let Some(layer) = self.tiles.get(&anc).and_then(|t| t.imagery) {
                let frame = self.frame;
                if let Some(t) = self.tiles.get_mut(&anc) {
                    t.last_used = frame;
                    t.last_drawn = frame;
                }
                return Some((layer as f32, Self::uv_transform(key, levels)));
            }
            if anc.z == 0 {
                return None;
            }
            levels += 1;
        }
    }

    fn resolve_imagery(&mut self, key: TileKey) -> TileShading {
        let own = self.find_imagery(key, 0);
        let coarse = if key.z > 0 {
            self.find_imagery(key, 1)
        } else {
            None
        };
        let mut shading = TileShading::default();
        match own {
            Some((layer, uvxf)) => {
                let arrived = self.tiles.get(&key).map(|t| t.imagery_at).unwrap_or(0);
                let age = self.frame.saturating_sub(arrived) as f64;
                shading.layer = layer;
                shading.uvxf = uvxf;
                shading.fade = (age / IMAGERY_FADE_FRAMES).clamp(0.0, 1.0) as f32;
                match coarse {
                    Some((prev, prev_uv)) if prev != layer => {
                        shading.prev_layer = prev;
                        shading.prev_uvxf = prev_uv;
                    }
                    _ => {
                        shading.prev_layer = layer;
                        shading.prev_uvxf = uvxf;
                    }
                }
            }
            None => {
                if let Some((prev, prev_uv)) = coarse {
                    shading.prev_layer = prev;
                    shading.prev_uvxf = prev_uv;
                }
                shading.fade = 0.0;
            }
        }
        shading
    }

    fn request_mesh(&mut self, key: TileKey, pool: &mut WorkerPool, priority: f32) {
        if self.frozen {
            return;
        }
        let frame = self.frame;
        let tile = self.tiles.entry(key).or_default();
        tile.last_used = frame;
        if tile.mesh.is_some()
            || tile.mesh_inbound
            || tile.mesh_fails >= MAX_RETRIES
            || tile.mesh_retry_at > frame
        {
            return;
        }
        let (url, uv) = terrain_source(key);
        pool.request(JobKind::Terrain, key, url, uv, priority);
    }

    fn covered_by_ancestor(&self, key: TileKey) -> bool {
        let mut levels = 1;
        loop {
            let anc = key.ancestor(levels);
            if self.tiles.get(&anc).is_some_and(|t| t.imagery.is_some()) {
                return true;
            }
            if anc.z == 0 {
                return false;
            }
            levels += 1;
        }
    }

    fn request_imagery(&mut self, key: TileKey, pool: &mut WorkerPool, priority: f32) {
        if self.frozen {
            return;
        }
        if key.z > imagery_max_zoom() {
            return;
        }
        if self.layer_pressure > ARENA_TRIM_HIGH && self.covered_by_ancestor(key) {
            return;
        }
        let frame = self.frame;
        let tile = self.tiles.entry(key).or_default();
        if tile.imagery.is_some()
            || tile.imagery_inbound
            || tile.imagery_fails >= MAX_RETRIES
            || tile.imagery_retry_at > frame
        {
            return;
        }
        pool.request(
            JobKind::Imagery,
            key,
            imagery_url(key),
            [1.0, 0.0, 0.0],
            priority,
        );
    }

    pub fn integrate(
        &mut self,
        gpu: &Gpu,
        renderer: &mut TerrainRenderer,
        pool: &mut WorkerPool,
        budget: &mut UploadBudget,
        deferred: &mut Vec<crate::stream::Incoming>,
    ) {
        if self.frozen {
            return;
        }
        for msg in pool.drain_inbox() {
            if msg.cancelled {
                continue;
            }
            if matches!(msg.kind, JobKind::Model) {
                self.models.push(msg);
                continue;
            }
            if matches!(msg.kind, JobKind::Icon) {
                self.icons.push(msg);
                continue;
            }
            let tile = self.tiles.entry(msg.key).or_default();
            match msg.kind {
                JobKind::Terrain => tile.mesh_inbound = true,
                JobKind::Imagery => tile.imagery_inbound = true,
                JobKind::Model | JobKind::Icon => {}
            }
            deferred.push(msg);
        }
        self.uploaded_bytes = 0;
        let frame = self.frame;
        let mut leftover = Vec::new();
        for msg in deferred.drain(..) {
            let bytes = msg
                .payload
                .as_ref()
                .map(|p| p.length() as usize)
                .unwrap_or(0);
            match msg.kind {
                JobKind::Model | JobKind::Icon => {}
                JobKind::Terrain => {
                    if !msg.ok {
                        let tile = self.tiles.entry(msg.key).or_default();
                        tile.mesh_inbound = false;
                        tile.mesh_fails += 1;
                        tile.mesh_retry_at = frame + RETRY_DELAY_FRAMES;
                        continue;
                    }
                    if !budget.fits(bytes) {
                        leftover.push(msg);
                        continue;
                    }
                    let Some(slot) = self.acquire_mesh_slot(renderer) else {
                        self.upload_stalls += 1;
                        leftover.push(msg);
                        continue;
                    };
                    budget.take(bytes);
                    let data = msg.payload.as_ref().unwrap().to_vec();
                    renderer.upload_mesh(gpu, slot, &data);
                    self.uploaded_bytes += bytes;
                    let tile = self.tiles.entry(msg.key).or_default();
                    tile.mesh_fails = 0;
                    tile.mesh_inbound = false;
                    let replaced = tile.mesh.replace(MeshRes {
                        slot,
                        center: msg.center,
                        min_height: msg.min_height,
                        max_height: msg.max_height,
                    });
                    if let Some(old) = replaced {
                        renderer.slots.free(old.slot);
                    }
                }
                JobKind::Imagery => {
                    if !msg.ok {
                        let tile = self.tiles.entry(msg.key).or_default();
                        tile.imagery_inbound = false;
                        tile.imagery_fails += 1;
                        tile.imagery_retry_at = frame + RETRY_DELAY_FRAMES;
                        continue;
                    }
                    if !budget.fits(bytes) {
                        leftover.push(msg);
                        continue;
                    }
                    let Some(layer) = self.acquire_imagery_layer(renderer) else {
                        self.upload_stalls += 1;
                        leftover.push(msg);
                        continue;
                    };
                    budget.take(bytes);
                    let data = msg.payload.as_ref().unwrap().to_vec();
                    renderer.upload_imagery(gpu, layer, &data);
                    self.uploaded_bytes += bytes;
                    let tile = self.tiles.entry(msg.key).or_default();
                    tile.imagery_fails = 0;
                    tile.imagery_inbound = false;
                    tile.imagery_at = frame;
                    if let Some(old) = tile.imagery.replace(layer) {
                        renderer.layers.free(old);
                    }
                }
            }
        }
        while leftover.len() > MAX_DEFERRED {
            let Some(msg) = leftover.pop() else { break };
            let tile = self.tiles.entry(msg.key).or_default();
            match msg.kind {
                JobKind::Terrain => tile.mesh_inbound = false,
                JobKind::Imagery => tile.imagery_inbound = false,
                JobKind::Model | JobKind::Icon => {}
            }
            self.dropped_uploads += 1;
        }
        self.deferred_uploads = leftover.len();
        *deferred = leftover;
        self.mesh_pressure = renderer.slots.used() as f32 / renderer.slots.capacity().max(1) as f32;
        self.layer_pressure =
            renderer.layers.used() as f32 / renderer.layers.capacity().max(1) as f32;
        self.retire(renderer);
        self.mesh_pressure = renderer.slots.used() as f32 / renderer.slots.capacity().max(1) as f32;
        self.layer_pressure =
            renderer.layers.used() as f32 / renderer.layers.capacity().max(1) as f32;
        self.prune();
    }

    fn retire(&mut self, renderer: &mut TerrainRenderer) {
        let frame = self.frame;
        let pressed = self.mesh_pressure > ARENA_TRIM_HIGH || self.layer_pressure > ARENA_TRIM_HIGH;
        let age = if !pressed {
            RETIRE_AFTER_FRAMES
        } else if self.camera_jumped {
            RETIRE_JUMP_FRAMES
        } else {
            RETIRE_PRESSURE_FRAMES
        };
        let mut meshes = 0;
        let mut layers = 0;
        for tile in self.tiles.values_mut() {
            if tile.last_used + age > frame {
                continue;
            }
            if let Some(mesh) = tile.mesh.take() {
                renderer.slots.free(mesh.slot);
                tile.split = false;
                meshes += 1;
            }
            if let Some(layer) = tile.imagery.take() {
                renderer.layers.free(layer);
                tile.imagery_at = 0;
                layers += 1;
            }
        }
        self.retired_meshes += meshes;
        self.retired_imagery += layers;
    }

    fn victim(&self, holds: impl Fn(&Tile) -> bool) -> Option<TileKey> {
        let frame = self.frame;
        self.tiles
            .iter()
            .filter(|(_, t)| {
                holds(t) && t.last_used < frame && t.last_drawn + EVICT_PROTECT_FRAMES <= frame
            })
            .min_by_key(|(k, t)| (t.last_drawn, t.last_used, std::cmp::Reverse(k.z)))
            .map(|(k, _)| *k)
    }

    fn acquire_mesh_slot(&mut self, renderer: &mut TerrainRenderer) -> Option<u32> {
        if let Some(slot) = renderer.slots.alloc() {
            return Some(slot);
        }
        let victim = self.victim(|t| t.mesh.is_some())?;
        let slot = {
            let tile = self.tiles.get_mut(&victim)?;
            tile.split = false;
            tile.mesh.take()?.slot
        };
        self.mesh_evictions += 1;
        Some(slot)
    }

    fn acquire_imagery_layer(&mut self, renderer: &mut TerrainRenderer) -> Option<u32> {
        if let Some(layer) = renderer.layers.alloc() {
            return Some(layer);
        }
        let victim = self.victim(|t| t.imagery.is_some())?;
        let layer = {
            let tile = self.tiles.get_mut(&victim)?;
            tile.imagery_at = 0;
            tile.imagery.take()?
        };
        self.imagery_evictions += 1;
        Some(layer)
    }

    fn prune(&mut self) {
        if self.tiles.len() < 8000 {
            return;
        }
        let frame = self.frame;
        self.tiles.retain(|_, t| {
            t.mesh.is_some()
                || t.imagery.is_some()
                || t.mesh_inbound
                || t.imagery_inbound
                || t.last_used + 600 > frame
        });
    }
}
