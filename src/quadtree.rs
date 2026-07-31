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
    split: bool,
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
    last_eye: DVec3,
    pub models: Vec<crate::stream::Incoming>,
}

impl TileTree {
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            frame: 0,
            drawn: Vec::new(),
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
            last_eye: DVec3::ZERO,
            models: Vec::new(),
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
        self.instances.clear();
    }

    pub fn clear_imagery(&mut self, renderer: &mut TerrainRenderer) {
        for tile in self.tiles.values_mut() {
            if let Some(layer) = tile.imagery.take() {
                renderer.layers.free(layer);
            }
            tile.imagery_fails = 0;
            tile.imagery_retry_at = 0;
        }
    }

    pub fn select(&mut self, camera: &Camera, screen_h: f32, pool: &mut WorkerPool) {
        self.frame += 1;
        self.drawn.clear();
        self.instances.clear();
        self.max_level_drawn = 0;
        self.min_level_drawn = u8::MAX;
        self.splits = 0;
        self.starved_splits = 0;
        self.visited = 0;
        self.culled = 0;
        self.blocked_splits = 0;

        let moved = (camera.eye - self.last_eye).length();
        self.camera_jumped =
            self.last_eye != DVec3::ZERO && moved > camera.distance * CAMERA_JUMP_FRACTION;
        self.last_eye = camera.eye;

        let frustum = Frustum::from_view_proj(camera.view_proj);
        let k = screen_h as f64 / (2.0 * (camera.fov_y as f64 * 0.5).tan());
        self.prefetch_focus(camera, k, pool);
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
            self.request_mesh(key, pool, -1000.0 + z as f32);
            self.request_imagery(key, pool, -1000.0 + z as f32);
            if key.ground_extent() / dist * k <= TILE_PIXEL_TARGET {
                break;
            }
        }
    }

    fn height_pad(&self, key: TileKey) -> f64 {
        let mut levels = 0;
        loop {
            let anc = key.ancestor(levels);
            if let Some(mesh) = self.tiles.get(&anc).and_then(|t| t.mesh.as_ref()) {
                return (mesh.max_height as f64).max(0.0) + HEIGHT_PAD_SLACK;
            }
            if anc.z == 0 {
                return MAX_TILE_HEIGHT;
            }
            levels += 1;
        }
    }

    fn visible(&self, key: TileKey, camera: &Camera, frustum: &Frustum) -> Option<f64> {
        let (center, radius) = key.bounding_sphere();
        if !horizon_visible(camera.eye, center, radius + MAX_TILE_HEIGHT) {
            return None;
        }
        let rel = (center - camera.eye).as_vec3();
        if !frustum.sphere_visible(rel, (radius + self.height_pad(key)) as f32) {
            return None;
        }
        Some(((center - camera.eye).length() - radius).max(1.0))
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
        frustum: &Frustum,
        k: f64,
        pool: &mut WorkerPool,
    ) {
        let Some(dist) = self.visible(key, camera, frustum) else {
            self.culled += 1;
            return;
        };
        self.visited += 1;

        let screen_px = key.ground_extent() / dist * k;
        let was_split = self.tiles.get(&key).is_some_and(|t| t.split);
        let threshold = if was_split {
            TILE_PIXEL_TARGET * SPLIT_HYSTERESIS
        } else {
            TILE_PIXEL_TARGET
        };
        let want_children = key.z < max_tile_zoom() && screen_px > threshold;

        if want_children {
            let children: [TileKey; 4] = [key.child(0), key.child(1), key.child(2), key.child(3)];
            let all_ready = children.iter().all(|c| {
                self.renderable(*c)
                    || self.exhausted(*c)
                    || self.visible(*c, camera, frustum).is_none()
            });
            if all_ready {
                self.splits += 1;
                self.tiles.entry(key).or_default().split = true;
                for c in children {
                    self.visit(c, camera, frustum, k, pool);
                }
                return;
            }
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
        if let Some(tile) = self.tiles.get_mut(&key) {
            tile.last_drawn = frame;
        }
        self.request_imagery(key, pool, dist as f32);
        let origin = (center - camera.eye).as_vec3();
        let shade = self.resolve_imagery(key);
        let morph = (2.0 * (1.0 - screen_px / TILE_PIXEL_TARGET)).clamp(0.0, 1.0) as f32;
        self.max_level_drawn = self.max_level_drawn.max(key.z);
        self.min_level_drawn = self.min_level_drawn.min(key.z);
        self.drawn.push(slot);
        self.instances.push(TileInstance {
            origin: origin.to_array(),
            morph,
            uvxf: shade.uvxf,
            prev_uvxf: shade.prev_uvxf,
            layers: [shade.layer, shade.prev_layer],
            fade: shade.fade,
            pad: 0.0,
        });
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
        let mut shading = TileShading::default();
        match own {
            Some((layer, uvxf)) => {
                let arrived = self.tiles.get(&key).map(|t| t.imagery_at).unwrap_or(0);
                let age = self.frame.saturating_sub(arrived) as f64;
                shading.layer = layer;
                shading.uvxf = uvxf;
                shading.fade = (age / IMAGERY_FADE_FRAMES).clamp(0.0, 1.0) as f32;
                if shading.fade < 1.0 && key.z > 0 {
                    if let Some((prev, prev_uv)) = self.find_imagery(key, 1) {
                        shading.prev_layer = prev;
                        shading.prev_uvxf = prev_uv;
                    }
                } else {
                    shading.prev_layer = layer;
                    shading.prev_uvxf = uvxf;
                }
            }
            None => {
                if let Some((prev, prev_uv)) = self.find_imagery(key, 1) {
                    shading.prev_layer = prev;
                    shading.prev_uvxf = prev_uv;
                }
                shading.fade = 0.0;
            }
        }
        shading
    }

    fn request_mesh(&mut self, key: TileKey, pool: &mut WorkerPool, priority: f32) {
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

    fn request_imagery(&mut self, key: TileKey, pool: &mut WorkerPool, priority: f32) {
        if key.z > imagery_max_zoom() {
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
        for msg in pool.drain_inbox() {
            if matches!(msg.kind, JobKind::Model) {
                self.models.push(msg);
                continue;
            }
            let tile = self.tiles.entry(msg.key).or_default();
            match msg.kind {
                JobKind::Terrain => tile.mesh_inbound = true,
                JobKind::Imagery => tile.imagery_inbound = true,
                JobKind::Model => {}
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
                JobKind::Model => {}
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
                JobKind::Model => {}
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
        let age = if self.camera_jumped {
            RETIRE_JUMP_FRAMES
        } else if pressed {
            RETIRE_PRESSURE_FRAMES
        } else {
            RETIRE_AFTER_FRAMES
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
