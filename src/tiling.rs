use crate::math::{geodetic_surface_normal, geodetic_to_ecef, MIN_RADIUS};
use glam::DVec3;
use std::cell::RefCell;

pub const MAX_TILE_HEIGHT: f64 = 9000.0;

pub const DEFAULT_TERRAIN: &str =
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
pub const DEFAULT_TERRAIN_MAX_ZOOM: u8 = 15;

pub const DEFAULT_IMAGERY: &str =
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
pub const DEFAULT_IMAGERY_MAX_ZOOM: u8 = 19;

struct Sources {
    terrain: String,
    terrain_max_zoom: u8,
    imagery: String,
    imagery_max_zoom: u8,
}

thread_local! {
    static SOURCES: RefCell<Sources> = RefCell::new(Sources {
        terrain: DEFAULT_TERRAIN.to_string(),
        terrain_max_zoom: DEFAULT_TERRAIN_MAX_ZOOM,
        imagery: DEFAULT_IMAGERY.to_string(),
        imagery_max_zoom: DEFAULT_IMAGERY_MAX_ZOOM,
    });
}

pub fn set_terrain_source(template: &str, max_zoom: u8) {
    SOURCES.with(|s| {
        let mut s = s.borrow_mut();
        s.terrain = template.to_string();
        s.terrain_max_zoom = max_zoom.min(22);
    });
}

pub fn set_imagery_source(template: &str, max_zoom: u8) {
    SOURCES.with(|s| {
        let mut s = s.borrow_mut();
        s.imagery = template.to_string();
        s.imagery_max_zoom = max_zoom.min(22);
    });
}

pub fn terrain_max_zoom() -> u8 {
    SOURCES.with(|s| s.borrow().terrain_max_zoom)
}

pub fn imagery_max_zoom() -> u8 {
    SOURCES.with(|s| s.borrow().imagery_max_zoom)
}

fn fill(template: &str, k: TileKey) -> String {
    template
        .replace("{z}", &k.z.to_string())
        .replace("{x}", &k.x.to_string())
        .replace("{y}", &k.y.to_string())
}

pub fn terrain_url(k: TileKey) -> String {
    SOURCES.with(|s| fill(&s.borrow().terrain, k))
}

pub fn imagery_url(k: TileKey) -> String {
    SOURCES.with(|s| fill(&s.borrow().imagery, k))
}

pub fn max_tile_zoom() -> u8 {
    terrain_max_zoom().max(imagery_max_zoom()).min(20)
}

pub fn terrain_source(k: TileKey) -> (String, [f64; 3]) {
    let max = terrain_max_zoom();
    if k.z <= max {
        return (terrain_url(k), [1.0, 0.0, 0.0]);
    }
    let levels = k.z - max;
    let src = k.ancestor(levels);
    let span = 1u32 << levels;
    let scale = 1.0 / span as f64;
    let u0 = (k.x & (span - 1)) as f64 * scale;
    let v0 = (k.y & (span - 1)) as f64 * scale;
    (terrain_url(src), [scale, u0, v0])
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct TileKey {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileKey {
    pub fn child(&self, i: u32) -> TileKey {
        TileKey {
            z: self.z + 1,
            x: self.x * 2 + (i & 1),
            y: self.y * 2 + (i >> 1),
        }
    }

    pub fn ancestor(&self, levels: u8) -> TileKey {
        TileKey {
            z: self.z - levels,
            x: self.x >> levels,
            y: self.y >> levels,
        }
    }

    pub fn lon_lat_bounds(&self) -> (f64, f64, f64, f64) {
        let n = (1u32 << self.z) as f64;
        let lon0 = self.x as f64 / n * std::f64::consts::TAU - std::f64::consts::PI;
        let lon1 = (self.x + 1) as f64 / n * std::f64::consts::TAU - std::f64::consts::PI;
        let lat0 = merc_y_to_lat(1.0 - 2.0 * self.y as f64 / n);
        let lat1 = merc_y_to_lat(1.0 - 2.0 * (self.y + 1) as f64 / n);
        (lon0, lat1, lon1, lat0)
    }

    pub fn bounding_sphere(&self) -> (DVec3, f64) {
        let (lon0, lat0, lon1, lat1) = self.lon_lat_bounds();
        let lonm = (lon0 + lon1) * 0.5;
        let latm = (lat0 + lat1) * 0.5;
        let corners = [
            geodetic_to_ecef(lon0, lat0, 0.0),
            geodetic_to_ecef(lon1, lat0, 0.0),
            geodetic_to_ecef(lon0, lat1, 0.0),
            geodetic_to_ecef(lon1, lat1, 0.0),
            geodetic_to_ecef(lonm, lat0, 0.0),
            geodetic_to_ecef(lonm, lat1, 0.0),
            geodetic_to_ecef(lon0, latm, 0.0),
            geodetic_to_ecef(lon1, latm, 0.0),
            geodetic_to_ecef(lonm, latm, 0.0),
        ];
        let mut center = DVec3::ZERO;
        for c in corners.iter() {
            center += *c;
        }
        center /= corners.len() as f64;
        let mut radius: f64 = 0.0;
        for c in corners.iter() {
            radius = radius.max((*c - center).length());
        }
        (center, radius)
    }

    pub fn bounding_box(&self, h_min: f64, h_max: f64) -> (DVec3, [DVec3; 3], [f64; 3]) {
        let (lon0, lat0, lon1, lat1) = self.lon_lat_bounds();
        let lonm = (lon0 + lon1) * 0.5;
        let latm = (lat0 + lat1) * 0.5;
        let (sin_lon, cos_lon) = lonm.sin_cos();
        let up = geodetic_surface_normal(lonm, latm);
        let east = DVec3::new(-sin_lon, cos_lon, 0.0);
        let axes = [east, up.cross(east), up];
        let origin = geodetic_to_ecef(lonm, latm, 0.0);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for lat in [lat0, latm, lat1] {
            for lon in [lon0, lonm, lon1] {
                let base = geodetic_to_ecef(lon, lat, 0.0) - origin;
                let normal = geodetic_surface_normal(lon, lat);
                for h in [h_min, h_max] {
                    let p = base + normal * h;
                    for i in 0..3 {
                        let d = p.dot(axes[i]);
                        lo[i] = lo[i].min(d);
                        hi[i] = hi[i].max(d);
                    }
                }
            }
        }
        let mut center = origin;
        let mut half = [0.0f64; 3];
        for i in 0..3 {
            center += axes[i] * ((lo[i] + hi[i]) * 0.5);
            half[i] = (hi[i] - lo[i]) * 0.5;
        }
        (center, axes, half)
    }

    pub fn ground_extent(&self) -> f64 {
        let (lon0, lat0, lon1, lat1) = self.lon_lat_bounds();
        let mid_lat = (lat0 + lat1) * 0.5;
        let w = (lon1 - lon0) * MIN_RADIUS * mid_lat.cos().abs().max(1e-3);
        let h = (lat1 - lat0) * MIN_RADIUS;
        w.max(h)
    }
}

pub fn tile_at(lon: f64, lat: f64, z: u8) -> TileKey {
    let n = (1u32 << z) as f64;
    let x = ((lon + std::f64::consts::PI) / std::f64::consts::TAU * n).floor();
    let clamped = lat.clamp(-1.4835, 1.4835);
    let merc = (clamped.tan() + 1.0 / clamped.cos()).ln() / std::f64::consts::PI;
    let y = ((1.0 - merc) / 2.0 * n).floor();
    TileKey {
        z,
        x: (x.max(0.0) as u32).min((1u32 << z) - 1),
        y: (y.max(0.0) as u32).min((1u32 << z) - 1),
    }
}

pub fn merc_y_to_lat(t: f64) -> f64 {
    (std::f64::consts::PI * t).sinh().atan()
}
