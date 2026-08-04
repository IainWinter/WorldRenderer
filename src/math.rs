use glam::{DVec3, Mat4, Vec3};

pub const WGS84_A: f64 = 6378137.0;
pub const WGS84_B: f64 = 6356752.314245;
pub const WGS84_E2: f64 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A);
pub const MIN_RADIUS: f64 = WGS84_B;

pub fn geodetic_to_ecef(lon_rad: f64, lat_rad: f64, height: f64) -> DVec3 {
    let (sin_lat, cos_lat) = lat_rad.sin_cos();
    let (sin_lon, cos_lon) = lon_rad.sin_cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    DVec3::new(
        (n + height) * cos_lat * cos_lon,
        (n + height) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + height) * sin_lat,
    )
}

pub fn geodetic_surface_normal(lon_rad: f64, lat_rad: f64) -> DVec3 {
    let (sin_lat, cos_lat) = lat_rad.sin_cos();
    let (sin_lon, cos_lon) = lon_rad.sin_cos();
    DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
}

pub fn ellipsoid_radius(dir: DVec3) -> f64 {
    let d = dir.normalize();
    let inv = (d.x * d.x + d.y * d.y) / (WGS84_A * WGS84_A) + d.z * d.z / (WGS84_B * WGS84_B);
    inv.sqrt().recip()
}

pub fn surface_point(dir: DVec3) -> DVec3 {
    dir.normalize() * ellipsoid_radius(dir)
}

pub fn dir_to_geodetic(dir: DVec3) -> (f64, f64) {
    let d = dir.normalize();
    let lon = d.y.atan2(d.x);
    let geocentric = d.z.clamp(-1.0, 1.0).asin();
    let lat = (geocentric.tan() / (1.0 - WGS84_E2)).atan();
    (lon, lat)
}

pub fn oct_encode(n: Vec3) -> [i16; 2] {
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(1e-12);
    let (mut u, mut v) = (n.x, n.y);
    if n.z < 0.0 {
        let su = if n.x >= 0.0 { 1.0 } else { -1.0 };
        let sv = if n.y >= 0.0 { 1.0 } else { -1.0 };
        let t = ((1.0 - n.y.abs()) * su, (1.0 - n.x.abs()) * sv);
        u = t.0;
        v = t.1;
    }
    [
        (u.clamp(-1.0, 1.0) * 32767.0).round() as i16,
        (v.clamp(-1.0, 1.0) * 32767.0).round() as i16,
    ]
}

pub fn reverse_z_infinite_perspective(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    Mat4::from_cols_array(&[
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        near,
        0.0,
    ])
}

pub fn ellipsoid_entry(eye: DVec3, dir: DVec3) -> Option<f64> {
    let scale = DVec3::new(1.0 / WGS84_A, 1.0 / WGS84_A, 1.0 / WGS84_B);
    let o = eye * scale;
    let d = dir * scale;
    let qa = d.dot(d);
    let qb = 2.0 * o.dot(d);
    let qc = o.dot(o) - 1.0;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 {
        return None;
    }
    let t = (-qb - disc.sqrt()) / (2.0 * qa);
    (t > 0.0).then_some(t)
}

pub fn horizon_distance(eye: DVec3) -> f64 {
    let r = eye.length();
    (r * r - MIN_RADIUS * MIN_RADIUS).max(0.0).sqrt()
        + (2.0 * MIN_RADIUS * crate::tiling::MAX_TILE_HEIGHT).sqrt()
}

pub fn ray_far_distance(eye: DVec3, dir: DVec3) -> f64 {
    ellipsoid_entry(eye, dir).unwrap_or_else(|| horizon_distance(eye))
}

pub struct Frustum {
    planes: [glam::Vec4; 6],
    corners: [Vec3; 8],
}

impl Frustum {
    pub fn from_camera(cam: &crate::camera::Camera) -> Self {
        let r = cam.view_proj.transpose();
        let mut planes = [
            r.w_axis + r.x_axis,
            r.w_axis - r.x_axis,
            r.w_axis + r.y_axis,
            r.w_axis - r.y_axis,
            r.w_axis + r.z_axis,
            r.w_axis - r.z_axis,
        ];
        for p in planes.iter_mut() {
            let len = Vec3::new(p.x, p.y, p.z).length().max(1e-9);
            *p /= len;
        }

        let forward = cam.ray(0.0, 0.0);
        const STEPS: usize = 8;
        const NDC: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut far = cam.near as f64 * 2.0;
        let mut open_sky = false;
        for i in 0..4 {
            let (x0, y0) = NDC[i];
            let (x1, y1) = NDC[(i + 1) % 4];
            for j in 0..STEPS {
                let t = j as f64 / STEPS as f64;
                let dir = cam.ray(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
                match ellipsoid_entry(cam.eye, dir) {
                    Some(hit) => far = far.max(hit * dir.dot(forward)),
                    None => open_sky = true,
                }
            }
        }
        if open_sky {
            far = far.max(horizon_distance(cam.eye));
        }
        far *= 1.02;
        planes[5] = glam::Vec4::new(
            -forward.x as f32,
            -forward.y as f32,
            -forward.z as f32,
            far as f32,
        );

        let mut corners = [Vec3::ZERO; 8];
        for (i, (x, y)) in NDC.iter().enumerate() {
            let dir = cam.ray(*x, *y);
            let axial = dir.dot(forward).max(1e-6);
            corners[i] = (dir * (cam.near as f64 / axial)).as_vec3();
            corners[i + 4] = (dir * (far / axial)).as_vec3();
        }

        Self { planes, corners }
    }

    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes
            .iter()
            .all(|pl| pl.x * p.x + pl.y * p.y + pl.z * p.z + pl.w >= 0.0)
    }

    pub fn box_visible(&self, center: Vec3, axes: [Vec3; 3], half: [f32; 3]) -> bool {
        for p in self.planes.iter() {
            let n = Vec3::new(p.x, p.y, p.z);
            let r = half[0] * n.dot(axes[0]).abs()
                + half[1] * n.dot(axes[1]).abs()
                + half[2] * n.dot(axes[2]).abs();
            if n.dot(center) + p.w < -r {
                return false;
            }
        }
        for i in 0..3 {
            let d = center.dot(axes[i]);
            let (lo, hi) = (d - half[i], d + half[i]);
            let mut cmin = f32::INFINITY;
            let mut cmax = f32::NEG_INFINITY;
            for c in self.corners.iter() {
                let t = c.dot(axes[i]);
                cmin = cmin.min(t);
                cmax = cmax.max(t);
            }
            if cmax < lo || cmin > hi {
                return false;
            }
        }
        true
    }
}

pub fn horizon_visible(camera: DVec3, center: DVec3, radius: f64) -> bool {
    let d = camera.length();
    if d <= MIN_RADIUS {
        return true;
    }
    center.dot(camera) + radius * d > MIN_RADIUS * MIN_RADIUS
}
