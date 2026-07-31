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

pub struct Frustum {
    planes: [glam::Vec4; 6],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let r = m.transpose();
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
        Self { planes }
    }

    pub fn sphere_visible(&self, center: Vec3, radius: f32) -> bool {
        for p in self.planes.iter().take(5) {
            if p.x * center.x + p.y * center.y + p.z * center.z + p.w < -radius {
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
