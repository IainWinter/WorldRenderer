use crate::math::{
    dir_to_geodetic, reverse_z_infinite_perspective, surface_point, MIN_RADIUS, WGS84_A, WGS84_B,
    WGS84_E2,
};
use glam::{DQuat, DVec3, Mat4};

pub const MIN_DISTANCE: f64 = 30.0;
pub const MAX_DISTANCE: f64 = 40_000_000.0;
pub const MAX_LAT: f64 = 1.4835;
pub const MAX_TILT: f64 = 1.45;

fn base_frame() -> DQuat {
    DQuat::from_axis_angle(DVec3::ONE.normalize(), std::f64::consts::TAU / 3.0)
}

pub fn anchor_at(lon: f64, geodetic_lat: f64) -> DQuat {
    let geocentric = ((1.0 - WGS84_E2) * geodetic_lat.tan()).atan();
    DQuat::from_rotation_z(lon) * DQuat::from_rotation_y(-geocentric) * base_frame()
}

fn canonical(anchor: DQuat) -> DQuat {
    let (lon, lat) = dir_to_geodetic(anchor * DVec3::Z);
    anchor_at(lon, lat.clamp(-MAX_LAT, MAX_LAT))
}

#[derive(Clone)]
pub struct Camera {
    anchor: DQuat,
    target_anchor: DQuat,
    pub distance: f64,
    target_distance: f64,
    pub heading: f64,
    target_heading: f64,
    pub tilt: f64,
    target_tilt: f64,
    pub fov_y: f32,
    pub eye: DVec3,
    pub orientation: DQuat,
    pub view_proj: Mat4,
    view: Mat4,
    pub near: f32,
    pub aspect: f32,
    pub ground_clearance: f64,
    grab: Option<DVec3>,
}

impl Camera {
    pub fn new() -> Self {
        let anchor = anchor_at(0.0, 20f64.to_radians());
        Self {
            anchor,
            target_anchor: anchor,
            distance: 22_000_000.0,
            target_distance: 22_000_000.0,
            heading: 0.0,
            target_heading: 0.0,
            tilt: 0.0,
            target_tilt: 0.0,
            fov_y: 50f64.to_radians() as f32,
            eye: DVec3::new(0.0, 0.0, 20_000_000.0),
            orientation: DQuat::IDENTITY,
            view_proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            near: 1.0,
            aspect: 1.0,
            ground_clearance: 0.0,
            grab: None,
        }
    }

    pub fn up(&self) -> DVec3 {
        self.anchor * DVec3::Z
    }

    pub fn target(&self) -> DVec3 {
        surface_point(self.up())
    }

    pub fn lon_lat(&self) -> (f64, f64) {
        dir_to_geodetic(self.up())
    }

    pub fn altitude(&self) -> f64 {
        self.eye.length() - crate::math::ellipsoid_radius(self.eye)
    }

    pub fn set_view(&mut self, lon: f64, lat: f64, distance: f64) {
        self.target_anchor = anchor_at(lon, lat.clamp(-MAX_LAT, MAX_LAT));
        self.target_distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
    }

    pub fn jump_view(&mut self, lon: f64, lat: f64, distance: f64) {
        self.set_view(lon, lat, distance);
        self.anchor = self.target_anchor;
        self.distance = self.target_distance;
    }

    pub fn set_orientation(&mut self, heading: f64, tilt: f64) {
        self.target_heading = heading;
        self.target_tilt = tilt.clamp(0.0, MAX_TILT);
    }

    pub fn update(&mut self, aspect: f32, dt: f64) {
        self.aspect = aspect;
        self.target_tilt = self.target_tilt.clamp(0.0, MAX_TILT);
        self.target_distance = self.target_distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.target_anchor = canonical(self.target_anchor);

        let k = 1.0 - (-dt.clamp(0.0, 0.1) * 14.0).exp();
        self.anchor = canonical(self.anchor.slerp(self.target_anchor, k));
        self.distance *= (self.target_distance / self.distance).powf(k);
        self.heading += (self.target_heading - self.heading) * k;
        self.tilt += (self.target_tilt - self.tilt) * k;
        self.distance = self.distance.clamp(MIN_DISTANCE, MAX_DISTANCE);

        self.place_eye();

        self.view = Mat4::from_quat(self.orientation.as_quat()).transpose();
        let ground_dist = self.altitude().max(1.0);
        self.near = (ground_dist * 0.002).clamp(0.5, 20_000.0) as f32;
        self.view_proj = reverse_z_infinite_perspective(self.fov_y, aspect, self.near) * self.view;
    }

    pub fn view_proj_for_aspect(&self, aspect: f32) -> Mat4 {
        reverse_z_infinite_perspective(self.fov_y, aspect, self.near) * self.view
    }

    pub fn ray(&self, ndc_x: f64, ndc_y: f64) -> DVec3 {
        let t = (self.fov_y as f64 * 0.5).tan();
        let local = DVec3::new(ndc_x * t * self.aspect as f64, ndc_y * t, -1.0);
        (self.orientation * local).normalize()
    }

    pub fn pick_dir(&self, ndc_x: f64, ndc_y: f64) -> Option<DVec3> {
        let dir = self.ray(ndc_x, ndc_y);
        let scale = DVec3::new(1.0 / WGS84_A, 1.0 / WGS84_A, 1.0 / WGS84_B);
        let o = self.eye * scale;
        let d = dir * scale;
        let a = d.dot(d);
        let b = 2.0 * o.dot(d);
        let c = o.dot(o) - 1.0;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let t0 = (-b - sq) / (2.0 * a);
        let t1 = (-b + sq) / (2.0 * a);
        let t = if t0 > 0.0 {
            t0
        } else if t1 > 0.0 {
            t1
        } else {
            return None;
        };
        Some((self.eye + dir * t).normalize())
    }

    fn limb_dir(&self, ndc_x: f64, ndc_y: f64) -> DVec3 {
        let dir = self.ray(ndc_x, ndc_y);
        let t = (-self.eye).dot(dir).max(0.0);
        (self.eye + dir * t).normalize()
    }

    fn sphere_dir(&self, ndc_x: f64, ndc_y: f64) -> DVec3 {
        self.pick_dir(ndc_x, ndc_y)
            .unwrap_or_else(|| self.limb_dir(ndc_x, ndc_y))
    }

    pub fn grab_start(&mut self, ndc_x: f64, ndc_y: f64) {
        self.grab = Some(self.sphere_dir(ndc_x, ndc_y));
    }

    pub fn grab_end(&mut self) {
        self.grab = None;
    }

    fn place_eye(&mut self) {
        self.orientation =
            self.anchor * DQuat::from_rotation_z(self.heading) * DQuat::from_rotation_x(self.tilt);
        self.eye = self.target() + (self.orientation * DVec3::Z) * self.distance;
        let eye_dir = self.eye.normalize();
        let floor = crate::math::ellipsoid_radius(eye_dir) + self.ground_clearance;
        if self.eye.length() < floor {
            self.eye = eye_dir * floor;
        }
    }

    pub fn grab_move(&mut self, ndc_x: f64, ndc_y: f64) -> bool {
        let Some(anchor_dir) = self.grab else {
            return false;
        };
        let mut moved = false;
        for _ in 0..8 {
            let current = self.sphere_dir(ndc_x, ndc_y);
            if current.dot(anchor_dir) > 1.0 - 1e-14 {
                break;
            }
            let rotation = DQuat::from_rotation_arc(current, anchor_dir);
            self.anchor = canonical(rotation * self.anchor);
            self.target_anchor = self.anchor;
            self.place_eye();
            moved = true;
        }
        moved
    }

    pub fn zoom_at(&mut self, ndc_x: f64, ndc_y: f64, delta: f64) {
        let before = self.target_distance;
        let after = (before * (delta * 0.0015).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.target_distance = after;
        if after >= before {
            return;
        }
        let Some(cursor) = self.pick_dir(ndc_x, ndc_y) else {
            return;
        };
        let t = ((1.0 - after / before) * 1.2).clamp(0.0, 0.85);
        let full = DQuat::from_rotation_arc(self.target_anchor * DVec3::Z, cursor);
        let partial = DQuat::IDENTITY.slerp(full, t);
        self.target_anchor = canonical(partial * self.target_anchor);
    }

    pub fn orbit_pixels(&mut self, dx: f64, dy: f64, screen_h: f64) {
        let t = (self.fov_y as f64 * 0.5).tan();
        let scale = (self.distance / MIN_RADIUS).min(1.0) * 2.0 * t / screen_h;
        self.move_ground(dy * scale, -dx * scale);
    }

    pub fn move_ground(&mut self, forward: f64, right: f64) {
        if forward == 0.0 && right == 0.0 {
            return;
        }
        let frame = self.target_anchor * DQuat::from_rotation_z(self.heading);
        let east = frame * DVec3::X;
        let north = frame * DVec3::Y;
        let rotation =
            DQuat::from_axis_angle(east, -forward) * DQuat::from_axis_angle(north, right);
        self.target_anchor = canonical(rotation * self.target_anchor);
    }

    pub fn nudge(&mut self, forward: f64, right: f64) {
        let scale = (self.distance / MIN_RADIUS).min(1.0) * 0.35;
        self.move_ground(forward * scale, right * scale);
    }

    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.target_heading -= dx * 0.006;
        self.target_tilt = (self.target_tilt - dy * 0.006).clamp(0.0, MAX_TILT);
    }
}
