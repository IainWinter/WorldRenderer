use crate::camera::Camera;
use crate::gpu::Gpu;
use crate::math::{
    dir_to_geodetic, ellipsoid_entry, ellipsoid_entry_at, ellipsoid_radius,
    geodetic_surface_normal, geodetic_to_ecef,
};
use crate::quadtree::TileTree;
use crate::tiling::MAX_TILE_HEIGHT;
use crate::vector::VectorRenderer;
use glam::DVec3;

const MARCH_STEPS: usize = 8192;
const REFINE_STEPS: usize = 32;
const BALL_RADIUS_PX: f64 = 20.0;

pub struct ToolContext<'a> {
    pub gpu: &'a Gpu,
    pub camera: &'a Camera,
    pub tree: &'a TileTree,
    pub vectors: &'a mut VectorRenderer,
}

fn shell_span(eye: DVec3, dir: DVec3) -> Option<(f64, f64)> {
    let radius = crate::math::WGS84_A + MAX_TILE_HEIGHT;
    let b = eye.dot(dir);
    let c = eye.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let exit = -b + sq;
    if exit <= 0.0 {
        return None;
    }
    Some(((-b - sq).max(0.0), exit))
}

impl ToolContext<'_> {
    fn terrain_gap(&self, point: DVec3) -> (f64, f64, f64, f64) {
        let (lon, lat) = dir_to_geodetic(point);
        let ground = self.tree.ground_height(lon, lat);
        let gap = point.length() - (ellipsoid_radius(point) + ground);
        (gap, lon, lat, ground)
    }

    pub fn screen_radius(&self, lon: f64, lat: f64, height: f64, pixels: f64) -> f64 {
        let point = geodetic_to_ecef(lon, lat, height);
        let distance = (point - self.camera.eye).length().max(1.0);
        let half_height = (self.gpu.config.height.max(1) as f64) * 0.5;
        let tan = (self.camera.fov_y as f64 * 0.5).tan();
        (distance * tan * pixels / half_height).max(0.05)
    }

    pub fn pick_ground(&self, ndc_x: f64, ndc_y: f64) -> Option<(f64, f64, f64)> {
        let eye = self.camera.eye;
        let dir = self.camera.ray(ndc_x, ndc_y);
        let (start, shell_exit) = shell_span(eye, dir)?;
        let end = ellipsoid_entry(eye, dir).unwrap_or(shell_exit).max(start);

        let span = end - start;
        let min_step = (span / MARCH_STEPS as f64).max(0.25);
        let max_step = (span / 32.0).max(min_step);

        let mut prev_t = start;
        let (mut prev_gap, ..) = self.terrain_gap(eye + dir * start);
        let mut t = start;
        let mut hit = prev_gap <= 0.0;
        while !hit && t < end {
            t = (t + (prev_gap * 0.2).clamp(min_step, max_step)).min(end);
            let (gap, ..) = self.terrain_gap(eye + dir * t);
            if gap <= 0.0 {
                hit = true;
                break;
            }
            prev_t = t;
            prev_gap = gap;
        }

        if !hit {
            let surface = ellipsoid_entry(eye, dir)?;
            let (_, lon, lat, ground) = self.terrain_gap(eye + dir * surface);
            return Some((lon, lat, ground));
        }

        let (mut lo, mut hi) = (prev_t, t);
        for _ in 0..REFINE_STEPS {
            let mid = (lo + hi) * 0.5;
            let (gap, ..) = self.terrain_gap(eye + dir * mid);
            if gap <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let (_, lon, lat, ground) = self.terrain_gap(eye + dir * hi);
        Some((lon, lat, ground))
    }
}

pub trait Tool {
    fn pointer_down(
        &mut self,
        _ctx: &mut ToolContext,
        _ndc_x: f64,
        _ndc_y: f64,
        _button: i16,
    ) -> bool {
        false
    }

    fn pointer_move(&mut self, _ctx: &mut ToolContext, _ndc_x: f64, _ndc_y: f64) -> bool {
        false
    }

    fn pointer_up(&mut self, _ctx: &mut ToolContext) -> bool {
        false
    }

    fn position(&self) -> Option<(f64, f64, f64)> {
        None
    }

    fn positions(&self) -> Vec<f64> {
        Vec::new()
    }

    fn detach(&mut self, _ctx: &mut ToolContext) {}
}

pub struct PlaceTool {
    balls: Vec<Ball>,
    active: Option<usize>,
    color: u32,
}

struct Ball {
    marker: usize,
    radius: f64,
    position: (f64, f64, f64),
}

impl PlaceTool {
    pub fn new() -> Self {
        Self {
            balls: Vec::new(),
            active: None,
            color: 0xff2a2aff,
        }
    }

    fn add(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64) {
        let Some((lon, lat, ground)) = ctx.pick_ground(ndc_x, ndc_y) else {
            return;
        };
        let radius = ctx.screen_radius(lon, lat, ground, BALL_RADIUS_PX);
        let center = geodetic_to_ecef(lon, lat, ground + radius);
        let marker = ctx.vectors.add_marker(ctx.gpu, center, radius, self.color);
        self.balls.push(Ball {
            marker,
            radius,
            position: (lon, lat, ground),
        });
        self.active = Some(self.balls.len() - 1);
    }

    fn drag(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64) {
        let Some(index) = self.active else {
            return;
        };
        let Some((lon, lat, ground)) = ctx.pick_ground(ndc_x, ndc_y) else {
            return;
        };
        let Some(ball) = self.balls.get_mut(index) else {
            return;
        };
        let radius = ctx.screen_radius(lon, lat, ground, BALL_RADIUS_PX);
        ball.position = (lon, lat, ground);
        ball.radius = radius;
        let marker = ball.marker;
        let center = geodetic_to_ecef(lon, lat, ground + radius);
        ctx.vectors
            .set_marker(ctx.gpu, marker, center, radius, self.color);
    }
}

impl Tool for PlaceTool {
    fn pointer_down(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64, button: i16) -> bool {
        if button != 0 {
            return false;
        }
        self.add(ctx, ndc_x, ndc_y);
        true
    }

    fn pointer_move(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64) -> bool {
        if self.active.is_none() {
            return false;
        }
        self.drag(ctx, ndc_x, ndc_y);
        true
    }

    fn pointer_up(&mut self, _ctx: &mut ToolContext) -> bool {
        self.active.take().is_some()
    }

    fn position(&self) -> Option<(f64, f64, f64)> {
        self.balls.last().map(|b| b.position)
    }

    fn positions(&self) -> Vec<f64> {
        self.balls
            .iter()
            .flat_map(|b| {
                [
                    b.position.0.to_degrees(),
                    b.position.1.to_degrees(),
                    b.position.2,
                ]
            })
            .collect()
    }

    fn detach(&mut self, ctx: &mut ToolContext) {
        self.balls.clear();
        self.active = None;
        ctx.vectors.clear_markers();
    }
}

struct Frame {
    anchor: DVec3,
    right: DVec3,
    forward: DVec3,
    height: f64,
}

pub struct SelectTool {
    frame: Option<Frame>,
    extent: (f64, f64),
    dragging: bool,
}

impl SelectTool {
    pub fn new() -> Self {
        Self {
            frame: None,
            extent: (0.0, 0.0),
            dragging: false,
        }
    }

    fn refresh(&self, ctx: &mut ToolContext) {
        let Some(frame) = self.frame.as_ref() else {
            ctx.vectors.clear_selection();
            return;
        };
        let (u, v) = self.extent;
        if u.abs() < 1.0 || v.abs() < 1.0 {
            ctx.vectors.clear_selection();
            return;
        }
        ctx.vectors.set_selection(
            ctx.gpu,
            frame.anchor,
            frame.right,
            frame.forward,
            u,
            v,
            frame.height,
        );
    }

    fn corners(&self) -> Option<[DVec3; 4]> {
        let frame = self.frame.as_ref()?;
        let (u, v) = self.extent;
        let r = frame.right * u;
        let f = frame.forward * v;
        Some([
            frame.anchor,
            frame.anchor + r,
            frame.anchor + r + f,
            frame.anchor + f,
        ])
    }
}

impl Tool for SelectTool {
    fn pointer_down(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64, button: i16) -> bool {
        if button != 0 {
            return false;
        }
        let Some((lon, lat, ground)) = ctx.pick_ground(ndc_x, ndc_y) else {
            return false;
        };
        let up = geodetic_surface_normal(lon, lat);
        let screen_x = ctx.camera.ray(0.5, 0.0) - ctx.camera.ray(-0.5, 0.0);
        let screen_y = ctx.camera.ray(0.0, 0.5) - ctx.camera.ray(0.0, -0.5);
        let flatten = |v: DVec3| v - up * v.dot(up);
        let mut right = flatten(screen_x);
        if right.length_squared() < 1e-12 {
            right = flatten(DVec3::Z.cross(up));
        }
        let right = right.normalize();
        let mut forward = flatten(screen_y);
        forward -= right * forward.dot(right);
        let forward = if forward.length_squared() < 1e-12 {
            up.cross(right)
        } else {
            forward.normalize()
        };
        self.frame = Some(Frame {
            anchor: geodetic_to_ecef(lon, lat, ground),
            right,
            forward,
            height: ground,
        });
        self.extent = (0.0, 0.0);
        self.dragging = true;
        ctx.vectors.clear_selection();
        true
    }

    fn pointer_move(&mut self, ctx: &mut ToolContext, ndc_x: f64, ndc_y: f64) -> bool {
        if !self.dragging {
            return false;
        }
        let Some(frame) = self.frame.as_ref() else {
            return true;
        };
        let dir = ctx.camera.ray(ndc_x, ndc_y);
        let Some(t) = ellipsoid_entry_at(ctx.camera.eye, dir, frame.height) else {
            return true;
        };
        let offset = ctx.camera.eye + dir * t - frame.anchor;
        self.extent = (offset.dot(frame.right), offset.dot(frame.forward));
        self.refresh(ctx);
        true
    }

    fn pointer_up(&mut self, ctx: &mut ToolContext) -> bool {
        if !self.dragging {
            return false;
        }
        self.dragging = false;
        self.refresh(ctx);
        true
    }

    fn position(&self) -> Option<(f64, f64, f64)> {
        let frame = self.frame.as_ref()?;
        let (lon, lat) = dir_to_geodetic(frame.anchor);
        Some((lon, lat, frame.height))
    }

    fn positions(&self) -> Vec<f64> {
        let Some(corners) = self.corners() else {
            return Vec::new();
        };
        let (u, v) = self.extent;
        let mut out: Vec<f64> = corners
            .iter()
            .flat_map(|c| {
                let (lon, lat) = dir_to_geodetic(*c);
                [lon.to_degrees(), lat.to_degrees()]
            })
            .collect();
        out.push(u.abs());
        out.push(v.abs());
        out
    }

    fn detach(&mut self, ctx: &mut ToolContext) {
        self.frame = None;
        self.extent = (0.0, 0.0);
        self.dragging = false;
        ctx.vectors.clear_selection();
    }
}

pub fn make(name: &str) -> Option<Box<dyn Tool>> {
    match name {
        "place" => Some(Box::new(PlaceTool::new())),
        "select" => Some(Box::new(SelectTool::new())),
        _ => None,
    }
}
