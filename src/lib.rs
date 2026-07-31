mod camera;
mod gpu;
mod math;
mod model;
mod model_gpu;
mod quadtree;
mod stream;
mod terrain_gpu;
mod terrain_mesh;
mod tiling;
mod vector;
mod worker;

use camera::Camera;
use gpu::{Gpu, UploadBudget};
use model_gpu::ModelRenderer;
use quadtree::TileTree;
use std::cell::RefCell;
use std::rc::Rc;
use stream::{Incoming, WorkerPool};
use terrain_gpu::TerrainRenderer;
use vector::VectorRenderer;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

pub const UPLOAD_BYTES_PER_FRAME: usize = 3 * 1024 * 1024;
pub const MIN_GROUND_CLEARANCE: f64 = 12.0;

#[derive(Default)]
struct Input {
    panning: bool,
    rotating: bool,
    last_x: f64,
    last_y: f64,
    keys: [bool; 8],
}

struct Flight {
    model: usize,
    instance: usize,
    waypoints: Vec<(f64, f64, f64)>,
    speed: f64,
    progress: f64,
    leg: usize,
    scale: f32,
    color: [f32; 4],
}

struct App {
    gpu: Gpu,
    terrain: TerrainRenderer,
    vectors: VectorRenderer,
    models: ModelRenderer,
    model_urls: Vec<String>,
    flights: Vec<Flight>,
    tree: TileTree,
    pool: WorkerPool,
    camera: Camera,
    budget: UploadBudget,
    deferred: Vec<Incoming>,
    canvas: HtmlCanvasElement,
    input: Input,
    resolution_scale: f64,
    frames: u32,
    last_frame_time: f64,
    last_fps_time: f64,
    frame_ms: f64,
    frame_ms_avg: f64,
    fps: f64,
    running: bool,
    uncapped: bool,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no window")
}

fn now() -> f64 {
    window().performance().map(|p| p.now()).unwrap_or(0.0)
}

fn with_app<T>(f: impl FnOnce(&mut App) -> T, fallback: T) -> T {
    APP.with(|slot| match slot.borrow_mut().as_mut() {
        Some(app) => f(app),
        None => fallback,
    })
}

#[wasm_bindgen]
pub async fn start(canvas_id: String) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Warn);

    let document = window().document().ok_or("no document")?;
    let canvas: HtmlCanvasElement = document
        .get_element_by_id(&canvas_id)
        .ok_or("canvas not found")?
        .dyn_into()?;

    let dpr = window().device_pixel_ratio().clamp(1.0, 1.5);
    let width = (canvas.client_width().max(1) as f64 * dpr) as u32;
    let height = (canvas.client_height().max(1) as f64 * dpr) as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let gpu = Gpu::new(canvas.clone()).await.map_err(JsValue::from)?;
    let terrain = TerrainRenderer::new(&gpu);
    let vectors = VectorRenderer::new(&gpu, &terrain.globals);
    let models = ModelRenderer::new(&gpu, &terrain.globals);

    let cores = window().navigator().hardware_concurrency() as usize;
    let worker_count = cores.saturating_sub(1).clamp(2, 8);
    let pool = WorkerPool::new(worker_count, 6)?;

    let app = App {
        gpu,
        terrain,
        vectors,
        models,
        model_urls: Vec::new(),
        flights: Vec::new(),
        tree: TileTree::new(),
        pool,
        camera: Camera::new(),
        budget: UploadBudget::new(UPLOAD_BYTES_PER_FRAME),
        deferred: Vec::new(),
        canvas: canvas.clone(),
        input: Input::default(),
        resolution_scale: 1.0,
        frames: 0,
        last_frame_time: now(),
        last_fps_time: now(),
        frame_ms: 0.0,
        frame_ms_avg: 0.0,
        fps: 0.0,
        running: true,
        uncapped: false,
    };

    APP.with(|slot| *slot.borrow_mut() = Some(app));
    install_input(&canvas)?;
    start_loop();
    Ok(())
}

fn ndc(app: &App, client_x: f64, client_y: f64) -> (f64, f64) {
    let rect = app.canvas.get_bounding_client_rect();
    let w = rect.width().max(1.0);
    let h = rect.height().max(1.0);
    let x = (client_x - rect.left()) / w * 2.0 - 1.0;
    let y = 1.0 - (client_y - rect.top()) / h * 2.0;
    (x, y)
}

fn key_index(code: &str) -> Option<usize> {
    match code {
        "KeyW" | "ArrowUp" => Some(0),
        "KeyS" | "ArrowDown" => Some(1),
        "KeyA" | "ArrowLeft" => Some(2),
        "KeyD" | "ArrowRight" => Some(3),
        "KeyQ" => Some(4),
        "KeyE" => Some(5),
        "KeyR" => Some(6),
        "KeyF" => Some(7),
        _ => None,
    }
}

fn install_input(canvas: &HtmlCanvasElement) -> Result<(), JsValue> {
    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
        e.prevent_default();
        with_app(
            |app| {
                let (x, y) = ndc(app, e.client_x() as f64, e.client_y() as f64);
                app.input.last_x = e.client_x() as f64;
                app.input.last_y = e.client_y() as f64;
                let tilt = e.button() == 2 || e.button() == 1 || e.shift_key() || e.ctrl_key();
                app.input.rotating = tilt;
                app.input.panning = !tilt && e.button() == 0;
                if app.input.panning {
                    app.camera.grab_start(x, y);
                }
            },
            (),
        );
    });
    canvas.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref())?;
    down.forget();

    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |_e| {
        with_app(
            |app| {
                app.input.panning = false;
                app.input.rotating = false;
                app.camera.grab_end();
            },
            (),
        );
    });
    canvas.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointerleave", up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointercancel", up.as_ref().unchecked_ref())?;
    up.forget();

    let motion =
        Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            with_app(
                |app| {
                    let cx = e.client_x() as f64;
                    let cy = e.client_y() as f64;
                    let dx = cx - app.input.last_x;
                    let dy = cy - app.input.last_y;
                    app.input.last_x = cx;
                    app.input.last_y = cy;
                    if app.input.rotating {
                        app.camera.rotate(dx, dy);
                    } else if app.input.panning {
                        let (x, y) = ndc(app, cx, cy);
                        if !app.camera.grab_move(x, y) {
                            let h = app.canvas.get_bounding_client_rect().height();
                            app.camera.orbit_pixels(dx, dy, h.max(1.0));
                        }
                    }
                },
                (),
            );
        });
    canvas.add_event_listener_with_callback("pointermove", motion.as_ref().unchecked_ref())?;
    motion.forget();

    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
        e.prevent_default();
        with_app(
            |app| {
                let (x, y) = ndc(app, e.client_x() as f64, e.client_y() as f64);
                let mut delta = e.delta_y();
                if e.delta_mode() == 1 {
                    delta *= 16.0;
                }
                app.camera.zoom_at(x, y, delta);
            },
            (),
        );
    });
    let opts = web_sys::AddEventListenerOptions::new();
    opts.set_passive(false);
    canvas.add_event_listener_with_callback_and_add_event_listener_options(
        "wheel",
        wheel.as_ref().unchecked_ref(),
        &opts,
    )?;
    wheel.forget();

    let menu = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
        e.prevent_default();
    });
    canvas.add_event_listener_with_callback("contextmenu", menu.as_ref().unchecked_ref())?;
    menu.forget();

    let key_down =
        Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            if let Some(i) = key_index(&e.code()) {
                e.prevent_default();
                with_app(|app| app.input.keys[i] = true, ());
            }
        });
    window().add_event_listener_with_callback("keydown", key_down.as_ref().unchecked_ref())?;
    key_down.forget();

    let key_up =
        Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            if let Some(i) = key_index(&e.code()) {
                with_app(|app| app.input.keys[i] = false, ());
            }
        });
    window().add_event_listener_with_callback("keyup", key_up.as_ref().unchecked_ref())?;
    key_up.forget();

    Ok(())
}

fn raf(cb: &Closure<dyn FnMut()>) {
    let _ = window().request_animation_frame(cb.as_ref().unchecked_ref());
}

fn soon(cb: &Closure<dyn FnMut()>) {
    let _ = window()
        .set_timeout_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 0);
}

fn schedule(cb: &Closure<dyn FnMut()>) {
    if with_app(|app| app.uncapped, false) {
        soon(cb);
    } else {
        raf(cb);
    }
}

fn start_loop() {
    let holder: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let clone = holder.clone();
    *holder.borrow_mut() = Some(Closure::<dyn FnMut()>::new(move || {
        with_app(
            |app| {
                if app.running {
                    app.frame(None);
                }
            },
            (),
        );
        let next = clone.borrow();
        if let Some(cb) = next.as_ref() {
            schedule(cb);
        }
    }));
    let first = holder.borrow();
    if let Some(cb) = first.as_ref() {
        schedule(cb);
    }
    drop(first);
}

impl App {
    fn sync_size(&mut self) {
        let dpr = window().device_pixel_ratio().clamp(1.0, 1.5) * self.resolution_scale;
        let w = (self.canvas.client_width().max(1) as f64 * dpr) as u32;
        let h = (self.canvas.client_height().max(1) as f64 * dpr) as u32;
        if w != self.gpu.config.width || h != self.gpu.config.height {
            self.canvas.set_width(w);
            self.canvas.set_height(h);
            self.gpu.resize(w, h);
        }
    }

    fn apply_keys(&mut self, dt: f64) {
        let k = &self.input.keys;
        let step = dt * 60.0;
        let fwd = (k[0] as i32 - k[1] as i32) as f64 * step;
        let right = (k[3] as i32 - k[2] as i32) as f64 * step;
        if fwd != 0.0 || right != 0.0 {
            self.camera.nudge(fwd * 0.02, right * 0.02);
        }
        let turn = (k[5] as i32 - k[4] as i32) as f64 * step;
        let tilt = (k[7] as i32 - k[6] as i32) as f64 * step;
        if turn != 0.0 || tilt != 0.0 {
            self.camera.rotate(turn * 2.0, tilt * 2.0);
        }
    }

    fn frame(&mut self, dt_override: Option<f64>) {
        let started = now();
        let dt = match dt_override {
            Some(dt) => dt.clamp(0.0, 0.1),
            None => ((started - self.last_frame_time) / 1000.0).clamp(0.0, 0.1),
        };
        self.last_frame_time = started;

        self.sync_size();
        self.apply_keys(dt);
        let (eye_lon, eye_lat) = math::dir_to_geodetic(self.camera.eye);
        self.camera.ground_clearance =
            self.tree.ground_height(eye_lon, eye_lat) + MIN_GROUND_CLEARANCE;
        self.camera.update(self.gpu.aspect(), dt);

        self.budget.reset();
        self.tree.integrate(
            &self.gpu,
            &mut self.terrain,
            &mut self.pool,
            &mut self.budget,
            &mut self.deferred,
        );

        self.integrate_models();
        self.tree
            .select(&self.camera, self.gpu.config.height as f32, &mut self.pool);
        self.advance_flights(dt);
        self.pool.dispatch();

        let sun = self.camera.eye.normalize().as_vec3();
        self.terrain.write_globals(
            &self.gpu,
            self.camera.view_proj,
            sun,
            self.gpu.config.width as f32,
            self.gpu.config.height as f32,
        );
        self.terrain
            .write_instances(&self.gpu, &self.tree.instances);
        self.models.write_instances(&self.gpu);
        self.vectors.update_origins(&self.gpu, self.camera.eye);

        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.gpu
                    .surface
                    .configure(&self.gpu.device, &self.gpu.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.record(&mut encoder, &view, &self.gpu.depth);
        self.gpu.queue.submit(Some(encoder.finish()));
        self.gpu.queue.present(frame);

        self.frames += 1;
        let t = now();
        self.frame_ms = t - started;
        self.frame_ms_avg = if self.frame_ms_avg == 0.0 {
            self.frame_ms
        } else {
            self.frame_ms_avg * 0.9 + self.frame_ms * 0.1
        };
        if t - self.last_fps_time >= 500.0 {
            self.fps = self.frames as f64 * 1000.0 / (t - self.last_fps_time);
            self.frames = 0;
            self.last_fps_time = t;
        }
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.008,
                        g: 0.015,
                        b: 0.04,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.terrain.draw(&mut pass, &self.tree.drawn);
        self.models.draw(&mut pass);
        self.vectors.draw(&mut pass);
    }

    fn integrate_models(&mut self) {
        for slot in 0..self.model_urls.len() {
            if self.models.ready(slot) {
                continue;
            }
            let url = self.model_urls[slot].clone();
            self.pool.request(
                stream::JobKind::Model,
                tiling::TileKey {
                    z: 255,
                    x: slot as u32,
                    y: 0,
                },
                url,
                [1.0, 0.0, 0.0],
                -10_000.0,
            );
        }
        let incoming = std::mem::take(&mut self.tree.models);
        for msg in incoming {
            let slot = msg.key.x as usize;
            let Some(payload) = msg.payload.as_ref() else {
                continue;
            };
            let Some(data) = model::decode(&payload.to_vec()) else {
                continue;
            };
            self.models.upload(&self.gpu, slot, &data);
        }
    }

    fn advance_flights(&mut self, dt: f64) {
        let eye = self.camera.eye;
        for flight in self.flights.iter_mut() {
            if flight.waypoints.len() < 2 {
                continue;
            }
            let a = flight.waypoints[flight.leg];
            let b = flight.waypoints[(flight.leg + 1) % flight.waypoints.len()];
            let start = math::geodetic_to_ecef(a.0.to_radians(), a.1.to_radians(), a.2);
            let end = math::geodetic_to_ecef(b.0.to_radians(), b.1.to_radians(), b.2);
            let leg_length = start.distance(end).max(1.0);
            flight.progress += flight.speed * dt / leg_length;
            while flight.progress >= 1.0 {
                flight.progress -= 1.0;
                flight.leg = (flight.leg + 1) % flight.waypoints.len();
            }
            let a = flight.waypoints[flight.leg];
            let b = flight.waypoints[(flight.leg + 1) % flight.waypoints.len()];
            let start = math::geodetic_to_ecef(a.0.to_radians(), a.1.to_radians(), a.2);
            let end = math::geodetic_to_ecef(b.0.to_radians(), b.1.to_radians(), b.2);
            let dir_start = start.normalize();
            let dir_end = end.normalize();
            let angle = dir_start.dot(dir_end).clamp(-1.0, 1.0).acos();
            let dir = if angle < 1e-6 {
                dir_start
            } else {
                let t = flight.progress;
                (dir_start * ((1.0 - t) * angle).sin() + dir_end * (t * angle).sin()) / angle.sin()
            };
            let dir = dir.normalize();
            let altitude = a.2 + (b.2 - a.2) * flight.progress;
            let position = dir * (math::ellipsoid_radius(dir) + altitude);

            let up = dir.as_vec3();
            let mut forward = (dir_end - dir_start).as_vec3();
            forward -= up * forward.dot(up);
            let forward = model_gpu::up_from(forward);
            let right = forward.cross(up).normalize_or(glam::Vec3::X);
            let basis = glam::Mat3::from_cols(right, forward, up);
            let rotation = glam::Quat::from_mat3(&basis) * model_gpu::model_to_enu();

            let instance = model_gpu::instance_from_transform(
                position,
                eye,
                rotation,
                flight.scale,
                flight.color,
            );
            if let Some(model) = self.models.models.get_mut(flight.model) {
                if flight.instance < model.live.len() {
                    model.live[flight.instance] = instance;
                }
            }
        }
    }
}

#[wasm_bindgen]
pub fn tick() {
    with_app(|app| app.frame(None), ());
}

#[wasm_bindgen]
pub fn tick_step(dt_seconds: f64) {
    with_app(|app| app.frame(Some(dt_seconds)), ());
}

#[wasm_bindgen]
pub fn set_running(running: bool) {
    with_app(|app| app.running = running, ());
}

#[wasm_bindgen]
pub fn set_uncapped(uncapped: bool) {
    with_app(|app| app.uncapped = uncapped, ());
}

#[wasm_bindgen]
pub fn stats() -> String {
    with_app(
        |app| {
            format!(
                "{:.0} fps | {:.1} ms | z{} | {} drawn | {} meshes | {} imagery | {} in flight",
                app.fps,
                app.frame_ms,
                app.tree.max_level_drawn,
                app.tree.drawn.len(),
                app.tree.resident_tiles(),
                app.tree.resident_imagery(),
                app.pool.active_count(),
            )
        },
        "starting".to_string(),
    )
}

#[wasm_bindgen]
pub fn stats_json() -> String {
    with_app(
        |app| {
            format!(
                concat!(
                    "{{\"fps\":{:.2},\"frameMs\":{:.3},\"drawn\":{},\"maxLevel\":{},",
                    "\"meshes\":{},\"imagery\":{},\"meshSlots\":{},\"meshSlotCap\":{},\"imageryLayers\":{},\"imageryLayerCap\":{},",
                    "\"inFlight\":{},\"dispatched\":{},\"dropped\":{},\"uploadedBytes\":{},",
                    "\"deferredUploads\":{},\"groundHeight\":{:.1},\"width\":{},\"height\":{}}}"
                ),
                app.fps,
                app.frame_ms,
                app.tree.drawn.len(),
                app.tree.max_level_drawn,
                app.tree.resident_tiles(),
                app.tree.resident_imagery(),
                app.terrain.slots.used(),
                app.terrain.slots.capacity(),
                app.terrain.layers.used(),
                app.terrain.layers.capacity(),
                app.pool.active_count(),
                app.pool.dispatched,
                app.pool.dropped,
                app.tree.uploaded_bytes,
                app.tree.deferred_uploads,
                app.camera.ground_clearance - MIN_GROUND_CLEARANCE,
                app.gpu.config.width,
                app.gpu.config.height,
            )
        },
        "{}".to_string(),
    )
}

#[wasm_bindgen]
pub fn debug_json() -> String {
    with_app(
        |app| {
            let d = app.tree.debug();
            let (lon, lat) = app.camera.lon_lat();
            let (t_active, i_active, m_active) = app.pool.active_kinds();
            let levels: Vec<String> = d
                .levels
                .iter()
                .map(|(z, s)| {
                    format!(
                        "{{\"z\":{},\"meshes\":{},\"imagery\":{},\"drawn\":{},\"pending\":{}}}",
                        z, s.meshes, s.imagery, s.drawn, s.pending
                    )
                })
                .collect();
            let workers: Vec<String> = app
                .pool
                .worker_load()
                .iter()
                .map(|(ready, flight)| format!("{{\"ready\":{},\"inFlight\":{}}}", ready, flight))
                .collect();
            let batches = app
                .vectors
                .batches
                .iter()
                .fold((0, 0, 0), |acc, b| match b.kind {
                    vector::BatchKind::Polygon { .. } => (acc.0 + 1, acc.1, acc.2),
                    vector::BatchKind::Line { .. } => (acc.0, acc.1 + 1, acc.2),
                    vector::BatchKind::Icon { .. } => (acc.0, acc.1, acc.2 + 1),
                });
            let model_instances: usize = app.models.models.iter().map(|m| m.live.len()).sum();
            let models_ready = (0..app.model_urls.len())
                .filter(|s| app.models.ready(*s))
                .count();
            format!(
                concat!(
                    "{{",
                    "\"frame\":{{\"fps\":{:.1},\"ms\":{:.2},\"msAvg\":{:.2},\"uncappedFps\":{:.0},",
                    "\"uncapped\":{},\"width\":{},\"height\":{},\"scale\":{:.2}}},",
                    "\"camera\":{{\"lon\":{:.5},\"lat\":{:.5},\"alt\":{:.1},\"distance\":{:.1},",
                    "\"heading\":{:.1},\"tilt\":{:.1},\"clearance\":{:.1}}},",
                    "\"draw\":{{\"drawn\":{},\"instances\":{},\"minLevel\":{},\"maxLevel\":{},",
                    "\"splits\":{},\"starvedSplits\":{},\"blockedSplits\":{},\"visited\":{},",
                    "\"culled\":{},\"maxDrawn\":{}}},",
                    "\"tree\":{{\"tiles\":{},\"meshes\":{},\"imagery\":{},\"meshInbound\":{},",
                    "\"imageryInbound\":{},\"meshFailed\":{},\"imageryFailed\":{},\"split\":{},",
                    "\"protected\":{},\"evictableMesh\":{},\"evictableImagery\":{}}},",
                    "\"arena\":{{\"meshUsed\":{},\"meshCap\":{},\"layerUsed\":{},\"layerCap\":{}}},",
                    "\"upload\":{{\"bytes\":{},\"budget\":{},\"spent\":{},\"deferred\":{},",
                    "\"stalls\":{},\"dropped\":{},\"meshEvictions\":{},\"imageryEvictions\":{},",
                    "\"retiredMeshes\":{},\"retiredImagery\":{},\"cameraJumped\":{}}},",
                    "\"pool\":{{\"active\":{},\"terrain\":{},\"imagery\":{},\"model\":{},",
                    "\"queueDepth\":{},\"sent\":{},\"dropped\":{},\"dispatched\":{},\"inbox\":{},\"maxInFlight\":{}}},",
                    "\"scene\":{{\"polygons\":{},\"lines\":{},\"icons\":{},\"models\":{},",
                    "\"modelsReady\":{},\"modelInstances\":{},\"flights\":{}}},",
                    "\"levels\":[{}],\"workers\":[{}]}}"
                ),
                app.fps,
                app.frame_ms,
                app.frame_ms_avg,
                if app.frame_ms_avg > 0.0 {
                    1000.0 / app.frame_ms_avg
                } else {
                    0.0
                },
                app.uncapped,
                app.gpu.config.width,
                app.gpu.config.height,
                app.resolution_scale,
                lon.to_degrees(),
                lat.to_degrees(),
                app.camera.altitude(),
                app.camera.distance,
                app.camera.heading.to_degrees(),
                app.camera.tilt.to_degrees(),
                app.camera.ground_clearance - MIN_GROUND_CLEARANCE,
                app.tree.drawn.len(),
                app.tree.instances.len(),
                app.tree.min_level_drawn,
                app.tree.max_level_drawn,
                app.tree.splits,
                app.tree.starved_splits,
                app.tree.blocked_splits,
                app.tree.visited,
                app.tree.culled,
                terrain_gpu::MAX_DRAWN_TILES,
                d.tiles,
                d.meshes,
                d.imagery,
                d.mesh_inbound,
                d.imagery_inbound,
                d.mesh_failed,
                d.imagery_failed,
                d.split,
                d.protected,
                d.evictable_mesh,
                d.evictable_imagery,
                app.terrain.slots.used(),
                app.terrain.slots.capacity(),
                app.terrain.layers.used(),
                app.terrain.layers.capacity(),
                app.tree.uploaded_bytes,
                app.budget.bytes_per_frame,
                app.budget.spent(),
                app.tree.deferred_uploads,
                app.tree.upload_stalls,
                app.tree.dropped_uploads,
                app.tree.mesh_evictions,
                app.tree.imagery_evictions,
                app.tree.retired_meshes,
                app.tree.retired_imagery,
                app.tree.camera_jumped,
                app.pool.active_count(),
                t_active,
                i_active,
                m_active,
                app.pool.queue_depth,
                app.pool.sent_last,
                app.pool.dropped,
                app.pool.dispatched,
                app.pool.inbox_len(),
                app.pool.max_in_flight(),
                batches.0,
                batches.1,
                batches.2,
                app.models.models.len(),
                models_ready,
                model_instances,
                app.flights.len(),
                levels.join(","),
                workers.join(","),
            )
        },
        "{}".to_string(),
    )
}

#[wasm_bindgen]
pub fn camera_state() -> Vec<f64> {
    with_app(
        |app| {
            let (lon, lat) = app.camera.lon_lat();
            vec![
                lon.to_degrees(),
                lat.to_degrees(),
                app.camera.distance,
                app.camera.heading.to_degrees(),
                app.camera.tilt.to_degrees(),
                app.camera.altitude(),
                app.camera.eye.x,
                app.camera.eye.y,
                app.camera.eye.z,
            ]
        },
        Vec::new(),
    )
}

#[wasm_bindgen]
pub fn pick(ndc_x: f64, ndc_y: f64) -> Vec<f64> {
    with_app(
        |app| match app.camera.pick_dir(ndc_x, ndc_y) {
            Some(dir) => {
                let (lon, lat) = math::dir_to_geodetic(dir);
                vec![lon.to_degrees(), lat.to_degrees()]
            }
            None => Vec::new(),
        },
        Vec::new(),
    )
}

#[wasm_bindgen]
pub fn set_resolution_scale(scale: f64) {
    with_app(|app| app.resolution_scale = scale.clamp(0.25, 1.0), ());
}

#[wasm_bindgen]
pub fn fly_to(lon_deg: f64, lat_deg: f64, distance_m: f64) {
    with_app(
        |app| {
            app.camera
                .set_view(lon_deg.to_radians(), lat_deg.to_radians(), distance_m)
        },
        (),
    );
}

#[wasm_bindgen]
pub fn jump_to(lon_deg: f64, lat_deg: f64, distance_m: f64) {
    with_app(
        |app| {
            app.camera
                .jump_view(lon_deg.to_radians(), lat_deg.to_radians(), distance_m)
        },
        (),
    );
}

#[wasm_bindgen]
pub fn set_orientation(heading_deg: f64, tilt_deg: f64) {
    with_app(
        |app| {
            app.camera
                .set_orientation(heading_deg.to_radians(), tilt_deg.to_radians())
        },
        (),
    );
}

#[wasm_bindgen]
pub fn set_imagery(url_template: String, max_zoom: u32) {
    with_app(
        |app| {
            tiling::set_imagery_source(&url_template, max_zoom as u8);
            app.tree.clear_imagery(&mut app.terrain);
        },
        (),
    );
}

#[wasm_bindgen]
pub fn set_terrain(url_template: String, max_zoom: u32) {
    with_app(
        |app| {
            tiling::set_terrain_source(&url_template, max_zoom as u8);
            app.tree.clear(&mut app.terrain);
        },
        (),
    );
}

#[wasm_bindgen]
pub fn add_polygon(coords: Vec<f64>, base_height: f64, top_height: f64, color: u32) -> i32 {
    with_app(
        |app| {
            app.vectors
                .add_polygon(&app.gpu, &coords, base_height, top_height, color)
                .map(|i| i as i32)
                .unwrap_or(-1)
        },
        -1,
    )
}

#[wasm_bindgen]
pub fn add_line(coords: Vec<f64>, width_px: f32, color: u32) -> i32 {
    with_app(
        |app| {
            app.vectors
                .add_line(&app.gpu, &coords, width_px, color)
                .map(|i| i as i32)
                .unwrap_or(-1)
        },
        -1,
    )
}

#[wasm_bindgen]
pub fn add_icons(coords: Vec<f64>, size_px: f32, color: u32) -> i32 {
    with_app(
        |app| {
            app.vectors
                .add_icons(&app.gpu, &coords, size_px, color)
                .map(|i| i as i32)
                .unwrap_or(-1)
        },
        -1,
    )
}

#[wasm_bindgen]
pub fn vector_batches() -> usize {
    with_app(|app| app.vectors.batches.len(), 0)
}

#[wasm_bindgen]
pub fn add_model(url: String) -> i32 {
    with_app(
        |app| {
            let slot = app.model_urls.len();
            app.model_urls.push(url);
            slot as i32
        },
        -1,
    )
}

#[wasm_bindgen]
pub fn model_ready(slot: usize) -> bool {
    with_app(|app| app.models.ready(slot), false)
}

#[wasm_bindgen]
pub fn model_instances(slot: usize) -> usize {
    with_app(
        |app| {
            app.models
                .models
                .get(slot)
                .map(|m| m.live.len())
                .unwrap_or(0)
        },
        0,
    )
}

#[wasm_bindgen]
pub fn add_flight(
    model: usize,
    waypoints: Vec<f64>,
    speed_mps: f64,
    scale: f32,
    color: u32,
) -> i32 {
    with_app(
        |app| {
            if waypoints.len() < 6 || app.models.models.len() <= model {
                return -1;
            }
            let points: Vec<(f64, f64, f64)> = waypoints
                .chunks_exact(3)
                .map(|c| (c[0], c[1], c[2]))
                .collect();
            let instance = {
                let entry = &mut app.models.models[model];
                entry.live.push(model_gpu::ModelInstance {
                    row0: [1.0, 0.0, 0.0, 0.0],
                    row1: [0.0, 1.0, 0.0, 0.0],
                    row2: [0.0, 0.0, 1.0, 0.0],
                    color: [0.0, 0.0, 0.0, 0.0],
                });
                entry.live.len() - 1
            };
            app.flights.push(Flight {
                model,
                instance,
                waypoints: points,
                speed: speed_mps,
                progress: 0.0,
                leg: 0,
                scale,
                color: [
                    ((color >> 24) & 0xff) as f32 / 255.0,
                    ((color >> 16) & 0xff) as f32 / 255.0,
                    ((color >> 8) & 0xff) as f32 / 255.0,
                    (color & 0xff) as f32 / 255.0,
                ],
            });
            (app.flights.len() - 1) as i32
        },
        -1,
    )
}

#[wasm_bindgen]
pub fn flight_position(index: usize) -> Vec<f64> {
    with_app(
        |app| match app.flights.get(index) {
            Some(f) => {
                let a = f.waypoints[f.leg];
                let b = f.waypoints[(f.leg + 1) % f.waypoints.len()];
                vec![
                    a.0 + (b.0 - a.0) * f.progress,
                    a.1 + (b.1 - a.1) * f.progress,
                    a.2 + (b.2 - a.2) * f.progress,
                ]
            }
            None => Vec::new(),
        },
        Vec::new(),
    )
}

#[wasm_bindgen]
pub fn clear_flights() {
    with_app(
        |app| {
            app.flights.clear();
            for model in app.models.models.iter_mut() {
                model.live.clear();
            }
        },
        (),
    );
}

#[wasm_bindgen]
pub fn clear_vectors() {
    with_app(|app| app.vectors.clear(), ());
}

async fn yield_task() {
    let Ok(channel) = web_sys::MessageChannel::new() else {
        return;
    };
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        channel.port1().set_onmessage(Some(resolve.unchecked_ref()));
        let _ = channel.port2().post_message(&JsValue::NULL);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[wasm_bindgen]
pub async fn read_pixels(size: u32) -> Vec<u8> {
    let size = size.clamp(16, 1024);
    let row_bytes = size * 4;
    let padded_row = row_bytes.div_ceil(256) * 256;
    let buffer = with_app(
        |app| {
            let color = app.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("readback color"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: app.gpu.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = app.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("readback depth"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gpu::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let buffer = app.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (padded_row * size) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let sun = app.camera.eye.normalize().as_vec3();
            app.terrain.write_globals(
                &app.gpu,
                app.camera.view_proj_for_aspect(1.0),
                sun,
                size as f32,
                size as f32,
            );
            let mut encoder =
                app.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("readback"),
                    });
            app.record(&mut encoder, &color_view, &depth_view);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(size),
                    },
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
            );
            app.gpu.queue.submit(Some(encoder.finish()));
            app.terrain.write_globals(
                &app.gpu,
                app.camera.view_proj,
                sun,
                app.gpu.config.width as f32,
                app.gpu.config.height as f32,
            );
            Some((buffer, color, depth))
        },
        None,
    );
    let Some((buffer, _color, _depth)) = buffer else {
        return Vec::new();
    };

    let done = Rc::new(RefCell::new(false));
    let flag = done.clone();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
        *flag.borrow_mut() = true;
    });
    for _ in 0..20000 {
        if *done.borrow() {
            break;
        }
        yield_task().await;
    }
    if !*done.borrow() {
        return Vec::new();
    }
    let bgra = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    let mut rgba = vec![255u8; (row_bytes * size) as usize];
    for row in 0..size as usize {
        let src = &bgra[row * padded_row as usize..][..row_bytes as usize];
        let dst = &mut rgba[row * row_bytes as usize..][..row_bytes as usize];
        for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 255;
        }
    }
    rgba
}
