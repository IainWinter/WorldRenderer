use crate::tiling::TileKey;
use glam::DVec3;
use js_sys::{Object, Reflect, Uint8Array};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum JobKind {
    Terrain,
    Imagery,
    Model,
    Icon,
}

impl JobKind {
    fn tag(&self) -> &'static str {
        match self {
            JobKind::Terrain => "terrain",
            JobKind::Imagery => "imagery",
            JobKind::Model => "model",
            JobKind::Icon => "icon",
        }
    }

    pub fn label(&self) -> &'static str {
        self.tag()
    }

    fn index(&self) -> usize {
        match self {
            JobKind::Terrain => 0,
            JobKind::Imagery => 1,
            JobKind::Model => 2,
            JobKind::Icon => 3,
        }
    }
}

pub const JOB_KINDS: [JobKind; 4] = [
    JobKind::Terrain,
    JobKind::Imagery,
    JobKind::Model,
    JobKind::Icon,
];

const LOAD_EMA: f64 = 0.15;

#[derive(Default, Clone, Copy)]
pub struct LoadStat {
    pub n: u32,
    pub failed: u32,
    pub hits: u32,
    pub last_ms: f64,
    pub avg_ms: f64,
    pub max_ms: f64,
    pub fetch_ms: f64,
    pub work_ms: f64,
    pub queue_ms: f64,
}

impl LoadStat {
    fn record(&mut self, total: f64, fetch: f64, work: f64, hit: bool, ok: bool) {
        self.n += 1;
        if !ok {
            self.failed += 1;
        }
        if hit {
            self.hits += 1;
        }
        self.last_ms = total;
        self.max_ms = self.max_ms.max(total);
        let blend = |old: f64, new: f64| {
            if old == 0.0 {
                new
            } else {
                old + (new - old) * LOAD_EMA
            }
        };
        self.avg_ms = blend(self.avg_ms, total);
        self.fetch_ms = blend(self.fetch_ms, fetch);
        self.work_ms = blend(self.work_ms, work);
        self.queue_ms = blend(self.queue_ms, (total - fetch - work).max(0.0));
    }
}

fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

pub struct Incoming {
    pub kind: JobKind,
    pub key: TileKey,
    pub ok: bool,
    pub cancelled: bool,
    pub center: DVec3,
    pub min_height: f32,
    pub max_height: f32,
    pub heights: Vec<f32>,
    pub payload: Option<Uint8Array>,
    pub load_ms: f32,
}

struct Request {
    kind: JobKind,
    key: TileKey,
    url: String,
    extra_url: String,
    uv: [f64; 3],
    priority: f32,
}

pub struct WorkerPool {
    workers: Vec<Worker>,
    ready: Rc<RefCell<Vec<bool>>>,
    in_flight: Rc<RefCell<Vec<u32>>>,
    active: Rc<RefCell<HashMap<(JobKind, TileKey), f64>>>,
    loads: Rc<RefCell<[LoadStat; 4]>>,
    inbox: Rc<RefCell<Vec<Incoming>>>,
    queue: Vec<Request>,
    queued: HashSet<(JobKind, TileKey)>,
    max_in_flight: u32,
    epoch: u32,
    pub dispatched: u32,
    pub dropped: u32,
    pub queue_depth: u32,
    pub sent_last: u32,
    pub cancelled: u32,
    _handlers: Vec<Closure<dyn FnMut(MessageEvent)>>,
}

fn get_f64(o: &JsValue, key: &str) -> f64 {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn get_bool(o: &JsValue, key: &str) -> bool {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn get_str(o: &JsValue, key: &str) -> String {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

impl WorkerPool {
    pub fn new(count: usize, max_in_flight: u32) -> Result<Self, JsValue> {
        let ready = Rc::new(RefCell::new(vec![false; count]));
        let in_flight = Rc::new(RefCell::new(vec![0u32; count]));
        let active: Rc<RefCell<HashMap<(JobKind, TileKey), f64>>> =
            Rc::new(RefCell::new(HashMap::new()));
        let loads: Rc<RefCell<[LoadStat; 4]>> = Rc::new(RefCell::new([LoadStat::default(); 4]));
        let inbox = Rc::new(RefCell::new(Vec::new()));
        let mut workers = Vec::with_capacity(count);
        let mut handlers = Vec::with_capacity(count);

        for i in 0..count {
            let opts = WorkerOptions::new();
            opts.set_type(WorkerType::Module);
            let worker = Worker::new_with_options("./worker.js", &opts)?;

            let ready_c = ready.clone();
            let flight_c = in_flight.clone();
            let active_c = active.clone();
            let loads_c = loads.clone();
            let inbox_c = inbox.clone();
            let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
                let data = e.data();
                let kind = get_str(&data, "kind");
                if kind == "ready" {
                    ready_c.borrow_mut()[i] = true;
                    return;
                }
                let job = match kind.as_str() {
                    "terrain" => JobKind::Terrain,
                    "imagery" => JobKind::Imagery,
                    "model" => JobKind::Model,
                    "icon" => JobKind::Icon,
                    _ => return,
                };
                {
                    let mut flight = flight_c.borrow_mut();
                    flight[i] = flight[i].saturating_sub(1);
                }
                let key = TileKey {
                    z: get_f64(&data, "z") as u8,
                    x: get_f64(&data, "x") as u32,
                    y: get_f64(&data, "y") as u32,
                };
                let started = active_c.borrow_mut().remove(&(job, key));
                let ok = get_bool(&data, "ok");
                let mut load_ms = 0.0;
                if let Some(t0) = started {
                    load_ms = now_ms() - t0;
                    if !get_bool(&data, "cancelled") {
                        loads_c.borrow_mut()[job.index()].record(
                            load_ms,
                            get_f64(&data, "fms"),
                            get_f64(&data, "wms"),
                            get_bool(&data, "hit"),
                            ok,
                        );
                    }
                }
                let payload = if ok {
                    let field = if matches!(job, JobKind::Imagery) {
                        "pixels"
                    } else {
                        "verts"
                    };
                    Reflect::get(&data, &JsValue::from_str(field))
                        .ok()
                        .and_then(|v| v.dyn_into::<Uint8Array>().ok())
                } else {
                    None
                };
                inbox_c.borrow_mut().push(Incoming {
                    kind: job,
                    key,
                    ok: ok && payload.is_some(),
                    cancelled: get_bool(&data, "cancelled"),
                    center: DVec3::new(
                        get_f64(&data, "cx"),
                        get_f64(&data, "cy"),
                        get_f64(&data, "cz"),
                    ),
                    min_height: get_f64(&data, "hmin") as f32,
                    max_height: get_f64(&data, "hmax") as f32,
                    heights: Reflect::get(&data, &JsValue::from_str("heights"))
                        .ok()
                        .and_then(|v| v.dyn_into::<js_sys::Float32Array>().ok())
                        .map(|a| a.to_vec())
                        .unwrap_or_default(),
                    payload,
                    load_ms: load_ms as f32,
                });
            });
            worker.set_onmessage(Some(handler.as_ref().unchecked_ref()));
            handlers.push(handler);
            workers.push(worker);
        }

        Ok(Self {
            workers,
            ready,
            in_flight,
            active,
            loads,
            inbox,
            queue: Vec::new(),
            queued: HashSet::new(),
            max_in_flight,
            epoch: 0,
            dispatched: 0,
            dropped: 0,
            queue_depth: 0,
            sent_last: 0,
            cancelled: 0,
            _handlers: handlers,
        })
    }

    pub fn active_count(&self) -> usize {
        self.active.borrow().len()
    }

    pub fn inbox_len(&self) -> usize {
        self.inbox.borrow().len()
    }

    pub fn max_in_flight(&self) -> u32 {
        self.max_in_flight
    }

    pub fn worker_load(&self) -> Vec<(bool, u32)> {
        let ready = self.ready.borrow();
        let flight = self.in_flight.borrow();
        (0..self.workers.len())
            .map(|i| (ready[i], flight[i]))
            .collect()
    }

    pub fn active_kinds(&self) -> (usize, usize, usize) {
        let active = self.active.borrow();
        let mut out = (0, 0, 0);
        for (kind, _) in active.keys() {
            match kind {
                JobKind::Terrain => out.0 += 1,
                JobKind::Imagery => out.1 += 1,
                JobKind::Model | JobKind::Icon => out.2 += 1,
            }
        }
        out
    }

    pub fn request(
        &mut self,
        kind: JobKind,
        key: TileKey,
        url: String,
        uv: [f64; 3],
        priority: f32,
    ) {
        self.request_with(kind, key, url, String::new(), uv, priority);
    }

    pub fn request_with(
        &mut self,
        kind: JobKind,
        key: TileKey,
        url: String,
        extra_url: String,
        uv: [f64; 3],
        priority: f32,
    ) {
        let id = (kind, key);
        if self.queued.contains(&id) || self.active.borrow().contains_key(&id) {
            return;
        }
        self.queued.insert(id);
        self.queue.push(Request {
            kind,
            key,
            url,
            extra_url,
            uv,
            priority,
        });
    }

    pub fn load_stats(&self) -> [LoadStat; 4] {
        *self.loads.borrow()
    }

    pub fn set_cache_limit_mb(&self, mb: f64) {
        let msg = Object::new();
        let _ = Reflect::set(&msg, &"kind".into(), &JsValue::from_str("cachelimit"));
        let _ = Reflect::set(&msg, &"mb".into(), &JsValue::from_f64(mb));
        for worker in self.workers.iter() {
            let _ = worker.post_message(&msg);
        }
    }

    pub fn cancel_stale(&mut self) {
        let live = self
            .active
            .borrow()
            .keys()
            .filter(|(kind, _)| matches!(kind, JobKind::Terrain | JobKind::Imagery))
            .count();
        if live == 0 {
            return;
        }
        self.epoch += 1;
        let msg = Object::new();
        let _ = Reflect::set(&msg, &"kind".into(), &JsValue::from_str("cancel"));
        let _ = Reflect::set(&msg, &"e".into(), &JsValue::from_f64(self.epoch as f64));
        for worker in self.workers.iter() {
            let _ = worker.post_message(&msg);
        }
        self.active
            .borrow_mut()
            .retain(|&(kind, _), _| matches!(kind, JobKind::Model | JobKind::Icon));
        self.cancelled += live as u32;
    }

    pub fn dispatch(&mut self) {
        self.queue_depth = self.queue.len() as u32;
        if self.queue.is_empty() {
            self.sent_last = 0;
            self.dropped = 0;
            return;
        }
        self.queue
            .sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());
        let mut sent = 0;
        for req in self.queue.iter() {
            let slot = {
                let ready = self.ready.borrow();
                let flight = self.in_flight.borrow();
                (0..self.workers.len())
                    .filter(|i| ready[*i] && flight[*i] < self.max_in_flight)
                    .min_by_key(|i| flight[*i])
            };
            let Some(slot) = slot else { break };
            let msg = Object::new();
            let _ = Reflect::set(&msg, &"kind".into(), &JsValue::from_str(req.kind.tag()));
            let _ = Reflect::set(&msg, &"z".into(), &JsValue::from_f64(req.key.z as f64));
            let _ = Reflect::set(&msg, &"x".into(), &JsValue::from_f64(req.key.x as f64));
            let _ = Reflect::set(&msg, &"y".into(), &JsValue::from_f64(req.key.y as f64));
            let _ = Reflect::set(&msg, &"url".into(), &JsValue::from_str(&req.url));
            if !req.extra_url.is_empty() {
                let _ = Reflect::set(&msg, &"url2".into(), &JsValue::from_str(&req.extra_url));
            }
            let _ = Reflect::set(&msg, &"e".into(), &JsValue::from_f64(self.epoch as f64));
            let _ = Reflect::set(&msg, &"us".into(), &JsValue::from_f64(req.uv[0]));
            let _ = Reflect::set(&msg, &"u0".into(), &JsValue::from_f64(req.uv[1]));
            let _ = Reflect::set(&msg, &"v0".into(), &JsValue::from_f64(req.uv[2]));
            if self.workers[slot].post_message(&msg).is_ok() {
                self.in_flight.borrow_mut()[slot] += 1;
                self.active
                    .borrow_mut()
                    .insert((req.kind, req.key), now_ms());
                self.dispatched += 1;
                sent += 1;
            }
        }
        self.sent_last = sent as u32;
        self.dropped = (self.queue.len() - sent) as u32;
        self.queue.clear();
        self.queued.clear();
    }

    pub fn drain_inbox(&mut self) -> Vec<Incoming> {
        std::mem::take(&mut *self.inbox.borrow_mut())
    }
}
