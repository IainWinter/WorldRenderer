use crate::tiling::TileKey;
use glam::DVec3;
use js_sys::{Object, Reflect, Uint8Array};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum JobKind {
    Terrain,
    Imagery,
    Model,
}

impl JobKind {
    fn tag(&self) -> &'static str {
        match self {
            JobKind::Terrain => "terrain",
            JobKind::Imagery => "imagery",
            JobKind::Model => "model",
        }
    }
}

pub struct Incoming {
    pub kind: JobKind,
    pub key: TileKey,
    pub ok: bool,
    pub center: DVec3,
    pub max_height: f32,
    pub payload: Option<Uint8Array>,
}

struct Request {
    kind: JobKind,
    key: TileKey,
    url: String,
    uv: [f64; 3],
    priority: f32,
}

pub struct WorkerPool {
    workers: Vec<Worker>,
    ready: Rc<RefCell<Vec<bool>>>,
    in_flight: Rc<RefCell<Vec<u32>>>,
    active: Rc<RefCell<HashSet<(JobKind, TileKey)>>>,
    inbox: Rc<RefCell<Vec<Incoming>>>,
    queue: Vec<Request>,
    queued: HashSet<(JobKind, TileKey)>,
    max_in_flight: u32,
    pub dispatched: u32,
    pub dropped: u32,
    pub queue_depth: u32,
    pub sent_last: u32,
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
        let active: Rc<RefCell<HashSet<(JobKind, TileKey)>>> =
            Rc::new(RefCell::new(HashSet::new()));
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
                active_c.borrow_mut().remove(&(job, key));
                let ok = get_bool(&data, "ok");
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
                    center: DVec3::new(
                        get_f64(&data, "cx"),
                        get_f64(&data, "cy"),
                        get_f64(&data, "cz"),
                    ),
                    max_height: get_f64(&data, "hmax") as f32,
                    payload,
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
            inbox,
            queue: Vec::new(),
            queued: HashSet::new(),
            max_in_flight,
            dispatched: 0,
            dropped: 0,
            queue_depth: 0,
            sent_last: 0,
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
        for (kind, _) in active.iter() {
            match kind {
                JobKind::Terrain => out.0 += 1,
                JobKind::Imagery => out.1 += 1,
                JobKind::Model => out.2 += 1,
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
        let id = (kind, key);
        if self.queued.contains(&id) || self.active.borrow().contains(&id) {
            return;
        }
        self.queued.insert(id);
        self.queue.push(Request {
            kind,
            key,
            url,
            uv,
            priority,
        });
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
            let _ = Reflect::set(&msg, &"us".into(), &JsValue::from_f64(req.uv[0]));
            let _ = Reflect::set(&msg, &"u0".into(), &JsValue::from_f64(req.uv[1]));
            let _ = Reflect::set(&msg, &"v0".into(), &JsValue::from_f64(req.uv[2]));
            if self.workers[slot].post_message(&msg).is_ok() {
                self.in_flight.borrow_mut()[slot] += 1;
                self.active.borrow_mut().insert((req.kind, req.key));
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
