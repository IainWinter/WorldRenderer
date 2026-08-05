use crate::model::{encode, parse_glb};
use crate::terrain_mesh::{
    build_mesh, decode_jpeg_rgba, decode_png_rgba, decode_terrarium, Heightmap,
};
use crate::tiling::TileKey;
use js_sys::{Array, ArrayBuffer, Object, Reflect, Uint8Array};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    AbortController, AbortSignal, Cache, DedicatedWorkerGlobalScope, MessageEvent, Request,
    RequestInit, Response,
};

thread_local! {
    static HEIGHT_CACHE: RefCell<VecDeque<(String, Rc<Heightmap>)>> = RefCell::new(VecDeque::new());
    static EPOCH: Cell<u32> = const { Cell::new(0) };
    static NEXT_ID: Cell<u32> = const { Cell::new(0) };
    static PENDING: RefCell<Vec<(u32, u32, AbortController)>> = const { RefCell::new(Vec::new()) };
    static DISK_MB: Cell<f64> = const { Cell::new(DEFAULT_DISK_MB) };
    static PUTS: Cell<u32> = const { Cell::new(0) };
    static SWEEPING: Cell<bool> = const { Cell::new(false) };
}

const HEIGHT_CACHE_LIMIT: usize = 24;
pub const DISK_CACHE_NAME: &str = "worldrenderer-tiles-v1";
const DEFAULT_DISK_MB: f64 = 16384.0;
const SWEEP_EVERY: u32 = 64;
const SWEEP_TARGET: f64 = 0.9;

fn stale(epoch: u32) -> bool {
    EPOCH.with(|e| e.get()) > epoch
}

fn register(epoch: u32) -> Option<(u32, AbortController)> {
    let ctrl = AbortController::new().ok()?;
    let id = NEXT_ID.with(|n| {
        let id = n.get();
        n.set(id.wrapping_add(1));
        id
    });
    PENDING.with(|p| p.borrow_mut().push((id, epoch, ctrl.clone())));
    Some((id, ctrl))
}

fn unregister(id: u32) {
    PENDING.with(|p| p.borrow_mut().retain(|(i, _, _)| *i != id));
}

fn abort_before(epoch: u32) {
    PENDING.with(|p| {
        p.borrow_mut().retain(|(_, e, ctrl)| {
            if *e < epoch {
                ctrl.abort();
                false
            } else {
                true
            }
        })
    });
}

fn cached_heightmap(url: &str) -> Option<Rc<Heightmap>> {
    HEIGHT_CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| k == url)
            .map(|(_, v)| v.clone())
    })
}

fn store_heightmap(url: &str, hm: Rc<Heightmap>) {
    HEIGHT_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        c.push_back((url.to_string(), hm));
        while c.len() > HEIGHT_CACHE_LIMIT {
            c.pop_front();
        }
    });
}

fn scope() -> DedicatedWorkerGlobalScope {
    js_sys::global().unchecked_into::<DedicatedWorkerGlobalScope>()
}

fn get_u32(o: &JsValue, key: &str) -> u32 {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u32
}

fn get_f64(o: &JsValue, key: &str) -> f64 {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn get_str(o: &JsValue, key: &str) -> String {
    Reflect::get(o, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

fn set(o: &Object, key: &str, v: &JsValue) {
    let _ = Reflect::set(o, &JsValue::from_str(key), v);
}

async fn open_disk_cache() -> Option<Cache> {
    if DISK_MB.with(|m| m.get()) <= 0.0 {
        return None;
    }
    let caches = scope().caches().ok()?;
    JsFuture::from(caches.open(DISK_CACHE_NAME))
        .await
        .ok()?
        .dyn_into::<Cache>()
        .ok()
}

async fn usage_bytes() -> Option<f64> {
    let est = JsFuture::from(scope().navigator().storage().estimate().ok()?)
        .await
        .ok()?;
    Reflect::get(&est, &JsValue::from_str("usage"))
        .ok()?
        .as_f64()
}

async fn sweep_disk_cache() {
    let limit = DISK_MB.with(|m| m.get()) * 1_048_576.0;
    let Some(used) = usage_bytes().await else {
        return;
    };
    if used <= limit {
        return;
    }
    let Some(cache) = open_disk_cache().await else {
        return;
    };
    let Ok(keys) = JsFuture::from(cache.keys()).await else {
        return;
    };
    let keys = Array::from(&keys);
    let total = keys.length();
    if total == 0 {
        return;
    }
    let fraction = 1.0 - (limit * SWEEP_TARGET / used).min(1.0);
    let victims = ((total as f64 * fraction).ceil() as u32).min(total);
    for i in 0..victims {
        let req: Request = keys.get(i).unchecked_into();
        let _ = JsFuture::from(cache.delete_with_request(&req)).await;
    }
}

async fn maybe_sweep() {
    let n = PUTS.with(|p| {
        let v = p.get().wrapping_add(1);
        p.set(v);
        v
    });
    if n % SWEEP_EVERY != 0 || SWEEPING.with(|s| s.get()) {
        return;
    }
    SWEEPING.with(|s| s.set(true));
    sweep_disk_cache().await;
    SWEEPING.with(|s| s.set(false));
}

pub struct Fetched {
    pub bytes: Vec<u8>,
    pub ms: f64,
    pub hit: bool,
}

fn now_ms() -> f64 {
    scope().performance().map(|p| p.now()).unwrap_or(0.0)
}

async fn fetch_bytes(url: &str, signal: Option<&AbortSignal>) -> Result<Fetched, JsValue> {
    let t0 = now_ms();
    let cache = open_disk_cache().await;
    if let Some(cache) = &cache {
        if let Ok(hit) = JsFuture::from(cache.match_with_str(url)).await {
            if let Ok(resp) = hit.dyn_into::<Response>() {
                let buf = JsFuture::from(resp.array_buffer()?).await?;
                return Ok(Fetched {
                    bytes: Uint8Array::new(&buf).to_vec(),
                    ms: now_ms() - t0,
                    hit: true,
                });
            }
        }
    }
    let init = RequestInit::new();
    init.set_signal(signal);
    let resp_value = JsFuture::from(scope().fetch_with_str_and_init(url, &init)).await?;
    let resp: Response = resp_value.dyn_into()?;
    if !resp.ok() {
        return Err(JsValue::from_str("http error"));
    }
    if let Some(cache) = cache {
        if let Ok(copy) = resp.clone() {
            let url = url.to_string();
            wasm_bindgen_futures::spawn_local(async move {
                if JsFuture::from(cache.put_with_str(&url, &copy))
                    .await
                    .is_ok()
                {
                    maybe_sweep().await;
                }
            });
        }
    }
    let buf = JsFuture::from(resp.array_buffer()?).await?;
    let arr = Uint8Array::new(&buf);
    Ok(Fetched {
        bytes: arr.to_vec(),
        ms: now_ms() - t0,
        hit: false,
    })
}

use wasm_bindgen_futures::JsFuture;

fn to_transferable(bytes: &[u8]) -> Uint8Array {
    let arr = Uint8Array::new_with_length(bytes.len() as u32);
    arr.copy_from(bytes);
    arr
}

fn post(msg: &Object, transfer: Option<&ArrayBuffer>) {
    let s = scope();
    match transfer {
        Some(buf) => {
            let list = Array::new();
            list.push(buf);
            let _ = s.post_message_with_transfer(msg, &list);
        }
        None => {
            let _ = s.post_message(msg);
        }
    }
}

fn fail(kind: &str, key: TileKey) {
    report(kind, key, false);
}

fn report(kind: &str, key: TileKey, cancelled: bool) {
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str(kind));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(false));
    set(&out, "cancelled", &JsValue::from_bool(cancelled));
    post(&out, None);
}

fn stamp(out: &Object, fetch_ms: f64, work_ms: f64, hit: bool) {
    set(out, "fms", &JsValue::from_f64(fetch_ms));
    set(out, "wms", &JsValue::from_f64(work_ms));
    set(out, "hit", &JsValue::from_bool(hit));
}

fn ended(kind: &str, key: TileKey, epoch: u32) {
    report(kind, key, stale(epoch));
}

async fn do_terrain(key: TileKey, url: String, uv: [f64; 3], epoch: u32) {
    if stale(epoch) {
        return report("terrain", key, true);
    }
    let mut fetch_ms = 0.0;
    let mut hit = true;
    let mut decode_ms = 0.0;
    let hm = match cached_heightmap(&url) {
        Some(hm) => hm,
        None => {
            let Some((id, ctrl)) = register(epoch) else {
                return fail("terrain", key);
            };
            let fetched = fetch_bytes(&url, Some(&ctrl.signal())).await;
            unregister(id);
            let fetched = match fetched {
                Ok(b) => b,
                Err(_) => return ended("terrain", key, epoch),
            };
            fetch_ms = fetched.ms;
            hit = fetched.hit;
            let t = now_ms();
            let out = match decode_terrarium(&fetched.bytes) {
                Ok(h) => {
                    let hm = Rc::new(h);
                    store_heightmap(&url, hm.clone());
                    hm
                }
                Err(_) => return fail("terrain", key),
            };
            decode_ms = now_ms() - t;
            out
        }
    };
    let t = now_ms();
    let mesh = build_mesh(key, &hm, uv);
    let work_ms = decode_ms + (now_ms() - t);
    let raw: &[u8] = bytemuck::cast_slice(&mesh.vertices);
    let arr = to_transferable(raw);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("terrain"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "cx", &JsValue::from_f64(mesh.center.x));
    set(&out, "cy", &JsValue::from_f64(mesh.center.y));
    set(&out, "cz", &JsValue::from_f64(mesh.center.z));
    set(&out, "hmin", &JsValue::from_f64(mesh.min_height as f64));
    set(&out, "hmax", &JsValue::from_f64(mesh.max_height as f64));
    set(
        &out,
        "heights",
        &js_sys::Float32Array::from(mesh.heights.as_slice()),
    );
    stamp(&out, fetch_ms, work_ms, hit);
    set(&out, "verts", &arr);
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

async fn do_imagery(key: TileKey, url: String, epoch: u32) {
    if stale(epoch) {
        return report("imagery", key, true);
    }
    let Some((id, ctrl)) = register(epoch) else {
        return fail("imagery", key);
    };
    let fetched = fetch_bytes(&url, Some(&ctrl.signal())).await;
    unregister(id);
    let fetched = match fetched {
        Ok(b) => b,
        Err(_) => return ended("imagery", key, epoch),
    };
    let t = now_ms();
    let (w, h, rgba) = match decode_jpeg_rgba(&fetched.bytes) {
        Ok(v) => v,
        Err(_) => return fail("imagery", key),
    };
    let work_ms = now_ms() - t;
    let arr = to_transferable(&rgba);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("imagery"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "w", &JsValue::from_f64(w as f64));
    set(&out, "h", &JsValue::from_f64(h as f64));
    set(&out, "pixels", &arr);
    stamp(&out, fetched.ms, work_ms, fetched.hit);
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

async fn do_model(key: TileKey, url: String) {
    let fetched = match fetch_bytes(&url, None).await {
        Ok(b) => b,
        Err(_) => return fail("model", key),
    };
    let t = now_ms();
    let data = match parse_glb(&fetched.bytes) {
        Ok(d) => d,
        Err(_) => return fail("model", key),
    };
    let blob = encode(&data);
    let work_ms = now_ms() - t;
    let arr = to_transferable(&blob);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("model"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "verts", &arr);
    stamp(&out, fetched.ms, work_ms, fetched.hit);
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

async fn do_icon(key: TileKey, url: String) {
    let fetched = match fetch_bytes(&url, None).await {
        Ok(b) => b,
        Err(_) => return fail("icon", key),
    };
    let t = now_ms();
    let (w, h, rgba) =
        match decode_png_rgba(&fetched.bytes).or_else(|_| decode_jpeg_rgba(&fetched.bytes)) {
            Ok(v) => v,
            Err(_) => return fail("icon", key),
        };
    let mut blob = Vec::with_capacity(8 + rgba.len());
    blob.extend_from_slice(&w.to_le_bytes());
    blob.extend_from_slice(&h.to_le_bytes());
    blob.extend_from_slice(&rgba);
    let work_ms = now_ms() - t;
    let arr = to_transferable(&blob);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("icon"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "verts", &arr);
    stamp(&out, fetched.ms, work_ms, fetched.hit);
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

#[wasm_bindgen]
pub fn worker_main() {
    console_error_panic_hook::set_once();
    let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let data = e.data();
        let kind = get_str(&data, "kind");
        let epoch = get_u32(&data, "e");
        if kind == "cachelimit" {
            DISK_MB.with(|m| m.set(get_f64(&data, "mb").max(0.0)));
            wasm_bindgen_futures::spawn_local(sweep_disk_cache());
            return;
        }
        if kind == "cancel" {
            EPOCH.with(|c| c.set(epoch));
            abort_before(epoch);
            return;
        }
        let key = TileKey {
            z: get_u32(&data, "z") as u8,
            x: get_u32(&data, "x"),
            y: get_u32(&data, "y"),
        };
        let url = get_str(&data, "url");
        let uv = [
            get_f64(&data, "us"),
            get_f64(&data, "u0"),
            get_f64(&data, "v0"),
        ];
        match kind.as_str() {
            "terrain" => wasm_bindgen_futures::spawn_local(do_terrain(key, url, uv, epoch)),
            "imagery" => wasm_bindgen_futures::spawn_local(do_imagery(key, url, epoch)),
            "model" => wasm_bindgen_futures::spawn_local(do_model(key, url)),
            "icon" => wasm_bindgen_futures::spawn_local(do_icon(key, url)),
            _ => {}
        }
    });
    scope().set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();

    let ready = Object::new();
    set(&ready, "kind", &JsValue::from_str("ready"));
    post(&ready, None);
}
