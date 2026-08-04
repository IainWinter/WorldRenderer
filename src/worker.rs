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
    AbortController, AbortSignal, DedicatedWorkerGlobalScope, MessageEvent, RequestInit, Response,
};

thread_local! {
    static HEIGHT_CACHE: RefCell<VecDeque<(String, Rc<Heightmap>)>> = RefCell::new(VecDeque::new());
    static EPOCH: Cell<u32> = const { Cell::new(0) };
    static NEXT_ID: Cell<u32> = const { Cell::new(0) };
    static PENDING: RefCell<Vec<(u32, u32, AbortController)>> = const { RefCell::new(Vec::new()) };
}

const HEIGHT_CACHE_LIMIT: usize = 24;

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

async fn fetch_bytes(url: &str, signal: Option<&AbortSignal>) -> Result<Vec<u8>, JsValue> {
    let init = RequestInit::new();
    init.set_signal(signal);
    let resp_value = JsFuture::from(scope().fetch_with_str_and_init(url, &init)).await?;
    let resp: Response = resp_value.dyn_into()?;
    if !resp.ok() {
        return Err(JsValue::from_str("http error"));
    }
    let buf = JsFuture::from(resp.array_buffer()?).await?;
    let arr = Uint8Array::new(&buf);
    Ok(arr.to_vec())
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

fn ended(kind: &str, key: TileKey, epoch: u32) {
    report(kind, key, stale(epoch));
}

async fn do_terrain(key: TileKey, url: String, uv: [f64; 3], epoch: u32) {
    if stale(epoch) {
        return report("terrain", key, true);
    }
    let hm = match cached_heightmap(&url) {
        Some(hm) => hm,
        None => {
            let Some((id, ctrl)) = register(epoch) else {
                return fail("terrain", key);
            };
            let fetched = fetch_bytes(&url, Some(&ctrl.signal())).await;
            unregister(id);
            let bytes = match fetched {
                Ok(b) => b,
                Err(_) => return ended("terrain", key, epoch),
            };
            match decode_terrarium(&bytes) {
                Ok(h) => {
                    let hm = Rc::new(h);
                    store_heightmap(&url, hm.clone());
                    hm
                }
                Err(_) => return fail("terrain", key),
            }
        }
    };
    let mesh = build_mesh(key, &hm, uv);
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
    let bytes = match fetched {
        Ok(b) => b,
        Err(_) => return ended("imagery", key, epoch),
    };
    let (w, h, rgba) = match decode_jpeg_rgba(&bytes) {
        Ok(v) => v,
        Err(_) => return fail("imagery", key),
    };
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
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

async fn do_model(key: TileKey, url: String) {
    let bytes = match fetch_bytes(&url, None).await {
        Ok(b) => b,
        Err(_) => return fail("model", key),
    };
    let data = match parse_glb(&bytes) {
        Ok(d) => d,
        Err(_) => return fail("model", key),
    };
    let blob = encode(&data);
    let arr = to_transferable(&blob);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("model"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "verts", &arr);
    let buffer = arr.buffer();
    post(&out, Some(&buffer));
}

async fn do_icon(key: TileKey, url: String) {
    let bytes = match fetch_bytes(&url, None).await {
        Ok(b) => b,
        Err(_) => return fail("icon", key),
    };
    let (w, h, rgba) = match decode_png_rgba(&bytes).or_else(|_| decode_jpeg_rgba(&bytes)) {
        Ok(v) => v,
        Err(_) => return fail("icon", key),
    };
    let mut blob = Vec::with_capacity(8 + rgba.len());
    blob.extend_from_slice(&w.to_le_bytes());
    blob.extend_from_slice(&h.to_le_bytes());
    blob.extend_from_slice(&rgba);
    let arr = to_transferable(&blob);
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str("icon"));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(true));
    set(&out, "verts", &arr);
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
