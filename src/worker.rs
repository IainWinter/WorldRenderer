use crate::model::{encode, parse_glb};
use crate::terrain_mesh::{build_mesh, decode_jpeg_rgba, decode_terrarium, Heightmap};
use crate::tiling::TileKey;
use js_sys::{Array, ArrayBuffer, Object, Reflect, Uint8Array};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, Response};

thread_local! {
    static HEIGHT_CACHE: RefCell<VecDeque<(String, Rc<Heightmap>)>> = RefCell::new(VecDeque::new());
}

const HEIGHT_CACHE_LIMIT: usize = 24;

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

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let resp_value = JsFuture::from(scope().fetch_with_str(url)).await?;
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
    let out = Object::new();
    set(&out, "kind", &JsValue::from_str(kind));
    set(&out, "z", &JsValue::from_f64(key.z as f64));
    set(&out, "x", &JsValue::from_f64(key.x as f64));
    set(&out, "y", &JsValue::from_f64(key.y as f64));
    set(&out, "ok", &JsValue::from_bool(false));
    post(&out, None);
}

async fn do_terrain(key: TileKey, url: String, uv: [f64; 3]) {
    let hm = match cached_heightmap(&url) {
        Some(hm) => hm,
        None => {
            let bytes = match fetch_bytes(&url).await {
                Ok(b) => b,
                Err(_) => return fail("terrain", key),
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

async fn do_imagery(key: TileKey, url: String) {
    let bytes = match fetch_bytes(&url).await {
        Ok(b) => b,
        Err(_) => return fail("imagery", key),
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
    let bytes = match fetch_bytes(&url).await {
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

#[wasm_bindgen]
pub fn worker_main() {
    console_error_panic_hook::set_once();
    let handler = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
        let data = e.data();
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
        match get_str(&data, "kind").as_str() {
            "terrain" => wasm_bindgen_futures::spawn_local(do_terrain(key, url, uv)),
            "imagery" => wasm_bindgen_futures::spawn_local(do_imagery(key, url)),
            "model" => wasm_bindgen_futures::spawn_local(do_model(key, url)),
            _ => {}
        }
    });
    scope().set_onmessage(Some(handler.as_ref().unchecked_ref()));
    handler.forget();

    let ready = Object::new();
    set(&ready, "kind", &JsValue::from_str("ready"));
    post(&ready, None);
}
