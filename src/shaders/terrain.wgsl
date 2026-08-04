struct Globals {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var atlas: texture_2d_array<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) uv_prev: vec2<f32>,
    @location(3) @interpolate(flat) layer: i32,
    @location(4) @interpolate(flat) prev_layer: i32,
    @location(5) @interpolate(flat) fade: f32,
    @location(6) morph: f32,
    @location(7) @interpolate(flat) dbg: vec4<f32>,
    @location(8) tile_uv: vec2<f32>,
};

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        let sx = select(-1.0, 1.0, v.x >= 0.0);
        let sy = select(-1.0, 1.0, v.y >= 0.0);
        let t = vec2<f32>((1.0 - abs(v.y)) * sx, (1.0 - abs(v.x)) * sy);
        v.x = t.x;
        v.y = t.y;
    }
    return normalize(v);
}

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) morph_delta: vec3<f32>,
    @location(4) origin: vec3<f32>,
    @location(5) morph_lo: f32,
    @location(6) uvxf: vec4<f32>,
    @location(7) prev_uvxf: vec4<f32>,
    @location(8) layers: vec2<f32>,
    @location(9) blend: vec4<f32>,
    @location(10) dbg: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    let rel = origin + pos;
    let d = length(rel);
    var m = clamp((d - morph_lo) / max(blend.y - morph_lo, 1.0), 0.0, 1.0);
    m = m * m * (3.0 - 2.0 * m);
    let floor_m = clamp(blend.z, 0.0, 1.0);
    let ceil_m = max(clamp(blend.w, 0.0, 1.0), floor_m);
    m = clamp(m, floor_m, ceil_m);
    out.clip = g.view_proj * vec4<f32>(rel + morph_delta * m, 1.0);
    out.nrm = oct_decode(nrm);
    out.uv = uv * uvxf.xy + uvxf.zw;
    out.uv_prev = uv * prev_uvxf.xy + prev_uvxf.zw;
    out.layer = i32(layers.x);
    out.prev_layer = i32(layers.y);
    out.fade = blend.x;
    out.morph = m;
    out.dbg = dbg;
    out.tile_uv = uv;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let fallback = vec3<f32>(0.22, 0.28, 0.24);
    let own = textureSample(atlas, atlas_sampler, in.uv, max(in.layer, 0)).rgb;
    let prev = textureSample(atlas, atlas_sampler, in.uv_prev, max(in.prev_layer, 0)).rgb;
    let base = select(fallback, prev, in.prev_layer >= 0);
    let own_color = select(base, own, in.layer >= 0);
    let detail = clamp(in.fade, 0.0, 1.0) * (1.0 - clamp(in.morph, 0.0, 1.0));
    let albedo = mix(base, own_color, detail);
    let n = normalize(in.nrm);
    let ndl = max(dot(n, g.sun_dir.xyz), 0.0);
    let light = 0.35 + 0.75 * ndl;
    let shaded = albedo * light;
    let e = min(min(in.tile_uv.x, 1.0 - in.tile_uv.x), min(in.tile_uv.y, 1.0 - in.tile_uv.y));
    let aa = max(fwidth(e), 1e-6);
    let on = step(0.001, in.dbg.a);
    let edge = (1.0 - smoothstep(aa, aa * 3.0, e)) * on;
    var col = mix(shaded, in.dbg.rgb, clamp(in.dbg.a, 0.0, 1.0));
    col = mix(col, vec3<f32>(1.0), edge * 0.8);
    return vec4<f32>(col, 1.0);
}
