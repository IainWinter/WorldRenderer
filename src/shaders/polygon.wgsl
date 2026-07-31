struct Globals {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    screen: vec4<f32>,
};

struct Batch {
    origin: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var<uniform> b: Batch;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(b.origin.xyz + pos, 1.0);
    out.nrm = nrm;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.nrm);
    let ndl = max(dot(n, g.sun_dir.xyz), 0.0);
    let light = 0.45 + 0.65 * ndl;
    return vec4<f32>(in.color.rgb * light, in.color.a);
}
