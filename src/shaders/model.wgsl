struct Globals {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) row0: vec4<f32>,
    @location(4) row1: vec4<f32>,
    @location(5) row2: vec4<f32>,
    @location(6) color: vec4<f32>,
) -> VsOut {
    let basis = mat3x3<f32>(
        vec3<f32>(row0.x, row1.x, row2.x),
        vec3<f32>(row0.y, row1.y, row2.y),
        vec3<f32>(row0.z, row1.z, row2.z),
    );
    let offset = vec3<f32>(row0.w, row1.w, row2.w);
    var out: VsOut;
    out.clip = g.view_proj * vec4<f32>(basis * pos + offset, 1.0);
    out.nrm = normalize(basis * nrm);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(albedo_tex, albedo_sampler, in.uv);
    let n = normalize(in.nrm);
    let ndl = max(dot(n, g.sun_dir.xyz), 0.0);
    let light = 0.4 + 0.8 * ndl;
    return vec4<f32>(tex.rgb * in.color.rgb * light, 1.0);
}
