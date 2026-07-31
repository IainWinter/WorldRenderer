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
@group(2) @binding(0) var icons: texture_2d<f32>;
@group(2) @binding(1) var icon_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec3<f32>,
    @location(1) size: vec2<f32>,
    @location(2) uv_rect: vec4<f32>,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.color = color;

    let center = g.view_proj * vec4<f32>(b.origin.xyz + pos, 1.0);
    if (center.w <= 0.0) {
        out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        out.uv = vec2<f32>(0.0, 0.0);
        return out;
    }

    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    let offset = (c - vec2<f32>(0.5, 0.5)) * size * 2.0 / g.screen.xy;
    out.clip = vec4<f32>(
        center.x + offset.x * center.w,
        center.y - offset.y * center.w,
        center.z,
        center.w,
    );
    out.uv = uv_rect.xy + c * uv_rect.zw;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(icons, icon_sampler, in.uv);
    let c = tex * in.color;
    if (c.a < 0.01) {
        discard;
    }
    return c;
}
