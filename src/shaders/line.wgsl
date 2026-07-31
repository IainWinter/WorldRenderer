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
    @location(0) color: vec4<f32>,
};

fn clip_near(a: vec4<f32>, bb: vec4<f32>) -> vec4<f32> {
    let t = (0.0001 - a.w) / (bb.w - a.w);
    return mix(a, bb, clamp(t, 0.0, 1.0));
}

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) pa: vec3<f32>,
    @location(1) pb: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
) -> VsOut {
    var out: VsOut;
    out.color = color;

    var ca = g.view_proj * vec4<f32>(b.origin.xyz + pa, 1.0);
    var cb = g.view_proj * vec4<f32>(b.origin.xyz + pb, 1.0);

    if (ca.w <= 0.0 && cb.w <= 0.0) {
        out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    if (ca.w <= 0.0) {
        ca = clip_near(ca, cb);
    }
    if (cb.w <= 0.0) {
        cb = clip_near(cb, ca);
    }

    let na = ca.xy / ca.w;
    let nb = cb.xy / cb.w;
    let screen = g.screen.xy;
    var dir = (nb - na) * screen * 0.5;
    if (length(dir) < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    }
    let perp = normalize(vec2<f32>(-dir.y, dir.x));

    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    let base = select(ca, cb, c.x > 0.5);
    let offset = perp * (width * 0.5 * c.y) * 2.0 / screen;
    out.clip = vec4<f32>(base.xy + offset * base.w, base.z, base.w);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
