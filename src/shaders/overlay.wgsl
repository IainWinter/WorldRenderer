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

const DASH_PERIOD: f32 = 18.0;
const DASH_DUTY: f32 = 0.55;

fn behind_horizon(rel: vec3<f32>, nrm: vec3<f32>) -> bool {
    return dot(normalize(nrm), normalize(rel)) > 0.08;
}

struct FillOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_fill(
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> FillOut {
    var out: FillOut;
    out.color = color;
    let rel = b.origin.xyz + pos;
    if (behind_horizon(rel, nrm)) {
        out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    out.clip = g.view_proj * vec4<f32>(rel, 1.0);
    return out;
}

@fragment
fn fs_fill(in: FillOut) -> @location(0) vec4<f32> {
    return in.color;
}

struct DashOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) travel: f32,
};

fn clip_near(a: vec4<f32>, bb: vec4<f32>) -> vec4<f32> {
    let t = (0.0001 - a.w) / (bb.w - a.w);
    return mix(a, bb, clamp(t, 0.0, 1.0));
}

@vertex
fn vs_dash(
    @builtin(vertex_index) vi: u32,
    @location(0) pa: vec3<f32>,
    @location(1) pb: vec3<f32>,
    @location(2) na: vec3<f32>,
    @location(3) nb: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) width: f32,
) -> DashOut {
    var out: DashOut;
    out.color = color;
    out.travel = 0.0;

    let rel_a = b.origin.xyz + pa;
    let rel_b = b.origin.xyz + pb;
    if (behind_horizon(rel_a, na) && behind_horizon(rel_b, nb)) {
        out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }

    var ca = g.view_proj * vec4<f32>(rel_a, 1.0);
    var cb = g.view_proj * vec4<f32>(rel_b, 1.0);
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

    let sa = ca.xy / ca.w;
    let sb = cb.xy / cb.w;
    let screen = g.screen.xy;
    var dir = (sb - sa) * screen * 0.5;
    let span = length(dir);
    if (span < 1e-6) {
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
    let along = select(0.0, span, c.x > 0.5);
    out.travel = along;
    let shift = perp * (width * 0.5 * c.y) * 2.0 / screen;
    out.clip = vec4<f32>(base.xy + shift * base.w, base.z, base.w);
    return out;
}

@fragment
fn fs_dash(in: DashOut) -> @location(0) vec4<f32> {
    let phase = fract(in.travel / DASH_PERIOD);
    if (phase > DASH_DUTY) {
        discard;
    }
    return in.color;
}
