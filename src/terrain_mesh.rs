use crate::math::{geodetic_surface_normal, geodetic_to_ecef, oct_encode};
use crate::tiling::{merc_y_to_lat, TileKey};
use glam::{DVec3, Vec3};

pub const GRID_N: u32 = 32;
pub const HEIGHT_N: usize = 16;
pub const HEIGHT_STRIDE: usize = HEIGHT_N + 1;
pub const CORE_VERTS: u32 = (GRID_N + 1) * (GRID_N + 1);
pub const SKIRT_VERTS: u32 = 4 * GRID_N;
pub const TOTAL_VERTS: u32 = CORE_VERTS + SKIRT_VERTS;
pub const TOTAL_INDICES: u32 = GRID_N * GRID_N * 6 + GRID_N * 4 * 6;
pub const VERTEX_SIZE: u32 = 32;
pub const SLOT_SIZE: u32 = (TOTAL_VERTS * VERTEX_SIZE).div_ceil(256) * 256;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    pub nrm: [i16; 2],
    pub uv: [u16; 2],
    pub morph: [f32; 3],
}

pub fn perimeter_cell(p: u32) -> (u32, u32) {
    let n = GRID_N;
    if p < n {
        (p, 0)
    } else if p < 2 * n {
        (n, p - n)
    } else if p < 3 * n {
        (n - (p - 2 * n), n)
    } else {
        (0, n - (p - 3 * n))
    }
}

pub fn build_indices() -> Vec<u16> {
    let n = GRID_N;
    let stride = n + 1;
    let mut idx = Vec::with_capacity(TOTAL_INDICES as usize);
    for y in 0..n {
        for x in 0..n {
            let i00 = (y * stride + x) as u16;
            let i10 = i00 + 1;
            let i01 = i00 + stride as u16;
            let i11 = i01 + 1;
            idx.extend_from_slice(&[i00, i01, i10, i10, i01, i11]);
        }
    }
    for p in 0..4 * n {
        let q = (p + 1) % (4 * n);
        let (cx, cy) = perimeter_cell(p);
        let (qx, qy) = perimeter_cell(q);
        let c0 = (cy * stride + cx) as u16;
        let c1 = (qy * stride + qx) as u16;
        let s0 = (CORE_VERTS + p) as u16;
        let s1 = (CORE_VERTS + q) as u16;
        idx.extend_from_slice(&[c0, c1, s1, c0, s1, s0]);
    }
    idx
}

pub struct Heightmap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Heightmap {
    pub fn sample(&self, u: f64, v: f64) -> f64 {
        let fx = (u * self.w as f64 - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (v * self.h as f64 - 0.5).clamp(0.0, (self.h - 1) as f64);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let a = self.data[y0 * self.w + x0] as f64;
        let b = self.data[y0 * self.w + x1] as f64;
        let c = self.data[y1 * self.w + x0] as f64;
        let d = self.data[y1 * self.w + x1] as f64;
        (a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty
    }
}

pub fn decode_terrarium(bytes: &[u8]) -> Result<Heightmap, String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let w = info.width as usize;
    let h = info.height as usize;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(format!("unsupported terrarium color type {:?}", other)),
    };
    let mut data = vec![0f32; w * h];
    for i in 0..w * h {
        let r = buf[i * channels] as f32;
        let g = buf[i * channels + 1] as f32;
        let b = buf[i * channels + 2] as f32;
        data[i] = r * 256.0 + g + b / 256.0 - 32768.0;
    }
    Ok(Heightmap { w, h, data })
}

pub struct BuiltMesh {
    pub center: DVec3,
    pub vertices: Vec<TerrainVertex>,
    pub min_height: f32,
    pub max_height: f32,
    pub heights: Vec<f32>,
}

pub fn build_mesh(key: TileKey, hm: &Heightmap, uv: [f64; 3]) -> BuiltMesh {
    let n = GRID_N;
    let stride = (n + 1) as usize;
    let tiles = (1u32 << key.z) as f64;
    let (lon0, _, lon1, _) = key.lon_lat_bounds();

    let mut lons = vec![0f64; stride];
    let mut lats = vec![0f64; stride];
    for i in 0..stride {
        let f = i as f64 / n as f64;
        lons[i] = lon0 + (lon1 - lon0) * f;
        lats[i] = merc_y_to_lat(1.0 - 2.0 * (key.y as f64 + f) / tiles);
    }

    let center = geodetic_to_ecef(
        (lon0 + lon1) * 0.5,
        merc_y_to_lat(1.0 - 2.0 * (key.y as f64 + 0.5) / tiles),
        0.0,
    );

    let mut world = vec![DVec3::ZERO; stride * stride];
    let mut ups = vec![DVec3::ZERO; stride * stride];
    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    let sample_step = stride / HEIGHT_N;
    let mut heights = Vec::with_capacity(HEIGHT_STRIDE * HEIGHT_STRIDE);
    for y in 0..stride {
        for x in 0..stride {
            let u = x as f64 / n as f64;
            let v = y as f64 / n as f64;
            let hgt = hm.sample(uv[1] + u * uv[0], uv[2] + v * uv[0]);
            min_h = min_h.min(hgt as f32);
            max_h = max_h.max(hgt as f32);
            if x % sample_step == 0 && y % sample_step == 0 {
                heights.push(hgt as f32);
            }
            world[y * stride + x] = geodetic_to_ecef(lons[x], lats[y], hgt);
            ups[y * stride + x] = geodetic_surface_normal(lons[x], lats[y]);
        }
    }

    let mut normals = vec![Vec3::ZERO; stride * stride];
    for y in 0..n as usize {
        for x in 0..n as usize {
            let i00 = y * stride + x;
            let i10 = i00 + 1;
            let i01 = i00 + stride;
            let i11 = i01 + 1;
            let p00 = world[i00];
            let e1 = (world[i01] - p00).as_vec3();
            let e2 = (world[i10] - p00).as_vec3();
            let na = e1.cross(e2);
            let p10 = world[i10];
            let f1 = (world[i01] - p10).as_vec3();
            let f2 = (world[i11] - p10).as_vec3();
            let nb = f1.cross(f2);
            for i in [i00, i01, i10] {
                normals[i] += na;
            }
            for i in [i10, i01, i11] {
                normals[i] += nb;
            }
        }
    }

    let mut coarse = vec![DVec3::ZERO; stride * stride];
    for y in 0..stride {
        for x in 0..stride {
            let (x0, x1) = if x % 2 == 0 { (x, x) } else { (x - 1, x + 1) };
            let (y0, y1) = if y % 2 == 0 { (y, y) } else { (y - 1, y + 1) };
            coarse[y * stride + x] = (world[y0 * stride + x0]
                + world[y0 * stride + x1]
                + world[y1 * stride + x0]
                + world[y1 * stride + x1])
                * 0.25;
        }
    }

    let mut vertices = Vec::with_capacity(TOTAL_VERTS as usize);
    for y in 0..stride {
        for x in 0..stride {
            let i = y * stride + x;
            let nrm = if normals[i].length_squared() > 1e-12 {
                normals[i].normalize()
            } else {
                ups[i].as_vec3()
            };
            vertices.push(TerrainVertex {
                pos: (world[i] - center).as_vec3().to_array(),
                nrm: oct_encode(nrm),
                uv: [
                    (x as f64 / n as f64 * 65535.0) as u16,
                    (y as f64 / n as f64 * 65535.0) as u16,
                ],
                morph: (coarse[i] - world[i]).as_vec3().to_array(),
            });
        }
    }

    let relief = (max_h - min_h).max(0.0) as f64;
    let skirt = (relief * 0.9 + key.ground_extent() * 0.004).clamp(25.0, 4000.0);
    for p in 0..4 * n {
        let (cx, cy) = perimeter_cell(p);
        let i = cy as usize * stride + cx as usize;
        let dropped = world[i] - ups[i] * skirt;
        let v = vertices[i];
        vertices.push(TerrainVertex {
            pos: (dropped - center).as_vec3().to_array(),
            nrm: v.nrm,
            uv: v.uv,
            morph: v.morph,
        });
    }

    BuiltMesh {
        center,
        vertices,
        min_height: min_h,
        max_height: max_h,
        heights,
    }
}

pub fn decode_png_rgba(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width, info.height);
    let count = (w * h) as usize;
    let mut out = vec![255u8; count * 4];
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        other => return Err(format!("unsupported png color type {:?}", other)),
    };
    for i in 0..count {
        let s = i * channels;
        match channels {
            1 => {
                out[i * 4] = buf[s];
                out[i * 4 + 1] = buf[s];
                out[i * 4 + 2] = buf[s];
            }
            _ => {
                out[i * 4] = buf[s];
                out[i * 4 + 1] = buf[s + 1];
                out[i * 4 + 2] = buf[s + 2];
                if channels == 4 {
                    out[i * 4 + 3] = buf[s + 3];
                }
            }
        }
    }
    Ok((w, h, out))
}

pub fn decode_jpeg_rgba(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let pixels = dec.decode().map_err(|e| e.to_string())?;
    let info = dec.info().ok_or_else(|| "no jpeg info".to_string())?;
    let (w, h) = (info.width as u32, info.height as u32);
    let count = (w * h) as usize;
    let mut out = vec![255u8; count * 4];
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            for i in 0..count {
                out[i * 4] = pixels[i * 3];
                out[i * 4 + 1] = pixels[i * 3 + 1];
                out[i * 4 + 2] = pixels[i * 3 + 2];
            }
        }
        jpeg_decoder::PixelFormat::L8 => {
            for i in 0..count {
                out[i * 4] = pixels[i];
                out[i * 4 + 1] = pixels[i];
                out[i * 4 + 2] = pixels[i];
            }
        }
        other => return Err(format!("unsupported jpeg pixel format {:?}", other)),
    }
    Ok((w, h, out))
}
