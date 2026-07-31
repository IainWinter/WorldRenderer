use crate::terrain_mesh::{decode_jpeg_rgba, decode_png_rgba};
use glam::{Mat4, Vec3};

pub const MODEL_HEADER_BYTES: usize = 16;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
}

pub struct ModelData {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub texture: Option<(u32, u32, Vec<u8>)>,
}

pub fn parse_glb(bytes: &[u8]) -> Result<ModelData, String> {
    let file = gltf::Gltf::from_slice(bytes).map_err(|e| e.to_string())?;
    let blob = file.blob.as_deref();
    let document = &file.document;

    let mut vertices: Vec<ModelVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut texture: Option<(u32, u32, Vec<u8>)> = None;

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| "glb has no scene".to_string())?;

    let mut stack: Vec<(gltf::Node, Mat4)> = scene
        .nodes()
        .map(|n| {
            let m = Mat4::from_cols_array_2d(&n.transform().matrix());
            (n, m)
        })
        .collect();

    while let Some((node, world)) = stack.pop() {
        for child in node.children() {
            let local = Mat4::from_cols_array_2d(&child.transform().matrix());
            stack.push((child, world * local));
        }
        let Some(mesh) = node.mesh() else { continue };
        let normal_matrix = Mat4::from_cols(
            world.x_axis.normalize_or_zero(),
            world.y_axis.normalize_or_zero(),
            world.z_axis.normalize_or_zero(),
            glam::Vec4::W,
        );
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => blob,
                gltf::buffer::Source::Uri(_) => None,
            });
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let base = vertices.len() as u32;
            for i in 0..positions.len() {
                let p = world.transform_point3(Vec3::from(positions[i]));
                let n = normal_matrix
                    .transform_vector3(Vec3::from(normals[i]))
                    .normalize_or(Vec3::Y);
                vertices.push(ModelVertex {
                    pos: p.to_array(),
                    nrm: n.to_array(),
                    uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                });
            }
            match reader.read_indices() {
                Some(read) => {
                    for i in read.into_u32() {
                        indices.push(base + i);
                    }
                }
                None => {
                    for i in 0..positions.len() as u32 {
                        indices.push(base + i);
                    }
                }
            }

            if texture.is_none() {
                if let Some(info) = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_texture()
                {
                    texture = read_texture(&info.texture(), blob);
                }
            }
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err("glb has no triangles".to_string());
    }

    Ok(ModelData {
        vertices,
        indices,
        texture,
    })
}

fn read_texture(texture: &gltf::Texture, blob: Option<&[u8]>) -> Option<(u32, u32, Vec<u8>)> {
    let source = texture.source().source();
    let gltf::image::Source::View { view, mime_type } = source else {
        return None;
    };
    let blob = blob?;
    let start = view.offset();
    let end = start + view.length();
    if end > blob.len() {
        return None;
    }
    let bytes = &blob[start..end];
    let decoded = if mime_type.contains("png") {
        decode_png_rgba(bytes)
    } else {
        decode_jpeg_rgba(bytes)
    };
    decoded.ok()
}

pub fn encode(data: &ModelData) -> Vec<u8> {
    let (tw, th, pixels) = match &data.texture {
        Some((w, h, p)) => (*w, *h, p.as_slice()),
        None => (0, 0, &[][..]),
    };
    let mut out = Vec::with_capacity(
        MODEL_HEADER_BYTES + data.vertices.len() * 32 + data.indices.len() * 4 + pixels.len(),
    );
    out.extend_from_slice(&(data.vertices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.indices.len() as u32).to_le_bytes());
    out.extend_from_slice(&tw.to_le_bytes());
    out.extend_from_slice(&th.to_le_bytes());
    out.extend_from_slice(bytemuck::cast_slice(&data.vertices));
    out.extend_from_slice(bytemuck::cast_slice(&data.indices));
    out.extend_from_slice(pixels);
    out
}

pub fn decode(bytes: &[u8]) -> Option<ModelData> {
    if bytes.len() < MODEL_HEADER_BYTES {
        return None;
    }
    let read = |i: usize| {
        u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
    };
    let vcount = read(0);
    let icount = read(4);
    let tw = read(8);
    let th = read(12);
    let vbytes = vcount * 32;
    let ibytes = icount * 4;
    let tbytes = tw * th * 4;
    if bytes.len() < MODEL_HEADER_BYTES + vbytes + ibytes + tbytes {
        return None;
    }
    let vstart = MODEL_HEADER_BYTES;
    let istart = vstart + vbytes;
    let tstart = istart + ibytes;
    let vertices: Vec<ModelVertex> = bytemuck::cast_slice(&bytes[vstart..istart]).to_vec();
    let indices: Vec<u32> = bytemuck::cast_slice(&bytes[istart..tstart]).to_vec();
    let texture = if tbytes > 0 {
        Some((
            tw as u32,
            th as u32,
            bytes[tstart..tstart + tbytes].to_vec(),
        ))
    } else {
        None
    };
    Some(ModelData {
        vertices,
        indices,
        texture,
    })
}
