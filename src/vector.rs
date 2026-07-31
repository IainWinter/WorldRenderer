use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::math::{geodetic_surface_normal, geodetic_to_ecef};
use glam::{DVec3, Vec3};

pub const BATCH_STRIDE: u64 = 256;
pub const MAX_BATCHES: u64 = 512;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolyVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub color: [u8; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInstance {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [u8; 4],
    pub width: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IconInstance {
    pub pos: [f32; 3],
    pub size: [f32; 2],
    pub uv_rect: [f32; 4],
    pub color: [u8; 4],
}

pub enum BatchKind {
    Polygon { indices: wgpu::Buffer, count: u32 },
    Line { count: u32 },
    Icon { count: u32 },
}

pub struct Batch {
    pub origin: DVec3,
    pub buffer: wgpu::Buffer,
    pub kind: BatchKind,
    pub slot: u32,
}

pub struct VectorRenderer {
    pub poly_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub icon_pipeline: wgpu::RenderPipeline,
    pub globals_bg: wgpu::BindGroup,
    pub batch_bg: wgpu::BindGroup,
    pub icon_bg: wgpu::BindGroup,
    _icon_texture: wgpu::Texture,
    pub batch_uniform: wgpu::Buffer,
    pub batches: Vec<Batch>,
    next_slot: u32,
}

fn unpack_color(rgba: u32) -> [u8; 4] {
    [
        (rgba >> 24) as u8,
        (rgba >> 16) as u8,
        (rgba >> 8) as u8,
        rgba as u8,
    ]
}

fn alpha_blend() -> Option<wgpu::BlendState> {
    Some(wgpu::BlendState::ALPHA_BLENDING)
}

impl VectorRenderer {
    pub fn new(gpu: &Gpu, globals: &wgpu::Buffer) -> Self {
        let device = &gpu.device;

        let bgl_globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector globals bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bgl_batch = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vector batch bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });

        let bgl_icons = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("icon bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let batch_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector batch uniform"),
            size: BATCH_STRIDE * MAX_BATCHES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector globals bg"),
            layout: &bgl_globals,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });

        let batch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vector batch bg"),
            layout: &bgl_batch,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &batch_uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
                }),
            }],
        });

        let (icon_texture, icon_view) = make_icon_atlas(gpu);
        let icon_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("icon sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let icon_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("icon bg"),
            layout: &bgl_icons,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&icon_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&icon_sampler),
                },
            ],
        });

        let poly_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polygon shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/polygon.wgsl").into()),
        });
        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("line shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });
        let icon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("icon shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/icon.wgsl").into()),
        });

        let pl2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vector pl"),
            bind_group_layouts: &[Some(&bgl_globals), Some(&bgl_batch)],
            immediate_size: 0,
        });
        let pl3 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("icon pl"),
            bind_group_layouts: &[Some(&bgl_globals), Some(&bgl_batch), Some(&bgl_icons)],
            immediate_size: 0,
        });

        let poly_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("polygon pipeline"),
            layout: Some(&pl2),
            vertex: wgpu::VertexState {
                module: &poly_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 28,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &poly_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: alpha_blend(),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line pipeline"),
            layout: Some(&pl2),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 28,
                            shader_location: 3,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: alpha_blend(),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let icon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("icon pipeline"),
            layout: Some(&pl3),
            vertex: wgpu::VertexState {
                module: &icon_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 20,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: 36,
                            shader_location: 3,
                        },
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &icon_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: alpha_blend(),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            poly_pipeline,
            line_pipeline,
            icon_pipeline,
            globals_bg,
            batch_bg,
            icon_bg,
            _icon_texture: icon_texture,
            batch_uniform,
            batches: Vec::new(),
            next_slot: 0,
        }
    }

    fn take_slot(&mut self) -> u32 {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % MAX_BATCHES as u32;
        slot
    }

    pub fn clear(&mut self) {
        self.batches.clear();
        self.next_slot = 0;
    }

    pub fn add_polygon(
        &mut self,
        gpu: &Gpu,
        lonlat: &[f64],
        base_h: f64,
        top_h: f64,
        color: u32,
    ) -> Option<usize> {
        let ring_len = lonlat.len() / 2;
        if ring_len < 3 {
            return None;
        }
        let mut ring: Vec<[f64; 2]> = (0..ring_len)
            .map(|i| [lonlat[i * 2], lonlat[i * 2 + 1]])
            .collect();
        let mut area = 0.0;
        for i in 0..ring_len {
            let j = (i + 1) % ring_len;
            area += ring[i][0] * ring[j][1] - ring[j][0] * ring[i][1];
        }
        if area < 0.0 {
            ring.reverse();
        }

        let flat: Vec<f64> = ring.iter().flat_map(|p| [p[0], p[1]]).collect();
        let tri = earcutr::earcut(&flat, &[], 2).ok()?;

        let rgba = unpack_color(color);
        let origin = geodetic_to_ecef(ring[0][0].to_radians(), ring[0][1].to_radians(), top_h);

        let mut top = Vec::with_capacity(ring_len);
        let mut bottom = Vec::with_capacity(ring_len);
        let mut ups = Vec::with_capacity(ring_len);
        for p in ring.iter() {
            let (lon, lat) = (p[0].to_radians(), p[1].to_radians());
            top.push(geodetic_to_ecef(lon, lat, top_h) - origin);
            bottom.push(geodetic_to_ecef(lon, lat, base_h) - origin);
            ups.push(geodetic_surface_normal(lon, lat));
        }

        let mut verts: Vec<PolyVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for i in 0..ring_len {
            verts.push(PolyVertex {
                pos: top[i].as_vec3().to_array(),
                nrm: ups[i].as_vec3().to_array(),
                color: rgba,
            });
        }
        for t in tri.iter() {
            indices.push(*t as u32);
        }

        let extruded = (top_h - base_h).abs() > 1e-6;
        if extruded {
            for i in 0..ring_len {
                let j = (i + 1) % ring_len;
                let edge = (top[j] - top[i]).as_vec3();
                let n = edge.cross(ups[i].as_vec3());
                let n = if n.length_squared() > 1e-12 {
                    n.normalize()
                } else {
                    Vec3::Z
                };
                let base = verts.len() as u32;
                for p in [bottom[i], bottom[j], top[j], top[i]] {
                    verts.push(PolyVertex {
                        pos: p.as_vec3().to_array(),
                        nrm: n.to_array(),
                        color: rgba,
                    });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        let vbuf = upload_buffer(
            gpu,
            "polygon verts",
            bytemuck::cast_slice(&verts),
            wgpu::BufferUsages::VERTEX,
        );
        let ibuf = upload_buffer(
            gpu,
            "polygon indices",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );
        let slot = self.take_slot();
        self.batches.push(Batch {
            origin,
            buffer: vbuf,
            kind: BatchKind::Polygon {
                indices: ibuf,
                count: indices.len() as u32,
            },
            slot,
        });
        Some(self.batches.len() - 1)
    }

    pub fn add_line(
        &mut self,
        gpu: &Gpu,
        lonlath: &[f64],
        width_px: f32,
        color: u32,
    ) -> Option<usize> {
        let count = lonlath.len() / 3;
        if count < 2 {
            return None;
        }
        let rgba = unpack_color(color);
        let origin = geodetic_to_ecef(lonlath[0].to_radians(), lonlath[1].to_radians(), lonlath[2]);
        let pts: Vec<DVec3> = (0..count)
            .map(|i| {
                geodetic_to_ecef(
                    lonlath[i * 3].to_radians(),
                    lonlath[i * 3 + 1].to_radians(),
                    lonlath[i * 3 + 2],
                ) - origin
            })
            .collect();
        let segments: Vec<LineInstance> = (0..count - 1)
            .map(|i| LineInstance {
                a: pts[i].as_vec3().to_array(),
                b: pts[i + 1].as_vec3().to_array(),
                color: rgba,
                width: width_px,
            })
            .collect();

        let buffer = upload_buffer(
            gpu,
            "line instances",
            bytemuck::cast_slice(&segments),
            wgpu::BufferUsages::VERTEX,
        );
        let slot = self.take_slot();
        self.batches.push(Batch {
            origin,
            buffer,
            kind: BatchKind::Line {
                count: segments.len() as u32,
            },
            slot,
        });
        Some(self.batches.len() - 1)
    }

    pub fn add_icons(
        &mut self,
        gpu: &Gpu,
        lonlath: &[f64],
        size_px: f32,
        color: u32,
    ) -> Option<usize> {
        let count = lonlath.len() / 3;
        if count == 0 {
            return None;
        }
        let rgba = unpack_color(color);
        let origin = geodetic_to_ecef(lonlath[0].to_radians(), lonlath[1].to_radians(), lonlath[2]);
        let icons: Vec<IconInstance> = (0..count)
            .map(|i| {
                let p = geodetic_to_ecef(
                    lonlath[i * 3].to_radians(),
                    lonlath[i * 3 + 1].to_radians(),
                    lonlath[i * 3 + 2],
                ) - origin;
                IconInstance {
                    pos: p.as_vec3().to_array(),
                    size: [size_px, size_px],
                    uv_rect: [0.0, 0.0, 1.0, 1.0],
                    color: rgba,
                }
            })
            .collect();

        let buffer = upload_buffer(
            gpu,
            "icon instances",
            bytemuck::cast_slice(&icons),
            wgpu::BufferUsages::VERTEX,
        );
        let slot = self.take_slot();
        self.batches.push(Batch {
            origin,
            buffer,
            kind: BatchKind::Icon {
                count: icons.len() as u32,
            },
            slot,
        });
        Some(self.batches.len() - 1)
    }

    pub fn update_origins(&self, gpu: &Gpu, eye: DVec3) {
        for batch in self.batches.iter() {
            let rel = (batch.origin - eye).as_vec3();
            let data = [rel.x, rel.y, rel.z, 0.0f32];
            gpu.queue.write_buffer(
                &self.batch_uniform,
                batch.slot as u64 * BATCH_STRIDE,
                bytemuck::cast_slice(&data),
            );
        }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.batches.is_empty() {
            return;
        }
        pass.set_bind_group(0, &self.globals_bg, &[]);

        pass.set_pipeline(&self.poly_pipeline);
        for batch in self.batches.iter() {
            if let BatchKind::Polygon { indices, count } = &batch.kind {
                pass.set_bind_group(1, &self.batch_bg, &[batch.slot * BATCH_STRIDE as u32]);
                pass.set_vertex_buffer(0, batch.buffer.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }

        pass.set_pipeline(&self.line_pipeline);
        for batch in self.batches.iter() {
            if let BatchKind::Line { count } = &batch.kind {
                pass.set_bind_group(1, &self.batch_bg, &[batch.slot * BATCH_STRIDE as u32]);
                pass.set_vertex_buffer(0, batch.buffer.slice(..));
                pass.draw(0..6, 0..*count);
            }
        }

        pass.set_pipeline(&self.icon_pipeline);
        pass.set_bind_group(2, &self.icon_bg, &[]);
        for batch in self.batches.iter() {
            if let BatchKind::Icon { count } = &batch.kind {
                pass.set_bind_group(1, &self.batch_bg, &[batch.slot * BATCH_STRIDE as u32]);
                pass.set_vertex_buffer(0, batch.buffer.slice(..));
                pass.draw(0..6, 0..*count);
            }
        }
    }
}

fn upload_buffer(gpu: &Gpu, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let mut padded = bytes.to_vec();
    while padded.len() < 4 || padded.len() % 4 != 0 {
        padded.push(0);
    }
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: padded.len() as u64,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .unwrap()
        .copy_from_slice(&padded);
    buffer.unmap();
    buffer
}

fn make_icon_atlas(gpu: &Gpu) -> (wgpu::Texture, wgpu::TextureView) {
    const N: u32 = 64;
    let mut pixels = vec![0u8; (N * N * 4) as usize];
    let c = (N as f32 - 1.0) * 0.5;
    for y in 0..N {
        for x in 0..N {
            let dx = x as f32 - c;
            let dy = y as f32 - c;
            let d = (dx * dx + dy * dy).sqrt();
            let outer = c - 1.0;
            let a = (1.0 - (d - (outer - 1.5)).clamp(0.0, 1.5) / 1.5).clamp(0.0, 1.0);
            let ring = if d > outer - 7.0 { 1.0 } else { 0.55 };
            let i = ((y * N + x) * 4) as usize;
            pixels[i] = (255.0 * ring) as u8;
            pixels[i + 1] = (255.0 * ring) as u8;
            pixels[i + 2] = (255.0 * ring) as u8;
            pixels[i + 3] = (255.0 * a) as u8;
        }
    }
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("icon atlas"),
        size: wgpu::Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(N * 4),
            rows_per_image: Some(N),
        },
        wgpu::Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
