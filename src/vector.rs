use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::math::{dir_to_geodetic, geodetic_surface_normal, geodetic_to_ecef};
use glam::{DVec3, Vec3};

pub const BATCH_STRIDE: u64 = 256;
pub const MAX_BATCHES: u64 = 4096;
pub const DEBUG_SLOT: u32 = MAX_BATCHES as u32 - 1;
pub const SELECT_FILL_SLOT: u32 = MAX_BATCHES as u32 - 2;
pub const SELECT_EDGE_SLOT: u32 = MAX_BATCHES as u32 - 3;

const SELECT_GRID: usize = 16;
const SELECT_EDGE_STEPS: usize = 12;
const SELECT_FILL_COLOR: u32 = 0xb9bcc82e;
const SELECT_EDGE_COLOR: u32 = 0xe4e7f0ff;
const SELECT_EDGE_WIDTH: f32 = 2.0;

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DashInstance {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub na: [f32; 3],
    pub nb: [f32; 3],
    pub color: [u8; 4],
    pub width: f32,
}

pub enum BatchKind {
    Polygon { indices: wgpu::Buffer, count: u32 },
    Tube { indices: wgpu::Buffer, count: u32 },
    Line { count: u32 },
    Icon { count: u32 },
    Dash { count: u32 },
}

pub struct Batch {
    pub origin: DVec3,
    pub buffer: wgpu::Buffer,
    pub kind: BatchKind,
    pub slot: u32,
    pub vertices: u32,
    pub triangles: u32,
}

pub struct VectorRenderer {
    pub poly_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub icon_pipeline: wgpu::RenderPipeline,
    pub overlay_fill_pipeline: wgpu::RenderPipeline,
    pub overlay_dash_pipeline: wgpu::RenderPipeline,
    pub globals_bg: wgpu::BindGroup,
    pub batch_bg: wgpu::BindGroup,
    pub icon_bg: wgpu::BindGroup,
    icon_layout: wgpu::BindGroupLayout,
    icon_sampler: wgpu::Sampler,
    _icon_texture: wgpu::Texture,
    pub icon_cols: u32,
    pub icon_rows: u32,
    pub icon_sheet_loaded: bool,
    pub batch_uniform: wgpu::Buffer,
    pub batches: Vec<Batch>,
    pub debug: Option<Batch>,
    pub markers: Vec<Batch>,
    pub selection: Option<(Batch, Batch)>,
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

        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/overlay.wgsl").into()),
        });

        let overlay_depth = || {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };

        let overlay_fill_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay fill pipeline"),
                layout: Some(&pl2),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs_fill"),
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
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: overlay_depth(),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_fill"),
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

        let overlay_dash_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay dash pipeline"),
                layout: Some(&pl2),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs_dash"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 56,
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
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 24,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 36,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Unorm8x4,
                                offset: 48,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 52,
                                shader_location: 5,
                            },
                        ],
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: overlay_depth(),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs_dash"),
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
            overlay_fill_pipeline,
            overlay_dash_pipeline,
            globals_bg,
            batch_bg,
            icon_bg,
            icon_layout: bgl_icons,
            icon_sampler,
            _icon_texture: icon_texture,
            icon_cols: 1,
            icon_rows: 1,
            icon_sheet_loaded: false,
            batch_uniform,
            batches: Vec::new(),
            debug: None,
            markers: Vec::new(),
            selection: None,
            next_slot: 0,
        }
    }

    pub fn set_icon_sheet(
        &mut self,
        gpu: &Gpu,
        width: u32,
        height: u32,
        rgba: &[u8],
        cols: u32,
        rows: u32,
    ) {
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            return;
        }
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("icon sheet"),
            size: wgpu::Extent3d {
                width,
                height,
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
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.icon_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("icon bg"),
            layout: &self.icon_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.icon_sampler),
                },
            ],
        });
        self._icon_texture = texture;
        self.icon_cols = cols.max(1);
        self.icon_rows = rows.max(1);
        self.icon_sheet_loaded = true;
    }

    pub fn icon_uv(&self, icon: u32) -> [f32; 4] {
        let cells = self.icon_cols * self.icon_rows;
        let i = if cells == 0 { 0 } else { icon % cells };
        let col = i % self.icon_cols;
        let row = i / self.icon_cols;
        let w = 1.0 / self.icon_cols as f32;
        let h = 1.0 / self.icon_rows as f32;
        [col as f32 * w, row as f32 * h, w, h]
    }

    pub fn totals(&self) -> (u32, u32, u32, u32, u32, u32) {
        let mut out = (0, 0, 0, 0, 0, 0);
        for b in self.batches.iter() {
            match b.kind {
                BatchKind::Polygon { .. } => out.0 += 1,
                BatchKind::Tube { .. } => out.1 += 1,
                BatchKind::Line { .. } => out.2 += 1,
                BatchKind::Icon { .. } => out.3 += 1,
                BatchKind::Dash { .. } => {}
            }
            out.4 += b.vertices;
            out.5 += b.triangles;
        }
        out
    }

    fn take_slot(&mut self) -> u32 {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % SELECT_EDGE_SLOT;
        slot
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_selection(
        &mut self,
        gpu: &Gpu,
        anchor: DVec3,
        right: DVec3,
        forward: DVec3,
        u: f64,
        v: f64,
        height: f64,
    ) {
        self.selection = Some(build_selection(gpu, anchor, right, forward, u, v, height));
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    pub fn add_marker(&mut self, gpu: &Gpu, position: DVec3, radius: f64, color: u32) -> usize {
        let slot = self.take_slot();
        let batch = build_marker(gpu, position, radius, color, slot);
        self.markers.push(batch);
        self.markers.len() - 1
    }

    pub fn set_marker(
        &mut self,
        gpu: &Gpu,
        index: usize,
        position: DVec3,
        radius: f64,
        color: u32,
    ) {
        let Some(existing) = self.markers.get(index) else {
            return;
        };
        let slot = existing.slot;
        self.markers[index] = build_marker(gpu, position, radius, color, slot);
    }

    pub fn clear_markers(&mut self) {
        self.markers.clear();
    }

    pub fn set_debug_lines(
        &mut self,
        gpu: &Gpu,
        origin: DVec3,
        segments: &[(DVec3, DVec3, u32)],
        width: f32,
    ) {
        if segments.is_empty() {
            self.debug = None;
            return;
        }
        let items: Vec<LineInstance> = segments
            .iter()
            .map(|(a, b, color)| LineInstance {
                a: (*a - origin).as_vec3().to_array(),
                b: (*b - origin).as_vec3().to_array(),
                color: unpack_color(*color),
                width,
            })
            .collect();
        let buffer = upload_buffer(
            gpu,
            "debug lines",
            bytemuck::cast_slice(&items),
            wgpu::BufferUsages::VERTEX,
        );
        let count = items.len() as u32;
        self.debug = Some(Batch {
            origin,
            buffer,
            kind: BatchKind::Line { count },
            slot: DEBUG_SLOT,
            vertices: count * 6,
            triangles: count * 2,
        });
    }

    pub fn clear_debug_lines(&mut self) {
        self.debug = None;
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
            vertices: verts.len() as u32,
            triangles: indices.len() as u32 / 3,
        });
        Some(self.batches.len() - 1)
    }

    pub fn add_tube(
        &mut self,
        gpu: &Gpu,
        lonlath: &[f64],
        radius: f64,
        sides: u32,
        color: u32,
    ) -> Option<usize> {
        let count = lonlath.len() / 3;
        if count < 2 || radius <= 0.0 {
            return None;
        }
        let sides = sides.clamp(3, 64) as usize;
        let rgba = unpack_color(color);
        let origin = geodetic_to_ecef(lonlath[0].to_radians(), lonlath[1].to_radians(), lonlath[2]);

        let mut pts = Vec::with_capacity(count);
        let mut ups = Vec::with_capacity(count);
        for i in 0..count {
            let lon = lonlath[i * 3].to_radians();
            let lat = lonlath[i * 3 + 1].to_radians();
            pts.push(geodetic_to_ecef(lon, lat, lonlath[i * 3 + 2]) - origin);
            ups.push(geodetic_surface_normal(lon, lat));
        }

        let mut verts: Vec<PolyVertex> = Vec::with_capacity(count * sides + 2);
        let mut indices: Vec<u32> = Vec::with_capacity(count * sides * 6);
        let mut tangents = Vec::with_capacity(count);

        for i in 0..count {
            let raw = if i == 0 {
                pts[1] - pts[0]
            } else if i == count - 1 {
                pts[i] - pts[i - 1]
            } else {
                pts[i + 1] - pts[i - 1]
            };
            let tangent = if raw.length_squared() > 1e-12 {
                raw.normalize()
            } else {
                ups[i].cross(DVec3::X).normalize()
            };
            let side = tangent.cross(ups[i]);
            let side = if side.length_squared() > 1e-12 {
                side.normalize()
            } else {
                tangent.cross(DVec3::Z).normalize()
            };
            let up = side.cross(tangent).normalize();
            tangents.push(tangent);
            for s in 0..sides {
                let a = std::f64::consts::TAU * s as f64 / sides as f64;
                let dir = side * a.cos() + up * a.sin();
                verts.push(PolyVertex {
                    pos: (pts[i] + dir * radius).as_vec3().to_array(),
                    nrm: dir.as_vec3().to_array(),
                    color: rgba,
                });
            }
        }

        for i in 0..count - 1 {
            for s in 0..sides {
                let n = (s + 1) % sides;
                let a = (i * sides + s) as u32;
                let b = (i * sides + n) as u32;
                let c = ((i + 1) * sides + n) as u32;
                let d = ((i + 1) * sides + s) as u32;
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }

        for (end, ring) in [(0usize, 0usize), (1usize, count - 1)] {
            let normal = if end == 0 {
                -tangents[0]
            } else {
                tangents[ring]
            };
            let center = verts.len() as u32;
            verts.push(PolyVertex {
                pos: pts[ring].as_vec3().to_array(),
                nrm: normal.as_vec3().to_array(),
                color: rgba,
            });
            for s in 0..sides {
                let a = (ring * sides + s) as u32;
                let b = (ring * sides + (s + 1) % sides) as u32;
                if end == 0 {
                    indices.extend_from_slice(&[center, b, a]);
                } else {
                    indices.extend_from_slice(&[center, a, b]);
                }
            }
        }

        let vbuf = upload_buffer(
            gpu,
            "tube verts",
            bytemuck::cast_slice(&verts),
            wgpu::BufferUsages::VERTEX,
        );
        let ibuf = upload_buffer(
            gpu,
            "tube indices",
            bytemuck::cast_slice(&indices),
            wgpu::BufferUsages::INDEX,
        );
        let slot = self.take_slot();
        self.batches.push(Batch {
            origin,
            buffer: vbuf,
            kind: BatchKind::Tube {
                indices: ibuf,
                count: indices.len() as u32,
            },
            slot,
            vertices: verts.len() as u32,
            triangles: indices.len() as u32 / 3,
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
        let segment_count = segments.len() as u32;
        self.batches.push(Batch {
            origin,
            buffer,
            kind: BatchKind::Line {
                count: segment_count,
            },
            slot,
            vertices: segment_count * 6,
            triangles: segment_count * 2,
        });
        Some(self.batches.len() - 1)
    }

    pub fn add_icons(
        &mut self,
        gpu: &Gpu,
        lonlath: &[f64],
        size_px: f32,
        color: u32,
        icon: u32,
    ) -> Option<usize> {
        let count = lonlath.len() / 3;
        if count == 0 {
            return None;
        }
        let rgba = unpack_color(color);
        let uv_rect = self.icon_uv(icon);
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
                    uv_rect,
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
        let icon_count = icons.len() as u32;
        self.batches.push(Batch {
            origin,
            buffer,
            kind: BatchKind::Icon { count: icon_count },
            slot,
            vertices: icon_count * 6,
            triangles: icon_count * 2,
        });
        Some(self.batches.len() - 1)
    }

    pub fn update_origins(&self, gpu: &Gpu, eye: DVec3) {
        let selection = self.selection.iter().flat_map(|(a, b)| [a, b]);
        for batch in self
            .batches
            .iter()
            .chain(self.debug.iter())
            .chain(self.markers.iter())
            .chain(selection)
        {
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
        if self.batches.is_empty()
            && self.debug.is_none()
            && self.markers.is_empty()
            && self.selection.is_none()
        {
            return;
        }
        pass.set_bind_group(0, &self.globals_bg, &[]);

        pass.set_pipeline(&self.poly_pipeline);
        for batch in self.batches.iter().chain(self.markers.iter()) {
            let (BatchKind::Polygon { indices, count } | BatchKind::Tube { indices, count }) =
                &batch.kind
            else {
                continue;
            };
            pass.set_bind_group(1, &self.batch_bg, &[batch.slot * BATCH_STRIDE as u32]);
            pass.set_vertex_buffer(0, batch.buffer.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..*count, 0, 0..1);
        }

        pass.set_pipeline(&self.line_pipeline);
        for batch in self.batches.iter().chain(self.debug.iter()) {
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

        if let Some((fill, edge)) = &self.selection {
            if let BatchKind::Polygon { indices, count } = &fill.kind {
                pass.set_pipeline(&self.overlay_fill_pipeline);
                pass.set_bind_group(1, &self.batch_bg, &[fill.slot * BATCH_STRIDE as u32]);
                pass.set_vertex_buffer(0, fill.buffer.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
            if let BatchKind::Dash { count } = &edge.kind {
                pass.set_pipeline(&self.overlay_dash_pipeline);
                pass.set_bind_group(1, &self.batch_bg, &[edge.slot * BATCH_STRIDE as u32]);
                pass.set_vertex_buffer(0, edge.buffer.slice(..));
                pass.draw(0..6, 0..*count);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_selection(
    gpu: &Gpu,
    anchor: DVec3,
    right: DVec3,
    forward: DVec3,
    u: f64,
    v: f64,
    height: f64,
) -> (Batch, Batch) {
    let on_surface = |s: f64, t: f64| {
        let flat = anchor + right * (u * s) + forward * (v * t);
        let (lon, lat) = dir_to_geodetic(flat);
        (geodetic_to_ecef(lon, lat, height), lon, lat)
    };
    let (origin, ..) = on_surface(0.5, 0.5);
    let fill_rgba = unpack_color(SELECT_FILL_COLOR);
    let edge_rgba = unpack_color(SELECT_EDGE_COLOR);
    let at = |s: f64, t: f64| {
        let (point, lon, lat) = on_surface(s, t);
        (
            (point - origin).as_vec3(),
            geodetic_surface_normal(lon, lat).as_vec3(),
        )
    };

    let mut verts: Vec<PolyVertex> = Vec::with_capacity((SELECT_GRID + 1) * (SELECT_GRID + 1));
    for row in 0..=SELECT_GRID {
        let t = row as f64 / SELECT_GRID as f64;
        for col in 0..=SELECT_GRID {
            let s = col as f64 / SELECT_GRID as f64;
            let (pos, nrm) = at(s, t);
            verts.push(PolyVertex {
                pos: pos.to_array(),
                nrm: nrm.to_array(),
                color: fill_rgba,
            });
        }
    }
    let stride = SELECT_GRID + 1;
    let mut indices: Vec<u32> = Vec::with_capacity(SELECT_GRID * SELECT_GRID * 6);
    for row in 0..SELECT_GRID {
        for col in 0..SELECT_GRID {
            let a = (row * stride + col) as u32;
            let b = a + 1;
            let c = ((row + 1) * stride + col) as u32 + 1;
            let d = ((row + 1) * stride + col) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    let mut loop_points: Vec<(f64, f64)> = Vec::with_capacity(SELECT_EDGE_STEPS * 4 + 1);
    let edges = [
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
    ];
    for (s0, t0, s1, t1) in edges {
        for step in 0..SELECT_EDGE_STEPS {
            let k = step as f64 / SELECT_EDGE_STEPS as f64;
            loop_points.push((s0 + (s1 - s0) * k, t0 + (t1 - t0) * k));
        }
    }
    loop_points.push((0.0, 0.0));

    let dashes: Vec<DashInstance> = loop_points
        .windows(2)
        .map(|pair| {
            let (pa, na) = at(pair[0].0, pair[0].1);
            let (pb, nb) = at(pair[1].0, pair[1].1);
            DashInstance {
                a: pa.to_array(),
                b: pb.to_array(),
                na: na.to_array(),
                nb: nb.to_array(),
                color: edge_rgba,
                width: SELECT_EDGE_WIDTH,
            }
        })
        .collect();

    let fill = Batch {
        origin,
        buffer: upload_buffer(
            gpu,
            "selection fill",
            bytemuck::cast_slice(&verts),
            wgpu::BufferUsages::VERTEX,
        ),
        kind: BatchKind::Polygon {
            indices: upload_buffer(
                gpu,
                "selection fill indices",
                bytemuck::cast_slice(&indices),
                wgpu::BufferUsages::INDEX,
            ),
            count: indices.len() as u32,
        },
        slot: SELECT_FILL_SLOT,
        vertices: verts.len() as u32,
        triangles: (indices.len() / 3) as u32,
    };
    let count = dashes.len() as u32;
    let edge = Batch {
        origin,
        buffer: upload_buffer(
            gpu,
            "selection edge",
            bytemuck::cast_slice(&dashes),
            wgpu::BufferUsages::VERTEX,
        ),
        kind: BatchKind::Dash { count },
        slot: SELECT_EDGE_SLOT,
        vertices: count * 6,
        triangles: count * 2,
    };
    (fill, edge)
}

fn build_marker(gpu: &Gpu, position: DVec3, radius: f64, color: u32, slot: u32) -> Batch {
    const RINGS: usize = 12;
    const SEGMENTS: usize = 20;
    let rgba = unpack_color(color);
    let mut verts: Vec<PolyVertex> = Vec::with_capacity((RINGS + 1) * (SEGMENTS + 1));
    let mut indices: Vec<u32> = Vec::with_capacity(RINGS * SEGMENTS * 6);
    for r in 0..=RINGS {
        let phi = std::f64::consts::PI * r as f64 / RINGS as f64;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for s in 0..=SEGMENTS {
            let theta = std::f64::consts::TAU * s as f64 / SEGMENTS as f64;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let dir = DVec3::new(sin_phi * cos_theta, sin_phi * sin_theta, cos_phi);
            verts.push(PolyVertex {
                pos: (dir * radius).as_vec3().to_array(),
                nrm: dir.as_vec3().to_array(),
                color: rgba,
            });
        }
    }
    let stride = SEGMENTS + 1;
    for r in 0..RINGS {
        for s in 0..SEGMENTS {
            let a = (r * stride + s) as u32;
            let b = (r * stride + s + 1) as u32;
            let c = ((r + 1) * stride + s + 1) as u32;
            let d = ((r + 1) * stride + s) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let vbuf = upload_buffer(
        gpu,
        "marker verts",
        bytemuck::cast_slice(&verts),
        wgpu::BufferUsages::VERTEX,
    );
    let ibuf = upload_buffer(
        gpu,
        "marker indices",
        bytemuck::cast_slice(&indices),
        wgpu::BufferUsages::INDEX,
    );
    Batch {
        origin: position,
        buffer: vbuf,
        kind: BatchKind::Polygon {
            indices: ibuf,
            count: indices.len() as u32,
        },
        slot,
        vertices: verts.len() as u32,
        triangles: indices.len() as u32 / 3,
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
