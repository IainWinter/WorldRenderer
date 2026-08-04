use crate::gpu::{Gpu, SlotAllocator, DEPTH_FORMAT};
use crate::terrain_mesh::{build_indices, SLOT_SIZE, TOTAL_INDICES};
use glam::Mat4;

pub const TERRAIN_SLOTS: u32 = 1400;
pub const IMAGERY_LAYERS: u32 = 768;
pub const IMAGERY_SIZE: u32 = 256;
pub const MAX_DRAWN_TILES: u32 = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [f32; 16],
    pub sun_dir: [f32; 4],
    pub screen: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileInstance {
    pub origin: [f32; 3],
    pub morph_lo: f32,
    pub uvxf: [f32; 4],
    pub prev_uvxf: [f32; 4],
    pub layers: [f32; 2],
    pub blend: [f32; 4],
    pub dbg: [f32; 4],
}

pub const INSTANCE_SIZE: u64 = std::mem::size_of::<TileInstance>() as u64;

pub struct TerrainRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub globals: wgpu::Buffer,
    pub arena: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub atlas: wgpu::Texture,
    pub slots: SlotAllocator,
    pub layers: SlotAllocator,
}

impl TerrainRenderer {
    pub fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let layer_cap = IMAGERY_LAYERS.min(device.limits().max_texture_array_layers);

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain arena"),
            size: (SLOT_SIZE as u64) * (TERRAIN_SLOTS as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_data = build_indices();
        let index_bytes: &[u8] = bytemuck::cast_slice(&index_data);
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain indices"),
            size: index_bytes.len() as u64,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        indices
            .slice(..)
            .get_mapped_range_mut()
            .unwrap()
            .copy_from_slice(index_bytes);
        indices.unmap();

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain instances"),
            size: (std::mem::size_of::<TileInstance>() as u64) * (MAX_DRAWN_TILES as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imagery atlas"),
            size: wgpu::Extent3d {
                width: IMAGERY_SIZE,
                height: IMAGERY_SIZE,
                depth_or_array_layers: layer_cap,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("imagery sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Snorm16x2,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Unorm16x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 20,
                                shader_location: 3,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: INSTANCE_SIZE,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 12,
                                shader_location: 5,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 6,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 7,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 48,
                                shader_location: 8,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 56,
                                shader_location: 9,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 72,
                                shader_location: 10,
                            },
                        ],
                    }),
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,
            globals,
            arena,
            indices,
            instances,
            atlas,
            slots: SlotAllocator::new(TERRAIN_SLOTS),
            layers: SlotAllocator::new(layer_cap),
        }
    }

    pub fn write_globals(&self, gpu: &Gpu, view_proj: Mat4, sun: glam::Vec3, w: f32, h: f32) {
        let globals = Globals {
            view_proj: view_proj.to_cols_array(),
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            screen: [w, h, 1.0 / w, 1.0 / h],
        };
        gpu.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
    }

    pub fn upload_mesh(&self, gpu: &Gpu, slot: u32, bytes: &[u8]) {
        gpu.queue
            .write_buffer(&self.arena, (slot as u64) * (SLOT_SIZE as u64), bytes);
    }

    pub fn upload_imagery(&self, gpu: &Gpu, layer: u32, pixels: &[u8]) {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(IMAGERY_SIZE * 4),
                rows_per_image: Some(IMAGERY_SIZE),
            },
            wgpu::Extent3d {
                width: IMAGERY_SIZE,
                height: IMAGERY_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn write_instances(&self, gpu: &Gpu, data: &[TileInstance]) {
        if !data.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(data));
        }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, slots: &[u32]) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        for (i, slot) in slots.iter().enumerate() {
            let vb = (*slot as u64) * (SLOT_SIZE as u64);
            let ib = (i as u64) * INSTANCE_SIZE;
            pass.set_vertex_buffer(0, self.arena.slice(vb..vb + SLOT_SIZE as u64));
            pass.set_vertex_buffer(1, self.instances.slice(ib..ib + INSTANCE_SIZE));
            pass.draw_indexed(0..TOTAL_INDICES, 0, 0..1);
        }
    }
}
