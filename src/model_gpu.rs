use crate::gpu::{Gpu, DEPTH_FORMAT};
use crate::model::ModelData;
use glam::{DVec3, Mat3, Quat, Vec3};

pub const MAX_INSTANCES: u64 = 4096;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelInstance {
    pub row0: [f32; 4],
    pub row1: [f32; 4],
    pub row2: [f32; 4],
    pub color: [f32; 4],
}

pub struct Model {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,
    pub bind_group: wgpu::BindGroup,
    pub instances: wgpu::Buffer,
    pub live: Vec<ModelInstance>,
}

pub struct ModelRenderer {
    pipeline: wgpu::RenderPipeline,
    globals_bg: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub models: Vec<Model>,
}

impl ModelRenderer {
    pub fn new(gpu: &Gpu, globals: &wgpu::Buffer) -> Self {
        let device = &gpu.device;

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model globals bgl"),
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

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model texture bgl"),
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

        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model globals bg"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/model.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model pl"),
            bind_group_layouts: &[Some(&globals_layout), Some(&texture_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model pipeline"),
            layout: Some(&layout),
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
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 24,
                                shader_location: 2,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 0,
                                shader_location: 3,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 4,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 32,
                                shader_location: 5,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 48,
                                shader_location: 6,
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
            globals_bg,
            texture_layout,
            sampler,
            models: Vec::new(),
        }
    }

    pub fn upload(&mut self, gpu: &Gpu, slot: usize, data: &ModelData) {
        let device = &gpu.device;
        let vertices = make_buffer(
            device,
            "model verts",
            bytemuck::cast_slice(&data.vertices),
            wgpu::BufferUsages::VERTEX,
        );
        let indices = make_buffer(
            device,
            "model indices",
            bytemuck::cast_slice(&data.indices),
            wgpu::BufferUsages::INDEX,
        );
        let (w, h, pixels) = match &data.texture {
            Some((w, h, p)) => (*w, *h, p.clone()),
            None => (1, 1, vec![255u8, 255, 255, 255]),
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
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
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model texture bg"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model instances"),
            size: 64 * MAX_INSTANCES,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        while self.models.len() <= slot {
            self.models.push(Model {
                vertices: make_buffer(device, "empty", &[0u8; 4], wgpu::BufferUsages::VERTEX),
                indices: make_buffer(device, "empty", &[0u8; 4], wgpu::BufferUsages::INDEX),
                index_count: 0,
                bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }),
                instances: device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                live: Vec::new(),
            });
        }

        let existing = std::mem::take(&mut self.models[slot].live);
        self.models[slot] = Model {
            vertices,
            indices,
            index_count: data.indices.len() as u32,
            bind_group,
            instances,
            live: existing,
        };
    }

    pub fn ready(&self, slot: usize) -> bool {
        self.models
            .get(slot)
            .map(|m| m.index_count > 0)
            .unwrap_or(false)
    }

    pub fn write_instances(&self, gpu: &Gpu) {
        for model in self.models.iter() {
            if !model.live.is_empty() {
                gpu.queue
                    .write_buffer(&model.instances, 0, bytemuck::cast_slice(&model.live));
            }
        }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        let mut bound = false;
        for model in self.models.iter() {
            if model.index_count == 0 || model.live.is_empty() {
                continue;
            }
            if !bound {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.globals_bg, &[]);
                bound = true;
            }
            pass.set_bind_group(1, &model.bind_group, &[]);
            pass.set_vertex_buffer(0, model.vertices.slice(..));
            pass.set_vertex_buffer(1, model.instances.slice(..));
            pass.set_index_buffer(model.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..model.index_count, 0, 0..model.live.len() as u32);
        }
    }
}

pub fn instance_from_transform(
    position: DVec3,
    eye: DVec3,
    rotation: Quat,
    scale: f32,
    color: [f32; 4],
) -> ModelInstance {
    let basis = Mat3::from_quat(rotation) * scale;
    let offset = (position - eye).as_vec3();
    ModelInstance {
        row0: [basis.x_axis.x, basis.y_axis.x, basis.z_axis.x, offset.x],
        row1: [basis.x_axis.y, basis.y_axis.y, basis.z_axis.y, offset.y],
        row2: [basis.x_axis.z, basis.y_axis.z, basis.z_axis.z, offset.z],
        color,
    }
}

pub fn model_to_enu() -> Quat {
    Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

pub fn up_from(v: Vec3) -> Vec3 {
    if v.length_squared() > 0.0 {
        v.normalize()
    } else {
        Vec3::Z
    }
}

fn make_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let mut padded = bytes.to_vec();
    while padded.len() < 4 || padded.len() % 4 != 0 {
        padded.push(0);
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
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
