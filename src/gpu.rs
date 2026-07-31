use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth: wgpu::TextureView,
}

impl Gpu {
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(
                canvas.unchecked_into::<HtmlCanvasElement>(),
            ))
            .map_err(|e| e.to_string())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| e.to_string())?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| e.to_string())?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| "surface not supported by adapter".to_string())?;
        let caps = surface.get_capabilities(&adapter);
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        config.format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(config.format);
        config.present_mode = wgpu::PresentMode::Fifo;
        config.desired_maximum_frame_latency = 2;
        surface.configure(&device, &config);
        let depth = make_depth(&device, width, height);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            depth,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = make_depth(&self.device, width, height);
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }
}

fn make_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

pub struct UploadBudget {
    pub bytes_per_frame: usize,
    spent: usize,
}

impl UploadBudget {
    pub fn new(bytes_per_frame: usize) -> Self {
        Self {
            bytes_per_frame,
            spent: 0,
        }
    }

    pub fn reset(&mut self) {
        self.spent = 0;
    }

    pub fn fits(&self, bytes: usize) -> bool {
        self.spent == 0 || self.spent + bytes <= self.bytes_per_frame
    }

    pub fn spent(&self) -> usize {
        self.spent
    }

    pub fn take(&mut self, bytes: usize) -> bool {
        if !self.fits(bytes) {
            return false;
        }
        self.spent += bytes;
        true
    }
}

pub struct SlotAllocator {
    free: Vec<u32>,
    capacity: u32,
}

impl SlotAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            free: (0..capacity).rev().collect(),
            capacity,
        }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    pub fn free(&mut self, slot: u32) {
        if !self.free.contains(&slot) {
            self.free.push(slot);
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn used(&self) -> u32 {
        self.capacity - self.free.len() as u32
    }
}
