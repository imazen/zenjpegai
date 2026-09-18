//! Device and queue setup, native (Vulkan / Metal / DX12) and browser (WebGPU).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{GpuError, Result};

/// How to pick the adapter.
#[derive(Clone, Debug, Default)]
pub struct ContextOptions {
    /// Case-insensitive substring of the adapter name (native only; the browser exposes one
    /// adapter per power preference). `None`: discrete GPU, then integrated, then anything else.
    pub adapter_name: Option<String>,
    /// Prefer the low-power adapter where the platform offers a choice.
    pub low_power: bool,
    /// Accept software rasterisers (llvmpipe, WARP, SwiftShader). Off by default: a CPU
    /// "GPU" is slower than the crate's own CPU engine, so callers should fall back instead.
    pub allow_software: bool,
}

/// A device, its queue and the compiled kernels. Create once, share between decoders.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    pub(crate) timestamps: bool,
    pub(crate) pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
}

impl core::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter", &self.info.name)
            .field("backend", &self.info.backend)
            .field("timestamps", &self.timestamps)
            .finish()
    }
}

/// Size of one `queue.write_buffer` call, a multiple of `COPY_BUFFER_ALIGNMENT`. Chosen well
/// under every backend's write staging (Dawn splits queue writes itself); the largest uploads
/// here are a few MiB, so this almost always means exactly one call.
const WRITE_CHUNK: usize = 16 << 20;

fn is_software(info: &wgpu::AdapterInfo) -> bool {
    info.device_type == wgpu::DeviceType::Cpu
}

impl GpuContext {
    /// Find an adapter and open a device. Works on every target; in the browser this is the only
    /// constructor.
    pub async fn new_async(opts: &ContextOptions) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = Self::pick_adapter(&instance, opts).await?;
        Self::from_adapter(adapter, opts).await
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn pick_adapter(
        instance: &wgpu::Instance,
        opts: &ContextOptions,
    ) -> Result<wgpu::Adapter> {
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if let Some(want) = &opts.adapter_name {
            let want = want.to_lowercase();
            adapters.retain(|a| a.get_info().name.to_lowercase().contains(&want));
        }
        let rank = |a: &wgpu::Adapter| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 2 * opts.low_power as u8,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 3,
            wgpu::DeviceType::Other => 4,
            wgpu::DeviceType::Cpu => 5,
        };
        adapters.sort_by_key(rank);
        adapters
            .into_iter()
            .next()
            .ok_or_else(|| GpuError::NoAdapter("no adapter matches the request".into()))
    }

    #[cfg(target_arch = "wasm32")]
    async fn pick_adapter(
        instance: &wgpu::Instance,
        opts: &ContextOptions,
    ) -> Result<wgpu::Adapter> {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if opts.low_power {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::NoAdapter(e.to_string()))
    }

    /// Open a device on an adapter the caller picked (e.g. the one its canvas surface needs).
    pub async fn from_adapter(adapter: wgpu::Adapter, opts: &ContextOptions) -> Result<Self> {
        let info = adapter.get_info();
        if is_software(&info) && !opts.allow_software {
            return Err(GpuError::NoAdapter(format!(
                "only a software adapter is available ({})",
                info.name
            )));
        }
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let have = adapter.limits();
        // The large feature maps of a 1024-px HOP tile need more than WebGPU's default 128 MiB
        // per binding; take what the adapter offers, up to 1 GiB.
        let mut limits = wgpu::Limits::defaults();
        let cap = 1u64 << 30;
        limits.max_storage_buffer_binding_size = have.max_storage_buffer_binding_size.min(cap);
        limits.max_buffer_size = have.max_buffer_size.min(cap);
        limits.max_compute_invocations_per_workgroup =
            have.max_compute_invocations_per_workgroup.min(256);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("zenjpegai-gpu"),
                required_features: if timestamps {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Self::from_device(device, queue, info))
    }

    /// Wrap a device the application already owns (so decoded pictures can be presented by the
    /// application's own pipelines without a copy). Timestamp queries are used when the device
    /// was created with [`wgpu::Features::TIMESTAMP_QUERY`].
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Self {
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        Self {
            device,
            queue,
            info,
            timestamps,
            pipelines: Mutex::new(HashMap::new()),
        }
    }

    /// Blocking constructor for native targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(opts: &ContextOptions) -> Result<Self> {
        pollster::block_on(Self::new_async(opts))
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Whether per-tile GPU timestamps are available ([`crate::Timing::gpu_ns`]).
    pub fn has_timestamps(&self) -> bool {
        self.timestamps
    }

    /// Largest storage buffer this device binds, in bytes.
    pub fn max_binding_bytes(&self) -> u64 {
        self.device
            .limits()
            .max_storage_buffer_binding_size
            .min(self.device.limits().max_buffer_size)
    }

    /// Compile (or fetch) the compute pipeline for a generated WGSL source. `key` must identify
    /// the source uniquely.
    pub(crate) fn pipeline(
        &self,
        key: &str,
        source: impl FnOnce() -> String,
    ) -> wgpu::ComputePipeline {
        let mut map = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = map.get(key) {
            return p.clone();
        }
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(source().into()),
            });
        let p = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        map.insert(key.to_string(), p.clone());
        p
    }

    /// Number of compiled pipelines (cold-start accounting in the benchmarks).
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Upload `data` into `buffer` (which must carry `COPY_DST`) through
    /// [`wgpu::Queue::write_buffer`], chunked so no single call has to stage the whole thing.
    ///
    /// Uploads never go through `create_buffer_init` / `mapped_at_creation`: on the browser's
    /// WebGPU backend a buffer mapped at creation is staged through a bounded shared-memory
    /// window in Dawn, and over that limit the JS `createBuffer` throws a synchronous
    /// RangeError that wgpu `unwrap`s into an unreachable wasm trap (`panic = "abort"`, so it
    /// escapes the calling promise entirely). Queue writes have no such failure mode on
    /// either backend.
    pub(crate) fn write_buffer(&self, buffer: &wgpu::Buffer, data: &[u8]) {
        for (i, chunk) in data.chunks(WRITE_CHUNK).enumerate() {
            self.queue
                .write_buffer(buffer, (i * WRITE_CHUNK) as u64, chunk);
        }
    }

    /// Wait for a buffer mapping. Native: drive the device until the callback ran. Browser: the
    /// event loop resolves it, the future just waits.
    pub(crate) async fn map_read(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<()> {
        let (tx, rx) = futures_channel::oneshot::channel();
        buf.slice(0..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Device(e.to_string()))?;
        rx.await
            .map_err(|_| GpuError::Device("map callback dropped".into()))?
            .map_err(|e| GpuError::Device(e.to_string()))
    }

    /// `map_read` for two buffers, issuing both map requests before waiting on either — one
    /// device-drain round-trip serves both instead of two sequential ones.
    pub(crate) async fn map_read2(
        &self,
        a: &wgpu::Buffer,
        a_bytes: u64,
        b: &wgpu::Buffer,
        b_bytes: u64,
    ) -> Result<()> {
        let (tx_a, rx_a) = futures_channel::oneshot::channel();
        a.slice(0..a_bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_a.send(r);
            });
        let (tx_b, rx_b) = futures_channel::oneshot::channel();
        b.slice(0..b_bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_b.send(r);
            });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::Device(e.to_string()))?;
        for rx in [rx_a, rx_b] {
            rx.await
                .map_err(|_| GpuError::Device("map callback dropped".into()))?
                .map_err(|e| GpuError::Device(e.to_string()))?;
        }
        Ok(())
    }
}
