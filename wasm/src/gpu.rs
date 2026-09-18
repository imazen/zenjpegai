//! The `gpu` feature: WebGPU synthesis through `zenjpegai-gpu`, behind the same one-instance
//! decoder as the CPU build.
//!
//! `initGpu` opens the adapter once per worker (all of wgpu's handle types are `!Send` on
//! wasm, so the state lives in a `thread_local` — the worker is single-threaded anyway).
//! `decode` then runs the entropy / latent stages on the CPU and synthesis on the GPU, and
//! falls back to the CPU engine on any [`GpuError`]: the browser package swaps in only when a
//! non-software adapter answered, and everything the GPU path cannot do (post-filters,
//! subsampled chroma, oversized feature maps, device loss) lands back on the engine that
//! already produces correct output.
//!
//! `present` draws a decoded picture onto a transferred `OffscreenCanvas` without a CPU
//! readback: the synthesised planes become an `rgba8unorm` texture on the GPU and are blitted
//! onto the canvas surface (the blit is the only render pass; the float planes never leave the
//! device).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use zenjpegai::Picture;
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai_gpu::{Blitter, ContextOptions, GpuContext, GpuDecoder, GpuError, Timing};

use crate::{decode_cpu, js_err, rgb_to_rgba, set, set_rgba, state};

/// GPU state for the worker's lifetime: the instance (surfaces are created from it), the
/// adapter (surface capabilities come from it) and the decoder (holds uploaded weights and the
/// pooled workspace).
struct GpuInner {
    // `instance`, `adapter` and `blitter` are only read by `present_texture`, which does its
    // real work on wasm32 (wgpu surfaces are `cfg(web)`); the host build still constructs them.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    instance: wgpu::Instance,
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    adapter: wgpu::Adapter,
    ctx: Arc<GpuContext>,
    decoder: GpuDecoder,
    adapter_name: String,
    backend: String,
    software: bool,
    /// One per surface format seen (in practice a single one), reused across presents.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    blitter: RefCell<Option<(wgpu::TextureFormat, Blitter)>>,
}

thread_local! {
    static GPU: RefCell<Option<Rc<GpuInner>>> = const { RefCell::new(None) };
    /// Why `initGpu` failed, once it ran. Init is once per worker; the error is remembered so
    /// `gpuStatus` can report it and a second `initGpu` does not re-request the adapter.
    static GPU_ERR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn gpu() -> Option<Rc<GpuInner>> {
    GPU.with(|g| g.borrow().clone())
}

/// Adapter names that mean "not a GPU" when `device_type` cannot be trusted (the web backend
/// reports `Cpu` only when the browser exposes `isFallbackAdapter`).
fn looks_like_software(name: &str) -> bool {
    let n = name.to_lowercase();
    ["swiftshader", "llvmpipe", "lavapipe", "warp", "software"]
        .iter()
        .any(|s| n.contains(s))
}

fn status_of(g: &GpuInner) -> js_sys::Object {
    let out = js_sys::Object::new();
    set(&out, "ok", true);
    set(&out, "adapter", g.adapter_name.clone());
    set(&out, "backend", g.backend.clone());
    set(&out, "software", g.software);
    out
}

/// Which adapters `initGpu` may settle for: `0` rejects software (the default; a CPU
/// rasteriser is slower than this crate's own CPU engine), `1` accepts them (exercising the
/// GPU path on a GPU-less host), `2` additionally passes `forceFallbackAdapter` so the browser
/// hands back its software adapter even when a real GPU exists.
///
/// Resolves to `{ ok: true, adapter, backend, software }`; rejects with the reason when no
/// usable adapter exists or device creation fails. Calling it twice returns the first result.
#[wasm_bindgen(js_name = initGpu)]
pub async fn init_gpu(software_mode: u32) -> Result<js_sys::Object, JsError> {
    if let Some(g) = gpu() {
        return Ok(status_of(&g));
    }
    if let Some(err) = GPU_ERR.with(|e| e.borrow().clone()) {
        return Err(JsError::new(&err));
    }
    match init_inner(software_mode).await {
        Ok(g) => {
            let out = status_of(&g);
            GPU.with(|s| *s.borrow_mut() = Some(g));
            Ok(out)
        }
        Err(e) => {
            let msg = e.to_string();
            GPU_ERR.with(|s| *s.borrow_mut() = Some(msg.clone()));
            Err(JsError::new(&msg))
        }
    }
}

async fn init_inner(software_mode: u32) -> Result<Rc<GpuInner>, GpuError> {
    // panic=abort means a Rust panic traps the whole worker (no CPU fallback is possible
    // after one); at least make the message visible in the console for diagnosis.
    console_error_panic_hook::set_once();
    // `requestAdapter` rejects when `navigator.gpu` is absent or offers no adapter.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: software_mode >= 2,
            ..Default::default()
        })
        .await
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;
    let info = adapter.get_info();
    let software = info.device_type == wgpu::DeviceType::Cpu || looks_like_software(&info.name);
    if software && software_mode == 0 {
        return Err(GpuError::NoAdapter(format!(
            "only a software adapter is available ({})",
            info.name
        )));
    }
    let ctx = Arc::new(
        GpuContext::from_adapter(
            adapter.clone(),
            &ContextOptions {
                allow_software: software_mode >= 1,
                ..Default::default()
            },
        )
        .await?,
    );
    // The same bundle store feeds both decoders: `addModels` bytes are shared, the CPU engine
    // runs the entropy / latent stages (single-threaded unless the build also has `threads`).
    let engine = Engine::with(Tier::detect(), cfg!(feature = "threads"));
    let decoder = GpuDecoder::with_engine(ctx.clone(), Box::new(state().models.clone()), engine);
    Ok(Rc::new(GpuInner {
        instance,
        adapter,
        ctx,
        decoder,
        adapter_name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        software,
        blitter: RefCell::new(None),
    }))
}

/// Permanently drop the GPU context: after a Rust panic trapped through the GPU path the
/// worker keeps running, but the GPU-side state is untrusted, so every later `decode`/`present`
/// takes the CPU engine. Called by the worker's `onerror` handler; a no-op on the CPU packages
/// (which don't export it). Records the reason so `gpuStatus`/`present` report it.
#[wasm_bindgen(js_name = disableGpu)]
pub fn disable_gpu() {
    GPU.with(|g| *g.borrow_mut() = None);
    GPU_ERR.with(|e| {
        let mut slot = e.borrow_mut();
        if slot.is_none() {
            *slot = Some("disabled after a trap in the GPU path".to_string());
        }
    });
}

/// `{ ok, adapter?, backend?, software?, error? }` — whether a GPU context exists.
#[wasm_bindgen(js_name = gpuStatus)]
pub fn gpu_status() -> js_sys::Object {
    if let Some(g) = gpu() {
        return status_of(&g);
    }
    let out = js_sys::Object::new();
    set(&out, "ok", false);
    if let Some(e) = GPU_ERR.with(|e| e.borrow().clone()) {
        set(&out, "error", e);
    }
    out
}

/// `decode` for the `gpu` build: GPU synthesis when a context exists, the CPU engine
/// otherwise; any [`GpuError`] mid-decode falls back to the CPU engine and reports
/// `gpuError`.
pub async fn decode(stream: &[u8]) -> Result<js_sys::Object, JsError> {
    let Some(g) = gpu() else {
        let out = decode_cpu(stream)?;
        set(&out, "path", "cpu");
        return Ok(out);
    };
    match decode_on_gpu(&g, stream).await {
        Ok(out) => Ok(out),
        Err(e) => {
            let out = decode_cpu(stream)?;
            set(&out, "path", "cpu");
            set(&out, "gpuError", e.to_string());
            Ok(out)
        }
    }
}

/// `{width, height, rgba, path:"gpu", gpu:{tiles,dispatches,plansBuilt,gpuNs?}}`.
async fn decode_on_gpu(g: &GpuInner, stream: &[u8]) -> Result<js_sys::Object, GpuError> {
    let decoded = g.decoder.decode_to_gpu(stream)?;
    let out = js_sys::Object::new();
    if decoded.presentable_on_gpu() {
        // 8-bit 4:4:4 BT.709, no post-filters: convert to RGBA on the GPU and read back
        // 4 bytes per pixel instead of the 12 the float planes would cost.
        let (h, w) = decoded.display_size();
        let tex = decoded.to_rgba_texture()?;
        let rgba = zenjpegai_gpu::read_rgba8(&g.ctx, &tex).await?;
        let gpu_ns = decoded.picture.gpu_time().await.ok().flatten();
        set(&out, "width", w as u32);
        set(&out, "height", h as u32);
        set_rgba(&out, &rgba);
        set_timing(&out, &decoded.picture.timing, gpu_ns);
    } else {
        // Post-filters / subsampled chroma / 10 bit: read the planes back once and finish
        // exactly as the CPU decoder does.
        let (picture, _planes, timing) = g.decoder.finish(decoded).await?;
        let gpu_ns = timing.gpu_ns;
        let Picture::Rgb(img) = picture else {
            return Err(GpuError::Unsupported(
                "the stream decodes to YUV planes, not RGB",
            ));
        };
        set(&out, "width", img.width as u32);
        set(&out, "height", img.height as u32);
        set_rgba(&out, &rgb_to_rgba(&img));
        set_timing(&out, &timing, gpu_ns);
    }
    set(&out, "path", "gpu");
    Ok(out)
}

fn set_timing(out: &js_sys::Object, t: &Timing, gpu_ns: Option<u64>) {
    let tm = js_sys::Object::new();
    set(&tm, "tiles", t.tiles as u32);
    set(&tm, "dispatches", t.dispatches as u32);
    set(&tm, "plansBuilt", t.plans_built as u32);
    if let Some(ns) = gpu_ns {
        set(&tm, "gpuNs", ns as f64);
    }
    set(out, "gpu", tm);
}

/// Decode `stream` and draw it onto `canvas` (a transferred `OffscreenCanvas`), without any
/// CPU readback. Resolves to `{width, height, path:"gpu", presented:"gpu"}`. Rejects when
/// there is no GPU context or the picture is not GPU-presentable — the caller falls back to
/// `decode` + a 2d `putImageData` on the same canvas.
#[wasm_bindgen(js_name = present)]
pub async fn present(
    stream: Vec<u8>,
    canvas: web_sys::OffscreenCanvas,
) -> Result<js_sys::Object, JsError> {
    let g = gpu().ok_or_else(|| {
        let why = GPU_ERR
            .with(|e| e.borrow().clone())
            .unwrap_or_else(|| "initGpu has not run".to_string());
        JsError::new(&format!("no GPU context: {why}"))
    })?;
    let decoded = g.decoder.decode_to_gpu(stream.as_slice()).map_err(js_err)?;
    if !decoded.presentable_on_gpu() {
        return Err(JsError::new(
            "GPU canvas presentation needs an 8-bit 4:4:4 BT.709 picture without post-filters",
        ));
    }
    let (h, w) = decoded.display_size();
    let tex = decoded.to_rgba_texture().map_err(js_err)?;
    // Waits for the GPU (the blit below is queued behind the synthesis work regardless);
    // fills in `gpuNs` for the timing report.
    let gpu_ns = decoded.picture.gpu_time().await.map_err(js_err)?;
    let timing = decoded.picture.timing;
    canvas.set_width(w as u32);
    canvas.set_height(h as u32);
    present_texture(&g, canvas, &tex).map_err(js_err)?;
    let out = js_sys::Object::new();
    set(&out, "width", w as u32);
    set(&out, "height", h as u32);
    set(&out, "path", "gpu");
    set(&out, "presented", "gpu");
    set_timing(&out, &timing, gpu_ns);
    Ok(out)
}

/// Blit `tex` onto a canvas surface sharing the decoder's device and present it.
#[cfg(target_arch = "wasm32")]
fn present_texture(
    g: &GpuInner,
    canvas: web_sys::OffscreenCanvas,
    tex: &wgpu::Texture,
) -> Result<(), GpuError> {
    let surface = g
        .instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
        .map_err(|e| GpuError::Device(e.to_string()))?;
    let caps = surface.get_capabilities(&g.adapter);
    let format = caps
        .formats
        .first()
        .copied()
        .ok_or_else(|| GpuError::Device("the canvas surface offers no formats".into()))?;
    surface.configure(
        g.ctx.device(),
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: tex.width(),
            height: tex.height(),
            present_mode: caps
                .present_modes
                .first()
                .copied()
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Opaque),
            view_formats: vec![],
        },
    );
    let frame = match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        other => {
            return Err(GpuError::Device(format!(
                "no current surface texture: {other:?}"
            )));
        }
    };
    let mut slot = g.blitter.borrow_mut();
    if slot.as_ref().map(|(f, _)| *f) != Some(format) {
        *slot = Some((format, Blitter::new(&g.ctx, format)));
    }
    let (_, blitter) = slot.as_ref().expect("blitter was just stored");
    blitter.blit(&g.ctx, tex, &frame.texture.create_view(&Default::default()));
    g.ctx.queue().present(frame);
    Ok(())
}

/// Canvas presentation exists only on the browser target (wgpu surfaces are `cfg(web)`).
#[cfg(not(target_arch = "wasm32"))]
fn present_texture(
    _g: &GpuInner,
    _canvas: web_sys::OffscreenCanvas,
    _tex: &wgpu::Texture,
) -> Result<(), GpuError> {
    Err(GpuError::Unsupported(
        "canvas presentation requires the wasm32 build",
    ))
}

/// Drop the GPU workspace buffers (called by `releaseBuffers`).
pub fn release_buffers() {
    if let Some(g) = gpu() {
        g.decoder.release_buffers();
    }
}
