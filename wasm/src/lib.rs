//! WebAssembly bindings of the zenjpegai decoder.
//!
//! One module instance holds one decoder: model bundles (`ZJB1`, written by
//! `zenjpegai pack-models`) are handed over with [`add_models`], streams are inspected with
//! [`info`] (no weights needed: a page fetches only the bundles its streams name) and decoded
//! with [`decode`]. Meant to live in a Web Worker: a decode blocks its thread for its duration.
//!
//! Two builds of this one source: the default single-threaded SIMD128 build, which runs anywhere
//! WebAssembly does, and the `threads` build (rayon on Web Workers over a shared memory), which
//! needs a cross-origin isolated page. Both produce the same pixels (wasm numeric policy,
//! `PORTING.md`).

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::sync::{Arc, OnceLock, RwLock};

use wasm_bindgen::prelude::*;
use zenjpegai::Decoder;
use zenjpegai::RgbImage;
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{self, ModelSource};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::PackedBundle;

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Bundles added so far. `read` copies the (few MB) file out from under the lock; that happens
/// once per (model, operating point), after which the decoder keeps the packed networks.
#[derive(Clone, Default)]
struct Shared(Arc<RwLock<PackedBundle>>);

impl ModelSource for Shared {
    fn read(&self, rel: &str) -> Result<Cow<'_, [u8]>, zenjpegai::Error> {
        let guard = self
            .0
            .read()
            .map_err(|_| zenjpegai::Error::Model("model store poisoned".into()))?;
        Ok(Cow::Owned(guard.read(rel)?.into_owned()))
    }
}

struct State {
    models: Shared,
    decoder: Decoder,
}

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(|| {
        let models = Shared::default();
        let engine = Engine::with(Tier::detect(), cfg!(feature = "threads"));
        State {
            decoder: Decoder::with_source(Box::new(models.clone()), engine),
            models,
        }
    })
}

fn js_err(e: impl core::fmt::Debug) -> JsError {
    JsError::new(&format!("{e:?}"))
}

fn op_name(op: OperatingPoint) -> &'static str {
    match op {
        OperatingPoint::Sop => "sop",
        OperatingPoint::Bop => "bop",
        OperatingPoint::Hop => "hop",
    }
}

/// `"threads"`, `"webgpu"` or `"simd"`: which build this is.
#[wasm_bindgen(js_name = buildMode)]
pub fn build_mode() -> String {
    if cfg!(feature = "gpu") {
        "webgpu".into()
    } else if cfg!(feature = "threads") {
        "threads".into()
    } else {
        "simd".into()
    }
}

/// SIMD tier the engine resolved to (`Wasm128` unless built without `simd128`).
#[wasm_bindgen(js_name = simdTier)]
pub fn simd_tier() -> String {
    format!("{:?}", state().decoder.engine().tier)
        .split('(')
        .next()
        .unwrap_or("")
        .to_string()
}

/// Add a `ZJB1` model bundle. Returns the upstream licence notice it carries.
#[wasm_bindgen(js_name = addModels)]
pub fn add_models(bundle: Vec<u8>) -> Result<String, JsError> {
    let mut store = state()
        .models
        .0
        .write()
        .map_err(|_| JsError::new("model store poisoned"))?;
    store.add(bundle).map_err(js_err)?;
    Ok(store.notice().to_string())
}

/// Whether the four checkpoints of (`model_id`, `op`) have been added.
#[wasm_bindgen(js_name = hasModels)]
pub fn has_models(model_id: usize, op: &str) -> Result<bool, JsError> {
    let op = parse_op(op)?;
    let store = state()
        .models
        .0
        .read()
        .map_err(|_| JsError::new("model store poisoned"))?;
    let mut all = true;
    for ccs in 0..2 {
        all &= store.contains(&model::common_path(model_id, ccs).map_err(js_err)?);
        all &= store.contains(&model::synthesis_path(model_id, ccs, op).map_err(js_err)?);
    }
    Ok(all)
}

fn parse_op(op: &str) -> Result<OperatingPoint, JsError> {
    match op {
        "sop" => Ok(OperatingPoint::Sop),
        "bop" => Ok(OperatingPoint::Bop),
        "hop" => Ok(OperatingPoint::Hop),
        _ => Err(JsError::new("operating point must be sop, bop or hop")),
    }
}

fn set(obj: &js_sys::Object, key: &str, value: impl Into<JsValue>) {
    // Setting a data property on a fresh plain object cannot fail.
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &value.into());
}

/// The coded chroma subsampling label for `info()` (factors are 1 or 2).
fn chroma_format(c_ver: u8, c_hor: u8) -> String {
    match (c_ver, c_hor) {
        (1, 1) => "4:4:4".to_string(),
        (1, 2) => "4:2:2".to_string(),
        (2, 2) => "4:2:0".to_string(),
        (2, 1) => "4:4:0".to_string(),
        (v, h) => format!("subsamp {v}x{h}"),
    }
}

fn memory_obj(r: zenjpegai::MemoryReport) -> js_sys::Object {
    let out = js_sys::Object::new();
    set(&out, "trackedPeakBytes", r.tracked_peak_bytes as f64);
    set(&out, "trackedLiveBytes", r.tracked_live_bytes as f64);
    set(&out, "poolBytes", r.pool_bytes as f64);
    out
}

/// Predicted heap of decoding `stream` (`zenjpegai::estimate_memory`):
/// `{ liveBytes, peakBytes }`. `liveBytes` is what admission control grants by; `peakBytes`
/// adds the recycle pool's cap. Needs no weights.
#[wasm_bindgen(js_name = estimateMemory)]
pub fn estimate_memory(stream: &[u8]) -> Result<js_sys::Object, JsError> {
    let est = state().decoder.estimate_memory(stream).map_err(js_err)?;
    let out = js_sys::Object::new();
    set(&out, "liveBytes", est.live_bytes as f64);
    set(&out, "peakBytes", est.peak_bytes() as f64);
    Ok(out)
}

/// This module's cumulative tracked-heap counters (`zenjpegai::MemoryReport`):
/// `{ trackedPeakBytes, trackedLiveBytes, poolBytes }`.
#[wasm_bindgen(js_name = memoryReport)]
pub fn memory_report() -> js_sys::Object {
    memory_obj(state().decoder.memory_report())
}

/// Header fields of a codestream: `{ width, height, bitDepth, modelId, operatingPoint,
/// operatingPoints, chromaFormat, postFilters }`. `operatingPoint` is the one [`decode`]
/// uses (the stream's first listed). `chromaFormat` is the coded subsampling; `postFilters`
/// is whether the stream switches any post-filter on (they run on the CPU after synthesis).
/// Needs no weights.
#[wasm_bindgen]
pub fn info(stream: &[u8]) -> Result<js_sys::Object, JsError> {
    let headers = state().decoder.read_headers(stream).map_err(js_err)?;
    let pic = &headers.picture;
    let out = js_sys::Object::new();
    set(&out, "width", pic.width);
    set(&out, "height", pic.height);
    set(&out, "bitDepth", pic.bit_depth as u32);
    set(&out, "modelId", pic.model_id as u32);
    set(&out, "chromaFormat", chroma_format(pic.c_ver, pic.c_hor));
    set(&out, "postFilters", headers.tools.any_post_filter());
    let ops = js_sys::Array::new();
    for &op in &pic.synthesis_transforms {
        ops.push(&JsValue::from_str(op_name(op)));
    }
    set(&out, "operatingPoint", ops.get(0));
    set(&out, "operatingPoints", ops);
    Ok(out)
}

/// Interleaved `RgbImage` at any bit depth as RGBA bytes for `ImageData` (alpha 255).
fn rgb_to_rgba(img: &RgbImage) -> Vec<u8> {
    let shift = img.bit_depth.saturating_sub(8) as u32;
    let mut rgba = vec![255u8; img.width * img.height * 4];
    for (dst, src) in rgba
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(img.data.as_chunks::<3>().0)
    {
        for c in 0..3 {
            dst[c] = (src[c] >> shift).min(255) as u8;
        }
    }
    rgba
}

fn set_rgba(out: &js_sys::Object, rgba: &[u8]) {
    let array = js_sys::Uint8ClampedArray::new_with_length(rgba.len() as u32);
    array.copy_from(rgba);
    set(out, "rgba", array);
}

/// Per-stage wall times of one decode, ms (a flat snapshot of `DecodeStats`; `[2]` fields
/// are per component — luma, chroma — and overlap each other on the thread pool, so their
/// sum can exceed `chains`). On builds without the `wasm-clock` feature every field is 0.
fn stats_obj(s: &zenjpegai::DecodeStats) -> js_sys::Object {
    let ms = |d: core::time::Duration| d.as_secs_f64() * 1000.0;
    let pair = |a: &[core::time::Duration; 2]| {
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from_f64(ms(a[0])));
        arr.push(&JsValue::from_f64(ms(a[1])));
        arr
    };
    let out = js_sys::Object::new();
    set(&out, "headers", ms(s.headers));
    set(&out, "models", ms(s.models));
    set(&out, "entropyZ", pair(&s.entropy_z));
    set(&out, "entropyQmap", ms(s.entropy_quality_map));
    set(&out, "entropyBody", pair(&s.entropy_body));
    set(&out, "entropyResidual", pair(&s.entropy_residual));
    set(&out, "entropyDequantize", pair(&s.entropy_dequantize));
    set(&out, "latent", pair(&s.latent));
    set(&out, "latentHyper", pair(&s.latent_hyper));
    set(&out, "latentMcm", pair(&s.latent_mcm));
    set(&out, "lsbs", pair(&s.post_process));
    set(&out, "chains", ms(s.chains));
    set(&out, "synthLuma", ms(s.synthesis_luma));
    set(&out, "synthesis", ms(s.synthesis));
    set(&out, "chroma", ms(s.chroma_convert));
    set(&out, "filters", ms(s.filters));
    set(&out, "output", ms(s.output));
    set(&out, "total", ms(s.total));
    out
}

/// `0` for "all coded channels" in the `decodePartial`/`presentPartial` arguments.
pub(crate) fn channel_cap(y: u16, uv: u16) -> [Option<u16>; 2] {
    [(y > 0).then_some(y), (uv > 0).then_some(uv)]
}

/// The CPU decode behind [`decode`], shared with the GPU path's fallback. `max_channels`
/// caps the latent channels decoded per component (progressive decode, `num_decode_chs`).
fn decode_cpu_opts(
    stream: &[u8],
    max_channels: Option<[Option<u16>; 2]>,
) -> Result<js_sys::Object, JsError> {
    let (picture, stats) = match max_channels {
        Some(mc) => state().decoder.decode_picture_stats_progressive(stream, mc),
        None => state().decoder.decode_picture_stats(stream),
    }
    .map_err(js_err)?;
    let zenjpegai::Picture::Rgb(img) = picture else {
        return Err(JsError::new("the stream decodes to YUV planes, not RGB"));
    };
    let out = js_sys::Object::new();
    set(&out, "width", img.width as u32);
    set(&out, "height", img.height as u32);
    set_rgba(&out, &rgb_to_rgba(&img));
    set(&out, "stats", stats_obj(&stats));
    // `stats.memory` is the per-call tracked-heap report (MemoryWatch over the decode).
    set(&out, "memory", memory_obj(stats.memory));
    Ok(out)
}

/// Decode a codestream: `{ width, height, rgba: Uint8ClampedArray, path, stats }` (alpha 255),
/// ready for `new ImageData(rgba, width, height)`. In the `gpu` build this is async (a
/// Promise) and `path` is `"gpu"` when synthesis ran on WebGPU or `"cpu"` after a fallback
/// (`gpuError` carries why); otherwise it is synchronous and `path` is `"cpu"`. `stats` is
/// the CPU stages' wall-time breakdown (present on the CPU path).
#[cfg(not(feature = "gpu"))]
#[wasm_bindgen]
pub fn decode(stream: &[u8]) -> Result<js_sys::Object, JsError> {
    let out = decode_cpu_opts(stream, None)?;
    set(&out, "path", "cpu");
    Ok(out)
}

/// See the non-`gpu` doc above.
#[cfg(feature = "gpu")]
#[wasm_bindgen]
pub async fn decode(stream: Vec<u8>) -> Result<js_sys::Object, JsError> {
    let watch = zenjpegai::MemoryWatch::new();
    let out = gpu::decode_opts(&stream, None).await?;
    set(&out, "memory", memory_obj(watch.report()));
    Ok(out)
}

/// [`decode`] reading only the first `max_y` / `max_uv` latent channels of each component
/// (`0` = all): the reference's `num_decode_chs` progressive decode — a coarser picture for
/// less entropy-stage work.
#[cfg(not(feature = "gpu"))]
#[wasm_bindgen(js_name = decodePartial)]
pub fn decode_partial(stream: &[u8], max_y: u16, max_uv: u16) -> Result<js_sys::Object, JsError> {
    let out = decode_cpu_opts(stream, Some(channel_cap(max_y, max_uv)))?;
    set(&out, "path", "cpu");
    Ok(out)
}

/// See the non-`gpu` doc above.
#[cfg(feature = "gpu")]
#[wasm_bindgen(js_name = decodePartial)]
pub async fn decode_partial(
    stream: Vec<u8>,
    max_y: u16,
    max_uv: u16,
) -> Result<js_sys::Object, JsError> {
    let watch = zenjpegai::MemoryWatch::new();
    let out = gpu::decode_opts(&stream, Some(channel_cap(max_y, max_uv))).await?;
    set(&out, "memory", memory_obj(watch.report()));
    Ok(out)
}

/// Drop the feature-map buffers kept between decodes (wasm memory itself never shrinks; this
/// lets the next, smaller picture reuse the space).
#[wasm_bindgen(js_name = releaseBuffers)]
pub fn release_buffers() {
    state().decoder.release_buffers();
    #[cfg(feature = "gpu")]
    gpu::release_buffers();
}
