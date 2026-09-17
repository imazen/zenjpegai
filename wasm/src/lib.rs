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
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{self, ModelSource};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::PackedBundle;

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

/// `"threads"` or `"simd"`: which build this is.
#[wasm_bindgen(js_name = buildMode)]
pub fn build_mode() -> String {
    if cfg!(feature = "threads") {
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

/// Header fields of a codestream: `{ width, height, bitDepth, modelId, operatingPoint,
/// operatingPoints }`. `operatingPoint` is the one [`decode`] uses (the stream's first listed).
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
    let ops = js_sys::Array::new();
    for &op in &pic.synthesis_transforms {
        ops.push(&JsValue::from_str(op_name(op)));
    }
    set(&out, "operatingPoint", ops.get(0));
    set(&out, "operatingPoints", ops);
    Ok(out)
}

/// Decode a codestream: `{ width, height, rgba: Uint8ClampedArray }` (alpha 255), ready for
/// `new ImageData(rgba, width, height)`.
#[wasm_bindgen]
pub fn decode(stream: &[u8]) -> Result<js_sys::Object, JsError> {
    let img = state().decoder.decode(stream).map_err(js_err)?;
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
    let out = js_sys::Object::new();
    set(&out, "width", img.width as u32);
    set(&out, "height", img.height as u32);
    let array = js_sys::Uint8ClampedArray::new_with_length(rgba.len() as u32);
    array.copy_from(&rgba);
    set(&out, "rgba", array);
    Ok(out)
}

/// Drop the feature-map buffers kept between decodes (wasm memory itself never shrinks; this
/// lets the next, smaller picture reuse the space).
#[wasm_bindgen(js_name = releaseBuffers)]
pub fn release_buffers() {
    state().decoder.release_buffers();
}
