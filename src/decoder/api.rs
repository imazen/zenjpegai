//! One-call decoding: codestream bytes in, RGB picture out.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use whereat::{At, at};

use super::output::{RgbImage, quantize, to_rgb_planes};
use super::reconstruct::{post_process_latent, reconstruct_latent_with, synthesize_with};
use super::{Headers, decode_entropy_stage_with, read_headers};
use crate::container::Codestream;
use crate::error::Error;
use crate::filters::{self, FilterContext};
use crate::header::OperatingPoint;
use crate::mans::AnsTables;
use crate::model::synthesis::{SynthesisPrimary, SynthesisSecondary};
use crate::model::{self, CommonModel, ModelDir, ModelSource};
use crate::nn::fast::Engine;

/// Networks of one (model, operating point) pair, packed for the decoder's engine.
struct ModelSet {
    common: [CommonModel; 2],
    luma: SynthesisPrimary,
    chroma: SynthesisSecondary,
}

/// A JPEG AI decoder bound to a directory of upstream checkpoints.
///
/// Checkpoints are parsed and packed on first use and kept, so decoding many streams pays the
/// model load once per (model, operating point). Large intermediate buffers are kept between
/// decodes too; see [`Decoder::release_buffers`]. `Decoder` is `Sync`; share it between threads.
pub struct Decoder {
    models: Box<dyn ModelSource + Send + Sync>,
    engine: Engine,
    tables: AnsTables,
    operating_point: Option<OperatingPoint>,
    cache: Mutex<HashMap<(usize, OperatingPoint), Arc<ModelSet>>>,
    icci_nets: filters::icci::NetCache,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        crate::nn::fast::release_buffers();
    }
}

impl Decoder {
    /// `models_dir` is laid out like the reference repository's `models/` directory.
    /// Uses the best SIMD tier of this CPU and, with the `parallel` feature, the rayon pool.
    pub fn new(models_dir: impl Into<std::path::PathBuf>) -> Self {
        Self::with_engine(models_dir, Engine::new())
    }

    pub fn with_engine(models_dir: impl Into<std::path::PathBuf>, engine: Engine) -> Self {
        Self::with_source(Box::new(ModelDir::new(models_dir)), engine)
    }

    /// Decoder over any checkpoint source, e.g. a [`crate::model::ModelBundle`] held in memory.
    pub fn with_source(models: Box<dyn ModelSource + Send + Sync>, engine: Engine) -> Self {
        Self {
            models,
            engine,
            tables: AnsTables::new(),
            operating_point: None,
            cache: Mutex::new(HashMap::new()),
            icci_nets: Default::default(),
        }
    }

    /// Decode with this synthesis transform instead of the stream's default (its first listed
    /// one). Decoding fails if the stream does not list it.
    pub fn operating_point(mut self, op: Option<OperatingPoint>) -> Self {
        self.operating_point = op;
        self
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    fn model_set(&self, id: usize, op: OperatingPoint) -> Result<Arc<ModelSet>, Error> {
        if let Some(set) = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(&(id, op)).cloned())
        {
            return Ok(set);
        }
        let eng = &self.engine;
        let set = Arc::new(ModelSet {
            common: [
                model::load_common(&*self.models, id, 0, eng)?,
                model::load_common(&*self.models, id, 1, eng)?,
            ],
            luma: model::load_synthesis_primary(&*self.models, id, op, eng)?,
            chroma: model::load_synthesis_secondary(&*self.models, id, op, eng)?,
        });
        if let Ok(mut c) = self.cache.lock() {
            c.insert((id, op), set.clone());
        }
        Ok(set)
    }

    /// Parse the headers only.
    pub fn read_headers(&self, stream: &[u8]) -> Result<Headers, At<Error>> {
        let cs = Codestream::parse(stream).map_err(|e| at!(e))?;
        read_headers(&cs).map_err(|e| at!(e))
    }

    /// Decode a codestream to interleaved RGB at the stream's bit depth.
    pub fn decode(&self, stream: &[u8]) -> Result<RgbImage, At<Error>> {
        self.decode_inner(stream, &enough::Unstoppable)
    }

    /// [`Decoder::decode`] with cooperative cancellation.
    ///
    /// `stop` is checked per region and channel chunk in the entropy stage, per region, network
    /// layer and synthesis tile afterwards, and between the output stages; never per sample.
    /// A stop request surfaces as [`Error::Cancelled`] and leaves the decoder reusable.
    pub fn decode_with(
        &self,
        stream: &[u8],
        stop: &dyn enough::Stop,
    ) -> Result<RgbImage, At<Error>> {
        self.decode_inner(stream, stop)
    }

    /// Free the feature-map buffers kept for reuse between decodes.
    ///
    /// Decoding recycles its large intermediate buffers through a process-wide pool (at most
    /// 24 buffers and 1 GiB) because faulting fresh memory in for every picture costs 10-25 % of
    /// the decode time. The pool is also emptied when a `Decoder` is dropped.
    pub fn release_buffers(&self) {
        crate::nn::fast::release_buffers();
    }

    fn decode_inner(&self, stream: &[u8], stop: &dyn enough::Stop) -> Result<RgbImage, At<Error>> {
        stop.check().map_err(|r| at!(Error::from(r)))?;
        let cs = Codestream::parse(stream).map_err(|e| at!(e))?;
        let headers = read_headers(&cs).map_err(|e| at!(e))?;
        let hdr = &headers.picture;
        if headers.tools.lsbs_enabled.iter().any(|&e| e) {
            return Err(at!(Error::Unsupported(
                "latent scaling before synthesis (LSBS)"
            )));
        }
        if hdr.bit_depth != 8 {
            return Err(at!(Error::Unsupported("10-bit pictures")));
        }
        let default_op = *hdr
            .synthesis_transforms
            .first()
            .ok_or_else(|| at!(Error::InvalidData("no synthesis transform listed")))?;
        let op = match self.operating_point {
            None => default_op,
            Some(op) if hdr.synthesis_transforms.contains(&op) => op,
            Some(_) => {
                return Err(at!(Error::InvalidArgument(
                    "the stream does not allow the requested operating point"
                )));
            }
        };
        let set = self
            .model_set(hdr.model_id as usize, op)
            .map_err(|e| at!(e))?;
        let eng = &self.engine;

        let ent = decode_entropy_stage_with(
            &self.tables,
            &cs,
            hdr,
            [&set.common[0], &set.common[1]],
            stop,
        )
        .map_err(|e| at!(e))?;
        let [ent_y, ent_uv] = ent;
        let mut ly = reconstruct_latent_with(eng, hdr, 0, &set.common[0], &ent_y, stop)
            .map_err(|e| at!(e))?;
        post_process_latent(hdr, &headers.tools, 0, &ent_y, &mut ly).map_err(|e| at!(e))?;
        // The LEF reads the luma scale map; everything else of the entropy stage can go.
        let luma_scale_log = ent_y.scale_log;
        let mut luv = reconstruct_latent_with(eng, hdr, 1, &set.common[1], &ent_uv, stop)
            .map_err(|e| at!(e))?;
        post_process_latent(hdr, &headers.tools, 1, &ent_uv, &mut luv).map_err(|e| at!(e))?;
        drop(ent_uv);
        let planes = synthesize_with(
            eng,
            hdr,
            &set.luma,
            &set.chroma,
            [&ly.y_hat, &luv.y_hat],
            stop,
        )
        .map_err(|e| at!(e))?;
        drop((ly, luv));
        let planes = if headers.tools.any_post_filter() {
            let ctx = FilterContext {
                eng,
                hdr,
                tools: &headers.tools,
                luma_scale_log: &luma_scale_log,
                models: &*self.models,
                op,
                icci_nets: &self.icci_nets,
            };
            filters::apply(&ctx, planes).map_err(|e| at!(e))?
        } else {
            planes
        };
        stop.check().map_err(|r| at!(Error::from(r)))?;
        let rgb = to_rgb_planes(hdr, &planes).map_err(|e| at!(e))?;
        drop(planes);
        quantize(&rgb, hdr.bit_depth).map_err(|e| at!(e))
    }
}
