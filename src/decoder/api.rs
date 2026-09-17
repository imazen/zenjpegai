//! One-call decoding: codestream bytes in, RGB picture out.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use whereat::{At, at};

use super::entropy::ComponentEntropy;
use super::limits::{Limits, MemoryEstimate};
use super::output::{Picture, RgbImage, finish, to_source_format};
use super::reconstruct::{post_process_latent, reconstruct_latent_with, synthesize_with};
use super::{Headers, decode_entropy_stage_progressive, read_headers};
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
    max_channels: [Option<u16>; 2],
    limits: Limits,
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
            max_channels: [None, None],
            limits: Limits::default(),
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

    /// Progressive decode: read only the first `luma` / `chroma` latent channels (the
    /// reference's `num_decode_chs`). Channels are coded in order of importance, so a prefix
    /// gives a coarser picture for less entropy-decoding work. `None` decodes all of them.
    pub fn max_channels(mut self, luma: Option<u16>, chroma: Option<u16>) -> Self {
        self.max_channels = [luma, chroma];
        self
    }

    /// Resource limits for every decode of this decoder (default: [`Limits::default`]).
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Predicted heap use of decoding `stream` with this decoder's operating point, from the
    /// headers alone (see [`crate::estimate_memory`]).
    pub fn estimate_memory(&self, stream: &[u8]) -> Result<MemoryEstimate, At<Error>> {
        let headers = self.read_headers(stream)?;
        let op = self.pick_operating_point(&headers)?;
        Ok(crate::estimate_memory(&headers.picture, op))
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

    /// Load and pack the networks `stream` needs, without decoding it. Decoding does this on
    /// first use; calling it up front moves the cost (tens of milliseconds and the models'
    /// memory) out of the first decode.
    pub fn preload(&self, stream: &[u8]) -> Result<(), At<Error>> {
        let headers = self.read_headers(stream)?;
        let op = self.pick_operating_point(&headers)?;
        self.model_set(headers.picture.model_id as usize, op)
            .map_err(|e| at!(e))?;
        Ok(())
    }

    /// The synthesis transform a decode of this stream would run.
    fn pick_operating_point(&self, headers: &Headers) -> Result<OperatingPoint, At<Error>> {
        let hdr = &headers.picture;
        let default_op = *hdr
            .synthesis_transforms
            .first()
            .ok_or_else(|| at!(Error::InvalidData("no synthesis transform listed")))?;
        match self.operating_point {
            None => Ok(default_op),
            Some(op) if hdr.synthesis_transforms.contains(&op) => Ok(op),
            Some(_) => Err(at!(Error::InvalidArgument(
                "the stream does not allow the requested operating point"
            ))),
        }
    }

    /// Parse the headers only.
    pub fn read_headers(&self, stream: &[u8]) -> Result<Headers, At<Error>> {
        let cs = Codestream::parse(stream).map_err(|e| at!(e))?;
        read_headers(&cs).map_err(|e| at!(e))
    }

    /// Decode a codestream to interleaved RGB at the stream's bit depth.
    ///
    /// Streams coded from a YUV source decode to YUV planes, not RGB: for those this returns
    /// `Error::Unsupported`; use [`Decoder::decode_picture`].
    pub fn decode(&self, stream: &[u8]) -> Result<RgbImage, At<Error>> {
        self.decode_with(stream, &enough::Unstoppable)
    }

    /// Decode a codestream to whatever it holds: RGB, or YUV planes in the source's chroma
    /// subsampling (4:4:4, 4:2:2 or 4:2:0), 8 or 10 bits per sample.
    pub fn decode_picture(&self, stream: &[u8]) -> Result<Picture, At<Error>> {
        self.decode_inner(stream, &enough::Unstoppable)
    }

    /// [`Decoder::decode_picture`] with cooperative cancellation (see [`Decoder::decode_with`]).
    pub fn decode_picture_with(
        &self,
        stream: &[u8],
        stop: &dyn enough::Stop,
    ) -> Result<Picture, At<Error>> {
        self.decode_inner(stream, stop)
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
        match self.decode_inner(stream, stop)? {
            Picture::Rgb(image) => Ok(image),
            Picture::Yuv(_) => Err(at!(Error::Unsupported(
                "the stream decodes to YUV planes: use decode_picture"
            ))),
        }
    }

    /// Free the feature-map buffers kept for reuse between decodes.
    ///
    /// Decoding recycles its large intermediate buffers through a process-wide pool (at most
    /// 24 buffers and 1 GiB) because faulting fresh memory in for every picture costs 10-25 % of
    /// the decode time. The pool is also emptied when a `Decoder` is dropped.
    pub fn release_buffers(&self) {
        crate::nn::fast::release_buffers();
    }

    fn decode_inner(&self, stream: &[u8], stop: &dyn enough::Stop) -> Result<Picture, At<Error>> {
        stop.check().map_err(|r| at!(Error::from(r)))?;
        self.limits.check_input(stream.len()).map_err(|e| at!(e))?;
        let cs = Codestream::parse(stream).map_err(|e| at!(e))?;
        let headers = read_headers(&cs).map_err(|e| at!(e))?;
        let hdr = &headers.picture;
        let op = self.pick_operating_point(&headers)?;
        // Limits are judged on the header alone, before any picture-sized allocation.
        let estimate = self.limits.check_header(hdr, op).map_err(|e| at!(e))?;
        if let Some(max) = self.limits.max_memory_bytes {
            let room = usize::try_from(max - estimate.live_bytes).unwrap_or(usize::MAX);
            if crate::nn::fast::pool_limit() > room {
                crate::nn::fast::set_pool_limit(room);
            }
        }
        let set = self
            .model_set(hdr.model_id as usize, op)
            .map_err(|e| at!(e))?;
        let eng = &self.engine;

        let ent = decode_entropy_stage_progressive(
            &self.tables,
            &cs,
            hdr,
            [&set.common[0], &set.common[1]],
            self.max_channels,
            stop,
        )
        .map_err(|e| at!(e))?;
        let [mut ent_y, mut ent_uv] = ent;
        // Reconstruction reads the hyper-latent and the residual (LSBS also the `likely` map),
        // the LEF reads the luma scale map; the rest of the entropy stage's output is dead.
        let filters = headers.tools.any_post_filter();
        shed_entropy(&mut ent_y, filters).map_err(|e| at!(e))?;
        shed_entropy(&mut ent_uv, false).map_err(|e| at!(e))?;
        let mut ly = reconstruct_latent_with(eng, hdr, 0, &set.common[0], &ent_y, stop)
            .map_err(|e| at!(e))?;
        post_process_latent(hdr, &headers.tools, 0, &ent_y, &mut ly).map_err(|e| at!(e))?;
        ly.psi = crate::tensor::Tensor::zeros(0, 0, 0).map_err(|e| at!(e))?;
        let luma_scale_log = ent_y.scale_log;
        drop((ent_y.z_hat, ent_y.residual, ent_y.likely));
        let mut luv = reconstruct_latent_with(eng, hdr, 1, &set.common[1], &ent_uv, stop)
            .map_err(|e| at!(e))?;
        post_process_latent(hdr, &headers.tools, 1, &ent_uv, &mut luv).map_err(|e| at!(e))?;
        luv.psi = crate::tensor::Tensor::zeros(0, 0, 0).map_err(|e| at!(e))?;
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
        // Coded chroma format -> source chroma format, then the post-filters, then colour.
        let planes = to_source_format(hdr, planes).map_err(|e| at!(e))?;
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
        finish(hdr, &planes).map_err(|e| at!(e))
    }
}

/// Free the entropy-stage tensors nothing downstream reads.
fn shed_entropy(e: &mut ComponentEntropy, keep_scale_log: bool) -> Result<(), Error> {
    e.skip_scale_log = crate::tensor::Tensor::zeros(0, 0, 0)?;
    e.mask = crate::tensor::Tensor::zeros(0, 0, 0)?;
    e.residual_q = crate::tensor::Tensor::zeros(0, 0, 0)?;
    if !keep_scale_log {
        e.scale_log = crate::tensor::Tensor::zeros(0, 0, 0)?;
    }
    Ok(())
}
