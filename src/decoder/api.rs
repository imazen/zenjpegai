//! One-call decoding: codestream bytes in, RGB picture out.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use enough::Stop;
use whereat::{At, at};

use super::limits::{Limits, MemoryEstimate};
use super::output::{Picture, RgbImage, finish_par, to_source_format_par};
use super::reconstruct;
use super::stats::{Gate, Probe, Tick, record};
use super::{DecodeStats, Headers, decode_components, read_headers};
use crate::container::Codestream;
use crate::error::Error;
use crate::filters::{self, FilterContext};
use crate::header::OperatingPoint;
use crate::mans::AnsTables;
use crate::model::synthesis::{SynthesisPrimary, SynthesisSecondary};
use crate::model::{self, CommonModel, ModelDir, ModelSource};
use crate::nn::fast::Engine;
use crate::tensor::Tensor;

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

    /// [`Decoder::new`] with an explicit [`Engine`] (SIMD tier, threading) instead of the
    /// best one this CPU supports.
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

    /// The [`Engine`] (SIMD tier, threading) this decoder runs the float networks on.
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
        self.decode_inner(stream, None, &enough::Unstoppable)
            .map(|(p, _)| p)
    }

    /// [`Decoder::decode_picture`] that also reports how long each pipeline stage took
    /// (see [`DecodeStats`]). The decode itself is identical either way; the timers add
    /// under a millisecond.
    pub fn decode_picture_stats(&self, stream: &[u8]) -> Result<(Picture, DecodeStats), At<Error>> {
        let probe = Probe::new();
        let (picture, _) = self.decode_inner(stream, Some(&probe), &enough::Unstoppable)?;
        Ok((picture, probe.finish()))
    }

    /// [`Decoder::decode_picture`] with cooperative cancellation (see [`Decoder::decode_with`]).
    pub fn decode_picture_with(
        &self,
        stream: &[u8],
        stop: &dyn enough::Stop,
    ) -> Result<Picture, At<Error>> {
        self.decode_inner(stream, None, stop).map(|(p, _)| p)
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
        match self.decode_inner(stream, None, stop)?.0 {
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

    fn decode_inner(
        &self,
        stream: &[u8],
        probe: Option<&Probe>,
        stop: &dyn enough::Stop,
    ) -> Result<(Picture, ()), At<Error>> {
        let t_total = Tick::now();
        // Checks from pooled tasks funnel through the gate: a counting `Stop` sees the same
        // check sequence a serial decode would produce.
        let gate = Gate::new(stop);
        let stop = &gate;
        let t = Tick::now();
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
        record(probe.map(|p| &p.headers), t);
        let t = Tick::now();
        let set = self
            .model_set(hdr.model_id as usize, op)
            .map_err(|e| at!(e))?;
        record(probe.map(|p| &p.models), t);
        let eng = &self.engine;

        // Synthesis geometry comes from the header alone (`latent_size(0)` is what
        // `y_hat[0]`'s dimensions will be): set the tiles and output planes up before the
        // chains so the luma chain can run its synthesis while chroma is still decoding.
        let (lat_h, lat_w) = hdr.latent_size(0);
        let (tiles, out_h, out_w) =
            reconstruct::synthesis_geometry(hdr, (lat_h as usize, lat_w as usize))
                .map_err(|e| at!(e))?;
        let (sv, sh) = (hdr.c_ver as usize, hdr.c_hor as usize);
        if sv == 0 || sh == 0 {
            return Err(at!(Error::InvalidData("chroma subsampling factor")));
        }
        let (ch, cw) = (out_h.div_ceil(sv), out_w.div_ceil(sh));
        let mut rec_y = Tensor::<f32>::zeros(1, out_h, out_w).map_err(|e| at!(e))?;
        let mut rec_u = Tensor::<f32>::zeros(1, ch, cw).map_err(|e| at!(e))?;
        let mut rec_v = Tensor::<f32>::zeros(1, ch, cw).map_err(|e| at!(e))?;

        // Entropy decode, latent reconstruction, LSBS and the luma synthesis run as two
        // per-component chains (luma ‖ chroma): chroma's entropy decode overlaps luma's
        // reconstruction, and luma's synthesis overlaps the chroma chain's tail. Only the
        // luma scale map survives the stage, and only when the post-filters read it.
        let t = Tick::now();
        let filters_on = headers.tools.any_post_filter();
        let (ly, luv, luma_scale_log) = decode_components(
            &self.tables,
            &cs,
            hdr,
            &headers.tools,
            eng,
            [&set.common[0], &set.common[1]],
            self.max_channels,
            filters_on,
            Some(reconstruct::LumaSynth {
                tiles: &tiles,
                model: &set.luma,
                out_h,
                out_w,
                rec_y: &mut rec_y,
            }),
            probe,
            stop,
        )
        .map_err(|e| at!(e))?;
        record(probe.map(|p| &p.chains), t);

        let t = Tick::now();
        if ly.y_hat.h != luv.y_hat.h || ly.y_hat.w != luv.y_hat.w {
            return Err(at!(Error::InvalidArgument(
                "luma / chroma latent size mismatch"
            )));
        }
        reconstruct::synthesize_chroma_into(
            eng,
            &tiles,
            &set.chroma,
            &ly.y_hat,
            &luv.y_hat,
            sv,
            sh,
            out_h,
            out_w,
            &mut rec_u,
            &mut rec_v,
            stop,
        )
        .map_err(|e| at!(e))?;
        let planes = reconstruct::Planes {
            y: rec_y,
            u: rec_u,
            v: rec_v,
        };
        drop((ly, luv));
        record(probe.map(|p| &p.synthesis), t);
        // Coded chroma format -> source chroma format, then the post-filters, then colour.
        let t = Tick::now();
        let planes = to_source_format_par(eng.parallel, hdr, planes).map_err(|e| at!(e))?;
        record(probe.map(|p| &p.chroma), t);
        let t = Tick::now();
        let planes = if filters_on {
            let ctx = FilterContext {
                eng,
                hdr,
                tools: &headers.tools,
                luma_scale_log: &luma_scale_log,
                models: &*self.models,
                op,
                icci_nets: &self.icci_nets,
                stop,
            };
            filters::apply(&ctx, planes).map_err(|e| at!(e))?
        } else {
            planes
        };
        drop(luma_scale_log);
        record(probe.map(|p| &p.filters), t);
        let t = Tick::now();
        stop.check().map_err(|r| at!(Error::from(r)))?;
        let picture = finish_par(eng.parallel, hdr, &planes).map_err(|e| at!(e))?;
        record(probe.map(|p| &p.output), t);
        record(probe.map(|p| &p.total), t_total);
        Ok((picture, ()))
    }
}
