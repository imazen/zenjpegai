//! Encoder: an RGB picture in, a JPEG AI codestream out.
//!
//! Ports the compress / encode direction of
//! `ref/src/codec/coding_tools/core_models/CCS_SGMM/` (`ccs_sgmm_tool.py::compress`,
//! `sep_chan_tool.py::compress`, `common_modules.py::{compress, _compress_z,
//! encoder_get_scales, _compress_ar_scale, encoder_skip_and_cubeflag_for_tiles, encode,
//! encode_z, encode_y, _ac_encode_y, _ac_encode_z}`).
//!
//! What is ported: a fixed model and operating point, one analysis tile, one region, one ANS
//! thread per substream, tools off. What is not: analysis tiling (pictures above ~1 MP),
//! regions, rate matching (`--bpp`), RVS / GRFS / LSBS / quality maps / post-filters, chroma
//! subsampling, bit depths other than 8. See `PORTING.md`.

// The per-channel loops index several parallel arrays with one counter, like the reference's
// tensor expressions; an iterator chain over one of them would hide that.
#![allow(clippy::needless_range_loop)]

mod colour;
mod rate;
mod tiles;

pub use colour::{AnalysisInput, preprocess_rgb};
pub use rate::{BDL_SEARCH_RANGE, RateMatch, search_beta};
pub use tiles::{AnalysisTile, analysis_tiles};

use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use whereat::{At, at};

use crate::container::{CodestreamWriter, Marker, join_threads};
use crate::decoder::entropy::{
    ComponentScales, channel_step, component_scales, distribution_index,
};
use crate::decoder::output::RgbImage;
use crate::error::{Error, Result};
use crate::header::{
    ColourTransform, ComponentHeader, LATENT_CHANNELS, OperatingPoint, PictureHeader,
    RenderingInfo, ToolHeader,
};
use crate::mans::AnsTables;
use crate::model::analysis::{AnalysisPrimary, AnalysisSecondary};
use crate::model::common::Z_OFFSET;
use crate::model::hyper_encoder::HyperEncoder;
use crate::model::mcm::{self, upshuffle_psi};
use crate::model::{self, CommonModel, ModelDir, ModelSource};
use crate::nn::fast::{BTensor, Engine};
use crate::tensor::Tensor;
use crate::tools::regions::Area;
use crate::tools::skip::skip_mask;

/// `skip_cube_thr`: a cube may be skipped while every latent sample in it is reconstructed
/// within this. **3, not the tool's Python default of 1**: `cfg/oper_point/common.json` sets it
/// on `model.CCS_SGMM.tools_common.model_common.common_modules.skip_mode`, which does reach
/// `SkipModeParams` (unlike `sigma_quant_level`, see `PORTING.md`). Checked against the
/// reference encoder's own cube flags: with 1 two cubes of `enc_img30_bop_m1_b0` would be
/// flagged (their reconstruction error is 1.01 and 1.13) and the reference flags none.
const SKIP_CUBE_THR: f32 = 3.0;
/// `level_idc` the reference encoder defaults to (`coding_engine/params.py`): the largest
/// picture-size level and the model set that permits every model.
const LEVEL_IDC: u8 = 52;

/// What a single encode is asked to do. The rate is set by `beta_displacement_log`, in
/// sigma-index units (Q7): lower is a lower rate. The reference clips it to `[-1069, 702]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeParams {
    /// Trained model, 0..=3 (`cfg/pipeline.json`'s betas 0.002, 0.012, 0.075, 0.5).
    pub model_id: u8,
    /// Quantiser displacement per component (luma, chroma).
    pub beta_displacement_log: [i32; 2],
    /// Operating point to code for: it selects the analysis transform (BOP for simple and base,
    /// HOP for high), the decoder profile, and the synthesis transforms the stream lists.
    pub op: OperatingPoint,
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            model_id: 1,
            beta_displacement_log: [0, 0],
            op: OperatingPoint::Bop,
        }
    }
}

/// `BDL_clipping_range` (`CCS_SGMM/params.py`).
pub const BDL_RANGE: (i32, i32) = (-1069, 702);

struct ModelSet {
    common: [CommonModel; 2],
    analysis_y: AnalysisPrimary,
    analysis_uv: AnalysisSecondary,
    hyper: [HyperEncoder; 2],
}

/// A JPEG AI encoder bound to a directory of upstream checkpoints.
///
/// Checkpoints are parsed and packed on first use and kept, so encoding many pictures pays the
/// model load once per (model, operating point). `Encoder` is `Sync`; share it between threads.
pub struct Encoder {
    models: Box<dyn ModelSource + Send + Sync>,
    engine: Engine,
    tables: AnsTables,
    cache: Mutex<HashMap<(usize, OperatingPoint), Arc<ModelSet>>>,
}

/// One model's analysis output for a whole picture: the two latents and the two hyper-latents.
/// They do not depend on the quantiser displacement, so rate matching computes them once.
type ModelLatents = ([Tensor<f32>; 2], [Tensor<i8>; 2]);

/// Everything one component contributes to the codestream.
struct Component {
    z_hat: Tensor<i8>,
    scales: ComponentScales,
    residual_q: Tensor<i16>,
    mask: Tensor<bool>,
    cube_flag: Vec<bool>,
    cube_flags: Option<Vec<bool>>,
}

/// What one component committed to the codestream: exactly the tensors
/// `scripts/ref_vectors/dump_encode.py` records, for parity tests and diagnostics.
#[derive(Clone, Debug)]
pub struct ComponentTrace {
    pub z_hat: Tensor<i8>,
    pub skip_scale_log: Tensor<i32>,
    pub scale_log: Tensor<i32>,
    pub residual_q: Tensor<i16>,
    /// Dequantised residual: what the decoder recovers.
    pub residual: Tensor<f32>,
    /// `[phase][cube_y][cube_x]`, `true` = the cube may be skipped.
    pub cube_flag: Vec<bool>,
    /// `true` where a residual symbol is written.
    pub mask: Tensor<bool>,
}

impl Encoder {
    /// `models_dir` is laid out like the reference repository's `models/` directory.
    pub fn new(models_dir: impl Into<std::path::PathBuf>) -> Self {
        Self::with_engine(models_dir, Engine::new())
    }

    pub fn with_engine(models_dir: impl Into<std::path::PathBuf>, engine: Engine) -> Self {
        Self::with_source(Box::new(ModelDir::new(models_dir)), engine)
    }

    pub fn with_source(models: Box<dyn ModelSource + Send + Sync>, engine: Engine) -> Self {
        Self {
            models,
            engine,
            tables: AnsTables::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    fn model_set(&self, id: usize, op: OperatingPoint) -> Result<Arc<ModelSet>> {
        if let Some(set) = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(&(id, op)).cloned())
        {
            return Ok(set);
        }
        let eng = &self.engine;
        // Simple-profile streams are encoded with the BOP analysis transform
        // (`cfg/oper_point/bopEnc_sopDec.json`); only the synthesis side is SOP.
        let analysis_op = match op {
            OperatingPoint::Sop => OperatingPoint::Bop,
            other => other,
        };
        let set = Arc::new(ModelSet {
            common: [
                model::load_common(&*self.models, id, 0, eng)?,
                model::load_common(&*self.models, id, 1, eng)?,
            ],
            analysis_y: model::load_analysis_primary(&*self.models, id, analysis_op, eng)?,
            analysis_uv: model::load_analysis_secondary(&*self.models, id, analysis_op, eng)?,
            hyper: [
                model::load_hyper_encoder(&*self.models, id, 0, eng)?,
                model::load_hyper_encoder(&*self.models, id, 1, eng)?,
            ],
        });
        if let Ok(mut c) = self.cache.lock() {
            c.insert((id, op), set.clone());
        }
        Ok(set)
    }

    /// Encode one 8-bit RGB picture.
    pub fn encode(
        &self,
        rgb: &RgbImage,
        params: EncodeParams,
    ) -> core::result::Result<Vec<u8>, At<Error>> {
        self.encode_with(rgb, params, &enough::Unstoppable)
    }

    /// [`Encoder::encode`] with cooperative cancellation. `stop` is checked between network
    /// layers and stages, never per sample.
    pub fn encode_with(
        &self,
        rgb: &RgbImage,
        params: EncodeParams,
        stop: &dyn enough::Stop,
    ) -> core::result::Result<Vec<u8>, At<Error>> {
        self.encode_inner(rgb, params, stop)
            .map(|(stream, _)| stream)
            .map_err(|e| at!(e))
    }

    /// [`Encoder::encode`] that also returns the intermediates of both components (luma,
    /// chroma). Used by the parity tests against the reference encoder's own dumps.
    pub fn encode_traced(
        &self,
        rgb: &RgbImage,
        params: EncodeParams,
    ) -> core::result::Result<(Vec<u8>, [ComponentTrace; 2]), At<Error>> {
        let (stream, components) = self
            .encode_inner(rgb, params, &enough::Unstoppable)
            .map_err(|e| at!(e))?;
        Self::trace(stream, components).map_err(|e| at!(e))
    }

    fn trace(
        stream: Vec<u8>,
        components: Vec<Component>,
    ) -> Result<(Vec<u8>, [ComponentTrace; 2])> {
        let traces = components
            .into_iter()
            .map(|c| {
                let residual =
                    crate::decoder::entropy::dequantize_residual(&c.scales, None, &c.residual_q)?;
                Ok(ComponentTrace {
                    z_hat: c.z_hat,
                    skip_scale_log: c.scales.skip_scale_log,
                    scale_log: c.scales.scale_log,
                    residual_q: c.residual_q,
                    residual,
                    cube_flag: c.cube_flag,
                    mask: c.mask,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let [y, uv] = <[ComponentTrace; 2]>::try_from(traces)
            .map_err(|_| Error::InvalidData("internal: component count"))?;
        Ok((stream, [y, uv]))
    }

    /// Everything after the analysis transforms, from the two latents on
    /// (`common_modules.py::compress` + `encode`). Exposed so that the parity tests can feed the
    /// reference encoder's own `y` and compare the integer decisions exactly: our analysis
    /// transform agrees with PyTorch's only to about 1e-4 (`PORTING.md`), which is enough to
    /// move a residual symbol that sits on a rounding boundary.
    ///
    /// `z_hats` supplies the hyper-latents. Pass `None` to derive them here, which is what the
    /// encoder does for a picture small enough to be one analysis tile; for a tiled picture each
    /// tile's `z` comes from that tile's own (un-merged) latent, so the merged `y` alone cannot
    /// reproduce it and the caller must hand them over.
    pub fn encode_latents(
        &self,
        latents: [&Tensor<f32>; 2],
        z_hats: Option<[&Tensor<i8>; 2]>,
        width: usize,
        height: usize,
        params: EncodeParams,
    ) -> core::result::Result<(Vec<u8>, [ComponentTrace; 2]), At<Error>> {
        let (stream, components) = self
            .encode_from_latents(
                latents,
                z_hats.map(|z| [z[0].clone(), z[1].clone()]),
                width,
                height,
                params,
                &enough::Unstoppable,
            )
            .map_err(|e| at!(e))?;
        Self::trace(stream, components).map_err(|e| at!(e))
    }

    fn encode_inner(
        &self,
        rgb: &RgbImage,
        params: EncodeParams,
        stop: &dyn enough::Stop,
    ) -> Result<(Vec<u8>, Vec<Component>)> {
        stop.check()?;
        let set = self.model_set(self.check_params(rgb.width, rgb.height, params)?, params.op)?;
        // 1. Colour pre-processing, then the analysis transform and the hyper-encoder, tile by
        //    tile (`sep_chan_tool.py::analysis_and_hyper_encoder`).
        let input = preprocess_rgb(rgb)?;
        let planes = [&input.luma, &input.chroma];
        let (mut ys, mut zs) = (Vec::with_capacity(2), Vec::with_capacity(2));
        for (ccs, plane) in planes.into_iter().enumerate() {
            let (y, z) = self.analyse_component(&set, ccs, plane, stop)?;
            ys.push(y);
            zs.push(z);
        }
        let z = [zs[0].clone(), zs[1].clone()];
        self.encode_from_latents(
            [&ys[0], &ys[1]],
            Some(z),
            rgb.width,
            rgb.height,
            params,
            stop,
        )
    }

    /// Encode to a target rate in bits per (luma) pixel, choosing the trained model and the
    /// quantiser displacement (`bitrate_matcher`; see [`rate`] for what is and is not ported).
    ///
    /// The analysis transform and the hyper-encoder do not depend on the displacement, so each
    /// model's latents are computed once and every trial re-codes from them; the reference
    /// re-runs the whole analysis per trial.
    pub fn encode_to_bpp(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        op: OperatingPoint,
    ) -> core::result::Result<(Vec<u8>, RateMatch), At<Error>> {
        self.encode_to_bpp_with(rgb, target_bpp, op, &enough::Unstoppable)
    }

    /// [`Encoder::encode_to_bpp`] with cooperative cancellation.
    pub fn encode_to_bpp_with(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        op: OperatingPoint,
        stop: &dyn enough::Stop,
    ) -> core::result::Result<(Vec<u8>, RateMatch), At<Error>> {
        self.rate_match(rgb, target_bpp, op, stop)
            .map_err(|e| at!(e))
    }

    fn rate_match(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        op: OperatingPoint,
        stop: &dyn enough::Stop,
    ) -> Result<(Vec<u8>, RateMatch)> {
        if !(target_bpp.is_finite() && target_bpp > 0.0) {
            return Err(Error::InvalidArgument("target bpp must be positive"));
        }
        let pixels = (rgb.width * rgb.height) as f64;
        let input = preprocess_rgb(rgb)?;
        let mut trials = 0usize;
        // One analysis per model; every displacement re-codes from its latents.
        let mut latents: Vec<Option<ModelLatents>> =
            (0..model::MODEL_BETAS.len()).map(|_| None).collect();
        let code = |ccs_model: usize,
                    beta: i32,
                    latents: &mut [Option<ModelLatents>],
                    trials: &mut usize|
         -> Result<Vec<u8>> {
            if latents[ccs_model].is_none() {
                let set = self.model_set(ccs_model, op)?;
                let mut ys = Vec::with_capacity(2);
                let mut zs = Vec::with_capacity(2);
                for (ccs, plane) in [&input.luma, &input.chroma].into_iter().enumerate() {
                    let (y, z) = self.analyse_component(&set, ccs, plane, stop)?;
                    ys.push(y);
                    zs.push(z);
                }
                latents[ccs_model] = Some((
                    [ys[0].clone(), ys[1].clone()],
                    [zs[0].clone(), zs[1].clone()],
                ));
            }
            let (y, z) = latents[ccs_model].as_ref().expect("just computed");
            *trials += 1;
            let params = EncodeParams {
                model_id: ccs_model as u8,
                beta_displacement_log: [beta; 2],
                op,
            };
            let (stream, _) = self.encode_from_latents(
                [&y[0], &y[1]],
                Some([z[0].clone(), z[1].clone()]),
                rgb.width,
                rgb.height,
                params,
                stop,
            )?;
            Ok(stream)
        };

        // `match_luma`: the model whose rate at displacement 0 is relatively closest.
        let mut best_model = 0usize;
        let mut best_diff = f64::INFINITY;
        let mut base: Vec<Option<f64>> = alloc::vec![None; model::MODEL_BETAS.len()];
        for id in 0..model::MODEL_BETAS.len() {
            stop.check()?;
            let bpp = code(id, 0, &mut latents, &mut trials)?.len() as f64 * 8.0 / pixels;
            base[id] = Some(bpp);
            let diff = (bpp - target_bpp).abs() / bpp;
            if diff < best_diff {
                best_diff = diff;
                best_model = id;
            }
            // Only the chosen model's latents are needed from here on.
        }
        for (id, slot) in latents.iter_mut().enumerate() {
            if id != best_model {
                *slot = None;
            }
        }

        let mut cache: std::collections::HashMap<i32, f64> = std::collections::HashMap::new();
        cache.insert(0, base[best_model].expect("measured above"));
        let beta = {
            let mut cached = |b: i32| -> Result<f64> {
                if let Some(&v) = cache.get(&b) {
                    return Ok(v);
                }
                stop.check()?;
                let bpp =
                    code(best_model, b, &mut latents, &mut trials)?.len() as f64 * 8.0 / pixels;
                cache.insert(b, bpp);
                Ok(bpp)
            };
            rate::search_beta(target_bpp, rate::BDL_SEARCH_RANGE[best_model], &mut cached)?
        };
        let stream = code(best_model, beta, &mut latents, &mut trials)?;
        let bpp = stream.len() as f64 * 8.0 / pixels;
        Ok((
            stream,
            RateMatch {
                model_id: best_model as u8,
                beta_displacement_log: beta.clamp(BDL_RANGE.0, BDL_RANGE.1),
                bpp,
                trials,
            },
        ))
    }

    /// `compress_colocated_tiles` over every analysis tile of one component: the analysis
    /// transform and the hyper-encoder run per tile and only each tile's core is kept.
    fn analyse_component(
        &self,
        set: &ModelSet,
        ccs: usize,
        plane: &Tensor<f32>,
        stop: &dyn enough::Stop,
    ) -> Result<(Tensor<f32>, Tensor<i8>)> {
        let (ph, pw) = (plane.h, plane.w);
        let d = tiles::LATENT_DOWNSCALE[ccs];
        let (lh, lw) = (ph.div_ceil(d), pw.div_ceil(d));
        let (hz, wz) = (ph.div_ceil(4 * d), pw.div_ceil(4 * d));
        let chs = LATENT_CHANNELS[ccs];
        let eng = &self.engine;
        let grid = tiles::analysis_tiles(ccs, ph, pw, lh, lw, hz, wz)?;
        let mut y = Tensor::<f32>::zeros(chs, lh, lw)?;
        let mut z_hat = Tensor::<i8>::zeros(chs, hz, wz)?;
        for t in &grid {
            stop.check()?;
            let tile = if grid.len() == 1 {
                plane.clone()
            } else {
                plane.window(t.image.x, t.image.y, t.image.width, t.image.height)?
            };
            let yt = match ccs {
                0 => set.analysis_y.forward(eng, &tile, stop)?,
                _ => set.analysis_uv.forward(eng, &tile, stop)?,
            };
            stop.check()?;
            let zt = set.hyper[ccs]
                .forward(eng, &yt, t.image.height, t.image.width, d, stop)?
                .to_planar()?;
            let yt = yt.to_planar()?;
            if (yt.h, yt.w) != (t.latent.height, t.latent.width)
                || (zt.h, zt.w) != (t.hyper.height, t.hyper.width)
            {
                return Err(Error::InvalidData("analysis tile: unexpected latent size"));
            }
            assign(&mut y, t.latent_core, &yt, t.latent_core_offset);
            let max = (Z_OFFSET - 1) as f32;
            let quantised = Tensor::from_vec(
                zt.c,
                zt.h,
                zt.w,
                zt.data
                    .iter()
                    .map(|&v| v.clamp(-(Z_OFFSET as f32), max).round_ties_even() as i8)
                    .collect(),
            )?;
            assign(&mut z_hat, t.hyper_core, &quantised, t.hyper_core_offset);
        }
        Ok((y, z_hat))
    }

    fn check_params(&self, w: usize, h: usize, params: EncodeParams) -> Result<usize> {
        if params.model_id as usize >= model::MODEL_BETAS.len() {
            return Err(Error::InvalidArgument("model_id out of range"));
        }
        if !(64..=65599).contains(&w) || !(64..=65599).contains(&h) {
            return Err(Error::Unsupported(
                "picture dimensions outside the format's 64..65599",
            ));
        }
        if w * h > 120_000_000 {
            return Err(Error::Unsupported("picture above 120 MP"));
        }
        Ok(params.model_id as usize)
    }

    fn encode_from_latents(
        &self,
        latents: [&Tensor<f32>; 2],
        z_hats: Option<[Tensor<i8>; 2]>,
        w: usize,
        h: usize,
        params: EncodeParams,
        stop: &dyn enough::Stop,
    ) -> Result<(Vec<u8>, Vec<Component>)> {
        let set = self.model_set(self.check_params(w, h, params)?, params.op)?;
        let eng = &self.engine;
        let beta = params
            .beta_displacement_log
            .map(|b| b.clamp(BDL_RANGE.0, BDL_RANGE.1));
        // Component plane sizes: the luma plane padded to an even size, and half of that.
        let (pw, ph) = (w + w % 2, h + h % 2);
        let sizes = [(ph, pw), (ph / 2, pw / 2)];

        // 2. Hyper-encoder and `z` quantisation, per component (already done when the caller
        //    supplied the hyper-latents, which a tiled analysis must).
        let mut z_hats = match z_hats {
            Some(z) => z.into_iter().collect(),
            None => Vec::with_capacity(2),
        };
        let need_z = z_hats.len() < 2;
        for (ccs, latent) in latents.iter().enumerate().filter(|_| need_z) {
            stop.check()?;
            let (ph, pw) = sizes[ccs];
            let divider = if ccs == 0 { 16 } else { 8 };
            let latent = BTensor::from_planar(latent, eng.tier.block())?;
            let z = set.hyper[ccs]
                .forward(eng, &latent, ph, pw, divider, stop)?
                .to_planar()?;
            let max = (Z_OFFSET - 1) as f32;
            let data = z
                .data
                .iter()
                .map(|&v| v.clamp(-(Z_OFFSET as f32), max).round_ties_even() as i8)
                .collect();
            z_hats.push(Tensor::from_vec(z.c, z.h, z.w, data)?);
        }

        // 3. A provisional header: everything the scale derivation reads is known now; the
        //    cube flags are filled in once the residual has been quantised.
        let mut hdr = picture_header(w as u32, h as u32, params.model_id, beta, params.op);
        hdr.check_conformance()?;

        let mut components = Vec::with_capacity(2);
        for (ccs, latent) in latents.into_iter().enumerate() {
            stop.check()?;
            let model = &set.common[ccs];
            let z_hat = &z_hats[ccs];
            let (lh, lw) = hdr.latent_size(ccs);
            let (lh, lw) = (lh as usize, lw as usize);
            if latent.h != lh || latent.w != lw || latent.c != model.chs {
                return Err(Error::InvalidData("analysis transform: unexpected latent"));
            }

            // 4. Scales, exactly as the decoder derives them, and the threshold skip mask.
            let scales = component_scales(&hdr, ccs, model, z_hat, None)?;
            let mask = skip_mask(&scales.skip_scale_log, None)?;

            // 5. psi, then the residual quantisation with its cube-flag decision.
            let psi = model.hyper_decoder.forward_with(
                eng,
                z_hat,
                lh.div_ceil(2),
                lw.div_ceil(2),
                stop,
            )?;
            let scaler = &scales.gain.scaler;
            let out = match &model.context {
                Some(ctx) => ctx.compress(eng, latent, &psi, scaler, &mask, SKIP_CUBE_THR, stop)?,
                None => compress_context_free(latent, &psi, scaler, &mask, SKIP_CUBE_THR)?,
            };

            // 6. `encoder_skip_and_cubeflag_for_tiles`: the final mask is the threshold mask
            //    widened by the cubes that must not be skipped; everything else is dropped.
            let cube_flag = out.cube_flag;
            let all_skippable = cube_flag.iter().all(|&f| f);
            let cube_flags = (!all_skippable).then(|| cube_flag.clone());
            let mask = skip_mask(&scales.skip_scale_log, cube_flags.as_deref())?;
            let mut residual_q = out.residual_q;
            for (q, &m) in residual_q.data.iter_mut().zip(&mask.data) {
                if !m {
                    *q = 0;
                }
            }
            components.push(Component {
                z_hat: z_hats[ccs].clone(),
                scales,
                residual_q,
                mask,
                cube_flag,
                cube_flags,
            });
        }
        for (ccs, c) in components.iter().enumerate() {
            hdr.components[ccs].cube_flags = c.cube_flags.clone();
        }

        // 7. Entropy coding. The encoder runs the decoder's call order backwards, so the
        //    chroma `z` is written before the luma `z` into the one SOZ payload.
        stop.check()?;
        let mut z_enc = self.tables.encoder(hdr.num_threads_z as usize)?;
        for ccs in [1usize, 0] {
            let z = &components[ccs].z_hat;
            let symbols: Vec<u8> = z
                .data
                .iter()
                .map(|&v| (v as i32 + Z_OFFSET) as u8)
                .collect();
            z_enc.encode_z(&set.common[ccs].z_cdfs, z.h * z.w, &symbols)?;
        }
        let soz_threads = z_enc.finish();
        let soz = join_threads(&soz_threads.iter().map(|t| t.as_slice()).collect::<Vec<_>>());

        let mut residual_payloads = Vec::with_capacity(2);
        for (ccs, c) in components.iter().enumerate() {
            stop.check()?;
            residual_payloads.push(self.encode_residual(&hdr, ccs, c)?);
        }

        // 8. Container. The reference writes the residual substreams before SOZ.
        let mut out = CodestreamWriter::new();
        out.substream(Marker::Pih, &hdr.write()?)?;
        out.substream(Marker::Ton, &ToolHeader::default().write(&hdr)?)?;
        out.substream(Marker::Rdi, &RenderingInfo::default().write()?)?;
        out.substream(Marker::Sorp, &residual_payloads[0])?;
        out.substream(Marker::Sors, &residual_payloads[1])?;
        out.substream(Marker::Soz, &soz)?;
        Ok((out.finish(), components))
    }

    /// `encode_y` / `_ac_encode_y` for one component: the decoder's channel-chunk loop, walked
    /// backwards (ANS is last-in-first-out).
    fn encode_residual(&self, hdr: &PictureHeader, ccs: usize, c: &Component) -> Result<Vec<u8>> {
        let comp = &hdr.components[ccs];
        let (lh, lw) = hdr.latent_size(ccs);
        let (lh, lw) = (lh as usize, lw as usize);
        let num_chs = (comp.num_chs as usize).min(c.residual_q.c);
        let threads = comp.num_threads_r as usize;
        let mut enc = self.tables.encoder(threads)?;
        let step = channel_step(lh, lw, num_chs, threads);
        let chunks: Vec<usize> = (0..num_chs).step_by(step).collect();
        let (mut sigma, mut coded, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for &c0 in chunks.iter().rev() {
            let c1 = (c0 + step).min(num_chs);
            let n = (c1 - c0) * lh * lw;
            sigma.clear();
            coded.clear();
            values.clear();
            sigma.reserve(n);
            coded.reserve(n);
            values.reserve(n);
            for ch in c0..c1 {
                sigma.extend(
                    c.scales.scale_log.plane(ch)[..lh * lw]
                        .iter()
                        .map(|&s| distribution_index(s)),
                );
                coded.extend_from_slice(&c.mask.plane(ch)[..lh * lw]);
                values.extend_from_slice(&c.residual_q.plane(ch)[..lh * lw]);
            }
            enc.encode_residual(&sigma, &coded, &values)?;
        }
        let parts = enc.finish();
        Ok(join_threads(
            &parts.iter().map(|t| t.as_slice()).collect::<Vec<_>>(),
        ))
    }
}

/// Read a PNG into the 8-bit RGB picture [`Encoder::encode`] takes (`cli` feature, zenpng).
///
/// 16-bit and paletted PNGs are reduced to 8-bit RGB by zenpng; alpha is dropped, as the
/// reference's own PNG reader does.
#[cfg(feature = "cli")]
pub fn read_png_rgb8(bytes: &[u8]) -> Result<RgbImage> {
    let info = zenpng::probe(bytes).map_err(|e| Error::InvalidArgument(png_err(e)))?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 {
        return Err(Error::InvalidArgument("empty PNG"));
    }
    let mut buf = alloc::vec![rgb::Rgb { r: 0u8, g: 0, b: 0 }; w * h];
    zenpng::PngDecoderConfig::new()
        .decode_into_rgb8(bytes, imgref::ImgRefMut::new(&mut buf, w, h))
        .map_err(|e| Error::InvalidArgument(png_err(e)))?;
    let mut data = Vec::with_capacity(w * h * 3);
    for p in &buf {
        data.extend_from_slice(&[p.r as u16, p.g as u16, p.b as u16]);
    }
    Ok(RgbImage {
        width: w,
        height: h,
        bit_depth: 8,
        data,
    })
}

#[cfg(feature = "cli")]
fn png_err<E>(_e: E) -> &'static str {
    "the input is not a readable PNG"
}

/// `tiling.get_data` + `tiling.assign_data`: copy `src[:, oy.., ox..]` into `dst` at `core`,
/// clamped the way tensor slicing clamps (the same helper `decoder::reconstruct` uses for
/// regions, over any element type).
fn assign<T: Copy + Default>(
    dst: &mut Tensor<T>,
    core: Area,
    src: &Tensor<T>,
    (ox, oy): (usize, usize),
) {
    let h = core
        .height
        .min(src.h.saturating_sub(oy))
        .min(dst.h.saturating_sub(core.y));
    let w = core
        .width
        .min(src.w.saturating_sub(ox))
        .min(dst.w.saturating_sub(core.x));
    let (dw, sw) = (dst.w, src.w);
    for c in 0..dst.c.min(src.c) {
        for y in 0..h {
            let s: &[T] = &src.plane(c)[(oy + y) * sw + ox..][..w];
            let d = (core.y + y) * dw + core.x;
            dst.plane_mut(c)[d..d + w].copy_from_slice(s);
        }
    }
}

/// `_compress_ar_scale`'s branch for a component without a context model (chroma): the mean is
/// `psi` up-shuffled, and the cube flags come from the whole reconstruction
/// (`skip_mode.gen_skip_cubeflag`) rather than stage by stage.
fn compress_context_free(
    y: &Tensor<f32>,
    psi: &BTensor,
    scaler: &[f32],
    mask: &Tensor<bool>,
    cube_thr: f32,
) -> Result<mcm::Compressed> {
    let (c, h, w) = (y.c, y.h, y.w);
    let mean = upshuffle_psi(psi, h, w)?;
    let mut residual_q = Tensor::<i16>::zeros(c, h, w)?;
    let mut residual = Tensor::<f32>::zeros(c, h, w)?;
    // Pass 1: the threshold mask alone; the error decides the cube flags.
    let (hh, hw) = (h.div_ceil(2), w.div_ceil(2));
    let (cube_h, cube_w) = (hh.div_ceil(mcm::CUBE_SIZE), hw.div_ceil(mcm::CUBE_SIZE));
    let mut cube_flag = alloc::vec![true; 4 * cube_h * cube_w];
    for ch in 0..c {
        for i in 0..h * w {
            let d = y.plane(ch)[i] - mean.plane(ch)[i];
            let (_, dq) = mcm::quantise(d, scaler[ch], mask.plane(ch)[i]);
            if (dq - d).abs() > cube_thr {
                let (yy, xx) = (i / w, i % w);
                let phase = (yy % 2) * 2 + xx % 2;
                cube_flag[(phase * cube_h + yy / 2 / mcm::CUBE_SIZE) * cube_w
                    + xx / 2 / mcm::CUBE_SIZE] = false;
            }
        }
    }
    // Pass 2: with the cubes that must not be skipped.
    let mask2 = skip_mask_or_cubes(mask, &cube_flag, cube_h, cube_w);
    for ch in 0..c {
        for i in 0..h * w {
            let d = y.plane(ch)[i] - mean.plane(ch)[i];
            let (q, dq) = mcm::quantise(d, scaler[ch], mask2.plane(ch)[i]);
            residual_q.plane_mut(ch)[i] = q;
            residual.plane_mut(ch)[i] = dq;
        }
    }
    Ok(mcm::Compressed {
        residual_q,
        residual,
        cube_flag,
    })
}

/// `torch.logical_or(mask2, cube_flags_full)` at latent resolution.
fn skip_mask_or_cubes(
    mask: &Tensor<bool>,
    cube_flag: &[bool],
    cube_h: usize,
    cube_w: usize,
) -> Tensor<bool> {
    let mut out = mask.clone();
    let (c, h, w) = (mask.c, mask.h, mask.w);
    for y in 0..h {
        for x in 0..w {
            let phase = (y % 2) * 2 + x % 2;
            let flag = cube_flag
                [(phase * cube_h + y / 2 / mcm::CUBE_SIZE) * cube_w + x / 2 / mcm::CUBE_SIZE];
            if !flag {
                for ch in 0..c {
                    out.data[(ch * h + y) * w + x] = true;
                }
            }
        }
    }
    out
}

/// The picture header of a fixed-model, single-region, tools-off encode.
fn picture_header(
    width: u32,
    height: u32,
    model_id: u8,
    beta_displacement_log: [i32; 2],
    op: OperatingPoint,
) -> PictureHeader {
    use OperatingPoint::{Bop, Hop, Sop};
    // `cfg/profiles/{simple,base,high}.json`.
    let (decoder_profile_id, synthesis_transforms) = match op {
        Sop => (0, alloc::vec![Sop]),
        Bop => (1, alloc::vec![Bop, Sop]),
        Hop => (2, alloc::vec![Hop, Bop, Sop]),
    };
    PictureHeader {
        stream_profile_idc: 0,
        decoder_profile_id,
        synthesis_transforms,
        level_idc: LEVEL_IDC,
        width,
        height,
        diff_display_width: 0,
        diff_display_height: 0,
        bit_depth: 8,
        s_ver: 1,
        s_hor: 1,
        c_ver: 1,
        c_hor: 1,
        colour_transform: ColourTransform::Bt709,
        model_id,
        num_threads_z: 1,
        beta_displacement_log,
        regions: None,
        // `tile_manager_synthesis` is set up from the coded luma size for both components.
        components: [0, 1].map(|ccs| ComponentHeader {
            num_threads_r: 1,
            num_chs: LATENT_CHANNELS[ccs] as u16,
            cube_flags: None,
            rvs_enabled: false,
            grfs_channel_flags: None,
            synthesis_tiling: tiles::synthesis_tiling(height as usize, width as usize),
        }),
        quality_map: None,
    }
}
