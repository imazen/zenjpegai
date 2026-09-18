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

use crate::container::{CodestreamWriter, Marker, join_dependent_regions, join_threads};
use crate::decoder::entropy::{
    ComponentScales, GrfsFlags, channel_step, component_scales_with, distribution_index,
};
use crate::decoder::output::RgbImage;
use crate::decoder::reconstruct::{HD_MCM_TILE_OVERLAP, hyper_crop};
use crate::error::{Error, Result};
use crate::header::{
    ColourTransform, ComponentHeader, LATENT_CHANNELS, OperatingPoint, PictureHeader, Regions,
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
use crate::tools::regions::{Area, Plane, region_grid};
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
    /// Residual variance scaling (`cfg/tools/ResVarScale.json`'s `rvs_enabled`): the quantiser
    /// step follows the block-wise sigma map.
    pub rvs: bool,
    /// Channel gain flags (`cwg_enabled`); the encoder picks the channels (`analyzeCWG`).
    pub grfs: bool,
    /// Latent scaling before synthesis (`cfg/tools/LSBS.json`). A decoder-side tool: it is
    /// signalled in the tool header and changes no coded symbol.
    pub lsbs: bool,
    /// ANS threads of the `z` substream and of each residual substream (1, 2, 4, 8 or 16).
    pub num_threads_z: u8,
    pub num_threads_r: u8,
    /// Region partitioning (`cfg/tools/{Dependent,Independent}Regions.json`). The grid itself is
    /// derived from the picture size (`calc_numHor_numVer_regions`); a picture of at most
    /// `NumSamplesInRegion` samples gets no regions whatever this says.
    pub regions: Option<RegionMode>,
}

/// How a region's residual reaches the codestream
/// (`region_residual_in_its_own_substream_flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionMode {
    /// One SORP / SORS substream for all regions, with their sizes at its head. Regions overlap
    /// and their contexts see each other.
    Dependent,
    /// One substream per region, each starting with its index: every region decodes on its own.
    Independent,
}

/// `NumSamplesInRegion` (`cfg/tools/*Regions.json`): a picture at most this large is not
/// partitioned.
pub const NUM_SAMPLES_IN_REGION: usize = 1_048_576;
/// `hyper_decoder_overlap_in_latent_samples` / `mcm_overlap_in_latent_samples`, as signalled.
pub const HYPER_DECODER_OVERLAP: u8 = 2;
pub const MCM_OVERLAP: u8 = 8;

/// `calc_numHor_numVer_regions`: `(num_ver, num_hor)`, or `None` when the picture is small
/// enough that the reference clears `region_partitioning_flag` again.
pub fn region_counts(height: usize, width: usize) -> Option<(u8, u8)> {
    if height * width <= NUM_SAMPLES_IN_REGION {
        return None;
    }
    let step = libm::sqrt(NUM_SAMPLES_IN_REGION as f64) as usize;
    let num_hor = (width / step).clamp(1, width.div_ceil(512));
    let num_ver = (height / step).clamp(1, height.div_ceil(256));
    Some((num_ver as u8, num_hor as u8))
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            model_id: 1,
            beta_displacement_log: [0, 0],
            op: OperatingPoint::Bop,
            rvs: false,
            grfs: false,
            lsbs: false,
            num_threads_z: 1,
            num_threads_r: 1,
            regions: None,
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
    psi: Tensor<f32>,
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
    /// The merged hyper-decoder output the context model saw.
    pub psi: Tensor<f32>,
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
                    psi: c.psi,
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
        let independent = (params.regions == Some(RegionMode::Independent))
            .then(|| region_counts(rgb.height, rgb.width))
            .flatten()
            .map(|(v, h)| (v as usize, h as usize));
        let input = preprocess_rgb(rgb)?;
        let planes = [&input.luma, &input.chroma];
        let (mut ys, mut zs) = (Vec::with_capacity(2), Vec::with_capacity(2));
        for (ccs, plane) in planes.into_iter().enumerate() {
            let (y, z) = self.analyse_component(&set, ccs, plane, independent, stop)?;
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
    /// `params.model_id` and `params.beta_displacement_log` are ignored (that is what the search
    /// decides); every other field applies.
    pub fn encode_to_bpp(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        params: EncodeParams,
    ) -> core::result::Result<(Vec<u8>, RateMatch), At<Error>> {
        self.encode_to_bpp_with(rgb, target_bpp, params, &enough::Unstoppable)
    }

    /// [`Encoder::encode_to_bpp`] with cooperative cancellation.
    pub fn encode_to_bpp_with(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        params: EncodeParams,
        stop: &dyn enough::Stop,
    ) -> core::result::Result<(Vec<u8>, RateMatch), At<Error>> {
        self.rate_match(rgb, target_bpp, params, stop)
            .map_err(|e| at!(e))
    }

    fn rate_match(
        &self,
        rgb: &RgbImage,
        target_bpp: f64,
        params: EncodeParams,
        stop: &dyn enough::Stop,
    ) -> Result<(Vec<u8>, RateMatch)> {
        let op = params.op;
        if !(target_bpp.is_finite() && target_bpp > 0.0) {
            return Err(Error::InvalidArgument("target bpp must be positive"));
        }
        let pixels = (rgb.width * rgb.height) as f64;
        let input = preprocess_rgb(rgb)?;
        let base_params = EncodeParams { op, ..params };
        let independent = (params.regions == Some(RegionMode::Independent))
            .then(|| region_counts(rgb.height, rgb.width))
            .flatten()
            .map(|(v, h)| (v as usize, h as usize));
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
                    let (y, z) = self.analyse_component(&set, ccs, plane, independent, stop)?;
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
                ..base_params
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

    /// The analysis tile grid [`Encoder::encode`] would use for component `ccs` of a
    /// `width x height` picture. Exposed for the parity tests and for diagnostics.
    pub fn analysis_tile_grid(
        width: usize,
        height: usize,
        ccs: usize,
        params: EncodeParams,
    ) -> Result<Vec<AnalysisTile>> {
        let (pw, ph) = (width + width % 2, height + height % 2);
        let (ph, pw) = if ccs == 0 { (ph, pw) } else { (ph / 2, pw / 2) };
        let d = tiles::LATENT_DOWNSCALE[ccs];
        let independent = (params.regions == Some(RegionMode::Independent))
            .then(|| region_counts(height, width))
            .flatten()
            .map(|(v, h)| (v as usize, h as usize));
        let region_areas;
        let regions = match independent {
            None => None,
            Some((num_ver, num_hor)) => {
                let scale = if ccs == 0 { 1 } else { 2 };
                let size =
                    tiles::conformance_tile_size(ph * scale, pw * scale, num_ver, num_hor) / scale;
                region_areas = region_area_grid(ph, pw, num_ver, num_hor);
                Some((region_areas.as_slice(), size))
            }
        };
        tiles::analysis_tiles_with(
            ccs,
            ph,
            pw,
            ph.div_ceil(d),
            pw.div_ceil(d),
            ph.div_ceil(4 * d),
            pw.div_ceil(4 * d),
            regions,
        )
    }

    /// `compress_colocated_tiles` over every analysis tile of one component: the analysis
    /// transform and the hyper-encoder run per tile and only each tile's core is kept.
    fn analyse_component(
        &self,
        set: &ModelSet,
        ccs: usize,
        plane: &Tensor<f32>,
        independent_regions: Option<(usize, usize)>,
        stop: &dyn enough::Stop,
    ) -> Result<(Tensor<f32>, Tensor<i8>)> {
        let (ph, pw) = (plane.h, plane.w);
        let d = tiles::LATENT_DOWNSCALE[ccs];
        let (lh, lw) = (ph.div_ceil(d), pw.div_ceil(d));
        let (hz, wz) = (ph.div_ceil(4 * d), pw.div_ceil(4 * d));
        let chs = LATENT_CHANNELS[ccs];
        let eng = &self.engine;
        // With independent regions the tile size is forced by `cfg_update_for_conformance` and
        // tiles grow only towards neighbours inside their own region.
        let region_areas;
        let regions = match independent_regions {
            None => None,
            Some((num_ver, num_hor)) => {
                // The conformance tile size is computed on the luma picture; chroma halves it.
                let scale = if ccs == 0 { 1 } else { 2 };
                let size =
                    tiles::conformance_tile_size(ph * scale, pw * scale, num_ver, num_hor) / scale;
                region_areas = region_area_grid(ph, pw, num_ver, num_hor);
                Some((region_areas.as_slice(), size))
            }
        };
        let grid = tiles::analysis_tiles_with(ccs, ph, pw, lh, lw, hz, wz, regions)?;
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
        let mut hdr = picture_header(w as u32, h as u32, params.model_id, beta, params);
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
            let scales = component_scales_with(
                &hdr,
                ccs,
                model,
                z_hat,
                None,
                params.rvs,
                GrfsFlags::Derive(params.grfs),
            )?;
            let mask = skip_mask(&scales.skip_scale_log, None)?;

            // 5. psi, then the residual quantisation with its cube-flag decision, region by
            //    region (`compress`'s two `iter_colocated_grids` loops).
            let q = EncoderQuantiser { scales: &scales };
            let out = compress_regions(
                eng, &hdr, ccs, model, z_hat, latent, &mask, &q, lh, lw, stop,
            )?;

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
            hdr.components[ccs].rvs_enabled = params.rvs;
            hdr.components[ccs].grfs_channel_flags = scales.grfs_flags.clone();
            components.push(Component {
                z_hat: z_hats[ccs].clone(),
                psi: out.psi,
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
        let tools = ToolHeader {
            lsbs_enabled: [params.lsbs; 2],
            ..ToolHeader::default()
        };
        out.substream(Marker::Ton, &tools.write(&hdr)?)?;
        out.substream(Marker::Rdi, &RenderingInfo::default().write()?)?;
        for (ccs, marker) in [Marker::Sorp, Marker::Sors].into_iter().enumerate() {
            let regions = &residual_payloads[ccs];
            if hdr.regions.is_some_and(|r| r.independent) {
                // One substream per region, each introduced by its index.
                for (i, payload) in regions.iter().enumerate() {
                    let mut body = alloc::vec![i as u8];
                    body.extend_from_slice(payload);
                    out.substream(marker, &body)?;
                }
            } else {
                let refs: Vec<&[u8]> = regions.iter().map(|p| p.as_slice()).collect();
                out.substream(marker, &join_dependent_regions(&refs))?;
            }
        }
        out.substream(Marker::Soz, &soz)?;
        Ok((out.finish(), components))
    }

    /// `encode_y` / `_ac_encode_y` for one component: one payload per region, each walking the
    /// decoder's channel-chunk loop backwards (ANS is last-in-first-out).
    fn encode_residual(
        &self,
        hdr: &PictureHeader,
        ccs: usize,
        c: &Component,
    ) -> Result<Vec<Vec<u8>>> {
        let comp = &hdr.components[ccs];
        let (lh, lw) = hdr.latent_size(ccs);
        let (lh, lw) = (lh as usize, lw as usize);
        let num_chs = (comp.num_chs as usize).min(c.residual_q.c);
        let threads = comp.num_threads_r as usize;
        let grid = region_grid(hdr, ccs, Plane::Latent);
        let mut out = Vec::with_capacity(grid.core.len());
        let (mut sigma, mut coded, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for area in &grid.core {
            let (rh, rw) = (
                area.height.min(lh - area.y.min(lh)),
                area.width.min(lw - area.x.min(lw)),
            );
            let mut enc = self.tables.encoder(threads)?;
            if rh != 0 && rw != 0 {
                let step = channel_step(rh, rw, num_chs, threads);
                let chunks: Vec<usize> = (0..num_chs).step_by(step).collect();
                for &c0 in chunks.iter().rev() {
                    let c1 = (c0 + step).min(num_chs);
                    sigma.clear();
                    coded.clear();
                    values.clear();
                    for ch in c0..c1 {
                        for y in area.y..area.y + rh {
                            let row = (ch * lh + y) * lw + area.x;
                            sigma.extend(
                                c.scales.scale_log.data[row..row + rw]
                                    .iter()
                                    .map(|&s| distribution_index(s)),
                            );
                            coded.extend_from_slice(&c.mask.data[row..row + rw]);
                            values.extend_from_slice(&c.residual_q.data[row..row + rw]);
                        }
                    }
                    enc.encode_residual(&sigma, &coded, &values)?;
                }
            }
            let parts = enc.finish();
            out.push(join_threads(
                &parts.iter().map(|t| t.as_slice()).collect::<Vec<_>>(),
            ));
        }
        Ok(out)
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

/// `common_modules.py::compress`: the hyper-decoder and the context model region by region,
/// merged the way `merge_psi_overlaps_of_tiles` / `compress_ar_scale_tile` merge them. The
/// mirror of `decoder::reconstruct::reconstruct_latent_with`, and it must stay one.
#[allow(clippy::too_many_arguments)]
fn compress_regions<Q: mcm::Quantiser>(
    eng: &Engine,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_hat: &Tensor<i8>,
    y: &Tensor<f32>,
    mask: &Tensor<bool>,
    q: &Q,
    lh: usize,
    lw: usize,
    stop: &dyn enough::Stop,
) -> Result<mcm::Compressed> {
    let chs = model.chs;
    let independent = hdr.regions.is_some_and(|r| r.independent);
    let img = region_grid(hdr, ccs, Plane::Image);
    let zg = region_grid(hdr, ccs, Plane::HyperLatent);
    let pg = region_grid(hdr, ccs, Plane::Psi);
    let lg = region_grid(hdr, ccs, Plane::Latent);
    let n = img.extended.len();
    let v = eng.tier.block();
    let (pic_h, pic_w) = (hdr.height as usize, hdr.width as usize);

    // 1. psi per region, merged (identical to the decoder's first loop).
    let mut psi = Tensor::<f32>::zeros(4 * chs, lh.div_ceil(2), lw.div_ceil(2))?;
    let mut psi_single = None;
    for r in 0..n {
        stop.check()?;
        let (it, zt) = (img.extended[r], zg.extended[r]);
        let (th, tw, divider) = if ccs == 0 {
            (it.height, it.width, 32)
        } else {
            (it.height.div_ceil(2), it.width.div_ceil(2), 16)
        };
        let out_h = (2 * zt.height)
            .checked_sub(hyper_crop(th, divider))
            .ok_or(Error::InvalidData("region geometry"))?;
        let out_w = (2 * zt.width)
            .checked_sub(hyper_crop(tw, divider))
            .ok_or(Error::InvalidData("region geometry"))?;
        let z_tile;
        let z = if n == 1 {
            z_hat
        } else {
            z_tile = z_hat.window(zt.x, zt.y, zt.width, zt.height)?;
            &z_tile
        };
        let t = model
            .hyper_decoder
            .forward_with(eng, z, out_h, out_w, stop)?;
        if n == 1 && t.h == psi.h && t.w == psi.w {
            psi_single = Some(t);
        } else {
            assign(&mut psi, pg.extended[r], &t.to_planar()?, (0, 0));
        }
        if let Some(p) = &psi_single {
            psi = p.to_planar()?;
        }
    }

    // 2. the residual per region, merged by cores.
    let mut residual_q = Tensor::<i16>::zeros(chs, lh, lw)?;
    let mut residual = Tensor::<f32>::zeros(chs, lh, lw)?;
    let (cube_h, cube_w) = (
        lh.div_ceil(2).div_ceil(mcm::CUBE_SIZE),
        lw.div_ceil(2).div_ceil(mcm::CUBE_SIZE),
    );
    let mut cube_flag = alloc::vec![true; 4 * cube_h * cube_w];
    for r in 0..n {
        stop.check()?;
        let (lt, pt, it) = (lg.extended[r], pg.extended[r], img.extended[r]);
        let (y_tile, mask_tile, psi_tile);
        let (yt, mt, psi_b) = match psi_single.take() {
            Some(p) => (y, mask, p),
            None => {
                y_tile = y.window(lt.x, lt.y, lt.width, lt.height)?;
                mask_tile = mask.window(lt.x, lt.y, lt.width, lt.height)?;
                psi_tile = psi.window(pt.x, pt.y, pt.width, pt.height)?;
                (&y_tile, &mask_tile, BTensor::from_planar(&psi_tile, v)?)
            }
        };
        let out = match &model.context {
            Some(ctx) => ctx.compress(eng, yt, &psi_b, q, mt, SKIP_CUBE_THR, stop)?,
            None => compress_context_free(yt, &psi_b, q, mt, SKIP_CUBE_THR)?,
        };
        // `compress_ar_scale_tile`: independent regions keep their whole tile, dependent ones
        // drop the half-overlap on every side facing another region.
        let (core, offset) = if independent || n == 1 {
            (lt, (0, 0))
        } else {
            let cut = HD_MCM_TILE_OVERLAP / 2 / 16;
            let left = if it.x == 0 { 0 } else { cut };
            let top = if it.y == 0 { 0 } else { cut };
            let right = if it.x + it.width >= pic_w { 0 } else { cut };
            let bottom = if it.y + it.height >= pic_h { 0 } else { cut };
            let (cw, ch) = (
                lt.width.checked_sub(left + right),
                lt.height.checked_sub(top + bottom),
            );
            let (Some(cw), Some(ch)) = (cw, ch) else {
                return Err(Error::InvalidData("region smaller than its overlap"));
            };
            (Area::new(lt.x + left, lt.y + top, cw, ch), (left, top))
        };
        assign(&mut residual_q, core, &out.residual_q, offset);
        assign(&mut residual, core, &out.residual, offset);
        // The cube flags of a region are merged whole, at the latent area downscaled by 16.
        let (rh, rw) = (
            lt.height.div_ceil(2).div_ceil(mcm::CUBE_SIZE),
            lt.width.div_ceil(2).div_ceil(mcm::CUBE_SIZE),
        );
        let (cx, cy) = (lt.x.div_ceil(16), lt.y.div_ceil(16));
        for phase in 0..4 {
            for y in 0..rh.min(cube_h.saturating_sub(cy)) {
                for x in 0..rw.min(cube_w.saturating_sub(cx)) {
                    cube_flag[(phase * cube_h + cy + y) * cube_w + cx + x] =
                        out.cube_flag[(phase * rh + y) * rw + x];
                }
            }
        }
    }
    Ok(mcm::Compressed {
        residual_q,
        residual,
        cube_flag,
        psi,
    })
}

/// The regions of one component's plane, in that plane's coordinates
/// (`calculate_region_coordinates` at depth 0, on the half-size plane for chroma).
fn region_area_grid(h: usize, w: usize, num_ver: usize, num_hor: usize) -> Vec<Area> {
    let axis = |size: usize, n: usize| -> Vec<(usize, usize)> {
        let region = (size.div_ceil(128) / n) * 128;
        (0..n)
            .map(|i| (i * region, if i + 1 < n { (i + 1) * region } else { size }))
            .collect()
    };
    let (ver, hor) = (axis(h, num_ver), axis(w, num_hor));
    let mut out = Vec::with_capacity(ver.len() * hor.len());
    for &(y0, y1) in &ver {
        for &(x0, x1) in &hor {
            out.push(Area::new(x0, y0, x1 - x0, y1 - y0));
        }
    }
    out
}

/// `quant_dequant` with the tools the encoder supports: the gain unit and RVS / GRFS.
struct EncoderQuantiser<'a> {
    scales: &'a ComponentScales,
}

impl mcm::Quantiser for EncoderQuantiser<'_> {
    #[inline]
    fn quantise(&self, ch: usize, index: usize, x: f32, coded: bool) -> (i16, f32) {
        let q = if coded {
            self.scales
                .quantize(None, ch, index, x)
                .unwrap_or(0.0)
                .clamp(-32768.0, 32767.0)
                .round_ties_even()
        } else {
            0.0
        };
        (q as i16, self.scales.dequantize(None, ch, index, q))
    }
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
fn compress_context_free<Q: mcm::Quantiser>(
    y: &Tensor<f32>,
    psi: &BTensor,
    q: &Q,
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
            let (_, dq) = q.quantise(ch, i, d, mask.plane(ch)[i]);
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
            let (sym, dq) = q.quantise(ch, i, d, mask2.plane(ch)[i]);
            residual_q.plane_mut(ch)[i] = sym;
            residual.plane_mut(ch)[i] = dq;
        }
    }
    Ok(mcm::Compressed {
        residual_q,
        residual,
        cube_flag,
        psi: Tensor::<f32>::zeros(1, 1, 1)?,
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
    params: EncodeParams,
) -> PictureHeader {
    use OperatingPoint::{Bop, Hop, Sop};
    let op = params.op;
    let regions = params.regions.and_then(|mode| {
        let (num_ver, num_hor) = region_counts(height as usize, width as usize)?;
        let independent = mode == RegionMode::Independent;
        Some(Regions {
            num_ver,
            num_hor,
            independent,
            hyper_decoder_overlap: if independent {
                0
            } else {
                HYPER_DECODER_OVERLAP
            },
            mcm_overlap: if independent { 0 } else { MCM_OVERLAP },
        })
    });
    let independent = regions.filter(|r| r.independent);
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
        num_threads_z: params.num_threads_z,
        beta_displacement_log,
        regions,
        // `tile_manager_synthesis` is set up from the coded luma size for both components.
        components: [0, 1].map(|ccs| ComponentHeader {
            num_threads_r: params.num_threads_r,
            num_chs: LATENT_CHANNELS[ccs] as u16,
            cube_flags: None,
            rvs_enabled: false,
            grfs_channel_flags: None,
            synthesis_tiling: tiles::synthesis_tiling(
                height as usize,
                width as usize,
                independent.map(|r| (r.num_ver as usize, r.num_hor as usize)),
            ),
        }),
        quality_map: None,
    }
}
