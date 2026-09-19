//! Entropy stage of the decoder: substreams → `z_hat`, sigma indices, skip mask, residual.
//!
//! Ports `CommonEncDecModules.decode` / `decode_z` / `decode_y` / `_ac_decode_y`
//! (`ref/src/codec/coding_tools/core_models/CCS_SGMM/common_modules.py`). Everything here is
//! integer arithmetic and must match the reference exactly; the only float output is the
//! dequantised residual, which is an elementwise float32 division and is exact as well.
//!
//! Order matters and is part of the format. Both components' `z` live in the one SOZ payload,
//! luma first; each component's residual is coded per region, and within a region in chunks of
//! whole channels whose symbol count is a multiple of `4 * threads` (so that no chunk but the
//! last takes the ANS tail path).

use alloc::vec::Vec;

use enough::Stop;

use crate::decoder::stats::{Gate, Probe, Tick, for_each_chunk, par_map, record};
use crate::error::{Error, Result};
use crate::header::PictureHeader;
use crate::mans::{AnsDecoder, AnsTables};
use crate::model::CommonModel;
use crate::model::common::{SIGMA_IDX_MAX, SIGMA_LEVELS, SIGMA_PRECISION, Z_OFFSET};
use crate::tensor::Tensor;
use crate::tools::gain::GainUnit;
use crate::tools::qualmap::QualityMap;
use crate::tools::regions::{Area, Plane, region_grid};
use crate::tools::rvs::{self, Rvs};
use crate::tools::skip::skip_mask_par;

/// Entropy-stage output of one component.
#[derive(Clone, Debug)]
pub struct ComponentEntropy {
    /// Hyper-latent, `[C, hz, wz]`.
    pub z_hat: Tensor<i8>,
    /// Sigma indices (Q7) driving skip mode: hyper-scale decoder output plus the gain unit.
    pub skip_scale_log: Tensor<i32>,
    /// Sigma indices driving the entropy coder (`skip_scale_log` plus RVS when enabled).
    pub scale_log: Tensor<i32>,
    /// Block-wise mean of `skip_scale_log` as a [`crate::tools::log2lin`] index: selects the RVS
    /// and LSBS table segments.
    pub likely: Tensor<u16>,
    /// `true` where a residual symbol is present in the stream.
    pub mask: Tensor<bool>,
    /// Quantised residual, `[C, H, W]`.
    pub residual_q: Tensor<i16>,
    /// Dequantised residual.
    pub residual: Tensor<f32>,
}

/// The scale maps of one component, and the quantiser state derived with them.
///
/// Ports `CommonEncDecModules.encoder_get_scales` / the first half of `decode_y`; both
/// directions must derive these identically, so both call [`component_scales`].
pub struct ComponentScales {
    /// Sigma indices (Q7) driving skip mode: hyper-scale decoder plus the gain unit (plus the
    /// quality map).
    pub skip_scale_log: Tensor<i32>,
    /// Sigma indices driving the entropy coder: [`Self::skip_scale_log`] plus RVS / GRFS.
    pub scale_log: Tensor<i32>,
    /// Block-wise mean of `skip_scale_log` as a [`crate::tools::log2lin`] index.
    pub likely: Tensor<u16>,
    pub gain: GainUnit,
    pub rvs: Option<Rvs>,
    /// The gain flags in force (`analyzeCWG` derived them when the encoder asked for GRFS);
    /// `None` when GRFS is off.
    pub grfs_flags: Option<Vec<bool>>,
}

impl ComponentScales {
    /// `quantizer.dequantize_resi`: the tools in reverse order (RVS, quality map, gain unit).
    #[inline]
    pub fn dequantize(&self, quality_map: Option<&QualityMap>, ch: usize, i: usize, q: f32) -> f32 {
        let mut x = q;
        if let Some(r) = &self.rvs {
            x = r.dequantize(ch, self.likely.plane(ch)[i], x);
        }
        if let Some(m) = quality_map {
            x = m.dequantize(i, x);
        }
        self.gain.dequantize(ch, x)
    }

    /// `quantizer.quantize_resi`: the tools in forward order (gain unit, quality map, RVS).
    #[inline]
    pub fn quantize(&self, quality_map: Option<&QualityMap>, ch: usize, i: usize, x: f32) -> f32 {
        let mut v = x * self.gain.scaler[ch];
        if let Some(m) = quality_map {
            v = m.quantize(i, v);
        }
        match &self.rvs {
            Some(r) => r.quantize(ch, self.likely.plane(ch)[i], v),
            None => v,
        }
    }
}

/// Where the GRFS gain flags come from.
#[derive(Clone, Copy, Debug)]
pub enum GrfsFlags<'a> {
    /// Decoding: the flags the picture header carries.
    Signalled(Option<&'a [bool]>),
    /// Encoding: `analyzeCWG` derives them from the scale map when GRFS is enabled.
    Derive(bool),
}

/// `_decode_scale` + `encoder_get_scales`: hyper-scale decoder, gain unit, quality map, the
/// block-wise `likely` map, then RVS / GRFS. Identical in both directions.
pub fn component_scales(
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_hat: &Tensor<i8>,
    quality_map: Option<&QualityMap>,
) -> Result<ComponentScales> {
    let flags = hdr.components[ccs].grfs_channel_flags.clone();
    component_scales_impl(
        hdr,
        ccs,
        model,
        z_hat,
        quality_map,
        hdr.components[ccs].rvs_enabled,
        GrfsFlags::Signalled(flags.as_deref()),
        cfg!(feature = "parallel"),
    )
}

/// [`component_scales`] with the RVS / GRFS signalling given explicitly, so that the encoder can
/// derive the gain flags from the scale map it is in the middle of computing.
#[allow(clippy::too_many_arguments)]
pub fn component_scales_with(
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_hat: &Tensor<i8>,
    quality_map: Option<&QualityMap>,
    rvs_enabled: bool,
    grfs: GrfsFlags<'_>,
) -> Result<ComponentScales> {
    component_scales_impl(
        hdr,
        ccs,
        model,
        z_hat,
        quality_map,
        rvs_enabled,
        grfs,
        cfg!(feature = "parallel"),
    )
}

/// The elementwise sections run on the rayon pool when `parallel` holds; the per-element
/// arithmetic is unchanged either way.
#[allow(clippy::too_many_arguments)]
fn component_scales_impl(
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_hat: &Tensor<i8>,
    quality_map: Option<&QualityMap>,
    rvs_enabled: bool,
    grfs: GrfsFlags<'_>,
    parallel: bool,
) -> Result<ComponentScales> {
    let (lh, lw) = hdr.latent_size(ccs);
    let (lh, lw) = (lh as usize, lw as usize);
    let gain = GainUnit::new(&model.gain_vector_log, hdr.beta_displacement_log[ccs]);
    let mut skip_scale_log = model.hsd.forward(z_hat, lh, lw, SIGMA_IDX_MAX)?;
    for &add in &gain.scaler_log {
        if !(-(1 << 13)..(1 << 13)).contains(&add) {
            return Err(Error::InvalidData("log-domain scaler out of range"));
        }
    }
    for_each_chunk(parallel, &mut skip_scale_log.data, lh * lw, |ch, plane| {
        let add = gain.scaler_log[ch];
        for v in plane {
            *v += add;
        }
    });
    if let Some(q) = quality_map {
        q.adjust_scale_par(parallel, &mut skip_scale_log)?;
    }
    let likely = rvs::likely_par(parallel, &skip_scale_log)?;
    // `analyzeCWG` runs on the scale map as it stands before RVS adds anything.
    let grfs_flags = match grfs {
        GrfsFlags::Signalled(f) => f.map(<[bool]>::to_vec),
        GrfsFlags::Derive(false) => None,
        GrfsFlags::Derive(true) => Some(rvs::grfs_flags(
            hdr.model_id as usize,
            ccs,
            &skip_scale_log,
        )?),
    };
    let rvs = Rvs::new(
        hdr.model_id as usize,
        model.chs,
        rvs_enabled,
        grfs_flags.as_deref(),
    )?;
    let mut scale_log = skip_scale_log.clone();
    if let Some(r) = &rvs {
        r.adjust_scale_par(parallel, &mut scale_log, &likely);
    }
    Ok(ComponentScales {
        skip_scale_log,
        scale_log,
        likely,
        gain,
        rvs,
        grfs_flags,
    })
}

/// `GMProbModel.build_indexes`: sigma index (Q7) → distribution number, round to nearest.
#[inline]
pub fn distribution_index(scale_log: i32) -> u8 {
    let half = 1 << (SIGMA_PRECISION - 1);
    ((scale_log + half) >> SIGMA_PRECISION).clamp(0, SIGMA_LEVELS - 1) as u8
}

/// `_cal_step_size`: channels per `decode_sgm` call for a region of `lh * lw` positions.
pub fn channel_step(lh: usize, lw: usize, num_chs: usize, num_threads: usize) -> usize {
    let delta = num_threads * 4;
    (1..=num_chs)
        .find(|item| (lh * lw * item).is_multiple_of(delta))
        .unwrap_or(num_chs.max(1))
}

/// Decode the `z` symbols of one component from the shared SOZ decoder.
pub fn decode_z(
    dec: &mut AnsDecoder<'_>,
    model: &CommonModel,
    hz: usize,
    wz: usize,
) -> Result<Tensor<i8>> {
    let mut symbols = alloc::vec![0u8; model.chs * hz * wz];
    {
        let _symbols_charge = crate::mem::Charge::of_vec(&symbols);
        dec.decode_z(&model.z_cdfs, hz * wz, &mut symbols)?;
    }
    let data = symbols
        .into_iter()
        .map(|s| (s as i32 - Z_OFFSET) as i8)
        .collect();
    Tensor::from_vec(model.chs, hz, wz, data)
}

/// Decode one component: its `z` from `z_dec`, then its residual from `region_payloads`
/// (the output of [`crate::container::split_regions`] for SORP or SORS).
#[allow(clippy::too_many_arguments)]
pub fn decode_component(
    tables: &AnsTables,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_dec: &mut AnsDecoder<'_>,
    region_payloads: &[Option<&[u8]>],
    quality_map: Option<&QualityMap>,
    max_channels: Option<u16>,
    stop: &dyn enough::Stop,
) -> Result<ComponentEntropy> {
    let (hz, wz) = hdr.hyper_latent_size(ccs);
    let z_hat = decode_z(z_dec, model, hz as usize, wz as usize)?;
    decode_component_body(
        tables,
        hdr,
        ccs,
        model,
        z_hat,
        region_payloads,
        quality_map,
        max_channels,
        cfg!(feature = "parallel"),
        None,
        stop,
    )
}

/// Everything of [`decode_component`] after the shared z substream. Both components' bodies
/// touch disjoint data — the caller may run them on the pool — and inside a body the
/// regions, the ANS threads of each region, and the elementwise passes are all
/// independent. `parallel` decides whether that independence reaches the rayon pool; the
/// output bits are identical either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_component_body(
    tables: &AnsTables,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    z_hat: Tensor<i8>,
    region_payloads: &[Option<&[u8]>],
    quality_map: Option<&QualityMap>,
    max_channels: Option<u16>,
    parallel: bool,
    probe: Option<&Probe>,
    stop: &dyn enough::Stop,
) -> Result<ComponentEntropy> {
    // Region and ANS-thread tasks check through the gate so a counting `Stop` stays exact.
    let gate = Gate::new(stop);
    let stop = &gate;
    stop.check()?;
    let t_body = Tick::now();
    let comp = &hdr.components[ccs];
    let chs = model.chs;
    let (lh, lw) = hdr.latent_size(ccs);
    let (lh, lw) = (lh as usize, lw as usize);

    // _decode_scale + the quantiser's log-domain additions; shared with the encoder.
    let t = Tick::now();
    let scales = component_scales_impl(
        hdr,
        ccs,
        model,
        &z_hat,
        quality_map,
        comp.rvs_enabled,
        GrfsFlags::Signalled(comp.grfs_channel_flags.as_deref()),
        parallel,
    )?;
    record(Probe::field(probe, ccs, |p| &p.scales), t);
    let ComponentScales {
        skip_scale_log,
        scale_log,
        ..
    } = &scales;
    let t = Tick::now();
    let mask = skip_mask_par(parallel, skip_scale_log, comp.cube_flags.as_deref())?;
    record(Probe::field(probe, ccs, |p| &p.mask), t);

    // decode_y: region by region, channel chunk by channel chunk. Regions carry their own
    // ANS payload — they decode independently, so with several regions each becomes a task.
    let num_chs = (comp.num_chs as usize).min(chs);
    // Progressive decode (`num_decode_chs`): stop after the first channels. The chunking is
    // computed from the reduced count, like the reference does; a chunk may still end past it.
    let num_decode = max_channels.map_or(num_chs, |m| (m as usize).min(num_chs));
    let num_threads = comp.num_threads_r as usize;
    let grid = region_grid(hdr, ccs, Plane::Latent);
    if region_payloads.len() != grid.core.len() {
        return Err(Error::InvalidData(
            "residual region count does not match the header",
        ));
    }
    let ctx = ResidualCtx {
        tables,
        scale_log,
        mask: &mask,
        lh,
        lw,
        num_chs,
        num_decode,
        num_threads,
        probe,
        ccs,
    };
    let mut residual_q = Tensor::<i16>::zeros(chs, lh, lw)?;
    if grid.core.len() == 1 {
        // One region spans the whole plane: decode straight into it, no tile copy.
        let area = grid.core[0];
        let (rh, rw) = (
            area.height.min(lh - area.y.min(lh)),
            area.width.min(lw - area.x.min(lw)),
        );
        if let (Some(payload), true) = (region_payloads[0], rh != 0 && rw != 0 && num_decode != 0) {
            decode_region_into(
                &ctx,
                &mut residual_q.data,
                lh * lw,
                lw,
                area,
                area,
                rh,
                rw,
                payload,
                parallel,
                stop,
            )?;
        }
    } else {
        // One `[chs, rh, rw]` tile per region, decoded in parallel, then merged channel by
        // channel (disjoint writes).
        let tiles = par_map(parallel, grid.core.len(), |r| {
            let area = grid.core[r];
            let Some(payload) = region_payloads[r] else {
                return Ok(None);
            };
            let (rh, rw) = (
                area.height.min(lh - area.y.min(lh)),
                area.width.min(lw - area.x.min(lw)),
            );
            if rh == 0 || rw == 0 || num_decode == 0 {
                return Ok(None);
            }
            let mut tile = alloc::vec![0i16; chs * rh * rw];
            let _tile_charge = crate::mem::Charge::of_vec(&tile);
            decode_region_into(
                &ctx,
                &mut tile,
                rh * rw,
                rw,
                Area::new(0, 0, rw, rh),
                area,
                rh,
                rw,
                payload,
                parallel,
                stop,
            )?;
            Ok(Some((tile, rh, rw)))
        })?;
        for_each_chunk(parallel, &mut residual_q.data, lh * lw, |ch, plane| {
            for (r, tile) in tiles.iter().enumerate() {
                let Some((tile, rh, rw)) = tile else { continue };
                let area = &grid.core[r];
                for y in 0..*rh {
                    let d = (area.y + y) * lw + area.x;
                    plane[d..d + rw].copy_from_slice(&tile[(ch * rh + y) * rw..][..*rw]);
                }
            }
        });
    }

    stop.check()?;
    let t = Tick::now();
    let residual = dequantize_residual_impl(parallel, &scales, quality_map, &residual_q)?;
    record(Probe::field(probe, ccs, |p| &p.dequantize), t);
    let ComponentScales {
        skip_scale_log,
        scale_log,
        likely,
        ..
    } = scales;
    record(Probe::field(probe, ccs, |p| &p.entropy_body), t_body);

    Ok(ComponentEntropy {
        z_hat,
        skip_scale_log,
        scale_log,
        likely,
        mask,
        residual_q,
        residual,
    })
}

/// Everything a region's residual decode needs that is shared with its siblings.
struct ResidualCtx<'a> {
    tables: &'a AnsTables,
    scale_log: &'a Tensor<i32>,
    mask: &'a Tensor<bool>,
    /// Global plane dims (the `scale_log`/`mask` rows the gather reads).
    lh: usize,
    lw: usize,
    num_chs: usize,
    num_decode: usize,
    num_threads: usize,
    probe: Option<&'a Probe>,
    ccs: usize,
}

/// One region's residual: gather `sigma`/`coded` for its `num_chs` channels at the region's
/// place in the global planes (`gather`), then the sequential `decode_sgm` channel chunks
/// (ANS state; internally threaded when the payload is multi-thread), and scatter the symbols
/// into `dst` at `scatter` — indexed `dst[ch * plane_stride + (scatter.y + y) * row_stride +
/// scatter.x + x]`, so the same routine fills a per-region tile (`plane_stride = rh*rw`,
/// `row_stride = rw`, `scatter = (0,0)`) or a region's rectangle of the whole plane.
#[allow(clippy::too_many_arguments)]
fn decode_region_into(
    ctx: &ResidualCtx<'_>,
    dst: &mut [i16],
    plane_stride: usize,
    row_stride: usize,
    scatter: Area,
    gather: Area,
    rh: usize,
    rw: usize,
    payload: &[u8],
    inner_par: bool,
    stop: &dyn enough::Stop,
) -> Result<()> {
    let threads = crate::container::split_threads(payload, ctx.num_threads)?;
    let mut dec = ctx.tables.decoder(&threads)?;
    dec.set_parallel(inner_par);
    let step = channel_step(rh, rw, ctx.num_decode, ctx.num_threads);
    let (lh, lw) = (ctx.lh, ctx.lw);
    let (gy, gx) = (gather.y, gather.x);
    let t = Tick::now();
    // `sigma`/`coded` for every coded channel of the region, region-local `[ch][y][x]`.
    let mut sigma = alloc::vec![0u8; ctx.num_chs * rh * rw];
    let mut coded = alloc::vec![false; ctx.num_chs * rh * rw];
    let _gather_charge =
        crate::mem::Charge::new(crate::mem::vec_bytes(&sigma) + crate::mem::vec_bytes(&coded));
    for_each_chunk(inner_par, &mut sigma, rh * rw, |ch, block| {
        for (y, drow) in block.chunks_exact_mut(rw).enumerate() {
            let src = (ch * lh + gy + y) * lw + gx;
            for (d, &s) in drow.iter_mut().zip(&ctx.scale_log.data[src..src + rw]) {
                *d = distribution_index(s);
            }
        }
    });
    for_each_chunk(inner_par, &mut coded, rh * rw, |ch, block| {
        for (y, drow) in block.chunks_exact_mut(rw).enumerate() {
            let src = (ch * lh + gy + y) * lw + gx;
            drow.copy_from_slice(&ctx.mask.data[src..src + rw]);
        }
    });
    record(Probe::field(ctx.probe, ctx.ccs, |p| &p.gather), t);

    let t = Tick::now();
    let (sy, sx) = (scatter.y, scatter.x);
    let mut symbols: Vec<i16> = Vec::new();
    let mut symbols_charge = crate::mem::Charge::EMPTY;
    for c0 in (0..ctx.num_decode).step_by(step) {
        stop.check()?;
        let c1 = (c0 + step).min(ctx.num_chs);
        let n = (c1 - c0) * rh * rw;
        symbols.clear();
        symbols.resize(n, 0);
        symbols_charge.resize(crate::mem::vec_bytes(&symbols));
        dec.decode_residual(
            &sigma[c0 * rh * rw..][..n],
            &coded[c0 * rh * rw..][..n],
            &mut symbols,
        )?;
        // Scatter, one channel block per task: `symbols` rows are contiguous per (ch, y).
        for_each_chunk(
            inner_par,
            &mut dst[c0 * plane_stride..c1 * plane_stride],
            plane_stride,
            |i, block| {
                for y in 0..rh {
                    let s = (i * rh + y) * rw;
                    let d = (sy + y) * row_stride + sx;
                    block[d..d + rw].copy_from_slice(&symbols[s..s + rw]);
                }
            },
        );
    }
    record(Probe::field(ctx.probe, ctx.ccs, |p| &p.residual), t);
    Ok(())
}

/// `quantizer.dequantize_resi` over a whole component.
pub fn dequantize_residual(
    scales: &ComponentScales,
    quality_map: Option<&QualityMap>,
    residual_q: &Tensor<i16>,
) -> Result<Tensor<f32>> {
    dequantize_residual_impl(cfg!(feature = "parallel"), scales, quality_map, residual_q)
}

/// [`dequantize_residual`], one task per channel when `parallel` holds.
fn dequantize_residual_impl(
    parallel: bool,
    scales: &ComponentScales,
    quality_map: Option<&QualityMap>,
    residual_q: &Tensor<i16>,
) -> Result<Tensor<f32>> {
    let (c, h, w) = (residual_q.c, residual_q.h, residual_q.w);
    let mut residual = Tensor::<f32>::zeros(c, h, w)?;
    for_each_chunk(parallel, &mut residual.data, h * w, |ch, plane| {
        let src = residual_q.plane(ch);
        for (i, (d, &q)) in plane.iter_mut().zip(src).enumerate() {
            *d = scales.dequantize(quality_map, ch, i, q as f32);
        }
    });
    Ok(residual)
}
