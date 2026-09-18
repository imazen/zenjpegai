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

use crate::error::{Error, Result};
use crate::header::PictureHeader;
use crate::mans::{AnsDecoder, AnsTables};
use crate::model::CommonModel;
use crate::model::common::{SIGMA_IDX_MAX, SIGMA_LEVELS, SIGMA_PRECISION, Z_OFFSET};
use crate::tensor::Tensor;
use crate::tools::gain::GainUnit;
use crate::tools::qualmap::QualityMap;
use crate::tools::regions::{Plane, region_grid};
use crate::tools::rvs::{self, Rvs};
use crate::tools::skip::skip_mask;

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
    component_scales_with(
        hdr,
        ccs,
        model,
        z_hat,
        quality_map,
        hdr.components[ccs].rvs_enabled,
        GrfsFlags::Signalled(flags.as_deref()),
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
    let (lh, lw) = hdr.latent_size(ccs);
    let gain = GainUnit::new(&model.gain_vector_log, hdr.beta_displacement_log[ccs]);
    let mut skip_scale_log = model
        .hsd
        .forward(z_hat, lh as usize, lw as usize, SIGMA_IDX_MAX)?;
    for (ch, &add) in gain.scaler_log.iter().enumerate() {
        if !(-(1 << 13)..(1 << 13)).contains(&add) {
            return Err(Error::InvalidData("log-domain scaler out of range"));
        }
        for v in skip_scale_log.plane_mut(ch) {
            *v += add;
        }
    }
    if let Some(q) = quality_map {
        q.adjust_scale(&mut skip_scale_log)?;
    }
    let likely = rvs::likely(&skip_scale_log)?;
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
        r.adjust_scale(&mut scale_log, &likely);
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
    dec.decode_z(&model.z_cdfs, hz * wz, &mut symbols)?;
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
    stop.check()?;
    let comp = &hdr.components[ccs];
    let chs = model.chs;
    let (lh, lw) = hdr.latent_size(ccs);
    let (lh, lw) = (lh as usize, lw as usize);
    let (hz, wz) = hdr.hyper_latent_size(ccs);

    let z_hat = decode_z(z_dec, model, hz as usize, wz as usize)?;
    stop.check()?;

    // _decode_scale + the quantiser's log-domain additions; shared with the encoder.
    let scales = component_scales(hdr, ccs, model, &z_hat, quality_map)?;
    let ComponentScales {
        skip_scale_log,
        scale_log,
        ..
    } = &scales;
    let mask = skip_mask(skip_scale_log, comp.cube_flags.as_deref())?;

    // decode_y: region by region, channel chunk by channel chunk.
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
    let mut residual_q = Tensor::<i16>::zeros(chs, lh, lw)?;
    let mut sigma: Vec<u8> = Vec::new();
    let mut coded: Vec<bool> = Vec::new();
    let mut symbols: Vec<i16> = Vec::new();
    for (area, payload) in grid.core.iter().zip(region_payloads) {
        // An absent region (dropped independent substream) decodes to zeros.
        let Some(payload) = payload else { continue };
        let (rh, rw) = (
            area.height.min(lh - area.y.min(lh)),
            area.width.min(lw - area.x.min(lw)),
        );
        if rh == 0 || rw == 0 || num_decode == 0 {
            continue;
        }
        let threads = crate::container::split_threads(payload, num_threads)?;
        let mut dec = tables.decoder(&threads)?;
        let step = channel_step(rh, rw, num_decode, num_threads);
        for c0 in (0..num_decode).step_by(step) {
            stop.check()?;
            let c1 = (c0 + step).min(num_chs);
            let n = (c1 - c0) * rh * rw;
            sigma.clear();
            coded.clear();
            sigma.reserve(n);
            coded.reserve(n);
            for ch in c0..c1 {
                for y in area.y..area.y + rh {
                    let row = (ch * lh + y) * lw + area.x;
                    sigma.extend(
                        scale_log.data[row..row + rw]
                            .iter()
                            .map(|&s| distribution_index(s)),
                    );
                    coded.extend_from_slice(&mask.data[row..row + rw]);
                }
            }
            symbols.clear();
            symbols.resize(n, 0);
            dec.decode_residual(&sigma, &coded, &mut symbols)?;
            let mut it = symbols.iter();
            for ch in c0..c1 {
                for y in area.y..area.y + rh {
                    let row = (ch * lh + y) * lw + area.x;
                    for d in &mut residual_q.data[row..row + rw] {
                        *d = *it.next().unwrap_or(&0);
                    }
                }
            }
        }
    }

    stop.check()?;
    let residual = dequantize_residual(&scales, quality_map, &residual_q)?;
    let ComponentScales {
        skip_scale_log,
        scale_log,
        likely,
        ..
    } = scales;

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

/// `quantizer.dequantize_resi` over a whole component.
pub fn dequantize_residual(
    scales: &ComponentScales,
    quality_map: Option<&QualityMap>,
    residual_q: &Tensor<i16>,
) -> Result<Tensor<f32>> {
    let (c, h, w) = (residual_q.c, residual_q.h, residual_q.w);
    let mut residual = Tensor::<f32>::zeros(c, h, w)?;
    for ch in 0..c {
        let src = residual_q.plane(ch);
        for (i, (d, &q)) in residual.plane_mut(ch).iter_mut().zip(src).enumerate() {
            *d = scales.dequantize(quality_map, ch, i, q as f32);
        }
    }
    Ok(residual)
}
