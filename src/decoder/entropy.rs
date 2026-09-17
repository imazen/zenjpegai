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

/// `GMProbModel.build_indexes`: sigma index (Q7) → distribution number, round to nearest.
#[inline]
fn distribution_index(scale_log: i32) -> u8 {
    let half = 1 << (SIGMA_PRECISION - 1);
    ((scale_log + half) >> SIGMA_PRECISION).clamp(0, SIGMA_LEVELS - 1) as u8
}

/// `_cal_step_size`: channels per `decode_sgm` call for a region of `lh * lw` positions.
fn channel_step(lh: usize, lw: usize, num_chs: usize, num_threads: usize) -> usize {
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

    // _decode_scale: hyper-scale decoder, then the quantiser's log-domain additions (gain unit).
    let gain = GainUnit::new(&model.gain_vector_log, hdr.beta_displacement_log[ccs]);
    let mut skip_scale_log = model.hsd.forward(&z_hat, lh, lw, SIGMA_IDX_MAX)?;
    for (ch, &add) in gain.scaler_log.iter().enumerate() {
        if !(-(1 << 13)..(1 << 13)).contains(&add) {
            return Err(Error::InvalidData("log-domain scaler out of range"));
        }
        for v in skip_scale_log.plane_mut(ch) {
            *v += add;
        }
    }
    // quantizer.analyze + quantize_scale(incl rvs): the block-wise "likely" map is always built
    // (LSBS reads it too); RVS / GRFS shift the scales the residual was coded with.
    if let Some(q) = quality_map {
        q.adjust_scale(&mut skip_scale_log)?;
    }
    let likely = rvs::likely(&skip_scale_log)?;
    let rvs = Rvs::new(
        hdr.model_id as usize,
        chs,
        comp.rvs_enabled,
        comp.grfs_channel_flags.as_deref(),
    )?;
    let mut scale_log = skip_scale_log.clone();
    if let Some(r) = &rvs {
        r.adjust_scale(&mut scale_log, &likely);
    }
    let mask = skip_mask(&skip_scale_log, comp.cube_flags.as_deref())?;

    // decode_y: region by region, channel chunk by channel chunk.
    let num_chs = (comp.num_chs as usize).min(chs);
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
        if rh == 0 || rw == 0 || num_chs == 0 {
            continue;
        }
        let threads = crate::container::split_threads(payload, num_threads)?;
        let mut dec = tables.decoder(&threads)?;
        let step = channel_step(rh, rw, num_chs, num_threads);
        for c0 in (0..num_chs).step_by(step) {
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
    let mut residual = Tensor::<f32>::zeros(chs, lh, lw)?;
    // dequantize_resi runs the tools in reverse: RVS, then the quality map, then the gain unit.
    for ch in 0..chs {
        let src = residual_q.plane(ch);
        let lk = likely.plane(ch);
        for (i, ((d, &q), &l)) in residual
            .plane_mut(ch)
            .iter_mut()
            .zip(src)
            .zip(lk)
            .enumerate()
        {
            let mut x = q as f32;
            if let Some(r) = &rvs {
                x = r.dequantize(ch, l, x);
            }
            if let Some(m) = quality_map {
                x = m.dequantize(i, x);
            }
            *d = gain.dequantize(ch, x);
        }
    }

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
