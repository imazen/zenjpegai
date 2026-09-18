//! Residual variance scaling (RVS), channel-wise gain flags (GRFS, "cwg" upstream) and the
//! block-wise "likely" sigma map they and LSBS are driven by.
//!
//! Ports `ref/src/codec/coding_tools/quantization/rvs/res_var_scale.py` (decoder direction:
//! `buildTables`, `analyze`, `quantize_scale`, `dequantize_resi`). The per-model thresholds and
//! scale lists come from `ref/cfg/pipeline.json` (`tools_N.model_common...rvs`); they are not in
//! the bitstream.

use alloc::vec::Vec;

use super::log2lin;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// `threshold_rvs_id1` per model id.
const THRESHOLDS: [[u32; 3]; 4] = [
    [4325, 7471, 18667],
    [4325, 6575, 17855],
    [4325, 7782, 18599],
    [4325, 6849, 17855],
];
/// `rvs_scale_list_id1` per model id.
const SCALES: [[u32; 4]; 4] = [
    [115, 125, 146, 160],
    [115, 122, 125, 160],
    [115, 130, 156, 175],
    [115, 130, 156, 160],
];
/// `cnum_list` per component, indexed by model id (`cfg/tools/ResVarScale.json`): how many
/// channels `analyzeCWG` raises the gain flag on. The fifth entry is for a model this port does
/// not have.
const CNUM: [[usize; 4]; 2] = [[5, 24, 35, 64], [10, 10, 10, 10]];
/// `quantize_resi`'s shift: the scale table is relative to 128 (`sigma_precision = 7`).
const SCALE_PRECISION: f32 = 128.0;

/// Value `F.pad` fills the partial 8x8 blocks with.
const PAD_VALUE: i64 = 1411;
const BLOCK: usize = 8;
/// `scaled_sigma_precision - rvs_thr_precision` = (5 + 5 + 7) - 13.
pub(crate) const THRESHOLD_SHIFT: u32 = 4;

/// `analyze`: per 8x8 block of the log-scale map, `(sum + 32) >> 6` with out-of-picture samples
/// counted as 1411, clamped to the table range and repeated over the block.
pub fn likely(scale_log: &Tensor<i32>) -> Result<Tensor<u16>> {
    likely_par(cfg!(feature = "parallel"), scale_log)
}

/// [`likely`], one task per channel when `parallel` holds.
pub(crate) fn likely_par(parallel: bool, scale_log: &Tensor<i32>) -> Result<Tensor<u16>> {
    let (c, h, w) = (scale_log.c, scale_log.h, scale_log.w);
    let mut out = Tensor::<u16>::zeros(c, h, w)?;
    let (bh, bw) = (h.div_ceil(BLOCK), w.div_ceil(BLOCK));
    crate::decoder::stats::for_each_chunk(parallel, &mut out.data, h * w, |ch, dst| {
        let src = scale_log.plane(ch);
        let mut block = alloc::vec![0i64; bh * bw];
        for by in 0..bh {
            for bx in 0..bw {
                let mut sum = 0i64;
                for y in by * BLOCK..(by + 1) * BLOCK {
                    for x in bx * BLOCK..(bx + 1) * BLOCK {
                        sum += if y < h && x < w {
                            src[y * w + x] as i64
                        } else {
                            PAD_VALUE
                        };
                    }
                }
                block[by * bw + bx] = ((sum + 32) >> 6).clamp(0, log2lin::LEN as i64 - 1);
            }
        }
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = block[(y / BLOCK) * bw + x / BLOCK] as u16;
            }
        }
    });
    Ok(out)
}

/// Piecewise-constant lookup over the "likely" index: segment `i` applies to indices above
/// `bounds[i - 1]` (and up to `bounds[i]`).
#[derive(Clone, Debug)]
pub(crate) struct Segments<T> {
    /// Ascending "likely" indices; segment `i + 1` starts right above `bounds[i]`.
    pub bounds: Vec<usize>,
    /// One value per segment (`bounds.len() + 1`).
    pub values: Vec<T>,
}

impl<T: Copy> Segments<T> {
    pub fn get(&self, likely: u16) -> T {
        // The reference adds each segment's delta at indices `> bound`; with bounds that are not
        // strictly increasing this still selects the last segment whose lower bound is passed.
        let i = self
            .bounds
            .iter()
            .take_while(|&&b| likely as usize > b)
            .count();
        self.values[i]
    }
}

/// Thresholds (13 fractional bits) → "likely" indices.
pub(crate) fn threshold_bounds(table: &[u32], thresholds: &[u32]) -> Vec<usize> {
    thresholds
        .iter()
        .map(|&t| log2lin::idx2log(table, (t as u64) << THRESHOLD_SHIFT))
        .collect()
}

/// The tables of one component for its signalled flags.
#[derive(Clone, Debug)]
pub struct Rvs {
    /// Log-domain addition to the scale map, for channels with the gain flag clear / set.
    log: [Segments<i32>; 2],
    /// Residual quantisation factor (relative to 128), same order.
    fwd: [Segments<f32>; 2],
    /// Residual dequantisation factor (16 fractional bits), same order.
    inv: [Segments<f32>; 2],
    /// Per channel: gain flag (all clear when GRFS is off).
    flags: Vec<bool>,
}

impl Rvs {
    /// `None` when neither RVS nor GRFS is enabled (the tool is then the identity).
    pub fn new(
        model_id: usize,
        chs: usize,
        rvs_enabled: bool,
        grfs_channel_flags: Option<&[bool]>,
    ) -> Result<Option<Self>> {
        if !rvs_enabled && grfs_channel_flags.is_none() {
            return Ok(None);
        }
        let (thresholds, scales) = THRESHOLDS
            .get(model_id)
            .zip(SCALES.get(model_id))
            .ok_or(Error::InvalidData("model_id out of range"))?;
        let table = log2lin::table();
        let (bounds, base): (Vec<usize>, Vec<u32>) = if rvs_enabled {
            (threshold_bounds(&table, thresholds), scales.to_vec())
        } else {
            (Vec::new(), alloc::vec![128])
        };
        // ln(scale / 128) in units of the sigma grid: log_k / 2^7 with log_k over 31 steps.
        let unit = (libm::log(54.82) - libm::log(0.11)) / 31.0 / 128.0;
        let build = |factor: Option<f64>| {
            let scaled: Vec<f64> = base
                .iter()
                .map(|&s| match factor {
                    Some(f) => libm::rint(s as f64 * f),
                    None => s as f64,
                })
                .collect();
            (
                Segments {
                    bounds: bounds.clone(),
                    values: scaled
                        .iter()
                        .map(|&s| libm::rint(libm::log(s / 128.0) / unit) as i32)
                        .collect(),
                },
                Segments {
                    bounds: bounds.clone(),
                    values: scaled.iter().map(|&s| s as f32).collect(),
                },
                Segments {
                    bounds: bounds.clone(),
                    values: scaled
                        .iter()
                        .map(|&s| libm::rint(8_388_608.0 / s) as f32)
                        .collect(),
                },
            )
        };
        let mut flags = alloc::vec![false; chs];
        let (clear, set) = match grfs_channel_flags {
            // cwg tables 2 (flag clear: x0.98) and 1 (flag set: x1.06)
            Some(f) => {
                for (d, &s) in flags.iter_mut().zip(f) {
                    *d = s;
                }
                (build(Some(0.98)), build(Some(1.06)))
            }
            None => (build(None), build(None)),
        };
        Ok(Some(Self {
            log: [clear.0, set.0],
            fwd: [clear.1, set.1],
            inv: [clear.2, set.2],
            flags,
        }))
    }

    /// `quantize_scale`: add the log-domain table value to every sample.
    pub fn adjust_scale(&self, scale_log: &mut Tensor<i32>, likely: &Tensor<u16>) {
        self.adjust_scale_par(cfg!(feature = "parallel"), scale_log, likely);
    }

    /// [`Self::adjust_scale`], one task per channel when `parallel` holds.
    pub(crate) fn adjust_scale_par(
        &self,
        parallel: bool,
        scale_log: &mut Tensor<i32>,
        likely: &Tensor<u16>,
    ) {
        let plane = scale_log.h * scale_log.w;
        crate::decoder::stats::for_each_chunk(parallel, &mut scale_log.data, plane, |ch, s| {
            let seg = &self.log[self.flags.get(ch).copied().unwrap_or(false) as usize];
            for (s, &l) in s.iter_mut().zip(likely.plane(ch)) {
                *s += seg.get(l);
            }
        });
    }

    /// `dequantize_resi`: `x * table / 2^16` in single precision.
    pub fn dequantize(&self, ch: usize, likely: u16, x: f32) -> f32 {
        let seg = &self.inv[self.flags.get(ch).copied().unwrap_or(false) as usize];
        x * seg.get(likely) / 65536.0
    }

    /// `quantize_resi`: `x * scale / 2^7` in single precision (encoder side).
    pub fn quantize(&self, ch: usize, likely: u16, x: f32) -> f32 {
        let seg = &self.fwd[self.flags.get(ch).copied().unwrap_or(false) as usize];
        x * seg.get(likely) / SCALE_PRECISION
    }
}

/// `analyzeCWG` (encoder side): the `cnum_list[model_id]` channels with the highest mean scale
/// index carry the gain flag. `scale_log` is the map before RVS adds anything
/// (`encoder_get_scales` hands `analyze` the `skip_scale_log` it just computed).
pub fn grfs_flags(model_id: usize, ccs: usize, scale_log: &Tensor<i32>) -> Result<Vec<bool>> {
    let cnum = *CNUM
        .get(ccs)
        .and_then(|c| c.get(model_id))
        .ok_or(Error::InvalidData("model_id out of range"))?;
    let n = (scale_log.h * scale_log.w) as f32;
    // `torch.mean` over a float32 plane, then `torch.sort(descending=True)`.
    let mut order: Vec<(usize, f32)> = (0..scale_log.c)
        .map(|ch| {
            let mean = scale_log.plane(ch).iter().map(|&v| v as f32).sum::<f32>() / n;
            (ch, mean)
        })
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut flags = alloc::vec![false; scale_log.c];
    for &(ch, _) in order.iter().take(cnum.min(scale_log.c)) {
        flags[ch] = true;
    }
    Ok(flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn likely_blocks() {
        // 9x9 of constant 640: full block -> (64 * 640 + 32) >> 6 = 640; partial blocks mix in 1411.
        let t = Tensor::from_vec(1, 9, 9, alloc::vec![640; 81]).unwrap();
        let l = likely(&t).unwrap();
        assert_eq!(l.at(0, 0, 0), 640);
        assert_eq!(l.at(0, 7, 7), 640);
        let edge = ((8 * 640 + 56 * 1411 + 32) >> 6) as u16;
        assert_eq!(l.at(0, 0, 8), edge);
        let corner = ((640 + 63 * 1411 + 32) >> 6) as u16;
        assert_eq!(l.at(0, 8, 8), corner);
    }

    #[test]
    fn identity_without_flags() {
        assert!(Rvs::new(0, 160, false, None).unwrap().is_none());
    }
}
