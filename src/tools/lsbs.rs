//! Latent scaling before synthesis (LSBS).
//!
//! Ports `ref/src/codec/coding_tools/ls_processing/lsbs/lsbs_scale_mode.py` (`buildTables`,
//! `post_processing`): `y_hat += (s0 * mean + s1 * residual) / 2^13` with `mean = y_hat -
//! residual` and `(s0, s1)` picked per sample from the block-wise "likely" sigma map
//! ([`super::rvs::likely`]). The scale lists are per model (`ref/cfg/pipeline.json`), the
//! thresholds are the tool's defaults; neither is in the bitstream.

use super::log2lin;
use super::rvs::{Segments, threshold_bounds};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// `threshold_lsbs` (default).
const THRESHOLDS: [u32; 2] = [7782, 8192];
/// `scale0_lsbs` per model id: multiplies the mean.
const SCALE0: [[i32; 3]; 4] = [[23, 0, 46], [17, 0, 34], [12, 0, 23], [9, 0, 28]];
/// `scale1_lsbs` per model id: multiplies the residual.
const SCALE1: [[i32; 3]; 4] = [[28, 0, 56], [23, 0, 46], [17, 0, 34], [12, 0, 26]];

/// `post_processing` for one component.
pub fn apply(
    model_id: usize,
    y_hat: &mut Tensor<f32>,
    residual: &Tensor<f32>,
    likely: &Tensor<u16>,
) -> Result<()> {
    apply_par(
        cfg!(feature = "parallel"),
        model_id,
        y_hat,
        residual,
        likely,
    )
}

/// [`apply`], chunked over the plane data when `parallel` holds.
pub(crate) fn apply_par(
    parallel: bool,
    model_id: usize,
    y_hat: &mut Tensor<f32>,
    residual: &Tensor<f32>,
    likely: &Tensor<u16>,
) -> Result<()> {
    let (s0, s1) = SCALE0
        .get(model_id)
        .zip(SCALE1.get(model_id))
        .ok_or(Error::InvalidData("model_id out of range"))?;
    if (y_hat.c, y_hat.h, y_hat.w) != (residual.c, residual.h, residual.w)
        || (y_hat.c, y_hat.h, y_hat.w) != (likely.c, likely.h, likely.w)
    {
        return Err(Error::InvalidArgument("LSBS: shapes differ"));
    }
    let bounds = threshold_bounds(&log2lin::table(), &THRESHOLDS);
    let pairs: alloc::vec::Vec<(f32, f32)> = s0
        .iter()
        .zip(s1)
        .map(|(&a, &b)| (a as f32, b as f32))
        .collect();
    let seg = Segments {
        bounds,
        values: pairs,
    };
    crate::decoder::stats::for_each_chunk(parallel, &mut y_hat.data, 16384, |i, chunk| {
        let base = i * 16384;
        for (j, y) in chunk.iter_mut().enumerate() {
            let (r, l) = (residual.data[base + j], likely.data[base + j]);
            let (a, b) = seg.get(l);
            let mean = *y - r;
            // Two products and a sum, then an exact power-of-two division, like the reference.
            let additive = (a * mean + b * r) / 8192.0;
            *y += additive;
        }
    });
    Ok(())
}
