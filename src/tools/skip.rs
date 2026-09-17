//! Skip mode: which latent positions carry a symbol in the residual substream.
//!
//! Ports `ref/src/codec/coding_tools/skip_ls/skip_mode.py` (decoder side). A position is coded
//! when its sigma index exceeds a fixed threshold, or when the cube it falls in is flagged as
//! "do not skip" in the picture header.

use crate::error::Result;
use crate::tensor::Tensor;

/// `thr_skip` in sigma-index units (Q7); positions with `sigma > THR_SKIP` are coded.
pub const THR_SKIP: i32 = 382;
/// Spatial cube edge, in down-shuffled (half resolution) latent samples.
const CUBE_SIZE: usize = 8;

/// Build the coded-position mask for a whole component.
///
/// `cube_flags` is the header's flag vector `[phase][cube_y][cube_x]` (`true` = the cube may be
/// skipped); phases are the 2x2 positions in the order (0,0), (0,1), (1,0), (1,1).
pub fn skip_mask(
    skip_scale_log: &Tensor<i32>,
    cube_flags: Option<&[bool]>,
) -> Result<Tensor<bool>> {
    let (c, h, w) = (skip_scale_log.c, skip_scale_log.h, skip_scale_log.w);
    let mut mask = Tensor::<bool>::zeros(c, h, w)?;
    for (m, &s) in mask.data.iter_mut().zip(&skip_scale_log.data) {
        *m = s > THR_SKIP;
    }
    if let Some(flags) = cube_flags {
        let cube_h = h.div_ceil(2).div_ceil(CUBE_SIZE);
        let cube_w = w.div_ceil(2).div_ceil(CUBE_SIZE);
        debug_assert_eq!(flags.len(), 4 * cube_h * cube_w);
        for y in 0..h {
            for x in 0..w {
                let phase = (y % 2) * 2 + (x % 2);
                let flag = flags[(phase * cube_h + y / 2 / CUBE_SIZE) * cube_w + x / 2 / CUBE_SIZE];
                if !flag {
                    for ch in 0..c {
                        mask.data[(ch * h + y) * w + x] = true;
                    }
                }
            }
        }
    }
    Ok(mask)
}
