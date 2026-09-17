//! LEF ("luma edge filter"): adaptive luma sharpening steered by the entropy stage's scale map.
//!
//! Ports `ref/src/codec/coding_tools/filters/LEF/LEFfilter.py` (`LEF.decompress`,
//! `adptive_sharpness`). One channel of the luma sigma-index map (`LEF_chIdx`) is
//! nearest-upsampled to the picture; where it crosses one of three per-model thresholds the
//! difference between the picture and a 3x3 smoothed copy is amplified:
//!
//! ```text
//! x      = Y / range
//! blur   = clamp(conv3x3(x, [[5,5,5],[5,24,5],[5,5,5]] / 64), 0, 1)     interior; border: x
//! out    = clamp(blur + (x - blur) * mag(region), 0, 1) * range
//! ```
//!
//! Chroma passes through. The border ring keeps its value but is still clamped to the range.
//!
//! Numerics: the 3x3 sum is accumulated in `(ky, kx)` order with the crate's multiply-add
//! (`nn::fmadd`: fused on native targets, unfused on wasm32); everything else is the reference's
//! operation sequence. On native targets the result is bit-identical to PyTorch 1.10's on every
//! reference stream measured (`PORTING.md`).

use alloc::vec::Vec;

use archmage::prelude::*;

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};
use crate::nn::fast::{Engine, for_each_row};
use crate::nn::fmadd;
use crate::tensor::Tensor;

/// `LEF.recSharpThrList`: sigma-index thresholds per `model_id`. Not signalled.
pub const THRESHOLDS: [[i32; 3]; 4] = [
    [600, 1200, 2000],
    [800, 1400, 2400],
    [1000, 1600, 2800],
    [1200, 2000, 3200],
];
/// `LEF.recSharpMagList`: sharpening gain for the three threshold bands per `model_id`. Not
/// signalled.
pub const MAGNITUDES: [[f32; 3]; 4] = [
    [1.10, 1.13, 1.16],
    [1.07, 1.09, 1.11],
    [1.03, 1.04, 1.05],
    [1.01, 1.02, 1.03],
];

/// `[5, 24, 5] / 64`: corner/edge and centre taps (exact in binary).
const TAP_OUTER: f32 = 5.0 / 64.0;
const TAP_CENTRE: f32 = 24.0 / 64.0;

pub(super) fn apply(
    ctx: &FilterContext<'_>,
    channel: u8,
    mut state: FilterState,
) -> Result<FilterState> {
    sharpen(
        ctx.eng,
        &mut state.image.y,
        ctx.luma_scale_log,
        channel as usize,
        ctx.hdr.model_id as usize,
        ctx.hdr.bit_depth,
    )?;
    Ok(state)
}

/// `torch.nn.functional.interpolate(mode='nearest')` source index per destination index:
/// `min(floor(dst * (in / out)), in - 1)` with the scale and the product in `f32`.
fn nearest_indices(src: usize, dst: usize) -> Vec<usize> {
    let scale = src as f32 / dst as f32;
    (0..dst)
        .map(|i| (libm::floorf(i as f32 * scale) as usize).min(src - 1))
        .collect()
}

/// One interior row: `rows` are the normalised rows above, at and below it, `gains` the
/// sharpening gain per column.
#[autoversion]
fn sharpen_row(rows: [&[f32]; 3], gains: &[f32], range: f32, out: &mut [f32]) {
    let w = out.len();
    let [up, cur, down] = rows;
    let (up, cur, down, gains) = (&up[..w], &cur[..w], &down[..w], &gains[..w]);
    out[0] = cur[0].clamp(0.0, 1.0) * range;
    out[w - 1] = cur[w - 1].clamp(0.0, 1.0) * range;
    for i in 1..w - 1 {
        let mut acc = 0.0f32;
        for (row, centre) in [(up, TAP_OUTER), (cur, TAP_CENTRE), (down, TAP_OUTER)] {
            acc = fmadd(row[i - 1], TAP_OUTER, acc);
            acc = fmadd(row[i], centre, acc);
            acc = fmadd(row[i + 1], TAP_OUTER, acc);
        }
        let blur = acc.clamp(0.0, 1.0);
        let diff = (cur[i] - blur) * gains[i];
        out[i] = (blur + diff).clamp(0.0, 1.0) * range;
    }
}

/// Filter the luma plane in place. `scale_log` is the luma sigma-index map
/// (`decisions['model_y']['scale_log']`, RVS included), `channel` the signalled `LEF_chIdx`.
pub fn sharpen(
    eng: &Engine,
    luma: &mut Tensor<f32>,
    scale_log: &Tensor<i32>,
    channel: usize,
    model_id: usize,
    bit_depth: u8,
) -> Result<()> {
    // The reference indexes its four-row tables with the model index and slices the scale map
    // with the channel; both fail (or filter nothing sensible) out of range.
    let (thr, mag) = THRESHOLDS
        .get(model_id)
        .zip(MAGNITUDES.get(model_id))
        .ok_or(Error::InvalidData("LEF: model_id out of range"))?;
    if channel >= scale_log.c {
        return Err(Error::InvalidData("LEF: channel index out of range"));
    }
    let (h, w) = (luma.h, luma.w);
    if luma.c != 1 || h == 0 || w == 0 || scale_log.h == 0 || scale_log.w == 0 {
        return Err(Error::InvalidArgument("LEF: empty picture or scale map"));
    }
    if !(1..=16).contains(&bit_depth) {
        return Err(Error::InvalidArgument("LEF: bit depth"));
    }
    let range = ((1u32 << bit_depth) - 1) as f32;
    let region = scale_log.plane(channel);
    let (rw, row_of, col_of) = (
        scale_log.w,
        nearest_indices(scale_log.h, h),
        nearest_indices(scale_log.w, w),
    );
    let gain = |s: i32| -> f32 {
        if s >= thr[2] {
            mag[2]
        } else if s >= thr[1] {
            mag[1]
        } else if s >= thr[0] {
            mag[0]
        } else {
            1.0
        }
    };

    let mut x = Tensor::<f32>::zeros(1, h, w)?;
    for (d, &s) in x.data.iter_mut().zip(&luma.data) {
        *d = s / range;
    }
    let x = &x.data;
    for_each_row(eng, &mut luma.data, w, |y, out| {
        let cur = &x[y * w..][..w];
        if y == 0 || y + 1 == h || w < 3 {
            for (o, &v) in out.iter_mut().zip(cur) {
                *o = v.clamp(0.0, 1.0) * range;
            }
            return;
        }
        let (up, down) = (&x[(y - 1) * w..][..w], &x[(y + 1) * w..][..w]);
        let region = &region[row_of[y] * rw..][..rw];
        let gains: Vec<f32> = col_of.iter().map(|&c| gain(region[c])).collect();
        sharpen_row([up, cur, down], &gains, range, out);
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_matches_torch_for_the_test_picture() {
        // 888 rows from 56: torch maps row 887 to floor(887 * (56 / 888)) = 55.
        let idx = nearest_indices(56, 888);
        assert_eq!((idx[0], idx[15], idx[16], idx[887]), (0, 0, 1, 55));
        assert_eq!(nearest_indices(3, 3), [0, 1, 2]);
    }

    #[test]
    fn flat_picture_and_borders_are_kept() {
        let eng = Engine::with(crate::nn::fast::Tier::Scalar, false);
        let mut y = Tensor::from_vec(1, 4, 4, alloc::vec![128.0f32; 16]).unwrap();
        y.data[0] = 300.0; // border sample above the range: clamped, not filtered
        let s = Tensor::from_vec(1, 1, 1, alloc::vec![5000i32]).unwrap();
        sharpen(&eng, &mut y, &s, 0, 0, 8).unwrap();
        assert_eq!(y.data[0], 255.0);
        assert!((y.data[15] - 128.0).abs() < 1e-4);
        assert!(sharpen(&eng, &mut y, &s, 1, 0, 8).is_err());
        assert!(sharpen(&eng, &mut y, &s, 0, 4, 8).is_err());
    }
}
