//! `pytorch_msssim == 0.2.1`'s `ms_ssim` for one-plane `[0, 1]` pictures — the metric the
//! eICCI model search scores candidates with (`icci_filter.py::calculate_msssim`, which calls
//! it per channel at `data_range = 1` and reports `1 - ms_ssim`).
//!
//! The port keeps the reference's elementwise arithmetic in `f32` (same op order — no FMA
//! contraction) and accumulates every reduction (the 11-tap Gaussian convolutions, the
//! `mean(-1)` over each SSIM/CS map, the 2x2 average pools) in `f64`, rounding once to `f32`.
//! That is a fixed summation order, so the result is bit-identical on every engine, tier and
//! thread count; the remaining difference to PyTorch's blocked `f32` reductions is measured
//! against a reference dump (`tests/encode_ref.rs`, `PORTING.md`).
//!
//! Reference details preserved: the `win_size = 11` / `win_sigma = 1.5` Gaussian kernel,
//! `K = (0.01, 0.03)` at `data_range = 1` (`C1 = 1e-4`, `C2 = 9e-4`), the five level weights
//! `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`, `relu` on the four contrast/structure means and
//! the last SSIM mean, `avg_pool2d(kernel = 2, stride = 2, padding = [h % 2, w % 2])` between
//! levels (zero pad applied *before* the first row/column, divisor always 4), the per-level
//! product `prod(cs_i ** w_i) * ssim_5 ** w_5`, and `gaussian_filter` skipping a spatial dim
//! below the window size.

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::nn::fast::{Engine, for_each_row};
use crate::tensor::Tensor;

/// `win_size`.
const WIN: usize = 11;
/// The five `weights` of `ms_ssim`.
const WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
/// `(K1 * data_range)^2` and `(K2 * data_range)^2` at `data_range = 1`.
const C1: f32 = 0.0001;
const C2: f32 = 0.0009;

/// `_fspecial_gauss_1d(11, 1.5)`: `exp(-c^2 / 4.5)` over `c = -5..=5`, normalised. The
/// reference computes it in `f32`; computing in `f64` and rounding once is the deterministic
/// choice (taps differ from PyTorch's by at most a few `f32` ulp).
fn gaussian_window() -> [f32; WIN] {
    let mut g = [0.0f64; WIN];
    let mut sum = 0.0f64;
    for (i, v) in g.iter_mut().enumerate() {
        let c = i as f64 - (WIN / 2) as f64;
        *v = libm::exp(-(c * c) / (2.0 * 1.5 * 1.5));
        sum += *v;
    }
    core::array::from_fn(|i| (g[i] / sum) as f32)
}

/// Valid (unpadded) 11-tap convolution down the rows: `h - 10` by `w`.
fn conv_rows(eng: &Engine, src: &[f32], h: usize, w: usize, win: &[f32; WIN]) -> Vec<f32> {
    let oh = h - (WIN - 1);
    let mut out = alloc::vec![0.0f32; oh * w];
    for_each_row(eng, &mut out, w, |y, row| {
        for (x, d) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &tap) in win.iter().enumerate() {
                acc += tap as f64 * src[(y + k) * w + x] as f64;
            }
            *d = acc as f32;
        }
    });
    out
}

/// Valid 11-tap convolution along the columns: `h` by `w - 10`.
fn conv_cols(eng: &Engine, src: &[f32], h: usize, w: usize, win: &[f32; WIN]) -> Vec<f32> {
    let ow = w - (WIN - 1);
    let mut out = alloc::vec![0.0f32; h * ow];
    for_each_row(eng, &mut out, ow, |y, row| {
        let src = &src[y * w..][..w];
        for (x, d) in row.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (k, &tap) in win.iter().enumerate() {
                acc += tap as f64 * src[x + k] as f64;
            }
            *d = acc as f32;
        }
    });
    out
}

/// `gaussian_filter`: the vertical pass first, then the horizontal one; a dimension below the
/// window size is skipped (the reference warns and leaves it untouched).
fn gaussian_filter(
    eng: &Engine,
    src: &[f32],
    mut h: usize,
    mut w: usize,
    win: &[f32; WIN],
) -> (Vec<f32>, usize, usize) {
    let mut buf = src.to_vec();
    if h >= WIN {
        buf = conv_rows(eng, &buf, h, w, win);
        h -= WIN - 1;
    }
    if w >= WIN {
        buf = conv_cols(eng, &buf, h, w, win);
        w -= WIN - 1;
    }
    (buf, h, w)
}

/// `avg_pool2d(x, kernel_size = 2, stride = 2, padding = [h % 2, w % 2])`: zero pad of `p`
/// samples on *every* side (the pad rows at the bottom/right are never inside a window),
/// each output the sum of its 2x2 window over 4 (`count_include_pad` is on). For odd `h` /
/// `w` the first row / column is therefore averaged at half weight.
fn avg_pool2(src: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h % 2, w % 2);
    let (oh, ow) = ((h + 2 * ph - 2) / 2 + 1, (w + 2 * pw - 2) / 2 + 1);
    let mut out = alloc::vec![0.0f32; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = 0.0f64;
            for dy in 0..2 {
                let Some(sy) = (2 * oy + dy).checked_sub(ph).filter(|&y| y < h) else {
                    continue;
                };
                for dx in 0..2 {
                    if let Some(sx) = (2 * ox + dx).checked_sub(pw).filter(|&x| x < w) {
                        acc += src[sy * w + sx] as f64;
                    }
                }
            }
            out[oy * ow + ox] = (acc * 0.25) as f32;
        }
    }
    (out, oh, ow)
}

/// `_ssim` on one channel: the per-channel SSIM and CS means over the `h - 10` by `w - 10`
/// valid-contraction map.
fn ssim(
    eng: &Engine,
    x: &[f32],
    y: &[f32],
    h: usize,
    w: usize,
    win: &[f32; WIN],
) -> Result<(f32, f32)> {
    // The unfiltered products live at the input's dims — keep them before the first filter
    // call shrinks `h`/`w` to the valid-contraction size.
    let (ih, iw) = (h, w);
    let (mu1, h, w) = gaussian_filter(eng, x, ih, iw, win);
    let (mu2, ..) = gaussian_filter(eng, y, ih, iw, win);
    let sq = |p: &[f32]| p.iter().map(|&v| v * v).collect::<Vec<f32>>();
    let prod: Vec<f32> = x.iter().zip(y).map(|(&a, &b)| a * b).collect();
    let (vx, ..) = gaussian_filter(eng, &sq(x), ih, iw, win);
    let (vy, ..) = gaussian_filter(eng, &sq(y), ih, iw, win);
    let (cxy, ..) = gaussian_filter(eng, &prod, ih, iw, win);
    let n = h * w;
    if n == 0 {
        return Err(Error::InvalidArgument("MS-SSIM: empty plane"));
    }
    let (mut ssum, mut csum) = (0.0f64, 0.0f64);
    for i in 0..n {
        let (m1, m2) = (mu1[i], mu2[i]);
        let (m1sq, m2sq, m1m2) = (m1 * m1, m2 * m2, m1 * m2);
        let (s1, s2, s12) = (vx[i] - m1sq, vy[i] - m2sq, cxy[i] - m1m2);
        // `cs_map` and `ssim_map`, elementwise and in the reference's op order.
        let cs = (2.0 * s12 + C2) / (s1 + s2 + C2);
        let sm = ((2.0 * m1m2 + C1) / (m1sq + m2sq + C1)) * cs;
        ssum += sm as f64;
        csum += cs as f64;
    }
    let n = n as f64;
    Ok(((ssum / n) as f32, (csum / n) as f32))
}

/// `torch.relu` elementwise (NaN passes through, `-0.0` keeps its sign — either way the later
/// `powf` sees a non-positive input exactly like PyTorch).
fn relu(v: f32) -> f32 {
    if v < 0.0 { 0.0 } else { v }
}

/// `ms_ssim(x, y, data_range = 1, size_average = True)` on a one-channel picture — which is
/// also `ms_ssim_val.mean(1)` for `size_average = False`, the only difference being a mean
/// over a single channel.
///
/// Fails when the smaller side is at most `(WIN - 1) * 2^4 = 160`, the reference's own assert
/// (the eICCI tile layout guarantees `MINIMUM_TILE_SIZE = 176`).
pub fn ms_ssim(eng: &Engine, x: &Tensor<f32>, y: &Tensor<f32>) -> Result<f32> {
    if x.c != 1 || y.c != 1 || (x.h, x.w) != (y.h, y.w) {
        return Err(Error::InvalidArgument(
            "MS-SSIM: one-channel planes of equal size",
        ));
    }
    let (mut h, mut w) = (x.h, x.w);
    if h.min(w) <= (WIN - 1) * 16 {
        return Err(Error::InvalidArgument(
            "MS-SSIM: side below the 5-level minimum of 161",
        ));
    }
    let win = gaussian_window();
    let (mut xb, mut yb) = (x.data.clone(), y.data.clone());
    let mut mcs = [0.0f32; 4];
    let mut ssim_last = 0.0f32;
    for i in 0..WEIGHTS.len() {
        let (spc, cs) = ssim(eng, &xb, &yb, h, w, &win)?;
        if i + 1 == WEIGHTS.len() {
            ssim_last = relu(spc);
        } else {
            mcs[i] = relu(cs);
            let (px, nh, nw) = avg_pool2(&xb, h, w);
            let (py, ..) = avg_pool2(&yb, h, w);
            (xb, yb, h, w) = (px, py, nh, nw);
        }
    }
    // `torch.prod(mcs_and_ssim ** weights)`: pow per level, then the product in level order.
    let mut ms = 1.0f32;
    for (i, &v) in mcs.iter().enumerate() {
        ms *= v.powf(WEIGHTS[i]);
    }
    Ok(ms * ssim_last.powf(WEIGHTS[4]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(h: usize, w: usize, f: impl Fn(usize, usize) -> f32) -> Tensor<f32> {
        let data = (0..h * w).map(|i| f(i / w, i % w)).collect();
        Tensor::from_vec(1, h, w, data).unwrap()
    }

    #[test]
    fn kernel_is_the_reference_window() {
        // `_fspecial_gauss_1d(11, 1.5)` as the installed pytorch_msssim computes it in f32.
        let win = gaussian_window();
        let want = [
            0.0010283804,
            0.007_598_758,
            0.036000773,
            0.10936069,
            0.21300553,
            0.26601172,
            0.21300553,
            0.10936069,
            0.036000773,
            0.007_598_758,
            0.0010283804,
        ];
        for (g, w) in win.iter().zip(want) {
            assert!((g - w).abs() < 2e-8, "{g} vs {w}");
        }
        assert!((win.iter().map(|&v| v as f64).sum::<f64>() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn identical_planes_score_one() {
        let eng = Engine::new();
        let a = plane(200, 224, |y, x| (((x * 7 + y * 13) % 251) as f32) / 255.0);
        assert_eq!(ms_ssim(&eng, &a, &a).unwrap(), 1.0);
    }

    /// `avg_pool2d(2, 2, padding = [h % 2, w % 2])` on an odd-sized plane: the first row and
    /// column carry half weight, the divisor stays 4.
    #[test]
    fn odd_size_pool_pads_at_the_start() {
        let (out, oh, ow) = avg_pool2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        assert_eq!((oh, ow), (2, 2));
        // (0,0): rows -1,0 x cols -1,0 -> only (0,0)=1 -> 1/4; (0,1): row 0, cols 1,2 -> 5/4;
        // (1,0): rows 1,2 col 0 -> (4+7)/4; (1,1): rows 1,2 cols 1,2 -> (5+6+8+9)/4.
        assert_eq!(out, [0.25, 1.25, 2.75, 7.0]);
    }

    #[test]
    fn small_planes_are_rejected() {
        let eng = Engine::new();
        let a = plane(160, 200, |_, _| 0.5);
        assert!(ms_ssim(&eng, &a, &a).is_err());
        let a = plane(161, 161, |_, _| 0.5);
        assert_eq!(ms_ssim(&eng, &a, &a).unwrap(), 1.0);
    }
}
