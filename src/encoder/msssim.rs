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

/// `avg_pool2d(x, kernel_size = 2, stride = 2, padding = [h % 2, w % 2])` —
/// non-overlapping 2x2 windows with the input's odd side zero-padded at the
/// top / left (pytorch `avg_pool2d` pads `padding` on both sides, but the
/// trailing pad lies beyond the last window, so the effective offset is the
/// leading one). The divisor stays 4; an odd side yields ceil dims.
fn avg_pool2(src: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h % 2, w % 2);
    let (oh, ow) = ((h + 1) / 2, (w + 1) / 2);
    let mut out = alloc::vec![0.0f32; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = 0.0f64;
            for dy in 0..2 {
                for dx in 0..2 {
                    if let (Some(sy), Some(sx)) = (
                        (2 * oy + dy).checked_sub(ph).filter(|&v| v < h),
                        (2 * ox + dx).checked_sub(pw).filter(|&v| v < w),
                    ) {
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

// --- VJP (input gradient, RDLR) --------------------------------------------

/// `conv_rows`'s / `conv_cols`'s input gradient for the separable Gaussian:
/// the forward is `out[j] = Σ_k win[k]·in[j+k]` (valid, `n → n-10`), so the
/// backward scatters `d_in[j+k] += win[k]·d_out[j]`.
fn conv_rows_backward(
    d: &[f32],
    h: usize,
    w: usize,
    win: &[f32; WIN],
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    // d is [oh][ow] (the output grad); result is [h][w] (the input grad).
    let _ = (h, w);
    let mut din = vec![0.0f32; (oh + WIN - 1) * ow];
    for y in 0..oh {
        for x in 0..ow {
            let g = d[y * ow + x];
            if g == 0.0 {
                continue;
            }
            for k in 0..WIN {
                din[(y + k) * ow + x] += win[k] * g;
            }
        }
    }
    din
}

fn conv_cols_backward(d: &[f32], oh: usize, ow: usize, win: &[f32; WIN]) -> Vec<f32> {
    let mut din = vec![0.0f32; oh * (ow + WIN - 1)];
    for y in 0..oh {
        for x in 0..ow {
            let g = d[y * ow + x];
            if g == 0.0 {
                continue;
            }
            for k in 0..WIN {
                din[y * (ow + WIN - 1) + x + k] += win[k] * g;
            }
        }
    }
    din
}

/// `gaussian_filter`'s input gradient: the output grad at `h-10 × w-10` → the
/// input grad at `h × w`. `gaussian_filter = conv_cols∘conv_rows` (valid), so
/// the backward is `conv_rows_backward∘conv_cols_backward`.
fn gaussian_backward(d: &[f32], h: usize, w: usize, win: &[f32; WIN]) -> Vec<f32> {
    let d2 = conv_cols_backward(d, h - (WIN - 1), w - (WIN - 1), win);
    conv_rows_backward(&d2, 0, 0, win, h - (WIN - 1), w)
}

/// `avg_pool2`'s input gradient: each 2x2 window's output grad spreads over
/// its four samples at the forward's `h % 2` / `w % 2` leading offset.
fn avg_pool2_backward(d: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (ph, pw) = (h % 2, w % 2);
    let (oh, ow) = ((h + 1) / 2, (w + 1) / 2);
    let mut din = vec![0.0f32; h * w];
    for y in 0..oh {
        for x in 0..ow {
            let g = d[y * ow + x] / 4.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    if let (Some(sy), Some(sx)) = (
                        (2 * y + dy).checked_sub(ph).filter(|&v| v < h),
                        (2 * x + dx).checked_sub(pw).filter(|&v| v < w),
                    ) {
                        din[sy * w + sx] += g;
                    }
                }
            }
        }
    }
    din
}

/// Per-level forward intermediates for `ms_ssim_vjp`.
struct SsimLevel {
    x: Vec<f32>,
    y: Vec<f32>,
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    cxy: Vec<f32>,
    spc: f32,
    cs: f32,
    h: usize,
    w: usize,
}

/// `ms_ssim`'s input gradient (RDLR VJP): given `d` = `dL/d(msssim)`, the
/// gradients on `x` and `y` at the input's resolution.
pub fn ms_ssim_vjp(
    eng: &Engine,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    d: f32,
) -> Result<(Tensor<f32>, Tensor<f32>)> {
    let win = gaussian_window();
    // Forward: per-level intermediates.
    let (mut h, mut w) = (x.h, x.w);
    let (mut xb, mut yb) = (x.data.clone(), y.data.clone());
    let mut levels: Vec<SsimLevel> = Vec::with_capacity(WEIGHTS.len());
    for i in 0..WEIGHTS.len() {
        let (ih, iw) = (h, w);
        let sq = |p: &[f32]| p.iter().map(|&v| v * v).collect::<Vec<f32>>();
        let prod: Vec<f32> = xb.iter().zip(&yb).map(|(&a, &b)| a * b).collect();
        let (mu1, h2, w2) = gaussian_filter(eng, &xb, ih, iw, &win);
        let (mu2, ..) = gaussian_filter(eng, &yb, ih, iw, &win);
        let (vx, ..) = gaussian_filter(eng, &sq(&xb), ih, iw, &win);
        let (vy, ..) = gaussian_filter(eng, &sq(&yb), ih, iw, &win);
        let (cxy, ..) = gaussian_filter(eng, &prod, ih, iw, &win);
        let n = h2 * w2;
        let (mut ssum, mut csum) = (0.0f64, 0.0f64);
        for k in 0..n {
            let (m1, m2) = (mu1[k], mu2[k]);
            let (s1, s2, s12) = (vx[k] - m1 * m1, vy[k] - m2 * m2, cxy[k] - m1 * m2);
            let cs = (2.0 * s12 + C2) / (s1 + s2 + C2);
            let sm = ((2.0 * m1 * m2 + C1) / (m1 * m1 + m2 * m2 + C1)) * cs;
            ssum += sm as f64;
            csum += cs as f64;
        }
        let n64 = n as f64;
        let spc = (ssum / n64) as f32;
        let cs = (csum / n64) as f32;
        levels.push(SsimLevel {
            x: xb,
            y: yb,
            mu1,
            mu2,
            vx,
            vy,
            cxy,
            spc,
            cs,
            h,
            w,
        });
        if i + 1 == WEIGHTS.len() {
            break;
        }
        let (px, nh, nw) = avg_pool2(&levels[i].x, h, w);
        let (py, ..) = avg_pool2(&levels[i].y, h, w);
        (xb, yb, h, w) = (px, py, nh, nw);
    }
    // Top: ms = Π relu(cs_i)^w_i · relu(ssim_5)^w_4 — scalar grads per level.
    let mut mcs = [0.0f32; 4];
    for i in 0..4 {
        mcs[i] = relu(levels[i].cs);
    }
    let ssim_last = relu(levels[4].spc);
    let mut ms = 1.0f32;
    for (i, &v) in mcs.iter().enumerate() {
        ms *= v.powf(WEIGHTS[i]);
    }
    ms *= ssim_last.powf(WEIGHTS[4]);
    // d_spc / d_cs scalars per level (relu backward gates on the mean's sign).
    let mut d_spc = [0.0f32; 5];
    let mut d_cs = [0.0f32; 5];
    for i in 0..4 {
        if levels[i].cs > 0.0 {
            d_cs[i] = d * ms * WEIGHTS[i] / levels[i].cs;
        }
    }
    if levels[4].spc > 0.0 {
        d_spc[4] = d * ms * WEIGHTS[4] / levels[4].spc;
    }
    // Walk levels deepest→shallowest: level i's own SSIM backprop plus the
    // deeper levels' contribution through `x_{i+1} = avg_pool2(x_i)`.
    let (mut carry_x, mut carry_y) = (Vec::<f32>::new(), Vec::<f32>::new());
    let (mut d_x, mut d_y) = (Vec::<f32>::new(), Vec::<f32>::new());
    for (i, lv) in levels.iter().enumerate().rev() {
        let (gh, gw) = (lv.h - (WIN - 1), lv.w - (WIN - 1));
        let n = (gh * gw) as f64;
        let mut dm1 = vec![0.0f32; gh * gw];
        let mut dm2 = vec![0.0f32; gh * gw];
        let mut dvx = vec![0.0f32; gh * gw];
        let mut dvy = vec![0.0f32; gh * gw];
        let mut dcxy = vec![0.0f32; gh * gw];
        for k in 0..gh * gw {
            let (m1, m2) = (lv.mu1[k], lv.mu2[k]);
            let (m1sq, m2sq, m1m2) = (m1 * m1, m2 * m2, m1 * m2);
            let (s1, s2, s12) = (lv.vx[k] - m1sq, lv.vy[k] - m2sq, lv.cxy[k] - m1m2);
            let cs = (2.0 * s12 + C2) / (s1 + s2 + C2);
            let core = (2.0 * m1m2 + C1) / (m1sq + m2sq + C1);
            // d(sm) = d_spc/n; d(cs) = d_cs/n + d(sm)·core.
            let d_sm = (d_spc[i] as f64 / n) as f32;
            let d_cs_k = (d_cs[i] as f64 / n) as f32 + d_sm * core;
            let d_core = d_sm * cs;
            let den1 = m1sq + m2sq + C1;
            let num1 = 2.0 * m1m2 + C1;
            let d_num1 = d_core / den1;
            let d_den1 = -d_core * num1 / (den1 * den1);
            dm1[k] += d_num1 * 2.0 * m2 + d_den1 * 2.0 * m1;
            dm2[k] += d_num1 * 2.0 * m1 + d_den1 * 2.0 * m2;
            let den2 = s1 + s2 + C2;
            let num2 = 2.0 * s12 + C2;
            let d_num2 = d_cs_k / den2;
            let d_den2 = -d_cs_k * num2 / (den2 * den2);
            dcxy[k] += 2.0 * d_num2;
            dvx[k] += d_den2;
            dvy[k] += d_den2;
            dm1[k] -= d_den2 * 2.0 * m1 + 2.0 * d_num2 * m2;
            dm2[k] -= d_den2 * 2.0 * m2 + 2.0 * d_num2 * m1;
        }
        // The five gaussian convolutions' backwards → level-input grads.
        let g = |p: &Vec<f32>| gaussian_backward(p, lv.h, lv.w, &win);
        let (gxm, gym) = (g(&dm1), g(&dm2));
        let (gxv, gyv, gxyv) = (g(&dvx), g(&dvy), g(&dcxy));
        let mut dx = vec![0.0f32; lv.h * lv.w];
        let mut dy = vec![0.0f32; lv.h * lv.w];
        for k in 0..lv.h * lv.w {
            dx[k] = gxm[k] + 2.0 * lv.x[k] * gxv[k] + lv.y[k] * gxyv[k];
            dy[k] = gym[k] + 2.0 * lv.y[k] * gyv[k] + lv.x[k] * gxyv[k];
        }
        // Add the deeper levels' contribution (level i+1's grad, scattered up).
        if !carry_x.is_empty() {
            for k in 0..dx.len() {
                dx[k] += carry_x[k];
                dy[k] += carry_y[k];
            }
        }
        if i > 0 {
            // `x_i = avg_pool2(x_{i-1})` at level i-1's dims — scatter down.
            let (ph, pw) = (levels[i - 1].h, levels[i - 1].w);
            carry_x = avg_pool2_backward(&dx, ph, pw);
            carry_y = avg_pool2_backward(&dy, ph, pw);
        } else {
            d_x = dx;
            d_y = dy;
        }
    }
    Ok((
        Tensor::from_vec(1, x.h, x.w, d_x)?,
        Tensor::from_vec(1, y.h, y.w, d_y)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(h: usize, w: usize, f: impl Fn(usize, usize) -> f32) -> Tensor<f32> {
        let data = (0..h * w).map(|i| f(i / w, i % w)).collect();
        Tensor::from_vec(1, h, w, data).unwrap()
    }

    /// `ms_ssim_vjp`'s input gradient against a directional finite difference of
    /// `ms_ssim` — `⟨vjp, v⟩` vs `(f(x+εv) − f(x−εv))/(2ε)` for a random `v`. The
    /// inner product tests the whole field at once and averages out the f32
    /// output quantization (~1e-7 ulp on a ~1.0 score) that swamps per-position
    /// differences at small gradient magnitudes.
    #[test]
    fn vjp_matches_finite_differences() {
        let eng = Engine::new();
        let (h, w) = (200, 200);
        let x = plane(h, w, |y, x| ((x * 37 + y * 91) % 251) as f32 / 251.0);
        let y0 = plane(h, w, |y, x| ((x * 53 + y * 17 + 11) % 241) as f32 / 241.0);
        let (dx, dy) = ms_ssim_vjp(&eng, &x, &y0, 1.0).unwrap();
        let mut v = x.clone();
        for (i, p) in v.data.iter_mut().enumerate() {
            *p = ((i * 2_654_435_761usize) % 997) as f32 / 997.0 - 0.5;
        }
        let lhs: f64 = dx
            .data
            .iter()
            .zip(&v.data)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let e = 1e-2f32;
        let mut xp = x.clone();
        let mut xm = x.clone();
        for ((p, m), &b) in xp.data.iter_mut().zip(&mut xm.data).zip(&v.data) {
            *p += e * b;
            *m -= e * b;
        }
        let rhs = (ms_ssim(&eng, &xp, &y0).unwrap() - ms_ssim(&eng, &xm, &y0).unwrap()) / (2.0 * e);
        eprintln!(
            "x-dir: vjp {lhs:.6e}  fd {rhs:.6e}  rel {:.3e}",
            (lhs - rhs as f64).abs() / (rhs as f64).abs()
        );
        assert!((lhs - rhs as f64).abs() < 0.02 * (rhs as f64).abs());
        // And the y (reconstruction-side) gradient the same way.
        let lhs: f64 = dy
            .data
            .iter()
            .zip(&v.data)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let mut yp = y0.clone();
        let mut ym = y0.clone();
        for ((p, m), &b) in yp.data.iter_mut().zip(&mut ym.data).zip(&v.data) {
            *p += e * b;
            *m -= e * b;
        }
        let rhs = (ms_ssim(&eng, &x, &yp).unwrap() - ms_ssim(&eng, &x, &ym).unwrap()) / (2.0 * e);
        eprintln!(
            "y-dir: vjp {lhs:.6e}  fd {rhs:.6e}  rel {:.3e}",
            (lhs - rhs as f64).abs() / (rhs as f64).abs()
        );
        assert!((lhs - rhs as f64).abs() < 0.02 * (rhs as f64).abs());
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

    /// `avg_pool2d(x, 2, padding = [h % 2, w % 2])` on an odd-sized plane: the
    /// leading pad gives (0,0) only input (0,0), the divisor stays 4.
    #[test]
    fn odd_size_pool_pads_at_the_start() {
        let (out, oh, ow) = avg_pool2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        assert_eq!((oh, ow), (2, 2));
        // (0,0): rows -1,0 x cols -1,0 -> only (0,0)=1 -> 1/4; (0,1): row 0, cols 1,2 -> 5/4;
        // (1,0): rows 1,2 col 0 -> (4+7)/4; (1,1): rows 1,2 cols 1,2 -> (5+6+8+9)/4.
        assert_eq!(out, [0.25, 1.25, 2.75, 7.0]);
    }

    /// `avg_pool2_backward` must mirror the forward's odd-side pad offset:
    /// on a 3x3 input, output (0,0) covers only input (0,0), so d 4.0 -> 1.0
    /// there and nothing elsewhere in its window.
    #[test]
    fn odd_size_pool_backward_shifts_windows() {
        let d = avg_pool2_backward(&[4.0, 0.0, 0.0, 0.0], 3, 3);
        assert_eq!(d, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // Output (1,1) covers inputs (1,1),(1,2),(2,1),(2,2) at quarter weight.
        let d = avg_pool2_backward(&[0.0, 0.0, 0.0, 8.0], 3, 3);
        assert_eq!(d, [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0]);
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
