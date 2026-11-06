//! Pointwise and reduction operations of the HOP attention blocks: `exp`, sigmoid, ELU gate,
//! layer norm over channels, L2 normalisation, channel attention.
//!
//! Numeric contract (what makes every tier and thread count agree bit for bit):
//!
//! - `exp` is one fixed algorithm (Cephes `expf`: range reduction by `ln 2`, degree-5 polynomial)
//!   written as straight IEEE operations, never a platform `exp`. It is applied per element, so
//!   vectorising it changes nothing.
//! - Sums over many elements are accumulated in `f64` from `f32` inputs. A product of two `f32`s
//!   is exact in `f64`, so only the additions round, and their order is fixed: either one
//!   accumulator per output element in index order (vectorised across output elements), or, for
//!   long dot products, eight interleaved accumulators (`index mod 8`) combined pairwise. Eight is
//!   a property of this file, not of the CPU.
//!
//! Against PyTorch these are bounded differences like the convolutions' (`PORTING.md`).

use alloc::vec::Vec;

use archmage::prelude::*;

use magetypes::simd::backends::{F32x8Convert, F32x16Convert};
use magetypes::simd::generic::{f32x8, f32x16, i32x8, i32x16};

use super::{Engine, Tier, for_each_row};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

const LOG2E: f32 = core::f32::consts::LOG2_E;
// Cephes' published constants, digits kept as printed there.
#[allow(clippy::excessive_precision)]
const LN2_HI: f32 = 0.693_359_375;
#[allow(clippy::excessive_precision)]
const LN2_LO: f32 = -2.121_944_4e-4;
#[allow(clippy::excessive_precision)]
const P: [f32; 6] = [
    1.987_569_15e-4,
    1.398_199_950_7e-3,
    8.333_451_907_3e-3,
    4.166_579_589_4e-2,
    1.666_666_545_9e-1,
    5.000_000_120_1e-1,
];

/// `e^x` for `x` clamped to `[-87, 88]` (so the result is a normal float; `exp(-87) ~ 1.6e-38`
/// stands in for anything smaller).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * LOG2E + 0.5).floor();
    let r = x - n * LN2_HI;
    let r = r - n * LN2_LO;
    let mut p = P[0];
    p = p * r + P[1];
    p = p * r + P[2];
    p = p * r + P[3];
    p = p * r + P[4];
    p = p * r + P[5];
    let y = p * (r * r) + r + 1.0;
    // 2^n, n in [-126, 127].
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    y * scale
}

// Scalar `sigmoid`/`elu_gate` per element — the Scalar-tier impl and the
// sub-lane tail of the `f32x` slice kernels below. Same `exp` per element
// so the tail is bit-identical to the vectorised lanes.
fn sigmoid_slice(data: &mut [f32]) {
    for v in data {
        *v = 1.0 / (1.0 + exp(-*v));
    }
}

/// `elu(a) * b` into `a`: `a > 0 ? a : exp(a) - 1`, times `b`.
fn elu_gate_slice(a: &mut [f32], b: &[f32]) {
    for (x, &g) in a.iter_mut().zip(b) {
        let e = exp(*x) - 1.0;
        let elu = if *x > 0.0 { *x } else { e };
        *x = elu * g;
    }
}

/// Lane-vectorised `sigmoid`/`elu_gate` kernels. Each is bit-identical to the
/// scalar [`exp`] / [`sigmoid_slice`] / [`elu_gate_slice`] on every element:
/// the Cephes `exp` is straight IEEE ops in the same order — *unfused* `*`/`+`
/// (Rust does not auto-contract `a*b+c` to FMA), `floor`, `to_i32`, an integer
/// `2^n` scale and `blend` selects — so every lane is per-lane deterministic.
/// The sub-lane tail falls back to the scalar slices, which are bit-identical
/// for the same reason. Modelled on `zenmetrics-math`'s `expf_libm_lanes!`.
macro_rules! exp_lanes {
    ($cexp:ident, $sig:ident, $elu:ident, $sexp:ident, $elub:ident, $F:ident, $I:ident, $conv:ident) => {
        /// Cephes `exp` on one `$F` lane vector, bit-identical to [`exp`].
        #[inline(always)]
        fn $cexp<T: $conv>(t: T, x: $F<T>) -> $F<T> {
            let s = $F::<T>::splat_t;
            let x = x.max(s(t, -87.0)).min(s(t, 88.0));
            let n = (x * s(t, LOG2E) + s(t, 0.5)).floor();
            let r = x - n * s(t, LN2_HI);
            let r = r - n * s(t, LN2_LO);
            let mut p = s(t, P[0]);
            p = p * r + s(t, P[1]);
            p = p * r + s(t, P[2]);
            p = p * r + s(t, P[3]);
            p = p * r + s(t, P[4]);
            p = p * r + s(t, P[5]);
            let y = p * (r * r) + r + s(t, 1.0);
            // `2^n`, `n` integral in [-126,127]: `(n + 127) << 23`, reinterpreted.
            let n_i = n.to_i32();
            let scale = (n_i + $I::<T>::splat_t(t, 127))
                .shl_const::<23>()
                .bitcast_to_f32();
            y * scale
        }

        /// `data[i] = 1 / (1 + exp(-data[i]))` — lane chunks + scalar tail.
        #[inline(always)]
        fn $sig<T: $conv>(t: T, data: &mut [f32]) {
            let s = $F::<T>::splat_t;
            let (chunks, tail) = $F::<T>::partition_slice_mut_t(t, data);
            for c in chunks.iter_mut() {
                let x = $F::<T>::load_t(t, c);
                let e = $cexp(t, s(t, 0.0) - x);
                (s(t, 1.0) / (s(t, 1.0) + e)).store(c);
            }
            sigmoid_slice(tail);
        }

        /// `a[i] = (a[i] > 0 ? a[i] : exp(a[i]) - 1) * b[i]` — lanes + scalar tail.
        #[inline(always)]
        fn $elu<T: $conv>(t: T, a: &mut [f32], b: &[f32]) {
            let s = $F::<T>::splat_t;
            let (ac, at) = $F::<T>::partition_slice_mut_t(t, a);
            let (bc, bt) = $F::<T>::partition_slice_t(t, b);
            for (x, g) in ac.iter_mut().zip(bc.iter()) {
                let xv = $F::<T>::load_t(t, x);
                let e = $cexp(t, xv) - s(t, 1.0);
                let elu = $F::<T>::blend(xv.simd_gt(s(t, 0.0)), xv, e);
                (elu * $F::<T>::load_t(t, g)).store(x);
            }
            elu_gate_slice(at, bt);
        }

        /// `data[i] = exp(data[i] - m)` — lane chunks + scalar tail. The softmax
        /// numerator; the `sum` stays a fixed-order scalar `f64` accumulate.
        #[inline(always)]
        fn $sexp<T: $conv>(t: T, data: &mut [f32], m: f32) {
            let sm = $F::<T>::splat_t(t, m);
            let (chunks, tail) = $F::<T>::partition_slice_mut_t(t, data);
            for c in chunks.iter_mut() {
                let x = $F::<T>::load_t(t, c);
                $cexp(t, x - sm).store(c);
            }
            for v in tail {
                *v = exp(*v - m);
            }
        }

        /// `elu_gate` backward: `d_a = elu'(a)·b·d`, `d_b = elu(a)·d` — one
        /// `exp` lane call yields both `elu`/`delu`. Lanes + scalar tail.
        #[inline(always)]
        fn $elub<T: $conv>(
            t: T,
            d: &[f32],
            a: &[f32],
            b: &[f32],
            d_a: &mut [f32],
            d_b: &mut [f32],
        ) {
            let s = $F::<T>::splat_t;
            let (dac, dat) = $F::<T>::partition_slice_mut_t(t, d_a);
            let (dbc, dbt) = $F::<T>::partition_slice_mut_t(t, d_b);
            let (ac, at) = $F::<T>::partition_slice_t(t, a);
            let (bc, bt) = $F::<T>::partition_slice_t(t, b);
            let (dc, dt) = $F::<T>::partition_slice_t(t, d);
            for (((da, db), (av, bv)), o) in dac
                .iter_mut()
                .zip(dbc.iter_mut())
                .zip(ac.iter().zip(bc.iter()))
                .zip(dc.iter())
            {
                let xv = $F::<T>::load_t(t, av);
                let e = $cexp(t, xv);
                let pos = xv.simd_gt(s(t, 0.0));
                let elu = $F::<T>::blend(pos, xv, e - s(t, 1.0));
                let delu = $F::<T>::blend(pos, s(t, 1.0), e);
                let ov = $F::<T>::load_t(t, o);
                (delu * $F::<T>::load_t(t, bv) * ov).store(da);
                (elu * ov).store(db);
            }
            for (((da, db), (&av, &bv)), &o) in dat
                .iter_mut()
                .zip(dbt.iter_mut())
                .zip(at.iter().zip(bt.iter()))
                .zip(dt.iter())
            {
                let elu = if av > 0.0 { av } else { exp(av) - 1.0 };
                let delu = if av > 0.0 { 1.0 } else { exp(av) };
                *da = delu * bv * o;
                *db = elu * o;
            }
        }
    };
}

exp_lanes!(
    cephes_exp8,
    sigmoid8,
    elu8,
    softmax_exp8,
    elu_bwd8,
    f32x8,
    i32x8,
    F32x8Convert
);
exp_lanes!(
    cephes_exp16,
    sigmoid16,
    elu16,
    softmax_exp16,
    elu_bwd16,
    f32x16,
    i32x16,
    F32x16Convert
);

#[cfg(feature = "avx512")]
#[arcane]
fn sigmoid_v4(t: X64V4Token, data: &mut [f32]) {
    sigmoid16::<X64V4Token>(t, data)
}
#[arcane]
fn sigmoid_v3(t: X64V3Token, data: &mut [f32]) {
    sigmoid8::<X64V3Token>(t, data)
}
#[arcane]
fn sigmoid_neon(t: NeonToken, data: &mut [f32]) {
    sigmoid8::<NeonToken>(t, data)
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn sigmoid_wasm(t: Wasm128Token, data: &mut [f32]) {
    sigmoid8::<Wasm128Token>(t, data)
}

#[cfg(feature = "avx512")]
#[arcane]
fn elu_v4(t: X64V4Token, a: &mut [f32], b: &[f32]) {
    elu16::<X64V4Token>(t, a, b)
}
#[arcane]
fn elu_v3(t: X64V3Token, a: &mut [f32], b: &[f32]) {
    elu8::<X64V3Token>(t, a, b)
}
#[arcane]
fn elu_neon(t: NeonToken, a: &mut [f32], b: &[f32]) {
    elu8::<NeonToken>(t, a, b)
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn elu_wasm(t: Wasm128Token, a: &mut [f32], b: &[f32]) {
    elu8::<Wasm128Token>(t, a, b)
}

/// `sigmoid` slice on `tier`'s lanes — bit-identical to scalar on every element.
fn run_sigmoid_slice(tier: Tier, data: &mut [f32]) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => sigmoid_v4(t, data),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => sigmoid_v3(t, data),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => sigmoid_neon(t, data),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => sigmoid_wasm(t, data),
        Tier::Scalar => sigmoid_slice(data),
    }
}

/// `elu_gate` slice on `tier`'s lanes — bit-identical to scalar on every element.
fn run_elu_gate_slice(tier: Tier, a: &mut [f32], b: &[f32]) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => elu_v4(t, a, b),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => elu_v3(t, a, b),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => elu_neon(t, a, b),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => elu_wasm(t, a, b),
        Tier::Scalar => elu_gate_slice(a, b),
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn softmax_v4(t: X64V4Token, data: &mut [f32], m: f32) {
    softmax_exp16::<X64V4Token>(t, data, m)
}
#[arcane]
fn softmax_v3(t: X64V3Token, data: &mut [f32], m: f32) {
    softmax_exp8::<X64V3Token>(t, data, m)
}
#[arcane]
fn softmax_neon(t: NeonToken, data: &mut [f32], m: f32) {
    softmax_exp8::<NeonToken>(t, data, m)
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn softmax_wasm(t: Wasm128Token, data: &mut [f32], m: f32) {
    softmax_exp8::<Wasm128Token>(t, data, m)
}

/// Softmax `exp(x - m)` on `tier`'s lanes — bit-identical to scalar on every element.
fn run_softmax_exp_slice(tier: Tier, data: &mut [f32], m: f32) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => softmax_v4(t, data, m),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => softmax_v3(t, data, m),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => softmax_neon(t, data, m),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => softmax_wasm(t, data, m),
        Tier::Scalar => {
            for v in data {
                *v = exp(*v - m);
            }
        }
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn elu_bwd_v4(t: X64V4Token, d: &[f32], a: &[f32], b: &[f32], da: &mut [f32], db: &mut [f32]) {
    elu_bwd16::<X64V4Token>(t, d, a, b, da, db)
}
#[arcane]
fn elu_bwd_v3(t: X64V3Token, d: &[f32], a: &[f32], b: &[f32], da: &mut [f32], db: &mut [f32]) {
    elu_bwd8::<X64V3Token>(t, d, a, b, da, db)
}
#[arcane]
fn elu_bwd_neon(t: NeonToken, d: &[f32], a: &[f32], b: &[f32], da: &mut [f32], db: &mut [f32]) {
    elu_bwd8::<NeonToken>(t, d, a, b, da, db)
}
#[cfg(target_arch = "wasm32")]
#[arcane]
fn elu_bwd_wasm(t: Wasm128Token, d: &[f32], a: &[f32], b: &[f32], da: &mut [f32], db: &mut [f32]) {
    elu_bwd8::<Wasm128Token>(t, d, a, b, da, db)
}

/// `elu_gate` backward on `tier`'s lanes — bit-identical to scalar on every element.
fn run_elu_bwd_slice(tier: Tier, d: &[f32], a: &[f32], b: &[f32], da: &mut [f32], db: &mut [f32]) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => elu_bwd_v4(t, d, a, b, da, db),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => elu_bwd_v3(t, d, a, b, da, db),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => elu_bwd_neon(t, d, a, b, da, db),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => elu_bwd_wasm(t, d, a, b, da, db),
        Tier::Scalar => elu_gate_backward_scalar(d, a, b, da, db),
    }
}

#[autoversion]
fn mul_slice(a: &mut [f32], b: &[f32]) {
    for (x, &g) in a.iter_mut().zip(b) {
        *x *= g;
    }
}

const CHUNK: usize = 1 << 14;

/// `1 / (1 + exp(-x))` in place.
pub fn sigmoid(eng: &Engine, data: &mut [f32]) {
    for_each_row(eng, data, CHUNK, |_, row| run_sigmoid_slice(eng.tier, row));
}

/// `a = elu(a) * b`.
pub fn elu_gate(eng: &Engine, a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::InvalidArgument("elu gate: lengths differ"));
    }
    for_each_row(eng, a, CHUNK, |i, row| {
        let n = row.len();
        run_elu_gate_slice(eng.tier, row, &b[i * CHUNK..][..n]);
    });
    Ok(())
}

/// `a *= b`.
pub fn mul_assign(eng: &Engine, a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::InvalidArgument("mul: lengths differ"));
    }
    for_each_row(eng, a, CHUNK, |i, row| {
        let n = row.len();
        mul_slice(row, &b[i * CHUNK..][..n]);
    });
    Ok(())
}

#[autoversion]
fn acc_plane(sum: &mut [f64], x: &[f32]) {
    for (s, &v) in sum.iter_mut().zip(x) {
        *s += v as f64;
    }
}

#[autoversion]
fn acc_plane_sq_dev(sum: &mut [f64], x: &[f32], mean: &[f64]) {
    for ((s, &v), &m) in sum.iter_mut().zip(x).zip(mean) {
        let d = v as f64 - m;
        *s += d * d;
    }
}

#[autoversion]
fn ln_apply(x: &mut [f32], mean: &[f32], denom: &[f32], w: f32, b: f32) {
    for ((v, &m), &d) in x.iter_mut().zip(mean).zip(denom) {
        *v = (*v - m) / d * w + b;
    }
}

/// The reference's `LayerNorm` over the channel axis, per position:
/// `(x - mean) / sqrt(var + 1e-5) * weight + bias` with the biased variance.
pub fn layer_norm_channels(x: &mut Tensor<f32>, weight: &[f32], bias: &[f32]) -> Result<()> {
    let (c, n) = (x.c, x.h * x.w);
    if weight.len() != c || bias.len() != c || c == 0 {
        return Err(Error::InvalidArgument("layer norm: parameter length"));
    }
    let mut mean = alloc::vec![0.0f64; n];
    for ch in 0..c {
        acc_plane(&mut mean, x.plane(ch));
    }
    for m in &mut mean {
        *m /= c as f64;
    }
    let mut var = alloc::vec![0.0f64; n];
    for ch in 0..c {
        acc_plane_sq_dev(&mut var, x.plane(ch), &mean);
    }
    let mean32: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
    let denom: Vec<f32> = var
        .iter()
        .map(|&s| ((s / c as f64) as f32 + 1e-5).sqrt())
        .collect();
    for ch in 0..c {
        ln_apply(x.plane_mut(ch), &mean32, &denom, weight[ch], bias[ch]);
    }
    Ok(())
}

/// Dot product in `f64` with eight interleaved accumulators, combined pairwise.
#[autoversion]
fn dot8(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0.0f64; 8];
    let (ca, ta) = a.as_chunks::<8>();
    let (cb, tb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] as f64 * y[l] as f64;
        }
    }
    for (l, (&x, &y)) in ta.iter().zip(tb).enumerate() {
        acc[l] += x as f64 * y as f64;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[autoversion]
fn div_slice(x: &mut [f32], d: f32) {
    for v in x {
        *v /= d;
    }
}

#[autoversion]
fn axpy64(acc: &mut [f64], a: f64, x: &[f32]) {
    for (s, &v) in acc.iter_mut().zip(x) {
        *s += a * v as f64;
    }
}

/// `Approximate`-precision dot: `f32` products summed in a fixed 8-lane `f32`
/// accumulator — autovectorises where [`dot8`]'s `f64` cannot; not tier- or
/// reference-identical (the opt-in fast path).
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, ta) = a.as_chunks::<8>();
    let (cb, tb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    for (l, (&x, &y)) in ta.iter().zip(tb).enumerate() {
        acc[l] += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// `Approximate`-precision `acc += a·x` in `f32` (see [`dot_f32`]).
fn axpy32(acc: &mut [f32], a: f32, x: &[f32]) {
    for (s, &v) in acc.iter_mut().zip(x) {
        *s += a * v;
    }
}

/// The reference's channel attention (`tam.py::Attention` without its output projection).
///
/// `qkv` is `[3 * dim, h, w]`: query, key, value, each `heads` groups of `dim / heads` channels.
/// Per head: rows of `q` and `k` are L2-normalised over the `h * w` positions (`F.normalize`,
/// eps `1e-12`), `attn = softmax(q k^T * temperature[head])` over the last axis, result
/// `attn v`. Returns `[dim, h, w]`.
pub fn channel_attention(
    eng: &Engine,
    qkv: &Tensor<f32>,
    heads: usize,
    temperature: &[f32],
) -> Result<Tensor<f32>> {
    Ok(channel_attention_impl(eng, qkv, heads, temperature)?.0)
}

/// The forward returning `(out, attn)` — shared by [`channel_attention`] and
/// [`channel_attention_tape`] so `attn` is computed exactly once.
fn channel_attention_impl(
    eng: &Engine,
    qkv: &Tensor<f32>,
    heads: usize,
    temperature: &[f32],
) -> Result<(Tensor<f32>, Vec<f32>)> {
    if heads == 0 || temperature.len() != heads || !qkv.c.is_multiple_of(3 * heads) {
        return Err(Error::InvalidArgument("attention: channel / head counts"));
    }
    let dim = qkv.c / 3;
    let c = dim / heads;
    let n = qkv.h * qkv.w;
    let approx = eng.precision == crate::nn::fast::Precision::Approximate;

    // Normalised q and k.
    let mut qk = qkv.data[..2 * dim * n].to_vec();
    let _qk_charge = crate::mem::Charge::of_vec(&qk);
    for_each_row(eng, &mut qk, n, |_, row| {
        let norm = if approx {
            dot_f32(row, row).sqrt()
        } else {
            dot8(row, row).sqrt() as f32
        };
        div_slice(row, norm.max(1e-12));
    });
    let (q, k) = qk.split_at(dim * n);
    let v = &qkv.data[2 * dim * n..];

    // attn[head][i][j], row-wise softmax.
    let mut attn = alloc::vec![0.0f32; dim * c];
    let _attn_charge = crate::mem::Charge::of_vec(&attn);
    for_each_row(eng, &mut attn, c, |row_idx, row| {
        let head = row_idx / c;
        let qi = &q[row_idx * n..][..n];
        let mut max = f32::NEG_INFINITY;
        for (j, a) in row.iter_mut().enumerate() {
            let kj = &k[(head * c + j) * n..][..n];
            *a = if approx {
                dot_f32(qi, kj)
            } else {
                dot8(qi, kj) as f32
            } * temperature[head];
            max = max.max(*a);
        }
        run_softmax_exp_slice(eng.tier, row, max);
        let mut sum = 0.0f64;
        for a in row.iter() {
            sum += *a as f64;
        }
        let sum = sum as f32;
        for a in row.iter_mut() {
            *a /= sum;
        }
    });

    let mut out = Tensor::<f32>::zeros(dim, qkv.h, qkv.w)?;
    for_each_row(eng, &mut out.data, n, |row_idx, row| {
        let head = row_idx / c;
        let weights = &attn[row_idx * c..][..c];
        if approx {
            scratch32(n, |acc| {
                for (j, &a) in weights.iter().enumerate() {
                    axpy32(acc, a, &v[(head * c + j) * n..][..n]);
                }
                for (o, &s) in row.iter_mut().zip(&*acc) {
                    *o = s;
                }
            });
        } else {
            scratch64(n, |acc| {
                for (j, &a) in weights.iter().enumerate() {
                    axpy64(acc, a as f64, &v[(head * c + j) * n..][..n]);
                }
                for (o, &s) in row.iter_mut().zip(&*acc) {
                    *o = s as f32;
                }
            });
        }
    });
    Ok((out, attn))
}

/// `f` runs with a zeroed `f32` scratch of `n`, reused per thread between calls
/// — the `Approximate`-precision accumulator (see [`scratch64`]).
fn scratch32(n: usize, f: impl FnOnce(&mut [f32])) {
    #[cfg(feature = "std")]
    {
        use std::cell::RefCell;
        thread_local! {
            static SCRATCH32: RefCell<(Vec<f32>, crate::mem::Charge)> =
                const { RefCell::new((Vec::new(), crate::mem::Charge::EMPTY)) };
        }
        SCRATCH32.with(|s| {
            let mut s = s.borrow_mut();
            s.0.clear();
            s.0.resize(n, 0.0);
            let bytes = crate::mem::vec_bytes(&s.0);
            s.1.resize(bytes);
            f(s.0.as_mut_slice());
        });
    }
    #[cfg(not(feature = "std"))]
    f(&mut alloc::vec![0.0f32; n]);
}

/// `f` runs with a zeroed `f64` scratch of `n`, reused per thread between calls
/// — the attention VJP's accumulators (an `f64` [`conv` module's `border_buf`]).
fn scratch64(n: usize, f: impl FnOnce(&mut [f64])) {
    #[cfg(feature = "std")]
    {
        use std::cell::RefCell;
        thread_local! {
            /// The scratch and its ledger token; both die at thread exit.
            static SCRATCH64: RefCell<(Vec<f64>, crate::mem::Charge)> =
                const { RefCell::new((Vec::new(), crate::mem::Charge::EMPTY)) };
        }
        SCRATCH64.with(|s| {
            let mut s = s.borrow_mut();
            s.0.clear();
            s.0.resize(n, 0.0);
            let bytes = crate::mem::vec_bytes(&s.0);
            s.1.resize(bytes);
            f(s.0.as_mut_slice());
        });
    }
    #[cfg(not(feature = "std"))]
    f(&mut alloc::vec![0.0f64; n]);
}

/// [`channel_attention`] that also returns the `attn` weights — the tape
/// `channel_attention_backward` needs (RDLR HOP VJP).
pub fn channel_attention_tape(
    eng: &Engine,
    qkv: &Tensor<f32>,
    heads: usize,
    temperature: &[f32],
) -> Result<(Tensor<f32>, Vec<f32>)> {
    channel_attention_impl(eng, qkv, heads, temperature)
}

/// The input gradient through [`channel_attention`] (RDLR HOP VJP): `d_out`
/// `[dim, h, w]` → `d(qkv)` `[3·dim, h, w]`. `attn` is the forward's softmax
/// weights (from [`channel_attention_tape`]). `d_attn = d_out·v^T`, softmax
/// VJP, `dq = T·d·k`, `dk = T·d^T·q`, `dv = attn^T·d_out`, then the L2-norm
/// backward on q,k — all sums in `dot8` order for tier bit-identity.
pub fn channel_attention_backward(
    eng: &Engine,
    qkv: &Tensor<f32>,
    attn: &[f32],
    d_out: &Tensor<f32>,
    heads: usize,
    temperature: &[f32],
) -> Result<Tensor<f32>> {
    if heads == 0 || temperature.len() != heads || !qkv.c.is_multiple_of(3 * heads) {
        return Err(Error::InvalidArgument("attention: channel / head counts"));
    }
    let dim = qkv.c / 3;
    let c = dim / heads;
    let n = qkv.h * qkv.w;
    let approx = eng.precision == crate::nn::fast::Precision::Approximate;
    // Re-normalise q,k and keep each row's norm for the L2-norm backward.
    let mut qk = qkv.data[..2 * dim * n].to_vec();
    let _qk_charge = crate::mem::Charge::of_vec(&qk);
    let mut norms = alloc::vec![0.0f32; 2 * dim];
    let _norms_charge = crate::mem::Charge::of_vec(&norms);
    // Each row's clamped L2 norm, for the L2-norm backward (one slice each).
    for_each_row(eng, &mut norms, 1, |i, slot| {
        let r = &qk[i * n..][..n];
        slot[0] = if approx {
            dot_f32(r, r).sqrt().max(1e-12)
        } else {
            (dot8(r, r).sqrt() as f32).max(1e-12)
        };
    });
    for_each_row(eng, &mut qk, n, |i, row| {
        div_slice(row, norms[i]);
    });
    let (q, k) = qk.split_at(dim * n);
    let v = &qkv.data[2 * dim * n..];
    // `d_attn = d_out·v^T` then the row softmax VJP — `da` is the shared
    // intermediate all three passes (dq, dk, dv) read, so it is computed once.
    let mut da = alloc::vec![0.0f32; dim * c];
    let _da_charge = crate::mem::Charge::of_vec(&da);
    for_each_row(eng, &mut da, c, |row_idx, row| {
        let head = row_idx / c;
        let a_row = &attn[row_idx * c..][..c];
        let d_out_row = &d_out.data[row_idx * n..][..n];
        let mut sdot = 0.0f64;
        for (j, a) in row.iter_mut().enumerate() {
            let vj = &v[(head * c + j) * n..][..n];
            *a = if approx {
                dot_f32(d_out_row, vj)
            } else {
                dot8(d_out_row, vj) as f32
            };
            sdot += a_row[j] as f64 * *a as f64;
        }
        for (j, a) in row.iter_mut().enumerate() {
            *a = a_row[j] * (*a - sdot as f32);
        }
    });
    let mut dqkv = Tensor::<f32>::zeros(3 * dim, qkv.h, qkv.w)?;
    for_each_row(eng, &mut dqkv.data, n, |row_idx, row| {
        // Rows 0..dim of dqkv hold dq / dk / dv for the corresponding qkv row.
        if row_idx < dim {
            // dq_row[l] = T · Σ_j da_row[j]·k_j[l], then the L2-norm backward.
            let head = row_idx / c;
            let da_row = &da[row_idx * c..][..c];
            let q_row = &q[row_idx * n..][..n];
            if approx {
                scratch32(n, |acc| {
                    for (j, &a) in da_row.iter().enumerate() {
                        axpy32(acc, temperature[head] * a, &k[(head * c + j) * n..][..n]);
                    }
                    let proj = dot_f32(q_row, acc);
                    let inv = 1.0 / norms[row_idx];
                    for (o, (&ql, &s)) in row.iter_mut().zip(q_row.iter().zip(acc.iter())) {
                        *o = (s - proj * ql) * inv;
                    }
                });
            } else {
                scratch64(n, |acc| {
                    for (j, &a) in da_row.iter().enumerate() {
                        let w = temperature[head] * a;
                        let kj = &k[(head * c + j) * n..][..n];
                        for (s, &kv) in acc.iter_mut().zip(kj) {
                            *s += w as f64 * kv as f64;
                        }
                    }
                    let mut proj = 0.0f64;
                    for (&ql, &s) in q_row.iter().zip(acc.iter()) {
                        proj += ql as f64 * s;
                    }
                    let inv = 1.0 / norms[row_idx] as f64;
                    for (o, (&ql, &s)) in row.iter_mut().zip(q_row.iter().zip(acc.iter())) {
                        *o = ((s - proj * ql as f64) * inv) as f32;
                    }
                });
            }
        } else if row_idx < 2 * dim {
            // dk_j[l] = T · Σ_i da_i[j]·q_i[l], then the L2-norm backward.
            let (head, j) = ((row_idx - dim) / c, (row_idx - dim) % c);
            let kj = &k[(head * c + j) * n..][..n];
            if approx {
                scratch32(n, |acc| {
                    for i in 0..c {
                        axpy32(
                            acc,
                            temperature[head] * da[(head * c + i) * c + j],
                            &q[(head * c + i) * n..][..n],
                        );
                    }
                    let proj = dot_f32(kj, acc);
                    let inv = 1.0 / norms[dim + head * c + j];
                    for (o, (&kl, &s)) in row.iter_mut().zip(kj.iter().zip(acc.iter())) {
                        *o = (s - proj * kl) * inv;
                    }
                });
            } else {
                scratch64(n, |acc| {
                    for i in 0..c {
                        let w = temperature[head] * da[(head * c + i) * c + j];
                        let qi = &q[(head * c + i) * n..][..n];
                        for (s, &qv) in acc.iter_mut().zip(qi) {
                            *s += w as f64 * qv as f64;
                        }
                    }
                    let mut proj = 0.0f64;
                    for (&kl, &s) in kj.iter().zip(acc.iter()) {
                        proj += kl as f64 * s;
                    }
                    let inv = 1.0 / norms[dim + head * c + j] as f64;
                    for (o, (&kl, &s)) in row.iter_mut().zip(kj.iter().zip(acc.iter())) {
                        *o = ((s - proj * kl as f64) * inv) as f32;
                    }
                });
            }
        } else {
            // dv_j[l] = Σ_i attn[i][j]·d_out_i[l].
            let (head, j) = ((row_idx - 2 * dim) / c, (row_idx - 2 * dim) % c);
            if approx {
                scratch32(n, |acc| {
                    for i in 0..c {
                        axpy32(
                            acc,
                            attn[(head * c + i) * c + j],
                            &d_out.data[(head * c + i) * n..][..n],
                        );
                    }
                    for (o, &s) in row.iter_mut().zip(acc.iter()) {
                        *o = s;
                    }
                });
            } else {
                scratch64(n, |acc| {
                    for i in 0..c {
                        let a = attn[(head * c + i) * c + j] as f64;
                        let d_out_row = &d_out.data[(head * c + i) * n..][..n];
                        for (s, &dv) in acc.iter_mut().zip(d_out_row) {
                            *s += a * dv as f64;
                        }
                    }
                    for (o, &s) in row.iter_mut().zip(acc.iter()) {
                        *o = s as f32;
                    }
                });
            }
        }
    });
    Ok(dqkv)
}

/// `elu_gate` backward: `a = elu(a)·b` upstream `d` → `(d_a, d_b)` where
/// `d_a = elu'(a)·b·d` (`elu'(x) = 1` for `x>0`, `exp(x)` else) and
/// `d_b = elu(a)·d`. `a`/`b` are the forward's pre-gating halves. RDLR VJP.
pub fn elu_gate_backward(d: &[f32], a: &[f32], b: &[f32], d_a: &mut [f32], d_b: &mut [f32]) {
    run_elu_bwd_slice(Tier::detect(), d, a, b, d_a, d_b)
}

fn elu_gate_backward_scalar(d: &[f32], a: &[f32], b: &[f32], d_a: &mut [f32], d_b: &mut [f32]) {
    for (((da, db), (&av, &bv)), &o) in d_a
        .iter_mut()
        .zip(d_b.iter_mut())
        .zip(a.iter().zip(b.iter()))
        .zip(d.iter())
    {
        let elu = if av > 0.0 { av } else { exp(av) - 1.0 };
        let delu = if av > 0.0 { 1.0 } else { exp(av) };
        *da = delu * bv * o;
        *db = elu * o;
    }
}

/// `layer_norm_channels` backward: per-position LN `out[c][i] =
/// (x[c][i]−μ[i])/σ[i]·w[c]+b[c]` → `d_x`. `x` is the forward's pre-LN input
/// (μ,σ recomputed). Per position `d_x[c] = (d_ŷ[c] − mean(d_ŷ) − ŷ[c]·
/// mean(d_ŷ·ŷ))/σ` with `d_ŷ = d·w`. RDLR VJP.
pub fn layer_norm_channels_backward(
    d: &Tensor<f32>,
    x: &Tensor<f32>,
    weight: &[f32],
    bias: &[f32],
) -> Result<Tensor<f32>> {
    let (c, n) = (x.c, x.h * x.w);
    if weight.len() != c || bias.len() != c || c == 0 || d.c != c {
        return Err(Error::InvalidArgument("layer norm backward: shape"));
    }
    // Recompute mean / denom / the normalised x per position (as the forward).
    let mut mean = alloc::vec![0.0f64; n];
    for ch in 0..c {
        acc_plane(&mut mean, x.plane(ch));
    }
    for m in &mut mean {
        *m /= c as f64;
    }
    let mut var = alloc::vec![0.0f64; n];
    for ch in 0..c {
        acc_plane_sq_dev(&mut var, x.plane(ch), &mean);
    }
    let mut xn = alloc::vec![0.0f32; c * n];
    let _xn_charge = crate::mem::Charge::of_vec(&xn);
    let mut denom = alloc::vec![0.0f32; n];
    let _denom_charge = crate::mem::Charge::of_vec(&denom);
    for i in 0..n {
        denom[i] = ((var[i] / c as f64) as f32 + 1e-5).sqrt();
    }
    for ch in 0..c {
        for (o, (&xv, i)) in xn[ch * n..][..n]
            .iter_mut()
            .zip(x.plane(ch).iter().zip(0..n))
        {
            *o = (xv - mean[i] as f32) / denom[i];
        }
    }
    // Per position: d_x[c] = (d_ŷ[c] − Σ d_ŷ/c − ŷ[c]·Σ d_ŷ·ŷ/c) / σ, d_ŷ = d·w.
    let mut dx = Tensor::<f32>::zeros(c, x.h, x.w)?;
    for i in 0..n {
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for ch in 0..c {
            let dy = d.data[ch * n + i] * weight[ch];
            s1 += dy as f64;
            s2 += dy as f64 * xn[ch * n + i] as f64;
        }
        let (m1, m2, sig) = (s1 as f32 / c as f32, s2 as f32 / c as f32, denom[i]);
        for ch in 0..c {
            let dy = d.data[ch * n + i] * weight[ch];
            dx.data[ch * n + i] = (dy - m1 - xn[ch * n + i] * m2) / sig;
        }
    }
    Ok(dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_tracks_libm() {
        let mut worst = 0.0f64;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (got, want) = (exp(x) as f64, libm::exp(x as f64));
            worst = worst.max(((got - want) / want).abs());
            x += 0.003_7;
        }
        assert!(worst < 3e-7, "worst relative error {worst:e}");
        assert_eq!(exp(0.0), 1.0);
        assert!(exp(-1000.0) > 0.0 && exp(-1000.0) < 2e-38);
        assert!(exp(f32::NAN).is_nan());
    }

    #[test]
    fn attention_of_one_position_is_value_mix() {
        // One position: normalised q, k are +-1, so attn is a softmax of +-temperature.
        let eng = Engine::with(super::super::Tier::Scalar, false);
        let qkv = Tensor::from_vec(6, 1, 1, alloc::vec![2.0, -3.0, 0.5, 4.0, 10.0, 20.0]).unwrap();
        let out = channel_attention(&eng, &qkv, 1, &[1.0]).unwrap();
        // row 0: q = 1, k = (1, 1) -> attn = (0.5, 0.5); row 1: q = -1 -> same
        assert_eq!(out.data, [15.0, 15.0]);
    }
}
