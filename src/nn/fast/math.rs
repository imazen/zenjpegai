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

use super::{Engine, for_each_row};
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

#[autoversion]
fn sigmoid_slice(data: &mut [f32]) {
    for v in data {
        *v = 1.0 / (1.0 + exp(-*v));
    }
}

/// `elu(a) * b` into `a`: `a > 0 ? a : exp(a) - 1`, times `b`.
#[autoversion]
fn elu_gate_slice(a: &mut [f32], b: &[f32]) {
    for (x, &g) in a.iter_mut().zip(b) {
        let e = exp(*x) - 1.0;
        let elu = if *x > 0.0 { *x } else { e };
        *x = elu * g;
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
    for_each_row(eng, data, CHUNK, |_, row| sigmoid_slice(row));
}

/// `a = elu(a) * b`.
pub fn elu_gate(eng: &Engine, a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::InvalidArgument("elu gate: lengths differ"));
    }
    for_each_row(eng, a, CHUNK, |i, row| {
        let n = row.len();
        elu_gate_slice(row, &b[i * CHUNK..][..n]);
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
    if heads == 0 || temperature.len() != heads || !qkv.c.is_multiple_of(3 * heads) {
        return Err(Error::InvalidArgument("attention: channel / head counts"));
    }
    let dim = qkv.c / 3;
    let c = dim / heads;
    let n = qkv.h * qkv.w;

    // Normalised q and k.
    let mut qk = qkv.data[..2 * dim * n].to_vec();
    let _qk_charge = crate::mem::Charge::of_vec(&qk);
    for_each_row(eng, &mut qk, n, |_, row| {
        let norm = dot8(row, row).sqrt() as f32;
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
            *a = dot8(qi, kj) as f32 * temperature[head];
            max = max.max(*a);
        }
        let mut sum = 0.0f64;
        for a in row.iter_mut() {
            *a = exp(*a - max);
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
        let mut acc = alloc::vec![0.0f64; n];
        let _acc_charge = crate::mem::Charge::of_vec(&acc);
        for (j, &a) in weights.iter().enumerate() {
            axpy64(&mut acc, a as f64, &v[(head * c + j) * n..][..n]);
        }
        for (o, &s) in row.iter_mut().zip(&acc) {
            *o = s as f32;
        }
    });
    Ok(out)
}

/// [`channel_attention`] that also returns the `attn` weights — the tape
/// `channel_attention_backward` needs (RDLR HOP VJP).
#[allow(dead_code)]
pub fn channel_attention_tape(
    eng: &Engine,
    qkv: &Tensor<f32>,
    heads: usize,
    temperature: &[f32],
) -> Result<(Tensor<f32>, Vec<f32>)> {
    let out = channel_attention(eng, qkv, heads, temperature)?;
    // Rebuild `attn` exactly as `channel_attention` did (the softmax weights are
    // cheap relative to storing a second forward of the QK products).
    let dim = qkv.c / 3;
    let c = dim / heads;
    let n = qkv.h * qkv.w;
    let mut qk = qkv.data[..2 * dim * n].to_vec();
    for_each_row(eng, &mut qk, n, |_, row| {
        let norm = dot8(row, row).sqrt() as f32;
        div_slice(row, norm.max(1e-12));
    });
    let (q, k) = qk.split_at(dim * n);
    let mut attn = alloc::vec![0.0f32; dim * c];
    for_each_row(eng, &mut attn, c, |row_idx, row| {
        let head = row_idx / c;
        let qi = &q[row_idx * n..][..n];
        let mut max = f32::NEG_INFINITY;
        for (j, a) in row.iter_mut().enumerate() {
            let kj = &k[(head * c + j) * n..][..n];
            *a = dot8(qi, kj) as f32 * temperature[head];
            max = max.max(*a);
        }
        let mut sum = 0.0f64;
        for a in row.iter_mut() {
            *a = exp(*a - max);
            sum += *a as f64;
        }
        let sum = sum as f32;
        for a in row.iter_mut() {
            *a /= sum;
        }
    });
    Ok((out, attn))
}

/// The input gradient through [`channel_attention`] (RDLR HOP VJP): `d_out`
/// `[dim, h, w]` → `d(qkv)` `[3·dim, h, w]`. `attn` is the forward's softmax
/// weights (from [`channel_attention_tape`]). `d_attn = d_out·v^T`, softmax
/// VJP, `dq = T·d·k`, `dk = T·d^T·q`, `dv = attn^T·d_out`, then the L2-norm
/// backward on q,k — all sums in `dot8` order for tier bit-identity.
#[allow(dead_code)]
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
    // Re-normalise q,k and keep each row's norm for the L2-norm backward.
    let mut qk = qkv.data[..2 * dim * n].to_vec();
    let _qk_charge = crate::mem::Charge::of_vec(&qk);
    let mut norms = alloc::vec![0.0f32; 2 * dim];
    let _norms_charge = crate::mem::Charge::of_vec(&norms);
    // Each row's clamped L2 norm, for the L2-norm backward (one slice each).
    for_each_row(eng, &mut norms, 1, |i, slot| {
        let r = &qk[i * n..][..n];
        slot[0] = (dot8(r, r).sqrt() as f32).max(1e-12);
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
            *a = dot8(d_out_row, vj) as f32;
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
            let mut acc = alloc::vec![0.0f64; n];
            for (j, &a) in da_row.iter().enumerate() {
                let w = temperature[head] * a;
                let kj = &k[(head * c + j) * n..][..n];
                for (s, &kv) in acc.iter_mut().zip(kj) {
                    *s += w as f64 * kv as f64;
                }
            }
            let q_row = &q[row_idx * n..][..n];
            let mut proj = 0.0f64;
            for (&ql, &s) in q_row.iter().zip(&acc) {
                proj += ql as f64 * s;
            }
            let inv = 1.0 / norms[row_idx] as f64;
            for (o, (&ql, &s)) in row.iter_mut().zip(q_row.iter().zip(&acc)) {
                *o = ((s - proj * ql as f64) * inv) as f32;
            }
        } else if row_idx < 2 * dim {
            // dk_j[l] = T · Σ_i da_i[j]·q_i[l], then the L2-norm backward.
            let (head, j) = ((row_idx - dim) / c, (row_idx - dim) % c);
            let kj = &k[(head * c + j) * n..][..n];
            let mut acc = alloc::vec![0.0f64; n];
            for i in 0..c {
                let w = temperature[head] * da[(head * c + i) * c + j];
                let qi = &q[(head * c + i) * n..][..n];
                for (s, &qv) in acc.iter_mut().zip(qi) {
                    *s += w as f64 * qv as f64;
                }
            }
            let mut proj = 0.0f64;
            for (&kl, &s) in kj.iter().zip(&acc) {
                proj += kl as f64 * s;
            }
            let inv = 1.0 / norms[dim + head * c + j] as f64;
            for (o, (&kl, &s)) in row.iter_mut().zip(kj.iter().zip(&acc)) {
                *o = ((s - proj * kl as f64) * inv) as f32;
            }
        } else {
            // dv_j[l] = Σ_i attn[i][j]·d_out_i[l].
            let (head, j) = ((row_idx - 2 * dim) / c, (row_idx - 2 * dim) % c);
            let mut acc = alloc::vec![0.0f64; n];
            for i in 0..c {
                let a = attn[(head * c + i) * c + j] as f64;
                let d_out_row = &d_out.data[(head * c + i) * n..][..n];
                for (s, &dv) in acc.iter_mut().zip(d_out_row) {
                    *s += a * dv as f64;
                }
            }
            for (o, &s) in row.iter_mut().zip(&acc) {
                *o = s as f32;
            }
        }
    });
    Ok(dqkv)
}

/// `elu_gate` backward: `a = elu(a)·b` upstream `d` → `(d_a, d_b)` where
/// `d_a = elu'(a)·b·d` (`elu'(x) = 1` for `x>0`, `exp(x)` else) and
/// `d_b = elu(a)·d`. `a`/`b` are the forward's pre-gating halves. RDLR VJP.
#[allow(dead_code)]
pub fn elu_gate_backward(d: &[f32], a: &[f32], b: &[f32], d_a: &mut [f32], d_b: &mut [f32]) {
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
#[allow(dead_code)]
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
