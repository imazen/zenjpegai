//! Depthwise convolution (`groups == channels`) on channel-blocked tensors.
//!
//! Every output channel reads one input channel, so a block of `V` channels is `V` independent
//! lanes: per output position the accumulator starts at the bias and adds `weight * input` tap by
//! tap, `(ky, kx)` ascending, with a fused multiply-add. That is [`crate::nn::reference`]'s order
//! for a one-channel group, so results are bit-identical to it on every tier.

use alloc::vec::Vec;

use archmage::prelude::*;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use magetypes::simd::generic::f32x8;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use magetypes::simd::generic::f32x16;

use super::simd::{Lanes8, SimdF32};
use super::tensor::BTensor;
use super::{Engine, Tier, for_each_row};
use crate::error::{Error, Result};
use crate::nn::Conv2d;

const MAX_TAPS: usize = 9;

struct DwRowJob<'a> {
    /// Output row of one block: `width * V`.
    out: &'a mut [f32],
    /// One padded input row per tap, already offset to the tap's column: `>= width * V` floats.
    rows: [&'a [f32]; MAX_TAPS],
    ntaps: usize,
    /// `[taps][V]`.
    w: &'a [f32],
    bias: &'a [f32],
}

#[inline(always)]
fn dw_row<F: SimdF32<V>, const V: usize>(t: F::Token, job: &mut DwRowJob<'_>) {
    let (out, _) = job.out.as_chunks_mut::<V>();
    let (w, _) = job.w.as_chunks::<V>();
    let mut bias = [0.0f32; V];
    bias.copy_from_slice(&job.bias[..V]);
    let bias = F::load(t, &bias);
    let mut wv = [F::splat(t, 0.0); MAX_TAPS];
    for (d, s) in wv.iter_mut().zip(w) {
        *d = F::load(t, s);
    }
    let mut rows: [&[[f32; V]]; MAX_TAPS] = [&[]; MAX_TAPS];
    for (d, s) in rows.iter_mut().zip(job.rows) {
        *d = &s.as_chunks::<V>().0[..out.len()];
    }
    let ntaps = job.ntaps;
    for (x, o) in out.iter_mut().enumerate() {
        let mut acc = bias;
        for k in 0..ntaps {
            acc = wv[k].mul_add(F::load(t, &rows[k][x]), acc);
        }
        acc.store(o);
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn dw_row_v4(t: X64V4Token, job: &mut DwRowJob<'_>) {
    dw_row::<f32x16<X64V4Token>, 16>(t, job)
}

#[arcane]
fn dw_row_v3(t: X64V3Token, job: &mut DwRowJob<'_>) {
    dw_row::<f32x8<X64V3Token>, 8>(t, job)
}

#[arcane]
fn dw_row_neon(t: NeonToken, job: &mut DwRowJob<'_>) {
    dw_row::<f32x8<NeonToken>, 8>(t, job)
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn dw_row_wasm128(t: Wasm128Token, job: &mut DwRowJob<'_>) {
    dw_row::<f32x8<Wasm128Token>, 8>(t, job)
}

fn dw_row_scalar(job: &mut DwRowJob<'_>) {
    dw_row::<Lanes8, 8>(ScalarToken, job)
}

/// A stride-1 depthwise convolution packed for one block size.
#[derive(Clone, Debug)]
pub struct PackedDepthwise {
    ch: usize,
    kh: usize,
    kw: usize,
    pad: (usize, usize),
    v: usize,
    /// `[cb][taps][V]`.
    weight: Vec<f32>,
    /// `[cb][V]`.
    bias: Vec<f32>,
}

impl PackedDepthwise {
    pub fn new(conv: &Conv2d, v: usize) -> Result<Self> {
        if conv.groups != conv.in_ch
            || conv.in_ch != conv.out_ch
            || conv.groups < 2
            || conv.stride != 1
            || conv.kh * conv.kw > MAX_TAPS
        {
            return Err(Error::Unsupported(
                "fast path: not a stride-1 depthwise convolution",
            ));
        }
        let (ch, ntaps) = (conv.in_ch, conv.kh * conv.kw);
        let cb = ch.div_ceil(v);
        let mut weight = alloc::vec![0.0f32; cb * ntaps * v];
        for c in 0..ch {
            for t in 0..ntaps {
                weight[((c / v) * ntaps + t) * v + c % v] = conv.weight[c * ntaps + t];
            }
        }
        let mut bias = alloc::vec![0.0f32; cb * v];
        if let Some(b) = &conv.bias {
            bias[..ch].copy_from_slice(b);
        }
        Ok(Self {
            ch,
            kh: conv.kh,
            kw: conv.kw,
            pad: (conv.pad_h, conv.pad_w),
            v,
            weight,
            bias,
        })
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let v = self.v;
        if x.c != self.ch || x.v != v {
            return Err(Error::InvalidArgument(
                "depthwise conv: channels / block size mismatch",
            ));
        }
        let (ph, pw) = self.pad;
        if x.h + 2 * ph < self.kh || x.w + 2 * pw < self.kw {
            return Err(Error::InvalidArgument(
                "depthwise conv: input smaller than the kernel",
            ));
        }
        let padded;
        let xp = if (ph, pw) == (0, 0) {
            x
        } else {
            padded = x.pad_par(eng, ph, pw, ph, pw)?;
            &padded
        };
        let (oh, ow) = (xp.h - self.kh + 1, xp.w - self.kw + 1);
        let mut out = BTensor::scratch(self.ch, oh, ow, v)?;
        let ntaps = self.kh * self.kw;
        let tier = eng.tier;
        for_each_row(eng, &mut out.data, ow * v, |idx, row| {
            let (cb, oy) = (idx / oh, idx % oh);
            let mut rows: [&[f32]; MAX_TAPS] = [&[]; MAX_TAPS];
            for ky in 0..self.kh {
                for kx in 0..self.kw {
                    rows[ky * self.kw + kx] =
                        &xp.data[((cb * xp.h + oy + ky) * xp.w + kx) * v..][..ow * v];
                }
            }
            let mut job = DwRowJob {
                out: row,
                rows,
                ntaps,
                w: &self.weight[cb * ntaps * v..][..ntaps * v],
                bias: &self.bias[cb * v..][..v],
            };
            match tier {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Tier::V4(t) => dw_row_v4(t, &mut job),
                #[cfg(target_arch = "x86_64")]
                Tier::V3(t) => dw_row_v3(t, &mut job),
                #[cfg(target_arch = "aarch64")]
                Tier::Neon(t) => dw_row_neon(t, &mut job),
                #[cfg(target_arch = "wasm32")]
                Tier::Wasm128(t) => dw_row_wasm128(t, &mut job),
                Tier::Scalar => dw_row_scalar(&mut job),
            }
        });
        Ok(out)
    }
}
