//! Convolution and transposed convolution on channel-blocked tensors.
//!
//! One micro-kernel serves both. For one output row of one output block it keeps `B` output
//! positions x `V` output channels in registers and, input channel by input channel and tap by
//! tap, adds `weight * input` with a fused multiply-add: the input sample is broadcast, the
//! weight vector holds the `V` output channels' weights for that (input channel, tap). That is
//! the accumulation order of [`crate::nn::reference`], so results are bit-identical to it for
//! every `V`, `B`, tier and thread count.
//!
//! A transposed convolution with stride 2 is run as one such pass per output-column parity: the
//! taps that reach a given output parity form a small ordinary kernel over the input.

use alloc::vec::Vec;

use archmage::prelude::*;
use magetypes::simd::generic::f32x8;
#[cfg(feature = "avx512")]
use magetypes::simd::generic::f32x16;

use super::simd::{Lanes8, SimdF32};
use super::tensor::BTensor;
use super::{Engine, Tier};
use crate::error::{Error, Result};
use crate::nn::{Conv2d, ConvTranspose2d};

/// Most taps a single kernel pass handles (3x3 = 9; a 4x4 stride-2 phase = 4).
const MAX_TAPS: usize = 9;

/// Work description for one output row of one output-channel block.
pub(crate) struct RowJob<'a> {
    /// The output row: `width * V` floats. Only positions `o0 + j * os`, `j < n`, are written.
    pub out: &'a mut [f32],
    pub o0: usize,
    pub os: usize,
    pub n: usize,
    /// Padded input, blocked: `[icb][ph][pw][V]`.
    pub xp: &'a [f32],
    pub ph: usize,
    pub pw: usize,
    /// Input blocks to accumulate, and the total number of real input channels.
    pub icb0: usize,
    pub icb1: usize,
    pub in_ch: usize,
    /// Per tap, in accumulation order: padded input row, and the input column of output `j = 0`.
    pub taps: &'a [(usize, usize)],
    /// Weights of this output block: `[icb1 - icb0][V][taps][V]`.
    pub w: &'a [f32],
    /// Bias of this output block: `V` floats.
    pub bias: &'a [f32],
}

/// `B` output positions of one input block: load accumulators, add every (channel, tap), store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block<F: SimdF32<V>, const V: usize, const B: usize>(
    t: F::Token,
    out: &mut [[f32; V]],
    os: usize,
    rows: &[&[[f32; V]]],
    j: usize,
    w: &[[f32; V]],
    vin: usize,
) {
    let mut acc = [F::splat(t, 0.0); B];
    for b in 0..B {
        acc[b] = F::load(t, &out[b * os]);
    }
    let mut wi = 0;
    for v in 0..vin {
        for r in rows {
            let r = &r[j..j + B];
            let wv = F::load(t, &w[wi]);
            wi += 1;
            for b in 0..B {
                // `v < V` always; the mask lets the compiler see it.
                acc[b] = wv.mul_add(F::splat(t, r[b][v & (V - 1)]), acc[b]);
            }
        }
    }
    for b in 0..B {
        acc[b].store(&mut out[b * os]);
    }
}

#[inline(always)]
fn conv_row<F: SimdF32<V>, const V: usize, const B: usize>(t: F::Token, job: &mut RowJob<'_>) {
    let (out, _) = job.out.as_chunks_mut::<V>();
    let (xp, _) = job.xp.as_chunks::<V>();
    let (w, _) = job.w.as_chunks::<V>();
    let mut bias = [0.0f32; V];
    bias.copy_from_slice(&job.bias[..V]);
    let ntaps = job.taps.len();
    let (o0, os, n) = (job.o0, job.os, job.n);
    for j in 0..n {
        out[o0 + j * os] = bias;
    }
    for icb in job.icb0..job.icb1 {
        let vin = (job.in_ch - icb * V).min(V);
        let wblk = &w[(icb - job.icb0) * V * ntaps..][..V * ntaps];
        let mut rows: [&[[f32; V]]; MAX_TAPS] = [&[]; MAX_TAPS];
        for (r, &(y, dx)) in rows.iter_mut().zip(job.taps) {
            *r = &xp[(icb * job.ph + y) * job.pw + dx..][..n];
        }
        let rows = &rows[..ntaps];
        let mut j = 0;
        while j + B <= n {
            block::<F, V, B>(t, &mut out[o0 + j * os..], os, rows, j, wblk, vin);
            j += B;
        }
        while j + 8 <= n {
            block::<F, V, 8>(t, &mut out[o0 + j * os..], os, rows, j, wblk, vin);
            j += 8;
        }
        while j + 4 <= n {
            block::<F, V, 4>(t, &mut out[o0 + j * os..], os, rows, j, wblk, vin);
            j += 4;
        }
        while j < n {
            block::<F, V, 1>(t, &mut out[o0 + j * os..], os, rows, j, wblk, vin);
            j += 1;
        }
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn conv_row_v4(t: X64V4Token, job: &mut RowJob<'_>) {
    conv_row::<f32x16<X64V4Token>, 16, 28>(t, job)
}

#[arcane]
fn conv_row_v3(t: X64V3Token, job: &mut RowJob<'_>) {
    conv_row::<f32x8<X64V3Token>, 8, 12>(t, job)
}

#[arcane]
fn conv_row_neon(t: NeonToken, job: &mut RowJob<'_>) {
    conv_row::<f32x8<NeonToken>, 8, 12>(t, job)
}

fn conv_row_scalar(job: &mut RowJob<'_>) {
    conv_row::<Lanes8, 8, 4>(ScalarToken, job)
}

pub(crate) fn run_row(tier: Tier, job: &mut RowJob<'_>) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => conv_row_v4(t, job),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => conv_row_v3(t, job),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => conv_row_neon(t, job),
        Tier::Scalar => conv_row_scalar(job),
    }
}

/// Run `f(index, row)` over the rows of `data`, in parallel when the engine allows it.
pub(crate) fn for_each_row<T: Send>(
    eng: &Engine,
    data: &mut [T],
    row_len: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    #[cfg(feature = "parallel")]
    if eng.parallel {
        use rayon::prelude::*;
        data.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
        return;
    }
    let _ = eng;
    for (i, row) in data.chunks_mut(row_len).enumerate() {
        f(i, row);
    }
}

/// Pack `[out][in][taps]` weights (any tap order already applied) into
/// `[ocb][icb][V in][taps][V out]`, zero for channels beyond the real counts.
#[allow(clippy::too_many_arguments)]
fn pack(
    v: usize,
    out_ch: usize,
    in_ch_group: usize,
    ntaps: usize,
    groups: usize,
    get: impl Fn(usize, usize, usize) -> f32,
) -> Vec<f32> {
    let ocb = out_ch.div_ceil(v);
    let icb = in_ch_group.div_ceil(v);
    let _ = groups;
    let mut w = alloc::vec![0.0f32; ocb * icb * v * ntaps * v];
    for ob in 0..ocb {
        for ib in 0..icb {
            for vi in 0..v {
                let ic = ib * v + vi;
                if ic >= in_ch_group {
                    continue;
                }
                for t in 0..ntaps {
                    let base = (((ob * icb + ib) * v + vi) * ntaps + t) * v;
                    for vo in 0..v {
                        let oc = ob * v + vo;
                        if oc < out_ch {
                            w[base + vo] = get(oc, ic, t);
                        }
                    }
                }
            }
        }
    }
    w
}

fn pack_bias(v: usize, out_ch: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut b = alloc::vec![0.0f32; out_ch.div_ceil(v) * v];
    if let Some(src) = bias {
        b[..out_ch].copy_from_slice(src);
    }
    b
}

/// A stride-1 convolution packed for one block size.
#[derive(Clone, Debug)]
pub struct PackedConv {
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    /// Zero padding: top, left, bottom, right.
    pad: [usize; 4],
    groups: usize,
    v: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl PackedConv {
    /// Pack `conv` for block size `v`. `extra_pad` (`[top, left, bottom, right]`) is added to the
    /// layer's own symmetric padding (the SOP upsampler pads right/bottom by one).
    pub fn new(conv: &Conv2d, v: usize, extra_pad: [usize; 4]) -> Result<Self> {
        let (icg, ocg) = (conv.in_ch / conv.groups, conv.out_ch / conv.groups);
        if conv.stride != 1 {
            return Err(Error::Unsupported("fast path: strided convolution"));
        }
        if conv.kh * conv.kw > MAX_TAPS {
            return Err(Error::Unsupported(
                "fast path: kernel with more than 9 taps",
            ));
        }
        if conv.groups > 1 && (!icg.is_multiple_of(v) || !ocg.is_multiple_of(v)) {
            return Err(Error::Unsupported(
                "fast path: group size not a multiple of the block size",
            ));
        }
        let ntaps = conv.kh * conv.kw;
        let weight = if conv.groups == 1 {
            pack(v, conv.out_ch, conv.in_ch, ntaps, 1, |oc, ic, t| {
                conv.weight[(oc * conv.in_ch + ic) * ntaps + t]
            })
        } else {
            // Groups are block-aligned: pack every group on its own and concatenate.
            let mut w = Vec::new();
            for g in 0..conv.groups {
                w.extend(pack(v, ocg, icg, ntaps, 1, |oc, ic, t| {
                    conv.weight[((g * ocg + oc) * icg + ic) * ntaps + t]
                }));
            }
            w
        };
        Ok(Self {
            in_ch: conv.in_ch,
            out_ch: conv.out_ch,
            kh: conv.kh,
            kw: conv.kw,
            pad: [
                conv.pad_h + extra_pad[0],
                conv.pad_w + extra_pad[1],
                conv.pad_h + extra_pad[2],
                conv.pad_w + extra_pad[3],
            ],
            groups: conv.groups,
            v,
            weight,
            bias: pack_bias(v, conv.out_ch, conv.bias.as_deref()),
        })
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let v = self.v;
        if x.c != self.in_ch || x.v != v {
            return Err(Error::InvalidArgument(
                "conv: input channels / block size mismatch",
            ));
        }
        let [pt, pl, pb, pr] = self.pad;
        if x.h + pt + pb < self.kh || x.w + pl + pr < self.kw {
            return Err(Error::InvalidArgument(
                "conv: input smaller than the kernel",
            ));
        }
        let padded;
        let xp = if self.pad == [0; 4] {
            x
        } else {
            padded = x.pad(pt, pl, pb, pr)?;
            &padded
        };
        let (oh, ow) = (xp.h - self.kh + 1, xp.w - self.kw + 1);
        let mut out = BTensor::zeros(self.out_ch, oh, ow, v)?;
        let ocb_total = self.out_ch.div_ceil(v);
        let (icb_group, ocb_group) = if self.groups == 1 {
            (self.in_ch.div_ceil(v), ocb_total)
        } else {
            (self.in_ch / self.groups / v, self.out_ch / self.groups / v)
        };
        let ntaps = self.kh * self.kw;
        let wlen = icb_group * v * ntaps * v;
        let tier = eng.tier;
        for_each_row(eng, &mut out.data, ow * v, |idx, row| {
            let (ocb, oy) = (idx / oh, idx % oh);
            let group = ocb / ocb_group;
            let mut taps = [(0usize, 0usize); MAX_TAPS];
            for ky in 0..self.kh {
                for kx in 0..self.kw {
                    taps[ky * self.kw + kx] = (oy + ky, kx);
                }
            }
            let mut job = RowJob {
                out: row,
                o0: 0,
                os: 1,
                n: ow,
                xp: &xp.data,
                ph: xp.h,
                pw: xp.w,
                icb0: group * icb_group,
                icb1: (group + 1) * icb_group,
                in_ch: self.in_ch,
                taps: &taps[..ntaps],
                w: &self.weight[ocb * wlen..][..wlen],
                bias: &self.bias[ocb * v..][..v],
            };
            run_row(tier, &mut job);
        });
        Ok(out)
    }
}

/// A stride-2 transposed convolution packed for one block size.
#[derive(Clone, Debug)]
pub struct PackedConvTranspose {
    in_ch: usize,
    out_ch: usize,
    k: usize,
    pad: usize,
    out_pad: usize,
    v: usize,
    /// Per (row parity, column parity) of `ky`/`kx`: the taps' kernel coordinates, ascending,
    /// and their packed weights.
    phases: [[Phase; 2]; 2],
    bias: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
struct Phase {
    ky: Vec<usize>,
    kx: Vec<usize>,
    weight: Vec<f32>,
}

/// Input padding on every side: enough for every tap of every output position.
const EXT: usize = 2;

impl PackedConvTranspose {
    pub fn new(conv: &ConvTranspose2d, v: usize) -> Result<Self> {
        if conv.stride != 2 || conv.k > 4 || conv.k.div_ceil(2) + 1 > EXT + 1 {
            return Err(Error::Unsupported(
                "fast path: transposed convolution other than stride 2, k <= 4",
            ));
        }
        let k = conv.k;
        let mut phases: [[Phase; 2]; 2] = Default::default();
        for (pyp, row) in phases.iter_mut().enumerate() {
            for (pxp, phase) in row.iter_mut().enumerate() {
                phase.ky = (0..k).filter(|ky| ky % 2 == pyp).collect();
                phase.kx = (0..k).filter(|kx| kx % 2 == pxp).collect();
                let (kys, kxs) = (phase.ky.clone(), phase.kx.clone());
                let ntaps = kys.len() * kxs.len();
                if ntaps == 0 {
                    continue;
                }
                // PyTorch layout: [in][out][ky][kx]
                phase.weight = pack(v, conv.out_ch, conv.in_ch, ntaps, 1, |oc, ic, t| {
                    let (ky, kx) = (kys[t / kxs.len()], kxs[t % kxs.len()]);
                    conv.weight[((ic * conv.out_ch + oc) * k + ky) * k + kx]
                });
            }
        }
        Ok(Self {
            in_ch: conv.in_ch,
            out_ch: conv.out_ch,
            k,
            pad: conv.pad,
            out_pad: conv.out_pad,
            v,
            phases,
            bias: pack_bias(v, conv.out_ch, conv.bias.as_deref()),
        })
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let v = self.v;
        if x.c != self.in_ch || x.v != v {
            return Err(Error::InvalidArgument(
                "conv_transpose: input channels / block size mismatch",
            ));
        }
        if x.h == 0 || x.w == 0 {
            return Err(Error::InvalidArgument("conv_transpose: empty input"));
        }
        let oh = (x.h - 1) * 2 + self.k + self.out_pad - 2 * self.pad;
        let ow = (x.w - 1) * 2 + self.k + self.out_pad - 2 * self.pad;
        let xp = x.pad(EXT, EXT, EXT, EXT)?;
        let mut out = BTensor::zeros(self.out_ch, oh, ow, v)?;
        let icb = self.in_ch.div_ceil(v);
        let tier = eng.tier;
        let pad = self.pad as isize;
        for_each_row(eng, &mut out.data, ow * v, |idx, row| {
            let (ocb, oy) = (idx / oh, idx % oh);
            // A tap (ky, kx) reaches output (oy, ox) iff oy + pad - ky and ox + pad - kx are even;
            // it then reads input ((oy + pad - ky) / 2, (ox + pad - kx) / 2).
            let pyp = (oy + self.pad) % 2;
            for q in 0..2usize.min(ow) {
                let pxp = (q + self.pad) % 2;
                let phase = &self.phases[pyp][pxp];
                let ntaps = phase.ky.len() * phase.kx.len();
                let n = (ow - q).div_ceil(2);
                let wlen = icb * v * ntaps * v;
                let mut taps = [(0usize, 0usize); MAX_TAPS];
                for (a, &ky) in phase.ky.iter().enumerate() {
                    let iy = (oy as isize + pad - ky as isize).div_euclid(2) + EXT as isize;
                    for (b, &kx) in phase.kx.iter().enumerate() {
                        let ix = (q as isize + pad - kx as isize).div_euclid(2) + EXT as isize;
                        taps[a * phase.kx.len() + b] = (iy as usize, ix as usize);
                    }
                }
                let mut job = RowJob {
                    out: &mut *row,
                    o0: q,
                    os: 2,
                    n,
                    xp: &xp.data,
                    ph: xp.h,
                    pw: xp.w,
                    icb0: 0,
                    icb1: icb,
                    in_ch: self.in_ch,
                    taps: &taps[..ntaps],
                    w: if ntaps == 0 {
                        &[]
                    } else {
                        &phase.weight[ocb * wlen..][..wlen]
                    },
                    bias: &self.bias[ocb * v..][..v],
                };
                run_row(tier, &mut job);
            }
        });
        Ok(out)
    }
}
