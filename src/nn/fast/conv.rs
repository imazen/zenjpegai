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
    /// Input, blocked: `[icb - icb_base][ph][pw][V]`.
    pub xp: &'a [f32],
    pub ph: usize,
    pub pw: usize,
    /// Block index of the first block stored in `xp`.
    pub icb_base: usize,
    /// Input blocks to accumulate, and the total number of real input channels.
    pub icb0: usize,
    pub icb1: usize,
    pub in_ch: usize,
    /// Input column step per output position (the convolution stride; 1 or 2).
    pub is: usize,
    /// Per tap, in accumulation order: input row (`None`: a row of the zero padding), and the
    /// input column of output `j = 0`.
    pub taps: &'a [(Option<usize>, usize)],
    /// Weights of this output block: `[icb1 - icb0][V][taps][V]`.
    pub w: &'a [f32],
    /// Bias of this output block: `V` floats.
    pub bias: &'a [f32],
}

/// Zero fill for taps that fall in the padding: `sr` rows point here when the input row is
/// `None`. Long enough for every (B, S, V) instantiation: `B * S * V` is at most 55 * 16.
static ZERO_PAD: [f32; 1024] = [0.0; 1024];

/// The `(tap) x B` body for one input channel `v`: one weight-vector load per tap, each
/// followed by `B` splat-load / multiply / add steps. The always-true `assert` lets the
/// optimizer drop the `r[b * S]` bounds checks from the unrolled position loop; the
/// accumulators must be indexed, not iterated — `iter_mut` forces the array to memory.
#[inline(always)]
fn tap_loop<F: SimdF32<V>, const V: usize, const B: usize, const S: usize>(
    t: F::Token,
    acc: &mut [F; B],
    wrow: &[[f32; V]],
    sr: &[&[[f32; V]]; MAX_TAPS],
    v: usize,
) {
    for (wt, &r) in wrow.iter().zip(sr.iter()) {
        // One always-true assert tells the optimizer `b * S` is in bounds, so the unrolled
        // loop below needs no element checks.
        assert!(r.len() > (B - 1) * S);
        let wv = F::load(t, wt);
        for b in 0..B {
            // `v < V` always; the mask lets the compiler see it.
            acc[b] = wv.mul_add(F::splat(t, r[b * S][v & (V - 1)]), acc[b]);
        }
    }
}

/// [`tap_loop`] with a compile-time tap count: the tap loop unrolls fully, so each weight
/// vector offset and `sr` slot is a constant. `NT` covers every layer in the models (1x1
/// convs, the 2x2 phases of a stride-2 transposed 4x4, 3x3 convs). Native only — see the
/// `vin == V` dispatch in [`block`].
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn tap_loop_n<F: SimdF32<V>, const V: usize, const B: usize, const S: usize, const NT: usize>(
    t: F::Token,
    acc: &mut [F; B],
    wrow: &[[f32; V]; NT],
    sr: &[&[[f32; V]]; NT],
    v: usize,
) {
    for t_ in 0..NT {
        let (wt, r) = (&wrow[t_], sr[t_]);
        assert!(r.len() > (B - 1) * S);
        let wv = F::load(t, wt);
        for b in 0..B {
            acc[b] = wv.mul_add(F::splat(t, r[b * S][v & (V - 1)]), acc[b]);
        }
    }
}

/// `B` output positions accumulated over every input block: the accumulators stay in
/// registers across the whole `(icb, channel, tap)` loop — the bias goes straight into them
/// and they are stored once — so an output position's value passes through memory exactly
/// once, at the store.
///
/// Per input block the `ntaps` input rows are first cut down to the `L = (B-1)*S + 1`
/// positions the block reads; the `assert` inside the inner loop then lets the compiler drop
/// every `r[b * S]` bounds check, leaving one weight-vector load plus `B` splat-load /
/// multiply / add steps per `(channel, tap)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn block<F: SimdF32<V>, const V: usize, const B: usize, const S: usize>(
    t: F::Token,
    job: &mut RowJob<'_>,
    xp: &[[f32; V]],
    w: &[[f32; V]],
    bias: &[f32; V],
    j: usize,
) {
    let l = (B - 1) * S + 1;
    let ntaps = job.taps.len();
    let (pad, _) = ZERO_PAD[..l * V].as_chunks::<V>();
    // Interior blocks have every tap inside the input; the `Option` test is per-`j`-block
    // invariant, so hoist it out of the input-block loop.
    let interior = job.taps.iter().all(|t| t.0.is_some());
    let mut acc = [F::load(t, bias); B];
    if ntaps != 0 {
        for icb in job.icb0..job.icb1 {
            let vin = (job.in_ch - icb * V).min(V);
            let wblk = &w[(icb - job.icb0) * V * ntaps..][..vin * ntaps];
            let xbase = (icb - job.icb_base) * job.ph * job.pw;
            let mut sr = [pad; MAX_TAPS];
            if interior {
                for (s, &(y, dx)) in sr.iter_mut().zip(job.taps) {
                    let y = y.unwrap();
                    *s = &xp[xbase + y * job.pw + dx + j * S..][..l];
                }
            } else {
                for (s, &(y, dx)) in sr.iter_mut().zip(job.taps) {
                    if let Some(y) = y {
                        *s = &xp[xbase + y * job.pw + dx + j * S..][..l];
                    }
                }
            }
            // `vin == V` (every input block but possibly the last) gets a constant trip count:
            // the v-loop unrolls and each splat offset folds into the load instruction.
            // Constant tap counts (1x1, 2x2 convT phases, 3x3) unroll the tap loop too —
            // native only: on wasm32 the unrolled bodies spill Cranelift's 16 v128
            // registers and cost ~45 % of decode (the ntaps==9 rejection in
            // benchmarks/wasm_kernel_2026-09-18.md generalised), so wasm keeps the
            // dynamic tap loop.
            if vin == V {
                #[cfg(not(target_arch = "wasm32"))]
                match ntaps {
                    1 => {
                        for v in 0..V {
                            tap_loop_n::<F, V, B, S, 1>(
                                t,
                                &mut acc,
                                wblk[v..][..1].try_into().unwrap(),
                                sr[..1].try_into().unwrap(),
                                v,
                            );
                        }
                        continue;
                    }
                    4 => {
                        for v in 0..V {
                            tap_loop_n::<F, V, B, S, 4>(
                                t,
                                &mut acc,
                                wblk[v * 4..][..4].try_into().unwrap(),
                                sr[..4].try_into().unwrap(),
                                v,
                            );
                        }
                        continue;
                    }
                    9 => {
                        for v in 0..V {
                            tap_loop_n::<F, V, B, S, 9>(
                                t,
                                &mut acc,
                                wblk[v * 9..][..9].try_into().unwrap(),
                                sr[..9].try_into().unwrap(),
                                v,
                            );
                        }
                        continue;
                    }
                    _ => {}
                }
                for v in 0..V {
                    tap_loop::<F, V, B, S>(t, &mut acc, &wblk[v * ntaps..][..ntaps], &sr, v);
                }
            } else {
                for (v, wrow) in wblk.chunks_exact(ntaps).enumerate() {
                    tap_loop::<F, V, B, S>(t, &mut acc, wrow, &sr, v);
                }
            }
        }
    }
    let (out, _) = job.out.as_chunks_mut::<V>();
    for b in 0..B {
        acc[b].store(&mut out[job.o0 + (j + b) * job.os]);
    }
}

#[inline(always)]
fn conv_row<F: SimdF32<V>, const V: usize, const B: usize>(t: F::Token, job: &mut RowJob<'_>) {
    if job.is == 2 {
        conv_row_s::<F, V, B, 2>(t, job)
    } else {
        conv_row_s::<F, V, B, 1>(t, job)
    }
}

#[inline(always)]
fn conv_row_s<F: SimdF32<V>, const V: usize, const B: usize, const S: usize>(
    t: F::Token,
    job: &mut RowJob<'_>,
) {
    // Copy the `&'a` inputs out of `job` first so the chunk views don't hold a borrow on it.
    let (xpf, wf) = (job.xp, job.w);
    let (xp, _) = xpf.as_chunks::<V>();
    let (w, _) = wf.as_chunks::<V>();
    let mut bias = [0.0f32; V];
    bias.copy_from_slice(&job.bias[..V]);
    let n = job.n;
    if n == 0 {
        return;
    }
    let mut j = 0;
    while j + B <= n {
        block::<F, V, B, S>(t, job, xp, w, &bias, j);
        j += B;
    }
    if j < n && n >= B {
        // One overlapping block covers the tail: positions n - B..j are recomputed with the
        // same values and stored twice — cheap next to a cascade of narrower blocks, each of
        // which pays the full (input channel, tap) sweep.
        block::<F, V, B, S>(t, job, xp, w, &bias, n - B);
        return;
    }
    while j + 8 <= n {
        block::<F, V, 8, S>(t, job, xp, w, &bias, j);
        j += 8;
    }
    while j + 4 <= n {
        block::<F, V, 4, S>(t, job, xp, w, &bias, j);
        j += 4;
    }
    while j < n {
        block::<F, V, 1, S>(t, job, xp, w, &bias, j);
        j += 1;
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn conv_row_v4(t: X64V4Token, job: &mut RowJob<'_>) {
    // AVX-512 has 32 vector registers: B = 24 keeps 24 accumulators plus the weight vector
    // and broadcast temporaries resident (B = 27+ spills, see benchmarks/conv_kernels_*).
    conv_row::<f32x16<X64V4Token>, 16, 24>(t, job)
}

#[arcane]
fn conv_row_v3(t: X64V3Token, job: &mut RowJob<'_>) {
    // 16 ymm registers: B = 12 accumulators + weight + broadcast temporaries just fit
    // (B = 14 spills one accumulator per FMA, benchmarks/conv_kernels_*).
    conv_row::<f32x8<X64V3Token>, 8, 12>(t, job)
}

#[arcane]
fn conv_row_neon(t: NeonToken, job: &mut RowJob<'_>) {
    conv_row::<f32x8<NeonToken>, 8, 12>(t, job)
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn conv_row_wasm128(t: Wasm128Token, job: &mut RowJob<'_>) {
    // 16 v128 registers: 4 positions x 2 halves of accumulators, the weight vector pair and
    // the broadcast input; B = 5 / 6 measured slower (spills), see benchmarks/wasm_profile_*.
    conv_row::<f32x8<Wasm128Token>, 8, 4>(t, job)
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
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => conv_row_wasm128(t, job),
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

/// A convolution (stride 1 or 2) packed for one block size.
#[derive(Clone, Debug)]
pub struct PackedConv {
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    /// Zero padding: top, left, bottom, right.
    pad: [usize; 4],
    groups: usize,
    stride: usize,
    v: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl PackedConv {
    /// Pack `conv` for block size `v`. `extra_pad` (`[top, left, bottom, right]`) is added to the
    /// layer's own symmetric padding (the SOP upsampler pads right/bottom by one).
    pub fn new(conv: &Conv2d, v: usize, extra_pad: [usize; 4]) -> Result<Self> {
        let (icg, ocg) = (conv.in_ch / conv.groups, conv.out_ch / conv.groups);
        if conv.stride != 1 && conv.stride != 2 {
            return Err(Error::Unsupported("fast path: stride above 2"));
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
            stride: conv.stride,
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
        let st = self.stride;
        let (h, w) = (x.h, x.w);
        let (oh, ow) = (
            (h + pt + pb - self.kh) / st + 1,
            (w + pl + pr - self.kw) / st + 1,
        );
        let mut out = BTensor::scratch(self.out_ch, oh, ow, v)?;
        let ocb_total = self.out_ch.div_ceil(v);
        let (icb_group, ocb_group) = if self.groups == 1 {
            (self.in_ch.div_ceil(v), ocb_total)
        } else {
            (self.in_ch / self.groups / v, self.out_ch / self.groups / v)
        };
        let ntaps = self.kh * self.kw;
        let wlen = icb_group * v * ntaps * v;
        let tier = eng.tier;
        // Output columns whose whole kernel window lies inside the input read `x` in place; the
        // zero padding is never materialised for them. `j0..j1` is that range.
        let j0 = pl.div_ceil(st).min(ow);
        let j1 = if w + pl >= self.kw {
            ((w + pl - self.kw) / st + 1).clamp(j0, ow)
        } else {
            j0
        };
        for_each_row(eng, &mut out.data, ow * v, |idx, row| {
            let (ocb, oy) = (idx / oh, idx % oh);
            let group = ocb / ocb_group;
            let (icb0, icb1) = (group * icb_group, (group + 1) * icb_group);
            let wts = &self.weight[ocb * wlen..][..wlen];
            let bias = &self.bias[ocb * v..][..v];
            // Input row of kernel row `ky`, if it is not padding.
            let in_row = |ky: usize| (oy * st + ky).checked_sub(pt).filter(|&iy| iy < h);

            if j1 > j0 {
                let mut taps = [(None, 0usize); MAX_TAPS];
                for ky in 0..self.kh {
                    for kx in 0..self.kw {
                        taps[ky * self.kw + kx] = (in_row(ky), j0 * st + kx - pl);
                    }
                }
                run_row(
                    tier,
                    &mut RowJob {
                        out: &mut *row,
                        o0: j0,
                        os: 1,
                        n: j1 - j0,
                        xp: &x.data,
                        ph: h,
                        pw: w,
                        icb_base: 0,
                        icb0,
                        icb1,
                        in_ch: self.in_ch,
                        is: st,
                        taps: &taps[..ntaps],
                        w: wts,
                        bias,
                    },
                );
            }
            // Border columns: copy their (few) input windows into a small padded buffer.
            // The scratch is per-thread: every row needs one, and a fresh `vec![0.0; n]` per
            // row contends on the global allocator under `threads`. Both sides share one
            // buffer, so there is at most one TLS access per row.
            let bwl = if j0 > 0 { (j0 - 1) * st + self.kw } else { 0 };
            let bwr = if ow > j1 {
                (ow - j1 - 1) * st + self.kw
            } else {
                0
            };
            if bwl + bwr > 0 {
                let nblk = icb1 - icb0;
                border_buf(nblk * self.kh * (bwl + bwr) * v, |buf| {
                    for (side, (ja, jb)) in [(0, j0), (j1, ow)].into_iter().enumerate() {
                        if jb <= ja {
                            continue;
                        }
                        let bw = (jb - ja - 1) * st + self.kw;
                        let buf =
                            &mut buf[side * nblk * self.kh * bwl * v..][..nblk * self.kh * bw * v];
                        self.border_row(&x.data, h, w, in_row, icb0, icb1, ja, bw, buf);
                        let mut taps = [(None, 0usize); MAX_TAPS];
                        for ky in 0..self.kh {
                            for kx in 0..self.kw {
                                taps[ky * self.kw + kx] = (Some(ky), kx);
                            }
                        }
                        run_row(
                            tier,
                            &mut RowJob {
                                out: &mut *row,
                                o0: ja,
                                os: 1,
                                n: jb - ja,
                                xp: buf,
                                ph: self.kh,
                                pw: bw,
                                icb_base: icb0,
                                icb0,
                                icb1,
                                in_ch: self.in_ch,
                                is: st,
                                taps: &taps[..ntaps],
                                w: wts,
                                bias,
                            },
                        );
                    }
                });
            }
        });
        Ok(out)
    }

    /// Copy the `kh` input rows covering border output columns `ja..` into `buf` (zeroed):
    /// `[icb][kh][bw][V]`, padding columns left zero.
    #[allow(clippy::too_many_arguments)]
    fn border_row(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        in_row: impl Fn(usize) -> Option<usize>,
        icb0: usize,
        icb1: usize,
        ja: usize,
        bw: usize,
        buf: &mut [f32],
    ) {
        let (v, st, pl) = (self.v, self.stride, self.pad[1]);
        for icb in icb0..icb1 {
            for ky in 0..self.kh {
                let Some(iy) = in_row(ky) else { continue };
                let src = &x[(icb * h + iy) * w * v..][..w * v];
                let dst = &mut buf[((icb - icb0) * self.kh + ky) * bw * v..][..bw * v];
                // The in-range cells `c` (input column `ja * st + c - pl` in `0..w`) form one
                // contiguous run: copy it in a single slice copy. Cells before/after it are
                // the padding and stay zero.
                let c0 = pl.saturating_sub(ja * st);
                let c1 = (w + pl).saturating_sub(ja * st).min(bw);
                if c0 < c1 {
                    let ix0 = ja * st + c0 - pl;
                    dst[c0 * v..c1 * v].copy_from_slice(&src[ix0 * v..(ix0 + c1 - c0) * v]);
                }
            }
        }
    }
}

/// `f` runs with a zeroed scratch of `n` floats, reused per thread between calls.
fn border_buf(n: usize, f: impl FnOnce(&mut [f32])) {
    #[cfg(feature = "std")]
    {
        use std::cell::RefCell;
        thread_local! {
            static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            s.clear();
            s.resize(n, 0.0);
            f(s.as_mut_slice());
        });
    }
    #[cfg(not(feature = "std"))]
    f(&mut alloc::vec![0.0f32; n]);
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
        let xp = x.pad_par(eng, EXT, EXT, EXT, EXT)?;
        let mut out = BTensor::scratch(self.out_ch, oh, ow, v)?;
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
                let mut taps = [(None, 0usize); MAX_TAPS];
                for (a, &ky) in phase.ky.iter().enumerate() {
                    let iy = (oy as isize + pad - ky as isize).div_euclid(2) + EXT as isize;
                    for (b, &kx) in phase.kx.iter().enumerate() {
                        let ix = (q as isize + pad - kx as isize).div_euclid(2) + EXT as isize;
                        taps[a * phase.kx.len() + b] = (Some(iy as usize), ix as usize);
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
                    icb_base: 0,
                    icb0: 0,
                    icb1: icb,
                    in_ch: self.in_ch,
                    is: 1,
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
