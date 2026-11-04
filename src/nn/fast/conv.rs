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

/// Winograd F(2x2, 3x3) for `Precision::Approximate` (opt-in, parity-breaking).
///
/// The 3x3 filter is pre-transformed to the 16 Winograd planes `u = G g G^T` at
/// pack time. At run time each output 2x2 tile's 4x4 input patch is transformed
/// `v = B^T d B`, the 16 planes each run one element-wise GEMM over input
/// channels `m[ξν] = Σ u[ξν] * v[ξν]` (16 multiplies per 2x2 output vs 36 for
/// the direct conv — 2.25x fewer), and `y = A^T m A` gives the tile.
///
/// The transforms are add/sub only and the GEMM accumulates over (icb, icv) in
/// a fixed order with `mul_add`, so every tier and thread count produces
/// identical output — but a different rounding than the direct conv.
const WINO_PLANES: usize = 16;

/// `B` tiles' `m[ξν]` accumulated over the `V` input-channel lanes of one input
/// block: `acc[b] += u[ξν][icv][..V_oc] * v[ξν][tile_b][icv]` — the Winograd
/// GEMM, the same op shape as [`tap_loop`] with the input channel varying per
/// step instead of fixed.
#[inline(always)]
fn wino_loop<F: SimdF32<V>, const V: usize, const B: usize>(
    t: F::Token,
    acc: &mut [F; B],
    wrow: &[[f32; V]], // [V_ic][V_oc] — `u[ξν]` for one input block
    xr: &[&[f32; V]],  // [B tiles][V_ic] — `v[ξν][tile]`
) {
    // `wrow` is always `V` rows; the indexed loop unrolls like `tap_loop_n`.
    for icv in 0..V {
        let wv = F::load(t, &wrow[icv]);
        for b in 0..B {
            acc[b] = wv.mul_add(F::splat(t, xr[b][icv]), acc[b]);
        }
    }
}

/// `G g G^T` of one 3x3 kernel — the 16 Winograd filter planes. `G` rows are
/// `(1,0,0)`, `(1/2,1/2,1/2)`, `(1/2,-1/2,1/2)`, `(0,0,1)`; `g` is row-major.
fn wino_filter(g: &[f32]) -> [f32; WINO_PLANES] {
    // `m = G g` (4x3): row `r` of `m` is `G[r]` dot the rows of `g`.
    let mut m = [[0.0f32; 3]; 4];
    for c in 0..3 {
        m[0][c] = g[c];
        m[1][c] = (g[c] + g[3 + c] + g[6 + c]) * 0.5;
        m[2][c] = (g[c] - g[3 + c] + g[6 + c]) * 0.5;
        m[3][c] = g[6 + c];
    }
    // `u = m G^T` (4x4): column `j` of `u` is `m` dot `G[j]`.
    let mut u = [0.0f32; WINO_PLANES];
    for r in 0..4 {
        u[r * 4] = m[r][0];
        u[r * 4 + 1] = (m[r][0] + m[r][1] + m[r][2]) * 0.5;
        u[r * 4 + 2] = (m[r][0] - m[r][1] + m[r][2]) * 0.5;
        u[r * 4 + 3] = m[r][2];
    }
    u
}

/// `B^T d B` of one tile's 4x4 input patch — the 16 Winograd input planes,
/// one `V`-lane vector each. `B^T` rows are `(1,0,-1,0)`, `(0,1,1,0)`,
/// `(0,-1,1,0)`, `(0,1,0,-1)`; `rows[r]` is the patch's four input cells of
/// row `r` (padding cells already zero — callers pass either a direct `x`
/// slice or a zero-padded gather).
fn wino_input<F: SimdF32<V>, const V: usize>(
    t: F::Token,
    rows: &[&[[f32; V]]; 4],
    dst: &mut [[f32; V]; WINO_PLANES],
) {
    // `tt = B^T d`: column `c` of `tt` is `B^T` applied to column `c` of `d`.
    let mut tt = [[F::splat(t, 0.0); 4]; 4];
    for (c, tt_c) in tt.iter_mut().enumerate().take(4) {
        let (d0, d1, d2, d3) = (
            F::load(t, &rows[0][c]),
            F::load(t, &rows[1][c]),
            F::load(t, &rows[2][c]),
            F::load(t, &rows[3][c]),
        );
        tt_c[0] = d0.sub(d2);
        tt_c[1] = d1.add(d2);
        tt_c[2] = d2.sub(d1);
        tt_c[3] = d1.sub(d3);
    }
    // `v = tt B`: row `r` of `v` is `tt[r]` applied to `B`.
    for r in 0..4 {
        let (t0, t1, t2, t3) = (tt[0][r], tt[1][r], tt[2][r], tt[3][r]);
        t0.sub(t2).store(&mut dst[r * 4]);
        t1.add(t2).store(&mut dst[r * 4 + 1]);
        t2.sub(t1).store(&mut dst[r * 4 + 2]);
        t1.sub(t3).store(&mut dst[r * 4 + 3]);
    }
}

/// `A^T m A + bias` of one tile's 4x4 `m` — the 2x2 output. `A^T` rows are
/// `(1,1,1,0)`, `(0,1,-1,-1)`; `m[e][v]` is the 16 GEMM planes.
fn wino_output<F: SimdF32<V>, const V: usize>(
    t: F::Token,
    m: &[[f32; V]; WINO_PLANES],
    bias: &[f32; V],
    y: &mut [[f32; V]; 4],
) {
    // `tt = A^T m` is 2x4: `tt[c]` = col `c` of row 0, `tt[4+c]` = col `c` of row 1.
    let mut tt = [F::splat(t, 0.0); 8];
    for c in 0..4 {
        let (m0, m1, m2, m3) = (
            F::load(t, &m[c]),
            F::load(t, &m[4 + c]),
            F::load(t, &m[8 + c]),
            F::load(t, &m[12 + c]),
        );
        tt[c] = m0.add(m1).add(m2);
        tt[4 + c] = m1.sub(m2).sub(m3);
    }
    let bias = F::load(t, bias);
    tt[0].add(tt[1]).add(tt[2]).add(bias).store(&mut y[0]);
    tt[1].sub(tt[2]).sub(tt[3]).add(bias).store(&mut y[1]);
    tt[4].add(tt[5]).add(tt[6]).add(bias).store(&mut y[2]);
    tt[5].sub(tt[6]).sub(tt[7]).add(bias).store(&mut y[3]);
}

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
    /// Winograd filter planes `[ocb][icb][ξν][V_ic][V_oc]` — packed only for
    /// 3x3 stride-1 dense convs (the [`Precision::Approximate`](super::Precision)
    /// path); empty otherwise.
    w16: Vec<f32>,
    bias: Vec<f32>,
    /// `weight` + `bias` bytes in the tracked ledger (see [`crate::mem`]).
    _charge: crate::mem::Charge,
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
        let bias = pack_bias(v, conv.out_ch, conv.bias.as_deref());
        // Winograd filter planes for the `Approximate` path — only the dense
        // 3x3 stride-1 convs (the hot case) carry the 16/9-larger transform.
        let w16 = if conv.kh == 3 && conv.kw == 3 && conv.stride == 1 && conv.groups == 1 {
            let icbg = conv.in_ch.div_ceil(v);
            let ocbg = conv.out_ch.div_ceil(v);
            let mut w16 = alloc::vec![0.0f32; ocbg * icbg * WINO_PLANES * v * v];
            for oc in 0..conv.out_ch {
                for ic in 0..conv.in_ch {
                    let u = wino_filter(&conv.weight[(oc * conv.in_ch + ic) * 9..][..9]);
                    let (ocb, icb, icv, ocv) = (oc / v, ic / v, ic % v, oc % v);
                    let base = (ocb * icbg + icb) * WINO_PLANES * v * v;
                    for e in 0..WINO_PLANES {
                        w16[base + (e * v + icv) * v + ocv] = u[e];
                    }
                }
            }
            w16
        } else {
            Vec::new()
        };
        let charge = crate::mem::Charge::new(
            crate::mem::vec_bytes(&weight)
                + crate::mem::vec_bytes(&w16)
                + crate::mem::vec_bytes(&bias),
        );
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
            w16,
            bias,
            _charge: charge,
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
        // The Winograd path handles the whole 3x3 stride-1 dense conv — opt-in
        // via `Precision::Approximate` (different rounding than the direct
        // kernel, so never the default).
        if eng.precision == super::Precision::Approximate && !self.w16.is_empty() {
            return self.forward_winograd(eng, x, oh, ow);
        }
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

    /// The Winograd F(2x2, 3x3) forward — `Precision::Approximate` only. Only
    /// reachable for dense 3x3 stride-1 convs (`self.w16` is packed for those
    /// alone). The output rows are processed in strips of two (one tile row
    /// each); every strip transforms its tiles once into `v16`, then each
    /// output block runs the 16 element-wise GEMMs and the inverse transform.
    fn forward_winograd(&self, eng: &Engine, x: &BTensor, oh: usize, ow: usize) -> Result<BTensor> {
        let v = self.v;
        let (h, w) = (x.h, x.w);
        let [pt, pl, _, _] = self.pad;
        let (tw, strips) = (ow.div_ceil(2), oh.div_ceil(2));
        let icbg = self.in_ch.div_ceil(v);
        let ocb_total = self.out_ch.div_ceil(v);
        let mut out = BTensor::scratch(self.out_ch, oh, ow, v)?;
        // Split `out.data` into per-strip units: `units[sy]` is every output
        // block's 2-row slice for strip `sy` (one row at the bottom edge).
        let mut units: Vec<Vec<(usize, &mut [f32])>> =
            (0..strips).map(|_| Vec::with_capacity(ocb_total)).collect();
        let mut rest: &mut [f32] = &mut out.data;
        for ocb in 0..ocb_total {
            let (d, r) = rest.split_at_mut(oh * ow * v);
            rest = r;
            let mut d = d;
            for (sy, slices) in units.iter_mut().enumerate() {
                let len = (oh - 2 * sy).min(2) * ow * v;
                let (s, rr) = d.split_at_mut(len);
                d = rr;
                slices.push((ocb, s));
            }
        }
        let w16 = &self.w16;
        let bias = &self.bias;
        let run = |sy: usize, slices: &mut Vec<(usize, &mut [f32])>| {
            run_wino(
                eng.tier,
                &mut WinoJob {
                    x: &x.data,
                    h,
                    w,
                    pt,
                    pl,
                    oh,
                    ow,
                    tw,
                    icbg,
                    v,
                    w16,
                    bias,
                    sy,
                    slices,
                },
            );
        };
        #[cfg(feature = "parallel")]
        if eng.parallel {
            use rayon::prelude::*;
            units
                .par_iter_mut()
                .enumerate()
                .for_each(|(sy, s)| run(sy, s));
            return Ok(out);
        }
        for (sy, s) in units.iter_mut().enumerate() {
            run(sy, s);
        }
        Ok(out)
    }
}

/// One strip's work in the Winograd path.
struct WinoJob<'a, 'b> {
    /// The input, blocked: `[icb][h][w][V]`.
    x: &'a [f32],
    h: usize,
    w: usize,
    /// Padding: top, left (bottom/right only change the output size).
    pt: usize,
    pl: usize,
    oh: usize,
    ow: usize,
    /// Tiles per strip: `ceil(ow / 2)`.
    tw: usize,
    /// Input blocks (`groups` is 1 here).
    icbg: usize,
    v: usize,
    /// `w16` and `bias`, packed (`[ocb][icb][ξν][V_ic][V_oc]` and `[ocb][V]`).
    w16: &'a [f32],
    bias: &'a [f32],
    /// The strip index: output rows `2 * sy`, `2 * sy + 1`.
    sy: usize,
    /// `(ocb, 2-row out slice)` for every output block in this strip — the
    /// inner `&'b` is the output tensor's own lifetime, longer than the `&'a`
    /// borrow of this strip's slice list.
    slices: &'a mut [(usize, &'b mut [f32])],
}

/// One strip of the Winograd forward: transform this strip's input tiles into
/// `v16` once, then for every output block run the 16 `m[ξν]` GEMMs (each a
/// `wino_loop` over the input blocks) and inverse-transform to the 2x2 output.
#[allow(clippy::needless_range_loop)]
fn wino_strip<F: SimdF32<V>, const V: usize, const B: usize>(
    t: F::Token,
    job: &mut WinoJob<'_, '_>,
) {
    let (v, tw, icbg) = (job.v, job.tw, job.icbg);
    let vlen = icbg * tw * WINO_PLANES;
    let mlen = tw * WINO_PLANES;
    // `scratch_buf`, not `border_buf`: both buffers are fully overwritten every
    // call, so the zero-fill is pure waste (a memset of ~MBs per strip).
    scratch_buf((vlen + mlen) * v, |buf| {
        let (v16, m16) = buf.split_at_mut(vlen * v);
        let (v16, _) = v16.as_chunks_mut::<V>();
        let (m16, _) = m16.as_chunks_mut::<V>();
        let (x, _) = job.x.as_chunks::<V>();
        let (row0, nrows) = (2 * job.sy, (job.oh - 2 * job.sy).min(2));
        // Transform every input-channel block's tiles of this strip into
        // `v16[icb][tile][ξν][V]`.
        for tx in 0..tw {
            let col0 = 2 * tx;
            // Interior tiles (the whole 4x4 patch inside the input) read `x`
            // in place — no gather copy. Only the border tiles take the
            // zero-padded `d` gather.
            let interior = col0 >= job.pl
                && col0 + 4 - job.pl <= job.w
                && row0 >= job.pt
                && row0 + 4 - job.pt <= job.h;
            for icb in 0..icbg {
                let dst: &mut [[f32; V]; WINO_PLANES] = (&mut v16[(icb * tw + tx) * WINO_PLANES..]
                    [..WINO_PLANES])
                    .try_into()
                    .unwrap();
                if interior {
                    let mut rows: [&[[f32; V]]; 4] = [&x[..1]; 4];
                    for (r, rr) in rows.iter_mut().enumerate() {
                        let iy = row0 + r - job.pt;
                        let xrow = &x[(icb * job.h + iy) * job.w..][..job.w];
                        *rr = &xrow[col0 - job.pl..col0 - job.pl + 4];
                    }
                    wino_input::<F, V>(t, &rows, dst);
                } else {
                    // The 4x4 patch covering output rows `row0..row0+2` /
                    // columns `col0..col0+2`, zero outside the input.
                    let mut d = [[[0.0f32; V]; 4]; 4];
                    for r in 0..4 {
                        let Some(iy) = (row0 + r).checked_sub(job.pt).filter(|&y| y < job.h) else {
                            continue;
                        };
                        let xrow = &x[(icb * job.h + iy) * job.w..][..job.w];
                        for c in 0..4 {
                            if let Some(ix) =
                                (col0 + c).checked_sub(job.pl).filter(|&ix| ix < job.w)
                            {
                                d[r][c] = xrow[ix];
                            }
                        }
                    }
                    let rows: [&[[f32; V]]; 4] = [&d[0], &d[1], &d[2], &d[3]];
                    wino_input::<F, V>(t, &rows, dst);
                }
            }
        }
        // Per output block: the 16 `m[ξν]` GEMMs, then the inverse transform.
        for (ocb, oslice) in job.slices.iter_mut() {
            let ocb = *ocb;
            let (w16c, _) = job.w16[ocb * icbg * WINO_PLANES * v * v..]
                [..icbg * WINO_PLANES * v * v]
                .as_chunks::<V>();
            let bias: &[f32; V] = job.bias[ocb * v..][..v].try_into().unwrap();
            for e in 0..WINO_PLANES {
                let mut tc = 0;
                while tc < tw {
                    let n = (tw - tc).min(B);
                    let mut acc = [F::splat(t, 0.0); B];
                    wino_block::<F, V, B>(t, &mut acc, w16c, v16, icbg, tw, e, tc, n);
                    for b in 0..n {
                        acc[b].store(&mut m16[(tc + b) * WINO_PLANES + e]);
                    }
                    tc += n;
                }
            }
            for tx in 0..tw {
                let mut y = [[0.0f32; V]; 4];
                wino_output::<F, V>(
                    t,
                    (&m16[tx * WINO_PLANES..][..WINO_PLANES])
                        .try_into()
                        .unwrap(),
                    bias,
                    &mut y,
                );
                let col0 = 2 * tx;
                for r in 0..nrows {
                    let orow = &mut oslice[r * job.ow * v..];
                    for c in 0..2 {
                        if col0 + c < job.ow {
                            orow[(col0 + c) * v..][..v].copy_from_slice(&y[r * 2 + c]);
                        }
                    }
                }
            }
        }
    });
}

/// `B` tiles' `m[ξν]` (plane `e`) accumulated over every input block and input
/// channel lane — the Winograd GEMM body.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn wino_block<F: SimdF32<V>, const V: usize, const B: usize>(
    t: F::Token,
    acc: &mut [F; B],
    w16c: &[[f32; V]], // `[icb][ξν][icv][V_oc]` as V-chunks
    v16: &[[f32; V]],  // `[icb][tile][ξν][V_ic]`
    icbg: usize,
    tw: usize,
    e: usize,
    tc: usize,
    n: usize,
) {
    for icb in 0..icbg {
        // `wrow[icv]` is the `V_oc` weight vector for (ξν = e, icv).
        let wrow = &w16c[(icb * WINO_PLANES + e) * V..][..V];
        let mut xr: [&[f32; V]; B] = [&v16[0]; B];
        for (b, xr_b) in xr.iter_mut().enumerate().take(n) {
            *xr_b = &v16[(icb * tw + tc + b) * WINO_PLANES + e];
        }
        wino_loop::<F, V, B>(t, acc, wrow, &xr);
    }
}

fn run_wino(tier: Tier, job: &mut WinoJob<'_, '_>) {
    match tier {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(t) => wino_v4(t, job),
        #[cfg(target_arch = "x86_64")]
        Tier::V3(t) => wino_v3(t, job),
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(t) => wino_neon(t, job),
        #[cfg(target_arch = "wasm32")]
        Tier::Wasm128(t) => wino_wasm(t, job),
        Tier::Scalar => wino_scalar(job),
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane]
fn wino_v4(t: X64V4Token, job: &mut WinoJob<'_, '_>) {
    wino_strip::<f32x16<X64V4Token>, 16, 24>(t, job)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn wino_v3(t: X64V3Token, job: &mut WinoJob<'_, '_>) {
    wino_strip::<f32x8<X64V3Token>, 8, 12>(t, job)
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn wino_neon(t: NeonToken, job: &mut WinoJob<'_, '_>) {
    wino_strip::<f32x8<NeonToken>, 8, 12>(t, job)
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn wino_wasm(t: Wasm128Token, job: &mut WinoJob<'_, '_>) {
    wino_strip::<f32x8<Wasm128Token>, 8, 4>(t, job)
}

fn wino_scalar(job: &mut WinoJob<'_, '_>) {
    wino_strip::<Lanes8, 8, 4>(ScalarToken, job)
}

/// `f` runs with a scratch of `n` floats, reused per thread between calls —
/// contents are *not* initialised (grows zeroed once, then keeps old values).
/// For buffers that are fully overwritten before any read, like the Winograd
/// transform planes, where `border_buf`'s memset would be pure waste.
fn scratch_buf(n: usize, f: impl FnOnce(&mut [f32])) {
    #[cfg(feature = "std")]
    {
        use std::cell::RefCell;
        thread_local! {
            /// The scratch and the ledger token for its capacity; both die at thread exit.
            static SCRATCH: RefCell<(Vec<f32>, crate::mem::Charge)> =
                const { RefCell::new((Vec::new(), crate::mem::Charge::EMPTY)) };
        }
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.0.len() < n {
                s.0.resize(n, 0.0);
                let bytes = crate::mem::vec_bytes(&s.0);
                s.1.resize(bytes);
            }
            f(&mut s.0[..n]);
        });
    }
    #[cfg(not(feature = "std"))]
    f(&mut alloc::vec![0.0f32; n]);
}

/// `f` runs with a zeroed scratch of `n` floats, reused per thread between calls.
fn border_buf(n: usize, f: impl FnOnce(&mut [f32])) {
    #[cfg(feature = "std")]
    {
        use std::cell::RefCell;
        thread_local! {
            /// The scratch and the ledger token for its capacity; both die at thread exit.
            static SCRATCH: RefCell<(Vec<f32>, crate::mem::Charge)> =
                const { RefCell::new((Vec::new(), crate::mem::Charge::EMPTY)) };
        }
        SCRATCH.with(|s| {
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
    /// `phases` + `bias` bytes in the tracked ledger (see [`crate::mem`]).
    _charge: crate::mem::Charge,
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
        let bias = pack_bias(v, conv.out_ch, conv.bias.as_deref());
        let phase_bytes: usize = phases
            .iter()
            .flatten()
            .map(|p| {
                crate::mem::vec_bytes(&p.ky)
                    + crate::mem::vec_bytes(&p.kx)
                    + crate::mem::vec_bytes(&p.weight)
            })
            .sum();
        let charge = crate::mem::Charge::new(phase_bytes + crate::mem::vec_bytes(&bias));
        Ok(Self {
            in_ch: conv.in_ch,
            out_ch: conv.out_ch,
            k,
            pad: conv.pad,
            out_pad: conv.out_pad,
            v,
            phases,
            bias,
            _charge: charge,
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
