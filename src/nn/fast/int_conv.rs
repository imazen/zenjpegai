//! Integer convolution for the hyper-scale decoder: int8 weights, inputs clamped to int8 range,
//! wrapping i32 accumulation.
//!
//! Integer addition is associative, so unlike the float kernels there is no evaluation order to
//! preserve: any arrangement is exact. Inputs and weights are held as i16 and input channels are
//! taken two at a time, so one `madd` (`pmaddwd` and friends: multiply i16 lanes, add adjacent
//! pairs into i32) performs `2 * V` multiply-accumulates. Products are at most 2^14 in magnitude,
//! so the pair sum cannot overflow its i32 lane; across taps the accumulator wraps like the
//! reference's int32 convolution.

// Lane-indexed loops read better than iterator chains in these kernels.
#![allow(clippy::needless_range_loop)]

use alloc::vec::Vec;

use archmage::prelude::*;
use magetypes::simd::backends::{I16x16Backend, I32x8Backend};
#[cfg(feature = "avx512")]
use magetypes::simd::backends::{I16x32Backend, I32x16Backend};
use magetypes::simd::generic::{i16x16, i32x8};
#[cfg(feature = "avx512")]
use magetypes::simd::generic::{i16x32, i32x16};

use super::{Engine, Tier, for_each_row};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// `V` i32 accumulator lanes fed by `V2 = 2 * V` i16 lanes.
trait SimdMadd<const V: usize, const V2: usize>: Copy {
    type Token: Copy + Send + Sync;
    type Acc: Copy;
    fn weights(t: Self::Token, w: &[i16; V2]) -> Self;
    /// `[x0, x1, x0, x1, ...]`.
    fn pair(t: Self::Token, x0: i16, x1: i16) -> Self;
    fn madd(self, o: Self) -> Self::Acc;
    fn acc_load(t: Self::Token, a: &[i32; V]) -> Self::Acc;
    fn acc_add(a: Self::Acc, b: Self::Acc) -> Self::Acc;
    fn acc_store(a: Self::Acc, out: &mut [i32; V]);
}

impl<T: I16x16Backend + I32x8Backend + Send + Sync> SimdMadd<8, 16> for i16x16<T> {
    type Token = T;
    type Acc = i32x8<T>;
    #[inline(always)]
    fn weights(t: T, w: &[i16; 16]) -> Self {
        i16x16::load(t, w)
    }
    #[inline(always)]
    fn pair(t: T, x0: i16, x1: i16) -> Self {
        let mut a = [x0; 16];
        for k in 0..8 {
            a[2 * k + 1] = x1;
        }
        i16x16::from_array(t, a)
    }
    #[inline(always)]
    fn madd(self, o: Self) -> i32x8<T> {
        self.madd_adjacent(o)
    }
    #[inline(always)]
    fn acc_load(t: T, a: &[i32; 8]) -> i32x8<T> {
        i32x8::load(t, a)
    }
    #[inline(always)]
    fn acc_add(a: i32x8<T>, b: i32x8<T>) -> i32x8<T> {
        a + b
    }
    #[inline(always)]
    fn acc_store(a: i32x8<T>, out: &mut [i32; 8]) {
        a.store(out)
    }
}

#[cfg(feature = "avx512")]
impl<T: I16x32Backend + I32x16Backend + Send + Sync> SimdMadd<16, 32> for i16x32<T> {
    type Token = T;
    type Acc = i32x16<T>;
    #[inline(always)]
    fn weights(t: T, w: &[i16; 32]) -> Self {
        i16x32::load(t, w)
    }
    #[inline(always)]
    fn pair(t: T, x0: i16, x1: i16) -> Self {
        let mut a = [x0; 32];
        for k in 0..16 {
            a[2 * k + 1] = x1;
        }
        i16x32::from_array(t, a)
    }
    #[inline(always)]
    fn madd(self, o: Self) -> i32x16<T> {
        self.madd_adjacent(o)
    }
    #[inline(always)]
    fn acc_load(t: T, a: &[i32; 16]) -> i32x16<T> {
        i32x16::load(t, a)
    }
    #[inline(always)]
    fn acc_add(a: i32x16<T>, b: i32x16<T>) -> i32x16<T> {
        a + b
    }
    #[inline(always)]
    fn acc_store(a: i32x16<T>, out: &mut [i32; 16]) {
        a.store(out)
    }
}

/// Scalar tier.
#[derive(Clone, Copy)]
struct ScalarMadd([i16; 16]);

impl SimdMadd<8, 16> for ScalarMadd {
    type Token = ScalarToken;
    type Acc = [i32; 8];
    #[inline(always)]
    fn weights(_: ScalarToken, w: &[i16; 16]) -> Self {
        Self(*w)
    }
    #[inline(always)]
    fn pair(_: ScalarToken, x0: i16, x1: i16) -> Self {
        let mut a = [x0; 16];
        for k in 0..8 {
            a[2 * k + 1] = x1;
        }
        Self(a)
    }
    #[inline(always)]
    fn madd(self, o: Self) -> [i32; 8] {
        let mut r = [0i32; 8];
        for k in 0..8 {
            let lo = self.0[2 * k] as i32 * o.0[2 * k] as i32;
            let hi = self.0[2 * k + 1] as i32 * o.0[2 * k + 1] as i32;
            r[k] = lo.wrapping_add(hi);
        }
        r
    }
    #[inline(always)]
    fn acc_load(_: ScalarToken, a: &[i32; 8]) -> [i32; 8] {
        *a
    }
    #[inline(always)]
    fn acc_add(a: [i32; 8], b: [i32; 8]) -> [i32; 8] {
        let mut r = [0i32; 8];
        for k in 0..8 {
            r[k] = a[k].wrapping_add(b[k]);
        }
        r
    }
    #[inline(always)]
    fn acc_store(a: [i32; 8], out: &mut [i32; 8]) {
        *out = a;
    }
}

struct IntRowJob<'a> {
    /// Output row of one output block: `width * V` i32.
    out: &'a mut [i32],
    /// Padded, clamped input, channel pairs interleaved: `[pair][ph][pw][2]` i16.
    xp: &'a [i16],
    ph: usize,
    pw: usize,
    pairs: usize,
    /// `(padded row, column offset)` per tap.
    taps: &'a [(usize, usize)],
    /// `[pair][tap][2V]` i16.
    w: &'a [i16],
    /// `V` i32.
    bias: &'a [i32],
}

/// `B` output positions accumulated over every `(channel pair, tap)`: the accumulators stay
/// in registers across the whole loop — each starts at the bias and is stored once — so an
/// output position's value passes through memory exactly once, at the store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn int_block<F: SimdMadd<V, V2>, const V: usize, const V2: usize, const B: usize>(
    t: F::Token,
    out: &mut [[i32; V]],
    taps: &[(usize, usize)],
    pairs: usize,
    ph: usize,
    pw: usize,
    xp: &[[i16; 2]],
    w: &[[i16; V2]],
    bias: F::Acc,
    j: usize,
) {
    let mut acc = [bias; B];
    // Fixed tap counts unroll the `(pair, tap)` sweep's inner loop — native only: the
    // unrolled body spills Cranelift's 16 v128 registers on wasm32 (same failure as the
    // float conv's ntaps==9 unroll, benchmarks/wasm_kernel_2026-09-18.md).
    #[cfg(target_arch = "wasm32")]
    int_taps_dyn::<F, V, V2, B>(t, &mut acc, taps, pairs, ph, pw, xp, w, j);
    #[cfg(not(target_arch = "wasm32"))]
    match taps.len() {
        1 => int_taps::<F, V, V2, B, 1>(
            t,
            &mut acc,
            taps.try_into().unwrap(),
            pairs,
            ph,
            pw,
            xp,
            w,
            j,
        ),
        9 => int_taps::<F, V, V2, B, 9>(
            t,
            &mut acc,
            taps.try_into().unwrap(),
            pairs,
            ph,
            pw,
            xp,
            w,
            j,
        ),
        _ => int_taps_dyn::<F, V, V2, B>(t, &mut acc, taps, pairs, ph, pw, xp, w, j),
    }
    for b in 0..B {
        F::acc_store(acc[b], &mut out[b]);
    }
}

/// The `(pair, tap)` sweep of [`int_block`], fixed tap count: the tap loop unrolls fully, so
/// each weight vector offset and tap entry is a constant. Native only — see [`int_block`].
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn int_taps<
    F: SimdMadd<V, V2>,
    const V: usize,
    const V2: usize,
    const B: usize,
    const NT: usize,
>(
    t: F::Token,
    acc: &mut [F::Acc; B],
    taps: &[(usize, usize); NT],
    pairs: usize,
    ph: usize,
    pw: usize,
    xp: &[[i16; 2]],
    w: &[[i16; V2]],
    j: usize,
) {
    for p in 0..pairs {
        let wp: &[[i16; V2]; NT] = w[p * NT..][..NT].try_into().unwrap();
        for k in 0..NT {
            let (y, dx) = taps[k];
            let r = &xp[(p * ph + y) * pw + dx + j..];
            // Always true (the padded input is wide enough); lets the optimizer drop the
            // `r[b]` bounds checks. The accumulators are indexed, not iterated — `iter_mut`
            // would force the array to memory.
            assert!(r.len() >= B);
            let wv = F::weights(t, &wp[k]);
            for b in 0..B {
                acc[b] = F::acc_add(acc[b], wv.madd(F::pair(t, r[b][0], r[b][1])));
            }
        }
    }
}

/// [`int_taps`] with a runtime tap count.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn int_taps_dyn<F: SimdMadd<V, V2>, const V: usize, const V2: usize, const B: usize>(
    t: F::Token,
    acc: &mut [F::Acc; B],
    taps: &[(usize, usize)],
    pairs: usize,
    ph: usize,
    pw: usize,
    xp: &[[i16; 2]],
    w: &[[i16; V2]],
    j: usize,
) {
    let ntaps = taps.len();
    for p in 0..pairs {
        for (wv, &(y, dx)) in w[p * ntaps..][..ntaps].iter().zip(taps) {
            let r = &xp[(p * ph + y) * pw + dx + j..];
            assert!(r.len() >= B);
            let wv = F::weights(t, wv);
            for b in 0..B {
                acc[b] = F::acc_add(acc[b], wv.madd(F::pair(t, r[b][0], r[b][1])));
            }
        }
    }
}

#[inline(always)]
fn int_row<F: SimdMadd<V, V2>, const V: usize, const V2: usize, const B: usize>(
    t: F::Token,
    job: &mut IntRowJob<'_>,
) {
    // Copy the `&'a` inputs and scalars out of `job` so the `out` chunk view can hold the
    // only borrow on it.
    let (taps, pairs, ph, pw) = (job.taps, job.pairs, job.ph, job.pw);
    let (out, _) = job.out.as_chunks_mut::<V>();
    let (xp, _) = job.xp.as_chunks::<2>();
    let (w, _) = job.w.as_chunks::<V2>();
    let n = out.len();
    let mut bias = [0i32; V];
    bias.copy_from_slice(&job.bias[..V]);
    let bias = F::acc_load(t, &bias);
    let mut j = 0;
    while j + B <= n {
        int_block::<F, V, V2, B>(t, &mut out[j..], taps, pairs, ph, pw, xp, w, bias, j);
        j += B;
    }
    if j < n && n >= B {
        // One overlapping block covers the tail: positions n - B..j are recomputed with the
        // same values and stored twice — cheap next to a cascade of B = 1 blocks, each of
        // which pays the full (channel pair, tap) sweep.
        int_block::<F, V, V2, B>(
            t,
            &mut out[n - B..],
            taps,
            pairs,
            ph,
            pw,
            xp,
            w,
            bias,
            n - B,
        );
        return;
    }
    while j + 8 <= n {
        int_block::<F, V, V2, 8>(t, &mut out[j..], taps, pairs, ph, pw, xp, w, bias, j);
        j += 8;
    }
    while j < n {
        int_block::<F, V, V2, 1>(t, &mut out[j..], taps, pairs, ph, pw, xp, w, bias, j);
        j += 1;
    }
}

#[cfg(feature = "avx512")]
#[arcane]
fn int_row_v4(t: X64V4Token, job: &mut IntRowJob<'_>) {
    int_row::<i16x32<X64V4Token>, 16, 32, 24>(t, job)
}

#[arcane]
fn int_row_v3(t: X64V3Token, job: &mut IntRowJob<'_>) {
    int_row::<i16x16<X64V3Token>, 8, 16, 6>(t, job)
}

#[arcane]
fn int_row_neon(t: NeonToken, job: &mut IntRowJob<'_>) {
    int_row::<i16x16<NeonToken>, 8, 16, 6>(t, job)
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn int_row_wasm128(t: Wasm128Token, job: &mut IntRowJob<'_>) {
    int_row::<i16x16<Wasm128Token>, 8, 16, 6>(t, job)
}

fn int_row_scalar(job: &mut IntRowJob<'_>) {
    int_row::<ScalarMadd, 8, 16, 2>(ScalarToken, job)
}

/// An int8 convolution (`Conv2di`), stride 1, zero padding `k / 2`, packed for one block size.
#[derive(Clone, Debug)]
pub struct PackedIntConv {
    in_ch: usize,
    out_ch: usize,
    k: usize,
    v: usize,
    /// `[ocb][pair][tap][2V]`.
    weight: Vec<i16>,
    /// `[ocb][V]`.
    bias: Vec<i32>,
    shift: Vec<u8>,
}

impl PackedIntConv {
    /// `weight` is `[out][in][k][k]`.
    pub fn new(
        out_ch: usize,
        in_ch: usize,
        k: usize,
        weight: &[i8],
        bias: &[i32],
        shift: &[u8],
        v: usize,
    ) -> Result<Self> {
        if weight.len() != out_ch * in_ch * k * k
            || bias.len() != out_ch
            || shift.len() != out_ch
            || k * k > 9
        {
            return Err(Error::Model(
                "int conv: inconsistent parameter sizes".into(),
            ));
        }
        let (ocb, pairs, ntaps) = (out_ch.div_ceil(v), in_ch.div_ceil(2), k * k);
        let mut w = alloc::vec![0i16; ocb * pairs * ntaps * 2 * v];
        for oc in 0..out_ch {
            let (ob, lane) = (oc / v, oc % v);
            for ic in 0..in_ch {
                let (p, half) = (ic / 2, ic % 2);
                for t in 0..ntaps {
                    let dst = ((ob * pairs + p) * ntaps + t) * 2 * v + 2 * lane + half;
                    w[dst] = weight[(oc * in_ch + ic) * ntaps + t] as i16;
                }
            }
        }
        let mut b = alloc::vec![0i32; ocb * v];
        b[..out_ch].copy_from_slice(bias);
        Ok(Self {
            in_ch,
            out_ch,
            k,
            v,
            weight: w,
            bias: b,
            shift: shift.to_vec(),
        })
    }

    /// `x` `[in, h, w]` (any i32; clamped to int8 range like the reference) → `[out, h, w]`:
    /// `(conv + bias) >> shift[oc]`, then `max(0, .)` if `relu`.
    pub fn forward(&self, eng: &Engine, x: &Tensor<i32>, relu: bool) -> Result<Tensor<i32>> {
        if x.c != self.in_ch || eng.tier.block() != self.v {
            return Err(Error::InvalidArgument(
                "int conv: input channels / block size mismatch",
            ));
        }
        let (h, w, k, v) = (x.h, x.w, self.k, self.v);
        let pad = k / 2;
        let (ph, pw) = (h + 2 * pad, w + 2 * pad);
        let pairs = self.in_ch.div_ceil(2);
        let mut xp = alloc::vec![0i16; pairs * ph * pw * 2];
        for c in 0..self.in_ch {
            let (p, half) = (c / 2, c % 2);
            for y in 0..h {
                let src = &x.data[(c * h + y) * w..][..w];
                let dst = &mut xp[((p * ph + y + pad) * pw + pad) * 2..][..w * 2];
                for (d, &s) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
                    d[half] = s.clamp(-128, 127) as i16;
                }
            }
        }
        let ocb = self.out_ch.div_ceil(v);
        let ntaps = k * k;
        let wlen = pairs * ntaps * 2 * v;
        let mut blocked = alloc::vec![0i32; ocb * h * w * v];
        let tier = eng.tier;
        for_each_row(eng, &mut blocked, w * v, |idx, row| {
            let (ob, oy) = (idx / h, idx % h);
            let mut taps = [(0usize, 0usize); 9];
            for ky in 0..k {
                for kx in 0..k {
                    taps[ky * k + kx] = (oy + ky, kx);
                }
            }
            let mut job = IntRowJob {
                out: row,
                xp: &xp,
                ph,
                pw,
                pairs,
                taps: &taps[..ntaps],
                w: &self.weight[ob * wlen..][..wlen],
                bias: &self.bias[ob * v..][..v],
            };
            match tier {
                #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
                Tier::V4(t) => int_row_v4(t, &mut job),
                #[cfg(target_arch = "x86_64")]
                Tier::V3(t) => int_row_v3(t, &mut job),
                #[cfg(target_arch = "aarch64")]
                Tier::Neon(t) => int_row_neon(t, &mut job),
                #[cfg(target_arch = "wasm32")]
                Tier::Wasm128(t) => int_row_wasm128(t, &mut job),
                Tier::Scalar => int_row_scalar(&mut job),
            }
        });
        let mut out = Tensor::<i32>::zeros(self.out_ch, h, w)?;
        for oc in 0..self.out_ch {
            let (ob, lane) = (oc / v, oc % v);
            let shift = self.shift[oc] as u32;
            let src = &blocked[ob * h * w * v..][..h * w * v];
            for (d, s) in out.plane_mut(oc).iter_mut().zip(src.chunks_exact(v)) {
                let val = s[lane] >> shift;
                *d = if relu { val.max(0) } else { val };
            }
        }
        Ok(out)
    }
}
