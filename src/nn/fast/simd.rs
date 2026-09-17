//! The vector abstraction the kernels are written against.
//!
//! `V` lanes of `f32`: magetypes' `f32x16` (AVX-512), `f32x8` (AVX2, or 2x NEON), or a plain
//! array whose lanes use `f32::mul_add` (the scalar tier). Every method is `#[inline(always)]`
//! so that the generic kernels inherit the `#[target_feature]` context of their entry point.

// Lane-indexed loops read better than iterator chains in these kernels.
#![allow(clippy::needless_range_loop)]

use archmage::ScalarToken;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8, f32x16};

pub(crate) trait SimdF32<const V: usize>: Copy {
    type Token: Copy + Send + Sync;
    fn splat(t: Self::Token, x: f32) -> Self;
    fn load(t: Self::Token, a: &[f32; V]) -> Self;
    fn store(self, a: &mut [f32; V]);
    /// `self * a + b`, fused (one rounding).
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl<T: F32x8Backend + Send + Sync> SimdF32<8> for f32x8<T> {
    type Token = T;
    #[inline(always)]
    fn splat(t: T, x: f32) -> Self {
        f32x8::splat(t, x)
    }
    #[inline(always)]
    fn load(t: T, a: &[f32; 8]) -> Self {
        f32x8::load(t, a)
    }
    #[inline(always)]
    fn store(self, a: &mut [f32; 8]) {
        f32x8::store(self, a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x8::mul_add(self, a, b)
    }
}

impl<T: F32x16Backend + Send + Sync> SimdF32<16> for f32x16<T> {
    type Token = T;
    #[inline(always)]
    fn splat(t: T, x: f32) -> Self {
        f32x16::splat(t, x)
    }
    #[inline(always)]
    fn load(t: T, a: &[f32; 16]) -> Self {
        f32x16::load(t, a)
    }
    #[inline(always)]
    fn store(self, a: &mut [f32; 16]) {
        f32x16::store(self, a)
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32x16::mul_add(self, a, b)
    }
}

/// Scalar tier: eight independent lanes, fused multiply-add through `f32::mul_add` (hardware
/// FMA when the build enables it, libm's exact `fma` otherwise). Deliberately *not* magetypes'
/// scalar backend, whose `mul_add` is an unfused multiply and add.
#[derive(Clone, Copy)]
pub(crate) struct Lanes8([f32; 8]);

impl SimdF32<8> for Lanes8 {
    type Token = ScalarToken;
    #[inline(always)]
    fn splat(_: ScalarToken, x: f32) -> Self {
        Self([x; 8])
    }
    #[inline(always)]
    fn load(_: ScalarToken, a: &[f32; 8]) -> Self {
        Self(*a)
    }
    #[inline(always)]
    fn store(self, a: &mut [f32; 8]) {
        *a = self.0;
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        let mut r = [0.0; 8];
        for i in 0..8 {
            r[i] = self.0[i].mul_add(a.0[i], b.0[i]);
        }
        Self(r)
    }
}
