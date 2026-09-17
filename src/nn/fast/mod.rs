//! Optimised layers: channel-blocked tensors, SIMD micro-kernels, row-level threading.
//!
//! Bit-identical to [`crate::nn::reference`] by construction (see the numeric contract in
//! [`crate::nn`]); `tests/fast_vs_reference.rs` checks it on every tier the machine has.

mod conv;
mod int_conv;
mod layers;
mod simd;
mod tensor;

use archmage::prelude::*;

pub(crate) use conv::for_each_row;
pub use conv::{PackedConv, PackedConvTranspose};
pub use int_conv::PackedIntConv;
pub use layers::{
    ConvLayer, ConvTransposeLayer, add_assign, gate, pixel_shuffle, pixel_shuffle_to_planar, relu,
    relu6,
};
pub use tensor::BTensor;

/// SIMD tier, resolved once per engine.
#[derive(Clone, Copy, Debug)]
pub enum Tier {
    /// AVX-512 (x86-64-v4), 16 lanes.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    V4(X64V4Token),
    /// AVX2 + FMA (x86-64-v3), 8 lanes.
    #[cfg(target_arch = "x86_64")]
    V3(X64V3Token),
    /// NEON, 8 lanes as two 128-bit registers.
    #[cfg(target_arch = "aarch64")]
    Neon(NeonToken),
    /// No SIMD: 8 scalar lanes with `f32::mul_add`.
    Scalar,
}

impl Tier {
    /// Best tier this CPU supports.
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = X64V4Token::summon() {
            return Tier::V4(t);
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = X64V3Token::summon() {
            return Tier::V3(t);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = NeonToken::summon() {
            return Tier::Neon(t);
        }
        Tier::Scalar
    }

    /// Every tier this CPU supports, best first (for tests and benchmarks).
    pub fn available() -> alloc::vec::Vec<Self> {
        let mut tiers = alloc::vec::Vec::new();
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = X64V4Token::summon() {
            tiers.push(Tier::V4(t));
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = X64V3Token::summon() {
            tiers.push(Tier::V3(t));
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(t) = NeonToken::summon() {
            tiers.push(Tier::Neon(t));
        }
        tiers.push(Tier::Scalar);
        tiers
    }

    /// Channel block size (SIMD lanes).
    pub fn block(self) -> usize {
        match self {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Tier::V4(_) => 16,
            _ => 8,
        }
    }
}

/// Execution context of the fast layers.
#[derive(Clone, Copy, Debug)]
pub struct Engine {
    pub tier: Tier,
    /// Spread rows over the rayon pool (needs the `parallel` feature).
    pub parallel: bool,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            tier: Tier::detect(),
            parallel: cfg!(feature = "parallel"),
        }
    }

    pub fn with(tier: Tier, parallel: bool) -> Self {
        Self { tier, parallel }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}
