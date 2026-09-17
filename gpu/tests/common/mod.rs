//! Shared helpers of the `gpu-tests` integration tests.
#![allow(dead_code)]

use std::sync::OnceLock;

use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::{ContextOptions, GpuContext};

/// The adapter under test. The *caller* chooses it (see `just gpu-test`):
/// `ZENJPEGAI_GPU_ADAPTER` = substring of the adapter name, `ZENJPEGAI_GPU_ALLOW_SOFTWARE=1` to
/// accept llvmpipe and friends. No usable adapter is a hard failure, never a skip.
pub fn context() -> &'static GpuContext {
    static CTX: OnceLock<GpuContext> = OnceLock::new();
    CTX.get_or_init(|| {
        let opts = ContextOptions {
            adapter_name: std::env::var("ZENJPEGAI_GPU_ADAPTER").ok(),
            low_power: false,
            allow_software: std::env::var("ZENJPEGAI_GPU_ALLOW_SOFTWARE").is_ok_and(|v| v == "1"),
        };
        let ctx = GpuContext::new(&opts).expect("the gpu-tests feature needs a GPU adapter");
        let i = ctx.adapter_info();
        println!(
            "adapter: {} ({:?}, {:?}, driver {} {})",
            i.name, i.backend, i.device_type, i.driver, i.driver_info
        );
        ctx
    })
}

/// SplitMix64, seeded from a label so every case is reproducible on its own.
pub struct Rng(u64);

impl Rng {
    pub fn new(label: &str) -> Self {
        Self(label.bytes().fold(0x9e37_79b9_7f4a_7c15, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        }))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform in `[-scale, scale)`.
    pub fn f32(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
    pub fn tensor(&mut self, c: usize, h: usize, w: usize) -> Tensor<f32> {
        self.tensor_scaled(c, h, w, 1.0)
    }
    pub fn tensor_scaled(&mut self, c: usize, h: usize, w: usize, scale: f32) -> Tensor<f32> {
        Tensor::from_vec(c, h, w, self.vec(c * h * w, scale)).unwrap()
    }
}

pub fn max_abs(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

pub fn max_mag(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}
