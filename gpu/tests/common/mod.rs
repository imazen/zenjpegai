//! Shared helpers of the `gpu-tests` integration tests.
#![allow(dead_code)]

use std::sync::OnceLock;

use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::{ContextOptions, GpuContext};

/// The adapter under test. The *caller* chooses it (see `just gpu-test`):
/// `ZENJPEGAI_GPU_ADAPTER` = substring of the adapter name, `ZENJPEGAI_GPU_ALLOW_SOFTWARE=1` to
/// accept llvmpipe and friends. No usable adapter is a hard failure, never a skip.
pub fn context() -> &'static GpuContext {
    context_arc_ref()
}

pub fn context_arc() -> std::sync::Arc<GpuContext> {
    context_arc_ref().clone()
}

fn context_arc_ref() -> &'static std::sync::Arc<GpuContext> {
    static CTX: OnceLock<std::sync::Arc<GpuContext>> = OnceLock::new();
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
        std::sync::Arc::new(ctx)
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

/// Root of the upstream reference checkout (`ZENJPEGAI_REF`, set by `just gpu-test`).
pub fn models_dir() -> std::path::PathBuf {
    let root = std::path::PathBuf::from(
        std::env::var_os("ZENJPEGAI_REF").expect("ZENJPEGAI_REF must point at the jpeg-ai-reference-software checkout (run via `just gpu-test`)"),
    );
    assert!(
        root.join("models").is_dir(),
        "{} has no models/ directory",
        root.display()
    );
    root.join("models")
}

pub fn vector_dir(name: &str) -> std::path::PathBuf {
    let root = std::env::var_os("ZENJPEGAI_VECTORS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"));
    let dir = root.join(name);
    assert!(
        dir.join("manifest.txt").is_file(),
        "{} is missing: run scripts/ref_vectors/make_reference_streams.sh",
        dir.display()
    );
    dir
}

/// `f32` tensors of a reference decoder dump, by name (format: see the core crate's
/// `tests/common/mod.rs`).
pub fn load_dump_f32(
    dir: &std::path::Path,
    names: &[&str],
) -> std::collections::HashMap<String, Vec<f32>> {
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).unwrap();
    let blob = std::fs::read(dir.join("tensors.bin")).unwrap();
    let mut out = std::collections::HashMap::new();
    for line in manifest
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        if !names.contains(&f[0]) {
            continue;
        }
        assert_eq!(f[1], "f32", "{}", f[0]);
        let ndim: usize = f[2].parse().unwrap();
        let offset: usize = f[3 + ndim].parse().unwrap();
        let nbytes: usize = f[4 + ndim].parse().unwrap();
        let data = blob[offset..offset + nbytes]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        out.insert(f[0].to_string(), data);
    }
    for n in names {
        assert!(
            out.contains_key(*n),
            "{n} missing from the dump in {}",
            dir.display()
        );
    }
    out
}
