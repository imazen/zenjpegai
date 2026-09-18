//! Scratch reproduction for the `car_bpp75` wasm trap: `GpuDecoder::decode_to_gpu` must return
//! a `GpuError` for streams it cannot run, never panic (the wasm build is panic=abort — a trap
//! kills the whole worker and makes CPU fallback impossible). Ignored by default: it needs a
//! real Vulkan adapter and the fetched demo assets. Run with:
//!   cargo test -p zenjpegai-wasm --features gpu --test gpu_repro -- --ignored --nocapture
#![cfg(feature = "gpu")]

use std::sync::Arc;

use zenjpegai::nn::fast::Engine;
use zenjpegai::weights::packed::PackedBundle;
use zenjpegai_gpu::{ContextOptions, GpuContext, GpuDecoder};

#[test]
#[ignore = "needs a Vulkan adapter and web/.demo-assets (node web/scripts/fetch-demo-assets.mjs)"]
fn car_bpp75_returns_error_not_panic() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../web/.demo-assets");
    let mut bundle = PackedBundle::default();
    for id in 0..4usize {
        for part in ["common", "bop"] {
            let p = dir.join(format!("m{id}_{part}.zjb"));
            if p.exists() {
                bundle.add(std::fs::read(&p).unwrap()).unwrap();
            }
        }
    }
    let ctx = GpuContext::new(&ContextOptions {
        allow_software: true,
        ..Default::default()
    })
    .expect("a Vulkan adapter (lavapipe) is needed for this test");
    let decoder = GpuDecoder::with_engine(
        Arc::new(ctx),
        Box::new(bundle),
        Engine::with(zenjpegai::nn::fast::Tier::detect(), false),
    );
    let stream = std::fs::read(dir.join("car_bpp75.jai")).unwrap();
    match decoder.decode_to_gpu(&stream) {
        Ok(_) => eprintln!("decoded fine"),
        Err(e) => eprintln!("clean GpuError: {e}"),
    }
}
