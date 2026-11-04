//! `GpuEncoder` (GPU analysis + CPU entropy) against the all-CPU `Encoder`: both produce a
//! valid stream, and their decoded pictures agree within the tolerance the GPU analysis
//! transform adds (`analysis_matches_cpu`, amplified through synthesis — not bit-exact).
#![cfg(feature = "gpu-tests")]

mod common;
use common::{context_arc, models_dir};
use zenjpegai::decoder::output::RgbImage;
use zenjpegai::encoder::{EncodeParams, Encoder};
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::ModelDir;
use zenjpegai::{Decoder, Engine};
use zenjpegai_gpu::GpuEncoder;

/// A deterministic small RGB picture (a smooth ramp, so codec error is exercised too).
fn source(w: usize, h: usize) -> RgbImage {
    RgbImage {
        width: w,
        height: h,
        bit_depth: 8,
        data: (0..w * h * 3)
            .map(|i| ((i / 3 + i % 3 * 64) % 256) as u16)
            .collect(),
    }
}

fn decode_rgb(stream: &[u8]) -> Vec<u16> {
    Decoder::new(models_dir()).decode(stream).unwrap().data
}

#[test]
fn gpu_encode_matches_cpu_encode() {
    let _ = Engine::new();
    let src = source(128, 96);
    let params = EncodeParams {
        model_id: 1,
        op: OperatingPoint::Bop,
        ..EncodeParams::default()
    };

    let cpu = Encoder::new(models_dir());
    let gpu = GpuEncoder::new(
        context_arc(),
        Encoder::new(models_dir()),
        Box::new(ModelDir::new(models_dir())),
    );

    let s_cpu = cpu.encode(src.clone(), params).unwrap();
    let s_gpu = gpu.encode(src, params).unwrap();

    // Both are conformant streams (the GPU one decodes at all).
    let (a, b) = (decode_rgb(&s_cpu), decode_rgb(&s_gpu));
    assert_eq!(a.len(), b.len());
    let (differing, worst) = a.iter().zip(&b).fold((0usize, 0u32), |(n, m), (x, y)| {
        let d = x.abs_diff(*y) as u32;
        (n + (d > 0) as usize, m.max(d))
    });
    println!(
        "gpu_encode vs cpu_encode: {differing}/{} samples differ, worst {worst}",
        a.len()
    );
    // GPU analysis ~3e-3 in [0,1] feeds the same CPU entropy path; the decoded
    // 8-bit reconstruction can move a few steps. Loose bound, measured on llvmpipe.
    assert!(worst <= 24, "worst {worst} over the gpu-analysis bound");
    assert!(
        differing < a.len() / 3,
        "{differing} samples differ — more than a small bound"
    );
}
