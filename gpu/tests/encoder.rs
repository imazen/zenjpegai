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

/// `encode` on a `SourceImage::Yuv` keeps the source's `colour_transform_idx`/`s_ver`/`s_hor`
/// (the `encode_latents_with` fix): the GPU stream's header matches the CPU encode's, and the
/// decoded pictures agree within the GPU analysis bound.
#[test]
fn gpu_encode_yuv_source_matches_cpu() {
    let _ = Engine::new();
    let vecroot = std::env::var_os("ZENJPEGAI_VECTORS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"));
    let raw = std::fs::read(vecroot.join("_inputs/img30_560x888_8bit_444.yuv")).unwrap();
    let (w, h) = (560usize, 888usize);
    let n = w * h;
    let src = zenjpegai::decoder::output::YuvImage {
        width: w,
        height: h,
        chroma_width: w,
        chroma_height: h,
        bit_depth: 8,
        y: raw[..n].iter().map(|&v| v as u16).collect(),
        u: raw[n..2 * n].iter().map(|&v| v as u16).collect(),
        v: raw[2 * n..3 * n].iter().map(|&v| v as u16).collect(),
    };
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
    let s_cpu = cpu
        .encode(zenjpegai::encoder::SourceImage::Yuv(src.clone()), params)
        .unwrap();
    let s_gpu = gpu
        .encode(zenjpegai::encoder::SourceImage::Yuv(src), params)
        .unwrap();
    // Same colour_transform_idx + sampling in both headers → both decode to YUV pictures.
    let pic = |s: &[u8]| match zenjpegai::Decoder::new(models_dir())
        .decode_picture(s)
        .unwrap()
    {
        zenjpegai::decoder::output::Picture::Yuv(i) => [i.y, i.u, i.v].concat(),
        zenjpegai::decoder::output::Picture::Rgb(i) => i.data,
    };
    let (a, b) = (pic(&s_cpu), pic(&s_gpu));
    assert_eq!(a.len(), b.len());
    let (differing, worst) = a.iter().zip(&b).fold((0usize, 0u32), |(n, m), (x, y)| {
        let d = x.abs_diff(*y) as u32;
        (n + (d > 0) as usize, m.max(d))
    });
    println!(
        "gpu_encode(yuv444) vs cpu_encode: {differing}/{} differ, worst {worst}",
        a.len()
    );
    assert!(
        worst <= 24,
        "worst {worst} — the Yuv header is wrong if this is ~244"
    );
    assert!(
        differing < a.len() / 3,
        "{differing} differ — header mismatch, not the analysis gap"
    );
}
