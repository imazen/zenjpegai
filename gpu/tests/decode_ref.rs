//! Whole decodes through [`GpuDecoder`] against the reference decoder's dumps: the same gate the
//! CPU engine passes (`tests/decode_ref.rs` of the core crate), plus the GPU-vs-CPU difference.
//!
//! Bounds: synthesised planes within 3e-3 (0..255) of the reference's `rec.a/b/c`; the 8-bit
//! RGB output differs from the reference's by at most 1, in fewer than 1/5000 samples.
#![cfg(feature = "gpu-tests")]

mod common;
use common::{context_arc, load_dump_f32, max_abs, models_dir, vector_dir};
use zenjpegai::Decoder;
use zenjpegai::container::Codestream;
use zenjpegai::decoder::output::{RgbPlanes, quantize, quantize_plane};
use zenjpegai::decoder::reconstruct::{
    Planes, post_process_latent, reconstruct_latent, synthesize,
};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai_gpu::{GpuDecoder, GpuError};

/// Synthesised planes of this crate's CPU engine.
fn cpu_planes(stream: &[u8]) -> Planes {
    let eng = Engine::new();
    let cs = Codestream::parse(stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(models_dir());
    let (id, op) = (hdr.model_id as usize, hdr.synthesis_transforms[0]);
    let common = [
        models.load_common(id, 0, &eng).unwrap(),
        models.load_common(id, 1, &eng).unwrap(),
    ];
    let luma = models.load_synthesis_primary(id, op, &eng).unwrap();
    let chroma = models.load_synthesis_secondary(id, op, &eng).unwrap();
    let ent = decode_entropy_stage(&AnsTables::new(), &cs, hdr, [&common[0], &common[1]]).unwrap();
    let mut ly = reconstruct_latent(&eng, hdr, 0, &common[0], &ent[0]).unwrap();
    let mut luv = reconstruct_latent(&eng, hdr, 1, &common[1], &ent[1]).unwrap();
    post_process_latent(hdr, &headers.tools, 0, &ent[0], &mut ly).unwrap();
    post_process_latent(hdr, &headers.tools, 1, &ent[1], &mut luv).unwrap();
    synthesize(&eng, hdr, &luma, &chroma, [&ly.y_hat, &luv.y_hat]).unwrap()
}

fn check(name: &str) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    // Region streams: the stock reference decoder mis-decodes their residuals (PORTING.md), so
    // the oracle is the dump taken with `dump_decode.py --contiguous-masks`.
    let dump_dir = if name.contains("regions") {
        dir.join("fixed_decoder")
    } else {
        dir.clone()
    };
    let dump = load_dump_f32(
        &dump_dir,
        &["rec.a", "rec.b", "rec.c", "out.a", "out.b", "out.c"],
    );

    let dec = GpuDecoder::new(context_arc(), Box::new(ModelDir::new(models_dir())));
    let decoded = dec.decode_to_gpu(&stream).unwrap();
    let (ours, planes, timing) = pollster::block_on(dec.finish(decoded)).unwrap();
    let zenjpegai::Picture::Rgb(ours) = ours else {
        panic!("{name}: expected an RGB picture");
    };
    println!("{name}: {timing:?}");

    let mut worst_plane = 0.0f32;
    for (key, got) in [
        ("rec.a", &planes.y),
        ("rec.b", &planes.u),
        ("rec.c", &planes.v),
    ] {
        let diff = max_abs(&dump[key], &got.data);
        worst_plane = worst_plane.max(diff);
        assert!(
            diff < 3e-3,
            "{name} {key}: max abs diff {diff:e} (range 0..255)"
        );
    }

    let theirs = quantize(
        &RgbPlanes {
            width: ours.width,
            height: ours.height,
            r: dump["out.a"].clone(),
            g: dump["out.b"].clone(),
            b: dump["out.c"].clone(),
        },
        8,
    )
    .unwrap();
    let count = |a: &[u16], b: &[u16]| {
        let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).abs())
            .max()
            .unwrap();
        (differing, worst)
    };
    let (differing, worst) = count(&ours.data, &theirs.data);
    assert!(
        worst <= 1,
        "{name}: an 8-bit sample differs from the reference by {worst}"
    );
    assert!(
        differing * 5000 < ours.data.len(),
        "{name}: {differing} of {} 8-bit samples differ from the reference",
        ours.data.len()
    );

    // Against this crate's CPU engine (not a gate against the reference, reported for the record).
    let cpu_p = cpu_planes(&stream);
    let vs_cpu = [
        (&cpu_p.y, &planes.y),
        (&cpu_p.u, &planes.u),
        (&cpu_p.v, &planes.v),
    ]
    .iter()
    .map(|(a, b)| max_abs(&a.data, &b.data))
    .fold(0.0, f32::max);
    assert!(
        vs_cpu < 3e-3,
        "{name}: planes differ from the CPU engine by {vs_cpu:e}"
    );
    let cpu = Decoder::new(models_dir()).decode(&stream).unwrap();
    let (cpu_differing, cpu_worst) = count(&ours.data, &cpu.data);
    assert!(
        cpu_worst <= 1,
        "{name}: an 8-bit sample differs from the CPU engine by {cpu_worst}"
    );
    println!(
        "{name}: planes max abs {worst_plane:.3e} vs reference, {vs_cpu:.3e} vs CPU engine; 8-bit samples differing by 1: {differing} vs reference, {cpu_differing} vs CPU engine, of {}",
        ours.data.len()
    );

    // `decode` takes the quantized readback (`GpuOut::Quantized`): the BT.709 conversion and
    // rounding ran on the GPU. Its quantisation of the same GPU planes can move a tie by one
    // step vs the CPU output stage — the recorded bound, measured here per stream.
    let fast = dec.decode(&stream).unwrap();
    let (q_differing, q_worst) = count(&fast.data, &ours.data);
    assert!(
        q_worst <= 1,
        "{name}: a GPU-quantised sample differs from the CPU output by {q_worst}"
    );
    assert!(
        q_differing * 5000 < ours.data.len(),
        "{name}: {q_differing} of {} GPU-quantised samples differ from the CPU output",
        ours.data.len()
    );
    println!(
        "{name}: GPU-quantised readback vs CPU output: {q_differing} of {} differ, worst {q_worst}",
        ours.data.len()
    );

    // A second decode reuses plans and buffers and must reproduce the first bit for bit.
    let again = dec.decode(&stream).unwrap();
    assert_eq!(
        again.data, fast.data,
        "{name}: warm decode differs from the cold one"
    );
}

macro_rules! vectors {
    ($($fn_name:ident => $dir:literal),* $(,)?) => { $( #[test] fn $fn_name() { check($dir); } )* };
}

vectors! {
    img30_simple_off_bpp050 => "img30_simple_off_bpp050",
    img30_base_off_bpp050 => "img30_base_off_bpp050",
    img30_high_off_bpp050 => "img30_high_off_bpp050",
    img01_base_off_bpp050 => "img01_base_off_bpp050",
    img01_base_off_indregions_m1 => "img01_base_off_indregions_m1",
}

/// The readback-free presentation path: YUV to `rgba8unorm` on the GPU, then a blit onto a render
/// target. Both must reproduce the CPU output stage up to rounding (the GPU rounds `x * 255` to
/// nearest in fixed function; ties and fused multiply-adds can move a sample by 1).
#[test]
fn gpu_rgba_presentation_matches_cpu_output() {
    let stream = std::fs::read(vector_dir("img30_base_off_bpp050").join("stream.bits")).unwrap();
    let ctx = context_arc();
    let dec = GpuDecoder::new(ctx.clone(), Box::new(ModelDir::new(models_dir())));
    let decoded = dec.decode_to_gpu(&stream).unwrap();
    assert!(decoded.presentable_on_gpu());
    let tex = decoded.to_rgba_texture().unwrap();
    let rgba = pollster::block_on(zenjpegai_gpu::read_rgba8(&ctx, &tex)).unwrap();
    let (zenjpegai::Picture::Rgb(rgb), _, _) = pollster::block_on(dec.finish(decoded)).unwrap()
    else {
        panic!("expected an RGB picture");
    };
    assert_eq!(rgba.len(), rgb.width * rgb.height * 4);
    let (mut differing, mut worst) = (0usize, 0i32);
    for (px, want) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.data.as_chunks::<3>().0)
    {
        assert_eq!(px[3], 255);
        for c in 0..3 {
            let d = (px[c] as i32 - want[c] as i32).abs();
            differing += (d != 0) as usize;
            worst = worst.max(d);
        }
    }
    println!(
        "rgba texture vs CPU output stage: {differing} of {} samples differ, worst {worst}",
        rgb.data.len()
    );
    assert!(worst <= 1 && differing * 5000 < rgb.data.len());

    // Blit onto an offscreen target of the same format: an exact copy.
    let target = ctx.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: tex.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    zenjpegai_gpu::Blitter::new(&ctx, wgpu::TextureFormat::Rgba8Unorm).blit(
        &ctx,
        &tex,
        &target.create_view(&Default::default()),
    );
    let blitted = pollster::block_on(zenjpegai_gpu::read_rgba8(&ctx, &target)).unwrap();
    assert!(blitted == rgba, "blit is not an exact copy");
}

/// The quantized readback for YUV streams (`GpuOut::Quantized`, `mode` 0): chroma is gathered
/// with the coded subsampling and quantised on the GPU, at the stream's bit depth. Compared
/// both to the reference dump's `out.*` planes quantised like `write_png` does, and to the
/// CPU output stage run on the same GPU planes (`finish`).
#[test]
fn gpu_quantized_yuv() {
    for name in [
        "img30yuv420_base_off_bpp050",
        "img30yuv422_base_off_bpp050",
        "img30yuv444_base_off_bpp050",
        "img30yuv444b10_base_off_bpp050",
        "img30cropyuv420_base_off_bpp075",
    ] {
        let dir = vector_dir(name);
        let stream = std::fs::read(dir.join("stream.bits")).unwrap();
        let cs = Codestream::parse(&stream).unwrap();
        let hdr = read_headers(&cs).unwrap().picture;
        let dump = load_dump_f32(&dir, &["out.a", "out.b", "out.c"]);

        let dec = GpuDecoder::new(context_arc(), Box::new(ModelDir::new(models_dir())));
        // CPU output stage on the GPU planes.
        let (want, _, _) =
            pollster::block_on(dec.finish(dec.decode_to_gpu(&stream).unwrap())).unwrap();
        let zenjpegai::Picture::Yuv(want) = want else {
            panic!("{name}: expected a YUV picture");
        };
        // The quantized path (`decode_picture` runs `GpuOut::Quantized`).
        let got = pollster::block_on(dec.decode_picture_async(&stream)).unwrap();
        let zenjpegai::Picture::Yuv(got) = got else {
            panic!("{name}: expected a YUV picture");
        };
        assert_eq!(
            (
                got.width,
                got.height,
                got.chroma_width,
                got.chroma_height,
                got.bit_depth
            ),
            (
                want.width,
                want.height,
                want.chroma_width,
                want.chroma_height,
                want.bit_depth
            ),
            "{name}: plane dims"
        );
        for (key, plane, ours) in [
            ("out.a", "y", &got.y),
            ("out.b", "u", &got.u),
            ("out.c", "v", &got.v),
        ] {
            let theirs = quantize_plane(&dump[key], hdr.bit_depth);
            assert_eq!(theirs.len(), ours.len(), "{name} {plane}: sample count");
            let mut differing = 0usize;
            let mut worst = 0i32;
            let mut vs_cpu = 0usize;
            let want_plane = match plane {
                "y" => &want.y,
                "u" => &want.u,
                _ => &want.v,
            };
            for ((&a, &b), &c) in ours.iter().zip(&theirs).zip(want_plane) {
                differing += (a != b) as usize;
                worst = worst.max((a as i32 - b as i32).abs());
                vs_cpu += (a != c) as usize;
            }
            println!(
                "{name} {plane}: {differing} of {} {}-bit samples differ from the reference \
                 (worst {worst}), {vs_cpu} vs CPU output",
                ours.len(),
                hdr.bit_depth
            );
            assert!(
                worst <= 1,
                "{name} {plane}: a sample differs from the reference by {worst}"
            );
            assert!(
                differing * 5000 < ours.len(),
                "{name} {plane}: {differing} of {} differ from the reference",
                ours.len()
            );
            assert!(
                vs_cpu * 5000 < ours.len(),
                "{name} {plane}: {vs_cpu} of {} differ from the CPU output",
                ours.len()
            );
        }
    }
}

/// Cancellation: the `enough` token the CPU decoder takes, checked between the CPU stages,
/// before the synthesis submit and around the readback. Trips at several points must all
/// surface `Error::Cancelled`, and the decoder stays usable and deterministic afterwards.
#[test]
fn stop_cancels_gpu_decode() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Trips once it has been asked `after` times.
    struct StopAfter {
        after: usize,
        checks: AtomicUsize,
    }
    impl enough::Stop for StopAfter {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.checks.fetch_add(1, Ordering::Relaxed) >= self.after {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    let stream = std::fs::read(vector_dir("img01_base_off_bpp050").join("stream.bits")).unwrap();
    let dec = GpuDecoder::new(context_arc(), Box::new(ModelDir::new(models_dir())));
    let count = StopAfter {
        after: usize::MAX,
        checks: AtomicUsize::new(0),
    };
    let full = dec.decode_with(&stream, &count).unwrap();
    let total = count.checks.load(Ordering::Relaxed);
    assert!(total >= 10, "only {total} stop checks in a GPU decode");
    for after in [0, 1, total / 2, total - 1] {
        let stop = StopAfter {
            after,
            checks: AtomicUsize::new(0),
        };
        let err = dec.decode_with(&stream, &stop).unwrap_err();
        assert!(
            matches!(
                err,
                GpuError::Codec(zenjpegai::Error::Cancelled(enough::StopReason::Cancelled))
            ),
            "after {after}: {err:?}"
        );
        assert_eq!(stop.checks.load(Ordering::Relaxed), after + 1);
    }
    assert_eq!(
        dec.decode(&stream).unwrap(),
        full,
        "decode after a cancellation differs"
    );
}

/// Progressive decode (`max_channels`, the reference's `num_decode_chs`) on the GPU path: only
/// a prefix of the latent channels is read. The oracle is the reference decoder's dump run
/// with the same limits (`<vector>/progressive_y*_uv*/`), quantised like `write_png`.
#[test]
fn gpu_progressive_decode_matches_reference() {
    let dir = vector_dir("img30_base_off_bpp050");
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    for (luma, chroma) in [(64u16, 32u16), (1, 1), (37, 0)] {
        let sub = dir.join(format!("progressive_y{luma}_uv{chroma}"));
        let dump = load_dump_f32(&sub, &["out.a", "out.b", "out.c"]);
        let dec = GpuDecoder::new(context_arc(), Box::new(ModelDir::new(models_dir())))
            .max_channels(Some(luma), Some(chroma));
        let img = dec.decode(&stream).unwrap();
        let theirs: Vec<Vec<u16>> = ["out.a", "out.b", "out.c"]
            .iter()
            .map(|k| quantize_plane(&dump[*k], 8))
            .collect();
        let theirs: Vec<u16> = theirs[0]
            .iter()
            .zip(&theirs[1])
            .zip(&theirs[2])
            .flat_map(|((&r, &g), &b)| [r, g, b])
            .collect();
        assert_eq!(img.data.len(), theirs.len());
        let differing = img.data.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        let worst = img
            .data
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        println!(
            "progressive y{luma} uv{chroma}: {differing} of {} samples differ, worst by {worst}",
            theirs.len()
        );
        assert!(worst <= 1 && differing * 5000 < theirs.len());
    }
}
