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
use zenjpegai::decoder::output::{RgbPlanes, quantize};
use zenjpegai::decoder::reconstruct::{
    Planes, post_process_latent, reconstruct_latent, synthesize,
};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai_gpu::GpuDecoder;

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

    // A second decode reuses plans and buffers and must reproduce the first bit for bit.
    let again = dec.decode(&stream).unwrap();
    assert_eq!(
        again.data, ours.data,
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
