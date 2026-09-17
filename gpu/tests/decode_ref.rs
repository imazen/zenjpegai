//! Whole decodes through [`GpuDecoder`] against the reference decoder's dumps: the same gate the
//! CPU engine passes (`tests/decode_ref.rs` of the core crate), plus the GPU-vs-CPU difference.
//!
//! Bounds: synthesised planes within 3e-3 (0..255) of the reference's `rec.a/b/c`; the 8-bit
//! RGB output differs from the reference's by at most 1, in fewer than 1/5000 samples.
#![cfg(feature = "gpu-tests")]

mod common;
use common::{context_arc, load_dump_f32, max_abs, models_dir, vector_dir};
use zenjpegai::Decoder;
use zenjpegai::decoder::output::{RgbPlanes, quantize};
use zenjpegai::model::ModelDir;
use zenjpegai_gpu::GpuDecoder;

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
    let cpu = Decoder::new(models_dir()).decode(&stream).unwrap();
    let (cpu_differing, cpu_worst) = count(&ours.data, &cpu.data);
    assert!(
        cpu_worst <= 1,
        "{name}: an 8-bit sample differs from the CPU engine by {cpu_worst}"
    );
    println!(
        "{name}: planes max abs vs reference {worst_plane:.3e}; 8-bit samples differing by 1: {differing} vs reference, {cpu_differing} vs CPU engine, of {}",
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
