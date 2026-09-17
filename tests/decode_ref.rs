//! Full decode against reference decoder dumps.
//!
//! The latent path and the synthesis transforms are float32 networks. PyTorch's convolution
//! kernels sum in their own order, so agreement is bounded, not exact: the bounds below are a
//! few times the measured worst case on these vectors (see `PORTING.md`), tight enough that a
//! wrong layer, weight or crop fails by orders of magnitude. Do not loosen them to make a change
//! pass.
//!
//! Between this crate's own SIMD tiers and thread counts the output must be *identical*; that is
//! checked here too, on real streams.
#![cfg(feature = "reference-tests")]

mod common;
use common::{load_dump, load_fixed_decoder_dump, ref_root, vector_dir};
use zenjpegai::container::Codestream;
use zenjpegai::decoder::output::{RgbPlanes, quantize, to_rgb_planes};
use zenjpegai::decoder::reconstruct::{
    Planes, post_process_latent, reconstruct_latent, synthesize,
};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::tensor::Tensor;

fn max_abs_diff(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

struct Decoded {
    psi: [Tensor<f32>; 2],
    y_hat: [Tensor<f32>; 2],
    planes: Planes,
}

fn decode(stream: &[u8], eng: &Engine) -> (zenjpegai::header::PictureHeader, Decoded) {
    let cs = Codestream::parse(stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = headers.picture;
    let models = ModelDir::new(ref_root().join("models"));
    let id = hdr.model_id as usize;
    let op = hdr.synthesis_transforms[0];
    let (ym, uvm) = (
        models.load_common(id, 0, eng).unwrap(),
        models.load_common(id, 1, eng).unwrap(),
    );
    let syn_y = models.load_synthesis_primary(id, op, eng).unwrap();
    let syn_uv = models.load_synthesis_secondary(id, op, eng).unwrap();
    let ent = decode_entropy_stage(&AnsTables::new(), &cs, &hdr, [&ym, &uvm]).unwrap();
    let mut ly = reconstruct_latent(eng, &hdr, 0, &ym, &ent[0]).unwrap();
    let mut luv = reconstruct_latent(eng, &hdr, 1, &uvm, &ent[1]).unwrap();
    post_process_latent(&hdr, &headers.tools, 0, &ent[0], &mut ly).unwrap();
    post_process_latent(&hdr, &headers.tools, 1, &ent[1], &mut luv).unwrap();
    let planes = synthesize(eng, &hdr, &syn_y, &syn_uv, [&ly.y_hat, &luv.y_hat]).unwrap();
    let psi = [ly.psi, luv.psi];
    (
        hdr,
        Decoded {
            psi,
            y_hat: [ly.y_hat, luv.y_hat],
            planes,
        },
    )
}

fn check(name: &str) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    // The stock reference decoder mis-decodes region streams and cannot decode quality maps
    // (PORTING.md); for those the oracle is the decoder with these defects patched.
    let dump = if name.contains("regions") || name.contains("qmap") {
        load_fixed_decoder_dump(&dir)
    } else {
        load_dump(&dir)
    };
    let (hdr, d) = decode(&stream, &Engine::new());

    for (key, got) in [
        ("y.psi", &d.psi[0]),
        ("y.y_hat", &d.y_hat[0]),
        ("uv.psi", &d.psi[1]),
        ("uv.y_hat", &d.y_hat[1]),
    ] {
        let diff = max_abs_diff(&dump[key].f32(), &got.data);
        assert!(diff < 5e-4, "{name} {key}: max abs diff {diff:e}");
    }
    for (key, got) in [
        ("rec.a", &d.planes.y),
        ("rec.b", &d.planes.u),
        ("rec.c", &d.planes.v),
    ] {
        let diff = max_abs_diff(&dump[key].f32(), &got.data);
        assert!(
            diff < 3e-3,
            "{name} {key}: max abs diff {diff:e} (range 0..255)"
        );
    }

    let rgb = to_rgb_planes(&hdr, &d.planes).unwrap();
    let theirs = RgbPlanes {
        width: rgb.width,
        height: rgb.height,
        r: dump["out.a"].f32(),
        g: dump["out.b"].f32(),
        b: dump["out.c"].f32(),
    };
    let (ours, theirs) = (quantize(&rgb, 8).unwrap(), quantize(&theirs, 8).unwrap());
    let differing = ours
        .data
        .iter()
        .zip(&theirs.data)
        .filter(|(a, b)| a != b)
        .count();
    let worst = ours
        .data
        .iter()
        .zip(&theirs.data)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(worst <= 1, "{name}: an 8-bit sample differs by {worst}");
    assert!(
        differing * 5000 < ours.data.len(),
        "{name}: {differing} of {} 8-bit samples differ",
        ours.data.len()
    );
    println!(
        "{name}: {differing} of {} 8-bit samples differ by 1",
        ours.data.len()
    );
}

macro_rules! vectors {
    ($($fn_name:ident => $dir:literal),* $(,)?) => { $( #[test] fn $fn_name() { check($dir); } )* };
}

vectors! {
    img30_base_off_bpp012 => "img30_base_off_bpp012",
    img30_base_off_bpp050 => "img30_base_off_bpp050",
    img30_base_off_bpp100 => "img30_base_off_bpp100",
    img30_simple_off_bpp050 => "img30_simple_off_bpp050",
    // Coding tools: latent scaling before synthesis, residual variance scaling + gain flags.
    img30_base_lsbs_bpp050 => "img30_base_lsbs_bpp050",
    img30_base_rvs_bpp050 => "img30_base_rvs_bpp050",
    img30_base_qmap_bpp050 => "img30_base_qmap_bpp050",
    img30_base_qmap_rvs_bpp025 => "img30_base_qmap_rvs_bpp025",
    img30_base_qmap_threads8_bpp100 => "img30_base_qmap_threads8_bpp100",
    img30_base_rvsonly_bpp050 => "img30_base_rvsonly_bpp050",
    img30_base_grfsonly_bpp075 => "img30_base_grfsonly_bpp075",
    img30_base_lsbs_rvs_bpp025 => "img30_base_lsbs_rvs_bpp025",
    img30_simple_lsbs_rvs_bpp100 => "img30_simple_lsbs_rvs_bpp100",
    // High profile: HOP synthesis (CAB + TAM attention).
    img30_high_off_bpp050 => "img30_high_off_bpp050",
    // 2096x1400: six overlapping synthesis tiles.
    img01_base_off_bpp050 => "img01_base_off_bpp050",
    img01_base_off_threads8_bpp050 => "img01_base_off_threads8_bpp050",
    img01_base_off_depregions_m1 => "img01_base_off_depregions_m1",
    img01_base_off_indregions_m1 => "img01_base_off_indregions_m1",
    img01_base_off_indregions_threads8_m2 => "img01_base_off_indregions_threads8_m2",
}

/// Every SIMD tier, threaded or not, must decode a real stream to identical bits.
#[test]
fn tiers_and_threads_agree_bit_for_bit() {
    for name in [
        "img30_base_off_bpp050",
        "img30_simple_off_bpp050",
        "img30_high_off_bpp050",
    ] {
        let stream = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
        let mut baseline: Option<(String, Decoded)> = None;
        for tier in Tier::available() {
            for parallel in [false, true] {
                // The scalar tier emulates FMA in software on CPUs without it; once is enough.
                if matches!(tier, Tier::Scalar) && parallel {
                    continue;
                }
                let eng = Engine::with(tier, parallel);
                let (_, d) = decode(&stream, &eng);
                let label = format!("{eng:?}");
                match &baseline {
                    None => baseline = Some((label, d)),
                    Some((base_label, b)) => {
                        let pairs = [
                            ("psi[y]", &b.psi[0], &d.psi[0]),
                            ("y_hat[y]", &b.y_hat[0], &d.y_hat[0]),
                            ("y_hat[uv]", &b.y_hat[1], &d.y_hat[1]),
                            ("Y", &b.planes.y, &d.planes.y),
                            ("U", &b.planes.u, &d.planes.u),
                            ("V", &b.planes.v, &d.planes.v),
                        ];
                        for (what, want, got) in pairs {
                            let bad = want
                                .data
                                .iter()
                                .zip(&got.data)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            assert_eq!(
                                bad, 0,
                                "{name} {what}: {label} differs from {base_label} in {bad} values"
                            );
                        }
                    }
                }
            }
        }
    }
}
