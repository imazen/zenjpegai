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
use zenjpegai::decoder::output::{Picture, finish, quantize_plane, to_source_format};
use zenjpegai::decoder::reconstruct::{
    Planes, post_process_latent, reconstruct_latent, synthesize,
};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::filters::FilterContext;
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
    /// `planes` after the post-filters the stream enables.
    filtered: Planes,
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
    // `model.decompress` hands the planes over in the source's chroma format.
    let planes = to_source_format(&hdr, planes).unwrap();
    let ctx = FilterContext {
        eng,
        hdr: &hdr,
        tools: &headers.tools,
        luma_scale_log: &ent[0].scale_log,
        models: &models,
        op,
        icci_nets: &Default::default(),
        stop: &enough::Unstoppable,
    };
    let filtered = zenjpegai::filters::apply(&ctx, planes.clone()).unwrap();
    let psi = [ly.psi, luv.psi];
    (
        hdr,
        Decoded {
            psi,
            y_hat: [ly.y_hat, luv.y_hat],
            planes,
            filtered,
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
        let want = &dump[key];
        assert_eq!(want.shape[2..], [got.h, got.w], "{name} {key}: plane size");
        let diff = max_abs_diff(&want.f32(), &got.data);
        assert!(
            diff < 3e-3,
            "{name} {key}: max abs diff {diff:e} (range 0..255)"
        );
    }

    // Final samples at the stream's bit depth: RGB interleaved, or the YUV planes one after
    // another. The reference's `out.*` planes are quantised exactly like its file writers do.
    let depth = hdr.bit_depth;
    let theirs: Vec<Vec<u16>> = ["out.a", "out.b", "out.c"]
        .iter()
        .map(|k| quantize_plane(&dump[*k].f32(), depth))
        .collect();
    let (ours, theirs): (Vec<u16>, Vec<u16>) = match finish(&hdr, &d.filtered).unwrap() {
        Picture::Rgb(img) => {
            let t = theirs[0]
                .iter()
                .zip(&theirs[1])
                .zip(&theirs[2])
                .flat_map(|((&r, &g), &b)| [r, g, b])
                .collect();
            (img.data, t)
        }
        Picture::Yuv(img) => ([img.y, img.u, img.v].concat(), theirs.concat()),
    };
    assert_eq!(ours.len(), theirs.len(), "{name}: sample count");
    let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    println!(
        "{name}: {differing} of {} {depth}-bit samples differ, worst by {worst}",
        ours.len()
    );
    assert!(
        worst <= 1,
        "{name}: a {depth}-bit sample differs by {worst}"
    );
    assert!(
        differing * 5000 < ours.len(),
        "{name}: {differing} of {} {depth}-bit samples differ",
        ours.len()
    );

    // The one-call API must produce exactly the same samples.
    let decoder = zenjpegai::Decoder::new(ref_root().join("models"));
    let api: Vec<u16> = match decoder.decode_picture(&stream).unwrap() {
        Picture::Rgb(img) => img.data,
        Picture::Yuv(img) => [img.y, img.u, img.v].concat(),
    };
    assert!(
        api == ours,
        "{name}: Decoder::decode_picture differs from the staged decode"
    );
}

/// The UDI substream is handed back byte for byte.
#[test]
fn user_data_is_passed_through() {
    let dir = vector_dir("img30_base_off_udi_m1");
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let payload = std::fs::read(dir.join("../../inputs/udi_payload.bin")).unwrap();
    let headers = zenjpegai::Decoder::new(ref_root().join("models"))
        .read_headers(&stream)
        .unwrap();
    assert_eq!(headers.user_data.as_deref(), Some(&payload[..]));
}

/// Progressive decode (`num_decode_chs`): only a prefix of the latent channels is read. The
/// oracle is the reference decoder run with the same limits (`<vector>/progressive_y*_uv*/`).
#[test]
fn progressive_decode_matches_reference() {
    let dir = vector_dir("img30_base_off_bpp050");
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    for (luma, chroma) in [(64u16, 32u16), (1, 1), (37, 0)] {
        let sub = dir.join(format!("progressive_y{luma}_uv{chroma}"));
        assert!(
            sub.join("manifest.txt").is_file(),
            "{} is missing: see scripts/ref_vectors/dump_decode.py --decoder-args",
            sub.display()
        );
        let dump = load_dump(&sub);
        let decoder = zenjpegai::Decoder::new(ref_root().join("models"))
            .max_channels(Some(luma), Some(chroma));
        let Picture::Rgb(img) = decoder.decode_picture(&stream).unwrap() else {
            panic!("RGB stream");
        };
        let theirs: Vec<Vec<u16>> = ["out.a", "out.b", "out.c"]
            .iter()
            .map(|k| quantize_plane(&dump[*k].f32(), 8))
            .collect();
        let theirs: Vec<u16> = theirs[0]
            .iter()
            .zip(&theirs[1])
            .zip(&theirs[2])
            .flat_map(|((&r, &g), &b)| [r, g, b])
            .collect();
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

macro_rules! vectors {
    ($($fn_name:ident => $dir:literal),* $(,)?) => { $( #[test] fn $fn_name() { check($dir); } )* };
}

vectors! {
    img30_base_off_bpp012 => "img30_base_off_bpp012",
    img30_base_off_bpp050 => "img30_base_off_bpp050",
    img30_base_off_bpp100 => "img30_base_off_bpp100",
    img30_simple_off_bpp050 => "img30_simple_off_bpp050",
    // YUV sources (4:2:0, 4:2:2, 4:4:4; 8 and 10 bit; odd size), RGB coded with subsampled
    // chroma (bicubic up-sampling), non-displayed border.
    img30yuv420_base_off_bpp050 => "img30yuv420_base_off_bpp050",
    img30yuv422_base_off_bpp050 => "img30yuv422_base_off_bpp050",
    img30yuv444_base_off_bpp050 => "img30yuv444_base_off_bpp050",
    img30yuv420b10_base_off_bpp050 => "img30yuv420b10_base_off_bpp050",
    img30yuv444b10_base_off_bpp050 => "img30yuv444b10_base_off_bpp050",
    img30cropyuv420_base_off_bpp075 => "img30cropyuv420_base_off_bpp075",
    img30_base_off_c420_bpp050 => "img30_base_off_c420_bpp050",
    img30_base_off_c422_bpp050 => "img30_base_off_c422_bpp050",
    img30_base_off_display_m1 => "img30_base_off_display_m1",
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
    // Post-filters: EFE linear, then EFE linear + non-linear with the encoder's filter choices
    // forced (every filter length and region split; odd picture size; four non-linear tiles).
    img30_base_efelin_bpp050 => "img30_base_efelin_bpp050",
    img30_efe_f2c1_f2c2_nl => "img30_efe_f2c1_f2c2_nl",
    img30_efe_f3c3_f3c4_nl => "img30_efe_f3c3_f3c4_nl",
    img30_efe_f3c5_f4c7_nl => "img30_efe_f3c5_f4c7_nl",
    img30_efe_f4c6_f1c5 => "img30_efe_f4c6_f1c5",
    crop277_efe_f4c5_f3c6_nl => "crop277_efe_f4c5_f3c6_nl",
    img01_efe_f2c0_f3c0_nl => "img01_efe_f2c0_f3c0_nl",
    // 2096x1400: six overlapping synthesis tiles.
    img01_base_off_bpp050 => "img01_base_off_bpp050",
    img01_base_off_threads8_bpp050 => "img01_base_off_threads8_bpp050",
    img01_base_off_depregions_m1 => "img01_base_off_depregions_m1",
    img01_base_off_indregions_m1 => "img01_base_off_indregions_m1",
    img01_base_off_indregions_threads8_m2 => "img01_base_off_indregions_threads8_m2",
    // Post-filters: LEF, eICCI.
    img30_base_lef_bpp050 => "img30_base_lef_bpp050",
    img30_base_eicci_bpp050 => "img30_base_eicci_bpp050",
    img01_base_eiccitiles_lef_bpp050 => "img01_base_eiccitiles_lef_bpp050",
    // Upstream's tools_on: all four post-filters in a row, on top of RVS / GRFS / LSBS.
    img30_base_on_bpp025 => "img30_base_on_bpp025",
    img30_base_on_bpp100 => "img30_base_on_bpp100",
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
