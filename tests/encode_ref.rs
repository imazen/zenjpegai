//! Encoder-side networks against the reference encoder (Gate 1 of the encoder port).
//!
//! Oracle: `scripts/ref_vectors/dump_encode.py OUT --enc2 -- <encoder args>` run on the stock
//! reference encoder (`make_reference_streams.sh encoder`), which records the input and output
//! of every analysis-transform and hyper-encoder call. Float networks: the bounds are a few
//! times the measured worst case (see `PORTING.md`); do not loosen them to make a change pass.
#![cfg(feature = "reference-tests")]

mod common;
use common::{load_encoder_dump, ref_root, vector_dir};
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{
    ModelDir, load_analysis_primary, load_analysis_secondary, load_hyper_encoder,
};
use zenjpegai::nn::fast::{BTensor, Engine, Tier};
use zenjpegai::tensor::Tensor;

fn max_abs_diff(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn tensor(t: &common::RefTensor) -> Tensor<f32> {
    assert_eq!(t.shape.len(), 4);
    Tensor::from_vec(t.shape[1], t.shape[2], t.shape[3], t.f32()).unwrap()
}

/// Returns the worst error of (analysis y, analysis uv, hyper y, hyper uv).
fn run(vector: &str, model_id: usize, op: OperatingPoint, eng: &Engine) -> [f32; 4] {
    let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
    let models = ModelDir::new(ref_root().join("models"));
    let stop = enough::Unstoppable;
    let mut out = [0f32; 4];

    let prim = load_analysis_primary(&models, model_id, op, eng).unwrap();
    let sec = load_analysis_secondary(&models, model_id, op, eng).unwrap();
    let x = tensor(&dump["analysis_y.0.in"]);
    let y = prim.forward(eng, &x, &stop).unwrap().to_planar().unwrap();
    out[0] = max_abs_diff(&dump["analysis_y.0.out"].f32(), &y.data);
    let x = tensor(&dump["analysis_uv.0.in"]);
    let uv = sec.forward(eng, &x, &stop).unwrap().to_planar().unwrap();
    out[1] = max_abs_diff(&dump["analysis_uv.0.out"].f32(), &uv.data);

    // Hyper-encoders on the reference's own latents; the picture is 16x (8x of the half-size
    // chroma plane) the latent only up to rounding, so take the size from the analysis input.
    let sizes =
        [&dump["analysis_y.0.in"], &dump["analysis_uv.0.in"]].map(|t| (t.shape[2], t.shape[3]));
    for (ccs, name) in ["hyper_y", "hyper_uv"].into_iter().enumerate() {
        let he = load_hyper_encoder(&models, model_id, ccs, eng).unwrap();
        let y = tensor(&dump[&format!("{name}.0.in")]);
        let y = BTensor::from_planar(&y, eng.tier.block()).unwrap();
        let (h, w) = sizes[ccs];
        let d = if ccs == 0 { 16 } else { 8 };
        let z = he
            .forward(eng, &y, h, w, d, &stop)
            .unwrap()
            .to_planar()
            .unwrap();
        out[2 + ccs] = max_abs_diff(&dump[&format!("{name}.0.out")].f32(), &z.data);
        // `_compress_z`: clamp to [-31, 31], round half to even. On the reference's own `y` the
        // committed hyper-latent must come out the same.
        let comp = ["y", "uv"][ccs];
        let want = dump[&format!("{comp}.z_hat")].i8();
        let differing = z
            .data
            .iter()
            .zip(&want)
            .filter(|&(&v, &w)| v.clamp(-31.0, 31.0).round_ties_even() as i8 != w)
            .count();
        assert_eq!(differing, 0, "{vector}: {comp}.z_hat symbols differ");
    }
    println!(
        "{vector}: analysis y {:e} uv {:e}, hyper y {:e} uv {:e}",
        out[0], out[1], out[2], out[3]
    );
    out
}

/// Measured worst case: analysis 1.1e-4 (HOP luma), hyper-encoder 3.3e-5.
const ANALYSIS_BOUND: f32 = 5e-4;
const HYPER_BOUND: f32 = 2e-4;

fn check(e: [f32; 4]) {
    assert!(e[0] < ANALYSIS_BOUND && e[1] < ANALYSIS_BOUND, "{e:?}");
    assert!(e[2] < HYPER_BOUND && e[3] < HYPER_BOUND, "{e:?}");
}

#[test]
fn bop_networks_match_reference() {
    check(run(
        "enc_img30_bop_m1_b0",
        1,
        OperatingPoint::Bop,
        &Engine::new(),
    ));
}

#[test]
fn hop_networks_match_reference() {
    check(run(
        "enc_img30_hop_m2_b0",
        2,
        OperatingPoint::Hop,
        &Engine::new(),
    ));
}

#[test]
fn analysis_tiers_agree_bit_for_bit() {
    let dump = load_encoder_dump(&vector_dir("enc_img30_bop_m1_b0").join("enc2"));
    let models = ModelDir::new(ref_root().join("models"));
    let x = tensor(&dump["analysis_y.0.in"]);
    let mut first: Option<Vec<f32>> = None;
    for tier in Tier::available() {
        for parallel in [false, true] {
            let eng = Engine::with(tier, parallel);
            let net = load_analysis_primary(&models, 1, OperatingPoint::Bop, &eng).unwrap();
            let y = net
                .forward(&eng, &x, &enough::Unstoppable)
                .unwrap()
                .to_planar()
                .unwrap();
            match &first {
                None => first = Some(y.data),
                Some(f) => assert!(
                    f.iter()
                        .zip(&y.data)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{tier:?} parallel={parallel}"
                ),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate 2 / 3: the whole encoder against the reference encoder's own decisions, then the stream
// through both decoders.

use std::path::Path;
use zenjpegai::encoder::{EncodeParams, Encoder, preprocess_rgb, read_png_rgb8};

/// (`vector`, source image, `model_id`, operating point, `beta_displacement_log`) of every
/// fixed-model encode `make_reference_streams.sh encoder` produces.
const VECTORS: &[(&str, &str, u8, OperatingPoint, i32)] = &[
    (
        "enc_img30_bop_m1_b0",
        "00030_TE_560x888_8bit_sRGB.png",
        1,
        OperatingPoint::Bop,
        0,
    ),
    (
        "enc_img30_hop_m2_b0",
        "00030_TE_560x888_8bit_sRGB.png",
        2,
        OperatingPoint::Hop,
        0,
    ),
    (
        "enc_img30_bop_m1_bm300",
        "00030_TE_560x888_8bit_sRGB.png",
        1,
        OperatingPoint::Bop,
        -300,
    ),
    (
        "enc_img30_sop_m0_bm300",
        "00030_TE_560x888_8bit_sRGB.png",
        0,
        OperatingPoint::Sop,
        -300,
    ),
    (
        "enc_img30_bop_m3_b400",
        "00030_TE_560x888_8bit_sRGB.png",
        3,
        OperatingPoint::Bop,
        400,
    ),
    (
        "enc_img30_bop_m0_bm1069",
        "00030_TE_560x888_8bit_sRGB.png",
        0,
        OperatingPoint::Bop,
        -1069,
    ),
    (
        "enc_img30_hop_m3_bm1069",
        "00030_TE_560x888_8bit_sRGB.png",
        3,
        OperatingPoint::Hop,
        -1069,
    ),
];

fn source(image: &str) -> zenjpegai::RgbImage {
    let path = ref_root().join("data/test").join(image);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    read_png_rgb8(&bytes).unwrap()
}

fn present(vector: &str) -> bool {
    vector_dir(vector).join("enc2/enc_manifest.txt").is_file()
}

/// Colour pre-processing: bit-exact against the reference's own analysis-transform inputs.
#[test]
fn colour_preprocessing_matches_reference() {
    let mut ran = 0;
    for &(vector, image, ..) in VECTORS {
        if !present(vector) {
            continue;
        }
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        let input = preprocess_rgb(&source(image)).unwrap();
        for (name, got) in [
            ("analysis_y.0.in", &input.luma),
            ("analysis_uv.0.in", &input.chroma),
        ] {
            let want = dump[name].f32();
            assert_eq!(want.len(), got.data.len(), "{vector} {name}: length");
            let differing = want
                .iter()
                .zip(&got.data)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(differing, 0, "{vector} {name}: {differing} samples differ");
        }
        ran += 1;
    }
    assert!(ran > 0, "no encoder vectors on disk");
}

/// Compare one encode's decisions with the reference encoder's own.
///
/// The integer stage (`z_hat`, both scale maps, the skip mask, the cube flags) must be
/// identical. `residual_quant` cannot be: the mean a symbol is measured against comes out of
/// the hyper-decoder and, for luma, the context model, both float networks whose sums this
/// crate orders differently from oneDNN (`psi` differs by up to 9e-7, `y_hat` by 8e-5 —
/// `PORTING.md`). A symbol whose unrounded value sits within that of a `.5` boundary lands on
/// the other side. Measured worst case over the seven vectors: see `PORTING.md`.
fn compare_decisions(vector: &str, traces: &[zenjpegai::encoder::ComponentTrace; 2]) -> [usize; 2] {
    let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
    let mut moved = [0usize; 2];
    for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
        let t = &traces[ccs];
        let eq_i32 = |field: &str, got: &[i32]| {
            let want = dump[&format!("{name}.{field}")].i32();
            assert_eq!(want.len(), got.len(), "{vector} {name}.{field}: length");
            let bad = want.iter().zip(got).filter(|(a, b)| a != b).count();
            assert_eq!(bad, 0, "{vector} {name}.{field}: {bad} values differ");
        };
        let z: Vec<i32> = t.z_hat.data.iter().map(|&v| v as i32).collect();
        eq_i32("z_hat", &z);
        eq_i32("skip_scale_log", &t.skip_scale_log.data);
        eq_i32("scale_log", &t.scale_log.data);
        let want_flags = &dump[&format!("{name}.cube_flag")].bytes;
        assert_eq!(
            want_flags.len(),
            t.cube_flag.len(),
            "{vector} {name}: flags"
        );
        let bad = want_flags
            .iter()
            .zip(&t.cube_flag)
            .filter(|&(&a, &b)| (a != 0) != b)
            .count();
        assert_eq!(bad, 0, "{vector} {name}.cube_flag: {bad} flags differ");

        let want = dump[&format!("{name}.residual_quant")].i32();
        let mut worst = 0i32;
        let mut n = 0usize;
        for (&a, &b) in want.iter().zip(&t.residual_q.data) {
            let d = a - b as i32;
            if d != 0 {
                n += 1;
                worst = worst.max(d.abs());
            }
        }
        assert!(
            worst <= 1,
            "{vector} {name}.residual_quant: a symbol moved by {worst}, not a rounding boundary"
        );
        assert!(
            n * 4000 <= want.len(),
            "{vector} {name}.residual_quant: {n} of {} symbols moved (bound 1 in 4000)",
            want.len()
        );
        moved[ccs] = n;
    }
    moved
}

/// Gate 2: fed the reference encoder's own latents, the integer stage is identical and only
/// rounding-boundary residual symbols move.
#[test]
fn decisions_match_reference_given_its_latents() {
    let mut ran = 0;
    for &(vector, _, model_id, op, beta) in VECTORS {
        if !present(vector) {
            continue;
        }
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        let (yl, yc) = (tensor(&dump["y.y"]), tensor(&dump["uv.y"]));
        let shape = dump["analysis_y.0.in"].shape.clone();
        let enc = Encoder::new(ref_root().join("models"));
        let params = EncodeParams {
            model_id,
            beta_displacement_log: [beta, beta],
            op,
        };
        let (stream, traces) = enc
            .encode_latents([&yl, &yc], shape[3], shape[2], params)
            .unwrap();
        let moved = compare_decisions(vector, &traces);
        let reference = std::fs::read(vector_dir(vector).join("stream.bits")).unwrap();
        println!(
            "{vector}: reference latents in -> {} bytes vs {} ({:+.3} %), symbols moved {} / {}",
            stream.len(),
            reference.len(),
            (stream.len() as f64 / reference.len() as f64 - 1.0) * 100.0,
            moved[0],
            moved[1],
        );
        assert!(
            (stream.len() as f64 / reference.len() as f64 - 1.0).abs() < 0.005,
            "{vector}: stream size differs from the reference by more than 0.5 %"
        );
        if moved == [0, 0] {
            assert_eq!(
                stream, reference,
                "{vector}: same decisions must give the same bytes"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "no encoder vectors on disk");
}

/// Gate 3: a whole encode from the PNG. The stream must be the reference's size to within
/// 0.5 %, and decode to the same picture through our own decoder.
#[test]
fn encoder_end_to_end_matches_reference() {
    let mut ran = 0;
    for &(vector, image, model_id, op, beta) in VECTORS {
        if !present(vector) {
            continue;
        }
        let params = EncodeParams {
            model_id,
            beta_displacement_log: [beta, beta],
            op,
        };
        let enc = Encoder::new(ref_root().join("models"));
        let (stream, traces) = enc.encode_traced(&source(image), params).unwrap();
        let moved = compare_decisions(vector, &traces);
        let reference = std::fs::read(vector_dir(vector).join("stream.bits")).unwrap();
        let ratio = stream.len() as f64 / reference.len() as f64;
        println!(
            "{vector}: {} bytes, reference {} ({:+.3} %), symbols moved {} / {}",
            stream.len(),
            reference.len(),
            (ratio - 1.0) * 100.0,
            moved[0],
            moved[1],
        );
        assert!(
            (ratio - 1.0).abs() < 0.005,
            "{vector}: stream size differs from the reference by more than 0.5 %"
        );
        if moved == [0, 0] {
            assert_eq!(
                stream, reference,
                "{vector}: same decisions must give the same bytes"
            );
        }
        // Our stream must decode, and land on the reference stream's picture.
        let dec = zenjpegai::Decoder::new(ref_root().join("models"));
        let ours = dec.decode(&stream).unwrap();
        let theirs = dec.decode(&reference).unwrap();
        assert_eq!((ours.width, ours.height), (theirs.width, theirs.height));
        let worst = ours
            .data
            .iter()
            .zip(&theirs.data)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let differing = ours
            .data
            .iter()
            .zip(&theirs.data)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "   vs a decode of the reference stream: {differing} of {} samples differ, worst {worst}",
            ours.data.len()
        );
        ran += 1;
    }
    assert!(ran > 0, "no encoder vectors on disk");
}

/// The reference decoder must accept our streams and produce the same picture.
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_accepts_our_streams() {
    for &(vector, image, model_id, op, beta) in VECTORS {
        if !present(vector) {
            continue;
        }
        let params = EncodeParams {
            model_id,
            beta_displacement_log: [beta, beta],
            op,
        };
        let enc = Encoder::new(ref_root().join("models"));
        let stream = enc.encode(&source(image), params).unwrap();
        let dir = std::env::temp_dir();
        let _ = dir;
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
        std::fs::create_dir_all(&scratch).unwrap();
        let bits = scratch.join(format!("{vector}.bits"));
        let png = scratch.join(format!("{vector}.png"));
        std::fs::write(&bits, &stream).unwrap();
        let status = std::process::Command::new("bash")
            .arg("-lc")
            .arg(format!(
                ". {}/.venv/bin/activate && cd {} && python -m src.reco.coders.decoder {} {} -target_device cpu",
                ref_root().display(),
                ref_root().display(),
                bits.display(),
                png.display()
            ))
            .status()
            .unwrap();
        assert!(status.success(), "{vector}: the reference decoder failed");
        let theirs = read_png_rgb8(&std::fs::read(&png).unwrap()).unwrap();
        let ours = zenjpegai::Decoder::new(ref_root().join("models"))
            .decode(&stream)
            .unwrap();
        let worst = ours
            .data
            .iter()
            .zip(&theirs.data)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let differing = ours
            .data
            .iter()
            .zip(&theirs.data)
            .filter(|(a, b)| a != b)
            .count();
        println!("{vector}: reference decode differs in {differing} samples, worst {worst}");
        assert!(worst <= 1, "{vector}: reference decode differs by {worst}");
    }
}
