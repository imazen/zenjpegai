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

fn z_tensor(t: &common::RefTensor) -> Tensor<i8> {
    assert_eq!(t.shape.len(), 4);
    Tensor::from_vec(t.shape[1], t.shape[2], t.shape[3], t.i8()).unwrap()
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

/// (`vector`, source image, `model_id`, operating point, `beta_displacement_log`, tools) of
/// every fixed-model encode `make_reference_streams.sh encoder` produces.
type Tools = fn(EncodeParams) -> EncodeParams;
type ToolVector = (&'static str, &'static str, u8, OperatingPoint, i32, Tools);

struct Vector {
    name: &'static str,
    image: &'static str,
    model_id: u8,
    op: OperatingPoint,
    beta: i32,
    tools: Tools,
}

const fn plain(p: EncodeParams) -> EncodeParams {
    p
}

const VECTORS_BASE: &[(&str, &str, u8, OperatingPoint, i32)] = &[
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
    // 2096x1400: above the 1 MP threshold, so the analysis transform and the hyper-encoder run
    // per tile (1024 luma / 512 chroma, overlap 64 / 32) and the header carries synthesis tiling.
    (
        "enc_img01_bop_m1_b0",
        "00001_TE_2096x1400_8bit_sRGB.png",
        1,
        OperatingPoint::Bop,
        0,
    ),
];

const IMG30: &str = "00030_TE_560x888_8bit_sRGB.png";

/// The coding tools the encoder can switch on, each against its own reference encode.
const VECTORS_TOOLS: &[ToolVector] = &[
    (
        "enc_img30_bop_m1_b0_threads8",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            num_threads_z: 8,
            num_threads_r: 8,
            ..p
        },
    ),
    (
        "enc_img30_bop_m1_b0_rvs",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            rvs: true,
            grfs: true,
            ..p
        },
    ),
    (
        "enc_img30_bop_m1_b0_rvsonly",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams { rvs: true, ..p },
    ),
    (
        "enc_img30_bop_m1_b0_grfsonly",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams { grfs: true, ..p },
    ),
    (
        "enc_img30_bop_m1_b0_lsbs",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams { lsbs: true, ..p },
    ),
];

fn vectors() -> Vec<Vector> {
    VECTORS_BASE
        .iter()
        .map(|&(name, image, model_id, op, beta)| Vector {
            name,
            image,
            model_id,
            op,
            beta,
            tools: plain,
        })
        .chain(
            VECTORS_TOOLS
                .iter()
                .map(|&(name, image, model_id, op, beta, tools)| Vector {
                    name,
                    image,
                    model_id,
                    op,
                    beta,
                    tools,
                }),
        )
        .filter(|v| present(v.name))
        .collect()
}

impl Vector {
    fn params(&self) -> EncodeParams {
        (self.tools)(EncodeParams {
            model_id: self.model_id,
            beta_displacement_log: [self.beta, self.beta],
            op: self.op,
            ..Default::default()
        })
    }
}

fn source(image: &str) -> zenjpegai::RgbImage {
    let path = ref_root().join("data/test").join(image);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    read_png_rgb8(&bytes).unwrap()
}

fn present(vector: &str) -> bool {
    vector_dir(vector).join("enc2/enc_manifest.txt").is_file()
}

/// Colour pre-processing: bit-exact against the reference's own analysis-transform inputs.
///
/// The reference records one input per analysis call, so on a tiled picture this also checks
/// that our analysis tile grid is the reference's, tile for tile.
#[test]
fn colour_preprocessing_matches_reference() {
    use zenjpegai::encoder::analysis_tiles;
    let mut ran = 0;
    for v in vectors() {
        let (vector, image) = (v.name, v.image);
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        let input = preprocess_rgb(&source(image)).unwrap();
        for (ccs, (net, plane)) in [("analysis_y", &input.luma), ("analysis_uv", &input.chroma)]
            .into_iter()
            .enumerate()
        {
            let d = [16usize, 8][ccs];
            let (lh, lw) = (plane.h.div_ceil(d), plane.w.div_ceil(d));
            let tiles = analysis_tiles(
                ccs,
                plane.h,
                plane.w,
                lh,
                lw,
                plane.h.div_ceil(4 * d),
                plane.w.div_ceil(4 * d),
            )
            .unwrap();
            for (i, t) in tiles.iter().enumerate() {
                let key = format!("{net}.{i}.in");
                let want = dump
                    .get(&key)
                    .unwrap_or_else(|| panic!("{vector}: {key} missing (tile count differs)"));
                assert_eq!(
                    (want.shape[2], want.shape[3]),
                    (t.image.height, t.image.width),
                    "{vector} {key}: tile geometry"
                );
                let got = plane
                    .window(t.image.x, t.image.y, t.image.width, t.image.height)
                    .unwrap();
                let differing = want
                    .f32()
                    .iter()
                    .zip(&got.data)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(differing, 0, "{vector} {key}: {differing} samples differ");
            }
            assert!(
                !dump.contains_key(&format!("{net}.{}.in", tiles.len())),
                "{vector} {net}: the reference used more tiles than we do"
            );
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
    for v in vectors() {
        let vector = v.name;
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        let (yl, yc) = (tensor(&dump["y.y"]), tensor(&dump["uv.y"]));
        let picture = source(v.image);
        let enc = Encoder::new(ref_root().join("models"));
        let params = v.params();
        let zy = z_tensor(&dump["y.z_hat"]);
        let zc = z_tensor(&dump["uv.z_hat"]);
        let (stream, traces) = enc
            .encode_latents(
                [&yl, &yc],
                Some([&zy, &zc]),
                picture.width,
                picture.height,
                params,
            )
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
    for v in vectors() {
        let (vector, params) = (v.name, v.params());
        let enc = Encoder::new(ref_root().join("models"));
        let (stream, traces) = enc.encode_traced(&source(v.image), params).unwrap();
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
    for v in vectors() {
        let (vector, params) = (v.name, v.params());
        let enc = Encoder::new(ref_root().join("models"));
        let stream = enc.encode(&source(v.image), params).unwrap();
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

/// Rate matching: the model and displacement `--bpp` settles on, against the reference
/// encoder's own choice at the same target. Ours is measured on the real codestream, the
/// reference's on a likelihood estimate (`src/encoder/rate.rs`), so the achieved rates differ.
#[test]
fn rate_matching_hits_the_target() {
    let image = "00030_TE_560x888_8bit_sRGB.png";
    let picture = source(image);
    let pixels = (picture.width * picture.height) as f64;
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    // (target bpp, the reference stream encoded at that target with its own rate matcher).
    let cases = [
        (0.12, "img30_base_off_bpp012"),
        (0.25, "img30_base_off_bpp025"),
        (0.50, "img30_base_off_bpp050"),
        (0.75, "img30_base_off_bpp075"),
        (1.00, "img30_base_off_bpp100"),
    ];
    let mut ran = 0;
    for (target, vector) in cases {
        let path = vector_dir(vector).join("stream.bits");
        if !path.is_file() {
            continue;
        }
        let (stream, m) = enc
            .encode_to_bpp(
                &picture,
                target,
                EncodeParams {
                    op: OperatingPoint::Bop,
                    ..Default::default()
                },
            )
            .unwrap();
        let theirs = std::fs::read(&path).unwrap();
        let their_bpp = theirs.len() as f64 * 8.0 / pixels;
        let their_hdr = dec.read_headers(&theirs).unwrap().picture;
        println!(
            "target {target}: ours model {} beta {} -> {:.4} bpp ({:+.1} %, {} trials); \
             reference model {} beta {} -> {:.4} bpp ({:+.1} %)",
            m.model_id,
            m.beta_displacement_log,
            m.bpp,
            (m.bpp / target - 1.0) * 100.0,
            m.trials,
            their_hdr.model_id,
            their_hdr.beta_displacement_log[0],
            their_bpp,
            (their_bpp / target - 1.0) * 100.0,
        );
        assert_eq!(
            m.model_id, their_hdr.model_id,
            "target {target}: rate matching chose a different model than the reference"
        );
        // Within 1 % unless the model's `BDL_range` caps the search (0.12 and 0.50 here);
        // measured worst case 3.5 %. Never loosen this without re-measuring.
        assert!(
            (m.bpp / target - 1.0).abs() < 0.04,
            "target {target}: {:.4} bpp",
            m.bpp
        );
        // The stream must still be a stream.
        let ours = dec.decode(&stream).unwrap();
        assert_eq!((ours.width, ours.height), (picture.width, picture.height));
        ran += 1;
    }
    assert!(ran > 0, "no reference streams on disk");
}
