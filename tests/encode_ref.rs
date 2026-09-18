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
use zenjpegai::tools::qualmap::QualityMap;

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
use zenjpegai::encoder::{
    EncodeParams, Encoder, RegionMode, SourceImage, preprocess_rgb, read_png_rgb8,
};

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
const IMG01: &str = "00001_TE_2096x1400_8bit_sRGB.png";

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
    (
        "enc_img30_bop_m1_b0_lef",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams { lef: true, ..p },
    ),
    (
        "enc_img30_bop_m1_b0_qmap",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        plain,
    ),
    (
        "enc_img30_bop_m1_b0_qmap_rvs",
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
        "enc_img01_bop_m1_b0_depregions",
        IMG01,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            regions: Some(RegionMode::Dependent),
            ..p
        },
    ),
    (
        "enc_img01_bop_m1_b0_indregions",
        IMG01,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            regions: Some(RegionMode::Independent),
            ..p
        },
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
    /// The ROI mask the `qmap` vectors were encoded with, as a quality map at latent resolution.
    fn quality_map(&self) -> Option<QualityMap> {
        if !self.name.contains("qmap") {
            return None;
        }
        let path = vector_dir(self.name)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("masks/img30_roi.png");
        let mask = read_png_rgb8(&std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (run make_reference_streams.sh encoder)",
                path.display()
            )
        }))
        .unwrap();
        let mut planes = Tensor::<u8>::zeros(3, mask.height, mask.width).unwrap();
        for (i, p) in mask.data.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in p.iter().enumerate() {
                planes.plane_mut(c)[i] = v as u8;
            }
        }
        let picture = source(self.image);
        Some(
            QualityMap::from_roi_mask(
                &planes,
                picture.height.div_ceil(16),
                picture.width.div_ceil(16),
            )
            .unwrap(),
        )
    }

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
    let mut ran = 0;
    for v in vectors() {
        let (vector, image) = (v.name, v.image);
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        let picture = source(image);
        let input = preprocess_rgb(&picture).unwrap();
        for (ccs, (net, plane)) in [("analysis_y", &input.luma), ("analysis_uv", &input.chroma)]
            .into_iter()
            .enumerate()
        {
            let tiles = Encoder::analysis_tile_grid(picture.width, picture.height, ccs, v.params())
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
        let qmap = v.quality_map();
        let (stream, traces) = enc
            .encode_latents(
                [&yl, &yc],
                Some([&zy, &zc]),
                picture.width,
                picture.height,
                params,
                qmap.as_ref(),
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
        let (stream, traces) = enc
            .encode_traced_with(source(v.image), params, v.quality_map().as_ref())
            .unwrap();
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
///
/// Region streams go through `dump_decode.py --contiguous-masks`: the stock reference decoder
/// mis-decodes *any* stream with `region_partitioning_flag = 1`, its own included (the
/// non-contiguous mask it hands the C++ coder; see `PORTING.md`), so the patched decoder is the
/// only oracle for them.
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_accepts_our_streams() {
    for v in vectors() {
        let (vector, params) = (v.name, v.params());
        let enc = Encoder::new(ref_root().join("models"));
        let stream = match v.quality_map() {
            None => enc.encode(source(v.image), params).unwrap(),
            Some(m) => enc
                .encode_with_quality_map(source(v.image), params, &m)
                .unwrap(),
        };
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
        std::fs::create_dir_all(&scratch).unwrap();
        let bits = scratch.join(format!("{vector}.bits"));
        let png = scratch.join(format!("{vector}.png"));
        std::fs::write(&bits, &stream).unwrap();
        // The stock reference decoder cannot read a region stream (any of them, its own
        // included) or a quality map; `dump_decode.py` patches both defects at runtime.
        let patched = params.regions.is_some() || vector.contains("qmap");
        let cmd = if patched {
            let out = scratch.join(vector);
            format!(
                "python {}/scripts/ref_vectors/dump_decode.py {} {} --contiguous-masks \
                 --fix-qmap-header && cp {}/decoded.png {}",
                env!("CARGO_MANIFEST_DIR"),
                bits.display(),
                out.display(),
                out.display(),
                png.display()
            )
        } else {
            format!(
                "python -m src.reco.coders.decoder {} {} -target_device cpu",
                bits.display(),
                png.display()
            )
        };
        let status = std::process::Command::new("bash")
            .arg("-lc")
            .arg(format!(
                ". {ref}/.venv/bin/activate && cd {ref} && PYTHONPATH=. {cmd}",
                ref = ref_root().display()
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
        println!(
            "{vector}: reference decode differs in {differing} of {} samples, worst {worst}",
            ours.data.len()
        );
        assert!(worst <= 1, "{vector}: reference decode differs by {worst}");
    }
}

/// `LEF.analyze`: the reference channel is the channel of the luma `scale_log` with the
/// highest mean. Checked against the reference encoder's own choice on the fixed-model LEF
/// vector (dumped by `dump_encode.py --lef` as `lef.ch_idx`, with `lef.avg_sig` holding the
/// per-channel means it chose from), and against the `LEF_chIdx` signalled on the
/// rate-matched LEF streams (their scale map comes from the decoder dump — the entropy stage
/// is integer-exact, so it is the map the reference encoder's `analyze` saw).
#[test]
fn lef_channel_matches_reference() {
    use zenjpegai::encoder::filters::lef::reference_channel;
    let mut ran = 0;

    let dir = vector_dir("enc_img30_bop_m1_b0_lef");
    if dir.join("enc2/enc_manifest.txt").is_file() {
        let dump = load_encoder_dump(&dir.join("enc2"));
        let s = &dump["y.scale_log"];
        let scale_log = Tensor::from_vec(s.shape[1], s.shape[2], s.shape[3], s.i32()).unwrap();
        let want = dump["lef.ch_idx"].i32()[0];
        let got = reference_channel(&scale_log).unwrap();
        assert_eq!(got as i32, want, "enc_img30_bop_m1_b0_lef: LEF_chIdx");
        // The means the argmax sees are torch's (`scale_log.float()` then `mean`): the planes
        // are small enough that the f32 sums are exact, so they must agree bit for bit.
        let n = (scale_log.h * scale_log.w) as f32;
        for (ch, &w) in dump["lef.avg_sig"].f32().iter().enumerate() {
            let m = scale_log.plane(ch).iter().map(|&v| v as f32).sum::<f32>() / n;
            assert_eq!(m.to_bits(), w.to_bits(), "lef.avg_sig[{ch}]");
        }
        ran += 1;
    }

    for name in [
        "img30_base_lef_bpp050",
        "img30_base_on_bpp025",
        "img30_base_on_bpp100",
        "img01_base_eiccitiles_lef_bpp050",
    ] {
        let dir = vector_dir(name);
        if !dir.join("manifest.txt").is_file() {
            continue;
        }
        let stream = std::fs::read(dir.join("stream.bits")).unwrap();
        let headers = zenjpegai::decoder::read_headers(
            &zenjpegai::container::Codestream::parse(&stream).unwrap(),
        )
        .unwrap();
        let want = headers
            .tools
            .lef_channel
            .expect("{name}: stream without LEF");
        let dump = common::load_dump(&dir);
        let s = &dump["y.scale_log"];
        let scale_log = Tensor::from_vec(s.shape[1], s.shape[2], s.shape[3], s.i32()).unwrap();
        let got = reference_channel(&scale_log).unwrap();
        assert_eq!(got, want, "{name}: LEF_chIdx");
        ran += 1;
    }
    assert!(ran > 0, "no LEF vectors on disk");
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

// ---------------------------------------------------------------------------------------------
// The `formats` vectors: chroma-subsampled, 10-bit and YUV sources, the `c_*_value` overrides
// and a non-displayed border (`make_reference_streams.sh`; inputs under `reference/inputs`).

/// One `formats` vector: the directory name, the input file (`reference/inputs` for `.yuv`,
/// `data/test` for PNG) and the encode options `make_reference_streams.sh` gave it.
struct FormatVector {
    name: &'static str,
    file: &'static str,
    /// `-c_ver_value`.
    c_ver: Option<u8>,
    /// `-c_hor_value`.
    c_hor: Option<u8>,
    /// `-diff_display_img_width`, `-diff_display_img_height`.
    diff_display: (u8, u8),
}

const VECTORS_FORMATS: &[FormatVector] = &[
    FormatVector {
        name: "img30yuv420_base_off_bpp050",
        file: "img30_560x888_8bit_420.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30yuv422_base_off_bpp050",
        file: "img30_560x888_8bit_422.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30yuv444_base_off_bpp050",
        file: "img30_560x888_8bit_444.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30yuv420b10_base_off_bpp050",
        file: "img30_560x888_10bit_420.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30yuv444b10_base_off_bpp050",
        file: "img30_560x888_10bit_444.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30cropyuv420_base_off_bpp075",
        file: "img30crop_203x301_8bit_420.yuv",
        c_ver: None,
        c_hor: None,
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30_base_off_c420_bpp050",
        file: IMG30,
        c_ver: Some(2),
        c_hor: Some(2),
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30_base_off_c422_bpp050",
        file: IMG30,
        c_ver: None,
        c_hor: Some(2),
        diff_display: (0, 0),
    },
    FormatVector {
        name: "img30_base_off_display_m1",
        file: IMG30,
        c_ver: None,
        c_hor: None,
        diff_display: (37, 5),
    },
];

/// `read_file`: `.yuv` input comes from the reference's `inputs/` directory (geometry, bit
/// depth and chroma format in the file name), PNG from `data/test`.
fn formats_source(file: &str) -> SourceImage {
    if file.ends_with(".yuv") {
        let dir = vector_dir("img30yuv420_base_off_bpp050")
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("inputs");
        let path = dir.join(file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        SourceImage::read_yuv(&path.to_string_lossy(), &bytes).unwrap()
    } else {
        SourceImage::from(source(file))
    }
}

/// Encode the `formats` vectors: the header must carry the source's subsampling, bit depth,
/// colour transform and display crop; the analysis-transform inputs must be bit-for-bit the
/// reference's (`enc2` dumps); the integer stage must match its decisions; and the stream must
/// be byte for byte when no residual symbol sits on a rounding boundary (the same rule as
/// `decisions_match_reference_given_its_latents`).
#[test]
fn formats_streams_match_reference() {
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new("");
    let mut ran = 0;
    for &FormatVector {
        name: vector,
        file,
        c_ver,
        c_hor,
        diff_display,
    } in VECTORS_FORMATS
    {
        let reference = std::fs::read(vector_dir(vector).join("stream.bits")).unwrap();
        let theirs = dec.read_headers(&reference).unwrap().picture;
        let src = formats_source(file);
        let params = EncodeParams {
            model_id: theirs.model_id,
            beta_displacement_log: theirs.beta_displacement_log,
            op: OperatingPoint::Bop,
            c_ver,
            c_hor,
            diff_display,
            ..Default::default()
        };
        let (stream, traces) = enc.encode_traced(&src, params).unwrap();
        let ours = dec.read_headers(&stream).unwrap().picture;
        assert_eq!(ours.bit_depth, theirs.bit_depth, "{vector}: bit_depth");
        assert_eq!(
            (ours.s_ver, ours.s_hor, ours.c_ver, ours.c_hor),
            (theirs.s_ver, theirs.s_hor, theirs.c_ver, theirs.c_hor),
            "{vector}: subsampling"
        );
        assert_eq!(
            ours.colour_transform, theirs.colour_transform,
            "{vector}: colour transform"
        );
        assert_eq!(
            (ours.diff_display_width, ours.diff_display_height),
            (theirs.diff_display_width, theirs.diff_display_height),
            "{vector}: display crop"
        );
        // The tensors committed to the analysis transforms (`enc2` dump) must come out
        // bit-for-bit: they are integer/float data flow, no network numerics.
        let meta = src.meta(c_ver, c_hor).unwrap();
        let input = zenjpegai::encoder::preprocess(&src, &meta).unwrap();
        let dump = load_encoder_dump(&vector_dir(vector).join("enc2"));
        assert_eq!(
            input.luma.data,
            dump["analysis_y.0.in"].f32(),
            "{vector}: analysis luma input"
        );
        assert_eq!(
            input.chroma.data,
            dump["analysis_uv.0.in"].f32(),
            "{vector}: analysis chroma input"
        );
        let moved = compare_decisions(vector, &traces);
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
        ran += 1;
    }
    assert!(ran > 0, "no formats vectors on disk");
}

/// The `formats` streams must decode in the reference decoder. Its output is compared with
/// our own decoder's on the same stream — a YUV stream to a `WxH_Nbit_FMT.yuv` (the file
/// name carries the format for `write_yuv`/`extract_info`), an RGB stream to `.png` — with
/// the same bound `check_dump` applies (worst 1 LSB, fewer than 1 in 5000 samples).
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_accepts_formats_streams() {
    use zenjpegai::Picture;
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
    std::fs::create_dir_all(&scratch).unwrap();
    for &FormatVector {
        name: vector,
        file,
        c_ver,
        c_hor,
        diff_display,
    } in VECTORS_FORMATS
    {
        let reference = std::fs::read(vector_dir(vector).join("stream.bits")).unwrap();
        let theirs = dec.read_headers(&reference).unwrap().picture;
        let src = formats_source(file);
        let stream = enc
            .encode(
                &src,
                EncodeParams {
                    model_id: theirs.model_id,
                    beta_displacement_log: theirs.beta_displacement_log,
                    op: OperatingPoint::Bop,
                    c_ver,
                    c_hor,
                    diff_display,
                    ..Default::default()
                },
            )
            .unwrap();
        let bits = scratch.join(format!("{vector}.bits"));
        std::fs::write(&bits, &stream).unwrap();
        let ours = dec.decode_picture(&stream).unwrap();
        // The output name must carry the picture geometry/format for `write_yuv`.
        let out = match &ours {
            Picture::Yuv(y) => {
                let fmt = match (theirs.s_ver, theirs.s_hor) {
                    (1, 1) => "444",
                    (1, 2) => "422",
                    (2, 2) => "420",
                    _ => unreachable!(),
                };
                scratch.join(format!(
                    "{vector}_{}x{}_{}bit_{}.yuv",
                    y.width, y.height, y.bit_depth, fmt
                ))
            }
            Picture::Rgb(_) => scratch.join(format!("{vector}.png")),
        };
        let status = std::process::Command::new("bash")
            .arg("-lc")
            .arg(format!(
                ". {ref}/.venv/bin/activate && cd {ref} && PYTHONPATH=. \
                 python -m src.reco.coders.decoder {} {} -target_device cpu",
                bits.display(),
                out.display(),
                ref = ref_root().display()
            ))
            .status()
            .unwrap();
        assert!(status.success(), "{vector}: the reference decoder failed");
        let reference_out = std::fs::read(&out).unwrap();
        // Samples in bit-depth units: 1 byte at 8 bit, little-endian u16 above.
        let plane_bytes = |p: &[u16], depth: u8| -> Vec<u8> {
            if depth > 8 {
                p.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                p.iter().map(|&v| v as u8).collect()
            }
        };
        let (got, want, depth): (Vec<i64>, Vec<i64>, u8) = match &ours {
            Picture::Yuv(y) => (
                [&y.y, &y.u, &y.v]
                    .iter()
                    .flat_map(|p| plane_bytes(p, y.bit_depth))
                    .map(i64::from)
                    .collect(),
                reference_out.iter().map(|&b| i64::from(b)).collect(),
                y.bit_depth,
            ),
            Picture::Rgb(rgb) => {
                let png = zenjpegai::encoder::read_png_rgb(&reference_out).unwrap();
                // A >8-bit PNG stores the samples shifted left to 16 bit.
                let shift = (png.bit_depth - rgb.bit_depth) as i64;
                (
                    rgb.data.iter().map(|&v| i64::from(v)).collect(),
                    png.data.iter().map(|&v| i64::from(v) >> shift).collect(),
                    rgb.bit_depth,
                )
            }
        };
        assert_eq!(got.len(), want.len(), "{vector}: reference output size");
        let mut n = 0usize;
        let mut worst = 0i64;
        for (&a, &b) in got.iter().zip(&want) {
            let d = (a - b).abs();
            n += (d != 0) as usize;
            worst = worst.max(d);
        }
        println!(
            "{vector}: reference output agrees within {worst} LSB ({n} of {} {depth}-bit \
             samples differ)",
            got.len()
        );
        assert!(
            worst <= 1 && n * 5000 < got.len(),
            "{vector}: reference output differs by {worst} in {n} samples"
        );
    }
}
