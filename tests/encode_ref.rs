//! Encoder-side networks against the reference encoder (Gate 1 of the encoder port).
//!
//! Oracle: `scripts/ref_vectors/dump_encode.py OUT --enc2 -- <encoder args>` run on the stock
//! reference encoder (`make_reference_streams.sh encoder`), which records the input and output
//! of every analysis-transform and hyper-encoder call. Float networks: the bounds are a few
//! times the measured worst case (see `PORTING.md`); do not loosen them to make a change pass.
#![cfg(feature = "reference-tests")]

mod common;
use common::{load_encoder_dump, ref_root, vector_dir, vectors_root};
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
use zenjpegai::decoder::reconstruct::Planes;
use zenjpegai::encoder::filters::icci::EicciConfig;
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

/// Colour pre-processing: bit-exact against the reference's own analysis-transform inputs.
///
/// The reference records one input per analysis call, so on a tiled picture this also checks
/// that our analysis tile grid is the reference's, tile for tile.
#[test]
fn colour_preprocessing_matches_reference() {
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
    }
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
    }
}

/// Gate 3: a whole encode from the PNG. The stream must be the reference's size to within
/// 0.5 %, and decode to the same picture through our own decoder.
#[test]
fn encoder_end_to_end_matches_reference() {
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
    }
}

/// Run `cmd` inside the upstream checkout's Python venv (PYTHONPATH set). Hard failure on
/// non-zero exit.
fn ref_run(cmd: &str, vector: &str) {
    let status = std::process::Command::new("bash")
        .arg("-lc")
        .arg(format!(
            ". {ref}/.venv/bin/activate && cd {ref} && PYTHONPATH=. {cmd}",
            ref = ref_root().display()
        ))
        .status()
        .unwrap();
    assert!(status.success(), "{vector}: the reference decoder failed");
}

/// `python -m src.reco.coders.decoder bits out -target_device cpu` on the stock decoder.
fn ref_decode(bits: &Path, out: &Path, vector: &str) {
    ref_run(
        &format!(
            "python -m src.reco.coders.decoder {} {} -target_device cpu",
            bits.display(),
            out.display()
        ),
        vector,
    );
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
        if patched {
            let out = scratch.join(vector);
            ref_run(
                &format!(
                    "python {}/scripts/ref_vectors/dump_decode.py {} {} --contiguous-masks \
                     --fix-qmap-header && cp {}/decoded.png {}",
                    env!("CARGO_MANIFEST_DIR"),
                    bits.display(),
                    out.display(),
                    out.display(),
                    png.display()
                ),
                vector,
            );
        } else {
            ref_decode(&bits, &png, vector);
        }
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

    let dir = vector_dir("enc_img30_bop_m1_b0_lef");
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

    for name in [
        "img30_base_lef_bpp050",
        "img30_base_on_bpp025",
        "img30_base_on_bpp100",
        "img01_base_eiccitiles_lef_bpp050",
    ] {
        let dir = vector_dir(name);
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
    }
}

/// Rate matching: the model and displacement `--bpp` settles on, against the reference
/// encoder's own choice at the same target. The default `RateEstimate::Coded` measures the
/// real codestream, the reference's a likelihood estimate (`src/encoder/rate.rs`), so the
/// achieved rates differ; `likelihood_estimate_picks_the_references_displacements` covers the
/// reference's own measurement (`RateEstimate::Likelihood` picks its displacements exactly).
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
    for (target, vector) in cases {
        let path = vector_dir(vector).join("stream.bits");
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
    }
}

/// E7: `RateEstimate::Likelihood` — the `ECLibLH` estimator the reference's bitrate matcher
/// searches with (`-sum(log2 p)` over `z_hat` and the residual, container excluded) — must
/// pick the reference's own `(model, beta)` at the five CTC rates on both test pictures, and
/// the estimate itself is checked against both the coded size and the reference encoder's own
/// logged per-trial estimates (`encoder.log`'s `bits = tensor([..])` lines).
///
/// The estimate-vs-coded bound is not the 0.5 % of the work-queue brief: the reference's own
/// estimator misses its coded stream by up to ~1.0 % at these rates (it prices the ideal
/// likelihood, not the quantised ANS tables, exp-Golomb tails, flush and container — ours
/// reproduces the reference's logged estimates to four decimals, so the gap is inherent).
/// The bound below is the measured worst case plus margin.
#[test]
fn likelihood_estimate_picks_the_references_displacements() {
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    // (source image, target bpp, reference stream vector)
    let img30 = "00030_TE_560x888_8bit_sRGB.png";
    let img01 = "00001_TE_2096x1400_8bit_sRGB.png";
    let cases = [
        (img30, 0.12, "img30_base_off_bpp012"),
        (img30, 0.25, "img30_base_off_bpp025"),
        (img30, 0.50, "img30_base_off_bpp050"),
        (img30, 0.75, "img30_base_off_bpp075"),
        (img30, 1.00, "img30_base_off_bpp100"),
        (img01, 0.12, "img01_base_off_bpp012"),
        (img01, 0.25, "img01_base_off_bpp025"),
        (img01, 0.50, "img01_base_off_bpp050"),
        (img01, 0.75, "img01_base_off_bpp075"),
        (img01, 1.00, "img01_base_off_bpp100"),
    ];
    for (image, target, vector) in cases {
        let dir = vector_dir(vector);
        let path = dir.join("stream.bits");
        let picture = source(image);
        let pixels = (picture.width * picture.height) as f64;
        let (stream, m) = enc
            .encode_to_bpp(
                &picture,
                target,
                EncodeParams {
                    op: OperatingPoint::Bop,
                    rate_estimate: zenjpegai::encoder::RateEstimate::Likelihood,
                    ..Default::default()
                },
            )
            .unwrap();
        let theirs = std::fs::read(&path).unwrap();
        let their_hdr = dec.read_headers(&theirs).unwrap().picture;
        let estimate = m.estimated_bpp.expect("likelihood mode must report it");
        // The stream the search returns is still a real codestream at the chosen displacement.
        assert_eq!(
            stream.len() as f64 * 8.0 / pixels,
            m.bpp,
            "{vector}: RateMatch::bpp must be the returned stream's rate"
        );
        println!(
            "{vector}: ours model {} beta {} -> {:.4} bpp (est {estimate:.6}); \
             reference model {} beta {} -> {:.4} bpp",
            m.model_id,
            m.beta_displacement_log,
            m.bpp,
            their_hdr.model_id,
            their_hdr.beta_displacement_log[0],
            theirs.len() as f64 * 8.0 / pixels,
        );
        assert_eq!(
            (m.model_id, m.beta_displacement_log),
            (their_hdr.model_id, their_hdr.beta_displacement_log[0]),
            "{vector}: likelihood search picked a different (model, beta) than the reference"
        );
        // The estimator's bit count vs the coded size it stands in for (measured worst case:
        // -1.0 % at img30 0.75 bpp — see the doc comment).
        let rel = estimate / m.bpp - 1.0;
        assert!(
            rel.abs() < 0.015,
            "{vector}: estimate {estimate:.6} vs coded {:.6} ({rel:+.3})",
            m.bpp
        );
        // Stronger oracle: the reference logs `bits = tensor([<est>])` for every trial; its
        // estimate at the winning beta must be ours (the log prints four decimals).
        let log = std::fs::read_to_string(dir.join("encoder.log")).unwrap();
        let marker = format!("beta = {},", m.beta_displacement_log);
        match log
            .lines()
            .rfind(|l| l.contains(&marker) && l.contains("bits = tensor"))
        {
            Some(line) => {
                let ref_est: f64 = line
                    .split("bits = tensor([")
                    .nth(1)
                    .and_then(|s| s.split(']').next())
                    .and_then(|s| s.trim().parse().ok())
                    .expect("encoder.log: unparseable trial line");
                assert!(
                    (estimate - ref_est).abs() < 2e-3,
                    "{vector}: estimate {estimate:.6} vs the reference's {ref_est} at beta {}",
                    m.beta_displacement_log
                );
            }
            // The reference logs the per-trial estimates only while its search iterates; when
            // the first candidate lands inside tolerance the log carries just the "best model"
            // decision (img30_base_off_bpp100), so there is no trial number to check — assert
            // the decision is logged rather than silently passing.
            None => assert!(
                log.contains("best model is"),
                "{vector}: encoder.log has neither a trial estimate nor the matcher's decision"
            ),
        }
    }
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
        let dir = vectors_root().parent().unwrap().join("inputs");
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
    }
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
        ref_decode(&bits, &out, vector);
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

// ---------------------------------------------------------------------------------------------
// Gate 4: eICCI encoder-side model selection (`icci_filter.py::compress` +
// `model_idxes.py::encode_header`) and the MS-SSIM loss it scores with.

/// `pytorch_msssim == 0.2.1`'s `ms_ssim(x, y, data_range = 1)` as the oracle:
/// `scripts/ref_vectors/dump_filters.py --msssim` wrote deterministic planes and the
/// reference's scores to `msssim_oracle/`. Tolerance covers torch's blocked f32 reductions vs
/// this port's f64 ones (measured in `PORTING.md`).
#[test]
fn msssim_matches_pytorch_msssim() {
    use zenjpegai::encoder::msssim::ms_ssim;
    let dump = common::load_dump(&vectors_root().join("msssim_oracle"));
    let eng = Engine::new();
    let (mut worst, mut n) = (0.0f32, 0);
    for i in 0..usize::MAX {
        let Some(x) = dump.get(&format!("msssim.{i}.x")) else {
            break;
        };
        let y = &dump[&format!("msssim.{i}.y")];
        let want = dump[&format!("msssim.{i}.val")].f32()[0];
        let x = Tensor::from_vec(1, x.shape[2], x.shape[3], x.f32()).unwrap();
        let y = Tensor::from_vec(1, y.shape[2], y.shape[3], y.f32()).unwrap();
        let got = ms_ssim(&eng, &x, &y).unwrap();
        println!(
            "msssim.{i}: reference {want} ours {got} (diff {:e})",
            got - want
        );
        worst = worst.max((got - want).abs());
        n += 1;
    }
    assert!(n >= 5, "msssim_oracle: {n} pairs dumped");
    assert!(worst <= 1e-5, "MS-SSIM vs pytorch_msssim: {worst:e}");
}

/// `eicci.selection` parity: `select` fed the reference's own `eicci.in` planes (the decoder
/// dump's filter input — the same picture `compress` decided on, EFE output included for the
/// tools_on vectors) must signal the stream's per-tile model indices. Mismatches are counted
/// and printed per channel for the PORTING.md tie/near-tie forensics.
fn check_eicci_selection(name: &str, image: &str, cfg: EicciConfig) -> (usize, usize) {
    use zenjpegai::decoder::read_headers;
    use zenjpegai::encoder::filters::icci;
    use zenjpegai::model::icci::NetCache;

    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let cs = zenjpegai::container::Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let want = headers.tools.icci.as_ref().expect("stream without eICCI");
    let dump = common::load_dump(&dir.join("filters_lef_icci"));
    let rec = Planes {
        y: plane(&dump["eicci.in.a"]),
        u: plane(&dump["eicci.in.b"]),
        v: plane(&dump["eicci.in.c"]),
    };
    let src = SourceImage::Rgb(source(image));
    let meta = src.meta(None, None).unwrap();
    let org = icci::org_planes(&src, &meta).unwrap();
    let models = ModelDir::new(ref_root().join("models"));
    let got = icci::select(
        &Engine::new(),
        &models,
        &NetCache::default(),
        &headers.picture,
        headers.picture.synthesis_transforms[0],
        &org,
        &rec,
        &cfg,
        &enough::Unstoppable,
    )
    .unwrap()
    .expect("eICCI selection");
    assert_eq!(got.tiling, want.tiling, "{name}: eICCI tiling");
    assert_eq!(
        got.tiles.len(),
        want.tiles.len(),
        "{name}: eICCI tile count"
    );
    let mut mismatched = 0usize;
    for (t, (g, w)) in got.tiles.iter().zip(&want.tiles).enumerate() {
        if g != w {
            mismatched += 1;
            println!("{name} tile {t}: ours {g:?} reference {w:?}");
        }
    }
    (mismatched, want.tiles.len())
}

/// One channel of a reference-dump plane (`eicci.in.*` etc., `[1, 1, h, w]` f32).
fn plane(t: &common::RefTensor) -> Tensor<f32> {
    assert_eq!(t.shape.len(), 4);
    Tensor::from_vec(t.shape[1], t.shape[2], t.shape[3], t.f32()).unwrap()
}

/// The vectors the reference encoder's eICCI search actually produced a selection for (the
/// forced 4:2:0/4:2:2 streams carry an override — `force_icci_encode.py` — not a search).
#[test]
fn eicci_selection_matches_reference() {
    let shipped = EicciConfig::default();
    let mut bad = 0usize;
    let mut total = 0usize;
    for (name, cfg) in [
        ("img30_base_eicci_bpp050", shipped),
        ("img30_base_on_bpp025", shipped),
        ("img30_base_on_bpp100", shipped),
        (
            "img01_base_eiccitiles_lef_bpp050",
            EicciConfig {
                tile_samples: 1_048_576,
                ..shipped
            },
        ),
    ] {
        let image = if name.starts_with("img01") {
            IMG01
        } else {
            IMG30
        };
        let (m, n) = check_eicci_selection(name, image, cfg);
        println!("{name}: {m} of {n} tile selections differ from the reference");
        bad += m;
        total += n;
    }
    println!("eICCI selection: {bad} of {total} tile selections differ");
    assert_eq!(bad, 0, "eICCI model selection diverges from the reference");
}

/// The reference decoder must accept our eICCI stream — the grammar puts `icci_enable_flag`
/// and the per-tile indices inside the tool header, so a desynchronising write would fail
/// loudly. Encodes `img30_base_eicci_bpp050`'s settings and hands the stream to
/// `src.reco.coders.decoder`.
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_accepts_eicci_stream() {
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    let reference =
        std::fs::read(vector_dir("img30_base_eicci_bpp050").join("stream.bits")).unwrap();
    let want = dec.read_headers(&reference).unwrap();
    let stream = Encoder::new(ref_root().join("models"))
        .encode(
            source(IMG30),
            EncodeParams {
                model_id: want.picture.model_id,
                beta_displacement_log: want.picture.beta_displacement_log,
                op: want.picture.synthesis_transforms[0],
                eicci: Some(EicciConfig::default()),
                ..Default::default()
            },
        )
        .unwrap();
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
    std::fs::create_dir_all(&scratch).unwrap();
    let bits = scratch.join("img30_eicci.bits");
    let png = scratch.join("img30_eicci.png");
    std::fs::write(&bits, &stream).unwrap();
    ref_decode(&bits, &png, "img30_eicci");
    let theirs = read_png_rgb8(&std::fs::read(&png).unwrap()).unwrap();
    let ours = dec.decode(&stream).unwrap();
    let worst = ours
        .data
        .iter()
        .zip(&theirs.data)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    println!("reference decode of our eICCI stream: worst diff {worst}");
    assert!(worst <= 1);
}

/// LEF on top of eICCI (`img01_base_eiccitiles_lef_bpp050` ran both tools).
fn lef_on(p: EncodeParams) -> EncodeParams {
    EncodeParams { lef: true, ..p }
}

/// A whole encode with eICCI on: the tool header our stream carries must be the reference
/// stream's, and the stream must still decode.
#[test]
fn eicci_encode_end_to_end() {
    use zenjpegai::decoder::read_headers;
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    for (name, image, cfg, tools) in [
        (
            "img30_base_eicci_bpp050",
            IMG30,
            EicciConfig::default(),
            plain as Tools,
        ),
        (
            "img01_base_eiccitiles_lef_bpp050",
            IMG01,
            EicciConfig {
                tile_samples: 1_048_576,
                ..EicciConfig::default()
            },
            lef_on as Tools,
        ),
    ] {
        let reference = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
        let want = dec.read_headers(&reference).unwrap();
        let want_icci = want.tools.icci.as_ref().expect("stream without eICCI");
        let stream = enc
            .encode(
                source(image),
                tools(EncodeParams {
                    model_id: want.picture.model_id,
                    beta_displacement_log: want.picture.beta_displacement_log,
                    op: want.picture.synthesis_transforms[0],
                    eicci: Some(cfg),
                    ..Default::default()
                }),
            )
            .unwrap();
        let got = read_headers(&zenjpegai::container::Codestream::parse(&stream).unwrap())
            .unwrap()
            .tools
            .icci
            .expect("our stream has no eICCI header");
        assert_eq!(
            got, *want_icci,
            "{name}: eICCI tool header differs from the reference's"
        );
        // And our stream still decodes to the reference picture's neighbourhood.
        let ours = dec.decode(&stream).unwrap();
        let theirs = dec.decode(&reference).unwrap();
        let worst = ours
            .data
            .iter()
            .zip(&theirs.data)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 1, "{name}: decoded output differs by {worst}");
        println!("{name}: eICCI encode end-to-end, decode worst diff {worst}");
    }
}

/// The reference only ever ships `process_short_list = 1`; the long list (all ten networks)
/// has no vector. Exercise it end to end: the stream must carry a parseable long-list header
/// and decode.
#[test]
fn eicci_long_list_encodes() {
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    let want = dec
        .read_headers(
            &std::fs::read(vector_dir("img30_base_eicci_bpp050").join("stream.bits")).unwrap(),
        )
        .unwrap();
    let stream = enc
        .encode(
            source(IMG30),
            EncodeParams {
                model_id: want.picture.model_id,
                beta_displacement_log: want.picture.beta_displacement_log,
                op: want.picture.synthesis_transforms[0],
                eicci: Some(EicciConfig {
                    short_list: false,
                    ..EicciConfig::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
    let got = dec.read_headers(&stream).unwrap();
    let icci = got.tools.icci.expect("our stream has no eICCI header");
    for t in &icci.tiles {
        assert!(!t.short_list, "long-list encode signalled the short list");
        if t.use_yuv[0] {
            assert!(t.index_y < 10);
        }
        if t.use_yuv[1] || t.use_yuv[2] {
            assert!(t.index_uv < 10);
        }
    }
    dec.decode(&stream).unwrap();
}

/// Selection is deterministic: every SIMD tier and thread mode must produce the identical
/// stream — the nets are bit-identical across tiers and `ms_ssim` reduces in `f64`.
#[test]
fn eicci_tiers_agree_bit_for_bit() {
    let src = source(IMG30);
    let want = zenjpegai::Decoder::new(ref_root().join("models"))
        .read_headers(
            &std::fs::read(vector_dir("img30_base_eicci_bpp050").join("stream.bits")).unwrap(),
        )
        .unwrap();
    let params = EncodeParams {
        model_id: want.picture.model_id,
        beta_displacement_log: want.picture.beta_displacement_log,
        op: want.picture.synthesis_transforms[0],
        eicci: Some(EicciConfig::default()),
        ..Default::default()
    };
    let mut first: Option<Vec<u8>> = None;
    for tier in Tier::available() {
        for parallel in [false, true] {
            let enc = Encoder::with_engine(ref_root().join("models"), Engine::with(tier, parallel));
            let stream = enc.encode(src.clone(), params).unwrap();
            match &first {
                None => first = Some(stream),
                Some(f) => assert_eq!(*f, stream, "{tier:?} parallel={parallel}"),
            }
        }
    }
}
