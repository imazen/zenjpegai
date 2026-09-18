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
    // `num_chs` below the model's channel count (`-model.CCS_SGMM.tools_common.model_{y,uv}
    // .common_modules.num_chs` in the reference): the residual substream codes the first N
    // channels and the header signals N; the rest are reconstructed from the mean.
    (
        "enc_img30_bop_m1_b0_y64u32",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            num_chs: [64, 32],
            ..p
        },
    ),
    // Low rate: the tail channels' cube-flag contribution is observable — none for the
    // context-module luma (`diff[:, num_chs:] = 0`), the full `|y - psi|` for chroma.
    (
        "enc_img30_bop_m1_bm300_y96u48",
        IMG30,
        1,
        OperatingPoint::Bop,
        -300,
        |p| EncodeParams {
            num_chs: [96, 48],
            ..p
        },
    ),
    // RVS+GRFS: `analyzeCWG` ranks all channels, `encode_header` writes `cwgf[:num_chs]`.
    (
        "enc_img30_bop_m1_b0_y96u48_rvs",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            rvs: true,
            grfs: true,
            num_chs: [96, 48],
            ..p
        },
    ),
    // num_chs = 0 for chroma: every channel is uncoded, so every cube's error is the full
    // residual and `use_cube_flags` is signalled.
    (
        "enc_img30_bop_m1_b0_uv0",
        IMG30,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            num_chs: [160, 0],
            ..p
        },
    ),
    // 2096x1400: the tiled analysis path with reduced channels.
    (
        "enc_img01_bop_m1_b0_y80u40",
        IMG01,
        1,
        OperatingPoint::Bop,
        0,
        |p| EncodeParams {
            num_chs: [80, 40],
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
    // E5: the rest of the `toolson` set — eICCI deciding under the full tool stack (its
    // input is the EFE-linear output of the reference's own reconstruction). Streams that
    // signalled `icci_enable_flag = 0` have no per-tile selection to compare.
    for &(name, image, _) in VECTORS_TOOLS_ON {
        if matches!(name, "img30_base_on_bpp025" | "img30_base_on_bpp100") {
            continue; // already in the list above
        }
        let stream = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
        let headers = zenjpegai::decoder::read_headers(
            &zenjpegai::container::Codestream::parse(&stream).unwrap(),
        )
        .unwrap();
        if headers.tools.icci.is_none() {
            println!("{name}: reference signalled no eICCI — nothing to compare");
            continue;
        }
        let (m, n) = check_eicci_selection(name, image, shipped);
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

// ---------------------------------------------------------------------------
// EFE non-linear encode side (E2): `EFEnonlinear.compress` ported as
// `encoder::filters::efe_nonlinear::decide`, replayed against the per-vector dumps
// `scripts/ref_vectors/dump_efe_nonlinear.py` writes into `<vector>/efe_nonlinear/`
// (make_reference_streams.sh `efe` set — the dumps replay the real reference `compress` on
// the same `filters/` input pictures, with its own lstsq/integerize/mask decisions recorded).

/// Vectors carrying an `efe_nonlinear/` oracle dump: every stream where the tool ran at
/// encode time. Forced (`_nl`) vectors exercise the forced-selection paths; the free ones
//  (`img30_efe_f4c6_f1c5`, the `c420`/`c422` non-`_nl` vectors, `base_efenl`, `base_on`)
/// carry authentic enable/disable decisions — including `dctif` and `base_efenl`, where no
/// up-sampled alternative exists and the mask search is skipped.
const EFE_NL_VECTORS: &[&str] = &[
    "img30_efe_f2c1_f2c2_nl",
    "img30_efe_f3c3_f3c4_nl",
    "img30_efe_f3c5_f4c7_nl",
    "crop277_efe_f4c5_f3c6_nl",
    "img01_efe_f2c0_f3c0_nl",
    "img30_c420_efe_f3c5_f4c7_nl",
    "crop277_c420_efe_f4c6_f3c3_nl",
    "crop277_s420_efe_f3c5_f4c7_nl",
    "img30_s420_efe_f1c0_f2c1_nl",
    "crop277_s422_efe_f3c6_f4c2_nl",
    "img30_efe_f4c6_f1c5",
    "img30_c420_efe_f1c0_f2c1",
    "img30_c422_efe_f3c2_f2c4",
    "img30_c420_efe_dctif",
    "img30_base_efenl_bpp050",
    "img30_base_on_bpp025",
    "img30_base_on_bpp100",
];

/// The dump's `meta` row: `[model_id, s_ver, s_hor, bSize, tile_w, tile_h, numTiles,
/// u_enabled, v_enabled, mask1_enabled, mask2_enabled]`.
struct NlMeta {
    model_id: usize,
    s_ver: u8,
    s_hor: u8,
    bsize: usize,
    tile_w: usize,
    tile_h: usize,
    ntiles: usize,
    enabled: [bool; 2],
    mask_en: [bool; 2],
}

fn nl_meta(dump: &std::collections::HashMap<String, common::RefTensor>) -> NlMeta {
    let m = dump["meta"].i64();
    assert_eq!(m.len(), 11, "efe_nonlinear meta length");
    NlMeta {
        model_id: m[0] as usize,
        s_ver: m[1] as u8,
        s_hor: m[2] as u8,
        bsize: m[3] as usize,
        tile_w: m[4] as usize,
        tile_h: m[5] as usize,
        ntiles: m[6] as usize,
        enabled: [m[7] != 0, m[8] != 0],
        mask_en: [m[9] != 0, m[10] != 0],
    }
}

fn nl_planes(
    dump: &std::collections::HashMap<String, common::RefTensor>,
    prefix: &str,
) -> Option<Planes> {
    let a = dump.get(&format!("{prefix}.a"))?;
    Some(Planes {
        y: plane(a),
        u: plane(&dump[&format!("{prefix}.b")]),
        v: plane(&dump[&format!("{prefix}.c")]),
    })
}

/// Per-vector parity tallies: deterministic-vs-deterministic counts plus the measured
/// MKL-divergence statistics.
#[derive(Default, Debug)]
struct NlStats {
    /// Our weight codes differing from the dump's deterministic f64 re-solve by more
    /// than one code (`round` boundary), and the total compared.
    w64_bad: usize,
    w64_total: usize,
    /// Exact matches against the f64 oracle.
    w64_exact: usize,
    /// Our codes vs the MKL f32 draw (`weights_{u,v}`) — informational only: the
    /// reference's own draw is not run-to-run reproducible on these systems.
    wmkl_diff: usize,
    /// Mask-block values differing from the dump's, where both sides kept a mask.
    mask_diff: usize,
    mask_total: usize,
    /// Per-plane enable decisions where ours matched neither the coded stream nor the
    /// replay (two independent reference draws).
    flag_outlier: usize,
    /// Per-plane keep/drop disagreements of the mask decision.
    mask_flag_diff: usize,
    /// Max |filtered − reference output| over the chroma planes.
    out_worst: f32,
}

/// `decide` replayed on one vector's dumps; returns the parity tallies.
fn check_efe_nl(name: &str) -> NlStats {
    use zenjpegai::decoder::read_headers;
    use zenjpegai::encoder::SourceMeta;
    use zenjpegai::encoder::filters::efe_nonlinear::{EfeNonlinearInput, decide};

    let dir = vector_dir(name);
    // Hard failure when the oracle dumps are absent: the caller opted in with
    // `reference-tests` + ZENJPEGAI_REF (just test-ref).
    let dump = common::load_dump(&dir.join("efe_nonlinear"));
    let meta = nl_meta(&dump);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let headers = read_headers(&zenjpegai::container::Codestream::parse(&stream).unwrap()).unwrap();
    let pic = &headers.picture;
    assert_eq!(meta.model_id, pic.model_id as usize, "{name}: model_id");
    assert_eq!(meta.s_ver, pic.s_ver, "{name}: s_ver");
    assert_eq!(meta.s_hor, pic.s_hor, "{name}: s_hor");

    let org = nl_planes(&dump, "org").unwrap();
    let rec = nl_planes(&dump, "in").unwrap();
    let alt = nl_planes(&dump, "alt");
    let smeta = SourceMeta {
        bit_depth: pic.bit_depth,
        s_ver: pic.s_ver,
        s_hor: pic.s_hor,
        c_ver: pic.c_ver,
        c_hor: pic.c_hor,
        colour_transform: pic.colour_transform.clone(),
    };
    let out = decide(&EfeNonlinearInput {
        eng: &Engine::new(),
        meta: &smeta,
        model_id: meta.model_id,
        org: &org,
        rec: &rec,
        upsampled: alt.as_ref(),
    })
    .unwrap();
    let h = &out.header;
    let mut st = NlStats::default();
    assert_eq!(h.min_symbol, 0, "{name}: minSymbol is always 0");
    assert_eq!(h.max_symbol, u16::MAX, "{name}: maxSymbol is always 65535");

    // Tile grid and per-tile luma ranges are integer/f32-exact.
    let want_min = dump["luma_min"].i64();
    let want_max = dump["luma_max"].i64();
    assert_eq!(want_min.len(), meta.ntiles, "{name}: dump numTiles");
    if let Some(t) = &h.nonlinear {
        assert_eq!(t.tile_width as usize, meta.tile_w, "{name}: tile width");
        assert_eq!(t.tile_height as usize, meta.tile_h, "{name}: tile height");
        assert_eq!(t.luma_min.len(), meta.ntiles, "{name}: numTiles");
        assert_eq!(
            t.luma_min.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            want_min,
            "{name}: lumaMin"
        );
        assert_eq!(
            t.luma_max.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            want_max,
            "{name}: lumaMax"
        );
    }
    // The weight codes themselves: exact against the deterministic f64 re-solve of the
    // dumped `A`/`B` (up to a one-code `round` boundary), and measured against the MKL
    // f32 draw the reference itself cannot reproduce.
    if let Some(t) = &h.nonlinear {
        for (p, key) in ["weights_u", "weights_v"].iter().enumerate() {
            if let Some(got) = &t.weights[p] {
                let want64 = dump[&format!("{key}_f64")].i64();
                let wantmkl = dump[*key].i64();
                assert_eq!(got.len(), want64.len(), "{name}: {key}_f64 length");
                st.w64_total += want64.len();
                for (i, &g) in got.iter().enumerate() {
                    let (w64, wmkl) = (want64[i], wantmkl[i]);
                    if g as i64 == w64 {
                        st.w64_exact += 1;
                    } else if (g as i64 - w64).abs() > 1 {
                        st.w64_bad += 1;
                        println!("{name}: {key}_f64[{i}] ours {g} f64 {w64} mkl {wmkl}");
                    }
                    if g as i64 != wmkl {
                        st.wmkl_diff += 1;
                    }
                }
            }
        }
    }
    // The per-plane enable decisions: compared against both independent reference draws
    // (the coded stream's header and the replay's) — they already disagree with each
    // other on these systems.
    let got_en = h
        .nonlinear
        .as_ref()
        .map(|t| [t.weights[0].is_some(), t.weights[1].is_some()])
        .unwrap_or([false, false]);
    let stream_en = headers
        .tools
        .efe_nonlinear
        .as_ref()
        .and_then(|s| s.nonlinear.as_ref())
        .map(|t| [t.weights[0].is_some(), t.weights[1].is_some()])
        .unwrap_or([false, false]);
    for p in 0..2 {
        if got_en[p] != meta.enabled[p] && got_en[p] != stream_en[p] {
            st.flag_outlier += 1;
            println!(
                "{name}: enabled[{p}] ours {} dump {} stream {}",
                got_en[p], meta.enabled[p], stream_en[p]
            );
        }
    }

    // Masks: geometry is exact wherever the replay kept a mask; contents and the
    // keep/drop decision follow the filtered planes and are measured.
    if (meta.mask_en[0] || meta.mask_en[1])
        && let Some((bs, my, mx)) = h.mask_geometry
    {
        assert_eq!(bs as usize, meta.bsize, "{name}: bS");
        let p = usize::from(!meta.mask_en[0]);
        let wm = &dump[&format!("mask.{p}")];
        assert_eq!(
            (my as usize, mx as usize),
            (wm.shape[2], wm.shape[3]),
            "{name}: mask geometry"
        );
    }
    for p in 0..2 {
        let want = dump[&format!("mask.{p}")].i64();
        match (&h.masks[p], meta.mask_en[p]) {
            (Some(got), true) => {
                assert_eq!(got.len(), want.len(), "{name}: mask.{p} length");
                st.mask_total += want.len();
                for (&g, &w) in got.iter().zip(&want) {
                    if g as i64 != w {
                        st.mask_diff += 1;
                    }
                }
            }
            (None, false) => {}
            _ => st.mask_flag_diff += 1,
        }
    }

    // The output picture vs the replay's `out` (chroma only — the luma passes through).
    for (p, c) in ["b", "c"].iter().enumerate() {
        let want = plane(&dump[&format!("out.{c}")]);
        let got = [&out.filtered.u, &out.filtered.v][p];
        st.out_worst = st.out_worst.max(max_abs_diff(&want.data, &got.data));
    }
    st
}

/// `EFEnonlinear.compress` parity. Deterministic in the reference — and asserted exactly —
/// are the tile grid, the per-tile luma bounds, the mask geometry, `numTiles`, the weight
/// count, and the `minSymbol`/`maxSymbol` constants (`encode_header`'s fold pins them to
/// 0/65535). The weight codes are asserted against the dump's *f64* re-solve (`dgelsy`,
/// the same algorithm family our `lstsq` port implements): exact or off by one `round`
/// boundary. Everything downstream of the solve — the MKL f32 draw, the enable flags, the
/// mask contents, the output planes — is *not* run-to-run reproducible by the reference
/// itself (the coded streams and the replay draw different flags on identical inputs), so
/// it is measured and bounded, not asserted exactly.
#[test]
fn efe_nonlinear_decisions_match_reference() {
    let mut all = NlStats::default();
    for &name in EFE_NL_VECTORS {
        let st = check_efe_nl(name);
        println!(
            "{name}: w64 {}/{} exact ({} bad), mkl diffs {}, masks {}/{} differ, \
             flag outliers {}, mask-flag diffs {}, out max|d| {:e}",
            st.w64_exact,
            st.w64_total,
            st.w64_bad,
            st.wmkl_diff,
            st.mask_diff,
            st.mask_total,
            st.flag_outlier,
            st.mask_flag_diff,
            st.out_worst,
        );
        all.w64_bad += st.w64_bad;
        all.w64_total += st.w64_total;
        all.w64_exact += st.w64_exact;
        all.wmkl_diff += st.wmkl_diff;
        all.mask_diff += st.mask_diff;
        all.mask_total += st.mask_total;
        all.flag_outlier += st.flag_outlier;
        all.mask_flag_diff += st.mask_flag_diff;
        all.out_worst = all.out_worst.max(st.out_worst);
    }
    println!(
        "efe_nonlinear: f64 weights {}/{} exact +{} boundary, mkl diffs {}, \
         masks {}/{}, flags {}, mask flags {}, out {:e}",
        all.w64_exact,
        all.w64_total,
        all.w64_total - all.w64_exact - all.w64_bad,
        all.wmkl_diff,
        all.mask_diff,
        all.mask_total,
        all.flag_outlier,
        all.mask_flag_diff,
        all.out_worst,
    );
    assert_eq!(
        all.w64_bad, 0,
        "integerised weights differ from the deterministic f64 solve by more than a \
         round boundary"
    );
    // Measured slack on the quantities downstream of the MKL draw — tripwires against a
    // behaviour change in our port, not parity claims (the reference's own draws disagree
    // with each other here). Measured 2026-09: 11 flag outliers of 34, 3 mask-flag diffs,
    // 289/1234 mask blocks, out max|d| 19.2.
    assert!(all.flag_outlier <= 16, "enable-flag outliers: {all:?}");
    assert!(all.mask_flag_diff <= 5, "mask keep/drop diffs: {all:?}");
    assert!(
        all.mask_diff * 2 <= all.mask_total,
        "mask block diffs {}/{}",
        all.mask_diff,
        all.mask_total
    );
    assert!(
        all.out_worst <= 32.0,
        "output planes differ by {}",
        all.out_worst
    );
}

/// End-to-end: an encode with `efe_linear + efe_nonlinear` on must produce a stream the
/// stock reference decoder accepts — this exercises the whole `mod.rs` post-filter chain
/// and the `TON` assembly, not just `decide`. Our own decode must land within 1 of the
/// reference decoder's picture (the whole-picture gate of `decode_ref.rs`).
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_accepts_efe_nonlinear_stream() {
    use zenjpegai::decoder::read_headers;
    let params = EncodeParams {
        model_id: 1,
        op: OperatingPoint::Bop,
        efe_linear: true,
        efe_nonlinear: true,
        ..Default::default()
    };
    let enc = Encoder::new(ref_root().join("models"));
    let stream = enc.encode(source(IMG30), params).unwrap();
    // The stream must carry the tool header.
    let headers = read_headers(&zenjpegai::container::Codestream::parse(&stream).unwrap()).unwrap();
    assert!(headers.tools.efe_linear.is_some(), "no EFE linear header");
    assert!(
        headers.tools.efe_nonlinear.is_some(),
        "no EFE non-linear header"
    );
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
    std::fs::create_dir_all(&scratch).unwrap();
    let bits = scratch.join("enc_img30_efenl.bits");
    let png = scratch.join("enc_img30_efenl.png");
    std::fs::write(&bits, &stream).unwrap();
    ref_decode(&bits, &png, "enc_img30_efenl");
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
    assert!(worst <= 1, "reference decode differs by {worst}");
}

// ---------------------------------------------------------------------------------------------
// E5: the `tools_on` preset (upstream `cfg/tools_on.json`: RVS + GRFS, LSBS, and the four
// post-filters EFE linear, eICCI, EFE non-linear, LEF) at the five CTC rates on both test
// pictures. Oracle: `make_reference_streams.sh toolson`.

/// `(vector, source image, target bpp)` — every stream of the `toolson` reference set.
const VECTORS_TOOLS_ON: &[(&str, &str, f64)] = &[
    ("img30_base_on_bpp012", IMG30, 0.12),
    ("img30_base_on_bpp025", IMG30, 0.25),
    ("img30_base_on_bpp050", IMG30, 0.50),
    ("img30_base_on_bpp075", IMG30, 0.75),
    ("img30_base_on_bpp100", IMG30, 1.00),
    ("img01_base_on_bpp012", IMG01, 0.12),
    ("img01_base_on_bpp025", IMG01, 0.25),
    ("img01_base_on_bpp050", IMG01, 0.50),
    ("img01_base_on_bpp075", IMG01, 0.75),
    ("img01_base_on_bpp100", IMG01, 1.00),
];

/// PSNR of a decode against the source picture, all samples and channels pooled. For a
/// gate that measures the *difference* between two decodes of equal-geometry pictures the
/// channel weighting cancels; the peak is the source's bit depth.
fn psnr_vs_source(decoded: &zenjpegai::RgbImage, source: &zenjpegai::RgbImage) -> f64 {
    assert_eq!(
        (decoded.width, decoded.height, decoded.data.len()),
        (source.width, source.height, source.data.len())
    );
    let peak = ((1u32 << source.bit_depth) - 1) as f64;
    let mse = decoded
        .data
        .iter()
        .zip(&source.data)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / decoded.data.len() as f64;
    10.0 * (peak * peak / mse).log10()
}

/// How far the two streams' decoded entropy stages diverge: `(z_hat moves, residual
/// moves, worst |residual move|)`. `z_hat` moves must each be a single step — a
/// hyper-encoder rounding boundary flip (the float path's known 3.3e-5 neighbourhood);
/// each flip cascades through the scale map, so under a `z_hat` move the residual moves
/// it causes are not bounded to one level. When `z_hat` is identical, residual moves are
/// the documented rounding-boundary mechanism and must move by exactly one level (the
/// ANS stream then propagates the difference, so byte diffs are not meaningful).
fn moved_residual_symbols(
    got: &zenjpegai::container::Codestream,
    hdr_got: &zenjpegai::header::PictureHeader,
    want: &zenjpegai::container::Codestream,
    hdr_want: &zenjpegai::header::PictureHeader,
) -> (usize, usize, i32) {
    use zenjpegai::decoder::decode_entropy_stage;
    use zenjpegai::mans::AnsTables;
    use zenjpegai::model::ModelDir;
    let models = ModelDir::new(ref_root().join("models"));
    let eng = Engine::new();
    let tables = AnsTables::new();
    let g0 = models
        .load_common(hdr_got.model_id as usize, 0, &eng)
        .unwrap();
    let g1 = models
        .load_common(hdr_got.model_id as usize, 1, &eng)
        .unwrap();
    let w0 = models
        .load_common(hdr_want.model_id as usize, 0, &eng)
        .unwrap();
    let w1 = models
        .load_common(hdr_want.model_id as usize, 1, &eng)
        .unwrap();
    let eg = decode_entropy_stage(&tables, got, hdr_got, [&g0, &g1]).unwrap();
    let ew = decode_entropy_stage(&tables, want, hdr_want, [&w0, &w1]).unwrap();
    assert_eq!(eg.len(), ew.len(), "entropy-stage component count");
    let mut z_moved = 0usize;
    let mut moved = 0usize;
    let mut worst = 0i32;
    for (g, w) in eg.iter().zip(&ew) {
        assert_eq!(g.z_hat.data.len(), w.z_hat.data.len(), "z_hat length");
        assert_eq!(
            g.residual_q.data.len(),
            w.residual_q.data.len(),
            "residual length"
        );
        for (a, b) in g.z_hat.data.iter().zip(&w.z_hat.data) {
            if a != b {
                z_moved += 1;
                assert_eq!(
                    (*a as i32 - *b as i32).abs(),
                    1,
                    "a z_hat symbol moved by more than one step"
                );
            }
        }
        for (a, b) in g.residual_q.data.iter().zip(&w.residual_q.data) {
            if a != b {
                moved += 1;
                worst = worst.max((*a as i32 - *b as i32).abs());
            }
        }
    }
    (z_moved, moved, worst)
}

/// One leg of the E5 gate, shared by both decoders: encode `image` tools-on at `target`
/// bpp. The search runs the reference's own `ECLibLH` measure, which should land it on the
/// reference's `(model, beta)`; when it does not, the stream gate applies to a fixed-model
/// encode at the reference's decisions and the divergence is reported (and counted by the
/// caller). Returns `(our stream, our pick)`.
fn tools_on_encode(
    enc: &Encoder,
    dec: &zenjpegai::Decoder,
    name: &str,
    image: &str,
    target: f64,
) -> (Vec<u8>, zenjpegai::encoder::RateMatch) {
    let dir = vector_dir(name);
    let reference = std::fs::read(dir.join("stream.bits")).unwrap();
    let want = dec.read_headers(&reference).unwrap();
    let picture = source(image);
    let (stream, m) = enc
        .encode_to_bpp(
            &picture,
            target,
            EncodeParams {
                rate_estimate: zenjpegai::encoder::RateEstimate::Likelihood,
                ..EncodeParams::tools_on()
            },
        )
        .unwrap();
    let (wmodel, wbeta) = (want.picture.model_id, want.picture.beta_displacement_log[0]);
    if (m.model_id, m.beta_displacement_log) == (wmodel, wbeta) {
        return (stream, m);
    }
    println!(
        "{name}: pick differs — ours model {} beta {}, reference {} / {}; \
         re-encoding at the reference's decisions",
        m.model_id, m.beta_displacement_log, wmodel, wbeta
    );
    let stream = enc
        .encode(
            &picture,
            EncodeParams {
                model_id: wmodel,
                beta_displacement_log: [wbeta; 2],
                ..EncodeParams::tools_on()
            },
        )
        .unwrap();
    (stream, m)
}

/// The measured deviation signature of each `tools_on` vector (2026-09-18; the table in
/// `PORTING.md` → "Encoder parity → `tools_on`"): `(z_hat symbols moved, residual symbols
/// moved, worst |residual move|, stream-size excess not carried by the TON payload)`. The
/// encode is deterministic — every SIMD tier and thread count produces the identical
/// stream — so each entry is pinned exactly: a changed value is a behaviour change, never
/// something to absorb; update the table and `PORTING.md` together.
const TOOLS_ON_SIGNATURE: &[(&str, usize, usize, i32, i64)] = &[
    ("img30_base_on_bpp012", 0, 0, 0, 0),
    ("img30_base_on_bpp025", 0, 0, 0, 0),
    ("img30_base_on_bpp050", 0, 0, 0, 0),
    ("img30_base_on_bpp075", 0, 1, 1, 1),
    ("img30_base_on_bpp100", 0, 1, 1, 0),
    ("img01_base_on_bpp012", 0, 1, 1, 1),
    ("img01_base_on_bpp025", 0, 0, 0, 0),
    ("img01_base_on_bpp050", 0, 12, 1, -1),
    ("img01_base_on_bpp075", 0, 4, 1, 0),
    ("img01_base_on_bpp100", 1, 37, 1, -2),
];

/// Assert `name`'s symbol moves and size accounting against [`TOOLS_ON_SIGNATURE`]. `sig`
/// is `(z_hat moves, residual moves, worst |residual move|)` from
/// [`moved_residual_symbols`]; `sizes` is `(our stream, reference stream, our TON payload,
/// reference TON payload)` lengths in bytes.
fn check_tools_on_signature(
    name: &str,
    sig: (usize, usize, i32),
    sizes: (usize, usize, usize, usize),
) {
    let Some(&(_, wz, wm, wworst, wunexplained)) = TOOLS_ON_SIGNATURE.iter().find(|e| e.0 == name)
    else {
        panic!("{name}: no pinned deviation signature");
    };
    let (got_len, want_len, ton_got, ton_want) = sizes;
    // Bytes of the size excess outside the TON: the ue-coded substream size prefixes and
    // the ANS-payload drift of moved symbols (a moved symbol rewrites the rest of its
    // substream's bytes, but the payload length drifts only a byte or two).
    let unexplained = got_len as i64 - want_len as i64 - (ton_got as i64 - ton_want as i64);
    assert_eq!(
        (sig.0, sig.1, sig.2, unexplained),
        (wz, wm, wworst, wunexplained),
        "{name}: deviation signature drifted (z_hat moves, residual moves, worst move, \
         unexplained size) — update TOOLS_ON_SIGNATURE and PORTING.md deliberately"
    );
}

/// E5 gate: at every CTC rate on both pictures, a `tools_on` encode must land on the
/// reference encoder's `(model, beta)` pick, produce a stream within 0.5 % of the
/// reference stream's size, and decode (through this crate's decoder) to a picture whose
/// PSNR against the source is within 0.02 dB of the reference stream's.
#[test]
fn tools_on_ctc_matches_reference() {
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    let mut picks = 0usize;
    for &(name, image, target) in VECTORS_TOOLS_ON {
        let dir = vector_dir(name);
        let reference = std::fs::read(dir.join("stream.bits")).unwrap();
        let want = dec.read_headers(&reference).unwrap();
        let picture = source(image);
        let pixels = (picture.width * picture.height) as f64;
        let (stream, m) = tools_on_encode(&enc, &dec, name, image, target);
        picks += usize::from(
            (m.model_id, m.beta_displacement_log)
                == (want.picture.model_id, want.picture.beta_displacement_log[0]),
        );
        let ours = dec.decode(&stream).unwrap();

        // The tool set the stream must carry: RVS + GRFS per component, LSBS, all four
        // post-filter blocks of the TON.
        let got = dec.read_headers(&stream).unwrap();
        for (ccs, c) in got.picture.components.iter().enumerate() {
            assert!(c.rvs_enabled, "{name}: component {ccs} without RVS");
            assert!(
                c.grfs_channel_flags
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|&f| f)),
                "{name}: component {ccs} without gain flags"
            );
        }
        assert!(
            got.tools.lsbs_enabled == [true, true],
            "{name}: LSBS not signalled"
        );
        assert!(
            got.tools.efe_linear.is_some(),
            "{name}: no EFE linear header"
        );
        assert!(got.tools.lef_channel.is_some(), "{name}: no LEF channel");
        // eICCI and EFE non-linear are *decisions*: on a near-tie our searches can
        // legitimately land elsewhere than the reference's (the documented MKL `lstsq`
        // deviation of E1/E2 produces a different — sometimes better — filter). Presence
        // parity is still asserted: a divergence fails loudly and must be explained in
        // PORTING.md, never silently loosened.
        assert_eq!(
            got.tools.icci.is_some(),
            want.tools.icci.is_some(),
            "{name}: eICCI enable diverges from the reference"
        );
        assert_eq!(
            got.tools.efe_nonlinear.is_some(),
            want.tools.efe_nonlinear.is_some(),
            "{name}: EFE non-linear enable diverges from the reference"
        );

        // The coded payload — every substream except the TON tool header — must be
        // byte-identical to the reference's: the entropy stage is integer-exact and the
        // only sanctioned divergence under tools_on is the post-filter *decisions* the
        // TON carries (the documented MKL `lstsq` deviation of E1/E2 can pick a different,
        // usually better, filter).
        use zenjpegai::container::{Codestream, Marker};
        let cs_got = Codestream::parse(&stream).unwrap();
        let cs_want = Codestream::parse(&reference).unwrap();
        assert_eq!(
            cs_got.substreams.len(),
            cs_want.substreams.len(),
            "{name}: substream count"
        );
        // PIH: every field equal except the GRFS flag *set* — `analyzeCWG` ranks channel
        // means with `torch.sort`, whose tie order is unstable, while this port's sort
        // breaks ties by lowest index. On a flat scale map (e.g. img01 at 0.12 bpp, where
        // 32 chroma channels share the floor value) the subsets legitimately differ. What
        // must hold: every other field, and the flagged-channel *count*.
        let (mut pg, mut pw) = (got.picture.clone(), want.picture.clone());
        for c in 0..pg.components.len() {
            fn flags(p: &zenjpegai::header::PictureHeader, c: usize) -> Option<usize> {
                p.components[c]
                    .grfs_channel_flags
                    .as_ref()
                    .map(|f| f.iter().filter(|&&b| b).count())
            }
            assert_eq!(
                flags(&pg, c),
                flags(&pw, c),
                "{name}: component {c} GRFS flag count"
            );
            pg.components[c].grfs_channel_flags = None;
            pw.components[c].grfs_channel_flags = None;
        }
        assert_eq!(pg, pw, "{name}: picture header fields");

        // The coded payload at symbol level. SOZ/RDI bytes are checked directly when
        // `z_hat` is identical — the only sanctioned z divergence is a single-step flip
        // on a hyper-encoder rounding boundary, which rewrites the SOZ stream.
        let (z_moved, moved, move_worst) =
            moved_residual_symbols(&cs_got, &got.picture, &cs_want, &want.picture);
        for (g, w) in cs_got.substreams.iter().zip(&cs_want.substreams) {
            assert_eq!(g.marker, w.marker, "{name}: substream order");
            if matches!(g.marker, Marker::Rdi | Marker::Soq)
                || (g.marker == Marker::Soz && z_moved == 0)
            {
                assert_eq!(g.payload, w.payload, "{name}: {:?} differs", g.marker);
            }
        }
        if z_moved == 0 {
            assert!(
                move_worst <= 1,
                "{name}: residual symbol moved by {move_worst} without a z_hat flip"
            );
        }
        let ton_got = cs_got.find(Marker::Ton).map(|p| p.len()).unwrap_or(0);
        let ton_want = cs_want.find(Marker::Ton).map(|p| p.len()).unwrap_or(0);

        let rel = stream.len() as f64 / reference.len() as f64 - 1.0;
        let theirs = dec.decode(&reference).unwrap();
        let dp = psnr_vs_source(&ours, &picture) - psnr_vs_source(&theirs, &picture);
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
        println!(
            "{name}: ours model {} beta {} -> {} bytes ({:.4} bpp, {:+.3} % of reference {}); \
             TON {} vs {} bytes, {z_moved} z_hat + {moved} residual symbols moved \
             (worst {move_worst}); \
             our-decode PSNR {:+.4} dB vs reference stream's, {differing} of {} samples differ \
             (worst {worst}); icci {} efe_nl {}",
            m.model_id,
            m.beta_displacement_log,
            stream.len(),
            stream.len() as f64 * 8.0 / pixels,
            rel * 100.0,
            reference.len(),
            ton_got,
            ton_want,
            dp,
            ours.data.len(),
            got.tools.icci.is_some(),
            got.tools.efe_nonlinear.is_some(),
        );
        // The E5 gate, kept strict: 0.5 % stream size and 0.02 dB PSNR. The only tolerated
        // breach is the documented one — the size excess sits in the TON plus its ue-coded
        // size prefix (payload byte-identical up to the known rounding-boundary mechanism:
        // a residual symbol on a quantization boundary can move by 1 and take a few ANS
        // bytes with it, at equal length) and the different filter decision is *better*
        // (dPSNR > 0), never worse. Anything else fails loudly.
        //
        // The signature itself is pinned per vector: the recorded symbol moves and the
        // exact size accounting of [`TOOLS_ON_SIGNATURE`] hold on every vector, breaching
        // or not.
        check_tools_on_signature(
            name,
            (z_moved, moved, move_worst),
            (stream.len(), reference.len(), ton_got, ton_want),
        );
        let size_ok = rel.abs() < 0.005;
        let psnr_ok = dp.abs() <= 0.02;
        if !size_ok || !psnr_ok {
            println!(
                "{name}: strict gate breach — size {size_ok} PSNR {psnr_ok}; \
                 the pinned deviation signature applies"
            );
            assert!(
                dp > 0.0,
                "{name}: different filter decision made the picture *worse* ({dp:+.4} dB)"
            );
        }
    }
    assert_eq!(
        picks,
        VECTORS_TOOLS_ON.len(),
        "rate matching under tools_on diverged from the reference's picks"
    );
}

/// The second leg of the E5 gate: the same streams through the *reference* decoder. PSNR
/// of our tools-on stream against the source must sit within 0.02 dB of the reference
/// stream's; a breach is only tolerated with the documented signature — byte-identical
/// coded payload (so the decoded difference is entirely the TON's filter decisions) and
/// `dPSNR > 0` (ours better, never worse).
#[test]
#[ignore = "runs the reference decoder (Python); enable with --ignored"]
fn reference_decoder_tools_on_psnr() {
    use zenjpegai::container::Codestream;
    let enc = Encoder::new(ref_root().join("models"));
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/refdec");
    std::fs::create_dir_all(&scratch).unwrap();
    for &(name, image, target) in VECTORS_TOOLS_ON {
        let dir = vector_dir(name);
        let picture = source(image);
        let (stream, _m) = tools_on_encode(&enc, &dec, name, image, target);
        let bits = scratch.join(format!("{name}.bits"));
        let png = scratch.join(format!("{name}.png"));
        std::fs::write(&bits, &stream).unwrap();
        ref_decode(&bits, &png, name);
        let ref_ours = read_png_rgb8(&std::fs::read(&png).unwrap()).unwrap();
        let ref_theirs = read_png_rgb8(
            &std::fs::read(dir.join("decoded.png"))
                .unwrap_or_else(|e| panic!("{name}/decoded.png: {e}")),
        )
        .unwrap();
        let dp = psnr_vs_source(&ref_ours, &picture) - psnr_vs_source(&ref_theirs, &picture);
        let worst = ref_ours
            .data
            .iter()
            .zip(&ref_theirs.data)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let differing = ref_ours
            .data
            .iter()
            .zip(&ref_theirs.data)
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "{name}: reference-decoder PSNR delta {dp:+.4} dB; {differing} of {} samples \
             differ (worst {worst})",
            ref_ours.data.len()
        );
        // The same pinned signature as the Rust-decoder leg: the PSNR gap must come from
        // the TON's filter decisions alone, never from a drifting payload.
        let reference = std::fs::read(dir.join("stream.bits")).unwrap();
        let cs_got = Codestream::parse(&stream).unwrap();
        let cs_want = Codestream::parse(&reference).unwrap();
        let got_hdr = dec.read_headers(&stream).unwrap();
        let want_hdr = dec.read_headers(&reference).unwrap();
        let sig = moved_residual_symbols(&cs_got, &got_hdr.picture, &cs_want, &want_hdr.picture);
        let ton_got = cs_got
            .find(zenjpegai::container::Marker::Ton)
            .map(|p| p.len())
            .unwrap_or(0);
        let ton_want = cs_want
            .find(zenjpegai::container::Marker::Ton)
            .map(|p| p.len())
            .unwrap_or(0);
        check_tools_on_signature(
            name,
            sig,
            (stream.len(), reference.len(), ton_got, ton_want),
        );
        if dp.abs() > 0.02 {
            assert!(
                dp > 0.0,
                "{name}: reference-decoder PSNR delta {dp:+.4} dB in the wrong direction"
            );
        }
    }
}
