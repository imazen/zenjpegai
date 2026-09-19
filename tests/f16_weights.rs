//! f16 model-weight storage (`ZJB2` bundles, `pack-models --f16`).
//!
//! What this pins down:
//! * every tensor an f16 bundle stores is either a byte-identical copy of the `.pth` tensor
//!   (integer dtypes, `vr_vec.c`) or its round-to-nearest-even f16 image — nothing else;
//! * the f16 decoder gets the SAME gate the f32 one has on the real reference vectors:
//!   8/10-bit output within 1 LSB of the reference dump in fewer than 1/5000 samples;
//! * every SIMD tier and thread mode produces bit-identical output with f16 weights, the
//!   `tiers_and_threads` property of the f32 path;
//! * the f32-vs-f16 output delta is *measured* on adversarial synthetic sources (this file
//!   prints the TSV rows `benchmarks/f16_weights_<date>.md` and `wasm_size_<date>.md` cite).
#![cfg(feature = "reference-tests")]

mod common;
#[path = "../scripts/ref_vectors/synthetic_patterns.rs"]
mod synthetic;

use common::{
    StagedDecode, decode_staged, decode_staged_with, icci_header_from_dump, load_dump,
    load_fixed_decoder_dump, ref_root, vector_dir,
};
use std::collections::BTreeMap;
use zenjpegai::container::{Codestream, Marker};
use zenjpegai::decoder::output::{Picture, finish, quantize_plane};
use zenjpegai::decoder::{Headers, read_headers};
use zenjpegai::encoder::{EncodeParams, Encoder};
use zenjpegai::header::{OperatingPoint, PictureHeader, ToolHeader};
use zenjpegai::model::{self, ModelDir, ModelSource};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::{PackedBundle, Recorder};
use zenjpegai::weights::{Checkpoint, DType, f16};

fn pack(ids: &[usize], ops: &[OperatingPoint], common_part: bool, synthesis: bool) -> Vec<u8> {
    let eng = Engine::with(Tier::Scalar, false);
    let rec = Recorder::new(ModelDir::new(ref_root().join("models")));
    for &id in ids {
        for ccs in 0..2 {
            if common_part {
                model::load_common(&rec, id, ccs, &eng).unwrap();
            }
        }
        for &op in ops.iter().filter(|_| synthesis) {
            model::load_synthesis_primary(&rec, id, op, &eng).unwrap();
            model::load_synthesis_secondary(&rec, id, op, &eng).unwrap();
        }
    }
    rec.pack_f16("notice text").unwrap()
}

/// The f32 (ZJB1) variant of [`pack`].
fn pack_lossless(
    ids: &[usize],
    ops: &[OperatingPoint],
    common_part: bool,
    synthesis: bool,
) -> Vec<u8> {
    let eng = Engine::with(Tier::Scalar, false);
    let rec = Recorder::new(ModelDir::new(ref_root().join("models")));
    for &id in ids {
        for ccs in 0..2 {
            if common_part {
                model::load_common(&rec, id, ccs, &eng).unwrap();
            }
        }
        for &op in ops.iter().filter(|_| synthesis) {
            model::load_synthesis_primary(&rec, id, op, &eng).unwrap();
            model::load_synthesis_secondary(&rec, id, op, &eng).unwrap();
        }
    }
    rec.pack("notice text").unwrap()
}

/// Which bundle parts are halved. `C16S32` keeps synthesis lossless; `C32S16` keeps the
/// entropy-stage-adjacent common part lossless.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Scheme {
    F16,
    C16S32,
    C32S16,
}

/// A bundle of one (model, operating point) under `scheme`, merging a common part and a
/// synthesis part packed separately so each can carry its own storage format.
fn pack_scheme(id: usize, op: OperatingPoint, scheme: Scheme) -> PackedBundle {
    let common = if scheme == Scheme::C32S16 {
        pack_lossless(&[id], &[], true, false)
    } else {
        pack(&[id], &[], true, false)
    };
    let synthesis = if scheme == Scheme::C16S32 {
        pack_lossless(&[id], &[op], false, true)
    } else {
        pack(&[id], &[op], false, true)
    };
    let mut b = PackedBundle::parse(common).unwrap();
    b.add(synthesis).unwrap();
    b
}

/// f16 bundle for the (model, operating point) a stream's headers ask for, built once per
/// (pair, scheme) and shared by every test that decodes several streams.
struct BundleCache(BTreeMap<(usize, u8, Scheme), PackedBundle>);

impl BundleCache {
    fn new() -> Self {
        Self(BTreeMap::new())
    }
    fn get(&mut self, id: usize, op: OperatingPoint, scheme: Scheme) -> &PackedBundle {
        self.0
            .entry((id, op as u8, scheme))
            .or_insert_with(|| pack_scheme(id, op, scheme))
    }
}

/// f16 model source for one stream: the packed bundle plus, when the stream's tool header
/// enables eICCI, that bank's networks f16-packed the same way (`write_zjm_f16`). The
/// post-filter checkpoints are not in `ZJB` bundles today (their loaders bypass
/// `model::with_checkpoint` — see `weights::packed`'s note), so a shipped f16 bundle would
/// cover them only once that gap closes; the measurement still wants them halved.
struct F16Source<'a> {
    bundle: &'a PackedBundle,
    /// f16-packed eICCI network checkpoints, by their `models/` path.
    extra: zenjpegai::model::ModelBundle,
}

impl ModelSource for F16Source<'_> {
    fn read(&self, rel: &str) -> Result<std::borrow::Cow<'_, [u8]>, zenjpegai::Error> {
        self.bundle.read(rel).or_else(|_| self.extra.read(rel))
    }
}

/// The eICCI bank of the stream's operating point, f16-packed under its models paths. The
/// header's per-tile selections are remapped through the op's short lists and the model id
/// (`filters::icci`'s `selection`), so the whole bank is the safe set to carry.
fn f16_icci_nets(
    tools: &ToolHeader,
    op: OperatingPoint,
    pth: &ModelDir,
) -> zenjpegai::model::ModelBundle {
    let mut extra = zenjpegai::model::ModelBundle::new();
    if tools.icci.is_none() {
        return extra;
    }
    for index in 0..10 {
        let rel = model::icci::checkpoint_path(op, index).unwrap();
        let file = pth.read(&rel).unwrap();
        let ck = Checkpoint::parse(&file).unwrap();
        let names: Vec<String> = ck.tensor_names().map(str::to_owned).collect();
        let zjm = zenjpegai::weights::packed::write_zjm_f16(&ck, &names).unwrap();
        extra.insert(rel, zjm);
    }
    extra
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Quantised samples of a staged decode and of the dump's `out.*` planes, laid out alike
/// (RGB interleaved or the YUV planes concatenated).
fn samples_pair(
    hdr: &PictureHeader,
    filtered: &zenjpegai::decoder::reconstruct::Planes,
    dump: &std::collections::HashMap<String, common::RefTensor>,
) -> (Vec<u16>, Vec<u16>) {
    let depth = hdr.bit_depth;
    let theirs: Vec<Vec<u16>> = ["out.a", "out.b", "out.c"]
        .iter()
        .map(|k| quantize_plane(&dump[*k].f32(), depth))
        .collect();
    match finish(hdr, filtered).unwrap() {
        Picture::Rgb(img) => (
            img.data,
            theirs[0]
                .iter()
                .zip(&theirs[1])
                .zip(&theirs[2])
                .flat_map(|((&r, &g), &b)| [r, g, b])
                .collect(),
        ),
        Picture::Yuv(img) => ([img.y, img.u, img.v].concat(), theirs.concat()),
    }
}

/// The decode_ref vectors (the macro list there) plus the forced-4:2:0 eICCI stream, whose
/// headers need the dump-rebuilt tool header.
fn vector_names() -> Vec<&'static str> {
    vec![
        "img30_base_off_bpp012",
        "img30_base_off_bpp050",
        "img30_base_off_bpp100",
        "img30_simple_off_bpp050",
        "img30yuv420_base_off_bpp050",
        "img30yuv422_base_off_bpp050",
        "img30yuv444_base_off_bpp050",
        "img30yuv420b10_base_off_bpp050",
        "img30yuv444b10_base_off_bpp050",
        "img30cropyuv420_base_off_bpp075",
        "img30_base_off_c420_bpp050",
        "img30_base_off_c422_bpp050",
        "img30_base_off_display_m1",
        "img30_base_lsbs_bpp050",
        "img30_base_rvs_bpp050",
        "img30_base_qmap_bpp050",
        "img30_base_qmap_rvs_bpp025",
        "img30_base_qmap_threads8_bpp100",
        "img30_base_rvsonly_bpp050",
        "img30_base_grfsonly_bpp075",
        "img30_base_lsbs_rvs_bpp025",
        "img30_simple_lsbs_rvs_bpp100",
        "img30_high_off_bpp050",
        "enc_img30_bop_m0_bm1069",
        "img30_base_efelin_bpp050",
        "img30_efe_f2c1_f2c2_nl",
        "img30_efe_f3c3_f3c4_nl",
        "img30_efe_f3c5_f4c7_nl",
        "img30_efe_f4c6_f1c5",
        "crop277_efe_f4c5_f3c6_nl",
        "img01_efe_f2c0_f3c0_nl",
        "img01_base_off_bpp050",
        "img01_base_off_threads8_bpp050",
        "img01_base_off_depregions_m1",
        "img01_base_off_indregions_m1",
        "img01_base_off_indregions_threads8_m2",
        "img30_base_lef_bpp050",
        "img30_base_eicci_bpp050",
        "img30yuv422_base_eicci",
        "img30yuv420_base_eicci",
        "img01_base_eiccitiles_lef_bpp050",
        "img30_base_on_bpp025",
        "img30_base_on_bpp100",
    ]
}

/// Staged decode of a reference vector with f16 weights; the forced-4:2:0 eICCI vector gets
/// its rebuilt tool header, exactly like `decode_ref`'s dedicated test.
fn decode_vector_f16(
    name: &str,
    dir: &std::path::Path,
    eng: &Engine,
    bundles: &mut BundleCache,
    scheme: Scheme,
) -> (PictureHeader, StagedDecode) {
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let cs = Codestream::parse(&stream).unwrap();
    // The PIH parses even on the forced-4:2:0 eICCI stream (only its tool header is broken),
    // so the model / operating point always come from the stream itself.
    let pic = PictureHeader::parse(cs.find(Marker::Pih).unwrap()).unwrap();
    let (id, op) = (pic.model_id as usize, pic.synthesis_transforms[0]);
    let headers = if name == "img30yuv420_base_eicci" {
        let sel = load_dump(&dir.join("filters_lef_icci"));
        Headers {
            picture: pic,
            tools: ToolHeader {
                icci: Some(icci_header_from_dump(&sel)),
                ..Default::default()
            },
            rendering: Default::default(),
            user_data: None,
        }
    } else {
        read_headers(&cs).unwrap()
    };
    let src = F16Source {
        bundle: bundles.get(id, op, scheme),
        extra: f16_icci_nets(
            &headers.tools,
            op,
            &ModelDir::new(ref_root().join("models")),
        ),
    };
    decode_staged_with(&src, &cs, eng, &headers)
}

/// Every tensor of an f16 bundle is either byte-identical to the `.pth` tensor (integer
/// dtypes, `vr_vec.c`) or its round-to-nearest-even f16 image; the entropy stage sees not
/// one changed byte.
#[test]
fn f16_bundle_preserves_the_integer_stage() {
    let bundle = PackedBundle::parse(pack(&[1], &[OperatingPoint::Bop], true, true)).unwrap();
    let pth = ModelDir::new(ref_root().join("models"));
    let mut f16_tensors = 0;
    let mut copied_tensors = 0;
    for rel in bundle.paths().map(str::to_owned).collect::<Vec<_>>() {
        let packed_bytes = bundle.read(&rel).unwrap();
        let orig_bytes = pth.read(&rel).unwrap();
        let packed = Checkpoint::parse(&packed_bytes).unwrap();
        let orig = Checkpoint::parse(&orig_bytes).unwrap();
        // Integer entries (epoch and friends) travel byte for byte.
        assert_eq!(packed.ints(), orig.ints(), "{rel}: integer entries differ");
        for name in packed.tensor_names() {
            let (dt, shape, bytes) = packed.raw(name).unwrap();
            let (dt0, shape0, bytes0) = orig.raw(name).unwrap();
            assert_eq!(shape, shape0, "{rel} {name}: shape changed");
            if dt0 == DType::F32 && name != "vr_vec.c" {
                assert_eq!(dt, DType::F16, "{rel} {name}: f32 tensor not halved");
                assert_eq!(bytes.len() * 2, bytes0.len());
                for (i, (h, w)) in bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(bytes0.as_chunks::<4>().0)
                    .enumerate()
                {
                    assert_eq!(
                        u16::from_le_bytes(*h),
                        f16::f32_to_f16(f32::from_le_bytes(*w)),
                        "{rel} {name}[{i}]: not the round-to-nearest-even image"
                    );
                }
                f16_tensors += 1;
            } else {
                assert_eq!(dt, dt0, "{rel} {name}: dtype changed");
                assert!(bytes == bytes0, "{rel} {name}: bytes changed");
                copied_tensors += 1;
            }
        }
    }
    assert!(f16_tensors > 0 && copied_tensors > 0);
    println!("f16 bundle: {f16_tensors} halved tensors, {copied_tensors} copied byte-exact");
}

/// Per-tensor rounding statistics of an f16 bundle: how many weights are not exactly
/// representable, the worst relative rounding error, and the counts falling into the f16
/// subnormal range, flushing to zero, or saturating to ±∞. TSV rows for benchmarks/.
#[test]
fn f16_weight_statistics() {
    // Common part of every model, synthesis of model 1 at all three operating points and of
    // model 3 at BOP — representative of what the browser bundles ship.
    let pth = ModelDir::new(ref_root().join("models"));
    println!(
        "file\ttensor\tn\tnot_exact\tmax_rel_err\tn_f16_subnormal\tn_flush_to_zero\tn_saturated"
    );
    let (mut t_n, mut t_ne) = (0u64, 0u64);
    let (mut t_sub, mut t_zero, mut t_sat) = (0u64, 0u64, 0u64);
    let mut t_rel = 0f64;
    let mut files_seen = std::collections::BTreeSet::new();
    for &(id, op) in &[
        (0usize, OperatingPoint::Bop),
        (1, OperatingPoint::Sop),
        (1, OperatingPoint::Bop),
        (1, OperatingPoint::Hop),
        (2, OperatingPoint::Bop),
        (3, OperatingPoint::Bop),
    ] {
        let bundle = PackedBundle::parse(pack(&[id], &[op], true, true)).unwrap();
        for rel in bundle.paths().map(str::to_owned).collect::<Vec<_>>() {
            if !files_seen.insert(rel.clone()) {
                continue;
            }
            let packed_bytes = bundle.read(&rel).unwrap();
            let orig_bytes = pth.read(&rel).unwrap();
            let packed = Checkpoint::parse(&packed_bytes).unwrap();
            let orig = Checkpoint::parse(&orig_bytes).unwrap();
            for name in packed.tensor_names() {
                let (dt, _, bytes) = packed.raw(name).unwrap();
                if dt != DType::F16 {
                    continue;
                }
                let (_, _, bytes0) = orig.raw(name).unwrap();
                let (mut not_exact, mut n_sub, mut n_zero, mut n_sat) = (0u64, 0u64, 0u64, 0u64);
                let mut max_rel = 0f64;
                for (h, w) in bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(bytes0.as_chunks::<4>().0)
                {
                    let h = u16::from_le_bytes(*h);
                    let w = f32::from_le_bytes(*w);
                    let up = f16::f16_to_f32(h);
                    if up.to_bits() != w.to_bits() {
                        not_exact += 1;
                    }
                    let mag = h & 0x7fff;
                    if mag == 0 && w != 0.0 {
                        n_zero += 1;
                    } else if mag < 0x0400 {
                        n_sub += 1;
                    } else if mag >= 0x7c00 {
                        n_sat += 1;
                    }
                    if w != 0.0 && up != 0.0 {
                        let rel = ((up - w) as f64 / w as f64).abs();
                        if rel.is_finite() {
                            max_rel = max_rel.max(rel);
                        }
                    }
                }
                let n = bytes.len() / 2;
                println!(
                    "{rel}\t{name}\t{n}\t{not_exact}\t{max_rel:.3e}\t{n_sub}\t{n_zero}\t{n_sat}"
                );
                t_n += n as u64;
                t_ne += not_exact;
                t_rel = t_rel.max(max_rel);
                t_sub += n_sub;
                t_zero += n_zero;
                t_sat += n_sat;
            }
        }
    }
    println!("TOTAL\t\t{t_n}\t{t_ne}\t{t_rel:.3e}\t{t_sub}\t{t_zero}\t{t_sat}");
}

/// The f16 decoder measured against the f32 decoder's gate on every real reference vector:
/// final samples within 1 LSB of the reference dump in fewer than 1/5000 samples. This is a
/// MEASUREMENT, not an assert — f16 storage is a candidate, and what the vectors show is
/// reported in `benchmarks/f16_weights_<date>.md` (the summary line says whether every vector
/// met the gate, and how far outside it the failures sit). Structural sanity — every stream
/// decodes, sample counts match — is asserted.
#[test]
fn f16_vs_reference_on_real_vectors() {
    let eng = Engine::new();
    let mut bundles = BundleCache::new();
    let (mut worst_abs, mut n_fail) = (0i32, 0u32);
    let mut worst_ratio = 0f64;
    println!(
        "vector\tsamples\tdiffering\tfraction\tgate_1_in_5000\tmax_abs\tmax_d_yhat\tmax_d_reca"
    );
    for name in vector_names() {
        let dir = vector_dir(name);
        // Region and quality-map streams, and the forced-4:2:0 eICCI stream, are oracle'd by
        // the reference decoder with its known defects patched (see `decode_ref`).
        let dump = if name.contains("regions")
            || name.contains("qmap")
            || name == "img30yuv420_base_eicci"
        {
            load_fixed_decoder_dump(&dir)
        } else {
            load_dump(&dir)
        };
        let (hdr, d) = decode_vector_f16(name, &dir, &eng, &mut bundles, Scheme::F16);
        // Intermediates vs the reference dump: measurements for benchmarks/f16_weights_*.
        let (dy, dp) = (
            max_abs_diff(&dump["y.y_hat"].f32(), &d.y_hat[0].data),
            max_abs_diff(&dump["rec.a"].f32(), &d.planes.y.data),
        );
        let (ours, theirs) = samples_pair(&hdr, &d.filtered, &dump);
        assert_eq!(ours.len(), theirs.len(), "{name}: sample count");
        let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        let worst = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        let pass = worst <= 1 && differing * 5000 < ours.len();
        n_fail += !pass as u32;
        worst_abs = worst_abs.max(worst);
        worst_ratio = worst_ratio.max(differing as f64 / ours.len() as f64);
        println!(
            "{name}\t{}\t{differing}\t{:.3e}\t{pass}\t{worst}\t{dy:.3e}\t{dp:.3e}",
            ours.len(),
            differing as f64 / ours.len() as f64
        );
    }
    println!(
        "f16 vs reference over {} vectors: {} outside the gate, worst sample diff {worst_abs}, worst differing fraction {worst_ratio:e}",
        vector_names().len(),
        n_fail
    );
}

/// Which bundle part drives the deviation: the same measurement as
/// [`f16_vs_reference_on_real_vectors`] with only the common part halved (`C16S32`,
/// synthesis kept lossless) or only the synthesis part (`C32S16`, common lossless), on a
/// representative subset. Says whether "f16 for some parts" could ever meet the gate.
#[test]
fn f16_hybrid_variants_on_real_vectors() {
    let eng = Engine::new();
    let mut bundles = BundleCache::new();
    println!("variant\tvector\tsamples\tdiffering\tfraction\tgate_1_in_5000\tmax_abs");
    for name in [
        "img30yuv420b10_base_off_bpp050", // worst of the full sweep
        "img30_base_off_bpp012",          // lowest rate
        "img30_base_off_bpp050",          // typical BOP
        "img30_high_off_bpp050",          // HOP synthesis (attention)
        "enc_img30_bop_m0_bm1069",        // model 0 at the rate floor
        "img01_base_off_bpp050",          // 8.8 MP, synthesis tiles
    ] {
        let dir = vector_dir(name);
        let dump = load_dump(&dir);
        for scheme in [Scheme::C32S16, Scheme::C16S32] {
            let (hdr, d) = decode_vector_f16(name, &dir, &eng, &mut bundles, scheme);
            let (ours, theirs) = samples_pair(&hdr, &d.filtered, &dump);
            let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
            let worst = ours
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap();
            let pass = worst <= 1 && differing * 5000 < ours.len();
            println!(
                "{scheme:?}\t{name}\t{}\t{differing}\t{:.3e}\t{pass}\t{worst}",
                ours.len(),
                differing as f64 / ours.len() as f64
            );
        }
    }
}

/// Every SIMD tier, threaded or not, must decode an f16 bundle to identical bits — the
/// `tiers_and_threads_agree_bit_for_bit` property with `ZJB2` storage.
#[test]
fn f16_tiers_and_threads_agree_bit_for_bit() {
    for name in ["img30_base_off_bpp050", "img30_high_off_bpp050"] {
        let dir = vector_dir(name);
        let stream = std::fs::read(dir.join("stream.bits")).unwrap();
        let cs = Codestream::parse(&stream).unwrap();
        let headers = read_headers(&cs).unwrap();
        let (id, op) = (
            headers.picture.model_id as usize,
            headers.picture.synthesis_transforms[0],
        );
        let bundle = PackedBundle::parse(pack(&[id], &[op], true, true)).unwrap();
        let mut baseline: Option<(String, StagedDecode)> = None;
        for tier in Tier::available() {
            for parallel in [false, true] {
                if matches!(tier, Tier::Scalar) && parallel {
                    continue;
                }
                let eng = Engine::with(tier, parallel);
                let (_, d) = decode_staged_with(&bundle, &cs, &eng, &headers);
                let label = format!("{eng:?}");
                match &baseline {
                    None => baseline = Some((label, d)),
                    Some((base_label, b)) => {
                        for (what, want, got) in [
                            ("psi[y]", &b.psi[0], &d.psi[0]),
                            ("y_hat[y]", &b.y_hat[0], &d.y_hat[0]),
                            ("y_hat[uv]", &b.y_hat[1], &d.y_hat[1]),
                            ("Y", &b.planes.y, &d.planes.y),
                            ("U", &b.planes.u, &d.planes.u),
                            ("V", &b.planes.v, &d.planes.v),
                        ] {
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
        println!("{name}: all tiers/threads bit-identical with f16 weights");
    }
}

/// The adversarial sweep: each synthetic source encoded by our encoder at the study's
/// (operating point, bpp) grid, then decoded with f32 and f16 weights. Prints the TSV rows:
/// differing 8-bit samples, max abs diff, PSNR between the two outputs, max abs diff on
/// `y_hat` and on the final f32 planes before quantisation.
#[test]
fn f16_vs_f32_on_synthetic_sources() {
    let enc = Encoder::new(ref_root().join("models"));
    let pth = ModelDir::new(ref_root().join("models"));
    let eng = Engine::new();
    let mut bundles = BundleCache::new();
    println!(
        "source\top\tbpp_target\tstream_bytes\tmodel\tdiffering\tsamples\tmax_abs\tpsnr_db\tmax_d_yhat\tmax_d_plane"
    );
    for img in synthetic::all() {
        let src = zenjpegai::decoder::output::RgbImage {
            width: img.width,
            height: img.height,
            bit_depth: 8,
            data: img.rgb.iter().map(|&b| b as u16).collect(),
        };
        for (op, bpp) in [
            (OperatingPoint::Bop, 0.12),
            (OperatingPoint::Bop, 0.25),
            (OperatingPoint::Bop, 0.5),
            (OperatingPoint::Bop, 1.0),
            (OperatingPoint::Sop, 0.5),
            (OperatingPoint::Hop, 0.5),
        ] {
            let (stream, _) = enc
                .encode_to_bpp(
                    &src,
                    bpp,
                    EncodeParams {
                        op,
                        ..Default::default()
                    },
                )
                .unwrap();
            let (hdr, d32) = decode_staged(&pth, &stream, &eng);
            let id = hdr.model_id as usize;
            let sop = hdr.synthesis_transforms[0];
            let bundle = bundles.get(id, sop, Scheme::F16);
            let (_, d16) = decode_staged(bundle, &stream, &eng);
            let (a32, a16) = (
                finish(&hdr, &d32.filtered).unwrap(),
                finish(&hdr, &d16.filtered).unwrap(),
            );
            let (s32, s16) = match (a32, a16) {
                (Picture::Rgb(x), Picture::Rgb(y)) => (x.data, y.data),
                (Picture::Yuv(x), Picture::Yuv(y)) => {
                    ([x.y, x.u, x.v].concat(), [y.y, y.u, y.v].concat())
                }
                _ => panic!("mixed output kinds"),
            };
            assert_eq!(s32.len(), s16.len());
            let differing = s32.iter().zip(&s16).filter(|(a, b)| a != b).count();
            let max_abs = s32
                .iter()
                .zip(&s16)
                .map(|(a, b)| (*a as i32 - *b as i32).abs())
                .max()
                .unwrap();
            let mse = s32
                .iter()
                .zip(&s16)
                .map(|(a, b)| (*a as i64 - *b as i64).pow(2) as f64)
                .sum::<f64>()
                / s32.len() as f64;
            let psnr = if mse == 0.0 {
                f64::INFINITY
            } else {
                10.0 * libm::log10(255.0 * 255.0 / mse)
            };
            let d_yhat = max_abs_diff(&d32.y_hat[0].data, &d16.y_hat[0].data)
                .max(max_abs_diff(&d32.y_hat[1].data, &d16.y_hat[1].data));
            let d_plane = max_abs_diff(&d32.filtered.y.data, &d16.filtered.y.data)
                .max(max_abs_diff(&d32.filtered.u.data, &d16.filtered.u.data))
                .max(max_abs_diff(&d32.filtered.v.data, &d16.filtered.v.data));
            println!(
                "{}\t{op:?}\t{bpp}\t{}\t{id}\t{differing}\t{}\t{max_abs}\t{psnr:.1}\t{d_yhat:.3e}\t{d_plane:.3e}",
                img.name,
                stream.len(),
                s32.len()
            );
        }
    }
}
