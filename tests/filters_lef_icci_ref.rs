//! LEF and eICCI post-filters against the reference's own planes.
//!
//! Oracle: `<vector>/filters_lef_icci/` written by `scripts/ref_vectors/dump_filters_lef_icci.py`
//! (the YUV picture entering and leaving each filter inside the reference decoder). Each filter
//! is fed the reference's *input* planes, so the measured error is the filter's own and the
//! filters are tested independently of each other and of the EFE pair.
//!
//! Bounds are a few times the measured worst case (`PORTING.md`); do not loosen them.
#![cfg(feature = "reference-tests")]

mod common;
use common::{RefTensor, load_dump, ref_root, vector_dir};
use std::collections::HashMap;
use zenjpegai::container::Codestream;
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::Planes;
use zenjpegai::filters::{icci, lef};
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

fn plane(t: &RefTensor) -> Tensor<f32> {
    assert_eq!(t.shape.len(), 4);
    Tensor::from_vec(t.shape[1], t.shape[2], t.shape[3], t.f32()).unwrap()
}

fn planes(dump: &HashMap<String, RefTensor>, prefix: &str) -> Planes {
    Planes {
        y: plane(&dump[&format!("{prefix}.a")]),
        u: plane(&dump[&format!("{prefix}.b")]),
        v: plane(&dump[&format!("{prefix}.c")]),
    }
}

fn filter_dump(name: &str) -> (zenjpegai::decoder::Headers, HashMap<String, RefTensor>) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let headers = read_headers(&Codestream::parse(&stream).unwrap()).unwrap();
    let sub = dir.join("filters_lef_icci");
    assert!(
        sub.join("manifest.txt").is_file(),
        "{} is missing: run scripts/ref_vectors/dump_filters_lef_icci.py",
        sub.display()
    );
    (headers, load_dump(&sub))
}

/// Every tier, threaded or not; the large picture only on the default engine (tier identity is
/// established on the small ones, and the scalar tier takes a minute on 3 MP).
fn engines(name: &str) -> Vec<Engine> {
    if name.starts_with("img01") {
        return vec![Engine::new()];
    }
    let mut v = Vec::new();
    for tier in Tier::available() {
        v.push(Engine::with(tier, true));
        v.push(Engine::with(tier, false));
    }
    v
}

fn check_lef(name: &str) {
    let (headers, dump) = filter_dump(name);
    let channel = headers.tools.lef_channel.expect("stream without LEF") as usize;
    let s = &dump["lef.scale_log"];
    let scale_log = Tensor::from_vec(s.shape[1], s.shape[2], s.shape[3], s.i32()).unwrap();
    let want = planes(&dump, "lef.out");
    let mut first: Option<Vec<f32>> = None;
    for eng in engines(name) {
        let mut got = planes(&dump, "lef.in");
        lef::sharpen(
            &eng,
            &mut got.y,
            &scale_log,
            channel,
            headers.picture.model_id as usize,
            headers.picture.bit_depth,
        )
        .unwrap();
        let diff = max_abs_diff(&want.y.data, &got.y.data);
        // Measured: identical to the reference's plane on every vector (the dump is fixed data
        // and this crate's output does not depend on the CPU), so exact equality is the gate.
        assert!(diff == 0.0, "{name} LEF luma: max abs diff {diff:e}");
        match &first {
            None => {
                let changed = max_abs_diff(&dump["lef.in.a"].f32(), &got.y.data);
                println!(
                    "{name}: LEF luma max abs diff {diff:e} (filter moved luma by up to {changed:.3})"
                );
                assert!(changed > 0.5, "{name}: the LEF did nothing");
                first = Some(got.y.data);
            }
            Some(f) => assert!(
                f.iter()
                    .zip(&got.y.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: LEF output differs between engines ({eng:?})"
            ),
        }
    }
    // Chroma passes through untouched.
    assert_eq!(dump["lef.in.b"].bytes, dump["lef.out.b"].bytes);
    assert_eq!(dump["lef.in.c"].bytes, dump["lef.out.c"].bytes);
}

fn check_icci(name: &str, expect_change: bool) {
    let (headers, dump) = filter_dump(name);
    let h = headers.tools.icci.as_ref().expect("stream without eICCI");
    let models = ModelDir::new(ref_root().join("models"));
    let want = planes(&dump, "eicci.out");
    let mut first: Option<Planes> = None;
    for eng in engines(name) {
        let cache = icci::NetCache::default();
        let got = icci::filter(
            &eng,
            &headers.picture,
            h,
            headers.picture.synthesis_transforms[0],
            &models,
            &cache,
            planes(&dump, "eicci.in"),
        )
        .unwrap();
        for (what, w, g, i) in [
            ("Y", &want.y, &got.y, "eicci.in.a"),
            ("U", &want.u, &got.u, "eicci.in.b"),
            ("V", &want.v, &got.v, "eicci.in.c"),
        ] {
            let diff = max_abs_diff(&w.data, &g.data);
            assert!(diff < 5e-4, "{name} eICCI {what}: max abs diff {diff:e}");
            if first.is_none() {
                let changed = max_abs_diff(&dump[i].f32(), &g.data);
                println!(
                    "{name}: eICCI {what} max abs diff {diff:e} (filter moved it by up to {changed:.3})"
                );
            }
        }
        match &first {
            None => {
                let moved = max_abs_diff(&dump["eicci.in.b"].f32(), &got.u.data);
                assert_eq!(moved > 0.5, expect_change, "{name}: eICCI chroma change");
                first = Some(got);
            }
            Some(f) => {
                for (a, b) in [(&f.y, &got.y), (&f.u, &got.u), (&f.v, &got.v)] {
                    assert!(
                        a.data
                            .iter()
                            .zip(&b.data)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{name}: eICCI output differs between engines ({eng:?})"
                    );
                }
            }
        }
    }
}

#[test]
fn lef_img30_base_lef_bpp050() {
    check_lef("img30_base_lef_bpp050");
}

#[test]
fn lef_img30_base_on_bpp025() {
    check_lef("img30_base_on_bpp025");
}

#[test]
fn lef_img30_base_on_bpp100() {
    check_lef("img30_base_on_bpp100");
}

#[test]
fn icci_img30_base_eicci_bpp050() {
    check_icci("img30_base_eicci_bpp050", true);
}

#[test]
fn icci_img30_base_on_bpp025() {
    check_icci("img30_base_on_bpp025", true);
}

/// All three planes unfiltered: the filter still clamps the picture to its range.
#[test]
fn icci_img30_base_on_bpp100() {
    check_icci("img30_base_on_bpp100", false);
}

/// 2096x1400 with the filter's own tiling on: six overlapping tiles, last column widened by
/// `_adjust_boundary_tiles`, all three planes filtered in every tile.
#[test]
fn icci_img01_base_eiccitiles_lef_bpp050() {
    check_icci("img01_base_eiccitiles_lef_bpp050", true);
}

#[test]
fn lef_img01_base_eiccitiles_lef_bpp050() {
    check_lef("img01_base_eiccitiles_lef_bpp050");
}

/// The public `Decoder` on the filter streams (LEF, eICCI, tiled eICCI + LEF, and upstream's
/// tools_on with all four filters): 8-bit output against the reference decoder's, same gate as
/// `tests/decode_ref.rs`. One decoder for all streams, so the eICCI network cache is exercised,
/// and every stream is decoded twice (second decode must be identical).
#[test]
fn decoder_api_on_filter_streams() {
    use zenjpegai::decoder::output::quantize_plane;
    let dec = zenjpegai::Decoder::new(ref_root().join("models"));
    for name in [
        "img30_base_lef_bpp050",
        "img30_base_eicci_bpp050",
        "img30_base_on_bpp025",
        "img30_base_on_bpp100",
        "img01_base_eiccitiles_lef_bpp050",
    ] {
        let dir = vector_dir(name);
        let stream = std::fs::read(dir.join("stream.bits")).unwrap();
        let dump = load_dump(&dir);
        let ours = dec.decode(&stream).unwrap();
        assert_eq!(ours, dec.decode(&stream).unwrap(), "{name}: second decode");
        let [r, g, b] = ["out.a", "out.b", "out.c"].map(|k| quantize_plane(&dump[k].f32(), 8));
        let theirs: Vec<u16> = (0..r.len()).flat_map(|i| [r[i], g[i], b[i]]).collect();
        assert_eq!(ours.data.len(), theirs.len(), "{name}");
        let differing = ours
            .data
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a != b)
            .count();
        let worst = ours
            .data
            .iter()
            .zip(&theirs)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        println!(
            "{name}: Decoder output, {differing} of {} samples differ, worst {worst}",
            theirs.len()
        );
        assert!(worst <= 1, "{name}: an 8-bit sample differs by {worst}");
        assert!(
            differing * 5000 < theirs.len(),
            "{name}: {differing} samples differ"
        );
    }
}
