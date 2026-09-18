//! EFE linear / EFE non-linear post-filters against the reference's per-filter dumps
//! (`scripts/ref_vectors/dump_filters.py` → `<vector>/filters/`).
//!
//! Each filter is run in isolation on the planes the reference handed to its own filter, so the
//! bounds below measure the filter alone (float summation order versus PyTorch), not the
//! networks in front of it. Bounds are on the 0..255 scale; measured values are in `PORTING.md`.
#![cfg(feature = "reference-tests")]

mod common;
use common::{RefTensor, load_dump, ref_root, vector_dir};
use std::collections::HashMap;
use zenjpegai::container::{Codestream, Marker};
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::Planes;
use zenjpegai::filters::{FilterContext, FilterState, efe_linear, efe_nonlinear};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::tensor::Tensor;

/// Largest absolute error allowed on a filtered plane (range 0..255).
const BOUND: f32 = 2e-4;

fn plane(t: &RefTensor) -> Tensor<f32> {
    let (h, w) = (t.shape[t.shape.len() - 2], t.shape[t.shape.len() - 1]);
    Tensor::from_vec(1, h, w, t.f32()).unwrap()
}

fn planes(dump: &HashMap<String, RefTensor>, prefix: &str) -> Option<Planes> {
    Some(Planes {
        y: plane(dump.get(&format!("{prefix}.a"))?),
        u: plane(&dump[&format!("{prefix}.b")]),
        v: plane(&dump[&format!("{prefix}.c")]),
    })
}

/// Max abs difference over the top-left `want`-sized... both planes have the picture's size.
fn diff(name: &str, what: &str, want: &Tensor<f32>, got: &Tensor<f32>) -> f32 {
    assert_eq!(
        (want.h, want.w),
        (got.h, got.w),
        "{name} {what}: plane size"
    );
    want.data
        .iter()
        .zip(&got.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check_planes(
    name: &str,
    what: &str,
    want: &Planes,
    got: &Planes,
    changed_from: Option<&Planes>,
) {
    for (c, w, g) in [
        ("Y", &want.y, &got.y),
        ("U", &want.u, &got.u),
        ("V", &want.v, &got.v),
    ] {
        let d = diff(name, what, w, g);
        println!("{name} {what} {c}: max abs diff {d:e}");
        assert!(d < BOUND, "{name} {what} {c}: max abs diff {d:e}");
    }
    if let Some(before) = changed_from {
        // Guard against a vacuous pass: the reference's filter must have changed the picture.
        let moved = diff(name, what, &before.u, &want.u).max(diff(name, what, &before.v, &want.v));
        println!("{name} {what}: the reference filter moved chroma by up to {moved:e}");
        assert!(
            moved > 100.0 * BOUND,
            "{name} {what}: filter is a no-op in this vector"
        );
    }
}

fn bits_identical(a: &Planes, b: &Planes) -> bool {
    [(&a.y, &b.y), (&a.u, &b.u), (&a.v, &b.v)]
        .iter()
        .all(|(x, y)| {
            x.data
                .iter()
                .zip(&y.data)
                .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// `expect_alt`: the vector must exercise the alternative ("up-sampled") picture.
fn check(name: &str, linear: bool, nonlinear: bool, expect_alt: bool) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    // The tool header must re-serialise to the reference's bytes (4:2:0 sources omit the eICCI
    // flag, 4:4:4-coded pictures code one chroma filter instead of four).
    let ton = cs.find(Marker::Ton).expect("no tool header");
    let rewritten = headers.tools.write(&headers.picture).unwrap();
    assert_eq!(rewritten, ton, "{name}: tool header round trip");
    let fdir = dir.join("filters");
    assert!(
        fdir.join("manifest.txt").is_file(),
        "{} is missing: run scripts/ref_vectors/make_reference_streams.sh efe",
        fdir.display()
    );
    let dump = load_dump(&fdir);
    let models = ModelDir::new(ref_root().join("models"));
    let scale_log = Tensor::<i32>::zeros(1, 1, 1).unwrap();

    let mut baseline: Option<Vec<Planes>> = None;
    for tier in Tier::available() {
        for parallel in [false, true] {
            let eng = Engine::with(tier, parallel);
            let ctx = FilterContext {
                eng: &eng,
                hdr: &headers.picture,
                tools: &headers.tools,
                luma_scale_log: &scale_log,
                models: &models,
                op: headers.picture.synthesis_transforms[0],
                icci_nets: &Default::default(),
                stop: &enough::Unstoppable,
            };
            let first = baseline.is_none();
            let mut results = Vec::new();
            if linear {
                let h = headers
                    .tools
                    .efe_linear
                    .as_ref()
                    .expect("EFE linear is off");
                let input = planes(&dump, "EFElinear.in").unwrap();
                let before = planes(&dump, "EFElinear.in").unwrap();
                let state = FilterState {
                    image: input,
                    upsampled: None,
                };
                let out = efe_linear::apply(&ctx, h, state).unwrap();
                let want_up = planes(&dump, "EFElinear.up");
                assert_eq!(
                    want_up.is_some(),
                    out.upsampled.is_some(),
                    "{name}: alt picture"
                );
                assert_eq!(
                    want_up.is_some(),
                    expect_alt,
                    "{name}: alt picture expected"
                );
                if first {
                    let want = planes(&dump, "EFElinear.out").unwrap();
                    check_planes(name, "EFElinear.out", &want, &out.image, Some(&before));
                    if let (Some(w), Some(g)) = (&want_up, &out.upsampled) {
                        check_planes(name, "EFElinear.up", w, g, Some(&before));
                    }
                }
                results.push(out.image);
                results.extend(out.upsampled);
            }
            if nonlinear {
                let h = headers
                    .tools
                    .efe_nonlinear
                    .as_ref()
                    .expect("EFE non-linear is off");
                let input = planes(&dump, "EFEnonlinear.in").unwrap();
                let before = planes(&dump, "EFEnonlinear.in").unwrap();
                let alt = planes(&dump, "EFEnonlinear.alt");
                let state = FilterState {
                    image: input,
                    upsampled: alt,
                };
                let out = efe_nonlinear::apply(&ctx, h, state).unwrap();
                if first {
                    let want = planes(&dump, "EFEnonlinear.out").unwrap();
                    check_planes(name, "EFEnonlinear.out", &want, &out.image, Some(&before));
                }
                results.push(out.image);
            }
            // Every tier, threaded or not, must produce identical bits.
            match &baseline {
                None => baseline = Some(results),
                Some(b) => {
                    assert_eq!(b.len(), results.len());
                    for (x, y) in b.iter().zip(&results) {
                        assert!(
                            bits_identical(x, y),
                            "{name}: {eng:?} differs from the first tier"
                        );
                    }
                }
            }
        }
    }
}

macro_rules! vectors {
    ($($fn_name:ident => ($dir:literal, $lin:literal, $nl:literal, $alt:literal)),* $(,)?) => {
        $( #[test] fn $fn_name() { check($dir, $lin, $nl, $alt); } )*
    };
}

vectors! {
    img30_base_efelin_bpp050 => ("img30_base_efelin_bpp050", true, false, false),
    img30_base_on_bpp025 => ("img30_base_on_bpp025", true, true, true),
    img30_base_on_bpp100 => ("img30_base_on_bpp100", true, true, true),
    // EFE decisions forced in the encoder (`force_efe_encode.py`); names give filter length and
    // region split per chroma plane. 4:4:4 source coded 4:4:4:
    img30_efe_f2c1_f2c2_nl => ("img30_efe_f2c1_f2c2_nl", true, true, true),
    img30_efe_f3c3_f3c4_nl => ("img30_efe_f3c3_f3c4_nl", true, true, true),
    img30_efe_f3c5_f4c7_nl => ("img30_efe_f3c5_f4c7_nl", true, true, true),
    img30_efe_f4c6_f1c5 => ("img30_efe_f4c6_f1c5", true, true, true),
    crop277_efe_f4c5_f3c6_nl => ("crop277_efe_f4c5_f3c6_nl", true, true, true),
    img01_efe_f2c0_f3c0_nl => ("img01_efe_f2c0_f3c0_nl", true, true, true),
    // 4:4:4 source coded 4:2:0 / 4:2:2 (DCT-IF kernels):
    img30_c420_efe_f1c0_f2c1 => ("img30_c420_efe_f1c0_f2c1", true, true, true),
    img30_c420_efe_f3c5_f4c7_nl => ("img30_c420_efe_f3c5_f4c7_nl", true, true, true),
    crop277_c420_efe_f4c6_f3c3_nl => ("crop277_c420_efe_f4c6_f3c3_nl", true, true, true),
    img30_c422_efe_f3c2_f2c4 => ("img30_c422_efe_f3c2_f2c4", true, true, true),
    // `DCTIF_only`: no coded linear filters; non-linear filter on U only, no masks.
    img30_c420_efe_dctif => ("img30_c420_efe_dctif", true, true, false),
    // 4:2:0 and 4:2:2 sources:
    crop277_s420_efe_f3c5_f4c7_nl => ("crop277_s420_efe_f3c5_f4c7_nl", true, true, true),
    img30_s420_efe_f1c0_f2c1_nl => ("img30_s420_efe_f1c0_f2c1_nl", true, true, true),
    crop277_s422_efe_f3c6_f4c2_nl => ("crop277_s422_efe_f3c6_f4c2_nl", true, true, true),
}
