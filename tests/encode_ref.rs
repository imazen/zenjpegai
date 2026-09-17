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
