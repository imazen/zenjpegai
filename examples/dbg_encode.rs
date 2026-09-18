//! Diagnostic: where the encoder's integer decisions diverge from the reference encoder's.
//!
//! `cargo run --release --all-features --example dbg_encode -- <vector> <model_id> <op> <beta>
//!  <height> <width>`
//! with the reference dumps in `$ZENJPEGAI_VECTORS/<vector>/enc2`.

#[path = "../tests/common/mod.rs"]
mod common;

use zenjpegai::encoder::{EncodeParams, Encoder};
use zenjpegai::header::OperatingPoint;
use zenjpegai::tensor::Tensor;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (vector, model_id, op, beta) = (
        a[0].clone(),
        a[1].parse::<u8>().unwrap(),
        match a[2].as_str() {
            "sop" => OperatingPoint::Sop,
            "hop" => OperatingPoint::Hop,
            _ => OperatingPoint::Bop,
        },
        a[3].parse::<i32>().unwrap(),
    );
    let dump = common::load_encoder_dump(&common::vector_dir(&vector).join("enc2"));
    let t = |n: &str| {
        let r = &dump[n];
        Tensor::from_vec(r.shape[1], r.shape[2], r.shape[3], r.f32()).unwrap()
    };
    let (yl, yc) = (t("y.y"), t("uv.y"));
    let zi = |n: &str| {
        let r = &dump[n];
        Tensor::from_vec(r.shape[1], r.shape[2], r.shape[3], r.i8()).unwrap()
    };
    let (zy, zc) = (zi("y.z_hat"), zi("uv.z_hat"));
    // The coded picture size, from the merged luma latent (16 samples per latent sample).
    let (h, w) = (
        a[4].parse::<usize>().unwrap(),
        a[5].parse::<usize>().unwrap(),
    );
    let enc = Encoder::new(common::ref_root().join("models"));
    let (_, traces) = enc
        .encode_latents(
            [&yl, &yc],
            Some([&zy, &zc]),
            w,
            h,
            EncodeParams {
                model_id,
                beta_displacement_log: [beta, beta],
                op,
            },
        )
        .unwrap();
    for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
        let want = dump[&format!("{name}.residual_quant")].i32();
        let got = &traces[ccs].residual_q;
        let mut worst = 0i32;
        let mut n = 0usize;
        let mut first = Vec::new();
        for (i, (&a, &b)) in want.iter().zip(&got.data).enumerate() {
            let d = a - b as i32;
            if d != 0 {
                n += 1;
                worst = worst.max(d.abs());
                if first.len() < 60 {
                    let (ch, rest) = (i / (got.h * got.w), i % (got.h * got.w));
                    first.push((ch, rest / got.w, rest % got.w, a, b as i32));
                }
            }
        }
        println!(
            "{name}: {n} of {} symbols differ, worst |delta| {worst}",
            want.len()
        );
        for (c, y, x, a, b) in first {
            println!("   ch {c} y {y} x {x}: reference {a}, ours {b}");
        }
        // How far the dequantised residual is from the reference's: the float-error scale.
        let wr = dump[&format!("{name}.residual")].f32();
        let max = wr
            .iter()
            .zip(&traces[ccs].residual.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("   dequantised residual max abs diff {max:e}");
    }
    for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
        let want = &dump[&format!("{name}.cube_flag")].bytes;
        let got = &traces[ccs].cube_flag;
        for (i, (&a, &b)) in want.iter().zip(got).enumerate() {
            if (a != 0) != b {
                println!("{name}.cube_flag[{i}]: reference {}, ours {b}", a != 0);
            }
        }
    }
    // Is `psi` itself already divergent, or is the difference inside the context model?
    let models = zenjpegai::model::ModelDir::new(common::ref_root().join("models"));
    let eng = zenjpegai::nn::fast::Engine::new();
    for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
        let cm = models.load_common(model_id as usize, ccs, &eng).unwrap();
        let zr = &dump[&format!("{name}.z_hat")];
        let z = Tensor::from_vec(zr.shape[1], zr.shape[2], zr.shape[3], zr.i8()).unwrap();
        let want = &dump[&format!("{name}.psi")];
        let psi = cm
            .hyper_decoder
            .forward(&eng, &z, want.shape[2], want.shape[3])
            .unwrap()
            .to_planar()
            .unwrap();
        let w = want.f32();
        let max = w
            .iter()
            .zip(&psi.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let bits = w
            .iter()
            .zip(&psi.data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{name}.psi: max abs diff {max:e}, {bits} of {} not bit-identical",
            w.len()
        );
    }
}
