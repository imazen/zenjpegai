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
    let regions = a.get(6).and_then(|m| match m.as_str() {
        "dep" => Some(zenjpegai::encoder::RegionMode::Dependent),
        "ind" => Some(zenjpegai::encoder::RegionMode::Independent),
        _ => None,
    });
    let enc = Encoder::new(common::ref_root().join("models"));
    let (stream_out, traces) = enc
        .encode_latents(
            [&yl, &yc],
            Some([&zy, &zc]),
            w,
            h,
            EncodeParams {
                model_id,
                beta_displacement_log: [beta, beta],
                op,
                regions,
                ..Default::default()
            },
        )
        .unwrap();
    {
        let cs = zenjpegai::container::Codestream::parse(&stream_out).unwrap();
        let ours = zenjpegai::decoder::read_headers(&cs).unwrap().picture;
        let refs = std::fs::read(common::vector_dir(&vector).join("stream.bits")).unwrap();
        let cs2 = zenjpegai::container::Codestream::parse(&refs).unwrap();
        let theirs = zenjpegai::decoder::read_headers(&cs2).unwrap().picture;
        println!("header regions ours {:?}", ours.regions);
        println!("header regions ref  {:?}", theirs.regions);
        println!(
            "header size ours {}x{} ref {}x{}",
            ours.width, ours.height, theirs.width, theirs.height
        );
    }
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
                if first.len() < 4 {
                    let (ch, rest) = (i / (got.h * got.w), i % (got.h * got.w));
                    first.push((ch, rest / got.w, rest % got.w, a, b as i32));
                }
            }
        }
        println!(
            "{name}: {n} of {} symbols differ, worst |delta| {worst}",
            want.len()
        );
        // Where, spatially?
        let mut cols = vec![0usize; got.w];
        let mut rows = vec![0usize; got.h];
        for (i, (&a, &b)) in want.iter().zip(&got.data).enumerate() {
            if a != b as i32 {
                let rest = i % (got.h * got.w);
                rows[rest / got.w] += 1;
                cols[rest % got.w] += 1;
            }
        }
        let nz: Vec<usize> = (0..got.w).filter(|&x| cols[x] > 0).collect();
        println!(
            "   differing columns {:?}..{:?} ({} of {})",
            nz.first(),
            nz.last(),
            nz.len(),
            got.w
        );
        let nzr: Vec<usize> = (0..got.h).filter(|&y| rows[y] > 0).collect();
        println!(
            "   differing rows {:?}..{:?} ({} of {})",
            nzr.first(),
            nzr.last(),
            nzr.len(),
            got.h
        );
        println!(
            "   per-column counts in 40..90: {:?}",
            &cols[40.min(got.w)..90.min(got.w)]
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
    for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
        let want = dump[&format!("{name}.psi")].f32();
        let got = &traces[ccs].psi;
        if got.data.len() != want.len() {
            println!(
                "{name}.psi: our merged psi is {} values, dump {}",
                got.data.len(),
                want.len()
            );
            continue;
        }
        let max = want
            .iter()
            .zip(&got.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mut cols = vec![0usize; got.w];
        for (i, (&a, &b)) in want.iter().zip(&got.data).enumerate() {
            if (a - b).abs() > 1e-3 {
                cols[(i % (got.h * got.w)) % got.w] += 1;
            }
        }
        let nz: Vec<usize> = (0..got.w).filter(|&x| cols[x] > 0).collect();
        println!(
            "{name}.psi merged: max abs diff {max:e}, differing columns {:?}..{:?}",
            nz.first(),
            nz.last()
        );
    }
    // Cross-check: the decoder's own region merge on the same z_hat.
    {
        use zenjpegai::decoder::entropy::ComponentEntropy;
        use zenjpegai::decoder::reconstruct::reconstruct_latent;
        let models = zenjpegai::model::ModelDir::new(common::ref_root().join("models"));
        let eng = zenjpegai::nn::fast::Engine::new();
        let stream = std::fs::read(common::vector_dir(&vector).join("stream.bits")).unwrap();
        let cs = zenjpegai::container::Codestream::parse(&stream).unwrap();
        let hdr = zenjpegai::decoder::read_headers(&cs).unwrap().picture;
        for (ccs, name) in ["y", "uv"].into_iter().enumerate() {
            let cm = models.load_common(model_id as usize, ccs, &eng).unwrap();
            let zr = &dump[&format!("{name}.z_hat")];
            let z = Tensor::from_vec(zr.shape[1], zr.shape[2], zr.shape[3], zr.i8()).unwrap();
            let (lh, lw) = hdr.latent_size(ccs);
            let e = ComponentEntropy {
                z_hat: z,
                skip_scale_log: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
                scale_log: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
                likely: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
                mask: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
                residual_q: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
                residual: Tensor::zeros(cm.chs, lh as usize, lw as usize).unwrap(),
            };
            let l = reconstruct_latent(&eng, &hdr, ccs, &cm, &e).unwrap();
            let want = dump[&format!("{name}.psi")].f32();
            let max = want
                .iter()
                .zip(&l.psi.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{name}.psi via the decoder's merge: max abs diff {max:e}");
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
