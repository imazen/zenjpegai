//! Scratch: run the whole decode and report timings plus float differences against a reference
//! decoder dump. usage: dbg_recon <vector_dir>   (ZENJPEGAI_THREADS=1 for single-threaded)
use std::time::Instant;

use zenjpegai::container::Codestream;
use zenjpegai::decoder::output::{RgbPlanes, quantize, to_rgb_planes};
use zenjpegai::decoder::reconstruct::{reconstruct_latent, synthesize};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::{Engine, Tier};

fn stats(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for (w, g) in want.iter().zip(got) {
        let d = (w - g).abs();
        max_abs = max_abs.max(d);
        sum_sq += (d as f64) * (d as f64);
    }
    let rmse = (sum_sq / want.len() as f64).sqrt();
    println!(
        "{name:12} n={:9} max_abs_diff={max_abs:.3e} rmse={rmse:.3e}",
        want.len()
    );
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).unwrap();
    let blob = std::fs::read(dir.join("tensors.bin")).unwrap();
    let get = |name: &str| -> Vec<f32> {
        let line = manifest.lines().find(|l| l.starts_with(name)).unwrap();
        let f: Vec<&str> = line.split_whitespace().collect();
        let ndim: usize = f[2].parse().unwrap();
        let off: usize = f[3 + ndim].parse().unwrap();
        let n: usize = f[4 + ndim].parse().unwrap();
        blob[off..off + n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    let eng = match std::env::var("ZENJPEGAI_THREADS").as_deref() {
        Ok("1") => Engine::with(Tier::detect(), false),
        _ => Engine::new(),
    };

    let t0 = Instant::now();
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let id = hdr.model_id as usize;
    let op = hdr.synthesis_transforms[0];
    let ym = models.load_common(id, 0, &eng).unwrap();
    let uvm = models.load_common(id, 1, &eng).unwrap();
    let syn_y = models.load_synthesis_primary(id, op, &eng).unwrap();
    let syn_uv = models.load_synthesis_secondary(id, op, &eng).unwrap();
    let t_load = t0.elapsed();

    for run in 0..3 {
        let t1 = Instant::now();
        let ent = decode_entropy_stage(&AnsTables::new(), &cs, hdr, [&ym, &uvm]).unwrap();
        let t_ent = t1.elapsed();
        let t2 = Instant::now();
        let ly = reconstruct_latent(&eng, hdr, 0, &ym, &ent[0]).unwrap();
        let luv = reconstruct_latent(&eng, hdr, 1, &uvm, &ent[1]).unwrap();
        let t_lat = t2.elapsed();
        let t3 = Instant::now();
        let planes = synthesize(&eng, hdr, &syn_y, &syn_uv, [&ly.y_hat, &luv.y_hat]).unwrap();
        let t_syn = t3.elapsed();
        let t4 = Instant::now();
        let rgb = to_rgb_planes(hdr, &planes).unwrap();
        let ours = quantize(&rgb, 8).unwrap();
        let t_out = t4.elapsed();
        println!(
            "run {run}: {eng:?} {}x{} model {id} {op:?}: load {t_load:.2?} | entropy {t_ent:.2?} latent {t_lat:.2?} synthesis {t_syn:.2?} output {t_out:.2?} | decode total {:.2?}",
            hdr.width,
            hdr.height,
            t1.elapsed()
        );
        if run == 2 {
            stats("y.psi", &get("y.psi "), &ly.psi.data);
            stats("y.y_hat", &get("y.y_hat "), &ly.y_hat.data);
            stats("uv.y_hat", &get("uv.y_hat "), &luv.y_hat.data);
            stats("rec.a", &get("rec.a "), &planes.y.data);
            stats("rec.b", &get("rec.b "), &planes.u.data);
            stats("rec.c", &get("rec.c "), &planes.v.data);
            let refp = RgbPlanes {
                width: rgb.width,
                height: rgb.height,
                r: get("out.a "),
                g: get("out.b "),
                b: get("out.c "),
            };
            let theirs = quantize(&refp, 8).unwrap();
            let differ = ours
                .data
                .iter()
                .zip(&theirs.data)
                .filter(|(a, b)| a != b)
                .count();
            println!("8-bit samples differing: {differ} of {}", ours.data.len());
        }
    }
}
