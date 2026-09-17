//! Scratch: run the reconstruction stage and report float differences against a reference
//! decoder dump. usage: dbg_recon <vector_dir>
use std::time::Instant;

use zenjpegai::container::Codestream;
use zenjpegai::decoder::reconstruct::{reconstruct_latent, synthesize};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;

fn stats(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut max_ref = 0f32;
    for (w, g) in want.iter().zip(got) {
        let d = (w - g).abs();
        max_abs = max_abs.max(d);
        sum_sq += (d as f64) * (d as f64);
        max_ref = max_ref.max(w.abs());
    }
    let rmse = (sum_sq / want.len() as f64).sqrt();
    println!(
        "{name:12} n={:9} max|ref|={max_ref:10.4} max_abs_diff={max_abs:.3e} rmse={rmse:.3e}",
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

    let t0 = Instant::now();
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let id = hdr.model_id as usize;
    let op = hdr.synthesis_transforms[0];
    let (ym, uvm) = (
        models.load_common(id, 0).unwrap(),
        models.load_common(id, 1).unwrap(),
    );
    let (syn_y, syn_uv) = (
        models.load_synthesis_primary(id, op).unwrap(),
        models.load_synthesis_secondary(id, op).unwrap(),
    );
    let t_load = t0.elapsed();

    let t1 = Instant::now();
    let ent = decode_entropy_stage(&AnsTables::new(), &cs, hdr, [&ym, &uvm]).unwrap();
    let t_ent = t1.elapsed();
    let t2 = Instant::now();
    let ly = reconstruct_latent(&ym, &ent[0]).unwrap();
    let luv = reconstruct_latent(&uvm, &ent[1]).unwrap();
    let t_lat = t2.elapsed();
    let t3 = Instant::now();
    let planes = synthesize(hdr, &syn_y, &syn_uv, [&ly.y_hat, &luv.y_hat]).unwrap();
    let t_syn = t3.elapsed();
    println!(
        "{}x{} model {id} {op:?}: load {t_load:.2?} entropy {t_ent:.2?} latent {t_lat:.2?} synthesis {t_syn:.2?}",
        hdr.width, hdr.height
    );

    stats("y.psi", &get("y.psi "), &ly.psi.data);
    stats("y.y_hat", &get("y.y_hat "), &ly.y_hat.data);
    stats("uv.psi", &get("uv.psi "), &luv.psi.data);
    stats("uv.y_hat", &get("uv.y_hat "), &luv.y_hat.data);
    stats("rec.a", &get("rec.a "), &planes.y.data);
    stats("rec.b", &get("rec.b "), &planes.u.data);
    stats("rec.c", &get("rec.c "), &planes.v.data);

    let rgb = zenjpegai::decoder::output::to_rgb_planes(hdr, &planes).unwrap();
    stats("out.a (R)", &get("out.a "), &rgb.r);
    stats("out.b (G)", &get("out.b "), &rgb.g);
    stats("out.c (B)", &get("out.c "), &rgb.b);
    // 8-bit agreement: quantise both the reference's float planes and ours the same way.
    let ours = zenjpegai::decoder::output::quantize(&rgb, 8).unwrap();
    let refp = zenjpegai::decoder::output::RgbPlanes {
        width: rgb.width,
        height: rgb.height,
        r: get("out.a "),
        g: get("out.b "),
        b: get("out.c "),
    };
    let theirs = zenjpegai::decoder::output::quantize(&refp, 8).unwrap();
    let differ = ours
        .data
        .iter()
        .zip(&theirs.data)
        .filter(|(a, b)| a != b)
        .count();
    let max = ours
        .data
        .iter()
        .zip(&theirs.data)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    println!(
        "8-bit samples differing: {differ} of {} (max |diff| {max})",
        ours.data.len()
    );
}
