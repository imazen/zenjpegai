//! Scratch: where does the entropy stage spend its time? (perf is unavailable on this box.)
use std::time::Instant;

use zenjpegai::container::{Codestream, Marker, split_threads};
use zenjpegai::decoder::entropy::decode_z;
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::model::common::SIGMA_IDX_MAX;
use zenjpegai::nn::fast::Engine;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let reps: u32 = std::env::args().nth(2).map_or(20, |s| s.parse().unwrap());
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let eng = Engine::new();
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let ym = models.load_common(hdr.model_id as usize, 0, &eng).unwrap();
    let uvm = models.load_common(hdr.model_id as usize, 1, &eng).unwrap();
    let tables = AnsTables::new();

    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(decode_entropy_stage(&tables, &cs, hdr, [&ym, &uvm]).unwrap());
    }
    println!("entropy stage total: {:.2?}", t.elapsed() / reps);

    let soz = cs.find(Marker::Soz).unwrap();
    let (hz, wz) = hdr.hyper_latent_size(0);
    let (lh, lw) = hdr.latent_size(0);
    let t = Instant::now();
    let mut z = None;
    for _ in 0..reps {
        let threads = split_threads(soz, hdr.num_threads_z as usize).unwrap();
        let mut dec = tables.decoder(&threads).unwrap();
        z = Some(decode_z(&mut dec, &ym, hz as usize, wz as usize).unwrap());
    }
    println!("  luma z decode:       {:.2?}", t.elapsed() / reps);
    let z = z.unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(
            ym.hsd
                .forward(&z, lh as usize, lw as usize, SIGMA_IDX_MAX)
                .unwrap(),
        );
    }
    println!("  luma hyper-scale dec: {:.2?}", t.elapsed() / reps);
    let t = Instant::now();
    for _ in 0..reps {
        let zc =
            zenjpegai::tensor::Tensor::from_vec(96, z.h, z.w, z.data[..96 * z.h * z.w].to_vec())
                .unwrap();
        std::hint::black_box(
            uvm.hsd
                .forward(&zc, lh as usize, lw as usize, SIGMA_IDX_MAX)
                .unwrap(),
        );
    }
    println!(
        "  chroma hyper-scale dec (luma z reused as input): {:.2?}",
        t.elapsed() / reps
    );
}
