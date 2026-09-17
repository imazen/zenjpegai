//! Time the LEF and eICCI post-filters on a reference vector.
//!
//!     ZENJPEGAI_REF=... cargo run --release --example prof_filters_lef_icci -- <vector dir> [reps]
//!
//! Reads `<vector>/stream.bits` for the headers and `<vector>/filters_lef_icci/` (written by
//! `scripts/ref_vectors/dump_filters_lef_icci.py`) for the planes the reference handed to its
//! filters. Prints tab-separated `filter threads best_ms first_ms`; `first` includes loading the
//! eICCI networks.
use std::time::Instant;

use zenjpegai::container::Codestream;
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::Planes;
use zenjpegai::filters::{icci, lef};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;

struct Dump {
    manifest: String,
    blob: Vec<u8>,
}

impl Dump {
    fn f32(&self, name: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        let line = self
            .manifest
            .lines()
            .find(|l| l.split(' ').next() == Some(name))?;
        let f: Vec<&str> = line.split(' ').collect();
        let n: Vec<usize> = f[2..].iter().map(|v| v.parse().unwrap()).collect();
        let (shape, off, len) = (n[1..5].to_vec(), n[5], n[6]);
        let bytes = &self.blob[off..off + len];
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| match f[1] {
                "i32" => i32::from_le_bytes(*b) as f32,
                _ => f32::from_le_bytes(*b),
            })
            .collect();
        Some((shape, data))
    }

    fn planes(&self, prefix: &str) -> Option<Planes> {
        let plane = |c: &str| {
            let (s, d) = self.f32(&format!("{prefix}.{c}"))?;
            Some(Tensor::from_vec(1, s[2], s[3], d).unwrap())
        };
        Some(Planes {
            y: plane("a")?,
            u: plane("b")?,
            v: plane("c")?,
        })
    }
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let reps: u32 = std::env::args().nth(2).map_or(20, |s| s.parse().unwrap());
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let headers = read_headers(&Codestream::parse(&stream).unwrap()).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let sub = dir.join("filters_lef_icci");
    let dump = Dump {
        manifest: std::fs::read_to_string(sub.join("manifest.txt")).unwrap(),
        blob: std::fs::read(sub.join("tensors.bin")).unwrap(),
    };
    for parallel in [false, true] {
        let eng = Engine::with(Engine::new().tier, parallel);
        let threads = if parallel {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            1
        };
        if let (Some(h), Some(input)) = (&headers.tools.icci, dump.planes("eicci.in")) {
            let cache = icci::NetCache::default();
            let (mut best, mut first) = (f64::MAX, None);
            for _ in 0..reps {
                let image = input.clone();
                let t = Instant::now();
                let op = hdr.synthesis_transforms[0];
                std::hint::black_box(
                    icci::filter(&eng, hdr, h, op, &models, &cache, image).unwrap(),
                );
                let ms = t.elapsed().as_secs_f64() * 1e3;
                first.get_or_insert(ms);
                best = best.min(ms);
            }
            println!("eICCI\t{threads}\t{best:.2}\t{:.2}", first.unwrap());
        }
        if let (Some(ch), Some(input)) = (headers.tools.lef_channel, dump.planes("lef.in")) {
            let (s, d) = dump.f32("lef.scale_log").unwrap();
            let scale_log =
                Tensor::from_vec(s[1], s[2], s[3], d.iter().map(|&v| v as i32).collect()).unwrap();
            let (mut best, mut first) = (f64::MAX, None);
            for _ in 0..reps {
                let mut y = input.y.clone();
                let t = Instant::now();
                lef::sharpen(
                    &eng,
                    &mut y,
                    &scale_log,
                    ch as usize,
                    hdr.model_id as usize,
                    hdr.bit_depth,
                )
                .unwrap();
                std::hint::black_box(&y);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                first.get_or_insert(ms);
                best = best.min(ms);
            }
            println!("LEF\t{threads}\t{best:.2}\t{:.2}", first.unwrap());
        }
    }
}
