//! Scratch: time the EFE linear / non-linear post-filters on a reference vector.
//!
//!     cargo run --release --example prof_filters_efe -- <vector dir> [reps]
//!
//! Reads `<vector>/stream.bits` for the headers and `<vector>/filters/` (written by
//! `scripts/ref_vectors/dump_filters.py`) for the planes the reference handed to its filters.
use std::time::Instant;

use zenjpegai::container::Codestream;
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::Planes;
use zenjpegai::filters::{FilterContext, FilterState, efe_linear, efe_nonlinear};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;

fn load(dir: &std::path::Path, prefix: &str) -> Option<Planes> {
    let manifest = std::fs::read_to_string(dir.join("manifest.txt")).unwrap();
    let blob = std::fs::read(dir.join("tensors.bin")).unwrap();
    let plane = |name: String| -> Option<Tensor<f32>> {
        let line = manifest
            .lines()
            .find(|l| l.split(' ').next() == Some(&name))?;
        let f: Vec<usize> = line
            .split(' ')
            .skip(2)
            .map(|v| v.parse().unwrap())
            .collect();
        let (h, w, off, len) = (f[3], f[4], f[5], f[6]);
        let data = blob[off..off + len]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Some(Tensor::from_vec(1, h, w, data).unwrap())
    };
    Some(Planes {
        y: plane(format!("{prefix}.a"))?,
        u: plane(format!("{prefix}.b"))?,
        v: plane(format!("{prefix}.c"))?,
    })
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let reps: u32 = std::env::args().nth(2).map_or(50, |s| s.parse().unwrap());
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let scale_log = Tensor::<i32>::zeros(1, 1, 1).unwrap();
    for parallel in [false, true] {
        let eng = Engine::with(Engine::new().tier, parallel);
        let ctx = FilterContext {
            eng: &eng,
            hdr: &headers.picture,
            tools: &headers.tools,
            luma_scale_log: &scale_log,
            models: &models,
            op: headers.picture.synthesis_transforms[0],
            icci_nets: &Default::default(),
        };
        if let (Some(h), Some(input)) = (
            &headers.tools.efe_linear,
            load(&dir.join("filters"), "EFElinear.in"),
        ) {
            let mut best = std::time::Duration::MAX;
            for _ in 0..reps {
                let state = FilterState {
                    image: input.clone(),
                    upsampled: None,
                };
                let t = Instant::now();
                std::hint::black_box(efe_linear::apply(&ctx, h, state).unwrap());
                best = best.min(t.elapsed());
            }
            println!("EFE linear     parallel={parallel}: best of {reps}: {best:.2?}");
        }
        if let (Some(h), Some(input)) = (
            &headers.tools.efe_nonlinear,
            load(&dir.join("filters"), "EFEnonlinear.in"),
        ) {
            let alt = load(&dir.join("filters"), "EFEnonlinear.alt");
            let mut best = std::time::Duration::MAX;
            for _ in 0..reps {
                let state = FilterState {
                    image: input.clone(),
                    upsampled: alt.clone(),
                };
                let t = Instant::now();
                std::hint::black_box(efe_nonlinear::apply(&ctx, h, state).unwrap());
                best = best.min(t.elapsed());
            }
            println!("EFE non-linear parallel={parallel}: best of {reps}: {best:.2?}");
        }
    }
}
