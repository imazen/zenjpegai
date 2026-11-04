//! `GpuAnalysis` against the CPU `model::analysis` transforms on the same weights and a random
//! picture, within the relative bound the whole-synthesis decode_ref uses (the GPU float
//! pipeline is not bit-exact — drivers choose FMA fusion and reduction order).
#![cfg(feature = "gpu-tests")]

mod common;
use common::{Rng, context_arc, max_abs, max_mag, models_dir};
use std::sync::{Arc, Mutex};
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{ModelDir, load_analysis_primary, load_analysis_secondary};
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::GpuAnalysis;
use zenjpegai_gpu::plan::{Graph, Pool, T};

fn run_analysis(
    f: impl FnOnce(&mut Graph<'_>, T) -> Result<T, zenjpegai_gpu::GpuError>,
    x: &Tensor<f32>,
) -> Tensor<f32> {
    let ctx = context_arc();
    let mut g = Graph::new(&ctx);
    let t = g.input(x.c, x.h, x.w).unwrap();
    let out = f(&mut g, t).unwrap();
    g.output(out);
    let plan = g.finish(&Arc::new(Mutex::new(Pool::default()))).unwrap();
    // Normalize `(v - 127.5) / 128` the way `GpuEncoder` does before upload.
    let norm: Vec<f32> = x.data.iter().map(|&v| (v - 127.5) / 128.0).collect();
    plan.write(&ctx, t, &norm).unwrap();
    let data = pollster::block_on(plan.run_and_read(&ctx, out)).unwrap();
    Tensor::from_vec(out.c, out.h, out.w, data).unwrap()
}

fn check(name: &str, want: &Tensor<f32>, got: &Tensor<f32>, rel: f32) {
    assert_eq!(
        (want.c, want.h, want.w),
        (got.c, got.h, got.w),
        "{name}: shape"
    );
    let (err, mag) = (max_abs(&want.data, &got.data), max_mag(&want.data));
    println!("{name}: max abs err {err:.3e} (rel {:.3e})", err / mag);
    assert!(err <= rel * mag, "{name}: {err:e} > {rel:e} * {mag:e}");
}

#[test]
fn analysis_matches_cpu() {
    let mut rng = Rng::new("analysis");
    let models = ModelDir::new(models_dir());
    let eng = Engine::new();
    let stop = enough::Unstoppable;
    for op in [OperatingPoint::Bop, OperatingPoint::Hop] {
        let gpu = GpuAnalysis::load(context_arc(), &ModelDir::new(models_dir()), 1, op).unwrap();
        // Primary (luma): [1, h, w] -> [160, h/16, w/16]. The CPU forward
        // normalizes its input itself, so `x` is handed over raw.
        for &(h, w) in &[(48usize, 64usize), (33, 47)] {
            let x = rng.tensor(1, h, w);
            let want = load_analysis_primary(&models, 1, op, &eng)
                .unwrap()
                .forward(&eng, &x, &stop)
                .unwrap();
            let got = run_analysis(|g, t| gpu.luma(g, t, h, w), &x);
            check(
                &format!("luma {op:?} {h}x{w}"),
                &want.to_planar().unwrap(),
                &got,
                3e-4,
            );
        }
        // Secondary (chroma): [12, h, w] -> [96, h/8, w/8].
        let (h, w) = (48usize, 64usize);
        let x = rng.tensor(12, h, w);
        let want = load_analysis_secondary(&models, 1, op, &eng)
            .unwrap()
            .forward(&eng, &x, &stop)
            .unwrap();
        let got = run_analysis(|g, t| gpu.chroma(g, t, h, w), &x);
        check(
            &format!("chroma {op:?}"),
            &want.to_planar().unwrap(),
            &got,
            3e-4,
        );
    }
}
