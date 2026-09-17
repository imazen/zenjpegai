//! The attention math dispatches by CPU features on its own (`#[autoversion]`), independent of
//! the engine's tier. This binary disables CPU feature tokens one permutation at a time
//! (process-wide, hence its own test binary) and demands bit-identical results throughout.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenjpegai::nn::fast::{Engine, Tier, math};
use zenjpegai::tensor::Tensor;

fn data(n: usize, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0
        })
        .collect()
}

#[test]
fn attention_math_is_identical_on_every_cpu_tier() {
    let (dim, h, w) = (32usize, 9usize, 21usize);
    let qkv = Tensor::from_vec(3 * dim, h, w, data(3 * dim * h * w, 0xA77E)).unwrap();
    let (wt, bs) = (data(3 * dim, 1), data(3 * dim, 2));
    let temperature = [0.7f32, 1.3, 2.1, 0.05];
    let run = || {
        let eng = Engine::with(Tier::Scalar, false);
        let attn = math::channel_attention(&eng, &qkv, 4, &temperature).unwrap();
        let mut ln = qkv.clone();
        math::layer_norm_channels(&mut ln, &wt, &bs).unwrap();
        let mut sg: Vec<f32> = qkv.data.iter().map(|v| v * 9.0).collect();
        math::sigmoid(&eng, &mut sg);
        let mut gate = qkv.data.clone();
        math::elu_gate(&eng, &mut gate, &ln.data).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        [bits(&attn.data), bits(&ln.data), bits(&sg), bits(&gate)]
    };
    let mut want = None;
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let got = run();
        match &want {
            None => want = Some(got),
            Some(w) => assert!(*w == got, "results changed at CPU tier permutation {perm}"),
        }
    });
    #[cfg(target_arch = "x86_64")]
    assert!(
        report.permutations_run >= 2,
        "only one CPU tier was exercised"
    );
    println!("{report:?}");
}
