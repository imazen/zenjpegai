//! Convolution kernel throughput on the layer shapes of the BOP luma synthesis transform, at the
//! latent size of a ~2.9 MP picture (131 x 88). Throughput is in multiply-accumulates per second.
//!
//!     cargo bench --bench conv_kernels
use zenbench::prelude::*;
use zenjpegai::nn::fast::{BTensor, Engine, PackedConv, PackedConvTranspose, Tier};
use zenjpegai::nn::{Conv2d, ConvTranspose2d};
use zenjpegai::tensor::Tensor;

fn filled(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn tier_name(t: Tier) -> &'static str {
    match t {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Tier::V4(_) => "v4",
        #[cfg(target_arch = "x86_64")]
        Tier::V3(_) => "v3",
        #[cfg(target_arch = "aarch64")]
        Tier::Neon(_) => "neon",
        Tier::Scalar => "scalar",
    }
}

/// `[in_ch, out_ch, k, groups, h, w]`
fn conv_group(suite: &mut Suite, name: &str, shape: [usize; 6]) {
    let [in_ch, out_ch, k, groups, h, w] = shape;
    let conv = Conv2d::new(
        in_ch,
        out_ch,
        (k, k),
        1,
        (k / 2, k / 2),
        groups,
        filled(out_ch * in_ch / groups * k * k, 1),
        Some(filled(out_ch, 2)),
    )
    .unwrap();
    let x = Tensor::from_vec(in_ch, h, w, filled(in_ch * h * w, 3)).unwrap();
    let macs = (out_ch * (in_ch / groups) * k * k * h * w) as u64;
    suite.group(name, |g| {
        g.throughput(Throughput::Elements(macs));
        for tier in Tier::available() {
            if matches!(tier, Tier::Scalar) {
                continue;
            }
            for parallel in [false, true] {
                let eng = Engine::with(tier, parallel);
                let v = tier.block();
                let packed = PackedConv::new(&conv, v, [0; 4]).unwrap();
                let bx = BTensor::from_planar(&x, v).unwrap();
                let label = format!(
                    "{}{}",
                    tier_name(tier),
                    if parallel { "_mt" } else { "_1t" }
                );
                g.bench(label, move |b| {
                    b.iter(|| packed.forward(&eng, &bx).unwrap())
                });
            }
        }
    });
}

fn convt_group(suite: &mut Suite, name: &str, in_ch: usize, out_ch: usize, h: usize, w: usize) {
    let conv = ConvTranspose2d::new(
        in_ch,
        out_ch,
        4,
        2,
        1,
        0,
        filled(in_ch * out_ch * 16, 4),
        Some(filled(out_ch, 5)),
    )
    .unwrap();
    let x = Tensor::from_vec(in_ch, h, w, filled(in_ch * h * w, 6)).unwrap();
    // every output position sees 2x2 taps of every input channel
    let macs = (out_ch * in_ch * 4 * (2 * h) * (2 * w)) as u64;
    suite.group(name, |g| {
        g.throughput(Throughput::Elements(macs));
        for tier in Tier::available() {
            if matches!(tier, Tier::Scalar) {
                continue;
            }
            for parallel in [false, true] {
                let eng = Engine::with(tier, parallel);
                let v = tier.block();
                let packed = PackedConvTranspose::new(&conv, v).unwrap();
                let bx = BTensor::from_planar(&x, v).unwrap();
                let label = format!(
                    "{}{}",
                    tier_name(tier),
                    if parallel { "_mt" } else { "_1t" }
                );
                g.bench(label, move |b| {
                    b.iter(|| packed.forward(&eng, &bx).unwrap())
                });
            }
        }
    });
}

fn benches(suite: &mut Suite) {
    let (h, w) = (88, 131);
    conv_group(suite, "res_conv3x3_160_160@1/16", [160, 160, 3, 1, h, w]);
    convt_group(suite, "up1_convT4x4_160_64@1/16", 160, 64, h, w);
    conv_group(
        suite,
        "resau_conv3x3_g4_64@1/8",
        [64, 64, 3, 4, 2 * h, 2 * w],
    );
    conv_group(
        suite,
        "resau_conv1x1_64_64@1/8",
        [64, 64, 1, 1, 2 * h, 2 * w],
    );
    convt_group(suite, "up2_convT4x4_64_64@1/8", 64, 64, 2 * h, 2 * w);
    conv_group(suite, "conv3_3x3_64_96@1/4", [64, 96, 3, 1, 4 * h, 4 * w]);
    conv_group(suite, "conv4_1x1_96_16@1/4", [96, 16, 1, 1, 4 * h, 4 * w]);
}

zenbench::main!(benches);
