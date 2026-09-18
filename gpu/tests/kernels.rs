//! Every WGSL kernel against the plain-loop oracle of the core crate (`zenjpegai::nn::reference`,
//! `zenjpegai::nn::fast::math` for the attention pieces) on random tensors.
//!
//! GPU float results are not bit-identical to the CPU contract (drivers choose whether to fuse
//! multiply-adds, reductions run in a different order), so each test asserts a bound and prints
//! the measured maximum. Bounds are relative to the largest reference magnitude.
#![cfg(feature = "gpu-tests")]

mod common;
use common::{Rng, context, max_abs, max_mag};
use std::sync::{Arc, Mutex};
use zenjpegai::nn::fast::{Engine, math};
use zenjpegai::nn::{Conv2d, ConvTranspose2d, reference};
use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::kernels::{Act, Pointwise};
use zenjpegai_gpu::layers::{GpuConv, GpuConvTranspose, GpuDepthwise, GpuLayerNorm, f32_buffer};
use zenjpegai_gpu::plan::{Graph, Pool, T};

fn run(build: impl FnOnce(&mut Graph<'_>, &[T]) -> T, inputs: &[&Tensor<f32>]) -> Tensor<f32> {
    let ctx = context();
    let mut g = Graph::new(ctx);
    let ins: Vec<T> = inputs
        .iter()
        .map(|t| g.input(t.c, t.h, t.w).unwrap())
        .collect();
    let out = build(&mut g, &ins);
    g.output(out);
    let plan = g.finish(&Arc::new(Mutex::new(Pool::default()))).unwrap();
    for (t, data) in ins.iter().zip(inputs) {
        plan.write(ctx, *t, &data.data).unwrap();
    }
    let data = pollster::block_on(plan.run_and_read(ctx, out)).unwrap();
    Tensor::from_vec(out.c, out.h, out.w, data).unwrap()
}

fn check(name: &str, want: &Tensor<f32>, got: &Tensor<f32>, rel: f32) {
    assert_eq!(
        (want.c, want.h, want.w),
        (got.c, got.h, got.w),
        "{name}: shape"
    );
    let (err, mag) = (max_abs(&want.data, &got.data), max_mag(&want.data));
    println!(
        "{name}: max abs err {err:.3e} (max |ref| {mag:.3e}, relative {:.3e})",
        err / mag
    );
    assert!(
        err <= rel * mag,
        "{name}: max abs err {err:e} exceeds {rel:e} * {mag:e}"
    );
}

#[allow(clippy::too_many_arguments)]
fn conv_case(
    name: &str,
    ic: usize,
    oc: usize,
    k: (usize, usize),
    stride: usize,
    pad: (usize, usize),
    groups: usize,
    bias: bool,
    h: usize,
    w: usize,
) {
    let mut rng = Rng::new(name);
    let c = Conv2d::new(
        ic,
        oc,
        k,
        stride,
        pad,
        groups,
        rng.vec(oc * (ic / groups) * k.0 * k.1, 0.5),
        bias.then(|| rng.vec(oc, 1.0)),
    )
    .unwrap();
    let x = rng.tensor(ic, h, w);
    let want = reference::conv2d(&c, &x).unwrap();
    let got = run(
        |g, i| g.conv(i[0], &GpuConv::new(context(), &c).unwrap()).unwrap(),
        &[&x],
    );
    check(name, &want, &got, 2e-6);
}

#[test]
fn conv_geometries() {
    conv_case(
        "conv3x3 160->160 bias",
        160,
        160,
        (3, 3),
        1,
        (1, 1),
        1,
        true,
        19,
        23,
    );
    conv_case(
        "conv3x3 256->48 bias",
        256,
        48,
        (3, 3),
        1,
        (1, 1),
        1,
        true,
        9,
        11,
    );
    conv_case(
        "conv3x3 grouped 64/4 groups",
        64,
        64,
        (3, 3),
        1,
        (1, 1),
        4,
        false,
        17,
        18,
    );
    conv_case(
        "conv3x3 grouped 96/6 groups (16 per group)",
        96,
        96,
        (3, 3),
        1,
        (1, 1),
        6,
        false,
        13,
        9,
    );
    conv_case(
        "conv1x1 96->16",
        96,
        16,
        (1, 1),
        1,
        (0, 0),
        1,
        false,
        33,
        31,
    );
    conv_case(
        "conv1x1 64->512 bias",
        64,
        512,
        (1, 1),
        1,
        (0, 0),
        1,
        true,
        10,
        12,
    );
    conv_case(
        "conv3x3 stride 2",
        64,
        64,
        (3, 3),
        2,
        (1, 1),
        1,
        true,
        18,
        22,
    );
    conv_case(
        "conv3x3 stride 2 odd size",
        32,
        32,
        (3, 3),
        2,
        (1, 1),
        1,
        true,
        17,
        21,
    );
    conv_case(
        "conv3x3 odd channels 5->7",
        5,
        7,
        (3, 3),
        1,
        (1, 1),
        1,
        true,
        12,
        10,
    );
    conv_case("conv1x3", 8, 8, (1, 3), 1, (0, 1), 1, true, 9, 9);
    conv_case(
        "conv3x3 24->24 (ob = 2)",
        24,
        24,
        (3, 3),
        1,
        (1, 1),
        1,
        true,
        9,
        9,
    );
}

/// Two convolutions that share every pipeline-key field but the baked output group size —
/// a regression test for the cache collision that ran the second with the first's shader.
#[test]
fn conv_grouped_pipeline_keys() {
    let mut rng = Rng::new("conv pipeline keys");
    let a = Conv2d::new(
        32,
        32,
        (1, 1),
        1,
        (0, 0),
        1,
        rng.vec(32 * 32, 0.5),
        Some(rng.vec(32, 1.0)),
    )
    .unwrap();
    // Same icg4, different ocg4 (32 -> 8 out blocks vs 32 -> 32 out blocks).
    let b = Conv2d::new(
        32,
        128,
        (1, 1),
        1,
        (0, 0),
        1,
        rng.vec(128 * 32, 0.5),
        Some(rng.vec(128, 1.0)),
    )
    .unwrap();
    let x = rng.tensor(32, 11, 13);
    let want = reference::conv2d(&b, &reference::conv2d(&a, &x).unwrap()).unwrap();
    let got = run(
        |g, i| {
            let y = g.conv(i[0], &GpuConv::new(context(), &a).unwrap()).unwrap();
            g.conv(y, &GpuConv::new(context(), &b).unwrap()).unwrap()
        },
        &[&x],
    );
    check("conv1x1 icg4 8 -> ocg4 8 then 32", &want, &got, 2e-6);
}

/// `F.pad(x, (0, 1, 0, 1))` + 2x2 convolution (SOP upsampling), and the fused ReLU / ReLU6.
#[test]
fn conv_extra_pad_and_fused_activations() {
    let mut rng = Rng::new("conv2x2");
    let c = Conv2d::new(
        64,
        128,
        (2, 2),
        1,
        (0, 0),
        1,
        rng.vec(128 * 64 * 4, 0.5),
        None,
    )
    .unwrap();
    let x = rng.tensor(64, 11, 13);
    let mut padded = Tensor::<f32>::zeros(64, 12, 14).unwrap();
    for ch in 0..64 {
        for y in 0..11 {
            padded.plane_mut(ch)[y * 14..][..13].copy_from_slice(&x.plane(ch)[y * 13..][..13]);
        }
    }
    let want = reference::conv2d(&c, &padded).unwrap();
    let got = run(
        |g, i| {
            g.conv_ex(
                i[0],
                &GpuConv::new(context(), &c).unwrap(),
                (1, 1),
                false,
                Act::None,
            )
            .unwrap()
        },
        &[&x],
    );
    check("conv2x2 with bottom/right zero pad", &want, &got, 2e-6);

    let c = Conv2d::new(
        32,
        32,
        (3, 3),
        1,
        (1, 1),
        2,
        rng.vec(32 * 16 * 9, 0.5),
        None,
    )
    .unwrap();
    let x = rng.tensor_scaled(32, 9, 10, 8.0);
    let mut act = x.clone();
    reference::relu6(&mut act);
    let mut want = reference::conv2d(&c, &act).unwrap();
    reference::relu(&mut want);
    let got = run(
        |g, i| {
            g.conv_ex(
                i[0],
                &GpuConv::new(context(), &c).unwrap(),
                (0, 0),
                true,
                Act::Relu,
            )
            .unwrap()
        },
        &[&x],
    );
    check("relu6 -> grouped conv3x3 -> relu, fused", &want, &got, 2e-6);
}

#[allow(clippy::too_many_arguments)]
fn convt_case(
    name: &str,
    ic: usize,
    oc: usize,
    k: usize,
    pad: usize,
    out_pad: usize,
    h: usize,
    w: usize,
) {
    let mut rng = Rng::new(name);
    let c = ConvTranspose2d::new(
        ic,
        oc,
        k,
        2,
        pad,
        out_pad,
        rng.vec(ic * oc * k * k, 0.5),
        Some(rng.vec(oc, 1.0)),
    )
    .unwrap();
    let x = rng.tensor(ic, h, w);
    let want = reference::conv_transpose2d(&c, &x).unwrap();
    let got = run(
        |g, i| {
            g.conv_transpose(i[0], &GpuConvTranspose::new(context(), &c).unwrap())
                .unwrap()
        },
        &[&x],
    );
    check(name, &want, &got, 2e-6);
}

#[test]
fn conv_transpose_geometries() {
    convt_case("convT 4x4 s2 p1 160->64", 160, 64, 4, 1, 0, 9, 11);
    convt_case("convT 4x4 s2 p1 144->64", 144, 64, 4, 1, 0, 7, 5);
    convt_case("convT 3x3 s2 p1 op1 128->128", 128, 128, 3, 1, 1, 8, 9);
    convt_case("convT 3x3 s2 p1 op1 128->1", 128, 1, 3, 1, 1, 10, 7);
    convt_case("convT 3x3 s2 p1 op1 64->8", 64, 8, 3, 1, 1, 6, 6);
}

#[test]
fn depthwise() {
    let mut rng = Rng::new("dw");
    for ch in [384, 6] {
        let c = Conv2d::new(ch, ch, (3, 3), 1, (1, 1), ch, rng.vec(ch * 9, 0.5), None).unwrap();
        let x = rng.tensor(ch, 13, 12);
        let want = reference::conv2d(&c, &x).unwrap();
        let got = run(
            |g, i| {
                g.depthwise(i[0], &GpuDepthwise::new(context(), &c).unwrap())
                    .unwrap()
            },
            &[&x],
        );
        check(&format!("depthwise 3x3, {ch} channels"), &want, &got, 2e-6);
    }
}

#[test]
fn shuffles_copies_and_pointwise() {
    let mut rng = Rng::new("misc");
    for (c, r) in [(64, 2), (16, 4), (128, 8), (12, 2)] {
        let x = rng.tensor(c, 5, 7);
        let want = reference::pixel_shuffle(&x, r).unwrap();
        let got = run(|g, i| g.pixel_shuffle(i[0], r).unwrap(), &[&x]);
        assert_eq!(want.data, got.data, "pixel shuffle {c} / {r}");
    }

    let (a, b) = (rng.tensor(48, 6, 5), rng.tensor(94, 6, 5));
    let want = reference::cat(&[&a, &b]).unwrap();
    let got = run(|g, i| g.cat(&[i[0], i[1]]).unwrap(), &[&a, &b]);
    assert_eq!(want.data, got.data, "cat");

    let x = rng.tensor(10, 9, 8);
    let want = x.window(2, 3, 5, 4).unwrap();
    let got = run(|g, i| g.window(i[0], 2, 3, 5, 4).unwrap(), &[&x]);
    assert_eq!(want.data, got.data, "window");

    let (x, m, t) = (
        rng.tensor_scaled(30, 7, 9, 4.0),
        rng.tensor(30, 7, 9),
        rng.tensor(30, 7, 9),
    );
    let pw = |op: Pointwise, n: usize| {
        run(
            |g, i| {
                g.pointwise(op, i[0], &i[1..1 + n]).unwrap();
                i[0]
            },
            &[&x, &m, &t],
        )
    };
    let map = |f: &dyn Fn(usize) -> f32| {
        Tensor::from_vec(30, 7, 9, (0..x.data.len()).map(f).collect()).unwrap()
    };
    check(
        "relu",
        &map(&|i| x.data[i].max(0.0)),
        &pw(Pointwise::Relu, 0),
        0.0,
    );
    check(
        "add",
        &map(&|i| x.data[i] + m.data[i]),
        &pw(Pointwise::Add, 1),
        0.0,
    );
    check(
        "gate x * (1 + m)",
        &map(&|i| x.data[i] * (1.0 + m.data[i])),
        &pw(Pointwise::Gate, 1),
        1e-6,
    );
    let mut sig = m.clone();
    math::sigmoid(&Engine::new(), &mut sig.data);
    check(
        "t + x * sigmoid(m)",
        &map(&|i| t.data[i] + x.data[i] * sig.data[i]),
        &pw(Pointwise::SigmoidMulAdd, 2),
        2e-6,
    );

    let x = rng.tensor_scaled(64, 6, 7, 3.0);
    let n = 32 * 42;
    let mut want = x.data[..n].to_vec();
    math::elu_gate(&Engine::new(), &mut want, &x.data[n..]).unwrap();
    let got = run(|g, i| g.elu_gate(i[0]).unwrap(), &[&x]);
    check(
        "elu gate",
        &Tensor::from_vec(32, 6, 7, want).unwrap(),
        &got,
        2e-6,
    );
}

#[test]
fn layer_norm_and_attention() {
    let mut rng = Rng::new("attn");
    for ch in [128usize, 64] {
        let (w, b) = (rng.vec(ch, 1.0), rng.vec(ch, 1.0));
        let x = rng.tensor_scaled(ch, 9, 11, 3.0);
        let mut want = x.clone();
        math::layer_norm_channels(&mut want, &w, &b).unwrap();
        let got = run(
            |g, i| {
                g.layer_norm(i[0], &GpuLayerNorm::new(context(), &w, &b).unwrap())
                    .unwrap()
            },
            &[&x],
        );
        check(&format!("layer norm over {ch} channels"), &want, &got, 5e-6);
    }
    // Sizes on both sides of the reduction chunk (256 pixels) and group (16 chunks) edges.
    for (dim, h, w) in [(128usize, 70, 61), (64, 16, 16), (64, 5, 3)] {
        let temperature = [0.7f32, 1.3, 2.1, 0.4];
        let qkv = rng.tensor(3 * dim, h, w);
        let want = math::channel_attention(&Engine::new(), &qkv, 4, &temperature).unwrap();
        let got = run(
            |g, i| {
                g.channel_attention(i[0], 4, &f32_buffer(context(), &temperature))
                    .unwrap()
            },
            &[&qkv],
        );
        check(
            &format!("channel attention dim {dim}, {h}x{w}"),
            &want,
            &got,
            2e-5,
        );
    }
}
