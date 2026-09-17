//! Layer-level agreement with PyTorch on tiny random tensors
//! (`scripts/ref_vectors/gen_nn_vectors.py`, torch 1.10.2). PyTorch sums in its own order, so the
//! comparison is tolerance-based; layer *definitions* (padding, stride, groups, weight layout,
//! transposed-convolution geometry, pixel-shuffle order) are what these tests pin down.

use std::collections::HashMap;

use zenjpegai::nn::{Conv2d, ConvTranspose2d, reference};
use zenjpegai::tensor::Tensor;

struct Arr {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn load(name: &str) -> HashMap<String, Arr> {
    let path = format!("{}/tests/vectors/nn/{name}.bin", env!("CARGO_MANIFEST_DIR"));
    let b = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&b[..4], b"ZJTB");
    let mut pos = 4;
    let u32_at = |pos: &mut usize| {
        let v = u32::from_le_bytes(b[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v as usize
    };
    let count = u32_at(&mut pos);
    let mut out = HashMap::new();
    for _ in 0..count {
        let nlen = u16::from_le_bytes(b[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8(b[pos..pos + nlen].to_vec()).unwrap();
        pos += nlen;
        let ndim = b[pos] as usize;
        pos += 1;
        let shape: Vec<usize> = (0..ndim).map(|_| u32_at(&mut pos)).collect();
        let n: usize = shape.iter().product();
        let data = b[pos..pos + 4 * n]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        pos += 4 * n;
        out.insert(name, Arr { shape, data });
    }
    assert_eq!(pos, b.len());
    out
}

fn tensor(a: &Arr) -> Tensor<f32> {
    assert_eq!(a.shape.len(), 3);
    Tensor::from_vec(a.shape[0], a.shape[1], a.shape[2], a.data.clone()).unwrap()
}

fn assert_close(name: &str, got: &Tensor<f32>, want: &Arr) {
    assert_eq!([got.c, got.h, got.w], want.shape[..], "{name}: shape");
    let mut worst = 0f32;
    for (g, w) in got.data.iter().zip(&want.data) {
        worst = worst.max((g - w).abs() / w.abs().max(1.0));
    }
    assert!(worst < 2e-6, "{name}: max relative error {worst:e}");
}

fn conv_case(name: &str, stride: usize, pad: (usize, usize), groups: usize) {
    let t = load(name);
    let w = &t["w"];
    let conv = Conv2d::new(
        w.shape[1] * groups,
        w.shape[0],
        (w.shape[2], w.shape[3]),
        stride,
        pad,
        groups,
        w.data.clone(),
        t.get("b").map(|b| b.data.clone()),
    )
    .unwrap();
    let y = reference::conv2d(&conv, &tensor(&t["x"])).unwrap();
    assert_close(name, &y, &t["y"]);
}

fn convt_case(name: &str, stride: usize, pad: usize, out_pad: usize) {
    let t = load(name);
    let w = &t["w"];
    let conv = ConvTranspose2d::new(
        w.shape[0],
        w.shape[1],
        w.shape[2],
        stride,
        pad,
        out_pad,
        w.data.clone(),
        t.get("b").map(|b| b.data.clone()),
    )
    .unwrap();
    let y = reference::conv_transpose2d(&conv, &tensor(&t["x"])).unwrap();
    assert_close(name, &y, &t["y"]);
}

#[test]
fn conv2d_variants() {
    conv_case("conv3x3_s1_p1_bias", 1, (1, 1), 1);
    conv_case("conv3x3_s1_p1_g4_nobias", 1, (1, 1), 4);
    conv_case("conv3x3_depthwise", 1, (1, 1), 12);
    conv_case("conv1x1_nobias", 1, (0, 0), 1);
    conv_case("conv3x3_s2_p1_bias", 2, (1, 1), 1);
    conv_case("conv2x2_s1_p0", 1, (0, 0), 1);
    conv_case("conv1x3_p01_bias", 1, (0, 1), 1);
    conv_case("conv3x1_p10_bias", 1, (1, 0), 1);
}

#[test]
fn conv_transpose2d_variants() {
    convt_case("convt4x4_s2_p1_bias", 2, 1, 0);
    convt_case("convt3x3_s2_p1_op1_bias", 2, 1, 1);
}

#[test]
fn pixel_shuffle_order() {
    let t = load("pixel_shuffle");
    let x = tensor(&t["x"]);
    for (r, key) in [(2, "y2"), (4, "y4")] {
        let y = reference::pixel_shuffle(&x, r).unwrap();
        assert_eq!([y.c, y.h, y.w], t[key].shape[..]);
        assert_eq!(
            y.data, t[key].data,
            "pixel_shuffle r={r} is a pure permutation"
        );
    }
}
