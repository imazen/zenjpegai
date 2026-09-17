//! The numeric contract: the fast layers are bit-identical to `nn::reference` on every SIMD
//! tier this machine has, for every block size, with and without threading.

use zenjpegai::nn::fast::{BTensor, Engine, PackedConv, PackedConvTranspose, Tier};
use zenjpegai::nn::{Conv2d, ConvTranspose2d, reference};
use zenjpegai::tensor::Tensor;

/// Small deterministic generator (xorshift); values in roughly [-2, 2] with a few exact zeros.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        if self.0.is_multiple_of(97) {
            0.0
        } else {
            (u - 0.5) * 4.0
        }
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn engines() -> Vec<Engine> {
    let mut out = Vec::new();
    for tier in Tier::available() {
        out.push(Engine::with(tier, false));
        out.push(Engine::with(tier, true));
    }
    assert!(!out.is_empty());
    out
}

fn assert_bits_eq(what: &str, eng: &Engine, want: &Tensor<f32>, got: &Tensor<f32>) {
    assert_eq!(
        (want.c, want.h, want.w),
        (got.c, got.h, got.w),
        "{what} {eng:?}: shape"
    );
    for (i, (a, b)) in want.data.iter().zip(&got.data).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{what} {eng:?}: element {i}: reference {a:e} vs fast {b:e}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_case(
    rng: &mut Rng,
    in_ch: usize,
    out_ch: usize,
    k: (usize, usize),
    pad: (usize, usize),
    groups: usize,
    bias: bool,
    h: usize,
    w: usize,
) {
    let weight = rng.vec(out_ch * (in_ch / groups) * k.0 * k.1);
    let b = bias.then(|| rng.vec(out_ch));
    let conv = Conv2d::new(in_ch, out_ch, k, 1, pad, groups, weight, b).unwrap();
    let x = Tensor::from_vec(in_ch, h, w, rng.vec(in_ch * h * w)).unwrap();
    let want = reference::conv2d(&conv, &x).unwrap();
    let what = format!("conv {in_ch}->{out_ch} k{k:?} pad{pad:?} g{groups} {h}x{w}");
    for eng in engines() {
        let v = eng.tier.block();
        let packed = PackedConv::new(&conv, v, [0; 4]).unwrap();
        let got = packed
            .forward(&eng, &BTensor::from_planar(&x, v).unwrap())
            .unwrap()
            .to_planar()
            .unwrap();
        assert_bits_eq(&what, &eng, &want, &got);
    }
}

#[test]
fn conv2d_bit_identical() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    // widths chosen to hit the 28 / 12 / 8 / 4 / 1 position blocks and their tails
    for &(h, w) in &[(1, 1), (3, 5), (5, 12), (4, 29), (7, 63), (2, 131)] {
        conv_case(&mut rng, 32, 48, (3, 3), (1, 1), 1, true, h, w);
        conv_case(&mut rng, 16, 16, (1, 1), (0, 0), 1, false, h, w);
    }
    // channel counts that do not fill a block, on both sides
    conv_case(&mut rng, 5, 7, (3, 3), (1, 1), 1, true, 6, 9);
    conv_case(&mut rng, 96, 16, (1, 1), (0, 0), 1, false, 5, 17);
    conv_case(&mut rng, 3, 1, (3, 3), (1, 1), 1, true, 4, 4);
    // grouped: ResAU (16 per group) and MCM (32 per group)
    conv_case(&mut rng, 64, 64, (3, 3), (1, 1), 4, false, 9, 14);
    conv_case(&mut rng, 160, 160, (3, 3), (1, 1), 5, true, 6, 7);
    // rectangular and even kernels
    conv_case(&mut rng, 16, 16, (1, 3), (0, 1), 1, true, 5, 9);
    conv_case(&mut rng, 16, 16, (3, 1), (1, 0), 1, true, 5, 9);
    conv_case(&mut rng, 16, 32, (2, 2), (0, 0), 1, false, 6, 7);
}

#[test]
fn conv2d_extra_padding_matches_explicit_padding() {
    // The SOP upsampler pads one zero row/column at the bottom/right before a 2x2 convolution.
    let mut rng = Rng(7);
    let (in_ch, out_ch, h, w) = (16, 32, 5, 6);
    let conv = Conv2d::new(
        in_ch,
        out_ch,
        (2, 2),
        1,
        (0, 0),
        1,
        rng.vec(out_ch * in_ch * 4),
        None,
    )
    .unwrap();
    let x = Tensor::from_vec(in_ch, h, w, rng.vec(in_ch * h * w)).unwrap();
    let mut xp = Tensor::<f32>::zeros(in_ch, h + 1, w + 1).unwrap();
    for c in 0..in_ch {
        for y in 0..h {
            for xx in 0..w {
                xp.data[(c * (h + 1) + y) * (w + 1) + xx] = x.at(c, y, xx);
            }
        }
    }
    let want = reference::conv2d(&conv, &xp).unwrap();
    for eng in engines() {
        let v = eng.tier.block();
        let packed = PackedConv::new(&conv, v, [0, 0, 1, 1]).unwrap();
        let got = packed
            .forward(&eng, &BTensor::from_planar(&x, v).unwrap())
            .unwrap()
            .to_planar()
            .unwrap();
        assert_bits_eq("conv2x2 + bottom/right pad", &eng, &want, &got);
    }
}

#[test]
fn conv_transpose2d_bit_identical() {
    let mut rng = Rng(0xD1B54A32D192ED03);
    for &(k, pad, out_pad) in &[(4usize, 1usize, 0usize), (3, 1, 1)] {
        for &(in_ch, out_ch, h, w) in &[
            (16, 16, 1, 1),
            (32, 16, 3, 5),
            (24, 40, 4, 15),
            (160, 64, 2, 31),
            (5, 3, 6, 4),
        ] {
            let weight = rng.vec(in_ch * out_ch * k * k);
            let conv = ConvTranspose2d::new(
                in_ch,
                out_ch,
                k,
                2,
                pad,
                out_pad,
                weight,
                Some(rng.vec(out_ch)),
            )
            .unwrap();
            let x = Tensor::from_vec(in_ch, h, w, rng.vec(in_ch * h * w)).unwrap();
            let want = reference::conv_transpose2d(&conv, &x).unwrap();
            let what = format!("convT {in_ch}->{out_ch} k{k} p{pad} op{out_pad} {h}x{w}");
            for eng in engines() {
                let v = eng.tier.block();
                let packed = PackedConvTranspose::new(&conv, v).unwrap();
                let got = packed
                    .forward(&eng, &BTensor::from_planar(&x, v).unwrap())
                    .unwrap()
                    .to_planar()
                    .unwrap();
                assert_bits_eq(&what, &eng, &want, &got);
            }
        }
    }
}

#[test]
fn blocked_tensor_roundtrip() {
    let mut rng = Rng(3);
    for v in [8, 16] {
        for c in [1, 7, 8, 9, 16, 17, 40] {
            let x = Tensor::from_vec(c, 3, 5, rng.vec(c * 15)).unwrap();
            let b = BTensor::from_planar(&x, v).unwrap();
            assert_eq!(b.to_planar().unwrap(), x);
            assert_eq!(
                b.pad(1, 2, 3, 4)
                    .unwrap()
                    .crop(4, 7)
                    .unwrap()
                    .to_planar()
                    .unwrap()
                    .c,
                c
            );
        }
    }
}

/// The int8 convolution of the hyper-scale decoder: every tier must equal plain wrapping-i32
/// loops exactly, including odd channel counts, out-of-range inputs (clamped) and accumulator
/// wrap-around (huge biases).
#[test]
fn int_conv_matches_scalar_loops() {
    use zenjpegai::nn::fast::PackedIntConv;
    use zenjpegai::tensor::Tensor;

    let mut r = Rng(0x1234_5678_9abc_def1);
    let mut next = move || {
        r.0 ^= r.0 << 13;
        r.0 ^= r.0 >> 7;
        r.0 ^= r.0 << 17;
        (r.0 >> 33) as i64
    };
    for &(out_ch, in_ch, k, h, w) in &[
        (16usize, 16usize, 1usize, 5usize, 7usize),
        (19, 7, 3, 6, 33),
        (40, 10, 3, 1, 1),
        (33, 32, 1, 9, 14),
        (8, 5, 3, 3, 47),
    ] {
        let weight: Vec<i8> = (0..out_ch * in_ch * k * k)
            .map(|_| (next() % 256 - 128) as i8)
            .collect();
        let bias: Vec<i32> = (0..out_ch)
            .map(|i| {
                if i % 3 == 0 {
                    i32::MAX - (next() % 1000) as i32
                } else {
                    (next() % 100_000) as i32 - 50_000
                }
            })
            .collect();
        let shift: Vec<u8> = (0..out_ch).map(|_| (next() % 12) as u8).collect();
        let x = Tensor::from_vec(
            in_ch,
            h,
            w,
            (0..in_ch * h * w)
                .map(|_| (next() % 700 - 350) as i32)
                .collect(),
        )
        .unwrap();

        for relu in [false, true] {
            let pad = k / 2;
            let mut want = Tensor::<i32>::zeros(out_ch, h, w).unwrap();
            for o in 0..out_ch {
                for y in 0..h {
                    for xo in 0..w {
                        let mut acc = bias[o];
                        for i in 0..in_ch {
                            for ky in 0..k {
                                for kx in 0..k {
                                    let (sy, sx) = (y + ky, xo + kx);
                                    if sy < pad || sx < pad || sy - pad >= h || sx - pad >= w {
                                        continue;
                                    }
                                    let v = x.at(i, sy - pad, sx - pad).clamp(-128, 127);
                                    let wv = weight[((o * in_ch + i) * k + ky) * k + kx] as i32;
                                    acc = acc.wrapping_add(v * wv);
                                }
                            }
                        }
                        let v = acc >> shift[o];
                        want.plane_mut(o)[y * w + xo] = if relu { v.max(0) } else { v };
                    }
                }
            }
            for eng in engines() {
                let conv =
                    PackedIntConv::new(out_ch, in_ch, k, &weight, &bias, &shift, eng.tier.block())
                        .unwrap();
                let got = conv.forward(&eng, &x, relu).unwrap();
                assert_eq!(
                    got.data, want.data,
                    "{eng:?} {out_ch}x{in_ch} k{k} {h}x{w} relu={relu}"
                );
            }
        }
    }
}
