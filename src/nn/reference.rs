//! Plain-loop implementations of every layer: the definition of the numeric contract
//! (see [`super`]) and the scalar dispatch tier.
//!
//! The loops are compiled once per CPU tier by `#[autoversion]`, so `f32::mul_add` is a hardware
//! FMA wherever one exists; without one it is libm's exact `fma`, which is slow but yields the
//! same bits. On wasm32 the multiply-add is unfused by policy ([`super::fmadd`]).

use alloc::vec::Vec;

use archmage::prelude::*;

use super::{Conv2d, ConvTranspose2d, fmadd};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Zero-padded copy of `x`: `[c, h + 2 * pad_h, w + 2 * pad_w]`.
fn zero_pad(x: &Tensor<f32>, pad_h: usize, pad_w: usize) -> Result<Tensor<f32>> {
    let (ph, pw) = (x.h + 2 * pad_h, x.w + 2 * pad_w);
    let mut out = Tensor::<f32>::zeros(x.c, ph, pw)?;
    for c in 0..x.c {
        for y in 0..x.h {
            let src = &x.data[(c * x.h + y) * x.w..][..x.w];
            out.data[(c * ph + y + pad_h) * pw + pad_w..][..x.w].copy_from_slice(src);
        }
    }
    Ok(out)
}

/// Geometry of one convolution output plane over an already padded input.
#[derive(Clone, Copy)]
struct ConvGeom {
    /// Padded input plane size.
    ph: usize,
    pw: usize,
    /// First input channel of the group and channels per group.
    ic0: usize,
    icg: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    oh: usize,
    ow: usize,
}

/// One output plane of a convolution over an already padded input.
#[autoversion]
fn conv_plane(
    xp: &[f32],
    g: ConvGeom,
    weight: &[f32],
    bias: f32,
    out: &mut [f32],
    acc: &mut [f32],
) {
    let ConvGeom {
        ph,
        pw,
        ic0,
        icg,
        kh,
        kw,
        stride,
        oh,
        ow,
    } = g;
    for oy in 0..oh {
        let acc = &mut acc[..ow];
        acc.fill(bias);
        for i in 0..icg {
            let plane = &xp[(ic0 + i) * ph * pw..][..ph * pw];
            for ky in 0..kh {
                let row = &plane[(oy * stride + ky) * pw..][..pw];
                for kx in 0..kw {
                    let wv = weight[(i * kh + ky) * kw + kx];
                    if stride == 1 {
                        for (a, &v) in acc.iter_mut().zip(&row[kx..kx + ow]) {
                            *a = fmadd(wv, v, *a);
                        }
                    } else {
                        for (ox, a) in acc.iter_mut().enumerate() {
                            *a = fmadd(wv, row[ox * stride + kx], *a);
                        }
                    }
                }
            }
        }
        out[oy * ow..][..ow].copy_from_slice(acc);
    }
}

/// Convolution (the contract's definition).
pub fn conv2d(conv: &Conv2d, x: &Tensor<f32>) -> Result<Tensor<f32>> {
    if x.c != conv.in_ch {
        return Err(Error::InvalidArgument(
            "conv2d: input channel count mismatch",
        ));
    }
    if x.h + 2 * conv.pad_h < conv.kh || x.w + 2 * conv.pad_w < conv.kw {
        return Err(Error::InvalidArgument(
            "conv2d: input smaller than the kernel",
        ));
    }
    let (oh, ow) = conv.out_size(x.h, x.w);
    let xp = zero_pad(x, conv.pad_h, conv.pad_w)?;
    let (icg, ocg) = (conv.in_ch / conv.groups, conv.out_ch / conv.groups);
    let wlen = icg * conv.kh * conv.kw;
    let mut out = Tensor::<f32>::zeros(conv.out_ch, oh, ow)?;
    let mut acc: Vec<f32> = alloc::vec![0.0; ow];
    for oc in 0..conv.out_ch {
        let group = oc / ocg;
        let geom = ConvGeom {
            ph: xp.h,
            pw: xp.w,
            ic0: group * icg,
            icg,
            kh: conv.kh,
            kw: conv.kw,
            stride: conv.stride,
            oh,
            ow,
        };
        let bias = conv.bias.as_ref().map_or(0.0, |b| b[oc]);
        conv_plane(
            &xp.data,
            geom,
            &conv.weight[oc * wlen..][..wlen],
            bias,
            out.plane_mut(oc),
            &mut acc,
        );
    }
    Ok(out)
}

/// Geometry of one transposed-convolution output plane.
#[derive(Clone, Copy)]
struct ConvTransposeGeom {
    /// Padded input plane size; the input is zero-padded by `ext` on every side.
    ph: usize,
    pw: usize,
    ext: usize,
    in_ch: usize,
    out_ch: usize,
    /// Output channel being computed.
    oc: usize,
    k: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
}

/// One output plane of a transposed convolution. The input is zero-padded by `ext` on every
/// side, so that every tap of every output position is addressable.
#[autoversion]
fn conv_transpose_plane(
    xp: &[f32],
    g: ConvTransposeGeom,
    weight: &[f32],
    bias: f32,
    out: &mut [f32],
) {
    let ConvTransposeGeom {
        ph,
        pw,
        ext,
        in_ch,
        out_ch,
        oc,
        k,
        stride,
        pad,
        oh,
        ow,
    } = g;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = bias;
            for ic in 0..in_ch {
                let plane = &xp[ic * ph * pw..][..ph * pw];
                let wbase = (ic * out_ch + oc) * k * k;
                for ky in 0..k {
                    // oy = iy * stride - pad + ky
                    let ty = oy + pad + ext * stride - ky;
                    if !ty.is_multiple_of(stride) {
                        continue;
                    }
                    let row = &plane[(ty / stride) * pw..][..pw];
                    for kx in 0..k {
                        let tx = ox + pad + ext * stride - kx;
                        if !tx.is_multiple_of(stride) {
                            continue;
                        }
                        acc = fmadd(weight[wbase + ky * k + kx], row[tx / stride], acc);
                    }
                }
            }
            out[oy * ow + ox] = acc;
        }
    }
}

/// Transposed convolution (the contract's definition).
pub fn conv_transpose2d(conv: &ConvTranspose2d, x: &Tensor<f32>) -> Result<Tensor<f32>> {
    if x.c != conv.in_ch {
        return Err(Error::InvalidArgument(
            "conv_transpose2d: input channel count mismatch",
        ));
    }
    if x.h == 0 || x.w == 0 {
        return Err(Error::InvalidArgument("conv_transpose2d: empty input"));
    }
    let (oh, ow) = conv.out_size(x.h, x.w);
    // Taps reach input indices in [-(k-1)/stride, (o + pad)/stride]; pad generously.
    let ext = conv.k.div_ceil(conv.stride) + 1;
    let xp = zero_pad(x, ext, ext)?;
    let mut out = Tensor::<f32>::zeros(conv.out_ch, oh, ow)?;
    for oc in 0..conv.out_ch {
        let geom = ConvTransposeGeom {
            ph: xp.h,
            pw: xp.w,
            ext,
            in_ch: conv.in_ch,
            out_ch: conv.out_ch,
            oc,
            k: conv.k,
            stride: conv.stride,
            pad: conv.pad,
            oh,
            ow,
        };
        let bias = conv.bias.as_ref().map_or(0.0, |b| b[oc]);
        conv_transpose_plane(&xp.data, geom, &conv.weight, bias, out.plane_mut(oc));
    }
    Ok(out)
}

/// `conv2d`'s input gradient (RDLR VJP): `d_in[ci,y,x] = Σ_co Σ_ky Σ_kx
/// d_out[co,oy,ox] · w[co, ci∈g, ky, kx]` — the 180°-flipped full conv of the
/// output gradient. Bias does not touch the input gradient. Deterministic
/// f32 accumulation in raster order.
pub fn conv2d_backward(
    dout: &Tensor<f32>,
    conv: &Conv2d,
    h: usize,
    w: usize,
) -> Result<Tensor<f32>> {
    if dout.c != conv.out_ch {
        return Err(Error::InvalidArgument(
            "conv2d_backward: output channel count mismatch",
        ));
    }
    let mut din = Tensor::<f32>::zeros(conv.in_ch, h, w)?;
    let (icg, ocg) = (conv.in_ch / conv.groups, conv.out_ch / conv.groups);
    let (oh, ow) = (dout.h, dout.w);
    for co in 0..conv.out_ch {
        let g = co / ocg;
        let wb = &conv.weight[(co * icg) * conv.kh * conv.kw..][..icg * conv.kh * conv.kw];
        for ci in 0..icg {
            let c_in = g * icg + ci;
            let dst = din.plane_mut(c_in);
            for oy in 0..oh {
                for ox in 0..ow {
                    let g_o = dout.plane(co)[oy * ow + ox];
                    if g_o == 0.0 {
                        continue;
                    }
                    for ky in 0..conv.kh {
                        let y = oy * conv.stride + ky;
                        if y < conv.pad_h || y - conv.pad_h >= h {
                            continue;
                        }
                        let iy = y - conv.pad_h;
                        for kx in 0..conv.kw {
                            let x = ox * conv.stride + kx;
                            if x < conv.pad_w || x - conv.pad_w >= w {
                                continue;
                            }
                            dst[iy * w + x - conv.pad_w] +=
                                g_o * wb[ci * conv.kh * conv.kw + ky * conv.kw + kx];
                        }
                    }
                }
            }
        }
    }
    Ok(din)
}

/// `conv_transpose2d`'s input gradient (RDLR VJP): an ordinary stride-`stride`
/// conv of the output gradient with the same kernel — `conv_transpose` scatters
/// `in[y,x]` to `out[s·y + ky - pad, s·x + kx - pad]`, so the backward gathers.
pub fn conv_transpose2d_backward(
    dout: &Tensor<f32>,
    conv: &ConvTranspose2d,
    h: usize,
    w: usize,
) -> Result<Tensor<f32>> {
    if dout.c != conv.out_ch {
        return Err(Error::InvalidArgument(
            "conv_transpose2d_backward: output channel count mismatch",
        ));
    }
    let mut din = Tensor::<f32>::zeros(conv.in_ch, h, w)?;
    let (oh, ow) = (dout.h, dout.w);
    // weight layout [in_ch, out_ch, k, k]: din[ic] sums dout[oc] · w[ic,oc].
    for ic in 0..conv.in_ch {
        let wb =
            &conv.weight[(ic * conv.out_ch) * conv.k * conv.k..][..conv.out_ch * conv.k * conv.k];
        let dst = din.plane_mut(ic);
        for oc in 0..conv.out_ch {
            let d = dout.plane(oc);
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0f32;
                    for ky in 0..conv.k {
                        let oy = y * conv.stride + ky;
                        if oy < conv.pad || oy - conv.pad >= oh {
                            continue;
                        }
                        for kx in 0..conv.k {
                            let ox = x * conv.stride + kx;
                            if ox < conv.pad || ox - conv.pad >= ow {
                                continue;
                            }
                            acc += d[(oy - conv.pad) * ow + ox - conv.pad]
                                * wb[oc * conv.k * conv.k + ky * conv.k + kx];
                        }
                    }
                    dst[y * w + x] += acc;
                }
            }
        }
    }
    Ok(din)
}

/// `max(x, 0)`.
pub fn relu(x: &mut Tensor<f32>) {
    for v in &mut x.data {
        *v = v.max(0.0);
    }
}

/// `min(max(x, 0), 6)`.
pub fn relu6(x: &mut Tensor<f32>) {
    for v in &mut x.data {
        *v = v.clamp(0.0, 6.0);
    }
}

/// `nn.PixelShuffle(r)`: `[c * r * r, h, w]` → `[c, h * r, w * r]`.
pub fn pixel_shuffle(x: &Tensor<f32>, r: usize) -> Result<Tensor<f32>> {
    if r == 0 || !x.c.is_multiple_of(r * r) {
        return Err(Error::InvalidArgument(
            "pixel_shuffle: channels not divisible by r^2",
        ));
    }
    let c = x.c / (r * r);
    let (h, w) = (x.h, x.w);
    let mut out = Tensor::<f32>::zeros(c, h * r, w * r)?;
    for oc in 0..c {
        for dy in 0..r {
            for dx in 0..r {
                let src = x.plane(oc * r * r + dy * r + dx);
                for y in 0..h {
                    let dst = &mut out.data[(oc * h * r + y * r + dy) * w * r..][..w * r];
                    for (xi, &v) in src[y * w..][..w].iter().enumerate() {
                        dst[xi * r + dx] = v;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Concatenate along the channel axis.
pub fn cat(parts: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
    let first = parts
        .first()
        .ok_or(Error::InvalidArgument("cat: no inputs"))?;
    if parts.iter().any(|p| p.h != first.h || p.w != first.w) {
        return Err(Error::InvalidArgument("cat: spatial sizes differ"));
    }
    let mut data = Vec::with_capacity(parts.iter().map(|p| p.data.len()).sum());
    for p in parts {
        data.extend_from_slice(&p.data);
    }
    Tensor::from_vec(parts.iter().map(|p| p.c).sum(), first.h, first.w, data)
}

/// Channels `c0..c1` as a new tensor.
pub fn slice_channels(x: &Tensor<f32>, c0: usize, c1: usize) -> Result<Tensor<f32>> {
    if c0 > c1 || c1 > x.c {
        return Err(Error::InvalidArgument(
            "slice_channels: range out of bounds",
        ));
    }
    let n = x.plane_len();
    Tensor::from_vec(c1 - c0, x.h, x.w, x.data[c0 * n..c1 * n].to_vec())
}

#[cfg(test)]
mod vjp_tests {
    use super::*;

    fn t(c: usize, h: usize, w: usize, seed: usize) -> Tensor<f32> {
        Tensor::from_vec(
            c,
            h,
            w,
            (0..c * h * w)
                .map(|i| ((i + seed) % 11) as f32 / 11.0 - 0.4)
                .collect(),
        )
        .unwrap()
    }
    fn conv2d_test(in_ch: usize, out_ch: usize, k: usize, groups: usize) -> Conv2d {
        let icg = in_ch / groups;
        let weight = (0..out_ch * icg * k * k)
            .map(|i| ((i % 7) as f32 / 7.0) - 0.3)
            .collect();
        Conv2d::new(
            in_ch,
            out_ch,
            (k, k),
            1,
            (k / 2, k / 2),
            groups,
            weight,
            None,
        )
        .unwrap()
    }

    /// `conv2d_backward` as the adjoint of `conv2d`: `⟨d_out, conv(x+εe) − conv(x)⟩/ε ≈ d_in[e]`.
    #[test]
    fn conv2d_backward_is_adjoint() {
        for &(ic, oc, k, g) in &[(4, 4, 3, 1), (8, 8, 3, 4), (4, 8, 1, 1)] {
            let conv = conv2d_test(ic, oc, k, g);
            let x = t(ic, 9, 9, 1);
            let dout = t(oc, 9, 9, 2);
            let din = conv2d_backward(&dout, &conv, 9, 9).unwrap();
            let base = conv2d(&conv, &x).unwrap();
            let e = 1e-3;
            let mut worst = 0.0f32;
            for &pos in &[0usize, ic * 40, ic * 81 - 1] {
                let mut xp = x.clone();
                xp.data[pos] += e;
                let op = conv2d(&conv, &xp).unwrap();
                let fd: f32 = dout
                    .data
                    .iter()
                    .zip(&op.data)
                    .zip(&base.data)
                    .map(|((&d, &a), &b)| d * (a - b))
                    .sum::<f32>()
                    / e;
                let rel = (din.data[pos] - fd).abs() / fd.abs().max(1e-9);
                worst = worst.max(rel);
            }
            assert!(worst < 0.02, "conv2d_backward g{g} rel {worst}");
        }
    }

    /// `conv_transpose2d_backward` adjoint of `conv_transpose2d`.
    #[test]
    fn conv_transpose2d_backward_is_adjoint() {
        let conv = ConvTranspose2d::new(
            4,
            6,
            4,
            2,
            1,
            0,
            (0..4 * 6 * 16)
                .map(|i| (i % 5) as f32 / 5.0 - 0.4)
                .collect(),
            None,
        )
        .unwrap();
        let x = t(4, 7, 7, 3);
        let base = conv_transpose2d(&conv, &x).unwrap();
        let dout = t(6, base.h, base.w, 4);
        let din = conv_transpose2d_backward(&dout, &conv, 7, 7).unwrap();
        let e = 1e-3;
        let mut worst = 0.0f32;
        for &pos in &[0usize, 4 * 24, 4 * 49 - 1] {
            let mut xp = x.clone();
            xp.data[pos] += e;
            let op = conv_transpose2d(&conv, &xp).unwrap();
            let fd: f32 = dout
                .data
                .iter()
                .zip(&op.data)
                .zip(&base.data)
                .map(|((&d, &a), &b)| d * (a - b))
                .sum::<f32>()
                / e;
            let rel = (din.data[pos] - fd).abs() / fd.abs().max(1e-9);
            worst = worst.max(rel);
        }
        assert!(worst < 0.02, "conv_transpose2d_backward rel {worst}");
    }
}
