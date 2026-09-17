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
