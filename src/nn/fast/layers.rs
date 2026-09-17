//! Layers as the networks use them: a packed fast path with a reference fallback, plus the
//! elementwise operations on blocked tensors.

use archmage::prelude::*;

use super::Engine;
use super::conv::{PackedConv, PackedConvTranspose};
use super::tensor::BTensor;
use crate::error::{Error, Result};
use crate::nn::{Conv2d, ConvTranspose2d, reference};
use crate::tensor::Tensor;

/// A convolution ready to run on one engine's block size.
#[derive(Clone, Debug)]
pub struct ConvLayer {
    /// Kept for layers the fast path does not cover (strided convolutions, large kernels).
    fallback: Option<(Conv2d, [usize; 4])>,
    packed: Option<PackedConv>,
    v: usize,
}

impl ConvLayer {
    pub fn new(conv: Conv2d, eng: &Engine) -> Result<Self> {
        Self::with_extra_pad(conv, eng, [0; 4])
    }

    /// `extra_pad` is `[top, left, bottom, right]` zero padding on top of the layer's own.
    pub fn with_extra_pad(conv: Conv2d, eng: &Engine, extra_pad: [usize; 4]) -> Result<Self> {
        let v = eng.tier.block();
        match PackedConv::new(&conv, v, extra_pad) {
            Ok(p) => Ok(Self {
                fallback: None,
                packed: Some(p),
                v,
            }),
            Err(Error::Unsupported(_)) => Ok(Self {
                fallback: Some((conv, extra_pad)),
                packed: None,
                v,
            }),
            Err(e) => Err(e),
        }
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        if let Some(p) = &self.packed {
            return p.forward(eng, x);
        }
        let (conv, extra) = self
            .fallback
            .as_ref()
            .ok_or(Error::InvalidArgument("conv layer without weights"))?;
        let planar = x.to_planar()?;
        let planar = if *extra == [0; 4] {
            planar
        } else {
            let (h, w) = (
                planar.h + extra[0] + extra[2],
                planar.w + extra[1] + extra[3],
            );
            let mut p = Tensor::<f32>::zeros(planar.c, h, w)?;
            for c in 0..planar.c {
                for y in 0..planar.h {
                    let src = &planar.data[(c * planar.h + y) * planar.w..][..planar.w];
                    p.data[(c * h + y + extra[0]) * w + extra[1]..][..planar.w]
                        .copy_from_slice(src);
                }
            }
            p
        };
        BTensor::from_planar(&reference::conv2d(conv, &planar)?, self.v)
    }
}

/// A transposed convolution ready to run on one engine's block size.
#[derive(Clone, Debug)]
pub struct ConvTransposeLayer {
    fallback: Option<ConvTranspose2d>,
    packed: Option<PackedConvTranspose>,
    v: usize,
}

impl ConvTransposeLayer {
    pub fn new(conv: ConvTranspose2d, eng: &Engine) -> Result<Self> {
        let v = eng.tier.block();
        match PackedConvTranspose::new(&conv, v) {
            Ok(p) => Ok(Self {
                fallback: None,
                packed: Some(p),
                v,
            }),
            Err(Error::Unsupported(_)) => Ok(Self {
                fallback: Some(conv),
                packed: None,
                v,
            }),
            Err(e) => Err(e),
        }
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        if let Some(p) = &self.packed {
            return p.forward(eng, x);
        }
        let conv = self
            .fallback
            .as_ref()
            .ok_or(Error::InvalidArgument("conv layer without weights"))?;
        BTensor::from_planar(&reference::conv_transpose2d(conv, &x.to_planar()?)?, self.v)
    }
}

#[autoversion]
fn relu_slice(data: &mut [f32]) {
    for v in data {
        *v = v.max(0.0);
    }
}

#[autoversion]
fn relu6_slice(data: &mut [f32]) {
    for v in data {
        *v = v.clamp(0.0, 6.0);
    }
}

#[autoversion]
fn add_slice(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// `x * (1 + mask)`: an add, then a multiply. Two IEEE operations, never fused.
#[autoversion]
fn gate_slice(x: &mut [f32], mask: &[f32]) {
    for (v, &m) in x.iter_mut().zip(mask) {
        *v *= 1.0 + m;
    }
}

/// `max(x, 0)` in place.
pub fn relu(x: &mut BTensor) {
    relu_slice(&mut x.data);
}

/// `clamp(x, 0, 6)` in place.
pub fn relu6(x: &mut BTensor) {
    relu6_slice(&mut x.data);
}

/// `dst += src`.
pub fn add_assign(dst: &mut BTensor, src: &BTensor) -> Result<()> {
    if (dst.c, dst.h, dst.w, dst.v) != (src.c, src.h, src.w, src.v) {
        return Err(Error::InvalidArgument("add: shapes differ"));
    }
    add_slice(&mut dst.data, &src.data);
    Ok(())
}

/// `x *= 1 + mask` (the ResAU gate).
pub fn gate(x: &mut BTensor, mask: &BTensor) -> Result<()> {
    if (x.c, x.h, x.w, x.v) != (mask.c, mask.h, mask.w, mask.v) {
        return Err(Error::InvalidArgument("gate: shapes differ"));
    }
    gate_slice(&mut x.data, &mask.data);
    Ok(())
}

/// `nn.PixelShuffle(r)` from a blocked tensor straight to planes: `[c * r * r, h, w]` →
/// `[c, h * r, w * r]`.
pub fn pixel_shuffle_to_planar(x: &BTensor, r: usize) -> Result<Tensor<f32>> {
    if r == 0 || !x.c.is_multiple_of(r * r) {
        return Err(Error::InvalidArgument(
            "pixel_shuffle: channels not divisible by r^2",
        ));
    }
    let (c, h, w, v) = (x.c / (r * r), x.h, x.w, x.v);
    let mut out = Tensor::<f32>::zeros(c, h * r, w * r)?;
    for oc in 0..c {
        for y in 0..h {
            for dy in 0..r {
                let dst = &mut out.data[(oc * h * r + y * r + dy) * w * r..][..w * r];
                for dx in 0..r {
                    let ch = oc * r * r + dy * r + dx;
                    let (b, lane) = (ch / v, ch % v);
                    let src = &x.data[(b * h + y) * w * v..][..w * v];
                    for (xi, s) in src.chunks_exact(v).enumerate() {
                        dst[xi * r + dx] = s[lane];
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `nn.PixelShuffle(r)` staying in blocked layout.
pub fn pixel_shuffle(x: &BTensor, r: usize) -> Result<BTensor> {
    BTensor::from_planar(&pixel_shuffle_to_planar(x, r)?, x.v)
}
