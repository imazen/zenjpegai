//! Neural-network layers.
//!
//! # Numeric contract
//!
//! The reference runs these networks in PyTorch float32. PyTorch's convolution kernels pick
//! their own summation order, so bit-exact agreement with it is not achievable; agreement is
//! measured instead (see `PORTING.md`). What this crate does guarantee is agreement with
//! *itself*: every implementation of a layer, on every SIMD tier and thread count, produces the
//! same bits as the plain loops in [`reference`]. That is possible because the contract fixes the
//! floating-point evaluation order of each output element:
//!
//! * convolution: `acc = bias; for ic { for ky { for kx { acc = fma(w, x, acc) } } }` over a
//!   zero-padded input (padding taps are accumulated like any other);
//! * transposed convolution: the same, over the taps whose stride parity reaches the output
//!   element, `ky` and `kx` ascending;
//! * everything else is elementwise IEEE arithmetic, written out op by op.
//!
//! Vectorising across output channels or positions never changes the order *within* an element,
//! so kernels are free to do that. They may not reassociate, and they may not replace the fused
//! multiply-add by a separate multiply and add.

pub mod reference;

use alloc::vec::Vec;

use crate::error::{Error, Result};

/// 2-D convolution parameters, stride 1 or 2, zero padding, optional groups.
/// Weight layout is PyTorch's: `[out_ch][in_ch / groups][kh][kw]`.
#[derive(Clone, Debug)]
pub struct Conv2d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub groups: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        (kh, kw): (usize, usize),
        stride: usize,
        (pad_h, pad_w): (usize, usize),
        groups: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if groups == 0
            || !in_ch.is_multiple_of(groups)
            || !out_ch.is_multiple_of(groups)
            || stride == 0
            || kh == 0
            || kw == 0
        {
            return Err(Error::Model(
                "conv2d: inconsistent channel/group/kernel configuration".into(),
            ));
        }
        if weight.len() != out_ch * (in_ch / groups) * kh * kw {
            return Err(Error::Model(
                "conv2d: weight size does not match its shape".into(),
            ));
        }
        if bias.as_ref().is_some_and(|b| b.len() != out_ch) {
            return Err(Error::Model(
                "conv2d: bias size does not match out_ch".into(),
            ));
        }
        Ok(Self {
            in_ch,
            out_ch,
            kh,
            kw,
            stride,
            pad_h,
            pad_w,
            groups,
            weight,
            bias,
        })
    }

    /// Output spatial size for an input of `h x w`.
    pub fn out_size(&self, h: usize, w: usize) -> (usize, usize) {
        (
            (h + 2 * self.pad_h - self.kh) / self.stride + 1,
            (w + 2 * self.pad_w - self.kw) / self.stride + 1,
        )
    }
}

/// Transposed convolution, square kernel, no groups, no dilation.
/// Weight layout is PyTorch's: `[in_ch][out_ch][k][k]`.
#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub out_pad: usize,
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl ConvTranspose2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        out_pad: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if stride == 0 || k == 0 || pad >= k || out_pad >= stride {
            return Err(Error::Model(
                "conv_transpose2d: unsupported geometry".into(),
            ));
        }
        if weight.len() != in_ch * out_ch * k * k {
            return Err(Error::Model(
                "conv_transpose2d: weight size does not match its shape".into(),
            ));
        }
        if bias.as_ref().is_some_and(|b| b.len() != out_ch) {
            return Err(Error::Model(
                "conv_transpose2d: bias size does not match out_ch".into(),
            ));
        }
        Ok(Self {
            in_ch,
            out_ch,
            k,
            stride,
            pad,
            out_pad,
            weight,
            bias,
        })
    }

    /// Output spatial size: `(n - 1) * stride - 2 * pad + k + out_pad`.
    pub fn out_size(&self, h: usize, w: usize) -> (usize, usize) {
        let f = |n: usize| (n - 1) * self.stride + self.k + self.out_pad - 2 * self.pad;
        (f(h), f(w))
    }
}
