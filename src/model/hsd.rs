//! Hyper-scale decoder: the integer network that turns `z_hat` into sigma indices.
//!
//! Ports `ref/src/codec/components/autoencoder_hyper/decoder_scale/basic.py`
//! (`HyperScaleDecoder`) and the quantised convolution in
//! `base_layers/conv_quant_layers.py` (`Conv2di`). This network must be bit-exact: its output
//! selects the entropy-coding distribution of every residual symbol.
//!
//! ```text
//! conv1x1 -> ReLU -> conv3x3 -> ReLU -> conv1x1 (C -> 16C) -> PixelShuffle(4) -> crop -> |x|
//!         -> clamp(0, sigma_idx_max)
//! ```
//!
//! Every `Conv2di` clamps its *input* to `[-128, 127]`, accumulates in wrapping `i32` (what
//! PyTorch's CPU int32 convolution does) and arithmetic-shifts the result right by a per-output-
//! channel amount.

use alloc::format;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `Conv2di` weights: `weight[out][in][ky][kx]` int8, `bias[out]` int32, `shift[out]`.
#[derive(Clone, Debug)]
struct QuantConv {
    out_ch: usize,
    in_ch: usize,
    k: usize,
    weight: Vec<i8>,
    bias: Vec<i32>,
    shift: Vec<u8>,
}

impl QuantConv {
    fn load(
        ck: &Checkpoint<'_>,
        prefix: &str,
        out_ch: usize,
        in_ch: usize,
        k: usize,
    ) -> Result<Self> {
        let quantized = ck.bool(&format!("{prefix}.is_quantized"))?;
        if quantized.data != [true] {
            return Err(Error::Model(format!(
                "{prefix}: checkpoint is not quantised (need VM_common_int)"
            )));
        }
        let weight = ck.i8(&format!("{prefix}.weight"))?;
        if weight.shape != [out_ch, in_ch, k, k] {
            return Err(Error::Model(format!(
                "{prefix}.weight: unexpected shape {:?}",
                weight.shape
            )));
        }
        let bias = ck.i32(&format!("{prefix}.bias"))?;
        let shifts = ck.i8(&format!("{prefix}.per_channel_shifts"))?;
        if bias.data.len() != out_ch || shifts.data.len() != out_ch {
            return Err(Error::Model(format!(
                "{prefix}: bias/shift length mismatch"
            )));
        }
        if shifts.data.iter().any(|&s| !(0..32).contains(&s)) {
            return Err(Error::Model(format!(
                "{prefix}: per-channel shift out of range"
            )));
        }
        Ok(Self {
            out_ch,
            in_ch,
            k,
            weight: weight.data,
            bias: bias.data,
            shift: shifts.data.iter().map(|&s| s as u8).collect(),
        })
    }

    /// Stride-1 convolution with zero padding `k / 2`. Input values are clamped to int8 range.
    fn forward(&self, x: &Tensor<i32>, relu: bool) -> Result<Tensor<i32>> {
        debug_assert_eq!(x.c, self.in_ch);
        let (h, w, k) = (x.h, x.w, self.k);
        let pad = k / 2;
        // Clamp once, into a zero-padded copy, so the inner loops carry no bounds logic.
        let (ph, pw) = (h + 2 * pad, w + 2 * pad);
        let mut xp = Tensor::<i32>::zeros(self.in_ch, ph, pw)?;
        for c in 0..self.in_ch {
            for y in 0..h {
                let src = &x.data[(c * h + y) * w..(c * h + y + 1) * w];
                let dst = &mut xp.data[(c * ph + y + pad) * pw + pad..][..w];
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = s.clamp(-128, 127);
                }
            }
        }
        let mut out = Tensor::<i32>::zeros(self.out_ch, h, w)?;
        let mut acc: Vec<i32> = alloc::vec![0; w];
        for o in 0..self.out_ch {
            let shift = self.shift[o] as u32;
            for y in 0..h {
                acc.fill(self.bias[o]);
                for i in 0..self.in_ch {
                    let wbase = (o * self.in_ch + i) * k * k;
                    for ky in 0..k {
                        let row = &xp.data[(i * ph + y + ky) * pw..][..pw];
                        for kx in 0..k {
                            let wv = self.weight[wbase + ky * k + kx] as i32;
                            if wv == 0 {
                                continue;
                            }
                            for (a, &v) in acc.iter_mut().zip(&row[kx..kx + w]) {
                                *a = a.wrapping_add(v.wrapping_mul(wv));
                            }
                        }
                    }
                }
                let dst = &mut out.data[(o * h + y) * w..][..w];
                for (d, &a) in dst.iter_mut().zip(&acc) {
                    let v = a >> shift;
                    *d = if relu { v.max(0) } else { v };
                }
            }
        }
        Ok(out)
    }
}

/// The integer hyper-scale decoder of one component.
#[derive(Clone, Debug)]
pub struct HyperScaleDecoder {
    chs: usize,
    conv1: QuantConv,
    depthwise: QuantConv,
    pointwise: QuantConv,
}

impl HyperScaleDecoder {
    pub fn load(ck: &Checkpoint<'_>, chs: usize) -> Result<Self> {
        Ok(Self {
            chs,
            conv1: QuantConv::load(ck, "hyper_scale_decoder.conv1", chs, chs, 1)?,
            // Named "depthwise" upstream, but it is a full (groups = 1) 3x3 convolution.
            depthwise: QuantConv::load(ck, "hyper_scale_decoder.depthwise", chs, chs, 3)?,
            pointwise: QuantConv::load(ck, "hyper_scale_decoder.pointwise", chs * 16, chs, 1)?,
        })
    }

    /// `z_hat` `[C, hz, wz]` → sigma map `[C, out_h, out_w]` in `[0, sigma_idx_max]`, where
    /// `(out_h, out_w)` is the latent size (the 4x upsampled map is cropped to it).
    pub fn forward(
        &self,
        z_hat: &Tensor<i8>,
        out_h: usize,
        out_w: usize,
        sigma_idx_max: i32,
    ) -> Result<Tensor<i32>> {
        if z_hat.c != self.chs {
            return Err(Error::InvalidArgument(
                "hyper-scale decoder: channel count mismatch",
            ));
        }
        if out_h > z_hat.h * 4 || out_w > z_hat.w * 4 {
            return Err(Error::InvalidArgument(
                "hyper-scale decoder: output larger than 4x input",
            ));
        }
        let x = Tensor::from_vec(
            z_hat.c,
            z_hat.h,
            z_hat.w,
            z_hat.data.iter().map(|&v| v as i32).collect(),
        )?;
        let x = self.conv1.forward(&x, true)?;
        let x = self.depthwise.forward(&x, true)?;
        let x = self.pointwise.forward(&x, false)?;

        // PixelShuffle(4) + crop + abs + clamp in one pass.
        let (hz, wz) = (z_hat.h, z_hat.w);
        let mut out = Tensor::<i32>::zeros(self.chs, out_h, out_w)?;
        for c in 0..self.chs {
            for y in 0..out_h {
                let (sy, dy) = (y / 4, y % 4);
                let dst = &mut out.data[(c * out_h + y) * out_w..][..out_w];
                for (xo, d) in dst.iter_mut().enumerate() {
                    let (sx, dx) = (xo / 4, xo % 4);
                    let v = x.data[((c * 16 + dy * 4 + dx) * hz + sy) * wz + sx];
                    *d = v.wrapping_abs().clamp(0, sigma_idx_max);
                }
            }
        }
        Ok(out)
    }
}
