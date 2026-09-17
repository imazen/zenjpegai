//! Layer parameters packed into the layout the kernels walk (see [`crate::kernels`]).

use wgpu::util::DeviceExt;
use zenjpegai::nn::{Conv2d, ConvTranspose2d};

use crate::context::GpuContext;
use crate::error::{GpuError, Result};

/// Planar `[C, H, W]` to HWC4 (flat `f32`, 4 per block, zero lanes past `C`).
pub fn pack_hwc4(planar: &[f32], c: usize, h: usize, w: usize) -> Result<Vec<f32>> {
    if planar.len() != c * h * w {
        return Err(GpuError::Shape(
            "planar data does not match its shape".into(),
        ));
    }
    let c4 = c.div_ceil(4);
    let mut out = vec![0.0f32; h * w * c4 * 4];
    let hw = h * w;
    for ch in 0..c {
        let plane = &planar[ch * hw..][..hw];
        for (px, &v) in plane.iter().enumerate() {
            out[px * c4 * 4 + ch] = v;
        }
    }
    Ok(out)
}

/// HWC4 back to planar `[C, H, W]`.
pub fn unpack_hwc4(packed: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let c4 = c.div_ceil(4);
    let hw = h * w;
    let mut out = vec![0.0f32; c * hw];
    for ch in 0..c {
        for px in 0..hw {
            out[ch * hw + px] = packed[px * c4 * 4 + ch];
        }
    }
    out
}

fn storage(ctx: &GpuContext, label: &str, data: &[f32]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
}

/// Largest of 4, 2, 1 output blocks per invocation that divides `blocks`.
fn pick_ob(blocks: usize) -> usize {
    if blocks.is_multiple_of(4) {
        4
    } else if blocks.is_multiple_of(2) {
        2
    } else {
        1
    }
}

fn bias_blocks(bias: Option<&[f32]>, oc4: usize) -> Vec<f32> {
    let mut b = vec![0.0f32; oc4 * 4];
    if let Some(src) = bias {
        b[..src.len()].copy_from_slice(src);
    }
    b
}

/// A convolution on the GPU.
#[derive(Clone, Debug)]
pub struct GpuConv {
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: (usize, usize),
    pub stride: usize,
    pub pad: (usize, usize),
    pub(crate) icg4: usize,
    pub(crate) ocg4: usize,
    pub(crate) ob: usize,
    pub(crate) weight: wgpu::Buffer,
    pub(crate) bias: wgpu::Buffer,
}

impl GpuConv {
    /// Groups must be whole blocks: with `groups > 1`, channels per group are multiples of 4.
    /// (Depthwise layers go through [`GpuDepthwise`].)
    pub fn new(ctx: &GpuContext, c: &Conv2d) -> Result<Self> {
        let (icg, ocg) = (c.in_ch / c.groups, c.out_ch / c.groups);
        if c.groups > 1 && (icg % 4 != 0 || ocg % 4 != 0) {
            return Err(GpuError::Shape(format!(
                "grouped convolution with {icg} -> {ocg} channels per group: not whole blocks"
            )));
        }
        let (icg4, ocg4, oc4) = (icg.div_ceil(4), ocg.div_ceil(4), c.out_ch.div_ceil(4));
        let ob = pick_ob(ocg4);
        let (kh, kw) = (c.kh, c.kw);
        // [out group][ky][kx][in block][block in out group], each a column-major mat4x4 whose
        // column j / row i is the weight from input lane j to output lane i.
        let mut w = vec![0.0f32; oc4 * kh * kw * icg4 * 16];
        for oc in 0..c.out_ch {
            let (og, b, row) = (oc / 4 / ob, oc / 4 % ob, oc % 4);
            for ic in 0..icg {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let m = ((((og * kh + ky) * kw + kx) * icg4) + ic / 4) * ob + b;
                        w[m * 16 + (ic % 4) * 4 + row] =
                            c.weight[((oc * icg + ic) * kh + ky) * kw + kx];
                    }
                }
            }
        }
        Ok(Self {
            in_ch: c.in_ch,
            out_ch: c.out_ch,
            k: (kh, kw),
            stride: c.stride,
            pad: (c.pad_h, c.pad_w),
            icg4,
            ocg4,
            ob,
            weight: storage(ctx, "conv weight", &w),
            bias: storage(ctx, "conv bias", &bias_blocks(c.bias.as_deref(), oc4)),
        })
    }
}

/// A transposed convolution on the GPU.
#[derive(Clone, Debug)]
pub struct GpuConvTranspose {
    pub in_ch: usize,
    pub out_ch: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub out_pad: usize,
    pub(crate) ob: usize,
    pub(crate) weight: wgpu::Buffer,
    pub(crate) bias: wgpu::Buffer,
}

impl GpuConvTranspose {
    pub fn new(ctx: &GpuContext, c: &ConvTranspose2d) -> Result<Self> {
        let (ic4, oc4) = (c.in_ch.div_ceil(4), c.out_ch.div_ceil(4));
        let ob = pick_ob(oc4);
        let k = c.k;
        let mut w = vec![0.0f32; oc4 * k * k * ic4 * 16];
        for ic in 0..c.in_ch {
            for oc in 0..c.out_ch {
                let (og, b, row) = (oc / 4 / ob, oc / 4 % ob, oc % 4);
                for ky in 0..k {
                    for kx in 0..k {
                        let m = ((((og * k + ky) * k + kx) * ic4) + ic / 4) * ob + b;
                        w[m * 16 + (ic % 4) * 4 + row] =
                            c.weight[((ic * c.out_ch + oc) * k + ky) * k + kx];
                    }
                }
            }
        }
        Ok(Self {
            in_ch: c.in_ch,
            out_ch: c.out_ch,
            k,
            stride: c.stride,
            pad: c.pad,
            out_pad: c.out_pad,
            ob,
            weight: storage(ctx, "convt weight", &w),
            bias: storage(ctx, "convt bias", &bias_blocks(c.bias.as_deref(), oc4)),
        })
    }
}

/// Depthwise 3x3 convolution (stride 1, padding 1, no bias).
#[derive(Clone, Debug)]
pub struct GpuDepthwise {
    pub ch: usize,
    pub(crate) weight: wgpu::Buffer,
}

impl GpuDepthwise {
    pub fn new(ctx: &GpuContext, c: &Conv2d) -> Result<Self> {
        if c.groups != c.in_ch
            || c.out_ch != c.in_ch
            || (c.kh, c.kw, c.stride, c.pad_h, c.pad_w) != (3, 3, 1, 1, 1)
            || c.bias.is_some()
        {
            return Err(GpuError::Shape(
                "not a bias-free depthwise 3x3 convolution".into(),
            ));
        }
        let c4 = c.in_ch.div_ceil(4);
        let mut w = vec![0.0f32; c4 * 9 * 4];
        for ch in 0..c.in_ch {
            for k in 0..9 {
                w[((ch / 4) * 9 + k) * 4 + ch % 4] = c.weight[ch * 9 + k];
            }
        }
        Ok(Self {
            ch: c.in_ch,
            weight: storage(ctx, "depthwise weight", &w),
        })
    }
}

/// Layer norm parameters: weight blocks, then bias blocks.
#[derive(Clone, Debug)]
pub struct GpuLayerNorm {
    pub ch: usize,
    pub(crate) wb: wgpu::Buffer,
}

impl GpuLayerNorm {
    pub fn new(ctx: &GpuContext, weight: &[f32], bias: &[f32]) -> Result<Self> {
        if weight.len() != bias.len() || weight.is_empty() {
            return Err(GpuError::Shape("layer norm parameter lengths".into()));
        }
        let c4 = weight.len().div_ceil(4);
        let mut wb = vec![0.0f32; c4 * 8];
        wb[..weight.len()].copy_from_slice(weight);
        wb[c4 * 4..][..bias.len()].copy_from_slice(bias);
        Ok(Self {
            ch: weight.len(),
            wb: storage(ctx, "layer norm", &wb),
        })
    }
}

/// A small read-only `f32` storage buffer (attention temperatures).
pub fn f32_buffer(ctx: &GpuContext, data: &[f32]) -> wgpu::Buffer {
    let mut padded = data.to_vec();
    padded.resize(data.len().div_ceil(4) * 4, 0.0);
    storage(ctx, "f32 parameters", &padded)
}
