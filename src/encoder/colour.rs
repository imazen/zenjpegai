//! Encoder-side colour pre-processing: an RGB picture to the two tensors the analysis
//! transforms take.
//!
//! Ports `ref/src/codec/coding_tools/core_models/CCS_SGMM/ccs_sgmm_tool.py::compress` (its first
//! half) with `ref/src/codec/common/{image.py, colorspace.py}`: `to_YUV_`, `convert_range_`,
//! `pad_` and the two `pixel_unshuffle` calls. Every step is plain `f32` arithmetic in the order
//! the reference evaluates it, which makes the result bit-exact against its tensor dumps.

use alloc::vec::Vec;

use crate::decoder::output::RgbImage;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// BT.709 luma weights and the two chroma denominators (`colorspace.py`).
const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;
const KBY: f32 = 1.8556;
const KRY: f32 = 1.5748;

/// Luma and chroma inputs of the analysis transforms.
pub struct AnalysisInput {
    /// `[1, ph, pw]` in `[0, 255]`, replicate-padded to an even size.
    pub luma: Tensor<f32>,
    /// `[12, ph / 2, pw / 2]`: the four luma phases (the "support information"), then U's four
    /// phases, then V's.
    pub chroma: Tensor<f32>,
}

/// `F.pixel_unshuffle(plane, 2)` into `out`'s channels `base .. base + 4`.
fn unshuffle_into(out: &mut Tensor<f32>, base: usize, plane: &[f32], ph: usize, pw: usize) {
    let (ch, cw) = (ph / 2, pw / 2);
    for k in 0..4 {
        let (dy, dx) = (k / 2, k % 2);
        let dst = out.plane_mut(base + k);
        for y in 0..ch {
            let src = &plane[(2 * y + dy) * pw..][..pw];
            for x in 0..cw {
                dst[y * cw + x] = src[2 * x + dx];
            }
        }
    }
}

/// Replicate the last row / column so that the plane has an even size (`Image.pad_`).
fn replicate_pad(src: &[f32], w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    let mut out = alloc::vec![0f32; ph * pw];
    for y in 0..ph {
        let sy = y.min(h - 1);
        let row = &mut out[y * pw..][..pw];
        row[..w].copy_from_slice(&src[sy * w..][..w]);
        row[w..].fill(src[sy * w + w - 1]);
    }
    out
}

/// RGB (8 bit, 4:4:4, BT.709) → the analysis transforms' inputs.
///
/// Only the case the reference's stock configuration produces is ported: an 8-bit RGB source
/// coded at full chroma resolution (`s_ver = s_hor = c_ver = c_hor = 1`).
pub fn preprocess_rgb(rgb: &RgbImage) -> Result<AnalysisInput> {
    if rgb.bit_depth != 8 {
        return Err(Error::Unsupported("encoder: only 8-bit input so far"));
    }
    let (w, h) = (rgb.width, rgb.height);
    if w == 0 || h == 0 || rgb.data.len() != w * h * 3 {
        return Err(Error::InvalidArgument("RGB buffer size does not match"));
    }
    let (mut yp, mut up, mut vp) = (
        alloc::vec![0f32; w * h],
        alloc::vec![0f32; w * h],
        alloc::vec![0f32; w * h],
    );
    for i in 0..w * h {
        // `convert_range_` to [0, 1], BT.709 forward, then `convert_range_` to [0, 255].
        let r = rgb.data[3 * i] as f32 / 255.0;
        let g = rgb.data[3 * i + 1] as f32 / 255.0;
        let b = rgb.data[3 * i + 2] as f32 / 255.0;
        let y = KR * r + KG * g + KB * b;
        yp[i] = y * 255.0;
        up[i] = ((b - y) / KBY + 0.5) * 255.0;
        vp[i] = ((r - y) / KRY + 0.5) * 255.0;
    }
    let (pw, ph) = (w + w % 2, h + h % 2);
    let yp = replicate_pad(&yp, w, h, pw, ph);
    let up = replicate_pad(&up, w, h, pw, ph);
    let vp = replicate_pad(&vp, w, h, pw, ph);

    let luma = Tensor::from_vec(1, ph, pw, yp.clone())?;
    let mut chroma = Tensor::<f32>::zeros(12, ph / 2, pw / 2)?;
    unshuffle_into(&mut chroma, 0, &yp, ph, pw);
    unshuffle_into(&mut chroma, 4, &up, ph, pw);
    unshuffle_into(&mut chroma, 8, &vp, ph, pw);
    Ok(AnalysisInput { luma, chroma })
}
