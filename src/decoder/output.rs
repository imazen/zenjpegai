//! Output stage: internal-range YUV planes → displayable samples.
//!
//! Ports the tail of `CodingEngine.decompress` for the default colour pipeline:
//! `ColourTransformation.post_processing` (`Image.to_RGB_`, BT.709, then clip) and
//! `ImageIO.write_png` (range conversion, round half to even, clamp). Every step is the same
//! float32 operation the reference performs, in the same order, including the range
//! conversions that are mathematically identities but not numerically.
//!
//! Also here: `Image.to_format_` (coded → source chroma format; the only conversion the
//! reference can need at the decoder is bicubic up-sampling to 4:4:4, `align_corners=True`) and
//! YUV output for sources that were YUV (`colour_transform_idx = 0`).
//!
//! **Not ported:** the user-defined colour transform (`colour_transform_idx = 2`). Upstream's
//! inverse uses the first row of the inverse matrix for all three output components, which
//! cannot be what is intended; without a trustworthy oracle such streams are rejected.

use alloc::vec::Vec;

use super::reconstruct::Planes;
use crate::error::{Error, Result};
use crate::header::{ColourTransform, PictureHeader};
use crate::tensor::Tensor;

// BT.709 (`YuvStd.params_dict['709']`), combined in f64 like the Python source, then narrowed
// to f32 where PyTorch multiplies a float32 tensor by a Python float.
const KR: f64 = 0.2126;
const KG: f64 = 0.7152;
const KB: f64 = 0.0722;
const KBY: f64 = 1.8556;
const KRY: f64 = 1.5748;

/// `RangesOps.convert_range` for ranges starting at zero: `(x / in_max) * out_max`.
#[inline]
fn convert_range(x: f32, in_max: f32, out_max: f32) -> f32 {
    ((x - 0.0) / in_max) * out_max + 0.0
}

/// Final picture: interleaved RGB samples, one `u16` per sample (8- and 10-bit share it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
    /// `[r, g, b, r, g, b, ...]`, row-major.
    pub data: Vec<u16>,
}

/// Float RGB planes in `[0, 255]`, before rounding (what the reference hashes as its "MD5").
#[derive(Clone, Debug)]
pub struct RgbPlanes {
    pub width: usize,
    pub height: usize,
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

const CUBIC_A: f32 = -0.75;

// PyTorch's CPU build contracts these polynomials into fused multiply-adds (checked by probing
// `F.interpolate` with impulses: only the fused evaluation reproduces its weights to the bit).
#[inline]
fn cubic1(x: f32) -> f32 {
    // ((A + 2) x - (A + 3)) x x + 1
    let p = libm::fmaf(CUBIC_A + 2.0, x, -(CUBIC_A + 3.0));
    libm::fmaf(p * x, x, 1.0)
}

#[inline]
fn cubic2(x: f32) -> f32 {
    // ((A x - 5A) x + 8A) x - 4A
    let p = libm::fmaf(CUBIC_A, x, -5.0 * CUBIC_A);
    let p = libm::fmaf(p, x, 8.0 * CUBIC_A);
    libm::fmaf(p, x, -4.0 * CUBIC_A)
}

/// `x0 c0 + x1 c1 + x2 c2 + x3 c3` exactly as PyTorch's compiled kernel evaluates it:
/// `fma(x3, c3, fma(x2, c2, fma(x0, c0, x1 * c1)))`. Found by matching its output bit for bit
/// over all 24 orders, fused and unfused (`scripts/ref_vectors/gen_resize_vectors.py` has the
/// vectors); any other order is off by an ulp or two in a quarter of the samples.
#[inline]
fn dot4(x: [f32; 4], c: [f32; 4]) -> f32 {
    let acc = x[1] * c[1];
    let acc = libm::fmaf(x[0], c[0], acc);
    let acc = libm::fmaf(x[2], c[2], acc);
    libm::fmaf(x[3], c[3], acc)
}

/// Source index, taps and weights of one output coordinate (`align_corners = True`).
fn cubic_taps(dst: usize, in_len: usize, out_len: usize) -> ([usize; 4], [f32; 4]) {
    let scale = if out_len > 1 {
        (in_len - 1) as f32 / (out_len - 1) as f32
    } else {
        0.0
    };
    let real = scale * dst as f32;
    let base = libm::floorf(real);
    let t = real - base;
    let base = base as isize;
    let idx =
        core::array::from_fn(|k| (base - 1 + k as isize).clamp(0, in_len as isize - 1) as usize);
    let u = 1.0 - t;
    (
        idx,
        [cubic2(t + 1.0), cubic1(t), cubic1(u), cubic2(u + 1.0)],
    )
}

/// `F.interpolate(x, size, mode="bicubic", align_corners=True)` as PyTorch's CPU kernel
/// computes it: per output sample, four horizontal 4-tap sums, then one vertical 4-tap sum, each
/// accumulated left to right in single precision; out-of-range taps repeat the border sample.
pub fn resize_bicubic(src: &Tensor<f32>, out_h: usize, out_w: usize) -> Result<Tensor<f32>> {
    if src.h == 0 || src.w == 0 {
        return Err(Error::InvalidArgument("resize: empty input"));
    }
    let mut out = Tensor::<f32>::zeros(src.c, out_h, out_w)?;
    let xs: Vec<([usize; 4], [f32; 4])> = (0..out_w).map(|x| cubic_taps(x, src.w, out_w)).collect();
    for c in 0..src.c {
        let plane = src.plane(c);
        for y in 0..out_h {
            let (iy, wy) = cubic_taps(y, src.h, out_h);
            let dst = &mut out.plane_mut(c)[y * out_w..][..out_w];
            for (d, (ix, wx)) in dst.iter_mut().zip(&xs) {
                // Four horizontal sums (one per source row), then the vertical sum.
                let rows: [f32; 4] = core::array::from_fn(|k| {
                    let row = &plane[iy[k] * src.w..][..src.w];
                    dot4([row[ix[0]], row[ix[1]], row[ix[2]], row[ix[3]]], *wx)
                });
                *d = dot4(rows, wy);
            }
        }
    }
    Ok(out)
}

/// `rec_img.to_format_(source format)`: the chroma planes arrive in the coded subsampling
/// (`c_ver`, `c_hor`) and leave in the source's (`s_ver`, `s_hor`). The encoder guarantees
/// `c >= s`, and the reference only implements the conversions to 4:4:4.
pub fn to_source_format(hdr: &PictureHeader, planes: Planes) -> Result<Planes> {
    let (c, s) = ((hdr.c_ver, hdr.c_hor), (hdr.s_ver, hdr.s_hor));
    if c == s {
        return Ok(planes);
    }
    if s != (1, 1) || !matches!(c, (2, 2) | (1, 2)) {
        return Err(Error::Unsupported(
            "chroma format conversion other than 4:2:0 / 4:2:2 to 4:4:4",
        ));
    }
    let (h, w) = (planes.y.h, planes.y.w);
    Ok(Planes {
        u: resize_bicubic(&planes.u, h, w)?,
        v: resize_bicubic(&planes.v, h, w)?,
        y: planes.y,
    })
}

/// `colour_processing.post_processing` for `colour_transform_idx = 1` on 4:4:4 planes.
pub fn to_rgb_planes(hdr: &PictureHeader, planes: &Planes) -> Result<RgbPlanes> {
    to_rgb_planes_owned(hdr, planes.clone())
}

/// [`to_rgb_planes`] that converts in place: the three input planes become the output planes,
/// so no second set of full-size planes exists at any point.
pub fn to_rgb_planes_owned(hdr: &PictureHeader, planes: Planes) -> Result<RgbPlanes> {
    if hdr.colour_transform != ColourTransform::Bt709 {
        return Err(Error::Unsupported(
            "colour transforms other than BT.709 YCbCr to RGB",
        ));
    }
    let (h, w) = (planes.y.h, planes.y.w);
    if (planes.u.h, planes.u.w) != (h, w) || (planes.v.h, planes.v.w) != (h, w) {
        return Err(Error::InvalidArgument("4:4:4 planes must have equal sizes"));
    }
    let kry = KRY as f32;
    let kby = KBY as f32;
    let gu = (KB * KBY / KG) as f32;
    let gv = (KR * KRY / KG) as f32;
    let (mut r, mut g, mut b) = (planes.y.data, planes.u.data, planes.v.data);
    for ((py, pu), pv) in r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()) {
        // to_RGB_: [0, 255] -> [0, 1], convert, back to [0, 255]; then clip_data_.
        let y = convert_range(*py, 255.0, 1.0);
        let u = convert_range(*pu, 255.0, 1.0) - 0.5;
        let v = convert_range(*pv, 255.0, 1.0) - 0.5;
        let rv = y + kry * v;
        let gvv = y - gu * u - gv * v;
        let bv = y + kby * u;
        *py = convert_range(rv, 1.0, 255.0).clamp(0.0, 255.0);
        *pu = convert_range(gvv, 1.0, 255.0).clamp(0.0, 255.0);
        *pv = convert_range(bv, 1.0, 255.0).clamp(0.0, 255.0);
    }
    Ok(RgbPlanes {
        width: w,
        height: h,
        r,
        g,
        b,
    })
}

/// Decoded YUV picture (`colour_transform_idx = 0`: the source was YUV and stays YUV), chroma
/// in the source's subsampling, one `u16` per sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YuvImage {
    pub width: usize,
    pub height: usize,
    pub chroma_width: usize,
    pub chroma_height: usize,
    pub bit_depth: u8,
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
}

/// What a stream decodes to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Picture {
    Rgb(RgbImage),
    Yuv(YuvImage),
}

/// One plane: clip to the internal range, scale to `[0, 2^depth - 1]`, round half to even.
pub fn quantize_plane(data: &[f32], bit_depth: u8) -> Vec<u16> {
    let max = ((1u32 << bit_depth) - 1) as f32;
    data.iter()
        .map(|&x| {
            libm::rintf(convert_range(x.clamp(0.0, 255.0), 255.0, max)).clamp(0.0, max) as u16
        })
        .collect()
}

/// The tail of the decoder for planes already in the source's chroma format (post-filters, if
/// any, have run): colour transform, clip, quantise to the stream's bit depth.
pub fn finish(hdr: &PictureHeader, planes: &Planes) -> Result<Picture> {
    if hdr.bit_depth != 8 && hdr.bit_depth != 10 {
        return Err(Error::InvalidData("bit depth must be 8 or 10"));
    }
    match hdr.colour_transform {
        ColourTransform::Bt709 => {
            if (hdr.s_ver, hdr.s_hor) != (1, 1) {
                // to_RGB_ would up-sample first; an RGB source is never subsampled.
                return Err(Error::Unsupported(
                    "RGB output from a chroma-subsampled source",
                ));
            }
            Ok(Picture::Rgb(quantize(
                &to_rgb_planes(hdr, planes)?,
                hdr.bit_depth,
            )?))
        }
        ColourTransform::None => Ok(Picture::Yuv(YuvImage {
            width: planes.y.w,
            height: planes.y.h,
            chroma_width: planes.u.w,
            chroma_height: planes.u.h,
            bit_depth: hdr.bit_depth,
            // post_processing: clip_data_(); write: convert range, round half to even, clip.
            y: quantize_plane(&planes.y.data, hdr.bit_depth),
            u: quantize_plane(&planes.u.data, hdr.bit_depth),
            v: quantize_plane(&planes.v.data, hdr.bit_depth),
        })),
        ColourTransform::Custom { .. } => Err(Error::Unsupported("user-defined colour transform")),
    }
}

/// `ImageIO.write_png` quantisation: internal range → `[0, 2^depth - 1]`, round half to even.
pub fn quantize(rgb: &RgbPlanes, bit_depth: u8) -> Result<RgbImage> {
    if bit_depth != 8 && bit_depth != 10 {
        return Err(Error::InvalidArgument("output bit depth must be 8 or 10"));
    }
    let max = ((1u32 << bit_depth) - 1) as f32;
    let q = |x: f32| -> u16 {
        convert_range(x, 255.0, max)
            .round_ties_even()
            .clamp(0.0, max) as u16
    };
    let mut data = Vec::with_capacity(rgb.r.len() * 3);
    for i in 0..rgb.r.len() {
        data.extend_from_slice(&[q(rgb.r[i]), q(rgb.g[i]), q(rgb.b[i])]);
    }
    Ok(RgbImage {
        width: rgb.width,
        height: rgb.height,
        bit_depth,
        data,
    })
}
