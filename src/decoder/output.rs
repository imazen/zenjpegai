//! Output stage: internal-range YUV planes → displayable samples.
//!
//! Ports the tail of `CodingEngine.decompress` for the default colour pipeline:
//! `ColourTransformation.post_processing` (`Image.to_RGB_`, BT.709, then clip) and
//! `ImageIO.write_png` (range conversion, round half to even, clamp). Every step is the same
//! float32 operation the reference performs, in the same order, including the range
//! conversions that are mathematically identities but not numerically.
//!
//! **Not ported yet:** chroma upsampling for 4:2:0 / 4:2:2 coded pictures (bicubic,
//! `align_corners=True`), the user-defined colour transform (`colour_transform_idx = 2`), and
//! YUV output (`colour_transform_idx = 0`).

use alloc::vec::Vec;

use super::reconstruct::Planes;
use crate::error::{Error, Result};
use crate::header::{ColourTransform, PictureHeader};

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

/// `colour_processing.post_processing` for `colour_transform_idx = 1` on 4:4:4 planes.
pub fn to_rgb_planes(hdr: &PictureHeader, planes: &Planes) -> Result<RgbPlanes> {
    if hdr.colour_transform != ColourTransform::Bt709 {
        return Err(Error::Unsupported(
            "colour transforms other than BT.709 YCbCr to RGB",
        ));
    }
    if (hdr.s_ver, hdr.s_hor, hdr.c_ver, hdr.c_hor) != (1, 1, 1, 1) {
        return Err(Error::Unsupported("chroma-subsampled pictures"));
    }
    let (h, w) = (planes.y.h, planes.y.w);
    if (planes.u.h, planes.u.w) != (h, w) || (planes.v.h, planes.v.w) != (h, w) {
        return Err(Error::InvalidArgument("4:4:4 planes must have equal sizes"));
    }
    let kry = KRY as f32;
    let kby = KBY as f32;
    let gu = (KB * KBY / KG) as f32;
    let gv = (KR * KRY / KG) as f32;
    let n = h * w;
    let (mut r, mut g, mut b) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        // to_RGB_: [0, 255] -> [0, 1], convert, back to [0, 255]; then clip_data_.
        let y = convert_range(planes.y.data[i], 255.0, 1.0);
        let u = convert_range(planes.u.data[i], 255.0, 1.0) - 0.5;
        let v = convert_range(planes.v.data[i], 255.0, 1.0) - 0.5;
        let rv = y + kry * v;
        let gvv = y - gu * u - gv * v;
        let bv = y + kby * u;
        r.push(convert_range(rv, 1.0, 255.0).clamp(0.0, 255.0));
        g.push(convert_range(gvv, 1.0, 255.0).clamp(0.0, 255.0));
        b.push(convert_range(bv, 1.0, 255.0).clamp(0.0, 255.0));
    }
    Ok(RgbPlanes {
        width: w,
        height: h,
        r,
        g,
        b,
    })
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
