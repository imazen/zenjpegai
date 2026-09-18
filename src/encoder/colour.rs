//! Encoder-side colour pre-processing: a source picture to the two tensors the analysis
//! transforms take.
//!
//! Ports `ref/src/codec/coding_tools/core_models/CCS_SGMM/ccs_sgmm_tool.py::compress` (its first
//! half) with `ref/src/codec/common/{image.py, colorspace.py}`: `to_YUV_`, `convert_range_`,
//! `pad_`, `to_format_` (chroma to the coded subsampling) and the `pixel_unshuffle` / row-split
//! / `repeat` packing that builds the 12-plane chroma input. Every step is plain `f32`
//! arithmetic in the order the reference evaluates it, which makes the result bit-exact against
//! its tensor dumps.
//!
//! Sources: interleaved RGB (`colour_transform_idx` 1, BT.709) and planar YUV
//! (`colour_transform_idx` 0), 4:4:4 / 4:2:2 / 4:2:0, any bit depth the header can signal.
//! Coded subsampling defaults to the source's and can be raised per axis (the reference's
//! `-c_ver_value` / `-c_hor_value`); a 4:2:2 source coded 4:2:0 is `NotImplementedError`
//! upstream and an [`Error::Unsupported`] here.

use alloc::vec::Vec;

use super::resample::resize_bilinear;
use crate::decoder::output::{RgbImage, YuvImage};
use crate::error::{Error, Result};
use crate::header::ColourTransform;
use crate::tensor::Tensor;

/// BT.709 luma weights and the two chroma denominators (`colorspace.py`).
const KR: f32 = 0.2126;
const KG: f32 = 0.7152;
const KB: f32 = 0.0722;
const KBY: f32 = 1.8556;
const KRY: f32 = 1.5748;

/// Luma and chroma inputs of the analysis transforms.
pub struct AnalysisInput {
    /// `[1, h, w]` in `[0, 255]`, the source's own size: `compress` pads luma to even only
    /// for the support-channel `pixel_unshuffle`; the analysis tiles run on the unpadded
    /// picture (`get_processed_img_shape`), so an odd-size picture is analysed as is.
    pub luma: Tensor<f32>,
    /// `[12, ph / 2, pw / 2]` (`ph`/`pw` even): the four luma phases (the "support
    /// information"), then eight chroma channels whose layout depends on the coded
    /// subsampling (see [`preprocess`]).
    pub chroma: Tensor<f32>,
}

/// A picture the encoder can take (`Image` on the input side of `coding_engine.py::compress`).
#[derive(Clone, Debug)]
pub enum SourceImage {
    /// Interleaved RGB; coded as `colour_transform_idx` 1 (BT.709), `s_ver = s_hor = 1`.
    Rgb(RgbImage),
    /// Planar YUV in the source's subsampling; coded as `colour_transform_idx` 0.
    Yuv(YuvImage),
}

/// What the picture header carries about the source and the coded chroma format
/// (`coding_engine.py`'s `s_ver`, `s_hor`, `c_ver`, `c_hor`, `image_data_bits`,
/// `colour_transform_idx`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceMeta {
    /// Bits per sample of the source (`bit_depth_idc`'s decoded value).
    pub bit_depth: u8,
    /// Source chroma subsampling factors (1 or 2), vertical and horizontal.
    pub s_ver: u8,
    /// See [`SourceMeta::s_ver`], horizontal factor.
    pub s_hor: u8,
    /// Coded chroma subsampling factors after the `c_*_value` overrides.
    pub c_ver: u8,
    /// See [`SourceMeta::c_ver`], horizontal factor.
    pub c_hor: u8,
    /// `ColourTransform::None` for a YUV source, `Bt709` for RGB.
    pub colour_transform: ColourTransform,
}

impl SourceMeta {
    /// `c_ver`/`c_hor` resolved the way `coding_engine.py` does: the override when given,
    /// otherwise the source's own subsampling (`self.c_ver = max(1, s_ver)`), with its asserts.
    pub fn resolve(
        s_ver: u8,
        s_hor: u8,
        bit_depth: u8,
        colour_transform: ColourTransform,
        c_ver: Option<u8>,
        c_hor: Option<u8>,
    ) -> Result<Self> {
        let c_ver = c_ver.unwrap_or(s_ver);
        let c_hor = c_hor.unwrap_or(s_hor);
        if c_ver < s_ver || c_hor < s_hor {
            return Err(Error::InvalidArgument(
                "coded chroma subsampling must not be finer than the source's",
            ));
        }
        if !matches!((c_ver, c_hor), (1, 1) | (1, 2) | (2, 2)) {
            return Err(Error::Unsupported(
                "coded chroma format must be 4:4:4, 4:2:2 or 4:2:0",
            ));
        }
        // `Image.to_420_` only converts from 4:4:4; a 4:2:2 source coded 4:2:0 is
        // `NotImplementedError` upstream.
        if (s_ver, s_hor) == (1, 2) && (c_ver, c_hor) == (2, 2) {
            return Err(Error::Unsupported(
                "a 4:2:2 source cannot be coded as 4:2:0",
            ));
        }
        Ok(Self {
            bit_depth,
            s_ver,
            s_hor,
            c_ver,
            c_hor,
            colour_transform,
        })
    }
}

impl SourceImage {
    /// Luma width in samples.
    pub fn width(&self) -> usize {
        match self {
            Self::Rgb(i) => i.width,
            Self::Yuv(i) => i.width,
        }
    }

    /// Luma height in samples.
    pub fn height(&self) -> usize {
        match self {
            Self::Rgb(i) => i.height,
            Self::Yuv(i) => i.height,
        }
    }

    /// Bits per sample of the source.
    pub fn bit_depth(&self) -> u8 {
        match self {
            Self::Rgb(i) => i.bit_depth,
            Self::Yuv(i) => i.bit_depth,
        }
    }

    /// The source's chroma subsampling `(s_ver, s_hor)`, derived from the chroma plane size.
    pub fn subsampling(&self) -> Result<(u8, u8)> {
        match self {
            Self::Rgb(_) => Ok((1, 1)),
            Self::Yuv(i) => {
                let (w, h, cw, ch) = (i.width, i.height, i.chroma_width, i.chroma_height);
                if (cw, ch) == (w, h) {
                    Ok((1, 1))
                } else if (cw, ch) == (w.div_ceil(2), h) {
                    Ok((1, 2))
                } else if (cw, ch) == (w.div_ceil(2), h.div_ceil(2)) {
                    Ok((2, 2))
                } else {
                    Err(Error::InvalidArgument(
                        "chroma plane size matches no 4:4:4 / 4:2:2 / 4:2:0 source format",
                    ))
                }
            }
        }
    }

    /// [`SourceMeta`] for this source with the `c_ver` / `c_hor` overrides applied.
    pub fn meta(&self, c_ver: Option<u8>, c_hor: Option<u8>) -> Result<SourceMeta> {
        let (s_ver, s_hor) = self.subsampling()?;
        SourceMeta::resolve(
            s_ver,
            s_hor,
            self.bit_depth(),
            match self {
                Self::Rgb(_) => ColourTransform::Bt709,
                Self::Yuv(_) => ColourTransform::None,
            },
            c_ver,
            c_hor,
        )
    }

    /// `Image.read_yuv` / `extract_info`: a planar YUV file whose name carries the picture
    /// size, bit depth and chroma format — `WxH`, `Nbit` and `444` / `420` / `422`, in the
    /// basename (defaults 8 bit, 4:2:0, like `read_file`'s). Samples are 1 byte at 8 bit,
    /// little-endian `u16` above, planes in Y-U-V order.
    pub fn read_yuv(filename: &str, data: &[u8]) -> Result<Self> {
        // `os.path.basename`: the name after the last separator (the metadata is matched in
        // the file name only, never in directory names).
        let name = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
        // `re.search(r"(\d+)x(\d+)")` / `r"(\d+)bit"` on the basename.
        let (w, h) = find_dims(name).ok_or(Error::InvalidArgument(
            "yuv input: no <width>x<height> in the file name",
        ))?;
        let bit_depth = find_bits(name).unwrap_or(8);
        if bit_depth == 0 || bit_depth > 16 {
            return Err(Error::InvalidArgument(
                "yuv input: bit depth outside 1..=16",
            ));
        }
        // `extract_info`'s order: 444, 420, 422, sRGB; `read_file` defaults to 4:2:0.
        let (cw, ch) = if name.contains("444") {
            (w, h)
        } else if name.contains("420") {
            (w.div_ceil(2), h.div_ceil(2))
        } else if name.contains("422") {
            (w.div_ceil(2), h)
        } else if name.contains("sRGB") {
            return Err(Error::InvalidArgument(
                "yuv input: chroma format in the file name is not 444, 420 or 422",
            ));
        } else {
            (w.div_ceil(2), h.div_ceil(2))
        };
        let planes = w * h + 2 * cw * ch;
        let bytes_per = if bit_depth > 8 { 2 } else { 1 };
        if data.len() < planes * bytes_per {
            return Err(Error::InvalidArgument(
                "yuv input: file shorter than the planes",
            ));
        }
        let mut pos = 0usize;
        let mut plane = |n: usize| -> Vec<u16> {
            let b = &data[pos..pos + n * bytes_per];
            pos += n * bytes_per;
            if bytes_per == 1 {
                b.iter().map(|&v| u16::from(v)).collect()
            } else {
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect()
            }
        };
        let y = plane(w * h);
        let u = plane(cw * ch);
        let v = plane(cw * ch);
        Ok(Self::Yuv(YuvImage {
            width: w,
            height: h,
            chroma_width: cw,
            chroma_height: ch,
            bit_depth,
            y,
            u,
            v,
        }))
    }
}

impl From<RgbImage> for SourceImage {
    fn from(rgb: RgbImage) -> Self {
        Self::Rgb(rgb)
    }
}

impl From<&RgbImage> for SourceImage {
    fn from(rgb: &RgbImage) -> Self {
        Self::Rgb(rgb.clone())
    }
}

impl From<YuvImage> for SourceImage {
    fn from(yuv: YuvImage) -> Self {
        Self::Yuv(yuv)
    }
}

impl From<&YuvImage> for SourceImage {
    fn from(yuv: &YuvImage) -> Self {
        Self::Yuv(yuv.clone())
    }
}

impl From<&SourceImage> for SourceImage {
    fn from(src: &SourceImage) -> Self {
        src.clone()
    }
}

/// The first `<digits>x<digits>` of `name` (`extract_info`'s `(?P<w>\d+)x(?P<h>\d+)`).
fn find_dims(name: &str) -> Option<(usize, usize)> {
    let b = name.as_bytes();
    for i in 0..b.len() {
        if b[i] != b'x' || i == 0 || !b[i - 1].is_ascii_digit() {
            continue;
        }
        let w_start = i - b[..i]
            .iter()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .count();
        let h_len = b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
        if h_len > 0
            && let (Ok(w), Ok(h)) = (name[w_start..i].parse(), name[i + 1..i + 1 + h_len].parse())
        {
            return Some((w, h));
        }
    }
    None
}

/// The first `<digits>bit` of `name` (`extract_info`'s `(?P<b>\d+)bit`).
fn find_bits(name: &str) -> Option<u8> {
    let idx = name.find("bit")?;
    let start = idx
        - name.as_bytes()[..idx]
            .iter()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .count();
    if start == idx {
        return None;
    }
    name[start..idx].parse().ok()
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

/// `Image.to_format_(coded)`: the chroma planes at the source subsampling to the coded one.
/// The only conversion the reference's encoder can reach is 4:4:4 down-sampling.
fn to_coded_format(
    u: Vec<f32>,
    v: Vec<f32>,
    sch: usize,
    scw: usize,
    s: (u8, u8),
    c: (u8, u8),
) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
    if s == c {
        return Ok((u, v, sch, scw));
    }
    let (tch, tcw) = match c {
        (1, 1) => (sch, scw),
        (1, 2) => (sch, scw.div_ceil(2)),
        (2, 2) => (sch.div_ceil(2), scw.div_ceil(2)),
        _ => {
            return Err(Error::Unsupported(
                "coded chroma format must be 4:4:4, 4:2:2 or 4:2:0",
            ));
        }
    };
    if s != (1, 1) {
        // `to_420_` / `to_422_` accept only a 4:4:4 source (and `to_444_` is unreachable:
        // c >= s is checked). `meta()` already rejected 4:2:2 -> 4:2:0.
        return Err(Error::Unsupported(
            "chroma conversion between subsampled formats",
        ));
    }
    let resample = |p: Vec<f32>| -> Result<Vec<f32>> {
        let t = Tensor::from_vec(1, sch, scw, p)?;
        Ok(resize_bilinear(&t, tch, tcw)?.data)
    };
    Ok((resample(u)?, resample(v)?, tch, tcw))
}

/// RGB (4:4:4) or planar YUV (4:4:4 / 4:2:2 / 4:2:0) → the analysis transforms' inputs, with
/// the chroma at the coded subsampling `meta` was resolved with ([`SourceImage::meta`]).
///
/// The 12-plane chroma tensor is `ccs_sgmm_tool.py::compress`'s: channels 0..4 are the luma's
/// four pixel-unshuffle phases (its "support information"); 4..12 are the coded chroma —
/// four U phases then four V phases at 4:4:4, U and V each repeated four times at 4:2:0, and
/// each chroma row phase repeated twice at 4:2:2.
pub fn preprocess(src: &SourceImage, meta: &SourceMeta) -> Result<AnalysisInput> {
    let (w, h) = (src.width(), src.height());
    let max = ((1u32 << meta.bit_depth) - 1) as f32;
    // The three components in `[0, 255]` (`convert_range_` to [0, 1], `to_YUV_`, then
    // `convert_range_` to the internal range). Chroma starts at the source's subsampling.
    let (yp, up, vp, sch, scw) = match src {
        SourceImage::Rgb(rgb) => {
            if rgb.data.len() != w * h * 3 {
                return Err(Error::InvalidArgument("RGB buffer size does not match"));
            }
            let (mut yp, mut up, mut vp) = (
                alloc::vec![0f32; w * h],
                alloc::vec![0f32; w * h],
                alloc::vec![0f32; w * h],
            );
            for i in 0..w * h {
                // `to_YUV_`: [0, 1] -> BT.709 -> back to the internal range.
                let r = rgb.data[3 * i] as f32 / max;
                let g = rgb.data[3 * i + 1] as f32 / max;
                let b = rgb.data[3 * i + 2] as f32 / max;
                let y = KR * r + KG * g + KB * b;
                yp[i] = y * 255.0;
                up[i] = ((b - y) / KBY + 0.5) * 255.0;
                vp[i] = ((r - y) / KRY + 0.5) * 255.0;
            }
            (yp, up, vp, h, w)
        }
        SourceImage::Yuv(yuv) => {
            let (cw, ch) = (yuv.chroma_width, yuv.chroma_height);
            if yuv.y.len() != w * h || yuv.u.len() != cw * ch || yuv.v.len() != cw * ch {
                return Err(Error::InvalidArgument("YUV plane size does not match"));
            }
            // `read_yuv` keeps the [0, 1] range; `convert_range_` to the internal range.
            let scale = |p: &[u16]| p.iter().map(|&v| v as f32 / max * 255.0).collect();
            (scale(&yuv.y), scale(&yuv.u), scale(&yuv.v), ch, cw)
        }
    };

    // `pad_` luma to an even size for the four support-channel phases (`chroma_sup_info`);
    // the analysis transform itself sees the luma unpadded.
    let (pw, ph) = (w + w % 2, h + h % 2);
    let luma = Tensor::from_vec(1, h, w, yp)?;
    let yp = replicate_pad(&luma.data, w, h, pw, ph);
    let mut chroma = Tensor::<f32>::zeros(12, ph / 2, pw / 2)?;
    unshuffle_into(&mut chroma, 0, &yp, ph, pw);

    let (up, vp, cch, ccw) = to_coded_format(
        up,
        vp,
        sch,
        scw,
        (meta.s_ver, meta.s_hor),
        (meta.c_ver, meta.c_hor),
    )?;
    match (meta.c_ver, meta.c_hor) {
        (1, 1) => {
            // `pad_` the chroma to the padded luma size, `pixel_unshuffle` each component.
            if (cch, ccw) != (h, w) {
                return Err(Error::InvalidData("4:4:4 chroma is not at luma size"));
            }
            let up = replicate_pad(&up, w, h, pw, ph);
            let vp = replicate_pad(&vp, w, h, pw, ph);
            unshuffle_into(&mut chroma, 4, &up, ph, pw);
            unshuffle_into(&mut chroma, 8, &vp, ph, pw);
        }
        (2, 2) => {
            if (cch, ccw) != (ph / 2, pw / 2) {
                return Err(Error::InvalidData("4:2:0 chroma is not at half size"));
            }
            // `repeat(1, 4, 1, 1)`: each chroma plane occupies four consecutive channels.
            for k in 0..4 {
                chroma.plane_mut(4 + k).copy_from_slice(&up);
                chroma.plane_mut(8 + k).copy_from_slice(&vp);
            }
        }
        (1, 2) => {
            // `pad_(0, h % 2, ['b', 'c'])`, then the even and odd chroma rows, each repeated
            // twice: [U_even, U_even, U_odd, U_odd], same for V.
            if (cch, ccw) != (h, pw / 2) {
                return Err(Error::InvalidData("4:2:2 chroma is not at full height"));
            }
            for (base, p) in [(4, &up), (8, &vp)] {
                let padded = replicate_pad(p, ccw, cch, ccw, ph);
                for k in 0..4 {
                    let phase = k / 2;
                    let dst = chroma.plane_mut(base + k);
                    for y in 0..ph / 2 {
                        dst[y * ccw..][..ccw]
                            .copy_from_slice(&padded[(2 * y + phase) * ccw..][..ccw]);
                    }
                }
            }
        }
        _ => {
            return Err(Error::Unsupported(
                "coded chroma must be 4:4:4, 4:2:2 or 4:2:0",
            ));
        }
    }
    Ok(AnalysisInput { luma, chroma })
}

/// RGB (8 bit, 4:4:4, BT.709) → the analysis transforms' inputs, coded at full chroma
/// resolution (`s_ver = s_hor = c_ver = c_hor = 1`).
pub fn preprocess_rgb(rgb: &RgbImage) -> Result<AnalysisInput> {
    let src = SourceImage::from(rgb);
    let meta = src.meta(None, None)?;
    preprocess(&src, &meta)
}
