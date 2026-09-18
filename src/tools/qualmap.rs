//! Quality map: spatially varying quantisation, one step index per latent position, shared by
//! both components.
//!
//! Ports `ref/src/codec/coding_tools/quality_map/quality_map.py`: `decode`, `quantize_scale`,
//! `dequantize_resi`, `matching_qp_to_scale{,_log}` for the decoder, and `encode`,
//! `quantize_resi`, `generate_qp_map_index` and the mask-file generator
//! `generate_qp_map_byROI_map` (`qp_map_type = 3`, the only one of upstream's four that works on
//! this code path) for the encoder.

use crate::container::{join_threads, split_threads};
use crate::error::{Error, Result};
use crate::header::QualityMapHeader;
use crate::mans::AnsTables;
use crate::tensor::Tensor;

/// `sigma_list`: the sigma the delta plane is coded with, by `quality_map_entropy_index`.
const SIGMAS: [u8; 8] = [0, 4, 5, 6, 8, 10, 14, 18];
/// `q_scale_log_map`: log-domain (Q7 sigma index) offset for `qp = -8..=8`.
const SCALE_LOG: [i32; 17] = [
    -886, -775, -664, -554, -443, -332, -221, -111, 0, 111, 221, 332, 443, 554, 664, 775, 886,
];
/// `q_scale_map`: linear step in 1/16 units for `qp = -8..=8`.
const SCALE: [i32; 17] = [
    4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 23, 27, 32, 39, 46, 54, 64,
];

/// The decoded map, `[1, H, W]` at latent resolution.
#[derive(Clone, Debug)]
pub struct QualityMap {
    pub qp: Tensor<i32>,
}

/// The reference matches `qp == i - 8` for `i in 0..17` and leaves anything else at zero.
fn lookup(table: &[i32; 17], qp: i32) -> i32 {
    usize::try_from(qp + 8)
        .ok()
        .and_then(|i| table.get(i))
        .copied()
        .unwrap_or(0)
}

/// `num_zero_list`: the fraction of zero deltas that selects `quality_map_entropy_index`.
const ZERO_FRACTIONS: [f32; 7] = [0.25, 0.5, 0.70, 0.80, 0.90, 0.95, 0.99];

impl QualityMap {
    /// A caller-supplied map, one `qp` per latent position, clamped to the table's `-8..=8`
    /// (`clip_qpmap`).
    pub fn from_qp(qp: &Tensor<i32>) -> Result<Self> {
        if qp.c != 1 {
            return Err(Error::InvalidArgument("quality map: one plane expected"));
        }
        let data = qp.data.iter().map(|&v| v.clamp(-8, 8)).collect();
        Ok(Self {
            qp: Tensor::from_vec(1, qp.h, qp.w, data)?,
        })
    }

    /// `generate_qp_map_byROI_map` with `adjust_qp = 1`: a white (255, 255, 255) pixel of the
    /// mask means +6, anything else -3. The mask is resized to a whole multiple of the latent
    /// size with nearest-neighbour interpolation, then max-pooled down to it.
    ///
    /// `mask` is `[3, mh, mw]`, 8-bit RGB.
    pub fn from_roi_mask(mask: &Tensor<u8>, h: usize, w: usize) -> Result<Self> {
        if mask.c != 3 || h == 0 || w == 0 || mask.h == 0 || mask.w == 0 {
            return Err(Error::InvalidArgument("quality map: bad ROI mask"));
        }
        let (dy, dx) = (mask.h.div_ceil(h), mask.w.div_ceil(w));
        let (ih, iw) = (dy * h, dx * w);
        let mut qp = Tensor::<i32>::zeros(1, h, w)?;
        for by in 0..h {
            for bx in 0..w {
                // `F.interpolate` with no mode is nearest neighbour (source index
                // `floor(dst * in / out)`), then `max_pool2d` per channel, and only then are the
                // three channels compared with 255.
                let mut maxima = [0u8; 3];
                for y in by * dy..(by + 1) * dy {
                    let sy = y * mask.h / ih;
                    for x in bx * dx..(bx + 1) * dx {
                        let sx = x * mask.w / iw;
                        for (c, m) in maxima.iter_mut().enumerate() {
                            *m = (*m).max(mask.plane(c)[sy * mask.w + sx]);
                        }
                    }
                }
                qp.data[by * w + bx] = if maxima == [255, 255, 255] { 6 } else { -3 };
            }
        }
        Self::from_qp(&qp)
    }

    /// `encode`'s DPCM: the delta of every position against its prediction.
    fn deltas(&self) -> alloc::vec::Vec<i16> {
        let (h, w) = (self.qp.h, self.qp.w);
        let mut out = alloc::vec![0i16; h * w];
        for y in 0..h {
            for x in 0..w {
                let pred = match (y, x) {
                    (0, 0) => 0,
                    (0, _) => self.qp.data[x - 1],
                    (_, 0) => self.qp.data[(y - 1) * w],
                    _ => (self.qp.data[y * w + x - 1] + self.qp.data[(y - 1) * w + x]) / 2,
                };
                out[y * w + x] = (self.qp.data[y * w + x] - pred) as i16;
            }
        }
        out
    }

    /// `generate_qp_map_index`: the more deltas are zero the lower the sigma the map is coded
    /// with. Note that the reference leaves the very first delta at zero here, although `encode`
    /// writes the value itself there; the count follows the reference.
    pub fn entropy_index(&self) -> u8 {
        let mut deltas = self.deltas();
        deltas[0] = 0;
        let zeros = deltas.iter().filter(|&&d| d == 0).count() as f32;
        let fraction = zeros / (self.qp.h * self.qp.w) as f32;
        // `len(list) - bisect_right(list, fraction)`.
        let above = ZERO_FRACTIONS.iter().filter(|&&t| t <= fraction).count();
        (ZERO_FRACTIONS.len() - above) as u8
    }

    /// `encode`: the DPCM deltas, coded with one fixed sigma and nothing skipped.
    pub fn encode(
        &self,
        tables: &AnsTables,
        hdr: &QualityMapHeader,
    ) -> Result<alloc::vec::Vec<u8>> {
        let sigma = *SIGMAS
            .get(hdr.entropy_index as usize)
            .ok_or(Error::InvalidData("quality_map_entropy_index out of range"))?;
        let deltas = self.deltas();
        let n = deltas.len();
        let mut enc = tables.encoder(hdr.num_threads as usize)?;
        enc.encode_residual(&alloc::vec![sigma; n], &alloc::vec![true; n], &deltas)?;
        let parts = enc.finish();
        Ok(join_threads(
            &parts
                .iter()
                .map(|t| t.as_slice())
                .collect::<alloc::vec::Vec<_>>(),
        ))
    }

    /// `quantize_resi` at flat position `i`: `x * scale / 16` in single precision.
    #[inline]
    pub fn quantize(&self, i: usize, x: f32) -> f32 {
        x * lookup(&SCALE, self.qp.data[i]) as f32 / 16.0
    }

    /// `QualityMap.decode`: every position carries a delta, coded like a residual with one fixed
    /// sigma and nothing skipped; the map is the running prediction plus the delta, predicting
    /// from the left neighbour in the first row, the upper one in the first column, and the
    /// average of both (rounded towards zero) elsewhere.
    pub fn decode(
        tables: &AnsTables,
        payload: &[u8],
        hdr: &QualityMapHeader,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let sigma = *SIGMAS
            .get(hdr.entropy_index as usize)
            .ok_or(Error::InvalidData("quality_map_entropy_index out of range"))?;
        let n = h * w;
        let threads = split_threads(payload, hdr.num_threads as usize)?;
        let mut dec = tables.decoder(&threads)?;
        let mut delta = alloc::vec![0i16; n];
        dec.decode_residual(&alloc::vec![sigma; n], &alloc::vec![true; n], &mut delta)?;

        let mut qp = Tensor::<i32>::zeros(1, h, w)?;
        for y in 0..h {
            for x in 0..w {
                let pred = match (y, x) {
                    (0, 0) => 0,
                    (0, _) => qp.data[x - 1],
                    (_, 0) => qp.data[(y - 1) * w],
                    // torch 1.10 `//` on these tensors truncates, like integer `/` here.
                    _ => (qp.data[y * w + x - 1] + qp.data[(y - 1) * w + x]) / 2,
                };
                qp.data[y * w + x] = pred + delta[y * w + x] as i32;
            }
        }
        Ok(Self { qp })
    }

    /// `quantize_scale`: the log-domain offset of every position, added to all channels.
    pub fn adjust_scale(&self, scale_log: &mut Tensor<i32>) -> Result<()> {
        if (scale_log.h, scale_log.w) != (self.qp.h, self.qp.w) {
            return Err(Error::InvalidData(
                "quality map size does not match the latent",
            ));
        }
        for ch in 0..scale_log.c {
            for (s, &q) in scale_log.plane_mut(ch).iter_mut().zip(&self.qp.data) {
                *s += lookup(&SCALE_LOG, q);
            }
        }
        Ok(())
    }

    /// `dequantize_resi` at flat position `i`: `x / (scale / 16 + 1e-9)` in single precision.
    #[inline]
    pub fn dequantize(&self, i: usize, x: f32) -> f32 {
        x / (lookup(&SCALE, self.qp.data[i]) as f32 / 16.0 + 1e-9)
    }
}
