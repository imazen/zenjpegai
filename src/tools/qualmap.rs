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

    /// `generate_qp_map` (`qp_map_type = 0`): the adaptive variance map. The source picture is
    /// variance-filtered in 4x4 windows (`cal_var`), min-pooled at `alignment_size / 4`
    /// (`alignment_size//4` upstream), and each latent cell loses one step per 0.8 / 1.5 / 2.5 /
    /// 4 times the mean variance — `qp = -(var > 0.8*mean) - (var > 1.5*mean) - (var > 2.5*mean)
    /// - (var > 4*mean)`. Upstream crashes on this path (`img` is an unbound name where the
    /// source tensor belongs); this takes `plane` = the source image in `[0, 2^bit_depth)`.
    /// The map lands on the `align`-cell latent grid — `ceil(h/align) x ceil(w/align)`.
    pub fn from_variance(plane: &Tensor<f32>, align: usize) -> Result<Self> {
        if plane.c != 1 || align < 4 || plane.h == 0 || plane.w == 0 {
            return Err(Error::InvalidArgument("quality map: variance source"));
        }
        // `cal_var(img, 4)` over the source padded with replicate to a multiple of `align`,
        // `window` = 4 samples: the cell variance of `img[:, 0, 4i:4i+4, 4j:4j+4]`.
        let (ph, pw) = (plane.h, plane.w);
        let (vh, vw) = (
            (ph + align - 1) / align * align / 4,
            (pw + align - 1) / align * align / 4,
        );
        let mut var = alloc::vec![0f32; vh * vw];
        for i in 0..vh {
            for j in 0..vw {
                let (mut s, mut s2, mut n) = (0f64, 0f64, 0usize);
                for y in (4 * i).min(ph)..(4 * i + 4).min(ph) {
                    for x in (4 * j).min(pw)..(4 * j + 4).min(pw) {
                        let v = plane.data[y * pw + x] as f64;
                        s += v;
                        s2 += v * v;
                        n += 1;
                    }
                }
                // `torch.var` is the unbiased (n-1) variance.
                let m = s / n as f64;
                var[i * vw + j] = if n > 1 {
                    ((s2 - s * m) / (n - 1) as f64) as f32
                } else {
                    0.0
                };
            }
        }
        // `-(maxpool(-var_map))` with kernel `align / 4`: the minimum variance in each
        // `align/4`-cell block — down to `align`-sized latent cells.
        let df = align / 4;
        let (lh, lw) = (vh.div_ceil(df), vw.div_ceil(df));
        let mut qp = Tensor::<i32>::zeros(1, lh, lw)?;
        let mean: f64 = var.iter().map(|&v| v as f64).sum::<f64>() / (vh * vw).max(1) as f64;
        for by in 0..lh {
            for bx in 0..lw {
                let mut mn = f32::INFINITY;
                for i in by * df..(by * df + df).min(vh) {
                    for j in bx * df..(bx * df + df).min(vw) {
                        mn = mn.min(var[i * vw + j]);
                    }
                }
                let mut q = 0;
                for &t in &[0.8, 1.5, 2.5, 4.0] {
                    if (mn as f64) > t * mean {
                        q -= 1;
                    }
                }
                qp.data[by * lw + bx] = q;
            }
        }
        Self::from_qp(&qp)
    }

    /// `generate_qp_map_byROI` (`qp_map_type = 1`): `-3` on the whole map, `0` inside each ROI
    /// rectangle `(x, y, wid, hei)` given in latent cells. Upstream reads the rectangles from
    /// `args_roi_*` config that is never set — it crashes there; the rects come from `rois`.
    pub fn from_roi_rects(
        rois: &[(usize, usize, usize, usize)],
        h: usize,
        w: usize,
    ) -> Result<Self> {
        if h == 0 || w == 0 {
            return Err(Error::InvalidArgument("quality map: ROI"));
        }
        let mut qp = Tensor::<i32>::zeros(1, h, w)?;
        qp.data.iter_mut().for_each(|v| *v = -3);
        for &(x, y, wid, hei) in rois {
            for by in y..(y + hei).min(h) {
                for bx in x..(x + wid).min(w) {
                    qp.data[by * w + bx] = 0;
                }
            }
        }
        Self::from_qp(&qp)
    }

    /// `generate_qp_map_byNoise` (`qp_map_type = 2`): a checkerboard of `block_qp`-cell blocks
    /// alternating `qp_min` and `qp_max` on `(i + j) % 2`. Upstream reads `block_qp`/`qp_min`/
    /// `qp_max` from config that is never set — it crashes there.
    pub fn from_noise(
        block_qp: usize,
        qp_min: i32,
        qp_max: i32,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        if block_qp == 0 || h == 0 || w == 0 {
            return Err(Error::InvalidArgument("quality map: noise"));
        }
        let mut qp = Tensor::<i32>::zeros(1, h, w)?;
        for i in 0..h.div_ceil(block_qp) {
            for j in 0..w.div_ceil(block_qp) {
                let v = if (i + j) % 2 == 1 { qp_max } else { qp_min };
                for y in i * block_qp..(i * block_qp + block_qp).min(h) {
                    for x in j * block_qp..(j * block_qp + block_qp).min(w) {
                        qp.data[y * w + x] = v;
                    }
                }
            }
        }
        Self::from_qp(&qp)
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
        self.adjust_scale_par(cfg!(feature = "parallel"), scale_log)
    }

    /// [`Self::adjust_scale`], one task per channel when `parallel` holds.
    pub(crate) fn adjust_scale_par(
        &self,
        parallel: bool,
        scale_log: &mut Tensor<i32>,
    ) -> Result<()> {
        if (scale_log.h, scale_log.w) != (self.qp.h, self.qp.w) {
            return Err(Error::InvalidData(
                "quality map size does not match the latent",
            ));
        }
        let plane = scale_log.h * scale_log.w;
        crate::decoder::stats::for_each_chunk(parallel, &mut scale_log.data, plane, |_, s| {
            for (s, &q) in s.iter_mut().zip(&self.qp.data) {
                *s += lookup(&SCALE_LOG, q);
            }
        });
        Ok(())
    }

    /// `dequantize_resi` at flat position `i`: `x / (scale / 16 + 1e-9)` in single precision.
    #[inline]
    pub fn dequantize(&self, i: usize, x: f32) -> f32 {
        x / (lookup(&SCALE, self.qp.data[i]) as f32 / 16.0 + 1e-9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `from_noise` is the `block_qp`-cell checkerboard alternating `qp_min`/`qp_max` on
    /// `(i + j) % 2` — `generate_qp_map_byNoise`.
    #[test]
    fn noise_is_a_checkerboard() {
        let m = QualityMap::from_noise(2, -4, 6, 6, 6).unwrap();
        for y in 0..6 {
            for x in 0..6 {
                let want = if (y / 2 + x / 2) % 2 == 1 { 6 } else { -4 };
                assert_eq!(m.qp.data[y * 6 + x], want, "({x}, {y})");
            }
        }
    }

    /// `from_roi_rects` is `-3` everywhere and `0` inside the rect.
    #[test]
    fn roi_rects_blank_and_mark() {
        let m = QualityMap::from_roi_rects(&[(1, 1, 2, 2)], 4, 4).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let want = if (1..3).contains(&x) && (1..3).contains(&y) {
                    0
                } else {
                    -3
                };
                assert_eq!(m.qp.data[y * 4 + x], want, "({x}, {y})");
            }
        }
    }

    /// `from_variance` on a flat image produces all zeros (no var passes the threshold), and
    /// on a two-region image marks the textured side — `generate_qp_map` (`qp_map_type = 0`).
    #[test]
    fn variance_map_marks_texture() {
        // 4x4 window, alignment 4 → one latent cell per 4x4 source window.
        let flat = Tensor::from_vec(1, 8, 8, alloc::vec![128.0; 64]).unwrap();
        let m = QualityMap::from_variance(&flat, 4).unwrap();
        assert!(m.qp.data.iter().all(|&q| q == 0), "flat image: {m:?}");
        // One half of the image alternates 0/255 (high variance), the other is flat.
        let mut data = alloc::vec![128.0; 8 * 8];
        for y in 0..8 {
            for x in 4..8 {
                data[y * 8 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
            }
        }
        let img = Tensor::from_vec(1, 8, 8, data).unwrap();
        let m = QualityMap::from_variance(&img, 4).unwrap();
        assert!(
            m.qp.data.iter().any(|&q| q < 0),
            "textured region should dip below 0: {m:?}"
        );
        // And it still encodes a valid DPCM map + entropy index.
        let _ = m.entropy_index();
        let _ = m.deltas();
    }
}
