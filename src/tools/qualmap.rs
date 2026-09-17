//! Quality map: spatially varying quantisation, one step index per latent position, shared by
//! both components.
//!
//! Ports the decoder direction of `ref/src/codec/coding_tools/quality_map/quality_map.py`
//! (`decode`, `quantize_scale`, `dequantize_resi`, `matching_qp_to_scale{,_log}`).

use crate::container::split_threads;
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

impl QualityMap {
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
