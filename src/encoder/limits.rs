//! Resource limits and the encoder's memory model.
//!
//! The model is calibrated on heaptrack measurements
//! (`benchmarks/memory_encode_2026-09-18.tsv`, buffer pool off): 560x888 and 2096x1400 RGB
//! sources, fixed-model and rate-matched encodes at the three operating points' common path
//! (BOP analysis; HOP is the same shape with bigger analysis networks). It reproduces those
//! points with headroom; other sizes are predictions of the model, not measurements.

use crate::decoder::limits::MemoryEstimate;
use crate::error::{Error, Result};
use crate::header::OperatingPoint;

/// Caller-set bounds on what an encode may consume. `None` means unlimited.
///
/// Every bound is checked against the source's size before any network runs or any
/// picture-sized buffer is allocated; a violation is [`Error::LimitExceeded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeLimits {
    /// Most luma samples (coded width x height).
    pub max_pixels: Option<u64>,
    /// Largest coded width.
    pub max_width: Option<u32>,
    /// Largest coded height.
    pub max_height: Option<u32>,
    /// Most heap the encode may hold at once, by the estimate of [`estimate_encode_memory`].
    /// The encoder also shrinks the process-wide buffer pool
    /// ([`crate::nn::fast::set_pool_limit`]) so that recycled buffers fit under this bound too.
    pub max_memory_bytes: Option<u64>,
}

impl EncodeLimits {
    /// Default pixel bound: 120 megapixels, the same as the decoder's.
    pub const DEFAULT_MAX_PIXELS: u64 = 120_000_000;
    /// Default memory bound: 4 GiB, the same as the decoder's.
    pub const DEFAULT_MAX_MEMORY: u64 = 4 << 30;

    /// No bounds at all. Only for trusted input.
    pub const fn none() -> Self {
        Self {
            max_pixels: None,
            max_width: None,
            max_height: None,
            max_memory_bytes: None,
        }
    }

    /// Bound the luma sample count.
    pub const fn with_max_pixels(mut self, pixels: u64) -> Self {
        self.max_pixels = Some(pixels);
        self
    }

    /// Bound width and height separately.
    pub const fn with_max_dimensions(mut self, width: u32, height: u32) -> Self {
        self.max_width = Some(width);
        self.max_height = Some(height);
        self
    }

    /// Bound the estimated peak heap use.
    pub const fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Check a `width` x `height` source against every bound and return the memory estimate it
    /// was judged by. `rate_matched` selects the estimate for [`Encoder::encode_to_bpp`]
    /// (which holds every model's weights and the best-so-far model's latents while it
    /// searches); `false` is a single fixed-model encode.
    ///
    /// [`Encoder::encode_to_bpp`]: crate::encoder::Encoder::encode_to_bpp
    pub fn check(
        &self,
        width: usize,
        height: usize,
        op: OperatingPoint,
        rate_matched: bool,
    ) -> Result<MemoryEstimate> {
        if self.max_width.is_some_and(|m| width as u64 > u64::from(m)) {
            return Err(Error::LimitExceeded("picture wider than max_width"));
        }
        if self
            .max_height
            .is_some_and(|m| height as u64 > u64::from(m))
        {
            return Err(Error::LimitExceeded("picture taller than max_height"));
        }
        let pixels = width as u64 * height as u64;
        if self.max_pixels.is_some_and(|m| pixels > m) {
            return Err(Error::LimitExceeded(
                "picture has more than max_pixels samples",
            ));
        }
        let est = estimate_encode_memory(width as u64, height as u64, op, rate_matched);
        if self.max_memory_bytes.is_some_and(|m| est.live_bytes > m) {
            return Err(Error::LimitExceeded(
                "estimated encode memory exceeds max_memory_bytes",
            ));
        }
        Ok(est)
    }
}

impl Default for EncodeLimits {
    /// [`Self::DEFAULT_MAX_PIXELS`] and [`Self::DEFAULT_MAX_MEMORY`]; nothing else bounded.
    fn default() -> Self {
        Self::none()
            .with_max_pixels(Self::DEFAULT_MAX_PIXELS)
            .with_max_memory(Self::DEFAULT_MAX_MEMORY)
    }
}

/// Estimate the heap one encode of a `width` x `height` picture needs at operating point `op`.
///
/// `live = fixed(op) + 160 B x picture samples + k(op) x samples of the largest analysis
/// tile` with `k` = 200 (SOP / BOP) or 260 (HOP) bytes per sample; an untiled picture is one
/// tile. `rate_matched` adds the other three models' packed networks plus one extra latent
/// set (the search keeps the best-so-far model's latents next to the one being measured).
/// See the module documentation for what this was measured on.
pub fn estimate_encode_memory(
    width: u64,
    height: u64,
    op: OperatingPoint,
    rate_matched: bool,
) -> MemoryEstimate {
    /// Per picture sample: the source planes (u16 RGB -> f32 Y/U/V), the preprocessed planes,
    /// quantiser displacement tables and the coded side of one component's tensors while the
    /// other component's compress runs.
    const PER_PICTURE_SAMPLE: u64 = 160;
    /// One extra model's encoder networks as rate matching holds them (common + analysis +
    /// hyper-encoder for both components): measured ~26 MB, rounded up.
    const EXTRA_MODEL: u64 = 28 << 20;
    /// One extra model's latents per picture sample: `y` 160+96 f32 channels at a 16th of the
    /// luma resolution each way plus the quantised hyper-latents (i8, four times coarser
    /// again) — measured at ~4 B/sample.
    const LATENTS_PER_SAMPLE: u64 = 4;
    /// Samples of the largest luma analysis tile (`tiles::ENC_SAMPLES_PER_TILE[0]`), the
    /// 1024-sample side the tile manager produces.
    const TILE_SIDE: u64 = 1024;
    const TILE_SAMPLES: u64 = TILE_SIDE * TILE_SIDE;
    // fixed(op): one model's packed encoder networks (common + analysis + hyper-encoder for
    // both components) plus the engine, the PNG decode and the transient checkpoint bytes of
    // the first load. SOP encodes with the BOP analysis transform (`encoder::model_set`), so
    // its fixed cost is BOP's; HOP's analysis networks are bigger (unmeasured for encode;
    // padded from the decoder side's ~1.7x HOP:BOP model ratio).
    // per_tile_sample: the analysis transform's feature maps over the largest tile.
    let (fixed, per_tile_sample): (u64, u64) = match op {
        OperatingPoint::Sop | OperatingPoint::Bop => (170 << 20, 200),
        OperatingPoint::Hop => (260 << 20, 260),
    };
    let (w, h) = (width.max(1), height.max(1));
    let tile = if w * h <= TILE_SAMPLES {
        w * h
    } else {
        w.min(TILE_SIDE) * h.min(TILE_SIDE)
    };
    let mut live_bytes = fixed
        .saturating_add(PER_PICTURE_SAMPLE.saturating_mul(w * h))
        .saturating_add(per_tile_sample.saturating_mul(tile));
    if rate_matched {
        // Three more model sets than a fixed encode, plus the kept competitor's latents.
        live_bytes = live_bytes
            .saturating_add(3 * EXTRA_MODEL)
            .saturating_add(LATENTS_PER_SAMPLE.saturating_mul(w * h));
    }
    MemoryEstimate {
        live_bytes,
        pool_bytes: crate::nn::fast::pool_limit() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_is_judged_against_every_bound() {
        let (w, h) = (560, 888);
        let op = OperatingPoint::Bop;
        assert!(EncodeLimits::default().check(w, h, op, false).is_ok());
        assert!(EncodeLimits::none().check(w, h, op, true).is_ok());
        assert!(
            EncodeLimits::none()
                .with_max_pixels(w as u64 * h as u64)
                .check(w, h, op, false)
                .is_ok()
        );
        for l in [
            EncodeLimits::none().with_max_pixels(w as u64 * h as u64 - 1),
            EncodeLimits::none().with_max_dimensions(559, 888),
            EncodeLimits::none().with_max_dimensions(560, 887),
            EncodeLimits::none().with_max_memory(20 << 20),
        ] {
            for rate_matched in [false, true] {
                assert!(
                    matches!(
                        l.check(w, h, op, rate_matched),
                        Err(Error::LimitExceeded(_))
                    ),
                    "{l:?} rate_matched={rate_matched}"
                );
            }
        }
    }

    #[test]
    fn estimate_orders_operating_points_and_rate_matching() {
        let e = |w, h, op, r| estimate_encode_memory(w, h, op, r).live_bytes;
        assert!(e(560, 888, OperatingPoint::Sop, false) < e(560, 888, OperatingPoint::Hop, false));
        assert!(e(560, 888, OperatingPoint::Bop, false) < e(560, 888, OperatingPoint::Bop, true));
        assert!(e(64, 64, OperatingPoint::Bop, false) < e(2096, 1400, OperatingPoint::Bop, false));
        // Tiling caps the per-tile term: above 1 MP the slope must not grow.
        let slope = e(2 * 2096, 2 * 1400, OperatingPoint::Bop, false)
            - e(2096, 1400, OperatingPoint::Bop, false);
        assert!(slope < 4 * PER_PICTURE_SAMPLE_BOUND * 2096 * 1400);
    }

    /// The test above re-derives the per-sample bound so the constant can't drift silently.
    const PER_PICTURE_SAMPLE_BOUND: u64 = 256;
}
