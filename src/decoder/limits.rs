//! Resource limits and the decoder's memory model.
//!
//! The model is calibrated on heaptrack measurements (`benchmarks/memory_2026-09-17.tsv`,
//! recycling off) and checked against the tracked-heap ledger (`src/mem.rs`,
//! `benchmarks/memory_tracked_*.tsv`): 560x888 and 2096x1400 streams at every operating point,
//! all 8-bit 4:4:4 without post-filters, the larger picture on 1024-sample synthesis tiles
//! decoded on a 32-thread pool. It reproduces those points with 20-50 % headroom; other sizes
//! are predictions of the model, not measurements. `live_bytes` is additionally floored at the
//! cold model-load peak of the stream's `model_id` (65-76 MiB tracked, margined), which a
//! fresh decoder transiently holds while packing the networks. `tests/memory_ref.rs` asserts
//! the estimate covers the tracked peak of every reference stream.

use crate::error::{Error, Result};
use crate::header::{OperatingPoint, PictureHeader};

/// Caller-set bounds on what a decode may consume. `None` means unlimited.
///
/// Every bound is checked against the picture header before any network runs or any
/// picture-sized buffer is allocated; a violation is [`Error::LimitExceeded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// Most luma samples (coded width x height).
    pub max_pixels: Option<u64>,
    /// Largest coded width.
    pub max_width: Option<u32>,
    /// Largest coded height.
    pub max_height: Option<u32>,
    /// Most bytes of codestream accepted.
    pub max_input_bytes: Option<u64>,
    /// Most heap the decode may hold at once, by the estimate of [`estimate_memory`]. The
    /// decoder also shrinks the process-wide buffer pool
    /// ([`crate::nn::fast::set_pool_limit`]) so that recycled buffers fit under this bound too.
    pub max_memory_bytes: Option<u64>,
}

impl Limits {
    /// Default pixel bound: 120 megapixels, the same as the other zen codecs.
    pub const DEFAULT_MAX_PIXELS: u64 = 120_000_000;
    /// Default memory bound: 4 GiB, the same as the other zen codecs.
    pub const DEFAULT_MAX_MEMORY: u64 = 4 << 30;

    /// No bounds at all. Only for trusted input.
    pub const fn none() -> Self {
        Self {
            max_pixels: None,
            max_width: None,
            max_height: None,
            max_input_bytes: None,
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

    /// Bound the codestream size.
    pub const fn with_max_input_bytes(mut self, bytes: u64) -> Self {
        self.max_input_bytes = Some(bytes);
        self
    }

    /// Bound the estimated peak heap use.
    pub const fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Check the codestream size.
    pub fn check_input(&self, len: usize) -> Result<()> {
        match self.max_input_bytes {
            Some(max) if len as u64 > max => Err(Error::LimitExceeded(
                "codestream larger than max_input_bytes",
            )),
            _ => Ok(()),
        }
    }

    /// Check a picture header against every bound; returns the memory estimate it was judged by.
    pub fn check_header(&self, hdr: &PictureHeader, op: OperatingPoint) -> Result<MemoryEstimate> {
        if self.max_width.is_some_and(|m| hdr.width > m) {
            return Err(Error::LimitExceeded("picture wider than max_width"));
        }
        if self.max_height.is_some_and(|m| hdr.height > m) {
            return Err(Error::LimitExceeded("picture taller than max_height"));
        }
        let pixels = hdr.width as u64 * hdr.height as u64;
        if self.max_pixels.is_some_and(|m| pixels > m) {
            return Err(Error::LimitExceeded(
                "picture has more than max_pixels samples",
            ));
        }
        let est = estimate_memory(hdr, op);
        if self.max_memory_bytes.is_some_and(|m| est.live_bytes > m) {
            return Err(Error::LimitExceeded(
                "estimated decode memory exceeds max_memory_bytes",
            ));
        }
        Ok(est)
    }
}

impl Default for Limits {
    /// [`Self::DEFAULT_MAX_PIXELS`] and [`Self::DEFAULT_MAX_MEMORY`]; nothing else bounded.
    fn default() -> Self {
        Self::none()
            .with_max_pixels(Self::DEFAULT_MAX_PIXELS)
            .with_max_memory(Self::DEFAULT_MAX_MEMORY)
    }
}

/// Predicted heap use of one decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryEstimate {
    /// Peak of the memory the decode itself holds: packed networks, entropy-stage tensors,
    /// latents, the feature maps of the largest synthesis tile, output planes and the result.
    pub live_bytes: u64,
    /// Upper bound of what the recycled-buffer pool may hold on top of that: its limit at the
    /// time of the call ([`crate::nn::fast::pool_limit`], 1 GiB by default).
    pub pool_bytes: u64,
}

impl MemoryEstimate {
    /// `live_bytes + pool_bytes`: upper bound of the heap held while decoding.
    pub const fn peak_bytes(&self) -> u64 {
        self.live_bytes.saturating_add(self.pool_bytes)
    }
}

/// Estimate the heap one decode of a picture with header `hdr` needs when synthesised with `op`.
///
/// `live = fixed(op) + 20 B x picture samples + k(op) x samples of the largest synthesis tile
/// + e(op) x samples of the other tiles that synthesise concurrently` — tiles map onto the
/// rayon pool, so a tiled picture holds up to `threads` tiles' working sets and every tile's
/// buffered output at once. `k` = 36/92/970 bytes per largest-tile sample and `e` = 36/60/600
/// per *extra* concurrently-scheduled tile sample for SOP/BOP/HOP — an extra tile's steady
/// working set, calibrated to the observed 32-thread peak + ~30 % (the concurrent peak is
/// schedule-dependent; that margin is the best fixed bound short of serialising tiles). An
/// untiled picture is one tile and pays no `e` term. `fixed` covers the packed networks
///   resident during the decode.
///
/// On top of that, `live` is at least `MODEL_LOAD_PEAK[model_id]`: a cold decoder loads and
/// packs the networks inside the decode, which transiently holds the checkpoint bytes plus
/// their parsed and packed copies — tracked peaks of ~65 MiB for the small-β models
/// (ids 0-1) and ~76 MiB for the large ones (ids 2-3, whose common checkpoints are
/// 50-61 MiB files). A decoder whose cache is already warm doesn't pay this, but the
/// estimate cannot know the cache state, so it covers the cold case.
pub fn estimate_memory(hdr: &PictureHeader, op: OperatingPoint) -> MemoryEstimate {
    /// Cold-start model-load peaks per `model_id` (β 0.002/0.012/0.075/0.5), measured
    /// tracked + ~13 % margin.
    const MODEL_LOAD_PEAK: [u64; 4] = [70 << 20, 70 << 20, 84 << 20, 84 << 20];
    const PER_PICTURE_SAMPLE: u64 = 20;
    let (fixed, per_tile_sample, extra_tile_sample): (u64, u64, u64) = match op {
        OperatingPoint::Sop => (32 << 20, 36, 36),
        OperatingPoint::Bop => (32 << 20, 92, 60),
        OperatingPoint::Hop => (44 << 20, 970, 600),
    };
    let (w, h) = (hdr.width as u64, hdr.height as u64);
    // Tile areas the way the decoder computes them (region-aware tiling only shrinks tiles,
    // so the region-free grid is the conservative case; a malformed tiling falls back to one
    // whole-picture tile, which is what the decoder's own error would face).
    let (lat_h, lat_w) = hdr.latent_size(0);
    let areas = crate::tools::tiles::synthesis_tiles(
        h as usize,
        w as usize,
        lat_h as usize,
        lat_w as usize,
        hdr.components[0].synthesis_tiling,
        None,
    )
    .map(|ts| {
        ts.iter()
            .map(|t| t.image.width as u64 * t.image.height as u64)
            .collect::<alloc::vec::Vec<u64>>()
    })
    .unwrap_or_else(|_| alloc::vec![w * h]);
    let largest = areas.iter().copied().max().unwrap_or(0);
    let total: u64 = areas.iter().sum();
    // How many tile working sets can be live at once: the rayon pool's width, capped by the
    // tile count. Without the `parallel` feature (or a 1-thread pool) tiles are sequential.
    #[cfg(feature = "parallel")]
    let par = rayon::current_num_threads() as u64;
    #[cfg(not(feature = "parallel"))]
    let par = 1u64;
    let extra = (total - largest).min(largest.saturating_mul(par.saturating_sub(1)));
    let model_load = MODEL_LOAD_PEAK
        .get(hdr.model_id as usize)
        .copied()
        .unwrap_or(MODEL_LOAD_PEAK[MODEL_LOAD_PEAK.len() - 1]);
    let live_bytes = fixed
        .saturating_add(PER_PICTURE_SAMPLE.saturating_mul(w * h))
        .saturating_add(per_tile_sample.saturating_mul(largest))
        .saturating_add(extra_tile_sample.saturating_mul(extra))
        .max(model_load);
    MemoryEstimate {
        live_bytes,
        pool_bytes: crate::nn::fast::pool_limit() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PIH of a 560x888 8-bit 4:4:4 stream (BOP, SOP), cut from a reference-encoder stream.
    fn header() -> PictureHeader {
        let bytes: alloc::vec::Vec<u8> = "011103401f00338000008a32c5001800"
            .as_bytes()
            .chunks(2)
            .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        PictureHeader::parse(&bytes).unwrap()
    }

    #[test]
    fn header_is_judged_against_every_bound() {
        let hdr = header();
        let op = OperatingPoint::Bop;
        assert!(Limits::default().check_header(&hdr, op).is_ok());
        assert!(Limits::none().check_header(&hdr, op).is_ok());
        let px = 560 * 888;
        assert!(
            Limits::none()
                .with_max_pixels(px)
                .check_header(&hdr, op)
                .is_ok()
        );
        for l in [
            Limits::none().with_max_pixels(px - 1),
            Limits::none().with_max_dimensions(559, 888),
            Limits::none().with_max_dimensions(560, 887),
            Limits::none().with_max_memory(50 << 20),
        ] {
            assert!(matches!(
                l.check_header(&hdr, op),
                Err(Error::LimitExceeded(_))
            ));
        }
        assert!(
            Limits::none()
                .with_max_input_bytes(10)
                .check_input(10)
                .is_ok()
        );
        assert!(
            Limits::none()
                .with_max_input_bytes(10)
                .check_input(11)
                .is_err()
        );
    }

    #[test]
    fn estimate_orders_operating_points_and_honours_tiling() {
        let mut hdr = header();
        let e = |h: &PictureHeader, op| estimate_memory(h, op).live_bytes;
        assert!(e(&hdr, OperatingPoint::Sop) < e(&hdr, OperatingPoint::Bop));
        assert!(e(&hdr, OperatingPoint::Bop) < e(&hdr, OperatingPoint::Hop));
        let untiled = e(&hdr, OperatingPoint::Hop);
        // A few big tiles still beats one whole-picture working set; many small ones do not —
        // they synthesise concurrently, each with its own working set, which the estimate
        // counts. Both directions are the point: tiling does not simply divide memory.
        hdr.components[0].synthesis_tiling = Some(crate::header::SynthesisTiling {
            tile_size: 512,
            overlap: 64,
        });
        assert!(e(&hdr, OperatingPoint::Hop) < untiled);
        hdr.components[0].synthesis_tiling = Some(crate::header::SynthesisTiling {
            tile_size: 128,
            overlap: 64,
        });
        assert!(e(&hdr, OperatingPoint::Hop) > e(&hdr, OperatingPoint::Sop));
    }
}
