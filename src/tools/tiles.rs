//! Synthesis tiling: large pictures are reconstructed tile by tile, with overlapping tiles whose
//! overlap halves are discarded.
//!
//! Ports the parts of `ref/src/codec/coding_tools/tiling/tiling.py::TileManager` the decoder's
//! synthesis stage uses: `_init_image_tiles_with_overlap`, `_init_image_tiles` (without region
//! overlap), `_get_latent_tile_from_image_tile` and the picture-border branch of
//! `_get_core_of_overlapping_tile`.
//!
//! With independent regions (`region_residual_in_its_own_substream_flag`) tiles start on a plain
//! grid, are grown by half the overlap towards neighbours *inside their region* (`_add_overlap`)
//! and lose exactly that growth again (region branch of `_get_core_of_overlapping_tile`), so no
//! tile reads latents of another region.

use alloc::vec::Vec;

use super::regions::{Area, RegionGrid};
use crate::error::{Error, Result};
use crate::header::SynthesisTiling;

/// Latent samples per picture sample along one axis, for both components (chroma is synthesised
/// at luma resolution).
pub const SYNTHESIS_DOWNSCALE: usize = 16;

/// One synthesis tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynthesisTile {
    /// Picture area the tile reconstructs, overlap included.
    pub image: Area,
    /// Latent (`y_hat`) area feeding it.
    pub latent: Area,
    /// Part of `image` that is kept, in picture coordinates.
    pub core: Area,
    /// Offset of `core` inside the tile's own output.
    pub core_offset: (usize, usize),
}

/// Tile layout of a `height x width` picture whose latent is `latent_h x latent_w`.
/// `tiling == None` (tiling disabled) yields the single whole-picture tile.
pub fn synthesis_tiles(
    height: usize,
    width: usize,
    latent_h: usize,
    latent_w: usize,
    tiling: Option<SynthesisTiling>,
    independent_regions: Option<&RegionGrid>,
) -> Result<Vec<SynthesisTile>> {
    let Some(t) = tiling else {
        let image = Area::new(0, 0, width, height);
        return Ok(alloc::vec![SynthesisTile {
            image,
            latent: Area::new(0, 0, latent_w, latent_h),
            core: image,
            core_offset: (0, 0),
        }]);
    };
    let (size, overlap) = (t.tile_size as usize, t.overlap as usize);
    let (tile_h, tile_w) = (size.min(height), size.min(width));
    let half = overlap / 2;
    if let Some(regions) = independent_regions {
        return region_tiles(
            height,
            width,
            latent_h,
            latent_w,
            (tile_h, tile_w),
            half,
            regions,
        );
    }
    // The reference iterates `range(0, dim - overlap, tile - overlap)`: a non-positive step or
    // an empty range makes it fail, so such headers are malformed.
    if tile_h <= overlap || tile_w <= overlap {
        return Err(Error::InvalidData(
            "synthesis tile not larger than its overlap",
        ));
    }
    let mut tiles = Vec::new();
    for y in (0..height - overlap).step_by(tile_h - overlap) {
        for x in (0..width - overlap).step_by(tile_w - overlap) {
            let image = Area::new(x, y, tile_w.min(width - x), tile_h.min(height - y));
            let latent = Area::new(
                x / SYNTHESIS_DOWNSCALE,
                y / SYNTHESIS_DOWNSCALE,
                latent_extent(image.x, image.width, latent_w)?,
                latent_extent(image.y, image.height, latent_h)?,
            );
            if latent.x + latent.width > latent_w || latent.y + latent.height > latent_h {
                return Err(Error::InvalidData("synthesis tile outside the latent"));
            }
            let left = if image.x == 0 { 0 } else { half };
            let top = if image.y == 0 { 0 } else { half };
            let right = if image.x + image.width >= width {
                0
            } else {
                half
            };
            let bottom = if image.y + image.height >= height {
                0
            } else {
                half
            };
            if image.width <= left + right || image.height <= top + bottom {
                return Err(Error::InvalidData("synthesis tile is all overlap"));
            }
            tiles.push(SynthesisTile {
                image,
                latent,
                core: Area::new(
                    image.x + left,
                    image.y + top,
                    image.width - left - right,
                    image.height - top - bottom,
                ),
                core_offset: (left, top),
            });
        }
    }
    Ok(tiles)
}

fn contains(r: &Area, x: usize, y: usize) -> bool {
    r.x <= x && x < r.x + r.width && r.y <= y && y < r.y + r.height
}

/// `_init_image_tiles` + `_add_overlap`, and the region branch of
/// `_get_core_of_overlapping_tile`. `regions` are picture areas of the (unextended) regions.
fn region_tiles(
    height: usize,
    width: usize,
    latent_h: usize,
    latent_w: usize,
    (tile_h, tile_w): (usize, usize),
    half: usize,
    regions: &RegionGrid,
) -> Result<Vec<SynthesisTile>> {
    let mut tiles = Vec::new();
    for ty in (0..height).step_by(tile_h) {
        for tx in (0..width).step_by(tile_w) {
            let (mut x, mut y) = (tx, ty);
            let (mut w, mut h) = (tile_w.min(width - tx), tile_h.min(height - ty));
            let (x_end, y_end) = (x + w - 1, y + h - 1);
            // The reference re-tests every region against the tile as grown so far.
            for r in &regions.extended {
                if !(contains(r, x, y) && contains(r, x_end, y_end)) {
                    continue;
                }
                if y > r.y {
                    y = y
                        .checked_sub(half)
                        .ok_or(Error::InvalidData("tile overlap"))?;
                    h += half;
                }
                if x > r.x {
                    x = x
                        .checked_sub(half)
                        .ok_or(Error::InvalidData("tile overlap"))?;
                    w += half;
                }
                if y_end < r.y + r.height - 1 {
                    h += half.min(height.saturating_sub(y + h));
                }
                if x_end < r.x + r.width - 1 {
                    w += half.min(width.saturating_sub(x + w));
                }
            }
            let image = Area::new(x, y, w, h);
            if x + w > width || y + h > height {
                return Err(Error::InvalidData("synthesis tile outside the picture"));
            }
            let latent = Area::new(
                x / SYNTHESIS_DOWNSCALE,
                y / SYNTHESIS_DOWNSCALE,
                latent_extent(x, w, latent_w)?,
                latent_extent(y, h, latent_h)?,
            );
            if latent.x + latent.width > latent_w || latent.y + latent.height > latent_h {
                return Err(Error::InvalidData("synthesis tile outside the latent"));
            }

            let (mut cx, mut cy, mut cw, mut ch) = (x, y, w, h);
            let (mut left, mut top) = (0, 0);
            let (bx, by) = (x + w - 1, y + h - 1);
            for r in &regions.extended {
                if !(contains(r, cx, cy) && contains(r, bx, by)) {
                    continue;
                }
                if cy > r.y {
                    top += half;
                    cy += half;
                    ch = ch
                        .checked_sub(half)
                        .ok_or(Error::InvalidData("tile overlap"))?;
                }
                if cx > r.x {
                    left += half;
                    cx += half;
                    cw = cw
                        .checked_sub(half)
                        .ok_or(Error::InvalidData("tile overlap"))?;
                }
                ch = ch.min(tile_h);
                cw = cw.min(tile_w);
            }
            if cw == 0 || ch == 0 || left + cw > w || top + ch > h {
                return Err(Error::InvalidData("synthesis tile is all overlap"));
            }
            tiles.push(SynthesisTile {
                image,
                latent,
                core: Area::new(cx, cy, cw, ch),
                core_offset: (left, top),
            });
        }
    }
    Ok(tiles)
}

/// `_get_latent_tile_from_image_tile` for one axis: a tile whose size is not a multiple of the
/// downscale factor must end at the picture border.
fn latent_extent(pos: usize, len: usize, latent_len: usize) -> Result<usize> {
    if !len.is_multiple_of(SYNTHESIS_DOWNSCALE)
        && (pos + len).div_ceil(SYNTHESIS_DOWNSCALE) < latent_len
    {
        return Err(Error::InvalidData(
            "unaligned synthesis tile inside the picture",
        ));
    }
    Ok(len.div_ceil(SYNTHESIS_DOWNSCALE))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_is_one_tile() {
        let t = synthesis_tiles(888, 560, 56, 35, None, None).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].core, Area::new(0, 0, 560, 888));
    }

    /// Layout the reference logs for a 2096x1400 picture, tile 1024, overlap 64:
    /// `Synthesis using these tiles for Luma on image level`.
    #[test]
    fn matches_reference_layout() {
        let tiling = SynthesisTiling {
            tile_size: 1024,
            overlap: 64,
        };
        let t = synthesis_tiles(1400, 2096, 88, 131, Some(tiling), None).unwrap();
        let images: Vec<Area> = t.iter().map(|t| t.image).collect();
        assert_eq!(
            images,
            [
                Area::new(0, 0, 1024, 1024),
                Area::new(960, 0, 1024, 1024),
                Area::new(1920, 0, 176, 1024),
                Area::new(0, 960, 1024, 440),
                Area::new(960, 960, 1024, 440),
                Area::new(1920, 960, 176, 440),
            ]
        );
        assert_eq!(t[1].latent, Area::new(60, 0, 64, 64));
        assert_eq!(t[5].latent, Area::new(120, 60, 11, 28));
        assert_eq!(t[1].core, Area::new(992, 0, 960, 992));
        assert_eq!(t[1].core_offset, (32, 0));
        assert_eq!(t[5].core, Area::new(1952, 992, 144, 408));
        // Cores partition the picture.
        let covered: usize = t.iter().map(|t| t.core.width * t.core.height).sum();
        assert_eq!(covered, 1400 * 2096);
    }

    /// Reference log for 2096x1400, two independent regions side by side (split at x = 1024),
    /// tile 640, overlap 128.
    #[test]
    fn matches_reference_layout_with_independent_regions() {
        let regions = RegionGrid {
            rows: 1,
            columns: 2,
            core: alloc::vec![Area::new(0, 0, 1024, 1400), Area::new(1024, 0, 1072, 1400)],
            extended: alloc::vec![Area::new(0, 0, 1024, 1400), Area::new(1024, 0, 1072, 1400)],
        };
        let tiling = SynthesisTiling {
            tile_size: 640,
            overlap: 128,
        };
        let t = synthesis_tiles(1400, 2096, 88, 131, Some(tiling), Some(&regions)).unwrap();
        let images: Vec<Area> = t.iter().map(|t| t.image).collect();
        assert_eq!(
            images,
            [
                Area::new(0, 0, 704, 704),
                Area::new(640, 0, 640, 640),
                Area::new(1216, 0, 768, 704),
                Area::new(1856, 0, 240, 704),
                Area::new(0, 576, 704, 768),
                Area::new(640, 640, 640, 640),
                Area::new(1216, 576, 768, 768),
                Area::new(1856, 576, 240, 768),
                Area::new(0, 1216, 704, 184),
                Area::new(640, 1280, 640, 120),
                Area::new(1216, 1216, 768, 184),
                Area::new(1856, 1216, 240, 184),
            ]
        );
        assert_eq!(t[6].latent, Area::new(76, 36, 48, 48));
        assert_eq!(t[6].core, Area::new(1280, 640, 640, 640));
        assert_eq!(t[6].core_offset, (64, 64));
        let covered: usize = t.iter().map(|t| t.core.width * t.core.height).sum();
        assert_eq!(covered, 1400 * 2096);
    }

    #[test]
    fn rejects_degenerate_headers() {
        let tiling = SynthesisTiling {
            tile_size: 64,
            overlap: 64,
        };
        assert!(synthesis_tiles(1400, 2096, 88, 131, Some(tiling), None).is_err());
    }
}
