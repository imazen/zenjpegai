//! Synthesis tiling: large pictures are reconstructed tile by tile, with overlapping tiles whose
//! overlap halves are discarded.
//!
//! Ports the parts of `ref/src/codec/coding_tools/tiling/tiling.py::TileManager` the decoder's
//! synthesis stage uses: `_init_image_tiles_with_overlap`, `_init_image_tiles` (without region
//! overlap), `_get_latent_tile_from_image_tile` and the picture-border branch of
//! `_get_core_of_overlapping_tile`.
//!
//! **Not ported:** the independent-region variant (tiles grown inside their region by
//! `_add_overlap`, cores cut by the region branch of `_get_core_of_overlapping_tile`).

use alloc::vec::Vec;

use super::regions::Area;
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
    // The reference iterates `range(0, dim - overlap, tile - overlap)`: a non-positive step or
    // an empty range makes it fail, so such headers are malformed.
    if tile_h <= overlap || tile_w <= overlap {
        return Err(Error::InvalidData(
            "synthesis tile not larger than its overlap",
        ));
    }
    let half = overlap / 2;
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
        let t = synthesis_tiles(888, 560, 56, 35, None).unwrap();
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
        let t = synthesis_tiles(1400, 2096, 88, 131, Some(tiling)).unwrap();
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

    #[test]
    fn rejects_degenerate_headers() {
        let tiling = SynthesisTiling {
            tile_size: 64,
            overlap: 64,
        };
        assert!(synthesis_tiles(1400, 2096, 88, 131, Some(tiling)).is_err());
    }
}
