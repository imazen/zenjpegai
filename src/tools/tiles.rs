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

use alloc::vec;
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

/// One RDLR optimization tile — `tiling.py`'s `TileManager` for `process_y`'s
/// encode-side tiling (`setup_tiles_enc` + `_init_image_tiles_with_overlap` +
/// `init_latent_tiles` + `get_core_of_overlapping_*_tile`).
#[derive(Clone, Copy, Debug)]
pub struct RdlrTile {
    /// `image_tile_with_overlap` — the image-space tile (image samples).
    pub image: Area,
    /// `latent_tile_with_overlap` — the matching latent tile (latent units, /16).
    pub latent: Area,
    /// The matching z tile (z units, /64).
    pub z: Area,
    /// `optimized_part_of_imgtile_rel_to_overlap` — the image core in
    /// tile-relative coords; the region `ms_ssim`/`psnr` evaluate.
    pub core_img_rel: Area,
    /// `optimized_part_of_lattile` — the latent core in absolute latent
    /// coords; where the refined latent is written back.
    pub core_lat: Area,
    /// `optimized_part_of_lattile_rel_to_overlap` — the latent core in
    /// tile-relative coords; the region the optimiser's leaf covers.
    pub core_lat_rel: Area,
}

/// `TileManager.setup_tiles_enc` for RDLR's `process_y`: `numSamplesPerTile`
/// samples per image tile (`numSamplesPer{Luma,Chroma}Tile`), `overlap` image
/// samples (`numSamplesTileOverlap{Y,UV}`), tiles aligned to `alignment`
/// (`tool.get_alignment_size()`), `minimum_tile_size` the MS-SSIM floor
/// (`tile_min_size_for_MSSSIM + overlap`, active only when the MS-SSIM loss is
/// mixed in). Tiles with overlap at image/latent(/z) resolution, each with the
/// "optimized part" (overlap-free core) in absolute and tile-relative coords.
///
/// Disabled (`numSamplesPerTile >= img_h * img_w`, or `-1` upstream — pass
/// `usize::MAX`) → a single whole-picture tile, identical to the untiled path.
/// `latent_{h,w}`/`z_{h,w}` are the latent / hyper-latent dims (`/16`, `/64` —
/// `latent_downscale_factor_{y,z}`); `z_{h,w}` may be `None` when the caller
/// does not need `z` areas (returns `z` covering nothing then).
#[allow(clippy::too_many_arguments)]
pub fn rdlr_tiles(
    img_h: usize,
    img_w: usize,
    latent_h: usize,
    latent_w: usize,
    z_h: usize,
    z_w: usize,
    num_samples_per_tile: usize,
    overlap: usize,
    alignment: usize,
    minimum_tile_size: Option<usize>,
) -> Result<Vec<RdlrTile>> {
    let enabled = num_samples_per_tile != usize::MAX && num_samples_per_tile < img_h * img_w;
    let (tile_h, tile_w) = if enabled {
        let ts = (num_samples_per_tile as f64).sqrt().floor() as usize;
        let ts = ts.div_ceil(alignment) * alignment;
        (ts.min(img_h), ts.min(img_w))
    } else {
        (img_h, img_w)
    };
    // `_init_image_tiles_with_overlap`: strides of `tile - overlap`, each tile
    // `min(tile, img - start)`.
    let step_y = tile_h - overlap;
    let step_x = tile_w - overlap;
    let mut grid: Vec<Vec<Area>> = Vec::new();
    let mut sy = 0;
    while sy < img_h.saturating_sub(overlap) {
        let mut row = Vec::new();
        let mut sx = 0;
        while sx < img_w.saturating_sub(overlap) {
            row.push(Area::new(
                sx,
                sy,
                tile_w.min(img_w - sx),
                tile_h.min(img_h - sy),
            ));
            sx += step_x;
        }
        grid.push(row);
        sy += step_y;
    }
    if grid.is_empty() || grid[0].is_empty() {
        // `img <= overlap` on an axis: upstream's `range(0, img - overlap, step)`
        // produces an empty grid — its RDLR is a silent no-op on such a
        // picture. Our contract returns the single whole-picture tile instead
        // (all borders, so its core is the whole picture).
        grid = vec![vec![Area::new(0, 0, img_w, img_h)]];
    }
    let (rows, cols) = (grid.len(), grid[0].len());
    if let Some(min) = minimum_tile_size {
        // `_adjust_boundary_tiles`: a last row/column shorter than `min` takes
        // `min` from its neighbour. `minimum_tile_size` is a multiple of
        // `alignment` upstream (asserted); `min` overlaps the same way.
        if min % alignment != 0 {
            return Err(Error::InvalidArgument("minimum_tile_size unaligned"));
        }
        let last = grid[rows - 1][cols - 1];
        if cols >= 2 && last.width < min {
            for row in grid.iter_mut() {
                row[cols - 2].width -= min;
                row[cols - 1].x -= min;
                row[cols - 1].width += min;
            }
        }
        if rows >= 2 && last.height < min {
            for t in grid[rows - 2].iter_mut() {
                t.height -= min;
            }
            for t in grid[rows - 1].iter_mut() {
                t.y -= min;
                t.height += min;
            }
        }
    }
    // `init_latent_tiles`/`_get_latent_tile_from_image_tile`: position by
    // floor-division, size by `ceil(size / f)` for non-aligned (border) tiles.
    // `alignment` is the component's latent downscale factor (`latent_downscale_factor`
    // — 16 luma, 8 chroma); `z` is a further 4x (`/64` luma, `/32` chroma).
    let map = |t: &Area, f: usize, lh: usize, lw: usize| {
        let (x, y) = (t.x / f, t.y / f);
        let h = if t.height.is_multiple_of(f) {
            t.height / f
        } else {
            t.height.div_ceil(f)
        };
        let w = if t.width.is_multiple_of(f) {
            t.width / f
        } else {
            t.width.div_ceil(f)
        };
        Area::new(x, y, w.min(lw - x), h.min(lh - y))
    };
    // `_get_core_of_overlapping_tile` (non-region): `overlap // 2 // align`
    // from each side that isn't on the image border.
    let core = |tile: &Area, img: &Area, align: usize, rel: bool| -> Area {
        let o = overlap / 2 / align;
        let (ol, or_) = (
            if img.x == 0 { 0 } else { o },
            if img.x + img.width >= img_w { 0 } else { o },
        );
        let (ot, ob) = (
            if img.y == 0 { 0 } else { o },
            if img.y + img.height >= img_h { 0 } else { o },
        );
        if rel {
            Area::new(ol, ot, tile.width - ol - or_, tile.height - ot - ob)
        } else {
            Area::new(
                tile.x + ol,
                tile.y + ot,
                tile.width - ol - or_,
                tile.height - ot - ob,
            )
        }
    };
    let mut tiles = Vec::with_capacity(rows * cols);
    for row in &grid {
        for image in row {
            let latent = map(image, alignment, latent_h, latent_w);
            tiles.push(RdlrTile {
                image: *image,
                latent,
                z: map(image, alignment * 4, z_h, z_w),
                core_img_rel: core(image, image, 1, true),
                core_lat: core(&latent, image, alignment, false),
                core_lat_rel: core(&latent, image, alignment, true),
            });
        }
    }
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `setup_tiles_enc` + `_init_image_tiles_with_overlap` + `init_latent_tiles` +
    /// `get_core_of_overlapping_*_tile` for a 2400x2400 picture at `numSamplesPerTile`
    /// = 1 MP (tile 1024, overlap 64, alignment 16): strides of 960 → a 3x3 grid,
    /// last column/row 480 wide/high.
    #[test]
    fn rdlr_tiles_grid_and_cores() {
        let t = rdlr_tiles(2400, 2400, 150, 150, 38, 38, 1 << 20, 64, 16, None).unwrap();
        assert_eq!(t.len(), 9);
        // image tiles: 1024-tiles on a 960 stride; edges clipped.
        assert_eq!(t[0].image, Area::new(0, 0, 1024, 1024));
        assert_eq!(t[1].image, Area::new(960, 0, 1024, 1024));
        assert_eq!(t[2].image, Area::new(1920, 0, 480, 1024));
        assert_eq!(t[6].image, Area::new(0, 1920, 1024, 480));
        assert_eq!(t[8].image, Area::new(1920, 1920, 480, 480));
        // latent tiles: image tile / 16 (with the ceil rule for the 480 edge).
        assert_eq!(t[0].latent, Area::new(0, 0, 64, 64));
        assert_eq!(t[1].latent, Area::new(60, 0, 64, 64));
        assert_eq!(t[2].latent, Area::new(120, 0, 30, 64));
        assert_eq!(t[8].latent, Area::new(120, 120, 30, 30));
        // z tiles: image tile / 64.
        assert_eq!(t[0].z, Area::new(0, 0, 16, 16));
        assert_eq!(t[2].z, Area::new(30, 0, 8, 16));
        // cores: `overlap/2` (32 img, 2 lat) from each non-border side.
        assert_eq!(t[0].core_img_rel, Area::new(0, 0, 992, 992));
        assert_eq!(t[8].core_img_rel, Area::new(32, 32, 448, 448));
        assert_eq!(t[4].core_img_rel, Area::new(32, 32, 960, 960));
        assert_eq!(t[0].core_lat_rel, Area::new(0, 0, 62, 62));
        assert_eq!(t[0].core_lat, Area::new(0, 0, 62, 62));
        assert_eq!(t[8].core_lat_rel, Area::new(2, 2, 28, 28));
        assert_eq!(t[8].core_lat, Area::new(122, 122, 28, 28));
        // cores partition the picture / latent (union area covers everything).
        let img_cov: usize = t
            .iter()
            .map(|t| {
                let a = core_abs(&t.image, &t.core_img_rel);
                a.width * a.height
            })
            .sum();
        assert_eq!(img_cov, 2400 * 2400);
        let lat_cov: usize = t.iter().map(|t| t.core_lat.width * t.core_lat.height).sum();
        assert_eq!(lat_cov, 150 * 150);
    }

    /// `_adjust_boundary_tiles` (`minimum_tile_size = 176 + 64 = 240`): a last
    /// column/row under 240 takes 240 from its neighbour. 1024-tile grid on
    /// 1100x1100 → last col/row 140 < 240 → adjusted.
    #[test]
    fn rdlr_tiles_boundary_adjust() {
        let t = rdlr_tiles(1100, 1100, 69, 69, 18, 18, 1 << 20, 64, 16, Some(240)).unwrap();
        assert_eq!(t.len(), 4);
        // last column: 140 < 240 → width 140+240, x 960-240=720; col 0 shrinks.
        // last row likewise: col adjust then row adjust (both fire upstream).
        assert_eq!(t[1].image, Area::new(720, 0, 380, 784));
        assert_eq!(t[0].image, Area::new(0, 0, 784, 784));
        assert_eq!(t[2].image, Area::new(0, 720, 784, 380));
        assert_eq!(t[3].image, Area::new(720, 720, 380, 380));
    }

    /// Disabled (`numSamplesPerTile >= img` or usize::MAX) → a single
    /// whole-picture tile with full-coverage cores.
    #[test]
    fn rdlr_tiles_disabled_is_whole() {
        for n in [usize::MAX, 1 << 20] {
            let t = rdlr_tiles(888, 560, 56, 35, 14, 9, n, 64, 16, None).unwrap();
            assert_eq!(t.len(), 1);
            assert_eq!(t[0].image, Area::new(0, 0, 560, 888));
            assert_eq!(t[0].latent, Area::new(0, 0, 35, 56));
            assert_eq!(t[0].core_img_rel, Area::new(0, 0, 560, 888));
            assert_eq!(t[0].core_lat, t[0].latent);
            assert_eq!(t[0].core_lat_rel, t[0].latent);
        }
    }

    fn core_abs(image: &Area, rel: &Area) -> Area {
        Area::new(image.x + rel.x, image.y + rel.y, rel.width, rel.height)
    }

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
