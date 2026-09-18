//! Analysis tiling: above about 1 MP the reference encoder runs the analysis transform and the
//! hyper-encoder tile by tile and keeps only each tile's core.
//!
//! Ports `ref/src/codec/coding_tools/tiling/tiling.py::TileManager`
//! (`setup_tiles_enc` / `setup_tiles_dec`, `_init_image_tiles_with_overlap`,
//! `_get_latent_tile_from_image_tile`, the picture-border branch of
//! `_get_core_of_overlapping_tile`) as `sep_chan_tool.py::setup_enc_tile_managers_of_model` and
//! `common_modules.py::compress_colocated_tiles` drive it.
//!
//! The tile size is not in the bitstream: it comes from `cfg/CTC.json`, which the reference's
//! shipped configuration always loads (`numSamplesPerTile` / `numSamplesTileOverlap` on
//! `model_y.tile_manager_enc` and `model_uv.tile_manager_enc`). `_adjust_boundary_tiles` does
//! not run on this tile manager (`setup_tiles_enc` is called without a minimum tile size).

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::tools::regions::Area;

/// `cfg/CTC.json`, per component: `numSamplesPerTile` and `numSamplesTileOverlap` of
/// `tile_manager_enc`, and the alignment size (= the latent downscale factor).
pub const ENC_SAMPLES_PER_TILE: [usize; 2] = [1_048_576, 262_144];
pub const ENC_TILE_OVERLAP: [usize; 2] = [64, 32];
/// Component plane samples per latent sample, and per hyper-latent sample.
pub const LATENT_DOWNSCALE: [usize; 2] = [16, 8];
pub const HYPER_DOWNSCALE: [usize; 2] = [64, 32];

/// `cfg/CTC.json`, `tile_manager_synthesis`: the same for both components, and computed from the
/// *luma* picture size (chroma is synthesised at luma resolution). When it is enabled the picture
/// header carries `synthesis_tile_size` / `synthesis_tile_overlap`.
pub const SYNTHESIS_SAMPLES_PER_TILE: usize = 1_048_576;
pub const SYNTHESIS_TILE_OVERLAP: u32 = 64;

/// One analysis tile of one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnalysisTile {
    /// Area of the component plane the analysis transform runs on.
    pub image: Area,
    /// Latent area it produces.
    pub latent: Area,
    /// Part of `latent` that is kept, in whole-latent coordinates.
    pub latent_core: Area,
    /// Offset of `latent_core` inside the tile's own latent.
    pub latent_core_offset: (usize, usize),
    /// Hyper-latent area the tile produces, and the part of it that is kept.
    pub hyper: Area,
    pub hyper_core: Area,
    pub hyper_core_offset: (usize, usize),
}

/// `tile_size`: `floor(sqrt(numSamplesPerTile))` rounded up to the alignment size. `None` when
/// the picture is small enough that `setup_tiles_enc` leaves tiling disabled.
fn tile_size(ccs: usize, h: usize, w: usize) -> Option<usize> {
    let n = ENC_SAMPLES_PER_TILE[ccs];
    if n >= h * w {
        return None;
    }
    let align = LATENT_DOWNSCALE[ccs];
    Some((libm::sqrt(n as f64).floor() as usize).div_ceil(align) * align)
}

/// Tiles of component `ccs` for a plane of `h x w` samples whose latent is `lh x lw` and whose
/// hyper-latent is `hz x wz`. One whole-plane tile when tiling is disabled.
pub fn analysis_tiles(
    ccs: usize,
    h: usize,
    w: usize,
    lh: usize,
    lw: usize,
    hz: usize,
    wz: usize,
) -> Result<Vec<AnalysisTile>> {
    let Some(size) = tile_size(ccs, h, w) else {
        return Ok(alloc::vec![AnalysisTile {
            image: Area::new(0, 0, w, h),
            latent: Area::new(0, 0, lw, lh),
            latent_core: Area::new(0, 0, lw, lh),
            latent_core_offset: (0, 0),
            hyper: Area::new(0, 0, wz, hz),
            hyper_core: Area::new(0, 0, wz, hz),
            hyper_core_offset: (0, 0),
        }]);
    };
    let overlap = ENC_TILE_OVERLAP[ccs];
    let (yd, zd) = (LATENT_DOWNSCALE[ccs], HYPER_DOWNSCALE[ccs]);
    let (tile_h, tile_w) = (size.min(h), size.min(w));
    if tile_h <= overlap || tile_w <= overlap {
        return Err(Error::InvalidData("analysis tile not larger than overlap"));
    }
    // `_get_core_of_overlapping_tile`: half the overlap, in units of the respective grid.
    let (yo, zo) = (overlap / 2 / yd, overlap / 2 / zd);

    let mut tiles = Vec::new();
    for y in (0..h - overlap).step_by(tile_h - overlap) {
        for x in (0..w - overlap).step_by(tile_w - overlap) {
            let image = Area::new(x, y, tile_w.min(w - x), tile_h.min(h - y));
            let grid = |down: usize, full_h: usize, full_w: usize| -> Result<Area> {
                Ok(Area::new(
                    image.x / down,
                    image.y / down,
                    extent(image.x, image.width, down, full_w)?,
                    extent(image.y, image.height, down, full_h)?,
                ))
            };
            let latent = grid(yd, lh, lw)?;
            let hyper = grid(zd, hz, wz)?;
            let core = |a: Area, o: usize| -> Result<(Area, (usize, usize))> {
                let left = if image.x == 0 { 0 } else { o };
                let top = if image.y == 0 { 0 } else { o };
                let right = if image.x + image.width >= w { 0 } else { o };
                let bottom = if image.y + image.height >= h { 0 } else { o };
                let (cw, ch) = (
                    a.width.checked_sub(left + right),
                    a.height.checked_sub(top + bottom),
                );
                let (Some(cw), Some(ch)) = (cw, ch) else {
                    return Err(Error::InvalidData("analysis tile is all overlap"));
                };
                Ok((Area::new(a.x + left, a.y + top, cw, ch), (left, top)))
            };
            let (latent_core, latent_core_offset) = core(latent, yo)?;
            let (hyper_core, hyper_core_offset) = core(hyper, zo)?;
            tiles.push(AnalysisTile {
                image,
                latent,
                latent_core,
                latent_core_offset,
                hyper,
                hyper_core,
                hyper_core_offset,
            });
        }
    }
    Ok(tiles)
}

/// `_get_latent_tile_from_image_tile` for one axis: a tile whose size is not a multiple of the
/// downscale factor must end at the plane border.
fn extent(pos: usize, len: usize, down: usize, full: usize) -> Result<usize> {
    if !len.is_multiple_of(down) && (pos + len).div_ceil(down) < full {
        return Err(Error::InvalidData(
            "unaligned analysis tile inside the plane",
        ));
    }
    Ok(len.div_ceil(down))
}

/// `tile_manager_synthesis` of both components, computed from the coded luma picture size: the
/// `synthesis_tiling` the picture header must carry, or `None`.
pub fn synthesis_tiling(height: usize, width: usize) -> Option<crate::header::SynthesisTiling> {
    if SYNTHESIS_SAMPLES_PER_TILE >= height * width {
        return None;
    }
    let size = libm::sqrt(SYNTHESIS_SAMPLES_PER_TILE as f64).floor() as u32;
    Some(crate::header::SynthesisTiling {
        tile_size: size.div_ceil(16) * 16,
        overlap: SYNTHESIS_TILE_OVERLAP,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout the reference logs for 2096x1400 (`enc_img01_bop_m1_b0/encoder.log`,
    /// "Analysis transforms (y and z) using these tiles on image level").
    #[test]
    fn matches_reference_layout() {
        let luma = analysis_tiles(0, 1400, 2096, 88, 131, 22, 33).unwrap();
        let images: Vec<Area> = luma.iter().map(|t| t.image).collect();
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
        let latents: Vec<Area> = luma.iter().map(|t| t.latent).collect();
        assert_eq!(latents[1], Area::new(60, 0, 64, 64));
        assert_eq!(latents[5], Area::new(120, 60, 11, 28));
        let hyper: Vec<Area> = luma.iter().map(|t| t.hyper).collect();
        assert_eq!(hyper[1], Area::new(15, 0, 16, 16));
        assert_eq!(hyper[5], Area::new(30, 15, 3, 7));
        // Latent cores partition the latent; hyper cores are whole tiles (64 / 2 / 64 == 0).
        let covered: usize = luma
            .iter()
            .map(|t| t.latent_core.width * t.latent_core.height)
            .sum();
        assert_eq!(covered, 88 * 131);
        assert_eq!(luma[0].latent_core, Area::new(0, 0, 62, 62));
        assert_eq!(luma[1].latent_core_offset, (2, 0));
        assert_eq!(hyper[0], luma[0].hyper_core);

        let chroma = analysis_tiles(1, 700, 1048, 88, 131, 22, 33).unwrap();
        let images: Vec<Area> = chroma.iter().map(|t| t.image).collect();
        assert_eq!(
            images,
            [
                Area::new(0, 0, 512, 512),
                Area::new(480, 0, 512, 512),
                Area::new(960, 0, 88, 512),
                Area::new(0, 480, 512, 220),
                Area::new(480, 480, 512, 220),
                Area::new(960, 480, 88, 220),
            ]
        );
        // The chroma plane is half the luma plane, so its latent / hyper grids coincide.
        assert_eq!(
            chroma.iter().map(|t| t.latent).collect::<Vec<_>>(),
            luma.iter().map(|t| t.latent).collect::<Vec<_>>()
        );
        assert_eq!(
            chroma.iter().map(|t| t.hyper).collect::<Vec<_>>(),
            luma.iter().map(|t| t.hyper).collect::<Vec<_>>()
        );
    }

    #[test]
    fn small_pictures_are_one_tile() {
        for ccs in 0..2 {
            let d = [1usize, 2][ccs];
            let t = analysis_tiles(ccs, 888 / d, 560 / d, 56, 35, 14, 9).unwrap();
            assert_eq!(t.len(), 1);
            assert_eq!(t[0].latent_core, Area::new(0, 0, 35, 56));
        }
    }

    #[test]
    fn synthesis_tiling_matches_the_reference_header() {
        assert_eq!(synthesis_tiling(888, 560), None);
        let t = synthesis_tiling(1400, 2096).unwrap();
        assert_eq!((t.tile_size, t.overlap), (1024, 64));
    }
}
