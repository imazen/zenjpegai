//! eICCI: a bank of small CNNs that enhance Y, U and V, selected per filter tile.
//!
//! Ports `ref/src/codec/coding_tools/filters/eICCI/{icci_filter.py, model_idxes.py, params.py}`
//! (decoder side) and the tile layout it asks of `tiling/tiling.py::TileManager`
//! (`_init_image_tiles_with_overlap`, `_adjust_boundary_tiles` with `minimum_tile_size = 176`,
//! picture-border branch of `_get_core_of_overlapping_tile`). The network itself is
//! [`crate::model::icci`].
//!
//! Per tile: normalise to `[0, 1]`, replicate-pad right/bottom to a multiple of 4, two Haar
//! levels per plane, run the selected luma network and/or the selected chroma network, inverse
//! transform, crop, and keep the tile's core. After the last tile the picture is clamped and
//! scaled back. Even a tile with nothing selected goes through normalise / clamp / scale.
//!
//! Tiles are filtered **in place**, in raster order, exactly like the reference decoder: a tile
//! reads the cores its predecessors already wrote inside its overlap margin (see `PORTING.md`).
//!
//! Chroma-subsampled pictures (`4:2:0`, `4:2:2` sources): the reference up-samples the chroma
//! planes to the luma size (`Image.to_444_`, bicubic), filters at 4:4:4, then down-samples back
//! (`Image.to_format_`, bilinear); [`crate::nn::resize_bilinear`] reproduces its kernel bit
//! for bit.
//! `4:2:0` can never be *signalled* (`icci_enable_flag` is not coded for `s_ver == s_hor == 2`),
//! but the filter path itself is exercised by a forced stream (see `PORTING.md`).

use alloc::vec::Vec;

use super::{FilterContext, FilterState};
use crate::decoder::reconstruct::Planes;
use crate::error::{Error, Result};
use crate::header::{IcciHeader, IcciTile, OperatingPoint, PictureHeader, SynthesisTiling};
use crate::model::ModelSource;
use crate::model::icci;
pub use crate::model::icci::NetCache;
use crate::nn::fast::Engine;
use crate::nn::resize_bilinear;
use crate::tensor::Tensor;
use crate::tools::regions::Area;

/// `EfficientICCIFilter.tile_min_size_for_MSSSIM`: the last tile column / row is at least this
/// large (taken from its neighbour). Not signalled.
pub const MINIMUM_TILE_SIZE: usize = 176;

/// `post_filters.eICCI.y_short_list` (`cfg/pipeline.json`): `[operating point][model_id]` → the
/// two luma networks a short-list index addresses. Not signalled.
pub const LUMA_SHORT_LIST: [[[u8; 2]; 5]; 3] = [
    [[5, 6], [2, 6], [2, 7], [3, 8], [3, 4]],
    [[5, 6], [3, 6], [2, 7], [3, 8], [4, 9]],
    [[5, 6], [6, 8], [2, 7], [3, 9], [4, 9]],
];
/// `post_filters.eICCI.uv_short_list`.
pub const CHROMA_SHORT_LIST: [[[u8; 2]; 5]; 3] = [
    [[0, 1], [0, 1], [1, 2], [1, 2], [3, 4]],
    [[0, 1], [1, 2], [2, 3], [2, 3], [1, 4]],
    [[0, 2], [1, 3], [2, 3], [2, 3], [3, 4]],
];

/// One filter tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilterTile {
    /// Picture area the network sees.
    pub image: Area,
    /// Part of it that is kept, in picture coordinates.
    pub core: Area,
    /// Offset of `core` inside `image`.
    pub core_offset: (usize, usize),
}

/// Tile starts and lengths along one axis.
fn axis(len: usize, t: SynthesisTiling) -> Result<Vec<(usize, usize)>> {
    let (tile, overlap) = ((t.tile_size as usize).min(len), t.overlap as usize);
    if tile <= overlap || len <= overlap {
        return Err(Error::InvalidData(
            "eICCI: tile not larger than its overlap",
        ));
    }
    let mut spans: Vec<(usize, usize)> = (0..len - overlap)
        .step_by(tile - overlap)
        .map(|p| (p, tile.min(len - p)))
        .collect();
    // `_adjust_boundary_tiles`: a short last tile takes MINIMUM_TILE_SIZE from its neighbour.
    let n = spans.len();
    if spans[n - 1].1 < MINIMUM_TILE_SIZE {
        // With a single tile the reference indexes `[-2]` onto the same tile and produces a
        // negative position; with a neighbour not larger than the minimum it produces an empty
        // or inverted tile. Neither is a layout.
        if n < 2 || spans[n - 2].1 <= MINIMUM_TILE_SIZE + overlap {
            return Err(Error::InvalidData(
                "eICCI: tiling leaves a tile below the minimum size",
            ));
        }
        spans[n - 2].1 -= MINIMUM_TILE_SIZE;
        spans[n - 1].0 -= MINIMUM_TILE_SIZE;
        spans[n - 1].1 += MINIMUM_TILE_SIZE;
    }
    Ok(spans)
}

/// Filter tiles of a `height x width` picture, raster order. `None` = one tile.
pub fn tile_layout(
    height: usize,
    width: usize,
    tiling: Option<SynthesisTiling>,
) -> Result<Vec<FilterTile>> {
    let Some(t) = tiling else {
        let image = Area::new(0, 0, width, height);
        return Ok(alloc::vec![FilterTile {
            image,
            core: image,
            core_offset: (0, 0),
        }]);
    };
    let half = t.overlap as usize / 2;
    let (rows, cols) = (axis(height, t)?, axis(width, t)?);
    let mut tiles = Vec::with_capacity(rows.len() * cols.len());
    for &(y, h) in &rows {
        for &(x, w) in &cols {
            let left = if x == 0 { 0 } else { half };
            let top = if y == 0 { 0 } else { half };
            let right = if x + w >= width { 0 } else { half };
            let bottom = if y + h >= height { 0 } else { half };
            if w <= left + right || h <= top + bottom {
                return Err(Error::InvalidData("eICCI: tile is all overlap"));
            }
            tiles.push(FilterTile {
                image: Area::new(x, y, w, h),
                core: Area::new(x + left, y + top, w - left - right, h - top - bottom),
                core_offset: (left, top),
            });
        }
    }
    Ok(tiles)
}

/// Networks (index into the bank of 10) filtering Y and U/V of one tile; `None` = plane kept.
/// `ICCIModelIndexes.decode_header`: `model_selection = map[signalled_idx] + 1`.
fn selection(
    t: &IcciTile,
    op: OperatingPoint,
    model_id: usize,
) -> Result<(Option<usize>, Option<usize>)> {
    let pick = |list: &[[[u8; 2]; 5]; 3], idx: u8| -> Result<usize> {
        if !t.short_list {
            return Ok(idx as usize);
        }
        list[op as usize]
            .get(model_id)
            .and_then(|l| l.get(idx as usize))
            .map(|&m| m as usize)
            .ok_or(Error::InvalidData("eICCI: short-list index out of range"))
    };
    let y = t.use_yuv[0]
        .then(|| pick(&LUMA_SHORT_LIST, t.index_y))
        .transpose()?;
    let uv = (t.use_yuv[1] || t.use_yuv[2])
        .then(|| pick(&CHROMA_SHORT_LIST, t.index_uv))
        .transpose()?;
    Ok((y, uv))
}

/// The core of a filtered tile (`full`: the tile at padded size) written back to `plane`.
fn write_core(plane: &mut Tensor<f32>, tile: &FilterTile, full: &Tensor<f32>) {
    let (ox, oy) = tile.core_offset;
    for y in 0..tile.core.height {
        let src = &full.data[(oy + y) * full.w + ox..][..tile.core.width];
        plane.data[(tile.core.y + y) * plane.w + tile.core.x..][..tile.core.width]
            .copy_from_slice(src);
    }
}

/// Apply the filter to a 4:4:4 picture in the codec range. `op` is the operating point the
/// picture was synthesised with: it selects the network bank and the short lists.
///
/// `stop` is checked once per tile: each tile runs a whole luma and/or chroma network, the
/// filter's most expensive unit of work.
#[allow(clippy::too_many_arguments)]
pub fn filter(
    eng: &Engine,
    hdr: &PictureHeader,
    h: &IcciHeader,
    op: OperatingPoint,
    models: &dyn ModelSource,
    cache: &NetCache,
    mut image: Planes,
    stop: &dyn enough::Stop,
) -> Result<Planes> {
    let (ph, pw) = (image.y.h, image.y.w);
    if image.y.c != 1 || ph == 0 || pw == 0 {
        return Err(Error::InvalidArgument("eICCI: empty picture"));
    }
    // `Image.to_444_()`: a subsampled source's chroma planes are bicubic-upsampled to the luma
    // size and the filter runs at 4:4:4; `to_format_()` after the filter bilinear-downsamples
    // them back. The header gives the source subsampling, the plane sizes must agree with it.
    let chroma = match (hdr.s_ver, hdr.s_hor) {
        (1, 1) => None,
        (2, 2) => Some((ph.div_ceil(2), pw.div_ceil(2))),
        (1, 2) => Some((ph, pw.div_ceil(2))),
        _ => {
            return Err(Error::Unsupported(
                "eICCI post-filter on a chroma format other than 4:2:0 / 4:2:2",
            ));
        }
    };
    let same = |p: &Tensor<f32>| p.c == 1 && p.h == ph && p.w == pw;
    if let Some((uh, uw)) = chroma {
        if image.u.c != 1
            || image.v.c != 1
            || (image.u.h, image.u.w) != (uh, uw)
            || (image.v.h, image.v.w) != (uh, uw)
        {
            return Err(Error::InvalidData(
                "eICCI: chroma planes do not match the source subsampling",
            ));
        }
        image.u = crate::decoder::output::resize_bicubic(&image.u, ph, pw)?;
        image.v = crate::decoder::output::resize_bicubic(&image.v, ph, pw)?;
    } else if !same(&image.u) || !same(&image.v) {
        return Err(Error::InvalidData(
            "eICCI: chroma planes do not match the source subsampling",
        ));
    }
    if !(1..=16).contains(&hdr.bit_depth) {
        return Err(Error::InvalidArgument("eICCI: bit depth"));
    }
    // The reference lays the tiles out on the coded picture size but filters the displayed
    // picture; with one tile that is the whole displayed picture.
    let tiles = if h.tiling.is_some() {
        if (ph, pw) != (hdr.height as usize, hdr.width as usize) {
            return Err(Error::Unsupported(
                "eICCI tiling on a picture with a non-displayed border",
            ));
        }
        tile_layout(ph, pw, h.tiling)?
    } else {
        tile_layout(ph, pw, None)?
    };
    if tiles.len() != h.tiles.len() {
        return Err(Error::InvalidData(
            "eICCI: tile count does not match the tiling",
        ));
    }
    let range = ((1u32 << hdr.bit_depth) - 1) as f32;

    for p in [&mut image.y, &mut image.u, &mut image.v] {
        for s in &mut p.data {
            *s /= range;
        }
    }
    for (tile, sel) in tiles.iter().zip(&h.tiles) {
        stop.check()?;
        let (net_y, net_uv) = selection(sel, op, hdr.model_id as usize)?;
        if net_y.is_none() && net_uv.is_none() {
            continue;
        }
        let a = tile.image;
        let input = icci::tile_input(
            eng,
            [&image.y, &image.u, &image.v],
            (a.x, a.y, a.width, a.height),
        )?;
        // Every correction is computed from the tile as it was read, then written back.
        let luma = net_y
            .map(|i| cache.get(models, op, i, eng)?.luma_correction(eng, &input))
            .transpose()?;
        let chroma = net_uv
            .map(|i| {
                let want = [sel.use_yuv[1], sel.use_yuv[2]];
                cache
                    .get(models, op, i, eng)?
                    .chroma_corrections(eng, &input, want)
            })
            .transpose()?;
        let [cu, cv] = chroma.unwrap_or([None, None]);
        for (p, (plane, c)) in [&mut image.y, &mut image.u, &mut image.v]
            .into_iter()
            .zip([luma, cu, cv])
            .enumerate()
        {
            if let Some(c) = c {
                write_core(plane, tile, &icci::tile_output(eng, &input, p, &c)?);
            }
        }
    }
    for p in [&mut image.y, &mut image.u, &mut image.v] {
        for s in &mut p.data {
            *s = s.clamp(0.0, 1.0) * range;
        }
    }
    // `img_flt.to_format_(source subsampling)`: chroma back down to its planes' size.
    if let Some((uh, uw)) = chroma {
        image.u = resize_bilinear(&image.u, uh, uw)?;
        image.v = resize_bilinear(&image.v, uh, uw)?;
    }
    Ok(image)
}

pub(super) fn apply(
    ctx: &FilterContext<'_>,
    h: &IcciHeader,
    mut state: FilterState,
) -> Result<FilterState> {
    state.image = filter(
        ctx.eng,
        ctx.hdr,
        h,
        ctx.op,
        ctx.models,
        ctx.icci_nets,
        state.image,
        ctx.stop,
    )?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untiled_is_one_tile() {
        let t = tile_layout(888, 560, None).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].core, Area::new(0, 0, 560, 888));
    }

    /// Layout the reference builds for 2096x1400, tile 1024, overlap 48: the last column would be
    /// 144 wide, so it takes 176 from its neighbour
    /// (`scripts/ref_vectors/dump_filters_lef_icci.py` on `img01_base_eiccitiles_lef_bpp050`).
    #[test]
    fn matches_reference_layout_with_adjusted_boundary() {
        let tiling = SynthesisTiling {
            tile_size: 1024,
            overlap: 48,
        };
        let t = tile_layout(1400, 2096, Some(tiling)).unwrap();
        let got: Vec<(Area, Area, (usize, usize))> =
            t.iter().map(|t| (t.image, t.core, t.core_offset)).collect();
        assert_eq!(
            got,
            [
                (
                    Area::new(0, 0, 1024, 1024),
                    Area::new(0, 0, 1000, 1000),
                    (0, 0)
                ),
                (
                    Area::new(976, 0, 848, 1024),
                    Area::new(1000, 0, 800, 1000),
                    (24, 0)
                ),
                (
                    Area::new(1776, 0, 320, 1024),
                    Area::new(1800, 0, 296, 1000),
                    (24, 0)
                ),
                (
                    Area::new(0, 976, 1024, 424),
                    Area::new(0, 1000, 1000, 400),
                    (0, 24)
                ),
                (
                    Area::new(976, 976, 848, 424),
                    Area::new(1000, 1000, 800, 400),
                    (24, 24)
                ),
                (
                    Area::new(1776, 976, 320, 424),
                    Area::new(1800, 1000, 296, 400),
                    (24, 24)
                ),
            ]
        );
        let covered: usize = t.iter().map(|t| t.core.width * t.core.height).sum();
        assert_eq!(covered, 1400 * 2096);
        // A picture narrower than the minimum, or tiles too small to give 176 away.
        assert!(tile_layout(1400, 160, Some(tiling)).is_err());
        let small = SynthesisTiling {
            tile_size: 192,
            overlap: 48,
        };
        assert!(tile_layout(1400, 2096, Some(small)).is_err());
    }

    #[test]
    fn short_lists_map_to_the_bank() {
        let t = IcciTile {
            use_yuv: [true, false, true],
            short_list: true,
            index_y: 1,
            index_uv: 0,
        };
        assert_eq!(
            selection(&t, OperatingPoint::Bop, 1).unwrap(),
            (Some(6), Some(1))
        );
        assert_eq!(
            selection(&t, OperatingPoint::Hop, 1).unwrap(),
            (Some(8), Some(1))
        );
        let long = IcciTile {
            short_list: false,
            index_y: 9,
            ..t
        };
        assert_eq!(
            selection(&long, OperatingPoint::Sop, 4).unwrap(),
            (Some(9), Some(0))
        );
        assert!(selection(&t, OperatingPoint::Bop, 5).is_err());
    }
}
