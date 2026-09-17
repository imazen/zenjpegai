//! Region geometry: how the picture, the latent `y`, `psi` and `z` are partitioned when
//! `region_partitioning_flag` is set.
//!
//! Ports `TileManagerHyper` (`ref/src/codec/coding_tools/tiling/tiling.py`) and the `Area`
//! arithmetic of `ref/src/codec/common/tiling.py`. Regions are laid out row-major: region index
//! `= row * num_hor + column`. Region boundaries are multiples of 128 luma samples; the last
//! row/column takes the remainder.

use alloc::vec::Vec;

use crate::header::{PictureHeader, Regions};

/// Axis-aligned rectangle in samples of whatever plane it refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Area {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Area {
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Region grid of one component at one resolution, with and without the overlap extension used
/// by dependent regions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionGrid {
    pub rows: usize,
    pub columns: usize,
    /// Core areas (no overlap), row-major.
    pub core: Vec<Area>,
    /// Areas grown by the overlap amount on every side that touches a neighbour.
    pub extended: Vec<Area>,
}

/// Region start/end coordinates along one axis at `1 / 2^depth` resolution
/// (`calculate_region_coordinates`).
fn axis_coords(
    size: usize,
    num_regions: usize,
    depth: u32,
    partitioned: bool,
) -> Vec<(usize, usize)> {
    let feature = size.div_ceil(1 << depth);
    if !partitioned {
        return alloc::vec![(0, feature)];
    }
    let region_size = (size.div_ceil(128) / num_regions) * 128;
    (0..num_regions)
        .map(|i| {
            let start = (i * region_size) >> depth;
            let end = if i + 1 < num_regions {
                ((i + 1) * region_size) >> depth
            } else {
                feature
            };
            (start, end)
        })
        .collect()
}

/// `_init_latent_tiles(extend, extend_amount)`.
fn grid(ver: &[(usize, usize)], hor: &[(usize, usize)], extend: usize) -> Vec<Area> {
    let mut out = Vec::with_capacity(ver.len() * hor.len());
    for (vi, &(y0, y1)) in ver.iter().enumerate() {
        for (hi, &(x0, x1)) in hor.iter().enumerate() {
            let (mut x, mut y) = (x0, y0);
            let (mut width, mut height) = (x1 - x0, y1 - y0);
            if hi > 0 {
                x -= extend.min(x);
                width += extend;
            }
            if hi + 1 < hor.len() {
                width += extend;
            }
            if vi > 0 {
                y -= extend.min(y);
                height += extend;
            }
            if vi + 1 < ver.len() {
                height += extend;
            }
            out.push(Area::new(x, y, width, height));
        }
    }
    out
}

/// Which plane a [`RegionGrid`] describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plane {
    /// Latent `y` (1/16 of luma).
    Latent,
    /// Hyper-decoder output `psi` (1/32).
    Psi,
    /// Hyper-latent `z` (1/64).
    HyperLatent,
}

/// Region grid for component `ccs` (0 luma, 1 chroma).
///
/// The reference computes region geometry on the luma-sized picture for both components (for
/// chroma it doubles the half-size picture back up), at depths 4 / 5 / 6.
pub fn region_grid(hdr: &PictureHeader, ccs: usize, plane: Plane) -> RegionGrid {
    let (h, w) = hdr.component_size(ccs);
    let (h, w) = if ccs == 0 {
        (h as usize, w as usize)
    } else {
        (h as usize * 2, w as usize * 2)
    };
    let regions = hdr.regions.unwrap_or(Regions {
        num_ver: 1,
        num_hor: 1,
        independent: false,
        hyper_decoder_overlap: 0,
        mcm_overlap: 0,
    });
    let partitioned = hdr.regions.is_some();
    let (rows, columns) = (regions.num_ver as usize, regions.num_hor as usize);
    // HyperDecoderOverlap = 2 * signalled value, McmOverlap = 4 * signalled value.
    let hd_overlap = regions.hyper_decoder_overlap as usize * 2;
    let mcm_overlap = regions.mcm_overlap as usize * 4;
    let (depth, extend) = match plane {
        Plane::Latent => (4, mcm_overlap >> 1),
        Plane::Psi => (5, mcm_overlap >> 2),
        Plane::HyperLatent => (6, hd_overlap),
    };
    let extend = if regions.independent { 0 } else { extend };
    let ver = axis_coords(h, rows, depth, partitioned);
    let hor = axis_coords(w, columns, depth, partitioned);
    RegionGrid {
        rows,
        columns,
        core: grid(&ver, &hor, 0),
        extended: grid(&ver, &hor, extend),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_region_covers_the_latent() {
        assert_eq!(axis_coords(888, 1, 4, false), [(0, 56)]);
        assert_eq!(axis_coords(560, 1, 6, false), [(0, 9)]);
    }

    #[test]
    fn partitioned_axis() {
        // 3000 rows, 3 regions: floor(ceil(3000/128)/3)*128 = floor(24/3)*128 = 1024
        assert_eq!(
            axis_coords(3000, 3, 4, true),
            [(0, 64), (64, 128), (128, 188)]
        );
        assert_eq!(axis_coords(3000, 3, 6, true), [(0, 16), (16, 32), (32, 47)]);
    }

    #[test]
    fn extension_only_towards_neighbours() {
        let ver = [(0, 64), (64, 128), (128, 188)];
        let hor = [(0, 100), (100, 250)];
        let g = grid(&ver, &hor, 4);
        assert_eq!(g[0], Area::new(0, 0, 104, 68));
        assert_eq!(g[1], Area::new(96, 0, 154, 68));
        assert_eq!(g[2], Area::new(0, 60, 104, 72));
        assert_eq!(g[5], Area::new(96, 124, 154, 64));
    }
}
