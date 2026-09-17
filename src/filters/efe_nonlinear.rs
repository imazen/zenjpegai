//! EFE non-linear post-filter (decoder side): a luma-driven piecewise-linear chroma correction,
//! followed by a block-wise switch between its result and the alternative picture the EFE
//! linear filter prepared.
//!
//! Ports `ref/src/codec/coding_tools/filters/EFEnonlinear/EFEnonlinear.py`: `decompress`,
//! `LumaAidedAdaptiveNonlinearFilter_apply`, `apply_OnoffSwitch`, `downsample`, `deinteger`
//! (the header syntax, `decode_header`, is in `crate::header`).
//!
//! Per tile and chroma plane the reference builds a two-layer 1x1 network out of eight coded
//! weights `w[0..8]` and the tile's coded luma range (`min`, `max`, eight equal bins):
//!
//! ```text
//! out = relu(w0 * (Y - min) + C) + sum_{i=1..7} w_i * relu(Y - (min + i * gap))
//! ```
//!
//! with `Y` the luma plane sampled at the chroma positions. Float order follows the reference's
//! two convolutions: taps in input-channel order, bias last (bound in `PORTING.md`).

use alloc::vec::Vec;

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};
use crate::header::{EfeNonlinearHeader, EfeNonlinearTiles};
use crate::nn::fast::for_each_row;
use crate::tensor::Tensor;

/// `hist_split_num`: luma bins per tile, which is also the number of weights per tile.
const BINS: usize = 8;

/// The reference's test for "the on/off switch is in use": `mask1.shape[2] > 0 or
/// mask1.shape[3] > 0`. Only the first (U) mask is looked at; a stream with the V mask alone
/// never switches, and the EFE linear filter does not build the alternative picture for it.
pub(super) fn has_first_mask(h: &EfeNonlinearHeader) -> bool {
    h.masks[0].is_some() && h.mask_geometry.is_some_and(|(_, my, mx)| my > 0 || mx > 0)
}

/// One tile of `LumaAidedAdaptiveNonlinearFilter_apply`.
struct Tile {
    rows: (usize, usize),
    cols: (usize, usize),
    /// `conv1.weight[0, 0]`, `conv1.bias[0]`.
    w0: f32,
    b0: f32,
    /// `conv1.bias[1..8]` and `conv2.weight[0, 1..8]`.
    ceiling: [f32; BINS - 1],
    w: [f32; BINS - 1],
}

/// `round(float(a) / n * n)` as the reference computes its tile borders (Python doubles, round
/// half to even).
fn tile_edge(a: usize, n: usize) -> usize {
    ((a as f64 / n as f64) * n as f64).round_ties_even() as usize
}

fn tiles(
    t: &EfeNonlinearTiles,
    min_symbol: u16,
    codes: &[u32],
    (hh, ww): (usize, usize),
) -> Result<Vec<Tile>> {
    let (th, tw) = (t.tile_height as usize, t.tile_width as usize);
    if th == 0 || tw == 0 {
        return Err(Error::InvalidData("EFE non-linear: zero tile size"));
    }
    let (ni, nj) = (hh.div_ceil(th), ww.div_ceil(tw));
    let count = ni * nj;
    if t.luma_min.len() < count || t.luma_max.len() < count || codes.len() < count * BINS {
        return Err(Error::InvalidData(
            "EFE non-linear: fewer tile parameters than tiles",
        ));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..ni {
        for j in 0..nj {
            let idx = i * nj + j;
            let wf = |k: usize| {
                super::efe_linear::deinteger(codes[idx * BINS + k] as i64 + min_symbol as i64)
            };
            // Python doubles, narrowed to f32 where they meet a tensor.
            let minimum = t.luma_min[idx] as f64 / 100.0;
            let gap = (t.luma_max[idx] as f64 / 100.0 - minimum) / BINS as f64;
            let ceil = |k: usize| -(1.0f32 * minimum as f32 + k as f32 * gap as f32);
            out.push(Tile {
                rows: (tile_edge(i * th, hh), tile_edge(((i + 1) * th).min(hh), hh)),
                cols: (tile_edge(j * tw, ww), tile_edge(((j + 1) * tw).min(ww), ww)),
                w0: wf(0),
                b0: wf(0) * ceil(0),
                ceiling: core::array::from_fn(|k| ceil(k + 1)),
                w: core::array::from_fn(|k| wf(k + 1)),
            });
        }
    }
    Ok(out)
}

/// `LumaAidedAdaptiveNonlinearFilter_apply` for one chroma plane, in place.
fn filter_plane(
    ctx: &FilterContext<'_>,
    t: &EfeNonlinearTiles,
    min_symbol: u16,
    codes: &[u32],
    luma: &Tensor<f32>,
    plane: &mut Tensor<f32>,
) -> Result<()> {
    let (sv, sh) = (ctx.hdr.s_ver as usize, ctx.hdr.s_hor as usize);
    // `downsample`: luma at the chroma positions, `Y[::s_ver, ::s_hor]`.
    let (hh, ww) = (luma.h.div_ceil(sv), luma.w.div_ceil(sh));
    if (plane.h, plane.w) != (hh, ww) {
        return Err(Error::InvalidArgument(
            "EFE non-linear: chroma plane size does not match the source format",
        ));
    }
    if hh == 0 || ww == 0 {
        return Ok(());
    }
    let tiles = tiles(t, min_symbol, codes, (hh, ww))?;
    let (lw, y) = (luma.w, &luma.data);
    for_each_row(ctx.eng, &mut plane.data, ww, |row_idx, row| {
        let yrow = &y[row_idx * sv * lw..][..lw];
        for tile in tiles
            .iter()
            .filter(|t| t.rows.0 <= row_idx && row_idx < t.rows.1)
        {
            for x in tile.cols.0..tile.cols.1.min(ww) {
                let l = yrow[x * sh];
                // conv1 (2 -> 8, bias) + ReLU, then conv2 (8 -> 1).
                let mut acc = (tile.w0 * l + row[x] + tile.b0).max(0.0);
                for k in 0..BINS - 1 {
                    acc += tile.w[k] * (l + tile.ceiling[k]).max(0.0);
                }
                row[x] = acc;
            }
        }
    });
    Ok(())
}

/// `apply_OnoffSwitch` for one plane: per `bs x bs` block, mask value 0 keeps the filtered
/// sample, 2 takes the alternative picture's, 1 averages the two.
fn switch_plane(
    ctx: &FilterContext<'_>,
    (bs, my, mx): (usize, usize, usize),
    mask: &[u8],
    alt: &Tensor<f32>,
    plane: &mut Tensor<f32>,
) -> Result<()> {
    let (h, w) = (plane.h, plane.w);
    if (alt.h, alt.w) != (h, w) {
        return Err(Error::InvalidArgument(
            "EFE non-linear: alternative picture size",
        ));
    }
    // The reference pads both pictures to a multiple of `bS` and multiplies by the mask repeated
    // `bS` times: any other mask size fails there.
    if bs == 0 || my != h.div_ceil(bs) || mx != w.div_ceil(bs) || mask.len() != my * mx {
        return Err(Error::InvalidData(
            "EFE non-linear: mask size does not cover the picture",
        ));
    }
    if w == 0 {
        return Ok(());
    }
    let a = &alt.data;
    for_each_row(ctx.eng, &mut plane.data, w, |y, row| {
        let arow = &a[y * w..][..w];
        let mrow = &mask[(y / bs) * mx..][..mx];
        for (bx, &m) in mrow.iter().enumerate() {
            let x0 = bx * bs;
            let x1 = (x0 + bs).min(w);
            match m {
                0 => {}
                2 => row[x0..x1].copy_from_slice(&arow[x0..x1]),
                _ => {
                    // (rec * mask + (2 - mask) * filt) / 2 with mask = 1
                    for (f, r) in row[x0..x1].iter_mut().zip(&arow[x0..x1]) {
                        *f = (*r + *f) / 2.0;
                    }
                }
            }
        }
    });
    Ok(())
}

/// `EFEnonlinear.decompress`.
pub fn apply(
    ctx: &FilterContext<'_>,
    h: &EfeNonlinearHeader,
    state: FilterState,
) -> Result<FilterState> {
    super::efe_linear::supported_format(ctx.hdr)?;
    let FilterState {
        mut image,
        upsampled,
    } = state;
    if let Some(t) = &h.nonlinear {
        if let Some(codes) = &t.weights[0] {
            filter_plane(ctx, t, h.min_symbol, codes, &image.y, &mut image.u)?;
        }
        if let Some(codes) = &t.weights[1] {
            filter_plane(ctx, t, h.min_symbol, codes, &image.y, &mut image.v)?;
        }
    }
    if has_first_mask(h) {
        let alt = upsampled.as_ref().ok_or(Error::InvalidData(
            "EFE non-linear: on/off masks without the EFE linear filter's alternative picture",
        ))?;
        let (bs, my, mx) = h.mask_geometry.unwrap_or_default();
        let geom = (bs as usize, my as usize, mx as usize);
        if let Some(mask) = &h.masks[0] {
            switch_plane(ctx, geom, mask, &alt.u, &mut image.u)?;
        }
        if let Some(mask) = &h.masks[1] {
            switch_plane(ctx, geom, mask, &alt.v, &mut image.v)?;
        }
    }
    Ok(FilterState { image, upsampled })
}
