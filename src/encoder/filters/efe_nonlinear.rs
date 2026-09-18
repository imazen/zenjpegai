//! EFE non-linear post-filter, encode side: the per-tile luma-binned chroma fit and the
//! on/off mask decision.
//!
//! Ports `ref/src/codec/coding_tools/filters/EFEnonlinear/EFEnonlinear.py`: `compress`,
//! `LumaAidedAdaptiveNonlinearFilter_encoder`, `calculateOnOff` and the `encode_header`
//! decisions (the apply half — `LumaAidedAdaptiveNonlinearFilter_apply`, `apply_OnoffSwitch`,
//! `downsample`, `deinteger` — is shared with the decoder, `crate::filters::efe_nonlinear`,
//! which the reference oracle verifies bit-identically).
//!
//! Per tile and chroma plane the reference solves `A x = B` with `A` the eight luma-bin
//! `relu(Y - (min + i * gap))` columns and `B = org - rec`, by `torch.linalg.lstsq` — MKL's
//! f32 `sgelsy`, which is **not run-to-run deterministic**. The solve here reuses the
//! deterministic f64 pivoted-QR `gelsy` of `encoder::filters::efe_linear`; on well-posed
//! systems the integerised codes are identical, on rank-borderline ones the reference's own
//! draw is unreproducible by construction (documented in `PORTING.md`).
//!
//! Two reference quirks are ported verbatim:
//!
//! - `encode_header`'s `minnn`/`maxxx` fold starts at `0` / `2^wP - 1`, so `minSymbol` is
//!   always 0 and `maxSymbol` always 65535 — the weight codes are signalled raw.
//! - The mask candidate for a plane is kept only when its loss is *strictly* positive, and
//!   each plane's mask is evaluated independently (the decoder only ever applies the U mask —
//!   `crate::filters::efe_nonlinear::has_first_mask` — but the encoder still codes a V-only
//!   win, and the reference encoder does too).

use alloc::vec::Vec;

use crate::decoder::reconstruct::Planes;
use crate::encoder::colour::SourceMeta;
use crate::error::{Error, Result};
use crate::filters::efe_linear::integerize;
use crate::filters::efe_nonlinear::{self as dec, BINS, tile_edge};
use crate::header::{EfeNonlinearHeader, EfeNonlinearTiles};
use crate::nn::fast::Engine;
use crate::tensor::Tensor;

use super::efe_linear::{lstsq, psnr};

/// `base_model_beta` (`core_models/CCS_SGMM/params.py`): the PSNR weight of the filter loss,
/// indexed by the model in use.
const BASE_MODEL_BETA: [f64; 4] = [0.002, 0.012, 0.075, 0.5];
/// `lossModifier` — 1.5 for the non-linear filter (`EFElinear` uses 0.5).
const LOSS_MODIFIER: f64 = 1.5;
/// `wP`, the signalled weight precision.
const WEIGHT_PRECISION: f64 = 16.0;
/// `NonlinearFilter_tile_width_base` / `_height_base`.
const TILE_BASE: usize = 1200;
/// `bSizes`: the on/off mask block size per model.
const BLOCK_SIZES: [usize; 5] = [128, 112, 96, 80, 64];

/// The inputs of `EFEnonlinear.compress`: the picture arriving at the filter, the EFE linear
/// filter's alternative picture, the source, and the parameters the header / config decide.
pub struct EfeNonlinearInput<'a> {
    /// SIMD engine the apply paths (`crate::filters::efe_nonlinear`) run on.
    pub eng: &'a Engine,
    /// Source chroma format (`s_ver` / `s_hor`) — the filter's `d_ver` / `d_hor`.
    pub meta: &'a SourceMeta,
    /// `get_base_model_id()`: the active model, selecting `BASE_MODEL_BETA` and `BLOCK_SIZES`.
    pub model_id: usize,
    /// `org_img_i` converted to YUV at the reconstruction's range (`img.to_YUV_()` +
    /// `convert_range_`): the source's planes at the source's chroma resolution.
    pub org: &'a Planes,
    /// `imgs[0]`: the picture arriving at the filter — the plain reconstruction, or the EFE
    /// linear / eICCI output when those tools ran first (`FiltersComposite.compress` chains
    /// `current_images[0]` through them).
    pub rec: &'a Planes,
    /// `imgs[1]`: the EFE linear filter's up-sampled picture (`filtersBest2` applied); without
    /// it the on/off mask search is skipped entirely (`rec_upsample is not None`).
    pub upsampled: Option<&'a Planes>,
}

/// Everything `EFEnonlinear.compress` produces that survives in the stream.
pub struct EfeNonlinearOutput {
    /// The `EFE_nonlinear_filter_enabled_flag` body of the tool header.
    pub header: EfeNonlinearHeader,
    /// `ans[0]`: the picture after the filter and the kept masks (chroma only — the luma
    /// plane passes through untouched). Nothing downstream of the encoder needs it; it is
    /// returned for the oracle tests.
    pub filtered: Planes,
}

/// `self.downsample`: the luma plane at the chroma positions, `Y[::s_ver, ::s_hor]` with the
/// reference's replicate pad for odd sizes (the pad row/column is a copy of the last one, so
/// plain indexing reaches it).
fn downsample(plane: &Tensor<f32>, (sv, sh): (usize, usize)) -> Tensor<f32> {
    let (h, w) = (plane.h.div_ceil(sv), plane.w.div_ceil(sh));
    let mut out = Vec::with_capacity(h * w);
    for y in 0..h {
        let row = &plane.data[(y * sv) * plane.w..][..plane.w];
        out.extend((0..w).map(|x| row[x * sh]));
    }
    // `h * w` is within the plane's own bound, so this cannot fail.
    Tensor::from_vec(1, h, w, out).expect("downsample shape")
}

/// The tile grid and per-tile fits `LumaAidedAdaptiveNonlinearFilter_encoder` produces.
struct TileSolve {
    /// `NonlinearFilter_tile_width` / `_height` (64-snapped).
    tile_width: u16,
    /// See `tile_width`.
    tile_height: u16,
    /// `lumaMin` / `lumaMax`, one pair per tile (`round(range * 100)`).
    luma_min: Vec<u16>,
    /// See `luma_min`.
    luma_max: Vec<u16>,
    /// `NonlinearFilterWeights_U` / `_V`: `BINS` integerised codes per tile, raster order.
    weights: [Vec<u32>; 2],
}

/// `LumaAidedAdaptiveNonlinearFilter_encoder`: the tile grid over the downsampled luma, the
/// per-tile luma range (`lumaMin` / `lumaMax`, `round(min * 100)` in f32) and the eight
/// integerised weights per tile per plane.
fn solve_tiles(rec_y: &Tensor<f32>, org: &Planes, rec: &Planes) -> Result<TileSolve> {
    let (hh, ww) = (rec_y.h, rec_y.w);
    // `count_w` / `count_h`, then the tile size snapped up to a multiple of 64.
    let count_w = ww.div_ceil(TILE_BASE);
    let count_h = hh.div_ceil(TILE_BASE);
    let tile_w = ww.div_ceil(64 * count_w) * 64;
    let tile_h = hh.div_ceil(64 * count_h) * 64;
    let (ni, nj) = (hh.div_ceil(tile_h), ww.div_ceil(tile_w));
    let mut luma_min = Vec::with_capacity(ni * nj);
    let mut luma_max = Vec::with_capacity(ni * nj);
    let mut weights = [Vec::new(), Vec::new()];
    for i in 0..ni {
        for j in 0..nj {
            // The cand fractions are exact (`float(i * tile_h) / hh` then `round(c * hh)`),
            // so the edges are `tile_edge` of the unpadded endpoints.
            let (r0, r1) = (
                tile_edge(i * tile_h, hh),
                tile_edge(((i + 1) * tile_h).min(hh), hh),
            );
            let (c0, c1) = (
                tile_edge(j * tile_w, ww),
                tile_edge(((j + 1) * tile_w).min(ww), ww),
            );
            // `torch.round(torch.min(tile) * 100)` — f32 all the way.
            let mut tmin = f32::MAX;
            let mut tmax = f32::MIN;
            for y in r0..r1 {
                for &v in &rec_y.data[y * ww + c0..y * ww + c1] {
                    tmin = tmin.min(v);
                    tmax = tmax.max(v);
                }
            }
            let minimum = (tmin * 100.0).round_ties_even();
            let maximum = (tmax * 100.0).round_ties_even();
            luma_min.push(minimum.max(0.0) as u16);
            luma_max.push(maximum.max(0.0) as u16);
            // `gap` and the bin ceilings are f32 tensor arithmetic in the reference.
            let (mn, mx) = (minimum / 100.0, maximum / 100.0);
            let gap = (mx - mn) / BINS as f32;
            let ceiling = |k: usize| mn + k as f32 * gap;
            let n = (r1 - r0) * (c1 - c0);
            let mut a = alloc::vec![0.0f64; n * BINS];
            let mut b = [alloc::vec![0.0f64; n], alloc::vec![0.0f64; n]];
            let (ou, ov, ru, rv) = (&org.u.data, &org.v.data, &rec.u.data, &rec.v.data);
            let mut r = 0usize;
            for y in r0..r1 {
                for x in c0..c1 {
                    let l = rec_y.data[y * ww + x];
                    for k in 0..BINS {
                        a[r * BINS + k] = (l - ceiling(k)).max(0.0) as f64;
                    }
                    let p = y * ww + x;
                    b[0][r] = (ou[p] - ru[p]) as f64;
                    b[1][r] = (ov[p] - rv[p]) as f64;
                    r += 1;
                }
            }
            for p in 0..2 {
                let x = lstsq(&a, &b[p], n, BINS)?;
                weights[p].extend(x.iter().map(|&w| integerize(w as f32)));
            }
        }
    }
    Ok(TileSolve {
        tile_width: tile_w.min(u16::MAX as usize) as u16,
        tile_height: tile_h.min(u16::MAX as usize) as u16,
        luma_min,
        luma_max,
        weights,
    })
}

/// Replicate-pad a plane to `h2 x w2` (`Image.pad_`, bottom/right edge extension).
fn replicate_pad(plane: &Tensor<f32>, h2: usize, w2: usize) -> Tensor<f32> {
    let mut out = Vec::with_capacity(h2 * w2);
    for y in 0..h2 {
        let row = &plane.data[y.min(plane.h - 1) * plane.w..][..plane.w];
        out.extend((0..w2).map(|x| row[x.min(plane.w - 1)]));
    }
    Tensor::from_vec(1, h2, w2, out).expect("pad shape")
}

/// `AvgPool2d((bS, bS), (bS, bS))` of `(a - b)²` at block `(by, bx)`: a sequential f32
/// accumulation over the block, then the divide — PyTorch CPU's order.
fn block_mse(a: &Tensor<f32>, b: &Tensor<f32>, y0: usize, x0: usize, bs: usize) -> f32 {
    let mut s = 0.0f32;
    for y in y0..y0 + bs {
        let (ar, br) = (&a.data[y * a.w..][..a.w], &b.data[y * b.w..][..b.w]);
        for x in x0..x0 + bs {
            let d = ar[x] - br[x];
            s += d * d;
        }
    }
    s / (bs * bs) as f32
}

/// `calculateOnOff` for one chroma plane: the `bS x bS` block mask over the
/// replicate-padded planes — `2` where the filtered picture loses to the alternative, `1`
/// where averaging the two beats the winner, `0` where the filtered picture stands.
fn on_off_mask(org: &Tensor<f32>, alt: &Tensor<f32>, filt: &Tensor<f32>, bs: usize) -> Vec<u8> {
    let (my, mx) = (org.h.div_ceil(bs), org.w.div_ceil(bs));
    let (hp, wp) = (my * bs, mx * bs);
    let org = replicate_pad(org, hp, wp);
    let alt = replicate_pad(alt, hp, wp);
    let filt = replicate_pad(filt, hp, wp);
    // `recUave` — the plain average of the two candidates, the `mask3` baseline.
    let ave = Tensor::from_vec(
        1,
        hp,
        wp,
        alt.data
            .iter()
            .zip(&filt.data)
            .map(|(&a, &f)| (a + f) / 2.0)
            .collect(),
    )
    .expect("ave shape");
    let mut mask = alloc::vec![0u8; my * mx];
    for by in 0..my {
        for bx in 0..mx {
            let (y0, x0) = (by * bs, bx * bs);
            // `mask1 = pool((org - filt)²) > pool((org - rec)²)`, `* 2`.
            mask[by * mx + bx] =
                (block_mse(&org, &filt, y0, x0, bs) > block_mse(&org, &alt, y0, x0, bs)) as u8 * 2;
        }
    }
    // `maskedU = (filt * (2 - mask) + mask * rec) / 2`, then `mask1[mask3] = 1` where the
    // winner-take-all block loses to the plain average.
    let masked = Tensor::from_vec(
        1,
        hp,
        wp,
        (0..hp * wp)
            .map(|i| {
                let m = mask[(i / wp / bs) * mx + (i % wp) / bs] as f32;
                (filt.data[i] * (2.0 - m) + m * alt.data[i]) / 2.0
            })
            .collect(),
    )
    .expect("masked shape");
    for by in 0..my {
        for bx in 0..mx {
            let (y0, x0) = (by * bs, bx * bs);
            if block_mse(&org, &masked, y0, x0, bs) > block_mse(&org, &ave, y0, x0, bs) {
                mask[by * mx + bx] = 1;
            }
        }
    }
    mask
}

/// `EFEnonlinear.compress`: solve the per-tile filters, apply them, take the per-plane
/// rate-distortion decision, then the on/off mask search against the up-sampled picture.
pub fn decide(i: &EfeNonlinearInput<'_>) -> Result<EfeNonlinearOutput> {
    let (sv, sh) = (i.meta.s_ver as usize, i.meta.s_hor as usize);
    if !matches!((sv, sh), (1, 1) | (1, 2) | (2, 2)) {
        return Err(Error::Unsupported(
            "EFE non-linear: chroma format the reference software does not support",
        ));
    }
    let (org, rec) = (i.org, i.rec);
    if rec.y.h != org.y.h || rec.y.w != org.y.w {
        return Err(Error::InvalidArgument(
            "EFE non-linear: reconstruction and source sizes differ",
        ));
    }
    let eng = i.eng;
    // `self.luma` — the source luma at the chroma positions; its size is also the weight-loss
    // normaliser `h * w`.
    let org_y = downsample(&org.y, (sv, sh));
    let (h, w) = (org_y.h, org_y.w);
    let rec_y = downsample(&rec.y, (sv, sh));
    if (rec_y.h, rec_y.w) != (h, w)
        || (org.u.h, org.u.w) != (h, w)
        || (org.u.h, org.u.w) != (org.v.h, org.v.w)
        || (org.u.h, org.u.w) != (rec.u.h, rec.u.w)
        || (org.u.h, org.u.w) != (rec.v.h, rec.v.w)
    {
        return Err(Error::InvalidArgument(
            "EFE non-linear: reconstruction and source sizes differ",
        ));
    }
    let beta = BASE_MODEL_BETA[i.model_id.min(BASE_MODEL_BETA.len() - 1)];

    // `LumaAidedAdaptiveNonlinearFilter_encoder` + `_apply` on `ans = rec`.
    let solve = solve_tiles(&rec_y, org, rec)?;
    let weights = &solve.weights;
    let tiles = EfeNonlinearTiles {
        tile_width: solve.tile_width,
        tile_height: solve.tile_height,
        luma_min: solve.luma_min.clone(),
        luma_max: solve.luma_max.clone(),
        weights: [Some(weights[0].clone()), Some(weights[1].clone())],
    };
    let ntiles = tiles.luma_min.len();
    let mut filt = [rec.u.clone(), rec.v.clone()];
    dec::filter_plane(eng, (sv, sh), &tiles, 0, &weights[0], &rec.y, &mut filt[0])?;
    dec::filter_plane(eng, (sv, sh), &tiles, 0, &weights[1], &rec.y, &mut filt[1])?;

    // `beta * (after - before) - len(weights) * wP * lossModifier / 4 / (h * w)`: a plane
    // stays enabled while its gain pays for `8 * ntiles` weight codes.
    let pixels = (h * w) as f64;
    let cost = 8.0 * ntiles as f64 * WEIGHT_PRECISION * LOSS_MODIFIER / 4.0 / pixels;
    let (org_p, rec_p) = ([&org.u, &org.v], [&rec.u, &rec.v]);
    let before = [psnr(org_p[0], rec_p[0]), psnr(org_p[1], rec_p[1])];
    let mut enabled = [true, true];
    let mut ans = [filt[0].clone(), filt[1].clone()];
    for p in 0..2 {
        let loss = beta * (psnr(org_p[p], &filt[p]) - before[p]) - cost;
        if loss < 0.0 {
            enabled[p] = false;
            ans[p] = rec_p[p].clone();
        }
    }

    // `calculateOnOff` + `apply_OnoffSwitch`: masks exist only when the EFE linear filter
    // supplied its alternative picture.
    let mut masks: [Option<Vec<u8>>; 2] = [None, None];
    let mut mask_geometry = None;
    if let Some(up) = i.upsampled {
        let bs = BLOCK_SIZES[i.model_id.min(BLOCK_SIZES.len() - 1)];
        let (my, mx) = (org.u.h.div_ceil(bs), org.u.w.div_ceil(bs));
        let m = [
            on_off_mask(&org.u, &up.u, &ans[0], bs),
            on_off_mask(&org.v, &up.v, &ans[1], bs),
        ];
        // The candidate picture with both masks applied.
        let mut masked = [ans[0].clone(), ans[1].clone()];
        let geom = (bs, my, mx);
        dec::switch_plane(eng, geom, &m[0], &up.u, &mut masked[0])?;
        dec::switch_plane(eng, geom, &m[1], &up.v, &mut masked[1])?;
        // `h, w` of the *luma* plane for this loss (`ans.get_component('a').shape`).
        let full = (rec.y.h * rec.y.w) as f64;
        let mask_cost = my as f64 * mx as f64 * LOSS_MODIFIER / 2.0 / full;
        for (p, o) in [&org.u, &org.v].into_iter().enumerate() {
            let loss = beta * (psnr(o, &masked[p]) - psnr(o, &ans[p])) - mask_cost;
            if loss > 0.0 {
                masks[p] = Some(m[p].clone());
                ans[p] = masked[p].clone();
            }
        }
        if masks.iter().any(Option::is_some) {
            mask_geometry = Some((bs as u16, my as u16, mx as u16));
        }
    }

    // `encode_header`: `minnn` / `maxxx` start at `0` / `2^wP - 1`, so the fold never moves
    // them — `minSymbol` is always 0 and `maxSymbol` 65535; the codes go out unshifted.
    let nonlinear = (enabled[0] || enabled[1]).then(|| EfeNonlinearTiles {
        weights: [
            enabled[0].then(|| weights[0].clone()),
            enabled[1].then(|| weights[1].clone()),
        ],
        ..tiles
    });
    Ok(EfeNonlinearOutput {
        header: EfeNonlinearHeader {
            min_symbol: 0,
            max_symbol: u16::MAX,
            mask_geometry,
            masks,
            nonlinear,
        },
        filtered: Planes {
            y: rec.y.clone(),
            u: ans[0].clone(),
            v: ans[1].clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::ColourTransform;

    fn plane(h: usize, w: usize, f: impl Fn(usize, usize) -> f32) -> Tensor<f32> {
        Tensor::from_vec(1, h, w, (0..h * w).map(|i| f(i / w, i % w)).collect()).unwrap()
    }

    fn planes(y: Tensor<f32>, u: Tensor<f32>, v: Tensor<f32>) -> Planes {
        Planes { y, u, v }
    }

    fn meta() -> SourceMeta {
        SourceMeta {
            bit_depth: 8,
            s_ver: 1,
            s_hor: 1,
            c_ver: 1,
            c_hor: 1,
            colour_transform: ColourTransform::None,
        }
    }

    /// `Y[::s_ver, ::s_hor]` with a replicate pad on odd sizes.
    #[test]
    fn downsample_strides_and_pads() {
        let p = plane(5, 7, |y, x| (y * 10 + x) as f32);
        assert_eq!(downsample(&p, (1, 1)).data, p.data);
        let s = downsample(&p, (2, 2));
        assert_eq!((s.h, s.w), (3, 4));
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(s.data[y * 4 + x], (y * 2) as f32 * 10.0 + (x * 2) as f32);
            }
        }
        let s = downsample(&p, (1, 2));
        assert_eq!((s.h, s.w), (5, 4));
        assert_eq!(s.data[6], 14.0);
    }

    /// `calculateOnOff`'s three outcomes on one `bS` block each: filtered kept (0),
    /// alternative taken (2), and the plain average beating the winner (1).
    #[test]
    fn on_off_mask_marks_filtered_alternative_and_average() {
        let bs = 4;
        let org = plane(bs, bs * 3, |_, x| 50.0 + (x / bs) as f32);
        // Block 0: filt wins outright. Block 1: alt wins outright. Block 2: filt and alt
        // err equally — the mask picks 0 — then the average beats both.
        let filt = plane(bs, bs * 3, |_, x| match x / bs {
            0 => 51.0,
            1 => 90.0,
            _ => 60.0,
        });
        let alt = plane(bs, bs * 3, |_, x| match x / bs {
            0 => 30.0,
            1 => 52.0,
            _ => 41.0,
        });
        let mask = on_off_mask(&org, &alt, &filt, bs);
        assert_eq!(mask, [0, 2, 1]);
    }

    /// The solve recovers an exactly-representable residual: `B = relu(Y - ceiling_4)`
    /// yields weight 4 = 1.0 (`34815`) and zeros elsewhere (`32767`), with `lumaMin` /
    /// `lumaMax` quantised as `round(100 * v)`.
    #[test]
    fn solve_recovers_a_single_bin_weight() {
        let luma = plane(96, 128, |y, x| ((x + 2 * y) % 160) as f32);
        // min 0, max 159 -> gap 19.875 -> ceiling_4 = 79.5 (exact in f32).
        let rec = planes(
            luma.clone(),
            plane(96, 128, |_, _| 0.0),
            plane(96, 128, |_, _| 0.0),
        );
        let org = planes(
            luma.clone(),
            plane(96, 128, |y, x| (((x + 2 * y) % 160) as f32 - 79.5).max(0.0)),
            plane(96, 128, |_, _| 0.0),
        );
        let solve = solve_tiles(&luma, &org, &rec).unwrap();
        assert_eq!(
            (solve.tile_width, solve.tile_height),
            (128, 128),
            "one tile snapped to 64"
        );
        assert_eq!(solve.luma_min, [0]);
        assert_eq!(solve.luma_max, [15900]);
        let want_u = {
            let mut v = vec![32767u32; BINS];
            v[4] = 34815;
            v
        };
        assert_eq!(solve.weights[0], want_u);
        assert_eq!(solve.weights[1], vec![32767u32; BINS]);
    }

    /// `decide` end-to-end: a U residual the filter fits almost exactly keeps the U
    /// filter (and drops V); the small quantisation remainder makes the `up = org`
    /// alternative win every mask block. The header serialises and re-parses intact.
    #[test]
    fn decide_enables_switches_and_roundtrips() {
        let luma = plane(96, 128, |y, x| ((x + 2 * y) % 160) as f32);
        // relu(l - 79.5) is exact (w4 = 1.0); the `0.001 * l` slope integerises to
        // 2/2048 = 0.0009765…, so the filter output is close to but not bitwise the
        // source — the on/off switch then prefers the (perfect) alternative.
        let org = planes(
            luma.clone(),
            plane(96, 128, |y, x| {
                let l = ((x + 2 * y) % 160) as f32;
                (l - 79.5).max(0.0) + 0.001 * l
            }),
            plane(96, 128, |_, _| 0.0),
        );
        let rec = planes(luma, plane(96, 128, |_, _| 0.0), plane(96, 128, |_, _| 0.0));
        let eng = Engine::new();
        let m = meta();
        let out = decide(&EfeNonlinearInput {
            eng: &eng,
            meta: &m,
            model_id: 0,
            org: &org,
            rec: &rec,
            upsampled: Some(&org),
        })
        .unwrap();
        let t = out.header.nonlinear.as_ref().expect("U filter on");
        // V's residual is exactly zero: `before == after == inf` makes the reference's
        // `loss < 0` comparison NaN — the plane stays enabled with all-zero weights
        // (an upstream quirk, ported verbatim).
        assert!(
            t.weights[0].is_some() && t.weights[1].is_some(),
            "U on, V quirk-on"
        );
        assert_eq!(t.weights[0].as_deref().unwrap()[4], 34815);
        assert_eq!(t.weights[1].as_deref().unwrap(), &[32767u32; BINS]);
        // The alternative is the source itself: every block switches (one 128x128 block
        // covers the 128x96 chroma plane), and the output takes it.
        assert_eq!(out.header.mask_geometry, Some((128, 1, 1)));
        assert_eq!(out.header.masks[0].as_deref(), Some(&[2u8][..]));
        assert!(out.header.masks[1].is_none());
        assert_eq!(out.filtered.u.data, org.u.data);

        let pih = crate::encoder::picture_header(
            128,
            96,
            1,
            [0, 0],
            crate::encoder::EncodeParams::default(),
            &m,
        );
        let tools = crate::header::ToolHeader {
            efe_nonlinear: Some(out.header.clone()),
            ..Default::default()
        };
        let bytes = tools.write(&pih).unwrap();
        let parsed = crate::header::ToolHeader::parse(&bytes, &pih).unwrap();
        assert_eq!(parsed.efe_nonlinear, Some(out.header));
    }

    /// A residual the hinge basis cannot fit — a `±0.5` checkerboard — leaves the PSNR
    /// delta under the signalling cost: both planes drop and the header carries no tile
    /// block at all.
    #[test]
    fn decide_disables_without_gain() {
        let luma = plane(96, 128, |y, x| (x + y) as f32);
        let noise = plane(96, 128, |y, x| if (x + y) % 2 == 0 { 0.5 } else { -0.5 });
        let rec = planes(
            luma.clone(),
            plane(96, 128, |_, _| 3.0),
            plane(96, 128, |_, _| 3.0),
        );
        let org = planes(
            luma,
            plane(96, 128, |y, x| 3.0 + noise.data[y * 128 + x]),
            plane(96, 128, |y, x| 3.0 + noise.data[y * 128 + x]),
        );
        let eng = Engine::new();
        let out = decide(&EfeNonlinearInput {
            eng: &eng,
            meta: &meta(),
            model_id: 0,
            org: &org,
            rec: &rec,
            // No EFE linear alternative: the mask search is skipped entirely.
            upsampled: None,
        })
        .unwrap();
        assert!(out.header.nonlinear.is_none());
        assert!(out.header.mask_geometry.is_none());
        assert_eq!(out.filtered.u.data, rec.u.data);
    }
}
