//! Reconstruction stage of the decoder: entropy-stage output → picture.
//!
//! Ports `CcsGvaeSGMM.forward` / `decompress` (`ccs_sgmm_tool.py`) and the hyper-decode /
//! context / synthesis steps of `common_modules.py`, including region partitioning (dependent and
//! independent) and synthesis tiling.

use enough::Stop;

use super::entropy::ComponentEntropy;
use crate::error::{Error, Result};
use crate::header::{PictureHeader, ToolHeader};
use crate::model::CommonModel;
use crate::model::mcm::upshuffle_psi;
use crate::model::synthesis::{SynthesisPrimary, SynthesisSecondary};
use crate::nn::fast::{BTensor, Engine};
use crate::tensor::Tensor;
use crate::tools::lsbs;
use crate::tools::regions::{Area, Plane, region_grid};
use crate::tools::tiles::synthesis_tiles;

/// Latent-domain result for one component.
#[derive(Clone, Debug)]
pub struct Latent {
    pub psi: Tensor<f32>,
    pub y_hat: Tensor<f32>,
}

/// `numSamplesTileOverlap` of the reference's hyper-decoder and context-model tile managers.
/// It is not signalled: both managers are built with the parameter's default, and with dependent
/// regions it decides how much of each region's result is dropped before merging
/// (`48 / 2 / 32 = 0` psi samples, `48 / 2 / 16 = 1` latent sample per interior side).
const HD_MCM_TILE_OVERLAP: usize = 48;

/// `tiling.get_data` + `tiling.assign_data`: copy `src[:, oy.., ox..]` into `dst` at `core`,
/// clamped the way tensor slicing clamps.
fn assign(dst: &mut Tensor<f32>, core: Area, src: &Tensor<f32>, (ox, oy): (usize, usize)) {
    let h = core
        .height
        .min(src.h.saturating_sub(oy))
        .min(dst.h.saturating_sub(core.y));
    let w = core
        .width
        .min(src.w.saturating_sub(ox))
        .min(dst.w.saturating_sub(core.x));
    let (dw, sw) = (dst.w, src.w);
    for c in 0..dst.c.min(src.c) {
        for y in 0..h {
            let s = &src.plane(c)[(oy + y) * sw + ox..][..w];
            let d = (core.y + y) * dw + core.x;
            dst.plane_mut(c)[d..d + w].copy_from_slice(s);
        }
    }
}

/// Rows/columns the hyper-decoder crops after its transposed convolution
/// (`cropping_layer(.., depth = 5)`, `parse_size_diff`); `divider` is 32 for luma and 16 for
/// chroma (`skip_depth_step`).
fn hyper_crop(len: usize, divider: usize) -> usize {
    2 * len.div_ceil(2 * divider) - len.div_ceil(divider)
}

/// Hyper-decoder and context model of one component, region by region
/// (`hyper_decode_tile`, `merge_psi_overlaps_of_tiles`, `extract_psi_for_mcm`,
/// `decompress_ar_scale_tile`, `merge_y_hat_overlaps_of_tiles`). Regions are processed in raster
/// order and merged by plain assignment, so where extended regions overlap the later one wins,
/// as in the reference.
pub fn reconstruct_latent(
    eng: &Engine,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    e: &ComponentEntropy,
) -> Result<Latent> {
    reconstruct_latent_with(eng, hdr, ccs, model, e, &enough::Unstoppable)
}

/// [`reconstruct_latent`] that checks `stop` per region and between network layers.
pub fn reconstruct_latent_with(
    eng: &Engine,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    e: &ComponentEntropy,
    stop: &dyn Stop,
) -> Result<Latent> {
    let (h, w) = (e.residual.h, e.residual.w);
    let chs = e.residual.c;
    let independent = hdr.regions.is_some_and(|r| r.independent);
    let img = region_grid(hdr, ccs, Plane::Image);
    let zg = region_grid(hdr, ccs, Plane::HyperLatent);
    let pg = region_grid(hdr, ccs, Plane::Psi);
    let lg = region_grid(hdr, ccs, Plane::Latent);
    let n = img.extended.len();
    let v = eng.tier.block();
    let (pic_h, pic_w) = (hdr.height as usize, hdr.width as usize);

    // 1. psi per region, merged.
    let mut psi = Tensor::<f32>::zeros(4 * chs, h.div_ceil(2), w.div_ceil(2))?;
    let mut psi_single = None;
    for r in 0..n {
        stop.check()?;
        let (it, zt) = (img.extended[r], zg.extended[r]);
        // The chroma hyper-decoder is told the half-size plane.
        let (th, tw, divider) = if ccs == 0 {
            (it.height, it.width, 32)
        } else {
            (it.height.div_ceil(2), it.width.div_ceil(2), 16)
        };
        let out_h = (2 * zt.height)
            .checked_sub(hyper_crop(th, divider))
            .ok_or(Error::InvalidData("region geometry"))?;
        let out_w = (2 * zt.width)
            .checked_sub(hyper_crop(tw, divider))
            .ok_or(Error::InvalidData("region geometry"))?;
        let z_tile;
        let z = if n == 1 {
            &e.z_hat
        } else {
            z_tile = e.z_hat.window(zt.x, zt.y, zt.width, zt.height)?;
            &z_tile
        };
        let t = model
            .hyper_decoder
            .forward_with(eng, z, out_h, out_w, stop)?;
        assign(&mut psi, pg.extended[r], &t.to_planar()?, (0, 0));
        if n == 1 && t.h == psi.h && t.w == psi.w {
            psi_single = Some(t);
        }
    }

    // 2. y_hat per region, merged.
    let mut y_hat = Tensor::<f32>::zeros(chs, h, w)?;
    for r in 0..n {
        stop.check()?;
        let (lt, pt, it) = (lg.extended[r], pg.extended[r], img.extended[r]);
        let (res_tile, psi_tile);
        let (res, psi_b) = match psi_single.take() {
            Some(p) => (&e.residual, p),
            None => {
                res_tile = e.residual.window(lt.x, lt.y, lt.width, lt.height)?;
                psi_tile = psi.window(pt.x, pt.y, pt.width, pt.height)?;
                (&res_tile, BTensor::from_planar(&psi_tile, v)?)
            }
        };
        let y = match &model.context {
            Some(ctx) => ctx.decompress_with(eng, res, &psi_b, stop)?,
            None => {
                let mut y = upshuffle_psi(&psi_b, res.h, res.w)?;
                for (o, &r) in y.data.iter_mut().zip(&res.data) {
                    *o += r;
                }
                y
            }
        };
        let (core, offset) = if independent {
            (lt, (0, 0))
        } else {
            // Picture-border branch of `_get_core_of_overlapping_tile`.
            let cut = HD_MCM_TILE_OVERLAP / 2 / 16;
            let left = if it.x == 0 { 0 } else { cut };
            let top = if it.y == 0 { 0 } else { cut };
            let right = if it.x + it.width >= pic_w { 0 } else { cut };
            let bottom = if it.y + it.height >= pic_h { 0 } else { cut };
            let cw = lt.width.checked_sub(left + right);
            let ch = lt.height.checked_sub(top + bottom);
            let (Some(cw), Some(ch)) = (cw, ch) else {
                return Err(Error::InvalidData("region smaller than its overlap"));
            };
            (Area::new(lt.x + left, lt.y + top, cw, ch), (left, top))
        };
        assign(&mut y_hat, core, &y, offset);
    }
    Ok(Latent { psi, y_hat })
}

/// Latent-space post-processing (`ls_processing.post_processing`): LSBS, when the tool header
/// enables it for component `ccs`. Runs on the merged `y_hat`, before synthesis.
pub fn post_process_latent(
    hdr: &PictureHeader,
    tools: &ToolHeader,
    ccs: usize,
    e: &ComponentEntropy,
    latent: &mut Latent,
) -> Result<()> {
    if tools.lsbs_enabled[ccs] {
        lsbs::apply(
            hdr.model_id as usize,
            &mut latent.y_hat,
            &e.residual,
            &e.likely,
        )?;
    }
    Ok(())
}

/// Reconstructed planes in the codec's internal range `[0, 255]`: luma at coded size minus the
/// non-displayed border, chroma subsampled to the coded chroma format.
#[derive(Clone, Debug)]
pub struct Planes {
    pub y: Tensor<f32>,
    pub u: Tensor<f32>,
    pub v: Tensor<f32>,
}

/// Synthesis of both components, tile by tile when the header enables synthesis tiling
/// (`ccs_sgmm_tool.py::forward`, `common_modules.py::decompress_y_hat_to_image_tile`).
pub fn synthesize(
    eng: &Engine,
    hdr: &PictureHeader,
    luma: &SynthesisPrimary,
    chroma: &SynthesisSecondary,
    y_hat: [&Tensor<f32>; 2],
) -> Result<Planes> {
    synthesize_with(eng, hdr, luma, chroma, y_hat, &enough::Unstoppable)
}

/// [`synthesize`] that checks `stop` per tile and between network layers.
pub fn synthesize_with(
    eng: &Engine,
    hdr: &PictureHeader,
    luma: &SynthesisPrimary,
    chroma: &SynthesisSecondary,
    y_hat: [&Tensor<f32>; 2],
    stop: &dyn Stop,
) -> Result<Planes> {
    let (h, w) = (hdr.height as usize, hdr.width as usize);
    let out_h = h - hdr.diff_display_height as usize;
    let out_w = w - hdr.diff_display_width as usize;

    // The reference looks the luma tile's y_hat up by the chroma tile's picture area, so both
    // components must be tiled identically.
    if hdr.components[0].synthesis_tiling != hdr.components[1].synthesis_tiling {
        return Err(Error::Unsupported(
            "different synthesis tiling for luma and chroma",
        ));
    }
    if y_hat[0].h != y_hat[1].h || y_hat[0].w != y_hat[1].w {
        return Err(Error::InvalidArgument("luma / chroma latent size mismatch"));
    }
    // Tiles only respect region borders when regions are independently decodable.
    let independent_regions = hdr
        .regions
        .filter(|r| r.independent)
        .map(|_| region_grid(hdr, 0, Plane::Image));
    let tiles = synthesis_tiles(
        h,
        w,
        y_hat[0].h,
        y_hat[0].w,
        hdr.components[0].synthesis_tiling,
        independent_regions.as_ref(),
    )?;

    let v = eng.tier.block();
    let mut rec_y = Tensor::<f32>::zeros(1, h, w)?;
    let mut rec_uv = Tensor::<f32>::zeros(2, h, w)?;
    for tile in &tiles {
        stop.check()?;
        let (img, lat) = (tile.image, tile.latent);
        let whole = lat.width == y_hat[0].w && lat.height == y_hat[0].h;
        let (by, buv) = if whole {
            (
                BTensor::from_planar(y_hat[0], v)?,
                BTensor::from_planar(y_hat[1], v)?,
            )
        } else {
            let win = |t: &Tensor<f32>| t.window(lat.x, lat.y, lat.width, lat.height);
            (
                BTensor::from_planar(&win(y_hat[0])?, v)?,
                BTensor::from_planar(&win(y_hat[1])?, v)?,
            )
        };
        let ty = luma.forward_with(eng, &by, img.height, img.width, stop)?;
        let tuv = chroma.forward_with(eng, &by, &buv, img.height, img.width, stop)?;
        for (dst, src) in [(&mut rec_y, &ty), (&mut rec_uv, &tuv)] {
            let (ox, oy) = tile.core_offset;
            for c in 0..dst.c {
                for y in 0..tile.core.height {
                    let s = &src.plane(c)[(oy + y) * src.w + ox..][..tile.core.width];
                    let d = (tile.core.y + y) * dst.w + tile.core.x;
                    dst.plane_mut(c)[d..d + tile.core.width].copy_from_slice(s);
                }
            }
        }
    }
    let rec_y = rec_y.crop(out_h, out_w)?;

    // rec_UV[:, :, :out_h:c_ver, :out_w:c_hor]
    let (sv, sh) = (hdr.c_ver as usize, hdr.c_hor as usize);
    let (ch, cw) = (out_h.div_ceil(sv), out_w.div_ceil(sh));
    let mut planes = [
        Tensor::<f32>::zeros(1, ch, cw)?,
        Tensor::<f32>::zeros(1, ch, cw)?,
    ];
    for (c, plane) in planes.iter_mut().enumerate() {
        for y in 0..ch {
            for x in 0..cw {
                plane.data[y * cw + x] = rec_uv.at(c, y * sv, x * sh);
            }
        }
    }
    let [u, v] = planes;
    Ok(Planes { y: rec_y, u, v })
}
