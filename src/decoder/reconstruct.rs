//! Reconstruction stage of the decoder: entropy-stage output → picture.
//!
//! Ports `CcsGvaeSGMM.forward` / `decompress` (`ccs_sgmm_tool.py`) and the hyper-decode /
//! context / synthesis steps of `common_modules.py`, including region partitioning (dependent and
//! independent) and synthesis tiling.

use enough::Stop;

use super::entropy::ComponentEntropy;
use super::stats::{Gate, Probe, Tick, for_each_chunk, join2, par_map, record};
use crate::error::{Error, Result};
use crate::header::{PictureHeader, ToolHeader};
use crate::model::CommonModel;
use crate::model::mcm::upshuffle_psi_par;
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
pub(crate) const HD_MCM_TILE_OVERLAP: usize = 48;

/// `tiling.get_data` + `tiling.assign_data`: copy `src[:, oy.., ox..]` into `dst` at `core`,
/// clamped the way tensor slicing clamps. One task per channel plane when `parallel` holds
/// (channels are disjoint).
fn assign_par(
    parallel: bool,
    dst: &mut Tensor<f32>,
    core: Area,
    src: &Tensor<f32>,
    (ox, oy): (usize, usize),
) {
    let h = core
        .height
        .min(src.h.saturating_sub(oy))
        .min(dst.h.saturating_sub(core.y));
    let w = core
        .width
        .min(src.w.saturating_sub(ox))
        .min(dst.w.saturating_sub(core.x));
    let (dw, sw) = (dst.w, src.w);
    let plane = dst.h * dst.w;
    for_each_chunk(parallel, &mut dst.data, plane, |c, dplane| {
        if c >= src.c {
            return;
        }
        let splane = src.plane(c);
        for y in 0..h {
            let s = &splane[(oy + y) * sw + ox..][..w];
            let d = (core.y + y) * dw + core.x;
            dplane[d..d + w].copy_from_slice(s);
        }
    });
}

/// Rows/columns the hyper-decoder crops after its transposed convolution
/// (`cropping_layer(.., depth = 5)`, `parse_size_diff`); `divider` is 32 for luma and 16 for
/// chroma (`skip_depth_step`).
pub(crate) fn hyper_crop(len: usize, divider: usize) -> usize {
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
    let gate = Gate::new(stop);
    reconstruct_latent_timed(eng, hdr, ccs, model, e, &gate, None)
}

/// [`reconstruct_latent_with`] that feeds `probe` with per-stage timings. Region work is
/// pooled (`eng.parallel`): the regions' extended windows overlap but only their disjoint
/// cores are merged — computed concurrently into tiles, then assigned in raster order, so
/// dependent regions keep the reference's "later wins" merge exactly.
pub(crate) fn reconstruct_latent_timed(
    eng: &Engine,
    hdr: &PictureHeader,
    ccs: usize,
    model: &CommonModel,
    e: &ComponentEntropy,
    stop: &dyn Stop,
    probe: Option<&Probe>,
) -> Result<Latent> {
    let t_lat = Tick::now();
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

    // 1. psi per region, merged. Each region's hyper-decoder run is independent work.
    let psi_tile = |r: usize| -> Result<(BTensor, Tensor<f32>)> {
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
        let planar = t.to_planar_par(eng)?;
        Ok((t, planar))
    };
    let t = Tick::now();
    let mut psi = Tensor::<f32>::zeros(4 * chs, h.div_ceil(2), w.div_ceil(2))?;
    let mut psi_single = None;
    if n == 1 {
        let (t, planar) = psi_tile(0)?;
        assign_par(eng.parallel, &mut psi, pg.extended[0], &planar, (0, 0));
        if t.h == psi.h && t.w == psi.w {
            psi_single = Some(t);
        }
    } else {
        let tiles = par_map(eng.parallel, n, |r| psi_tile(r).map(|(_, p)| p))?;
        for (r, planar) in tiles.iter().enumerate() {
            assign_par(eng.parallel, &mut psi, pg.extended[r], planar, (0, 0));
        }
    }
    record(Probe::field(probe, ccs, |p| &p.hyper), t);

    // 2. y_hat per region, merged.
    let t = Tick::now();
    let y_tile = |r: usize| -> Result<Tensor<f32>> {
        stop.check()?;
        let (lt, pt) = (lg.extended[r], pg.extended[r]);
        let (res_tile, psi_tile, psi_b);
        let (res, psi_r): (&Tensor<f32>, &BTensor) = match psi_single.as_ref() {
            Some(p) => (&e.residual, p),
            None => {
                res_tile = e.residual.window(lt.x, lt.y, lt.width, lt.height)?;
                psi_tile = psi.window(pt.x, pt.y, pt.width, pt.height)?;
                psi_b = BTensor::from_planar_par(eng, &psi_tile, v)?;
                (&res_tile, &psi_b)
            }
        };
        match &model.context {
            Some(ctx) => ctx.decompress_with(eng, res, psi_r, stop),
            None => {
                let mut y = upshuffle_psi_par(eng.parallel, psi_r, res.h, res.w)?;
                for_each_chunk(eng.parallel, &mut y.data, 16384, |i, chunk| {
                    let base = i * 16384;
                    let src = &res.data[base..base + chunk.len()];
                    for (o, &r) in chunk.iter_mut().zip(src) {
                        *o += r;
                    }
                });
                Ok(y)
            }
        }
    };
    let ys = par_map(eng.parallel, n, y_tile)?;
    let mut y_hat = Tensor::<f32>::zeros(chs, h, w)?;
    for (r, y) in ys.iter().enumerate() {
        let (lt, it) = (lg.extended[r], img.extended[r]);
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
        assign_par(eng.parallel, &mut y_hat, core, y, offset);
    }
    record(Probe::field(probe, ccs, |p| &p.mcm), t);
    record(Probe::field(probe, ccs, |p| &p.latent), t_lat);
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
    post_process_latent_par(cfg!(feature = "parallel"), hdr, tools, ccs, e, latent)
}

/// [`post_process_latent`], pooled elementwise work when `parallel` holds.
pub(crate) fn post_process_latent_par(
    parallel: bool,
    hdr: &PictureHeader,
    tools: &ToolHeader,
    ccs: usize,
    e: &ComponentEntropy,
    latent: &mut Latent,
) -> Result<()> {
    if tools.lsbs_enabled[ccs] {
        lsbs::apply_par(
            parallel,
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
    let gate = Gate::new(stop);
    let stop = &gate;
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
    // Tiles are written straight into the output planes: luma cropped to the displayed size,
    // chroma subsampled (`rec_UV[:, :, :out_h:c_ver, :out_w:c_hor]`). Full-size staging planes
    // would triple the memory held here.
    let (sv, sh) = (hdr.c_ver as usize, hdr.c_hor as usize);
    if sv == 0 || sh == 0 {
        return Err(Error::InvalidData("chroma subsampling factor"));
    }
    let (ch, cw) = (out_h.div_ceil(sv), out_w.div_ceil(sh));
    let mut rec_y = Tensor::<f32>::zeros(1, out_h, out_w)?;
    let mut rec_u = Tensor::<f32>::zeros(1, ch, cw)?;
    let mut rec_v = Tensor::<f32>::zeros(1, ch, cw)?;
    for tile in &tiles {
        stop.check()?;
        let (img, lat) = (tile.image, tile.latent);
        let whole = lat.width == y_hat[0].w && lat.height == y_hat[0].h;
        let blocked = |t: &Tensor<f32>| -> Result<BTensor> {
            let win;
            let src = if whole {
                t
            } else {
                win = t.window(lat.x, lat.y, lat.width, lat.height)?;
                &win
            };
            BTensor::from_planar_par(eng, src, v)
        };
        let (by, buv) = join2(eng.parallel, || blocked(y_hat[0]), || blocked(y_hat[1]));
        let (by, buv) = (by?, buv?);
        let (ox, oy) = tile.core_offset;
        let core = tile.core;

        // The two transforms share only their inputs: run them side by side so the tail of
        // one overlaps the ramp-up of the other.
        let (ty, tuv) = join2(
            eng.parallel,
            || luma.forward_with(eng, &by, img.height, img.width, stop),
            || chroma.forward_with(eng, &by, &buv, img.height, img.width, stop),
        );
        let (ty, tuv) = (ty?, tuv?);
        drop((by, buv));
        let cols = core.width.min(out_w.saturating_sub(core.x));
        if ty.h < oy + core.height || ty.w < ox + core.width {
            return Err(Error::InvalidData("synthesis tile smaller than its core"));
        }
        for y in 0..core.height.min(out_h.saturating_sub(core.y)) {
            let s = &ty.data[(oy + y) * ty.w + ox..][..cols];
            let d = (core.y + y) * out_w + core.x;
            rec_y.data[d..d + cols].copy_from_slice(s);
        }
        drop(ty);
        if tuv.c != 2 || tuv.h < oy + core.height || tuv.w < ox + core.width {
            return Err(Error::InvalidData("synthesis tile smaller than its core"));
        }
        for (c, dst) in [&mut rec_u, &mut rec_v].into_iter().enumerate() {
            let src = tuv.plane(c);
            // Picture rows / columns of this tile's core that survive the subsampling.
            for py in (core.y.next_multiple_of(sv)..(core.y + core.height).min(out_h)).step_by(sv) {
                let srow = &src[(oy + py - core.y) * tuv.w..][..tuv.w];
                let drow = &mut dst.data[(py / sv) * cw..][..cw];
                for px in
                    (core.x.next_multiple_of(sh)..(core.x + core.width).min(out_w)).step_by(sh)
                {
                    drow[px / sh] = srow[ox + px - core.x];
                }
            }
        }
    }
    Ok(Planes {
        y: rec_y,
        u: rec_u,
        v: rec_v,
    })
}
