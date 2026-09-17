//! Reconstruction stage of the decoder: entropy-stage output → picture.
//!
//! Ports `CcsGvaeSGMM.forward` / `decompress` (`ccs_sgmm_tool.py`) and the hyper-decode /
//! context / synthesis steps of `common_modules.py`, for a picture that is one region and one
//! synthesis tile. **Region and tile partitioning of this stage is not ported yet.**

use super::entropy::ComponentEntropy;
use crate::error::{Error, Result};
use crate::header::PictureHeader;
use crate::model::CommonModel;
use crate::model::mcm::upshuffle_psi;
use crate::model::synthesis::{SynthesisPrimary, SynthesisSecondary};
use crate::nn::fast::{BTensor, Engine};
use crate::tensor::Tensor;
use crate::tools::tiles::synthesis_tiles;

/// Latent-domain result for one component.
#[derive(Clone, Debug)]
pub struct Latent {
    pub psi: BTensor,
    pub y_hat: Tensor<f32>,
}

/// `hyper_decode_tile` + `decompress_ar_scale_tile` for a single tile covering the component.
pub fn reconstruct_latent(
    eng: &Engine,
    model: &CommonModel,
    e: &ComponentEntropy,
) -> Result<Latent> {
    let (h, w) = (e.residual.h, e.residual.w);
    let psi = model
        .hyper_decoder
        .forward(eng, &e.z_hat, h.div_ceil(2), w.div_ceil(2))?;
    let y_hat = match &model.context {
        Some(ctx) => ctx.decompress(eng, &e.residual, &psi)?,
        None => {
            let mut y = upshuffle_psi(&psi, h, w)?;
            for (v, &r) in y.data.iter_mut().zip(&e.residual.data) {
                *v += r;
            }
            y
        }
    };
    Ok(Latent { psi, y_hat })
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
    if hdr.regions.is_some() {
        return Err(Error::Unsupported(
            "region partitioning in the reconstruction stage",
        ));
    }
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
    let tiles = synthesis_tiles(
        h,
        w,
        y_hat[0].h,
        y_hat[0].w,
        hdr.components[0].synthesis_tiling,
    )?;

    let v = eng.tier.block();
    let mut rec_y = Tensor::<f32>::zeros(1, h, w)?;
    let mut rec_uv = Tensor::<f32>::zeros(2, h, w)?;
    for tile in &tiles {
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
        let ty = luma.forward(eng, &by, img.height, img.width)?;
        let tuv = chroma.forward(eng, &by, &buv, img.height, img.width)?;
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
