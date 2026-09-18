//! eICCI model selection — the encode side of `icci_filter.py::compress` and
//! `model_idxes.py::encode_header`.
//!
//! Per filter tile the search scores every `(luma, chroma)` candidate pair of the active map
//! (the per-model two-entry short list, or the whole ten-network bank) with the configured
//! loss — MSE, MS-SSIM ([`crate::encoder::msssim`]) or the shipped `mixed` weighting — against
//! the source. The luma model that strictly improves the running luma error wins Y; the chroma
//! model whose largest of the three gains `[u + v, u, v]` strictly beats the running best wins
//! the chroma planes its argmax covers (both, only U, or only V). `use_YUV` /
//! `icci_use_shortList` / `icci_model_signalled_idx` follow the selection exactly as
//! `encode_header` writes them. The networks, the Haar front-end and the tile layout are the
//! decoder's (`crate::filters::icci`, `crate::model::icci`); the search only picks and signals.
//!
//! The reference decides on the reconstruction once, after `model.compress`
//! (`coding_engine.py::compress`), and only when the source is 4:4:4 — `compress` disables the
//! tool for a subsampled source whatever the config says.

use alloc::vec::Vec;

use crate::decoder::reconstruct::Planes;
use crate::encoder::colour::{SourceImage, SourceMeta};
use crate::encoder::msssim;
use crate::error::{Error, Result};
use crate::filters::icci::{CHROMA_SHORT_LIST, FilterTile, LUMA_SHORT_LIST};
use crate::header::{IcciHeader, IcciTile, OperatingPoint, PictureHeader, SynthesisTiling};
use crate::model::ModelSource;
use crate::model::icci::{self, NetCache};
use crate::nn::fast::Engine;
use crate::tensor::Tensor;

/// `post_filters.eICCI.loss_type` (`cfg/tools/eICCI.json`): the loss the per-tile search
/// minimizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EicciLoss {
    /// `calculate_mse` alone.
    Mse,
    /// `calculate_msssim` alone (`1 - ms_ssim` per channel).
    MsSsim,
    /// `calculate_mixed`: `weight[0] * mse + weight[1] * msssim` per channel — the shipped
    /// configuration (`luma_loss_weights = [5, 1]`, `chroma_loss_weights = [1, 0]`).
    #[default]
    Mixed,
}

/// `IcciParams.process_short_list`, the loss weights and the filter's tile manager — the
/// encoder-facing knobs of `post_filters.eICCI`. `tile_samples == u32::MAX` is the
/// reference's `-1` (never tile). [`Default`] is the shipped `cfg/tools/eICCI.json`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EicciConfig {
    /// `loss_type`.
    pub loss: EicciLoss,
    /// `process_short_list`: candidates and signalled indices through the per-model short
    /// lists instead of the full bank.
    pub short_list: bool,
    /// `numSamplesPerTile`: the filter tiles when this is below the displayed sample count.
    pub tile_samples: u32,
    /// `numSamplesTileOverlap` (a multiple of the 16-sample alignment).
    pub tile_overlap: u32,
    /// `luma_loss_weights` (`[mse, ms-ssim]` of the `mixed` loss on the Y channel).
    pub luma_weights: [f32; 2],
    /// `chroma_loss_weights`, applied to U and V alike.
    pub chroma_weights: [f32; 2],
}

impl Default for EicciConfig {
    /// The shipped `cfg/tools/eICCI.json`: `mixed` loss, weights `[5, 1]` / `[1, 0]`, the
    /// short lists, `numSamplesPerTile = 4194304`, `numSamplesTileOverlap = 48`.
    fn default() -> Self {
        Self {
            loss: EicciLoss::Mixed,
            short_list: true,
            tile_samples: 4_194_304,
            tile_overlap: 48,
            luma_weights: [5.0, 1.0],
            chroma_weights: [1.0, 0.0],
        }
    }
}

impl EicciConfig {
    /// `self.skip_luma`: `sum(luma_loss_weights) == 0` — the luma answer is the unfiltered
    /// plane, so no luma candidate can strictly improve the running error and Y stays
    /// unfiltered.
    fn skip_luma(&self) -> bool {
        self.luma_weights[0] + self.luma_weights[1] == 0.0
    }

    /// `self.skip_msssim`: `loss_weights[1].sum() == 0` — no MS-SSIM term on any channel, the
    /// reference substitutes `[1, 1, 1]` (which the zero weights then discard anyway).
    fn skip_msssim(&self) -> bool {
        self.luma_weights[1] + 2.0 * self.chroma_weights[1] == 0.0
    }
}

/// `torch.argmax` returns the first maximum.
fn argmax3(v: [f32; 3]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `calculate_mse`: `mean((a - b)^2)` per channel — `f64` sums (fixed order, identical on
/// every engine), rounded once.
fn mse(a: &Tensor<f32>, b: &Tensor<f32>) -> f32 {
    let n = a.data.len() as f64;
    let s: f64 = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum();
    (s / n) as f32
}

/// `self.loss(tile_org, tile_enh)`: the per-channel error of the candidate picture.
fn channel_errors(
    eng: &Engine,
    cfg: &EicciConfig,
    org: &[Tensor<f32>; 3],
    enh: &[Tensor<f32>; 3],
) -> Result<[f32; 3]> {
    let mut out = [0.0f32; 3];
    for (c, e) in out.iter_mut().enumerate() {
        *e = match cfg.loss {
            EicciLoss::Mse => mse(&org[c], &enh[c]),
            EicciLoss::MsSsim => 1.0 - msssim::ms_ssim(eng, &org[c], &enh[c])?,
            EicciLoss::Mixed => {
                // `loss_weights` is `torch.tensor([lw, cw, cw]).T`: row 0 the MSE weights,
                // row 1 the MS-SSIM weights, luma on channel 0, chroma on 1 and 2.
                let w = if c == 0 {
                    cfg.luma_weights
                } else {
                    cfg.chroma_weights
                };
                // `skip_msssim` substitutes `msssim = 1` per channel — a zero error.
                let ms_err = if cfg.skip_msssim() {
                    0.0
                } else {
                    1.0 - msssim::ms_ssim(eng, &org[c], &enh[c])?
                };
                w[0] * mse(&org[c], &enh[c]) + w[1] * ms_err
            }
        };
    }
    Ok(out)
}

/// `get_current_error`: the luma error and the three-entry chroma error `[u + v, u, v]`.
fn current_error(
    eng: &Engine,
    cfg: &EicciConfig,
    org: &[Tensor<f32>; 3],
    enh: &[Tensor<f32>; 3],
) -> Result<(f32, [f32; 3])> {
    let e = channel_errors(eng, cfg, org, enh)?;
    Ok((e[0], [e[1] + e[2], e[1], e[2]]))
}

/// `enhanced[:, :, :h, :w]` — the multiple-of-4 padding cropped — clamped to `[0, 1]`
/// (`current_tile_enhanced` right before the loss).
fn crop_clamp(full: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
    let mut t = full.window(0, 0, w, h)?;
    for v in &mut t.data {
        *v = v.clamp(0.0, 1.0);
    }
    Ok(t)
}

/// `setup_tiles_enc` on the displayed size, then `tile_layout`
/// (`_init_image_tiles_with_overlap` + `_adjust_boundary_tiles`, minimum 176). `None` — the
/// filter is one tile — when `numSamplesPerTile` is `-1` (`u32::MAX`) or covers the picture.
pub fn enc_tile_layout(
    height: usize,
    width: usize,
    cfg: &EicciConfig,
) -> Result<(Option<SynthesisTiling>, Vec<FilterTile>)> {
    let enabled =
        cfg.tile_samples != u32::MAX && (cfg.tile_samples as u64) < height as u64 * width as u64;
    let tiling = enabled.then(|| {
        // `floor(sqrt(numSamplesPerTile))` rounded up to the 16-sample alignment.
        let size = (libm::sqrt(cfg.tile_samples as f64) as usize).div_ceil(16) * 16;
        SynthesisTiling {
            tile_size: size as u32,
            overlap: cfg.tile_overlap,
        }
    });
    if let Some(t) = tiling
        && (t.overlap % 16 != 0 || t.tile_size / 16 > u8::MAX as u32 || t.overlap / 16 > 31)
    {
        return Err(Error::InvalidArgument("eICCI: tile size/overlap"));
    }
    let tiles = crate::filters::icci::tile_layout(height, width, tiling)?;
    Ok((tiling, tiles))
}

/// `org_img_i` of `compress`: the source's YUV planes in the filter's internal `[0, 1]`
/// range at the source's own size (`convert_range_` + `to_YUV_`; `to_444_` is the identity
/// for the only sources the tool runs on). A `diff_display` border means this is larger
/// than the displayed `rec` — [`select`] reads its top-left `rec`-sized area.
pub fn org_planes(src: &SourceImage, meta: &SourceMeta) -> Result<Planes> {
    let sp = crate::encoder::colour::source_planes_01(src, meta)?;
    let (w, h) = (src.width(), src.height());
    let plane = |data: Vec<f32>, ph, pw| Tensor::from_vec(1, ph, pw, data);
    Ok(Planes {
        y: plane(sp.luma, h, w)?,
        u: plane(sp.u, sp.chroma_height, sp.chroma_width)?,
        v: plane(sp.v, sp.chroma_height, sp.chroma_width)?,
    })
}

/// `EfficientICCIFilter.compress`: pick the per-tile luma and chroma networks of `op`'s bank
/// (`model_id` rows the short lists) and build the [`IcciHeader`] carrying the selection.
///
/// `org` is the source's YUV planes already in the internal `[0, 1]` range ([`org_planes`])
/// and `rec` the just-decoded picture in `[0, 255]`, both at 4:4:4 (the reference's
/// `to_444_()` upsamples chroma, but `compress` runs this filter only for a 4:4:4 source —
/// `Ok(None)` otherwise, matching its `set_enable(False)`).
#[allow(clippy::too_many_arguments)]
pub fn select(
    eng: &Engine,
    models: &dyn ModelSource,
    cache: &NetCache,
    hdr: &PictureHeader,
    op: OperatingPoint,
    org: &Planes,
    rec: &Planes,
    cfg: &EicciConfig,
    stop: &dyn enough::Stop,
) -> Result<Option<IcciHeader>> {
    // `compress`'s first act: a subsampled source disables the tool.
    if hdr.s_ver != 1 || hdr.s_hor != 1 {
        return Ok(None);
    }
    let (h, w) = (rec.y.h, rec.y.w);
    let same = |p: &Tensor<f32>| p.c == 1 && p.h == h && p.w == w;
    // `org` holds the source at 4:4:4 — a `diff_display` border can make it larger than the
    // displayed `rec`; the tile windows read its top-left `h` x `w` (`tiling.get_data`).
    if h == 0
        || w == 0
        || rec.y.c != 1
        || org.y.c != 1
        || org.y.h < h
        || org.y.w < w
        || org.u.c != 1
        || org.v.c != 1
        || (org.u.h, org.u.w) != (org.y.h, org.y.w)
        || (org.v.h, org.v.w) != (org.y.h, org.y.w)
        || !same(&rec.u)
        || !same(&rec.v)
    {
        return Err(Error::InvalidArgument("eICCI: expects 4:4:4 planes"));
    }
    let (tiling, tiles) = enc_tile_layout(h, w, cfg)?;
    // `IcciHeader::parse` derives the tile count from the coded size: with a non-displayed
    // border it would differ (the decoder rejects the combination the same way).
    if tiling.is_some() && (hdr.height as usize, hdr.width as usize) != (h, w) {
        return Err(Error::Unsupported(
            "eICCI tiling on a picture with a non-displayed border",
        ));
    }
    // `map_idx`: the search space per channel for this model at this op. The long map is
    // `all_models` — every one of the ten networks, for every model id.
    const LONG: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let model_id = hdr.model_id as usize;
    let (map_y, map_uv): (&[u8], &[u8]) = if cfg.short_list {
        let out_of_range = || Error::InvalidArgument("eICCI: model id out of range");
        (
            LUMA_SHORT_LIST[op as usize]
                .get(model_id)
                .ok_or_else(out_of_range)?,
            CHROMA_SHORT_LIST[op as usize]
                .get(model_id)
                .ok_or_else(out_of_range)?,
        )
    } else {
        if model_id >= 5 {
            return Err(Error::InvalidArgument("eICCI: model id out of range"));
        }
        (&LONG, &LONG)
    };

    // `img.convert_range_(internal_range)`: the decoder-side `[0, 255]` picture back to
    // `[0, 1]` (`org` already is — `source_planes_01` normalises directly).
    let scale = |p: &Tensor<f32>| -> Result<Tensor<f32>> {
        let mut t = p.clone();
        for v in &mut t.data {
            *v /= 255.0;
        }
        Ok(t)
    };
    let rec01 = [scale(&rec.y)?, scale(&rec.u)?, scale(&rec.v)?];

    let mut out = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        stop.check()?;
        let a = tile.image;
        let (th, tw) = (a.height, a.width);
        let win = |p: &Tensor<f32>| p.window(a.x, a.y, tw, th);
        let tile_org = [win(&org.y)?, win(&org.u)?, win(&org.v)?];
        let tile_rec = [win(&rec01[0])?, win(&rec01[1])?, win(&rec01[2])?];
        // `padding_layer` + `process_dwt444_2`: the replicate-padded tile as 48 sub-bands.
        let input = icci::tile_input(eng, [&rec01[0], &rec01[1], &rec01[2]], (a.x, a.y, tw, th))?;

        let (mut running_y, initial_uv) = current_error(eng, cfg, &tile_org, &tile_rec)?;
        // `uv_gain` starts at zeros and `gain_index` at the first argmax (0).
        let mut uv_gain = [0.0f32; 3];
        let mut gain_index = 0usize;
        let mut selection = [0u8; 3];

        for (&my, &muv) in map_y.iter().zip(map_uv) {
            // `compress_tile`: with `skip_luma` the luma answer is the unfiltered plane, so
            // the candidate's luma error equals the initial one and can never win Y.
            let enh_y = if cfg.skip_luma() {
                tile_rec[0].clone()
            } else {
                let net = cache.get(models, op, my as usize, eng)?;
                let corr = net.luma_correction(eng, &input)?;
                crop_clamp(&icci::tile_output(eng, &input, 0, &corr)?, th, tw)?
            };
            let net_uv = cache.get(models, op, muv as usize, eng)?;
            let [cu, cv] = net_uv.chroma_corrections(eng, &input, [true, true])?;
            let enh = [
                enh_y,
                crop_clamp(
                    &icci::tile_output(
                        eng,
                        &input,
                        1,
                        &cu.ok_or(Error::InvalidData("internal: eICCI chroma"))?,
                    )?,
                    th,
                    tw,
                )?,
                crop_clamp(
                    &icci::tile_output(
                        eng,
                        &input,
                        2,
                        &cv.ok_or(Error::InvalidData("internal: eICCI chroma"))?,
                    )?,
                    th,
                    tw,
                )?,
            ];
            let (cur_y, cur_uv) = current_error(eng, cfg, &tile_org, &enh)?;
            if cur_y < running_y {
                running_y = cur_y;
                selection[0] = my + 1;
            }
            let cur_gain = [
                initial_uv[0] - cur_uv[0],
                initial_uv[1] - cur_uv[1],
                initial_uv[2] - cur_uv[2],
            ];
            let cur_index = argmax3(cur_gain);
            if cur_gain[cur_index] > uv_gain[gain_index] {
                uv_gain = cur_gain;
                gain_index = cur_index;
                selection[1] = muv + 1;
                selection[2] = muv + 1;
                // `gain_index` 1 keeps only U, 2 keeps only V.
                match gain_index {
                    1 => selection[2] = 0,
                    2 => selection[1] = 0,
                    _ => {}
                }
            }
        }

        // `encode_header`: `use_YUV` from the selection, then `map.index(model - 1)` — the
        // position of the model inside the active map, which is what the stream signals.
        let index_of = |map: &[u8], sel: u8| -> Result<u8> {
            map.iter()
                .position(|&m| m == sel - 1)
                .map(|i| i as u8)
                .ok_or(Error::InvalidData("eICCI: selected model not in the map"))
        };
        out.push(IcciTile {
            use_yuv: [selection[0] != 0, selection[1] != 0, selection[2] != 0],
            // `icci_use_shortList` is coded only when a plane is enabled; with nothing
            // selected the parsed field is the decoder default — emit the same.
            short_list: cfg.short_list && selection.iter().any(|&s| s != 0),
            index_y: if selection[0] != 0 {
                index_of(map_y, selection[0])?
            } else {
                0
            },
            index_uv: if selection[1] != 0 || selection[2] != 0 {
                // `(model_selection[1] or model_selection[2]) - 1`.
                index_of(
                    map_uv,
                    if selection[1] != 0 {
                        selection[1]
                    } else {
                        selection[2]
                    },
                )?
            } else {
                0
            },
        });
    }
    Ok(Some(IcciHeader { tiling, tiles: out }))
}
