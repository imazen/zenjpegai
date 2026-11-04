//! `coding_tools/rdlr/` — Rate-Distortion Latent Refinement: the encoder-side
//! iterative latent optimiser. `process_y` (luma `rdlr`, chroma `--num-ite-uv`),
//! `process_z` (`--num-ite-z`, both `z_Y` and `z_UV`), `feature_clipping`, QMAP
//! 0/1/2 and `independent_beta_UV` are all ported — the reference ships every
//! one at `0`/off, so each is opt-in here.
//!
//! At b9e573f the reference's RDLR is dead code — five stale call sites crash
//! before the first iteration (`scripts/ref_vectors/dump_rdlr.py` restores each
//! to the API it was written against and dumps every iteration's tensors; the
//! oracle tests below replay them).
//!
//! This module ports `rdlr_comp_Y.py::get_current_rate_point`: on a candidate
//! luma latent it re-runs `_compress_ar_scale` (the context model's
//! quantise/dequantise), reconstructs `y_hat` exactly as the decoder does,
//! decodes it through the luma synthesis, and measures
//!
//! - the *rate point*: `z` bits from the `ECLibLH` factorized model (the real
//!   entropy stream is unchanged by refinement) plus the `GMProbModel`
//!   likelihood of the continuous residual `y - means_hat`, with
//!   `means_hat = y_hat - resi_hat` — all divided by the source pixel count;
//! - PSNR and MS-SSIM on the optimised part of the reconstruction.
//!
//! The gradient path (a hand-derived VJP through the same chain) and the
//! Adagrad/StepLR driver land on top of this forward evaluation.

use alloc::vec::Vec;

use crate::encoder::msssim;
use crate::error::{Error, Result};
use crate::model::common::{SIGMA_IDX_MAX, SIGMA_PRECISION, Z_OFFSET};
use crate::model::hyper_decoder;
use crate::model::mcm;
use crate::model::synthesis::{SynthesisPrimary, SynthesisSecondary};
use crate::nn::fast::math;
use crate::nn::fast::{BTensor, Engine};
use crate::tensor::Tensor;
use crate::tools::gain::log_k;
use crate::tools::regions::Area;

use super::compress_context_free;
use super::{compress_regions, lh};

/// `ln(sigma_quant_min)` — `pipeline.json`'s `sigma_quant_min = 0.11`, the same
/// value `GMProbModel` lower-bounds its scales at (`scale_bound` defaults to it).
#[allow(clippy::excessive_precision)]
const LN_SIGMA_MIN: f32 = -2.207274913787842;
/// `GMProbModel.scale_bound`, the `LowerBound` on the linear scales in `_likelihood`.
const SCALE_BOUND: f32 = 0.11;
/// `GMProbModel.likelihood_bound` — `LowerBound(1e-9)` on every likelihood.
const LIKELIHOOD_BOUND: f64 = 1e-9;

/// What `get_current_rate_point` measures — plus the intermediates the gradient
/// path reuses (`y_hat` through synthesis, `resi_q`/`psi` through the context
/// model, `rec` through the metrics).
pub(crate) struct RatePoint {
    /// `test_bpp_Y`: (factorized `z` bits + `-log2` GM likelihood bits) / pixels.
    pub bpp: f64,
    /// `20·log10(255) - 10·log10(mse + 1e-10)` on the optimised part, luma.
    pub psnr: f64,
    /// `ms_ssim(rec, original, data_range = 255)` on the optimised part, luma.
    pub msssim: f64,
    /// `resi_hat` — the quantised residual as integers (`residual_quant`).
    pub residual_q: Tensor<i16>,
    /// The merged hyper-decoder output the quantisation was measured against.
    pub psi: Tensor<f32>,
    /// `y_hat = residual_dq + means` — the reconstructed latent, decoder-exact.
    pub y_hat: Tensor<f32>,
    /// The reconstructed picture plane (0..255) at the component's resolution.
    pub rec: Tensor<f32>,
    /// `-log2` GM likelihood of `y - means_hat`, in bits (the `rate_y` term).
    /// Read by the oracle test only.
    #[allow(dead_code)]
    pub rate_y: f64,
    /// The synthesis's VJP tape (present when `rate_point_y` built `rec`).
    pub synth_tape: Option<crate::model::synthesis::PrimaryTape>,
    /// `compress`'s per-stage forward tape — `backward`'s `compress_backward`
    /// consumes it instead of replaying the whole MCM forward.
    pub tape: Vec<mcm::StageTape>,
}

/// `rate_y`/`rate_uv` — `Σ -log2` of the `GMProbModel` likelihood of
/// `resi = y - means_hat = y - y_hat + residual_q` (mixed units, the reference's
/// own formula), over every position: no skip mask, no `num_chs` truncation —
/// `_compress_ar_scale`'s continuous-residual rate term.
pub(crate) fn gm_rate(
    y: &Tensor<f32>,
    y_hat: &Tensor<f32>,
    residual_q: &Tensor<i16>,
    scale_log: &Tensor<i32>,
) -> f64 {
    let log_k = log_k() as f32;
    let inv = 1.0f32 / (1u32 << SIGMA_PRECISION) as f32;
    let mut rate = 0.0f64;
    for ch in 0..residual_q.c {
        let (r, s, yq, yh) = (
            residual_q.plane(ch),
            scale_log.plane(ch),
            y.plane(ch),
            y_hat.plane(ch),
        );
        for i in 0..r.len() {
            let resi = yq[i] - yh[i] + r[i] as f32;
            let scale_lin = math::exp(s[i] as f32 * inv * log_k + LN_SIGMA_MIN);
            let idx = (scale_lin + 64.0)
                .mul_add(1.0 / 128.0, 0.0)
                .clamp(0.0, 34.0);
            let sigma = math::exp(idx * log_k + LN_SIGMA_MIN).max(SCALE_BOUND);
            rate += gm_nbits(resi, sigma);
        }
    }
    rate
}

/// `RDLRCompY.get_current_rate_point` for one luma latent tile: the rate point
/// and both metrics. `original` is the source luma at 0..255 over `area` (the
/// optimised part of the picture tile; the whole picture for a single tile).
/// `pixels` is the *source* picture's pixel count — `test_bpp_Y` divides by it
/// even when `y` is a tile.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rate_point_y(
    eng: &Engine,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    synthesis: &SynthesisPrimary,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    area: Area,
    pixels: f64,
    orig_ms: Option<&msssim::MsOrig>,
    psi_in: Option<&Tensor<f32>>,
    stop: &dyn enough::Stop,
) -> Result<RatePoint> {
    // `_compress_ar_scale(y, decisions)`: residual quantisation per region. The
    // straight-through step the gradient path replaces with identity lives
    // inside (residual_q vs residual_dq). `psi` is `z_hat`'s hyper-decoder output —
    // fixed across `optimize_y`'s iterations — precomputed once by the caller.
    let out = compress_regions(
        eng, hdr, 0, model, z_hat, y, mask, q, num_chs, original.h, original.w, y.h, y.w, psi_in,
        stop,
    )?;
    let mcm::Compressed {
        residual_q,
        psi,
        y_hat,
        tape: mcm_tape,
        ..
    } = out;

    // `means_hat = y_hat - resi_hat`; `resi = y - means_hat = y - y_hat + resi_q`
    // (the residual's quantiser *index* — mixed units is the reference's own
    // formula). The GM likelihood sums over every position (no skip mask, no
    // num_chs truncation). `entropy(resi, scale_lin, 0)` with
    // `scale_table_precision = 7` applies `GMProbModel._index_to_scale` — a
    // float "bitwise shift" `(scale_lin + 2^6) / 2^7` clamped to `[0, level-1]`,
    // then `exp(idx·log_k + ln σ_min) · 2^7` — and `_likelihood` divides by
    // `2^7` again, so the effective sigma is
    // `max(exp(clamp((scale_lin+64)/128, 0, 34)·log_k + ln σ_min), 0.11)` where
    // `scale_lin = exp(scale_log/2^7 · log_k + ln σ_min)`.
    let rate_y = gm_rate(y, &y_hat, &residual_q, scale_log);

    // `tool.decoder(y_new2, h, w)` — the luma synthesis on the reconstructed
    // latent.
    let (rec, tape) = synthesis.forward_tape(
        eng,
        &BTensor::from_planar(&y_hat, eng.tier.block())?,
        original.h,
        original.w,
        stop,
    )?;
    let (psnr, msssim) = distortion(eng, &rec, original, area, orig_ms)?;

    let z_bits = lh.z_bits(z_hat);
    let bpp = (z_bits + rate_y) / pixels;
    Ok(RatePoint {
        bpp,
        psnr,
        msssim,
        residual_q,
        psi,
        y_hat,
        rec,
        synth_tape: Some(tape),
        rate_y,
        tape: mcm_tape,
    })
}

/// `get_luma_tile_rate` / `get_current_rate_point(only_rate=True)` — the luma
/// tile's rate term (`z` + residual bits per pixel) that chroma RDLR's joint
/// rate adds. Compresses the luma tile and skips the synthesis/distortion the
/// full `rate_point_y` computes — upstream's `only_rate` skips the same.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_tile_rate(
    eng: &Engine,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    img_h: usize,
    img_w: usize,
    pixels: f64,
    stop: &dyn enough::Stop,
) -> Result<f64> {
    let out = compress_regions(
        eng, hdr, 0, model, z_hat, y, mask, q, num_chs, img_h, img_w, y.h, y.w, None, stop,
    )?;
    let rate_y = gm_rate(y, &out.y_hat, &out.residual_q, scale_log);
    Ok((lh.z_bits(z_hat) + rate_y) / pixels)
}

/// `RDLRCompUV.get_current_rate_point` for one chroma latent tile: the UV rate
/// (`test_bpp_UV`) and the per-plane PSNRs. `sup` is `sup_info` — the (refined)
/// luma `y_hat` decimated `[:, :, ::2, ::2]` into the chroma decoder's input
/// planes. `original` is the source chroma `[2, H, W]` over `area` at 0..255.
pub(crate) struct RatePointUV {
    /// `test_bpp_UV`: (factorized `z` bits + `-log2` GM likelihood bits) / pixels.
    pub bpp: f64,
    /// `psnr` of the reconstructed U plane against `original[0]` over `area`.
    pub psnr_u: f64,
    /// `psnr` of the reconstructed V plane against `original[1]` over `area`.
    pub psnr_v: f64,
    pub residual_q: Tensor<i16>,
    pub psi: Tensor<f32>,
    pub y_hat: Tensor<f32>,
    /// The two reconstructed chroma planes `[2, H, W]`.
    pub rec: Tensor<f32>,
    #[allow(dead_code)]
    pub rate_uv: f64,
    /// The cube flags — `compress_context_free_backward` rebuilds `mask2` from
    /// them (the context-free path keeps no MCM tape).
    pub cube_flag: Vec<bool>,
    /// The secondary synthesis's VJP tape (`backward_uv` consumes it).
    pub synth_tape: Option<crate::model::synthesis::SecondaryTape>,
}

/// `RDLRCompUV.get_current_rate_point`: the chroma analogue of [`rate_point_y`]
/// — `_compress_ar_scale` on the chroma latent, the UV residual rate, then the
/// secondary synthesis on `cat(sup_info, y_hat)` and per-plane PSNR.
///
/// Called by the chroma `rdlr_y_iteration` driver (the next piece of the
/// chroma-RDLR port); the evaluator itself is complete and mirrors `rate_point_y`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn rate_point_uv(
    eng: &Engine,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    synthesis_uv: &SynthesisSecondary,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    sup: &BTensor,
    area: Area,
    pixels: f64,
    psi_in: Option<&Tensor<f32>>,
    stop: &dyn enough::Stop,
) -> Result<RatePointUV> {
    let out = compress_regions(
        eng,
        hdr,
        1,
        model,
        z_hat,
        y,
        mask,
        q,
        num_chs,
        original.h * 2,
        original.w * 2,
        y.h,
        y.w,
        psi_in,
        stop,
    )?;
    let mcm::Compressed {
        residual_q,
        psi,
        y_hat,
        cube_flag,
        ..
    } = out;

    let rate_uv = gm_rate(y, &y_hat, &residual_q, scale_log);

    // `tool.decoder(cat(sup_info, y_hat), h, w)` — the secondary synthesis takes
    // the luma latent as its first-stage input planes.
    let (rec, synth_tape) = synthesis_uv.forward_tape(
        eng,
        sup,
        &BTensor::from_planar(&y_hat, eng.tier.block())?,
        original.h,
        original.w,
        stop,
    )?;
    let psnr_u = psnr_plane(&rec, 0, original, 0, area);
    let psnr_v = psnr_plane(&rec, 1, original, 1, area);

    let z_bits = lh.z_bits(z_hat);
    let bpp = (z_bits + rate_uv) / pixels;
    Ok(RatePointUV {
        bpp,
        psnr_u,
        psnr_v,
        residual_q,
        psi,
        y_hat,
        rec,
        rate_uv,
        cube_flag,
        synth_tape: Some(synth_tape),
    })
}

/// `psnr` of `rec[plane]` against `original[plane]` inside `area` —
/// `20·log10(255) - 10·log10(mse + 1e-10)`, the same formula `distortion` uses
/// on the luma plane.
#[allow(dead_code)]
fn psnr_plane(
    rec: &Tensor<f32>,
    rec_plane: usize,
    original: &Tensor<f32>,
    orig_plane: usize,
    area: Area,
) -> f64 {
    let (ah, aw) = (area.height, area.width);
    let mut mse = 0.0f64;
    for row in 0..ah {
        for x in 0..aw {
            let o = original.plane(orig_plane)[(area.y + row) * original.w + area.x + x];
            let r = rec.plane(rec_plane)[(area.y + row) * rec.w + area.x + x];
            mse += ((o - r) as f64) * ((o - r) as f64);
        }
    }
    mse /= (ah * aw) as f64;
    20.0 * (255.0f64).log10() - 10.0 * (mse + 1e-10).log10()
}

/// `PSNR`'s input gradient: `psnr = C - 10·log10(mse + eps)`, `mse =
/// mean((a-b)²)` over `area` — `d(rec[i]) = d_psnr · (-10/(ln10·mse)) ·
/// 2·(rec[i]-orig[i])/N` inside `area`, `0` outside. `plane` selects the
/// channel of the multi-plane `rec`/`original` pair (luma `0`; chroma `0`/`1`
/// for U/V).
fn psnr_vjp(
    d_psnr: f64,
    rec: &Tensor<f32>,
    original: &Tensor<f32>,
    area: Area,
    plane: usize,
) -> Tensor<f32> {
    let n = (area.width * area.height) as f64;
    let mse = rec
        .plane(plane)
        .iter()
        .zip(original.plane(plane))
        .enumerate()
        .filter(|(i, _)| {
            let (x, y) = (i % rec.w, i / rec.w);
            (area.x..area.x + area.width).contains(&x)
                && (area.y..area.y + area.height).contains(&y)
        })
        .map(|(_, (&a, &b))| ((a - b) as f64).powi(2))
        .sum::<f64>()
        / n;
    let mut out = Tensor::<f32>::zeros(1, rec.h, rec.w).unwrap();
    let k = d_psnr * (-10.0 / (core::f64::consts::LN_10 * (mse + 1e-10))) * (2.0 / n);
    for y in 0..rec.h {
        for x in 0..rec.w {
            if (area.x..area.x + area.width).contains(&x)
                && (area.y..area.y + area.height).contains(&y)
            {
                let i = y * rec.w + x;
                out.plane_mut(0)[i] =
                    (k * (rec.plane(plane)[i] - original.plane(plane)[i]) as f64) as f32;
            }
        }
    }
    out
}

/// `rdlr_y_iteration`'s backward: the full `d(y)` — the leaf gradient — for one
/// iteration's rate point and upstream scalars.
///
/// The chain (see the module docs): `d(resi)` from the GM likelihood + `d(rec)`
/// from the two metrics → `d(y_hat)` through the synthesis → `d(y)` through the
/// MCM's STE/context VJP — then the direct `resi = y - means_hat` `+1` term.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward(
    eng: &Engine,
    ctx: &mcm::ContextModel,
    synthesis: &SynthesisPrimary,
    up: &LossUpstream,
    rp: &RatePoint,
    y: &Tensor<f32>,
    _z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    area: Area,
    pixels: f64,
    orig_ms: Option<&msssim::MsOrig>,
    stop: &dyn enough::Stop,
) -> Result<Tensor<f32>> {
    let psi_b = BTensor::from_planar(&rp.psi, eng.tier.block())?;
    // d(resi) — the GM rate term.
    let d_resi = rate_vjp(up.d_rate / pixels, y, &rp.y_hat, &rp.residual_q, scale_log);
    // d(rec) — psnr + ms-ssim (ms-ssim on the [0,1]-scaled `area` crop, as
    // `distortion` builds it).
    let d_rec = rec_grad(eng, up, &rp.rec, original, area, orig_ms)?;
    // d(y_hat) — synthesis backward minus the resi term's -means_hat.
    let d_yhat_synth = synthesis.backward(
        eng,
        rp.synth_tape
            .as_ref()
            .ok_or(Error::InvalidArgument("RDLR: no synthesis tape"))?,
        &d_rec,
    )?;
    let mut d_out_yhat = d_yhat_synth.clone();
    for (a, &b) in d_out_yhat.data.iter_mut().zip(&d_resi.data) {
        *a -= b;
    }
    if cfg!(debug_assertions) {
        let am =
            |t: &Tensor<f32>| t.data.iter().map(|v| v.abs()).sum::<f32>() / t.data.len() as f32;
        eprintln!(
            "  d_resi {:.3e}  d_rec {:.3e}  d_yhat_synth {:.3e}  d_out_yhat {:.3e}",
            am(&d_resi),
            am(&d_rec),
            am(&d_yhat_synth),
            am(&d_out_yhat)
        );
    }
    // d(y) — the MCM's internal grad plus the resi term's direct +1.
    let mut d_y = ctx.compress_backward(
        eng,
        y,
        &psi_b,
        q,
        mask,
        num_chs,
        crate::encoder::SKIP_CUBE_THR,
        &d_out_yhat,
        &d_resi,
        Some(rp.tape.clone()),
        stop,
        None,
    )?;
    for (a, &b) in d_y.data.iter_mut().zip(&d_resi.data) {
        *a += b;
    }
    Ok(d_y)
}

/// `GMProbModel._likelihood` of a continuous residual with `means = 0`, as
/// `-log2` bits: the unit-interval Gaussian mass `Φ((0.5-|v|)/σ) - Φ((-0.5-|v|)/σ)`
/// written with `erfc`, floored at `likelihood_bound`.
fn gm_nbits(resi: f32, sigma: f32) -> f64 {
    // `get_std_cum_freq`'s `const = -(2**-0.5)`.
    const C: f64 = -std::f64::consts::FRAC_1_SQRT_2;
    let s = sigma as f64;
    let v = resi.abs() as f64;
    let upper = 0.5 * libm::erfc(C * (0.5 - v) / s);
    let lower = 0.5 * libm::erfc(C * (-0.5 - v) / s);
    let p = (upper - lower).max(LIKELIHOOD_BOUND);
    -p.log2()
}

/// `d(-log2 l)/d(resi)` at `(resi, sigma)` — the rate term's upstream VJP.
/// `l = 0.5·erfc(-0.7071·(0.5-|v|)/σ) - 0.5·erfc(-0.7071·(-0.5-|v|)/σ)`;
/// `d(erfc u)/du = -2·e^{-u²}/√π`; where `l` hit the `1e-9` floor,
/// `LowerBound`'s backward passes `1`, giving `d/d(l) = -1/(l·ln2)` with
/// `l = 1e-9` — the reference's own semantics (`LowerBoundFunction` returns
/// `max(x, bound)` and its backward zeroes nothing; autograd sees the clamped
/// value and passes the gradient through `x`'s gradient when `x > bound` —
/// below the bound the output is constant and the grad is 0). Torch's
/// `torch.clamp`-equivalent custom Function here is `LowerBound`: forward
/// `max`, backward grad where `x >= bound` else 0.
/// `d(-log2 LowerBound(l))/d(resi)` — the gradient of the *bounded* log
/// likelihood. The erfc factor `C² = 1/2` halves the squared terms:
/// `dl/dv = (1/(σ√(2π)))·(e^{−dl²/2} − e^{−du²/2})` where `dl=(0.5+v)/σ`,
/// `du=(0.5−v)/σ`. `LowerBound`'s backward passes the gradient where the raw
/// likelihood was under the floor ONLY when the upstream grad is negative
/// (`lower_bound_bwd`: `x >= bound | grad < 0` — the "unstick" direction);
/// `grad_output` there is `d_rate·(−1/(l_b ln2))`, negative iff `d_rate > 0`.
fn gm_nbits_vjp(resi: f32, sigma: f32, d_rate_positive: bool) -> f64 {
    const SQRT_2PI: f64 = 2.5066282746310002;
    let s = sigma as f64;
    let v = resi.abs() as f64;
    let du = (0.5 - v) / s;
    let dl = (0.5 + v) / s;
    // lik = Φ(du) − Φ(−dl) via erfc (the forward's exact form, for `p`).
    const C: f64 = -std::f64::consts::FRAC_1_SQRT_2;
    let upper = 0.5 * libm::erfc(C * du);
    let lower = 0.5 * libm::erfc(C * (-dl));
    let p = upper - lower;
    let floored = p < LIKELIHOOD_BOUND;
    if floored && !d_rate_positive {
        return 0.0;
    }
    let p_b = if floored { LIKELIHOOD_BOUND } else { p };
    let dl_dv = (libm::exp(-0.5 * dl * dl) - libm::exp(-0.5 * du * du)) / (s * SQRT_2PI);
    -(1.0 / (p_b * core::f64::consts::LN_2)) * dl_dv * resi.signum() as f64
}

/// The iteration's loss and its upstream scalars — `rdlr_y_iteration`'s
/// change vectors and slope-normal projections.
pub(crate) struct LossUpstream {
    /// `loss.item()` — the mixed loss for the best-iterate comparison.
    pub loss: f64,
    /// `dL/d(rate)` — `rate` = `test_bpp_Y`.
    pub d_rate: f64,
    /// `dL/d(psnr)`.
    pub d_psnr: f64,
    /// `dL/d(msssim)`.
    pub d_msssim: f64,
}

/// `rdlr_y_iteration`'s loss composition and the upstream scalar gradients —
/// the piece of the backward that doesn't touch the latent. `rp` is the
/// current rate point, `ref` the `get_ref_rate_data` point; the slopes are
/// `_get_slopes_parameters`' angles, `ratio` the PSNR/MS-SSIM mix.
pub(crate) fn loss_upstream(
    rp: &RatePoint,
    ref_rate: f64,
    ref_psnr: f64,
    ref_msssim: f64,
    slope_psnr: f64,
    slope_msssim: f64,
    ratio: f64,
) -> LossUpstream {
    let rate_log_diff = rp.bpp.ln() - ref_rate.ln();
    let psnr_diff = rp.psnr - ref_psnr;
    let msssim_diff = rp.msssim - ref_msssim;
    let (np_x, np_y) = (-slope_psnr.sin(), slope_psnr.cos());
    let (nm_x, nm_y) = (-slope_msssim.sin(), slope_msssim.cos());
    let loss_psnr = -(np_x * rate_log_diff + np_y * psnr_diff);
    let loss_msssim = -20.0 * (nm_x * rate_log_diff + nm_y * msssim_diff);
    let loss = loss_msssim * ratio + loss_psnr * (1.0 - ratio);
    // Upstream: dL/d(rate_log_diff), dL/d(psnr), dL/d(msssim).
    let d_rate_log = -(20.0 * ratio * nm_x + (1.0 - ratio) * np_x);
    LossUpstream {
        loss,
        d_rate: d_rate_log / rp.bpp,
        d_psnr: -(1.0 - ratio) * np_y,
        d_msssim: -(20.0 * ratio * nm_y),
    }
}

/// `process_y`'s luma driver: `num_ite` Adagrad + StepLR iterations on the
/// tile-core latent, tracking the lowest-loss iterate. The leaf is the whole
/// `y` here (single tile; a core crop for tiled RDLR — a later item). `refs`
/// is `get_ref_rate_data`'s un-optimised rate point (rate, psnr, msssim);
/// `lr0`/`lr_decay`/`ratio`/`slopes` come from `__calc_lr` /
/// `lossTypeBDcurveSlope` / `_get_slopes_parameters`.
///
/// Adagrad (`eps = 1e-10` like `torch.optim.Adagrad`): `sum_sq += g²`,
/// `leaf -= lr·g/(sqrt(sum_sq)+eps)`; StepLR `lr *= lr_decay` per iteration.
/// `best` is the lowest-loss iterate's `y` — `_compress_y` then re-derives its
/// residual/cube-flag decisions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_y(
    eng: &Engine,
    ctx: &mcm::ContextModel,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    synthesis: &SynthesisPrimary,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    area: Area,
    pixels: f64,
    num_ite: usize,
    lr0: f32,
    lr_decay: f32,
    refs: (f64, f64, f64),
    slope_psnr: f64,
    slope_msssim: f64,
    ratio: f64,
    optimize_area: Option<Area>,
    stop: &dyn enough::Stop,
) -> Result<(Tensor<f32>, f64)> {
    let mut leaf = y.clone();
    let mut sum_sq = vec![0.0f32; leaf.data.len()];
    let mut lr = lr0;
    let mut best = leaf.clone();
    let mut best_loss = f64::INFINITY;
    // `ms_ssim`'s `x = original` side is the same every iteration — compute it once.
    let orig_ms = orig_ms(eng, original, area)?;
    // `psi` is `z_hat`'s hyper-decoder output — `z_hat` is fixed, so iter 0's
    // `psi` serves all iterations (the hyper-decoder isn't re-run).
    let mut psi = None;
    // `Precision::Approximate` experiment knob: stop the search once the best iterate has
    // not improved for `EARLY_EXIT_ITERS` straight iterations — `Exact` always runs the
    // full `num_ite`. This is parity-breaking by construction (a different `best`), so it
    // stays behind the opt-in enum.
    let mut nochange = 0usize;
    for _ in 0..num_ite {
        let rp = rate_point_y(
            eng,
            hdr,
            model,
            synthesis,
            lh,
            &leaf,
            z_hat,
            scale_log,
            mask,
            q,
            num_chs,
            original,
            area,
            pixels,
            Some(&orig_ms),
            psi.as_ref(),
            stop,
        )?;
        psi.get_or_insert_with(|| rp.psi.clone());
        let up = loss_upstream(&rp, refs.0, refs.1, refs.2, slope_psnr, slope_msssim, ratio);
        let d_y = backward(
            eng,
            ctx,
            synthesis,
            &up,
            &rp,
            &leaf,
            z_hat,
            scale_log,
            mask,
            q,
            num_chs,
            original,
            area,
            pixels,
            Some(&orig_ms),
            stop,
        )?;
        // Upstream's leaf is `lat_tile_optimized_part` — only the overlap-free
        // core is stepped; the overlap context stays at its initial values.
        match optimize_area {
            Some(a) => {
                let stride = leaf.w * leaf.h;
                for (i, (&g, acc)) in d_y.data.iter().zip(&mut sum_sq).enumerate() {
                    let pos = i % stride;
                    let (x, y) = (pos % leaf.w, pos / leaf.w);
                    if x >= a.x && x < a.x + a.width && y >= a.y && y < a.y + a.height {
                        *acc += g * g;
                        leaf.data[i] -= lr * g / (acc.sqrt() + 1e-10);
                    }
                }
            }
            None => {
                for (i, (&g, acc)) in d_y.data.iter().zip(&mut sum_sq).enumerate() {
                    *acc += g * g;
                    leaf.data[i] -= lr * g / (acc.sqrt() + 1e-10);
                }
            }
        }
        lr *= lr_decay;
        if up.loss < best_loss {
            best.clone_from(&leaf);
            best_loss = up.loss;
            nochange = 0;
        } else {
            nochange += 1;
        }
        if eng.precision == crate::nn::fast::Precision::Approximate && nochange == EARLY_EXIT_ITERS
        {
            break;
        }
    }
    Ok((best, best_loss))
}

/// `cfg/tools/RDLR.json`'s `numIteYLuma` — the luma Adagrad iteration count.
/// (`numIteYChroma = 0` in the shipped config; chroma RDLR is a later item.)
pub(crate) const NUM_ITE_Y: usize = 8;

/// `optimize_*`'s early-exit plateau length — consecutive iterations without a
/// new best-loss before `Precision::Approximate` gives up (Exact never does).
const EARLY_EXIT_ITERS: usize = 4;

/// The chroma RDLR's upstream scalars — `dL` through the direction-projection
/// loss without touching the latent.
pub(crate) struct LossUpstreamUV {
    /// `loss.item()` — `(loss_psnrU + loss_psnrV) / 2`.
    pub loss: f64,
    /// `dL/d(rate_UV)` — `(sin sU + sin sV) / (2·joined)`, `joined =
    /// rate_UV + rate_ref_Y_part`.
    pub d_rate: f64,
    /// `dL/d(psnrU)` — `-cos sU / 2`.
    pub d_psnr_u: f64,
    /// `dL/d(psnrV)` — `-cos sV / 2`.
    pub d_psnr_v: f64,
}

/// `RDLRCompUV.rdlr_y_iteration`'s loss composition and upstream gradients.
/// `rp` is the current chroma rate point; `refs` is `get_ref_rate_data`'s
/// `(joint_rate_ref, psnrU_ref, psnrV_ref, rate_ref_Y_part)` where the joint
/// reference and the Y part each carry `calc_additional_rate`'s luma bpp.
#[allow(clippy::too_many_arguments)]
pub(crate) fn loss_upstream_uv(
    rp: &RatePointUV,
    joint_rate_ref: f64,
    psnr_u_ref: f64,
    psnr_v_ref: f64,
    rate_ref_y_part: f64,
    slope_u: f64,
    slope_v: f64,
) -> LossUpstreamUV {
    let joined = rp.bpp + rate_ref_y_part;
    let rate_log_diff = joined.ln() - joint_rate_ref.ln();
    let du = rp.psnr_u - psnr_u_ref;
    let dv = rp.psnr_v - psnr_v_ref;
    let loss_u = -(-slope_u.sin() * rate_log_diff + slope_u.cos() * du);
    let loss_v = -(-slope_v.sin() * rate_log_diff + slope_v.cos() * dv);
    LossUpstreamUV {
        loss: (loss_u + loss_v) / 2.0,
        d_rate: (slope_u.sin() + slope_v.sin()) / (2.0 * joined),
        d_psnr_u: -slope_u.cos() / 2.0,
        d_psnr_v: -slope_v.cos() / 2.0,
    }
}

/// `RDLRCompUV.rdlr_y_iteration`'s backward: `d(chroma y)` — the leaf gradient —
/// for one iteration's chroma rate point and upstream scalars. Mirrors
/// [`backward`]; the luma `sup` input is detached inside the secondary
/// synthesis's tape.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_uv(
    eng: &Engine,
    synthesis_uv: &SynthesisSecondary,
    up: &LossUpstreamUV,
    rp: &RatePointUV,
    y: &Tensor<f32>,
    _z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    area: Area,
    pixels: f64,
    _stop: &dyn enough::Stop,
) -> Result<Tensor<f32>> {
    let d_resi = rate_vjp(up.d_rate / pixels, y, &rp.y_hat, &rp.residual_q, scale_log);
    // `d(rec)`: the U and V PSNR gradients stacked into the two-plane map.
    let du = psnr_vjp(up.d_psnr_u, &rp.rec, original, area, 0);
    let dv = psnr_vjp(up.d_psnr_v, &rp.rec, original, area, 1);
    let mut d_rec = Tensor::<f32>::zeros(2, rp.rec.h, rp.rec.w)?;
    d_rec.plane_mut(0).copy_from_slice(&du.data);
    d_rec.plane_mut(1).copy_from_slice(&dv.data);
    // `d(y_hat)`: secondary synthesis backward minus the resi term's -means_hat.
    let d_yhat_synth = synthesis_uv.backward(
        eng,
        rp.synth_tape
            .as_ref()
            .ok_or(Error::InvalidArgument("RDLR: no secondary tape"))?,
        &d_rec,
    )?;
    let mut d_out_yhat = d_yhat_synth.clone();
    for (a, &b) in d_out_yhat.data.iter_mut().zip(&d_resi.data) {
        *a -= b;
    }
    // `d(y)`: the context-free STE-gated resi path plus the direct +1.
    let mut d_y = crate::encoder::compress_context_free_backward(
        y,
        mask,
        &rp.cube_flag,
        num_chs,
        q,
        &d_out_yhat,
        &d_resi,
    )?;
    for (a, &b) in d_y.data.iter_mut().zip(&d_resi.data) {
        *a += b;
    }
    Ok(d_y)
}

/// `process_y`'s chroma driver for `numIteYChroma > 0`: `num_ite` Adagrad +
/// StepLR iterations on the chroma latent, tracking the lowest-loss iterate.
/// `sup` is `sup_info` — the refined luma `y_hat` at chroma-latent size;
/// `refs` is `get_ref_rate_data` `(joint_rate, psnrU, psnrV, rate_ref_Y_part)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_uv(
    eng: &Engine,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    synthesis_uv: &SynthesisSecondary,
    lh: &lh::Likelihood,
    sup: &BTensor,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    area: Area,
    pixels: f64,
    num_ite: usize,
    lr0: f32,
    lr_decay: f32,
    refs: (f64, f64, f64, f64),
    slope_u: f64,
    slope_v: f64,
    optimize_area: Option<Area>,
    stop: &dyn enough::Stop,
) -> Result<(Tensor<f32>, f64)> {
    let mut leaf = y.clone();
    let mut sum_sq = vec![0.0f32; leaf.data.len()];
    let mut lr = lr0;
    let mut best = leaf.clone();
    let mut best_loss = f64::INFINITY;
    let mut psi = None;
    let mut nochange = 0usize;
    for _ in 0..num_ite {
        let rp = rate_point_uv(
            eng,
            hdr,
            model,
            synthesis_uv,
            lh,
            &leaf,
            z_hat,
            scale_log,
            mask,
            q,
            num_chs,
            original,
            sup,
            area,
            pixels,
            psi.as_ref(),
            stop,
        )?;
        psi.get_or_insert_with(|| rp.psi.clone());
        let up = loss_upstream_uv(&rp, refs.0, refs.1, refs.2, refs.3, slope_u, slope_v);
        let d_y = backward_uv(
            eng,
            synthesis_uv,
            &up,
            &rp,
            &leaf,
            z_hat,
            scale_log,
            mask,
            q,
            num_chs,
            original,
            area,
            pixels,
            stop,
        )?;
        // Same masked leaf as `optimize_y` — only `core_lat_rel` is stepped.
        match optimize_area {
            Some(a) => {
                let stride = leaf.w * leaf.h;
                for (i, (&g, acc)) in d_y.data.iter().zip(&mut sum_sq).enumerate() {
                    let pos = i % stride;
                    let (x, y) = (pos % leaf.w, pos / leaf.w);
                    if x >= a.x && x < a.x + a.width && y >= a.y && y < a.y + a.height {
                        *acc += g * g;
                        leaf.data[i] -= lr * g / (acc.sqrt() + 1e-10);
                    }
                }
            }
            None => {
                for (i, (&g, acc)) in d_y.data.iter().zip(&mut sum_sq).enumerate() {
                    *acc += g * g;
                    leaf.data[i] -= lr * g / (acc.sqrt() + 1e-10);
                }
            }
        }
        lr *= lr_decay;
        if up.loss < best_loss {
            best.clone_from(&leaf);
            best_loss = up.loss;
            nochange = 0;
        } else {
            nochange += 1;
        }
        if eng.precision == crate::nn::fast::Precision::Approximate && nochange == EARLY_EXIT_ITERS
        {
            break;
        }
    }
    Ok((best, best_loss))
}

/// `mean_psnrU_slopes` / `mean_psnrV_slopes` keyed by `int(target_bpp·100)`
/// (degrees → radians) — the chroma `fixed_slopes` counterpart.
fn fixed_slopes_uv(target_bpp_int: i32) -> (f64, f64) {
    let deg = std::f64::consts::PI / 180.0;
    let u = match target_bpp_int {
        6 => 64.131075,
        12 => 62.749463,
        25 => 63.404102,
        50 => 58.369855,
        75 => 44.556739,
        _ => 58.642247,
    };
    let v = match target_bpp_int {
        6 => 64.006979,
        12 => 62.568593,
        25 => 62.318648,
        50 => 56.189655,
        75 => 42.682759,
        _ => 57.553327,
    };
    (u * deg, v * deg)
}

/// `process_y`'s whole-picture chroma path for `EncodeParams::num_ite_uv > 0`:
/// like [`luma`], but `RDLRCompUV` — `original` is the source chroma
/// `[2, ch, cw]` at 0..255, `sup` the refined luma `y_hat`, `y_rate_part` the
/// luma rate point's bpp (`calc_additional_rate`/`get_luma_tile_rate`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma(
    eng: &Engine,
    models: &dyn crate::model::ModelSource,
    model_id: u8,
    synth_op: crate::header::OperatingPoint,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    sup: &BTensor,
    latent: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    y_rate_parts: &[f64],
    pixels: f64,
    num_ite: usize,
    stop: &dyn enough::Stop,
) -> Result<Tensor<f32>> {
    let synthesis_uv =
        crate::model::load_synthesis_secondary(models, model_id as usize, synth_op, eng)?;
    // `numSamplesPerChromaTile` picture tiles — `get_luma_tile_rate` adds each
    // UV tile's co-located luma-tile rate (`y_rate_parts[i]`). The chroma
    // component's latent factor is 8 (its "picture" is the half-res 4:2:0
    // view); the UV loss is per-plane PSNR only — no MS-SSIM floor.
    let tiles = crate::tools::tiles::rdlr_tiles(
        original.h,
        original.w,
        latent.h,
        latent.w,
        z_hat.h,
        z_hat.w,
        NUM_SAMPLES_PER_CHROMA_TILE,
        RDLR_TILE_OVERLAP,
        CHROMA_LATENT_FACTOR,
        None,
    )?;
    let (su, sv) = fixed_slopes_uv(target_bpp_centi(model_id));
    let lr0 = calc_lr(pixels);
    let sup_p = sup.to_planar()?;
    let mut best = latent.clone();
    for (i, t) in tiles.iter().enumerate() {
        let lat_t = latent.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        let im_t = original.window(t.image.x, t.image.y, t.image.width, t.image.height)?;
        let z_t = z_hat.window(t.z.x, t.z.y, t.z.width, t.z.height)?;
        let sl_t = scale_log.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        let mask_t = mask.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        // `lat_suppinfo = get_data(sup_info, latent_tile)` — sup is already
        // `::2` decimated to chroma-latent res.
        let sup_t = BTensor::from_planar(
            &sup_p.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?,
            eng.tier.block(),
        )?;
        // `ref_data[0]` (joint) and `ref_data[3]` (Y part) each get the luma
        // tile's rate — `additional_rate`. `y_rate_parts[i]` is the luma tile's
        // bpp; for a single tile it is the whole-picture `y_rate_part`.
        let yrp = y_rate_parts.get(i).copied().unwrap_or(0.0);
        let core_px = (t.core_img_rel.width * t.core_img_rel.height) as f64;
        let r = rate_point_uv(
            eng,
            hdr,
            model,
            &synthesis_uv,
            lh,
            &lat_t,
            &z_t,
            &sl_t,
            &mask_t,
            q,
            num_chs,
            &im_t,
            &sup_t,
            t.core_img_rel,
            core_px,
            None,
            stop,
        )?;
        let (best_t, _loss) = optimize_uv(
            eng,
            hdr,
            model,
            &synthesis_uv,
            lh,
            &sup_t,
            &lat_t,
            &z_t,
            &sl_t,
            &mask_t,
            q,
            num_chs,
            &im_t,
            t.core_img_rel,
            core_px,
            num_ite,
            lr0,
            0.7,
            (r.bpp + yrp, r.psnr_u, r.psnr_v, yrp),
            su,
            sv,
            Some(t.core_lat_rel),
            stop,
        )?;
        assign_area(&mut best, &t.core_lat, &best_t, &t.core_lat_rel)?;
    }
    Ok(best)
}

/// `numSamplesPerLumaTile` — `setup_tiles_enc`'s luma tile threshold
/// (`RdlrCompParams` default: image samples per tile → 1024-tile).
pub(crate) const NUM_SAMPLES_PER_LUMA_TILE: usize = 1 << 20;
/// `numSamplesPerChromaTile` — the chroma threshold (→ 720-tile).
pub(crate) const NUM_SAMPLES_PER_CHROMA_TILE: usize = 500_000;
/// `numSamplesTileOverlap{Y,UV}` — tile overlap in image samples.
pub(crate) const RDLR_TILE_OVERLAP: usize = 64;
/// `tool.get_alignment_size()` — the CCS tile alignment.
pub(crate) const RDLR_ALIGN: usize = 16;
/// The chroma component's `latent_downscale_factor` — its "picture" is the
/// half-res 4:2:0 view, so `chroma_image / 8` lands on the same picture/16
/// latent resolution as luma's `/16`. Its `z` factor is `CHROMA_LATENT_FACTOR * 4`.
pub(crate) const CHROMA_LATENT_FACTOR: usize = 8;
/// `tile_min_size_for_MSSSIM` — MS-SSIM needs tiles larger than 160; the floor
/// is `tile_min_size + overlap` (240) when the MS-SSIM loss is mixed in.
pub(crate) const TILE_MIN_FOR_MSSSIM: usize = 176;

/// `tiling.assign_data` — `dst[dst_area] = src[src_area]` per channel plane.
fn assign_area(
    dst: &mut Tensor<f32>,
    dst_area: &Area,
    src: &Tensor<f32>,
    src_area: &Area,
) -> Result<()> {
    if src_area.width != dst_area.width || src_area.height != dst_area.height {
        return Err(Error::InvalidArgument("assign_area shape mismatch"));
    }
    let (sd, ss) = (dst.w * dst.h, src.w * src.h);
    for c in 0..dst.data.len() / sd {
        for y in 0..dst_area.height {
            for x in 0..dst_area.width {
                dst.data[c * sd + (dst_area.y + y) * dst.w + dst_area.x + x] =
                    src.data[c * ss + (src_area.y + y) * src.w + src_area.x + x];
            }
        }
    }
    Ok(())
}

/// `__calc_lr` — the per-resolution learning rate (`LearningRateYAutomaticPerResolution`).
fn calc_lr(num_samples: f64) -> f32 {
    if num_samples >= (3840.0 * 2160.0) {
        0.02
    } else if num_samples >= (832.0 * 480.0) {
        0.03
    } else {
        0.05
    }
}

/// The operating points' `target_bpps` in centi-bpp (the `-target_bpps` CLI
/// values — `target_bpps[bpp_idx]` indexes `mean_*_slopes`). `get_target_bpp`
/// is `target_bpps[idx]·0.01`; `get_target_bpp_int` = `int(target_bpp·100)`
/// recovers the centi-bpp key. Beta [0.002, 0.012, 0.075, 0.5] → [6, 12, 25, 50].
fn target_bpp_centi(model_id: u8) -> i32 {
    const TARGET_BPPS: [i32; 4] = [6, 12, 25, 50];
    TARGET_BPPS.get(model_id as usize).copied().unwrap_or(-1)
}

/// `_get_slopes_parameters` `single_seq_bpp_slopes` fallback: the fixed
/// `mean_{psnrY,msssimY}_slopes` keyed by `int(target_bpp·100)` (degrees →
/// radians). The dumped vector's slopes (76.474952° / 1.200354°) match bucket
/// 12 — beta 0.012's `-target_bpps 12`. Non-listed targets take `'other'`.
fn fixed_slopes(target_bpp_int: i32) -> (f64, f64) {
    let deg = std::f64::consts::PI / 180.0;
    let psnr = match target_bpp_int {
        6 => 74.868811,
        12 => 76.474952,
        25 => 76.252456,
        50 => 69.157414,
        75 => 57.958624,
        _ => 70.942451,
    };
    let msssim = match target_bpp_int {
        6 => 1.629650,
        12 => 1.200354,
        25 => 0.785844,
        50 => 0.372070,
        75 => 0.175778,
        _ => 0.832739,
    };
    (psnr * deg, msssim * deg)
}

/// `process_y`'s whole-picture luma path for `EncodeParams::rdlr`: the source
/// luma `original` `[1, h, w]` at 0..255, the same `scales`/`mask`/`q` the
/// residual pass uses, `num_ite` Adagrad iterations. Returns the refined latent —
/// `compress_regions` then re-derives its residual/cube-flag decisions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma(
    eng: &Engine,
    models: &dyn crate::model::ModelSource,
    model_id: u8,
    synth_op: crate::header::OperatingPoint,
    hdr: &crate::header::PictureHeader,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    latent: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    scale_log: &Tensor<i32>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    original: &Tensor<f32>,
    pixels: f64,
    num_ite: usize,
    stop: &dyn enough::Stop,
) -> Result<Tensor<f32>> {
    let ctx = model
        .context
        .as_ref()
        .ok_or_else(|| Error::Model("RDLR: no context model".into()))?;
    let synthesis = crate::model::load_synthesis_primary(models, model_id as usize, synth_op, eng)?;
    // `process_y`: `setup_tiles_enc` over `numSamplesPerLumaTile` picture tiles
    // (single whole-picture tile when the picture fits). `minimum_tile_size` is
    // set whenever `ratio != 0` — our `loss_upstream` always mixes MS-SSIM
    // (`loss = msssim·0.8 + psnr·0.2`).
    let tiles = crate::tools::tiles::rdlr_tiles(
        original.h,
        original.w,
        latent.h,
        latent.w,
        z_hat.h,
        z_hat.w,
        NUM_SAMPLES_PER_LUMA_TILE,
        RDLR_TILE_OVERLAP,
        RDLR_ALIGN,
        Some(TILE_MIN_FOR_MSSSIM + RDLR_TILE_OVERLAP),
    )?;
    let (sp, sm) = fixed_slopes(target_bpp_centi(model_id));
    let lr0 = calc_lr(pixels);
    let mut best = latent.clone();
    for t in &tiles {
        let lat_t = latent.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        let im_t = original.window(t.image.x, t.image.y, t.image.width, t.image.height)?;
        let z_t = z_hat.window(t.z.x, t.z.y, t.z.width, t.z.height)?;
        let sl_t = scale_log.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        let mask_t = mask.window(t.latent.x, t.latent.y, t.latent.width, t.latent.height)?;
        // `get_ref_rate_data` — the un-optimised rate point of this tile.
        let core_px = (t.core_img_rel.width * t.core_img_rel.height) as f64;
        let r = rate_point_y(
            eng,
            hdr,
            model,
            &synthesis,
            lh,
            &lat_t,
            &z_t,
            &sl_t,
            &mask_t,
            q,
            num_chs,
            &im_t,
            t.core_img_rel,
            core_px,
            None,
            None,
            stop,
        )?;
        let (best_t, _loss) = optimize_y(
            eng,
            ctx,
            hdr,
            model,
            &synthesis,
            lh,
            &lat_t,
            &z_t,
            &sl_t,
            &mask_t,
            q,
            num_chs,
            &im_t,
            t.core_img_rel,
            core_px,
            num_ite,
            lr0,
            0.7,
            (r.bpp, r.psnr, r.msssim),
            sp,
            sm,
            0.8,
            Some(t.core_lat_rel),
            stop,
        )?;
        // `assign_data(best_decisions['y'], optimized_part_of_lattile, best_y_tile)`.
        assign_area(&mut best, &t.core_lat, &best_t, &t.core_lat_rel)?;
    }
    Ok(best)
}

/// `d(rec)` on the whole reconstruction plane — `ms_ssim_vjp` (on the
/// `[0,1]`-scaled `area` crop, scattered back to the picture grid) plus
/// `psnr_vjp`. This is `x_hat_grad`'s analogue.
pub(crate) fn rec_grad(
    eng: &Engine,
    up: &LossUpstream,
    rec: &Tensor<f32>,
    original: &Tensor<f32>,
    area: Area,
    orig_ms: Option<&msssim::MsOrig>,
) -> Result<Tensor<f32>> {
    let (dx, _dy) = {
        const INV: f32 = 1.0 / 255.0;
        let (ah, aw) = (area.height, area.width);
        let (mut a, mut b) = (Vec::with_capacity(ah * aw), Vec::with_capacity(ah * aw));
        for row in 0..ah {
            for x in 0..aw {
                a.push(original.plane(0)[(area.y + row) * original.w + area.x + x] * INV);
                b.push(rec.plane(0)[(area.y + row) * rec.w + area.x + x] * INV);
            }
        }
        // `ms_ssim` runs on 1/255-scaled inputs, so `dx` is d/d(rec/255) — the
        // rec-domain gradient is smaller by a further 1/255: fold that into the
        // upstream scalar. `y = original` is fixed across iterations — its side
        // is in `orig_ms` (`x = rec` is `ms_ssim_vjp`'s first arg here).
        match orig_ms {
            Some(orig) => msssim::ms_ssim_vjp_with_orig(
                eng,
                orig,
                &Tensor::from_vec(1, ah, aw, b)?,
                up.d_msssim as f32 * INV,
            )?,
            None => msssim::ms_ssim_vjp(
                eng,
                &Tensor::from_vec(1, ah, aw, b)?,
                &Tensor::from_vec(1, ah, aw, a)?,
                up.d_msssim as f32 * INV,
            )?,
        }
    };
    let mut d_rec = Tensor::<f32>::zeros(1, rec.h, rec.w)?;
    for row in 0..area.height {
        for x in 0..area.width {
            d_rec.plane_mut(0)[(area.y + row) * rec.w + area.x + x] = dx.data[row * area.width + x];
        }
    }
    let d_psnr_map = psnr_vjp(up.d_psnr, rec, original, area, 0);
    for (a, &b) in d_rec.data.iter_mut().zip(&d_psnr_map.data) {
        *a += b;
    }
    Ok(d_rec)
}

/// `d(rate_y)/d(resi)` at every position — `dL/d(resi)` with the upstream
/// `dL/d(rate_y)` folded in. `rate_y = Σ -log2 l`; `dL/d(rate_y) = d_rate /
/// pixels`.
pub(crate) fn rate_vjp(
    d_rate_y: f64,
    y: &Tensor<f32>,
    y_hat: &Tensor<f32>,
    residual_q: &Tensor<i16>,
    scale_log: &Tensor<i32>,
) -> Tensor<f32> {
    let log_k = log_k() as f32;
    let inv = 1.0f32 / (1u32 << SIGMA_PRECISION) as f32;
    let mut out = Tensor::<f32>::zeros(residual_q.c, residual_q.h, residual_q.w).unwrap();
    for ch in 0..residual_q.c {
        let (r, s, yq, yh) = (
            residual_q.plane(ch),
            scale_log.plane(ch),
            y.plane(ch),
            y_hat.plane(ch),
        );
        for i in 0..r.len() {
            let resi = yq[i] - yh[i] + r[i] as f32;
            let scale_lin = math::exp(s[i] as f32 * inv * log_k + LN_SIGMA_MIN);
            // The same sigma the forward uses — `_index_to_scale`'s float
            // `>>` (`(scale_lin + 2^6) / 2^7`, fraction preserved, then
            // `exp(idx·log_k + ln σ_min)·2^7`) and `_likelihood`'s `/2^7`
            // — reproduces the dumped `rate_y` to 0.08 bits.
            let idx = (scale_lin + 64.0)
                .mul_add(1.0 / 128.0, 0.0)
                .clamp(0.0, 34.0);
            let sigma = math::exp(idx * log_k + LN_SIGMA_MIN).max(SCALE_BOUND);
            out.plane_mut(ch)[i] = (d_rate_y * gm_nbits_vjp(resi, sigma, d_rate_y > 0.0)) as f32;
        }
    }
    out
}

// --- VJP (input-gradient) primitives ---------------------------------------
//
// Every backward is the input gradient only (model weights are frozen — no
// dL/dw anywhere in RDLR). Determinism rule: f32 accumulation in fixed order,
// matching the forward path's op-order/FMA policy; `math::` kernels reused
// where a forward op already has a fixed implementation.

/// `distortion`'s scaled `x = original` crop as an [`MsOrig`] — `ms_ssim`'s
/// `x` side is fixed across `optimize_y`'s iterations, so it is precomputed once.
fn orig_ms(eng: &Engine, original: &Tensor<f32>, area: Area) -> Result<msssim::MsOrig> {
    const INV: f32 = 1.0 / 255.0;
    let (ah, aw) = (area.height, area.width);
    let mut a = Vec::with_capacity(ah * aw);
    for row in 0..ah {
        for x in 0..aw {
            a.push(original.plane(0)[(area.y + row) * original.w + area.x + x] * INV);
        }
    }
    msssim::ms_ssim_precompute_orig(eng, &Tensor::from_vec(1, ah, aw, a)?)
}

/// `calc_distortion_loss` for the luma tile: `mean((a-b)²)` over `area` (+1e-10),
/// `ms_ssim(a, b, data_range = 255)`.
fn distortion(
    eng: &Engine,
    rec: &Tensor<f32>,
    original: &Tensor<f32>,
    area: Area,
    orig_ms: Option<&msssim::MsOrig>,
) -> Result<(f64, f64)> {
    // MS-SSIM's `data_range = 255` is `data_range = 1` on inputs scaled by 1/255
    // (the C1/C2 constants are the only place the range enters).
    const INV: f32 = 1.0 / 255.0;
    let (ah, aw) = (area.height, area.width);
    let (mut a, mut b) = (Vec::new(), Vec::with_capacity(ah * aw));
    let mut mse = 0.0f64;
    for row in 0..ah {
        for x in 0..aw {
            let o = original.plane(0)[(area.y + row) * original.w + area.x + x];
            let r = rec.plane(0)[(area.y + row) * rec.w + area.x + x];
            if orig_ms.is_none() {
                a.push(o * INV);
            }
            b.push(r * INV);
            mse += ((o - r) as f64) * ((o - r) as f64);
        }
    }
    mse /= (ah * aw) as f64;
    let psnr = 20.0 * (255.0f64).log10() - 10.0 * (mse + 1e-10).log10();
    let tb = Tensor::from_vec(1, ah, aw, b)?;
    let msssim = match orig_ms {
        // `x = original` is fixed across the optimiser's iterations — its side
        // of the cascade was computed once.
        Some(orig) => msssim::ms_ssim_with_orig(eng, orig, &tb)? as f64,
        None => msssim::ms_ssim(eng, &Tensor::from_vec(1, ah, aw, a)?, &tb)? as f64,
    };
    Ok((psnr, msssim))
}

// `Likelihood`'s `z_bits` accessor lives in `lh.rs` (it reads the private
// `z_nbits` table directly).

#[cfg(all(test, feature = "reference-tests"))]
mod oracle {
    //! Replay `dump_rdlr.py`'s captures on `rdlr_img30_bop_m1_b0`: every
    //! iteration's `rate_point` must land on the dumped `(rate, psnr, msssim)`,
    //! the latent/scales intermediates are asserted exactly, and `y_hat`
    //! replays the merged MCM output.

    use super::*;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use crate::decoder::entropy::{GrfsFlags, component_scales_with};
    use crate::encoder::colour::SourceMeta;
    use crate::encoder::{EncodeParams, EncoderQuantiser, picture_header};
    use crate::header::ColourTransform;
    use crate::header::OperatingPoint;
    use crate::tools::skip::skip_mask;

    struct Dump {
        dtype: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn load_dump(dir: &Path) -> HashMap<String, Dump> {
        let read = |name: &str| -> Vec<u8> {
            std::fs::read(dir.join(name)).unwrap_or_else(|e| {
                panic!(
                    "{}: {e} — run scripts/ref_vectors/dump_rdlr.py on the reference checkout",
                    dir.join(name).display()
                )
            })
        };
        let manifest = String::from_utf8(read("rdlr_manifest.txt")).unwrap();
        let blob = read("rdlr_tensors.bin");
        let mut out = HashMap::new();
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            // Tensor names can contain spaces (e.g. decisions keyed by an `Area`
            // repr): anchor on the dtype token — `name* dtype ndim dims[ndim]
            // offset nbytes`.
            const DTYPES: [&str; 9] =
                ["f32", "f64", "i8", "u8", "i16", "i32", "i64", "u32", "bool"];
            let n = f.len();
            let i = (0..n)
                .find(|&i| {
                    DTYPES.contains(&f[i])
                        && f[i + 1].parse::<usize>().is_ok_and(|d| i + 4 + d == n)
                        && f[i + 2..n - 2].iter().all(|t| t.parse::<usize>().is_ok())
                })
                .unwrap_or_else(|| panic!("unparsable manifest line: {line}"));
            let ndim: usize = f[i + 1].parse().unwrap();
            let shape: Vec<usize> = f[i + 2..i + 2 + ndim]
                .iter()
                .map(|d| d.parse().unwrap())
                .collect();
            let offset: usize = f[i + 2 + ndim].parse().unwrap();
            let nbytes: usize = f[i + 3 + ndim].parse().unwrap();
            out.insert(
                f[..i].join(" "),
                Dump {
                    dtype: f[i].to_string(),
                    shape,
                    bytes: blob[offset..offset + nbytes].to_vec(),
                },
            );
        }
        out
    }

    impl Dump {
        fn f32(&self) -> Vec<f32> {
            assert_eq!(self.dtype, "f32");
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        fn f32s(&self) -> f32 {
            let v = self.f32();
            assert_eq!(v.len(), 1, "scalar dump");
            v[0]
        }
        fn tensor_f32(&self) -> Tensor<f32> {
            assert_eq!(self.shape[0], 1, "batch dim");
            Tensor::from_vec(self.shape[1], self.shape[2], self.shape[3], self.f32()).unwrap()
        }
        fn tensor_i8(&self) -> Tensor<i8> {
            assert_eq!(self.dtype, "f32");
            Tensor::from_vec(
                self.shape[1],
                self.shape[2],
                self.shape[3],
                self.f32().into_iter().map(|v| v as i8).collect(),
            )
            .unwrap()
        }
        /// `resi_hat` indices exceed the `i8` symbol range on HOP (±244) — read
        /// them as `i16` so the comparison does not saturate at ±127.
        fn tensor_i16(&self) -> Tensor<i16> {
            assert_eq!(self.dtype, "f32");
            Tensor::from_vec(
                self.shape[1],
                self.shape[2],
                self.shape[3],
                self.f32().into_iter().map(|v| v as i16).collect(),
            )
            .unwrap()
        }
    }

    fn vectors_root() -> PathBuf {
        std::env::var_os("ZENJPEGAI_VECTORS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"))
    }

    /// The dumped encode's whole context is rebuilt from the dumped `z_hat`
    /// (`component_scales_with` output is asserted exactly), then every dumped
    /// `rdlr.i.y` is run through `rate_point_y` and compared to the
    /// `gcrp.(i+1)` rate point the reference measured on it.
    #[test]
    fn rate_point_matches_reference_iterations() {
        check_vector("rdlr_img30_bop_m1_b0", 1, OperatingPoint::Bop);
    }

    /// The same replay on the high profile's dump (`rdlr_img30_hop_m2_b0`):
    /// the HOP `Primary` tape/backward must also land on the reference's
    /// per-iteration rate points and gradients.
    #[test]
    fn rate_point_matches_reference_iterations_hop() {
        check_vector("rdlr_img30_hop_m2_b0", 2, OperatingPoint::Hop);
    }

    fn check_vector(dir_name: &str, model_id: usize, op: OperatingPoint) {
        let dump = load_dump(&vectors_root().join(dir_name));
        let models = std::env::var_os("ZENJPEGAI_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/lilith/work/zen/jpeg-ai-reference-software/models")
            });
        let enc = crate::encoder::Encoder::new(models);
        let eng = &enc.engine;
        let params = EncodeParams {
            model_id: model_id as u8,
            op,
            ..EncodeParams::default()
        };
        let set = enc.model_set(model_id, op).unwrap();
        let meta = SourceMeta {
            bit_depth: 8,
            s_ver: 1,
            s_hor: 1,
            c_ver: 2,
            c_hor: 2,
            colour_transform: ColourTransform::Bt709,
        };
        let hdr = picture_header(
            560,
            888,
            params.model_id,
            params.beta_displacement_log,
            params,
            &meta,
        );

        let z_hat = dump["rdlr.decisions.z_hat"].tensor_i8();
        let scales = component_scales_with(
            &hdr,
            0,
            &set.common[0],
            &z_hat,
            None,
            params.rvs,
            GrfsFlags::Derive(params.grfs),
        )
        .unwrap();
        // The derived integer scales equal the reference's decisions exactly.
        assert_eq!(
            scales.scale_log.data,
            dump["rdlr.decisions.scale_log"]
                .f32()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            scales.skip_scale_log.data,
            dump["rdlr.decisions.skip_scale_log"]
                .f32()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        );

        let mask = skip_mask(&scales.skip_scale_log, None).unwrap();
        let q = EncoderQuantiser {
            scales: &scales,
            quality_map: None,
        };
        let synthesis =
            crate::model::load_synthesis_primary(enc.models.as_ref(), model_id, op, eng).unwrap();
        let lh = &set.lh[0];
        let original = dump["rdlr.0.original"].tensor_f32();
        let pixels = 888.0 * 560.0;
        let area = Area {
            x: 0,
            y: 0,
            width: 560,
            height: 888,
        };

        for i in 0..8usize {
            let y = dump[&format!("rdlr.{i}.y")].tensor_f32();
            let rp = rate_point_y(
                eng,
                &hdr,
                &set.common[0],
                &synthesis,
                lh,
                &y,
                &z_hat,
                &scales.scale_log,
                &mask,
                &q,
                160,
                &original,
                area,
                pixels,
                None,
                None,
                &enough::Unstoppable,
            )
            .unwrap();
            // `_compress_ar_scale`'s outputs replayed exactly (`cas.0/1` are
            // the main encode, `cas.2` the ref point, `cas.3+i` iteration i).
            let c = i + 3;
            // `resi_hat` is an index into the scale table — HOP's wider
            // residuals exceed `i8`, so read it as `i16` (`tensor_i8` would
            // saturate at ±127 and hide nothing-real diffs).
            let want_rq = dump[&format!("rdlr.cas.{c}.resi_hat")].tensor_i16();
            let got: Vec<i16> = want_rq.data;
            let ndiff = rp
                .residual_q
                .data
                .iter()
                .zip(&got)
                .filter(|&(&a, &b)| a != b)
                .count();
            let max_step = rp
                .residual_q
                .data
                .iter()
                .zip(&got)
                .map(|(&a, &b)| (a - b).abs())
                .max()
                .unwrap_or(0);
            // Quantise-boundary flips only: the context-mean nets are f32-exact
            // so a `data_q` landing within an ULP of `.5` can round the other
            // way — but only to the adjacent index, and only on a handful of
            // positions out of 313,600.
            assert!(max_step <= 1, "iter {i}: resi_hat step {max_step}");
            assert!(
                ndiff * 1000 <= rp.residual_q.data.len(),
                "iter {i}: resi_hat {ndiff} diffs of {}",
                rp.residual_q.data.len()
            );
            let want_yh = dump[&format!("rdlr.cas.{c}.y_hat")].tensor_f32();
            // `y_hat = resi_dq + mu`: where `resi_q` flipped an index, `y_hat`
            // moves by one dequant step (~0.6) — the flip's own signature.
            // Elsewhere the reconstruction is f32-close.
            let mut unflipped = 0.0f32;
            for k in 0..rp.y_hat.data.len() {
                if got[k] != rp.residual_q.data[k] {
                    continue;
                }
                unflipped = unflipped.max((rp.y_hat.data[k] - want_yh.data[k]).abs());
            }
            // The context is autoregressive on the reconstructed residual: a
            // `resi_q` index flip upstream propagates ~one dequant step (~0.6)
            // through the context conv into *downstream* means, so an off-flip
            // position can legitimately reach ~that magnitude (measured 0.25,
            // one position after a flip). Bound at one full dequant step; a
            // real `mu` defect exceeds it.
            assert!(unflipped < 1.0, "iter {i}: y_hat off-flip max {unflipped}");
            // The GM's actual inputs: `resi = y - means_hat`, `scale = scale_lin`.
            let gm_resi = dump[&format!("rdlr.gm.{}.resi", i + 1)].tensor_f32();
            let gm_scale = dump[&format!("rdlr.gm.{}.scale", i + 1)].tensor_f32();
            let (mut md, mut ms) = (0.0f32, 0.0f32);
            for ch in 0..160 {
                for k in 0..gm_resi.plane(ch).len() {
                    // resi carries `resi_q` in index units — a ±1 flip moves it
                    // by a whole dequant step; compare off-flip only.
                    if rp.residual_q.plane(ch)[k] != got[ch * 1960 + k] {
                        continue;
                    }
                    let resi =
                        y.plane(ch)[k] - rp.y_hat.plane(ch)[k] + rp.residual_q.plane(ch)[k] as f32;
                    md = md.max((resi - gm_resi.plane(ch)[k]).abs());
                    let sl = scales.scale_log.plane(ch)[k];
                    let lin = math::exp(sl as f32 * (1.0 / 128.0) * log_k() as f32 + LN_SIGMA_MIN);
                    ms = ms.max((lin - gm_scale.plane(ch)[k]).abs() / gm_scale.plane(ch)[k]);
                }
            }
            // `scale_lin` reaches ~82 on HOP — compare the exp-precision (≲1 ULP)
            // *relative* diff, not an absolute bound calibrated to BOP's range.
            assert!(ms < 1e-6, "iter {i}: gm scale rel diff {ms}");
            // `resi` inherits `y_hat`'s propagated diff — bound at one dequant
            // step like `unflipped`.
            assert!(md < 1.0, "iter {i}: gm resi diff {md}");
            // The rate term's upstream: `dL/d(resi)` = `dL/d(rate)·d(rate)/d(rate_y)·
            // d(rate_y)/d(resi)` — `gm_resi_grad` is exactly that.
            let ref_rate = dump["rdlr.gcrp.0.rate"].f32s() as f64;
            let ref_psnr = dump["rdlr.gcrp.0.psnr"].f32s() as f64;
            let ref_msssim = dump["rdlr.gcrp.0.msssim"].f32s() as f64;
            let slope_psnr = dump["rdlr.slope.psnrY"].f32s() as f64;
            let slope_msssim = dump["rdlr.slope.msssim_Torch"].f32s() as f64;
            let ratio = dump["rdlr.ratio_psnr_msssim"].f32s() as f64;
            let up = loss_upstream(
                &rp,
                ref_rate,
                ref_psnr,
                ref_msssim,
                slope_psnr,
                slope_msssim,
                ratio,
            );
            let want_loss = dump[&format!("rdlr.{i}.loss")].f32s() as f64;
            assert!(
                (up.loss - want_loss).abs() < 1e-4,
                "iter {i}: loss {} vs reference {want_loss}",
                up.loss
            );
            let d_rate_y = up.d_rate / pixels;
            let d_resi = rate_vjp(d_rate_y, &y, &rp.y_hat, &rp.residual_q, &scales.scale_log);
            let want_dresi = dump[&format!("rdlr.{i}.gm_resi_grad")].tensor_f32();
            let (mut md, mut am, mut c_xy, mut c_yy) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
            for k in 0..d_resi.data.len() {
                let (a, b) = (d_resi.data[k] as f64, want_dresi.data[k] as f64);
                md = md.max((a - b) as f32);
                am += (a - b).abs();
                c_xy += a * b;
                c_yy += b * b;
            }
            eprintln!(
                "  d_resi vs gm_resi_grad: max {md:.3e} mean {:.3e} corr {:.4} scale {:.3}",
                am / d_resi.data.len() as f64,
                c_xy / (c_xy.abs().sqrt() * c_yy.sqrt()),
                c_xy / c_yy
            );
            // The full backward: `d(y)` vs `cas_y_in_grad`, `d(leaf)` vs `y_grad`.
            let ctx = set.common[0].context.as_ref().unwrap();
            let d_y = backward(
                eng,
                ctx,
                &synthesis,
                &up,
                &rp,
                &y,
                &z_hat,
                &scales.scale_log,
                &mask,
                &q,
                160,
                &original,
                area,
                pixels,
                None,
                &enough::Unstoppable,
            )
            .unwrap();
            // Isolate the MCM backward: feed the reference's OWN y_hat/resi_hat
            // gradients (exact upstreams) — if d_y then matches cas_y_in_grad,
            // the gap is in the synthesis backward, not the MCM.
            if i == 0 {
                let psi_b = BTensor::from_planar(&rp.psi, eng.tier.block()).unwrap();
                let want_uh = dump[&format!("rdlr.{i}.y_hat_grad")].tensor_f32();
                let want_ur = dump[&format!("rdlr.{i}.gm_resi_grad")].tensor_f32();
                let ctx = set.common[0].context.as_ref().unwrap();
                let d_y2 = ctx
                    .compress_backward(
                        eng,
                        &y,
                        &psi_b,
                        &q,
                        &mask,
                        160,
                        crate::encoder::SKIP_CUBE_THR,
                        &want_uh,
                        &want_ur,
                        None,
                        &enough::Unstoppable,
                        None,
                    )
                    .unwrap();
                // `compress_backward` alone must land on `cas_y_in_grad`; with
                // the resi term's direct +1 added it must land on `y_grad`.
                let want = dump[&format!("rdlr.{i}.cas_y_in_grad")].tensor_f32();
                let want_l = dump[&format!("rdlr.{i}.y_grad")].tensor_f32();
                let (mut am0, mut s_xy0, mut s_yy0) = (0.0f64, 0.0f64, 0.0f64);
                for k in 0..d_y2.data.len() {
                    am0 += (d_y2.data[k] - want.data[k]).abs() as f64;
                    s_xy0 += d_y2.data[k] as f64 * want.data[k] as f64;
                    s_yy0 += (want.data[k] as f64).powi(2);
                }
                let mut d_y2 = d_y2;
                for (a, &b) in d_y2.data.iter_mut().zip(&want_ur.data) {
                    *a += b;
                }
                let (mut md, mut am, mut s_xy, mut s_yy) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
                for k in 0..d_y2.data.len() {
                    let (a, b) = (d_y2.data[k] as f64, want_l.data[k] as f64);
                    md = md.max((a - b) as f32 * ((a - b) as f32).signum());
                    am += (a - b).abs();
                    s_xy += a * b;
                    s_yy += b * b;
                }
                eprintln!(
                    "  MCM-only: cas diff {:.3e} cas corr {:.4} | leaf diff {:.3e} leaf corr {:.4} (d_out=ref upstreams)",
                    am0 / d_y2.data.len() as f64,
                    s_xy0 / s_yy0,
                    am / d_y2.data.len() as f64,
                    s_xy / s_yy
                );
            }
            if i == 1 {
                let am = |t: &Tensor<f32>| {
                    t.data.iter().map(|v| v.abs()).sum::<f32>() / t.data.len() as f32
                };
                // Is `rp.rec` the reference's `x_hat`/`rec_Y_in`? `d_rec` is
                // computed on `rp.rec`; `x_hat_grad` on `x_hat` — a rec
                // mismatch misdirects the metric Jacobian.
                // `rec.{n}` numbers decoder calls: rec.0 is the baseline ref
                // point, rec.{i+1} is iteration `i`'s rec_Y_in — so iter `i`'s
                // rec is `rec.{i+1}`, not `rec.{i}`.
                let want_rec = dump[&format!("rdlr.rec.{}.x_hat", i + 1)].tensor_f32();
                let mut recd = 0.0f64;
                let mut reca = 0.0f64;
                for k in 0..rp.rec.data.len().min(want_rec.data.len()) {
                    recd += (rp.rec.data[k] - want_rec.data[k]).abs() as f64;
                    reca += want_rec.data[k].abs() as f64;
                }
                eprintln!(
                    "  rec vs x_hat: absdiff {:.3e} ref absmean {:.3e}",
                    recd / want_rec.data.len() as f64,
                    reca / want_rec.data.len() as f64
                );
                // `rp.y_hat` vs the reference `y_hat` — a systematic shift
                // changes the synth_tape activation pattern → ref_synth off.
                let want_yh1 = dump[&format!("rdlr.cas.{}.y_hat", i + 3)].tensor_f32();
                let mut yhd = 0.0f64;
                for k in 0..rp.y_hat.data.len().min(want_yh1.data.len()) {
                    yhd += (rp.y_hat.data[k] - want_yh1.data[k]).abs() as f64;
                }
                eprintln!(
                    "  y_hat vs cas.{}.y_hat: absdiff {:.3e}",
                    i + 3,
                    yhd / want_yh1.data.len() as f64
                );
                // `LightPrimary(y_hat)` vs `x_hat`/`decoder(y_hat)` on the
                // exact reference y_hat — any residual is a real
                // LightPrimary/decoder forward difference (not the y_hat
                // diff amplified), which would explain the ~20% Jacobian gap.
                let yh_b = BTensor::from_planar(&want_yh1, eng.tier.block()).unwrap();
                let (lh_rec, _) = synthesis
                    .forward_tape(eng, &yh_b, want_rec.h, want_rec.w, &enough::Unstoppable)
                    .unwrap();
                let (mut lhd, mut lhmax) = (0.0f64, 0.0f32);
                for k in 0..lh_rec.data.len().min(want_rec.data.len()) {
                    lhd += (lh_rec.data[k] - want_rec.data[k]).abs() as f64;
                    lhmax = lhmax.max((lh_rec.data[k] - want_rec.data[k]).abs());
                }
                eprintln!(
                    "  LightPrimary(y_hat) vs x_hat: absdiff {:.3e} max {:.3e}",
                    lhd / want_rec.data.len() as f64,
                    lhmax
                );
                // Where does LightPrimary differ from decoder? Edge vs interior,
                // and is it at clip-boundary (rec near 0/255) positions?
                let (mut e_edge, mut e_int, mut e_clip, mut e_noclip, mut n_clip) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
                let (rh, rw) = (want_rec.h, want_rec.w);
                for y in 0..rh {
                    for x in 0..rw {
                        let k = y * rw + x;
                        let d = (lh_rec.data[k] - want_rec.data[k]).abs() as f64;
                        if y < 11 || y >= rh - 11 || x < 11 || x >= rw - 11 {
                            e_edge += d;
                        } else {
                            e_int += d;
                        }
                        if want_rec.data[k] < 1.0 || want_rec.data[k] > 254.0 {
                            e_clip += d;
                            n_clip += 1;
                        } else {
                            e_noclip += d;
                        }
                    }
                }
                eprintln!(
                    "    edge absdiff {:.3e}  interior {:.3e}  clip {:.3e} (n {})  noclip {:.3e}",
                    e_edge / want_rec.data.len() as f64,
                    e_int / want_rec.data.len() as f64,
                    e_clip / want_rec.data.len() as f64,
                    n_clip,
                    e_noclip / want_rec.data.len() as f64
                );
                // `x_hat_grad` is indexed by decoder CALL, not iteration:
                // rec.0 is the ref point, rec.{i+1} is iteration i — so iter
                // i's upstream is `rdlr.{i+1}.x_hat_grad` (dump off-by-one).
                let want_dr = dump[&format!("rdlr.{}.x_hat_grad", i + 1)].tensor_f32();
                // Feed the reference's own `x_hat_grad` through the synthesis
                // backward — if downstream is right, `synth - d_resi` should
                // then equal `y_hat_grad` (isolating `d_rec` as the sole gap).
                let ref_synth = synthesis
                    .backward(eng, rp.synth_tape.as_ref().unwrap(), &want_dr)
                    .unwrap();
                let want_dyh = dump[&format!("rdlr.{i}.y_hat_grad")].tensor_f32();
                let (mut e_xy, mut e_xx, mut e_yy) = (0.0f64, 0.0f64, 0.0f64);
                for k in 0..ref_synth.data.len().min(want_dyh.data.len()) {
                    let a = ref_synth.data[k] as f64 - want_dresi.data[k] as f64;
                    let b = want_dyh.data[k] as f64;
                    e_xy += a * b;
                    e_xx += a * a;
                    e_yy += b * b;
                }
                eprintln!(
                    "  [ref x_hat_grad] synth-resi vs y_hat_grad: corr {:.4} scale {:.3}",
                    e_xy / (e_xx.sqrt() * e_yy.sqrt()),
                    e_xy / e_yy,
                );
                // Per-position adjoint check on `LightPrimary::backward`: for
                // the true adjoint `J^T`, `backward(v)[i]` equals the
                // directional derivative `⟨fwd(y+εe_i)−fwd(y), v⟩/ε`. Probing
                // ~40 positions localises any ~20% directional error the
                // whole-field inner product averages out.
                if i == 1 {
                    let v = &want_dr;
                    let bw = synthesis
                        .backward(eng, rp.synth_tape.as_ref().unwrap(), v)
                        .unwrap();
                    let eps = 1e-3f32;
                    let n = want_yh1.data.len();
                    let (ih, iw) = (want_rec.h, want_rec.w);
                    let mut base = want_yh1.clone();
                    let (base_rec, _) = synthesis
                        .forward_tape(
                            eng,
                            &BTensor::from_planar(&base, eng.tier.block()).unwrap(),
                            ih,
                            iw,
                            &enough::Unstoppable,
                        )
                        .unwrap();
                    let mut worst = 0.0f64;
                    let mut n_bad = 0usize;
                    for p in 0..40usize {
                        let idx = (p * 37 + 11) % n;
                        base.data[idx] += eps;
                        let (pf, _) = synthesis
                            .forward_tape(
                                eng,
                                &BTensor::from_planar(&base, eng.tier.block()).unwrap(),
                                ih,
                                iw,
                                &enough::Unstoppable,
                            )
                            .unwrap();
                        base.data[idx] -= eps;
                        let mut rhs = 0.0f64;
                        for k in 0..base_rec.data.len().min(v.data.len()) {
                            rhs += (pf.data[k] as f64 - base_rec.data[k] as f64) * v.data[k] as f64;
                        }
                        let lhs = bw.data[idx] as f64 * eps as f64;
                        let rel = (lhs - rhs).abs() / rhs.abs().max(1e-9);
                        worst = worst.max(rel);
                        if rel > 0.05 {
                            n_bad += 1;
                        }
                        if p < 6 {
                            eprintln!(
                                "    adjoint[{idx}] lhs {lhs:.5e} rhs {rhs:.5e} rel {rel:.3}"
                            );
                        }
                    }
                    eprintln!(
                        "  LightPrimary::backward per-pos adjoint: worst rel {worst:.3} bad(>5%) {n_bad}/40"
                    );
                }
                // Layer-by-layer `LightPrimary` vs reference `bop_prim`
                // (dumped to `/home/lilith/tmp/bop_layers`): the tape's
                // `act[i]`/`res_conv_out`/`pre_denorm` are the intermediates.
                if i == 1 && op == OperatingPoint::Bop {
                    let bop = load_dump(Path::new("/home/lilith/tmp/bop_layers"));
                    let crate::model::synthesis::PrimaryTape::Light(tape) =
                        rp.synth_tape.as_ref().unwrap()
                    else {
                        return;
                    };
                    let cmp = |mine: &Tensor<f32>, want: &Tensor<f32>, tag: &str| {
                        let mut md = 0.0f64;
                        let mut ad = 0.0f64;
                        for k in 0..mine.data.len().min(want.data.len()) {
                            let d = (mine.data[k] - want.data[k]).abs() as f64;
                            ad += d;
                            md = md.max(d);
                        }
                        eprintln!(
                            "  layer {tag}: absdiff {:.3e} max {:.3e} ({} vs {})",
                            ad / want.data.len() as f64,
                            md,
                            mine.data.len(),
                            want.data.len()
                        );
                    };
                    let rc = tape.res_conv_out.to_planar().unwrap();
                    cmp(&rc, &bop["bop.res_conv"].tensor_f32(), "res_conv");
                    for (j, tag) in ["act1_x", "act2_x", "act3_x"].iter().enumerate() {
                        let x = tape.act[j].0.to_planar().unwrap();
                        let m = tape.act[j].1.to_planar().unwrap();
                        cmp(&x, &bop[&format!("bop.{tag}")].tensor_f32(), tag);
                        cmp(
                            &m,
                            &bop[&format!("bop.act{}_mask", j + 1)].tensor_f32(),
                            &format!("act{}_mask", j + 1),
                        );
                    }
                    cmp(
                        &tape.pre_denorm,
                        &bop["bop.ps"].tensor_f32(),
                        "pre_denorm/ps",
                    );
                }
                let my_dr = rec_grad(eng, &up, &rp.rec, &original, area, None).unwrap();
                let (mut s_xy, mut s_xx, mut s_yy) = (0.0f64, 0.0f64, 0.0f64);
                for k in 0..my_dr.data.len() {
                    let (a, b) = (my_dr.data[k] as f64, want_dr.data[k] as f64);
                    s_xy += a * b;
                    s_xx += a * a;
                    s_yy += b * b;
                }
                // Edge-vs-interior: if the reference conv pads (same-size maps),
                // `x_hat_grad` is non-trivial in the outer 5px while the
                // valid-conv backward leaves it ~0.
                let (h, w) = (my_dr.h, my_dr.w);
                let (mut em, mut ew, mut im, mut iw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for y in 0..h {
                    for x in 0..w {
                        let edge = y < 5 || x < 5 || y + 5 >= h || x + 5 >= w;
                        let (a, b) = (my_dr.data[y * w + x], want_dr.data[y * w + x]);
                        if edge {
                            em += a.abs() as f64;
                            ew += b.abs() as f64;
                        } else {
                            im += a.abs() as f64;
                            iw += b.abs() as f64;
                        }
                    }
                }
                eprintln!(
                    "  d_rec edge mine {em:.4e} ref {ew:.4e} | interior mine {im:.4e} ref {iw:.4e}"
                );
                // Fit `x_hat_grad = c1·ms_grad + c2·psnr_grad` by least squares.
                // If the Jacobians are right, `c2/c1` should equal the
                // reference's `d_psnr/d_msssim` upstream ratio — isolating a
                // scalar-mixing error from a Jacobian error.
                {
                    let u = |m: f64, p: f64| LossUpstream {
                        d_msssim: m,
                        d_psnr: p,
                        d_rate: 0.0,
                        loss: 0.0,
                    };
                    let mg = rec_grad(eng, &u(1.0, 0.0), &rp.rec, &original, area, None).unwrap();
                    let pg = rec_grad(eng, &u(0.0, 1.0), &rp.rec, &original, area, None).unwrap();
                    let (mut mm, mut pp, mut mp, mut mw, mut pw) =
                        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    for k in 0..mg.data.len() {
                        let (m, p, w) =
                            (mg.data[k] as f64, pg.data[k] as f64, want_dr.data[k] as f64);
                        mm += m * m;
                        pp += p * p;
                        mp += m * p;
                        mw += m * w;
                        pw += p * w;
                    }
                    let det = mm * pp - mp * mp;
                    let (c1, c2) = ((mw * pp - pw * mp) / det, (pw * mm - mw * mp) / det);
                    // Residual R² of the two-Jacobian model on `x_hat_grad`.
                    let (mut res, mut tot) = (0.0f64, 0.0f64);
                    for k in 0..mg.data.len() {
                        let fit = c1 * mg.data[k] as f64 + c2 * pg.data[k] as f64;
                        res += (want_dr.data[k] as f64 - fit).powi(2);
                        tot += (want_dr.data[k] as f64).powi(2);
                    }
                    eprintln!(
                        "  upstream fit: c1 {:.4e} c2 {:.4e} R² {:.4} | mine d_ms {:.3e} d_psnr {:.3e}",
                        c1,
                        c2,
                        1.0 - res / tot,
                        up.d_msssim,
                        up.d_psnr,
                    );
                }
                eprintln!(
                    "  d_rec vs x_hat_grad: corr {:.4} scale {:.3} | gm_resi {:.3e} resi_hat {:.3e} y_hat {:.3e}",
                    s_xy / (s_xx.sqrt() * s_yy.sqrt()),
                    s_xy / s_yy,
                    am(&want_dresi),
                    am(&dump[&format!("rdlr.{i}.resi_hat_grad")].tensor_f32()),
                    am(&dump[&format!("rdlr.{i}.y_hat_grad")].tensor_f32()),
                );
            }
            let want_dy = dump[&format!("rdlr.{i}.cas_y_in_grad")].tensor_f32();
            let want_leaf = dump[&format!("rdlr.{i}.y_grad")].tensor_f32();
            for nm in ["x_hat_grad", "y_hat_grad"] {
                if let Some(d) = dump.get(&format!("rdlr.{i}.{nm}")) {
                    let t = d.tensor_f32();
                    let am = t.data.iter().map(|v| v.abs()).sum::<f32>() / t.data.len() as f32;
                    eprintln!("iter {i} ref {nm} absmean {am:.3e}");
                }
            }
            // `d_y` is the leaf gradient — compare against `y_grad`, not
            // `cas_y_in_grad` (which excludes the resi term's direct +1;
            // `cas_y_in + gm_resi == y_grad` holds exactly in the dump).
            let (mut md_y, mut md_leaf) = (0.0f32, 0.0f32);
            let (mut am_y, mut am_w) = (0.0f64, 0.0f64);
            let (mut s_xy, mut s_xx, mut s_yy) = (0.0f64, 0.0f64, 0.0f64);
            for ch in 0..160 {
                for k in 0..d_y.plane(ch).len() {
                    let (a, b) = (d_y.plane(ch)[k] as f64, want_leaf.plane(ch)[k] as f64);
                    md_y = md_y.max((d_y.plane(ch)[k] - want_dy.plane(ch)[k]).abs());
                    md_leaf = md_leaf.max((d_y.plane(ch)[k] - want_leaf.plane(ch)[k]).abs());
                    am_y += (a - b).abs();
                    am_w += b.abs();
                    s_xy += a * b;
                    s_xx += a * a;
                    s_yy += b * b;
                }
            }
            let corr = s_xy / (s_xx.sqrt() * s_yy.sqrt());
            eprintln!(
                "iter {i}: cas max {md_y:.3e} leaf {md_leaf:.3e} diff {:.3e} want {:.3e} corr {corr:.4} scale {:.3}",
                am_y / d_y.data.len() as f64,
                am_w / d_y.data.len() as f64,
                s_xy / s_yy
            );
            let want_rate = dump[&format!("rdlr.gcrp.{}.rate", i + 1)].f32s() as f64;
            let want_psnr = dump[&format!("rdlr.gcrp.{}.psnr", i + 1)].f32s() as f64;
            let want_msssim = dump[&format!("rdlr.gcrp.{}.msssim", i + 1)].f32s() as f64;
            let want_ry = dump[&format!("rdlr.gm.{}.rate_y", i + 1)].f32s() as f64;
            if i == 0 {
                eprintln!(
                    "  msssim mine {:.7} ref {:.7} (diff {:.2e})",
                    rp.msssim,
                    want_msssim,
                    rp.msssim - want_msssim
                );
            }
            assert!(
                (rp.rate_y - want_ry).abs() < want_ry * 0.002,
                "iter {i}: rate_y {} vs reference {want_ry}",
                rp.rate_y
            );
            assert!(
                (rp.bpp - want_rate).abs() < 2e-3,
                "iter {i}: bpp {} vs reference {want_rate}",
                rp.bpp
            );
            assert!(
                (rp.psnr - want_psnr).abs() < 0.05,
                "iter {i}: psnr {} vs reference {want_psnr}",
                rp.psnr
            );
            assert!(
                (rp.msssim - want_msssim).abs() < 2e-3,
                "iter {i}: msssim {} vs reference {want_msssim}",
                rp.msssim
            );
            // Adjoint check: `⟨d_rec, rec(y+εv) − rec(y)⟩/ε ≈ ⟨d_yhat, v⟩`.
            if i == 0 {
                let (rec_t, tape) = synthesis
                    .forward_tape(
                        eng,
                        &BTensor::from_planar(&rp.y_hat, eng.tier.block()).unwrap(),
                        original.h,
                        original.w,
                        &enough::Unstoppable,
                    )
                    .unwrap();
                // A deterministic pseudo-random `d_rec` (not the real one — the
                // adjoint identity holds for any upstream).
                let mut d_rec = Tensor::<f32>::zeros(1, rec_t.h, rec_t.w).unwrap();
                for (k, v) in d_rec.data.iter_mut().enumerate() {
                    *v = ((k * 7919) % 977) as f32 / 977.0 - 0.5;
                }
                let d_yh = synthesis.backward(eng, &tape, &d_rec).unwrap();
                // Full inner-product adjoint: a single random `v` verifies the
                // whole backward — ⟨d_yh, v⟩ must equal ⟨d_rec, rec(y+εv)−rec(y)⟩/ε.
                let mut v = rp.y_hat.clone();
                for (k, x) in v.data.iter_mut().enumerate() {
                    *x = ((k * 2654435761usize) % 997) as f32 / 997.0 - 0.5;
                }
                let lhs: f64 = d_yh
                    .data
                    .iter()
                    .zip(&v.data)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let mut yp = rp.y_hat.clone();
                let e = 1e-2f32;
                for (x, &b) in yp.data.iter_mut().zip(&v.data) {
                    *x += e * b;
                }
                let rec_p = synthesis
                    .forward_with(
                        eng,
                        &BTensor::from_planar(&yp, eng.tier.block()).unwrap(),
                        original.h,
                        original.w,
                        &enough::Unstoppable,
                    )
                    .unwrap();
                let rhs: f64 = d_rec
                    .data
                    .iter()
                    .zip(&rec_p.data)
                    .zip(&rec_t.data)
                    .map(|((&d, &rp_), &rt)| d as f64 * (rp_ - rt) as f64)
                    .sum::<f64>()
                    / e as f64;
                eprintln!(
                    "  synthesis adjoint: bwd {lhs:.4e}  fd {rhs:.4e}  ratio {:.3}",
                    lhs / rhs
                );
            }
        }
        // End-to-end driver smoke: `optimize_y` on the reference's own
        // constants — the loss trajectory should improve (the gradient is
        // ~0.72-correlated, so the path differs but descends).
        {
            let y0 = dump["rdlr.0.y"].tensor_f32();
            let ctx = set.common[0].context.as_ref().unwrap();
            let refs = (
                dump["rdlr.gcrp.0.rate"].f32s() as f64,
                dump["rdlr.gcrp.0.psnr"].f32s() as f64,
                dump["rdlr.gcrp.0.msssim"].f32s() as f64,
            );
            let (best, best_loss) = optimize_y(
                eng,
                ctx,
                &hdr,
                &set.common[0],
                &synthesis,
                lh,
                &y0,
                &z_hat,
                &scales.scale_log,
                &mask,
                &q,
                160,
                &original,
                area,
                pixels,
                8,
                0.03,
                0.7,
                refs,
                dump["rdlr.slope.psnrY"].f32s() as f64,
                dump["rdlr.slope.msssim_Torch"].f32s() as f64,
                dump["rdlr.ratio_psnr_msssim"].f32s() as f64,
                None,
                &enough::Unstoppable,
            )
            .unwrap();
            let ref_last_loss = dump["rdlr.7.loss"].f32s() as f64;
            eprintln!(
                "optimize_y: best_loss {best_loss:.4} (ref iter-7 {ref_last_loss:.4}), best y absmean {:.4}",
                best.data.iter().map(|v| v.abs()).sum::<f32>() / best.data.len() as f32
            );
            assert!(
                best_loss.is_finite() && best_loss < 0.0,
                "optimize_y best_loss {best_loss}"
            );
        }
    }

    /// The HOP synthesis VJP is the network adjoint: `⟨backward(d), v⟩` must
    /// equal `⟨d, forward(y+εv) − forward(y)⟩/ε` for any deterministic `v`/`d`.
    /// Self-contained — no reference dump, the same identity the conv VJPs
    /// (`conv2d_backward_is_adjoint`) already satisfy.
    #[test]
    fn hop_primary_backward_is_adjoint() {
        let models = std::env::var_os("ZENJPEGAI_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/lilith/work/zen/jpeg-ai-reference-software/models")
            });
        let enc = crate::encoder::Encoder::new(models);
        let eng = &enc.engine;
        let synthesis =
            crate::model::load_synthesis_primary(enc.models.as_ref(), 2, OperatingPoint::Hop, eng)
                .unwrap();
        // A small deterministic latent `[160, 4, 4]` over a 64x64 picture.
        let (h, w, lh, lw) = (64usize, 64usize, 4usize, 4usize);
        let mut y = crate::tensor::Tensor::<f32>::zeros(160, lh, lw).unwrap();
        for (k, v) in y.data.iter_mut().enumerate() {
            *v = ((k * 2654435761usize) % 1021) as f32 / 1021.0 - 0.5;
        }
        let (rec_t, tape) = synthesis
            .forward_tape(
                eng,
                &BTensor::from_planar(&y, eng.tier.block()).unwrap(),
                h,
                w,
                &enough::Unstoppable,
            )
            .unwrap();
        let mut d_rec = Tensor::<f32>::zeros(1, rec_t.h, rec_t.w).unwrap();
        for (k, v) in d_rec.data.iter_mut().enumerate() {
            *v = ((k * 7919) % 977) as f32 / 977.0 - 0.5;
        }
        let d_yh = synthesis.backward(eng, &tape, &d_rec).unwrap();
        let mut v = y.clone();
        for (k, x) in v.data.iter_mut().enumerate() {
            *x = ((k * 40503usize) % 1021) as f32 / 1021.0 - 0.5;
        }
        let lhs: f64 = d_yh
            .data
            .iter()
            .zip(&v.data)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let mut yp = y.clone();
        let e = 1e-3f32;
        for (x, &b) in yp.data.iter_mut().zip(&v.data) {
            *x += e * b;
        }
        let rec_p = synthesis
            .forward_with(
                eng,
                &BTensor::from_planar(&yp, eng.tier.block()).unwrap(),
                h,
                w,
                &enough::Unstoppable,
            )
            .unwrap();
        let rhs: f64 = d_rec
            .data
            .iter()
            .zip(&rec_p.data)
            .zip(&rec_t.data)
            .map(|((&d, &rp_), &rt)| d as f64 * (rp_ - rt) as f64)
            .sum::<f64>()
            / e as f64;
        eprintln!(
            "hop synthesis adjoint: bwd {lhs:.5e}  fd {rhs:.5e}  ratio {:.4}",
            lhs / rhs
        );
        assert!(
            ((lhs - rhs) / rhs.abs().max(1.0)).abs() < 5e-3,
            "hop synthesis backward not the adjoint: {lhs} vs {rhs}"
        );
    }
}

/// `process_z` rate point — the `encode_z + encode_y` total-bits eval for one
/// `z_hat` (the reference's `ECLibLH`-equivalent `ac.get_total_bits()`). The
/// z-RDLR path (`numIteZLuma`/`numIteZChroma`, dead upstream) optimizes the
/// hyper-latent `z_hat` toward the *total* component rate — a pure bitrate
/// objective, no distortion term.
pub(crate) struct ZRatePoint {
    /// `encode_z(z_hat)` + `encode_y(z_hat→psi→scale_log→residual)` total bits.
    total: f64,
    /// The integer `z_hat` the entropy / `hsd` / compress calls consume.
    z_hat_i8: Tensor<i8>,
    /// `hyper_decoder(z_hat)` at psi-res — the `HyperDecoder::backward` leaf.
    psi: BTensor,
    /// `hyper_decoder`'s VJP intermediates.
    hyper_tape: hyper_decoder::HyperTape,
    /// `compress`'s MCM tape (luma `compress_backward`) — `None` on the chroma
    /// context-free path.
    mcm_tape: Option<Vec<mcm::StageTape>>,
    /// `compress`'s reconstructed latent (mean + resi_dq) at latent res.
    y_hat: Tensor<f32>,
    /// The quantized residual (the `+ residual_q` term of `resi`).
    residual_q: Tensor<i16>,
    /// `hsd(z_hat)` — the sigma map (`rate_y`'s scale, held fixed in the VJP).
    scale_log: Tensor<i32>,
}

/// `process_z`'s `encode_z + encode_y` eval: `z_hat_cont` (the STE-quantized
/// continuous hyper-latent) → `z_hat_i8` (integer) → `psi` (hyper-decoder,
/// differentiable) + `scale_log` (int `hsd`) + `compress` → `rate_y` + `z_bits`.
pub(crate) fn rate_point_z(
    eng: &Engine,
    ccs: usize,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat_cont: &Tensor<f32>,
    z_hat_i8: &Tensor<i8>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    lh_y: usize,
    lw_y: usize,
    stop: &dyn enough::Stop,
) -> Result<ZRatePoint> {
    // psi at psi-res = latent/2 (`upshuffle_psi` produces the latent-res mean).
    let (psi_h, psi_w) = (lh_y.div_ceil(2), lw_y.div_ceil(2));
    let (psi, hyper_tape) = model
        .hyper_decoder
        .forward_tape(eng, z_hat_cont, psi_h, psi_w, stop)?;
    let scale_log = model.hsd.forward(z_hat_i8, lh_y, lw_y, SIGMA_IDX_MAX)?;
    let (out, mcm_tape) = if ccs == 0 {
        // luma: MCM `compress` — `d(psi)` flows through the stage VJPs.
        let ctx = model
            .context
            .as_ref()
            .ok_or(Error::InvalidArgument("process_z luma: no context model"))?;
        let o = ctx.compress(
            eng,
            y,
            &psi,
            q,
            mask,
            num_chs,
            crate::encoder::SKIP_CUBE_THR,
            stop,
        )?;
        (o.clone(), Some(o.tape.clone()))
    } else {
        (
            compress_context_free(y, &psi, q, mask, num_chs, crate::encoder::SKIP_CUBE_THR)?,
            None,
        )
    };
    let rate_y = gm_rate(y, &out.y_hat, &out.residual_q, &scale_log);
    let z_bits = lh.z_bits(z_hat_i8);
    Ok(ZRatePoint {
        total: z_bits + rate_y,
        z_hat_i8: z_hat_i8.clone(),
        psi,
        hyper_tape,
        mcm_tape,
        y_hat: out.y_hat,
        residual_q: out.residual_q,
        scale_log,
    })
}

/// `process_z`'s backward: `d(encode_z + encode_y)/d(z_hat)` at z-res. The
/// `encode_y` chain is `z_hat→psi→mean→resi→rate_y` (`d(mean) = -d_resi`,
/// `d(psi) = upshuffle_psi_backward`, `d(z_hat) = hyper_decoder.backward`);
/// `scale_log` (the int `hsd` path) is held fixed, matching the reference's
/// zero gradient through its integer convolutions. The `encode_z` term is the
/// `z_nbits` LUT's local slope.
pub(crate) fn backward_z(
    eng: &Engine,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    rp: &ZRatePoint,
    q: &impl mcm::Quantiser,
    mask: &Tensor<bool>,
    num_chs: usize,
    stop: &dyn enough::Stop,
) -> Result<Tensor<f32>> {
    let _ = eng;
    // `d(rate_y)/d(resi)` — `d_rate_y = 1` (the loss *is* the total bits).
    let d_resi = rate_vjp(1.0, y, &rp.y_hat, &rp.residual_q, &rp.scale_log);
    let (hh, hw) = (rp.psi.h, rp.psi.w);
    let d_psi = if let Some(mcm_tape) = &rp.mcm_tape {
        // luma MCM: `means_hat = y_hat - resi_dq` → `d(y_hat) = -d_resi`,
        // `d(resi_dq) = +d_resi`; `compress_backward` maps them onto `d(psi)`
        // (and a `d_y` the z-path ignores).
        let mut d_out_yhat = d_resi.clone();
        for v in d_out_yhat.data.iter_mut() {
            *v = -*v;
        }
        let mut d_psi = Tensor::<f32>::zeros(rp.psi.c, hh, hw)?;
        let ctx = model
            .context
            .as_ref()
            .ok_or(Error::InvalidArgument("process_z luma: no context model"))?;
        let _d_y = ctx.compress_backward(
            eng,
            y,
            &rp.psi,
            q,
            mask,
            num_chs,
            crate::encoder::SKIP_CUBE_THR,
            &d_out_yhat,
            &d_resi,
            Some(mcm_tape.clone()),
            stop,
            Some(&mut d_psi),
        )?;
        d_psi
    } else {
        // chroma context-free: `resi = y - means_hat`, `means_hat = mean =
        // upshuffle_psi(psi)` → `d(mean) = -d_resi` → `d(psi)`.
        let mut d_mean = d_resi.clone();
        for v in d_mean.data.iter_mut() {
            *v = -*v;
        }
        mcm::upshuffle_psi_backward(&d_mean, rp.psi.c, hh, hw)?
    };
    // `d(z_hat) = hyper_decoder.backward(d_psi)` + `d(z_bits)`.
    let mut d_z = model.hyper_decoder.backward(&rp.hyper_tape, &d_psi)?;
    let d_z_ent = lh.z_bits_vjp(&rp.z_hat_i8);
    for (o, &e) in d_z.data.iter_mut().zip(&d_z_ent.data) {
        *o += e;
    }
    Ok(d_z)
}

/// The reference's `Substract` STE: `z_hat = z_new + clamp(round(z_new).detach()
/// - z_new.detach(), -1, 1)` — `z_hat` keeps `z_new`'s value under the ±1 clamp
/// of the rounding residual, `d(z_hat)/d(z_new) = 1` where the clamp holds.
fn ste_z(z_new: &Tensor<f32>) -> Tensor<f32> {
    let mut out = z_new.clone();
    for (o, &z) in out.data.iter_mut().zip(&z_new.data) {
        *o = z + (z.round_ties_even() - z).clamp(-1.0, 1.0);
    }
    out
}

/// `z_hat` (continuous, post-STE) → the integer symbol tensor.
fn quant_z(z_hat: &Tensor<f32>) -> Result<Tensor<i8>> {
    let max = (crate::mans::MAX_Z as i32 - 1 - Z_OFFSET) as f32;
    Tensor::from_vec(
        z_hat.c,
        z_hat.h,
        z_hat.w,
        z_hat
            .data
            .iter()
            .map(|&v| v.clamp(-(Z_OFFSET as f32), max).round_ties_even() as i8)
            .collect(),
    )
}

/// `process_z`: optimize `z_Y`'s `z_hat` toward `encode_z + encode_y` total
/// bits via the `Substract` additive filter + Adagrad(lr·0.9/iter). Returns the
/// best integer `z_hat` (`best['z_hat'].round()`).
pub(crate) fn optimize_z(
    eng: &Engine,
    ccs: usize,
    model: &crate::model::common::CommonModel,
    lh: &lh::Likelihood,
    y: &Tensor<f32>,
    z_hat: &Tensor<i8>,
    mask: &Tensor<bool>,
    q: &impl mcm::Quantiser,
    num_chs: usize,
    lh_y: usize,
    lw_y: usize,
    num_ite: usize,
    stop: &dyn enough::Stop,
) -> Result<Tensor<i8>> {
    let (c, zh, zw) = (z_hat.c, z_hat.h, z_hat.w);
    let n = c * zh * zw;
    // `Substract`: `z_new = z_Y + additive2 + additive`.
    let mut additive2 = vec![0.0f32; n];
    let mut additive = 0.0f32;
    let mut sq2 = vec![0.0f32; n];
    let mut sq_scalar = 0.0f32;
    let mut lr = 0.05f32;
    let mut best = f64::INFINITY;
    let mut best_z = z_hat.clone();
    let mut nochange = 0usize;
    for _ in 0..num_ite {
        lr *= 0.9;
        // `z_new = z_Y + additive2 + additive`, then the STE `z_hat`.
        let mut z_new = Tensor::<f32>::zeros(c, zh, zw)?;
        for (o, (&zy, &a2)) in z_new.data.iter_mut().zip(z_hat.data.iter().zip(&additive2)) {
            *o = zy as f32 + a2 + additive;
        }
        let z_hat_cont = ste_z(&z_new);
        let z_hat_i8 = quant_z(&z_hat_cont)?;
        let rp = rate_point_z(
            eng,
            ccs,
            model,
            lh,
            y,
            &z_hat_cont,
            &z_hat_i8,
            mask,
            q,
            num_chs,
            lh_y,
            lw_y,
            stop,
        )?;
        if rp.total < best {
            best = rp.total;
            best_z = z_hat_i8;
            nochange = 0;
        } else {
            nochange += 1;
            if nochange == 7 {
                break;
            }
        }
        // `d(total)/d(z_hat)` at z-res; STE makes `d(z_new) = d(z_hat)`, so the
        // additive leaf gradients are `d(additive2) = d(z_new)`,
        // `d(additive) = sum(d(z_new))`.
        let d_z = backward_z(eng, model, lh, y, &rp, q, mask, num_chs, stop)?;
        for (i, (a2, sq)) in additive2.iter_mut().zip(&mut sq2).enumerate() {
            let g = d_z.data[i];
            *sq += g * g;
            *a2 -= lr * g / (sq.sqrt() + 1e-10);
        }
        let g_scalar: f32 = d_z.data.iter().sum();
        sq_scalar += g_scalar * g_scalar;
        additive -= lr * g_scalar / (sq_scalar.sqrt() + 1e-10);
    }
    // `best['z_hat'].round()` — already integer.
    Ok(best_z)
}
