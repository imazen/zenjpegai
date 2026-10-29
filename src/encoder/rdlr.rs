//! `coding_tools/rdlr/` — Rate-Distortion Latent Refinement: the encoder-side
//! iterative latent optimiser, the last coding tool not ported.
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

use crate::error::{Error, Result};
use crate::model::common::SIGMA_PRECISION;
use crate::model::mcm;
use crate::model::synthesis::SynthesisPrimary;
use crate::nn::fast::math;
use crate::nn::fast::{BTensor, Engine};
use crate::tensor::Tensor;
use crate::tools::gain::log_k;
use crate::tools::regions::Area;

use super::{compress_regions, lh};

/// `ln(sigma_quant_min)` — `pipeline.json`'s `sigma_quant_min = 0.11`, the same
/// value `GMProbModel` lower-bounds its scales at (`scale_bound` defaults to it).
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
    /// `residual_dq` — the dequantised residual.
    pub residual_dq: Tensor<f32>,
    /// The merged hyper-decoder output the quantisation was measured against.
    pub psi: Tensor<f32>,
    /// `y_hat = residual_dq + means` — the reconstructed latent, decoder-exact.
    pub y_hat: Tensor<f32>,
    /// The reconstructed picture plane (0..255) at the component's resolution.
    pub rec: Tensor<f32>,
    /// `-log2` GM likelihood of `y - means_hat`, in bits (the `rate_y` term).
    pub rate_y: f64,
    /// Factorized `z_hat` bit count (the `bs.encode_term()` term).
    pub z_bits: f64,
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
    stop: &dyn enough::Stop,
) -> Result<RatePoint> {
    // `_compress_ar_scale(y, decisions)`: residual quantisation per region. The
    // straight-through step the gradient path replaces with identity lives
    // inside (residual_q vs residual_dq).
    let out = compress_regions(
        eng, hdr, 0, model, z_hat, y, mask, q, num_chs, y.h, y.w, stop,
    )?;
    let mcm::Compressed {
        residual_q,
        residual,
        psi,
        ..
    } = out;
    let ctx = model.context.as_ref().ok_or(Error::InvalidArgument(
        "RDLR: luma component has no context model",
    ))?;
    let psi_b = BTensor::from_planar(&psi, eng.tier.block())?;
    let y_hat = ctx.decompress_with(eng, &residual, &psi_b, stop)?;

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
    let log_k = log_k() as f32;
    let inv = 1.0f32 / (1u32 << SIGMA_PRECISION) as f32;
    let mut rate_y = 0.0f64;
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
            rate_y += gm_nbits(resi, sigma);
        }
    }

    // `tool.decoder(y_new2, h, w)` — the luma synthesis on the reconstructed
    // latent.
    let rec = synthesis.forward_with(
        eng,
        &BTensor::from_planar(&y_hat, eng.tier.block())?,
        original.h,
        original.w,
        stop,
    )?;
    let (psnr, msssim) = distortion(eng, &rec, original, area)?;

    let z_bits = lh.z_bits(z_hat);
    let bpp = (z_bits + rate_y) / pixels;
    Ok(RatePoint {
        bpp,
        psnr,
        msssim,
        residual_q,
        residual_dq: residual,
        psi,
        y_hat,
        rec,
        rate_y,
        z_bits,
    })
}

/// `GMProbModel._likelihood` of a continuous residual with `means = 0`, as
/// `-log2` bits: the unit-interval Gaussian mass `Φ((0.5-|v|)/σ) - Φ((-0.5-|v|)/σ)`
/// written with `erfc`, floored at `likelihood_bound`.
fn gm_nbits(resi: f32, sigma: f32) -> f64 {
    // `get_std_cum_freq`'s `const = -(2**-0.5)`.
    const C: f64 = -0.7071067811865476;
    let s = sigma as f64;
    let v = resi.abs() as f64;
    let upper = 0.5 * libm::erfc(C * (0.5 - v) / s);
    let lower = 0.5 * libm::erfc(C * (-0.5 - v) / s);
    let p = (upper - lower).max(LIKELIHOOD_BOUND);
    -p.log2()
}

/// `calc_distortion_loss` for the luma tile: `mean((a-b)²)` over `area` (+1e-10),
/// `ms_ssim(a, b, data_range = 255)`.
fn distortion(
    eng: &Engine,
    rec: &Tensor<f32>,
    original: &Tensor<f32>,
    area: Area,
) -> Result<(f64, f64)> {
    // MS-SSIM's `data_range = 255` is `data_range = 1` on inputs scaled by 1/255
    // (the C1/C2 constants are the only place the range enters).
    const INV: f32 = 1.0 / 255.0;
    let (ah, aw) = (area.height, area.width);
    let (mut a, mut b) = (Vec::with_capacity(ah * aw), Vec::with_capacity(ah * aw));
    let mut mse = 0.0f64;
    for row in 0..ah {
        for x in 0..aw {
            let o = original.plane(0)[(area.y + row) * original.w + area.x + x];
            let r = rec.plane(0)[(area.y + row) * rec.w + area.x + x];
            a.push(o * INV);
            b.push(r * INV);
            mse += ((o - r) as f64) * ((o - r) as f64);
        }
    }
    mse /= (ah * aw) as f64;
    let psnr = 20.0 * (255.0f64).log10() - 10.0 * (mse + 1e-10).log10();
    let (ta, tb) = (
        Tensor::from_vec(1, ah, aw, a)?,
        Tensor::from_vec(1, ah, aw, b)?,
    );
    let msssim = crate::encoder::msssim::ms_ssim(eng, &ta, &tb)? as f64;
    Ok((psnr, msssim))
}

/// `Likelihood`'s `z_bits` accessor lives in `lh.rs` (it reads the private
/// `z_nbits` table directly).

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
        fn tensor_i32(&self) -> Tensor<i32> {
            assert_eq!(self.dtype, "f32");
            Tensor::from_vec(
                self.shape[1],
                self.shape[2],
                self.shape[3],
                self.f32().into_iter().map(|v| v as i32).collect(),
            )
            .unwrap()
        }
    }

    fn vectors_root() -> PathBuf {
        std::env::var_os("ZENJPEGAI_VECTORS")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"))
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The dumped encode's whole context is rebuilt from the dumped `z_hat`
    /// (`component_scales_with` output is asserted exactly), then every dumped
    /// `rdlr.i.y` is run through `rate_point_y` and compared to the
    /// `gcrp.(i+1)` rate point the reference measured on it.
    #[test]
    fn rate_point_matches_reference_iterations() {
        let dump = load_dump(&vectors_root().join("rdlr_img30_bop_m1_b0"));
        let models = std::env::var_os("ZENJPEGAI_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/home/lilith/work/zen/jpeg-ai-reference-software/models")
            });
        let enc = crate::encoder::Encoder::new(models);
        let eng = &enc.engine;
        let params = EncodeParams::default();
        let set = enc.model_set(1, OperatingPoint::Bop).unwrap();
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
            crate::model::load_synthesis_primary(enc.models.as_ref(), 1, OperatingPoint::Bop, eng)
                .unwrap();
        let lh = &set.lh[0];
        let original = dump["rdlr.0.original"].tensor_f32();
        let pixels = (888.0 * 560.0) as f64;
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
                &enough::Unstoppable,
            )
            .unwrap();
            // `_compress_ar_scale`'s outputs replayed exactly (`cas.0/1` are
            // the main encode, `cas.2` the ref point, `cas.3+i` iteration i).
            let c = i + 3;
            let want_rq = dump[&format!("rdlr.cas.{c}.resi_hat")].tensor_i8();
            let got: Vec<i16> = want_rq.data.iter().map(|&v| v as i16).collect();
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
            assert!(unflipped < 0.15, "iter {i}: y_hat off-flip max {unflipped}");
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
                    ms = ms.max((lin - gm_scale.plane(ch)[k]).abs());
                }
            }
            assert!(ms < 4e-6, "iter {i}: gm scale diff {ms}");
            assert!(md < 0.15, "iter {i}: gm resi diff {md}");
            let want_rate = dump[&format!("rdlr.gcrp.{}.rate", i + 1)].f32s() as f64;
            let want_psnr = dump[&format!("rdlr.gcrp.{}.psnr", i + 1)].f32s() as f64;
            let want_msssim = dump[&format!("rdlr.gcrp.{}.msssim", i + 1)].f32s() as f64;
            let want_ry = dump[&format!("rdlr.gm.{}.rate_y", i + 1)].f32s() as f64;
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
        }
    }
}
