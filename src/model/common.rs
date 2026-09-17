//! The "common modules" of one component and one trained model: everything the entropy stage
//! and the latent reconstruction need (`CommonEncDecModules` in the reference).
//!
//! Loaded from `models/VM_common_int/{Y,UV}_<beta>.pth`.

use alloc::format;
use alloc::vec::Vec;

use super::hsd::HyperScaleDecoder;
use super::hyper_decoder::HyperDecoder;
use super::mcm::ContextModel;
use crate::error::{Error, Result};
use crate::mans::{MAX_Z, normalize_z_cdf};
use crate::weights::Checkpoint;

/// Fractional bits of a sigma index (`QuantizerParams.sigma_precision`).
pub const SIGMA_PRECISION: u32 = 7;
/// Number of sigma quantisation levels.
///
/// `cfg/pipeline.json` says `sigma_quant_level: 35` (and `sigma_quant_max: 100`), but it sets
/// them on `tools_common`, where no tool owns such a parameter; the entropy modules therefore
/// run with their defaults: 32 levels, sigma in `[0.11, 54.82]`. Verified at runtime against the
/// reference (`hyper_scale_decoder.sigma_idx_max_value == 3967`). 32 is also the number of
/// distributions the entropy coder has tables for.
pub const SIGMA_LEVELS: i32 = 32;
/// Upper clamp of the hyper-scale decoder output: `(levels - 1) * 2^precision - 1`.
pub const SIGMA_IDX_MAX: i32 = (SIGMA_LEVELS - 1) * (1 << SIGMA_PRECISION) - 1;
/// `z` symbols are `z_hat + Z_OFFSET`.
pub const Z_OFFSET: i32 = 31;
/// Lower clamp of the log-domain gain vector (`gain_vector_log_bitdepth = 12`).
const GAIN_VECTOR_LOG_MIN: i32 = -(1 << 11);

/// Entropy-stage parameters of one component.
#[derive(Clone, Debug)]
pub struct CommonModel {
    /// Latent channel count (160 luma, 96 chroma).
    pub chs: usize,
    /// 8-bit CDF of `z` per channel, built from `hyper_entropy.freqs_int`.
    pub z_cdfs: Vec<[u8; MAX_Z]>,
    pub hsd: HyperScaleDecoder,
    /// Log-domain gain vector (`GainUnit.get_gain_vector_log`), one entry per latent channel.
    pub gain_vector_log: Vec<i32>,
    pub hyper_decoder: HyperDecoder,
    /// Present for luma only (`use_context_module`); chroma takes its mean straight from `psi`.
    pub context: Option<ContextModel>,
}

impl CommonModel {
    pub fn load(ck: &Checkpoint<'_>, chs: usize, eng: &crate::nn::fast::Engine) -> Result<Self> {
        if ck.bool("hyper_entropy.is_quantized")?.data != [true] {
            return Err(Error::Model(
                "hyper_entropy is not quantised (need VM_common_int)".into(),
            ));
        }
        let freqs = ck.i32("hyper_entropy.freqs_int")?;
        if freqs.shape != [chs, MAX_Z] {
            return Err(Error::Model(format!(
                "hyper_entropy.freqs_int: unexpected shape {:?}",
                freqs.shape
            )));
        }
        let mut z_cdfs = Vec::with_capacity(chs);
        for row in freqs.data.as_chunks::<MAX_Z>().0 {
            if row.iter().any(|&f| f < 0) || row.iter().all(|&f| f == 0) {
                return Err(Error::Model(
                    "hyper_entropy.freqs_int: invalid frequency row".into(),
                ));
            }
            z_cdfs.push(normalize_z_cdf(row));
        }

        // Gain unit: `vr_vec.c` is [chs, num_betas] with exactly one populated column (the
        // model's own beta); the log-domain gain is that column in Q7, rounded half to even.
        let c = ck.f32("vr_vec.c")?;
        if c.shape.len() != 2 || c.shape[0] != chs || c.shape[1] == 0 {
            return Err(Error::Model(format!(
                "vr_vec.c: unexpected shape {:?}",
                c.shape
            )));
        }
        let n_beta = c.shape[1];
        let column_mean = |j: usize| -> f32 {
            // torch.mean over a float32 column; only compared against exactly zero.
            (0..chs).map(|i| c.data[i * n_beta + j]).sum::<f32>() / chs as f32
        };
        let all_ones = c.data.iter().all(|&v| (v - 1.0).abs() < 1e-5);
        let vec_idx = if all_ones {
            0
        } else {
            let means: Vec<f32> = (0..n_beta).map(column_mean).collect();
            let (mut min_n, mut max_n) = (0usize, n_beta - 1);
            if means.iter().all(|&m| m == 0.0) {
                max_n = 0;
            } else {
                for i in 0..n_beta - 1 {
                    if means[i] == 0.0 && means[i + 1] != 0.0 {
                        min_n = i + 1;
                    }
                    if means[i] != 0.0 && means[i + 1] == 0.0 {
                        max_n = i;
                    }
                }
            }
            if min_n != max_n {
                return Err(Error::Model(
                    "vr_vec.c: more than one populated gain column".into(),
                ));
            }
            min_n
        };
        let gain_vector_log = (0..chs)
            .map(|i| {
                let v = c.data[i * n_beta + vec_idx] * (1 << SIGMA_PRECISION) as f32;
                (v.round_ties_even() as i32).max(GAIN_VECTOR_LOG_MIN)
            })
            .collect();

        let context = if ck.contains("context.MCM.0.fusion_pred_net.conv1.weight") {
            Some(ContextModel::load(ck, chs, eng)?)
        } else {
            None
        };
        Ok(Self {
            chs,
            z_cdfs,
            hsd: HyperScaleDecoder::load(ck, chs, eng)?,
            gain_vector_log,
            hyper_decoder: HyperDecoder::load(ck, chs, eng)?,
            context,
        })
    }
}
