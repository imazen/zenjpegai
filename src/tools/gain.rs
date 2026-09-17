//! Gain unit: per-channel quantisation step, signalled through `beta_displacement_log`.
//!
//! Ports `ref/src/codec/coding_tools/quantization/gain_unit/gain_unit.py`.
//!
//! In the log domain (sigma-index units, Q7) the scaler of channel `c` is
//! `gain_vector_log[c] + beta_displacement_log`; it is *added* to the sigma indices. In the
//! linear domain it is `exp(log * log_k / 128)` rounded to 10 fractional bits; residuals are
//! *divided* by it on the decoder side.

use alloc::vec::Vec;

/// `log_k = (ln(sigma_max) - ln(sigma_min)) / (levels - 1)` with the effective
/// `sigma_quant_min = 0.11`, `sigma_quant_max = 54.82`, `sigma_quant_level = 32`
/// (see [`crate::model::common::SIGMA_LEVELS`]): 0.20036548400207613.
pub fn log_k() -> f64 {
    (libm::log(54.82) - libm::log(0.11)) / 31.0
}

/// Fractional bits of the linear scaler (`gain_vector_precision + beta_displacement_precision`).
const SCALER_PRECISION: u32 = 10;

/// Linear-domain scaler for a log-domain value (`GainUnit._beta_displacement_log_updated`).
///
/// The reference computes this in float32 with `torch.exp`:
/// `round(exp(f32(log) * f32(log_k) / 128) * 1024) / 1024`, rounding half to even.
pub fn scaler_from_log(scaler_log: i32) -> f32 {
    let x = (scaler_log as f32 * log_k() as f32) / 128.0;
    let m = (1u32 << SCALER_PRECISION) as f32;
    (libm::expf(x) * m).round_ties_even() / m
}

/// Per-channel scalers of one component.
#[derive(Clone, Debug)]
pub struct GainUnit {
    /// Log-domain scaler per channel, added to the sigma indices.
    pub scaler_log: Vec<i32>,
    /// Linear-domain scaler per channel.
    pub scaler: Vec<f32>,
}

impl GainUnit {
    pub fn new(gain_vector_log: &[i32], beta_displacement_log: i32) -> Self {
        let scaler_log: Vec<i32> = gain_vector_log
            .iter()
            .map(|&g| g + beta_displacement_log)
            .collect();
        let scaler = scaler_log.iter().map(|&l| scaler_from_log(l)).collect();
        Self { scaler_log, scaler }
    }

    /// `dequantize_resi`: `x / (scaler + 1e-9)`, evaluated in float32 like the reference.
    #[inline]
    pub fn dequantize(&self, ch: usize, residual_q: f32) -> f32 {
        residual_q / (self.scaler[ch] + 1e-9f32)
    }
}
