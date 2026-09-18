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

#[cfg(test)]
mod tests {
    use super::*;

    /// `scaler_from_log` against `torch.exp` on every reachable input.
    ///
    /// `tests/vectors/gain_scaler.bin` (made by
    /// `scripts/ref_vectors/gen_gain_scaler_vectors.py`) holds the reference scaler for
    /// `gain_vector_log[c] + beta_displacement_log` over every gain entry of the eight
    /// `VM_common_int` checkpoints and every beta the 12-bit header field can signal
    /// (-2048..=2047 — the -1069..702 clip is encoder-side only). The reachable sums tile one
    /// contiguous range; the file stores the sorted gain values, then the f32 bits of the
    /// reference scaler for each sum in it.
    #[test]
    fn scaler_matches_torch_exp_on_every_reachable_input() {
        let v = include_bytes!("../../tests/vectors/gain_scaler.bin");
        let (head, rest) = v.split_at(12);
        let min_log = i32::from_le_bytes(head[0..4].try_into().unwrap());
        let n_gain = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
        let n_val = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let (gains, vals) = rest.split_at(n_gain * 2);
        assert_eq!(vals.len(), n_val * 4);
        let gains: Vec<i32> = gains
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| i16::from_le_bytes(b) as i32)
            .collect();
        let want = |scaler_log: i32| -> u32 {
            let i = (scaler_log - min_log) as usize;
            u32::from_le_bytes(vals[i * 4..i * 4 + 4].try_into().unwrap())
        };

        let mut checked = 0usize;
        for &g in &gains {
            for beta in -(1 << 11)..=(1 << 11) - 1 {
                let scaler_log = g + beta;
                assert!(
                    (min_log..min_log + n_val as i32).contains(&scaler_log),
                    "{scaler_log} outside the dumped range"
                );
                let got = scaler_from_log(scaler_log).to_bits();
                assert_eq!(
                    got,
                    want(scaler_log),
                    "scaler_from_log({scaler_log}) (gain {g} + beta {beta})"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, n_gain * (1 << 12));
    }
}
