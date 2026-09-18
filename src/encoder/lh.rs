//! Likelihood-based rate estimation: the `ECLibLH` backend the reference's bitrate matcher
//! measures every trial with
//! (`ref/src/codec/entropy_coding/lib_wrappers/lh/`).
//!
//! Instead of coding a trial, the reference runs its entropy stage over the decisions and
//! accumulates `-sum(log2(p))` of two float probability models (`ec_module.py`'s
//! `get_total_bits`):
//!
//! - `z_hat` — `CustomProbWrapper`: `FactorizedProbModel.forward(z_hat)` —
//!   `|sigmoid(s·L(z+0.5)) - sigmoid(s·L(z-0.5))|` with `s = -sign(L(z+0.5)+L(z-0.5))`, the
//!   cumulative logits `L` being a per-channel `[1→3→3→3→1]` MLP over `softplus(matrix)`
//!   weights with `tanh` residual factors (`prob_models/factorized.py`), clamped to
//!   `[1e-9, 1]`. `z_hat` takes only `MAX_Z` values, so each channel's likelihoods are a
//!   small table.
//! - residual — `SgtProbWrapper` with `use_simple_model`: the *unquantised* Gaussian of
//!   `GMProbModel.forward` (`prob_models/gm.py`): the sigma index (Q7) maps through
//!   `max(exp(idx·log_k + ln 0.11), 0.11)` and the likelihood of `|v|` is
//!   `Φ((0.5-v)/σ) - Φ((-0.5-v)/σ)`, clamped to `[1e-9, 1-1e-9]`.
//!
//! The wrapper ignores the skip mask: masked-out positions enter the sum as the zero
//! symbol, which is what `Component.residual_q` already holds (the reference zeroes
//! `residual_quant` there before estimating). The estimate is what the *search* sees;
//! the returned stream is still coded for real.

use alloc::vec::Vec;

use crate::decoder::entropy::distribution_index;
use crate::error::{Error, Result};
use crate::mans::MAX_Z;
use crate::model::common::{SIGMA_LEVELS, Z_OFFSET};
use crate::tensor::Tensor;
use crate::tools::gain::log_k;
use crate::weights::Checkpoint;

/// `sigma_quant_min` / `scale_bound` (`pipeline.json` effective value).
const SIGMA_MIN: f64 = 0.11;
/// `likelihood_bound` of `GMProbModel` and `freq_bound` of `FactorizedProbModel`.
const FREQ_BOUND: f64 = 1e-9;
/// Layer widths of `FactorizedProbModel.filters` plus the scalar output.
const LAYER_WIDTHS: [usize; 4] = [3, 3, 3, 1];

/// The factorized `z` model of one component, as per-channel likelihood tables.
///
/// `z_nbits[ch][sym]` is `-log2(freq)` for `z_hat = sym - Z_OFFSET` — the value
/// `CustomProbWrapper.compute_bits` accumulates for that symbol.
pub(crate) struct Likelihood {
    z_nbits: Vec<[f64; MAX_Z]>,
}

impl Likelihood {
    /// Reads `hyper_entropy.{matrices,biases,factors}` of a `VM_common_int` checkpoint.
    pub(crate) fn load(ck: &Checkpoint<'_>) -> Result<Self> {
        // `get_cum_logits`: `softplus(matrix) @ logits + bias`, then `+= tanh(factor) *
        // tanh(logits)` on every layer but the last. Weights are softplus'ed and factors
        // tanh'ed once here.
        let mut w: [Vec<f64>; 4] = Default::default();
        let mut b: [Vec<f64>; 4] = Default::default();
        let mut f: [Vec<f64>; 3] = Default::default();
        let mut chs = 0usize;
        for (l, &out) in LAYER_WIDTHS.iter().enumerate() {
            let inp = if l == 0 { 1 } else { LAYER_WIDTHS[l - 1] };
            let m = ck.f32(&alloc::format!("hyper_entropy.matrices.{l}"))?;
            if m.shape.len() != 3 || m.shape[1] != out || m.shape[2] != inp {
                return Err(Error::Model(alloc::format!(
                    "hyper_entropy.matrices.{l}: unexpected shape {:?}",
                    m.shape
                )));
            }
            chs = m.shape[0];
            w[l] = m.data.iter().map(|&v| softplus(v as f64)).collect();
            b[l] = ck
                .f32(&alloc::format!("hyper_entropy.biases.{l}"))?
                .data
                .iter()
                .map(|&v| v as f64)
                .collect();
            if l < 3 {
                f[l] = ck
                    .f32(&alloc::format!("hyper_entropy.factors.{l}"))?
                    .data
                    .iter()
                    .map(|&v| (v as f64).tanh())
                    .collect();
            }
        }

        let mut z_nbits = alloc::vec![[0.0f64; MAX_Z]; chs];
        for (ch, lut) in z_nbits.iter_mut().enumerate() {
            // Cumulative logits on the half-integer grid `-Z_OFFSET - 0.5 ..= Z_OFFSET - 0.5`
            // (`get_freq` evaluates `x + 0.5` and `x - 0.5` for integer `x`).
            let mut cum = [0.0f64; MAX_Z + 1];
            for (t, c) in cum.iter_mut().enumerate() {
                *c = cum_logit(&w, &b, &f, ch, t as f64 - Z_OFFSET as f64 - 0.5);
            }
            for (sym, n) in lut.iter_mut().enumerate() {
                let (upper, lower) = (cum[sym + 1], cum[sym]);
                // `sign = -(upper + lower).sign()`: flip towards the sigmoid's left tail.
                let sign = if upper + lower > 0.0 {
                    -1.0
                } else if upper + lower < 0.0 {
                    1.0
                } else {
                    0.0
                };
                let freq = (sigmoid(sign * upper) - sigmoid(sign * lower))
                    .abs()
                    .clamp(FREQ_BOUND, 1.0);
                *n = -freq.log2();
            }
        }
        Ok(Self { z_nbits })
    }

    /// `ECLibLH.total_bits` of one component: the factorized estimate of `z_hat` plus the
    /// Gaussian estimate of the (already masked) `residual_q`. `num_chs` truncates the
    /// residual only — `z` is coded over all channels whatever the residual's channel count
    /// (`_ac_encode_z` never slices `z_hat`).
    pub(crate) fn bits(
        &self,
        z_hat: &Tensor<i8>,
        scale_log: &Tensor<i32>,
        residual_q: &Tensor<i16>,
        num_chs: usize,
    ) -> f64 {
        let mut bits = 0.0;
        for ch in 0..z_hat.c {
            let lut = &self.z_nbits[ch];
            for &z in z_hat.plane(ch) {
                let sym = (z as i32 + Z_OFFSET).clamp(0, MAX_Z as i32 - 1) as usize;
                bits += lut[sym];
            }
        }
        // Per-distribution memo: `-log2` of the Gaussian likelihood of `|v|`, grown on
        // demand — a trial visits at most a few hundred distinct magnitudes.
        let mut memo: [Vec<f64>; SIGMA_LEVELS as usize] = core::array::from_fn(|_| Vec::new());
        for ch in 0..residual_q.c.min(num_chs).min(scale_log.c) {
            let (s, r) = (scale_log.plane(ch), residual_q.plane(ch));
            for i in 0..r.len() {
                let idx = distribution_index(s[i]) as usize;
                let v = r[i].unsigned_abs() as usize;
                let m = &mut memo[idx];
                if v >= m.len() {
                    let sigma = sigma_of_index(idx);
                    m.extend((m.len()..=v).map(|vv| gaussian_nbits(vv, sigma)));
                }
                bits += m[v];
            }
        }
        bits
    }
}

/// Linear scale of one sigma-table entry: `GMProbModel._index_to_scale`, `exp(idx·log_k +
/// ln(sigma_min))` lower-bounded at `sigma_min` (the `·2^7`/`/2^7` of the reference's Q7
/// fixed point cancels).
fn sigma_of_index(idx: usize) -> f64 {
    (idx as f64 * log_k() + SIGMA_MIN.ln()).exp().max(SIGMA_MIN)
}

/// `GMProbModel._likelihood` for `means = None`: `Φ((0.5-|v|)/σ) - Φ((-0.5-|v|)/σ)` as
/// `erfc`, then `-log2` of it clamped the way `SgtProbWrapper.compute_bits` does
/// (`freq.clamp(max=1-1e-9)`; the model itself lower-bounds at `1e-9`).
fn gaussian_nbits(v: usize, sigma: f64) -> f64 {
    let v = v as f64;
    let upper = 0.5 * libm::erfc(core::f64::consts::FRAC_1_SQRT_2 * (v - 0.5) / sigma);
    let lower = 0.5 * libm::erfc(core::f64::consts::FRAC_1_SQRT_2 * (v + 0.5) / sigma);
    let freq = (upper - lower).clamp(FREQ_BOUND, 1.0 - FREQ_BOUND);
    -freq.log2()
}

/// `F.softplus` (`beta = 1`, `threshold = 20`).
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `get_cum_logits` for one channel: `softplus(W) @ x + b`, then `+= tanh(a)·tanh(·)` on
/// all but the last layer.
fn cum_logit(w: &[Vec<f64>; 4], b: &[Vec<f64>; 4], f: &[Vec<f64>; 3], ch: usize, x: f64) -> f64 {
    let mut v = [x, 0.0, 0.0];
    let mut n_in = 1usize;
    for (l, &n_out) in LAYER_WIDTHS.iter().enumerate() {
        let mut nv = [0.0f64; 3];
        for k in 0..n_out {
            let row = (ch * n_out + k) * n_in;
            let mut s = b[l][ch * n_out + k];
            for j in 0..n_in {
                s = w[l][row + j].mul_add(v[j], s);
            }
            if l < 3 {
                s += f[l][ch * n_out + k] * s.tanh();
            }
            nv[k] = s;
        }
        v = nv;
        n_in = n_out;
    }
    v[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// The Gaussian likelihood is symmetric in `v` and integrates to ~1 over the symbol
    /// range: for `σ = 1` `p(0) ≈ 0.383`, `p(1) ≈ 0.242`, `p(2) ≈ 0.061` (normal pmf of
    /// `|X| ∈ [v-0.5, v+0.5]`).
    #[test]
    fn gaussian_likelihood_matches_scipy() {
        // Φ(0.5) - Φ(-0.5) = erf(0.5/√2) ≈ 0.38292492254802624
        let p0 = 2f64.powf(-gaussian_nbits(0, 1.0));
        assert!((p0 - 0.38292492254802624).abs() < 1e-15, "{p0}");
        // Φ(1.5) - Φ(0.5) ≈ 0.2417303374571288
        let p1 = 2f64.powf(-gaussian_nbits(1, 1.0));
        assert!((p1 - 0.2417303374571288).abs() < 1e-15, "{p1}");
    }

    /// `sigma_of_index` reproduces the reference's log-spaced table: index 0 → 0.11,
    /// index 31 → 54.82, geometric in between.
    #[test]
    fn sigma_table_is_log_spaced() {
        assert!((sigma_of_index(0) - SIGMA_MIN).abs() < 1e-16);
        let last = sigma_of_index(31);
        assert!((last - 54.82).abs() < 1e-12, "{last}");
        let ratio = sigma_of_index(1) / sigma_of_index(0);
        assert!((ratio - log_k().exp()).abs() < 1e-15, "{ratio}");
    }

    /// `bits` sums `z` and residual terms; a zero residual at index 0 costs `-log2 p(0)`.
    #[test]
    fn bits_sum_over_planes() {
        let lh = Likelihood {
            z_nbits: vec![[2.0; MAX_Z]; 1],
        };
        let z_hat = Tensor::from_vec(1, 1, 4, vec![-31, 0, 15, 30]).unwrap();
        let scale_log = Tensor::from_vec(1, 1, 4, vec![0; 4]).unwrap();
        let residual_q = Tensor::from_vec(1, 1, 4, vec![0, 1, -1, 0]).unwrap();
        let got = lh.bits(&z_hat, &scale_log, &residual_q, 1);
        // z: 4 symbols * 2 bits; residual at sigma idx 0: 2·nbits(0) + 2·nbits(1)
        let sigma0 = sigma_of_index(0);
        let want = 8.0 + 2.0 * gaussian_nbits(0, sigma0) + 2.0 * gaussian_nbits(1, sigma0);
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}
