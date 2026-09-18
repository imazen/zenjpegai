//! Rate matching: choose the trained model and the quantiser displacement that hit a target
//! bits-per-pixel.
//!
//! Ports the structure of `ref/src/codec/coding_tools/bitrate_matcher/bitrate_matcher.py`
//! (`match_luma` + `beta_linear_interpolation`): pick the model whose rate at displacement 0 is
//! relatively closest to the target, then bracket the displacement with the model's search
//! range, interpolate in the log-rate domain and bisect a +/-100 window, keeping the tightest
//! trial inside the configured tolerance.
//!
//! The reference measures each trial with a *likelihood* estimate (`ECLibLH`: `-sum(log2(p))`
//! of the entropy model, excluding the container) instead of coding it; that is ported as
//! [`RateEstimate::Likelihood`]. The default [`RateEstimate::Coded`] keeps this port's earlier
//! choice of measuring the real codestream — the rate you get, not one the estimator
//! predicted — at the parameter file's default tolerance. `find_UV_beta_with_hyperopt` is not
//! ported either: the shipped configuration (`cfg/BRM/default.json`, `independent_beta_UV: 1`)
//! never reaches it, and it is a stochastic TPE search.

use crate::error::Result;

/// `BDL_range` per model (`cfg/BRM/default.json`): the displacements the rate matcher searches.
/// Narrower than the clipping range `BDL_clipping_range` that [`super::BDL_RANGE`] applies.
pub const BDL_SEARCH_RANGE: [(i32, i32); 4] = [(-1024, 259), (-443, 259), (-443, 702), (-443, 702)];

/// `tolerance_min` / `tolerance_max` (`bitrate_matcher/params.py`).
pub(crate) const TOLERANCE_MIN: f64 = -0.01;
pub(crate) const TOLERANCE_MAX: f64 = 0.01;

/// The tolerances the reference actually searches with: `--set_target_bpp` always appends
/// `cfg/BRM/regen_list.json`, which overrides the parameter defaults to `tolerance_min =
/// -0.10`, `tolerance_max = +0.05`. [`RateEstimate::Likelihood`] uses them so the search
/// accepts the same window the reference does.
pub(crate) const TOLERANCE_LH: (f64, f64) = (-0.10, 0.05);

/// How a rate-matching trial is measured (`EncodeParams::rate_estimate`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RateEstimate {
    /// Code the trial and measure the codestream size. The default: our rate is the one you
    /// get, and the tight +/-1 % parameter default applies.
    #[default]
    Coded,
    /// `ECLibLH` — the `-sum(log2(p))` likelihood estimate ([`super::lh`]) the reference's
    /// bitrate matcher searches with, at the reference's effective tolerances
    /// (`cfg/BRM/regen_list.json`: -10 % / +5 %). The search then picks the same
    /// `(model, beta)` the reference does; the returned stream is still coded for real.
    Likelihood,
}

/// What a rate-matched encode settled on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateMatch {
    pub model_id: u8,
    pub beta_displacement_log: i32,
    /// Achieved rate, over the coded picture's luma samples.
    pub bpp: f64,
    /// The likelihood estimate at the chosen displacement, when
    /// [`EncodeParams::rate_estimate`](crate::encoder::EncodeParams::rate_estimate) is
    /// [`RateEstimate::Likelihood`]; `None` for [`RateEstimate::Coded`].
    pub estimated_bpp: Option<f64>,
    /// Trial encodes performed (both stages).
    pub trials: usize,
}

/// `beta_linear_interpolation` for one model. `bpp_at(beta)` codes the picture at that
/// displacement and returns its rate; it is never called twice with the same displacement.
pub fn search_beta(
    target: f64,
    range: (i32, i32),
    bpp_at: impl FnMut(i32) -> Result<f64>,
) -> Result<i32> {
    search_beta_with(target, range, (TOLERANCE_MIN, TOLERANCE_MAX), true, bpp_at)
}

/// [`search_beta`] with explicit `(tolerance_min, tolerance_max)` and window handling:
/// `clamp_window` clips the +/-100 bisection window to `range`, which the reference does not
/// do (only the encoder's `BDL_clipping_range` applies downstream); [`RateEstimate::Coded`]
/// keeps the clipping because it only skips trials that re-code an endpoint already measured.
pub(crate) fn search_beta_with(
    target: f64,
    range: (i32, i32),
    tol: (f64, f64),
    clamp_window: bool,
    mut bpp_at: impl FnMut(i32) -> Result<f64>,
) -> Result<i32> {
    let (mut tol_min, mut tol_max) = tol;
    let base = bpp_at(0)?;
    let base_mismatch = (base - target) / target;
    if (tol_min..=tol_max).contains(&base_mismatch) {
        return Ok(0);
    }
    let (mut min_beta, mut max_beta) = range;
    let (mut min_bits, mut max_bits) = (bpp_at(min_beta)?, bpp_at(max_beta)?);
    // `judge_condition`: 0 searches upward from displacement 0, 1 downward.
    let up = base < target;
    let mut best_bits = if up { f64::INFINITY } else { f64::NEG_INFINITY };
    if up {
        tol_max = 0.0;
        min_beta = 0;
        min_bits = base;
    } else {
        tol_min = 0.0;
        max_beta = 0;
        max_bits = base;
        // The reference widens the window at its lowest CTC rate.
        if (target - 0.12).abs() < 1e-12 {
            tol_max -= 0.03;
        }
    }
    let span = f64::from(max_beta - min_beta);
    let now = if (max_bits - min_bits).abs() < f64::EPSILON || min_bits <= 0.0 || max_bits <= 0.0 {
        0
    } else {
        (span * (libm::log(target) - libm::log(min_bits))
            / (libm::log(max_bits) - libm::log(min_bits))
            + f64::from(min_beta)) as i32
    };
    // The reference bisects `[now - 100, now + 100]` unclamped; `clamp_window` restores this
    // port's clipping to `range`, which only removes trials that would code the same stream
    // anyway (the encoder clips the displacement it is given).
    let (mut lo, mut hi) = if clamp_window {
        ((now - 100).max(range.0), (now + 100).min(range.1))
    } else {
        (now - 100, now + 100)
    };
    let (mut best, mut last) = (
        None,
        if clamp_window {
            now.clamp(range.0, range.1)
        } else {
            now
        },
    );
    while lo <= hi {
        let beta = lo + (hi - lo + 1) / 2;
        let bits = bpp_at(beta)?;
        let mismatch = (bits - target) / target;
        let inside = mismatch <= tol_max && mismatch >= tol_min;
        let search = if up {
            if inside && bits < best_bits {
                best = Some(beta);
                best_bits = bits;
            }
            (bits - target * (1.0 + tol_min)) / (target * (1.0 + tol_min))
        } else {
            if inside && bits > best_bits {
                best = Some(beta);
                best_bits = bits;
            }
            (bits - target * (1.0 + tol_max)) / (target * (1.0 + tol_max))
        };
        if search > 0.0 {
            hi = beta - 1;
        } else {
            lo = beta + 1;
        }
        last = beta;
    }
    Ok(best.unwrap_or(last))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    /// A monotone toy rate curve: `bpp = exp(beta / 400)`. The search must land inside the
    /// tolerance and never repeat a displacement.
    fn run(target: f64) -> (i32, f64, usize) {
        let mut seen: BTreeMap<i32, f64> = BTreeMap::new();
        let range = BDL_SEARCH_RANGE[1];
        let beta = search_beta(target, range, |b| {
            let v = libm::exp(f64::from(b.clamp(range.0, range.1)) / 400.0);
            assert!(seen.insert(b, v).is_none(), "displacement {b} tried twice");
            Ok(v)
        })
        .unwrap();
        let bpp = libm::exp(f64::from(beta.clamp(range.0, range.1)) / 400.0);
        (beta, bpp, seen.len())
    }

    /// `exp(beta / 400)` over model 1's range reaches 0.330 .. 1.911.
    #[test]
    fn hits_the_target_within_tolerance() {
        for &target in &[0.4, 0.6, 0.9, 1.2, 1.8] {
            let (beta, bpp, trials) = run(target);
            let mismatch = (bpp - target) / target;
            assert!(
                mismatch.abs() <= 0.011,
                "target {target}: beta {beta} gives {bpp} ({mismatch})"
            );
            assert!(trials < 14, "target {target}: {trials} trials");
        }
    }

    /// A target the model cannot reach lands on the end of its range, not on an error.
    #[test]
    fn an_unreachable_target_saturates() {
        let (beta, bpp, _) = run(0.25);
        assert_eq!(beta, BDL_SEARCH_RANGE[1].0);
        assert!(bpp > 0.25);
        let (beta, bpp, _) = run(3.0);
        assert_eq!(beta, BDL_SEARCH_RANGE[1].1);
        assert!(bpp < 3.0);
    }

    #[test]
    fn a_target_already_met_costs_one_trial() {
        let mut n = 0;
        let beta = search_beta(1.0, BDL_SEARCH_RANGE[2], |_| {
            n += 1;
            Ok(1.0)
        })
        .unwrap();
        assert_eq!((beta, n), (0, 1));
    }
}
