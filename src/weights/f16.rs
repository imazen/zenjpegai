//! IEEE-754 half-precision storage for float weights (`ZJB2` bundles, [`super::packed`]).
//!
//! The conversion is pure integer arithmetic, so it is deterministic on every target (native,
//! wasm, any SIMD tier): `f16` is a *storage* format. [`crate::weights::Checkpoint::f32`]
//! upcasts at load time and every f16 value is exactly representable in f32, so the compute
//! path sees ordinary f32 tensors — identical bits on every tier, same as a `.pth` load whose
//! floats had been pre-rounded. Only the rounding itself loses information.

/// `f32` → f16 bits, round to nearest even. NaN stays NaN (quiet, payload truncated), finite
/// values above the f16 range round to ±∞, values below half the smallest f16 subnormal
/// (≈ 3e-8) flush to ±0.
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;
    if exp == 255 {
        // ±∞ or NaN; a NaN keeps a nonzero (quiet) payload.
        let payload = ((mant >> 13) as u16) | 0x200;
        return sign | 0x7c00 | if mant != 0 { payload.max(1) } else { 0 };
    }
    // Biased f16 exponent of the same value before rounding.
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // f16 subnormal (or zero): keep the top bits of `1.mant` under a 2^-24 quantum.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let dropped = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let mut r = m >> shift;
        if dropped > half || (dropped == half && r & 1 == 1) {
            r += 1;
        }
        // Rounding may carry into 0x400, which is exactly the smallest normal — the layout
        // handles it for free.
        return sign | r as u16;
    }
    // Normal: drop 13 mantissa bits, round to nearest even. A mantissa carry (r = 0x400) must
    // be *added* to the exponent — OR-ing loses it whenever e is odd — which bumps the result to
    // the next binade, or to ±∞ at e = 30.
    let dropped = mant & 0x1fff;
    let mut r = mant >> 13;
    if dropped > 0x1000 || (dropped == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | (((e as u16) << 10) + r as u16)
}

/// f16 bits → `f32`. Exact: every f16 (subnormals, ±0, ±∞, NaN) has an exact f32 image.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        // Subnormal: `mant * 2^-24`, exact (mant < 2^10 and the scale is a power of two).
        0 => sign | ((mant as f32) * f32::from_bits(0x3380_0000)).to_bits(),
        31 => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp as u32 + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// `f32` → `f16` → `f32`: the value the decoder's compute path sees for a stored f16 weight.
pub fn round_trip(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_SUBNORMAL: f32 = f32::from_bits(0x3380_0000); // 2^-24 ≈ 5.96e-8

    #[test]
    fn upcast_examples() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), MIN_SUBNORMAL);
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * MIN_SUBNORMAL);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!(f16_to_f32(0x7fff).is_nan());
    }

    #[test]
    fn every_f16_is_a_round_trip_fixed_point() {
        // Upcast is exact, so an already-f16 value must come through f32 -> f16 unchanged.
        for h in 0u32..0x10000 {
            let h = h as u16;
            let v = f16_to_f32(h);
            if v.is_nan() {
                assert!(f16_to_f32(f32_to_f16(v)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(v), h, "f16 {h:#06x} did not round-trip");
        }
    }

    #[test]
    fn round_to_nearest_even() {
        // Midpoint between 1.0 (0x3c00) and the next f16 (0x3c01): ties to even 0x3c00.
        assert_eq!(f32_to_f16(f32::from_bits(0x3f80_1000)), 0x3c00);
        // Just above the midpoint: rounds up.
        assert_eq!(f32_to_f16(f32::from_bits(0x3f80_1001)), 0x3c01);
        // Midpoint between 0x3c01 (odd) and 0x3c02 (even): ties to 0x3c02.
        assert_eq!(f32_to_f16(f32::from_bits(0x3f80_3000)), 0x3c02);
    }

    #[test]
    fn range_edges() {
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // above the last midpoint: +inf
        assert_eq!(f32_to_f16(-65520.0), 0xfc00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400); // smallest normal
        assert_eq!(f32_to_f16(MIN_SUBNORMAL), 0x0001); // 2^-24
        assert_eq!(f32_to_f16(2.98e-8), 0x0000); // half of it: tie to even zero
        assert_eq!(f32_to_f16(8.9e-8), 0x0001);
        assert_eq!(f32_to_f16(-2.98e-8), 0x8000); // -0
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        // The largest f16 subnormal rounds into the smallest normal at its midpoint.
        assert_eq!(f32_to_f16(6.100_6e-5), 0x0400);
        // Mantissa rounding carries into the exponent — including when the exponent is odd
        // (regression: OR-ing the carry instead of adding it halved the result).
        assert_eq!(f32_to_f16(1.220_521_5e-4), 0x0800); // just below 2^-13
        assert_eq!(f32_to_f16(1.999_999_9), 0x3c00 + (1 << 10)); // rounds up to 2.0
    }

    #[test]
    fn error_bounds_hold() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..2_000_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let v = f32::from_bits((rng >> 16) as u32);
            if v.is_nan() {
                continue;
            }
            let w = round_trip(v);
            assert_eq!(round_trip(w), w, "round trip not idempotent for {v}");
            let a = v.abs();
            if !v.is_finite() {
                assert!(!w.is_finite() || a < 65520.0);
                continue;
            }
            if a >= 65520.0 {
                assert_eq!(w, f32::INFINITY.copysign(v));
            } else if a >= 2f32.powi(-14) {
                // Normal f16 range: at most half an ulp of 2^-11 relative.
                assert!(
                    (w - v).abs() <= a * 4.883e-4 + 1e-9,
                    "{v} -> {w}: relative error too large"
                );
            } else {
                // Subnormal / flush-to-zero: absolute error under the 2^-24 quantum.
                assert!((w - v).abs() <= MIN_SUBNORMAL / 2.0 + 1e-12);
            }
        }
    }
}
