//! me-tANS table construction.
//!
//! Ports `ref/src/codec/entropy_coding/lib_wrappers/mans/utils.py` (`get_cdf_matrix`,
//! `get_encode_transitions`, `get_state_maps`, `get_decode_transitions`) and the table
//! assembly in `ECLibMans.init_quant_params`. The reference computes bit counts through
//! floating-point `log2`; everything here is integer arithmetic. `tests::tables_match_reference`
//! pins the result to hashes of the reference-built tables.
//!
//! The coder is a tANS with 8 mass bits. The state is kept as `x - 256` in a `u8`
//! (`x` in `[256, 512)`). Each symbol owns two state ranges, one in the lower half of the state
//! space and one in the upper half (`cdf_first` / `cdf_second` below).

use alloc::boxed::Box;

use super::pdf_tables::{BOUND_TABLE_R, NUM_DISTRIBUTIONS_R, PDF_R_FLAT, PDF_R_OFFSETS};

/// Probability mass bits.
pub(crate) const MASS_BITS: u32 = 8;
/// Number of states.
pub(crate) const NUM_STATES: usize = 1 << MASS_BITS;
/// Largest regular `z` symbol plus one; symbol `MAX_Z` is the escape.
pub const MAX_Z: usize = 63;

/// Number of bits to shift in (decode) / out (encode) for a `state_next` or pmf value `v >= 1`:
/// `8 - floor(log2(v))`, clamped at 0. The reference's `get_clz`.
#[inline]
fn renorm_bits(v: u32) -> u32 {
    debug_assert!(v >= 1);
    MASS_BITS.saturating_sub(31 - v.leading_zeros())
}

/// Symbol value at PDF column `col`: `0, -1, 1, -2, 2, ...` (`get_sequence(256) - 128`).
#[inline]
fn column_symbol(col: usize) -> i32 {
    if col.is_multiple_of(2) {
        (col / 2) as i32
    } else {
        -((col / 2) as i32) - 1
    }
}

/// PDF column of symbol `v` in `[-128, 127]` (`get_inverse_sequence`).
#[cfg(test)]
fn symbol_column(v: i32) -> usize {
    if v >= 0 {
        (2 * v) as usize
    } else {
        (-2 * v - 1) as usize
    }
}

/// Tables for the residual (`sgm`) coder: 32 distributions selected by the sigma index.
pub(crate) struct ResidualTables {
    /// `decode[d][state] = nbits << 24 | next_state_base << 16 | (symbol as u16)`.
    pub decode: Box<[[u32; NUM_STATES]; NUM_DISTRIBUTIONS_R]>,
    /// `encode[d][symbol + 128] = delta_bits << 16 | adder`.
    pub encode: Box<[[u32; NUM_STATES]; NUM_DISTRIBUTIONS_R]>,
    /// `state_map[d][slot]`: cumulative slot number to state.
    pub state_map: Box<[[u8; NUM_STATES]; NUM_DISTRIBUTIONS_R]>,
    /// Escape bound per distribution.
    pub bounds: [u8; NUM_DISTRIBUTIONS_R],
}

impl ResidualTables {
    pub fn build() -> Self {
        let mut decode = Box::new([[0u32; NUM_STATES]; NUM_DISTRIBUTIONS_R]);
        let mut encode = Box::new([[0u32; NUM_STATES]; NUM_DISTRIBUTIONS_R]);
        let mut state_map = Box::new([[0u8; NUM_STATES]; NUM_DISTRIBUTIONS_R]);

        for d in 0..NUM_DISTRIBUTIONS_R {
            // pmf over PDF columns; one unit of mass for the escape symbol `-bound`.
            let mut pmf = [0u32; NUM_STATES];
            let row = &PDF_R_FLAT[PDF_R_OFFSETS[d] as usize..PDF_R_OFFSETS[d + 1] as usize];
            for (p, &m) in pmf.iter_mut().zip(row) {
                *p = m as u32;
            }
            pmf[BOUND_TABLE_R[d] as usize * 2 - 1] = 1;

            let mut cdf = [0u32; NUM_STATES + 1];
            for c in 0..NUM_STATES {
                cdf[c + 1] = cdf[c] + pmf[c];
            }
            debug_assert_eq!(cdf[NUM_STATES], NUM_STATES as u32);

            let first = |c: u32| c - (c >> 1);
            let second = |c: u32| (c >> 1) + 128;

            for col in 0..NUM_STATES {
                let p = pmf[col];
                // Encode transition, indexed by symbol + 128. Columns with zero mass keep the
                // reference's filler value (never used by a valid encoder).
                let nb = if p == 0 {
                    MASS_BITS + 2
                } else {
                    renorm_bits(p)
                };
                let delta_bits = ((nb + 1) << MASS_BITS) - (p << nb);
                let adder = (cdf[col].wrapping_sub(p)) & 0xFF;
                let sym = column_symbol(col);
                encode[d][(sym + 128) as usize] = (delta_bits << 16) | adder;

                if p == 0 {
                    continue;
                }
                let (f0, f1) = (first(cdf[col]), first(cdf[col + 1]));
                let (s0, s1) = (second(cdf[col]), second(cdf[col + 1]));
                let n_first = f1 - f0;
                let sym16 = (sym as i16 as u16) as u32;
                let entry = |state_next: u32| -> u32 {
                    let nbits = renorm_bits(state_next);
                    let base = (state_next << nbits) ^ NUM_STATES as u32;
                    (nbits << 24) | (base << 16) | sym16
                };
                for (k, s) in (f0..f1).enumerate() {
                    decode[d][s as usize] = entry(p + k as u32);
                    state_map[d][(cdf[col] + k as u32) as usize] = s as u8;
                }
                for (k, s) in (s0..s1).enumerate() {
                    decode[d][s as usize] = entry(p + n_first + k as u32);
                    state_map[d][(cdf[col] + n_first + k as u32) as usize] = s as u8;
                }
            }
        }
        Self {
            decode,
            encode,
            state_map,
            bounds: BOUND_TABLE_R,
        }
    }
}

/// `transitionsZPreprocess` from the reference decoder: for `k` in `1..512`,
/// `((k << nbits) ^ 256) | nbits << 8` with `nbits = 8 - floor(log2(k))` (0 for `k >= 256`).
pub(crate) fn z_decode_preprocess() -> [u16; 512] {
    let mut t = [0u16; 512];
    for (k, e) in t.iter_mut().enumerate().skip(1) {
        let nbits = renorm_bits(k as u32);
        *e = ((((k as u32) << nbits) ^ 256) | (nbits << 8)) as u16;
    }
    t
}

/// `deltaBits` from the reference encoder: for pmf `p` in `1..256`,
/// `((nbits + 1) << 8) - (p << nbits)`... expressed as the reference does: `(nb << 8) - (p << nb) + 256`
/// with `nb = 8 - floor(log2(p))`.
pub(crate) fn z_encode_delta_bits() -> [u16; 256] {
    let mut t = [0u16; 256];
    for (p, e) in t.iter_mut().enumerate().skip(1) {
        let nb = renorm_bits(p as u32);
        *e = ((nb << 8) + 256 - ((p as u32) << nb)) as u16;
    }
    t
}

/// Normalise a row of integer frequencies to the 8-bit CDF the `z` coder uses
/// (`CustomProbWrapper.normalize_z`): `cdf[i] = (cumsum[i] * 255 + total / 2) / total`.
///
/// The last entry is always 255; state 255 is reserved for the escape symbol.
#[allow(dead_code)] // wired in by the z-substream decoder
pub(crate) fn normalize_z_cdf(freqs: &[i32; MAX_Z]) -> [u8; MAX_Z] {
    let total: i64 = freqs.iter().map(|&f| f as i64).sum();
    let mut out = [0u8; MAX_Z];
    let mut acc = 0i64;
    for (o, &f) in out.iter_mut().zip(freqs) {
        acc += f as i64;
        // numpy floor division on int64; all operands are non-negative for valid tables.
        *o = ((acc * 255 + (total >> 1)).div_euclid(total.max(1))) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a64(bytes: impl Iterator<Item = u8>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Hashes are FNV-1a 64 over the little-endian bytes of the tables the reference builds
    /// (`get_encode_transitions`, `get_state_maps`, `get_decode_transitions` on `pdf_r`),
    /// computed with the reference's own Python at upstream commit b9e573f.
    #[test]
    fn tables_match_reference() {
        let t = ResidualTables::build();
        let enc = fnv1a64(t.encode.iter().flatten().flat_map(|v| v.to_le_bytes()));
        let dec = fnv1a64(t.decode.iter().flatten().flat_map(|v| v.to_le_bytes()));
        let sm = fnv1a64(t.state_map.iter().flatten().copied());
        assert_eq!(dec, 0xae57_8f60_1a57_39ff, "decode transitions");
        assert_eq!(sm, 0x175e_35e0_eef4_7ad9, "state maps");
        assert_eq!(enc, 0x81b7_251e_16cc_1da7, "encode transitions");
    }

    #[test]
    fn symbol_column_roundtrip() {
        for v in -128..128 {
            assert_eq!(column_symbol(symbol_column(v)), v);
        }
        assert_eq!(
            (0..6).map(column_symbol).collect::<alloc::vec::Vec<_>>(),
            [0, -1, 1, -2, 2, -3]
        );
    }

    #[test]
    fn z_preprocess_matches_reference_loop() {
        // The reference computes nBits with a running counter from 511 down to 1.
        let mut want = [0u16; 512];
        let mut nbits = 0u32;
        for i in (1..512u32).rev() {
            if (i << nbits) < 256 {
                nbits += 1;
            }
            want[i as usize] = (((i << nbits) ^ 256) | (nbits << 8)) as u16;
        }
        assert_eq!(z_decode_preprocess(), want);

        let mut want = [0u16; 256];
        let mut nbits = 1u32;
        for i in (1..256u32).rev() {
            if (i << nbits) < 256 {
                nbits += 1;
            }
            want[i as usize] = ((nbits << 8) + 256 - (i << nbits)) as u16;
        }
        assert_eq!(z_encode_delta_bits(), want);
    }
}
