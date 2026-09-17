//! The reference's sigma index → linear scale table (`common/log2lin.py::Log2LinConvertion`).
//!
//! Upstream ships 4352 literal integers. They are exactly
//! `round(2^17 * exp(i * ln(100 / 0.11) / 4352 + ln 0.11))` in double precision (checked against
//! the upstream file: the unit test pins the table's hash), so the table is computed instead of
//! stored.

use alloc::vec::Vec;

/// Number of entries; also the exclusive upper bound of a "likely" index.
pub const LEN: usize = 4352;

/// `Log2LinConvertion.table`.
pub fn table() -> Vec<u32> {
    let ln_min = libm::log(0.11);
    let step = (libm::log(100.0) - ln_min) / LEN as f64;
    (0..LEN)
        .map(|i| libm::rint(131072.0 * libm::exp(i as f64 * step + ln_min)) as u32)
        .collect()
}

/// `idx2log_table`: index of the first entry `>= value`, clamped to the last index
/// (`torch.searchsorted` + `clamp(max = len - 1)`).
pub fn idx2log(table: &[u32], value: u64) -> usize {
    table
        .partition_point(|&t| (t as u64) < value)
        .min(table.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_upstream_literals() {
        let t = table();
        assert_eq!(
            (t.len(), t[0], t[1], t[LEN - 1]),
            (4352, 14418, 14441, 13_086_699)
        );
        // FNV-1a 64 over the little-endian bytes of upstream's literal table.
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for v in &t {
            for b in v.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(h, 0xae24_746d_99d7_1994);
    }

    #[test]
    fn idx2log_is_searchsorted_left() {
        let t = table();
        assert_eq!(idx2log(&t, 0), 0);
        assert_eq!(idx2log(&t, 14418), 0);
        assert_eq!(idx2log(&t, 14419), 1);
        assert_eq!(idx2log(&t, u64::MAX), LEN - 1);
    }
}
