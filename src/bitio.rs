//! MSB-first bit reader/writer and the Exp-Golomb binarizations used by the container.
//!
//! Ports `ref/src/codec/entropy_coding/bits_coders.py` (`StreamBitReader`, `StreamBitWriter`),
//! `binarizers.py` (`Binarizers`) and the C++ `EcLibDirect` bit IO used for header substreams
//! (`cpp_exts/direct/ec_lib_direct.cpp`). All three are the same thing: big-endian,
//! most-significant-bit first.

use alloc::vec::Vec;

use crate::error::{Error, Result};

/// Number of bits the reference uses for a value bounded by `max_symbol_value`:
/// `ceil(log2(max + 1))`, computed in integers (the reference uses floating point).
#[inline]
pub(crate) fn bits_for_max(max_symbol_value: u32) -> u32 {
    // ceil(log2(m + 1)) == number of bits needed to represent m (0 for m == 0).
    32 - max_symbol_value.leading_zeros()
}

/// MSB-first bit reader over a byte slice.
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bit position from the start of the buffer.
    #[inline]
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Bits not read yet.
    #[inline]
    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    /// Number of whole bytes touched so far (a partially read byte counts).
    #[inline]
    pub fn bytes_consumed(&self) -> usize {
        self.pos.div_ceil(8)
    }

    /// Bytes after the last touched byte.
    #[inline]
    pub fn remaining_bytes(&self) -> &'a [u8] {
        &self.data[self.bytes_consumed().min(self.data.len())..]
    }

    /// Read `n` bits (`n <= 32`), MSB first.
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        let end = self.pos + n as usize;
        if end > self.data.len() * 8 {
            return Err(Error::UnexpectedEof);
        }
        let mut v: u64 = 0;
        let mut pos = self.pos;
        let mut left = n;
        while left > 0 {
            let byte = self.data[pos >> 3] as u64;
            let bit_off = (pos & 7) as u32;
            let avail = 8 - bit_off;
            let take = avail.min(left);
            let chunk = (byte >> (avail - take)) & ((1u64 << take) - 1);
            v = (v << take) | chunk;
            pos += take as usize;
            left -= take;
        }
        self.pos = end;
        Ok(v as u32)
    }

    #[inline]
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read a value coded with `ceil(log2(max + 1))` bits.
    #[inline]
    pub fn read_bounded(&mut self, max_symbol_value: u32) -> Result<u32> {
        self.read_bits(bits_for_max(max_symbol_value))
    }

    /// Unsigned Exp-Golomb, order 0: unary prefix of zeros terminated by a one, then the suffix.
    pub fn read_ue(&mut self) -> Result<u64> {
        let mut k = 0u32;
        while !self.read_bit()? {
            k += 1;
            if k > 32 {
                return Err(Error::InvalidData("exp-golomb prefix longer than 32 bits"));
            }
        }
        let suffix = self.read_bits(k)? as u64;
        Ok((1u64 << k) + suffix - 1)
    }

    /// Signed Exp-Golomb as the reference defines it: `value = 2|x| - (x > 0)`.
    pub fn read_se(&mut self) -> Result<i64> {
        let v = self.read_ue()?;
        let mag = v.div_ceil(2) as i64;
        Ok(if v & 1 == 0 { -mag } else { mag })
    }

    /// Skip to the next byte boundary.
    #[inline]
    pub fn align(&mut self) {
        self.pos = self.bytes_consumed() * 8;
    }
}

/// MSB-first bit writer into a growable buffer.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bits written so far.
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.out.len() * 8 + self.nbits as usize
    }

    /// Write the low `n` bits of `value` (`n <= 32`), MSB first.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        debug_assert!(n == 32 || value >> n == 0, "value does not fit in {n} bits");
        let mut left = n;
        while left > 0 {
            let space = 8 - self.nbits;
            let take = space.min(left);
            let chunk = ((value as u64 >> (left - take)) & ((1u64 << take) - 1)) as u8;
            self.cur = if take == 8 {
                chunk
            } else {
                (self.cur << take) | chunk
            };
            self.nbits += take;
            left -= take;
            if self.nbits == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(bit as u32, 1);
    }

    /// Write a value with `ceil(log2(max + 1))` bits.
    #[inline]
    pub fn write_bounded(&mut self, value: u32, max_symbol_value: u32) {
        debug_assert!(value <= max_symbol_value);
        self.write_bits(value, bits_for_max(max_symbol_value));
    }

    /// Unsigned Exp-Golomb, order 0.
    pub fn write_ue(&mut self, value: u64) {
        let v = value + 1;
        let k = 63 - v.leading_zeros();
        for _ in 0..k {
            self.write_bit(false);
        }
        self.write_bit(true);
        let suffix = v ^ (1u64 << k);
        if k > 0 {
            // k <= 32 for every value the container writes (sizes fit in u32).
            debug_assert!(k <= 32);
            self.write_bits(suffix as u32, k);
        }
    }

    /// Signed Exp-Golomb (see [`BitReader::read_se`]).
    pub fn write_se(&mut self, value: i64) {
        let mag = value.unsigned_abs();
        let sign = (value > 0) as u64;
        self.write_ue(2 * mag - sign);
    }

    /// Pad the last byte with zero bits and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits != 0 {
            self.out.push(self.cur << (8 - self.nbits));
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_for_max_matches_reference_formula() {
        // int(ceil(log2(m + 1))) for the values the reference uses.
        for (m, want) in [
            (0u32, 0u32),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (15, 4),
            (16, 5),
            (65535, 16),
            (65535 + 64 - 64, 16),
        ] {
            assert_eq!(bits_for_max(m), want, "max={m}");
        }
    }

    #[test]
    fn ue_roundtrip_and_known_codes() {
        // Known ue(v) codewords: 0 -> "1", 1 -> "010", 2 -> "011", 3 -> "00100".
        let mut w = BitWriter::new();
        for v in [0u64, 1, 2, 3] {
            w.write_ue(v);
        }
        let bytes = w.finish();
        // 1 010 011 00100 -> 1010 0110 0100 (pad) -> 0xA6 0x40
        assert_eq!(bytes, [0xA6, 0x40]);

        let mut w = BitWriter::new();
        let vals: Vec<u64> = (0..2000).chain([65535, 1 << 20, (1 << 31) - 1]).collect();
        for &v in &vals {
            w.write_ue(v);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &v in &vals {
            assert_eq!(r.read_ue().unwrap(), v);
        }
    }

    #[test]
    fn se_roundtrip() {
        let vals: Vec<i64> = (-300..300).chain([-(1 << 20), 1 << 20]).collect();
        let mut w = BitWriter::new();
        for &v in &vals {
            w.write_se(v);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &v in &vals {
            assert_eq!(r.read_se().unwrap(), v, "v={v}");
        }
    }

    #[test]
    fn fixed_bits_roundtrip_and_eof() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0xABCD, 16);
        w.write_bits(0xFFFF_FFFF, 32);
        w.write_bits(0, 1);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(16).unwrap(), 0xABCD);
        assert_eq!(r.read_bits(32).unwrap(), 0xFFFF_FFFF);
        assert_eq!(r.read_bits(1).unwrap(), 0);
        assert_eq!(r.bytes_consumed(), bytes.len());
        assert!(matches!(r.read_bits(8), Err(Error::UnexpectedEof)));
    }
}
