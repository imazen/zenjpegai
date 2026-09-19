//! me-tANS encoder.
//!
//! Ports `ref/src/codec/entropy_coding/cpp_exts/mans/compressor.{h,cpp}`.
//!
//! ANS is last-in-first-out: whatever the decoder reads first must be written last. Callers
//! therefore issue encode calls in the exact reverse of the decoder's call order, and inside a
//! call symbols are walked from the last to the first. See [`super::decoder`] for the layout
//! rules (two interleaved states with one thread, one state per thread otherwise, groups of
//! four symbols per refill).

use alloc::vec::Vec;

use super::tables::{MAX_Z, ResidualTables, z_encode_delta_bits};
use crate::error::{Error, Result};

/// Forward bit accumulator for one thread (`BitStreamEncode` in the reference).
#[derive(Clone, Debug, Default)]
struct Stream {
    out: Vec<u8>,
    head: u64,
    bit_pos: u32,
    state1: u8,
    state2: u8,
    /// `out`'s bytes in the tracked ledger (see [`crate::mem`]).
    charge: crate::mem::Charge,
}

impl Stream {
    #[inline(always)]
    fn write(&mut self, n: u64, step: u32) {
        self.head |= n << self.bit_pos;
        self.bit_pos += step;
        debug_assert!(self.bit_pos < 64);
    }

    /// Move whole bytes from the accumulator to the output.
    #[inline(always)]
    fn flush(&mut self) {
        let nbytes = (self.bit_pos >> 3) as usize;
        self.out
            .extend_from_slice(&self.head.to_le_bytes()[..nbytes]);
        self.charge.resize(crate::mem::vec_bytes(&self.out));
        self.head = if nbytes == 8 {
            0
        } else {
            self.head >> (nbytes * 8)
        };
        self.bit_pos &= 7;
    }

    /// Append the terminator (`1` bit above the final state or states) and return the bytes.
    fn close(mut self, two_states: bool) -> Vec<u8> {
        if two_states {
            self.write(65536 | ((self.state1 as u64) << 8) | self.state2 as u64, 17);
        } else {
            self.write(256 | self.state1 as u64, 9);
        }
        let nbytes = self.bit_pos.div_ceil(8) as usize;
        self.out
            .extend_from_slice(&self.head.to_le_bytes()[..nbytes]);
        self.charge.resize(crate::mem::vec_bytes(&self.out));
        self.out
    }
}

/// `encodeWithTransition`: shift out just enough low state bits that the symbol's slot fits,
/// then move to the slot.
#[inline(always)]
fn encode_with_transition(s: &mut Stream, second: bool, transition: u32) -> u8 {
    let state = if second { s.state2 } else { s.state1 } as u32;
    let nbits = (state + (transition >> 16)) >> 8;
    s.write((state & ((1u32 << nbits) - 1)) as u64, nbits);
    (((state | 256) >> nbits).wrapping_add(transition)) as u8
}

/// Encoder for one entropy-coded payload.
pub struct AnsEncoder<'t> {
    tables: &'t ResidualTables,
    streams: Vec<Stream>,
    z_delta_bits: [u16; 256],
}

impl<'t> AnsEncoder<'t> {
    pub(crate) fn new(tables: &'t ResidualTables, num_threads: usize) -> Result<Self> {
        if num_threads == 0 || num_threads > 16 {
            return Err(Error::InvalidArgument("thread count out of range"));
        }
        Ok(Self {
            tables,
            streams: alloc::vec![Stream::default(); num_threads],
            z_delta_bits: z_encode_delta_bits(),
        })
    }

    /// Encode residual symbols (`encode_sgm`). Mirror of
    /// [`super::decoder::AnsDecoder::decode_residual`]; `values` outside the distribution's
    /// bound are escape-coded.
    pub fn encode_residual(
        &mut self,
        sigma_idx: &[u8],
        mask: &[bool],
        values: &[i16],
    ) -> Result<()> {
        if sigma_idx.len() != values.len() || mask.len() != values.len() {
            return Err(Error::InvalidArgument(
                "encode_residual: slice lengths differ",
            ));
        }
        let n = self.streams.len();
        let tables = self.tables;
        if n == 1 {
            encode_residual_single(tables, &mut self.streams[0], sigma_idx, mask, values);
        } else {
            for (i, s) in self.streams.iter_mut().enumerate() {
                encode_residual_strided(tables, s, i, n, sigma_idx, mask, values);
            }
        }
        Ok(())
    }

    /// Encode hyper-latent symbols (`encode_factorize`): one 63-entry CDF per channel, symbols
    /// in `[0, 63]` channel-major. Symbols with zero mass (and symbol 63) are escape-coded.
    pub fn encode_z(
        &mut self,
        cdfs: &[[u8; MAX_Z]],
        size_per_channel: usize,
        symbols: &[u8],
    ) -> Result<()> {
        if symbols.len()
            != cdfs
                .len()
                .checked_mul(size_per_channel)
                .ok_or(Error::InvalidArgument("encode_z: size overflow"))?
        {
            return Err(Error::InvalidArgument("encode_z: input length mismatch"));
        }
        if symbols.iter().any(|&v| v as usize > MAX_Z) {
            return Err(Error::InvalidArgument("encode_z: symbol out of range"));
        }
        if size_per_channel == 0 {
            return Ok(());
        }
        let n = self.streams.len();
        for (cdf, row) in cdfs
            .iter()
            .zip(symbols.chunks_exact(size_per_channel))
            .rev()
        {
            let transitions = z_transitions(&self.z_delta_bits, cdf)?;
            if n == 1 {
                encode_z_row_single(&transitions, &mut self.streams[0], row);
            } else {
                for (i, s) in self.streams.iter_mut().enumerate() {
                    encode_z_row_strided(&transitions, s, i, n, row);
                }
            }
        }
        Ok(())
    }

    /// Terminate every thread and return their byte ranges in order.
    pub fn finish(self) -> Vec<Vec<u8>> {
        let two_states = self.streams.len() == 1;
        self.streams
            .into_iter()
            .map(|s| s.close(two_states))
            .collect()
    }
}

/// `setZDistributions` (encoder side): symbol → `delta_bits << 16 | adder`, 0 for zero mass.
fn z_transitions(delta_bits: &[u16; 256], cdf: &[u8; MAX_Z]) -> Result<[u32; MAX_Z + 1]> {
    let mut t = [0u32; MAX_Z + 1];
    let mut curr = 0u32;
    for (e, &c) in t.iter_mut().zip(cdf) {
        let c = c as u32;
        if c < curr {
            return Err(Error::InvalidArgument("z CDF is not monotonic"));
        }
        let pmf = c - curr;
        if pmf != 0 {
            *e = ((delta_bits[pmf as usize] as u32) << 16) | (curr.wrapping_sub(pmf) & 0xFF);
        }
        curr = c;
    }
    t[MAX_Z] = ((delta_bits[1] as u32) << 16) | 254;
    Ok(t)
}

/// `encodeYOutbound`: returns the symbol to hand to the ANS state machine.
#[inline(always)]
fn y_outbound(tables: &ResidualTables, s: &mut Stream, d: u8, value: i16) -> i16 {
    let bound = tables.bounds[(d & 31) as usize] as i32;
    let value = value as i32;
    if value >= bound || value <= -bound {
        let sign = value >> 15; // 0 or -1
        let coded = (((value + (sign ^ -bound)) << 1) ^ sign) as u32 as u64;
        if coded >= 8 {
            s.write(coded, 17);
        } else {
            s.write(coded | 8, 4);
        }
        s.flush();
        -bound as i16
    } else {
        value as i16
    }
}

#[inline(always)]
fn y_inbound(tables: &ResidualTables, s: &mut Stream, second: bool, d: u8, symbol: i16) {
    let d = (d & 31) as usize;
    let transition = tables.encode[d][(symbol as i32 + 128) as usize & 255];
    let slot = encode_with_transition(s, second, transition);
    let next = tables.state_map[d][slot as usize];
    if second {
        s.state2 = next;
    } else {
        s.state1 = next;
    }
}

/// `encodeRSingleThread`.
fn encode_residual_single(
    tables: &ResidualTables,
    s: &mut Stream,
    sigma: &[u8],
    mask: &[bool],
    values: &[i16],
) {
    let mut index = values.len();
    if index & 1 != 0 {
        index -= 1;
        if mask[index] {
            let sym = y_outbound(tables, s, sigma[index], values[index]);
            y_inbound(tables, s, false, sigma[index], sym);
        }
        s.flush();
    }
    if index & 2 != 0 {
        index -= 2;
        for j in [1usize, 0] {
            if mask[index + j] {
                let sym = y_outbound(tables, s, sigma[index + j], values[index + j]);
                y_inbound(tables, s, j == 1, sigma[index + j], sym);
            }
        }
        s.flush();
    }
    while index != 0 {
        index -= 4;
        let mut syms = [0i16; 4];
        for j in (0..4).rev() {
            if mask[index + j] {
                syms[j] = y_outbound(tables, s, sigma[index + j], values[index + j]);
            }
        }
        for j in (0..4).rev() {
            if mask[index + j] {
                y_inbound(tables, s, j & 1 == 1, sigma[index + j], syms[j]);
            }
        }
        s.flush();
    }
}

/// One thread of `encodeRMultiThread`.
fn encode_residual_strided(
    tables: &ResidualTables,
    s: &mut Stream,
    first: usize,
    n: usize,
    sigma: &[u8],
    mask: &[bool],
    values: &[i16],
) {
    let len = values.len() as isize;
    let n = n as isize;
    let num_rounds = len / n;
    let num_groups = num_rounds >> 2;
    let round_thres = num_rounds * n;
    let group_thres = (num_groups * n) << 2;
    let first = first as isize;
    let mut index = round_thres + first - if first + round_thres >= len { n } else { 0 };

    while index >= group_thres {
        let i = index as usize;
        if mask[i] {
            let sym = y_outbound(tables, s, sigma[i], values[i]);
            y_inbound(tables, s, false, sigma[i], sym);
            s.flush();
        }
        index -= n;
    }
    while index > 0 {
        let mut syms = [0i16; 4];
        for j in 0..4 {
            let i = (index - n * j as isize) as usize;
            if mask[i] {
                syms[j] = y_outbound(tables, s, sigma[i], values[i]);
            }
        }
        for j in 0..4 {
            let i = (index - n * j as isize) as usize;
            if mask[i] {
                y_inbound(tables, s, false, sigma[i], syms[j]);
            }
        }
        index -= n * 4;
        s.flush();
    }
}

#[inline(always)]
fn z_outbound(transitions: &[u32; MAX_Z + 1], s: &mut Stream, value: u8) -> u8 {
    if transitions[value as usize & MAX_Z] == 0 {
        s.write(value as u64, 6);
        MAX_Z as u8
    } else {
        value
    }
}

#[inline(always)]
fn z_inbound(transitions: &[u32; MAX_Z + 1], s: &mut Stream, second: bool, symbol: u8) {
    let next = encode_with_transition(s, second, transitions[symbol as usize & MAX_Z]);
    if second {
        s.state2 = next;
    } else {
        s.state1 = next;
    }
}

/// `encodeZRowSingleThread`.
fn encode_z_row_single(transitions: &[u32; MAX_Z + 1], s: &mut Stream, row: &[u8]) {
    let mut len = row.len();
    if len & 1 != 0 {
        len -= 1;
        let sym = z_outbound(transitions, s, row[len]);
        z_inbound(transitions, s, false, sym);
    }
    if len & 2 != 0 {
        len -= 1;
        let sym = z_outbound(transitions, s, row[len]);
        z_inbound(transitions, s, true, sym);
        len -= 1;
        let sym = z_outbound(transitions, s, row[len]);
        z_inbound(transitions, s, false, sym);
    }
    s.flush();
    while len != 0 {
        len -= 4;
        let mut syms = [0u8; 4];
        for j in (0..4).rev() {
            syms[j] = z_outbound(transitions, s, row[len + j]);
        }
        for j in (0..4).rev() {
            z_inbound(transitions, s, j & 1 == 1, syms[j]);
        }
        s.flush();
    }
}

/// One thread of `encodeZRowMultiThread`.
fn encode_z_row_strided(
    transitions: &[u32; MAX_Z + 1],
    s: &mut Stream,
    first: usize,
    n: usize,
    row: &[u8],
) {
    let len = row.len() as isize;
    let n = n as isize;
    let num_rounds = len / n;
    let num_groups = num_rounds >> 2;
    let round_thres = num_rounds * n;
    let group_thres = (num_groups * n) << 2;
    let first = first as isize;
    let mut index = round_thres + first - if first + round_thres >= len { n } else { 0 };

    while index >= group_thres {
        let sym = z_outbound(transitions, s, row[index as usize]);
        z_inbound(transitions, s, false, sym);
        index -= n;
    }
    s.flush();
    while index > 0 {
        let mut syms = [0u8; 4];
        for j in 0..4 {
            syms[j] = z_outbound(transitions, s, row[(index - n * j as isize) as usize]);
        }
        for j in 0..4 {
            z_inbound(transitions, s, false, syms[j]);
        }
        index -= n * 4;
        s.flush();
    }
}
