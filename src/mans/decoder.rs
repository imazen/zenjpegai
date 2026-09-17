//! me-tANS decoder.
//!
//! Ports `ref/src/codec/entropy_coding/cpp_exts/mans/decompressor.{h,cpp}`.
//!
//! ANS is last-in-first-out, so the encoder writes the payload front to back and the decoder
//! consumes it back to front: every thread's byte range ends with a sentinel `1` bit above the
//! final coder state(s), and bits are pulled from a 64-bit window that slides towards the start
//! of the range.
//!
//! Two layouts exist and they are not interchangeable:
//!
//! * one thread: two interleaved coder states (`state1`, `state2`) alternate symbol by symbol;
//! * `n > 1` threads: thread `i` owns symbols `i, i + n, i + 2n, ...` and uses a single state.
//!
//! Symbols are processed in groups of four with one window refill per group; a call whose length
//! is not a multiple of the group size takes a differently ordered tail path. The grouping is
//! therefore part of the bitstream format, and callers must split their data into calls exactly
//! as the reference does.

use alloc::vec::Vec;

use super::tables::{MAX_Z, NUM_STATES, ResidualTables, z_decode_preprocess};
use crate::error::{Error, Result};

/// Bytes of zero padding in front of the payload so the 64-bit window can always be loaded.
const FRONT_PAD: usize = 8;

/// Backward bit window over one thread's byte range (`BitStreamDecode` in the reference).
#[derive(Clone, Debug)]
struct Stream {
    /// Index into the padded buffer of the window's first byte.
    ptr_end: usize,
    head: u64,
    /// Number of unread bits in `head` (they are its low bits).
    bit_pos: u32,
    state1: u8,
    state2: u8,
    /// Set when the window was asked to slide in front of the buffer: the stream is corrupt.
    overrun: bool,
}

#[inline(always)]
fn load64(buf: &[u8], pos: usize) -> u64 {
    // One bounds check per refill; `pos + 8 <= buf.len()` holds for every position a valid
    // stream can reach, and corrupt streams are clamped by `Stream::flush`.
    match buf.get(pos..pos + 8) {
        Some(b) => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
        None => 0,
    }
}

impl Stream {
    /// `initBackward` + `initWithState(s)`: `end` is the padded-buffer offset one past the
    /// thread's last byte.
    fn open(buf: &[u8], end: usize, two_states: bool) -> Result<Self> {
        if end < FRONT_PAD || end > buf.len() {
            return Err(Error::InvalidData("ANS thread range out of bounds"));
        }
        let ptr_end = end - FRONT_PAD;
        let mut head = load64(buf, ptr_end);
        if head == 0 {
            return Err(Error::InvalidData("ANS stream has no terminator bit"));
        }
        let bit_pos = 63 - head.leading_zeros();
        head ^= 1u64 << bit_pos;
        let need = if two_states { 16 } else { 8 };
        if bit_pos < need {
            return Err(Error::InvalidData(
                "ANS stream too short for its initial state",
            ));
        }
        let mut s = Stream {
            ptr_end,
            head,
            bit_pos,
            state1: 0,
            state2: 0,
            overrun: false,
        };
        s.state1 = s.read(8) as u8;
        if two_states {
            s.state2 = s.read(8) as u8;
        }
        s.flush(buf);
        Ok(s)
    }

    /// Pull `step` bits. Callers never request more than the window holds between refills
    /// (at most 56 bits); the saturating subtraction only keeps corrupt input panic-free.
    #[inline(always)]
    fn read(&mut self, step: u32) -> u64 {
        self.bit_pos = self.bit_pos.saturating_sub(step);
        let result = self.head >> (self.bit_pos & 63);
        self.head ^= result << (self.bit_pos & 63);
        result
    }

    /// Slide the window so that it holds at least 56 unread bits again.
    #[inline(always)]
    fn flush(&mut self, buf: &[u8]) {
        let consumed = (7 ^ (self.bit_pos >> 3)) as usize;
        match self.ptr_end.checked_sub(consumed) {
            Some(p) => self.ptr_end = p,
            None => {
                self.ptr_end = 0;
                self.overrun = true;
            }
        }
        let keep = 56 + (self.bit_pos & 7);
        self.head = load64(buf, self.ptr_end) & ((1u64 << keep) - 1);
        self.bit_pos |= 56;
    }
}

/// Decoder over one entropy-coded payload (one substream, or one region of one).
pub struct AnsDecoder<'t> {
    tables: &'t ResidualTables,
    buf: Vec<u8>,
    streams: Vec<Stream>,
    z_preprocess: [u16; 512],
}

impl<'t> AnsDecoder<'t> {
    /// `threads` are the byte ranges from [`crate::container::split_threads`], in order; they
    /// must be consecutive slices of one payload (they are copied into one padded buffer).
    pub(crate) fn new(tables: &'t ResidualTables, threads: &[&[u8]]) -> Result<Self> {
        if threads.is_empty() || threads.len() > 16 {
            return Err(Error::InvalidData("thread count out of range"));
        }
        let total: usize = threads.iter().map(|t| t.len()).sum();
        let mut buf = Vec::with_capacity(FRONT_PAD + total);
        buf.resize(FRONT_PAD, 0);
        let mut ends = Vec::with_capacity(threads.len());
        for t in threads {
            buf.extend_from_slice(t);
            ends.push(buf.len());
        }
        let two_states = threads.len() == 1;
        let streams = ends
            .iter()
            .map(|&end| Stream::open(&buf, end, two_states))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tables,
            buf,
            streams,
            z_preprocess: z_decode_preprocess(),
        })
    }

    /// Number of ANS threads in this payload.
    pub fn num_threads(&self) -> usize {
        self.streams.len()
    }

    /// True if any thread tried to read in front of its data. Valid streams never do.
    pub fn overrun(&self) -> bool {
        self.streams.iter().any(|s| s.overrun)
    }

    /// Decode residual symbols (`decode_sgm`).
    ///
    /// `sigma_idx[i]` selects the distribution of symbol `i` (values above 31 are clamped, see
    /// `PORTING.md`), `mask[i] == false` means symbol `i` is skipped (not in the stream) and
    /// `out[i]` is left untouched. All three slices have the same length, and that length is part
    /// of the format: split calls exactly like the reference.
    pub fn decode_residual(
        &mut self,
        sigma_idx: &[u8],
        mask: &[bool],
        out: &mut [i16],
    ) -> Result<()> {
        if sigma_idx.len() != out.len() || mask.len() != out.len() {
            return Err(Error::InvalidArgument(
                "decode_residual: slice lengths differ",
            ));
        }
        if self.streams.len() == 1 {
            let mut s = self.streams[0].clone();
            decode_residual_single(self.tables, &self.buf, &mut s, sigma_idx, mask, out);
            self.streams[0] = s;
        } else {
            let n = self.streams.len();
            for (i, s) in self.streams.iter_mut().enumerate() {
                decode_residual_strided(self.tables, &self.buf, s, i, n, sigma_idx, mask, out);
            }
        }
        if self.overrun() {
            return Err(Error::InvalidData("ANS stream exhausted"));
        }
        Ok(())
    }

    /// Decode hyper-latent symbols (`decode_factorize`): `cdfs` holds one 63-entry 8-bit CDF per
    /// channel, `out` holds `channels * size_per_channel` symbols in `[0, 63]`, channel-major.
    pub fn decode_z(
        &mut self,
        cdfs: &[[u8; MAX_Z]],
        size_per_channel: usize,
        out: &mut [u8],
    ) -> Result<()> {
        if out.len()
            != cdfs
                .len()
                .checked_mul(size_per_channel)
                .ok_or(Error::InvalidArgument("decode_z: size overflow"))?
        {
            return Err(Error::InvalidArgument("decode_z: output length mismatch"));
        }
        if size_per_channel == 0 {
            return Ok(());
        }
        let n = self.streams.len();
        for (cdf, row) in cdfs.iter().zip(out.chunks_exact_mut(size_per_channel)) {
            let transitions = z_transitions(&self.z_preprocess, cdf)?;
            if n == 1 {
                let mut s = self.streams[0].clone();
                decode_z_row_single(&transitions, &self.buf, &mut s, row);
                self.streams[0] = s;
            } else {
                for (i, s) in self.streams.iter_mut().enumerate() {
                    decode_z_row_strided(&transitions, &self.buf, s, i, n, row);
                }
            }
        }
        if self.overrun() {
            return Err(Error::InvalidData("ANS stream exhausted"));
        }
        Ok(())
    }
}

/// `setZDistributions` (decoder side): state → `nbits << 24 | next_state_base << 16 | symbol`.
fn z_transitions(preprocess: &[u16; 512], cdf: &[u8; MAX_Z]) -> Result<[u32; NUM_STATES]> {
    let mut t = [0u32; NUM_STATES];
    let mut curr = 0usize;
    for (sym, &c) in cdf.iter().enumerate() {
        let c = c as usize;
        if c < curr {
            return Err(Error::InvalidData("z CDF is not monotonic"));
        }
        let pmf = c - curr;
        for (k, e) in t[curr..c].iter_mut().enumerate() {
            *e = ((preprocess[pmf + k] as u32) << 16) | sym as u32;
        }
        curr = c;
    }
    t[NUM_STATES - 1] = ((preprocess[1] as u32) << 16) | MAX_Z as u32;
    Ok(t)
}

#[inline(always)]
fn y_inbound(tables: &ResidualTables, s: &mut Stream, second: bool, d: u8, out: &mut i16) {
    let state = if second { s.state2 } else { s.state1 };
    let transition = tables.decode[(d & 31) as usize][state as usize];
    let next = ((transition >> 16) as u8) | (s.read(transition >> 24) as u8);
    if second {
        s.state2 = next;
    } else {
        s.state1 = next;
    }
    *out = transition as u16 as i16;
}

#[inline(always)]
fn y_outbound(tables: &ResidualTables, buf: &[u8], s: &mut Stream, d: u8, out: &mut i16) {
    let bound = tables.bounds[(d & 31) as usize] as i32;
    if *out as i32 + bound == 0 {
        let nbits = if s.read(1) != 0 { 3 } else { 16 };
        let v = s.read(nbits) as i32;
        let sign = v & 1;
        *out = (((v >> 1) + (bound - sign)) ^ (-sign)) as i16;
        s.flush(buf);
    }
}

/// `decodeSingleThread`.
fn decode_residual_single(
    tables: &ResidualTables,
    buf: &[u8],
    s: &mut Stream,
    sigma: &[u8],
    mask: &[bool],
    out: &mut [i16],
) {
    let len = out.len();
    let mut index = 0usize;
    while index + 3 < len {
        let (sg, mk, o) = (
            &sigma[index..index + 4],
            &mask[index..index + 4],
            &mut out[index..index + 4],
        );
        for j in 0..4 {
            if mk[j] {
                y_inbound(tables, s, j & 1 == 1, sg[j], &mut o[j]);
            }
        }
        s.flush(buf);
        for j in 0..4 {
            if mk[j] {
                y_outbound(tables, buf, s, sg[j], &mut o[j]);
            }
        }
        index += 4;
    }
    if len & 2 != 0 {
        for j in 0..2 {
            if mask[index + j] {
                y_inbound(tables, s, j == 1, sigma[index + j], &mut out[index + j]);
                y_outbound(tables, buf, s, sigma[index + j], &mut out[index + j]);
            }
        }
        index += 2;
        s.flush(buf);
    }
    if len & 1 != 0 {
        if mask[index] {
            y_inbound(tables, s, false, sigma[index], &mut out[index]);
            y_outbound(tables, buf, s, sigma[index], &mut out[index]);
        }
        s.flush(buf);
    }
}

/// One thread of `decodeMultiThread`: symbols `first, first + n, ...`, single state.
#[allow(clippy::too_many_arguments)]
fn decode_residual_strided(
    tables: &ResidualTables,
    buf: &[u8],
    s: &mut Stream,
    first: usize,
    n: usize,
    sigma: &[u8],
    mask: &[bool],
    out: &mut [i16],
) {
    let len = out.len();
    let thres = len / (n << 2) * (n << 2);
    let mut index = first;
    while index < thres {
        for j in 0..4 {
            let i = index + n * j;
            if mask[i] {
                y_inbound(tables, s, false, sigma[i], &mut out[i]);
            }
        }
        s.flush(buf);
        for j in 0..4 {
            let i = index + n * j;
            if mask[i] {
                y_outbound(tables, buf, s, sigma[i], &mut out[i]);
            }
        }
        index += n * 4;
    }
    while index < len {
        if mask[index] {
            y_inbound(tables, s, false, sigma[index], &mut out[index]);
            y_outbound(tables, buf, s, sigma[index], &mut out[index]);
        }
        index += n;
        s.flush(buf);
    }
}

#[inline(always)]
fn z_inbound(transitions: &[u32; NUM_STATES], s: &mut Stream, second: bool, out: &mut u8) {
    let state = if second { s.state2 } else { s.state1 };
    let transition = transitions[state as usize];
    let next = ((transition >> 16) as u8) | (s.read(transition >> 24) as u8);
    if second {
        s.state2 = next;
    } else {
        s.state1 = next;
    }
    *out = transition as u8;
}

#[inline(always)]
fn z_outbound(s: &mut Stream, out: &mut u8) {
    if *out as usize == MAX_Z {
        *out = s.read(6) as u8;
    }
}

/// `decodeRowSingleThread`.
fn decode_z_row_single(
    transitions: &[u32; NUM_STATES],
    buf: &[u8],
    s: &mut Stream,
    row: &mut [u8],
) {
    let len = row.len();
    let mut index = 0usize;
    while index + 3 < len {
        let o = &mut row[index..index + 4];
        for j in 0..4 {
            z_inbound(transitions, s, j & 1 == 1, &mut o[j]);
        }
        for v in o.iter_mut() {
            z_outbound(s, v);
        }
        s.flush(buf);
        index += 4;
    }
    if len & 2 != 0 {
        z_inbound(transitions, s, false, &mut row[index]);
        z_outbound(s, &mut row[index]);
        z_inbound(transitions, s, true, &mut row[index + 1]);
        z_outbound(s, &mut row[index + 1]);
        index += 2;
    }
    if len & 1 != 0 {
        z_inbound(transitions, s, false, &mut row[index]);
        z_outbound(s, &mut row[index]);
    }
    s.flush(buf);
}

/// One thread of `decodeRowMultiThread`.
fn decode_z_row_strided(
    transitions: &[u32; NUM_STATES],
    buf: &[u8],
    s: &mut Stream,
    first: usize,
    n: usize,
    row: &mut [u8],
) {
    let len = row.len();
    let thres = len / (n << 2) * (n << 2);
    let mut index = first;
    while index < thres {
        for j in 0..4 {
            z_inbound(transitions, s, false, &mut row[index + n * j]);
        }
        for j in 0..4 {
            z_outbound(s, &mut row[index + n * j]);
        }
        s.flush(buf);
        index += n * 4;
    }
    while index < len {
        z_inbound(transitions, s, false, &mut row[index]);
        z_outbound(s, &mut row[index]);
        index += n;
    }
    s.flush(buf);
}
