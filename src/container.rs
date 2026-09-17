//! Marker-delimited codestream container.
//!
//! Ports `ref/src/codec/bitstream_structure/` (`layouts_def.py`, `substream.py`,
//! `bitstream_structure.py`, `aemem.py`).
//!
//! ```text
//! SOC | PIH | [TON] | [SOQ] | SOZ | SORP.. | SORS.. | [UDI] | [RDI] | EOC
//! ```
//!
//! Every substream is `marker (2 bytes, big-endian) | size ue(v), byte-aligned | payload`.
//! Entropy-coded substreams may be split into independently decodable *threads*, and the two
//! residual substreams may additionally be split into spatial *regions*. Those splits depend on
//! values signalled in the picture header, so parsing is two-phase: [`Codestream::parse`] splits
//! the file at markers without interpreting payloads, and [`split_regions`] / [`split_threads`]
//! are applied once the header is known.

use alloc::vec::Vec;

use crate::bitio::{BitReader, BitWriter};
use crate::error::{Error, Result};

/// Substream markers (`SubstreamLayouts` in the reference).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Marker {
    /// Start of codestream.
    Soc = 0xFF80,
    /// End of codestream.
    Eoc = 0xFF81,
    /// Picture header.
    Pih = 0xFF82,
    /// Tool header.
    Ton = 0xFF83,
    /// Rendering information.
    Rdi = 0xFF84,
    /// Hyper-latent `z` substream.
    Soz = 0xFF88,
    /// Residual substream, primary (luma) component.
    Sorp = 0xFF89,
    /// Residual substream, secondary (chroma) component.
    Sors = 0xFF8A,
    /// Quality map.
    Soq = 0xFF8B,
    /// User-defined information.
    Udi = 0xFF8C,
}

impl Marker {
    pub fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            0xFF80 => Marker::Soc,
            0xFF81 => Marker::Eoc,
            0xFF82 => Marker::Pih,
            0xFF83 => Marker::Ton,
            0xFF84 => Marker::Rdi,
            0xFF88 => Marker::Soz,
            0xFF89 => Marker::Sorp,
            0xFF8A => Marker::Sors,
            0xFF8B => Marker::Soq,
            0xFF8C => Marker::Udi,
            _ => return None,
        })
    }

    /// Whether the payload is me-tANS coded (as opposed to plain bit fields).
    pub fn is_entropy_coded(self) -> bool {
        matches!(
            self,
            Marker::Soz | Marker::Sorp | Marker::Sors | Marker::Soq
        )
    }

    /// Whether the payload may be split into spatial regions.
    pub fn has_regions(self) -> bool {
        matches!(self, Marker::Sorp | Marker::Sors)
    }

    /// Whether the marker introduces a sized substream (everything except SOC / EOC).
    pub fn has_payload(self) -> bool {
        !matches!(self, Marker::Soc | Marker::Eoc)
    }
}

/// One marker-delimited substream, borrowed from the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Substream<'a> {
    pub marker: Marker,
    pub payload: &'a [u8],
}

/// A codestream split at its markers. Payloads are not interpreted.
#[derive(Clone, Debug, Default)]
pub struct Codestream<'a> {
    pub substreams: Vec<Substream<'a>>,
}

impl<'a> Codestream<'a> {
    /// Split `data` at markers.
    ///
    /// Stricter than the reference reader, which skips two bytes at a time over anything it does
    /// not recognise (and never terminates on a truncated file): here the stream must start
    /// with SOC, contain only known markers, put PIH first, and end with EOC.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let mut pos = 0usize;
        let read_marker = |pos: &mut usize| -> Result<u16> {
            let b = data.get(*pos..*pos + 2).ok_or(Error::UnexpectedEof)?;
            *pos += 2;
            Ok(u16::from_be_bytes([b[0], b[1]]))
        };

        if read_marker(&mut pos)? != Marker::Soc as u16 {
            return Err(Error::InvalidData("codestream does not start with SOC"));
        }
        let mut substreams = Vec::new();
        loop {
            let raw = read_marker(&mut pos)?;
            let marker = Marker::from_u16(raw).ok_or(Error::InvalidData("unknown marker"))?;
            match marker {
                Marker::Eoc => break,
                Marker::Soc => return Err(Error::InvalidData("SOC inside codestream")),
                _ => {}
            }
            let mut r = BitReader::new(&data[pos..]);
            let size = r.read_ue()?;
            pos += r.bytes_consumed();
            let size =
                usize::try_from(size).map_err(|_| Error::InvalidData("substream size overflow"))?;
            let end = pos
                .checked_add(size)
                .ok_or(Error::InvalidData("substream size overflow"))?;
            let payload = data.get(pos..end).ok_or(Error::UnexpectedEof)?;
            pos = end;
            if substreams.is_empty() && marker != Marker::Pih {
                return Err(Error::InvalidData("PIH must be the first substream"));
            }
            substreams.push(Substream { marker, payload });
        }
        if substreams.is_empty() {
            return Err(Error::InvalidData("codestream has no picture header"));
        }
        Ok(Self { substreams })
    }

    /// First substream with the given marker.
    pub fn find(&self, marker: Marker) -> Option<&'a [u8]> {
        self.substreams
            .iter()
            .find(|s| s.marker == marker)
            .map(|s| s.payload)
    }

    /// All substreams with the given marker, in file order.
    pub fn find_all(&self, marker: Marker) -> impl Iterator<Item = &'a [u8]> + '_ {
        self.substreams
            .iter()
            .filter(move |s| s.marker == marker)
            .map(|s| s.payload)
    }
}

/// How the residual of one component is laid out across regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionLayout {
    /// `region_residual_in_its_own_substream_flag == 1`: one substream per region, each starting
    /// with a one-byte region index. Regions may be absent.
    Independent,
    /// `region_residual_in_its_own_substream_flag == 0`: one substream holding `num_regions - 1`
    /// ue(v) sizes (byte-aligned as a group) followed by the region payloads.
    Dependent,
}

/// Split the residual substream(s) of one component into per-region payloads.
///
/// `substreams` are the payloads of every SORP (or SORS) substream in file order. The result
/// has `num_regions` entries; `None` marks a region that is absent from the stream (only
/// possible with [`RegionLayout::Independent`]).
pub fn split_regions<'a>(
    substreams: &[&'a [u8]],
    layout: RegionLayout,
    num_regions: usize,
) -> Result<Vec<Option<&'a [u8]>>> {
    if num_regions == 0 || num_regions > 128 * 128 {
        return Err(Error::InvalidData("region count out of range"));
    }
    let mut out: Vec<Option<&'a [u8]>> = alloc::vec![None; num_regions];
    match layout {
        RegionLayout::Independent => {
            for s in substreams {
                let (&idx, rest) = s.split_first().ok_or(Error::UnexpectedEof)?;
                let slot = out
                    .get_mut(idx as usize)
                    .ok_or(Error::InvalidData("region index out of range"))?;
                if slot.is_some() {
                    return Err(Error::InvalidData("duplicate region substream"));
                }
                *slot = Some(rest);
            }
        }
        RegionLayout::Dependent => {
            let [s] = substreams else {
                return Err(Error::InvalidData(
                    "dependent regions need exactly one residual substream",
                ));
            };
            let mut r = BitReader::new(s);
            let mut sizes = Vec::with_capacity(num_regions);
            let mut total = 0usize;
            for _ in 0..num_regions - 1 {
                let v = usize::try_from(r.read_ue()?)
                    .map_err(|_| Error::InvalidData("region size overflow"))?;
                total = total
                    .checked_add(v)
                    .ok_or(Error::InvalidData("region size overflow"))?;
                sizes.push(v);
            }
            let mut rest = r.remaining_bytes();
            if total > rest.len() {
                return Err(Error::InvalidData("region sizes exceed substream"));
            }
            sizes.push(rest.len() - total);
            for (slot, size) in out.iter_mut().zip(sizes) {
                let (head, tail) = rest.split_at(size);
                *slot = Some(head);
                rest = tail;
            }
        }
    }
    Ok(out)
}

/// Split an entropy-coded payload into its `num_threads` byte ranges.
///
/// Layout: `num_threads - 1` se(v) deltas (byte-aligned as a group), then the thread data.
/// `size_i = mean - delta_i` with `mean = floor(data_len / num_threads)`; the last thread takes
/// the remainder (`AEMemObject` in the reference).
pub fn split_threads(payload: &[u8], num_threads: usize) -> Result<Vec<&[u8]>> {
    if num_threads == 0 || num_threads > 16 {
        return Err(Error::InvalidData("thread count out of range"));
    }
    let mut r = BitReader::new(payload);
    let mut deltas = Vec::with_capacity(num_threads - 1);
    for _ in 0..num_threads - 1 {
        deltas.push(r.read_se()?);
    }
    let mut rest = r.remaining_bytes();
    let mean = (rest.len() / num_threads) as i64;
    let mut out = Vec::with_capacity(num_threads);
    for d in deltas {
        let size = mean
            .checked_sub(d)
            .ok_or(Error::InvalidData("thread size overflow"))?;
        let size = usize::try_from(size).map_err(|_| Error::InvalidData("negative thread size"))?;
        if size > rest.len() {
            return Err(Error::InvalidData("thread sizes exceed substream"));
        }
        let (head, tail) = rest.split_at(size);
        out.push(head);
        rest = tail;
    }
    out.push(rest);
    Ok(out)
}

/// Serialises substreams into a codestream.
#[derive(Default)]
pub struct CodestreamWriter {
    out: Vec<u8>,
    wrote_any: bool,
}

impl CodestreamWriter {
    pub fn new() -> Self {
        let mut out = Vec::new();
        out.extend_from_slice(&(Marker::Soc as u16).to_be_bytes());
        Self {
            out,
            wrote_any: false,
        }
    }

    /// Append one substream. The first one must be PIH.
    pub fn substream(&mut self, marker: Marker, payload: &[u8]) -> Result<()> {
        if !marker.has_payload() {
            return Err(Error::InvalidArgument("SOC/EOC carry no payload"));
        }
        if !self.wrote_any && marker != Marker::Pih {
            return Err(Error::InvalidArgument("PIH must be the first substream"));
        }
        self.wrote_any = true;
        self.out.extend_from_slice(&(marker as u16).to_be_bytes());
        let mut w = BitWriter::new();
        w.write_ue(payload.len() as u64);
        self.out.extend_from_slice(&w.finish());
        self.out.extend_from_slice(payload);
        Ok(())
    }

    /// Append EOC and return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.out
            .extend_from_slice(&(Marker::Eoc as u16).to_be_bytes());
        self.out
    }
}

/// Join per-thread data into one entropy-coded payload (inverse of [`split_threads`]).
pub fn join_threads(threads: &[&[u8]]) -> Vec<u8> {
    let total: usize = threads.iter().map(|t| t.len()).sum();
    let mean = (total / threads.len().max(1)) as i64;
    let mut w = BitWriter::new();
    for t in &threads[..threads.len().saturating_sub(1)] {
        w.write_se(mean - t.len() as i64);
    }
    let mut out = w.finish();
    out.reserve(total);
    for t in threads {
        out.extend_from_slice(t);
    }
    out
}

/// Join per-region payloads for [`RegionLayout::Dependent`] (inverse of [`split_regions`]).
pub fn join_dependent_regions(regions: &[&[u8]]) -> Vec<u8> {
    let mut w = BitWriter::new();
    for r in &regions[..regions.len().saturating_sub(1)] {
        w.write_ue(r.len() as u64);
    }
    let mut out = w.finish();
    for r in regions {
        out.extend_from_slice(r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_roundtrip() {
        let pih = [1u8, 2, 3];
        let soz = alloc::vec![7u8; 300];
        let mut w = CodestreamWriter::new();
        w.substream(Marker::Pih, &pih).unwrap();
        w.substream(Marker::Soz, &soz).unwrap();
        w.substream(Marker::Sorp, &[]).unwrap();
        let bytes = w.finish();
        let cs = Codestream::parse(&bytes).unwrap();
        assert_eq!(cs.substreams.len(), 3);
        assert_eq!(cs.find(Marker::Pih).unwrap(), pih);
        assert_eq!(cs.find(Marker::Soz).unwrap(), &soz[..]);
        assert_eq!(cs.find(Marker::Sorp).unwrap(), &[] as &[u8]);
        assert!(cs.find(Marker::Ton).is_none());
    }

    #[test]
    fn rejects_malformed() {
        assert!(Codestream::parse(&[]).is_err());
        assert!(Codestream::parse(&[0xFF, 0x80]).is_err());
        // SOC EOC with no PIH
        assert!(Codestream::parse(&[0xFF, 0x80, 0xFF, 0x81]).is_err());
        // first substream is not PIH
        let mut w = CodestreamWriter::new();
        assert!(w.substream(Marker::Soz, &[1]).is_err());
        // truncated payload
        let mut w = CodestreamWriter::new();
        w.substream(Marker::Pih, &[1, 2, 3, 4]).unwrap();
        let bytes = w.finish();
        assert!(Codestream::parse(&bytes[..bytes.len() - 4]).is_err());
    }

    #[test]
    fn threads_roundtrip() {
        let a = alloc::vec![1u8; 10];
        let b = alloc::vec![2u8; 13];
        let c = alloc::vec![3u8; 7];
        let d = alloc::vec![4u8; 11];
        let joined = join_threads(&[&a, &b, &c, &d]);
        let parts = split_threads(&joined, 4).unwrap();
        assert_eq!(parts, [&a[..], &b[..], &c[..], &d[..]]);
        // single thread: no deltas at all
        assert_eq!(join_threads(&[&a]), a);
        assert_eq!(split_threads(&a, 1).unwrap(), [&a[..]]);
    }

    #[test]
    fn regions_roundtrip() {
        let a = alloc::vec![1u8; 100];
        let b = alloc::vec![2u8; 0];
        let c = alloc::vec![3u8; 300];
        let joined = join_dependent_regions(&[&a, &b, &c]);
        let parts = split_regions(&[&joined], RegionLayout::Dependent, 3).unwrap();
        assert_eq!(parts, [Some(&a[..]), Some(&b[..]), Some(&c[..])]);

        let r0 = [&[0u8][..], &a[..]].concat();
        let r2 = [&[2u8][..], &c[..]].concat();
        let parts = split_regions(&[&r2, &r0], RegionLayout::Independent, 3).unwrap();
        assert_eq!(parts, [Some(&a[..]), None, Some(&c[..])]);
        assert!(split_regions(&[&r0, &r0], RegionLayout::Independent, 3).is_err());
        assert!(split_regions(&[&r2], RegionLayout::Independent, 2).is_err());
    }
}
