//! Minimal reader for the uncompressed ZIP archives `torch.save` writes.
//!
//! Only what checkpoints need: locate the central directory, list entries, and hand out the byte
//! range of *stored* (method 0) entries. ZIP64 records are understood because torch emits them for
//! large checkpoints. Nothing is decompressed and nothing is copied.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};

const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

fn bad(msg: &'static str) -> Error {
    Error::Model(String::from(msg))
}

fn u16_at(d: &[u8], pos: usize) -> Result<u16> {
    let b = d.get(pos..pos + 2).ok_or_else(|| bad("zip: truncated"))?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(d: &[u8], pos: usize) -> Result<u32> {
    let b = d.get(pos..pos + 4).ok_or_else(|| bad("zip: truncated"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(d: &[u8], pos: usize) -> Result<u64> {
    let b = d.get(pos..pos + 8).ok_or_else(|| bad("zip: truncated"))?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// One archive member.
#[derive(Clone, Debug)]
pub(crate) struct Entry<'a> {
    pub name: String,
    pub data: &'a [u8],
}

/// Parsed archive: borrowed member byte ranges.
#[derive(Clone, Debug)]
pub(crate) struct Archive<'a> {
    pub entries: Vec<Entry<'a>>,
}

impl<'a> Archive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        // End-of-central-directory record: scan backwards over the (at most 64 KiB) comment.
        if data.len() < 22 {
            return Err(bad("zip: file too small"));
        }
        let lo = data.len().saturating_sub(22 + 65535);
        let eocd = (lo..=data.len() - 22)
            .rev()
            .find(|&p| u32_at(data, p).is_ok_and(|s| s == EOCD_SIG))
            .ok_or_else(|| bad("zip: end of central directory not found"))?;

        let mut count = u16_at(data, eocd + 10)? as u64;
        let mut cd_size = u32_at(data, eocd + 12)? as u64;
        let mut cd_offset = u32_at(data, eocd + 16)? as u64;

        // ZIP64: a locator sits right before the EOCD and points at the 64-bit record.
        if eocd >= 20 && u32_at(data, eocd - 20)? == EOCD64_LOCATOR_SIG {
            let rec = usize::try_from(u64_at(data, eocd - 20 + 8)?)
                .map_err(|_| bad("zip: offset overflow"))?;
            if u32_at(data, rec)? != EOCD64_SIG {
                return Err(bad("zip: bad zip64 record"));
            }
            count = u64_at(data, rec + 32)?;
            cd_size = u64_at(data, rec + 40)?;
            cd_offset = u64_at(data, rec + 48)?;
        }
        if count > 1_000_000 {
            return Err(bad("zip: implausible entry count"));
        }
        let cd_offset = usize::try_from(cd_offset).map_err(|_| bad("zip: offset overflow"))?;
        let cd_end = cd_offset
            .checked_add(usize::try_from(cd_size).map_err(|_| bad("zip: size overflow"))?)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| bad("zip: central directory out of bounds"))?;

        let mut entries = Vec::with_capacity(count as usize);
        let mut pos = cd_offset;
        for _ in 0..count {
            if pos + 46 > cd_end || u32_at(data, pos)? != CENTRAL_SIG {
                return Err(bad("zip: bad central directory entry"));
            }
            let method = u16_at(data, pos + 10)?;
            let mut comp_size = u32_at(data, pos + 20)? as u64;
            let mut size = u32_at(data, pos + 24)? as u64;
            let name_len = u16_at(data, pos + 28)? as usize;
            let extra_len = u16_at(data, pos + 30)? as usize;
            let comment_len = u16_at(data, pos + 32)? as usize;
            let mut local_offset = u32_at(data, pos + 42)? as u64;
            let name_bytes = data
                .get(pos + 46..pos + 46 + name_len)
                .ok_or_else(|| bad("zip: truncated name"))?;
            let name = core::str::from_utf8(name_bytes)
                .map_err(|_| bad("zip: entry name is not UTF-8"))?;

            // ZIP64 extended information (header id 1): values appear only for saturated fields,
            // in the order size, compressed size, local header offset.
            let extra = data
                .get(pos + 46 + name_len..pos + 46 + name_len + extra_len)
                .ok_or_else(|| bad("zip: truncated extra"))?;
            let mut e = 0usize;
            while e + 4 <= extra.len() {
                let id = u16_at(extra, e)?;
                let len = u16_at(extra, e + 2)? as usize;
                let body = extra
                    .get(e + 4..e + 4 + len)
                    .ok_or_else(|| bad("zip: truncated extra field"))?;
                if id == 1 {
                    let mut q = 0usize;
                    for field in [&mut size, &mut comp_size, &mut local_offset] {
                        let saturated = *field == u32::MAX as u64;
                        if saturated {
                            *field = u64_at(body, q)?;
                            q += 8;
                        }
                    }
                }
                e += 4 + len;
            }

            if method != 0 || comp_size != size {
                return Err(Error::Model(alloc::format!(
                    "zip: entry `{name}` is compressed; torch checkpoints are stored"
                )));
            }
            let local = usize::try_from(local_offset).map_err(|_| bad("zip: offset overflow"))?;
            if u32_at(data, local)? != LOCAL_SIG {
                return Err(bad("zip: bad local header"));
            }
            // The local header has its own name/extra lengths (torch pads the extra field so the
            // payload is 64-byte aligned); the central directory's lengths do not apply here.
            let l_name = u16_at(data, local + 26)? as usize;
            let l_extra = u16_at(data, local + 28)? as usize;
            let start = local + 30 + l_name + l_extra;
            let size = usize::try_from(size).map_err(|_| bad("zip: size overflow"))?;
            let body = start
                .checked_add(size)
                .and_then(|end| data.get(start..end))
                .ok_or_else(|| bad("zip: entry data out of bounds"))?;
            entries.push(Entry {
                name: String::from(name),
                data: body,
            });
            pos += 46 + name_len + extra_len + comment_len;
        }
        Ok(Self { entries })
    }

    pub fn get(&self, name: &str) -> Option<&'a [u8]> {
        self.entries.iter().find(|e| e.name == name).map(|e| e.data)
    }
}
