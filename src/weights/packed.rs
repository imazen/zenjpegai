//! Packed model files: only the tensors a decoder reads, as plain little-endian bytes.
//!
//! The upstream `.pth` checkpoints carry encoder-side tensors and, in places, optimizer state:
//! the four files one (model, operating point) pair needs add up to tens of megabytes of which
//! the decoder reads a fraction. Two containers, both little-endian, both lossless (tensor bytes
//! are copied, never re-quantised, so decoded pixels are bit-identical to the `.pth` path):
//!
//! **`ZJM1`**: one checkpoint. [`super::Checkpoint::parse`] accepts it wherever it accepts a
//! `.pth` file.
//!
//! ```text
//! 0   "ZJM1"
//! 4   u32 version (1)
//! 8   u32 tensor count
//! 12  u32 integer-entry count
//! 16  per tensor:  u16 name length, name (UTF-8), u8 dtype code, u8 rank, rank x u64 dims,
//!                  u64 byte offset from the start of the file (a multiple of 64), u64 byte length
//!     per integer: u16 name length, name, i64 value
//!     zero padding to a multiple of 64, then the tensor bytes (row-major, contiguous), each
//!     tensor zero-padded to a multiple of 64
//! ```
//!
//! **`ZJB1`**: a bundle of files addressed by their path relative to the reference `models/`
//! directory (`VM_common_int/Y_0.012.pth`, ...), plus a notice text (the upstream licence, which
//! has to travel with redistributed weights). [`PackedBundle`] serves it as a
//! [`crate::model::ModelSource`] without copying.
//!
//! ```text
//! 0   "ZJB1"
//! 4   u32 version (1)
//! 8   u32 file count
//! 12  u32 notice length, then the notice (UTF-8)
//!     per file: u16 path length, path, u64 byte offset (multiple of 64), u64 byte length
//!     zero padding to a multiple of 64, then the files, each padded to a multiple of 64
//! ```

use alloc::borrow::Cow;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::dtype::{DType, TensorRef};
use super::{Checkpoint, model_err};
use crate::error::{Error, Result};

pub const ZJM_MAGIC: &[u8; 4] = b"ZJM1";
pub const ZJB_MAGIC: &[u8; 4] = b"ZJB1";
const VERSION: u32 = 1;
const ALIGN: usize = 64;
/// Bounds on table sizes, against hostile inputs.
const MAX_ENTRIES: u32 = 1 << 20;
const MAX_RANK: usize = 8;

fn dtype_code(d: DType) -> u8 {
    match d {
        DType::F64 => 0,
        DType::F32 => 1,
        DType::F16 => 2,
        DType::I64 => 3,
        DType::I32 => 4,
        DType::I16 => 5,
        DType::I8 => 6,
        DType::U8 => 7,
        DType::Bool => 8,
    }
}

fn dtype_from_code(c: u8) -> Result<DType> {
    Ok(match c {
        0 => DType::F64,
        1 => DType::F32,
        2 => DType::F16,
        3 => DType::I64,
        4 => DType::I32,
        5 => DType::I16,
        6 => DType::I8,
        7 => DType::U8,
        8 => DType::Bool,
        _ => return Err(model_err("packed model: unknown dtype code")),
    })
}

/// Bounds-checked little-endian cursor.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| model_err("packed model: truncated table"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(a))
    }
    fn usize(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| model_err("packed model: offset exceeds usize"))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        core::str::from_utf8(self.take(n)?)
            .map(ToString::to_string)
            .map_err(|_| model_err("packed model: name is not UTF-8"))
    }
}

fn header<'a>(file: &'a [u8], magic: &[u8; 4]) -> Result<Reader<'a>> {
    let mut r = Reader { data: file, pos: 0 };
    if r.take(4)? != magic {
        return Err(model_err("packed model: bad magic"));
    }
    if r.u32()? != VERSION {
        return Err(model_err("packed model: unsupported version"));
    }
    Ok(r)
}

/// A tensor table entry and the integer entries of a `ZJM1` file.
pub(super) type ZjmTables = (Vec<(String, TensorRef)>, Vec<(String, i64)>);

/// Parse the tables of a `ZJM1` file. Tensors come back as contiguous [`TensorRef`]s whose
/// `storage_offset` (in elements) indexes the file itself.
pub(super) fn parse_zjm(file: &[u8]) -> Result<ZjmTables> {
    let mut r = header(file, ZJM_MAGIC)?;
    let (nt, ni) = (r.u32()?, r.u32()?);
    if nt > MAX_ENTRIES || ni > MAX_ENTRIES {
        return Err(Error::LimitExceeded("packed model: too many entries"));
    }
    let mut tensors = Vec::new();
    for _ in 0..nt {
        let name = r.string()?;
        let dtype = dtype_from_code(r.u8()?)?;
        let rank = r.u8()? as usize;
        if rank > MAX_RANK {
            return Err(model_err("packed model: tensor rank above 8"));
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(r.usize()?);
        }
        let (offset, len) = (r.usize()?, r.usize()?);
        let es = dtype.size();
        let numel = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .and_then(|n| n.checked_mul(es))
            .ok_or_else(|| model_err("packed model: tensor size overflow"))?;
        let in_file = offset.checked_add(len).is_some_and(|e| e <= file.len());
        if numel != len || !offset.is_multiple_of(ALIGN) || !in_file {
            return Err(model_err(alloc::format!(
                "packed model: tensor `{name}` has an invalid extent"
            )));
        }
        let mut stride = alloc::vec![1usize; rank];
        for ax in (0..rank.saturating_sub(1)).rev() {
            stride[ax] = stride[ax + 1].saturating_mul(shape[ax + 1]);
        }
        tensors.push((
            name,
            TensorRef {
                storage_key: String::new(),
                dtype,
                storage_offset: offset / es,
                shape,
                stride,
            },
        ));
    }
    let mut ints = Vec::new();
    for _ in 0..ni {
        let name = r.string()?;
        ints.push((name, r.u64()? as i64));
    }
    Ok((tensors, ints))
}

fn pad(out: &mut Vec<u8>) {
    out.resize(out.len().next_multiple_of(ALIGN), 0);
}

fn put_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let n = u16::try_from(s.len()).map_err(|_| model_err("packed model: name too long"))?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

/// Write the tensors `names` of `ck` (plus all of its integer entries) as a `ZJM1` file.
/// Tensor bytes are copied unchanged (strided views are made contiguous).
pub fn write_zjm(ck: &Checkpoint<'_>, names: &[String]) -> Result<Vec<u8>> {
    let mut payloads = Vec::with_capacity(names.len());
    for name in names {
        payloads.push(ck.raw(name)?);
    }
    // Table size first: offsets are absolute.
    let mut table = 16usize;
    for (name, (_, shape, _)) in names.iter().zip(&payloads) {
        table += 2 + name.len() + 2 + 8 * shape.len() + 16;
    }
    for (name, _) in ck.ints() {
        table += 2 + name.len() + 8;
    }
    let mut offset = table.next_multiple_of(ALIGN);
    let mut out = Vec::new();
    out.extend_from_slice(ZJM_MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(names.len() as u32).to_le_bytes());
    out.extend_from_slice(&(ck.ints().len() as u32).to_le_bytes());
    for (name, (dtype, shape, bytes)) in names.iter().zip(&payloads) {
        if shape.len() > MAX_RANK {
            return Err(model_err("packed model: tensor rank above 8"));
        }
        put_str(&mut out, name)?;
        out.push(dtype_code(*dtype));
        out.push(shape.len() as u8);
        for &d in shape {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        offset = (offset + bytes.len()).next_multiple_of(ALIGN);
    }
    for (name, v) in ck.ints() {
        put_str(&mut out, name)?;
        out.extend_from_slice(&v.to_le_bytes());
    }
    debug_assert_eq!(out.len(), table);
    pad(&mut out);
    for (_, _, bytes) in &payloads {
        out.extend_from_slice(bytes);
        pad(&mut out);
    }
    Ok(out)
}

/// Write a `ZJB1` bundle of `(relative path, file bytes)` pairs. Files are usually `ZJM1`
/// checkpoints but any bytes work.
pub fn write_bundle(files: &[(String, Vec<u8>)], notice: &str) -> Result<Vec<u8>> {
    let mut table = 16 + notice.len();
    for (rel, _) in files {
        table += 2 + rel.len() + 16;
    }
    let mut offset = table.next_multiple_of(ALIGN);
    let mut out = Vec::new();
    out.extend_from_slice(ZJB_MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(files.len() as u32).to_le_bytes());
    let n = u32::try_from(notice.len()).map_err(|_| model_err("bundle: notice too long"))?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(notice.as_bytes());
    for (rel, bytes) in files {
        put_str(&mut out, rel)?;
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        offset = (offset + bytes.len()).next_multiple_of(ALIGN);
    }
    pad(&mut out);
    for (_, bytes) in files {
        out.extend_from_slice(bytes);
        pad(&mut out);
    }
    Ok(out)
}

/// A `ZJB1` bundle held in memory, served as a [`crate::model::ModelSource`] without copying.
/// Several bundles can be merged into one source with [`PackedBundle::add`] (a page that needs a
/// second (model, operating point) pair fetches a second bundle).
#[derive(Clone, Debug, Default)]
pub struct PackedBundle {
    parts: Vec<Part>,
}

#[derive(Clone, Debug)]
struct Part {
    data: Vec<u8>,
    notice: core::ops::Range<usize>,
    /// (relative path, byte range in `data`), sorted by path.
    files: Vec<(String, core::ops::Range<usize>)>,
}

impl PackedBundle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a `ZJB1` file.
    pub fn parse(data: Vec<u8>) -> Result<Self> {
        let mut b = Self::new();
        b.add(data)?;
        Ok(b)
    }

    /// Add the files of another `ZJB1` bundle. A path already present keeps its first copy
    /// (bundles of the same model share the common checkpoints, byte for byte).
    pub fn add(&mut self, data: Vec<u8>) -> Result<()> {
        let mut r = header(&data, ZJB_MAGIC)?;
        let count = r.u32()?;
        if count > MAX_ENTRIES {
            return Err(Error::LimitExceeded("bundle: too many files"));
        }
        let notice_len = r.u32()? as usize;
        let notice = r.pos..r.pos + r.take(notice_len)?.len();
        if core::str::from_utf8(&data[notice.clone()]).is_err() {
            return Err(model_err("bundle: notice is not UTF-8"));
        }
        let mut files = Vec::new();
        for _ in 0..count {
            let rel = r.string()?;
            let (offset, len) = (r.usize()?, r.usize()?);
            let end = offset
                .checked_add(len)
                .filter(|&e| e <= data.len() && offset.is_multiple_of(ALIGN))
                .ok_or_else(|| {
                    model_err(alloc::format!("bundle: `{rel}` has an invalid extent"))
                })?;
            files.push((rel, offset..end));
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        self.parts.push(Part {
            data,
            notice,
            files,
        });
        Ok(())
    }

    /// Relative paths of the files held.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.parts
            .iter()
            .flat_map(|p| p.files.iter().map(|(n, _)| n.as_str()))
    }

    pub fn contains(&self, rel: &str) -> bool {
        self.get(rel).is_some()
    }

    fn get(&self, rel: &str) -> Option<&[u8]> {
        self.parts.iter().find_map(|p| {
            let i = p
                .files
                .binary_search_by(|(n, _)| n.as_str().cmp(rel))
                .ok()?;
            Some(&p.data[p.files[i].1.clone()])
        })
    }

    /// The notice text of the first bundle added (the upstream licence of the weights).
    pub fn notice(&self) -> &str {
        self.parts
            .first()
            .and_then(|p| core::str::from_utf8(&p.data[p.notice.clone()]).ok())
            .unwrap_or("")
    }
}

impl crate::model::ModelSource for PackedBundle {
    fn read(&self, rel: &str) -> Result<Cow<'_, [u8]>> {
        self.get(rel)
            .map(Cow::Borrowed)
            .ok_or_else(|| Error::Model(alloc::format!("{rel}: not in the model bundle")))
    }
}

/// A [`crate::model::ModelSource`] wrapper that notes which tensors of which files the loaders
/// look up; [`Recorder::pack`] then writes exactly those into a `ZJB1` bundle.
#[cfg(feature = "std")]
pub struct Recorder<S> {
    inner: S,
    seen: std::sync::Mutex<
        alloc::collections::BTreeMap<String, alloc::collections::BTreeSet<String>>,
    >,
}

#[cfg(feature = "std")]
impl<S: crate::model::ModelSource> Recorder<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            seen: Default::default(),
        }
    }

    /// Pack every file read so far, reduced to the tensors looked up in it. Tensors keep their
    /// file order.
    pub fn pack(&self, notice: &str) -> Result<Vec<u8>> {
        let seen = self
            .seen
            .lock()
            .map_err(|_| model_err("recorder poisoned"))?;
        let mut files = Vec::new();
        for (rel, names) in seen.iter() {
            let file = self.inner.read(rel)?;
            let ck = Checkpoint::parse(&file)?;
            let ordered: Vec<String> = ck
                .tensor_names()
                .filter(|n| names.contains(*n))
                .map(ToString::to_string)
                .collect();
            files.push((rel.clone(), write_zjm(&ck, &ordered)?));
        }
        write_bundle(&files, notice)
    }
}

#[cfg(feature = "std")]
impl<S: crate::model::ModelSource> crate::model::ModelSource for Recorder<S> {
    fn read(&self, rel: &str) -> Result<Cow<'_, [u8]>> {
        self.inner.read(rel)
    }

    fn accessed(&self, rel: &str, tensors: &[String]) {
        if let Ok(mut seen) = self.seen.lock() {
            seen.entry(rel.to_string())
                .or_default()
                .extend(tensors.iter().cloned());
        }
    }
}
