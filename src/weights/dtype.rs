//! Tensor element type and reference, shared by the packed-bundle reader (always compiled) and
//! the PyTorch pickle interpreter (`pickle.rs`, gated behind the `pth` feature — see its module
//! doc comment). Split out so a `pth`-less build (the browser wasm target: it only ever reads
//! packed `ZJM1`/`ZJB1` bundles) does not need to pull in the pickle interpreter just for these
//! two small types.

use alloc::string::String;
use alloc::vec::Vec;

/// Element type of a tensor storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    F16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    #[cfg(feature = "pth")]
    pub(super) fn from_storage_class(name: &str) -> Option<Self> {
        Some(match name {
            "DoubleStorage" => DType::F64,
            "FloatStorage" => DType::F32,
            "HalfStorage" => DType::F16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ShortStorage" => DType::I16,
            "CharStorage" => DType::I8,
            "ByteStorage" => DType::U8,
            "BoolStorage" => DType::Bool,
            _ => return None,
        })
    }
}

/// A tensor as described by the pickle: a view into a named storage.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    /// Storage key: the archive member `<prefix>/data/<key>`.
    pub storage_key: String,
    pub dtype: DType,
    /// Offset into the storage, in elements.
    pub storage_offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}
