//! Reader for the reference software's PyTorch checkpoints (`models/**/*.pth`).
//!
//! A `.pth` file is an uncompressed ZIP holding `<prefix>/data.pkl` (a pickled state dict whose
//! tensors are persistent references to storages) and one `<prefix>/data/<key>` member per
//! storage, little-endian. [`Checkpoint::parse`] runs the pickle through a data-only interpreter
//! (no code execution) and exposes the top-level tensors by name; tensor bytes are only touched
//! when a tensor is requested, so the optimizer state some upstream files carry costs nothing.

mod pickle;
mod zip;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use pickle::Value;
pub use pickle::{DType, TensorRef};

use crate::error::{Error, Result};

/// A parsed checkpoint borrowing the file bytes.
pub struct Checkpoint<'a> {
    archive: zip::Archive<'a>,
    prefix: String,
    /// Top-level tensors in file order.
    tensors: Vec<(String, TensorRef)>,
    /// Top-level integer entries (`epoch`, ...).
    ints: Vec<(String, i64)>,
}

/// A materialised tensor: contiguous, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

impl<'a> Checkpoint<'a> {
    pub fn parse(file: &'a [u8]) -> Result<Self> {
        let archive = zip::Archive::parse(file)?;
        let pkl = archive
            .entries
            .iter()
            .find(|e| e.name.ends_with("/data.pkl") || e.name == "data.pkl")
            .ok_or_else(|| model_err("checkpoint has no data.pkl"))?;
        let prefix = pkl.name.strip_suffix("data.pkl").unwrap_or("").to_string();
        let root = pickle::load(pkl.data)?;
        let Value::Dict(items) = root else {
            return Err(model_err("checkpoint root is not a state dict"));
        };
        let mut tensors = Vec::new();
        let mut ints = Vec::new();
        for (k, v) in items {
            let Value::Str(name) = k else { continue };
            match v {
                Value::Tensor(t) => tensors.push((name, t)),
                Value::Int(i) => ints.push((name, i)),
                // nested dicts (optimizer state) and anything else are not model weights
                _ => {}
            }
        }
        Ok(Self {
            archive,
            prefix,
            tensors,
            ints,
        })
    }

    /// Names of the top-level tensors, in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n == name)
    }

    /// Shape and dtype of a tensor without touching its data.
    pub fn info(&self, name: &str) -> Option<&TensorRef> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// A top-level integer entry such as `epoch`.
    pub fn int(&self, name: &str) -> Option<i64> {
        self.ints.iter().find(|(n, _)| n == name).map(|&(_, v)| v)
    }

    /// Contiguous little-endian bytes of a tensor, row-major, plus its shape.
    fn bytes(&self, name: &str, want: DType) -> Result<(Vec<usize>, Vec<u8>)> {
        let t = self
            .info(name)
            .ok_or_else(|| model_err(alloc::format!("tensor `{name}` not found")))?;
        if t.dtype != want {
            return Err(model_err(alloc::format!(
                "tensor `{name}` is {:?}, expected {want:?}",
                t.dtype
            )));
        }
        if t.shape.len() != t.stride.len() {
            return Err(model_err(alloc::format!(
                "tensor `{name}`: shape/stride rank mismatch"
            )));
        }
        let storage = self
            .archive
            .get(&alloc::format!("{}data/{}", self.prefix, t.storage_key))
            .ok_or_else(|| {
                model_err(alloc::format!(
                    "tensor `{name}`: storage `{}` missing",
                    t.storage_key
                ))
            })?;
        let es = t.dtype.size();
        let numel = t
            .shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| model_err("tensor size overflow"))?;
        let total = numel
            .checked_mul(es)
            .ok_or_else(|| model_err("tensor size overflow"))?;
        let storage_elems = storage.len() / es;

        // Row-major strides of a contiguous tensor with this shape.
        let mut contiguous = true;
        let mut expect = 1usize;
        for (&d, &s) in t.shape.iter().zip(&t.stride).rev() {
            if d != 1 && s != expect {
                contiguous = false;
            }
            expect = expect.saturating_mul(d);
        }

        let mut out = Vec::new();
        out.try_reserve_exact(total)
            .map_err(|_| Error::LimitExceeded("out of memory loading tensor"))?;
        if contiguous {
            let start = t
                .storage_offset
                .checked_mul(es)
                .ok_or_else(|| model_err("tensor offset overflow"))?;
            let src = start
                .checked_add(total)
                .and_then(|end| storage.get(start..end))
                .ok_or_else(|| model_err(alloc::format!("tensor `{name}` exceeds its storage")))?;
            out.extend_from_slice(src);
        } else {
            // General strided gather (views saved without `.contiguous()`).
            let mut idx = alloc::vec![0usize; t.shape.len()];
            for _ in 0..numel {
                let mut elem = t.storage_offset;
                for (&i, &s) in idx.iter().zip(&t.stride) {
                    elem = elem
                        .checked_add(
                            i.checked_mul(s)
                                .ok_or_else(|| model_err("stride overflow"))?,
                        )
                        .ok_or_else(|| model_err("stride overflow"))?;
                }
                if elem >= storage_elems {
                    return Err(model_err(alloc::format!(
                        "tensor `{name}` exceeds its storage"
                    )));
                }
                out.extend_from_slice(&storage[elem * es..(elem + 1) * es]);
                for ax in (0..idx.len()).rev() {
                    idx[ax] += 1;
                    if idx[ax] < t.shape[ax] {
                        break;
                    }
                    idx[ax] = 0;
                }
            }
        }
        Ok((t.shape.clone(), out))
    }

    pub fn f32(&self, name: &str) -> Result<Tensor<f32>> {
        let (shape, b) = self.bytes(name, DType::F32)?;
        let data = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(Tensor { shape, data })
    }

    pub fn i32(&self, name: &str) -> Result<Tensor<i32>> {
        let (shape, b) = self.bytes(name, DType::I32)?;
        let data = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        Ok(Tensor { shape, data })
    }

    pub fn i64(&self, name: &str) -> Result<Tensor<i64>> {
        let (shape, b) = self.bytes(name, DType::I64)?;
        let data = b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        Ok(Tensor { shape, data })
    }

    pub fn i8(&self, name: &str) -> Result<Tensor<i8>> {
        let (shape, b) = self.bytes(name, DType::I8)?;
        Ok(Tensor {
            shape,
            data: b.into_iter().map(|v| v as i8).collect(),
        })
    }

    pub fn bool(&self, name: &str) -> Result<Tensor<bool>> {
        let (shape, b) = self.bytes(name, DType::Bool)?;
        Ok(Tensor {
            shape,
            data: b.into_iter().map(|v| v != 0).collect(),
        })
    }
}
