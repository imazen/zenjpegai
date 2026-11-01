//! Reader for the reference software's PyTorch checkpoints (`models/**/*.pth`) and packed
//! bundles.
//!
//! A `.pth` file is an uncompressed ZIP holding `<prefix>/data.pkl` (a pickled state dict whose
//! tensors are persistent references to storages) and one `<prefix>/data/<key>` member per
//! storage, little-endian. [`Checkpoint::parse`] runs the pickle through a data-only interpreter
//! (no code execution) and exposes the top-level tensors by name; tensor bytes are only touched
//! when a tensor is requested, so the optimizer state some upstream files carry costs nothing.
//! Reading raw `.pth` files needs the `pth` feature (on by default; off in the browser wasm
//! build, which only ever reads packed `ZJM1`/`ZJB1` bundles — see `wasm/Cargo.toml` and
//! `benchmarks/wasm_size_*.md`). With `pth` off, [`Checkpoint::parse`] accepts packed files only.

pub mod dtype;
pub mod f16;
pub mod packed;
#[cfg(feature = "pth")]
mod pickle;
#[cfg(feature = "pth")]
mod zip;

use alloc::string::String;
#[cfg(feature = "pth")]
use alloc::string::ToString;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

pub use dtype::{DType, TensorRef};
#[cfg(feature = "pth")]
use pickle::Value;

use crate::error::{Error, Result};

/// Where tensor bytes live.
enum Backing<'a> {
    /// A PyTorch ZIP: storages are archive members `<prefix>data/<key>`. Only ever constructed
    /// with the `pth` feature on (see [`Checkpoint::parse`]), but the variant itself stays
    /// unconditional so [`Backing`]'s other match arms don't need `#[cfg]` scattered on them.
    #[cfg(feature = "pth")]
    Torch {
        archive: zip::Archive<'a>,
        prefix: String,
    },
    /// A packed `ZJM1` checkpoint ([`packed`]): one storage, the file itself; every tensor is
    /// contiguous and its `storage_offset` (in elements) points into the file.
    Packed(&'a [u8]),
}

/// A parsed checkpoint borrowing the file bytes: an upstream PyTorch `.pth` file or a packed
/// `ZJM1` file, behind the same accessors.
pub struct Checkpoint<'a> {
    backing: Backing<'a>,
    /// Top-level tensors in file order.
    tensors: Vec<(String, TensorRef)>,
    /// Per tensor: looked up by name (`contains`, `info` or a data accessor) since parsing.
    touched: Vec<AtomicBool>,
    /// Top-level integer entries (`epoch`, ...).
    ints: Vec<(String, i64)>,
    /// Numeric scalar lists found in nested dicts (e.g. `clip_thres.E1B.MinList`), only from
    /// `.pth` checkpoints — packed files have no dicts.
    lists: Vec<(String, Vec<f64>)>,
}

/// A materialised tensor: contiguous, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
    /// `data`'s bytes in the tracked ledger (see [`crate::mem`]); not part of the value.
    pub(crate) charge: crate::mem::Charge,
}

impl<T> Tensor<T> {
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Flatten a nested state-dict dict into dotted names, up to `MAX_DEPTH` levels: tensor
/// leaves join `tensors` (`key.layer.weight`), lists of numbers join `lists`
/// (`key.layer.MinList` — the shape `clip_thres` tables have). Other values are ignored.
#[cfg(feature = "pth")]
fn flatten_nested(
    prefix: &str,
    items: Vec<(pickle::Value, pickle::Value)>,
    depth: usize,
    tensors: &mut Vec<(String, pickle::TensorRef)>,
    lists: &mut Vec<(String, Vec<f64>)>,
) {
    const MAX_DEPTH: usize = 3;
    if depth >= MAX_DEPTH {
        return;
    }
    for (k, v) in items {
        let pickle::Value::Str(leaf) = k else {
            continue;
        };
        match v {
            pickle::Value::Tensor(t) => {
                tensors.push((alloc::format!("{prefix}.{leaf}"), t));
            }
            pickle::Value::Dict(inner) => {
                flatten_nested(
                    &alloc::format!("{prefix}.{leaf}"),
                    inner,
                    depth + 1,
                    tensors,
                    lists,
                );
            }
            pickle::Value::List(xs) => {
                let xs: Option<Vec<f64>> = xs
                    .iter()
                    .map(|x| match x {
                        pickle::Value::Float(f) => Some(*f),
                        pickle::Value::Int(i) => Some(*i as f64),
                        _ => None,
                    })
                    .collect();
                if let Some(xs) = xs {
                    lists.push((alloc::format!("{prefix}.{leaf}"), xs));
                }
            }
            _ => {}
        }
    }
}

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

impl<'a> Checkpoint<'a> {
    /// Parse a PyTorch `.pth` file (needs the `pth` feature) or a packed `ZJM1` file (told apart
    /// by the magic).
    pub fn parse(file: &'a [u8]) -> Result<Self> {
        if file.starts_with(packed::ZJM_MAGIC) {
            let (tensors, ints) = packed::parse_zjm(file)?;
            return Ok(Self {
                backing: Backing::Packed(file),
                touched: tensors.iter().map(|_| AtomicBool::new(false)).collect(),
                tensors,
                ints,
                lists: Vec::new(),
            });
        }
        #[cfg(feature = "pth")]
        {
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
            let mut lists = Vec::new();
            for (k, v) in items {
                let Value::Str(name) = k else { continue };
                match v {
                    Value::Tensor(t) => tensors.push((name, t)),
                    Value::Int(i) => ints.push((name, i)),
                    // Nested dicts (`clip_thres` tables; optimizer state) are not model
                    // weights, but their tensor / scalar-list leaves are addressable as
                    // `key.layer.leaf` for loaders that want them.
                    Value::Dict(inner) => flatten_nested(&name, inner, 0, &mut tensors, &mut lists),
                    _ => {}
                }
            }
            Ok(Self {
                backing: Backing::Torch { archive, prefix },
                touched: tensors.iter().map(|_| AtomicBool::new(false)).collect(),
                tensors,
                ints,
                lists,
            })
        }
        #[cfg(not(feature = "pth"))]
        Err(model_err(
            "not a packed ZJM1/ZJB1 file, and this build has no `.pth` (PyTorch checkpoint) support",
        ))
    }

    /// Names of the top-level tensors, in file order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.info(name).is_some()
    }

    /// Shape and dtype of a tensor without touching its data.
    pub fn info(&self, name: &str) -> Option<&TensorRef> {
        let i = self.tensors.iter().position(|(n, _)| n == name)?;
        self.touched[i].store(true, Ordering::Relaxed);
        Some(&self.tensors[i].1)
    }

    /// Names of the tensors looked up by name since parsing, in file order: what a loader
    /// actually needs from this file (the input of [`packed::write_zjm`]).
    pub fn touched_names(&self) -> Vec<String> {
        self.tensors
            .iter()
            .zip(&self.touched)
            .filter(|(_, t)| t.load(Ordering::Relaxed))
            .map(|((n, _), _)| n.clone())
            .collect()
    }

    /// All top-level integer entries.
    pub fn ints(&self) -> &[(String, i64)] {
        &self.ints
    }

    /// A top-level integer entry such as `epoch`.
    pub fn int(&self, name: &str) -> Option<i64> {
        self.ints.iter().find(|(n, _)| n == name).map(|&(_, v)| v)
    }

    /// A numeric scalar list found in a nested dict (`clip_thres.E1B.MinList`); `None` when
    /// the name is absent (packed checkpoints never have lists).
    pub fn list(&self, name: &str) -> Option<&[f64]> {
        self.lists
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
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
        // Without the `pth` feature `Backing` has one variant, and clippy asks for a `let`.
        #[allow(clippy::infallible_destructuring_match)]
        let storage = match &self.backing {
            #[cfg(feature = "pth")]
            Backing::Torch { archive, prefix } => archive
                .get(&alloc::format!("{prefix}data/{}", t.storage_key))
                .ok_or_else(|| {
                    model_err(alloc::format!(
                        "tensor `{name}`: storage `{}` missing",
                        t.storage_key
                    ))
                })?,
            Backing::Packed(file) => file,
        };
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

    /// Contiguous little-endian bytes of a tensor of any dtype, with its dtype and shape.
    pub fn raw(&self, name: &str) -> Result<(DType, Vec<usize>, Vec<u8>)> {
        let dtype = self
            .info(name)
            .ok_or_else(|| model_err(alloc::format!("tensor `{name}` not found")))?
            .dtype;
        let (shape, bytes) = self.bytes(name, dtype)?;
        Ok((dtype, shape, bytes))
    }

    /// f32 tensor, by name. A tensor stored as `f16` (packed `ZJB2` bundles) is upcast to f32
    /// here — the upcast is exact and deterministic, so the rest of the pipeline sees plain
    /// f32 weights whichever container they came from.
    pub fn f32(&self, name: &str) -> Result<Tensor<f32>> {
        if self
            .info(name)
            .ok_or_else(|| model_err(alloc::format!("tensor `{name}` not found")))?
            .dtype
            == DType::F16
        {
            let (shape, b) = self.bytes(name, DType::F16)?;
            let _b_charge = crate::mem::Charge::of_vec(&b);
            let data = b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::f16_to_f32(u16::from_le_bytes(*c)))
                .collect();
            let charge = crate::mem::Charge::of_vec(&data);
            return Ok(Tensor {
                shape,
                data,
                charge,
            });
        }
        let (shape, b) = self.bytes(name, DType::F32)?;
        let _b_charge = crate::mem::Charge::of_vec(&b);
        let data: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let charge = crate::mem::Charge::of_vec(&data);
        Ok(Tensor {
            shape,
            data,
            charge,
        })
    }

    pub fn i32(&self, name: &str) -> Result<Tensor<i32>> {
        let (shape, b) = self.bytes(name, DType::I32)?;
        let _b_charge = crate::mem::Charge::of_vec(&b);
        let data: Vec<i32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let charge = crate::mem::Charge::of_vec(&data);
        Ok(Tensor {
            shape,
            data,
            charge,
        })
    }

    pub fn i64(&self, name: &str) -> Result<Tensor<i64>> {
        let (shape, b) = self.bytes(name, DType::I64)?;
        let _b_charge = crate::mem::Charge::of_vec(&b);
        let data: Vec<i64> = b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect();
        let charge = crate::mem::Charge::of_vec(&data);
        Ok(Tensor {
            shape,
            data,
            charge,
        })
    }

    pub fn i8(&self, name: &str) -> Result<Tensor<i8>> {
        let (shape, b) = self.bytes(name, DType::I8)?;
        let _b_charge = crate::mem::Charge::of_vec(&b);
        let data: Vec<i8> = b.into_iter().map(|v| v as i8).collect();
        let charge = crate::mem::Charge::of_vec(&data);
        Ok(Tensor {
            shape,
            data,
            charge,
        })
    }

    pub fn bool(&self, name: &str) -> Result<Tensor<bool>> {
        let (shape, b) = self.bytes(name, DType::Bool)?;
        let _b_charge = crate::mem::Charge::of_vec(&b);
        let data: Vec<bool> = b.into_iter().map(|v| v != 0).collect();
        let charge = crate::mem::Charge::of_vec(&data);
        Ok(Tensor {
            shape,
            data,
            charge,
        })
    }
}

#[cfg(all(test, feature = "pth"))]
mod tests {
    use super::*;

    /// A `.pth` `state_dict` with a nested `clip_thres` dict (the shape
    /// `set_clip_thres`/`feature_clipping` write) flattens its scalar lists into
    /// `Checkpoint::list` under `clip_thres.<layer>.<key>`, alongside the tensors and ints.
    #[test]
    fn clip_thres_nested_dict_flattens() {
        let bytes = std::fs::read("/tmp/clip_test.pth")
            .expect("run the /tmp/clip_test.pth generator (torch.save a nested clip_thres dict)");
        let ck = Checkpoint::parse(&bytes).unwrap();
        assert_eq!(
            ck.list("clip_thres.E1B.MinList"),
            Some(vec![-1.5; 160].as_slice())
        );
        assert_eq!(
            ck.list("clip_thres.E1B.MaxList"),
            Some(vec![1.5; 160].as_slice())
        );
        assert_eq!(
            ck.list("clip_thres.HE1.MinList"),
            Some(vec![-2.0; 64].as_slice())
        );
        // The regular state dict is untouched by the flatten.
        assert!(ck.contains("conv1.weight"));
        assert_eq!(ck.int("epoch"), Some(3));
    }
}
