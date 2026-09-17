//! A data-only pickle interpreter, sufficient for `torch.save`d state dicts.
//!
//! Python's pickle format is a program for a small stack machine. This interpreter runs the
//! opcodes torch emits (protocol 2, plus the protocol 3/4 framing opcodes) but never calls
//! anything: `GLOBAL` just records a name and `REDUCE` builds a value from a short allow-list of
//! constructors (`OrderedDict`, `_rebuild_tensor_v2`, `_rebuild_parameter`). Everything else
//! becomes an opaque [`Value::Object`]. Loading a checkpoint therefore cannot execute code.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// Upper bound on interpreter stack depth / container nesting, against hostile inputs.
const MAX_STACK: usize = 1 << 20;
const MAX_MEMO: usize = 1 << 22;

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

    fn from_storage_class(name: &str) -> Option<Self> {
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

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    /// Insertion-ordered mapping (`dict` and `OrderedDict`).
    Dict(Vec<(Value, Value)>),
    /// `module.name`, as written by `GLOBAL` / `STACK_GLOBAL`.
    Global(String, String),
    /// A persistent-id storage reference.
    Storage {
        key: String,
        dtype: DType,
    },
    Tensor(TensorRef),
    /// Anything constructed by a callable outside the allow-list.
    Object(Box<Value>, Box<Value>),
    /// Interpreter-internal stack marker.
    Mark,
}

fn bad(msg: &'static str) -> Error {
    Error::Model(String::from(msg))
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("pickle: length overflow"))?;
        let s = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| bad("pickle: truncated"))?;
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
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let nl = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| bad("pickle: unterminated line"))?;
        let s = core::str::from_utf8(&rest[..nl]).map_err(|_| bad("pickle: line is not UTF-8"))?;
        self.pos += nl + 1;
        Ok(s)
    }
    fn utf8(&mut self, n: usize) -> Result<String> {
        let b = self.take(n)?;
        core::str::from_utf8(b)
            .map(String::from)
            .map_err(|_| bad("pickle: string is not UTF-8"))
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value> {
    match stack.pop() {
        Some(Value::Mark) | None => Err(bad("pickle: stack underflow")),
        Some(v) => Ok(v),
    }
}

fn pop_to_mark(stack: &mut Vec<Value>) -> Result<Vec<Value>> {
    let at = stack
        .iter()
        .rposition(|v| matches!(v, Value::Mark))
        .ok_or_else(|| bad("pickle: mark not found"))?;
    let items = stack.split_off(at + 1);
    stack.pop();
    Ok(items)
}

fn as_usize(v: &Value) -> Result<usize> {
    match v {
        Value::Int(i) => usize::try_from(*i).map_err(|_| bad("pickle: negative size")),
        _ => Err(bad("pickle: expected an integer")),
    }
}

fn as_usize_list(v: &Value) -> Result<Vec<usize>> {
    match v {
        Value::Tuple(items) | Value::List(items) => items.iter().map(as_usize).collect(),
        _ => Err(bad("pickle: expected a tuple of integers")),
    }
}

/// Little-endian two's-complement integer of up to 8 bytes (`LONG1`).
fn long_from_le(bytes: &[u8]) -> Result<i64> {
    if bytes.len() > 8 {
        return Err(bad("pickle: integer wider than 64 bits"));
    }
    let mut buf = if bytes.last().is_some_and(|b| b & 0x80 != 0) {
        [0xFFu8; 8]
    } else {
        [0u8; 8]
    };
    buf[..bytes.len()].copy_from_slice(bytes);
    Ok(i64::from_le_bytes(buf))
}

/// `REDUCE` / `NEWOBJ`: build a value from a callable and its argument tuple.
fn reduce(callable: Value, args: Value) -> Result<Value> {
    if let (Value::Global(module, name), Value::Tuple(items)) = (&callable, &args) {
        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") if items.is_empty() => {
                return Ok(Value::Dict(Vec::new()));
            }
            ("torch._utils", "_rebuild_tensor_v2") if items.len() >= 4 => {
                let Value::Storage { key, dtype } = &items[0] else {
                    return Err(bad("pickle: tensor without a storage"));
                };
                return Ok(Value::Tensor(TensorRef {
                    storage_key: key.clone(),
                    dtype: *dtype,
                    storage_offset: as_usize(&items[1])?,
                    shape: as_usize_list(&items[2])?,
                    stride: as_usize_list(&items[3])?,
                }));
            }
            // Parameter(data, requires_grad, backward_hooks) -> data
            ("torch._utils", "_rebuild_parameter") if !items.is_empty() => {
                return Ok(items[0].clone());
            }
            _ => {}
        }
    }
    Ok(Value::Object(Box::new(callable), Box::new(args)))
}

/// Persistent id for a storage: `('storage', <StorageClass>, key, location, numel)`.
fn persistent_load(pid: Value) -> Result<Value> {
    let Value::Tuple(items) = pid else {
        return Err(bad("pickle: persistent id is not a tuple"));
    };
    match items.as_slice() {
        [
            Value::Str(tag),
            Value::Global(_, class),
            Value::Str(key),
            ..,
        ] if tag == "storage" => {
            let dtype = DType::from_storage_class(class).ok_or_else(|| {
                Error::Model(alloc::format!(
                    "pickle: unsupported storage class `{class}`"
                ))
            })?;
            Ok(Value::Storage {
                key: key.clone(),
                dtype,
            })
        }
        _ => Err(bad("pickle: unrecognised persistent id")),
    }
}

/// Run a pickle program and return the value left on the stack by `STOP`.
pub(crate) fn load(data: &[u8]) -> Result<Value> {
    let mut r = Reader { data, pos: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: Vec<Option<Value>> = Vec::new();

    let memo_put = |memo: &mut Vec<Option<Value>>, idx: usize, v: Value| -> Result<()> {
        if idx >= MAX_MEMO {
            return Err(bad("pickle: memo index too large"));
        }
        if idx >= memo.len() {
            memo.resize(idx + 1, None);
        }
        memo[idx] = Some(v);
        Ok(())
    };
    let memo_get = |memo: &[Option<Value>], idx: usize| -> Result<Value> {
        memo.get(idx)
            .cloned()
            .flatten()
            .ok_or_else(|| bad("pickle: memo miss"))
    };

    loop {
        if stack.len() > MAX_STACK {
            return Err(bad("pickle: stack too deep"));
        }
        let op = r.u8()?;
        match op {
            0x80 => {
                let proto = r.u8()?;
                if proto > 5 {
                    return Err(bad("pickle: unsupported protocol"));
                }
            }
            0x95 => {
                r.u64()?; // FRAME: length hint only
            }
            b'.' => return pop(&mut stack),
            b'(' => stack.push(Value::Mark),
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'J' => stack.push(Value::Int(r.u32()? as i32 as i64)),
            b'K' => stack.push(Value::Int(r.u8()? as i64)),
            b'M' => stack.push(Value::Int(r.u16()? as i64)),
            0x8a => {
                let n = r.u8()? as usize;
                stack.push(Value::Int(long_from_le(r.take(n)?)?));
            }
            b'G' => stack.push(Value::Float(f64::from_bits(u64::from_be_bytes(
                r.take(8)?.try_into().unwrap(),
            )))),
            b'X' => {
                let n = r.u32()? as usize;
                stack.push(Value::Str(r.utf8(n)?));
            }
            0x8c => {
                let n = r.u8()? as usize;
                stack.push(Value::Str(r.utf8(n)?));
            }
            0x8d => {
                let n = usize::try_from(r.u64()?).map_err(|_| bad("pickle: length overflow"))?;
                stack.push(Value::Str(r.utf8(n)?));
            }
            b'U' => {
                let n = r.u8()? as usize;
                stack.push(Value::Bytes(r.take(n)?.to_vec()));
            }
            b'T' | b'B' => {
                let n = r.u32()? as usize;
                stack.push(Value::Bytes(r.take(n)?.to_vec()));
            }
            b'C' => {
                let n = r.u8()? as usize;
                stack.push(Value::Bytes(r.take(n)?.to_vec()));
            }
            b'c' => {
                let module = String::from(r.line()?);
                let name = String::from(r.line()?);
                stack.push(Value::Global(module, name));
            }
            0x93 => {
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                match (module, name) {
                    (Value::Str(m), Value::Str(n)) => stack.push(Value::Global(m, n)),
                    _ => return Err(bad("pickle: STACK_GLOBAL operands are not strings")),
                }
            }
            b')' => stack.push(Value::Tuple(Vec::new())),
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Value::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(alloc::vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(alloc::vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(Value::Tuple(alloc::vec![a, b, c]));
            }
            b']' => stack.push(Value::List(Vec::new())),
            b'l' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(Value::List(items));
            }
            b'}' => stack.push(Value::Dict(Vec::new())),
            b'd' => {
                let items = pop_to_mark(&mut stack)?;
                let mut pairs = Vec::with_capacity(items.len() / 2);
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    pairs.push((k, v));
                }
                stack.push(Value::Dict(pairs));
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.push(v),
                    _ => return Err(bad("pickle: APPEND target is not a list")),
                }
            }
            b'e' => {
                let items = pop_to_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::List(l)) => l.extend(items),
                    _ => return Err(bad("pickle: APPENDS target is not a list")),
                }
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::Dict(d)) => d.push((k, v)),
                    // setting items on an opaque object: ignore
                    Some(Value::Object(..)) => {}
                    _ => return Err(bad("pickle: SETITEM target is not a dict")),
                }
            }
            b'u' => {
                let items = pop_to_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Value::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    Some(Value::Object(..)) => {}
                    _ => return Err(bad("pickle: SETITEMS target is not a dict")),
                }
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(callable, args)?);
            }
            0x81 => {
                let args = pop(&mut stack)?;
                let cls = pop(&mut stack)?;
                stack.push(reduce(cls, args)?);
            }
            b'b' => {
                // BUILD: object state (e.g. OrderedDict._metadata). Carries no tensor data.
                pop(&mut stack)?;
                if stack.is_empty() {
                    return Err(bad("pickle: BUILD without an object"));
                }
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(persistent_load(pid)?);
            }
            b'q' => {
                let idx = r.u8()? as usize;
                let v = stack
                    .last()
                    .cloned()
                    .ok_or_else(|| bad("pickle: BINPUT on empty stack"))?;
                memo_put(&mut memo, idx, v)?;
            }
            b'r' => {
                let idx = r.u32()? as usize;
                let v = stack
                    .last()
                    .cloned()
                    .ok_or_else(|| bad("pickle: LONG_BINPUT on empty stack"))?;
                memo_put(&mut memo, idx, v)?;
            }
            0x94 => {
                let v = stack
                    .last()
                    .cloned()
                    .ok_or_else(|| bad("pickle: MEMOIZE on empty stack"))?;
                let idx = memo.len();
                memo_put(&mut memo, idx, v)?;
            }
            b'h' => {
                let idx = r.u8()? as usize;
                stack.push(memo_get(&memo, idx)?);
            }
            b'j' => {
                let idx = r.u32()? as usize;
                stack.push(memo_get(&memo, idx)?);
            }
            b'0' => {
                pop(&mut stack)?;
            }
            b'2' => {
                let v = stack
                    .last()
                    .cloned()
                    .ok_or_else(|| bad("pickle: DUP on empty stack"))?;
                stack.push(v);
            }
            _ => {
                return Err(Error::Model(alloc::format!(
                    "pickle: unsupported opcode 0x{op:02x} at byte {}",
                    r.pos - 1
                )));
            }
        }
    }
}
