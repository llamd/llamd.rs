//! GGUF parsing: header, metadata, tensor descriptors, memory-mapped data.
//!
//! PROVENANCE. The format is defined by llama.cpp / ggml (MIT, © 2023-2026
//! The ggml authors), `ggml/include/gguf.h` and `ggml/src/gguf.cpp`. This is a
//! reimplementation from the format, not a translation of the parser.
//!
//! ```text
//!   magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
//!   kv pairs     : name(string) type(u32) value
//!   tensor infos : name(string) n_dims(u32) dims[n_dims](u64) type(u32) offset(u64)
//!   padding to `general.alignment` (default 32)
//!   tensor data
//! ```
//!
//! Strings are `len: u64` followed by unterminated UTF-8.
//!
//! EVERY LENGTH IN THIS FILE IS ATTACKER-CONTROLLED. A GGUF is a binary blob
//! that may be truncated, corrupt, or hostile, and it declares its own array
//! counts as u64. A parser that trusts them will try to allocate 2^64 elements
//! on a one-byte file. So every count is checked against the remaining bytes
//! before anything is reserved.

// This crate is ONE OF FOUR in the workspace permitted to use unsafe, and it
// uses it exactly once: the mmap in `Gguf::open`. `deny` rather than `forbid`
// so that single call can be allowed with its justification attached — and
// every other use in this crate is still a compile error.
#![deny(unsafe_code)]

use llamd_core::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata value. Mirrors `gguf_type`; the discriminants are ggml's.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Value::U8(v) => *v as u64,
            Value::U16(v) => *v as u64,
            Value::U32(v) => *v as u64,
            Value::U64(v) => *v,
            Value::I8(v) if *v >= 0 => *v as u64,
            Value::I16(v) if *v >= 0 => *v as u64,
            Value::I32(v) if *v >= 0 => *v as u64,
            Value::I64(v) if *v >= 0 => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
    /// A short description for dumps — arrays print their length, not 150k
    /// tokens.
    pub fn summary(&self) -> String {
        match self {
            Value::Array(a) => format!("[{} items]", a.len()),
            Value::String(s) if s.len() > 48 => format!("{:?}…", &s[..48]),
            v => format!("{v:?}"),
        }
    }
}

/// Where a tensor's data lives and how it is shaped.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the fastest-varying axis.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset from the start of the tensor-data section, not the file.
    pub offset: u64,
}

impl TensorInfo {
    pub fn elems(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// A parsed GGUF file, with its data still on disk.
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// File offset where the tensor-data section begins.
    pub data_offset: u64,
    pub alignment: u64,
    map: memmap2::Mmap,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        // SAFETY: mmap of a file we opened read-only. The documented hazard is
        // another process truncating the file while it is mapped, which would
        // turn reads into SIGBUS. LLAMD accepts that for model files, which
        // are not expected to change under a running process — the same
        // assumption llama.cpp makes. This is one of the four places in the
        // workspace where unsafe is permitted, and it is confined to this call.
        #[allow(unsafe_code)]
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Self::parse(map)
    }

    fn parse(map: memmap2::Mmap) -> Result<Self> {
        let mut r = Reader { b: &map, p: 0 };

        let magic = r.take(4)?;
        if magic != MAGIC {
            return Err(Error::malformed(
                "gguf header",
                format!("magic is {magic:?}, expected {MAGIC:?}"),
            ));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(Error::unsupported(
                "gguf version",
                format!("version {version}; this parser implements 3"),
            ));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;

        // A count larger than the file can possibly describe is corruption,
        // and checking it here is what stops a hostile file from making us
        // reserve gigabytes. The smallest possible entry is well over 8 bytes,
        // so file length is a sound upper bound.
        let cap = map.len() as u64;
        if n_tensors > cap || n_kv > cap {
            return Err(Error::malformed(
                "gguf header",
                format!("{n_kv} kv / {n_tensors} tensors exceeds the {cap}-byte file"),
            ));
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let val = r.value(ty)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(n_tensors.min(4096) as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims > 4 {
                return Err(Error::malformed(
                    "tensor info",
                    format!("{name} claims {n_dims} dimensions; ggml allows at most 4"),
                ));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::malformed(
                "general.alignment",
                format!("{alignment} is not a power of two"),
            ));
        }

        // Tensor data starts at the next alignment boundary after the header.
        let pos = r.p as u64;
        let data_offset = pos.div_ceil(alignment) * alignment;
        if data_offset > map.len() as u64 {
            return Err(Error::malformed(
                "gguf layout",
                format!(
                    "data section starts at {data_offset}, past the {}-byte file",
                    map.len()
                ),
            ));
        }

        Ok(Gguf {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
            map,
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// `general.architecture` — the key that decides which model code runs.
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(Value::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw bytes of one tensor, borrowed from the mapping.
    ///
    /// No copy: for a 23 GiB model, copying tensors to read them would defeat
    /// the point of mapping the file.
    pub fn tensor_data(&self, t: &TensorInfo) -> Result<&[u8]> {
        let ty = llamd_quant::QuantType::from_ggml(t.ggml_type)?;
        let n = usize::try_from(t.elems())
            .map_err(|_| Error::malformed("tensor", format!("{} has too many elements", t.name)))?;
        let len = ty.bytes_for(n)?;
        let start = usize::try_from(self.data_offset + t.offset)
            .map_err(|_| Error::malformed("tensor", "offset does not fit in usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::malformed("tensor", format!("{} extent overflows", t.name)))?;
        if end > self.map.len() {
            return Err(Error::malformed(
                "tensor",
                format!(
                    "{} spans {start}..{end}, past the {}-byte file",
                    t.name,
                    self.map.len()
                ),
            ));
        }
        Ok(&self.map[start..end])
    }

    /// Dequantize a tensor to f32.
    pub fn dequantize(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let ty = llamd_quant::QuantType::from_ggml(t.ggml_type)?;
        let n = t.elems() as usize;
        let mut out = Vec::with_capacity(n);
        llamd_quant::dequantize(ty, self.tensor_data(t)?, n, &mut out)?;
        Ok(out)
    }

    pub fn file_size(&self) -> usize {
        self.map.len()
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| Error::malformed("gguf", "read length overflows"))?;
        if end > self.b.len() {
            return Err(Error::malformed(
                "gguf",
                format!(
                    "truncated: wanted {n} bytes at {}, file has {}",
                    self.p,
                    self.b.len()
                ),
            ));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        let n = usize::try_from(n)
            .map_err(|_| Error::malformed("gguf string", "length does not fit in usize"))?;
        let s = self.take(n)?;
        // Lossy on purpose: a single bad byte in one metadata string should
        // not make a 23 GiB model unopenable.
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()?;
                if inner == 9 {
                    return Err(Error::unsupported("gguf array", "nested arrays"));
                }
                // The remaining-bytes check is what prevents a declared count
                // of 2^63 from reserving until the process dies.
                let remaining = (self.b.len() - self.p) as u64;
                if n > remaining {
                    return Err(Error::malformed(
                        "gguf array",
                        format!("{n} elements declared, only {remaining} bytes remain"),
                    ));
                }
                let mut v = Vec::with_capacity(n.min(65536) as usize);
                for _ in 0..n {
                    v.push(self.value(inner)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => {
                return Err(Error::malformed(
                    "gguf value",
                    format!("unknown type {other}"),
                ));
            }
        })
    }
}
