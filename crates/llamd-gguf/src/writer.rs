//! A minimal GGUF v3 writer.
//!
//! For tests and tooling: synthetic models small enough to build in a unit
//! test, and files produced by preprocessing steps. It writes exactly the
//! layout `Gguf::open` reads (see the crate docs), with every tensor's data
//! padded to `general.alignment`, so a round trip exercises the real parser.
//!
//! It is not a converter. Tensor bytes are written as given; producing
//! correctly quantized blocks is the caller's business.
//!
//! SPARSE TENSORS. [`GgufWriter::tensor_sparse`] declares a tensor of full
//! size without supplying bytes; [`GgufWriter::write`] leaves a hole there, so
//! a file with the exact tensor table of a 500 GB model occupies a few MB of
//! disk. That is how out-of-core code is exercised at the target's real shape
//! on a machine that cannot store the target. Reads of a hole return zeros.

use crate::{DEFAULT_ALIGNMENT, MAGIC, Value};
use llamd_core::{Error, Result};
use std::path::Path;

pub struct GgufWriter {
    kv: Vec<(String, Value)>,
    tensors: Vec<PendingTensor>,
    alignment: u64,
}

struct PendingTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    /// `None`: sparse — `len` bytes of hole.
    data: Option<Vec<u8>>,
    len: u64,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        GgufWriter {
            kv: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
        }
    }

    /// Set `general.alignment`. Must be a power of two.
    pub fn alignment(&mut self, a: u64) -> &mut Self {
        self.alignment = a;
        self
    }

    pub fn kv(&mut self, key: &str, v: Value) -> &mut Self {
        self.kv.push((key.to_string(), v));
        self
    }

    /// Add a tensor. `dims` in ggml order (dims[0] fastest).
    pub fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, data: Vec<u8>) -> &mut Self {
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            len: data.len() as u64,
            data: Some(data),
        });
        self
    }

    /// Declare a tensor WITHOUT data. Its length is computed from the shape
    /// and block format, exactly as the reader will compute it.
    pub fn tensor_sparse(&mut self, name: &str, dims: &[u64], ggml_type: u32) -> Result<&mut Self> {
        let ty = llamd_quant::QuantType::from_ggml(ggml_type)?;
        let n: u64 = dims.iter().product();
        let len = ty.bytes_for(
            usize::try_from(n).map_err(|_| Error::malformed("gguf writer", "tensor too large"))?,
        )? as u64;
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            dims: dims.to_vec(),
            ggml_type,
            data: None,
            len,
        });
        Ok(self)
    }

    /// Header and tensor table, padded to the data section, plus each
    /// tensor's data-relative offset and the data section's total length.
    fn layout(&self) -> Result<(Vec<u8>, Vec<u64>, u64)> {
        let a = self.alignment;
        if a == 0 || !a.is_power_of_two() {
            return Err(Error::malformed(
                "gguf writer",
                format!("alignment {a} is not a power of two"),
            ));
        }
        let mut kv = self.kv.clone();
        if a != DEFAULT_ALIGNMENT && !kv.iter().any(|(k, _)| k == "general.alignment") {
            kv.push(("general.alignment".into(), Value::U32(a as u32)));
        }

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, v) in &kv {
            put_str(&mut out, k);
            out.extend_from_slice(&type_id(v)?.to_le_bytes());
            put_value(&mut out, v)?;
        }

        // Offsets are relative to the data section, each tensor aligned.
        let mut off = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            off = off.div_ceil(a) * a;
            offsets.push(off);
            off += t.len;
        }
        for (t, o) in self.tensors.iter().zip(&offsets) {
            put_str(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&o.to_le_bytes());
        }

        let data_start = (out.len() as u64).div_ceil(a) * a;
        out.resize(data_start as usize, 0);
        Ok((out, offsets, off))
    }

    /// The whole file in memory. Sparse tensors become zeros — use
    /// [`GgufWriter::write`] for anything large.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let (mut out, offsets, _) = self.layout()?;
        let data_start = out.len() as u64;
        for (t, o) in self.tensors.iter().zip(&offsets) {
            out.resize((data_start + o) as usize, 0);
            match &t.data {
                Some(d) => out.extend_from_slice(d),
                None => out.resize(out.len() + t.len as usize, 0),
            }
        }
        Ok(out)
    }

    /// Write to `path`, leaving holes for sparse tensors.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let (head, offsets, data_len) = self.layout()?;
        let f = std::fs::File::create(path)?;
        f.write_all_at(&head, 0)?;
        let data_start = head.len() as u64;
        for (t, o) in self.tensors.iter().zip(&offsets) {
            if let Some(d) = &t.data {
                f.write_all_at(d, data_start + o)?;
            }
        }
        f.set_len(data_start + data_len)?;
        Ok(())
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn type_id(v: &Value) -> Result<u32> {
    Ok(match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    })
}

fn put_value(out: &mut Vec<u8>, v: &Value) -> Result<()> {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => put_str(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(items) => {
            // The element type is declared once; mixed arrays are not GGUF.
            let inner = items.first().map(type_id).transpose()?.unwrap_or(4);
            if inner == 9 {
                return Err(Error::unsupported("gguf writer", "nested arrays"));
            }
            for it in items {
                if type_id(it)? != inner {
                    return Err(Error::malformed("gguf writer", "mixed-type array"));
                }
            }
            out.extend_from_slice(&inner.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                put_value(out, it)?;
            }
        }
    }
    Ok(())
}
