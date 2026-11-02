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
//! TWO WAYS TO OPEN. `open` maps the whole file and hands out zero-copy slices —
//! right when the model fits in RAM. `open_unmapped` parses the header only
//! and reads tensor bytes on request with `pread` (`read_at` /
//! `tensor_extent`). That is the mode for models larger than RAM: a 508 GB
//! file mapped and touched would grow the page cache without bound, while
//! here memory is bounded by the header plus whatever the caller reads.
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

pub mod writer;
pub use writer::GgufWriter;

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

/// A byte range in the model file: ABSOLUTE offset (not data-relative) and
/// exact length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub offset: u64,
    pub len: u64,
}

impl Extent {
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }

    /// The largest power of two dividing the offset, capped at 4096.
    ///
    /// 4096 means the extent can be read with O_DIRECT as-is; anything less
    /// means the reader must widen the request to an aligned superset and
    /// discard the head. GGUF only promises `general.alignment` (32).
    pub fn alignment(&self) -> u64 {
        if self.offset == 0 {
            4096
        } else {
            (1u64 << self.offset.trailing_zeros()).min(4096)
        }
    }
}

/// How the tensor bytes are reached.
enum Storage {
    Mapped(memmap2::Mmap),
    /// Header parsed, data left on disk, read with `pread`.
    File {
        file: std::fs::File,
        len: u64,
    },
}

/// A parsed GGUF file, with its data still on disk.
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// File offset where the tensor-data section begins.
    pub data_offset: u64,
    pub alignment: u64,
    storage: Storage,
}

/// Everything the header says, independent of how the data is reached.
struct Header {
    version: u32,
    metadata: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
    alignment: u64,
}

/// First read for `open_unmapped`; doubled until the header fits. Real
/// headers are a few MB (the vocabulary dominates).
const HEADER_PROBE: usize = 1 << 20;

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
        let mut r = Reader::new(&map, map.len() as u64);
        let h = parse_header(&mut r)?;
        Ok(Self::from_header(h, Storage::Mapped(map)))
    }

    /// Parse the header without mapping the file. Tensor bytes are then read
    /// on demand with [`Gguf::read_at`]; [`Gguf::tensor_data`] is refused.
    ///
    /// Memory use is bounded by twice the header size, whatever the file size.
    pub fn open_unmapped(path: impl AsRef<Path>) -> Result<Self> {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(path.as_ref())?;
        let len = file.metadata()?.len();
        let mut want = HEADER_PROBE.min(usize::try_from(len).unwrap_or(usize::MAX));
        loop {
            let mut buf = vec![0u8; want];
            file.read_exact_at(&mut buf, 0)?;
            let mut r = Reader::new(&buf, len);
            match parse_header(&mut r) {
                Ok(h) => return Ok(Self::from_header(h, Storage::File { file, len })),
                // Ran off the end of the PREFIX, not the file: read more.
                Err(_) if r.ran_short && (want as u64) < len => {
                    want = want
                        .saturating_mul(2)
                        .min(usize::try_from(len).unwrap_or(usize::MAX));
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn from_header(h: Header, storage: Storage) -> Self {
        Gguf {
            version: h.version,
            metadata: h.metadata,
            tensors: h.tensors,
            data_offset: h.data_offset,
            alignment: h.alignment,
            storage,
        }
    }

    /// True when opened with [`Gguf::open`] (zero-copy slices available).
    pub fn is_mapped(&self) -> bool {
        matches!(self.storage, Storage::Mapped(_))
    }
}

fn parse_header(r: &mut Reader<'_>) -> Result<Header> {
    {
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
        let cap = r.file_len;
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
        if data_offset > r.file_len {
            return Err(Error::malformed(
                "gguf layout",
                format!(
                    "data section starts at {data_offset}, past the {}-byte file",
                    r.file_len
                ),
            ));
        }

        Ok(Header {
            version,
            metadata,
            tensors,
            data_offset,
            alignment,
        })
    }
}

impl Gguf {
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

    /// Where one tensor's bytes are in the file, checked against its size.
    ///
    /// The length comes from the element count and the block format, never
    /// from the next tensor's offset — padding between tensors is not data.
    pub fn tensor_extent(&self, t: &TensorInfo) -> Result<Extent> {
        let ty = llamd_quant::QuantType::from_ggml(t.ggml_type)?;
        let n = usize::try_from(t.elems())
            .map_err(|_| Error::malformed("tensor", format!("{} has too many elements", t.name)))?;
        let len = ty.bytes_for(n)? as u64;
        let offset = self
            .data_offset
            .checked_add(t.offset)
            .ok_or_else(|| Error::malformed("tensor", format!("{} offset overflows", t.name)))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::malformed("tensor", format!("{} extent overflows", t.name)))?;
        if end > self.file_size() as u64 {
            return Err(Error::malformed(
                "tensor",
                format!(
                    "{} spans {offset}..{end}, past the {}-byte file",
                    t.name,
                    self.file_size()
                ),
            ));
        }
        Ok(Extent { offset, len })
    }

    /// The raw bytes of one tensor, borrowed from the mapping.
    ///
    /// No copy: for a 23 GiB model, copying tensors to read them would defeat
    /// the point of mapping the file. Refused for an unmapped open — use
    /// [`Gguf::tensor_extent`] and [`Gguf::read_at`].
    pub fn tensor_data(&self, t: &TensorInfo) -> Result<&[u8]> {
        let e = self.tensor_extent(t)?;
        match &self.storage {
            Storage::Mapped(map) => Ok(&map[e.offset as usize..e.end() as usize]),
            Storage::File { .. } => Err(Error::unsupported(
                "tensor_data",
                format!(
                    "{}: file opened without mmap; read it with tensor_extent + read_at",
                    t.name
                ),
            )),
        }
    }

    /// Copy `buf.len()` bytes starting at absolute file offset `offset`.
    ///
    /// Works in both modes. Unmapped, this is a positioned read (`pread`), so
    /// concurrent readers need no shared cursor.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .filter(|&e| e <= self.file_size() as u64)
            .ok_or_else(|| {
                Error::malformed(
                    "read_at",
                    format!(
                        "{offset}+{} is past the {}-byte file",
                        buf.len(),
                        self.file_size()
                    ),
                )
            })?;
        match &self.storage {
            Storage::Mapped(map) => {
                buf.copy_from_slice(&map[offset as usize..end as usize]);
                Ok(())
            }
            Storage::File { file, .. } => {
                use std::os::unix::fs::FileExt;
                Ok(file.read_exact_at(buf, offset)?)
            }
        }
    }

    /// Read a whole extent into a new buffer.
    pub fn read_extent(&self, e: Extent) -> Result<Vec<u8>> {
        let len = usize::try_from(e.len)
            .map_err(|_| Error::malformed("read_extent", "length does not fit in usize"))?;
        let mut v = vec![0u8; len];
        self.read_at(e.offset, &mut v)?;
        Ok(v)
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
        match &self.storage {
            Storage::Mapped(map) => map.len(),
            Storage::File { len, .. } => *len as usize,
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
    /// Length of the whole FILE. `b` may be only a prefix of it.
    file_len: u64,
    /// Set when a read ran past `b` but not past the file: the prefix was
    /// too short, which is not corruption.
    ran_short: bool,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8], file_len: u64) -> Self {
        Reader {
            b,
            p: 0,
            file_len,
            ran_short: false,
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| Error::malformed("gguf", "read length overflows"))?;
        if end > self.b.len() {
            if (end as u64) <= self.file_len {
                self.ran_short = true;
            }
            return Err(Error::malformed(
                "gguf",
                format!(
                    "truncated: wanted {n} bytes at {}, file has {}",
                    self.p, self.file_len
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
                // Against the FILE, not the prefix: a long vocabulary array
                // may extend past a header probe and is still well-formed.
                let remaining = self.file_len.saturating_sub(self.p as u64);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A file in the temp dir that removes itself.
    struct TempFile(std::path::PathBuf);
    impl TempFile {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            TempFile(
                std::env::temp_dir()
                    .join(format!("llamd-gguf-{tag}-{}-{n}.gguf", std::process::id())),
            )
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn sample(vocab: usize) -> GgufWriter {
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("qwen3moe".into()));
        w.kv("qwen3moe.block_count", Value::U32(2));
        w.kv(
            "tokenizer.ggml.tokens",
            Value::Array(
                (0..vocab)
                    .map(|i| Value::String(format!("tok{i:06}")))
                    .collect(),
            ),
        );
        w.tensor("a", &[4, 3], 0, f32_bytes(&[1.0; 12]));
        w.tensor("b", &[5], 0, f32_bytes(&[2.0, 3.0, 4.0, 5.0, 6.0]));
        w.tensor(
            "c",
            &[8, 2, 3],
            0,
            f32_bytes(&(0..48).map(|i| i as f32).collect::<Vec<_>>()),
        );
        w
    }

    #[test]
    fn mapped_and_unmapped_agree_on_everything() {
        let f = TempFile::new("agree");
        sample(10).write(&f.0).unwrap();
        let m = Gguf::open(&f.0).unwrap();
        let u = Gguf::open_unmapped(&f.0).unwrap();
        assert!(m.is_mapped() && !u.is_mapped());
        assert_eq!(m.metadata, u.metadata);
        assert_eq!(m.data_offset, u.data_offset);
        assert_eq!(m.file_size(), u.file_size());
        for t in &m.tensors {
            let em = m.tensor_extent(t).unwrap();
            let eu = u.tensor_extent(u.tensor(&t.name).unwrap()).unwrap();
            assert_eq!(em, eu, "{}", t.name);
            // pread returns exactly the bytes the mapping exposes.
            assert_eq!(u.read_extent(eu).unwrap(), m.tensor_data(t).unwrap());
            assert_eq!(m.read_extent(em).unwrap(), m.tensor_data(t).unwrap());
        }
    }

    #[test]
    fn a_header_larger_than_the_first_probe_is_read_in_full() {
        // ~250k tokens of 12 bytes each: several MB of header, well past the
        // 1 MiB probe, so the doubling path is what parses it.
        let f = TempFile::new("bighdr");
        sample(250_000).write(&f.0).unwrap();
        let m = Gguf::open(&f.0).unwrap();
        let u = Gguf::open_unmapped(&f.0).unwrap();
        assert!(m.data_offset as usize > HEADER_PROBE);
        assert_eq!(m.metadata, u.metadata);
        let c = u.tensor("c").unwrap();
        assert_eq!(
            u.read_extent(u.tensor_extent(c).unwrap()).unwrap(),
            m.tensor_data(m.tensor("c").unwrap()).unwrap()
        );
    }

    #[test]
    fn unmapped_refuses_zero_copy_slices_instead_of_pretending() {
        let f = TempFile::new("refuse");
        sample(4).write(&f.0).unwrap();
        let u = Gguf::open_unmapped(&f.0).unwrap();
        let err = u.tensor_data(u.tensor("a").unwrap()).unwrap_err();
        assert!(format!("{err}").contains("without mmap"), "{err}");
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_loop() {
        let f = TempFile::new("trunc");
        let bytes = sample(50_000).to_bytes().unwrap();
        // Cut inside the vocabulary array: the header cannot complete.
        std::fs::write(&f.0, &bytes[..bytes.len() / 3]).unwrap();
        assert!(Gguf::open_unmapped(&f.0).is_err());
        assert!(Gguf::open(&f.0).is_err());
    }

    #[test]
    fn a_hostile_array_count_is_refused_in_both_modes() {
        let f = TempFile::new("hostile");
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // kv
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b'k');
        b.extend_from_slice(&9u32.to_le_bytes()); // array
        b.extend_from_slice(&0u32.to_le_bytes()); // of u8
        b.extend_from_slice(&(u64::MAX / 2).to_le_bytes()); // absurd count
        std::fs::write(&f.0, &b).unwrap();
        assert!(Gguf::open(&f.0).is_err());
        assert!(Gguf::open_unmapped(&f.0).is_err());
    }

    #[test]
    fn reads_past_the_end_are_refused() {
        let f = TempFile::new("past");
        sample(4).write(&f.0).unwrap();
        for g in [
            Gguf::open(&f.0).unwrap(),
            Gguf::open_unmapped(&f.0).unwrap(),
        ] {
            let n = g.file_size() as u64;
            let mut buf = [0u8; 8];
            assert!(g.read_at(n - 4, &mut buf).is_err());
            assert!(g.read_at(u64::MAX - 2, &mut buf).is_err());
            assert!(g.read_at(n - 8, &mut buf).is_ok());
        }
    }

    #[test]
    fn tensor_offsets_honour_the_declared_alignment() {
        let f = TempFile::new("align");
        let mut w = sample(3);
        w.alignment(4096);
        w.write(&f.0).unwrap();
        let g = Gguf::open_unmapped(&f.0).unwrap();
        assert_eq!(g.alignment, 4096);
        for t in &g.tensors {
            assert_eq!(g.tensor_extent(t).unwrap().alignment(), 4096, "{}", t.name);
        }
    }

    #[test]
    fn a_sparse_file_has_the_full_tensor_table_but_no_data_on_disk() {
        use std::os::unix::fs::MetadataExt;
        let f = TempFile::new("sparse");
        let mut w = sample(3);
        // 2 GiB of declared Q8_0 data that is never written.
        w.tensor_sparse("huge", &[32, 1 << 26], 8).unwrap();
        w.write(&f.0).unwrap();
        let u = Gguf::open_unmapped(&f.0).unwrap();
        let e = u.tensor_extent(u.tensor("huge").unwrap()).unwrap();
        assert_eq!(e.len, (1u64 << 26) * 34);
        assert_eq!(u.file_size() as u64, e.end());
        let blocks = std::fs::metadata(&f.0).unwrap().blocks() * 512;
        assert!(
            blocks < 16 << 20,
            "{blocks} bytes allocated for a sparse file"
        );
        // A hole reads as zeros; the real tensors keep their bytes.
        let mut z = [1u8; 64];
        u.read_at(e.offset + 12345, &mut z).unwrap();
        assert!(z.iter().all(|&b| b == 0));
        let a = u.tensor("a").unwrap();
        assert_eq!(
            u.read_extent(u.tensor_extent(a).unwrap()).unwrap()[..4],
            1f32.to_le_bytes()
        );
    }

    #[test]
    fn extent_alignment_is_the_largest_power_of_two_capped_at_4096() {
        let e = |offset| Extent { offset, len: 1 };
        assert_eq!(e(0).alignment(), 4096);
        assert_eq!(e(32).alignment(), 32);
        assert_eq!(e(96).alignment(), 32);
        assert_eq!(e(8192).alignment(), 4096);
        assert_eq!(e(4096 + 512).alignment(), 512);
        assert_eq!(e(7).alignment(), 1);
    }
}
