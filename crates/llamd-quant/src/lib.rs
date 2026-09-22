//! Quantized block formats, bit-exact with ggml.
//!
//! PROVENANCE. The block layouts and dequantization procedures here are
//! derived from llama.cpp / ggml, MIT licensed, © 2023-2026 The ggml authors:
//!
//!   - `ggml/src/ggml-common.h`  — `block_q6_K`, `block_q8_0` layout
//!   - `ggml/src/ggml-quants.c`  — `dequantize_row_q6_K`, `dequantize_row_q8_0`
//!
//! Reference revision: see `REFERENCE` at the repository root.
//!
//! WHY THIS CRATE IS A BIT-EXACT OBLIGATION AND NOT AN APPROXIMATION. These
//! are not numerical algorithms with acceptable error; they are a file format.
//! The bytes in a GGUF were produced by ggml's quantizer, and reading them any
//! other way produces different weights — not slightly different, arbitrarily
//! different. So `llamd-core::tolerance::EXACT` applies to everything here,
//! and the tests compare against values computed by hand from the layout.

#![forbid(unsafe_code)]

use llamd_core::{Error, Result};

/// Super-block size for the k-quant family. `QK_K` in ggml.
pub const QK_K: usize = 256;

/// Block size for Q8_0. `QK8_0` in ggml.
pub const QK8_0: usize = 32;

/// Serialized size of one `block_q6_K`, in bytes.
///
/// `ql[128] + qh[64] + scales[16] + d(f16)` = 210. ggml asserts this with a
/// `static_assert`; the equivalent here is [`tests::q6k_block_size_matches_ggml`].
pub const Q6K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

/// Serialized size of one `block_q8_0`: `d(f16) + qs[32]` = 34 bytes.
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

/// The quantization formats LLAMD can read.
///
/// The discriminants are ggml's `ggml_type` values, so a GGUF tensor's type
/// field maps directly. Using our own numbering would require a conversion
/// table that could drift from the file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q6K = 14,
}

impl QuantType {
    pub fn from_ggml(v: u32) -> Result<Self> {
        Ok(match v {
            0 => QuantType::F32,
            1 => QuantType::F16,
            8 => QuantType::Q8_0,
            14 => QuantType::Q6K,
            other => {
                return Err(Error::unsupported(
                    "ggml quantization type",
                    format!("type {other} is not implemented yet"),
                ));
            }
        })
    }

    /// Elements per serialized block. 1 for unquantized types.
    pub fn block_elems(self) -> usize {
        match self {
            QuantType::F32 | QuantType::F16 => 1,
            QuantType::Q8_0 => QK8_0,
            QuantType::Q6K => QK_K,
        }
    }

    /// Bytes per serialized block.
    pub fn block_bytes(self) -> usize {
        match self {
            QuantType::F32 => 4,
            QuantType::F16 => 2,
            QuantType::Q8_0 => Q8_0_BLOCK_BYTES,
            QuantType::Q6K => Q6K_BLOCK_BYTES,
        }
    }

    /// Bytes needed to store `n` elements of this type.
    pub fn bytes_for(self, n: usize) -> Result<usize> {
        let be = self.block_elems();
        if n % be != 0 {
            return Err(Error::malformed(
                "tensor length",
                format!("{n} elements is not a multiple of the {be}-element block for {self:?}"),
            ));
        }
        Ok(n / be * self.block_bytes())
    }
}

/// f16 → f32, matching `GGML_FP16_TO_FP32`.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

/// Dequantize `n` elements from `src`, appending to `out`.
pub fn dequantize(ty: QuantType, src: &[u8], n: usize, out: &mut Vec<f32>) -> Result<()> {
    let need = ty.bytes_for(n)?;
    if src.len() < need {
        return Err(Error::malformed(
            "quantized data",
            format!(
                "need {need} bytes for {n} elements of {ty:?}, have {}",
                src.len()
            ),
        ));
    }
    out.reserve(n);
    match ty {
        QuantType::F32 => {
            for i in 0..n {
                let o = i * 4;
                out.push(f32::from_le_bytes([
                    src[o],
                    src[o + 1],
                    src[o + 2],
                    src[o + 3],
                ]));
            }
        }
        QuantType::F16 => {
            for i in 0..n {
                out.push(f16_to_f32(rd_u16(src, i * 2)));
            }
        }
        QuantType::Q8_0 => dequantize_q8_0(src, n, out),
        QuantType::Q6K => dequantize_q6_k(src, n, out),
    }
    Ok(())
}

/// Q8_0: one f16 scale per 32 int8 quants.
///
/// Derived from `dequantize_row_q8_0` in `ggml/src/ggml-quants.c`.
fn dequantize_q8_0(src: &[u8], n: usize, out: &mut Vec<f32>) {
    for b in 0..(n / QK8_0) {
        let base = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(rd_u16(src, base));
        for j in 0..QK8_0 {
            out.push(d * (src[base + 2 + j] as i8) as f32);
        }
    }
}

/// Q6_K: 6 bits per weight, in super-blocks of 256.
///
/// Derived from `dequantize_row_q6_K` in `ggml/src/ggml-quants.c`. The
/// structure is preserved deliberately, including the `n += 128` outer step
/// and the sub-block stride, because this is a bit-exact obligation and a
/// tidier-looking rewrite is a chance to get it silently wrong.
///
/// Layout per 210-byte block:
/// ```text
///   ql[0..128]    low 4 bits of each weight
///   qh[128..192]  high 2 bits, packed 4 weights per byte
///   scales[192..208]  16 signed 8-bit sub-block scales
///   d[208..210]       f16 super-block scale
/// ```
/// Each weight is `(low4 | high2 << 4) - 32`, scaled by `d * scales[is]`.
/// The `- 32` centres the 6-bit value on zero; omitting it biases every
/// weight, which is the kind of error that still produces fluent output.
fn dequantize_q6_k(src: &[u8], n: usize, out: &mut Vec<f32>) {
    const QL: usize = 0;
    const QH: usize = QK_K / 2;
    const SC: usize = QK_K / 2 + QK_K / 4;
    const D: usize = SC + QK_K / 16;

    let start = out.len();
    out.resize(start + n, 0.0);

    for b in 0..(n / QK_K) {
        let base = b * Q6K_BLOCK_BYTES;
        let d = f16_to_f32(rd_u16(src, base + D));
        let blk = &src[base..base + Q6K_BLOCK_BYTES];
        let y0 = start + b * QK_K;

        // Two halves of 128 elements each; ggml walks them with moving
        // pointers, which is the same as these offsets.
        for half in 0..2 {
            let ql = QL + half * 64;
            let qh = QH + half * 32;
            let sc = SC + half * 8;
            let y = y0 + half * 128;

            for l in 0..32 {
                let is = l / 16;
                let h = blk[qh + l];
                let lo0 = blk[ql + l];
                let lo1 = blk[ql + l + 32];

                // `>> 0` is deliberate: these four lines mirror ggml's
                // source as one parallel group, and breaking the symmetry is
                // how a shift gets silently wrong in a bit-exact routine.
                #[allow(clippy::identity_op)]
                let q1 = ((lo0 & 0x0F) | ((h >> 0) & 3) << 4) as i8 - 32;
                let q2 = ((lo1 & 0x0F) | ((h >> 2) & 3) << 4) as i8 - 32;
                let q3 = ((lo0 >> 4) | ((h >> 4) & 3) << 4) as i8 - 32;
                let q4 = ((lo1 >> 4) | ((h >> 6) & 3) << 4) as i8 - 32;

                out[y + l] = d * (blk[sc + is] as i8) as f32 * q1 as f32;
                out[y + l + 32] = d * (blk[sc + is + 2] as i8) as f32 * q2 as f32;
                out[y + l + 64] = d * (blk[sc + is + 4] as i8) as f32 * q3 as f32;
                out[y + l + 96] = d * (blk[sc + is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ggml enforces these with `static_assert`. If they drift, every tensor
    /// read from a GGUF is misaligned and nothing downstream can be right.
    #[test]
    fn q6k_block_size_matches_ggml() {
        assert_eq!(Q6K_BLOCK_BYTES, 210);
        assert_eq!(Q6K_BLOCK_BYTES, 2 + QK_K / 16 + 3 * QK_K / 4);
        assert_eq!(QuantType::Q6K.block_bytes(), 210);
        assert_eq!(QuantType::Q6K.block_elems(), 256);
    }

    #[test]
    fn q8_0_block_size_matches_ggml() {
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
        assert_eq!(QuantType::Q8_0.block_bytes(), 34);
        assert_eq!(QuantType::Q8_0.block_elems(), 32);
    }

    #[test]
    fn ggml_type_numbers_are_not_invented() {
        // These must be ggml's own values: a GGUF stores the number, not a name.
        assert_eq!(QuantType::from_ggml(0).unwrap(), QuantType::F32);
        assert_eq!(QuantType::from_ggml(1).unwrap(), QuantType::F16);
        assert_eq!(QuantType::from_ggml(8).unwrap(), QuantType::Q8_0);
        assert_eq!(QuantType::from_ggml(14).unwrap(), QuantType::Q6K);
        // An unimplemented type must say so rather than guess.
        let e = QuantType::from_ggml(12).unwrap_err();
        assert!(matches!(e, Error::Unsupported { .. }), "got {e:?}");
    }

    #[test]
    fn q8_0_dequantizes_by_hand() {
        // d = 2.0, quants 1, -1, then zeros.
        let mut blk = vec![0u8; Q8_0_BLOCK_BYTES];
        blk[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        blk[2] = 1u8;
        blk[3] = (-1i8) as u8;

        let mut out = Vec::new();
        dequantize(QuantType::Q8_0, &blk, QK8_0, &mut out).unwrap();
        assert_eq!(out.len(), QK8_0);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -2.0);
        assert!(out[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6k_centres_on_minus_32() {
        // All quant bits zero means every weight is (0 - 32) = -32, not 0.
        // Forgetting the offset produces a model that still generates fluent
        // text, which is exactly why it needs its own test.
        let mut blk = vec![0u8; Q6K_BLOCK_BYTES];
        blk[QK_K / 2 + QK_K / 4] = 1; // scales[0] = 1
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        blk[d_off..d_off + 2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        let mut out = Vec::new();
        dequantize(QuantType::Q6K, &blk, QK_K, &mut out).unwrap();
        assert_eq!(out.len(), QK_K);
        assert_eq!(out[0], -32.0, "the -32 centring offset is missing");
    }

    #[test]
    fn q6k_reads_high_bits_from_qh() {
        // low nibble 5, high 2 bits 3 -> (5 | 3<<4) = 53, minus 32 = 21.
        // If qh were ignored the answer would be 5 - 32 = -27.
        let mut blk = vec![0u8; Q6K_BLOCK_BYTES];
        blk[0] = 0x05;
        blk[QK_K / 2] = 0b0000_0011;
        blk[QK_K / 2 + QK_K / 4] = 1;
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        blk[d_off..d_off + 2].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

        let mut out = Vec::new();
        dequantize(QuantType::Q6K, &blk, QK_K, &mut out).unwrap();
        assert_eq!(out[0], 21.0, "high 2 bits from qh were not applied");
    }

    #[test]
    fn short_input_is_refused_not_read_past() {
        let blk = vec![0u8; 10];
        let mut out = Vec::new();
        let e = dequantize(QuantType::Q6K, &blk, QK_K, &mut out).unwrap_err();
        assert!(matches!(e, Error::Malformed { .. }));
    }

    #[test]
    fn partial_blocks_are_refused() {
        // 100 elements is not a whole number of 256-element blocks.
        let e = QuantType::Q6K.bytes_for(100).unwrap_err();
        assert!(matches!(e, Error::Malformed { .. }));
        assert_eq!(QuantType::Q6K.bytes_for(512).unwrap(), 420);
    }
}
