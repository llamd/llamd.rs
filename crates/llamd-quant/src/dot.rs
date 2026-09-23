//! The quantized dot product: Q6_K weights against Q8_K activations.
//!
//! PROVENANCE. Derived from llama.cpp / ggml (MIT, © 2023-2026 The ggml
//! authors):
//!   - `ggml/src/ggml-quants.c` — `quantize_row_q8_K_ref`, `nearest_int`
//!   - `ggml/src/ggml-cpu/quants.c` — `ggml_vec_dot_q6_K_q8_K_generic`
//!
//! WHY THIS EXISTS. LLAMD's f32 reference dequantizes weights and multiplies
//! in f32. ggml does not: it quantizes the ACTIVATION to Q8_K and performs an
//! integer dot product. Those are different algorithms, and the difference is
//! small — until it meets a near-tied MoE routing decision, where it flips
//! which expert runs. Measured on this model, layer 47's router puts eight
//! experts within a factor of 2.4, the selection flips, and the emitted token
//! changes.
//!
//! So matching ggml at the token level requires performing the same
//! arithmetic, not a more accurate version of it. This module is that
//! arithmetic, reproduced deliberately — including the parts that look like
//! mistakes.
//!
//! THREE DETAILS THAT LOOK WRONG AND ARE NOT:
//!
//! 1. **The scale is negative.** `iscale = -127 / max`, where `max` is the
//!    SIGNED value at the largest magnitude, so the element of peak magnitude
//!    always maps to −127 whatever its sign.
//!
//!    A control showed this matters LESS than it looks for this path: `d` is
//!    `1/iscale`, so flipping the sign flips both `qs` and `d` and their
//!    product is unchanged — the dot product is identical either way. It is
//!    reproduced because other quantizations read `qs` directly and the
//!    upstream comment says the sign exists for IQ2_XXS's AVX path. Matching
//!    a convention costs nothing; diverging from one silently is how a
//!    format drifts.
//! 2. **The clamp is one-sided.** `MIN(127, v)` bounds above only. With the
//!    negative scale the minimum is already −127, so no lower clamp is needed.
//! 3. **Rounding is half-to-even.** `nearest_int` adds 1.5 × 2²³ to force the
//!    FPU's round-to-nearest-even, then extracts the mantissa. Rust's
//!    `f32::round` is half-away-from-zero and would disagree on every exact
//!    .5 — so `round_ties_even` is used.
//!
//!    Also a documented blind spot: swapping it for `round` does NOT change
//!    any token or fail the ggml comparison, because exact .5 values are
//!    vanishingly rare in real activations. Only the unit test below catches
//!    it, by asking directly. A stage-level property that end-to-end tests
//!    cannot see needs its own test — the same lesson the tokenizer's digit
//!    rule taught.

use crate::{Q6K_BLOCK_BYTES, QK_K};
use llamd_core::{Error, Result};

/// Serialized size of one `block_q8_K`: `d(f32) + qs[256] + bsums[16](i16)`.
pub const Q8K_BLOCK_BYTES: usize = 4 + QK_K + (QK_K / 16) * 2;

/// One quantized activation block, in the layout ggml uses.
#[derive(Debug, Clone)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    /// Sums of each group of 16 quants. Unused by the Q6_K dot product, but
    /// part of the format and required by other quantizations.
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        BlockQ8K {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        }
    }
}

/// ggml's `nearest_int`: round half to even.
///
/// The original adds 1.5 × 2²³ so the mantissa holds the rounded integer, then
/// reads it back — a way of asking the FPU for its default rounding mode.
/// `round_ties_even` is the same operation said directly.
#[inline]
fn nearest_int(v: f32) -> i32 {
    debug_assert!(
        v.abs() <= 4_194_303.0,
        "nearest_int is only valid within ±2^22"
    );
    v.round_ties_even() as i32
}

/// Quantize `x` (a multiple of 256 elements) to Q8_K blocks.
pub fn quantize_q8_k(x: &[f32]) -> Vec<BlockQ8K> {
    assert!(
        x.len() % QK_K == 0,
        "Q8_K quantizes whole 256-element blocks"
    );
    let mut out = Vec::with_capacity(x.len() / QK_K);

    for chunk in x.chunks_exact(QK_K) {
        let mut b = BlockQ8K::default();

        // The SIGNED value at the largest magnitude, not the largest value.
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if a > amax {
                amax = a;
                max = v;
            }
        }
        if amax == 0.0 {
            // An all-zero block has no scale. ggml leaves d = 0 and the
            // quants zero; anything else would divide by zero.
            out.push(b);
            continue;
        }

        let iscale = -127.0f32 / max;
        for (q, &v) in b.qs.iter_mut().zip(chunk) {
            // Clamped above only: the negative scale puts the peak at -127.
            *q = nearest_int(iscale * v).min(127) as i8;
        }
        for (g, bs) in b.bsums.iter_mut().enumerate() {
            *bs = b.qs[g * 16..(g + 1) * 16].iter().map(|&q| q as i16).sum();
        }
        b.d = 1.0 / iscale;
        out.push(b);
    }
    out
}

/// Dot product of Q6_K weights with Q8_K activations.
///
/// `w` is the packed Q6_K bytes for `n` elements; `a` is the matching
/// activation blocks. The accumulation order follows the reference exactly:
/// eight parallel i32 lanes per block, scaled per 16-element sub-block, summed
/// into eight f32 lanes, and only then added together. A different order is a
/// different answer in the last bits, and the last bits are what decide a
/// near-tied routing choice.
pub fn dot_q6k_q8k(w: &[u8], a: &[BlockQ8K], n: usize) -> f32 {
    assert!(n % QK_K == 0, "the dot product runs over whole blocks");
    let nb = n / QK_K;
    assert!(w.len() >= nb * Q6K_BLOCK_BYTES, "weight buffer is short");
    assert!(a.len() >= nb, "activation buffer is short");

    const QL: usize = 0;
    const QH: usize = QK_K / 2;
    const SC: usize = QK_K / 2 + QK_K / 4;
    const D: usize = SC + QK_K / 16;

    let mut sums = [0.0f32; 8];

    for i in 0..nb {
        let blk = &w[i * Q6K_BLOCK_BYTES..(i + 1) * Q6K_BLOCK_BYTES];
        let y = &a[i];

        // Unpack the 6-bit weights to signed bytes, centred on -32. Same
        // structure as `dequantize_q6_k`, but stopping at integers: the whole
        // point is that no f32 multiply happens here.
        let mut aux8 = [0i8; QK_K];
        for half in 0..2 {
            let ql = QL + half * 64;
            let qh = QH + half * 32;
            let o = half * 128;
            for l in 0..32 {
                let h = blk[qh + l];
                let lo0 = blk[ql + l];
                let lo1 = blk[ql + l + 32];
                aux8[o + l] = ((lo0 & 0x0F) | ((h) & 3) << 4) as i8 - 32;
                aux8[o + l + 32] = ((lo1 & 0x0F) | ((h >> 2) & 3) << 4) as i8 - 32;
                aux8[o + l + 64] = ((lo0 >> 4) | ((h >> 4) & 3) << 4) as i8 - 32;
                aux8[o + l + 96] = ((lo1 >> 4) | ((h >> 6) & 3) << 4) as i8 - 32;
            }
        }

        // Eight i32 lanes, one scale per 16 elements, two 8-wide passes each.
        let mut aux32 = [0i32; 8];
        for j in 0..(QK_K / 16) {
            let scale = blk[SC + j] as i8 as i32;
            for half in 0..2 {
                let o = j * 16 + half * 8;
                for l in 0..8 {
                    // i16 intermediate, exactly as the reference: the product
                    // of two int8 values cannot overflow it, and widening
                    // earlier would change nothing but is worth not risking.
                    let p = (y.qs[o + l] as i16) * (aux8[o + l] as i16);
                    aux32[l] += scale * p as i32;
                }
            }
        }

        let d = crate::f16_to_f32(u16::from_le_bytes([blk[D], blk[D + 1]])) * y.d;
        for (s, &v) in sums.iter_mut().zip(&aux32) {
            *s += d * v as f32;
        }
    }
    sums.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantType, dequantize};

    #[test]
    fn block_size_matches_ggml() {
        // d(4) + qs(256) + bsums(16 * 2) = 292
        assert_eq!(Q8K_BLOCK_BYTES, 292);
    }

    #[test]
    fn rounding_is_half_to_even_not_half_away_from_zero() {
        // The difference shows only on exact .5, and only in the last bit of
        // one quant — which is precisely enough to flip a near-tied router.
        assert_eq!(nearest_int(0.5), 0, "0.5 must round to 0, not 1");
        assert_eq!(nearest_int(1.5), 2);
        assert_eq!(nearest_int(2.5), 2, "2.5 must round to 2, not 3");
        assert_eq!(nearest_int(-0.5), 0);
        assert_eq!(nearest_int(-1.5), -2);
        assert_eq!(nearest_int(-2.5), -2);
        // Rust's own round would disagree on three of those.
        assert_ne!(nearest_int(2.5), 2.5f32.round() as i32);
    }

    #[test]
    fn the_peak_magnitude_maps_to_minus_127() {
        // True whatever its sign, because iscale is negative.
        let mut v = vec![0.1f32; QK_K];
        v[7] = 5.0;
        assert_eq!(quantize_q8_k(&v)[0].qs[7], -127);

        let mut v = vec![0.1f32; QK_K];
        v[7] = -5.0;
        assert_eq!(
            quantize_q8_k(&v)[0].qs[7],
            -127,
            "a negative peak also maps to -127"
        );
    }

    #[test]
    fn an_all_zero_block_has_no_scale() {
        let b = &quantize_q8_k(&vec![0.0f32; QK_K])[0];
        assert_eq!(b.d, 0.0);
        assert!(b.qs.iter().all(|&q| q == 0));
        assert!(b.d.is_finite(), "must not divide by zero");
    }

    #[test]
    fn bsums_are_group_sums_of_sixteen() {
        let v: Vec<f32> = (0..QK_K).map(|i| (i as f32 * 0.01).sin()).collect();
        let b = &quantize_q8_k(&v)[0];
        for g in 0..QK_K / 16 {
            let want: i16 = b.qs[g * 16..(g + 1) * 16].iter().map(|&q| q as i16).sum();
            assert_eq!(b.bsums[g], want);
        }
    }

    /// Build one Q6_K block with a known scale and given 6-bit codes.
    fn q6k_block(codes: &[u8; QK_K], scale: i8, d: f32) -> Vec<u8> {
        let mut blk = vec![0u8; Q6K_BLOCK_BYTES];
        for half in 0..2 {
            let ql = half * 64;
            let qh = QK_K / 2 + half * 32;
            let o = half * 128;
            for l in 0..32 {
                let (c0, c1, c2, c3) = (
                    codes[o + l],
                    codes[o + l + 32],
                    codes[o + l + 64],
                    codes[o + l + 96],
                );
                blk[ql + l] = (c0 & 0x0F) | ((c2 & 0x0F) << 4);
                blk[ql + l + 32] = (c1 & 0x0F) | ((c3 & 0x0F) << 4);
                blk[qh + l] = (c0 >> 4) | ((c1 >> 4) << 2) | ((c2 >> 4) << 4) | ((c3 >> 4) << 6);
            }
        }
        for j in 0..QK_K / 16 {
            blk[QK_K / 2 + QK_K / 4 + j] = scale as u8;
        }
        let d_off = QK_K / 2 + QK_K / 4 + QK_K / 16;
        blk[d_off..d_off + 2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        blk
    }

    #[test]
    fn the_dot_product_agrees_with_dequantize_then_multiply() {
        // Not bit-exact — that is the whole point, they are different
        // algorithms — but they must agree to the accuracy Q8_K allows. A
        // large disagreement would mean the unpacking or the scaling is wrong.
        let mut codes = [32u8; QK_K]; // 32 - 32 = 0
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (32 + (i as i32 % 7) - 3) as u8;
        }
        let w = q6k_block(&codes, 2, 0.5);

        let x: Vec<f32> = (0..QK_K).map(|i| ((i as f32) * 0.05).sin()).collect();
        let a = quantize_q8_k(&x);
        let quantized = dot_q6k_q8k(&w, &a, QK_K);

        let mut wf = Vec::new();
        dequantize(QuantType::Q6K, &w, QK_K, &mut wf).unwrap();
        let exact: f32 = wf.iter().zip(&x).map(|(a, b)| a * b).sum();

        // These are DIFFERENT ALGORITHMS, so they cannot be bit-equal. This
        // input also sums 256 terms down to about -0.27 — near-total
        // cancellation, which inflates any measure relative to the result.
        // Scaling by the magnitude of the terms is the honest comparison.
        //
        // The tight check belongs elsewhere: `quantized_dot_matches_ggml` in
        // llamd-testing compares this function against ggml's own and holds
        // it to 2e-7.
        let scale: f32 = wf
            .iter()
            .zip(&x)
            .map(|(a, b)| (a * b).abs())
            .sum::<f32>()
            .max(1e-6);
        let rel = (quantized - exact).abs() / scale;
        assert!(
            rel < 5e-3,
            "quantized {quantized} vs f32 {exact}: {rel:.2e} of the term magnitude"
        );
    }

    #[test]
    fn a_zero_weight_block_gives_zero() {
        // Every code 32 means every weight is exactly 0 after the -32 offset.
        let w = q6k_block(&[32u8; QK_K], 1, 1.0);
        let x: Vec<f32> = (0..QK_K).map(|i| i as f32).collect();
        let a = quantize_q8_k(&x);
        assert_eq!(dot_q6k_q8k(&w, &a, QK_K), 0.0);
    }

    #[test]
    fn multiple_blocks_accumulate() {
        let w1 = q6k_block(&[33u8; QK_K], 1, 1.0); // weight = +1 * scale * d
        let mut w = w1.clone();
        w.extend_from_slice(&w1);
        let x = vec![1.0f32; 2 * QK_K];
        let a = quantize_q8_k(&x);
        let one = dot_q6k_q8k(&w1, &a[..1], QK_K);
        let two = dot_q6k_q8k(&w, &a, 2 * QK_K);
        assert!(
            (two - 2.0 * one).abs() < 1e-3 * one.abs().max(1.0),
            "{two} vs 2x{one}"
        );
    }
}

/// Byte offsets of the four planes in a planar-repacked Q6_K tensor.
///
/// WHY REPACK AT ALL. ggml stores Q6_K as 210-byte blocks: 128 bytes of low
/// nibbles, 64 of high bits, 16 int8 scales, one f16. 210 is not a multiple of
/// 4, so block `b` sits at `b * 210` and is only 2-byte aligned — every block
/// after the first has a different alignment mod 16, and a 16-byte vector load
/// is impossible. The kernel is forced into single-byte loads.
///
/// That costs most of the card. Measured on this R9700: a cold streaming read
/// reaches 637.8 GB/s with `uint4` (16-byte) loads — 100% of the 640 GB/s the
/// memory spec allows — while the Q6_K matvec sustains 259 GB/s in the engine.
///
/// Splitting the four components into separate planes fixes the alignment
/// without padding: each plane's per-block stride (128, 64, 16, 2) divides
/// evenly, so `ql`, `qh` and `sc` are all 16-byte aligned everywhere and can be
/// read as `uint4`. Total bytes are unchanged — this is a permutation, not an
/// expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanarQ6K {
    /// Low nibbles: `rows * blocks * 128` bytes at offset 0.
    pub ql: usize,
    /// High bits: `rows * blocks * 64`.
    pub qh: usize,
    /// Per-group int8 scales: `rows * blocks * 16`.
    pub sc: usize,
    /// Per-block f16 super-scale: `rows * blocks * 2`.
    pub d: usize,
    pub total: usize,
}

impl PlanarQ6K {
    pub fn layout(rows: usize, blocks: usize) -> PlanarQ6K {
        let n = rows * blocks;
        let ql = 0;
        let qh = ql + n * (QK_K / 2);
        let sc = qh + n * (QK_K / 4);
        let d = sc + n * (QK_K / 16);
        PlanarQ6K {
            ql,
            qh,
            sc,
            d,
            total: d + n * 2,
        }
    }
}

/// Repack interleaved Q6_K blocks into the four planes above.
///
/// Pure permutation: every byte of the input appears exactly once in the
/// output, at a different offset. `planar_repack_is_a_permutation` asserts that.
pub fn repack_q6k_planar(src: &[u8], rows: usize, blocks: usize) -> Result<Vec<u8>> {
    let n = rows * blocks;
    let want = n * Q6K_BLOCK_BYTES;
    if src.len() < want {
        return Err(Error::Shape {
            expected: vec![want],
            got: vec![src.len()],
        });
    }
    let l = PlanarQ6K::layout(rows, blocks);
    let mut out = vec![0u8; l.total];
    for i in 0..n {
        let b = &src[i * Q6K_BLOCK_BYTES..(i + 1) * Q6K_BLOCK_BYTES];
        out[l.ql + i * (QK_K / 2)..l.ql + (i + 1) * (QK_K / 2)].copy_from_slice(&b[..QK_K / 2]);
        out[l.qh + i * (QK_K / 4)..l.qh + (i + 1) * (QK_K / 4)]
            .copy_from_slice(&b[QK_K / 2..QK_K / 2 + QK_K / 4]);
        let sc0 = QK_K / 2 + QK_K / 4;
        out[l.sc + i * (QK_K / 16)..l.sc + (i + 1) * (QK_K / 16)]
            .copy_from_slice(&b[sc0..sc0 + QK_K / 16]);
        let d0 = sc0 + QK_K / 16;
        out[l.d + i * 2..l.d + i * 2 + 2].copy_from_slice(&b[d0..d0 + 2]);
    }
    Ok(out)
}

#[cfg(test)]
mod planar_tests {
    use super::*;

    #[test]
    fn planar_repack_is_a_permutation() {
        // Every input byte must appear once in the output. A repack that loses
        // or duplicates a byte would change weights, and the symptom would be
        // degraded text rather than an error.
        let (rows, blocks) = (7usize, 5usize);
        let mut src = vec![0u8; rows * blocks * Q6K_BLOCK_BYTES];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let out = repack_q6k_planar(&src, rows, blocks).expect("repack");
        let l = PlanarQ6K::layout(rows, blocks);
        assert_eq!(out.len(), src.len(), "repack must not change the size");
        assert_eq!(l.total, src.len());

        let mut hist_in = [0usize; 256];
        let mut hist_out = [0usize; 256];
        for &b in &src {
            hist_in[b as usize] += 1;
        }
        for &b in &out {
            hist_out[b as usize] += 1;
        }
        assert_eq!(hist_in, hist_out, "byte multiset changed");

        // And spot-check that a specific block landed where the layout says.
        let i = 3usize;
        let blk = &src[i * Q6K_BLOCK_BYTES..(i + 1) * Q6K_BLOCK_BYTES];
        assert_eq!(&out[l.ql + i * 128..l.ql + i * 128 + 128], &blk[..128]);
        assert_eq!(&out[l.qh + i * 64..l.qh + i * 64 + 64], &blk[128..192]);
        assert_eq!(&out[l.sc + i * 16..l.sc + i * 16 + 16], &blk[192..208]);
        assert_eq!(&out[l.d + i * 2..l.d + i * 2 + 2], &blk[208..210]);
    }

    #[test]
    fn planar_layout_keeps_every_plane_16_byte_aligned() {
        // The whole point: uint4 loads need 16-byte alignment, and the
        // interleaved 210-byte layout cannot provide it.
        for (rows, blocks) in [(1usize, 1usize), (4096, 8), (2048, 16), (151936, 8)] {
            let l = PlanarQ6K::layout(rows, blocks);
            assert_eq!(l.ql % 16, 0, "ql plane");
            assert_eq!(l.qh % 16, 0, "qh plane {rows}x{blocks}");
            assert_eq!(l.sc % 16, 0, "sc plane {rows}x{blocks}");
            // Per-block strides are what the kernel indexes by.
            assert_eq!((QK_K / 2) % 16, 0);
            assert_eq!((QK_K / 4) % 16, 0);
            assert_eq!((QK_K / 16) % 16, 0);
        }
    }

    #[test]
    fn a_short_source_is_refused() {
        assert!(repack_q6k_planar(&[0u8; 10], 1, 1).is_err());
    }
}
