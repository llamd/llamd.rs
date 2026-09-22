//! The CPU reference backend.
//!
//! PROVENANCE. Numerical conventions follow llama.cpp / ggml (MIT, © 2023-2026
//! The ggml authors): `ggml/src/ggml-cpu/ops.cpp` for rms_norm, soft_max and
//! rope, `src/llama-graph.cpp` for how they are composed.
//!
//! CLARITY OVER SPEED, DELIBERATELY. This backend exists to be obviously
//! correct so that it can adjudicate a GPU bug. When a ROCm kernel and this
//! disagree, the question must be "what did the kernel do differently", never
//! "which of these two clever implementations is wrong". So: no SIMD
//! intrinsics, no manual unrolling, no fused shortcuts. It is allowed to be
//! slow. `llamd-cpu` is the only crate in the workspace whose performance is
//! explicitly a non-goal.
//!
//! The one thing it may NOT be is differently-rounded on purpose. Where the
//! order of a reduction is observable, it follows the reference's order.

#![forbid(unsafe_code)]

use llamd_core::{Error, Result};
use llamd_tensor::{Backend, RopeConfig, Shape, Tensor};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }

    /// Elementwise with a shape check, since every binary op needs both.
    fn zip(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor> {
        a.shape().same_as(b.shape())?;
        let data = a
            .data()
            .iter()
            .zip(b.data())
            .map(|(&x, &y)| f(x, y))
            .collect();
        Tensor::new(*a.shape(), data)
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu-reference"
    }

    /// `x / sqrt(mean(x^2) + eps) `, per row.
    ///
    /// NOTE: no learned weight here. ggml's `rms_norm` is the normalisation
    /// alone and the model graph multiplies by the weight afterwards. Folding
    /// the weight in would be faster and would stop this matching the
    /// reference op, which is the only thing it is for.
    fn rms_norm(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        let n = x.row_len();
        let mut out = Tensor::zeros(*x.shape());
        for r in 0..x.rows() {
            let src = x.row(r).ok_or_else(|| Error::other("row out of range"))?;
            // Sum of squares in f32, in order. Accumulating in f64 would be
            // more accurate and would stop matching the reference.
            let mut sum = 0.0f32;
            for &v in src {
                sum += v * v;
            }
            let scale = 1.0 / (sum / n as f32 + eps).sqrt();
            let dst = out
                .row_mut(r)
                .ok_or_else(|| Error::other("row out of range"))?;
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * scale;
            }
        }
        Ok(out)
    }

    /// Softmax per row, shifted by the row maximum.
    ///
    /// The shift is not an optimization: without it, attention logits of a few
    /// hundred overflow `exp` to inf and the row becomes NaN.
    fn softmax(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::zeros(*x.shape());
        for r in 0..x.rows() {
            let src = x.row(r).ok_or_else(|| Error::other("row out of range"))?;
            let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            let dst = out
                .row_mut(r)
                .ok_or_else(|| Error::other("row out of range"))?;
            for (d, &s) in dst.iter_mut().zip(src) {
                let e = (s - max).exp();
                *d = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for d in dst.iter_mut() {
                *d *= inv;
            }
        }
        Ok(out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.data().iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Tensor::new(*x.shape(), data)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::zip(a, b, |x, y| x + y)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Self::zip(a, b, |x, y| x * y)
    }

    fn scale(&self, x: &Tensor, by: f32) -> Result<Tensor> {
        Tensor::new(*x.shape(), x.data().iter().map(|v| v * by).collect())
    }

    /// `w` is `[k, m]`, `x` is `[k, n]`, result is `[m, n]`.
    ///
    /// Both operands are contiguous along `k`, so this is a dot product of two
    /// rows — no transpose, no strided access. That is exactly why ggml stores
    /// weights this way, and why the dimension convention in `llamd-tensor`
    /// matters more than it looks.
    fn matmul(&self, w: &Tensor, x: &Tensor) -> Result<Tensor> {
        let k = w.row_len();
        if x.row_len() != k {
            return Err(Error::Shape {
                expected: vec![k],
                got: vec![x.row_len()],
            });
        }
        let m = w.rows();
        let n = x.rows();
        let mut out = Tensor::zeros(Shape::new(&[m, n])?);
        for j in 0..n {
            let xr = x.row(j).ok_or_else(|| Error::other("x row out of range"))?;
            for i in 0..m {
                let wr = w.row(i).ok_or_else(|| Error::other("w row out of range"))?;
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += wr[t] * xr[t];
                }
                out.data_mut()[j * m + i] = acc;
            }
        }
        Ok(out)
    }

    /// Rotary position embedding, NeoX style — the pair for element `i` is
    /// `i + head_dim/2`, not `i + 1`.
    ///
    /// THE TWO CONVENTIONS ARE NOT INTERCHANGEABLE. The original GPT-NeoX
    /// layout rotates `(x[i], x[i + d/2])`; the "interleaved" layout rotates
    /// `(x[2i], x[2i+1])`. Both are called RoPE, both produce finite
    /// plausible numbers, and only one matches the weights. Qwen uses NeoX.
    fn rope(&self, x: &Tensor, positions: &[u32], cfg: RopeConfig) -> Result<Tensor> {
        let d = cfg.head_dim;
        if d % 2 != 0 {
            return Err(Error::malformed(
                "rope",
                format!("head_dim {d} must be even"),
            ));
        }
        if x.row_len() % d != 0 {
            return Err(Error::malformed(
                "rope",
                format!(
                    "row length {} is not a multiple of head_dim {d}",
                    x.row_len()
                ),
            ));
        }
        if positions.len() != x.rows() {
            return Err(Error::malformed(
                "rope",
                format!("{} positions for {} rows", positions.len(), x.rows()),
            ));
        }

        let heads = x.row_len() / d;
        let half = d / 2;
        let mut out = x.clone();
        for (r, &p) in positions.iter().enumerate() {
            let pos = p as f32 * cfg.freq_scale;
            let src = x
                .row(r)
                .ok_or_else(|| Error::other("row out of range"))?
                .to_vec();
            let dst = out
                .row_mut(r)
                .ok_or_else(|| Error::other("row out of range"))?;
            for h in 0..heads {
                let base = h * d;
                for i in 0..half {
                    // theta = pos / base^(2i/d)
                    let exponent = 2.0 * i as f32 / d as f32;
                    let theta = pos / cfg.freq_base.powf(exponent);
                    let (sin, cos) = theta.sin_cos();
                    let a = src[base + i];
                    let b = src[base + i + half];
                    dst[base + i] = a * cos - b * sin;
                    dst[base + i + half] = a * sin + b * cos;
                }
            }
        }
        Ok(out)
    }

    fn get_rows(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let n = table.row_len();
        let mut data = Vec::with_capacity(ids.len() * n);
        for &id in ids {
            let row = table.row(id as usize).ok_or_else(|| {
                Error::malformed(
                    "get_rows",
                    format!("id {id} is outside a table of {} rows", table.rows()),
                )
            })?;
            data.extend_from_slice(row);
        }
        Tensor::new(Shape::new(&[n, ids.len()])?, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llamd_core::tolerance::{ATTENTION, F32_ULP, close};

    fn t(dims: &[usize], v: &[f32]) -> Tensor {
        Tensor::from_slice(dims, v).unwrap()
    }

    #[test]
    fn rms_norm_matches_hand_computation() {
        let b = CpuBackend::new();
        // rms of [3,4] = sqrt((9+16)/2) = sqrt(12.5) = 3.5355
        let out = b.rms_norm(&t(&[2], &[3.0, 4.0]), 0.0).unwrap();
        let rms = (12.5f32).sqrt();
        assert!(close(out.data()[0], 3.0 / rms, F32_ULP));
        assert!(close(out.data()[1], 4.0 / rms, F32_ULP));
    }

    #[test]
    fn rms_norm_is_per_row_not_global() {
        // Two rows with very different magnitudes must normalise independently.
        // A global normalisation would leave the second row tiny.
        let b = CpuBackend::new();
        let out = b
            .rms_norm(&t(&[2, 2], &[3.0, 4.0, 300.0, 400.0]), 0.0)
            .unwrap();
        assert!(
            close(out.data()[0], out.data()[2], ATTENTION),
            "rows must normalise separately"
        );
        assert!(close(out.data()[1], out.data()[3], ATTENTION));
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_logits() {
        let b = CpuBackend::new();
        let out = b.softmax(&t(&[3], &[1.0, 2.0, 3.0])).unwrap();
        let sum: f32 = out.data().iter().sum();
        assert!(close(sum, 1.0, F32_ULP), "sum was {sum}");

        // Without the max shift these overflow to inf and the row is NaN.
        let big = b.softmax(&t(&[3], &[1000.0, 1001.0, 1002.0])).unwrap();
        assert!(
            big.data().iter().all(|v| v.is_finite()),
            "large logits overflowed"
        );
        assert!(close(big.data().iter().sum::<f32>(), 1.0, ATTENTION));
        // Shift-invariance: the same distribution, offset.
        for (a, c) in out.data().iter().zip(big.data()) {
            assert!(close(*a, *c, ATTENTION));
        }
    }

    #[test]
    fn silu_has_the_right_shape_at_known_points() {
        let b = CpuBackend::new();
        let out = b.silu(&t(&[3], &[0.0, 1.0, -1.0])).unwrap();
        assert!(close(out.data()[0], 0.0, F32_ULP));
        assert!(close(out.data()[1], 1.0 / (1.0 + (-1.0f32).exp()), F32_ULP));
        // Negative inputs stay negative but shrink — the property that makes
        // SiLU different from ReLU.
        assert!(out.data()[2] < 0.0 && out.data()[2] > -1.0);
    }

    #[test]
    fn matmul_is_a_row_dot_product_not_a_transpose() {
        let b = CpuBackend::new();
        // w = [k=2, m=3]: three rows of two. x = [k=2, n=1]: one row of two.
        let w = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let x = t(&[2, 1], &[1., 1.]);
        let out = b.matmul(&w, &x).unwrap();
        assert_eq!(out.shape().dims(), &[3, 1]);
        // Each output is the sum of a w row: 1+2, 3+4, 5+6.
        assert_eq!(out.data(), &[3.0, 7.0, 11.0]);
    }

    #[test]
    fn matmul_refuses_a_mismatched_contraction() {
        let b = CpuBackend::new();
        assert!(
            b.matmul(&t(&[2, 3], &[0.; 6]), &t(&[3, 1], &[0.; 3]))
                .is_err()
        );
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        // theta = 0 for every element, so cos=1 and sin=0.
        let b = CpuBackend::new();
        let x = t(&[4, 1], &[1., 2., 3., 4.]);
        let out = b.rope(&x, &[0], RopeConfig::new(4, 10000.0)).unwrap();
        for (a, c) in x.data().iter().zip(out.data()) {
            assert!(close(*a, *c, F32_ULP), "position 0 must not rotate");
        }
    }

    #[test]
    fn rope_pairs_across_the_half_boundary() {
        // NeoX pairs element i with i + d/2. For head_dim 4 that is (0,2) and
        // (1,3) — NOT (0,1) and (2,3). This test fails loudly if the
        // interleaved convention is used instead.
        let b = CpuBackend::new();
        let x = t(&[4, 1], &[1., 0., 0., 0.]);
        let cfg = RopeConfig::new(4, 10000.0);
        let out = b.rope(&x, &[1], cfg).unwrap();
        // Element 0 rotates with theta = 1/base^0 = 1 rad.
        assert!(close(out.data()[0], 1.0f32.cos(), ATTENTION));
        // Its partner is element 2, which receives sin(theta).
        assert!(
            close(out.data()[2], 1.0f32.sin(), ATTENTION),
            "pair must be i and i+d/2"
        );
        // Element 1's partner is 3, and both were zero.
        assert!(close(out.data()[1], 0.0, ATTENTION));
    }

    #[test]
    fn rope_preserves_the_norm_of_each_pair() {
        // A rotation cannot change length. This holds for any base, position
        // or input, which makes it a strong check on the whole routine.
        let b = CpuBackend::new();
        let x = t(&[8, 1], &[0.3, -1.2, 0.7, 2.0, -0.5, 1.1, 0.9, -0.2]);
        let out = b.rope(&x, &[37], RopeConfig::new(8, 10_000_000.0)).unwrap();
        for i in 0..4 {
            let before = x.data()[i].hypot(x.data()[i + 4]);
            let after = out.data()[i].hypot(out.data()[i + 4]);
            assert!(close(before, after, ATTENTION), "pair {i} changed length");
        }
    }

    #[test]
    fn get_rows_selects_and_reports_bad_ids() {
        let b = CpuBackend::new();
        let table = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let out = b.get_rows(&table, &[2, 0]).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2]);
        assert_eq!(out.data(), &[5., 6., 1., 2.]);
        // An out-of-range id must be an error, not a silent zero row: in MoE
        // routing that would be a different expert.
        assert!(b.get_rows(&table, &[9]).is_err());
    }
}
