//! Grouped-query attention, the CPU reference.
//!
//! PROVENANCE. Follows the composition in `llm_graph_context::build_attn` in
//! `llama.cpp/src/llama-graph.cpp` (MIT, © 2023-2026 The ggml authors).
//!
//! GROUPED QUERY, NOT MULTI-HEAD. This model has 32 query heads and 4 KV
//! heads, so eight query heads SHARE each KV head. The mapping is
//! `kv_head = q_head / gqa_ratio` — integer division, so heads 0..7 all read
//! KV head 0.
//!
//! Getting that mapping wrong (say `q_head % n_head_kv`) still runs, still
//! produces finite attention weights, and simply attends to the wrong keys.
//! It has its own test for exactly that reason.
//!
//! THE SCALE IS `1/sqrt(head_dim)`, applied to the logits before softmax.
//! Omitting it makes the softmax far too peaked and the model degenerate — a
//! failure that looks like a bad checkpoint rather than a missing constant.
//!
//! CAUSAL MASKING is implicit here: a query at position `p` attends to keys
//! `0..=p` only, because that is the whole KV cache at the moment it is
//! computed. A separate mask is only needed once several tokens are processed
//! in one batch, which is `llamd-runtime`'s next piece of work.

use llamd_core::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct AttnParams {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
}

impl AttnParams {
    /// Query heads per KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.n_head / self.n_head_kv
    }
    pub fn validate(&self) -> Result<()> {
        if self.n_head_kv == 0 || self.n_head % self.n_head_kv != 0 {
            return Err(Error::malformed(
                "attention",
                format!(
                    "{} query heads do not group into {} kv heads",
                    self.n_head, self.n_head_kv
                ),
            ));
        }
        if self.head_dim == 0 {
            return Err(Error::malformed("attention", "head_dim is zero"));
        }
        Ok(())
    }
}

/// Attend one query position against `n_kv` cached positions.
///
/// * `q` — `[n_head * head_dim]`, this position's queries
/// * `k`, `v` — `[n_kv * n_head_kv * head_dim]`, the cache in position order
///
/// Returns `[n_head * head_dim]`, the per-head context vectors concatenated in
/// head order, ready for the output projection.
pub fn attend(q: &[f32], k: &[f32], v: &[f32], n_kv: usize, p: AttnParams) -> Result<Vec<f32>> {
    p.validate()?;
    let AttnParams {
        n_head,
        n_head_kv,
        head_dim,
    } = p;
    let kv_dim = n_head_kv * head_dim;

    if q.len() != n_head * head_dim {
        return Err(Error::Shape {
            expected: vec![n_head * head_dim],
            got: vec![q.len()],
        });
    }
    if k.len() != n_kv * kv_dim || v.len() != n_kv * kv_dim {
        return Err(Error::Shape {
            expected: vec![n_kv * kv_dim],
            got: vec![k.len().min(v.len())],
        });
    }
    if n_kv == 0 {
        return Err(Error::malformed("attention", "nothing to attend to"));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let ratio = p.gqa_ratio();
    let mut out = vec![0.0f32; n_head * head_dim];
    let mut logits = vec![0.0f32; n_kv];

    for h in 0..n_head {
        // Integer division: eight consecutive query heads share one KV head.
        let kvh = h / ratio;
        let qo = h * head_dim;

        for (t, logit) in logits.iter_mut().enumerate() {
            let ko = t * kv_dim + kvh * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[qo + d] * k[ko + d];
            }
            *logit = dot * scale;
        }

        // Softmax over positions, shifted: attention logits reach the tens and
        // exp overflows without it.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for l in logits.iter_mut() {
            *l = (*l - max).exp();
            sum += *l;
        }
        let inv = 1.0 / sum;

        for (t, &w) in logits.iter().enumerate() {
            let w = w * inv;
            let vo = t * kv_dim + kvh * head_dim;
            for d in 0..head_dim {
                out[qo + d] += w * v[vo + d];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llamd_core::tolerance::{ATTENTION, close};

    fn p(n_head: usize, n_head_kv: usize, head_dim: usize) -> AttnParams {
        AttnParams {
            n_head,
            n_head_kv,
            head_dim,
        }
    }

    #[test]
    fn a_single_position_returns_that_value_exactly() {
        // With one key, softmax gives weight 1 and the output IS the value.
        // Anything else means the weighting is wrong.
        let q = vec![0.5; 4];
        let k = vec![0.25; 4];
        let v = vec![7.0, 8.0, 9.0, 10.0];
        let out = attend(&q, &k, &v, 1, p(1, 1, 4)).unwrap();
        for (o, e) in out.iter().zip(&v) {
            assert!(close(*o, *e, ATTENTION), "got {o}, want {e}");
        }
    }

    #[test]
    fn attention_weights_form_a_convex_combination() {
        // The output must lie between the smallest and largest value, because
        // the weights are non-negative and sum to one.
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let v = vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0];
        let out = attend(&q, &k, &v, 3, p(1, 1, 2)).unwrap();
        assert!(out[0] > 10.0 && out[0] < 30.0, "got {}", out[0]);
        // The key most aligned with q is the first, so it dominates.
        assert!(out[0] < 20.0, "the aligned key should pull the result low");
    }

    #[test]
    fn grouped_query_heads_share_kv_heads_by_integer_division() {
        // 4 query heads over 2 kv heads: q0,q1 -> kv0 and q2,q3 -> kv1.
        // A `%` mapping would give q0,q2 -> kv0, which also runs.
        let head_dim = 2;
        let (n_head, n_head_kv) = (4, 2);
        // kv0 has value 100, kv1 has value 200.
        let k = vec![1.0, 0.0, /*kv0*/ 1.0, 0.0 /*kv1*/];
        let v = vec![100.0, 100.0, 200.0, 200.0];
        let q = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = attend(&q, &k, &v, 1, p(n_head, n_head_kv, head_dim)).unwrap();
        assert!(close(out[0], 100.0, ATTENTION), "head 0 must read kv 0");
        assert!(close(out[2], 100.0, ATTENTION), "head 1 must read kv 0");
        assert!(close(out[4], 200.0, ATTENTION), "head 2 must read kv 1");
        assert!(close(out[6], 200.0, ATTENTION), "head 3 must read kv 1");
    }

    #[test]
    fn the_scale_is_one_over_sqrt_head_dim() {
        // Two orthogonal keys, query aligned with the first. The resulting
        // weight depends on the scale, so this pins it numerically.
        let head_dim = 4;
        let q = vec![2.0, 0.0, 0.0, 0.0];
        let k = vec![2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = attend(&q, &k, &v, 2, p(1, 1, head_dim)).unwrap();
        // logits: [4/2, 0] = [2, 0]; softmax -> e^2/(e^2+1) = 0.8808
        let want = (2.0f32).exp() / ((2.0f32).exp() + 1.0);
        assert!(
            close(out[0], want, ATTENTION),
            "got {}, want {want}",
            out[0]
        );
    }

    #[test]
    fn large_logits_do_not_overflow() {
        let head_dim = 4;
        let q = vec![100.0; 4];
        let k = vec![100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0];
        let v = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let out = attend(&q, &k, &v, 2, p(1, 1, head_dim)).unwrap();
        assert!(out.iter().all(|x| x.is_finite()), "overflowed: {out:?}");
        assert!(
            close(out[0], 1.0, ATTENTION),
            "the aligned key should win outright"
        );
    }

    #[test]
    fn shape_and_grouping_violations_are_refused() {
        assert!(
            attend(&[0.0; 4], &[0.0; 4], &[0.0; 4], 1, p(3, 2, 2)).is_err(),
            "3 does not group into 2"
        );
        assert!(
            attend(&[0.0; 3], &[0.0; 4], &[0.0; 4], 1, p(1, 1, 4)).is_err(),
            "q wrong length"
        );
        assert!(
            attend(&[0.0; 4], &[0.0; 4], &[0.0; 4], 0, p(1, 1, 4)).is_err(),
            "no positions"
        );
    }

    #[test]
    fn the_real_shape_runs() {
        // 32 heads, 4 kv heads, head_dim 128 — this model's attention.
        let (n_head, n_head_kv, head_dim, n_kv) = (32, 4, 128, 7);
        let q: Vec<f32> = (0..n_head * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let k: Vec<f32> = (0..n_kv * n_head_kv * head_dim)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();
        let v: Vec<f32> = (0..n_kv * n_head_kv * head_dim)
            .map(|i| (i as f32 * 0.03).sin())
            .collect();
        let out = attend(&q, &k, &v, n_kv, p(n_head, n_head_kv, head_dim)).unwrap();
        assert_eq!(out.len(), n_head * head_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
