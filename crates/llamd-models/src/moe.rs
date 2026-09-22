//! Mixture-of-experts routing.
//!
//! PROVENANCE. The order of operations is derived from
//! `llm_graph_context::build_moe_ffn` in `llama.cpp/src/llama-graph.cpp`
//! (MIT, © 2023-2026 The ggml authors), as called by `src/models/qwen3moe.cpp`
//! with `gating_op = SOFTMAX` and `norm_w = true`.
//!
//! THE ORDER IS THE ALGORITHM, and it is easy to get plausibly wrong:
//!
//! ```text
//!   1. logits  = router(x)                over ALL n_expert
//!   2. probs   = softmax(logits)          over ALL n_expert  ← before top-k
//!   3. chosen  = top_k(probs, n_used)     by probability
//!   4. weights = probs[chosen]
//!   5. weights = weights / sum(weights)   renormalise the chosen few
//! ```
//!
//! A CORRECTION to this comment, because the first version was wrong and a
//! negative control caught it. Softmaxing after top-k instead of before is
//! NOT a difference: renormalising a subset of a softmax is algebraically
//! identical to softmaxing the subset,
//!
//! ```text
//!   softmax(x)_i / SUM_{j in S} softmax(x)_j  ==  e^{x_i} / SUM_{j in S} e^{x_j}
//! ```
//!
//! and selection is unaffected because softmax is monotonic. The order below
//! matches the reference; swapping it would not change an answer.
//!
//! What DOES change the answer:
//!
//!   * **Skipping step 5.** Weights then sum to less than one, quietly
//!     scaling down every FFN contribution.
//!   * **A SECOND softmax on the gathered weights.** That is llama.cpp's
//!     `SOFTMAX_WEIGHT` gating, used by other architectures. qwen3moe uses
//!     plain `SOFTMAX`, so the gathered probabilities are renormalised, never
//!     re-softmaxed. Re-softmaxing flattens them toward uniform: eight
//!     nearly-equal experts instead of a sharp choice.
//!
//! EXPERT SELECTION ADMITS NO TOLERANCE. A different expert is a different
//! model, so `chosen` is compared with `tolerance::EXACT` and ties are broken
//! the way the reference breaks them: by ascending expert index.

use llamd_core::{Error, Result};

/// The smallest normal f16, used by ggml to clamp the weight sum away from
/// zero. Reproduced exactly: a different floor changes the weights whenever
/// the router is nearly uniform.
#[allow(clippy::excessive_precision)]
pub const WEIGHT_SUM_FLOOR: f32 = 6.103515625e-5; // exactly 2^-14

#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Selected expert indices, best first.
    pub experts: Vec<u32>,
    /// Their weights, renormalised to sum to one.
    pub weights: Vec<f32>,
}

/// Route one token.
///
/// `logits` holds one score per expert, as produced by `ffn_gate_inp`.
pub fn route(logits: &[f32], n_used: usize) -> Result<Routing> {
    if n_used == 0 || n_used > logits.len() {
        return Err(Error::malformed(
            "moe routing",
            format!("{n_used} experts requested of {}", logits.len()),
        ));
    }

    // 1-2. softmax over EVERY expert, shifted for stability.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }

    // 3. top-k by probability. Ties break toward the LOWER expert index,
    //    matching ggml's argsort — otherwise two experts with identical
    //    scores could be selected in either order and the result would not be
    //    reproducible.
    let mut idx: Vec<u32> = (0..probs.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        probs[b as usize]
            .partial_cmp(&probs[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let experts: Vec<u32> = idx.into_iter().take(n_used).collect();

    // 4-5. gather and renormalise, with ggml's clamp on the divisor.
    let mut weights: Vec<f32> = experts.iter().map(|&e| probs[e as usize]).collect();
    let sum = weights.iter().sum::<f32>().max(WEIGHT_SUM_FLOOR);
    for w in &mut weights {
        *w /= sum;
    }

    Ok(Routing { experts, weights })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llamd_core::tolerance::{ATTENTION, close};

    #[test]
    fn selects_the_highest_scoring_experts_in_order() {
        let logits = [0.1, 5.0, 0.2, 3.0, 0.3];
        let r = route(&logits, 2).unwrap();
        assert_eq!(r.experts, vec![1, 3], "best first");
    }

    #[test]
    fn weights_sum_to_one_after_renormalisation() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = route(&logits, 3).unwrap();
        let s: f32 = r.weights.iter().sum();
        assert!(close(s, 1.0, ATTENTION), "sum was {s}");
        // Without step 5 the three would sum to well under one, silently
        // scaling down every FFN contribution.
        assert!(r.weights[0] > r.weights[1] && r.weights[1] > r.weights[2]);
    }

    #[test]
    fn weights_are_renormalised_probabilities_not_re_softmaxed() {
        // THE DISTINCTION THAT MATTERS. qwen3moe uses gating SOFTMAX: the
        // gathered probabilities are renormalised. SOFTMAX_WEIGHT gating
        // applies a second softmax, which flattens the weights toward uniform
        // and turns a sharp routing decision into a mushy one.
        let logits = [10.0, 9.0, 0.0, 0.0];
        let r = route(&logits, 2).unwrap();
        assert!(
            close(r.weights[0], 0.7310586, ATTENTION),
            "got {}",
            r.weights[0]
        );
        assert!(close(r.weights[1], 0.2689414, ATTENTION));

        // Re-softmaxing those two probabilities gives about 0.61/0.39 -
        // visibly flatter. This asserts we did NOT do that.
        let resoftmaxed = {
            let p = [0.7310586f32, 0.2689414];
            let e: Vec<f32> = p.iter().map(|v| (v - p[0]).exp()).collect();
            e[0] / e.iter().sum::<f32>()
        };
        assert!(
            (r.weights[0] - resoftmaxed).abs() > 0.1,
            "weights look re-softmaxed: {} vs {}",
            r.weights[0],
            resoftmaxed
        );
    }

    #[test]
    fn softmax_order_is_irrelevant_but_renormalisation_is_not() {
        // Demonstrated rather than asserted in prose, because the prose was
        // wrong once: renormalising a subset equals softmaxing the subset.
        let logits = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let r = route(&logits, 2).unwrap();

        let direct = {
            let chosen = [logits[4], logits[2]];
            let m = chosen.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = chosen.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            vec![e[0] / s, e[1] / s]
        };
        for (a, b) in r.weights.iter().zip(&direct) {
            assert!(close(*a, *b, ATTENTION), "orders must agree: {a} vs {b}");
        }

        // Dropping the renormalisation DOES change it.
        let raw_sum: f32 = {
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = e.iter().sum();
            (e[4] + e[2]) / s
        };
        assert!(
            raw_sum < 0.95,
            "un-normalised weights would sum to {raw_sum}"
        );
    }

    #[test]
    fn ties_break_toward_the_lower_index() {
        // Reproducibility: identical scores must always give the same choice.
        let logits = [1.0, 1.0, 1.0, 1.0];
        let r = route(&logits, 2).unwrap();
        assert_eq!(r.experts, vec![0, 1]);
        for _ in 0..8 {
            assert_eq!(
                route(&logits, 2).unwrap().experts,
                vec![0, 1],
                "must be deterministic"
            );
        }
    }

    #[test]
    fn the_real_shape_routes_eight_of_one_hundred_twenty_eight() {
        let mut logits = vec![0.0f32; 128];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = (i as f32 * 0.37).sin() * 3.0;
        }
        let r = route(&logits, 8).unwrap();
        assert_eq!(r.experts.len(), 8);
        assert_eq!(r.weights.len(), 8);
        assert!(close(r.weights.iter().sum::<f32>(), 1.0, ATTENTION));
        // All distinct: selecting the same expert twice would double-count it.
        let uniq: std::collections::HashSet<_> = r.experts.iter().collect();
        assert_eq!(uniq.len(), 8, "experts must be distinct");
    }

    #[test]
    fn a_degenerate_router_does_not_divide_by_zero() {
        // Logits far below the max underflow to zero after the shift. The
        // clamp is what stops the renormalisation producing NaN.
        let logits = [0.0, -800.0, -800.0, -800.0];
        let r = route(&logits, 3).unwrap();
        assert!(
            r.weights.iter().all(|w| w.is_finite()),
            "weights were {:?}",
            r.weights
        );
    }

    #[test]
    fn impossible_requests_are_refused() {
        assert!(route(&[1.0, 2.0], 0).is_err());
        assert!(route(&[1.0, 2.0], 3).is_err());
    }
}
