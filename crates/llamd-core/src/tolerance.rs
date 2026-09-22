//! Named numerical tolerances.
//!
//! WHY THESE ARE NAMED CONSTANTS AND NOT LITERALS. A comparison against a
//! reference implementation is only as meaningful as its tolerance. If the
//! tolerance is a literal in a test, the path of least resistance when the
//! test fails is to raise it — which silently changes the claim the test
//! makes without anybody reviewing that change.
//!
//! Here, widening a tolerance is an edit to this file, with a reason, in a
//! diff. That is the whole point.

/// Bit-exact. No tolerance at all.
///
/// Used where a difference of one bit means the implementation is wrong, not
/// merely imprecise: dequantized weights, token ids, expert selection, tensor
/// shapes, and any comparison at temperature 0.
pub const EXACT: f32 = 0.0;

/// Single f32 rounding step at unit magnitude.
///
/// For operations that should produce identical results but may legitimately
/// differ in the last bit because of instruction selection or fused
/// multiply-add.
pub const F32_ULP: f32 = 1.2e-7;

/// Same arithmetic, different summation order.
///
/// A reduction over `n` terms accumulates about `sqrt(n)` rounding steps, and
/// two implementations that block or vectorise the sum differently will
/// disagree by roughly that much. This is NOT a licence for a different
/// algorithm: only for the same one summed in a different order.
///
/// Sized for reductions up to a few thousand terms — `sqrt(4096) = 64` ULP is
/// about 7.6e-6, so this leaves headroom without being able to hide a
/// structural error, which would be O(1) rather than O(1e-6).
///
/// Introduced when `rms_norm` over 128 elements differed from ggml by 1.9 ULP
/// and `F32_ULP` was found to be the wrong category for any reduction.
pub const REDUCTION: f32 = 1e-5;

/// Accumulated error through one attention block.
///
/// Justification: a reduction over `n` terms accumulates roughly `sqrt(n)`
/// rounding steps. For head_dim 128 that is ~11 ULP; this leaves an order of
/// magnitude of headroom. If a comparison needs more than this, the cause is
/// an algorithmic difference, not rounding, and raising the number hides it.
pub const ATTENTION: f32 = 1e-5;

/// Accumulated error through a full transformer layer.
pub const LAYER: f32 = 1e-4;

/// Logits after a full forward pass over many layers.
///
/// Deliberately the loosest value here. Logits are the end of a long
/// accumulation chain, and a difference at this scale does not change the
/// sampled token — which is what `EXACT` on the token id is for. If logits
/// agree to 2e-3 but the tokens differ, the disagreement is at a decision
/// boundary and must be investigated, not tolerated.
pub const LOGITS: f32 = 2e-3;

/// Are two values within tolerance, relative where it is meaningful?
///
/// Uses relative error above unit magnitude and absolute error below it: a
/// purely relative check is meaningless near zero, and a purely absolute one
/// is far too strict for logits in the tens.
pub fn close(a: f32, b: f32, tol: f32) -> bool {
    if tol == EXACT {
        // Bit-exact means bit-exact, including the sign of zero and NaN
        // payloads — a comparison that treats -0.0 and 0.0 as equal is not
        // checking what it claims to check.
        return a.to_bits() == b.to_bits();
    }
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff / scale <= tol
}

/// Largest relative deviation across two slices, and where it occurred.
///
/// Returns the index as well as the value: "max deviation 3e-4" is not
/// actionable, "max deviation 3e-4 at index 4021" is.
pub fn max_deviation(a: &[f32], b: &[f32]) -> Option<(usize, f32)> {
    if a.len() != b.len() {
        return None;
    }
    let mut worst = (0usize, 0.0f32);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        let scale = x.abs().max(y.abs()).max(1.0);
        let d = (x - y).abs() / scale;
        if d > worst.1 {
            worst = (i, d);
        }
    }
    Some(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_means_bitwise() {
        assert!(close(1.0, 1.0, EXACT));
        assert!(!close(1.0, 1.0 + F32_ULP, EXACT));
        // -0.0 and 0.0 compare equal under `==` but are different bit
        // patterns. A bit-exact check must say so.
        assert!(!close(0.0, -0.0, EXACT));
    }

    #[test]
    fn nan_is_only_close_to_nan() {
        assert!(close(f32::NAN, f32::NAN, LOGITS));
        assert!(!close(f32::NAN, 0.0, LOGITS));
        assert!(!close(0.0, f32::NAN, LOGITS));
    }

    #[test]
    fn small_values_use_absolute_error() {
        // A purely relative check would call these wildly different.
        assert!(close(1e-9, 2e-9, ATTENTION));
        // And must still reject a genuine difference at unit scale.
        assert!(!close(1.0, 1.1, ATTENTION));
    }

    #[test]
    // The values ARE constants; clippy is right that the comparison is
    // compile-time. That is the point: this is a guard on the constants
    // themselves, so that widening one out of order fails the build.
    #[allow(clippy::assertions_on_constants)]
    fn tolerances_are_ordered() {
        // Each stage of accumulation admits more error than the one before.
        // If this ever fails somebody has widened the wrong constant.
        assert!(EXACT < F32_ULP);
        assert!(F32_ULP < REDUCTION);
        assert!(REDUCTION <= ATTENTION);
        assert!(ATTENTION < LAYER);
        assert!(LAYER < LOGITS);
    }

    #[test]
    fn max_deviation_reports_where() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.5];
        let (i, d) = max_deviation(&a, &b).unwrap();
        assert_eq!(i, 2);
        assert!((d - 0.5 / 3.5).abs() < 1e-6);
        assert!(
            max_deviation(&a, &[1.0]).is_none(),
            "length mismatch must not be silently ignored"
        );
    }
}
