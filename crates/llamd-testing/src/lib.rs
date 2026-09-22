//! The validation harness: LLAMD against llama.cpp, on identical inputs.
//!
//! THE RULE THIS CRATE EXISTS TO ENFORCE. When LLAMD and llama.cpp disagree,
//! llama.cpp is presumed right until proven otherwise. Every comparison here
//! names its tolerance from `llamd_core::tolerance`, and widening one is an
//! edit to that file with a reason — never a local literal bumped until a test
//! passes.
//!
//! Oracle tests need a real model and are skipped when `LLAMD_TEST_MODEL` is
//! unset, so `cargo test` works on a machine with no 23 GiB file. A skip is
//! printed, never silently passed: a test that quietly does nothing is worse
//! than one that fails.

#![forbid(unsafe_code)]

use std::path::PathBuf;

/// The model to validate against, from `LLAMD_TEST_MODEL`.
pub fn oracle_model() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("LLAMD_TEST_MODEL").ok()?);
    p.is_file().then_some(p)
}

/// Announce a skip loudly enough that nobody mistakes it for a pass.
pub fn skip(test: &str, why: &str) {
    eprintln!("  SKIP {test}: {why}");
}

/// Compare two f32 slices at a named tolerance, reporting where the worst
/// disagreement is rather than merely that one exists.
pub fn compare(label: &str, ours: &[f32], oracle: &[f32], tol: f32) -> Result<(), String> {
    if ours.len() != oracle.len() {
        return Err(format!(
            "{label}: length {} vs oracle {}",
            ours.len(),
            oracle.len()
        ));
    }
    match llamd_core::tolerance::max_deviation(ours, oracle) {
        None => Err(format!("{label}: incomparable")),
        Some((i, d)) if d > tol => Err(format!(
            "{label}: max relative deviation {d:.3e} at index {i} exceeds {tol:.1e} \
             (ours {:?}, oracle {:?})",
            ours.get(i),
            oracle.get(i)
        )),
        Some((i, d)) => {
            eprintln!("  {label}: max deviation {d:.3e} at {i} (tolerance {tol:.1e})");
            Ok(())
        }
    }
}
