//! Which compute backend to use, and why.
//!
//! VULKAN IS THE DEFAULT GPU PATH. It is the API this class of AMD card exposes
//! natively through RADV, it is vendor-neutral, and on gfx1201 it provides
//! everything the ROCm path relies on: `integerDotProduct4x8BitPacked­Unsigned­
//! Accelerated` (the `udot4` equivalent, in hardware), `shaderInt8`, 8-bit
//! storage, subgroup arithmetic and size control, and the full 31.75 GiB
//! device-local heap. Choosing it as primary costs no arithmetic and allows the
//! kernels and memory strategy to be tuned hard for one vendor's hardware
//! without binding the engine to that vendor's runtime.
//!
//! ROCM REMAINS FULLY SUPPORTED — a selectable execution path and the
//! benchmarking reference. Both backends implement the same identity over the
//! same planar layout and are validated against the same scalar reference, so a
//! comparison between them on identical hardware measures the backend.
//!
//! CPU IS THE FALLBACK, and remains the correctness reference: it is the path
//! that agrees with llama.cpp token for token, and every device kernel is held
//! to it.
//!
//! WHAT THIS MODULE IS NOT. It does not abstract the operations — the device
//! op set lives in each backend crate, because the whole point of a
//! Vulkan-first design is that Vulkan-specific kernels and memory strategies can
//! be optimised aggressively. What stays backend-agnostic is everything above:
//! GGUF parsing, the model description, tensor shapes, sampling, and scheduling.

use llamd_core::{Error, Result};

/// The compute path in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Vulkan compute — the default on supported AMD hardware.
    Vulkan,
    /// ROCm/HIP — selectable, and the reference the Vulkan path is measured
    /// against.
    Rocm,
    /// Scalar CPU — the fallback, and the correctness reference.
    Cpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Rocm => "rocm",
            Backend::Cpu => "cpu",
        }
    }

    /// Is this backend actually usable on this machine right now?
    pub fn available(self) -> bool {
        match self {
            Backend::Vulkan => llamd_vulkan::available(),
            Backend::Rocm => llamd_rocm::available(),
            Backend::Cpu => true,
        }
    }

    pub fn parse(s: &str) -> Result<Backend> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vulkan" | "vk" => Ok(Backend::Vulkan),
            "rocm" | "hip" => Ok(Backend::Rocm),
            "cpu" => Ok(Backend::Cpu),
            other => Err(Error::malformed(
                "backend",
                format!("{other:?} is not one of vulkan, rocm, cpu"),
            )),
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The order preference is expressed in: Vulkan first, ROCm second, CPU last.
pub const PREFERENCE: [Backend; 3] = [Backend::Vulkan, Backend::Rocm, Backend::Cpu];

/// Choose a backend from an explicit request, or by preference if there is none.
///
/// Pure, so the policy is testable without mutating the process environment —
/// which this crate could not do anyway, being `#![forbid(unsafe_code)]`.
///
/// AN EXPLICIT REQUEST IS AN ERROR IF UNAVAILABLE, not a silent downgrade.
/// Someone who asked for ROCm in order to benchmark it, and silently got
/// Vulkan, would publish a number labelled with the wrong backend.
pub fn select_from(requested: Option<&str>) -> Result<Backend> {
    if let Some(v) = requested {
        if !v.trim().is_empty() {
            let want = Backend::parse(v)?;
            if !want.available() {
                return Err(Error::unsupported(
                    "backend",
                    format!(
                        "{} was requested but is not available on this machine",
                        want.name()
                    ),
                ));
            }
            return Ok(want);
        }
    }
    Ok(PREFERENCE
        .iter()
        .copied()
        .find(|b| b.available())
        .unwrap_or(Backend::Cpu))
}

/// [`select_from`] with the request taken from `LLAMD_BACKEND`.
pub fn select() -> Result<Backend> {
    let env = std::env::var("LLAMD_BACKEND").ok();
    select_from(env.as_deref())
}

/// One line per backend: available or not, and what the device reports.
///
/// Used by the CLI and by every benchmark, so a recorded number always carries
/// the backend and driver it was produced with.
pub fn report() -> String {
    let chosen = select().ok();
    let mut s = String::new();
    for b in PREFERENCE {
        let mark = if Some(b) == chosen {
            " <- selected"
        } else {
            ""
        };
        let detail = match b {
            Backend::Vulkan => match llamd_vulkan::device_info() {
                Ok(i) => i.to_string(),
                Err(_) if !llamd_vulkan::built_with_vulkan() => {
                    "not built with Vulkan support".into()
                }
                Err(e) => format!("unavailable: {e}"),
            },
            Backend::Rocm => match llamd_rocm::device_info(0) {
                Ok(i) => i.to_string(),
                Err(_) if !llamd_rocm::built_with_rocm() => "not built with ROCm support".into(),
                Err(e) => format!("unavailable: {e}"),
            },
            Backend::Cpu => "scalar reference, always available".into(),
        };
        s.push_str(&format!("  {:<7} {detail}{mark}\n", b.name()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available_so_selection_cannot_fail() {
        assert!(Backend::Cpu.available());
        let b = select_from(None).expect("selection must succeed");
        assert!(b.available(), "{b} was selected but is not available");
    }

    #[test]
    fn vulkan_is_preferred_over_rocm_over_cpu() {
        // The order IS the policy, so assert it rather than trusting the array
        // to stay in the intended order through future edits.
        assert_eq!(PREFERENCE[0], Backend::Vulkan);
        assert_eq!(PREFERENCE[1], Backend::Rocm);
        assert_eq!(PREFERENCE[2], Backend::Cpu);

        // And that preference actually decides: with nothing requested, the
        // selection is the first available in that order.
        let want = PREFERENCE.iter().copied().find(|b| b.available()).unwrap();
        assert_eq!(select_from(None).unwrap(), want);
    }

    #[test]
    fn names_round_trip_and_rubbish_is_refused() {
        for b in PREFERENCE {
            assert_eq!(Backend::parse(b.name()).unwrap(), b);
        }
        assert_eq!(Backend::parse("VK").unwrap(), Backend::Vulkan);
        assert_eq!(Backend::parse(" HIP ").unwrap(), Backend::Rocm);
        assert!(Backend::parse("cuda").is_err());
        assert!(Backend::parse("").is_err());
    }

    #[test]
    fn an_explicit_request_is_honoured_when_available() {
        for b in PREFERENCE {
            if b.available() {
                assert_eq!(select_from(Some(b.name())).unwrap(), b);
            }
        }
    }

    #[test]
    fn an_unavailable_explicit_request_is_an_error_not_a_downgrade() {
        // Silently falling back would mislabel a benchmark, which is worse than
        // refusing to run.
        for b in [Backend::Vulkan, Backend::Rocm] {
            if b.available() {
                continue;
            }
            assert!(
                select_from(Some(b.name())).is_err(),
                "{b} is unavailable but selection accepted it"
            );
        }
    }

    #[test]
    fn an_empty_request_falls_back_to_preference() {
        // An unset-but-present variable should not be treated as a request.
        assert_eq!(select_from(Some("")).unwrap(), select_from(None).unwrap());
        assert_eq!(
            select_from(Some("   ")).unwrap(),
            select_from(None).unwrap()
        );
    }
}
