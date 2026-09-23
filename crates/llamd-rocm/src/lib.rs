//! The AMD backend.
//!
//! WHAT MAKES THIS DIFFERENT FROM A PORT. llama.cpp's AMD support is its CUDA
//! backend hipified at build time — `ggml/src/ggml-hip/` contains exactly one
//! file, a CMakeLists. That works, and it inherits NVIDIA's execution model:
//! warp size 32 as an accident rather than a choice, tile shapes tuned for a
//! different memory hierarchy, and staging patterns that assume shared memory
//! is cheap. LLAMD's kernels are written against RDNA4's actual behaviour.
//! Whether that wins is a question for `llamd-bench`, not for this comment.
//!
//! UNSAFE POLICY. This is one of four crates permitted `unsafe`, and it uses
//! it only to cross the HIP FFI boundary. Every such call is wrapped in a safe
//! function that checks its own preconditions first, so no caller outside this
//! crate can reach an unchecked pointer. `deny` rather than `forbid`, so the
//! exceptions must be written where a reviewer sees them.
//!
//! AVAILABILITY IS A RUNTIME QUESTION. `cargo build --workspace` must work on
//! a machine with no AMD GPU, so `build.rs` detects ROCm and this crate
//! compiles to a stub when it is absent. Callers ask [`available`] rather than
//! assuming.

#![deny(unsafe_code)]

use llamd_core::{Error, Result};

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub total_mem: usize,
    pub free_mem: usize,
    /// Workgroup processors. Each holds TWO compute units on RDNA, so the
    /// R9700's 64 CUs report as 32 WGPs — the number HIP gives is the one
    /// that matters for occupancy, but calling it "CUs" halves it.
    pub wgp_count: i32,
    /// Native wave width. 32 on RDNA; 64 on CDNA and on older GCN parts.
    pub wave_size: i32,
    /// The full target, e.g. "gfx1201". From `gcnArchName`: major/minor alone
    /// give 12.0 and would name a different binary than hipcc built.
    pub arch: String,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} — {:.2} GiB ({:.2} free), {} WGPs ({} CUs), wave{}, {}",
            self.name,
            self.total_mem as f64 / (1u64 << 30) as f64,
            self.free_mem as f64 / (1u64 << 30) as f64,
            self.wgp_count,
            self.wgp_count * 2,
            self.wave_size,
            self.arch,
        )
    }
}

#[cfg(rocm)]
#[allow(unsafe_code)]
mod ffi {
    //! The HIP boundary. Every declaration here is unchecked by definition;
    //! the safe wrappers above validate lengths before any of them is called.

    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        pub fn llamd_rocm_device_count() -> c_int;
        pub fn llamd_rocm_device_info(
            dev: c_int,
            name: *mut c_char,
            name_len: c_int,
            arch: *mut c_char,
            arch_len: c_int,
            total_mem: *mut usize,
            wgp_count: *mut c_int,
            warp_size: *mut c_int,
        ) -> c_int;
        pub fn llamd_rocm_free_mem() -> usize;
        pub fn llamd_rocm_matvec_q6k(
            w: *const u8,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
        ) -> c_int;
    }
}

/// Was this crate built against a real ROCm installation?
pub const fn built_with_rocm() -> bool {
    cfg!(rocm)
}

/// Is a device actually present and usable now?
///
/// Distinct from [`built_with_rocm`]: the binary can carry kernels for a card
/// that is not in this machine, and that failure should be reported rather
/// than crashing on the first launch.
pub fn available() -> bool {
    device_count() > 0
}

pub fn device_count() -> i32 {
    #[cfg(rocm)]
    {
        // SAFETY: the function takes no arguments and returns a plain int; it
        // reports 0 rather than failing when the runtime is unusable.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_rocm_device_count()
        }
    }
    #[cfg(not(rocm))]
    0
}

pub fn device_info(index: i32) -> Result<DeviceInfo> {
    #[cfg(not(rocm))]
    {
        let _ = index;
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        if index < 0 || index >= device_count() {
            return Err(Error::malformed(
                "rocm device",
                format!("index {index} of {} devices", device_count()),
            ));
        }
        let mut name = [0i8; 256];
        let mut arch = [0i8; 64];
        let (mut total, mut wgps, mut wave) = (0usize, 0, 0);
        // SAFETY: `name` is 256 bytes and the length is passed alongside it;
        // every out-pointer targets a live local. The callee writes a NUL
        // within the buffer, which the conversion below relies on.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_rocm_device_info(
                index,
                name.as_mut_ptr(),
                name.len() as i32,
                arch.as_mut_ptr(),
                arch.len() as i32,
                &mut total,
                &mut wgps,
                &mut wave,
            )
        };
        if rc != 0 {
            return Err(Error::other(format!(
                "hipGetDeviceProperties failed for device {index}"
            )));
        }
        let cstr = |b: &[i8]| -> String {
            let v: Vec<u8> = b
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&v).into_owned()
        };
        // The driver strings are not guaranteed UTF-8, so both are converted
        // lossily rather than unwrapped.
        let name = cstr(&name);
        let arch = cstr(&arch);

        // SAFETY: no arguments; returns 0 when the query fails.
        #[allow(unsafe_code)]
        let free = unsafe { ffi::llamd_rocm_free_mem() };

        Ok(DeviceInfo {
            index,
            name,
            total_mem: total,
            free_mem: free,
            wgp_count: wgps,
            wave_size: wave,
            arch,
        })
    }
}

/// Q6_K × Q8_K matrix-vector product on the GPU.
///
/// `w` holds `rows × blocks` packed Q6_K blocks; the activation is supplied
/// already quantized, because quantizing it once and reusing it across rows is
/// the whole point — re-quantizing per row would cost more than the matmul.
///
/// Returns one f32 per row, and must agree with
/// [`llamd_quant::dot_q6k_q8k`]: a difference here flips a near-tied MoE
/// expert and changes the emitted token.
pub fn matvec_q6k(w: &[u8], act: &[llamd_quant::BlockQ8K], rows: usize) -> Result<Vec<f32>> {
    let blocks = act.len();
    if blocks == 0 || rows == 0 {
        return Err(Error::malformed(
            "rocm matvec",
            format!("{rows} rows, {blocks} blocks"),
        ));
    }
    let need = rows * blocks * llamd_quant::Q6K_BLOCK_BYTES;
    if w.len() < need {
        return Err(Error::malformed(
            "rocm matvec",
            format!(
                "need {need} weight bytes for {rows}x{blocks}, have {}",
                w.len()
            ),
        ));
    }

    // The kernel wants quants and scales separately: one contiguous i8 array
    // it can read coalesced, and one small f32 array. Splitting here keeps the
    // device side free of struct-of-array gymnastics.
    let mut aq = Vec::with_capacity(blocks * llamd_quant::QK_K);
    let mut ad = Vec::with_capacity(blocks);
    for b in act {
        aq.extend_from_slice(&b.qs);
        ad.push(b.d);
    }

    #[cfg(not(rocm))]
    {
        let _ = (aq, ad);
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        if !available() {
            return Err(Error::unsupported("rocm", "no device present"));
        }
        let mut out = vec![0.0f32; rows];
        // SAFETY: every buffer length was checked above — `w` holds at least
        // `rows*blocks*210` bytes, `aq` exactly `blocks*256`, `ad` exactly
        // `blocks`, and `out` exactly `rows`. The callee reads and writes no
        // more than it is told, and does not retain the pointers.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_rocm_matvec_q6k(
                w.as_ptr(),
                aq.as_ptr(),
                ad.as_ptr(),
                out.as_mut_ptr(),
                blocks as i32,
                rows as i32,
            )
        };
        if rc != 0 {
            return Err(Error::other("HIP matvec failed — see the ROCm log"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_is_reported_not_assumed() {
        // Must not panic on a machine without a GPU: this is what lets the
        // workspace build and test anywhere.
        let n = device_count();
        assert!(n >= 0);
        assert_eq!(available(), n > 0);
        if !built_with_rocm() {
            assert_eq!(n, 0, "a stub build cannot see devices");
        }
    }

    #[test]
    fn bad_arguments_are_refused_before_the_ffi() {
        // Checked on both build configurations, so the guard is not something
        // only a GPU machine exercises.
        assert!(matvec_q6k(&[], &[], 0).is_err());
        let one = vec![llamd_quant::BlockQ8K::default()];
        assert!(
            matvec_q6k(&[0u8; 10], &one, 1).is_err(),
            "short weights must be refused"
        );
    }
}
