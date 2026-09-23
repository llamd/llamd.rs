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
        pub fn llamd_rocm_upload(
            host: *const std::ffi::c_void,
            bytes: usize,
        ) -> *mut std::ffi::c_void;
        pub fn llamd_rocm_release(dev: *mut std::ffi::c_void);
        pub fn llamd_rocm_matvec_q6k_resident(
            d_w: *const std::ffi::c_void,
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

/// Weights resident on the device.
///
/// MEASURED, NOT ASSUMED. The first version uploaded the weights on every
/// call: 13.25 ms for a 6.6 MiB projection against 5.02 ms for the scalar CPU
/// — the GPU was 2.6× SLOWER, and only 0.34 ms of that was the transfer. The
/// cost was allocation churn. Weights live for the life of the model, so they
/// are uploaded once and this handle owns that allocation.
pub struct DeviceWeights {
    #[cfg(rocm)]
    ptr: *mut std::ffi::c_void,
    rows: usize,
    blocks: usize,
}

// SAFETY: the handle is an opaque device pointer that is only dereferenced
// inside HIP calls, and it is never aliased — `Drop` frees it exactly once.
// Sending it between threads is sound; sharing it is too, because every use
// is a read.
#[allow(unsafe_code)]
unsafe impl Send for DeviceWeights {}
#[allow(unsafe_code)]
unsafe impl Sync for DeviceWeights {}

impl DeviceWeights {
    /// Upload packed Q6_K weights for `rows` rows of `blocks` blocks each.
    pub fn upload(w: &[u8], rows: usize, blocks: usize) -> Result<Self> {
        let need = rows * blocks * llamd_quant::Q6K_BLOCK_BYTES;
        if w.len() < need {
            return Err(Error::malformed(
                "device weights",
                format!("need {need} bytes for {rows}x{blocks}, have {}", w.len()),
            ));
        }
        #[cfg(not(rocm))]
        {
            let _ = (rows, blocks);
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            if !available() {
                return Err(Error::unsupported("rocm", "no device present"));
            }
            // SAFETY: `w` is at least `need` bytes and `need` is what is
            // copied; the callee returns null on failure rather than a
            // partially-initialised pointer.
            #[allow(unsafe_code)]
            let ptr = unsafe { ffi::llamd_rocm_upload(w.as_ptr() as *const _, need) };
            if ptr.is_null() {
                return Err(Error::other(format!(
                    "hipMalloc/hipMemcpy failed for {:.1} MiB of weights",
                    need as f64 / (1 << 20) as f64
                )));
            }
            Ok(DeviceWeights { ptr, rows, blocks })
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Matvec against the resident weights. Only the activation and the
    /// result cross the bus.
    pub fn matvec(&self, act: &[llamd_quant::BlockQ8K]) -> Result<Vec<f32>> {
        if act.len() != self.blocks {
            return Err(Error::Shape {
                expected: vec![self.blocks],
                got: vec![act.len()],
            });
        }
        let mut aq = Vec::with_capacity(self.blocks * llamd_quant::QK_K);
        let mut ad = Vec::with_capacity(self.blocks);
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
            let mut out = vec![0.0f32; self.rows];
            // SAFETY: `self.ptr` was returned by a successful upload of
            // exactly `rows*blocks*210` bytes and has not been freed — `Drop`
            // is the only release and consumes the handle. `aq`, `ad` and
            // `out` are sized from the same `blocks`/`rows`.
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_rocm_matvec_q6k_resident(
                    self.ptr,
                    aq.as_ptr(),
                    ad.as_ptr(),
                    out.as_mut_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP resident matvec failed"));
            }
            Ok(out)
        }
    }
}

impl Drop for DeviceWeights {
    fn drop(&mut self) {
        #[cfg(rocm)]
        {
            // SAFETY: freeing a pointer this handle owns, exactly once.
            #[allow(unsafe_code)]
            unsafe {
                ffi::llamd_rocm_release(self.ptr)
            };
        }
    }
}

/// Q6_K × Q8_K matrix-vector product, uploading the weights each call.
///
/// Kept for the validation tests, which check correctness rather than speed.
/// Anything that runs repeatedly should use [`DeviceWeights`]: this form was
/// measured at 2.6× slower than the CPU, almost entirely in allocation.
pub fn matvec_q6k(w: &[u8], act: &[llamd_quant::BlockQ8K], rows: usize) -> Result<Vec<f32>> {
    // Expressed on top of DeviceWeights rather than as a second kernel: two
    // implementations of the same arithmetic is two things to keep in
    // agreement, and the validation tests exist to check the kernel, not the
    // convenience wrapper around it.
    if act.is_empty() || rows == 0 {
        return Err(Error::malformed(
            "rocm matvec",
            format!("{rows} rows, {} blocks", act.len()),
        ));
    }
    let dw = DeviceWeights::upload(w, rows, act.len())?;
    dw.matvec(act)
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
