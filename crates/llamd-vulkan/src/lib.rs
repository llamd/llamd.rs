//! The Vulkan compute backend — LLAMD's primary GPU path on AMD hardware.
//!
//! WHY VULKAN FIRST. It is the API this card exposes natively through RADV, it
//! is vendor-neutral, and on this device it provides everything the ROCm path
//! relies on: `integerDotProduct4x8BitPackedUnsignedAccelerated` (the `udot4`
//! equivalent, in hardware), `shaderInt8`, 8-bit storage, subgroup arithmetic
//! and size control, and a 31.75 GiB device-local heap. That means a
//! Vulkan-first engine gives up no arithmetic, and it can be optimised around
//! one vendor's hardware without being tied to that vendor's runtime.
//!
//! ROCM REMAINS FULLY SUPPORTED, as a selectable execution path and as the
//! benchmarking reference. The two backends implement the SAME arithmetic —
//! `shaders/q6k_matvec.comp` and `kernels/llamd_udot.hip` are the same identity,
//! validated against the same scalar reference — so a comparison between them on
//! identical hardware measures the backend rather than the algorithm.
//!
//! NO NEW DEPENDENCIES. The workspace policy is that the numerical path carries
//! none, so the Vulkan host is C++ behind a small C API with a thin safe wrapper
//! here, exactly as `llamd-rocm` does. The SPIR-V is compiled by `build.rs` and
//! embedded in the library.

#![deny(unsafe_code)]

pub mod ops;
pub use ops::{Pipeline, Push, begin, end_and_wait, subgroup_size};

use llamd_core::{Error, Result};
use std::ffi::c_void;

/// Was this crate built against a real Vulkan toolchain?
pub const fn built_with_vulkan() -> bool {
    cfg!(vulkan)
}

/// Initialise the device. Idempotent; safe to call from anywhere.
///
/// Picks a DISCRETE GPU deliberately: this machine also exposes `llvmpipe`, a
/// software Vulkan device that would make every measurement meaningless while
/// reporting success.
pub fn init() -> Result<()> {
    #[cfg(not(vulkan))]
    {
        Err(Error::unsupported(
            "vulkan",
            "not built with Vulkan support",
        ))
    }
    #[cfg(vulkan)]
    {
        // SAFETY: no arguments; reports failure by return value and is
        // idempotent on the C side.
        #[allow(unsafe_code)]
        let rc = unsafe { ffi::llamd_vk_init() };
        if rc != 0 {
            return Err(Error::other("Vulkan initialisation failed"));
        }
        Ok(())
    }
}

pub fn available() -> bool {
    #[cfg(not(vulkan))]
    {
        false
    }
    #[cfg(vulkan)]
    {
        if init().is_err() {
            return false;
        }
        // SAFETY: no arguments.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_vk_available() != 0
        }
    }
}

/// What the device reports about itself.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub driver: String,
    pub heap_bytes: u64,
    pub subgroup_size: u32,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} via {} — {:.2} GiB device-local, subgroup {}",
            self.name,
            self.driver,
            self.heap_bytes as f64 / (1u64 << 30) as f64,
            self.subgroup_size
        )
    }
}

pub fn device_info() -> Result<DeviceInfo> {
    init()?;
    #[cfg(not(vulkan))]
    {
        Err(Error::unsupported(
            "vulkan",
            "not built with Vulkan support",
        ))
    }
    #[cfg(vulkan)]
    {
        let mut name = vec![0u8; 256];
        let mut driver = vec![0u8; 256];
        let mut heap = 0u64;
        let mut sg = 0u32;
        // SAFETY: both buffers are 256 bytes and their lengths are passed; the
        // scalars are initialised.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_vk_device_info(
                name.as_mut_ptr() as *mut i8,
                256,
                driver.as_mut_ptr() as *mut i8,
                256,
                &mut heap,
                &mut sg,
            )
        };
        if rc != 0 {
            return Err(Error::other("could not query the Vulkan device"));
        }
        let cut = |v: Vec<u8>| {
            let n = v.iter().position(|&c| c == 0).unwrap_or(v.len());
            String::from_utf8_lossy(&v[..n]).into_owned()
        };
        Ok(DeviceInfo {
            name: cut(name),
            driver: cut(driver),
            heap_bytes: heap,
            subgroup_size: sg,
        })
    }
}

/// A device-local buffer, freed on drop.
///
/// Not `Clone`: two owners of one free is a double free, and the type system is
/// the cheapest place to prevent it.
pub struct VkBuf {
    h: *mut c_void,
    bytes: usize,
}

// SAFETY: the handle is an address into the Vulkan implementation, valid on any
// host thread. Access is serialised by the caller holding `&mut`, and the C side
// serialises its own command buffer.
#[allow(unsafe_code)]
unsafe impl Send for VkBuf {}
#[allow(unsafe_code)]
unsafe impl Sync for VkBuf {}

impl VkBuf {
    pub fn new(bytes: usize) -> Result<Self> {
        init()?;
        if bytes == 0 {
            return Err(Error::malformed("vulkan alloc", "zero bytes"));
        }
        #[cfg(not(vulkan))]
        {
            Err(Error::unsupported(
                "vulkan",
                "not built with Vulkan support",
            ))
        }
        #[cfg(vulkan)]
        {
            // SAFETY: a plain allocation; the result is checked for null.
            #[allow(unsafe_code)]
            let h = unsafe { ffi::llamd_vk_alloc(bytes as u64) };
            if h.is_null() {
                return Err(Error::other(format!(
                    "Vulkan allocation of {} MiB failed",
                    bytes / (1 << 20)
                )));
            }
            Ok(VkBuf { h, bytes })
        }
    }

    pub fn from_bytes(host: &[u8]) -> Result<Self> {
        let b = VkBuf::new(host.len())?;
        b.upload(host)?;
        Ok(b)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn handle(&self) -> *mut c_void {
        self.h
    }

    pub fn upload(&self, host: &[u8]) -> Result<()> {
        if host.len() > self.bytes {
            return Err(Error::Shape {
                expected: vec![self.bytes],
                got: vec![host.len()],
            });
        }
        #[cfg(not(vulkan))]
        {
            Err(Error::unsupported(
                "vulkan",
                "not built with Vulkan support",
            ))
        }
        #[cfg(vulkan)]
        {
            // SAFETY: `host.len()` bytes are read from a live slice into an
            // allocation just checked to be at least that large.
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_vk_upload(self.h, host.as_ptr() as *const c_void, host.len() as u64)
            };
            if rc != 0 {
                return Err(Error::other("Vulkan upload failed"));
            }
            Ok(())
        }
    }

    pub fn download(&self, out: &mut [u8]) -> Result<()> {
        if out.len() > self.bytes {
            return Err(Error::Shape {
                expected: vec![self.bytes],
                got: vec![out.len()],
            });
        }
        #[cfg(not(vulkan))]
        {
            Err(Error::unsupported(
                "vulkan",
                "not built with Vulkan support",
            ))
        }
        #[cfg(vulkan)]
        {
            // SAFETY: as `upload`, in the other direction.
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_vk_download(self.h, out.as_mut_ptr() as *mut c_void, out.len() as u64)
            };
            if rc != 0 {
                return Err(Error::other("Vulkan download failed"));
            }
            Ok(())
        }
    }
}

impl Drop for VkBuf {
    fn drop(&mut self) {
        #[cfg(vulkan)]
        {
            // SAFETY: `self.h` came from `llamd_vk_alloc` and this is the only
            // release; `VkBuf` is not `Clone`, so no second owner exists.
            #[allow(unsafe_code)]
            unsafe {
                ffi::llamd_vk_free(self.h)
            };
        }
    }
}

/// Helpers for moving typed data, since the FFI boundary speaks bytes.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only called for plain-old-data (u8, i32, f32), which have no
    // padding or invalid bit patterns, so their bytes are a valid &[u8].
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
    }
}

fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    let n = std::mem::size_of_val(v);
    // SAFETY: as `as_bytes`.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n)
    }
}

impl VkBuf {
    pub fn from_f32(host: &[f32]) -> Result<Self> {
        VkBuf::from_bytes(as_bytes(host))
    }
    pub fn from_i32(host: &[i32]) -> Result<Self> {
        VkBuf::from_bytes(as_bytes(host))
    }
    pub fn download_f32(&self, out: &mut [f32]) -> Result<()> {
        self.download(as_bytes_mut(out))
    }
    pub fn upload_i32(&self, host: &[i32]) -> Result<()> {
        self.upload(as_bytes(host))
    }
    pub fn download_i32(&self, out: &mut [i32]) -> Result<()> {
        self.download(as_bytes_mut(out))
    }
}

/// Planar Q6_K weights on the Vulkan device, with their plane offsets.
///
/// The layout is `llamd_quant::PlanarQ6K`, identical to what the ROCm backend
/// uploads — the same bytes in the same order, so a comparison between backends
/// is not confounded by a different data layout.
pub struct PlanarWeights {
    buf: VkBuf,
    off: [u64; 4],
    rows: usize,
    blocks: usize,
}

impl PlanarWeights {
    pub fn upload(src: &[u8], rows: usize, blocks: usize) -> Result<Self> {
        let packed = llamd_quant::repack_q6k_planar(src, rows, blocks)?;
        let l = llamd_quant::PlanarQ6K::layout(rows, blocks);
        Ok(PlanarWeights {
            buf: VkBuf::from_bytes(&packed)?,
            off: [l.ql as u64, l.qh as u64, l.sc as u64, l.d as u64],
            rows,
            blocks,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn blocks(&self) -> usize {
        self.blocks
    }
    pub fn buf(&self) -> &VkBuf {
        &self.buf
    }
    pub fn offsets(&self) -> &[u64; 4] {
        &self.off
    }

    /// Q6_K × Q8_K matvec for `nseq` activations.
    ///
    /// `iters` > 1 dispatches repeatedly with a barrier between, for timing the
    /// shader rather than the submission.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec(
        &self,
        qb: &VkBuf,
        sb: &VkBuf,
        ad: &VkBuf,
        out: &VkBuf,
        nseq: usize,
        iters: usize,
    ) -> Result<()> {
        if nseq == 0 || nseq > 8 {
            return Err(Error::malformed("vulkan matvec", format!("nseq {nseq}")));
        }
        if out.bytes() < self.rows * nseq * 4 {
            return Err(Error::malformed("vulkan matvec", "output buffer too small"));
        }
        #[cfg(not(vulkan))]
        {
            Err(Error::unsupported(
                "vulkan",
                "not built with Vulkan support",
            ))
        }
        #[cfg(vulkan)]
        {
            // SAFETY: every handle is a live VkBuf; the output size was checked
            // against rows*nseq, which is what the shader writes, and the plane
            // offsets come from the layout the buffer was packed with.
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_vk_matvec(
                    self.buf.handle(),
                    self.off.as_ptr(),
                    qb.handle(),
                    sb.handle(),
                    ad.handle(),
                    out.handle(),
                    self.blocks as i32,
                    self.rows as i32,
                    nseq as i32,
                    iters as i32,
                )
            };
            if rc != 0 {
                return Err(Error::other("Vulkan matvec dispatch failed"));
            }
            Ok(())
        }
    }
}

#[cfg(vulkan)]
#[allow(unsafe_code)]
pub(crate) mod ffi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        pub fn llamd_vk_init() -> c_int;
        pub fn llamd_vk_available() -> c_int;
        pub fn llamd_vk_device_info(
            name: *mut c_char,
            name_len: c_int,
            driver: *mut c_char,
            driver_len: c_int,
            heap: *mut u64,
            subgroup: *mut u32,
        ) -> c_int;
        pub fn llamd_vk_alloc(bytes: u64) -> *mut c_void;
        pub fn llamd_vk_pipeline_count() -> c_int;
        pub fn llamd_vk_pipeline_name(i: c_int) -> *const c_char;
        pub fn llamd_vk_begin() -> c_int;
        pub fn llamd_vk_dispatch(
            pipe: c_int,
            bufs: *const *mut c_void,
            nbuf: c_int,
            push: *const u32,
            gx: u32,
            gy: u32,
        ) -> c_int;
        pub fn llamd_vk_end_and_wait() -> c_int;
        pub fn llamd_vk_subgroup_size() -> u32;
        pub fn llamd_vk_free(h: *mut c_void);
        pub fn llamd_vk_upload(h: *mut c_void, src: *const c_void, bytes: u64) -> c_int;
        pub fn llamd_vk_download(h: *mut c_void, dst: *mut c_void, bytes: u64) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_vk_matvec(
            w: *mut c_void,
            off: *const u64,
            qb: *mut c_void,
            sb: *mut c_void,
            ad: *mut c_void,
            out: *mut c_void,
            blocks: c_int,
            rows: c_int,
            nseq: c_int,
            iters: c_int,
        ) -> c_int;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_is_reported_not_assumed() {
        // Must not panic without Vulkan: this is what lets the workspace build
        // and test on a machine with no GPU.
        let a = available();
        if !built_with_vulkan() {
            assert!(!a, "a stub build cannot see a device");
        }
    }
}
