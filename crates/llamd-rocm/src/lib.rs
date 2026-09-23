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

pub mod batch;
pub mod forward;
pub mod udot;
pub use batch::{bop, max_seq};
pub use forward::{DevBuf, free_memory, graph, op, stream_init, sync};

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
        pub fn llamd_rocm_matvec_grouped(
            d_w: *const std::ffi::c_void,
            eid: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        ) -> c_int;
        pub fn llamd_rocm_matvec_grouped_bench(
            d_w: *const std::ffi::c_void,
            eid: *const i32,
            aq: *const i8,
            ad: *const f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            iters: c_int,
            ms_out: *mut f32,
        ) -> c_int;
        pub fn llamd_rocm_quantize_q8k(
            x: *const f32,
            nblocks: c_int,
            qs: *mut i8,
            d: *mut f32,
            bsums: *mut i16,
        ) -> c_int;
        pub fn llamd_rocm_matvec_bench(
            d_w: *const std::ffi::c_void,
            aq: *const i8,
            ad: *const f32,
            blocks: c_int,
            rows: c_int,
            iters: c_int,
            ms_out: *mut f32,
        ) -> c_int;
    }
}

/// Serialises access to the HIP layer's shared scratch buffers.
///
/// WHY THIS IS NECESSARY, and it was found the hard way. `g_aq`, `g_ad`,
/// `g_out` and `g_eid` in llamd.hip are process-global statics, grown on
/// demand and reused across calls — the design that took a small projection
/// from 0.23 ms to well under it, by not allocating per call. But
/// `DeviceWeights` is `Send + Sync`, and `cargo test` runs tests concurrently,
/// so two threads entered the same scratch and overwrote each other's
/// activations. The symptom was a result with the RIGHT MAGNITUDE AND THE
/// WRONG SIGN — `-1752.9983` where `+1753.0044` was correct — which is exactly
/// the kind of corruption that would surface later as an occasional wrong
/// token and be blamed on the arithmetic.
///
/// A mutex rather than per-thread scratch: inference is one stream of work,
/// the lock is uncontended on that path, and thread-local device buffers would
/// multiply VRAM use by the thread count for no benefit.
static SCRATCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            let _guard = SCRATCH.lock().expect("scratch mutex");
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

/// Kernel-only time for `iters` launches, in milliseconds for the whole run.
///
/// WHY THIS IS SEPARATE FROM [`DeviceWeights::matvec`]. `matvec` is the honest
/// API: it takes host activations and returns host results, so it pays two
/// `hipMemcpy`s and a full device synchronise every call. That is the right
/// shape for a caller that has data on the host, and the wrong shape for
/// answering "how fast is the kernel". On a 1.2 MiB expert matrix the
/// round-trip is 30x the kernel, so a benchmark built on `matvec` measures
/// PCIe and concludes the kernel is slow.
///
/// This launches back to back with no synchronise between, timed with HIP
/// events on the device, after one warm launch. It is what the kernel costs
/// inside a resident forward pass — the number the design depends on.
impl DeviceWeights {
    pub fn bench_kernel_only(&self, act: &[llamd_quant::BlockQ8K], iters: u32) -> Result<f32> {
        if act.len() != self.blocks {
            return Err(Error::Shape {
                expected: vec![self.blocks],
                got: vec![act.len()],
            });
        }
        if iters == 0 {
            return Err(Error::malformed("bench iters", "0"));
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
            let mut ms = 0.0f32;
            // SAFETY: as `matvec` — `self.ptr` is a live upload of exactly
            // `rows*blocks*210` bytes, and `aq`/`ad` are sized from the same
            // `blocks`. `ms` is a single initialised f32.
            let _guard = SCRATCH.lock().expect("scratch mutex");
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_rocm_matvec_bench(
                    self.ptr,
                    aq.as_ptr(),
                    ad.as_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                    iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP kernel-only bench failed"));
            }
            Ok(ms)
        }
    }
}

/// The routed-expert path: all `expert_ids` experts in one launch.
///
/// The weights are the layer's whole 3-D expert tensor, uploaded once; each
/// workgroup indexes into it by `expert_ids[e]`. Returns `[n_used][rows]` —
/// one row block per routed expert, in the order given, unscaled. Applying the
/// router weights is the caller's job, because the caller is what knows
/// whether softmax came before or after top-k.
impl DeviceWeights {
    pub fn matvec_grouped(
        &self,
        act: &[llamd_quant::BlockQ8K],
        expert_ids: &[i32],
        rows_per_expert: usize,
    ) -> Result<Vec<f32>> {
        if act.len() != self.blocks {
            return Err(Error::Shape {
                expected: vec![self.blocks],
                got: vec![act.len()],
            });
        }
        if expert_ids.is_empty() {
            return Err(Error::malformed("expert_ids", "empty"));
        }
        // The upload holds n_expert matrices of rows_per_expert rows; a stated
        // row count that does not divide it means the caller has the layout
        // wrong, and reading past the allocation is the alternative.
        if rows_per_expert == 0 || self.rows % rows_per_expert != 0 {
            return Err(Error::malformed(
                "rows_per_expert",
                format!(
                    "{rows_per_expert} does not divide {} uploaded rows",
                    self.rows
                ),
            ));
        }
        let n_expert = self.rows / rows_per_expert;
        if let Some(bad) = expert_ids
            .iter()
            .find(|&&e| e < 0 || e as usize >= n_expert)
        {
            return Err(Error::malformed(
                "expert id",
                format!("{bad} outside 0..{n_expert}"),
            ));
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
            let mut out = vec![0.0f32; rows_per_expert * expert_ids.len()];
            // SAFETY: `self.ptr` is a live upload of rows*blocks*210 bytes;
            // every id was just bounds-checked against the matrix count, so
            // the largest offset the kernel forms stays inside it. `aq`/`ad`
            // are sized from `blocks`, `out` from rows_per_expert*n_used.
            let _guard = SCRATCH.lock().expect("scratch mutex");
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_rocm_matvec_grouped(
                    self.ptr,
                    expert_ids.as_ptr(),
                    aq.as_ptr(),
                    ad.as_ptr(),
                    out.as_mut_ptr(),
                    self.blocks as i32,
                    rows_per_expert as i32,
                    expert_ids.len() as i32,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP grouped matvec failed"));
            }
            Ok(out)
        }
    }

    /// Kernel-only milliseconds for `iters` grouped launches. See
    /// [`DeviceWeights::bench_kernel_only`] for why this is not `matvec_grouped`
    /// in a loop.
    pub fn bench_grouped_kernel_only(
        &self,
        act: &[llamd_quant::BlockQ8K],
        expert_ids: &[i32],
        rows_per_expert: usize,
        iters: u32,
    ) -> Result<f32> {
        if act.len() != self.blocks || expert_ids.is_empty() || iters == 0 {
            return Err(Error::malformed("grouped bench args", "invalid"));
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
            let mut ms = 0.0f32;
            // SAFETY: as `matvec_grouped`, with no output buffer — the kernel
            // writes only to device scratch sized by the same arguments.
            let _guard = SCRATCH.lock().expect("scratch mutex");
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_rocm_matvec_grouped_bench(
                    self.ptr,
                    expert_ids.as_ptr(),
                    aq.as_ptr(),
                    ad.as_ptr(),
                    self.blocks as i32,
                    rows_per_expert as i32,
                    expert_ids.len() as i32,
                    iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP grouped bench failed"));
            }
            Ok(ms)
        }
    }
}

/// Quantize to Q8_K on the device.
///
/// The point is not speed on its own — it is that while this conversion lives
/// on the host, no forward pass can stay resident: every matvec would hand a
/// buffer back across the bus. Validated against
/// `llamd_quant::quantize_q8_k` for exact equality, because a differing quant
/// shifts a routing decision and changes the token.
pub fn quantize_q8k_on_device(x: &[f32]) -> Result<Vec<llamd_quant::BlockQ8K>> {
    if x.is_empty() || x.len() % llamd_quant::QK_K != 0 {
        return Err(Error::malformed(
            "q8k input",
            format!("{} is not a multiple of {}", x.len(), llamd_quant::QK_K),
        ));
    }
    let nblocks = x.len() / llamd_quant::QK_K;
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        let mut qs = vec![0i8; nblocks * llamd_quant::QK_K];
        let mut d = vec![0.0f32; nblocks];
        let mut bsums = vec![0i16; nblocks * (llamd_quant::QK_K / 16)];
        // SAFETY: all four buffers are sized from the same `nblocks` the
        // kernel is told about, and `x` is non-empty and block-aligned.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_rocm_quantize_q8k(
                x.as_ptr(),
                nblocks as i32,
                qs.as_mut_ptr(),
                d.as_mut_ptr(),
                bsums.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(Error::other("HIP q8_k quantize failed"));
        }
        let mut out = Vec::with_capacity(nblocks);
        for b in 0..nblocks {
            let mut blk = llamd_quant::BlockQ8K {
                d: d[b],
                qs: [0; llamd_quant::QK_K],
                bsums: [0; llamd_quant::QK_K / 16],
            };
            blk.qs
                .copy_from_slice(&qs[b * llamd_quant::QK_K..(b + 1) * llamd_quant::QK_K]);
            blk.bsums.copy_from_slice(
                &bsums[b * (llamd_quant::QK_K / 16)..(b + 1) * (llamd_quant::QK_K / 16)],
            );
            out.push(blk);
        }
        Ok(out)
    }
}

/// A planar-repacked Q6_K tensor on the device, with its plane offsets.
///
/// Separate from [`DeviceWeights`] because the kernels index it differently:
/// four planes at fixed offsets rather than 210-byte interleaved blocks. The
/// repack happens on the host during upload and is a permutation, so the bytes
/// on the device are the same bytes the file holds.
pub struct PlanarWeights {
    buf: crate::forward::DevBuf,
    off: [usize; 4],
    rows: usize,
    blocks: usize,
}

impl PlanarWeights {
    /// Host-in, host-out udot4 matvec for `nseq` activations, for validation.
    ///
    /// The udot4 route replaces 16 multiply-accumulates per 16 weights with 4
    /// dot instructions, using the identity in llamd_udot.hip. Values must agree
    /// with the scalar reference to within reduction error; the integer part of
    /// the identity is exact, only the float regrouping differs.
    pub fn matvec_udot(&self, acts: &[llamd_quant::BlockQ8K], nseq: usize) -> Result<Vec<f32>> {
        if nseq == 0 || acts.len() != self.blocks * nseq {
            return Err(Error::Shape {
                expected: vec![self.blocks * nseq],
                got: vec![acts.len()],
            });
        }
        let mut qs = Vec::with_capacity(acts.len() * llamd_quant::QK_K);
        let mut ad = Vec::with_capacity(acts.len());
        for b in acts {
            qs.extend_from_slice(&b.qs);
            ad.push(b.d);
        }
        #[cfg(not(rocm))]
        {
            let _ = (qs, ad);
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            let mut out = vec![0.0f32; self.rows * nseq];
            let _guard = SCRATCH.lock().expect("scratch mutex");
            // SAFETY: the planar buffer holds rows*blocks blocks and `off` came
            // from the same layout; qs/ad are sized blocks*nseq and out
            // rows*nseq, matching what the entry point is told.
            #[allow(unsafe_code)]
            let rc = unsafe {
                crate::udot::ffi::llamd_u_matvec_host(
                    self.buf.ptr(),
                    self.off.as_ptr(),
                    qs.as_ptr(),
                    ad.as_ptr(),
                    out.as_mut_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                    nseq as i32,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP udot matvec failed"));
            }
            Ok(out)
        }
    }

    /// Repack interleaved Q6_K bytes and upload.
    pub fn upload(src: &[u8], rows: usize, blocks: usize) -> Result<Self> {
        let packed = llamd_quant::repack_q6k_planar(src, rows, blocks)?;
        let l = llamd_quant::PlanarQ6K::layout(rows, blocks);
        let buf = crate::forward::DevBuf::from_bytes(&packed)?;
        Ok(PlanarWeights {
            buf,
            off: [l.ql, l.qh, l.sc, l.d],
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
    pub fn offsets(&self) -> &[usize; 4] {
        &self.off
    }
    pub fn buf(&self) -> &crate::forward::DevBuf {
        &self.buf
    }

    /// Host-in, host-out planar matvec, for validation against the scalar path.
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
            let _guard = SCRATCH.lock().expect("scratch mutex");
            // SAFETY: the buffer holds a full planar repack of rows*blocks
            // blocks; `off` came from the same layout call, so every plane read
            // stays inside it. aq/ad/out are sized from blocks and rows.
            #[allow(unsafe_code)]
            let rc = unsafe {
                crate::forward::ffi::llamd_rocm_planar_matvec(
                    self.buf.ptr(),
                    self.off.as_ptr(),
                    aq.as_ptr(),
                    ad.as_ptr(),
                    out.as_mut_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP planar matvec failed"));
            }
            Ok(out)
        }
    }

    /// Kernel-only milliseconds for `iters` udot launches, matching the Vulkan
    /// backend's timing discipline so the two are comparable.
    pub fn bench_udot(
        &self,
        acts: &[llamd_quant::BlockQ8K],
        nseq: usize,
        iters: u32,
    ) -> Result<f32> {
        if nseq == 0 || acts.len() != self.blocks * nseq || iters == 0 {
            return Err(Error::malformed("udot bench args", "invalid"));
        }
        let mut qs = Vec::with_capacity(acts.len() * llamd_quant::QK_K);
        let mut ad = Vec::with_capacity(acts.len());
        for b in acts {
            qs.extend_from_slice(&b.qs);
            ad.push(b.d);
        }
        #[cfg(not(rocm))]
        {
            let _ = (qs, ad);
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            let mut ms = 0.0f32;
            let _guard = SCRATCH.lock().expect("scratch mutex");
            // SAFETY: as `matvec_udot`; buffers sized from blocks*nseq and rows.
            #[allow(unsafe_code)]
            let rc = unsafe {
                crate::udot::ffi::llamd_u_matvec_bench(
                    self.buf.ptr(),
                    self.off.as_ptr(),
                    qs.as_ptr(),
                    ad.as_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                    nseq as i32,
                    iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP udot bench failed"));
            }
            Ok(ms)
        }
    }

    /// Kernel-only milliseconds for `iters` planar launches.
    pub fn bench(&self, act: &[llamd_quant::BlockQ8K], iters: u32) -> Result<f32> {
        if act.len() != self.blocks || iters == 0 {
            return Err(Error::malformed("planar bench args", "invalid"));
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
            let mut ms = 0.0f32;
            let _guard = SCRATCH.lock().expect("scratch mutex");
            // SAFETY: as `matvec`, with no host output buffer.
            #[allow(unsafe_code)]
            let rc = unsafe {
                crate::forward::ffi::llamd_rocm_planar_bench(
                    self.buf.ptr(),
                    self.off.as_ptr(),
                    aq.as_ptr(),
                    ad.as_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                    iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP planar bench failed"));
            }
            Ok(ms)
        }
    }
}

impl DeviceWeights {
    /// Kernel-only milliseconds for `iters` split-k launches with `nsub` waves
    /// per row. Exists so `nsub` is chosen by measurement: reasoning about
    /// occupancy predicted a gain on attn_out and delivered a small loss.
    pub fn bench_splitk(
        &self,
        act: &[llamd_quant::BlockQ8K],
        nsub: u32,
        iters: u32,
    ) -> Result<f32> {
        if act.len() != self.blocks || iters == 0 || nsub == 0 || 8 % nsub != 0 {
            return Err(Error::malformed("splitk bench args", "invalid"));
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
            let mut ms = 0.0f32;
            let _guard = SCRATCH.lock().expect("scratch mutex");
            // SAFETY: as the other bench entry points — live upload, buffers
            // sized from the same `blocks`/`rows`.
            #[allow(unsafe_code)]
            let rc = unsafe {
                crate::forward::ffi::llamd_rocm_splitk_bench(
                    self.ptr,
                    aq.as_ptr(),
                    ad.as_ptr(),
                    self.blocks as i32,
                    self.rows as i32,
                    nsub as i32,
                    iters as i32,
                    &mut ms,
                )
            };
            if rc != 0 {
                return Err(Error::other("HIP split-k bench failed"));
            }
            Ok(ms)
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
