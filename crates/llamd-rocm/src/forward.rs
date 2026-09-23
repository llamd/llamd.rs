//! The resident forward pass: a whole token computed on the device.
//!
//! WHAT THIS REPLACES. `llamd-runtime::forward` computes a token on the CPU,
//! dequantizing each weight as it is used, at 2.755 s/token. Routing every
//! matmul through `DeviceWeights::matvec` instead would be worse, not better:
//! that API copies activations in, synchronises and copies results out, and a
//! token runs roughly 1400 matmuls. At the measured 0.12 ms round-trip that is
//! 167 ms/token — about 6 tok/s, a ceiling no kernel improvement can break.
//!
//! SO NOTHING CROSSES THE BUS. Weights are uploaded once and stay. Activations
//! live in device buffers for the life of the model. Every operation is a
//! launch taking device pointers, queued without synchronising. One
//! `hipDeviceSynchronize` happens after the whole token, and the only value
//! read back is the argmax — a single `i32`.
//!
//! CORRECTNESS IS NOT NEGOTIABLE HERE. This must reproduce
//! `llamd-runtime::forward_layer` op for op, because that path is what agrees
//! with llama.cpp on all five token-agreement prompts. The device kernels
//! carry comments naming the CPU line each mirrors, and
//! `llamd-testing/tests/device_forward.rs` compares the two directly.

use crate::{Error, Result};
use std::ffi::c_void;

/// A device allocation, freed on drop.
///
/// Not `Clone`: two owners of one `hipFree` is a double free, and the type
/// system is the cheapest place to prevent it.
pub struct DevBuf {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device pointer is just an address; it is valid on any host thread
// for the same device. Access is serialised by the caller holding &mut, and
// nothing here touches the shared scratch that `SCRATCH` in lib.rs guards.
#[allow(unsafe_code)]
unsafe impl Send for DevBuf {}
#[allow(unsafe_code)]
unsafe impl Sync for DevBuf {}

impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::malformed("device alloc", "zero bytes"));
        }
        #[cfg(not(rocm))]
        {
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: a plain allocation of a non-zero size; the result is
            // checked for null before it is stored.
            #[allow(unsafe_code)]
            let ptr = unsafe { ffi::llamd_fwd_alloc(bytes) };
            if ptr.is_null() {
                return Err(Error::other(format!(
                    "device allocation of {} MiB failed",
                    bytes / (1 << 20)
                )));
            }
            Ok(DevBuf { ptr, bytes })
        }
    }

    pub fn from_bytes(host: &[u8]) -> Result<Self> {
        let b = DevBuf::new(host.len())?;
        b.upload_bytes(host)?;
        Ok(b)
    }

    pub fn from_f32(host: &[f32]) -> Result<Self> {
        let b = DevBuf::new(std::mem::size_of_val(host))?;
        // SAFETY: f32 has no padding or invalid bit patterns, so its bytes are
        // a valid &[u8] of exactly this length.
        #[allow(unsafe_code)]
        let raw = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        b.upload_bytes(raw)?;
        Ok(b)
    }

    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = DevBuf::new(bytes)?;
        b.zero()?;
        Ok(b)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn as_f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }

    pub fn upload_bytes(&self, host: &[u8]) -> Result<()> {
        if host.len() > self.bytes {
            return Err(Error::Shape {
                expected: vec![self.bytes],
                got: vec![host.len()],
            });
        }
        #[cfg(not(rocm))]
        {
            let _ = host;
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: `host.len()` bytes are read from a live slice and written
            // to an allocation just checked to be at least that large.
            #[allow(unsafe_code)]
            let rc = unsafe {
                ffi::llamd_fwd_upload(self.ptr, host.as_ptr() as *const c_void, host.len())
            };
            if rc != 0 {
                return Err(Error::other("device upload failed"));
            }
            Ok(())
        }
    }

    pub fn download_f32(&self, out: &mut [f32]) -> Result<()> {
        let want = std::mem::size_of_val(out);
        if want > self.bytes {
            return Err(Error::Shape {
                expected: vec![self.bytes],
                got: vec![want],
            });
        }
        #[cfg(not(rocm))]
        {
            let _ = want;
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: `want` bytes are written into a slice that is exactly
            // that many bytes long, read from an allocation at least as large.
            #[allow(unsafe_code)]
            let rc =
                unsafe { ffi::llamd_fwd_download(out.as_mut_ptr() as *mut c_void, self.ptr, want) };
            if rc != 0 {
                return Err(Error::other("device download failed"));
            }
            Ok(())
        }
    }

    /// Upload i32 values — used for the per-token `[token, pos]` metadata.
    pub fn upload_i32(&self, host: &[i32]) -> Result<()> {
        // SAFETY: i32 has no padding or invalid bit patterns, so its bytes are
        // a valid &[u8] of exactly this length.
        #[allow(unsafe_code)]
        let raw = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        self.upload_bytes(raw)
    }

    pub fn download_i32(&self, out: &mut [i32]) -> Result<()> {
        let want = std::mem::size_of_val(out);
        #[cfg(not(rocm))]
        {
            let _ = want;
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: as `download_f32`; i32 has no invalid bit patterns.
            #[allow(unsafe_code)]
            let rc =
                unsafe { ffi::llamd_fwd_download(out.as_mut_ptr() as *mut c_void, self.ptr, want) };
            if rc != 0 {
                return Err(Error::other("device download failed"));
            }
            Ok(())
        }
    }

    pub fn zero(&self) -> Result<()> {
        #[cfg(not(rocm))]
        {
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: writes `self.bytes` zeroes into its own allocation.
            #[allow(unsafe_code)]
            let rc = unsafe { ffi::llamd_fwd_memset(self.ptr, 0, self.bytes) };
            if rc != 0 {
                return Err(Error::other("device memset failed"));
            }
            Ok(())
        }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        #[cfg(rocm)]
        {
            // SAFETY: `self.ptr` came from `llamd_fwd_alloc` and this is the
            // only release; `DevBuf` is not `Clone`, so no second owner exists.
            #[allow(unsafe_code)]
            unsafe {
                ffi::llamd_fwd_free(self.ptr)
            };
        }
    }
}

/// Free VRAM in bytes, for deciding whether a model fits before trying.
pub fn free_memory() -> usize {
    #[cfg(not(rocm))]
    {
        0
    }
    #[cfg(rocm)]
    {
        // SAFETY: a query with no arguments.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_fwd_free_mem()
        }
    }
}

/// Block until every queued launch has finished, and report a launch error.
///
/// Called ONCE per token. Calling it between operations would reintroduce
/// exactly the stall this module exists to remove.
pub fn sync() -> Result<()> {
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        // SAFETY: no arguments; both report status through their return value.
        #[allow(unsafe_code)]
        let rc = unsafe { ffi::llamd_fwd_sync() };
        if rc != 0 {
            return Err(Error::other("device synchronise failed"));
        }
        #[allow(unsafe_code)]
        let err = unsafe { ffi::llamd_fwd_last_error() };
        if err != 0 {
            return Err(Error::other("a queued kernel reported an error"));
        }
        Ok(())
    }
}

#[allow(unsafe_code)]
pub(crate) mod ffi {
    //! The device-pointer launch surface. Every function here queues work and
    //! returns immediately; none synchronises.
    use std::ffi::c_void;
    use std::os::raw::c_int;

    unsafe extern "C" {
        pub fn llamd_fwd_alloc(bytes: usize) -> *mut c_void;
        pub fn llamd_fwd_free(p: *mut c_void);
        pub fn llamd_fwd_upload(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn llamd_fwd_download(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn llamd_fwd_memset(p: *mut c_void, value: c_int, bytes: usize) -> c_int;
        pub fn llamd_fwd_sync() -> c_int;
        pub fn llamd_fwd_free_mem() -> usize;
        pub fn llamd_fwd_last_error() -> c_int;

        pub fn llamd_fwd_embed(table: *const c_void, token: c_int, out: *mut f32, blocks: c_int);
        pub fn llamd_fwd_rms_norm_mul(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            n: c_int,
            eps: f32,
        );
        pub fn llamd_fwd_head_rms_norm_mul(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            n_head: c_int,
            hd: c_int,
            eps: f32,
        );
        pub fn llamd_fwd_rope(x: *mut f32, pos: c_int, n_head: c_int, hd: c_int, freq_base: f32);
        pub fn llamd_fwd_kv_store(
            kc: *mut f32,
            vc: *mut f32,
            k: *const f32,
            v: *const f32,
            pos: c_int,
            kv_dim: c_int,
        );
        pub fn llamd_fwd_attention(
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            out: *mut f32,
            scores: *mut f32,
            n_head: c_int,
            n_head_kv: c_int,
            hd: c_int,
            n_kv: c_int,
        );
        pub fn llamd_fwd_add(a: *const f32, b: *const f32, out: *mut f32, n: c_int);
        pub fn llamd_fwd_silu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: c_int);
        pub fn llamd_fwd_router(
            logits: *const f32,
            n_expert: c_int,
            n_used: c_int,
            ids: *mut i32,
            weights: *mut f32,
            probs: *mut f32,
        );
        pub fn llamd_fwd_moe_combine(
            down: *const f32,
            weights: *const f32,
            out: *mut f32,
            n_used: c_int,
            n_embd: c_int,
        );
        pub fn llamd_fwd_argmax(
            x: *const f32,
            n: c_int,
            scratch_v: *mut f32,
            scratch_i: *mut i32,
            out: *mut i32,
            nblocks: c_int,
        );
        pub fn llamd_fwd_matvec_f32(
            w: *const f32,
            x: *const f32,
            out: *mut f32,
            k: c_int,
            rows: c_int,
        );
        pub fn llamd_fwd_matvec(
            w: *const c_void,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
        );
        pub fn llamd_fwd_matvec_grouped(
            w: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );
        pub fn llamd_fwd_matvec_grouped_pa(
            w: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );
        pub fn llamd_fwd_quantize(x: *const f32, qs: *mut i8, d: *mut f32, nblocks: c_int);
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_matvec_x3_splitk(
            w0: *const c_void,
            w1: *const c_void,
            w2: *const c_void,
            aq: *const i8,
            ad: *const f32,
            o0: *mut f32,
            o1: *mut f32,
            o2: *mut f32,
            blocks: c_int,
            rows0: c_int,
            rows1: c_int,
            rows2: c_int,
            nsub: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_matvec_grouped_x2_splitk(
            wa: *const c_void,
            wb: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            oa: *mut f32,
            ob: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            nsub: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_matvec_grouped_pa_splitk(
            w: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            nsub: c_int,
        );
        pub fn llamd_fwd_planar_matvec(
            base: *const c_void,
            off: *const usize,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_planar_grouped(
            base: *const c_void,
            off: *const usize,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_planar_grouped_pa(
            base: *const c_void,
            off: *const usize,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_planar_x3(
            b0: *const c_void,
            f0: *const usize,
            b1: *const c_void,
            f1: *const usize,
            b2: *const c_void,
            f2: *const usize,
            aq: *const i8,
            ad: *const f32,
            o0: *mut f32,
            o1: *mut f32,
            o2: *mut f32,
            blocks: c_int,
            rows0: c_int,
            rows1: c_int,
            rows2: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_planar_grouped_x2(
            ba: *const c_void,
            fa: *const usize,
            bb: *const c_void,
            fb: *const usize,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            oa: *mut f32,
            ob: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );
        pub fn llamd_rocm_planar_matvec(
            base: *const c_void,
            off: *const usize,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_rocm_planar_bench(
            base: *const c_void,
            off: *const usize,
            aq: *const i8,
            ad: *const f32,
            blocks: c_int,
            rows: c_int,
            iters: c_int,
            ms_out: *mut f32,
        ) -> c_int;
        pub fn llamd_rocm_splitk_bench(
            w: *const c_void,
            aq: *const i8,
            ad: *const f32,
            blocks: c_int,
            rows: c_int,
            nsub: c_int,
            iters: c_int,
            ms_out: *mut f32,
        ) -> c_int;
        pub fn llamd_fwd_matvec_splitk(
            w: *const c_void,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            nsub: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_matvec_x3(
            w0: *const c_void,
            w1: *const c_void,
            w2: *const c_void,
            aq: *const i8,
            ad: *const f32,
            o0: *mut f32,
            o1: *mut f32,
            o2: *mut f32,
            blocks: c_int,
            rows0: c_int,
            rows1: c_int,
            rows2: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_matvec_grouped_x2(
            wa: *const c_void,
            wb: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            oa: *mut f32,
            ob: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
        );

        pub fn llamd_fwd_stream_init() -> c_int;
        pub fn llamd_fwd_norm_rope(
            x: *mut f32,
            w: *const f32,
            n_head: c_int,
            hd: c_int,
            meta: *const i32,
            freq_base: f32,
            eps: f32,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_k_norm_rope_store(
            k: *mut f32,
            w: *const f32,
            v: *const f32,
            kc: *mut f32,
            vc: *mut f32,
            n_head_kv: c_int,
            hd: c_int,
            kv_dim: c_int,
            meta: *const i32,
            freq_base: f32,
            eps: f32,
        );
        pub fn llamd_fwd_rms_norm_quant(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            qs: *mut i8,
            d: *mut f32,
            inv_scratch: *mut f32,
            n: c_int,
            eps: f32,
        );
        pub fn llamd_fwd_moe_combine_add(
            down: *const f32,
            weights: *const f32,
            residual: *const f32,
            out: *mut f32,
            n_used: c_int,
            n_embd: c_int,
        );
        pub fn llamd_fwd_embed_meta(
            table: *const c_void,
            meta: *const i32,
            out: *mut f32,
            blocks: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_fwd_attention_meta(
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            out: *mut f32,
            scores: *mut f32,
            n_head: c_int,
            n_head_kv: c_int,
            hd: c_int,
            meta: *const i32,
            max_kv: c_int,
        );
        pub fn llamd_fwd_begin_capture() -> c_int;
        pub fn llamd_fwd_end_capture() -> c_int;
        pub fn llamd_fwd_graph_launch() -> c_int;
        pub fn llamd_fwd_has_graph() -> c_int;
        pub fn llamd_fwd_graph_destroy();
    }
}

/// Create the stream every device operation uses.
///
/// Must be called before any of them. The default stream cannot be captured
/// into a graph, so everything goes to a stream of our own.
pub fn stream_init() -> Result<()> {
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        // SAFETY: no arguments; idempotent, reports failure by return value.
        #[allow(unsafe_code)]
        let rc = unsafe { ffi::llamd_fwd_stream_init() };
        if rc != 0 {
            return Err(Error::other("could not create the device stream"));
        }
        Ok(())
    }
}

/// Graph capture. See the note in `llamd_forward.hip`.
///
/// NOTHING EXECUTES between `begin` and `end` — capture records the launches, it
/// does not run them. So the first real output comes from the first `replay`,
/// and any buffer the captured token reads must already be in a sane state.
pub mod graph {
    use super::{Error, Result, ffi};

    pub fn begin() -> Result<()> {
        #[cfg(not(rocm))]
        {
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: no arguments; fails by return value.
            #[allow(unsafe_code)]
            let rc = unsafe { ffi::llamd_fwd_begin_capture() };
            if rc != 0 {
                return Err(Error::other("graph capture could not start"));
            }
            Ok(())
        }
    }

    pub fn end() -> Result<()> {
        #[cfg(not(rocm))]
        {
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: no arguments; fails by return value.
            #[allow(unsafe_code)]
            let rc = unsafe { ffi::llamd_fwd_end_capture() };
            if rc != 0 {
                return Err(Error::other("graph capture could not be instantiated"));
            }
            Ok(())
        }
    }

    pub fn replay() -> Result<()> {
        #[cfg(not(rocm))]
        {
            Err(Error::unsupported("rocm", "not built with ROCm support"))
        }
        #[cfg(rocm)]
        {
            // SAFETY: no arguments; refuses if no graph is instantiated.
            #[allow(unsafe_code)]
            let rc = unsafe { ffi::llamd_fwd_graph_launch() };
            if rc != 0 {
                return Err(Error::other("graph launch failed"));
            }
            Ok(())
        }
    }

    pub fn exists() -> bool {
        #[cfg(not(rocm))]
        {
            false
        }
        #[cfg(rocm)]
        {
            // SAFETY: no arguments.
            #[allow(unsafe_code)]
            unsafe {
                ffi::llamd_fwd_has_graph() != 0
            }
        }
    }

    pub fn destroy() {
        #[cfg(rocm)]
        {
            // SAFETY: no arguments; safe to call with no graph.
            #[allow(unsafe_code)]
            unsafe {
                ffi::llamd_fwd_graph_destroy()
            }
        }
    }
}

/// Queue one operation. Each is a thin, checked wrapper over a launch.
///
/// These are `pub` so `llamd-runtime` can compose a forward pass without this
/// crate needing to know about models, and deliberately take `&DevBuf` rather
/// than raw pointers so a caller cannot pass a freed or mis-sized buffer.
pub mod op {
    use super::{DevBuf, Error, Result, ffi};

    /// Guards every launch: a buffer too small for what the kernel will write
    /// is a memory error on the device, which surfaces later and elsewhere.
    fn need(buf: &DevBuf, floats: usize, what: &str) -> Result<()> {
        if buf.bytes() < floats * 4 {
            return Err(Error::malformed(
                "device buffer",
                format!(
                    "{what}: buffer holds {} floats, needs {floats}",
                    buf.bytes() / 4
                ),
            ));
        }
        Ok(())
    }

    macro_rules! launch {
        ($call:expr) => {{
            #[cfg(not(rocm))]
            {
                return Err(Error::unsupported("rocm", "not built with ROCm support"));
            }
            #[cfg(rocm)]
            // SAFETY: every pointer comes from a live DevBuf whose size was
            // checked above against what the kernel writes.
            #[allow(unsafe_code)]
            unsafe {
                $call
            }
        }};
    }

    pub fn embed(table: &DevBuf, token: u32, out: &DevBuf, blocks: usize) -> Result<()> {
        need(out, blocks * 256, "embed out")?;
        launch!(ffi::llamd_fwd_embed(
            table.ptr(),
            token as i32,
            out.as_f32(),
            blocks as i32
        ));
        Ok(())
    }

    pub fn rms_norm_mul(x: &DevBuf, w: &DevBuf, out: &DevBuf, n: usize, eps: f32) -> Result<()> {
        need(x, n, "rms_norm x")?;
        need(w, n, "rms_norm w")?;
        need(out, n, "rms_norm out")?;
        launch!(ffi::llamd_fwd_rms_norm_mul(
            x.as_f32(),
            w.as_f32(),
            out.as_f32(),
            n as i32,
            eps
        ));
        Ok(())
    }

    pub fn head_rms_norm_mul(
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        n_head: usize,
        hd: usize,
        eps: f32,
    ) -> Result<()> {
        need(x, n_head * hd, "head_rms x")?;
        need(w, hd, "head_rms w")?;
        need(out, n_head * hd, "head_rms out")?;
        launch!(ffi::llamd_fwd_head_rms_norm_mul(
            x.as_f32(),
            w.as_f32(),
            out.as_f32(),
            n_head as i32,
            hd as i32,
            eps
        ));
        Ok(())
    }

    pub fn rope(x: &DevBuf, pos: u32, n_head: usize, hd: usize, freq_base: f32) -> Result<()> {
        need(x, n_head * hd, "rope x")?;
        launch!(ffi::llamd_fwd_rope(
            x.as_f32(),
            pos as i32,
            n_head as i32,
            hd as i32,
            freq_base
        ));
        Ok(())
    }

    pub fn kv_store(
        kc: &DevBuf,
        vc: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        pos: usize,
        kv_dim: usize,
    ) -> Result<()> {
        need(kc, (pos + 1) * kv_dim, "k cache")?;
        need(vc, (pos + 1) * kv_dim, "v cache")?;
        need(k, kv_dim, "k")?;
        need(v, kv_dim, "v")?;
        launch!(ffi::llamd_fwd_kv_store(
            kc.as_f32(),
            vc.as_f32(),
            k.as_f32(),
            v.as_f32(),
            pos as i32,
            kv_dim as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        q: &DevBuf,
        kc: &DevBuf,
        vc: &DevBuf,
        out: &DevBuf,
        scores: &DevBuf,
        n_head: usize,
        n_head_kv: usize,
        hd: usize,
        n_kv: usize,
    ) -> Result<()> {
        if n_kv == 0 {
            return Err(Error::malformed("attention", "nothing to attend to"));
        }
        if n_head % n_head_kv != 0 {
            return Err(Error::malformed(
                "attention",
                format!("{n_head} query heads do not divide into {n_head_kv} kv heads"),
            ));
        }
        need(q, n_head * hd, "attn q")?;
        need(out, n_head * hd, "attn out")?;
        need(scores, n_head * n_kv, "attn scores")?;
        launch!(ffi::llamd_fwd_attention(
            q.as_f32(),
            kc.as_f32(),
            vc.as_f32(),
            out.as_f32(),
            scores.as_f32(),
            n_head as i32,
            n_head_kv as i32,
            hd as i32,
            n_kv as i32
        ));
        Ok(())
    }

    pub fn add(a: &DevBuf, b: &DevBuf, out: &DevBuf, n: usize) -> Result<()> {
        need(a, n, "add a")?;
        need(b, n, "add b")?;
        need(out, n, "add out")?;
        launch!(ffi::llamd_fwd_add(
            a.as_f32(),
            b.as_f32(),
            out.as_f32(),
            n as i32
        ));
        Ok(())
    }

    pub fn silu_mul(gate: &DevBuf, up: &DevBuf, out: &DevBuf, n: usize) -> Result<()> {
        need(gate, n, "silu gate")?;
        need(up, n, "silu up")?;
        need(out, n, "silu out")?;
        launch!(ffi::llamd_fwd_silu_mul(
            gate.as_f32(),
            up.as_f32(),
            out.as_f32(),
            n as i32
        ));
        Ok(())
    }

    pub fn router(
        logits: &DevBuf,
        n_expert: usize,
        n_used: usize,
        ids: &DevBuf,
        weights: &DevBuf,
        probs: &DevBuf,
    ) -> Result<()> {
        if n_used == 0 || n_used > n_expert {
            return Err(Error::malformed(
                "moe routing",
                format!("{n_used} experts requested of {n_expert}"),
            ));
        }
        // The kernel stages the probabilities in a fixed 1024-float LDS array.
        if n_expert > 1024 {
            return Err(Error::unsupported(
                "moe routing",
                format!("{n_expert} experts exceeds the kernel's 1024 LDS slots"),
            ));
        }
        need(logits, n_expert, "router logits")?;
        need(probs, n_expert, "router probs")?;
        need(ids, n_used, "router ids")?;
        need(weights, n_used, "router weights")?;
        launch!(ffi::llamd_fwd_router(
            logits.as_f32(),
            n_expert as i32,
            n_used as i32,
            ids.ptr() as *mut i32,
            weights.as_f32(),
            probs.as_f32()
        ));
        Ok(())
    }

    pub fn moe_combine(
        down: &DevBuf,
        weights: &DevBuf,
        out: &DevBuf,
        n_used: usize,
        n_embd: usize,
    ) -> Result<()> {
        need(down, n_used * n_embd, "moe down")?;
        need(out, n_embd, "moe out")?;
        launch!(ffi::llamd_fwd_moe_combine(
            down.as_f32(),
            weights.as_f32(),
            out.as_f32(),
            n_used as i32,
            n_embd as i32
        ));
        Ok(())
    }

    pub fn argmax(
        x: &DevBuf,
        n: usize,
        scratch_v: &DevBuf,
        scratch_i: &DevBuf,
        out: &DevBuf,
        nblocks: usize,
    ) -> Result<()> {
        need(x, n, "argmax x")?;
        need(scratch_v, nblocks, "argmax scratch")?;
        launch!(ffi::llamd_fwd_argmax(
            x.as_f32(),
            n as i32,
            scratch_v.as_f32(),
            scratch_i.ptr() as *mut i32,
            out.ptr() as *mut i32,
            nblocks as i32
        ));
        Ok(())
    }

    /// Per-head RMS norm then RoPE, one launch. `w` may be `None` for a model
    /// without QK-norm. Reads the position from `meta`.
    pub fn norm_rope(
        x: &DevBuf,
        w: Option<&DevBuf>,
        meta: &DevBuf,
        n_head: usize,
        hd: usize,
        freq_base: f32,
        eps: f32,
    ) -> Result<()> {
        need(x, n_head * hd, "norm_rope x")?;
        if let Some(w) = w {
            need(w, hd, "norm_rope w")?;
        }
        let wp = w
            .map(|b| b.as_f32() as *const f32)
            .unwrap_or(std::ptr::null());
        launch!(ffi::llamd_fwd_norm_rope(
            x.as_f32(),
            wp,
            n_head as i32,
            hd as i32,
            meta.ptr() as *const i32,
            freq_base,
            eps
        ));
        Ok(())
    }

    /// The K side: norm, RoPE, then store K and V into the cache. One launch
    /// where there were three. The store must follow the rotation — the cache
    /// holds roped keys.
    #[allow(clippy::too_many_arguments)]
    pub fn k_norm_rope_store(
        k: &DevBuf,
        w: Option<&DevBuf>,
        v: &DevBuf,
        kc: &DevBuf,
        vc: &DevBuf,
        meta: &DevBuf,
        n_head_kv: usize,
        hd: usize,
        freq_base: f32,
        eps: f32,
    ) -> Result<()> {
        let kv_dim = n_head_kv * hd;
        need(k, kv_dim, "k_norm_rope k")?;
        need(v, kv_dim, "k_norm_rope v")?;
        if let Some(w) = w {
            need(w, hd, "k_norm_rope w")?;
        }
        let wp = w
            .map(|b| b.as_f32() as *const f32)
            .unwrap_or(std::ptr::null());
        launch!(ffi::llamd_fwd_k_norm_rope_store(
            k.as_f32(),
            wp,
            v.as_f32(),
            kc.as_f32(),
            vc.as_f32(),
            n_head_kv as i32,
            hd as i32,
            kv_dim as i32,
            meta.ptr() as *const i32,
            freq_base,
            eps
        ));
        Ok(())
    }

    /// rms_norm(x) * w, emitted as f32 AND quantized, in one pass.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_quant(
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        inv: &DevBuf,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        need(x, n, "rms_norm_quant x")?;
        need(w, n, "rms_norm_quant w")?;
        need(out, n, "rms_norm_quant out")?;
        need(d, n / 256, "rms_norm_quant d")?;
        need(inv, 1, "rms_norm_quant inv")?;
        if qs.bytes() < n {
            return Err(Error::malformed("rms_norm_quant qs", "too small"));
        }
        launch!(ffi::llamd_fwd_rms_norm_quant(
            x.as_f32(),
            w.as_f32(),
            out.as_f32(),
            qs.ptr() as *mut i8,
            d.as_f32(),
            inv.as_f32(),
            n as i32,
            eps
        ));
        Ok(())
    }

    /// `out = residual + sum_e weights[e] * down[e]`, one launch.
    pub fn moe_combine_add(
        down: &DevBuf,
        weights: &DevBuf,
        residual: &DevBuf,
        out: &DevBuf,
        n_used: usize,
        n_embd: usize,
    ) -> Result<()> {
        need(down, n_used * n_embd, "moe down")?;
        need(out, n_embd, "moe out")?;
        launch!(ffi::llamd_fwd_moe_combine_add(
            down.as_f32(),
            weights.as_f32(),
            residual.as_f32(),
            out.as_f32(),
            n_used as i32,
            n_embd as i32
        ));
        Ok(())
    }

    /// Embedding lookup reading the token id from `meta`, so a captured graph
    /// does not need re-recording per token.
    pub fn embed_meta(table: &DevBuf, meta: &DevBuf, out: &DevBuf, blocks: usize) -> Result<()> {
        need(out, blocks * 256, "embed out")?;
        launch!(ffi::llamd_fwd_embed_meta(
            table.ptr(),
            meta.ptr() as *const i32,
            out.as_f32(),
            blocks as i32
        ));
        Ok(())
    }

    /// Attention reading n_kv from `meta`. `max_kv` is the scores stride, fixed
    /// for the life of the model, which is what lets the graph reuse it.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_meta(
        q: &DevBuf,
        kc: &DevBuf,
        vc: &DevBuf,
        out: &DevBuf,
        scores: &DevBuf,
        meta: &DevBuf,
        n_head: usize,
        n_head_kv: usize,
        hd: usize,
        max_kv: usize,
    ) -> Result<()> {
        if n_head % n_head_kv != 0 {
            return Err(Error::malformed(
                "attention",
                format!("{n_head} query heads do not divide into {n_head_kv} kv heads"),
            ));
        }
        need(q, n_head * hd, "attn q")?;
        need(out, n_head * hd, "attn out")?;
        need(scores, n_head * max_kv, "attn scores")?;
        launch!(ffi::llamd_fwd_attention_meta(
            q.as_f32(),
            kc.as_f32(),
            vc.as_f32(),
            out.as_f32(),
            scores.as_f32(),
            n_head as i32,
            n_head_kv as i32,
            hd as i32,
            meta.ptr() as *const i32,
            max_kv as i32
        ));
        Ok(())
    }

    pub fn matvec_f32(w: &DevBuf, x: &DevBuf, out: &DevBuf, k: usize, rows: usize) -> Result<()> {
        need(w, k * rows, "matvec_f32 w")?;
        need(x, k, "matvec_f32 x")?;
        need(out, rows, "matvec_f32 out")?;
        launch!(ffi::llamd_fwd_matvec_f32(
            w.as_f32(),
            x.as_f32(),
            out.as_f32(),
            k as i32,
            rows as i32
        ));
        Ok(())
    }

    /// Matvec with `nsub` waves cooperating on each output row.
    ///
    /// For matrices with too few rows to fill the card: attn_out's 2048 rows
    /// are 2048 waves against a 4096-wave capacity, so half the GPU idles.
    /// `nsub` must divide 8 (the waves in a workgroup) and should not exceed
    /// `blocks`, or waves are assigned no work.
    pub fn matvec_splitk(
        w: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
        nsub: usize,
    ) -> Result<()> {
        if nsub == 0 || 8 % nsub != 0 {
            return Err(Error::malformed(
                "matvec_splitk",
                format!("nsub {nsub} must divide 8"),
            ));
        }
        if nsub > blocks {
            return Err(Error::malformed(
                "matvec_splitk",
                format!("nsub {nsub} exceeds {blocks} blocks; waves would idle"),
            ));
        }
        if nsub == 1 {
            return matvec(w, qs, d, out, blocks, rows);
        }
        need(out, rows, "splitk out")?;
        need(d, blocks, "splitk d")?;
        launch!(ffi::llamd_fwd_matvec_splitk(
            w.ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            nsub as i32
        ));
        Ok(())
    }

    /// Q, K and V in one launch. All three read the same quantized activation.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_x3(
        w: (&DevBuf, &DevBuf, &DevBuf),
        qs: &DevBuf,
        d: &DevBuf,
        out: (&DevBuf, &DevBuf, &DevBuf),
        blocks: usize,
        rows: (usize, usize, usize),
    ) -> Result<()> {
        need(out.0, rows.0, "x3 out0")?;
        need(out.1, rows.1, "x3 out1")?;
        need(out.2, rows.2, "x3 out2")?;
        need(d, blocks, "x3 d")?;
        launch!(ffi::llamd_fwd_matvec_x3(
            w.0.ptr(),
            w.1.ptr(),
            w.2.ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.0.as_f32(),
            out.1.as_f32(),
            out.2.as_f32(),
            blocks as i32,
            rows.0 as i32,
            rows.1 as i32,
            rows.2 as i32
        ));
        Ok(())
    }

    /// Validates `nsub` against the shape. Measured per shape rather than
    /// reasoned about: an occupancy argument predicted a gain for attn_out and
    /// the engine got slower, while attn_q gained 1.52x.
    fn check_nsub(nsub: usize, blocks: usize) -> Result<()> {
        if nsub == 0 || 8 % nsub != 0 {
            return Err(Error::malformed("nsub", format!("{nsub} must divide 8")));
        }
        if nsub > blocks {
            return Err(Error::malformed(
                "nsub",
                format!("{nsub} exceeds {blocks} blocks; waves would idle"),
            ));
        }
        Ok(())
    }

    /// Q, K and V in one launch, with `nsub` waves per row.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_x3_splitk(
        w: (&DevBuf, &DevBuf, &DevBuf),
        qs: &DevBuf,
        d: &DevBuf,
        out: (&DevBuf, &DevBuf, &DevBuf),
        blocks: usize,
        rows: (usize, usize, usize),
        nsub: usize,
    ) -> Result<()> {
        check_nsub(nsub, blocks)?;
        // At nsub=1 there is nothing to combine, and the split-k kernel still
        // pays an LDS write, a barrier and the combine loop — measured ~5% on
        // the engine. Use the dedicated kernel instead.
        if nsub == 1 {
            return matvec_x3(w, qs, d, out, blocks, rows);
        }
        need(out.0, rows.0, "x3 out0")?;
        need(out.1, rows.1, "x3 out1")?;
        need(out.2, rows.2, "x3 out2")?;
        need(d, blocks, "x3 d")?;
        launch!(ffi::llamd_fwd_matvec_x3_splitk(
            w.0.ptr(),
            w.1.ptr(),
            w.2.ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.0.as_f32(),
            out.1.as_f32(),
            out.2.as_f32(),
            blocks as i32,
            rows.0 as i32,
            rows.1 as i32,
            rows.2 as i32,
            nsub as i32
        ));
        Ok(())
    }

    /// Expert gate and up in one launch, with `nsub` waves per row.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_x2_splitk(
        wa: &DevBuf,
        wb: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        oa: &DevBuf,
        ob: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
        nsub: usize,
    ) -> Result<()> {
        check_nsub(nsub, blocks)?;
        if nsub == 1 {
            return matvec_grouped_x2(wa, wb, ids, qs, d, oa, ob, blocks, rows, n_used);
        }
        need(oa, rows * n_used, "x2 oa")?;
        need(ob, rows * n_used, "x2 ob")?;
        need(d, blocks, "x2 d")?;
        need(ids, n_used, "x2 ids")?;
        launch!(ffi::llamd_fwd_matvec_grouped_x2_splitk(
            wa.ptr(),
            wb.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            oa.as_f32(),
            ob.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32,
            nsub as i32
        ));
        Ok(())
    }

    /// The down projection — per-expert activations — with `nsub` waves per row.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_per_act_splitk(
        w: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
        nsub: usize,
    ) -> Result<()> {
        check_nsub(nsub, blocks)?;
        if nsub == 1 {
            return matvec_grouped_per_act(w, ids, qs, d, out, blocks, rows, n_used);
        }
        need(out, rows * n_used, "pa out")?;
        need(d, blocks * n_used, "pa d")?;
        need(ids, n_used, "pa ids")?;
        launch!(ffi::llamd_fwd_matvec_grouped_pa_splitk(
            w.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32,
            nsub as i32
        ));
        Ok(())
    }

    /// The expert gate and up projections in one launch: both read the same
    /// activation, and 768 workgroups each becomes 1536 together.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_x2(
        wa: &DevBuf,
        wb: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        oa: &DevBuf,
        ob: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(oa, rows * n_used, "x2 oa")?;
        need(ob, rows * n_used, "x2 ob")?;
        need(d, blocks, "x2 d")?;
        need(ids, n_used, "x2 ids")?;
        launch!(ffi::llamd_fwd_matvec_grouped_x2(
            wa.ptr(),
            wb.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            oa.as_f32(),
            ob.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }

    // ---- planar variants -------------------------------------------------
    //
    // Same arithmetic, 16-byte loads. Measured on a cold 243 MiB tensor —
    // large enough to defeat this card's 64 MB Infinity Cache — the planar
    // kernel reaches 510.6 GB/s against the interleaved 439.7, a 1.16x gain
    // and 80% of the 637.8 GB/s a pure streaming read achieves.

    pub fn planar_matvec(
        w: &crate::PlanarWeights,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        rows: usize,
    ) -> Result<()> {
        need(out, rows, "planar out")?;
        need(d, w.blocks(), "planar d")?;
        launch!(ffi::llamd_fwd_planar_matvec(
            w.buf().ptr(),
            w.offsets().as_ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            w.blocks() as i32,
            rows as i32
        ));
        Ok(())
    }

    pub fn planar_grouped(
        w: &crate::PlanarWeights,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(out, rows * n_used, "planar grouped out")?;
        need(d, w.blocks(), "planar grouped d")?;
        need(ids, n_used, "planar grouped ids")?;
        launch!(ffi::llamd_fwd_planar_grouped(
            w.buf().ptr(),
            w.offsets().as_ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            w.blocks() as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }

    /// The down projection: one activation per routed expert.
    pub fn planar_grouped_per_act(
        w: &crate::PlanarWeights,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(out, rows * n_used, "planar pa out")?;
        need(d, w.blocks() * n_used, "planar pa d")?;
        need(ids, n_used, "planar pa ids")?;
        launch!(ffi::llamd_fwd_planar_grouped_pa(
            w.buf().ptr(),
            w.offsets().as_ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            w.blocks() as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }

    pub fn planar_x3(
        w: (
            &crate::PlanarWeights,
            &crate::PlanarWeights,
            &crate::PlanarWeights,
        ),
        qs: &DevBuf,
        d: &DevBuf,
        out: (&DevBuf, &DevBuf, &DevBuf),
        rows: (usize, usize, usize),
    ) -> Result<()> {
        need(out.0, rows.0, "planar x3 out0")?;
        need(out.1, rows.1, "planar x3 out1")?;
        need(out.2, rows.2, "planar x3 out2")?;
        need(d, w.0.blocks(), "planar x3 d")?;
        if w.1.blocks() != w.0.blocks() || w.2.blocks() != w.0.blocks() {
            return Err(Error::malformed(
                "planar x3",
                "all three matrices must share the activation's block count",
            ));
        }
        launch!(ffi::llamd_fwd_planar_x3(
            w.0.buf().ptr(),
            w.0.offsets().as_ptr(),
            w.1.buf().ptr(),
            w.1.offsets().as_ptr(),
            w.2.buf().ptr(),
            w.2.offsets().as_ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.0.as_f32(),
            out.1.as_f32(),
            out.2.as_f32(),
            w.0.blocks() as i32,
            rows.0 as i32,
            rows.1 as i32,
            rows.2 as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn planar_grouped_x2(
        wa: &crate::PlanarWeights,
        wb: &crate::PlanarWeights,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        oa: &DevBuf,
        ob: &DevBuf,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(oa, rows * n_used, "planar x2 oa")?;
        need(ob, rows * n_used, "planar x2 ob")?;
        need(d, wa.blocks(), "planar x2 d")?;
        need(ids, n_used, "planar x2 ids")?;
        if wb.blocks() != wa.blocks() {
            return Err(Error::malformed("planar x2", "block counts differ"));
        }
        launch!(ffi::llamd_fwd_planar_grouped_x2(
            wa.buf().ptr(),
            wa.offsets().as_ptr(),
            wb.buf().ptr(),
            wb.offsets().as_ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            oa.as_f32(),
            ob.as_f32(),
            wa.blocks() as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }

    pub fn quantize(x: &DevBuf, qs: &DevBuf, d: &DevBuf, nblocks: usize) -> Result<()> {
        need(x, nblocks * 256, "quantize x")?;
        need(d, nblocks, "quantize d")?;
        if qs.bytes() < nblocks * 256 {
            return Err(Error::malformed("quantize qs", "too small"));
        }
        launch!(ffi::llamd_fwd_quantize(
            x.as_f32(),
            qs.ptr() as *mut i8,
            d.as_f32(),
            nblocks as i32
        ));
        Ok(())
    }

    pub fn matvec(
        w: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
    ) -> Result<()> {
        need(out, rows, "matvec out")?;
        need(d, blocks, "matvec d")?;
        launch!(ffi::llamd_fwd_matvec(
            w.ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32
        ));
        Ok(())
    }

    /// The grouped matvec with a SEPARATE activation per routed expert.
    ///
    /// Used for the down projection only. The gate and up projections share
    /// one activation (every expert reads the same normalised x); the down
    /// projection does not, because each expert's input is its own
    /// silu(gate_e) * up_e. Feeding it the shared form gives expert 0's
    /// activation to all eight and produces fluent, wrong output.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped_per_act(
        w: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(out, rows * n_used, "grouped_pa out")?;
        need(d, blocks * n_used, "grouped_pa d")?;
        need(ids, n_used, "grouped_pa ids")?;
        if qs.bytes() < blocks * 256 * n_used {
            return Err(Error::malformed("grouped_pa qs", "too small"));
        }
        launch!(ffi::llamd_fwd_matvec_grouped_pa(
            w.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matvec_grouped(
        w: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
    ) -> Result<()> {
        need(out, rows * n_used, "grouped out")?;
        need(d, blocks, "grouped d")?;
        need(ids, n_used, "grouped ids")?;
        launch!(ffi::llamd_fwd_matvec_grouped(
            w.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32
        ));
        Ok(())
    }
}
