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
