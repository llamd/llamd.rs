//! Batched decode: several independent sequences per pass.
//!
//! WHAT THIS BUYS, and what it does not. A token at batch 1 reads 2.485 GB of
//! weights, and three attempts to raise the ~260 GB/s the engine sustains were
//! rejected by measurement (benchmarks.md §8-§10). Batching attacks the bytes
//! instead of the bandwidth: the attention projections (743 MB/token) and the
//! output projection (255 MB) are the SAME matrices for every sequence, so
//! their cost divides by the batch size.
//!
//! The routed experts do not divide. Each sequence picks its own 8 of 128, and
//! at batch 2 only ~3% of those picks coincide. Experts are 1486 MB of the 2485,
//! so per-token traffic goes to 1986 MB at batch 2 (1.25x) and 1736 at batch 4
//! (1.43x) — not 1/2 and 1/4. That is mixture-of-experts, not an implementation
//! limit.
//!
//! AGGREGATE THROUGHPUT, NOT LATENCY. Token N+1 of a sequence depends on token
//! N, so the sequences batched here must be independent — separate streams, or
//! speculative candidates. Each sequence still sees roughly batch-1 latency.

use crate::{DevBuf, Error, Result};

/// The largest batch the kernels' register arrays are sized for.
pub fn max_seq() -> usize {
    #[cfg(not(rocm))]
    {
        0
    }
    #[cfg(rocm)]
    {
        // SAFETY: returns a compile-time constant.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_batch_max_seq() as usize
        }
    }
}

#[allow(unsafe_code)]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    unsafe extern "C" {
        pub fn llamd_batch_max_seq() -> c_int;
        pub fn llamd_b_embed(
            table: *const c_void,
            meta: *const i32,
            out: *mut f32,
            blocks: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_rms_norm_quant(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            qs: *mut i8,
            d: *mut f32,
            inv: *mut f32,
            n: c_int,
            nseq: c_int,
            eps: f32,
        );
        pub fn llamd_b_rms_norm_mul(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            n: c_int,
            nseq: c_int,
            eps: f32,
        );
        pub fn llamd_b_quantize(x: *const f32, qs: *mut i8, d: *mut f32, nblocks: c_int);
        pub fn llamd_b_matvec(
            w: *const c_void,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_matvec_x3(
            w0: *const c_void,
            w1: *const c_void,
            w2: *const c_void,
            aq: *const i8,
            ad: *const f32,
            o0: *mut f32,
            o1: *mut f32,
            o2: *mut f32,
            blocks: c_int,
            r0: c_int,
            r1: c_int,
            r2: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_grouped_x2(
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
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_grouped_pa(
            w: *const c_void,
            ids: *const i32,
            aq: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_norm_rope(
            x: *mut f32,
            w: *const f32,
            n_head: c_int,
            hd: c_int,
            meta: *const i32,
            nseq: c_int,
            freq_base: f32,
            eps: f32,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_k_norm_rope_store(
            k: *mut f32,
            w: *const f32,
            v: *const f32,
            kc: *mut f32,
            vc: *mut f32,
            n_head_kv: c_int,
            hd: c_int,
            kv_dim: c_int,
            max_kv: c_int,
            meta: *const i32,
            nseq: c_int,
            freq_base: f32,
            eps: f32,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_attention(
            q: *const f32,
            kc: *const f32,
            vc: *const f32,
            out: *mut f32,
            scores: *mut f32,
            n_head: c_int,
            n_head_kv: c_int,
            hd: c_int,
            max_kv: c_int,
            meta: *const i32,
            nseq: c_int,
        );
        pub fn llamd_b_add(a: *const f32, b: *const f32, o: *mut f32, n: c_int);
        pub fn llamd_b_silu_mul(g: *const f32, u: *const f32, o: *mut f32, n: c_int);
        pub fn llamd_b_matvec_f32(
            w: *const f32,
            x: *const f32,
            out: *mut f32,
            k: c_int,
            rows: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_b_router(
            logits: *const f32,
            n_expert: c_int,
            n_used: c_int,
            ids: *mut i32,
            weights: *mut f32,
            probs: *mut f32,
            nseq: c_int,
        );
        pub fn llamd_b_moe_combine_add(
            down: *const f32,
            weights: *const f32,
            residual: *const f32,
            out: *mut f32,
            n_used: c_int,
            n_embd: c_int,
            nseq: c_int,
        );
        pub fn llamd_b_argmax(
            x: *const f32,
            n: c_int,
            sv: *mut f32,
            si: *mut i32,
            out: *mut i32,
            nchunk: c_int,
            nseq: c_int,
        );
    }
}

/// The batched operations. Each queues one launch covering every sequence.
pub mod bop {
    use super::{DevBuf, Error, Result, ffi};

    fn need(buf: &DevBuf, floats: usize, what: &str) -> Result<()> {
        if buf.bytes() < floats * 4 {
            return Err(Error::malformed(
                "batch buffer",
                format!("{what}: holds {} floats, needs {floats}", buf.bytes() / 4),
            ));
        }
        Ok(())
    }

    fn check(nseq: usize) -> Result<()> {
        let m = super::max_seq();
        if nseq == 0 || nseq > m {
            return Err(Error::malformed(
                "batch size",
                format!("{nseq} outside 1..={m}"),
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
            // SAFETY: every pointer is a live DevBuf checked above for the
            // element count the kernel writes, and nseq is bounds-checked
            // against the kernels' MAX_SEQ.
            #[allow(unsafe_code)]
            unsafe {
                $call
            }
        }};
    }

    pub fn embed(
        table: &DevBuf,
        meta: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(out, blocks * 256 * nseq, "embed out")?;
        need(meta, nseq * 2, "meta")?;
        launch!(ffi::llamd_b_embed(
            table.ptr(),
            meta.ptr() as *const i32,
            out.as_f32(),
            blocks as i32,
            nseq as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_quant(
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        inv: &DevBuf,
        n: usize,
        nseq: usize,
        eps: f32,
    ) -> Result<()> {
        check(nseq)?;
        need(x, n * nseq, "x")?;
        need(w, n, "w")?;
        need(out, n * nseq, "out")?;
        need(d, (n / 256) * nseq, "d")?;
        need(inv, nseq, "inv")?;
        if qs.bytes() < n * nseq {
            return Err(Error::malformed("qs", "too small"));
        }
        launch!(ffi::llamd_b_rms_norm_quant(
            x.as_f32(),
            w.as_f32(),
            out.as_f32(),
            qs.ptr() as *mut i8,
            d.as_f32(),
            inv.as_f32(),
            n as i32,
            nseq as i32,
            eps
        ));
        Ok(())
    }

    pub fn rms_norm_mul(
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        n: usize,
        nseq: usize,
        eps: f32,
    ) -> Result<()> {
        check(nseq)?;
        need(out, n * nseq, "out")?;
        launch!(ffi::llamd_b_rms_norm_mul(
            x.as_f32(),
            w.as_f32(),
            out.as_f32(),
            n as i32,
            nseq as i32,
            eps
        ));
        Ok(())
    }

    pub fn quantize(x: &DevBuf, qs: &DevBuf, d: &DevBuf, nblocks_total: usize) -> Result<()> {
        need(d, nblocks_total, "d")?;
        if qs.bytes() < nblocks_total * 256 {
            return Err(Error::malformed("qs", "too small"));
        }
        launch!(ffi::llamd_b_quantize(
            x.as_f32(),
            qs.ptr() as *mut i8,
            d.as_f32(),
            nblocks_total as i32
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
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(out, rows * nseq, "out")?;
        need(d, blocks * nseq, "d")?;
        launch!(ffi::llamd_b_matvec(
            w.ptr(),
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            nseq as i32
        ));
        Ok(())
    }

    /// Q, K and V: one launch, and each weight byte serves every sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_x3(
        w: (&DevBuf, &DevBuf, &DevBuf),
        qs: &DevBuf,
        d: &DevBuf,
        out: (&DevBuf, &DevBuf, &DevBuf),
        blocks: usize,
        rows: (usize, usize, usize),
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(out.0, rows.0 * nseq, "o0")?;
        need(out.1, rows.1 * nseq, "o1")?;
        need(out.2, rows.2 * nseq, "o2")?;
        need(d, blocks * nseq, "d")?;
        launch!(ffi::llamd_b_matvec_x3(
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
            nseq as i32
        ));
        Ok(())
    }

    /// Expert gate and up. `ids` is `[nseq][n_used]`; no weight is shared
    /// between sequences, because they route independently.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_x2(
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
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(oa, rows * n_used * nseq, "oa")?;
        need(ob, rows * n_used * nseq, "ob")?;
        need(ids, n_used * nseq, "ids")?;
        launch!(ffi::llamd_b_grouped_x2(
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
            nseq as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn grouped_pa(
        w: &DevBuf,
        ids: &DevBuf,
        qs: &DevBuf,
        d: &DevBuf,
        out: &DevBuf,
        blocks: usize,
        rows: usize,
        n_used: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(out, rows * n_used * nseq, "out")?;
        need(d, blocks * n_used * nseq, "d")?;
        launch!(ffi::llamd_b_grouped_pa(
            w.ptr(),
            ids.ptr() as *const i32,
            qs.ptr() as *const i8,
            d.as_f32(),
            out.as_f32(),
            blocks as i32,
            rows as i32,
            n_used as i32,
            nseq as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn norm_rope(
        x: &DevBuf,
        w: Option<&DevBuf>,
        meta: &DevBuf,
        n_head: usize,
        hd: usize,
        nseq: usize,
        freq_base: f32,
        eps: f32,
    ) -> Result<()> {
        check(nseq)?;
        need(x, n_head * hd * nseq, "x")?;
        let wp = w
            .map(|b| b.as_f32() as *const f32)
            .unwrap_or(std::ptr::null());
        launch!(ffi::llamd_b_norm_rope(
            x.as_f32(),
            wp,
            n_head as i32,
            hd as i32,
            meta.ptr() as *const i32,
            nseq as i32,
            freq_base,
            eps
        ));
        Ok(())
    }

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
        max_kv: usize,
        nseq: usize,
        freq_base: f32,
        eps: f32,
    ) -> Result<()> {
        check(nseq)?;
        let kv_dim = n_head_kv * hd;
        need(k, kv_dim * nseq, "k")?;
        need(v, kv_dim * nseq, "v")?;
        need(kc, max_kv * kv_dim * nseq, "kc")?;
        need(vc, max_kv * kv_dim * nseq, "vc")?;
        let wp = w
            .map(|b| b.as_f32() as *const f32)
            .unwrap_or(std::ptr::null());
        launch!(ffi::llamd_b_k_norm_rope_store(
            k.as_f32(),
            wp,
            v.as_f32(),
            kc.as_f32(),
            vc.as_f32(),
            n_head_kv as i32,
            hd as i32,
            kv_dim as i32,
            max_kv as i32,
            meta.ptr() as *const i32,
            nseq as i32,
            freq_base,
            eps
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
        meta: &DevBuf,
        n_head: usize,
        n_head_kv: usize,
        hd: usize,
        max_kv: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        if n_head % n_head_kv != 0 {
            return Err(Error::malformed("attention", "kv heads do not divide"));
        }
        need(q, n_head * hd * nseq, "q")?;
        need(out, n_head * hd * nseq, "out")?;
        need(scores, n_head * max_kv * nseq, "scores")?;
        launch!(ffi::llamd_b_attention(
            q.as_f32(),
            kc.as_f32(),
            vc.as_f32(),
            out.as_f32(),
            scores.as_f32(),
            n_head as i32,
            n_head_kv as i32,
            hd as i32,
            max_kv as i32,
            meta.ptr() as *const i32,
            nseq as i32
        ));
        Ok(())
    }

    pub fn add(a: &DevBuf, b: &DevBuf, o: &DevBuf, n: usize) -> Result<()> {
        need(o, n, "out")?;
        launch!(ffi::llamd_b_add(
            a.as_f32(),
            b.as_f32(),
            o.as_f32(),
            n as i32
        ));
        Ok(())
    }

    pub fn silu_mul(g: &DevBuf, u: &DevBuf, o: &DevBuf, n: usize) -> Result<()> {
        need(o, n, "out")?;
        launch!(ffi::llamd_b_silu_mul(
            g.as_f32(),
            u.as_f32(),
            o.as_f32(),
            n as i32
        ));
        Ok(())
    }

    pub fn matvec_f32(
        w: &DevBuf,
        x: &DevBuf,
        out: &DevBuf,
        k: usize,
        rows: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(out, rows * nseq, "out")?;
        launch!(ffi::llamd_b_matvec_f32(
            w.as_f32(),
            x.as_f32(),
            out.as_f32(),
            k as i32,
            rows as i32,
            nseq as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router(
        logits: &DevBuf,
        n_expert: usize,
        n_used: usize,
        ids: &DevBuf,
        weights: &DevBuf,
        probs: &DevBuf,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        if n_used == 0 || n_used > n_expert || n_expert > 1024 {
            return Err(Error::malformed(
                "moe routing",
                format!("{n_used} of {n_expert}"),
            ));
        }
        need(logits, n_expert * nseq, "logits")?;
        need(probs, n_expert * nseq, "probs")?;
        need(ids, n_used * nseq, "ids")?;
        need(weights, n_used * nseq, "weights")?;
        launch!(ffi::llamd_b_router(
            logits.as_f32(),
            n_expert as i32,
            n_used as i32,
            ids.ptr() as *mut i32,
            weights.as_f32(),
            probs.as_f32(),
            nseq as i32
        ));
        Ok(())
    }

    pub fn moe_combine_add(
        down: &DevBuf,
        weights: &DevBuf,
        residual: &DevBuf,
        out: &DevBuf,
        n_used: usize,
        n_embd: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(down, n_used * n_embd * nseq, "down")?;
        need(out, n_embd * nseq, "out")?;
        launch!(ffi::llamd_b_moe_combine_add(
            down.as_f32(),
            weights.as_f32(),
            residual.as_f32(),
            out.as_f32(),
            n_used as i32,
            n_embd as i32,
            nseq as i32
        ));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn argmax(
        x: &DevBuf,
        n: usize,
        sv: &DevBuf,
        si: &DevBuf,
        out: &DevBuf,
        nchunk: usize,
        nseq: usize,
    ) -> Result<()> {
        check(nseq)?;
        need(x, n * nseq, "logits")?;
        need(sv, nchunk * nseq, "scratch")?;
        launch!(ffi::llamd_b_argmax(
            x.as_f32(),
            n as i32,
            sv.as_f32(),
            si.ptr() as *mut i32,
            out.ptr() as *mut i32,
            nchunk as i32,
            nseq as i32
        ));
        Ok(())
    }
}
