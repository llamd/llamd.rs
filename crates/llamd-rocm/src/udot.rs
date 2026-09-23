//! The udot4 register-tiled path: bindings for `kernels/llamd_udot.hip`.
//!
//! gfx1201 has no `sdot4` (that needs target feature `dot1-insts`) but does have
//! `udot4` and `udot8` — probed, not assumed. A signed Q6_K × Q8_K dot product
//! rides on the unsigned instruction through the identity documented in the
//! kernel: with `b = a + 128`, `Σ sc·(w−32)·a = sc·[Σw·b − 128·Σw − 32·Σb +
//! 65536]` over a 16-value scale group. `Σw` is independent of the activation,
//! so at batch N it is computed once and reused N times — the register tile.

#[allow(unsafe_code)]
pub(crate) mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    unsafe extern "C" {
        pub fn llamd_u_set_tile(t: c_int);
        pub fn llamd_u_get_tile() -> c_int;
        pub fn llamd_u_bias_sums(qs: *const i8, qb: *mut u8, sb: *mut i32, nblocks: c_int);
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_norm_quant(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            qb: *mut u8,
            sb: *mut i32,
            d: *mut f32,
            inv: *mut f32,
            n: c_int,
            nseq: c_int,
            eps: f32,
        );
        pub fn llamd_u_quantize(
            x: *const f32,
            qb: *mut u8,
            sb: *mut i32,
            d: *mut f32,
            nblocks: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_matvec(
            base: *const c_void,
            off: *const usize,
            qb: *const u8,
            sb: *const i32,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_x3(
            b0: *const c_void,
            f0: *const usize,
            b1: *const c_void,
            f1: *const usize,
            b2: *const c_void,
            f2: *const usize,
            qb: *const u8,
            sb: *const i32,
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
        pub fn llamd_u_grouped_x2(
            ba: *const c_void,
            fa: *const usize,
            bb: *const c_void,
            fb: *const usize,
            ids: *const i32,
            qb: *const u8,
            sb: *const i32,
            ad: *const f32,
            oa: *mut f32,
            ob: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_grouped_pa(
            base: *const c_void,
            off: *const usize,
            ids: *const i32,
            qb: *const u8,
            sb: *const i32,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            n_used: c_int,
            nseq: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_norm_quant_host(
            x: *const f32,
            w: *const f32,
            n: c_int,
            nseq: c_int,
            eps: f32,
            out: *mut f32,
            qb: *mut u8,
            sb: *mut i32,
            d: *mut f32,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_matvec_bench(
            base: *const c_void,
            off: *const usize,
            qs: *const i8,
            ad: *const f32,
            blocks: c_int,
            rows: c_int,
            nseq: c_int,
            iters: c_int,
            ms_out: *mut f32,
        ) -> c_int;
        pub fn llamd_u_quantize_host(
            x: *const f32,
            nblocks: c_int,
            qb: *mut u8,
            sb: *mut i32,
            d: *mut f32,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn llamd_u_matvec_host(
            base: *const c_void,
            off: *const usize,
            qs: *const i8,
            ad: *const f32,
            out: *mut f32,
            blocks: c_int,
            rows: c_int,
            nseq: c_int,
        ) -> c_int;
    }
}

use crate::{DevBuf, Error, PlanarWeights, Result};

macro_rules! ulaunch {
    ($call:expr) => {{
        #[cfg(not(rocm))]
        {
            return Err(Error::unsupported("rocm", "not built with ROCm support"));
        }
        #[cfg(rocm)]
        // SAFETY: every pointer is a live DevBuf or PlanarWeights buffer, and
        // the element counts were checked by the caller against what the kernel
        // writes.
        #[allow(unsafe_code)]
        unsafe {
            $call
        }
    }};
}

fn need(b: &DevBuf, floats: usize, what: &str) -> Result<()> {
    if b.bytes() < floats * 4 {
        return Err(Error::malformed(
            "udot buffer",
            format!("{what}: holds {} floats, needs {floats}", b.bytes() / 4),
        ));
    }
    Ok(())
}

/// Bias the quantized activation to unsigned and sum each group of 16.
///
/// Must run after every activation quantization and before any udot kernel:
/// `udot4` needs unsigned bytes, and `Σb` is the correction term that makes the
/// unsigned instruction reproduce the signed dot product.
pub fn bias_sums(qs: &DevBuf, qb: &DevBuf, sb: &DevBuf, nblocks_total: usize) -> Result<()> {
    if nblocks_total == 0 {
        return Err(Error::malformed("bias_sums", "no blocks"));
    }
    if qb.bytes() < nblocks_total * 256 || sb.bytes() < nblocks_total * 16 * 4 {
        return Err(Error::malformed("bias_sums", "output buffers too small"));
    }
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        // SAFETY: both outputs were just checked against nblocks_total, which is
        // what the kernel is told.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_u_bias_sums(
                qs.ptr() as *const i8,
                qb.ptr() as *mut u8,
                sb.ptr() as *mut i32,
                nblocks_total as i32,
            )
        };
        Ok(())
    }
}

/// rms_norm(x)*w, emitting f32 AND the biased/summed Q8_K form in one pass.
///
/// Replaces a quantize followed by a separate bias-and-sum kernel: that pass
/// cost a launch at each of the five quantization points in a layer, 240 per
/// batched pass, for no arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn norm_quant(
    x: &DevBuf,
    w: &DevBuf,
    out: &DevBuf,
    qb: &DevBuf,
    sb: &DevBuf,
    d: &DevBuf,
    inv: &DevBuf,
    n: usize,
    nseq: usize,
    eps: f32,
) -> Result<()> {
    need(out, n * nseq, "out")?;
    need(d, (n / 256) * nseq, "d")?;
    need(inv, nseq, "inv")?;
    if qb.bytes() < n * nseq || sb.bytes() < (n / 16) * nseq * 4 {
        return Err(Error::malformed("norm_quant", "qb or sb too small"));
    }
    ulaunch!(ffi::llamd_u_norm_quant(
        x.as_f32(),
        w.as_f32(),
        out.as_f32(),
        qb.ptr() as *mut u8,
        sb.ptr() as *mut i32,
        d.as_f32(),
        inv.as_f32(),
        n as i32,
        nseq as i32,
        eps
    ));
    Ok(())
}

/// Quantize an activation already in its final f32 form, emitting the biased
/// form and group sums directly.
pub fn quantize(x: &DevBuf, qb: &DevBuf, sb: &DevBuf, d: &DevBuf, nblocks: usize) -> Result<()> {
    need(d, nblocks, "d")?;
    if qb.bytes() < nblocks * 256 || sb.bytes() < nblocks * 16 * 4 {
        return Err(Error::malformed("quantize", "qb or sb too small"));
    }
    ulaunch!(ffi::llamd_u_quantize(
        x.as_f32(),
        qb.ptr() as *mut u8,
        sb.ptr() as *mut i32,
        d.as_f32(),
        nblocks as i32
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn matvec(
    w: &PlanarWeights,
    qb: &DevBuf,
    sb: &DevBuf,
    ad: &DevBuf,
    out: &DevBuf,
    rows: usize,
    nseq: usize,
) -> Result<()> {
    need(out, rows * nseq, "out")?;
    need(ad, w.blocks() * nseq, "ad")?;
    ulaunch!(ffi::llamd_u_matvec(
        w.buf().ptr(),
        w.offsets().as_ptr(),
        qb.ptr() as *const u8,
        sb.ptr() as *const i32,
        ad.as_f32(),
        out.as_f32(),
        w.blocks() as i32,
        rows as i32,
        nseq as i32
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn x3(
    w: (&PlanarWeights, &PlanarWeights, &PlanarWeights),
    qb: &DevBuf,
    sb: &DevBuf,
    ad: &DevBuf,
    out: (&DevBuf, &DevBuf, &DevBuf),
    rows: (usize, usize, usize),
    nseq: usize,
) -> Result<()> {
    if w.1.blocks() != w.0.blocks() || w.2.blocks() != w.0.blocks() {
        return Err(Error::malformed("udot x3", "block counts differ"));
    }
    need(out.0, rows.0 * nseq, "o0")?;
    need(out.1, rows.1 * nseq, "o1")?;
    need(out.2, rows.2 * nseq, "o2")?;
    ulaunch!(ffi::llamd_u_x3(
        w.0.buf().ptr(),
        w.0.offsets().as_ptr(),
        w.1.buf().ptr(),
        w.1.offsets().as_ptr(),
        w.2.buf().ptr(),
        w.2.offsets().as_ptr(),
        qb.ptr() as *const u8,
        sb.ptr() as *const i32,
        ad.as_f32(),
        out.0.as_f32(),
        out.1.as_f32(),
        out.2.as_f32(),
        w.0.blocks() as i32,
        rows.0 as i32,
        rows.1 as i32,
        rows.2 as i32,
        nseq as i32
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_x2(
    wa: &PlanarWeights,
    wb: &PlanarWeights,
    ids: &DevBuf,
    qb: &DevBuf,
    sb: &DevBuf,
    ad: &DevBuf,
    oa: &DevBuf,
    ob: &DevBuf,
    rows: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    if wb.blocks() != wa.blocks() {
        return Err(Error::malformed("udot x2", "block counts differ"));
    }
    need(oa, rows * n_used * nseq, "oa")?;
    need(ob, rows * n_used * nseq, "ob")?;
    ulaunch!(ffi::llamd_u_grouped_x2(
        wa.buf().ptr(),
        wa.offsets().as_ptr(),
        wb.buf().ptr(),
        wb.offsets().as_ptr(),
        ids.ptr() as *const i32,
        qb.ptr() as *const u8,
        sb.ptr() as *const i32,
        ad.as_f32(),
        oa.as_f32(),
        ob.as_f32(),
        wa.blocks() as i32,
        rows as i32,
        n_used as i32,
        nseq as i32
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_per_act(
    w: &PlanarWeights,
    ids: &DevBuf,
    qb: &DevBuf,
    sb: &DevBuf,
    ad: &DevBuf,
    out: &DevBuf,
    rows: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    need(out, rows * n_used * nseq, "out")?;
    ulaunch!(ffi::llamd_u_grouped_pa(
        w.buf().ptr(),
        w.offsets().as_ptr(),
        ids.ptr() as *const i32,
        qb.ptr() as *const u8,
        sb.ptr() as *const i32,
        ad.as_f32(),
        out.as_f32(),
        w.blocks() as i32,
        rows as i32,
        n_used as i32,
        nseq as i32
    ));
    Ok(())
}

/// Run the fused quantize on the host's behalf and return `(qb, sb, d)`.
///
/// For validation only: the point is to compare the fused kernel against
/// `llamd_quant::quantize_q8_k` directly, rather than inferring a bug from a
/// wrong token forty-eight layers downstream.
pub fn quantize_host(x: &[f32]) -> Result<(Vec<u8>, Vec<i32>, Vec<f32>)> {
    if x.is_empty() || x.len() % 256 != 0 {
        return Err(Error::malformed(
            "quantize_host",
            "not a whole number of blocks",
        ));
    }
    let nb = x.len() / 256;
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        let mut qb = vec![0u8; nb * 256];
        let mut sb = vec![0i32; nb * 16];
        let mut d = vec![0.0f32; nb];
        // SAFETY: all three outputs are sized from the same `nb` the entry
        // point is told, and `x` is non-empty and block-aligned.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_u_quantize_host(
                x.as_ptr(),
                nb as i32,
                qb.as_mut_ptr(),
                sb.as_mut_ptr(),
                d.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(Error::other("fused quantize failed"));
        }
        Ok((qb, sb, d))
    }
}

/// Run the fused rms_norm+quantize and return `(out, qb, sb, d)`. Validation only.
pub fn norm_quant_host(
    x: &[f32],
    w: &[f32],
    nseq: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<u8>, Vec<i32>, Vec<f32>)> {
    let n = w.len();
    if n % 256 != 0 || x.len() != n * nseq {
        return Err(Error::malformed("norm_quant_host", "shape mismatch"));
    }
    let nb = n / 256;
    #[cfg(not(rocm))]
    {
        Err(Error::unsupported("rocm", "not built with ROCm support"))
    }
    #[cfg(rocm)]
    {
        let mut out = vec![0.0f32; n * nseq];
        let mut qb = vec![0u8; n * nseq];
        let mut sb = vec![0i32; nb * 16 * nseq];
        let mut d = vec![0.0f32; nb * nseq];
        // SAFETY: every buffer is sized from n, nb and nseq, which are the
        // values the entry point is told.
        #[allow(unsafe_code)]
        let rc = unsafe {
            ffi::llamd_u_norm_quant_host(
                x.as_ptr(),
                w.as_ptr(),
                n as i32,
                nseq as i32,
                eps,
                out.as_mut_ptr(),
                qb.as_mut_ptr(),
                sb.as_mut_ptr(),
                d.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return Err(Error::other("fused norm_quant failed"));
        }
        Ok((out, qb, sb, d))
    }
}

/// Rows per wave in the udot matvec: 1 (no tiling), 2 or 4.
///
/// Tiling loads each activation group once and consumes it `tile` times. The
/// activation is cache-resident, so this trades registers for load
/// instructions — and which way that trade goes is a question for the engine,
/// not for a microbenchmark. Every value is measured by the sweep in
/// llamd-testing/tests/batch_decode.rs.
pub fn set_tile(t: usize) {
    #[cfg(rocm)]
    {
        // SAFETY: sets an int; out-of-range values are clamped to 1 in the C.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_u_set_tile(t as i32)
        }
    }
    #[cfg(not(rocm))]
    {
        let _ = t;
    }
}

pub fn tile() -> usize {
    #[cfg(not(rocm))]
    {
        1
    }
    #[cfg(rocm)]
    {
        // SAFETY: reads an int.
        #[allow(unsafe_code)]
        unsafe {
            ffi::llamd_u_get_tile() as usize
        }
    }
}
