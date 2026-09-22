//! One transformer layer, forward, on the CPU reference backend.
//!
//! PROVENANCE. The composition follows `src/models/qwen3moe.cpp` and
//! `llm_graph_context::build_attn` / `build_moe_ffn` in llama.cpp
//! (MIT, © 2023-2026 The ggml authors).
//!
//! THE ORDER, for one token:
//!
//! ```text
//!   x                              [n_embd]
//!   norm      = rms_norm(x)
//!   attn_norm = norm * w_attn_norm
//!   Q = Wq·attn_norm  [n_head*hd]   K,V = Wk,Wv·attn_norm  [n_kv*hd]
//!   Q,K per-head RMS-normed by w_q_norm / w_k_norm     ← Qwen3 QK-norm
//!   Q,K rotated by RoPE at this position
//!   append K,V to the cache
//!   attn      = GQA(Q, cache_K, cache_V)
//!   kqv_out   = Wo·attn
//!   ffn_inp   = x + kqv_out                            ← residual
//!   ffn_norm  = rms_norm(ffn_inp) * w_ffn_norm
//!   logits    = W_router·ffn_norm   [n_expert]
//!   routing   = softmax → top-k → renormalise
//! ```
//!
//! QK-NORM IS APPLIED BEFORE RoPE, and per head rather than across the whole
//! projection. Both details matter: normalising after rotation, or over all
//! 4096 values at once, produces finite plausible numbers and a different
//! model. Qwen2 has no QK-norm at all, so its presence is also what
//! distinguishes the architectures.

use llamd_core::Result;
use llamd_models::{Model, Routing};
use llamd_tensor::{Backend, RopeConfig, Tensor};

use crate::{AttnParams, KvCache, attend};

/// Intermediates from one layer, named to match llama.cpp's debug output so a
/// comparison can be made tensor by tensor rather than only at the end.
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub norm: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub q_roped: Vec<f32>,
    pub k: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub k_roped: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub moe_logits: Vec<f32>,
    pub moe_probs: Vec<f32>,
    pub routing: Routing,
}

/// RMS-normalise each head independently, then scale by a per-head-dim weight.
///
/// `x` is `[n_head * head_dim]`; `w` is `[head_dim]` and is shared by heads.
fn per_head_rms_norm(
    be: &dyn Backend,
    x: &[f32],
    w: &[f32],
    n_head: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let t = Tensor::from_slice(&[head_dim, n_head], x)?;
    let n = be.rms_norm(&t, eps)?;
    let mut out = n.data().to_vec();
    for h in 0..n_head {
        for d in 0..head_dim {
            out[h * head_dim + d] *= w[d];
        }
    }
    Ok(out)
}

/// Run one layer for a single token at `pos`, recording every intermediate.
///
/// Single-token only, deliberately: batching is the next piece of work and
/// adding it here would mean validating two things at once.
pub fn forward_layer(
    be: &dyn Backend,
    model: &Model,
    il: usize,
    x: &[f32],
    pos: u32,
    kv: &mut KvCache,
) -> Result<LayerTrace> {
    let h = &model.hparams;
    let l = &model.layers[il];
    let (n_embd, n_head, n_head_kv, hd) = (h.n_embd, h.n_head, h.n_head_kv, h.n_head_dim);

    // SINGLE-TOKEN SHAPE CONVENTION. `matmul` returns rank-2 `[m, n]`, so with
    // one token every projection comes back as `[dim, 1]` while the residual
    // path carries `[dim]`. Rather than making `add` silently tolerate a rank
    // mismatch — which would hide real shape errors — each projection is
    // squeezed back to rank 1 explicitly, at the point it happens.
    let xt = Tensor::from_slice(&[n_embd], x)?;
    let vec1 = |t: Tensor| -> Result<Tensor> {
        let n = t.elems();
        t.reshape(&[n])
    };

    // ── attention norm ───────────────────────────────────────────────────
    let norm = be.rms_norm(&xt, h.rms_eps)?;
    let w_attn_norm = model.tensor(&l.attn_norm)?;
    let attn_norm = be.mul(&norm, &w_attn_norm)?;

    // ── projections ──────────────────────────────────────────────────────
    let q = vec1(be.matmul(&model.tensor(&l.attn_q)?, &attn_norm)?)?;
    let k = vec1(be.matmul(&model.tensor(&l.attn_k)?, &attn_norm)?)?;
    let v = vec1(be.matmul(&model.tensor(&l.attn_v)?, &attn_norm)?)?;

    // ── QK-norm, per head, BEFORE RoPE ───────────────────────────────────
    let q_normed = match &l.attn_q_norm {
        Some(n) => per_head_rms_norm(be, q.data(), model.tensor(n)?.data(), n_head, hd, h.rms_eps)?,
        None => q.data().to_vec(),
    };
    let k_normed = match &l.attn_k_norm {
        Some(n) => per_head_rms_norm(
            be,
            k.data(),
            model.tensor(n)?.data(),
            n_head_kv,
            hd,
            h.rms_eps,
        )?,
        None => k.data().to_vec(),
    };

    // ── RoPE ─────────────────────────────────────────────────────────────
    let cfg = RopeConfig::new(hd, h.rope_freq_base);
    let q_roped = be
        .rope(&Tensor::from_slice(&[n_head * hd], &q_normed)?, &[pos], cfg)?
        .data()
        .to_vec();
    let k_roped = be
        .rope(
            &Tensor::from_slice(&[n_head_kv * hd], &k_normed)?,
            &[pos],
            cfg,
        )?
        .data()
        .to_vec();

    // ── cache and attend ─────────────────────────────────────────────────
    kv.append(il, pos as usize, &k_roped, v.data())?;
    let n_kv = (pos as usize) + 1;
    let ap = AttnParams {
        n_head,
        n_head_kv,
        head_dim: hd,
    };
    let attn_out = attend(
        &q_roped,
        &kv.keys(il)?[..n_kv * h.kv_dim()],
        &kv.values(il)?[..n_kv * h.kv_dim()],
        n_kv,
        ap,
    )?;

    // ── output projection and residual ───────────────────────────────────
    let kqv_out = vec1(be.matmul(
        &model.tensor(&l.attn_out)?,
        &Tensor::from_slice(&[n_head * hd], &attn_out)?,
    )?)?;
    let ffn_inp = be.add(&xt, &kqv_out)?;

    // ── FFN norm and routing ─────────────────────────────────────────────
    let fnorm = be.rms_norm(&ffn_inp, h.rms_eps)?;
    let ffn_norm = be.mul(&fnorm, &model.tensor(&l.ffn_norm)?)?;

    let router = l
        .ffn_gate_inp
        .as_ref()
        .ok_or_else(|| llamd_core::Error::unsupported("layer", "dense FFN is not implemented"))?;
    let moe_logits = vec1(be.matmul(&model.tensor(router)?, &ffn_norm)?)?;
    let moe_probs = be.softmax(&moe_logits)?;
    let routing = llamd_models::route(moe_logits.data(), h.n_expert_used)?;

    Ok(LayerTrace {
        norm: norm.data().to_vec(),
        attn_norm: attn_norm.data().to_vec(),
        q: q.data().to_vec(),
        q_normed,
        q_roped,
        k: k.data().to_vec(),
        k_normed,
        k_roped,
        v: v.data().to_vec(),
        attn_out,
        kqv_out: kqv_out.data().to_vec(),
        ffn_inp: ffn_inp.data().to_vec(),
        ffn_norm: ffn_norm.data().to_vec(),
        moe_logits: moe_logits.data().to_vec(),
        moe_probs: moe_probs.data().to_vec(),
        routing,
    })
}

/// Look up a token's embedding row.
pub fn embed(be: &dyn Backend, model: &Model, token: u32) -> Result<Vec<f32>> {
    let table = model.tensor(&model.tok_embd)?;
    Ok(be.get_rows(&table, &[token])?.data().to_vec())
}
