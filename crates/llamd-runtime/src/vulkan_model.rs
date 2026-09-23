//! `VulkanModel` — the whole model resident in VRAM, a token computed on Vulkan.
//!
//! This is the Vulkan counterpart of [`crate::device::DeviceModel`], and it is
//! deliberately the same shape: weights uploaded once, activations never leaving
//! the device, every operation recorded without waiting, ONE submission per
//! token, and only the argmax read back.
//!
//! WHY MIRROR THE ROCM PATH INSTEAD OF ABSTRACTING OVER IT. A single trait over
//! both APIs would force each backend into the shape of the other, and the point
//! of a Vulkan-first design is that Vulkan's descriptors, barriers and dispatch
//! shapes can be tuned on their own terms. What the two paths share is the
//! planar data layout and the arithmetic — which is what makes a comparison
//! between them measure the backend rather than the algorithm.
//!
//! CORRECTNESS IS DEFINED BY THE CPU REFERENCE, as everywhere else in this
//! crate: the path that agrees with llama.cpp token for token.

use llamd_core::{Error, Result};
use llamd_models::Model;
use llamd_vulkan::{PlanarWeights, VkBuf, ops};

struct Layer {
    attn_norm: VkBuf,
    attn_q: PlanarWeights,
    attn_k: PlanarWeights,
    attn_v: PlanarWeights,
    attn_q_norm: Option<VkBuf>,
    attn_k_norm: Option<VkBuf>,
    attn_out: PlanarWeights,
    ffn_norm: VkBuf,
    /// The MoE gate: F32 in this model, not Q6_K.
    ffn_gate_inp: VkBuf,
    ffn_gate_exps: PlanarWeights,
    ffn_up_exps: PlanarWeights,
    ffn_down_exps: PlanarWeights,
}

struct Scratch {
    x: VkBuf,
    xb: VkBuf,
    xb2: VkBuf,
    q: VkBuf,
    k: VkBuf,
    v: VkBuf,
    attn: VkBuf,
    qb: VkBuf,
    sb: VkBuf,
    ad: VkBuf,
    inv: VkBuf,
    router_logits: VkBuf,
    router_probs: VkBuf,
    expert_ids: VkBuf,
    expert_w: VkBuf,
    e_gate: VkBuf,
    e_up: VkBuf,
    e_act: VkBuf,
    e_down: VkBuf,
    scores: VkBuf,
    logits: VkBuf,
    amax_v: VkBuf,
    amax_i: VkBuf,
    out_token: VkBuf,
    /// `[token, pos]` per sequence — the only per-token host write.
    meta: VkBuf,
}

pub struct VulkanModel {
    hp: llamd_models::Hparams,
    tok_embd: PlanarWeights,
    layers: Vec<Layer>,
    output_norm: VkBuf,
    output: PlanarWeights,
    kc: Vec<VkBuf>,
    vc: Vec<VkBuf>,
    max_kv: usize,
    max_seq: usize,
    s: Scratch,
    weight_bytes: usize,
}

const ARGMAX_CHUNKS: usize = 512;

fn raw<'a>(model: &'a Model, name: &str) -> Result<&'a [u8]> {
    let info = model.info(name)?;
    model.gguf().tensor_data(info)
}

fn up_planar(model: &Model, name: &str) -> Result<PlanarWeights> {
    let info = model.info(name)?;
    let k = info.dims[0] as usize;
    let rows = info.elems() as usize / k;
    let ty = llamd_quant::QuantType::from_ggml(info.ggml_type)?;
    if ty != llamd_quant::QuantType::Q6K {
        return Err(Error::unsupported(
            "vulkan upload",
            format!("{name} is {ty:?}, not Q6_K"),
        ));
    }
    PlanarWeights::upload(raw(model, name)?, rows, k / 256)
        .map_err(|e| Error::other(format!("{name}: {e}")))
}

fn up_f32(model: &Model, name: &str) -> Result<VkBuf> {
    let t = model.tensor(name)?;
    VkBuf::from_f32(t.data()).map_err(|e| Error::other(format!("{name}: {e}")))
}

impl VulkanModel {
    pub fn load(model: &Model, max_kv: usize, max_seq: usize) -> Result<Self> {
        llamd_vulkan::init()?;
        let hp = model.hparams.clone();
        if !hp.arch.is_moe() {
            return Err(Error::unsupported(
                "vulkan model",
                "only MoE models are wired up",
            ));
        }
        if max_kv == 0 || max_seq == 0 || max_seq > 8 {
            return Err(Error::malformed(
                "vulkan model",
                format!("max_kv {max_kv}, max_seq {max_seq}"),
            ));
        }

        let (n_embd, n_head, hd) = (hp.n_embd, hp.n_head, hp.n_head_dim);
        let kv_dim = hp.kv_dim();
        let (n_expert, n_used, n_ffe) = (hp.n_expert, hp.n_expert_used, hp.n_ff_expert);

        let tok_embd = up_planar(model, &model.tok_embd)?;
        let output_norm = up_f32(model, &model.output_norm)?;
        let output = match &model.output {
            Some(o) => up_planar(model, o)?,
            None => {
                return Err(Error::unsupported(
                    "vulkan model",
                    "tied output embeddings are not wired up",
                ));
            }
        };

        let mut layers = Vec::with_capacity(hp.n_layer);
        let mut weight_bytes = raw(model, &model.tok_embd).map(|b| b.len()).unwrap_or(0)
            + model
                .output
                .as_ref()
                .and_then(|o| raw(model, o).ok())
                .map(|b| b.len())
                .unwrap_or(0);
        for il in 0..hp.n_layer {
            let l = &model.layers[il];
            // Every Q6_K tensor, not just the attention ones: a reported
            // figure that silently omits 60% of the model is worse than none.
            for n in [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out] {
                weight_bytes += raw(model, n).map(|b| b.len()).unwrap_or(0);
            }
            for n in [&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps] {
                if let Some(n) = n.as_ref() {
                    weight_bytes += raw(model, n).map(|b| b.len()).unwrap_or(0);
                }
            }
            layers.push(Layer {
                attn_norm: up_f32(model, &l.attn_norm)?,
                attn_q: up_planar(model, &l.attn_q)?,
                attn_k: up_planar(model, &l.attn_k)?,
                attn_v: up_planar(model, &l.attn_v)?,
                attn_q_norm: l
                    .attn_q_norm
                    .as_ref()
                    .map(|n| up_f32(model, n))
                    .transpose()?,
                attn_k_norm: l
                    .attn_k_norm
                    .as_ref()
                    .map(|n| up_f32(model, n))
                    .transpose()?,
                attn_out: up_planar(model, &l.attn_out)?,
                ffn_norm: up_f32(model, &l.ffn_norm)?,
                ffn_gate_inp: up_f32(
                    model,
                    l.ffn_gate_inp
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no MoE router"))?,
                )?,
                ffn_gate_exps: up_planar(
                    model,
                    l.ffn_gate_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no gate experts"))?,
                )?,
                ffn_up_exps: up_planar(
                    model,
                    l.ffn_up_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no up experts"))?,
                )?,
                ffn_down_exps: up_planar(
                    model,
                    l.ffn_down_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no down experts"))?,
                )?,
            });
        }

        // Activations carry a sequence dimension; the weights do not, which is
        // the whole point of batching.
        let f = |n: usize| VkBuf::new(n * max_seq * 4);
        // The widest quantized activation is the down projection's, indexed by
        // (sequence, expert, block).
        let blocks_max = (n_embd.max(n_head * hd).max(n_used * n_ffe)).div_ceil(256);
        let blocks_wide = blocks_max.max((n_ffe / 256) * n_used);

        let s = Scratch {
            x: f(n_embd)?,
            xb: f(n_embd)?,
            xb2: f(n_embd)?,
            q: f(n_head * hd)?,
            k: f(kv_dim)?,
            v: f(kv_dim)?,
            attn: f(n_head * hd)?,
            qb: VkBuf::new(blocks_wide * 256 * max_seq)?,
            sb: VkBuf::new(blocks_wide * 16 * 4 * max_seq)?,
            ad: f(blocks_wide)?,
            inv: f(1)?,
            router_logits: f(n_expert)?,
            router_probs: f(n_expert)?,
            expert_ids: VkBuf::new(n_used * 4 * max_seq)?,
            expert_w: f(n_used)?,
            e_gate: f(n_used * n_ffe)?,
            e_up: f(n_used * n_ffe)?,
            e_act: f(n_used * n_ffe)?,
            e_down: f(n_used * n_embd)?,
            scores: f(n_head * max_kv)?,
            logits: f(hp.n_vocab)?,
            amax_v: f(ARGMAX_CHUNKS)?,
            amax_i: VkBuf::new(ARGMAX_CHUNKS * 4 * max_seq)?,
            out_token: VkBuf::new(4 * max_seq)?,
            meta: VkBuf::new(8 * max_seq)?,
        };

        let mut kc = Vec::with_capacity(hp.n_layer);
        let mut vc = Vec::with_capacity(hp.n_layer);
        for _ in 0..hp.n_layer {
            kc.push(VkBuf::new(max_kv * kv_dim * 4 * max_seq)?);
            vc.push(VkBuf::new(max_kv * kv_dim * 4 * max_seq)?);
        }

        Ok(VulkanModel {
            hp,
            tok_embd,
            layers,
            output_norm,
            output,
            kc,
            vc,
            max_kv,
            max_seq,
            s,
            weight_bytes,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// One pass over `nseq` independent sequences, returning a token each.
    pub fn forward_batch(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<u32>> {
        let nseq = tokens.len();
        if nseq == 0 || nseq != positions.len() || nseq > self.max_seq {
            return Err(Error::malformed(
                "vulkan forward",
                format!(
                    "{nseq} tokens, {} positions, max {}",
                    positions.len(),
                    self.max_seq
                ),
            ));
        }
        for (i, &p) in positions.iter().enumerate() {
            if p >= self.max_kv {
                return Err(Error::malformed(
                    "vulkan forward",
                    format!(
                        "sequence {i} position {p} beyond the {} allocated",
                        self.max_kv
                    ),
                ));
            }
        }

        // The only per-token host write, and it must happen BEFORE recording:
        // there is one command buffer, and an upload mid-recording is refused.
        let mut meta = Vec::with_capacity(nseq * 2);
        for (t, p) in tokens.iter().zip(positions) {
            meta.push(*t as i32);
            meta.push(*p as i32);
        }
        self.s.meta.upload_i32(&meta)?;

        let hp = &self.hp;
        let (n_embd, n_head, n_head_kv, hd) = (hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_head_dim);
        let kv_dim = hp.kv_dim();
        let (n_expert, n_used, n_ffe) = (hp.n_expert, hp.n_expert_used, hp.n_ff_expert);
        let eps = hp.rms_eps;
        let fb = hp.rope_freq_base;
        let max_kv = self.max_kv;
        let s = &self.s;

        ops::begin()?;
        // From here to end_and_wait, nothing waits and nothing crosses the bus.
        let queued = (|| -> Result<()> {
            ops::embed(&self.tok_embd, &s.meta, &s.x, n_embd / 256, nseq)?;

            for il in 0..hp.n_layer {
                let l = &self.layers[il];

                // One norm and one quantization serve Q, K and V: all three
                // read the same activation.
                ops::rms_scale(&s.x, &s.inv, n_embd, nseq, eps)?;
                ops::norm_quant(
                    &s.x,
                    &l.attn_norm,
                    &s.inv,
                    &s.xb,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    n_embd,
                    nseq,
                )?;
                ops::matvec(&l.attn_q, &s.qb, &s.sb, &s.ad, &s.q, n_head * hd, nseq)?;
                ops::matvec(&l.attn_k, &s.qb, &s.sb, &s.ad, &s.k, kv_dim, nseq)?;
                ops::matvec(&l.attn_v, &s.qb, &s.sb, &s.ad, &s.v, kv_dim, nseq)?;

                // QK-norm then RoPE, then the cache, then attend.
                ops::norm_rope(
                    &s.q,
                    l.attn_q_norm.as_ref(),
                    &s.meta,
                    n_head,
                    hd,
                    nseq,
                    fb,
                    eps,
                )?;
                ops::k_norm_rope_store(
                    &s.k,
                    l.attn_k_norm.as_ref(),
                    &s.v,
                    &self.kc[il],
                    &self.vc[il],
                    &s.meta,
                    n_head_kv,
                    hd,
                    max_kv,
                    nseq,
                    fb,
                    eps,
                )?;
                ops::attention(
                    &s.q,
                    &self.kc[il],
                    &self.vc[il],
                    &s.attn,
                    &s.scores,
                    &s.meta,
                    n_head,
                    n_head_kv,
                    hd,
                    max_kv,
                    nseq,
                )?;

                ops::quantize(&s.attn, &s.qb, &s.sb, &s.ad, ((n_head * hd) / 256) * nseq)?;
                ops::matvec(&l.attn_out, &s.qb, &s.sb, &s.ad, &s.xb2, n_embd, nseq)?;
                ops::add(&s.x, &s.xb2, &s.x, n_embd * nseq)?;

                // FFN norm: f32 for the router, quantized for the experts.
                ops::rms_scale(&s.x, &s.inv, n_embd, nseq, eps)?;
                ops::norm_quant(
                    &s.x,
                    &l.ffn_norm,
                    &s.inv,
                    &s.xb,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    n_embd,
                    nseq,
                )?;
                ops::matvec_f32(
                    &l.ffn_gate_inp,
                    &s.xb,
                    &s.router_logits,
                    n_embd,
                    n_expert,
                    nseq,
                )?;
                ops::router(
                    &s.router_logits,
                    &s.expert_ids,
                    &s.expert_w,
                    &s.router_probs,
                    n_expert,
                    n_used,
                    nseq,
                )?;

                ops::grouped(
                    &l.ffn_gate_exps,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    &s.e_gate,
                    &s.expert_ids,
                    n_ffe,
                    n_used,
                    nseq,
                )?;
                ops::grouped(
                    &l.ffn_up_exps,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    &s.e_up,
                    &s.expert_ids,
                    n_ffe,
                    n_used,
                    nseq,
                )?;
                // SiLU on the GATE only, then elementwise with up.
                ops::silu_mul(&s.e_gate, &s.e_up, &s.e_act, n_used * n_ffe * nseq)?;
                // Each expert's down input is its OWN activation.
                ops::quantize(
                    &s.e_act,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    ((n_used * n_ffe) / 256) * nseq,
                )?;
                ops::grouped_per_act(
                    &l.ffn_down_exps,
                    &s.qb,
                    &s.sb,
                    &s.ad,
                    &s.e_down,
                    &s.expert_ids,
                    n_embd,
                    n_used,
                    nseq,
                )?;
                ops::moe_combine_add(&s.e_down, &s.expert_w, &s.x, &s.x, n_used, n_embd, nseq)?;
            }

            ops::rms_norm_mul(&s.x, &self.output_norm, &s.xb, n_embd, nseq, eps)?;
            ops::quantize(&s.xb, &s.qb, &s.sb, &s.ad, (n_embd / 256) * nseq)?;
            ops::matvec(
                &self.output,
                &s.qb,
                &s.sb,
                &s.ad,
                &s.logits,
                hp.n_vocab,
                nseq,
            )?;
            ops::argmax(
                &s.logits,
                &s.amax_v,
                &s.amax_i,
                &s.out_token,
                hp.n_vocab,
                ARGMAX_CHUNKS,
                nseq,
            )
        })();
        // Close the recording whatever happened: leaving it open would refuse
        // every later upload and wedge the device.
        let submitted = ops::end_and_wait();
        queued?;
        submitted?;

        let mut got = vec![0i32; nseq];
        s.out_token.download_i32(&mut got)?;
        let mut out = Vec::with_capacity(nseq);
        for (i, &g) in got.iter().enumerate() {
            if g < 0 || g as usize >= hp.n_vocab {
                return Err(Error::other(format!("sequence {i} argmax returned {g}")));
            }
            out.push(g as u32);
        }
        Ok(out)
    }

    /// The logits for the last pass, copied back. For comparing against the CPU
    /// reference; generation does not need them.
    pub fn logits(&self, nseq: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.hp.n_vocab * nseq];
        self.s.logits.download_f32(&mut out)?;
        Ok(out)
    }
}
