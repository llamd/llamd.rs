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
    /// The tile's (token, slot) pairs sorted by expert, built on the device
    /// between the router and the expert matmuls.
    g_start: VkBuf,
    g_count: VkBuf,
    p_token: VkBuf,
    p_slot: VkBuf,
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
    /// Independent-sequence slices the KV cache holds.
    ///
    /// DECOUPLED FROM `max_seq` on purpose. A prefill tile of N tokens needs ONE
    /// slice — the slots are positions in one sequence and share the cache, which
    /// is what `kv_stride = 0` expresses — while N independent decode sequences
    /// need N. Sizing the cache by the prefill tile would cost 7.55 GB at 64
    /// tokens against 0.94 GB at 8, on top of 23.4 GB of weights.
    kv_slots: usize,
    s: Scratch,
    weight_bytes: usize,
    /// Tokens per pass at or above which the expert matmuls are grouped BY
    /// EXPERT rather than by token. `usize::MAX` never, `1` always.
    ///
    /// EXPERT-MAJOR ONLY PAYS WHEN EXPERTS ARE SHARED. Token-major reads an
    /// expert's weights once per token that chose it; expert-major reads them
    /// once for all of them, which is worth 2.3 reads at 32 tokens and exactly
    /// one at a single token. Measured on a 512-token prompt:
    ///
    ///   tile  8   348.6 -> 351.2 t/s   1.01x
    ///   tile 16   465.8 -> 520.9 t/s   1.12x
    ///   tile 32   481.7 -> 591.3 t/s   1.23x
    ///
    /// and in decode, where nothing is shared, it is a 1.41x LOSS: 122.3 tok/s
    /// token-major against 86.5 expert-major, because the grouping dispatch and
    /// the gathered activation reads are then pure overhead.
    ///
    /// So the default is 16: prefill tiles take the expert-major path and decode
    /// does not. The threshold is a field rather than a constant so the two can
    /// still be A/B'd in one process.
    pub expert_major_min_tokens: usize,
    /// Tokens per prefill pass. Defaults to 32, MEASURED.
    ///
    /// A wider tile shares each expert's weights across more tokens — 4.07 tokens
    /// per expert at 64 against 2.29 at 32 — and costs registers, because the
    /// accumulator is that wide. On a 512-token prompt:
    ///
    ///   tile 16   542.3 tok/s
    ///   tile 32   638.7 tok/s
    ///   tile 64   551.0 tok/s
    ///
    /// 64 loses despite the extra sharing: its variant needs 192 VGPRs, which is
    /// one wave per SIMD where 32 fits three, and this kernel is latency-bound —
    /// the same reason the tiled GEMM attempt failed (benchmarks.md §16).
    pub prefill_tile: usize,
    /// Issue Q, K and V as one dispatch rather than three.
    ///
    /// Switchable so the two can be A/B'd IN ONE PROCESS. Comparing across
    /// separate runs is how this project twice accepted a change that
    /// measurement later rejected: run-to-run drift on this card is larger than
    /// the effect being looked for.
    pub fuse_qkv: bool,
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
    /// `max_seq` is the widest pass — the prefill tile. `kv_slots` is how many
    /// INDEPENDENT sequences may decode at once, and is what the KV cache is
    /// sized by; prefill needs only one slice however wide its tile.
    pub fn load(model: &Model, max_kv: usize, max_seq: usize) -> Result<Self> {
        Self::load_with_slots(model, max_kv, max_seq, max_seq.min(8))
    }

    pub fn load_with_slots(
        model: &Model,
        max_kv: usize,
        max_seq: usize,
        kv_slots: usize,
    ) -> Result<Self> {
        llamd_vulkan::init()?;
        let hp = model.hparams.clone();
        if !hp.arch.is_moe() {
            return Err(Error::unsupported(
                "vulkan model",
                "only MoE models are wired up",
            ));
        }
        if kv_slots == 0 || kv_slots > max_seq {
            return Err(Error::malformed(
                "vulkan model",
                format!("kv_slots {kv_slots} must be in 1..={max_seq}"),
            ));
        }
        if max_kv == 0 || max_seq == 0 || max_seq > llamd_vulkan::MAX_TILE {
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
            g_start: VkBuf::new(n_expert * 4)?,
            g_count: VkBuf::new(n_expert * 4)?,
            p_token: VkBuf::new(n_used * max_seq * 4)?,
            p_slot: VkBuf::new(n_used * max_seq * 4)?,
        };

        let mut kc = Vec::with_capacity(hp.n_layer);
        let mut vc = Vec::with_capacity(hp.n_layer);
        for _ in 0..hp.n_layer {
            kc.push(VkBuf::new(max_kv * kv_dim * 4 * kv_slots)?);
            vc.push(VkBuf::new(max_kv * kv_dim * 4 * kv_slots)?);
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
            kv_slots,
            s,
            weight_bytes,
            expert_major_min_tokens: 16,
            // The tile is no longer bounded by the accumulator width — the shaders
            // chunk the token dimension — so this is now a free measurement rather
            // than a register limit. 64 measured best and the surface is flat from
            // 64 to 512, which is itself the finding: see benchmarks.md §20.
            prefill_tile: max_seq.min(64),
            fuse_qkv: true,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    /// One pass over `nseq` independent sequences, returning a token each.
    pub fn forward_batch(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<u32>> {
        if tokens.len() > self.kv_slots {
            return Err(Error::malformed(
                "forward_batch",
                format!(
                    "{} independent sequences exceeds the {} KV slices allocated; \
                     a wider PREFILL tile does not imply more slices",
                    tokens.len(),
                    self.kv_slots
                ),
            ));
        }
        let kv_dim = self.hp.kv_dim();
        let stride = self.max_kv * kv_dim;
        self.run(tokens, positions, stride, true)?
            .ok_or_else(|| Error::other("no logits produced"))
    }

    /// The shared engine behind `forward_batch` and `prefill`.
    ///
    /// `kv_stride` is the distance between one slot's cache slice and the next:
    /// `max_kv * kv_dim` for independent sequences, 0 for prefill where the slots
    /// are positions in one sequence. `want_logits` false skips the final norm,
    /// the output projection and the argmax, which is what makes a non-final
    /// prefill tile cheap.
    fn run(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        kv_stride: usize,
        want_logits: bool,
    ) -> Result<Option<Vec<u32>>> {
        let nseq = tokens.len();
        if nseq == 0 || nseq != positions.len() || nseq > self.max_seq {
            return Err(Error::malformed(
                "vulkan run",
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
        // Decided once per pass, from the token count.
        let expert_major = self.expert_major_min_tokens > 0 && nseq >= self.expert_major_min_tokens;
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
                // ONE dispatch for all three, or three — measured, not assumed.
                // Separately they are 1024, 128 and 128 workgroups; fused, the
                // row space is 5120 so K and V fill the card alongside Q, and
                // two compute barriers disappear. The same change was worth 23%
                // on the ROCm side.
                if self.fuse_qkv {
                    ops::matvec_x3(
                        (&l.attn_q, &l.attn_k, &l.attn_v),
                        &s.qb,
                        &s.sb,
                        &s.ad,
                        (&s.q, &s.k, &s.v),
                        (n_head * hd, kv_dim, kv_dim),
                        nseq,
                    )?;
                } else {
                    ops::matvec(&l.attn_q, &s.qb, &s.sb, &s.ad, &s.q, n_head * hd, nseq)?;
                    ops::matvec(&l.attn_k, &s.qb, &s.sb, &s.ad, &s.k, kv_dim, nseq)?;
                    ops::matvec(&l.attn_v, &s.qb, &s.sb, &s.ad, &s.v, kv_dim, nseq)?;
                }

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
                    kv_stride,
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
                    kv_stride,
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

                let grp = (&s.g_start, &s.g_count, &s.p_token, &s.p_slot);
                if expert_major {
                    // Sort the tile's (token, slot) pairs by expert, then read each
                    // expert's weights once for every token that chose it.
                    ops::moe_group(
                        &s.expert_ids,
                        &s.g_start,
                        &s.g_count,
                        &s.p_token,
                        &s.p_slot,
                        n_expert,
                        n_used,
                        nseq,
                    )?;
                    if ops::expert_rt_enabled() {
                        ops::expert_row_tiled(
                            &l.ffn_gate_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_gate,
                            grp,
                            n_ffe,
                            n_used,
                            nseq,
                            n_expert,
                            false,
                        )?;
                    } else {
                        ops::expert_major(
                            &l.ffn_gate_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_gate,
                            grp,
                            n_ffe,
                            n_used,
                            nseq,
                            n_expert,
                        )?;
                    }
                    if ops::expert_rt_enabled() {
                        ops::expert_row_tiled(
                            &l.ffn_up_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_up,
                            grp,
                            n_ffe,
                            n_used,
                            nseq,
                            n_expert,
                            false,
                        )?;
                    } else {
                        ops::expert_major(
                            &l.ffn_up_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_up,
                            grp,
                            n_ffe,
                            n_used,
                            nseq,
                            n_expert,
                        )?;
                    }
                } else {
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
                }
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
                if expert_major {
                    // The grouping above still holds: the router has not run again. The
                    // activation here is per (token, slot), NOT per token, because each
                    // expert consumed its own silu(gate)*up.
                    let dg = (&s.g_start, &s.g_count, &s.p_token, &s.p_slot);
                    if ops::expert_rt_enabled() {
                        ops::expert_row_tiled(
                            &l.ffn_down_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_down,
                            dg,
                            n_embd,
                            n_used,
                            nseq,
                            n_expert,
                            // by_slot: the down input is per (token, slot).
                            true,
                        )?;
                    } else {
                        ops::expert_major_per_act(
                            &l.ffn_down_exps,
                            &s.qb,
                            &s.sb,
                            &s.ad,
                            &s.e_down,
                            dg,
                            n_embd,
                            n_used,
                            nseq,
                            n_expert,
                        )?;
                    }
                } else {
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
                }
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

        if !want_logits {
            return Ok(None);
        }
        let mut got = vec![0i32; nseq];
        self.s.out_token.download_i32(&mut got)?;
        let mut out = Vec::with_capacity(nseq);
        for (i, &g) in got.iter().enumerate() {
            if g < 0 || g as usize >= self.hp.n_vocab {
                return Err(Error::other(format!("slot {i} argmax returned {g}")));
            }
            out.push(g as u32);
        }
        Ok(Some(out))
    }

    /// Process `tokens` of ONE sequence in tiles, returning the token that
    /// follows them.
    ///
    /// WHY THIS EXISTS, and it is the gap that mattered most. `forward_batch`
    /// decodes independent sequences; a prompt is not that. Without prefill a
    /// 512-token prompt had to be fed one token at a time, reading the whole
    /// 2.485 GB of active weights for every one of them — 4.5 s against
    /// llama.cpp's 0.31 s. Decode speed is irrelevant if ingesting the prompt
    /// costs fourteen times more.
    ///
    /// PREFILL IS BATCHING WITH A SHARED CACHE. The layer loop is the same code:
    /// every token multiplies the same weights, so one read serves the whole
    /// tile. The single difference is that the slots are POSITIONS IN ONE
    /// SEQUENCE rather than separate streams, so they share the KV cache —
    /// expressed as `kv_stride = 0`.
    ///
    /// CAUSALITY NEEDS NO MASK. Each slot's `n_kv` is its own position plus one,
    /// so token i reads exactly positions 0..pos_i. Every slot writes its K and V
    /// before the attention dispatch — the barrier between them guarantees it —
    /// so what token i reads is the earlier tokens of the same tile.
    ///
    /// Returns the argmax of the LAST token, which is the first token to generate.
    pub fn prefill(&mut self, tokens: &[u32], base_pos: usize) -> Result<u32> {
        if tokens.is_empty() {
            return Err(Error::malformed("prefill", "no tokens"));
        }
        if base_pos + tokens.len() > self.max_kv {
            return Err(Error::malformed(
                "prefill",
                format!(
                    "{} tokens from {base_pos} exceeds the {} allocated",
                    tokens.len(),
                    self.max_kv
                ),
            ));
        }
        let tile = self.prefill_tile.clamp(1, self.max_seq);
        let mut last = None;
        for (chunk, toks) in tokens.chunks(tile).enumerate() {
            let start = base_pos + chunk * tile;
            let positions: Vec<usize> = (0..toks.len()).map(|i| start + i).collect();
            // Only the final tile needs logits: the output projection is 255 MB,
            // 10% of a token's traffic, and every earlier token's logits are
            // discarded. Skipping it on the other tiles is free correctness.
            let is_last = start + toks.len() == base_pos + tokens.len();
            let out = self.run(toks, &positions, 0, is_last)?;
            if is_last {
                last = out.map(|v| v[toks.len() - 1]);
            }
        }
        last.ok_or_else(|| Error::other("prefill produced no logits"))
    }

    /// The logits for the last pass, copied back. For comparing against the CPU
    /// reference; generation does not need them.
    pub fn logits(&self, nseq: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.hp.n_vocab * nseq];
        self.s.logits.download_f32(&mut out)?;
        Ok(out)
    }
}
