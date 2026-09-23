//! `DeviceModel` — the whole model resident in VRAM, a token computed on the GPU.
//!
//! THE NUMBER THIS EXISTS TO CHANGE. `forward` in this crate computes a token
//! on the CPU at 2.755 s — 0.363 tok/s, against llama.cpp's measured 94.39 on
//! the same card. The grouped expert kernel showed the matmul path can beat
//! llama.cpp (129 tok/s of weight traffic against its 94.39), but none of that
//! reaches a user while the forward pass runs on the host.
//!
//! WHY NOT JUST CALL THE GPU KERNELS FROM `forward`. Because
//! `DeviceWeights::matvec` copies activations in, synchronises, and copies
//! results out. A token runs ~1400 matmuls, and at the measured 0.12 ms
//! round-trip that is 167 ms/token — 6 tok/s, a ceiling no kernel can break.
//! The round-trip, not the arithmetic, is what had to go.
//!
//! SO: weights uploaded once, activations never leaving VRAM, every operation
//! queued without synchronising, ONE synchronise after the whole token, and
//! the only value read back is the argmax — a single `i32`.
//!
//! CORRECTNESS. This reproduces `forward_layer` op for op. Where the CPU path
//! makes a choice that affects the emitted token — SiLU on the gate only,
//! QK-norm before RoPE, NeoX pairing, ties toward the lower expert index, the
//! 2^-14 clamp on the routing divisor — the device kernel makes the same one,
//! and `llamd-testing/tests/device_forward.rs` compares them directly.

use llamd_core::{Error, Result};
use llamd_models::Model;
use llamd_rocm::{DevBuf, op};

/// One layer's weights, already on the device.
struct DeviceLayer {
    attn_norm: DevBuf,
    attn_q: DevBuf,
    attn_k: DevBuf,
    attn_v: DevBuf,
    attn_q_norm: Option<DevBuf>,
    attn_k_norm: Option<DevBuf>,
    attn_out: DevBuf,
    ffn_norm: DevBuf,
    /// The MoE gate. F32 in this model, not Q6_K.
    ffn_gate_inp: DevBuf,
    ffn_gate_exps: DevBuf,
    ffn_up_exps: DevBuf,
    ffn_down_exps: DevBuf,
}

/// Activation buffers, allocated once and reused for every token.
struct Scratch {
    x: DevBuf,
    xb: DevBuf,
    xb2: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn: DevBuf,
    aq: DevBuf,
    ad: DevBuf,
    router_logits: DevBuf,
    router_probs: DevBuf,
    expert_ids: DevBuf,
    expert_w: DevBuf,
    e_gate: DevBuf,
    e_up: DevBuf,
    e_act: DevBuf,
    e_down: DevBuf,
    moe_out: DevBuf,
    scores: DevBuf,
    logits: DevBuf,
    amax_v: DevBuf,
    amax_i: DevBuf,
    out_token: DevBuf,
}

pub struct DeviceModel {
    hp: llamd_models::Hparams,
    tok_embd: DevBuf,
    embd_blocks: usize,
    layers: Vec<DeviceLayer>,
    output_norm: DevBuf,
    output: DevBuf,
    kc: Vec<DevBuf>,
    vc: Vec<DevBuf>,
    max_kv: usize,
    s: Scratch,
    /// Bytes of weights uploaded, for reporting.
    weight_bytes: usize,
    /// Per-phase timings, filled only when profiling.
    ///
    /// WHY THIS IS OPT-IN. Measuring a phase means synchronising after it,
    /// which is precisely the stall this module exists to avoid — a profiled
    /// token is slower than a real one and its total must never be quoted as
    /// throughput. It is here because "23.85 ms against a 7.73 ms matmul
    /// budget" is a question that guesswork answers badly.
    pub profile: Option<Profile>,
}

/// Accumulated seconds per phase of a token.
#[derive(Default, Clone, Debug)]
pub struct Profile {
    pub embed: f64,
    pub norms: f64,
    pub qkv: f64,
    pub qk_norm_rope: f64,
    pub attention: f64,
    pub attn_out: f64,
    pub router: f64,
    pub experts: f64,
    pub moe_tail: f64,
    pub output: f64,
    pub tokens: usize,
}

impl Profile {
    pub fn total(&self) -> f64 {
        self.embed
            + self.norms
            + self.qkv
            + self.qk_norm_rope
            + self.attention
            + self.attn_out
            + self.router
            + self.experts
            + self.moe_tail
            + self.output
    }

    pub fn report(&self) -> String {
        let n = self.tokens.max(1) as f64;
        let t = self.total() / n * 1e3;
        let row = |name: &str, v: f64| {
            let ms = v / n * 1e3;
            format!("    {name:<16} {ms:>8.3} ms  {:>5.1}%\n", 100.0 * ms / t)
        };
        let mut s = format!("  PHASE BREAKDOWN, {} profiled tokens\n", self.tokens);
        s.push_str(&row("embed", self.embed));
        s.push_str(&row("norms", self.norms));
        s.push_str(&row("q/k/v proj", self.qkv));
        s.push_str(&row("qk-norm+rope", self.qk_norm_rope));
        s.push_str(&row("attention", self.attention));
        s.push_str(&row("attn_out+res", self.attn_out));
        s.push_str(&row("router", self.router));
        s.push_str(&row("experts", self.experts));
        s.push_str(&row("moe tail", self.moe_tail));
        s.push_str(&row("output+argmax", self.output));
        s.push_str(&format!(
            "    {:<16} {t:>8.3} ms  (profiled, includes per-phase sync)\n",
            "total"
        ));
        s
    }
}

const ARGMAX_BLOCKS: usize = 512;

/// Raw quantized bytes for a tensor, as stored in the file.
fn raw<'a>(model: &'a Model, name: &str) -> Result<&'a [u8]> {
    let info = model.info(name)?;
    model.gguf().tensor_data(info)
}

/// Upload a Q6_K (or any quantized) tensor verbatim — no dequantization.
fn up_quant(model: &Model, name: &str) -> Result<DevBuf> {
    DevBuf::from_bytes(raw(model, name)?).map_err(|e| Error::other(format!("{name}: {e}")))
}

/// Upload an F32 tensor. Norms and the MoE gate are stored F32, and
/// `Model::tensor` returns them without conversion.
fn up_f32(model: &Model, name: &str) -> Result<DevBuf> {
    let t = model.tensor(name)?;
    DevBuf::from_f32(t.data()).map_err(|e| Error::other(format!("{name}: {e}")))
}

impl DeviceModel {
    /// Upload the model. `max_kv` is the longest sequence this instance can
    /// serve — the KV cache is allocated once, because growing it mid-token
    /// would mean a device allocation inside the hot path.
    pub fn load(model: &Model, max_kv: usize) -> Result<Self> {
        let hp = model.hparams.clone();
        if !hp.arch.is_moe() {
            return Err(Error::unsupported(
                "device model",
                "only MoE models are wired up; a dense FFN has no expert path",
            ));
        }
        if max_kv == 0 {
            return Err(Error::malformed("device model", "max_kv of 0"));
        }

        let (n_embd, n_head, hd) = (hp.n_embd, hp.n_head, hp.n_head_dim);
        let kv_dim = hp.kv_dim();
        let n_used = hp.n_expert_used;
        let n_ffe = hp.n_ff_expert;

        // Refuse early and legibly rather than failing on some allocation
        // partway through a 23 GiB upload.
        let need = raw(model, &model.tok_embd)?.len() as u64
            + model
                .output
                .as_ref()
                .map(|o| raw(model, o).map(|b| b.len() as u64))
                .transpose()?
                .unwrap_or(0)
            + (0..hp.n_layer)
                .map(|il| {
                    let l = &model.layers[il];
                    let mut n = 0u64;
                    for name in [&l.attn_q, &l.attn_k, &l.attn_v, &l.attn_out] {
                        n += raw(model, name).map(|b| b.len() as u64).unwrap_or(0);
                    }
                    for name in [&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps] {
                        if let Some(n2) = name.as_ref() {
                            n += raw(model, n2).map(|b| b.len() as u64).unwrap_or(0);
                        }
                    }
                    n
                })
                .sum::<u64>();
        let free = llamd_rocm::free_memory() as u64;
        if free > 0 && need + (1 << 30) > free {
            return Err(Error::other(format!(
                "model needs about {:.1} GiB of weights but only {:.1} GiB is free",
                need as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64
            )));
        }

        let tok_embd = up_quant(model, &model.tok_embd)?;
        let embd_blocks = n_embd / 256;
        let output_norm = up_f32(model, &model.output_norm)?;
        let output = match &model.output {
            Some(o) => up_quant(model, o)?,
            None => {
                return Err(Error::unsupported(
                    "device model",
                    "tied output embeddings are not wired up",
                ));
            }
        };

        let mut layers = Vec::with_capacity(hp.n_layer);
        for il in 0..hp.n_layer {
            let l = &model.layers[il];
            layers.push(DeviceLayer {
                attn_norm: up_f32(model, &l.attn_norm)?,
                attn_q: up_quant(model, &l.attn_q)?,
                attn_k: up_quant(model, &l.attn_k)?,
                attn_v: up_quant(model, &l.attn_v)?,
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
                attn_out: up_quant(model, &l.attn_out)?,
                ffn_norm: up_f32(model, &l.ffn_norm)?,
                ffn_gate_inp: up_f32(
                    model,
                    l.ffn_gate_inp
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no MoE router"))?,
                )?,
                ffn_gate_exps: up_quant(
                    model,
                    l.ffn_gate_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no gate experts"))?,
                )?,
                ffn_up_exps: up_quant(
                    model,
                    l.ffn_up_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no up experts"))?,
                )?,
                ffn_down_exps: up_quant(
                    model,
                    l.ffn_down_exps
                        .as_ref()
                        .ok_or_else(|| Error::unsupported("layer", "no down experts"))?,
                )?,
            });
        }

        let f = |n: usize| DevBuf::zeroed(n * 4);
        // The largest activation any matvec consumes decides the quantized
        // activation scratch: attn_out takes n_head*hd, the output projection
        // n_embd, the experts n_ff_expert.
        // The down projection quantizes all n_used expert activations at
        // once, so the scratch must hold n_used * n_ff_expert — larger than
        // any single projection's input.
        let max_act = n_embd.max(n_head * hd).max(n_used * n_ffe);
        let max_blocks = max_act.div_ceil(256);

        let s = Scratch {
            x: f(n_embd)?,
            xb: f(n_embd)?,
            xb2: f(n_embd)?,
            q: f(n_head * hd)?,
            k: f(kv_dim)?,
            v: f(kv_dim)?,
            attn: f(n_head * hd)?,
            aq: DevBuf::zeroed(max_blocks * 256)?,
            ad: f(max_blocks)?,
            router_logits: f(hp.n_expert)?,
            router_probs: f(hp.n_expert)?,
            expert_ids: DevBuf::zeroed(n_used * 4)?,
            expert_w: f(n_used)?,
            e_gate: f(n_used * n_ffe)?,
            e_up: f(n_used * n_ffe)?,
            e_act: f(n_used * n_ffe)?,
            e_down: f(n_used * n_embd)?,
            moe_out: f(n_embd)?,
            scores: f(n_head * max_kv)?,
            logits: f(hp.n_vocab)?,
            amax_v: f(ARGMAX_BLOCKS)?,
            amax_i: DevBuf::zeroed(ARGMAX_BLOCKS * 4)?,
            out_token: DevBuf::zeroed(4)?,
        };

        let mut kc = Vec::with_capacity(hp.n_layer);
        let mut vc = Vec::with_capacity(hp.n_layer);
        for _ in 0..hp.n_layer {
            kc.push(DevBuf::zeroed(max_kv * kv_dim * 4)?);
            vc.push(DevBuf::zeroed(max_kv * kv_dim * 4)?);
        }

        Ok(DeviceModel {
            hp,
            tok_embd,
            embd_blocks,
            layers,
            output_norm,
            output,
            kc,
            vc,
            max_kv,
            s,
            weight_bytes: need as usize,
            profile: None,
        })
    }

    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn hparams(&self) -> &llamd_models::Hparams {
        &self.hp
    }

    /// Quantize the activation in `src` and matvec it against `w`.
    ///
    /// Both launches go into the queue; neither waits. The activation is
    /// quantized ON THE DEVICE, which is what lets the whole token stay here —
    /// with that conversion on the host every matmul would need a round-trip.
    fn qmatvec(&self, w: &DevBuf, src: &DevBuf, out: &DevBuf, k: usize, rows: usize) -> Result<()> {
        let blocks = k / 256;
        op::quantize(src, &self.s.aq, &self.s.ad, blocks)?;
        op::matvec(w, &self.s.aq, &self.s.ad, out, blocks, rows)?;
        Ok(())
    }

    /// Matvec against the activation ALREADY quantized in the scratch.
    ///
    /// WHY THIS EXISTS. Q, K and V all consume the same normalised x, so
    /// `qmatvec` for each of them quantized identical data three times. The
    /// phase breakdown put q/k/v at 4.07 ms of a 23.8 ms profiled token, well
    /// above the 2.16 ms its matmuls need — two thirds of the excess was
    /// redundant quantization. Quantize once, then use this.
    fn matvec_pre(&self, w: &DevBuf, out: &DevBuf, k: usize, rows: usize) -> Result<()> {
        op::matvec(w, &self.s.aq, &self.s.ad, out, k / 256, rows)?;
        Ok(())
    }

    /// One token. Queues every operation, synchronises once, returns the argmax.
    ///
    /// `pos` is the position in the sequence; the KV cache for every layer is
    /// written at that position before the layer attends, because a layer reads
    /// only its own keys.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<u32> {
        if pos >= self.max_kv {
            return Err(Error::malformed(
                "device forward",
                format!("position {pos} beyond the {} allocated", self.max_kv),
            ));
        }
        let hp = &self.hp;
        let (n_embd, n_head, n_head_kv, hd) = (hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_head_dim);
        let kv_dim = hp.kv_dim();
        let (n_expert, n_used, n_ffe) = (hp.n_expert, hp.n_expert_used, hp.n_ff_expert);
        let eps = hp.rms_eps;
        let n_kv = pos + 1;
        let profiling = self.profile.is_some();
        let s = &self.s;

        // When profiling, each phase ends with a synchronise so its cost can
        // be attributed. Otherwise this closure is a no-op and the whole token
        // stays queued, which is the point of the module.
        macro_rules! phase {
            ($field:ident, $body:block) => {{
                if profiling {
                    let t = std::time::Instant::now();
                    $body
                    llamd_rocm::sync()?;
                    if let Some(p) = self.profile.as_mut() {
                        p.$field += t.elapsed().as_secs_f64();
                    }
                } else {
                    $body
                }
            }};
        }

        phase!(embed, {
            op::embed(&self.tok_embd, token, &s.x, self.embd_blocks)?;
        });

        for il in 0..hp.n_layer {
            let l = &self.layers[il];

            // ── attention norm ───────────────────────────────────────────
            phase!(norms, {
                op::rms_norm_mul(&s.x, &l.attn_norm, &s.xb, n_embd, eps)?;
            });

            // ── projections ──────────────────────────────────────────────
            phase!(qkv, {
                // One quantization for all three projections: they read the
                // same normalised x.
                op::quantize(&s.xb, &s.aq, &s.ad, n_embd / 256)?;
                self.matvec_pre(&l.attn_q, &s.q, n_embd, n_head * hd)?;
                self.matvec_pre(&l.attn_k, &s.k, n_embd, kv_dim)?;
                self.matvec_pre(&l.attn_v, &s.v, n_embd, kv_dim)?;
            });

            // ── QK-norm, per head, BEFORE RoPE ───────────────────────────
            phase!(qk_norm_rope, {
                if let Some(qn) = &l.attn_q_norm {
                    op::head_rms_norm_mul(&s.q, qn, &s.q, n_head, hd, eps)?;
                }
                if let Some(kn) = &l.attn_k_norm {
                    op::head_rms_norm_mul(&s.k, kn, &s.k, n_head_kv, hd, eps)?;
                }
                op::rope(&s.q, pos as u32, n_head, hd, hp.rope_freq_base)?;
                op::rope(&s.k, pos as u32, n_head_kv, hd, hp.rope_freq_base)?;
                op::kv_store(&self.kc[il], &self.vc[il], &s.k, &s.v, pos, kv_dim)?;
            });

            phase!(attention, {
                op::attention(
                    &s.q,
                    &self.kc[il],
                    &self.vc[il],
                    &s.attn,
                    &s.scores,
                    n_head,
                    n_head_kv,
                    hd,
                    n_kv,
                )?;
            });

            // ── output projection and residual ───────────────────────────
            phase!(attn_out, {
                self.qmatvec(&l.attn_out, &s.attn, &s.xb2, n_head * hd, n_embd)?;
                op::add(&s.x, &s.xb2, &s.x, n_embd)?;
            });

            // ── FFN norm and routing ─────────────────────────────────────
            phase!(router, {
                op::rms_norm_mul(&s.x, &l.ffn_norm, &s.xb, n_embd, eps)?;
                op::matvec_f32(&l.ffn_gate_inp, &s.xb, &s.router_logits, n_embd, n_expert)?;
                op::router(
                    &s.router_logits,
                    n_expert,
                    n_used,
                    &s.expert_ids,
                    &s.expert_w,
                    &s.router_probs,
                )?;
            });

            // ── the experts, all of them in one launch each ──────────────
            phase!(experts, {
                let blocks_in = n_embd / 256;
                op::quantize(&s.xb, &s.aq, &s.ad, blocks_in)?;
                op::matvec_grouped(
                    &l.ffn_gate_exps,
                    &s.expert_ids,
                    &s.aq,
                    &s.ad,
                    &s.e_gate,
                    blocks_in,
                    n_ffe,
                    n_used,
                )?;
                op::matvec_grouped(
                    &l.ffn_up_exps,
                    &s.expert_ids,
                    &s.aq,
                    &s.ad,
                    &s.e_up,
                    blocks_in,
                    n_ffe,
                    n_used,
                )?;
                // SiLU on the GATE only, then elementwise with up. SwiGLU is not
                // symmetric: the other order still produces fluent text.
                op::silu_mul(&s.e_gate, &s.e_up, &s.e_act, n_used * n_ffe)?;

                // The down projection consumes each expert's own activation, so
                // its input is n_used blocks of n_ffe, quantized together.
                // n_used * n_ffe / 256 blocks, laid out as n_used runs of
                // n_ffe/256 — exactly what matvec_grouped_per_act indexes.
                op::quantize(&s.e_act, &s.aq, &s.ad, (n_used * n_ffe) / 256)?;
                op::matvec_grouped_per_act(
                    &l.ffn_down_exps,
                    &s.expert_ids,
                    &s.aq,
                    &s.ad,
                    &s.e_down,
                    n_ffe / 256,
                    n_embd,
                    n_used,
                )?;
            });
            phase!(moe_tail, {
                op::moe_combine(&s.e_down, &s.expert_w, &s.moe_out, n_used, n_embd)?;
                op::add(&s.x, &s.moe_out, &s.x, n_embd)?;
            });
        }

        phase!(output, {
            op::rms_norm_mul(&s.x, &self.output_norm, &s.xb, n_embd, eps)?;
            self.qmatvec(&self.output, &s.xb, &s.logits, n_embd, hp.n_vocab)?;
            op::argmax(
                &s.logits,
                hp.n_vocab,
                &s.amax_v,
                &s.amax_i,
                &s.out_token,
                ARGMAX_BLOCKS,
            )?;
        });
        if let Some(p) = self.profile.as_mut() {
            p.tokens += 1;
        }

        // THE ONE SYNCHRONISE. Everything above was queued.
        llamd_rocm::sync()?;

        let mut got = [0i32; 1];
        s.out_token.download_i32(&mut got)?;
        if got[0] < 0 || got[0] as usize >= hp.n_vocab {
            return Err(Error::other(format!("argmax returned {}", got[0])));
        }
        Ok(got[0] as u32)
    }

    /// The logits for the last token, copied back. For comparing against the
    /// CPU reference; generation does not need them.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.hp.n_vocab];
        self.s.logits.download_f32(&mut out)?;
        Ok(out)
    }
}
