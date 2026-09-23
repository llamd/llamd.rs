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
use llamd_rocm::{DevBuf, bop, op};

/// One layer's weights, already on the device.
struct DeviceLayer {
    attn_norm: DevBuf,
    /// INTERLEAVED, i.e. ggml's own 210-byte blocks, and not planar.
    ///
    /// A planar repack into four 16-byte-aligned planes is implemented
    /// (`llamd_quant::repack_q6k_planar`, `llamd_rocm::PlanarWeights`) and
    /// measures 439.7 -> 510.6 GB/s on a 243 MiB tensor. In the engine it
    /// measured 102.0 tok/s against 103.1 interleaved — no better, and 18.3 s
    /// of repack at load against 3.8 s. Kept as evidence, not used. See
    /// benchmarks.md §9.
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
    scores: DevBuf,
    logits: DevBuf,
    amax_v: DevBuf,
    amax_i: DevBuf,
    out_token: DevBuf,
    /// One float: 1/rms, handed from fk_rms_scale to fk_norm_quant.
    inv: DevBuf,
    /// `[token, pos]`, device-resident.
    ///
    /// WHY THESE LIVE ON THE DEVICE. A captured HIP graph freezes its kernel
    /// arguments. While the token id and position were arguments, the graph had
    /// to be re-recorded every token, which costs more than it saves. Kernels
    /// read them from here instead, so the graph is structurally invariant and
    /// the per-token host work is an 8-byte upload.
    meta: DevBuf,
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
    /// Sequences this instance's buffers are sized for.
    max_seq: usize,
    s: Scratch,
    /// Bytes of weights uploaded, for reporting.
    weight_bytes: usize,
    /// This instance's identity, so a graph recorded by another model is never
    /// replayed against these buffers.
    id: u64,
    /// Waves per output row for the q/k/v, gate/up and down projections.
    ///
    /// `(1, 1, 1)` — split-k OFF — because that measured fastest.
    ///
    /// MEASURED IN THE ENGINE, not standalone, and the two disagree completely.
    /// A kernel-only benchmark looping one matrix 1000 times keeps it in this
    /// card's 64 MB Infinity Cache; it reported attn_q going 38.8 -> 25.5 us at
    /// nsub=4, a 1.52x gain, and the expert gate 44.5 -> 31.3. Applied to the
    /// engine, where a token reads each layer's weights once from HBM and
    /// nothing is resident, the same settings cost 5-13%.
    ///
    /// The lesson is about the benchmark, not the kernel: a microbenchmark that
    /// loops over data small enough to cache measures cache bandwidth, and on a
    /// card with 64 MB of it almost every individual projection qualifies.
    pub nsub: (usize, usize, usize),
    /// Whether to submit tokens through a captured HIP graph.
    ///
    /// OFF by default. It measured slower before the launches were fused
    /// (13.851 ms/token against 12.545) and is now indistinguishable from
    /// queueing them individually — fusion took ~1200 launches per token down
    /// to ~750, which removed the overhead a graph would have amortised. Kept,
    /// with the capture machinery, so the comparison stays reproducible and a
    /// future ROCm can be retested by flipping one flag.
    pub use_graph: bool,
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

/// Which `DeviceModel` the captured graph belongs to.
///
/// WHY THIS IS NEEDED, and it was found by a memory fault rather than by
/// thinking. The HIP graph is a process-global object holding raw pointers into
/// the buffers of whichever model recorded it. Load a model, capture, drop it,
/// load another: the new model sees a graph already instantiated, replays it,
/// and the kernels read freed memory. The fault surfaced as
/// `HSA_STATUS_ERROR_MEMORY_FAULT ... kernel: fk_embed_q6k_meta` — reading an
/// embedding table that no longer existed.
///
/// A comment saying "do not do that" would not have prevented it. Each model
/// takes an id, and a graph recorded by a different id is discarded rather than
/// replayed.
static GRAPH_OWNER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn kv_dim_of(hp: &llamd_models::Hparams) -> usize {
    hp.kv_dim()
}

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
        Self::load_batched(model, max_kv, 1)
    }

    /// Load with room for `max_seq` concurrent sequences.
    ///
    /// Activations and the KV cache scale with `max_seq`; the weights do not,
    /// which is the entire point of batching.
    pub fn load_batched(model: &Model, max_kv: usize, max_seq: usize) -> Result<Self> {
        // Everything queues on a stream of our own: the default stream cannot
        // be captured into a graph.
        llamd_rocm::stream_init()?;

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

        if max_seq == 0 || max_seq > llamd_rocm::max_seq().max(1) {
            return Err(Error::malformed(
                "device model",
                format!(
                    "max_seq {max_seq} outside 1..={}",
                    llamd_rocm::max_seq().max(1)
                ),
            ));
        }
        // Every activation carries a sequence dimension. The weights do not.
        let f = |n: usize| DevBuf::zeroed(n * max_seq * 4);
        // The largest activation any matvec consumes decides the quantized
        // activation scratch: attn_out takes n_head*hd, the output projection
        // n_embd, the experts n_ff_expert.
        // The down projection quantizes all n_used expert activations at
        // once, so the scratch must hold n_used * n_ff_expert — larger than
        // any single projection's input.
        let max_act = n_embd.max(n_head * hd).max(n_used * n_ffe);
        let max_blocks = max_act.div_ceil(256);
        // `ad` is indexed by (sequence, expert, block) on the down projection,
        // which is the widest use.
        let ad_len = max_blocks.max((n_ffe / 256) * n_used);

        let s = Scratch {
            x: f(n_embd)?,
            xb: f(n_embd)?,
            xb2: f(n_embd)?,
            q: f(n_head * hd)?,
            k: f(kv_dim)?,
            v: f(kv_dim)?,
            attn: f(n_head * hd)?,
            aq: DevBuf::zeroed(max_blocks * 256 * max_seq)?,
            ad: f(ad_len)?,
            router_logits: f(hp.n_expert)?,
            router_probs: f(hp.n_expert)?,
            expert_ids: DevBuf::zeroed(n_used * 4 * max_seq)?,
            expert_w: f(n_used)?,
            e_gate: f(n_used * n_ffe)?,
            e_up: f(n_used * n_ffe)?,
            e_act: f(n_used * n_ffe)?,
            e_down: f(n_used * n_embd)?,
            scores: f(n_head * max_kv)?,
            logits: f(hp.n_vocab)?,
            amax_v: f(ARGMAX_BLOCKS)?,
            amax_i: DevBuf::zeroed(ARGMAX_BLOCKS * 4 * max_seq)?,
            out_token: DevBuf::zeroed(4 * max_seq)?,
            meta: DevBuf::zeroed(8 * max_seq)?,
            inv: f(1)?,
        };

        let mut kc = Vec::with_capacity(hp.n_layer);
        let mut vc = Vec::with_capacity(hp.n_layer);
        for _ in 0..hp.n_layer {
            kc.push(DevBuf::zeroed(max_kv * kv_dim * 4 * max_seq)?);
            vc.push(DevBuf::zeroed(max_kv * kv_dim * 4 * max_seq)?);
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
            max_seq,
            s,
            weight_bytes: need as usize,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            // ALL ONE, i.e. split-k off, measured by the in-situ sweep in
            // device_forward.rs. Every split-k configuration tried was slower:
            // (1,1,1) 9.432 ms, (1,2,1) 9.566, (2,1,1) 9.942, (1,1,2) 10.261,
            // (2,2,2) 10.800. The kernels are kept because the sweep is the
            // evidence and it should stay runnable.
            nsub: (1, 1, 1),
            // Off by default. Graph replay measured SLOWER before the
            // launches were fused (13.851 ms/token against 12.545) and is now
            // indistinguishable from queueing them (9.54-9.66 either way),
            // because fusion removed the launches that would have justified it.
            // Neutral and more machinery to go wrong, so not the default.
            use_graph: false,
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

    /// Queue one token's worth of work. Does not synchronise.
    ///
    /// Split out from `forward` so graph capture can record exactly this and
    /// nothing else. Reads the token and position from the device-resident
    /// `meta` buffer, so the recorded graph is valid for every later token.
    fn queue_token(&mut self) -> Result<()> {
        let hp = &self.hp;
        let (n_embd, n_head, n_head_kv, hd) = (hp.n_embd, hp.n_head, hp.n_head_kv, hp.n_head_dim);
        let (n_expert, n_used, n_ffe) = (hp.n_expert, hp.n_expert_used, hp.n_ff_expert);
        let eps = hp.rms_eps;
        let fb = hp.rope_freq_base;
        let max_kv = self.max_kv;
        let (nsub_qkv, nsub_gu, nsub_dn) = self.nsub;
        let profiling = self.profile.is_some();
        let s = &self.s;

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
            op::embed_meta(&self.tok_embd, &s.meta, &s.x, self.embd_blocks)?;
        });

        for il in 0..hp.n_layer {
            let l = &self.layers[il];

            // ── attention norm, emitting f32 and quantized in one pass ───
            // Q, K and V all read the same activation, so one norm and one
            // quantization serve all three.
            phase!(norms, {
                op::rms_norm_quant(&s.x, &l.attn_norm, &s.xb, &s.aq, &s.ad, &s.inv, n_embd, eps)?;
            });

            phase!(qkv, {
                // One launch for all three: same activation, and the small K
                // and V projections fill the card alongside Q instead of
                // running alone at 64 workgroups.
                // nsub=4 measured: attn_q goes 38.8 -> 25.5 us, 177 -> 269 GB/s.
                op::matvec_x3_splitk(
                    (&l.attn_q, &l.attn_k, &l.attn_v),
                    &s.aq,
                    &s.ad,
                    (&s.q, &s.k, &s.v),
                    n_embd / 256,
                    (n_head * hd, kv_dim_of(hp), kv_dim_of(hp)),
                    nsub_qkv,
                )?;
            });

            // ── QK-norm then RoPE, fused; K also stores into the cache ───
            phase!(qk_norm_rope, {
                op::norm_rope(&s.q, l.attn_q_norm.as_ref(), &s.meta, n_head, hd, fb, eps)?;
                op::k_norm_rope_store(
                    &s.k,
                    l.attn_k_norm.as_ref(),
                    &s.v,
                    &self.kc[il],
                    &self.vc[il],
                    &s.meta,
                    n_head_kv,
                    hd,
                    fb,
                    eps,
                )?;
            });

            phase!(attention, {
                op::attention_meta(
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
                )?;
            });

            phase!(attn_out, {
                // attn_out is [4096 -> 2048]: 2048 rows is 2048 waves against
                // a 4096-wave card, so one wave per row leaves half the GPU
                // idle and it measured 202 GB/s. Four waves per row, each
                // taking 4 of the 16 blocks.
                op::quantize(&s.attn, &s.aq, &s.ad, (n_head * hd) / 256)?;
                // nsub=1: the standalone bench liked 4 here, the engine did not.
                op::matvec(
                    &l.attn_out,
                    &s.aq,
                    &s.ad,
                    &s.xb2,
                    (n_head * hd) / 256,
                    n_embd,
                )?;
                op::add(&s.x, &s.xb2, &s.x, n_embd)?;
            });

            // ── FFN norm (f32 for the router, quantized for the experts) ──
            phase!(router, {
                op::rms_norm_quant(&s.x, &l.ffn_norm, &s.xb, &s.aq, &s.ad, &s.inv, n_embd, eps)?;
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

            phase!(experts, {
                // Gate and up in one launch, for the same reason as q/k/v.
                // nsub=2 measured: 44.5 -> 31.3 us per matrix, 232 -> 330 GB/s.
                op::matvec_grouped_x2_splitk(
                    &l.ffn_gate_exps,
                    &l.ffn_up_exps,
                    &s.expert_ids,
                    &s.aq,
                    &s.ad,
                    &s.e_gate,
                    &s.e_up,
                    n_embd / 256,
                    n_ffe,
                    n_used,
                    nsub_gu,
                )?;
                // SiLU on the GATE only, then elementwise with up.
                op::silu_mul(&s.e_gate, &s.e_up, &s.e_act, n_used * n_ffe)?;
                // Each expert's down input is its OWN activation, hence the
                // per-activation grouped kernel.
                op::quantize(&s.e_act, &s.aq, &s.ad, (n_used * n_ffe) / 256)?;
                // blocks is only 3 here, so nsub=2 is the most the shape allows.
                // Measured 41.8 -> 35.0 us, 247 -> 295 GB/s.
                op::matvec_grouped_per_act_splitk(
                    &l.ffn_down_exps,
                    &s.expert_ids,
                    &s.aq,
                    &s.ad,
                    &s.e_down,
                    n_ffe / 256,
                    n_embd,
                    n_used,
                    nsub_dn,
                )?;
            });

            phase!(moe_tail, {
                op::moe_combine_add(&s.e_down, &s.expert_w, &s.x, &s.x, n_used, n_embd)?;
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
        Ok(())
    }

    /// Record the token as a HIP graph, so later tokens submit as one operation.
    ///
    /// CAPTURE DOES NOT EXECUTE. Nothing runs between begin and end; the graph
    /// is a recording of the launches. So this is called with `meta` already set
    /// to a real (token, pos), and the first replay produces the first result.
    ///
    /// Profiling and capture are mutually exclusive: profiling synchronises
    /// inside the token, which is illegal during capture and would be the wrong
    /// measurement anyway.
    fn capture(&mut self) -> Result<()> {
        if self.profile.is_some() {
            return Err(Error::malformed(
                "graph capture",
                "cannot capture while profiling: profiling synchronises mid-token",
            ));
        }
        llamd_rocm::graph::begin()?;
        // If queueing fails mid-capture the stream is left capturing, which
        // poisons every later launch. End it before propagating.
        if let Err(e) = self.queue_token() {
            let _ = llamd_rocm::graph::end();
            llamd_rocm::graph::destroy();
            return Err(e);
        }
        llamd_rocm::graph::end()
    }

    /// One token: set the metadata, run it, return the argmax.
    ///
    /// Uses the captured graph when there is one — which submits ~900 kernels
    /// as a single operation instead of ~900 driver calls — and falls back to
    /// queueing them individually when profiling or before capture.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<u32> {
        if pos >= self.max_kv {
            return Err(Error::malformed(
                "device forward",
                format!("position {pos} beyond the {} allocated", self.max_kv),
            ));
        }
        // The only per-token host work: eight bytes.
        self.s.meta.upload_i32(&[token as i32, pos as i32])?;

        if self.use_graph && self.profile.is_none() {
            use std::sync::atomic::Ordering::SeqCst;
            // A graph belonging to a different model points at buffers this one
            // does not own — possibly freed. Discard rather than replay.
            if !llamd_rocm::graph::exists() || GRAPH_OWNER.load(SeqCst) != self.id {
                llamd_rocm::graph::destroy();
                self.capture()?;
                GRAPH_OWNER.store(self.id, SeqCst);
            }
            llamd_rocm::graph::replay()?;
        } else {
            self.queue_token()?;
        }

        // THE ONE SYNCHRONISE.
        llamd_rocm::sync()?;

        let mut got = [0i32; 1];
        self.s.out_token.download_i32(&mut got)?;
        if got[0] < 0 || got[0] as usize >= self.hp.n_vocab {
            return Err(Error::other(format!("argmax returned {}", got[0])));
        }
        Ok(got[0] as u32)
    }

    /// Decode `nseq` INDEPENDENT sequences in one pass, returning one token each.
    ///
    /// WHY THIS IS THE ONLY REMAINING LEVER. At batch 1 a token reads 2.485 GB
    /// of weights at ~260 GB/s cold, and three attempts to raise that bandwidth
    /// were rejected by whole-token measurement (HIP graphs, split-k, planar
    /// Q6_K — benchmarks.md §8-§10). Batching attacks the bytes instead: the Q,
    /// K, V and O projections and the output projection are the SAME matrices
    /// for every sequence, so one read serves `nseq` tokens.
    ///
    /// WHAT DOES NOT DIVIDE. The routed experts. Each sequence picks its own 8
    /// of 128 and at batch 2 only ~3% coincide, so 1486 of the 2485 MB is paid
    /// per sequence regardless. Per-token traffic therefore goes to 1986 MB at
    /// batch 2 (1.25x) and 1736 at batch 4 (1.43x) — not 1/2 and 1/4.
    ///
    /// INDEPENDENT is load-bearing. Token N+1 of a sequence depends on token N,
    /// so these must be separate streams or speculative candidates; this raises
    /// aggregate throughput, not single-stream latency.
    pub fn forward_batch(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<u32>> {
        let nseq = tokens.len();
        if nseq == 0 || nseq != positions.len() {
            return Err(Error::malformed(
                "batch forward",
                format!("{nseq} tokens, {} positions", positions.len()),
            ));
        }
        if nseq > self.max_seq {
            return Err(Error::malformed(
                "batch forward",
                format!("{nseq} sequences exceeds the {} allocated", self.max_seq),
            ));
        }
        for (i, &p) in positions.iter().enumerate() {
            if p >= self.max_kv {
                return Err(Error::malformed(
                    "batch forward",
                    format!(
                        "sequence {i} position {p} beyond the {} allocated",
                        self.max_kv
                    ),
                ));
            }
        }

        // meta is [token0, pos0, token1, pos1, ...] so a kernel reads its own
        // sequence's pair with one index.
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

        bop::embed(&self.tok_embd, &s.meta, &s.x, self.embd_blocks, nseq)?;

        for il in 0..hp.n_layer {
            let l = &self.layers[il];

            // One norm and one quantization serve Q, K and V for every sequence.
            bop::rms_norm_quant(
                &s.x,
                &l.attn_norm,
                &s.xb,
                &s.aq,
                &s.ad,
                &s.inv,
                n_embd,
                nseq,
                eps,
            )?;
            // THE SHARING: three matrices, nseq sequences, each weight byte read
            // once and used nseq times.
            bop::matvec_x3(
                (&l.attn_q, &l.attn_k, &l.attn_v),
                &s.aq,
                &s.ad,
                (&s.q, &s.k, &s.v),
                n_embd / 256,
                (n_head * hd, kv_dim, kv_dim),
                nseq,
            )?;

            bop::norm_rope(
                &s.q,
                l.attn_q_norm.as_ref(),
                &s.meta,
                n_head,
                hd,
                nseq,
                fb,
                eps,
            )?;
            bop::k_norm_rope_store(
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
            bop::attention(
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

            bop::quantize(&s.attn, &s.aq, &s.ad, ((n_head * hd) / 256) * nseq)?;
            bop::matvec(
                &l.attn_out,
                &s.aq,
                &s.ad,
                &s.xb2,
                (n_head * hd) / 256,
                n_embd,
                nseq,
            )?;
            bop::add(&s.x, &s.xb2, &s.x, n_embd * nseq)?;

            bop::rms_norm_quant(
                &s.x,
                &l.ffn_norm,
                &s.xb,
                &s.aq,
                &s.ad,
                &s.inv,
                n_embd,
                nseq,
                eps,
            )?;
            bop::matvec_f32(
                &l.ffn_gate_inp,
                &s.xb,
                &s.router_logits,
                n_embd,
                n_expert,
                nseq,
            )?;
            bop::router(
                &s.router_logits,
                n_expert,
                n_used,
                &s.expert_ids,
                &s.expert_w,
                &s.router_probs,
                nseq,
            )?;

            // No weight sharing here — each sequence routes independently — but
            // still one launch, which the small expert matrices need for
            // occupancy.
            bop::grouped_x2(
                &l.ffn_gate_exps,
                &l.ffn_up_exps,
                &s.expert_ids,
                &s.aq,
                &s.ad,
                &s.e_gate,
                &s.e_up,
                n_embd / 256,
                n_ffe,
                n_used,
                nseq,
            )?;
            bop::silu_mul(&s.e_gate, &s.e_up, &s.e_act, n_used * n_ffe * nseq)?;
            bop::quantize(&s.e_act, &s.aq, &s.ad, ((n_used * n_ffe) / 256) * nseq)?;
            bop::grouped_pa(
                &l.ffn_down_exps,
                &s.expert_ids,
                &s.aq,
                &s.ad,
                &s.e_down,
                n_ffe / 256,
                n_embd,
                n_used,
                nseq,
            )?;
            bop::moe_combine_add(&s.e_down, &s.expert_w, &s.x, &s.x, n_used, n_embd, nseq)?;
        }

        bop::rms_norm_mul(&s.x, &self.output_norm, &s.xb, n_embd, nseq, eps)?;
        bop::quantize(&s.xb, &s.aq, &s.ad, (n_embd / 256) * nseq)?;
        bop::matvec(
            &self.output,
            &s.aq,
            &s.ad,
            &s.logits,
            n_embd / 256,
            hp.n_vocab,
            nseq,
        )?;
        bop::argmax(
            &s.logits,
            hp.n_vocab,
            &s.amax_v,
            &s.amax_i,
            &s.out_token,
            ARGMAX_BLOCKS,
            nseq,
        )?;

        // ONE synchronise for the whole batch.
        llamd_rocm::sync()?;

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

    /// The logits for the last token, copied back. For comparing against the
    /// CPU reference; generation does not need them.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.hp.n_vocab];
        self.s.logits.download_f32(&mut out)?;
        Ok(out)
    }
}

impl Drop for DeviceModel {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::SeqCst;
        // The graph references this model's buffers, which are about to be
        // freed by the DevBuf drops below. Leaving it instantiated would let a
        // later model replay it against freed memory.
        if GRAPH_OWNER.load(SeqCst) == self.id {
            llamd_rocm::graph::destroy();
            GRAPH_OWNER.store(0, SeqCst);
        }
    }
}
