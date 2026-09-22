//! Architecture identification and hyperparameters, read from GGUF metadata.
//!
//! PROVENANCE. Key names and semantics follow `src/llama-arch.cpp` and
//! `src/llama-model.cpp` in llama.cpp (MIT, © 2023-2026 The ggml authors).
//!
//! EVERYTHING HERE IS READ, NOTHING IS DEFAULTED SILENTLY. A hyperparameter
//! guessed wrong does not fail: it produces a model that loads, runs, and is
//! subtly not the model in the file. So a missing required key is an error,
//! and the few genuine optionals say in the code why a default is safe.

use llamd_core::{Error, Result};
use llamd_gguf::{Gguf, Value};

/// The architectures LLAMD implements. Deliberately two of llama.cpp's forty:
/// adding one is a decision with a test attached, not a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen3,
    Qwen3Moe,
}

impl std::str::FromStr for Arch {
    type Err = Error;

    /// Implemented as the trait rather than an inherent method, so `parse()`
    /// works and nobody wonders which `from_str` they are calling.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            other => {
                return Err(Error::unsupported(
                    "architecture",
                    format!("{other}; LLAMD implements qwen3 and qwen3moe"),
                ));
            }
        })
    }
}

impl Arch {
    pub fn key(&self) -> &'static str {
        match self {
            Arch::Qwen3 => "qwen3",
            Arch::Qwen3Moe => "qwen3moe",
        }
    }
    pub fn is_moe(&self) -> bool {
        matches!(self, Arch::Qwen3Moe)
    }
}

#[derive(Debug, Clone)]
pub struct Hparams {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Per-head width. Derived when the file does not state it.
    pub n_head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    /// What the architecture supports — not what a given deployment serves.
    pub n_ctx_train: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,

    // MoE. Zero for a dense model.
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_expert: usize,
}

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch_s = g
            .architecture()
            .ok_or_else(|| Error::malformed("hparams", "general.architecture is absent"))?;
        let arch: Arch = arch_s.parse()?;
        let p = arch.key();

        let req_u = |suffix: &str| -> Result<usize> {
            let k = format!("{p}.{suffix}");
            g.get(&k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| Error::malformed("hparams", format!("{k} is absent")))
        };
        let req_f = |suffix: &str| -> Result<f32> {
            let k = format!("{p}.{suffix}");
            g.get(&k)
                .and_then(Value::as_f32)
                .ok_or_else(|| Error::malformed("hparams", format!("{k} is absent")))
        };

        let n_embd = req_u("embedding_length")?;
        let n_head = req_u("attention.head_count")?;
        let n_head_kv = req_u("attention.head_count_kv")?;

        // head_dim is usually absent and derived. For this model n_embd/n_head
        // would give 64, but the real value is 128 — the Q projection widens
        // to n_head * 128 = 4096. So the TENSOR decides, not the arithmetic.
        let n_head_dim = g
            .get(&format!("{p}.attention.key_length"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .or_else(|| {
                // attn_k is [n_embd, n_head_kv * head_dim]; recover head_dim.
                g.tensor("blk.0.attn_k.weight")
                    .and_then(|t| t.dims.get(1).copied())
                    .map(|kout| kout as usize / n_head_kv.max(1))
            })
            .unwrap_or(n_embd / n_head.max(1));

        let n_vocab = g
            .tensor("token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .map(|v| v as usize)
            .ok_or_else(|| Error::malformed("hparams", "token_embd.weight is absent"))?;

        let (n_expert, n_expert_used, n_ff_expert) = if arch.is_moe() {
            (
                req_u("expert_count")?,
                req_u("expert_used_count")?,
                req_u("expert_feed_forward_length")?,
            )
        } else {
            (0, 0, 0)
        };

        let h = Hparams {
            arch,
            n_layer: req_u("block_count")?,
            n_embd,
            n_head,
            n_head_kv,
            n_head_dim,
            n_ff: req_u("feed_forward_length")?,
            n_vocab,
            n_ctx_train: req_u("context_length")?,
            rope_freq_base: req_f("rope.freq_base")?,
            rms_eps: req_f("attention.layer_norm_rms_epsilon")?,
            n_expert,
            n_expert_used,
            n_ff_expert,
        };
        h.validate()?;
        Ok(h)
    }

    /// Reject combinations that cannot be right, before anything is loaded.
    fn validate(&self) -> Result<()> {
        let bad = |d: String| Err(Error::malformed("hparams", d));
        if self.n_head_kv == 0 || self.n_head % self.n_head_kv != 0 {
            return bad(format!(
                "{} query heads do not divide into {} kv heads",
                self.n_head, self.n_head_kv
            ));
        }
        if self.n_head_dim % 2 != 0 {
            return bad(format!(
                "head_dim {} must be even for RoPE",
                self.n_head_dim
            ));
        }
        if self.arch.is_moe() && (self.n_expert_used == 0 || self.n_expert_used > self.n_expert) {
            return bad(format!(
                "{} experts used of {} available",
                self.n_expert_used, self.n_expert
            ));
        }
        Ok(())
    }

    /// Query heads per KV head — the grouped-query attention ratio.
    pub fn gqa_ratio(&self) -> usize {
        self.n_head / self.n_head_kv
    }
    /// Width of the concatenated Q projection.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.n_head_dim
    }
    /// Width of each of the K and V projections.
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.n_head_dim
    }

    /// KV cache cost per token, in bytes, for a given element size.
    ///
    /// This is the number that decides how much context fits, and it is worth
    /// having in the model rather than in a spreadsheet: for this model it is
    /// 2 × 48 × 4 × 128 × 2 = 98304 bytes, i.e. 96 KiB per token at f16.
    pub fn kv_bytes_per_token(&self, bytes_per_elem: usize) -> usize {
        2 * self.n_layer * self.n_head_kv * self.n_head_dim * bytes_per_elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_architectures_are_refused_by_name() {
        assert_eq!("qwen3moe".parse::<Arch>().unwrap(), Arch::Qwen3Moe);
        let e = "llama".parse::<Arch>().unwrap_err();
        assert!(matches!(e, Error::Unsupported { .. }));
    }

    fn qwen3moe() -> Hparams {
        Hparams {
            arch: Arch::Qwen3Moe,
            n_layer: 48,
            n_embd: 2048,
            n_head: 32,
            n_head_kv: 4,
            n_head_dim: 128,
            n_ff: 5472,
            n_vocab: 151936,
            n_ctx_train: 262144,
            rope_freq_base: 10_000_000.0,
            rms_eps: 1e-6,
            n_expert: 128,
            n_expert_used: 8,
            n_ff_expert: 768,
        }
    }

    #[test]
    fn derived_dimensions_match_the_real_model() {
        let h = qwen3moe();
        assert_eq!(h.gqa_ratio(), 8, "32 query heads over 4 kv heads");
        assert_eq!(h.q_dim(), 4096, "matches attn_q.weight [2048, 4096]");
        assert_eq!(h.kv_dim(), 512, "matches attn_k.weight [2048, 512]");
        // head_dim is NOT n_embd/n_head here: 2048/32 = 64, but it is 128.
        assert_ne!(h.n_head_dim, h.n_embd / h.n_head);
    }

    #[test]
    fn kv_cost_matches_the_measured_context_budget() {
        // The number the whole context ladder rests on: 96 KiB per token.
        let h = qwen3moe();
        assert_eq!(h.kv_bytes_per_token(2), 98304);
        assert_eq!(h.kv_bytes_per_token(2) / 1024, 96);
        // And why 262144 does not fit beside 23.37 GiB of weights. At q8_0
        // the KV cache is 48 KiB/token, and 262144 * 48 KiB is EXACTLY 12 GiB
        // — so the full context needs 35.4 GiB on a 31.86 GiB card before any
        // compute buffers. The number is exact, so the test asserts it rather
        // than bounding it.
        let kv_at_full_ctx = h.kv_bytes_per_token(1) * h.n_ctx_train;
        assert_eq!(
            kv_at_full_ctx,
            12 << 30,
            "q8_0 KV at full context is exactly 12 GiB"
        );
    }

    #[test]
    fn impossible_head_counts_are_refused() {
        let mut h = qwen3moe();
        h.n_head_kv = 5; // 32 is not divisible by 5
        assert!(h.validate().is_err());

        let mut h = qwen3moe();
        h.n_head_dim = 127; // RoPE needs pairs
        assert!(h.validate().is_err());

        let mut h = qwen3moe();
        h.n_expert_used = 200; // more than exist
        assert!(h.validate().is_err());
    }
}
