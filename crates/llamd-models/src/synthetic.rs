//! A tiny, structurally complete `qwen3moe` GGUF, for tests that must not
//! need a 23 GiB file.
//!
//! STRUCTURE, NOT NUMBERS. Every tensor `Model::open` requires is present with
//! the right name and shape, and expert tensors are Q6_K with the expert on the
//! slowest axis, exactly as in a real file. The bytes are deterministic noise:
//! enough to tell one expert from another and to catch a wrong extent, but
//! not a model anyone should run. Numerical validation stays with the
//! llama.cpp oracle tests.

use llamd_core::Result;
use llamd_gguf::{GgufWriter, Value};
use llamd_quant::{Q6K_BLOCK_BYTES, QK_K};
use std::path::PathBuf;

use crate::expert_index::Proj;

const F32: u32 = 0;
const Q6K: u32 = 14;

#[derive(Debug, Clone)]
pub struct SyntheticMoe {
    pub n_layer: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Must be a multiple of 256 (the Q6_K block).
    pub n_embd: usize,
    /// Must be a multiple of 256.
    pub n_ff_expert: usize,
    pub n_vocab: usize,
    pub alignment: u64,
    /// Declare a different `expert_count` than the tensors carry — for
    /// negative tests.
    pub declared_experts: Option<usize>,
}

impl Default for SyntheticMoe {
    fn default() -> Self {
        SyntheticMoe {
            n_layer: 2,
            n_expert: 4,
            n_expert_used: 2,
            n_embd: 256,
            n_ff_expert: 256,
            n_vocab: 8,
            alignment: 32,
            declared_experts: None,
        }
    }
}

/// A model file in the temp dir that removes itself.
pub struct TempModel(pub PathBuf);

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl SyntheticMoe {
    const N_HEAD: usize = 2;
    const N_HEAD_KV: usize = 1;
    const HEAD_DIM: usize = 128;

    /// The exact bytes written for one expert of one projection.
    pub fn expert_bytes(&self, layer: usize, proj: Proj, expert: usize) -> Vec<u8> {
        let blocks = self.n_embd * self.n_ff_expert / QK_K;
        let seed = ((layer * 3 + proj as usize) * 1000 + expert) as u64;
        noise(seed, blocks * Q6K_BLOCK_BYTES)
    }

    pub fn write(&self, tag: &str) -> Result<TempModel> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "llamd-synth-{tag}-{}-{}.gguf",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        self.writer().write(&path)?;
        Ok(TempModel(path))
    }

    pub fn writer(&self) -> GgufWriter {
        let (e, ff) = (self.n_embd as u64, self.n_ff_expert as u64);
        let q = (Self::N_HEAD * Self::HEAD_DIM) as u64;
        let kv = (Self::N_HEAD_KV * Self::HEAD_DIM) as u64;
        let ne = self.n_expert as u64;

        let mut w = GgufWriter::new();
        w.alignment(self.alignment);
        let a = "qwen3moe";
        let u = |v: usize| Value::U32(v as u32);
        w.kv("general.architecture", Value::String(a.into()));
        w.kv(&format!("{a}.block_count"), u(self.n_layer));
        w.kv(&format!("{a}.context_length"), u(4096));
        w.kv(&format!("{a}.embedding_length"), u(self.n_embd));
        w.kv(&format!("{a}.feed_forward_length"), u(self.n_ff_expert * 4));
        w.kv(&format!("{a}.attention.head_count"), u(Self::N_HEAD));
        w.kv(&format!("{a}.attention.head_count_kv"), u(Self::N_HEAD_KV));
        w.kv(&format!("{a}.attention.key_length"), u(Self::HEAD_DIM));
        w.kv(&format!("{a}.rope.freq_base"), Value::F32(1_000_000.0));
        w.kv(
            &format!("{a}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-6),
        );
        w.kv(
            &format!("{a}.expert_count"),
            u(self.declared_experts.unwrap_or(self.n_expert)),
        );
        w.kv(&format!("{a}.expert_used_count"), u(self.n_expert_used));
        w.kv(
            &format!("{a}.expert_feed_forward_length"),
            u(self.n_ff_expert),
        );

        let f32t = |w: &mut GgufWriter, name: &str, dims: &[u64], seed: u64| {
            let n: u64 = dims.iter().product();
            let data: Vec<u8> = noise(seed, n as usize / 4 * 4 + 4)
                .chunks(4)
                .take(n as usize)
                .flat_map(|c| {
                    // Small finite values, never NaN.
                    let x = (u32::from_le_bytes([c[0], c[1], c[2], c[3]]) % 2001) as f32;
                    ((x - 1000.0) / 1000.0).to_le_bytes()
                })
                .collect();
            w.tensor(name, dims, F32, data);
        };

        f32t(&mut w, "token_embd.weight", &[e, self.n_vocab as u64], 1);
        for il in 0..self.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            let s0 = 100 + il as u64 * 100;
            f32t(&mut w, &b("attn_norm.weight"), &[e], s0);
            f32t(&mut w, &b("attn_q.weight"), &[e, q], s0 + 1);
            f32t(&mut w, &b("attn_k.weight"), &[e, kv], s0 + 2);
            f32t(&mut w, &b("attn_v.weight"), &[e, kv], s0 + 3);
            f32t(
                &mut w,
                &b("attn_q_norm.weight"),
                &[Self::HEAD_DIM as u64],
                s0 + 4,
            );
            f32t(
                &mut w,
                &b("attn_k_norm.weight"),
                &[Self::HEAD_DIM as u64],
                s0 + 5,
            );
            f32t(&mut w, &b("attn_output.weight"), &[q, e], s0 + 6);
            f32t(&mut w, &b("ffn_norm.weight"), &[e], s0 + 7);
            f32t(&mut w, &b("ffn_gate_inp.weight"), &[e, ne], s0 + 8);
            for proj in Proj::ALL {
                let dims = match proj {
                    Proj::Gate | Proj::Up => [e, ff, ne],
                    Proj::Down => [ff, e, ne],
                };
                let data: Vec<u8> = (0..self.n_expert)
                    .flat_map(|x| self.expert_bytes(il, proj, x))
                    .collect();
                w.tensor(&proj.tensor_name(il), &dims, Q6K, data);
            }
        }
        f32t(&mut w, "output_norm.weight", &[e], 2);
        f32t(&mut w, "output.weight", &[e, self.n_vocab as u64], 3);
        w
    }
}

/// Deterministic bytes (splitmix64). Not random-quality; just distinct.
fn noise(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n + 8);
    while out.len() < n {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    out.truncate(n);
    out
}
