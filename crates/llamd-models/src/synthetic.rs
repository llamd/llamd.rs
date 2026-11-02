//! A structurally complete `qwen3moe` GGUF, for tests that must not need a
//! real model file.
//!
//! STRUCTURE, NOT NUMBERS. Every tensor `Model::open` requires is present with
//! the right name and shape, and expert tensors carry the expert on the
//! slowest axis, exactly as in a real file. Dense bytes are deterministic
//! noise: enough to tell one expert from another and to catch a wrong extent,
//! but not a model anyone should run. Numerical validation stays with the
//! llama.cpp oracle tests.
//!
//! SPARSE MODE writes the full tensor table with no tensor data: the file has
//! the real model's size and layout but occupies a few MB of disk. Presets
//! such as [`SyntheticMoe::qwen3_coder_480b_q8_0`] use it to exercise the
//! out-of-core path at the TARGET's shape on a machine that cannot store it.

use llamd_core::Result;
use llamd_gguf::{GgufWriter, Value};
use llamd_quant::QuantType;
use std::path::PathBuf;

use crate::expert_index::Proj;

#[derive(Debug, Clone)]
pub struct SyntheticMoe {
    pub n_layer: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Must be a multiple of the expert/shared block size.
    pub n_embd: usize,
    /// Must be a multiple of the expert block size.
    pub n_ff_expert: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub expert_qtype: QuantType,
    /// Projections and embeddings. Norms and the router stay F32, as in
    /// real files.
    pub shared_qtype: QuantType,
    pub alignment: u64,
    /// No tensor data: holes of the right size.
    pub sparse: bool,
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
            n_head: 2,
            n_head_kv: 1,
            head_dim: 128,
            expert_qtype: QuantType::Q6K,
            shared_qtype: QuantType::F32,
            alignment: 32,
            sparse: false,
            declared_experts: None,
        }
    }
}

impl SyntheticMoe {
    /// The tensor table of Qwen3-Coder-480B-A35B-Instruct at Q8_0 — the
    /// probable "~508 GB" target — as a SPARSE file.
    ///
    /// Shapes are from the model's published config (62 layers, 160 experts,
    /// top-8, hidden 6144, expert FFN 2560, 96 query / 8 KV heads of 128,
    /// vocabulary 151936). Real Q8_0 GGUFs may keep a few small tensors in
    /// other formats; the expert tensors, which are 97% of the bytes, are
    /// what this reproduces.
    pub fn qwen3_coder_480b_q8_0() -> Self {
        SyntheticMoe {
            n_layer: 62,
            n_expert: 160,
            n_expert_used: 8,
            n_embd: 6144,
            n_ff_expert: 2560,
            n_vocab: 151936,
            n_head: 96,
            n_head_kv: 8,
            head_dim: 128,
            expert_qtype: QuantType::Q8_0,
            shared_qtype: QuantType::Q8_0,
            alignment: 32,
            sparse: true,
            declared_experts: None,
        }
    }

    /// The exact bytes written for one expert of one projection (dense mode).
    pub fn expert_bytes(&self, layer: usize, proj: Proj, expert: usize) -> Vec<u8> {
        let n = self
            .expert_qtype
            .bytes_for(self.n_embd * self.n_ff_expert)
            .expect("expert shape is block-aligned");
        let seed = ((layer * 3 + proj as usize) * 1000 + expert) as u64;
        noise(seed, n)
    }

    pub fn write(&self, tag: &str) -> Result<TempModel> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "llamd-synth-{tag}-{}-{}.gguf",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        self.writer()?.write(&path)?;
        Ok(TempModel(path))
    }

    pub fn writer(&self) -> Result<GgufWriter> {
        let (e, ff) = (self.n_embd as u64, self.n_ff_expert as u64);
        let q = (self.n_head * self.head_dim) as u64;
        let kv = (self.n_head_kv * self.head_dim) as u64;
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
        w.kv(&format!("{a}.attention.head_count"), u(self.n_head));
        w.kv(&format!("{a}.attention.head_count_kv"), u(self.n_head_kv));
        w.kv(&format!("{a}.attention.key_length"), u(self.head_dim));
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

        let f32q = QuantType::F32;
        let sh = self.shared_qtype;
        let mut seed = 0u64;
        let mut t = |w: &mut GgufWriter, name: &str, dims: &[u64], ty: QuantType| -> Result<()> {
            seed += 1;
            if self.sparse {
                w.tensor_sparse(name, dims, ty as u32)?;
            } else {
                let n: u64 = dims.iter().product();
                w.tensor(name, dims, ty as u32, dense(ty, n as usize, seed)?);
            }
            Ok(())
        };

        t(&mut w, "token_embd.weight", &[e, self.n_vocab as u64], sh)?;
        for il in 0..self.n_layer {
            let b = |s: &str| format!("blk.{il}.{s}");
            t(&mut w, &b("attn_norm.weight"), &[e], f32q)?;
            t(&mut w, &b("attn_q.weight"), &[e, q], sh)?;
            t(&mut w, &b("attn_k.weight"), &[e, kv], sh)?;
            t(&mut w, &b("attn_v.weight"), &[e, kv], sh)?;
            t(
                &mut w,
                &b("attn_q_norm.weight"),
                &[self.head_dim as u64],
                f32q,
            )?;
            t(
                &mut w,
                &b("attn_k_norm.weight"),
                &[self.head_dim as u64],
                f32q,
            )?;
            t(&mut w, &b("attn_output.weight"), &[q, e], sh)?;
            t(&mut w, &b("ffn_norm.weight"), &[e], f32q)?;
            t(&mut w, &b("ffn_gate_inp.weight"), &[e, ne], f32q)?;
            for proj in Proj::ALL {
                let dims = match proj {
                    Proj::Gate | Proj::Up => [e, ff, ne],
                    Proj::Down => [ff, e, ne],
                };
                if self.sparse {
                    w.tensor_sparse(&proj.tensor_name(il), &dims, self.expert_qtype as u32)?;
                } else {
                    let data: Vec<u8> = (0..self.n_expert)
                        .flat_map(|x| self.expert_bytes(il, proj, x))
                        .collect();
                    w.tensor(&proj.tensor_name(il), &dims, self.expert_qtype as u32, data);
                }
            }
        }
        t(&mut w, "output_norm.weight", &[e], f32q)?;
        t(&mut w, "output.weight", &[e, self.n_vocab as u64], sh)?;
        Ok(w)
    }
}

/// A model file in the temp dir that removes itself.
pub struct TempModel(pub PathBuf);

impl Drop for TempModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Dense filler of the right byte length. F32 gets small finite values (never
/// NaN); other formats get noise, which is fine for structure tests.
fn dense(ty: QuantType, n: usize, seed: u64) -> Result<Vec<u8>> {
    let len = ty.bytes_for(n)?;
    if ty != QuantType::F32 {
        return Ok(noise(seed, len));
    }
    Ok(noise(seed, n * 4)
        .chunks(4)
        .flat_map(|c| {
            let x = (u32::from_le_bytes([c[0], c[1], c[2], c[3]]) % 2001) as f32;
            ((x - 1000.0) / 1000.0).to_le_bytes()
        })
        .collect())
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
