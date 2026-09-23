//! Model loading: resolving tensor names and fetching weights on demand.
//!
//! LAZY BY DESIGN. This model is 23.37 GiB quantized and would be ~122 GiB
//! dequantized to f32 — more than the machine has. So nothing is materialised
//! until it is asked for, and the GGUF stays memory-mapped. A loader that
//! eagerly converted everything would work on a toy model and fail on the only
//! model that matters.
//!
//! PROVENANCE. Tensor naming follows `src/llama-arch.cpp` in llama.cpp
//! (MIT, © 2023-2026 The ggml authors).

use llamd_core::{Error, Result};
use llamd_gguf::{Gguf, TensorInfo};
use llamd_tensor::{Shape, Tensor};

use crate::hparams::{Arch, Hparams};

/// The tensors one transformer block needs, by name.
///
/// Names are built once and checked against the file at load, so a missing
/// tensor is reported when the model is opened rather than part-way through
/// the first forward pass.
#[derive(Debug, Clone)]
pub struct LayerNames {
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_q_norm: Option<String>,
    pub attn_k_norm: Option<String>,
    pub attn_out: String,
    pub ffn_norm: String,
    // Dense FFN
    pub ffn_gate: Option<String>,
    pub ffn_up: Option<String>,
    pub ffn_down: Option<String>,
    // MoE
    pub ffn_gate_inp: Option<String>,
    pub ffn_gate_exps: Option<String>,
    pub ffn_up_exps: Option<String>,
    pub ffn_down_exps: Option<String>,
}

impl LayerNames {
    fn new(il: usize, arch: Arch, g: &Gguf) -> Self {
        let b = |s: &str| format!("blk.{il}.{s}");
        let opt = |s: String| g.tensor(&s).map(|_| s);
        let moe = arch.is_moe();
        LayerNames {
            attn_norm: b("attn_norm.weight"),
            attn_q: b("attn_q.weight"),
            attn_k: b("attn_k.weight"),
            attn_v: b("attn_v.weight"),
            // Qwen3 normalises Q and K per head before RoPE. Optional because
            // Qwen2 does not, and presence is the honest way to tell.
            attn_q_norm: opt(b("attn_q_norm.weight")),
            attn_k_norm: opt(b("attn_k_norm.weight")),
            attn_out: b("attn_output.weight"),
            ffn_norm: b("ffn_norm.weight"),
            ffn_gate: (!moe).then(|| b("ffn_gate.weight")),
            ffn_up: (!moe).then(|| b("ffn_up.weight")),
            ffn_down: (!moe).then(|| b("ffn_down.weight")),
            ffn_gate_inp: moe.then(|| b("ffn_gate_inp.weight")),
            ffn_gate_exps: moe.then(|| b("ffn_gate_exps.weight")),
            ffn_up_exps: moe.then(|| b("ffn_up_exps.weight")),
            ffn_down_exps: moe.then(|| b("ffn_down_exps.weight")),
        }
    }

    fn required(&self) -> Vec<&str> {
        let mut v = vec![
            self.attn_norm.as_str(),
            self.attn_q.as_str(),
            self.attn_k.as_str(),
            self.attn_v.as_str(),
            self.attn_out.as_str(),
            self.ffn_norm.as_str(),
        ];
        v.extend(
            [
                &self.ffn_gate,
                &self.ffn_up,
                &self.ffn_down,
                &self.ffn_gate_inp,
                &self.ffn_gate_exps,
                &self.ffn_up_exps,
                &self.ffn_down_exps,
            ]
            .into_iter()
            .flatten()
            .map(String::as_str),
        );
        v
    }
}

pub struct Model {
    gguf: Gguf,
    pub hparams: Hparams,
    pub layers: Vec<LayerNames>,
    pub tok_embd: String,
    pub output_norm: String,
    /// Absent when the model ties output to the embedding table.
    pub output: Option<String>,
}

impl Model {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        let hparams = Hparams::from_gguf(&gguf)?;
        let layers: Vec<LayerNames> = (0..hparams.n_layer)
            .map(|il| LayerNames::new(il, hparams.arch, &gguf))
            .collect();

        let m = Model {
            tok_embd: "token_embd.weight".into(),
            output_norm: "output_norm.weight".into(),
            // Some models tie the output projection to the embeddings. Saying
            // so explicitly beats discovering it as a missing-tensor error.
            output: gguf
                .tensor("output.weight")
                .map(|_| "output.weight".to_string()),
            gguf,
            hparams,
            layers,
        };
        m.verify_present()?;
        Ok(m)
    }

    /// Check every named tensor exists BEFORE inference starts.
    ///
    /// A missing tensor discovered at layer 31 of a forward pass is a much
    /// worse error message than one reported at load.
    fn verify_present(&self) -> Result<()> {
        let mut missing = Vec::new();
        for n in [self.tok_embd.as_str(), self.output_norm.as_str()] {
            if self.gguf.tensor(n).is_none() {
                missing.push(n.to_string());
            }
        }
        for l in &self.layers {
            for n in l.required() {
                if self.gguf.tensor(n).is_none() {
                    missing.push(n.to_string());
                }
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::malformed(
                "model",
                format!(
                    "{} tensors missing, first: {:?}",
                    missing.len(),
                    &missing[..missing.len().min(3)]
                ),
            ))
        }
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.gguf
            .tensor(name)
            .ok_or_else(|| Error::malformed("model", format!("tensor {name} is absent")))
    }

    /// Dequantize a whole tensor to f32. Costly for large weights, which is
    /// why the expert accessor below exists.
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        let info = self.info(name)?;
        let data = self.gguf.dequantize(info)?;
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        Tensor::new(Shape::new(&dims)?, data)
    }

    /// One expert's slice of a 3D expert tensor.
    ///
    /// Expert weights are stored as `[d0, d1, n_expert]` with the expert index
    /// on the SLOWEST axis, so one expert is a contiguous block. Materialising
    /// the whole tensor to reach one expert would cost 128× the memory and is
    /// the difference between routing being feasible and not.
    pub fn expert(&self, name: &str, expert: usize) -> Result<Tensor> {
        let info = self.info(name)?;
        if info.dims.len() != 3 {
            return Err(Error::malformed(
                "expert tensor",
                format!("{name} has {} dims, expected 3", info.dims.len()),
            ));
        }
        let (d0, d1, ne) = (
            info.dims[0] as usize,
            info.dims[1] as usize,
            info.dims[2] as usize,
        );
        if expert >= ne {
            return Err(Error::malformed(
                "expert tensor",
                format!("expert {expert} of {ne} in {name}"),
            ));
        }
        let per = d0 * d1;
        let ty = llamd_quant::QuantType::from_ggml(info.ggml_type)?;
        // Slice in the QUANTIZED domain: the block size must divide the
        // per-expert element count, or an expert would start mid-block.
        let block = ty.block_elems();
        if per % block != 0 {
            return Err(Error::unsupported(
                "expert slicing",
                format!("{per} elements per expert is not a multiple of the {block}-element block"),
            ));
        }
        let bytes_per = ty.bytes_for(per)?;
        let all = self.gguf.tensor_data(info)?;
        let start = expert * bytes_per;
        let raw = all.get(start..start + bytes_per).ok_or_else(|| {
            Error::malformed("expert tensor", format!("{name} slice out of range"))
        })?;
        let mut out = Vec::with_capacity(per);
        llamd_quant::dequantize(ty, raw, per, &mut out)?;
        Tensor::new(Shape::new(&[d0, d1])?, out)
    }

    /// Matrix-vector product against a QUANTIZED weight, without
    /// dequantizing it.
    ///
    /// This is the path that matches llama.cpp. The f32 reference
    /// (`tensor()` then `Backend::matmul`) performs different arithmetic, and
    /// while the difference is tiny it is larger than the gap between
    /// near-tied MoE experts — so it changes which expert runs, and therefore
    /// the emitted token. See docs/porting-matrix.md.
    ///
    /// Falls back to dequantizing for types without a quantized kernel, so a
    /// model mixing F32 and Q6_K tensors still runs; the fallback is exact for
    /// F32 and so costs nothing in fidelity there.
    pub fn matvec_quant(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let ty = llamd_quant::QuantType::from_ggml(info.ggml_type)?;
        let k = info.dims[0] as usize;
        let rows = info.elems() as usize / k;
        if x.len() != k {
            return Err(Error::Shape {
                expected: vec![k],
                got: vec![x.len()],
            });
        }

        if ty != llamd_quant::QuantType::Q6K {
            // F32 and F16 weights are exact either way; dequantizing them
            // introduces nothing. Only the k-quants have a separate path.
            let w = self.tensor(name)?;
            return Ok((0..rows)
                .map(|r| {
                    let row = &w.data()[r * k..(r + 1) * k];
                    row.iter().zip(x).map(|(a, b)| a * b).sum()
                })
                .collect());
        }

        let a = llamd_quant::quantize_q8_k(x);
        let bytes = ty.bytes_for(k)?;
        let data = self.gguf.tensor_data(info)?;
        (0..rows)
            .map(|r| {
                let row = data
                    .get(r * bytes..(r + 1) * bytes)
                    .ok_or_else(|| Error::malformed("matvec", format!("{name} row {r}")))?;
                Ok(llamd_quant::dot_q6k_q8k(row, &a, k))
            })
            .collect()
    }

    /// The same, for one expert of a 3D expert tensor.
    pub fn matvec_expert_quant(&self, name: &str, expert: usize, x: &[f32]) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        let ty = llamd_quant::QuantType::from_ggml(info.ggml_type)?;
        if info.dims.len() != 3 || ty != llamd_quant::QuantType::Q6K {
            // Keep the general path available rather than refusing: a model
            // with F32 experts is unusual but not wrong.
            let w = self.expert(name, expert)?;
            let k = w.row_len();
            return Ok((0..w.rows())
                .map(|r| {
                    w.data()[r * k..(r + 1) * k]
                        .iter()
                        .zip(x)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect());
        }
        let (k, rows) = (info.dims[0] as usize, info.dims[1] as usize);
        if x.len() != k {
            return Err(Error::Shape {
                expected: vec![k],
                got: vec![x.len()],
            });
        }
        let per_expert = ty.bytes_for(k * rows)?;
        let row_bytes = ty.bytes_for(k)?;
        let data = self.gguf.tensor_data(info)?;
        let base = expert * per_expert;
        let a = llamd_quant::quantize_q8_k(x);
        (0..rows)
            .map(|r| {
                let o = base + r * row_bytes;
                let row = data
                    .get(o..o + row_bytes)
                    .ok_or_else(|| Error::malformed("matvec", format!("{name} e{expert} r{r}")))?;
                Ok(llamd_quant::dot_q6k_q8k(row, &a, k))
            })
            .collect()
    }

    /// Total parameters, counted from the tensor descriptors.
    pub fn n_params(&self) -> u64 {
        self.gguf.tensors.iter().map(|t| t.elems()).sum()
    }
}
