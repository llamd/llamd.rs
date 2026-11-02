//! The expert index: where every routed expert's weights are in the file.
//!
//! WHY THIS EXISTS. A resident model can treat a layer's `ffn_*_exps` tensor
//! as one blob. A model larger than memory cannot: it must fetch the eight
//! experts the router chose, and nothing else. That needs, for every
//! (layer, expert, projection), the exact byte range, the block format and
//! the shape — known BEFORE the first token, checked against the
//! hyperparameters, and computable without touching tensor data.
//!
//! THE LAYOUT MAKES IT CHEAP. GGUF stores expert tensors as
//! `[cols, rows, n_expert]` with the expert on the SLOWEST axis, so one expert
//! is one contiguous run of `rows * cols` elements:
//!
//! ```text
//!   expert e of (layer, proj)  =  tensor.offset + e * per_expert   ..  + per_expert
//! ```
//!
//! No repacking is needed to address an expert; the index is arithmetic over
//! the tensor table. What the index does NOT promise is O_DIRECT alignment —
//! GGUF only aligns tensor starts to `general.alignment`, so
//! [`ExpertIndex::alignment_census`] reports what a reader will face.
//!
//! EVERYTHING IS CHECKED, because a wrong extent is not an error at runtime: it
//! is a different expert's weights, i.e. a different model that still emits
//! plausible text.

use llamd_core::{Error, Result};
use llamd_gguf::{Extent, Gguf};
use llamd_quant::QuantType;
use std::collections::BTreeMap;

use crate::hparams::Hparams;

/// One of an expert's three projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Proj {
    Gate,
    Up,
    Down,
}

impl Proj {
    pub const ALL: [Proj; 3] = [Proj::Gate, Proj::Up, Proj::Down];

    pub fn tensor_name(self, layer: usize) -> String {
        let s = match self {
            Proj::Gate => "ffn_gate_exps",
            Proj::Up => "ffn_up_exps",
            Proj::Down => "ffn_down_exps",
        };
        format!("blk.{layer}.{s}.weight")
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One `[cols, rows, n_expert]` expert tensor.
#[derive(Debug, Clone)]
pub struct ExpertTensor {
    pub name: String,
    pub qtype: QuantType,
    /// Input width — ggml `dims[0]`.
    pub cols: usize,
    /// Output rows of ONE expert — ggml `dims[1]`.
    pub rows: usize,
    /// The whole tensor, every expert.
    pub extent: Extent,
    /// Bytes of one expert.
    pub per_expert: u64,
}

impl ExpertTensor {
    pub fn expert(&self, expert: usize) -> Extent {
        Extent {
            offset: self.extent.offset + expert as u64 * self.per_expert,
            len: self.per_expert,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertIndex {
    pub n_layer: usize,
    pub n_expert: usize,
    /// `[layer * 3 + proj]`.
    tensors: Vec<ExpertTensor>,
}

impl ExpertIndex {
    /// Build and verify the index from the tensor table alone. Reads no
    /// tensor data, so it works on a file opened with `Gguf::open_unmapped`.
    pub fn build(g: &Gguf, h: &Hparams) -> Result<Self> {
        if !h.arch.is_moe() {
            return Err(Error::unsupported(
                "expert index",
                format!("{} is not a mixture-of-experts architecture", h.arch.key()),
            ));
        }
        let mut tensors = Vec::with_capacity(h.n_layer * 3);
        for layer in 0..h.n_layer {
            for proj in Proj::ALL {
                tensors.push(Self::one(g, h, layer, proj)?);
            }
        }
        check_disjoint(tensors.iter().map(|t| (t.name.as_str(), t.extent)))?;
        Ok(ExpertIndex {
            n_layer: h.n_layer,
            n_expert: h.n_expert,
            tensors,
        })
    }

    fn one(g: &Gguf, h: &Hparams, layer: usize, proj: Proj) -> Result<ExpertTensor> {
        let name = proj.tensor_name(layer);
        let bad = |d: String| Error::malformed("expert index", format!("{name}: {d}"));
        let info = g.tensor(&name).ok_or_else(|| bad("absent".into()))?;
        if info.dims.len() != 3 {
            return Err(bad(format!("{} dims, expected 3", info.dims.len())));
        }
        let (cols, rows, ne) = (
            info.dims[0] as usize,
            info.dims[1] as usize,
            info.dims[2] as usize,
        );
        if ne != h.n_expert {
            return Err(bad(format!("{ne} experts, hparams say {}", h.n_expert)));
        }
        // gate/up map n_embd -> n_ff_expert; down maps back.
        let (want_cols, want_rows) = match proj {
            Proj::Gate | Proj::Up => (h.n_embd, h.n_ff_expert),
            Proj::Down => (h.n_ff_expert, h.n_embd),
        };
        if (cols, rows) != (want_cols, want_rows) {
            return Err(bad(format!(
                "shape [{cols}, {rows}] per expert, expected [{want_cols}, {want_rows}]"
            )));
        }
        let qtype = QuantType::from_ggml(info.ggml_type)?;
        let per = cols * rows;
        // An expert must start on a block boundary, or slicing it would split
        // a block between two experts.
        if per % qtype.block_elems() != 0 {
            return Err(bad(format!(
                "{per} elements per expert is not a multiple of the {}-element {qtype:?} block",
                qtype.block_elems()
            )));
        }
        let per_expert = qtype.bytes_for(per)? as u64;
        let extent = g.tensor_extent(info)?;
        if extent.len != per_expert * ne as u64 {
            return Err(bad(format!(
                "{} bytes, expected {ne} x {per_expert}",
                extent.len
            )));
        }
        Ok(ExpertTensor {
            name,
            qtype,
            cols,
            rows,
            extent,
            per_expert,
        })
    }

    pub fn tensor(&self, layer: usize, proj: Proj) -> Result<&ExpertTensor> {
        if layer >= self.n_layer {
            return Err(Error::malformed(
                "expert index",
                format!("layer {layer} of {}", self.n_layer),
            ));
        }
        Ok(&self.tensors[layer * 3 + proj.index()])
    }

    /// Byte range of one projection of one expert.
    pub fn extent(&self, layer: usize, expert: usize, proj: Proj) -> Result<Extent> {
        if expert >= self.n_expert {
            return Err(Error::malformed(
                "expert index",
                format!("expert {expert} of {}", self.n_expert),
            ));
        }
        Ok(self.tensor(layer, proj)?.expert(expert))
    }

    /// Bytes to bring one expert of `layer` in: gate + up + down.
    pub fn expert_bytes(&self, layer: usize) -> Result<u64> {
        Ok(Proj::ALL
            .iter()
            .map(|&p| self.tensor(layer, p).map(|t| t.per_expert))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum())
    }

    /// The largest single expert in the model — the unit a staging slot must hold.
    pub fn max_expert_bytes(&self) -> u64 {
        (0..self.n_layer)
            .filter_map(|l| self.expert_bytes(l).ok())
            .max()
            .unwrap_or(0)
    }

    /// Every routed expert, every layer.
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.extent.len).sum()
    }

    /// How many expert extents fall into each alignment class (see
    /// [`Extent::alignment`]). Decides whether O_DIRECT reads can be issued
    /// as-is or must be widened.
    pub fn alignment_census(&self) -> BTreeMap<u64, usize> {
        let mut m = BTreeMap::new();
        for t in &self.tensors {
            for e in 0..self.n_expert {
                *m.entry(t.expert(e).alignment()).or_default() += 1;
            }
        }
        m
    }

    /// Quantization formats used by the expert tensors.
    pub fn qtypes(&self) -> BTreeMap<String, usize> {
        let mut m = BTreeMap::new();
        for t in &self.tensors {
            *m.entry(format!("{:?}", t.qtype)).or_default() += 1;
        }
        m
    }
}

/// No two tensors may share a byte. A hostile or corrupt file could point two
/// tensors at one range; reading "expert 3" would then silently return
/// another tensor's weights.
fn check_disjoint<'a>(it: impl Iterator<Item = (&'a str, Extent)>) -> Result<()> {
    let mut v: Vec<_> = it.collect();
    v.sort_by_key(|(_, e)| e.offset);
    for w in v.windows(2) {
        let ((na, a), (nb, b)) = (w[0], w[1]);
        if a.end() > b.offset {
            return Err(Error::malformed(
                "expert index",
                format!(
                    "{na} [{}..{}) overlaps {nb} [{}..{})",
                    a.offset,
                    a.end(),
                    b.offset,
                    b.end()
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{SyntheticMoe, TempModel};

    fn open_both(s: &SyntheticMoe) -> (TempModel, Gguf, Gguf) {
        let f = s.write("index").unwrap();
        let m = Gguf::open(&f.0).unwrap();
        let u = Gguf::open_unmapped(&f.0).unwrap();
        (f, m, u)
    }

    #[test]
    fn every_expert_extent_is_exactly_that_experts_bytes() {
        let s = SyntheticMoe::default();
        let (_f, m, u) = open_both(&s);
        let h = Hparams::from_gguf(&u).unwrap();
        let idx = ExpertIndex::build(&u, &h).unwrap();
        for layer in 0..s.n_layer {
            for proj in Proj::ALL {
                let t = idx.tensor(layer, proj).unwrap();
                let whole = m.tensor_data(m.tensor(&t.name).unwrap()).unwrap();
                let per = t.per_expert as usize;
                for e in 0..s.n_expert {
                    let got = u.read_extent(idx.extent(layer, e, proj).unwrap()).unwrap();
                    assert_eq!(got, &whole[e * per..(e + 1) * per], "{} e{e}", t.name);
                    // And the generator's own record of what it wrote.
                    assert_eq!(got, s.expert_bytes(layer, proj, e), "{} e{e}", t.name);
                }
            }
        }
    }

    #[test]
    fn mapped_and_unmapped_build_the_same_index() {
        let (_f, m, u) = open_both(&SyntheticMoe::default());
        let h = Hparams::from_gguf(&m).unwrap();
        let a = ExpertIndex::build(&m, &h).unwrap();
        let b = ExpertIndex::build(&u, &h).unwrap();
        for (x, y) in a.tensors.iter().zip(&b.tensors) {
            assert_eq!((x.extent, x.per_expert), (y.extent, y.per_expert));
        }
    }

    #[test]
    fn sizes_add_up() {
        let s = SyntheticMoe::default();
        let (_f, _m, u) = open_both(&s);
        let idx = ExpertIndex::build(&u, &Hparams::from_gguf(&u).unwrap()).unwrap();
        let per = (s.n_embd * s.n_ff_expert / 256 * llamd_quant::Q6K_BLOCK_BYTES) as u64;
        assert_eq!(idx.expert_bytes(0).unwrap(), 3 * per);
        assert_eq!(idx.max_expert_bytes(), 3 * per);
        assert_eq!(idx.total_bytes(), 3 * per * (s.n_expert * s.n_layer) as u64);
        let census: usize = idx.alignment_census().values().sum();
        assert_eq!(census, 3 * s.n_expert * s.n_layer);
    }

    #[test]
    fn out_of_range_requests_are_refused() {
        let s = SyntheticMoe::default();
        let (_f, _m, u) = open_both(&s);
        let idx = ExpertIndex::build(&u, &Hparams::from_gguf(&u).unwrap()).unwrap();
        assert!(idx.extent(s.n_layer, 0, Proj::Gate).is_err());
        assert!(idx.extent(0, s.n_expert, Proj::Up).is_err());
    }

    #[test]
    fn an_expert_count_that_disagrees_with_the_tensors_is_refused() {
        let s = SyntheticMoe {
            declared_experts: Some(8), // tensors carry 4
            ..SyntheticMoe::default()
        };
        let (_f, _m, u) = open_both(&s);
        let h = Hparams::from_gguf(&u).unwrap();
        let e = ExpertIndex::build(&u, &h).unwrap_err();
        assert!(format!("{e}").contains("4 experts, hparams say 8"), "{e}");
    }

    #[test]
    fn overlapping_extents_are_refused() {
        let a = Extent {
            offset: 0,
            len: 100,
        };
        let b = Extent {
            offset: 99,
            len: 10,
        };
        let c = Extent {
            offset: 100,
            len: 10,
        };
        assert!(check_disjoint([("a", a), ("b", b)].into_iter()).is_err());
        assert!(check_disjoint([("c", c), ("a", a)].into_iter()).is_ok());
    }
}
