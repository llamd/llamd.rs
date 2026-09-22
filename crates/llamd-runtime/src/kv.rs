//! The KV cache.
//!
//! WHY THIS IS A REDESIGN, NOT A PORT. llama.cpp has four cache
//! implementations — unified, iswa, recurrent, hybrid — totalling 7,882 lines,
//! because they accumulated as architectures needed them. LLAMD starts with
//! the one Qwen3MoE requires and adds a variant when a model actually needs
//! one. Porting all four first would be carrying complexity for models that
//! are out of scope.
//!
//! THE LAYOUT IS AN AMD-FIRST DECISION and is deliberately not inherited.
//! Here it is the simple one: per layer, keys and values stored as
//! `[n_head_kv * head_dim]` per position, contiguous in position order, so a
//! sequential read over a sequence is a linear scan. Whether that survives
//! contact with RDNA4's memory hierarchy is a question for the ROCm backend —
//! and the point of putting it behind an interface is that the answer can
//! change without the model code noticing.
//!
//! THE COST IS THE CONTEXT CEILING. For this model: 2 × 48 layers × 4 kv heads
//! × 128 dim × 2 bytes = 96 KiB per token at f16. That single number decides
//! that 262144 context does not fit on a 32 GiB card, so it is computed here
//! rather than remembered.

use llamd_core::{Error, Result};

/// Keys and values for every layer of one sequence.
#[derive(Debug)]
pub struct KvCache {
    n_layer: usize,
    /// Width of one position's K (and V) entry: `n_head_kv * head_dim`.
    kv_dim: usize,
    capacity: usize,
    /// Positions populated PER LAYER, in order from zero.
    ///
    /// Originally a single global length that advanced only when the last
    /// layer was written, on the theory that a half-written token must not be
    /// readable. That was wrong about the access pattern: a layer reads only
    /// its OWN keys and values, never another layer's, so there is no
    /// cross-layer state to protect — and the global rule made a single-layer
    /// forward pass read an empty cache.
    len: Vec<usize>,
    /// Per layer, `capacity * kv_dim` values.
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layer: usize, kv_dim: usize, capacity: usize) -> Result<Self> {
        if n_layer == 0 || kv_dim == 0 || capacity == 0 {
            return Err(Error::malformed(
                "kv cache",
                format!("{n_layer} layers, {kv_dim} kv_dim, {capacity} capacity"),
            ));
        }
        Ok(KvCache {
            n_layer,
            kv_dim,
            capacity,
            len: vec![0; n_layer],
            k: vec![vec![0.0; capacity * kv_dim]; n_layer],
            v: vec![vec![0.0; capacity * kv_dim]; n_layer],
        })
    }

    /// Positions available on `layer`.
    pub fn len_at(&self, layer: usize) -> usize {
        self.len.get(layer).copied().unwrap_or(0)
    }
    /// Positions available on every layer — the length of a complete token.
    pub fn len(&self) -> usize {
        self.len.iter().copied().min().unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Bytes this cache occupies at a given element size.
    pub fn bytes(&self, bytes_per_elem: usize) -> usize {
        Self::bytes_for(self.n_layer, self.kv_dim, self.capacity, bytes_per_elem)
    }

    /// Bytes a cache of these dimensions WOULD occupy.
    ///
    /// Deliberately an associated function. This is the number that decides
    /// how much context fits, so it is asked before a cache is created — and
    /// a version that required an instance made a test allocate 25 GB to
    /// compute one multiplication.
    pub fn bytes_for(
        n_layer: usize,
        kv_dim: usize,
        capacity: usize,
        bytes_per_elem: usize,
    ) -> usize {
        2 * n_layer * capacity * kv_dim * bytes_per_elem
    }

    /// The largest capacity that fits in `budget` bytes.
    pub fn capacity_for_budget(
        n_layer: usize,
        kv_dim: usize,
        bytes_per_elem: usize,
        budget: usize,
    ) -> usize {
        let per_token = 2 * n_layer * kv_dim * bytes_per_elem;
        budget.checked_div(per_token).unwrap_or(0)
    }

    /// Append one position's K and V for one layer.
    ///
    /// Appending per layer rather than per token is deliberate: a forward pass
    /// visits layers in order and would otherwise need to buffer the whole
    /// token's worth of state before writing any of it.
    pub fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if layer >= self.n_layer {
            return Err(Error::malformed(
                "kv cache",
                format!("layer {layer} of {}", self.n_layer),
            ));
        }
        if pos >= self.capacity {
            return Err(Error::malformed(
                "kv cache",
                format!("position {pos} exceeds capacity {}", self.capacity),
            ));
        }
        if k.len() != self.kv_dim || v.len() != self.kv_dim {
            return Err(Error::Shape {
                expected: vec![self.kv_dim],
                got: vec![k.len().max(v.len())],
            });
        }
        let o = pos * self.kv_dim;
        self.k[layer][o..o + self.kv_dim].copy_from_slice(k);
        self.v[layer][o..o + self.kv_dim].copy_from_slice(v);
        self.len[layer] = self.len[layer].max(pos + 1);
        Ok(())
    }

    /// Keys for `layer`, positions `0..len`.
    pub fn keys(&self, layer: usize) -> Result<&[f32]> {
        self.slice(&self.k, layer)
    }
    /// Values for `layer`, positions `0..len`.
    pub fn values(&self, layer: usize) -> Result<&[f32]> {
        self.slice(&self.v, layer)
    }

    fn slice<'a>(&'a self, store: &'a [Vec<f32>], layer: usize) -> Result<&'a [f32]> {
        store
            .get(layer)
            .map(|b| &b[..self.len_at(layer) * self.kv_dim])
            .ok_or_else(|| {
                Error::malformed("kv cache", format!("layer {layer} of {}", self.n_layer))
            })
    }

    /// Forget everything. The storage is kept: a new sequence in the same
    /// session should not pay to reallocate a multi-gigabyte buffer.
    pub fn clear(&mut self) {
        self.len.iter_mut().for_each(|l| *l = 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_and_reads_back_in_position_order() {
        let mut c = KvCache::new(2, 4, 8).unwrap();
        for pos in 0..3 {
            for layer in 0..2 {
                let k = vec![pos as f32 + layer as f32 * 100.0; 4];
                let v = vec![-(pos as f32); 4];
                c.append(layer, pos, &k, &v).unwrap();
            }
        }
        assert_eq!(c.len(), 3);
        let k0 = c.keys(0).unwrap();
        assert_eq!(k0.len(), 3 * 4);
        assert_eq!(k0[0], 0.0);
        assert_eq!(k0[4], 1.0, "position 1 follows position 0 contiguously");
        assert_eq!(c.keys(1).unwrap()[0], 100.0, "layers are independent");
    }

    #[test]
    fn each_layer_tracks_its_own_length() {
        // A layer reads only its own keys and values, so its length is its
        // own business. `len()` is the minimum across layers — the length of
        // a token that is complete everywhere.
        let mut c = KvCache::new(3, 2, 4).unwrap();
        c.append(0, 0, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(c.len_at(0), 1, "layer 0 can be read immediately");
        assert_eq!(c.len_at(1), 0, "layer 1 is untouched");
        assert_eq!(c.len(), 0, "the token is not complete everywhere yet");
        assert_eq!(c.keys(0).unwrap().len(), 2, "layer 0 has one position");
        assert!(c.keys(1).unwrap().is_empty());

        c.append(1, 0, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        c.append(2, 0, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(c.len(), 1, "now every layer has the position");
    }

    #[test]
    fn capacity_and_shape_violations_are_refused() {
        let mut c = KvCache::new(1, 4, 2).unwrap();
        assert!(
            c.append(0, 2, &[0.0; 4], &[0.0; 4]).is_err(),
            "past capacity"
        );
        assert!(
            c.append(1, 0, &[0.0; 4], &[0.0; 4]).is_err(),
            "no such layer"
        );
        assert!(
            c.append(0, 0, &[0.0; 3], &[0.0; 4]).is_err(),
            "wrong kv_dim"
        );
        assert!(KvCache::new(0, 4, 2).is_err());
    }

    #[test]
    fn bytes_reproduces_the_measured_context_budget() {
        // 48 layers, 4 kv heads x 128 dim = 512. Computed WITHOUT allocating:
        // an earlier version of this test built the full-context cache to ask
        // its size and allocated 25 GB to perform one multiplication.
        assert_eq!(
            KvCache::bytes_for(48, 512, 1, 2),
            98304,
            "96 KiB per token at f16"
        );
        assert_eq!(
            KvCache::bytes_for(48, 512, 262144, 1) >> 30,
            12,
            "q8_0 at the model's full context is 12 GiB"
        );

        // And the inverse: what actually fits beside 23.37 GiB of weights on
        // a 31.86 GiB card, leaving ~1.2 GiB of compute buffers.
        let free = (31.86 - 23.37 - 1.2) * (1u64 << 30) as f64;
        let fits = KvCache::capacity_for_budget(48, 512, 1, free as usize);
        assert!(
            (130_000..=170_000).contains(&fits),
            "computed ceiling {fits} should be near the measured 131072"
        );
    }

    #[test]
    fn clear_forgets_positions_without_freeing() {
        let mut c = KvCache::new(1, 2, 4).unwrap();
        c.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(c.len(), 1);
        c.clear();
        assert_eq!(c.len(), 0);
        assert_eq!(c.capacity(), 4, "capacity survives a clear");
        assert!(c.keys(0).unwrap().is_empty());
    }
}
