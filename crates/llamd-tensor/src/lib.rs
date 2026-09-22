//! Tensors, and the `Backend` trait every accelerator implements.
//!
//! WHY THIS IS A REDESIGN AND NOT A PORT. ggml's arena plus `ggml_context`
//! exists because C has no ownership: tensors are allocated from a scratch
//! buffer and freed together because tracking them individually would be
//! error-prone. Rust has ownership, so LLAMD uses it — a tensor owns its
//! storage, lifetimes are real, and there is no context to forget to free.
//!
//! WHAT IS NOT REDESIGNED is the dimension order and the numerical
//! conventions. Those are interface, not implementation: getting them "better"
//! means getting different answers from the reference.
//!
//! THE BACKEND SEAM. Nothing above this crate may know which backend is in
//! use. That is what allows a ROCm or Vulkan backend to be added later without
//! touching the model or runtime code, and it is why the trait takes and
//! returns owned tensors rather than exposing device memory.

#![forbid(unsafe_code)]

pub mod shape;

use llamd_core::{Error, Result};
pub use shape::{MAX_DIMS, Shape};

/// A tensor's element type.
///
/// The reference path materialises everything as `F32`. Quantized types
/// survive in the GGUF and are dequantized on load; keeping them packed is a
/// GPU-backend optimization, and doing it in the reference path would trade
/// the one property that path exists for — being obviously correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

/// An owned tensor with f32 storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Shape, data: Vec<f32>) -> Result<Self> {
        if data.len() != shape.elems() {
            return Err(Error::malformed(
                "tensor",
                format!(
                    "shape {shape} needs {} elements, got {}",
                    shape.elems(),
                    data.len()
                ),
            ));
        }
        Ok(Tensor { shape, data })
    }

    pub fn zeros(shape: Shape) -> Self {
        let n = shape.elems();
        Tensor {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn from_slice(dims: &[usize], data: &[f32]) -> Result<Self> {
        Tensor::new(Shape::new(dims)?, data.to_vec())
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        DType::F32
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn elems(&self) -> usize {
        self.data.len()
    }

    /// One contiguous row. Rows are the unit almost every kernel works in,
    /// because axis 0 is the contiguous one.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        let n = self.shape.row_len();
        self.data.get(i * n..(i + 1) * n)
    }
    pub fn row_mut(&mut self, i: usize) -> Option<&mut [f32]> {
        let n = self.shape.row_len();
        self.data.get_mut(i * n..(i + 1) * n)
    }
    pub fn rows(&self) -> usize {
        self.shape.rows()
    }
    pub fn row_len(&self) -> usize {
        self.shape.row_len()
    }

    /// Reshape without moving data. Only valid when the element count matches.
    pub fn reshape(&self, dims: &[usize]) -> Result<Tensor> {
        let s = Shape::new(dims)?;
        if s.elems() != self.data.len() {
            return Err(Error::malformed(
                "reshape",
                format!("{} elements cannot become {s}", self.data.len()),
            ));
        }
        Ok(Tensor {
            shape: s,
            data: self.data.clone(),
        })
    }
}

/// How RoPE is applied.
///
/// `freq_base` is `rope.freq_base` from the GGUF — 10,000,000 for this model,
/// not the 10,000 of the original paper. A wrong base does not fail: it
/// rotates by the wrong angle at every position, and the model degrades with
/// context length in a way that looks like a quality problem rather than a bug.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
}

impl RopeConfig {
    pub fn new(head_dim: usize, freq_base: f32) -> Self {
        RopeConfig {
            head_dim,
            freq_base,
            freq_scale: 1.0,
        }
    }
}

/// The operations a backend must provide.
///
/// Deliberately small. Every op here is one a Qwen forward pass needs, and
/// each is a place a GPU backend can do something clever. Fusion belongs
/// behind this trait, not in front of it: a fused kernel must still be
/// checkable against the unfused reference.
pub trait Backend {
    fn name(&self) -> &str;

    /// Root-mean-square normalisation, applied per row.
    fn rms_norm(&self, x: &Tensor, eps: f32) -> Result<Tensor>;

    /// Softmax along the contiguous axis, per row.
    fn softmax(&self, x: &Tensor) -> Result<Tensor>;

    /// SiLU, also called swish: `x * sigmoid(x)`.
    fn silu(&self, x: &Tensor) -> Result<Tensor>;

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn scale(&self, x: &Tensor, by: f32) -> Result<Tensor>;

    /// `w` is `[k, m]` and `x` is `[k, n]` in ggml order; the result is
    /// `[m, n]`. Both operands share the contiguous axis `k`, which is what
    /// makes this a dot product over rows rather than a transpose.
    fn matmul(&self, w: &Tensor, x: &Tensor) -> Result<Tensor>;

    /// Rotary position embedding, in place of `x`, for the given positions.
    fn rope(&self, x: &Tensor, positions: &[u32], cfg: RopeConfig) -> Result<Tensor>;

    /// Gather whole rows by index — embeddings, and MoE expert selection.
    fn get_rows(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_length_must_match_the_shape() {
        let s = Shape::new(&[2, 3]).unwrap();
        assert!(Tensor::new(s, vec![0.0; 6]).is_ok());
        assert!(
            Tensor::new(s, vec![0.0; 5]).is_err(),
            "short data must be refused"
        );
        assert!(
            Tensor::new(s, vec![0.0; 7]).is_err(),
            "long data must be refused"
        );
    }

    #[test]
    fn rows_are_contiguous_slices() {
        let t = Tensor::from_slice(&[3, 2], &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.row_len(), 3);
        // Row 0 is the FIRST THREE elements, because axis 0 is contiguous.
        assert_eq!(t.row(0), Some(&[1., 2., 3.][..]));
        assert_eq!(t.row(1), Some(&[4., 5., 6.][..]));
        assert_eq!(t.row(2), None);
    }

    #[test]
    fn reshape_preserves_data_and_checks_the_count() {
        let t = Tensor::from_slice(&[6], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.data(), t.data());
        assert_eq!(r.rows(), 2);
        assert!(
            t.reshape(&[4, 2]).is_err(),
            "a reshape that changes the count must fail"
        );
    }
}
