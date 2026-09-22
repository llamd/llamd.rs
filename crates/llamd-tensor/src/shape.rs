//! Shapes and strides.
//!
//! GGML'S DIMENSION ORDER, which LLAMD keeps. `ne[0]` is the FASTEST-VARYING
//! axis — the one that is contiguous in memory. A GGUF tensor written as
//! `[2048, 151936]` therefore has rows of length 2048, and 151936 of them.
//!
//! This is the opposite of the row-major convention most people carry from
//! numpy, and it is the single easiest thing to get backwards in this project.
//! Getting it backwards does not crash: it produces a transposed matmul, which
//! yields plausible-looking numbers and a model that generates confident
//! nonsense. So the order is stated here, tested, and never re-derived.

use llamd_core::{Error, Result};

/// ggml allows at most 4 dimensions, so a shape needs no allocation.
pub const MAX_DIMS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_DIMS],
    rank: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.is_empty() || dims.len() > MAX_DIMS {
            return Err(Error::malformed(
                "shape",
                format!("rank {} is outside 1..={MAX_DIMS}", dims.len()),
            ));
        }
        if dims.contains(&0) {
            return Err(Error::malformed(
                "shape",
                format!("{dims:?} has a zero dimension"),
            ));
        }
        let mut d = [1usize; MAX_DIMS];
        d[..dims.len()].copy_from_slice(dims);
        Ok(Shape {
            dims: d,
            rank: dims.len(),
        })
    }

    /// Build from a GGUF tensor descriptor, which uses the same order.
    pub fn from_gguf(dims: &[u64]) -> Result<Self> {
        let d: Vec<usize> = dims.iter().map(|&x| x as usize).collect();
        Shape::new(&d)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }
    pub fn dim(&self, i: usize) -> usize {
        if i < self.rank { self.dims[i] } else { 1 }
    }
    pub fn elems(&self) -> usize {
        self.dims[..self.rank].iter().product()
    }

    /// Length of a contiguous row: `ne[0]`.
    pub fn row_len(&self) -> usize {
        self.dims[0]
    }
    /// How many rows the tensor holds: the product of every axis above 0.
    pub fn rows(&self) -> usize {
        self.dims[1..self.rank].iter().product()
    }

    /// Element strides, in ggml order: `stride[0]` is always 1.
    pub fn strides(&self) -> [usize; MAX_DIMS] {
        let mut s = [0usize; MAX_DIMS];
        let mut acc = 1usize;
        for (slot, &d) in s.iter_mut().zip(&self.dims[..self.rank]) {
            *slot = acc;
            acc *= d;
        }
        s
    }

    /// Flat index of a coordinate, or `None` if out of bounds.
    pub fn offset(&self, idx: &[usize]) -> Option<usize> {
        if idx.len() != self.rank {
            return None;
        }
        let st = self.strides();
        let mut off = 0;
        for (i, &c) in idx.iter().enumerate() {
            if c >= self.dims[i] {
                return None;
            }
            off += c * st[i];
        }
        Some(off)
    }

    pub fn same_as(&self, other: &Shape) -> Result<()> {
        if self == other {
            Ok(())
        } else {
            Err(Error::Shape {
                expected: self.dims().to_vec(),
                got: other.dims().to_vec(),
            })
        }
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ne0_is_the_contiguous_axis() {
        // The whole convention in one assertion. A GGUF [2048, 151936] is
        // 151936 rows of 2048, NOT 2048 rows of 151936.
        let s = Shape::new(&[2048, 151936]).unwrap();
        assert_eq!(s.row_len(), 2048);
        assert_eq!(s.rows(), 151936);
        assert_eq!(s.strides()[0], 1, "axis 0 must be contiguous");
        assert_eq!(s.strides()[1], 2048);
    }

    #[test]
    fn offsets_follow_the_stride_order() {
        let s = Shape::new(&[4, 3]).unwrap();
        assert_eq!(s.offset(&[0, 0]), Some(0));
        // Advancing axis 0 moves one element; advancing axis 1 moves a row.
        assert_eq!(s.offset(&[1, 0]), Some(1));
        assert_eq!(s.offset(&[0, 1]), Some(4));
        assert_eq!(s.offset(&[3, 2]), Some(11));
        assert_eq!(s.offset(&[4, 0]), None, "out of range must be rejected");
        assert_eq!(s.offset(&[0]), None, "wrong rank must be rejected");
    }

    #[test]
    fn elems_is_the_product_and_rows_times_row_len() {
        for dims in [vec![7], vec![4, 3], vec![2, 3, 5], vec![2, 3, 4, 5]] {
            let s = Shape::new(&dims).unwrap();
            assert_eq!(s.elems(), dims.iter().product::<usize>());
            // An invariant the backends rely on when iterating rows.
            assert_eq!(s.rows() * s.row_len(), s.elems());
        }
    }

    #[test]
    fn degenerate_shapes_are_refused() {
        // A zero dimension makes elems() == 0 while rank stays positive, and
        // every loop over it silently does nothing.
        assert!(Shape::new(&[0]).is_err());
        assert!(Shape::new(&[4, 0]).is_err());
        assert!(Shape::new(&[]).is_err());
        assert!(Shape::new(&[1, 2, 3, 4, 5]).is_err());
    }

    #[test]
    fn rank_one_has_a_single_row() {
        let s = Shape::new(&[128]).unwrap();
        assert_eq!(s.rows(), 1);
        assert_eq!(s.row_len(), 128);
    }

    #[test]
    fn mismatch_reports_both_shapes() {
        let a = Shape::new(&[4, 3]).unwrap();
        let b = Shape::new(&[3, 4]).unwrap();
        match a.same_as(&b) {
            Err(Error::Shape { expected, got }) => {
                assert_eq!(expected, vec![4, 3]);
                assert_eq!(got, vec![3, 4]);
            }
            other => panic!("expected a shape error, got {other:?}"),
        }
    }
}
