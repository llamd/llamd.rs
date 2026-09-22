//! Shared vocabulary for the workspace.
//!
//! Two things live here that nothing else should own: the error type, and the
//! definition of what "close enough" means. The second matters more than it
//! looks. This project compares its own numbers against llama.cpp's, and a
//! tolerance that is a bare `f32` literal scattered through test files becomes
//! a tolerance that gets widened whenever something fails. Tolerances are
//! named, documented, and live in one place.

#![forbid(unsafe_code)]

use std::fmt;

pub mod tolerance;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// The file is not what it claims to be — bad magic, impossible length,
    /// a count that would not fit in memory.
    #[error("malformed {what}: {detail}")]
    Malformed { what: &'static str, detail: String },

    /// Well-formed, but describes something LLAMD does not implement. Kept
    /// separate from `Malformed` on purpose: one means the file is broken,
    /// the other means we are. Conflating them sends people to debug the
    /// wrong thing.
    #[error("unsupported {what}: {detail}")]
    Unsupported { what: &'static str, detail: String },

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    Shape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn malformed(what: &'static str, detail: impl fmt::Display) -> Self {
        Error::Malformed {
            what,
            detail: detail.to_string(),
        }
    }
    pub fn unsupported(what: &'static str, detail: impl fmt::Display) -> Self {
        Error::Unsupported {
            what,
            detail: detail.to_string(),
        }
    }
    pub fn other(detail: impl fmt::Display) -> Self {
        Error::Other(detail.to_string())
    }
}
