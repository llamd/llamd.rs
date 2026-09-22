//! Model architectures. Qwen3 and Qwen3MoE only, deliberately.
//!
//! Not yet implemented. See docs/porting-matrix.md for what this crate owes
//! and how it will be validated against the llama.cpp oracle.

#![forbid(unsafe_code)]

pub mod hparams;
pub mod model;
pub mod moe;

pub use hparams::{Arch, Hparams};
pub use model::{LayerNames, Model};
pub use moe::{Routing, route};
