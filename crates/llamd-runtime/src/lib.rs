//! Scheduling, KV cache, batching, the decode loop, memory ownership.
//!
//! Not yet implemented. See docs/porting-matrix.md for what this crate owes
//! and how it will be validated against the llama.cpp oracle.

#![forbid(unsafe_code)]

pub mod attention;
pub mod kv;
pub mod layer;

pub use attention::{AttnParams, attend};
pub use kv::KvCache;
pub use layer::{LayerTrace, embed, forward_layer};
