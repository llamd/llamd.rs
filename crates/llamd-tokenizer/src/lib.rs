//! Tokenization. BPE first, because Qwen is BPE.
//!
//! PROVENANCE. The pre-tokenizer split, the byte mapping and the BPE merge
//! order are derived from llama.cpp (MIT, © 2023-2026 The ggml authors):
//!   - `src/unicode.cpp`     — `unicode_regex_split_custom_qwen2`, byte map
//!   - `src/llama-vocab.cpp` — the BPE bigram loop
//!
//! Reference revision: see `REFERENCE`.
//!
//! THERE IS NO TOLERANCE HERE. Tokenization is the one stage of the pipeline
//! where "close" is meaningless: a single different token id produces a
//! different prompt, and every number downstream is then answering a question
//! nobody asked. `llamd-core::tolerance::EXACT` applies, and the oracle test
//! compares whole id sequences.

#![forbid(unsafe_code)]

pub mod bpe;
pub mod bytes;
pub mod pretokenize;

pub use bpe::{TokenType, Vocab};
