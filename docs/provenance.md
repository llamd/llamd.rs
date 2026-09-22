# Provenance and licensing

## Upstream

LLAMD.rs derives functionality from **llama.cpp / ggml**:

- Repository: https://github.com/ggml-org/llama.cpp
- Licence: **MIT**
- Copyright: © 2023-2026 The ggml authors
- Pinned revision: `0f8a414b7` (see [REFERENCE](../REFERENCE))

The MIT licence permits derivation and reimplementation with attribution.
That attribution is a requirement, not a courtesy, and this file plus the
per-file headers are how it is met.

## What is derived, and what is not

| LLAMD code | derived from | nature |
|---|---|---|
| `llamd-quant` block layouts | `ggml/src/ggml-common.h` | **Format.** `block_q6_K`, `block_q8_0`, `QK_K=256`. A file format, reproduced exactly because reading it any other way yields different weights. |
| `llamd-quant` dequantization | `ggml/src/ggml-quants.c` | **Procedure.** `dequantize_row_q6_K` / `_q8_0`. Structure preserved deliberately: a tidier rewrite is a chance to be silently wrong. |
| `llamd-gguf` | `ggml/include/gguf.h`, `ggml/src/gguf.cpp` | **Format.** Reimplemented from the layout, not translated. |
| RoPE, attention, MoE routing | `src/llama-graph.cpp`, `src/llama-model.cpp` | **Numerical conventions.** Must match to produce the same outputs. |
| `llamd-tokenizer` | `src/llama-vocab.cpp` | **Behaviour.** BPE must produce identical token ids. |
| `llamd-rocm` | — | **Not derived.** Written for RDNA4 rather than adapted from CUDA. This is the project's reason for existing. |
| `llamd-tensor`, `llamd-runtime` | — | **Not derived.** ggml's arena model exists because C has no ownership; Rust does. |

## Per-file attribution

Any file implementing something derived from upstream carries a header naming
the upstream file:

```rust
//! PROVENANCE. Derived from llama.cpp / ggml, MIT, © 2023-2026 The ggml authors:
//!   - `ggml/src/ggml-common.h`  — block layouts
//!   - `ggml/src/ggml-quants.c`  — dequantization
//! Reference revision: see `REFERENCE`.
```

## The reference checkout

`llama.cpp/` is a git checkout, **excluded from LLAMD's history** by
`.gitignore`. Restore it with:

```bash
./scripts/clone-reference.sh
```

It is not vendored because: it is 614 MB of another project's history;
committing it would obscure which code is LLAMD's; and a pinned commit hash is
a stronger provenance record than a copy.

## LLAMD's own licence

MIT, for compatibility with the upstream it derives from.
