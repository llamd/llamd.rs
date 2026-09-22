# Subsystem-to-Rust porting matrix

Every llama.cpp subsystem, what LLAMD does with it, and how it is validated.

**Status legend:** `—` not started · `spec` designed · `wip` in progress ·
`cpu` CPU path done · `gpu` accelerated · `ok` validated against the oracle

## Reading this table

"Port" does not mean translate. For each row the question is: *what behaviour
must LLAMD reproduce, and what is the cheapest honest way to prove it does?*
Some rows are bit-exact obligations (quant block formats). Some are
behavioural (sampling). Some are deliberately **not** ported at all, because
inheriting the reference's design would defeat the purpose.

| # | llama.cpp | lines | LLAMD crate | approach | validated by | status |
|---|---|---|---|---|---|---|
| 1 | `ggml/src/gguf.cpp`, `gguf.h` | ~2k | `llamd-gguf` | **Reimplement.** A file format with a written spec; a parser is the natural Rust artifact. Memory-map, zero-copy tensor views. | Parse the real Qwen3-Coder Q6_K; every KV pair and tensor descriptor must equal `gguf_dump` output | `—` |
| 2 | `ggml-common.h` block structs | ~1k | `llamd-quant` | **Bit-exact obligation.** `block_q6_K` (210 B), `block_q8_0` (34 B), `QK_K=256`. Layout must match byte for byte or every weight is wrong. | `static_assert`-equivalent size tests; dequantize a real tensor and compare against llama.cpp's own dequant, bit for bit | `—` |
| 3 | `src/llama-vocab.cpp` | 4,732 | `llamd-tokenizer` | **Reimplement BPE first** (Qwen is BPE). SPM/WPM/UGM deferred until a model needs them. | Tokenize a corpus through both; **exact id-sequence match is mandatory**, not a tolerance | `—` |
| 4 | `src/llama-arch.cpp/.h` | 1,963 | `llamd-models` | **Reimplement as Rust enums + serde.** A 2k-line C++ enum-to-string table becomes a derive. | Architecture and tensor names resolve identically for every Qwen variant | `—` |
| 5 | `src/llama-model.cpp` | 6,842 | `llamd-models` | **Selectively reimplement.** Qwen3 and Qwen3MoE only. The other ~40 architectures are explicitly out of scope; adding one is a deliberate decision, not a sweep. | Tensor wiring matches; shapes equal the oracle's for the same file | `—` |
| 6 | `ggml` tensor core | large | `llamd-tensor` | **Redesign.** ggml's arena + `ggml_context` model exists because C has no ownership. Rust has one. Shapes/strides/dtypes with real lifetimes, and the `Backend` trait seam. | Shape and stride arithmetic property-tested; results compared per op | `—` |
| 7 | `ggml-cpu` | 93,114 | `llamd-cpu` | **Reference path, clarity over speed.** Its job is to be obviously correct so it can adjudicate GPU bugs. Not a port of 93k lines of SIMD. | It *is* the reference; validated directly against llama.cpp CPU output | `—` |
| 8 | RoPE (in `ggml`/`llama-graph`) | — | `llamd-tensor` | **Reimplement.** Position encoding, incl. NTK/YaRN scaling for long context. Off-by-one here is silent and poisons everything downstream. | Per-position tensor comparison at documented tolerance | `—` |
| 9 | attention + Flash Attention | — | `llamd-tensor`, backends | **CPU reference first, then fused GPU.** FA is an optimization of a thing that must already be correct. | Attention output vs oracle; FA vs own non-FA path must agree | `—` |
| 10 | `src/llama-kv-cache*.cpp` | 7,882 | `llamd-runtime` | **Redesign.** Four cache variants upstream reflect accumulated requirements. LLAMD starts with unified only, and adds variants when a model needs one. Layout is an AMD-first decision, not inherited. | Long-context generation matches the oracle token for token | `—` |
| 11 | `src/llama-graph.cpp` | 5,305 | `llamd-runtime` | **Redesign.** Graph building in Rust wants a typed builder, not a C struct-graph. | Same logical graph → same numerics | `—` |
| 12 | `src/llama-context.cpp` | 4,813 | `llamd-runtime` | **Reimplement.** Decode loop, batching, session state. | End-to-end sequence equality | `—` |
| 13 | `src/llama-batch.cpp` | 1,161 | `llamd-runtime` | **Reimplement.** Batch assembly and splitting. | Batched output equals sequential output | `—` |
| 14 | MoE routing (`llama-model`) | — | `llamd-models` | **Reimplement.** Qwen3MoE: top-k expert selection and gather. ~3B of 30B active per token, so this is a first-order performance path, not a detail. | Expert selection must match **exactly** — a different expert is a different model | `—` |
| 15 | `src/llama-sampler.cpp` | 4,431 | `llamd-sampling` | **Reimplement the chain.** temp, top-k, top-p, min-p, typical, penalties, mirostat, grammar. | Same seed + same logits → same token. Exact. | `—` |
| 16 | `ggml-cuda` (hipified) | 46,122 | `llamd-rocm` | **Do NOT port.** This is the whole point: a backend written *for* RDNA4 rather than CUDA-with-substitutions. wave32/64, LDS budget, matrix cores, occupancy. | Every kernel vs the CPU reference; then benchmarked against hipified llama.cpp on the same card | `—` |
| 17 | `ggml-vulkan` | 38,029 | `llamd-vulkan` | **Reimplement, first-class.** Portable path and a cross-check on ROCm: two independent GPU implementations disagreeing is a strong bug signal. | vs CPU reference; benchmarked vs llama.cpp Vulkan | `—` |
| 18 | `tools/server` | — | `llamd-server` | **Reimplement, narrow.** OpenAI-compatible surface only — what OpenCode actually calls. Not the full upstream server. | OpenCode drives LLAMD end to end | `—` |
| 19 | `src/llama-chat.cpp` | 1,032 | `llamd-server` | **Reimplement Qwen templates only.** Getting the chat template wrong makes an instruct model continue the document instead of following it — a failure this project has already met once. | Rendered prompt must be byte-identical to the oracle's | `—` |
| 20 | speculative decoding (`tools/`) | — | `llamd-runtime` | **Reimplement, verified only.** Draft tokens are **always** verified by the full target model. Acceptance may change speed; it may never change output. | Output with MTP enabled must equal output with it disabled. Non-negotiable. | `—` |
| 21 | `src/llama-grammar.cpp` | 1,720 | `llamd-sampling` | **Deferred.** GBNF constrained sampling. Not needed for the first milestones. | — | `—` |
| 22 | `src/llama-quant.cpp` | 1,487 | `llamd-quant` | **Deferred.** LLAMD *reads* quantized models first; producing them is later. | — | `—` |
| 23 | `src/llama-adapter.cpp` | — | — | **Out of scope initially.** LoRA/control vectors. | — | `—` |
| 24 | `src/llama-mmap.cpp` | 905 | `llamd-gguf` | **Reimplement** with `memmap2`. Isolated unsafe, documented. | Large-model load without copying | `—` |

## Unsafe policy

Rust is the point. Unsafe is confined to four places, each documented at its
boundary with what invariant makes it sound:

1. `llamd-gguf` — mmap and zero-copy tensor views over a borrowed file
2. `llamd-rocm` — HIP FFI
3. `llamd-vulkan` — Vulkan FFI
4. `llamd-cpu` — explicit SIMD, only where a benchmark justifies it

Everything above the backend trait — scheduling, ownership, model execution,
the API layer — is safe Rust. If safe Rust cannot express something there, that
is a design question, not a reason to reach for `unsafe`.

## What "validated" means

A row reaches `ok` when its comparison runs **automatically** in
`llamd-testing` against the real oracle, with the tolerance written down. Not
when it looks right, not when a demo works. A row that cannot be validated
automatically does not get a status — it gets a note explaining why.
