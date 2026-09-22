# LLAMD.rs

**Large Language Acceleration for AMD.** A Rust-native LLM inference engine,
AMD-first by design, with llama.cpp retained permanently as an upstream
reference and regression oracle.

> Not a Rust clone of llama.cpp. llama.cpp's AMD support is its **CUDA backend
> hipified at build time** — `ggml/src/ggml-hip/` contains one file, a
> CMakeLists. LLAMD exists to find out what a backend designed against RDNA4's
> actual execution model can do instead, and to prove it by benchmarking both
> on the same card.

## Status

Phases 1–3 of the plan in [docs/architecture.md](docs/architecture.md).

| | |
|---|---|
| Reference analysed | `llama.cpp@0f8a414b9`, 11,109 commits |
| Subsystems mapped | 24, in [docs/porting-matrix.md](docs/porting-matrix.md) |
| Workspace | 13 crates, builds clean |
| GGUF parser | **working on the real 23.37 GiB model, 28 ms** |
| Quantization | Q6_K and Q8_0, bit-exact block layouts |
| Oracle agreement | **579 tensors, 44 metadata keys — exact** |

Everything else is `—` in the matrix. Nothing is claimed that is not tested.

## Quick start

```bash
./scripts/clone-reference.sh       # restore the pinned llama.cpp checkout
cargo build --release
./target/release/llamd inspect /path/to/model.gguf

LLAMD_TEST_MODEL=/path/to/model.gguf cargo test --workspace
```

## Correctness before speed

A fast wrong answer is worthless, and a fast *slightly* wrong answer is worse,
because it survives casual inspection. So the order is fixed: CPU reference
path → automated validation against llama.cpp → only then GPU.

Tolerances are **named constants** in `llamd-core::tolerance`, not literals in
test files. Widening one is an edit to that file, with a reason, in a diff —
because the path of least resistance when a comparison fails must not be to
quietly change what the test claims.

Comparisons that admit **no** tolerance at all: token ids, tensor shapes,
dequantized weights, MoE expert selection, and anything at temperature 0.

## Target

AMD Radeon AI PRO R9700, 31.86 GiB, gfx1201 (Navi 48, RDNA4), ROCm 10.0, Linux.
Reference workload: Qwen3-Coder-30B-A3B-Instruct Q6_K — 48 layers, 128 experts
with 8 active per token, which makes routing and expert-gather a first-order
performance concern rather than a detail.

## Layout

```
crates/
  llamd-core       errors, and the tolerance vocabulary
  llamd-gguf       GGUF parsing, memory-mapped
  llamd-quant      Q6_K / Q8_0, bit-exact with ggml
  llamd-tokenizer  BPE
  llamd-tensor     shapes, strides, the Backend trait
  llamd-cpu        reference backend: clarity over speed
  llamd-models     Qwen3, Qwen3MoE
  llamd-runtime    scheduling, KV cache, decode loop
  llamd-sampling   logits to token
  llamd-server     OpenAI-compatible API
  llamd-testing    the validation harness
  llamd-bench      LLAMD vs llama.cpp, ROCm vs Vulkan
  llamd-cli        the llamd binary
llama.cpp/         pinned reference checkout (NOT committed)
```

## Unsafe policy

Four crates may use `unsafe`, each at a documented boundary: `llamd-gguf`
(mmap), `llamd-rocm` (HIP FFI), `llamd-vulkan` (Vulkan FFI), `llamd-cpu`
(explicit SIMD, only where a benchmark justifies it). Everything else is
`#![forbid(unsafe_code)]`. `llamd-gguf` uses `deny` rather than `forbid`
precisely so its single exception must be written where a reviewer sees it.

## Licence

MIT. Derives from llama.cpp / ggml (MIT, © 2023-2026 The ggml authors) — see
[docs/provenance.md](docs/provenance.md).
