# LLAMD.rs — architecture

**Large Language Acceleration for AMD.** An AMD-first, hardware-aware Rust
inference runtime. llama.cpp is retained permanently as an upstream reference
and regression oracle — not as something to be replaced or mechanically
translated.

Reference revision analysed: `ggml-org/llama.cpp@0f8a414b7` (2026-09-22, 11109 commits).

## 1. What the reference actually looks like

Measured, not assumed:

| area | files | lines |
|---|---|---|
| `src/` (llama runtime) | 221 | 98,783 |
| `ggml/src/` (tensor library + backends) | 772 | 352,467 |
| `ggml/src/ggml-cpu` | — | 93,114 |
| `ggml/src/ggml-cuda` | 282 | 46,122 |
| `ggml/src/ggml-vulkan` | 198 | 38,029 |
| `common/` | 89 | 41,897 |

Core subsystems by weight:

| subsystem | lines | what it owns |
|---|---|---|
| `llama-kv-cache` | 7,882 | KV storage, unified/iswa/recurrent/hybrid variants |
| `llama-model` | 6,842 | architecture definitions, tensor wiring, per-arch graphs |
| `llama-graph` | 5,305 | computation graph construction |
| `llama-context` | 4,813 | session state, batching, decode loop |
| `llama-vocab` | 4,732 | BPE/SPM/WPM/UGM tokenizers |
| `llama-sampler` | 4,431 | the sampling chain |
| `llama-arch` | 1,963 | architecture and tensor-name enums |
| `llama-quant` | 1,487 | quantization |

### The finding that shapes this project

**`ggml/src/ggml-hip/` contains exactly one file: `CMakeLists.txt`.**

llama.cpp has no AMD-native backend. AMD support is produced by *hipifying the
CUDA sources* at build time — the 282 files under `ggml-cuda/` are compiled
through HIP, guarded by `GGML_USE_HIP` and `__HIP_PLATFORM_AMD__`. It works,
and it is the pragmatic engineering choice for a project that must support both
vendors from one codebase.

It is also the gap LLAMD exists to fill. A backend written as CUDA-with-
substitutions inherits NVIDIA's execution model: warp size 32, NVIDIA's
preferred tile shapes, memory hierarchy assumptions, and instruction selection
tuned for a different machine. RDNA4 has wave32 *and* wave64, a different LDS
budget, different matrix-core availability, and different occupancy behaviour.

**LLAMD's thesis:** an AMD-first backend designed against RDNA4's actual
execution model can beat a hipified CUDA backend on AMD hardware — and the only
honest way to find out is to build both paths and benchmark them side by side
on the same card.

## 2. Non-goals

- **Not a line-by-line port.** A mechanical translation would inherit every C++
  ownership pattern that Rust exists to improve on, and would be permanently
  behind upstream.
- **Not a replacement for llama.cpp.** It stays as the oracle. When LLAMD and
  llama.cpp disagree, llama.cpp is presumed right until proven otherwise.
- **Not speed-first.** Correctness and numerical fidelity outrank tokens/second.
  A fast wrong answer is worthless, and a fast *slightly* wrong answer is worse,
  because it survives casual inspection.

## 3. Layering

```
  llamd-server        OpenAI-compatible HTTP  (OpenCode talks to this)
        │
  llamd-sampling      logits → token
        │
  llamd-models        Qwen family: dense and MoE graphs
        │
  llamd-runtime       scheduling, KV cache, batching, memory ownership
        │
  llamd-tensor        shapes, strides, dtypes, the Backend trait
        │
  ┌─────┴─────────────┬──────────────────┐
  llamd-cpu       llamd-rocm         llamd-vulkan
  reference       AMD-first          portable
  (correctness)   (performance)      (first-class alternative)
        │
  llamd-quant   ← Q6_K, Q8_0 block formats, bit-exact with ggml
  llamd-gguf    ← model file parsing
  llamd-tokenizer
```

The **Backend trait** is the seam. A backend supplies allocation, transfer and
a set of kernels; nothing above `llamd-tensor` knows which one is in use. That
is what allows a fourth backend later without touching the model or runtime
architecture.

## 4. Correctness before speed

The validation harness (`llamd-testing`) runs identical inputs through both
engines and compares at every level where comparison is meaningful:

```
  same model file, prompt, seed, context, sampling parameters
        │
  ┌─────┴──────┐
llama.cpp    LLAMD
  └─────┬──────┘
        ▼
  token ids            exact match required
  tensor shapes        exact match required
  dequantized weights  bit-exact required (same block format)
  intermediate tensors where practical   documented tolerance
  logits               documented tolerance
  top-k agreement      rank correlation
  generated tokens     exact match at temperature 0
  full sequences       exact match at temperature 0
```

Tolerances are **documented per comparison**, never global, and never widened to
make a test pass. Widening a tolerance is a change to the claim the test makes,
and is reviewed as such.

## 5. Order of work

Deliberately sequential. Each phase must pass before the next begins.

1. **Analysis** — this document and the porting matrix.
2. **Workspace** — crates, CI, licensing, provenance.
3. **GGUF + CPU Qwen** — load a real model, run a small one on CPU, prove it.
4. **Correctness harness** — automated llama.cpp-vs-LLAMD comparison.
5. *Only then:* GPU. ROCm backend, then RDNA4-specific optimization.
6. Larger Qwen models, MoE routing, long context.
7. Verified MTP/speculative decoding — candidates **always** verified by the
   full target model, so acceleration can never trade away correctness.
8. Server integration.

## 6. Target hardware

AMD Radeon AI PRO R9700, 31.86 GiB, **gfx1201** (Navi 48, RDNA4), ROCm 10.0.
Linux. The reference workload is the model already running on this machine:
Qwen3-Coder-30B-A3B-Instruct Q6_K, 23.37 GiB — a MoE with ~3B active parameters
per token, which makes routing and expert-gather performance a first-order
concern rather than an afterthought.

## 7. Provenance and licensing

llama.cpp and ggml are **MIT**, © 2023-2026 The ggml authors. Any LLAMD
functionality derived from them — quantization block formats, RoPE conventions,
GGUF layout, numerical procedures — carries a file-header attribution naming the
upstream file and the revision it was derived from. See
[provenance.md](provenance.md). The vendored clone is a git checkout, not a
copy: it is excluded from LLAMD's own history.

---

## Backends: Vulkan first, ROCm capable, CPU as the reference

**Vulkan is the default GPU compute path.** It is the API this class of AMD card
exposes natively through RADV, it is vendor-neutral, and on gfx1201 it provides
everything the ROCm path relies on — verified by query, not assumed:

| feature | gfx1201 via RADV |
|---|---|
| `integerDotProduct4x8BitPackedUnsignedAccelerated` | **true** — the `udot4` equivalent, in hardware |
| `shaderInt8`, `storageBuffer8BitAccess` | true |
| subgroup arithmetic / ballot / shuffle | true |
| `subgroupSizeControl`, `computeFullSubgroups` | true |
| device-local heap | 31.75 GiB |
| compute-only queue family | present |
| `VK_KHR_cooperative_matrix` | present (unused so far) |

So choosing Vulkan as primary costs no arithmetic, and it lets the kernels and
memory strategy be tuned hard for one vendor's hardware without binding the
engine to that vendor's runtime.

**ROCm remains fully supported** — a selectable execution path and the
benchmarking reference. Both backends implement the *same identity* over the
*same planar layout* and are validated against the *same scalar reference*, so a
comparison between them on identical hardware measures the backend rather than
the algorithm.

**CPU is the fallback and the correctness reference.** It is the path that agrees
with llama.cpp token for token, and every device kernel is held to it.

### Selection

`llamd_runtime::backend` — `Vulkan`, then `Rocm`, then `Cpu`, first available
wins. `LLAMD_BACKEND=vulkan|rocm|cpu` overrides.

An explicit request that is unavailable is an **error, not a silent downgrade**.
Someone who asked for ROCm in order to benchmark it, and silently got Vulkan,
would publish a number labelled with the wrong backend. `select_from` is pure so
the policy is testable without mutating the environment — which this crate could
not do anyway, being `#![forbid(unsafe_code)]`.

```
  vulkan  AMD Radeon AI PRO R9700 (RADV GFX1201) via radv Mesa 26.0.8 — 31.75 GiB, subgroup 64  <- selected
  rocm    AMD Radeon AI PRO R9700 — 31.86 GiB, 32 WGPs (64 CUs), wave32, gfx1201
  cpu     scalar reference, always available
```

### What stays backend-agnostic, and what deliberately does not

Backend-agnostic, and unchanged by adding Vulkan: GGUF parsing (`llamd-gguf`),
the quantization formats and scalar reference (`llamd-quant`), hyperparameters
and model description (`llamd-models`), tensor shapes (`llamd-tensor`), the
tokenizer, sampling, and the decode loop's structure.

**Not abstracted, on purpose:** the device operation set. A single trait over
both APIs would force each backend to the shape of the other, and the point of a
Vulkan-first design is that Vulkan-specific kernels, descriptor strategies and
memory layouts can be optimised aggressively. Each backend crate owns its own op
surface; what they share is the planar data layout and the arithmetic, which is
what makes them comparable.

### No new dependencies

The workspace policy is that the numerical path carries none. The Vulkan host is
therefore C++ behind a small C API with a thin safe Rust wrapper — exactly the
shape `llamd-rocm` already had — and the SPIR-V is compiled by `build.rs` with
`glslc` and embedded in the library. `llamd-vulkan` adds zero crates, and builds
to a stub that reports unavailability when the toolchain is absent, so
`cargo build --workspace` still works on a machine with no GPU.
