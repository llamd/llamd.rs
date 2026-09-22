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
| 3 | `src/llama-vocab.cpp` | 4,732 | `llamd-tokenizer` | **BPE done** — gpt2 model, qwen2 pre-tokenizer, byte mapping, merge loop, special tokens, decode. SPM/WPM/UGM deferred. | **24/24 prompts exactly equal to llama.cpp** on the real Qwen3-Coder vocabulary (151,936 tokens). Verified non-vacuous by injecting a whitespace-rule bug and watching it fail. | `ok` |
| 4 | `src/llama-arch.cpp/.h` | 1,963 | `llamd-models` | **Done for qwen3/qwen3moe.** `Arch` enum, `Hparams` read from GGUF with validation, per-layer tensor naming. Nothing defaulted silently. | All 579 tensors resolve on the real model; shapes checked against what the hyperparameters imply | `ok` |
| 5 | `src/llama-model.cpp` | 6,842 | `llamd-models` | **Loader done, lazily.** 30.53B params over 579 tensors opened in 73 ms; nothing materialised until asked for, since f32 would be ~122 GiB. One expert extracted in 23 ms without its 127 siblings. | Every named tensor verified present at open; shapes match hparams | `cpu` |
| 6 | `ggml` tensor core | large | `llamd-tensor` | **Shape, Tensor and the `Backend` trait done.** ggml dimension order kept (`ne[0]` contiguous); ownership replaces the arena. | Stride arithmetic tested; the convention asserted directly, since getting it backwards yields a transposed matmul and confident nonsense | `cpu` |
| 7 | `ggml-cpu` | 93,114 | `llamd-cpu` | **Reference path done** for rms_norm, softmax, silu, add/mul/scale, matmul, rope, get_rows. No SIMD, no fusion — deliberately. | **6/6 ops match ggml** via `tools/ggml-oracle`: rms_norm 6.6e-7, matmul 2.4e-6, rope 5.4e-7, get_rows **exact**. Non-vacuity proven by injecting an interleaved-RoPE bug → 6.5e-1 | `ok` |
| 8 | RoPE | — | `llamd-cpu` | **NeoX pairing done** — element `i` pairs with `i + head_dim/2`, at the model's real `freq_base` 1e7. YaRN/NTK scaling for long context still outstanding. | Matches ggml at 5.4e-7 for head_dim 128. The interleaved convention is also "RoPE", also finite, and wrong — so it has its own test | `cpu` |
| 9 | attention | — | `llamd-runtime` | **GQA done** — 32 query heads over 4 KV heads, `kv_head = q_head / 8`, scaled by 1/sqrt(128). Flash Attention still outstanding. | Matches llama.cpp's `kqv_out-0` at 3.2e-4 in the layer trace; the `%` vs `/` head mapping has its own test | `cpu` |
| 10 | `src/llama-kv-cache*.cpp` | 7,882 | `llamd-runtime` | **Unified cache done**, per-layer lengths. Layout kept simple and explicitly provisional — it is an AMD-first decision for the ROCm backend to revisit. | Used by the layer trace; `bytes_for` reproduces the 96 KiB/token that sets the context ceiling | `cpu` |
| 11 | `src/llama-graph.cpp` | 5,305 | `llamd-runtime` | **Redesign.** Graph building in Rust wants a typed builder, not a C struct-graph. | Same logical graph → same numerics | `—` |
| 12 | `src/llama-context.cpp` | 4,813 | `llamd-runtime` | **Single-token layer forward done.** Full decode loop over 48 layers, batching and session state still outstanding. | **13 tensors of layer 0 match llama.cpp's own trace**; verified non-vacuous by skipping QK-norm → 2.3e0 | `cpu` |
| 13 | `src/llama-batch.cpp` | 1,161 | `llamd-runtime` | **Reimplement.** Batch assembly and splitting. | Batched output equals sequential output | `—` |
| 14 | MoE routing | — | `llamd-models` | **Done.** softmax over 128 → top-8 → renormalise, with ggml's 6.1e-5 clamp and ties broken toward the lower index. | **Selection identical to ggml across 4 seeds**, set-compared since ggml documents top-k order as unspecified. Weights validated by invariant only — see the gap below | `ok` |
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

## Gaps worth knowing about

**~~MoE weights are not compared against ggml~~ — CLOSED.** The layer trace
from `llama-eval-callback` exposes the routing in four stages, so all three
meaningful ones are now compared independently: the gathered probabilities
(1.35e-3), their clamped sum (1.35e-3), and the renormalised weights (3.4e-8).

Closing it corrected a misreading: `ffn_moe_weights-0` is the gather BEFORE
normalisation. Comparing our normalised weights against it gave 1.000000 vs
0.349709 — not a precision failure but two different quantities. The top 8
experts of 128 hold about 35% of the probability mass.

**The layer comparison uses sums, which a permutation would survive.**
`eval-callback` truncates the values it prints but sums over the whole tensor,
so the sum is exact and covers every element — enough to catch a wrong
projection, a missing QK-norm or a RoPE convention error. It is not enough to
catch a tensor with the right elements in the wrong order. Shapes are compared
too, which narrows it without closing it.

**A claim I had to retract.** This module's comment asserted that softmaxing
after top-k instead of before would change the weights. A negative control
disproved it: renormalising a subset of a softmax is algebraically identical to
softmaxing the subset, so the order changes nothing. The real distinction is
`SOFTMAX` versus `SOFTMAX_WEIGHT` gating, and that now has its own test. Prose
in a comment is not evidence; the control was.

## A blind spot worth knowing about

Row 3 reached `ok` on an end-to-end comparison. Deliberately breaking the
per-digit split rule (`\p{N}` → `\p{N}+`) did **not** fail that comparison,
because BPE merges the digits back and the final ids are identical.

An end-to-end comparison proves the *pipeline* agrees, not that each stage
does. Stages whose errors are absorbed downstream need their own tests — and
the only way to learn which stages those are is to try breaking them. Every
row that reaches `ok` should get the same treatment.

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
