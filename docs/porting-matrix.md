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
| 2 | `ggml-common.h`, `ggml-quants.c`, `ggml-cpu/quants.c` | ~3k | `llamd-quant` | **Block layouts AND the quantized dot product done.** `block_q6_K` (210 B), `block_q8_0` (34 B), `block_q8_K` (292 B), `quantize_q8_k`, `dot_q6k_q8k`. | Size tests; dequantization checked on real weights; **the dot product agrees with ggml at 1.9e-7** | `ok` |
| 3 | `src/llama-vocab.cpp` | 4,732 | `llamd-tokenizer` | **BPE done** — gpt2 model, qwen2 pre-tokenizer, byte mapping, merge loop, special tokens, decode. SPM/WPM/UGM deferred. | **24/24 prompts exactly equal to llama.cpp** on the real Qwen3-Coder vocabulary (151,936 tokens). Verified non-vacuous by injecting a whitespace-rule bug and watching it fail. | `ok` |
| 4 | `src/llama-arch.cpp/.h` | 1,963 | `llamd-models` | **Done for qwen3/qwen3moe.** `Arch` enum, `Hparams` read from GGUF with validation, per-layer tensor naming. Nothing defaulted silently. | All 579 tensors resolve on the real model; shapes checked against what the hyperparameters imply | `ok` |
| 5 | `src/llama-model.cpp` | 6,842 | `llamd-models` | **Loader done, lazily.** 30.53B params over 579 tensors opened in 73 ms; nothing materialised until asked for, since f32 would be ~122 GiB. One expert extracted in 23 ms without its 127 siblings. | Every named tensor verified present at open; shapes match hparams | `cpu` |
| 6 | `ggml` tensor core | large | `llamd-tensor` | **Shape, Tensor and the `Backend` trait done.** ggml dimension order kept (`ne[0]` contiguous); ownership replaces the arena. | Stride arithmetic tested; the convention asserted directly, since getting it backwards yields a transposed matmul and confident nonsense | `cpu` |
| 7 | `ggml-cpu` | 93,114 | `llamd-cpu` | **Reference path done** for rms_norm, softmax, silu, add/mul/scale, matmul, rope, get_rows. No SIMD, no fusion — deliberately. | **6/6 ops match ggml** via `tools/ggml-oracle`: rms_norm 6.6e-7, matmul 2.4e-6, rope 5.4e-7, get_rows **exact**. Non-vacuity proven by injecting an interleaved-RoPE bug → 6.5e-1 | `ok` |
| 8 | RoPE | — | `llamd-cpu` | **NeoX pairing done** — element `i` pairs with `i + head_dim/2`, at the model's real `freq_base` 1e7. YaRN/NTK scaling for long context still outstanding. | Matches ggml at 5.4e-7 for head_dim 128. The interleaved convention is also "RoPE", also finite, and wrong — so it has its own test | `cpu` |
| 9 | attention | — | `llamd-runtime` | **GQA done** — 32 query heads over 4 KV heads, `kv_head = q_head / 8`, scaled by 1/sqrt(128). Flash Attention still outstanding. | Matches llama.cpp's `kqv_out-0` at 3.2e-4 in the layer trace; the `%` vs `/` head mapping has its own test | `cpu` |
| 10 | `src/llama-kv-cache*.cpp` | 7,882 | `llamd-runtime` | **Unified cache done**, per-layer lengths. Layout kept simple and explicitly provisional — it is an AMD-first decision for the ROCm backend to revisit. | Used by the layer trace; `bytes_for` reproduces the 96 KiB/token that sets the context ceiling | `cpu` |
| 11 | `src/llama-graph.cpp` | 5,305 | `llamd-runtime` | **Redesign.** Graph building in Rust wants a typed builder, not a C struct-graph. | Same logical graph → same numerics | `—` |
| 12 | `src/llama-context.cpp` | 4,813 | `llamd-runtime` | **Full 48-layer forward to logits done** — MoE FFN included, 7.3 s per token on the reference. Batching and multi-token sequences still outstanding. | 13 tensors of layer 0 match the trace; routing matches **exactly through layer 45**. Token equality NOT achieved — see the MoE routing limitation below | `cpu` |
| 13 | `src/llama-batch.cpp` | 1,161 | `llamd-runtime` | **Reimplement.** Batch assembly and splitting. | Batched output equals sequential output | `—` |
| 14 | MoE routing | — | `llamd-models` | **Done.** softmax over 128 → top-8 → renormalise, with ggml's 6.1e-5 clamp and ties broken toward the lower index. | **Selection identical to ggml across 4 seeds**, set-compared since ggml documents top-k order as unspecified. Weights validated by invariant only — see the gap below | `ok` |
| 15 | `src/llama-sampler.cpp` | 4,431 | `llamd-sampling` | **Reimplement the chain.** temp, top-k, top-p, min-p, typical, penalties, mirostat, grammar. | Same seed + same logits → same token. Exact. | `—` |
| 16 | `ggml-cuda` (hipified) | 46,122 | `llamd-rocm` | **First kernel written for RDNA4**, not ported: Q6_K×Q8_K matvec, one wave32 per output row, no LDS staging. ROCm detected at build time so the workspace still builds without a GPU. | **Matches the CPU reference at 7.6e-8 – 4.8e-6** on the real card, held to `REDUCTION` since both run the same integer arithmetic. Wave-width control fails at 5.0e-1. Not yet benchmarked | `gpu` |
| 17 | `ggml-vulkan` | 38,029 | `llamd-vulkan` | **Reimplement, first-class.** Portable path and a cross-check on ROCm: two independent GPU implementations disagreeing is a strong bug signal. | vs CPU reference; benchmarked vs llama.cpp Vulkan | `—` |
| 18 | `tools/server` | — | `llamd-server` | **Reimplement, narrow.** OpenAI-compatible surface only — what OpenCode actually calls. Not the full upstream server. | OpenCode drives LLAMD end to end | `—` |
| 19 | `src/llama-chat.cpp` | 1,032 | `llamd-server` | **Reimplement Qwen templates only.** Getting the chat template wrong makes an instruct model continue the document instead of following it — a failure this project has already met once. | Rendered prompt must be byte-identical to the oracle's | `—` |
| 20 | speculative decoding (`tools/`) | — | `llamd-runtime` | **Reimplement, verified only.** Draft tokens are **always** verified by the full target model. Acceptance may change speed; it may never change output. | Output with MTP enabled must equal output with it disabled. Non-negotiable. | `—` |
| 21 | `src/llama-grammar.cpp` | 1,720 | `llamd-sampling` | **Deferred.** GBNF constrained sampling. Not needed for the first milestones. | — | `—` |
| 22 | `src/llama-quant.cpp` | 1,487 | `llamd-quant` | **Deferred.** LLAMD *reads* quantized models first; producing them is later. | — | `—` |
| 23 | `src/llama-adapter.cpp` | — | — | **Out of scope initially.** LoRA/control vectors. | — | `—` |
| 24 | `src/llama-mmap.cpp` | 905 | `llamd-gguf` | **Reimplement** with `memmap2`. Isolated unsafe, documented. | Large-model load without copying | `—` |

## The limitation that shaped this work — RESOLVED

An f32 reference could not match a quantized MoE implementation token-for-token.
Routing matched llama.cpp exactly through layer 45; at layer 47 the router put
eight experts within a factor of 2.4, the quantized-path difference flipped the
selection, and llama.cpp's greedy token ranked **second**, 0.62 logit below ours.

**Fixed by performing the same arithmetic.** `llamd-quant::dot_q6k_q8k`
quantizes activations to Q8_K and runs the integer dot product ggml runs,
rather than dequantizing weights to f32. Against ggml's own `mul_mat` on Q6_K
weights it agrees to **1.1e-7 – 1.9e-7**, versus 5.5e-4 for the f32 path.

Result on the real model:

| | f32 reference | quantized path |
|---|---|---|
| llama.cpp's token | rank 1, −0.62 | **rank 0** |
| logit[0] | 7.0420 (oracle 5.9153) | **5.9729** |
| logit sum gap | 2.95e-2 | **8.2e-3** |
| time per token | 7.3 s | **3.0 s** |

Faster as well as correct: the quantized path never materialises 23 GiB of f32.

**Five prompts now agree with llama.cpp's greedy token**, two of them with
margins under 0.2 — `"Hi"` at 0.1196 and `"def"` at 0.1946 — which are exactly
the near-ties the f32 path got wrong.

The general lesson stands for the GPU work: **MoE routing is discrete, so it
amplifies arbitrarily small numerical differences into arbitrarily large output
differences.** A ROCm or Vulkan kernel must match the quantized path, not
approximate it — and it should be compared against ggml at `REDUCTION`, not
against the f32 reference at `QUANTIZED_MATMUL`.

## The first RDNA4 kernel, and why it is shaped that way

`q6k_q8k_matvec` assigns **one wave32 per output row**, accumulates in
registers, and reduces with `__shfl_down` — no LDS, no barriers.

The CUDA idiom would stage a tile of weights through shared memory so several
outputs can reuse it. With a single activation vector **there is no reuse**:
each weight is read exactly once. Staging would add a barrier and an occupancy
cost for nothing. That is the first place the two designs part company, and it
is the kind of decision a hipified CUDA backend cannot make.

Reported correctly now, after two mistakes worth keeping in mind on this
hardware: `multiProcessorCount` counts **workgroup processors**, each holding
two CUs, so the R9700's 64 CUs appear as 32; and `major`/`minor` give `12.0`
where `gcnArchName` gives the real `gfx1201`, which is what hipcc compiled for.

### What the first benchmark measured, including the part that was wrong

The kernel was 2.6× SLOWER than the scalar CPU when first timed — 13.25 ms
against 5.02 ms for a 6.6 MiB projection, of which only 0.34 ms was the
transfer. The design reasoning was sound; the *configuration* was not. Two
fixes, each measured separately:

| | attn_q 6.6 MiB | all six shapes |
|---|---|---|
| upload per call, 1 wave/workgroup | 13.25 ms | — |
| weights resident, 8 waves/workgroup | 0.349 ms | 3.611 ms |
| + persistent activation scratch | **0.232 ms** | **2.335 ms** |

Against the scalar CPU reference, on the real projection shapes:

```text
  attn_q      [2048→4096]      0.232 ms    46.6x     29.6 GB/s
  attn_k      [2048→512]       0.158 ms     5.2x      5.4 GB/s
  attn_out    [4096→2048]      0.175 ms    38.8x     39.3 GB/s
  expert_gate [2048→768]       0.173 ms     6.8x      7.5 GB/s
  expert_down [768→2048]       0.166 ms     7.1x      7.8 GB/s
  output      [2048→151936]    0.823 ms   252.0x    310.1 GB/s
```

**A floor of ~0.16 ms remains** on every small shape — launch plus
synchronise, not arithmetic. The MoE experts sit squarely inside it, and eight
of them run per layer, so that floor is the next thing worth attacking:
batching the experts into one launch, or an asynchronous stream that does not
synchronise per call.

Note the CPU reference is deliberately unoptimised, so these ratios say
nothing about llama.cpp.

### The bar, measured

llama.cpp built with `GGML_HIP=ON` for gfx1201, on this card:

```text
  qwen3moe 30B.A3B Q6_K · ROCm · ngl 99
    pp512  1636.66 ± 460.41 t/s
    tg64     94.39 ±   7.73 t/s
```

Measured on an IDLE card. An earlier 73.04 t/s recorded here was taken
while a llama-server held 32 GB of VRAM — benchmarking against a busy card
understates the bar by 23%.

A live `llama-server` on the same card, configured as a user would run one
(98304-token context, `-ctk q8_0 -ctv q8_0 -fa on`), measures **57.7 t/s**
instead. The bar depends on configuration, so both are recorded.

**LLAMD has no end-to-end GPU number to set against that yet.** The GPU kernel
is validated but is not wired into the forward pass, which runs on the CPU
reference at **2.755 s/token — 0.363 t/s**, now measured end to end by
`llamd-testing/tests/throughput.rs` rather than inferred from a single forward.
For scale, llama.cpp on the same CPU with 12 threads reaches 0.75 t/s: the
single-threaded scalar reference is within ~2× of it on one twelfth of the
cores. Quoting the kernel speedups as though they were a comparison with
llama.cpp would still be dishonest — they are measured against that
deliberately unoptimised path.

What the kernel bandwidth does say is more useful. A MoE token touches roughly
2.5 GiB of active weights, so at the **310–319 GB/s** measured on the output
projection the arithmetic alone would permit ~118 t/s. **That estimate is
wrong, and the per-shape sweep in [benchmarks.md](benchmarks.md) shows why:**
it applies the largest shape's bandwidth to all traffic, but the expert
matrices are 90% of the bytes and sustain only 9–24% of peak. Weighted by what
the model actually reads, today's kernel ceiling is **~75 t/s** — parity with
llama.cpp, not 1.6× it. (An earlier figure of 178 GB/s here was a hardcoded constant that had
drifted from what the benchmark measured; it now prints the measurement.)
**The kernel is not the bottleneck; the missing integration is.** That makes
the next step concrete rather than speculative: hold weights resident across the
whole model, run the forward pass on the device, and only then is there a number
that tests the premise.

Full methodology, the live token-agreement re-derivation, and the near-tie
experiment that shows why the tolerance discipline matters: **[benchmarks.md](benchmarks.md)**.

## Gaps worth knowing about

**~~MoE weights are not compared against ggml~~ — CLOSED.** The layer trace
from `llama-eval-callback` exposes the routing in four stages, so all three
meaningful ones are now compared independently: the gathered probabilities
(1.35e-3), their clamped sum (1.35e-3), and the renormalised weights (3.4e-8).

Closing it corrected a misreading: `ffn_moe_weights-0` is the gather BEFORE
normalisation. Comparing our normalised weights against it gave 1.000000 vs
0.349709 — not a precision failure but two different quantities. The top 8
experts of 128 hold about 35% of the probability mass.

**Two blind spots in the quantized path, found by trying to break it.**
Swapping `round_ties_even` for Rust's `round` changes no token and fails no
oracle comparison — exact .5 values are vanishingly rare in real activations.
Dropping the negative scale likewise changes nothing here, because `d` is
`1/iscale` so the sign cancels in the product. Both are still reproduced
faithfully, and both are covered only by unit tests that ask directly. A
stage-level property that end-to-end tests cannot see needs its own test.

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
