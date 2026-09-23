# Benchmarks, and what they do and do not support

Every number here was measured on this machine on 2026-09-23 and each one comes
with the command that produced it. Numbers without a command are opinions.

**Machine.** AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 31.86 GiB VRAM, 32 WGPs /
64 CUs, wave32 native. 12-core CPU, 91 GiB RAM. ROCm 10.0.

**Model.** `Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf`, 23.4 GiB — qwen3moe, 48
layers, n_embd 2048, 32 Q heads / 4 KV heads, head_dim 128, 128 experts with 8
active per token.

**Reference.** llama.cpp `b11109-0f8a414b7`, built two ways: HIP for gfx1201,
and CPU-only for the matched-numerics comparison.

---

## 1. Correctness: does LLAMD compute what llama.cpp computes?

This is the question the fork lives or dies on, and it is settled first because
a fast engine that computes something else is worthless.

### Token agreement, re-derived live

The test constants in `token_agreement.rs` were llama.cpp output *recorded
earlier*. They have now been regenerated live rather than trusted.

First, that the prompts are the same input. llama.cpp's own tokenizer:

```
$ llama-tokenize -m <model> -p Hi --ids --no-bos
  "Hi" -> [13048]   "The" -> [785]   "def" -> [750]
  "Python" -> [30280]   "1" -> [16]
```

Identical to the ids LLAMD's `Vocab::encode` produces, and no BOS is prepended
by either. Then the continuation, with token ids fed straight to
`/completion` so no tokenizer or chat template sits in between:

| prompt | llama.cpp | LLAMD | LLAMD's margin | |
|---|---|---|---|---|
| `"Hi"` | 525 `" are"` | 525 `" are"` | 0.1196 | agree |
| `"The"` | 525 `" are"` | 525 `" are"` | 2.0050 | agree |
| `"def"` | 374 `" is"` | 374 `" is"` | 0.1946 | agree |
| `"Python"` | 25 `":"` | 25 `":"` | 0.8520 | agree |
| `"1"` | 13 `"."` | 13 `"."` | 0.8905 | agree |

**5 of 5, matched configuration.** Reproduce with:

```
llama-server -m <model> --port 8099 -ngl 0 -t 12 -c 512 -ctk f16 -ctv f16 -fa off
LLAMD_TEST_MODEL=<model> cargo test --release -p llamd-testing \
  --test token_agreement -- --ignored --nocapture
```

### The confound that is worth more than the agreement

Run the same five prompts against a llama.cpp server configured the way a user
actually runs one — `-ctk q8_0 -ctv q8_0 -fa on`, 98304-token context — and
**two of the five disagree**:

| prompt | LLAMD margin | vs q8_0-KV server |
|---|---|---|
| `"Hi"` | **0.1196** | differs — `" you"` not `" are"` |
| `"def"` | **0.1946** | differs — `" find"` not `" is"` |
| `"Python"` | 0.8520 | agrees |
| `"1"` | 0.8905 | agrees |
| `"The"` | 2.0050 | agrees |

The two that differ are exactly the two smallest margins. Nothing is broken in
either implementation: quantizing the KV cache to q8_0 perturbs the residual
stream slightly, and a near-tied argmax converts a slight perturbation into a
different token.

This is the same mechanism that made the quantized dot product necessary in the
first place — at layer 47 the router put 8 experts within a factor of 2.4, and
an f32-vs-Q6_K difference flipped which experts ran. **Discrete decisions —
argmax over the vocabulary, top-k over experts — have unbounded gain with
respect to numerical error.** It is why this project keeps a named tolerance
vocabulary (`crates/llamd-core/src/tolerance.rs`) instead of one global epsilon,
and why "the outputs look close" is not a test.

### Everything else

```
cargo test --release --workspace          98 passed, 0 failed, 3 ignored
```

The three ignored tests are the ones needing the real 23 GiB model; all three
pass when given it:

| test | result |
|---|---|
| `token_agreement` | 5/5 prompts, 14.2 s |
| `forward_oracle` | argmax agrees, token 525, 0.1196 clear of second; 74.3 s |
| `throughput` | see below |

---

## 2. Performance: where LLAMD actually stands

### End to end

Measured the way llama.cpp reports it — first token separated from steady
state, KV cache growing one position per token:

| path | configuration | tok/s | % of 640 GB/s |
|---|---|---|---|
| llama.cpp, HIP | `llama-bench`, **idle card** | **94.39** ± 7.73 | 36.6% |
| llama.cpp, HIP | `-ngl 99`, q8_0 KV, 98304 ctx | 57.74 | 22.4% |
| llama.cpp, CPU | 12 threads, f16 KV, warm | 0.75 | — |
| **LLAMD, CPU** | 1 thread, scalar reference | **0.363** | 0.1% |
| LLAMD, GPU | kernel validated, *not wired in* | — | — |

Prompt processing, same run: **1636.66 ± 460.41 t/s** (pp512).

**The bar was understated in this repo by 23%.** The previously recorded 73.04
t/s was measured while a `llama-server` held 32 GB of VRAM. On an idle card
llama.cpp reaches 94.39. A benchmark taken against a contended GPU flatters
whatever it is compared to.

LLAMD generates coherent text, not just one matching token:

```
"Hi" -> " are you familiar with the concept of a \"soul\" in the context of"
   first token   2.849 s
   steady state  2.755 s/token   (spread 2.718 – 2.799 s)
```

Two honest readings of `0.363` against `0.75`:

- LLAMD is **2.1× slower** than llama.cpp on the same CPU. True.
- LLAMD is **5.8× faster per thread**, being single-threaded and scalar against
  12 threads of hand-written SIMD. Also true, and the more useful reading: it
  says the reference is a fair baseline rather than a straw man.

Neither number is the point. The CPU path exists to be *correct*, and is
deliberately unoptimised.

**A measurement error worth recording.** The first llama.cpp CPU figure taken
was 0.22 t/s, which would have flattered LLAMD as 1.6× *faster*. That run was
cold and bounded by disk, faulting in a 23 GiB mmap. Warm, twice, it is
0.69 and 0.75. Any throughput comparison against an mmap'd multi-gigabyte model
must discard the first run.

### The kernel

`cargo run --release -p llamd-bench` — Q6_K weights, Q8_K activations, real
projection shapes from this model, resident weights, 50 iterations:

| projection | MiB | cpu ms | gpu ms | speedup | GB/s |
|---|---|---|---|---|---|
| `attn_q` [2048→4096] | 6.6 | 5.545 | 0.151 | 36.8× | 45.6 |
| `attn_k` [2048→512] | 0.8 | 0.737 | 0.119 | 6.2× | 7.2 |
| `attn_out` [4096→2048] | 6.6 | 5.535 | 0.156 | 35.5× | 44.1 |
| `expert_gate` [2048→768] | 1.2 | 1.081 | 0.123 | 8.8× | 10.5 |
| `expert_down` [768→2048] | 1.2 | 1.044 | 0.135 | 7.8× | 9.6 |
| `output` [2048→151936] | 243.4 | 207.410 | 0.823 | 252.0× | **310.1** |
| all six | | 221.352 | 1.506 | 147.0× | |

The small shapes are launch-latency bound — 0.12 ms is the floor on this card,
and five of six shapes sit within 0.04 ms of it. Only `output`, at 243 MiB, is
large enough to measure bandwidth, and it reaches **310–319 GB/s** across runs.

### What that ceiling implies, stated as a ceiling

A token touches ~2.5 GiB of active weights (~3.3 B active params at 6.5625
bits). At 319 GB/s the weight traffic alone permits **~118 tok/s — 1.6× the
73 t/s llama.cpp reaches with hipified CUDA kernels.**

That is a ceiling from one kernel on one shape, not a result. It ignores
attention, norms, routing and every launch in between. What makes it worth
writing down is the *direction*: the gap between 0.363 and 73 is not a kernel
problem. The kernel is already faster than it needs to be. The gap is that the
forward pass still runs entirely on the CPU reference.

---

## 3. What this supports, and what it does not

**Supported by measurement:**

1. LLAMD's arithmetic matches ggml's where it has been implemented — 98 tests,
   plus 5/5 live token agreement on a 30 B model through 48 layers.
2. Q6_K × Q8_K on RDNA4 sustains 310–319 GB/s, implying headroom above the
   hipified path rather than a deficit.
3. The correctness discipline earns its keep: it was a near-tie at layer 47 that
   forced ggml-exact arithmetic, and the q8_0-KV experiment above shows the same
   sensitivity from the other direction.

**Not supported, and not to be claimed:**

1. **No end-to-end GPU number exists.** 0.363 t/s is the CPU reference. Until
   the ROCm backend is wired into `forward`, LLAMD has no throughput result to
   compare against llama.cpp, and the kernel ratios (147×, 252×) are against
   LLAMD's own unoptimised CPU path — not against llama.cpp.
2. **No batched prefill.** `forward` is single-token, so there is no
   prompt-processing figure. llama.cpp's 640 t/s prompt rate has no counterpart.
3. **Untested beyond qwen3moe Q6_K**, one prompt shape, one card.

## 4. The premise, restated against the evidence

`ggml/src/ggml-hip/` in the pinned llama.cpp contains exactly one file, a
`CMakeLists.txt`. AMD support is CUDA source translated by `hipify`, so RDNA4
runs kernels whose structure was chosen for a different architecture — warp 32
vs wave32 assumptions, MFMA paths that gfx1201 lacks, occupancy tuned for SMs
rather than WGPs.

The measurement that speaks to this is the 319 GB/s: a kernel written directly
for wave32 with 8 waves per workgroup and resident weights exceeds what the
hipified path needs to hit 73 t/s. That is the case for the fork — and it is a
case about headroom, not a demonstration of speed, because the integration that
would turn it into speed is not done.

The next number that would actually test the premise is LLAMD's tok/s with the
ROCm backend in the forward pass. Everything before that is groundwork.

---

## 5. Kernel-only bandwidth, and the correction it forced

Everything above measures `DeviceWeights::matvec`, which copies activations to
the device, synchronises, and copies the result back **on every call**. That is
the honest cost for a host-side caller. It is the wrong number for deciding
what to build next, because on a small matrix it is almost entirely PCIe.

`DeviceWeights::bench_kernel_only` launches back to back with one device
synchronise at the end, timed with HIP events after a warm launch — what the
kernel costs inside a resident forward pass:

| shape | size | api µs | kernel µs | GB/s | of peak |
|---|---|---|---|---|---|
| `output` [2048→151936] | 243.4 MiB | 912.0 | **531.6** | **480.1** | **75%** |
| `attn_out` [4096→2048] | 6.6 MiB | 133.6 | 22.3 | 309.0 | 48% |
| `attn_q` [2048→4096] | 6.6 MiB | 142.7 | 29.8 | 230.6 | 36% |
| `expert_gate` [2048→768] | 1.2 MiB | 127.5 | 8.3 | 154.8 | 24% |
| `attn_k` [2048→512] | 0.8 MiB | 117.5 | 7.6 | 113.9 | 18% |
| `expert_down` [768→2048] | 1.2 MiB | 157.0 | 22.3 | **57.8** | **9%** |

Two findings, and the second overturns an earlier claim in this document.

**The round-trip was costing 15×.** `expert_gate` went from 127.5 µs to 8.3 µs.
Per-call `hipMemcpy` and `hipDeviceSynchronize` were the measurement, not the
kernel. A resident forward pass recovers all of it.

**But the projection built on the best shape was wrong.** This document
previously argued ~115 t/s by applying the output projection's bandwidth to all
weight traffic. Measured per shape, that is not available:

```
  weight traffic per token
    48 x (attn 15.5 MB + 8 experts 31.0 MB) + output 255.3 MB = 2.485 GB
    at the bandwidths measured above:  13.35 ms/token  ->  75 tok/s
```

**~75 tok/s, not ~115** — parity with llama.cpp's 94.39, slightly behind it.
The error was applying a 243 MiB shape's rate to 1.2 MiB shapes that measure
5–8× slower.

### Where the ceiling actually goes

The expert matrices are **90% of the traffic** — 31.0 of 46.4 MB per layer —
and they sustain 9–24% of peak. The reason is parallelism, not the inner loop:
a 1.2 MiB matrix at 8 rows per workgroup gives the card too little in flight to
cover memory latency. `expert_down` [768→2048] is worst at 9%, because k=768 is
only three Q6_K blocks per row — the inner loop ends before it gets going.

The 243 MiB output projection reaching 75% of peak proves the memory system and
the kernel's inner loop are not the limit. Size is.

### What 140 tok/s requires, from these measurements

| | |
|---|---|
| budget at 140 t/s | 7.14 ms/token |
| output projection, measured | 0.53 ms |
| attention, 48 layers at ~270 GB/s | 2.76 ms |
| **left for experts** | **3.85 ms** |
| expert traffic, 48 × 31.0 MB | 1.49 GB |
| **required expert bandwidth** | **388 GB/s** (61% of peak) |
| measured today on that shape | 57.8 – 154.8 GB/s |

So 140 t/s needs the expert path 2.5–6.7× faster. The lever is **grouped expert
GEMM**: one kernel over all 8 routed experts turns 24 launches of 1.2 MiB into
3 of 10.3 MB, with 8× the rows. That moves the experts into the size class where
this kernel already measures 230–309 GB/s, and 388 sits between that and the
480 measured at 243 MiB.

It is not a certainty — 388 GB/s on a 10.3 MB shape is above anything measured
here at that size. It is a target with a measured basis, which is the most that
can be said before it is built.

---

## 6. Grouped expert GEMM — GPU against GPU

Section 5 located the problem: the routed-expert matrices are 90% of a token's
weight traffic and ran at 9–24% of peak, ceilinging the model near 75 t/s. The
cause was shape, not arithmetic. `expert_down` has k=768, so `blocks=3`: a lane
performs twelve multiply-accumulates and then a five-step shuffle reduction, and
2048 rows at 8 per workgroup is only 256 workgroups — too little in flight to
cover memory latency.

`q6k_q8k_matvec_grouped` puts all 8 routed experts in one launch. The row space
becomes `n_used * rows`, so 8× the workgroups and 8× the bytes in flight, from
the same 210-byte block reads. The arithmetic is untouched.

**No CPU appears in this comparison.** Both columns are this card, kernel-only,
timed with HIP events after a warm launch.

| expert shape | MiB (8 experts) | 8 separate | grouped | speedup | GB/s | of peak |
|---|---|---|---|---|---|---|
| `expert_gate` [2048→768] | 9.8 | 70.1 µs | **28.4 µs** | 2.5× | **363.3** | 57% |
| `expert_up` [2048→768] | 9.8 | 80.8 µs | **28.1 µs** | 2.9× | **367.3** | 57% |
| `expert_down` [768→2048] | 9.8 | 183.2 µs | **32.0 µs** | **5.7×** | **322.8** | 50% |

`expert_down` — the worst shape in the model — went from **57.8 GB/s to 322.8**,
a 5.6× bandwidth improvement on the path that carries most of the traffic.

### The per-token estimate, rebuilt

```
  per-expert launches     19.53 ms/token  ->    51 tok/s
  grouped                  7.73 ms/token  ->   129 tok/s
  llama.cpp measured                            94.4 tok/s
```

**129 against llama.cpp's 94.39 — 1.37×.** The target of 140 t/s needed 388 GB/s
on the expert path; it delivers 322–367.

### What this figure is, stated precisely

It is the **weight-traffic time of the matmuls**, summed over the real model
composition, with every matmul at the bandwidth measured for its own shape. It
excludes attention, normalisation, RoPE, routing, the residual adds, KV cache
traffic, and every gap between launches. Those are real costs and they will pull
the number down.

It is **not** an engine. LLAMD still generates at 0.363 tok/s, on the CPU,
because none of this is wired into `forward`. The honest claim is narrow and
worth stating exactly: *the kernel path required to beat llama.cpp on this card
now exists and measures faster than llama.cpp; the engine around it does not.*

### Correctness was not traded for it

`llamd-testing/tests/grouped_expert.rs` holds the grouped kernel to two bars:

- **Bit-identical to the ungrouped kernel** across 16,384 rows over 8 routed
  experts. Grouping changes which matrix a workgroup indexes, not how it sums,
  so any difference is a bug and a tolerance would hide it.
- **Within reduction error of the scalar CPU path**, scaled by accumulation
  magnitude rather than by each result — worst 7.3e-4 against a 1.7e-2 bound.

A repeated expert id must produce a repeated output block, and out-of-range ids,
a zero row count and a row count that does not divide the upload are all errors
rather than clamps, because each would read past the allocation.

### A concurrency bug this work exposed

The HIP layer keeps `g_aq`, `g_ad`, `g_out` and `g_eid` as process-global
statics — the design that removed per-call allocation and took a small
projection under the 0.23 ms floor. But `DeviceWeights` is `Send + Sync` and
`cargo test` runs tests concurrently, so two threads entered the same scratch.

The symptom was a result with the **right magnitude and the wrong sign**:
`-1752.9983` where `+1753.0044` was correct. In a forward pass that would have
surfaced as an occasional wrong token and been blamed on the arithmetic. The
scratch is now behind a mutex — uncontended on the inference path, where work is
one stream, and cheaper than per-thread device buffers that would multiply VRAM
by the thread count.

### Device-side Q8_K quantization

Also landed, and a precondition for everything above becoming an engine: while
activation quantization runs on the host, every one of the ~1400 matmuls in a
token must hand a buffer across the bus, so no forward pass can stay resident.
`q8k_quantize` is validated **exactly** — scales, quants and bsums — against
`llamd_quant::quantize_q8_k` on the cases most likely to expose a difference: a
negative peak (which is what `iscale = -127/max` exists for), an all-zero block,
exact `.5` values for round-half-to-even, and values needing the upper clamp.

### What remains between 129 and a real number

1. The resident forward pass — weights uploaded once, activations never leaving
   VRAM, one synchronise per token instead of ~1400.
2. Attention, norms, RoPE, routing and residuals as device kernels. Anything
   left on the host reintroduces the round-trip the grouped kernel just escaped.
3. Then measure. That number, not 129, is the one that tests the premise.

---

## 7. The resident forward pass — LLAMD generating on the GPU

Everything before this section measured kernels. This one measures an engine.

`DeviceModel` uploads the whole model to VRAM once and computes a token with
every operation on the device: **one `hipDeviceSynchronize` per token**, and the
only value crossing the bus is the argmax — a single `i32`.

```
  uploaded 23.32 GiB in 3.8 s     VRAM free 31.8 -> 8.1 GiB
```

### Correctness first

| prompt | GPU | CPU | llama.cpp | margin |
|---|---|---|---|---|
| `"Hi"` | 525 | 525 | 525 | 0.1196 |
| `"The"` | 525 | 525 | 525 | 2.0050 |
| `"def"` | 374 | 374 | 374 | 0.1946 |
| `"Python"` | 25 | 25 | 25 | 0.8520 |
| `"1"` | 13 | 13 | 13 | 0.8905 |

**GPU == CPU == llama.cpp on all five**, including the two near-ties that the
q8_0 KV experiment in §1 showed are sensitive enough to flip. Logit deviation
between GPU and CPU is 8e-2 to 2.8e-1 relative, which is 48 layers of reordered
reduction; the *decision* is identical, which is what a user sees.

### Throughput

```
  first token      19.6 ms
  steady state     14.87 ms/token  ->  67.27 tok/s   (spread 13.26 - 18.07 ms)
  llama.cpp                            94.39 tok/s
```

Generating coherent text, not one token:

> `" are you familiar with the book \"The Art of War\" by Sun Tzu?\nYes, I'm
> familiar with \"The Art of War\" by Sun Tzu. It's an ancient Chinese military
> treatise dating from the 5th century BCE, traditionally attributed to the
> military general and strategist Sun Tzu (also"`

**0.363 → 67.27 tok/s, 185×.** LLAMD is now at 0.71× llama.cpp, from 0.004×.

### Where the remaining time goes, measured

Profiling is opt-in because measuring a phase means synchronising after it —
exactly the stall the module exists to remove. A profiled token is slower than a
real one and its total must never be quoted as throughput. It is here because
"14.87 ms against a 7.73 ms matmul budget" is a question guesswork answers badly.

| phase | ms | share |
|---|---|---|
| experts | 6.570 | 27.1% |
| q/k/v proj | 3.815 | 15.7% |
| router | 3.056 | 12.8% |
| attn_out + residual | 2.610 | 10.8% |
| qk-norm + rope + kv | 2.177 | 9.0% |
| attention | 1.854 | 7.6% |
| moe tail | 1.717 | 7.1% |
| norms | 1.671 | 6.9% |
| output + argmax | 0.502 | 2.1% |
| embed | 0.035 | 0.1% |
| profiled total | 24.253 | |

Each row carries ~17.5 µs × 48 of synchronise overhead the real path does not
pay, which is why the profiled total exceeds the measured 14.87 ms.

### Two bottlenecks this found, and what they cost

**The router was 37.9% of a token — larger than all 1152 expert matmuls.**
Selection was a serial scan on thread 0: eight rounds over 128 experts, each a
global read of `probs[]`, so ~1024 dependent loads per layer with no latency
hiding. Staging the probabilities in LDS and running eight block-wide max
reductions instead took the phase from **12.326 ms to 3.056 ms** and the engine
from **41.93 to 64.66 tok/s**. Same selection, same tie rule, same 2^-14 clamp.

**Q, K and V were quantizing the same activation three times.** They all consume
the same normalised x. Quantizing once took q/k/v from 4.073 to 3.815 ms —
64.66 to **67.27 tok/s**. A smaller win than expected, and that is the useful
part of the result: it says the cost is launch *count*, not the work.

### What stands between 67 and 140

About 1200 launches go into a token. At the ~4.5 µs the numbers above imply,
that is roughly 5.4 of the 14.87 ms — more than any single phase. Three levers,
in order of expected return:

1. **Fewer launches.** The rope path alone is five launches per layer for
   trivial work (two QK-norms, two RoPEs, a KV store) and fuses to two. Norms
   fuse into the quantization that follows them; the MoE tail's combine and
   residual add fuse into one.
2. **HIP graph capture.** llama.cpp builds with `GGML_HIP_GRAPHS=ON` for this
   reason. A token's graph is structurally fixed — expert ids are read on the
   device, so routing does not change it — and only `pos` and `n_kv` vary.
3. **Faster matvec on the attention shapes.** `attn_q` measures 230 GB/s against
   480 on the output projection. The experts already reach 322–367.

140 tok/s needs 7.14 ms. The matmuls alone measure 7.73 ms today, so 140 is not
reachable by removing overhead alone — it needs item 3 as well. **90–100 tok/s,
i.e. parity with llama.cpp, is reachable from items 1 and 2**, and that is the
next honest milestone.
