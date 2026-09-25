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

---

## 8. Parity, and past it — 101.9 tok/s against llama.cpp's 95.6

§7 left LLAMD at 67.27 tok/s against llama.cpp's 94.39, with a plan: fuse
launches, capture a HIP graph, and expect ~90–100. Both were built. One worked.

**Final, warmed, 64 generated tokens, matching `llama-bench tg64`:**

| | tok/s | ms/token |
|---|---|---|
| **LLAMD** | **101.9 – 103.9** | 9.62 – 9.84 |
| llama.cpp (two runs) | 94.39 ± 7.73, 95.55 ± 5.59 | |
| **ratio** | **1.07 – 1.10×** | |

Three consecutive runs gave 103.94, 103.39, 102.13 — spread 8.99 to 10.89 ms per
token. **GPU == CPU == llama.cpp on all five agreement prompts throughout.**

### What actually produced the gain

| step | tok/s | |
|---|---|---|
| resident forward, §7 | 67.27 | |
| fuse rope path, norm+quant, moe tail | 79.71 | +18% |
| parallelise rms_norm_quant | 83.84 | +5% |
| one workgroup per row in the F32 matvec | 82.84 → 101.60 with the next step | |
| **fuse Q/K/V into one launch, gate/up into one** | **101.6 – 103.9** | **+23%** |

The last step was by far the largest, and not for the reason the fusions were
begun. Q, K and V are 512, 64 and 64 workgroups. Issued separately, the K and V
projections each occupy 64 workgroups on a 32-WGP card — the GPU is nearly idle
while they run. Fused, all three are 640 workgroups in one launch and K and V
fill the card alongside Q. The same applies to the expert gate and up: 768
workgroups each becomes 1536 together.

**So the win was occupancy, not launch overhead.** That distinction is the
useful result here, because it is the opposite of what §7 predicted.

### The HIP graph was a dead end, and the measurement says so

§7 argued from a ~4.5 µs/launch estimate that ~1200 launches cost roughly 5.4 of
14.87 ms, and that graph capture should recover most of it. Capture was
implemented — device-resident `[token, pos]` so the graph is structurally
invariant, a dedicated stream, ownership tracking — and measured:

| | before fusion | after fusion |
|---|---|---|
| graph replay | 13.851 ms | 9.538 – 9.662 ms |
| individual launches | **12.545 ms** | 9.581 – 9.676 ms |

**Slower before fusion, indistinguishable after.** The launch-overhead estimate
was inferred, not measured, and it was wrong. `use_graph` is off by default and
the machinery is kept so the comparison stays reproducible and a future ROCm can
be retested by flipping one flag.

Recording this because a plan that half-failed is more informative than one that
worked: the reasoning behind the graph was plausible and the number refuted it.

### Two bugs found, both by symptom rather than by review

**A graph replayed against freed memory.** The HIP graph is a process-global
object holding raw pointers into whichever model recorded it. Load a model,
capture, drop it, load another: the new model saw `graph::exists()` and replayed
a graph pointing at freed buffers. It surfaced as
`HSA_STATUS_ERROR_MEMORY_FAULT ... kernel: fk_embed_q6k_meta` — reading an
embedding table that no longer existed. Fixed with instance ownership: each
model takes an id, and a graph recorded under a different id is discarded rather
than replayed. A comment would not have prevented it.

**A hardcoded epsilon that happened to be right.** The fused QK-norm kernels were
written with `1e-6f` inline. This model's `rms_eps` *is* 1e-6, so every test
passed; another model would have been silently wrong. Now passed as an argument.

### Where the remaining time goes

| phase | ms (profiled) | share |
|---|---|---|
| experts | 4.966 | 24.9% |
| router | 2.667 | 13.4% |
| attn_out + residual | 2.648 | 13.3% |
| q/k/v proj | 2.353 | 11.8% |
| profiled total | 19.970 | |

Profiling synchronises after each phase, so its total overstates the 9.8 ms real
token; the proportions are what to read.

### Toward 140

140 tok/s needs 7.14 ms against today's 9.62–9.84. The remaining levers are no
longer integration:

1. **The matvec kernel itself.** `attn_q` measures 230 GB/s where the output
   projection reaches 480, and the experts 322–367. Closing that on the
   attention shapes is worth roughly 1 ms.
2. **Fuse the residual add into the projection** that produces it, as
   `moe_combine_add` already does — the attention output path still runs a
   separate `add`.
3. **The remaining small kernels.** `qk-norm+rope` and `moe tail` are ~1.5 ms
   combined for work that touches a few thousand floats.

Those three plausibly reach ~8 ms, i.e. ~125 tok/s. 140 would need the expert
path near 480 GB/s, which nothing has yet demonstrated at a 10 MB shape.

---

## 9. Attempting 140: two more measurements that said no

§8 named the next lever: `attn_q` measures 230 GB/s where the output projection
reaches 480, and closing that should be worth ~1 ms. The lever was built,
measured, and does not exist.

### Split-k: several waves per output row

`attn_out` is [4096 → 2048]. At one wave per row that is 2048 waves — 65536
threads against this card's 131072-thread capacity, so half the GPU is idle for
the whole projection. Row count is fixed by the model, so the only way to fill
the card on a short matrix is to put several waves on one row, each summing a
strided subset of the row's blocks, combining through LDS.

Kernel-only, every legal `nsub`:

| shape | rows | blocks | nsub=1 | best | gain |
|---|---|---|---|---|---|
| `attn_q` [2048→4096] | 4096 | 8 | 38.8 µs / 177 GB/s | **nsub=4: 25.5 µs / 269** | **1.52×** |
| `attn_out` [4096→2048] | 2048 | 16 | 21.9 µs / 314 | nsub=4: 19.7 µs / 350 | 1.11× |
| `expert_gate` ×8 | 6144 | 8 | 44.5 µs / 232 | **nsub=2: 31.3 µs / 330** | **1.42×** |
| `expert_down` ×8 | 16384 | 3 | 41.8 µs / 247 | nsub=2: 35.0 µs / 295 | 1.19× |
| `output` [2048→151936] | 151936 | 8 | **608 µs / 419** | nsub=1 | — |

A clear result, pointing at a 1.4–1.5× gain on the two phases that carry most of
the traffic. **Applied to the engine it made things slower.**

| nsub (q/k/v, gate+up, down) | ms/token | tok/s |
|---|---|---|
| **(1, 1, 1)** | **9.432** | **106.02** |
| (1, 2, 1) | 9.566 | 104.54 |
| (2, 1, 1) | 9.942 | 100.58 |
| (1, 1, 2) | 10.261 | 97.45 |
| (4, 2, 2) | 10.317 | 96.93 |
| (2, 2, 2) | 10.800 | 92.60 |

Every split-k configuration lost. Split-k is off.

### Why the benchmark lied, and it is not the kernel's fault

**The R9700 has 64 MB of Infinity Cache.** The benchmark loops one matrix 1000
times, so after the first iteration a 6.6 MiB projection is entirely resident:
it measures cache bandwidth. Every shape in that table except the 243 MiB output
projection fits — which is exactly why `output` is the one shape where the
benchmark and the engine agree that nsub=1 wins.

In a real token, each layer's weights are read **once**, cold, from HBM. There is
no reuse to cache: 2.5 GB streams through per token and 48 layers never revisit
each other's weights. Split-k adds LDS reductions and barriers for parallelism
that a bandwidth-bound cold read cannot use.

This also retires the "230 GB/s vs 480 GB/s" gap §8 pointed at. That comparison
was between a cache-resident 6.6 MiB matrix and a genuinely HBM-bound 243 MiB
one — not between a slow kernel and a fast one.

A second, smaller finding from the same exercise: the split-k kernels cost ~5% at
`nsub=1`, because they still write to LDS, barrier, and run a combine loop with
nothing to combine. `nsub == 1` now dispatches to the dedicated kernel.

### Where this leaves 140

**Final: 102.6 – 103.3 tok/s, 1.07 – 1.08× llama.cpp**, three consecutive runs,
warmed, 64 generated tokens, tokens identical to llama.cpp throughout.

140 tok/s needs 7.14 ms/token. The engine is at 9.68–9.75 ms and reads 2.5 GB of
weights per token, so 140 requires **355 GB/s sustained on a cold stream**. The
only shape ever measured above that here was the output projection at 419–480
GB/s — 243 MiB, and the one shape large enough that its number is honest.

So the remaining 26% is not an integration problem and not an occupancy problem.
It is the cold-stream bandwidth of the Q6_K dot product itself, and nothing
tried so far moves it. Candidates, in order of expected return at the time:

1. **Wider loads** — the inner loop reads `ql` and `qh` a byte at a time.
   **Attempted; see §10. It does not help the engine.**
2. `__builtin_nontemporal_load` for weights that are read once and never reused.
3. **Two tokens at a time.** Batch-2 decode halves the weight traffic per token
   outright — a change to the engine's shape rather than a kernel.

---

## 10. Planar Q6_K: a 1.16× kernel that the engine does not feel

§9 ended by naming load width as the most promising remaining lever, and the
reasoning was sound enough to be worth building.

### The measurement that motivated it

A cold streaming read on this card, 2 GiB buffer so nothing caches:

| load width | GB/s | of the 640 GB/s spec |
|---|---|---|
| `uint4` (16 bytes) | **637.8** | **100%** |

The card delivers its full rated bandwidth. The Q6_K matvec sustains ~260 GB/s
in the engine, and the reason it cannot use a wide load is alignment: ggml
stores Q6_K as **210-byte blocks**, so block `b` sits at `b × 210` and
`210 % 16 = 2`. Every block has a different alignment mod 16 and the kernel is
forced into single-byte loads.

### The fix, and it is a clean one

`repack_q6k_planar` splits the four components into separate planes — 128 bytes
of low nibbles, 64 of high bits, 16 scales, one f16 — whose per-block strides all
divide 16. `ql`, `qh` and `sc` become 16-byte aligned everywhere. It is a
**permutation**: total bytes, and therefore VRAM, are unchanged, asserted
byte-for-byte by `planar_repack_is_a_permutation`.

The kernel then reads one `uint4` of `ql`, one of `qh` and two of activation per
32 multiply-accumulates, where the interleaved kernel spends 14 byte loads per 8.

Measured on a 243 MiB tensor, and it agrees with the scalar reference on every
shape the model uses (3, 8 and 16 blocks per row):

| | µs | GB/s | of 637.8 |
|---|---|---|---|
| interleaved | 580.5 | 439.7 | 69% |
| **planar** | **499.9** | **510.6** | **80%** |
| speedup | | **1.16×** | |

### In the engine it is worth nothing

| | tok/s (3 runs) | mean |
|---|---|---|
| interleaved | 102.56, 103.31, 103.27 | **103.1** |
| planar | 100.80, 102.52, 102.58 | 102.0 |

No better, and 18.3 s of host-side repack at load against 3.8 s. The engine runs
interleaved; the planar path is kept, tested and documented, because the finding
is worth more than the code.

The detail that explains it: the `output+argmax` phase — the one place a single
243 MiB projection dominates, the exact shape the benchmark measured 1.16× on —
did not move. 0.495–0.500 ms planar against 0.498–0.502 interleaved.

### Which means even the 243 MiB benchmark was cache-assisted

This card has **64 MB of Infinity Cache**. A 50-iteration loop over 243 MiB keeps
about a quarter of it resident, and the planar layout's contiguous plane scans
benefit from that far more than a single cold pass does. §9 already caught this
at 6.6 MiB and compensated by going to 243 MiB. That was not enough.

**The rule this establishes for this repo: a repeated-iteration microbenchmark on
this card cannot be trusted at any tensor size.** Only whole-token timing can
decide a kernel change. `examples/splitk.rs` and `tests/planar.rs` both carry
that warning now.

There is a second, more interesting reading. Planar reaches 510 GB/s on one big
matrix, and the engine averages 260 GB/s across 2.5 GB of weight traffic. The
gap is not layout and not load width — it is that **most of the model is small
matrices**. 46.4 MB per layer arrives as tensors of 0.86 to 10.3 MB, where the
fixed costs of a launch, an occupancy ramp and a reduction dominate whatever the
bytes are doing.

### Final state, and the honest answer about 140

**101.1 – 102.1 tok/s, 1.06 – 1.07× llama.cpp's 94.4 – 95.6.** Tokens identical
to llama.cpp and to the CPU reference on all five agreement prompts, every run.

Three optimisations were built and measured against the engine in this round and
the last. All three were rejected by measurement:

| | standalone | in the engine |
|---|---|---|
| HIP graph capture | — | slower, then neutral |
| split-k, several waves per row | up to **1.52×** | 5–13% slower |
| planar Q6_K, 16-byte loads | **1.16×** | neutral |

140 tok/s needs 7.14 ms/token against today's 9.80–9.89, i.e. 355 GB/s sustained
on 2.5 GB of cold weight traffic. Nothing measured here reaches that outside a
cache-assisted benchmark, and the three most plausible kernel-level routes are
now closed.

**The remaining route is not a kernel.** Decoding two tokens per pass halves the
weight traffic per token: the same 2.5 GB serves two tokens, so 140 tok/s would
need ~180 GB/s, well under the ~260 already achieved. That is a change to the
engine's shape — a second KV position, activations in pairs, and a batch
dimension through every kernel — and it is the only path to 140 on this card that
the measurements support.

---

## 11. Batched decode: 212.7 tok/s, and where llama.cpp still wins

§10 concluded that the remaining route was not a kernel: read each weight once
for several tokens instead of once for one. That is now built and measured.

**What batching can share, and what it cannot.** The Q, K, V and O projections
(743.2 MB/token) and the output projection (255.3 MB) are the *same matrices* for
every sequence, so one read serves the whole batch. The routed experts
(1486.4 MB, 60% of the traffic) are not: each sequence picks its own 8 of 128 and
at batch 2 only ~3% of those picks coincide. Per-token traffic therefore falls to
1986 MB at batch 2 and 1736 at batch 4 — 1.25× and 1.43×, not 2× and 4×. That is
mixture-of-experts, not an implementation limit.

**It is aggregate throughput, not latency.** Token N+1 of a sequence depends on
token N, so the sequences batched here are independent — separate streams, or
speculative candidates. Each individual sequence still sees batch-1 latency.

### Measured, 64 decode steps, warmed

| batch | ms/pass | ms/token | **tok/s total** | vs b=1 |
|---|---|---|---|---|
| 1 | 10.72 | 10.722 | **93.3** | 1.00× |
| 2 | 14.65 | 7.324 | **136.5** | 1.46× |
| 4 | 22.44 | 5.609 | **178.3** | 1.91× |
| 8 | 37.61 | 4.701 | **212.7** | 2.28× |

**Better than the bandwidth model predicted** (128 / 146 / 157), because batching
also fixes occupancy: the expert matrices are 0.86–10.3 MB and at batch 1 simply
do not have enough rows to fill 32 WGPs. Batching multiplies the row space.

Correctness is asserted against the single-sequence path at every width: batch 2,
4 and 8 each produce exactly the tokens eight separate `forward` calls produce.
A batched kernel that leaked one sequence's activation into another's accumulator
would still emit fluent text, so this comparison is the test.

Staging the activation in LDS was worth 3.5%: the global version re-reads each
sequence's activation for every output row, up to 64 times per workgroup at
batch 8. Kept, gated on the activation fitting in 24 KiB so occupancy survives.

### The honest comparison: llama.cpp batched, same card, same model

`llama-batched-bench -npp 1 -ntg 64 -npl 1,2,4,8`:

| batch | **llamd** | **llama.cpp** | |
|---|---|---|---|
| 1 | **93.3** | 69.9 | llamd **+33%** |
| 2 | 136.5 | **145.8** | llama.cpp +7% |
| 4 | 178.3 | **229.5** | llama.cpp +29% |
| 8 | 212.7 | **290.9** | llama.cpp +37% |

*(llama.cpp's batch-1 figure is 69.9 in this harness and 94.4–95.6 under
`llama-bench`; the harnesses differ in context allocation. Within
`llama-batched-bench` the two engines are measured identically.)*

**LLAMD wins at batch 1 and loses from batch 2 up.** Scaling tells the story:
llamd 1.46×/1.91×/2.28×, llama.cpp 2.09×/3.28×/4.16×.

The reason is structural, not a tuning gap. At batch 1 the problem is pure
bandwidth and a tight matrix-vector kernel is the right answer — which is why
llamd's batch-1 path, after §8's fusions, is 1.08× llama.cpp. At batch B the
arithmetic intensity rises by B and the right answer becomes a matrix-*matrix*
kernel: tile over rows *and* sequences, hold a weight tile in registers, reuse it
across every sequence in the tile. llama.cpp does that. LLAMD's batched kernel
still computes one row per wave with `nseq` accumulators, which shares the weight
*read* but not the weight *register*.

### State of the fork

| | llamd | llama.cpp | |
|---|---|---|---|
| single-stream decode | **102.98 tok/s** | 94.4 – 95.6 | **1.08×** |
| batch 8 aggregate | 212.7 | 290.9 | 0.73× |
| token agreement | 5/5 every run | — | exact |

The premise — that a kernel written for RDNA4 can beat CUDA hipified to it —
holds where it was tested: batch-1 decode, which is what a single user
experiences. It does not yet hold for batched serving, and the reason is
identified rather than mysterious.

**Next, concretely:** a row×sequence register-tiled Q6_K kernel. Each wave takes
4 rows and all `nseq` sequences, decoding a weight block once into registers and
consuming it 4×nseq times instead of nseq times. That is the change that turns
2.28× scaling into something nearer 4×, and it is the last structural gap between
this engine and llama.cpp.

---

## 12. The register-tiled kernel: udot4, and llamd ahead at batch 1 and 2

§11 named the gap precisely: at batch B the arithmetic intensity rises by B and
the right answer is a matrix-*matrix* kernel that reuses a weight tile across
sequences. llama.cpp does that; llamd was still computing one row per wave with
`nseq` accumulators, sharing the weight *read* but not the weight *register*.

### What RDNA4 actually has

`__builtin_amdgcn_sdot4` needs target feature `dot1-insts`, which gfx1201 does
not have. It has **`udot4` and `udot8` — unsigned**. Probed, not assumed.

That looks fatal: Q6_K weights are signed after the `−32`, and Q8_K activations
are signed int8. It is not, because of an identity. Q6_K stores `w` in 0..63
*before* the subtraction, and an int8 `a` becomes unsigned by `b = a + 128`,
which is exactly `a XOR 0x80`. Over a group of 16 values sharing one scale:

```
  Σ sc·(w−32)·a  =  sc · [ Σw·b  −  128·Σw  −  32·Σb  +  32·128·16 ]
```

- **`Σw·b`** — 4 `udot4` per 16 values, **per sequence**
- **`Σw`** — 4 `udot4` per 16 values, **shared by every sequence** ← the tile
- **`Σb`** — precomputed per activation group; Q8_K's `bsums` is this quantity

So 16 weights against `nseq` sequences cost `4·nseq + 4` dot instructions where
the previous kernel spent `16·nseq` MACs. At batch 8 that is **36 against 128**.

The identity was verified on device against the reference before the kernel was
written, including the signed and clamping cases.

**Planar is a precondition, which is what it was for.** Packing four weights
into a `uint32` needs four *consecutive* value indices, and they must share a
scale group. ggml's interleaved layout gives a lane two strided bytes per half;
the planar repack (§10, measured neutral on its own) gives it 16 consecutive
ones — exactly one scale group.

### Results

| batch | **llamd** | **llama.cpp** | |
|---|---|---|---|
| 1 | **103.76** | 94.4 – 95.6 | **llamd 1.09×** |
| 2 | **148.36** | 145.84 | **llamd 1.02×** |
| 4 | 214.85 | 229.51 | llama.cpp 1.07× |
| 8 | 279.69 | 290.90 | llama.cpp 1.04× |

Against §11's kernel: batch 2 136.5 → 148.4, batch 4 178.3 → 214.9, batch 8
212.7 → 279.7 — **+31% at batch 8**. Every batch width still produces exactly
the tokens the single-sequence path produces, and the planar and udot kernels
now agree on all eight test prompts.

**udot4 loses at batch 1** (96.0 against planar's 103.8) and the engine uses
planar there. A single sequence is purely bandwidth-bound, so the packing and
the correction terms are added instructions with nothing to amortise them over.
The crossover is between batch 1 and 2.

Fusing the bias-and-sum pass into the quantizer was worth the rest: it had been
a separate kernel at each of the five quantization points in a layer, 240 extra
launches per batched pass, plus a full read and write of the activation.

### A bug worth recording, because of how it presented

Mid-refactor, batched output became fluent nonsense — `"The"` produced `"arf"`,
then `"-no"` on a rerun of the same input. Non-determinism says uninitialised or
aliased memory, not arithmetic, and that is what it was: two `rms_norm_quant`
call sites failed to be replaced while the edit that *removed* the `bias_sums`
after them succeeded, so the expert path consumed the biased activation left
over from the previous `attn_out` quantization.

What made it cheap to find was refusing to guess. The fused quantizer and the
fused norm+quantizer were each compared against `quantize_q8_k` directly and
both were exact, which moved suspicion off the kernels and onto the wiring in
one step. **A kernel that is correct in isolation and wrong in the engine is a
plumbing bug**, and the way to tell them apart is a test at the boundary.

It also exposed a weakness in the batch test: it compared the batched path
against `forward()`, which after this change runs a *different kernel*. The
reference is now `forward_batch` at `nseq=1`, holding the kernel fixed and
varying only the batch width, with the cross-kernel comparison reported
separately.

### Row tiling: built, measured, neutral

One row per wave means every row re-reads the whole activation — for `attn_q` at
batch 8 that is ~67 MB of activation reads per layer against 6.9 MB of weights.
Cache-resident, so it does not appear as bandwidth; it appears as load
instructions, which is what remains after `udot4` cuts the arithmetic.

A templated `UR`-rows-per-wave kernel loads each activation group once and
consumes it `UR` times. In the engine at batch 8: **tile 1 → 279.18, tile 2 →
280.37, tile 4 → 278.65 tok/s.** Inside the run-to-run spread. Default stays 1.

The honest caveat: tiling is wired only into `uk_matvec`, which serves
`attn_out` and the output projection. `x3` and the grouped expert kernels carry
most of the traffic and are not tiled. **Extending it there is the open work**,
and the sweep stays in the test suite so the answer keeps coming from the engine.

### Where the fork stands

| | llamd | llama.cpp | |
|---|---|---|---|
| single-stream decode | **103.76 tok/s** | 94.4 – 95.6 | **1.09×** |
| batch 2 | **148.36** | 145.84 | **1.02×** |
| batch 4 | 214.85 | 229.51 | 0.94× |
| batch 8 | 279.69 | 290.90 | 0.96× |
| token agreement | exact, every run | — | |

An RDNA4-native engine written in Rust is ahead of hipified CUDA at batch 1 and
batch 2, and within 4–6% at batch 4 and 8 — from 0.004× four commits ago. The
remaining gap is one identified piece of work, not a mystery.

---

## 13. The Vulkan backend: Vulkan against ROCm on identical hardware

LLAMD now has a second GPU backend, and Vulkan is the default. The reason to
build it is in §12's shape: the engine had become tightly coupled to one vendor's
runtime, and the interesting question — whether an AMD-native engine can be
optimised harder through the vendor-neutral compute API the card exposes — could
not be asked without a second path.

### It computes the same thing

`shaders/q6k_matvec.comp` implements the identity from §12 over the same planar
layout the ROCm backend uses. RADV reports
`integerDotProduct4x8BitPackedUnsignedAccelerated = true`, so
`dotPacked4x8AccSatEXT` is the hardware instruction, not an emulation.

Validated against `llamd_quant::dot_q6k_q8k` at every block count the model uses
and every batch width:

| rows × k | blocks | nseq | worst deviation | bound |
|---|---|---|---|---|
| 256 × 768 | 3 | 1 | 2.44e-4 | 6.45e-3 |
| 1024 × 2048 | 8 | 1 | 4.88e-4 | 1.12e-2 |
| 512 × 4096 | 16 | 1 | 9.77e-4 | 1.55e-2 |
| 1024 × 2048 | 8 | 4 | 4.88e-4 | 1.15e-2 |
| 256 × 2048 | 8 | 8 | 4.88e-4 | 1.20e-2 |
| 7 × 2048 | 8 | 1 | 3.66e-4 | 1.07e-2 |

And against the ROCm backend directly: **worst 9.77e-4 against a 1.17e-2 bound
over 4096 values.** Each sequence gets a different activation, so a shader that
used sequence 0's for all of them would fail rather than pass.

### A fairness correction made before reporting any number

The first comparison had Vulkan 2.0–2.7× ahead, and part of that was not the
backend. Writing the shader made an improvement obvious that the HIP kernel did
not have: the packed word already holds the four weights in the order the dot
product wants, so **one mask-and-shift transforms all four at once** where the
HIP kernel extracted each of sixteen bytes separately — four operations instead
of sixteen.

That is better code, not a better API, so it was ported back to HIP before
measuring. It improved ROCm on `expert_down` from 32.5 to ~20 µs.

### The comparison, three runs, same card and same planar bytes

| shape | size | Vulkan µs | ROCm µs | |
|---|---|---|---|---|
| `attn_q` [2048→4096] nseq 1 | 6.6 MiB | **7.1 – 7.4** | 28.3 – 29.7 | **Vulkan 3.9 – 4.2×** |
| `attn_q` [2048→4096] nseq 8 | 6.6 MiB | **19.2 – 19.9** | 35.4 – 43.7 | **Vulkan 1.8 – 2.3×** |
| `attn_out` [4096→2048] nseq 8 | 6.6 MiB | **18.8** | 32.1 – 34.8 | **Vulkan 1.7 – 1.8×** |
| `expert_down` ×8 nseq 1 | 9.8 MiB | **13.3 – 13.4** | 19.9 – 22.5 | **Vulkan 1.5 – 1.7×** |
| `output` [2048→151936] nseq 1 | 243 MiB | 449 – 470 | **422 – 429** | ROCm 1.04 – 1.11× |
| `output` [2048→151936] nseq 8 | 243 MiB | 626 – 644 | 632 – 642 | parity |

The pattern is consistent and interpretable. **On the one genuinely HBM-bound
shape the two backends are equivalent** — same memory system, same bytes, and
neither API changes physics. On the smaller, cache-resident shapes Vulkan is
1.5–4× faster, plausibly from RADV's wider subgroup (64 against wave32, so one
subgroup per row costs proportionally less reduction) and its scheduling.

**And the caveat that this repo has earned the hard way applies to both columns:**
a repeated dispatch over a 6.6 MiB tensor on a card with 64 MB of Infinity Cache
measures cache bandwidth. §9 and §10 record two optimisations that looked
excellent under exactly this kind of measurement and were rejected by the engine.
The shapes above are labelled with their size for that reason, and the
engine-level verdict is not in yet.

### What is NOT done, stated plainly

**End-to-end inference still runs on ROCm.** The Vulkan backend has *one* of the
roughly fourteen kernels a token needs — the Q6_K matvec, which is the hardest
and the one carrying most of the traffic, but one. Norms, RoPE, attention, the
router, the MoE combine, the argmax and the quantize/bias shaders exist only in
HIP, so `DeviceModel` is still a ROCm path.

So "Vulkan is the default backend" is true of the selection policy and of the
kernel that is implemented and validated. It is **not** yet true that a token is
computed on Vulkan. The honest reading of the table above is *a strong signal on
the dominant kernel*, and the next increment is the remaining shaders — after
which a whole-token Vulkan-versus-ROCm number can be put beside llama.cpp's.

---

## 14. Vulkan end to end: ahead of llama.cpp at every batch size

§13 delivered the Vulkan backend with one kernel and said plainly that a token
was not yet computed on it. The remaining seventeen shaders are now written, and
`VulkanModel` computes a whole token on Vulkan: weights resident, activations
never crossing the bus, every operation recorded without waiting, **one
submission per token**, and only the argmax read back.

### Correctness first

| prompt | Vulkan | llama.cpp | |
|---|---|---|---|
| `"Hi"` | 525 `" are"` | 525 | ok |
| `"The"` | 525 `" are"` | 525 | ok |
| `"def"` | 374 `" is"` | 374 | ok |
| `"Python"` | 25 `":"` | 25 | ok |
| `"1"` | 13 `"."` | 13 | ok |

**All five, including the two near-ties** (margins 0.1196 and 0.1946) that the
q8_0-KV experiment in §1 showed are sensitive enough to flip. Batch 2 and batch 4
produce exactly what `nseq=1` produces, so batching is verified against the same
kernel rather than across a kernel boundary.

It generates coherent text, not one matching token:

> `" are you familiar with the concept of \"the 10,000 hour rule\"?\nYes, I'm
> familiar with the \"10,000 hour rule.\" It's a concept popularized by Malcolm
> Gladwell in his 2008 book *Outliers*. The rule suggests that it"`

### Throughput, three runs, warmed, 64 decode passes

| batch | **Vulkan** | llamd ROCm | llama.cpp | vs llama.cpp | vs ROCm |
|---|---|---|---|---|---|
| 1 | **111.5 – 114.6** | 103.8 | 94.4 – 95.6 | **1.17 – 1.21×** | 1.07 – 1.10× |
| 2 | **183.6 – 186.5** | 147.6 | 145.8 | **1.26 – 1.28×** | 1.24 – 1.26× |
| 4 | **258.1 – 265.6** | 214.8 | 229.5 | **1.12 – 1.16×** | 1.20 – 1.24× |
| 8 | **396.2 – 399.9** | 279.6 | 290.9 | **1.36 – 1.37×** | 1.42 – 1.43× |

**Vulkan is ahead of hipified CUDA at every batch size, and ahead of LLAMD's own
ROCm path at every batch size.** The batch-8 figure is the one §12 could not
reach: llama.cpp led there by 4% and now trails by 37%.

This also closes the question §11 opened. The gap at batch was blamed on
matrix-matrix tiling, and the fix turned out to be elsewhere — the same
`udot4`-equivalent identity, the same planar layout, expressed through an API
whose subgroup is 64 wide and whose driver schedules it better.

### What the Vulkan path is made of

Eighteen compute shaders: the Q6_K matvec, two grouped expert variants (shared
activation for gate and up, per-`(sequence, expert)` activation for down), an F32
matvec for the router, the embedding dequantizer, `rms_scale`, the fused
norm-and-quantize, a standalone quantize, `rms_norm_mul`, per-head norm with NeoX
RoPE, the K-side variant that also writes the cache, attention with GQA, `add`,
`silu_mul`, the router's softmax and top-k, the MoE combine, and a two-stage
argmax.

Three structural decisions carried the work:

**One descriptor layout for every shader.** Eight storage buffers pushed inline
with `VK_KHR_push_descriptor` plus 64 bytes of push constants. No descriptor
pool, nothing to recycle, and adding a shader is adding a file — a token is ~15
dispatches per layer and allocating a set for each would have dominated.

**The Q6_K decode lives in one place.** GLSL has no buffer pointers and the
weight tensor differs per shader, so the arithmetic is a macro in `q6k.glsl`
instantiated against whichever buffer a shader declares. Four hand-written copies
would have been four places for a divergence that flips a near-tied expert.

**The headers are split by what they cost.** `quant.glsl` brings 3 KiB of shared
memory the matvec shaders must not pay for; `q6k.glsl` needs a `uint w[]` binding
a quantize shader does not have. Appending everything to one header broke three
shaders and silently taxed the rest.

### Two bugs the design caught rather than shipped

**One field, two meanings.** The Q6_K macro loops `_s < pc.nseq` while the expert
shaders need exactly one sequence per row — a row belongs to one
`(sequence, expert)` pair — and the row-space bound needs the real batch width.
Sharing one push field made those disagree. The sequence count is now an explicit
macro parameter, documented as load-bearing.

**One command buffer, two users.** `upload` and `download` record into the same
command buffer the token is being recorded into, so calling either between
`begin` and `end_and_wait` would reset it and discard a half-built token —
producing plausible wrong output rather than an error. Both now refuse while
recording, `begin` refuses to nest, and a dispatch outside a recording is
refused. Three tests assert it.

### Still true, and worth repeating

The ROCm path is unchanged and unregressed at 102.0 tok/s, and remains the
benchmarking reference: both backends implement the same identity over the same
planar bytes and are validated against the same scalar reference, which is what
makes the column above a measurement of the backend.

The CPU path remains the correctness reference — the one that agrees with
llama.cpp token for token, and the one every device kernel is held to.

---

## 15. Batched prefill: the gap that mattered most

Every number before this one measured decode. That made the engine look better
than it was: without prefill a prompt had to be fed **one token at a time**, so a
512-token prompt read all 2.485 GB of active weights 512 times.

```
  512-token prompt, before:   5.83 s    (87.9 tok/s)
  512-token prompt, after:    1.05 s    (485.9 tok/s)     5.6x
  llama.cpp pp512:            0.31 s    (1632 tok/s)
```

### Prefill is batching with a shared cache

The whole layer loop is the same code `forward_batch` already ran. Every token in
the tile multiplies the same weights, so one weight read serves all of them — the
sharing that made batch-8 decode fast is exactly the sharing a prompt needs.

The only difference is that the slots are **positions in one sequence** rather
than separate streams, so they share the KV cache instead of owning slices of it.
Expressed as one push constant, `kv_stride`: `max_kv * kv_dim` for independent
sequences, **0** for prefill. Two shaders changed; none were added.

**Causality needs no mask.** Each slot's `n_kv` is its own position plus one, so
token *i* reads exactly positions 0..pos_i and never a later one. Every slot
writes its K and V before the attention dispatch — the barrier between them
guarantees it — so what token *i* reads is the earlier tokens of the same tile.

Non-final tiles skip the final norm, the output projection and the argmax. The
output projection is 255 MB, 10% of a token's traffic, and every token's logits
but the last are discarded.

### The tile is the whole story

| tile | 512 tokens | tok/s |
|---|---|---|
| 1 | 5.827 s | 87.9 |
| 4 | 2.119 | 241.7 |
| 8 | 1.429 | 358.3 |
| 16 | 1.087 | 471.0 |
| **32** | **1.054** | **485.9** |

Raising `MAX_SEQ` from 8 to 32 was worth 36%. The returns flatten from 16 to 32
(+3%), which says the limit has moved: the routed experts do not share across
tokens the way the attention projections do, and attention cost grows with
context.

| prompt | prefill | tok/s | one at a time | |
|---|---|---|---|---|
| 64 | 0.104 s | **613.7** | 119.4 | 5.14× |
| 128 | 0.209 | **611.1** | 112.5 | 5.43× |
| 256 | 0.466 | **549.9** | 101.0 | 5.45× |
| 512 | 1.050 | **487.8** | 87.4 | 5.58× |

Shorter prompts ingest faster per token because attention cost grows with `n_kv`.
A prompt shorter than the tile is slower per token again — 11 tokens managed only
232 tok/s — because the tile is not full and the fixed costs per pass dominate.

### Correctness: equivalence, not plausibility

A prefill that gets causality wrong — a token seeing one that follows it — still
produces fluent text and would pass any eyeball test. So the bar is that prefill
returns **exactly** what sequential decode returns:

**20 (tile, length) cases, tiles 1 through 32**, each straddling a tile boundary
— below it, exactly it, one over, and a ragged last chunk. All exact.

The intermediate predictions are visibly sensible too, which is a useful smell
test on top: *"The quick brown fox jumps over the"* → `" lazy"` → `" dog"` → `"."`.

And the two paths compose — generation continues from the cache prefill built:

> `" The sky appears blue due to a phenomenon called Rayleigh scattering. When
> sunlight enters Earth's atmosphere, it collides with tiny gas molecules,
> primarily nitrogen and oxygen. Sunlight contains all colors of the rainbow, but
> blue light has a shorter wavelength"`

### Where this leaves llamd against llama.cpp

**Prefill is still 3.3× behind** — 487.8 tok/s against 1632. llama.cpp processes
the whole prompt in one GEMM; llamd processes it in tiles of 32 with a
matrix-vector kernel that happens to carry 32 accumulators.

That makes the comparison depend on the shape of the request:

| | llamd | llama.cpp |
|---|---|---|
| 512-token prompt | 1.05 s | 0.31 s |
| then 128 tokens out | 1.13 s | 1.35 s |
| **total** | **2.18 s** | **1.66 s** |

llama.cpp wins that one. The break-even is around **440 generated tokens**: below
it llama.cpp's faster prefill dominates, above it llamd's faster decode does.

The honest summary is that llamd now has a prefill path rather than a good one.
The next step is the one llama.cpp already took — a real tiled GEMM over tokens,
with the activation staged in shared memory and specialization constants so the
accumulator loop is compile-time bounded and stays in registers rather than
scratch.

---

## 16. The tiled GEMM: built, measured, and the wrong target

§15 named a tiled GEMM as the way to close prefill's 3.3× gap to llama.cpp. It is
built and validated, it is **slower than the kernel it replaces**, and the more
useful result is why.

### Three variants, all slower

On `attn_q` [2048→4096], kernel-only, against the plain matvec:

| variant | scratch | vs matvec |
|---|---|---|
| a lane owns (row, group) | 0 | 1.09 – 1.68× slower |
| 4 rows per lane | **32 KB** | 1.45 – 4.59× slower |
| 2 rows per lane | 0 | 1.09 – 1.76× slower |

All correct — validated against the scalar reference at 3, 8 and 16 blocks per
row, at 1 to 32 tokens, and at row counts that do not divide the tile (37, 7).

### What each variant ruled out

**The first was a parallelism error.** Giving a lane one (row, group) pair means a
row is covered by 8 lanes instead of 64 — it bought 8× activation reuse by giving
up 8× parallelism.

**The second was registers, and measurement said so directly.** RADV reported
`Scratch size: 32768`: `acc[4][32]` is 128 floats indexed by a loop whose bound is
a push constant, so the compiler cannot unroll it and the array goes to scratch.
Worth noting that the *original* matvec's `acc[32]` does **not** spill —
`Scratch size: 0, Spilled VGPRs: 0, VGPRs: 72`. Registers were never the matvec's
problem, so the specialization-constant fix §15 proposed would have fixed nothing.

**The third ruled out the remaining explanation.** Two rows per lane does not
spill, and has a strictly better instruction mix than the matvec — 16 dot4 per 8
activation loads against 8 per 8. It is still slower.

So not registers, and not instruction count. **What is left is latency.** The
matvec runs one row per subgroup at 72 VGPRs, which puts 262144 lanes on
`attn_q` — two waves per hardware slot, so a stalled wave always has a partner to
hide behind. Halving the subgroup count to buy fewer instructions removes exactly
the latency hiding that was paying for them.

### The premise was wrong anyway, and the earlier data said so

This kernel serves the attention projections. At 32 tokens those are **3.5% of a
prefill tile's weight traffic**:

| tokens | distinct experts | expert MB/layer | attention % | expert % |
|---|---|---|---|---|
| 1 | 8 | 31 | 33.3% | 66.7% |
| 8 | 52 | 200 | 7.2% | 92.8% |
| 16 | 82 | 319 | 4.6% | 95.4% |
| **32** | **112** | **433** | **3.5%** | **96.5%** |
| 512 | 128 | 495 | 3.0% | 97.0% |

The attention weights are read once per tile however many tokens it holds. The
routed experts are not: each token picks its own eight of 128, and at 32 tokens
112 distinct experts get touched. **A perfect kernel here would win 3.5%.**

§15's own tile sweep had already said this and it was not read carefully enough:
16 → 32 tokens gained **3%**. That flattening is the signature of a cost that does
not share, and it was pointing at the expert path.

### What would actually work

At 32 tokens, 112 distinct experts serve 256 token-expert pairs — **2.3 tokens per
chosen expert**. Today the grouped kernels run one row per `(token, expert)` pair
with `NS = 1`, so an expert's weights are read once per token that chose it.

Grouping **expert-major** instead — sort the tile's tokens by expert, gather their
activations, run one matmul per expert against all its tokens, scatter the results
— reads each expert's weights once and serves 2.3 tokens with them. That is ~2.3×
on 96.5% of the traffic, against 3.5% for anything done to the attention path. It
is also what `ggml_mul_mat_id` does.

The kernel and its tests are kept, as the HIP graph, split-k and planar
experiments were, because the finding is the deliverable: **the matvec's
decomposition is already near-optimal for a latency-bound kernel, and prefill is
limited somewhere else entirely.**

A note on method. This is the fourth optimisation in this repo that measured well
in theory and lost in practice, and the pattern across all four is the same:
arithmetic about bytes or instructions predicted a win, and occupancy or latency
decided otherwise. The arithmetic that *did* hold up was the traffic breakdown
above — which was available before the kernel was written, and would have said
not to write it.

---

## 17. Expert-major grouping: prefill 598.5 tok/s

§16 got the target wrong and said so: the tiled GEMM optimised the attention
projections, which are 3.5% of a prefill tile's weight traffic, while the routed
experts are 96.5%. This is the fix that follows from that arithmetic.

### The idea

Token-major runs one row per `(token, expert)` pair, so an expert chosen by three
tokens has its weights read three times. Expert-major sorts the tile's pairs by
expert and reads each expert's weights ONCE for every token that chose it. At 32
tokens, 112 distinct experts serve 256 pairs — **2.3 tokens per read**.

**The grouping has to happen on the device.** The routing is chosen by the router
shader, so it only exists on the GPU; reading the expert ids back to sort on the
host would mean a submit-and-wait per layer, 48 per tile, which is the stall the
whole engine is built to avoid. `moe_group.comp` is a counting sort in ONE
workgroup — at most 256 pairs over 128 experts, so shared-memory atomics for the
histogram and the scatter, one thread for the prefix sum, no global
synchronisation.

The dispatch then uses `gl_WorkGroupID.y` as the expert id, with empty groups
returning immediately. That avoids needing the group count — which only exists on
the device — back on the host to size the grid. At 32 tokens only 16 of 128
workgroups are empty.

### Results, A/B in one process

| tile | token-major | expert-major | |
|---|---|---|---|
| 8 | 348.6 t/s | 351.2 | 1.01× |
| 16 | 465.8 | **520.9** | 1.12× |
| 32 | 481.7 | **591.3** | **1.23×** |

And in decode, where a pass holds one token and no expert is shared, it is a
**1.41× loss** — 122.3 tok/s token-major against 86.5 — because the grouping
dispatch and the gathered activation reads are then pure overhead.

So the policy is a threshold, set from the measurement: expert-major at **16 or
more tokens per pass**, token-major below. Prefill tiles take one path and decode
the other.

### Prompt ingestion, end to end

| prompt | before §15 | after §15 | **now** | vs one-at-a-time |
|---|---|---|---|---|
| 64 | — | 613.7 | **810.7** | 7.10× |
| 128 | — | 611.1 | **774.4** | 7.32× |
| 256 | — | 549.9 | **695.7** | 6.98× |
| 512 | 87.9 | 487.8 | **598.5** | 6.99× |

A 512-token prompt: **5.83 s → 0.855 s.**

### Correctness

A gather/scatter that feeds the wrong token's activation to an expert, or writes a
result into the wrong slot, still produces fluent text. So the bar is that the two
groupings agree **exactly**: every tile width from 1 to 32, at lengths straddling
each boundary, plus a decode step where the grouping degenerates to 8 groups of 1
— the case most likely to expose an off-by-one in the prefix sum. All exact. And
prefill still equals sequential decode across 20 (tile, length) cases, and decode
still matches llama.cpp on all five prompts.

### Where llamd now stands against llama.cpp

| | llamd | llama.cpp |
|---|---|---|
| decode, 1 stream | **109 – 113** | 94.4 – 95.6 |
| decode, batch 8 | **397** | 290.9 |
| **prefill (pp512)** | 598.5 | **1632** |

Prefill is still **2.7× behind**, improved from 3.3×. For a request of a 512-token
prompt and 128 generated tokens, llama.cpp is still ahead overall — 1.66 s against
1.99 s — and the **break-even is now 323 generated tokens**, down from 441.

What remains on the prefill path is the part expert-major does not address: an
expert serving 2.3 tokens still runs a matrix-VECTOR kernel 2.3 times rather than
one matrix-matrix kernel over 2.3 columns. §16 established that widening that
kernel costs more in latency hiding than it saves in instructions at this tile
size, so the next thing worth trying is a larger tile — which needs the
accumulator loop compile-time bounded through specialization constants, and is now
a measurement about the expert path rather than the attention path.

---

## 18. Specialization constants: a wider tile without the register cost

§17 ended by naming the next experiment: a larger prefill tile, which needs the
accumulator loop compile-time bounded. This is that experiment, and it produced
two results — one expected, one not.

### The obstacle, measured before it was fixed

Every matvec-shaped shader holds its per-token partial sums in `float acc[MAX_SEQ]`.
GLSL has no variable-length arrays, so `MAX_SEQ` is the widest batch the engine
will ever run, and **every** dispatch pays for it — a batch-1 decode allocates 32
(now 64) registers and uses one.

Raising `MAX_SEQ` from 32 to 64, so a 64-token tile could exist at all, was
therefore a regression before it was an improvement:

| | prefill, tile 32 |
|---|---|
| `MAX_SEQ 32` | 598.5 t/s |
| `MAX_SEQ 64`, unspecialized | **541.7 t/s** |

`RADV_DEBUG=shaderstats` explains it exactly: 192 VGPRs, which on gfx1201 is
**one wave per SIMD** instead of three. The kernel is latency-bound — it waits on
memory — so occupancy is the thing it cannot afford to lose.

### The fix

```glsl
layout(constant_id = 0) const int ACC_N = 64;
...
float acc[ACC_N];
for (int i = 0; i < ACC_N; ++i) { if (i >= count) break; ... }
```

A specialization constant is substituted at pipeline-creation time, so the array
is sized and the loop unrolled per *variant*, from one source. The host builds
seven variants of each of the four shaders — `ACC_N` ∈ {1, 2, 4, 8, 16, 32, 64} —
and `dispatch_acc` picks the narrowest variant that fits the call. Register cost
now spans **12 to 192 VGPRs** where it was flat at 192.

### What it is worth, A/B in one process

`ops::set_acc_specialized(false)` forces every dispatch to the widest variant,
which is the unspecialized engine exactly. Alternating arms, best of 3:

| batch | specialized | widest variant | |
|---|---|---|---|
| 1 | **126.46 t/s** | 58.41 t/s | **2.165×** |
| 2 | **200.24** | 102.88 | 1.946× |
| 4 | **276.89** | 167.39 | 1.654× |
| 8 | **410.96** | 276.04 | 1.489× |

The gain shrinks with batch size, as it should: at batch 8 the wide accumulator is
half used, at batch 1 it is 1/64 used.

This also settled a false alarm. Decode read ~105 t/s in a single run just after
the `MAX_SEQ` change and looked 4% down. The card was idle (21 °C, 9 W), so it was
not thermal — and best-of-3 puts batch 1 at 126.46 t/s, **above every earlier
reading in this document**. The dip was single-run noise, not a regression.

### And the answer to §17's actual question: tile 64 loses

With the register cost removed, the tile sweep could finally be run honestly:

| prefill tile | tok/s |
|---|---|
| 16 | 542.3 |
| **32** | **638.7** |
| 64 | 551.0 |

**Tile 64 loses despite sharing 1.77× more expert weight reads** — 4.07 tokens per
chosen expert against 2.29 at tile 32. The widest variant is still the 192-VGPR
one, so the tile that most needs sharing is also the tile with the least
occupancy, and on a latency-bound kernel that trade goes the wrong way. Tile 32
stays the default. The experiment §17 proposed was worth running and the answer
was no.

### A cache-size consequence that had to be fixed first

A prefill tile shares one KV slice (`kv_stride = 0`), so the number of tile slots
and the number of KV slots are different quantities. They had been the same field.
At 64 they would have allocated **7.55 GB** of KV cache against 0.94 GB at 8, on a
31.9 GiB card holding a 23.37 GiB model — it would not have fit. `load_with_slots`
now takes them separately.

### Prompt ingestion, end to end

| prompt | §15 | §17 | **now** | vs one-at-a-time |
|---|---|---|---|---|
| 64 | 613.7 | 810.7 | **889.0** | 7.79× |
| 128 | 611.1 | 774.4 | **843.7** | 7.97× |
| 256 | 549.9 | 695.7 | **751.4** | 7.54× |
| 512 | 487.8 | 598.5 | **639.1** | 8.38× |

A 512-token prompt: **5.83 s → 0.801 s.**

### Correctness

Specialization changes which binary runs, per dispatch, per batch width — exactly
the kind of change that can be right at one width and wrong at another. So the
whole ladder is re-checked: `vulkan_oracle` 6/6 against the scalar reference,
prefill equal to sequential decode across every tile width from 1 to 32,
expert-major equal to token-major, and all five prompts still token-for-token
equal to llama.cpp. 131 passed, 0 failed, 18 ignored.

### Where llamd stands against llama.cpp

| | llamd | llama.cpp |
|---|---|---|
| decode, 1 stream | **113 – 126** | 94.4 – 95.6 |
| decode, batch 8 | **397 – 411** | 290.9 |
| **prefill (pp512)** | 639.1 | **1632** |

Prefill is **2.55×** behind, from 3.3× at §15. Decode is ahead by 1.18–1.33× at one
stream and 1.37–1.41× at batch 8.

---

## 19. Where the prefill gap actually is, measured

§18 left prefill 2.55× behind on a llama.cpp figure of 1632 t/s that was quoted
from an earlier run. That figure was wrong, and the real gap is larger. This
section replaces it and decomposes it.

### llama.cpp is not one number — it is a curve in n_ubatch

Same card, same model, `llama-bench -p 512 -n 0`:

| n_ubatch | llama.cpp ROCm | llama.cpp Vulkan | llamd (tile) |
|---|---|---|---|
| 32 | 523.8 | 676.1 | **656.4** |
| 64 | 673.8 | — | 550.8 |
| 128 | 778.2 | — | — |
| 256 | 1540.4 | — | — |
| 512 | 2040.6 | **3213.6** | — |

**At matched width, llamd is competitive**: 656.4 against llama.cpp Vulkan's
676.1 at ubatch 32 — 3% behind — and 25% *ahead* of its ROCm path. The deficit
is not a worse kernel. It is that llama.cpp can run a 512-wide ubatch and llamd
cannot get past 32.

### Why llamd cannot widen, in one number

Weight traffic to ingest 512 tokens, from the model geometry (one expert-layer is
3.87 MB, attention is 15.48 MB/layer, 48 layers, 8 of 128 experts per token, and
`E(p) = 128(1-(127/128)^p)` distinct experts for `p` routed pairs):

| tile | tokens/expert | traffic | measured | achieved GB/s | % of 637.8 |
|---|---|---|---|---|---|
| 8 | 0.50 | 648.5 GB | 1.381 s | 470 | 74% |
| 16 | 1.00 | 506.2 GB | 0.909 s | **557** | **87%** |
| 32 | 2.00 | 341.6 GB | 0.780 s | 438 | 69% |
| 64 | 4.00 | 193.0 GB | 0.930 s | **208** | **33%** |
| 128 | 8.00 | 98.3 GB | — | — | — |
| 256 | 16.00 | 49.3 GB | — | — | — |
| 512 | 32.00 | **24.8 GB** | — | — | — |

Traffic falls 13.8× from tile 32 to tile 512 — and **achieved bandwidth falls
with it**, from 87% of peak at `ACC_N=16` to 33% at `ACC_N=64`. That is the
`float acc[ACC_N]` array of §18: it is the thing that makes a wide tile
worth having, and the thing that makes a wide tile impossible. Tile 64 has 1.8×
less traffic than tile 32 and is still slower, because it moves that traffic at
half the rate.

### How much is tiling, and how much is hardware

`GGML_VK_DISABLE_COOPMAT=1` answers this directly:

| llama.cpp Vulkan, pp512, ub 512 | t/s |
|---|---|
| with `KHR_coopmat` | **3213.6** |
| without | 1834.3 |

The device reports `matrix cores: KHR_coopmat` and `int dot: 1`. So the 4.9× gap
factors cleanly:

- **2.8× is tiling** — a wide ubatch and a matrix-matrix kernel, no special
  hardware. llama.cpp reaches 1834 t/s this way.
- **1.75× is cooperative matrix** on top of that.

2.8 × 1.75 = 4.9, which is the gap.

### What that means for the work

The budget is not tight once the tile is wide. At tile 512 the traffic is 24.8 GB,
so **3213 t/s needs only 156 GB/s** — a third of what the tile-32 kernel already
achieves. The binding constraint becomes arithmetic: 2.78 TFLOP for 512 tokens,
so 3213 t/s is **17.5 TFLOPS effective**, against roughly 66 TFLOPS of `udot4`
ceiling on this part. llamd is at 3.56 TFLOPS today.

So the order is forced, and it is the opposite of what §16 tried:

1. **Widen the tile by moving the token dimension out of the per-lane accumulator
   array** — into lanes and LDS, i.e. a real register-tiled GEMM, this time on the
   expert path (96.5% of traffic) rather than the attention path (3.5%) that §16
   spent itself on. Expected: 654 → ~1800 t/s.
2. **Then `VK_KHR_cooperative_matrix`.** It is worth 1.75× to llama.cpp here and
   is the one mechanism this device exposes that llamd does not touch. Expected:
   ~1800 → ~3200 t/s, parity.
3. **To pass it**, the candidate is int8 coopmat fed straight from the planar
   Q6_K layout. llama.cpp's coopmat path dequantizes to fp16; the §V2 identity
   keeps the whole product integral, so int8 matrix ops would halve the operand
   bytes in LDS and, on RDNA4, run at roughly twice the fp16 matrix rate. This is
   unverified and is the first thing to check, not the first thing to build.

Doing 2 before 1 would measure nothing: at 69% of memory peak the ALU is not the
limit, which is the same mistake §16 made and the same reason it is recorded.

---

## 20. The wide-tile expert GEMM: built, and the two models it refuted

§19 laid out an ordered plan whose first step was worth 2.8×: widen the prefill
tile so each expert's weights are read once instead of once per 32-token tile.
Both halves of that were built. The first was worth **0%**, the second **3%**, and
what they ruled out is worth more than what they gained.

### What was built

**Token chunking, in the four matvec-shaped shaders.** `float acc[ACC_N]` no
longer bounds the tile: the shaders loop the token dimension in chunks of `ACC_N`
internally, re-reading the weight row per chunk. `MAX_SEQ` (64) is now the
accumulator width and the new `MAX_TILE` (512) is the tile width — they had been
one number, which is what capped the tile at 32.

**A row-tiled expert kernel**, `q6k_expert_rt.comp`, the 24th pipeline. The
activation index `qb[ab]` depends on `(token, K)` and **not on row**, so with one
row per subgroup every activation load feeds exactly one row of dots. This one
hoists the token loop above a row loop: one load, `ROWS` rows of dots. Counted per
lane per K-group:

| | 1 row × 32 tokens | 4 rows × 16 tokens |
|---|---|---|
| weight loads | 8 | 32 |
| activation loads | 352 | 176 |
| work units | 32 | 64 |
| **loads per unit** | **11.25** | **3.25** |

### Result 1: the traffic model was wrong

| prefill tile | before | after chunking |
|---|---|---|
| 32 | 631.7 | 659.5 |
| **64** | **550.8** | **681.9** |
| 128 | — | 676.4 |
| 256 | — | 669.1 |
| 512 | — | 670.8 |

Chunking **removed the tile-64 cliff** — 550.8 → 681.9, and that part worked
exactly as intended. But the surface is then **flat from 64 to 512**, where §19's
model predicted 5.5× from the 13.8× drop in expert weight traffic. A 2-D sweep of
tile against chunk is flat in both:

| tile | chunk 8 | chunk 16 | chunk 32 | chunk 64 |
|---|---|---|---|---|
| 32 | 631.8 | 644.9 | 636.4 | 636.8 |
| 64 | 652.1 | **666.5** | 657.0 | 543.7 |
| 512 | 620.8 | 633.0 | 644.6 | 542.6 |

`RADV_DEBUG=shaderstats` rules out the usual suspects across all 161 pipeline
variants: VGPRs 12/24/48/72/96 with only the `ACC_N=64` variants at 192, **zero
spilled registers and zero scratch anywhere**. At chunk 32 that is 5–7 waves per
SIMD. And at tile 512 the engine moves 24.8 GB in 0.76 s — **31 GB/s, 5% of
peak**. So above tile 64 it is neither bandwidth-bound nor occupancy-bound.

### Result 2: the load-count model was wrong too

| tile | 1 row | 4 rows | |
|---|---|---|---|
| 32 | 638.0 | 663.1 | 1.039× |
| 64 | 662.6 | **681.4** | 1.029× |
| 512 | 645.0 | 670.1 | 1.039× |

Cutting inner-loop loads per unit of work by 3.5× is worth **3%**.

### What that proves, and it is the useful part

Two independent kernel-level improvements to the expert path — 13.8× less weight
traffic, 3.5× fewer loads — produced 3% between them. The expert matmuls cannot be
where the time is. §17's token-major A/B bounds it directly: token-major reads
2.3× more expert weight at tile 32 and cost 1.23× overall, so if `f` is the expert
matmuls' share of prefill time,

    f × 2.3 + (1 − f) = 1.23   →   f ≈ 0.18

**The routed expert matmuls are ~18% of prefill time.** They are 96.5% of its
weight traffic and 67% of its arithmetic, and they are 18% of its wall clock. No
amount of work on that kernel can exceed 1.2×, and two attempts have now returned
1.03× of it.

So §19's factorisation was wrong in its first term. The 2.8× that llama.cpp gains
between ubatch 32 and 512 without coopmat is not expert weight reuse — llamd now
has that reuse and gained nothing. It has to be the other 82%: the per-token
attention half, its norms, RoPE, the O(T²) scores, the quantize steps and the
combine, which in llamd are ~20 separate dispatches per layer over small kernels.

### Where this leaves the engine

| | before | after |
|---|---|---|
| pp512 | 654.6 t/s | **680.3** (+3.9%) |
| pp256 | 851.8 | **880.6** |
| pp128 | 992.5 | **1033.2** |
| pp64 | 1028.0 | **1079.6** |
| tile 64 | 550.8 | **681.9** (+23.8%) |
| vs one-at-a-time | 7.46× | **7.71×** |

Correctness held throughout: prefill equals sequential decode across **100**
(tile, length) cases with a 290-token prompt that actually exercises multi-chunk
paths — the previous 26-token prompt made every tile ≥32 a single chunk, so the
chunk loop was never tested twice; the row-tiled kernel is **exact** against the
one it replaces across 45 cases; 131 passed, 0 failed.

### The next step is a profiler, not another model

Two models have now predicted large gains and delivered ~0. Both were built on
counting — traffic, then instructions — and both were consistent with every number
available at the time. The thing neither could see is where 82% of the time goes.
That needs RGP or RADV performance counters against the real 48-layer pass, and
until it exists any further projection about prefill should be treated as what the
last two turned out to be.

---

## 21. The profile: attention is 61% of prefill

§19 and §20 each built a counting model, predicted a large prefill gain, and got
nothing. This is the measurement that should have come first.

### Method

`vkCmdWriteTimestamp` around every dispatch, read back after the pass, aggregated
by pipeline. **It is exact, not sampled**: every dispatch in this engine is already
followed by a full memory barrier, so nothing overlaps and the intervals sum to the
pass. That is also why the ~960 extra commands per token perturb almost nothing.
`ops::profile(true)` then `ops::profile_read()`; the test is
`vulkan_profile.rs::where_prefill_time_goes`.

### One 48-layer pass, 512-token prompt

| kernel | calls | tile 64 | tile 512 |
|---|---|---|---|
| **`attention`** | 48 | **61.7%** | **60.5%** |
| `q6k_expert_rt` | 144 | 23.6% | 23.6% |
| `q6k_matvec` | 49 | 6.6% | 7.2% |
| `q6k_x3` | 48 | 4.6% | 5.5% |
| `matvec_f32` (router) | 48 | 1.3% | 1.5% |
| the other 14 kernels | 580 | 2.2% | 1.7% |

**Attention is 61% of prefill and it does not move with the tile.** That is the
whole explanation of §20's flat surface: the thing being optimised was 23.6% of the
time, and the thing that dominates is invariant to tile width. The algebraic
estimate in §20 put the expert matmuls at 18%; measured, they are 23.6%.

### How far off attention is

Converting each kernel's real MAC count against its measured time:

| kernel | G MAC/s |
|---|---|
| `q6k_expert_rt` | 15473 |
| `q6k_x3` | 6127 |
| **`attention`** | **112** |

**55× slower per MAC than the Q/K/V projection**, and about 1.4% of this part's
fp32 FMA rate. It is not a tuning gap; the kernel is the wrong shape.

### Why, from the source

`attention.comp` is decode-shaped — written when a pass held one token, where none
of this mattered. Four defects, all of which make it latency-bound:

1. **The QK inner loop is a 128-deep dependent FMA chain.**
   `for (d) dot += q[qb+d] * kc[ko+d]` — one accumulator, so ~4 cycles per MAC
   with no instruction-level parallelism, ~512 cycles to do 128 MACs.
2. **The K reads are fully uncoalesced.** Lanes are mapped to KEYS, and
   consecutive keys are `kv_dim` = 512 floats apart, so one load instruction
   across 64 lanes touches 64 different cache lines.
3. **The query is re-read from global memory 128 times per thread.** It is 128
   floats and never staged in LDS or registers.
4. **The AV pass uses half the workgroup and is serial in keys.**
   `for (d = t; d < hd; d += 256)` with `hd` = 128 leaves 128 of 256 threads
   idle, and each surviving thread walks all `n_kv` keys in another single-
   accumulator dependent chain.

### What fixing it is worth

At tile 512 the pass is 761 ms of GPU time: 460 attention, 301 everything else.
Attention at even 2000 G MAC/s — still under a quarter of fp32 peak, and the
projections already reach 6127 — makes it 26 ms:

| | now | attention fixed |
|---|---|---|
| pp512 | 680 t/s | **≈1550 t/s** |
| vs llama.cpp Vulkan, no coopmat (1834) | 0.37× | ≈0.85× |
| vs llama.cpp Vulkan, coopmat (3214) | 0.21× | ≈0.48× |

So: **~2.3× on prefill from one kernel**, which is more than every prefill change
in §15 through §20 combined. It does not close the gap alone, and the remaining
factor is the coopmat one §19 measured at 1.75×.

Decode is barely affected — at one token `n_kv` is the context length and attention
is a small share — so this is a prefill and long-context fix, not a decode one.

### The lesson, stated plainly

Three sections of kernel work went into the expert path on the strength of a
traffic argument: experts are 96.5% of prefill's weight traffic and 67% of its
arithmetic. Both figures are true. Neither is a share of TIME. A profiler was
available the whole time and would have said `attention 61%` in one run.

---

## 22. The attention rewrite: prefill 680 → 1337 tok/s

§21 said attention was 61% of prefill and named four defects. This fixes them. It
is the largest single change in the engine's history, and it also vindicates §20,
whose traffic model turned out to be right and merely masked.

### What changed in `attention.comp`

| defect | fix |
|---|---|
| QK dot was a 128-deep **dependent** FMA chain on one accumulator | four independent `vec4` accumulators |
| K/V read 4 bytes per lane | `vec4` views of the same bindings — 16 bytes per request, 4× fewer instructions, a 64-byte line consumed in 4 iterations instead of 16 |
| the query re-read from global 128× per thread | staged once in LDS (`shared vec4 sq[128]`), read as a broadcast |
| AV pass idled **half** the workgroup and walked every key in one chain | every thread owns an (output `vec4`, key partition) pair — 32 × 8 = exactly 256 at head_dim 128 — plus two accumulators each |

The `vec4` view requires `head_dim % 4 == 0`, so the host now **refuses** a model
where it isn't rather than reading across element boundaries and returning
plausible nonsense.

### The kernel

| | before | after |
|---|---|---|
| `attention`, per 512-token pass | 460.38 ms | **68.49 ms** |
| share of prefill | 60.5% | **17.9%** |
| rate | 112 G MAC/s | **753 G MAC/s** |

**6.7× on the kernel.**

### The engine

| prefill tile | §20 | now |
|---|---|---|
| 32 | 659.5 | 843.0 |
| 64 | 681.9 | 1062.6 |
| 128 | 676.4 | 1215.4 |
| 256 | 669.1 | 1291.9 |
| **512** | 670.8 | **1337.1** |

**The sweep is now monotonic to 512, and 512 is the best tile.** That is §20's
traffic model being correct after all: widening the tile does cut expert weight
traffic 13.8×, and it was worth nothing only because a kernel that dominated the
time and ignored the tile stood in front of it. The default is now `MAX_TILE`.

| prompt | §20 | now | vs one-at-a-time |
|---|---|---|---|
| 64 | 1079.6 | 1127.6 | 9.7× |
| 128 | 1033.2 | 1349.8 | 11.7× |
| 256 | 880.6 | **1425.4** | 12.3× |
| 512 | 680.3 | **1337.1** | 11.8× |

**pp512 1.98×.** A 512-token prompt: 0.753 s → **0.383 s**.

Long-context decode improved too, as a side effect: one-at-a-time ingestion of 512
tokens — which is decode with a growing cache — went 88.2 → 113.7 tok/s, **+29%**,
because attention is the part of decode that grows with context.

### The new profile

| kernel | tile 512 |
|---|---|
| `q6k_expert_rt` | **50.0%** |
| `attention` | 17.9% |
| `q6k_matvec` | 14.8% |
| `q6k_x3` | 10.6% |
| `matvec_f32` | 3.2% |

The routed experts are now the top cost, which is where the traffic argument said
they were all along. At tile 512 they move 23.8 GB in 191 ms — 124 GB/s, 19% of
peak — and run at 4852 G MAC/s against a ~32800 ceiling, so there is headroom.

### Where llamd stands

| | llamd | llama.cpp Vulkan |
|---|---|---|
| decode, 1 stream | **112 – 117** | 94.4 – 95.6 |
| decode, batch 8 | **398 – 402** | 290.9 |
| **prefill pp512** | **1337** | 3214 (1834 without coopmat) |

Prefill is **2.40× behind**, from 4.9×. Against llama.cpp's non-coopmat path it is
**0.73×**, from 0.37×.

### The decode question, now settled by an A/B

It is settled the only way it could be: **both kernels kept as pipelines and
alternated inside one process**, best of three, with the arm order flipped between
batch sizes so a warming trend cannot favour one consistently
(`attention_rewrite_against_the_previous_kernel`).

| case | rewritten | previous | |
|---|---|---|---|
| decode batch 1 | 112.50 | 97.38 | **1.155×** |
| decode batch 2 | 213.25 | 203.96 | 1.046× |
| decode batch 4 | 377.46 | 351.70 | 1.073× |
| decode batch 8 | 552.56 | 516.42 | 1.070× |
| **prefill pp512** | **1337.7** | 674.2 | **1.984×** |

Both arms emit the same next token (1119), so this compares two kernels and not two
models.

**The rewrite is faster at every batch size — 4.6% to 15.5% on decode.** There was
no regression. The ~3% was drift between two builds, which is exactly what the
three failed explanations below were evidence of and what I should have concluded
from them at the time instead of leaving it open.

(These decode figures are higher than the table above because this harness holds
context at 64 tokens and repeats 16 passes; the throughput test generates 64 tokens
with a growing cache. The two are not comparable to each other — only within
themselves, which is the point of an A/B.)

### The three explanations that failed, and why that was the answer

Before the A/B existed, decode at batch 4–8 read ~3% below a single-run baseline of
276.59/411.30 and best-of-four gave 267.7/398.1. Three candidate causes were
tested and all three came back null:

1. The profiler shows attention is **2.5%** of decode at batch 8 — it cannot
   account for 3%.
2. A single-chunk fast path with literal-zero bases in `q6k_matvec`/`q6k_x3`,
   restoring the exact pre-chunking inner loop, changed nothing.
3. Removing the macro's runtime `break` entirely changed nothing (190.07 vs 189.93
   at batch 2, 400.02 vs 398.08 at batch 8) — a third counting model refuted.

And the run-to-run spread at batch 1 across four runs was 98.82, 111.55, 112.20,
117.18 — 19%. A single-run baseline sits anywhere in that.

The claim made here first — that deciding it "needs an in-process A/B, which a
shader change cannot have" — was **wrong**, and it was the one claim not tested.
A shader change can have one: keep both shaders as pipelines and flip between them
at runtime, which is the pattern `set_expert_rt` and `set_acc_specialized` already
established in this same file. `attention_scalar.comp` is retained for exactly that
and must not be "fixed" — it is only useful as long as it is what was measured.
The fast path was kept too: it is the correct shape and costs nothing.

### Correctness

All five prompts still token-for-token equal to llama.cpp. Prefill equals
sequential decode across 100 (tile, length) cases. Row-tiled experts exact across
45. 131 passed, 0 failed, 22 ignored.

The summation order **did** change — four partial accumulators sum a 128-term f32
dot differently than one chain — so results differ at ~1e-7 relative. The two
near-tied prompts have margins of 0.1196 and 0.1946, six orders of magnitude
above that, and the agreement suite is what verifies it rather than an argument.

---

## 23. The expert kernel's occupancy, and the shape that changes everything

§22 left the routed experts at 50% of prefill, running at 124 GB/s (19% of memory
peak) and 4852 G MAC/s (15% of the dot-product ceiling) — bound by neither.

### Why: two waves per SIMD

`RADV_DEBUG=shaderstats` on the pipeline set: the row-tiled expert kernel at
ROWS 4 × ACC 16 compiles to **252 VGPRs, two waves per SIMD**. Accumulators are
`ROWS × ACC` and dominate the register file, so the reuse row-tiling bought in
§20 was paid straight back in lost latency hiding. That is the whole reason it
measured 1.03×.

`ROWS` moved from a `#define` to three thin wrappers around
`expert_rt_body.glsl`, so the trade is measurable rather than asserted:

| ROWS | ACC | acc regs | tok/s |
|---|---|---|---|
| 1 | 32 | 32 | 1286.5 |
| 2 | 8 | 16 | 1374.3 |
| **2** | **16** | **32** | **1401.9** |
| 2 | 32 | 64 | 1341.0 |
| 4 | 4 | 16 | 1300.8 |
| 4 | 8 | 32 | 1375.0 |
| 4 | 16 | 64 | 1336.6 |
| 8 | 4 | 32 | 1181.5 |
| 8 | 8 | 64 | 1241.5 |

**32 accumulators is the sweet spot, and 64 is the cliff** — every 64-accumulator
arm loses to its 32-accumulator sibling. Every arm emits the same token, asserted
rather than eyeballed. Default is now 2 × 16.

| | before | after |
|---|---|---|
| expert kernel | 191.20 ms | **169.61 ms** |
| its share | 50.0% | 46.6% |
| pp512 | 1337 | **1402** |

+4.9%, which is honest and small. Occupancy was a real factor and not the main one.

### Where the ceiling actually is

| | pp512 | t/s | % of roofline |
|---|---|---|---|
| roofline (24.8 GB at 600 GB/s; 1.39 T MAC at the dot ceiling) | ~42 ms | ~12000 | 100% |
| llama.cpp Vulkan + coopmat | 159 ms | 3214 | **26%** |
| llama.cpp Vulkan, no coopmat | 279 ms | 1834 | 15% |
| **llamd now** | **365 ms** | **1402** | **12%** |

llama.cpp is not near the limit either. The gap is not exotic; it is a factor of
two in efficiency on the matmul kernels, which are 77% of the pass.

### And the instruction that closes it

`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` on this part
(`examples/coopmat.rs`) reports 12 shapes, all 16×16×16 at subgroup scope,
including:

```
16x16x16  A u8   B u8   C u32  acc u32
16x16x16  A s8   B s8   C s32  acc s32
16x16x16  A f16  B f16  C f32  acc f32
```

**`u8 × u8 → u32` is the §V2 identity in hardware.** That identity is why this
engine exists: Q6_K weights are unsigned 0..63, the Q8_K activations are already
biased to unsigned by `q XOR 0x80`, and every correction term
(`−128·Σw − 32·Σb + 32·128·16`) is an integer. So llamd can feed matrix cores
**with no dequantisation to float at any point** — the planar repack of §V3 already
produces the operand layout.

llama.cpp's Vulkan backend reports `fp16: 1` and takes the f16 path, dequantising
Q6_K to feed it. On RDNA4 int8 matrix throughput is typically twice f16.

The arithmetic density: one 16×16×16 coopmat instruction is 4096 MACs per
subgroup — **64 MACs per lane against 4 for `dotPacked4x8AccSatEXT`, 16×**. The
matmul kernels (expert 46.6%, matvec 15.8%, x3 11.1%, router 3.4% = 77%) are 280 ms
of the 365 ms pass:

| matmuls speed up by | pp512 | t/s | vs llama.cpp |
|---|---|---|---|
| 2× | 225 ms | 2276 | 0.71× |
| **4×** | **155 ms** | **3310** | **1.03×** |
| 6× | 132 ms | 3888 | 1.21× |

**4× on the matmuls passes llama.cpp**, and 4× is a quarter of the instruction
density the shape provides. That is the target, and unlike §19's and §20's
projections it rests on an enumerated hardware capability rather than a counting
argument — though on this project's record, it is a hypothesis until measured.

---

## 24. Cooperative matrix: enabled, measured, and the fold solved

§23 named the shape. This enables it, prices it, and settles the kernel design
before a kernel is written — because three projections in this document have
predicted large prefill gains and returned ~0, and the pattern in all three was
reasoning where a measurement was available.

### Enabled, and optional

`VK_KHR_cooperative_matrix` plus `cooperativeMatrix`, `vulkanMemoryModel` and
`vulkanMemoryModelDeviceScope` are now requested — **if the device has them**. A
device without is not broken: the shader table skips any shader whose name contains
`coopmat`, `llamd_vk_dispatch` refuses a null pipeline, and the dot-product kernels
remain the fallback and the correctness reference. A coopmat shader cannot merely
fail at dispatch — pipeline creation fails at load, for the whole table — so this
had to be a build-time skip rather than a runtime branch.

`shaderInt8` and `storageBuffer8BitAccess` were already on, and the toolchain
(glslang 16.1.0, shaderc 2026.1) compiles `GL_KHR_cooperative_matrix` against the
existing `--target-env=vulkan1.3`.

### The problem the shape creates

Q6_K carries an **int8 scale per 16 weights** and an f16 super-scale per 256. The
MMA's K dimension is **also 16**. So one MMA consumes exactly one scale group, and
its 16×16 result must be scaled per row before the next group can join it. The scale
cannot be folded into the operand: weights are 0..63 and an int8 scale multiplied by
them overflows u8.

Done naively — store the accumulator to LDS, read it back, scale each element —
that is **256 scalar operations per 4096-MAC instruction**.

### Measured, same harness, same occupancy

| arm | G MAC/s | vs dot | % of ceiling |
|---|---|---|---|
| MMA, no fold | **59307** | 33.2× | 100% |
| MMA + naive fold | 9381 | 5.3× | 16% |
| **MMA + matrix-domain fold** | **21651** | **12.1×** | **37%** |
| `dotPacked4x8AccSatEXT` | 1786 | 1× | 3% |

The naive fold keeps 16% of the instruction's ceiling. **The fear was justified and
the fix works.**

### The fix: fold without leaving the matrix domain

The weight scale is per row, therefore **constant along the token axis**. A
stride-0 load expresses exactly that: loaded row-major with stride 0, element
`[r][c]` comes from `mem[r*0 + c] = mem[c]`. So compute the product **transposed** —
tokens down, weight rows across — and the per-row scale becomes a per-**column**
broadcast that a 16-element array supplies:

```glsl
MAT_C mc = coopMatMulAdd(ma, mb, MAT_C(0u));
MAT_F ms;
coopMatLoad(ms, s_bcast, 0, 0, gl_CooperativeMatrixLayoutRowMajor);  // stride 0
facc += ms * MAT_F(mc);                                              // elementwise
```

16 shared writes per group instead of 256 extractions, and the accumulator never
leaves the matrix domain until the end. **2.31× over the naive fold**, and both
arms return the same verified value (512 for all-ones operands over 8 iterations),
which is what proves the stride-0 load means what the design relies on.

### And the simplification measuring it exposed

The device reports **all eight** signed/unsigned 8-bit combinations, including:

```
16x16x16  A s8  B s8  C s32  acc s32
```

The §V2 unsigned identity — `Σ sc·(w−32)·a` rewritten over unsigned operands with
`−128·Σw − 32·Σb + 32·128·16` corrections — exists because **gfx1201's HIP lacks
`sdot4`**. The matrix cores do not lack signed int8. So the coopmat kernel computes

    out[r][t] = d[r] · sc[r][g] · Σ_j (w[r][j] − 32) · a[t][j]

directly: `w−32` is −32..31 and fits s8, the Q8_K activation is already int8, and
**every correction term disappears** — no `Σw`, no `Σb`, no `+65536`, and no XOR
bias. The coopmat path is *simpler* than the dot path it replaces, which is not how
this usually goes.

### What this predicts, and the standing caveat

21651 G MAC/s against the expert kernel's measured 4852 is **4.46×** on its
arithmetic. §23's arithmetic said 4× on the matmuls reaches 3310 tok/s, past
llama.cpp's 3214.

That is now a measured instruction rate and a verified fold, not a counting
argument — which is better footing than §19, §20 or §23 had. It is still not a
kernel. What remains is the unglamorous part: unpacking planar Q6_K into s8 tiles in
LDS, staging activations transposed, and driving 128 groups per row-block. A kernel
realises less than its instruction ceiling, always, and how much less is the next
measurement rather than the next paragraph.

---

## 25. The coopmat expert kernel: correct, at parity, and the reason why

§24 measured the instruction at 21651 G MAC/s with the fold, 4.46× the expert
kernel's arithmetic, and said "a kernel realises less than its instruction ceiling,
always, and how much less is the next measurement rather than the next paragraph."
This is that measurement. It realises **24%** of it, which is parity.

### The kernel

`q6k_expert_cm.comp`, signed 8-bit, and simpler than the kernel it replaces:

    out[r][t] = d[r][b] · sc[r][g] · Σ_j (w[r][j] − 32) · a[t][j]

No `Σw`, no `Σb`, no `+65536`, no XOR bias — the corrections exist only because
gfx1201's HIP lacks `sdot4`, and the matrix cores do not. The product is computed
**transposed** so the per-weight-row scale is a per-column broadcast from a stride-0
row-major load, and the activation's block scale a per-row broadcast from a stride-0
column-major load. The accumulator never leaves the matrix domain.

**24 of 24 (tile, length) cases emit the same token as the dot kernel**, across
token counts either side of TT·16 = 32 and tiles either side of the row blocking.

### Three things it was slow for, each found by measuring

| | pp512 | vs dot |
|---|---|---|
| first working version | 690.6 | 0.266× |
| 8 scale groups per barrier pair | 1253.6 | 0.90× |
| weights row-major, B loaded column-major | **1372.0** | **0.987×** |

1. **Barriers.** One scale group per barrier pair means a 128-group K loop costs 256
   workgroup barriers per output tile — and a barrier costs more than the 16-deep
   matrix multiply it protects. Staging 8 groups per pair: **1.8×**.
2. **16 groups per pair was worse** (906.4), not better: LDS went to 34 KiB and took
   occupancy with it. 8 is the measured point.
3. **Byte-scattered LDS stores.** Stored K-major, a lane unpacking four consecutive
   K of one row writes four bytes *sixteen apart*, and sub-dword scattered shared
   stores are read-modify-write. Storing row-major and loading the B operand
   **column-major** makes them contiguous — the layout argument is free. **1.09×**.

### Why parity is the ceiling here

| kernel | ms per 512-token pass | G MAC/s |
|---|---|---|
| `q6k_expert_rt2` (dot) | 169.61 | 5466 |
| `q6k_expert_cm` (matrix) | 178.96 | 5180 |

The same 927 G MACs. §24's fold arm would run them in **43 ms**; the kernel takes
179. So **136 ms of it — 76% — is not matrix arithmetic**: it is unpacking Q6_K into
s8 tiles, staging activations, and the barriers between. The matrix units accelerate
the quarter that was arithmetic and cannot touch the rest.

This is consistent with everything measured before it. §23 found the dot kernel at
19% of memory peak and 15% of the dot ceiling — **bound by neither**, therefore
bound by what sits between them. Replacing the arithmetic engine does not move a
kernel whose arithmetic was never the constraint. §24's 4.46× was a property of the
instruction, not of the problem, and the projection was the fourth in this document
to survive every check available and still be wrong.

### What this actually implies for beating llama.cpp

Not more matrix work on this path. **Dequantisation is the cost, so the lever is
reuse per unpack.** A weight tile unpacked once currently feeds TT·16 = 32 tokens,
which is exactly what an expert sees at a 512-token tile — so at pp512 the
amortisation is already maximal and there is nothing to win. It becomes the right
kernel when an expert sees more tokens, which needs a tile wider than 512, which
needs prompts longer than 512. For pp512 specifically, **the expert path is done**
until the unpack itself gets cheaper.

The kernel is kept and defaulted OFF: correct, measured, 0.987×, and the right
shape for a workload this benchmark does not contain. Parity is not a reason to put
more machinery on the hot path.

---

## 26. Hardware counters by ablation, and the reduction nobody was looking at

Four optimisations had by now failed on the matmul kernels (§19, §20, §24, and a
re-sweep of the token chunk that returned the existing default). Every one reasoned
from op counts. This section stops reasoning and gets the counters.

### There are no counters, so they were measured instead

RADV exposes **no `VK_KHR_performance_query`**. It does support SQTT with
`RADV_THREAD_TRACE_CACHE_COUNTERS` and `RADV_THREAD_TRACE_INSTRUCTION_TIMING`, and
`MESA_VK_TRACE_TRIGGER` even allows a capture without a swapchain — but the `.rgp`
needs AMD's GUI to read, and `rocprofv3` only reaches HIP/HSA, not Vulkan.

So: run the real kernel over a real tensor and **remove one class of work at a
time**. `matvec_ablate.comp`, five arms plus a vectorised sixth, every arm consuming
what it computes so nothing is eliminated.

### What it says

| arm | attn_output 4096→2048 | one expert gate 2048→768 |
|---|---|---|
| full kernel | 2397 G MAC/s | 1842 G MAC/s |
| dot products | **31%** | **13%** |
| activation reads | **40%** | 28% |
| weight traffic | 3% | 3% |
| Q6_K unpack | ~0% | ~0% |
| **unaccounted** | **28%** | **57%** |

Two things nobody had measured:

**The weights are not the problem — the activations are.** Removing the weight
traffic saves 3%; removing the activation reads saves 28–40%. The inner loop issues
**11 activation loads per (K-group, sequence)** — eight `qb` uints, two `sb`, one
`ad` — against **eight weight loads per K-group**. At 32 sequences that is 352 to 8,
a ratio of 44:1. Three sections of this document optimised weight traffic.

**And 57% of the expert kernel is none of its work at all.**

### The vectorisation that follows from it

The eight `qb` uints are four-consecutive, twice. Every index the kernel forms for a
run of four is 4-aligned, so both buffers become `uvec4` views and one 16-byte load
replaces four. On the isolated kernel: **1.21×**.

In the engine — `q6k.glsl` serves `q6k_matvec`, `q6k_x3` and both decode grouped
kernels:

| | before | after |
|---|---|---|
| decode, one-at-a-time | 113.7 | **127.5 t/s** (+12%) |
| pp128 | 1349.8 | **1414.3** |
| pp256 | 1425.4 | **1521.7** |
| pp512 | 1396 | 1391 (flat) |

Vectorising the **expert** kernel too was **1.5% slower** (1370.7 against 1391.4,
consistent over three runs each) — `uvec4` temporaries widen a kernel §23 already
showed to be register-bound. Reverted. The same change helps one kernel and hurts
another, which is only knowable by measuring both.

### The 57%, explained by geometry

The expert gate is K = 2048, so `n_u4 = blocks * 8 = 64` — and the subgroup is **64
lanes wide**. Each lane runs **exactly one iteration** of the K loop. Then it pays
the epilogue: a `subgroupAdd` per accumulator, 32 of them, each ~6 DPP steps on
wave64. That is roughly **192 reduction operations against 256 dot instructions of
actual work.**

**The expert kernel spends more time reducing across lanes than multiplying.** The
one-subgroup-per-output-row decomposition — which measured best three times in §16,
on shapes with a long K — is pathological when K is short enough that the loop
disappears.

None of the four failed optimisations touched this. Wide tiles, row tiling,
cooperative matrix and chunk width all changed what happened *inside* the loop.

### What follows, stated as a hypothesis this time

The fix that removes the reduction entirely is **one lane per output row** instead of
one subgroup per row: a lane accumulates a whole row's K, so there is nothing to
reduce. Its cost is that lanes then read different rows, so weight loads go
uncoalesced — and the ablation says weight traffic is **3%**, which is what makes
this affordable here and would not have been on a shape where weights dominated.
Activations become a broadcast across lanes, which is the cheapest access there is.

If the reduction is 43% of the expert kernel and a quarter of the matvec kernels,
that is ~97 ms of a 371 ms pass, or about **1870 tok/s**. Against llama.cpp's 3214
that is still behind, and this is the fifth projection in this document — the
difference from the previous four is that it rests on a measured 57% with a
mechanical explanation rather than on an op count.

---

## 27. The clustered reduction: two failures bracketing a win

§26 measured 57% of the expert kernel as unaccounted for by any work it does, and
blamed the `subgroupAdd` epilogue: the expert gate is K = 2048, so `n_u4 = 64` and
the subgroup is 64 lanes wide — each lane runs **one** loop iteration, then pays 32
reductions of ~6 DPP steps each.

### Failure one: remove the reduction entirely

`q6k_expert_lane.comp` — one lane per output row, walking the whole K itself, no
cross-lane reduction at all. **Correct across 18 of 18 cases, and 2.15× slower**
(644.5 against 1386.3 at pp512).

The reason is a flaw in how §26's numbers were read. The ablation says weight traffic
is **3%** — but that was measured with **coalesced** loads. One lane per row makes
each lane stream its own row, so a single load instruction touches 64 cache lines,
and the 3% became the dominant cost.

> **An ablation prices a component in the arrangement it was measured in. It does not
> price the same component after the arrangement changes around it.**

That is the general lesson, and it is the fifth distinct way this document has been
wrong about these kernels.

### Failure two is the old one: 64 lanes per row

Keeping the whole subgroup on one row is the other extreme: coalescing is perfect and
the reduction costs ~12288 lane-ops per row against ~51200 of inner work.

### The win is between them

`CLUSTER` lanes per row group: the reduction is `log2(CLUSTER)` steps over `CLUSTER`
lanes instead of 6 over 64, while the lanes of a cluster still read **consecutive K**
and stay coalesced. Four clusters means four weight streams, not sixty-four.

| prefill tile | subgroup 64 | cluster 32 | **cluster 16** | cluster 8 | 1 lane/row |
|---|---|---|---|---|---|
| 128 | 1237.8 | 1358.4 | **1380.5** | 1345.5 | 521.7 |
| 256 | 1325.6 | 1471.3 | **1491.7** | 1465.7 | 599.2 |
| 512 | 1364.7 | 1529.8 | **1547.1** | 1530.4 | 644.5 |

A clean inverted U with its peak at 16, which is what the two failures predicted:
too few lanes and the loads scatter, too many and the epilogue dominates. **1.134×**,
and 15 of 15 cases emit the same token.

### Where the engine stands

| | before §26 | now |
|---|---|---|
| pp64 | 1127.6 | **1249.4** |
| pp128 | 1349.8 | **1549.3** |
| pp256 | 1425.4 | **1689.4** |
| **pp512** | **1396** | **1572.6** |
| decode, 1 stream | 109–113 | 112.4 |
| decode, batch 8 | 396–411 | 395.0 |

Prefill is **2.04× behind** llama.cpp Vulkan's 3214, from 4.9× at §19 and 2.3× at
§25. Its non-coopmat path (1834) is now within **14%**.

Correctness unchanged throughout: all five prompts token-for-token equal to
llama.cpp, prefill equal to sequential decode across 100 (tile, length) cases,
132 passed, 0 failed.

### The scoreboard, honestly

| | llamd | llama.cpp | |
|---|---|---|---|
| decode, 1 stream | **112.4** | 94.4–95.6 | win 1.18× |
| decode, batch 2 | **189.8** | 145.8 | win 1.30× |
| decode, batch 4 | **265.5** | 229.5 | win 1.16× |
| decode, batch 8 | **395.0** | 290.9 | win 1.36× |
| prefill pp512 | 1572.6 | **3214** | **lose 0.49×** |

Four of five. The goal of being faster in every category is **not met**, and prefill
is the sole holdout.

Both remaining leads are measured rather than reasoned. The ablation's largest single
line is still **activation reads at 28–40%**, and the `uvec4` vectorisation only
collected that on the kernels where register pressure allowed it — the expert kernel
refused it (§26). And the clustered reduction has only been applied to the expert
kernel; `q6k_matvec` and `q6k_x3` have the same 64-lane epilogue over a K that is
twice as long, so the same change is worth less there but is not worth nothing.

---

## 28. The same fix on the attention projections: 0.957×

§27's clustered reduction was worth 1.134× on the expert kernel. `q6k_matvec` and
`q6k_x3` carry the identical epilogue — and Q/K/V is K = 2048, so `n_u4 = 64` against
a 64-lane subgroup, **exactly** the one-iteration-per-lane shape §26 diagnosed. The
same change applied to them is **4% slower**.

| prefill tile | clustered | subgroup 64 | |
|---|---|---|---|
| 128 | 1337.8 | 1403.9 | 0.953× |
| 256 | 1447.9 | 1512.9 | 0.957× |
| 512 | 1497.6 | 1564.3 | 0.957× |
| decode batch 8 | 479.1 | 497.6 | 0.963× |

15 of 15 cases emit the same token, so this is a performance result and not a bug.

### Why it does not transfer

The two kernels differ in what shares the activation. The expert kernel carries
`ROWS = 2` per lane, so one activation read feeds two rows and the epilogue is a
larger fraction of what is left. The projections are one row per lane group, so
clustering only quarters the coalescing without the same offset — four weight streams
per subgroup instead of one, paying for a cheaper reduction that was a smaller share
to begin with.

`Q6K_LANE`/`Q6K_LANES` stay in `q6k.glsl` and the two `_c` variants stay in the tree,
defaulted off, because a negative result with a measurement attached is worth more
than the two lines it costs. Anyone reading §27 will otherwise try exactly this.

### The pattern across this document

Three separate changes now have **opposite signs on two kernels that look alike**:

| change | expert kernel | attention projections |
|---|---|---|
| `uvec4` vectorised loads (§26) | **0.985×** | **1.21×** |
| clustered reduction (§27, §28) | **1.134×** | **0.957×** |
| cooperative matrix (§25) | 0.987× | not attempted |

Every one was predicted to help both. Nothing in the op counts distinguishes them —
what distinguishes them is register pressure, how much the activation is reused, and
how much coalescing there is to lose, none of which a count of instructions exposes.
**The only reliable method found in 28 sections is to measure each kernel separately,
in the engine, with both arms alternating in one process.**

---

## 29. Re-sweeping ROWS × ACC × vec4: the default was already right

§23 chose ROWS 2 / ACC 16 for the expert kernel before §26's vectorisation and §27's
clustering existed, and all three trade through the same register file. §26 tested
`uvec4` at that single point, found it 1.5% slower, and drew a conclusion from one
sample. This re-measures the whole grid, with every arm asserted to emit the same
token.

| ROWS | ACC | accumulators | scalar | vec4 | vec4 |
|---|---|---|---|---|---|
| 1 | 16 | 16 | 1281.3 | 1273.1 | 0.994× |
| 1 | 32 | 32 | 1271.8 | 1274.0 | 1.002× |
| 2 | 8 | 16 | 1479.2 | 1428.0 | 0.965× |
| **2** | **16** | **32** | **1559.8** | 1516.9 | 0.973× |
| 2 | 32 | 64 | 1530.6 | 1509.3 | 0.986× |
| 4 | 8 | 32 | 1456.7 | 1422.4 | 0.976× |
| 4 | 16 | 64 | 1541.7 | 1515.7 | 0.983× |

**No change. ROWS 2 / ACC 16 / scalar remains the optimum**, and the sweep is worth
more than the null result suggests because of three things it settles:

**`uvec4` loses everywhere in this kernel, not just at one point.** 0.965× to 1.002×
across all seven. §26's single sample was representative after all — but that was
luck, and the grid is what shows it.

**And the loss correlates with ROWS, not with the accumulator count.** At ROWS 1 it is
neutral (0.994×, 1.002×); at ROWS 2 and 4 it costs 2–3.5%. That is a mechanism, not a
coincidence: with ROWS > 1 the `uvec4` temporaries stay live across the inner row
loop, so they add ~16 VGPRs of pressure where the scalar version can retire each word
as it is consumed. The projections have one row per lane group, which is exactly why
the same change wins 1.21× there (§28).

**Activation reuse across rows dominates everything else.** ROWS 1 is 1271–1281
against 1559.8 at ROWS 2 — a **22% penalty** for giving up the sharing. That is
consistent with the ablation putting activation reads at 28–40% (§26) and it is why
`q6k_expert_lane`, which went to one row per lane, could never have won.

**32 accumulators is still the cliff edge**, now confirmed under clustering: 2×16 (32)
beats both 2×8 (16) and 2×32 (64).

The three knobs are now measured together rather than separately, which is what §28
concluded was the only reliable method. The defaults stand where they were.
