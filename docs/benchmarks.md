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

| path | configuration | tok/s |
|---|---|---|
| llama.cpp, HIP | `-ngl 99`, q8_0 KV, 98304 ctx | **57.7** |
| llama.cpp, HIP | `llama-bench`, f16 KV, short ctx | **73.0** |
| llama.cpp, CPU | 12 threads, f16 KV, warm | **0.75** |
| **LLAMD, CPU** | 1 thread, scalar reference | **0.363** |
| LLAMD, GPU | kernel validated, *not wired in* | — |

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
