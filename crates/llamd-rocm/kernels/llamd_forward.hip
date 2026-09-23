// LLAMD resident forward pass — every operation of a token on the device.
//
// WHY A SECOND FILE. llamd.hip holds the matvec kernels and their host-facing
// entry points, which copy activations in and results out. Those are the right
// shape for a caller holding host data and the wrong shape for a forward pass:
// a token runs roughly 1400 matmuls, and at the measured 0.12 ms round-trip
// that is 167 ms per token — 6 tok/s, a ceiling no kernel can break.
//
// Everything here takes DEVICE pointers and launches asynchronously on the
// default stream. Nothing synchronises. The caller synchronises once, after
// the whole token, and only the argmax crosses the bus.
//
// CORRECTNESS IS DEFINED BY THE CPU REFERENCE, not by what looks reasonable:
// these kernels must reproduce llamd-runtime's forward_layer op for op, since
// that path is what agrees with llama.cpp on all five token-agreement prompts.
// Where a choice exists, the comment says which CPU line it mirrors.
//
// PROVENANCE: the Q6_K layout, NeoX RoPE pairing and MoE gating follow
// llama.cpp / ggml (MIT, (c) 2023-2026 The ggml authors).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#define QK_K            256
#define Q6K_BLOCK_BYTES 210
#define WAVE            32
#define FWD_BLOCK       256

// ---------------------------------------------------------------------------
// Shared reductions. One workgroup, FWD_BLOCK threads.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float v, float *s) {
    const int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    return s[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float *s) {
    const int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] = fmaxf(s[t], s[t + stride]);
        __syncthreads();
    }
    return s[0];
}

// ---------------------------------------------------------------------------
// Q6_K row dequantization — the embedding lookup.
//
// Mirrors CpuBackend::get_rows on a quantized table. The decode must match the
// dot kernel's byte handling exactly; it is the same layout read for a
// different purpose.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void q6k_decode_block(
    const uint8_t *blk, float *out /* 256 floats */, int tid, int nthreads)
{
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + QK_K / 2;
    const int8_t  *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
    const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                         | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
    const float d = __half2float(__ushort_as_half(dbits));

    for (int half = 0; half < 2; ++half) {
        const int o = half * 128;
        for (int l = tid; l < 32; l += nthreads) {
            const uint8_t h   = qh[half * 32 + l];
            const uint8_t lo0 = ql[half * 64 + l];
            const uint8_t lo1 = ql[half * 64 + l + 32];
            const int v0 = (int)(int8_t)((lo0 & 0x0F) | (((h >> 0) & 3) << 4)) - 32;
            const int v1 = (int)(int8_t)((lo1 & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
            const int v2 = (int)(int8_t)((lo0 >>   4) | (((h >> 4) & 3) << 4)) - 32;
            const int v3 = (int)(int8_t)((lo1 >>   4) | (((h >> 6) & 3) << 4)) - 32;
            out[o + l]      = d * (float)sc[(o + l)      / 16] * (float)v0;
            out[o + l + 32] = d * (float)sc[(o + l + 32) / 16] * (float)v1;
            out[o + l + 64] = d * (float)sc[(o + l + 64) / 16] * (float)v2;
            out[o + l + 96] = d * (float)sc[(o + l + 96) / 16] * (float)v3;
        }
    }
}

__global__ void fk_embed_q6k(
    const uint8_t * __restrict__ table, int token, float * __restrict__ out, int blocks)
{
    const uint8_t *row = table + (size_t)token * blocks * Q6K_BLOCK_BYTES;
    for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
        q6k_decode_block(row + (size_t)b * Q6K_BLOCK_BYTES, out + b * QK_K,
                         threadIdx.x, blockDim.x);
    }
}

// ---------------------------------------------------------------------------
// rms_norm(x) * w, as CpuBackend::rms_norm followed by mul.
// One workgroup over the whole vector.
// ---------------------------------------------------------------------------
__global__ void fk_rms_norm_mul(
    const float * __restrict__ x, const float * __restrict__ w,
    float * __restrict__ out, int n, float eps)
{
    __shared__ float s[FWD_BLOCK];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    const float total = block_reduce_sum(acc, s);
    const float inv = rsqrtf(total / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

// Per-head RMS norm for QK-norm: one workgroup per head, head_dim threads.
// Mirrors per_head_rms_norm in llamd-runtime/src/layer.rs, which normalises
// each head independently BEFORE RoPE.
__global__ void fk_head_rms_norm_mul(
    const float * __restrict__ x, const float * __restrict__ w,
    float * __restrict__ out, int hd, float eps)
{
    extern __shared__ float s[];
    const int head = blockIdx.x;
    const float *xi = x + (size_t)head * hd;
    float *oi = out + (size_t)head * hd;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += xi[i] * xi[i];
    const float total = block_reduce_sum(acc, s);
    const float inv = rsqrtf(total / (float)hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) oi[i] = xi[i] * inv * w[i];
}

// ---------------------------------------------------------------------------
// NeoX RoPE. Pairs element i with i + hd/2 — NOT adjacent elements. Mirrors
// CpuBackend::rope: theta = pos / freq_base^(2i/hd), applied per head.
// ---------------------------------------------------------------------------
__global__ void fk_rope_neox(
    float * __restrict__ x, int pos, int hd, float freq_base)
{
    const int head = blockIdx.x;
    const int half = hd / 2;
    float *xi = x + (size_t)head * hd;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float exponent = 2.0f * (float)i / (float)hd;
        const float theta = (float)pos / powf(freq_base, exponent);
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float a = xi[i];
        const float b = xi[i + half];
        xi[i]        = a * cs - b * sn;
        xi[i + half] = a * sn + b * cs;
    }
}

// Store this position's K and V into the cache.
__global__ void fk_kv_store(
    float * __restrict__ kc, float * __restrict__ vc,
    const float * __restrict__ k, const float * __restrict__ v,
    int pos, int kv_dim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    kc[(size_t)pos * kv_dim + i] = k[i];
    vc[(size_t)pos * kv_dim + i] = v[i];
}

// ---------------------------------------------------------------------------
// Attention, one workgroup per query head.
//
// Mirrors llamd_runtime::attend: scale 1/sqrt(head_dim), GQA by integer
// division (h / ratio), softmax shifted by the max, then the weighted sum of V.
//
// Each thread computes whole dot products for a subset of positions, so no
// cross-thread reduction is needed for the scores themselves — only for the
// max and the sum. Scores live in a global scratch buffer rather than LDS
// because n_kv grows with the sequence and would overflow 64 KiB.
// ---------------------------------------------------------------------------
__global__ void fk_attention(
    const float * __restrict__ q, const float * __restrict__ kc,
    const float * __restrict__ vc, float * __restrict__ out,
    float * __restrict__ scores, int n_head_kv, int hd, int n_kv, int ratio)
{
    __shared__ float s[FWD_BLOCK];
    const int h   = blockIdx.x;
    const int kvh = h / ratio;
    const int kv_dim = n_head_kv * hd;
    const float scale = rsqrtf((float)hd);

    const float *qh = q + (size_t)h * hd;
    float *sc = scores + (size_t)h * n_kv;

    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float *kt = kc + (size_t)t * kv_dim + (size_t)kvh * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
        const float sv = dot * scale;
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    const float mx = block_reduce_max(local_max, s);

    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float e = __expf(sc[t] - mx);
        sc[t] = e;
        local_sum += e;
    }
    const float sum = block_reduce_sum(local_sum, s);
    const float inv = 1.0f / sum;
    __syncthreads();

    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            acc += sc[t] * vc[(size_t)t * kv_dim + (size_t)kvh * hd + d];
        }
        out[(size_t)h * hd + d] = acc * inv;
    }
}

__global__ void fk_add(
    const float * __restrict__ a, const float * __restrict__ b,
    float * __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// silu(gate) * up. SiLU on the GATE only — SwiGLU is not symmetric, and
// applying it to `up` or after the multiply still produces fluent text.
__global__ void fk_silu_mul(
    const float * __restrict__ gate, const float * __restrict__ up,
    float * __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        out[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

// ---------------------------------------------------------------------------
// Router: softmax over every expert, then top-k.
//
// Mirrors llamd_models::route exactly, including the two details that decide
// reproducibility: ties break toward the LOWER expert index, matching ggml's
// argsort, and the renormalising divisor is clamped at 2^-14.
//
// Selection is a serial scan on thread 0. n_expert is 128 and n_used is 8, so
// this is 1024 comparisons — far cheaper than a parallel sort's overhead, and
// exactly reproducible, which matters more here than speed.
// ---------------------------------------------------------------------------
#define WEIGHT_SUM_FLOOR 6.103515625e-5f

__global__ void fk_router_topk(
    const float * __restrict__ logits, int n_expert, int n_used,
    int32_t * __restrict__ ids, float * __restrict__ weights,
    float * __restrict__ probs)
{
    __shared__ float s[FWD_BLOCK];
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x)
        local_max = fmaxf(local_max, logits[i]);
    const float mx = block_reduce_max(local_max, s);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
        const float e = __expf(logits[i] - mx);
        probs[i] = e;
        local_sum += e;
    }
    const float total = block_reduce_sum(local_sum, s);
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) probs[i] /= total;
    __syncthreads();

    // TOP-K, PARALLEL.
    //
    // WHY NOT THE OBVIOUS SERIAL SCAN. The first version did the selection on
    // thread 0: eight rounds over n_expert, each reading probs[] from global
    // memory, with an inner check against the already-chosen ids. Correct, and
    // it cost 12.3 of a token's 32.5 profiled milliseconds -- 37.9%, the single
    // largest phase, larger than all 1152 expert matmuls combined. One thread
    // issuing ~1024 dependent global loads per layer hides no latency at all.
    //
    // This stages the probabilities in LDS once, then runs n_used rounds of a
    // block-wide max reduction, masking each winner to -INF. Same selection,
    // same tie rule, 56 LDS-resident reduction steps instead of 1024 global
    // loads.
    __shared__ float sp[1024];
    __shared__ int   sidx[FWD_BLOCK];
    __shared__ float sval[FWD_BLOCK];
    if (n_expert > 1024) return;   // guarded by the Rust wrapper; belt and braces
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) sp[i] = probs[i];
    __syncthreads();

    for (int k = 0; k < n_used; ++k) {
        float bv = -INFINITY;
        int   bi = -1;
        for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
            // Strictly greater keeps the LOWER index on a tie, matching
            // route()'s .then(a.cmp(&b)) and therefore ggml's argsort.
            if (sp[i] > bv || (sp[i] == bv && i < bi)) { bv = sp[i]; bi = i; }
        }
        sval[threadIdx.x] = bv;
        sidx[threadIdx.x] = bi;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const int j = threadIdx.x + stride;
                if (sval[j] > sval[threadIdx.x]
                    || (sval[j] == sval[threadIdx.x] && sidx[j] < sidx[threadIdx.x])) {
                    sval[threadIdx.x] = sval[j];
                    sidx[threadIdx.x] = sidx[j];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            ids[k] = sidx[0];
            weights[k] = sval[0];
            sp[sidx[0]] = -INFINITY;   // so the next round cannot pick it again
        }
        __syncthreads();
    }

    // Renormalise with ggml's clamp on the divisor. Serial over n_used (8), on
    // values already in registers-worth of LDS, so it costs nothing.
    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < n_used; ++k) wsum += weights[k];
        wsum = fmaxf(wsum, WEIGHT_SUM_FLOOR);
        for (int k = 0; k < n_used; ++k) weights[k] /= wsum;
    }
}

// out[d] = sum_e weights[e] * down[e][d]
__global__ void fk_moe_combine(
    const float * __restrict__ down, const float * __restrict__ weights,
    float * __restrict__ out, int n_used, int n_embd)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_used; ++e) acc += weights[e] * down[(size_t)e * n_embd + d];
    out[d] = acc;
}

// Argmax over the vocabulary. Two stages so the final answer is a single int
// and the logits never cross the bus: 151936 floats is 600 KiB per token.
__global__ void fk_argmax_partial(
    const float * __restrict__ x, int n,
    float * __restrict__ bestv, int32_t * __restrict__ besti)
{
    __shared__ float sv[FWD_BLOCK];
    __shared__ int   si[FWD_BLOCK];
    const int t = threadIdx.x;
    float bv = -INFINITY; int bi = -1;
    for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
        if (x[i] > bv) { bv = x[i]; bi = i; }
    }
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            // Strictly greater, and the lower index on a tie — the same rule
            // the CPU's max_by follows, so the emitted token is reproducible.
            if (sv[t + stride] > sv[t] || (sv[t + stride] == sv[t] && si[t + stride] < si[t])) {
                sv[t] = sv[t + stride]; si[t] = si[t + stride];
            }
        }
        __syncthreads();
    }
    if (t == 0) { bestv[blockIdx.x] = sv[0]; besti[blockIdx.x] = si[0]; }
}

__global__ void fk_argmax_final(
    const float * __restrict__ bestv, const int32_t * __restrict__ besti,
    int n, int32_t * __restrict__ out)
{
    if (threadIdx.x != 0) return;
    float bv = -INFINITY; int bi = -1;
    for (int i = 0; i < n; ++i) {
        if (bestv[i] > bv || (bestv[i] == bv && besti[i] < bi)) { bv = bestv[i]; bi = besti[i]; }
    }
    *out = bi;
}

// ---------------------------------------------------------------------------
// FUSED KERNELS.
//
// WHY. The phase breakdown put ~1200 launches in a token at roughly 4.5 us
// each — about 5.4 of 14.87 ms, more than any single phase. The rope path was
// five launches per layer (two QK-norms, two RoPEs, a KV store) for work that
// touches 4096 and 512 floats. Fusing is not micro-optimisation here; launch
// overhead was the largest single cost in the engine.
//
// Each fused kernel performs EXACTLY the operations it replaces, in the same
// order, so the arithmetic is unchanged and the token-agreement test still
// holds. Nothing is approximated to make a fusion possible.
// ---------------------------------------------------------------------------

// Per-head RMS norm followed by NeoX RoPE, one workgroup per head.
//
// Both stages are confined to a single head — the norm reduces over head_dim,
// the rotation pairs i with i + head_dim/2 inside the same head — so one
// workgroup can do both with a barrier between. Replaces
// fk_head_rms_norm_mul + fk_rope_neox.
//
// `w` may be null: a model without QK-norm skips the norm and still ropes.
__global__ void fk_norm_rope(
    float * __restrict__ x, const float * __restrict__ w,
    int hd, const int32_t * __restrict__ meta, float freq_base, float eps)
{
    extern __shared__ float sh[];
    const int head = blockIdx.x;
    const int pos  = meta[1];
    float *xi = x + (size_t)head * hd;

    if (w) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += xi[i] * xi[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        const float inv = rsqrtf(sh[0] / (float)hd + eps);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) xi[i] = xi[i] * inv * w[i];
        __syncthreads();
    }

    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float exponent = 2.0f * (float)i / (float)hd;
        const float theta = (float)pos / powf(freq_base, exponent);
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float a = xi[i];
        const float b = xi[i + half];
        xi[i]        = a * cs - b * sn;
        xi[i + half] = a * sn + b * cs;
    }
}

// The K side: norm, rope, and store both K and V into the cache. Replaces
// fk_head_rms_norm_mul + fk_rope_neox + fk_kv_store.
//
// The store must come after the rotation — the cache holds ROPED keys, which is
// what attention reads — so it cannot be hoisted out or run concurrently.
__global__ void fk_k_norm_rope_store(
    float * __restrict__ k, const float * __restrict__ w,
    const float * __restrict__ v, float * __restrict__ kc, float * __restrict__ vc,
    int hd, int kv_dim, const int32_t * __restrict__ meta, float freq_base,
    float eps)
{
    extern __shared__ float sh[];
    const int head = blockIdx.x;
    const int pos  = meta[1];
    float *ki = k + (size_t)head * hd;

    if (w) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += ki[i] * ki[i];
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
            __syncthreads();
        }
        const float inv = rsqrtf(sh[0] / (float)hd + eps);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ki[i] = ki[i] * inv * w[i];
        __syncthreads();
    }

    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float exponent = 2.0f * (float)i / (float)hd;
        const float theta = (float)pos / powf(freq_base, exponent);
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float a = ki[i];
        const float b = ki[i + half];
        ki[i]        = a * cs - b * sn;
        ki[i + half] = a * sn + b * cs;
    }
    __syncthreads();

    // Store this head's slice of K (now roped) and V.
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const int off = head * hd + i;
        kc[(size_t)pos * kv_dim + off] = ki[i];
        vc[(size_t)pos * kv_dim + off] = v[off];
    }
}

// rms_norm(x) * w, written as f32 AND quantized to Q8_K in one pass.
//
// Replaces fk_rms_norm_mul + q8k_quantize. The f32 output is still needed by
// the F32 router matvec; the quantized form feeds every Q6_K matvec. Computing
// the norm twice, or writing f32 and reading it back in a second kernel, was
// two launches and a round trip through L2 for no reason.
//
// The quantization must match llamd_quant::quantize_q8_k exactly: iscale is
// -127/max where max is the SIGNED value of largest magnitude, the clamp is
// upper-only, and rounding is half-to-even.
__global__ void fk_rms_scale(
    const float * __restrict__ x, float * __restrict__ inv_out, int n, float eps)
{
    __shared__ float s[FWD_BLOCK];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    const float total = block_reduce_sum(acc, s);
    if (threadIdx.x == 0) *inv_out = rsqrtf(total / (float)n + eps);
}

// Scale by the precomputed 1/rms, multiply by w, and quantize — ONE WORKGROUP
// PER QUANTIZATION BLOCK, so the blocks run concurrently.
//
// WHY THIS IS TWO KERNELS AND NOT ONE. The fused single-workgroup version
// computed the norm and then quantized all eight blocks SEQUENTIALLY, because a
// 256-element block max needs a reduction across the whole workgroup. That cost
// ~22 us and it runs 96 times per token — about 2.1 of 12.5 ms, on one of the
// card's 32 WGPs. Splitting the reduction out costs an extra launch and lets
// eight workgroups quantize at once. Measured, the launch is much the cheaper
// half: see benchmarks.md §8.
//
// Must match llamd_quant::quantize_q8_k exactly: iscale = -127/max with max the
// SIGNED value of largest magnitude, upper-only clamp, round-half-to-even.
__global__ void fk_norm_quant(
    const float * __restrict__ x, const float * __restrict__ w,
    const float * __restrict__ inv_in, float * __restrict__ out,
    int8_t * __restrict__ qs, float * __restrict__ d)
{
    __shared__ float sa[QK_K];
    __shared__ float sm[QK_K];
    const int b = blockIdx.x;
    const int i = b * QK_K + threadIdx.x;
    const float inv = *inv_in;

    const float v = x[i] * inv * w[i];
    out[i] = v;

    sa[threadIdx.x] = fabsf(v);
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int stride = QK_K / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            // Strictly greater, so a tie keeps the earlier element — the one a
            // sequential host scan keeps.
            if (sa[threadIdx.x + stride] > sa[threadIdx.x]) {
                sa[threadIdx.x] = sa[threadIdx.x + stride];
                sm[threadIdx.x] = sm[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    const float amax = sa[0], mx = sm[0];
    if (amax == 0.0f) {
        qs[i] = 0;
        if (threadIdx.x == 0) d[b] = 0.0f;
        return;
    }
    const float iscale = -127.0f / mx;
    int q = (int)rintf(iscale * v);
    if (q > 127) q = 127;
    qs[i] = (int8_t)q;
    if (threadIdx.x == 0) d[b] = 1.0f / iscale;
}

// out = residual + sum_e weights[e] * down[e]. Replaces fk_moe_combine + fk_add.
__global__ void fk_moe_combine_add(
    const float * __restrict__ down, const float * __restrict__ weights,
    const float * __restrict__ residual, float * __restrict__ out,
    int n_used, int n_embd)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    float acc = 0.0f;
    for (int e = 0; e < n_used; ++e) acc += weights[e] * down[(size_t)e * n_embd + d];
    out[d] = residual[d] + acc;
}

// The embedding lookup, reading the token id from the device-resident meta
// buffer rather than taking it as an argument. See the graph-capture note in
// llamd_fwd_begin_capture: a captured graph's kernel arguments are fixed, so
// anything that changes per token must be read from memory.
__global__ void fk_embed_q6k_meta(
    const uint8_t * __restrict__ table, const int32_t * __restrict__ meta,
    float * __restrict__ out, int blocks)
{
    const uint8_t *row = table + (size_t)meta[0] * blocks * Q6K_BLOCK_BYTES;
    for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
        q6k_decode_block(row + (size_t)b * Q6K_BLOCK_BYTES, out + b * QK_K,
                         threadIdx.x, blockDim.x);
    }
}

// Attention reading n_kv from the meta buffer, for the same reason.
__global__ void fk_attention_meta(
    const float * __restrict__ q, const float * __restrict__ kc,
    const float * __restrict__ vc, float * __restrict__ out,
    float * __restrict__ scores, int n_head_kv, int hd,
    const int32_t * __restrict__ meta, int ratio, int max_kv)
{
    __shared__ float s[FWD_BLOCK];
    const int n_kv = meta[1] + 1;
    const int h   = blockIdx.x;
    const int kvh = h / ratio;
    const int kv_dim = n_head_kv * hd;
    const float scale = rsqrtf((float)hd);

    const float *qh = q + (size_t)h * hd;
    // The scores buffer is [n_head][max_kv]; n_kv varies per token but the
    // stride does not, which is what lets a captured graph reuse it.
    float *sc = scores + (size_t)h * max_kv;

    float local_max = -INFINITY;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float *kt = kc + (size_t)t * kv_dim + (size_t)kvh * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
        const float sv = dot * scale;
        sc[t] = sv;
        local_max = fmaxf(local_max, sv);
    }
    const float mx = block_reduce_max(local_max, s);

    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float e = __expf(sc[t] - mx);
        sc[t] = e;
        local_sum += e;
    }
    const float sum = block_reduce_sum(local_sum, s);
    const float inv = 1.0f / sum;
    __syncthreads();

    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            acc += sc[t] * vc[(size_t)t * kv_dim + (size_t)kvh * hd + d];
        }
        out[(size_t)h * hd + d] = acc * inv;
    }
}

// F32 matvec, for the router.
//
// The MoE gate (ffn_gate_inp) is stored F32, not Q6_K — 241 of this model's
// tensors are F32: the norms and the 48 routers. It is small ([2048, 128]) but
// it decides which experts run, so it cannot be skipped or approximated.
// One wave per output row, matching the quantized kernel's shape.
__global__ void fk_matvec_f32(
    const float * __restrict__ w, const float * __restrict__ x,
    float * __restrict__ out, int k)
{
    // ONE WORKGROUP PER ROW, not one wave.
    //
    // The first version gave each row a single 32-lane wave and packed eight
    // rows into a workgroup. For the MoE router — [2048, 128], the only F32
    // matvec in the model — that is 16 workgroups on a 32-WGP card, so half the
    // GPU idles and the rest is latency-bound on 8 KiB per row. The router phase
    // was 13.9% of a token for 1 MiB of reads.
    //
    // One workgroup of FWD_BLOCK threads per row gives 128 workgroups and eight
    // elements per thread, then one block reduction.
    __shared__ float s[FWD_BLOCK];
    const int row = blockIdx.x;
    const float *wr = w + (size_t)row * k;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) acc += wr[i] * x[i];
    const float total = block_reduce_sum(acc, s);
    if (threadIdx.x == 0) out[row] = total;
}

// ---------------------------------------------------------------------------
// STREAM AND GRAPH CAPTURE.
//
// WHY A GRAPH. About 1200 kernels go into a token. Each hipLaunchKernelGGL
// costs a few microseconds of driver work — command packet assembly, argument
// marshalling, doorbell — and measured against the phase breakdown that came to
// roughly 5.4 of 14.87 ms, more than any single phase of actual computation.
// A graph pays that cost ONCE, at instantiation, and each replay submits the
// whole token as one operation. llama.cpp builds with GGML_HIP_GRAPHS=ON for
// exactly this reason.
//
// WHAT MADE IT POSSIBLE. A captured graph's kernel arguments are frozen. The
// token id, the position and n_kv change every token, so as long as they were
// kernel arguments the graph had to be rebuilt each time — which costs more
// than it saves. They now live in a two-int device buffer that the kernels
// READ, so the graph is structurally invariant and the only per-token host work
// is a 8-byte upload. Expert routing was never a problem: the ids are chosen on
// the device and consumed there, so the graph does not branch on them.
//
// Everything launches on g_stream so capture can intercept it. The default
// stream cannot be captured.
static hipStream_t  g_stream = nullptr;
static hipGraphExec_t g_exec = nullptr;
static hipGraph_t     g_graph = nullptr;

extern "C" {

// The stream, for the entry points in llamd.hip. They are a separate
// translation unit and cannot see the static, but their launches MUST go to the
// same stream: a launch on the default stream during capture is not captured,
// and mixing a captured stream with an uncaptured one is an error.
void *llamd_fwd_get_stream() { return (void *)g_stream; }

// Create the stream all launches use. Idempotent.
int llamd_fwd_stream_init() {
    if (g_stream) return 0;
    return hipStreamCreate(&g_stream) == hipSuccess ? 0 : -1;
}

void *llamd_fwd_alloc(size_t bytes) {
    void *p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    return p;
}

void llamd_fwd_free(void *p) { if (p) (void)hipFree(p); }

int llamd_fwd_upload(void *dst, const void *src, size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess ? 0 : -1;
}

int llamd_fwd_download(void *dst, const void *src, size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

int llamd_fwd_memset(void *p, int value, size_t bytes) {
    return hipMemsetAsync(p, value, bytes, g_stream) == hipSuccess ? 0 : -1;
}

int llamd_fwd_sync() {
    // The stream, not the device: everything this module queues goes to
    // g_stream, and a device-wide synchronise would also wait on unrelated
    // work (the matvec entry points in llamd.hip still use the default stream).
    if (g_stream) return hipStreamSynchronize(g_stream) == hipSuccess ? 0 : -1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : -1;
}

size_t llamd_fwd_free_mem() {
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) return 0;
    return f;
}

void llamd_fwd_embed(const void *table, int token, float *out, int blocks) {
    hipLaunchKernelGGL(fk_embed_q6k, dim3(blocks < 256 ? blocks : 256),
                       dim3(FWD_BLOCK), 0, g_stream,
                       (const uint8_t *)table, token, out, blocks);
}

void llamd_fwd_rms_norm_mul(const float *x, const float *w, float *out,
                            int n, float eps) {
    hipLaunchKernelGGL(fk_rms_norm_mul, dim3(1), dim3(FWD_BLOCK), 0, g_stream,
                       x, w, out, n, eps);
}

void llamd_fwd_head_rms_norm_mul(const float *x, const float *w, float *out,
                                 int n_head, int hd, float eps) {
    const int threads = hd < FWD_BLOCK ? hd : FWD_BLOCK;
    hipLaunchKernelGGL(fk_head_rms_norm_mul, dim3(n_head), dim3(threads),
                       threads * sizeof(float), g_stream, x, w, out, hd, eps);
}

void llamd_fwd_rope(float *x, int pos, int n_head, int hd, float freq_base) {
    const int threads = (hd / 2) < FWD_BLOCK ? (hd / 2) : FWD_BLOCK;
    hipLaunchKernelGGL(fk_rope_neox, dim3(n_head), dim3(threads), 0, g_stream,
                       x, pos, hd, freq_base);
}

void llamd_fwd_kv_store(float *kc, float *vc, const float *k, const float *v,
                        int pos, int kv_dim) {
    hipLaunchKernelGGL(fk_kv_store, dim3((kv_dim + 255) / 256), dim3(256), 0, g_stream,
                       kc, vc, k, v, pos, kv_dim);
}

void llamd_fwd_attention(const float *q, const float *kc, const float *vc,
                         float *out, float *scores, int n_head, int n_head_kv,
                         int hd, int n_kv) {
    hipLaunchKernelGGL(fk_attention, dim3(n_head), dim3(FWD_BLOCK), 0, g_stream,
                       q, kc, vc, out, scores, n_head_kv, hd, n_kv,
                       n_head / n_head_kv);
}

void llamd_fwd_matvec_f32(const float *w, const float *x, float *out,
                          int k, int rows) {
    hipLaunchKernelGGL(fk_matvec_f32, dim3(rows), dim3(FWD_BLOCK),
                       0, g_stream, w, x, out, k);
}

void llamd_fwd_add(const float *a, const float *b, float *out, int n) {
    hipLaunchKernelGGL(fk_add, dim3((n + 255) / 256), dim3(256), 0, g_stream, a, b, out, n);
}

void llamd_fwd_silu_mul(const float *gate, const float *up, float *out, int n) {
    hipLaunchKernelGGL(fk_silu_mul, dim3((n + 255) / 256), dim3(256), 0, g_stream,
                       gate, up, out, n);
}

void llamd_fwd_router(const float *logits, int n_expert, int n_used,
                      int32_t *ids, float *weights, float *probs) {
    hipLaunchKernelGGL(fk_router_topk, dim3(1), dim3(FWD_BLOCK), 0, g_stream,
                       logits, n_expert, n_used, ids, weights, probs);
}

void llamd_fwd_moe_combine(const float *down, const float *weights, float *out,
                           int n_used, int n_embd) {
    hipLaunchKernelGGL(fk_moe_combine, dim3((n_embd + 255) / 256), dim3(256), 0, g_stream,
                       down, weights, out, n_used, n_embd);
}

void llamd_fwd_argmax(const float *x, int n, float *scratch_v, int32_t *scratch_i,
                      int32_t *out, int nblocks) {
    hipLaunchKernelGGL(fk_argmax_partial, dim3(nblocks), dim3(FWD_BLOCK), 0, g_stream,
                       x, n, scratch_v, scratch_i);
    hipLaunchKernelGGL(fk_argmax_final, dim3(1), dim3(1), 0, g_stream,
                       scratch_v, scratch_i, nblocks, out);
}

void llamd_fwd_norm_rope(float *x, const float *w, int n_head, int hd,
                         const int32_t *meta, float freq_base, float eps) {
    const int threads = hd < FWD_BLOCK ? hd : FWD_BLOCK;
    hipLaunchKernelGGL(fk_norm_rope, dim3(n_head), dim3(threads),
                       threads * sizeof(float), g_stream, x, w, hd, meta,
                       freq_base, eps);
}

void llamd_fwd_k_norm_rope_store(float *k, const float *w, const float *v,
                                 float *kc, float *vc, int n_head_kv, int hd,
                                 int kv_dim, const int32_t *meta,
                                 float freq_base, float eps) {
    const int threads = hd < FWD_BLOCK ? hd : FWD_BLOCK;
    hipLaunchKernelGGL(fk_k_norm_rope_store, dim3(n_head_kv), dim3(threads),
                       threads * sizeof(float), g_stream,
                       k, w, v, kc, vc, hd, kv_dim, meta, freq_base, eps);
}

void llamd_fwd_rms_norm_quant(const float *x, const float *w, float *out,
                              int8_t *qs, float *d, float *inv_scratch,
                              int n, float eps) {
    hipLaunchKernelGGL(fk_rms_scale, dim3(1), dim3(FWD_BLOCK), 0, g_stream,
                       x, inv_scratch, n, eps);
    // One workgroup per 256-element quantization block, QK_K threads each, so
    // every block's max reduction happens at the same time as the others'.
    hipLaunchKernelGGL(fk_norm_quant, dim3(n / QK_K), dim3(QK_K), 0, g_stream,
                       x, w, inv_scratch, out, qs, d);
}

void llamd_fwd_moe_combine_add(const float *down, const float *weights,
                               const float *residual, float *out,
                               int n_used, int n_embd) {
    hipLaunchKernelGGL(fk_moe_combine_add, dim3((n_embd + 255) / 256), dim3(256),
                       0, g_stream, down, weights, residual, out, n_used, n_embd);
}

void llamd_fwd_embed_meta(const void *table, const int32_t *meta, float *out,
                          int blocks) {
    hipLaunchKernelGGL(fk_embed_q6k_meta, dim3(blocks < 256 ? blocks : 256),
                       dim3(FWD_BLOCK), 0, g_stream,
                       (const uint8_t *)table, meta, out, blocks);
}

void llamd_fwd_attention_meta(const float *q, const float *kc, const float *vc,
                              float *out, float *scores, int n_head,
                              int n_head_kv, int hd, const int32_t *meta,
                              int max_kv) {
    hipLaunchKernelGGL(fk_attention_meta, dim3(n_head), dim3(FWD_BLOCK), 0, g_stream,
                       q, kc, vc, out, scores, n_head_kv, hd, meta,
                       n_head / n_head_kv, max_kv);
}

// ---- graph capture ---------------------------------------------------------
//
// Capture is NOT a dry run: nothing executes between begin and end. So the
// buffers a captured token reads must already hold sane values, and the first
// real result comes from the first replay, not from the capture.
int llamd_fwd_begin_capture() {
    if (!g_stream) return -1;
    if (g_exec) { (void)hipGraphExecDestroy(g_exec); g_exec = nullptr; }
    if (g_graph) { (void)hipGraphDestroy(g_graph); g_graph = nullptr; }
    return hipStreamBeginCapture(g_stream, hipStreamCaptureModeGlobal) == hipSuccess
        ? 0 : -1;
}

int llamd_fwd_end_capture() {
    if (!g_stream) return -1;
    if (hipStreamEndCapture(g_stream, &g_graph) != hipSuccess) return -1;
    if (hipGraphInstantiate(&g_exec, g_graph, nullptr, nullptr, 0) != hipSuccess)
        return -1;
    return 0;
}

int llamd_fwd_graph_launch() {
    if (!g_exec) return -1;
    return hipGraphLaunch(g_exec, g_stream) == hipSuccess ? 0 : -1;
}

int llamd_fwd_has_graph() { return g_exec ? 1 : 0; }

void llamd_fwd_graph_destroy() {
    if (g_exec) { (void)hipGraphExecDestroy(g_exec); g_exec = nullptr; }
    if (g_graph) { (void)hipGraphDestroy(g_graph); g_graph = nullptr; }
}

// Did anything in the queued work fail? Checked once per token, after the
// synchronise, so a launch error is reported rather than producing silence.
int llamd_fwd_last_error() {
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // extern "C"
