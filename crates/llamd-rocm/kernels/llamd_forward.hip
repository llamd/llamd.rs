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

// F32 matvec, for the router.
//
// The MoE gate (ffn_gate_inp) is stored F32, not Q6_K — 241 of this model's
// tensors are F32: the norms and the 48 routers. It is small ([2048, 128]) but
// it decides which experts run, so it cannot be skipped or approximated.
// One wave per output row, matching the quantized kernel's shape.
__global__ void fk_matvec_f32(
    const float * __restrict__ w, const float * __restrict__ x,
    float * __restrict__ out, int k, int rows)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * (FWD_BLOCK / WAVE) + wave;
    if (row >= rows) return;
    const float *wr = w + (size_t)row * k;
    float acc = 0.0f;
    for (int i = lane; i < k; i += WAVE) acc += wr[i] * x[i];
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) acc += __shfl_down(acc, off, WAVE);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Device buffers and the launch surface.
//
// Every function below takes device pointers and returns immediately. None
// synchronises. That is the whole point: a token is ~1400 launches queued back
// to back, and the one hipDeviceSynchronize belongs after all of them.
// ---------------------------------------------------------------------------
extern "C" {

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
    return hipMemsetAsync(p, value, bytes, 0) == hipSuccess ? 0 : -1;
}

int llamd_fwd_sync() {
    return hipDeviceSynchronize() == hipSuccess ? 0 : -1;
}

size_t llamd_fwd_free_mem() {
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) return 0;
    return f;
}

void llamd_fwd_embed(const void *table, int token, float *out, int blocks) {
    hipLaunchKernelGGL(fk_embed_q6k, dim3(blocks < 256 ? blocks : 256),
                       dim3(FWD_BLOCK), 0, 0,
                       (const uint8_t *)table, token, out, blocks);
}

void llamd_fwd_rms_norm_mul(const float *x, const float *w, float *out,
                            int n, float eps) {
    hipLaunchKernelGGL(fk_rms_norm_mul, dim3(1), dim3(FWD_BLOCK), 0, 0,
                       x, w, out, n, eps);
}

void llamd_fwd_head_rms_norm_mul(const float *x, const float *w, float *out,
                                 int n_head, int hd, float eps) {
    const int threads = hd < FWD_BLOCK ? hd : FWD_BLOCK;
    hipLaunchKernelGGL(fk_head_rms_norm_mul, dim3(n_head), dim3(threads),
                       threads * sizeof(float), 0, x, w, out, hd, eps);
}

void llamd_fwd_rope(float *x, int pos, int n_head, int hd, float freq_base) {
    const int threads = (hd / 2) < FWD_BLOCK ? (hd / 2) : FWD_BLOCK;
    hipLaunchKernelGGL(fk_rope_neox, dim3(n_head), dim3(threads), 0, 0,
                       x, pos, hd, freq_base);
}

void llamd_fwd_kv_store(float *kc, float *vc, const float *k, const float *v,
                        int pos, int kv_dim) {
    hipLaunchKernelGGL(fk_kv_store, dim3((kv_dim + 255) / 256), dim3(256), 0, 0,
                       kc, vc, k, v, pos, kv_dim);
}

void llamd_fwd_attention(const float *q, const float *kc, const float *vc,
                         float *out, float *scores, int n_head, int n_head_kv,
                         int hd, int n_kv) {
    hipLaunchKernelGGL(fk_attention, dim3(n_head), dim3(FWD_BLOCK), 0, 0,
                       q, kc, vc, out, scores, n_head_kv, hd, n_kv,
                       n_head / n_head_kv);
}

void llamd_fwd_matvec_f32(const float *w, const float *x, float *out,
                          int k, int rows) {
    const int per = FWD_BLOCK / WAVE;
    hipLaunchKernelGGL(fk_matvec_f32, dim3((rows + per - 1) / per), dim3(FWD_BLOCK),
                       0, 0, w, x, out, k, rows);
}

void llamd_fwd_add(const float *a, const float *b, float *out, int n) {
    hipLaunchKernelGGL(fk_add, dim3((n + 255) / 256), dim3(256), 0, 0, a, b, out, n);
}

void llamd_fwd_silu_mul(const float *gate, const float *up, float *out, int n) {
    hipLaunchKernelGGL(fk_silu_mul, dim3((n + 255) / 256), dim3(256), 0, 0,
                       gate, up, out, n);
}

void llamd_fwd_router(const float *logits, int n_expert, int n_used,
                      int32_t *ids, float *weights, float *probs) {
    hipLaunchKernelGGL(fk_router_topk, dim3(1), dim3(FWD_BLOCK), 0, 0,
                       logits, n_expert, n_used, ids, weights, probs);
}

void llamd_fwd_moe_combine(const float *down, const float *weights, float *out,
                           int n_used, int n_embd) {
    hipLaunchKernelGGL(fk_moe_combine, dim3((n_embd + 255) / 256), dim3(256), 0, 0,
                       down, weights, out, n_used, n_embd);
}

void llamd_fwd_argmax(const float *x, int n, float *scratch_v, int32_t *scratch_i,
                      int32_t *out, int nblocks) {
    hipLaunchKernelGGL(fk_argmax_partial, dim3(nblocks), dim3(FWD_BLOCK), 0, 0,
                       x, n, scratch_v, scratch_i);
    hipLaunchKernelGGL(fk_argmax_final, dim3(1), dim3(1), 0, 0,
                       scratch_v, scratch_i, nblocks, out);
}

// Did anything in the queued work fail? Checked once per token, after the
// synchronise, so a launch error is reported rather than producing silence.
int llamd_fwd_last_error() {
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // extern "C"
