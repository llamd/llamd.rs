// Batched decode: several sequences per pass, sharing every weight read.
//
// WHY. At batch 1 a token reads 2.485 GB of weights and the engine sustains
// ~260 GB/s cold, which is 9.85 ms and ~102 tok/s. Three kernel-level attempts
// to raise that bandwidth were rejected by measurement (HIP graphs, split-k,
// planar Q6_K — benchmarks.md §8-§10). What remains is to read each weight once
// for MORE THAN ONE token.
//
// WHAT BATCHING CAN AND CANNOT SHARE, because the difference decides the result:
//
//   attention projections  743.2 MB/token   SHARED — every sequence multiplies
//                                           the same Q, K, V, O matrices
//   output projection      255.3 MB/token   SHARED
//   routed experts        1486.4 MB/token   NOT SHARED — each sequence routes
//                                           to its own 8 of 128, and at batch 2
//                                           only ~3% of those coincide
//
// So per-token traffic goes 2485 -> 1986 MB at batch 2 (1.25x) and -> 1736 at
// batch 4 (1.43x). The experts are 60% of the bytes and they do not divide.
// This is a property of mixture-of-experts, not of the implementation.
//
// AND IT IS AGGREGATE THROUGHPUT, NOT LATENCY. Token N+1 of a sequence depends
// on token N, so the sequences here must be INDEPENDENT — different streams, or
// speculative candidates. Each individual sequence sees roughly the batch-1
// latency; what improves is tokens per second across all of them.
//
// PROVENANCE: the Q6_K arithmetic follows llama.cpp / ggml (MIT, (c) 2023-2026
// The ggml authors), unchanged.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

#define QK_K            256
#define Q6K_BLOCK_BYTES 210
#define WAVE            32
#define BATCH_BLOCK     256
#define BWAVES          8      // waves per workgroup
#define BROWS           8      // output rows per workgroup
#define MAX_SEQ         8      // register arrays are sized by this

// The stream every launch uses, owned by llamd_forward.hip. MUST be declared
// extern "C": without it the name is C++-mangled and the link fails with an
// undefined symbol that differs from the defined one only by mangling.
extern "C" void *llamd_fwd_get_stream();
static inline hipStream_t bstream() { return (hipStream_t)llamd_fwd_get_stream(); }

extern "C" {

// ---------------------------------------------------------------------------
// The batched Q6_K row dot product — THE WHOLE POINT OF THIS FILE.
//
// The weight block is decoded ONCE and then multiplied against `nseq`
// activations. That is the only reason batching is faster: the 210 bytes cross
// the memory bus once and do nseq tokens' worth of work.
//
// Per-term arithmetic is identical to the single-sequence kernel; each
// sequence's accumulator sees exactly the terms it would have seen alone, in
// the same order. Batching therefore cannot change any token it would not have
// changed at batch 1 — asserted by comparing batch-2 output against two
// separate batch-1 forwards.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void q6k_lane_partial_batched(
    const uint8_t * __restrict__ wrow,
    const int8_t  * __restrict__ aq,   // [nseq][blocks*QK_K]
    const float   * __restrict__ ad,   // [nseq][blocks]
    int blocks, int lane, int nseq, int aq_stride, int ad_stride,
    float *acc /* [nseq] */)
{
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float dw = __half2float(__ushort_as_half(dbits));

        int32_t iacc[MAX_SEQ];
        #pragma unroll
        for (int s = 0; s < MAX_SEQ; ++s) iacc[s] = 0;

        for (int half = 0; half < 2; ++half) {
            const int o = half * 128;
            for (int l = lane; l < 32; l += WAVE) {
                // Decoded once, used nseq times. Everything below this line
                // that touches `w` happens exactly once per weight byte.
                const uint8_t h   = qh[half * 32 + l];
                const uint8_t lo0 = ql[half * 64 + l];
                const uint8_t lo1 = ql[half * 64 + l + 32];
                const int v0 = (int)(int8_t)((lo0 & 0x0F) | (((h >> 0) & 3) << 4)) - 32;
                const int v1 = (int)(int8_t)((lo1 & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
                const int v2 = (int)(int8_t)((lo0 >>   4) | (((h >> 4) & 3) << 4)) - 32;
                const int v3 = (int)(int8_t)((lo1 >>   4) | (((h >> 6) & 3) << 4)) - 32;
                const int s0 = sc[(o + l) / 16],       s1 = sc[(o + l + 32) / 16];
                const int s2 = sc[(o + l + 64) / 16],  s3 = sc[(o + l + 96) / 16];

                for (int s = 0; s < nseq; ++s) {
                    const int8_t *q8 = aq + (size_t)s * aq_stride + (size_t)b * QK_K;
                    iacc[s] += s0 * v0 * (int)q8[o + l];
                    iacc[s] += s1 * v1 * (int)q8[o + l + 32];
                    iacc[s] += s2 * v2 * (int)q8[o + l + 64];
                    iacc[s] += s3 * v3 * (int)q8[o + l + 96];
                }
            }
        }
        for (int s = 0; s < nseq; ++s) {
            acc[s] += dw * ad[(size_t)s * ad_stride + b] * (float)iacc[s];
        }
    }
}

__device__ __forceinline__ float bwave_reduce(float a) {
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) a += __shfl_down(a, off, WAVE);
    return a;
}

// The same, reading the activation from LDS.
//
// WHY. The global-memory version above re-reads each sequence's activation for
// every output row: 8 rows per workgroup times nseq sequences is up to 64 reads
// of the same few kilobytes, and at batch 8 that dominates. The activation is
// small — nseq * blocks * 256 bytes, 16 KiB at nseq=8 and blocks=8 — so a
// workgroup can stage it once in LDS and every row reads it at LDS latency.
//
// Measured against llama.cpp's own batched harness, llamd scaled 1.40x/1.82x/
// 2.16x at batch 2/4/8 where llama.cpp scaled 2.09x/3.28x/4.16x. llama.cpp
// switches to matrix-matrix kernels with real tiling at batch; this is the
// cheapest step in that direction.
__device__ __forceinline__ void q6k_lane_partial_lds(
    const uint8_t * __restrict__ wrow,
    const int8_t  * __restrict__ saq,   // LDS: [nseq][blocks*QK_K]
    const float   * __restrict__ sad,   // LDS: [nseq][blocks]
    int blocks, int lane, int nseq, float *acc)
{
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float dw = __half2float(__ushort_as_half(dbits));

        int32_t iacc[MAX_SEQ];
        #pragma unroll
        for (int s = 0; s < MAX_SEQ; ++s) iacc[s] = 0;

        for (int half = 0; half < 2; ++half) {
            const int o = half * 128;
            for (int l = lane; l < 32; l += WAVE) {
                const uint8_t h   = qh[half * 32 + l];
                const uint8_t lo0 = ql[half * 64 + l];
                const uint8_t lo1 = ql[half * 64 + l + 32];
                const int v0 = (int)(int8_t)((lo0 & 0x0F) | (((h >> 0) & 3) << 4)) - 32;
                const int v1 = (int)(int8_t)((lo1 & 0x0F) | (((h >> 2) & 3) << 4)) - 32;
                const int v2 = (int)(int8_t)((lo0 >>   4) | (((h >> 4) & 3) << 4)) - 32;
                const int v3 = (int)(int8_t)((lo1 >>   4) | (((h >> 6) & 3) << 4)) - 32;
                const int c0 = sc[(o + l) / 16],       c1 = sc[(o + l + 32) / 16];
                const int c2 = sc[(o + l + 64) / 16],  c3 = sc[(o + l + 96) / 16];

                for (int s = 0; s < nseq; ++s) {
                    const int8_t *q8 = saq + (size_t)s * blocks * QK_K + (size_t)b * QK_K;
                    iacc[s] += c0 * v0 * (int)q8[o + l];
                    iacc[s] += c1 * v1 * (int)q8[o + l + 32];
                    iacc[s] += c2 * v2 * (int)q8[o + l + 64];
                    iacc[s] += c3 * v3 * (int)q8[o + l + 96];
                }
            }
        }
        for (int s = 0; s < nseq; ++s) acc[s] += dw * sad[s * blocks + b] * (float)iacc[s];
    }
}

// Stage nseq activations into LDS. Every thread participates; the caller
// barriers afterwards.
__device__ __forceinline__ void stage_activation(
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    int8_t *saq, float *sad, int blocks, int nseq)
{
    const int nq = nseq * blocks * QK_K;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) saq[i] = aq[i];
    const int nd = nseq * blocks;
    for (int i = threadIdx.x; i < nd; i += blockDim.x) sad[i] = ad[i];
}

// Batched matvec against one matrix. `out` is [nseq][rows].
__global__ void q6k_batched_matvec(
    const uint8_t * __restrict__ w,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out,
    int blocks, int rows, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * BROWS + wave;
    if (row >= rows) return;

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;

    q6k_lane_partial_batched(w + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                             aq, ad, blocks, lane, nseq,
                             blocks * QK_K, blocks, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = bwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

// Batched matvec, activation staged in LDS.
__global__ void q6k_batched_matvec_lds(
    const uint8_t * __restrict__ w,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out,
    int blocks, int rows, int nseq)
{
    extern __shared__ char lds[];
    int8_t *saq = (int8_t *)lds;
    float  *sad = (float *)(lds + nseq * blocks * QK_K);
    stage_activation(aq, ad, saq, sad, blocks, nseq);
    __syncthreads();

    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * BROWS + wave;
    if (row >= rows) return;

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;
    q6k_lane_partial_lds(w + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                         saq, sad, blocks, lane, nseq, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = bwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

// Batched Q, K and V with the activation in LDS.
__global__ void q6k_batched_x3_lds(
    const uint8_t * __restrict__ w0, const uint8_t * __restrict__ w1,
    const uint8_t * __restrict__ w2,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2,
    int blocks, int rows0, int rows1, int rows2, int nseq)
{
    extern __shared__ char lds[];
    int8_t *saq = (int8_t *)lds;
    float  *sad = (float *)(lds + nseq * blocks * QK_K);
    stage_activation(aq, ad, saq, sad, blocks, nseq);
    __syncthreads();

    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * BROWS + wave;
    if (grow >= rows0 + rows1 + rows2) return;

    const uint8_t *w; float *out; int row, rows;
    if (grow < rows0)              { w=w0; out=o0; row=grow;             rows=rows0; }
    else if (grow < rows0+rows1)   { w=w1; out=o1; row=grow-rows0;       rows=rows1; }
    else                           { w=w2; out=o2; row=grow-rows0-rows1; rows=rows2; }

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;
    q6k_lane_partial_lds(w + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                         saq, sad, blocks, lane, nseq, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = bwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

// Batched Q, K and V: three matrices, nseq sequences, one launch.
__global__ void q6k_batched_x3(
    const uint8_t * __restrict__ w0, const uint8_t * __restrict__ w1,
    const uint8_t * __restrict__ w2,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2,
    int blocks, int rows0, int rows1, int rows2, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * BROWS + wave;
    if (grow >= rows0 + rows1 + rows2) return;

    const uint8_t *w; float *out; int row, rows;
    if (grow < rows0)              { w=w0; out=o0; row=grow;               rows=rows0; }
    else if (grow < rows0+rows1)   { w=w1; out=o1; row=grow-rows0;         rows=rows1; }
    else                           { w=w2; out=o2; row=grow-rows0-rows1;   rows=rows2; }

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;
    q6k_lane_partial_batched(w + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                             aq, ad, blocks, lane, nseq,
                             blocks * QK_K, blocks, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = bwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

// ---------------------------------------------------------------------------
// The expert path. NO WEIGHT SHARING IS POSSIBLE HERE, and pretending otherwise
// would be the bug: sequence 0 routes to its own 8 experts, sequence 1 to its
// own, and at batch 2 only about 3% of those coincide. So the row space is
// nseq * n_used * rows and each (sequence, expert) pair reads its own matrix.
//
// It is still ONE launch rather than nseq, which matters for occupancy: the
// expert matrices are small enough that issuing them per sequence leaves the
// card short of work.
// ---------------------------------------------------------------------------
__global__ void q6k_batched_grouped_x2(
    const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb,
    const int32_t * __restrict__ ids,      // [nseq][n_used]
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ oa, float * __restrict__ ob,
    int blocks, int rows, int n_used, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * BROWS + wave;
    const int per_seq = rows * n_used;
    const int total   = per_seq * nseq;
    if (grow >= 2 * total) return;

    const bool second = grow >= total;
    const int idx = second ? grow - total : grow;
    const uint8_t *w = second ? wb : wa;
    float *out = second ? ob : oa;

    const int s   = idx / per_seq;
    const int r0  = idx - s * per_seq;
    const int e   = r0 / rows;
    const int row = r0 - e * rows;
    const size_t mat = (size_t)ids[s * n_used + e] * rows * blocks * Q6K_BLOCK_BYTES;

    // One sequence's activation only — there is nothing to share.
    float acc[MAX_SEQ];
    acc[0] = 0.0f;
    q6k_lane_partial_batched(w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                             aq + (size_t)s * blocks * QK_K,
                             ad + (size_t)s * blocks,
                             blocks, lane, 1, 0, 0, acc);
    const float t = bwave_reduce(acc[0]);
    if (lane == 0) out[idx] = t;
}

// The down projection: per (sequence, expert) activation.
__global__ void q6k_batched_grouped_pa(
    const uint8_t * __restrict__ w,
    const int32_t * __restrict__ ids,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out,
    int blocks, int rows, int n_used, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * BROWS + wave;
    const int per_seq = rows * n_used;
    if (grow >= per_seq * nseq) return;

    const int s   = grow / per_seq;
    const int r0  = grow - s * per_seq;
    const int e   = r0 / rows;
    const int row = r0 - e * rows;
    const size_t mat = (size_t)ids[s * n_used + e] * rows * blocks * Q6K_BLOCK_BYTES;
    // Activation index is (sequence, expert): each expert consumed its own
    // silu(gate)*up for this sequence.
    const size_t a = ((size_t)s * n_used + e) * blocks;

    float acc[MAX_SEQ];
    acc[0] = 0.0f;
    q6k_lane_partial_batched(w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES,
                             aq + a * QK_K, ad + a, blocks, lane, 1, 0, 0, acc);
    const float t = bwave_reduce(acc[0]);
    if (lane == 0) out[grow] = t;
}

// ---------------------------------------------------------------------------
// Everything else, given a sequence dimension. These are per-sequence
// independent, so the only change is that the grid covers nseq as well.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float breduce_sum(float v, float *s) {
    const int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) s[t] += s[t + st];
        __syncthreads();
    }
    return s[0];
}

__device__ __forceinline__ float breduce_max(float v, float *s) {
    const int t = threadIdx.x;
    s[t] = v;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) s[t] = fmaxf(s[t], s[t + st]);
        __syncthreads();
    }
    return s[0];
}

// One workgroup per sequence: 1/rms of that sequence's vector.
__global__ void bk_rms_scale(
    const float * __restrict__ x, float * __restrict__ inv, int n, float eps)
{
    __shared__ float sh[BATCH_BLOCK];
    const float *xi = x + (size_t)blockIdx.x * n;
    float a = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a += xi[i] * xi[i];
    const float tot = breduce_sum(a, sh);
    if (threadIdx.x == 0) inv[blockIdx.x] = rsqrtf(tot / (float)n + eps);
}

// One workgroup per (sequence, quantization block).
__global__ void bk_norm_quant(
    const float * __restrict__ x, const float * __restrict__ w,
    const float * __restrict__ inv, float * __restrict__ out,
    int8_t * __restrict__ qs, float * __restrict__ d, int nblocks)
{
    __shared__ float sa[QK_K];
    __shared__ float sm[QK_K];
    const int s = blockIdx.x / nblocks;
    const int b = blockIdx.x - s * nblocks;
    const size_t base = (size_t)s * nblocks * QK_K;
    const int i = b * QK_K + threadIdx.x;
    const float sc = inv[s];

    const float v = x[base + i] * sc * w[i];
    out[base + i] = v;

    sa[threadIdx.x] = fabsf(v);
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int st = QK_K / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            if (sa[threadIdx.x + st] > sa[threadIdx.x]) {
                sa[threadIdx.x] = sa[threadIdx.x + st];
                sm[threadIdx.x] = sm[threadIdx.x + st];
            }
        }
        __syncthreads();
    }
    const float amax = sa[0], mx = sm[0];
    if (amax == 0.0f) {
        qs[base + i] = 0;
        if (threadIdx.x == 0) d[(size_t)s * nblocks + b] = 0.0f;
        return;
    }
    const float iscale = -127.0f / mx;
    int q = (int)rintf(iscale * v);
    if (q > 127) q = 127;
    qs[base + i] = (int8_t)q;
    if (threadIdx.x == 0) d[(size_t)s * nblocks + b] = 1.0f / iscale;
}

// Plain rms_norm * w, per sequence (the final norm, whose output is not
// quantized here because the output projection quantizes it itself).
__global__ void bk_rms_norm_mul(
    const float * __restrict__ x, const float * __restrict__ w,
    float * __restrict__ out, int n, float eps)
{
    __shared__ float sh[BATCH_BLOCK];
    const size_t base = (size_t)blockIdx.x * n;
    float a = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a += x[base + i] * x[base + i];
    const float tot = breduce_sum(a, sh);
    const float inv = rsqrtf(tot / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[base + i] = x[base + i] * inv * w[i];
}

__global__ void bk_quantize(
    const float * __restrict__ x, int8_t * __restrict__ qs,
    float * __restrict__ d, int nblocks_total)
{
    __shared__ float sa[QK_K];
    __shared__ float sm[QK_K];
    const int b = blockIdx.x;
    if (b >= nblocks_total) return;
    const size_t i = (size_t)b * QK_K + threadIdx.x;
    const float v = x[i];
    sa[threadIdx.x] = fabsf(v);
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int st = QK_K / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            if (sa[threadIdx.x + st] > sa[threadIdx.x]) {
                sa[threadIdx.x] = sa[threadIdx.x + st];
                sm[threadIdx.x] = sm[threadIdx.x + st];
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

// Per-head norm then RoPE, for every sequence. grid = nseq * n_head.
__global__ void bk_norm_rope(
    float * __restrict__ x, const float * __restrict__ w,
    int n_head, int hd, const int32_t * __restrict__ meta,
    float freq_base, float eps)
{
    extern __shared__ float sh[];
    const int s    = blockIdx.x / n_head;
    const int head = blockIdx.x - s * n_head;
    const int pos  = meta[2 * s + 1];
    float *xi = x + ((size_t)s * n_head + head) * hd;

    if (w) {
        float a = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) a += xi[i] * xi[i];
        sh[threadIdx.x] = a;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
            __syncthreads();
        }
        const float inv = rsqrtf(sh[0] / (float)hd + eps);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) xi[i] = xi[i] * inv * w[i];
        __syncthreads();
    }
    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float)pos / powf(freq_base, 2.0f * (float)i / (float)hd);
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float a = xi[i], b = xi[i + half];
        xi[i]        = a * cs - b * sn;
        xi[i + half] = a * sn + b * cs;
    }
}

// K side: norm, RoPE, store into that sequence's cache slice.
__global__ void bk_k_norm_rope_store(
    float * __restrict__ k, const float * __restrict__ w,
    const float * __restrict__ v, float * __restrict__ kc, float * __restrict__ vc,
    int n_head_kv, int hd, int kv_dim, int max_kv,
    const int32_t * __restrict__ meta, float freq_base, float eps)
{
    extern __shared__ float sh[];
    const int s    = blockIdx.x / n_head_kv;
    const int head = blockIdx.x - s * n_head_kv;
    const int pos  = meta[2 * s + 1];
    float *ki = k + ((size_t)s * n_head_kv + head) * hd;

    if (w) {
        float a = 0.0f;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) a += ki[i] * ki[i];
        sh[threadIdx.x] = a;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
            __syncthreads();
        }
        const float inv = rsqrtf(sh[0] / (float)hd + eps);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) ki[i] = ki[i] * inv * w[i];
        __syncthreads();
    }
    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float theta = (float)pos / powf(freq_base, 2.0f * (float)i / (float)hd);
        float sn, cs;
        __sincosf(theta, &sn, &cs);
        const float a = ki[i], b = ki[i + half];
        ki[i]        = a * cs - b * sn;
        ki[i + half] = a * sn + b * cs;
    }
    __syncthreads();

    // Each sequence owns a max_kv * kv_dim slice of the layer's cache.
    float *kcs = kc + (size_t)s * max_kv * kv_dim;
    float *vcs = vc + (size_t)s * max_kv * kv_dim;
    const float *vs = v + (size_t)s * kv_dim;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const int off = head * hd + i;
        kcs[(size_t)pos * kv_dim + off] = ki[i];
        vcs[(size_t)pos * kv_dim + off] = vs[off];
    }
}

// Attention. grid = nseq * n_head; each sequence attends only to its own cache.
__global__ void bk_attention(
    const float * __restrict__ q, const float * __restrict__ kc,
    const float * __restrict__ vc, float * __restrict__ out,
    float * __restrict__ scores, int n_head, int n_head_kv, int hd,
    int max_kv, const int32_t * __restrict__ meta, int ratio)
{
    __shared__ float sh[BATCH_BLOCK];
    const int s    = blockIdx.x / n_head;
    const int h    = blockIdx.x - s * n_head;
    const int n_kv = meta[2 * s + 1] + 1;
    const int kvh  = h / ratio;
    const int kv_dim = n_head_kv * hd;
    const float scale = rsqrtf((float)hd);

    const float *qh  = q + ((size_t)s * n_head + h) * hd;
    const float *kcs = kc + (size_t)s * max_kv * kv_dim;
    const float *vcs = vc + (size_t)s * max_kv * kv_dim;
    float *sc = scores + ((size_t)s * n_head + h) * max_kv;

    float lmax = -INFINITY;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float *kt = kcs + (size_t)t * kv_dim + (size_t)kvh * hd;
        float dot = 0.0f;
        for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
        const float sv = dot * scale;
        sc[t] = sv;
        lmax = fmaxf(lmax, sv);
    }
    const float mx = breduce_max(lmax, sh);
    float lsum = 0.0f;
    for (int t = threadIdx.x; t < n_kv; t += blockDim.x) {
        const float e = __expf(sc[t] - mx);
        sc[t] = e;
        lsum += e;
    }
    const float sum = breduce_sum(lsum, sh);
    const float inv = 1.0f / sum;
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float a = 0.0f;
        for (int t = 0; t < n_kv; ++t)
            a += sc[t] * vcs[(size_t)t * kv_dim + (size_t)kvh * hd + d];
        out[((size_t)s * n_head + h) * hd + d] = a * inv;
    }
}

__global__ void bk_add(const float * __restrict__ a, const float * __restrict__ b,
                       float * __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
}

__global__ void bk_silu_mul(const float * __restrict__ g, const float * __restrict__ u,
                            float * __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float x = g[i]; o[i] = (x / (1.0f + __expf(-x))) * u[i]; }
}

// F32 matvec for the router, batched: one workgroup per (sequence, row).
__global__ void bk_matvec_f32(
    const float * __restrict__ w, const float * __restrict__ x,
    float * __restrict__ out, int k, int rows)
{
    __shared__ float sh[BATCH_BLOCK];
    const int s   = blockIdx.x / rows;
    const int row = blockIdx.x - s * rows;
    const float *wr = w + (size_t)row * k;
    const float *xs = x + (size_t)s * k;
    float a = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) a += wr[i] * xs[i];
    const float tot = breduce_sum(a, sh);
    if (threadIdx.x == 0) out[(size_t)s * rows + row] = tot;
}

#define WEIGHT_SUM_FLOOR 6.103515625e-5f

// Router, one workgroup per sequence. Same selection rule as route(): softmax
// over every expert, top-k with ties to the LOWER index, divisor clamped at
// 2^-14.
__global__ void bk_router(
    const float * __restrict__ logits, int n_expert, int n_used,
    int32_t * __restrict__ ids, float * __restrict__ weights,
    float * __restrict__ probs)
{
    __shared__ float sh[BATCH_BLOCK];
    __shared__ float sp[1024];
    __shared__ int   sidx[BATCH_BLOCK];
    __shared__ float sval[BATCH_BLOCK];
    const int s = blockIdx.x;
    const float *lg = logits + (size_t)s * n_expert;
    float *pr = probs + (size_t)s * n_expert;
    int32_t *id = ids + (size_t)s * n_used;
    float *wt = weights + (size_t)s * n_used;

    float lmax = -INFINITY;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) lmax = fmaxf(lmax, lg[i]);
    const float mx = breduce_max(lmax, sh);
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
        const float e = __expf(lg[i] - mx);
        pr[i] = e;
        lsum += e;
    }
    const float tot = breduce_sum(lsum, sh);
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) pr[i] /= tot;
    __syncthreads();
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) sp[i] = pr[i];
    __syncthreads();

    for (int k = 0; k < n_used; ++k) {
        float bv = -INFINITY; int bi = -1;
        for (int i = threadIdx.x; i < n_expert; i += blockDim.x) {
            if (sp[i] > bv || (sp[i] == bv && i < bi)) { bv = sp[i]; bi = i; }
        }
        sval[threadIdx.x] = bv; sidx[threadIdx.x] = bi;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) {
                const int j = threadIdx.x + st;
                if (sval[j] > sval[threadIdx.x]
                    || (sval[j] == sval[threadIdx.x] && sidx[j] < sidx[threadIdx.x])) {
                    sval[threadIdx.x] = sval[j];
                    sidx[threadIdx.x] = sidx[j];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) { id[k] = sidx[0]; wt[k] = sval[0]; sp[sidx[0]] = -INFINITY; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float ws = 0.0f;
        for (int k = 0; k < n_used; ++k) ws += wt[k];
        ws = fmaxf(ws, WEIGHT_SUM_FLOOR);
        for (int k = 0; k < n_used; ++k) wt[k] /= ws;
    }
}

// out = residual + sum_e w[e]*down[e], per sequence.
__global__ void bk_moe_combine_add(
    const float * __restrict__ down, const float * __restrict__ weights,
    const float * __restrict__ residual, float * __restrict__ out,
    int n_used, int n_embd, int nseq)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd * nseq) return;
    const int s = i / n_embd;
    const int d = i - s * n_embd;
    const float *dn = down + (size_t)s * n_used * n_embd;
    const float *wt = weights + (size_t)s * n_used;
    float a = 0.0f;
    for (int e = 0; e < n_used; ++e) a += wt[e] * dn[(size_t)e * n_embd + d];
    out[i] = residual[i] + a;
}

// Embedding lookup for every sequence.
__global__ void bk_embed(
    const uint8_t * __restrict__ table, const int32_t * __restrict__ meta,
    float * __restrict__ out, int blocks, int nseq)
{
    const int s = blockIdx.y;
    if (s >= nseq) return;
    const uint8_t *row = table + (size_t)meta[2 * s] * blocks * Q6K_BLOCK_BYTES;
    float *o = out + (size_t)s * blocks * QK_K;
    for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
        const uint8_t *blk = row + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql = blk, *qh = blk + QK_K / 2;
        const int8_t *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t db = (uint16_t)blk[QK_K/2 + QK_K/4 + QK_K/16]
                          | ((uint16_t)blk[QK_K/2 + QK_K/4 + QK_K/16 + 1] << 8);
        const float d = __half2float(__ushort_as_half(db));
        float *ob = o + b * QK_K;
        for (int half = 0; half < 2; ++half) {
            const int off = half * 128;
            for (int l = threadIdx.x; l < 32; l += blockDim.x) {
                const uint8_t h = qh[half*32+l];
                const uint8_t lo0 = ql[half*64+l], lo1 = ql[half*64+l+32];
                const int v0 = (int)(int8_t)((lo0 & 0x0F) | (((h>>0)&3)<<4)) - 32;
                const int v1 = (int)(int8_t)((lo1 & 0x0F) | (((h>>2)&3)<<4)) - 32;
                const int v2 = (int)(int8_t)((lo0 >>   4) | (((h>>4)&3)<<4)) - 32;
                const int v3 = (int)(int8_t)((lo1 >>   4) | (((h>>6)&3)<<4)) - 32;
                ob[off+l]    = d * (float)sc[(off+l)/16]      * (float)v0;
                ob[off+l+32] = d * (float)sc[(off+l+32)/16]   * (float)v1;
                ob[off+l+64] = d * (float)sc[(off+l+64)/16]   * (float)v2;
                ob[off+l+96] = d * (float)sc[(off+l+96)/16]   * (float)v3;
            }
        }
    }
}

// Argmax per sequence: one workgroup per (sequence, chunk), then one combine.
__global__ void bk_argmax_partial(
    const float * __restrict__ x, int n, int nchunk,
    float * __restrict__ bv, int32_t * __restrict__ bi)
{
    __shared__ float sv[BATCH_BLOCK];
    __shared__ int   si[BATCH_BLOCK];
    const int s = blockIdx.y;
    const int c = blockIdx.x;
    const float *xs = x + (size_t)s * n;
    float v = -INFINITY; int idx = -1;
    for (int i = c * blockDim.x + threadIdx.x; i < n; i += nchunk * blockDim.x) {
        if (xs[i] > v) { v = xs[i]; idx = i; }
    }
    sv[threadIdx.x] = v; si[threadIdx.x] = idx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) {
            const int j = threadIdx.x + st;
            if (sv[j] > sv[threadIdx.x] || (sv[j] == sv[threadIdx.x] && si[j] < si[threadIdx.x])) {
                sv[threadIdx.x] = sv[j]; si[threadIdx.x] = si[j];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { bv[s * nchunk + c] = sv[0]; bi[s * nchunk + c] = si[0]; }
}

__global__ void bk_argmax_final(
    const float * __restrict__ bv, const int32_t * __restrict__ bi,
    int nchunk, int32_t * __restrict__ out)
{
    const int s = blockIdx.x;
    if (threadIdx.x != 0) return;
    float v = -INFINITY; int idx = -1;
    for (int i = 0; i < nchunk; ++i) {
        const float c = bv[s * nchunk + i];
        const int   j = bi[s * nchunk + i];
        if (c > v || (c == v && j < idx)) { v = c; idx = j; }
    }
    out[s] = idx;
}

// ---------------------------------------------------------------------------
// Launch surface. Device pointers, no synchronise, same stream as the
// single-sequence path so a capture would see all of it.
// ---------------------------------------------------------------------------
int llamd_batch_max_seq() { return MAX_SEQ; }

void llamd_b_embed(const void *table, const int32_t *meta, float *out,
                   int blocks, int nseq) {
    hipLaunchKernelGGL(bk_embed, dim3(blocks < 64 ? blocks : 64, nseq),
                       dim3(BATCH_BLOCK), 0, bstream(),
                       (const uint8_t *)table, meta, out, blocks, nseq);
}

void llamd_b_rms_norm_quant(const float *x, const float *w, float *out,
                            int8_t *qs, float *d, float *inv,
                            int n, int nseq, float eps) {
    hipLaunchKernelGGL(bk_rms_scale, dim3(nseq), dim3(BATCH_BLOCK), 0, bstream(),
                       x, inv, n, eps);
    hipLaunchKernelGGL(bk_norm_quant, dim3(nseq * (n / QK_K)), dim3(QK_K), 0, bstream(),
                       x, w, inv, out, qs, d, n / QK_K);
}

void llamd_b_rms_norm_mul(const float *x, const float *w, float *out,
                          int n, int nseq, float eps) {
    hipLaunchKernelGGL(bk_rms_norm_mul, dim3(nseq), dim3(BATCH_BLOCK), 0, bstream(),
                       x, w, out, n, eps);
}

void llamd_b_quantize(const float *x, int8_t *qs, float *d, int nblocks_total) {
    hipLaunchKernelGGL(bk_quantize, dim3(nblocks_total), dim3(QK_K), 0, bstream(),
                       x, qs, d, nblocks_total);
}

// LDS staging is worth it only when the activation fits without collapsing
// occupancy. 24 KiB leaves room for two workgroups per 64 KiB of LDS.
static inline size_t lds_bytes(int blocks, int nseq) {
    return (size_t)nseq * blocks * QK_K + (size_t)nseq * blocks * sizeof(float);
}
static inline bool lds_worth_it(int blocks, int nseq) {
    return nseq > 1 && lds_bytes(blocks, nseq) <= 24 * 1024;
}

void llamd_b_matvec(const void *w, const int8_t *aq, const float *ad, float *out,
                    int blocks, int rows, int nseq) {
    const int grid = (rows + BROWS - 1) / BROWS;
    if (lds_worth_it(blocks, nseq)) {
        hipLaunchKernelGGL(q6k_batched_matvec_lds, dim3(grid), dim3(WAVE * BWAVES),
                           lds_bytes(blocks, nseq), bstream(),
                           (const uint8_t *)w, aq, ad, out, blocks, rows, nseq);
        return;
    }
    hipLaunchKernelGGL(q6k_batched_matvec, dim3(grid),
                       dim3(WAVE * BWAVES), 0, bstream(),
                       (const uint8_t *)w, aq, ad, out, blocks, rows, nseq);
}

void llamd_b_matvec_x3(const void *w0, const void *w1, const void *w2,
                       const int8_t *aq, const float *ad,
                       float *o0, float *o1, float *o2,
                       int blocks, int r0, int r1, int r2, int nseq) {
    const int total = r0 + r1 + r2;
    const int grid = (total + BROWS - 1) / BROWS;
    if (lds_worth_it(blocks, nseq)) {
        hipLaunchKernelGGL(q6k_batched_x3_lds, dim3(grid), dim3(WAVE * BWAVES),
                           lds_bytes(blocks, nseq), bstream(),
                           (const uint8_t *)w0, (const uint8_t *)w1, (const uint8_t *)w2,
                           aq, ad, o0, o1, o2, blocks, r0, r1, r2, nseq);
        return;
    }
    hipLaunchKernelGGL(q6k_batched_x3, dim3(grid),
                       dim3(WAVE * BWAVES), 0, bstream(),
                       (const uint8_t *)w0, (const uint8_t *)w1, (const uint8_t *)w2,
                       aq, ad, o0, o1, o2, blocks, r0, r1, r2, nseq);
}

void llamd_b_grouped_x2(const void *wa, const void *wb, const int32_t *ids,
                        const int8_t *aq, const float *ad, float *oa, float *ob,
                        int blocks, int rows, int n_used, int nseq) {
    const int total = 2 * rows * n_used * nseq;
    hipLaunchKernelGGL(q6k_batched_grouped_x2, dim3((total + BROWS - 1) / BROWS),
                       dim3(WAVE * BWAVES), 0, bstream(),
                       (const uint8_t *)wa, (const uint8_t *)wb, ids,
                       aq, ad, oa, ob, blocks, rows, n_used, nseq);
}

void llamd_b_grouped_pa(const void *w, const int32_t *ids, const int8_t *aq,
                        const float *ad, float *out,
                        int blocks, int rows, int n_used, int nseq) {
    const int total = rows * n_used * nseq;
    hipLaunchKernelGGL(q6k_batched_grouped_pa, dim3((total + BROWS - 1) / BROWS),
                       dim3(WAVE * BWAVES), 0, bstream(),
                       (const uint8_t *)w, ids, aq, ad, out, blocks, rows, n_used, nseq);
}

void llamd_b_norm_rope(float *x, const float *w, int n_head, int hd,
                       const int32_t *meta, int nseq, float freq_base, float eps) {
    const int th = hd < BATCH_BLOCK ? hd : BATCH_BLOCK;
    hipLaunchKernelGGL(bk_norm_rope, dim3(nseq * n_head), dim3(th),
                       th * sizeof(float), bstream(),
                       x, w, n_head, hd, meta, freq_base, eps);
}

void llamd_b_k_norm_rope_store(float *k, const float *w, const float *v,
                               float *kc, float *vc, int n_head_kv, int hd,
                               int kv_dim, int max_kv, const int32_t *meta,
                               int nseq, float freq_base, float eps) {
    const int th = hd < BATCH_BLOCK ? hd : BATCH_BLOCK;
    hipLaunchKernelGGL(bk_k_norm_rope_store, dim3(nseq * n_head_kv), dim3(th),
                       th * sizeof(float), bstream(),
                       k, w, v, kc, vc, n_head_kv, hd, kv_dim, max_kv,
                       meta, freq_base, eps);
}

void llamd_b_attention(const float *q, const float *kc, const float *vc,
                       float *out, float *scores, int n_head, int n_head_kv,
                       int hd, int max_kv, const int32_t *meta, int nseq) {
    hipLaunchKernelGGL(bk_attention, dim3(nseq * n_head), dim3(BATCH_BLOCK), 0, bstream(),
                       q, kc, vc, out, scores, n_head, n_head_kv, hd, max_kv,
                       meta, n_head / n_head_kv);
}

void llamd_b_add(const float *a, const float *b, float *o, int n) {
    hipLaunchKernelGGL(bk_add, dim3((n + 255) / 256), dim3(256), 0, bstream(), a, b, o, n);
}

void llamd_b_silu_mul(const float *g, const float *u, float *o, int n) {
    hipLaunchKernelGGL(bk_silu_mul, dim3((n + 255) / 256), dim3(256), 0, bstream(), g, u, o, n);
}

void llamd_b_matvec_f32(const float *w, const float *x, float *out,
                        int k, int rows, int nseq) {
    hipLaunchKernelGGL(bk_matvec_f32, dim3(nseq * rows), dim3(BATCH_BLOCK), 0, bstream(),
                       w, x, out, k, rows);
}

void llamd_b_router(const float *logits, int n_expert, int n_used,
                    int32_t *ids, float *weights, float *probs, int nseq) {
    hipLaunchKernelGGL(bk_router, dim3(nseq), dim3(BATCH_BLOCK), 0, bstream(),
                       logits, n_expert, n_used, ids, weights, probs);
}

void llamd_b_moe_combine_add(const float *down, const float *weights,
                             const float *residual, float *out,
                             int n_used, int n_embd, int nseq) {
    const int n = n_embd * nseq;
    hipLaunchKernelGGL(bk_moe_combine_add, dim3((n + 255) / 256), dim3(256), 0, bstream(),
                       down, weights, residual, out, n_used, n_embd, nseq);
}

void llamd_b_argmax(const float *x, int n, float *sv, int32_t *si,
                    int32_t *out, int nchunk, int nseq) {
    hipLaunchKernelGGL(bk_argmax_partial, dim3(nchunk, nseq), dim3(BATCH_BLOCK),
                       0, bstream(), x, n, nchunk, sv, si);
    hipLaunchKernelGGL(bk_argmax_final, dim3(nseq), dim3(1), 0, bstream(),
                       sv, si, nchunk, out);
}

} // extern "C"
