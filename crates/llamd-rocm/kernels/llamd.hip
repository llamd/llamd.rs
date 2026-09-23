// LLAMD kernels for RDNA4.
//
// WRITTEN FOR THIS ARCHITECTURE, not adapted from CUDA. That is the project's
// premise: llama.cpp's AMD support is its CUDA backend hipified at build time
// (ggml/src/ggml-hip/ contains one file, a CMakeLists), which inherits
// NVIDIA's warp size, tile shapes and occupancy assumptions.
//
// RDNA4 SPECIFICS THESE KERNELS ASSUME:
//   * wave32 is the native execution width. RDNA can run wave64, but wave32
//     is the default for compute and halves the cost of a divergent branch.
//   * 64 KiB of LDS per workgroup, but occupancy falls off well before that.
//   * No tensor/matrix cores usable from plain HIP on gfx1201, so the dot
//     product is a straightforward reduction rather than an MFMA fragment.
//
// THE FORMAT IS NOT NEGOTIABLE. These read the same Q6_K bytes ggml writes and
// must produce the same numbers as llamd-quant's scalar version, because a
// difference there flips a near-tied MoE expert and changes the emitted token.
//
// PROVENANCE: the Q6_K layout and the Q8_K dot product are derived from
// llama.cpp / ggml (MIT, (c) 2023-2026 The ggml authors).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K            256
#define Q6K_BLOCK_BYTES 210
#define WAVE            32      // RDNA4 native wave width

extern "C" {

// ---------------------------------------------------------------------------
// Device inventory. The first thing to get working: if this cannot report the
// card, nothing below is worth debugging.
// ---------------------------------------------------------------------------
int llamd_rocm_device_count() {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n;
}

int llamd_rocm_device_info(int dev, char *name, int name_len,
                           char *arch, int arch_len,
                           size_t *total_mem, int *wgp_count,
                           int *warp_size) {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) return -1;
    int i = 0;
    for (; i < name_len - 1 && p.name[i]; ++i) name[i] = p.name[i];
    name[i] = 0;
    // gcnArchName carries the full target ("gfx1201"); major/minor give only
    // 12.0 and would name a different binary than the one hipcc built.
    for (i = 0; i < arch_len - 1 && p.gcnArchName[i]; ++i) arch[i] = p.gcnArchName[i];
    arch[i] = 0;
    *total_mem = p.totalGlobalMem;
    // On RDNA this counts workgroup processors, not compute units: each WGP
    // holds two CUs, so the R9700's 64 CUs report as 32 here.
    *wgp_count = p.multiProcessorCount;
    *warp_size = p.warpSize;
    return 0;
}

size_t llamd_rocm_free_mem() {
    size_t f = 0, t = 0;
    if (hipMemGetInfo(&f, &t) != hipSuccess) return 0;
    return f;
}

// ---------------------------------------------------------------------------
// Q6_K x Q8_K matrix-vector product.
//
// ONE WAVE PER OUTPUT ROW. Each wave walks its row's blocks, accumulates in
// registers, and reduces across the wave at the end. That shape is chosen for
// RDNA4 deliberately:
//
//   * A row is 210 bytes per 256 elements, read linearly — the access is
//     coalesced within the wave without any staging through LDS.
//   * No LDS means no barriers and no occupancy penalty from shared memory,
//     which on RDNA4 is often the binding constraint rather than registers.
//   * The cross-wave reduction uses __shfl_down at wave32, which needs no
//     synchronisation at all.
//
// The CUDA idiom here would stage a tile of weights through shared memory for
// reuse across several outputs. With one activation vector there IS no reuse —
// each weight is read exactly once — so that would add cost for nothing. This
// is the first place the two designs part company.
// ---------------------------------------------------------------------------
__global__ void q6k_q8k_matvec(
    const uint8_t * __restrict__ w,      // [rows][blocks][210]
    const int8_t  * __restrict__ aq,     // [blocks][256] activation quants
    const float   * __restrict__ ad,     // [blocks] activation scales
    float         * __restrict__ out,    // [rows]
    int blocks, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;               // 0..WAVE-1

    const uint8_t *wrow = w + (size_t)row * blocks * Q6K_BLOCK_BYTES;
    float acc = 0.0f;

    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad[b];
        const int8_t  *q8  = aq + (size_t)b * QK_K;

        // 256 elements over WAVE lanes: each lane takes 8, strided so the
        // wave reads contiguous bytes together.
        int32_t iacc = 0;
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

                // The scale is per 16 elements; these four land in different
                // sub-blocks, so each needs its own.
                iacc += sc[(o + l)       / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32)  / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64)  / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96)  / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }

    // Wave reduction. No __syncthreads: a wave is lockstep on RDNA.
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, WAVE);
    }
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// PERSISTENT WEIGHTS. Measured first, then fixed: allocating and freeing the
// weight buffer on every call cost 13.25 ms for a 6.6 MiB projection, against
// 5.02 ms for the scalar CPU — the GPU was 2.6x SLOWER, and only 0.34 ms of
// that was the transfer itself. Weights live on the device for the life of the
// model, so they should be uploaded once.
// ---------------------------------------------------------------------------
void *llamd_rocm_upload(const void *host, size_t bytes) {
    void *dev = nullptr;
    if (hipMalloc(&dev, bytes) != hipSuccess) return nullptr;
    if (hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dev);
        return nullptr;
    }
    return dev;
}

void llamd_rocm_release(void *dev) {
    if (dev) (void)hipFree(dev);
}

// Matvec against weights ALREADY on the device. Only the activation (a few
// KiB) and the result cross the bus.
//
// LAUNCH SHAPE, also measured rather than assumed. The first version used one
// 32-thread workgroup per row: 4096 workgroups of a single wave each. RDNA4
// schedules by workgroup, so a one-wave workgroup leaves the rest of the SIMD
// idle and pays full launch overhead per row. ROWS_PER_BLOCK waves share a
// workgroup instead, which is the difference between 4096 launches worth of
// scheduling and 512.
#define ROWS_PER_BLOCK 8
#define FWD_WAVES      8    // waves per workgroup (256 threads / wave32)

// The Q6_K x Q8_K row dot product, as ONE definition.
//
// This arithmetic was copied into four kernels as they were written, which is
// four places for a divergence that would flip a near-tied expert and change
// the emitted token. Factored here; `grouped_matches_the_ungrouped_kernel_exactly`
// asserts bit-identity across kernels, so a mistake in this refactor fails a
// test rather than producing subtly different text.
//
// Returns this LANE's partial sum. The caller does the shuffle reduction,
// because the reduction is where the kernels legitimately differ in shape.
__device__ __forceinline__ float q6k_lane_partial(
    const uint8_t * __restrict__ wrow,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    int blocks, int lane)
{
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad[b];
        const int8_t  *q8  = aq + (size_t)b * QK_K;

        int32_t iacc = 0;
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
                iacc += sc[(o + l)      / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32) / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64) / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96) / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }
    return acc;
}

// The same dot product over a STRIDED subset of a row's blocks.
//
// For split-k: wave `sub` of `nsub` takes blocks sub, sub+nsub, sub+2*nsub...
// Identical arithmetic per block; only which blocks a wave owns differs.
__device__ __forceinline__ float q6k_lane_partial_strided(
    const uint8_t * __restrict__ wrow,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    int blocks, int lane, int sub, int nsub)
{
    float acc = 0.0f;
    for (int b = sub; b < blocks; b += nsub) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad[b];
        const int8_t  *q8  = aq + (size_t)b * QK_K;

        int32_t iacc = 0;
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
                iacc += sc[(o + l)      / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32) / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64) / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96) / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }
    return acc;
}

__device__ __forceinline__ float wave_reduce(float acc) {
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) acc += __shfl_down(acc, off, WAVE);
    return acc;
}

// THREE MATRICES, ONE LAUNCH — the Q, K and V projections.
//
// They all consume the same activation, and separately they are 512 + 64 + 64
// workgroups issued as three launches. Together they are 640 workgroups in one,
// which both removes two launches per layer and lets the small K and V
// projections fill the card alongside Q instead of running alone at 64
// workgroups on 32 WGPs.
__global__ void q6k_q8k_matvec_x3(
    const uint8_t * __restrict__ w0, const uint8_t * __restrict__ w1,
    const uint8_t * __restrict__ w2,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2,
    int blocks, int rows0, int rows1, int rows2)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (grow >= rows0 + rows1 + rows2) return;

    const uint8_t *w;
    float *out;
    int row;
    if (grow < rows0)                { w = w0; out = o0; row = grow; }
    else if (grow < rows0 + rows1)   { w = w1; out = o1; row = grow - rows0; }
    else                             { w = w2; out = o2; row = grow - rows0 - rows1; }

    const float acc = wave_reduce(
        q6k_lane_partial(w + (size_t)row * blocks * Q6K_BLOCK_BYTES, aq, ad, blocks, lane));
    if (lane == 0) out[row] = acc;
}

// TWO GROUPED MATRICES, ONE LAUNCH — the expert gate and up projections.
//
// Same reasoning: both read the same activation, and 768 workgroups each
// becomes 1536 in one launch.
__global__ void q6k_q8k_matvec_grouped_x2(
    const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb,
    const int32_t * __restrict__ expert_ids,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ oa, float * __restrict__ ob,
    int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    const int total = rows * n_used;
    if (grow >= 2 * total) return;

    const bool second = grow >= total;
    const int idx = second ? grow - total : grow;
    const uint8_t *w = second ? wb : wa;
    float *out = second ? ob : oa;

    const int e   = idx / rows;
    const int row = idx - e * rows;
    const size_t mat = (size_t)expert_ids[e] * rows * blocks * Q6K_BLOCK_BYTES;

    const float acc = wave_reduce(q6k_lane_partial(
        w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES, aq, ad, blocks, lane));
    if (lane == 0) out[idx] = acc;
}


__global__ void q6k_q8k_matvec_resident(
    const uint8_t * __restrict__ w,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    float         * __restrict__ out,
    int blocks, int rows)
{
    const int wave = threadIdx.x / WAVE;             // which row within the block
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (row >= rows) return;

    const uint8_t *wrow = w + (size_t)row * blocks * Q6K_BLOCK_BYTES;
    float acc = 0.0f;

    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad[b];
        const int8_t  *q8  = aq + (size_t)b * QK_K;

        int32_t iacc = 0;
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

                iacc += sc[(o + l)      / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32) / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64) / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96) / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }

    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, WAVE);
    }
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// DEVICE-SIDE Q8_K QUANTIZATION
//
// WHY THIS IS THE KEYSTONE. Every matvec needs its activation as Q8_K. While
// that conversion lives on the host, each of the ~1400 matmuls in a token must
// hand a buffer across the bus, so the forward pass cannot stay on the device
// no matter how fast the kernels are. Measured: 127.5 us per call through the
// host path against 8.3 us of actual kernel. The host was 15/16ths of the cost.
//
// MUST MATCH llamd_quant::quantize_q8_k BIT FOR BIT. iscale is -127/max where
// max is the SIGNED value of largest magnitude, so the peak lands on -127; the
// clamp is upper-only; rounding is round-half-to-even. A difference here shifts
// a routing decision and changes the emitted token, which is the failure this
// project exists to avoid.
//
// One workgroup per 256-element block. 256 threads, so one element each.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int q8k_nearest_int(float v) {
    // rintf() is round-half-to-even under the default rounding mode, which is
    // what f32::round_ties_even() gives on the host. roundf() is NOT: it
    // rounds halves away from zero and would disagree on exact .5 values.
    return (int)rintf(v);
}

__global__ void q8k_quantize(
    const float * __restrict__ x,
    int8_t      * __restrict__ qs,     // [nblocks][256]
    float       * __restrict__ d,      // [nblocks]
    int16_t     * __restrict__ bsums,  // [nblocks][16], may be null
    int nblocks)
{
    const int blk = blockIdx.x;
    if (blk >= nblocks) return;
    const int tid = threadIdx.x;              // 0..255
    const float *src = x + (size_t)blk * QK_K;

    // Reduce for the signed value of largest magnitude. Reducing |v| alone
    // loses the sign, and the sign is what decides iscale's direction.
    __shared__ float s_amax[QK_K];
    __shared__ float s_max[QK_K];
    const float v = src[tid];
    s_amax[tid] = fabsf(v);
    s_max[tid]  = v;
    __syncthreads();
    for (int stride = QK_K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // Strictly greater, so ties keep the earlier element — the same
            // element the host's sequential scan keeps.
            if (s_amax[tid + stride] > s_amax[tid]) {
                s_amax[tid] = s_amax[tid + stride];
                s_max[tid]  = s_max[tid + stride];
            }
        }
        __syncthreads();
    }
    const float amax = s_amax[0];
    const float mx   = s_max[0];

    int8_t *bq = qs + (size_t)blk * QK_K;
    if (amax == 0.0f) {
        bq[tid] = 0;
        if (tid == 0) d[blk] = 0.0f;
        if (bsums && tid < QK_K / 16) bsums[(size_t)blk * (QK_K / 16) + tid] = 0;
        return;
    }

    const float iscale = -127.0f / mx;
    int q = q8k_nearest_int(iscale * v);
    if (q > 127) q = 127;                  // upper clamp only, as on the host
    bq[tid] = (int8_t)q;
    if (tid == 0) d[blk] = 1.0f / iscale;

    if (bsums) {
        __syncthreads();
        if (tid < QK_K / 16) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += (int)bq[tid * 16 + i];
            bsums[(size_t)blk * (QK_K / 16) + tid] = (int16_t)sum;
        }
    }
}

// ---------------------------------------------------------------------------
// GROUPED EXPERT MATVEC
//
// THE MEASUREMENT THAT DEMANDED THIS. Per-shape kernel-only bandwidth on this
// card: 480 GB/s on the 243 MiB output projection (75% of peak), but 154.8 on
// expert_gate [2048->768] and 57.8 on expert_down [768->2048] — 24% and 9%.
// The expert matrices are 90% of a token's weight traffic (31.0 of 46.4 MB per
// layer), so the whole model inherits the 9-24% figure, ceilinging at ~75 t/s.
//
// WHY THEY ARE SLOW, and it is not the inner loop. expert_down has k=768, i.e.
// blocks=3, so a lane runs the block loop three times and the inner loop ONCE:
// twelve multiply-accumulates, then a five-step shuffle reduction. With 2048
// rows at 8 rows per workgroup there are only 256 workgroups — too little in
// flight to cover memory latency, and most of what does run is reduction.
//
// WHAT THIS CHANGES. One launch covers all `n_used` routed experts, so the row
// space is n_used*rows instead of rows: 8x the workgroups, 8x the bytes in
// flight, from the same 210-byte-per-block reads. The arithmetic per row is
// untouched and must stay untouched — it is the arithmetic the token agreement
// tests validate.
//
// `expert_ids` holds the routed expert indices, so a workgroup finds its
// weights at expert_ids[e] * (rows*blocks*210) within the layer's 3D tensor.
// Output is [n_used][rows], the caller scales by the routing weights.
__global__ void q6k_q8k_matvec_grouped(
    const uint8_t * __restrict__ w,        // the whole [.., .., n_expert] tensor
    const int32_t * __restrict__ expert_ids,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    float         * __restrict__ out,      // [n_used][rows]
    int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;   // global row
    const int total = rows * n_used;
    if (grow >= total) return;

    const int e   = grow / rows;            // which routed expert
    const int row = grow - e * rows;         // row within that expert
    const size_t mat = (size_t)expert_ids[e] * rows * blocks * Q6K_BLOCK_BYTES;
    const uint8_t *wrow = w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES;

    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad[b];
        const int8_t  *q8  = aq + (size_t)b * QK_K;

        int32_t iacc = 0;
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

                iacc += sc[(o + l)      / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32) / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64) / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96) / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }

    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, WAVE);
    }
    if (lane == 0) out[grow] = acc;
}

// GROUPED MATVEC WITH PER-EXPERT ACTIVATIONS.
//
// WHY A SECOND GROUPED KERNEL. The one above shares a single activation across
// every routed expert, which is exactly right for the gate and up projections:
// all 8 experts read the same normalised x. It is WRONG for the down
// projection. Each expert's down input is its own silu(gate_e) * up_e, so the
// activation is per-expert and indexing it as shared would feed expert 0's
// activation to all 8 — producing plausible, fluent, wrong output.
//
// `aq` and `ad` are therefore indexed by the routed slot e, not shared.
__global__ void q6k_q8k_matvec_grouped_pa(
    const uint8_t * __restrict__ w,
    const int32_t * __restrict__ expert_ids,
    const int8_t  * __restrict__ aq,       // [n_used][blocks*256]
    const float   * __restrict__ ad,       // [n_used][blocks]
    float         * __restrict__ out,      // [n_used][rows]
    int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (grow >= rows * n_used) return;

    const int e   = grow / rows;
    const int row = grow - e * rows;
    const size_t mat = (size_t)expert_ids[e] * rows * blocks * Q6K_BLOCK_BYTES;
    const uint8_t *wrow = w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES;
    const int8_t *aq_e = aq + (size_t)e * blocks * QK_K;
    const float  *ad_e = ad + (size_t)e * blocks;

    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t *blk = wrow + (size_t)b * Q6K_BLOCK_BYTES;
        const uint8_t *ql  = blk;
        const uint8_t *qh  = blk + QK_K / 2;
        const int8_t  *sc  = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        const uint16_t dbits = (uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16]
                             | ((uint16_t)blk[QK_K / 2 + QK_K / 4 + QK_K / 16 + 1] << 8);
        const float    d   = __half2float(__ushort_as_half(dbits)) * ad_e[b];
        const int8_t  *q8  = aq_e + (size_t)b * QK_K;

        int32_t iacc = 0;
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
                iacc += sc[(o + l)      / 16] * v0 * (int)q8[o + l];
                iacc += sc[(o + l + 32) / 16] * v1 * (int)q8[o + l + 32];
                iacc += sc[(o + l + 64) / 16] * v2 * (int)q8[o + l + 64];
                iacc += sc[(o + l + 96) / 16] * v3 * (int)q8[o + l + 96];
            }
        }
        acc += d * (float)iacc;
    }
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) acc += __shfl_down(acc, off, WAVE);
    if (lane == 0) out[grow] = acc;
}

// SPLIT-K MATVEC — several waves per output row.
//
// THE MEASUREMENT. attn_out is [4096 -> 2048]: 2048 rows at one wave each is
// 2048 waves, 65536 threads, against this card's 131072-thread capacity. Half
// the GPU is idle for the whole projection, and it measured 202 GB/s where the
// output projection — 151936 rows, so never short of waves — reaches 480.
//
// Row count is fixed by the model, so the only way to fill the card on a short
// matrix is to put more than one wave on a row. Each of `nsub` waves sums a
// strided subset of the row's blocks; the partials are combined through LDS.
//
// NUMERICS. This changes the ORDER of the float additions across blocks: the
// single-wave kernel accumulates block 0,1,2..., this one sums
// {0,nsub,2*nsub...} per wave and then adds nsub partials. Algebraically
// identical, not bit-identical, and therefore capable of flipping a near-tied
// argmax — which is why device_forward.rs asserts token equality with llama.cpp
// rather than a tolerance on logits.
__global__ void q6k_q8k_matvec_splitk(
    const uint8_t * __restrict__ w,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    float         * __restrict__ out,
    int blocks, int rows, int nsub)
{
    __shared__ float part[FWD_WAVES];
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int rows_here = FWD_WAVES / nsub;      // rows this workgroup covers
    const int row_local = wave / nsub;
    const int sub       = wave - row_local * nsub;
    const int row = blockIdx.x * rows_here + row_local;

    float acc = 0.0f;
    if (row < rows) {
        acc = q6k_lane_partial_strided(
            w + (size_t)row * blocks * Q6K_BLOCK_BYTES, aq, ad, blocks, lane, sub, nsub);
    }
    acc = wave_reduce(acc);
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (sub == 0 && lane == 0 && row < rows) {
        float t = 0.0f;
        for (int i = 0; i < nsub; ++i) t += part[row_local * nsub + i];
        out[row] = t;
    }
}

// SPLIT-K VERSIONS OF THE FUSED KERNELS.
//
// Measured per shape (kernel-only, examples/splitk.rs), best nsub and the gain
// over one wave per row:
//
//   attn_q      [2048->4096]     nsub 4   38.8 -> 25.5 us   1.52x   177 -> 269 GB/s
//   expert_gate [2048->768]x8    nsub 2   44.5 -> 31.3 us   1.42x   232 -> 330 GB/s
//   expert_down [768->2048]x8    nsub 2   41.8 -> 35.0 us   1.19x   247 -> 295 GB/s
//   attn_out    [4096->2048]     nsub 4   21.9 -> 19.7 us   1.11x   314 -> 350 GB/s
//   output      [2048->151936]   nsub 1  608.5 us best              419 GB/s
//
// The pattern is not "more waves is better": output regresses steadily with
// nsub (419 -> 221 GB/s at nsub 8) because 151936 rows already saturate and
// splitting only adds LDS reductions. nsub belongs to the shape, measured.
//
// `nsub` must divide FWD_WAVES and should not exceed `blocks`.
__global__ void q6k_q8k_matvec_x3_splitk(
    const uint8_t * __restrict__ w0, const uint8_t * __restrict__ w1,
    const uint8_t * __restrict__ w2,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2,
    int blocks, int rows0, int rows1, int rows2, int nsub)
{
    __shared__ float part[FWD_WAVES];
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int rows_here = FWD_WAVES / nsub;
    const int row_local = wave / nsub;
    const int sub       = wave - row_local * nsub;
    const int grow = blockIdx.x * rows_here + row_local;
    const int total = rows0 + rows1 + rows2;

    const uint8_t *w = nullptr;
    float *out = nullptr;
    int row = 0;
    if (grow < total) {
        if (grow < rows0)              { w = w0; out = o0; row = grow; }
        else if (grow < rows0 + rows1) { w = w1; out = o1; row = grow - rows0; }
        else                           { w = w2; out = o2; row = grow - rows0 - rows1; }
    }

    float acc = 0.0f;
    if (w) {
        acc = q6k_lane_partial_strided(
            w + (size_t)row * blocks * Q6K_BLOCK_BYTES, aq, ad, blocks, lane, sub, nsub);
    }
    acc = wave_reduce(acc);
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (sub == 0 && lane == 0 && w) {
        float t = 0.0f;
        for (int i = 0; i < nsub; ++i) t += part[row_local * nsub + i];
        out[row] = t;
    }
}

__global__ void q6k_q8k_matvec_grouped_x2_splitk(
    const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb,
    const int32_t * __restrict__ expert_ids,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ oa, float * __restrict__ ob,
    int blocks, int rows, int n_used, int nsub)
{
    __shared__ float part[FWD_WAVES];
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int rows_here = FWD_WAVES / nsub;
    const int row_local = wave / nsub;
    const int sub       = wave - row_local * nsub;
    const int grow = blockIdx.x * rows_here + row_local;
    const int total = rows * n_used;

    float acc = 0.0f;
    bool live = grow < 2 * total;
    int idx = 0;
    float *out = nullptr;
    if (live) {
        const bool second = grow >= total;
        idx = second ? grow - total : grow;
        const uint8_t *w = second ? wb : wa;
        out = second ? ob : oa;
        const int e   = idx / rows;
        const int row = idx - e * rows;
        const size_t mat = (size_t)expert_ids[e] * rows * blocks * Q6K_BLOCK_BYTES;
        acc = q6k_lane_partial_strided(
            w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES, aq, ad, blocks, lane, sub, nsub);
    }
    acc = wave_reduce(acc);
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (sub == 0 && lane == 0 && live) {
        float t = 0.0f;
        for (int i = 0; i < nsub; ++i) t += part[row_local * nsub + i];
        out[idx] = t;
    }
}

// The down projection: per-expert activations, split-k.
__global__ void q6k_q8k_matvec_grouped_pa_splitk(
    const uint8_t * __restrict__ w,
    const int32_t * __restrict__ expert_ids,
    const int8_t  * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out,
    int blocks, int rows, int n_used, int nsub)
{
    __shared__ float part[FWD_WAVES];
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int rows_here = FWD_WAVES / nsub;
    const int row_local = wave / nsub;
    const int sub       = wave - row_local * nsub;
    const int grow = blockIdx.x * rows_here + row_local;
    const int total = rows * n_used;

    float acc = 0.0f;
    const bool live = grow < total;
    if (live) {
        const int e   = grow / rows;
        const int row = grow - e * rows;
        const size_t mat = (size_t)expert_ids[e] * rows * blocks * Q6K_BLOCK_BYTES;
        acc = q6k_lane_partial_strided(
            w + mat + (size_t)row * blocks * Q6K_BLOCK_BYTES,
            aq + (size_t)e * blocks * QK_K, ad + (size_t)e * blocks,
            blocks, lane, sub, nsub);
    }
    acc = wave_reduce(acc);
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (sub == 0 && lane == 0 && live) {
        float t = 0.0f;
        for (int i = 0; i < nsub; ++i) t += part[row_local * nsub + i];
        out[grow] = t;
    }
}

// ---------------------------------------------------------------------------
// PLANAR Q6_K — the same arithmetic, read 16 bytes at a time.
//
// THE MEASUREMENT THAT DEMANDED THIS. A cold streaming read on this card
// reaches 637.8 GB/s with uint4 loads, 100% of what the 256-bit GDDR6 spec
// allows. The interleaved Q6_K matvec sustains 259 GB/s in the engine — 41%.
// The difference is load width: ggml's 210-byte blocks are 2-byte aligned
// (210 % 16 = 2), so a 16-byte vector load is impossible and every byte of
// weight is fetched by its own instruction.
//
// llamd_quant::repack_q6k_planar splits the four components into separate
// planes. Per-block strides become 128, 64, 16 and 2, all dividing 16 for the
// first three, so ql, qh and sc are 16-byte aligned EVERYWHERE. No padding:
// the repack is a permutation, asserted byte-for-byte by
// planar_repack_is_a_permutation.
//
// THE MAPPING, because it is easy to get wrong. In the interleaved kernel lane
// `l` handles ql bytes l and l+32 of each half. Here a lane owns 16 CONSECUTIVE
// ql bytes. For ql byte at position jb within a block (0..127):
//
//   h  = jb / 64          which half
//   m  = jb % 64          position within the half
//   low nibble  -> value index h*128 + m
//   high nibble -> value index h*128 + m + 64
//   qh byte      = qh[h*32 + (m % 32)]
//   shift: low = (m < 32 ? 0 : 2),  high = low + 4
//
// Both nibbles of one ql byte therefore land 64 apart, and a 16-byte group
// shares one `sc` entry for each — because 16 consecutive value indices are
// exactly one scale group. So one uint4 of ql needs one uint4 of qh, two uint4
// of activation, and two scale bytes, for 32 multiply-accumulates. The
// interleaved kernel spends 14 byte loads per 8 accumulates.
//
// NUMERICS. Identical per-term arithmetic; the float additions regroup, because
// `d` now multiplies each 32-value group rather than the whole 256-value block.
// d*(A+B) against d*A+d*B — algebraically equal, not bit-equal. Held to
// REDUCTION against the scalar path, and to token equality with llama.cpp.
__device__ __forceinline__ float q6k_planar_lane_partial(
    const uint8_t * __restrict__ ql_row,
    const uint8_t * __restrict__ qh_row,
    const int8_t  * __restrict__ sc_row,
    const uint16_t * __restrict__ d_row,
    const int8_t  * __restrict__ aq,
    const float   * __restrict__ ad,
    int blocks, int lane)
{
    float acc = 0.0f;
    const int n_u4 = blocks * 8;          // 128 bytes of ql per block / 16
    for (int u = lane; u < n_u4; u += WAVE) {
        const int b   = u >> 3;
        const int jb0 = (u & 7) << 4;     // 0,16,...,112 — always inside one half
        const int h   = jb0 >> 6;
        const int m0  = jb0 & 63;         // 0, 16, 32 or 48
        const int shl = (m0 >= 32) ? 2 : 0;

        const uint4 qlv = *(const uint4 *)(ql_row + (size_t)u * 16);
        const uint4 qhv = *(const uint4 *)(qh_row + (size_t)b * 64 + h * 32 + (m0 & 31));
        const int base_lo = b * QK_K + h * 128 + m0;
        const uint4 q8lo = *(const uint4 *)(aq + base_lo);
        const uint4 q8hi = *(const uint4 *)(aq + base_lo + 64);
        // 16 consecutive value indices are exactly one scale group.
        const int sclo = sc_row[b * 16 + h * 8 + (m0 >> 4)];
        const int schi = sc_row[b * 16 + h * 8 + (m0 >> 4) + 4];
        const float d = __half2float(__ushort_as_half(d_row[b])) * ad[b];

        const uint8_t *qlb = (const uint8_t *)&qlv;
        const uint8_t *qhb = (const uint8_t *)&qhv;
        const int8_t  *q8l = (const int8_t *)&q8lo;
        const int8_t  *q8h = (const int8_t *)&q8hi;

        int alo = 0, ahi = 0;
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int hb = qhb[t];
            const int vlo = (int)(int8_t)((qlb[t] & 0x0F) | (((hb >> shl) & 3) << 4)) - 32;
            const int vhi = (int)(int8_t)((qlb[t] >> 4) | (((hb >> (shl + 4)) & 3) << 4)) - 32;
            alo += vlo * (int)q8l[t];
            ahi += vhi * (int)q8h[t];
        }
        acc += d * (float)(sclo * alo + schi * ahi);
    }
    return acc;
}

// Planar matvec. `planes` are the four offsets from PlanarQ6K::layout.
__global__ void q6k_planar_matvec(
    const uint8_t * __restrict__ base,
    size_t off_ql, size_t off_qh, size_t off_sc, size_t off_d,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out, int blocks, int rows)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (row >= rows) return;

    const size_t i = (size_t)row * blocks;
    const float acc = wave_reduce(q6k_planar_lane_partial(
        base + off_ql + i * (QK_K / 2),
        base + off_qh + i * (QK_K / 4),
        (const int8_t *)(base + off_sc + i * (QK_K / 16)),
        (const uint16_t *)(base + off_d + i * 2),
        aq, ad, blocks, lane));
    if (lane == 0) out[row] = acc;
}

// Planar, grouped over routed experts, shared activation (gate and up).
__global__ void q6k_planar_grouped(
    const uint8_t * __restrict__ base,
    size_t off_ql, size_t off_qh, size_t off_sc, size_t off_d,
    const int32_t * __restrict__ expert_ids,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out, int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (grow >= rows * n_used) return;
    const int e   = grow / rows;
    const int row = grow - e * rows;
    const size_t i = ((size_t)expert_ids[e] * rows + row) * blocks;
    const float acc = wave_reduce(q6k_planar_lane_partial(
        base + off_ql + i * (QK_K / 2),
        base + off_qh + i * (QK_K / 4),
        (const int8_t *)(base + off_sc + i * (QK_K / 16)),
        (const uint16_t *)(base + off_d + i * 2),
        aq, ad, blocks, lane));
    if (lane == 0) out[grow] = acc;
}

// Planar, grouped, PER-EXPERT activation (the down projection).
__global__ void q6k_planar_grouped_pa(
    const uint8_t * __restrict__ base,
    size_t off_ql, size_t off_qh, size_t off_sc, size_t off_d,
    const int32_t * __restrict__ expert_ids,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ out, int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (grow >= rows * n_used) return;
    const int e   = grow / rows;
    const int row = grow - e * rows;
    const size_t i = ((size_t)expert_ids[e] * rows + row) * blocks;
    const float acc = wave_reduce(q6k_planar_lane_partial(
        base + off_ql + i * (QK_K / 2),
        base + off_qh + i * (QK_K / 4),
        (const int8_t *)(base + off_sc + i * (QK_K / 16)),
        (const uint16_t *)(base + off_d + i * 2),
        aq + (size_t)e * blocks * QK_K, ad + (size_t)e * blocks,
        blocks, lane));
    if (lane == 0) out[grow] = acc;
}

// Planar x3 — Q, K and V, which share an activation.
__global__ void q6k_planar_x3(
    const uint8_t * __restrict__ b0, const uint8_t * __restrict__ b1,
    const uint8_t * __restrict__ b2,
    size_t q0, size_t h0, size_t s0, size_t d0,
    size_t q1, size_t h1, size_t s1, size_t d1,
    size_t q2, size_t h2, size_t s2, size_t d2,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ o0, float * __restrict__ o1, float * __restrict__ o2,
    int blocks, int rows0, int rows1, int rows2)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    if (grow >= rows0 + rows1 + rows2) return;

    const uint8_t *base; size_t oq, oh, os, od; float *out; int row;
    if (grow < rows0)              { base=b0; oq=q0; oh=h0; os=s0; od=d0; out=o0; row=grow; }
    else if (grow < rows0+rows1)   { base=b1; oq=q1; oh=h1; os=s1; od=d1; out=o1; row=grow-rows0; }
    else                           { base=b2; oq=q2; oh=h2; os=s2; od=d2; out=o2; row=grow-rows0-rows1; }

    const size_t i = (size_t)row * blocks;
    const float acc = wave_reduce(q6k_planar_lane_partial(
        base + oq + i * (QK_K / 2),
        base + oh + i * (QK_K / 4),
        (const int8_t *)(base + os + i * (QK_K / 16)),
        (const uint16_t *)(base + od + i * 2),
        aq, ad, blocks, lane));
    if (lane == 0) out[row] = acc;
}

// Planar, grouped, two matrices sharing an activation (expert gate and up).
__global__ void q6k_planar_grouped_x2(
    const uint8_t * __restrict__ ba, const uint8_t * __restrict__ bb,
    size_t qa, size_t ha, size_t sa, size_t da,
    size_t qb, size_t hb_, size_t sb, size_t db,
    const int32_t * __restrict__ expert_ids,
    const int8_t * __restrict__ aq, const float * __restrict__ ad,
    float * __restrict__ oa, float * __restrict__ ob,
    int blocks, int rows, int n_used)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * ROWS_PER_BLOCK + wave;
    const int total = rows * n_used;
    if (grow >= 2 * total) return;

    const bool second = grow >= total;
    const int idx = second ? grow - total : grow;
    const uint8_t *base = second ? bb : ba;
    const size_t oq = second ? qb : qa, oh = second ? hb_ : ha;
    const size_t os = second ? sb : sa, od = second ? db : da;
    float *out = second ? ob : oa;

    const int e   = idx / rows;
    const int row = idx - e * rows;
    const size_t i = ((size_t)expert_ids[e] * rows + row) * blocks;
    const float acc = wave_reduce(q6k_planar_lane_partial(
        base + oq + i * (QK_K / 2),
        base + oh + i * (QK_K / 4),
        (const int8_t *)(base + os + i * (QK_K / 16)),
        (const uint16_t *)(base + od + i * 2),
        aq, ad, blocks, lane));
    if (lane == 0) out[idx] = acc;
}

// Per-device scratch for the activation and the result. Grown on demand and
// never freed until the process ends.
//
// WHY THIS EXISTS. With per-call allocation every small projection timed at
// ~0.23 ms regardless of its size — a fixed floor, not arithmetic. The MoE
// experts in this model are 768x2048, firmly inside that floor, so eight of
// them per layer paid allocation cost instead of doing work.
static int8_t *g_aq   = nullptr;  static size_t g_aq_cap  = 0;
static float  *g_ad   = nullptr;  static size_t g_ad_cap  = 0;
static float  *g_out  = nullptr;  static size_t g_out_cap = 0;

static bool ensure(void **p, size_t *cap, size_t want) {
    if (*cap >= want) return true;
    if (*p) (void)hipFree(*p);
    *p = nullptr; *cap = 0;
    if (hipMalloc(p, want) != hipSuccess) return false;
    *cap = want;
    return true;
}

void llamd_rocm_free_scratch() {
    if (g_aq)  { (void)hipFree(g_aq);  g_aq  = nullptr; g_aq_cap  = 0; }
    if (g_ad)  { (void)hipFree(g_ad);  g_ad  = nullptr; g_ad_cap  = 0; }
    if (g_out) { (void)hipFree(g_out); g_out = nullptr; g_out_cap = 0; }
}

int llamd_rocm_matvec_q6k_resident(
    const void *d_w, const int8_t *h_aq, const float *h_ad,
    float *h_out, int blocks, int rows)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);

    int rc = -1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) goto done;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) goto done;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  goto done;

    {
    int8_t *d_aq = g_aq; float *d_ad = g_ad, *d_out = g_out;
    if (hipMemcpy(d_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) goto done;
    if (hipMemcpy(d_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) goto done;

    {
        const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        hipLaunchKernelGGL(q6k_q8k_matvec_resident,
                           dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                           (const uint8_t *)d_w, d_aq, d_ad, d_out, blocks, rows);
    }
    if (hipGetLastError() != hipSuccess) goto done;
    if (hipDeviceSynchronize() != hipSuccess) goto done;
    if (hipMemcpy(h_out, d_out, obytes, hipMemcpyDeviceToHost) != hipSuccess) goto done;
    rc = 0;
    }

done:
    return rc;   // the scratch is kept; llamd_rocm_free_scratch releases it
}

// Kernel-only timing. WHY THIS EXISTS: llamd_rocm_matvec_q6k_resident above
// copies activations in, synchronises, and copies the result out on EVERY
// call. For a 243 MiB weight matrix that overhead is noise, but for a 1.2 MiB
// expert matrix it is everything -- the benchmark reported 10.5 GB/s on a
// shape whose weights would stream in 4 us at full rate. Attributing that to
// the kernel, and then projecting end-to-end throughput from it, would draw
// exactly the wrong conclusion about what needs fixing.
//
// So: upload once, launch `iters` times back to back with no intervening
// synchronise, then synchronise once and report the elapsed device time. That
// is the cost the kernel would have inside a resident forward pass, which is
// the number the design decision actually depends on.
//
// `ms_out` receives the total for all iterations; the caller divides.
int llamd_rocm_matvec_bench(
    const void *d_w, const int8_t *h_aq, const float *h_ad,
    int blocks, int rows, int iters, float *ms_out)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);

    int rc = -1;
    hipEvent_t t0, t1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (hipMemcpy(g_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;

    if (hipEventCreate(&t0) != hipSuccess) return -1;
    if (hipEventCreate(&t1) != hipSuccess) { (void)hipEventDestroy(t0); return -1; }

    {
        const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

        // One warm launch, so the measurement excludes first-launch costs
        // (module load, code object paging) that a steady forward pass does
        // not pay per token.
        hipLaunchKernelGGL(q6k_q8k_matvec_resident,
                           dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                           (const uint8_t *)d_w, g_aq, g_ad, g_out, blocks, rows);
        if (hipDeviceSynchronize() != hipSuccess) goto cleanup;

        if (hipEventRecord(t0, 0) != hipSuccess) goto cleanup;
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(q6k_q8k_matvec_resident,
                               dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                               (const uint8_t *)d_w, g_aq, g_ad, g_out, blocks, rows);
        }
        if (hipEventRecord(t1, 0) != hipSuccess) goto cleanup;
        if (hipEventSynchronize(t1) != hipSuccess) goto cleanup;
        if (hipGetLastError() != hipSuccess) goto cleanup;
        if (hipEventElapsedTime(ms_out, t0, t1) != hipSuccess) goto cleanup;
        rc = 0;
    }

cleanup:
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return rc;
}

// ---------------------------------------------------------------------------
// Grouped expert matvec: correctness path and timing path.
//
// The correctness path exists because the grouped kernel is only worth having
// if it computes exactly what the per-expert kernel computes. It takes host
// activations, returns host results for all n_used experts, and the Rust test
// compares those against n_used separate calls to the scalar CPU dot product.
// ---------------------------------------------------------------------------
static int32_t *g_eid  = nullptr;  static size_t g_eid_cap  = 0;

void llamd_rocm_free_grouped_scratch() {
    if (g_eid) { (void)hipFree(g_eid); g_eid = nullptr; g_eid_cap = 0; }
}

int llamd_rocm_matvec_grouped(
    const void *d_w, const int32_t *h_eid, const int8_t *h_aq,
    const float *h_ad, float *h_out, int blocks, int rows, int n_used)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * n_used * sizeof(float);
    const size_t ebytes  = (size_t)n_used * sizeof(int32_t);

    int rc = -1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (!ensure((void **)&g_eid, &g_eid_cap, ebytes))  return -1;

    if (hipMemcpy(g_aq,  h_aq,  aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad,  h_ad,  adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_eid, h_eid, ebytes,  hipMemcpyHostToDevice) != hipSuccess) return -1;

    {
        const int total = rows * n_used;
        const int grid  = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        hipLaunchKernelGGL(q6k_q8k_matvec_grouped,
                           dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                           (const uint8_t *)d_w, g_eid, g_aq, g_ad, g_out,
                           blocks, rows, n_used);
    }
    if (hipGetLastError() != hipSuccess) return -1;
    if (hipDeviceSynchronize() != hipSuccess) return -1;
    if (hipMemcpy(h_out, g_out, obytes, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    rc = 0;
    return rc;
}

// Kernel-only timing for the grouped kernel, same discipline as
// llamd_rocm_matvec_bench: upload once, launch `iters` times with no
// intervening synchronise, one warm launch first, HIP events on the device.
int llamd_rocm_matvec_grouped_bench(
    const void *d_w, const int32_t *h_eid, const int8_t *h_aq,
    const float *h_ad, int blocks, int rows, int n_used,
    int iters, float *ms_out)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * n_used * sizeof(float);
    const size_t ebytes  = (size_t)n_used * sizeof(int32_t);

    int rc = -1;
    hipEvent_t t0, t1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (!ensure((void **)&g_eid, &g_eid_cap, ebytes))  return -1;
    if (hipMemcpy(g_aq,  h_aq,  aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad,  h_ad,  adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_eid, h_eid, ebytes,  hipMemcpyHostToDevice) != hipSuccess) return -1;

    if (hipEventCreate(&t0) != hipSuccess) return -1;
    if (hipEventCreate(&t1) != hipSuccess) { (void)hipEventDestroy(t0); return -1; }

    {
        const int total = rows * n_used;
        const int grid  = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

        hipLaunchKernelGGL(q6k_q8k_matvec_grouped,
                           dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                           (const uint8_t *)d_w, g_eid, g_aq, g_ad, g_out,
                           blocks, rows, n_used);
        if (hipDeviceSynchronize() != hipSuccess) goto cleanup;

        if (hipEventRecord(t0, 0) != hipSuccess) goto cleanup;
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(q6k_q8k_matvec_grouped,
                               dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, 0,
                               (const uint8_t *)d_w, g_eid, g_aq, g_ad, g_out,
                               blocks, rows, n_used);
        }
        if (hipEventRecord(t1, 0) != hipSuccess) goto cleanup;
        if (hipEventSynchronize(t1) != hipSuccess) goto cleanup;
        if (hipGetLastError() != hipSuccess) goto cleanup;
        if (hipEventElapsedTime(ms_out, t0, t1) != hipSuccess) goto cleanup;
        rc = 0;
    }

cleanup:
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return rc;
}

// Device-side Q8_K quantization, validated against the host implementation.
// Takes f32 on the host only so the test can drive it; inside a resident
// forward pass the input is already a device buffer and this kernel is
// launched directly.
int llamd_rocm_quantize_q8k(
    const float *h_x, int nblocks,
    int8_t *h_qs, float *h_d, int16_t *h_bsums)
{
    const size_t xbytes  = (size_t)nblocks * QK_K * sizeof(float);
    const size_t qbytes  = (size_t)nblocks * QK_K;
    const size_t dbytes  = (size_t)nblocks * sizeof(float);
    const size_t sbytes  = (size_t)nblocks * (QK_K / 16) * sizeof(int16_t);

    float   *d_x = nullptr; int8_t *d_qs = nullptr;
    float   *d_d = nullptr; int16_t *d_bs = nullptr;
    int rc = -1;
    if (hipMalloc((void **)&d_x,  xbytes) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_qs, qbytes) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_d,  dbytes) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_bs, sbytes) != hipSuccess) goto out;
    if (hipMemcpy(d_x, h_x, xbytes, hipMemcpyHostToDevice) != hipSuccess) goto out;

    hipLaunchKernelGGL(q8k_quantize, dim3(nblocks), dim3(QK_K), 0, 0,
                       d_x, d_qs, d_d, d_bs, nblocks);
    if (hipGetLastError() != hipSuccess) goto out;
    if (hipDeviceSynchronize() != hipSuccess) goto out;

    if (hipMemcpy(h_qs, d_qs, qbytes, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_d,  d_d,  dbytes, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (h_bsums && hipMemcpy(h_bsums, d_bs, sbytes, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    rc = 0;

out:
    if (d_x)  (void)hipFree(d_x);
    if (d_qs) (void)hipFree(d_qs);
    if (d_d)  (void)hipFree(d_d);
    if (d_bs) (void)hipFree(d_bs);
    return rc;
}

// ---------------------------------------------------------------------------
// DEVICE-POINTER ENTRY POINTS for the resident forward pass.
//
// The functions above are the host-facing API: they copy activations in,
// synchronise, and copy results out. Inside a token that is fatal — ~1400
// matmuls at the measured 0.12 ms round-trip is 167 ms, about 6 tok/s.
//
// These take device pointers and return immediately. The caller queues a whole
// token's worth and synchronises once. They live in this file because the
// kernels do, and a __global__ cannot be launched from another translation
// unit.
// ---------------------------------------------------------------------------
// The stream these launch on, owned by llamd_forward.hip. Fetched per call
// rather than cached: it is created after this translation unit is loaded, and
// a stale null would silently put these kernels on the default stream — outside
// any graph capture, which is an error rather than a slowdown.
void *llamd_fwd_get_stream();
static inline hipStream_t fwd_stream() { return (hipStream_t)llamd_fwd_get_stream(); }

void llamd_fwd_matvec(const void *d_w, const int8_t *d_aq, const float *d_ad,
                      float *d_out, int blocks, int rows)
{
    const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_q8k_matvec_resident,
                       dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, fwd_stream(),
                       (const uint8_t *)d_w, d_aq, d_ad, d_out, blocks, rows);
}

void llamd_fwd_matvec_grouped(const void *d_w, const int32_t *d_ids,
                              const int8_t *d_aq, const float *d_ad,
                              float *d_out, int blocks, int rows, int n_used)
{
    const int total = rows * n_used;
    const int grid  = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_q8k_matvec_grouped,
                       dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, fwd_stream(),
                       (const uint8_t *)d_w, d_ids, d_aq, d_ad, d_out,
                       blocks, rows, n_used);
}

void llamd_fwd_matvec_grouped_pa(const void *d_w, const int32_t *d_ids,
                                 const int8_t *d_aq, const float *d_ad,
                                 float *d_out, int blocks, int rows, int n_used)
{
    const int total = rows * n_used;
    const int grid  = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_q8k_matvec_grouped_pa,
                       dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, fwd_stream(),
                       (const uint8_t *)d_w, d_ids, d_aq, d_ad, d_out,
                       blocks, rows, n_used);
}

// Kernel-only timing for the split-k kernel, so nsub can be chosen by
// measurement instead of by reasoning about occupancy — which was wrong twice.
int llamd_rocm_splitk_bench(
    const void *d_w, const int8_t *h_aq, const float *h_ad,
    int blocks, int rows, int nsub, int iters, float *ms_out)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);
    hipEvent_t t0, t1;
    int rc = -1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (hipMemcpy(g_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipEventCreate(&t0) != hipSuccess) return -1;
    if (hipEventCreate(&t1) != hipSuccess) { (void)hipEventDestroy(t0); return -1; }
    {
        const int rows_here = FWD_WAVES / nsub;
        const int grid = (rows + rows_here - 1) / rows_here;
        hipLaunchKernelGGL(q6k_q8k_matvec_splitk, dim3(grid), dim3(WAVE * FWD_WAVES),
                           0, 0, (const uint8_t *)d_w, g_aq, g_ad, g_out,
                           blocks, rows, nsub);
        if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
        if (hipEventRecord(t0, 0) != hipSuccess) goto cleanup;
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(q6k_q8k_matvec_splitk, dim3(grid), dim3(WAVE * FWD_WAVES),
                               0, 0, (const uint8_t *)d_w, g_aq, g_ad, g_out,
                               blocks, rows, nsub);
        }
        if (hipEventRecord(t1, 0) != hipSuccess) goto cleanup;
        if (hipEventSynchronize(t1) != hipSuccess) goto cleanup;
        if (hipGetLastError() != hipSuccess) goto cleanup;
        if (hipEventElapsedTime(ms_out, t0, t1) != hipSuccess) goto cleanup;
        rc = 0;
    }
cleanup:
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return rc;
}

void llamd_fwd_matvec_splitk(const void *d_w, const int8_t *d_aq,
                             const float *d_ad, float *d_out,
                             int blocks, int rows, int nsub)
{
    const int rows_here = FWD_WAVES / nsub;
    const int grid = (rows + rows_here - 1) / rows_here;
    hipLaunchKernelGGL(q6k_q8k_matvec_splitk,
                       dim3(grid), dim3(WAVE * FWD_WAVES), 0, fwd_stream(),
                       (const uint8_t *)d_w, d_aq, d_ad, d_out, blocks, rows, nsub);
}

void llamd_fwd_matvec_x3(const void *w0, const void *w1, const void *w2,
                         const int8_t *d_aq, const float *d_ad,
                         float *o0, float *o1, float *o2,
                         int blocks, int rows0, int rows1, int rows2)
{
    const int total = rows0 + rows1 + rows2;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_q8k_matvec_x3,
                       dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, fwd_stream(),
                       (const uint8_t *)w0, (const uint8_t *)w1, (const uint8_t *)w2,
                       d_aq, d_ad, o0, o1, o2, blocks, rows0, rows1, rows2);
}

void llamd_fwd_matvec_grouped_x2(const void *wa, const void *wb,
                                 const int32_t *d_ids, const int8_t *d_aq,
                                 const float *d_ad, float *oa, float *ob,
                                 int blocks, int rows, int n_used)
{
    const int total = 2 * rows * n_used;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_q8k_matvec_grouped_x2,
                       dim3(grid), dim3(WAVE * ROWS_PER_BLOCK), 0, fwd_stream(),
                       (const uint8_t *)wa, (const uint8_t *)wb, d_ids,
                       d_aq, d_ad, oa, ob, blocks, rows, n_used);
}

void llamd_fwd_matvec_x3_splitk(const void *w0, const void *w1, const void *w2,
                                const int8_t *d_aq, const float *d_ad,
                                float *o0, float *o1, float *o2,
                                int blocks, int rows0, int rows1, int rows2,
                                int nsub)
{
    const int rows_here = FWD_WAVES / nsub;
    const int total = rows0 + rows1 + rows2;
    const int grid = (total + rows_here - 1) / rows_here;
    hipLaunchKernelGGL(q6k_q8k_matvec_x3_splitk,
                       dim3(grid), dim3(WAVE * FWD_WAVES), 0, fwd_stream(),
                       (const uint8_t *)w0, (const uint8_t *)w1, (const uint8_t *)w2,
                       d_aq, d_ad, o0, o1, o2, blocks, rows0, rows1, rows2, nsub);
}

void llamd_fwd_matvec_grouped_x2_splitk(const void *wa, const void *wb,
                                        const int32_t *d_ids, const int8_t *d_aq,
                                        const float *d_ad, float *oa, float *ob,
                                        int blocks, int rows, int n_used, int nsub)
{
    const int rows_here = FWD_WAVES / nsub;
    const int total = 2 * rows * n_used;
    const int grid = (total + rows_here - 1) / rows_here;
    hipLaunchKernelGGL(q6k_q8k_matvec_grouped_x2_splitk,
                       dim3(grid), dim3(WAVE * FWD_WAVES), 0, fwd_stream(),
                       (const uint8_t *)wa, (const uint8_t *)wb, d_ids,
                       d_aq, d_ad, oa, ob, blocks, rows, n_used, nsub);
}

void llamd_fwd_matvec_grouped_pa_splitk(const void *d_w, const int32_t *d_ids,
                                        const int8_t *d_aq, const float *d_ad,
                                        float *d_out, int blocks, int rows,
                                        int n_used, int nsub)
{
    const int rows_here = FWD_WAVES / nsub;
    const int total = rows * n_used;
    const int grid = (total + rows_here - 1) / rows_here;
    hipLaunchKernelGGL(q6k_q8k_matvec_grouped_pa_splitk,
                       dim3(grid), dim3(WAVE * FWD_WAVES), 0, fwd_stream(),
                       (const uint8_t *)d_w, d_ids, d_aq, d_ad, d_out,
                       blocks, rows, n_used, nsub);
}

// Planar entry points. `off` is [ql, qh, sc, d] from PlanarQ6K::layout.
void llamd_fwd_planar_matvec(const void *base, const size_t *off,
                             const int8_t *d_aq, const float *d_ad,
                             float *d_out, int blocks, int rows)
{
    const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_planar_matvec, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                       0, fwd_stream(), (const uint8_t *)base,
                       off[0], off[1], off[2], off[3], d_aq, d_ad, d_out, blocks, rows);
}

void llamd_fwd_planar_grouped(const void *base, const size_t *off,
                              const int32_t *d_ids, const int8_t *d_aq,
                              const float *d_ad, float *d_out,
                              int blocks, int rows, int n_used)
{
    const int total = rows * n_used;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_planar_grouped, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                       0, fwd_stream(), (const uint8_t *)base,
                       off[0], off[1], off[2], off[3], d_ids, d_aq, d_ad, d_out,
                       blocks, rows, n_used);
}

void llamd_fwd_planar_grouped_pa(const void *base, const size_t *off,
                                 const int32_t *d_ids, const int8_t *d_aq,
                                 const float *d_ad, float *d_out,
                                 int blocks, int rows, int n_used)
{
    const int total = rows * n_used;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_planar_grouped_pa, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                       0, fwd_stream(), (const uint8_t *)base,
                       off[0], off[1], off[2], off[3], d_ids, d_aq, d_ad, d_out,
                       blocks, rows, n_used);
}

void llamd_fwd_planar_x3(const void *b0, const size_t *f0,
                         const void *b1, const size_t *f1,
                         const void *b2, const size_t *f2,
                         const int8_t *d_aq, const float *d_ad,
                         float *o0, float *o1, float *o2,
                         int blocks, int rows0, int rows1, int rows2)
{
    const int total = rows0 + rows1 + rows2;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_planar_x3, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                       0, fwd_stream(),
                       (const uint8_t *)b0, (const uint8_t *)b1, (const uint8_t *)b2,
                       f0[0], f0[1], f0[2], f0[3],
                       f1[0], f1[1], f1[2], f1[3],
                       f2[0], f2[1], f2[2], f2[3],
                       d_aq, d_ad, o0, o1, o2, blocks, rows0, rows1, rows2);
}

void llamd_fwd_planar_grouped_x2(const void *ba, const size_t *fa,
                                 const void *bb, const size_t *fb,
                                 const int32_t *d_ids, const int8_t *d_aq,
                                 const float *d_ad, float *oa, float *ob,
                                 int blocks, int rows, int n_used)
{
    const int total = 2 * rows * n_used;
    const int grid = (total + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hipLaunchKernelGGL(q6k_planar_grouped_x2, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                       0, fwd_stream(),
                       (const uint8_t *)ba, (const uint8_t *)bb,
                       fa[0], fa[1], fa[2], fa[3],
                       fb[0], fb[1], fb[2], fb[3],
                       d_ids, d_aq, d_ad, oa, ob, blocks, rows, n_used);
}

// Host-facing planar matvec, for the validation test: takes host activations
// and returns host results, like llamd_rocm_matvec_q6k_resident.
int llamd_rocm_planar_matvec(const void *base, const size_t *off,
                             const int8_t *h_aq, const float *h_ad,
                             float *h_out, int blocks, int rows)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (hipMemcpy(g_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    {
        const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        hipLaunchKernelGGL(q6k_planar_matvec, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                           0, 0, (const uint8_t *)base,
                           off[0], off[1], off[2], off[3], g_aq, g_ad, g_out, blocks, rows);
    }
    if (hipGetLastError() != hipSuccess) return -1;
    if (hipDeviceSynchronize() != hipSuccess) return -1;
    if (hipMemcpy(h_out, g_out, obytes, hipMemcpyDeviceToHost) != hipSuccess) return -1;
    return 0;
}

// Kernel-only timing for the planar matvec, same discipline as the others.
int llamd_rocm_planar_bench(const void *base, const size_t *off,
                            const int8_t *h_aq, const float *h_ad,
                            int blocks, int rows, int iters, float *ms_out)
{
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);
    hipEvent_t t0, t1;
    int rc = -1;
    if (!ensure((void **)&g_aq,  &g_aq_cap,  aqbytes)) return -1;
    if (!ensure((void **)&g_ad,  &g_ad_cap,  adbytes)) return -1;
    if (!ensure((void **)&g_out, &g_out_cap, obytes))  return -1;
    if (hipMemcpy(g_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipMemcpy(g_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) return -1;
    if (hipEventCreate(&t0) != hipSuccess) return -1;
    if (hipEventCreate(&t1) != hipSuccess) { (void)hipEventDestroy(t0); return -1; }
    {
        const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        hipLaunchKernelGGL(q6k_planar_matvec, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                           0, 0, (const uint8_t *)base, off[0], off[1], off[2], off[3],
                           g_aq, g_ad, g_out, blocks, rows);
        if (hipDeviceSynchronize() != hipSuccess) goto cleanup;
        if (hipEventRecord(t0, 0) != hipSuccess) goto cleanup;
        for (int i = 0; i < iters; ++i) {
            hipLaunchKernelGGL(q6k_planar_matvec, dim3(grid), dim3(WAVE * ROWS_PER_BLOCK),
                               0, 0, (const uint8_t *)base, off[0], off[1], off[2], off[3],
                               g_aq, g_ad, g_out, blocks, rows);
        }
        if (hipEventRecord(t1, 0) != hipSuccess) goto cleanup;
        if (hipEventSynchronize(t1) != hipSuccess) goto cleanup;
        if (hipGetLastError() != hipSuccess) goto cleanup;
        if (hipEventElapsedTime(ms_out, t0, t1) != hipSuccess) goto cleanup;
        rc = 0;
    }
cleanup:
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
    return rc;
}

void llamd_fwd_quantize(const float *d_x, int8_t *d_qs, float *d_d, int nblocks)
{
    hipLaunchKernelGGL(q8k_quantize, dim3(nblocks), dim3(QK_K), 0, fwd_stream(),
                       d_x, d_qs, d_d, (int16_t *)nullptr, nblocks);
}

} // extern "C"
