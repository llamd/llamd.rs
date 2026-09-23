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

int llamd_rocm_matvec_q6k(
    const uint8_t *h_w, const int8_t *h_aq, const float *h_ad,
    float *h_out, int blocks, int rows)
{
    const size_t wbytes  = (size_t)rows * blocks * Q6K_BLOCK_BYTES;
    const size_t aqbytes = (size_t)blocks * QK_K;
    const size_t adbytes = (size_t)blocks * sizeof(float);
    const size_t obytes  = (size_t)rows * sizeof(float);

    uint8_t *d_w = nullptr; int8_t *d_aq = nullptr;
    float *d_ad = nullptr, *d_out = nullptr;
    int rc = -1;

    if (hipMalloc(&d_w,  wbytes)  != hipSuccess) goto done;
    if (hipMalloc(&d_aq, aqbytes) != hipSuccess) goto done;
    if (hipMalloc(&d_ad, adbytes) != hipSuccess) goto done;
    if (hipMalloc(&d_out, obytes) != hipSuccess) goto done;

    if (hipMemcpy(d_w,  h_w,  wbytes,  hipMemcpyHostToDevice) != hipSuccess) goto done;
    if (hipMemcpy(d_aq, h_aq, aqbytes, hipMemcpyHostToDevice) != hipSuccess) goto done;
    if (hipMemcpy(d_ad, h_ad, adbytes, hipMemcpyHostToDevice) != hipSuccess) goto done;

    hipLaunchKernelGGL(q6k_q8k_matvec, dim3(rows), dim3(WAVE), 0, 0,
                       d_w, d_aq, d_ad, d_out, blocks, rows);
    if (hipGetLastError() != hipSuccess) goto done;
    if (hipDeviceSynchronize() != hipSuccess) goto done;
    if (hipMemcpy(h_out, d_out, obytes, hipMemcpyDeviceToHost) != hipSuccess) goto done;
    rc = 0;

done:
    if (d_w)   (void)hipFree(d_w);
    if (d_aq)  (void)hipFree(d_aq);
    if (d_ad)  (void)hipFree(d_ad);
    if (d_out) (void)hipFree(d_out);
    return rc;
}

} // extern "C"
