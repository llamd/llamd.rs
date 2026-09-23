// Register-tiled Q6_K x Q8_K using RDNA4's unsigned 4-way int8 dot product.
//
// WHY. At batch 8 llamd reached 212.7 tok/s against llama.cpp's 290.9. The gap
// is not bandwidth — batching already amortised that — it is instruction count.
// The previous batched kernel spends 16 multiply-accumulates per 16 weights PER
// SEQUENCE, so at batch 8 the inner loop issues 128 MAC instructions per 16
// weight bytes. llama.cpp switches to matrix-matrix kernels at batch; this is
// the equivalent step.
//
// WHAT gfx1201 ACTUALLY HAS. Not sdot4: `__builtin_amdgcn_sdot4` needs target
// feature dot1-insts, which RDNA4 lacks. It has udot4 and udot8 — UNSIGNED.
// Probed, not assumed.
//
// THE IDENTITY THAT MAKES AN UNSIGNED INSTRUCTION CARRY A SIGNED DOT PRODUCT.
// Q6_K stores w in 0..63 and the kernel subtracts 32; Q8_K activations are
// signed int8. With b = a + 128 (which for int8 is exactly a XOR 0x80):
//
//   S sc*(w-32)*a = sc * [ S w*b  -  128*S w  -  32*S b  +  32*128*16 ]
//
// over a group of 16 values sharing one scale. Verified exactly against the
// reference on device before this kernel was written.
//
//   S w*b   4 udot4 per 16 values, PER SEQUENCE
//   S w     4 udot4 per 16 values, SHARED by every sequence  <- the tiling win
//   S b     precomputed per activation group, which is what Q8_K's bsums are
//
// So 16 weights against nseq sequences costs 4*nseq + 4 dot instructions where
// the previous kernel spent 16*nseq MACs. At nseq=8 that is 36 against 128.
//
// WHY PLANAR IS REQUIRED HERE. Packing four weights into a uint32 needs four
// CONSECUTIVE value indices, and consecutive indices must share a scale group.
// ggml's interleaved layout gives a lane two strided bytes per half; the planar
// repack (llamd_quant::repack_q6k_planar) gives it 16 consecutive ones, which is
// exactly one scale group. The repack was measured neutral at batch 1 in §10 and
// is a precondition for this kernel — that is what it is for.
//
// PROVENANCE: the Q6_K layout and the arithmetic being reproduced are from
// llama.cpp / ggml (MIT, (c) 2023-2026 The ggml authors).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK_K            256
#define Q6K_BLOCK_BYTES 210
#define WAVE            32
#define UWAVES          8
#define UROWS           8
#define MAX_SEQ         8

extern "C" void *llamd_fwd_get_stream();
static inline hipStream_t ustream() { return (hipStream_t)llamd_fwd_get_stream(); }

extern "C" {

// Bias the quantized activation to unsigned and sum each group of 16.
//
// b = q + 128, which for int8 is q XOR 0x80 — no arithmetic, and it is the
// form udot4 needs. The group sums are the `S b` term above; Q8_K already
// defines bsums as exactly this quantity over the signed values, so this is the
// same thing offset by 128*16.
__global__ void uk_bias_sums(
    const int8_t * __restrict__ qs, uint8_t * __restrict__ qb,
    int32_t * __restrict__ sb, int nblocks_total)
{
    const int b = blockIdx.x;
    if (b >= nblocks_total) return;
    const size_t i = (size_t)b * QK_K + threadIdx.x;
    qb[i] = (uint8_t)(qs[i] ^ (int8_t)0x80);
    __syncthreads();
    if (threadIdx.x < QK_K / 16) {
        const size_t g = (size_t)b * (QK_K / 16) + threadIdx.x;
        int acc = 0;
        for (int j = 0; j < 16; ++j) acc += (int)qb[(size_t)b * QK_K + threadIdx.x * 16 + j];
        sb[g] = acc;
    }
}

// Quantize AND bias AND sum, in one kernel.
//
// WHY. The udot path needs the activation biased to unsigned and its per-16
// group sums. Doing that as a separate pass over the quantizer's output cost a
// launch at every one of the five quantization points in a layer — 240 extra
// launches per batched pass — plus a full read and write of the activation for
// no arithmetic reason. These kernels emit `qb` and `sb` directly and never
// materialise the signed form, which the udot kernels do not read.
//
// `out` still receives the f32 normalised values, because the F32 router matvec
// consumes them.
__global__ void uk_rms_scale(
    const float * __restrict__ x, float * __restrict__ inv, int n, float eps)
{
    __shared__ float sh[QK_K];
    const float *xi = x + (size_t)blockIdx.x * n;
    float a = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) a += xi[i] * xi[i];
    sh[threadIdx.x] = a;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv[blockIdx.x] = rsqrtf(sh[0] / (float)n + eps);
}

// Shared tail: given the value this thread owns, quantize to Q8_K, bias, and
// contribute to the group sum. One workgroup per 256-element block, one thread
// per element. Must match llamd_quant::quantize_q8_k: iscale = -127/max with max
// the SIGNED value of largest magnitude, upper-only clamp, round-half-to-even.
__device__ __forceinline__ void quant_bias_sum(
    float v, size_t i, size_t gbase, uint8_t * __restrict__ qb,
    int32_t * __restrict__ sb, float * __restrict__ d, int b)
{
    __shared__ float sa[QK_K];
    __shared__ float sm[QK_K];
    __shared__ int   sq[QK_K];
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
    int q;
    if (amax == 0.0f) {
        q = 0;
        if (threadIdx.x == 0) d[b] = 0.0f;
    } else {
        const float iscale = -127.0f / mx;
        q = (int)rintf(iscale * v);
        if (q > 127) q = 127;
        if (threadIdx.x == 0) d[b] = 1.0f / iscale;
    }
    // Bias to unsigned: for int8, +128 is exactly XOR 0x80.
    const uint8_t bq = (uint8_t)(((int8_t)q) ^ (int8_t)0x80);
    qb[i] = bq;
    sq[threadIdx.x] = (int)bq;
    __syncthreads();
    if (threadIdx.x < QK_K / 16) {
        int acc = 0;
        for (int j = 0; j < 16; ++j) acc += sq[threadIdx.x * 16 + j];
        sb[gbase + threadIdx.x] = acc;
    }
}

// Per-sequence rms_norm * w, emitting f32 plus the biased/summed Q8_K form.
__global__ void uk_norm_quant(
    const float * __restrict__ x, const float * __restrict__ w,
    const float * __restrict__ inv, float * __restrict__ out,
    uint8_t * __restrict__ qb, int32_t * __restrict__ sb,
    float * __restrict__ d, int nblocks)
{
    const int s = blockIdx.x / nblocks;
    const int b = blockIdx.x - s * nblocks;
    const size_t base = (size_t)s * nblocks * QK_K;
    const size_t i = base + (size_t)b * QK_K + threadIdx.x;
    const float v = x[i] * inv[s] * w[b * QK_K + threadIdx.x];
    out[i] = v;
    quant_bias_sum(v, i,
                   (size_t)s * nblocks * (QK_K / 16) + (size_t)b * (QK_K / 16),
                   qb, sb, d + (size_t)s * nblocks, b);
}

// Quantize an activation that is already in its final f32 form.
__global__ void uk_quantize(
    const float * __restrict__ x, uint8_t * __restrict__ qb,
    int32_t * __restrict__ sb, float * __restrict__ d, int nblocks_total)
{
    const int b = blockIdx.x;
    if (b >= nblocks_total) return;
    const size_t i = (size_t)b * QK_K + threadIdx.x;
    quant_bias_sum(x[i], i, (size_t)b * (QK_K / 16), qb, sb, d, b);
}

// One row's contribution for every sequence, using udot4.
//
// `acc` receives nseq partial sums, one per sequence, for this lane's share of
// the row. The caller reduces across the wave.
__device__ __forceinline__ void q6k_udot_lane(
    const uint8_t  * __restrict__ ql_row,
    const uint8_t  * __restrict__ qh_row,
    const int8_t   * __restrict__ sc_row,
    const uint16_t * __restrict__ d_row,
    const uint8_t  * __restrict__ qb,    // [nseq][blocks*QK_K], biased
    const int32_t  * __restrict__ sb,    // [nseq][blocks*16]
    const float    * __restrict__ ad,    // [nseq][blocks]
    int blocks, int lane, int nseq, float *acc)
{
    const int n_u4 = blocks * 8;          // 128 ql bytes per block / 16
    const int gper  = blocks * (QK_K / 16);
    const int aper  = blocks * QK_K;
    for (int u = lane; u < n_u4; u += WAVE) {
        const int b   = u >> 3;
        const int jb0 = (u & 7) << 4;     // 0,16,...,112 — inside one half
        const int h   = jb0 >> 6;
        const int m0  = jb0 & 63;
        const int shl = (m0 >= 32) ? 2 : 0;

        const uint4 qlv = *(const uint4 *)(ql_row + (size_t)u * 16);
        const uint4 qhv = *(const uint4 *)(qh_row + (size_t)b * 64 + h * 32 + (m0 & 31));
        const int base_lo = b * QK_K + h * 128 + m0;
        const int glo = base_lo >> 4;
        const int ghi = (base_lo + 64) >> 4;
        const int sclo = sc_row[b * 16 + h * 8 + (m0 >> 4)];
        const int schi = sc_row[b * 16 + h * 8 + (m0 >> 4) + 4];
        const float dw = __half2float(__ushort_as_half(d_row[b]));

        // Pack 16 low and 16 high 6-bit weights into 4 + 4 uint32. Done ONCE
        // and reused by every sequence — this is the register tile.
        unsigned wlo[4], whi[4];
        const uint8_t *qlb = (const uint8_t *)&qlv;
        const uint8_t *qhb = (const uint8_t *)&qhv;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            unsigned pl = 0, ph = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = g * 4 + j;
                const int hb = qhb[t];
                const unsigned lo = (unsigned)((qlb[t] & 0x0F) | (((hb >> shl) & 3) << 4));
                const unsigned hi = (unsigned)((qlb[t] >> 4) | (((hb >> (shl + 4)) & 3) << 4));
                pl |= lo << (8 * j);
                ph |= hi << (8 * j);
            }
            wlo[g] = pl;
            whi[g] = ph;
        }

        // S w, shared by every sequence.
        unsigned swlo = 0, swhi = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            swlo = __builtin_amdgcn_udot4(wlo[g], 0x01010101u, swlo, false);
            swhi = __builtin_amdgcn_udot4(whi[g], 0x01010101u, swhi, false);
        }
        const int corr_lo = -128 * (int)swlo + 32 * 128 * 16;
        const int corr_hi = -128 * (int)swhi + 32 * 128 * 16;

        for (int s = 0; s < nseq; ++s) {
            const uint8_t *bq = qb + (size_t)s * aper;
            // 16-byte aligned: base_lo is a multiple of 16 and the buffer is a
            // device allocation.
            const unsigned *blo = (const unsigned *)(bq + base_lo);
            const unsigned *bhi = (const unsigned *)(bq + base_lo + 64);
            unsigned wblo = 0, wbhi = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                wblo = __builtin_amdgcn_udot4(wlo[g], blo[g], wblo, false);
                wbhi = __builtin_amdgcn_udot4(whi[g], bhi[g], wbhi, false);
            }
            const int32_t *sbs = sb + (size_t)s * gper;
            const int ilo = sclo * ((int)wblo + corr_lo - 32 * sbs[glo]);
            const int ihi = schi * ((int)wbhi + corr_hi - 32 * sbs[ghi]);
            acc[s] += dw * ad[(size_t)s * blocks + b] * (float)(ilo + ihi);
        }
    }
}

} // extern "C" — templates below need C++ linkage

// ROW-TILED variant: one wave computes UR rows, loading each activation group
// ONCE and consuming it UR times.
//
// THE MEASUREMENT THAT MOTIVATED IT. With one row per wave, every row re-reads
// the whole activation: for attn_q that is 4096 rows x 16 KiB at batch 8, about
// 67 MB of activation reads per layer against 6.9 MB of weight reads. Those hit
// L1/L2 rather than HBM, so they do not show up as memory bandwidth — they show
// up as load instructions, which is exactly what was left after udot4 cut the
// arithmetic.
//
// UR rows share the activation registers; the weights differ per row and are
// packed per row. Register pressure is the limit: UR*8 packed weights, UR*nseq
// accumulators, 8 activation words. UR is a template parameter so the right
// value is chosen by measuring the engine, not by argument.
template <int UR>
__device__ __forceinline__ void q6k_udot_lane_tiled(
    const uint8_t  * __restrict__ base,
    const size_t   *off,
    size_t row_base_i,                   // (first row) * blocks
    int rows_valid,                      // how many of UR are in range
    const uint8_t  * __restrict__ qb,
    const int32_t  * __restrict__ sb,
    const float    * __restrict__ ad,
    int blocks, int lane, int nseq, float acc[UR][MAX_SEQ])
{
    const int n_u4  = blocks * 8;
    const int gper  = blocks * (QK_K / 16);
    const int aper  = blocks * QK_K;

    for (int u = lane; u < n_u4; u += WAVE) {
        const int b   = u >> 3;
        const int jb0 = (u & 7) << 4;
        const int h   = jb0 >> 6;
        const int m0  = jb0 & 63;
        const int shl = (m0 >= 32) ? 2 : 0;
        const int base_lo = b * QK_K + h * 128 + m0;
        const int glo = base_lo >> 4;
        const int ghi = (base_lo + 64) >> 4;

        // Pack the weights for every row in the tile.
        unsigned wlo[UR][4], whi[UR][4];
        int sclo[UR], schi[UR];
        float dw[UR];
        unsigned swlo[UR], swhi[UR];
        #pragma unroll
        for (int r = 0; r < UR; ++r) {
            if (r >= rows_valid) { dw[r] = 0.0f; sclo[r] = schi[r] = 0;
                                   swlo[r] = swhi[r] = 0;
                                   #pragma unroll
                                   for (int g = 0; g < 4; ++g) { wlo[r][g] = 0; whi[r][g] = 0; }
                                   continue; }
            const size_t i = row_base_i + (size_t)r * blocks;
            const uint8_t *ql_row = base + off[0] + i * (QK_K / 2);
            const uint8_t *qh_row = base + off[1] + i * (QK_K / 4);
            const int8_t  *sc_row = (const int8_t *)(base + off[2] + i * (QK_K / 16));
            const uint16_t *d_row = (const uint16_t *)(base + off[3] + i * 2);

            const uint4 qlv = *(const uint4 *)(ql_row + (size_t)u * 16);
            const uint4 qhv = *(const uint4 *)(qh_row + (size_t)b * 64 + h * 32 + (m0 & 31));
            const uint8_t *qlb = (const uint8_t *)&qlv;
            const uint8_t *qhb = (const uint8_t *)&qhv;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                unsigned pl = 0, ph = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int t = g * 4 + j;
                    const int hb = qhb[t];
                    pl |= (unsigned)((qlb[t] & 0x0F) | (((hb >> shl) & 3) << 4)) << (8 * j);
                    ph |= (unsigned)((qlb[t] >> 4) | (((hb >> (shl + 4)) & 3) << 4)) << (8 * j);
                }
                wlo[r][g] = pl;
                whi[r][g] = ph;
            }
            unsigned a = 0, c = 0;
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                a = __builtin_amdgcn_udot4(wlo[r][g], 0x01010101u, a, false);
                c = __builtin_amdgcn_udot4(whi[r][g], 0x01010101u, c, false);
            }
            swlo[r] = a;
            swhi[r] = c;
            sclo[r] = sc_row[b * 16 + h * 8 + (m0 >> 4)];
            schi[r] = sc_row[b * 16 + h * 8 + (m0 >> 4) + 4];
            dw[r]   = __half2float(__ushort_as_half(d_row[b]));
        }

        // One activation load per sequence, consumed by every row in the tile.
        for (int s = 0; s < nseq; ++s) {
            const uint8_t *bq = qb + (size_t)s * aper;
            const unsigned *blo = (const unsigned *)(bq + base_lo);
            const unsigned *bhi = (const unsigned *)(bq + base_lo + 64);
            unsigned b0 = blo[0], b1 = blo[1], b2 = blo[2], b3 = blo[3];
            unsigned h0 = bhi[0], h1 = bhi[1], h2 = bhi[2], h3 = bhi[3];
            const int32_t *sbs = sb + (size_t)s * gper;
            const int Sblo = sbs[glo], Sbhi = sbs[ghi];
            const float adv = ad[(size_t)s * blocks + b];
            #pragma unroll
            for (int r = 0; r < UR; ++r) {
                if (r >= rows_valid) continue;
                unsigned wb = 0, wh = 0;
                wb = __builtin_amdgcn_udot4(wlo[r][0], b0, wb, false);
                wb = __builtin_amdgcn_udot4(wlo[r][1], b1, wb, false);
                wb = __builtin_amdgcn_udot4(wlo[r][2], b2, wb, false);
                wb = __builtin_amdgcn_udot4(wlo[r][3], b3, wb, false);
                wh = __builtin_amdgcn_udot4(whi[r][0], h0, wh, false);
                wh = __builtin_amdgcn_udot4(whi[r][1], h1, wh, false);
                wh = __builtin_amdgcn_udot4(whi[r][2], h2, wh, false);
                wh = __builtin_amdgcn_udot4(whi[r][3], h3, wh, false);
                const int ilo = sclo[r] * ((int)wb - 128 * (int)swlo[r] + 32 * 128 * 16 - 32 * Sblo);
                const int ihi = schi[r] * ((int)wh - 128 * (int)swhi[r] + 32 * 128 * 16 - 32 * Sbhi);
                acc[r][s] += dw[r] * adv * (float)(ilo + ihi);
            }
        }
    }
}

extern "C" {

__device__ __forceinline__ float uwave_reduce(float a) {
    #pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) a += __shfl_down(a, off, WAVE);
    return a;
}

#define PLANES(base, i, blocks) \
    (base) + off[0] + (size_t)(i) * (QK_K / 2), \
    (base) + off[1] + (size_t)(i) * (QK_K / 4), \
    (const int8_t *)((base) + off[2] + (size_t)(i) * (QK_K / 16)), \
    (const uint16_t *)((base) + off[3] + (size_t)(i) * 2)

__global__ void uk_matvec(
    const uint8_t * __restrict__ base, size_t o0, size_t o1, size_t o2, size_t o3,
    const uint8_t * __restrict__ qb, const int32_t * __restrict__ sb,
    const float * __restrict__ ad, float * __restrict__ out,
    int blocks, int rows, int nseq)
{
    const size_t off[4] = {o0, o1, o2, o3};
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row  = blockIdx.x * UROWS + wave;
    if (row >= rows) return;
    const size_t i = (size_t)row * blocks;

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;
    q6k_udot_lane(PLANES(base, i, blocks), qb, sb, ad, blocks, lane, nseq, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = uwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

} // extern "C"

template <int UR>
__global__ void uk_matvec_tiled(
    const uint8_t * __restrict__ base, size_t o0, size_t o1, size_t o2, size_t o3,
    const uint8_t * __restrict__ qb, const int32_t * __restrict__ sb,
    const float * __restrict__ ad, float * __restrict__ out,
    int blocks, int rows, int nseq)
{
    const size_t off[4] = {o0, o1, o2, o3};
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row0 = (blockIdx.x * UWAVES + wave) * UR;
    if (row0 >= rows) return;
    const int valid = (rows - row0) < UR ? (rows - row0) : UR;

    float acc[UR][MAX_SEQ];
    #pragma unroll
    for (int r = 0; r < UR; ++r)
        #pragma unroll
        for (int s = 0; s < MAX_SEQ; ++s) acc[r][s] = 0.0f;

    q6k_udot_lane_tiled<UR>(base, off, (size_t)row0 * blocks, valid,
                            qb, sb, ad, blocks, lane, nseq, acc);
    #pragma unroll
    for (int r = 0; r < UR; ++r) {
        for (int s = 0; s < nseq; ++s) {
            const float t = uwave_reduce(acc[r][s]);
            if (lane == 0 && r < valid) out[(size_t)s * rows + row0 + r] = t;
        }
    }
}

template __global__ void uk_matvec_tiled<2>(
    const uint8_t *, size_t, size_t, size_t, size_t, const uint8_t *,
    const int32_t *, const float *, float *, int, int, int);
template __global__ void uk_matvec_tiled<4>(
    const uint8_t *, size_t, size_t, size_t, size_t, const uint8_t *,
    const int32_t *, const float *, float *, int, int, int);

extern "C" {

__global__ void uk_x3(
    const uint8_t * __restrict__ b0, size_t a0, size_t a1, size_t a2, size_t a3,
    const uint8_t * __restrict__ b1, size_t c0, size_t c1, size_t c2, size_t c3,
    const uint8_t * __restrict__ b2, size_t e0, size_t e1, size_t e2, size_t e3,
    const uint8_t * __restrict__ qb, const int32_t * __restrict__ sb,
    const float * __restrict__ ad,
    float * __restrict__ oo0, float * __restrict__ oo1, float * __restrict__ oo2,
    int blocks, int rows0, int rows1, int rows2, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * UROWS + wave;
    if (grow >= rows0 + rows1 + rows2) return;

    const uint8_t *base; size_t off[4]; float *out; int row, rows;
    if (grow < rows0) {
        base = b0; off[0]=a0; off[1]=a1; off[2]=a2; off[3]=a3;
        out = oo0; row = grow; rows = rows0;
    } else if (grow < rows0 + rows1) {
        base = b1; off[0]=c0; off[1]=c1; off[2]=c2; off[3]=c3;
        out = oo1; row = grow - rows0; rows = rows1;
    } else {
        base = b2; off[0]=e0; off[1]=e1; off[2]=e2; off[3]=e3;
        out = oo2; row = grow - rows0 - rows1; rows = rows2;
    }
    const size_t i = (size_t)row * blocks;

    float acc[MAX_SEQ];
    #pragma unroll
    for (int s = 0; s < MAX_SEQ; ++s) acc[s] = 0.0f;
    q6k_udot_lane(PLANES(base, i, blocks), qb, sb, ad, blocks, lane, nseq, acc);
    for (int s = 0; s < nseq; ++s) {
        const float t = uwave_reduce(acc[s]);
        if (lane == 0) out[(size_t)s * rows + row] = t;
    }
}

// Expert gate and up. Each sequence routes independently, so a row belongs to
// exactly one (sequence, expert) and there is no weight sharing — but udot4
// still cuts the instruction count fourfold.
__global__ void uk_grouped_x2(
    const uint8_t * __restrict__ ba, size_t a0, size_t a1, size_t a2, size_t a3,
    const uint8_t * __restrict__ bb, size_t c0, size_t c1, size_t c2, size_t c3,
    const int32_t * __restrict__ ids,
    const uint8_t * __restrict__ qb, const int32_t * __restrict__ sb,
    const float * __restrict__ ad,
    float * __restrict__ oa, float * __restrict__ ob,
    int blocks, int rows, int n_used, int nseq)
{
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * UROWS + wave;
    const int per_seq = rows * n_used;
    const int total   = per_seq * nseq;
    if (grow >= 2 * total) return;

    const bool second = grow >= total;
    const int idx = second ? grow - total : grow;
    const uint8_t *base = second ? bb : ba;
    size_t off[4];
    if (second) { off[0]=c0; off[1]=c1; off[2]=c2; off[3]=c3; }
    else        { off[0]=a0; off[1]=a1; off[2]=a2; off[3]=a3; }
    float *out = second ? ob : oa;

    const int s   = idx / per_seq;
    const int r0  = idx - s * per_seq;
    const int e   = r0 / rows;
    const int row = r0 - e * rows;
    const size_t i = ((size_t)ids[s * n_used + e] * rows + row) * blocks;

    float acc[MAX_SEQ];
    acc[0] = 0.0f;
    q6k_udot_lane(PLANES(base, i, blocks),
                  qb + (size_t)s * blocks * QK_K,
                  sb + (size_t)s * blocks * (QK_K / 16),
                  ad + (size_t)s * blocks, blocks, lane, 1, acc);
    const float t = uwave_reduce(acc[0]);
    if (lane == 0) out[idx] = t;
}

// The down projection: activation is per (sequence, expert).
__global__ void uk_grouped_pa(
    const uint8_t * __restrict__ base, size_t o0, size_t o1, size_t o2, size_t o3,
    const int32_t * __restrict__ ids,
    const uint8_t * __restrict__ qb, const int32_t * __restrict__ sb,
    const float * __restrict__ ad, float * __restrict__ out,
    int blocks, int rows, int n_used, int nseq)
{
    const size_t off[4] = {o0, o1, o2, o3};
    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * UROWS + wave;
    const int per_seq = rows * n_used;
    if (grow >= per_seq * nseq) return;

    const int s   = grow / per_seq;
    const int r0  = grow - s * per_seq;
    const int e   = r0 / rows;
    const int row = r0 - e * rows;
    const size_t i = ((size_t)ids[s * n_used + e] * rows + row) * blocks;
    const size_t a = (size_t)s * n_used + e;

    float acc[MAX_SEQ];
    acc[0] = 0.0f;
    q6k_udot_lane(PLANES(base, i, blocks),
                  qb + a * blocks * QK_K,
                  sb + a * blocks * (QK_K / 16),
                  ad + a * blocks, blocks, lane, 1, acc);
    const float t = uwave_reduce(acc[0]);
    if (lane == 0) out[grow] = t;
}

// ---- launch surface -------------------------------------------------------
void llamd_u_norm_quant(const float *x, const float *w, float *out,
                        uint8_t *qb, int32_t *sb, float *d, float *inv,
                        int n, int nseq, float eps) {
    hipLaunchKernelGGL(uk_rms_scale, dim3(nseq), dim3(QK_K), 0, ustream(),
                       x, inv, n, eps);
    hipLaunchKernelGGL(uk_norm_quant, dim3(nseq * (n / QK_K)), dim3(QK_K), 0, ustream(),
                       x, w, inv, out, qb, sb, d, n / QK_K);
}

void llamd_u_quantize(const float *x, uint8_t *qb, int32_t *sb, float *d,
                      int nblocks_total) {
    hipLaunchKernelGGL(uk_quantize, dim3(nblocks_total), dim3(QK_K), 0, ustream(),
                       x, qb, sb, d, nblocks_total);
}

void llamd_u_bias_sums(const int8_t *qs, uint8_t *qb, int32_t *sb, int nblocks_total) {
    hipLaunchKernelGGL(uk_bias_sums, dim3(nblocks_total), dim3(QK_K), 0, ustream(),
                       qs, qb, sb, nblocks_total);
}

// Rows per wave. 1 disables tiling.
//
// MEASURED NEUTRAL, so 1 is the default. At batch 8: tile 1 gave 279.18 tok/s,
// tile 2 gave 280.37, tile 4 gave 278.65 — inside the run-to-run spread. The
// honest reading is that tiling is only wired into uk_matvec, which serves
// attn_out and the output projection; x3 and the grouped expert kernels carry
// most of the traffic and are not tiled. Extending it there is the open work,
// and the sweep in batch_decode.rs stays so the answer keeps coming from the
// engine rather than from an argument.
static int g_udot_tile = 1;
void llamd_u_set_tile(int t) { g_udot_tile = (t == 2 || t == 4) ? t : 1; }
int  llamd_u_get_tile() { return g_udot_tile; }

void llamd_u_matvec(const void *base, const size_t *off, const uint8_t *qb,
                    const int32_t *sb, const float *ad, float *out,
                    int blocks, int rows, int nseq) {
    if (g_udot_tile == 2) {
        const int per = UWAVES * 2;
        hipLaunchKernelGGL((uk_matvec_tiled<2>), dim3((rows + per - 1) / per),
                           dim3(WAVE * UWAVES), 0, ustream(), (const uint8_t *)base,
                           off[0], off[1], off[2], off[3], qb, sb, ad, out,
                           blocks, rows, nseq);
        return;
    }
    if (g_udot_tile == 4) {
        const int per = UWAVES * 4;
        hipLaunchKernelGGL((uk_matvec_tiled<4>), dim3((rows + per - 1) / per),
                           dim3(WAVE * UWAVES), 0, ustream(), (const uint8_t *)base,
                           off[0], off[1], off[2], off[3], qb, sb, ad, out,
                           blocks, rows, nseq);
        return;
    }
    hipLaunchKernelGGL(uk_matvec, dim3((rows + UROWS - 1) / UROWS), dim3(WAVE * UWAVES),
                       0, ustream(), (const uint8_t *)base,
                       off[0], off[1], off[2], off[3], qb, sb, ad, out,
                       blocks, rows, nseq);
}

void llamd_u_x3(const void *b0, const size_t *f0,
                const void *b1, const size_t *f1,
                const void *b2, const size_t *f2,
                const uint8_t *qb, const int32_t *sb, const float *ad,
                float *o0, float *o1, float *o2,
                int blocks, int r0, int r1, int r2, int nseq) {
    const int total = r0 + r1 + r2;
    hipLaunchKernelGGL(uk_x3, dim3((total + UROWS - 1) / UROWS), dim3(WAVE * UWAVES),
                       0, ustream(),
                       (const uint8_t *)b0, f0[0], f0[1], f0[2], f0[3],
                       (const uint8_t *)b1, f1[0], f1[1], f1[2], f1[3],
                       (const uint8_t *)b2, f2[0], f2[1], f2[2], f2[3],
                       qb, sb, ad, o0, o1, o2, blocks, r0, r1, r2, nseq);
}

void llamd_u_grouped_x2(const void *ba, const size_t *fa,
                        const void *bb, const size_t *fb,
                        const int32_t *ids, const uint8_t *qb, const int32_t *sb,
                        const float *ad, float *oa, float *ob,
                        int blocks, int rows, int n_used, int nseq) {
    const int total = 2 * rows * n_used * nseq;
    hipLaunchKernelGGL(uk_grouped_x2, dim3((total + UROWS - 1) / UROWS), dim3(WAVE * UWAVES),
                       0, ustream(),
                       (const uint8_t *)ba, fa[0], fa[1], fa[2], fa[3],
                       (const uint8_t *)bb, fb[0], fb[1], fb[2], fb[3],
                       ids, qb, sb, ad, oa, ob, blocks, rows, n_used, nseq);
}

void llamd_u_grouped_pa(const void *base, const size_t *off, const int32_t *ids,
                        const uint8_t *qb, const int32_t *sb, const float *ad,
                        float *out, int blocks, int rows, int n_used, int nseq) {
    const int total = rows * n_used * nseq;
    hipLaunchKernelGGL(uk_grouped_pa, dim3((total + UROWS - 1) / UROWS), dim3(WAVE * UWAVES),
                       0, ustream(), (const uint8_t *)base,
                       off[0], off[1], off[2], off[3], ids, qb, sb, ad, out,
                       blocks, rows, n_used, nseq);
}

// Host-in, host-out rms_norm + quantize, for the same reason.
int llamd_u_norm_quant_host(const float *h_x, const float *h_w, int n, int nseq,
                            float eps, float *h_out, uint8_t *h_qb,
                            int32_t *h_sb, float *h_d)
{
    const int nb = n / QK_K;
    const size_t xb = (size_t)n * nseq * sizeof(float);
    const size_t wb = (size_t)n * sizeof(float);
    const size_t qb = (size_t)n * nseq;
    const size_t sbb = (size_t)nb * (QK_K / 16) * nseq * sizeof(int32_t);
    const size_t db = (size_t)nb * nseq * sizeof(float);
    float *d_x=nullptr,*d_w=nullptr,*d_out=nullptr,*d_d=nullptr,*d_inv=nullptr;
    uint8_t *d_qb=nullptr; int32_t *d_sb=nullptr;
    int rc = -1;
    if (hipMalloc((void**)&d_x, xb) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_w, wb) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_out, xb) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_qb, qb) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_sb, sbb) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_d, db) != hipSuccess) goto out;
    if (hipMalloc((void**)&d_inv, (size_t)nseq*sizeof(float)) != hipSuccess) goto out;
    if (hipMemcpy(d_x, h_x, xb, hipMemcpyHostToDevice) != hipSuccess) goto out;
    if (hipMemcpy(d_w, h_w, wb, hipMemcpyHostToDevice) != hipSuccess) goto out;
    hipLaunchKernelGGL(uk_rms_scale, dim3(nseq), dim3(QK_K), 0, 0, d_x, d_inv, n, eps);
    hipLaunchKernelGGL(uk_norm_quant, dim3(nseq*nb), dim3(QK_K), 0, 0,
                       d_x, d_w, d_inv, d_out, d_qb, d_sb, d_d, nb);
    if (hipGetLastError() != hipSuccess) goto out;
    if (hipDeviceSynchronize() != hipSuccess) goto out;
    if (hipMemcpy(h_out, d_out, xb, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_qb, d_qb, qb, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_sb, d_sb, sbb, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_d, d_d, db, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    rc = 0;
out:
    if(d_x)(void)hipFree(d_x); if(d_w)(void)hipFree(d_w); if(d_out)(void)hipFree(d_out);
    if(d_qb)(void)hipFree(d_qb); if(d_sb)(void)hipFree(d_sb); if(d_d)(void)hipFree(d_d);
    if(d_inv)(void)hipFree(d_inv);
    return rc;
}

// Host-in, host-out quantize, to check the fused kernel against
// llamd_quant::quantize_q8_k directly rather than inferring a bug from a wrong
// token forty-eight layers downstream.
int llamd_u_quantize_host(const float *h_x, int nblocks,
                          uint8_t *h_qb, int32_t *h_sb, float *h_d)
{
    const size_t xb = (size_t)nblocks * QK_K * sizeof(float);
    const size_t qb = (size_t)nblocks * QK_K;
    const size_t sb = (size_t)nblocks * (QK_K / 16) * sizeof(int32_t);
    const size_t db = (size_t)nblocks * sizeof(float);
    float *d_x = nullptr; uint8_t *d_qb = nullptr;
    int32_t *d_sb = nullptr; float *d_d = nullptr;
    int rc = -1;
    if (hipMalloc((void **)&d_x, xb) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_qb, qb) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_sb, sb) != hipSuccess) goto out;
    if (hipMalloc((void **)&d_d, db) != hipSuccess) goto out;
    if (hipMemcpy(d_x, h_x, xb, hipMemcpyHostToDevice) != hipSuccess) goto out;
    hipLaunchKernelGGL(uk_quantize, dim3(nblocks), dim3(QK_K), 0, 0,
                       d_x, d_qb, d_sb, d_d, nblocks);
    if (hipGetLastError() != hipSuccess) goto out;
    if (hipDeviceSynchronize() != hipSuccess) goto out;
    if (hipMemcpy(h_qb, d_qb, qb, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_sb, d_sb, sb, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    if (hipMemcpy(h_d, d_d, db, hipMemcpyDeviceToHost) != hipSuccess) goto out;
    rc = 0;
out:
    if (d_x) (void)hipFree(d_x);
    if (d_qb) (void)hipFree(d_qb);
    if (d_sb) (void)hipFree(d_sb);
    if (d_d) (void)hipFree(d_d);
    return rc;
}

// Host-in, host-out, for validation against the scalar reference. Allocates,
// biases, runs, copies back. Never used on the hot path.
int llamd_u_matvec_host(const void *base, const size_t *off,
                        const int8_t *h_qs, const float *h_ad,
                        float *h_out, int blocks, int rows, int nseq)
{
    const size_t qbytes = (size_t)nseq * blocks * QK_K;
    const size_t gbytes = (size_t)nseq * blocks * (QK_K / 16) * sizeof(int32_t);
    const size_t abytes = (size_t)nseq * blocks * sizeof(float);
    const size_t obytes = (size_t)nseq * rows * sizeof(float);

    int8_t *d_qs = nullptr; uint8_t *d_qb = nullptr;
    int32_t *d_sb = nullptr; float *d_ad = nullptr; float *d_out = nullptr;
    int rc = -1;
    if (hipMalloc((void **)&d_qs, qbytes) != hipSuccess) goto done;
    if (hipMalloc((void **)&d_qb, qbytes) != hipSuccess) goto done;
    if (hipMalloc((void **)&d_sb, gbytes) != hipSuccess) goto done;
    if (hipMalloc((void **)&d_ad, abytes) != hipSuccess) goto done;
    if (hipMalloc((void **)&d_out, obytes) != hipSuccess) goto done;
    if (hipMemcpy(d_qs, h_qs, qbytes, hipMemcpyHostToDevice) != hipSuccess) goto done;
    if (hipMemcpy(d_ad, h_ad, abytes, hipMemcpyHostToDevice) != hipSuccess) goto done;

    hipLaunchKernelGGL(uk_bias_sums, dim3(nseq * blocks), dim3(QK_K), 0, 0,
                       d_qs, d_qb, d_sb, nseq * blocks);
    hipLaunchKernelGGL(uk_matvec, dim3((rows + UROWS - 1) / UROWS), dim3(WAVE * UWAVES),
                       0, 0, (const uint8_t *)base, off[0], off[1], off[2], off[3],
                       d_qb, d_sb, d_ad, d_out, blocks, rows, nseq);
    if (hipGetLastError() != hipSuccess) goto done;
    if (hipDeviceSynchronize() != hipSuccess) goto done;
    if (hipMemcpy(h_out, d_out, obytes, hipMemcpyDeviceToHost) != hipSuccess) goto done;
    rc = 0;

done:
    if (d_qs) (void)hipFree(d_qs);
    if (d_qb) (void)hipFree(d_qb);
    if (d_sb) (void)hipFree(d_sb);
    if (d_ad) (void)hipFree(d_ad);
    if (d_out) (void)hipFree(d_out);
    return rc;
}

} // extern "C"
