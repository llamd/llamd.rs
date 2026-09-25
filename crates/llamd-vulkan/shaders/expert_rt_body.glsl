#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 256) in;

// VEC4LOAD picks how the two hot buffers are read. Every index this kernel forms for
// a run of four uints is 4-aligned, so a uvec4 view is exact — one 16-byte load
// instead of four. It was measured 1.21x on the attention projections (§26) and 1.5%
// SLOWER here at the then-current ROWS 2 / ACC 16, because uvec4 temporaries widen a
// kernel §23 had already tuned to the register file. But §23's sweep predates both
// the vectorisation and §27's clustering, and all three trade through the same
// registers — so the optimum is re-measured over all three together rather than
// assumed to have stayed put.
#ifdef VEC4LOAD
layout(std430, binding = 0) readonly buffer W   { uvec4 w[];      };
layout(std430, binding = 1) readonly buffer QB  { uvec4 qb[];     };
#define WS(i) (w[(i) >> 2u][(i) & 3u])
#else
layout(std430, binding = 0) readonly buffer W   { uint  w[];      };
layout(std430, binding = 1) readonly buffer QB  { uint  qb[];     };
#define WS(i) (w[i])
#endif
layout(std430, binding = 2) readonly buffer SB  { int   sb[];     };
layout(std430, binding = 3) readonly buffer AD  { float ad[];     };
layout(std430, binding = 4) buffer          O   { float outv[];   };
layout(std430, binding = 5) readonly buffer GS  { int   gstart[]; };
layout(std430, binding = 6) readonly buffer GC  { int   gcount[]; };
layout(std430, binding = 7) readonly buffer PT  { int   ptoken[]; };
layout(std430, binding = 8) readonly buffer PS  { int   pslot[];  };

layout(push_constant) uniform Push {
    uint ql_u; uint qh_u; uint sc_u; uint d_u;
    uint blocks; uint rows; uint n_used; uint nseq;
    uint by_slot;
} pc;
#include "common.glsl"

// Tokens per chunk. Specialized, as in every matvec-shaped shader here.
layout(constant_id = 0) const int ACC_N = 16;

// ROWS comes from the including file. Three thin wrappers set it — 2, 4 and 8 —
// because it and ACC_N trade against each other through the register file and the
// best point is a measurement, not a derivation:
//
//   accumulators = ROWS * ACC_N, and that is the dominant register cost.
//
// At ROWS 4 / ACC_N 16 this kernel compiles to 252 VGPRs, which is TWO waves per
// SIMD. That is why row-tiling was only worth 3% (§20): the instructions it saved
// were paid back in lost latency hiding. A wrapper per ROWS keeps the sweep honest
// without adding a second axis to all 25 shaders' variant table.
#ifndef ROWS
#error "include expert_rt_body.glsl from a wrapper that defines ROWS"
#endif

// LANES PER OUTPUT ROW-GROUP. 0 (or absent) means the whole subgroup, which is the
// original decomposition. A smaller cluster is the middle of two failures:
//
//   64 lanes per row  the reduction is ~19% of lane-ops, because K = 2048 leaves each
//                     lane ONE loop iteration to amortise 32 subgroupAdds over (§26)
//    1 lane per row   no reduction at all, and 2.15x SLOWER — each lane streams its
//                     own row, so one load instruction touches 64 cache lines. The
//                     ablation's "weight traffic is 3%" was measured with COALESCED
//                     loads; it does not survive being reorganised.
//
// At CLUSTER lanes the reduction is log2(CLUSTER) steps over CLUSTER lanes instead of
// 6 over 64 — 2048 lane-ops per row at 16 against 12288 — while the lanes of a
// cluster still read CONSECUTIVE K and stay coalesced. Four clusters means four
// streams, not sixty-four.
#ifndef CLUSTER
#define CLUSTER 0
#endif

// ADDRHOIST — the address arithmetic, lifted out of the K loop.
//
// §30 disassembled what actually runs: scalar work (address arithmetic, loop control,
// branches and waits) is 44-49% of every kernel here, against 9-11% for the packed dot
// products this engine exists to perform. Five optimisations aimed at the 4-11% and
// returned 0-3%; the one that worked touched the scalar half.
//
// Two things in the inner loop are recomputed for every (K-group, token) pair when
// neither has to be:
//
//   1. `pslot[p]` / `ptoken[p]` is a GLOBAL LOAD, re-fetched per K-group even though
//      it depends only on the token. Hoisted into shared memory, one entry per token
//      per chunk — 64 bytes, filled once, read as a broadcast.
//   2. `a * pc.blocks * 64`, `a * pc.blocks * 16` and `a * pc.blocks` are three
//      multiplies per (K-group, token). Only `a * pc.blocks` is needed; the rest are
//      shifts of it. Hoisting that product leaves two shifts and an add.
//
// And the per-row weight plane bases — `(expert*rows + row) * blocks` and its three
// plane offsets — are loop-INVARIANT in K and were being recomputed every iteration.
//
// Shared memory means a barrier, so the row count must divide the workgroup's rows
// exactly: a lane group that returned early would desert it. The host checks.
// ADDRHOIST puts the per-token base in SHARED memory, which needs a barrier.
// ADDRHOIST_REG puts it in REGISTERS, which does not — at ACC_N uints of pressure.
// Which is better is not derivable: the shared version measured 0.981x because two
// barriers per chunk cost more than the arithmetic saved, and because the `pslot`
// load it removed was a cache hit, not a memory trip.
#if defined(ADDRHOIST) || defined(ADDRHOIST_REG) || defined(PREFETCH)
#define HOIST_ROWBASE 1
#endif

// PREFETCH — software pipelining, the one thing never tried.
//
// §31 eliminated every other explanation. Removing weight traffic saves 3%; making the
// arithmetic twelve times denser gives parity; removing 5.7% of the instructions and
// four points of scalar share gives zero; occupancy is 5-7 waves with no spills. What
// is left is LATENCY: the loop issues a load and consumes it almost at once, so each
// wave has nearly nothing in flight, and `s_waitcnt` is 8-13% of the stream.
//
// This issues the NEXT K-group's weight words before unpacking the current ones, so
// the memory latency of iteration k+1 hides under the unpack-and-dot of iteration k.
// It is what a tuned GEMM does and what this kernel has never done.
//
// The cost is registers — 8 uints per row in flight, 16 at ROWS 2 — and registers are
// the OTHER latency-hiding mechanism on this part. The two could cancel exactly. That
// is the experiment.
#ifdef ADDRHOIST
shared uint s_abase[64];
#endif

// LDSACC — the accumulator in shared memory instead of registers.
//
// §32 found that waves beat software pipelining on this part, and the accumulator is
// the largest register consumer: ROWS * ACC_N = 32 floats of 84. The apparent
// corollary is to move it to LDS and buy waves with it.
//
// IT IS REFUTED, AND NOT INSTANTIATED. LDS is not a free resource — it is the same
// occupancy currency as registers, and a worse one here. Per-lane accumulators need
// 256 * ROWS * ACC_N * 4 bytes:
//
//   ACC_N 16 -> 32 KiB, which caps a workgroup at 4 waves per SIMD
//   ACC_N 32 -> 64 KiB, exactly the hard limit
//   ACC_N 64 -> 128 KiB, over it; the pipeline cannot be created at all
//
// The 84 VGPRs it replaces already permit 6 waves per SIMD, so it trades 6 waves for 4
// before any LDS traffic is paid — and at ACC_N 64 it cannot be created at all, because
// this device's maxComputeSharedMemorySize is 65536. The code stays as the record; no
// wrapper includes it. See benchmarks.md §33.
#ifdef LDSACC
shared float s_acc[256 * ROWS * ACC_N];
#define ACCREF(r, i) s_acc[(gl_LocalInvocationID.x * uint(ROWS) + uint(r)) * uint(ACC_N) + uint(i)]
#else
#define ACCREF(r, i) acc[r][i]
#endif
#if CLUSTER > 0
#define CL_LANES uint(CLUSTER)
#else
#define CL_LANES gl_SubgroupSize
#endif

void main() {
    uint expert = gl_WorkGroupID.y;
    int count = gcount[expert];
    if (count <= 0) return;
    int start = gstart[expert];

    // Which row group this lane belongs to. With CLUSTER = 0 this is the original
    // one-group-per-subgroup mapping; with a cluster it is one group per cluster.
    uint cl_id = gl_SubgroupInvocationID / CL_LANES;      // cluster within the subgroup
    uint cl_lane = gl_SubgroupInvocationID % CL_LANES;    // lane within the cluster
    uint cl_per_sg = gl_SubgroupSize / CL_LANES;
    uint rbase = ((gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID) * cl_per_sg
                  + cl_id) * uint(ROWS);
    if (rbase >= pc.rows) return;
    // How many of this lane group's ROWS rows are real. pc.rows is 768 or 2048
    // and ROWS divides both, but a caller with a ragged row count must not write
    // past the output.
    uint live_rows = min(uint(ROWS), pc.rows - rbase);
    bool by_slot = pc.by_slot != 0u;

    uint n_u4 = pc.blocks * 8u;

#ifdef HOIST_ROWBASE
    // Loop-invariant in K: computed once per row, not once per row per K-group.
    uint h_ri[ROWS], h_qlr[ROWS], h_qhr[ROWS], h_scr[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        uint ri = (expert * pc.rows + rbase + uint(r)) * pc.blocks;
        h_ri[r] = ri;
        h_qlr[r] = pc.ql_u + ri * 32u;
        h_qhr[r] = pc.qh_u + ri * 16u;
        h_scr[r] = pc.sc_u + ri * 4u;
    }
#endif

    for (int base = 0; base < count; base += ACC_N) {
        int live = min(ACC_N, count - base);
#ifdef ADDRHOIST
        // One global fetch and one multiply per token per CHUNK, where the inner loop
        // was doing both per token per K-group.
        for (int i = int(gl_LocalInvocationID.x); i < live; i += 256) {
            int pr = start + base + i;
            uint a = pc.by_slot != 0u ? uint(pslot[pr]) : uint(ptoken[pr]);
            s_abase[i] = a * pc.blocks;
        }
        barrier();
#endif
#ifdef ADDRHOIST_REG
        // The same hoist with no barrier: ACC_N uints of register pressure instead.
        uint r_abase[ACC_N];
        for (int i = 0; i < ACC_N; ++i) {
            int pr = start + base + min(i, live - 1);
            uint a = pc.by_slot != 0u ? uint(pslot[pr]) : uint(ptoken[pr]);
            r_abase[i] = a * pc.blocks;
        }
#endif

#ifndef LDSACC
        float acc[ROWS][ACC_N];
#endif
        for (int r = 0; r < ROWS; ++r) {
            for (int i = 0; i < ACC_N; ++i) ACCREF(r, i) = 0.0;
        }

#ifdef PREFETCH
        // The prologue order, so the first iteration also arrives to a load already
        // in flight rather than starting one.
        uint nq[ROWS][4], nw[ROWS][4];
        {
            uint u0 = cl_lane;
            uint b0 = u0 >> 3u, j0 = (u0 & 7u) << 4u;
            uint h0i = j0 >> 6u, m0i = j0 & 63u;
            uint ho0 = (m0i & 31u) >> 2u;
            for (int r = 0; r < ROWS; ++r) {
                uint ql = h_qlr[r] + u0 * 4u;
                uint qh = h_qhr[r] + b0 * 16u + h0i * 8u + ho0;
                for (uint j = 0u; j < 4u; ++j) {
                    nq[r][j] = u0 < n_u4 ? w[ql + j] : 0u;
                    nw[r][j] = u0 < n_u4 ? w[qh + j] : 0u;
                }
            }
        }
#endif
        for (uint u = cl_lane; u < n_u4; u += CL_LANES) {
            uint b = u >> 3u, jb0 = (u & 7u) << 4u;
            uint h = jb0 >> 6u, m0 = jb0 & 63u;
            uint shl = (m0 >= 32u) ? 2u : 0u, s4 = shl + 4u;
            uint si = b * 16u + h * 8u + (m0 >> 4u);
            uint sj = si + 4u;
            uint bu = b * 64u + h * 32u + (m0 >> 2u);
            uint g16 = b * 16u + h * 8u + (m0 >> 4u);

            // The per-ROW weight state, unpacked ONCE and then held while every
            // token in the chunk is multiplied against it.
            uint l0[ROWS], l1[ROWS], l2[ROWS], l3[ROWS];
            uint i0[ROWS], i1[ROWS], i2[ROWS], i3[ROWS];
            int  cl[ROWS], ch[ROWS], scl[ROWS], sch[ROWS];
            float dw[ROWS];

#ifdef PREFETCH
            // Take delivery of what the previous iteration ordered, then order the next
            // BEFORE unpacking — so the load is in flight across the whole unpack.
            uint cq[ROWS][4], cw[ROWS][4];
            for (int r = 0; r < ROWS; ++r) {
                for (uint j = 0u; j < 4u; ++j) { cq[r][j] = nq[r][j]; cw[r][j] = nw[r][j]; }
            }
            uint un = u + CL_LANES;
            if (un < n_u4) {
                uint bn = un >> 3u, jn = (un & 7u) << 4u;
                uint hn = jn >> 6u, mn = jn & 63u;
                uint hon = (mn & 31u) >> 2u;
                for (int r = 0; r < ROWS; ++r) {
                    uint ql = h_qlr[r] + un * 4u;
                    uint qh = h_qhr[r] + bn * 16u + hn * 8u + hon;
                    for (uint j = 0u; j < 4u; ++j) {
                        nq[r][j] = w[ql + j];
                        nw[r][j] = w[qh + j];
                    }
                }
            }
#endif
            for (int r = 0; r < ROWS; ++r) {
#ifdef HOIST_ROWBASE
                uint ri = h_ri[r], qlr = h_qlr[r], qhr = h_qhr[r], scr = h_scr[r];
#else
                uint ri = (expert * pc.rows + rbase + uint(r)) * pc.blocks;
                uint qlr = pc.ql_u + ri * 32u;
                uint qhr = pc.qh_u + ri * 16u;
                uint scr = pc.sc_u + ri * 4u;
#endif

#ifdef PREFETCH
                uint q0 = cq[r][0], q1 = cq[r][1], q2 = cq[r][2], q3 = cq[r][3];
                uint h0 = cw[r][0], h1 = cw[r][1], h2 = cw[r][2], h3 = cw[r][3];
#else
                uint hb = qhr + b * 16u + h * 8u + ((m0 & 31u) >> 2u);
#ifdef VEC4LOAD
                uvec4 qv = w[(qlr + u * 4u) >> 2u];
                uint q0 = qv.x, q1 = qv.y, q2 = qv.z, q3 = qv.w;
                uvec4 hv = w[hb >> 2u];
                uint h0 = hv.x, h1 = hv.y, h2 = hv.z, h3 = hv.w;
#else
                uint q0 = w[qlr + u * 4u + 0u], q1 = w[qlr + u * 4u + 1u];
                uint q2 = w[qlr + u * 4u + 2u], q3 = w[qlr + u * 4u + 3u];
                uint h0 = w[hb + 0u], h1 = w[hb + 1u], h2 = w[hb + 2u], h3 = w[hb + 3u];
#endif
#endif

                l0[r] = (q0 & 0x0F0F0F0Fu) | (((h0 >> shl) & 0x03030303u) << 4u);
                l1[r] = (q1 & 0x0F0F0F0Fu) | (((h1 >> shl) & 0x03030303u) << 4u);
                l2[r] = (q2 & 0x0F0F0F0Fu) | (((h2 >> shl) & 0x03030303u) << 4u);
                l3[r] = (q3 & 0x0F0F0F0Fu) | (((h3 >> shl) & 0x03030303u) << 4u);
                i0[r] = ((q0 >> 4u) & 0x0F0F0F0Fu) | (((h0 >> s4) & 0x03030303u) << 4u);
                i1[r] = ((q1 >> 4u) & 0x0F0F0F0Fu) | (((h1 >> s4) & 0x03030303u) << 4u);
                i2[r] = ((q2 >> 4u) & 0x0F0F0F0Fu) | (((h2 >> s4) & 0x03030303u) << 4u);
                i3[r] = ((q3 >> 4u) & 0x0F0F0F0Fu) | (((h3 >> s4) & 0x03030303u) << 4u);

                // Sum of the weights, activation-independent, so it is computed
                // once per row here rather than once per (row, token).
                uint swl = dotPacked4x8AccSatEXT(l0[r], 0x01010101u, 0u);
                swl = dotPacked4x8AccSatEXT(l1[r], 0x01010101u, swl);
                swl = dotPacked4x8AccSatEXT(l2[r], 0x01010101u, swl);
                swl = dotPacked4x8AccSatEXT(l3[r], 0x01010101u, swl);
                uint swh = dotPacked4x8AccSatEXT(i0[r], 0x01010101u, 0u);
                swh = dotPacked4x8AccSatEXT(i1[r], 0x01010101u, swh);
                swh = dotPacked4x8AccSatEXT(i2[r], 0x01010101u, swh);
                swh = dotPacked4x8AccSatEXT(i3[r], 0x01010101u, swh);
                cl[r] = -128 * int(swl) + 32 * 128 * 16;
                ch[r] = -128 * int(swh) + 32 * 128 * 16;

                scl[r] = bitfieldExtract(int(WS(scr + (si >> 2u))), int(8u * (si & 3u)), 8);
                sch[r] = bitfieldExtract(int(WS(scr + (sj >> 2u))), int(8u * (sj & 3u)), 8);
                vec2 dp = unpackHalf2x16(WS(pc.d_u + ((ri + b) >> 1u)));
                dw[r] = (((ri + b) & 1u) == 0u) ? dp.x : dp.y;
            }

            // THE REUSE. One activation load per token, ROWS rows of dots.
            for (int i = 0; i < ACC_N; ++i) {
                if (i >= live) break;
#if defined(ADDRHOIST) || defined(ADDRHOIST_REG)
#ifdef ADDRHOIST
                uint abase = s_abase[i];
#else
                uint abase = r_abase[i];
#endif
                uint ab = (abase << 6u) + bu;
                uint ah = ab + 16u;
                uint sbase = (abase << 4u) + g16;
                uint adi = abase + b;
#else
                int p = start + base + i;
                uint a = by_slot ? uint(pslot[p]) : uint(ptoken[p]);
                uint ab = a * pc.blocks * 64u + bu;
                uint ah = ab + 16u;
                uint sbase = a * pc.blocks * 16u + g16;
                uint adi = a * pc.blocks + b;
#endif

#ifdef VEC4LOAD
                uvec4 av = qb[ab >> 2u];
                uvec4 bv = qb[ah >> 2u];
                uint a0 = av.x, a1 = av.y, a2 = av.z, a3 = av.w;
                uint a4 = bv.x, a5 = bv.y, a6 = bv.z, a7 = bv.w;
#else
                uint a0 = qb[ab + 0u], a1 = qb[ab + 1u];
                uint a2 = qb[ab + 2u], a3 = qb[ab + 3u];
                uint a4 = qb[ah + 0u], a5 = qb[ah + 1u];
                uint a6 = qb[ah + 2u], a7 = qb[ah + 3u];
#endif
                int sbl = sb[sbase];
                int sbh = sb[sbase + 4u];
                float as = ad[adi];

                for (int r = 0; r < ROWS; ++r) {
                    uint wl = dotPacked4x8AccSatEXT(l0[r], a0, 0u);
                    wl = dotPacked4x8AccSatEXT(l1[r], a1, wl);
                    wl = dotPacked4x8AccSatEXT(l2[r], a2, wl);
                    wl = dotPacked4x8AccSatEXT(l3[r], a3, wl);
                    uint wh = dotPacked4x8AccSatEXT(i0[r], a4, 0u);
                    wh = dotPacked4x8AccSatEXT(i1[r], a5, wh);
                    wh = dotPacked4x8AccSatEXT(i2[r], a6, wh);
                    wh = dotPacked4x8AccSatEXT(i3[r], a7, wh);
                    int il = scl[r] * (int(wl) + cl[r] - 32 * sbl);
                    int ih = sch[r] * (int(wh) + ch[r] - 32 * sbh);
                    ACCREF(r, i) += dw[r] * as * float(il + ih);
                }
            }
        }

#ifdef ADDRHOIST
        barrier();
#endif
        for (int r = 0; r < ROWS; ++r) {
            if (uint(r) >= live_rows) break;
            for (int i = 0; i < ACC_N; ++i) {
                if (i >= live) break;
#if CLUSTER > 0
                float sum = subgroupClusteredAdd(ACCREF(r, i), CLUSTER);
#else
                float sum = subgroupAdd(ACCREF(r, i));
#endif
                if (cl_lane == 0u) {
                    outv[uint(pslot[start + base + i]) * pc.rows + rbase + uint(r)] = sum;
                }
            }
        }
    }
}
