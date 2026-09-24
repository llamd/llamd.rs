#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer W   { uint  w[];      };
layout(std430, binding = 1) readonly buffer QB  { uint  qb[];     };
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

void main() {
    uint expert = gl_WorkGroupID.y;
    int count = gcount[expert];
    if (count <= 0) return;
    int start = gstart[expert];

    uint rbase = (gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID) * uint(ROWS);
    if (rbase >= pc.rows) return;
    // How many of this lane group's ROWS rows are real. pc.rows is 768 or 2048
    // and ROWS divides both, but a caller with a ragged row count must not write
    // past the output.
    uint live_rows = min(uint(ROWS), pc.rows - rbase);
    bool by_slot = pc.by_slot != 0u;

    uint n_u4 = pc.blocks * 8u;

    for (int base = 0; base < count; base += ACC_N) {
        int live = min(ACC_N, count - base);

        float acc[ROWS][ACC_N];
        for (int r = 0; r < ROWS; ++r) {
            for (int i = 0; i < ACC_N; ++i) acc[r][i] = 0.0;
        }

        for (uint u = gl_SubgroupInvocationID; u < n_u4; u += gl_SubgroupSize) {
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

            for (int r = 0; r < ROWS; ++r) {
                uint ri = (expert * pc.rows + rbase + uint(r)) * pc.blocks;
                uint qlr = pc.ql_u + ri * 32u;
                uint qhr = pc.qh_u + ri * 16u;
                uint scr = pc.sc_u + ri * 4u;

                uint q0 = w[qlr + u * 4u + 0u], q1 = w[qlr + u * 4u + 1u];
                uint q2 = w[qlr + u * 4u + 2u], q3 = w[qlr + u * 4u + 3u];
                uint hb = qhr + b * 16u + h * 8u + ((m0 & 31u) >> 2u);
                uint h0 = w[hb + 0u], h1 = w[hb + 1u], h2 = w[hb + 2u], h3 = w[hb + 3u];

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

                scl[r] = bitfieldExtract(int(w[scr + (si >> 2u)]), int(8u * (si & 3u)), 8);
                sch[r] = bitfieldExtract(int(w[scr + (sj >> 2u)]), int(8u * (sj & 3u)), 8);
                vec2 dp = unpackHalf2x16(w[pc.d_u + ((ri + b) >> 1u)]);
                dw[r] = (((ri + b) & 1u) == 0u) ? dp.x : dp.y;
            }

            // THE REUSE. One activation load per token, ROWS rows of dots.
            for (int i = 0; i < ACC_N; ++i) {
                if (i >= live) break;
                int p = start + base + i;
                uint a = by_slot ? uint(pslot[p]) : uint(ptoken[p]);
                uint ab = a * pc.blocks * 64u + bu;
                uint ah = ab + 16u;

                uint a0 = qb[ab + 0u], a1 = qb[ab + 1u];
                uint a2 = qb[ab + 2u], a3 = qb[ab + 3u];
                uint a4 = qb[ah + 0u], a5 = qb[ah + 1u];
                uint a6 = qb[ah + 2u], a7 = qb[ah + 3u];
                int sbl = sb[a * pc.blocks * 16u + g16];
                int sbh = sb[a * pc.blocks * 16u + g16 + 4u];
                float as = ad[a * pc.blocks + b];

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
                    acc[r][i] += dw[r] * as * float(il + ih);
                }
            }
        }

        for (int r = 0; r < ROWS; ++r) {
            if (uint(r) >= live_rows) break;
            for (int i = 0; i < ACC_N; ++i) {
                if (i >= live) break;
                float sum = subgroupAdd(acc[r][i]);
                if (subgroupElect()) {
                    outv[uint(pslot[start + base + i]) * pc.rows + rbase + uint(r)] = sum;
                }
            }
        }
    }
}
