// The Q6_K weight decode and the batched dot-product accumulation.
//
// REQUIRES the including shader to declare `layout(std430, binding = 0) readonly
// buffer W { uint w[]; }` plus `qb`, `sb`, `ad`, and a `pc` with the four plane
// offsets and `blocks`. Separate from common.glsl for exactly that reason: a
// shader with no weight buffer cannot include it.
// ---------------------------------------------------------------------------
// The Q6_K x Q8_K lane accumulation, as a macro.
//
// WHY A MACRO AND NOT A FUNCTION: GLSL has no pointers to buffers, and the
// weight tensor differs per shader (one matrix, an expert tensor indexed by a
// routed id). A macro lets the one copy of this arithmetic be instantiated
// against whichever buffer a shader declares, instead of being duplicated by
// hand in four shaders — which is four places for a divergence that would flip
// a near-tied expert and change the emitted token.
//
// WB   the weight buffer, as uvec4[]. ABLATION SAID THIS IS THE KERNEL: removing the
//      activation reads saved 40% of a real matvec while removing the weight traffic
//      saved 3%, because the inner loop issues 11 activation loads per (K-group,
//      sequence) against 8 weight loads per K-group — 352 to 8 at 32 sequences. Every
//      index formed for a run of four uints here is 4-aligned, so one 16-byte load
//      replaces four and scalar reads become a component select. Measured 1.25x on
//      the kernel; see benchmarks.md §26.
// QLU/QHU/SCU/DU  that buffer's four plane offsets, in uints
// RI   the row's flat block index, i.e. row*blocks (or expert_row*blocks)
// AQ   activation base, in uints
// AS   activation group-sum base
// AA   activation scale base
// NS    the COMPILE-TIME accumulator width to loop over (ACC_N, or 1)
// NLIVE the RUNTIME number of those slots that hold a real sequence
// ACC   inout float[MAX_SEQ]
//
// NS AND NLIVE ARE SEPARATE ON PURPOSE. NS must be a constant or the loop cannot
// unroll and ACC lands in scratch — that was the whole point of specializing it.
// But a wide tile is chunked ACC_N tokens at a time, so the LAST chunk is short,
// and looping the full NS there would read activation slots past the tile. Hence
// a constant bound with a runtime break, which is the same shape the expert
// shaders already use.
//
// NS IS EXPLICIT, NOT pc.nseq, and that distinction is load-bearing. For the
// attention and output projections every sequence multiplies the same matrix, so
// NS is the batch width and one weight read serves all of them. For the expert
// projections a row belongs to exactly ONE (sequence, expert) — the sequences
// route independently — so NS is 1 and the bases select that sequence. Sharing
// one field for both meanings is how the row-space bound and the accumulation
// loop end up disagreeing.
//
// THE PLANE OFFSETS ARE PARAMETERS, not pc fields. The fused Q/K/V shader binds
// three different weight tensors, each with its own offsets; reading pc.ql_u
// would silently use Q's planes for K and V and produce fluent, wrong output.
//
// The scale and super-scale reads are inlined for the same reason: as functions
// they closed over the global `w`, which is only correct when there is one
// weight buffer.
//
// Requires pc.blocks.
// WHICH LANES SHARE A ROW. By default the whole subgroup, which is what this kernel
// has always done. A shader may define these to a CLUSTER instead, so that
// log2(CLUSTER) reduction steps over CLUSTER lanes replace 6 over 64 while the lanes
// sharing a row still read CONSECUTIVE K and stay coalesced. §27 measured that as a
// clean inverted U peaking at 16 on the expert kernel: 64 lanes pay too much
// epilogue, one lane per row loses coalescing and is 2.15x slower.
//
// It matters most where K is short enough that the loop cannot amortise the epilogue.
// Q, K and V are K = 2048, so n_u4 = 64 against a 64-lane subgroup — exactly one
// iteration per lane, the same shape §26 found pathological.
#ifndef Q6K_LANE
#define Q6K_LANE  gl_SubgroupInvocationID
#endif
#ifndef Q6K_LANES
#define Q6K_LANES gl_SubgroupSize
#endif

#define Q6K_ACC(WB, QLU, QHU, SCU, DU, RI, AQ, AS, AA, NS, NLIVE, ACC)         \
{                                                                              \
    uint _nu4 = pc.blocks * 8u;                                                \
    uint _qlr = (QLU) + (RI) * 32u;                                            \
    uint _qhr = (QHU) + (RI) * 16u;                                            \
    uint _scr = (SCU) + (RI) * 4u;                                             \
    for (uint _u = Q6K_LANE; _u < _nu4; _u += Q6K_LANES) {                     \
        uint _b = _u >> 3u;                                                    \
        uint _j = (_u & 7u) << 4u;                                             \
        uint _h = _j >> 6u;                                                    \
        uint _m = _j & 63u;                                                    \
        uint _sh = (_m >= 32u) ? 2u : 0u;                                       \
        uvec4 _qv = WB[(_qlr + _u * 4u) >> 2u];                                  \
        uint _q0 = _qv.x, _q1 = _qv.y, _q2 = _qv.z, _q3 = _qv.w;                 \
        uint _hb = _qhr + _b * 16u + _h * 8u + ((_m & 31u) >> 2u);               \
        uvec4 _hv = WB[_hb >> 2u];                                               \
        uint _h0 = _hv.x, _h1 = _hv.y, _h2 = _hv.z, _h3 = _hv.w;                 \
        /* Whole-word unpack: the packed word already holds four weights in the \
           order the dot product wants, so one mask-and-shift does all four. */ \
        uint _l0 = (_q0 & 0x0F0F0F0Fu) | (((_h0 >> _sh) & 0x03030303u) << 4u);   \
        uint _l1 = (_q1 & 0x0F0F0F0Fu) | (((_h1 >> _sh) & 0x03030303u) << 4u);   \
        uint _l2 = (_q2 & 0x0F0F0F0Fu) | (((_h2 >> _sh) & 0x03030303u) << 4u);   \
        uint _l3 = (_q3 & 0x0F0F0F0Fu) | (((_h3 >> _sh) & 0x03030303u) << 4u);   \
        uint _s4 = _sh + 4u;                                                    \
        uint _i0 = ((_q0 >> 4u) & 0x0F0F0F0Fu) | (((_h0 >> _s4) & 0x03030303u) << 4u); \
        uint _i1 = ((_q1 >> 4u) & 0x0F0F0F0Fu) | (((_h1 >> _s4) & 0x03030303u) << 4u); \
        uint _i2 = ((_q2 >> 4u) & 0x0F0F0F0Fu) | (((_h2 >> _s4) & 0x03030303u) << 4u); \
        uint _i3 = ((_q3 >> 4u) & 0x0F0F0F0Fu) | (((_h3 >> _s4) & 0x03030303u) << 4u); \
        /* S w — independent of the activation, so computed once for every       \
           sequence. This is the register tile. */                              \
        uint _swl = dotPacked4x8AccSatEXT(_l0, 0x01010101u, 0u);                \
        _swl = dotPacked4x8AccSatEXT(_l1, 0x01010101u, _swl);                   \
        _swl = dotPacked4x8AccSatEXT(_l2, 0x01010101u, _swl);                   \
        _swl = dotPacked4x8AccSatEXT(_l3, 0x01010101u, _swl);                   \
        uint _swh = dotPacked4x8AccSatEXT(_i0, 0x01010101u, 0u);                \
        _swh = dotPacked4x8AccSatEXT(_i1, 0x01010101u, _swh);                   \
        _swh = dotPacked4x8AccSatEXT(_i2, 0x01010101u, _swh);                   \
        _swh = dotPacked4x8AccSatEXT(_i3, 0x01010101u, _swh);                   \
        uint _bu = _b * 64u + _h * 32u + (_m >> 2u);                             \
        uint _g  = _b * 16u + _h * 8u + (_m >> 4u);                              \
        uint _si = _b * 16u + _h * 8u + (_m >> 4u);                              \
        int _scl = bitfieldExtract(int(WB[(_scr + (_si >> 2u)) >> 2u][(_scr + (_si >> 2u)) & 3u]),                  \
                                   int(8u * (_si & 3u)), 8);                     \
        uint _sj = _si + 4u;                                                     \
        int _sch = bitfieldExtract(int(WB[(_scr + (_sj >> 2u)) >> 2u][(_scr + (_sj >> 2u)) & 3u]),                  \
                                   int(8u * (_sj & 3u)), 8);                     \
        uint _di = (DU) + (((RI) + _b) >> 1u);                                   \
        vec2 _dp = unpackHalf2x16(WB[_di >> 2u][_di & 3u]);                                      \
        float _dw = ((((RI) + _b) & 1u) == 0u) ? _dp.x : _dp.y;                  \
        int _cl = -128 * int(_swl) + 32 * 128 * 16;                              \
        int _ch = -128 * int(_swh) + 32 * 128 * 16;                              \
        for (uint _s = 0u; _s < uint(NS); ++_s) {                                 \
            if (_s >= uint(NLIVE)) break;                                        \
            uint _ab = (AQ) + _s * pc.blocks * 64u + _bu;                        \
            uvec4 _av = qb[_ab >> 2u];                                           \
            uvec4 _bv = qb[(_ab + 16u) >> 2u];                                    \
            uint _wl = dotPacked4x8AccSatEXT(_l0, _av.x, 0u);                     \
            _wl = dotPacked4x8AccSatEXT(_l1, _av.y, _wl);                         \
            _wl = dotPacked4x8AccSatEXT(_l2, _av.z, _wl);                         \
            _wl = dotPacked4x8AccSatEXT(_l3, _av.w, _wl);                         \
            uint _wh = dotPacked4x8AccSatEXT(_i0, _bv.x, 0u);                     \
            _wh = dotPacked4x8AccSatEXT(_i1, _bv.y, _wh);                         \
            _wh = dotPacked4x8AccSatEXT(_i2, _bv.z, _wh);                         \
            _wh = dotPacked4x8AccSatEXT(_i3, _bv.w, _wh);                         \
            int _sl = sb[(AS) + _s * pc.blocks * 16u + _g];                       \
            int _sx = sb[(AS) + _s * pc.blocks * 16u + _g + 4u];                  \
            int _il = _scl * (int(_wl) + _cl - 32 * _sl);                         \
            int _ih = _sch * (int(_wh) + _ch - 32 * _sx);                         \
            ACC[_s] += _dw * ad[(AA) + _s * pc.blocks + _b] * float(_il + _ih);   \
        }                                                                        \
    }                                                                            \
}

// NOTE. The `sc_at` / `d_at` helpers that used to live here closed over a global
// `w`, which is only correct when a shader has exactly one weight buffer. The
// fused Q/K/V dispatch has three, so the macro above inlines those reads against
// its WB parameter instead. `embed.comp` is the only shader that still wants them
// as functions and declares them itself.
