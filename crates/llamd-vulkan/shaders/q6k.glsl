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
// WB   the weight buffer (uint[])
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
#define Q6K_ACC(WB, QLU, QHU, SCU, DU, RI, AQ, AS, AA, NS, NLIVE, ACC)         \
{                                                                              \
    uint _nu4 = pc.blocks * 8u;                                                \
    uint _qlr = (QLU) + (RI) * 32u;                                            \
    uint _qhr = (QHU) + (RI) * 16u;                                            \
    uint _scr = (SCU) + (RI) * 4u;                                             \
    for (uint _u = gl_SubgroupInvocationID; _u < _nu4; _u += gl_SubgroupSize) { \
        uint _b = _u >> 3u;                                                    \
        uint _j = (_u & 7u) << 4u;                                             \
        uint _h = _j >> 6u;                                                    \
        uint _m = _j & 63u;                                                    \
        uint _sh = (_m >= 32u) ? 2u : 0u;                                       \
        uint _q0 = WB[_qlr + _u * 4u + 0u];                                     \
        uint _q1 = WB[_qlr + _u * 4u + 1u];                                     \
        uint _q2 = WB[_qlr + _u * 4u + 2u];                                     \
        uint _q3 = WB[_qlr + _u * 4u + 3u];                                     \
        uint _hb = _qhr + _b * 16u + _h * 8u + ((_m & 31u) >> 2u);               \
        uint _h0 = WB[_hb + 0u], _h1 = WB[_hb + 1u];                            \
        uint _h2 = WB[_hb + 2u], _h3 = WB[_hb + 3u];                            \
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
        int _scl = bitfieldExtract(int(WB[_scr + (_si >> 2u)]),                  \
                                   int(8u * (_si & 3u)), 8);                     \
        uint _sj = _si + 4u;                                                     \
        int _sch = bitfieldExtract(int(WB[_scr + (_sj >> 2u)]),                  \
                                   int(8u * (_sj & 3u)), 8);                     \
        uint _di = (DU) + (((RI) + _b) >> 1u);                                   \
        vec2 _dp = unpackHalf2x16(WB[_di]);                                      \
        float _dw = ((((RI) + _b) & 1u) == 0u) ? _dp.x : _dp.y;                  \
        int _cl = -128 * int(_swl) + 32 * 128 * 16;                              \
        int _ch = -128 * int(_swh) + 32 * 128 * 16;                              \
        for (uint _s = 0u; _s < uint(NS); ++_s) {                                 \
            if (_s >= uint(NLIVE)) break;                                        \
            uint _ab = (AQ) + _s * pc.blocks * 64u + _bu;                        \
            uint _wl = dotPacked4x8AccSatEXT(_l0, qb[_ab + 0u], 0u);             \
            _wl = dotPacked4x8AccSatEXT(_l1, qb[_ab + 1u], _wl);                 \
            _wl = dotPacked4x8AccSatEXT(_l2, qb[_ab + 2u], _wl);                 \
            _wl = dotPacked4x8AccSatEXT(_l3, qb[_ab + 3u], _wl);                 \
            uint _ah = _ab + 16u;                                                \
            uint _wh = dotPacked4x8AccSatEXT(_i0, qb[_ah + 0u], 0u);             \
            _wh = dotPacked4x8AccSatEXT(_i1, qb[_ah + 1u], _wh);                 \
            _wh = dotPacked4x8AccSatEXT(_i2, qb[_ah + 2u], _wh);                 \
            _wh = dotPacked4x8AccSatEXT(_i3, qb[_ah + 3u], _wh);                 \
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
