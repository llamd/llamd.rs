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
// RI   the row's flat block index, i.e. row*blocks (or expert_row*blocks)
// AQ   activation base, in uints
// AS   activation group-sum base
// AA   activation scale base
// NS   how many sequences to accumulate into ACC
// ACC  inout float[MAX_SEQ]
//
// NS IS EXPLICIT, NOT pc.nseq, and that distinction is load-bearing. For the
// attention and output projections every sequence multiplies the same matrix, so
// NS is the batch width and one weight read serves all of them. For the expert
// projections a row belongs to exactly ONE (sequence, expert) — the sequences
// route independently — so NS is 1 and the bases select that sequence. Sharing
// one field for both meanings is how the row-space bound and the accumulation
// loop end up disagreeing.
//
// Requires pc.ql_u/qh_u/sc_u/d_u (plane offsets in uints) and pc.blocks.
#define Q6K_ACC(WB, RI, AQ, AS, AA, NS, ACC)                                   \
{                                                                              \
    uint _nu4 = pc.blocks * 8u;                                                \
    uint _qlr = pc.ql_u + (RI) * 32u;                                          \
    uint _qhr = pc.qh_u + (RI) * 16u;                                          \
    uint _scr = pc.sc_u + (RI) * 4u;                                           \
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
        int _scl = sc_at(_scr, _b * 16u + _h * 8u + (_m >> 4u));                 \
        int _sch = sc_at(_scr, _b * 16u + _h * 8u + (_m >> 4u) + 4u);            \
        float _dw = d_at(pc.d_u, (RI) + _b);                                     \
        int _cl = -128 * int(_swl) + 32 * 128 * 16;                              \
        int _ch = -128 * int(_swh) + 32 * 128 * 16;                              \
        for (uint _s = 0u; _s < uint(NS); ++_s) {                                 \
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

// One int8 group scale out of the sc plane.
int sc_at(uint base_u, uint idx) {
    uint word = w[base_u + (idx >> 2u)];
    return bitfieldExtract(int(word), int(8u * (idx & 3u)), 8);
}

// One f16 super-block scale out of the d plane.
float d_at(uint base_u, uint idx) {
    uint word = w[base_u + (idx >> 1u)];
    vec2 pair = unpackHalf2x16(word);
    return ((idx & 1u) == 0u) ? pair.x : pair.y;
}
