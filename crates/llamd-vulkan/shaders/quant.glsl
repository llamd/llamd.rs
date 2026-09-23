// Q8_K activation quantization, for the two shaders that produce activations.
//
// Costs 3 KiB of shared memory, so it is a separate include: the matvec shaders
// do not need it and should not pay for it.
//
// MUST MATCH llamd_quant::quantize_q8_k: iscale = -127/max where max is the
// SIGNED value of largest magnitude, the clamp is upper-only, and rounding is
// half-to-even. The result is biased to unsigned (q + 128, exactly q XOR 0x80 for
// int8) because the packed dot product is unsigned, and each group of 16 is
// summed because that sum is the correction term the identity needs.
shared float s_amax[QK_K];
shared float s_max[QK_K];
shared int   s_q[QK_K];

// Returns the biased byte for this invocation's value and writes the block's
// scale and group sums. `t` is gl_LocalInvocationID.x and must span QK_K.
uint quant_bias_group(float v, uint t, uint gbase, out float dscale) {
    s_amax[t] = abs(v);
    s_max[t]  = v;
    barrier();
    for (uint st = QK_K / 2u; st > 0u; st >>= 1u) {
        if (t < st) {
            // Strictly greater, so a tie keeps the earlier element — the one a
            // sequential host scan keeps.
            if (s_amax[t + st] > s_amax[t]) {
                s_amax[t] = s_amax[t + st];
                s_max[t]  = s_max[t + st];
            }
        }
        barrier();
    }
    float amax = s_amax[0], mx = s_max[0];
    int q;
    if (amax == 0.0) {
        q = 0;
        dscale = 0.0;
    } else {
        float iscale = -127.0 / mx;
        // roundEven is round-half-to-even, matching f32::round_ties_even.
        q = int(roundEven(iscale * v));
        if (q > 127) q = 127;
        dscale = 1.0 / iscale;
    }
    uint bq = uint(q + 128) & 0xFFu;
    s_q[t] = int(bq);
    barrier();
    return bq;
}

