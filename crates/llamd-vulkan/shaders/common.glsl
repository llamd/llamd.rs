// Shared by every LLAMD compute shader.
//
// ONE DESCRIPTOR LAYOUT FOR ALL SHADERS: eight storage buffers, pushed inline
// with VK_KHR_push_descriptor, plus 64 bytes of push constants. Each shader
// declares only the bindings it uses and names the push fields it needs; the
// Vulkan side never has to know which shader wants what, so adding a shader is
// adding a file rather than adding host plumbing.
//
// THE ARITHMETIC MIRRORS crates/llamd-rocm/kernels/*.hip ON PURPOSE. A
// Vulkan-versus-ROCm comparison on identical hardware is only evidence if both
// compute the same thing, so where a choice affects the emitted token — SiLU on
// the gate only, QK-norm before RoPE, NeoX pairing, ties to the lower expert
// index, the 2^-14 clamp — this side makes the same one.
//
// PROVENANCE: quantization formats and the MoE gating follow llama.cpp / ggml
// (MIT, (c) 2023-2026 The ggml authors).

#define QK_K 256
// The widest tile the shaders' accumulator arrays are sized for.
//
// It bounds two different things: how many independent sequences decode together,
// and how many tokens of one prompt prefill together. The second is what makes it
// worth raising — prefill shares a weight read across the whole tile, so 32
// shares four times as much as 8.
//
// The cost is registers: `float acc[MAX_SEQ]` per invocation, indexed by a loop
// bound that is a push constant, so the compiler cannot unroll and the array may
// land in scratch. Whether raising it pays is therefore a question for the
// engine, not for arithmetic — measured in vulkan_prefill.rs.
#define MAX_SEQ 32
#define WG 256

// Q8_K quantization, shared by the two shaders that produce activations.
//
// MUST MATCH llamd_quant::quantize_q8_k: iscale = -127/max where max is the
// SIGNED value of largest magnitude, the clamp is upper-only, and rounding is
// half-to-even. The result is then biased to unsigned (q + 128, which for int8
// is q XOR 0x80) because the packed dot product is unsigned, and each group of
// 16 is summed because that sum is the correction term the identity needs.
// Split deliberately: including a helper costs shared memory and requires the
// bindings it references. `quant.glsl` brings the Q8_K quantizer and its 3 KiB of
// shared arrays; `q6k.glsl` brings the weight decode and needs a `uint w[]`
// binding. A shader includes only what it uses.
