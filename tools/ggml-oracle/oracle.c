// A minimal ggml oracle for LLAMD's validation harness.
//
// Computes ONE op with ggml and writes the f32 result to stdout, so LLAMD can
// be compared against the reference implementation rather than against its own
// idea of what the reference does.
//
//   oracle <op> <ne0> <ne1> [params...]   < input.f32  > output.f32
//
// Input and output are raw little-endian f32, in ggml's layout: ne0 is the
// contiguous axis. Nothing is printed to stdout but the result; diagnostics go
// to stderr, so a mistake there cannot corrupt the comparison.
//
// WHY A SEPARATE BINARY rather than FFI into the Rust tests: the oracle must
// be able to crash, leak or abort without taking the test runner with it, and
// linking ggml into every test build would put the thing under test and the
// thing judging it in the same address space.
//
// Derived from the ggml public API (MIT, (c) 2023-2026 The ggml authors).
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *m) { fprintf(stderr, "oracle: %s\n", m); exit(2); }

int main(int argc, char **argv) {
    if (argc < 4) die("usage: oracle <op> <ne0> <ne1> [params...]");
    const char *op = argv[1];
    const int64_t ne0 = atoll(argv[2]);
    const int64_t ne1 = atoll(argv[3]);
    if (ne0 <= 0 || ne1 <= 0) die("dimensions must be positive");

    // Second operand dims, for the binary ops.
    const int64_t be0 = argc > 4 ? atoll(argv[4]) : 0;
    const int64_t be1 = argc > 5 ? atoll(argv[5]) : 0;

    struct ggml_init_params ip = {
        .mem_size   = (size_t)512*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(ip);
    if (!ctx) die("ggml_init failed");

    struct ggml_tensor *a = NULL;
    struct ggml_tensor *r = NULL;
    size_t na = 0;

    // Operand `a` is created per-op, because RoPE needs a 3D tensor and the
    // input can only be read once: stdin is a pipe and cannot be rewound.
    if (strcmp(op, "rope") != 0) {
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        na = (size_t)ne0*ne1;
        if (fread(a->data, sizeof(float), na, stdin) != na) die("short read for operand a");
    }

    if (!strcmp(op, "rms_norm")) {
        float eps = argc > 4 ? atof(argv[4]) : 1e-6f;
        r = ggml_rms_norm(ctx, a, eps);
    } else if (!strcmp(op, "soft_max")) {
        r = ggml_soft_max(ctx, a);
    } else if (!strcmp(op, "silu")) {
        r = ggml_silu(ctx, a);
    } else if (!strcmp(op, "mul_mat")) {
        if (be0 <= 0 || be1 <= 0) die("mul_mat needs b dims");
        struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, be0, be1);
        size_t nb = (size_t)be0*be1;
        if (fread(b->data, sizeof(float), nb, stdin) != nb) die("short read for operand b");
        r = ggml_mul_mat(ctx, a, b);
    } else if (!strcmp(op, "rope")) {
        // rope <head_dim> <n_heads> <n_tokens> <freq_base>; positions follow.
        //
        // ggml asserts a->ne[2] == b->ne[0]: RoPE expects the tensor as
        // [head_dim, n_heads, n_tokens] and takes the token count from ne[2].
        // The memory layout is identical to a flat [head_dim*n_heads,
        // n_tokens] - head_dim varies fastest either way - but built as 2D the
        // op sees ne[2]==1 and silently claims there is one token.
        if (argc < 6) die("rope needs <head_dim> <n_heads> <n_tokens> <freq_base>");
        int64_t n_tokens = atoll(argv[4]);
        float freq_base  = atof(argv[5]);
        int    n_dims    = (int)ne0;          // rotate the whole head
        a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, n_tokens);
        na = (size_t)ne0*ne1*n_tokens;
        if (fread(a->data, sizeof(float), na, stdin) != na) die("short read for rope input");
        struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        if (fread(pos->data, sizeof(int32_t), n_tokens, stdin) != (size_t)n_tokens)
            die("short read for positions");
        // mode 2 = GGML_ROPE_TYPE_NEOX, which pairs i with i + n_dims/2.
        r = ggml_rope_ext(ctx, a, pos, NULL, n_dims, GGML_ROPE_TYPE_NEOX,
                          0, freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    } else if (!strcmp(op, "top_k")) {
        // top_k <ne0> <ne1> <k> : softmax over ne0, then ggml's top-k.
        //
        // This is the expert-selection half of build_moe_ffn. ggml documents
        // the indices as being in NO PARTICULAR ORDER, so the comparison on
        // the Rust side is set-based; asserting an order would be testing an
        // implementation detail the reference does not promise.
        //
        // Indices come back as i32 and are written as i32, so the caller must
        // read them as such rather than as float.
        if (argc < 5) die("top_k needs k");
        int k = atoi(argv[4]);
        struct ggml_tensor *p = ggml_soft_max(ctx, a);
        r = ggml_argsort_top_k(ctx, p, k);
        struct ggml_cgraph *g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, r);
        ggml_graph_compute_with_ctx(ctx, g2, 1);
        size_t n2 = ggml_nelements(r);
        if (fwrite(r->data, sizeof(int32_t), n2, stdout) != n2) die("short write");
        fflush(stdout);
        fprintf(stderr, "oracle: top_k -> %zu indices\n", n2);
        ggml_free(ctx);
        return 0;
    } else if (!strcmp(op, "get_rows")) {
        // get_rows <ne0> <ne1> <n_ids> ; ids follow the data as i32
        if (argc < 5) die("get_rows needs n_ids");
        int64_t n_ids = atoll(argv[4]);
        struct ggml_tensor *ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
        if (fread(ids->data, sizeof(int32_t), n_ids, stdin) != (size_t)n_ids)
            die("short read for ids");
        r = ggml_get_rows(ctx, a, ids);
    } else {
        die("unknown op");
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);   // one thread: deterministic order

    size_t n = ggml_nelements(r);
    if (fwrite(r->data, sizeof(float), n, stdout) != n) die("short write");
    fflush(stdout);
    fprintf(stderr, "oracle: %s -> %zu elements\n", op, n);
    ggml_free(ctx);
    return 0;
}
