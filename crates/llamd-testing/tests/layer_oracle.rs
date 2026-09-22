//! A whole transformer layer, compared against llama.cpp's own trace.
//!
//! The oracle is `llama-eval-callback`, which dumps every intermediate tensor
//! of a real forward pass with its shape and its full-tensor sum. LLAMD runs
//! the same token through layer 0 and the two traces are compared
//! tensor-by-tensor.
//!
//! WHAT A SUM COMPARISON PROVES, AND WHAT IT DOES NOT. eval-callback truncates
//! the values it prints but sums over the whole tensor, so the sum is exact
//! and covers every element. It will catch a wrong projection, a missing
//! QK-norm, a RoPE convention error, a wrong residual — anything structural.
//!
//! It will NOT catch a permutation: a tensor with the right elements in the
//! wrong order has the same sum. That is a real limitation and is recorded in
//! the porting matrix rather than glossed over. Shapes are compared too, which
//! narrows it, but does not close it.
//!
//!   LLAMD_TEST_MODEL=/path/to/model.gguf cargo test -p llamd-testing

use llamd_core::tolerance::{LAYER, QUANTIZED_LAYER, QUANTIZED_MATMUL, close};
use llamd_cpu::CpuBackend;
use llamd_models::Model;
use llamd_runtime::{KvCache, embed, forward_layer};
use llamd_testing::{oracle_model, skip};
use std::collections::HashMap;

/// Run eval-callback and collect `name -> (shape, sum)` for the first
/// occurrence of each tensor. Later occurrences are reshapes and views of the
/// same data, which would overwrite the value we want.
fn oracle_trace(
    model: &std::path::Path,
    prompt: &str,
) -> Option<HashMap<String, (Vec<usize>, f64)>> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?;
    let bin = repo.join("llama.cpp/build/bin/llama-eval-callback");
    if !bin.is_file() {
        skip(
            "layer_oracle",
            "llama-eval-callback not built — cmake --build llama.cpp/build --target llama-eval-callback",
        );
        return None;
    }
    let out = std::process::Command::new(&bin)
        .args(["-m", model.to_str()?, "-p", prompt, "-ngl", "0"])
        .output()
        .ok()?;
    assert!(
        out.status.success(),
        "eval-callback failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);

    let mut map: HashMap<String, (Vec<usize>, f64)> = HashMap::new();
    let mut pending: Option<(String, Vec<usize>)> = None;
    for line in text.lines() {
        if let Some(rest) = line.split("common_debug_cb_eval: ").nth(1) {
            // "<name> = (f32) OP(...) = {a, b, c, d}"
            let name = rest.split(" = ").next().unwrap_or("").trim().to_string();
            let shape = rest
                .rsplit_once('{')
                .and_then(|(_, s)| s.split_once('}'))
                .map(|(s, _)| {
                    s.split(',')
                        .filter_map(|x| x.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            pending = Some((name, shape));
        } else if let Some(v) = line.trim().strip_prefix("sum = ") {
            if let (Some((name, shape)), Ok(sum)) = (pending.take(), v.trim().parse::<f64>()) {
                map.entry(name).or_insert((shape, sum));
            }
        }
    }
    Some(map)
}

fn sum(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum()
}

/// Compare one tensor's sum, scaled by its magnitude so the tolerance means
/// the same thing for a 128-element router and a 4096-element projection.
fn cmp(
    label: &str,
    ours: &[f32],
    oracle: &HashMap<String, (Vec<usize>, f64)>,
    key: &str,
    tol: f32,
) {
    let Some((shape, want)) = oracle.get(key) else {
        panic!("oracle has no tensor named {key}");
    };
    let n: usize = shape.iter().product();
    assert_eq!(
        ours.len(),
        n,
        "{label}: {} elements vs oracle's {n} {shape:?}",
        ours.len()
    );

    let got = sum(ours);
    // Relative to the magnitude of the values, not of the sum: a sum near zero
    // from cancellation would otherwise demand impossible precision.
    let scale = ours.iter().map(|x| x.abs() as f64).sum::<f64>().max(1.0);
    let rel = ((got - want).abs() / scale) as f32;
    assert!(
        close(rel, 0.0, tol),
        "{label} ({key}): sum {got:.6} vs oracle {want:.6}, relative {rel:.3e} > {tol:.1e}"
    );
    eprintln!("  {label:<16} sum {got:>14.6}  oracle {want:>14.6}  rel {rel:.2e}");
}

#[test]
fn layer_zero_matches_llama_cpp() {
    let Some(path) = oracle_model() else {
        skip("layer_zero_matches_llama_cpp", "set LLAMD_TEST_MODEL");
        return;
    };
    // One token keeps the comparison unambiguous: no batching, no mask, and
    // every tensor in the trace belongs to this token.
    const PROMPT: &str = "Hi";

    let Some(oracle) = oracle_trace(&path, PROMPT) else {
        return;
    };
    assert!(
        oracle.contains_key("ffn_moe_weights-0"),
        "oracle trace is missing layer-0 MoE tensors; got {} tensors",
        oracle.len()
    );

    let be = CpuBackend::new();
    let model = Model::open(&path).expect("open model");
    let g = model.gguf();
    let vocab = llamd_tokenizer::Vocab::from_gguf(g).expect("vocab");
    let ids = vocab.encode(PROMPT, false);
    assert_eq!(ids.len(), 1, "expected a single-token prompt, got {ids:?}");

    let h = model.hparams.clone();
    let x = embed(&be, &model, ids[0]).expect("embed");
    cmp("embd", &x, &oracle, "embd", LAYER);

    let mut kv = KvCache::new(h.n_layer, h.kv_dim(), 8).unwrap();
    let t = forward_layer(&be, &model, 0, &x, 0, &mut kv).expect("layer 0");

    // Each name is llama.cpp's own, so a reader can find it in the trace.
    cmp("rms_norm", &t.norm, &oracle, "norm-0", LAYER);
    cmp("attn_norm", &t.attn_norm, &oracle, "attn_norm-0", LAYER);
    cmp("Q", &t.q, &oracle, "Qcur-0", QUANTIZED_MATMUL);
    cmp(
        "Q_normed",
        &t.q_normed,
        &oracle,
        "Qcur_normed-0",
        QUANTIZED_MATMUL,
    );
    cmp(
        "K_normed",
        &t.k_normed,
        &oracle,
        "Kcur_normed-0",
        QUANTIZED_MATMUL,
    );
    cmp("V", &t.v, &oracle, "Vcur-0", QUANTIZED_MATMUL);
    // llama.cpp's `kqv_out-0` is {4096} — the concatenated heads BEFORE the
    // output projection. Its projection is an unnamed node, so the projected
    // value is checked through `ffn_inp` below, which is that plus the
    // residual. Matching their name to ours here, rather than renaming our
    // field, keeps our names describing what the values are.
    cmp(
        "attn_heads",
        &t.attn_out,
        &oracle,
        "kqv_out-0",
        QUANTIZED_MATMUL,
    );
    cmp("ffn_inp", &t.ffn_inp, &oracle, "ffn_inp-0", QUANTIZED_LAYER);
    cmp(
        "ffn_norm",
        &t.ffn_norm,
        &oracle,
        "ffn_norm-0",
        QUANTIZED_LAYER,
    );

    // The router, where a mistake picks different experts entirely.
    cmp(
        "moe_logits",
        &t.moe_logits,
        &oracle,
        "ffn_moe_logits-0",
        QUANTIZED_LAYER,
    );
    cmp(
        "moe_probs",
        &t.moe_probs,
        &oracle,
        "ffn_moe_probs-0",
        QUANTIZED_LAYER,
    );

    // THE GAP FROM THE PREVIOUS COMMIT, NOW CLOSED — and in three places
    // rather than one, because llama.cpp dumps each stage of the routing.
    //
    // `ffn_moe_weights-0` is the gathered probabilities BEFORE
    // renormalisation. Comparing our normalised weights against it gave
    // 1.000000 vs 0.349709 — not a precision failure, two different
    // quantities. The top 8 experts of 128 hold ~35% of the probability mass.
    cmp(
        "moe_w_raw",
        &t.routing.weights_raw,
        &oracle,
        "ffn_moe_weights-0",
        QUANTIZED_LAYER,
    );
    cmp(
        "moe_w_sum",
        &[t.routing.weight_sum],
        &oracle,
        "ffn_moe_weights_sum_clamped-0",
        QUANTIZED_LAYER,
    );
    cmp(
        "moe_w_norm",
        &t.routing.weights,
        &oracle,
        "ffn_moe_weights_norm-0",
        QUANTIZED_LAYER,
    );

    eprintln!(
        "  routing chose experts {:?} with weights summing to {:.6}",
        t.routing.experts,
        t.routing.weights.iter().sum::<f32>()
    );
}
