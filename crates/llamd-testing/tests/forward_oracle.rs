//! The full 48-layer forward pass, compared against llama.cpp.
//!
//! This is the first test where a single wrong expert becomes visible as a
//! different token. Everything before it compared intermediates; this compares
//! the thing the model is actually for.
//!
//! SLOW ON PURPOSE. The CPU reference dequantizes every weight as it is used —
//! roughly 2.7 billion elements for one token across 48 layers — because
//! caching 23 GiB as f32 is impossible and caching selectively would be an
//! optimization in the one crate whose performance is explicitly a non-goal.
//! Run it in release, and expect tens of seconds.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing --test forward_oracle

use llamd_cpu::CpuBackend;
use llamd_models::Model;
use llamd_runtime::{KvCache, forward};
use llamd_testing::{oracle_model, skip};

/// `result_output` from llama.cpp's trace: the full logit vector's sum, and
/// the vocabulary size it covers.
fn oracle_logits(model: &std::path::Path, prompt: &str) -> Option<(usize, f64)> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?;
    let bin = repo.join("llama.cpp/build/bin/llama-eval-callback");
    if !bin.is_file() {
        skip("forward_oracle", "llama-eval-callback not built");
        return None;
    }
    let out = std::process::Command::new(&bin)
        .args(["-m", model.to_str()?, "-p", prompt, "-ngl", "0"])
        .output()
        .ok()?;
    assert!(out.status.success(), "eval-callback failed");
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);

    let mut want: Option<(usize, f64)> = None;
    let mut pending: Option<usize> = None;
    for line in text.lines() {
        if let Some(rest) = line.split("common_debug_cb_eval: ").nth(1) {
            let name = rest.split(" = ").next().unwrap_or("").trim();
            pending = (name == "result_output")
                .then(|| {
                    rest.rsplit_once('{')
                        .and_then(|(_, s)| s.split_once('}'))
                        .map(|(s, _)| {
                            s.split(',')
                                .filter_map(|x| x.trim().parse::<usize>().ok())
                                .product()
                        })
                })
                .flatten();
        } else if let Some(v) = line.trim().strip_prefix("sum = ") {
            if let (Some(n), Ok(sum)) = (pending.take(), v.trim().parse::<f64>()) {
                want.get_or_insert((n, sum));
            }
        }
    }
    want
}

#[test]
#[ignore = "runs the whole model on the CPU reference; takes minutes"]
fn full_forward_matches_llama_cpp() {
    let Some(path) = oracle_model() else {
        skip("full_forward_matches_llama_cpp", "set LLAMD_TEST_MODEL");
        return;
    };
    const PROMPT: &str = "Hi";

    let Some((n_vocab, want_sum)) = oracle_logits(&path, PROMPT) else {
        return;
    };

    let be = CpuBackend::new();
    let model = Model::open(&path).expect("open");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let ids = vocab.encode(PROMPT, false);
    assert_eq!(ids.len(), 1, "expected one token, got {ids:?}");

    let h = model.hparams.clone();
    let mut kv = KvCache::new(h.n_layer, h.kv_dim(), 8).unwrap();

    let t0 = std::time::Instant::now();
    let logits = forward(&be, &model, ids[0], 0, &mut kv).expect("forward");
    let secs = t0.elapsed().as_secs_f32();

    assert_eq!(
        logits.len(),
        n_vocab,
        "logit count disagrees with the oracle"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "forward produced NaN or inf"
    );

    let got: f64 = logits.iter().map(|&x| x as f64).sum();
    let scale = logits.iter().map(|x| x.abs() as f64).sum::<f64>().max(1.0);
    let rel = ((got - want_sum).abs() / scale) as f32;

    eprintln!("  {} layers in {secs:.1}s", h.n_layer);
    eprintln!("  logits: sum {got:.4}, oracle {want_sum:.4}, relative {rel:.3e}");
    // The top few, with their gaps. If the winner changes because the top two
    // are separated by less than the quantized-path error, that is a
    // reproducible consequence of a different algorithm. If the reference's
    // choice is not even near the top, it is a bug — and these are different
    // conclusions, so the ranking is printed rather than only the winner.
    eprintln!(
        "  first 3 logits: {:.4} {:.4} {:.4}   (oracle prints 5.9153 3.7523 2.0300)",
        logits[0], logits[1], logits[2]
    );
    eprintln!(
        "  last 3 logits:  {:.4} {:.4} {:.4}   (oracle prints 0.1184 0.1177 0.1058)",
        logits[n_vocab - 3],
        logits[n_vocab - 2],
        logits[n_vocab - 1]
    );

    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("  top 6 tokens:");
    for (i, (tok, lg)) in ranked.iter().take(6).enumerate() {
        eprintln!(
            "    {i}. {tok:>6} {:>14?}  logit {lg:>9.4}  gap {:.4}",
            vocab.text_of(*tok as u32).unwrap_or("?"),
            ranked[0].1 - lg
        );
    }

    // WHAT THIS TEST CAN AND CANNOT ASSERT.
    //
    // It cannot assert logit equality, and the reason is structural rather
    // than a matter of precision. Routing in this model matches llama.cpp
    // EXACTLY through layer 45 and closely at 46, then diverges at layer 47 —
    // where the router is nearly uniform (0.178, 0.150, 0.135, … 0.125,
    // 0.079, 0.074). A near-tied discrete choice, perturbed by the
    // quantized-path difference, selects different experts, and a different
    // expert is a different model.
    //
    // So an f32 reference cannot be expected to agree token-for-token with a
    // Q8_K implementation of an MoE model. Closing that needs LLAMD to
    // implement the same quantized dot product, not a looser tolerance here.
    //
    // What is assertable: the pass completes, the shape is right, the values
    // are finite, and the reference's own top token is near the top of ours
    // rather than anywhere in 151,936.
    let oracle_top = 525u32; // " are" — llama.cpp's greedy choice for "Hi"
    let rank = ranked
        .iter()
        .position(|(t, _)| *t as u32 == oracle_top)
        .expect("the reference's token is not in our ranking at all");
    assert!(
        rank < 8,
        "llama.cpp's token {oracle_top} ranks {rank} in ours — far enough to \
         indicate a bug rather than a routing tie"
    );
    eprintln!(
        "  llama.cpp's token ranks {rank} here, {:.4} below the top",
        ranked[0].1 - ranked[rank].1
    );
    eprintln!("  (logit sum gap {rel:.3e} — see the note above on why this is not asserted)");
}
