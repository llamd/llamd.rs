//! End-to-end generation throughput, measured the way llama.cpp reports it.
//!
//! WHY THIS EXISTS. Every throughput number LLAMD had was inferred from a
//! single-token oracle test: one forward, divide one by the seconds. That
//! conflates the first token — which faults the whole weight file in from page
//! cache — with the steady state, and it never exercised the KV cache growing
//! across positions. llama.cpp reports `prompt_ms` and `predicted_ms`
//! separately for exactly that reason, and a comparison that does not separate
//! them is comparing two different quantities.
//!
//! WHAT IS MEASURED. Greedy generation from a single-token prompt, argmax each
//! step, KV cache growing by one position per token. The first token is
//! reported on its own; `tg` is the mean over the remainder. That matches
//! llama.cpp's `predicted_per_second` closely enough to put the two numbers in
//! one table honestly.
//!
//! WHAT IS NOT MEASURED. Batched prefill — LLAMD's forward is single-token, so
//! there is no prompt-processing number to report, and the docs must not
//! invent one.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing \
//!     --test throughput -- --ignored --nocapture

use llamd_cpu::CpuBackend;
use llamd_models::Model;
use llamd_runtime::{KvCache, forward};
use llamd_testing::{oracle_model, skip};

const N_GEN: usize = 16;

#[test]
#[ignore = "generates tokens on the CPU reference; takes minutes"]
fn generation_throughput_cpu() {
    let Some(path) = oracle_model() else {
        skip("generation_throughput_cpu", "set LLAMD_TEST_MODEL");
        return;
    };
    let be = CpuBackend::new();
    let load0 = std::time::Instant::now();
    let model = Model::open(&path).expect("open");
    let load_ms = load0.elapsed().as_millis();
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let h = model.hparams.clone();

    let mut kv = KvCache::new(h.n_layer, h.kv_dim(), N_GEN + 2).unwrap();
    let mut token = 13048u32; // "Hi", the same prompt llama.cpp was given
    let mut text = String::new();
    let mut per_token = Vec::with_capacity(N_GEN);

    for pos in 0..N_GEN {
        let t0 = std::time::Instant::now();
        let logits = forward(&be, &model, token, pos as u32, &mut kv).expect("forward");
        per_token.push(t0.elapsed().as_secs_f64());
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        text.push_str(vocab.text_of(token).unwrap_or("?"));
    }

    let first = per_token[0];
    let rest: f64 = per_token[1..].iter().sum();
    let tg = (per_token.len() - 1) as f64 / rest;

    eprintln!("\n  model load          {load_ms} ms");
    eprintln!("  first token         {:.3} s", first);
    eprintln!(
        "  steady state        {:.3} s/token  ->  {tg:.4} tok/s",
        rest / (per_token.len() - 1) as f64
    );
    eprintln!("  generated           {text:?}");
    eprintln!(
        "  per-token spread    min {:.3} s  max {:.3} s",
        per_token[1..].iter().cloned().fold(f64::MAX, f64::min),
        per_token[1..].iter().cloned().fold(0.0, f64::max)
    );

    // The point of the test is the measurement, but a collapse to nonsense
    // should still fail rather than print a pretty number.
    assert!(tg > 0.0 && tg < 1000.0, "implausible throughput {tg}");
    assert!(!text.is_empty());
}
