//! Greedy token agreement with llama.cpp across several prompts.
//!
//! THE CLAIM THIS TEST MAKES. Given the same single-token prompt, LLAMD's
//! argmax over 151,936 logits is the same token llama.cpp emits. That is only
//! possible because `llamd-quant::dot_q6k_q8k` performs the same Q6_K × Q8_K
//! arithmetic ggml does — with the f32 reference, a near-tied MoE routing
//! decision at layer 47 flipped and the tokens differed.
//!
//! WHY SEVERAL PROMPTS. One agreement could be luck: most routing decisions
//! are not close, so most prompts would agree even with a slightly wrong
//! implementation. Each additional prompt is another chance for a near-tie to
//! expose a difference.
//!
//! The expectations are llama.cpp's own greedy output, captured with a
//! pass-through chat template so nothing wraps the prompt.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing --test token_agreement -- --ignored

use llamd_cpu::CpuBackend;
use llamd_models::Model;
use llamd_runtime::{KvCache, forward};
use llamd_testing::{oracle_model, skip};

/// (prompt, prompt token, llama.cpp's greedy next token, its text)
const CASES: &[(&str, u32, u32, &str)] = &[
    ("Hi", 13048, 525, " are"),
    ("The", 785, 525, " are"),
    ("def", 750, 374, " is"),
    ("Python", 30280, 25, ":"),
    ("1", 16, 13, "."),
];

#[test]
#[ignore = "runs the whole model once per prompt on the CPU reference"]
fn greedy_tokens_agree_with_llama_cpp() {
    let Some(path) = oracle_model() else {
        skip("greedy_tokens_agree_with_llama_cpp", "set LLAMD_TEST_MODEL");
        return;
    };
    let be = CpuBackend::new();
    let model = Model::open(&path).expect("open");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let h = model.hparams.clone();

    let mut wrong = Vec::new();
    for &(prompt, want_in, want_out, want_text) in CASES {
        // The prompt must tokenize to exactly the id the case claims, or the
        // comparison is against a different input than llama.cpp saw.
        let ids = vocab.encode(prompt, false);
        assert_eq!(
            ids,
            vec![want_in],
            "prompt {prompt:?} did not tokenize as expected"
        );

        let mut kv = KvCache::new(h.n_layer, h.kv_dim(), 4).unwrap();
        let t0 = std::time::Instant::now();
        let logits = forward(&be, &model, ids[0], 0, &mut kv).expect("forward");
        let secs = t0.elapsed().as_secs_f32();

        let (top, &top_logit) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let second = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != top)
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let ok = top as u32 == want_out;
        eprintln!(
            "  {:<8} -> {:<10} want {:<10} {}  margin {:.4}  ({secs:.1}s)",
            format!("{prompt:?}"),
            format!("{:?}", vocab.text_of(top as u32).unwrap_or("?")),
            format!("{want_text:?}"),
            if ok { "ok" } else { "MISMATCH" },
            top_logit - second,
        );
        if !ok {
            wrong.push(format!(
                "{prompt:?}: llama.cpp {want_out} {want_text:?}, LLAMD {top} {:?}",
                vocab.text_of(top as u32).unwrap_or("?")
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} prompts disagree:\n  {}",
        wrong.len(),
        CASES.len(),
        wrong.join("\n  ")
    );
    eprintln!("  all {} prompts agree with llama.cpp", CASES.len());
}
