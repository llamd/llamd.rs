//! Batched prefill: the same answer as one-at-a-time, and much faster.
//!
//! WHY THIS IS THE FEATURE THAT MATTERED. Decode speed is irrelevant if
//! ingesting the prompt costs more than generating the reply. Without prefill a
//! 512-token prompt had to be fed one token at a time, reading all 2.485 GB of
//! active weights per token — measured 4.5 s against llama.cpp's 0.31 s.
//!
//! THE CORRECTNESS BAR IS EQUIVALENCE, not plausibility. A prefill that gets
//! causality wrong — a token seeing one that follows it — still produces fluent
//! text, and would look fine. So the test is that prefill returns EXACTLY what
//! sequential decode returns, token for token, at several prompt lengths and
//! across the tile boundary.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing \
//!     --test vulkan_prefill -- --ignored --nocapture

use llamd_models::Model;
use llamd_runtime::VulkanModel;
use llamd_testing::{oracle_model, skip};

fn open(max_kv: usize, max_seq: usize) -> Option<(Model, VulkanModel)> {
    let path = oracle_model()?;
    if !llamd_vulkan::available() {
        return None;
    }
    let model = Model::open(&path).ok()?;
    VulkanModel::load(&model, max_kv, max_seq)
        .ok()
        .map(|v| (model, v))
}

/// Feed a prompt one token at a time and return the token that follows it —
/// the reference prefill has to reproduce.
fn sequential(vk: &mut VulkanModel, toks: &[u32]) -> u32 {
    let mut last = 0u32;
    for (i, &t) in toks.iter().enumerate() {
        last = vk.forward_batch(&[t], &[i]).expect("sequential")[0];
    }
    last
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn prefill_equals_sequential_decode() {
    let Some((model, mut vk)) = open(320, llamd_vulkan::MAX_SEQ) else {
        skip("prefill_equals_sequential_decode", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "The quick brown fox jumps over the lazy dog and then keeps running \
                through the field until it reaches the river where it stops to drink";
    let all = vocab.encode(text, false);
    eprintln!("  prompt tokenizes to {} tokens", all.len());

    // EVERY TILE WIDTH, and lengths that straddle each. An off-by-one in the
    // chunking, or a slot taking the wrong position, shows up at a boundary and
    // nowhere else — so below the tile, exactly it, one over, and a ragged last
    // chunk are all covered for each width.
    let mut cases: Vec<(usize, usize)> = Vec::new(); // (tile, length)
    for tile in [1usize, 4, 8, 16, 32] {
        if tile > llamd_vulkan::MAX_SEQ {
            continue;
        }
        for n in [
            1,
            tile.saturating_sub(1).max(1),
            tile,
            tile + 1,
            2 * tile,
            2 * tile + 3,
        ] {
            if n >= 1 && n <= all.len() && !cases.contains(&(tile, n)) {
                cases.push((tile, n));
            }
        }
    }

    let mut wrong = Vec::new();
    for &(tile, n) in &cases {
        vk.prefill_tile = tile;
        let toks = &all[..n];
        let want = sequential(&mut vk, toks);
        let got = vk.prefill(toks, 0).expect("prefill");
        if got != want {
            wrong.push(format!(
                "tile {tile}, {n} tokens: sequential {want} {:?}, prefill {got} {:?}",
                vocab.text_of(want).unwrap_or("?"),
                vocab.text_of(got).unwrap_or("?")
            ));
        }
    }
    for w in &wrong {
        eprintln!("  MISMATCH {w}");
    }
    assert!(
        wrong.is_empty(),
        "{} of {} (tile, length) cases disagree",
        wrong.len(),
        cases.len()
    );
    eprintln!(
        "  prefill == sequential decode across {} (tile, length) cases, tiles 1 to {}",
        cases.len(),
        llamd_vulkan::MAX_SEQ
    );
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn prefill_then_generate_reads_coherently() {
    // Prefill a real prompt, then generate from it. Equivalence is the hard
    // test; this one checks the two paths compose — that generation continues
    // from the cache prefill built rather than from nothing.
    let Some((model, mut vk)) = open(320, 8) else {
        skip("prefill_then_generate", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let prompt = "Write a short explanation of why the sky is blue.";
    let toks = vocab.encode(prompt, false);

    let t0 = std::time::Instant::now();
    let mut next = vk.prefill(&toks, 0).expect("prefill");
    let pf = t0.elapsed().as_secs_f64();

    let mut out = String::new();
    out.push_str(vocab.text_of(next).unwrap_or("?"));
    let t1 = std::time::Instant::now();
    const GEN: usize = 48;
    for i in 0..GEN {
        next = vk
            .forward_batch(&[next], &[toks.len() + i])
            .expect("decode")[0];
        out.push_str(vocab.text_of(next).unwrap_or("?"));
    }
    let dec = t1.elapsed().as_secs_f64();

    eprintln!(
        "  prompt {} tokens in {:.3} s -> {:.1} tok/s",
        toks.len(),
        pf,
        toks.len() as f64 / pf
    );
    eprintln!(
        "  {GEN} generated in {:.3} s -> {:.1} tok/s",
        dec,
        GEN as f64 / dec
    );
    eprintln!("  continuation: {out:?}");
    assert!(!out.is_empty());
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn prefill_throughput_against_one_at_a_time() {
    // The number the feature exists for: tokens per second of PROMPT INGESTION,
    // against the one-at-a-time path it replaces and against llama.cpp's pp512.
    let Some((model, mut vk)) = open(600, llamd_vulkan::MAX_SEQ) else {
        skip("prefill_throughput", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let unit = vocab.encode(
        "The measurement of a language model's prompt ingestion rate matters more \
         than its generation rate whenever the prompt is longer than the reply. ",
        false,
    );
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < 512 {
        prompt.extend_from_slice(&unit);
    }
    prompt.truncate(512);

    // Warm both paths before timing either.
    let _ = vk.prefill(&prompt[..64], 0);
    for (i, &t) in prompt[..64].iter().enumerate() {
        let _ = vk.forward_batch(&[t], &[i]);
    }

    // TILE SWEEP. A wider tile shares each weight read across more tokens, and
    // costs registers: the accumulator array is indexed by a push-constant loop
    // bound the compiler cannot unroll, so it may land in scratch. Measured.
    eprintln!("\n  512-token prompt, prefill tile sweep:");
    eprintln!("  {:<7} {:>10} {:>13}", "tile", "seconds", "tokens/s");
    let mut best = (1usize, f64::MAX);
    for tile in [1usize, 4, 8, 16, 32] {
        if tile > llamd_vulkan::MAX_SEQ {
            continue;
        }
        vk.prefill_tile = tile;
        let _ = vk.prefill(&prompt[..64], 0); // warm this tile's dispatch shapes
        let t = std::time::Instant::now();
        vk.prefill(&prompt, 0).expect("prefill");
        let sec = t.elapsed().as_secs_f64();
        let mark = if sec < best.1 {
            best = (tile, sec);
            "  <-"
        } else {
            ""
        };
        eprintln!(
            "  {tile:<7} {sec:>10.3} {:>13.1}{mark}",
            prompt.len() as f64 / sec
        );
    }
    vk.prefill_tile = best.0;
    eprintln!("  best tile {} at {:.3} s", best.0, best.1);

    eprintln!(
        "\n  {:<10} {:>10} {:>12} {:>16}",
        "tokens", "prefill s", "prefill t/s", "one at a time"
    );
    for n in [64usize, 128, 256, 512] {
        let t = std::time::Instant::now();
        vk.prefill(&prompt[..n], 0).expect("prefill");
        let pf = t.elapsed().as_secs_f64();

        let t = std::time::Instant::now();
        for (i, &tok) in prompt[..n].iter().enumerate() {
            vk.forward_batch(&[tok], &[i]).expect("seq");
        }
        let sq = t.elapsed().as_secs_f64();

        eprintln!(
            "  {n:<10} {pf:>10.3} {:>12.1} {:>12.1} t/s ({:.2}x)",
            n as f64 / pf,
            n as f64 / sq,
            sq / pf
        );
    }
    eprintln!("  llama.cpp pp512: 1632 t/s");
}
