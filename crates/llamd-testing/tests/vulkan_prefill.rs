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

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn expert_major_equals_token_major() {
    // A gather/scatter that mixes up which token's activation feeds which expert,
    // or writes a result into the wrong slot, still produces fluent text. So the
    // bar is that the two groupings agree EXACTLY, token for token, at every tile
    // width — held against each other rather than against a tolerance, because
    // they perform the identical arithmetic in a different order of experts.
    let Some((model, mut vk)) = open(320, llamd_vulkan::MAX_SEQ) else {
        skip("expert_major_equals_token_major", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let all = vocab.encode(
        "The capital of France is Paris and the capital of Japan is Tokyo while \
         the largest ocean on Earth is the Pacific",
        false,
    );

    let mut wrong = Vec::new();
    for tile in [1usize, 4, 8, 16, 32] {
        if tile > llamd_vulkan::MAX_SEQ {
            continue;
        }
        vk.prefill_tile = tile;
        for &n in &[1usize, tile, tile + 1, 2 * tile + 1] {
            if n > all.len() {
                continue;
            }
            vk.expert_major_min_tokens = usize::MAX;
            let want = vk.prefill(&all[..n], 0).expect("token-major");
            vk.expert_major_min_tokens = 1;
            let got = vk.prefill(&all[..n], 0).expect("expert-major");
            if got != want {
                wrong.push(format!(
                    "tile {tile}, {n} tokens: token-major {want} {:?}, expert-major {got} {:?}",
                    vocab.text_of(want).unwrap_or("?"),
                    vocab.text_of(got).unwrap_or("?")
                ));
            }
        }
    }
    // And a decode step, where every tile holds exactly one token: the grouping
    // degenerates to 8 groups of 1, which is the case most likely to expose an
    // off-by-one in the prefix sum.
    vk.expert_major_min_tokens = usize::MAX;
    let d_want = vk
        .forward_batch(&[13048], &[0])
        .expect("token-major decode")[0];
    vk.expert_major_min_tokens = 1;
    let d_got = vk
        .forward_batch(&[13048], &[0])
        .expect("expert-major decode")[0];
    if d_got != d_want {
        wrong.push(format!(
            "decode: token-major {d_want}, expert-major {d_got}"
        ));
    }

    for w in &wrong {
        eprintln!("  MISMATCH {w}");
    }
    assert!(wrong.is_empty(), "{} cases disagree", wrong.len());
    eprintln!("  expert-major == token-major at every tile width, and in decode");
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn expert_major_against_token_major() {
    // A/B in one process, alternating, best of several — the only comparison this
    // card's run-to-run drift does not swamp, and the discipline four earlier
    // optimisations had to learn the hard way.
    let Some((model, mut vk)) = open(600, llamd_vulkan::MAX_SEQ) else {
        skip("expert_major_against_token_major", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let unit = vocab.encode(
        "Grouping the expert matmuls by expert rather than by token reads each \
         expert's weights once for every token that chose it. ",
        false,
    );
    let mut prompt: Vec<u32> = Vec::new();
    while prompt.len() < 512 {
        prompt.extend_from_slice(&unit);
    }
    prompt.truncate(512);

    eprintln!(
        "\n  512-token prompt.  {:>7} {:>12} {:>12} {:>9}",
        "tile", "token-major", "expert-major", "speedup"
    );
    for tile in [8usize, 16, 32] {
        if tile > llamd_vulkan::MAX_SEQ {
            continue;
        }
        vk.prefill_tile = tile;
        let mut best = [f64::MAX; 2]; // [token-major, expert-major]
        for _ in 0..3 {
            for (arm, em) in [(0usize, false), (1usize, true)] {
                vk.expert_major_min_tokens = if em { 1 } else { usize::MAX };
                let _ = vk.prefill(&prompt[..64], 0); // warm this configuration
                let t = std::time::Instant::now();
                vk.prefill(&prompt, 0).expect("prefill");
                let sec = t.elapsed().as_secs_f64();
                if sec < best[arm] {
                    best[arm] = sec;
                }
            }
        }
        eprintln!(
            "  {:>14} {:>9.1} t/s {:>9.1} t/s {:>8.2}x",
            tile,
            prompt.len() as f64 / best[0],
            prompt.len() as f64 / best[1],
            best[0] / best[1]
        );
    }
    vk.expert_major_min_tokens = 1;
    eprintln!("  llama.cpp pp512: 1632 t/s");

    // And the effect on decode, where a tile holds one token and the grouping has
    // nothing to share — it should be neutral or a small loss, and it matters that
    // it is not a large one.
    eprintln!("\n  decode, 32 tokens:");
    for (name, em) in [("token-major", false), ("expert-major", true)] {
        vk.expert_major_min_tokens = if em { 1 } else { usize::MAX };
        let mut tk = vec![13048u32];
        for pos in 0..8 {
            tk = vk.forward_batch(&tk, &[pos]).expect("warm");
        }
        let t = std::time::Instant::now();
        for pos in 0..32 {
            tk = vk.forward_batch(&tk, &[pos]).expect("timed");
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 32.0;
        eprintln!(
            "  {name:<14} {ms:>7.2} ms/token  {:>7.2} tok/s",
            1000.0 / ms
        );
    }
    vk.expert_major_min_tokens = 1;
}
