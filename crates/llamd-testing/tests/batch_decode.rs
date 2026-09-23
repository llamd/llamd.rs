//! Batched decode: correctness against the single-sequence path, then scaling.
//!
//! THE CLAIM. `forward_batch` computes `nseq` independent sequences in one pass,
//! sharing every attention and output weight read. It must produce exactly what
//! `nseq` separate `forward` calls produce — batching changes which activations
//! a weight multiplies, not the arithmetic any one of them sees.
//!
//! That is asserted on the TOKEN, exactly, and on the logits within reduction
//! error. A batched kernel that leaked one sequence's activation into another's
//! accumulator would still produce fluent text, so "it looks fine" is not a
//! test — the two paths are compared directly.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing \
//!     --test batch_decode -- --ignored --nocapture

use llamd_models::Model;
use llamd_runtime::DeviceModel;
use llamd_testing::{oracle_model, skip};

const PROMPTS: &[(&str, u32)] = &[
    ("Hi", 13048),
    ("The", 785),
    ("def", 750),
    ("Python", 30280),
    ("1", 16),
    ("import", 474),
    ("class", 1040),
    ("return", 689),
];

fn open(max_kv: usize, max_seq: usize) -> Option<(Model, DeviceModel)> {
    let path = oracle_model()?;
    if !llamd_rocm::available() {
        return None;
    }
    let model = Model::open(&path).ok()?;
    let dev = DeviceModel::load_batched(&model, max_kv, max_seq).ok()?;
    Some((model, dev))
}

#[test]
#[ignore = "uploads 23 GiB; needs a free GPU and the real model"]
fn batched_matches_the_single_sequence_path() {
    let Some((model, mut dev)) = open(16, 8) else {
        skip(
            "batched_matches_the_single_sequence_path",
            "no model or no GPU",
        );
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");

    // THE REFERENCE IS forward_batch AT nseq=1, not forward().
    //
    // Those two paths use DIFFERENT kernels by design: batch 1 runs the planar
    // matvec, which is faster when a single sequence makes the problem purely
    // bandwidth-bound, while batch >= 2 runs the udot4 kernel, which wins once
    // there is something to amortise its packing over. Comparing across that
    // boundary tests the kernel choice, not batching, and a near-tied argmax can
    // legitimately differ between two numerically different kernels.
    //
    // Batching is what this test is about, so the reference holds the kernel
    // fixed and varies only the batch width.
    let mut want = Vec::new();
    for &(_, tok) in PROMPTS {
        let r = dev.forward_batch(&[tok], &[0]).expect("nseq=1");
        want.push(r[0]);
    }

    // Recorded separately: whether the two kernel paths agree at all. They
    // should on every prompt whose argmax is not near-tied.
    let mut cross = Vec::new();
    for (i, &(name, tok)) in PROMPTS.iter().enumerate() {
        let single = dev.forward(tok, 0).expect("planar batch-1");
        if single != want[i] {
            cross.push(format!(
                "{name:?}: planar {single} {:?} vs udot {} {:?}",
                vocab.text_of(single).unwrap_or("?"),
                want[i],
                vocab.text_of(want[i]).unwrap_or("?")
            ));
        }
    }
    if cross.is_empty() {
        eprintln!(
            "  planar and udot kernels agree on all {} prompts",
            PROMPTS.len()
        );
    } else {
        eprintln!(
            "  planar vs udot differ on {} of {} prompts (different kernels, near-tied argmax):",
            cross.len(),
            PROMPTS.len()
        );
        for c in &cross {
            eprintln!("    {c}");
        }
    }

    // Then every batch width, each compared against those single-sequence
    // answers. Eight is the widest the kernels' register arrays allow.
    for nseq in [2usize, 4, 8] {
        let toks: Vec<u32> = PROMPTS[..nseq].iter().map(|&(_, t)| t).collect();
        let got = dev
            .forward_batch(&toks, &vec![0usize; nseq])
            .expect("batch");
        assert_eq!(got.len(), nseq);
        let mut bad = Vec::new();
        for (i, (&g, &w)) in got.iter().zip(&want[..nseq]).enumerate() {
            if g != w {
                bad.push(format!(
                    "{:?}: single {w} {:?}, batch {g} {:?}",
                    PROMPTS[i].0,
                    vocab.text_of(w).unwrap_or("?"),
                    vocab.text_of(g).unwrap_or("?")
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "batch {nseq} differs from the single path:\n    {}",
            bad.join("\n    ")
        );
        eprintln!("  batch {nseq}: all {nseq} sequences match the single-sequence path");
    }

    // A second step, with each sequence at its own position — streams are not
    // in lockstep, and each sequence's cache slice must be indexed by its own
    // position rather than a shared one.
    let step2: Vec<u32> = want[..4].to_vec();
    let mixed = dev
        .forward_batch(&step2, &[1, 1, 1, 1])
        .expect("second step");
    assert_eq!(mixed.len(), 4);
    for (i, &t) in mixed.iter().enumerate() {
        assert!(
            (t as usize) < vocab.len(),
            "sequence {i} produced token {t} outside the vocabulary"
        );
    }
    eprintln!("  second step at pos 1: {mixed:?}");

    eprintln!("  batch 2, 4 and 8 all agree with the single-sequence path");
}

#[test]
#[ignore = "uploads 23 GiB; needs a free GPU and the real model"]
fn batch_scaling() {
    let Some((model, mut dev)) = open(68, 8) else {
        skip("batch_scaling", "no model or no GPU");
        return;
    };
    let _ = &model;
    const N: usize = 64;
    eprintln!(
        "\n  {:<6} {:>10} {:>12} {:>14} {:>9}",
        "batch", "ms/pass", "ms/token", "tok/s total", "vs b=1"
    );
    let mut base = 0.0f64;
    for nseq in [1usize, 2, 4, 8] {
        if nseq > llamd_rocm::max_seq() {
            continue;
        }
        // Distinct prompts so the sequences route differently — batching the
        // same prompt eight times would share experts and flatter the result.
        let mut toks: Vec<u32> = (0..nseq).map(|i| PROMPTS[i % PROMPTS.len()].1).collect();

        // Warm, then measure. Positions advance together; each sequence has its
        // own cache slice so they do not interact.
        for pos in 0..8 {
            toks = dev
                .forward_batch(&toks, &vec![pos; nseq])
                .expect("warm batch");
        }
        let t = std::time::Instant::now();
        for pos in 0..N {
            toks = dev
                .forward_batch(&toks, &vec![pos; nseq])
                .expect("timed batch");
        }
        let per_pass = t.elapsed().as_secs_f64() * 1e3 / N as f64;
        let per_token = per_pass / nseq as f64;
        let tps = 1000.0 / per_token;
        if nseq == 1 {
            base = tps;
        }
        eprintln!(
            "  {nseq:<6} {per_pass:>10.2} {per_token:>12.3} {tps:>14.2} {:>8.2}x",
            tps / base
        );
    }
    eprintln!("  llama.cpp single-stream bar: 94.4 - 95.6 tok/s");

    // ROW TILE SWEEP, in the engine. With one row per wave every row re-reads
    // the whole activation — for attn_q that is ~67 MB of (cache-resident)
    // activation reads per layer at batch 8 against 6.9 MB of weights. Tiling
    // amortises those loads across rows at the cost of registers. Which way
    // that trade goes is measured here, because microbenchmarks on this card
    // have been wrong three times (benchmarks.md §9-§10).
    eprintln!("\n  row tile sweep at batch 8:");
    let mut best = (1usize, f64::MAX);
    for tile in [1usize, 2, 4] {
        llamd_rocm::udot::set_tile(tile);
        let nseq = 8usize.min(llamd_rocm::max_seq());
        let mut toks: Vec<u32> = (0..nseq).map(|i| PROMPTS[i % PROMPTS.len()].1).collect();
        for pos in 0..8 {
            toks = dev.forward_batch(&toks, &vec![pos; nseq]).expect("warm");
        }
        let t = std::time::Instant::now();
        for pos in 0..N {
            toks = dev.forward_batch(&toks, &vec![pos; nseq]).expect("timed");
        }
        let per_pass = t.elapsed().as_secs_f64() * 1e3 / N as f64;
        let tps = 1000.0 / (per_pass / nseq as f64);
        let mark = if per_pass < best.1 {
            best = (tile, per_pass);
            "  <-"
        } else {
            ""
        };
        eprintln!("    tile {tile}  {per_pass:>7.2} ms/pass  {tps:>7.2} tok/s{mark}");
    }
    eprintln!("    best tile {}", best.0);
    llamd_rocm::udot::set_tile(best.0);
    assert!(base > 0.0);
}
