//! Where prefill time actually goes, per dispatch, measured on the device.
//!
//! §19 and §20 each built a counting model that predicted a large prefill gain
//! and delivered nothing. Both were consistent with every number available when
//! they were written, and neither could see where the time went. This stops
//! modelling: it reads GPU timestamps around every dispatch of a real 48-layer
//! pass and attributes the wall clock to the kernel that spent it.
//!
//! IT IS EXACT, not sampled. Every dispatch in this engine is followed by a full
//! memory barrier, so nothing overlaps — the intervals sum to the pass.

use llamd_models::Model;
use llamd_runtime::VulkanModel;
use llamd_testing::oracle_model;
use llamd_vulkan::ops;
use std::collections::HashMap;

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn where_prefill_time_goes() {
    let Some(path) = oracle_model() else {
        eprintln!("  SKIP where_prefill_time_goes: no LLAMD_TEST_MODEL");
        return;
    };
    if !llamd_vulkan::available() {
        eprintln!("  SKIP where_prefill_time_goes: no Vulkan");
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut toks = Vec::new();
    while toks.len() < 512 {
        toks.extend_from_slice(&vocab.encode(text, false));
    }
    toks.truncate(512);

    for &tile in &[64usize, 512] {
        vk.prefill_tile = tile;
        // Warm, unprofiled, so pipeline creation and first-touch are not charged.
        let _ = vk.prefill(&toks, 0).expect("warm");

        ops::profile(true).expect("timestamps");
        let wall = std::time::Instant::now();
        let _ = vk.prefill(&toks, 0).expect("prefill");
        let wall = wall.elapsed().as_secs_f64() * 1000.0;
        // prefill() submits one command buffer per tile and the reader reports the
        // LAST pass, so a multi-tile prefill is profiled tile by tile. Read what
        // the final tile did and scale by the tile count for the whole prompt.
        let last = ops::profile_read();
        ops::profile(false).expect("timestamps off");

        let tiles = toks.len().div_ceil(tile);
        let mut by: HashMap<&'static str, (f64, usize)> = HashMap::new();
        let mut total = 0.0;
        for (p, ms) in &last {
            let e = by.entry(p.shader_name()).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
            total += ms;
        }
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());

        eprintln!();
        eprintln!(
            "  ===== tile {tile}: {} dispatches in the last of {tiles} tiles, {:.1} ms on the GPU \
             ({:.0} ms wall for the whole 512-token prompt)",
            last.len(),
            total,
            wall
        );
        eprintln!(
            "  {:<24}{:>8}{:>11}{:>10}{:>11}",
            "kernel", "calls", "ms", "% of tile", "us/call"
        );
        for (name, (ms, n)) in &rows {
            eprintln!(
                "  {:<24}{:>8}{:>11}{:>10}{:>11}",
                name,
                n,
                format!("{ms:.2}"),
                format!("{:.1}%", 100.0 * ms / total),
                format!("{:.1}", ms * 1000.0 / *n as f64)
            );
        }
        // The question §20 left open, answered as one number.
        let expert: f64 = rows
            .iter()
            .filter(|(n, _)| n.contains("expert"))
            .map(|(_, (ms, _))| *ms)
            .sum();
        eprintln!(
            "  routed expert matmuls: {:.1}% of the tile — the other {:.1}% is everything else",
            100.0 * expert / total,
            100.0 * (1.0 - expert / total)
        );
    }

    // And decode, because the attention rewrite cost batched decode ~3.7% and the
    // point of having a profiler is not to guess why.
    for &batch in &[1usize, 8] {
        let toks: Vec<u32> = std::iter::repeat_n(toks[0], batch).collect();
        let pos: Vec<usize> = (0..batch).map(|_| 64).collect();
        let _ = vk.forward_batch(&toks, &pos).expect("warm");
        ops::profile(true).expect("timestamps");
        let _ = vk.forward_batch(&toks, &pos).expect("decode");
        let last = ops::profile_read();
        ops::profile(false).expect("off");

        let mut by: HashMap<&'static str, (f64, usize)> = HashMap::new();
        let mut total = 0.0;
        for (p, ms) in &last {
            let e = by.entry(p.shader_name()).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
            total += ms;
        }
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        eprintln!();
        eprintln!("  ===== decode batch {batch}, 64 tokens of context: {total:.2} ms on the GPU");
        for (name, (ms, n)) in rows.iter().take(6) {
            eprintln!(
                "  {:<24}{:>8}{:>11}{:>10}",
                name,
                n,
                format!("{ms:.2}"),
                format!("{:.1}%", 100.0 * ms / total)
            );
        }
    }
}

/// Did the attention rewrite cost decode anything?
///
/// §22 left this open. A before/after across two builds could not decide it: four
/// runs of ONE build spanned 98.82 to 117.18 tok/s at batch 1, which is 19%, and
/// the question was whether a 3% change was real. So both kernels are kept as
/// pipelines and this alternates them inside one process, best of three — the only
/// measurement shape this card's 64 MB of Infinity Cache and its run-to-run drift
/// cannot fake.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn attention_rewrite_against_the_previous_kernel() {
    let Some(path) = oracle_model() else {
        eprintln!("  SKIP attention_rewrite_ab: no LLAMD_TEST_MODEL");
        return;
    };
    if !llamd_vulkan::available() {
        eprintln!("  SKIP attention_rewrite_ab: no Vulkan");
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions,                 covering lexing, parsing, type checking, optimisation and code generation.";
    let mut prompt = Vec::new();
    while prompt.len() < 512 {
        prompt.extend_from_slice(&vocab.encode(text, false));
    }
    prompt.truncate(512);

    // Decode: `batch` streams, 16 passes, at a fixed 64 tokens of context.
    let decode = |vk: &mut VulkanModel, batch: usize, scalar: bool| -> f64 {
        ops::set_attention_scalar(scalar);
        let toks: Vec<u32> = std::iter::repeat_n(prompt[0], batch).collect();
        let pos: Vec<usize> = (0..batch).map(|_| 64).collect();
        let _ = vk.forward_batch(&toks, &pos).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for _ in 0..16 {
                let _ = vk.forward_batch(&toks, &pos).expect("decode");
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        (batch * 16) as f64 / best
    };
    let prefill = |vk: &mut VulkanModel, scalar: bool| -> f64 {
        ops::set_attention_scalar(scalar);
        let _ = vk.prefill(&prompt, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&prompt, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };

    eprintln!("  alternating arms in one process, best of 3");
    eprintln!(
        "  {:<18}{:>12}{:>12}{:>10}",
        "case", "rewritten", "previous", "speedup"
    );
    for batch in [1usize, 2, 4, 8] {
        // Rewritten first on even batches, previous first on odd, so a warming
        // trend cannot favour one arm consistently.
        let (new, old) = if batch % 2 == 0 {
            let n = decode(&mut vk, batch, false);
            (n, decode(&mut vk, batch, true))
        } else {
            let o = decode(&mut vk, batch, true);
            (decode(&mut vk, batch, false), o)
        };
        eprintln!(
            "  {:<18}{:>12}{:>12}{:>10}",
            format!("decode batch {batch}"),
            format!("{new:.2}"),
            format!("{old:.2}"),
            format!("{:.3}x", new / old)
        );
    }
    let po = prefill(&mut vk, true);
    let pn = prefill(&mut vk, false);
    eprintln!(
        "  {:<18}{:>12}{:>12}{:>10}",
        "prefill pp512",
        format!("{pn:.1}"),
        format!("{po:.1}"),
        format!("{:.3}x", pn / po)
    );
    ops::set_attention_scalar(false);

    // And the arms must agree, or the comparison is between two different models.
    ops::set_attention_scalar(true);
    let a = vk.prefill(&prompt, 0).expect("prefill scalar");
    ops::set_attention_scalar(false);
    let b = vk.prefill(&prompt, 0).expect("prefill vec4");
    eprintln!(
        "  same next token from both kernels: {} ({a} vs {b})",
        if a == b { "yes" } else { "NO" }
    );
    assert_eq!(a, b, "the two attention kernels disagree on the next token");
}
