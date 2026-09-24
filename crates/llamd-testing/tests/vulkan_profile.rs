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
}
