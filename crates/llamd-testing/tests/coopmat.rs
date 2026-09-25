//! What a u8 cooperative matrix is worth on this part, measured before a kernel is
//! built on it.
//!
//! §23 established the shape exists — 16x16x16, A u8, B u8, C u32, subgroup scope —
//! and that it is the §V2 identity in hardware. It did not establish what a Q6_K
//! kernel could get from it, and there is a specific reason to doubt the headline
//! density: Q6_K carries an int8 scale per 16 weights, the MMA's K dimension is
//! also 16, and the scale cannot be folded into the operand because weights are
//! 0..63 and their product with an int8 scale overflows u8. So every MMA must be
//! scaled per row and corrected before the next group joins it, which is 256
//! scalar operations per 4096-MAC instruction.
//!
//! Three projections in this project have predicted large gains and returned ~0.
//! This one gets measured first.

use llamd_models::Model;
use llamd_runtime::VulkanModel;
use llamd_testing::oracle_model;
use llamd_vulkan::{VkBuf, ops};

fn ok() -> bool {
    llamd_vulkan::available() && llamd_vulkan::has_coopmat()
}

#[test]
fn reports_whether_cooperative_matrix_is_available() {
    if !llamd_vulkan::available() {
        eprintln!("  SKIP: no Vulkan");
        return;
    }
    match llamd_vulkan::coopmat_report() {
        Some(r) if !r.is_empty() => {
            let n = r.lines().count();
            let u8u8 = r
                .lines()
                .filter(|l| l.contains("A u8") && l.contains("B u8"))
                .count();
            eprintln!("  cooperative matrix: {n} shapes, {u8u8} of them u8 x u8");
            eprintln!("  enabled on the device: {}", llamd_vulkan::has_coopmat());
            assert!(u8u8 > 0, "expected a u8 x u8 shape on gfx1201");
        }
        _ => eprintln!("  no VK_KHR_cooperative_matrix on this device"),
    }
}

#[test]
#[ignore = "needs a free GPU"]
fn what_the_u8_matrix_is_worth() {
    if !ok() {
        eprintln!("  SKIP what_the_u8_matrix_is_worth: no coopmat");
        return;
    }
    // One 16x16 u8 tile each, and a 4 KiB dword view for the dot-product arm.
    let a = VkBuf::from_bytes(&vec![1u8; 256]).expect("a");
    let b = VkBuf::from_bytes(&vec![1u8; 256]).expect("b");
    let a4 = VkBuf::from_bytes(&vec![1u8; 4096]).expect("a4");
    let b4 = VkBuf::from_bytes(&vec![1u8; 4096]).expect("b4");
    let groups: u32 = 2048; // enough subgroups to fill the card
    let out = VkBuf::new(groups as usize * 4).expect("out");

    let iters = 4096usize;
    // MACs per subgroup-iteration is one 16x16x16 product.
    let macs = groups as f64 * iters as f64 * 16.0 * 16.0 * 16.0;

    let run = |mode: usize| -> f64 {
        // Warm, then best of three.
        ops::begin().expect("begin");
        ops::coopmat_probe(&a, &b, &a4, &b4, &out, groups, iters, mode).expect("probe");
        ops::end_and_wait().expect("wait");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            ops::begin().expect("begin");
            ops::coopmat_probe(&a, &b, &a4, &b4, &out, groups, iters, mode).expect("probe");
            ops::end_and_wait().expect("wait");
            best = best.min(t.elapsed().as_secs_f64());
        }
        macs / best / 1e9
    };

    let names = [
        "MMA, no fold",
        "MMA + naive fold",
        "dotPacked4x8",
        "MMA + matrix fold",
    ];
    let mut rate = [0.0f64; 4];
    for mode in 0..4 {
        rate[mode] = run(mode);
        eprintln!(
            "  {:<20}{:>12} G MAC/s",
            names[mode],
            format!("{:.0}", rate[mode])
        );
    }

    // Correctness of the MMA itself: all-ones operands, so every element of C is
    // 16 per iteration, and lane 0 sums elements 0, 64, 128 and 192.
    ops::begin().expect("begin");
    ops::coopmat_probe(&a, &b, &a4, &b4, &out, 1, 8, 0).expect("probe");
    ops::end_and_wait().expect("wait");
    let mut got = vec![0.0f32; 1];
    out.download_f32(&mut got).expect("download");
    let want = 4.0 * 16.0 * 8.0;
    eprintln!(
        "  MMA of all-ones, 8 iterations: {} (expected {want})",
        got[0]
    );
    assert_eq!(
        got[0], want as f32,
        "the u8 MMA does not compute what it should"
    );

    // The matrix-domain fold must give the SAME answer as no fold at all, since
    // its broadcast scale is 1.0 — otherwise the stride-0 load does not mean what
    // it is being relied on to mean.
    ops::begin().expect("begin");
    ops::coopmat_probe(&a, &b, &a4, &b4, &out, 1, 8, 3).expect("probe");
    ops::end_and_wait().expect("wait");
    let mut got3 = vec![0.0f32; 1];
    out.download_f32(&mut got3).expect("download");
    eprintln!(
        "  matrix-domain fold, same case: {} (expected {want})",
        got3[0]
    );
    assert_eq!(
        got3[0], want as f32,
        "a stride-0 row-major load is not the per-column broadcast the design needs"
    );

    eprintln!();
    eprintln!("  MMA ceiling over the dot path: {:.2}x", rate[0] / rate[2]);
    eprintln!("  naive fold over the dot path:  {:.2}x", rate[1] / rate[2]);
    eprintln!("  MATRIX fold over the dot path: {:.2}x", rate[3] / rate[2]);
    eprintln!(
        "  the naive fold keeps:          {:.0}% of the ceiling",
        100.0 * rate[1] / rate[0]
    );
    eprintln!(
        "  the matrix fold keeps:         {:.0}% of the ceiling",
        100.0 * rate[3] / rate[0]
    );
}

/// The cooperative-matrix expert kernel against the kernel it replaces.
///
/// It cannot be bit-exact and is not asked to be: an MMA sums 16 products in
/// hardware order and the scale is applied as one f32 multiply per group, where the
/// dot path sums two 16-groups as integers and converts once. Both lose the same
/// kind of precision at ~1e-7 relative. The bar is the one that matters — the same
/// emitted token — across shapes that straddle the kernel's tiling: token counts
/// either side of TT*16 = 32, and prefill tiles either side of the row blocking.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn coopmat_experts_agree_with_the_dot_kernel() {
    if !ok() {
        eprintln!("  SKIP coopmat_experts_agree: no coopmat");
        return;
    }
    let Some(path) = oracle_model() else {
        eprintln!("  SKIP coopmat_experts_agree: no LLAMD_TEST_MODEL");
        return;
    };
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions,                 covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 300 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(300);

    let mut wrong = Vec::new();
    let mut cases = 0;
    for &tile in &[32usize, 64, 128, 256] {
        for &n in &[17usize, 33, 64, 65, 129, 300] {
            if n > all.len() {
                continue;
            }
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_expert_coopmat(false);
            let want = vk.prefill(toks, 0).expect("dot kernel");
            ops::set_expert_coopmat(true);
            let got = vk.prefill(toks, 0).expect("coopmat kernel");
            cases += 1;
            if got != want {
                wrong.push(format!(
                    "tile {tile}, {n} tokens: dot {want} {:?}, coopmat {got} {:?}",
                    vocab.text_of(want).unwrap_or("?"),
                    vocab.text_of(got).unwrap_or("?")
                ));
            }
        }
    }
    ops::set_expert_coopmat(false);
    for w in &wrong {
        eprintln!("  MISMATCH {w}");
    }
    eprintln!("  {} of {cases} cases agree", cases - wrong.len());
    assert!(
        wrong.is_empty(),
        "{} of {cases} cases disagree",
        wrong.len()
    );
}

/// And what it is worth, both arms alternating in one process.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn coopmat_experts_against_the_dot_kernel() {
    if !ok() {
        eprintln!("  SKIP coopmat_experts_perf: no coopmat");
        return;
    }
    let Some(path) = oracle_model() else {
        return;
    };
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

    let run = |vk: &mut VulkanModel, cm: bool| -> f64 {
        ops::set_expert_coopmat(cm);
        let _ = vk.prefill(&prompt, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&prompt, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };

    eprintln!(
        "  {:>6}{:>12}{:>12}{:>10}",
        "tile", "coopmat", "dot", "speedup"
    );
    let mut best = (0usize, 0.0f64);
    for tile in [64usize, 128, 256, 512] {
        vk.prefill_tile = tile;
        let dot = run(&mut vk, false);
        let cm = run(&mut vk, true);
        if cm > best.1 {
            best = (tile, cm);
        }
        eprintln!(
            "  {tile:>6}{:>12}{:>12}{:>10}",
            format!("{cm:.1}"),
            format!("{dot:.1}"),
            format!("{:.3}x", cm / dot)
        );
    }
    ops::set_expert_coopmat(false);
    eprintln!(
        "  best coopmat: tile {} at {:.1} tok/s   (llama.cpp Vulkan pp512 3213.6)",
        best.0, best.1
    );
}

/// Where the matvec kernels' time goes, by ablation on the real weights.
///
/// RADV exposes no VK_KHR_performance_query and its SQTT needs AMD's RGP to read, so
/// the counters come from removing one class of work at a time and reading the
/// deltas. Mode 3 is the control that separates waiting from working: if dropping
/// the weight LOADS collapses the time while dropping the dot products does not,
/// the kernel is stalled on memory rather than busy.
#[test]
#[ignore = "uploads a real tensor; needs a free GPU and the real model"]
fn where_the_matvec_time_goes() {
    let Some(path) = oracle_model() else {
        eprintln!("  SKIP where_the_matvec_time_goes: no LLAMD_TEST_MODEL");
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    // Two shapes, because they behave differently and the engine spends most of
    // prefill in the second: the attention output projection that q6k_matvec runs 48
    // times, and ONE EXPERT's gate projection — the shape the expert kernel runs
    // 128 x 3 x 48 times and that four separate optimisations have failed to move.
    // Same arithmetic, same harness, so the difference is the shape.
    for &(name, label) in &[
        ("blk.0.attn_output.weight", "attn_output 4096->2048"),
        ("blk.0.ffn_gate_exps.weight", "ONE expert gate 2048->768"),
    ] {
        run_one(&model, name, label);
    }
}

fn run_one(model: &Model, name: &str, label: &str) {
    let info = model.info(name).expect("info");
    let k = info.dims[0] as usize;
    // For the expert tensor this takes the FIRST expert's slice: the ablation is
    // about the shape and the access pattern, not about routing.
    let rows = if name.contains("exps") {
        768
    } else {
        info.elems() as usize / k
    };
    let blocks = k / 256;
    let raw = model.gguf().tensor_data(info).expect("bytes");
    let w = llamd_vulkan::PlanarWeights::upload(raw, rows, blocks).expect("upload");
    let nseq = 32usize;

    let qb = VkBuf::from_bytes(&vec![0x40u8; nseq * blocks * 256]).expect("qb");
    let sb = VkBuf::from_i32(&vec![16i32; nseq * blocks * 16]).expect("sb");
    let ad = VkBuf::from_f32(&vec![0.01f32; nseq * blocks]).expect("ad");
    let out = VkBuf::new(rows * nseq * 4).expect("out");

    // MACs one dispatch performs: every row, every K, every sequence.
    let macs = rows as f64 * (blocks * 256) as f64 * nseq as f64;
    let reps = 64usize;

    let run = |mode: usize| -> f64 {
        ops::begin().expect("begin");
        for _ in 0..4 {
            ops::matvec_ablate(&w, &qb, &sb, &ad, &out, rows, nseq, mode).expect("ablate");
        }
        ops::end_and_wait().expect("wait");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            ops::begin().expect("begin");
            for _ in 0..reps {
                ops::matvec_ablate(&w, &qb, &sb, &ad, &out, rows, nseq, mode).expect("ablate");
            }
            ops::end_and_wait().expect("wait");
            best = best.min(t.elapsed().as_secs_f64());
        }
        best / reps as f64 * 1e3
    };

    let names = [
        "0 full kernel",
        "1 no dot products",
        "2 no Q6_K unpack",
        "3 no WEIGHT loads",
        "4 no activation loads",
        "5 FULL, vec4 loads",
    ];
    let mut ms = [0.0f64; 6];
    eprintln!();
    eprintln!(
        "  {label}, {nseq} sequences, {:.2} G MAC per dispatch",
        macs / 1e9
    );
    eprintln!(
        "  {:<24}{:>10}{:>12}{:>11}",
        "arm", "ms", "G MAC/s", "time saved"
    );
    for mode in 0..6 {
        ms[mode] = run(mode);
        eprintln!(
            "  {:<24}{:>10}{:>12}{:>11}",
            names[mode],
            format!("{:.3}", ms[mode]),
            format!("{:.0}", macs / (ms[mode] / 1e3) / 1e9),
            if mode == 0 {
                "-".to_string()
            } else {
                format!("{:.0}%", 100.0 * (1.0 - ms[mode] / ms[0]))
            }
        );
    }
    eprintln!();
    eprintln!(
        "  the dot products are   {:>5.0}% of the kernel",
        100.0 * (1.0 - ms[1] / ms[0])
    );
    eprintln!(
        "  the Q6_K unpack is     {:>5.0}%",
        100.0 * (1.0 - ms[2] / ms[0])
    );
    eprintln!(
        "  the weight traffic is  {:>5.0}%",
        100.0 * (1.0 - ms[3] / ms[0])
    );
    eprintln!(
        "  the activation reads   {:>5.0}%",
        100.0 * (1.0 - ms[4] / ms[0])
    );
    eprintln!();
    eprintln!("  VECTORISING THE LOADS: {:.3}x", ms[0] / ms[5]);
}

/// The lane-per-row expert kernel: correctness, then what removing the reduction is
/// worth. §26 measured 57% of the expert kernel as unaccounted for by any work it
/// does, and the geometry says it is the `subgroupAdd` epilogue — K = 2048 leaves each
/// lane one loop iteration to amortise 32 reductions over.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn lane_per_row_experts() {
    let Some(path) = oracle_model() else {
        eprintln!("  SKIP lane_per_row_experts: no LLAMD_TEST_MODEL");
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 512 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(512);

    // Correctness first, across shapes that straddle ACC_N and the row blocking.
    let mut wrong = 0;
    let mut cases = 0;
    for &tile in &[32usize, 128, 512] {
        for &n in &[15usize, 16, 17, 33, 300, 512] {
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_expert_lane(false);
            let want = vk.prefill(toks, 0).expect("subgroup kernel");
            ops::set_expert_lane(true);
            let got = vk.prefill(toks, 0).expect("lane kernel");
            cases += 1;
            if got != want {
                wrong += 1;
                eprintln!("  MISMATCH tile {tile}, {n} tokens: {want} vs {got}");
            }
        }
    }
    ops::set_expert_lane(false);
    eprintln!("  {} of {cases} cases agree", cases - wrong);
    assert_eq!(wrong, 0, "the lane kernel disagrees");

    // Then the number, both arms alternating in one process.
    let run = |vk: &mut VulkanModel, lane: bool| -> f64 {
        ops::set_expert_lane(lane);
        let _ = vk.prefill(&all, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&all, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };
    eprintln!(
        "  {:>6}{:>14}{:>14}{:>10}",
        "tile", "lane-per-row", "subgroup", "speedup"
    );
    let mut best = (0usize, 0.0f64);
    for tile in [128usize, 256, 512] {
        vk.prefill_tile = tile;
        let sub = run(&mut vk, false);
        let lane = run(&mut vk, true);
        if lane > best.1 {
            best = (tile, lane);
        }
        eprintln!(
            "  {tile:>6}{:>14}{:>14}{:>10}",
            format!("{lane:.1}"),
            format!("{sub:.1}"),
            format!("{:.3}x", lane / sub)
        );
    }
    ops::set_expert_lane(false);
    eprintln!(
        "  best lane-per-row: tile {} at {:.1} tok/s   (llama.cpp Vulkan pp512 3213.6)",
        best.0, best.1
    );
}

/// The clustered reduction: 16 lanes per row group instead of 64.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn clustered_reduction_experts() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 512 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(512);

    let mut wrong = 0;
    let mut cases = 0;
    for &tile in &[32usize, 128, 512] {
        for &n in &[15usize, 17, 33, 300, 512] {
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_expert_cluster(0);
            let want = vk.prefill(toks, 0).expect("subgroup");
            ops::set_expert_cluster(16);
            let got = vk.prefill(toks, 0).expect("clustered");
            cases += 1;
            if got != want {
                wrong += 1;
                eprintln!("  MISMATCH tile {tile}, {n} tokens: {want} vs {got}");
            }
        }
    }
    ops::set_expert_cluster(0);
    eprintln!("  {} of {cases} cases agree", cases - wrong);
    assert_eq!(wrong, 0, "the clustered kernel disagrees");

    let run = |vk: &mut VulkanModel, lanes: usize| -> f64 {
        ops::set_expert_cluster(lanes);
        let _ = vk.prefill(&all, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&all, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };
    eprintln!(
        "  {:>6}{:>12}{:>12}{:>12}{:>12}",
        "tile", "subgroup 64", "cluster 32", "cluster 16", "cluster 8"
    );
    let mut best = (0usize, 0usize, 0.0f64);
    for tile in [128usize, 256, 512] {
        vk.prefill_tile = tile;
        let mut row = Vec::new();
        for lanes in [0usize, 32, 16, 8] {
            let t = run(&mut vk, lanes);
            if t > best.2 {
                best = (tile, lanes, t);
            }
            row.push(format!("{t:.1}"));
        }
        eprintln!(
            "  {tile:>6}{:>12}{:>12}{:>12}{:>12}",
            row[0], row[1], row[2], row[3]
        );
    }
    ops::set_expert_cluster(0);
    eprintln!(
        "  best: tile {} cluster {} at {:.1} tok/s   (llama.cpp Vulkan pp512 3213.6)",
        best.0,
        if best.1 == 0 { 64 } else { best.1 },
        best.2
    );
}

/// The clustered reduction applied to the attention projections — §27's fix on the
/// same pathology in the other 26% of the pass.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn clustered_reduction_projections() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 512 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(512);

    let mut wrong = 0;
    let mut cases = 0;
    for &tile in &[32usize, 128, 512] {
        for &n in &[15usize, 17, 33, 300, 512] {
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_matvec_cluster(false);
            let want = vk.prefill(toks, 0).expect("subgroup");
            ops::set_matvec_cluster(true);
            let got = vk.prefill(toks, 0).expect("clustered");
            cases += 1;
            if got != want {
                wrong += 1;
                eprintln!("  MISMATCH tile {tile}, {n} tokens: {want} vs {got}");
            }
        }
    }
    ops::set_matvec_cluster(false);
    eprintln!("  {} of {cases} cases agree", cases - wrong);
    assert_eq!(wrong, 0, "the clustered projections disagree");

    let run = |vk: &mut VulkanModel, on: bool| -> f64 {
        ops::set_matvec_cluster(on);
        let _ = vk.prefill(&all, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&all, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };
    eprintln!(
        "  {:>6}{:>14}{:>14}{:>10}",
        "tile", "clustered", "subgroup 64", "speedup"
    );
    let mut best = 0.0f64;
    for tile in [128usize, 256, 512] {
        vk.prefill_tile = tile;
        let off = run(&mut vk, false);
        let on = run(&mut vk, true);
        best = best.max(on);
        eprintln!(
            "  {tile:>6}{:>14}{:>14}{:>10}",
            format!("{on:.1}"),
            format!("{off:.1}"),
            format!("{:.3}x", on / off)
        );
    }
    ops::set_matvec_cluster(false);
    eprintln!("  best: {best:.1} tok/s   (llama.cpp Vulkan pp512 3213.6, no coopmat 1834.3)");

    // Decode too: q6k_matvec and q6k_x3 run there as well, over a much shorter K.
    let dec = |vk: &mut VulkanModel, on: bool| -> f64 {
        ops::set_matvec_cluster(on);
        let toks = [all[0]; 8];
        let pos = [64usize; 8];
        let _ = vk.forward_batch(&toks, &pos).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for _ in 0..16 {
                let _ = vk.forward_batch(&toks, &pos).expect("decode");
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        (8 * 16) as f64 / best
    };
    let d_off = dec(&mut vk, false);
    let d_on = dec(&mut vk, true);
    ops::set_matvec_cluster(false);
    eprintln!(
        "  decode batch 8: clustered {d_on:.1} vs subgroup {d_off:.1} = {:.3}x",
        d_on / d_off
    );
}

/// ROWS x ACC x vec4, re-swept together.
///
/// §23 chose ROWS 2 / ACC 16 before §26's vectorisation and §27's clustering existed,
/// and all three trade through the same register file. §26 tested vec4 only at that
/// one point and found it 1.5% slower, which says nothing about the other points.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn expert_rows_acc_vec4_sweep() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut prompt = Vec::new();
    while prompt.len() < 512 {
        prompt.extend_from_slice(&vocab.encode(text, false));
    }
    prompt.truncate(512);
    vk.prefill_tile = 512;

    // Every arm must emit this.
    ops::set_expert_cluster(16);
    ops::set_expert_vec4(false);
    ops::set_expert_tile(2, 16);
    let want = vk.prefill(&prompt, 0).expect("reference");

    eprintln!("  512-token prompt, tile 512, cluster 16. accumulators = ROWS * ACC.");
    eprintln!(
        "  {:>5}{:>5}{:>6}{:>12}{:>12}{:>9}",
        "ROWS", "ACC", "regs", "scalar", "vec4", "vec4 win"
    );
    let mut best = (0usize, 0usize, false, 0.0f64);
    for &(rows, acc) in &[
        (1usize, 16usize),
        (1, 32),
        (2, 8),
        (2, 16),
        (2, 32),
        (4, 8),
        (4, 16),
    ] {
        let mut t = [0.0f64; 2];
        for (vi, &v) in [false, true].iter().enumerate() {
            ops::set_expert_vec4(v);
            ops::set_expert_tile(rows, acc);
            let got = vk.prefill(&prompt, 0).expect("prefill");
            assert_eq!(
                got, want,
                "ROWS {rows} ACC {acc} vec4 {v} changed the token"
            );
            let mut bestrun = f64::MAX;
            for _ in 0..2 {
                let s = std::time::Instant::now();
                let _ = vk.prefill(&prompt, 0).expect("prefill");
                bestrun = bestrun.min(s.elapsed().as_secs_f64());
            }
            t[vi] = 512.0 / bestrun;
            if t[vi] > best.3 {
                best = (rows, acc, v, t[vi]);
            }
        }
        eprintln!(
            "  {rows:>5}{acc:>5}{:>6}{:>12}{:>12}{:>9}",
            rows * acc,
            format!("{:.1}", t[0]),
            format!("{:.1}", t[1]),
            format!("{:.3}x", t[1] / t[0])
        );
    }
    ops::set_expert_tile(best.0, best.1);
    ops::set_expert_vec4(best.2);
    eprintln!(
        "  best: ROWS {} ACC {} {} at {:.1} tok/s",
        best.0,
        best.1,
        if best.2 { "vec4" } else { "scalar" },
        best.3
    );
}

/// Hoisting the address arithmetic out of the K loop — §30's measured target.
#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn expert_address_hoist() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 512 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(512);

    let mut wrong = 0;
    let mut cases = 0;
    for &tile in &[32usize, 128, 512] {
        for &n in &[15usize, 17, 33, 300, 512] {
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_expert_addr(0);
            let want = vk.prefill(toks, 0).expect("plain");
            for m in [1usize, 2, 3] {
                ops::set_expert_addr(m);
                let got = vk.prefill(toks, 0).expect("hoisted");
                cases += 1;
                if got != want {
                    wrong += 1;
                    eprintln!("  MISMATCH mode {m} tile {tile}, {n} tokens: {want} vs {got}");
                }
            }
        }
    }
    ops::set_expert_addr(0);
    eprintln!("  {} of {cases} cases agree", cases - wrong);
    assert_eq!(wrong, 0, "the hoisted kernel disagrees");

    let run = |vk: &mut VulkanModel, on: usize| -> f64 {
        ops::set_expert_addr(on);
        let _ = vk.prefill(&all, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(&all, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        512.0 / best
    };
    eprintln!(
        "  {:>6}{:>12}{:>14}{:>14}{:>14}",
        "tile", "plain", "shared+bar", "registers", "PREFETCH"
    );
    let mut best = 0.0f64;
    for tile in [128usize, 256, 512] {
        vk.prefill_tile = tile;
        let a = run(&mut vk, 0);
        let b = run(&mut vk, 1);
        let c = run(&mut vk, 2);
        let d = run(&mut vk, 3);
        best = best.max(a.max(b).max(c.max(d)));
        eprintln!(
            "  {tile:>6}{:>12}{:>14}{:>14}{:>14}",
            format!("{a:.1}"),
            format!("{b:.1} {:.3}x", b / a),
            format!("{c:.1} {:.3}x", c / a),
            format!("{d:.1} {:.3}x", d / a)
        );
    }
    ops::set_expert_addr(0);
    eprintln!("  best hoisted: {best:.1} tok/s   (llama.cpp Vulkan pp512 3213.6)");
}

/// How prefill scales past 512 tokens — the baseline any per-layer dequantisation
/// scheme has to beat, because that scheme's cost is per LAYER and its benefit is per
/// TILE, so it can only amortise over prompts long enough to need several tiles.
#[test]
#[ignore = "uploads 23 GiB and a large KV cache; needs a free GPU and the real model"]
fn prefill_scaling_past_512() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk = VulkanModel::load_with_slots(&model, 2100, llamd_vulkan::MAX_TILE, 8)
        .expect("load with a 2100-token cache");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 2048 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(2048);
    vk.prefill_tile = 512;

    eprintln!(
        "  {:>8}{:>8}{:>12}{:>12}",
        "prompt", "tiles", "seconds", "tok/s"
    );
    for n in [512usize, 1024, 1536, 2048] {
        let toks = &all[..n];
        let _ = vk.prefill(toks, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(toks, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        eprintln!(
            "  {n:>8}{:>8}{:>12}{:>12}",
            n.div_ceil(512),
            format!("{best:.3}"),
            format!("{:.1}", n as f64 / best)
        );
    }
}

/// Flash attention: one pass over the keys with an online softmax, four lanes per key.
///
/// §34 put attention at 21% of a 512-token prefill and 52% of a 2048-token one, running
/// 7.3x slower per MAC than the matmul kernels. The bar is the same emitted token, then
/// the number — measured at BOTH prompt lengths, because the whole point is the long one.
#[test]
#[ignore = "uploads 23 GiB and a large KV cache; needs a free GPU and the real model"]
fn flash_attention() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 2100, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 2048 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(2048);

    // Correctness first, over lengths that straddle the 64-key block and the tile.
    let mut wrong = 0;
    let mut cases = 0;
    for &tile in &[32usize, 128, 512] {
        for &n in &[1usize, 63, 64, 65, 129, 300, 512, 1024] {
            vk.prefill_tile = tile;
            let toks = &all[..n];
            ops::set_attention_flash(0);
            let want = vk.prefill(toks, 0).expect("three-pass");
            for m in [1usize, 2] {
                ops::set_attention_flash(m);
                let got = vk.prefill(toks, 0).expect("flash");
                cases += 1;
                if got != want {
                    wrong += 1;
                    if wrong <= 6 {
                        eprintln!("  MISMATCH mode {m} tile {tile}, {n} tokens: {want} vs {got}");
                    }
                }
            }
        }
    }
    ops::set_attention_flash(0);
    eprintln!("  {} of {cases} cases agree", cases - wrong);
    assert_eq!(
        wrong, 0,
        "flash attention disagrees in {wrong} of {cases} cases"
    );

    // Then the number, alternating arms in one process.
    let run = |vk: &mut VulkanModel, n: usize, on: usize| -> f64 {
        ops::set_attention_flash(on);
        let toks = &all[..n];
        let _ = vk.prefill(toks, 0).expect("warm");
        let mut best = f64::MAX;
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let _ = vk.prefill(toks, 0).expect("prefill");
            best = best.min(t.elapsed().as_secs_f64());
        }
        n as f64 / best
    };
    vk.prefill_tile = 512;
    eprintln!(
        "  {:>8}{:>14}{:>14}{:>10}",
        "prompt", "flash", "three-pass", "speedup"
    );
    for n in [512usize, 1024, 1536, 2048] {
        let old = run(&mut vk, n, 1);
        let new = run(&mut vk, n, 2);
        eprintln!(
            "  {n:>8}{:>14}{:>14}{:>10}",
            format!("{new:.1}"),
            format!("{old:.1}"),
            format!("{:.3}x", new / old)
        );
    }
    ops::set_attention_flash(0);
}

/// Flash attention at DECODE, where n_kv is the context and attention is a small share —
/// so the bar here is "no regression", not a win.
#[test]
#[ignore = "uploads 23 GiB; needs a free GPU and the real model"]
fn flash_attention_decode() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let mut vk =
        VulkanModel::load_with_slots(&model, 600, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let tok = vocab.encode("The", false)[0];
    let run = |vk: &mut VulkanModel, batch: usize, on: usize| -> f64 {
        ops::set_attention_flash(on);
        let toks: Vec<u32> = std::iter::repeat_n(tok, batch).collect();
        let pos: Vec<usize> = (0..batch).map(|_| 256).collect();
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
    eprintln!("  256 tokens of context");
    eprintln!(
        "  {:>7}{:>12}{:>14}{:>10}",
        "batch", "flash", "three-pass", "ratio"
    );
    for b in [1usize, 4, 8] {
        let off = run(&mut vk, b, 1);
        let on = run(&mut vk, b, 2);
        eprintln!(
            "  {b:>7}{:>12}{:>14}{:>10}",
            format!("{on:.1}"),
            format!("{off:.1}"),
            format!("{:.3}x", on / off)
        );
    }
    ops::set_attention_flash(0);
}

/// The merge, checked on the LOGITS — per kernel, per length, with real metrics.
///
/// Token equality is not sufficient and this test exists because it was not: the
/// four-stream kernel passed 48 of 48 token-equality cases while deviating by 8% of the
/// logit range. If the reference says 12.81347 / 11.92731 and we say 12.81291 / 11.92802,
/// the same token wins and numerical error has still been introduced.
///
/// Recorded against the three-pass kernel, which is the long-standing reference that
/// agrees with llama.cpp token for token:
///   E_abs  = max |L_i - R_i|
///   MAE    = mean |L_i - R_i|
///   E_rel  = max |L_i - R_i| / max(|R_i|, eps)
///   top-10 and top-20 set agreement
///
/// The lengths are every structural transition, because that is where ownership of blocks
/// between the four streams changes: n_kv 1..3 leaves three streams EMPTY, 63/64/65 is the
/// first full round, and 127/128/129, 255/256/257, 511/512/513, 1023/1024/1025,
/// 2047/2048/2049 are the later ones.
#[test]
#[ignore = "uploads 23 GiB and a 2100-token cache; needs a free GPU and the real model"]
fn flash_merge_logits_agree() {
    let Some(path) = oracle_model() else {
        return;
    };
    if !llamd_vulkan::available() {
        return;
    }
    let model = Model::open(&path).expect("open");
    let nv = model.hparams.n_vocab;
    let mut vk =
        VulkanModel::load_with_slots(&model, 2100, llamd_vulkan::MAX_TILE, 8).expect("load");
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    let text = "Explain how a compiler turns source code into machine instructions, \
                covering lexing, parsing, type checking, optimisation and code generation.";
    let mut all = Vec::new();
    while all.len() < 2060 {
        all.extend_from_slice(&vocab.encode(text, false));
    }
    all.truncate(2060);

    fn topk(v: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        idx.truncate(k);
        idx
    }
    let overlap =
        |a: &[usize], b: &[usize]| -> usize { a.iter().filter(|i| b.contains(i)).count() };

    const EPS: f32 = 1e-3;
    // Every structural transition, because that is where block ownership between the four
    // streams changes.
    let lengths = [
        1usize, 2, 3, 4, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
        2047, 2048, 2049,
    ];
    eprintln!(
        "  {:>6}{:>6}{:>12}{:>12}{:>12}{:>9}{:>9}{:>12}",
        "n_kv", "mode", "E_abs", "MAE", "E_rel", "top10", "top20", "vs mode1"
    );
    eprintln!("  (E_* are against mode 0; the last column is against mode 1)");
    let mut worst: Vec<(f32, usize, usize)> = Vec::new();
    for &n in &lengths {
        if n > all.len() {
            continue;
        }
        vk.prefill_tile = 512;
        let toks = &all[..n];
        // §38: prefill now runs the output projection for the FINAL TOKEN ONLY, so the
        // buffer holds exactly one token's logits and it is the one this test wants.
        let last_tile = 1usize;
        let tail = |v: Vec<f32>| v[..nv].to_vec();
        // The reference: the vec4 three-pass kernel, with the scalar arm explicitly OFF.
        // Leaving it on is how the first version of this test accidentally produced the
        // control below, and the leak is worth naming because the result is load-bearing.
        ops::set_attention_scalar(false);
        ops::set_attention_flash(0);
        let _ = vk.prefill(toks, 0).expect("three-pass");
        let r = tail(vk.logits(last_tile).expect("logits"));
        let r10 = topk(&r, 10);
        let r20 = topk(&r, 20);
        // Mode 0 against itself is the determinism control; it comes out exactly zero,
        // so any deviation below is real. Mode 1 against mode 2 is the OTHER control: two
        // independently written flash kernels. If they agree with each other far more
        // closely than either agrees with mode 0, the difference is the softmax SUMMATION
        // ORDER amplified through 48 layers rather than a bug in either.
        ops::set_attention_flash(1);
        let _ = vk.prefill(toks, 0).expect("one-pass");
        let ref1 = tail(vk.logits(last_tile).expect("logits"));
        // THE CONTROL THAT DECIDES THE METHOD. Mode 3 here is not a flash kernel at all:
        // it is the PRE-§22 scalar three-pass attention, kept as an A/B arm and validated
        // against llama.cpp for a dozen sections. It differs from mode 0 only in summation
        // order (scalar loads versus vec4). If IT also moves the logits by ~1, then an
        // end-to-end logit comparison cannot distinguish a correct kernel from a buggy one
        // in this model, and §36's verdict was an artefact of the method rather than a bug.
        for m in [0usize, 1, 2, 3] {
            if m == 3 {
                ops::set_attention_flash(0);
                ops::set_attention_scalar(true);
            } else {
                ops::set_attention_scalar(false);
                ops::set_attention_flash(m);
            }
            let _ = vk.prefill(toks, 0).expect("flash");
            let g = tail(vk.logits(last_tile).expect("logits"));
            let mut eabs = 0.0f32;
            let mut sum = 0.0f64;
            let mut erel = 0.0f32;
            for (a, b) in r.iter().zip(&g) {
                let d = (a - b).abs();
                eabs = eabs.max(d);
                sum += d as f64;
                erel = erel.max(d / a.abs().max(EPS));
            }
            let mae = (sum / r.len() as f64) as f32;
            let t10 = overlap(&r10, &topk(&g, 10));
            let t20 = overlap(&r20, &topk(&g, 20));
            // and how far this kernel is from the OTHER flash kernel
            let v1 = ref1
                .iter()
                .zip(&g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "  {n:>6}{m:>6}{:>12}{:>12}{:>12}{:>9}{:>9}{:>12}",
                format!("{eabs:.3e}"),
                format!("{mae:.3e}"),
                format!("{erel:.3e}"),
                format!("{t10}/10"),
                format!("{t20}/20"),
                format!("{v1:.3e}")
            );
            worst.push((eabs, n, m));
        }
    }
    ops::set_attention_scalar(false);
    ops::set_attention_flash(0);
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!(
        "  worst E_abs: {:.3e} at n_kv {} mode {}",
        worst[0].0, worst[0].1, worst[0].2
    );
}
