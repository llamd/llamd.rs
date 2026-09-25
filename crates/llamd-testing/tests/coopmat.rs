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
