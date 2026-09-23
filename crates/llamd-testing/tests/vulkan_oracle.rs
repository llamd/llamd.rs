//! The Vulkan backend, validated against the same reference as ROCm.
//!
//! WHY THIS TEST IS THE WHOLE POINT OF A SECOND BACKEND. A Vulkan-vs-ROCm
//! comparison on identical hardware is only evidence if both backends compute
//! the same thing. `shaders/q6k_matvec.comp` and `kernels/llamd_udot.hip`
//! implement the same identity over the same planar layout, and both are held
//! here to `llamd_quant::dot_q6k_q8k`.
//!
//! The bar is `REDUCTION`, scaled by accumulation magnitude. The integer part of
//! the identity is exact on both backends; what differs is the order of the
//! float additions, and on Vulkan also the subgroup width — RADV gives 64 where
//! the HIP kernels use 32, so the reduction tree has a different shape. That is
//! reduction error, not disagreement.

use llamd_core::tolerance::REDUCTION;
use llamd_quant::{QK_K, dot_q6k_q8k, quantize_q8_k};

const B: usize = 210;

fn weights(rows: usize, blocks: usize, seed: u32) -> Vec<u8> {
    let mut w = vec![0u8; rows * blocks * B];
    let mut st = seed.wrapping_mul(2246822519).wrapping_add(7);
    for b in w.iter_mut() {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (st >> 16) as u8;
    }
    for blk in 0..rows * blocks {
        let o = blk * B + QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = half::f16::from_f32(0.008 + (blk % 17) as f32 * 0.002);
        w[o..o + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    w
}

/// Bias an activation to unsigned and sum each group of 16 — the two correction
/// terms the packed unsigned dot product needs. Done on the host here; in the
/// engine a shader does it.
fn bias_and_sums(acts: &[llamd_quant::BlockQ8K]) -> (Vec<u8>, Vec<i32>, Vec<f32>) {
    let mut qb = Vec::with_capacity(acts.len() * QK_K);
    let mut sb = Vec::with_capacity(acts.len() * 16);
    let mut ad = Vec::with_capacity(acts.len());
    for a in acts {
        for &q in &a.qs {
            qb.push((q as i32 + 128) as u8);
        }
        for g in 0..16 {
            sb.push((0..16).map(|j| a.qs[g * 16 + j] as i32 + 128).sum::<i32>());
        }
        ad.push(a.d);
    }
    (qb, sb, ad)
}

fn skip(what: &str) -> bool {
    if !llamd_vulkan::available() {
        eprintln!("  skipping {what}: no Vulkan device");
        return true;
    }
    false
}

#[test]
fn reports_the_device_or_says_why_not() {
    if !llamd_vulkan::built_with_vulkan() {
        eprintln!("  skipping: built without Vulkan");
        return;
    }
    if skip("reports_the_device") {
        return;
    }
    let info = llamd_vulkan::device_info().expect("device info");
    eprintln!("  {info}");
    // A software device would make every measurement meaningless while
    // reporting success, so the backend picks a discrete GPU and this checks it.
    assert!(
        !info.name.to_lowercase().contains("llvmpipe"),
        "picked the software device: {}",
        info.name
    );
    assert!(
        info.heap_bytes > (8u64 << 30),
        "implausibly small device-local heap"
    );
    assert!(
        info.subgroup_size == 32 || info.subgroup_size == 64,
        "unexpected subgroup size {}",
        info.subgroup_size
    );
}

#[test]
fn vulkan_matvec_agrees_with_the_scalar_reference() {
    if skip("vulkan_matvec_agrees_with_the_scalar_reference") {
        return;
    }
    // Every block count the model uses — 3 (expert down), 8 (most), 16
    // (attn_out) — and every batch width, since the shader loops over nseq.
    for (rows, k, nseq) in [
        (256usize, 768usize, 1usize),
        (1024, 2048, 1),
        (512, 4096, 1),
        (1024, 2048, 4),
        (256, 2048, 8),
        (7, 2048, 1), // a row count that does not fill a workgroup
    ] {
        let blocks = k / QK_K;
        let w = weights(rows, blocks, 13 + rows as u32 + nseq as u32);
        let pw = llamd_vulkan::PlanarWeights::upload(&w, rows, blocks).expect("upload");

        // A different activation per sequence: a shader that used sequence 0's
        // activation for all of them would pass an identical-input test.
        let mut acts = Vec::new();
        let mut per_seq = Vec::new();
        for s in 0..nseq {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as f32) * 0.017 + s as f32 * 1.7).sin())
                .collect();
            let a = quantize_q8_k(&x);
            per_seq.push(a.clone());
            acts.extend(a);
        }
        let (qb, sb, ad) = bias_and_sums(&acts);

        let d_qb = llamd_vulkan::VkBuf::from_bytes(&qb).expect("qb");
        let d_sb = llamd_vulkan::VkBuf::from_i32(&sb).expect("sb");
        let d_ad = llamd_vulkan::VkBuf::from_f32(&ad).expect("ad");
        let d_out = llamd_vulkan::VkBuf::new(rows * nseq * 4).expect("out");
        pw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
            .expect("dispatch");

        let mut got = vec![0.0f32; rows * nseq];
        d_out.download_f32(&mut got).expect("download");

        let mut worst = 0.0f32;
        let mut acc = 0.0f64;
        let mut n = 0usize;
        for (s, a) in per_seq.iter().enumerate() {
            for r in 0..rows {
                let want = dot_q6k_q8k(&w[r * blocks * B..(r + 1) * blocks * B], a, k);
                worst = worst.max((got[s * rows + r] - want).abs());
                acc += (want as f64) * (want as f64);
                n += 1;
            }
        }
        let rms = (acc / n as f64).sqrt() as f32;
        let bound = REDUCTION * rms.max(1.0);
        assert!(
            worst <= bound,
            "{rows}x{k} nseq {nseq}: worst deviation {worst:.3e} exceeds {bound:.3e} (rms {rms:.3e})"
        );
        eprintln!(
            "  {rows:>5} rows, {blocks:>2} blocks, nseq {nseq}: worst {worst:.3e}, bound {bound:.3e}"
        );
    }
}

#[test]
fn vulkan_and_rocm_agree_with_each_other() {
    // The comparison the second backend exists to make. Both read the identical
    // planar bytes and implement the identical identity, so they should land
    // within reduction error of each other as well as of the reference.
    if skip("vulkan_and_rocm_agree_with_each_other") {
        return;
    }
    if !llamd_rocm::available() {
        eprintln!("  skipping: no ROCm device to compare against");
        return;
    }
    let (rows, k, nseq) = (1024usize, 2048usize, 4usize);
    let blocks = k / QK_K;
    let w = weights(rows, blocks, 4242);

    let mut acts = Vec::new();
    for s in 0..nseq {
        let x: Vec<f32> = (0..k)
            .map(|i| ((i as f32) * 0.021 + s as f32 * 0.9).cos())
            .collect();
        acts.extend(quantize_q8_k(&x));
    }
    let (qb, sb, ad) = bias_and_sums(&acts);

    let vw = llamd_vulkan::PlanarWeights::upload(&w, rows, blocks).expect("vk upload");
    let d_qb = llamd_vulkan::VkBuf::from_bytes(&qb).expect("qb");
    let d_sb = llamd_vulkan::VkBuf::from_i32(&sb).expect("sb");
    let d_ad = llamd_vulkan::VkBuf::from_f32(&ad).expect("ad");
    let d_out = llamd_vulkan::VkBuf::new(rows * nseq * 4).expect("out");
    vw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
        .expect("vk dispatch");
    let mut vk = vec![0.0f32; rows * nseq];
    d_out.download_f32(&mut vk).expect("download");

    let rw = llamd_rocm::PlanarWeights::upload(&w, rows, blocks).expect("rocm upload");
    let rocm = rw.matvec_udot(&acts, nseq).expect("rocm udot");

    let rms = (rocm.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / rocm.len() as f64)
        .sqrt() as f32;
    let (idx, worst) = vk
        .iter()
        .zip(&rocm)
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("non-empty");
    let bound = REDUCTION * rms.max(1.0);
    assert!(
        worst <= bound,
        "Vulkan and ROCm differ by {worst:.3e} at {idx} (bound {bound:.3e}); vk {:e} rocm {:e}",
        vk[idx],
        rocm[idx]
    );
    eprintln!(
        "  Vulkan vs ROCm: worst {worst:.3e}, bound {bound:.3e} over {} values",
        vk.len()
    );
}

#[test]
fn vulkan_against_rocm_on_identical_hardware() {
    // THE COMPARISON THE SECOND BACKEND EXISTS TO MAKE. Same card, same model
    // shapes, same planar bytes, same identity, same timing discipline: upload
    // once, dispatch `iters` times with nothing between, one wait at the end.
    //
    // A caveat that applies to BOTH columns and is easy to forget: this card has
    // 64 MB of Infinity Cache, so a repeated dispatch over a small tensor
    // measures cache bandwidth. That mistake cost this project two failed
    // optimisations (benchmarks.md §9-§10). The shapes below are labelled with
    // their size so the reader can tell which numbers are cache-assisted, and
    // the engine-level verdict belongs to a whole-token measurement, not here.
    if skip("vulkan_against_rocm_on_identical_hardware") {
        return;
    }
    if !llamd_rocm::available() {
        eprintln!("  skipping: no ROCm device to compare against");
        return;
    }
    eprintln!(
        "\n  {:<26} {:>8} {:>5} {:>11} {:>11} {:>9}",
        "shape", "MiB", "nseq", "vulkan us", "rocm us", "vk/rocm"
    );
    eprintln!("  {}", "-".repeat(76));

    let shapes: &[(&str, usize, usize, usize)] = &[
        ("attn_q      [2048->4096]", 4096, 2048, 1),
        ("attn_q      [2048->4096]", 4096, 2048, 8),
        ("attn_out    [4096->2048]", 2048, 4096, 8),
        ("expert_down [768->2048]x8", 2048 * 8, 768, 1),
        ("output      [2048->151936]", 151936, 2048, 1),
        ("output      [2048->151936]", 151936, 2048, 8),
    ];
    for &(name, rows, k, nseq) in shapes {
        let blocks = k / QK_K;
        let w = weights(rows, blocks, 31);
        let bytes = (rows * blocks * B) as f64;

        let mut acts = Vec::new();
        for s in 0..nseq {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as f32) * 0.019 + s as f32 * 0.5).sin())
                .collect();
            acts.extend(quantize_q8_k(&x));
        }
        let (qb, sb, ad) = bias_and_sums(&acts);
        let iters = if bytes > 1e8 { 50u32 } else { 500u32 };

        let vw = llamd_vulkan::PlanarWeights::upload(&w, rows, blocks).expect("vk upload");
        let d_qb = llamd_vulkan::VkBuf::from_bytes(&qb).expect("qb");
        let d_sb = llamd_vulkan::VkBuf::from_i32(&sb).expect("sb");
        let d_ad = llamd_vulkan::VkBuf::from_f32(&ad).expect("ad");
        let d_out = llamd_vulkan::VkBuf::new(rows * nseq * 4).expect("out");
        // Warm: first dispatch pays pipeline and page-in costs.
        vw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
            .expect("warm");
        let t = std::time::Instant::now();
        vw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, iters as usize)
            .expect("vk bench");
        let vk_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let rw = llamd_rocm::PlanarWeights::upload(&w, rows, blocks).expect("rocm upload");
        let rocm_ms = rw.bench_udot(&acts, nseq, iters).expect("rocm bench");
        let rocm_us = rocm_ms as f64 * 1e3 / iters as f64;

        eprintln!(
            "  {name:<26} {:>8.1} {nseq:>5} {vk_us:>11.1} {rocm_us:>11.1} {:>8.2}x",
            bytes / (1u64 << 20) as f64,
            rocm_us / vk_us
        );
    }
    eprintln!();
}

#[test]
fn the_tiled_gemm_agrees_with_the_scalar_reference() {
    // Same arithmetic as the matvec, different work decomposition: a lane owns
    // (row, group) instead of (row), and the 8 lanes sharing a row combine with
    // subgroupClusteredAdd. That reshuffles which lane accumulates what, so the
    // float reduction order changes and the bar is REDUCTION — but a wrong
    // cluster boundary or row mapping is an outright error, not a tolerance
    // question, and would fail this badly.
    if skip("the_tiled_gemm_agrees_with_the_scalar_reference") {
        return;
    }
    for (rows, k, nseq) in [
        (256usize, 768usize, 1usize), // 3 blocks, one token
        (1024, 2048, 1),
        (512, 4096, 1), // 16 blocks
        (1024, 2048, 8),
        (256, 2048, 32), // the widest tile
        (2048, 2048, 16),
        (37, 2048, 4), // rows not a multiple of 32: the tail
        (7, 2048, 1),  // fewer rows than one workgroup covers
    ] {
        let blocks = k / QK_K;
        let w = weights(rows, blocks, 71 + rows as u32 + nseq as u32);
        let pw = llamd_vulkan::PlanarWeights::upload(&w, rows, blocks).expect("upload");

        let mut acts = Vec::new();
        let mut per_seq = Vec::new();
        for s in 0..nseq {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as f32) * 0.013 + s as f32 * 0.77).sin())
                .collect();
            let a = quantize_q8_k(&x);
            per_seq.push(a.clone());
            acts.extend(a);
        }
        let (qb, sb, ad) = bias_and_sums(&acts);
        let d_qb = llamd_vulkan::VkBuf::from_bytes(&qb).expect("qb");
        let d_sb = llamd_vulkan::VkBuf::from_i32(&sb).expect("sb");
        let d_ad = llamd_vulkan::VkBuf::from_f32(&ad).expect("ad");
        let d_out = llamd_vulkan::VkBuf::new(rows * nseq * 4).expect("out");
        pw.matvec_gemm_host(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
            .expect("gemm");
        let mut got = vec![0.0f32; rows * nseq];
        d_out.download_f32(&mut got).expect("download");

        let mut worst = 0.0f32;
        let mut acc = 0.0f64;
        let mut n = 0usize;
        for (s, a) in per_seq.iter().enumerate() {
            for r in 0..rows {
                let want = dot_q6k_q8k(&w[r * blocks * B..(r + 1) * blocks * B], a, k);
                worst = worst.max((got[s * rows + r] - want).abs());
                acc += (want as f64) * (want as f64);
                n += 1;
            }
        }
        let rms = (acc / n as f64).sqrt() as f32;
        let bound = REDUCTION * rms.max(1.0);
        assert!(
            worst <= bound,
            "{rows}x{k} nseq {nseq}: worst {worst:.3e} exceeds {bound:.3e} (rms {rms:.3e})"
        );
        eprintln!(
            "  {rows:>5} rows, {blocks:>2} blocks, nseq {nseq:>2}: worst {worst:.3e}, bound {bound:.3e}"
        );
    }
}

#[test]
fn tiled_gemm_against_the_matvec_by_token_count() {
    // The question the GEMM was written to answer: at how many tokens does
    // staging the activation in shared memory and covering 32 rows per workgroup
    // beat one subgroup per row?
    //
    // Kernel-level, so cache-assisted — this card has 64 MB of Infinity Cache and
    // §9-§10 record two optimisations that looked good here and were rejected by
    // the engine. Treated as a direction to look, not a verdict; the engine A/B
    // in vulkan_prefill.rs decides.
    if skip("tiled_gemm_against_the_matvec_by_token_count") {
        return;
    }
    let (rows, k) = (4096usize, 2048usize); // attn_q's shape
    let blocks = k / QK_K;
    let w = weights(rows, blocks, 5);
    let pw = llamd_vulkan::PlanarWeights::upload(&w, rows, blocks).expect("upload");

    eprintln!("\n  attn_q [2048->4096], kernel only");
    eprintln!(
        "  {:>6} {:>12} {:>12} {:>10}",
        "tokens", "matvec us", "gemm us", "gemm/mv"
    );
    for nseq in [1usize, 2, 4, 8, 16, 32] {
        let mut acts = Vec::new();
        for s in 0..nseq {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i as f32) * 0.011 + s as f32).sin())
                .collect();
            acts.extend(quantize_q8_k(&x));
        }
        let (qb, sb, ad) = bias_and_sums(&acts);
        let d_qb = llamd_vulkan::VkBuf::from_bytes(&qb).expect("qb");
        let d_sb = llamd_vulkan::VkBuf::from_i32(&sb).expect("sb");
        let d_ad = llamd_vulkan::VkBuf::from_f32(&ad).expect("ad");
        let d_out = llamd_vulkan::VkBuf::new(rows * nseq * 4).expect("out");
        let iters = 200usize;

        pw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
            .expect("warm mv");
        let t = std::time::Instant::now();
        pw.matvec(&d_qb, &d_sb, &d_ad, &d_out, nseq, iters)
            .expect("mv");
        let mv = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        pw.matvec_gemm_host(&d_qb, &d_sb, &d_ad, &d_out, nseq, 1)
            .expect("warm gemm");
        let t = std::time::Instant::now();
        pw.matvec_gemm_host(&d_qb, &d_sb, &d_ad, &d_out, nseq, iters)
            .expect("gemm");
        let gm = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

        eprintln!("  {nseq:>6} {mv:>12.1} {gm:>12.1} {:>9.2}x", gm / mv);
    }
    eprintln!("  (< 1.00x means the GEMM is faster)");
}
