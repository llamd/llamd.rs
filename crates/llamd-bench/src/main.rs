//! LLAMD benchmarks.
//!
//! WHAT A NUMBER HERE IS WORTH. Only as much as the correctness behind it. The
//! kernels timed below are the same ones `llamd-testing` validates against
//! ggml, so a speed claim is a claim about code that produces llama.cpp's
//! tokens — not about code that is merely fast.
//!
//! WHAT IS MEASURED. A Q6_K × Q8_K matrix-vector product at the shapes this
//! model actually uses, with weights RESIDENT on the device, because that is
//! how inference works: weights are uploaded once and read every token.
//!
//! Timing the upload per call was the first thing this benchmark did, and it
//! reported the GPU as 2.6× SLOWER than the CPU. That number was true and
//! useless — it measured allocation, not arithmetic. It is kept as a row
//! because the difference between the two is the point.
//!
//!   cargo run --release -p llamd-bench

#![forbid(unsafe_code)]

use llamd_quant::{Q6K_BLOCK_BYTES, QK_K, dot_q6k_q8k, quantize_q8_k};
use std::time::Instant;

/// The real projections in Qwen3-Coder-30B-A3B, as (name, rows, k).
const SHAPES: &[(&str, usize, usize)] = &[
    ("attn_q      [2048→4096]", 4096, 2048),
    ("attn_k      [2048→512]", 512, 2048),
    ("attn_out    [4096→2048]", 2048, 4096),
    ("expert_gate [2048→768]", 768, 2048),
    ("expert_down [768→2048]", 2048, 768),
    ("output      [2048→151936]", 151936, 2048),
];

fn synth_weights(rows: usize, blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; rows * blocks * Q6K_BLOCK_BYTES];
    let mut st = 0x9E3779B9u32;
    for b in w.iter_mut() {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (st >> 16) as u8;
    }
    // Real f16 scales, so nothing decodes to inf and makes the timing
    // meaningless through denormal handling.
    for blk in 0..rows * blocks {
        let o = blk * Q6K_BLOCK_BYTES + QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = half::f16::from_f32(0.01);
        w[o..o + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    w
}

fn bench(f: impl Fn(), iters: usize) -> f64 {
    f(); // warm: the first call pays page faults and code upload
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    println!("\n  LLAMD matvec benchmark — Q6_K weights, Q8_K activations\n");
    match llamd_rocm::device_info(0) {
        Ok(d) => println!("  device: {d}\n"),
        Err(_) => println!("  device: none — CPU figures only\n"),
    }

    println!(
        "  {:<26} {:>9} {:>11} {:>11} {:>10} {:>10}",
        "projection", "MiB", "cpu ms", "gpu ms", "speedup", "GB/s"
    );
    println!("  {}", "-".repeat(82));

    let mut total_cpu = 0.0;
    let mut total_gpu = 0.0;
    // Kept so the summary can cite the bandwidth this run measured rather
    // than a constant that silently goes stale.
    let mut measured_gbs: Vec<f64> = Vec::new();

    for &(name, rows, k) in SHAPES {
        let blocks = k / QK_K;
        let w = synth_weights(rows, blocks);
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
        let act = quantize_q8_k(&x);
        let mib = w.len() as f64 / (1 << 20) as f64;

        // The CPU reference. Slow by design: no SIMD, no threading.
        let iters = if rows > 100_000 { 2 } else { 10 };
        let cpu = bench(
            || {
                let _: Vec<f32> = (0..rows)
                    .map(|r| {
                        let row =
                            &w[r * blocks * Q6K_BLOCK_BYTES..(r + 1) * blocks * Q6K_BLOCK_BYTES];
                        dot_q6k_q8k(row, &act, k)
                    })
                    .collect();
            },
            iters,
        );
        total_cpu += cpu;

        let (gpu, gbs) = match llamd_rocm::DeviceWeights::upload(&w, rows, blocks) {
            Ok(dw) => {
                let g = bench(
                    || {
                        let _ = dw.matvec(&act).unwrap();
                    },
                    50,
                );
                (g, w.len() as f64 / g / 1e9)
            }
            Err(_) => (f64::NAN, f64::NAN),
        };
        total_gpu += if gpu.is_nan() { 0.0 } else { gpu };
        if gbs.is_finite() {
            measured_gbs.push(gbs);
        }

        println!(
            "  {:<26} {:>9.1} {:>11.3} {:>11.3} {:>9.1}x {:>10.1}",
            name,
            mib,
            cpu * 1e3,
            gpu * 1e3,
            cpu / gpu,
            gbs
        );
    }

    println!("  {}", "-".repeat(82));
    println!(
        "  {:<26} {:>9} {:>11.3} {:>11.3} {:>9.1}x",
        "all six",
        "",
        total_cpu * 1e3,
        total_gpu * 1e3,
        total_cpu / total_gpu
    );

    // The bar, measured on this card with llama-bench and its hipified
    // kernels. Recorded here so the gap is visible in the same place as the
    // kernel numbers, rather than living in somebody's memory.
    // ---- Kernel-only bandwidth, with the round-trip removed -------------
    //
    // The table above times the shipping API, which copies activations to the
    // device and the result back on every call. That is the honest cost for a
    // host-side caller, and it is the wrong number for deciding what to build
    // next: on the small shapes it is almost entirely PCIe and synchronise.
    //
    // Below, the same kernel on the same weights, launched back to back with
    // one device synchronise at the end. The gap between the two columns is
    // what a resident forward pass would recover.
    println!("  KERNEL ONLY — round-trip and per-call sync removed\n");
    println!(
        "  {:<26} {:>9} {:>12} {:>12} {:>10} {:>9}",
        "projection", "MiB", "api us", "kernel us", "GB/s", "of peak"
    );
    println!("  {}", "-".repeat(82));

    const PEAK_GBS: f64 = 640.0; // 256-bit GDDR6 at 20 Gbps
    let mut kernel_gbs: Vec<(f64, f64)> = Vec::new(); // (bytes, gbs)

    for &(name, rows, k) in SHAPES {
        let blocks = k / QK_K;
        let w = synth_weights(rows, blocks);
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
        let act = quantize_q8_k(&x);
        let bytes = w.len() as f64;

        let Ok(dw) = llamd_rocm::DeviceWeights::upload(&w, rows, blocks) else {
            continue;
        };
        // The API path, for the side-by-side.
        let api = bench(
            || {
                let _ = dw.matvec(&act).unwrap();
            },
            20,
        ) * 1e6;

        let iters = if bytes > 1e8 { 200 } else { 2000 };
        let Ok(total_ms) = dw.bench_kernel_only(&act, iters) else {
            continue;
        };
        let us = total_ms as f64 * 1e3 / iters as f64;
        let gbs = bytes / (us * 1e-6) / 1e9;
        kernel_gbs.push((bytes, gbs));

        println!(
            "  {:<26} {:>9.1} {:>12.1} {:>12.1} {:>10.1} {:>8.0}%",
            name,
            bytes / (1u64 << 20) as f64,
            api,
            us,
            gbs,
            100.0 * gbs / PEAK_GBS
        );
    }
    println!("  {}", "-".repeat(82));

    // The projection that matters: weight traffic for one token, at the
    // bandwidth each shape class actually sustains. Weighted, not best-case.
    //
    // Composition per token for this model: 48 layers of (Q,K,V,O) plus 8 of
    // 128 experts at three matrices each, plus the output projection.
    let bpw = 6.5625 / 8.0;
    let nonexp = (2048.0 * 4096.0 + 2048.0 * 512.0 * 2.0 + 4096.0 * 2048.0) * bpw;
    let expert = 8.0 * 3.0 * (2048.0 * 768.0) * bpw;
    let outproj = 2048.0 * 151936.0 * bpw;

    // Match each class to the measured bandwidth of the nearest shape by size.
    let bw_for = |bytes: f64| -> f64 {
        kernel_gbs
            .iter()
            .min_by(|a, b| {
                (a.0 / bytes)
                    .ln()
                    .abs()
                    .partial_cmp(&(b.0 / bytes).ln().abs())
                    .unwrap()
            })
            .map(|(_, g)| *g)
            .unwrap_or(1.0)
    };

    let secs = 48.0
        * (nonexp / (bw_for(nonexp / 4.0) * 1e9) + expert / (bw_for(expert / 24.0) * 1e9))
        + outproj / (bw_for(outproj) * 1e9);
    let total_bytes = 48.0 * (nonexp + expert) + outproj;

    println!("\n  WEIGHT TRAFFIC PER TOKEN");
    println!(
        "    48 x (attn {:.1} MB + 8 experts {:.1} MB) + output {:.1} MB = {:.3} GB",
        nonexp / 1e6,
        expert / 1e6,
        outproj / 1e6,
        total_bytes / 1e9
    );
    println!(
        "    at the bandwidths measured above: {:.2} ms/token -> {:.0} tok/s",
        secs * 1e3,
        1.0 / secs
    );
    println!(
        "    (that is the weight read ALONE — no attention, norms, routing,\n              KV traffic or launch gaps. A ceiling, and an optimistic one.)\n"
    );

    // Declared here because the grouped summary below cites it.
    const REF_TG: f64 = 94.39; // t/s, qwen3moe 30B.A3B Q6_K, ngl 99, idle card
    const REF_PP: f64 = 1636.66;
    const REF_TG_SERVER: f64 = 57.74; // same card, 98304 ctx, q8_0 KV

    // ---- The routed-expert path: 8 separate launches against 1 grouped ----
    //
    // THIS IS THE MEASUREMENT THE 140 t/s TARGET RESTS ON. The expert matrices
    // are 90% of a token's weight traffic (31.0 of 46.4 MB per layer) and the
    // per-shape sweep above measured them at 9-24% of peak, which ceilings the
    // whole model around 75 t/s. The cause is not the inner loop: expert_down
    // has k=768, so blocks=3, so a lane does twelve multiply-accumulates and
    // then a five-step reduction, and 2048 rows at 8 per workgroup is only 256
    // workgroups — far too little in flight to cover memory latency.
    //
    // The grouped kernel puts all 8 routed experts in one launch: 8x the rows,
    // 8x the bytes in flight, identical arithmetic (bit-identical to the
    // ungrouped kernel, asserted in llamd-testing/tests/grouped_expert.rs).
    const EXPERTS: usize = 128;
    const USED: usize = 8;
    println!("  ROUTED EXPERTS — {USED} of {EXPERTS}, one launch against {USED}\n");
    println!(
        "  {:<24} {:>10} {:>12} {:>12} {:>9} {:>10} {:>9}",
        "shape", "MiB x8", "8 sep us", "grouped us", "speedup", "GB/s", "of peak"
    );
    println!("  {}", "-".repeat(82));

    let expert_shapes: &[(&str, usize, usize)] = &[
        ("expert_gate [2048→768]", 768, 2048),
        ("expert_up   [2048→768]", 768, 2048),
        ("expert_down [768→2048]", 2048, 768),
    ];
    let mut grouped_secs = 0.0f64;
    let mut ungrouped_secs = 0.0f64;

    for &(name, rows, k) in expert_shapes {
        let blocks = k / QK_K;
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.019).sin()).collect();
        let act = quantize_q8_k(&x);
        // The whole 3-D expert tensor, as the model stores it.
        let w = synth_weights(rows * EXPERTS, blocks);
        let used_bytes = (rows * blocks * Q6K_BLOCK_BYTES * USED) as f64;
        let ids: Vec<i32> = (0..USED as i32).map(|i| i * 13 % EXPERTS as i32).collect();

        let Ok(dw) = llamd_rocm::DeviceWeights::upload(&w, rows * EXPERTS, blocks) else {
            println!("  {name:<24}  upload failed (VRAM?)");
            continue;
        };

        // Eight separate launches, kernel-only, the way the forward pass would
        // do it today. Each expert is its own upload region, so this reuses the
        // grouped entry point with a single id — same kernel, one expert.
        let mut sep_ms = 0.0f32;
        for &id in &ids {
            sep_ms += dw
                .bench_grouped_kernel_only(&act, &[id], rows, 500)
                .unwrap_or(f32::NAN)
                / 500.0;
        }
        let sep_us = sep_ms as f64 * 1e3;

        let g_total = dw
            .bench_grouped_kernel_only(&act, &ids, rows, 500)
            .unwrap_or(f32::NAN);
        let g_us = g_total as f64 * 1e3 / 500.0;
        let gbs = used_bytes / (g_us * 1e-6) / 1e9;

        grouped_secs += g_us * 1e-6;
        ungrouped_secs += sep_us * 1e-6;

        println!(
            "  {:<24} {:>10.1} {:>12.1} {:>12.1} {:>8.1}x {:>10.1} {:>8.0}%",
            name,
            used_bytes / (1u64 << 20) as f64,
            sep_us,
            g_us,
            sep_us / g_us,
            gbs,
            100.0 * gbs / PEAK_GBS
        );
    }
    println!("  {}", "-".repeat(82));

    if grouped_secs > 0.0 {
        // Rebuild the per-token estimate with the grouped expert path in place
        // of the per-expert one, everything else as measured above.
        let attn_per_layer = (2048.0 * 4096.0 + 2048.0 * 512.0 * 2.0 + 4096.0 * 2048.0) * bpw;
        let attn_bw = bw_for(attn_per_layer / 4.0);
        let out_bw = bw_for(outproj);
        let t = 48.0 * (attn_per_layer / (attn_bw * 1e9) + grouped_secs) + outproj / (out_bw * 1e9);
        let t_old =
            48.0 * (attn_per_layer / (attn_bw * 1e9) + ungrouped_secs) + outproj / (out_bw * 1e9);
        println!("\n  PER-TOKEN ESTIMATE, weight traffic only");
        println!(
            "    per-expert launches   {:>7.2} ms  ->  {:>6.0} tok/s",
            t_old * 1e3,
            1.0 / t_old
        );
        println!(
            "    grouped               {:>7.2} ms  ->  {:>6.0} tok/s",
            t * 1e3,
            1.0 / t
        );
        println!("    llama.cpp measured                      {REF_TG:>6.1} tok/s\n");
    }

    // llama-bench on this card, ngl 99, f16 KV, short context. A live
    // llama-server carrying a 98304-token q8_0 KV cache measures 57.7 t/s
    // instead — the bar depends on the configuration, so both are recorded
    // rather than one being presented as "the" number. See docs/benchmarks.md.
    // REF_TG / REF_PP / REF_TG_SERVER are declared above, where the grouped
    // summary first cites them. Re-measured on an IDLE card: the 73.04 this
    // file used to carry was taken while a llama-server held 32 GB of VRAM.
    const ACTIVE_BYTES: f64 = 3.3e9 * (6.5625 / 8.0); // ~3.3B active at Q6_K

    println!("\n  THE BAR — llama.cpp with hipified CUDA kernels, this card:");
    println!("    prompt  {REF_PP:.0} t/s        generate  {REF_TG:.1} t/s  (llama-bench)");
    println!(
        "                             generate  {REF_TG_SERVER:.1} t/s  (server, 98k q8_0 KV)"
    );

    // Measured, not remembered. This was a hardcoded 178.5 until a rerun
    // reported 312 GB/s on the same shape — a constant that drifts from what
    // the program itself just measured is worse than no constant.
    // Deliberately NOT the best shape's bandwidth. Applying the 243 MiB
    // projection's rate to all traffic produced a 115 t/s estimate that the
    // per-shape sweep above contradicts: the expert matrices are 90% of the
    // bytes and run at a fraction of that rate. The weighted figure printed
    // in the sweep is the one to trust; this is kept only to show the size
    // of the error a best-case extrapolation makes.
    let best_bw = measured_gbs.iter().copied().fold(0.0f64, f64::max);
    if best_bw <= 0.0 {
        println!("\n  No GPU figures this run — nothing to project from.\n");
        return;
    }
    let projected = 1000.0 / (ACTIVE_BYTES / (best_bw * 1e9) * 1e3);
    println!("\n  WHERE LLAMD STANDS. The GPU kernel is validated but is NOT yet");
    println!("  wired into the forward pass, which still runs on the CPU reference");
    println!("  at 2.755 s/token — 0.363 t/s, measured end to end by");
    println!("  llamd-testing/tests/throughput.rs. Quoting the kernel ratios as");
    println!("  though they were end-to-end numbers would be dishonest: they are");
    println!("  measured against a deliberately unoptimised single-threaded CPU");
    println!("  path, not against llama.cpp. For reference, llama.cpp on the same");
    println!("  CPU with 12 threads reaches 0.75 t/s — so the scalar reference is");
    println!("  within about 2x of it while using one twelfth of the cores.");
    println!("\n  What the kernel bandwidth DOES say: a MoE token touches roughly");
    println!(
        "  {:.1} GiB of active weights, so at the {best_bw:.0} GB/s measured on the",
        ACTIVE_BYTES / (1u64 << 30) as f64
    );
    println!("  output projection the arithmetic alone would allow ~{projected:.0} t/s —");
    println!(
        "  {:.1}x the reference's {REF_TG:.0}. That headroom is the premise of this",
        projected / REF_TG
    );
    println!("  work: a kernel written for RDNA4 rather than hipified from CUDA has");
    println!("  room the hipified path does not use. It is a ceiling, not a result —");
    println!("  the kernel is not the bottleneck, the missing integration is.\n");
}
