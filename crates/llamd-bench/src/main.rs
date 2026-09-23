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

    println!("\n  NOTE ON WHAT THIS IS NOT. These are synthetic weights at real");
    println!("  shapes, one activation vector, weights already resident. It is a");
    println!("  kernel measurement, not tokens/second: no attention, no KV cache,");
    println!("  no sampling. The comparison against llama.cpp's hipified kernels");
    println!("  is the number that decides the project's premise, and it needs a");
    println!("  HIP build of the reference.\n");
}
