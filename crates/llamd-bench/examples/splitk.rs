//! Measure nsub for each real projection shape, kernel-only.
//!
//! READ THE WARNING BELOW BEFORE TRUSTING THESE NUMBERS.
//!
//! This loops one matrix 1000 times. On a card with 64 MB of Infinity Cache —
//! which the R9700 has — a 6.6 MiB projection is entirely resident after the
//! first iteration, so what is measured is CACHE bandwidth, not the HBM
//! bandwidth a token actually sees. Every shape here except the 243 MiB output
//! projection fits.
//!
//! The consequence, measured: this program reports attn_q going 38.8 -> 25.5 us
//! at nsub=4 (a 1.52x gain, 177 -> 269 GB/s) and the expert gate 44.5 -> 31.3.
//! Applying exactly those settings to the engine — where a token reads each
//! layer's weights once, cold — made it 5-13% SLOWER, and an in-situ sweep
//! found nsub=1 fastest for every shape.
//!
//! So this is kept as evidence for HOW MUCH a microbenchmark can mislead, and
//! as a way to compare kernel variants when the data genuinely is resident. For
//! choosing what the engine should do, use the sweep in
//! llamd-testing/tests/device_forward.rs, which times whole tokens.
use llamd_quant::{QK_K, quantize_q8_k};

const B: usize = 210;

fn synth(rows: usize, blocks: usize) -> Vec<u8> {
    let mut w = vec![0u8; rows * blocks * B];
    let mut st = 0x9E3779B9u32;
    for b in w.iter_mut() {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (st >> 16) as u8;
    }
    for blk in 0..rows * blocks {
        let o = blk * B + QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = half::f16::from_f32(0.01);
        w[o..o + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    w
}

fn main() {
    const PEAK: f64 = 640.0;
    let shapes: &[(&str, usize, usize)] = &[
        ("attn_q      [2048->4096]", 4096, 2048),
        ("attn_out    [4096->2048]", 2048, 4096),
        ("expert_gate [2048->768]x8", 768 * 8, 2048),
        ("expert_down [768->2048]x8", 2048 * 8, 768),
        ("output      [2048->151936]", 151936, 2048),
    ];
    println!("\n  shape                        rows  blocks   nsub    us     GB/s  of peak");
    println!("  {}", "-".repeat(74));
    for &(name, rows, k) in shapes {
        let blocks = k / QK_K;
        let w = synth(rows, blocks);
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.019).sin()).collect();
        let act = quantize_q8_k(&x);
        let Ok(dw) = llamd_rocm::DeviceWeights::upload(&w, rows, blocks) else {
            println!("  {name:<28} upload failed");
            continue;
        };
        let bytes = w.len() as f64;
        let iters = if bytes > 1e8 { 100 } else { 1000 };
        let mut best = (0usize, f64::MAX);
        for nsub in [1usize, 2, 4, 8] {
            if nsub > blocks {
                continue;
            }
            let Ok(ms) = dw.bench_splitk(&act, nsub as u32, iters) else {
                continue;
            };
            let us = ms as f64 * 1e3 / iters as f64;
            let gbs = bytes / (us * 1e-6) / 1e9;
            let mark = if us < best.1 {
                best = (nsub, us);
                "  <-"
            } else {
                ""
            };
            println!(
                "  {:<28} {rows:>5} {blocks:>7} {nsub:>6} {us:>7.1} {gbs:>8.1} {:>7.0}%{mark}",
                if nsub == 1 { name } else { "" },
                100.0 * gbs / PEAK
            );
        }
        println!("     best nsub={} at {:.1} us", best.0, best.1);
    }
    println!();
}
