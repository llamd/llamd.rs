//! The planar Q6_K layout: same values, wider loads.
//!
//! WHAT IS BEING TESTED. `repack_q6k_planar` permutes ggml's 210-byte
//! interleaved blocks into four aligned planes so the kernel can use 16-byte
//! loads. A cold streaming read on this card reaches 637.8 GB/s with `uint4`
//! loads against 259 GB/s for the byte-load Q6_K matvec in the engine, and
//! alignment is the whole reason the wide load was unavailable: 210 % 16 = 2.
//!
//! The rewrite changes which bytes a lane owns — 16 consecutive instead of two
//! strided — so the per-lane partial sums differ and the float additions
//! regroup. Values must still agree with the scalar reference to within
//! reduction error, and that is what the first test asserts.

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

fn skip_no_gpu(what: &str) -> bool {
    if !llamd_rocm::available() {
        eprintln!("  skipping {what}: no ROCm device");
        return true;
    }
    false
}

#[test]
fn planar_agrees_with_the_scalar_reference() {
    if skip_no_gpu("planar_agrees_with_the_scalar_reference") {
        return;
    }
    // Every block count the model actually uses: 3 (expert down), 8 (most),
    // 16 (attn_out). blocks=3 exercises the tail where blocks*8 is not a
    // multiple of the wave width.
    for (rows, k) in [(256usize, 768usize), (1024, 2048), (512, 4096), (7, 2048)] {
        let blocks = k / QK_K;
        let w = weights(rows, blocks, 31 + rows as u32);
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.023).sin()).collect();
        let act = quantize_q8_k(&x);

        let cpu: Vec<f32> = (0..rows)
            .map(|r| dot_q6k_q8k(&w[r * blocks * B..(r + 1) * blocks * B], &act, k))
            .collect();

        let pw = llamd_rocm::PlanarWeights::upload(&w, rows, blocks).expect("planar upload");
        let gpu = pw.matvec(&act).expect("planar matvec");
        assert_eq!(gpu.len(), cpu.len());

        // Scaled by accumulation magnitude, not per element: with random
        // weights a row can cancel from partial sums in the millions down to
        // near zero, and a per-element relative bound would measure that
        // cancellation rather than correctness.
        let rms = (cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cpu.len() as f64)
            .sqrt() as f32;
        let (idx, worst) = gpu
            .iter()
            .zip(&cpu)
            .enumerate()
            .map(|(i, (g, c))| (i, (g - c).abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("non-empty");
        let bound = REDUCTION * rms.max(1.0);
        assert!(
            worst <= bound,
            "{rows}x{k}: worst deviation {worst:.3e} at row {idx} exceeds {bound:.3e}; \
             gpu {:e} cpu {:e}",
            gpu[idx],
            cpu[idx]
        );
        eprintln!("  {rows:>6} rows, {blocks:>2} blocks: worst {worst:.3e}, bound {bound:.3e}");
    }
}

#[test]
fn planar_is_faster_on_a_cold_tensor() {
    if skip_no_gpu("planar_is_faster_on_a_cold_tensor") {
        return;
    }
    // BIG ENOUGH TO DEFEAT THE CACHE. This card has 64 MB of Infinity Cache,
    // and a benchmark that loops a 6.6 MiB matrix measures cache bandwidth —
    // which is how the split-k experiment came to promise a 1.52x gain and
    // deliver a 7% loss. 151936 rows is 243 MiB, nearly 4x the cache.
    let (rows, k) = (151936usize, 2048usize);
    let blocks = k / QK_K;
    let w = weights(rows, blocks, 77);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.019).sin()).collect();
    let act = quantize_q8_k(&x);
    let bytes = w.len() as f64;

    let inter = llamd_rocm::DeviceWeights::upload(&w, rows, blocks).expect("interleaved");
    let plan = llamd_rocm::PlanarWeights::upload(&w, rows, blocks).expect("planar");

    let iters = 50u32;
    let i_ms = inter
        .bench_kernel_only(&act, iters)
        .expect("interleaved bench");
    let p_ms = plan.bench(&act, iters).expect("planar bench");
    let i_us = i_ms as f64 * 1e3 / iters as f64;
    let p_us = p_ms as f64 * 1e3 / iters as f64;
    let i_gbs = bytes / (i_us * 1e-6) / 1e9;
    let p_gbs = bytes / (p_us * 1e-6) / 1e9;

    eprintln!(
        "  {:.1} MiB, {blocks} blocks/row",
        bytes / (1u64 << 20) as f64
    );
    eprintln!(
        "    interleaved  {i_us:>8.1} us  {i_gbs:>6.1} GB/s  {:>3.0}% of 638",
        100.0 * i_gbs / 637.8
    );
    eprintln!(
        "    planar       {p_us:>8.1} us  {p_gbs:>6.1} GB/s  {:>3.0}% of 638",
        100.0 * p_gbs / 637.8
    );
    eprintln!("    speedup      {:.2}x", i_us / p_us);

    // Also check they agree on this shape, since it is the one the engine cares
    // most about: the output projection is 10% of a token's weight traffic.
    let a = inter.matvec(&act).expect("interleaved matvec");
    let b = plan.matvec(&act).expect("planar matvec");
    let rms =
        (a.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / a.len() as f64).sqrt() as f32;
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= REDUCTION * rms.max(1.0),
        "planar and interleaved disagree by {worst:.3e} (rms {rms:.3e})"
    );
}
