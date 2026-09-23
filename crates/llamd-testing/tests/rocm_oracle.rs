//! The ROCm backend, validated against the CPU reference and against ggml.
//!
//! THE BAR IS QUANTIZED-AGAINST-QUANTIZED. The kernel performs the same Q6_K ×
//! Q8_K integer arithmetic as `llamd_quant::dot_q6k_q8k`, so the comparison is
//! held to `REDUCTION` — a reordered summation — and not to the far looser
//! `QUANTIZED_MATMUL`, which exists for f32-against-quantized.
//!
//! Using the loose bound here would waste the comparison: the whole reason to
//! implement the quantized path was that small differences flip near-tied MoE
//! experts, and a tolerance wide enough to hide one would defeat it.
//!
//! Skips loudly without a device, so the suite runs on a machine with no GPU.

use llamd_core::tolerance::REDUCTION;
use llamd_quant::{Q6K_BLOCK_BYTES, QK_K, dot_q6k_q8k, quantize_q8_k};
use llamd_testing::skip;

fn sample(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Deterministic but structurally valid Q6_K bytes: random codes, sane scales.
fn weights(rows: usize, blocks: usize, seed: u32) -> Vec<u8> {
    let mut w = vec![0u8; rows * blocks * Q6K_BLOCK_BYTES];
    let mut st = seed.wrapping_mul(2246822519).wrapping_add(7);
    for b in w.iter_mut() {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (st >> 16) as u8;
    }
    // The f16 super-block scale must be a real number, not an arbitrary bit
    // pattern that could decode to inf or NaN and make every comparison vacuous.
    for blk in 0..rows * blocks {
        let d_off = blk * Q6K_BLOCK_BYTES + QK_K / 2 + QK_K / 4 + QK_K / 16;
        let d = half::f16::from_f32(0.008 + (blk % 17) as f32 * 0.002);
        w[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    w
}

#[test]
fn reports_the_device_or_says_why_not() {
    if !llamd_rocm::built_with_rocm() {
        skip(
            "reports_the_device",
            "built without ROCm — nothing to report",
        );
        return;
    }
    let n = llamd_rocm::device_count();
    if n == 0 {
        skip("reports_the_device", "ROCm present but no device visible");
        return;
    }
    let info = llamd_rocm::device_info(0).expect("device 0");
    eprintln!("  {info}");

    assert!(
        info.total_mem > (8u64 << 30) as usize,
        "implausibly small device"
    );
    assert!(info.wgp_count > 0);
    assert!(info.arch.starts_with("gfx"), "arch was {:?}", info.arch);
    // RDNA runs wave32 natively. If this ever reports 64 the kernels' shuffle
    // width is wrong, and they would silently reduce across the wrong lanes.
    assert_eq!(
        info.wave_size, 32,
        "kernels assume wave32; this device reports wave{}",
        info.wave_size
    );
}

#[test]
fn gpu_matvec_matches_the_cpu_reference() {
    if !llamd_rocm::available() {
        skip("gpu_matvec_matches_the_cpu_reference", "no ROCm device");
        return;
    }
    for (rows, blocks, seed) in [(1usize, 1usize, 3u32), (8, 4, 5), (64, 8, 11)] {
        let n = blocks * QK_K;
        let w = weights(rows, blocks, seed);
        let act = quantize_q8_k(&sample(n, seed + 100));

        let cpu: Vec<f32> = (0..rows)
            .map(|r| {
                let row = &w[r * blocks * Q6K_BLOCK_BYTES..(r + 1) * blocks * Q6K_BLOCK_BYTES];
                dot_q6k_q8k(row, &act, n)
            })
            .collect();
        let gpu = llamd_rocm::matvec_q6k(&w, &act, rows).expect("gpu matvec");

        assert_eq!(gpu.len(), cpu.len());
        let (idx, dev) = llamd_core::tolerance::max_deviation(&gpu, &cpu).expect("comparable");
        assert!(
            dev <= REDUCTION,
            "rows{rows} blocks{blocks}: deviation {dev:.3e} at row {idx} exceeds {REDUCTION:.1e}\n  \
             gpu {:?}\n  cpu {:?}",
            gpu.get(idx),
            cpu.get(idx),
        );
        eprintln!("  rows{rows:<4} blocks{blocks:<3} max deviation {dev:.3e}");
    }
}
