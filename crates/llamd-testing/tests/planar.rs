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

#[test]
fn udot_agrees_with_the_scalar_reference() {
    // The udot4 route replaces 16 multiply-accumulates per 16 weights with 4
    // dot instructions, via the identity
    //   S sc*(w-32)*a = sc*[S w*b - 128*S w - 32*S b + 65536],  b = a+128.
    // The integer part of that is exact; only the float regrouping differs, so
    // the bar is reduction error and a failure here means the identity or the
    // packing is wrong, not that the hardware rounded differently.
    if skip_no_gpu("udot_agrees_with_the_scalar_reference") {
        return;
    }
    for (rows, k, nseq) in [
        (256usize, 768usize, 1usize), // expert down: 3 blocks
        (1024, 2048, 1),              // the common shape
        (512, 4096, 1),               // attn_out: 16 blocks
        (1024, 2048, 4),              // batched
        (256, 2048, 8),               // widest batch
    ] {
        let blocks = k / QK_K;
        let w = weights(rows, blocks, 97 + rows as u32);
        let pw = llamd_rocm::PlanarWeights::upload(&w, rows, blocks).expect("planar");

        // A different activation per sequence, so a kernel that used sequence
        // 0's activation for all of them would fail rather than pass.
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

        let got = pw.matvec_udot(&acts, nseq).expect("udot matvec");
        assert_eq!(got.len(), rows * nseq);

        let mut worst = 0.0f32;
        let mut rms_acc = 0.0f64;
        let mut n = 0usize;
        for (s, a) in per_seq.iter().enumerate() {
            for r in 0..rows {
                let want = dot_q6k_q8k(&w[r * blocks * B..(r + 1) * blocks * B], a, k);
                let have = got[s * rows + r];
                worst = worst.max((have - want).abs());
                rms_acc += (want as f64) * (want as f64);
                n += 1;
            }
        }
        let rms = (rms_acc / n as f64).sqrt() as f32;
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
fn the_fused_quantize_matches_the_host() {
    // The fused kernel does rms-scale, quantize, bias to unsigned and group-sum
    // in one pass. Each of those is a place to be subtly wrong, and a wrong
    // activation shows up as a wrong token forty-eight layers later — which is
    // exactly how it was found. Compared here against quantize_q8_k directly.
    if skip_no_gpu("the_fused_quantize_matches_the_host") {
        return;
    }
    let mut x: Vec<f32> = Vec::new();
    x.extend((0..QK_K).map(|i| ((i as f32) * 0.11).sin() * 3.0));
    x.extend((0..QK_K).map(|i| -((i as f32) * 0.07).cos().abs() * 2.0)); // negative peak
    x.extend(std::iter::repeat_n(0.0f32, QK_K)); // all zero
    x.extend((0..QK_K).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })); // ties
    x.extend((0..QK_K).map(|i| if i == 0 { -8.0 } else { 8.0 })); // upper clamp

    let host = quantize_q8_k(&x);
    let (qb, sb, d) = llamd_rocm::udot::quantize_host(&x).expect("fused quantize");
    assert_eq!(d.len(), host.len());

    for (b, h) in host.iter().enumerate() {
        assert_eq!(h.d, d[b], "block {b} scale");
        for i in 0..QK_K {
            // The biased form must be exactly q + 128.
            let want = (h.qs[i] as i32 + 128) as u8;
            assert_eq!(
                qb[b * QK_K + i],
                want,
                "block {b} element {i}: host q {} -> want biased {want}, got {}",
                h.qs[i],
                qb[b * QK_K + i]
            );
        }
        for g in 0..16 {
            let want: i32 = (0..16).map(|j| h.qs[g * 16 + j] as i32 + 128).sum();
            assert_eq!(sb[b * 16 + g], want, "block {b} group {g} sum");
        }
    }
    eprintln!(
        "  {} blocks: biased quants and group sums all exact",
        host.len()
    );
}

#[test]
fn the_fused_norm_quant_matches_the_host() {
    if skip_no_gpu("the_fused_norm_quant_matches_the_host") {
        return;
    }
    let n = 2048usize;
    let nseq = 2usize;
    let w: Vec<f32> = (0..n)
        .map(|i| 0.9 + ((i as f32) * 0.003).sin() * 0.2)
        .collect();
    let mut x = Vec::with_capacity(n * nseq);
    for s in 0..nseq {
        x.extend((0..n).map(|i| ((i as f32) * 0.013 + s as f32 * 2.1).sin() * 1.7));
    }
    let eps = 1e-6f32;
    let (out, qb, _sb, d) =
        llamd_rocm::udot::norm_quant_host(&x, &w, nseq, eps).expect("fused norm_quant");

    for s in 0..nseq {
        let xs = &x[s * n..(s + 1) * n];
        let ss: f32 = xs.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / n as f32 + eps).sqrt();
        let want: Vec<f32> = xs.iter().zip(&w).map(|(v, ww)| v * inv * ww).collect();
        let worst = want
            .iter()
            .zip(&out[s * n..(s + 1) * n])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "sequence {s}: f32 output differs by {worst:e}"
        );

        let hq = quantize_q8_k(&want);
        for (b, h) in hq.iter().enumerate() {
            assert!(
                (h.d - d[s * hq.len() + b]).abs() <= h.d.abs() * 1e-5,
                "sequence {s} block {b} scale: host {} gpu {}",
                h.d,
                d[s * hq.len() + b]
            );
            for i in 0..QK_K {
                let want_b = (h.qs[i] as i32 + 128) as u8;
                let got_b = qb[s * n + b * QK_K + i];
                assert_eq!(
                    got_b, want_b,
                    "sequence {s} block {b} element {i}: host {} gpu {}",
                    want_b, got_b
                );
            }
        }
    }
    eprintln!("  fused norm+quant matches the host for {nseq} sequences");
}
