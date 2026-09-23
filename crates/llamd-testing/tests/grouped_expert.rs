//! The grouped expert kernel: identical arithmetic, then speed.
//!
//! WHY CORRECTNESS COMES FIRST HERE. The grouped kernel exists to make the
//! expert path fast, and the expert path is where this model's routing
//! decisions are decided. A grouped kernel that is fast and differs would flip
//! a near-tied expert selection and change the emitted token — the failure mode
//! that cost this project a day when the f32 reference disagreed with ggml at
//! layer 47.
//!
//! TWO DIFFERENT BARS, AND THE DISTINCTION MATTERS.
//!
//! Against the UNGROUPED GPU kernel: EXACT. Grouping changes which matrix a
//! workgroup indexes, not how it sums, so every float operation happens in the
//! same order on the same values. Any difference at all is a bug, and a
//! tolerance here would hide exactly the indexing errors this test is for.
//!
//! Against the SCALAR CPU dot product: `REDUCTION`. This is not a concession —
//! it is arithmetic. The CPU multiplies each block's complete integer sum by
//! that block's scale; the kernel gives each lane a partial integer sum,
//! scales it, and adds across lanes with a shuffle tree. Algebraically the
//! same, and a parallel reduction cannot be bit-identical to a sequential one.
//! Measured deviation on the shapes below is ~1.3e-7 relative — one ULP.
//!
//! An earlier version of this file asserted exact equality against the CPU and
//! failed on 11608 of 16384 rows at 6.1e-5 absolute on values near 461. The
//! kernel was right and the test was wrong, which is worth recording: the
//! first instinct of "assert exact, tolerances hide bugs" is correct for the
//! GPU-to-GPU comparison and incorrect for the GPU-to-CPU one.

use llamd_core::tolerance::REDUCTION;
use llamd_quant::{QK_K, dot_q6k_q8k, quantize_q8_k};

const Q6K_BLOCK_BYTES: usize = 210;

/// Deterministic pseudo-Q6_K bytes. Not a real quantization — the dot product
/// does not care, and both paths read the identical bytes.
fn synth(rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut v = vec![0u8; rows * blocks * Q6K_BLOCK_BYTES];
    for (i, b) in v.iter_mut().enumerate() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = ((s >> 33) as u8) ^ (i as u8);
    }
    // The f16 scale sits at the end of each block; keep it small and finite so
    // accumulated values stay in a range where f32 comparison is meaningful.
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q6K_BLOCK_BYTES + QK_K / 2 + QK_K / 4 + QK_K / 16;
            v[off] = 0x00;
            v[off + 1] = 0x2c; // ~0.0156 in f16
        }
    }
    v
}

fn skip_if_no_gpu(what: &str) -> bool {
    if !llamd_rocm::available() {
        eprintln!("  skipping {what}: no ROCm device");
        return true;
    }
    false
}

#[test]
fn grouped_matches_the_ungrouped_kernel_exactly() {
    if skip_if_no_gpu("grouped_matches_the_ungrouped_kernel_exactly") {
        return;
    }
    // expert_down's shape, the worst performer: k=768 is only three blocks.
    let (rows, k, n_expert) = (2048usize, 768usize, 16usize);
    let blocks = k / QK_K;
    let w = synth(rows * n_expert, blocks, 0x5eed);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let act = quantize_q8_k(&x);

    let ids: Vec<i32> = vec![3, 0, 15, 7, 7, 1, 11, 2];
    let dw = llamd_rocm::DeviceWeights::upload(&w, rows * n_expert, blocks).expect("upload");
    let got = dw.matvec_grouped(&act, &ids, rows).expect("grouped");
    assert_eq!(got.len(), rows * ids.len());

    // The ungrouped kernel over each expert's slice, which is the arithmetic
    // the token-agreement tests already validate end to end.
    let mut mismatched = 0usize;
    let mut first: Option<String> = None;
    for (e, &eid) in ids.iter().enumerate() {
        let base = eid as usize * rows * blocks * Q6K_BLOCK_BYTES;
        let slice = &w[base..base + rows * blocks * Q6K_BLOCK_BYTES];
        let single = llamd_rocm::DeviceWeights::upload(slice, rows, blocks).expect("upload one");
        let want = single.matvec(&act).expect("ungrouped");
        for r in 0..rows {
            let have = got[e * rows + r];
            if want[r] != have {
                mismatched += 1;
                if first.is_none() {
                    first = Some(format!(
                        "expert {eid} row {r}: ungrouped {:e}, grouped {have:e}, delta {:e}",
                        want[r],
                        have - want[r]
                    ));
                }
            }
        }
    }
    assert_eq!(
        mismatched,
        0,
        "{mismatched} of {} rows differ from the ungrouped kernel. First: {}",
        rows * ids.len(),
        first.unwrap_or_default()
    );
    eprintln!(
        "  {} rows across {} routed experts, bit-identical to the ungrouped kernel",
        rows * ids.len(),
        ids.len()
    );
}

#[test]
fn grouped_matches_the_scalar_cpu_within_reduction() {
    if skip_if_no_gpu("grouped_matches_the_scalar_cpu_within_reduction") {
        return;
    }
    let (rows, k, n_expert) = (512usize, 768usize, 8usize);
    let blocks = k / QK_K;
    let w = synth(rows * n_expert, blocks, 0x1234);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.021).cos()).collect();
    let act = quantize_q8_k(&x);
    let ids: Vec<i32> = vec![5, 2, 7, 0];

    let dw = llamd_rocm::DeviceWeights::upload(&w, rows * n_expert, blocks).expect("upload");
    let got = dw.matvec_grouped(&act, &ids, rows).expect("grouped");

    let mut cpu = Vec::with_capacity(got.len());
    for &eid in &ids {
        let base = eid as usize * rows * blocks * Q6K_BLOCK_BYTES;
        for r in 0..rows {
            let off = base + r * blocks * Q6K_BLOCK_BYTES;
            cpu.push(dot_q6k_q8k(
                &w[off..off + blocks * Q6K_BLOCK_BYTES],
                &act,
                k,
            ));
        }
    }
    // WHY NOT A RELATIVE TOLERANCE PER ELEMENT. `max_deviation` scales the
    // difference by the values being compared, which is right when those values
    // represent the size of the computation. In a dot product over random
    // weights it is not: one row here accumulates partial sums in the millions
    // and cancels down to 0.7956, so a 2.1e-4 absolute difference — ordinary
    // f32 reduction noise at that accumulation scale — reads as 2.7e-4
    // relative and fails a 1e-5 bound. The bound would be measuring
    // cancellation, not correctness.
    //
    // The honest scale for reduction error is the magnitude of what was summed.
    // RMS over the whole result vector stands in for it: it is dominated by the
    // rows that did not cancel, which is where the accumulation magnitude shows.
    let rms = (cpu.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cpu.len() as f64).sqrt()
        as f32;
    let (idx, worst) = got
        .iter()
        .zip(&cpu)
        .enumerate()
        .map(|(i, (g, c))| (i, (g - c).abs()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("non-empty");
    let bound = REDUCTION * rms;
    assert!(
        worst <= bound,
        "worst absolute deviation {worst:.3e} at row {idx} exceeds {bound:.3e} \
         (= REDUCTION {REDUCTION:.1e} x rms {rms:.3e});  gpu {:e}, cpu {:e}",
        got[idx],
        cpu[idx]
    );
    eprintln!(
        "  {} rows: worst absolute deviation {worst:.3e}, bound {bound:.3e} (rms {rms:.3e})",
        got.len()
    );
}

#[test]
fn a_repeated_expert_gives_a_repeated_block() {
    // Routing can select the same expert twice in principle; the kernel must
    // treat each slot independently rather than aliasing them.
    if skip_if_no_gpu("a_repeated_expert_gives_a_repeated_block") {
        return;
    }
    let (rows, k, n_expert) = (256usize, 512usize, 4usize);
    let blocks = k / QK_K;
    let w = synth(rows * n_expert, blocks, 0xabcd);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.03).cos()).collect();
    let act = quantize_q8_k(&x);
    let dw = llamd_rocm::DeviceWeights::upload(&w, rows * n_expert, blocks).expect("upload");
    let got = dw.matvec_grouped(&act, &[2, 2], rows).expect("grouped");
    assert_eq!(
        &got[..rows],
        &got[rows..],
        "the same expert twice must match"
    );
}

#[test]
fn bad_expert_ids_are_refused() {
    if skip_if_no_gpu("bad_expert_ids_are_refused") {
        return;
    }
    let (rows, k, n_expert) = (64usize, 256usize, 2usize);
    let blocks = k / QK_K;
    let w = synth(rows * n_expert, blocks, 7);
    let x: Vec<f32> = vec![0.5; k];
    let act = quantize_q8_k(&x);
    let dw = llamd_rocm::DeviceWeights::upload(&w, rows * n_expert, blocks).expect("upload");
    // Out of range, and a row count that does not divide the upload: both would
    // read past the allocation, so both must be errors and not clamps.
    assert!(dw.matvec_grouped(&act, &[2], rows).is_err(), "id 2 of 2");
    assert!(dw.matvec_grouped(&act, &[-1], rows).is_err(), "negative id");
    assert!(dw.matvec_grouped(&act, &[0], 0).is_err(), "zero rows");
    assert!(
        dw.matvec_grouped(&act, &[0], rows + 1).is_err(),
        "non-divisor"
    );
    assert!(dw.matvec_grouped(&act, &[], rows).is_err(), "no experts");
}

#[test]
fn device_quantization_is_identical_to_the_host() {
    if skip_if_no_gpu("device_quantization_is_identical_to_the_host") {
        return;
    }
    // Cases chosen to hit the parts of quantize_q8_k that are easy to get
    // wrong: a negative peak (which is what -127/max is for), an all-zero
    // block, exact .5 values for round-half-to-even, and a value that needs
    // the upper clamp.
    let mut x: Vec<f32> = Vec::new();
    x.extend((0..QK_K).map(|i| ((i as f32) * 0.11).sin() * 3.0)); // mixed
    x.extend((0..QK_K).map(|i| -((i as f32) * 0.07).cos().abs() * 2.0)); // negative peak
    x.extend(std::iter::repeat_n(0.0f32, QK_K)); // all zero
    x.extend((0..QK_K).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })); // ties
    x.extend((0..QK_K).map(|i| if i == 0 { -8.0 } else { 8.0 })); // clamp

    let host = quantize_q8_k(&x);
    let dev = llamd_rocm::quantize_q8k_on_device(&x).expect("device quantize");
    assert_eq!(host.len(), dev.len());

    for (b, (h, g)) in host.iter().zip(&dev).enumerate() {
        assert_eq!(h.d, g.d, "block {b} scale: host {:e} gpu {:e}", h.d, g.d);
        let bad: Vec<usize> = (0..QK_K).filter(|&i| h.qs[i] != g.qs[i]).collect();
        assert!(
            bad.is_empty(),
            "block {b}: {} quants differ, first at {} (host {} gpu {})",
            bad.len(),
            bad[0],
            h.qs[bad[0]],
            g.qs[bad[0]]
        );
        assert_eq!(h.bsums, g.bsums, "block {b} bsums");
    }
    eprintln!("  {} blocks, scales quants and bsums all exact", host.len());
}
