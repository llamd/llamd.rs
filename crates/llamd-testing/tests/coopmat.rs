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
