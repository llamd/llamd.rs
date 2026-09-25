//! The attention kernels against a CPU reference, driven directly.
//!
//! §36 found both flash kernels numerically wrong and could not say WHERE, because it
//! compared final logits — 48 layers downstream of the kernel, where a relative error of
//! 1e-5 becomes an absolute error of 1. Debugging through that amplification produced two
//! wrong kernels in a row.
//!
//! This drives `ops::attention` with synthetic Q, K and V and compares against softmax
//! computed in f64 on the host. No model, no layers, no amplification: whatever deviates
//! here deviates in the kernel, and the reference says WHICH kernel is wrong rather than
//! merely that two disagree.

use llamd_vulkan::{VkBuf, ops};

const HD: usize = 128; // mode 2 fixes eight vec4 per lane, so head_dim is exactly this
const NH: usize = 4;
const NKVH: usize = 1;

fn ok() -> bool {
    llamd_vulkan::available()
}

/// Deterministic pseudo-random, so a failure is reproducible.
fn rnd(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 8) as f32 / 8388608.0) - 1.0
}

/// Softmax attention in f64, the thing every kernel has to reproduce.
fn reference(q: &[f32], k: &[f32], v: &[f32], n_kv: usize) -> Vec<f32> {
    let kv_dim = NKVH * HD;
    let scale = 1.0f64 / (HD as f64).sqrt();
    let mut out = vec![0.0f32; NH * HD];
    for h in 0..NH {
        let kvh = h / (NH / NKVH);
        let mut s = vec![0.0f64; n_kv];
        for (j, sj) in s.iter_mut().enumerate() {
            let mut d = 0.0f64;
            for i in 0..HD {
                d += q[h * HD + i] as f64 * k[j * kv_dim + kvh * HD + i] as f64;
            }
            *sj = d * scale;
        }
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0.0f64;
        for sj in s.iter_mut() {
            *sj = (*sj - m).exp();
            l += *sj;
        }
        for i in 0..HD {
            let mut a = 0.0f64;
            for (j, sj) in s.iter().enumerate() {
                a += sj * v[j * kv_dim + kvh * HD + i] as f64;
            }
            out[h * HD + i] = (a / l) as f32;
        }
    }
    out
}

/// `shape` selects the score distribution, because a softmax bug can hide in a benign one.
fn build(n_kv: usize, shape: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_dim = NKVH * HD;
    let mut seed = 0x1234_5678u32;
    let mut q = vec![0.0f32; NH * HD];
    for x in q.iter_mut() {
        *x = rnd(&mut seed) * 0.5;
    }
    let mut k = vec![0.0f32; n_kv * kv_dim];
    let mut v = vec![0.0f32; n_kv * kv_dim];
    for x in v.iter_mut() {
        *x = rnd(&mut seed);
    }
    for (j, chunk) in k.chunks_mut(kv_dim).enumerate() {
        for x in chunk.iter_mut() {
            *x = rnd(&mut seed) * 0.5;
        }
        match shape {
            // One key overwhelmingly dominant: the running max moves once, hugely.
            "dominant" if j == n_kv / 2 => {
                chunk.iter_mut().zip(&q).for_each(|(x, qq)| *x = qq * 40.0)
            }
            // Every score nearly equal: no rescale should happen at all.
            "flat" => chunk.iter_mut().zip(&q).for_each(|(x, qq)| *x = qq * 0.01),
            // All scores far below zero.
            "negative" => chunk
                .iter_mut()
                .zip(&q)
                .for_each(|(x, qq)| *x = -qq.abs() * 8.0),
            // The max arrives LAST, so every earlier block must be rescaled.
            "rising" => {
                let f = 1.0 + 30.0 * (j as f32) / (n_kv as f32);
                chunk.iter_mut().zip(&q).for_each(|(x, qq)| *x = qq * f);
            }
            _ => {}
        }
    }
    (q, k, v)
}

#[test]
#[ignore = "needs a free GPU"]
fn attention_kernels_against_a_cpu_reference() {
    if !ok() {
        eprintln!("  SKIP: no Vulkan");
        return;
    }
    let max_kv = 4096usize;
    let kv_dim = NKVH * HD;
    let lengths = [
        1usize, 2, 3, 4, 5, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023,
        1024, 2048,
    ];
    let shapes = ["random", "dominant", "flat", "negative", "rising"];

    eprintln!(
        "  {:>6}{:>10}{:>12}{:>12}{:>12}",
        "n_kv", "shape", "mode 0", "mode 1", "mode 2"
    );
    eprintln!("  (max |kernel - f64 reference| over {} outputs)", NH * HD);
    let mut worst = [0.0f32; 3];
    let mut worst_at = [(0usize, ""); 3];
    for &shape in &shapes {
        for &n_kv in &lengths {
            let (q, k, v) = build(n_kv, shape);
            let want = reference(&q, &k, &v, n_kv);

            let qb = VkBuf::from_f32(&q).expect("q");
            let mut kk = vec![0.0f32; max_kv * kv_dim];
            kk[..k.len()].copy_from_slice(&k);
            let mut vv = vec![0.0f32; max_kv * kv_dim];
            vv[..v.len()].copy_from_slice(&v);
            let kb = VkBuf::from_f32(&kk).expect("k");
            let vb = VkBuf::from_f32(&vv).expect("v");
            let out = VkBuf::new(NH * HD * 4).expect("out");
            let sc = VkBuf::new(NH * max_kv * 4).expect("scores");
            let meta = VkBuf::from_i32(&[0i32, n_kv as i32 - 1]).expect("meta");

            let mut row = [0.0f32; 3];
            for m in 0..3usize {
                ops::set_attention_flash(m);
                ops::begin().expect("begin");
                ops::attention(&qb, &kb, &vb, &out, &sc, &meta, NH, NKVH, HD, max_kv, 1, 0)
                    .expect("attention");
                ops::end_and_wait().expect("wait");
                let mut got = vec![0.0f32; NH * HD];
                out.download_f32(&mut got).expect("download");
                let e = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                row[m] = e;
                if e > worst[m] {
                    worst[m] = e;
                    worst_at[m] = (n_kv, shape);
                }
            }
            // Only print the interesting rows, or the table is 110 lines.
            if row.iter().any(|&e| e > 1e-4) || n_kv <= 4 {
                eprintln!(
                    "  {n_kv:>6}{shape:>10}{:>12}{:>12}{:>12}",
                    format!("{:.2e}", row[0]),
                    format!("{:.2e}", row[1]),
                    format!("{:.2e}", row[2])
                );
            }
        }
    }
    ops::set_attention_flash(0);
    eprintln!();
    for m in 0..3 {
        eprintln!(
            "  mode {m}: worst {:.3e} at n_kv {} shape {}",
            worst[m], worst_at[m].0, worst_at[m].1
        );
    }
}
