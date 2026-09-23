//! CPU backend ops, validated against ggml itself.
//!
//! The oracle is `tools/ggml-oracle`, a small C binary linked against the
//! pinned ggml. It computes one op and writes raw f32, so the comparison is
//! against the reference implementation rather than against our idea of it.
//!
//! Build it with `tools/ggml-oracle/build.sh`; the tests skip loudly if it is
//! absent, and FAIL if it is present and errors.
//!
//! TOLERANCES ARE NAMED. Each comparison below states which constant from
//! `llamd_core::tolerance` it uses and why that one. A comparison whose
//! tolerance is chosen to make it pass is not a comparison.

use llamd_core::tolerance::{ATTENTION, EXACT, LAYER, REDUCTION, max_deviation};
use llamd_cpu::CpuBackend;
use llamd_tensor::{Backend, RopeConfig, Tensor};
use llamd_testing::skip;
use std::io::Write;
use std::process::{Command, Stdio};

fn oracle_path() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .join("tools/ggml-oracle/oracle");
    p.is_file().then_some(p)
}

/// Run the oracle: args after the op, f32 input, optional trailing i32 block.
fn oracle(op: &str, args: &[String], f32s: &[f32], i32s: &[i32]) -> Option<Vec<f32>> {
    let bin = oracle_path()?;
    let mut c = Command::new(&bin)
        .arg(op)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    {
        let si = c.stdin.as_mut()?;
        for v in f32s {
            si.write_all(&v.to_le_bytes()).ok()?;
        }
        for v in i32s {
            si.write_all(&v.to_le_bytes()).ok()?;
        }
    }
    let out = c.wait_with_output().ok()?;
    // Present and failing is a FAILURE, never a skip: skipping here would
    // report success while comparing nothing.
    assert!(
        out.status.success(),
        "ggml oracle failed for {op}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(
        out.stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Deterministic pseudo-random inputs. A fixed sequence, so a failure is
/// reproducible — random test data that cannot be replayed is a bug report
/// nobody can act on.
fn sample(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn check(label: &str, ours: &[f32], theirs: &[f32], tol: f32) {
    assert_eq!(
        ours.len(),
        theirs.len(),
        "{label}: length {} vs {}",
        ours.len(),
        theirs.len()
    );
    let (i, d) = max_deviation(ours, theirs).expect("comparable");
    assert!(
        d <= tol,
        "{label}: max relative deviation {d:.3e} at index {i} exceeds {tol:.1e}\n  \
         ours {:?}\n  ggml {:?}",
        &ours[i.saturating_sub(2)..(i + 3).min(ours.len())],
        &theirs[i.saturating_sub(2)..(i + 3).min(theirs.len())],
    );
    eprintln!("  {label}: max deviation {d:.3e} (tolerance {tol:.1e})");
}

#[test]
fn rms_norm_matches_ggml() {
    if oracle_path().is_none() {
        skip("rms_norm_matches_ggml", "build tools/ggml-oracle first");
        return;
    }
    let b = CpuBackend::new();
    for (ne0, ne1) in [(2usize, 1usize), (128, 4), (2048, 3)] {
        let x = sample(ne0 * ne1, 7);
        let t = Tensor::from_slice(&[ne0, ne1], &x).unwrap();
        let ours = b.rms_norm(&t, 1e-6).unwrap();
        let theirs = oracle(
            "rms_norm",
            &[ne0.to_string(), ne1.to_string(), "0.000001".into()],
            &x,
            &[],
        )
        .unwrap();
        // REDUCTION, not F32_ULP. The sum of squares is a reduction, and
        // ggml's CPU path blocks it differently from our straight loop. At
        // ne0=128 the observed disagreement was 1.9 ULP — consistent with a
        // reordered sum (sqrt(128) = 11 ULP would be the bound) and far too
        // small to be a structural difference, which would be O(1).
        check(
            &format!("rms_norm {ne0}x{ne1}"),
            ours.data(),
            &theirs,
            REDUCTION,
        );
    }
}

#[test]
fn softmax_matches_ggml() {
    if oracle_path().is_none() {
        skip("softmax_matches_ggml", "build tools/ggml-oracle first");
        return;
    }
    let b = CpuBackend::new();
    for (ne0, ne1) in [(8usize, 1usize), (512, 4)] {
        let x = sample(ne0 * ne1, 11);
        let t = Tensor::from_slice(&[ne0, ne1], &x).unwrap();
        let ours = b.softmax(&t).unwrap();
        let theirs = oracle("soft_max", &[ne0.to_string(), ne1.to_string()], &x, &[]).unwrap();
        // ATTENTION: exp and the normalising division admit a little more
        // than one ULP, and softmax is an attention-block operation.
        check(
            &format!("softmax {ne0}x{ne1}"),
            ours.data(),
            &theirs,
            ATTENTION,
        );
    }
}

#[test]
fn silu_matches_ggml() {
    if oracle_path().is_none() {
        skip("silu_matches_ggml", "build tools/ggml-oracle first");
        return;
    }
    let b = CpuBackend::new();
    let (ne0, ne1) = (256usize, 2usize);
    let x = sample(ne0 * ne1, 13);
    let ours = b
        .silu(&Tensor::from_slice(&[ne0, ne1], &x).unwrap())
        .unwrap();
    let theirs = oracle("silu", &[ne0.to_string(), ne1.to_string()], &x, &[]).unwrap();
    // ggml may use a polynomial approximation for exp; ATTENTION covers that
    // without hiding a structural difference.
    check("silu", ours.data(), &theirs, ATTENTION);
}

#[test]
fn matmul_matches_ggml() {
    if oracle_path().is_none() {
        skip("matmul_matches_ggml", "build tools/ggml-oracle first");
        return;
    }
    let b = CpuBackend::new();
    // a = [k, m], b = [k, n] -> [m, n]. If the dimension convention were
    // backwards this would produce a transposed result and fail loudly.
    for (k, m, n) in [(4usize, 3usize, 1usize), (64, 32, 5), (256, 128, 2)] {
        let wa = sample(k * m, 17);
        let xb = sample(k * n, 19);
        let ours = b
            .matmul(
                &Tensor::from_slice(&[k, m], &wa).unwrap(),
                &Tensor::from_slice(&[k, n], &xb).unwrap(),
            )
            .unwrap();
        let mut input = wa.clone();
        input.extend_from_slice(&xb);
        let theirs = oracle(
            "mul_mat",
            &[k.to_string(), m.to_string(), k.to_string(), n.to_string()],
            &input,
            &[],
        )
        .unwrap();
        // LAYER: a k-term reduction accumulates error with k, and ggml's CPU
        // path may block the sum differently from our straight loop.
        check(
            &format!("matmul k{k} m{m} n{n}"),
            ours.data(),
            &theirs,
            LAYER,
        );
    }
}

#[test]
fn rope_matches_ggml() {
    if oracle_path().is_none() {
        skip("rope_matches_ggml", "build tools/ggml-oracle first");
        return;
    }
    let b = CpuBackend::new();
    // head_dim 128 and freq_base 10^7 are this model's real values. A test
    // that only used 10^4 would pass while the model was configured wrong.
    for (head_dim, heads, ntok, base) in [
        (4usize, 1usize, 1usize, 10000.0f32),
        (128, 1, 6, 10_000_000.0),
        (64, 2, 3, 10_000_000.0),
    ] {
        let ne0 = head_dim * heads;
        let x = sample(ne0 * ntok, 23);
        let pos: Vec<u32> = (0..ntok as u32).map(|i| i * 3 + 1).collect();
        let ours = b
            .rope(
                &Tensor::from_slice(&[ne0, ntok], &x).unwrap(),
                &pos,
                RopeConfig::new(head_dim, base),
            )
            .unwrap();
        // ggml takes the token count from ne[2], so the oracle is given
        // [head_dim, n_heads, n_tokens] — the same memory, described as 3D.
        let theirs = oracle(
            "rope",
            &[
                head_dim.to_string(),
                heads.to_string(),
                ntok.to_string(),
                format!("{base}"),
            ],
            &x,
            &pos.iter().map(|&p| p as i32).collect::<Vec<_>>(),
        )
        .unwrap();
        // ATTENTION: sin/cos differ in the last bits between libm and ggml's
        // own tables. A convention error would be O(1), not O(1e-6).
        check(
            &format!("rope d{head_dim} h{heads} n{ntok}"),
            ours.data(),
            &theirs,
            ATTENTION,
        );
    }
}

#[test]
fn get_rows_matches_ggml_exactly() {
    if oracle_path().is_none() {
        skip(
            "get_rows_matches_ggml_exactly",
            "build tools/ggml-oracle first",
        );
        return;
    }
    let b = CpuBackend::new();
    let (ne0, ne1) = (16usize, 10usize);
    let x = sample(ne0 * ne1, 29);
    let ids: Vec<u32> = vec![3, 0, 9, 9, 1];
    let ours = b
        .get_rows(&Tensor::from_slice(&[ne0, ne1], &x).unwrap(), &ids)
        .unwrap();
    let theirs = oracle(
        "get_rows",
        &[ne0.to_string(), ne1.to_string(), ids.len().to_string()],
        &x,
        &ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
    )
    .unwrap();
    // EXACT: gathering copies bytes. There is no arithmetic to round, so any
    // difference at all means the wrong rows were selected — which in MoE
    // routing would be a different expert.
    check("get_rows", ours.data(), &theirs, EXACT);
}

/// Expert selection must match ggml EXACTLY — a different expert is a
/// different model, so there is no tolerance to apply.
///
/// ggml documents `argsort_top_k` as returning indices in no particular order,
/// so the comparison is set-based. Asserting an order would be testing an
/// implementation detail the reference does not promise, and would fail for
/// the wrong reason the day ggml changes its sort.
#[test]
fn moe_expert_selection_matches_ggml_exactly() {
    if oracle_path().is_none() {
        skip(
            "moe_expert_selection_matches_ggml_exactly",
            "build tools/ggml-oracle first",
        );
        return;
    }
    // The real routing shape: 8 experts chosen from 128.
    let (n_expert, n_used) = (128usize, 8usize);

    for seed in [31u32, 37, 41, 43] {
        let logits = sample(n_expert, seed);
        let ours = llamd_models::route(&logits, n_used).unwrap();

        let bin = oracle_path().unwrap();
        let mut c = std::process::Command::new(&bin)
            .args(["top_k", &n_expert.to_string(), "1", &n_used.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let si = c.stdin.as_mut().unwrap();
            for v in &logits {
                si.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let out = c.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "oracle top_k failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let theirs: Vec<u32> = out
            .stdout
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32)
            .collect();

        let mut a = ours.experts.clone();
        let mut b = theirs.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(
            a, b,
            "seed {seed}: expert SETS differ\n  ours {a:?}\n  ggml {b:?}"
        );

        // Weights must still sum to one after renormalising the chosen few.
        let s: f32 = ours.weights.iter().sum();
        assert!(
            (s - 1.0).abs() < ATTENTION,
            "seed {seed}: weights sum to {s}, not 1"
        );
    }
    eprintln!("  expert selection: 8 of 128 identical to ggml across 4 seeds");
}

/// The quantized dot product, compared against ggml's own.
///
/// This is the comparison that decides whether LLAMD can ever match
/// llama.cpp at the token level. The f32 reference cannot: it performs
/// different arithmetic, and on a near-tied MoE routing decision that flips
/// which expert runs. This path performs the SAME arithmetic, so the bar is
/// much tighter — `REDUCTION`, not `QUANTIZED_MATMUL`.
#[test]
fn quantized_dot_matches_ggml() {
    if oracle_path().is_none() {
        skip(
            "quantized_dot_matches_ggml",
            "build tools/ggml-oracle first",
        );
        return;
    }
    use llamd_quant::{Q6K_BLOCK_BYTES, QK_K, dot_q6k_q8k, quantize_q8_k};

    for (blocks, rows, seed) in [(1usize, 1usize, 3u32), (4, 1, 5), (8, 3, 7)] {
        let n = blocks * QK_K;

        // Deterministic Q6_K bytes. Arbitrary but valid: every field is used,
        // so a mistake in unpacking shows up rather than cancelling.
        let mut w = vec![0u8; rows * blocks * Q6K_BLOCK_BYTES];
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(11);
        for b in w.iter_mut() {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (st >> 16) as u8;
        }
        // The f16 super-block scale must be a sane number, not a random bit
        // pattern that could be inf or NaN.
        for r in 0..rows * blocks {
            let d_off = r * Q6K_BLOCK_BYTES + QK_K / 2 + QK_K / 4 + QK_K / 16;
            let d = half::f16::from_f32(0.01 + (r as f32) * 0.003);
            w[d_off..d_off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
        }

        let x = sample(n, seed.wrapping_add(100));
        let a = quantize_q8_k(&x);

        let mut ours = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &w[r * blocks * Q6K_BLOCK_BYTES..(r + 1) * blocks * Q6K_BLOCK_BYTES];
            ours.push(dot_q6k_q8k(row, &a, n));
        }

        // The oracle takes raw bytes, so the weights go over as-is.
        let bin = oracle_path().unwrap();
        let mut c = std::process::Command::new(&bin)
            .args(["mul_mat_q6k", &n.to_string(), &rows.to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let si = c.stdin.as_mut().unwrap();
            si.write_all(&w).unwrap();
            for v in &x {
                si.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        let out = c.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "oracle mul_mat_q6k failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let theirs: Vec<f32> = out
            .stdout
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // REDUCTION: the same integer arithmetic in the same order should
        // agree to within a reordered-summation step. Needing more than this
        // would mean the quantizer or the unpacking differs.
        check(
            &format!("q6k·q8k n{n} rows{rows}"),
            &ours,
            &theirs,
            REDUCTION,
        );
    }
}
