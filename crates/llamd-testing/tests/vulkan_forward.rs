//! A whole token computed on Vulkan, against the CPU reference and llama.cpp.
//!
//! THE CLAIM THIS TEST MAKES. `VulkanModel::forward_batch` computes a token with
//! every operation on the Vulkan device — weights resident, activations never
//! crossing the bus, one submission per token — and emits the same token as the
//! ROCm path and the CPU reference, which is the path that agrees with llama.cpp.
//!
//! Asserted on the TOKEN, exactly. The logits will not match bit for bit: the
//! subgroup width differs (RADV gives 64 where the HIP kernels use 32), so the
//! reduction tree has a different shape and 48 layers of reordered float
//! addition accumulate. What must match is the decision.
//!
//!   LLAMD_TEST_MODEL=... cargo test --release -p llamd-testing \
//!     --test vulkan_forward -- --ignored --nocapture

use llamd_models::Model;
use llamd_runtime::VulkanModel;
use llamd_testing::{oracle_model, skip};

/// (prompt, prompt token, llama.cpp's greedy next token)
const CASES: &[(&str, u32, u32)] = &[
    ("Hi", 13048, 525),
    ("The", 785, 525),
    ("def", 750, 374),
    ("Python", 30280, 25),
    ("1", 16, 13),
];

fn open(max_kv: usize, max_seq: usize) -> Option<(Model, VulkanModel)> {
    let path = oracle_model()?;
    if !llamd_vulkan::available() {
        return None;
    }
    let model = Model::open(&path).ok()?;
    match VulkanModel::load(&model, max_kv, max_seq) {
        Ok(v) => Some((model, v)),
        Err(e) => {
            eprintln!("  could not load on Vulkan: {e}");
            None
        }
    }
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn vulkan_emits_the_same_tokens_as_llama_cpp() {
    let t0 = std::time::Instant::now();
    let Some((model, mut vk)) = open(64, 4) else {
        skip(
            "vulkan_emits_the_same_tokens_as_llama_cpp",
            "no model or no Vulkan",
        );
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    eprintln!(
        "  uploaded {:.2} GiB of attention weights in {:.1} s on {}",
        vk.weight_bytes() as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f32(),
        llamd_vulkan::device_info()
            .map(|i| i.name)
            .unwrap_or_default()
    );

    let mut wrong = Vec::new();
    for &(prompt, tok_in, want) in CASES {
        let t = std::time::Instant::now();
        let got = vk.forward_batch(&[tok_in], &[0]).expect("vulkan forward");
        let ms = t.elapsed().as_secs_f32() * 1e3;
        let ok = got[0] == want;
        eprintln!(
            "  {:<9} vulkan {:<6} {:<10} llama.cpp {:<6} {:<10} {}  {ms:>6.1} ms",
            format!("{prompt:?}"),
            got[0],
            format!("{:?}", vocab.text_of(got[0]).unwrap_or("?")),
            want,
            format!("{:?}", vocab.text_of(want).unwrap_or("?")),
            if ok { "ok" } else { "MISMATCH" }
        );
        if !ok {
            wrong.push(format!(
                "{prompt:?}: llama.cpp {want} {:?}, vulkan {} {:?}",
                vocab.text_of(want).unwrap_or("?"),
                got[0],
                vocab.text_of(got[0]).unwrap_or("?")
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} prompts disagree:\n  {}",
        wrong.len(),
        CASES.len(),
        wrong.join("\n  ")
    );
    eprintln!("  all {} prompts: Vulkan == llama.cpp", CASES.len());
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn vulkan_batching_matches_its_own_single_sequence_path() {
    // Holds the kernel fixed and varies only the batch width, so a failure is a
    // batching bug rather than a numerical difference between two kernels.
    let Some((model, mut vk)) = open(16, 4) else {
        skip("vulkan_batching_matches", "no model or no Vulkan");
        return;
    };
    let _ = &model;
    let toks: Vec<u32> = CASES.iter().map(|&(_, t, _)| t).collect();
    let mut want = Vec::new();
    for &t in &toks {
        want.push(vk.forward_batch(&[t], &[0]).expect("nseq=1")[0]);
    }
    for nseq in [2usize, 4] {
        let got = vk
            .forward_batch(&toks[..nseq], &vec![0usize; nseq])
            .expect("batch");
        assert_eq!(&got, &want[..nseq], "batch {nseq} differs from nseq=1");
        eprintln!("  batch {nseq}: matches the single-sequence path");
    }
}

#[test]
#[ignore = "uploads 23 GiB to the device; needs a free GPU and the real model"]
fn vulkan_generation_throughput() {
    let Some((model, mut vk)) = open(68, 8) else {
        skip("vulkan_generation_throughput", "no model or no Vulkan");
        return;
    };
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).expect("vocab");
    const N: usize = 64;

    // Warm first: a cold pass pays pipeline creation and first-touch costs a
    // steady state does not.
    let mut tk = vec![13048u32];
    for pos in 0..N {
        tk = vk.forward_batch(&tk, &[pos]).expect("warm");
    }

    tk = vec![13048u32];
    let mut text = String::new();
    let t = std::time::Instant::now();
    for pos in 0..N {
        tk = vk.forward_batch(&tk, &[pos]).expect("timed");
        text.push_str(vocab.text_of(tk[0]).unwrap_or("?"));
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / N as f64;
    eprintln!("\n  batch 1: {ms:.2} ms/token -> {:.2} tok/s", 1000.0 / ms);
    eprintln!("  generated {text:?}");

    eprintln!(
        "\n  {:<6} {:>10} {:>12} {:>14}",
        "batch", "ms/pass", "ms/token", "tok/s total"
    );
    for nseq in [1usize, 2, 4, 8] {
        let mut toks: Vec<u32> = (0..nseq).map(|i| CASES[i % CASES.len()].1).collect();
        for pos in 0..8 {
            toks = vk.forward_batch(&toks, &vec![pos; nseq]).expect("warm");
        }
        let t = std::time::Instant::now();
        for pos in 0..N {
            toks = vk.forward_batch(&toks, &vec![pos; nseq]).expect("timed");
        }
        let per_pass = t.elapsed().as_secs_f64() * 1e3 / N as f64;
        eprintln!(
            "  {nseq:<6} {per_pass:>10.2} {:>12.3} {:>14.2}",
            per_pass / nseq as f64,
            1000.0 / (per_pass / nseq as f64)
        );
    }
    eprintln!("  llama.cpp: 94.4-95.6 single stream, 145.8/229.5/290.9 at batch 2/4/8");
}
