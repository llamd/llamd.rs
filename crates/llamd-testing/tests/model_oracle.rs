//! Loading the real Qwen3-Coder-30B-A3B model.
//!
//! These tests run against 23.37 GiB on disk. They check the things that must
//! be true before a forward pass can be attempted at all: that every tensor
//! the architecture names exists, that the shapes are what the hyperparameters
//! imply, and that a single expert can be extracted without materialising its
//! 128 siblings.
//!
//!   LLAMD_TEST_MODEL=/path/to/model.gguf cargo test -p llamd-testing

use llamd_models::{Arch, Model};
use llamd_testing::{oracle_model, skip};

#[test]
fn loads_the_real_model_and_finds_every_tensor() {
    let Some(path) = oracle_model() else {
        skip(
            "loads_the_real_model_and_finds_every_tensor",
            "set LLAMD_TEST_MODEL",
        );
        return;
    };
    let t0 = std::time::Instant::now();
    // Model::open verifies every named tensor is present, so this succeeding
    // IS the assertion that naming is right for all 48 layers.
    let m = Model::open(&path).expect("failed to open the model");
    let ms = t0.elapsed().as_millis();

    let h = &m.hparams;
    assert_eq!(h.arch, Arch::Qwen3Moe);
    assert_eq!(h.n_layer, 48);
    assert_eq!(h.n_embd, 2048);
    assert_eq!(h.n_head, 32);
    assert_eq!(h.n_head_kv, 4);
    assert_eq!(h.n_head_dim, 128, "head_dim is 128, not n_embd/n_head = 64");
    assert_eq!(h.n_expert, 128);
    assert_eq!(h.n_expert_used, 8);
    assert_eq!(h.n_ff_expert, 768);
    assert_eq!(h.n_vocab, 151936);
    assert_eq!(h.n_ctx_train, 262144);
    assert_eq!(h.rope_freq_base, 10_000_000.0);
    assert_eq!(m.layers.len(), 48);

    // Qwen3 has QK-norm; its absence would mean we loaded a Qwen2-shaped model.
    assert!(
        m.layers[0].attn_q_norm.is_some(),
        "Qwen3 must have attn_q_norm"
    );
    assert!(m.layers[0].attn_k_norm.is_some());
    assert!(
        m.layers[0].ffn_gate_exps.is_some(),
        "MoE must have expert tensors"
    );
    assert!(
        m.layers[0].ffn_gate.is_none(),
        "a MoE layer has no dense ffn_gate"
    );

    eprintln!(
        "  loaded in {ms} ms: {:.2}B params over {} tensors",
        m.n_params() as f64 / 1e9,
        m.gguf().tensors.len()
    );
}

#[test]
fn tensor_shapes_match_the_hyperparameters() {
    let Some(path) = oracle_model() else {
        skip(
            "tensor_shapes_match_the_hyperparameters",
            "set LLAMD_TEST_MODEL",
        );
        return;
    };
    let m = Model::open(&path).unwrap();
    let h = &m.hparams;
    let l = &m.layers[0];

    // The shapes the hyperparameters IMPLY must be the shapes on disk. A
    // mismatch means a hyperparameter was misread, which would otherwise show
    // up as a confusing matmul failure deep in a forward pass.
    let dims = |n: &str| {
        m.info(n)
            .unwrap()
            .dims
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        dims(&l.attn_q),
        vec![h.n_embd, h.q_dim()],
        "q: [n_embd, n_head*head_dim]"
    );
    assert_eq!(
        dims(&l.attn_k),
        vec![h.n_embd, h.kv_dim()],
        "k: [n_embd, n_head_kv*head_dim]"
    );
    assert_eq!(dims(&l.attn_v), vec![h.n_embd, h.kv_dim()]);
    assert_eq!(dims(&l.attn_out), vec![h.q_dim(), h.n_embd]);
    assert_eq!(dims(&l.attn_norm), vec![h.n_embd]);
    assert_eq!(
        dims(l.attn_q_norm.as_ref().unwrap()),
        vec![h.n_head_dim],
        "QK-norm is per head_dim"
    );

    // The router projects n_embd to one logit per expert.
    assert_eq!(
        dims(l.ffn_gate_inp.as_ref().unwrap()),
        vec![h.n_embd, h.n_expert]
    );
    // Experts: [n_embd, n_ff_expert, n_expert], expert on the slowest axis.
    assert_eq!(
        dims(l.ffn_gate_exps.as_ref().unwrap()),
        vec![h.n_embd, h.n_ff_expert, h.n_expert]
    );
    assert_eq!(
        dims(l.ffn_down_exps.as_ref().unwrap()),
        vec![h.n_ff_expert, h.n_embd, h.n_expert],
        "down projects back: [n_ff_expert, n_embd, n_expert]"
    );
    eprintln!("  all shapes agree with the hyperparameters");
}

#[test]
fn one_expert_can_be_read_without_the_other_127() {
    let Some(path) = oracle_model() else {
        skip(
            "one_expert_can_be_read_without_the_other_127",
            "set LLAMD_TEST_MODEL",
        );
        return;
    };
    let m = Model::open(&path).unwrap();
    let h = &m.hparams;
    let name = m.layers[0].ffn_gate_exps.clone().unwrap();

    let t0 = std::time::Instant::now();
    let e0 = m.expert(&name, 0).expect("expert 0");
    let ms = t0.elapsed().as_millis();

    assert_eq!(e0.shape().dims(), &[h.n_embd, h.n_ff_expert]);
    assert!(
        e0.data().iter().all(|v| v.is_finite()),
        "expert weights must be finite"
    );

    // Different experts must give different weights — if slicing ignored the
    // index every expert would be expert 0, and the model would still run.
    let e7 = m.expert(&name, 7).unwrap();
    assert_ne!(
        e0.data()[..64],
        e7.data()[..64],
        "experts 0 and 7 are identical"
    );

    // Out of range must be an error, not a wrap-around to a real expert.
    assert!(m.expert(&name, h.n_expert).is_err());

    let max = e0.data().iter().fold(0.0f32, |a, b| a.max(b.abs()));
    eprintln!(
        "  expert 0: {} elements in {ms} ms, max |w| {:.4} (whole tensor would be {}x)",
        e0.elems(),
        max,
        h.n_expert
    );
}
