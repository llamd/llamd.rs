//! How the quantized-path gap grows with depth.
//!
//! Not a pass/fail test of the model so much as a measurement of the error
//! budget: `l_out-N` is dumped for every layer, so the gap can be watched
//! accumulate. Smooth growth is quantization; a jump is a bug, and telling
//! those apart is the whole reason to plot it rather than assert on the end.

use llamd_cpu::CpuBackend;
use llamd_models::Model;
use llamd_runtime::{KvCache, embed, forward_layer, moe_ffn};
use llamd_testing::{oracle_model, skip};
use std::collections::HashMap;

fn oracle_l_out(model: &std::path::Path, prompt: &str) -> Option<HashMap<String, f64>> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?;
    let bin = repo.join("llama.cpp/build/bin/llama-eval-callback");
    if !bin.is_file() {
        skip("layer_growth", "llama-eval-callback not built");
        return None;
    }
    let out = std::process::Command::new(&bin)
        .args(["-m", model.to_str()?, "-p", prompt, "-ngl", "0"])
        .output()
        .ok()?;
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    let mut map = HashMap::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        if let Some(rest) = line.split("common_debug_cb_eval: ").nth(1) {
            let n = rest.split(" = ").next().unwrap_or("").trim().to_string();
            pending = n.starts_with("l_out-").then_some(n);
        } else if let Some(v) = line.trim().strip_prefix("sum = ") {
            if let (Some(n), Ok(s)) = (pending.take(), v.trim().parse::<f64>()) {
                map.entry(n).or_insert(s);
            }
        }
    }
    Some(map)
}

#[test]
#[ignore = "diagnostic: runs the whole model on the CPU reference"]
fn error_grows_smoothly_with_depth() {
    let Some(path) = oracle_model() else {
        skip("error_grows_smoothly_with_depth", "set LLAMD_TEST_MODEL");
        return;
    };
    let Some(oracle) = oracle_l_out(&path, "Hi") else {
        return;
    };

    let be = CpuBackend::new();
    let model = Model::open(&path).unwrap();
    let vocab = llamd_tokenizer::Vocab::from_gguf(model.gguf()).unwrap();
    let ids = vocab.encode("Hi", false);
    let h = model.hparams.clone();
    let mut kv = KvCache::new(h.n_layer, h.kv_dim(), 8).unwrap();
    let mut x = embed(&be, &model, ids[0]).unwrap();

    eprintln!("  layer   ours sum      oracle sum    relative");
    let mut prev = 0.0f32;
    for il in 0..h.n_layer {
        let t = forward_layer(&be, &model, il, &x, 0, &mut kv).unwrap();
        let ffn = moe_ffn(&be, &model, il, &t.ffn_norm, &t.routing).unwrap();
        x = t.ffn_inp.iter().zip(&ffn).map(|(a, b)| a + b).collect();

        if let Some(&want) = oracle.get(&format!("l_out-{il}")) {
            let got: f64 = x.iter().map(|&v| v as f64).sum();
            let scale = x.iter().map(|v| v.abs() as f64).sum::<f64>().max(1.0);
            let rel = ((got - want).abs() / scale) as f32;
            if !(3..41).contains(&il) {
                eprintln!("  {il:>5}   {got:>11.4}   {want:>11.4}   {rel:.3e}");
            }
            // A single layer must not multiply the error: that would be a bug
            // rather than accumulation.
            // A jump rather than a drift is the signature of a discrete
            // event — in this model, a routing decision flipping.
            if il > 0 && prev > 1e-6 && rel > prev * 4.0 + 1e-3 {
                eprintln!("  ^^ layer {il}: JUMP from {prev:.3e} to {rel:.3e}");
            }
            prev = rel;
        }
    }
    eprintln!(
        "  final relative gap after {} layers: {prev:.3e}",
        h.n_layer
    );
}
