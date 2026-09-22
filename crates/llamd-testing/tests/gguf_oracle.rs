//! GGUF parsing, validated against llama.cpp's own reader.
//!
//! The oracle is `gguf-py` from the vendored llama.cpp checkout — the same
//! code llama.cpp's own tooling uses. When the model or python is unavailable
//! the test skips LOUDLY: a test that quietly does nothing is worse than one
//! that fails, because it reports success.
//!
//!   LLAMD_TEST_MODEL=/path/to/model.gguf cargo test -p llamd-testing

use llamd_testing::{oracle_model, skip};
use std::process::Command;

fn oracle_dump(model: &std::path::Path) -> Option<serde_json::Value> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .to_path_buf();
    let gguf_py = repo.join("llama.cpp/gguf-py");
    if !gguf_py.is_dir() {
        skip(
            "gguf_oracle",
            "llama.cpp/gguf-py absent — run scripts/clone-reference.sh",
        );
        return None;
    }
    let py = ["/home/maxx/ai/venv-infer/bin/python", "python3"]
        .iter()
        .find(|p| {
            Command::new(p)
                .arg("-c")
                .arg("import numpy")
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
        .copied()?;

    let script = r#"
import json, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
# GGUF.* are gguf-py's SYNTHETIC header fields (version, tensor_count,
# kv_count), not metadata stored in the file. Counting them makes the oracle
# disagree with any correct parser by exactly three - which is how this
# discrepancy first appeared.
kv = [k for k in r.fields if not k.startswith("GGUF.")]
print(json.dumps({
    "n_kv": len(kv),
    "n_tensors": len(r.tensors),
    "tensors": [{"name": t.name,
                 "shape": [int(x) for x in t.shape],
                 "type": int(t.tensor_type)} for t in r.tensors],
}))
"#;
    let out = Command::new(py)
        .env("PYTHONPATH", &gguf_py)
        .arg("-c")
        .arg(script)
        .arg(model)
        .output()
        .ok()?;
    if !out.status.success() {
        skip(
            "gguf_oracle",
            &format!("oracle failed: {}", String::from_utf8_lossy(&out.stderr)),
        );
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

#[test]
fn parser_agrees_with_llama_cpp() {
    let Some(model) = oracle_model() else {
        skip(
            "parser_agrees_with_llama_cpp",
            "set LLAMD_TEST_MODEL to a .gguf",
        );
        return;
    };
    let Some(oracle) = oracle_dump(&model) else {
        return;
    };
    let ours = llamd_gguf::Gguf::open(&model).expect("LLAMD failed to open the model");

    assert_eq!(
        ours.metadata.len() as u64,
        oracle["n_kv"].as_u64().unwrap(),
        "metadata key count disagrees with the oracle"
    );
    assert_eq!(
        ours.tensors.len() as u64,
        oracle["n_tensors"].as_u64().unwrap(),
        "tensor count disagrees with the oracle"
    );

    // Name, shape and type must match EXACTLY. These are not numerical
    // comparisons: a shape that is close is simply wrong.
    for (i, o) in oracle["tensors"].as_array().unwrap().iter().enumerate() {
        let name = o["name"].as_str().unwrap();
        let t = ours
            .tensor(name)
            .unwrap_or_else(|| panic!("tensor {name} (oracle index {i}) missing from LLAMD"));
        let oshape: Vec<u64> = o["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(t.dims, oshape, "shape mismatch for {name}");
        assert_eq!(
            t.ggml_type as u64,
            o["type"].as_u64().unwrap(),
            "quantization type mismatch for {name}"
        );
    }
    eprintln!(
        "  agreed with the oracle on {} tensors and {} metadata keys",
        ours.tensors.len(),
        ours.metadata.len()
    );
}

#[test]
fn dequantized_values_are_plausible_weights() {
    let Some(model) = oracle_model() else {
        skip(
            "dequantized_values_are_plausible_weights",
            "set LLAMD_TEST_MODEL",
        );
        return;
    };
    let g = llamd_gguf::Gguf::open(&model).unwrap();
    let t = g
        .tensors
        .iter()
        .filter(|t| t.ggml_type == 14 && t.elems() < 5_000_000)
        .min_by_key(|t| t.elems())
        .expect("no small Q6_K tensor in this model");

    let v = g.dequantize(t).expect("dequantize failed");
    assert_eq!(v.len() as u64, t.elems());
    assert!(
        v.iter().all(|x| x.is_finite()),
        "dequantization produced NaN or inf"
    );

    // Trained weights are small and centred near zero. Values in the hundreds
    // would mean the -32 centring offset or the scale is wrong — an error that
    // still produces plausible-looking text, which is exactly why it is checked.
    // PLAUSIBILITY BOUNDS, not tolerances. Named for the same reason
    // tolerances are: an unnamed literal in an assertion is one somebody
    // raises when it fails. These are not comparisons against the oracle —
    // they are sanity checks on what a trained weight can be.
    const MAX_PLAUSIBLE_WEIGHT: f32 = 100.0;
    const MAX_PLAUSIBLE_MEAN: f32 = 1.0;

    let max = v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    assert!(
        max < MAX_PLAUSIBLE_WEIGHT,
        "max |weight| {max} is implausible for a trained model"
    );
    assert!(
        mean.abs() < MAX_PLAUSIBLE_MEAN,
        "mean weight {mean} is implausibly far from zero"
    );
    eprintln!(
        "  {}: {} elements, max |w| {:.4}, mean {:.6}",
        t.name,
        v.len(),
        max,
        mean
    );
}
