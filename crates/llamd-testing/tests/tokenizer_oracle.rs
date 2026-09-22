//! Tokenization, validated against llama.cpp — with NO tolerance.
//!
//! This is the one comparison in the project where "close" is meaningless. A
//! single different id produces a different prompt, and every number measured
//! downstream is then answering a question nobody asked. So the assertion is
//! whole-sequence equality, and a mismatch prints both sequences and the
//! decoded pieces, because "sequences differ" is not a debuggable message.
//!
//!   LLAMD_TEST_MODEL=/path/to/model.gguf cargo test -p llamd-testing

use llamd_testing::{oracle_model, skip};
use std::process::Command;

/// Prompts chosen to exercise the rules that are easy to get wrong.
const CORPUS: &[&str] = &[
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "    four spaces then text",
    "it's a test, isn't it? we'll see — I'd say so.",
    "IT'S UPPERCASE AND it's lowercase",
    "2024 and 007 and 1.5e-10",
    "fn main() {\n    println!(\"hello\");\n}\n",
    "def f(x):\n\treturn x ** 2\n\n\n",
    "a\n\nb\r\nc",
    "path/to/file.rs:42:8",
    "SELECT * FROM t WHERE id = 1;",
    "日本語のテキストです",
    "Здравствуй, мир",
    "emoji: 🎉🔥 and combining: é ñ",
    "<|im_start|>user\nhi<|im_end|>",
    "",
    " ",
    "\n",
    "​zero-width space",
    "mixed  　unicode\u{3000}spaces",
    "very_long_identifier_name_that_will_be_split_by_bpe_somewhere",
    "0x7FFFFFFF & 0b1010 | 0o777",
];

/// Ask llama.cpp to tokenize, via gguf-py's vocabulary plus the model's own
/// rules. We shell out to the reference rather than reimplementing it.
fn oracle_tokenize(model: &std::path::Path, texts: &[&str]) -> Option<Vec<Vec<u32>>> {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .to_path_buf();

    // llama-tokenize is the reference CLI. Built from the pinned checkout, it
    // is the same code path llama.cpp itself uses at inference time.
    let bin = [
        "llama.cpp/build/bin/llama-tokenize",
        "llama.cpp/build/bin/llama-cli",
    ]
    .iter()
    .map(|p| repo.join(p))
    .find(|p| p.is_file());
    let Some(bin) = bin else {
        skip(
            "tokenizer_oracle",
            "llama-tokenize not built — cmake -B llama.cpp/build llama.cpp && \
             cmake --build llama.cpp/build --target llama-tokenize",
        );
        return None;
    };

    let mut all = Vec::new();
    for t in texts {
        // llama-tokenize treats `--prompt ""` as "no prompt given" and exits
        // with an error. An empty input has no tokens by definition, so the
        // answer is known without asking — but it is still compared, because
        // our encoder must agree that it produces none.
        if t.is_empty() {
            all.push(Vec::new());
            continue;
        }
        let out = Command::new(&bin)
            .arg("-m")
            .arg(model)
            .arg("--prompt")
            .arg(t)
            .arg("--ids")
            .arg("--no-bos")
            .output()
            .ok()?;
        if !out.status.success() {
            // The oracle is PRESENT and errored. That is a failure, not a
            // reason to skip: skipping here made this test report ok while
            // comparing nothing, which is precisely the vacuous pass the
            // harness exists to prevent.
            panic!(
                "oracle failed on {t:?}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        let s = String::from_utf8_lossy(&out.stdout);
        // `--ids` prints a bracketed list on the last non-empty line.
        let line = s
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('['))
            .unwrap_or_else(|| panic!("oracle printed no id list for {t:?}:\n{s}"));
        let ids: Vec<u32> = line
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        all.push(ids);
    }
    Some(all)
}

#[test]
fn tokenization_matches_llama_cpp_exactly() {
    let Some(model) = oracle_model() else {
        skip(
            "tokenization_matches_llama_cpp_exactly",
            "set LLAMD_TEST_MODEL",
        );
        return;
    };
    let g = llamd_gguf::Gguf::open(&model).expect("open model");
    let v = llamd_tokenizer::Vocab::from_gguf(&g).expect("load vocabulary");
    eprintln!("  vocabulary: {} tokens", v.len());

    let Some(oracle) = oracle_tokenize(&model, CORPUS) else {
        return;
    };

    let mut failures = Vec::new();
    for (text, want) in CORPUS.iter().zip(&oracle) {
        let got = v.encode(text, true);
        if &got != want {
            // A bare "sequences differ" is not debuggable. Show the first
            // divergence and what both sides made of it.
            let at = got
                .iter()
                .zip(want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            failures.push(format!(
                "\n  input   {text:?}\n  ours    {got:?}\n  oracle  {want:?}\n  \
                 first difference at index {at}: ours {:?} ({:?}) vs oracle {:?} ({:?})",
                got.get(at),
                got.get(at).and_then(|&i| v.text_of(i)),
                want.get(at),
                want.get(at).and_then(|&i| v.text_of(i)),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} prompts tokenized differently:{}",
        failures.len(),
        CORPUS.len(),
        failures.join("")
    );
    eprintln!(
        "  all {} prompts tokenized identically to llama.cpp",
        CORPUS.len()
    );
}

#[test]
fn decode_round_trips_encode() {
    let Some(model) = oracle_model() else {
        skip("decode_round_trips_encode", "set LLAMD_TEST_MODEL");
        return;
    };
    let g = llamd_gguf::Gguf::open(&model).unwrap();
    let v = llamd_tokenizer::Vocab::from_gguf(&g).unwrap();

    // Round-tripping does not need the oracle: it is a property of our own
    // implementation, and a failure means encode and decode disagree about
    // the byte mapping.
    for t in CORPUS {
        let ids = v.encode(t, true);
        let back = v.decode(&ids);
        assert_eq!(
            &back, t,
            "round-trip failed\n  in  {t:?}\n  out {back:?}\n  ids {ids:?}"
        );
    }
    eprintln!("  {} prompts round-tripped", CORPUS.len());
}
