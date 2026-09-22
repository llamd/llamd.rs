//! `llamd` — the command-line entry point.
//!
//! Only one subcommand so far, and deliberately so: `inspect` proves the GGUF
//! parser works against a real model file, which is the milestone the porting
//! matrix puts first.

#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("inspect") => match args.get(2) {
            Some(path) => {
                if let Err(e) = inspect(path) {
                    eprintln!("llamd: {e}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("usage: llamd inspect <model.gguf>");
                std::process::exit(2);
            }
        },
        _ => {
            eprintln!(
                "llamd {}\n\nusage:\n  llamd inspect <model.gguf>",
                env!("CARGO_PKG_VERSION")
            );
            std::process::exit(2);
        }
    }
}

fn inspect(path: &str) -> llamd_core::Result<()> {
    let t0 = std::time::Instant::now();
    let g = llamd_gguf::Gguf::open(path)?;
    let open_ms = t0.elapsed().as_millis();

    println!("  file          {path}");
    println!(
        "  size          {:.2} GiB",
        g.file_size() as f64 / (1 << 30) as f64
    );
    println!("  gguf version  {}", g.version);
    println!("  alignment     {}", g.alignment);
    println!("  data offset   {}", g.data_offset);
    println!("  metadata      {} keys", g.metadata.len());
    println!("  tensors       {}", g.tensors.len());
    println!("  parsed in     {open_ms} ms");

    println!(
        "\n  architecture  {}",
        g.architecture().unwrap_or("(absent)")
    );

    // The keys that decide how the model runs. Printed first because they are
    // what a reader actually needs; the full dump is noise by comparison.
    println!("\n  key metadata");
    let arch = g.architecture().unwrap_or("").to_string();
    for suffix in [
        "block_count",
        "context_length",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "rope.freq_base",
        "attention.layer_norm_rms_epsilon",
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
    ] {
        let key = format!("{arch}.{suffix}");
        if let Some(v) = g.get(&key) {
            println!("    {:<44} {}", key, v.summary());
        }
    }
    for key in ["general.name", "general.file_type", "tokenizer.ggml.model"] {
        if let Some(v) = g.get(key) {
            println!("    {:<44} {}", key, v.summary());
        }
    }

    // Tensor type census: which quantizations the file actually uses.
    let mut by_type: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
    for t in &g.tensors {
        let e = by_type.entry(t.ggml_type).or_default();
        e.0 += 1;
        e.1 += t.elems();
    }
    println!("\n  tensor types");
    for (ty, (count, elems)) in &by_type {
        let name = llamd_quant::QuantType::from_ggml(*ty)
            .map(|q| format!("{q:?}"))
            .unwrap_or_else(|_| format!("ggml_type {ty} (unsupported)"));
        println!(
            "    {:<16} {:>5} tensors  {:>14} elements",
            name, count, elems
        );
    }

    println!("\n  first tensors");
    for t in g.tensors.iter().take(5) {
        println!("    {:<40} {:?} type={}", t.name, t.dims, t.ggml_type);
    }
    Ok(())
}
