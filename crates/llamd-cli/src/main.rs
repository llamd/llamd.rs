//! `llamd` — the command-line entry point.
//!
//! Only one subcommand so far, and deliberately so: `inspect` proves the GGUF
//! parser works against a real model file, which is the milestone the porting
//! matrix puts first.

#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("experts") => match args.get(2) {
            Some(path) => {
                let verify = args
                    .iter()
                    .position(|a| a == "--verify")
                    .and_then(|i| args.get(i + 1))
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0usize);
                if let Err(e) = experts(path, verify) {
                    eprintln!("llamd: {e}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("usage: llamd experts <model.gguf> [--verify N]");
                std::process::exit(2);
            }
        },
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
                "llamd {}\n\nusage:\n  llamd inspect <model.gguf>\n  llamd experts <model.gguf> [--verify N]",
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

fn rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

/// Open a MoE model WITHOUT mapping it, build the expert index, and report
/// what out-of-core loading will face: sizes, formats, alignment, and the
/// memory the open itself cost. `--verify N` preads N experts spread over the
/// model and reports throughput — a cold-ish read figure, not a benchmark.
fn experts(path: &str, verify: usize) -> llamd_core::Result<()> {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    let rss0 = rss_mib();
    let t0 = std::time::Instant::now();
    let m = llamd_models::Model::open_unmapped(path)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    let idx = m.expert_index()?;
    let index_ms = t1.elapsed().as_secs_f64() * 1e3;
    let rss1 = rss_mib();

    let file = m.gguf().file_size() as u64;
    let experts = idx.total_bytes();
    let h = &m.hparams;
    println!("  file            {path}");
    println!(
        "  size            {:.2} GiB (opened unmapped)",
        file as f64 / GIB
    );
    println!("  architecture    {}", h.arch.key());
    println!(
        "  layers x experts {} x {} (top-{} routed)",
        h.n_layer, h.n_expert, h.n_expert_used
    );
    println!("  open / index    {open_ms:.1} ms / {index_ms:.2} ms");
    println!("  RSS             {rss0:.1} -> {rss1:.1} MiB");
    println!(
        "  routed experts  {:.2} GiB ({:.1}% of file)",
        experts as f64 / GIB,
        100.0 * experts as f64 / file as f64
    );
    println!(
        "  everything else {:.2} GiB (shared weights + header: the always-resident part)",
        (file - experts) as f64 / GIB
    );
    println!(
        "  one expert      {:.2} MiB (gate+up+down, largest layer)",
        idx.max_expert_bytes() as f64 / MIB
    );
    let per_token = idx.max_expert_bytes() * (h.n_expert_used * h.n_layer) as u64;
    println!(
        "  per token, all misses  {:.2} GiB of expert reads ({} layers x top-{})",
        per_token as f64 / GIB,
        h.n_layer,
        h.n_expert_used
    );
    println!("  expert formats  {:?}", idx.qtypes());
    println!(
        "  alignment       {:?}  (bytes -> extents)",
        idx.alignment_census()
    );

    if verify > 0 {
        let total = h.n_layer * h.n_expert * 3;
        let step = (total / verify).max(1);
        let mut bytes = 0u64;
        let t = std::time::Instant::now();
        for k in (0..total).step_by(step).take(verify) {
            let (l, rest) = (k / (h.n_expert * 3), k % (h.n_expert * 3));
            let (e, p) = (rest / 3, llamd_models::Proj::ALL[rest % 3]);
            let ext = idx.extent(l, e, p)?;
            bytes += m.gguf().read_extent(ext)?.len() as u64;
        }
        let s = t.elapsed().as_secs_f64();
        println!(
            "  verify          {verify} extents, {:.1} MiB in {:.3} s = {:.0} MiB/s (buffered pread)",
            bytes as f64 / MIB,
            s,
            bytes as f64 / MIB / s
        );
        println!("  RSS after       {:.1} MiB", rss_mib());
    }
    Ok(())
}
