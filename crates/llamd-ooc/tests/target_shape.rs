//! The acceptance criterion at the TARGET's shape, without the target's bytes.
//!
//! A sparse GGUF with the full tensor table of Qwen3-Coder-480B-A35B at Q8_0
//! (~510 GB apparent size, a few MB on disk) must open without loading, index
//! every expert, and produce a memory plan for 32 GB VRAM / 16 GB RAM that
//! never exceeds either budget. What this CANNOT show — real read throughput
//! and real token rates — needs the real file on real storage.

use llamd_models::{Model, synthetic::SyntheticMoe};
use llamd_ooc::{BudgetConfig, MemoryPlan, ModelFacts, Tier};

fn rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 1024
}

#[test]
fn a_480b_shaped_model_opens_indexes_and_plans_within_32g_vram_16g_ram() {
    let f = SyntheticMoe::qwen3_coder_480b_q8_0()
        .write("480b")
        .expect("sparse write");
    let before = rss();
    let m = Model::open_unmapped(&f.0).expect("open");
    let idx = m.expert_index().expect("index");
    let grew = rss().saturating_sub(before);

    let file = m.gguf().file_size() as u64;
    assert!(file > 450 << 30, "apparent size {file}");
    assert_eq!(idx.n_layer * idx.n_expert, 62 * 160);
    // One expert at Q8_0: 3 x 6144 x 2560 x 34/32 bytes.
    assert_eq!(idx.max_expert_bytes(), 3 * 6144 * 2560 / 32 * 34);
    assert!(
        grew < 64 << 20,
        "RSS grew {grew} opening a {file}-byte model"
    );

    let cfg = BudgetConfig::target_32g_16g();
    let plan = MemoryPlan::new(&cfg, &ModelFacts::from_model(&m, &idx).unwrap()).expect("plan");
    eprintln!("{}", plan.report());
    assert!(!plan.fits_resident);
    assert!(plan.used(Tier::Vram) <= cfg.vram_total);
    assert!(plan.used(Tier::Ram) <= cfg.ram_total);
    assert!(plan.vram_slots >= 8, "need top-k resident");
}
