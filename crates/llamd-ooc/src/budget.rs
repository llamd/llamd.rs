//! The memory plan: where every byte of VRAM and RAM goes, decided up front.
//!
//! FIXED COSTS FIRST, EXPERTS GET THE REST. Shared weights, KV cache,
//! activations, staging and reserves are not negotiable — a token cannot be
//! computed without them — so they are subtracted first. Whatever remains in
//! VRAM becomes expert SLOTS, and whatever remains in RAM becomes the expert
//! CACHE. If the fixed costs alone exceed a tier, the plan is refused with a
//! breakdown: that is a configuration the engine cannot run, and finding out
//! at plan time beats finding out from the OOM killer.
//!
//! THE FLOOR. Computing one MoE layer needs its `top_k` experts resident at
//! once, so fewer than `top_k` VRAM slots is refused too. More slots than that
//! is what buys cache hits.
//!
//! ESTIMATES ARE LABELLED. Shared weights and expert sizes are exact (from the
//! file's tensor table). KV is exact for the f32 layout `VulkanModel` uses.
//! Activations are an ESTIMATE from the same formulas `VulkanModel` allocates
//! its scratch with, and can be overridden.

use llamd_core::{Error, Result};
use llamd_models::{ExpertIndex, Model};

use crate::size::fmt_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    Vram,
    Ram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// Attention, norms, router, embeddings, output — every non-expert tensor.
    SharedWeights,
    KvCache,
    Activations,
    /// Upload staging (VRAM side) or the aligned read-buffer pool (RAM side).
    Staging,
    /// Driver / fragmentation headroom on the card.
    DriverReserve,
    /// RAM left to the OS, desktop and everything that is not llamd.
    OsReserve,
    /// Parsed header (vocabulary), expert index, router statistics.
    Metadata,
    /// VRAM expert slots — active and hot experts.
    ExpertSlots,
    /// RAM expert cache — warm experts.
    ExpertCache,
}

impl Category {
    pub const ALL: [Category; 9] = [
        Category::SharedWeights,
        Category::KvCache,
        Category::Activations,
        Category::Staging,
        Category::DriverReserve,
        Category::OsReserve,
        Category::Metadata,
        Category::ExpertSlots,
        Category::ExpertCache,
    ];
    pub fn index(self) -> usize {
        self as usize
    }
}

/// What the plan needs to know about the model. Exact where the file says so.
#[derive(Debug, Clone)]
pub struct ModelFacts {
    pub n_layer: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub kv_dim: usize,
    pub q_dim: usize,
    pub n_ff_expert: usize,
    /// Every non-expert tensor's bytes.
    pub shared_bytes: u64,
    /// The largest single expert (gate + up + down) — one slot's size.
    pub max_expert_bytes: u64,
    /// All routed experts, all layers.
    pub expert_bytes_total: u64,
    /// Header + tensor table, roughly what the parsed metadata costs.
    pub header_bytes: u64,
}

impl ModelFacts {
    pub fn from_model(m: &Model, idx: &ExpertIndex) -> Result<Self> {
        let g = m.gguf();
        let mut tensor_bytes = 0u64;
        for t in &g.tensors {
            tensor_bytes += g.tensor_extent(t)?.len;
        }
        let experts = idx.total_bytes();
        let h = &m.hparams;
        Ok(ModelFacts {
            n_layer: h.n_layer,
            n_expert: h.n_expert,
            n_expert_used: h.n_expert_used,
            n_embd: h.n_embd,
            n_head: h.n_head,
            n_vocab: h.n_vocab,
            kv_dim: h.kv_dim(),
            q_dim: h.q_dim(),
            n_ff_expert: h.n_ff_expert,
            shared_bytes: tensor_bytes.saturating_sub(experts),
            max_expert_bytes: idx.max_expert_bytes(),
            expert_bytes_total: experts,
            header_bytes: g.data_offset,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BudgetConfig {
    pub vram_total: u64,
    pub ram_total: u64,
    /// RAM never touched by llamd.
    pub os_reserve: u64,
    /// VRAM headroom for the driver and fragmentation.
    pub driver_reserve: u64,
    /// Context length the KV cache is sized for.
    pub ctx: usize,
    /// Independent sequences (KV slices).
    pub n_seq: usize,
    /// Tokens computed together (prefill tile / batch width).
    pub max_batch: usize,
    /// Bytes per KV element; 4 for the f32 cache `VulkanModel` uses today.
    pub kv_elem_bytes: usize,
    /// VRAM upload staging.
    pub vram_staging: u64,
    /// RAM aligned read-buffer pool. Deliberately NOT a multiple of the
    /// expert size: requests are chunked, so 64 in flight need 64 chunks.
    pub ram_staging: u64,
    /// Override the activation estimate.
    pub activations: Option<u64>,
}

impl BudgetConfig {
    /// The spec's target configuration: 32 GB card, 16 GB RAM.
    pub fn target_32g_16g() -> Self {
        BudgetConfig {
            vram_total: 32 << 30,
            ram_total: 16 << 30,
            os_reserve: 3 << 30,
            driver_reserve: 1 << 30,
            ctx: 4096,
            n_seq: 1,
            max_batch: 32,
            kv_elem_bytes: 4,
            vram_staging: 256 << 20,
            ram_staging: 256 << 20,
            activations: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// `(category, bytes)` per tier, in `Category::ALL` order, zeros omitted.
    pub vram: Vec<(Category, u64)>,
    pub ram: Vec<(Category, u64)>,
    pub vram_total: u64,
    pub ram_total: u64,
    pub slot_bytes: u64,
    /// Experts that can be resident on the card at once.
    pub vram_slots: usize,
    /// Experts the RAM cache can hold.
    pub ram_slots: usize,
    /// Every expert fits in the VRAM slots: streaming is unnecessary.
    pub fits_resident: bool,
    pub facts: ModelFacts,
}

impl MemoryPlan {
    pub fn new(cfg: &BudgetConfig, f: &ModelFacts) -> Result<Self> {
        if f.n_expert_used == 0 || f.max_expert_bytes == 0 {
            return Err(Error::unsupported(
                "memory plan",
                "not a mixture-of-experts model",
            ));
        }
        let kv = (2 * f.n_layer * cfg.ctx * f.kv_dim * cfg.kv_elem_bytes * cfg.n_seq) as u64;
        let act = cfg
            .activations
            .unwrap_or_else(|| activation_estimate(cfg, f));

        let vram_fixed = [
            (Category::SharedWeights, f.shared_bytes),
            (Category::KvCache, kv),
            (Category::Activations, act),
            (Category::Staging, cfg.vram_staging),
            (Category::DriverReserve, cfg.driver_reserve),
        ];
        let vram_used: u64 = vram_fixed.iter().map(|(_, b)| b).sum();
        if vram_used > cfg.vram_total {
            return Err(Error::other(format!(
                "memory plan: fixed VRAM costs {} exceed the {} budget before a single expert:\n{}",
                fmt_bytes(vram_used),
                fmt_bytes(cfg.vram_total),
                breakdown(&vram_fixed)
            )));
        }
        let slot = f.max_expert_bytes;
        let vram_slots = ((cfg.vram_total - vram_used) / slot) as usize;
        if vram_slots < f.n_expert_used {
            return Err(Error::other(format!(
                "memory plan: {} VRAM left for experts holds {vram_slots} of {} MiB each; \
                 one layer needs its top-{} resident at once:\n{}",
                fmt_bytes(cfg.vram_total - vram_used),
                slot >> 20,
                f.n_expert_used,
                breakdown(&vram_fixed)
            )));
        }

        // Metadata: the parsed header plus the index and per-expert stats —
        // a few hundred bytes per expert, rounded up generously.
        let n_experts = (f.n_layer * f.n_expert) as u64;
        let meta = f.header_bytes * 2 + n_experts * 256;
        let ram_fixed = [
            (Category::OsReserve, cfg.os_reserve),
            (Category::Staging, cfg.ram_staging),
            (Category::Metadata, meta),
        ];
        let ram_used: u64 = ram_fixed.iter().map(|(_, b)| b).sum();
        if ram_used > cfg.ram_total {
            return Err(Error::other(format!(
                "memory plan: fixed RAM costs {} exceed the {} budget:\n{}",
                fmt_bytes(ram_used),
                fmt_bytes(cfg.ram_total),
                breakdown(&ram_fixed)
            )));
        }
        let ram_slots = ((cfg.ram_total - ram_used) / slot) as usize;
        let fits_resident = vram_slots as u64 >= n_experts;

        // Experts get what they can USE, not more than exists.
        let vram_slots = vram_slots.min(n_experts as usize);
        let ram_slots = ram_slots.min(n_experts as usize);

        let mut vram: Vec<(Category, u64)> = vram_fixed.to_vec();
        vram.push((Category::ExpertSlots, vram_slots as u64 * slot));
        let mut ram: Vec<(Category, u64)> = ram_fixed.to_vec();
        ram.push((Category::ExpertCache, ram_slots as u64 * slot));
        vram.retain(|(_, b)| *b > 0);
        ram.retain(|(_, b)| *b > 0);

        Ok(MemoryPlan {
            vram,
            ram,
            vram_total: cfg.vram_total,
            ram_total: cfg.ram_total,
            slot_bytes: slot,
            vram_slots,
            ram_slots,
            fits_resident,
            facts: f.clone(),
        })
    }

    pub fn get(&self, tier: Tier, c: Category) -> u64 {
        let v = match tier {
            Tier::Vram => &self.vram,
            Tier::Ram => &self.ram,
        };
        v.iter()
            .find(|(k, _)| *k == c)
            .map(|(_, b)| *b)
            .unwrap_or(0)
    }

    pub fn used(&self, tier: Tier) -> u64 {
        match tier {
            Tier::Vram => self.vram.iter().map(|(_, b)| b).sum(),
            Tier::Ram => self.ram.iter().map(|(_, b)| b).sum(),
        }
    }

    /// Fraction of all experts that can be resident in VRAM, and in VRAM+RAM.
    pub fn coverage(&self) -> (f64, f64) {
        let n = (self.facts.n_layer * self.facts.n_expert) as f64;
        (
            self.vram_slots as f64 / n,
            (self.vram_slots + self.ram_slots) as f64 / n,
        )
    }

    /// Worst case per token: every routed expert misses both caches.
    pub fn worst_case_read_per_token(&self) -> u64 {
        (self.facts.n_layer * self.facts.n_expert_used) as u64 * self.slot_bytes
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        for (tier, name, total) in [
            (Tier::Vram, "VRAM", self.vram_total),
            (Tier::Ram, "RAM", self.ram_total),
        ] {
            s += &format!("  {name} budget {}\n", fmt_bytes(total));
            let v = if tier == Tier::Vram {
                &self.vram
            } else {
                &self.ram
            };
            for (c, b) in v {
                s += &format!("    {:<16} {:>12}\n", format!("{c:?}"), fmt_bytes(*b));
            }
            s += &format!(
                "    {:<16} {:>12}\n",
                "(unassigned)",
                fmt_bytes(total - self.used(tier))
            );
        }
        let (cv, cr) = self.coverage();
        s += &format!(
            "  expert slot {} | VRAM slots {} | RAM cache slots {} | of {} experts\n",
            fmt_bytes(self.slot_bytes),
            self.vram_slots,
            self.ram_slots,
            self.facts.n_layer * self.facts.n_expert
        );
        s += &format!(
            "  coverage: {:.1}% in VRAM, {:.1}% in VRAM+RAM; worst case {} read per token\n",
            cv * 100.0,
            cr * 100.0,
            fmt_bytes(self.worst_case_read_per_token())
        );
        if self.fits_resident {
            s += "  every expert fits in VRAM: streaming is not needed for this budget\n";
        }
        s
    }
}

/// Scratch the forward pass allocates, by the same formulas `VulkanModel`
/// uses (f32 everywhere, sized for `max_batch` tokens).
fn activation_estimate(cfg: &BudgetConfig, f: &ModelFacts) -> u64 {
    let b = cfg.max_batch as u64;
    let per_token = 6 * f.n_embd // x, xb, xb2, attn, residual copies
        + f.q_dim
        + 2 * f.kv_dim
        + 2 * f.n_expert // router logits + probs
        + 2 * f.n_expert_used // ids + weights
        + 3 * f.n_expert_used * f.n_ff_expert // gate, up, act per routed expert
        + f.n_expert_used * f.n_embd // down outputs before combine
        + f.n_vocab; // logits
    let scores = (f.n_head * cfg.ctx) as u64; // attention scores per token
    (per_token as u64 + scores) * 4 * b
}

fn breakdown(items: &[(Category, u64)]) -> String {
    items
        .iter()
        .map(|(c, b)| format!("    {:<16} {}", format!("{c:?}"), fmt_bytes(*b)))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shapes of Qwen3-Coder-30B-A3B Q6_K (the model llamd validates on).
    fn qwen30b() -> ModelFacts {
        let expert = 3 * (2048 * 768 / 256 * 210) as u64; // gate+up+down, Q6_K
        ModelFacts {
            n_layer: 48,
            n_expert: 128,
            n_expert_used: 8,
            n_embd: 2048,
            n_head: 32,
            n_vocab: 151936,
            kv_dim: 512,
            q_dim: 4096,
            n_ff_expert: 768,
            shared_bytes: 1_400 << 20,
            max_expert_bytes: expert,
            expert_bytes_total: expert * 48 * 128,
            header_bytes: 8 << 20,
        }
    }

    #[test]
    fn the_validation_model_fits_resident_on_the_r9700() {
        let mut c = BudgetConfig::target_32g_16g();
        c.ram_total = 91 << 30;
        let p = MemoryPlan::new(&c, &qwen30b()).unwrap();
        assert!(p.fits_resident, "{}", p.report());
    }

    #[test]
    fn a_small_vram_budget_forces_streaming_and_keeps_the_floor() {
        let mut c = BudgetConfig::target_32g_16g();
        c.vram_total = 4 << 30;
        let p = MemoryPlan::new(&c, &qwen30b()).unwrap();
        assert!(!p.fits_resident);
        assert!(p.vram_slots >= 8);
        assert!(p.used(Tier::Vram) <= c.vram_total);
        assert!(p.used(Tier::Ram) <= c.ram_total);
    }

    #[test]
    fn nothing_is_ever_planned_past_a_budget() {
        for vram in [3u64, 4, 8, 16, 32] {
            for ram in [4u64, 8, 16, 64] {
                let mut c = BudgetConfig::target_32g_16g();
                c.vram_total = vram << 30;
                c.ram_total = ram << 30;
                if let Ok(p) = MemoryPlan::new(&c, &qwen30b()) {
                    assert!(p.used(Tier::Vram) <= c.vram_total, "{vram}G/{ram}G");
                    assert!(p.used(Tier::Ram) <= c.ram_total, "{vram}G/{ram}G");
                }
            }
        }
    }

    #[test]
    fn fixed_costs_that_do_not_fit_are_refused_with_a_breakdown() {
        let mut c = BudgetConfig::target_32g_16g();
        c.vram_total = 1 << 30;
        let e = MemoryPlan::new(&c, &qwen30b()).unwrap_err().to_string();
        assert!(e.contains("SharedWeights") && e.contains("KvCache"), "{e}");
    }

    #[test]
    fn fewer_slots_than_top_k_are_refused() {
        let f = qwen30b();
        let mut c = BudgetConfig::target_32g_16g();
        c.activations = Some(0);
        c.vram_staging = 0;
        c.driver_reserve = 0;
        c.ctx = 1;
        let fixed = f.shared_bytes + (2 * f.n_layer * f.kv_dim * 4) as u64;
        c.vram_total = fixed + 5 * f.max_expert_bytes; // room for 5 of 8
        let e = MemoryPlan::new(&c, &f).unwrap_err().to_string();
        assert!(e.contains("top-8"), "{e}");
        c.vram_total = fixed + 8 * f.max_expert_bytes;
        assert_eq!(MemoryPlan::new(&c, &f).unwrap().vram_slots, 8);
    }

    #[test]
    fn kv_cache_is_exact_for_the_f32_layout() {
        let f = qwen30b();
        let c = BudgetConfig::target_32g_16g();
        let p = MemoryPlan::new(&c, &f).unwrap();
        // 2 (K,V) x 48 layers x 4096 positions x 512 x 4 bytes = 768 MiB.
        assert_eq!(p.get(Tier::Vram, Category::KvCache), 768 << 20);
    }
}
