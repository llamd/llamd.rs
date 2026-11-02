//! The expert cache: which experts live in VRAM, which in RAM, and which go.
//!
//! TWO TIERS, ONE POLICY. VRAM holds a fixed number of expert SLOTS (the
//! plan's `vram_slots`); this module decides which expert occupies which slot
//! and the device code owns the memory. RAM holds expert BYTES, each entry
//! reserved against the ledger's `ExpertCache` category. Both tiers evict by
//! the same score:
//!
//! ```text
//!   score = frequency * 2^(-(now - last_use) / half_life)
//! ```
//!
//! — LFU with exponential aging. Pure LFU keeps an expert that was hot an
//! hour ago forever; pure LRU throws out an expert used every few tokens the
//! moment a burst of one-offs arrives. Aging makes frequency decay, so
//! "frequent AND recent" wins, and `half_life` (in routing events) says how
//! fast the past is forgotten.
//!
//! PINNED MEANS UNTOUCHABLE. A VRAM slot referenced by recorded-but-unfinished
//! GPU work is pinned; eviction skips it, and if every slot is pinned the
//! request fails instead of overwriting weights a kernel is about to read.
//! RAM entries are handed out as `Arc`s, so an entry evicted while an upload
//! still reads it stays alive until that reader drops it.
//!
//! THE PREDICTOR is deliberately simple and measures itself: for the next
//! layers it proposes the experts with the highest aged frequency at that
//! layer, plus those the PREVIOUS token chose there (token-to-token reuse is
//! the strongest signal MoE routing usually has). Every prediction is scored
//! against the router's actual choice, so its precision is a number in the
//! stats, not a hope.

use llamd_core::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

use crate::budget::Category;
use crate::ledger::{Ledger, Reservation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertKey {
    pub fn new(layer: usize, expert: usize) -> Self {
        ExpertKey {
            layer: layer as u16,
            expert: expert as u16,
        }
    }
}

/// Where an expert is right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Vram(usize),
    Ram,
    Disk,
}

#[derive(Debug, Clone, Copy, Default)]
struct Heat {
    freq: f64,
    last: u64,
}

impl Heat {
    fn score(&self, now: u64, half_life: f64) -> f64 {
        self.freq * (-((now - self.last) as f64) / half_life).exp2()
    }
    fn touch(&mut self, now: u64, half_life: f64) {
        self.freq = self.score(now, half_life) + 1.0;
        self.last = now;
    }
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub misses: u64,
    pub vram_evictions: u64,
    pub ram_evictions: u64,
    /// Slot requests refused because every slot was pinned.
    pub pinned_refusals: u64,
    pub predicted: u64,
    pub predicted_correct: u64,
    pub routed: u64,
}

impl CacheStats {
    pub fn vram_hit_rate(&self) -> f64 {
        ratio(self.vram_hits, self.lookups())
    }
    pub fn any_hit_rate(&self) -> f64 {
        ratio(self.vram_hits + self.ram_hits, self.lookups())
    }
    /// Of the experts predicted, the fraction the router then chose.
    pub fn prediction_precision(&self) -> f64 {
        ratio(self.predicted_correct, self.predicted)
    }
    /// Of the experts routed, the fraction that had been predicted.
    pub fn prediction_recall(&self) -> f64 {
        ratio(self.predicted_correct, self.routed)
    }
    pub fn lookups(&self) -> u64 {
        self.vram_hits + self.ram_hits + self.misses
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

struct Slot {
    key: Option<ExpertKey>,
    pins: u32,
}

struct RamEntry {
    bytes: Arc<Vec<u8>>,
    _res: Reservation,
}

pub struct ExpertCache {
    half_life: f64,
    now: u64,
    heat: HashMap<ExpertKey, Heat>,
    slots: Vec<Slot>,
    in_vram: HashMap<ExpertKey, usize>,
    ram: HashMap<ExpertKey, RamEntry>,
    ram_slots: usize,
    ledger: Arc<Ledger>,
    /// Previous token's choice per layer.
    last_routing: HashMap<u16, Vec<u16>>,
    /// Outstanding predictions per layer, scored when that layer routes.
    pending: HashMap<u16, Vec<u16>>,
    stats: CacheStats,
}

impl ExpertCache {
    /// `vram_slots` and `ram_slots` from the plan; RAM bytes are reserved
    /// against `ledger`'s `ExpertCache` category per entry.
    pub fn new(vram_slots: usize, ram_slots: usize, half_life: f64, ledger: Arc<Ledger>) -> Self {
        ExpertCache {
            half_life: half_life.max(1.0),
            now: 0,
            heat: HashMap::new(),
            slots: (0..vram_slots)
                .map(|_| Slot { key: None, pins: 0 })
                .collect(),
            in_vram: HashMap::new(),
            ram: HashMap::new(),
            ram_slots,
            ledger,
            last_routing: HashMap::new(),
            pending: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    pub fn residency(&self, k: ExpertKey) -> Residency {
        if let Some(&s) = self.in_vram.get(&k) {
            Residency::Vram(s)
        } else if self.ram.contains_key(&k) {
            Residency::Ram
        } else {
            Residency::Disk
        }
    }

    /// Record one router decision for `layer`: updates heat, scores any
    /// prediction for this layer, counts hits by tier, and returns where each
    /// chosen expert is.
    pub fn route(&mut self, layer: usize, experts: &[u32]) -> Vec<Residency> {
        let l = layer as u16;
        self.now += 1;
        let (now, hl) = (self.now, self.half_life);
        if let Some(pred) = self.pending.remove(&l) {
            let correct = experts
                .iter()
                .filter(|e| pred.contains(&(**e as u16)))
                .count();
            self.stats.predicted += pred.len() as u64;
            self.stats.predicted_correct += correct as u64;
        }
        self.stats.routed += experts.len() as u64;
        let mut out = Vec::with_capacity(experts.len());
        for &e in experts {
            let k = ExpertKey::new(layer, e as usize);
            self.heat.entry(k).or_default().touch(now, hl);
            let r = self.residency(k);
            match r {
                Residency::Vram(_) => self.stats.vram_hits += 1,
                Residency::Ram => self.stats.ram_hits += 1,
                Residency::Disk => self.stats.misses += 1,
            }
            out.push(r);
        }
        self.last_routing
            .insert(l, experts.iter().map(|&e| e as u16).collect());
        out
    }

    /// Give `k` a VRAM slot and pin it. Returns the slot and whether its
    /// weights must be uploaded (false when already resident).
    ///
    /// Evicts the coldest UNPINNED occupant when full; fails when every slot
    /// is pinned rather than overwrite weights in use.
    pub fn acquire_vram(&mut self, k: ExpertKey) -> Result<(usize, bool)> {
        if let Some(&s) = self.in_vram.get(&k) {
            self.slots[s].pins += 1;
            return Ok((s, false));
        }
        let s = match self.slots.iter().position(|s| s.key.is_none()) {
            Some(free) => free,
            None => {
                let (now, hl) = (self.now, self.half_life);
                let victim = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.pins == 0)
                    .min_by(|(_, a), (_, b)| {
                        let sa = self
                            .heat
                            .get(&a.key.unwrap())
                            .map_or(0.0, |h| h.score(now, hl));
                        let sb = self
                            .heat
                            .get(&b.key.unwrap())
                            .map_or(0.0, |h| h.score(now, hl));
                        sa.total_cmp(&sb)
                    })
                    .map(|(i, _)| i);
                match victim {
                    Some(v) => {
                        let old = self.slots[v].key.take().expect("occupied");
                        self.in_vram.remove(&old);
                        self.stats.vram_evictions += 1;
                        v
                    }
                    None => {
                        self.stats.pinned_refusals += 1;
                        return Err(Error::other(format!(
                            "all {} VRAM expert slots are pinned by in-flight work",
                            self.slots.len()
                        )));
                    }
                }
            }
        };
        self.slots[s] = Slot {
            key: Some(k),
            pins: 1,
        };
        self.in_vram.insert(k, s);
        Ok((s, true))
    }

    /// Release one pin taken by `acquire_vram` (call when the GPU work that
    /// reads the slot has completed).
    pub fn unpin(&mut self, k: ExpertKey) -> Result<()> {
        let s = *self
            .in_vram
            .get(&k)
            .ok_or_else(|| Error::other(format!("unpin of {k:?}, which is not in VRAM")))?;
        let slot = &mut self.slots[s];
        if slot.pins == 0 {
            return Err(Error::other(format!("unpin of {k:?}, which is not pinned")));
        }
        slot.pins -= 1;
        Ok(())
    }

    pub fn pins(&self, k: ExpertKey) -> u32 {
        self.in_vram.get(&k).map_or(0, |&s| self.slots[s].pins)
    }

    /// The bytes of `k` if the RAM tier holds them.
    pub fn ram_get(&self, k: ExpertKey) -> Option<Arc<Vec<u8>>> {
        self.ram.get(&k).map(|e| Arc::clone(&e.bytes))
    }

    /// Keep `bytes` for `k` in the RAM tier, evicting colder entries as
    /// needed. Returns false (and keeps nothing) when `k` is colder than
    /// everything it would displace — caching it would only churn.
    pub fn ram_insert(&mut self, k: ExpertKey, bytes: Vec<u8>) -> Result<bool> {
        if self.ram_slots == 0 {
            return Ok(false);
        }
        if self.ram.contains_key(&k) {
            return Ok(true);
        }
        let (now, hl) = (self.now, self.half_life);
        let score = |c: &Self, key: &ExpertKey| c.heat.get(key).map_or(0.0, |h| h.score(now, hl));
        let mine = score(self, &k);
        while self.ram.len() >= self.ram_slots {
            let coldest = self
                .ram
                .keys()
                .copied()
                .min_by(|a, b| score(self, a).total_cmp(&score(self, b)))
                .expect("nonempty");
            if score(self, &coldest) > mine {
                return Ok(false);
            }
            self.ram.remove(&coldest);
            self.stats.ram_evictions += 1;
        }
        let res = self
            .ledger
            .reserve(Category::ExpertCache, bytes.len() as u64)?;
        self.ram.insert(
            k,
            RamEntry {
                bytes: Arc::new(bytes),
                _res: res,
            },
        );
        Ok(true)
    }

    /// Predict up to `n` experts for `layer` and remember the prediction so
    /// the next `route` for that layer scores it.
    pub fn predict(&mut self, layer: usize, n: usize) -> Vec<u16> {
        let l = layer as u16;
        let (now, hl) = (self.now, self.half_life);
        let mut pick: Vec<u16> = self.last_routing.get(&l).cloned().unwrap_or_default();
        pick.truncate(n);
        if pick.len() < n {
            let mut hot: Vec<(f64, u16)> = self
                .heat
                .iter()
                .filter(|(k, _)| k.layer == l && !pick.contains(&k.expert))
                .map(|(k, h)| (h.score(now, hl), k.expert))
                .collect();
            hot.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            pick.extend(hot.into_iter().take(n - pick.len()).map(|(_, e)| e));
        }
        self.pending.insert(l, pick.clone());
        pick
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    pub fn vram_occupancy(&self) -> (usize, usize) {
        (self.in_vram.len(), self.slots.len())
    }

    pub fn ram_occupancy(&self) -> (usize, usize) {
        (self.ram.len(), self.ram_slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Tier;

    fn cache(vram: usize, ram: usize) -> ExpertCache {
        let l = Ledger::with_caps(Tier::Ram, &[(Category::ExpertCache, 1 << 30)]);
        ExpertCache::new(vram, ram, 16.0, l)
    }
    fn k(l: usize, e: usize) -> ExpertKey {
        ExpertKey::new(l, e)
    }

    #[test]
    fn a_pinned_slot_is_never_evicted() {
        let mut c = cache(2, 0);
        c.route(0, &[1, 2]);
        c.acquire_vram(k(0, 1)).unwrap(); // pinned
        c.acquire_vram(k(0, 2)).unwrap(); // pinned
        // Full and all pinned: refuse rather than overwrite live weights.
        assert!(c.acquire_vram(k(0, 3)).is_err());
        assert_eq!(c.stats().pinned_refusals, 1);
        c.unpin(k(0, 2)).unwrap();
        let (slot, upload) = c.acquire_vram(k(0, 3)).unwrap();
        assert!(upload);
        assert_eq!(c.residency(k(0, 1)), Residency::Vram(c.in_vram[&k(0, 1)]));
        assert_eq!(
            c.residency(k(0, 2)),
            Residency::Disk,
            "the unpinned one went"
        );
        assert_eq!(c.residency(k(0, 3)), Residency::Vram(slot));
    }

    #[test]
    fn already_resident_experts_need_no_upload_and_stack_pins() {
        let mut c = cache(2, 0);
        let (s1, up1) = c.acquire_vram(k(1, 7)).unwrap();
        let (s2, up2) = c.acquire_vram(k(1, 7)).unwrap();
        assert_eq!((s1, up1, up2), (s2, true, false));
        assert_eq!(c.pins(k(1, 7)), 2);
        c.unpin(k(1, 7)).unwrap();
        c.unpin(k(1, 7)).unwrap();
        assert!(
            c.unpin(k(1, 7)).is_err(),
            "unbalanced unpin is a bug, not a no-op"
        );
    }

    #[test]
    fn frequent_and_recent_beats_a_burst_of_one_offs() {
        let mut c = cache(3, 0);
        for _ in 0..20 {
            c.route(0, &[5]); // expert 5 is hot
        }
        c.acquire_vram(k(0, 5)).unwrap();
        c.unpin(k(0, 5)).unwrap();
        // A stream of one-off experts must cycle through the other slots,
        // not evict the hot one.
        for e in 100..120 {
            c.route(0, &[e]);
            c.acquire_vram(k(0, e as usize)).unwrap();
            c.unpin(k(0, e as usize)).unwrap();
        }
        assert!(matches!(c.residency(k(0, 5)), Residency::Vram(_)));
    }

    #[test]
    fn aging_lets_a_stale_hot_expert_go() {
        let mut c = cache(1, 0);
        for _ in 0..10 {
            c.route(0, &[5]);
        }
        c.acquire_vram(k(0, 5)).unwrap();
        c.unpin(k(0, 5)).unwrap();
        // Many routing events later (half-life 16), expert 9 used a few times
        // recently outscores expert 5's old popularity.
        for _ in 0..200 {
            c.route(1, &[0]);
        }
        for _ in 0..3 {
            c.route(0, &[9]);
        }
        c.acquire_vram(k(0, 9)).unwrap();
        assert_eq!(c.residency(k(0, 5)), Residency::Disk);
    }

    #[test]
    fn the_ram_tier_respects_its_slots_and_its_ledger() {
        let l = Ledger::with_caps(Tier::Ram, &[(Category::ExpertCache, 250)]);
        let mut c = ExpertCache::new(0, 10, 16.0, Arc::clone(&l));
        for e in 0..3 {
            c.route(0, &[e]);
            assert!(c.ram_insert(k(0, e as usize), vec![0u8; 100]).is_ok() || e == 2);
        }
        // Ten slots but only 250 bytes: the third 100-byte entry is refused
        // by the ledger, not silently admitted.
        assert!(l.usage(Category::ExpertCache).used <= 250);
        assert_eq!(c.ram_occupancy().0, 2);
    }

    #[test]
    fn an_evicted_ram_entry_stays_alive_for_a_reader_still_holding_it() {
        let mut c = cache(0, 1);
        c.route(0, &[1]);
        c.ram_insert(k(0, 1), vec![7u8; 8]).unwrap();
        let held = c.ram_get(k(0, 1)).unwrap();
        for _ in 0..5 {
            c.route(0, &[2]);
        }
        assert!(c.ram_insert(k(0, 2), vec![9u8; 8]).unwrap());
        assert!(c.ram_get(k(0, 1)).is_none(), "evicted from the cache");
        assert_eq!(held[0], 7, "but the reader's Arc keeps the bytes valid");
    }

    #[test]
    fn a_cold_newcomer_does_not_churn_a_hot_ram_tier() {
        let mut c = cache(0, 1);
        for _ in 0..10 {
            c.route(0, &[1]);
        }
        c.ram_insert(k(0, 1), vec![0; 8]).unwrap();
        c.route(0, &[2]);
        assert!(!c.ram_insert(k(0, 2), vec![0; 8]).unwrap());
        assert!(c.ram_get(k(0, 1)).is_some());
    }

    #[test]
    fn the_predictor_scores_itself() {
        let mut c = cache(0, 0);
        // Perfectly repetitive routing: prediction from the previous token
        // must be exact after the first token.
        for _ in 0..10 {
            for l in 0..4 {
                c.predict(l, 2);
                c.route(l, &[3, 4]);
            }
        }
        let s = c.stats();
        assert!(
            s.prediction_precision() > 0.85,
            "{}",
            s.prediction_precision()
        );
        assert!(s.prediction_recall() > 0.85);
    }

    #[test]
    fn hit_rates_count_by_tier() {
        let mut c = cache(1, 1);
        c.acquire_vram(k(0, 1)).unwrap();
        c.route(0, &[2]);
        c.ram_insert(k(0, 2), vec![0; 4]).unwrap();
        let r = c.route(0, &[1, 2, 3]);
        assert!(matches!(r[0], Residency::Vram(_)));
        assert_eq!(r[1], Residency::Ram);
        assert_eq!(r[2], Residency::Disk);
        let s = c.stats();
        assert_eq!((s.vram_hits, s.ram_hits, s.misses), (1, 1, 2));
    }
}
