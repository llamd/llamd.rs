//! Allocation-time enforcement of a [`MemoryPlan`].
//!
//! RESERVE, THEN ALLOCATE. Every buffer that belongs to a budgeted category is
//! preceded by a [`Reservation`] for its size. The reservation is checked
//! against the category's cap under a lock, so two threads cannot both pass a
//! check that only one of them fits. If it fails, nothing was allocated; if it
//! succeeds, the bytes are accounted until the `Reservation` is dropped.
//!
//! WHY A LEDGER AND NOT JUST THE ALLOCATOR. Device allocators fail late and
//! differently per driver (an OOM, a hang, a silent spill to host memory on
//! some stacks). Page cache and RSS are not allocator-visible at all. A ledger
//! fails early, the same way everywhere, with the category and the numbers in
//! the message — and its peak is a measurement the benchmarks can report.

use llamd_core::{Error, Result};
use std::sync::{Arc, Mutex};

use crate::budget::{Category, MemoryPlan, Tier};
use crate::size::fmt_bytes;

const N: usize = Category::ALL.len();

#[derive(Debug, Default, Clone, Copy)]
struct Counter {
    cap: u64,
    used: u64,
    peak: u64,
    refused: u64,
}

/// Per-category caps and usage for ONE tier.
#[derive(Debug)]
pub struct Ledger {
    tier: Tier,
    c: Mutex<[Counter; N]>,
}

/// Bytes held against a category; released on drop.
#[derive(Debug)]
pub struct Reservation {
    ledger: Arc<Ledger>,
    cat: Category,
    bytes: u64,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn category(&self) -> Category {
        self.cat
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut c = self.ledger.c.lock().unwrap_or_else(|p| p.into_inner());
        let k = &mut c[self.cat.index()];
        k.used = k.used.saturating_sub(self.bytes);
    }
}

/// A snapshot of one category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    pub cap: u64,
    pub used: u64,
    pub peak: u64,
    /// Reservations refused because they would have exceeded `cap`.
    pub refused: u64,
}

impl Ledger {
    /// Caps from a plan, for one tier. Categories the plan gives nothing get a
    /// cap of zero — reserving against them is an error, which is the point.
    pub fn from_plan(plan: &MemoryPlan, tier: Tier) -> Arc<Self> {
        let mut c = [Counter::default(); N];
        for cat in Category::ALL {
            c[cat.index()].cap = plan.get(tier, cat);
        }
        Arc::new(Ledger {
            tier,
            c: Mutex::new(c),
        })
    }

    /// Explicit caps — for tests and tools.
    pub fn with_caps(tier: Tier, caps: &[(Category, u64)]) -> Arc<Self> {
        let mut c = [Counter::default(); N];
        for (cat, b) in caps {
            c[cat.index()].cap = *b;
        }
        Arc::new(Ledger {
            tier,
            c: Mutex::new(c),
        })
    }

    /// Reserve `bytes` against `cat`, or fail without reserving anything.
    pub fn reserve(self: &Arc<Self>, cat: Category, bytes: u64) -> Result<Reservation> {
        let mut c = self.c.lock().unwrap_or_else(|p| p.into_inner());
        let k = &mut c[cat.index()];
        let after = k.used.checked_add(bytes).filter(|&a| a <= k.cap);
        match after {
            Some(a) => {
                k.used = a;
                k.peak = k.peak.max(a);
                Ok(Reservation {
                    ledger: Arc::clone(self),
                    cat,
                    bytes,
                })
            }
            None => {
                k.refused += 1;
                Err(Error::other(format!(
                    "{:?} budget: {cat:?} would reach {} of its {} cap ({} requested, {} in use)",
                    self.tier,
                    fmt_bytes(k.used.saturating_add(bytes)),
                    fmt_bytes(k.cap),
                    fmt_bytes(bytes),
                    fmt_bytes(k.used)
                )))
            }
        }
    }

    pub fn usage(&self, cat: Category) -> Usage {
        let c = self.c.lock().unwrap_or_else(|p| p.into_inner());
        let k = c[cat.index()];
        Usage {
            cap: k.cap,
            used: k.used,
            peak: k.peak,
            refused: k.refused,
        }
    }

    /// Total in use across categories.
    pub fn used(&self) -> u64 {
        let c = self.c.lock().unwrap_or_else(|p| p.into_inner());
        c.iter().map(|k| k.used).sum()
    }

    /// Sum of per-category peaks: an UPPER bound on the joint peak (the
    /// categories need not have peaked at the same moment).
    pub fn peak_upper_bound(&self) -> u64 {
        let c = self.c.lock().unwrap_or_else(|p| p.into_inner());
        c.iter().map(|k| k.peak).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reservation_past_the_cap_fails_and_reserves_nothing() {
        let l = Ledger::with_caps(Tier::Ram, &[(Category::ExpertCache, 100)]);
        let a = l.reserve(Category::ExpertCache, 60).unwrap();
        let e = l
            .reserve(Category::ExpertCache, 50)
            .unwrap_err()
            .to_string();
        assert!(e.contains("ExpertCache"), "{e}");
        let u = l.usage(Category::ExpertCache);
        assert_eq!((u.used, u.peak, u.refused), (60, 60, 1));
        drop(a);
        assert_eq!(l.usage(Category::ExpertCache).used, 0);
        // Freed capacity is reusable.
        let _b = l.reserve(Category::ExpertCache, 100).unwrap();
    }

    #[test]
    fn categories_do_not_borrow_from_each_other() {
        let l = Ledger::with_caps(
            Tier::Vram,
            &[(Category::ExpertSlots, 10), (Category::Staging, 1000)],
        );
        assert!(l.reserve(Category::ExpertSlots, 11).is_err());
        assert!(
            l.reserve(Category::KvCache, 1).is_err(),
            "uncapped means zero"
        );
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_cap() {
        let l = Ledger::with_caps(Tier::Ram, &[(Category::Staging, 1000)]);
        let held = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| {
                    for _ in 0..200 {
                        if let Ok(r) = l.reserve(Category::Staging, 7) {
                            held.lock().unwrap().push(r);
                        }
                        let mut h = held.lock().unwrap();
                        if h.len() > 100 {
                            h.drain(..50);
                        }
                    }
                });
            }
        });
        let u = l.usage(Category::Staging);
        assert!(u.peak <= 1000, "peak {} past cap", u.peak);
        drop(held);
        assert_eq!(l.usage(Category::Staging).used, 0);
    }

    #[test]
    fn overflowing_requests_are_refused_not_wrapped() {
        let l = Ledger::with_caps(Tier::Ram, &[(Category::Staging, u64::MAX)]);
        let _a = l.reserve(Category::Staging, u64::MAX - 1).unwrap();
        assert!(l.reserve(Category::Staging, 10).is_err());
    }
}
