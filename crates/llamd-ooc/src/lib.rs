//! Out-of-core mixture-of-experts: running a model larger than RAM and VRAM.
//!
//! THE PROBLEM. A ~508 GB MoE model on a 32 GB card and 16 GB of RAM. Every
//! token needs only `top_k` experts per layer, so the model never has to be
//! resident — but every byte that IS resident must be accounted for, because
//! the failure mode of "assume it fits" is not an error message: it is the OOM
//! killer, a driver reset, or a machine swapping itself to a standstill.
//!
//! WHAT LIVES HERE.
//!
//! * [`size`] — byte counts as people write them (`32G`, `512MiB`).
//! * [`budget`] — the memory PLAN: from the model's own index and a config,
//!   how much VRAM and RAM each category gets, and how many expert slots fit.
//!   Refuses, with a breakdown, when the fixed costs alone do not fit.
//! * [`ledger`] — allocation-time ENFORCEMENT of that plan. Memory is
//!   reserved against a category's cap before it is allocated; a reservation
//!   that would exceed the cap fails then, not after the allocation succeeded.
//!
//! * [`io`] — the expert LOADER: bounded, prioritised, chunked reads with
//!   O_DIRECT and a buffered fallback, staging reserved against the ledger.
//!
//! The expert cache and scheduler integration build on these.
//!
//! No unsafe and no new dependencies: this crate decides and accounts; it
//! does not touch device memory itself.

#![forbid(unsafe_code)]

pub mod budget;
pub mod io;
pub mod ledger;
pub mod size;

pub use budget::{BudgetConfig, Category, MemoryPlan, ModelFacts, Tier};
pub use io::{Loader, LoaderConfig, LoaderStats, Priority, ReadMode, Ticket};
pub use ledger::{Ledger, Reservation};
pub use size::{fmt_bytes, parse_bytes};
