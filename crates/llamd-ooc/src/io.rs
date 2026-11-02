//! Asynchronous, bounded, prioritised reads of expert extents.
//!
//! THE SHAPE. A request is one extent (an expert's projection, say 16 MB) and
//! a destination buffer. It is split into CHUNKS; each chunk is read by a
//! worker into a pooled, 4 KiB-aligned staging buffer and copied into the
//! destination. So memory in flight is `queue_depth` chunks — never
//! `queue_depth` experts — and it is reserved against the `Staging` budget in
//! the [`Ledger`] when the loader is built, not discovered later.
//!
//! PRIORITY IS STRUCTURAL. Required reads (an expert the router just chose)
//! and speculative reads (a prefetch guess) live in separate queues. Workers
//! always take required work first; speculative chunks in flight are capped
//! below the worker count, so there is always a worker free for a required
//! chunk; and speculative submissions are REFUSED when their queue is full,
//! rather than queued behind or ahead of anything. A wrong guess costs at most
//! the chunks already in flight.
//!
//! O_DIRECT, WITH A FALLBACK. Direct reads bypass the page cache, which is
//! what keeps a 500 GB model from turning 16 GB of RAM into page cache. They
//! need aligned offsets, lengths and buffers; GGUF only aligns tensors to 32
//! bytes, so each chunk is widened to 4 KiB boundaries and the head trimmed.
//! When the filesystem refuses O_DIRECT (EINVAL at open or read), the loader
//! falls back to buffered reads for that file and says so in its stats.
//!
//! THREADS, NOT io_uring — YET. Each worker issues one blocking positioned
//! read, so `queue_depth` workers keep `queue_depth` requests outstanding at
//! the device, which is what the NVMe queue sees either way. No unsafe and no
//! new dependency. An io_uring engine can replace the worker loop behind the
//! same `Loader` API; the measurements below are its baseline.

use llamd_core::{Error, Result};
use llamd_gguf::Extent;
use std::collections::VecDeque;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::budget::Category;
use crate::ledger::{Ledger, Reservation};

/// Direct-I/O alignment. 4096 satisfies every logical block size in use.
pub const IO_ALIGN: u64 = 4096;

#[cfg(target_arch = "x86_64")]
const O_DIRECT: Option<i32> = Some(0o40000);
#[cfg(target_arch = "aarch64")]
const O_DIRECT: Option<i32> = Some(0o200000);
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const O_DIRECT: Option<i32> = None;

const EINVAL: i32 = 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Required,
    Speculative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    Direct,
    Buffered,
}

#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Worker threads = chunk reads outstanding at the device.
    pub queue_depth: usize,
    /// Staging buffer size; a multiple of 4 KiB. Each chunk carries up to
    /// `chunk_bytes - 4096` payload (the slack absorbs head misalignment).
    pub chunk_bytes: usize,
    /// Try O_DIRECT first.
    pub direct: bool,
    /// Speculative chunks allowed in flight at once (< queue_depth).
    pub max_speculative_in_flight: usize,
    /// Speculative chunks allowed to wait; beyond this, submissions are refused.
    pub max_speculative_queued: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            queue_depth: 64,
            chunk_bytes: 4 << 20,
            direct: true,
            max_speculative_in_flight: 16,
            max_speculative_queued: 256,
        }
    }
}

impl LoaderConfig {
    pub fn staging_bytes(&self) -> u64 {
        (self.queue_depth * self.chunk_bytes) as u64
    }
    fn payload(&self) -> u64 {
        self.chunk_bytes as u64 - IO_ALIGN
    }
    fn validate(&self) -> Result<()> {
        let bad = |d: String| Err(Error::malformed("loader config", d));
        if self.queue_depth == 0 {
            return bad("queue_depth must be at least 1".into());
        }
        if self.chunk_bytes < 2 * IO_ALIGN as usize || self.chunk_bytes % IO_ALIGN as usize != 0 {
            return bad(format!(
                "chunk_bytes {} must be a multiple of {IO_ALIGN} and at least {}",
                self.chunk_bytes,
                2 * IO_ALIGN
            ));
        }
        if self.max_speculative_in_flight >= self.queue_depth && self.queue_depth > 1 {
            return bad(format!(
                "max_speculative_in_flight {} must leave a worker for required reads (depth {})",
                self.max_speculative_in_flight, self.queue_depth
            ));
        }
        Ok(())
    }
}

/// A heap buffer whose usable window starts on an `IO_ALIGN` boundary.
/// Over-allocates by one alignment unit instead of using an aligned
/// allocator, which keeps this crate free of unsafe.
struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let raw = vec![0u8; len + IO_ALIGN as usize];
        let addr = raw.as_ptr() as usize;
        let off = (IO_ALIGN as usize - addr % IO_ALIGN as usize) % IO_ALIGN as usize;
        AlignedBuf { raw, off, len }
    }
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

/// One request's shared state.
struct Job {
    priority: Priority,
    dest: Mutex<Vec<u8>>,
    remaining: Mutex<usize>,
    done: Condvar,
    error: Mutex<Option<String>>,
    cancelled: AtomicBool,
    submitted: Instant,
    finished_at: Mutex<Option<Instant>>,
    /// Global completion order — lets tests and stats see who finished first.
    finished_seq: AtomicU64,
}

struct Chunk {
    job: Arc<Job>,
    /// Absolute file offset and length of the payload wanted.
    offset: u64,
    len: u64,
    /// Where in `dest` it goes.
    dest_off: usize,
    queued: Instant,
}

/// The handle a submitter keeps.
pub struct Ticket {
    job: Arc<Job>,
}

impl Ticket {
    /// Block until every chunk is read (or the job failed / was cancelled),
    /// then hand back the destination buffer.
    pub fn wait(self) -> Result<Vec<u8>> {
        let mut r = self.job.remaining.lock().unwrap_or_else(|p| p.into_inner());
        while *r > 0 {
            r = self.job.done.wait(r).unwrap_or_else(|p| p.into_inner());
        }
        drop(r);
        if self.job.cancelled.load(Ordering::Acquire) {
            return Err(Error::other("load cancelled"));
        }
        if let Some(e) = self
            .job
            .error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
        {
            return Err(Error::other(e));
        }
        Ok(std::mem::take(
            &mut *self.job.dest.lock().unwrap_or_else(|p| p.into_inner()),
        ))
    }

    /// [`Ticket::wait`], plus the submit-to-completion latency.
    pub fn wait_timed(self) -> Result<(Vec<u8>, Duration)> {
        let job = Arc::clone(&self.job);
        let data = self.wait()?;
        let end = job
            .finished_at
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .unwrap_or_else(Instant::now);
        Ok((data, end - job.submitted))
    }

    pub fn is_done(&self) -> bool {
        *self.job.remaining.lock().unwrap_or_else(|p| p.into_inner()) == 0
    }

    /// Drop the chunks not yet started. In-flight chunks finish harmlessly.
    pub fn cancel(&self) {
        self.job.cancelled.store(true, Ordering::Release);
    }

    /// Submit-to-finish latency, once done.
    pub fn latency(&self) -> Option<Duration> {
        self.job
            .finished_at
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .map(|t| t - self.job.submitted)
    }

    pub fn finished_seq(&self) -> u64 {
        self.job.finished_seq.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoaderStats {
    pub bytes: u64,
    pub chunks: u64,
    pub direct_chunks: u64,
    pub buffered_chunks: u64,
    pub required_jobs: u64,
    pub speculative_jobs: u64,
    pub speculative_refused: u64,
    pub cancelled_chunks: u64,
    pub errors: u64,
    /// Summed time chunks waited in queue, and summed read time.
    pub queue_wait: Duration,
    pub read_time: Duration,
    /// Wall time from the first chunk start to the last chunk end.
    pub busy_wall: Duration,
}

impl LoaderStats {
    pub fn mib_per_s(&self) -> f64 {
        let s = self.busy_wall.as_secs_f64();
        if s == 0.0 {
            0.0
        } else {
            self.bytes as f64 / (1u64 << 20) as f64 / s
        }
    }
}

struct Queues {
    required: VecDeque<Chunk>,
    speculative: VecDeque<Chunk>,
    spec_in_flight: usize,
    paused: bool,
    shutdown: bool,
}

struct Shared {
    cfg: LoaderConfig,
    file: File,
    /// Buffered fallback handle (also used when direct is off).
    buffered: File,
    mode: Mutex<ReadMode>,
    file_len: u64,
    q: Mutex<Queues>,
    work: Condvar,
    stats: Mutex<LoaderStats>,
    first_start: Mutex<Option<Instant>>,
    seq: AtomicU64,
    pool: Mutex<Vec<AlignedBuf>>,
}

pub struct Loader {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    _staging: Reservation,
    path: PathBuf,
}

impl Loader {
    /// Open `path` for loading. The staging pool is reserved against the
    /// ledger's `Staging` category first; if it does not fit, nothing is
    /// allocated and no thread is started.
    pub fn new(path: impl AsRef<Path>, cfg: LoaderConfig, ledger: &Arc<Ledger>) -> Result<Self> {
        cfg.validate()?;
        let staging = ledger.reserve(Category::Staging, cfg.staging_bytes())?;
        let path = path.as_ref().to_path_buf();
        let buffered = File::open(&path)?;
        let file_len = buffered.metadata()?.len();
        let (file, mode) = match (cfg.direct, O_DIRECT) {
            (true, Some(flag)) => {
                use std::os::unix::fs::OpenOptionsExt;
                match std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(flag)
                    .open(&path)
                {
                    Ok(f) => (f, ReadMode::Direct),
                    Err(_) => (File::open(&path)?, ReadMode::Buffered),
                }
            }
            _ => (File::open(&path)?, ReadMode::Buffered),
        };
        let pool = (0..cfg.queue_depth)
            .map(|_| AlignedBuf::new(cfg.chunk_bytes))
            .collect();
        let shared = Arc::new(Shared {
            cfg: cfg.clone(),
            file,
            buffered,
            mode: Mutex::new(mode),
            file_len,
            q: Mutex::new(Queues {
                required: VecDeque::new(),
                speculative: VecDeque::new(),
                spec_in_flight: 0,
                paused: false,
                shutdown: false,
            }),
            work: Condvar::new(),
            stats: Mutex::new(LoaderStats::default()),
            first_start: Mutex::new(None),
            seq: AtomicU64::new(0),
            pool: Mutex::new(pool),
        });
        let workers = (0..cfg.queue_depth)
            .map(|i| {
                let s = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("llamd-io-{i}"))
                    .spawn(move || worker(s))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Loader {
            shared,
            workers,
            _staging: staging,
            path,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The mode in use now (may have dropped from Direct to Buffered).
    pub fn mode(&self) -> ReadMode {
        *self.shared.mode.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Read `extent` into `dest` (resized to the extent's length).
    ///
    /// Speculative submissions are refused when the speculative queue is
    /// full — a prefetch that cannot start soon is worth nothing.
    pub fn submit(&self, extent: Extent, priority: Priority, mut dest: Vec<u8>) -> Result<Ticket> {
        if extent.end() > self.shared.file_len {
            return Err(Error::malformed(
                "load",
                format!(
                    "extent {}..{} is past the {}-byte file",
                    extent.offset,
                    extent.end(),
                    self.shared.file_len
                ),
            ));
        }
        let len = usize::try_from(extent.len)
            .map_err(|_| Error::malformed("load", "extent does not fit in memory"))?;
        dest.resize(len, 0);
        let payload = self.shared.cfg.payload();
        let n_chunks = extent.len.div_ceil(payload).max(1) as usize;
        let job = Arc::new(Job {
            priority,
            dest: Mutex::new(dest),
            remaining: Mutex::new(if extent.len == 0 { 0 } else { n_chunks }),
            done: Condvar::new(),
            error: Mutex::new(None),
            cancelled: AtomicBool::new(false),
            submitted: Instant::now(),
            finished_at: Mutex::new(None),
            finished_seq: AtomicU64::new(0),
        });
        if extent.len == 0 {
            return Ok(Ticket { job });
        }
        let mut q = self.shared.q.lock().unwrap_or_else(|p| p.into_inner());
        if priority == Priority::Speculative
            && q.speculative.len() + n_chunks > self.shared.cfg.max_speculative_queued
        {
            drop(q);
            self.shared
                .stats
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .speculative_refused += 1;
            return Err(Error::other("speculative queue full"));
        }
        let now = Instant::now();
        for i in 0..n_chunks {
            let start = i as u64 * payload;
            let c = Chunk {
                job: Arc::clone(&job),
                offset: extent.offset + start,
                len: payload.min(extent.len - start),
                dest_off: start as usize,
                queued: now,
            };
            match priority {
                Priority::Required => q.required.push_back(c),
                Priority::Speculative => q.speculative.push_back(c),
            }
        }
        drop(q);
        {
            let mut st = self.shared.stats.lock().unwrap_or_else(|p| p.into_inner());
            match priority {
                Priority::Required => st.required_jobs += 1,
                Priority::Speculative => st.speculative_jobs += 1,
            }
        }
        self.shared.work.notify_all();
        Ok(Ticket { job })
    }

    /// Stop handing out work (in-flight chunks finish). For quiescing and tests.
    pub fn pause(&self) {
        self.shared
            .q
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .paused = true;
    }

    pub fn resume(&self) {
        self.shared
            .q
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .paused = false;
        self.shared.work.notify_all();
    }

    pub fn stats(&self) -> LoaderStats {
        self.shared
            .stats
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Chunks waiting (required, speculative).
    pub fn queued(&self) -> (usize, usize) {
        let q = self.shared.q.lock().unwrap_or_else(|p| p.into_inner());
        (q.required.len(), q.speculative.len())
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.shared
            .q
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .shutdown = true;
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(s: Arc<Shared>) {
    loop {
        // Take work: required first; speculative only under its in-flight cap.
        let chunk = {
            let mut q = s.q.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if q.shutdown {
                    // Fail whatever is left so no waiter hangs.
                    let qs = &mut *q;
                    let left: Vec<Chunk> = qs
                        .required
                        .drain(..)
                        .chain(qs.speculative.drain(..))
                        .collect();
                    drop(q);
                    for c in left {
                        finish(&s, &c, Some("loader shut down".into()), false);
                    }
                    return;
                }
                if !q.paused {
                    if let Some(c) = q.required.pop_front() {
                        break c;
                    }
                    if q.spec_in_flight < s.cfg.max_speculative_in_flight.max(1) {
                        if let Some(c) = q.speculative.pop_front() {
                            q.spec_in_flight += 1;
                            break c;
                        }
                    }
                }
                q = s.work.wait(q).unwrap_or_else(|p| p.into_inner());
            }
        };
        let spec = chunk.job.priority == Priority::Speculative;

        if chunk.job.cancelled.load(Ordering::Acquire) {
            s.stats
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .cancelled_chunks += 1;
            finish(&s, &chunk, None, spec);
            continue;
        }

        let mut buf = s
            .pool
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .expect("one buffer per worker");
        let started = Instant::now();
        s.first_start
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_or_insert(started);
        let res = read_chunk(&s, &chunk, &mut buf);
        let read_time = started.elapsed();
        let err = match res {
            Ok((data_off, direct)) => {
                let mut d = chunk.job.dest.lock().unwrap_or_else(|p| p.into_inner());
                let n = chunk.len as usize;
                d[chunk.dest_off..chunk.dest_off + n]
                    .copy_from_slice(&buf.as_mut()[data_off..data_off + n]);
                drop(d);
                let mut st = s.stats.lock().unwrap_or_else(|p| p.into_inner());
                st.bytes += chunk.len;
                st.chunks += 1;
                if direct {
                    st.direct_chunks += 1;
                } else {
                    st.buffered_chunks += 1;
                }
                st.queue_wait += started - chunk.queued;
                st.read_time += read_time;
                let first = s
                    .first_start
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .unwrap_or(started);
                st.busy_wall = st.busy_wall.max(Instant::now() - first);
                None
            }
            Err(e) => {
                s.stats.lock().unwrap_or_else(|p| p.into_inner()).errors += 1;
                Some(e.to_string())
            }
        };
        s.pool.lock().unwrap_or_else(|p| p.into_inner()).push(buf);
        finish(&s, &chunk, err, spec);
    }
}

/// Account a chunk as done (successfully or not) and wake the waiter when
/// it was the job's last.
fn finish(s: &Shared, c: &Chunk, err: Option<String>, spec: bool) {
    if spec {
        let mut q = s.q.lock().unwrap_or_else(|p| p.into_inner());
        q.spec_in_flight = q.spec_in_flight.saturating_sub(1);
        drop(q);
        s.work.notify_all();
    }
    if let Some(e) = err {
        let mut slot = c.job.error.lock().unwrap_or_else(|p| p.into_inner());
        slot.get_or_insert(e);
    }
    let mut r = c.job.remaining.lock().unwrap_or_else(|p| p.into_inner());
    *r -= 1;
    if *r == 0 {
        *c.job.finished_at.lock().unwrap_or_else(|p| p.into_inner()) = Some(Instant::now());
        c.job
            .finished_seq
            .store(s.seq.fetch_add(1, Ordering::AcqRel) + 1, Ordering::Release);
        c.job.done.notify_all();
    }
}

/// Read one chunk. Returns where the payload starts inside the buffer and
/// whether the read was direct.
fn read_chunk(s: &Shared, c: &Chunk, buf: &mut AlignedBuf) -> std::io::Result<(usize, bool)> {
    let mode = *s.mode.lock().unwrap_or_else(|p| p.into_inner());
    if mode == ReadMode::Direct {
        let start = c.offset / IO_ALIGN * IO_ALIGN;
        let head = (c.offset - start) as usize;
        let end = (c.offset + c.len).div_ceil(IO_ALIGN) * IO_ALIGN;
        let want = (end - start) as usize;
        debug_assert!(want <= buf.len);
        match read_full(
            &s.file,
            &mut buf.as_mut()[..want],
            start,
            head + c.len as usize,
        ) {
            Ok(()) => return Ok((head, true)),
            // The filesystem refuses direct I/O here: fall back for good.
            Err(e) if e.raw_os_error() == Some(EINVAL) => {
                *s.mode.lock().unwrap_or_else(|p| p.into_inner()) = ReadMode::Buffered;
            }
            Err(e) => return Err(e),
        }
    }
    let n = c.len as usize;
    read_full(&s.buffered, &mut buf.as_mut()[..n], c.offset, n)?;
    Ok((0, false))
}

/// Read into `buf` at `offset` until at least `need` bytes are in. A direct
/// read of the file's last block returns short at EOF; that is fine as long
/// as the payload is covered.
fn read_full(f: &File, buf: &mut [u8], offset: u64, need: usize) -> std::io::Result<()> {
    let mut got = 0usize;
    while got < need {
        match f.read_at(&mut buf[got..], offset + got as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("short read at {}: {got} of {need}", offset + got as u64),
                ));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Tier;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// A file of `n` deterministic bytes, in a directory that may or may not
    /// support O_DIRECT (both paths are valid outcomes for these tests).
    fn file(n: usize, tag: &str) -> (Tmp, Vec<u8>) {
        use std::sync::atomic::AtomicUsize;
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "llamd-io-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let data: Vec<u8> = (0..n)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 9)) as u8)
            .collect();
        std::fs::write(&p, &data).unwrap();
        (Tmp(p), data)
    }

    fn ledger(bytes: u64) -> Arc<Ledger> {
        Ledger::with_caps(Tier::Ram, &[(Category::Staging, bytes)])
    }

    fn small(depth: usize, direct: bool) -> LoaderConfig {
        LoaderConfig {
            queue_depth: depth,
            chunk_bytes: 16 << 10,
            direct,
            max_speculative_in_flight: if depth > 1 { depth - 1 } else { 1 },
            max_speculative_queued: 64,
        }
    }

    #[test]
    fn every_byte_matches_a_plain_read_in_both_modes() {
        let (f, data) = file(3 << 20, "bytes");
        for direct in [true, false] {
            let l = Loader::new(&f.0, small(8, direct), &ledger(1 << 30)).unwrap();
            // Odd offsets and lengths: misaligned heads, tails, chunk seams,
            // and the file's very end.
            let cases = [
                (0u64, 1u64),
                (1, 4095),
                (4095, 2),
                (33, 100_000),
                (12_345, 16 << 10),
                (100_001, 777_777),
                ((3 << 20) - 5000, 5000),
                ((3 << 20) - 1, 1),
            ];
            let tickets: Vec<_> = cases
                .iter()
                .map(|&(o, n)| {
                    l.submit(Extent { offset: o, len: n }, Priority::Required, Vec::new())
                        .unwrap()
                })
                .collect();
            for (t, &(o, n)) in tickets.into_iter().zip(&cases) {
                let got = t.wait().unwrap();
                assert!(
                    got == data[o as usize..(o + n) as usize],
                    "direct={direct} mode={:?} extent {o}+{n}",
                    l.mode()
                );
            }
            let st = l.stats();
            assert_eq!(st.errors, 0);
            if !direct {
                assert_eq!(st.direct_chunks, 0);
            }
        }
    }

    #[test]
    fn the_staging_pool_is_reserved_up_front_or_not_at_all() {
        let (f, _) = file(64 << 10, "ledger");
        let cfg = small(8, false);
        let too_small = ledger(cfg.staging_bytes() - 1);
        assert!(Loader::new(&f.0, cfg.clone(), &too_small).is_err());
        assert_eq!(too_small.usage(Category::Staging).used, 0);

        let fits = ledger(cfg.staging_bytes());
        let l = Loader::new(&f.0, cfg.clone(), &fits).unwrap();
        assert_eq!(fits.usage(Category::Staging).used, cfg.staging_bytes());
        drop(l);
        assert_eq!(fits.usage(Category::Staging).used, 0, "released on drop");
    }

    #[test]
    fn a_required_read_overtakes_queued_speculative_work() {
        let (f, _) = file(4 << 20, "prio");
        let mut cfg = small(2, false);
        cfg.max_speculative_in_flight = 1;
        let l = Loader::new(&f.0, cfg, &ledger(1 << 30)).unwrap();
        l.pause();
        // Ten chunks of speculative work queued first...
        let spec = l
            .submit(
                Extent {
                    offset: 0,
                    len: 10 * (12 << 10),
                },
                Priority::Speculative,
                Vec::new(),
            )
            .unwrap();
        // ...then one required chunk.
        let req = l
            .submit(
                Extent {
                    offset: 1 << 20,
                    len: 1000,
                },
                Priority::Required,
                Vec::new(),
            )
            .unwrap();
        l.resume();
        let (rs, ss) = (
            req.finished_seq_after_wait(),
            spec.finished_seq_after_wait(),
        );
        assert!(rs < ss, "required finished #{rs}, speculative #{ss}");
    }

    #[test]
    fn speculative_submissions_are_refused_when_their_queue_is_full() {
        let (f, _) = file(4 << 20, "full");
        let mut cfg = small(2, false);
        cfg.max_speculative_queued = 4;
        let l = Loader::new(&f.0, cfg, &ledger(1 << 30)).unwrap();
        l.pause();
        let big = Extent {
            offset: 0,
            len: 5 * (12 << 10),
        }; // 5 chunks > 4
        assert!(l.submit(big, Priority::Speculative, Vec::new()).is_err());
        // Required work is never refused for queue length.
        let t = l.submit(big, Priority::Required, Vec::new()).unwrap();
        l.resume();
        t.wait().unwrap();
        assert_eq!(l.stats().speculative_refused, 1);
    }

    #[test]
    fn a_cancelled_load_reports_cancelled_and_skips_its_queued_chunks() {
        let (f, _) = file(4 << 20, "cancel");
        let l = Loader::new(&f.0, small(2, false), &ledger(1 << 30)).unwrap();
        l.pause();
        let t = l
            .submit(
                Extent {
                    offset: 0,
                    len: 40 * (12 << 10),
                },
                Priority::Speculative,
                Vec::new(),
            )
            .unwrap();
        t.cancel();
        l.resume();
        assert!(t.wait().is_err());
        assert!(l.stats().cancelled_chunks > 0);
    }

    #[test]
    fn extents_past_the_end_are_refused_at_submission() {
        let (f, _) = file(10_000, "eof");
        let l = Loader::new(&f.0, small(2, true), &ledger(1 << 30)).unwrap();
        assert!(
            l.submit(
                Extent {
                    offset: 9_000,
                    len: 2_000
                },
                Priority::Required,
                Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn a_config_that_could_starve_required_reads_is_refused() {
        let (f, _) = file(10_000, "cfg");
        let mut cfg = small(4, false);
        cfg.max_speculative_in_flight = 4;
        assert!(Loader::new(&f.0, cfg, &ledger(1 << 30)).is_err());
    }

    impl Ticket {
        fn finished_seq_after_wait(self) -> u64 {
            let job = Arc::clone(&self.job);
            let _ = self.wait();
            job.finished_seq.load(Ordering::Acquire)
        }
    }
}
