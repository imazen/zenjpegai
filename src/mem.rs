//! Tracked-byte accounting of the codec's large allocations.
//!
//! `#![forbid(unsafe_code)]` rules out a `#[global_allocator]`, so the ledger is kept where
//! the big allocations are made instead: `Tensor`/`BTensor` buffers, packed model weights,
//! the ANS payload copy, entropy-stage scratch, checkpoint bytes and output planes each own a
//! [`Charge`] — an RAII token whose drop releases the bytes — or pass through the `nn::fast`
//! buffer pool, which moves bytes between "live" and "parked" rather than freeing them.
//! Anything under ~64 KiB (indices, headers, table rows) is not tracked; it is noise next to
//! the picture-sized buffers this exists to bound.
//!
//! The counters are process-wide because the heap they describe is: the buffer pool and the
//! model caches are shared, and one decode's peak is the process's peak. [`MemoryWatch`]
//! reports what one call added on top of whatever was already held (under concurrent callers
//! it sees their traffic too — it measures the process, not the thread), and
//! `Decoder::memory_report` / `Encoder::memory_report` the cumulative high-water mark.
//!
//! Buffers handed to the caller — the decoded `RgbImage`/`YuvImage`, the encoded stream — are
//! counted while the codec holds them and leave the accounting at hand-off.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Bytes owned by live tracked buffers right now.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Bytes parked in the `nn::fast` buffer pool (still heap the codec holds).
static PARKED: AtomicUsize = AtomicUsize::new(0);
/// `LIVE + PARKED` kept in one atomic. Peaks are sampled off the `fetch_add` return of this
/// counter — a `LIVE + PARKED` sum of two loads can be torn by a concurrent drop (peak missed)
/// or `parked_move` (bytes counted twice), which under heavy rayon traffic measurably skews
/// the reported peak both ways. `TOTAL` leads its sub-counters on release so it never
/// over-reports a free that is in flight.
static TOTAL: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of `LIVE + PARKED` over the process lifetime.
static HIGH: AtomicUsize = AtomicUsize::new(0);

/// `LIVE + PARKED`: every tracked byte the codec currently holds.
#[inline]
fn total() -> usize {
    TOTAL.load(Ordering::Relaxed)
}

/// `n` bytes newly held. Updates the high-water mark and every live [`MemoryWatch`].
fn bump(n: usize) {
    let t = TOTAL.fetch_add(n, Ordering::Relaxed) + n;
    LIVE.fetch_add(n, Ordering::Relaxed);
    HIGH.fetch_max(t, Ordering::Relaxed);
    #[cfg(feature = "std")]
    if WATCHING.load(Ordering::Relaxed) != 0 {
        notify(t);
    }
}

/// `n` bytes released (freed — for parking see `parked_move`).
fn unbump(n: usize) {
    TOTAL.fetch_sub(n, Ordering::Relaxed);
    LIVE.fetch_sub(n, Ordering::Relaxed);
}

/// A parked buffer left the pool (`nn::fast` `take`, or a parked buffer freed by eviction or
/// [`crate::nn::fast::release_buffers`]): the pool's side of the ledger shrinks. A `take`n
/// buffer's new owner charges `LIVE` itself, so this only ever touches `PARKED`.
#[inline]
pub(crate) fn unparked(n: usize) {
    TOTAL.fetch_sub(n, Ordering::Relaxed);
    PARKED.fetch_sub(n, Ordering::Relaxed);
}

/// A live buffer was parked in the pool without changing the total: `n` moves from `LIVE` to
/// `PARKED`. The pool calls this with the bytes a [`Charge::release`] handed over.
#[inline]
pub(crate) fn parked_move(n: usize) {
    LIVE.fetch_sub(n, Ordering::Relaxed);
    PARKED.fetch_add(n, Ordering::Relaxed);
}

/// Bytes a `Vec` occupies, by capacity (what the allocator is holding for it).
#[inline]
pub(crate) fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity().saturating_mul(core::mem::size_of::<T>())
}

/// RAII token for `n` tracked bytes: [`bump`]ed on construction, [`unbump`]ed on drop.
///
/// Stored as a field next to the `Vec` it measures (`Tensor`, `BTensor`, the packed weight
/// structs, `AnsDecoder::buf`) or bound to a `_guard` local to cover a scratch allocation.
/// Cloning charges the same bytes again — a clone holds a second allocation of that size.
/// [`PartialEq`] is always true: the charge is accounting, not part of the container's value.
pub(crate) struct Charge(usize);

impl Charge {
    /// A token for nothing.
    pub(crate) const EMPTY: Self = Self(0);

    /// Charge `bytes` as newly held.
    pub(crate) fn new(bytes: usize) -> Self {
        bump(bytes);
        Self(bytes)
    }

    /// Charge `v`'s current capacity as newly held.
    pub(crate) fn of_vec<T>(v: &Vec<T>) -> Self {
        Self::new(vec_bytes(v))
    }

    /// Grow or shrink this charge to `bytes` (for buffers whose capacity changes in place).
    pub(crate) fn resize(&mut self, bytes: usize) {
        if bytes > self.0 {
            bump(bytes - self.0);
        } else {
            unbump(self.0 - bytes);
        }
        self.0 = bytes;
    }

    /// Hand the charge to the buffer pool: returns the bytes without discharging them
    /// (`parked_move` takes them from `LIVE` to `PARKED`).
    pub(crate) fn release(mut self) -> usize {
        core::mem::take(&mut self.0)
    }
}

impl Clone for Charge {
    fn clone(&self) -> Self {
        Self::new(self.0)
    }
}

impl Default for Charge {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        unbump(self.0);
    }
}

impl core::fmt::Debug for Charge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} tracked bytes", self.0)
    }
}

impl PartialEq for Charge {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for Charge {}

/// Measured heap use of a decode or encode, in bytes.
///
/// Counted are the codec's large allocations (see the module documentation); small headers,
/// indices and row tables are not tracked, so these numbers sit a few percent under a
/// heaptrack peak. Values are process-wide.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryReport {
    /// Most tracked bytes attributable to the measurement. Per call ([`MemoryWatch`]): the
    /// peak of `total - baseline` seen during the call — what the call added to the process
    /// heap at most. Cumulative (`memory_report` on the codec): the process's all-time
    /// high-water mark of tracked bytes.
    pub tracked_peak_bytes: u64,
    /// Tracked bytes still held (per call: what the call added and has not released).
    /// Includes buffers parked in the recycle pool; excludes what was handed to the caller.
    pub tracked_live_bytes: u64,
    /// The subset of `tracked_live_bytes` currently parked in the `nn::fast` recycle pool —
    /// not owned by any live object but still heap held (see `release_buffers` on the codec).
    pub pool_bytes: u64,
}

/// Per-call memory measurement (`std` only).
///
/// `MemoryWatch::new()` snapshots the current tracked total as the baseline; every later
/// charge raises the watch's peak to `total - baseline`. `report` reads the peak and what
/// the call still holds. Under concurrent decodes a watch sees the other calls' charges too —
/// it measures the process, so for a per-decode number decode sequentially or read it as an
/// upper bound.
///
/// ```ignore
/// let watch = MemoryWatch::new();
/// let image = decoder.decode(&stream)?;
/// let report = watch.report();
/// ```
#[cfg(feature = "std")]
pub struct MemoryWatch {
    inner: alloc::sync::Arc<Watch>,
}

#[cfg(feature = "std")]
struct Watch {
    /// `total()` when the watch started: growth is measured above it.
    baseline: AtomicUsize,
    /// Largest `total() - baseline` observed so far.
    peak: AtomicUsize,
}

/// Registered watches; dead entries are purged on registration.
#[cfg(feature = "std")]
static WATCHES: std::sync::Mutex<Vec<alloc::sync::Weak<Watch>>> = std::sync::Mutex::new(Vec::new());

/// Live [`MemoryWatch`]es — [`bump`] skips the registry lock when this is 0, which is the
/// common case (decodes only watch while reporting).
#[cfg(feature = "std")]
static WATCHING: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
impl MemoryWatch {
    /// Start measuring; registered until dropped.
    pub fn new() -> Self {
        let inner = alloc::sync::Arc::new(Watch {
            baseline: AtomicUsize::new(total()),
            peak: AtomicUsize::new(0),
        });
        // Incremented unconditionally — `Drop` decrements unconditionally. If the registry
        // lock is poisoned the watch still counts itself live (bump then takes the lock and
        // fails quietly, matching the unregistered case).
        WATCHING.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut w) = WATCHES.lock() {
            w.retain(|e| e.upgrade().is_some());
            w.push(alloc::sync::Arc::downgrade(&inner));
        }
        Self { inner }
    }

    /// Re-baseline at the current total and clear the peak.
    pub fn reset(&self) {
        self.inner.peak.store(0, Ordering::Relaxed);
        self.inner.baseline.store(total(), Ordering::Relaxed);
    }

    /// The measurement so far (or since `reset`).
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            tracked_peak_bytes: self.inner.peak.load(Ordering::Relaxed) as u64,
            tracked_live_bytes: total().saturating_sub(self.inner.baseline.load(Ordering::Relaxed))
                as u64,
            pool_bytes: PARKED.load(Ordering::Relaxed) as u64,
        }
    }
}

#[cfg(feature = "std")]
impl Default for MemoryWatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl Drop for MemoryWatch {
    fn drop(&mut self) {
        WATCHING.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "std")]
impl core::fmt::Debug for MemoryWatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemoryWatch")
            .field("report", &self.report())
            .finish()
    }
}

/// Notify every live watch that the tracked total is now `t`.
#[cfg(feature = "std")]
fn notify(t: usize) {
    let Ok(watches) = WATCHES.lock() else {
        return;
    };
    for w in watches.iter() {
        if let Some(w) = w.upgrade() {
            w.peak.fetch_max(
                t.saturating_sub(w.baseline.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
        }
    }
}

/// Cumulative report: all-time tracked high-water mark, bytes held now, bytes parked now.
#[cfg(feature = "std")]
pub(crate) fn cumulative() -> MemoryReport {
    MemoryReport {
        tracked_peak_bytes: HIGH.load(Ordering::Relaxed) as u64,
        tracked_live_bytes: total() as u64,
        pool_bytes: PARKED.load(Ordering::Relaxed) as u64,
    }
}
