//! Admission control over the heap bytes concurrent decodes and encodes may hold.
//!
//! [`Limits::max_memory_bytes`] bounds *one* job by [`estimate_memory`]; [`MemoryBudget`]
//! bounds the *sum* over every job that shares it. A [`Decoder`] or [`Encoder`] built with a
//! budget acquires the job's estimate before any picture-sized allocation and holds the grant
//! until the call returns — on success, error, or cancellation (an RAII [`BudgetGuard`]).
//! When the estimate would not fit, the call waits for room (`BudgetPolicy::Wait`, the
//! default) or fails fast (`BudgetPolicy::FailFast` → [`Error::ResourceBusy`]).
//!
//! Grants are strictly first-come-first-served: a job never overtakes one that queued before
//! it, so a stream of small jobs cannot starve a big one — but a queued big job does
//! head-of-line block smaller ones behind it. A job whose estimate exceeds the whole budget
//! is refused outright (`Error::LimitExceeded`) unless `allow_oversize` is set, in which case
//! it waits for a completely idle budget and then runs alone.
//!
//! ```ignore
//! let budget = MemoryBudget::new(2 << 30);            // 2 GiB across all users
//! let decoder = Decoder::new(dir).budget(budget.clone());
//! std::thread::scope(|s| for _ in 0..8 {
//!     s.spawn(|| decoder.decode(&stream));            // serialised by the budget
//! });
//! ```
//!
//! The grant is for the job's *estimate* (`estimate_memory` / `estimate_encode_memory`),
//! which has headroom over the measured peak, so `budget` bytes can over-book the actual
//! heap; the recycled-buffer pool is bounded separately (`nn::fast::set_pool_limit`) and is
//! not charged to jobs.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use crate::error::{Error, Result};

/// What a job does when its estimate does not currently fit the budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetPolicy {
    /// Wait until earlier jobs release enough bytes (or the job's `stop` token fires →
    /// [`Error::Cancelled`]). Strict FIFO.
    #[default]
    Wait,
    /// Refuse immediately with [`Error::ResourceBusy`].
    FailFast,
}

/// A shared, cloneable budget of heap bytes for concurrent decodes/encodes.
///
/// `Clone` shares the same underlying budget (it is an `Arc`): every clone grants and
/// releases against the same counters.
#[derive(Clone)]
pub struct MemoryBudget {
    inner: Arc<Inner>,
}

struct Inner {
    /// Total bytes the budget may have granted at once.
    bytes: u64,
    /// Most jobs granted at once (`usize::MAX` = unlimited).
    max_jobs: usize,
    policy: BudgetPolicy,
    /// A job bigger than `bytes` still runs — alone — when set.
    allow_oversize: bool,
    mu: Mutex<State>,
    cv: Condvar,
}

#[derive(Default)]
struct State {
    /// Bytes currently granted.
    used: u64,
    /// Jobs currently granted.
    jobs: usize,
    /// Next ticket number.
    next: u64,
    /// Waiting tickets, oldest first.
    queue: VecDeque<u64>,
}

/// A grant of `bytes` from a [`MemoryBudget`]; drop returns them.
pub struct BudgetGuard {
    inner: Arc<Inner>,
    /// Bytes this grant holds (also the oversize marker: > `inner.bytes` runs alone).
    bytes: u64,
}

impl core::fmt::Debug for BudgetGuard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BudgetGuard")
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl Drop for BudgetGuard {
    fn drop(&mut self) {
        let mut st = self.inner.mu.lock().unwrap_or_else(|e| e.into_inner());
        st.used = st.used.saturating_sub(self.bytes);
        st.jobs = st.jobs.saturating_sub(1);
        drop(st);
        self.inner.cv.notify_all();
    }
}

/// How long a waiter sleeps between re-checks: the `stop` token is polled, not signalled, so
/// the wait has a poll granularity. Decodes are tens of milliseconds and up; 5 ms is fine.
const WAIT_POLL_MS: u64 = 5;

impl MemoryBudget {
    /// A budget of `bytes` over all jobs that acquire from it: wait policy, no concurrency
    /// cap, oversize jobs refused.
    pub fn new(bytes: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                bytes,
                max_jobs: usize::MAX,
                policy: BudgetPolicy::Wait,
                allow_oversize: false,
                mu: Mutex::new(State::default()),
                cv: Condvar::new(),
            }),
        }
    }

    /// Cap the number of jobs granted at once, independent of bytes.
    pub fn with_max_jobs(mut self, jobs: usize) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("MemoryBudget shared before configuration")
            .max_jobs = jobs;
        self
    }

    /// What to do when a job does not fit right now (default [`BudgetPolicy::Wait`]).
    pub fn with_policy(mut self, policy: BudgetPolicy) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("MemoryBudget shared before configuration")
            .policy = policy;
        self
    }

    /// Let a single job larger than the whole budget run, alone. Without this it is refused
    /// with [`Error::LimitExceeded`].
    pub fn with_allow_oversize(mut self, allow: bool) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("MemoryBudget shared before configuration")
            .allow_oversize = allow;
        self
    }

    /// Total budget in bytes.
    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    /// Bytes currently granted (for diagnostics and tests).
    pub fn held(&self) -> u64 {
        self.inner.mu.lock().map(|s| s.used).unwrap_or(0)
    }

    /// Jobs currently granted (for diagnostics and tests).
    pub fn jobs(&self) -> usize {
        self.inner.mu.lock().map(|s| s.jobs).unwrap_or(0)
    }

    /// Jobs currently queued waiting for a grant (for diagnostics and tests).
    pub fn waiting(&self) -> usize {
        self.inner.mu.lock().map(|s| s.queue.len()).unwrap_or(0)
    }

    /// Acquire `estimate` bytes of budget, per the configured policy.
    ///
    /// Under [`BudgetPolicy::Wait`] this blocks until the job fits — strict FIFO order — or
    /// `stop` fires, which removes the ticket and returns [`Error::Cancelled`]. Under
    /// [`BudgetPolicy::FailFast`] it returns [`Error::ResourceBusy`] unless the job fits
    /// immediately.
    pub fn acquire(&self, estimate: u64, stop: &dyn enough::Stop) -> Result<BudgetGuard> {
        if estimate > self.inner.bytes && !self.inner.allow_oversize {
            return Err(Error::LimitExceeded(
                "job estimate exceeds the memory budget",
            ));
        }
        let oversize = estimate > self.inner.bytes;
        let inner = &*self.inner;
        // `into_inner` on poison: a panic while holding the lock leaves the queue state
        // structurally valid — grants and tickets are plain counters, not invariants a panic
        // can corrupt — so a poisoned mutex still serialises correctly.
        let mut st = inner.mu.lock().unwrap_or_else(|e| e.into_inner());
        let ticket = st.next;
        st.next += 1;
        st.queue.push_back(ticket);
        let can_run = |st: &State| {
            st.queue.front() == Some(&ticket)
                && if oversize {
                    // An oversize job runs alone: wait for a completely idle budget. It is
                    // still charged, so nothing else can join while it runs.
                    st.used == 0 && st.jobs == 0
                } else {
                    st.used + estimate <= inner.bytes && st.jobs < inner.max_jobs
                }
        };
        loop {
            if can_run(&st) {
                st.queue.pop_front();
                st.used += estimate;
                st.jobs += 1;
                return Ok(BudgetGuard {
                    inner: Arc::clone(&self.inner),
                    bytes: estimate,
                });
            }
            if inner.policy == BudgetPolicy::FailFast {
                Self::leave(inner, &mut st, ticket);
                return Err(Error::ResourceBusy(
                    "memory budget cannot fit the job right now",
                ));
            }
            if let Err(r) = stop.check() {
                Self::leave(inner, &mut st, ticket);
                return Err(Error::Cancelled(r));
            }
            let (guard, _t) = inner
                .cv
                .wait_timeout(st, std::time::Duration::from_millis(WAIT_POLL_MS))
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
        }
    }

    /// Drop `ticket` out of the queue (a waiter leaving before its grant).
    fn leave(inner: &Inner, st: &mut State, ticket: u64) {
        if let Some(pos) = st.queue.iter().position(|&t| t == ticket) {
            st.queue.remove(pos);
            inner.cv.notify_all();
        }
    }
}

impl core::fmt::Debug for MemoryBudget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let st = self.inner.mu.lock();
        f.debug_struct("MemoryBudget")
            .field("bytes", &self.inner.bytes)
            .field("held", &st.as_ref().map(|s| s.used).unwrap_or(0))
            .field("jobs", &st.as_ref().map(|s| s.jobs).unwrap_or(0))
            .field("policy", &self.inner.policy)
            .field("allow_oversize", &self.inner.allow_oversize)
            .finish()
    }
}
