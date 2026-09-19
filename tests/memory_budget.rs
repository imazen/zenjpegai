//! Deterministic unit tests for `MemoryBudget` admission control. These need no reference
//! vectors, so the file is not gated on `reference-tests`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use enough::{Stop, StopReason, Unstoppable};
use zenjpegai::{BudgetPolicy, Error, MemoryBudget};

/// A stop token the test flips from another thread.
struct FlagStop(Arc<AtomicBool>);

impl FlagStop {
    fn pair() -> (Arc<AtomicBool>, Self) {
        let flag = Arc::new(AtomicBool::new(false));
        (flag.clone(), Self(flag))
    }
}

impl Stop for FlagStop {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.load(Ordering::Relaxed) {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Spin until `cond` or panic after ~10 s (tests must never hang forever).
fn until(cond: impl Fn() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn acquire_counts_and_drop_releases() {
    let b = MemoryBudget::new(1_000);
    let g1 = b.acquire(400, &Unstoppable).unwrap();
    let g2 = b.acquire(600, &Unstoppable).unwrap();
    assert_eq!(b.held(), 1_000);
    assert_eq!(b.jobs(), 2);
    drop(g1);
    assert_eq!(b.held(), 600);
    assert_eq!(b.jobs(), 1);
    drop(g2);
    assert_eq!(b.held(), 0);
    assert_eq!(b.jobs(), 0);
}

#[test]
fn full_budget_blocks_then_frees() {
    let b = MemoryBudget::new(1_000);
    let g = b.acquire(1_000, &Unstoppable).unwrap();
    let b2 = b.clone();
    let t = thread::spawn(move || b2.acquire(1, &Unstoppable));
    until(|| b.waiting() == 1, "waiter to queue");
    thread::sleep(Duration::from_millis(20));
    assert_eq!(b.jobs(), 1, "waiter granted while budget is full");
    drop(g);
    let g2 = t.join().unwrap().unwrap();
    assert_eq!(b.jobs(), 1);
    drop(g2);
}

#[test]
fn grants_are_fifo() {
    // One job holds the whole budget; four waiters queue behind it. On release each waiter
    // needs the whole budget again, so the grant order is exactly the queue order.
    let b = MemoryBudget::new(1_000);
    let g = b.acquire(1_000, &Unstoppable).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for i in 0..4usize {
        let b2 = b.clone();
        let order2 = order.clone();
        handles.push(thread::spawn(move || {
            let _g = b2.acquire(1_000, &Unstoppable).unwrap();
            order2.lock().unwrap().push(i);
        }));
        // Deterministic queue order: wait for this waiter to be registered before
        // spawning the next, so tickets are 0,1,2,3.
        until(|| b.waiting() == i + 1, "waiter to queue");
    }
    drop(g);
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn queued_big_job_head_of_line_blocks_small() {
    // FIFO is strict: a small job that *would* fit cannot overtake a queued big job.
    let b = MemoryBudget::new(1_000);
    let g = b.acquire(700, &Unstoppable).unwrap();
    let b_big = b.clone();
    let big = thread::spawn(move || b_big.acquire(600, &Unstoppable));
    until(|| b.waiting() == 1, "big job to queue");
    // 300 bytes are free: a 100-byte job would fit, but it queued behind the big job.
    let b_small = b.clone();
    let small = thread::spawn(move || b_small.acquire(100, &Unstoppable));
    until(|| b.waiting() == 2, "small job to queue");
    thread::sleep(Duration::from_millis(20));
    assert_eq!(
        b.jobs(),
        1,
        "small job took the free bytes ahead of the queued big job (FIFO violated)"
    );
    drop(g);
    // Big is granted first, then small fits beside it (600 + 100 <= 1000).
    big.join().unwrap().unwrap();
    small.join().unwrap().unwrap();
}

#[test]
fn cancelled_waiter_leaves_the_queue() {
    let b = MemoryBudget::new(1_000);
    let g = b.acquire(1_000, &Unstoppable).unwrap();
    let (flag, stop) = FlagStop::pair();
    let b2 = b.clone();
    let t = thread::spawn(move || b2.acquire(1, &stop));
    until(|| b.waiting() == 1, "waiter to queue");
    flag.store(true, Ordering::Relaxed);
    match t.join().unwrap() {
        Err(Error::Cancelled(StopReason::Cancelled)) => {}
        other => panic!("expected Cancelled, got {other:?}"),
    }
    assert_eq!(b.waiting(), 0, "cancelled waiter left its ticket behind");
    drop(g);
    // The queue is clean: a new job acquires immediately.
    b.acquire(1_000, &Unstoppable).unwrap();
}

#[test]
fn fail_fast_reports_busy() {
    let b = MemoryBudget::new(1_000).with_policy(BudgetPolicy::FailFast);
    let g = b.acquire(800, &Unstoppable).unwrap();
    match b.acquire(300, &Unstoppable) {
        Err(Error::ResourceBusy(_)) => {}
        other => panic!("expected ResourceBusy, got {other:?}"),
    }
    // A job that still fits is admitted.
    let g2 = b.acquire(200, &Unstoppable).unwrap();
    drop((g, g2));
}

#[test]
fn oversize_refused_then_allowed_alone() {
    let b = MemoryBudget::new(1_000);
    match b.acquire(1_500, &Unstoppable) {
        Err(Error::LimitExceeded(_)) => {}
        other => panic!("expected LimitExceeded, got {other:?}"),
    }

    let b = MemoryBudget::new(1_000).with_allow_oversize(true);
    // While a normal job runs, the oversize job waits for a completely idle budget.
    let g = b.acquire(100, &Unstoppable).unwrap();
    let b2 = b.clone();
    let t = thread::spawn(move || b2.acquire(1_500, &Unstoppable));
    until(|| b.waiting() == 1, "oversize job to queue");
    thread::sleep(Duration::from_millis(20));
    assert_eq!(b.jobs(), 1, "oversize job joined a busy budget");
    drop(g);
    let g_big = t.join().unwrap().unwrap();
    assert_eq!(b.held(), 1_500, "oversize grant is still charged");
    // And while it runs, nothing else is admitted (held > budget).
    let b3 = b.clone();
    let t2 = thread::spawn(move || b3.acquire(1, &Unstoppable));
    until(|| b.waiting() == 1, "job to queue behind oversize");
    drop(g_big);
    t2.join().unwrap().unwrap();
    assert_eq!(b.held(), 0);
}

#[test]
fn max_jobs_serialises_small_jobs() {
    let b = MemoryBudget::new(1_000_000).with_max_jobs(1);
    let g = b.acquire(1, &Unstoppable).unwrap();
    let b2 = b.clone();
    let t = thread::spawn(move || b2.acquire(1, &Unstoppable));
    until(|| b.waiting() == 1, "second job to queue");
    thread::sleep(Duration::from_millis(20));
    assert_eq!(b.jobs(), 1, "max_jobs cap let a second job in");
    drop(g);
    t.join().unwrap().unwrap();
}

#[test]
fn total_granted_never_exceeds_budget() {
    // Eight threads × 400 bytes on a 1000-byte budget: at most two run at once. Each grantee
    // records the high-water mark of held() while it owns a grant.
    let b = MemoryBudget::new(1_000);
    let max_held = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let b2 = b.clone();
        let mh = max_held.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..5 {
                let _g = b2.acquire(400, &Unstoppable).unwrap();
                mh.fetch_max(b2.held(), Ordering::Relaxed);
                thread::sleep(Duration::from_millis(1));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(b.held(), 0);
    assert_eq!(b.jobs(), 0);
    assert!(
        max_held.load(Ordering::Relaxed) <= 1_000,
        "grants exceeded the budget: {}",
        max_held.load(Ordering::Relaxed)
    );
}
