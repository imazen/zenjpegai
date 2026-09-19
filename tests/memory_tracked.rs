//! Accounting tests for the tracked-heap ledger (`src/mem.rs`) through the public surface:
//! `MemoryWatch` deltas plus `nn::fast` tensors and the recycle pool. No reference vectors.
//!
//! Everything runs under one lock: the counters are process-wide, so two tests charging at
//! once would see each other's bytes.

#![cfg(all(feature = "std", feature = "unstable-internals"))]

use std::sync::Mutex;

use zenjpegai::MemoryWatch;
use zenjpegai::nn::fast::{BTensor, release_buffers, set_pool_limit};
use zenjpegai::tensor::Tensor;

static SERIAL: Mutex<()> = Mutex::new(());

/// Default pool cap (1 GiB as floats), restored by `clean` so a test's pool setting cannot
/// leak into the next.
const DEFAULT_POOL_BYTES: usize = 1 << 30;

/// Start each test from an empty pool and the default limit.
fn clean() {
    release_buffers();
    set_pool_limit(DEFAULT_POOL_BYTES);
}

#[test]
fn tensor_charges_live_and_releases_on_drop() {
    let _s = SERIAL.lock().unwrap();
    clean();
    let w = MemoryWatch::new();
    let floats = 1 << 20; // 4 MiB
    let t = BTensor::scratch(1024, 32, 32, 8).unwrap();
    assert_eq!(t.data.len(), floats);
    let held = w.report().tracked_live_bytes;
    assert!(
        held >= floats as u64 * 4,
        "scratch not charged: held {held}"
    );
    drop(t);
    // The buffer parked: still held (parked counts as held) but now on the pool side.
    let r = w.report();
    assert!(r.pool_bytes >= floats as u64 * 4, "not parked: {r:?}");
    release_buffers();
    let r = w.report();
    assert_eq!(r.pool_bytes, 0);
    assert!(
        r.tracked_live_bytes < floats as u64 * 4,
        "parked buffer still counted after release: {r:?}"
    );
}

#[test]
fn pooled_buffer_is_reused_not_double_counted() {
    let _s = SERIAL.lock().unwrap();
    clean();
    let w = MemoryWatch::new();
    let floats = 1 << 20;
    drop(BTensor::scratch(1024, 32, 32, 8).unwrap()); // parks 4 MiB
    let parked = w.report().pool_bytes;
    assert!(parked >= floats as u64 * 4);
    // Same size again: takes the parked buffer — total held must not grow.
    let before = w.report().tracked_live_bytes;
    let t = BTensor::scratch(1024, 32, 32, 8).unwrap();
    let after = w.report().tracked_live_bytes;
    assert!(
        after <= before,
        "pool take double-charged: before {before}, after {after}"
    );
    assert!(
        w.report().pool_bytes < parked,
        "taken buffer still on the pool's side"
    );
    drop(t);
    release_buffers();
    assert_eq!(w.report().tracked_live_bytes, 0);
}

#[test]
fn pool_limit_zero_frees_instead_of_parking() {
    let _s = SERIAL.lock().unwrap();
    clean();
    set_pool_limit(0);
    let w = MemoryWatch::new();
    drop(BTensor::scratch(1024, 32, 32, 8).unwrap());
    let r = w.report();
    assert_eq!(r.pool_bytes, 0);
    assert_eq!(r.tracked_live_bytes, 0, "buffer held with pool off: {r:?}");
}

#[test]
fn planar_tensor_charges_and_releases() {
    let _s = SERIAL.lock().unwrap();
    clean();
    let w = MemoryWatch::new();
    let bytes = (3 * 64 * 64 * 4) as u64; // 3×64×64 f32
    let t = Tensor::<f32>::zeros(3, 64, 64).unwrap();
    assert!(w.report().tracked_live_bytes >= bytes);
    drop(t);
    assert_eq!(
        w.report().tracked_live_bytes,
        0,
        "plain Tensor does not pool: drop must release"
    );
}

#[test]
fn watch_reset_rebaselines() {
    let _s = SERIAL.lock().unwrap();
    clean();
    let w = MemoryWatch::new();
    let _t = BTensor::scratch(1024, 32, 32, 8).unwrap();
    assert!(w.report().tracked_peak_bytes >= 4 * (1 << 20));
    w.reset();
    assert_eq!(w.report().tracked_peak_bytes, 0);
    // The baseline moved up over the live tensor, so its bytes no longer count as growth.
    assert_eq!(w.report().tracked_live_bytes, 0);
    release_buffers();
}

#[test]
fn watch_sees_peak_not_just_live() {
    let _s = SERIAL.lock().unwrap();
    clean();
    set_pool_limit(0); // no parking: drops really free
    let w = MemoryWatch::new();
    {
        let _big = BTensor::scratch(1024, 64, 64, 8).unwrap(); // 16 MiB
        let _small = BTensor::scratch(1024, 32, 32, 8).unwrap(); // 4 MiB
    }
    let r = w.report();
    assert_eq!(r.tracked_live_bytes, 0);
    assert!(
        r.tracked_peak_bytes >= 20 * (1 << 20) as u64,
        "peak missed a transient allocation: {r:?}"
    );
}
