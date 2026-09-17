//! Public `Decoder` API against real streams: cancellation, limits, memory estimate.
#![cfg(feature = "reference-tests")]

mod common;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{ref_root, vector_dir};
use zenjpegai::{Decoder, Error};

/// Trips once it has been asked `after` times.
struct StopAfter {
    after: usize,
    checks: AtomicUsize,
}

impl enough::Stop for StopAfter {
    fn check(&self) -> Result<(), enough::StopReason> {
        if self.checks.fetch_add(1, Ordering::Relaxed) >= self.after {
            Err(enough::StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}

fn big_stream() -> Vec<u8> {
    std::fs::read(vector_dir("img01_base_off_bpp050").join("stream.bits")).unwrap()
}

#[test]
fn stop_aborts_a_large_decode_early() {
    let stream = big_stream();
    let dec = Decoder::new(ref_root().join("models"));
    // How many checks a full 2096x1400 decode performs.
    let count = StopAfter {
        after: usize::MAX,
        checks: AtomicUsize::new(0),
    };
    let full = dec.decode_with(&stream, &count).unwrap();
    assert_eq!((full.width, full.height), (2096, 1400));
    let total = count.checks.load(Ordering::Relaxed);
    assert!(total >= 40, "only {total} stop checks in a 2.9 MP decode");

    // Trip at every eighth of the way: each must abort, at exactly the tripping check.
    for after in [0, 1, total / 8, total / 4, total / 2, total - 1] {
        let stop = StopAfter {
            after,
            checks: AtomicUsize::new(0),
        };
        let err = dec.decode_with(&stream, &stop).unwrap_err();
        assert_eq!(
            *err.error(),
            Error::Cancelled(enough::StopReason::Cancelled),
            "after {after}"
        );
        assert_eq!(stop.checks.load(Ordering::Relaxed), after + 1);
    }
    // The decoder is still usable and deterministic afterwards.
    assert_eq!(dec.decode(&stream).unwrap(), full);
}
