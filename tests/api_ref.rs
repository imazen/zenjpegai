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

/// Same as [`stop_aborts_a_large_decode_early`] but on a stream that enables post-filters
/// (tiled eICCI + LEF): `filters::apply` did not check `stop` at all until this test was added
/// (see `PORTING.md` "Zen codec standards"). eICCI runs a whole network per tile, so this also
/// exercises the per-tile check in `filters::icci::filter`, not just the once-per-filter one in
/// `filters::apply`.
#[test]
fn stop_aborts_a_filtered_decode_early() {
    let stream =
        std::fs::read(vector_dir("img01_base_eiccitiles_lef_bpp050").join("stream.bits")).unwrap();
    let dec = Decoder::new(ref_root().join("models"));
    let count = StopAfter {
        after: usize::MAX,
        checks: AtomicUsize::new(0),
    };
    let full = dec.decode_with(&stream, &count).unwrap();
    assert_eq!((full.width, full.height), (2096, 1400));
    let total = count.checks.load(Ordering::Relaxed);
    assert!(
        total >= 40,
        "only {total} stop checks in a 2.9 MP filtered decode"
    );

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
    assert_eq!(dec.decode(&stream).unwrap(), full);
}

/// Peak tracked heap of `zenjpegai decode --discard --pool-mb 0` (multi-thread — the estimate
/// covers concurrent-tile peaks and the cold model-load transient, so a single-thread
/// reference under-represents what it must bound), `benchmarks/memory_tracked_2026-09-19.tsv`
/// regime. The estimate must cover each and stay within 2x of it.
const MEASURED_LIVE: [(&str, u64); 4] = [
    ("img30_simple_off_bpp050", 40_640_000),
    ("img30_base_off_bpp050", 65_700_000),
    ("img30_high_off_bpp050", 443_340_000),
    ("img01_base_off_bpp050", 237_840_000),
];

#[test]
fn memory_estimate_covers_the_measured_peaks() {
    let dec = Decoder::new(ref_root().join("models"));
    for (name, measured) in MEASURED_LIVE {
        let stream = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
        let est = dec.estimate_memory(&stream).unwrap();
        assert!(
            est.live_bytes >= measured && est.live_bytes <= 2 * measured,
            "{name}: estimated {} for a measured {measured}",
            est.live_bytes
        );
        assert_eq!(est.peak_bytes(), est.live_bytes + est.pool_bytes);
    }
}

/// Peak heap of `zenjpegai encode --pool-mb 0 --single-thread`, heaptrack,
/// `benchmarks/memory_encode_2026-09-18.tsv` (label `encode-pool0`), on the reference's own
/// test images: (width, height, rate_matched, measured). The estimate must cover each and
/// stay within 2x of it.
const MEASURED_LIVE_ENCODE: [(u64, u64, bool, u64); 4] = [
    (560, 888, false, 266_470_000),
    (560, 888, true, 344_910_000),
    (2096, 1400, false, 593_750_000),
    (2096, 1400, true, 682_140_000),
];

#[test]
fn encode_memory_estimate_covers_the_measured_peaks() {
    use zenjpegai::header::OperatingPoint;
    for (w, h, rate_matched, measured) in MEASURED_LIVE_ENCODE {
        let est = zenjpegai::estimate_encode_memory(w, h, OperatingPoint::Bop, rate_matched);
        assert!(
            est.live_bytes >= measured && est.live_bytes <= 2 * measured,
            "{w}x{h} rate_matched={rate_matched}: estimated {} for a measured {measured}",
            est.live_bytes
        );
        assert_eq!(est.peak_bytes(), est.live_bytes + est.pool_bytes);
    }
}

/// `EncodeLimits` must refuse before any checkpoint or picture-sized buffer is touched.
#[test]
fn encode_limits_reject_from_the_size_alone() {
    use zenjpegai::EncodeLimits;
    // A models directory that does not exist: a limit violation must surface before any
    // checkpoint is touched.
    let nowhere = ref_root().join("no-such-models-dir");
    let image = zenjpegai::encoder::SourceImage::from(zenjpegai::RgbImage {
        width: 560,
        height: 888,
        bit_depth: 8,
        data: vec![0u16; 3 * 560 * 888],
    });
    for limits in [
        EncodeLimits::none().with_max_pixels(560 * 888 - 1),
        EncodeLimits::none().with_max_dimensions(559, 888),
        EncodeLimits::none().with_max_memory(100 << 20),
    ] {
        let err = zenjpegai::Encoder::new(&nowhere)
            .limits(limits)
            .encode(&image, zenjpegai::EncodeParams::default())
            .unwrap_err();
        assert!(matches!(err.error(), Error::LimitExceeded(_)), "{err:?}");
    }
}

#[test]
fn limits_reject_from_the_header_alone() {
    use zenjpegai::Limits;
    let stream = big_stream();
    // A models directory that does not exist: a limit violation must surface before any
    // checkpoint is touched.
    let nowhere = ref_root().join("no-such-models-dir");
    for limits in [
        Limits::none().with_max_pixels(2096 * 1400 - 1),
        Limits::none().with_max_dimensions(2095, 1400),
        Limits::none().with_max_memory(100 << 20),
        Limits::none().with_max_input_bytes(1000),
    ] {
        let err = Decoder::new(&nowhere)
            .limits(limits)
            .decode(&stream)
            .unwrap_err();
        assert!(matches!(err.error(), Error::LimitExceeded(_)), "{err:?}");
    }
    // Within bounds the same decoder gets as far as the missing checkpoints.
    let err = Decoder::new(&nowhere).decode(&stream).unwrap_err();
    assert!(matches!(err.error(), Error::Model(_)), "{err:?}");
}
