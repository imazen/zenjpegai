//! Tracked-heap and budget tests against the reference vectors: `estimate_memory` must cover
//! the measured peak of every stream the decoder accepts, and a shared `MemoryBudget` must
//! serialise concurrent decodes at the estimate's granularity.

#![cfg(feature = "reference-tests")]

mod common;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use common::{ref_root, vector_dir, vectors_root};
use zenjpegai::encoder::{EncodeParams, Encoder, read_png_rgb8};
use zenjpegai::header::OperatingPoint;
use zenjpegai::{BudgetPolicy, Decoder, Error, MemoryBudget, estimate_encode_memory};

/// The tracked counters are process-wide and a `MemoryWatch` sees every thread's charges;
/// serialise every test in this file so a measurement cannot catch a neighbour's allocations.
static SERIAL: Mutex<()> = Mutex::new(());

/// Decode `name` twice: once with the recycle pool off (the tracked peak is then the pure
/// live-side peak `estimate.live_bytes` was calibrated against) and once with it on (the
/// peak includes retained pool bytes and is judged against `estimate.peak_bytes()`).
/// Returns `(pool_off_report, pooled_report, estimate)`. A fresh decoder each pass, so the
/// report includes loading the models the stream needs — the honest case the estimate's
/// fixed part has to cover.
fn measured(
    name: &str,
) -> (
    zenjpegai::MemoryReport,
    zenjpegai::MemoryReport,
    zenjpegai::MemoryEstimate,
) {
    let stream = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
    let pool = zenjpegai::nn::fast::pool_limit();
    let mut reports = Vec::new();
    let mut est = None;
    for limit in [0, pool] {
        zenjpegai::nn::fast::set_pool_limit(limit);
        zenjpegai::nn::fast::release_buffers();
        let decoder = Decoder::new(ref_root().join("models"));
        let e = decoder.estimate_memory(&stream).unwrap();
        let (_picture, report) = decoder
            .decode_picture_report(&stream, &enough::Unstoppable)
            .unwrap();
        reports.push(report);
        est = Some(e);
    }
    zenjpegai::nn::fast::set_pool_limit(pool);
    // The second estimate was taken under the real pool limit, so its peak_bytes() carries
    // the pool's true cap (live_bytes is the same in both).
    (reports[0], reports[1], est.unwrap())
}

/// The bench's six standard cases: 560x888 and 2096x1400 at each operating point.
#[test]
fn estimate_covers_tracked_peak_six_cases() {
    let _s = SERIAL.lock().unwrap();
    let cases = [
        ("img30_simple_off_bpp050", "560x888 SOP"),
        ("img30_base_off_bpp050", "560x888 BOP"),
        ("img30_high_off_bpp050", "560x888 HOP"),
        ("img01_simple_off_bpp050", "2096x1400 SOP"),
        ("img01_base_off_bpp050", "2096x1400 BOP"),
        ("img01_high_off_bpp050", "2096x1400 HOP"),
    ];
    eprintln!(
        "case\ttracked_peak_live\ttracked_peak_pooled\testimate_live_bytes\testimate_peak_bytes"
    );
    for (name, label) in cases {
        let (off, pooled, est) = measured(name);
        eprintln!(
            "{label}\t{}\t{}\t{}\t{}",
            off.tracked_peak_bytes,
            pooled.tracked_peak_bytes,
            est.live_bytes,
            est.peak_bytes()
        );
        assert!(off.tracked_peak_bytes > 0, "{name}: nothing was tracked");
        assert!(
            off.tracked_peak_bytes <= est.live_bytes,
            "{name}: live estimate {} < tracked live-side peak {}",
            est.live_bytes,
            off.tracked_peak_bytes
        );
        assert!(
            pooled.tracked_peak_bytes <= est.peak_bytes(),
            "{name}: estimate peak {} < tracked pooled peak {}",
            est.peak_bytes(),
            pooled.tracked_peak_bytes
        );
    }
}

/// The estimate must cover the tracked peak of *every* stream the decoder accepts — the
/// admission control hands out grants sized by the estimate, so a stream that under-estimates
/// would over-book the budget.
#[test]
fn estimate_covers_tracked_peak_every_vector() {
    let _s = SERIAL.lock().unwrap();
    let pool = zenjpegai::nn::fast::pool_limit();
    zenjpegai::nn::fast::set_pool_limit(0);
    let mut checked = 0;
    let mut skipped = Vec::new();
    let mut violations = Vec::new();
    for entry in std::fs::read_dir(vectors_root()).unwrap() {
        let dir = entry.unwrap().path();
        let stream_path = dir.join("stream.bits");
        if !stream_path.is_file() {
            continue;
        }
        let name = dir.file_name().unwrap().to_str().unwrap().to_owned();
        let stream = std::fs::read(&stream_path).unwrap();
        let decoder = Decoder::new(ref_root().join("models"));
        let Ok(est) = decoder.estimate_memory(&stream) else {
            // Not a decodable vector (e.g. the forced-4:2:0 eICCI stream, whose tool header a
            // conformant parser cannot read — it has its own dedicated test).
            skipped.push(name);
            continue;
        };
        let (_p, r) = decoder
            .decode_picture_report(&stream, &enough::Unstoppable)
            .unwrap_or_else(|e| panic!("{name}: estimate parsed but decode failed: {e:?}"));
        if r.tracked_peak_bytes > est.live_bytes {
            violations.push(format!(
                "{name}: estimate {} < tracked {}",
                est.live_bytes, r.tracked_peak_bytes
            ));
        }
        checked += 1;
    }
    zenjpegai::nn::fast::set_pool_limit(pool);
    eprintln!("estimate>=tracked checked on {checked} vectors; skipped {skipped:?}");
    assert!(checked >= 40, "only {checked} vectors checked");
    assert!(
        violations.is_empty(),
        "estimate under-covers the tracked peak:\n{}",
        violations.join("\n")
    );
}

/// A budget the size of one estimate admits exactly one decode at a time.
#[test]
fn budget_serialises_concurrent_decodes() {
    let _s = SERIAL.lock().unwrap();
    let stream = std::fs::read(vector_dir("img30_base_off_bpp050").join("stream.bits")).unwrap();
    let decoder = Decoder::new(ref_root().join("models"));
    let est = decoder.estimate_memory(&stream).unwrap();
    let budget = MemoryBudget::new(est.live_bytes);
    let decoder = Arc::new(decoder.budget(Some(budget.clone())));
    let stream = Arc::new(stream);

    // Continuous overlap is what we are testing for: held() sits at its grant for the whole
    // decode, so a 200 us sampler cannot miss a two-job overlap.
    let max_jobs = Arc::new(AtomicUsize::new(0));
    let max_held = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let (budget, max_jobs, max_held, done) = (
            budget.clone(),
            max_jobs.clone(),
            max_held.clone(),
            done.clone(),
        );
        thread::spawn(move || {
            while done.load(Ordering::Relaxed) < 4 {
                max_jobs.fetch_max(budget.jobs(), Ordering::Relaxed);
                max_held.fetch_max(budget.held(), Ordering::Relaxed);
                thread::sleep(Duration::from_micros(200));
            }
        })
    };
    let mut handles = Vec::new();
    for _ in 0..4 {
        let (d, s, done) = (decoder.clone(), stream.clone(), done.clone());
        handles.push(thread::spawn(move || {
            d.decode_picture(&s).unwrap();
            done.fetch_add(1, Ordering::Relaxed);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    monitor.join().unwrap();
    assert_eq!(max_jobs.load(Ordering::Relaxed), 1, "decodes overlapped");
    assert_eq!(max_held.load(Ordering::Relaxed), est.live_bytes);
    assert_eq!(budget.held(), 0, "a grant leaked");
}

/// Fail-fast: with the budget already granted away, a decode reports busy, not wrong.
#[test]
fn budget_fail_fast_decode() {
    let _s = SERIAL.lock().unwrap();
    let stream = std::fs::read(vector_dir("img30_base_off_bpp050").join("stream.bits")).unwrap();
    let decoder = Decoder::new(ref_root().join("models"));
    let est = decoder.estimate_memory(&stream).unwrap();
    let budget = MemoryBudget::new(est.live_bytes).with_policy(BudgetPolicy::FailFast);
    let decoder = decoder.budget(Some(budget.clone()));
    let _grant = budget
        .acquire(est.live_bytes, &enough::Unstoppable)
        .unwrap();
    match decoder.decode_picture(&stream) {
        Err(e) => assert!(
            matches!(e.error(), Error::ResourceBusy(_)),
            "expected ResourceBusy, got {e:?}"
        ),
        Ok(_) => panic!("decode ran while its budget was fully granted"),
    }
    drop(_grant);
    decoder.decode_picture(&stream).unwrap();
}

/// The tracked peak of a rate-matched encode stays under `estimate_encode_memory`, like the
/// decode side. `encode_to_bpp` runs several trial encodes internally — the estimate's
/// rate-matched variant covers exactly that.
#[test]
fn estimate_covers_tracked_peak_encode() {
    let _s = SERIAL.lock().unwrap();
    let image = "00030_TE_560x888_8bit_sRGB.png";
    let path = ref_root().join("data/test").join(image);
    let picture = read_png_rgb8(&std::fs::read(&path).unwrap()).unwrap();
    let (w, h) = (picture.width as u64, picture.height as u64);
    let encoder = Encoder::new(ref_root().join("models"));

    // Fixed-model encode.
    let (stream, r) = encoder
        .encode_report(
            picture.clone(),
            EncodeParams {
                op: OperatingPoint::Bop,
                ..Default::default()
            },
            &enough::Unstoppable,
        )
        .unwrap();
    let est = estimate_encode_memory(w, h, OperatingPoint::Bop, false);
    assert!(r.tracked_peak_bytes > 0);
    assert!(
        r.tracked_peak_bytes <= est.peak_bytes(),
        "encode: estimate {} < tracked peak {}",
        est.peak_bytes(),
        r.tracked_peak_bytes
    );

    // Rate-matched encode (`encode_to_bpp` keeps the source + several trial streams alive).
    let watch = zenjpegai::MemoryWatch::new();
    let (stream2, _m) = encoder
        .encode_to_bpp(
            &picture,
            0.5,
            EncodeParams {
                op: OperatingPoint::Bop,
                ..Default::default()
            },
        )
        .unwrap();
    let r2 = watch.report();
    let est2 = estimate_encode_memory(w, h, OperatingPoint::Bop, true);
    assert!(
        r2.tracked_peak_bytes <= est2.peak_bytes(),
        "rate-matched encode: estimate {} < tracked peak {}",
        est2.peak_bytes(),
        r2.tracked_peak_bytes
    );
    assert_ne!(stream, stream2, "rate matching did not re-encode");
}

/// One budget shared by a decoder and an encoder bounds their combined heap.
#[test]
fn budget_shared_by_decoder_and_encoder() {
    let _s = SERIAL.lock().unwrap();
    let stream = std::fs::read(vector_dir("img30_base_off_bpp050").join("stream.bits")).unwrap();
    let path = ref_root()
        .join("data/test")
        .join("00030_TE_560x888_8bit_sRGB.png");
    let picture = read_png_rgb8(&std::fs::read(&path).unwrap()).unwrap();
    let decoder = Decoder::new(ref_root().join("models"));
    let encoder = Encoder::new(ref_root().join("models"));
    // Sized to the bigger of the two estimates: each job fits alone, but not together.
    let budget = MemoryBudget::new(
        encoder
            .estimate_memory(560, 888, OperatingPoint::Bop, false)
            .live_bytes,
    )
    .with_policy(BudgetPolicy::FailFast);
    let decoder = decoder.budget(Some(budget.clone()));
    let encoder = encoder.budget(Some(budget.clone()));

    let _grant = budget
        .acquire(budget.bytes(), &enough::Unstoppable)
        .unwrap();
    assert!(matches!(
        decoder
            .decode_picture(&stream)
            .as_ref()
            .map_err(|e| e.error()),
        Err(Error::ResourceBusy(_))
    ));
    assert!(matches!(
        encoder
            .encode(
                &picture,
                EncodeParams {
                    op: OperatingPoint::Bop,
                    ..Default::default()
                },
            )
            .as_ref()
            .map_err(|e| e.error()),
        Err(Error::ResourceBusy(_))
    ));
    drop(_grant);
    decoder.decode_picture(&stream).unwrap();
    encoder
        .encode(
            &picture,
            EncodeParams {
                op: OperatingPoint::Bop,
                ..Default::default()
            },
        )
        .unwrap();
}
