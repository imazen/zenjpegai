//! Fuzz crash regression suite.
//!
//! Runs every file in `fuzz/regression/` through every decoder entry point that
//! has a fuzz target. Each seed is a previously-found crash/timeout that has
//! been fixed; this test makes sure none of them re-introduce a panic.
//!
//! It `include!`s `fuzz/fuzz_targets/common.rs` — the same code the libFuzzer
//! bins run — so replay can never drift from the fuzzed paths. Runs on stable
//! Rust, no nightly or sanitizer needed; needs the `unstable-internals`
//! feature for the internal parser/entropy entry points.
//!
//! To add a seed: minimize the crash (`cargo fuzz tmin <target> <artifact>`)
//! and drop the result into `fuzz/regression/` — no other action needed.

use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fuzz/fuzz_targets/common.rs"
));

fn regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression")
}

/// Every fuzz target entry point, in the order the fuzz targets run them.
type EntryPoint = (&'static str, fn(&[u8]));
const ENTRY_POINTS: &[EntryPoint] = &[
    ("container_headers", run_container_headers),
    ("entropy_stage", run_entropy),
    ("decode_full", run_decode),
];

fn run_one(name: &str, entry: fn(&[u8]), input: &[u8]) -> Result<(), String> {
    panic::catch_unwind(AssertUnwindSafe(|| entry(input))).map_err(|_| format!("{name} panicked"))
}

/// Every regression seed must go through every entry point without panicking.
/// Non-panic `Err` returns are the correct outcome — the decoder rejects
/// malformed input, it never crashes on it.
#[test]
fn regression_seeds_do_not_panic() {
    let dir = regression_dir();
    // The directory is committed (it holds `.gitkeep`): a missing one means a broken
    // checkout, and replaying nothing must not pass as if everything replayed clean.
    let rd = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: unreadable regression dir: {e}", dir.display()));
    let mut seeds: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    seeds.sort();
    for seed in &seeds {
        let input = fs::read(seed).expect("read regression seed");
        for (name, entry) in ENTRY_POINTS {
            if let Err(e) = run_one(name, *entry, &input) {
                panic!("{}: {e}", seed.display());
            }
        }
    }
}

/// The committed seed streams (`fuzz/seeds/`: TON, region, qmap, truncations)
/// are the campaign's starting corpus — replay them through every entry point
/// so the real syntax paths are exercised on stable CI too. Missing dir is a
/// broken checkout, same as `fuzz/regression`.
#[test]
fn seed_streams_do_not_panic() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds");
    let rd = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: unreadable seeds dir: {e}", dir.display()));
    let mut seeds: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    seeds.sort();
    assert!(!seeds.is_empty(), "{}: no seed streams", dir.display());
    for seed in &seeds {
        let input = fs::read(seed).expect("read seed stream");
        for (name, entry) in ENTRY_POINTS {
            if let Err(e) = run_one(name, *entry, &input) {
                panic!("{}: {e}", seed.display());
            }
        }
    }
}

/// A regression directory may be empty on a fresh clone — still smoke-test the
/// entry points on a handful of boundary inputs so the harness itself is
/// exercised by `cargo test`.
#[test]
fn smoke_inputs_do_not_panic() {
    let fixed: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0xFF],
        vec![0xFF, 0x80],
        // SOC EOC — empty codestream.
        vec![0xFF, 0x80, 0xFF, 0x81],
        // SOC PIH ue(0) EOC — empty picture header.
        vec![0xFF, 0x80, 0xFF, 0x82, 0x80, 0xFF, 0x81],
        // SOC PIH ue(1) 0x00 EOC — one zero byte of header.
        vec![0xFF, 0x80, 0xFF, 0x82, 0x40, 0x00, 0xFF, 0x81],
        // SOC PIH ue(4) zeros TON ue(0) SOZ ue(0) SORP ue(0) SORS ue(0) EOC.
        vec![
            0xFF, 0x80, 0xFF, 0x82, 0x20, 0, 0, 0, 0, 0xFF, 0x83, 0x80, 0xFF, 0x88, 0x80, 0xFF,
            0x89, 0x80, 0xFF, 0x8A, 0x80, 0xFF, 0x81,
        ],
    ];
    for (i, input) in fixed.iter().enumerate() {
        for (name, entry) in ENTRY_POINTS {
            if let Err(e) = run_one(name, *entry, input) {
                panic!("smoke input {i}: {e}");
            }
        }
    }
}
