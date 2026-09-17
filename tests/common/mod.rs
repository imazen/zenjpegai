//! Shared helpers for the `reference-tests` integration tests.
#![allow(dead_code)]

use std::path::PathBuf;

/// Root of the upstream reference checkout. Hard failure if unset: these tests only compile with
/// `--features reference-tests`, and the caller (see `just test-ref`) owns that decision.
pub fn ref_root() -> PathBuf {
    let root = std::env::var_os("ZENJPEGAI_REF").expect("ZENJPEGAI_REF must point at the jpeg-ai-reference-software checkout (run via `just test-ref`)");
    let root = PathBuf::from(root);
    assert!(
        root.join("models").is_dir(),
        "{} has no models/ directory (git lfs pull?)",
        root.display()
    );
    root
}

pub fn read_model(rel: &str) -> Vec<u8> {
    let path = ref_root().join("models").join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
