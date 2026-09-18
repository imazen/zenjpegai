//! Fuzz the codestream container and header parsers: marker splitting, PIH
//! (region/tiling/qmap syntax), TON tool headers, RDI, UDI and the
//! thread/region split helpers. Everything here is safe Rust over untrusted
//! bytes; a panic is a bug.

#![no_main]
// Each bin exercises one entry point; the rest of common.rs serves the
// other targets and the regression test.
#![allow(dead_code)]

use libfuzzer_sys::fuzz_target;

mod common;

fuzz_target!(|data: &[u8]| {
    common::run_container_headers(data);
});
