//! Fuzz the whole decode path: `Decoder::decode_picture_with` on arbitrary
//! bytes, under tight `Limits` (1 MP, 512 MiB estimate, 1024px sides) with a
//! cooperative-cancellation token, against in-memory synthetic checkpoints.

#![no_main]
// Each bin exercises one entry point; the rest of common.rs serves the
// other targets and the regression test.
#![allow(dead_code)]

use libfuzzer_sys::fuzz_target;

mod common;

fuzz_target!(|data: &[u8]| {
    common::run_decode(data);
});
