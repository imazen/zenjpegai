//! Fuzz the entropy stage on arbitrary codestreams: me-tANS z/residual decode,
//! thread and region splitting, hyper-scale decoder, gain/RVS/GRFS and the
//! quality map, against fixed small synthetic models (no trained weights).

#![no_main]
// Each bin exercises one entry point; the rest of common.rs serves the
// other targets and the regression test.
#![allow(dead_code)]

use libfuzzer_sys::fuzz_target;

mod common;

fuzz_target!(|data: &[u8]| {
    common::run_entropy(data);
});
