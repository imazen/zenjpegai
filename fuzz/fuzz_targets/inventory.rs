//! Fuzz `DecodeJob::inventory`: it must not panic and the part list must tile the input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zencodec::decode::{DecodeJob, DecoderConfig};

fuzz_target!(|data: &[u8]| {
    if let Ok(Some(inv)) = zenjpegai::codec::JpegAiDecoderConfig::default()
        .job()
        .inventory(data)
    {
        inv.validate().expect("inventory must tile the input");
    }
});
