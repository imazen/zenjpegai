//! `check_inventory` (valid input, 37 appended bytes, every truncation) on every reference
//! stream, which includes the `enc_*` vectors the encoder is pinned against.
#![cfg(all(feature = "zencodec", feature = "reference-tests"))]

mod common;

use zenjpegai::codec::JpegAiDecoderConfig;

#[test]
fn check_inventory_passes_on_every_reference_stream() {
    let dir = common::vectors_root();
    let mut streams: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path().join("stream.bits"))
        .filter(|p| p.is_file())
        .collect();
    streams.sort();
    assert!(
        streams.len() >= 80,
        "only {} streams under {}",
        streams.len(),
        dir.display()
    );
    for s in &streams {
        let data = std::fs::read(s).unwrap();
        zencodec_testkit::check_inventory(JpegAiDecoderConfig::default(), &data)
            .unwrap_or_else(|e| panic!("{}: {e:?}", s.display()));
    }
    eprintln!(
        "check_inventory passed on {} reference streams",
        streams.len()
    );
}
