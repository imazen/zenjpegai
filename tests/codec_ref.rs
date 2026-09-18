//! `zencodec` trait bridge (`src/codec.rs`) against real streams: the zencodec-driven decode
//! must produce exactly the same pixels as `Decoder::decode` directly.
#![cfg(all(feature = "zencodec", feature = "reference-tests"))]

mod common;
use std::sync::Arc;

use common::{ref_root, vector_dir};
use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zenjpegai::codec::JpegAiDecoderConfig;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;

#[test]
fn decode_matches_the_plain_api_bit_for_bit() {
    let stream = std::fs::read(vector_dir("img01_base_off_bpp050").join("stream.bits")).unwrap();

    let direct = zenjpegai::Decoder::new(ref_root().join("models"));
    let want = direct.decode(&stream).unwrap();

    let models: Arc<dyn zenjpegai::model::ModelSource + Send + Sync> =
        Arc::new(ModelDir::new(ref_root().join("models")));
    let config = JpegAiDecoderConfig::new(models, Engine::new());
    let job = config.clone().job();
    let info = job.probe(&stream).unwrap();
    assert_eq!(
        (info.width, info.height),
        (want.width as u32, want.height as u32)
    );

    let job = config.job();
    let output = job
        .decoder(std::borrow::Cow::Borrowed(stream.as_slice()), &[])
        .unwrap()
        .decode()
        .unwrap();
    let pixels = output.pixels();
    assert_eq!(pixels.width() as usize, want.width);
    assert_eq!(pixels.rows() as usize, want.height);

    // `want.data` is interleaved u16 RGB at `want.bit_depth`; the bridge narrows to u8 when
    // `bit_depth <= 8` (this stream is 8-bit).
    assert!(want.bit_depth <= 8, "test assumes an 8-bit stream");
    let mut expected = Vec::with_capacity(want.data.len());
    for &s in &want.data {
        expected.push(s as u8);
    }
    let mut got = Vec::with_capacity(expected.len());
    for row in 0..pixels.rows() {
        got.extend_from_slice(pixels.row(row));
    }
    assert_eq!(got, expected);
}

/// `estimate_decode_resources` should at least be in the right ballpark of the calibrated
/// `Decoder::estimate_memory` for the same stream (see `src/codec.rs`'s second documented
/// mismatch — it cannot call the calibrated model directly, so this only checks order of
/// magnitude, not tight agreement).
#[test]
fn resource_estimate_is_same_order_of_magnitude_as_the_calibrated_one() {
    let stream = std::fs::read(vector_dir("img01_base_off_bpp050").join("stream.bits")).unwrap();
    let direct = zenjpegai::Decoder::new(ref_root().join("models"));
    let calibrated = direct.estimate_memory(&stream).unwrap();

    let models: Arc<dyn zenjpegai::model::ModelSource + Send + Sync> =
        Arc::new(ModelDir::new(ref_root().join("models")));
    let config = JpegAiDecoderConfig::new(models, Engine::new());
    let image = zencodec::estimate::ImageCharacteristics::new(
        2096,
        1400,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    );
    let compute = zencodec::estimate::ComputeEnvironment::new().with_cores(1);
    let est = config.estimate_decode_resources(&image, &compute);
    let structural = est.peak_memory_bytes_est().unwrap();
    let calibrated_live = calibrated.live_bytes;
    let ratio = structural as f64 / calibrated_live as f64;
    assert!(
        (0.1..10.0).contains(&ratio),
        "structural estimate {structural} vs calibrated {calibrated_live} (ratio {ratio})"
    );
}
