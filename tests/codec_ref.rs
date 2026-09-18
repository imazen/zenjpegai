//! `zencodec` trait bridge (`src/codec.rs`) against real streams: the zencodec-driven decode
//! must produce exactly the same pixels as `Decoder::decode` directly, and the zencodec-driven
//! encode exactly the same bytes as `Encoder::encode`.
#![cfg(all(feature = "zencodec", feature = "reference-tests"))]

mod common;
use std::sync::Arc;

use common::{ref_root, vector_dir};
use zencodec::CategorizedError;
use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zencodec::encode::{EncodeJob, Encoder as _, EncoderConfig as _};
use zenjpegai::codec::{JpegAiDecoderConfig, JpegAiEncoderConfig};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai::{EncodeLimits, EncodeParams};

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

// ===========================================================================
// Encode side
// ===========================================================================

fn models() -> Arc<dyn zenjpegai::model::ModelSource + Send + Sync> {
    Arc::new(ModelDir::new(ref_root().join("models")))
}

/// Decode `img01_base_off_bpp050` once and hand its 8-bit interleaved RGB bytes back — the
/// encode source every test below shares, so no PNG reader is needed.
fn decoded_rgb8() -> (zenjpegai::RgbImage, Vec<u8>) {
    let stream = std::fs::read(vector_dir("img01_base_off_bpp050").join("stream.bits")).unwrap();
    let img = zenjpegai::Decoder::new(ref_root().join("models"))
        .decode(&stream)
        .unwrap();
    assert!(img.bit_depth <= 8, "test assumes an 8-bit stream");
    (img.clone(), img.data.iter().map(|&s| s as u8).collect())
}

fn rgb8_slice<'a>(img: &zenjpegai::RgbImage, rgb: &'a [u8]) -> zenpixels::PixelSlice<'a> {
    zenpixels::PixelSlice::new(
        rgb,
        img.width as u32,
        img.height as u32,
        img.width * 3,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}

/// The E9 gate: the trait path encodes byte-for-byte what `Encoder::encode` does.
#[test]
fn encode_matches_the_plain_api_byte_for_byte() {
    let (img, rgb) = decoded_rgb8();
    let params = EncodeParams::default();

    let direct = zenjpegai::Encoder::new(ref_root().join("models"))
        .encode(&img, params)
        .unwrap();

    let out = JpegAiEncoderConfig::new(models(), Engine::new())
        .params(params)
        .job()
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap();
    assert_eq!(out.data(), direct.as_slice());
    // JPEG AI codestream: JPEG XS-style SOC/SOT markers 0xFF80 / 0xFF82.
    assert_eq!(&out.data()[..4], &[0xFF, 0x80, 0xFF, 0x82]);
}

/// `with_generic_quality` selects the rate matcher: the trait path must then return what
/// `Encoder::encode_to_bpp` returns, and surface the search's choice as a `RateMatch`
/// extension.
#[test]
fn rate_matched_encode_matches_the_plain_api() {
    let (img, rgb) = decoded_rgb8();
    let params = EncodeParams::default();

    let (direct, matched) = zenjpegai::Encoder::new(ref_root().join("models"))
        .encode_to_bpp(&img, 0.5, params)
        .unwrap();

    let config = JpegAiEncoderConfig::new(models(), Engine::new())
        .params(params)
        .with_generic_quality(50.0); // 50 / 100 = 0.5 bpp
    assert_eq!(config.generic_quality(), Some(50.0));
    let out = config
        .job()
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap();
    assert_eq!(out.data(), direct.as_slice());
    let reported = out
        .extensions()
        .get::<zenjpegai::encoder::RateMatch>()
        .unwrap();
    assert_eq!(*reported, matched);
}

/// 16-bit input takes the `Rgb16` branch of the slice conversion and is still byte-identical
/// to the plain API on the same `RgbImage`.
#[test]
fn encode_16bit_matches_the_plain_api() {
    let (img, _) = decoded_rgb8();
    // Re-label the 8-bit decode as 16-bit source samples (the encoder reads u16 lanes).
    let src = zenjpegai::RgbImage {
        bit_depth: 16,
        data: img.data.iter().map(|&s| s << 8).collect(),
        ..img.clone()
    };
    let params = EncodeParams::default();

    let direct = zenjpegai::Encoder::new(ref_root().join("models"))
        .encode(&src, params)
        .unwrap();

    let mut buf = zenpixels::PixelBuffer::new(
        src.width as u32,
        src.height as u32,
        zenpixels::PixelDescriptor::RGB16_SRGB,
    );
    {
        let mut slice = buf.as_slice_mut();
        for y in 0..slice.rows() {
            let row = slice.row_mut(y);
            for (dst, &s) in row
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&src.data[y as usize * src.width * 3..(y as usize + 1) * src.width * 3])
            {
                *dst = s.to_ne_bytes();
            }
        }
    }
    let out = JpegAiEncoderConfig::new(models(), Engine::new())
        .params(params)
        .job()
        .encoder()
        .unwrap()
        .encode(buf.as_slice())
        .unwrap();
    assert_eq!(out.data(), direct.as_slice());
}

/// A stop token that is already cancelled aborts the encode as `Stopped` — the same
/// `enough` mechanism the decode side uses.
#[test]
fn stop_token_cancels_encode() {
    struct Cancelled;
    impl enough::Stop for Cancelled {
        fn check(&self) -> Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }

    let (img, rgb) = decoded_rgb8();
    let err = JpegAiEncoderConfig::new(models(), Engine::new())
        .job()
        .with_stop(zencodec::StopToken::new(Cancelled))
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap_err();
    assert_eq!(
        err.category(),
        zencodec::ErrorCategory::Stopped(enough::StopReason::Cancelled)
    );
}

/// `with_limits` layers a per-job check on top of the config's: `max_pixels` below the
/// source refuses before any model work, `max_output_bytes` below the coded size refuses
/// after it, and generous limits leave the bytes alone.
#[test]
fn encode_job_limits_are_enforced() {
    let (img, rgb) = decoded_rgb8();
    let pixels = img.width as u64 * img.height as u64;
    let config = JpegAiEncoderConfig::new(models(), Engine::new());

    let err = config
        .clone()
        .job()
        .with_limits(zencodec::ResourceLimits::none().with_max_pixels(pixels - 1))
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap_err();
    assert_eq!(
        err.category(),
        zencodec::ErrorCategory::Resource(zencodec::ResourceError::Limits(
            zencodec::LimitKind::Pixels
        ))
    );

    // Smaller than any valid codestream → refused after coding.
    let err = config
        .clone()
        .job()
        .with_limits(zencodec::ResourceLimits::none().with_max_output(64))
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap_err();
    assert_eq!(
        err.category(),
        zencodec::ErrorCategory::Resource(zencodec::ResourceError::Limits(
            zencodec::LimitKind::OutputSize
        ))
    );

    // Generous limits are a pass-through.
    let direct = zenjpegai::Encoder::new(ref_root().join("models"))
        .encode(&img, EncodeParams::default())
        .unwrap();
    let out = config
        .job()
        .with_limits(
            zencodec::ResourceLimits::none()
                .with_max_pixels(pixels)
                .with_max_output(u64::MAX),
        )
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap();
    assert_eq!(out.data(), direct.as_slice());
}

/// The config's own `EncodeLimits` bake into the shared `Encoder` and refuse on their own.
#[test]
fn encode_config_limits_are_enforced() {
    let (img, rgb) = decoded_rgb8();
    let config = JpegAiEncoderConfig::with_limits(
        models(),
        Engine::new(),
        EncodeLimits::none().with_max_pixels(1024),
    );
    let err = config
        .job()
        .encoder()
        .unwrap()
        .encode(rgb8_slice(&img, &rgb))
        .unwrap_err();
    assert_eq!(
        err.category(),
        zencodec::ErrorCategory::Resource(zencodec::ResourceError::Limits(
            zencodec::LimitKind::Pixels
        ))
    );
}

/// Non-RGB / non-full-range / non-sRGB descriptors are refused as `PixelFormat`, never
/// silently mislabelled (the `RDI` this encoder writes carries no CICP).
#[test]
fn encode_refuses_unsupported_descriptors() {
    use zenpixels::PixelDescriptor;
    let config = JpegAiEncoderConfig::new(models(), Engine::new());
    for desc in [
        PixelDescriptor::GRAY8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB8_SRGB.with_signal_range(zenpixels::SignalRange::Narrow),
        PixelDescriptor::RGB16_BT2100_PQ,
    ] {
        // `PixelBuffer` (not a raw `Vec<u8>`) so 16-bit descriptors pass its alignment check.
        let buf = zenpixels::PixelBuffer::new(8, 8, desc);
        let pixels = buf.as_slice();
        let err = config
            .clone()
            .job()
            .encoder()
            .unwrap()
            .encode(pixels)
            .unwrap_err();
        assert_eq!(
            err.category(),
            zencodec::ErrorCategory::Request(zencodec::RequestError::Unsupported(
                zencodec::UnsupportedOperation::PixelFormat
            )),
            "{desc:?}"
        );
    }
}

/// `encode_srgba8` with `make_opaque` strips the padding alpha and encodes byte-for-byte what
/// the RGB path produces; meaningful alpha is refused.
#[test]
fn encode_srgba8_strips_padding_alpha() {
    let (img, rgb) = decoded_rgb8();
    let (w, h) = (img.width as u32, img.height as u32);
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.as_chunks::<3>().0 {
        rgba.extend_from_slice(px);
        rgba.push(0);
    }
    let params = EncodeParams::default();
    let direct = zenjpegai::Encoder::new(ref_root().join("models"))
        .encode(&img, params)
        .unwrap();

    let out = JpegAiEncoderConfig::new(models(), Engine::new())
        .params(params)
        .job()
        .encoder()
        .unwrap()
        .encode_srgba8(&mut rgba, true, w, h, w)
        .unwrap();
    assert_eq!(out.data(), direct.as_slice());

    // `make_opaque = false` declares the alpha meaningful → refused before any work.
    let err = JpegAiEncoderConfig::new(models(), Engine::new())
        .job()
        .encoder()
        .unwrap()
        .encode_srgba8(&mut rgba, false, w, h, w)
        .unwrap_err();
    assert_eq!(
        err.category(),
        zencodec::ErrorCategory::Request(zencodec::RequestError::Unsupported(
            zencodec::UnsupportedOperation::PixelFormat
        ))
    );
}

/// `estimate_encode_resources` reports the calibrated [`estimate_encode_memory`] model
/// verbatim (E8's live-vs-pool split maps onto est vs max).
#[test]
fn encode_resource_estimate_is_the_calibrated_model() {
    let (img, _) = decoded_rgb8();
    let image = zencodec::estimate::ImageCharacteristics::new(
        img.width as u32,
        img.height as u32,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    );
    let compute = zencodec::estimate::ComputeEnvironment::new().with_cores(1);
    let calibrated = zenjpegai::estimate_encode_memory(
        img.width as u64,
        img.height as u64,
        zenjpegai::header::OperatingPoint::Bop,
        false,
    );

    let est = JpegAiEncoderConfig::new(models(), Engine::new())
        .estimate_encode_resources(&image, &compute);
    assert_eq!(est.peak_memory_bytes_est(), Some(calibrated.live_bytes));
    assert_eq!(
        est.peak_memory_bytes_max(),
        Some(calibrated.live_bytes + calibrated.pool_bytes)
    );

    // A rate-matched encode estimates the rate-matched model and a much larger wall time.
    let est_rm = JpegAiEncoderConfig::new(models(), Engine::new())
        .target_bpp(Some(0.5))
        .estimate_encode_resources(&image, &compute);
    let calibrated_rm = zenjpegai::estimate_encode_memory(
        img.width as u64,
        img.height as u64,
        zenjpegai::header::OperatingPoint::Bop,
        true,
    );
    assert_eq!(
        est_rm.peak_memory_bytes_est(),
        Some(calibrated_rm.live_bytes)
    );
    assert!(est_rm.wall_ms().unwrap() > est.wall_ms().unwrap());
}
