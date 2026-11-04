//! Encode→decode roundtrips across the source-format matrix. The sources are
//! synthetic `SourceImage`s — no reference vectors needed, only the model
//! checkpoints (`ref_root()/models`, the same `ZENJPEGAI_REF` the other
//! reference tests use). Each case encodes at a fixed model/operating point and
//! decodes back, asserting the output format matches the source's
//! (`to_format_` / `to_source_format_par` coded→source on the decode side).

mod common;

use common::ref_root;
use zenjpegai::encoder::{EncodeParams, SourceImage};
use zenjpegai::{Decoder, Encoder, Picture, RgbImage, YuvImage};

fn models() -> std::path::PathBuf {
    ref_root().join("models")
}

/// A deterministic gradient plane — lossy encode/decode only needs the format
/// to survive, but a non-trivial pattern exercises the chroma paths better than
/// a constant.
fn plane(len: usize, bit_depth: u8, seed: usize) -> Vec<u16> {
    let max = (1usize << bit_depth) - 1;
    (0..len)
        .map(|i| ((i * 7 + seed * 131 + i / 97 * 31) & max) as u16)
        .collect()
}

/// A planar YUV source at `w×h`, subsampling `s`, bit depth `bd`.
fn yuv(w: usize, h: usize, s: (u8, u8), bd: u8) -> SourceImage {
    let (cw, ch) = (w.div_ceil(s.1 as usize), h.div_ceil(s.0 as usize));
    SourceImage::Yuv(YuvImage {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth: bd,
        y: plane(w * h, bd, 1),
        u: plane(cw * ch, bd, 2),
        v: plane(cw * ch, bd, 3),
    })
}

fn rgb(w: usize, h: usize, bd: u8) -> SourceImage {
    SourceImage::Rgb(RgbImage {
        width: w,
        height: h,
        bit_depth: bd,
        data: plane(w * h * 3, bd, 4),
    })
}

fn roundtrip(src: SourceImage, params: EncodeParams) -> Picture {
    let stream = Encoder::new(models())
        .encode(src, params)
        .expect("encode failed");
    Decoder::new(models())
        .decode_picture(&stream)
        .expect("decode failed")
}

fn assert_yuv(pic: &Picture, w: usize, h: usize, s: (u8, u8), bd: u8) {
    let Picture::Yuv(y) = pic else {
        panic!("expected Picture::Yuv, got Picture::Rgb");
    };
    assert_eq!((y.width, y.height), (w, h));
    assert_eq!(
        (y.chroma_width, y.chroma_height),
        (w.div_ceil(s.1 as usize), h.div_ceil(s.0 as usize)),
        "decoded chroma plane size must match the source subsampling"
    );
    assert_eq!(y.bit_depth, bd);
    assert_eq!(y.y.len(), w * h);
    assert_eq!(y.u.len(), y.chroma_width * y.chroma_height);
}

/// Every YUV source subsampling × both bit depths, even and odd dimensions.
#[test]
fn yuv_sources_roundtrip_at_source_format() {
    for &(w, h) in &[(96, 64), (203, 301)] {
        for &s in &[(1u8, 1u8), (1, 2), (2, 2)] {
            for &bd in &[8u8, 10] {
                let pic = roundtrip(yuv(w, h, s, bd), EncodeParams::default());
                assert_yuv(&pic, w, h, s, bd);
            }
        }
    }
}

/// 8-bit RGB sources decode back to `Picture::Rgb` (`colour_transform_idx = 1`).
/// A 16-bit RGB source encodes `bit_depth_idc = 4` (upstream `read_png` accepts
/// it) but the decoder rejects it — `assert image_data_bits_idx in [0, 1]` is
/// upstream's own check; our `InvalidData` mirrors it.
#[test]
fn rgb_sources_roundtrip_to_rgb() {
    let pic = roundtrip(rgb(96, 72, 8), EncodeParams::default());
    let Picture::Rgb(r) = pic else {
        panic!("expected Picture::Rgb for an RGB source");
    };
    assert_eq!((r.width, r.height), (96, 72));
    assert_eq!(r.data.len(), 96 * 72 * 3);
}

/// The upstream asymmetry documented above: 16-bit sources encode
/// (`bit_depth_idc = 4`) but no decoder — ours or the reference's — accepts a
/// depth outside `BIT_DEPTHS[0..2]`.
#[test]
fn rgb16_encodes_but_decoders_reject() {
    let stream = Encoder::new(models())
        .encode(rgb(96, 72, 16), EncodeParams::default())
        .expect("16-bit RGB encode should succeed (upstream accepts it)");
    let err = Decoder::new(models())
        .decode_picture(&stream)
        .expect_err("16-bit streams must be rejected, matching upstream's assert");
    assert!(format!("{err:?}").contains("8 and 10"));
}

/// A 4:4:4 source coded at 4:2:0 (`c_ver`/`c_hor` overrides) decodes back in
/// the source's 4:4:4 format — `to_format_`'s bicubic upsample arm.
#[test]
fn coded_below_source_upsamples_back() {
    let params = EncodeParams {
        c_ver: Some(2),
        c_hor: Some(2),
        ..EncodeParams::default()
    };
    let pic = roundtrip(yuv(96, 64, (1, 1), 8), params);
    assert_yuv(&pic, 96, 64, (1, 1), 8);
}

/// The rejected coded-format combinations still error at encode time
/// (`SourceMeta::resolve`): 4:2:2 source coded 4:2:0 and coded chroma finer
/// than the source.
#[test]
fn rejected_coded_formats_error() {
    let enc = Encoder::new(models());
    // 4:2:2 source coded 4:2:0 — upstream `NotImplementedError`.
    assert!(
        enc.encode(
            yuv(64, 64, (1, 2), 8),
            EncodeParams {
                c_ver: Some(2),
                c_hor: Some(2),
                ..EncodeParams::default()
            },
        )
        .is_err()
    );
    // Coded chroma finer than the source (4:2:0 source at 4:4:4).
    assert!(
        enc.encode(
            yuv(64, 64, (2, 2), 8),
            EncodeParams {
                c_ver: Some(1),
                c_hor: Some(1),
                ..EncodeParams::default()
            },
        )
        .is_err()
    );
}
