//! Packed model bundles (`ZJB1` / `ZJM1`): decoding from a bundle yields the same bytes as
//! decoding from the upstream `.pth` files, and a bundle holds nothing the decoder does not read.
#![cfg(feature = "reference-tests")]

mod common;
use common::{ref_root, vector_dir};
use zenjpegai::Decoder;
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{self, ModelDir};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::{PackedBundle, Recorder};

fn pack(ids: &[usize], ops: &[OperatingPoint], common: bool, synthesis: bool) -> Vec<u8> {
    let eng = Engine::with(Tier::Scalar, false);
    let rec = Recorder::new(ModelDir::new(ref_root().join("models")));
    for &id in ids {
        for ccs in 0..2 {
            if common {
                model::load_common(&rec, id, ccs, &eng).unwrap();
            }
        }
        for &op in ops.iter().filter(|_| synthesis) {
            model::load_synthesis_primary(&rec, id, op, &eng).unwrap();
            model::load_synthesis_secondary(&rec, id, op, &eng).unwrap();
        }
    }
    rec.pack("notice text").unwrap()
}

#[test]
fn bundle_decodes_bit_identically() {
    use OperatingPoint::*;
    // (vector, model id, operating point); model ids as printed by `zenjpegai info`.
    for (name, op) in [
        ("img30_base_off_bpp050", Bop),
        ("img30_simple_off_bpp050", Sop),
        ("img30_high_off_bpp050", Hop),
    ] {
        let stream = std::fs::read(vector_dir(name).join("stream.bits")).unwrap();
        let engine = Engine::new();
        let from_pth = Decoder::with_engine(ref_root().join("models"), engine);
        let id = from_pth.read_headers(&stream).unwrap().picture.model_id as usize;
        let expect = from_pth.decode(&stream).unwrap();

        let bytes = pack(&[id], &[op], true, true);
        let bundle = PackedBundle::parse(bytes).unwrap();
        assert_eq!(bundle.paths().count(), 4, "{name}");
        assert_eq!(bundle.notice(), "notice text");
        let got = Decoder::with_source(Box::new(bundle), engine)
            .decode(&stream)
            .unwrap();
        assert_eq!((got.width, got.height), (expect.width, expect.height));
        assert!(got.data == expect.data, "{name}: bundle pixels differ");

        // The same from a common part and a synthesis part fetched separately.
        let mut split = PackedBundle::parse(pack(&[id], &[op], true, false)).unwrap();
        assert_eq!(split.paths().count(), 2);
        split.add(pack(&[id], &[op], false, true)).unwrap();
        assert_eq!(split.paths().count(), 4);
        let got = Decoder::with_source(Box::new(split), engine)
            .decode(&stream)
            .unwrap();
        assert!(
            got.data == expect.data,
            "{name}: split bundle pixels differ"
        );
    }
}

#[test]
fn repacking_a_bundle_is_a_fixed_point() {
    // Every tensor in a bundle is one the loaders look up: packing from the bundle itself
    // reproduces it byte for byte.
    let first = pack(&[1], &[OperatingPoint::Bop], true, true);
    let eng = Engine::with(Tier::Scalar, false);
    let rec = Recorder::new(PackedBundle::parse(first.clone()).unwrap());
    for ccs in 0..2 {
        model::load_common(&rec, 1, ccs, &eng).unwrap();
    }
    model::load_synthesis_primary(&rec, 1, OperatingPoint::Bop, &eng).unwrap();
    model::load_synthesis_secondary(&rec, 1, OperatingPoint::Bop, &eng).unwrap();
    assert!(rec.pack("notice text").unwrap() == first);
}

#[test]
fn damaged_bundles_are_rejected() {
    let bytes = pack(&[1], &[OperatingPoint::Sop], false, true);
    assert!(PackedBundle::parse(bytes[..bytes.len() - 100].to_vec()).is_err());
    assert!(PackedBundle::parse(bytes[..40].to_vec()).is_err());
    let mut bad = bytes.clone();
    bad[4] = 9; // version
    assert!(PackedBundle::parse(bad).is_err());
    assert!(PackedBundle::parse(b"ZJB1".to_vec()).is_err());
    let mut bad2 = pack_f16(&[1], &[OperatingPoint::Sop], false, true);
    bad2[4] = 9;
    assert!(PackedBundle::parse(bad2).is_err());
    assert!(PackedBundle::parse(b"ZJB2".to_vec()).is_err());
}

fn pack_f16(ids: &[usize], ops: &[OperatingPoint], common: bool, synthesis: bool) -> Vec<u8> {
    let eng = Engine::with(Tier::Scalar, false);
    let rec = Recorder::new(ModelDir::new(ref_root().join("models")));
    for &id in ids {
        for ccs in 0..2 {
            if common {
                model::load_common(&rec, id, ccs, &eng).unwrap();
            }
        }
        for &op in ops.iter().filter(|_| synthesis) {
            model::load_synthesis_primary(&rec, id, op, &eng).unwrap();
            model::load_synthesis_secondary(&rec, id, op, &eng).unwrap();
        }
    }
    rec.pack_f16("notice text").unwrap()
}

/// `ZJB2` members keep `ZJM1` magic (so `Checkpoint::parse` takes them through the same
/// `ModelSource` path), store every f32 network weight as `f16`, and copy integer tensors and
/// `vr_vec.c` byte for byte. Checked tensor by tensor against the `.pth` originals.
#[test]
fn zjb2_members_hold_rounded_f16_and_exact_integers() {
    use zenjpegai::model::ModelSource;
    use zenjpegai::weights::{Checkpoint, DType, f16};
    let bundle = PackedBundle::parse(pack_f16(&[1], &[OperatingPoint::Bop], true, true)).unwrap();
    assert_eq!(bundle.paths().count(), 4);
    for rel in bundle.paths().map(str::to_string).collect::<Vec<_>>() {
        let member = bundle.read(&rel).unwrap().into_owned();
        let half = Checkpoint::parse(&member).unwrap();
        let pth_file = std::fs::read(ref_root().join("models").join(&rel)).unwrap();
        let pth = Checkpoint::parse(&pth_file).unwrap();
        for name in pth.tensor_names().filter(|n| half.contains(n)) {
            let info = half.info(name).unwrap();
            let (dtype, _, want) = pth.raw(name).unwrap();
            match dtype {
                DType::F32 if name != "vr_vec.c" => {
                    assert_eq!(info.dtype, DType::F16, "{rel}/{name}");
                    let got = half.raw(name).unwrap().2;
                    assert_eq!(got.len() * 2, want.len(), "{rel}/{name}");
                    for (h, w) in got.as_chunks::<2>().0.iter().zip(want.as_chunks::<4>().0) {
                        let w = f32::from_le_bytes(*w);
                        let expect = f16::f32_to_f16(w);
                        assert_eq!(u16::from_le_bytes(*h), expect, "{rel}/{name}: {w:e}");
                    }
                    // `Checkpoint::f32` upcasts to exactly the f16-rounded values.
                    let up = half.f32(name).unwrap();
                    let orig = pth.f32(name).unwrap();
                    for (u, o) in up.data.iter().zip(&orig.data) {
                        assert_eq!(*u, f16::f16_to_f32(f16::f32_to_f16(*o)), "{rel}/{name}");
                    }
                }
                _ => {
                    assert_eq!(info.dtype, dtype, "{rel}/{name}");
                    assert!(half.raw(name).unwrap().2 == want, "{rel}/{name}");
                }
            }
        }
        assert!(pth.ints() == half.ints(), "{rel}: integer entries differ");
    }
    // A ZJB1 part and a ZJB2 part merge into one source (common f32 + synthesis f16).
    let mut mixed = PackedBundle::parse(pack(&[1], &[OperatingPoint::Bop], true, false)).unwrap();
    mixed
        .add(pack_f16(&[1], &[OperatingPoint::Bop], false, true))
        .unwrap();
    assert_eq!(mixed.paths().count(), 4);
    let stream = std::fs::read(vector_dir("img30_base_off_bpp050").join("stream.bits")).unwrap();
    Decoder::with_source(Box::new(mixed), Engine::new())
        .decode(&stream)
        .unwrap();
}
