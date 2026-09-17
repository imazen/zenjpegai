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
}
