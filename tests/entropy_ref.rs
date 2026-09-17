//! Entropy stage against reference dumps: `z_hat`, sigma indices, quantised and dequantised
//! residual must all be identical (integers exactly, floats bit for bit).
//!
//! Streams without regions are checked against what the reference *decoder* recovered. Streams
//! with regions are checked against what the reference *encoder* wrote, because the reference
//! decoder mis-decodes them: it hands a non-contiguous view of the skip mask to its C++ entropy
//! coder, which ignores strides (the encoder side `.copy()`s the view and is fine). Its own
//! encoder and decoder MD5s disagree on these streams. See PORTING.md.
#![cfg(feature = "reference-tests")]

mod common;
use common::{load_dump, load_encoder_dump, ref_root, vector_dir};
use zenjpegai::container::Codestream;
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;

#[derive(Clone, Copy, PartialEq)]
enum Oracle {
    Decoder,
    Encoder,
}

fn check(name: &str) {
    check_against(name, Oracle::Decoder);
}

fn check_regions(name: &str) {
    check_against(name, Oracle::Encoder);
}

fn check_against(name: &str, oracle: Oracle) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let dump = if oracle == Oracle::Decoder {
        load_dump(&dir)
    } else {
        load_encoder_dump(&dir)
    };

    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(ref_root().join("models"));
    let eng = Engine::new();
    let y_model = models.load_common(hdr.model_id as usize, 0, &eng).unwrap();
    let uv_model = models.load_common(hdr.model_id as usize, 1, &eng).unwrap();
    let tables = AnsTables::new();
    let out = decode_entropy_stage(&tables, &cs, hdr, [&y_model, &uv_model]).unwrap();

    for (comp, e) in ["y", "uv"].into_iter().zip(&out) {
        let z = &dump[&format!("{comp}.z_hat")];
        assert_eq!(
            z.shape[1..],
            [e.z_hat.c, e.z_hat.h, e.z_hat.w],
            "{name} {comp}.z_hat shape"
        );
        assert_eq!(z.i8(), e.z_hat.data, "{name} {comp}.z_hat");
        if oracle == Oracle::Encoder {
            assert!(
                headers.picture.regions.is_some(),
                "{name}: encoder oracle is for region streams"
            );
        }

        let s = &dump[&format!("{comp}.skip_scale_log")];
        assert_eq!(
            s.shape[1..],
            [e.skip_scale_log.c, e.skip_scale_log.h, e.skip_scale_log.w]
        );
        assert_eq!(
            s.i32(),
            e.skip_scale_log.data,
            "{name} {comp}.skip_scale_log"
        );
        assert_eq!(
            dump[&format!("{comp}.scale_log")].i32(),
            e.scale_log.data,
            "{name} {comp}.scale_log"
        );

        let rq: Vec<i32> = e.residual_q.data.iter().map(|&v| v as i32).collect();
        assert_eq!(
            dump[&format!("{comp}.residual_quant")].i32(),
            rq,
            "{name} {comp}.residual_quant"
        );

        let want = dump[&format!("{comp}.residual")].f32();
        let bad = want
            .iter()
            .zip(&e.residual.data)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "{name} {comp}.residual: {bad} of {} floats differ",
            want.len()
        );
    }
}

macro_rules! vectors {
    ($check:ident: $($fn_name:ident => $dir:literal),* $(,)?) => {
        $( #[test] fn $fn_name() { $check($dir); } )*
    };
}

vectors! { check:
    img30_base_off_bpp012 => "img30_base_off_bpp012",
    img30_base_off_bpp025 => "img30_base_off_bpp025",
    img30_base_off_bpp050 => "img30_base_off_bpp050",
    img30_base_off_bpp075 => "img30_base_off_bpp075",
    img30_base_off_bpp100 => "img30_base_off_bpp100",
    img30_simple_off_bpp050 => "img30_simple_off_bpp050",
    img30_high_off_bpp050 => "img30_high_off_bpp050",
    img01_base_off_bpp050 => "img01_base_off_bpp050",
    img01_base_off_threads8_bpp050 => "img01_base_off_threads8_bpp050",
}

vectors! { check_regions:
    img01_base_off_depregions_m1 => "img01_base_off_depregions_m1",
    img01_base_off_indregions_m1 => "img01_base_off_indregions_m1",
    img01_base_off_indregions_threads8_m2 => "img01_base_off_indregions_threads8_m2",
}
