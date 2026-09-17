//! Full decode against reference decoder dumps.
//!
//! The latent path and the synthesis transforms are float32 networks. PyTorch's convolution
//! kernels sum in their own order, so agreement is bounded, not exact: the bounds below are a
//! few times the measured worst case on these vectors (see `PORTING.md`), tight enough that a
//! wrong layer, weight or crop fails by orders of magnitude. Do not loosen them to make a change
//! pass.
#![cfg(feature = "reference-tests")]

mod common;
use common::{load_dump, ref_root, vector_dir};
use zenjpegai::container::Codestream;
use zenjpegai::decoder::output::{RgbPlanes, quantize, to_rgb_planes};
use zenjpegai::decoder::reconstruct::{reconstruct_latent, synthesize};
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;

fn max_abs_diff(want: &[f32], got: &[f32]) -> f32 {
    assert_eq!(want.len(), got.len());
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check(name: &str) {
    let dir = vector_dir(name);
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let dump = load_dump(&dir);
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    let models = ModelDir::new(ref_root().join("models"));
    let id = hdr.model_id as usize;
    let op = hdr.synthesis_transforms[0];
    let (ym, uvm) = (
        models.load_common(id, 0).unwrap(),
        models.load_common(id, 1).unwrap(),
    );
    let syn_y = models.load_synthesis_primary(id, op).unwrap();
    let syn_uv = models.load_synthesis_secondary(id, op).unwrap();

    let ent = decode_entropy_stage(&AnsTables::new(), &cs, hdr, [&ym, &uvm]).unwrap();
    let ly = reconstruct_latent(&ym, &ent[0]).unwrap();
    let luv = reconstruct_latent(&uvm, &ent[1]).unwrap();
    for (key, got) in [
        ("y.psi", &ly.psi),
        ("y.y_hat", &ly.y_hat),
        ("uv.psi", &luv.psi),
        ("uv.y_hat", &luv.y_hat),
    ] {
        let d = max_abs_diff(&dump[key].f32(), &got.data);
        assert!(d < 5e-4, "{name} {key}: max abs diff {d:e}");
    }

    let planes = synthesize(hdr, &syn_y, &syn_uv, [&ly.y_hat, &luv.y_hat]).unwrap();
    for (key, got) in [
        ("rec.a", &planes.y),
        ("rec.b", &planes.u),
        ("rec.c", &planes.v),
    ] {
        let d = max_abs_diff(&dump[key].f32(), &got.data);
        assert!(d < 3e-3, "{name} {key}: max abs diff {d:e} (range 0..255)");
    }

    let rgb = to_rgb_planes(hdr, &planes).unwrap();
    let theirs = RgbPlanes {
        width: rgb.width,
        height: rgb.height,
        r: dump["out.a"].f32(),
        g: dump["out.b"].f32(),
        b: dump["out.c"].f32(),
    };
    let (ours, theirs) = (quantize(&rgb, 8).unwrap(), quantize(&theirs, 8).unwrap());
    let differing = ours
        .data
        .iter()
        .zip(&theirs.data)
        .filter(|(a, b)| a != b)
        .count();
    let worst = ours
        .data
        .iter()
        .zip(&theirs.data)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap();
    assert!(worst <= 1, "{name}: an 8-bit sample differs by {worst}");
    assert!(
        differing * 5000 < ours.data.len(),
        "{name}: {differing} of {} 8-bit samples differ",
        ours.data.len()
    );
    println!(
        "{name}: {differing} of {} 8-bit samples differ by 1",
        ours.data.len()
    );
}

macro_rules! vectors {
    ($($fn_name:ident => $dir:literal),* $(,)?) => { $( #[test] fn $fn_name() { check($dir); } )* };
}

vectors! {
    img30_base_off_bpp012 => "img30_base_off_bpp012",
    img30_base_off_bpp050 => "img30_base_off_bpp050",
    img30_base_off_bpp100 => "img30_base_off_bpp100",
    img30_simple_off_bpp050 => "img30_simple_off_bpp050",
}
