//! Scratch: compare the Rust entropy stage with a reference dump (decoder- or encoder-side).
//! usage: dbg_entropy <vector_dir> [enc]
use zenjpegai::container::Codestream;
use zenjpegai::decoder::{decode_entropy_stage, read_headers};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::ModelDir;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let enc = std::env::args().nth(2).as_deref() == Some("enc");
    let (mf, tf) = if enc {
        ("enc_manifest.txt", "enc_tensors.bin")
    } else {
        ("manifest.txt", "tensors.bin")
    };
    let stream = std::fs::read(dir.join("stream.bits")).unwrap();
    let manifest = std::fs::read_to_string(dir.join(mf)).unwrap();
    let blob = std::fs::read(dir.join(tf)).unwrap();
    let get = |name: &str| -> Vec<i32> {
        let line = manifest.lines().find(|l| l.starts_with(name)).unwrap();
        let f: Vec<&str> = line.split_whitespace().collect();
        let ndim: usize = f[2].parse().unwrap();
        let off: usize = f[3 + ndim].parse().unwrap();
        let n: usize = f[4 + ndim].parse().unwrap();
        let b = &blob[off..off + n];
        match f[1] {
            "i32" => b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect(),
            "i16" => b
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as i32)
                .collect(),
            "i8" => b.iter().map(|&v| v as i8 as i32).collect(),
            t => panic!("dtype {t}"),
        }
    };
    let cs = Codestream::parse(&stream).unwrap();
    let headers = read_headers(&cs).unwrap();
    let hdr = &headers.picture;
    println!(
        "model_id {} beta_log {:?} regions {:?}",
        hdr.model_id, hdr.beta_displacement_log, hdr.regions
    );
    let models = ModelDir::new(std::env::var("ZENJPEGAI_REF").unwrap() + "/models");
    let y = models.load_common(hdr.model_id as usize, 0).unwrap();
    let uv = models.load_common(hdr.model_id as usize, 1).unwrap();
    let out = decode_entropy_stage(&AnsTables::new(), &cs, hdr, [&y, &uv]).unwrap();
    for (comp, e) in [("y", &out[0]), ("uv", &out[1])] {
        let report = |what: &str, want: Vec<i32>, got: Vec<i32>| {
            let bad = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            let first = want.iter().zip(&got).position(|(a, b)| a != b);
            println!(
                "{comp}.{what}: {bad} of {} differ (first at {first:?})",
                want.len()
            );
        };
        report(
            "z_hat",
            get(&format!("{comp}.z_hat ")),
            e.z_hat.data.iter().map(|&v| v as i32).collect(),
        );
        report(
            "scale_log",
            get(&format!("{comp}.scale_log ")),
            e.scale_log.data.clone(),
        );
        report(
            "residual_quant",
            get(&format!("{comp}.residual_quant ")),
            e.residual_q.data.iter().map(|&v| v as i32).collect(),
        );
        let coded = e.mask.data.iter().filter(|m| **m).count();
        println!("{comp}: {coded} of {} positions coded", e.mask.data.len());
    }
}
