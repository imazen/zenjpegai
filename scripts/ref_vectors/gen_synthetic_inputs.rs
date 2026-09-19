//! Writes the synthetic adversarial sources of [`synthetic_patterns`] as 8-bit RGB PNGs
//! into the reference inputs tree, for the f16 model-weight study (`tests/f16_weights.rs`
//! encodes the same pixels in memory). Imazen tooling only: zenpng.
//!
//! ```text
//! cargo run --release --features cli --example gen_synthetic_inputs -- [output dir]
//! ```
//!
//! Output dir defaults to `/mnt/v/output/zenjpegai/reference/inputs/synthetic/`.
//! The PNGs are generated artefacts (like the reference vectors): they stay out of git.

#[path = "synthetic_patterns.rs"]
mod synthetic_patterns;

use synthetic_patterns::SyntheticImage;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/mnt/v/output/zenjpegai/reference/inputs/synthetic".to_string());
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).expect("create output dir");
    for SyntheticImage {
        name,
        width,
        height,
        rgb,
    } in synthetic_patterns::all()
    {
        let pixels: Vec<rgb::Rgb<u8>> = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| rgb::Rgb {
                r: p[0],
                g: p[1],
                b: p[2],
            })
            .collect();
        let png = zenpng::encode_rgb8(
            imgref::ImgRef::new(&pixels, width, height),
            None,
            &zenpng::EncodeConfig::default(),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .expect("png encode");
        let path = dir.join(format!("{name}_{width}x{height}_8bit_sRGB.png"));
        std::fs::write(&path, &png).expect("write png");
        eprintln!("{}: {} bytes", path.display(), png.len());
    }
}
