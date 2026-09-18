//! Downscale demo source PNGs to ~1 MP (Imazen tooling only: zenpng decode/encode, zenresize
//! Lanczos resample). usage: `prep-demo-images <out_dir> <in.png>...`
use std::env;
use std::fs;
use std::path::PathBuf;

use enough::Unstoppable;
use imgref::ImgRef;
use rgb::FromSlice;
use zenpixels_convert::PixelBufferConvertTypedExt;
use zenpng::{decode, encode_rgba8, Compression, EncodeConfig, PngDecodeConfig};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

const TARGET_PIXELS: f64 = 1_000_000.0;

fn main() {
    let mut args = env::args().skip(1);
    let out_dir = PathBuf::from(args.next().expect("usage: prep-demo-images <out_dir> <in.png>..."));
    fs::create_dir_all(&out_dir).unwrap();
    for path in args {
        let path = PathBuf::from(path);
        let bytes = fs::read(&path).unwrap();
        let output = decode(&bytes, &PngDecodeConfig::default(), &Unstoppable).unwrap();
        let rgba = output.pixels.to_rgba8();
        let (w, h) = (output.info.width as usize, output.info.height as usize);
        let scale = (TARGET_PIXELS / (w as f64 * h as f64)).sqrt().min(1.0);
        let (ow, oh) = (((w as f64 * scale).round() as usize).max(1), ((h as f64 * scale).round() as usize).max(1));

        let config = ResizeConfig::builder(w as u32, h as u32, ow as u32, oh as u32)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .build();
        let resized = Resizer::new(&config).resize(&rgba.copy_to_contiguous_bytes());

        let img = ImgRef::new(resized.as_rgba(), ow, oh);
        let encoded = encode_rgba8(img, None, &EncodeConfig::default().with_compression(Compression::High), &Unstoppable, &Unstoppable).unwrap();

        let stem = path.file_stem().unwrap().to_string_lossy();
        let out = out_dir.join(format!("{stem}_{ow}x{oh}.png"));
        fs::write(&out, &encoded).unwrap();
        println!("{} {w}x{h} -> {ow}x{oh} ({} bytes)", out.display(), encoded.len());
    }
}
