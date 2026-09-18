//! Command line front end: `zenjpegai encode in.png out.bits`, `zenjpegai decode in.bits
//! out.png`, `zenjpegai info in.bits`.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{self, ModelDir};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::packed::{PackedBundle, Recorder};

/// The upstream licence travels inside every bundle: the weights are upstream's.
const UPSTREAM_NOTICE: &str = concat!(
    "Model weights: JPEG AI reference software (https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software), ",
    "repacked without modification by zenjpegai.\n\n",
    include_str!("../../upstream-notices/LICENSE")
);
use zenjpegai::{Decoder, EncodeParams, Encoder, Picture};

const USAGE: &str = "\
zenjpegai - JPEG AI (ISO/IEC 6048) codec

USAGE:
    zenjpegai encode <in.png> <out.bits> [--model <0..3>] [--beta-disp <n>] [--op <sop|bop|hop>]
    zenjpegai decode <in.bits> <out.png | out.yuv> [options]
    zenjpegai info <in.bits>
    zenjpegai pack-models --models <dir> --out <file.zjb> [--model <0..3>]... [--op <sop|bop|hop>]...
                          [--only <common|synthesis>]
    zenjpegai preload <in.bits>      load the networks the stream needs, decode nothing

OPTIONS:
    --models <path>    directory of upstream checkpoints (the reference software's models/), or a
                       packed bundle written by `pack-models`; default: $ZENJPEGAI_MODELS
    --op <sop|bop|hop> encode: the operating point to code for (default: bop). decode: the
                       synthesis transform to use (default: the stream's first listed one)
    --beta-disp <n>    encode: quantiser displacement, -1069..702 (default 0; lower = lower rate)
    --max-channels <y,uv>  progressive decode: read only the first latent channels
    --single-thread    do not use the thread pool
    --scalar           no SIMD (for debugging; every tier produces identical pixels)
    --repeat <n>       decode n times and print per-run timing (models stay loaded)
    --time             print timing
    --pool-mb <n>      cap the recycled-buffer pool at n MiB (default 1024; 0 = no recycling)
    --discard          decode only, write no file (profiling; <out.png> is ignored)

pack-models writes a ZJB1 bundle holding only the tensors the decoder reads for the given models
(default: all four) and operating points (default: all three), byte for byte: pixels decoded
from a bundle are identical to pixels decoded from the .pth files. `--only common` keeps the
per-model entropy / latent networks, `--only synthesis` the per-(model, operating point) synthesis
transforms: a client that has the common part of a model fetches only the other for a new
operating point (`PackedBundle::add` merges them).
";

struct Args {
    positional: Vec<String>,
    models: Option<PathBuf>,
    op: Option<OperatingPoint>,
    max_channels: (Option<u16>, Option<u16>),
    ops: Vec<OperatingPoint>,
    model_ids: Vec<usize>,
    out: Option<PathBuf>,
    only: Option<String>,
    beta_disp: i32,
    single_thread: bool,
    scalar: bool,
    repeat: usize,
    time: bool,
    discard: bool,
    pool_mb: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        positional: Vec::new(),
        models: std::env::var_os("ZENJPEGAI_MODELS").map(PathBuf::from),
        op: None,
        max_channels: (None, None),
        ops: Vec::new(),
        model_ids: Vec::new(),
        out: None,
        only: None,
        beta_disp: 0,
        single_thread: false,
        scalar: false,
        repeat: 1,
        time: false,
        discard: false,
        pool_mb: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--models" => a.models = Some(PathBuf::from(value("--models")?)),
            "--op" => {
                let op = match value("--op")?.as_str() {
                    "sop" => OperatingPoint::Sop,
                    "bop" => OperatingPoint::Bop,
                    "hop" => OperatingPoint::Hop,
                    other => return Err(format!("unknown operating point `{other}`")),
                };
                a.op = Some(op);
                a.ops.push(op);
            }
            "--model" => {
                let id: usize = value("--model")?
                    .parse()
                    .map_err(|e| format!("--model: {e}"))?;
                if id >= zenjpegai::model::MODEL_BETAS.len() {
                    return Err(format!("--model: {id} is not one of 0..=3"));
                }
                a.model_ids.push(id);
            }
            "--only" => {
                let v = value("--only")?;
                if v != "common" && v != "synthesis" {
                    return Err(format!("--only: `{v}` is not `common` or `synthesis`"));
                }
                a.only = Some(v);
            }
            "--out" => a.out = Some(PathBuf::from(value("--out")?)),
            "--beta-disp" => {
                a.beta_disp = value("--beta-disp")?
                    .parse()
                    .map_err(|e| format!("--beta-disp: {e}"))?;
            }
            "--max-channels" => {
                let v = value("--max-channels")?;
                let (y, uv) = v.split_once(',').ok_or("--max-channels wants <y,uv>")?;
                let parse = |s: &str| s.parse::<u16>().map_err(|e| format!("--max-channels: {e}"));
                a.max_channels = (Some(parse(y)?), Some(parse(uv)?));
            }
            "--single-thread" => a.single_thread = true,
            "--scalar" => a.scalar = true,
            "--repeat" => {
                a.repeat = value("--repeat")?
                    .parse()
                    .map_err(|e| format!("--repeat: {e}"))?;
                a.time = true;
            }
            "--time" => a.time = true,
            "--discard" => a.discard = true,
            "--pool-mb" => {
                a.pool_mb = Some(
                    value("--pool-mb")?
                        .parse()
                        .map_err(|e| format!("--pool-mb: {e}"))?,
                )
            }
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with("--") => return Err(format!("unknown option `{s}`")),
            _ => a.positional.push(arg),
        }
    }
    Ok(a)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let pos: Vec<&str> = args.positional.iter().map(String::as_str).collect();
    match pos.as_slice() {
        ["info", input] => {
            let stream = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
            // Headers need no checkpoints.
            let headers = Decoder::new("")
                .read_headers(&stream)
                .map_err(|e| format!("{e:?}"))?;
            println!("{headers:#?}");
            Ok(())
        }
        ["pack-models"] => {
            let models = args
                .models
                .ok_or("no checkpoint directory: pass --models or set ZENJPEGAI_MODELS")?;
            let out = args.out.ok_or("pack-models needs --out <file>")?;
            let ids = if args.model_ids.is_empty() {
                (0..zenjpegai::model::MODEL_BETAS.len()).collect()
            } else {
                args.model_ids
            };
            let ops = if args.ops.is_empty() {
                vec![
                    OperatingPoint::Sop,
                    OperatingPoint::Bop,
                    OperatingPoint::Hop,
                ]
            } else {
                args.ops
            };
            // Loading is where the decoder reads tensors; the tier does not change which.
            let eng = Engine::with(Tier::Scalar, false);
            let rec = Recorder::new(ModelDir::new(&models));
            let (common, synthesis) = match args.only.as_deref() {
                Some("common") => (true, false),
                Some(_) => (false, true),
                None => (true, true),
            };
            for &id in &ids {
                for ccs in 0..2 {
                    if common {
                        model::load_common(&rec, id, ccs, &eng).map_err(|e| format!("{e:?}"))?;
                    }
                }
                for &op in ops.iter().filter(|_| synthesis) {
                    model::load_synthesis_primary(&rec, id, op, &eng)
                        .map_err(|e| format!("{e:?}"))?;
                    model::load_synthesis_secondary(&rec, id, op, &eng)
                        .map_err(|e| format!("{e:?}"))?;
                }
            }
            let bundle = rec.pack(UPSTREAM_NOTICE).map_err(|e| format!("{e:?}"))?;
            let packed = PackedBundle::parse(bundle.clone()).map_err(|e| format!("{e:?}"))?;
            let mut source_bytes = 0u64;
            for rel in packed.paths() {
                source_bytes += std::fs::metadata(models.join(rel)).map_or(0, |m| m.len());
            }
            std::fs::write(&out, &bundle).map_err(|e| format!("{out:?}: {e}"))?;
            eprintln!(
                "{}: {} files, {} bytes (from {} bytes of .pth)",
                out.display(),
                packed.paths().count(),
                bundle.len(),
                source_bytes
            );
            Ok(())
        }
        ["preload", input] => {
            let models = args
                .models
                .ok_or("no checkpoint directory: pass --models or set ZENJPEGAI_MODELS")?;
            let stream = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
            let t = Instant::now();
            Decoder::new(models)
                .operating_point(args.op)
                .preload(&stream)
                .map_err(|e| format!("{e:?}"))?;
            eprintln!("models loaded: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            Ok(())
        }
        ["encode", input, output] => {
            let models = args
                .models
                .ok_or("no checkpoint directory: pass --models or set ZENJPEGAI_MODELS")?;
            if models.is_file() {
                return Err(
                    "the encoder needs the checkpoint directory, not a packed bundle \
                            (bundles hold decoder tensors only)"
                        .into(),
                );
            }
            let tier = if args.scalar {
                Tier::Scalar
            } else {
                Tier::detect()
            };
            let engine = Engine::with(tier, !args.single_thread && cfg!(feature = "parallel"));
            let png = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
            let image = zenjpegai::encoder::read_png_rgb8(&png).map_err(|e| format!("{e:?}"))?;
            let params = EncodeParams {
                model_id: args.model_ids.first().copied().unwrap_or(1) as u8,
                beta_displacement_log: [args.beta_disp; 2],
                op: args.op.unwrap_or(OperatingPoint::Bop),
            };
            let encoder = Encoder::with_engine(models, engine);
            let mut stream = Vec::new();
            for run in 0..args.repeat.max(1) {
                let t = Instant::now();
                stream = encoder
                    .encode(&image, params)
                    .map_err(|e| format!("{e:?}"))?;
                if args.time {
                    eprintln!(
                        "encode {run}: {:.1} ms ({}x{}, {:?}{})",
                        t.elapsed().as_secs_f64() * 1e3,
                        image.width,
                        image.height,
                        engine.tier,
                        if run == 0 {
                            ", includes model load"
                        } else {
                            ""
                        },
                    );
                }
            }
            std::fs::write(output, &stream).map_err(|e| format!("{output}: {e}"))?;
            eprintln!(
                "{output}: {} bytes ({:.4} bpp)",
                stream.len(),
                stream.len() as f64 * 8.0 / (image.width * image.height) as f64
            );
            Ok(())
        }
        ["decode", input, output] => {
            let models = args
                .models
                .ok_or("no checkpoint directory: pass --models or set ZENJPEGAI_MODELS")?;
            let tier = if args.scalar {
                Tier::Scalar
            } else {
                Tier::detect()
            };
            let engine = Engine::with(tier, !args.single_thread && cfg!(feature = "parallel"));
            let stream = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
            if let Some(mb) = args.pool_mb {
                zenjpegai::nn::fast::set_pool_limit(mb << 20);
            }
            let decoder = if models.is_file() {
                let bytes = std::fs::read(&models).map_err(|e| format!("{models:?}: {e}"))?;
                let bundle = PackedBundle::parse(bytes).map_err(|e| format!("{e:?}"))?;
                Decoder::with_source(Box::new(bundle), engine)
            } else {
                Decoder::with_engine(models, engine)
            }
            .operating_point(args.op)
            .max_channels(args.max_channels.0, args.max_channels.1);
            let mut image = None;
            for run in 0..args.repeat.max(1) {
                let t = Instant::now();
                let img = decoder
                    .decode_picture(&stream)
                    .map_err(|e| format!("{e:?}"))?;
                if args.time {
                    let (width, height) = match &img {
                        Picture::Rgb(i) => (i.width, i.height),
                        Picture::Yuv(i) => (i.width, i.height),
                    };
                    eprintln!(
                        "decode {run}: {:.1} ms ({}x{}, {:?}{})",
                        t.elapsed().as_secs_f64() * 1e3,
                        width,
                        height,
                        engine.tier,
                        if run == 0 {
                            ", includes model load"
                        } else {
                            ""
                        },
                    );
                }
                image = Some(img);
            }
            let image = image.ok_or("nothing decoded")?;
            // The PNG encoder needs memory of its own: hand the decoder's recycled buffers back.
            decoder.release_buffers();
            if args.discard {
                return Ok(());
            }
            let t = Instant::now();
            let bytes = match image {
                // 8 bit: 8-bit PNG. 10 bit: 16-bit PNG, samples in the top bits with the low
                // bits set, like the reference's `write_file`.
                Picture::Rgb(image) if image.bit_depth == 8 => {
                    let pixels: Vec<rgb::Rgb<u8>> = image
                        .data
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| rgb::Rgb {
                            r: p[0] as u8,
                            g: p[1] as u8,
                            b: p[2] as u8,
                        })
                        .collect();
                    zenpng::encode_rgb8(
                        imgref::ImgRef::new(&pixels, image.width, image.height),
                        None,
                        &zenpng::EncodeConfig::default()
                            .with_compression(zenpng::Compression::Fast),
                        &enough::Unstoppable,
                        &enough::Unstoppable,
                    )
                    .map_err(|e| format!("png: {e:?}"))?
                }
                Picture::Rgb(image) => {
                    let shift = 16 - image.bit_depth as u32;
                    let up = |v: u16| (v << shift) | ((1u16 << shift) - 1);
                    let pixels: Vec<rgb::Rgb<u16>> = image
                        .data
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|p| rgb::Rgb {
                            r: up(p[0]),
                            g: up(p[1]),
                            b: up(p[2]),
                        })
                        .collect();
                    zenpng::encode_rgb16(
                        imgref::ImgRef::new(&pixels, image.width, image.height),
                        None,
                        &zenpng::EncodeConfig::default()
                            .with_compression(zenpng::Compression::Fast),
                        &enough::Unstoppable,
                        &enough::Unstoppable,
                    )
                    .map_err(|e| format!("png: {e:?}"))?
                }
                // A YUV source stays YUV: raw planar Y, U, V (one byte per sample at 8 bit, two
                // little-endian bytes at 10 bit), chroma in the source's subsampling.
                Picture::Yuv(image) => {
                    if !output.to_ascii_lowercase().ends_with(".yuv") {
                        return Err(format!(
                            "{input} decodes to YUV {}x{} (chroma {}x{}, {} bit): give an output name ending in .yuv",
                            image.width,
                            image.height,
                            image.chroma_width,
                            image.chroma_height,
                            image.bit_depth
                        ));
                    }
                    let mut raw = Vec::new();
                    for plane in [&image.y, &image.u, &image.v] {
                        for &v in plane {
                            if image.bit_depth == 8 {
                                raw.push(v as u8);
                            } else {
                                raw.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    }
                    raw
                }
            };
            std::fs::write(output, bytes).map_err(|e| format!("{output}: {e}"))?;
            if args.time {
                eprintln!("output write: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
            }
            Ok(())
        }
        _ => Err(String::new()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) if msg.is_empty() => {
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
