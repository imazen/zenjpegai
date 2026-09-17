//! Command line front end: `zenjpegai decode in.bits out.png`, `zenjpegai info in.bits`.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use zenjpegai::Decoder;
use zenjpegai::header::OperatingPoint;
use zenjpegai::nn::fast::{Engine, Tier};

const USAGE: &str = "\
zenjpegai - JPEG AI (ISO/IEC 6048) codec

USAGE:
    zenjpegai decode <in.bits> <out.png> [options]
    zenjpegai info <in.bits>

OPTIONS:
    --models <dir>     directory of upstream checkpoints (the reference software's models/);
                       default: $ZENJPEGAI_MODELS
    --op <sop|bop|hop> synthesis transform (default: the stream's first listed one)
    --single-thread    do not use the thread pool
    --scalar           no SIMD (for debugging; every tier produces identical pixels)
    --repeat <n>       decode n times and print per-run timing (models stay loaded)
    --time             print timing
";

struct Args {
    positional: Vec<String>,
    models: Option<PathBuf>,
    op: Option<OperatingPoint>,
    single_thread: bool,
    scalar: bool,
    repeat: usize,
    time: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        positional: Vec::new(),
        models: std::env::var_os("ZENJPEGAI_MODELS").map(PathBuf::from),
        op: None,
        single_thread: false,
        scalar: false,
        repeat: 1,
        time: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--models" => a.models = Some(PathBuf::from(value("--models")?)),
            "--op" => {
                a.op = Some(match value("--op")?.as_str() {
                    "sop" => OperatingPoint::Sop,
                    "bop" => OperatingPoint::Bop,
                    "hop" => OperatingPoint::Hop,
                    other => return Err(format!("unknown operating point `{other}`")),
                })
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
            let decoder = Decoder::with_engine(models, engine).operating_point(args.op);
            let mut image = None;
            for run in 0..args.repeat.max(1) {
                let t = Instant::now();
                let img = decoder.decode(&stream).map_err(|e| format!("{e:?}"))?;
                if args.time {
                    eprintln!(
                        "decode {run}: {:.1} ms ({}x{}, {:?}{})",
                        t.elapsed().as_secs_f64() * 1e3,
                        img.width,
                        img.height,
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
            let t = Instant::now();
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
            let png = zenpng::encode_rgb8(
                imgref::ImgRef::new(&pixels, image.width, image.height),
                None,
                &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fast),
                &enough::Unstoppable,
                &enough::Unstoppable,
            )
            .map_err(|e| format!("png: {e:?}"))?;
            std::fs::write(output, png).map_err(|e| format!("{output}: {e}"))?;
            if args.time {
                eprintln!("png write: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
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
