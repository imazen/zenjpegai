//! Command line front end: `zenjpegai decode in.bits out.png`, `zenjpegai info in.bits`.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use zenjpegai::Decoder;
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

const USAGE: &str = "\
zenjpegai - JPEG AI (ISO/IEC 6048) codec

USAGE:
    zenjpegai decode <in.bits> <out.png> [options]
    zenjpegai info <in.bits>
    zenjpegai pack-models --models <dir> --out <file.zjb> [--model <0..3>]... [--op <sop|bop|hop>]...
                          [--only <common|synthesis>]

OPTIONS:
    --models <path>    directory of upstream checkpoints (the reference software's models/), or a
                       packed bundle written by `pack-models`; default: $ZENJPEGAI_MODELS
    --op <sop|bop|hop> synthesis transform (default: the stream's first listed one)
    --single-thread    do not use the thread pool
    --scalar           no SIMD (for debugging; every tier produces identical pixels)
    --repeat <n>       decode n times and print per-run timing (models stay loaded)
    --time             print timing

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
    ops: Vec<OperatingPoint>,
    model_ids: Vec<usize>,
    out: Option<PathBuf>,
    only: Option<String>,
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
        ops: Vec::new(),
        model_ids: Vec::new(),
        out: None,
        only: None,
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
            let decoder = if models.is_file() {
                let bytes = std::fs::read(&models).map_err(|e| format!("{models:?}: {e}"))?;
                let bundle = PackedBundle::parse(bytes).map_err(|e| format!("{e:?}"))?;
                Decoder::with_source(Box::new(bundle), engine)
            } else {
                Decoder::with_engine(models, engine)
            }
            .operating_point(args.op);
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
