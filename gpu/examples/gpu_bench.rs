//! GPU vs CPU synthesis and whole-decode timing.
//!
//!     cargo run --release -p zenjpegai-gpu --example gpu_bench -- \
//!         --models ~/work/zen/jpeg-ai-reference-software/models \
//!         --vectors /mnt/v/output/zenjpegai/reference/vectors --out benchmarks/gpu_decode_<date>.tsv
//!
//! Options: `--adapter <name substring>`, `--allow-software`, `--rounds N` (default 7),
//! `--max-size N` (largest square of the size sweep, default 4096), `--ops sop,bop,hop`,
//! `--profile` (instead of the sweep: per-dispatch device time at 560x888 and 1024x1024, one
//! compute pass per dispatch; the `single_pass_ms` column is the same picture timed as one pass,
//! so the profiling overhead is visible).
//!
//! What is measured (all wall-clock numbers are medians over interleaved rounds: every round
//! runs GPU, CPU threaded and CPU 1-thread once, in that order, so drift hits all three alike):
//!
//! * size sweep, synthesis only, random latents: `gpu_wall_ms` = latent upload + submit +
//!   wait + readback of the three f32 planes; `gpu_device_ms` = sum of the tiles' compute-pass
//!   durations from GPU timestamp queries (empty when the adapter has none);
//!   `gpu_submit_ms` = host time spent encoding / submitting (not GPU work);
//!   `gpu_first_ms` = first run at that size in a warm context (plan building; pipelines may
//!   already be compiled by earlier sizes); `cpu_mt_ms` / `cpu_1t_ms` = the CPU engine's
//!   `synthesize` on the same latents.
//! * `cold` rows: new context + checkpoint parse + weight upload + first picture (pipeline
//!   compilation included), per operating point.
//! * `decode` rows: whole reference streams, codestream bytes to 8-bit RGB.
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zenjpegai::Decoder;
use zenjpegai::container::Codestream;
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::synthesize;
use zenjpegai::header::{OperatingPoint, PictureHeader, SynthesisTiling};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::{ContextOptions, GpuContext, GpuDecoder, GpuOut, GpuSynthesis, Workspace};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn latents(h: usize, w: usize) -> [Tensor<f32>; 2] {
    let (lh, lw) = (h.div_ceil(16), w.div_ceil(16));
    let mut s = 0x2545_f491u32;
    let mut fill = |c: usize| {
        let data = (0..c * lh * lw)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 4.0
            })
            .collect();
        Tensor::from_vec(c, lh, lw, data).unwrap()
    };
    [fill(160), fill(96)]
}

struct Args {
    models: PathBuf,
    vectors: PathBuf,
    out: Option<PathBuf>,
    opts: ContextOptions,
    rounds: usize,
    max_size: usize,
    ops: Vec<OperatingPoint>,
    profile: bool,
}

fn args() -> Args {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut a = Args {
        models: PathBuf::from(format!("{home}/work/zen/jpeg-ai-reference-software/models")),
        vectors: PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"),
        out: None,
        opts: ContextOptions::default(),
        rounds: 7,
        max_size: 4096,
        ops: vec![
            OperatingPoint::Sop,
            OperatingPoint::Bop,
            OperatingPoint::Hop,
        ],
        profile: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--models" => a.models = val().into(),
            "--vectors" => a.vectors = val().into(),
            "--out" => a.out = Some(val().into()),
            "--adapter" => a.opts.adapter_name = Some(val()),
            "--allow-software" => a.opts.allow_software = true,
            "--profile" => a.profile = true,
            "--rounds" => a.rounds = val().parse().expect("--rounds"),
            "--max-size" => a.max_size = val().parse().expect("--max-size"),
            "--ops" => {
                a.ops = val()
                    .split(',')
                    .map(|o| match o {
                        "sop" => OperatingPoint::Sop,
                        "bop" => OperatingPoint::Bop,
                        "hop" => OperatingPoint::Hop,
                        _ => panic!("unknown operating point {o}"),
                    })
                    .collect();
            }
            _ => panic!("unknown option {flag}"),
        }
    }
    a
}

fn header_for(template: &PictureHeader, h: usize, w: usize) -> PictureHeader {
    let mut hdr = template.clone();
    hdr.height = h as u32;
    hdr.width = w as u32;
    hdr.diff_display_height = 0;
    hdr.diff_display_width = 0;
    hdr.regions = None;
    let tiling = (h > 1024 || w > 1024).then_some(SynthesisTiling {
        tile_size: 1024,
        overlap: 64,
    });
    for c in &mut hdr.components {
        c.synthesis_tiling = tiling;
    }
    hdr
}

/// `--profile`: per-dispatch device time at a couple of sizes, one compute pass per dispatch.
fn profile(a: &Args, models: &ModelDir, template: &PictureHeader, model_id: usize) {
    let ctx = Arc::new(GpuContext::new(&a.opts).expect("adapter"));
    let i = ctx.adapter_info();
    let adapter = format!(
        "{} [{:?} {:?}; {} {}]",
        i.name, i.backend, i.device_type, i.driver, i.driver_info
    );
    assert!(ctx.has_timestamps(), "adapter has no timestamp queries");
    let mut tsv = String::from(
        "kind\tadapter\top\twidth\theight\tindex\tkernel\tgrid\tns\tpct\tdispatch_ms_total\tsingle_pass_ms\tparams\n",
    );
    for &op in &a.ops {
        let syn = GpuSynthesis::load(ctx.clone(), models, model_id, op).expect("load");
        let mut sizes = vec![(888usize, 560usize)];
        if a.max_size >= 1024 {
            sizes.push((1024, 1024));
        }
        for (h, w) in sizes {
            let hdr = header_for(template, h, w);
            let y = latents(h, w);
            // Warm: build the plan and compile pipelines outside the measurement.
            let mut ws = Workspace::new();
            let mut plain = vec![];
            for _ in 0..=a.rounds {
                let mut pic = syn
                    .run_for_header(&mut ws, &hdr, [&y[0], &y[1]])
                    .expect("run");
                pollster::block_on(pic.read_planes()).expect("readback");
                if let Some(ns) = pic.timing.gpu_ns {
                    plain.push(ns as f64 / 1e6);
                }
            }
            let single_pass = median(&mut plain);
            ws.set_profile(true);
            let mut rows: Vec<Vec<u64>> = vec![];
            let mut labels: Vec<String> = vec![];
            for _ in 0..a.rounds {
                let mut pic = syn
                    .run_for_header(&mut ws, &hdr, [&y[0], &y[1]])
                    .expect("run");
                pollster::block_on(pic.read_planes()).expect("readback");
                let per = pollster::block_on(pic.dispatch_ns())
                    .expect("timestamps")
                    .expect("profile mode");
                if labels.is_empty() {
                    labels = per.iter().map(|(l, _)| l.clone()).collect();
                    rows = vec![vec![]; per.len()];
                }
                for (r, (_, ns)) in rows.iter_mut().zip(&per) {
                    r.push(*ns);
                }
            }
            let med: Vec<f64> = rows
                .iter_mut()
                .map(|r| {
                    r.sort_unstable();
                    r[r.len() / 2] as f64
                })
                .collect();
            let total: f64 = med.iter().sum();
            for (idx, (label, &ns)) in labels.iter().zip(&med).enumerate() {
                // `key [gx x gy x gz] p1 p2 ...` — params feed the roofline analysis.
                let (kernel, rest) = label.split_once(" [").unwrap_or((label, ""));
                let (grid, params) = rest.split_once("] ").unwrap_or((rest, ""));
                writeln!(
                    tsv,
                    "profile\t{adapter}\t{op:?}\t{w}\t{h}\t{idx}\t{kernel}\t{grid}\t{ns:.0}\t{:.2}\t{:.3}\t{single_pass:.3}\t{params}",
                    100.0 * ns / total,
                    total / 1e6,
                )
                .unwrap();
            }
            eprintln!(
                "{op:?} {w}x{h}: {} dispatches, {:.2} ms summed (single pass {single_pass:.2} ms)",
                labels.len(),
                total / 1e6
            );
        }
    }
    print!("{tsv}");
    if let Some(out) = &a.out {
        std::fs::write(out, &tsv).expect("write tsv");
        eprintln!("wrote {}", out.display());
    }
}

fn main() {
    let a = args();
    let models = ModelDir::new(&a.models);
    let template = {
        let s = std::fs::read(a.vectors.join("img30_base_off_bpp050/stream.bits"))
            .expect("template stream");
        read_headers(&Codestream::parse(&s).unwrap())
            .unwrap()
            .picture
    };
    let model_id = template.model_id as usize;
    if a.profile {
        profile(&a, &models, &template, model_id);
        return;
    }
    let cpu_mt = Engine::new();
    let cpu_1t = Engine::with(cpu_mt.tier, false);
    let mut tsv = String::from(
        "kind\tadapter\top\twidth\theight\ttiles\tdispatches\tgpu_wall_ms\tgpu_device_ms\tgpu_submit_ms\tgpu_first_ms\tcpu_mt_ms\tcpu_1t_ms\tgpu_mem_mb\tlargest_map_mb\tnote\n",
    );
    let mut adapter = String::new();
    // Write after every row: a sweep that dies late (a big operating point can exhaust GPU
    // memory) must not take the rows it already produced with it.
    let flush = |tsv: &str| {
        if let Some(out) = &a.out {
            std::fs::write(out, tsv).expect("write tsv");
        }
    };

    for &op in &a.ops {
        // Cold: context, weights, first picture (560 wide x 888 high like the reference test image).
        let t = Instant::now();
        let ctx = Arc::new(GpuContext::new(&a.opts).expect("adapter"));
        let ctx_ms = ms(t);
        let i = ctx.adapter_info();
        adapter = format!(
            "{} [{:?} {:?}; {} {}]",
            i.name, i.backend, i.device_type, i.driver, i.driver_info
        );
        let t = Instant::now();
        let syn = GpuSynthesis::load(ctx.clone(), &models, model_id, op).expect("load");
        let load_ms = ms(t);
        let mut ws = Workspace::new();
        let y = latents(888, 560);
        let t = Instant::now();
        let mut pic = syn
            .run(&mut ws, [&y[0], &y[1]], (888, 560), None, None)
            .expect("run");
        pollster::block_on(pic.read_planes()).expect("readback");
        let first_ms = ms(t);
        writeln!(
            tsv,
            "cold\t{adapter}\t{op:?}\t560\t888\t1\t{}\t{:.2}\t\t\t{first_ms:.2}\t\t\t\t\tcontext {ctx_ms:.1} ms + checkpoint parse/upload {load_ms:.1} ms + first picture {first_ms:.1} ms, of which plan build (pipeline compilation + bind groups) {:.1} ms ({} pipelines)",
            pic.timing.dispatches,
            ctx_ms + load_ms + first_ms,
            pic.timing.plan_host_ns as f64 / 1e6,
            ctx.pipeline_count(),
        )
        .unwrap();
        flush(&tsv);

        let luma = models
            .load_synthesis_primary(model_id, op, &cpu_mt)
            .unwrap();
        let chroma = models
            .load_synthesis_secondary(model_id, op, &cpu_mt)
            .unwrap();
        let mut sizes: Vec<(usize, usize)> =
            [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
                .into_iter()
                .filter(|&s| s <= a.max_size)
                .map(|s| (s, s))
                .collect();
        sizes.push((888, 560));
        for (h, w) in sizes {
            let hdr = header_for(&template, h, w);
            let y = latents(h, w);
            let t = Instant::now();
            let first = syn.run_for_header(&mut ws, &hdr, [&y[0], &y[1]]);
            let mut first = match first {
                Ok(p) => p,
                Err(e) => {
                    writeln!(
                        tsv,
                        "synthesis\t{adapter}\t{op:?}\t{w}\t{h}\t\t\t\t\t\t\t\t\t\t\tGPU: {e}"
                    )
                    .unwrap();
                    flush(&tsv);
                    continue;
                }
            };
            pollster::block_on(first.read_planes()).expect("readback");
            let first_ms = ms(t);
            // Fewer rounds where one CPU 1-thread run takes seconds.
            let rounds = if h * w >= 2048 * 2048 {
                a.rounds.min(3)
            } else {
                a.rounds
            };
            let (mut wall, mut dev, mut sub, mut mt, mut st) =
                (vec![], vec![], vec![], vec![], vec![]);
            let (mut tiles, mut dispatches, mut largest) = (0, 0, 0);
            for _ in 0..rounds {
                let t = Instant::now();
                let mut pic = syn
                    .run_for_header(&mut ws, &hdr, [&y[0], &y[1]])
                    .expect("run");
                pollster::block_on(pic.read_planes()).expect("readback");
                wall.push(ms(t));
                assert_eq!(pic.timing.plans_built, 0, "warm run rebuilt a plan");
                if let Some(ns) = pic.timing.gpu_ns {
                    dev.push(ns as f64 / 1e6);
                }
                sub.push((pic.timing.upload_host_ns + pic.timing.submit_host_ns) as f64 / 1e6);
                (tiles, dispatches) = (pic.timing.tiles, pic.timing.dispatches);
                largest = pic.timing.largest_binding_bytes;
                let t = Instant::now();
                synthesize(&cpu_mt, &hdr, &luma, &chroma, [&y[0], &y[1]]).expect("cpu");
                mt.push(ms(t));
                let t = Instant::now();
                synthesize(&cpu_1t, &hdr, &luma, &chroma, [&y[0], &y[1]]).expect("cpu");
                st.push(ms(t));
            }
            let dev = if dev.is_empty() {
                String::new()
            } else {
                format!("{:.2}", median(&mut dev))
            };
            writeln!(
                tsv,
                "synthesis\t{adapter}\t{op:?}\t{w}\t{h}\t{tiles}\t{dispatches}\t{:.2}\t{dev}\t{:.2}\t{first_ms:.2}\t{:.2}\t{:.2}\t{:.1}\t{:.1}\t",
                median(&mut wall),
                median(&mut sub),
                median(&mut mt),
                median(&mut st),
                ws.bytes() as f64 / (1 << 20) as f64,
                largest as f64 / (1 << 20) as f64,
            )
            .unwrap();
            flush(&tsv);
            eprintln!("{op:?} {w}x{h} done");
        }
    }

    // Whole streams.
    let ctx = Arc::new(GpuContext::new(&a.opts).expect("adapter"));
    let gpu = GpuDecoder::new(ctx, Box::new(ModelDir::new(&a.models)));
    let cpu_dec_mt = Decoder::new(&a.models);
    let cpu_dec_1t = Decoder::with_engine(&a.models, cpu_1t);
    for name in [
        "img30_simple_off_bpp050",
        "img30_base_off_bpp050",
        "img30_high_off_bpp050",
        "img01_base_off_bpp050",
    ] {
        let stream = std::fs::read(a.vectors.join(name).join("stream.bits")).expect("stream");
        let hdr = read_headers(&Codestream::parse(&stream).unwrap())
            .unwrap()
            .picture;
        let op = hdr.synthesis_transforms[0];
        if !a.ops.contains(&op) {
            continue;
        }
        let t = Instant::now();
        gpu.decode(&stream).expect("gpu decode");
        let first_ms = ms(t);
        cpu_dec_mt.decode(&stream).expect("cpu decode");
        cpu_dec_1t.decode(&stream).expect("cpu decode");
        let (mut wall, mut dev, mut sub, mut mt, mut st) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut tiles, mut dispatches) = (0, 0);
        for _ in 0..a.rounds {
            let t = Instant::now();
            // The quantized tail: output-format conversion on the GPU, packed u16 readback.
            let d = gpu
                .decode_to_gpu_with(&stream, GpuOut::Quantized)
                .expect("gpu decode");
            let (_, timing) = pollster::block_on(gpu.finish_picture(d)).expect("gpu finish");
            wall.push(ms(t));
            if let Some(ns) = timing.gpu_ns {
                dev.push(ns as f64 / 1e6);
            }
            sub.push((timing.upload_host_ns + timing.submit_host_ns) as f64 / 1e6);
            (tiles, dispatches) = (timing.tiles, timing.dispatches);
            let t = Instant::now();
            cpu_dec_mt.decode(&stream).expect("cpu decode");
            mt.push(ms(t));
            let t = Instant::now();
            cpu_dec_1t.decode(&stream).expect("cpu decode");
            st.push(ms(t));
        }
        let dev = if dev.is_empty() {
            String::new()
        } else {
            format!("{:.2}", median(&mut dev))
        };
        writeln!(
            tsv,
            "decode\t{adapter}\t{op:?}\t{}\t{}\t{tiles}\t{dispatches}\t{:.2}\t{dev}\t{:.2}\t{first_ms:.2}\t{:.2}\t{:.2}\t\t\t{name}; first = model load + first decode",
            hdr.width,
            hdr.height,
            median(&mut wall),
            median(&mut sub),
            median(&mut mt),
            median(&mut st),
        )
        .unwrap();
        flush(&tsv);
        eprintln!("{name} done");
    }

    print!("{tsv}");
    flush(&tsv);
    if let Some(out) = &a.out {
        eprintln!("wrote {}", out.display());
    }
}
