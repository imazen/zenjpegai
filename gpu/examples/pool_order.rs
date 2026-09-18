//! Does a workspace that has synthesised a large picture keep a small picture slow?
//!
//!     cargo run --release -p zenjpegai-gpu --example pool_order -- [--adapter <substring>]
//!
//! Replays the `gpu_bench` sweep order (ascending sizes with CPU synthesis interleaved, then
//! the 560x888 picture again) in one `Workspace`, then a fresh workspace, then back-to-back
//! reruns. On the RTX 2080 the post-sweep small picture measures slow only while the card is
//! still at its idle clock (the interleaved CPU phases let it down-clock); a few back-to-back
//! runs ramp it back to small-picture speed. The retained-memory column shows the bounded
//! `Pool`/`Workspace` retention.
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use zenjpegai::container::Codestream;
use zenjpegai::decoder::read_headers;
use zenjpegai::decoder::reconstruct::synthesize;
use zenjpegai::header::{OperatingPoint, SynthesisTiling};
use zenjpegai::model::ModelDir;
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;
use zenjpegai_gpu::{ContextOptions, GpuContext, GpuSynthesis, Workspace};

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

fn run_once(
    syn: &GpuSynthesis,
    ws: &mut Workspace,
    hdr: &zenjpegai::header::PictureHeader,
    y: &[Tensor<f32>; 2],
) -> (f64, Option<f64>) {
    let t = Instant::now();
    let mut pic = syn.run_for_header(ws, hdr, [&y[0], &y[1]]).expect("run");
    pollster::block_on(pic.read_planes()).expect("readback");
    let wall = t.elapsed().as_secs_f64() * 1e3;
    let dev = pic.timing.gpu_ns.map(|ns| ns as f64 / 1e6);
    (wall, dev)
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut models = PathBuf::from(format!("{home}/work/zen/jpeg-ai-reference-software/models"));
    let mut vectors = PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors");
    let mut opts = ContextOptions::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--models" => models = it.next().unwrap().into(),
            "--vectors" => vectors = it.next().unwrap().into(),
            "--adapter" => opts.adapter_name = Some(it.next().unwrap()),
            "--allow-software" => opts.allow_software = true,
            _ => panic!("unknown option {flag}"),
        }
    }
    let models = ModelDir::new(&models);
    let template = {
        let s = std::fs::read(vectors.join("img30_base_off_bpp050/stream.bits")).expect("stream");
        read_headers(&Codestream::parse(&s).unwrap())
            .unwrap()
            .picture
    };
    let ctx = Arc::new(GpuContext::new(&opts).expect("adapter"));
    let i = ctx.adapter_info();
    eprintln!("adapter: {} [{:?}]", i.name, i.backend);
    let syn = GpuSynthesis::load(
        ctx.clone(),
        &models,
        template.model_id as usize,
        OperatingPoint::Bop,
    )
    .expect("load");

    let mut small = template.clone();
    small.height = 888;
    small.width = 560;
    small.diff_display_height = 0;
    small.diff_display_width = 0;
    small.regions = None;
    for c in &mut small.components {
        c.synthesis_tiling = None;
    }
    let ys = latents(888, 560);

    // Ascending size sweep like gpu_bench (CPU synthesis interleaved between GPU runs — that
    // pairing is what the benchmark does and what reproduces the stall), then the small
    // picture again.
    let cpu_mt = Engine::new();
    let cpu_1t = Engine::with(cpu_mt.tier, false);
    let luma = models
        .load_synthesis_primary(template.model_id as usize, OperatingPoint::Bop, &cpu_mt)
        .unwrap();
    let chroma = models
        .load_synthesis_secondary(template.model_id as usize, OperatingPoint::Bop, &cpu_mt)
        .unwrap();
    let sizes: [usize; 8] = [256, 512, 1024, 1536, 2048, 3072, 4096, 0];
    let mut ws = Workspace::new();
    for &s in &sizes {
        let (hdr, y) = if s == 0 {
            (small.clone(), ys.clone())
        } else {
            let mut h = small.clone();
            h.height = s as u32;
            h.width = s as u32;
            if s > 1024 {
                for c in &mut h.components {
                    c.synthesis_tiling = Some(SynthesisTiling {
                        tile_size: 1024,
                        overlap: 64,
                    });
                }
            }
            (h, latents(s, s))
        };
        let (w, d) = run_once(&syn, &mut ws, &hdr, &y);
        synthesize(&cpu_mt, &hdr, &luma, &chroma, [&y[0], &y[1]]).expect("cpu mt");
        let (w2, d2) = run_once(&syn, &mut ws, &hdr, &y);
        synthesize(&cpu_1t, &hdr, &luma, &chroma, [&y[0], &y[1]]).expect("cpu 1t");
        eprintln!(
            "{:>5}x{:<5} wall {w:7.2}/{w2:7.2} ms  device {d:?}/{d2:?}  ws {} MiB",
            hdr.width,
            hdr.height,
            ws.bytes() / (1 << 20)
        );
    }
    // Fresh workspace, same synthesis: what the small picture should cost.
    let mut ws2 = Workspace::new();
    let (w, d) = run_once(&syn, &mut ws2, &small, &ys);
    let (w2, d2) = run_once(&syn, &mut ws2, &small, &ys);
    eprintln!("small fresh workspace  wall {w:7.2}/{w2:7.2} ms  device {d:?}/{d2:?}");

    // Recovery: hammer the GPU back to back; if the stall is a clock ramp the later runs
    // return to small-picture speed without touching the workspace.
    for i in 0..8 {
        let (w, d) = run_once(&syn, &mut ws, &small, &ys);
        eprintln!("recovery #{i} wall {w:7.2} ms  device {d:?}");
    }
}
