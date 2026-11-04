//! `GpuEncoder`: GPU analysis + CPU entropy encode.
//!
//! The conv-heavy analysis transforms run on the GPU (`GpuAnalysis`), the
//! resulting `y_hat`/`uv_hat` latents read back, and the rest of the encode —
//! hyper-latents, RDLR, headers, arithmetic coding — runs on the CPU via
//! `zenjpegai::encoder::Encoder::encode_latents`, the same seam the parity
//! tests use. The GPU latents differ from the CPU path's by a small amount
//! (same order as the CPU-vs-PyTorch analysis gap the reference encoder
//! tolerates, ~1e-4), so this produces a *valid* JPEG AI codestream, not a
//! bit-identical one — use `Encoder` directly for bit-exact HOP output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zenjpegai::encoder::{
    EncodeParams, Encoder, HYPER_DOWNSCALE, LATENT_DOWNSCALE, SourceImage, analysis_tiles_with,
    preprocess,
};
use zenjpegai::header::{LATENT_CHANNELS, OperatingPoint};
use zenjpegai::model::ModelSource;
use zenjpegai::model::common::Z_OFFSET;
use zenjpegai::model::hyper_encoder::HyperEncoder;
use zenjpegai::model::load_hyper_encoder;
use zenjpegai::nn::fast::{BTensor, Engine};
use zenjpegai::tensor::Tensor;
use zenjpegai::tools::regions::Area;

use crate::analysis::GpuAnalysis;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::layers::unpack_hwc4;
use crate::plan::{Graph, Plan, Pool, T};

/// GPU analysis + CPU `encode_latents`. `GpuEncoder` is `Sync`.
pub struct GpuEncoder {
    ctx: Arc<GpuContext>,
    cpu: Encoder,
    eng: Engine,
    models: Box<dyn ModelSource + Send + Sync>,
    hypers: Mutex<HashMap<(usize, OperatingPoint), Arc<[HyperEncoder; 2]>>>,
    analyses: Mutex<HashMap<(usize, OperatingPoint), Arc<GpuAnalysis>>>,
    pool: Arc<Mutex<Pool>>,
}

impl GpuEncoder {
    /// `cpu` is a ready `zenjpegai` encoder (entropy/hyper/header path); the
    /// GPU handles only the analysis transforms. `models` backs `GpuAnalysis`.
    pub fn new(
        ctx: Arc<GpuContext>,
        cpu: Encoder,
        models: Box<dyn ModelSource + Send + Sync>,
    ) -> Self {
        Self {
            ctx,
            cpu,
            eng: Engine::new(),
            models,
            hypers: Mutex::new(HashMap::new()),
            analyses: Mutex::new(HashMap::new()),
            pool: Arc::new(Mutex::new(Pool::default())),
        }
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// The `GpuAnalysis` for `(model_id, op)`, loading on first use.
    fn analysis(&self, model_id: usize, op: OperatingPoint) -> Result<Arc<GpuAnalysis>> {
        if let Some(a) = self.analyses.lock().unwrap().get(&(model_id, op)) {
            return Ok(a.clone());
        }
        let a: Arc<GpuAnalysis> = Arc::new(GpuAnalysis::load(
            self.ctx.clone(),
            &*self.models,
            model_id,
            op,
        )?);
        self.analyses
            .lock()
            .unwrap()
            .insert((model_id, op), a.clone());
        Ok(a)
    }

    /// The `[HyperEncoder; 2]` for `(model_id, op)`, loading on first use.
    /// `GpuEncoder` owns its own — `Encoder`'s `hyper` is private and the
    /// per-tile `z_hat` path needs the nets directly (they stay on the CPU;
    /// they are small compared to the analysis transforms).
    fn hyper(&self, model_id: usize, op: OperatingPoint) -> Result<Arc<[HyperEncoder; 2]>> {
        if let Some(h) = self.hypers.lock().unwrap().get(&(model_id, op)) {
            return Ok(h.clone());
        }
        let load = |ccs: usize| load_hyper_encoder(&*self.models, model_id, ccs, &self.eng);
        let h: Arc<[HyperEncoder; 2]> = Arc::new([
            load(0).map_err(GpuError::Codec)?,
            load(1).map_err(GpuError::Codec)?,
        ]);
        self.hypers
            .lock()
            .unwrap()
            .insert((model_id, op), h.clone());
        Ok(h)
    }

    /// Run one analysis forward (`luma` or `chroma`) on a planar `[C, h, w]`
    /// input tile, returning the latent as planar `[C, H, W]`. The multi-tile
    /// path runs it per tile — one graph + submission per tile (the reference
    /// tiles the analysis, so each tile is its own forward).
    fn analyse_one(
        &self,
        input: &Tensor<f32>,
        ccs: usize,
        model_id: usize,
        op: OperatingPoint,
    ) -> Result<Tensor<f32>> {
        let analysis = self.analysis(model_id, op)?;
        let (h, w) = (input.h, input.w);
        let mut g = Graph::new(&self.ctx);
        let x = g.input(input.c, h, w)?;
        let lat = if ccs == 0 {
            analysis.luma(&mut g, x, h, w)?
        } else {
            analysis.chroma(&mut g, x, h, w)?
        };
        g.output(lat);
        let plan = g.finish(&self.pool)?;
        let norm: Vec<f32> = input.data.iter().map(|&v| (v - 127.5) / 128.0).collect();
        plan.write(&self.ctx, x, &norm)?;
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        plan.encode(&mut enc, None);
        self.ctx.queue().submit([enc.finish()]);
        let data = self.read_latent(&plan, lat)?;
        Tensor::from_vec(lat.c, lat.h, lat.w, data).map_err(|e| GpuError::Shape(e.to_string()))
    }

    /// Copy `t`'s HWC4 buffer back and unpack to planar `[C, H, W]`.
    fn read_latent(&self, plan: &Plan, t: T) -> Result<Vec<f32>> {
        // `len4` is the vec4 count (`h*w*c4`) — 16 bytes each, not 4.
        let bytes = (t.len4() * 16) as u64;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("latent staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(plan.buffer(t)?, 0, &staging, 0, bytes);
        self.ctx.queue().submit([enc.finish()]);
        pollster::block_on(self.ctx.map_read(&staging, bytes))?;
        let data = {
            let view = staging
                .slice(0..bytes)
                .get_mapped_range()
                .map_err(|e| GpuError::Device(e.to_string()))?;
            unpack_hwc4(bytemuck::cast_slice(&view), t.c, t.h, t.w)
        };
        staging.unmap();
        Ok(data)
    }

    /// Whole-picture encode: the source (`RgbImage`, `YuvImage`, `Picture`)
    /// is preprocessed to the analysis inputs (colour transform + chroma
    /// split at half resolution), the transforms run on the GPU and the
    /// latents are coded by the CPU `encode_latents`. Mirrors
    /// `Encoder::encode`; GPU analysis is approximate: valid but not
    /// bit-identical to `Encoder::encode`.
    pub fn encode(&self, src: impl Into<SourceImage>, params: EncodeParams) -> Result<Vec<u8>> {
        let src = src.into();
        let meta = src
            .meta(params.c_ver, params.c_hor)
            .map_err(GpuError::Codec)?;
        let input = preprocess(&src, &meta).map_err(GpuError::Codec)?;
        self.encode_input(&input.luma, &input.chroma, params.model_id as usize, params)
    }

    /// Whole-picture encode from pre-made analysis inputs: `luma`
    /// `[1, h, w]`, `chroma` `[12, h/2, w/2]`, both in `[0, 255]` — the
    /// `AnalysisInput` `preprocess` produces. Lower-level than `encode`.
    /// Errors on a picture the encoder would split into more than one
    /// analysis tile — the reference analysis is *tiled* there, and a
    /// whole-picture GPU pass would diverge, not just approximate.
    pub fn encode_input(
        &self,
        luma: &Tensor<f32>,
        chroma: &Tensor<f32>,
        model_id: usize,
        params: EncodeParams,
    ) -> Result<Vec<u8>> {
        for ccs in 0..2 {
            let (w, h) = if ccs == 0 {
                (luma.w, luma.h)
            } else {
                (chroma.w * 2, chroma.h * 2)
            };
            if Encoder::analysis_tile_grid(w, h, ccs, params)
                .map_err(GpuError::Codec)?
                .len()
                != 1
            {
                return self.encode_tiled(luma, chroma, model_id, params);
            }
        }
        let analysis = self.analysis(model_id, params.op)?;
        let (h, w) = (luma.h, luma.w);
        let (ch, cw) = (chroma.h, chroma.w);
        // Luma and chroma share one graph, one plan and one submission — the
        // two analyses are independent dispatches, so this halves the
        // plan-build + submit + sync the split version paid twice.
        let mut g = Graph::new(&self.ctx);
        let xl = g.input(luma.c, h, w)?;
        let yl = analysis.luma(&mut g, xl, h, w)?;
        g.output(yl);
        let xc = g.input(chroma.c, ch, cw)?;
        let uvl = analysis.chroma(&mut g, xc, ch, cw)?;
        g.output(uvl);
        let plan = g.finish(&self.pool)?;
        let norm = |t: &Tensor<f32>| {
            t.data
                .iter()
                .map(|&v| (v - 127.5) / 128.0)
                .collect::<Vec<f32>>()
        };
        plan.write(&self.ctx, xl, &norm(luma))?;
        plan.write(&self.ctx, xc, &norm(chroma))?;
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        plan.encode(&mut enc, None);
        self.ctx.queue().submit([enc.finish()]);
        let yd = self.read_latent(&plan, yl)?;
        let ud = self.read_latent(&plan, uvl)?;
        let y =
            Tensor::from_vec(yl.c, yl.h, yl.w, yd).map_err(|e| GpuError::Shape(e.to_string()))?;
        let uv = Tensor::from_vec(uvl.c, uvl.h, uvl.w, ud)
            .map_err(|e| GpuError::Shape(e.to_string()))?;
        self.cpu
            .encode_latents([&y, &uv], None, w, h, params, None)
            .map(|(stream, _)| stream)
            .map_err(|e| GpuError::Codec(e.error().clone()))
    }

    /// Per-tile encode: the reference analysis is *tiled* (`analysis_tiles`),
    /// so each tile runs `GpuAnalysis` on its `window`ed plane, the
    /// `HyperEncoder`s produce per-tile `z_hat`s on the CPU, and the kept
    /// cores merge via `assign`. Produces the same `(y, z_hat)` merge the CPU
    /// `analyse_component` does, then `encode_latents` with the real z_hats.
    fn encode_tiled(
        &self,
        luma: &Tensor<f32>,
        chroma: &Tensor<f32>,
        model_id: usize,
        params: EncodeParams,
    ) -> Result<Vec<u8>> {
        let hyper = self.hyper(model_id, params.op)?;
        let mut parts: [Option<(Tensor<f32>, Tensor<i8>)>; 2] = [None, None];
        for (ccs, plane) in [luma, chroma].iter().enumerate() {
            let (ph, pw) = (plane.h, plane.w);
            let d = LATENT_DOWNSCALE[ccs];
            let (lh, lw) = (ph.div_ceil(d), pw.div_ceil(d));
            let (hz, wz) = (
                ph.div_ceil(HYPER_DOWNSCALE[ccs]),
                pw.div_ceil(HYPER_DOWNSCALE[ccs]),
            );
            let chs = LATENT_CHANNELS[ccs];
            let grid =
                analysis_tiles_with(ccs, ph, pw, lh, lw, hz, wz, None).map_err(GpuError::Codec)?;
            let mut yy =
                Tensor::<f32>::zeros(chs, lh, lw).map_err(|e| GpuError::Shape(e.to_string()))?;
            let mut zz =
                Tensor::<i8>::zeros(chs, hz, wz).map_err(|e| GpuError::Shape(e.to_string()))?;
            for t in &grid {
                let tile = plane
                    .window(t.image.x, t.image.y, t.image.width, t.image.height)
                    .map_err(|e| GpuError::Shape(e.to_string()))?;
                let yt = self.analyse_one(&tile, ccs, model_id, params.op)?;
                let bt = BTensor::from_planar_par(&self.eng, &yt, self.eng.tier.block())
                    .map_err(|e| GpuError::Shape(e.to_string()))?;
                let zt = hyper[ccs]
                    .forward_clip(
                        &self.eng,
                        &bt,
                        t.image.height,
                        t.image.width,
                        d,
                        params.experimental.clipping,
                        &enough::Unstoppable,
                    )
                    .map_err(GpuError::Codec)?
                    .to_planar()
                    .map_err(|e| GpuError::Shape(e.to_string()))?;
                let max = (Z_OFFSET - 1) as f32;
                let zq = Tensor::from_vec(
                    zt.c,
                    zt.h,
                    zt.w,
                    zt.data
                        .iter()
                        .map(|&v| v.clamp(-(Z_OFFSET as f32), max).round_ties_even() as i8)
                        .collect(),
                )
                .map_err(|e| GpuError::Shape(e.to_string()))?;
                assign(&mut yy, t.latent_core, &yt, t.latent_core_offset);
                assign(&mut zz, t.hyper_core, &zq, t.hyper_core_offset);
            }
            parts[ccs] = Some((yy, zz));
        }
        let (y, zy) = parts[0].take().unwrap();
        let (uv, zuv) = parts[1].take().unwrap();
        self.cpu
            .encode_latents([&y, &uv], Some([&zy, &zuv]), luma.w, luma.h, params, None)
            .map(|(stream, _)| stream)
            .map_err(|e| GpuError::Codec(e.error().clone()))
    }
}

/// `assign` from `encoder/mod.rs` — copy the kept core of a tile's latent
/// into the merged latent (the parts the overlap discards are dropped).
fn assign<T: Copy + Default>(
    dst: &mut Tensor<T>,
    core: Area,
    src: &Tensor<T>,
    (ox, oy): (usize, usize),
) {
    let h = core
        .height
        .min(src.h.saturating_sub(oy))
        .min(dst.h.saturating_sub(core.y));
    let w = core
        .width
        .min(src.w.saturating_sub(ox))
        .min(dst.w.saturating_sub(core.x));
    let (dw, sw) = (dst.w, src.w);
    for c in 0..dst.c.min(src.c) {
        for y in 0..h {
            let s: &[T] = &src.plane(c)[(oy + y) * sw + ox..][..w];
            let d = (core.y + y) * dw + core.x;
            dst.plane_mut(c)[d..d + w].copy_from_slice(s);
        }
    }
}
