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

use zenjpegai::encoder::{EncodeParams, Encoder, SourceImage, preprocess};
use zenjpegai::header::OperatingPoint;
use zenjpegai::model::ModelSource;
use zenjpegai::tensor::Tensor;

use crate::analysis::GpuAnalysis;
use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::layers::unpack_hwc4;
use crate::plan::{Graph, Plan, Pool, T};

/// GPU analysis + CPU `encode_latents`. `GpuEncoder` is `Sync`.
pub struct GpuEncoder {
    ctx: Arc<GpuContext>,
    cpu: Encoder,
    models: Box<dyn ModelSource + Send + Sync>,
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
            models,
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

    /// Copy `t`'s HWC4 buffer back and unpack to planar `[C, H, W]`.
    fn read_latent(&self, plan: &Plan, t: T) -> Result<Vec<f32>> {
        let bytes = (t.len4() * 4) as u64;
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
        // The GPU analysis is whole-picture: if the encoder would split either
        // component into more than one analysis tile, the full CPU encode —
        // which tiles the analysis natively — is the correct path, not a
        // diverging whole-picture GPU pass.
        for ccs in 0..2 {
            if Encoder::analysis_tile_grid(src.width(), src.height(), ccs, params)
                .map_err(GpuError::Codec)?
                .len()
                != 1
            {
                return self
                    .cpu
                    .encode(src, params)
                    .map_err(|e| GpuError::Codec(e.error().clone()));
            }
        }
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
                return Err(GpuError::Shape(format!(
                    "tiled GPU analysis not implemented (ccs {ccs} is multi-tile)"
                )));
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
}
