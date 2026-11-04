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

    /// Run `f` (the analysis forward) on a normalized input map, returning
    /// the latent as planar `[C, H, W]`.
    fn run_analysis(
        &self,
        input: &Tensor<f32>,
        f: impl FnOnce(&mut Graph<'_>, T) -> Result<T>,
    ) -> Result<(usize, usize, usize, Vec<f32>)> {
        let (h, w) = (input.h, input.w);
        let mut g = Graph::new(&self.ctx);
        let x = g.input(input.c, h, w)?;
        let lat = f(&mut g, x)?;
        g.output(lat);
        let plan = g.finish(&self.pool)?;
        // Normalize `(v - 127.5) / 128` on the host during upload.
        let norm: Vec<f32> = input.data.iter().map(|&v| (v - 127.5) / 128.0).collect();
        plan.write(&self.ctx, x, &norm)?;
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        plan.encode(&mut enc, None);
        self.ctx.queue().submit([enc.finish()]);
        let data = self.read_latent(&plan, lat)?;
        Ok((lat.c, lat.h, lat.w, data))
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
        let meta = src
            .meta(params.c_ver, params.c_hor)
            .map_err(|e| GpuError::Codec(e))?;
        let input = preprocess(&src, &meta).map_err(|e| GpuError::Codec(e))?;
        self.encode_input(&input.luma, &input.chroma, params.model_id as usize, params)
    }

    /// Whole-picture encode from pre-made analysis inputs: `luma`
    /// `[1, h, w]`, `chroma` `[12, h/2, w/2]`, both in `[0, 255]` — the
    /// `AnalysisInput` `preprocess` produces. Lower-level than `encode`.
    pub fn encode_input(
        &self,
        luma: &Tensor<f32>,
        chroma: &Tensor<f32>,
        model_id: usize,
        params: EncodeParams,
    ) -> Result<Vec<u8>> {
        let analysis = self.analysis(model_id, params.op)?;
        let (h, w) = (luma.h, luma.w);
        let (yc, yh, yw, yd) = self.run_analysis(luma, |g, x| analysis.luma(g, x, h, w))?;
        let (ch, cw) = (chroma.h, chroma.w);
        let (_c, uh, uw, ud) = self.run_analysis(chroma, |g, x| analysis.chroma(g, x, ch, cw))?;
        let y = Tensor::from_vec(yc, yh, yw, yd).map_err(|e| GpuError::Shape(e.to_string()))?;
        let uv = Tensor::from_vec(_c, uh, uw, ud).map_err(|e| GpuError::Shape(e.to_string()))?;
        self.cpu
            .encode_latents([&y, &uv], None, w, h, params, None)
            .map(|(stream, _)| stream)
            .map_err(|e| GpuError::Codec(e.error().clone()))
    }
}
