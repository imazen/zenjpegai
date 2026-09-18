//! Whole-stream decoding: CPU entropy + latent stage, GPU synthesis, CPU (or GPU) output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zenjpegai::container::Codestream;
use zenjpegai::decoder::output;
use zenjpegai::decoder::reconstruct::{Planes, post_process_latent, reconstruct_latent};
use zenjpegai::decoder::{Headers, decode_entropy_stage, read_headers};
use zenjpegai::filters::{self, FilterContext};
use zenjpegai::header::{OperatingPoint, PictureHeader};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::{self, CommonModel, ModelSource};
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;
use zenjpegai::{Picture, RgbImage};

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::synthesis::{GpuPicture, GpuSynthesis, Timing, Workspace};

struct ModelSet {
    common: [CommonModel; 2],
    synthesis: GpuSynthesis,
}

/// A picture decoded up to the synthesis output, still on the GPU.
pub struct GpuDecoded {
    pub headers: Headers,
    pub picture: GpuPicture,
    /// Luma sigma indices (`scale_log`), which the LEF post-filter reads.
    luma_scale_log: Tensor<i32>,
    /// Operating point the picture was synthesised with.
    op: OperatingPoint,
}

impl GpuDecoded {
    /// Displayed size (`height`, `width`): coded size minus the non-displayed border.
    pub fn display_size(&self) -> (usize, usize) {
        let h = &self.headers.picture;
        (
            (h.height - h.diff_display_height as u32) as usize,
            (h.width - h.diff_display_width as u32) as usize,
        )
    }

    /// Whether [`GpuDecoded::to_rgba_texture`] can show this picture as is: 4:4:4, BT.709,
    /// 8 bit, no post-filters. Everything else has to go through [`GpuDecoder::finish`].
    pub fn presentable_on_gpu(&self) -> bool {
        let h = &self.headers.picture;
        h.bit_depth == 8
            && (h.s_ver, h.s_hor, h.c_ver, h.c_hor) == (1, 1, 1, 1)
            && h.colour_transform == zenjpegai::header::ColourTransform::Bt709
            && !self.headers.tools.any_post_filter()
    }

    /// `rgba8unorm` texture of the displayed picture, without a CPU readback.
    pub fn to_rgba_texture(&self) -> Result<wgpu::Texture> {
        if !self.presentable_on_gpu() {
            return Err(GpuError::Unsupported(
                "GPU presentation needs an 8-bit 4:4:4 BT.709 picture without post-filters",
            ));
        }
        let (h, w) = self.display_size();
        self.picture.to_rgba_texture(h, w)
    }
}

/// `rec[:, :out_h:sv, :out_w:sh]` like `zenjpegai::decoder::reconstruct::synthesize`.
fn to_planes(hdr: &PictureHeader, rec: &Tensor<f32>) -> Result<Planes> {
    let out_h = hdr.height as usize - hdr.diff_display_height as usize;
    let out_w = hdr.width as usize - hdr.diff_display_width as usize;
    let (sv, sh) = (hdr.c_ver as usize, hdr.c_hor as usize);
    let (ch, cw) = (out_h.div_ceil(sv), out_w.div_ceil(sh));
    let mut y = Tensor::<f32>::zeros(1, out_h, out_w)?;
    for row in 0..out_h {
        y.data[row * out_w..][..out_w].copy_from_slice(&rec.plane(0)[row * rec.w..][..out_w]);
    }
    let sub = |c: usize| -> Result<Tensor<f32>> {
        let mut p = Tensor::<f32>::zeros(1, ch, cw)?;
        for row in 0..ch {
            for col in 0..cw {
                p.data[row * cw + col] = rec.at(c, row * sv, col * sh);
            }
        }
        Ok(p)
    };
    Ok(Planes {
        y,
        u: sub(1)?,
        v: sub(2)?,
    })
}

/// JPEG AI decoder with the synthesis transforms on the GPU.
///
/// Models are parsed, packed and uploaded on first use and kept. All `async` methods work on
/// every target; the blocking wrappers exist on native targets only.
pub struct GpuDecoder {
    ctx: Arc<GpuContext>,
    models: Box<dyn ModelSource + Send + Sync>,
    engine: Engine,
    tables: AnsTables,
    operating_point: Option<OperatingPoint>,
    cache: Mutex<HashMap<(usize, OperatingPoint), Arc<ModelSet>>>,
    workspace: Mutex<Option<Workspace>>,
    icci_nets: zenjpegai::model::icci::NetCache,
}

impl GpuDecoder {
    pub fn new(ctx: Arc<GpuContext>, models: Box<dyn ModelSource + Send + Sync>) -> Self {
        Self::with_engine(ctx, models, Engine::new())
    }

    /// `engine` runs the CPU stages (hyper-decoder, context model, post-filters).
    pub fn with_engine(
        ctx: Arc<GpuContext>,
        models: Box<dyn ModelSource + Send + Sync>,
        engine: Engine,
    ) -> Self {
        Self {
            ctx,
            models,
            engine,
            tables: AnsTables::new(),
            operating_point: None,
            cache: Mutex::new(HashMap::new()),
            workspace: Mutex::new(None),
            icci_nets: Default::default(),
        }
    }

    /// Decode with this synthesis transform instead of the stream's first listed one.
    pub fn operating_point(mut self, op: Option<OperatingPoint>) -> Self {
        self.operating_point = op;
        self
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Drop the pooled GPU buffers (they are rebuilt on the next decode).
    pub fn release_buffers(&self) {
        if let Ok(mut ws) = self.workspace.lock() {
            *ws = None;
        }
    }

    fn model_set(&self, id: usize, op: OperatingPoint) -> Result<Arc<ModelSet>> {
        if let Some(set) = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(&(id, op)).cloned())
        {
            return Ok(set);
        }
        let set = Arc::new(ModelSet {
            common: [
                model::load_common(&*self.models, id, 0, &self.engine)?,
                model::load_common(&*self.models, id, 1, &self.engine)?,
            ],
            synthesis: GpuSynthesis::load(self.ctx.clone(), &*self.models, id, op)?,
        });
        if let Ok(mut c) = self.cache.lock() {
            c.insert((id, op), set.clone());
        }
        Ok(set)
    }

    /// Load and upload the networks of `(model_id, op)` ahead of the first decode.
    pub fn preload(&self, model_id: usize, op: OperatingPoint) -> Result<()> {
        self.model_set(model_id, op).map(|_| ())
    }

    /// Entropy + latent stage on the CPU, synthesis submitted to the GPU. Returns as soon as the
    /// work is queued; nothing is read back.
    pub fn decode_to_gpu(&self, stream: &[u8]) -> Result<GpuDecoded> {
        let cs = Codestream::parse(stream)?;
        let headers = read_headers(&cs)?;
        let hdr = &headers.picture;
        let default_op = *hdr
            .synthesis_transforms
            .first()
            .ok_or(zenjpegai::Error::InvalidData(
                "no synthesis transform listed",
            ))?;
        let op = match self.operating_point {
            None => default_op,
            Some(op) if hdr.synthesis_transforms.contains(&op) => op,
            Some(_) => {
                return Err(zenjpegai::Error::InvalidArgument(
                    "the stream does not allow the requested operating point",
                )
                .into());
            }
        };
        let set = self.model_set(hdr.model_id as usize, op)?;
        let eng = &self.engine;
        let [ent_y, ent_uv] =
            decode_entropy_stage(&self.tables, &cs, hdr, [&set.common[0], &set.common[1]])?;
        let mut ly = reconstruct_latent(eng, hdr, 0, &set.common[0], &ent_y)?;
        post_process_latent(hdr, &headers.tools, 0, &ent_y, &mut ly)?;
        let mut luv = reconstruct_latent(eng, hdr, 1, &set.common[1], &ent_uv)?;
        post_process_latent(hdr, &headers.tools, 1, &ent_uv, &mut luv)?;

        let mut ws = self
            .workspace
            .lock()
            .ok()
            .and_then(|mut w| w.take())
            .unwrap_or_default();
        let picture = set
            .synthesis
            .run_for_header(&mut ws, hdr, [&ly.y_hat, &luv.y_hat]);
        if let Ok(mut slot) = self.workspace.lock() {
            *slot = Some(ws);
        }
        Ok(GpuDecoded {
            picture: picture?,
            luma_scale_log: ent_y.scale_log,
            op,
            headers,
        })
    }

    /// Read the planes back: post-filters, colour conversion and rounding on the CPU, exactly as
    /// the CPU decoder does them (RGB, or YUV planes for streams coded that way; 8 or 10 bit).
    /// Also returns the synthesised planes before the post-filters (for parity measurements) and
    /// the timing with `gpu_ns` filled in.
    pub async fn finish(&self, mut decoded: GpuDecoded) -> Result<(Picture, Planes, Timing)> {
        let hdr = &decoded.headers.picture;
        let rec = decoded.picture.read_planes().await?;
        let synthesized = to_planes(hdr, &rec)?;
        drop(rec);
        let planes = if decoded.headers.tools.any_post_filter() {
            let fctx = FilterContext {
                eng: &self.engine,
                hdr,
                tools: &decoded.headers.tools,
                luma_scale_log: &decoded.luma_scale_log,
                models: &*self.models,
                op: decoded.op,
                icci_nets: &self.icci_nets,
                stop: &enough::Unstoppable,
            };
            filters::apply(&fctx, synthesized.clone())?
        } else {
            synthesized.clone()
        };
        Ok((
            output::finish(hdr, &planes)?,
            synthesized,
            decoded.picture.timing,
        ))
    }

    /// Codestream to whatever it holds (RGB or YUV planes), like `Decoder::decode_picture`.
    pub async fn decode_picture_async(&self, stream: &[u8]) -> Result<Picture> {
        let decoded = self.decode_to_gpu(stream)?;
        Ok(self.finish(decoded).await?.0)
    }

    /// Codestream to interleaved RGB.
    pub async fn decode_async(&self, stream: &[u8]) -> Result<RgbImage> {
        match self.decode_picture_async(stream).await? {
            Picture::Rgb(image) => Ok(image),
            Picture::Yuv(_) => Err(GpuError::Unsupported(
                "the stream decodes to YUV planes: use decode_picture_async",
            )),
        }
    }

    /// Blocking [`decode_async`](Self::decode_async).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode(&self, stream: &[u8]) -> Result<RgbImage> {
        pollster::block_on(self.decode_async(stream))
    }
}
