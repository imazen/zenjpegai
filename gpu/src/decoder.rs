//! Whole-stream decoding: CPU entropy + latent stage, GPU synthesis, CPU (or GPU) output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zenjpegai::container::Codestream;
use zenjpegai::decoder::output;
use zenjpegai::decoder::reconstruct::{Planes, post_process_latent, reconstruct_latent_with};
use zenjpegai::decoder::{Headers, decode_entropy_stage_progressive, read_headers};
use zenjpegai::filters::{self, FilterContext};
use zenjpegai::header::{ColourTransform, OperatingPoint, PictureHeader};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::{self, CommonModel, ModelSource};
use zenjpegai::nn::fast::Engine;
use zenjpegai::tensor::Tensor;
use zenjpegai::{Picture, RgbImage, YuvImage};

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::synthesis::{GpuPicture, GpuSynthesis, OutSpec, RunTail, Timing, Workspace, now, since};

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
        presentable(&self.headers)
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

/// The `presentable_on_gpu` check, before `GpuDecoded` exists (the decode tail depends on it).
fn presentable(headers: &Headers) -> bool {
    let h = &headers.picture;
    h.bit_depth == 8
        && (h.s_ver, h.s_hor, h.c_ver, h.c_hor) == (1, 1, 1, 1)
        && h.colour_transform == zenjpegai::header::ColourTransform::Bt709
        && !headers.tools.any_post_filter()
}

/// What `decode_to_gpu_with` records into the decode's single submission, beyond the tile
/// passes themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuOut {
    /// Copy the planar picture into the readback staging buffer (`read_planes` / `finish` then
    /// only map, no extra submit).
    Planes,
    /// For presentable pictures, also run the `rgba8unorm` conversion (`to_rgba_texture` then
    /// returns the finished texture without a second submit). Non-presentable pictures behave
    /// like `Planes`.
    Rgba,
    /// `Rgba` plus a copy of the RGBA texture into the staging buffer, so `read_rgba` is one
    /// map and no additional submission.
    RgbaReadback,
    /// Convert to the stream's output format on the GPU (the same arithmetic
    /// `zenjpegai::decoder::output::finish` runs on the CPU) and stage the packed `u16`
    /// samples — half the bytes of the `f32` planes. Streams the GPU conversion does not cover
    /// (post-filters, coded chroma ≠ source, custom transform) fall back to `Planes`.
    /// `GpuDecoder::finish_picture` reads the result; `finish` keeps working too (it re-copies
    /// the planes into the staging buffer itself).
    Quantized,
}

/// The `rgba8unorm` texture descriptor shared by the decode tail and `to_rgba_texture`.
pub(crate) fn rgba_texture(ctx: &GpuContext, h: usize, w: usize) -> wgpu::Texture {
    ctx.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("zenjpegai rgba"),
        size: wgpu::Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// The GPU output conversion [`GpuOut::Quantized`] asks for, or `None` when it cannot produce
/// what `zenjpegai::decoder::output::finish` would: post-filters run on the CPU planes, coded
/// chroma in a different subsampling than the source needs bicubic upsampling first, and the
/// custom colour transform is unsupported everywhere.
fn quantized_spec(headers: &Headers) -> Option<OutSpec> {
    let h = &headers.picture;
    if headers.tools.any_post_filter() || (h.bit_depth != 8 && h.bit_depth != 10) {
        return None;
    }
    let out_w = h.width - h.diff_display_width as u32;
    let out_h = h.height - h.diff_display_height as u32;
    let (pic_w, pic_h) = (h.width, h.height);
    match h.colour_transform {
        // to_rgb_planes on 4:4:4 planes (an RGB source is never subsampled).
        ColourTransform::Bt709 if (h.s_ver, h.s_hor, h.c_ver, h.c_hor) == (1, 1, 1, 1) => {
            let n_y = out_w * out_h;
            Some(OutSpec {
                mode: 1,
                bit_depth: h.bit_depth,
                out_w,
                out_h,
                pic_w,
                pic_h,
                sv: 1,
                sh: 1,
                cw: 0,
                ch: 0,
                n_y,
                n_c: 0,
                n: 3 * n_y,
            })
        }
        // Planar YUV at the source's subsampling = the coded subsampling (c == s).
        ColourTransform::None if h.c_ver == h.s_ver && h.c_hor == h.s_hor => {
            let (sv, sh) = (h.c_ver as u32, h.c_hor as u32);
            let (cw, ch) = (out_w.div_ceil(sh), out_h.div_ceil(sv));
            let (n_y, n_c) = (out_w * out_h, cw * ch);
            Some(OutSpec {
                mode: 0,
                bit_depth: h.bit_depth,
                out_w,
                out_h,
                pic_w,
                pic_h,
                sv,
                sh,
                cw,
                ch,
                n_y,
                n_c,
                n: n_y + 2 * n_c,
            })
        }
        _ => None,
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
    max_channels: [Option<u16>; 2],
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
            max_channels: [None, None],
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

    /// Progressive decode: read only the first `luma` / `chroma` latent channels (the
    /// reference's `num_decode_chs`), like `zenjpegai::Decoder::max_channels`. `None` decodes
    /// all of them.
    pub fn max_channels(mut self, luma: Option<u16>, chroma: Option<u16>) -> Self {
        self.max_channels = [luma, chroma];
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

    /// The `(model, op)` networks — parsed and uploaded on first use, then cached. `t` is
    /// charged with the load time on a miss (`common_host_ns` for the CPU-side common
    /// networks, `weights_host_ns` for the synthesis checkpoint parse + GPU upload).
    fn model_set(&self, id: usize, op: OperatingPoint, t: &mut Timing) -> Result<Arc<ModelSet>> {
        if let Some(set) = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(&(id, op)).cloned())
        {
            return Ok(set);
        }
        let t0 = now();
        let common = [
            model::load_common(&*self.models, id, 0, &self.engine)?,
            model::load_common(&*self.models, id, 1, &self.engine)?,
        ];
        t.common_host_ns += since(t0);
        let t0 = now();
        let synthesis = GpuSynthesis::load(self.ctx.clone(), &*self.models, id, op)?;
        t.weights_host_ns += since(t0);
        let set = Arc::new(ModelSet { common, synthesis });
        if let Ok(mut c) = self.cache.lock() {
            c.insert((id, op), set.clone());
        }
        Ok(set)
    }

    /// Load and upload the networks of `(model_id, op)` ahead of the first decode.
    pub fn preload(&self, model_id: usize, op: OperatingPoint) -> Result<()> {
        self.model_set(model_id, op, &mut Timing::default())
            .map(|_| ())
    }

    /// Entropy + latent stage on the CPU, synthesis submitted to the GPU. Returns as soon as the
    /// work is queued; nothing is read back. The planes readback is staged in the same
    /// submission ([`GpuOut::Planes`]); callers wanting the RGBA texture should use
    /// [`decode_to_gpu_with`](Self::decode_to_gpu_with) instead.
    pub fn decode_to_gpu(&self, stream: &[u8]) -> Result<GpuDecoded> {
        self.decode_to_gpu_stop(stream, GpuOut::Planes, &enough::Unstoppable)
    }

    /// [`decode_to_gpu`](Self::decode_to_gpu) with a choice of what the single submission also
    /// produces ([`GpuOut`]). Presentable pictures decoded with `Rgba`/`RgbaReadback` need no
    /// later submit for `to_rgba_texture`/`read_rgba`.
    pub fn decode_to_gpu_with(&self, stream: &[u8], out: GpuOut) -> Result<GpuDecoded> {
        self.decode_to_gpu_stop(stream, out, &enough::Unstoppable)
    }

    /// [`decode_to_gpu_with`](Self::decode_to_gpu_with) reading only the first `max_channels`
    /// latent channels per component for this call (progressive decode, the reference's
    /// `num_decode_chs`; the remaining channels decode as zero residual). Unlike
    /// [`GpuDecoder::max_channels`], which caps every decode of this decoder, this applies to
    /// one call; a component's `None` keeps the decoder's own setting.
    pub fn decode_to_gpu_progressive(
        &self,
        stream: &[u8],
        out: GpuOut,
        max_channels: [Option<u16>; 2],
    ) -> Result<GpuDecoded> {
        let mc = [
            max_channels[0].or(self.max_channels[0]),
            max_channels[1].or(self.max_channels[1]),
        ];
        self.decode_to_gpu_inner(stream, out, mc, &enough::Unstoppable)
    }

    /// [`decode_to_gpu_with`](Self::decode_to_gpu_with) with cooperative cancellation: the
    /// `stop` token the CPU decoder uses, checked in the entropy and latent stages (they accept
    /// it directly) and before the synthesis submission — the GPU cannot be recalled once
    /// queued. A stop request surfaces as `GpuError::Codec(Error::Cancelled)`.
    pub fn decode_to_gpu_stop(
        &self,
        stream: &[u8],
        out: GpuOut,
        stop: &dyn enough::Stop,
    ) -> Result<GpuDecoded> {
        self.decode_to_gpu_inner(stream, out, self.max_channels, stop)
    }

    fn decode_to_gpu_inner(
        &self,
        stream: &[u8],
        out: GpuOut,
        max_channels: [Option<u16>; 2],
        stop: &dyn enough::Stop,
    ) -> Result<GpuDecoded> {
        stop.check().map_err(zenjpegai::Error::from)?;
        let t = now();
        let cs = Codestream::parse(stream)?;
        let headers = read_headers(&cs)?;
        let headers_ns = since(t);
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
        let mut phases = Timing::default();
        let set = self.model_set(hdr.model_id as usize, op, &mut phases)?;
        let eng = &self.engine;
        let t = now();
        let [ent_y, ent_uv] = decode_entropy_stage_progressive(
            &self.tables,
            &cs,
            hdr,
            [&set.common[0], &set.common[1]],
            max_channels,
            stop,
        )?;
        phases.entropy_host_ns = since(t);
        let t = now();
        let mut ly = reconstruct_latent_with(eng, hdr, 0, &set.common[0], &ent_y, stop)?;
        post_process_latent(hdr, &headers.tools, 0, &ent_y, &mut ly)?;
        let mut luv = reconstruct_latent_with(eng, hdr, 1, &set.common[1], &ent_uv, stop)?;
        post_process_latent(hdr, &headers.tools, 1, &ent_uv, &mut luv)?;
        phases.latent_host_ns = since(t);
        // The last point a stop can still keep work off the GPU.
        stop.check().map_err(zenjpegai::Error::from)?;

        // Decide the tail before borrowing the workspace: for presentable pictures the RGBA
        // conversion (and its staged readback for `RgbaReadback`) rides along in the decode's
        // single submission; `to_rgba_texture`/`read_rgba` then only map.
        let rgba_tex = match out {
            GpuOut::Rgba | GpuOut::RgbaReadback if presentable(&headers) => {
                let (dh, dw) = (
                    hdr.height as usize - hdr.diff_display_height as usize,
                    hdr.width as usize - hdr.diff_display_width as usize,
                );
                Some((rgba_texture(&self.ctx, dh, dw), dh, dw))
            }
            _ => None,
        };
        let mut ws = self
            .workspace
            .lock()
            .ok()
            .and_then(|mut w| w.take())
            .unwrap_or_default();
        let tail = match &rgba_tex {
            Some((tex, dh, dw)) => RunTail::Rgba {
                tex,
                out: (*dh, *dw),
                stage: out == GpuOut::RgbaReadback,
            },
            None => match out {
                GpuOut::Quantized => quantized_spec(&headers)
                    .map(RunTail::Output)
                    .unwrap_or(RunTail::Planes),
                _ => RunTail::Planes,
            },
        };
        let picture =
            set.synthesis
                .run_for_header_tailed(&mut ws, hdr, [&ly.y_hat, &luv.y_hat], tail);
        if let Ok(mut slot) = self.workspace.lock() {
            *slot = Some(ws);
        }
        let mut picture = picture?;
        picture.timing.headers_host_ns = headers_ns;
        picture.timing.common_host_ns = phases.common_host_ns;
        picture.timing.weights_host_ns = phases.weights_host_ns;
        picture.timing.entropy_host_ns = phases.entropy_host_ns;
        picture.timing.latent_host_ns = phases.latent_host_ns;
        Ok(GpuDecoded {
            picture,
            luma_scale_log: ent_y.scale_log,
            op,
            headers,
        })
    }

    /// Read the planes back: post-filters, colour conversion and rounding on the CPU, exactly as
    /// the CPU decoder does them (RGB, or YUV planes for streams coded that way; 8 or 10 bit).
    /// Also returns the synthesised planes before the post-filters (for parity measurements) and
    /// the timing with `gpu_ns` filled in.
    pub async fn finish(&self, decoded: GpuDecoded) -> Result<(Picture, Planes, Timing)> {
        self.finish_stop(decoded, &enough::Unstoppable).await
    }

    /// [`finish`](Self::finish) with cooperative cancellation: `stop` gates the readback wait
    /// and reaches the post-filter networks (they check it per tile / layer).
    pub async fn finish_stop(
        &self,
        mut decoded: GpuDecoded,
        stop: &dyn enough::Stop,
    ) -> Result<(Picture, Planes, Timing)> {
        stop.check().map_err(zenjpegai::Error::from)?;
        let hdr = &decoded.headers.picture;
        let rec = decoded.picture.read_planes().await?;
        stop.check().map_err(zenjpegai::Error::from)?;
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
                stop,
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

    /// [`finish`](Self::finish) for a decode staged with [`GpuOut::Quantized`]: the picture was
    /// converted to its output format on the GPU, so this maps the packed `u16` buffer (half
    /// the `f32` planes' bytes) instead of the planes. When the decode had to fall back to
    /// `Planes` (post-filters, chroma resampling, custom transform) this runs the CPU path.
    /// Does not return the pre-filter planes — use [`finish`](Self::finish) for those.
    pub async fn finish_picture(&self, decoded: GpuDecoded) -> Result<(Picture, Timing)> {
        self.finish_picture_stop(decoded, &enough::Unstoppable)
            .await
    }

    /// [`finish_picture`](Self::finish_picture) with cooperative cancellation.
    pub async fn finish_picture_stop(
        &self,
        mut decoded: GpuDecoded,
        stop: &dyn enough::Stop,
    ) -> Result<(Picture, Timing)> {
        stop.check().map_err(zenjpegai::Error::from)?;
        if decoded.picture.staged_output().is_some() {
            let (spec, data) = decoded.picture.read_output().await?;
            let picture = match spec.mode {
                1 => Picture::Rgb(RgbImage {
                    width: spec.out_w as usize,
                    height: spec.out_h as usize,
                    bit_depth: spec.bit_depth,
                    data,
                }),
                _ => {
                    let (ny, nc) = (spec.n_y as usize, spec.n_c as usize);
                    Picture::Yuv(YuvImage {
                        width: spec.out_w as usize,
                        height: spec.out_h as usize,
                        chroma_width: spec.cw as usize,
                        chroma_height: spec.ch as usize,
                        bit_depth: spec.bit_depth,
                        y: data[..ny].to_vec(),
                        u: data[ny..ny + nc].to_vec(),
                        v: data[ny + nc..ny + 2 * nc].to_vec(),
                    })
                }
            };
            return Ok((picture, decoded.picture.timing));
        }
        let (picture, _, timing) = self.finish_stop(decoded, stop).await?;
        Ok((picture, timing))
    }

    /// Codestream to whatever it holds (RGB or YUV planes), like `Decoder::decode_picture`.
    /// Converts to the output format on the GPU where it can ([`GpuOut::Quantized`]) and reads
    /// back the packed samples; other streams read the `f32` planes and finish on the CPU.
    pub async fn decode_picture_async(&self, stream: &[u8]) -> Result<Picture> {
        self.decode_picture_stop(stream, &enough::Unstoppable).await
    }

    /// [`decode_picture_async`](Self::decode_picture_async) with cooperative cancellation: the
    /// `stop` token the CPU decoder takes, checked between the CPU stages, before the synthesis
    /// submit and before the readback.
    pub async fn decode_picture_stop(
        &self,
        stream: &[u8],
        stop: &dyn enough::Stop,
    ) -> Result<Picture> {
        let decoded = self.decode_to_gpu_stop(stream, GpuOut::Quantized, stop)?;
        Ok(self.finish_picture_stop(decoded, stop).await?.0)
    }

    /// Codestream to interleaved RGB with cooperative cancellation.
    pub async fn decode_stop(&self, stream: &[u8], stop: &dyn enough::Stop) -> Result<RgbImage> {
        match self.decode_picture_stop(stream, stop).await? {
            Picture::Rgb(image) => Ok(image),
            Picture::Yuv(_) => Err(GpuError::Unsupported(
                "the stream decodes to YUV planes: use decode_picture_stop",
            )),
        }
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

    /// Blocking [`decode_stop`](Self::decode_stop).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn decode_with(&self, stream: &[u8], stop: &dyn enough::Stop) -> Result<RgbImage> {
        pollster::block_on(self.decode_stop(stream, stop))
    }
}
