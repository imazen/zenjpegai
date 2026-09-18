//! The synthesis transforms (`y_hat` to reconstructed planes) on the GPU: SOP, BOP and HOP,
//! luma and chroma, tile by tile.
//!
//! Layer structure, tensor names and crops mirror `zenjpegai::model::{synthesis, attention}`;
//! the weights come from the same checkpoints.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use zenjpegai::header::{OperatingPoint, PictureHeader, SynthesisTiling};
use zenjpegai::model::{ModelSource, synthesis_path};
use zenjpegai::nn::{Conv2d, ConvTranspose2d};
use zenjpegai::tensor::Tensor;
use zenjpegai::tools::regions::{Plane, RegionGrid, region_grid};
use zenjpegai::tools::tiles::{SynthesisTile, synthesis_tiles};
use zenjpegai::weights::Checkpoint;

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::kernels::{Act, Pointwise, Res};
use crate::layers::{GpuConv, GpuConvTranspose, GpuDepthwise, GpuLayerNorm, f32_buffer, pack_hwc4};
use crate::plan::{Graph, Plan, Pool, RETAIN_FACTOR, RETAIN_SLACK, T};

// ---------------------------------------------------------------- checkpoint loading

struct Loader<'a> {
    ctx: &'a GpuContext,
    ck: &'a Checkpoint<'a>,
}

impl Loader<'_> {
    fn model_err(msg: String) -> GpuError {
        GpuError::Codec(zenjpegai::Error::Model(msg))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Conv2d> {
        let w = self.ck.f32(&format!("{prefix}.weight"))?;
        if w.shape != [out_ch, in_ch / groups, k, k] {
            return Err(Self::model_err(format!(
                "{prefix}.weight: shape {:?}",
                w.shape
            )));
        }
        let b = bias
            .then(|| self.ck.f32(&format!("{prefix}.bias")).map(|b| b.data))
            .transpose()?;
        if !bias && self.ck.contains(&format!("{prefix}.bias")) {
            return Err(Self::model_err(format!("{prefix}: unexpected bias")));
        }
        Ok(Conv2d::new(
            in_ch,
            out_ch,
            (k, k),
            stride,
            (pad, pad),
            groups,
            w.data,
            b,
        )?)
    }

    fn conv3x3(&self, prefix: &str, i: usize, o: usize, bias: bool) -> Result<GpuConv> {
        GpuConv::new(self.ctx, &self.conv2d(prefix, i, o, 3, 1, 1, 1, bias)?)
    }

    fn conv3x3_s2(&self, prefix: &str, c: usize) -> Result<GpuConv> {
        GpuConv::new(self.ctx, &self.conv2d(prefix, c, c, 3, 2, 1, 1, true)?)
    }

    fn conv1x1(&self, prefix: &str, i: usize, o: usize, bias: bool) -> Result<GpuConv> {
        GpuConv::new(self.ctx, &self.conv2d(prefix, i, o, 1, 1, 0, 1, bias)?)
    }

    fn depthwise(&self, prefix: &str, c: usize) -> Result<GpuDepthwise> {
        GpuDepthwise::new(self.ctx, &self.conv2d(prefix, c, c, 3, 1, 1, c, false)?)
    }

    fn conv_t(
        &self,
        prefix: &str,
        i: usize,
        o: usize,
        k: usize,
        pad: usize,
        out_pad: usize,
    ) -> Result<GpuConvTranspose> {
        let w = self.ck.f32(&format!("{prefix}.weight"))?;
        if w.shape != [i, o, k, k] {
            return Err(Self::model_err(format!(
                "{prefix}.weight: shape {:?}",
                w.shape
            )));
        }
        let b = self.ck.f32(&format!("{prefix}.bias"))?.data;
        GpuConvTranspose::new(
            self.ctx,
            &ConvTranspose2d::new(i, o, k, 2, pad, out_pad, w.data, Some(b))?,
        )
    }

    /// `conv3x3_t`: exactly doubles the size.
    fn conv3x3_t(&self, prefix: &str, i: usize, o: usize) -> Result<GpuConvTranspose> {
        self.conv_t(prefix, i, o, 3, 1, 1)
    }

    fn layer_norm(&self, prefix: &str, dim: usize) -> Result<GpuLayerNorm> {
        let (w, b) = (
            self.ck.f32(&format!("{prefix}.weight"))?,
            self.ck.f32(&format!("{prefix}.bias"))?,
        );
        if w.shape != [dim] || b.shape != [dim] {
            return Err(Self::model_err(format!("{prefix}: layer norm shape")));
        }
        GpuLayerNorm::new(self.ctx, &w.data, &b.data)
    }
}

// ---------------------------------------------------------------- building blocks

/// `y = x * (1 + conv1x1(conv3x3_grouped(relu6(x))))`.
struct ResAu {
    conv: GpuConv,
    conv2: GpuConv,
}

impl ResAu {
    fn load(l: &Loader<'_>, prefix: &str, chs: usize) -> Result<Self> {
        let name = format!("{prefix}.conv.weight");
        let per_group =
            l.ck.info(&name)
                .and_then(|t| t.shape.get(1).copied())
                .filter(|&g| g != 0 && chs.is_multiple_of(g))
                .ok_or_else(|| Loader::model_err(format!("{name}: missing or malformed")))?;
        Ok(Self {
            conv: GpuConv::new(
                l.ctx,
                &l.conv2d(
                    &format!("{prefix}.conv"),
                    chs,
                    chs,
                    3,
                    1,
                    1,
                    chs / per_group,
                    false,
                )?,
            )?,
            conv2: l.conv1x1(&format!("{prefix}.conv2"), chs, chs, false)?,
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        let m = g.conv_ex(x, &self.conv, (0, 0), true, Act::None)?;
        g.conv_res(m, &self.conv2, Act::None, Res::Gate, x)
    }
}

enum Upsample {
    Transposed(GpuConvTranspose),
    Conv2x2Shuffle(GpuConv),
}

impl Upsample {
    fn load(l: &Loader<'_>, op: OperatingPoint, prefix: &str, i: usize, o: usize) -> Result<Self> {
        match op {
            OperatingPoint::Bop => Ok(Self::Transposed(l.conv_t(prefix, i, o, 4, 1, 0)?)),
            OperatingPoint::Sop => Ok(Self::Conv2x2Shuffle(GpuConv::new(
                l.ctx,
                &l.conv2d(&format!("{prefix}.conv"), i, o * 4, 2, 1, 0, 1, false)?,
            )?)),
            OperatingPoint::Hop => Err(GpuError::Shape("HOP is not a light transform".into())),
        }
    }

    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        match self {
            Self::Transposed(c) => g.conv_transpose(x, c),
            Self::Conv2x2Shuffle(c) => {
                let t = g.conv_ex(x, c, (1, 1), false, Act::None)?;
                g.pixel_shuffle(t, 2)
            }
        }
    }
}

struct ResidualBlock {
    conv1: GpuConv,
    conv2: GpuConv,
}

impl ResidualBlock {
    fn load(l: &Loader<'_>, prefix: &str, chs: usize) -> Result<Self> {
        Ok(Self {
            conv1: l.conv3x3(&format!("{prefix}.conv1"), chs, chs, true)?,
            conv2: l.conv3x3(&format!("{prefix}.conv2"), chs, chs, true)?,
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        let t = g.conv_ex(x, &self.conv1, (0, 0), false, Act::Relu)?;
        g.conv_res(t, &self.conv2, Act::None, Res::Add, x)
    }
}

/// `x + trunk(x) * sigmoid(mask(x))`, the mask computed at half resolution.
struct Cab {
    trunk: [ResidualBlock; 2],
    subscale: GpuConv,
    mask2: ResidualBlock,
    mask3: ResidualBlock,
    upscale: GpuConvTranspose,
}

impl Cab {
    fn load(l: &Loader<'_>, prefix: &str, chs: usize) -> Result<Self> {
        let rb = |name: &str| ResidualBlock::load(l, &format!("{prefix}.{name}"), chs);
        Ok(Self {
            trunk: [rb("residual_trunk.0")?, rb("residual_trunk.1")?],
            subscale: l.conv3x3_s2(&format!("{prefix}.subscale"), chs)?,
            mask2: rb("residual_mask2.0")?,
            mask3: rb("residual_mask3.0")?,
            upscale: l.conv3x3_t(&format!("{prefix}.upscale"), chs, chs)?,
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        let t = self.trunk[0].forward(g, x)?;
        let trunk = self.trunk[1].forward(g, t)?;
        let m = g.conv(x, &self.subscale)?;
        let m = self.mask2.forward(g, m)?;
        let m = self.mask3.forward(g, m)?;
        let m = g.conv_transpose(m, &self.upscale)?;
        if (m.h, m.w) != (trunk.h, trunk.w) {
            return Err(GpuError::Codec(zenjpegai::Error::InvalidArgument(
                "CAB: odd feature map size",
            )));
        }
        g.pointwise(Pointwise::SigmoidMulAdd, trunk, &[m, x])?;
        Ok(trunk)
    }
}

const HEADS: usize = 4;
const FFN_GAMMA: usize = 4;
const FFN_SPLIT: usize = 2;

/// Output channels `first .. first + n` of an ungrouped convolution.
fn out_slice(c: &Conv2d, first: usize, n: usize) -> Result<Conv2d> {
    let per = c.in_ch * c.kh * c.kw;
    Ok(Conv2d::new(
        c.in_ch,
        n,
        (c.kh, c.kw),
        c.stride,
        (c.pad_h, c.pad_w),
        1,
        c.weight[first * per..(first + n) * per].to_vec(),
        c.bias.as_ref().map(|b| b[first..first + n].to_vec()),
    )?)
}

/// Channels `first .. first + n` of a bias-free depthwise 3x3 convolution.
fn depthwise_slice(c: &Conv2d, first: usize, n: usize) -> Result<Conv2d> {
    Ok(Conv2d::new(
        n,
        n,
        (3, 3),
        1,
        (1, 1),
        n,
        c.weight[first * 9..(first + n) * 9].to_vec(),
        None,
    )?)
}

struct TransformerBlock {
    prep_norm: GpuLayerNorm,
    prep_conv1: GpuConv,
    prep_conv2: GpuDepthwise,
    temperature: wgpu::Buffer,
    attn_out: GpuConv,
    ffn_norm: GpuLayerNorm,
    /// The feed-forward's expansion (`project_in` 1x1 + depthwise 3x3 to `2 * hidden` channels),
    /// split by output channel into `FFN_SPLIT` (value, gate) pairs: every channel is computed
    /// exactly as in the unsplit layer, but the largest feature map is `1 / (2 * FFN_SPLIT)` of it
    /// (128 MiB instead of 256 MiB for a 1024-px luma tile: WebGPU's default binding limit).
    ffn_expand: Vec<[(GpuConv, GpuDepthwise); 2]>,
    ffn_out: GpuConv,
    hidden: usize,
}

impl TransformerBlock {
    fn load(l: &Loader<'_>, prefix: &str, dim: usize) -> Result<Self> {
        let hidden = dim * FFN_GAMMA;
        let p = |name: &str| format!("{prefix}.{name}");
        let temperature = l.ck.f32(&p("attn.temperature"))?;
        if temperature.shape != [HEADS, 1, 1] {
            return Err(Loader::model_err(format!("{prefix}: temperature shape")));
        }
        Ok(Self {
            prep_norm: l.layer_norm(&p("prep_data.norm1"), dim)?,
            prep_conv1: l.conv1x1(&p("prep_data.conv1"), dim, 3 * dim, false)?,
            prep_conv2: l.depthwise(&p("prep_data.conv2"), 3 * dim)?,
            temperature: f32_buffer(l.ctx, &temperature.data),
            attn_out: l.conv1x1(&p("attn.project_out"), dim, dim, false)?,
            ffn_norm: l.layer_norm(&p("ffn.norm1"), dim)?,
            ffn_expand: (0..FFN_SPLIT)
                .map(|q| -> Result<_> {
                    let n = hidden / FFN_SPLIT;
                    let part = |first: usize| -> Result<(GpuConv, GpuDepthwise)> {
                        let pw =
                            l.conv2d(&p("ffn.project_in"), dim, 2 * hidden, 1, 1, 0, 1, false)?;
                        let dw = l.conv2d(
                            &p("ffn.dwconv"),
                            2 * hidden,
                            2 * hidden,
                            3,
                            1,
                            1,
                            2 * hidden,
                            false,
                        )?;
                        Ok((
                            GpuConv::new(l.ctx, &out_slice(&pw, first, n)?)?,
                            GpuDepthwise::new(l.ctx, &depthwise_slice(&dw, first, n)?)?,
                        ))
                    };
                    Ok([part(q * n)?, part(hidden + q * n)?])
                })
                .collect::<Result<_>>()?,
            hidden,
            ffn_out: l.conv1x1(&p("ffn.project_out"), hidden, dim, false)?,
        })
    }

    /// In place on `x`.
    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        let t = g.layer_norm(x, &self.prep_norm)?;
        let t = g.conv(t, &self.prep_conv1)?;
        let t = g.depthwise(t, &self.prep_conv2)?;
        let a = g.channel_attention(t, HEADS, &self.temperature)?;
        let a = g.conv(a, &self.attn_out)?;
        g.pointwise(Pointwise::Add, x, &[a])?;

        let t = g.layer_norm(x, &self.ffn_norm)?;
        let f = g.alloc(self.hidden, t.h, t.w)?;
        for (q, [(pw_a, dw_a), (pw_b, dw_b)]) in self.ffn_expand.iter().enumerate() {
            let a = g.conv(t, pw_a)?;
            let a = g.depthwise(a, dw_a)?;
            let b = g.conv(t, pw_b)?;
            let b = g.depthwise(b, dw_b)?;
            g.elu_gate_into(a, b, f, q * (self.hidden / FFN_SPLIT) / 4)?;
        }
        let f = g.conv(f, &self.ffn_out)?;
        g.pointwise(Pointwise::Add, x, &[f])?;
        Ok(x)
    }
}

struct Tam {
    resample: Option<(GpuConv, GpuConvTranspose)>,
    blocks: [TransformerBlock; 2],
}

impl Tam {
    fn load(l: &Loader<'_>, prefix: &str, dim: usize, downsample: bool) -> Result<Self> {
        let resample = downsample
            .then(|| -> Result<_> {
                Ok((
                    l.conv3x3_s2(&format!("{prefix}.ds_conv"), dim)?,
                    l.conv3x3_t(&format!("{prefix}.us_conv"), dim, dim)?,
                ))
            })
            .transpose()?;
        let tb = |i: usize| TransformerBlock::load(l, &format!("{prefix}.TABs.{i}"), dim);
        Ok(Self {
            resample,
            blocks: [tb(0)?, tb(1)?],
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T) -> Result<T> {
        let x = match &self.resample {
            Some((ds, _)) => g.conv(x, ds)?,
            None => x,
        };
        let x = self.blocks[0].forward(g, x)?;
        let x = self.blocks[1].forward(g, x)?;
        match &self.resample {
            Some((_, us)) => g.conv_transpose(x, us),
            None => Ok(x),
        }
    }
}

// ---------------------------------------------------------------- the transforms

/// What the last feature map of a transform looks like: PixelShuffle factor and plane count.
struct Tail {
    x: T,
    r: usize,
    planes: usize,
}

struct LightPrimary {
    res_conv: GpuConv,
    up1: Upsample,
    act1: ResAu,
    up2: Upsample,
    act2: ResAu,
    conv3: GpuConv,
    act3: ResAu,
    conv4: GpuConv,
}

struct HopPrimary {
    res: ResidualBlock,
    up1: GpuConvTranspose,
    act1: ResAu,
    up2: GpuConvTranspose,
    cab: Cab,
    act2: ResAu,
    conv3: GpuConv,
    tam: Tam,
    act3: ResAu,
    up4: GpuConvTranspose,
}

enum Primary {
    Light(Box<LightPrimary>),
    Hop(Box<HopPrimary>),
}

impl Primary {
    fn load(l: &Loader<'_>, op: OperatingPoint) -> Result<Self> {
        let (c1, c2, c3) = match op {
            OperatingPoint::Sop => (64, 32, 32),
            OperatingPoint::Bop => (64, 64, 96),
            OperatingPoint::Hop => {
                let c = 128;
                return Ok(Self::Hop(Box::new(HopPrimary {
                    res: ResidualBlock::load(l, "first_stage.0.0", 160)?,
                    up1: l.conv3x3_t("first_stage.1", 160, c)?,
                    act1: ResAu::load(l, "first_stage.2", c)?,
                    up2: l.conv3x3_t("conv2_t", c, c)?,
                    cab: Cab::load(l, "CAB", c)?,
                    act2: ResAu::load(l, "iact2", c)?,
                    conv3: l.conv1x1("conv3_t", c, 4 * c, true)?,
                    tam: Tam::load(l, "TAM", c, true)?,
                    act3: ResAu::load(l, "iact3", c)?,
                    up4: l.conv3x3_t("conv4_t", c, 1)?,
                })));
            }
        };
        Ok(Self::Light(Box::new(LightPrimary {
            res_conv: l.conv3x3("first_stage.0.0.conv1", 160, 160, true)?,
            up1: Upsample::load(l, op, "first_stage.1", 160, c1)?,
            act1: ResAu::load(l, "first_stage.2", c1)?,
            up2: Upsample::load(l, op, "conv2_t", c1, c2)?,
            act2: ResAu::load(l, "iact2", c2)?,
            conv3: l.conv3x3("conv3", c2, c3, true)?,
            act3: ResAu::load(l, "iact3", c3)?,
            conv4: l.conv1x1("conv4", c3, 16, false)?,
        })))
    }

    fn forward(&self, g: &mut Graph<'_>, y: T, h: usize, w: usize) -> Result<Tail> {
        match self {
            Self::Light(m) => {
                // LightResidualBlock: relu(conv(x)) + x
                let x = g.conv_res(y, &m.res_conv, Act::Relu, Res::Add, y)?;
                let x = m.up1.forward(g, x)?;
                let x = m.act1.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(8), w.div_ceil(8))?;
                let x = m.up2.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(4), w.div_ceil(4))?;
                let x = m.act2.forward(g, x)?;
                let x = g.conv(x, &m.conv3)?;
                let x = m.act3.forward(g, x)?;
                let x = g.conv(x, &m.conv4)?;
                Ok(Tail { x, r: 4, planes: 1 })
            }
            Self::Hop(m) => {
                let x = m.res.forward(g, y)?;
                let x = g.conv_transpose(x, &m.up1)?;
                let x = m.act1.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(8), w.div_ceil(8))?;
                let x = g.conv_transpose(x, &m.up2)?;
                let x = m.cab.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(4), w.div_ceil(4))?;
                let x = m.act2.forward(g, x)?;
                let x = g.conv(x, &m.conv3)?;
                let x = g.pixel_shuffle(x, 2)?;
                let x = m.tam.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(2), w.div_ceil(2))?;
                let x = m.act3.forward(g, x)?;
                let x = g.conv_transpose(x, &m.up4)?;
                Ok(Tail { x, r: 1, planes: 1 })
            }
        }
    }
}

struct LightSecondary {
    up: Upsample,
    act2: ResAu,
    conv3: GpuConv,
    act3: ResAu,
    conv4: GpuConv,
}

struct HopSecondary {
    up2: GpuConvTranspose,
    cab: Cab,
    act2: ResAu,
    conv3: GpuConv,
    tam: Tam,
    act3: ResAu,
    up4: GpuConvTranspose,
}

enum SecondaryTail {
    Light(Box<LightSecondary>),
    Hop(Box<HopSecondary>),
}

struct Secondary {
    /// `LightCombineBlock`: conv3x3(160 + 96 -> 48); its output is `cat(info, chroma latent)`.
    combine: GpuConv,
    tail: SecondaryTail,
}

impl Secondary {
    fn load(l: &Loader<'_>, op: OperatingPoint) -> Result<Self> {
        let combine = l.conv3x3("first_stage.conv1", 256, 48, true)?;
        let tail = match op {
            OperatingPoint::Hop => {
                let c = 64;
                SecondaryTail::Hop(Box::new(HopSecondary {
                    up2: l.conv3x3_t("conv2_t", 144, c)?,
                    cab: Cab::load(l, "CAB", c)?,
                    act2: ResAu::load(l, "iact2", c)?,
                    conv3: l.conv1x1("conv3_t", c, 4 * c, true)?,
                    tam: Tam::load(l, "TAM", c, false)?,
                    act3: ResAu::load(l, "iact3", c)?,
                    up4: l.conv3x3_t("conv4_t", c, 8)?,
                }))
            }
            _ => {
                let (c2, c3) = if op == OperatingPoint::Sop {
                    (32, 32)
                } else {
                    (64, 128)
                };
                SecondaryTail::Light(Box::new(LightSecondary {
                    up: Upsample::load(l, op, "conv2_t", 144, c2)?,
                    act2: ResAu::load(l, "iact2", c2)?,
                    conv3: l.conv3x3("conv3", c2, c3, true)?,
                    act3: ResAu::load(l, "iact3", c3)?,
                    conv4: l.conv1x1("conv4", c3, 128, false)?,
                }))
            }
        };
        Ok(Self { combine, tail })
    }

    fn forward(
        &self,
        g: &mut Graph<'_>,
        y_luma: T,
        y_chroma: T,
        h: usize,
        w: usize,
    ) -> Result<Tail> {
        let x = g.cat(&[y_luma, y_chroma])?;
        let info = g.conv(x, &self.combine)?;
        let x = g.cat(&[info, y_chroma])?;
        match &self.tail {
            SecondaryTail::Light(m) => {
                let x = m.up.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(8), w.div_ceil(8))?;
                let x = m.act2.forward(g, x)?;
                let x = g.conv(x, &m.conv3)?;
                let x = m.act3.forward(g, x)?;
                let x = g.conv(x, &m.conv4)?;
                Ok(Tail { x, r: 8, planes: 2 })
            }
            SecondaryTail::Hop(m) => {
                let x = g.conv_transpose(x, &m.up2)?;
                let x = m.cab.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(8), w.div_ceil(8))?;
                let x = m.act2.forward(g, x)?;
                let x = g.conv(x, &m.conv3)?;
                let x = g.pixel_shuffle(x, 2)?;
                let x = m.tam.forward(g, x)?;
                let x = g.crop(x, h.div_ceil(4), w.div_ceil(4))?;
                let x = m.act3.forward(g, x)?;
                let x = g.conv_transpose(x, &m.up4)?;
                Ok(Tail { x, r: 2, planes: 2 })
            }
        }
    }
}

// ---------------------------------------------------------------- picture-level state

const MAX_TIMED_TILES: u32 = 64;
/// Cap on the dispatches a profiling run times individually (query-set size).
const MAX_PROFILED_DISPATCHES: usize = 2048;

struct Queries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

/// GPU memory that outlives a picture: activation pool, latent upload buffers, the picture
/// buffer, the readback staging buffer. Reused across pictures and across models; buffers only
/// ever grow. One workspace serves one decode at a time.
#[derive(Default)]
pub struct Workspace {
    pool: Arc<Mutex<Pool>>,
    latents: [Option<wgpu::Buffer>; 2],
    rec: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,
    queries: Option<Queries>,
    generation: u64,
    profile: bool,
}

/// Keep the buffer at `bytes`: grow when too small, and shrink when it exceeds
/// `RETAIN_FACTOR * bytes + RETAIN_SLACK` — the same bounded-retention rule as the
/// activation pool, so a picture-level buffer sized for a large picture does not stay bound
/// (and resident) for later small ones. Returns `true` when the buffer was replaced; the
/// caller bumps `Workspace::generation` so plans that bound the old buffer are dropped.
fn fit(
    slot: &mut Option<wgpu::Buffer>,
    ctx: &GpuContext,
    bytes: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> Result<bool> {
    let oversized = slot.as_ref().is_some_and(|b| {
        b.size()
            > bytes
                .saturating_mul(RETAIN_FACTOR)
                .saturating_add(RETAIN_SLACK)
    });
    if !oversized && slot.as_ref().is_some_and(|b| b.size() >= bytes) {
        return Ok(false);
    }
    if bytes > ctx.device().limits().max_buffer_size {
        return Err(GpuError::TooLarge {
            needed: bytes,
            limit: ctx.device().limits().max_buffer_size,
        });
    }
    *slot = Some(ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage,
        mapped_at_creation: false,
    }));
    Ok(true)
}

impl Workspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Time every dispatch separately (one compute pass each) so
    /// [`GpuPicture::dispatch_ns`] can report where the device time went. Off by default:
    /// splitting the pass perturbs the total. Needs an adapter with timestamp queries.
    pub fn set_profile(&mut self, on: bool) {
        self.profile = on;
    }

    /// Drop every retained buffer (activation slots and the latent / picture / readback
    /// buffers); the next run rebuilds them. Bounded retention usually makes this unneeded —
    /// it is for callers that want the memory back without dropping the workspace, like
    /// `zenjpegai::nn::fast::release_buffers` on the CPU side.
    pub fn release_buffers(&mut self) {
        if let Ok(mut p) = self.pool.lock() {
            p.release();
        }
        self.latents = [None, None];
        self.rec = None;
        self.staging = None;
        self.queries = None;
        self.generation += 1;
    }

    /// Bytes of GPU memory currently held.
    pub fn bytes(&self) -> u64 {
        let pool = self.pool.lock().map(|p| p.bytes()).unwrap_or(0);
        pool + self
            .latents
            .iter()
            .chain([&self.rec, &self.staging])
            .flatten()
            .map(|b| b.size())
            .sum::<u64>()
    }

    fn stamp(&self) -> (u64, u64) {
        (
            self.generation,
            self.pool.lock().map(|p| p.generation).unwrap_or(0),
        )
    }
}

/// Where the time of one [`GpuSynthesis::run`] went. Host-side numbers only say how long the
/// CPU spent *submitting*; `gpu_ns` is the device's own clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub tiles: usize,
    pub dispatches: usize,
    /// Plans built (pipelines compiled, bind groups created) during this run; 0 when warm.
    pub plans_built: usize,
    /// Largest feature map bound by any tile's plan, in bytes.
    pub largest_binding_bytes: u64,
    /// Host time packing and queueing the latent upload.
    pub upload_host_ns: u64,
    /// Host time building plans (cold only).
    pub plan_host_ns: u64,
    /// Host time encoding and submitting the tile command buffers.
    pub submit_host_ns: u64,
    /// Host time in `GpuDecoder` before synthesis: codestream parse + header read.
    pub headers_host_ns: u64,
    /// Host time parsing the common (entropy/latent) networks — a `(model, op)` cache miss
    /// only, 0 on warm decodes.
    pub common_host_ns: u64,
    /// Host time parsing the synthesis checkpoints and uploading their weights — a cache miss
    /// only, 0 on warm decodes.
    pub weights_host_ns: u64,
    /// Host time in the entropy stage (z substream, residuals, quality map; both components).
    pub entropy_host_ns: u64,
    /// Host time in latent reconstruction + post-processing (hyper-decoder, context model,
    /// LSBS; both components). This is the serial CPU stage the GPU work waits on.
    pub latent_host_ns: u64,
    /// Sum of the tiles' compute-pass durations from GPU timestamp queries, when the device has
    /// them and the timestamps were read back (through [`GpuPicture::read_planes`],
    /// [`GpuPicture::read_rgba`] or [`GpuPicture::gpu_time`]).
    pub gpu_ns: Option<u64>,
}

/// Extra commands [`GpuSynthesis::run_tailed`] appends to its single submission, so the whole
/// decode is one queue submit and at most one map round-trip.
#[derive(Default)]
pub(crate) enum RunTail<'a> {
    /// The planes stay in the picture buffer; readers submit their own copies.
    #[default]
    None,
    /// Copy the picture buffer into the staging buffer — `read_planes` then only maps.
    Planes,
    /// Convert the picture to the caller's `rgba8unorm` texture (display `out_h x out_w`) and,
    /// when `stage`, copy it into the staging buffer for `read_rgba`. The texture is stored on
    /// the picture (`GpuPicture::rgba_texture`).
    Rgba {
        tex: &'a wgpu::Texture,
        out: (usize, usize),
        stage: bool,
    },
}

/// What the run's tail left in the staging buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Staged {
    None,
    /// Planar `f32` picture, tightly packed `[3, h, w]`.
    Planes,
    /// `rgba8` rows at the padded copy stride.
    Rgba {
        stride: usize,
        h: usize,
        w: usize,
    },
}

/// A synthesised picture still on the GPU: planar `f32` YUV `[3, height, width]` in `[0, 255]`
/// at coded size, chroma at luma resolution.
pub struct GpuPicture {
    ctx: Arc<GpuContext>,
    /// The picture buffer (`STORAGE | COPY_SRC`); valid until the workspace's next run.
    pub buffer: wgpu::Buffer,
    pub height: usize,
    pub width: usize,
    pub timing: Timing,
    staging: wgpu::Buffer,
    staged: Staged,
    rgba_tex: Option<wgpu::Texture>,
    queries: Option<(wgpu::Buffer, u32)>,
    profile: Option<(wgpu::Buffer, Vec<String>)>,
}

// `Instant` panics on wasm32-unknown-unknown; the browser timer is `Date.now` (ms as f64).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn now() -> std::time::Instant {
    std::time::Instant::now()
}
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn since(t: std::time::Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}
#[cfg(target_arch = "wasm32")]
pub(crate) fn now() -> f64 {
    js_sys::Date::now()
}
#[cfg(target_arch = "wasm32")]
pub(crate) fn since(t: f64) -> u64 {
    ((js_sys::Date::now() - t) * 1e6) as u64
}

impl GpuPicture {
    /// Sum the tile-pass durations out of a mapped timestamp-result buffer.
    fn read_ticks(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<u64> {
        let view = buf
            .slice(0..bytes)
            .get_mapped_range()
            .map_err(|e| GpuError::Device(e.to_string()))?;
        let ticks: u64 = bytemuck::cast_slice::<u8, u64>(&view)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|t| t[1].saturating_sub(t[0]))
            .sum();
        drop(view);
        buf.unmap();
        Ok(ticks)
    }

    /// Map the staging buffer and (when timestamp queries ran) the query-result buffer in one
    /// wait, then read the timestamps into `timing.gpu_ns`.
    async fn map_staging(&mut self, bytes: u64) -> Result<()> {
        let ctx = &self.ctx;
        match &self.queries {
            Some((qbuf, tiles)) => {
                let qbytes = *tiles as u64 * 16;
                ctx.map_read2(&self.staging, bytes, qbuf, qbytes).await?;
                let ticks = self.read_ticks(qbuf, qbytes)?;
                self.timing.gpu_ns =
                    Some((ticks as f64 * ctx.queue().get_timestamp_period() as f64) as u64);
            }
            None => ctx.map_read(&self.staging, bytes).await?,
        }
        Ok(())
    }

    /// Read the three planes back: `[3, height, width]`. Also fills `timing.gpu_ns`.
    pub async fn read_planes(&mut self) -> Result<Tensor<f32>> {
        let ctx = &self.ctx;
        let bytes = (3 * self.height * self.width * 4) as u64;
        if self.staged != Staged::Planes {
            let mut enc = ctx.device().create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.buffer, 0, &self.staging, 0, bytes);
            ctx.queue().submit([enc.finish()]);
        }
        self.map_staging(bytes).await?;
        let data = {
            let view = self
                .staging
                .slice(0..bytes)
                .get_mapped_range()
                .map_err(|e| GpuError::Device(e.to_string()))?;
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        self.staging.unmap();
        Ok(Tensor::from_vec(3, self.height, self.width, data)?)
    }

    /// Read back the RGBA picture the run staged with `RunTail::Rgba { stage: true }` — the map is
    /// the only wait, everything was recorded into the decode's single submission. Also fills
    /// `timing.gpu_ns`.
    pub async fn read_rgba(&mut self) -> Result<Vec<u8>> {
        let Staged::Rgba { stride, h, w } = self.staged else {
            return Err(GpuError::Shape(
                "no staged RGBA readback (the run needs RunTail::Rgba stage)".into(),
            ));
        };
        self.map_staging((stride * h) as u64).await?;
        let view = self
            .staging
            .slice(0..(stride * h) as u64)
            .get_mapped_range()
            .map_err(|e| GpuError::Device(e.to_string()))?;
        let mut out = Vec::with_capacity(w * h * 4);
        for row in 0..h {
            out.extend_from_slice(&view[row * stride..][..w * 4]);
        }
        drop(view);
        self.staging.unmap();
        Ok(out)
    }

    /// Device time of each dispatch, in submission order, after a run of a workspace with
    /// [`Workspace::set_profile(true)`](Workspace::set_profile). `None` otherwise. Every
    /// dispatch was its own compute pass, so the sum is slightly above the single-pass total.
    pub async fn dispatch_ns(&self) -> Result<Option<Vec<(String, u64)>>> {
        let Some((buf, labels)) = &self.profile else {
            return Ok(None);
        };
        let bytes = labels.len() as u64 * 16;
        self.ctx.map_read(buf, bytes).await?;
        let period = self.ctx.queue().get_timestamp_period() as f64;
        let out = {
            let view = buf
                .slice(0..bytes)
                .get_mapped_range()
                .map_err(|e| GpuError::Device(e.to_string()))?;
            bytemuck::cast_slice::<u8, u64>(&view)
                .as_chunks::<2>()
                .0
                .iter()
                .zip(labels)
                .map(|(t, l)| {
                    (
                        l.clone(),
                        (t[1].saturating_sub(t[0]) as f64 * period) as u64,
                    )
                })
                .collect()
        };
        buf.unmap();
        Ok(Some(out))
    }

    /// Device time of the tile passes (waits for the GPU). `None` without timestamp queries.
    pub async fn gpu_time(&self) -> Result<Option<u64>> {
        let Some((buf, tiles)) = &self.queries else {
            return Ok(None);
        };
        let bytes = *tiles as u64 * 16;
        self.ctx.map_read(buf, bytes).await?;
        let ticks = self.read_ticks(buf, bytes)?;
        Ok(Some(
            (ticks as f64 * self.ctx.queue().get_timestamp_period() as f64) as u64,
        ))
    }

    /// The `rgba8unorm` texture a [`RunTail::Rgba`] run already produced, if any.
    pub fn rgba_texture(&self) -> Option<&wgpu::Texture> {
        self.rgba_tex.as_ref()
    }

    /// Convert to an `rgba8unorm` texture (BT.709, 4:4:4) on the GPU, cropped to
    /// `out_h x out_w`, for presentation without a readback. The texture has
    /// `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC` usage. Returns the texture the run
    /// already made when it was built with a matching [`RunTail::Rgba`] instead of submitting a
    /// second conversion.
    pub fn to_rgba_texture(&self, out_h: usize, out_w: usize) -> Result<wgpu::Texture> {
        if let Some(tex) = &self.rgba_tex
            && tex.width() as usize == out_w
            && tex.height() as usize == out_h
        {
            return Ok(tex.clone());
        }
        if out_h > self.height || out_w > self.width || out_h == 0 || out_w == 0 {
            return Err(GpuError::Shape(
                "display size outside the coded picture".into(),
            ));
        }
        let tex = crate::decoder::rgba_texture(&self.ctx, out_h, out_w);
        let view = tex.create_view(&Default::default());
        let mut g = Graph::new(&self.ctx);
        g.yuv_to_rgba(
            &self.buffer,
            (self.height, self.width),
            (out_h, out_w),
            &view,
        );
        let plan = g.finish(&Arc::new(Mutex::new(Pool::default())))?;
        let mut enc = self
            .ctx
            .device()
            .create_command_encoder(&Default::default());
        plan.encode(&mut enc, None);
        self.ctx.queue().submit([enc.finish()]);
        Ok(tex)
    }
}

/// Cached plans and the workspace stamp they were built against.
type PlanCache = (HashMap<PlanKey, Arc<Plan>>, (u64, u64));

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PlanKey {
    tile: [usize; 10],
    latent: (usize, usize),
    picture: (usize, usize),
}

/// Both synthesis transforms of one (model, operating point) on the GPU.
pub struct GpuSynthesis {
    ctx: Arc<GpuContext>,
    op: OperatingPoint,
    luma: Primary,
    chroma: Secondary,
    plans: Mutex<PlanCache>,
}

impl GpuSynthesis {
    /// Load `VM_{sop,bop,hop}/decoder_{Y,UV}_<beta>.pth` of `model_id` from `source` and upload
    /// the packed weights.
    pub fn load(
        ctx: Arc<GpuContext>,
        source: &dyn ModelSource,
        model_id: usize,
        op: OperatingPoint,
    ) -> Result<Self> {
        // Same steps as `zenjpegai::model::with_checkpoint` (`.pth` or packed `ZJM1`; the source
        // is told which tensors were read so `pack-models` keeps them).
        let luma = {
            let rel = synthesis_path(model_id, 0, op)?;
            let file = source.read(&rel)?;
            let ck = Checkpoint::parse(&file)?;
            let net = Primary::load(&Loader { ctx: &ctx, ck: &ck }, op)?;
            source.accessed(&rel, &ck.touched_names());
            net
        };
        let chroma = {
            let rel = synthesis_path(model_id, 1, op)?;
            let file = source.read(&rel)?;
            let ck = Checkpoint::parse(&file)?;
            let net = Secondary::load(&Loader { ctx: &ctx, ck: &ck }, op)?;
            source.accessed(&rel, &ck.touched_names());
            net
        };
        Ok(Self {
            ctx,
            op,
            luma,
            chroma,
            plans: Mutex::new((HashMap::new(), (0, 0))),
        })
    }

    pub fn operating_point(&self) -> OperatingPoint {
        self.op
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    fn build_plan(
        &self,
        ws: &Workspace,
        tile: &SynthesisTile,
        latent: (usize, usize),
        picture: (usize, usize),
    ) -> Result<Plan> {
        let (Some(by), Some(buv), Some(rec)) = (&ws.latents[0], &ws.latents[1], &ws.rec) else {
            return Err(GpuError::Shape("workspace buffers missing".into()));
        };
        let mut g = Graph::new(&self.ctx);
        let (lat, img) = (tile.latent, tile.image);
        let y = g.external(by, 160, latent.0, latent.1)?;
        let uv = g.external(buv, 96, latent.0, latent.1)?;
        let y = g.window(y, lat.x, lat.y, lat.width, lat.height)?;
        let uv = g.window(uv, lat.x, lat.y, lat.width, lat.height)?;
        let core = (tile.core.x, tile.core.y, tile.core.width, tile.core.height);
        let t = self.luma.forward(&mut g, y, img.height, img.width)?;
        g.emit(t.x, t.r, t.planes, 0, rec, picture, core, tile.core_offset)?;
        let t = self.chroma.forward(&mut g, y, uv, img.height, img.width)?;
        g.emit(t.x, t.r, t.planes, 1, rec, picture, core, tile.core_offset)?;
        g.finish(&ws.pool)
    }

    /// Synthesize a whole picture of `height x width` coded samples from the two latents
    /// (`[160, H, W]` and `[96, H, W]`, planar). The result stays on the GPU.
    pub fn run(
        &self,
        ws: &mut Workspace,
        y_hat: [&Tensor<f32>; 2],
        (height, width): (usize, usize),
        tiling: Option<SynthesisTiling>,
        independent_regions: Option<&RegionGrid>,
    ) -> Result<GpuPicture> {
        self.run_tailed(
            ws,
            y_hat,
            (height, width),
            tiling,
            independent_regions,
            RunTail::None,
        )
    }

    /// [`run`](Self::run) with extra commands appended to the same submission (`Tail`).
    pub(crate) fn run_tailed(
        &self,
        ws: &mut Workspace,
        y_hat: [&Tensor<f32>; 2],
        (height, width): (usize, usize),
        tiling: Option<SynthesisTiling>,
        independent_regions: Option<&RegionGrid>,
        tail: RunTail<'_>,
    ) -> Result<GpuPicture> {
        let ctx = &self.ctx;
        let (lh, lw) = (y_hat[0].h, y_hat[0].w);
        if (y_hat[0].c, y_hat[1].c) != (160, 96) || (y_hat[1].h, y_hat[1].w) != (lh, lw) {
            return Err(GpuError::Codec(zenjpegai::Error::InvalidArgument(
                "luma / chroma latent size mismatch",
            )));
        }
        let tiles = synthesis_tiles(height, width, lh, lw, tiling, independent_regions)?;
        let mut timing = Timing {
            tiles: tiles.len(),
            ..Timing::default()
        };

        // Picture-level buffers. `fit` bounds them like the pool: a workspace that ran a
        // large picture releases the excess on the next small one instead of keeping the
        // large buffers bound (their old plans are dropped via `ws.generation`).
        let t0 = now();
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mut changed = false;
        for (i, t) in y_hat.iter().enumerate() {
            let packed = pack_hwc4(&t.data, t.c, lh, lw)?;
            changed |= fit(
                &mut ws.latents[i],
                ctx,
                packed.len() as u64 * 4,
                storage,
                "zenjpegai latent",
            )?;
            if let Some(b) = &ws.latents[i] {
                ctx.write_buffer(b, bytemuck::cast_slice(&packed));
            }
        }
        let rec_bytes = (3 * height * width * 4) as u64;
        if rec_bytes > ctx.max_binding_bytes() {
            return Err(GpuError::TooLarge {
                needed: rec_bytes,
                limit: ctx.max_binding_bytes(),
            });
        }
        changed |= fit(
            &mut ws.rec,
            ctx,
            rec_bytes,
            storage | wgpu::BufferUsages::COPY_SRC,
            "zenjpegai picture",
        )?;
        // The staging buffer must also fit a padded-stride RGBA copy when the tail stages one
        // (it always does for real pictures, but a degenerate tiny picture can pad up past the
        // planar size). Staging is only ever a copy destination, never bound, so replacing it
        // does not touch `ws.generation`.
        let stage_bytes = match &tail {
            RunTail::Rgba {
                stage: true, out, ..
            } => {
                let stride =
                    (out.1 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
                rec_bytes.max((stride * out.0) as u64)
            }
            _ => rec_bytes,
        };
        fit(
            &mut ws.staging,
            ctx,
            stage_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "zenjpegai readback",
        )?;
        if changed {
            ws.generation += 1;
        }
        if ctx.timestamps && ws.queries.is_none() {
            let bytes = MAX_TIMED_TILES as u64 * 16;
            ws.queries = Some(Queries {
                set: ctx.device().create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("zenjpegai tile timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: MAX_TIMED_TILES * 2,
                }),
                resolve: ctx.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                read: ctx.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            });
        }
        timing.upload_host_ns = since(t0);

        let timed = (tiles.len() as u32).min(MAX_TIMED_TILES);
        let mut resolved: Vec<Arc<Plan>> = Vec::with_capacity(tiles.len());
        for tile in tiles.iter() {
            let key = PlanKey {
                tile: [
                    tile.image.width,
                    tile.image.height,
                    tile.latent.x,
                    tile.latent.y,
                    tile.latent.width,
                    tile.latent.height,
                    tile.core.x,
                    tile.core.y,
                    tile.core_offset.0,
                    tile.core_offset.1,
                ],
                latent: (lh, lw),
                picture: (height, width),
            };
            let cached = {
                let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
                if plans.1 != ws.stamp() || plans.0.len() > 256 {
                    plans.0.clear();
                    plans.1 = ws.stamp();
                }
                plans.0.get(&key).cloned()
            };
            let plan = match cached {
                Some(p) => p,
                None => {
                    let t = now();
                    let p = Arc::new(self.build_plan(ws, tile, (lh, lw), (height, width))?);
                    timing.plan_host_ns += since(t);
                    timing.plans_built += 1;
                    let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
                    // Building may have grown the pool: older plans then pin stale buffers.
                    if plans.1 != ws.stamp() {
                        plans.0.clear();
                        plans.1 = ws.stamp();
                    }
                    plans.0.insert(key, p.clone());
                    p
                }
            };
            timing.dispatches += plan.dispatches();
            timing.largest_binding_bytes =
                timing.largest_binding_bytes.max(plan.largest_binding_bytes);
            resolved.push(plan);
        }

        // Per-dispatch profiling: its own query set, sized for this picture.
        let profile = (ws.profile
            && ctx.timestamps
            && timing.dispatches <= MAX_PROFILED_DISPATCHES)
            .then(|| {
                let count = 2 * timing.dispatches as u32;
                let bytes = count as u64 * 8;
                let labels: Vec<String> = resolved
                    .iter()
                    .flat_map(|p| p.labels().iter().cloned())
                    .collect();
                let set = ctx.device().create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("zenjpegai dispatch timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count,
                });
                let mk = |usage| {
                    ctx.device().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("zenjpegai dispatch timestamps"),
                        size: bytes,
                        usage,
                        mapped_at_creation: false,
                    })
                };
                let resolve = mk(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
                let read = mk(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                (set, resolve, read, labels)
            });

        // All tile plans, the timestamp resolve and the tail (readback copies / RGBA convert)
        // go into one command encoder: one queue submit per picture, not one per tile.
        let t_submit = now();
        let mut enc = ctx.device().create_command_encoder(&Default::default());
        let mut at = 0u32;
        for (i, plan) in resolved.iter().enumerate() {
            match &profile {
                Some((set, ..)) => {
                    plan.encode_profiled(&mut enc, set, at);
                    at += 2 * plan.dispatches() as u32;
                }
                None => {
                    let stamps = ws
                        .queries
                        .as_ref()
                        .filter(|_| (i as u32) < timed)
                        .map(|q| (&q.set, 2 * i as u32));
                    plan.encode(&mut enc, stamps);
                }
            }
        }
        if let Some((set, resolve, read, _)) = &profile {
            enc.resolve_query_set(set, 0..at, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, read, 0, at as u64 * 8);
        } else if let Some(q) = &ws.queries {
            enc.resolve_query_set(&q.set, 0..2 * timed, &q.resolve, 0);
            enc.copy_buffer_to_buffer(&q.resolve, 0, &q.read, 0, timed as u64 * 16);
        }

        let (Some(rec), Some(staging)) = (&ws.rec, &ws.staging) else {
            return Err(GpuError::Shape("workspace buffers missing".into()));
        };
        let mut staged = Staged::None;
        match &tail {
            RunTail::None => {}
            RunTail::Planes => {
                enc.copy_buffer_to_buffer(rec, 0, staging, 0, rec_bytes);
                staged = Staged::Planes;
            }
            RunTail::Rgba { tex, out, stage } => {
                if out.0 > height || out.1 > width || out.0 == 0 || out.1 == 0 {
                    return Err(GpuError::Shape(
                        "display size outside the coded picture".into(),
                    ));
                }
                // The conversion pass has no intermediate tensors: the pool is untouched.
                let mut g = Graph::new(ctx);
                g.yuv_to_rgba(
                    rec,
                    (height, width),
                    *out,
                    &tex.create_view(&Default::default()),
                );
                g.finish(&ws.pool)?.encode(&mut enc, None);
                if *stage {
                    let stride =
                        (out.1 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
                    enc.copy_texture_to_buffer(
                        tex.as_image_copy(),
                        wgpu::TexelCopyBufferInfo {
                            buffer: staging,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(stride as u32),
                                rows_per_image: None,
                            },
                        },
                        tex.size(),
                    );
                    staged = Staged::Rgba {
                        stride,
                        h: out.0,
                        w: out.1,
                    };
                }
            }
        }
        ctx.queue().submit([enc.finish()]);
        timing.submit_host_ns = since(t_submit);

        Ok(GpuPicture {
            ctx: self.ctx.clone(),
            buffer: rec.clone(),
            height,
            width,
            timing,
            staging: staging.clone(),
            staged,
            rgba_tex: match &tail {
                RunTail::Rgba { tex, .. } => Some((*tex).clone()),
                _ => None,
            },
            // In profile mode the per-tile query set is not written, so it must not be read.
            queries: profile
                .is_none()
                .then(|| ws.queries.as_ref().map(|q| (q.read.clone(), timed)))
                .flatten(),
            profile: profile.map(|(_, _, read, labels)| (read, labels)),
        })
    }

    /// [`run`](Self::run) with the tiling and region layout of a picture header, like
    /// `zenjpegai::decoder::reconstruct::synthesize`.
    pub fn run_for_header(
        &self,
        ws: &mut Workspace,
        hdr: &PictureHeader,
        y_hat: [&Tensor<f32>; 2],
    ) -> Result<GpuPicture> {
        self.run_for_header_tailed(ws, hdr, y_hat, RunTail::None)
    }

    /// [`run_for_header`](Self::run_for_header) with extra commands appended to the same
    /// submission (`Tail`).
    pub(crate) fn run_for_header_tailed(
        &self,
        ws: &mut Workspace,
        hdr: &PictureHeader,
        y_hat: [&Tensor<f32>; 2],
        tail: RunTail<'_>,
    ) -> Result<GpuPicture> {
        if hdr.components[0].synthesis_tiling != hdr.components[1].synthesis_tiling {
            return Err(GpuError::Codec(zenjpegai::Error::Unsupported(
                "different synthesis tiling for luma and chroma",
            )));
        }
        let regions = hdr
            .regions
            .filter(|r| r.independent)
            .map(|_| region_grid(hdr, 0, Plane::Image));
        self.run_tailed(
            ws,
            y_hat,
            (hdr.height as usize, hdr.width as usize),
            hdr.components[0].synthesis_tiling,
            regions.as_ref(),
            tail,
        )
    }
}
