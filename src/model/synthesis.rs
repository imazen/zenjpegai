//! Synthesis transforms: latent `y_hat` → reconstructed samples.
//!
//! Ports `ref/src/codec/components/autoencoder_data/decoder/{sop,bop,hop}_{prim,sec}.py`,
//! `activations/resau.py` and the residual blocks of `base_layers/conv_layers.py`; the HOP
//! attention blocks live in [`super::attention`].
//!
//! The primary (luma) transform maps `[160, H, W]` to one plane at 16x the resolution. The
//! secondary (chroma) transform takes the luma latent as side information (`cat(y_hat_luma,
//! y_hat_chroma)`) and produces both chroma planes at *luma* resolution; subsampling to the
//! coded chroma format happens afterwards.

use alloc::format;

use enough::{Stop, Unstoppable};

use crate::error::{Error, Result};
use crate::header::OperatingPoint;
use crate::model::attention::{Cab, ResidualBlock, Tam, conv3x3_t};
use crate::model::load;
use crate::nn::fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `ResAU`: `y = x * (1 + conv1x1(conv3x3_grouped(relu6(x))))`, both convolutions bias-free.
#[derive(Clone, Debug)]
pub(crate) struct ResAu {
    conv: ConvLayer,
    conv2: ConvLayer,
}

impl ResAu {
    pub(crate) fn load(
        ck: &Checkpoint<'_>,
        prefix: &str,
        chs: usize,
        eng: &Engine,
    ) -> Result<Self> {
        // The group count differs between transforms (16 channels per group in SOP / BOP, 4
        // groups in HOP); the checkpoint's weight shape `[chs, chs / groups, 3, 3]` decides.
        let name = format!("{prefix}.conv.weight");
        let per_group = ck
            .info(&name)
            .and_then(|t| t.shape.get(1).copied())
            .filter(|&g| g != 0 && chs.is_multiple_of(g))
            .ok_or_else(|| Error::Model(format!("{name}: missing or malformed")))?;
        let conv = load::conv3x3(
            ck,
            &format!("{prefix}.conv"),
            chs,
            chs,
            chs / per_group,
            false,
        )?;
        let conv2 = load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?;
        Ok(Self {
            conv: ConvLayer::new(conv, eng)?,
            conv2: ConvLayer::new(conv2, eng)?,
        })
    }

    pub(crate) fn forward(&self, eng: &Engine, x: BTensor) -> Result<BTensor> {
        let mut act = x.clone();
        fast::relu6(&mut act);
        let mask = self.conv2.forward(eng, &self.conv.forward(eng, &act)?)?;
        let mut y = x;
        // Two IEEE operations, like the reference's `x * (1 + mask)`; not an FMA.
        fast::gate(&mut y, &mask)?;
        Ok(y)
    }
}

/// 2x upsampling layer: transposed convolution (BOP) or 2x2 convolution + PixelShuffle(2) (SOP).
#[derive(Clone, Debug)]
enum Upsample {
    /// `conv4x4_t`: kernel 4, stride 2, padding 1, with bias.
    Transposed(ConvTransposeLayer),
    /// `conv2x2_pxl2`: pad right/bottom by one zero, 2x2 convolution to 4x channels, shuffle.
    Conv2x2Shuffle(ConvLayer),
}

impl Upsample {
    fn load(
        ck: &Checkpoint<'_>,
        op: OperatingPoint,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        eng: &Engine,
    ) -> Result<Self> {
        match op {
            OperatingPoint::Bop => {
                let c = load::conv_transpose(ck, prefix, in_ch, out_ch, 4, 2, 1, 0)?;
                Ok(Self::Transposed(ConvTransposeLayer::new(c, eng)?))
            }
            OperatingPoint::Sop => {
                let c = load::conv(
                    ck,
                    &format!("{prefix}.conv"),
                    in_ch,
                    out_ch * 4,
                    (2, 2),
                    1,
                    (0, 0),
                    1,
                    false,
                )?;
                // F.pad(x, (0, 1, 0, 1)): one zero column on the right, one zero row at the bottom.
                Ok(Self::Conv2x2Shuffle(ConvLayer::with_extra_pad(
                    c,
                    eng,
                    [0, 0, 1, 1],
                )?))
            }
            OperatingPoint::Hop => Err(Error::InvalidArgument("HOP is not a light transform")),
        }
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        match self {
            Self::Transposed(c) => c.forward(eng, x),
            Self::Conv2x2Shuffle(c) => fast::pixel_shuffle(eng, &c.forward(eng, x)?, 2),
        }
    }
}

/// `x * 128 + 127.5`, clamped to `[0, 255]` (`denormalize` + `clip_image_sgl`).
fn denormalize(x: &mut Tensor<f32>) {
    for v in &mut x.data {
        *v = (*v * 128.0 + 127.5).clamp(0.0, 255.0);
    }
}

/// Primary (luma) synthesis transform.
#[derive(Clone, Debug)]
pub struct SynthesisPrimary(Primary);

#[derive(Clone, Debug)]
enum Primary {
    Light(alloc::boxed::Box<LightPrimary>),
    Hop(alloc::boxed::Box<HopPrimary>),
}

impl SynthesisPrimary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        Ok(Self(match op {
            OperatingPoint::Hop => Primary::Hop(alloc::boxed::Box::new(HopPrimary::load(ck, eng)?)),
            _ => Primary::Light(alloc::boxed::Box::new(LightPrimary::load(ck, op, eng)?)),
        }))
    }

    /// `y_hat` `[160, H, W]` → luma `[1, >= h, >= w]` in `[0, 255]`, where the intermediate maps
    /// are cropped for a target picture (tile) of `h x w` samples. The caller crops to `h x w`.
    pub fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor<f32>> {
        self.forward_with(eng, y_hat, h, w, &Unstoppable)
    }

    /// [`Self::forward`] that checks `stop` between layers.
    pub fn forward_with(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        match &self.0 {
            Primary::Light(m) => m.forward(eng, y_hat, h, w, stop),
            Primary::Hop(m) => m.forward(eng, y_hat, h, w, stop),
        }
    }
}

/// `DecoderHOPPrim`.
#[derive(Clone, Debug)]
struct HopPrimary {
    res: ResidualBlock,
    up1: ConvTransposeLayer,
    act1: ResAu,
    up2: ConvTransposeLayer,
    cab: Cab,
    act2: ResAu,
    conv3: ConvLayer,
    tam: Tam,
    act3: ResAu,
    up4: ConvTransposeLayer,
}

impl HopPrimary {
    const C: usize = 128;

    fn load(ck: &Checkpoint<'_>, eng: &Engine) -> Result<Self> {
        let c = Self::C;
        Ok(Self {
            res: ResidualBlock::load(ck, "first_stage.0.0", 160, eng)?,
            up1: conv3x3_t(ck, "first_stage.1", 160, c, eng)?,
            act1: ResAu::load(ck, "first_stage.2", c, eng)?,
            up2: conv3x3_t(ck, "conv2_t", c, c, eng)?,
            cab: Cab::load(ck, "CAB", c, eng)?,
            act2: ResAu::load(ck, "iact2", c, eng)?,
            conv3: ConvLayer::new(load::conv1x1(ck, "conv3_t", c, 4 * c, true)?, eng)?,
            tam: Tam::load(ck, "TAM", c, true, eng)?,
            act3: ResAu::load(ck, "iact3", c, eng)?,
            up4: conv3x3_t(ck, "conv4_t", c, 1, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let x = self.res.forward(eng, y_hat)?;
        stop.check()?;
        let x = self.act1.forward(eng, self.up1.forward(eng, &x)?)?;
        let x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        let x = self.cab.forward(eng, &self.up2.forward(eng, &x)?)?;
        let x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        let x = self.act2.forward(eng, x)?;
        stop.check()?;
        let x = fast::pixel_shuffle(eng, &self.conv3.forward(eng, &x)?, 2)?;
        stop.check()?;
        let x = self.tam.forward(eng, x)?;
        let x = x.crop_par(eng, h.div_ceil(2), w.div_ceil(2))?;
        stop.check()?;
        let x = self.act3.forward(eng, x)?;
        stop.check()?;
        let mut x = self.up4.forward(eng, &x)?.to_planar()?;
        denormalize(&mut x);
        Ok(x)
    }
}

/// `DecoderSOPPrim` / `DecoderBOPPrim`.
#[derive(Clone, Debug)]
struct LightPrimary {
    res_conv: ConvLayer,
    up1: Upsample,
    act1: ResAu,
    up2: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl LightPrimary {
    fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        // hidden channel counts: (after up1, after up2, after conv3)
        let (c1, c2, c3) = match op {
            OperatingPoint::Sop => (64, 32, 32),
            OperatingPoint::Bop => (64, 64, 96),
            OperatingPoint::Hop => {
                return Err(Error::InvalidArgument("HOP is not a light transform"));
            }
        };
        Ok(Self {
            res_conv: ConvLayer::new(
                load::conv3x3(ck, "first_stage.0.0.conv1", 160, 160, 1, true)?,
                eng,
            )?,
            up1: Upsample::load(ck, op, "first_stage.1", 160, c1, eng)?,
            act1: ResAu::load(ck, "first_stage.2", c1, eng)?,
            up2: Upsample::load(ck, op, "conv2_t", c1, c2, eng)?,
            act2: ResAu::load(ck, "iact2", c2, eng)?,
            conv3: ConvLayer::new(load::conv3x3(ck, "conv3", c2, c3, 1, true)?, eng)?,
            act3: ResAu::load(ck, "iact3", c3, eng)?,
            conv4: ConvLayer::new(load::conv1x1(ck, "conv4", c3, 16, false)?, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        // LightResidualBlock: relu(conv(x)) + x
        let mut x = self.res_conv.forward(eng, y_hat)?;
        fast::relu(&mut x);
        fast::add_assign(&mut x, y_hat)?;
        stop.check()?;
        let x = self.act1.forward(eng, self.up1.forward(eng, &x)?)?;
        let x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        let x = self
            .up2
            .forward(eng, &x)?
            .crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        let x = self.act2.forward(eng, x)?;
        stop.check()?;
        let x = self.conv3.forward(eng, &x)?;
        stop.check()?;
        let x = self.act3.forward(eng, x)?;
        stop.check()?;
        let mut x = fast::pixel_shuffle_to_planar(&self.conv4.forward(eng, &x)?, 4)?;
        denormalize(&mut x);
        Ok(x)
    }
}

/// Secondary (chroma) synthesis transform.
#[derive(Clone, Debug)]
pub struct SynthesisSecondary {
    /// `LightCombineBlock`: conv3x3(160 + 96 → 48); its output is `cat(info, chroma latent)`.
    combine: ConvLayer,
    tail: SecondaryTail,
}

#[derive(Clone, Debug)]
enum SecondaryTail {
    Light(alloc::boxed::Box<LightSecondary>),
    Hop(alloc::boxed::Box<HopSecondary>),
}

/// `DecoderHOPSec` behind the combine block.
#[derive(Clone, Debug)]
struct HopSecondary {
    up2: ConvTransposeLayer,
    cab: Cab,
    act2: ResAu,
    conv3: ConvLayer,
    tam: Tam,
    act3: ResAu,
    up4: ConvTransposeLayer,
}

impl HopSecondary {
    const C: usize = 64;

    fn load(ck: &Checkpoint<'_>, eng: &Engine) -> Result<Self> {
        let c = Self::C;
        Ok(Self {
            up2: conv3x3_t(ck, "conv2_t", 144, c, eng)?,
            cab: Cab::load(ck, "CAB", c, eng)?,
            act2: ResAu::load(ck, "iact2", c, eng)?,
            conv3: ConvLayer::new(load::conv1x1(ck, "conv3_t", c, 4 * c, true)?, eng)?,
            tam: Tam::load(ck, "TAM", c, false, eng)?,
            act3: ResAu::load(ck, "iact3", c, eng)?,
            up4: conv3x3_t(ck, "conv4_t", c, 8, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        x: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let x = self.cab.forward(eng, &self.up2.forward(eng, x)?)?;
        let x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        let x = self.act2.forward(eng, x)?;
        stop.check()?;
        let x = fast::pixel_shuffle(eng, &self.conv3.forward(eng, &x)?, 2)?;
        stop.check()?;
        let x = self.tam.forward(eng, x)?;
        let x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        let x = self.act3.forward(eng, x)?;
        stop.check()?;
        fast::pixel_shuffle_to_planar(&self.up4.forward(eng, &x)?, 2)
    }
}

/// `DecoderSOPSec` / `DecoderBOPSec` behind the combine block.
#[derive(Clone, Debug)]
struct LightSecondary {
    up: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl LightSecondary {
    fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        let (c2, c3) = match op {
            OperatingPoint::Sop => (32, 32),
            OperatingPoint::Bop => (64, 128),
            OperatingPoint::Hop => {
                return Err(Error::InvalidArgument("HOP is not a light transform"));
            }
        };
        Ok(Self {
            up: Upsample::load(ck, op, "conv2_t", 144, c2, eng)?,
            act2: ResAu::load(ck, "iact2", c2, eng)?,
            conv3: ConvLayer::new(load::conv3x3(ck, "conv3", c2, c3, 1, true)?, eng)?,
            act3: ResAu::load(ck, "iact3", c3, eng)?,
            conv4: ConvLayer::new(load::conv1x1(ck, "conv4", c3, 128, false)?, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        x: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let x = self
            .up
            .forward(eng, x)?
            .crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        let x = self.act2.forward(eng, x)?;
        stop.check()?;
        let x = self.conv3.forward(eng, &x)?;
        stop.check()?;
        let x = self.act3.forward(eng, x)?;
        stop.check()?;
        fast::pixel_shuffle_to_planar(&self.conv4.forward(eng, &x)?, 8)
    }
}

impl SynthesisSecondary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        Ok(Self {
            combine: ConvLayer::new(
                load::conv3x3(ck, "first_stage.conv1", 256, 48, 1, true)?,
                eng,
            )?,
            tail: match op {
                OperatingPoint::Hop => {
                    SecondaryTail::Hop(alloc::boxed::Box::new(HopSecondary::load(ck, eng)?))
                }
                _ => {
                    SecondaryTail::Light(alloc::boxed::Box::new(LightSecondary::load(ck, op, eng)?))
                }
            },
        })
    }

    /// `y_hat_luma` `[160, H, W]`, `y_hat_chroma` `[96, H', W']` → both chroma planes
    /// `[2, 8 * ceil(h/8), ..]` at luma resolution, for a target picture (tile) of `h x w` luma
    /// samples. The two latents are cropped to their common size first.
    pub fn forward(
        &self,
        eng: &Engine,
        y_hat_luma: &BTensor,
        y_hat_chroma: &BTensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor<f32>> {
        self.forward_with(eng, y_hat_luma, y_hat_chroma, h, w, &Unstoppable)
    }

    /// [`Self::forward`] that checks `stop` between layers.
    pub fn forward_with(
        &self,
        eng: &Engine,
        y_hat_luma: &BTensor,
        y_hat_chroma: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let (lh, lw) = (
            y_hat_luma.h.min(y_hat_chroma.h),
            y_hat_luma.w.min(y_hat_chroma.w),
        );
        if y_hat_chroma.h != lh || y_hat_chroma.w != lw {
            return Err(Error::InvalidArgument(
                "chroma latent larger than the luma latent",
            ));
        }
        let x = BTensor::cat(&[&y_hat_luma.crop(lh, lw)?, y_hat_chroma])?;
        let info = self.combine.forward(eng, &x)?;
        stop.check()?;
        let x = BTensor::cat(&[&info, y_hat_chroma])?;
        let mut x = match &self.tail {
            SecondaryTail::Light(m) => m.forward(eng, &x, h, w, stop)?,
            SecondaryTail::Hop(m) => m.forward(eng, &x, h, w, stop)?,
        };
        denormalize(&mut x);
        Ok(x)
    }
}
