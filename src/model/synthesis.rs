//! Synthesis transforms: latent `y_hat` → reconstructed samples.
//!
//! Ports `ref/src/codec/components/autoencoder_data/decoder/{sop,bop}_{prim,sec}.py`,
//! `activations/resau.py` and the residual blocks of `base_layers/conv_layers.py`.
//!
//! The primary (luma) transform maps `[160, H, W]` to one plane at 16x the resolution. The
//! secondary (chroma) transform takes the luma latent as side information (`cat(y_hat_luma,
//! y_hat_chroma)`) and produces both chroma planes at *luma* resolution; subsampling to the
//! coded chroma format happens afterwards.
//!
//! Not ported yet: the HOP transforms (`hop_prim.py`, `hop_sec.py`: CAB + TAM attention).

use alloc::format;

use crate::error::{Error, Result};
use crate::header::OperatingPoint;
use crate::model::load;
use crate::nn::fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `ResAU`: `y = x * (1 + conv1x1(conv3x3_grouped(relu6(x))))`, both convolutions bias-free,
/// 16 channels per group.
#[derive(Clone, Debug)]
struct ResAu {
    conv: ConvLayer,
    conv2: ConvLayer,
}

impl ResAu {
    fn load(ck: &Checkpoint<'_>, prefix: &str, chs: usize, eng: &Engine) -> Result<Self> {
        if !chs.is_multiple_of(16) {
            return Err(Error::Model(format!(
                "{prefix}: ResAU needs a multiple of 16 channels"
            )));
        }
        let conv = load::conv3x3(ck, &format!("{prefix}.conv"), chs, chs, chs / 16, false)?;
        let conv2 = load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?;
        Ok(Self {
            conv: ConvLayer::new(conv, eng)?,
            conv2: ConvLayer::new(conv2, eng)?,
        })
    }

    fn forward(&self, eng: &Engine, x: BTensor) -> Result<BTensor> {
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
            OperatingPoint::Hop => Err(Error::Unsupported("HOP synthesis transform")),
        }
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        match self {
            Self::Transposed(c) => c.forward(eng, x),
            Self::Conv2x2Shuffle(c) => fast::pixel_shuffle(&c.forward(eng, x)?, 2),
        }
    }
}

/// `x * 128 + 127.5`, clamped to `[0, 255]` (`denormalize` + `clip_image_sgl`).
fn denormalize(x: &mut Tensor<f32>) {
    for v in &mut x.data {
        *v = (*v * 128.0 + 127.5).clamp(0.0, 255.0);
    }
}

/// Primary (luma) synthesis transform, SOP or BOP.
#[derive(Clone, Debug)]
pub struct SynthesisPrimary {
    res_conv: ConvLayer,
    up1: Upsample,
    act1: ResAu,
    up2: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl SynthesisPrimary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        // hidden channel counts: (after up1, after up2, after conv3)
        let (c1, c2, c3) = match op {
            OperatingPoint::Sop => (64, 32, 32),
            OperatingPoint::Bop => (64, 64, 96),
            OperatingPoint::Hop => return Err(Error::Unsupported("HOP synthesis transform")),
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

    /// `y_hat` `[160, H, W]` → luma `[1, 16H', 16W']` in `[0, 255]`, where the intermediate maps
    /// are cropped for a target picture (tile) of `h x w` samples. The caller crops to `h x w`.
    pub fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor<f32>> {
        // LightResidualBlock: relu(conv(x)) + x
        let mut x = self.res_conv.forward(eng, y_hat)?;
        fast::relu(&mut x);
        fast::add_assign(&mut x, y_hat)?;
        let x = self.act1.forward(eng, self.up1.forward(eng, &x)?)?;
        let x = x.crop(h.div_ceil(8), w.div_ceil(8))?;
        let x = self
            .up2
            .forward(eng, &x)?
            .crop(h.div_ceil(4), w.div_ceil(4))?;
        let x = self.act2.forward(eng, x)?;
        let x = self.act3.forward(eng, self.conv3.forward(eng, &x)?)?;
        let mut x = fast::pixel_shuffle_to_planar(&self.conv4.forward(eng, &x)?, 4)?;
        denormalize(&mut x);
        Ok(x)
    }
}

/// Secondary (chroma) synthesis transform, SOP or BOP.
#[derive(Clone, Debug)]
pub struct SynthesisSecondary {
    combine: ConvLayer,
    up: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl SynthesisSecondary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        let (c2, c3) = match op {
            OperatingPoint::Sop => (32, 32),
            OperatingPoint::Bop => (64, 128),
            OperatingPoint::Hop => return Err(Error::Unsupported("HOP synthesis transform")),
        };
        Ok(Self {
            // LightCombineBlock: conv3x3(160 + 96 → 48), output cat(info, chroma latent) = 144 ch
            combine: ConvLayer::new(
                load::conv3x3(ck, "first_stage.conv1", 256, 48, 1, true)?,
                eng,
            )?,
            up: Upsample::load(ck, op, "conv2_t", 144, c2, eng)?,
            act2: ResAu::load(ck, "iact2", c2, eng)?,
            conv3: ConvLayer::new(load::conv3x3(ck, "conv3", c2, c3, 1, true)?, eng)?,
            act3: ResAu::load(ck, "iact3", c3, eng)?,
            conv4: ConvLayer::new(load::conv1x1(ck, "conv4", c3, 128, false)?, eng)?,
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
        let x = BTensor::cat(&[&info, y_hat_chroma])?;
        let x = self
            .up
            .forward(eng, &x)?
            .crop(h.div_ceil(8), w.div_ceil(8))?;
        let x = self.act2.forward(eng, x)?;
        let x = self.act3.forward(eng, self.conv3.forward(eng, &x)?)?;
        let mut x = fast::pixel_shuffle_to_planar(&self.conv4.forward(eng, &x)?, 8)?;
        denormalize(&mut x);
        Ok(x)
    }
}
