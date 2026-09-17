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
use crate::nn::{Conv2d, ConvTranspose2d, reference as ops};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `ResAU`: `y = x * (1 + conv1x1(conv3x3_grouped(relu6(x))))`, both convolutions bias-free,
/// 16 channels per group.
#[derive(Clone, Debug)]
struct ResAu {
    conv: Conv2d,
    conv2: Conv2d,
}

impl ResAu {
    fn load(ck: &Checkpoint<'_>, prefix: &str, chs: usize) -> Result<Self> {
        if !chs.is_multiple_of(16) {
            return Err(Error::Model(format!(
                "{prefix}: ResAU needs a multiple of 16 channels"
            )));
        }
        Ok(Self {
            conv: load::conv3x3(ck, &format!("{prefix}.conv"), chs, chs, chs / 16, false)?,
            conv2: load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?,
        })
    }

    fn forward(&self, x: Tensor<f32>) -> Result<Tensor<f32>> {
        let mut act = x.clone();
        ops::relu6(&mut act);
        let mask = ops::conv2d(&self.conv2, &ops::conv2d(&self.conv, &act)?)?;
        let mut y = x;
        for (v, &m) in y.data.iter_mut().zip(&mask.data) {
            // Two IEEE operations, like the reference's `x * (1 + mask)`; not an FMA.
            *v *= 1.0 + m;
        }
        Ok(y)
    }
}

/// 2x upsampling layer: transposed convolution (BOP) or 2x2 convolution + PixelShuffle(2) (SOP).
#[derive(Clone, Debug)]
enum Upsample {
    /// `conv4x4_t`: kernel 4, stride 2, padding 1, with bias.
    Transposed(ConvTranspose2d),
    /// `conv2x2_pxl2`: pad right/bottom by one zero, 2x2 convolution to 4x channels, shuffle.
    Conv2x2Shuffle(Conv2d),
}

impl Upsample {
    fn load(
        ck: &Checkpoint<'_>,
        op: OperatingPoint,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
    ) -> Result<Self> {
        match op {
            OperatingPoint::Bop => Ok(Self::Transposed(load::conv_transpose(
                ck, prefix, in_ch, out_ch, 4, 2, 1, 0,
            )?)),
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
                Ok(Self::Conv2x2Shuffle(c))
            }
            OperatingPoint::Hop => Err(Error::Unsupported("HOP synthesis transform")),
        }
    }

    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        match self {
            Self::Transposed(c) => ops::conv_transpose2d(c, x),
            Self::Conv2x2Shuffle(c) => {
                // F.pad(x, (0, 1, 0, 1)): one zero column on the right, one zero row at the bottom.
                let mut padded = Tensor::<f32>::zeros(x.c, x.h + 1, x.w + 1)?;
                for ch in 0..x.c {
                    for y in 0..x.h {
                        let src = &x.data[(ch * x.h + y) * x.w..][..x.w];
                        padded.data[(ch * (x.h + 1) + y) * (x.w + 1)..][..x.w].copy_from_slice(src);
                    }
                }
                ops::pixel_shuffle(&ops::conv2d(c, &padded)?, 2)
            }
        }
    }
}

/// `x * 128 + 127.5`, clamped to `[0, 255]` (`denormalize` + `clip_image_sgl`).
fn denormalize(x: &mut Tensor<f32>) {
    for v in &mut x.data {
        *v = (*v * 128.0 + 127.5).clamp(0.0, 255.0);
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Primary (luma) synthesis transform, SOP or BOP.
#[derive(Clone, Debug)]
pub struct SynthesisPrimary {
    res_conv: Conv2d,
    up1: Upsample,
    act1: ResAu,
    up2: Upsample,
    act2: ResAu,
    conv3: Conv2d,
    act3: ResAu,
    conv4: Conv2d,
}

impl SynthesisPrimary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint) -> Result<Self> {
        // hidden channel counts: (after up1, after up2, after conv3)
        let (c1, c2, c3) = match op {
            OperatingPoint::Sop => (64, 32, 32),
            OperatingPoint::Bop => (64, 64, 96),
            OperatingPoint::Hop => return Err(Error::Unsupported("HOP synthesis transform")),
        };
        Ok(Self {
            res_conv: load::conv3x3(ck, "first_stage.0.0.conv1", 160, 160, 1, true)?,
            up1: Upsample::load(ck, op, "first_stage.1", 160, c1)?,
            act1: ResAu::load(ck, "first_stage.2", c1)?,
            up2: Upsample::load(ck, op, "conv2_t", c1, c2)?,
            act2: ResAu::load(ck, "iact2", c2)?,
            conv3: load::conv3x3(ck, "conv3", c2, c3, 1, true)?,
            act3: ResAu::load(ck, "iact3", c3)?,
            conv4: load::conv1x1(ck, "conv4", c3, 16, false)?,
        })
    }

    /// `y_hat` `[160, H, W]` → luma `[1, 16H', 16W']` in `[0, 255]`, where the intermediate maps
    /// are cropped for a target picture (tile) of `h x w` samples. The caller crops to `h x w`.
    pub fn forward(&self, y_hat: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
        // LightResidualBlock: relu(conv(x)) + x
        let mut x = ops::conv2d(&self.res_conv, y_hat)?;
        ops::relu(&mut x);
        for (v, &s) in x.data.iter_mut().zip(&y_hat.data) {
            *v += s;
        }
        let x = self.act1.forward(self.up1.forward(&x)?)?;
        let x = x.crop(ceil_div(h, 8), ceil_div(w, 8))?;
        let x = self.up2.forward(&x)?.crop(ceil_div(h, 4), ceil_div(w, 4))?;
        let x = self.act2.forward(x)?;
        let x = self.act3.forward(ops::conv2d(&self.conv3, &x)?)?;
        let mut x = ops::pixel_shuffle(&ops::conv2d(&self.conv4, &x)?, 4)?;
        denormalize(&mut x);
        Ok(x)
    }
}

/// Secondary (chroma) synthesis transform, SOP or BOP.
#[derive(Clone, Debug)]
pub struct SynthesisSecondary {
    combine: Conv2d,
    up: Upsample,
    act2: ResAu,
    conv3: Conv2d,
    act3: ResAu,
    conv4: Conv2d,
}

impl SynthesisSecondary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint) -> Result<Self> {
        let (c2, c3) = match op {
            OperatingPoint::Sop => (32, 32),
            OperatingPoint::Bop => (64, 128),
            OperatingPoint::Hop => return Err(Error::Unsupported("HOP synthesis transform")),
        };
        Ok(Self {
            // LightCombineBlock: conv3x3(160 + 96 → 48), output cat(info, chroma latent) = 144 ch
            combine: load::conv3x3(ck, "first_stage.conv1", 256, 48, 1, true)?,
            up: Upsample::load(ck, op, "conv2_t", 144, c2)?,
            act2: ResAu::load(ck, "iact2", c2)?,
            conv3: load::conv3x3(ck, "conv3", c2, c3, 1, true)?,
            act3: ResAu::load(ck, "iact3", c3)?,
            conv4: load::conv1x1(ck, "conv4", c3, 128, false)?,
        })
    }

    /// `y_hat_luma` `[160, H, W]`, `y_hat_chroma` `[96, H', W']` → both chroma planes
    /// `[2, 8 * ceil(h/8), ..]` at luma resolution, for a target picture (tile) of `h x w` luma
    /// samples. The two latents are cropped to their common size first.
    pub fn forward(
        &self,
        y_hat_luma: &Tensor<f32>,
        y_hat_chroma: &Tensor<f32>,
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
        let x = ops::cat(&[&y_hat_luma.crop(lh, lw)?, y_hat_chroma])?;
        let info = ops::conv2d(&self.combine, &x)?;
        let x = ops::cat(&[&info, y_hat_chroma])?;
        let x = self.up.forward(&x)?.crop(ceil_div(h, 8), ceil_div(w, 8))?;
        let x = self.act2.forward(x)?;
        let x = self.act3.forward(ops::conv2d(&self.conv3, &x)?)?;
        let mut x = ops::pixel_shuffle(&ops::conv2d(&self.conv4, &x)?, 8)?;
        denormalize(&mut x);
        Ok(x)
    }
}
