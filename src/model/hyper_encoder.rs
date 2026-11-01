//! Hyper-encoder: latent `y` → hyper-latent `z` (before clamping and rounding).
//!
//! Ports `ref/src/codec/components/autoencoder_hyper/encoder/basic.py` (`HyperEncoderBasic`,
//! `abs_in_hyperprior = 1`):
//!
//! ```text
//! |y| → conv3x3 → LeakyReLU → conv3x3 → pad → conv3x3/2 → LeakyReLU → conv3x3 → pad → conv3x3/2
//! ```
//!
//! The weights live next to the decoder-side modules in `models/VM_common_int/*.pth` under
//! `hyper_encoder.*`.

use enough::Stop;

use crate::error::{Error, Result};
use crate::model::analysis::{pad_amount, replicate_pad};
use crate::model::load;
use crate::nn::fast::{BTensor, ConvLayer, Engine};
use crate::weights::Checkpoint;

/// `nn.LeakyReLU()` default slope.
const NEGATIVE_SLOPE: f32 = 0.01;

fn leaky_relu(x: &mut BTensor) {
    for v in &mut x.data {
        if *v < 0.0 {
            *v *= NEGATIVE_SLOPE;
        }
    }
}

#[derive(Clone, Debug)]
pub struct HyperEncoder {
    conv1: ConvLayer,
    conv2: ConvLayer,
    conv3: ConvLayer,
    conv4: ConvLayer,
    conv5: ConvLayer,
    /// `clip_thres` of `conv1..conv5` (`HE1`..`HE5`), when the checkpoint carries it.
    clip: [Option<(Vec<f32>, Vec<f32>)>; 5],
}

impl HyperEncoder {
    pub fn load(ck: &Checkpoint<'_>, chs: usize, eng: &Engine) -> Result<Self> {
        let conv = |name: &str, stride: usize| -> Result<ConvLayer> {
            ConvLayer::new(
                load::conv(
                    ck,
                    &alloc::format!("hyper_encoder.{name}"),
                    chs,
                    chs,
                    (3, 3),
                    stride,
                    (1, 1),
                    1,
                    true,
                )?,
                eng,
            )
        };
        let clip = |stage: usize| {
            load::clip_bounds(ck, &alloc::format!("hyper_encoder.clip_thres.HE{stage}"))
        };
        Ok(Self {
            conv1: conv("conv1", 1)?,
            conv2: conv("conv2", 1)?,
            conv3: conv("conv3", 2)?,
            conv4: conv("conv4", 1)?,
            conv5: conv("conv5", 2)?,
            clip: [clip(1), clip(2), clip(3), clip(4), clip(5)],
        })
    }

    /// `y` is the latent of a picture (tile) whose component plane is `h x w` samples;
    /// `latent_divider` is 16 for luma and 8 for chroma (whose plane has half the luma size).
    /// Returns the unrounded `z`, `[C, ceil(h / (4 d)), ceil(w / (4 d))]`.
    pub fn forward(
        &self,
        eng: &Engine,
        y: &BTensor,
        h: usize,
        w: usize,
        latent_divider: usize,
        stop: &dyn Stop,
    ) -> Result<BTensor> {
        self.forward_clip(eng, y, h, w, latent_divider, false, stop)
    }

    /// `forward` with the checkpoint's `clip_thres` applied after each conv when `clip`
    /// (`clipping_mode = 1` — `autoencoder_hyper/encoder/basic.py::forward` under `is_clip`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_clip(
        &self,
        eng: &Engine,
        y: &BTensor,
        h: usize,
        w: usize,
        latent_divider: usize,
        clip: bool,
        stop: &dyn Stop,
    ) -> Result<BTensor> {
        let d = latent_divider;
        if y.h != h.div_ceil(d) || y.w != w.div_ceil(d) {
            return Err(Error::InvalidArgument("hyper-encoder: latent size"));
        }
        let mut x = y.clone();
        for v in &mut x.data {
            *v = v.abs();
        }
        let mut x = self.conv1.forward(eng, &x)?;
        if clip && let Some(b) = &self.clip[0] {
            load::clip_channels(&mut x, b);
        }
        leaky_relu(&mut x);
        stop.check()?;
        let mut x = self.conv2.forward(eng, &x)?;
        if clip && let Some(b) = &self.clip[1] {
            load::clip_channels(&mut x, b);
        }
        let x = replicate_pad(x, pad_amount(h, d), pad_amount(w, d))?;
        let mut x = self.conv3.forward(eng, &x)?;
        if clip && let Some(b) = &self.clip[2] {
            load::clip_channels(&mut x, b);
        }
        leaky_relu(&mut x);
        stop.check()?;
        let mut x = self.conv4.forward(eng, &x)?;
        if clip && let Some(b) = &self.clip[3] {
            load::clip_channels(&mut x, b);
        }
        let x = replicate_pad(x, pad_amount(h, 2 * d), pad_amount(w, 2 * d))?;
        let mut x = self.conv5.forward(eng, &x)?;
        if clip && let Some(b) = &self.clip[4] {
            load::clip_channels(&mut x, b);
        }
        Ok(x)
    }
}
