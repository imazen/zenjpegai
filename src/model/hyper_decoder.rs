//! Hyper-decoder: `z_hat` → `psi`, the conditioning tensor of the context model.
//!
//! Ports `ref/src/codec/components/autoencoder_hyper/decoder/base.py` (`HyperDecoderBase`):
//!
//! ```text
//! conv1x1 → convT4x4 (x2) → crop → ReLU6 → conv3x3 → ReLU6 → conv3x3 (C → 4C)
//! ```

use crate::error::Result;
use crate::model::load;
use crate::nn::fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

#[derive(Clone, Debug)]
pub struct HyperDecoder {
    conv1: ConvLayer,
    conv2: ConvTransposeLayer,
    conv3: ConvLayer,
    conv4: ConvLayer,
}

impl HyperDecoder {
    pub fn load(ck: &Checkpoint<'_>, chs: usize, eng: &Engine) -> Result<Self> {
        Ok(Self {
            conv1: ConvLayer::new(
                load::conv1x1(ck, "hyper_decoder.conv1", chs, chs, false)?,
                eng,
            )?,
            conv2: ConvTransposeLayer::new(
                load::conv_transpose(ck, "hyper_decoder.conv2", chs, chs, 4, 2, 1, 0)?,
                eng,
            )?,
            conv3: ConvLayer::new(
                load::conv3x3(ck, "hyper_decoder.conv3", chs, chs, 1, true)?,
                eng,
            )?,
            conv4: ConvLayer::new(
                load::conv3x3(ck, "hyper_decoder.conv4", chs, 4 * chs, 1, true)?,
                eng,
            )?,
        })
    }

    /// `z_hat` `[C, hz, wz]` → `psi` `[4C, out_h, out_w]`; the 2x upsampled map is cropped to
    /// `(out_h, out_w)` (the reference's `cropping_layer(.., depth=5)`).
    pub fn forward(
        &self,
        eng: &Engine,
        z_hat: &Tensor<i8>,
        out_h: usize,
        out_w: usize,
    ) -> Result<BTensor> {
        let x = Tensor::from_vec(
            z_hat.c,
            z_hat.h,
            z_hat.w,
            z_hat.data.iter().map(|&v| v as f32).collect(),
        )?;
        let x = BTensor::from_planar(&x, eng.tier.block())?;
        let x = self.conv1.forward(eng, &x)?;
        let mut x = self.conv2.forward(eng, &x)?.crop(out_h, out_w)?;
        fast::relu6(&mut x);
        let mut x = self.conv3.forward(eng, &x)?;
        fast::relu6(&mut x);
        self.conv4.forward(eng, &x)
    }
}
