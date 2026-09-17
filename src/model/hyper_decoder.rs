//! Hyper-decoder: `z_hat` → `psi`, the conditioning tensor of the context model.
//!
//! Ports `ref/src/codec/components/autoencoder_hyper/decoder/base.py` (`HyperDecoderBase`):
//!
//! ```text
//! conv1x1 → convT4x4 (x2) → crop → ReLU6 → conv3x3 → ReLU6 → conv3x3 (C → 4C)
//! ```

use crate::error::Result;
use crate::model::load;
use crate::nn::{Conv2d, ConvTranspose2d, reference as ops};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

#[derive(Clone, Debug)]
pub struct HyperDecoder {
    conv1: Conv2d,
    conv2: ConvTranspose2d,
    conv3: Conv2d,
    conv4: Conv2d,
}

impl HyperDecoder {
    pub fn load(ck: &Checkpoint<'_>, chs: usize) -> Result<Self> {
        Ok(Self {
            conv1: load::conv1x1(ck, "hyper_decoder.conv1", chs, chs, false)?,
            conv2: load::conv_transpose(ck, "hyper_decoder.conv2", chs, chs, 4, 2, 1, 0)?,
            conv3: load::conv3x3(ck, "hyper_decoder.conv3", chs, chs, 1, true)?,
            conv4: load::conv3x3(ck, "hyper_decoder.conv4", chs, 4 * chs, 1, true)?,
        })
    }

    /// `z_hat` `[C, hz, wz]` → `psi` `[4C, out_h, out_w]`; the 2x upsampled map is cropped to
    /// `(out_h, out_w)` (the reference's `cropping_layer(.., depth=5)`).
    pub fn forward(&self, z_hat: &Tensor<i8>, out_h: usize, out_w: usize) -> Result<Tensor<f32>> {
        let x = Tensor::from_vec(
            z_hat.c,
            z_hat.h,
            z_hat.w,
            z_hat.data.iter().map(|&v| v as f32).collect(),
        )?;
        let x = ops::conv2d(&self.conv1, &x)?;
        let x = ops::conv_transpose2d(&self.conv2, &x)?;
        let mut x = x.crop(out_h, out_w)?;
        ops::relu6(&mut x);
        let mut x = ops::conv2d(&self.conv3, &x)?;
        ops::relu6(&mut x);
        ops::conv2d(&self.conv4, &x)
    }
}
