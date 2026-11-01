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
        self.forward_with(eng, z_hat, out_h, out_w, &enough::Unstoppable)
    }

    /// [`Self::forward`] that checks `stop` between layers.
    pub fn forward_with(
        &self,
        eng: &Engine,
        z_hat: &Tensor<i8>,
        out_h: usize,
        out_w: usize,
        stop: &dyn enough::Stop,
    ) -> Result<BTensor> {
        let x = Tensor::from_vec(
            z_hat.c,
            z_hat.h,
            z_hat.w,
            z_hat.data.iter().map(|&v| v as f32).collect(),
        )?;
        // `x = f(&x)` frees each map as soon as the next exists.
        let mut x = BTensor::from_planar(&x, eng.tier.block())?;
        x = self.conv1.forward(eng, &x)?;
        stop.check()?;
        x = self.conv2.forward(eng, &x)?;
        if (x.h, x.w) != (out_h, out_w) {
            x = x.crop(out_h, out_w)?;
        }
        fast::relu6(&mut x);
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        fast::relu6(&mut x);
        stop.check()?;
        self.conv4.forward(eng, &x)
    }

    /// [`Self::forward_with`] on a real-valued `z_hat` — `process_z`'s
    /// STE-quantized (but continuous-domain) latent; identical output to
    /// [`Self::forward`] on the same integer values, differentiable in `z_hat`.
    pub fn forward_f32(
        &self,
        eng: &Engine,
        z_hat: &Tensor<f32>,
        out_h: usize,
        out_w: usize,
        stop: &dyn enough::Stop,
    ) -> Result<BTensor> {
        let mut x = BTensor::from_planar(z_hat, eng.tier.block())?;
        x = self.conv1.forward(eng, &x)?;
        stop.check()?;
        x = self.conv2.forward(eng, &x)?;
        if (x.h, x.w) != (out_h, out_w) {
            x = x.crop(out_h, out_w)?;
        }
        fast::relu6(&mut x);
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        fast::relu6(&mut x);
        stop.check()?;
        self.conv4.forward(eng, &x)
    }

    /// [`Self::forward_f32`] that records the intermediates its VJP needs
    /// (RDLR `process_z`): each pre-`relu6` blocked tensor and every conv's
    /// input dims.
    pub fn forward_tape(
        &self,
        eng: &Engine,
        z_hat: &Tensor<f32>,
        out_h: usize,
        out_w: usize,
        stop: &dyn enough::Stop,
    ) -> Result<(BTensor, HyperTape)> {
        let mut x = BTensor::from_planar(z_hat, eng.tier.block())?;
        let x1 = (x.h, x.w);
        x = self.conv1.forward(eng, &x)?;
        stop.check()?;
        let x2 = (x.h, x.w);
        x = self.conv2.forward(eng, &x)?;
        if (x.h, x.w) != (out_h, out_w) {
            x = x.crop(out_h, out_w)?;
        }
        stop.check()?;
        let c2_pre = x.clone();
        fast::relu6(&mut x);
        let x3 = (x.h, x.w);
        x = self.conv3.forward(eng, &x)?;
        let c3_pre = x.clone();
        fast::relu6(&mut x);
        let x = self.conv4.forward(eng, &x)?;
        Ok((
            x,
            HyperTape {
                c2_pre,
                c3_pre,
                x1,
                x2,
                x3,
            },
        ))
    }

    /// `d(psi)` (planar `[4C, out_h, out_w]`) → `d(z_hat)` (planar
    /// `[C, hz, wz]`): the reverse of `forward` — conv4 → `relu6'` → conv3 →
    /// `relu6'` → pad → conv2 → conv1.
    pub fn backward(&self, tape: &HyperTape, d: &Tensor<f32>) -> Result<Tensor<f32>> {
        let d = self.conv4.backward(d, tape.x3.0, tape.x3.1)?;
        // relu6 backward on conv3's output.
        let mut d = d;
        let c3p = tape.c3_pre.to_planar()?;
        for (o, &p) in d.data.iter_mut().zip(&c3p.data) {
            if !(p > 0.0 && p < 6.0) {
                *o = 0.0;
            }
        }
        let d = self.conv3.backward(&d, tape.x3.0, tape.x3.1)?;
        // `crop(out_h, out_w)` backward: pad to conv2's output dims.
        let c2p = tape.c2_pre.to_planar()?;
        let mut d = zero_pad(&d, c2p.h, c2p.w)?;
        // relu6 backward on conv2's (cropped) output — applied to the padded map's
        // in-bounds part (zeros outside, where relu6's input did not exist).
        for (o, &p) in d.data.iter_mut().zip(&c2p.data) {
            if !(p > 0.0 && p < 6.0) {
                *o = 0.0;
            }
        }
        let d = self.conv2.backward(&d, tape.x2.0, tape.x2.1)?;
        self.conv1.backward(&d, tape.x1.0, tape.x1.1)
    }
}

/// `crop(out_h, out_w)`'s backward — pad `d` out to `(h, w)` with zeros (the
/// cropped-away positions contributed nothing).
fn zero_pad(d: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
    let mut out = Tensor::<f32>::zeros(d.c, h, w)?;
    let (dh, dw) = (d.h.min(h), d.w.min(w));
    for c in 0..d.c {
        for y in 0..dh {
            out.data[(c * h + y) * w..(c * h + y) * w + dw]
                .copy_from_slice(&d.data[(c * d.h + y) * d.w..(c * d.h + y) * d.w + dw]);
        }
    }
    Ok(out)
}

/// `HyperDecoder::backward`'s forward intermediates.
pub struct HyperTape {
    /// `conv2`'s (cropped) output before `relu6` — the relu6 gate.
    c2_pre: BTensor,
    /// `conv3`'s output before `relu6` — the relu6 gate.
    c3_pre: BTensor,
    /// The `z_hat`/`conv1` input dims.
    x1: (usize, usize),
    /// `conv1`'s output dims (`conv2` input).
    x2: (usize, usize),
    /// `conv3`'s input dims (post-crop + relu6).
    x3: (usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_models() -> Option<crate::model::ModelDir> {
        let root = std::env::var_os("ZENJPEGAI_REF")?;
        let dir = std::path::PathBuf::from(root).join("models");
        dir.is_dir().then(|| crate::model::ModelDir::new(dir))
    }

    fn filled(c: usize, h: usize, w: usize, scale: f32, seed: u32) -> Tensor<f32> {
        let mut data = Vec::with_capacity(c * h * w);
        for i in 0..c * h * w {
            let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) & 0xffff;
            data.push(scale * ((v as f32 / 65536.0) - 0.5));
        }
        Tensor::from_vec(c, h, w, data).unwrap()
    }

    /// `HyperDecoder::backward`'s `d(z_hat)` vs a central finite difference of
    /// `forward` — the `process_z` `z_hat→psi` VJP.
    #[test]
    fn hyper_decoder_backward_finite_diff() {
        let Some(models) = ref_models() else { return };
        let eng = Engine::new();
        let stop = enough::Unstoppable;
        let (hz, wz) = (8usize, 8usize);
        let (oh, ow) = (16usize, 16usize); // psi is at 2x the z size (conv2 x2)
        for ccs in [0usize, 1] {
            let chs = if ccs == 0 { 160usize } else { 96usize };
            let z_hat = filled(chs, hz, wz, 8.0, ccs as u32 + 5);
            let Ok(hd) = crate::model::load_hyper_decoder(&models, 1, ccs, &eng) else {
                continue;
            };
            let (_psi, tape) = hd.forward_tape(&eng, &z_hat, oh, ow, &stop).unwrap();
            let d_psi = filled(4 * chs, oh, ow, 0.5, 41);
            let d_z = hd.backward(&tape, &d_psi).unwrap();
            let loss = |z2: &Tensor<f32>| {
                let p = hd
                    .forward_f32(&eng, z2, oh, ow, &stop)
                    .unwrap()
                    .to_planar()
                    .unwrap();
                p.data
                    .iter()
                    .zip(&d_psi.data)
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum::<f64>()
            };
            let eps = 1e-2f32;
            for &(ch, yy, xx) in &[(0usize, 3usize, 4usize), (chs / 2, 5, 2), (chs - 1, 7, 6)] {
                let mut zp = z_hat.clone();
                let mut zm = z_hat.clone();
                zp.data[(ch * hz + yy) * wz + xx] += eps;
                zm.data[(ch * hz + yy) * wz + xx] -= eps;
                let numeric = (loss(&zp) - loss(&zm)) / (2.0 * eps) as f64;
                let got = d_z.data[(ch * hz + yy) * wz + xx] as f64;
                assert!(
                    (got - numeric).abs() <= 0.02 * (1.0 + numeric.abs()),
                    "ccs{ccs} ch{ch} y{yy} x{xx}: backward {got} vs numeric {numeric}"
                );
            }
        }
    }
}
