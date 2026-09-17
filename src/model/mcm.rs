//! Multi-stage context model (MCM): reconstructs the latent `y_hat` from the dequantised
//! residual and `psi` in four passes, one per position of a 2x2 grid.
//!
//! Ports `ref/src/codec/components/contexts/` (`context.py::Context.decompress/pred`,
//! `MCM_phases.py`, `fusion_pred_net.py`, `utils.py::ContextUtils`).
//!
//! Stage order over the 2x2 grid is (0,0), (1,1), (0,1), (1,0) as `(row, column)`. Stage `s`
//! predicts a mean from the `s`-th quarter of `psi` and from every stage reconstructed so far;
//! the stage's latent is `residual + mean`.

use alloc::format;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::model::load;
use crate::nn::{Conv2d, reference as ops};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `(row, column)` of each stage inside a 2x2 cell.
pub const STAGE_POSITIONS: [(usize, usize); 4] = [(0, 0), (1, 1), (0, 1), (1, 0)];

/// `FusionPredNet`: three bias-free 1x1 convolutions with ReLU in between.
#[derive(Clone, Debug)]
struct FusionPredNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
}

impl FusionPredNet {
    fn load(ck: &Checkpoint<'_>, prefix: &str, in_ch: usize, chs: usize) -> Result<Self> {
        Ok(Self {
            conv1: load::conv1x1(ck, &format!("{prefix}.conv1"), in_ch, chs, false)?,
            conv2: load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?,
            conv3: load::conv1x1(ck, &format!("{prefix}.conv3"), chs, chs, false)?,
        })
    }

    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let mut x = ops::conv2d(&self.conv1, x)?;
        ops::relu(&mut x);
        let mut x = ops::conv2d(&self.conv2, &x)?;
        ops::relu(&mut x);
        ops::conv2d(&self.conv3, &x)
    }
}

/// One MCM phase. Phase 0 has no spatial context.
#[derive(Clone, Debug)]
struct Phase {
    /// `conv.0` (1x1, `s * C → C`, no bias) and `conv.1` (3x3, groups `C / 32`, bias).
    context: Option<(Conv2d, Conv2d)>,
    fusion: FusionPredNet,
}

#[derive(Clone, Debug)]
pub struct ContextModel {
    chs: usize,
    phases: Vec<Phase>,
}

impl ContextModel {
    pub fn load(ck: &Checkpoint<'_>, chs: usize) -> Result<Self> {
        if !chs.is_multiple_of(32) {
            return Err(Error::Model(
                "context model: channel count must be a multiple of 32".into(),
            ));
        }
        let mut phases = Vec::with_capacity(4);
        for s in 0..4 {
            let prefix = format!("context.MCM.{s}");
            let context = if s == 0 {
                None
            } else {
                Some((
                    load::conv1x1(ck, &format!("{prefix}.conv.0"), s * chs, chs, false)?,
                    load::conv3x3(ck, &format!("{prefix}.conv.1"), chs, chs, chs / 32, true)?,
                ))
            };
            let fusion_in = if s == 0 { chs } else { 2 * chs };
            phases.push(Phase {
                context,
                fusion: FusionPredNet::load(
                    ck,
                    &format!("{prefix}.fusion_pred_net"),
                    fusion_in,
                    chs,
                )?,
            });
        }
        Ok(Self { chs, phases })
    }

    /// `Context.decompress`: `residual` `[C, H, W]` (dequantised), `psi` `[4C, ceil(H/2),
    /// ceil(W/2)]` → `y_hat` `[C, H, W]`.
    pub fn decompress(&self, residual: &Tensor<f32>, psi: &Tensor<f32>) -> Result<Tensor<f32>> {
        let (c, h, w) = (self.chs, residual.h, residual.w);
        let (hh, hw) = (h.div_ceil(2), w.div_ceil(2));
        if residual.c != c || psi.c != 4 * c || psi.h != hh || psi.w != hw {
            return Err(Error::InvalidArgument(
                "context model: residual / psi shape mismatch",
            ));
        }
        let mut y_hat = Tensor::<f32>::zeros(c, h, w)?;
        // Reconstructed stages so far, concatenated along channels (the spatial context).
        let mut context = Tensor::<f32>::zeros(0, hh, hw)?;
        for (s, phase) in self.phases.iter().enumerate() {
            let psi_s = ops::slice_channels(psi, s * c, (s + 1) * c)?;
            let mean = match &phase.context {
                None => phase.fusion.forward(&psi_s)?,
                Some((pointwise, grouped)) => {
                    let t = ops::conv2d(pointwise, &context)?;
                    let t = ops::conv2d(grouped, &t)?;
                    phase.fusion.forward(&ops::cat(&[&t, &psi_s])?)?
                }
            };
            // y_hat_s = residual_s + mean, on the half-resolution grid. Positions of the padded
            // (odd-size) border read a zero residual, like the reference's F.pad.
            let (py, px) = STAGE_POSITIONS[s];
            let mut stage = mean;
            for ch in 0..c {
                for y in 0..hh {
                    let sy = 2 * y + py;
                    for x in 0..hw {
                        let sx = 2 * x + px;
                        let r = if sy < h && sx < w {
                            residual.data[(ch * h + sy) * w + sx]
                        } else {
                            0.0
                        };
                        let v = &mut stage.data[(ch * hh + y) * hw + x];
                        *v += r;
                        if sy < h && sx < w {
                            y_hat.data[(ch * h + sy) * w + sx] = *v;
                        }
                    }
                }
            }
            context = if s == 0 {
                stage
            } else {
                ops::cat(&[&context, &stage])?
            };
        }
        Ok(y_hat)
    }
}

/// Mean of a component without a context model: `psi` quarters up-shuffled to latent
/// resolution (`Upsample_proc(chunk(psi, 4))`), cropped to `[C, h, w]`.
pub fn upshuffle_psi(psi: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
    if !psi.c.is_multiple_of(4) || psi.h * 2 < h || psi.w * 2 < w {
        return Err(Error::InvalidArgument("upshuffle_psi: shape mismatch"));
    }
    let c = psi.c / 4;
    let mut out = Tensor::<f32>::zeros(c, h, w)?;
    for (s, &(py, px)) in STAGE_POSITIONS.iter().enumerate() {
        for ch in 0..c {
            let src = psi.plane(s * c + ch);
            for y in (py..h).step_by(2) {
                for x in (px..w).step_by(2) {
                    out.data[(ch * h + y) * w + x] = src[(y / 2) * psi.w + x / 2];
                }
            }
        }
    }
    Ok(out)
}
