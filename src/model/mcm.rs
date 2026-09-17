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
use crate::nn::fast::{self, BTensor, ConvLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `(row, column)` of each stage inside a 2x2 cell.
pub const STAGE_POSITIONS: [(usize, usize); 4] = [(0, 0), (1, 1), (0, 1), (1, 0)];

/// `FusionPredNet`: three bias-free 1x1 convolutions with ReLU in between.
#[derive(Clone, Debug)]
struct FusionPredNet {
    conv1: ConvLayer,
    conv2: ConvLayer,
    conv3: ConvLayer,
}

impl FusionPredNet {
    fn load(
        ck: &Checkpoint<'_>,
        prefix: &str,
        in_ch: usize,
        chs: usize,
        eng: &Engine,
    ) -> Result<Self> {
        Ok(Self {
            conv1: ConvLayer::new(
                load::conv1x1(ck, &format!("{prefix}.conv1"), in_ch, chs, false)?,
                eng,
            )?,
            conv2: ConvLayer::new(
                load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?,
                eng,
            )?,
            conv3: ConvLayer::new(
                load::conv1x1(ck, &format!("{prefix}.conv3"), chs, chs, false)?,
                eng,
            )?,
        })
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut x = self.conv1.forward(eng, x)?;
        fast::relu(&mut x);
        let mut x = self.conv2.forward(eng, &x)?;
        fast::relu(&mut x);
        self.conv3.forward(eng, &x)
    }
}

/// One MCM phase. Phase 0 has no spatial context.
#[derive(Clone, Debug)]
struct Phase {
    /// `conv.0` (1x1, `s * C → C`, no bias) and `conv.1` (3x3, groups `C / 32`, bias).
    context: Option<(ConvLayer, ConvLayer)>,
    fusion: FusionPredNet,
}

#[derive(Clone, Debug)]
pub struct ContextModel {
    chs: usize,
    phases: Vec<Phase>,
}

impl ContextModel {
    pub fn load(ck: &Checkpoint<'_>, chs: usize, eng: &Engine) -> Result<Self> {
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
                    ConvLayer::new(
                        load::conv1x1(ck, &format!("{prefix}.conv.0"), s * chs, chs, false)?,
                        eng,
                    )?,
                    ConvLayer::new(
                        load::conv3x3(ck, &format!("{prefix}.conv.1"), chs, chs, chs / 32, true)?,
                        eng,
                    )?,
                ))
            };
            let fusion_in = if s == 0 { chs } else { 2 * chs };
            let fusion = FusionPredNet::load(
                ck,
                &format!("{prefix}.fusion_pred_net"),
                fusion_in,
                chs,
                eng,
            )?;
            phases.push(Phase { context, fusion });
        }
        Ok(Self { chs, phases })
    }

    /// `Context.decompress`: `residual` `[C, H, W]` (dequantised), `psi` `[4C, ceil(H/2),
    /// ceil(W/2)]` → `y_hat` `[C, H, W]`.
    pub fn decompress(
        &self,
        eng: &Engine,
        residual: &Tensor<f32>,
        psi: &BTensor,
    ) -> Result<Tensor<f32>> {
        let (c, h, w) = (self.chs, residual.h, residual.w);
        let (hh, hw) = (h.div_ceil(2), w.div_ceil(2));
        if residual.c != c || psi.c != 4 * c || psi.h != hh || psi.w != hw {
            return Err(Error::InvalidArgument(
                "context model: residual / psi shape mismatch",
            ));
        }
        let v = psi.v;
        let mut y_hat = Tensor::<f32>::zeros(c, h, w)?;
        // Reconstructed stages so far, concatenated along channels (the spatial context).
        let mut context: Option<BTensor> = None;
        for (s, phase) in self.phases.iter().enumerate() {
            let psi_s = psi.slice_channels(s * c, (s + 1) * c)?;
            let mean = match (&phase.context, &context) {
                (Some((pointwise, grouped)), Some(ctx)) => {
                    let t = grouped.forward(eng, &pointwise.forward(eng, ctx)?)?;
                    phase.fusion.forward(eng, &BTensor::cat(&[&t, &psi_s])?)?
                }
                _ => phase.fusion.forward(eng, &psi_s)?,
            };
            // y_hat_s = residual_s + mean, on the half-resolution grid. Positions of the padded
            // (odd-size) border read a zero residual, like the reference's F.pad.
            let (py, px) = STAGE_POSITIONS[s];
            let mut stage = mean;
            for ch in 0..c {
                let (b, lane) = (ch / v, ch % v);
                for y in 0..hh {
                    let sy = 2 * y + py;
                    let row = &mut stage.data[(b * hh + y) * hw * v..][..hw * v];
                    if sy >= h {
                        continue;
                    }
                    let res_row = &residual.data[(ch * h + sy) * w..][..w];
                    let out_row = &mut y_hat.data[(ch * h + sy) * w..][..w];
                    for x in 0..hw {
                        let sx = 2 * x + px;
                        if sx < w {
                            let val = &mut row[x * v + lane];
                            *val += res_row[sx];
                            out_row[sx] = *val;
                        }
                    }
                }
            }
            context = Some(match context {
                None => stage,
                Some(ctx) => BTensor::cat(&[&ctx, &stage])?,
            });
        }
        Ok(y_hat)
    }
}

/// Mean of a component without a context model: `psi` quarters up-shuffled to latent
/// resolution (`Upsample_proc(chunk(psi, 4))`), cropped to `[C, h, w]`.
pub fn upshuffle_psi(psi: &BTensor, h: usize, w: usize) -> Result<Tensor<f32>> {
    if !psi.c.is_multiple_of(4) || psi.h * 2 < h || psi.w * 2 < w {
        return Err(Error::InvalidArgument("upshuffle_psi: shape mismatch"));
    }
    let c = psi.c / 4;
    let v = psi.v;
    let mut out = Tensor::<f32>::zeros(c, h, w)?;
    for (s, &(py, px)) in STAGE_POSITIONS.iter().enumerate() {
        for ch in 0..c {
            let pch = s * c + ch;
            let (b, lane) = (pch / v, pch % v);
            for y in (py..h).step_by(2) {
                let src = &psi.data[(b * psi.h + y / 2) * psi.w * v..][..psi.w * v];
                for x in (px..w).step_by(2) {
                    out.data[(ch * h + y) * w + x] = src[(x / 2) * v + lane];
                }
            }
        }
    }
    Ok(out)
}
