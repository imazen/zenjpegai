//! Multi-stage context model (MCM): reconstructs the latent `y_hat` from the dequantised
//! residual and `psi` in four passes, one per position of a 2x2 grid.
//!
//! Ports `ref/src/codec/components/contexts/` (`context.py::Context.decompress/pred`,
//! `MCM_phases.py`, `fusion_pred_net.py`, `utils.py::ContextUtils`).
//!
//! Stage order over the 2x2 grid is (0,0), (1,1), (0,1), (1,0) as `(row, column)`. Stage `s`
//! predicts a mean from the `s`-th quarter of `psi` and from every stage reconstructed so far;
//! the stage's latent is `residual + mean`.

// The stage loops walk a channel index across several parallel arrays (a blocked mean, a planar
// latent, a planar mask, a per-channel scaler); an iterator chain over one of them would hide
// that correspondence.
#![allow(clippy::needless_range_loop)]

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
        x = self.conv2.forward(eng, &x)?;
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
        self.decompress_with(eng, residual, psi, &enough::Unstoppable)
    }

    /// [`Self::decompress`] that checks `stop` before each of the four stages.
    pub fn decompress_with(
        &self,
        eng: &Engine,
        residual: &Tensor<f32>,
        psi: &BTensor,
        stop: &dyn enough::Stop,
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
            stop.check()?;
            let psi_s = psi.slice_channels(s * c, (s + 1) * c)?;
            let mean = match (&phase.context, &context) {
                (Some((pointwise, grouped)), Some(ctx)) => {
                    let mut t = pointwise.forward(eng, ctx)?;
                    t = grouped.forward(eng, &t)?;
                    let joined = BTensor::cat(&[&t, &psi_s])?;
                    drop(t);
                    phase.fusion.forward(eng, &joined)?
                }
                _ => phase.fusion.forward(eng, &psi_s)?,
            };
            drop(psi_s);
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

/// Skip mode's spatial cube edge, in down-shuffled (half resolution) latent samples.
pub const CUBE_SIZE: usize = 8;

/// `quant_dequant` for one latent sample: `quantize_resi`, the skip mask, the int16 clamp, round
/// half to even, then `dequantize_resi`.
///
/// `index` is the sample's offset inside its channel plane *at latent resolution*, because the
/// per-position tools (RVS / GRFS, the quality map) index their tables there.
pub trait Quantiser {
    fn quantise(&self, ch: usize, index: usize, x: f32, coded: bool) -> (i16, f32);
}

/// The gain unit alone: `round_ties_even(clamp(x * scaler))`, then `q / (scaler + 1e-9)`.
#[derive(Clone, Copy, Debug)]
pub struct GainQuantiser<'a> {
    pub scaler: &'a [f32],
}

impl Quantiser for GainQuantiser<'_> {
    #[inline]
    fn quantise(&self, ch: usize, _index: usize, x: f32, coded: bool) -> (i16, f32) {
        let s = self.scaler[ch];
        let q = if coded {
            (x * s).clamp(-32768.0, 32767.0).round_ties_even()
        } else {
            0.0
        };
        (q as i16, q / (s + 1e-9f32))
    }
}

/// Encoder-side output of one component's residual quantisation.
#[derive(Clone, Debug)]
pub struct Compressed {
    pub residual_q: Tensor<i16>,
    /// Dequantised residual: what the decoder recovers.
    pub residual: Tensor<f32>,
    /// Skip-mode cube flags, `[phase][cube_y][cube_x]` with phases in raster order
    /// ((0,0), (0,1), (1,0), (1,1)); `true` = the cube may be skipped. Exactly the layout
    /// [`crate::header::ComponentHeader::cube_flags`] and [`crate::tools::skip::skip_mask`] use.
    pub cube_flag: Vec<bool>,
}

/// Maximum of `err` over each `CUBE_SIZE x CUBE_SIZE` block of every channel, thresholded:
/// `gen_skip_cubeflag`'s 3-D max-pool over (all channels, 8, 8). `err` is `[C, h, w]` on the
/// down-shuffled grid; the result is `[cube_h][cube_w]`, `true` = the cube may be skipped.
fn cube_flags_of(err: &Tensor<f32>, thr: f32) -> Vec<bool> {
    let (cube_h, cube_w) = (err.h.div_ceil(CUBE_SIZE), err.w.div_ceil(CUBE_SIZE));
    let mut flags = alloc::vec![true; cube_h * cube_w];
    for ch in 0..err.c {
        let plane = err.plane(ch);
        for y in 0..err.h {
            let row = &plane[y * err.w..][..err.w];
            let fy = (y / CUBE_SIZE) * cube_w;
            for (x, &e) in row.iter().enumerate() {
                if e > thr {
                    flags[fy + x / CUBE_SIZE] = false;
                }
            }
        }
    }
    flags
}

/// `_mask_redundant_padding_*`: the row / column that only exists because an odd-sized latent was
/// padded to even carries no information; stages 1 and 3 own the padded row, 1 and 2 the padded
/// column.
#[inline]
fn is_padding(stage: usize, h: usize, w: usize, hh: usize, hw: usize, y: usize, x: usize) -> bool {
    (h % 2 == 1 && matches!(stage, 1 | 3) && y + 1 == hh)
        || (w % 2 == 1 && matches!(stage, 1 | 2) && x + 1 == hw)
}

impl ContextModel {
    /// `Context.forward` / `pred` in the encoder direction: quantise `y - mean` stage by stage,
    /// deriving each stage's skip-mode cube flags from its own reconstruction error and
    /// re-quantising with the cubes it must not skip.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn compress<Q: Quantiser>(
        &self,
        eng: &Engine,
        y: &Tensor<f32>,
        psi: &BTensor,
        q: &Q,
        mask: &Tensor<bool>,
        cube_thr: f32,
        stop: &dyn enough::Stop,
    ) -> Result<Compressed> {
        let (c, h, w) = (self.chs, y.h, y.w);
        let (hh, hw) = (h.div_ceil(2), w.div_ceil(2));
        if y.c != c
            || mask.c != c
            || mask.h != h
            || mask.w != w
            || psi.c != 4 * c
            || psi.h != hh
            || psi.w != hw
        {
            return Err(Error::InvalidArgument(
                "context model: y / psi / mask shape mismatch",
            ));
        }
        let v = psi.v;
        let mut residual_q = Tensor::<i16>::zeros(c, h, w)?;
        let mut residual = Tensor::<f32>::zeros(c, h, w)?;
        let (cube_h, cube_w) = (hh.div_ceil(CUBE_SIZE), hw.div_ceil(CUBE_SIZE));
        let mut cube_flag = alloc::vec![true; 4 * cube_h * cube_w];
        let mut err = Tensor::<f32>::zeros(c, hh, hw)?;
        let mut diff = Tensor::<f32>::zeros(c, hh, hw)?;
        let mut coded = Tensor::<bool>::zeros(c, hh, hw)?;

        let mut context: Option<BTensor> = None;
        for (s, phase) in self.phases.iter().enumerate() {
            stop.check()?;
            let psi_s = psi.slice_channels(s * c, (s + 1) * c)?;
            let mut stage = match (&phase.context, &context) {
                (Some((pointwise, grouped)), Some(ctx)) => {
                    let t = grouped.forward(eng, &pointwise.forward(eng, ctx)?)?;
                    phase.fusion.forward(eng, &BTensor::cat(&[&t, &psi_s])?)?
                }
                _ => phase.fusion.forward(eng, &psi_s)?,
            };
            let (py, px) = STAGE_POSITIONS[s];

            // Pass 1: quantise with the sigma-threshold mask only, and record the error the
            // cube flags are decided on (zeroed on the padded row / column).
            for ch in 0..c {
                let (b, lane) = (ch / v, ch % v);
                for yy in 0..hh {
                    let sy = 2 * yy + py;
                    let row = &stage.data[(b * hh + yy) * hw * v..][..hw * v];
                    for xx in 0..hw {
                        let sx = 2 * xx + px;
                        // F.pad(y, ..) fills the padded row / column with zero, and the padded
                        // positions of the mask with false.
                        let inside = sy < h && sx < w;
                        let m = inside && mask.data[(ch * h + sy) * w + sx];
                        let d = if inside {
                            y.data[(ch * h + sy) * w + sx] - row[xx * v + lane]
                        } else {
                            -row[xx * v + lane]
                        };
                        let i = (ch * hh + yy) * hw + xx;
                        diff.data[i] = d;
                        coded.data[i] = m;
                        let li = if inside { sy * w + sx } else { 0 };
                        let (_, dq) = q.quantise(ch, li, d, m);
                        err.data[i] = if is_padding(s, h, w, hh, hw, yy, xx) {
                            0.0
                        } else {
                            (dq - d).abs()
                        };
                    }
                }
            }
            let flags = cube_flags_of(&err, cube_thr);
            // Stage -> raster phase channel: (0,0), (1,1), (0,1), (1,0) -> 0, 3, 1, 2.
            let out_phase = py * 2 + px;
            cube_flag[out_phase * cube_h * cube_w..][..cube_h * cube_w].copy_from_slice(&flags);

            // Pass 2: the cubes that must not be skipped join the mask.
            for ch in 0..c {
                let (b, lane) = (ch / v, ch % v);
                for yy in 0..hh {
                    let sy = 2 * yy + py;
                    let row = &mut stage.data[(b * hh + yy) * hw * v..][..hw * v];
                    for xx in 0..hw {
                        let sx = 2 * xx + px;
                        let i = (ch * hh + yy) * hw + xx;
                        let cube = flags[(yy / CUBE_SIZE) * cube_w + xx / CUBE_SIZE];
                        let inside = sy < h && sx < w;
                        let code = (coded.data[i] || !cube) && !is_padding(s, h, w, hh, hw, yy, xx);
                        let li = if inside { sy * w + sx } else { 0 };
                        let (sym, dq) = q.quantise(ch, li, diff.data[i], code);
                        let val = &mut row[xx * v + lane];
                        *val += dq;
                        if inside {
                            residual_q.data[(ch * h + sy) * w + sx] = sym;
                            residual.data[(ch * h + sy) * w + sx] = dq;
                        }
                    }
                }
            }
            context = Some(match context {
                None => stage,
                Some(ctx) => BTensor::cat(&[&ctx, &stage])?,
            });
        }
        Ok(Compressed {
            residual_q,
            residual,
            cube_flag,
        })
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
