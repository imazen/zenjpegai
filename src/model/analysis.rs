//! Analysis transforms: picture samples → latent `y` (encoder side).
//!
//! Ports `ref/src/codec/components/autoencoder_data/encoder/{bop,hop}_{prim,sec}.py` and the
//! helpers `normalize` / `padding_layer` of `base_layers/utils.py`. There is no SOP analysis
//! transform: simple-profile streams are produced with the BOP one.
//!
//! ```text
//! primary:   normalize → [pad → conv3x3/2 → ResAU] x3 → pad → conv3x3/2 → conv1x1      (/16)
//! secondary: normalize → [pad → conv3x3/2 → ResAU] x3 → conv1x1                          (/8)
//! HOP adds a TAM in front of the second convolution and a CAB in front of the third.
//! ```
//!
//! `pad` replicates the last row / column so that the map has an even size before every
//! stride-2 convolution. The secondary transform's input is 12 planes at half the luma
//! resolution: luma pixel-unshuffled by 2, then U and V pixel-unshuffled by 2 (4:4:4 coding).
//!
//! `feature_clipping` is not ported: no shipped checkpoint carries `clip_thres` and the
//! reference runs with `clipping_mode = 0`.

use enough::Stop;

use crate::error::{Error, Result};
use crate::header::OperatingPoint;
use crate::model::attention::{Cab, Tam};
use crate::model::load;
use crate::model::synthesis::ResAu;
use crate::nn::fast::{BTensor, ConvLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// Hidden width of every analysis transform.
const C: usize = 128;

/// `normalize`: `(x - 127.5) / 128`.
fn normalize(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data.iter().map(|&v| (v - 127.5) / 128.0).collect();
    Tensor::from_vec(x.c, x.h, x.w, data)
}

/// Rows / columns `padding_layer(.., depth)` adds for a picture (tile) dimension `len`:
/// `ceil(len / 2^(depth+1)) * 2 - ceil(len / 2^depth)`, i.e. 1 when the map is odd.
pub(crate) fn pad_amount(len: usize, divider: usize) -> usize {
    2 * len.div_ceil(2 * divider) - len.div_ceil(divider)
}

/// `F.pad(x, (0, right, 0, bottom), mode = "replicate")` on a blocked tensor.
pub(crate) fn replicate_pad(x: BTensor, bottom: usize, right: usize) -> Result<BTensor> {
    if bottom == 0 && right == 0 {
        return Ok(x);
    }
    if x.h == 0 || x.w == 0 {
        return Err(Error::InvalidArgument("replicate padding of an empty map"));
    }
    let (h, w, v) = (x.h + bottom, x.w + right, x.v);
    let mut out = BTensor::scratch(x.c, h, w, v)?;
    for b in 0..x.blocks() {
        for y in 0..h {
            let sy = y.min(x.h - 1);
            let src = &x.data[(b * x.h + sy) * x.w * v..][..x.w * v];
            let dst = &mut out.data[(b * h + y) * w * v..][..w * v];
            dst[..x.w * v].copy_from_slice(src);
            let last = &src[(x.w - 1) * v..];
            for px in dst[x.w * v..].chunks_exact_mut(v) {
                px.copy_from_slice(last);
            }
        }
    }
    Ok(out)
}

/// `padding_layer` for the map a picture of `h x w` samples has at `divider`.
fn pad_for(x: BTensor, h: usize, w: usize, divider: usize) -> Result<BTensor> {
    let (want_h, want_w) = (h.div_ceil(divider), w.div_ceil(divider));
    if x.h != want_h || x.w != want_w {
        return Err(Error::InvalidArgument(
            "analysis transform: map size does not match the picture size",
        ));
    }
    replicate_pad(x, pad_amount(h, divider), pad_amount(w, divider))
}

fn conv_s2(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    eng: &Engine,
) -> Result<ConvLayer> {
    ConvLayer::new(
        load::conv(ck, prefix, in_ch, out_ch, (3, 3), 2, (1, 1), 1, true)?,
        eng,
    )
}

/// The three `conv → ResAU` stages both transforms share, with HOP's attention blocks.
#[derive(Clone, Debug)]
struct Trunk {
    conv1: ConvLayer,
    act1: ResAu,
    tam: Option<Tam>,
    conv2: ConvLayer,
    act2: ResAu,
    cab: Option<Cab>,
    conv3: ConvLayer,
    act3: ResAu,
}

impl Trunk {
    fn load(
        ck: &Checkpoint<'_>,
        op: OperatingPoint,
        in_ch: usize,
        tam_downsamples: bool,
        eng: &Engine,
    ) -> Result<Self> {
        let hop = op == OperatingPoint::Hop;
        Ok(Self {
            conv1: conv_s2(ck, "conv1", in_ch, C, eng)?,
            act1: ResAu::load(ck, "act1", C, eng)?,
            tam: if hop {
                Some(Tam::load(ck, "TAM", C, tam_downsamples, eng)?)
            } else {
                None
            },
            conv2: conv_s2(ck, "conv2", C, C, eng)?,
            act2: ResAu::load(ck, "act2", C, eng)?,
            cab: if hop {
                Some(Cab::load(ck, "cab", C, eng)?)
            } else {
                None
            },
            conv3: conv_s2(ck, "conv3", C, C, eng)?,
            act3: ResAu::load(ck, "act3", C, eng)?,
        })
    }

    /// Input in `[0, 255]`; output at 1/8 of the input resolution, not padded.
    fn forward(
        &self,
        eng: &Engine,
        x: &Tensor<f32>,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<BTensor> {
        if x.h != h || x.w != w {
            return Err(Error::InvalidArgument("analysis transform: input size"));
        }
        let x = BTensor::from_planar_par(eng, &normalize(x)?, eng.tier.block())?;
        let x = pad_for(x, h, w, 1)?;
        let x = self.act1.forward(eng, self.conv1.forward(eng, &x)?)?;
        stop.check()?;
        let mut x = pad_for(x, h, w, 2)?;
        if let Some(tam) = &self.tam {
            x = tam.forward(eng, x)?;
            stop.check()?;
        }
        let x = self.act2.forward(eng, self.conv2.forward(eng, &x)?)?;
        stop.check()?;
        let mut x = pad_for(x, h, w, 4)?;
        if let Some(cab) = &self.cab {
            x = cab.forward(eng, &x)?;
            stop.check()?;
        }
        let x = self.act3.forward(eng, self.conv3.forward(eng, &x)?)?;
        stop.check()?;
        Ok(x)
    }
}

fn check_op(op: OperatingPoint) -> Result<()> {
    match op {
        OperatingPoint::Bop | OperatingPoint::Hop => Ok(()),
        OperatingPoint::Sop => Err(Error::InvalidArgument(
            "there is no SOP analysis transform (use BOP)",
        )),
    }
}

/// Primary (luma) analysis transform: `[1, h, w]` in `[0, 255]` → `[160, ceil(h/16), ceil(w/16)]`.
#[derive(Clone, Debug)]
pub struct AnalysisPrimary {
    trunk: Trunk,
    conv4: ConvLayer,
    conv5: ConvLayer,
}

impl AnalysisPrimary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        check_op(op)?;
        let chs = crate::header::LATENT_CHANNELS[0];
        Ok(Self {
            trunk: Trunk::load(ck, op, 1, true, eng)?,
            conv4: conv_s2(ck, "conv4", C, chs, eng)?,
            conv5: ConvLayer::new(load::conv1x1(ck, "conv5", chs, chs, false)?, eng)?,
        })
    }

    pub fn forward(&self, eng: &Engine, x: &Tensor<f32>, stop: &dyn Stop) -> Result<BTensor> {
        let (h, w) = (x.h, x.w);
        let t = self.trunk.forward(eng, x, h, w, stop)?;
        let t = pad_for(t, h, w, 8)?;
        let t = self.conv4.forward(eng, &t)?;
        stop.check()?;
        self.conv5.forward(eng, &t)
    }
}

/// Secondary (chroma) analysis transform: `[12, h, w]` (half the luma resolution, luma planes
/// first) in `[0, 255]` → `[96, ceil(h/8), ceil(w/8)]`.
#[derive(Clone, Debug)]
pub struct AnalysisSecondary {
    trunk: Trunk,
    conv5: ConvLayer,
}

impl AnalysisSecondary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        check_op(op)?;
        let chs = crate::header::LATENT_CHANNELS[1];
        Ok(Self {
            trunk: Trunk::load(ck, op, 12, false, eng)?,
            conv5: ConvLayer::new(load::conv1x1(ck, "conv5", C, chs, false)?, eng)?,
        })
    }

    pub fn forward(&self, eng: &Engine, x: &Tensor<f32>, stop: &dyn Stop) -> Result<BTensor> {
        let t = self.trunk.forward(eng, x, x.h, x.w, stop)?;
        self.conv5.forward(eng, &t)
    }
}
