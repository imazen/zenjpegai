//! Synthesis transforms: latent `y_hat` → reconstructed samples.
//!
//! Ports `ref/src/codec/components/autoencoder_data/decoder/{sop,bop,hop}_{prim,sec}.py`,
//! `activations/resau.py` and the residual blocks of `base_layers/conv_layers.py`; the HOP
//! attention blocks live in [`super::attention`].
//!
//! The primary (luma) transform maps `[160, H, W]` to one plane at 16x the resolution. The
//! secondary (chroma) transform takes the luma latent as side information (`cat(y_hat_luma,
//! y_hat_chroma)`) and produces both chroma planes at *luma* resolution; subsampling to the
//! coded chroma format happens afterwards.

use alloc::format;

use enough::{Stop, Unstoppable};

use crate::error::{Error, Result};
use crate::header::OperatingPoint;
use crate::model::attention::{Cab, ResidualBlock, Tam, conv3x3_t};
use crate::model::load;
use crate::nn::fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `ResAU`: `y = x * (1 + conv1x1(conv3x3_grouped(relu6(x))))`, both convolutions bias-free.
#[derive(Clone, Debug)]
pub(crate) struct ResAu {
    conv: ConvLayer,
    conv2: ConvLayer,
}

impl ResAu {
    pub(crate) fn load(
        ck: &Checkpoint<'_>,
        prefix: &str,
        chs: usize,
        eng: &Engine,
    ) -> Result<Self> {
        // The group count differs between transforms (16 channels per group in SOP / BOP, 4
        // groups in HOP); the checkpoint's weight shape `[chs, chs / groups, 3, 3]` decides.
        let name = format!("{prefix}.conv.weight");
        let per_group = ck
            .info(&name)
            .and_then(|t| t.shape.get(1).copied())
            .filter(|&g| g != 0 && chs.is_multiple_of(g))
            .ok_or_else(|| Error::Model(format!("{name}: missing or malformed")))?;
        let conv = load::conv3x3(
            ck,
            &format!("{prefix}.conv"),
            chs,
            chs,
            chs / per_group,
            false,
        )?;
        let conv2 = load::conv1x1(ck, &format!("{prefix}.conv2"), chs, chs, false)?;
        Ok(Self {
            conv: ConvLayer::new(conv, eng)?,
            conv2: ConvLayer::new(conv2, eng)?,
        })
    }

    pub(crate) fn forward(&self, eng: &Engine, x: BTensor) -> Result<BTensor> {
        // Every intermediate is dropped as soon as the next layer has consumed it: these maps
        // are the decoder's largest allocations.
        let mut act = x.clone();
        fast::relu6(&mut act);
        let t = self.conv.forward(eng, &act)?;
        drop(act);
        let mask = self.conv2.forward(eng, &t)?;
        drop(t);
        let mut y = x;
        // Two IEEE operations, like the reference's `x * (1 + mask)`; not an FMA.
        fast::gate(&mut y, &mask)?;
        Ok(y)
    }

    /// [`Self::forward`] that returns `(out, x_input, mask)` — the pair its VJP
    /// needs (`x` is the gate's `x` operand and relu6's forward input).
    fn forward_tape(&self, eng: &Engine, x: BTensor) -> Result<(BTensor, BTensor, BTensor)> {
        let mut act = x.clone();
        fast::relu6(&mut act);
        let t = self.conv.forward(eng, &act)?;
        drop(act);
        let mask = self.conv2.forward(eng, &t)?;
        drop(t);
        let x_save = x.clone();
        let mut y = x;
        // Two IEEE operations, like the reference's `x * (1 + mask)`; not an FMA.
        fast::gate(&mut y, &mask)?;
        Ok((y, x_save, mask))
    }

    /// `x * (1 + conv2(conv(relu6(x))))`'s input gradient (RDLR VJP): given
    /// `d_y` (planar) and the saved `x` / `mask`, the gradient on `x`.
    fn backward(&self, d_y: &Tensor<f32>, x: &BTensor, mask: &BTensor) -> Result<Tensor<f32>> {
        // y = x * (1 + mask) → d_x += d_y·(1+mask), d_mask = d_y·x.
        let xp = x.to_planar()?;
        let mp = mask.to_planar()?;
        let mut d_x = d_y.clone();
        let mut d_mask = d_y.clone();
        for i in 0..d_x.data.len() {
            d_x.data[i] *= 1.0 + mp.data[i];
            d_mask.data[i] *= xp.data[i];
        }
        let d_t = self.conv2.backward(&d_mask, mask.h, mask.w)?;
        let d_act = self.conv.backward(&d_t, xp.h, xp.w)?;
        // relu6 backward: pass d_act where 0 < x < 6.
        for i in 0..d_x.data.len() {
            let xv = xp.data[i];
            if xv > 0.0 && xv < 6.0 {
                d_x.data[i] += d_act.data[i];
            }
        }
        Ok(d_x)
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
            OperatingPoint::Hop => Err(Error::InvalidArgument("HOP is not a light transform")),
        }
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        match self {
            Self::Transposed(c) => c.forward(eng, x),
            Self::Conv2x2Shuffle(c) => {
                let t = c.forward(eng, x)?;
                fast::pixel_shuffle(eng, &t, 2)
            }
        }
    }

    /// The input gradient through this upsample (RDLR VJP), planar in/out.
    /// `Transposed`: an ordinary stride-2 conv of the output gradient.
    /// `Conv2x2Shuffle`: pixel-shuffle backward then the conv's.
    fn backward(&self, dout: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
        match self {
            Self::Transposed(c) => c.backward(dout, h, w),
            Self::Conv2x2Shuffle(c) => {
                let d_t = pixel_shuffle_backward(dout, 2)?;
                c.backward(&d_t, h, w)
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

/// `denormalize`'s input gradient (RDLR VJP): `d = d·128` where the output was
/// inside the `[0,255]` clamp, else `0`.
fn denormalize_backward(d: &Tensor<f32>, pre: &Tensor<f32>) -> Tensor<f32> {
    let mut out = d.clone();
    for (o, &p) in out.data.iter_mut().zip(&pre.data) {
        let v = p * 128.0 + 127.5;
        *o = if v > 0.0 && v < 255.0 {
            *o * 128.0
        } else {
            0.0
        };
    }
    out
}

/// `pixel_shuffle_to_planar`'s input gradient (RDLR VJP): `d_out[c, y·r+dy,
/// x·r+dx]` gathers to `d_in[c·r² + dy·r + dx, y, x]` — planar both sides.
fn pixel_shuffle_backward(d_out: &Tensor<f32>, r: usize) -> Result<Tensor<f32>> {
    let (c, h, w) = (d_out.c, d_out.h, d_out.w);
    let (ih, iw) = (h / r, w / r);
    let mut din = Tensor::<f32>::zeros(c * r * r, ih, iw)?;
    for oc in 0..c {
        for y in 0..ih {
            for dy in 0..r {
                for dx in 0..r {
                    let src = &d_out.data[(oc * h + y * r + dy) * w..][..w];
                    let dst = din.plane_mut(oc * r * r + dy * r + dx);
                    for x in 0..iw {
                        dst[y * iw + x] = src[x * r + dx];
                    }
                }
            }
        }
    }
    Ok(din)
}

/// Primary (luma) synthesis transform.
#[derive(Clone, Debug)]
pub struct SynthesisPrimary(Primary);

#[derive(Clone, Debug)]
enum Primary {
    Light(alloc::boxed::Box<LightPrimary>),
    Hop(alloc::boxed::Box<HopPrimary>),
}

impl SynthesisPrimary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        Ok(Self(match op {
            OperatingPoint::Hop => Primary::Hop(alloc::boxed::Box::new(HopPrimary::load(ck, eng)?)),
            _ => Primary::Light(alloc::boxed::Box::new(LightPrimary::load(ck, op, eng)?)),
        }))
    }

    /// `y_hat` `[160, H, W]` → luma `[1, >= h, >= w]` in `[0, 255]`, where the intermediate maps
    /// are cropped for a target picture (tile) of `h x w` samples. The caller crops to `h x w`.
    pub fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor<f32>> {
        self.forward_with(eng, y_hat, h, w, &Unstoppable)
    }

    /// [`Self::forward`] that checks `stop` between layers.
    pub fn forward_with(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        match &self.0 {
            Primary::Light(m) => m.forward(eng, y_hat, h, w, stop),
            Primary::Hop(m) => m.forward(eng, y_hat, h, w, stop),
        }
    }

    /// [`Self::forward_with`] that returns the VJP tape alongside the output
    /// (RDLR). `Light` primary only — the HOP decoder's VJP is a later item.
    pub fn forward_tape(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<(Tensor<f32>, PrimaryTape)> {
        match &self.0 {
            Primary::Light(m) => {
                let (rec, t) = m.forward_tape(eng, y_hat, h, w, stop)?;
                Ok((rec, PrimaryTape::Light(t)))
            }
            Primary::Hop(_) => Err(Error::InvalidArgument(
                "RDLR VJP: HOP synthesis backward not yet implemented",
            )),
        }
    }

    /// The input gradient through the synthesis (RDLR VJP): `d_rec` → `d(y_hat)`
    /// as a planar `[160, lh, lw]` tensor.
    pub fn backward(&self, tape: &PrimaryTape, d_rec: &Tensor<f32>) -> Result<Tensor<f32>> {
        match (&self.0, tape) {
            (Primary::Light(m), PrimaryTape::Light(t)) => m.backward(t, d_rec),
            _ => Err(Error::InvalidArgument("RDLR VJP: tape / primary mismatch")),
        }
    }
}

/// The synthesis's per-primary VJP tape (RDLR).
pub enum PrimaryTape {
    /// `LightPrimary`'s tape.
    Light(SynthTape),
}

/// `DecoderHOPPrim`.
#[derive(Clone, Debug)]
struct HopPrimary {
    res: ResidualBlock,
    up1: ConvTransposeLayer,
    act1: ResAu,
    up2: ConvTransposeLayer,
    cab: Cab,
    act2: ResAu,
    conv3: ConvLayer,
    tam: Tam,
    act3: ResAu,
    up4: ConvTransposeLayer,
}

impl HopPrimary {
    const C: usize = 128;

    fn load(ck: &Checkpoint<'_>, eng: &Engine) -> Result<Self> {
        let c = Self::C;
        Ok(Self {
            res: ResidualBlock::load(ck, "first_stage.0.0", 160, eng)?,
            up1: conv3x3_t(ck, "first_stage.1", 160, c, eng)?,
            act1: ResAu::load(ck, "first_stage.2", c, eng)?,
            up2: conv3x3_t(ck, "conv2_t", c, c, eng)?,
            cab: Cab::load(ck, "CAB", c, eng)?,
            act2: ResAu::load(ck, "iact2", c, eng)?,
            conv3: ConvLayer::new(load::conv1x1(ck, "conv3_t", c, 4 * c, true)?, eng)?,
            tam: Tam::load(ck, "TAM", c, true, eng)?,
            act3: ResAu::load(ck, "iact3", c, eng)?,
            up4: conv3x3_t(ck, "conv4_t", c, 1, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        // `x = f(&x)` frees the previous map as soon as the next one exists (a shadowing `let`
        // would keep every one of them alive until the function returns).
        let mut x = self.res.forward(eng, y_hat)?;
        stop.check()?;
        x = self.up1.forward(eng, &x)?;
        x = self.act1.forward(eng, x)?;
        x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        x = self.up2.forward(eng, &x)?;
        x = self.cab.forward(eng, &x)?;
        x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        x = self.act2.forward(eng, x)?;
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        x = fast::pixel_shuffle(eng, &x, 2)?;
        stop.check()?;
        x = self.tam.forward(eng, x)?;
        x = x.crop_par(eng, h.div_ceil(2), w.div_ceil(2))?;
        stop.check()?;
        x = self.act3.forward(eng, x)?;
        stop.check()?;
        x = self.up4.forward(eng, &x)?;
        let mut x = x.to_planar()?;
        denormalize(&mut x);
        Ok(x)
    }
}

/// `DecoderSOPPrim` / `DecoderBOPPrim`.
#[derive(Clone, Debug)]
struct LightPrimary {
    res_conv: ConvLayer,
    up1: Upsample,
    act1: ResAu,
    up2: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl LightPrimary {
    fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        // hidden channel counts: (after up1, after up2, after conv3)
        let (c1, c2, c3) = match op {
            OperatingPoint::Sop => (64, 32, 32),
            OperatingPoint::Bop => (64, 64, 96),
            OperatingPoint::Hop => {
                return Err(Error::InvalidArgument("HOP is not a light transform"));
            }
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

    fn forward(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        // LightResidualBlock: relu(conv(x)) + x
        let mut x = self.res_conv.forward(eng, y_hat)?;
        fast::relu(&mut x);
        fast::add_assign(&mut x, y_hat)?;
        stop.check()?;
        x = self.up1.forward(eng, &x)?;
        x = self.act1.forward(eng, x)?;
        x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        x = self.up2.forward(eng, &x)?;
        x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        x = self.act2.forward(eng, x)?;
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        stop.check()?;
        x = self.act3.forward(eng, x)?;
        stop.check()?;
        x = self.conv4.forward(eng, &x)?;
        let mut x = fast::pixel_shuffle_to_planar(&x, 4)?;
        denormalize(&mut x);
        Ok(x)
    }

    /// [`Self::forward`] that records the intermediates the VJP needs.
    #[allow(clippy::too_many_lines)]
    fn forward_tape(
        &self,
        eng: &Engine,
        y_hat: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<(Tensor<f32>, SynthTape)> {
        // res block: relu(conv(y)) + y
        let mut x = self.res_conv.forward(eng, y_hat)?;
        let res_conv_out = x.clone();
        fast::relu(&mut x);
        fast::add_assign(&mut x, y_hat)?;
        stop.check()?;
        x = self.up1.forward(eng, &x)?;
        let (x, a1x, a1m) = self.act1.forward_tape(eng, x)?;
        let a1_dims = (x.h, x.w);
        let mut x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        let crop8_dims = (x.h, x.w);
        stop.check()?;
        x = self.up2.forward(eng, &x)?;
        let up2_dims = (x.h, x.w);
        let x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        let (x, a2x, a2m) = self.act2.forward_tape(eng, x)?;
        let mut x = x;
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        stop.check()?;
        let (x, a3x, a3m) = self.act3.forward_tape(eng, x)?;
        let mut x = x;
        stop.check()?;
        x = self.conv4.forward(eng, &x)?;
        let mut x = fast::pixel_shuffle_to_planar(&x, 4)?;
        let pre_denorm = x.clone();
        denormalize(&mut x);
        Ok((
            x,
            SynthTape {
                res_conv_out,
                act: [(a1x, a1m), (a2x, a2m), (a3x, a3m)],
                pre_denorm,
                a1_dims,
                crop8_dims,
                up2_dims,
            },
        ))
    }

    /// The input gradient through `forward` (RDLR VJP): `d_rec` (the upstream
    /// gradient on the reconstructed luma `[1, H, W]`) → the gradient on
    /// `y_hat` `[160, latent_h, latent_w]`, planar.
    fn backward(&self, tape: &SynthTape, d_rec: &Tensor<f32>) -> Result<Tensor<f32>> {
        // denormalize → pixel_shuffle → conv4 → act3 → conv3 → act2 →
        //   pad(up2_dims) → up2 → pad(a1_dims) → act1 → up1 → res block.
        let d = denormalize_backward(d_rec, &tape.pre_denorm);
        let d = pixel_shuffle_backward(&d, 4)?;
        // conv4/conv3 are same-size convs: the input's h,w are the output's.
        let d = self.conv4.backward(&d, d.h, d.w)?;
        let d = self.act3.backward(&d, &tape.act[2].0, &tape.act[2].1)?;
        let d = self.conv3.backward(&d, d.h, d.w)?;
        let d = self.act2.backward(&d, &tape.act[1].0, &tape.act[1].1)?;
        // crop_par(4) backward: pad the gradient to up2's output dims.
        let d = zero_pad(&d, tape.up2_dims.0, tape.up2_dims.1)?;
        // up2's input was the crop8 output (h/8, w/8).
        let d = self
            .up2
            .backward(&d, tape.crop8_dims.0, tape.crop8_dims.1)?;
        // crop_par(8) backward: pad to act1's (= up1's) output dims.
        let d = zero_pad(&d, tape.a1_dims.0, tape.a1_dims.1)?;
        let d = self.act1.backward(&d, &tape.act[0].0, &tape.act[0].1)?;
        // up1's input was the residual sum `relu(conv(y)) + y` at latent size.
        let (lh, lw) = (tape.res_conv_out.h, tape.res_conv_out.w);
        let d = self.up1.backward(&d, lh, lw)?;
        // Residual block: `x1 = relu(x0) + y` → d(y) += d_x1 and, through the
        // conv branch, `res_conv.backward(relu'(x0)·d_x1)`.
        let mut d_y = d.clone();
        let mut d_x0 = d;
        let rco = tape.res_conv_out.to_planar()?;
        for (g, &p) in d_x0.data.iter_mut().zip(&rco.data) {
            if p <= 0.0 {
                *g = 0.0;
            }
        }
        let d_conv = self.res_conv.backward(&d_x0, lh, lw)?;
        for (a, &b) in d_y.data.iter_mut().zip(&d_conv.data) {
            *a += b;
        }
        Ok(d_y)
    }
}

/// Zero-pad a planar gradient to `h × w` — `crop_par`'s backward.
fn zero_pad(d: &Tensor<f32>, h: usize, w: usize) -> Result<Tensor<f32>> {
    if d.h == h && d.w == w {
        return Ok(d.clone());
    }
    let mut out = Tensor::<f32>::zeros(d.c, h, w)?;
    for c in 0..d.c {
        for y in 0..d.h.min(h) {
            let n = d.w.min(w);
            out.plane_mut(c)[y * w..][..n].copy_from_slice(&d.plane(c)[y * d.w..][..n]);
        }
    }
    Ok(out)
}

/// The forward intermediates `LightPrimary`'s backward needs (RDLR VJP).
pub struct SynthTape {
    /// `res_conv`'s output before the ReLU (the residual-block input path).
    pub(crate) res_conv_out: BTensor,
    /// Each `ResAu`'s `(x, mask)`.
    pub(crate) act: [(BTensor, BTensor); 3],
    /// The picture before `denormalize` (for its clamp's backward).
    pub(crate) pre_denorm: Tensor<f32>,
    /// `act1`'s pre-crop dims (the `crop_par(8)` backward's pad target).
    pub(crate) a1_dims: (usize, usize),
    /// The `crop_par(8)` output dims — `up2`'s input.
    pub(crate) crop8_dims: (usize, usize),
    /// `up2`'s pre-crop dims (the `crop_par(4)` backward's pad target).
    pub(crate) up2_dims: (usize, usize),
}

/// Secondary (chroma) synthesis transform.
#[derive(Clone, Debug)]
pub struct SynthesisSecondary {
    /// `LightCombineBlock`: conv3x3(160 + 96 → 48); its output is `cat(info, chroma latent)`.
    combine: ConvLayer,
    tail: SecondaryTail,
}

#[derive(Clone, Debug)]
enum SecondaryTail {
    Light(alloc::boxed::Box<LightSecondary>),
    Hop(alloc::boxed::Box<HopSecondary>),
}

/// `DecoderHOPSec` behind the combine block.
#[derive(Clone, Debug)]
struct HopSecondary {
    up2: ConvTransposeLayer,
    cab: Cab,
    act2: ResAu,
    conv3: ConvLayer,
    tam: Tam,
    act3: ResAu,
    up4: ConvTransposeLayer,
}

impl HopSecondary {
    const C: usize = 64;

    fn load(ck: &Checkpoint<'_>, eng: &Engine) -> Result<Self> {
        let c = Self::C;
        Ok(Self {
            up2: conv3x3_t(ck, "conv2_t", 144, c, eng)?,
            cab: Cab::load(ck, "CAB", c, eng)?,
            act2: ResAu::load(ck, "iact2", c, eng)?,
            conv3: ConvLayer::new(load::conv1x1(ck, "conv3_t", c, 4 * c, true)?, eng)?,
            tam: Tam::load(ck, "TAM", c, false, eng)?,
            act3: ResAu::load(ck, "iact3", c, eng)?,
            up4: conv3x3_t(ck, "conv4_t", c, 8, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        x: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let mut x = self.up2.forward(eng, x)?;
        x = self.cab.forward(eng, &x)?;
        x = x.crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        x = self.act2.forward(eng, x)?;
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        x = fast::pixel_shuffle(eng, &x, 2)?;
        stop.check()?;
        x = self.tam.forward(eng, x)?;
        x = x.crop_par(eng, h.div_ceil(4), w.div_ceil(4))?;
        stop.check()?;
        x = self.act3.forward(eng, x)?;
        stop.check()?;
        x = self.up4.forward(eng, &x)?;
        fast::pixel_shuffle_to_planar(&x, 2)
    }
}

/// `DecoderSOPSec` / `DecoderBOPSec` behind the combine block.
#[derive(Clone, Debug)]
struct LightSecondary {
    up: Upsample,
    act2: ResAu,
    conv3: ConvLayer,
    act3: ResAu,
    conv4: ConvLayer,
}

impl LightSecondary {
    fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        let (c2, c3) = match op {
            OperatingPoint::Sop => (32, 32),
            OperatingPoint::Bop => (64, 128),
            OperatingPoint::Hop => {
                return Err(Error::InvalidArgument("HOP is not a light transform"));
            }
        };
        Ok(Self {
            up: Upsample::load(ck, op, "conv2_t", 144, c2, eng)?,
            act2: ResAu::load(ck, "iact2", c2, eng)?,
            conv3: ConvLayer::new(load::conv3x3(ck, "conv3", c2, c3, 1, true)?, eng)?,
            act3: ResAu::load(ck, "iact3", c3, eng)?,
            conv4: ConvLayer::new(load::conv1x1(ck, "conv4", c3, 128, false)?, eng)?,
        })
    }

    fn forward(
        &self,
        eng: &Engine,
        x: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
    ) -> Result<Tensor<f32>> {
        let mut x = self
            .up
            .forward(eng, x)?
            .crop_par(eng, h.div_ceil(8), w.div_ceil(8))?;
        stop.check()?;
        x = self.act2.forward(eng, x)?;
        stop.check()?;
        x = self.conv3.forward(eng, &x)?;
        stop.check()?;
        x = self.act3.forward(eng, x)?;
        stop.check()?;
        x = self.conv4.forward(eng, &x)?;
        fast::pixel_shuffle_to_planar(&x, 8)
    }
}

impl SynthesisSecondary {
    pub fn load(ck: &Checkpoint<'_>, op: OperatingPoint, eng: &Engine) -> Result<Self> {
        Ok(Self {
            combine: ConvLayer::new(
                load::conv3x3(ck, "first_stage.conv1", 256, 48, 1, true)?,
                eng,
            )?,
            tail: match op {
                OperatingPoint::Hop => {
                    SecondaryTail::Hop(alloc::boxed::Box::new(HopSecondary::load(ck, eng)?))
                }
                _ => {
                    SecondaryTail::Light(alloc::boxed::Box::new(LightSecondary::load(ck, op, eng)?))
                }
            },
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
        self.forward_with(eng, y_hat_luma, y_hat_chroma, h, w, &Unstoppable)
    }

    /// [`Self::forward`] that checks `stop` between layers.
    pub fn forward_with(
        &self,
        eng: &Engine,
        y_hat_luma: &BTensor,
        y_hat_chroma: &BTensor,
        h: usize,
        w: usize,
        stop: &dyn Stop,
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
        let x = if (y_hat_luma.h, y_hat_luma.w) == (lh, lw) {
            BTensor::cat(&[y_hat_luma, y_hat_chroma])?
        } else {
            BTensor::cat(&[&y_hat_luma.crop(lh, lw)?, y_hat_chroma])?
        };
        let info = self.combine.forward(eng, &x)?;
        drop(x);
        stop.check()?;
        let x = BTensor::cat(&[&info, y_hat_chroma])?;
        drop(info);
        let mut x = match &self.tail {
            SecondaryTail::Light(m) => m.forward(eng, &x, h, w, stop)?,
            SecondaryTail::Hop(m) => m.forward(eng, &x, h, w, stop)?,
        };
        denormalize(&mut x);
        Ok(x)
    }
}
