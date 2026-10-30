//! Building blocks of the HOP synthesis transforms: `ResidualBlock`, the convolutional attention
//! block (`CAB`) and the transformer attention module (`TAM`).
//!
//! Ports `ref/src/codec/components/base_layers/{conv_layers.py::ResidualBlock, cab.py, tam.py}`.

use alloc::format;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::model::load;
use crate::nn::fast::{self, BTensor, ConvLayer, ConvTransposeLayer, Engine, math};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `conv3x3 -> ReLU -> conv3x3`, plus the input.
#[derive(Clone, Debug)]
pub(crate) struct ResidualBlock {
    conv1: ConvLayer,
    conv2: ConvLayer,
}

impl ResidualBlock {
    pub fn load(ck: &Checkpoint<'_>, prefix: &str, chs: usize, eng: &Engine) -> Result<Self> {
        let c = |name: &str| -> Result<ConvLayer> {
            ConvLayer::new(
                load::conv3x3(ck, &format!("{prefix}.{name}"), chs, chs, 1, true)?,
                eng,
            )
        };
        Ok(Self {
            conv1: c("conv1")?,
            conv2: c("conv2")?,
        })
    }

    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut t = self.conv1.forward(eng, x)?;
        fast::relu(&mut t);
        let mut out = self.conv2.forward(eng, &t)?;
        drop(t);
        fast::add_assign(&mut out, x)?;
        Ok(out)
    }

    /// [`Self::forward`] that also returns `conv1`'s output before the ReLU — the
    /// mask `backward` needs. `out = conv2(relu(conv1(x))) + x`. RDLR HOP VJP —
    /// used by `Cab`/`HopPrimary::forward_tape` (wired up next).
    #[allow(dead_code)]
    pub fn forward_tape(&self, eng: &Engine, x: &BTensor) -> Result<(BTensor, BTensor)> {
        let mut t = self.conv1.forward(eng, x)?;
        let conv1_out = t.clone();
        fast::relu(&mut t);
        let mut out = self.conv2.forward(eng, &t)?;
        drop(t);
        fast::add_assign(&mut out, x)?;
        Ok((out, conv1_out))
    }

    /// `d_out` (planar) → `d_x` (planar): the skip contributes `d_out` verbatim,
    /// the trunk `conv1^T(relu'(conv1_out) · conv2^T d_out)`. Same-size convs —
    /// `h,w` are the input's throughout. RDLR HOP VJP — used by `Cab::backward`.
    #[allow(dead_code)]
    pub fn backward(&self, d_out: &Tensor<f32>, conv1_out: &BTensor) -> Result<Tensor<f32>> {
        let (h, w) = (conv1_out.h, conv1_out.w);
        let mut d = self.conv2.backward(d_out, h, w)?;
        let c1 = conv1_out.to_planar()?;
        for (g, &p) in d.data.iter_mut().zip(&c1.data) {
            if p <= 0.0 {
                *g = 0.0;
            }
        }
        let mut dx = self.conv1.backward(&d, h, w)?;
        for (a, &b) in dx.data.iter_mut().zip(&d_out.data) {
            *a += b;
        }
        Ok(dx)
    }
}

/// `conv3x3_t`: transposed convolution, kernel 3, stride 2, padding 1, output padding 1 (exactly
/// doubles the size), with bias.
pub(crate) fn conv3x3_t(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    eng: &Engine,
) -> Result<ConvTransposeLayer> {
    ConvTransposeLayer::new(
        load::conv_transpose(ck, prefix, in_ch, out_ch, 3, 2, 1, 1)?,
        eng,
    )
}

fn conv3x3_s2(ck: &Checkpoint<'_>, prefix: &str, chs: usize, eng: &Engine) -> Result<ConvLayer> {
    ConvLayer::new(
        load::conv(ck, prefix, chs, chs, (3, 3), 2, (1, 1), 1, true)?,
        eng,
    )
}

/// `CAB`: `x + trunk(x) * sigmoid(mask(x))`, the mask computed at half resolution.
#[derive(Clone, Debug)]
pub(crate) struct Cab {
    trunk: [ResidualBlock; 2],
    subscale: ConvLayer,
    mask2: ResidualBlock,
    mask3: ResidualBlock,
    upscale: ConvTransposeLayer,
}

impl Cab {
    pub fn load(ck: &Checkpoint<'_>, prefix: &str, chs: usize, eng: &Engine) -> Result<Self> {
        let rb = |name: &str| ResidualBlock::load(ck, &format!("{prefix}.{name}"), chs, eng);
        Ok(Self {
            trunk: [rb("residual_trunk.0")?, rb("residual_trunk.1")?],
            subscale: conv3x3_s2(ck, &format!("{prefix}.subscale"), chs, eng)?,
            mask2: rb("residual_mask2.0")?,
            mask3: rb("residual_mask3.0")?,
            upscale: conv3x3_t(ck, &format!("{prefix}.upscale"), chs, chs, eng)?,
        })
    }

    /// The reference's `gama` argument is always 1 in the decoder, so it is not a parameter.
    pub fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        // `x = f(&x)` frees each map as soon as the next exists; see `synthesis.rs`.
        let mut trunk = self.trunk[0].forward(eng, x)?;
        trunk = self.trunk[1].forward(eng, &trunk)?;
        let mut m = self.subscale.forward(eng, x)?;
        m = self.mask2.forward(eng, &m)?;
        m = self.mask3.forward(eng, &m)?;
        m = self.upscale.forward(eng, &m)?;
        if (m.h, m.w) != (trunk.h, trunk.w) {
            // Odd input sizes: the reference would fail on the shape mismatch as well.
            return Err(Error::InvalidArgument("CAB: odd feature map size"));
        }
        math::sigmoid(eng, &mut m.data);
        math::mul_assign(eng, &mut trunk.data, &m.data)?;
        fast::add_assign(&mut trunk, x)?;
        Ok(trunk)
    }

    /// [`Self::forward`] recording what `backward` needs: the pre-multiply trunk,
    /// the post-sigmoid mask, each `ResidualBlock`'s `conv1` output, and the
    /// half-resolution mask input's dims. RDLR HOP VJP — used by
    /// `HopPrimary::forward_tape` (wired up next).
    #[allow(dead_code)]
    #[allow(clippy::type_complexity)]
    pub fn forward_tape(&self, eng: &Engine, x: &BTensor) -> Result<(BTensor, CabTape)> {
        let (mut trunk, rb0_c1) = self.trunk[0].forward_tape(eng, x)?;
        let (t1, rb1_c1) = self.trunk[1].forward_tape(eng, &trunk)?;
        trunk = t1;
        let trunk_pre = trunk.clone();
        let mut m = self.subscale.forward(eng, x)?;
        let (m2, m2_c1) = self.mask2.forward_tape(eng, &m)?;
        let (m3, m3_c1) = self.mask3.forward_tape(eng, &m2)?;
        m = m3;
        let mask_dims = (m.h, m.w);
        m = self.upscale.forward(eng, &m)?;
        if (m.h, m.w) != (trunk.h, trunk.w) {
            return Err(Error::InvalidArgument("CAB: odd feature map size"));
        }
        math::sigmoid(eng, &mut m.data);
        let sig = m.clone();
        math::mul_assign(eng, &mut trunk.data, &m.data)?;
        fast::add_assign(&mut trunk, x)?;
        Ok((
            trunk,
            CabTape {
                trunk_pre,
                sig,
                rb_c1: [rb0_c1, rb1_c1],
                mask_c1: [m2_c1, m3_c1],
                mask_dims,
            },
        ))
    }

    /// `d_out` (planar) → `d_x`: `out = trunk·sigmoid(m) + x`. The trunk branch
    /// gets `d_out·sigmoid(m)`, the mask branch `d_out·trunk` through `σ'` and
    /// the mask chain back to full resolution, the skip `d_out` verbatim. RDLR
    /// HOP VJP — used by `HopPrimary::backward` (wired up next).
    #[allow(dead_code)]
    pub fn backward(&self, d_out: &Tensor<f32>, tape: &CabTape) -> Result<Tensor<f32>> {
        let (h, w) = (tape.trunk_pre.h, tape.trunk_pre.w);
        let trunk = tape.trunk_pre.to_planar()?;
        let sig = tape.sig.to_planar()?;
        // d_trunk = d_out · sig ; d_sig = d_out · trunk (elementwise).
        let mut d_trunk = d_out.clone();
        let mut d_sig = d_out.clone();
        for (dt, (&s, &o)) in d_trunk
            .data
            .iter_mut()
            .zip(sig.data.iter().zip(&d_out.data))
        {
            *dt = o * s;
        }
        for (ds, (&t, &o)) in d_sig
            .data
            .iter_mut()
            .zip(trunk.data.iter().zip(&d_out.data))
        {
            *ds = o * t;
        }
        // Trunk branch: res[1] then res[0] backwards.
        let d = self.trunk[1].backward(&d_trunk, &tape.rb_c1[1])?;
        let d = self.trunk[0].backward(&d, &tape.rb_c1[0])?;
        let mut dx = d;
        // Mask branch: σ'(m) = sig·(1−sig) on d_sig, then upscale (half res
        // input) → mask3 → mask2 → subscale (back to full res).
        for (ds, &s) in d_sig.data.iter_mut().zip(&sig.data) {
            *ds *= s * (1.0 - s);
        }
        let (mh, mw) = tape.mask_dims;
        let d = self.upscale.backward(&d_sig, mh, mw)?;
        let d = self.mask3.backward(&d, &tape.mask_c1[1])?;
        let d = self.mask2.backward(&d, &tape.mask_c1[0])?;
        let d = self.subscale.backward(&d, h, w)?;
        for (a, &b) in dx.data.iter_mut().zip(&d.data) {
            *a += b;
        }
        // Skip: `+ x`.
        for (a, &b) in dx.data.iter_mut().zip(&d_out.data) {
            *a += b;
        }
        Ok(dx)
    }
}

/// `Cab::backward`'s forward intermediates.
pub(crate) struct CabTape {
    /// `trunk` before the sigmoid mask multiply.
    trunk_pre: BTensor,
    /// `sigmoid(mask)`, upscaled to trunk size.
    sig: BTensor,
    /// Each trunk `ResidualBlock`'s `conv1` output.
    rb_c1: [BTensor; 2],
    /// `mask2`/`mask3`'s `conv1` outputs.
    mask_c1: [BTensor; 2],
    /// `upscale`'s input dims (half resolution).
    mask_dims: (usize, usize),
}

#[derive(Clone, Debug)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    /// `weight` + `bias` bytes in the tracked ledger (see [`crate::mem`]).
    _charge: crate::mem::Charge,
}

impl LayerNorm {
    fn load(ck: &Checkpoint<'_>, prefix: &str, dim: usize) -> Result<Self> {
        let weight = ck.f32(&format!("{prefix}.weight"))?;
        let bias = ck.f32(&format!("{prefix}.bias"))?;
        if weight.shape != [dim] || bias.shape != [dim] {
            return Err(Error::Model(format!(
                "{prefix}: unexpected layer norm shape"
            )));
        }
        let (weight, bias) = (weight.data, bias.data);
        let charge =
            crate::mem::Charge::new(crate::mem::vec_bytes(&weight) + crate::mem::vec_bytes(&bias));
        Ok(Self {
            weight,
            bias,
            _charge: charge,
        })
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut p = x.to_planar_par(eng)?;
        math::layer_norm_channels(&mut p, &self.weight, &self.bias)?;
        BTensor::from_planar_par(eng, &p, x.v)
    }
}

const HEADS: usize = 4;
const FFN_GAMMA: usize = 4;

/// `TransformerBlock`: channel attention and a gated feed-forward, each behind a layer norm and
/// added to the input. All convolutions are bias-free.
#[derive(Clone, Debug)]
struct TransformerBlock {
    prep_norm: LayerNorm,
    prep_conv1: ConvLayer,
    prep_conv2: ConvLayer,
    temperature: Vec<f32>,
    attn_out: ConvLayer,
    ffn_norm: LayerNorm,
    ffn_in: ConvLayer,
    ffn_dw: ConvLayer,
    ffn_out: ConvLayer,
    hidden: usize,
    /// `temperature`'s bytes in the tracked ledger (see [`crate::mem`]).
    _charge: crate::mem::Charge,
}

impl TransformerBlock {
    fn load(ck: &Checkpoint<'_>, prefix: &str, dim: usize, eng: &Engine) -> Result<Self> {
        let hidden = dim * FFN_GAMMA;
        let p = |name: &str| format!("{prefix}.{name}");
        let temperature = ck.f32(&p("attn.temperature"))?;
        if temperature.shape != [HEADS, 1, 1] {
            return Err(Error::Model(format!(
                "{prefix}: unexpected temperature shape"
            )));
        }
        let pw = |name: &str, i: usize, o: usize| -> Result<ConvLayer> {
            ConvLayer::new(load::conv1x1(ck, &p(name), i, o, false)?, eng)
        };
        let dw = |name: &str, c: usize| -> Result<ConvLayer> {
            ConvLayer::new(load::conv3x3(ck, &p(name), c, c, c, false)?, eng)
        };
        let temperature = temperature.data;
        let charge = crate::mem::Charge::of_vec(&temperature);
        Ok(Self {
            prep_norm: LayerNorm::load(ck, &p("prep_data.norm1"), dim)?,
            prep_conv1: pw("prep_data.conv1", dim, 3 * dim)?,
            prep_conv2: dw("prep_data.conv2", 3 * dim)?,
            temperature,
            attn_out: pw("attn.project_out", dim, dim)?,
            ffn_norm: LayerNorm::load(ck, &p("ffn.norm1"), dim)?,
            ffn_in: pw("ffn.project_in", dim, 2 * hidden)?,
            ffn_dw: dw("ffn.dwconv", 2 * hidden)?,
            ffn_out: pw("ffn.project_out", hidden, dim)?,
            hidden,
            _charge: charge,
        })
    }

    fn forward(&self, eng: &Engine, mut x: BTensor) -> Result<BTensor> {
        // x = x + attn(prep(x))
        // Each map is freed as soon as the next exists: the 3x and 8x wide ones here are the
        // largest allocations of a HOP decode.
        let mut t = self.prep_norm.forward(eng, &x)?;
        t = self.prep_conv1.forward(eng, &t)?;
        t = self.prep_conv2.forward(eng, &t)?;
        let planar = t.to_planar()?;
        drop(t);
        let a = math::channel_attention(eng, &planar, HEADS, &self.temperature)?;
        drop(planar);
        let blocked = BTensor::from_planar(&a, x.v)?;
        drop(a);
        let a = self.attn_out.forward(eng, &blocked)?;
        drop(blocked);
        fast::add_assign(&mut x, &a)?;
        drop(a);

        // x = x + ffn(x): elu(first half) * second half
        let mut t = self.ffn_norm.forward(eng, &x)?;
        t = self.ffn_in.forward(eng, &t)?;
        t = self.ffn_dw.forward(eng, &t)?;
        let f = if self.hidden.is_multiple_of(t.v) {
            // Channel blocks are the outermost axis, so the two halves are two slices.
            let n = self.hidden / t.v * t.h * t.w * t.v;
            let (x1, x2) = t.data.split_at_mut(n);
            math::elu_gate(eng, x1, x2)?;
            t.data.truncate(n);
            t.c = self.hidden;
            self.ffn_out.forward(eng, &t)?
        } else {
            let mut p = t.to_planar_par(eng)?;
            drop(t);
            let n = self.hidden * p.h * p.w;
            let (x1, x2) = p.data.split_at_mut(n);
            math::elu_gate(eng, x1, x2)?;
            p.data.truncate(n);
            p.c = self.hidden;
            let b = BTensor::from_planar_par(eng, &p, x.v)?;
            drop(p);
            self.ffn_out.forward(eng, &b)?
        };
        fast::add_assign(&mut x, &f)?;
        Ok(x)
    }
}

/// `TAM`: two transformer blocks, optionally run at half resolution between a stride-2
/// convolution and a transposed convolution.
#[derive(Clone, Debug)]
pub(crate) struct Tam {
    resample: Option<(ConvLayer, ConvTransposeLayer)>,
    blocks: [TransformerBlock; 2],
}

impl Tam {
    pub fn load(
        ck: &Checkpoint<'_>,
        prefix: &str,
        dim: usize,
        downsample: bool,
        eng: &Engine,
    ) -> Result<Self> {
        let resample = if downsample {
            Some((
                conv3x3_s2(ck, &format!("{prefix}.ds_conv"), dim, eng)?,
                conv3x3_t(ck, &format!("{prefix}.us_conv"), dim, dim, eng)?,
            ))
        } else {
            None
        };
        let tb = |i: usize| TransformerBlock::load(ck, &format!("{prefix}.TABs.{i}"), dim, eng);
        Ok(Self {
            resample,
            blocks: [tb(0)?, tb(1)?],
        })
    }

    pub fn forward(&self, eng: &Engine, x: BTensor) -> Result<BTensor> {
        let x = match &self.resample {
            Some((ds, _)) => {
                let t = ds.forward(eng, &x)?;
                drop(x);
                t
            }
            None => x,
        };
        let x = self.blocks[0].forward(eng, x)?;
        let x = self.blocks[1].forward(eng, x)?;
        match &self.resample {
            Some((_, us)) => us.forward(eng, &x),
            None => Ok(x),
        }
    }
}
