//! The analysis transforms (input planes -> latents) on the GPU: BOP and HOP.
//!
//! Layer structure and tensor names mirror `zenjpegai::model::analysis`: the
//! trunk is `[pad -> conv3x3_s2 -> ResAu] x3` (HOP inserts a TAM before the
//! second conv and a CAB before the third), the primary then pads, applies a
//! fourth stride-2 conv and a 1x1; the secondary drops the fourth conv.

use std::sync::Arc;

use zenjpegai::header::OperatingPoint;
use zenjpegai::model::{ModelSource, analysis_path};
use zenjpegai::weights::Checkpoint;

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::layers::GpuConv;
use crate::plan::{Graph, T};
use crate::synthesis::{Cab, Loader, ResAu, Tam};

/// Feature width of the trunk (`zenjpegai::model::analysis::C`).
const C: usize = 128;

/// `padding_layer` rows/cols for a picture of `len` samples at `divider`.
fn pad_amount(len: usize, divider: usize) -> usize {
    2 * len.div_ceil(2 * divider) - len.div_ceil(divider)
}

/// `conv3x3 stride 2` (`conv2d` helper is `c -> c` square only, so the head's
/// `conv4` goes through `conv2d` directly).
fn conv_s2(l: &Loader<'_>, prefix: &str, i: usize, o: usize) -> Result<GpuConv> {
    GpuConv::new(l.ctx, &l.conv2d(prefix, i, o, 3, 2, 1, 1, true)?)
}

/// The three `[conv -> ResAu]` stages both transforms share.
struct Trunk {
    conv1: GpuConv,
    conv2: GpuConv,
    conv3: GpuConv,
    act1: ResAu,
    act2: ResAu,
    act3: ResAu,
    tam: Option<Tam>,
    cab: Option<Cab>,
}

impl Trunk {
    fn load(
        l: &Loader<'_>,
        op: OperatingPoint,
        in_ch: usize,
        tam_downsamples: bool,
    ) -> Result<Self> {
        let hop = op == OperatingPoint::Hop;
        Ok(Self {
            conv1: conv_s2(l, "conv1", in_ch, C)?,
            act1: ResAu::load(l, "act1", C)?,
            tam: hop
                .then(|| Tam::load(l, "TAM", C, tam_downsamples))
                .transpose()?,
            conv2: conv_s2(l, "conv2", C, C)?,
            act2: ResAu::load(l, "act2", C)?,
            cab: hop.then(|| Cab::load(l, "cab", C)).transpose()?,
            conv3: conv_s2(l, "conv3", C, C)?,
            act3: ResAu::load(l, "act3", C)?,
        })
    }

    /// `x` is the normalized picture at `h x w`; out at `/8`, padded at
    /// every stage.
    fn forward(&self, g: &mut Graph<'_>, x: T, h: usize, w: usize) -> Result<T> {
        let x = g.pad(x, pad_amount(h, 1), pad_amount(w, 1))?;
        let x = g.conv(x, &self.conv1)?;
        let x = self.act1.forward(g, x)?;
        let x = g.pad(x, pad_amount(h, 2), pad_amount(w, 2))?;
        let x = if let Some(t) = &self.tam {
            t.forward(g, x)?
        } else {
            x
        };
        let x = g.conv(x, &self.conv2)?;
        let x = self.act2.forward(g, x)?;
        let x = g.pad(x, pad_amount(h, 4), pad_amount(w, 4))?;
        let x = if let Some(c) = &self.cab {
            c.forward(g, x)?
        } else {
            x
        };
        let x = g.conv(x, &self.conv3)?;
        self.act3.forward(g, x)
    }
}

/// Luma (`[1, h, w] -> [160, h/16, w/16]`): the trunk plus `conv4`/`conv5`.
struct Primary {
    trunk: Trunk,
    conv4: GpuConv,
    conv5: GpuConv,
}

impl Primary {
    fn load(l: &Loader<'_>, op: OperatingPoint) -> Result<Self> {
        let chs = zenjpegai::header::LATENT_CHANNELS[0];
        Ok(Self {
            trunk: Trunk::load(l, op, 1, true)?,
            conv4: conv_s2(l, "conv4", C, chs)?,
            conv5: l.conv1x1("conv5", chs, chs, false)?,
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T, h: usize, w: usize) -> Result<T> {
        let x = self.trunk.forward(g, x, h, w)?;
        let x = g.pad(x, pad_amount(h, 8), pad_amount(w, 8))?;
        let x = g.conv(x, &self.conv4)?;
        g.conv(x, &self.conv5)
    }
}

/// Chroma (`[12, h, w] -> [96, h/8, w/8]`): the trunk plus `conv5`.
struct Secondary {
    trunk: Trunk,
    conv5: GpuConv,
}

impl Secondary {
    fn load(l: &Loader<'_>, op: OperatingPoint) -> Result<Self> {
        let chs = zenjpegai::header::LATENT_CHANNELS[1];
        Ok(Self {
            trunk: Trunk::load(l, op, 12, false)?,
            conv5: l.conv1x1("conv5", C, chs, false)?,
        })
    }

    fn forward(&self, g: &mut Graph<'_>, x: T, h: usize, w: usize) -> Result<T> {
        let x = self.trunk.forward(g, x, h, w)?;
        g.conv(x, &self.conv5)
    }
}

/// The analysis transforms on the GPU, one for each channel class.
pub struct GpuAnalysis {
    ctx: Arc<GpuContext>,
    op: OperatingPoint,
    luma: Primary,
    chroma: Secondary,
}

impl GpuAnalysis {
    /// Load `VM_{bop,hop}/encoder_{Y,UV}_<beta>.pth` of `model_id` from
    /// `source` and upload the packed weights.
    pub fn load(
        ctx: Arc<GpuContext>,
        source: &dyn ModelSource,
        model_id: usize,
        op: OperatingPoint,
    ) -> Result<Self> {
        let luma = {
            let rel = analysis_path(model_id, 0, op)?;
            let file = source
                .read(&rel)
                .map_err(|e| GpuError::Codec(zenjpegai::Error::Model(e.to_string())))?;
            let ck = Checkpoint::parse(&file)?;
            let net = Primary::load(&Loader { ctx: &ctx, ck: &ck }, op)?;
            source.accessed(&rel, &ck.touched_names());
            net
        };
        let chroma = {
            let rel = analysis_path(model_id, 1, op)?;
            let file = source
                .read(&rel)
                .map_err(|e| GpuError::Codec(zenjpegai::Error::Model(e.to_string())))?;
            let ck = Checkpoint::parse(&file)?;
            let net = Secondary::load(&Loader { ctx: &ctx, ck: &ck }, op)?;
            source.accessed(&rel, &ck.touched_names());
            net
        };
        Ok(Self {
            ctx,
            op,
            luma,
            chroma,
        })
    }

    pub fn operating_point(&self) -> OperatingPoint {
        self.op
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.ctx
    }

    /// Luma analysis: `g` gets `[1, h, w]` input -> `[160, ceil(h/16),
    /// ceil(w/16)]` latent.
    pub fn luma(&self, g: &mut Graph<'_>, x: T, h: usize, w: usize) -> Result<T> {
        self.luma.forward(g, x, h, w)
    }

    /// Chroma analysis: `[12, h, w]` (half-luma resolution, luma planes
    /// first) -> `[96, ceil(h/8), ceil(w/8)]` latent.
    pub fn chroma(&self, g: &mut Graph<'_>, x: T, h: usize, w: usize) -> Result<T> {
        self.chroma.forward(g, x, h, w)
    }
}
