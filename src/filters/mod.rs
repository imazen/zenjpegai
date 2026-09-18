//! Enhancement post-filters, applied to the reconstructed YUV picture before the colour
//! transform.
//!
//! Ports the chain of `ref/src/codec/coding_tools/filters/` (`FiltersComposite.decompress`,
//! order from `cfg/pipeline.json`): EFE linear → eICCI → EFE non-linear → LEF. Each enabled
//! filter maps a [`FilterState`] to the next one; the state carries the picture and, for the EFE
//! pair, the alternative up-sampled picture the linear filter prepares for the non-linear one.
//!
//! Status: see `PORTING.md`. A stream that enables a filter that is not ported yet is rejected
//! with `Error::Unsupported` naming the filter.

pub mod efe_linear;
pub mod efe_nonlinear;
pub mod icci;
pub mod lef;

use super::decoder::reconstruct::Planes;
use crate::error::Result;
use crate::header::{PictureHeader, ToolHeader};
use crate::model::ModelSource;
use crate::nn::fast::Engine;
use crate::tensor::Tensor;

/// Everything a filter may read besides the picture.
pub struct FilterContext<'a> {
    pub eng: &'a Engine,
    pub hdr: &'a PictureHeader,
    pub tools: &'a ToolHeader,
    /// Luma sigma indices the residual was coded with (`decisions['model_y']['scale_log']`,
    /// RVS included): the LEF steers its sharpening with one channel of it.
    pub luma_scale_log: &'a Tensor<i32>,
    /// Source of the eICCI network checkpoints.
    pub models: &'a dyn ModelSource,
    /// Operating point the picture was synthesised with (selects the eICCI bank and short lists).
    pub op: crate::header::OperatingPoint,
    /// Loaded eICCI networks, kept between decodes.
    pub icci_nets: &'a icci::NetCache,
    /// Cooperative cancellation, checked once per enabled filter and, for eICCI, once per tile
    /// (it runs a whole network per tile).
    pub stop: &'a dyn enough::Stop,
}

/// What one filter hands to the next (`[img, upsampled_img]` in the reference).
pub struct FilterState {
    /// The picture: planes in `[0, 255]`, chroma in the source's subsampling format.
    pub image: Planes,
    /// EFE linear's second output, consumed by EFE non-linear.
    pub upsampled: Option<Planes>,
}

/// Run the enabled filters in the normative order.
pub fn apply(ctx: &FilterContext<'_>, image: Planes) -> Result<Planes> {
    let mut state = FilterState {
        image,
        upsampled: None,
    };
    if let Some(h) = &ctx.tools.efe_linear {
        ctx.stop.check()?;
        state = efe_linear::apply(ctx, h, state)?;
    }
    if let Some(h) = &ctx.tools.icci {
        ctx.stop.check()?;
        state = icci::apply(ctx, h, state)?;
    }
    if let Some(h) = &ctx.tools.efe_nonlinear {
        ctx.stop.check()?;
        state = efe_nonlinear::apply(ctx, h, state)?;
    }
    if let Some(channel) = ctx.tools.lef_channel {
        ctx.stop.check()?;
        state = lef::apply(ctx, channel, state)?;
    }
    Ok(state.image)
}
