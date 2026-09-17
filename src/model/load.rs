//! Helpers for pulling layers out of a checkpoint with shape validation.

use alloc::format;

use crate::error::{Error, Result};
use crate::nn::{Conv2d, ConvTranspose2d};
use crate::weights::Checkpoint;

/// `nn.Conv2d` stored as `<prefix>.weight` (+ `<prefix>.bias`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: (usize, usize),
    stride: usize,
    pad: (usize, usize),
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let w = ck.f32(&format!("{prefix}.weight"))?;
    if w.shape != [out_ch, in_ch / groups, k.0, k.1] {
        return Err(Error::Model(format!(
            "{prefix}.weight: shape {:?}, expected [{out_ch}, {}, {}, {}]",
            w.shape,
            in_ch / groups,
            k.0,
            k.1
        )));
    }
    let b = if bias {
        Some(ck.f32(&format!("{prefix}.bias"))?.data)
    } else {
        None
    };
    if !bias && ck.contains(&format!("{prefix}.bias")) {
        return Err(Error::Model(format!(
            "{prefix}: unexpected bias in checkpoint"
        )));
    }
    Conv2d::new(in_ch, out_ch, k, stride, pad, groups, w.data, b)
}

/// 3x3, stride 1, padding 1.
pub(crate) fn conv3x3(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    conv(ck, prefix, in_ch, out_ch, (3, 3), 1, (1, 1), groups, bias)
}

/// 1x1, stride 1, no padding.
pub(crate) fn conv1x1(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    bias: bool,
) -> Result<Conv2d> {
    conv(ck, prefix, in_ch, out_ch, (1, 1), 1, (0, 0), 1, bias)
}

/// `nn.ConvTranspose2d` with bias, weight `[in_ch, out_ch, k, k]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
    out_pad: usize,
) -> Result<ConvTranspose2d> {
    let w = ck.f32(&format!("{prefix}.weight"))?;
    if w.shape != [in_ch, out_ch, k, k] {
        return Err(Error::Model(format!(
            "{prefix}.weight: shape {:?}, expected [{in_ch}, {out_ch}, {k}, {k}]",
            w.shape
        )));
    }
    let b = ck.f32(&format!("{prefix}.bias"))?.data;
    ConvTranspose2d::new(in_ch, out_ch, k, stride, pad, out_pad, w.data, Some(b))
}
