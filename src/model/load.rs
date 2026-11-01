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

/// `clip_thres.<layer>` of a checkpoint — the per-channel `[MinList, MaxList]` bounds of
/// `base_layers/utils.py::feature_clipping` — or `None` when the checkpoint carries no
/// `clip_thres` for the layer (every shipped model file).
pub(crate) fn clip_bounds(ck: &Checkpoint<'_>, key: &str) -> Option<(Vec<f32>, Vec<f32>)> {
    let lo = ck.list(&format!("{key}.MinList"))?;
    let hi = ck.list(&format!("{key}.MaxList"))?;
    Some((
        lo.iter().map(|&x| x as f32).collect(),
        hi.iter().map(|&x| x as f32).collect(),
    ))
}

/// `feature_clipping(x, layer, clip_thres, is_clip)` on a blocked tensor: `x[c]` clamped to
/// `[lo[c], hi[c]]`, leaving channels the lists do not cover alone.
pub(crate) fn clip_channels(x: &mut crate::nn::fast::BTensor, bounds: &(Vec<f32>, Vec<f32>)) {
    let (lo, hi) = bounds;
    let (v, plane) = (x.v, x.h * x.w);
    for (blk, b) in x.data.chunks_exact_mut(plane * v).enumerate() {
        for lane in 0..v {
            let c = blk * v + lane;
            let (Some(&lo), Some(&hi)) = (lo.get(c), hi.get(c)) else {
                break;
            };
            for x in b[lane..].iter_mut().step_by(v) {
                *x = x.clamp(lo, hi);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::fast::BTensor;

    /// `clip_channels` clamps each channel to its `[MinList, MaxList]` entry across the
    /// blocked layout, leaving channels beyond the lists untouched.
    #[test]
    fn clip_channels_clamps_per_channel() {
        // c = 10 channels over v = 4 lanes → three blocks, the last one half-full.
        let mut x = BTensor::zeros(10, 2, 3, 4).unwrap();
        let plane = x.h * x.w;
        for c in 0..x.c {
            let (blk, lane) = (c / x.v, c % x.v);
            for i in 0..plane {
                x.data[blk * plane * x.v + i * x.v + lane] = c as f32 * 10.0;
            }
        }
        let lo: Vec<f32> = (0..10).map(|_| 20.0).collect();
        let hi: Vec<f32> = (0..10).map(|_| 70.0).collect();
        clip_channels(&mut x, &(lo, hi));
        for c in 0..x.c {
            let (blk, lane) = (c / x.v, c % x.v);
            let want = (c as f32 * 10.0).clamp(20.0, 70.0);
            for i in 0..plane {
                assert_eq!(x.data[blk * plane * x.v + i * x.v + lane], want, "ch {c}");
            }
        }
    }

    /// A `MinList`/`MaxList` shorter than the channel count leaves the tail channels alone.
    #[test]
    fn clip_channels_short_lists_skip_tail() {
        let mut x = BTensor::zeros(10, 1, 4, 4).unwrap();
        let plane = x.h * x.w;
        for c in 0..x.c {
            let (blk, lane) = (c / x.v, c % x.v);
            for i in 0..plane {
                x.data[blk * plane * x.v + i * x.v + lane] = c as f32 * 10.0;
            }
        }
        // Lists only cover channels 0..5 — channels 5..9 stay unclamped.
        let lo = vec![0.0; 5];
        let hi = vec![30.0; 5];
        clip_channels(&mut x, &(lo, hi));
        for c in 0..x.c {
            let (blk, lane) = (c / x.v, c % x.v);
            let want = if c < 5 {
                (c as f32 * 10.0).clamp(0.0, 30.0)
            } else {
                c as f32 * 10.0
            };
            for i in 0..plane {
                assert_eq!(x.data[blk * plane * x.v + i * x.v + lane], want, "ch {c}");
            }
        }
    }
}
