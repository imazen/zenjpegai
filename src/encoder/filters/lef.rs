//! LEF ("luma edge filter"), encode side: choosing the reference channel.
//!
//! Ports `LEFfilter.py::analyze` (called from `LEF.compress`): `LEF_chIdx` is the channel of
//! the luma sigma-index map with the highest mean. The sharpening itself is decoder-only
//! (`crate::filters::lef`); the encoder just signals the channel.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// `LEF.analyze`: `torch.argmax(torch.mean(scale_log, dim=[2, 3]))` over the luma
/// `scale_log` (`decisions['model_y']['scale_log']`, RVS included — the same map
/// `crate::filters::lef::sharpen` reads on the decoder side).
///
/// The mean is accumulated in single precision, like `tools::rvs::grfs_flags`, the other
/// consumer of these per-channel means. `torch.argmax` returns the first maximum, so a tie
/// goes to the lowest channel index. `LEF_chIdx` is coded with `max_symbol_value = 255`.
pub fn reference_channel(scale_log: &Tensor<i32>) -> Result<u8> {
    if scale_log.c == 0 || scale_log.h == 0 || scale_log.w == 0 {
        return Err(Error::InvalidData("LEF: empty luma scale map"));
    }
    if scale_log.c > 256 {
        return Err(Error::InvalidData(
            "LEF: more channels than LEF_chIdx can signal",
        ));
    }
    let n = (scale_log.h * scale_log.w) as f32;
    let mut best = (0usize, f32::NEG_INFINITY);
    for ch in 0..scale_log.c {
        let mean = scale_log.plane(ch).iter().map(|&v| v as f32).sum::<f32>() / n;
        if mean > best.1 {
            best = (ch, mean);
        }
    }
    Ok(best.0 as u8)
}
