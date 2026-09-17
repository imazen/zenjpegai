//! The eICCI enhancement network (`ThreeStageYUVLite_DWT_V8_444_jointUV`).
//!
//! Ports `ref/src/codec/coding_tools/filters/eICCI/icci_models.py` and
//! `base_layers/conv_layers.py::ResidualBlock_BN_RectKernel`. The network works on a two-level
//! Haar decomposition of the picture: every plane becomes 16 sub-bands at quarter resolution, the
//! 48 sub-bands of Y, U and V feed two small residual trunks (luma: 2 blocks, chroma: 4 blocks,
//! 48 features), and each trunk predicts a correction of its plane's 16 sub-bands.
//!
//! ```text
//! trunk(x) = blocks(relu(BN(conv3x3(x))))          block(f) = f + s * BN(conv3x1(relu(BN(conv1x3(f)))))
//! Y16' = Y16 + sY * conv3x3(trunk_Y(YUV16))        U16' / V16' likewise from the shared chroma trunk
//! ```
//!
//! Inference-mode batch norm is an affine map per channel, so it is folded into the convolution
//! in front of it, together with the learned residual scale behind it (`s`, `sY`, `sU`, `sV`):
//! `w' = w * a`, `b' = (b - mean) * a + beta` with `a = gamma / sqrt(var + eps) * s`, computed
//! in `f64` and rounded once. PyTorch applies the three steps one after another in `f32`; the
//! difference is part of the bound recorded in `PORTING.md`.

use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::header::OperatingPoint;
use crate::model::{ModelSource, load};
use crate::nn::fast::{self, BTensor, ConvLayer, Engine};
use crate::tensor::Tensor;
use crate::weights::Checkpoint;

/// `cfg/pipeline.json`, `post_filters.eICCI`: feature channels and residual blocks per trunk.
pub const FEATURES: usize = 48;
pub const LUMA_BLOCKS: usize = 2;
pub const CHROMA_BLOCKS: usize = 4;
/// Sub-bands of the three planes after two Haar levels.
pub const BANDS: usize = 16;
/// `post_filters.eICCI.ckpt_model_name`.
pub const CHECKPOINT_DIR: &str = "eICCI_bophop_2d020448_20240229";
/// `nn.BatchNorm2d` default.
const BN_EPS: f64 = 1e-5;

/// Checkpoint of network `index` (`0..10`) of the bank serving `op`. The bank of the base
/// operating point also serves the simple one (`eicci_type = 'bop' if base_model_type == 'sop'`).
/// Order as in `post_filters.eICCI.ckpt_files`: five MSE-trained networks, then five MS-SSIM ones.
pub fn checkpoint_path(op: OperatingPoint, index: usize) -> Result<String> {
    const RATES: [&str; 5] = ["012", "025", "050", "075", "100"];
    if index >= 2 * RATES.len() {
        return Err(Error::InvalidData("eICCI: network index out of range"));
    }
    let bank = match op {
        OperatingPoint::Hop => "hop",
        OperatingPoint::Sop | OperatingPoint::Bop => "bop",
    };
    let loss = if index < RATES.len() { "mse" } else { "mss" };
    Ok(format!(
        "{CHECKPOINT_DIR}/eicci_{bank}_{loss}{}_300k.pth",
        RATES[index % RATES.len()]
    ))
}

/// Per-channel affine map of an inference-mode batch norm times a scalar: `(a, shift)` with
/// `bn(x) * s = a * x + shift`.
fn batch_norm(ck: &Checkpoint<'_>, prefix: &str, ch: usize, s: f64) -> Result<Vec<(f64, f64)>> {
    let get = |what: &str| -> Result<Vec<f32>> {
        let t = ck.f32(&format!("{prefix}.{what}"))?;
        if t.shape != [ch] {
            return Err(Error::Model(format!(
                "{prefix}.{what}: shape {:?}",
                t.shape
            )));
        }
        Ok(t.data)
    };
    let (gamma, beta, mean, var) = (
        get("weight")?,
        get("bias")?,
        get("running_mean")?,
        get("running_var")?,
    );
    Ok((0..ch)
        .map(|c| {
            let a = gamma[c] as f64 / libm::sqrt(var[c] as f64 + BN_EPS);
            (a * s, (beta[c] as f64 - mean[c] as f64 * a) * s)
        })
        .collect())
}

fn scalar(ck: &Checkpoint<'_>, name: &str) -> Result<f64> {
    let t = ck.f32(name)?;
    match t.data.as_slice() {
        [v] => Ok(*v as f64),
        _ => Err(Error::Model(format!("{name}: not a scalar"))),
    }
}

/// Convolution `prefix` followed by the per-output-channel affine map `post`.
#[allow(clippy::too_many_arguments)]
fn folded_conv(
    ck: &Checkpoint<'_>,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: (usize, usize),
    post: &[(f64, f64)],
    eng: &Engine,
) -> Result<ConvLayer> {
    let mut conv = load::conv(ck, prefix, in_ch, out_ch, k, 1, (k.0 / 2, k.1 / 2), 1, true)?;
    let per_out = in_ch * k.0 * k.1;
    let bias = conv
        .bias
        .as_mut()
        .ok_or_else(|| Error::Model(format!("{prefix}: no bias")))?;
    for (oc, &(a, shift)) in post.iter().enumerate() {
        for w in &mut conv.weight[oc * per_out..][..per_out] {
            *w = (*w as f64 * a) as f32;
        }
        bias[oc] = (bias[oc] as f64 * a + shift) as f32;
    }
    ConvLayer::new(conv, eng)
}

/// `conv_first` + BN + ReLU, then the residual blocks.
#[derive(Clone, Debug)]
struct Trunk {
    first: ConvLayer,
    /// `(conv1x3 + BN1, conv3x1 + BN2 * scale)` per block.
    blocks: Vec<(ConvLayer, ConvLayer)>,
}

impl Trunk {
    fn load(
        ck: &Checkpoint<'_>,
        first: &str,
        bn: &str,
        hidden: &str,
        blocks: usize,
        eng: &Engine,
    ) -> Result<Self> {
        let nf = FEATURES;
        if ck.contains(&format!("{hidden}.{blocks}.scale")) {
            return Err(Error::Model(format!(
                "{hidden}: more than {blocks} residual blocks"
            )));
        }
        let first = folded_conv(
            ck,
            first,
            3 * BANDS,
            nf,
            (3, 3),
            &batch_norm(ck, bn, nf, 1.0)?,
            eng,
        )?;
        let blocks = (0..blocks)
            .map(|i| {
                let p = format!("{hidden}.{i}");
                let s = scalar(ck, &format!("{p}.scale"))?;
                let bn1 = batch_norm(ck, &format!("{p}.BN1"), nf, 1.0)?;
                let bn2 = batch_norm(ck, &format!("{p}.BN2"), nf, s)?;
                Ok((
                    folded_conv(ck, &format!("{p}.conv1"), nf, nf, (1, 3), &bn1, eng)?,
                    folded_conv(ck, &format!("{p}.conv2"), nf, nf, (3, 1), &bn2, eng)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { first, blocks })
    }

    fn forward(&self, eng: &Engine, x: &BTensor) -> Result<BTensor> {
        let mut f = self.first.forward(eng, x)?;
        fast::relu(&mut f);
        for (conv1, conv2) in &self.blocks {
            let mut o = conv1.forward(eng, &f)?;
            fast::relu(&mut o);
            fast::add_assign(&mut f, &conv2.forward(eng, &o)?)?;
        }
        Ok(f)
    }
}

/// One network of the eICCI bank, packed for one engine.
#[derive(Clone, Debug)]
pub struct IcciNet {
    luma: Trunk,
    last_y: ConvLayer,
    chroma: Trunk,
    last_u: ConvLayer,
    last_v: ConvLayer,
}

impl IcciNet {
    pub fn load(ck: &Checkpoint<'_>, eng: &Engine) -> Result<Self> {
        let last = |conv: &str, scale: &str| -> Result<ConvLayer> {
            let s = scalar(ck, scale)?;
            let post = alloc::vec![(s, 0.0); BANDS];
            folded_conv(ck, conv, FEATURES, BANDS, (3, 3), &post, eng)
        };
        Ok(Self {
            luma: Trunk::load(
                ck,
                "conv_first_Y",
                "BNY",
                "hidden_layer_Y",
                LUMA_BLOCKS,
                eng,
            )?,
            last_y: last("conv_last_Y", "scaleY")?,
            chroma: Trunk::load(
                ck,
                "conv_firstUV",
                "BNUV",
                "hidden_layer_UV",
                CHROMA_BLOCKS,
                eng,
            )?,
            last_u: last("conv_last_u", "scaleU")?,
            last_v: last("conv_last_v", "scaleV")?,
        })
    }

    /// `process_Y` up to the inverse transform: the scaled correction of the 16 luma sub-bands
    /// from the 48 sub-bands of all planes.
    pub fn luma_correction(&self, eng: &Engine, yuv16: &BTensor) -> Result<BTensor> {
        self.last_y.forward(eng, &self.luma.forward(eng, yuv16)?)
    }

    /// `process_UV` up to the inverse transform: scaled corrections of the U and V sub-bands
    /// (either can be skipped; the trunk runs once).
    pub fn chroma_corrections(
        &self,
        eng: &Engine,
        yuv16: &BTensor,
        want: [bool; 2],
    ) -> Result<[Option<BTensor>; 2]> {
        let f = self.chroma.forward(eng, yuv16)?;
        let u = want[0].then(|| self.last_u.forward(eng, &f)).transpose()?;
        let v = want[1].then(|| self.last_v.forward(eng, &f)).transpose()?;
        Ok([u, v])
    }
}

/// One Haar level (`my_tf_dwt`): `[h, w]` → four `[h / 2, w / 2]` bands LL, HL, LH, HH, written
/// to `dst[0..4]`. Operation order as in the reference (halve, then add left to right).
fn dwt(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    let (oh, ow) = (h / 2, w / 2);
    let n = oh * ow;
    let (ll, rest) = dst.split_at_mut(n);
    let (hl, rest) = rest.split_at_mut(n);
    let (lh, hh) = rest.split_at_mut(n);
    for y in 0..oh {
        let (r0, r1) = (&src[2 * y * w..][..w], &src[(2 * y + 1) * w..][..w]);
        for x in 0..ow {
            let (x1, x2) = (r0[2 * x] / 2.0, r1[2 * x] / 2.0);
            let (x3, x4) = (r0[2 * x + 1] / 2.0, r1[2 * x + 1] / 2.0);
            let i = y * ow + x;
            ll[i] = x1 + x2 + x3 + x4;
            hl[i] = -x1 - x2 + x3 + x4;
            lh[i] = -x1 + x2 - x3 + x4;
            hh[i] = x1 - x2 - x3 + x4;
        }
    }
}

/// Inverse of [`dwt`] (`my_tf_idwt`): four `[h, w]` bands → `[2h, 2w]`.
fn idwt(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    let n = h * w;
    let ow = 2 * w;
    for y in 0..h {
        let (top, bottom) = dst[2 * y * ow..][..2 * ow].split_at_mut(ow);
        for x in 0..w {
            let i = y * w + x;
            let (x1, x2) = (src[i] / 2.0, src[n + i] / 2.0);
            let (x3, x4) = (src[2 * n + i] / 2.0, src[3 * n + i] / 2.0);
            top[2 * x] = x1 - x2 - x3 + x4;
            bottom[2 * x] = x1 - x2 + x3 - x4;
            top[2 * x + 1] = x1 + x2 - x3 - x4;
            bottom[2 * x + 1] = x1 + x2 + x3 + x4;
        }
    }
}

/// Two Haar levels of one plane (`my_tf_dwt` + `my_tf_dwt_2`): `[h, w]` → 16 bands of
/// `[h / 4, w / 4]`, the four second-level bands of first-level band `i` at `4 * i ..`.
/// `h` and `w` must be multiples of 4.
pub fn dwt2(plane: &[f32], h: usize, w: usize, out: &mut [f32]) -> Result<()> {
    if !h.is_multiple_of(4) || !w.is_multiple_of(4) || plane.len() != h * w {
        return Err(Error::InvalidArgument("dwt2: size not a multiple of 4"));
    }
    let n1 = (h / 2) * (w / 2);
    let n2 = (h / 4) * (w / 4);
    if out.len() != BANDS * n2 {
        return Err(Error::InvalidArgument("dwt2: output size"));
    }
    let mut level1 = Tensor::<f32>::zeros(4, h / 2, w / 2)?.data;
    dwt(plane, h, w, &mut level1);
    for (band, dst) in level1
        .chunks_exact(n1.max(1))
        .zip(out.chunks_exact_mut(4 * n2))
    {
        dwt(band, h / 2, w / 2, dst);
    }
    Ok(())
}

/// Inverse of [`dwt2`] (`my_tf_idwt_2`): 16 bands of `[h, w]` → `[4h, 4w]`.
pub fn idwt2(bands: &[f32], h: usize, w: usize) -> Result<Tensor<f32>> {
    if bands.len() != BANDS * h * w {
        return Err(Error::InvalidArgument("idwt2: input size"));
    }
    let mut level1 = Tensor::<f32>::zeros(4, 2 * h, 2 * w)?.data;
    let n1 = 4 * h * w;
    for (src, dst) in bands
        .chunks_exact((4 * h * w).max(1))
        .zip(level1.chunks_exact_mut(n1.max(1)))
    {
        idwt(src, h, w, dst);
    }
    let mut out = Tensor::<f32>::zeros(1, 4 * h, 4 * w)?;
    idwt(&level1, 2 * h, 2 * w, &mut out.data);
    Ok(out)
}

/// Loaded eICCI networks, by checkpoint path and engine block size. Shared by every decode of a
/// [`crate::Decoder`], so a checkpoint is parsed and packed once.
#[derive(Default)]
pub struct NetCache {
    #[cfg(feature = "std")]
    nets: std::sync::Mutex<alloc::collections::BTreeMap<(String, usize), Arc<IcciNet>>>,
}

impl core::fmt::Debug for NetCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("NetCache")
    }
}

impl NetCache {
    /// Network `index` of the bank serving `op`, loaded from `src` on first use.
    pub fn get(
        &self,
        src: &dyn ModelSource,
        op: OperatingPoint,
        index: usize,
        eng: &Engine,
    ) -> Result<Arc<IcciNet>> {
        let path = checkpoint_path(op, index)?;
        #[cfg(feature = "std")]
        let key = (path.clone(), eng.tier.block());
        #[cfg(feature = "std")]
        if let Some(net) = self.nets.lock().ok().and_then(|c| c.get(&key).cloned()) {
            return Ok(net);
        }
        let file = src.read(&path)?;
        let net = Arc::new(IcciNet::load(&Checkpoint::parse(&file)?, eng)?);
        #[cfg(feature = "std")]
        if let Ok(mut c) = self.nets.lock() {
            c.insert(key, net.clone());
        }
        Ok(net)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_round_trip_and_band_order() {
        let (h, w) = (8, 12);
        let plane: Vec<f32> = (0..h * w).map(|i| ((i * 37) % 101) as f32 / 8.0).collect();
        let mut bands = alloc::vec![0.0f32; BANDS * (h / 4) * (w / 4)];
        dwt2(&plane, h, w, &mut bands).unwrap();
        // Band 0 is LL of LL: the 4x4 block sum over 4 (two halvings per level, 16 samples).
        let sum: f32 = (0..4)
            .flat_map(|y| (0..4).map(move |x| (y, x)))
            .map(|(y, x)| plane[y * w + x])
            .sum();
        assert_eq!(bands[0], sum / 4.0);
        let back = idwt2(&bands, h / 4, w / 4).unwrap();
        assert_eq!((back.h, back.w), (h, w));
        for (a, b) in plane.iter().zip(&back.data) {
            assert!((a - b).abs() < 1e-5);
        }
        assert!(dwt2(&plane[..h * w - 1], h, w, &mut bands).is_err());
    }

    #[test]
    fn checkpoint_names() {
        assert_eq!(
            checkpoint_path(OperatingPoint::Sop, 0).unwrap(),
            "eICCI_bophop_2d020448_20240229/eicci_bop_mse012_300k.pth"
        );
        assert_eq!(
            checkpoint_path(OperatingPoint::Hop, 9).unwrap(),
            "eICCI_bophop_2d020448_20240229/eicci_hop_mss100_300k.pth"
        );
        assert!(checkpoint_path(OperatingPoint::Bop, 10).is_err());
    }
}
