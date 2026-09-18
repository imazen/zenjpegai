//! EFE linear post-filter (decoder side): a luma-aided linear filter on the chroma planes.
//!
//! Ports `ref/src/codec/coding_tools/filters/EFElinear/EFElinear.py`: `decompress`,
//! `SplitApply`, `LumaAidedUpsampler_apply`, `pixelUnshuffleGeneral`, `pixelShuffleGeneral`,
//! `deinteger` (the header syntax, `decode_header` / `decode_filters`, is in `crate::header`).
//!
//! The reference splits luma into its four 2x2 sampling phases and chroma into as many phases as
//! the source format leaves room for (four for 4:4:4, two for 4:2:2, one for 4:2:0), pads each
//! phase plane by replication and runs two grouped convolutions per region of the picture:
//!
//! ```text
//! out[phase] = conv(chroma[phase] - mean) + mean + convY(luma[phase])
//! ```
//!
//! `conv` is the coded chroma filter plus the identity; when the picture was coded with fewer
//! chroma samples than the source has (`c_ver > s_ver` or `c_hor > s_hor`) it is a 4x4 kernel
//! that also carries fixed DCT-IF interpolation taps. Nothing is materialised here: the phase
//! planes are read straight out of the full-resolution planes, one output row at a time.
//!
//! Float order: every tap is a separate multiply and add, taps in `(ky, kx)` order, bias last.
//! Versus PyTorch the result differs by float summation order only (bound in `PORTING.md`).

use alloc::vec;
use alloc::vec::Vec;

use super::{FilterContext, FilterState};
use crate::decoder::reconstruct::Planes;
use crate::error::{Error, Result};
use crate::header::{EFE_LINEAR_SPLITS, EfeLinearHeader, EfeLinearSet};
use crate::nn::fast::for_each_row;
use crate::tensor::Tensor;

/// `EFElinear.cands[k][2..]`: the regions of each candidate split, as fractions
/// `[row_start, row_end, col_start, col_end]` of the phase plane.
pub(crate) const SPLITS: [&[[f64; 4]]; 8] = [
    &[[0.0, 1.0, 0.0, 1.0]],
    &[[0.0, 0.5, 0.0, 1.0], [0.5, 1.0, 0.0, 1.0]],
    &[[0.0, 1.0, 0.0, 0.5], [0.0, 1.0, 0.5, 1.0]],
    &[
        [0.0, 1.0, 0.0, 0.33],
        [0.0, 1.0, 0.33, 0.66],
        [0.0, 1.0, 0.66, 1.0],
    ],
    &[
        [0.0, 0.33, 0.0, 1.0],
        [0.33, 0.66, 0.0, 1.0],
        [0.66, 1.0, 0.0, 1.0],
    ],
    &[
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 0.0, 0.5],
        [0.5, 1.0, 0.5, 1.0],
    ],
    &[
        [0.0, 0.33, 0.0, 0.5],
        [0.0, 0.33, 0.5, 1.0],
        [0.33, 0.66, 0.0, 0.5],
        [0.33, 0.66, 0.5, 1.0],
        [0.66, 1.0, 0.0, 0.5],
        [0.66, 1.0, 0.5, 1.0],
    ],
    &[
        [0.0, 0.5, 0.0, 0.33],
        [0.5, 1.0, 0.0, 0.33],
        [0.0, 0.5, 0.33, 0.66],
        [0.5, 1.0, 0.33, 0.66],
        [0.0, 0.5, 0.66, 1.0],
        [0.5, 1.0, 0.66, 1.0],
    ],
];

/// `EFElinear.DCT_IF_4TAP`, phases 1 (horizontal), 2 (vertical) and 3 (diagonal); phase 0 is
/// all zero. Unsignalled constants of the reference.
pub(crate) const DCT_IF_4TAP: [[f32; 16]; 4] = [
    [0.0; 16],
    [
        0.0, 0.0, 0.0, 0.0, //
        -0.0625, -0.4375, 0.5625, -0.0625, //
        0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0,
    ],
    [
        0.0, -0.0625, 0.0, 0.0, //
        0.0, -0.4375, 0.0, 0.0, //
        0.0, 0.5625, 0.0, 0.0, //
        0.0, -0.0625, 0.0, 0.0,
    ],
    [
        0.003_906_25,
        -0.035_156_25,
        -0.035_156_25,
        0.003_906_25, //
        -0.035_156_25,
        -0.683_593_75,
        0.316_406_25,
        -0.035_156_25, //
        -0.035_156_25,
        0.316_406_25,
        0.316_406_25,
        -0.035_156_25, //
        0.003_906_25,
        -0.035_156_25,
        -0.035_156_25,
        0.003_906_25,
    ],
];

/// `deinteger`: 16-bit weight code → float, `(code - 32767) / 2^11`. Exact in f32.
pub(crate) fn deinteger(code: i64) -> f32 {
    (code - 32767) as f32 / 2048.0
}

/// `integerize`: float weight → 16-bit code, `clamp(round(w * 2^11) + 32767, 0, 65535)` —
/// Python's `round`, half to even, on the f64 of the f32 weight (`integerize` /
/// `integerizeTensor`).
pub(crate) fn integerize(w: f32) -> u32 {
    ((w as f64 * 2048.0).round_ties_even() + 32767.0).clamp(0.0, 65535.0) as u32
}

/// One region of one chroma plane with its convolution kernels.
pub(crate) struct Region {
    pub(crate) rows: (usize, usize),
    pub(crate) cols: (usize, usize),
    /// Kernel side: `fL` for pictures coded at the source's chroma resolution, else 4.
    pub(crate) k: usize,
    /// Rows / columns of context before the output sample (`start` in `SplitApply`).
    pub(crate) before: usize,
    /// Chroma kernels of the four phases, `k * k` taps each, identity (and DCT-IF) included.
    pub(crate) chroma: [Vec<f32>; 4],
    /// Luma kernel, shared by the phases.
    pub(crate) luma: Vec<f32>,
}

/// `round(split * (s + 31) / 32) * 32`, Python's round (half to even) on doubles.
pub(crate) fn split_edge(fraction: f64, s: usize) -> usize {
    let v = (fraction * (s + 31) as f64 / 32.0).round_ties_even();
    v as usize * 32
}

/// Regions and kernels of plane `p` (`0` = U, `1` = V) for one coded filter set.
pub(crate) fn regions(
    set: &EfeLinearSet,
    p: usize,
    coded_444: bool,
    (s0, s1): (usize, usize),
) -> Result<Vec<Region>> {
    // An unfiltered plane still goes through `SplitApply` in the reference, with
    // `cands[-1]` and no weights. The split is irrelevant then: every region gets the same
    // kernel, so one region covering the plane is equivalent.
    let (cand, fl) = match set.cand[p] {
        Some(c) => (c as usize, set.filter_len[p] as usize),
        None => (0, 0),
    };
    if cand >= EFE_LINEAR_SPLITS.len() || fl > 4 || (set.cand[p].is_some() && fl == 0) {
        return Err(Error::InvalidData("EFE linear: filter set out of range"));
    }
    let weighted = set.cand[p].is_some();
    if weighted
        && (set.chroma_weights[p].len() != EFE_LINEAR_SPLITS[cand]
            || set.luma_weights[p].len() != EFE_LINEAR_SPLITS[cand])
    {
        return Err(Error::InvalidData("EFE linear: filter count"));
    }
    let code = |sym: u32| deinteger(sym as i64 + set.min_symbol as i64);
    let mut out = Vec::new();
    for (num, split) in SPLITS[cand].iter().enumerate() {
        let rows = (split_edge(split[0], s0), split_edge(split[1], s0).min(s0));
        let cols = (split_edge(split[2], s1), split_edge(split[3], s1).min(s1));
        if rows.0 >= rows.1 || cols.0 >= cols.1 {
            // Small pictures leave some regions empty (the reference then convolves an empty
            // tensor, which PyTorch rejects for kernels above 1x1). Nothing to filter.
            continue;
        }
        // Kernel geometry: `conv[fL - 1]` with `start` / `end` context for pictures coded at
        // the source's chroma resolution, a 4x4 kernel holding the fL x fL filter otherwise.
        let (k, before, off) = if coded_444 {
            (fl, (fl.saturating_sub(1)) / 2, 0)
        } else {
            (4, 1, 1 - (fl.max(1) - 1) / 2)
        };
        let centre = if coded_444 { (fl - 1) / 2 } else { 1 };
        let mut luma = vec![0.0f32; k * k];
        let mut chroma: [Vec<f32>; 4] = core::array::from_fn(|_| vec![0.0f32; k * k]);
        if weighted {
            let (cw, lw) = (&set.chroma_weights[p][num], &set.luma_weights[p][num]);
            let phases = if coded_444 { 1 } else { 4 };
            if cw.len() != phases * fl * fl || lw.len() != fl * fl {
                return Err(Error::InvalidData("EFE linear: filter size"));
            }
            for ky in 0..fl {
                for kx in 0..fl {
                    let dst = (off + ky) * k + off + kx;
                    luma[dst] = code(lw[ky * fl + kx]);
                    for (i, c) in chroma.iter_mut().enumerate() {
                        let ph = if coded_444 { 0 } else { i };
                        c[dst] = code(cw[(ph * fl + ky) * fl + kx]);
                    }
                }
            }
        }
        for (i, c) in chroma.iter_mut().enumerate() {
            if !coded_444 {
                for (w, d) in c.iter_mut().zip(&DCT_IF_4TAP[i]) {
                    *w += d;
                }
            }
            c[centre * k + centre] += 1.0;
        }
        out.push(Region {
            rows,
            cols,
            k,
            before,
            chroma,
            luma,
        });
    }
    Ok(out)
}

/// One sampling phase of a plane, replicate-padded (`pixelUnshuffleGeneral` + `pad`), with
/// `offset` subtracted from every sample.
pub(crate) struct Phase {
    /// Row stride: `s1 + PAD_BEFORE + PAD_AFTER`.
    pub(crate) stride: usize,
    /// Padded samples: row `r` holds phase-plane row `r - PAD_BEFORE` (clamped).
    pub(crate) data: Vec<f32>,
}

/// Rows / columns of replicated border around a phase plane: the kernels reach at most one
/// sample before and two after the output position.
pub(crate) const PAD_BEFORE: usize = 1;
pub(crate) const PAD_AFTER: usize = 2;

/// Phase `(py, px)` of `src` sampled every `(fv, fh)` samples, as an `s0 x s1` plane plus border.
/// Rows and columns the (odd-sized) plane lacks replicate its last ones, like the reference's
/// padding before the split.
#[allow(clippy::too_many_arguments)]
pub(crate) fn phase_plane(
    eng: &crate::nn::fast::Engine,
    src: &Tensor<f32>,
    (fv, fh): (usize, usize),
    (py, px): (usize, usize),
    (s0, s1): (usize, usize),
    offset: f32,
) -> Result<Phase> {
    let stride = s1 + PAD_BEFORE + PAD_AFTER;
    let rows = s0 + PAD_BEFORE + PAD_AFTER;
    let mut data = Tensor::<f32>::zeros(1, rows, stride)?.data;
    let (h, w) = (src.h, src.w);
    // Samples of the phase that exist in `src` without clamping the column.
    let direct = if w > px {
        (w - px).div_ceil(fh).min(s1)
    } else {
        0
    };
    for_each_row(eng, &mut data, stride, |r, dst| {
        let sy = r.saturating_sub(PAD_BEFORE).min(s0 - 1);
        let row = &src.data[(sy * fv + py).min(h - 1) * w..][..w];
        let (body, tail) = dst[PAD_BEFORE..].split_at_mut(direct);
        for (d, v) in body.iter_mut().zip(row[px.min(w - 1)..].iter().step_by(fh)) {
            *d = v - offset;
        }
        let last = row[w - 1] - offset;
        let edge = if direct > 0 { body[direct - 1] } else { last };
        // Columns past the plane (odd width), then the right border: the last phase sample.
        let fill = if direct < s1 { last } else { edge };
        tail.fill(fill);
        let first = dst[PAD_BEFORE];
        dst[..PAD_BEFORE].fill(first);
    });
    Ok(Phase { stride, data })
}

/// The phase planes `SplitApply` works on, shared by both chroma planes' filter sets.
pub(crate) struct Phases {
    /// Chroma phase factors (`3 - s_ver`, `3 - s_hor`): 2 where the source has full chroma
    /// resolution, 1 where it is subsampled.
    pub(crate) fv: usize,
    /// See `fv`, horizontal factor.
    pub(crate) fh: usize,
    /// Phase plane size (`ceil(H / 2)`, `ceil(W / 2)` of luma).
    pub(crate) s0: usize,
    /// See `s0`, horizontal size.
    pub(crate) s1: usize,
    /// Luma phases by conv group (`py * 2 + px`); only the groups the output uses are built.
    pub(crate) luma: [Option<Phase>; 4],
}

impl Phases {
    /// Luma phase planes for a picture whose source subsampling is `s_ver` x `s_hor`.
    pub(crate) fn new(
        eng: &crate::nn::fast::Engine,
        s_ver: u8,
        s_hor: u8,
        luma: &Tensor<f32>,
    ) -> Result<Self> {
        let (fv, fh) = (3 - s_ver as usize, 3 - s_hor as usize);
        let (s0, s1) = (luma.h.div_ceil(2), luma.w.div_ceil(2));
        if luma.h == 0 || luma.w == 0 {
            return Err(Error::InvalidArgument("EFE linear: empty picture"));
        }
        let mut planes: [Option<Phase>; 4] = [None, None, None, None];
        for py in 0..fv {
            for px in 0..fh {
                let phase = phase_plane(eng, luma, (2, 2), (py, px), (s0, s1), 0.0)?;
                planes[py * 2 + px] = Some(phase);
            }
        }
        Ok(Self {
            fv,
            fh,
            s0,
            s1,
            luma: planes,
        })
    }
}

/// `acc[x] += w * line[x]`, one multiply and one add per sample (no FMA), which LLVM vectorises
/// without changing any sample's result.
#[inline]
fn axpy(acc: &mut [f32], w: f32, line: &[f32]) {
    for (a, &v) in acc.iter_mut().zip(line) {
        *a += w * v;
    }
}

/// `SplitApply` for one chroma plane. `coded_444` is `PictureHeader::efe_coded_444`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_plane(
    eng: &crate::nn::fast::Engine,
    coded_444: bool,
    set: &EfeLinearSet,
    p: usize,
    mean: f32,
    g: &Phases,
    chroma: &[Option<Phase>; 4],
    src: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    if set.cand[p].is_none() && coded_444 {
        // "Plane not filtered" at full coded chroma resolution: the reference (encoder and
        // decoder alike) fails on the missing luma weights, so there is no behaviour to match.
        // (With DCT-IF taps, i.e. chroma coded below the source's resolution, it works.)
        return Err(Error::Unsupported(
            "EFE linear: unfiltered plane in a picture coded at the source's chroma resolution",
        ));
    }
    let regions = regions(set, p, coded_444, (g.s0, g.s1))?;
    let mut out = Tensor::<f32>::zeros(1, src.h, src.w)?;
    let cw = src.w;
    for_each_row(eng, &mut out.data, cw, |oy, row| {
        let (py, yy) = (oy % g.fv, oy / g.fv);
        let mut acc = Vec::new();
        let mut lum = Vec::new();
        for r in regions.iter().filter(|r| r.rows.0 <= yy && yy < r.rows.1) {
            let n = r.cols.1 - r.cols.0;
            for px in 0..g.fh {
                // Conv group of this output phase: chroma phase (py, px), luma phase likewise.
                let group = py * 2 + px;
                let (Some(c), Some(l)) = (&chroma[group], &g.luma[group]) else {
                    continue;
                };
                acc.clear();
                acc.resize(n, 0.0f32);
                lum.clear();
                lum.resize(n, 0.0f32);
                for ky in 0..r.k {
                    let line = (yy + PAD_BEFORE + ky - r.before) * c.stride;
                    for kx in 0..r.k {
                        let x0 = line + r.cols.0 + PAD_BEFORE + kx - r.before;
                        // A zero tap adds nothing: skipping it leaves the sum unchanged.
                        let (wc, wl) = (r.chroma[group][ky * r.k + kx], r.luma[ky * r.k + kx]);
                        if wc != 0.0 {
                            axpy(&mut acc, wc, &c.data[x0..x0 + n]);
                        }
                        if wl != 0.0 {
                            axpy(&mut lum, wl, &l.data[x0..x0 + n]);
                        }
                    }
                }
                // conv's bias, then `+ convY(..)`; pixelShuffleGeneral, cropped to the plane.
                let dst = row[r.cols.0 * g.fh + px..].iter_mut().step_by(g.fh);
                for (d, (a, l)) in dst.zip(acc.iter().zip(&lum)) {
                    *d = (a + mean) + l;
                }
            }
        }
    });
    Ok(out)
}

/// The source / coded chroma formats the reference can produce and decode, all verified here:
/// 4:4:4 coded as 4:4:4, 4:2:2 or 4:2:0, and 4:2:2 / 4:2:0 coded as themselves. The header can
/// spell others (vertical-only subsampling; 4:2:2 coded 4:2:0) that the reference rejects.
pub(super) fn supported_format(hdr: &crate::header::PictureHeader) -> Result<()> {
    let (s, c) = ((hdr.s_ver, hdr.s_hor), (hdr.c_ver, hdr.c_hor));
    let ok = matches!(
        (s, c),
        ((1, 1), (1, 1) | (1, 2) | (2, 2)) | ((1, 2), (1, 2)) | ((2, 2), (2, 2))
    );
    if ok {
        Ok(())
    } else {
        Err(Error::Unsupported(
            "EFE filters: chroma format the reference software does not support",
        ))
    }
}

/// `EFElinear.decompress`.
pub fn apply(
    ctx: &FilterContext<'_>,
    h: &EfeLinearHeader,
    state: FilterState,
) -> Result<FilterState> {
    supported_format(ctx.hdr)?;
    let img = state.image;
    let g = Phases::new(ctx.eng, ctx.hdr.s_ver, ctx.hdr.s_hor, &img.y)?;
    if img.u.h.div_ceil(g.fv) != g.s0
        || img.u.w.div_ceil(g.fh) != g.s1
        || (img.v.h, img.v.w) != (img.u.h, img.u.w)
    {
        return Err(Error::InvalidArgument(
            "EFE linear: chroma plane size does not match the source format",
        ));
    }
    // `float(int(code)) / 100` is a Python double; PyTorch narrows it to f32 where it meets a
    // tensor (`rec_UV - meanUV`, `conv.bias[:] = meanUV`).
    let mean = h.mean.map(|code| (code as f64 / 100.0) as f32);
    let mut chroma: [[Option<Phase>; 4]; 2] = Default::default();
    for (p, src) in [&img.u, &img.v].into_iter().enumerate() {
        for py in 0..g.fv {
            for px in 0..g.fh {
                let phase =
                    phase_plane(ctx.eng, src, (g.fv, g.fh), (py, px), (g.s0, g.s1), mean[p])?;
                chroma[p][py * 2 + px] = Some(phase);
            }
        }
    }
    let coded_444 = ctx.hdr.efe_coded_444();
    let run = |set: &EfeLinearSet| -> Result<[Tensor<f32>; 2]> {
        Ok([
            filter_plane(ctx.eng, coded_444, set, 0, mean[0], &g, &chroma[0], &img.u)?,
            filter_plane(ctx.eng, coded_444, set, 1, mean[1], &g, &chroma[1], &img.v)?,
        ])
    };
    // The alternative picture for the EFE non-linear filter's on/off switch: built from the
    // first coded set, only when that filter is on and carries its first mask (the reference
    // tests `mask1` alone).
    let wants_alt = h.upsample_set.cand.iter().any(Option::is_some)
        && ctx
            .tools
            .efe_nonlinear
            .as_ref()
            .is_some_and(super::efe_nonlinear::has_first_mask);
    let upsampled = if wants_alt {
        let [u, v] = run(&h.upsample_set)?;
        Some(Planes {
            y: img.y.clone(),
            u,
            v,
        })
    } else {
        None
    };
    let [u, v] = run(&h.set)?;
    drop((g, chroma));
    let image = Planes { y: img.y, u, v };
    Ok(FilterState { image, upsampled })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::header::{PictureHeader, ToolHeader};
    use crate::model::ModelBundle;
    use crate::nn::fast::Engine;

    /// A reference-written 4:4:4 picture header (`header::tests::PIH_BASE_TOOLS_OFF`).
    pub fn picture_header() -> PictureHeader {
        let hex = "011103401f00338000008a32c5001800";
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        PictureHeader::parse(&bytes).unwrap()
    }

    /// Run `f` with a filter context for `hdr` / `tools`.
    pub fn with_ctx<R>(
        hdr: &PictureHeader,
        tools: &ToolHeader,
        f: impl FnOnce(&FilterContext<'_>) -> R,
    ) -> R {
        let eng = Engine::new();
        let scale_log = Tensor::<i32>::zeros(1, 1, 1).unwrap();
        let models = ModelBundle::new();
        f(&FilterContext {
            eng: &eng,
            hdr,
            tools,
            luma_scale_log: &scale_log,
            models: &models,
            op: hdr.synthesis_transforms[0],
            icci_nets: &Default::default(),
            stop: &enough::Unstoppable,
        })
    }

    pub fn ramp(h: usize, w: usize, scale: f32, base: f32) -> Tensor<f32> {
        let data = (0..h * w).map(|i| base + scale * i as f32).collect();
        Tensor::from_vec(1, h, w, data).unwrap()
    }

    /// Code of a weight value (`integerize` without the clamp).
    fn code(w: f32) -> u32 {
        ((w * 2048.0).round() as i32 + 32767) as u32
    }

    #[test]
    fn split_edges_round_half_to_even_like_python() {
        // round(f * (s + 31) / 32) * 32 evaluated by CPython for f = 0.33, 0.5, 0.66, 1.
        for (s, want) in [
            (444, [160, 224, 320, 480]),
            (280, [96, 160, 192, 320]),
            (101, [32, 64, 96, 128]),
            (50, [32, 32, 64, 96]),
            (20, [32, 32, 32, 64]),
            // 0.5 * 160 / 32 = 2.5 rounds to 2, not 3.
            (129, [64, 64, 96, 160]),
        ] {
            let got = [0.33, 0.5, 0.66, 1.0].map(|f| split_edge(f, s));
            assert_eq!(got, want, "s = {s}");
        }
    }

    #[test]
    fn one_tap_filter_matches_the_formula() {
        // fL = 1: out = (1 + w) * (u - mean) + mean + wy * y, per sample, on an odd-sized picture.
        let hdr = picture_header();
        let (h, w) = (5, 7);
        let y = ramp(h, w, 1.5, 16.0);
        let mut set = EfeLinearSet {
            cand: [Some(0), Some(0)],
            filter_len: [1, 1],
            ..Default::default()
        };
        set.chroma_weights = [vec![vec![code(0.5)]], vec![vec![code(-0.25)]]];
        set.luma_weights = [vec![vec![code(0.25)]], vec![vec![code(0.0)]]];
        let header = EfeLinearHeader {
            mean: [12000, 13000],
            upsample_set: EfeLinearSet::default(),
            set,
        };
        let state = FilterState {
            image: Planes {
                y: y.clone(),
                u: ramp(h, w, -0.5, 130.0),
                v: ramp(h, w, 0.25, 110.0),
            },
            upsampled: None,
        };
        let input = state.image.clone();
        let out = with_ctx(&hdr, &ToolHeader::default(), |ctx| {
            apply(ctx, &header, state)
        })
        .unwrap();
        assert!(out.upsampled.is_none());
        assert_eq!(out.image.y.data, y.data);
        for i in 0..h * w {
            let u = (input.u.data[i] - 120.0) * 1.5 + 120.0 + 0.25 * y.data[i];
            let v = (input.v.data[i] - 130.0) * 0.75 + 130.0 + 0.0;
            assert_eq!(out.image.u.data[i], u, "U sample {i}");
            assert_eq!(out.image.v.data[i], v, "V sample {i}");
        }
    }

    #[test]
    fn formats_without_an_oracle_are_rejected() {
        let mut hdr = picture_header();
        let header = EfeLinearHeader::default(); // both planes "not filtered"
        let state = || FilterState {
            image: Planes {
                y: ramp(4, 4, 1.0, 0.0),
                u: ramp(4, 4, 1.0, 0.0),
                v: ramp(4, 4, 1.0, 0.0),
            },
            upsampled: None,
        };
        let run = |hdr: &PictureHeader| {
            with_ctx(hdr, &ToolHeader::default(), |ctx| {
                apply(ctx, &header, state()).map(|_| ())
            })
        };
        assert!(matches!(run(&hdr), Err(Error::Unsupported(_))));
        (hdr.s_ver, hdr.s_hor, hdr.c_ver, hdr.c_hor) = (2, 1, 2, 1);
        assert!(matches!(run(&hdr), Err(Error::Unsupported(_))));
        (hdr.s_ver, hdr.s_hor, hdr.c_ver, hdr.c_hor) = (1, 2, 2, 2);
        assert!(matches!(run(&hdr), Err(Error::Unsupported(_))));
    }
}
