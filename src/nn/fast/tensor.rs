//! Channel-blocked tensor: `[ceil(c / v)][h][w][v]`, i.e. NCHWc with block size `v`.
//!
//! The block size equals the SIMD width of the tier in use, so that one vector load/store moves
//! `v` channels of one position. Lanes beyond `c` in the last block are padding; kernels never
//! read them as inputs and nothing reads them as outputs.

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct BTensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// Block size (8 or 16).
    pub v: usize,
    pub data: Vec<f32>,
}

impl BTensor {
    pub fn blocks(&self) -> usize {
        self.c.div_ceil(self.v)
    }

    pub fn zeros(c: usize, h: usize, w: usize, v: usize) -> Result<Self> {
        let n = c
            .div_ceil(v)
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .and_then(|n| n.checked_mul(v))
            .ok_or(Error::LimitExceeded("tensor size overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(n)
            .map_err(|_| Error::LimitExceeded("out of memory"))?;
        data.resize(n, 0.0);
        Ok(Self { c, h, w, v, data })
    }

    pub fn from_planar(x: &Tensor<f32>, v: usize) -> Result<Self> {
        let mut out = Self::zeros(x.c, x.h, x.w, v)?;
        let plane = x.h * x.w;
        for ch in 0..x.c {
            let (b, lane) = (ch / v, ch % v);
            let src = &x.data[ch * plane..][..plane];
            let dst = &mut out.data[b * plane * v..][..plane * v];
            for (d, &s) in dst.chunks_exact_mut(v).zip(src) {
                d[lane] = s;
            }
        }
        Ok(out)
    }

    pub fn to_planar(&self) -> Result<Tensor<f32>> {
        let mut out = Tensor::<f32>::zeros(self.c, self.h, self.w)?;
        let plane = self.h * self.w;
        let v = self.v;
        for ch in 0..self.c {
            let (b, lane) = (ch / v, ch % v);
            let src = &self.data[b * plane * v..][..plane * v];
            let dst = &mut out.data[ch * plane..][..plane];
            for (d, s) in dst.iter_mut().zip(src.chunks_exact(v)) {
                *d = s[lane];
            }
        }
        Ok(out)
    }

    /// Zero-padded copy.
    pub fn pad(&self, top: usize, left: usize, bottom: usize, right: usize) -> Result<Self> {
        let (ph, pw) = (self.h + top + bottom, self.w + left + right);
        let mut out = Self::zeros(self.c, ph, pw, self.v)?;
        let v = self.v;
        for b in 0..self.blocks() {
            for y in 0..self.h {
                let src = &self.data[(b * self.h + y) * self.w * v..][..self.w * v];
                out.data[((b * ph + y + top) * pw + left) * v..][..self.w * v].copy_from_slice(src);
            }
        }
        Ok(out)
    }

    /// Top-left `h x w` corner.
    pub fn crop(&self, h: usize, w: usize) -> Result<Self> {
        if h > self.h || w > self.w {
            return Err(Error::InvalidArgument("crop larger than tensor"));
        }
        if h == self.h && w == self.w {
            return Ok(self.clone());
        }
        let mut out = Self::zeros(self.c, h, w, self.v)?;
        let v = self.v;
        for b in 0..self.blocks() {
            for y in 0..h {
                let src = &self.data[(b * self.h + y) * self.w * v..][..w * v];
                out.data[(b * h + y) * w * v..][..w * v].copy_from_slice(src);
            }
        }
        Ok(out)
    }

    /// Concatenate along channels. Every part but the last must fill its blocks completely.
    pub fn cat(parts: &[&BTensor]) -> Result<Self> {
        let first = parts
            .first()
            .ok_or(Error::InvalidArgument("cat: no inputs"))?;
        let (h, w, v) = (first.h, first.w, first.v);
        let n = parts.len();
        for (i, p) in parts.iter().enumerate() {
            if p.h != h || p.w != w || p.v != v {
                return Err(Error::InvalidArgument("cat: shapes differ"));
            }
            if i + 1 < n && !p.c.is_multiple_of(v) {
                return Err(Error::InvalidArgument(
                    "cat: inner part does not fill its blocks",
                ));
            }
        }
        let mut data = Vec::with_capacity(parts.iter().map(|p| p.data.len()).sum());
        for p in parts {
            data.extend_from_slice(&p.data);
        }
        Ok(Self {
            c: parts.iter().map(|p| p.c).sum(),
            h,
            w,
            v,
            data,
        })
    }

    /// Channels `c0..c1`; `c0` must be block-aligned.
    pub fn slice_channels(&self, c0: usize, c1: usize) -> Result<Self> {
        if c0 > c1 || c1 > self.c || !c0.is_multiple_of(self.v) {
            return Err(Error::InvalidArgument(
                "slice_channels: range not block-aligned",
            ));
        }
        if c1 != self.c && !c1.is_multiple_of(self.v) {
            return Err(Error::InvalidArgument(
                "slice_channels: range not block-aligned",
            ));
        }
        let per_block = self.h * self.w * self.v;
        let (b0, b1) = (c0 / self.v, c1.div_ceil(self.v));
        Ok(Self {
            c: c1 - c0,
            h: self.h,
            w: self.w,
            v: self.v,
            data: self.data[b0 * per_block..b1 * per_block].to_vec(),
        })
    }
}
