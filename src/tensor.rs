//! Minimal dense tensor: one image's worth of planes, `[channels, height, width]`, row-major.
//!
//! The reference works on `[1, C, H, W]` PyTorch tensors; batch is always 1, so it is dropped.

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::mem::Charge;

/// `[c, h, w]` tensor with contiguous row-major storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<T>,
    /// `data`'s bytes in the tracked ledger (see [`crate::mem`]); not part of the value.
    charge: Charge,
}

impl<T: Copy + Default> Tensor<T> {
    /// Zero-initialised tensor. Fails instead of aborting when the allocation is refused.
    pub fn zeros(c: usize, h: usize, w: usize) -> Result<Self> {
        let n = c
            .checked_mul(h)
            .and_then(|v| v.checked_mul(w))
            .ok_or(Error::LimitExceeded("tensor size overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(n)
            .map_err(|_| Error::LimitExceeded("out of memory"))?;
        data.resize(n, T::default());
        let charge = Charge::of_vec(&data);
        Ok(Self {
            c,
            h,
            w,
            data,
            charge,
        })
    }

    pub fn from_vec(c: usize, h: usize, w: usize, data: Vec<T>) -> Result<Self> {
        if Some(data.len()) != c.checked_mul(h).and_then(|v| v.checked_mul(w)) {
            return Err(Error::InvalidArgument(
                "tensor data length does not match its shape",
            ));
        }
        let charge = Charge::of_vec(&data);
        Ok(Self {
            c,
            h,
            w,
            data,
            charge,
        })
    }

    /// Split into the buffer and its tracked-bytes token: callers that need the bare `Vec`
    /// keep it tracked by holding the [`Charge`] for the `Vec`'s lifetime.
    pub(crate) fn into_parts(mut self) -> (Vec<T>, Charge) {
        (
            core::mem::take(&mut self.data),
            core::mem::take(&mut self.charge),
        )
    }

    #[inline]
    pub fn plane_len(&self) -> usize {
        self.h * self.w
    }

    #[inline]
    pub fn plane(&self, ch: usize) -> &[T] {
        let n = self.plane_len();
        &self.data[ch * n..(ch + 1) * n]
    }

    #[inline]
    pub fn plane_mut(&mut self, ch: usize) -> &mut [T] {
        let n = self.plane_len();
        &mut self.data[ch * n..(ch + 1) * n]
    }

    #[inline]
    pub fn at(&self, ch: usize, y: usize, x: usize) -> T {
        self.data[(ch * self.h + y) * self.w + x]
    }

    /// Copy of the top-left `[c, h, w]` corner (`x[:, :h, :w]`).
    pub fn crop(&self, h: usize, w: usize) -> Result<Self> {
        if h > self.h || w > self.w {
            return Err(Error::InvalidArgument("crop larger than tensor"));
        }
        let mut out = Self::zeros(self.c, h, w)?;
        for ch in 0..self.c {
            for y in 0..h {
                let src = (ch * self.h + y) * self.w;
                let dst = (ch * h + y) * w;
                out.data[dst..dst + w].copy_from_slice(&self.data[src..src + w]);
            }
        }
        Ok(out)
    }

    /// Copy of the window `x[:, y0..y0 + h, x0..x0 + w]`.
    pub fn window(&self, x0: usize, y0: usize, w: usize, h: usize) -> Result<Self> {
        if y0 + h > self.h || x0 + w > self.w {
            return Err(Error::InvalidArgument("window outside tensor"));
        }
        let mut out = Self::zeros(self.c, h, w)?;
        for ch in 0..self.c {
            for y in 0..h {
                let src = (ch * self.h + y0 + y) * self.w + x0;
                let dst = (ch * h + y) * w;
                out.data[dst..dst + w].copy_from_slice(&self.data[src..src + w]);
            }
        }
        Ok(out)
    }
}
