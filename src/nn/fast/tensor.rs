//! Channel-blocked tensor: `[ceil(c / v)][h][w][v]`, i.e. NCHWc with block size `v`.
//!
//! The block size equals the SIMD width of the tier in use, so that one vector load/store moves
//! `v` channels of one position. Lanes beyond `c` in the last block are padding; kernels never
//! read them as inputs and nothing reads them as outputs.

use alloc::vec::Vec;

use super::{Engine, for_each_row};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Recycled tensor storage.
///
/// A synthesis pass allocates a few hundred feature maps of tens of megabytes; handing each back
/// to the allocator means an `munmap` per tensor and a page fault per 4 KiB of the next one.
/// Dropped [`BTensor`]s park their buffer here instead and [`BTensor::scratch`] reuses it without
/// clearing. The pool is bounded (24 buffers, 1 GiB unless [`set_pool_limit`] says otherwise) and
/// [`release_buffers`] empties it.
#[cfg(feature = "std")]
mod pool {
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const MAX_BUFFERS: usize = 24;
    /// Most floats parked at once by default (1 GiB).
    pub const DEFAULT_MAX_FLOATS: usize = 1 << 28;
    /// Most floats parked at once.
    static MAX_FLOATS: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_FLOATS);

    pub fn set_limit(floats: usize) {
        MAX_FLOATS.store(floats, Ordering::Relaxed);
        // Shrink to the new limit, smallest buffers first (the policy of `give`).
        let Ok(mut pool) = POOL.lock() else { return };
        while pool.iter().map(Vec::capacity).sum::<usize>() > floats {
            let Some((i, _)) = pool.iter().enumerate().min_by_key(|(_, b)| b.capacity()) else {
                break;
            };
            pool.swap_remove(i);
        }
    }

    pub fn limit() -> usize {
        MAX_FLOATS.load(Ordering::Relaxed)
    }
    /// Below this many floats the allocator is cheaper than the lock.
    const MIN_LEN: usize = 1 << 14;

    static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

    /// A buffer of length `n` with arbitrary (but initialised) contents, if one fits.
    pub fn take(n: usize) -> Option<Vec<f32>> {
        if n < MIN_LEN {
            return None;
        }
        let mut pool = POOL.lock().ok()?;
        // Smallest buffer that fits without wasting more than half of itself.
        let best = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity() >= n && b.capacity() / 2 <= n)
            .min_by_key(|(_, b)| b.capacity())
            .map(|(i, _)| i)?;
        let mut buf = pool.swap_remove(best);
        drop(pool);
        buf.resize(n, 0.0);
        Some(buf)
    }

    pub fn give(buf: Vec<f32>) {
        if buf.capacity() < MIN_LEN {
            return;
        }
        let max_floats = limit();
        if buf.capacity() > max_floats {
            return;
        }
        let Ok(mut pool) = POOL.lock() else { return };
        // Make room by dropping the smallest buffers: the large ones are the expensive ones to
        // fault in again. Give up if the smallest is no smaller than the newcomer.
        loop {
            let held: usize = pool.iter().map(Vec::capacity).sum();
            if pool.len() < MAX_BUFFERS && held + buf.capacity() <= max_floats {
                break;
            }
            let Some((i, _)) = pool.iter().enumerate().min_by_key(|(_, b)| b.capacity()) else {
                return;
            };
            if pool[i].capacity() >= buf.capacity() {
                return;
            }
            pool.swap_remove(i);
        }
        pool.push(buf);
    }

    pub fn clear() {
        if let Ok(mut pool) = POOL.lock() {
            pool.clear();
        }
    }
}

/// Free the tensor buffers kept for reuse.
pub fn release_buffers() {
    #[cfg(feature = "std")]
    pool::clear();
}

/// Most bytes the process-wide buffer pool may keep parked (default 1 GiB; 0 turns recycling
/// off). Buffers beyond the new limit are freed at once. Without `std` there is no pool.
pub fn set_pool_limit(bytes: usize) {
    #[cfg(feature = "std")]
    pool::set_limit(bytes / core::mem::size_of::<f32>());
    #[cfg(not(feature = "std"))]
    let _ = bytes;
}

/// Current limit of the buffer pool in bytes (see [`set_pool_limit`]); 0 without `std`.
pub fn pool_limit() -> usize {
    #[cfg(feature = "std")]
    return pool::limit() * core::mem::size_of::<f32>();
    #[cfg(not(feature = "std"))]
    0
}

#[derive(Debug)]
pub struct BTensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// Block size (8 or 16).
    pub v: usize,
    pub data: Vec<f32>,
}

impl Clone for BTensor {
    fn clone(&self) -> Self {
        let mut out = Self::scratch_unchecked(self.c, self.h, self.w, self.v, self.data.len());
        out.data.copy_from_slice(&self.data);
        out
    }
}

impl Drop for BTensor {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        pool::give(core::mem::take(&mut self.data));
    }
}

impl BTensor {
    fn scratch_unchecked(c: usize, h: usize, w: usize, v: usize, n: usize) -> Self {
        #[cfg(feature = "std")]
        if let Some(data) = pool::take(n) {
            return Self { c, h, w, v, data };
        }
        Self {
            c,
            h,
            w,
            v,
            data: alloc::vec![0.0; n],
        }
    }

    /// Like [`Self::zeros`] but with arbitrary contents: for outputs a kernel overwrites
    /// completely.
    pub fn scratch(c: usize, h: usize, w: usize, v: usize) -> Result<Self> {
        #[cfg_attr(not(feature = "std"), allow(unused_variables))]
        let n = c
            .div_ceil(v)
            .checked_mul(h)
            .and_then(|n| n.checked_mul(w))
            .and_then(|n| n.checked_mul(v))
            .ok_or(Error::LimitExceeded("tensor size overflow"))?;
        #[cfg(feature = "std")]
        if let Some(data) = pool::take(n) {
            return Ok(Self { c, h, w, v, data });
        }
        Self::zeros(c, h, w, v)
    }

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
        // `vec![0.0; n]` is one `calloc`: fresh pages arrive zeroed from the kernel and are only
        // touched by whoever writes them first. Reserve-then-fill would write everything twice.
        let data = alloc::vec![0.0f32; n];
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

    /// [`Self::from_planar`], one task per channel block.
    pub fn from_planar_par(eng: &Engine, x: &Tensor<f32>, v: usize) -> Result<Self> {
        let mut out = Self::zeros(x.c, x.h, x.w, v)?;
        let plane = x.h * x.w;
        if plane == 0 {
            return Ok(out);
        }
        for_each_row(eng, &mut out.data, plane * v, |b, dst| {
            let lanes = (x.c - b * v).min(v);
            for lane in 0..lanes {
                let src = &x.data[(b * v + lane) * plane..][..plane];
                for (d, &s) in dst.chunks_exact_mut(v).zip(src) {
                    d[lane] = s;
                }
            }
        });
        Ok(out)
    }

    /// [`Self::to_planar`], one task per channel block.
    pub fn to_planar_par(&self, eng: &Engine) -> Result<Tensor<f32>> {
        let mut out = Tensor::<f32>::zeros(self.c, self.h, self.w)?;
        let plane = self.h * self.w;
        let v = self.v;
        if plane == 0 {
            return Ok(out);
        }
        for_each_row(eng, &mut out.data, plane * v, |b, dst| {
            let src = &self.data[b * plane * v..][..plane * v];
            for (lane, d) in dst.chunks_exact_mut(plane).enumerate() {
                for (o, s) in d.iter_mut().zip(src.chunks_exact(v)) {
                    *o = s[lane];
                }
            }
        });
        Ok(out)
    }

    /// [`Self::pad`], one task per row.
    pub fn pad_par(
        &self,
        eng: &Engine,
        top: usize,
        left: usize,
        bottom: usize,
        right: usize,
    ) -> Result<Self> {
        let (ph, pw) = (self.h + top + bottom, self.w + left + right);
        let mut out = Self::zeros(self.c, ph, pw, self.v)?;
        let v = self.v;
        if pw == 0 {
            return Ok(out);
        }
        for_each_row(eng, &mut out.data, pw * v, |idx, row| {
            let (b, y) = (idx / ph, idx % ph);
            if y >= top && y < top + self.h {
                let src = &self.data[(b * self.h + y - top) * self.w * v..][..self.w * v];
                row[left * v..][..self.w * v].copy_from_slice(src);
            }
        });
        Ok(out)
    }

    /// [`Self::crop`], one task per row; consumes `self` so an unchanged size costs nothing.
    pub fn crop_par(self, eng: &Engine, h: usize, w: usize) -> Result<Self> {
        if h > self.h || w > self.w {
            return Err(Error::InvalidArgument("crop larger than tensor"));
        }
        if h == self.h && w == self.w {
            return Ok(self);
        }
        let mut out = Self::scratch(self.c, h, w, self.v)?;
        let v = self.v;
        if w == 0 {
            return Ok(out);
        }
        for_each_row(eng, &mut out.data, w * v, |idx, row| {
            let (b, y) = (idx / h, idx % h);
            row.copy_from_slice(&self.data[(b * self.h + y) * self.w * v..][..w * v]);
        });
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
