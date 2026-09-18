//! `F.interpolate(x, size, mode="bilinear", align_corners=True)` as PyTorch 1.10.2's CPU
//! build evaluates it — the chroma down-sampler behind `Image.to_420_` / `Image.to_422_`
//! (`TensorOps.resize_tensor`).
//!
//! For a `[1, 1, H, W]` float32 tensor `upsample_bilinear2d` takes the channels-last kernel:
//! per output sample the four corner weights are pre-multiplied (`hλ * wλ`, each in f32) and
//! accumulated `fma(p0, a, p1 * b)`, `fma(p2, c, acc)`, `fma(p3, d, acc)`. Probing it with
//! impulses over ~135k samples found that expression bit-exact; the unfused sum and every
//! other contraction order are off by an ulp in up to a third of the samples
//! (`scripts/ref_vectors/gen_resize_vectors.py` has the vectors `nn_vectors.rs` checks).

use alloc::vec::Vec;

use crate::error::{Error, Result};
use crate::nn::fmadd;
use crate::tensor::Tensor;

/// Source indices and weights of one output coordinate (`align_corners = True`), the kernel's
/// `compute_source_index_and_lambda` on `scalar_t` (f32) arithmetic.
fn linear_taps(dst: usize, in_len: usize, out_len: usize) -> (usize, usize, f32, f32) {
    if in_len == out_len {
        return (dst, dst, 1.0, 0.0);
    }
    let scale = if out_len > 1 {
        (in_len - 1) as f32 / (out_len - 1) as f32
    } else {
        0.0
    };
    let real = scale * dst as f32;
    let i0 = real as i64;
    let i1 = i0 + (i0 < in_len as i64 - 1) as i64;
    let l1 = real - i0 as f32;
    let l0 = 1.0 - l1;
    (i0 as usize, i1 as usize, l0, l1)
}

/// Bilinear-resample every plane of `src` to `out_h x out_w`, bit-identical to PyTorch's CPU
/// kernel (see the module docs). Used to take 4:4:4 chroma to 4:2:2 / 4:2:0.
pub fn resize_bilinear(src: &Tensor<f32>, out_h: usize, out_w: usize) -> Result<Tensor<f32>> {
    if src.h == 0 || src.w == 0 || out_h == 0 || out_w == 0 {
        return Err(Error::InvalidArgument("resize: empty dimension"));
    }
    let mut out = Tensor::<f32>::zeros(src.c, out_h, out_w)?;
    let xs: Vec<(usize, usize, f32, f32)> =
        (0..out_w).map(|x| linear_taps(x, src.w, out_w)).collect();
    for c in 0..src.c {
        let plane = src.plane(c);
        let dst = out.plane_mut(c);
        for y in 0..out_h {
            let (y0, y1, h0, h1) = linear_taps(y, src.h, out_h);
            let (r0, r1) = (y0 * src.w, y1 * src.w);
            for (x, &(x0, x1, w0, w1)) in xs.iter().enumerate() {
                // The kernel's four pre-multiplied weights and its accumulation order.
                let p0 = h0 * w0;
                let p1 = h0 * w1;
                let p2 = h1 * w0;
                let p3 = h1 * w1;
                let acc = fmadd(p0, plane[r0 + x0], p1 * plane[r0 + x1]);
                let acc = fmadd(p2, plane[r1 + x0], acc);
                dst[y * out_w + x] = fmadd(p3, plane[r1 + x1], acc);
            }
        }
    }
    Ok(out)
}
