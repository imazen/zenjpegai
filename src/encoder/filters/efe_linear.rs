//! EFE linear post-filter, encode side: choosing the per-region chroma filters.
//!
//! Ports `ref/src/codec/coding_tools/filters/EFElinear/EFElinear.py`: `compress`,
//! `SplitDecide` / `searchCore`, `LumaAidedUpsampler_encoder` (the least-squares fit),
//! `integerize` / `integerizeTensor` / `deinteger`, and `encode_header` / `encode_filters`
//! (`minSymbol` / `maxSymbol`). The decoder half is `crate::filters::efe_linear`; the
//! region-split tables, split-edge rounding and the DCT-IF taps are shared with it.
//!
//! The reference solves each region's `A x = B` with `torch.linalg.lstsq` — MKL's
//! `sgelsy` in f32, which is **not run-to-run deterministic** (its blocked kernels pick
//! threading/alignment-dependent summation orders, and the rank estimate occasionally
//! flips). The minimizer is what is signalled, so the solve here runs in `f64` with a
//! fixed operation order — a LAPACK `?gelsy`-equivalent path (pivoted Householder QR,
//! `?laic1` rank estimation, `?tzrzf`/`?ormrz` min-norm completion when rank-deficient).
//! On every captured reference solve the f64 minimizer integerizes to the same codes the
//! reference stream carries; where the reference's own f32 solve truncated rank, the codes
//! can differ (documented in `PORTING.md`).

use alloc::vec::Vec;

use crate::decoder::reconstruct::Planes;
use crate::encoder::colour::SourceMeta;
use crate::error::{Error, Result};
use crate::filters::efe_linear::{
    DCT_IF_4TAP, PAD_BEFORE, Phase, Phases, SPLITS, filter_plane, integerize, phase_plane,
    split_edge,
};
use crate::header::{EfeLinearHeader, EfeLinearSet};
use crate::nn::fast::Engine;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// `?gelsy` in f64 (LAPACK): the least-squares solver `LumaAidedUpsampler_encoder`
// reaches through `torch.linalg.lstsq` -> MKL `sgelsy`.
// ---------------------------------------------------------------------------

/// LAPACK `?nrm2`: scaled sum of squares, no overflow on large components.
fn nrm2(x: &[f64]) -> f64 {
    let (mut scale, mut ssq) = (0.0f64, 1.0f64);
    for &v in x {
        if v != 0.0 {
            let ax = v.abs();
            if scale < ax {
                let t = scale / ax;
                ssq = 1.0 + ssq * t * t;
                scale = ax;
            } else {
                let t = ax / scale;
                ssq += t * t;
            }
        }
    }
    scale * ssq.sqrt()
}

/// LAPACK `?lapy2`: `sqrt(a² + b²)` without overflow.
fn lapy2(a: f64, b: f64) -> f64 {
    let (w, z) = if a.abs() > b.abs() { (a, b) } else { (b, a) };
    if z == 0.0 {
        w.abs()
    } else {
        let t = z / w;
        w.abs() * (1.0 + t * t).sqrt()
    }
}

/// Column-major view of an `m x n` matrix (LAPACK storage).
struct Mat {
    m: usize,
    n: usize,
    a: Vec<f64>,
}

impl Mat {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.a[i + j * self.m]
    }
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.a[i + j * self.m] = v;
    }
    fn col(&self, j: usize) -> &[f64] {
        &self.a[j * self.m..j * self.m + self.m]
    }
}

/// `?larfg`: a Householder reflector `H = I - tau v v^T` with `v = (1, x)` such that
/// `H (alpha, x) = (beta, 0)`. `x` is overwritten by `v`'s tail; returns `(beta, tau)`.
fn larfg(alpha: &mut f64, x: &mut [f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let xnorm = nrm2(x);
    if xnorm == 0.0 {
        return 0.0;
    }
    let safmin = f64::MIN_POSITIVE / f64::EPSILON;
    let mut beta = -lapy2(*alpha, xnorm).copysign(*alpha);
    if beta.abs() < safmin {
        // Rare underflow path of the reference: rescale until `beta` is representable.
        let rsafmn = 1.0 / safmin;
        for _ in 0..20 {
            for v in x.iter_mut() {
                *v *= rsafmn;
            }
            beta *= rsafmn;
            *alpha *= rsafmn;
            if beta.abs() >= safmin {
                break;
            }
        }
        beta = -lapy2(*alpha, nrm2(x)).copysign(*alpha);
    }
    let tau = (beta - *alpha) / beta;
    let s = 1.0 / (*alpha - beta);
    for v in x.iter_mut() {
        *v *= s;
    }
    *alpha = beta;
    tau
}

/// `?laqp2` (unblocked `?geqp3`): QR with column pivoting of `a` (`m x n`). Returns the
/// pivot permutation and the reflector `tau`s; the reflector tails stay below `a`'s
/// diagonal and `R` above/on it.
fn geqp2(a: &mut Mat) -> (Vec<usize>, Vec<f64>) {
    let (m, n) = (a.m, a.n);
    let mn = m.min(n);
    let mut jpvt: Vec<usize> = (0..n).collect();
    let mut tau = vec![0.0; mn];
    // vn1 = column norms, vn2 = their unchanged copies for the downdating trick.
    let mut vn1: Vec<f64> = (0..n).map(|j| nrm2(a.col(j))).collect();
    let mut vn2 = vn1.clone();
    let tol3z = f64::EPSILON.sqrt();
    for i in 0..mn {
        let offpi = i;
        // Pivot: column of largest remaining norm (`isamax` takes the first on ties).
        let mut pvt = i;
        for j in i + 1..n {
            if vn1[j] > vn1[pvt] {
                pvt = j;
            }
        }
        if pvt != i {
            for r in 0..m {
                let t = a.at(r, i);
                a.set(r, i, a.at(r, pvt));
                a.set(r, pvt, t);
            }
            jpvt.swap(pvt, i);
            vn1[pvt] = vn1[i];
            vn2[pvt] = vn2[i];
        }
        // Reflector zeroing the column below `offpi`.
        {
            let mut col: Vec<f64> = a.col(i).to_vec();
            let mut alpha = col[offpi];
            let t = larfg(&mut alpha, &mut col[offpi + 1..]);
            tau[i] = t;
            for r in offpi + 1..m {
                a.set(r, i, col[r]);
            }
            a.set(offpi, i, alpha);
        }
        if i + 1 < n {
            // `?larf` left: apply H to the trailing columns, with the leading 1.
            let t = tau[i];
            if t != 0.0 {
                let aii = a.at(offpi, i);
                a.set(offpi, i, 1.0);
                let v: Vec<f64> = (offpi..m).map(|r| a.at(r, i)).collect();
                for j in i + 1..n {
                    // dot(v, col), then col -= t*dot*v.
                    let mut d = 0.0f64;
                    for (k, &vv) in v.iter().enumerate() {
                        d += vv * a.at(offpi + k, j);
                    }
                    let c = -t * d;
                    for (k, &vv) in v.iter().enumerate() {
                        let nv = a.at(offpi + k, j) + c * vv;
                        a.set(offpi + k, j, nv);
                    }
                }
                a.set(offpi, i, aii);
            }
        }
        // Downdate the remaining norms (slaqp2's `temp2` path).
        for j in i + 1..n {
            if vn1[j] != 0.0 {
                let mut temp = 1.0 - (a.at(offpi, j).abs() / vn1[j]).powi(2);
                temp = temp.max(0.0);
                let temp2 = temp * (vn1[j] / vn2[j]).powi(2);
                if temp2 <= tol3z {
                    if offpi + 1 < m {
                        vn1[j] = nrm2(&a.col(j)[offpi + 1..]);
                        vn2[j] = vn1[j];
                    } else {
                        vn1[j] = 0.0;
                        vn2[j] = 0.0;
                    }
                } else {
                    vn1[j] *= temp.sqrt();
                }
            }
        }
    }
    (jpvt, tau)
}

/// `?ormqr` side 'L', trans 'T', unblocked (`?orm2r`): `C -= H_i C` for reflectors
/// `i = 0..k` in order, where `a` carries `geqp2`'s output. `c` is `m x nrhs` col-major.
fn ormqr_lt(a: &Mat, k: usize, tau: &[f64], c: &mut Mat) {
    let m = a.m;
    for i in 0..k {
        if tau[i] == 0.0 {
            continue;
        }
        // v = (1, a[i+1..m, i]); C[i..m,:] -= tau_i v (v^T C[i..m,:]).
        let v: Vec<f64> = (i..m)
            .map(|r| if r == i { 1.0 } else { a.at(r, i) })
            .collect();
        for j in 0..c.n {
            let mut d = 0.0f64;
            for (r, &vv) in v.iter().enumerate() {
                d += vv * c.at(i + r, j);
            }
            let f = -tau[i] * d;
            for (r, &vv) in v.iter().enumerate() {
                let nv = c.at(i + r, j) + f * vv;
                c.set(i + r, j, nv);
            }
        }
    }
}

/// `?trtrs` 'U','N','N' on a `rank x rank` upper triangle in `a`, solving `R X = B` for
/// `c`'s first `rank` rows in place.
fn trtrs_upper(a: &Mat, rank: usize, c: &mut Mat) {
    for j in 0..c.n {
        for i in (0..rank).rev() {
            let mut s = c.at(i, j);
            for kk in i + 1..rank {
                s -= a.at(i, kk) * c.at(kk, j);
            }
            c.set(i, j, s / a.at(i, i));
        }
    }
}

/// `?laic1`: one step of the incremental singular-value estimate. `job` 1 estimates the
/// largest, 2 the smallest singular value of the matrix `[W w; 0 gamma]` whose previous
/// estimate is `sest`. Returns `(sestpr, s, c)`.
fn laic1(job: u8, x: &[f64], sest: f64, w: &[f64], gamma: f64) -> (f64, f64, f64) {
    let eps = f64::EPSILON;
    let alpha: f64 = x.iter().zip(w).map(|(a, b)| a * b).sum();
    let absalp = alpha.abs();
    let absgam = gamma.abs();
    let absest = sest.abs();
    if job == 2 {
        // Smallest singular value estimate.
        if sest == 0.0 {
            let (sine, cosine) = if absgam.max(absalp) == 0.0 {
                (1.0, 0.0)
            } else {
                (-gamma, alpha)
            };
            let s1 = sine.abs().max(cosine.abs());
            let (mut s, mut c) = (sine / s1, cosine / s1);
            let tmp = (s * s + c * c).sqrt();
            s /= tmp;
            c /= tmp;
            return (0.0, s, c);
        }
        if absgam <= eps * absest {
            return (absgam, 0.0, 1.0);
        }
        if absalp <= eps * absest {
            let (s1, s2) = (absgam, absest);
            return if s1 <= s2 {
                (s1, 0.0, 1.0)
            } else {
                (s2, 1.0, 0.0)
            };
        }
        if absest <= eps * absalp || absest <= eps * absgam {
            let (s1, s2) = (absgam, absalp);
            if s1 <= s2 {
                let tmp = s1 / s2;
                let s = (1.0 + tmp * tmp).sqrt();
                let sestpr = s2 * s;
                let c = (gamma / s2) / s;
                let s = 1.0f64.copysign(alpha) / s;
                return (sestpr, s, c);
            }
            let tmp = s2 / s1;
            let c = (1.0 + tmp * tmp).sqrt();
            let sestpr = s1 * c;
            let s = (alpha / s1) / c;
            let c = 1.0f64.copysign(gamma) / c;
            return (sestpr, s, c);
        }
        let zeta1 = alpha / absest;
        let zeta2 = gamma / absest;
        let b = (1.0 - zeta1 * zeta1 - zeta2 * zeta2) * 0.5;
        let c0 = zeta1 * zeta1;
        let t = if b > 0.0 {
            c0 / (b + (b * b + c0).sqrt())
        } else {
            (b * b + c0).sqrt() - b
        };
        let sine = -zeta1 / t;
        let cosine = -zeta2 / (1.0 + t);
        let tmp = (sine * sine + cosine * cosine).sqrt();
        ((t + 1.0).sqrt() * absest, sine / tmp, cosine / tmp)
    } else {
        // Largest singular value estimate (job == 1).
        if sest == 0.0 {
            let s1 = absgam.max(absalp);
            if s1 == 0.0 {
                return (0.0, 0.0, 1.0);
            }
            let s = alpha / s1;
            let c = gamma / s1;
            let tmp = (s * s + c * c).sqrt();
            return (s1 * tmp, s / tmp, c / tmp);
        }
        if absgam <= eps * absest {
            let tmp = absest.max(absalp);
            let s1 = absest / tmp;
            let s2 = absalp / tmp;
            return (tmp * (s1 * s1 + s2 * s2).sqrt(), 1.0, 0.0);
        }
        if absalp <= eps * absest {
            let (s1, s2) = (absgam, absest);
            return if s1 <= s2 {
                (s2, 1.0, 0.0)
            } else {
                (s1, 0.0, 1.0)
            };
        }
        if absest <= eps * absalp || absest <= eps * absgam {
            let (s1, s2) = (absgam, absalp);
            if s1 <= s2 {
                let tmp = s1 / s2;
                let c = (1.0 + tmp * tmp).sqrt();
                let sestpr = absest * (tmp / c);
                let s = -(gamma / s2) / c;
                let c = 1.0f64.copysign(alpha) / c;
                return (sestpr, s, c);
            }
            let tmp = s2 / s1;
            let s = (1.0 + tmp * tmp).sqrt();
            let sestpr = absest / s;
            let c = (alpha / s1) / s;
            let s = -1.0f64.copysign(gamma) / s;
            return (sestpr, s, c);
        }
        let zeta1 = alpha / absest;
        let zeta2 = gamma / absest;
        let norma = (1.0 + zeta1 * zeta1 + (zeta1 * zeta2).abs())
            .max((zeta1 * zeta2).abs() + zeta2 * zeta2);
        let test = 1.0 + 2.0 * (zeta1 - zeta2) * (zeta1 + zeta2);
        let (sine, cosine, sestpr);
        if test >= 0.0 {
            let b = (zeta1 * zeta1 + zeta2 * zeta2 + 1.0) * 0.5;
            let c0 = zeta2 * zeta2;
            let t = c0 / (b + (b * b - c0).abs().sqrt());
            sine = zeta1 / (1.0 - t);
            cosine = -zeta2 / t;
            sestpr = (t + 4.0 * eps * eps * norma).sqrt() * absest;
        } else {
            let b = (zeta2 * zeta2 + zeta1 * zeta1 - 1.0) * 0.5;
            let c0 = zeta1 * zeta1;
            let t = if b >= 0.0 {
                -c0 / (b + (b * b + c0).sqrt())
            } else {
                b - (b * b + c0).sqrt()
            };
            sine = -zeta1 / t;
            cosine = -zeta2 / (1.0 + t);
            sestpr = (1.0 + t + 4.0 * eps * eps * norma).sqrt() * absest;
        }
        let tmp = (sine * sine + cosine * cosine).sqrt();
        (sestpr, sine / tmp, cosine / tmp)
    }
}

/// `?latrz` (unblocked `?tzrzf`): LQ factorization of the `m x n` upper trapezoidal
/// matrix in `a` (`m <= n`). The `m x m` `T` ends in `a`'s left part; reflector tails go
/// to `a[i][n - l + j]`. Returns `tau`.
fn latrz(a: &mut Mat, l: usize) -> Vec<f64> {
    let (m, n) = (a.m, a.n);
    let mut tau = vec![0.0; m];
    if m == n {
        return tau;
    }
    for i in (0..m).rev() {
        // Reflector over (a[i][i], a[i][n-l..]) — the tail is a ROW segment.
        let mut alpha = a.at(i, i);
        let mut tail: Vec<f64> = (n - l..n).map(|j| a.at(i, j)).collect();
        tau[i] = larfg(&mut alpha, &mut tail);
        a.set(i, i, alpha);
        for (k, &v) in tail.iter().enumerate() {
            a.set(i, n - l + k, v);
        }
        // `?larz` right: rows 0..i, columns i..n get C := C H with v = (1, tail) applied
        // to (column i, columns n-l..n).
        if tau[i] != 0.0 && i > 0 {
            // work[r] = C[r][i] + sum_k tail[k] * C[r][n-l+k]
            for r in 0..i {
                let mut w = a.at(r, i);
                for (k, &v) in tail.iter().enumerate() {
                    w += a.at(r, n - l + k) * v;
                }
                let f = -tau[i] * w;
                a.set(r, i, a.at(r, i) + f);
                for (k, &v) in tail.iter().enumerate() {
                    a.set(r, n - l + k, a.at(r, n - l + k) + f * v);
                }
            }
        }
    }
    tau
}

/// `?ormr3` side 'L', trans 'T': apply the `latrz` reflectors to `c` (`n x nrhs`), in
/// forward order. `k` reflectors stored in `a`'s rows, tail length `l`.
fn ormrz_lt(a: &Mat, k: usize, l: usize, tau: &[f64], c: &mut Mat) {
    let n = c.m;
    for i in 0..k {
        if tau[i] == 0.0 {
            continue;
        }
        // v = (1, a[i][n-l..]) acts on (row i, rows n-l..n).
        for j in 0..c.n {
            let mut w = c.at(i, j);
            for r in 0..l {
                w += a.at(i, n - l + r) * c.at(n - l + r, j);
            }
            let f = -tau[i] * w;
            c.set(i, j, c.at(i, j) + f);
            for r in 0..l {
                let nv = c.at(n - l + r, j) + f * a.at(i, n - l + r);
                c.set(n - l + r, j, nv);
            }
        }
    }
}

/// `torch.linalg.lstsq(A, B)` for `A` `m x n`, `B` `m x 1`, driver 'gelsy', as an f64
/// `?gelsy`: pivoted QR, `?laic1` rank test at `rcond = eps(f32) * max(m, n)` (torch's
/// default), triangular solve, `?tzrzf`/`?ormrz` min-norm completion when short.
///
/// `a` is row-major `m x n`, `b` length `m`; returns the `n` coefficients.
fn lstsq(a: &[f64], b: &[f64], m: usize, n: usize) -> Result<Vec<f64>> {
    if a.len() != m * n || b.len() != m {
        return Err(Error::InvalidArgument("EFE linear: lstsq input shape"));
    }
    if m == 0 || n == 0 {
        return Ok(vec![0.0; n]);
    }
    let mn = m.min(n);
    // f64 port of LAPACK's safe-minimum scaling (`slabad` on `dlamch('S')/('P')`).
    let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
    let bignum = 1.0 / smlnum;
    let mut amat = Mat {
        m,
        n,
        a: {
            let mut v = vec![0.0; m * n];
            for j in 0..n {
                for i in 0..m {
                    v[i + j * m] = a[i * n + j];
                }
            }
            v
        },
    };
    let mut bmat = Mat {
        m: m.max(n),
        n: 1,
        a: {
            let mut v = vec![0.0; m.max(n)];
            v[..m].copy_from_slice(b);
            v
        },
    };
    // `slange('M')`: max |a_ij|.
    let anrm = amat.a.iter().fold(0.0f64, |s, &v| s.max(v.abs()));
    if anrm == 0.0 {
        return Ok(vec![0.0; n]);
    }
    let mut iascl = 0u8;
    if anrm < smlnum {
        for v in amat.a.iter_mut() {
            *v *= smlnum / anrm;
        }
        iascl = 1;
    } else if anrm > bignum {
        for v in amat.a.iter_mut() {
            *v *= bignum / anrm;
        }
        iascl = 2;
    }
    let bnrm = bmat.a[..m].iter().fold(0.0f64, |s, &v| s.max(v.abs()));
    let mut ibscl = 0u8;
    if bnrm > 0.0 && bnrm < smlnum {
        for v in bmat.a[..m].iter_mut() {
            *v *= smlnum / bnrm;
        }
        ibscl = 1;
    } else if bnrm > bignum {
        for v in bmat.a[..m].iter_mut() {
            *v *= bignum / bnrm;
        }
        ibscl = 2;
    }

    let (jpvt, tau) = geqp2(&mut amat);

    // Rank estimation loop (`sgelsy`'s `slaic1` walk over R's columns).
    let rcond = f32::EPSILON as f64 * m.max(n) as f64;
    let mut rank;
    if amat.at(0, 0).abs() == 0.0 {
        return Ok(vec![0.0; n]);
    } else {
        rank = 1;
    }
    let mut smin = amat.at(0, 0).abs();
    let mut smax = smin;
    let mut wmin = vec![1.0f64];
    let mut wmax = vec![1.0f64];
    while rank < mn {
        let i = rank; // next column index (0-based)
        let acol: Vec<f64> = (0..rank).map(|r| amat.at(r, i)).collect();
        let (sminpr, s1, c1) = laic1(2, &wmin, smin, &acol, amat.at(i, i));
        let (smaxpr, s2, c2) = laic1(1, &wmax, smax, &acol, amat.at(i, i));
        if smaxpr * rcond <= sminpr {
            for w in wmin.iter_mut() {
                *w *= s1;
            }
            for w in wmax.iter_mut() {
                *w *= s2;
            }
            wmin.push(c1);
            wmax.push(c2);
            smin = sminpr;
            smax = smaxpr;
            rank += 1;
        } else {
            break;
        }
    }

    // `stzrzf`: LQ of the `rank x n` upper trapezoid, when rank-deficient. `tz` keeps
    // T in its left `rank x rank` block and the reflector tails in its last `n - rank`
    // columns.
    let mut tz: Option<(Mat, Vec<f64>)> = None;
    if rank < n {
        let mut z = Mat {
            m: rank,
            n,
            a: {
                let mut v = vec![0.0; rank * n];
                for j in 0..n {
                    for i in 0..rank {
                        v[i + j * rank] = amat.at(i, j);
                    }
                }
                v
            },
        };
        let ztau = latrz(&mut z, n - rank);
        tz = Some((z, ztau));
    }
    // `sormqr` 'L','T': B := Q^T B.
    ormqr_lt(&amat, mn, &tau, &mut bmat);
    // `strsm` on T (rank-deficient) or R (full rank).
    match &tz {
        Some((z, _)) => trtrs_upper(z, rank, &mut bmat),
        None => trtrs_upper(&amat, rank, &mut bmat),
    }
    for i in rank..n {
        bmat.set(i, 0, 0.0);
    }
    // `sormrz` 'L','T': x := Z^T (y; 0).
    if let Some((z, ztau)) = &tz {
        ormrz_lt(z, rank, n - rank, ztau, &mut bmat);
    }
    // Unpermute: work[jpvt[i]] = b[i].
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[jpvt[i]] = bmat.at(i, 0);
    }
    if iascl == 1 {
        let s = anrm / smlnum;
        for v in out.iter_mut() {
            *v *= s;
        }
    } else if iascl == 2 {
        let s = anrm / bignum;
        for v in out.iter_mut() {
            *v *= s;
        }
    }
    if ibscl == 1 {
        let s = smlnum / bnrm;
        for v in out.iter_mut() {
            *v *= s;
        }
    } else if ibscl == 2 {
        let s = bignum / bnrm;
        for v in out.iter_mut() {
            *v *= s;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// `SplitDecide` / `searchCore` / `LumaAidedUpsampler_encoder`.
// ---------------------------------------------------------------------------

/// `betalist` (`SplitDecide`): the per-model weight of the PSNR term in the filter loss.
const BETAS: [f64; 4] = [0.002, 0.012, 0.075, 0.5];
/// `lossModifier`.
const LOSS_MODIFIER: f64 = 0.5;
/// `wP`, the signalled weight precision.
const WEIGHT_PRECISION: f64 = 16.0;
/// `max_samples` of `linearSolve`: above it the design matrix is subsampled.
const MAX_SOLVE_SAMPLES: f64 = 2_000_000.0;

/// One chroma plane's winning filters: `filters` dict entries `U`/`U2` (or `V`/`V2`) plus
/// `candU` and the filtered picture (`rec_U_best` at source resolution).
#[derive(Clone)]
struct PlaneDecision {
    /// `best_cand_*_idx`: index into [`SPLITS`]; the baseline run starts at candidate 0.
    cand: usize,
    /// `fL` of the winning filters.
    filter_len: usize,
    /// `filtersU` / `filtersV`: per split, the integerised chroma weights (16-bit codes,
    /// `phases * fL * fL` of them; `fL * fL` when coded at full chroma resolution).
    chroma: Vec<Vec<u32>>,
    /// `filtersU2` / `filtersV2`: per split, the integerised luma-aid weights (`fL * fL`).
    luma: Vec<Vec<u32>>,
    /// `rec_U_best` / `rec_V_best`: the filtered chroma plane at source resolution.
    filtered: Tensor<f32>,
}

/// What `SplitDecide` returns: the filtered picture and both planes' decisions.
struct DecideOut {
    /// `filters` dict equivalent for the two chroma planes.
    planes: [PlaneDecision; 2],
    /// `ans` chroma planes (source resolution): the picture after the winning filters.
    filtered: Planes,
}

/// The inputs of `EFElinear.compress`: the reconstruction and the source picture, both at
/// source resolution in `[0, 255]`, plus the parameters the header / config decide.
pub(crate) struct EfeLinearInput<'a> {
    /// SIMD engine the convolutions and phase extraction run on.
    pub eng: &'a Engine,
    /// Source / coded chroma formats and the colour transform of the stream being written.
    pub meta: &'a SourceMeta,
    /// `get_base_model_id()`: the active model, selecting `BETAS`.
    pub model_id: usize,
    /// `EFElinearParams.DCTIF_only`: signal the enable flag with no filters at all.
    pub dctif_only: bool,
    /// `EFEnonlinear.enabled`: without it the up-sampled-picture set is never searched.
    pub efe_nonlinear: bool,
    /// `org_img_i` converted to YUV at the reconstruction's range (`img.to_YUV_()` +
    /// `convert_range_`): the source's planes at the source's chroma resolution.
    pub org: &'a Planes,
    /// `imgs[0]`: the reconstruction at source resolution (`res_changer.backward_transform`
    /// + `to_format_` of `model.decompress`'s output).
    pub rec: &'a Planes,
}

/// Everything `EFElinear.compress` produces that survives in the stream.
pub(crate) struct EfeLinearOutput {
    /// `filtersBest` + `filtersBest2` + `mean` as [`EfeLinearHeader`].
    pub header: EfeLinearHeader,
    /// `ans[0]`: the picture after the winning filters (input of the EFE non-linear filter).
    pub filtered: Planes,
    /// `ans[1]`: the up-sampled picture (`filtersBest2` applied), when it was searched.
    pub upsampled: Option<Planes>,
}

/// Phase `(i / 2, i % 2)` channel of a `pixelUnshuffleGeneral(2, 2)` — always the identity on
/// the luma side; for the chroma planes it depends on the source's subsampling.
fn chroma_channel_phase(i: usize, (fv, fh): (usize, usize)) -> usize {
    let py = if fv == 2 { i / 2 } else { 0 };
    let px = if fh == 2 { i % 2 } else { 0 };
    py * 2 + px
}

/// `[round(split_lo), min(round(split_hi), s)]` for one axis of one region, the org slices of
/// `searchCore` (`sli_*_org`).
fn region_extent(split: &[f64; 4], s: usize, axis: usize) -> (usize, usize) {
    (
        split_edge(split[axis], s),
        split_edge(split[axis + 1], s).min(s),
    )
}

/// `mean(plane)` in f64, then `round(m * 100) / 100` like the reference (`SplitDecide`'s
/// `mean1`/`mean2`). The f64 sum can differ from torch's f32 `mean` by ~1e-4; only a value
/// landing within that of a rounding boundary (p ≈ 0.2% per plane) flips the coded mean.
fn plane_mean(plane: &Tensor<f32>) -> f64 {
    let m = plane.data.iter().map(|&v| v as f64).sum::<f64>() / plane.data.len() as f64;
    (m * 100.0).round_ties_even() / 100.0
}

/// `conv_UV_fix`: the fixed DCT-interpolation that turns the padded phase planes into the
/// `upsampled_UV` subtracted in `B` (and added back in `rec_add`). All-zero kernels when the
/// picture is coded at the source's chroma resolution.
fn upsampled_phases(
    phase: &[Option<Phase>; 4],
    coded_444: bool,
    (fv, fh): (usize, usize),
    (s0, s1): (usize, usize),
) -> [Vec<f32>; 4] {
    core::array::from_fn(|i| {
        let mut out = alloc::vec![0.0f32; s0 * s1];
        if coded_444 {
            return out;
        }
        let taps = &DCT_IF_4TAP[i];
        let Some(p) = &phase[chroma_channel_phase(i, (fv, fh))] else {
            return out;
        };
        // `conv_UV_fix(rec_*_pad[:, :, 1:, 1:])`: the 4x4 kernel's output at (y, x) reads
        // phase rows/cols `y - 1 ..= y + 2`, i.e. padded-plane indices `y ..= y + 3`.
        for y in 0..s0 {
            for x in 0..s1 {
                let mut acc = 0.0f32;
                for ky in 0..4 {
                    let line = (y + ky) * p.stride + x;
                    for kx in 0..4 {
                        acc += taps[ky * 4 + kx] * p.data[line + kx];
                    }
                }
                out[y * s1 + x] = acc;
            }
        }
        out
    })
}

/// One `linearSolve(A, B)` call: `subsample` strided `lstsq` solves, averaged. `rows` are the
/// design-matrix rows as index `t -> (phase_row, phase_col)` pairs over the region; `fill`
/// writes the `cols` design values of row `t` and its target.
fn linear_solve(
    n_cols: usize,
    rows: usize,
    mut fill: impl FnMut(usize, &mut [f64]) -> f64,
) -> Result<Vec<f64>> {
    // `subsample = max(round(A.shape[1] / max_samples), 1)`: Python `round`, half to even.
    let subsample = (rows as f64 / MAX_SOLVE_SAMPLES).round_ties_even().max(1.0) as usize;
    let stride = subsample * 4;
    let mut x = alloc::vec![0.0f64; n_cols];
    for i in (0..stride).step_by(4) {
        let count = (rows - i).div_ceil(stride);
        if count == 0 {
            continue;
        }
        let mut a = alloc::vec![0.0f64; count * n_cols];
        let mut b = alloc::vec![0.0f64; count];
        let mut r = 0usize;
        let mut t = i;
        while t < rows {
            b[r] = fill(t, &mut a[r * n_cols..(r + 1) * n_cols]);
            r += 1;
            t += stride;
        }
        let s = lstsq(&a, &b, count, n_cols)?;
        for (o, v) in x.iter_mut().zip(s) {
            *o += v;
        }
    }
    for v in x.iter_mut() {
        *v /= subsample as f64;
    }
    Ok(x)
}

/// The phases `SplitDecide` unshuffles once and `searchCore` then slices.
struct DecideCtx {
    /// `s`: phase-plane size `(ceil(luma_h / 2), ceil(luma_w / 2))`.
    s0: usize,
    s1: usize,
    /// `fv` / `fh`: chroma phase factors of the source subsampling.
    fv: usize,
    fh: usize,
    /// Coded at the source's chroma resolution (`scale_ver == 2 and scale_hor == 2`).
    coded_444: bool,
    /// `rec_Y_pad`'s four unshuffle channels (offset 0), index `i` = phase `(i/2, i%2)`.
    luma: [Phase; 4],
    /// Per chroma plane `p`: `rec_{U,V}_pad`'s channels at group `py * 2 + px`, minus
    /// `mean[p]`; only the `fv * fh` distinct phases are `Some`.
    chroma: [[Option<Phase>; 4]; 2],
    /// Per chroma plane: `U_org_pad` / `V_org_pad` (offset 0), same group layout.
    org: [[Option<Phase>; 4]; 2],
    /// Per chroma plane, per channel `i`: `upsampled_{U,V}` (`conv_UV_fix` output), `s0*s1`.
    upsampled: [[Vec<f32>; 4]; 2],
    /// The decoder-side apply context: luma phases of the *output* groups only.
    apply_luma: Phases,
}

/// `LumaAidedUpsampler_encoder` for one region of one chroma plane: returns the raw
/// (pre-integerisation) chroma filters `(4, fL*fL)` — one phase, or four replicated when the
/// picture is coded 4:4:4 — and the luma filter `fL*fL`.
#[allow(clippy::too_many_arguments)]
fn solve_region(
    ctx: &DecideCtx,
    p: usize,
    (r_lo, r_hi): (usize, usize),
    (c_lo, c_hi): (usize, usize),
    fl: usize,
    mean: f64,
) -> Result<([Vec<f32>; 4], Vec<f32>)> {
    let taps = fl * fl;
    let positions = (r_hi - r_lo) * (c_hi - c_lo);
    let s1 = ctx.s1;
    // An empty region solves an empty system upstream (the unfold output has no positions);
    // lstsq on it yields zeros -> every code is 32767.
    if positions == 0 {
        let z = || alloc::vec![0.0f32; taps];
        return Ok(([z(), z(), z(), z()], z()));
    }
    let padd_up = if fl % 2 == 1 { fl / 2 } else { fl / 2 - 1 };
    let phases_to_solve: &[usize] = if ctx.coded_444 { &[0] } else { &[0, 1, 2, 3] };
    let mut chroma: [Vec<f32>; 4] = core::array::from_fn(|_| alloc::vec![0.0; taps]);
    let mut luma = alloc::vec![0.0f32; taps];
    for &i in phases_to_solve {
        let Some(cp) = &ctx.chroma[p][chroma_channel_phase(i, (ctx.fv, ctx.fh))] else {
            return Err(Error::InvalidData("EFE linear: missing chroma phase"));
        };
        let (lp, op, up) = (
            &ctx.luma[i],
            ctx.org[p][chroma_channel_phase(i, (ctx.fv, ctx.fh))]
                .as_ref()
                .ok_or(Error::InvalidData("EFE linear: missing source phase"))?,
            &ctx.upsampled[p][i],
        );
        // `A = cat(unfold(rec_UV - mean), unfold(rec_Y))`, `B = org - rec - upsampled`.
        // Row `t` walks the region in raster order; `fill` writes `2 * taps` design values.
        let x = linear_solve(2 * taps, positions, |t, row| {
            let (ty, tx) = (t / (c_hi - c_lo) + r_lo, t % (c_hi - c_lo) + c_lo);
            let wy = ty + PAD_BEFORE - padd_up;
            let wx = tx + PAD_BEFORE - padd_up;
            for ky in 0..fl {
                let (cl, ll) = ((wy + ky) * cp.stride + wx, (wy + ky) * lp.stride + wx);
                for kx in 0..fl {
                    row[ky * fl + kx] = cp.data[cl + kx] as f64;
                    row[taps + ky * fl + kx] = lp.data[ll + kx] as f64;
                }
            }
            let o = op.data[(ty + PAD_BEFORE) * op.stride + tx + PAD_BEFORE] as f64;
            let r = cp.data[(ty + PAD_BEFORE) * cp.stride + tx + PAD_BEFORE] as f64 + mean;
            o - r - up[ty * s1 + tx] as f64
        })?;
        chroma[i] = x[..taps].iter().map(|&v| v as f32).collect();
        luma.copy_from_slice(&x[taps..].iter().map(|&v| v as f32).collect::<Vec<_>>());
    }
    if !ctx.coded_444 {
        // The refine solve: `rec_add = rec + conv_UV(rec - mean) + upsampled`, then a
        // luma-only fit `org - rec_add ~ unfold(rec_Y)` over the four channels' rows.
        let rec_add: [Vec<f32>; 4] = core::array::from_fn(|i| {
            let mut out = alloc::vec![0.0f32; positions];
            let cp = ctx.chroma[p][chroma_channel_phase(i, (ctx.fv, ctx.fh))]
                .as_ref()
                .expect("checked above");
            let w = &chroma[i];
            for t in 0..positions {
                let (ty, tx) = (t / (c_hi - c_lo) + r_lo, t % (c_hi - c_lo) + c_lo);
                let wy = ty + PAD_BEFORE - padd_up;
                let wx = tx + PAD_BEFORE - padd_up;
                let mut conv = 0.0f32;
                for ky in 0..fl {
                    let line = (wy + ky) * cp.stride + wx;
                    for kx in 0..fl {
                        conv += w[ky * fl + kx] * cp.data[line + kx];
                    }
                }
                let centre = (ty + PAD_BEFORE) * cp.stride + tx + PAD_BEFORE;
                out[t] = (cp.data[centre] + mean as f32) + conv + ctx.upsampled[p][i][ty * s1 + tx];
            }
            out
        });
        let x = linear_solve(taps, 4 * positions, |t, row| {
            let (i, t) = (t / positions, t % positions);
            let (ty, tx) = (t / (c_hi - c_lo) + r_lo, t % (c_hi - c_lo) + c_lo);
            let lp = &ctx.luma[i];
            let op = ctx.org[p][chroma_channel_phase(i, (ctx.fv, ctx.fh))]
                .as_ref()
                .expect("checked above");
            let wy = ty + PAD_BEFORE - padd_up;
            let wx = tx + PAD_BEFORE - padd_up;
            for ky in 0..fl {
                let line = (wy + ky) * lp.stride + wx;
                for kx in 0..fl {
                    row[ky * fl + kx] = lp.data[line + kx] as f64;
                }
            }
            op.data[(ty + PAD_BEFORE) * op.stride + tx + PAD_BEFORE] as f64 - rec_add[i][t] as f64
        })?;
        luma.copy_from_slice(&x.iter().map(|&v| v as f32).collect::<Vec<_>>());
    }
    Ok((chroma, luma))
}

/// `integerizeTensor` over a whole filter tensor.
fn codes(w: &[f32]) -> Vec<u32> {
    w.iter().map(|&v| integerize(v)).collect()
}

/// `searchCore(fl, cand)`: solve every region of `cand` for both chroma planes and apply the
/// integerised filters. Returns the coded filters and the two filtered planes (at source
/// resolution — `pixelShuffleGeneral(rec_*_new)` of the phase-space result, which is what
/// `filter_plane` writes directly).
fn search_core(
    ctx: &DecideCtx,
    eng: &Engine,
    cand_idx: usize,
    fl: usize,
    mean: [f64; 2],
    rec: &Planes,
) -> Result<(PlaneDecision, PlaneDecision)> {
    let mut sets: [Vec<(Vec<u32>, Vec<u32>)>; 2] = [Vec::new(), Vec::new()];
    for split in SPLITS[cand_idx].iter() {
        let rows = region_extent(split, ctx.s0, 0);
        let cols = region_extent(split, ctx.s1, 2);
        for p in 0..2 {
            let (ch, lu) = solve_region(ctx, p, rows, cols, fl, mean[p])?;
            // The header carries one chroma filter when coded 4:4:4, all four phases
            // otherwise (`encode_filters`' `filt[0:1]` vs `filt`).
            let flat: Vec<f32> = if ctx.coded_444 {
                ch[0].clone()
            } else {
                ch.iter().flatten().copied().collect()
            };
            sets[p].push((codes(&flat), codes(&lu)));
        }
    }
    // The apply half of `searchCore`: integerised weights, `SplitApply` geometry.
    let set = EfeLinearSet {
        cand: [Some(cand_idx as u8), Some(cand_idx as u8)],
        filter_len: [fl as u8, fl as u8],
        min_symbol: 0,
        max_symbol: u16::MAX,
        chroma_weights: [
            sets[0].iter().map(|(c, _)| c.clone()).collect(),
            sets[1].iter().map(|(c, _)| c.clone()).collect(),
        ],
        luma_weights: [
            sets[0].iter().map(|(_, l)| l.clone()).collect(),
            sets[1].iter().map(|(_, l)| l.clone()).collect(),
        ],
    };
    let build = |p: usize, src: &Tensor<f32>| -> Result<PlaneDecision> {
        let filtered = filter_plane(
            eng,
            ctx.coded_444,
            &set,
            p,
            mean[p] as f32,
            &ctx.apply_luma,
            &ctx.chroma[p],
            src,
        )?;
        Ok(PlaneDecision {
            cand: cand_idx,
            filter_len: fl,
            chroma: sets[p].iter().map(|(c, _)| c.clone()).collect(),
            luma: sets[p].iter().map(|(_, l)| l.clone()).collect(),
            filtered,
        })
    };
    Ok((build(0, &rec.u)?, build(1, &rec.v)?))
}

/// `mean((org - filtered)^2)` of one chroma plane → PSNR on the `[0, 255]` scale, the
/// `10*log10(255^2 / mse)` of `SplitDecide`. `org` and `filtered` are source-resolution
/// planes; the phase-space un/shuffle of the reference is a bijection and leaves the SSE
/// unchanged.
fn psnr(org: &Tensor<f32>, filtered: &Tensor<f32>) -> f64 {
    let mse = org
        .data
        .iter()
        .zip(&filtered.data)
        .map(|(&o, &f)| {
            let d = o - f;
            d as f64 * d as f64
        })
        .sum::<f64>()
        / org.data.len() as f64;
    10.0 * libm::log10(255.0 * 255.0 / mse)
}

/// `SplitDecide`: the baseline (candidate 0, `fL` 1), then the `fl x cand` search with its
/// early exits, deciding U and V independently.
#[allow(clippy::too_many_arguments)]
fn split_decide(
    ctx: &DecideCtx,
    eng: &Engine,
    filter_lengths: &[usize],
    candidates: &[usize],
    beta: f64,
    mean: [f64; 2],
    org: &Planes,
    rec: &Planes,
) -> Result<DecideOut> {
    let pixels = (rec.y.h * rec.y.w) as f64;
    // `searchCore(1, cands[0])` — the unfiltered-picture baseline.
    let (u0, v0) = search_core(ctx, eng, 0, 1, mean, rec)?;
    let mut best = [u0, v0];
    // `numFilters` counts the coded filters: 4 chroma phases + 1 luma, or 1 + 1 when coded
    // at the source's chroma resolution.
    let num_filters = if ctx.coded_444 { 2.0 } else { 5.0 };
    let mut loss_best = [
        beta * psnr(&org.u, &best[0].filtered) - num_filters * LOSS_MODIFIER / pixels,
        beta * psnr(&org.v, &best[1].filtered) - num_filters * LOSS_MODIFIER / pixels,
    ];
    let mut break_iteration = false;
    for &fl in filter_lengths {
        if break_iteration {
            break;
        }
        for &cand_idx in candidates {
            if fl == 1 && SPLITS[cand_idx].len() == 1 {
                continue;
            }
            let (u, v) = search_core(ctx, eng, cand_idx, fl, mean, rec)?;
            // `5 * fL^2 * wP * numSplit * lossModifier / (h * w)`.
            let cost = 5.0
                * (fl * fl) as f64
                * WEIGHT_PRECISION
                * SPLITS[cand_idx].len() as f64
                * LOSS_MODIFIER
                / pixels;
            let loss_u = beta * psnr(&org.u, &u.filtered) - cost;
            let loss_v = beta * psnr(&org.v, &v.filtered) - cost;
            if loss_best[0] < loss_u {
                loss_best[0] = loss_u;
                best[0] = u;
            }
            if loss_best[1] < loss_v {
                loss_best[1] = loss_v;
                best[1] = v;
            }
            // The early exits test the *best-so-far* filter length and candidates.
            if (fl == 2
                && best[0].filter_len == 1
                && best[1].filter_len == 1
                && best[0].cand == 0
                && best[1].cand == 0)
                || (fl == 3 && best[0].filter_len < 2 && best[1].filter_len < 2)
                || (fl == 4 && best[0].filter_len < 3 && best[1].filter_len < 3)
            {
                break_iteration = true;
                break;
            }
        }
    }
    let [u, v] = best;
    Ok(DecideOut {
        filtered: Planes {
            y: rec.y.clone(),
            u: u.filtered.clone(),
            v: v.filtered.clone(),
        },
        planes: [u, v],
    })
}

/// `encode_filters`'s second half: `minSymbol` / `maxSymbol` over the codes of the enabled
/// planes, then every weight stored as `code - min`.
fn assemble_set(planes: &[PlaneDecision; 2], coded_444: bool) -> Result<EfeLinearSet> {
    let mut set = EfeLinearSet {
        cand: [None, None],
        filter_len: [0, 0],
        min_symbol: u16::MAX,
        max_symbol: u16::MAX,
        chroma_weights: [Vec::new(), Vec::new()],
        luma_weights: [Vec::new(), Vec::new()],
    };
    // `minnn`/`maxxx` start at `2^wP - 1` / `0` and fold over every coded weight.
    let mut min = u16::MAX as u32;
    let mut max = 0u32;
    for (p, d) in planes.iter().enumerate() {
        set.cand[p] = Some(d.cand as u8);
        set.filter_len[p] = d.filter_len as u8;
        for f in d.chroma.iter().chain(&d.luma) {
            for &c in f {
                min = min.min(c);
                max = max.max(c);
            }
        }
    }
    set.min_symbol = min as u16;
    set.max_symbol = (max as i64 - min as i64).unsigned_abs() as u16;
    for (p, d) in planes.iter().enumerate() {
        let phases = if coded_444 { 1 } else { 4 };
        let taps = d.filter_len * d.filter_len;
        if SPLITS[d.cand].len() != d.chroma.len() || d.luma.len() != d.chroma.len() {
            return Err(Error::InvalidData("EFE linear: filter count"));
        }
        for f in &d.chroma {
            if f.len() != phases * taps {
                return Err(Error::InvalidData("EFE linear: chroma filter size"));
            }
            set.chroma_weights[p].push(f.iter().map(|&c| c - min).collect());
        }
        for f in &d.luma {
            if f.len() != taps {
                return Err(Error::InvalidData("EFE linear: luma filter size"));
            }
            set.luma_weights[p].push(f.iter().map(|&c| c - min).collect());
        }
    }
    Ok(set)
}

/// `EFElinear.compress`: run `SplitDecide` for the up-sampled picture set (when the EFE
/// non-linear filter is enabled and the picture is small enough), then for the real filter
/// set, and assemble the tool header. The stream order is `filtersBest2` then `filtersBest`.
pub(crate) fn decide(i: &EfeLinearInput<'_>) -> Result<EfeLinearOutput> {
    let meta = i.meta;
    let (fv, fh) = (3 - meta.s_ver as usize, 3 - meta.s_hor as usize);
    let coded_444 = meta.c_ver == meta.s_ver && meta.c_hor == meta.s_hor;
    let (org, rec) = (i.org, i.rec);
    // `SplitDecide` compares planes of the same size: `org` at source size, `rec` cropped by
    // `diff_display` — the reference misaligns the same way when they differ.
    let (s0, s1) = (org.u.h.div_ceil(fv), org.u.w.div_ceil(fh));
    if rec.y.h.div_ceil(2) != s0
        || rec.y.w.div_ceil(2) != s1
        || (org.u.h, org.u.w) != (org.v.h, org.v.w)
        || (org.u.h, org.u.w) != (rec.u.h, rec.u.w)
        || rec.y.h != org.y.h
        || rec.y.w != org.y.w
    {
        return Err(Error::InvalidArgument(
            "EFE linear: reconstruction and source sizes differ",
        ));
    }
    let eng = i.eng;
    let mean = [plane_mean(&rec.u), plane_mean(&rec.v)];
    let luma: Vec<Phase> = (0..4)
        .map(|i| phase_plane(eng, &rec.y, (2, 2), (i / 2, i % 2), (s0, s1), 0.0))
        .collect::<Result<_>>()?;
    let luma: [Phase; 4] = match luma.try_into() {
        Ok(l) => l,
        Err(_) => return Err(Error::InvalidData("internal: luma phases")),
    };
    let apply_luma = Phases::new(eng, meta.s_ver, meta.s_hor, &rec.y)?;
    let build_chroma = |src: &Tensor<f32>, offset: f32| -> Result<[Option<Phase>; 4]> {
        let mut out: [Option<Phase>; 4] = [None, None, None, None];
        for py in 0..fv {
            for px in 0..fh {
                out[py * 2 + px] =
                    Some(phase_plane(eng, src, (fv, fh), (py, px), (s0, s1), offset)?);
            }
        }
        Ok(out)
    };
    let chroma = [
        build_chroma(&rec.u, mean[0] as f32)?,
        build_chroma(&rec.v, mean[1] as f32)?,
    ];
    let org_phases = [build_chroma(&org.u, 0.0)?, build_chroma(&org.v, 0.0)?];
    let upsampled = [
        upsampled_phases(&chroma[0], coded_444, (fv, fh), (s0, s1)),
        upsampled_phases(&chroma[1], coded_444, (fv, fh), (s0, s1)),
    ];
    let ctx = DecideCtx {
        s0,
        s1,
        fv,
        fh,
        coded_444,
        luma,
        chroma,
        org: org_phases,
        upsampled,
        apply_luma,
    };
    let pixels = (org.y.h * org.y.w) as f64;
    let beta = BETAS[i.model_id.min(BETAS.len() - 1)];

    // The up-sampled picture set: `SplitDecide(rec, [1], cands[0:1])` — the candidate-0
    // baseline alone — searched only when the EFE non-linear filter is enabled and the
    // picture is under 4000x4000.
    let want_upsample = pixels < 4000.0 * 4000.0 && !i.dctif_only && i.efe_nonlinear;
    let upsample_out = if want_upsample {
        Some(split_decide(&ctx, eng, &[1], &[0], beta, mean, org, rec)?)
    } else {
        None
    };

    // The `filtL` / `cands` subsets of `compress` by picture area and model.
    let (filter_lengths, candidates): (&[usize], &[usize]) = if pixels <= 1e6 {
        if i.model_id < 2 {
            (
                if i.model_id == 0 {
                    &[1, 2][..]
                } else {
                    &[3, 4][..]
                },
                &[0, 1, 2][..],
            )
        } else {
            (&[3, 4][..], &[4, 5, 6, 7][..])
        }
    } else if pixels <= 4e6 {
        (
            match i.model_id {
                0 => &[1, 2, 3][..],
                1 => &[2, 3, 4][..],
                _ => &[3, 4][..],
            },
            if i.model_id < 2 {
                &[0, 1, 2, 3, 4, 5, 6, 7][..]
            } else {
                &[4, 5, 6, 7][..]
            },
        )
    } else if pixels <= 9e6 {
        (
            match i.model_id {
                0 => &[1, 2, 3][..],
                1 => &[3, 4][..],
                _ => &[4][..],
            },
            if i.model_id < 2 {
                &[3, 4, 5, 6, 7][..]
            } else {
                &[6, 7][..]
            },
        )
    } else if pixels < 16e6 {
        (
            if i.model_id == 0 {
                &[1, 2, 3][..]
            } else {
                &[4][..]
            },
            if i.model_id < 2 {
                &[5, 6, 7][..]
            } else {
                &[6, 7][..]
            },
        )
    } else {
        (
            if i.model_id == 0 {
                &[3, 4][..]
            } else {
                &[4][..]
            },
            &[6, 7][..],
        )
    };
    let (filter_lengths, candidates) = if i.dctif_only {
        (&[1][..], &[0][..])
    } else {
        (filter_lengths, candidates)
    };
    let out = split_decide(&ctx, eng, filter_lengths, candidates, beta, mean, org, rec)?;
    // `int(round(mean*100))`, coded `B1`; `mean` is the two-decimal `mean1`/`mean2`.
    let mean_code = |m: f64| ((m * 100.0).round_ties_even() as i64).clamp(0, 32767) as u16;
    let empty = |min_symbol: u16| EfeLinearSet {
        cand: [None, None],
        min_symbol,
        max_symbol: u16::MAX,
        ..Default::default()
    };
    if i.dctif_only {
        // `filtersBest`/`filtersBest2` are both reset to the empty dict; its means are the
        // initial `0`s because the up-sampled search never ran (`not DCTIF_only` gates it).
        return Ok(EfeLinearOutput {
            header: EfeLinearHeader {
                mean: [0, 0],
                upsample_set: empty(0),
                set: empty(u16::MAX),
            },
            filtered: out.filtered,
            upsampled: None,
        });
    }
    // `filtersBest2` codes `minSymbol 0` / `maxSymbol 65535` and raw codes; `filtersBest`
    // gets the computed range.
    let upsample_set = match &upsample_out {
        Some(o) => EfeLinearSet {
            cand: [Some(o.planes[0].cand as u8), Some(o.planes[1].cand as u8)],
            filter_len: [o.planes[0].filter_len as u8, o.planes[1].filter_len as u8],
            min_symbol: 0,
            max_symbol: u16::MAX,
            chroma_weights: [o.planes[0].chroma.clone(), o.planes[1].chroma.clone()],
            luma_weights: [o.planes[0].luma.clone(), o.planes[1].luma.clone()],
        },
        None => empty(0),
    };
    let set = assemble_set(&out.planes, coded_444)?;
    Ok(EfeLinearOutput {
        header: EfeLinearHeader {
            mean: [mean_code(mean[0]), mean_code(mean[1])],
            upsample_set,
            set,
        },
        filtered: out.filtered,
        upsampled: upsample_out.map(|o| o.filtered),
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// One tensor of a `dump_solves.py` dump (`manifest.txt` + `tensors.bin`, the same
    /// manifest+blob format `tests/common` reads).
    struct Dump {
        dtype: String,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    fn load_dump(dir: &std::path::Path) -> Option<HashMap<String, Dump>> {
        let manifest = std::fs::read_to_string(dir.join("manifest.txt")).ok()?;
        let blob = std::fs::read(dir.join("tensors.bin")).ok()?;
        let mut out = HashMap::new();
        for line in manifest
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let f: Vec<&str> = line.split_whitespace().collect();
            let ndim: usize = f[2].parse().unwrap();
            let shape: Vec<usize> = f[3..3 + ndim].iter().map(|d| d.parse().unwrap()).collect();
            let offset: usize = f[3 + ndim].parse().unwrap();
            let nbytes: usize = f[4 + ndim].parse().unwrap();
            out.insert(
                f[0].to_string(),
                Dump {
                    dtype: f[1].to_string(),
                    shape,
                    bytes: blob[offset..offset + nbytes].to_vec(),
                },
            );
        }
        Some(out)
    }

    impl Dump {
        fn f32(&self) -> Vec<f32> {
            assert_eq!(self.dtype, "f32");
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        }
        fn i64(&self) -> Vec<i64> {
            assert_eq!(self.dtype, "i64");
            self.bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect()
        }
        fn i32(&self) -> Vec<i32> {
            assert_eq!(self.dtype, "i32");
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect()
        }
    }

    /// `integerize` is `clamp(round_half_even(w * 2048) + 32767, 0, 65535)`.
    #[test]
    fn integerize_rounds_half_even() {
        // Exact halfway cases: ties go to the even neighbour, like Python's `round`.
        assert_eq!(integerize(0.5 / 2048.0), 32767);
        assert_eq!(integerize(1.5 / 2048.0), 32769);
        assert_eq!(integerize(-0.5 / 2048.0), 32767);
        assert_eq!(integerize(-1.5 / 2048.0), 32765);
        assert_eq!(integerize(0.0), 32767);
        // Saturation.
        assert_eq!(integerize(64.0), u16::MAX as u32);
        assert_eq!(integerize(-64.0), 0);
    }

    /// Every `int.N.in` / `int.N.out` pair the dump holds: `integerizeTensor`'s exact output.
    #[test]
    fn integerize_matches_reference_pairs() {
        let Some(dir) = std::env::var_os("ZENJPEGAI_EFE_DUMP").map(std::path::PathBuf::from) else {
            return;
        };
        let Some(dump) = load_dump(&dir) else { return };
        let mut n = 0;
        while let Some(pair) = dump
            .get(&format!("int.{n}.in"))
            .zip(dump.get(&format!("int.{n}.out")))
        {
            let (inp, out) = (pair.0.f32(), pair.1.i64());
            assert_eq!(inp.len(), out.len());
            for (k, (&w, &want)) in inp.iter().zip(&out).enumerate() {
                assert_eq!(
                    integerize(w),
                    want as u32,
                    "int.{n} element {k}: {w} -> {want}"
                );
            }
            n += 1;
        }
        assert!(n > 0, "{dir:?}: no int.N pairs");
    }

    /// Every `solve.N.{A,B,X}` triple: the deterministic f64 `lstsq` must produce the same
    /// *integerised* weights as MKL's dumped solution (the signalled quantity). The raw
    /// coefficient gap is reported for diagnosis: MKL's f32 solve sits ~1e-5 off the f64
    /// minimizer in the captured solves, and can diverge entirely when its rank estimate
    /// flips — those cases are logged, not asserted (see the module docs).
    #[test]
    fn lstsq_matches_reference_solves() {
        let Some(dir) = std::env::var_os("ZENJPEGAI_EFE_DUMP").map(std::path::PathBuf::from) else {
            return;
        };
        let Some(dump) = load_dump(&dir) else { return };
        let mut n = 0;
        let mut bad_codes = 0usize;
        while let (Some(a), Some(b), Some(x)) = (
            dump.get(&format!("solve.{n}.A")),
            dump.get(&format!("solve.{n}.B")),
            dump.get(&format!("solve.{n}.X")),
        ) {
            assert_eq!(a.shape[0], 1);
            let (m, cols) = (a.shape[1], a.shape[2]);
            assert_eq!(b.shape[1], m);
            let (a, b, x) = (a.f32(), b.f32(), x.f32());
            let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
            let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
            let got = lstsq(&af, &bf, m, cols).unwrap();
            assert_eq!(got.len(), cols);
            let mut max_gap = 0.0f64;
            let mut code_diffs = 0usize;
            for (k, (&g, &r)) in got.iter().zip(&x).enumerate() {
                max_gap = max_gap.max((g - r as f64).abs());
                if integerize(g as f32) != integerize(r) {
                    code_diffs += 1;
                    eprintln!("solve.{n} col {k}: got {g} ref {r} (codes differ)");
                }
            }
            eprintln!(
                "solve.{n}: {m}x{cols}, max |x - x_ref| = {max_gap:.3e}, code diffs {code_diffs}/{cols}"
            );
            bad_codes += code_diffs;
            n += 1;
        }
        assert!(n > 0, "{dir:?}: no solve.N triples");
        assert_eq!(bad_codes, 0, "{bad_codes} weight codes diverged");
    }

    /// `f64` tensor of the dump (the `*.means` entries).
    impl Dump {
        fn f64(&self) -> Vec<f64> {
            assert_eq!(self.dtype, "f64");
            self.bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect()
        }
    }

    fn tensor(d: &Dump) -> Tensor<f32> {
        let f = d.f32();
        Tensor::from_vec(d.shape[0] * d.shape[1], d.shape[2], d.shape[3], f).unwrap()
    }

    /// The forced-`4:5` capture of `dump_solves.py` (`crop277_f4c5`): `SplitDecide` with the
    /// candidate list pinned to `[cands[5]]` and `fL` to 4, on a 201x277 4:4:4 crop coded
    /// 4:4:4. Both planes decided for the baseline (candidate 0); the test replays the same
    /// pinned search and expects the same codes and filtered planes, then replays `up2` (the
    /// up-sampled set's `[1]`/`cands[0]` search).
    #[test]
    fn split_decide_matches_forced_reference() {
        let Some(dir) = std::env::var_os("ZENJPEGAI_EFE_DUMP").map(std::path::PathBuf::from) else {
            return;
        };
        let Some(dump) = load_dump(&dir) else { return };
        let planes = |k: &str| Planes {
            y: tensor(&dump[&format!("{k}.a")]),
            u: tensor(&dump[&format!("{k}.b")]),
            v: tensor(&dump[&format!("{k}.c")]),
        };
        let org = planes("org");
        let rec = planes("rec");
        let meta = SourceMeta {
            bit_depth: 8,
            s_ver: 1,
            s_hor: 1,
            c_ver: 1,
            c_hor: 1,
            colour_transform: crate::header::ColourTransform::None,
        };
        let eng = Engine::new();
        // The same construction `decide` does, factored out for the pinned search.
        let (fv, fh) = (2usize, 2usize);
        let (s0, s1) = (org.u.h.div_ceil(fv), org.u.w.div_ceil(fh));
        let mean = [plane_mean(&rec.u), plane_mean(&rec.v)];
        let want_mean = dump["4:5.means"].f64();
        assert_eq!(mean, [want_mean[0], want_mean[1]], "mean1/mean2");
        let luma: Vec<Phase> = (0..4)
            .map(|i| phase_plane(&eng, &rec.y, (2, 2), (i / 2, i % 2), (s0, s1), 0.0))
            .collect::<Result<_>>()
            .unwrap();
        let luma: [Phase; 4] = luma.try_into().map_err(|_| ()).unwrap();
        let apply_luma = Phases::new(&eng, 1, 1, &rec.y).unwrap();
        let build = |src: &Tensor<f32>, off: f32| -> [Option<Phase>; 4] {
            let mut out: [Option<Phase>; 4] = [None, None, None, None];
            for py in 0..fv {
                for px in 0..fh {
                    out[py * 2 + px] =
                        Some(phase_plane(&eng, src, (fv, fh), (py, px), (s0, s1), off).unwrap());
                }
            }
            out
        };
        let ctx = DecideCtx {
            s0,
            s1,
            fv,
            fh,
            coded_444: true,
            luma,
            chroma: [build(&rec.u, mean[0] as f32), build(&rec.v, mean[1] as f32)],
            org: [build(&org.u, 0.0), build(&org.v, 0.0)],
            upsampled: [
                upsampled_phases(&[None, None, None, None], true, (fv, fh), (s0, s1)),
                upsampled_phases(&[None, None, None, None], true, (fv, fh), (s0, s1)),
            ],
            apply_luma,
        };
        // Pinned `[4]` x `[cands[5]]`: the capture's own spec.
        let beta = BETAS[1];
        let out = split_decide(&ctx, &eng, &[4], &[5], beta, mean, &org, &rec).unwrap();
        let meta_ref = dump["4:5.meta"].i32();
        assert_eq!(out.planes[0].cand, meta_ref[0] as usize, "candU");
        assert_eq!(out.planes[1].cand, meta_ref[1] as usize, "candV");
        for (p, (u, u2)) in [("U", "U2"), ("V", "V2")].into_iter().enumerate() {
            let want_c = dump[&format!("4:5.{u}.0")].i64();
            let want_l = dump[&format!("4:5.{u2}.0")].i64();
            // Coded 4:4:4: the dumped (4,1,fL,fL) chroma filter is four identical phases; the
            // set carries one.
            assert_eq!(out.planes[p].chroma[0].len(), want_c.len() / 4);
            for (k, &w) in out.planes[p].chroma[0].iter().enumerate() {
                assert_eq!(w as i64, want_c[k], "4:5.{u}.0 tap {k}");
            }
            for (k, &w) in out.planes[p].luma[0].iter().enumerate() {
                assert_eq!(w as i64, want_l[k], "4:5.{u2}.0 tap {k}");
            }
            let want_rec = dump[&format!("4:5.rec_{}", if p == 0 { "u" } else { "v" })].f32();
            let got = &out.planes[p].filtered.data;
            assert_eq!(got.len(), want_rec.len());
            let max_diff = got
                .iter()
                .zip(&want_rec)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "4:5 rec_{} max diff {max_diff:.3e}",
                if p == 0 { "u" } else { "v" }
            );
            assert!(max_diff < 1e-3, "filtered plane deviates by {max_diff}");
        }

        // `up2`: the up-sampled set's pinned `[1]` x `cands[0]` search.
        let up = split_decide(&ctx, &eng, &[1], &[0], beta, mean, &org, &rec).unwrap();
        let meta_ref = dump["up2.meta"].i32();
        assert_eq!(up.planes[0].cand, meta_ref[0] as usize, "up2 candU");
        assert_eq!(up.planes[1].cand, meta_ref[1] as usize, "up2 candV");
        for (p, (u, u2)) in [("U", "U2"), ("V", "V2")].into_iter().enumerate() {
            let want_c = dump[&format!("up2.{u}.0")].i64();
            let want_l = dump[&format!("up2.{u2}.0")].i64();
            assert_eq!(up.planes[p].chroma[0].len(), want_c.len() / 4);
            for (k, &w) in up.planes[p].chroma[0].iter().enumerate() {
                assert_eq!(w as i64, want_c[k], "up2.{u}.0 tap {k}");
            }
            for (k, &w) in up.planes[p].luma[0].iter().enumerate() {
                assert_eq!(w as i64, want_l[k], "up2.{u2}.0 tap {k}");
            }
        }

        // The real `decide` path: 201x277 takes the <= 1e6 branch, model 1 -> `fL` [3, 4] and
        // candidates 0..2. Whatever it picks, the header must serialise and re-parse intact.
        let out = decide(&EfeLinearInput {
            eng: &eng,
            meta: &meta,
            model_id: 1,
            dctif_only: false,
            efe_nonlinear: true,
            org: &org,
            rec: &rec,
        })
        .unwrap();
        let pih = crate::encoder::picture_header(
            org.y.w as u32,
            org.y.h as u32,
            1,
            [0, 0],
            crate::encoder::EncodeParams::default(),
            &meta,
        );
        let tools = crate::header::ToolHeader {
            efe_linear: Some(out.header),
            ..Default::default()
        };
        let bytes = tools.write(&pih).unwrap();
        let parsed = crate::header::ToolHeader::parse(&bytes, &pih).unwrap();
        assert_eq!(parsed.efe_linear, Some(tools.efe_linear.unwrap()));
    }
}
