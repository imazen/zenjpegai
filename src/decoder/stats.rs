//! Per-stage decode timing.
//!
//! [`DecodeStats`] is the committed `std`-side API: a flat snapshot of where one decode
//! spent its wall-clock time, populated by [`crate::Decoder::decode_picture_stats`].
//! The decoder internals are `no_std`, so they cannot name `Instant`; instead they take a
//! shared [`Probe`] — a set of atomic nanosecond counters — and feed it [`Tick`]s, which
//! are monotonic timestamps on `std` builds and no-ops where no clock exists. Parallel
//! tasks share one `&Probe` and add into their own slots.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use core::time::Duration;

use crate::error::Result;

/// A monotonic timestamp for stage timing. On `std` builds this wraps
/// [`std::time::Instant`]; without `std` it is a zero-sized value whose
/// [`elapsed`](Self::elapsed) is always [`Duration::ZERO`].
#[derive(Clone, Copy)]
pub(crate) struct Tick {
    #[cfg(feature = "std")]
    t: std::time::Instant,
}

impl Tick {
    #[inline]
    pub(crate) fn now() -> Self {
        Self {
            #[cfg(feature = "std")]
            t: std::time::Instant::now(),
        }
    }

    #[inline]
    pub(crate) fn elapsed(self) -> Duration {
        #[cfg(feature = "std")]
        return self.t.elapsed();
        #[cfg(not(feature = "std"))]
        Duration::ZERO
    }
}

/// Add one elapsed measurement to a slot; `None` skips the clock read entirely.
#[inline]
pub(crate) fn record(slot: Option<&AtomicU64>, t: Tick) {
    if let Some(slot) = slot {
        slot.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Shared nanosecond counters the pipeline stages add into. Every field is one measured
/// section; `[2]` fields are indexed by component (`0` = luma/Y, `1` = chroma/UV).
#[derive(Debug, Default)]
pub(crate) struct Probe {
    /// Codestream scan + header parse + limits check (serial front).
    pub headers: AtomicU64,
    /// Checkpoint load + weight packing (first decode of a model only).
    pub models: AtomicU64,
    /// `decode_z`, per component (the z substream is shared and decodes in order).
    pub z: [AtomicU64; 2],
    /// `QualityMap::decode` (the SOQ substream; zero without the quality-map tool).
    pub quality_map: AtomicU64,
    /// Whole `decode_component_body` per component, task wall time.
    pub entropy_body: [AtomicU64; 2],
    /// `component_scales` per component (hyper-scale decoder + gain/RVS tools).
    pub scales: [AtomicU64; 2],
    /// `skip_mask` per component.
    pub mask: [AtomicU64; 2],
    /// Build of the per-region `sigma`/`coded` gather (pure reads; overlapped work).
    pub gather: [AtomicU64; 2],
    /// `AnsDecoder` residual symbol decode + scatter per component.
    pub residual: [AtomicU64; 2],
    /// `dequantize_residual` per component.
    pub dequantize: [AtomicU64; 2],
    /// `reconstruct_latent` per component, task wall time.
    pub latent: [AtomicU64; 2],
    /// `hyper_decoder` forward + tile merge per component.
    pub hyper: [AtomicU64; 2],
    /// `ContextModel` (`decompress` + merge) per component.
    pub mcm: [AtomicU64; 2],
    /// `post_process_latent` (LSBS) per component.
    pub lsbs: [AtomicU64; 2],
    /// The joined per-component chains (entropy + latent + LSBS), wall time.
    pub chains: AtomicU64,
    /// Synthesis transform.
    pub synthesis: AtomicU64,
    /// Coded chroma format → source chroma format.
    pub chroma: AtomicU64,
    /// Post-filters.
    pub filters: AtomicU64,
    /// Colour transform + quantise.
    pub output: AtomicU64,
    /// Whole call.
    pub total: AtomicU64,
}

impl Probe {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The `ccs` slot of a per-component field: `field(probe, 0, |p| &p.scales)`.
    #[inline]
    pub(crate) fn field<'a>(
        this: Option<&'a Self>,
        ccs: usize,
        f: impl FnOnce(&'a Self) -> &'a [AtomicU64; 2],
    ) -> Option<&'a AtomicU64> {
        this.map(|p| &f(p)[ccs])
    }

    /// Read every counter into the public snapshot.
    pub(crate) fn finish(&self) -> DecodeStats {
        let d = |a: &AtomicU64| Duration::from_nanos(a.load(Ordering::Relaxed));
        let pair = |f: &[AtomicU64; 2]| [d(&f[0]), d(&f[1])];
        DecodeStats {
            headers: d(&self.headers),
            models: d(&self.models),
            entropy_z: pair(&self.z),
            entropy_quality_map: d(&self.quality_map),
            entropy_body: pair(&self.entropy_body),
            entropy_scales: pair(&self.scales),
            entropy_mask: pair(&self.mask),
            entropy_gather: pair(&self.gather),
            entropy_residual: pair(&self.residual),
            entropy_dequantize: pair(&self.dequantize),
            latent: pair(&self.latent),
            latent_hyper: pair(&self.hyper),
            latent_mcm: pair(&self.mcm),
            post_process: pair(&self.lsbs),
            chains: d(&self.chains),
            synthesis: d(&self.synthesis),
            chroma_convert: d(&self.chroma),
            filters: d(&self.filters),
            output: d(&self.output),
            total: d(&self.total),
        }
    }
}

/// Serialized [`Stop`] for the parallel sections.
///
/// Pooled tasks would race on `stop.check()` — a `Stop` that counts its checks (the
/// cancellation tests do) would see the count overshoot the tripping call. The gate funnels
/// every `check` through a short spin section and, once one call has tripped, answers all
/// later calls with the stored reason without touching `inner` again. The sequence `inner`
/// observes is exactly `Ok..Ok, Err(reason)` — what a serial decode produces — while parallel
/// tasks still bail as soon as the trip lands.
pub(crate) struct Gate<'a> {
    inner: &'a dyn enough::Stop,
    state: AtomicU8,
}

const GATE_OPEN: u8 = 0;
const GATE_BUSY: u8 = 1;
const GATE_CANCELLED: u8 = 2;
const GATE_TIMED_OUT: u8 = 3;
const GATE_OTHER: u8 = 4;

impl<'a> Gate<'a> {
    pub(crate) fn new(inner: &'a dyn enough::Stop) -> Self {
        Self {
            inner,
            state: AtomicU8::new(GATE_OPEN),
        }
    }
}

impl enough::Stop for Gate<'_> {
    fn check(&self) -> core::result::Result<(), enough::StopReason> {
        use core::sync::atomic::Ordering::{Acquire, Release};
        loop {
            match self.state.load(Acquire) {
                GATE_OPEN => {
                    if self
                        .state
                        .compare_exchange_weak(GATE_OPEN, GATE_BUSY, Acquire, Acquire)
                        .is_err()
                    {
                        core::hint::spin_loop();
                        continue;
                    }
                    let r = self.inner.check();
                    let back = match r {
                        Ok(()) => GATE_OPEN,
                        Err(enough::StopReason::Cancelled) => GATE_CANCELLED,
                        Err(enough::StopReason::TimedOut) => GATE_TIMED_OUT,
                        Err(_) => GATE_OTHER,
                    };
                    self.state.store(back, Release);
                    return r;
                }
                GATE_BUSY => core::hint::spin_loop(),
                GATE_CANCELLED => return Err(enough::StopReason::Cancelled),
                GATE_TIMED_OUT => return Err(enough::StopReason::TimedOut),
                _ => return Err(enough::StopReason::Cancelled),
            }
        }
    }

    fn may_stop(&self) -> bool {
        self.inner.may_stop()
    }
}

/// Run `a` and `b` on the rayon pool when `parallel` holds, else sequentially.
#[inline]
pub(crate) fn join2<A, B>(
    parallel: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    if parallel {
        return rayon::join(a, b);
    }
    let _ = parallel;
    (a(), b())
}

/// `f(i)` for `i in 0..n`, collecting into a `Vec`: pooled when `parallel`, sequential
/// otherwise. Errors short-circuit the sequential path and surface (first, in order) on the
/// pooled one.
pub(crate) fn par_map<T: Send>(
    parallel: bool,
    n: usize,
    f: impl Fn(usize) -> Result<T> + Sync + Send,
) -> Result<Vec<T>> {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(&f).collect();
    }
    let _ = parallel;
    (0..n).map(f).collect()
}

/// `f(i, chunk)` for every `row_len`-sized chunk of `data`: pooled when `parallel`,
/// sequential otherwise. [`crate::nn::fast::conv::for_each_row`] without an [`Engine`].
#[inline]
pub(crate) fn for_each_chunk<T: Send>(
    parallel: bool,
    data: &mut [T],
    row_len: usize,
    f: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        data.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
        return;
    }
    let _ = parallel;
    for (i, row) in data.chunks_mut(row_len).enumerate() {
        f(i, row);
    }
}

/// Wall-clock breakdown of one [`crate::Decoder`] decode, returned by
/// [`crate::Decoder::decode_picture_stats`]. Sections measured inside the parallel
/// component chains are task wall times: `entropy_body[0]` and `entropy_body[1]` overlap
/// each other when the pool is in use, so their sum can exceed the `chains` wall time.
/// Stages a stream doesn't use (e.g. the quality map) stay at zero.
#[derive(Clone, Debug, Default)]
pub struct DecodeStats {
    /// Marker scan + header parse + limits check.
    pub headers: Duration,
    /// Checkpoint load and weight packing; nonzero only on the first decode of a model.
    pub models: Duration,
    /// z symbol decode per component — a single shared substream, always serial.
    pub entropy_z: [Duration; 2],
    /// Quality-map substream decode (overlaps the z decode on the pool).
    pub entropy_quality_map: Duration,
    /// Whole per-component entropy body (scales + mask + residual + dequantise).
    pub entropy_body: [Duration; 2],
    /// Per-component scale derivation (hyper-scale decoder + tools), task time.
    pub entropy_scales: [Duration; 2],
    /// Per-component coded-channel mask, task time.
    pub entropy_mask: [Duration; 2],
    /// Per-component `sigma`/`coded` gather, task time.
    pub entropy_gather: [Duration; 2],
    /// Per-component residual ANS decode + scatter, task time.
    pub entropy_residual: [Duration; 2],
    /// Per-component residual dequantise, task time.
    pub entropy_dequantize: [Duration; 2],
    /// Per-component latent reconstruction (hyper-decoder + context model), task wall.
    pub latent: [Duration; 2],
    /// Per-component hyper-decoder time (incl. region tile merge), task time.
    pub latent_hyper: [Duration; 2],
    /// Per-component context-model time (incl. region merge), task time.
    pub latent_mcm: [Duration; 2],
    /// LSBS latent post-processing per component.
    pub post_process: [Duration; 2],
    /// The joined per-component chains, wall time (≈ the longer of the two).
    pub chains: Duration,
    /// Synthesis transform.
    pub synthesis: Duration,
    /// Coded chroma format → source chroma format (e.g. 4:4:4 → 4:2:0 upsampling).
    pub chroma_convert: Duration,
    /// Post-filters (LEF, eICCI, …).
    pub filters: Duration,
    /// Colour transform + quantise to output samples.
    pub output: Duration,
    /// Whole call.
    pub total: Duration,
}
