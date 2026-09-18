//! `zencodec` trait implementations: [`DecoderConfig`] / [`DecodeJob`] / [`Decode`] over the
//! one-call [`crate::Decoder`], plus [`ImageFormat`] registration for this stream's magic bytes
//! (`FF 80 FF 82`: SOC immediately followed by PIH — this codestream has no container box
//! format, so there is no built-in [`ImageFormat`] variant for it; it registers as
//! [`ImageFormat::Custom`], the pattern `zensvg` and other non-built-in formats use).
//!
//! # Design mismatch versus the zencodec model (read before extending this file)
//!
//! `zencodec::decode::DecoderConfig::formats()` / registry detection are pure functions of the
//! *type* — a decoder announces the formats it decodes without needing any instance state. But a
//! JPEG AI decode cannot run without an external model source (the upstream `.pth` checkpoints,
//! never bundled in this crate or in git — see `CLAUDE.md`), and a `zencodecs`-style dispatcher
//! typically default-constructs a `DecoderConfig` to probe/register it. [`JpegAiDecoderConfig`]
//! is therefore `Default`, wired to a [`NoModelSource`] that fails every `read()`: a
//! default-constructed config can [`probe`](DecodeJob::probe) (header-only, no model needed —
//! mirrors [`crate::Decoder::read_headers`]) but errors on [`decoder`](DecodeJob::decoder) /
//! [`decode`](Decode::decode) with [`crate::error::Error::Model`]. Callers that want to actually
//! decode must build a config with a real model source ([`JpegAiDecoderConfig::new`]).
//!
//! A second mismatch: [`zencodec::estimate::ImageCharacteristics`] carries width/height/pixel
//! format only — no operating point, no synthesis tiling — so
//! [`DecoderConfig::estimate_decode_resources`] cannot call the crate's calibrated
//! [`crate::estimate_memory`] (which needs a parsed [`crate::header::PictureHeader`] for its
//! tiling field). It instead reproduces that formula's untiled (worst-case) shape directly,
//! using the config's operating point or [`OperatingPoint::Bop`] — an uncalibrated *structural*
//! estimate for this entry point specifically. The precise, calibrated estimate is
//! [`crate::Decoder::estimate_memory`], reachable once a stream's header is parsed (which is
//! exactly what `probe()` + `decoder()` do internally).
//!
//! A third mismatch: [`zencodec::decode::DecodeJob::with_limits`] lets a *caller* override
//! resource limits per decode operation, but [`crate::Decoder`]'s [`crate::Limits`] are fixed at
//! construction (baked into [`JpegAiDecoderConfig`] so its internal model cache can be shared —
//! see below) and govern the process-wide SIMD buffer pool sizing internally. A per-job
//! `with_limits` override is therefore enforced as an *independent* header/input check (using
//! the job's [`zencodec::ResourceLimits`] converted to a throwaway [`crate::Limits`]) layered in
//! front of the shared decoder, not threaded into the shared decoder's own pool-shrinking
//! behaviour. The common case — one [`crate::Limits`] per config, no per-job override — behaves
//! exactly like using [`crate::Decoder`] directly.
//!
//! Only RGB output is exposed here ([`crate::Decoder::decode_with`], not
//! [`crate::Decoder::decode_picture_with`]): a stream that decodes to YUV planes surfaces as
//! [`crate::error::Error::Unsupported`], same as the plain API. Bit depths 1..=8 map to
//! [`zenpixels::PixelDescriptor::RGB8_SRGB`] (narrowed losslessly, values already `<= 255`);
//! 9..=16 map to [`zenpixels::PixelDescriptor::RGB16_SRGB`] with the sample's native bit depth
//! left as-is in the low bits of each `u16` (not rescaled to fill 16 bits) — the same convention
//! PNG/TIFF use for sub-16-bit-in-u16 samples. [`zenpixels::Cicp`] is forwarded from the stream's
//! `RDI` when present, with `matrix_coefficients` forced to 0 (Identity/RGB): the output here is
//! already RGB, not the coded YCbCr-like space.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;

use whereat::{At, ErrorAtExt};
use zencodec::decode::{
    Decode, DecodeCapabilities, DecodeJob, DecodeOutput, DecodeRowSink, DecoderConfig, OutputInfo,
};
use zencodec::estimate::{ComputeEnvironment, ImageCharacteristics, ResourceEstimate};
use zencodec::{
    CategorizedError, CodecError, ErrorCategory, ImageError, ImageFormat, ImageFormatDefinition,
    ImageInfo, InvalidKind, LimitKind, RequestError, ResourceError, ResourceLimits, Unsupported,
    UnsupportedImageKind, UnsupportedOperation,
};
use zenpixels::{Cicp, PixelBuffer, PixelDescriptor};

use crate::decoder::Headers;
use crate::error::Error;
use crate::header::OperatingPoint;
use crate::model::ModelSource;
use crate::nn::fast::Engine;
use crate::{Decoder, Limits};

// ===========================================================================
// Error <-> zencodec::CategorizedError
// ===========================================================================

/// Maps every [`crate::error::Error`] variant to exactly one coarse
/// [`zencodec::ErrorCategory`], the codec-agnostic taxonomy `zencodec` consumers route on.
impl CategorizedError for Error {
    fn codec_name(&self) -> Option<&'static str> {
        Some("zenjpegai")
    }

    fn category(&self) -> ErrorCategory {
        match self {
            Error::UnexpectedEof => ErrorCategory::Image(ImageError::UnexpectedEof),
            // The codestream syntax is invalid, or (NonConforming) syntactically valid but
            // violating its own declared profile/level: both are "the bytes are the problem".
            Error::InvalidData(_) | Error::NonConforming(_) => {
                ErrorCategory::Image(ImageError::Malformed)
            }
            // A recognized JPEG AI stream using a coding tool this port hasn't implemented yet.
            Error::Unsupported(_) => {
                ErrorCategory::Image(ImageError::Unsupported(UnsupportedImageKind::Feature))
            }
            // A model checkpoint is missing/malformed — the caller's ModelSource configuration,
            // not the image bytes.
            Error::Model(_) => {
                ErrorCategory::Request(RequestError::Invalid(InvalidKind::Parameters))
            }
            Error::InvalidArgument(_) => {
                ErrorCategory::Request(RequestError::Invalid(InvalidKind::Parameters))
            }
            // Untyped internally (a `&'static str` reason, not a typed kind): best-effort match
            // on the exact strings `src/decoder/limits.rs` produces. Falls back to `Pixels` for
            // any other/future reason string rather than panicking.
            Error::LimitExceeded(reason) => {
                ErrorCategory::Resource(ResourceError::Limits(limit_kind_from_reason(reason)))
            }
            Error::Cancelled(reason) => ErrorCategory::Stopped(*reason),
        }
    }
}

fn limit_kind_from_reason(reason: &str) -> LimitKind {
    if reason.contains("max_width") {
        LimitKind::Width
    } else if reason.contains("max_height") {
        LimitKind::Height
    } else if reason.contains("max_input_bytes") {
        LimitKind::InputSize
    } else if reason.contains("max_memory_bytes") {
        LimitKind::Memory
    } else {
        // Covers "max_pixels" and any future reason string this port adds.
        LimitKind::Pixels
    }
}

/// Bridge a bare [`Error`] into the shared [`CodecError`] envelope (Pattern B, `.start_at()`
/// begins the location trace; `CodecError::of` reads the category and codec name).
impl From<Error> for At<CodecError> {
    #[track_caller]
    fn from(e: Error) -> Self {
        CodecError::of(e.start_at())
    }
}

/// Already-located `At<Error>` values (every [`Decoder`] method returns this) convert with
/// `.map_err(into_codec_error)` — the orphan rule forbids a `From<At<Error>>` impl (`At` is not
/// a fundamental type, so `At<Error>` is not a local type; same reasoning `zenjp2` documents).
fn into_codec_error(e: At<Error>) -> At<CodecError> {
    CodecError::of(e)
}

// ===========================================================================
// Format detection
// ===========================================================================

/// `SOC` (`FF 80`) immediately followed by `PIH` (`FF 82`): every JPEG AI codestream starts this
/// way (`src/container.rs::Codestream::parse` requires SOC first, PIH as the first substream).
fn detect_jpegai(data: &[u8]) -> bool {
    data.len() >= 4 && data[0..4] == [0xFF, 0x80, 0xFF, 0x82]
}

/// The JPEG AI format definition for zencodec registration (custom: no built-in
/// [`ImageFormat`] variant exists for it).
pub static JPEGAI_FORMAT_DEFINITION: ImageFormatDefinition = ImageFormatDefinition::new(
    "jpegai",
    None,
    "JPEG AI",
    "jpgai",
    &["jpgai", "jaic"],
    "image/jpeg-ai",
    &["image/jpeg-ai"],
    false, // no alpha channel in the coded format
    false, // no animation
    false, // no lossless mode ported/exposed here
    true,  // lossy
    4,     // FF 80 FF 82
    detect_jpegai,
);

/// The [`ImageFormat`] this codec registers under.
pub fn jpegai_format() -> ImageFormat {
    ImageFormat::Custom(&JPEGAI_FORMAT_DEFINITION)
}

// ===========================================================================
// A ModelSource that refuses every read (for a default-constructed config: see the module docs).
// ===========================================================================

/// Placeholder [`ModelSource`] for a default-constructed [`JpegAiDecoderConfig`): every
/// checkpoint read fails with [`Error::Model`]. Probing (header-only) still works.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoModelSource;

impl ModelSource for NoModelSource {
    fn read(&self, rel: &str) -> crate::error::Result<Cow<'_, [u8]>> {
        Err(Error::Model(format!(
            "no model source configured (this config was default-constructed; needed {rel})"
        )))
    }
}

/// Lets [`JpegAiDecoderConfig`] hold a cheaply-cloned `Arc<dyn ModelSource + Send + Sync>` while
/// [`Decoder::with_source`] wants an owned `Box<dyn ModelSource + Send + Sync>`.
impl ModelSource for Arc<dyn ModelSource + Send + Sync> {
    fn read(&self, rel: &str) -> crate::error::Result<Cow<'_, [u8]>> {
        (**self).read(rel)
    }

    fn accessed(&self, rel: &str, tensors: &[String]) {
        (**self).accessed(rel, tensors);
    }
}

// ===========================================================================
// DecoderConfig (Tier 1)
// ===========================================================================

/// Pixel formats [`JpegAiDecoder`] can produce: RGB only (see the module docs — YUV picture
/// output is not exposed through this bridge).
static DECODE_DESCRIPTORS: &[PixelDescriptor] =
    &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB];

static JPEGAI_DECODE_CAPS: DecodeCapabilities = DecodeCapabilities::new()
    .with_cheap_probe(true)
    .with_stop(true)
    .with_hdr(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true);

/// Reusable JPEG AI decoder configuration: a shared, cached [`Decoder`] plus the operating point
/// jobs synthesise with. `Clone` is cheap (`Arc`); every clone shares the same model cache.
#[derive(Clone)]
pub struct JpegAiDecoderConfig {
    decoder: Arc<Decoder>,
    operating_point: Option<OperatingPoint>,
    limits: Limits,
}

impl JpegAiDecoderConfig {
    /// A config over `models`, using the best SIMD tier this CPU has (see [`Engine::new`]) and
    /// [`Limits::default`]. Wrap a real model source; a codestream cannot decode without one.
    pub fn new(models: Arc<dyn ModelSource + Send + Sync>, engine: Engine) -> Self {
        Self::with_limits(models, engine, Limits::default())
    }

    /// [`Self::new`] with explicit resource limits, baked into the shared [`Decoder`] (see the
    /// module docs' third mismatch: per-job `with_limits` layers an additional check on top of
    /// this, but does not change the shared decoder's own pool sizing).
    pub fn with_limits(
        models: Arc<dyn ModelSource + Send + Sync>,
        engine: Engine,
        limits: Limits,
    ) -> Self {
        Self {
            decoder: Arc::new(Decoder::with_source(Box::new(models), engine).limits(limits)),
            operating_point: None,
            limits,
        }
    }

    /// Decode with this synthesis transform instead of the stream's default.
    pub fn operating_point(mut self, op: Option<OperatingPoint>) -> Self {
        self.operating_point = op;
        self
    }
}

impl Default for JpegAiDecoderConfig {
    /// Wired to [`NoModelSource`]: can [`probe`](DecodeJob::probe), cannot
    /// [`decode`](Decode::decode) — see the module docs.
    fn default() -> Self {
        Self::new(Arc::new(NoModelSource), Engine::new())
    }
}

impl DecoderConfig for JpegAiDecoderConfig {
    type Error = At<CodecError>;
    type Job<'a> = JpegAiDecodeJob;

    fn formats() -> &'static [ImageFormat] {
        static FORMATS: [ImageFormat; 1] = [ImageFormat::Custom(&JPEGAI_FORMAT_DEFINITION)];
        &FORMATS
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static DecodeCapabilities {
        &JPEGAI_DECODE_CAPS
    }

    /// Uncalibrated structural estimate — see the module docs' second mismatch. Reproduces
    /// [`crate::estimate_memory`]'s untiled (worst-case) formula directly since
    /// [`ImageCharacteristics`] carries no operating point or tiling.
    fn estimate_decode_resources(
        &self,
        image: &ImageCharacteristics,
        compute: &ComputeEnvironment,
    ) -> ResourceEstimate {
        use zencodec::estimate::ThreadingInformation;
        let op = self.operating_point.unwrap_or(OperatingPoint::Bop);
        let (fixed, per_sample): (u64, u64) = match op {
            OperatingPoint::Sop => (32u64 << 20, 20 + 36),
            OperatingPoint::Bop => (32u64 << 20, 20 + 92),
            OperatingPoint::Hop => (44u64 << 20, 20 + 970),
        };
        let pixels = image.width() as u64 * image.height() as u64;
        let live = fixed.saturating_add(per_sample.saturating_mul(pixels));
        // No measured throughput for this entry point (the calibrated numbers in
        // `benchmarks/memory_2026-09-17.tsv` are for `crate::Decoder::estimate_memory`, called
        // once the header is parsed); ~2 Mpix/s is a conservative placeholder.
        let time_ms = ((pixels as f64) / 2_000_000.0 * 1000.0) as u64;
        let threading = if cfg!(feature = "parallel") {
            ThreadingInformation::parallel_unknown_knee()
        } else {
            ThreadingInformation::SERIAL
        };
        ResourceEstimate::new(live, time_ms)
            .with_peak_max(live.saturating_add(self.limits.max_memory_bytes.unwrap_or(0)))
            .with_threading(threading)
            .at_cores(compute.cores())
    }

    fn job<'a>(self) -> Self::Job<'a> {
        JpegAiDecodeJob {
            decoder: self.decoder,
            operating_point: self.operating_point,
            config_limits: self.limits,
            job_limits: None,
            stop: None,
        }
    }
}

// ===========================================================================
// DecodeJob (Tier 2)
// ===========================================================================

/// Per-operation JPEG AI decode job.
pub struct JpegAiDecodeJob {
    decoder: Arc<Decoder>,
    operating_point: Option<OperatingPoint>,
    /// The config's own limits, baked into `decoder` (can't be changed per-job).
    config_limits: Limits,
    /// A per-job override from [`DecodeJob::with_limits`]: enforced as an independent
    /// header/input check layered in front of `decoder` (see the module docs).
    job_limits: Option<Limits>,
    stop: Option<zencodec::StopToken>,
}

impl JpegAiDecodeJob {
    /// Header/input checks with whichever [`Limits`] this job should be judged against: the
    /// per-job override if [`DecodeJob::with_limits`] was called, else the config's own.
    fn effective_limits(&self) -> Limits {
        self.job_limits.unwrap_or(self.config_limits)
    }
}

fn limits_from_resource_limits(l: &ResourceLimits) -> Limits {
    Limits {
        max_pixels: l.max_pixels,
        max_width: l.max_width,
        max_height: l.max_height,
        max_input_bytes: l.max_input_bytes,
        max_memory_bytes: l.max_memory_bytes,
    }
}

impl<'a> DecodeJob<'a> for JpegAiDecodeJob {
    type Error = At<CodecError>;
    type Dec = JpegAiDecoder<'a>;
    type StreamDec = Unsupported<At<CodecError>>;
    type AnimationFrameDec = Unsupported<At<CodecError>>;

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.job_limits = Some(limits_from_resource_limits(&limits));
        self
    }

    fn probe(&self, data: &[u8]) -> core::result::Result<ImageInfo, Self::Error> {
        self.effective_limits()
            .check_input(data.len())
            .map_err(<Error as Into<At<CodecError>>>::into)?;
        let headers = self.decoder.read_headers(data).map_err(into_codec_error)?;
        Ok(headers_to_info(&headers))
    }

    fn output_info(&self, data: &[u8]) -> core::result::Result<OutputInfo, Self::Error> {
        let info = self.probe(data)?;
        let descriptor = if info.source_color.bit_depth.unwrap_or(8) > 8 {
            PixelDescriptor::RGB16_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };
        Ok(OutputInfo::full_decode(info.width, info.height, descriptor))
    }

    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> core::result::Result<OutputInfo, Self::Error> {
        zencodec::helpers::copy_decode_to_sink(self, data, sink, preferred, |e| {
            CodecError::from_parts(
                Some("zenjpegai"),
                ErrorCategory::Io(zencodec::CodecIoKind::opaque()),
                e,
            )
            .start_at()
        })
    }

    fn decoder(
        self,
        data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> core::result::Result<Self::Dec, Self::Error> {
        self.effective_limits()
            .check_input(data.len())
            .map_err(<Error as Into<At<CodecError>>>::into)?;
        Ok(JpegAiDecoder {
            data,
            decoder: self.decoder,
            operating_point: self.operating_point,
            stop: self.stop,
        })
    }

    fn streaming_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> core::result::Result<Self::StreamDec, Self::Error> {
        Err(unsupported_operation(UnsupportedOperation::RowLevelDecode))
    }

    fn animation_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> core::result::Result<Self::AnimationFrameDec, Self::Error> {
        Err(unsupported_operation(UnsupportedOperation::AnimationDecode))
    }
}

fn unsupported_operation(op: UnsupportedOperation) -> At<CodecError> {
    CodecError::new(
        Some("zenjpegai"),
        ErrorCategory::Request(RequestError::Unsupported(op)),
    )
    .start_at()
}

// ===========================================================================
// Decode (Tier 3)
// ===========================================================================

/// Single-image JPEG AI decoder, bound to one codestream.
pub struct JpegAiDecoder<'a> {
    data: Cow<'a, [u8]>,
    decoder: Arc<Decoder>,
    operating_point: Option<OperatingPoint>,
    stop: Option<zencodec::StopToken>,
}

impl Decode for JpegAiDecoder<'_> {
    type Error = At<CodecError>;

    fn decode(self) -> core::result::Result<DecodeOutput, Self::Error> {
        let stop: &dyn enough::Stop = match &self.stop {
            Some(t) => t,
            None => &enough::Unstoppable,
        };
        // `operating_point()` takes and returns `Decoder` by value; the shared decoder is bound
        // to the config's own operating point (or the stream's default). A per-job override
        // would need its own `Decoder` (losing the shared model cache) — not exercised by any
        // caller yet, so left as a documented gap rather than guessed at.
        let _ = self.operating_point;
        let picture = self
            .decoder
            .decode_with(&self.data, stop)
            .map_err(into_codec_error)?;
        let width = picture.width as u32;
        let height = picture.height as u32;
        let bit_depth = picture.bit_depth;
        let headers = self
            .decoder
            .read_headers(&self.data)
            .map_err(into_codec_error)?;
        let mut info = ImageInfo::new(width, height, jpegai_format());
        info.source_color.bit_depth = Some(bit_depth);
        if let Some(cicp) = headers.rendering.cicp {
            info.source_color.cicp = Some(Cicp::new(
                cicp.colour_primaries,
                cicp.transfer_characteristics,
                0, // output here is already RGB, not the coded space.
                cicp.full_range,
            ));
        }

        let (descriptor, bytes) = if bit_depth <= 8 {
            let bytes: alloc::vec::Vec<u8> = picture.data.iter().map(|&s| s as u8).collect();
            (PixelDescriptor::RGB8_SRGB, bytes)
        } else {
            let mut bytes = alloc::vec::Vec::with_capacity(picture.data.len() * 2);
            for s in &picture.data {
                bytes.extend_from_slice(&s.to_ne_bytes());
            }
            (PixelDescriptor::RGB16_SRGB, bytes)
        };
        let pixel_buffer =
            PixelBuffer::from_vec(bytes, width, height, descriptor).map_err(|e| {
                let boxed: alloc::boxed::Box<dyn core::error::Error + Send + Sync> =
                    alloc::boxed::Box::new(e);
                CodecError::from_parts(
                    Some("zenjpegai"),
                    ErrorCategory::Internal(zencodec::InternalKind::Bug),
                    boxed,
                )
                .start_at()
            })?;

        Ok(DecodeOutput::new(pixel_buffer, info))
    }
}

fn headers_to_info(headers: &Headers) -> ImageInfo {
    let hdr = &headers.picture;
    let mut info = ImageInfo::new(hdr.width, hdr.height, jpegai_format());
    info.source_color.bit_depth = Some(hdr.bit_depth);
    if let Some(cicp) = headers.rendering.cicp {
        info.source_color.cicp = Some(Cicp::new(
            cicp.colour_primaries,
            cicp.transfer_characteristics,
            0,
            cicp.full_range,
        ));
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_soc_pih() {
        assert!(detect_jpegai(&[0xFF, 0x80, 0xFF, 0x82, 0, 0]));
        assert!(!detect_jpegai(&[0xFF, 0x80, 0xFF, 0x83]));
        assert!(!detect_jpegai(&[0xFF, 0x80]));
        assert!(!detect_jpegai(&[]));
        assert!(!detect_jpegai(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn format_definition_round_trips_through_custom() {
        let fmt = jpegai_format();
        assert_eq!(fmt, ImageFormat::Custom(&JPEGAI_FORMAT_DEFINITION));
        assert_eq!(JPEGAI_FORMAT_DEFINITION.name, "jpegai");
        assert!((JPEGAI_FORMAT_DEFINITION.detect)(&[
            0xFF, 0x80, 0xFF, 0x82, 0, 0
        ]));
    }

    #[test]
    fn error_categories_are_stable() {
        assert_eq!(
            Error::UnexpectedEof.category(),
            ErrorCategory::Image(ImageError::UnexpectedEof)
        );
        assert_eq!(
            Error::InvalidData("x").category(),
            ErrorCategory::Image(ImageError::Malformed)
        );
        assert_eq!(
            Error::NonConforming("x").category(),
            ErrorCategory::Image(ImageError::Malformed)
        );
        assert_eq!(
            Error::Unsupported("x").category(),
            ErrorCategory::Image(ImageError::Unsupported(UnsupportedImageKind::Feature))
        );
        assert_eq!(
            Error::Cancelled(enough::StopReason::Cancelled).category(),
            ErrorCategory::Stopped(enough::StopReason::Cancelled)
        );
        assert_eq!(Error::UnexpectedEof.codec_name(), Some("zenjpegai"));
    }

    #[test]
    fn limit_kind_matches_reason_text() {
        // Exact strings from `src/decoder/limits.rs`, so this test breaks (loudly) if that
        // wording ever changes without updating `limit_kind_from_reason`.
        assert_eq!(
            limit_kind_from_reason("picture wider than max_width"),
            LimitKind::Width
        );
        assert_eq!(
            limit_kind_from_reason("picture taller than max_height"),
            LimitKind::Height
        );
        assert_eq!(
            limit_kind_from_reason("picture has more than max_pixels samples"),
            LimitKind::Pixels
        );
        assert_eq!(
            limit_kind_from_reason("estimated decode memory exceeds max_memory_bytes"),
            LimitKind::Memory
        );
        assert_eq!(
            limit_kind_from_reason("codestream larger than max_input_bytes"),
            LimitKind::InputSize
        );
        assert_eq!(limit_kind_from_reason("something new"), LimitKind::Pixels);
    }

    /// A default-constructed config can probe (header-only) but not decode — the module docs'
    /// first mismatch. Exercised without any reference data: any well-formed-looking PIH bytes
    /// would do, but real header bytes from `decoder::limits` tests keep this honest.
    #[test]
    fn default_config_probes_but_cannot_decode() {
        let bytes: alloc::vec::Vec<u8> = "011103401f00338000008a32c5001800"
            .as_bytes()
            .chunks(2)
            .map(|p| u8::from_str_radix(core::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        let mut w = crate::container::CodestreamWriter::new();
        w.substream(crate::container::Marker::Pih, &bytes).unwrap();
        let stream = w.finish();

        let config = JpegAiDecoderConfig::default();
        let job = config.job();
        let info = job
            .probe(&stream)
            .expect("header-only probe needs no model");
        assert_eq!((info.width, info.height), (560, 888));

        let job = JpegAiDecoderConfig::default().job();
        let err = job
            .decoder(alloc::borrow::Cow::Borrowed(&stream), &[])
            .and_then(|d| d.decode())
            .unwrap_err();
        assert_eq!(
            err.category(),
            ErrorCategory::Request(RequestError::Invalid(InvalidKind::Parameters))
        );
    }
}
