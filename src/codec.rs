//! `zencodec` trait implementations: [`DecoderConfig`] / [`DecodeJob`] / [`Decode`] over the
//! one-call [`crate::Decoder`] and [`EncoderConfig`] / [`EncodeJob`] /
//! [`zencodec::encode::Encoder`] over [`crate::Encoder`], plus [`ImageFormat`] registration
//! for this stream's magic bytes
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
//!
//! # The encode half
//!
//! [`JpegAiEncoderConfig`] carries the [`ModelSource`], the [`EncodeParams`] every job encodes
//! with, and [`EncodeLimits`] baked into the shared [`crate::Encoder`] — the encode-side
//! mirror of [`JpegAiDecoderConfig`], including the `Default` wired to [`NoModelSource`] (a
//! default config can estimate but cannot encode). [`EncoderConfig::with_generic_quality`]
//! maps `q` to a `q / 100` bits-per-pixel target — the units the reference CLI's
//! `--set_target_bpp` uses — and selects [`crate::Encoder::encode_to_bpp`] (the rate matcher;
//! `EncodeParams::model_id` / `beta_displacement_log` then become its outputs).
//!
//! Per-job [`EncodeJob::with_limits`] is enforced the same way as on the decode side: an
//! independent check layered in front of the shared encoder (which still applies its baked-in
//! limits too — the strictest wins), plus `max_output_bytes` applied to the coded stream.
//! Cancellation goes through the same [`enough`] token the decoder uses.
//!
//! Inputs are interleaved RGB, 8 or 16 bits, full range, tagged sRGB/BT.709 or untagged. The
//! `RDI` substream this encoder writes is the reference's default (no CICP), so a descriptor
//! claiming any other colour space — or narrow range — is refused with
//! [`UnsupportedOperation::PixelFormat`] rather than silently mislabelled; forwarding a
//! descriptor's CICP into `RDI` is a possible extension. RGBA8 reaches
//! [`encode_srgba8`](zencodec::encode::Encoder::encode_srgba8) only when its alpha is declared
//! padding (`make_opaque`); the format has no alpha channel. Metadata is dropped, not embedded
//! (the capabilities declare no ICC/EXIF/XMP carrier), and row-level, pull and animation encode
//! are rejected: the analysis transform needs the whole picture and the format has no
//! animation.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;

use whereat::{At, ErrorAtExt};
use zencodec::decode::{
    Decode, DecodeCapabilities, DecodeJob, DecodeOutput, DecodeRowSink, DecoderConfig, OutputInfo,
};
use zencodec::encode::{EncodeCapabilities, EncodeJob, EncodeOutput, EncoderConfig};
use zencodec::estimate::{ComputeEnvironment, ImageCharacteristics, ResourceEstimate};
use zencodec::{
    CategorizedError, CodecError, ErrorCategory, ImageError, ImageFormat, ImageFormatDefinition,
    ImageInfo, InvalidKind, LimitKind, Metadata, RequestError, ResourceError, ResourceLimits,
    Unsupported, UnsupportedImageKind, UnsupportedOperation,
};
use zenpixels::{Cicp, PixelBuffer, PixelDescriptor, PixelSlice};

use crate::decoder::Headers;
use crate::error::Error;
use crate::header::OperatingPoint;
use crate::model::ModelSource;
use crate::nn::fast::Engine;
use crate::{
    Decoder, EncodeLimits, EncodeParams, Encoder, Limits, RgbImage, estimate_encode_memory,
};

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
    } else if reason.contains("max_output_bytes") {
        LimitKind::OutputSize
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

// ===========================================================================
// EncoderConfig (Tier 1) — the encode half of the bridge
// ===========================================================================

/// Pixel formats [`JpegAiEncoder`] accepts natively: interleaved RGB at 8 or 16 bits per
/// sample, full range, sRGB-tagged or untagged (see the module docs for why other colour
/// tagging is refused).
static ENCODE_DESCRIPTORS: &[PixelDescriptor] =
    &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB16_SRGB];

static JPEGAI_ENCODE_CAPS: EncodeCapabilities = EncodeCapabilities::new()
    .with_stop(true)
    .with_lossy(true)
    .with_native_16bit(true)
    .with_enforces_max_pixels(true)
    .with_enforces_max_memory(true)
    .with_quality_range(0.0, 100.0)
    .with_threads_supported_range(1, 16);

/// Reusable JPEG AI encoder configuration: a shared, cached [`Encoder`] plus the
/// [`EncodeParams`] every job of it encodes with. `Clone` is cheap (`Arc`); every clone
/// shares the same model cache — the encode-side mirror of [`JpegAiDecoderConfig`].
#[derive(Clone)]
pub struct JpegAiEncoderConfig {
    encoder: Arc<Encoder>,
    params: EncodeParams,
    /// `Some` = rate-match to this bits-per-pixel target ([`Encoder::encode_to_bpp`]); set by
    /// [`EncoderConfig::with_generic_quality`] or [`Self::target_bpp`].
    target_bpp: Option<f64>,
    /// Baked into `encoder` (so the shared model cache can be kept); kept here too so jobs
    /// can report the config's own bounds next to a per-job override.
    limits: EncodeLimits,
}

impl JpegAiEncoderConfig {
    /// A config over `models`, using `engine` (see [`Engine::new`]), [`EncodeParams::default`]
    /// (the reference encoder's defaults: model 1, displacement 0, BOP, tools off) and
    /// [`EncodeLimits::default`]. Wrap a real model source; a picture cannot encode without
    /// one.
    pub fn new(models: Arc<dyn ModelSource + Send + Sync>, engine: Engine) -> Self {
        Self::with_limits(models, engine, EncodeLimits::default())
    }

    /// [`Self::new`] with explicit resource limits, baked into the shared [`Encoder`] (a
    /// per-job `with_limits` layers an additional check on top of these but does not change
    /// the shared encoder's own pool sizing — the module docs' third mismatch, applied to the
    /// encode side identically).
    pub fn with_limits(
        models: Arc<dyn ModelSource + Send + Sync>,
        engine: Engine,
        limits: EncodeLimits,
    ) -> Self {
        Self {
            encoder: Arc::new(Encoder::with_source(Box::new(models), engine).limits(limits)),
            params: EncodeParams::default(),
            target_bpp: None,
            limits,
        }
    }

    /// The [`EncodeParams`] every job of this config encodes with.
    #[must_use]
    pub fn params(mut self, params: EncodeParams) -> Self {
        self.params = params;
        self
    }

    /// Rate-match to `bpp` bits per pixel ([`Encoder::encode_to_bpp`]): the model and the
    /// quantiser displacement become the search's outputs instead of `params`' inputs. `None`
    /// — the default — is a fixed-model encode ([`Encoder::encode`]).
    #[must_use]
    pub fn target_bpp(mut self, bpp: Option<f64>) -> Self {
        self.target_bpp = bpp;
        self
    }
}

impl Default for JpegAiEncoderConfig {
    /// Wired to [`NoModelSource`]: can [`estimate_encode_resources`](EncoderConfig::estimate_encode_resources),
    /// cannot encode — see the module docs.
    fn default() -> Self {
        Self::new(Arc::new(NoModelSource), Engine::new())
    }
}

impl EncoderConfig for JpegAiEncoderConfig {
    type Error = At<CodecError>;
    type Job = JpegAiEncodeJob;

    fn format() -> ImageFormat {
        jpegai_format()
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static EncodeCapabilities {
        &JPEGAI_ENCODE_CAPS
    }

    /// JPEG AI's quality dial is bitrate: `q` maps to a `q / 100` bits-per-pixel target — the
    /// units the reference CLI's `--set_target_bpp` uses — and the encode then runs the rate
    /// matcher (an order of magnitude slower than a fixed-model encode; it re-codes ~13
    /// trials). `q` outside `0..=100` clamps; `q <= 0` is refused at encode time (a target
    /// rate must be positive).
    fn with_generic_quality(mut self, quality: f32) -> Self {
        self.target_bpp = Some(f64::from(quality.clamp(0.0, 100.0)) / 100.0);
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        self.target_bpp.map(|b| (b * 100.0) as f32)
    }

    /// Calibrated heap estimate via [`estimate_encode_memory`] (E8); wall time from
    /// `benchmarks/encode_end_to_end_2026-09-18.tsv`: ~1.5 Mpx/s single-threaded at SOP/BOP
    /// (SOP encodes through the BOP analysis transform — `Encoder::model_set`), ~0.29 Mpx/s
    /// at HOP, and a rate-matched encode re-codes roughly thirteen trials on top of its
    /// analyses.
    fn estimate_encode_resources(
        &self,
        image: &ImageCharacteristics,
        compute: &ComputeEnvironment,
    ) -> ResourceEstimate {
        use zencodec::estimate::ThreadingInformation;
        let op = self.params.op;
        let rate_matched = self.target_bpp.is_some();
        let est = estimate_encode_memory(
            u64::from(image.width()),
            u64::from(image.height()),
            op,
            rate_matched,
        );
        let mpix_per_s = match op {
            OperatingPoint::Sop | OperatingPoint::Bop => 1_500_000.0,
            OperatingPoint::Hop => 290_000.0,
        };
        let trials = if rate_matched { 13.0 } else { 1.0 };
        let time_ms = (image.pixels() as f64 / mpix_per_s * 1000.0 * trials) as u64;
        let threading = if cfg!(feature = "parallel") {
            ThreadingInformation::parallel_unknown_knee()
        } else {
            ThreadingInformation::SERIAL
        };
        ResourceEstimate::new(est.live_bytes, time_ms)
            .with_peak_max(est.live_bytes.saturating_add(est.pool_bytes))
            .with_threading(threading)
            .at_cores(compute.cores())
    }

    fn job(self) -> JpegAiEncodeJob {
        JpegAiEncodeJob {
            encoder: self.encoder,
            params: self.params,
            target_bpp: self.target_bpp,
            config_limits: self.limits,
            job_limits: None,
            stop: None,
        }
    }
}

// ===========================================================================
// EncodeJob (Tier 2)
// ===========================================================================

/// Per-operation JPEG AI encode job.
pub struct JpegAiEncodeJob {
    encoder: Arc<Encoder>,
    params: EncodeParams,
    target_bpp: Option<f64>,
    /// The config's own limits, baked into `encoder` (can't be changed per-job).
    config_limits: EncodeLimits,
    /// A per-job override from [`EncodeJob::with_limits`]: enforced as an independent check
    /// layered in front of `encoder`, with `max_output_bytes` applied to the coded stream
    /// (see the module docs).
    job_limits: Option<ResourceLimits>,
    stop: Option<zencodec::StopToken>,
}

fn encode_limits_from_resource_limits(l: &ResourceLimits) -> EncodeLimits {
    EncodeLimits {
        max_pixels: l.max_pixels,
        max_width: l.max_width,
        max_height: l.max_height,
        max_memory_bytes: l.max_memory_bytes,
    }
}

impl EncodeJob for JpegAiEncodeJob {
    type Error = At<CodecError>;
    type Enc = JpegAiEncoder;
    type AnimationFrameEnc = ();

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.job_limits = Some(limits);
        self
    }

    /// The format this port writes carries no ICC/EXIF/XMP (the capabilities declare none),
    /// so metadata is dropped, not embedded — the "silently skips the rest" half of the
    /// trait's contract.
    #[allow(deprecated)]
    fn with_metadata(self, _meta: Metadata) -> Self {
        self
    }

    fn encoder(self) -> core::result::Result<JpegAiEncoder, At<CodecError>> {
        // The per-job limits replace the config's for the layered check, as on the decode
        // side (`effective_limits`); the encoder's baked-in limits still apply inside the
        // encode, so the strictest of the two wins either way.
        let (limits, max_output_bytes) = match self.job_limits {
            Some(l) => (encode_limits_from_resource_limits(&l), l.max_output_bytes),
            None => (self.config_limits, None),
        };
        Ok(JpegAiEncoder {
            encoder: self.encoder,
            params: self.params,
            target_bpp: self.target_bpp,
            limits,
            max_output_bytes,
            stop: self.stop,
        })
    }

    fn animation_frame_encoder(self) -> core::result::Result<(), At<CodecError>> {
        Err(unsupported_operation(UnsupportedOperation::AnimationEncode))
    }
}

// ===========================================================================
// Encoder (Tier 3)
// ===========================================================================

/// Single-image JPEG AI encoder, bound to a shared [`Encoder`]'s model cache. Produces
/// exactly the bytes [`Encoder::encode`] (fixed model) or [`Encoder::encode_to_bpp`] (when a
/// rate target is set) produce on the same source — that byte-identity is what
/// `tests/codec_ref.rs` gates.
pub struct JpegAiEncoder {
    encoder: Arc<Encoder>,
    params: EncodeParams,
    target_bpp: Option<f64>,
    /// Whichever bounds this encode is judged against: the per-job override if
    /// [`EncodeJob::with_limits`] ran, else the config's own.
    limits: EncodeLimits,
    /// `max_output_bytes` from the job's [`ResourceLimits`], applied after coding.
    max_output_bytes: Option<u64>,
    stop: Option<zencodec::StopToken>,
}

impl JpegAiEncoder {
    fn encode_rgb(self, src: RgbImage) -> core::result::Result<EncodeOutput, At<CodecError>> {
        if src.width == 0 || src.height == 0 {
            return Err(<Error as Into<At<CodecError>>>::into(
                Error::InvalidArgument("empty pixel buffer"),
            ));
        }
        // The layered per-job limit check; the encoder's own (baked-in) limits are checked
        // again inside `encode_with` / `encode_to_bpp_with`, so the strictest wins.
        self.limits
            .check(
                src.width,
                src.height,
                self.params.op,
                self.target_bpp.is_some(),
            )
            .map_err(<Error as Into<At<CodecError>>>::into)?;
        let stop: &dyn enough::Stop = match &self.stop {
            Some(t) => t,
            None => &enough::Unstoppable,
        };
        let out = match self.target_bpp {
            Some(bpp) => {
                let (stream, matched) = self
                    .encoder
                    .encode_to_bpp_with(src, bpp, self.params, stop)
                    .map_err(into_codec_error)?;
                let mut out = EncodeOutput::new(stream, jpegai_format());
                // What the rate search decided, for callers that want it back.
                out.extensions_mut().insert(matched);
                out
            }
            None => EncodeOutput::new(
                self.encoder
                    .encode_with(src, self.params, stop)
                    .map_err(into_codec_error)?,
                jpegai_format(),
            ),
        };
        if self
            .max_output_bytes
            .is_some_and(|max| out.len() as u64 > max)
        {
            return Err(<Error as Into<At<CodecError>>>::into(Error::LimitExceeded(
                "encoded stream larger than max_output_bytes",
            )));
        }
        Ok(out)
    }
}

/// A [`PixelSlice`] to the [`RgbImage`] [`Encoder::encode`] takes. Only the physical layout
/// is converted (u8 samples widen to u16; u16 rows are native-endian); the colour contract is
/// in the module docs.
fn slice_to_rgb(pixels: &PixelSlice<'_>) -> core::result::Result<RgbImage, At<CodecError>> {
    use zenpixels::{ColorPrimaries, PixelFormat, SignalRange, TransferFunction};
    let d = pixels.descriptor();
    let depth = match d.pixel_format() {
        PixelFormat::Rgb8 => 8,
        PixelFormat::Rgb16 => 16,
        _ => return Err(unsupported_operation(UnsupportedOperation::PixelFormat)),
    };
    if d.signal_range != SignalRange::Full
        || d.primaries != ColorPrimaries::Bt709
        || !matches!(
            d.transfer,
            TransferFunction::Srgb | TransferFunction::Unknown
        )
    {
        return Err(unsupported_operation(UnsupportedOperation::PixelFormat));
    }
    let (w, h) = (pixels.width() as usize, pixels.rows() as usize);
    let mut data = alloc::vec::Vec::with_capacity(w * h * 3);
    if depth == 8 {
        for y in 0..pixels.rows() {
            data.extend(pixels.row(y).iter().map(|&v| u16::from(v)));
        }
    } else {
        for y in 0..pixels.rows() {
            data.extend(
                pixels
                    .row(y)
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_ne_bytes(*c)),
            );
        }
    }
    Ok(RgbImage {
        width: w,
        height: h,
        bit_depth: depth,
        data,
    })
}

impl zencodec::encode::Encoder for JpegAiEncoder {
    type Error = At<CodecError>;

    fn reject(op: UnsupportedOperation) -> At<CodecError> {
        unsupported_operation(op)
    }

    fn encode(self, pixels: PixelSlice<'_>) -> core::result::Result<EncodeOutput, At<CodecError>> {
        self.encode_rgb(slice_to_rgb(&pixels)?)
    }

    /// RGBA8 input is encodable only when its alpha is declared padding (`make_opaque`):
    /// JPEG AI has no alpha channel, so meaningful alpha is refused.
    fn encode_srgba8(
        self,
        data: &mut [u8],
        make_opaque: bool,
        width: u32,
        height: u32,
        stride_pixels: u32,
    ) -> core::result::Result<EncodeOutput, At<CodecError>> {
        if !make_opaque {
            return Err(unsupported_operation(UnsupportedOperation::PixelFormat));
        }
        let (w, h, stride) = (width as usize, height as usize, stride_pixels as usize);
        let needed = stride
            .checked_mul(h)
            .and_then(|n| n.checked_mul(4))
            .expect("encode_srgba8: stride_pixels * height overflows");
        assert!(
            data.len() >= needed,
            "encode_srgba8: data.len() < stride_pixels * height * 4"
        );
        let mut rgb = alloc::vec::Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for px in data[y * stride * 4..y * stride * 4 + w * 4]
                .as_chunks::<4>()
                .0
            {
                rgb.extend(px[..3].iter().map(|&v| u16::from(v)));
            }
        }
        self.encode_rgb(RgbImage {
            width: w,
            height: h,
            bit_depth: 8,
            data: rgb,
        })
    }
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
