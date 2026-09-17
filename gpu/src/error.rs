//! Error type of the GPU backend.

use core::fmt;

/// What can go wrong on the GPU path. Callers that want a strict or GPU-less fallback should
/// treat every variant except [`GpuError::Codec`] as "use the CPU decoder instead".
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// No usable adapter (none present, only a software rasteriser, or the name filter matched
    /// nothing).
    NoAdapter(String),
    /// Device creation, submission or buffer mapping failed.
    Device(String),
    /// A feature map does not fit this device's buffer limits.
    TooLarge { needed: u64, limit: u64 },
    /// Internal shape mismatch (a bug, or a checkpoint with an unexpected layout).
    Shape(String),
    /// The stream is valid but this path does not handle it (the CPU decoder may).
    Unsupported(&'static str),
    /// Error from the CPU stages (parsing, entropy decoding, model loading).
    Codec(zenjpegai::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(s) => write!(f, "no GPU adapter: {s}"),
            Self::Device(s) => write!(f, "GPU device error: {s}"),
            Self::TooLarge { needed, limit } => write!(
                f,
                "a feature map needs {needed} bytes, the device binds at most {limit}"
            ),
            Self::Shape(s) => write!(f, "shape error: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported on the GPU path: {s}"),
            Self::Codec(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for GpuError {}

impl From<zenjpegai::Error> for GpuError {
    fn from(e: zenjpegai::Error) -> Self {
        Self::Codec(e)
    }
}

pub(crate) type Result<T> = core::result::Result<T, GpuError>;
