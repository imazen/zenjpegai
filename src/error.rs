//! Error type.
//!
//! Internals use the plain [`Error`] (cheap in hot loops); the public API wraps it in
//! [`whereat::At`] so failures carry the call sites they passed through.

use core::fmt;

/// Everything that can go wrong while parsing, decoding or encoding a JPEG AI codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The input ended before a complete syntax element could be read.
    UnexpectedEof,
    /// The codestream violates the syntax. The string names the violated rule.
    InvalidData(&'static str),
    /// The codestream is valid but uses a feature this implementation does not support yet.
    Unsupported(&'static str),
    /// The stream does not conform to its declared profile or level.
    NonConforming(&'static str),
    /// A model checkpoint is missing, malformed, or has an unexpected tensor layout.
    Model(alloc::string::String),
    /// A caller-supplied argument is out of range.
    InvalidArgument(&'static str),
    /// A resource limit set by the caller would be exceeded.
    LimitExceeded(&'static str),
    /// A shared resource (`MemoryBudget`) cannot admit the job right now; retry may succeed.
    ResourceBusy(&'static str),
    /// The caller's [`enough::Stop`] token asked the operation to stop (cancelled or timed out).
    Cancelled(enough::StopReason),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnexpectedEof => f.write_str("unexpected end of data"),
            Error::InvalidData(s) => write!(f, "invalid codestream: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::NonConforming(s) => write!(f, "non-conforming codestream: {s}"),
            Error::Model(s) => write!(f, "model error: {s}"),
            Error::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Error::LimitExceeded(s) => write!(f, "limit exceeded: {s}"),
            Error::ResourceBusy(s) => write!(f, "resource busy: {s}"),
            Error::Cancelled(r) => write!(f, "stopped: {r}"),
        }
    }
}

impl core::error::Error for Error {}

impl From<enough::StopReason> for Error {
    fn from(r: enough::StopReason) -> Self {
        Error::Cancelled(r)
    }
}

/// Internal result alias (untraced).
pub(crate) type Result<T> = core::result::Result<T, Error>;
