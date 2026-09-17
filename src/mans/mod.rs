//! me-tANS: the multi-symbol, table-based ANS entropy coder of JPEG AI.
//!
//! Ports `ref/src/codec/entropy_coding/cpp_exts/mans/` (C++) and the table construction in
//! `ref/src/codec/entropy_coding/lib_wrappers/mans/`. Bit-exact with the reference in both
//! directions; `tests/mans_vectors.rs` checks that against streams produced by the reference's
//! own C++ extension.

// The coder loops index several parallel arrays with one counter, in the same interleaved order
// as the C++ they port; iterator chains would obscure that correspondence.
#![allow(clippy::needless_range_loop)]

mod decoder;
mod encoder;
mod pdf_tables;
mod tables;

pub use decoder::AnsDecoder;
pub use encoder::AnsEncoder;
pub use tables::MAX_Z;
#[allow(unused_imports)] // used by the z-substream decoder
pub(crate) use tables::normalize_z_cdf;

use alloc::boxed::Box;
use tables::ResidualTables;

/// The shared, immutable coder tables. Build once and reuse for every substream.
pub struct AnsTables {
    residual: Box<ResidualTables>,
}

impl AnsTables {
    pub fn new() -> Self {
        Self {
            residual: Box::new(ResidualTables::build()),
        }
    }

    /// Open a decoder over the thread byte ranges of one payload
    /// (see [`crate::container::split_threads`]).
    pub fn decoder(&self, threads: &[&[u8]]) -> Result<AnsDecoder<'_>, crate::Error> {
        AnsDecoder::new(&self.residual, threads)
    }

    /// Start an encoder with `num_threads` ANS threads (1, 2, 4, 8 or 16 in conforming streams).
    pub fn encoder(&self, num_threads: usize) -> Result<AnsEncoder<'_>, crate::Error> {
        AnsEncoder::new(&self.residual, num_threads)
    }

    /// Escape bound of distribution `sigma_idx`: symbols with `|v| >= bound` are escape-coded.
    pub fn bound(&self, sigma_idx: u8) -> u8 {
        self.residual.bounds[(sigma_idx & 31) as usize]
    }
}

impl Default for AnsTables {
    fn default() -> Self {
        Self::new()
    }
}
