//! JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) learned image codec in safe Rust.
//!
//! A port of the JPEG AI reference software (the "VM"). See `PORTING.md` for the
//! module-by-module map to the upstream Python/C++ sources and the parity test that
//! gates each one.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
// Without `unstable-internals`, `nn`/`model`/`tools`/`mans`/`bitio`/`container`/`tensor`/
// `weights`/`filters` are `pub(crate)`: unreachable from outside the crate, so `dead_code`
// correctly flags every item in them with no internal caller — mostly public-API-only
// convenience wrappers (e.g. `ModelDir::load_common`) and in-progress encoder scaffolding
// (`model::analysis`, `HyperEncoder`, `AnsEncoder`, ...) that exists for external/future callers,
// not for this crate's own use. That is real signal when the internals ARE public (hence no
// blanket `allow` inside the `unstable-internals` build) but noise otherwise.
#![cfg_attr(not(feature = "unstable-internals"), allow(dead_code, unused_imports))]
// Doc-completeness is enforced on the committed public API only (this crate without
// `unstable-internals`: `Decoder`, `header`, the error/limit types, and `codec` when
// `zencodec` is on). Internal modules made `pub` by `unstable-internals` are explicitly not a
// committed API (see the feature's doc comment in Cargo.toml) and are not held to this bar yet.
#![cfg_attr(not(feature = "unstable-internals"), warn(missing_docs))]

extern crate alloc;

whereat::define_at_crate_info!();

// Internal modules: the crate-external API is the one-call `Decoder` re-exported below (plus
// `header` for the types that API takes/returns, and `codec` for the zencodec bridge). Direct
// access to `nn`, `model`, `tools`, `mans`, `bitio`, `container`, `tensor`, `weights` and
// `filters` is unstable — needed by this crate's own CLI/tests/benches and by the `gpu`/`wasm`
// sibling crates, but not a committed public API yet (see PORTING.md "API hygiene"). Their
// paths are identical either way (`unstable-internals` only changes `pub` vs `pub(crate)`), so
// enabling the feature never breaks code that already compiles against it.
#[cfg(feature = "unstable-internals")]
pub mod bitio;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod bitio;
#[cfg(feature = "zencodec")]
pub mod codec;
#[cfg(feature = "unstable-internals")]
pub mod container;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod container;
#[cfg(feature = "unstable-internals")]
pub mod decoder;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod decoder;
// The encoder is unconditionally `pub` (unlike the decoder-side internals above, which are
// gated behind `unstable-internals`) but its API is still settling — PORTING.md lists what is
// not ported yet. `missing_docs` on the committed public API is not applied to it for that
// reason — revisit once its shape settles.
#[cfg(feature = "std")]
#[allow(missing_docs)]
pub mod encoder;
mod error;
#[cfg(feature = "unstable-internals")]
pub mod filters;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod filters;
pub mod header;
#[cfg(feature = "unstable-internals")]
pub mod mans;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod mans;
mod mem;
#[cfg(feature = "unstable-internals")]
pub mod model;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod model;
#[cfg(feature = "unstable-internals")]
pub mod nn;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod nn;
#[cfg(feature = "unstable-internals")]
pub mod tensor;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod tensor;
#[cfg(feature = "unstable-internals")]
pub mod tools;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod tools;
#[cfg(feature = "unstable-internals")]
pub mod weights;
#[cfg(not(feature = "unstable-internals"))]
pub(crate) mod weights;

#[cfg(feature = "std")]
pub use decoder::DecodeStats;
#[cfg(feature = "std")]
pub use decoder::Decoder;
pub use decoder::limits::{Limits, MemoryEstimate, estimate_memory};
pub use decoder::output::{Picture, RgbImage, YuvImage};
#[cfg(feature = "std")]
pub use encoder::{EncodeLimits, EncodeParams, Encoder, estimate_encode_memory};
pub use error::Error;
pub use mem::MemoryReport;
#[cfg(feature = "std")]
pub use mem::MemoryWatch;
// `Decoder::with_engine` / `Decoder::engine` (always public, `std`-gated only) take/return
// `nn::fast::Engine`, so it must stay reachable regardless of `unstable-internals` — otherwise
// those two methods would take a type callers could not name.
#[cfg(feature = "std")]
pub use nn::fast::Engine;
