//! JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) learned image codec in safe Rust.
//!
//! A port of the JPEG AI reference software (the "VM"). See `PORTING.md` for the
//! module-by-module map to the upstream Python/C++ sources and the parity test that
//! gates each one.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

whereat::define_at_crate_info!();

pub mod bitio;
pub mod container;
mod error;
pub mod header;
pub mod mans;
pub mod weights;

pub use error::Error;
