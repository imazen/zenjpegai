//! WebGPU (wgpu) synthesis backend for the `zenjpegai` JPEG AI decoder.
//!
//! The entropy stage and the small latent-domain networks stay on the CPU (the former must be
//! bit-exact); the synthesis transforms, where nearly all decode time goes, run as WGSL compute
//! shaders. See `README.md` for the API a browser build should call.
#![forbid(unsafe_code)]

mod context;
mod error;
pub mod kernels;
pub mod layers;
pub mod plan;

pub use context::{ContextOptions, GpuContext};
pub use error::GpuError;
