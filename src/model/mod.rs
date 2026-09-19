//! Trained model parameters and the networks that use them.

pub mod analysis;
mod attention;
pub mod common;
pub mod hsd;
pub mod hyper_decoder;
pub mod hyper_encoder;
pub mod icci;
pub(crate) mod load;
pub mod mcm;
pub mod synthesis;

pub use common::CommonModel;

#[cfg(feature = "std")]
use crate::header::OperatingPoint;
#[cfg(feature = "std")]
use crate::nn::fast::Engine;

/// Betas of the four trained models, indexed by `model_id` (`cfg/pipeline.json`).
pub const MODEL_BETAS: [&str; 4] = ["0.002", "0.012", "0.075", "0.5"];
/// Checkpoint file-name prefix per component.
pub const COMPONENT_NAMES: [&str; 2] = ["Y", "UV"];

/// Where checkpoint files come from. Paths are relative to a `models/` directory laid out like
/// the reference repository (`VM_common_int/Y_0.012.pth`, `VM_bop/decoder_UV_0.5.pth`, ...).
pub trait ModelSource {
    fn read(&self, rel: &str) -> crate::error::Result<alloc::borrow::Cow<'_, [u8]>>;

    /// Told by the loaders which tensors of `rel` they looked up (see [`with_checkpoint`]).
    /// Sources ignore it, except the model packer's [`crate::weights::packed::Recorder`].
    fn accessed(&self, _rel: &str, _tensors: &[alloc::string::String]) {}
}

/// Read and parse the checkpoint at `rel` (a `.pth` or a packed `ZJM1` file), run `load` on it,
/// and report the tensors it looked up to the source. Every loader should go through this so
/// that `zenjpegai pack-models` sees what it needs.
pub fn with_checkpoint<T>(
    src: &dyn ModelSource,
    rel: &str,
    load: impl FnOnce(&crate::weights::Checkpoint<'_>) -> crate::error::Result<T>,
) -> crate::error::Result<T> {
    let file = src.read(rel)?;
    // Owned checkpoint bytes (a `.pth` read off disk is the biggest transient of a first
    // decode); a borrowed bundle's bytes are the caller's and aren't counted.
    let _file_charge = match &file {
        alloc::borrow::Cow::Owned(b) => crate::mem::Charge::of_vec(b),
        alloc::borrow::Cow::Borrowed(_) => crate::mem::Charge::EMPTY,
    };
    let ck = crate::weights::Checkpoint::parse(&file)?;
    let out = load(&ck)?;
    src.accessed(rel, &ck.touched_names());
    Ok(out)
}

/// Checkpoints held in memory (for targets without a file system, such as the browser).
#[derive(Clone, Debug, Default)]
pub struct ModelBundle {
    files: alloc::collections::BTreeMap<alloc::string::String, alloc::vec::Vec<u8>>,
}

impl ModelBundle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) the file at `rel`, e.g. `"VM_common_int/Y_0.012.pth"`.
    pub fn insert(&mut self, rel: impl Into<alloc::string::String>, bytes: alloc::vec::Vec<u8>) {
        self.files.insert(rel.into(), bytes);
    }

    pub fn contains(&self, rel: &str) -> bool {
        self.files.contains_key(rel)
    }
}

impl ModelSource for ModelBundle {
    fn read(&self, rel: &str) -> crate::error::Result<alloc::borrow::Cow<'_, [u8]>> {
        self.files
            .get(rel)
            .map(|b| alloc::borrow::Cow::Borrowed(b.as_slice()))
            .ok_or_else(|| crate::Error::Model(alloc::format!("{rel}: not in the model bundle")))
    }
}

fn beta(model_id: usize) -> crate::error::Result<&'static str> {
    MODEL_BETAS
        .get(model_id)
        .copied()
        .ok_or(crate::Error::InvalidData("model_id out of range"))
}

/// Relative path of the common (entropy-stage and latent) checkpoint of component `ccs`.
pub fn common_path(model_id: usize, ccs: usize) -> crate::error::Result<alloc::string::String> {
    Ok(alloc::format!(
        "VM_common_int/{}_{}.pth",
        COMPONENT_NAMES[ccs],
        beta(model_id)?
    ))
}

/// Relative path of the synthesis checkpoint of component `ccs` at operating point `op`.
pub fn synthesis_path(
    model_id: usize,
    ccs: usize,
    op: crate::header::OperatingPoint,
) -> crate::error::Result<alloc::string::String> {
    use crate::header::OperatingPoint::*;
    let dir = match op {
        Sop => "VM_sop",
        Bop => "VM_bop",
        Hop => "VM_hop",
    };
    Ok(alloc::format!(
        "{dir}/decoder_{}_{}.pth",
        COMPONENT_NAMES[ccs],
        beta(model_id)?
    ))
}

/// Relative path of the analysis (encoder-side) checkpoint of component `ccs`. `op` must be
/// BOP or HOP; simple-profile streams are encoded with the BOP analysis transform.
pub fn analysis_path(
    model_id: usize,
    ccs: usize,
    op: crate::header::OperatingPoint,
) -> crate::error::Result<alloc::string::String> {
    use crate::header::OperatingPoint::*;
    let dir = match op {
        Sop | Bop => "VM_bop",
        Hop => "VM_hop",
    };
    Ok(alloc::format!(
        "{dir}/encoder_{}_{}.pth",
        COMPONENT_NAMES[ccs],
        beta(model_id)?
    ))
}

/// Luma analysis transform of model `model_id` (`op`: BOP or HOP).
pub fn load_analysis_primary(
    src: &dyn ModelSource,
    model_id: usize,
    op: crate::header::OperatingPoint,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<analysis::AnalysisPrimary> {
    with_checkpoint(src, &analysis_path(model_id, 0, op)?, |ck| {
        analysis::AnalysisPrimary::load(ck, op, eng)
    })
}

/// Chroma analysis transform of model `model_id` (`op`: BOP or HOP).
pub fn load_analysis_secondary(
    src: &dyn ModelSource,
    model_id: usize,
    op: crate::header::OperatingPoint,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<analysis::AnalysisSecondary> {
    with_checkpoint(src, &analysis_path(model_id, 1, op)?, |ck| {
        analysis::AnalysisSecondary::load(ck, op, eng)
    })
}

/// Hyper-encoder of component `ccs` (stored in the common checkpoint).
pub fn load_hyper_encoder(
    src: &dyn ModelSource,
    model_id: usize,
    ccs: usize,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<hyper_encoder::HyperEncoder> {
    with_checkpoint(src, &common_path(model_id, ccs)?, |ck| {
        hyper_encoder::HyperEncoder::load(ck, crate::header::LATENT_CHANNELS[ccs], eng)
    })
}

/// Common modules of component `ccs` for model `model_id`, packed for `eng`.
pub fn load_common(
    src: &dyn ModelSource,
    model_id: usize,
    ccs: usize,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<CommonModel> {
    with_checkpoint(src, &common_path(model_id, ccs)?, |ck| {
        CommonModel::load(ck, crate::header::LATENT_CHANNELS[ccs], eng)
    })
}

/// Luma synthesis transform of model `model_id` at operating point `op`.
pub fn load_synthesis_primary(
    src: &dyn ModelSource,
    model_id: usize,
    op: crate::header::OperatingPoint,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<synthesis::SynthesisPrimary> {
    with_checkpoint(src, &synthesis_path(model_id, 0, op)?, |ck| {
        synthesis::SynthesisPrimary::load(ck, op, eng)
    })
}

/// Chroma synthesis transform of model `model_id` at operating point `op`.
pub fn load_synthesis_secondary(
    src: &dyn ModelSource,
    model_id: usize,
    op: crate::header::OperatingPoint,
    eng: &crate::nn::fast::Engine,
) -> crate::error::Result<synthesis::SynthesisSecondary> {
    with_checkpoint(src, &synthesis_path(model_id, 1, op)?, |ck| {
        synthesis::SynthesisSecondary::load(ck, op, eng)
    })
}

/// Checkpoints in a directory on disk.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct ModelDir {
    root: std::path::PathBuf,
}

#[cfg(feature = "std")]
impl ModelSource for ModelDir {
    fn read(&self, rel: &str) -> crate::error::Result<alloc::borrow::Cow<'_, [u8]>> {
        let path = self.root.join(rel);
        std::fs::read(&path)
            .map(alloc::borrow::Cow::Owned)
            .map_err(|e| crate::Error::Model(alloc::format!("{}: {e}", path.display())))
    }
}

#[cfg(feature = "std")]
impl ModelDir {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// See [`load_common`].
    pub fn load_common(
        &self,
        model_id: usize,
        ccs: usize,
        eng: &Engine,
    ) -> crate::error::Result<CommonModel> {
        load_common(self, model_id, ccs, eng)
    }

    /// See [`load_synthesis_primary`].
    pub fn load_synthesis_primary(
        &self,
        model_id: usize,
        op: OperatingPoint,
        eng: &Engine,
    ) -> crate::error::Result<synthesis::SynthesisPrimary> {
        load_synthesis_primary(self, model_id, op, eng)
    }

    /// See [`load_synthesis_secondary`].
    pub fn load_synthesis_secondary(
        &self,
        model_id: usize,
        op: OperatingPoint,
        eng: &Engine,
    ) -> crate::error::Result<synthesis::SynthesisSecondary> {
        load_synthesis_secondary(self, model_id, op, eng)
    }
}
