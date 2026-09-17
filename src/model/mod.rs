//! Trained model parameters and the networks that use them.

pub mod common;
pub mod hsd;
pub mod hyper_decoder;
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

/// Locates the upstream checkpoints under a `models/` directory laid out like the reference
/// repository (`VM_common_int/`, `VM_bop/`, ...). Models are packed for the engine's SIMD tier
/// at load time.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub struct ModelDir {
    root: std::path::PathBuf,
}

#[cfg(feature = "std")]
impl ModelDir {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn read(&self, rel: &str) -> crate::error::Result<alloc::vec::Vec<u8>> {
        let path = self.root.join(rel);
        std::fs::read(&path)
            .map_err(|e| crate::Error::Model(alloc::format!("{}: {e}", path.display())))
    }

    fn beta(model_id: usize) -> crate::error::Result<&'static str> {
        MODEL_BETAS
            .get(model_id)
            .copied()
            .ok_or(crate::Error::InvalidData("model_id out of range"))
    }

    /// Common (entropy-stage and latent) modules of component `ccs` for model `model_id`.
    pub fn load_common(
        &self,
        model_id: usize,
        ccs: usize,
        eng: &Engine,
    ) -> crate::error::Result<CommonModel> {
        let file = self.read(&alloc::format!(
            "VM_common_int/{}_{}.pth",
            COMPONENT_NAMES[ccs],
            Self::beta(model_id)?
        ))?;
        let ck = crate::weights::Checkpoint::parse(&file)?;
        CommonModel::load(&ck, crate::header::LATENT_CHANNELS[ccs], eng)
    }

    fn synthesis_file(
        &self,
        model_id: usize,
        ccs: usize,
        op: OperatingPoint,
    ) -> crate::error::Result<alloc::vec::Vec<u8>> {
        let dir = match op {
            OperatingPoint::Sop => "VM_sop",
            OperatingPoint::Bop => "VM_bop",
            OperatingPoint::Hop => "VM_hop",
        };
        self.read(&alloc::format!(
            "{dir}/decoder_{}_{}.pth",
            COMPONENT_NAMES[ccs],
            Self::beta(model_id)?
        ))
    }

    /// Luma synthesis transform of model `model_id` at operating point `op`.
    pub fn load_synthesis_primary(
        &self,
        model_id: usize,
        op: OperatingPoint,
        eng: &Engine,
    ) -> crate::error::Result<synthesis::SynthesisPrimary> {
        let file = self.synthesis_file(model_id, 0, op)?;
        synthesis::SynthesisPrimary::load(&crate::weights::Checkpoint::parse(&file)?, op, eng)
    }

    /// Chroma synthesis transform of model `model_id` at operating point `op`.
    pub fn load_synthesis_secondary(
        &self,
        model_id: usize,
        op: OperatingPoint,
        eng: &Engine,
    ) -> crate::error::Result<synthesis::SynthesisSecondary> {
        let file = self.synthesis_file(model_id, 1, op)?;
        synthesis::SynthesisSecondary::load(&crate::weights::Checkpoint::parse(&file)?, op, eng)
    }
}
