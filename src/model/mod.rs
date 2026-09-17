//! Trained model parameters and the networks that use them.

pub mod common;
pub mod hsd;

pub use common::CommonModel;

/// Betas of the four trained models, indexed by `model_id` (`cfg/pipeline.json`).
pub const MODEL_BETAS: [&str; 4] = ["0.002", "0.012", "0.075", "0.5"];
/// Checkpoint file-name prefix per component.
pub const COMPONENT_NAMES: [&str; 2] = ["Y", "UV"];

/// Locates the upstream checkpoints under a `models/` directory laid out like the reference
/// repository (`VM_common_int/`, `VM_bop/`, ...).
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

    /// Common (entropy-stage) modules of component `ccs` for model `model_id`.
    pub fn load_common(&self, model_id: usize, ccs: usize) -> crate::error::Result<CommonModel> {
        let beta = MODEL_BETAS
            .get(model_id)
            .ok_or(crate::Error::InvalidData("model_id out of range"))?;
        let file = self.read(&alloc::format!(
            "VM_common_int/{}_{beta}.pth",
            COMPONENT_NAMES[ccs]
        ))?;
        let ck = crate::weights::Checkpoint::parse(&file)?;
        CommonModel::load(&ck, crate::header::LATENT_CHANNELS[ccs])
    }
}
