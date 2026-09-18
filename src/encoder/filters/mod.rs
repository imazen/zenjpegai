//! Encode side of the post-filters: the decisions `filters/*::compress` takes upstream and
//! the tool-header bits that carry them. The filters themselves are not applied here — the
//! reference runs them once on the reconstruction *after* `model.compress` returns
//! (`coding_engine.py::compress`), so outside the rate loop and outside anything that reaches
//! the codestream. The decoder-side filter bodies live in `crate::filters`.

pub mod efe_linear;
pub mod icci;
pub mod lef;
