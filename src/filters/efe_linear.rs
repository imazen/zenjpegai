//! EFE linear post-filter. **Not ported yet** (`ref/src/codec/coding_tools/filters/EFElinear/EFElinear.py`).

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};
use crate::header::EfeLinearHeader;

pub(super) fn apply(
    _ctx: &FilterContext<'_>,
    _h: &EfeLinearHeader,
    _state: FilterState,
) -> Result<FilterState> {
    Err(Error::Unsupported("EFE linear post-filter"))
}
