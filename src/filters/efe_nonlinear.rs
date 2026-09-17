//! EFE non-linear post-filter. **Not ported yet** (`ref/src/codec/coding_tools/filters/EFEnonlinear/EFEnonlinear.py`).

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};
use crate::header::EfeNonlinearHeader;

pub(super) fn apply(
    _ctx: &FilterContext<'_>,
    _h: &EfeNonlinearHeader,
    _state: FilterState,
) -> Result<FilterState> {
    Err(Error::Unsupported("EFE non-linear post-filter"))
}
