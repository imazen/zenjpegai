//! eICCI post-filter. **Not ported yet** (`ref/src/codec/coding_tools/filters/eICCI/icci_filter.py`).

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};
use crate::header::IcciHeader;

pub(super) fn apply(
    _ctx: &FilterContext<'_>,
    _h: &IcciHeader,
    _state: FilterState,
) -> Result<FilterState> {
    Err(Error::Unsupported("eICCI post-filter"))
}
