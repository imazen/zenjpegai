//! LEF post-filter. **Not ported yet** (`ref/src/codec/coding_tools/filters/LEF/LEFfilter.py`).

use super::{FilterContext, FilterState};
use crate::error::{Error, Result};

pub(super) fn apply(
    _ctx: &FilterContext<'_>,
    _channel: u8,
    _state: FilterState,
) -> Result<FilterState> {
    Err(Error::Unsupported("LEF post-filter"))
}
